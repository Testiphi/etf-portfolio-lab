"""路线 C：NiceGUI 实算应用。

与路线 A（静态站）的关系
------------------------
两条路线**共用同一套计算路径**（``reports.static_site.compute_preset`` 与 ``core/``）。
区别只在于交互：静态站是"算好给你看"，本应用是"改参数立刻重算"。
如果两条路线各写一套计算，数字迟早会不一致——那时教学工具本身就在误导人。

已落实的架构约束（来自调研结论）
--------------------------------
* **计算与界面分离**：页面只收集输入、展示输出，不做任何计算；
* **重计算走进程池**：``services.pool.run_heavy`` 优先用 ``run.cpu_bound``，
  失败回退主进程并明确告知，避免"一个访客的计算卡死所有访客"；
* **单进程部署**：NiceGUI 要求浏览器粘住最初服务它的进程，因此不使用多 worker；
  将来要扩容必须加 sticky session 与 Redis，这一点写在页面说明里；
* **不做大数据回传**：只推送"摘要 + 图表数据"，不整表塞进表格
  （NiceGUI 的 WebSocket 单条消息上限 1,000,000 字节）。

两个实测踩出来的坑（都在这里留档）
---------------------------------
1. **进程池作业必须只读打开数据库**。DuckDB 里只要有任一进程以读写打开就取独占锁，
   而数据库实例在进程内**常驻**——worker 算完任务、``con.close()`` 之后实例仍持锁，
   于是第二个 worker 直接失败（``IOException: 另一个程序正在使用此文件``），
   表现是"有的组合能打开、有的报 500"，且取决于访问顺序。
2. ``ui.page`` 的 **``response_timeout`` 默认只有 3 秒**，而一个组合要跑
   利率、蒙特卡洛与 Delta-Gamma 复制模拟，冷启动时必然超时
   （报 ``Response ... not ready after 3.0 seconds``，随后是
   ``Client has been deleted but is still being used``）。
   本项目把该页面的 ``response_timeout`` 放宽到 60 秒，并在启动时**后台预热**缓存，
   让第一个访客也不必等进程池冷启动。
"""

from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path
from typing import Any

from nicegui import app, ui

from etf_lab import __version__, universe
from etf_lab.content import teaching
from etf_lab.core import dca as dca_core
from etf_lab.core import returns as returns_core
from etf_lab.core import synthetic as synthetic_core
from etf_lab.data import repo
from etf_lab.presets import PRESETS, PRESETS_BY_KEY
from etf_lab.reports import figures, insights as insights_mod, static_site
from etf_lab.services import auth
from etf_lab.services.jobs import compute_custom_job, compute_preset_job
from etf_lab.services.pool import run_heavy

PRESET_PAGE_TIMEOUT = 60.0
"""组合页的构建时限（秒）。默认 3 秒对这里的计算量完全不现实。"""


def _users_path(db_path: str | Path | None = None) -> Path:
    """用户库路径（与行情库分开，见 data/users_schema.sql）。"""
    if db_path is None:
        return repo.DEFAULT_USERS_DB_PATH
    # 允许测试把用户库放在行情库旁边，便于隔离
    return Path(db_path).with_name("users.duckdb")


def _storage_secret() -> str:
    """NiceGUI 的会话签名密钥：优先取环境变量，否则在 data/ 下落一个随机值。

    必须**跨重启稳定**，否则每次重启都会让所有人的登录态失效
    （表现为"刚登录完一刷新又变回未登录"，很难查）。
    """
    from_env = os.environ.get("ETF_LAB_STORAGE_SECRET")
    if from_env:
        return from_env
    path = Path("data") / ".storage_secret"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    path.write_text(value, encoding="utf-8")
    return value


def _current_user() -> str | None:
    return app.storage.user.get("username")


def _collect_dca(controls: Mapping[str, Any], param_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """把界面控件收成定投设置。

    **只收当前模式真正用得到的参数**：否则存档里会混进一堆与所选模式无关的值，
    下次载入时看起来"参数变了"，其实是上一模式残留的。
    """
    mode = str(controls["mode"].value or "fixed")
    wanted = {spec[0] for spec in dca_core.PARAM_SPECS.get(mode, ())}
    params = {
        name: float(widget.value)
        for name, widget in param_inputs.items()
        if name in wanted and widget.value is not None
    }
    day = controls["day"].value
    return {
        "amount": float(controls["amount"].value or 2000.0),
        "freq": str(controls["freq"].value or "monthly"),
        "day": int(day) if day else None,
        "mode": mode,
        "params": params,
    }


def _collect_rebalance(controls: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": str(controls["policy"].value or "daily"),
        "threshold": float(controls["threshold"].value or 0.05),
        "cost_bps": float(controls["cost_bps"].value or 0.0),
    }


def _collect_cash(controls: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "usd_annual_rate": float(controls["usd_rate"].value or 0.0),
        "cash_tenor": "CN1Y",
    }


def _apply_definition(
    definition: Mapping[str, Any], controls: Mapping[str, Any], param_inputs: Mapping[str, Any]
) -> None:
    """把保存的组合设置填回控件（只预填，不自动计算）。"""
    dca_payload = dict(definition.get("dca") or {})
    if dca_payload.get("amount"):
        controls["amount"].value = float(dca_payload["amount"])
    if dca_payload.get("freq") in dca_core.FREQ_LABELS:
        controls["freq"].value = dca_payload["freq"]
    if dca_payload.get("day"):
        controls["day"].value = int(dca_payload["day"])
    if dca_payload.get("mode") in dca_core.MODES:
        # 赋值会触发参数显示逻辑（on_value_change），参数框随模式同步
        controls["mode"].value = dca_payload["mode"]
    for name, value in (dca_payload.get("params") or {}).items():
        if name in param_inputs and value is not None:
            param_inputs[name].value = float(value)

    rebalance_payload = dict(definition.get("rebalance") or {})
    if rebalance_payload.get("policy") in returns_core.REBALANCE_POLICIES:
        controls["policy"].value = rebalance_payload["policy"]
    for key in ("threshold", "cost_bps"):
        if rebalance_payload.get(key) is not None:
            controls[key].value = float(rebalance_payload[key])

    cash_payload = dict(definition.get("cash") or {})
    if cash_payload.get("usd_annual_rate") is not None:
        controls["usd_rate"].value = float(cash_payload["usd_annual_rate"])


def _user_header() -> None:
    """页头。

    **必须把"登录只用于保存"写在界面上**——否则访客会以为不登录就看不到东西，
    而本项目的定位恰恰是匿名可用全部功能。
    """
    with ui.row().classes("items-center gap-4"):
        ui.link("← 返回首页", "/")
        user = _current_user()
        if user:
            ui.label(f"已登录：{user}").classes("muted")
            ui.link("我的组合", "/portfolios")
            ui.link("退出", "/logout")
        else:
            ui.link("登录 / 注册", "/login")
    ui.label("所有分析功能都无需登录；登录只用于保存你自己配好的组合。").classes("muted")

CSS = """
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.metric-card { min-width: 150px; background:#151a22; border:1px solid #252c3a; }
.formula { font-family: ui-monospace, Consolas, monospace; background:#0e1116; border:1px solid #252c3a;
  color:#e8c07d; padding:5px 8px; border-radius:6px; font-size:12px; }
.warn { color:#ff7a59; }
.muted { color:#8b95a7; font-size:13px; }
/* 注入的表格块（与静态站共用同一套 HTML）需要这些基础样式，
   否则表格会挤在一起、数字对不齐，读起来比缺失更糟。 */
html table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 13px; }
html th, html td { border: 1px solid #252c3a; padding: 4px 8px; text-align: right; }
html th:first-child, html td:first-child { text-align: left; }
html th { background: #151a22; color: #8b95a7; font-weight: 600; }
html .note { color: #8b95a7; font-size: 12px; margin-top: 6px; line-height: 1.6; }
html .ok { color: #3ddc97; }
html .warn-text { color: #ff7a59; }
html .badge { border-radius: 4px; padding: 1px 6px; font-size: 11px; }
html .badge.pending { background: #3a2a1a; color: #e8c07d; }
"""

# 进程内缓存：key = f"{spec_key}|{data_version}"。小站点上这就够用；
# 多实例部署时必须换成 Redis（NiceGUI 的 app.storage 只是本地 JSON 文件）。
_RESULT_CACHE: dict[str, dict[str, Any]] = {}

# 在途任务表：同一档位可能被多个访客同时打开（启动预热也可能正在跑），
# 重复提交同一个重计算既浪费 CPU、也会让进程池排队。
_INFLIGHT: dict[str, asyncio.Task] = {}


def _db_path(db_path: str | Path | None) -> str | None:
    return str(db_path) if db_path is not None else None


def _data_version(db_path: str | Path | None) -> str:
    con = repo.connect(_db_path(db_path), read_only=True)
    try:
        return repo.latest_data_version(con)
    finally:
        con.close()


def _counts(db_path: str | Path | None) -> dict[str, int]:
    con = repo.connect(_db_path(db_path), read_only=True)
    try:
        return repo.table_counts(con)
    except Exception:  # noqa: BLE001 - 库还没建好时页面仍要能打开
        return {}
    finally:
        con.close()


async def _run_preset_job(key: str, db_path: str | Path | None, cache_key: str) -> tuple[dict[str, Any], str, str | None]:
    """真正执行一次重计算，并把结果写进进程内缓存。"""
    outcome = await run_heavy(compute_preset_job, key, _db_path(db_path))
    if isinstance(outcome.value, dict):
        _RESULT_CACHE[cache_key] = outcome.value
    return outcome.value, outcome.via, outcome.note


async def _compute_preset_cached(key: str, db_path: str | Path | None) -> tuple[dict[str, Any], str, str | None]:
    """带缓存地计算预设组合，返回 ``(结果, 执行途径, 提示)``。

    同一档位的并发请求会**合并到同一个在途任务**——否则三个访客同时打开同一档位，
    就会让进程池排三次完全相同的重计算。
    """
    version = _data_version(db_path)
    cache_key = f"{key}|{version}"
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key], "cache", None

    task = _INFLIGHT.get(cache_key)
    joined = task is not None
    if task is None:
        task = asyncio.ensure_future(_run_preset_job(key, db_path, cache_key))
        _INFLIGHT[cache_key] = task
    try:
        # shield：即使这一个访客断开，也不要取消已经在跑的重计算——
        # 否则另一个正在等待的访客会连带失败。
        result, via, note = await asyncio.shield(task)
    finally:
        if _INFLIGHT.get(cache_key) is task:
            _INFLIGHT.pop(cache_key, None)
    return result, ("合并到同一计算" if joined else via), note


def _metric_tiles(result: dict[str, Any]) -> None:
    m = result["metrics"]
    tiles = [
        ("年化收益", _pct(m.get("annualized_return"))),
        ("年化波动", _pct(m.get("annualized_volatility"))),
        ("夏普", _num(m.get("sharpe"))),
        ("索提诺", _num(m.get("sortino"))),
        ("卡玛", _num(m.get("calmar"))),
        ("最大回撤", _pct(m.get("max_drawdown"))),
        ("VaR 95%（历史）", _pct(m.get("var_95_historical"))),
        ("VaR 95%（参数）", _pct(m.get("var_95_parametric"))),
        ("CVaR 95%", _pct(m.get("cvar_95_historical"))),
        ("偏度", _num(m.get("skewness"), 3)),
        ("超额峰度", _num(m.get("excess_kurtosis"), 3)),
        ("样本交易日", str(m.get("n_obs"))),
    ]
    with ui.row().classes("gap-3 flex-wrap"):
        for label, value in tiles:
            with ui.card().classes("metric-card items-center"):
                ui.label(label).classes("muted")
                ui.label(value).classes("text-xl font-semibold")


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


ANALYSIS_TABS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (标签, 图表函数名, 表格块键)。图表与表格都来自 reports/，两条路线共用同一套表述。
    ("净值与回撤", ("fig_nav",), ("drawdown",)),
    ("收益与风险", ("fig_return_contribution", "fig_risk_vs_weight", "fig_rolling_sharpe"), ()),
    ("定投", ("fig_dca",), ("dca",)),
    ("各标的", ("fig_per_asset",), ("per_asset",)),
    ("因子敞口", ("fig_exposure_heatmap",), ("exposure",)),
    ("利率与久期", ("fig_yield_curve", "fig_yield_history", "fig_rate_scenarios"), ("rates", "duration")),
    ("波动率与期权", ("fig_vol_term_structure", "fig_protection_curve"), ("derivatives",)),
    ("蒙特卡洛", ("fig_mc_fan", "fig_mc_histogram", "fig_mc_convergence"), ("monte_carlo",)),
    ("Delta-Gamma 复制", ("fig_hedge_tradeoff",), ("hedge",)),
    ("历史情节重放", ("fig_episodes",), ()),
)
"""应用侧的模块清单。

**必须与静态站的仪表盘覆盖同一批分析。** 之前应用只渲染了净值/定投/各标的，
于是同一个组合在静态站上有利率、久期、蒙特卡洛、Delta-Gamma 复制，
在应用里却看不到——两条路线共用计算却不共用表述，是最容易产生"互相矛盾"的地方。
"""


def _render_analysis_tabs(result: dict[str, Any]) -> None:
    """按模块渲染标签页；图表来自 figures，表格来自 static_site.html_blocks。"""
    blocks = static_site.html_blocks(result)
    with ui.tabs().classes("w-full") as tabs:
        labels = [ui.tab(label) for label, _, _ in ANALYSIS_TABS]
    with ui.tab_panels(tabs, value=labels[0]).classes("w-full"):
        for (label, figure_names, block_keys), tab in zip(ANALYSIS_TABS, labels):
            with ui.tab_panel(tab):
                rendered = False
                for name in figure_names:
                    figure = getattr(figures, name)(result)
                    # 没有数据时图表是空的：画一堆空坐标轴比不画更糟
                    if getattr(figure, "data", None):
                        ui.plotly(figure).classes("w-full")
                        rendered = True
                for key in block_keys:
                    html = blocks.get(key) or ""
                    if html.strip():
                        ui.html(html)
                        rendered = True
                if not rendered:
                    ui.label("该模块需要更多数据或特定持仓结构，当前不可用。").classes("muted")


def _render_result(result: dict[str, Any]) -> None:
    ui.label(f"{result['name']}｜{result['start']} ~ {result['end']}（{result['n_obs']} 个交易日）").classes("muted")
    if result.get("question"):
        ui.label(result["question"]).classes("text-lg")
    if result.get("caveat"):
        with ui.card().classes("bg-amber-50"):
            ui.label(result["caveat"]).classes("text-sm")
    _metric_tiles(result)
    _render_analysis_tabs(result)

    ui.label("洞察（由你的数据触发）").classes("text-xl font-semibold mt-6")
    _insights_block(result)
    ui.label("可解锁模块").classes("text-xl font-semibold mt-6")
    _unlocks_block(result)


def _insights_block(result: dict[str, Any]) -> None:
    """洞察条：由数据触发，点开才展开知识与证据。"""
    found = insights_mod.evaluate(result)
    if not found:
        ui.label("当前数据没有触发任何提醒——没有哪一项越过阈值，这本身也是一个结论。").classes("muted")
        return
    for item in found:
        color = "border-l-4 border-orange-500" if item.level == "warn" else "border-l-4 border-blue-500"
        with ui.expansion(item.title).classes(f"w-full {color} bg-gray-800"):
            card = teaching.card(item.card)
            ui.label(card["title"]).classes("font-semibold")
            ui.html(f'<div class="formula">{card["formula"]}</div>')
            ui.label(f"说明什么：{card['means']}")
            ui.label(f"什么时候会骗人：{card['misleads']}").classes("warn")
            if item.evidence:
                ui.table(
                    columns=[
                        {"name": "k", "label": "依据", "field": "k", "align": "left"},
                        {"name": "v", "label": "数值", "field": "v"},
                    ],
                    rows=[{"k": k, "v": str(v)} for k, v in item.evidence.items()],
                ).classes("w-full")


def _unlocks_block(result: dict[str, Any]) -> None:
    """解锁清单：配出对应结构才会出现。"""
    items = insights_mod.evaluate_unlocks(result.get("composition"))
    triggered = sum(1 for i in items if i["triggered"])
    ui.label(f"已触发 {triggered}/{len(items)} 个模块——知识是被你的组合问出来的，不是翻页找到的。").classes("muted")
    with ui.row().classes("gap-3 flex-wrap"):
        for item in items:
            state = "已解锁" if item["triggered"] and item["implemented"] else ("已触发 · 待接入" if item["triggered"] else f"🔒 需{item['requirement_label']}")
            color = "border-green-500" if item["triggered"] and item["implemented"] else ("border-yellow-500" if item["triggered"] else "border-gray-600")
            with ui.card().classes(f"w-72 border {color}"):
                ui.label(f"{item['title']}｜{state}").classes("font-semibold")
                ui.label(f"触发条件：{item['trigger']}").classes("muted")
                if item["triggered"] and not item["implemented"]:
                    ui.label(f"待办：{item['pending']}").classes("warn")
                if item["triggered"]:
                    with ui.expansion("先了解这个概念").classes("w-full"):
                        card = teaching.card(item["card"])
                        ui.label(card["title"]).classes("font-semibold")
                        ui.html(f'<div class="formula">{card["formula"]}</div>')
                        ui.label(card["means"])
                        ui.label(card["misleads"]).classes("warn")


def _concept_cards(keys: list[str]) -> None:
    for key in keys:
        card = teaching.card(key)
        with ui.card().classes("w-full"):
            ui.label(card["title"]).classes("text-lg font-semibold")
            ui.html(f'<div class="formula">{card["formula"]}</div>')
            ui.label(f"说明什么：{card['means']}")
            ui.label(f"什么时候会骗人：{card['misleads']}").classes("warn")


def run(host: str = "127.0.0.1", port: int = 8080, db_path: str | Path | None = None) -> None:
    """启动应用。"""
    # shared=True 是必需的：用了 ui.page 之后，全局作用域注入的样式必须显式声明为共享，
    # 否则 NiceGUI 会在启动时直接抛 RuntimeError（本项目实测踩过）。
    ui.add_head_html(f"<style>{CSS}</style>", shared=True)

    @ui.page("/")
    def home() -> None:
        ui.dark_mode().enable()
        ui.label("ETF 组合数值实验室").classes("text-3xl font-bold")
        ui.label("不是告诉你买什么，而是让你看清那些数字怎么算、代表什么、什么时候会骗你。").classes("text-lg")
        _user_header()
        with ui.card().classes("bg-red-50 w-full"):
            ui.label(teaching.DISCLAIMER).classes("text-sm")
        ui.label("示例组合").classes("text-xl font-semibold mt-4")
        for spec in PRESETS:
            with ui.card().classes("w-full"):
                ui.link(spec.name, f"/preset/{spec.key}").classes("text-lg")
                ui.label(spec.question).classes("muted")
        ui.label("自定义实验室").classes("text-xl font-semibold mt-4")
        ui.link("打开实验室：拖动权重，看风险与收益怎么变", "/lab")
        counts = _counts(db_path)
        ui.label("数据底座").classes("text-xl font-semibold mt-4")
        ui.table(
            columns=[
                {"name": "table", "label": "表", "field": "table", "align": "left"},
                {"name": "rows", "label": "行数", "field": "rows"},
            ],
            rows=[{"table": k, "rows": f"{v:,}"} for k, v in counts.items() if v],
        ).classes("w-96")
        ui.label(
            f"数据版本 {_data_version(db_path)}｜本应用与静态报告站共用同一套计算代码｜版本 {__version__}"
        ).classes("muted")
        ui.label(
            "部署说明：NiceGUI 要求浏览器粘住最初服务它的进程，因此只能单进程运行；"
            "将来扩容需要 sticky session 与 Redis，而不是简单加 worker。"
        ).classes("muted")

    @ui.page("/preset/{key}", response_timeout=PRESET_PAGE_TIMEOUT)
    async def preset_page(key: str) -> None:
        ui.dark_mode().enable()
        ui.link("← 返回首页", "/")
        if key not in PRESETS_BY_KEY:
            ui.label(f"未知组合：{key}").classes("warn")
            return
        spec = PRESETS_BY_KEY[key]
        ui.label(spec.name).classes("text-2xl font-bold")
        spinner = ui.spinner(size="lg")
        container = ui.column().classes("w-full")
        try:
            result, via, note = await _compute_preset_cached(key, db_path)
            spinner.delete()
            with container:
                _render_result(result)
                ui.label(f"本次结果来源：{via}").classes("muted")
                if note:
                    ui.label(note).classes("warn")
        except Exception as exc:  # noqa: BLE001 - 页面要给出可操作的原因
            spinner.delete()
            with container:
                ui.label(f"计算失败：{type(exc).__name__}: {exc}").classes("warn")
                ui.label("请先采集数据：python -m etf_lab.cli fetch --preset core").classes("muted")

    def _save_controls(sliders, controls, param_inputs) -> None:
        """保存组合——**唯一需要登录的功能**。

        没登录时这里只显示一句说明与链接，而不是把实验室锁起来。
        """
        user = _current_user()
        with ui.card().classes("w-full"):
            ui.label("保存这个组合").classes("text-lg")
            if not user:
                ui.label("登录后可以把当前权重存下来；不登录也能照常调参、计算与查看全部结果。").classes("muted")
                ui.link("去登录 / 注册", "/login")
                return
            name_input = ui.input("组合名称", value="我的组合").classes("w-64")

            def save() -> None:
                raw = {s: float(sliders[s].value or 0) for s in sliders}
                total = sum(raw.values())
                if total <= 0:
                    ui.notify("权重之和不能为 0", type="warning")
                    return
                weights = {s: v / total for s, v in raw.items() if v > 0}
                # 权重、定投、再平衡、现金假设**全部**存下来：
                # 只存一部分会让"打开后重算"得到与保存时不同的数字。
                definition = {
                    "weights": weights,
                    "dca": _collect_dca(controls, param_inputs),
                    "rebalance": _collect_rebalance(controls),
                    "cash": _collect_cash(controls),
                }
                con = repo.connect_users(_users_path(db_path))
                try:
                    auth.save_portfolio(con, user, str(name_input.value or "我的组合"), definition)
                except auth.AuthError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                finally:
                    con.close()
                ui.notify("已保存，可在「我的组合」里打开", type="positive")

            ui.button("保存", on_click=save)
            ui.label(
                "保存权重、定投设置、再平衡规则与现金假设，**但不保存计算结果**——"
                "结果随数据版本变化，存下来只会变成过期数字。"
            ).classes("muted")

    @ui.page("/lab")
    async def lab() -> None:
        ui.dark_mode().enable()
        _user_header()
        ui.label("自定义组合实验室").classes("text-2xl font-bold")
        ui.label("拖动权重，然后点计算——所有数字与示例页面走完全相同的一套计算代码。").classes("muted")

        con = repo.connect(_db_path(db_path), read_only=True)
        try:
            # 用 etf_meta 而不是 DISTINCT symbol：新采集的标的会自动出现在这里，
            # 而且能显示名称与板块——**只给代码没人看得懂哪个是哪只**。
            meta = repo.read_etf_meta(con)
        finally:
            con.close()
        rows = [
            {
                "symbol": str(record.symbol),
                "name": str(getattr(record, "name", None) or record.symbol),
                "asset_class": universe.asset_class_label(getattr(record, "asset_class", None)),
                "index": str(getattr(record, "underlying_index", None) or ""),
                "t_plus": getattr(record, "t_plus", None),
                "cross": bool(getattr(record, "is_cross_border", False)),
            }
            for record in meta.itertuples(index=False)
        ]
        symbols = [row["symbol"] for row in rows]
        if not symbols:
            ui.label("本地数据仓还没有行情数据，请先运行：python -m etf_lab.cli fetch --preset core").classes("warn")
            return

        amounts: dict[str, Any] = {}
        sliders: dict[str, Any] = {}
        output = ui.column().classes("w-full")

        # 合成资产（现金/美元）也进可配列表：它们不是 ETF，但同样是**仓位**，
        # 而且是降低波动、承担汇率敞口的主要工具。
        synthetic_rows = [
            {
                "symbol": symbol,
                "name": info["name"],
                "asset_class": universe.asset_class_label(info["asset_class"]),
                "index": "",
                "t_plus": None,
                "cross": symbol == synthetic_core.USD_SYMBOL,
                "note": info["note"],
            }
            for symbol, info in synthetic_core.SYNTHETIC_ASSETS.items()
        ]
        all_rows = rows + synthetic_rows

        with ui.card().classes("w-full"):
            ui.label(
                f"标的权重：{len(all_rows)} 项（{len(rows)} 只已采集 ETF + {len(synthetic_rows)} 项现金类）"
            ).classes("muted")
            for row in all_rows:
                with ui.row().classes("items-center w-full"):
                    with ui.column().classes("w-64 gap-0"):
                        ui.label(f"{row['symbol']}　{row['name']}")
                        detail = row["asset_class"]
                        if row["index"]:
                            detail += f"｜跟踪 {row['index']}"
                        if row["cross"]:
                            detail += "｜跨境"
                        if row["t_plus"] is not None:
                            detail += f"｜T+{int(row['t_plus'])}"
                        if row.get("note"):
                            detail += f"｜{row['note']}"
                        ui.label(detail).classes("muted text-xs")
                    sliders[row["symbol"]] = ui.slider(min=0, max=100, step=5, value=0).classes("flex-grow")
                    amounts[row["symbol"]] = ui.label("0%").classes("w-16 text-right")
                    sliders[row["symbol"]].on_value_change(
                        lambda event, s=row["symbol"]: amounts[s].set_text(f"{int(event.value)}%")
                    )
            ui.label(
                "权重自动归一化到 100%；5% 只是滑动步长，**不要求整数份额、也不限制标的个数**。"
            ).classes("muted")

        controls: dict[str, Any] = {}
        with ui.card().classes("w-full"):
            ui.label("定投设置").classes("text-lg")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                controls["amount"] = ui.number("每期金额（元）", value=2000, min=100, step=100)
                controls["freq"] = ui.select(
                    dict(dca_core.FREQ_LABELS), value="monthly", label="频率"
                )
                controls["day"] = ui.number("周期内第几个交易日（留空＝首个）", value=None, min=1, max=23, step=1)
                controls["mode"] = ui.select(
                    {mode: dca_core.MODE_LABELS[mode] for mode in dca_core.MODES}, value="fixed", label="方式"
                )
            # 参数输入框按引擎的参数表生成（去重后每个参数只建一个控件），
            # 再按当前模式显示/隐藏——避免摆一堆与所选模式无关的输入框。
            unique_params: dict[str, tuple[float, float, float, float, str]] = {}
            for specs in dca_core.PARAM_SPECS.values():
                for name, default, low, high, step, label in specs:
                    unique_params.setdefault(name, (default, low, high, step, label))
            param_inputs: dict[str, Any] = {}
            with ui.row().classes("items-center gap-4 flex-wrap"):
                for name, (default, low, high, step, label) in unique_params.items():
                    param_inputs[name] = ui.number(label, value=default, min=low, max=high, step=step)

            def _sync_mode_params() -> None:
                wanted = {spec[0] for spec in dca_core.PARAM_SPECS.get(str(controls["mode"].value), ())}
                for name, widget in param_inputs.items():
                    widget.set_visibility(name in wanted)

            controls["mode"].on_value_change(lambda _: _sync_mode_params())
            _sync_mode_params()

        with ui.card().classes("w-full"):
            ui.label("再平衡设置").classes("text-lg")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                controls["policy"] = ui.select(
                    dict(returns_core.REBALANCE_POLICIES), value="daily", label="再平衡规则"
                )
                controls["threshold"] = ui.number("偏离阈值（仅阈值规则用）", value=0.05, min=0.01, max=1.0, step=0.01)
                controls["cost_bps"] = ui.number("单边交易成本（bp）", value=0.0, min=0.0, max=100.0, step=1.0)

            def _sync_policy() -> None:
                controls["threshold"].set_visibility(str(controls["policy"].value) == "threshold")

            controls["policy"].on_value_change(lambda _: _sync_policy())
            _sync_policy()
            ui.label(
                "规则影响的是**权重漂移**：每天归位＝恒定权重；从不归位＝让赢家跑。"
                "成本只在你再平衡时发生，所以「多久调一次」是一个真实的权衡。"
            ).classes("muted")

        with ui.card().classes("w-full"):
            ui.label("现金与美元").classes("text-lg")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                controls["usd_rate"] = ui.number(
                    "美元年化利率假设（0＝不生息）", value=0.0, min=0.0, max=0.10, step=0.005, format="%.3f"
                )
            ui.label(
                "人民币现金按国债曲线短端逐日计息（利率数据从 2015 年起，含现金会把样本推到那之后）；"
                "美元现金 = 汇率变动 + 上面的利率假设。"
                "**默认不生息**：美债利率历史只有近 4 年，套用当前利率会系统性高估 2012–2021 年，"
                "因此这里默认不叠加，意味着它**低估**了持有美元的实际收益。"
            ).classes("muted")

        ui.button("计算", on_click=lambda: _run_lab(sliders, controls, param_inputs, db_path, output))

        # 从「我的组合」带过来的定义：只用于预填，不自动计算
        # （自动算会让页面在打开瞬间就跑一次重计算，而用户可能只是想改一改）
        loaded = app.storage.user.pop("load_definition", None)
        if loaded:
            for symbol, weight in (loaded.get("weights") or {}).items():
                if symbol in sliders:
                    percent = int(round(float(weight) * 100))
                    sliders[symbol].value = percent
                    amounts[symbol].set_text(f"{percent}%")
            _apply_definition(loaded, controls, param_inputs)
            ui.label("已载入你保存的组合设置，点「计算」即可重算。").classes("muted")
        elif symbols:
            # 给个合理初值，让人一进来就能点"计算"看到东西
            sliders[symbols[0]].value = 100
            amounts[symbols[0]].set_text("100%")

        _save_controls(sliders, controls, param_inputs)

    async def _run_lab(sliders, controls, param_inputs, db_path, output) -> None:
        raw = {s: float(sliders[s].value or 0) for s in sliders}
        total = sum(raw.values())
        if total <= 0:
            ui.notify("权重之和不能为 0", type="warning")
            return
        weights = {s: v / total for s, v in raw.items() if v > 0}
        dca_payload = _collect_dca(controls, param_inputs)
        rebalance_payload = _collect_rebalance(controls)
        cash_payload = {"usd_annual_rate": float(controls["usd_rate"].value or 0.0), "cash_tenor": "CN1Y"}
        output.clear()
        with output:
            spinner = ui.spinner(size="lg")
            try:
                outcome = await run_heavy(
                    compute_custom_job,
                    weights,
                    dca_payload,
                    rebalance_payload,
                    cash_payload,
                    _db_path(db_path),
                )
                spinner.delete()
                _render_result(outcome.value)
                names = outcome.value.get("names") or {}
                rebalance_label = returns_core.REBALANCE_POLICIES.get(str(rebalance_payload["policy"]), "")
                ui.label(
                    f"本次结果来源：{outcome.via}｜定投：{dca_core.MODE_LABELS.get(dca_payload['mode'], '')}"
                    f"（{dca_core.FREQ_LABELS.get(dca_payload['freq'], '')}）"
                    f"｜再平衡：{rebalance_label}"
                    + "｜权重（已归一化）："
                    + "、".join(f"{names.get(s, s)} {w:.1%}" for s, w in weights.items())
                ).classes("muted")
                if outcome.note:
                    ui.label(outcome.note).classes("warn")
            except Exception as exc:  # noqa: BLE001
                spinner.delete()
                ui.label(f"计算失败：{type(exc).__name__}: {exc}").classes("warn")

    @ui.page("/login")
    def login_page() -> None:
        ui.dark_mode().enable()
        _user_header()
        ui.label("登录 / 注册").classes("text-2xl font-bold")
        ui.label(
            "账号只用来保存你自己的组合。不登录也能使用全部分析功能，"
            "匿名状态下不会往磁盘写任何一行。"
        ).classes("muted")

        username = ui.input("用户名").classes("w-64")
        password = ui.input("口令", password=True, password_toggle_button=True).classes("w-64")
        message = ui.label().classes("warn")

        def do_register() -> None:
            try:
                con = repo.connect_users(_users_path(db_path))
                try:
                    name = auth.register(con, str(username.value or ""), str(password.value or ""))
                finally:
                    con.close()
            except auth.AuthError as exc:
                message.set_text(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                message.set_text(f"注册失败：{type(exc).__name__}: {exc}")
                return
            app.storage.user["username"] = name
            ui.notify(f"已注册并登录：{name}", type="positive")
            ui.navigate.to("/portfolios")

        def do_login() -> None:
            try:
                con = repo.connect_users(_users_path(db_path))
                try:
                    ok = auth.authenticate(con, str(username.value or ""), str(password.value or ""))
                finally:
                    con.close()
            except Exception as exc:  # noqa: BLE001
                message.set_text(f"登录失败：{type(exc).__name__}: {exc}")
                return
            if not ok:
                # 不区分"用户不存在"与"口令错误"
                message.set_text("用户名或口令不对")
                return
            app.storage.user["username"] = auth.normalize_username(str(username.value))
            ui.notify("已登录", type="positive")
            ui.navigate.to("/portfolios")

        with ui.row():
            ui.button("登录", on_click=do_login)
            ui.button("注册新账号", on_click=do_register).props("outline")
        ui.label(f"口令至少 {auth.MIN_PASSWORD_LENGTH} 位；用 PBKDF2-HMAC-SHA256 加盐存储。").classes("muted")
        ui.label(
            "部署到公网请务必走 HTTPS——本项目没有做登录限流与口令找回，"
            "面向的是本地或自托管的单机使用。"
        ).classes("muted")

    @ui.page("/logout")
    def logout_page() -> None:
        ui.dark_mode().enable()
        app.storage.user.pop("username", None)
        app.storage.user.pop("load_definition", None)
        _user_header()
        ui.label("已退出登录。未登录状态下依然可以使用全部功能。").classes("muted")

    @ui.page("/portfolios")
    def portfolios_page() -> None:
        ui.dark_mode().enable()
        _user_header()
        user = _current_user()
        if not user:
            ui.label("这个页面需要登录——它是**唯一**需要登录的地方。").classes("text-xl")
            ui.link("去登录", "/login")
            return

        ui.label(f"{user} 保存的组合").classes("text-2xl font-bold")
        listing = ui.column().classes("w-full")

        def refresh() -> None:
            listing.clear()
            con = repo.connect_users(_users_path(db_path))
            try:
                rows = auth.list_portfolios(con, user)
            finally:
                con.close()
            with listing:
                if not rows:
                    ui.label("还没有保存过组合。去实验室调好权重后点保存。").classes("muted")
                    return
                for row in rows:
                    with ui.row().classes("items-center gap-2"):
                        ui.label(row["name"]).classes("w-48")
                        ui.label(row["updated_at"]).classes("muted")

                        def load(identifier: str = row["id"]) -> None:
                            con2 = repo.connect_users(_users_path(db_path))
                            try:
                                payload = auth.load_portfolio(con2, user, identifier)
                            finally:
                                con2.close()
                            if payload is None:
                                ui.notify("这个组合已经不在了", type="warning")
                                refresh()
                                return
                            app.storage.user["load_definition"] = payload["definition"]
                            ui.navigate.to("/lab")

                        def remove(identifier: str = row["id"], name: str = row["name"]) -> None:
                            con2 = repo.connect_users(_users_path(db_path))
                            try:
                                auth.delete_portfolio(con2, user, identifier)
                            finally:
                                con2.close()
                            ui.notify(f"已删除「{name}」", type="info")
                            refresh()

                        ui.button("在实验室打开", on_click=load).props("flat")
                        ui.button("删除", on_click=remove).props("flat color=negative")

        refresh()
        ui.link("去实验室调权重", "/lab")

    @ui.page("/concepts")
    def concepts() -> None:
        ui.dark_mode().enable()
        ui.link("← 返回首页", "/")
        ui.label("概念与陷阱").classes("text-2xl font-bold")
        _concept_cards(list(teaching.CARDS))

    async def _warm_cache() -> None:
        """启动时后台预热示例组合的缓存。

        冷启动的第一次计算要付进程池启动与子进程导入 numpy/scipy 的代价
        （实测让页面构建超过 3 秒，正好撞上默认的 response_timeout）。
        预热后第一个访客也能立刻看到结果。预热失败只是少了个优化，不该拦住启动。
        """
        for spec in PRESETS:
            try:
                await _compute_preset_cached(spec.key, db_path)
            except Exception:  # noqa: BLE001
                continue

    app.on_startup(_warm_cache)
    ui.run(
        host=host,
        port=port,
        title="ETF 组合数值实验室",
        reload=False,
        show=False,
        favicon="📊",
        storage_secret=_storage_secret(),
    )
