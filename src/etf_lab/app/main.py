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
from pathlib import Path
from typing import Any

from nicegui import app, ui

from etf_lab import __version__
from etf_lab.content import teaching
from etf_lab.data import repo
from etf_lab.presets import PRESETS, PRESETS_BY_KEY
from etf_lab.reports import figures, insights as insights_mod, static_site
from etf_lab.services.jobs import compute_custom_job, compute_preset_job
from etf_lab.services.pool import run_heavy

PRESET_PAGE_TIMEOUT = 60.0
"""组合页的构建时限（秒）。默认 3 秒对这里的计算量完全不现实。"""

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

    @ui.page("/lab")
    async def lab() -> None:
        ui.dark_mode().enable()
        ui.link("← 返回首页", "/")
        ui.label("自定义组合实验室").classes("text-2xl font-bold")
        ui.label("拖动权重，然后点计算——所有数字与示例页面走完全相同的一套计算代码。").classes("muted")

        con = repo.connect(_db_path(db_path), read_only=True)
        try:
            symbols = [row[0] for row in con.execute("SELECT DISTINCT symbol FROM etf_price ORDER BY symbol").fetchall()]
        finally:
            con.close()
        if not symbols:
            ui.label("本地数据仓还没有行情数据，请先运行：python -m etf_lab.cli fetch --preset core").classes("warn")
            return

        amounts: dict[str, Any] = {}
        sliders: dict[str, Any] = {}
        output = ui.column().classes("w-full")
        with ui.card().classes("w-full"):
            for symbol in symbols:
                with ui.row().classes("items-center w-full"):
                    ui.label(symbol).classes("w-24")
                    sliders[symbol] = ui.slider(min=0, max=100, step=5, value=0).classes("flex-grow")
                    amounts[symbol] = ui.label("0%").classes("w-16 text-right")
                    sliders[symbol].on_value_change(
                        lambda event, s=symbol: amounts[s].set_text(f"{int(event.value)}%")
                    )
            amount_input = ui.number("每期定投金额（元）", value=2000, min=100, step=100)
            mode_select = ui.select({"fixed": "固定金额", "value_avg": "价值平均"}, value="fixed", label="定投方式")
            ui.button("计算", on_click=lambda: _run_lab(symbols, sliders, amount_input, mode_select, db_path, output))

        # 给个合理初值，让人一进来就能点"计算"看到东西
        sliders[symbols[0]].value = 100
        amounts[symbols[0]].set_text("100%")

    async def _run_lab(symbols, sliders, amount_input, mode_select, db_path, output) -> None:
        raw = {s: float(sliders[s].value or 0) for s in symbols}
        total = sum(raw.values())
        if total <= 0:
            ui.notify("权重之和不能为 0", type="warning")
            return
        weights = {s: v / total for s, v in raw.items() if v > 0}
        output.clear()
        with output:
            spinner = ui.spinner(size="lg")
            try:
                outcome = await run_heavy(
                    compute_custom_job,
                    weights,
                    float(amount_input.value or 2000),
                    str(mode_select.value),
                    _db_path(db_path),
                )
                spinner.delete()
                _render_result(outcome.value)
                ui.label(f"本次结果来源：{outcome.via}｜权重（已归一化）："
                         + "、".join(f"{s} {w:.1%}" for s, w in weights.items())).classes("muted")
                if outcome.note:
                    ui.label(outcome.note).classes("warn")
            except Exception as exc:  # noqa: BLE001
                spinner.delete()
                ui.label(f"计算失败：{type(exc).__name__}: {exc}").classes("warn")

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
    ui.run(host=host, port=port, title="ETF 组合数值实验室", reload=False, show=False, favicon="📊")
