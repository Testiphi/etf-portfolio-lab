"""路线 A：把计算结果渲染成**纯静态、可离线打开**的深色密集仪表盘。

三条设计纪律（对应产品定位）
----------------------------
1. **计算与渲染分离**：:func:`compute_preset` 只返回可 JSON 序列化的字典
   （因此可缓存、可被路线 C 复用、也能直接喂给路线 B 的 Pyodide）；
2. **默认视图零解释文字**：页面上只有数字、表格、图。公式与「何时会骗人」放在
   ``<details>`` 里，点开才出现——由 :mod:`etf_lab.reports.theme` 统一提供；
3. **知识由数据触发**：:mod:`etf_lab.reports.insights` 的规则引擎读结果字典，
   数据满足条件时才浮现洞察条；组合结构满足时才解锁对应模块。

口径全部写进页面：复权方式、区间、无风险利率取值、数据版本。
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from etf_lab import __version__
from etf_lab.content import teaching
from etf_lab.content.episodes import EPISODES
from etf_lab.core import correlation, dca, derivatives as derivatives_mod, episodes as episodes_mod, exposure as exposure_mod, metrics, rates as rates_mod, returns, simulate as simulate_mod
from etf_lab.data import repo
from etf_lab.etl import fund_nav
from etf_lab.presets import PRESETS, PortfolioSpec
from etf_lab.reports import figures, insights as insights_mod, theme

RF_ANNUAL_DEFAULT = 0.02
"""教学场景下固定的无风险利率。真实使用时应从国债收益率曲线取，并在页面标注。"""
ROLLING_WINDOW = 252
"""滚动夏普窗口（约一年）。"""
ADJUSTMENT_STEP = 0.01
"""复权因子单日变化超过 1% 视为一次分红/份额折算事件。"""

# 敞口因子：刻意选**经济含义不同**的少数几个（大盘/中盘/成长/红利/债），
# 而不是把所有宽基指数都塞进去——高度同质的因子会让 beta 互相稀释，数字看着精确但没意义。
EXPOSURE_FACTORS: tuple[tuple[str, str], ...] = (
    ("000300", "沪深300"),
    ("000905", "中证500"),
    ("399006", "创业板指"),
    ("000922", "中证红利"),
    ("000012", "上证国债"),
)
BENCHMARK_INDEX = "000300"


# --------------------------------------------------------------------------- #
# 计算层：只产出可序列化的字典
# --------------------------------------------------------------------------- #
def _series_to_pairs(
    series: pd.Series,
    precision: int = 6,
    step: int = 1,
    keep_extremes: bool = False,
) -> list[list[Any]]:
    """序列 → ``[[日期, 值], ...]``。

    ``step`` 用于抽稀：日频曲线在约 1000 像素宽的图上根本分辨不出 3000 个点，
    而它却是页面体积的大头（三档合计 446 KB，占整页 85%）。

    ``keep_extremes=True`` 时额外保留首尾与极值点——单纯等距抽稀有可能
    恰好丢掉最深的那一天，那正是读者最需要看到的点。
    """
    if series.empty:
        return []
    if step > 1:
        picked = series.iloc[::step]
        if keep_extremes and len(series) > 2:
            extras = {series.index[0], series.index[-1], series.idxmin(), series.idxmax()}
            picked = series.loc[sorted(set(picked.index) | extras)]
    else:
        picked = series
    return [[str(idx.date()), round(float(val), precision)] for idx, val in picked.items()]


def _asset_summary(nav: pd.Series) -> dict[str, Any]:
    rets = nav.pct_change().dropna()
    info = metrics.max_drawdown(nav)
    return {
        "annualized_return": round(metrics.annualized_return(nav), 6),
        "annualized_volatility": round(metrics.annualized_volatility(rets), 6),
        "max_drawdown": round(info.depth, 6),
        "total_return": round(returns.total_return(nav), 6),
    }


def _exposure_block(
    con,
    aligned_prices: pd.DataFrame,
    weights: Mapping[str, float],
    start: Any,
    name_by_symbol: Mapping[str, str],
) -> dict[str, Any]:
    """因子敞口矩阵：对每只标的与整个组合各做一次 RBSA。"""
    try:
        codes = [code for code, _ in EXPOSURE_FACTORS]
        index_panel = repo.read_index_panel(con, codes, start=start)
        if index_panel.empty:
            return {}
        # 因子之间也要求同日有数据，保证各行的样本区间一致、beta 可比
        index_panel = index_panel.dropna(how="any")
        if len(index_panel) < exposure_mod.DEFAULT_MIN_OBS:
            return {}
        factor_returns = returns.to_returns(index_panel, method="simple")
        factor_returns = factor_returns.rename(columns=dict(EXPOSURE_FACTORS))
        asset_returns = returns.to_returns(aligned_prices, method="simple")
        matrix = exposure_mod.exposure_matrix(asset_returns, weights, factor_returns)
        for row in matrix["rows"]:
            row["name"] = "组合" if row["key"] == "__portfolio__" else name_by_symbol.get(row["key"], row["key"])
        matrix["factor_labels"] = {label: label for _, label in EXPOSURE_FACTORS}
        return matrix
    except Exception:  # noqa: BLE001 - 敞口算不出来不应影响其它面板
        return {}


def _episodes_block(result: Mapping[str, Any]) -> str:
    """历史情节重放：先把数字摆出来，再让"当时发生了什么"解释它。"""
    items = list(result.get("episodes") or [])
    if not items:
        return "<p class='note'>没有可重放的历史情节。</p>"

    blocks: list[str] = []
    for episode in items:
        portfolio = episode.get("portfolio") or {}
        market = episode.get("market") or {}
        chips: list[str] = []
        if portfolio:
            css = "warn" if float(portfolio["total_return"]) < 0 else "ok"
            chips.append(f'组合 <b class="{css}">{theme.pct(portfolio["total_return"])}</b>')
            chips.append(f'最深回撤 <b>{theme.pct(portfolio["max_drawdown"])}</b>')
            chips.append(f'最差单日 <b>{theme.pct(portfolio["worst_day"])}</b>')
            chips.append(f'<span>{portfolio["n_obs"]} 个交易日</span>')
        else:
            chips.append('<b class="warn">组合未覆盖</b>')
        if market:
            chips.append(f'沪深300 <b>{theme.pct(market["total_return"])}</b>')
        summary = theme.esc(episode["title"]) + "　" + "　".join(chips)

        coverage = episodes_mod.COVERAGE_LABELS.get(str(episode.get("coverage")), "")
        body = f'<p class="hook">{theme.esc(episode["hook"])}</p>'
        if coverage:
            body += f'<p class="note">{theme.esc(coverage)}</p>'
        body += f"<p>{theme.esc(episode['what_happened'])}</p>"
        cards = "".join(theme.card_html(t) for t in episode.get("terms", []) if t in teaching.CARDS)
        if cards:
            body += f'<div style="display:grid;gap:8px;margin-top:8px">{cards}</div>'
        blocks.append(f'<details class="episode"><summary>{summary}</summary><div class="body">{body}</div></details>')

    return '<div class="insights">' + "".join(blocks) + "</div>"


def _exposure_table(result: Mapping[str, Any]) -> str:
    """敞口明细表：把矩阵的数值也列出来（热力图看形状，表格看数字）。

    数值表与热力图都用全宽面板：早先挤在半宽面板里，
    行标签带 R² 后过长、列也放不下，才出现"渲染有误"的观感。
    """
    block = result.get("exposure") or {}
    rows = block.get("rows") or []
    factors = list(block.get("factors") or [])
    if not rows:
        return "<p class='note'>敞口矩阵需要指数数据，当前不可用。</p>"

    head = "".join(f"<th>{theme.esc(f)}</th>" for f in factors)
    body = ""
    for row in rows:
        if row.get("error"):
            body += (
                f'<tr><td>{theme.esc(row["name"])}</td>'
                f'<td colspan="{len(factors) + 4}" class="warn">{theme.esc(row["error"][:60])}</td></tr>'
            )
            continue
        cells = "".join(f'<td>{row["betas"].get(f, 0.0):.3f}</td>' for f in factors)
        body += (
            f'<tr><td>{theme.esc(row["name"])}</td>{cells}'
            f'<td>{theme.num(row["r_squared"], 3)}</td>'
            f'<td>{theme.pct(row["alpha_annual"])}</td>'
            f'<td>{theme.pct(row["tracking_error"])}</td>'
            f'<td>{row["n_obs"]}</td></tr>'
        )
    warnings = block.get("warnings") or []
    warn_html = "".join(f'<p class="note warn-text">⚠ {theme.esc(w)}</p>' for w in warnings)
    return (
        "<table><thead><tr><th>标的</th>"
        f"{head}<th>R²</th><th>Alpha<br>(年化)</th><th>跟踪误差</th><th>样本</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f"<p class='note'>{theme.esc(block.get('note', ''))}</p>"
        f"{warn_html}"
    )


def _monte_carlo_block(nav: pd.Series, rf_annual: float) -> dict[str, Any]:
    """蒙特卡洛：把点估计变成分布，并**并排展示多个模型**。

    为什么要跑四个模型：同一份历史数据，用不同假设模拟出来的达标概率可以差出几十个百分点。
    这个差异本身就是最重要的结论——它说明"未来收益分布"从来不是从数据里读出来的，
    而是你**假设**出来的。只给一个模型的结果，等于把假设藏起来。
    """
    try:
        returns = nav.pct_change().dropna()
        if len(returns) < 250:
            return {}
        results = simulate_mod.simulate_all_models(
            returns,
            # 4000 条路径：长期限下 1% 分位的标准误约为 8%（2000 条时会到 11%），
            # 路径数太少时那个数字本身在抖，不该拿来当结论
            n_paths=4000,
            horizon_days=1260,
            seed=20260101,
            goal_annual=0.08,
            deep_drawdown=-0.30,
        )
        if not results:
            return {}
        default = next((r for r in results if r.model == "bootstrap"), results[0])
        return {
            "params": {
                "n_paths": default.n_paths,
                "horizon_days": default.horizon_days,
                "horizon_years": round(default.horizon_days / 252, 2),
                "goal_annual": default.risk["goal_annual"],
                "deep_drawdown": default.risk["deep_drawdown"],
                "seed": default.seed,
            },
            "default_model": default.model,
            "models": {r.model: r.as_dict() for r in results},
            "comparison": simulate_mod.model_comparison(results),
            "warning": default.warning,
            "note": (
                "所有模型都只用**历史收益**拟合：它模拟不出历史里没出现过的极端事件。"
                "达标概率 = 终值 ≥ 期初 ×(1+目标年化)^年数 的路径占比；"
                "深度回撤概率 = 途中触及阈值（比只看终值更贴近实际体验）"
            ),
        }
    except Exception:  # noqa: BLE001 - 模拟失败不应影响其它面板
        return {}


def _monte_carlo_tables(result: Mapping[str, Any]) -> str:
    block = result.get("monte_carlo") or {}
    rows = block.get("comparison") or []
    if not rows:
        return "<p class='note'>蒙特卡洛需要足够长的历史样本，当前不可用。</p>"

    params = block.get("params") or {}
    body = ""
    goal_probs = [float(row["prob_goal"]) for row in rows]
    spread = (max(goal_probs) - min(goal_probs)) / max(min(goal_probs), 1e-9) if goal_probs else 0.0
    for row in rows:
        warn = " warn-text" if row.get("warning") else ""
        body += (
            f'<tr><td>{theme.esc(row["label"])}</td>'
            f'<td>{theme.pct(row["prob_goal"])}</td>'
            f'<td>{theme.pct(row["prob_loss"])}</td>'
            f'<td class="warn">{theme.pct(row["prob_deep_drawdown"])}</td>'
            f'<td>{theme.num(row["median_terminal"], 3)}</td>'
            f'<td>{theme.num(row["p5_terminal"], 3)}</td>'
            f'<td>{theme.pct(row["p95_max_drawdown"])}</td></tr>'
        )
    warning = block.get("warning")
    warn_html = f'<p class="note warn-text">⚠ {theme.esc(warning)}</p>' if warning else ""
    # 措辞要随事实变化：模型结论接近时说"不敏感"，差距大时说"敏感"。
    # 一句写死的"模型差异很大"在四个模型意见一致时就是误导。
    if spread < 0.05:
        spread_text = (
            f'四个模型的达标概率相对差距只有 <b>{spread:.0%}</b>——这次结论对模型选择<b>不敏感</b>，'
            "可以当作相对稳健的结果。"
        )
    elif spread < 0.20:
        spread_text = (
            f'四个模型的达标概率相对差距约 <b>{spread:.0%}</b>——属于中等敏感：'
            "结论方向一致，但具体数字不要当成精确值。"
        )
    else:
        spread_text = (
            f'四个模型的达标概率相对差距达 <b>{spread:.0%}</b>——结论对模型选择<b>高度敏感</b>，'
            "此时任何单一模型的数字都不能单独看。"
        )
    return (
        f'<p class="note">路径数 {params.get("n_paths")}，期限 {params.get("horizon_days")} 个交易日'
        f'（约 {params.get("horizon_years")} 年），目标年化 {theme.pct(params.get("goal_annual"))}，'
        f'深度回撤阈值 {theme.pct(params.get("deep_drawdown"))}，随机种子 {params.get("seed")}（可复现）。</p>'
        "<table><thead><tr><th>模型</th><th>达标概率</th><th>亏损概率</th>"
        "<th>深度回撤概率</th><th>中位终值</th><th>下5%终值</th><th>下5%回撤</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<p class="note">{spread_text}</p>'
        "<p class='note'>注意「下 5% 终值」在长期限下也与正态接近——日频厚尾在几百天求和后会被平均掉。"
        "厚尾真正影响的是**路径回撤**与短期限风险，这一点在 GARCH 与 Student-t 的深度回撤概率上看得出来。</p>"
        f"{warn_html}"
        f"<p class='note'>{theme.esc(block.get('note', ''))}</p>"
    )


def _rates_block(
    curve: pd.DataFrame | None,
    env: Any,
    aligned: pd.DataFrame,
    weights: Mapping[str, float],
    name_by_symbol: Mapping[str, str],
) -> dict[str, Any]:
    """利率环境 + 各标的久期估计 + 利率冲击情景。

    **只有 R² 达标的标的才进入情景表**。实测沪深300ETF 的"久期"是 −9.57 年、
    黄金 ETF 是 +4.50 年，但两者的 R² 都只有 0.01~0.02——它们根本不是被利率驱动的。
    把这种数字放进情景表，就是把噪声当结论。
    """
    environment = env.as_dict()
    if curve is None or curve.empty:
        return {"available": False, "environment": environment, "estimates": [], "scenarios": []}

    history = {}
    for code in ("CN1Y", "CN10Y"):
        subset = curve[curve["code"] == code].sort_values("date")
        if subset.empty:
            continue
        series = subset.set_index("date")["yield"]
        history[code] = _series_to_pairs(series, step=21)

    estimates: list[dict[str, Any]] = []
    for symbol in aligned.columns:
        estimate = rates_mod.estimate_duration(symbol, aligned[symbol].pct_change().dropna(), curve)
        if estimate is None:
            continue
        row = estimate.as_dict()
        row["name"] = name_by_symbol.get(symbol, symbol)
        estimates.append(row)

    reliable = {row["symbol"]: float(row["duration"]) for row in estimates if row["reliable"]}
    scenarios = rates_mod.rate_scenarios(reliable, weights) if reliable else []
    translated = [
        {
            **{k: v for k, v in row.items() if k != "per_holding"},
            "per_holding": {name_by_symbol.get(k, k): v for k, v in row["per_holding"].items()},
        }
        for row in scenarios
    ]
    return {
        "available": True,
        "environment": environment,
        "history": history,
        "estimates": estimates,
        "scenarios": translated,
        "reliable_symbols": sorted(reliable),
        "bond_weight": round(float(sum(weights.get(s, 0.0) for s in reliable)), 6),
        "note": (
            "久期用「该标的的日收益对收益率变动的回归」反推（ΔP/P ≈ −D·Δy）；"
            "只有 R² ≥ 0.2 的标的进入情景表，其余列入估计表但标注为无参考价值。"
            "未建模的资产（股票、商品等）的利率敏感性不计入冲击。"
        ),
    }


def _rates_tables(result: Mapping[str, Any]) -> str:
    block = result.get("rates") or {}
    env = block.get("environment") or {}
    if not block.get("available"):
        return "<p class='note'>本地没有收益率曲线数据，无风险利率退回假设值。运行 <code>etf-lab fetch --preset macro</code> 可补齐。</p>"

    source_label = "国债收益率曲线" if env.get("source") == "curve" else "兜底假设值"
    curve_rows = ""
    for code, value in sorted((env.get("curve") or {}).items(), key=lambda kv: kv[0]):
        previous = (env.get("curve_last_year") or {}).get(code)
        # 收益率以小数存放，变动用基点展示更直观
        change = "—" if previous is None else f"{(value - previous) * 10000:+.0f} bp"
        curve_rows += f"<tr><td>{theme.esc(code)}</td><td>{theme.pct(value)}</td><td>{change}</td></tr>"
    environment_html = (
        f'<p class="note">数据截止 {theme.esc(env.get("as_of"))}，无风险利率取 '
        f'<b>{theme.esc(env.get("tenor_used"))} = {theme.pct(env.get("risk_free"))}</b>（来源：{source_label}）。'
        "它直接影响夏普、索提诺与卡玛——此前页面写死 2%，那是把假设当成了事实。</p>"
        "<table><thead><tr><th>期限</th><th>当前</th><th>较一年前</th></tr></thead>"
        f"<tbody>{curve_rows}</tbody></table>"
    )
    slope = env.get("slope_10y_1y")
    if slope is not None:
        environment_html += (
            f'<p class="note">10 年 − 1 年期限利差 {slope * 10000:+.0f} bp，'
            f'曲线{"正常向上倾斜" if slope > 0 else "倒挂（短端高于长端）"}。</p>'
        )
    return environment_html


def _duration_tables(result: Mapping[str, Any]) -> str:
    block = result.get("rates") or {}
    estimates = block.get("estimates") or []
    scenarios = block.get("scenarios") or []
    if not estimates:
        return "<p class='note'>没有可用于久期估计的标的。</p>"

    estimate_rows = ""
    for row in estimates:
        flag = "" if row.get("reliable") else " <span class='badge pending'>不可靠</span>"
        estimate_rows += (
            f'<tr><td>{theme.esc(row["name"])}</td><td>{row["duration"]:+.2f}</td>'
            f'<td>{theme.esc(row["code"])}</td><td>{theme.num(row["r_squared"], 3)}</td>'
            f'<td>{row["n_obs"]}</td><td class="note">{theme.esc(row.get("warning") or "")}{flag}</td></tr>'
        )
    estimate_html = (
        "<table><thead><tr><th>标的</th><th>估计久期（年）</th><th>对哪条曲线</th><th>R²</th><th>样本</th><th>说明</th></tr></thead>"
        f"<tbody>{estimate_rows}</tbody></table>"
    )

    if not scenarios:
        return estimate_html + "<p class='note'>没有 R² 达标的标的，因此不给利率冲击情景——把噪声当结论比不给数字更糟。</p>"

    scenario_rows = ""
    for row in scenarios:
        sleeve = "—" if row.get("bond_sleeve_impact") is None else theme.pct(row["bond_sleeve_impact"])
        scenario_rows += (
            f'<tr><td>{row["shock_bp"]:+.0f} bp</td>'
            f'<td class="{"warn" if row["portfolio_impact"] < 0 else "ok"}">{theme.pct(row["portfolio_impact"])}</td>'
            f'<td>{sleeve}</td></tr>'
        )
    scenario_html = (
        f'<p class="note">债券腿权重合计 {theme.pct(block.get("bond_weight"))}；'
        "组合影响 = Σ（权重 × 久期 × 冲击）。「债券腿自身」= 组合影响 ÷ 债券权重，"
        "即那条腿单独会跌多少。</p>"
        "<table><thead><tr><th>收益率平行移动</th><th>组合影响</th><th>债券腿自身</th></tr></thead>"
        f"<tbody>{scenario_rows}</tbody></table>"
    )
    return estimate_html + scenario_html + f'<p class="note">{theme.esc(block.get("note", ""))}</p>'


def _derivatives_block(nav: pd.Series, aligned: pd.DataFrame, rf_annual: float) -> dict[str, Any]:
    """波动率期限结构 + 保护成本表（买保险要花多少钱）。

    **必须说清的边界**：本项目没有期权行情，因此这里全部是**历史波动率**下的理论值，
    不是市场报价的隐含波动率。隐含波动率是市场对未来波动的定价，
    市场恐慌时它远高于历史波动率——这正是"最需要保险时保险最贵"的来源。
    把这个区别写清楚，比多给几个希腊字母重要。
    """
    try:
        port_returns = nav.pct_change().dropna()
        term = derivatives_mod.volatility_term_structure(port_returns)
        tenor_map = {1 / 12: 21, 3 / 12: 63, 6 / 12: 126, 1.0: 252}
        tenor_vol = {
            str(tenor): (None if not np.isfinite(term.get(window, float("nan"))) else float(term[window]))
            for tenor, window in tenor_map.items()
        }
        usable = {float(k): v for k, v in tenor_vol.items() if v}
        protection = derivatives_mod.protection_table(spot=1.0, vol_by_tenor=usable)
        per_asset = {
            str(symbol): {
                str(window): (None if not np.isfinite(value) else round(float(value), 6))
                for window, value in derivatives_mod.volatility_term_structure(aligned[symbol].pct_change().dropna()).items()
            }
            for symbol in aligned.columns
        }
        return {
            "term_structure": {str(k): (None if not np.isfinite(v) else round(float(v), 6)) for k, v in term.items()},
            "tenor_vol": {k: (None if v is None else round(v, 6)) for k, v in tenor_vol.items()},
            "protection": [
                {k: (None if isinstance(v, float) and not np.isfinite(v) else round(float(v), 8)) for k, v in row.items()}
                for row in protection
            ],
            "per_asset": per_asset,
            "implied_vol_available": False,
            "note": (
                "波动率取自历史波动率（不同回看窗口），Greeks 是在该假设下的理论值；"
                "隐含波动率需要期权行情，本项目尚未接入"
            ),
        }
    except Exception:  # noqa: BLE001 - 期权模块算不出来不应影响其它面板
        return {}


def _derivatives_table(result: Mapping[str, Any]) -> str:
    block = result.get("derivatives") or {}
    rows = block.get("protection") or []
    if not rows:
        return "<p class='note'>期权模块需要价格数据，当前不可用。</p>"
    body = ""
    for row in rows:
        body += (
            f'<tr><td>{row["tenor_months"]:.0f} 个月</td><td>{row["moneyness"]:.0%}</td>'
            f'<td>{theme.pct(row["sigma"])}</td>'
            f'<td class="warn">{theme.pct(row["cost_pct"])}</td>'
            f'<td>{theme.pct(row["annualized_cost_pct"])}</td>'
            f'<td>{theme.num(row["delta"], 3)}</td><td>{theme.num(row["gamma"], 4)}</td>'
            f'<td>{theme.num(row["vega"], 4)}</td><td>{theme.num(row["theta"], 5)}</td></tr>'
        )
    return (
        "<table><thead><tr><th>期限</th><th>行权价 / 现价</th><th>所用波动率</th>"
        "<th>成本</th><th>年化成本</th><th>Delta</th><th>Gamma</th><th>Vega</th><th>Theta</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        "<p class='note'>「成本」是买入认沽期权的权利金占标的价值的比例；"
        "「年化成本」= 成本 ÷ 期限，用来回答「给持仓买一年保险，每年要付出本金的百分之几」。</p>"
        f"<p class='note'>{theme.esc(block.get('note', ''))}</p>"
    )


def _premium_block(con, symbols: Sequence[str], weights: Mapping[str, float], start: Any) -> dict[str, Any]:
    """折溢价率：**未复权市场价**与基金单位净值之比 − 1。

    必须用未复权价：前复权价已被分红调整过，拿它算折溢价会把历史分红误算成折价。
    没有净值数据时返回空字典，页面显示"—"，不阻塞整页。
    """
    try:
        raw_panel = repo.read_price_panel(con, symbols, start=start, field="close")
        nav_panel = repo.read_nav_panel(con, symbols, start=start)
        if raw_panel.empty or nav_panel.empty:
            return {}
        per_symbol: dict[str, Any] = {}
        for symbol in symbols:
            if symbol not in raw_panel.columns or symbol not in nav_panel.columns:
                continue
            series = fund_nav.premium_discount(raw_panel[symbol], nav_panel[symbol])
            if series.empty:
                continue
            per_symbol[symbol] = {
                "latest": round(float(series.iloc[-1]), 6),
                "mean": round(float(series.mean()), 6),
                "max_abs": round(float(series.abs().max()), 6),
                "as_of": str(series.index[-1].date()),
                "n_obs": int(len(series)),
            }
        if not per_symbol:
            return {}
        total_weight = sum(float(weights[s]) for s in per_symbol) or 1.0
        weighted_latest = sum(float(weights[s]) * per_symbol[s]["latest"] for s in per_symbol) / total_weight
        worst = max(per_symbol, key=lambda s: abs(per_symbol[s]["latest"]))
        return {
            "per_symbol": per_symbol,
            "weighted_latest": round(weighted_latest, 6),
            "max_abs_symbol": worst,
            "as_of": max(per_symbol[s]["as_of"] for s in per_symbol),
            "note": "折溢价 = 未复权收盘价 / 单位净值 − 1；净值为基金公司披露口径",
        }
    except Exception:  # noqa: BLE001 - 缺净值数据不影响其它面板
        return {}


def _risk_contribution(rets: pd.DataFrame, weights: Mapping[str, float]) -> dict[str, Any]:
    """风险贡献（成分 VaR 占比）+ 收益归因（**对数贡献**）。

    为什么用对数贡献而不是算术贡献
    ------------------------------
    算术贡献 ``Σ_t wᵢ·rᵢₜ`` 会被**该标的自身的波动拖累**主导：实测一个 13 年的股债金组合里，
    沪深300ETF 的算术贡献约 +8000bp，而组合实际累计只有 +21%——两个数字摆在一起无法解读，
    也正是图表"拱出屏幕"的根源。

    改用对数贡献 ``Σ_t wᵢ·ln(1+rᵢₜ)``：它扣掉了每个标的自身的复利效应，量级与实际收益可比；
    各项之和与组合**实际对数收益**之间的差额，恰好就是**再平衡/分散化效应**
    （由 Jensen 不等式可知它恒为非负），可以作为一个独立的数字展示，
    而不是含糊地叫"误差"。

    成分 VaR 用欧拉分解，各标的占比之和为 1（这条是精确的）。
    """
    simple = returns.to_simple(rets, method="simple")
    log_rets = np.log1p(simple)
    contribution = {symbol: float((log_rets[symbol] * float(weights[symbol])).sum()) for symbol in weights}
    component = correlation.component_var(simple, weights, level=0.95)
    total = float(component.sum())
    share = {symbol: (float(component[symbol]) / total if total else None) for symbol in weights}
    return {
        "component_var": {symbol: round(float(component[symbol]), 8) for symbol in weights},
        "component_var_share": {symbol: (None if share[symbol] is None else round(share[symbol], 6)) for symbol in weights},
        "log_contribution": {symbol: round(value, 8) for symbol, value in contribution.items()},
        "log_contribution_sum": round(sum(contribution.values()), 8),
    }


def _diagnostics(
    rets: pd.DataFrame,
    weights: Mapping[str, float],
    contribution: Mapping[str, float],
    adjustment_frame: pd.DataFrame | None,
    n_obs: int,
) -> dict[str, Any]:
    """喂给洞察规则引擎的诊断量。"""
    corr = rets.corr(min_periods=60)
    max_pair: dict[str, Any] = {}
    best = -np.inf
    columns = list(corr.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and float(value) > best:
                best = float(value)
                max_pair = {"a": a, "b": b, "rho": round(best, 4)}

    positives = {s: float(c) for s, c in contribution.items() if float(c) > 0}
    total_positive = sum(positives.values())
    concentration: dict[str, Any] = {}
    if positives and total_positive > 0:
        symbol = max(positives, key=lambda k: positives[k])
        concentration = {
            "symbol": symbol,
            "return_share": round(positives[symbol] / total_positive, 6),
            "basis": "positive",
            "total_positive_contribution": round(total_positive, 6),
        }

    adjustment_days = 0
    adjustment_symbols: list[str] = []
    if adjustment_frame is not None and not adjustment_frame.empty:
        change = adjustment_frame.pct_change()
        flagged = (change.abs() > ADJUSTMENT_STEP)
        adjustment_days = int(flagged.any(axis=1).sum())
        adjustment_symbols = [str(c) for c in flagged.columns[flagged.any(axis=0)]]

    return {
        "max_corr_pair": max_pair,
        "concentration": concentration,
        "adjustment_events": adjustment_days,
        "adjustment_symbols": adjustment_symbols,
        "n_obs": int(n_obs),
    }


def _rolling_sharpe(nav: pd.Series, rf_annual: float, window: int = ROLLING_WINDOW) -> pd.Series:
    rets = nav.pct_change().dropna()
    if len(rets) < window + 20:
        return pd.Series(dtype=float)
    # 防御：1 + r < 0 时分数次幂会给出复数，随后被静默截断成实数。
    # 正常情况下不会遇到（无风险利率是小数），但真出现过一次，所以显式挡住。
    if 1.0 + rf_annual <= 0:
        return pd.Series(dtype=float)
    rf_period = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
    excess = rets - rf_period
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std * np.sqrt(252)).dropna()


def compute_preset(
    con,
    spec: PortfolioSpec,
    rf_annual: float | None = None,
    start: str | None = None,
) -> dict[str, Any]:
    """算一个组合的全部展示数据（纯数据，不含任何渲染）。

    ``rf_annual=None`` 表示**从收益率曲线自动取无风险利率**；取不到时才退回
    ``RF_ANNUAL_DEFAULT`` 假设值，并在结果里标明来源是曲线还是假设。
    """
    symbols = list(spec.weights)
    panel = repo.read_price_panel(con, symbols, start=start, field="close_adj")
    if panel.empty:
        raise RuntimeError(f"组合 {spec.key} 的标在库中没有数据，请先运行：etf-lab fetch --preset core")

    missing = [s for s in symbols if s not in panel.columns]
    if missing:
        raise RuntimeError(f"组合 {spec.key} 缺少标的 {missing} 的价格数据")

    # 只保留全部标的都有价格的日期：缺失值不做填充（否则会造出不存在的收益）
    aligned = panel.dropna(how="any")
    if len(aligned) < 60:
        raise RuntimeError(
            f"组合 {spec.key} 对齐后仅剩 {len(aligned)} 个交易日，样本不足；"
            "通常是因为某只 ETF 上市太晚，请改用指数补历史或缩短组合"
        )

    nav = returns.nav_from_prices(aligned, weights=spec.weights)
    drawdown = metrics.drawdown_series(nav)
    rets = returns.to_returns(aligned, method="simple")

    # 无风险利率优先取真实国债收益率曲线；没有曲线数据时才退回假设值
    try:
        curve = repo.read_bond_yield(con)
    except Exception:  # noqa: BLE001 - 缺曲线不应影响其它面板
        curve = None
    rate_env = rates_mod.rate_environment(curve, fallback=RF_ANNUAL_DEFAULT)
    rf_used = float(rf_annual) if rf_annual is not None else rate_env.risk_free
    # 组合层面的指标必须用**单列**的组合收益；rets 是多资产面板，只用于相关性与分解
    port_rets = nav.pct_change().dropna()
    port_rets.name = "portfolio"
    info = metrics.max_drawdown(nav)

    corr = correlation.correlation_matrix(rets, method="pearson", min_obs=60)
    order = correlation.cluster_order(corr)
    corr_ordered = corr.loc[order, order]

    per_asset = {symbol: _asset_summary(returns.nav_from_prices(aligned[symbol])) for symbol in aligned.columns}

    risk = _risk_contribution(rets, spec.weights)
    # 对数贡献之和与组合实际对数收益的差额 = 再平衡/分散化效应（Jensen 不等式保证非负）
    risk["log_total_return"] = round(float(np.log1p(metrics.total_return(nav))), 8)
    risk["rebalancing_effect"] = round(risk["log_total_return"] - risk["log_contribution_sum"], 8)

    adjustment_frame = None
    try:
        adjustment_frame = repo.read_price_panel(con, symbols, start=start, field="adj_factor")
    except Exception:  # noqa: BLE001 - 缺 adj_factor 时不阻塞整页
        adjustment_frame = None

    diagnostics = _diagnostics(rets, spec.weights, risk["log_contribution"], adjustment_frame, len(aligned))

    meta = repo.read_etf_meta(con, symbols)
    class_by_symbol = dict(zip(meta.get("symbol", []), meta.get("asset_class", []))) if not meta.empty else {}
    name_by_symbol = dict(zip(meta.get("symbol", []), meta.get("name", []))) if not meta.empty else {}
    by_class: dict[str, float] = {}
    for symbol, weight in spec.weights.items():
        asset_class = str(class_by_symbol.get(symbol, "unknown"))
        by_class[asset_class] = round(by_class.get(asset_class, 0.0) + float(weight), 6)
    composition = {
        "by_asset_class": by_class,
        **{f"has_{key}": key in by_class for key in ("broad", "bond", "gold", "cross_border", "industry")},
    }

    premium = _premium_block(con, symbols, spec.weights, start)
    exposure_block = _exposure_block(con, aligned, spec.weights, start, name_by_symbol)
    derivatives_block = _derivatives_block(nav, aligned, rf_used)
    monte_carlo_block = _monte_carlo_block(nav, rf_used)
    rates_block = _rates_block(curve, rate_env, aligned, spec.weights, name_by_symbol)

    # 历史情节重放：基准指数可回溯到 2005 年，组合净值则受成分标的上市时间限制
    try:
        benchmark_panel = repo.read_index_panel(con, [BENCHMARK_INDEX], start="2005-01-01")
        benchmark = benchmark_panel[BENCHMARK_INDEX] if not benchmark_panel.empty else None
    except Exception:  # noqa: BLE001
        benchmark = None
    episode_block = episodes_mod.replay(nav, benchmark, EPISODES)

    # 定投：按组合净值定投（隐含"每日再平衡"假设，页面上必须写明）
    dca_runs: dict[str, Any] = {}
    for mode in ("fixed", "value_avg"):
        plan = dca.DcaPlan(
            amount=float(spec.dca.get("amount", 2000.0)),
            freq=str(spec.dca.get("freq", "monthly")),
            day=spec.dca.get("day"),
            mode=mode,  # type: ignore[arg-type]
        )
        try:
            result = dca.simulate(plan, nav)
            invested_curve = result.invested_curve
            first_contrib = invested_curve[invested_curve > 0]
            years = (invested_curve.index[-1] - first_contrib.index[0]).days / 365.25 if len(first_contrib) else float("nan")
            simple_on_invested = float(result.metrics.get("simple_return_on_invested", float("nan")))
            # 把累计收益错误地"当年化"（等价于假设所有钱在期初一次性投入）
            naive = (
                (1.0 + simple_on_invested) ** (1.0 / years) - 1.0
                if years and years > 0 and simple_on_invested > -1
                else float("nan")
            )
            dca_runs[mode] = {
                "value": _series_to_pairs(result.value, step=5),
                "invested": _series_to_pairs(invested_curve, step=5),
                "xirr": None if not np.isfinite(result.xirr) else round(float(result.xirr), 6),
                "invested_total": round(result.invested_total, 2),
                "final_value": round(result.final_value, 2),
                "cumulative_return_on_invested": round(simple_on_invested, 6),
                "naive_annualized_return": None if not np.isfinite(naive) else round(float(naive), 6),
                "years": None if not np.isfinite(years) else round(float(years), 3),
                "n_contributions": int(result.metrics.get("n_contributions", 0)),
                "total_fees": round(float(result.metrics.get("total_fees", 0.0)), 2),
            }
        except Exception as exc:  # noqa: BLE001 - 单个模式失败不应让整页打不开
            dca_runs[mode] = {"error": f"{type(exc).__name__}: {exc}"}

    rolling = _rolling_sharpe(nav, rf_used)

    return {
        "key": spec.key,
        "name": spec.name,
        "question": spec.question,
        "caveat": spec.caveat,
        "weights": {k: float(v) for k, v in spec.weights.items()},
        "dca_plan": dict(spec.dca),
        "start": str(aligned.index[0].date()),
        "end": str(aligned.index[-1].date()),
        "n_obs": int(len(aligned)),
        "data_version": repo.latest_data_version(con),
        "rf_annual": rf_used,
        "rf_source": rate_env.source,
        "rf_tenor": rate_env.tenor_used,
        "nav": _series_to_pairs(nav, step=3, keep_extremes=True),
        "drawdown": _series_to_pairs(drawdown, step=3, keep_extremes=True),
        "rolling_sharpe": _series_to_pairs(rolling, step=5),
        "correlation": {
            "labels": [str(c) for c in corr_ordered.columns],
            "matrix": [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in corr_ordered.to_numpy()],
        },
        "metrics": {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in metrics.summary(nav, port_rets, rf_annual=rf_used).items()
        },
        "time_weighted_annualized": round(metrics.annualized_return(nav), 6),
        "top_drawdowns": [
            {
                "depth": round(d.depth, 6),
                "peak": str(d.peak_date.date()),
                "trough": str(d.trough_date.date()),
                "recovery": str(d.recovery_date.date()) if d.recovery_date is not None else None,
                "duration_days": d.duration_days,
                "recovery_days": d.recovery_days,
            }
            for d in metrics.top_drawdowns(nav, top_n=5)
        ],
        "per_asset": per_asset,
        "dca": dca_runs,
        "risk_contribution": risk,
        "composition": composition,
        "premium_discount": premium,
        "exposure": exposure_block,
        "derivatives": derivatives_block,
        "monte_carlo": monte_carlo_block,
        "rates": rates_block,
        "episodes": episode_block,
        "diagnostics": diagnostics,
        "max_drawdown_info": {
            "depth": round(info.depth, 6),
            "peak": str(info.peak_date.date()),
            "trough": str(info.trough_date.date()),
            "recovery": str(info.recovery_date.date()) if info.recovery_date is not None else None,
        },
    }


# --------------------------------------------------------------------------- #
# 渲染层
# --------------------------------------------------------------------------- #
PAGE = Template(
    """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<link rel="stylesheet" href="$rootassets/style.css">
<script src="$rootassets/plotly.min.js"></script>
</head>
<body>
<header class="site-header">
  <a class="brand" href="$rootindex">ETF 组合数值实验室</a>
  <nav>
    <a href="$rootindex">组合工作台</a>
    <a href="$rootconcepts">知识附录</a>
    <a href="$rootabout">口径</a>
  </nav>
</header>
<main>
$body
</main>
<footer>
  <p class="disclaimer">$disclaimer</p>
  <p class="meta">版本 $version · 数据版本 $data_version · 生成于 $generated</p>
</footer>
</body>
</html>
"""
)


def _page(title: str, body: str, *, root: str = "", data_version: str = "—") -> str:
    return PAGE.substitute(
        title=title,
        rootassets=f"{root}assets",
        rootindex=f"{root}index.html",
        rootconcepts=f"{root}concepts.html",
        rootabout=f"{root}about.html",
        body=body,
        disclaimer=theme.esc(teaching.DISCLAIMER),
        version=__version__,
        data_version=theme.esc(data_version),
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _metrics_keyboard(result: Mapping[str, Any]) -> str:
    """指标键盘：每格默认只有标签+数值，点开才出现公式与陷阱。"""
    m = result["metrics"]
    tiles: list[tuple[str, str, str | None, bool]] = [
        ("年化收益（复合）", theme.pct(m.get("annualized_return")), "annualized_return", False),
        ("算术平均年化", theme.pct(m.get("arithmetic_annualized_return")), "arithmetic_vs_geometric", False),
        ("波动拖累", theme.pct(m.get("volatility_drag")), "arithmetic_vs_geometric", True),
        ("年化波动", theme.pct(m.get("annualized_volatility")), "volatility", False),
        ("夏普", theme.num(m.get("sharpe")), "sharpe", False),
        ("索提诺", theme.num(m.get("sortino")), "sortino", False),
        ("卡玛", theme.num(m.get("calmar")), "calmar", False),
        ("最大回撤", theme.pct(m.get("max_drawdown")), "max_drawdown", True),
        ("VaR 95% 历史", theme.pct(m.get("var_95_historical")), "var", False),
        ("VaR 95% 参数", theme.pct(m.get("var_95_parametric")), "var", False),
        ("CVaR 95%", theme.pct(m.get("cvar_95_historical")), "cvar", False),
        ("偏度", theme.num(m.get("skewness"), 3), None, False),
        ("超额峰度", theme.num(m.get("excess_kurtosis"), 3), None, False),
        ("样本交易日", str(m.get("n_obs")), None, False),
        ("无风险利率", theme.pct(m.get("rf_annual_used")), "sharpe", False),
        ("组合折溢价", theme.pct((result.get("premium_discount") or {}).get("weighted_latest")), "premium_discount", False),
        ("时间加权年化", theme.pct(result.get("time_weighted_annualized")), "xirr", False),
    ]
    return '<div class="tiles">' + "".join(
        theme.metric_tile(label, value, card, warn=warn) for label, value, card, warn in tiles
    ) + "</div>"


def _drawdown_table(result: Mapping[str, Any]) -> str:
    rows = ""
    for item in result["top_drawdowns"]:
        recovery = item["recovery"] or "未修复"
        if item["recovery_days"] is not None:
            recovery = f"{recovery}（{item['recovery_days']}d）"
        rows += (
            f"<tr><td>{item['peak']} → {item['trough']}</td>"
            f"<td class='warn'>{theme.pct(item['depth'])}</td>"
            f"<td>{item['duration_days']}d</td><td>{recovery}</td></tr>"
        )
    return (
        "<table><thead><tr><th>区间</th><th>深度</th><th>下跌</th><th>修复</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p class='note'>只报一个最大回撤会掩盖路径差异：跌得快恢复快与阴跌两年是完全不同的体验。</p>"
    )


def _dca_table(result: Mapping[str, Any]) -> str:
    rows = ""
    for mode, payload in result["dca"].items():
        label = "固定金额" if mode == "fixed" else "价值平均"
        if "error" in payload:
            rows += f"<tr><td>{theme.esc(label)}</td><td colspan='6' class='warn'>{theme.esc(payload['error'])}</td></tr>"
            continue
        rows += (
            f"<tr><td>{label}</td><td>{payload['n_contributions']}</td>"
            f"<td>{payload['invested_total']:,.0f}</td><td>{payload['final_value']:,.0f}</td>"
            f"<td>{theme.pct(payload['cumulative_return_on_invested'])}</td>"
            f"<td class='warn'>{theme.pct(payload['naive_annualized_return'])}</td>"
            f"<td class='ok'><strong>{theme.pct(payload['xirr'])}</strong></td></tr>"
        )
    return (
        "<table><thead><tr><th>方式</th><th>期数</th><th>累计投入</th><th>期末市值</th>"
        "<th>累计收益</th><th>错误年化</th><th>XIRR</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p class='note'>「累计收益」是全过程总涨幅；「错误年化」= 把它按年数折年化"
        f"（隐含假设所有钱期初就投入，定投里不成立）；只有 XIRR 按每笔钱的实际在场时间折算。"
        f"组合本身的时间加权年化为 {theme.pct(result.get('time_weighted_annualized'))}。</p>"
    )


def _per_asset_table(result: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{theme.esc(s)}</td><td>{theme.pct(v['annualized_return'])}</td>"
        f"<td>{theme.pct(v['annualized_volatility'])}</td><td class='warn'>{theme.pct(v['max_drawdown'])}</td></tr>"
        for s, v in result["per_asset"].items()
    )
    return (
        "<table><thead><tr><th>标的</th><th>年化</th><th>波动</th><th>最大回撤</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _insights_block(result: Mapping[str, Any]) -> str:
    found = insights_mod.evaluate(result)
    if not found:
        return "<p class='note'>当前数据没有触发任何提醒——这本身也是一个结论：没有哪一项越过了阈值。</p>"
    return '<div class="insights">' + "".join(
        theme.insight_html(
            {"level": i.level, "title": i.title, "card": i.card, "evidence": i.evidence}
        )
        for i in found
    ) + "</div>"


def _unlocks_block(result: Mapping[str, Any]) -> str:
    items = insights_mod.evaluate_unlocks(result.get("composition"))
    cards: list[str] = []
    for item in items:
        triggered = bool(item["triggered"])
        implemented = bool(item["implemented"])
        if triggered and implemented:
            badge, css = '<span class="badge on">已解锁</span>', "unlock on"
        elif triggered:
            badge, css = '<span class="badge pending">已触发 · 待接入</span>', "unlock on"
        else:
            badge, css = f'<span class="badge">🔒 需{item["requirement_label"]}</span>', "unlock"
        pending = f'<div class="s">待办：{theme.esc(item["pending"])}</div>' if not implemented and triggered else ""
        hint = f'<div class="s">触发条件：{theme.esc(item["trigger"])}</div>'
        body = ""
        if triggered:
            body = (
                f'<details class="insight"><summary>先了解这个概念 →</summary>'
                f'<div class="body">{theme.card_html(str(item["card"]))}</div></details>'
            )
        cards.append(
            f'<div class="{css}"><div class="t">{theme.esc(item["title"])} {badge}</div>{hint}{pending}{body}</div>'
        )
    unlocked = sum(1 for i in items if i["triggered"])
    return (
        f"<p class='note'>已触发 {unlocked}/{len(items)} 个模块。"
        "配出对应结构才会出现——知识不是靠翻页找到的，是被你的组合问出来的。</p>"
        '<div class="unlocks">' + "".join(cards) + "</div>"
    )


def render_dashboard(result: Mapping[str, Any], *, prefix: str = "", figs: dict[str, Any] | None = None) -> str:
    """一个组合的完整仪表盘（不含页头页脚），供独立页与首屏档位切换共用。

    ``prefix`` 会加到每个图表容器的 id 前面。**这是必需的**：首屏把多个组合的仪表盘
    内联在同一页里，若 id 重复，浏览器 ``getElementById`` 只返回第一个匹配，
    Plotly 会把所有图都画进第一档的容器，其余档位一片空白——本项目真的踩过这个坑。

    ``figs`` 收集各图的 data/layout，由调用方写进独立的 ``data/*.figs.js``。
    """
    weights_chips = " · ".join(f"{theme.esc(s)} {w:.0%}" for s, w in result["weights"].items())

    # 条件渲染：面板与图表数据必须**同时**出现或同时缺席，
    # 否则会出现"有容器没数据"或"有数据没容器"（后者会让 id 对齐测试失败，前者是空白图）。
    rates_data = result.get("rates") or {}
    rate_panels = ""
    if rates_data.get("available"):
        rate_panels = (
            theme.panel("利率环境与无风险利率", _rates_tables(result), span=4)
            + theme.panel("国债收益率曲线", theme.figure_div(figures.fig_yield_curve(result), f"{prefix}fig-ycurve", figs), span=4)
            + theme.panel("国债收益率历史", theme.figure_div(figures.fig_yield_history(result), f"{prefix}fig-yhist", figs), span=4)
        )
    duration_panel = ""
    if (result.get("composition") or {}).get("has_bond") and rates_data.get("scenarios"):
        duration_panel = theme.panel(
            "久期与利率冲击",
            theme.figure_div(figures.fig_rate_scenarios(result), f"{prefix}fig-scen", figs) + _duration_tables(result),
            span=12,
        )

    return f"""
<div class="titlebar">
  <div>
    <h1>{theme.esc(result['name'])}</h1>
    <div class="q">{theme.esc(result['question'])}</div>
  </div>
  <div class="sub">{theme.esc(weights_chips)}<br>{theme.esc(result['start'])} ~ {theme.esc(result['end'])}（{result['n_obs']} 个交易日）</div>
</div>

{_metrics_keyboard(result)}

<div class="grid" style="margin-top:12px">
  {theme.panel("净值与水下曲线", theme.figure_div(figures.fig_nav(result), f"{prefix}fig-nav", figs), span=8)}
  {theme.panel("收益归因", theme.figure_div(figures.fig_return_contribution(result), f"{prefix}fig-attrib", figs)
    + "<p class='note'>归因用<b>对数贡献</b>（各标的的对数收益 × 权重）：它扣掉了每个标的自身的复利效应，"
    + "量级与实际收益可比。算术贡献会被各标的自身的波动拖累主导（长周期里单一标的能到 +8000bp，"
    + "而组合实际累计只有几十个百分点），那种数字无法解读。</p>"
    + "<p class='note'>各标的对数贡献之和 "
    + theme.pct((result.get('risk_contribution') or {}).get('log_contribution_sum'))
    + "，组合实际对数收益 "
    + theme.pct((result.get('risk_contribution') or {}).get('log_total_return'))
    + "，差额 <b>"
    + theme.pct((result.get('risk_contribution') or {}).get('rebalancing_effect'))
    + "</b> 就是<b>再平衡/分散化效应</b>——由 Jensen 不等式它恒为非负："
    + "每日再平衡会在波动中不断把权重拉回目标，从而多得一部分收益。</p>", span=4)}
  {theme.panel("权重 vs 风险贡献", theme.figure_div(figures.fig_risk_vs_weight(result), f"{prefix}fig-risk", figs), span=6)}
  {theme.panel("回撤最深的前五段", _drawdown_table(result), span=6)}
  {theme.panel("定投：三种收益率口径", _dca_table(result), span=6)}
  {theme.panel("定投：市值 vs 累计投入", theme.figure_div(figures.fig_dca(result), f"{prefix}fig-dca", figs), span=6)}
  {theme.panel("滚动一年夏普", theme.figure_div(figures.fig_rolling_sharpe(result), f"{prefix}fig-roll", figs), span=4)}
  {theme.panel("各标的单独持有", _per_asset_table(result), span=4)}
  {theme.panel("各标的年化 vs 最大回撤", theme.figure_div(figures.fig_per_asset(result), f"{prefix}fig-asset", figs), span=4)}
  {theme.panel("因子敞口矩阵（热力图）", theme.figure_div(figures.fig_exposure_heatmap(result), f"{prefix}fig-expo", figs), span=12)}
  {theme.panel("敞口明细与拟合质量", _exposure_table(result), span=12)}
  {rate_panels}
  {duration_panel}
  {theme.panel("波动率期限结构", theme.figure_div(figures.fig_vol_term_structure(result), f"{prefix}fig-vol", figs)
    + "<p class='note'>不同回看窗口下的<b>历史</b>波动率。它不是隐含波动率——"
    + "隐含波动率是市场对<b>未来</b>波动的定价，恐慌时会显著高于历史波动率，"
    + "这正是「最需要保险时保险最贵」的来源。</p>", span=5)}
  {theme.panel("保护成本曲线", theme.figure_div(figures.fig_protection_curve(result), f"{prefix}fig-prot", figs), span=7)}
  {theme.panel("保护成本与 Greeks（理论值）", _derivatives_table(result), span=12)}
  {theme.panel("蒙特卡洛：终值分布扇形图", theme.figure_div(figures.fig_mc_fan(result), f"{prefix}fig-mcfan", figs), span=7)}
  {theme.panel("终值分布直方图", theme.figure_div(figures.fig_mc_histogram(result), f"{prefix}fig-mchist", figs), span=5)}
  {theme.panel("收敛诊断：路径数够不够", theme.figure_div(figures.fig_mc_convergence(result), f"{prefix}fig-mcconv", figs)
    + "<p class='note'>标准误应大致按 1/√N 下降（双对数图上是一条斜率 −0.5 的直线）。"
    + "偏离这条线说明结果对路径数仍敏感，那个数字就还在抖。</p>", span=5)}
  {theme.panel("四个模型的结论对比", _monte_carlo_tables(result), span=7)}
  {theme.panel("历史情节重放", theme.figure_div(figures.fig_episodes(result), f"{prefix}fig-epi", figs)
    + _episodes_block(result), span=12)}
  {theme.panel("洞察（由数据触发）", _insights_block(result), span=12)}
  {theme.panel("可解锁模块", _unlocks_block(result), span=12)}
</div>
"""


def render_preset_page(result: Mapping[str, Any], *, dashboard: str, root: str = "") -> str:
    """独立组合页。复用首屏已经渲染好的仪表盘 HTML，避免重复构建图表。

    图表数据与首屏共用同一个 ``data/<key>.figs.js``（容器 id 带同一个前缀），
    因此同一份数据只写一次、两处都能用。
    """
    key = str(result.get("key", "preset"))
    body = (
        dashboard
        + f'\n<script src="{root}assets/lab.js"></script>'
        + f'\n<script>labInitSingle("{key}", "{root}data/{key}.figs.js");</script>'
    )
    return _page(
        f"{result['name']} · ETF 组合数值实验室",
        body,
        root=root,
        data_version=str(result.get("data_version", "—")),
    )


def _gear_css(count: int) -> str:
    """档位切换所需的 :checked 规则（数量随组合数变化，因此动态生成）。"""
    rules = []
    for index in range(count):
        rules.append(f'#gear{index}:checked ~ .gear-panes > #pane{index} {{ display:block; }}')
        rules.append(
            f'#gear{index}:checked ~ .gear-labels label[for="gear{index}"]'
            " { background:var(--accent); color:#0b0e13; border-color:var(--accent); }"
        )
    return "<style>" + "".join(rules) + "</style>"


def render_index(
    results: Sequence[Mapping[str, Any]],
    *,
    dashboards: Mapping[str, str],
    counts: Mapping[str, int],
    data_version: str,
    root: str = "",
) -> str:
    """首屏：**一进来就是数据**。档位切换用纯 CSS，图表数据按档位**懒加载**。

    懒加载是必需的：三档的图表 JSON 合计约 785 KB（占原页面 85%），
    一次性全塞进 HTML 既拖慢首屏也不可 diff。现在首屏只装载当前档位的数据，
    切换时再取——每个档位一个独立文件，浏览器可缓存。
    """
    inputs = "".join(
        f'<input class="gear-input" type="radio" name="gear" id="gear{i}"'
        f' data-key="{theme.esc(str(r.get("key")))}" data-src="{root}data/{theme.esc(str(r.get("key")))}.figs.js"'
        f'{" checked" if i == 0 else ""}>'
        for i, r in enumerate(results)
    )
    labels = "".join(
        f'<label for="gear{i}">{theme.esc(r.get("name", r.get("key")))}</label>' for i, r in enumerate(results)
    )
    custom = (
        '<details class="gear-custom"><summary>自定义组合 →</summary>'
        '<div class="card" style="margin-top:8px">'
        "<p>自定义权重要实时重算，属于<b>实算应用</b>（路线 C）。本地启动：</p>"
        '<div class="formula">python -m etf_lab.cli app</div>'
        '<p>然后打开 <code>http://127.0.0.1:8080/lab</code> 拖动权重。<br>'
        "静态页只做预计算——这样它零服务器、离线可用、也永远不会挂。</p>"
        "</div></details>"
    )
    panes = "".join(
        f'<section class="gear-pane" id="pane{i}">{dashboards.get(str(r.get("key")), "")}</section>'
        for i, r in enumerate(results)
    )
    counts_rows = "".join(f"<tr><td>{theme.esc(k)}</td><td>{v:,}</td></tr>" for k, v in counts.items() if v)
    body = f"""
{_gear_css(len(results))}
<div class="gear-wrap">
  {inputs}
  <div class="gear-labels">{labels}{custom}</div>
  <div class="gear-panes">{panes}</div>
</div>
<div class="grid" style="margin-top:12px">
  {theme.panel("数据底座", "<table><thead><tr><th>表</th><th>行数</th></tr></thead><tbody>" + counts_rows + "</tbody></table>"
    + f"<p class='note'>数据版本 <code>{theme.esc(data_version)}</code>。行情来自公开接口，数据不随仓库分发。</p>", span=6)}
  {theme.panel("怎么读这个站", '''<p class="note">上面每一格数字里都有 <span class="hintmark">◂</span>，点开才是公式与「什么时候会骗人」。默认视图不放讲解。</p>
  <p class="note">页面里的<b>洞察条</b>不是写好的文案，而是规则引擎读你这份组合算出来的数字后浮出来的——
  换个组合，浮出来的提醒就变了。配出特定结构（含跨境、含债券、带对冲）还会解锁对应模块。</p>
  <p class="note">图表数据按档位单独存放、切换时按需加载；这样首页只有几十 KB，
  而每个档位的数据文件都会被浏览器缓存。曲线做了抽稀并保留极值点，
  画面上看不出差别，体积却小很多。</p>''', span=6)}
</div>
<script src="{root}assets/lab.js"></script>
<script>labInitTabs();</script>
"""
    return _page("ETF 组合数值实验室 · 组合工作台", body, root=root, data_version=data_version)


def render_concepts(*, root: str = "") -> str:
    keys = [
        "annualized_return", "volatility", "sharpe", "sortino", "calmar", "max_drawdown",
        "var", "cvar", "xirr", "correlation", "diversification_ratio", "beta",
        "tracking_error", "adjustment", "risk_contribution", "premium_discount",
        "arithmetic_vs_geometric", "leverage_unwind", "liquidity_spiral",
        "implied_volatility", "credit_spread_cds", "fx_exposure", "duration",
        "greeks", "protection_cost", "monte_carlo", "tail_crossover", "hedging",
    ]
    cards = "".join(theme.card_html(k) for k in keys)
    body = f"""
<div class="titlebar"><div><h1>知识附录</h1>
<div class="q">这里只是索引。正常使用不需要来这一页——每张卡片都能从某个数字或某条洞察就地展开。</div></div></div>
{theme.panel("全部知识卡片", f'<div style="columns:2;column-gap:14px">{cards}</div>', span=12)}
"""
    return _page("知识附录 · ETF 组合数值实验室", body, root=root)


def render_about(*, counts: Mapping[str, int], data_version: str, root: str = "") -> str:
    counts_rows = "".join(f"<tr><td>{theme.esc(k)}</td><td>{v:,}</td></tr>" for k, v in counts.items())
    body = f"""
<div class="titlebar"><div><h1>关于与口径</h1><div class="q">{theme.esc(teaching.DISCLAIMER)}</div></div></div>
<div class="grid">
  {theme.panel("数据来源与口径", '''<table><tbody>
  <tr><td>ETF 行情</td><td>腾讯公开接口（主源），按区间分页取完整历史</td></tr>
  <tr><td>独立校验源</td><td>搜狐公开接口（未复权价），逐日交叉校验，中位差异为 0</td></tr>
  <tr><td>复权口径</td><td><strong>前复权</strong>用于一切收益计算；同时保留未复权价与复权因子</td></tr>
  <tr><td>指数成分股</td><td>不使用成分股名单；风险敞口将改用收益法风格分析（RBSA）</td></tr>
  <tr><td>再平衡假设</td><td>每日再平衡（权重每天回到目标值）</td></tr>
  <tr><td>无风险利率</td><td>教学场景固定 ''' + theme.pct(RF_ANNUAL_DEFAULT) + '''，页面显示具体取值</td></tr>
  <tr><td>缺失值</td><td>不做任何填充；标的未上市期间直接排除该日期</td></tr>
  <tr><td>风险贡献</td><td>成分 VaR 的欧拉分解（正态近似），之和等于组合 VaR</td></tr>
  </tbody></table>''', span=6)}
  {theme.panel("数据底座", "<table><thead><tr><th>表</th><th>行数</th></tr></thead><tbody>" + counts_rows + "</tbody></table>"
    + f"<p class='note'>数据版本 <code>{theme.esc(data_version)}</code>。</p>", span=6)}
  {theme.panel("尚未接入", '''<p class="note">国债收益率曲线、股指期货基差与展期成本、ETF 期权与隐含波动率、汇率——
  这四类数据需要另找公开接口。它们对应的概念卡片（久期、Greeks、汇率贡献、对冲成本）
  已经可以在「可解锁模块」里提前读到，但暂无实测数字。</p>''', span=12)}
</div>
"""
    return _page("口径 · ETF 组合数值实验室", body, root=root, data_version=data_version)


def _copy_plotly_js(out_dir: Path) -> Path:
    """把 plotly.min.js 复制一份到 assets/，各页面共享（每页内联会让站点膨胀到几十 MB）。"""
    import plotly

    source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not source.exists():  # pragma: no cover - 依赖包结构变化时给出明确指引
        raise RuntimeError(f"未找到 plotly.min.js：{source}；请确认 plotly 版本")
    target_dir = out_dir / "assets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "plotly.min.js"
    shutil.copyfile(source, target)
    return target


def export_preset_prices(con, out_dir: Path) -> Path | None:
    """导出示例组合用到的价格序列（供路线 B 的 Pyodide 页面直接吃）。"""
    symbols: list[str] = []
    for spec in PRESETS:
        symbols.extend(s for s in spec.weights if s not in symbols)
    panel = repo.read_price_panel(con, symbols, field="close_adj")
    if panel.empty:
        return None
    payload = {
        "data_version": repo.latest_data_version(con),
        "field": "close_adj",
        "note": "前复权收盘价；日期为交易日，缺失表示该标的当日无数据（不做填充）",
        "dates": [str(idx.date()) for idx in panel.index],
        "series": {c: [None if pd.isna(v) else round(float(v), 4) for v in panel[c]] for c in panel.columns},
    }
    target_dir = out_dir / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "preset_prices.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def build(out_dir: str | Path = "docs", db_path: str | Path | None = None, rf_annual: float | None = None) -> Path:
    """生成整站，返回输出目录。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = repo.connect(db_path)
    counts = repo.table_counts(con)
    data_version = repo.latest_data_version(con)
    if counts.get("etf_price", 0) == 0:
        raise RuntimeError("本地数据仓还没有行情数据，请先运行：python -m etf_lab.cli fetch --preset core")

    _copy_plotly_js(out)
    (out / "assets" / "style.css").write_text(theme.STYLE, encoding="utf-8")

    results: list[dict[str, Any]] = []
    dashboards: dict[str, str] = {}
    figs_by_key: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for spec in PRESETS:
        try:
            result = compute_preset(con, spec, rf_annual=rf_annual)
            figs: dict[str, Any] = {}
            # 仪表盘只渲染一次，首屏与独立页共用；图表数据同时被收集起来写文件
            dashboards[spec.key] = render_dashboard(result, prefix=f"{spec.key}-", figs=figs)
            figs_by_key[spec.key] = figs
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - 单个组合作不出来不应让整站失败
            failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")

    # 图表 JSON 外置：页面因此只有几十 KB，数据文件按档位懒加载且可被缓存
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, figs in figs_by_key.items():
        (data_dir / f"{key}.figs.js").write_text(theme.figure_data_js(figs), encoding="utf-8")
    (out / "assets" / "lab.js").write_text(theme.LAB_JS, encoding="utf-8")

    (out / "index.html").write_text(
        render_index(results, dashboards=dashboards, counts=counts, data_version=data_version), encoding="utf-8"
    )
    for result in results:
        key = str(result["key"])
        (out / f"{key}.html").write_text(
            render_preset_page(result, dashboard=dashboards[key]), encoding="utf-8"
        )
    (out / "concepts.html").write_text(render_concepts(), encoding="utf-8")
    (out / "about.html").write_text(render_about(counts=counts, data_version=data_version), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    export_preset_prices(con, out)

    if failures:
        print("以下组合未能生成（已跳过，未做填充）：")
        for line in failures:
            print(f"  - {line}")
    return out
