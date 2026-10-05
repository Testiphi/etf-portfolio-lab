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
from etf_lab.core import correlation, dca, metrics, returns
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


# --------------------------------------------------------------------------- #
# 计算层：只产出可序列化的字典
# --------------------------------------------------------------------------- #
def _series_to_pairs(series: pd.Series, precision: int = 6, step: int = 1) -> list[list[Any]]:
    """序列 → ``[[日期, 值], ...]``。``step`` 用于抽稀，避免 JSON 过大。"""
    trimmed = series.iloc[::step]
    return [[str(idx.date()), round(float(val), precision)] for idx, val in trimmed.items()]


def _asset_summary(nav: pd.Series) -> dict[str, Any]:
    rets = nav.pct_change().dropna()
    info = metrics.max_drawdown(nav)
    return {
        "annualized_return": round(metrics.annualized_return(nav), 6),
        "annualized_volatility": round(metrics.annualized_volatility(rets), 6),
        "max_drawdown": round(info.depth, 6),
        "total_return": round(returns.total_return(nav), 6),
    }


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
    """风险贡献（成分 VaR 占比）+ 收益归因（算术贡献，单位：小数）。

    **必须说清楚的一件事**：收益归因用的是**算术贡献** ``Σ_t wᵢ·rᵢₜ``。
    按每日再平衡假设，各标的算术贡献之和 = 组合各日收益之和，
    但它**不等于**复利后的累计收益——两者的差就是复利/再平衡效应。
    这是归因的经典难题（需要 Carino/Menchero 之类的链接方法才能精确分解），
    本项目选择**如实把差额展示出来**，而不是伪造一条"加总等于累计收益"的假不变量。

    成分 VaR 用欧拉分解，各标的占比之和为 1（这条是精确的，有测试守住）。
    """
    simple = returns.to_simple(rets, method="simple")
    contribution = {symbol: float((simple[symbol] * float(weights[symbol])).sum()) for symbol in weights}
    component = correlation.component_var(simple, weights, level=0.95)
    total = float(component.sum())
    share = {symbol: (float(component[symbol]) / total if total else None) for symbol in weights}
    return {
        "component_var": {symbol: round(float(component[symbol]), 8) for symbol in weights},
        "component_var_share": {symbol: (None if share[symbol] is None else round(share[symbol], 6)) for symbol in weights},
        "return_contribution": {symbol: round(value, 8) for symbol, value in contribution.items()},
        "return_contribution_sum": round(sum(contribution.values()), 8),
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
    rf_period = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
    excess = rets - rf_period
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std * np.sqrt(252)).dropna()


def compute_preset(
    con,
    spec: PortfolioSpec,
    rf_annual: float = RF_ANNUAL_DEFAULT,
    start: str | None = None,
) -> dict[str, Any]:
    """算一个组合的全部展示数据（纯数据，不含任何渲染）。"""
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
    # 组合层面的指标必须用**单列**的组合收益；rets 是多资产面板，只用于相关性与分解
    port_rets = nav.pct_change().dropna()
    port_rets.name = "portfolio"
    info = metrics.max_drawdown(nav)

    corr = correlation.correlation_matrix(rets, method="pearson", min_obs=60)
    order = correlation.cluster_order(corr)
    corr_ordered = corr.loc[order, order]

    per_asset = {symbol: _asset_summary(returns.nav_from_prices(aligned[symbol])) for symbol in aligned.columns}

    risk = _risk_contribution(rets, spec.weights)

    adjustment_frame = None
    try:
        adjustment_frame = repo.read_price_panel(con, symbols, start=start, field="adj_factor")
    except Exception:  # noqa: BLE001 - 缺 adj_factor 时不阻塞整页
        adjustment_frame = None

    diagnostics = _diagnostics(rets, spec.weights, risk["return_contribution"], adjustment_frame, len(aligned))

    meta = repo.read_etf_meta(con, symbols)
    class_by_symbol = dict(zip(meta.get("symbol", []), meta.get("asset_class", []))) if not meta.empty else {}
    by_class: dict[str, float] = {}
    for symbol, weight in spec.weights.items():
        asset_class = str(class_by_symbol.get(symbol, "unknown"))
        by_class[asset_class] = round(by_class.get(asset_class, 0.0) + float(weight), 6)
    composition = {
        "by_asset_class": by_class,
        **{f"has_{key}": key in by_class for key in ("broad", "bond", "gold", "cross_border", "industry")},
    }

    premium = _premium_block(con, symbols, spec.weights, start)

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

    rolling = _rolling_sharpe(nav, rf_annual)

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
        "rf_annual": rf_annual,
        "nav": _series_to_pairs(nav, step=1),
        "drawdown": _series_to_pairs(drawdown, step=1),
        "rolling_sharpe": _series_to_pairs(rolling, step=5),
        "correlation": {
            "labels": [str(c) for c in corr_ordered.columns],
            "matrix": [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in corr_ordered.to_numpy()],
        },
        "metrics": {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in metrics.summary(nav, port_rets, rf_annual=rf_annual).items()
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


def render_dashboard(result: Mapping[str, Any], *, prefix: str = "") -> str:
    """一个组合的完整仪表盘（不含页头页脚），供独立页与首屏档位切换共用。

    ``prefix`` 会加到每个图表容器的 id 前面。**这是必需的**：首屏把多个组合的仪表盘
    内联在同一页里，若 id 重复，浏览器 ``getElementById`` 只返回第一个匹配，
    Plotly 会把所有图都画进第一档的容器，其余档位一片空白——本项目真的踩过这个坑。
    """
    weights_chips = " · ".join(f"{theme.esc(s)} {w:.0%}" for s, w in result["weights"].items())
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
  {theme.panel("净值与水下曲线", theme.figure_html(figures.fig_nav(result), f"{prefix}fig-nav"), span=8)}
  {theme.panel("收益归因", theme.figure_html(figures.fig_return_contribution(result), f"{prefix}fig-attrib")
    + "<p class='note'>算术贡献各项之和 "
    + theme.pct((result.get('risk_contribution') or {}).get('return_contribution_sum'))
    + " 与复利后的实际累计收益 "
    + theme.pct((result.get('metrics') or {}).get('total_return'))
    + " 之间的差额，就是复利与再平衡效应——归因相加不等于累计收益，这是它的固有难点。</p>", span=4)}
  {theme.panel("权重 vs 风险贡献", theme.figure_html(figures.fig_risk_vs_weight(result), f"{prefix}fig-risk"), span=6)}
  {theme.panel("回撤最深的前五段", _drawdown_table(result), span=6)}
  {theme.panel("定投：三种收益率口径", _dca_table(result), span=6)}
  {theme.panel("定投：市值 vs 累计投入", theme.figure_html(figures.fig_dca(result), f"{prefix}fig-dca"), span=6)}
  {theme.panel("滚动一年夏普", theme.figure_html(figures.fig_rolling_sharpe(result), f"{prefix}fig-roll"), span=4)}
  {theme.panel("各标的单独持有", _per_asset_table(result), span=4)}
  {theme.panel("各标的年化 vs 最大回撤", theme.figure_html(figures.fig_per_asset(result), f"{prefix}fig-asset"), span=4)}
  {theme.panel("洞察（由数据触发）", _insights_block(result), span=12)}
  {theme.panel("可解锁模块", _unlocks_block(result), span=12)}
</div>
"""


def render_preset_page(result: Mapping[str, Any], *, root: str = "") -> str:
    return _page(
        f"{result['name']} · ETF 组合数值实验室",
        render_dashboard(result, prefix=f"{result.get('key', 'p')}-"),
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


def render_index(results: Sequence[Mapping[str, Any]], *, counts: Mapping[str, int], data_version: str, root: str = "") -> str:
    """首屏：**一进来就是数据**。档位切换预先把每个组合的仪表盘都渲染进同一页，
    纯 CSS 切换，因此切组合不刷新、不跳页。"""
    inputs = "".join(
        f'<input class="gear-input" type="radio" name="gear" id="gear{i}"{" checked" if i == 0 else ""}>'
        for i in range(len(results))
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
        f'<section class="gear-pane" id="pane{i}">'
        f'{render_dashboard(r, prefix=str(r.get("key", f"p{i}")) + "-")}</section>'
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
  换个组合，浮出来的提醒就变了。配出特定结构（含跨境、含债券、带对冲）还会解锁对应模块。</p>''', span=6)}
</div>
"""
    return _page("ETF 组合数值实验室 · 组合工作台", body, root=root, data_version=data_version)


def render_concepts(*, root: str = "") -> str:
    keys = [
        "annualized_return", "volatility", "sharpe", "sortino", "calmar", "max_drawdown",
        "var", "cvar", "xirr", "correlation", "diversification_ratio", "beta",
        "tracking_error", "adjustment", "risk_contribution", "fx_exposure", "duration",
        "greeks", "monte_carlo", "hedging",
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


def build(out_dir: str | Path = "docs", db_path: str | Path | None = None, rf_annual: float = RF_ANNUAL_DEFAULT) -> Path:
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
    failures: list[str] = []
    for spec in PRESETS:
        try:
            result = compute_preset(con, spec, rf_annual=rf_annual)
            (out / f"{spec.key}.html").write_text(render_preset_page(result, root=""), encoding="utf-8")
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - 单个组合作不出来不应让整站失败
            failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")

    (out / "index.html").write_text(render_index(results, counts=counts, data_version=data_version), encoding="utf-8")
    (out / "concepts.html").write_text(render_concepts(), encoding="utf-8")
    (out / "about.html").write_text(render_about(counts=counts, data_version=data_version), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    export_preset_prices(con, out)

    if failures:
        print("以下组合未能生成（已跳过，未做填充）：")
        for line in failures:
            print(f"  - {line}")
    return out
