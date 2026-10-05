"""路线 A：把计算结果渲染成**纯静态、可离线打开**的交互式报告站。

设计要点
--------
1. **计算与渲染分离**：:func:`compute_preset` 只返回可 JSON 序列化的字典
   （因此可缓存、可被路线 C 的 NiceGUI 复用、也可以直接喂给路线 B 的 Pyodide），
   :func:`render_*` 才负责把它变成 HTML 与图表。
2. **plotly.js 只放一份**：每个页面都内联一份 plotly.min.js 会让站点变成几十 MB；
   这里统一放 ``assets/plotly.min.js`` 由各页共享。
3. **口径全部写进页面**：复权方式、区间、无风险利率取值、数据版本。
   一个不写口径的收益数字，本质上不可复核。
4. **静态产物寿命长**：不依赖任何后端，拷贝即部署，GitHub Pages 直接可用。
"""

from __future__ import annotations

import datetime as dt
import html
import json
import shutil
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from etf_lab import __version__
from etf_lab.content import teaching
from etf_lab.core import correlation, dca, metrics, returns
from etf_lab.data import repo
from etf_lab.presets import PRESETS, PortfolioSpec

RF_ANNUAL_DEFAULT = 0.02
"""教学场景下固定的无风险利率。真实使用时应从国债收益率曲线取，并在页面标注。"""


# --------------------------------------------------------------------------- #
# 计算层：只产出可序列化的字典
# --------------------------------------------------------------------------- #
def _series_to_pairs(series: pd.Series, precision: int = 6) -> list[list[Any]]:
    return [[str(idx.date()), round(float(val), precision)] for idx, val in series.items()]


def _asset_summary(nav: pd.Series) -> dict[str, Any]:
    rets = nav.pct_change().dropna()
    info = metrics.max_drawdown(nav)
    return {
        "annualized_return": round(metrics.annualized_return(nav), 6),
        "annualized_volatility": round(metrics.annualized_volatility(rets), 6),
        "max_drawdown": round(info.depth, 6),
        "total_return": round(returns.total_return(nav), 6),
    }


def compute_preset(
    con,
    spec: PortfolioSpec,
    rf_annual: float = RF_ANNUAL_DEFAULT,
    start: str | None = None,
) -> dict[str, Any]:
    """算一个示例组合的全部展示数据。"""
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
    # 组合层面的指标必须用**单列**的组合收益；rets 是多资产面板，只用于相关性分析
    port_rets = nav.pct_change().dropna()
    port_rets.name = "portfolio"
    info = metrics.max_drawdown(nav)

    corr = correlation.correlation_matrix(rets, method="pearson", min_obs=60)
    order = correlation.cluster_order(corr)
    corr_ordered = corr.loc[order, order]

    per_asset: dict[str, Any] = {}
    for symbol in aligned.columns:
        per_asset[symbol] = _asset_summary(returns.nav_from_prices(aligned[symbol]))

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
            years = (
                (invested_curve.index[-1] - first_contrib.index[0]).days / 365.25 if len(first_contrib) else float("nan")
            )
            simple_on_invested = float(result.metrics.get("simple_return_on_invested", float("nan")))
            # 把累计收益错误地"当年化"（等价于假设所有钱在期初一次性投入）
            naive_annualized = (
                (1.0 + simple_on_invested) ** (1.0 / years) - 1.0 if years and years > 0 and simple_on_invested > -1 else float("nan")
            )
            dca_runs[mode] = {
                "value": _series_to_pairs(result.value),
                "invested": _series_to_pairs(invested_curve),
                "xirr": None if not np.isfinite(result.xirr) else round(float(result.xirr), 6),
                "invested_total": round(result.invested_total, 2),
                "final_value": round(result.final_value, 2),
                "cumulative_return_on_invested": round(simple_on_invested, 6),
                "naive_annualized_return": None if not np.isfinite(naive_annualized) else round(float(naive_annualized), 6),
                "years": None if not np.isfinite(years) else round(float(years), 3),
                "n_contributions": int(result.metrics.get("n_contributions", 0)),
                "total_fees": round(float(result.metrics.get("total_fees", 0.0)), 2),
            }
        except Exception as exc:  # noqa: BLE001 - 单个模式失败不应让整页打不开
            dca_runs[mode] = {"error": f"{type(exc).__name__}: {exc}"}

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
        "nav": _series_to_pairs(nav),
        "drawdown": _series_to_pairs(drawdown),
        "correlation": {
            "labels": [str(c) for c in corr_ordered.columns],
            "matrix": [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in corr_ordered.to_numpy()],
        },
        "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.summary(nav, port_rets, rf_annual=rf_annual).items()},
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
            for d in metrics.top_drawdowns(nav, top_n=4)
        ],
        "per_asset": per_asset,
        "dca": dca_runs,
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
    <a href="$rootindex">示例组合</a>
    <a href="$rootconcepts">概念与陷阱</a>
    <a href="$rootabout">关于与口径</a>
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


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _figure_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config={"displaylogo": False, "responsive": True})


def _metric_row(label: str, value: str, card_key: str | None = None) -> str:
    anchor = f' <a class="cardlink" href="#card-{card_key}">?</a>' if card_key else ""
    return f'<div class="metric"><span class="metric-label">{_esc(label)}{anchor}</span><span class="metric-value">{_esc(value)}</span></div>'


def _pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _card_html(key: str) -> str:
    c = teaching.card(key)
    return (
        f'<div class="card" id="card-{key}">'
        f'<h3>{_esc(c["title"])}</h3>'
        f'<p class="formula">{_esc(c["formula"])}</p>'
        f'<p><strong>说明什么：</strong>{_esc(c["means"])}</p>'
        f'<p class="warn"><strong>什么时候会骗人：</strong>{_esc(c["misleads"])}</p>'
        f"</div>"
    )


# ---- 图表 ----------------------------------------------------------------- #
def fig_nav(result: Mapping[str, Any]) -> go.Figure:
    dates = [row[0] for row in result["nav"]]
    values = [row[1] for row in result["nav"]]
    dd_dates = [row[0] for row in result["drawdown"]]
    dd_values = [row[1] for row in result["drawdown"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name="组合净值", line={"color": "#2563eb", "width": 2}))
    fig.add_trace(
        go.Scatter(
            x=dd_dates,
            y=dd_values,
            name="回撤（右轴）",
            yaxis="y2",
            fill="tozeroy",
            line={"color": "#dc2626", "width": 1},
            opacity=0.35,
        )
    )
    fig.update_layout(
        title="净值曲线与水下曲线",
        yaxis={"title": "组合净值（起点 = 1）"},
        yaxis2={"title": "回撤", "overlaying": "y", "side": "right", "tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 60, "r": 60, "t": 70, "b": 40},
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def fig_dca(result: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    colors = {"fixed": "#2563eb", "value_avg": "#059669"}
    labels = {"fixed": "固定金额定投：组合市值", "value_avg": "价值平均定投：组合市值"}
    for mode, payload in result["dca"].items():
        if "error" in payload:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row[0] for row in payload["value"]],
                y=[row[1] for row in payload["value"]],
                name=labels.get(mode, mode),
                line={"color": colors.get(mode, "#666"), "width": 2},
            )
        )
    first = next((p for p in result["dca"].values() if "error" not in p), None)
    if first is not None:
        fig.add_trace(
            go.Scatter(
                x=[row[0] for row in first["invested"]],
                y=[row[1] for row in first["invested"]],
                name="累计投入本金",
                line={"color": "#9ca3af", "width": 2, "dash": "dash"},
            )
        )
    fig.update_layout(
        title="定投：组合市值 vs 累计投入本金",
        yaxis={"title": "金额（元）"},
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 60, "r": 40, "t": 70, "b": 40},
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def fig_correlation(result: Mapping[str, Any]) -> go.Figure:
    corr = result["correlation"]
    fig = go.Figure(
        go.Heatmap(
            z=corr["matrix"],
            x=corr["labels"],
            y=corr["labels"],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=[[("" if v is None else f"{v:.2f}") for v in row] for row in corr["matrix"]],
            texttemplate="%{text}",
            colorbar={"title": "ρ"},
        )
    )
    fig.update_layout(
        title="相关性矩阵（按聚类重排，方块结构越明显越容易分散）",
        margin={"l": 80, "r": 40, "t": 70, "b": 40},
        template="plotly_white",
    )
    return fig


def fig_per_asset(result: Mapping[str, Any]) -> go.Figure:
    symbols = list(result["per_asset"])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=[result["per_asset"][s]["annualized_return"] for s in symbols],
            name="年化收益",
            marker_color="#2563eb",
        )
    )
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=[result["per_asset"][s]["max_drawdown"] for s in symbols],
            name="最大回撤",
            marker_color="#dc2626",
        )
    )
    fig.update_layout(
        title="各标的单独持有：年化收益 vs 最大回撤",
        yaxis={"tickformat": ".0%"},
        barmode="group",
        legend={"orientation": "h", "y": 1.15},
        margin={"l": 60, "r": 40, "t": 70, "b": 40},
        template="plotly_white",
    )
    return fig


# ---- 页面 ----------------------------------------------------------------- #
def render_preset_page(result: Mapping[str, Any], *, root: str = "") -> str:
    m = result["metrics"]
    weights_rows = "".join(
        f"<tr><td>{_esc(symbol)}</td><td>{weight:.0%}</td></tr>" for symbol, weight in result["weights"].items()
    )
    dd_rows = "".join(
        f"<tr><td>{_pct(d['depth'])}</td><td>{d['peak']}</td><td>{d['trough']}</td>"
        f"<td>{d['recovery'] or '尚未修复'}</td><td>{d['duration_days']} 天</td>"
        f"<td>{d['recovery_days'] if d['recovery_days'] is not None else '—'}</td></tr>"
        for d in result["top_drawdowns"]
    )
    asset_rows = "".join(
        f"<tr><td>{_esc(s)}</td><td>{_pct(v['annualized_return'])}</td>"
        f"<td>{_pct(v['annualized_volatility'])}</td><td>{_pct(v['max_drawdown'])}</td></tr>"
        for s, v in result["per_asset"].items()
    )
    dca_rows = ""
    for mode, payload in result["dca"].items():
        label = "固定金额" if mode == "fixed" else "价值平均"
        if "error" in payload:
            dca_rows += f'<tr><td>{_esc(label)}</td><td colspan="6">计算失败：{_esc(payload["error"])}</td></tr>'
            continue
        dca_rows += (
            f"<tr><td>{_esc(label)}</td>"
            f"<td>{payload['n_contributions']}</td>"
            f"<td>{payload['invested_total']:,.0f}</td>"
            f"<td>{payload['final_value']:,.0f}</td>"
            f"<td>{_pct(payload['cumulative_return_on_invested'])}</td>"
            f"<td class='warn-cell'>{_pct(payload['naive_annualized_return'])}</td>"
            f"<td><strong>{_pct(payload['xirr'])}</strong></td></tr>"
        )
    tw = result.get("time_weighted_annualized")

    metric_cards = "".join(
        _metric_row(label, value, key)
        for label, value, key in [
            ("年化收益", _pct(m.get("annualized_return")), "annualized_return"),
            ("年化波动率", _pct(m.get("annualized_volatility")), "volatility"),
            ("夏普比率", _num(m.get("sharpe")), "sharpe"),
            ("索提诺比率", _num(m.get("sortino")), "sortino"),
            ("卡玛比率", _num(m.get("calmar")), "calmar"),
            ("最大回撤", _pct(m.get("max_drawdown")), "max_drawdown"),
            ("VaR 95%（历史法）", _pct(m.get("var_95_historical")), "var"),
            ("CVaR 95%（历史法）", _pct(m.get("cvar_95_historical")), "cvar"),
            ("VaR 95%（参数法）", _pct(m.get("var_95_parametric")), "var"),
            ("偏度", _num(m.get("skewness"), 3), None),
            ("超额峰度", _num(m.get("excess_kurtosis"), 3), None),
            ("样本交易日", f"{m.get('n_obs')}", None),
        ]
    )

    teaching_keys = [
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "var",
        "cvar",
        "xirr",
        "correlation",
        "adjustment",
    ]

    body = f"""
<section class="hero">
  <h1>{_esc(result['name'])}</h1>
  <p class="question">这个组合要回答的问题：{_esc(result['question'])}</p>
  <p class="range">{_esc(result['start'])} ~ {_esc(result['end'])}（{result['n_obs']} 个交易日）</p>
  {f'<p class="caveat"><strong>本组合的特别之处：</strong>{_esc(result["caveat"])}</p>' if result.get('caveat') else ''}
</section>

<section>
  <h2>核心指标</h2>
  <div class="metrics">{metric_cards}</div>
  <p class="note">无风险利率取 {_pct(result['rf_annual'])}（教学场景固定值）。
  真实使用时应从国债收益率曲线读取，且页面必须显示所用取值——它直接改变夏普与索提诺。</p>
</section>

<section>
  <h2>净值与回撤</h2>
  {_figure_div(fig_nav(result), 'fig-nav')}
</section>

<section>
  <h2>回撤最深的前几段</h2>
  <table>
    <thead><tr><th>深度</th><th>高点</th><th>低点</th><th>修复日</th><th>下跌持续</th><th>修复用时</th></tr></thead>
    <tbody>{dd_rows}</tbody>
  </table>
  <p class="note">只报一个"最大回撤"会掩盖路径的差异：同样 −30%，跌得快恢复快和阴跌两年是完全不同的体验。</p>
</section>

<section>
  <h2>定投：两种收益率算法的差别</h2>
  {_figure_div(fig_dca(result), 'fig-dca')}
  <table>
    <thead><tr><th>定投方式</th><th>期数</th><th>累计投入</th><th>期末市值</th>
    <th>累计收益<br>（总收益 ÷ 总投入）</th><th>错误地把它当年化<br>（假设期初一次性投入）</th>
    <th>XIRR<br>（资金加权年化，正确）</th></tr></thead>
    <tbody>{dca_rows}</tbody>
  </table>
  <p class="note">三个数字放在一起才能看清问题：<strong>累计收益</strong>是全过程的总涨幅；
  把它按年数折成年化（第三列）隐含了"所有钱在期初就投入"的假设，定投里这个假设不成立；
  只有 <strong>XIRR</strong> 按每一笔钱的实际在场时间折算，才是定投的真实年化。
  作为参照，这段时间里<strong>组合本身</strong>的时间加权年化是 {_pct(tw)}——
  它衡量的是"标的涨了多少"，与"你的钱赚了多少"是两个不同的问题。</p>
  <p class="note">本页定投按<strong>组合净值</strong>成交，隐含"每日再平衡"假设；真实场内定投以 100 份为一手，
  零头会留在现金里，因此实盘结果会与此略有差异。</p>
</section>

<section>
  <h2>相关性结构</h2>
  {_figure_div(fig_correlation(result), 'fig-corr')}
  <p class="note">相关系数用过去的日收益估计。危机时相关性会上升，"分散化在最需要它的时候变弱"
  是这套方法最重要的局限。</p>
</section>

<section>
  <h2>各标的单独持有的表现</h2>
  {_figure_div(fig_per_asset(result), 'fig-asset')}
  <table>
    <thead><tr><th>标的</th><th>年化收益</th><th>年化波动</th><th>最大回撤</th></tr></thead>
    <tbody>{asset_rows}</tbody>
  </table>
</section>

<section>
  <h2>组合权重</h2>
  <table><thead><tr><th>标的</th><th>目标权重</th></tr></thead><tbody>{weights_rows}</tbody></table>
  <p class="note">计算采用<strong>每日再平衡</strong>假设，即权重每天回到目标值。
  真实的买入持有组合权重会随时间漂移，实际波动通常介于"不再平衡"与"每日再平衡"之间。</p>
</section>

<section>
  <h2>概念与陷阱</h2>
  <div class="cards">{''.join(_card_html(k) for k in teaching_keys)}</div>
</section>
"""
    return PAGE.substitute(
        title=f"{result['name']} · ETF 组合数值实验室",
        rootassets=f"{root}assets",
        rootindex=f"{root}index.html",
        rootconcepts=f"{root}concepts.html",
        rootabout=f"{root}about.html",
        body=body,
        disclaimer=_esc(teaching.DISCLAIMER),
        version=__version__,
        data_version=_esc(result["data_version"]),
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_index(preset_meta: Sequence[Mapping[str, Any]], *, counts: Mapping[str, int], data_version: str, root: str = "") -> str:
    items = "".join(
        f'<li><a href="{root}{p["key"]}.html"><strong>{_esc(p["name"])}</strong></a>'
        f'<span class="q">{_esc(p["question"])}</span>'
        f'<span class="r">{_esc(p.get("start", "—"))} ~ {_esc(p.get("end", "—"))}'
        f'{"（数据不完整）" if p.get("error") else ""}</span></li>'
        for p in preset_meta
    )
    counts_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>" for k, v in counts.items() if v)
    body = f"""
<section class="hero">
  <h1>组合数值实验室</h1>
  <p class="question">不是告诉你买什么，而是让你看清那些数字是怎么算出来的、代表什么、什么时候会骗你。</p>
  <p class="caveat">{_esc(teaching.DISCLAIMER)}</p>
</section>

<section>
  <h2>示例组合</h2>
  <ul class="presets">{items}</ul>
  <p class="note">每个示例组合都刻意对应一个教学问题。这些组合只是演示对象，不构成任何推荐。</p>
</section>

<section>
  <h2>本站想讲清楚的四件事</h2>
  <div class="cards">
    {''.join(_card_html(k) for k in ['xirr', 'max_drawdown', 'var', 'correlation'])}
  </div>
</section>

<section>
  <h2>数据底座</h2>
  <table><thead><tr><th>表</th><th>行数</th></tr></thead><tbody>{counts_rows}</tbody></table>
  <p class="note">数据版本 <code>{_esc(data_version)}</code>。
  行情来自公开接口（主源腾讯，独立校验源搜狐），<strong>数据不随仓库分发</strong>，请用采集脚本自行获取。</p>
</section>
"""
    return PAGE.substitute(
        title="ETF 组合数值实验室 · 教学演示",
        rootassets=f"{root}assets",
        rootindex=f"{root}index.html",
        rootconcepts=f"{root}concepts.html",
        rootabout=f"{root}about.html",
        body=body,
        disclaimer=_esc(teaching.DISCLAIMER),
        version=__version__,
        data_version=_esc(data_version),
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_concepts(*, root: str = "") -> str:
    keys = [
        "annualized_return",
        "volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "var",
        "cvar",
        "xirr",
        "correlation",
        "diversification_ratio",
        "beta",
        "tracking_error",
        "adjustment",
        "monte_carlo",
        "hedging",
    ]
    body = f"""
<section class="hero">
  <h1>概念与陷阱</h1>
  <p class="question">每个数字都配三件事：怎么算的、说明什么、什么时候会骗人。</p>
</section>
<section>
  <div class="cards">{''.join(_card_html(k) for k in keys)}</div>
</section>
"""
    return PAGE.substitute(
        title="概念与陷阱 · ETF 组合数值实验室",
        rootassets=f"{root}assets",
        rootindex=f"{root}index.html",
        rootconcepts=f"{root}concepts.html",
        rootabout=f"{root}about.html",
        body=body,
        disclaimer=_esc(teaching.DISCLAIMER),
        version=__version__,
        data_version="—",
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def render_about(*, counts: Mapping[str, int], data_version: str, root: str = "") -> str:
    counts_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{v:,}</td></tr>" for k, v in counts.items())
    body = f"""
<section class="hero">
  <h1>关于与口径</h1>
  <p class="caveat">{_esc(teaching.DISCLAIMER)}</p>
</section>
<section>
  <h2>数据来源与口径</h2>
  <table>
    <thead><tr><th>项目</th><th>说明</th></tr></thead>
    <tbody>
      <tr><td>ETF 行情</td><td>腾讯公开行情接口（主源），区间分页取完整历史</td></tr>
      <tr><td>独立校验源</td><td>搜狐公开行情接口（未复权价），用于交叉校验主源收盘价</td></tr>
      <tr><td>复权口径</td><td><strong>前复权</strong>用于一切收益计算；同时保留未复权价与复权因子</td></tr>
      <tr><td>指数成分股</td><td>不使用指数成分股名单；风险敞口改用收益法风格分析（RBSA）</td></tr>
      <tr><td>再平衡假设</td><td>每日再平衡（权重每天回到目标值）</td></tr>
      <tr><td>无风险利率</td><td>教学场景固定取值，页面显示具体数值</td></tr>
      <tr><td>缺失值</td><td>不做任何填充；标的未上市期间直接排除该日期</td></tr>
    </tbody>
  </table>
</section>
<section>
  <h2>数据底座</h2>
  <table><thead><tr><th>表</th><th>行数</th></tr></thead><tbody>{counts_rows}</tbody></table>
  <p class="note">数据版本 <code>{_esc(data_version)}</code>。</p>
</section>
<section>
  <h2>尚未接入</h2>
  <p class="note">国债收益率曲线、股指期货基差与展期成本、ETF 期权与隐含波动率、汇率——
  这四类数据需要另找公开接口。它们对应的教学内容（对冲成本、汇率贡献）在<a href="{root}concepts.html">概念页</a>已有文字说明，
  但暂未提供实测数字。</p>
</section>
"""
    return PAGE.substitute(
        title="关于与口径 · ETF 组合数值实验室",
        rootassets=f"{root}assets",
        rootindex=f"{root}index.html",
        rootconcepts=f"{root}concepts.html",
        rootabout=f"{root}about.html",
        body=body,
        disclaimer=_esc(teaching.DISCLAIMER),
        version=__version__,
        data_version=_esc(data_version),
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


STYLE = """
:root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --warn:#b91c1c; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; color:var(--fg); line-height:1.7; background:#fafafa; }
.site-header { display:flex; justify-content:space-between; align-items:center; padding:14px 24px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
.brand { font-weight:700; text-decoration:none; color:var(--fg); }
.site-header nav a { margin-left:16px; color:var(--muted); text-decoration:none; font-size:14px; }
.site-header nav a:hover { color:var(--accent); }
main { max-width: 980px; margin: 0 auto; padding: 24px 20px 60px; }
h1 { font-size: 30px; margin: 0 0 8px; }
h2 { font-size: 21px; margin: 36px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }
.hero { background:#fff; padding:22px; border:1px solid var(--line); border-radius:10px; }
.question { font-size:17px; color:var(--fg); margin:6px 0; }
.range { color:var(--muted); font-size:14px; margin:4px 0 0; }
.caveat { background:#fffbeb; border-left:3px solid #f59e0b; padding:10px 12px; margin-top:12px; font-size:14px; }
.metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap:10px; }
.metric { background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px 12px; display:flex; flex-direction:column; }
.metric-label { font-size:13px; color:var(--muted); }
.metric-value { font-size:20px; font-weight:600; }
.cardlink { text-decoration:none; color:var(--accent); font-weight:700; }
table { width:100%; border-collapse: collapse; background:#fff; font-size:14px; }
th, td { border:1px solid var(--line); padding:7px 10px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#f3f4f6; font-weight:600; }
.note { font-size:13px; color:var(--muted); background:#fff; border:1px dashed var(--line); padding:10px 12px; border-radius:8px; }
.cards { display:grid; gap:12px; }
.card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:8px; padding:14px 16px; }
.card h3 { margin:0 0 6px; font-size:17px; }
.card .formula { font-family: ui-monospace, Consolas, monospace; background:#f9fafb; padding:6px 8px; border-radius:6px; font-size:13px; }
.card .warn { color:var(--warn); }
ul.presets { list-style:none; padding:0; display:grid; gap:12px; }
ul.presets li { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
ul.presets a { text-decoration:none; color:var(--accent); font-size:17px; }
ul.presets .q, ul.presets .r { display:block; font-size:14px; color:var(--muted); }
footer { border-top:1px solid var(--line); background:#fff; padding:20px 24px; font-size:13px; color:var(--muted); }
.disclaimer { max-width: 980px; margin: 0 auto 6px; }
.meta { max-width: 980px; margin: 0 auto; }
code { background:#f3f4f6; padding:1px 5px; border-radius:4px; }
.warn-cell { color: var(--warn); }
"""


def _copy_plotly_js(out_dir: Path) -> Path:
    """把 plotly.min.js 复制一份到 assets/，各页面共享。"""
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
    (out / "assets" / "style.css").write_text(STYLE, encoding="utf-8")

    meta: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in PRESETS:
        try:
            result = compute_preset(con, spec, rf_annual=rf_annual)
            (out / f"{spec.key}.html").write_text(render_preset_page(result, root=""), encoding="utf-8")
            meta.append({"key": spec.key, "name": spec.name, "question": spec.question, "start": result["start"], "end": result["end"]})
        except Exception as exc:  # noqa: BLE001 - 单个组合作不出来不应让整站失败
            failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")
            meta.append({"key": spec.key, "name": spec.name, "question": spec.question, "error": str(exc)})

    (out / "index.html").write_text(render_index(meta, counts=counts, data_version=data_version), encoding="utf-8")
    (out / "concepts.html").write_text(render_concepts(), encoding="utf-8")
    (out / "about.html").write_text(render_about(counts=counts, data_version=data_version), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    export_preset_prices(con, out)

    if failures:
        print("以下组合未能生成（已跳过，未做填充）：")
        for line in failures:
            print(f"  - {line}")
    return out
