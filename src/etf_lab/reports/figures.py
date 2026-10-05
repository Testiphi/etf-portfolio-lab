"""图表构建：每个函数接收 ``compute_preset`` 的结果字典，返回 Plotly 图。

与 ``theme.dark`` 配合统一深色样式。这里只做**排布**，不做计算——
所有数字都来自结果字典，因此同样的图在静态站与 NiceGUI 里完全一致。
"""

from __future__ import annotations

import plotly.graph_objects as go

from etf_lab.reports import theme

P = theme.PALETTE


def _xy(pairs: list[list]) -> tuple[list, list]:
    return [row[0] for row in pairs], [row[1] for row in pairs]


def fig_nav(result: dict) -> go.Figure:
    dates, values = _xy(result["nav"])
    dd_dates, dd_values = _xy(result["drawdown"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name="组合净值", line={"color": P["accent"], "width": 2}))
    fig.add_trace(
        go.Scatter(
            x=dd_dates,
            y=dd_values,
            name="回撤",
            yaxis="y2",
            fill="tozeroy",
            line={"color": P["warn"], "width": 1},
            opacity=0.35,
        )
    )
    fig.update_layout(
        title="净值与水下曲线",
        yaxis={"title": "净值（起点 1.0）"},
        yaxis2={"title": "回撤", "overlaying": "y", "side": "right", "tickformat": ".0%", "gridcolor": "rgba(0,0,0,0)"},
    )
    return theme.dark(fig, height=300)


def fig_return_contribution(result: dict) -> go.Figure:
    """收益归因：每只标的的算术贡献（bp）。

    算术贡献 ``Σ_t wᵢ·rᵢₜ`` 的各项之和等于组合各日收益之和，
    但**不等于**复利后的累计收益——面板标题里会把两个数都写出来，
    差额就是复利与再平衡效应。
    """
    contrib = result.get("risk_contribution", {}).get("return_contribution", {}) or {}
    symbols = list(contrib)
    values_bp = [float(contrib[s]) * 10000 for s in symbols]
    colors = [P["ok"] if v >= 0 else P["warn"] for v in values_bp]
    total_true = (result.get("metrics", {}) or {}).get("total_return")
    title = "收益归因（算术贡献 bp）"
    if total_true is not None:
        title += f"｜加总 {theme.pct(result.get('risk_contribution', {}).get('return_contribution_sum'))} vs 实际累计 {theme.pct(total_true)}"
    fig = go.Figure(
        go.Bar(
            x=values_bp,
            y=symbols,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.0f}" for v in values_bp],
            textposition="auto",
        )
    )
    fig.update_layout(title=title, xaxis={"title": "贡献（基点）"})
    return theme.dark(fig, height=260)


def fig_risk_vs_weight(result: dict) -> go.Figure:
    """权重 vs 风险贡献：两条横条并排，失衡一眼可见。"""
    shares = result.get("risk_contribution", {}).get("component_var_share", {}) or {}
    weights = result.get("weights", {}) or {}
    symbols = list(weights)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[float(weights.get(s, 0)) for s in symbols],
            y=symbols,
            orientation="h",
            name="权重",
            marker_color=P["accent"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=[float(shares.get(s, 0) or 0) for s in symbols],
            y=symbols,
            orientation="h",
            name="风险贡献",
            marker_color=P["warn"],
        )
    )
    fig.update_layout(
        title="权重 vs 风险贡献（成分 VaR）",
        barmode="group",
        xaxis={"title": "占比", "tickformat": ".0%"},
    )
    return theme.dark(fig, height=260)


def fig_dca(result: dict) -> go.Figure:
    fig = go.Figure()
    labels = {"fixed": "固定金额定投：市值", "value_avg": "价值平均定投：市值"}
    for index, (mode, payload) in enumerate(result["dca"].items()):
        if "error" in payload:
            continue
        dates, values = _xy(payload["value"])
        fig.add_trace(
            go.Scatter(x=dates, y=values, name=labels.get(mode, mode), line={"color": theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)], "width": 2})
        )
    first = next((p for p in result["dca"].values() if "error" not in p), None)
    if first is not None:
        dates, values = _xy(first["invested"])
        fig.add_trace(go.Scatter(x=dates, y=values, name="累计投入本金", line={"color": P["muted"], "width": 2, "dash": "dash"}))
    fig.update_layout(title="定投：市值 vs 累计投入", yaxis={"title": "金额（元）"})
    return theme.dark(fig, height=280)


def fig_correlation(result: dict) -> go.Figure:
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
            colorbar={"title": "ρ", "thickness": 12, "outlinewidth": 0},
        )
    )
    fig.update_layout(title="相关性矩阵（按聚类重排）")
    return theme.dark(fig, height=320)


def fig_per_asset(result: dict) -> go.Figure:
    symbols = list(result["per_asset"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=symbols, y=[result["per_asset"][s]["annualized_return"] for s in symbols], name="年化收益", marker_color=P["accent"]))
    fig.add_trace(go.Bar(x=symbols, y=[result["per_asset"][s]["max_drawdown"] for s in symbols], name="最大回撤", marker_color=P["warn"]))
    fig.update_layout(title="各标的单独持有：年化收益 vs 最大回撤", barmode="group", yaxis={"tickformat": ".0%"})
    return theme.dark(fig, height=260)


def fig_rolling_sharpe(result: dict) -> go.Figure:
    """滚动 1 年夏普：让"指标本身在飘"这件事可见。"""
    series = result.get("rolling_sharpe") or []
    fig = go.Figure()
    if series:
        dates, values = _xy(series)
        fig.add_trace(go.Scatter(x=dates, y=values, name="滚动 252 日夏普", line={"color": P["gold"], "width": 1.6}))
    fig.update_layout(title="滚动一年夏普（无风险利率见口径页）", yaxis={"title": "夏普"})
    return theme.dark(fig, height=260)
