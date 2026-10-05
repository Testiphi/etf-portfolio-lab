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
    """收益归因：各标的的**对数贡献**（纵条，单位 %）。

    用纵条而不是横条，是因为横条在窄面板里配合长标的代码会把文字拱出画布；
    用对数贡献而不是算术贡献，是因为后者会被各标的自身的波动拖累主导
    （长周期里单一标的能到 +8000bp，而组合实际累计只有 +21%），量级无法解读。
    """
    risk = result.get("risk_contribution") or {}
    contrib = risk.get("log_contribution") or {}
    symbols = list(contrib)
    values = [float(contrib[s]) * 100 for s in symbols]
    colors = [P["ok"] if v >= 0 else P["warn"] for v in values]
    fig = go.Figure(
        go.Bar(
            x=symbols,
            y=values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in values],
            textposition="auto",
        )
    )
    note = ""
    if risk.get("log_total_return") is not None:
        note = f"（合计 {theme.pct(risk.get('log_contribution_sum'))}，组合实际对数收益 {theme.pct(risk.get('log_total_return'))}）"
    fig.update_layout(
        title=f"收益归因：对数贡献{note}",
        yaxis={"title": "对数贡献（%）", "ticksuffix": "%"},
    )
    return theme.dark(fig, height=280)


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


def fig_exposure_heatmap(result: dict) -> go.Figure:
    """因子敞口矩阵：行是各标的与组合，列是因子，格子里是受约束回归的 β。"""
    block = result.get("exposure") or {}
    rows = [r for r in (block.get("rows") or []) if r.get("betas")]
    factors = list(block.get("factors") or [])
    if not rows or not factors:
        fig = go.Figure()
        fig.update_layout(title="因子敞口矩阵（数据不足）")
        return theme.dark(fig, height=220)

    z = [[float(r["betas"].get(f, 0.0)) for f in factors] for r in rows]
    labels = [[f"{v:.2f}" for v in row] for row in z]
    # 把 R² 写进行标签：R² 很低时 β 本身没有意义，让不可靠一眼可见（表格里也有同样信息）
    row_labels = [f'{r["name"]} · R²{r["r_squared"]:.2f}' for r in rows]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=factors,
            y=row_labels,
            colorscale="Blues",
            text=labels,
            texttemplate="%{text}",
            colorbar={"title": "β", "thickness": 12, "outlinewidth": 0},
        )
    )
    fig.update_layout(title="因子敞口矩阵（β，受约束回归）")
    return theme.dark(fig, height=max(220, 70 + 26 * len(rows)))


def fig_episodes(result: dict) -> go.Figure:
    """把几个历史情节里的组合收益与市场收益并排放，落差一眼可见。"""
    items = [e for e in (result.get("episodes") or []) if e.get("portfolio") or e.get("market")]
    if not items:
        fig = go.Figure()
        fig.update_layout(title="历史情节（数据不足）")
        return theme.dark(fig, height=200)
    titles = [e["title"] for e in items]
    portfolio = [float(e["portfolio"]["total_return"]) if e.get("portfolio") else None for e in items]
    market = [float(e["market"]["total_return"]) if e.get("market") else None for e in items]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=titles, y=portfolio, name="你的组合", marker_color=P["accent"]))
    fig.add_trace(go.Bar(x=titles, y=market, name="沪深300", marker_color=P["muted"]))
    fig.update_layout(title="历史情节重放：组合 vs 市场", barmode="group", yaxis={"tickformat": ".0%"})
    return theme.dark(fig, height=300)


def fig_vol_term_structure(result: dict) -> go.Figure:
    """历史波动率的"期限结构"：各标的在不同回看窗口下的年化波动率。"""
    block = result.get("derivatives") or {}
    per_asset = block.get("per_asset") or {}
    fig = go.Figure()
    for index, (symbol, term) in enumerate(per_asset.items()):
        windows = sorted(int(k) for k in term)
        fig.add_trace(
            go.Scatter(
                x=windows,
                y=[term[str(w)] for w in windows],
                name=symbol,
                mode="lines+markers",
                line={"color": theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)]},
            )
        )
    overall = block.get("term_structure") or {}
    if overall:
        windows = sorted(int(k) for k in overall)
        fig.add_trace(
            go.Scatter(
                x=windows,
                y=[overall[str(w)] for w in windows],
                name="组合",
                mode="lines+markers",
                line={"color": P["fg"], "width": 3, "dash": "dot"},
            )
        )
    fig.update_layout(
        title="历史波动率期限结构",
        xaxis={"title": "回看交易日"},
        yaxis={"title": "年化波动率", "tickformat": ".0%"},
    )
    return theme.dark(fig, height=300)


def fig_protection_curve(result: dict) -> go.Figure:
    """保护成本曲线：买入认沽期权的权利金占标的价值的比例。"""
    rows = (result.get("derivatives") or {}).get("protection") or []
    fig = go.Figure()
    for index, tenor in enumerate(sorted({row["tenor_years"] for row in rows})):
        subset = sorted([row for row in rows if row["tenor_years"] == tenor], key=lambda row: row["moneyness"])
        fig.add_trace(
            go.Scatter(
                x=[row["moneyness"] for row in subset],
                y=[row["cost_pct"] for row in subset],
                name=f"{tenor * 12:.0f} 个月",
                mode="lines+markers",
                line={"color": theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)]},
            )
        )
    fig.update_layout(
        title="保护成本：买入认沽期权要花多少",
        xaxis={"title": "行权价 / 现价", "tickformat": ".0%"},
        yaxis={"title": "成本（占标的价值）", "tickformat": ".1%"},
    )
    return theme.dark(fig, height=300)


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
