"""相关性与分散化：纯函数，无 I/O。

教学要点：相关性是"分散化"这一整套说法的地基，但它有三个常见陷阱：
1. **相关性不是恒定的**——危机时相关性上升，分散化在最需要它的时候变弱；
2. **低相关 ≠ 降低风险**——一个长期负收益的资产也是低相关甚至负相关，
   但它只会把组合拖低。所以相关性必须和收益、波动一起看；
3. **对短样本估计的相关性极不稳定**，样本少于约 250 个交易日时标准误很大，
   页面上必须显示样本量。
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd
from scipy import cluster, stats

TRADING_DAYS_PER_YEAR = 252


def correlation_matrix(
    returns: pd.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
    min_obs: int = 60,
) -> pd.DataFrame:
    """相关矩阵。

    Parameters
    ----------
    method
        ``pearson`` 是线性相关；``spearman`` 是秩相关，对极端值更稳健，
        在尖峰厚尾的 A 股收益上两者差异本身就值得展示。
    min_obs
        两两配对的**最小有效样本量**，少于它的组合结果为 NaN 而不是勉强计算。
    """
    if returns.empty:
        raise ValueError("收益面板为空")
    if method == "pearson":
        corr = returns.corr(method="pearson", min_periods=min_obs)
    elif method == "spearman":
        corr = returns.corr(method="spearman", min_periods=min_obs)
    else:
        raise ValueError(f"未知 method：{method!r}")
    return corr


def pairwise_obs(returns: pd.DataFrame) -> pd.DataFrame:
    """两两配对的有效样本量——配合相关矩阵一起展示，避免过度解读。"""
    notna = returns.notna().astype(int)
    return notna.T @ notna


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """滚动相关系数，用来展示"相关性会变"。"""
    if window < 5:
        raise ValueError("window 至少为 5")
    return a.rolling(window).corr(b).rename(f"corr_{a.name}_{b.name}")


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """按层次聚类给出列顺序，使相关矩阵的方块结构可见。"""
    if corr.empty:
        return []
    # pandas 3.x 的 to_numpy() 可能返回只读视图，而 fill_diagonal 需要可写数组
    matrix = np.array(corr.fillna(0.0).to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(matrix, 1.0)
    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)
    condensed = distance[np.triu_indices_from(distance, k=1)]
    if len(condensed) == 0:
        return list(corr.columns)
    linkage = cluster.hierarchy.linkage(condensed, method="average")
    dendro = cluster.hierarchy.dendrogram(linkage, no_plot=True)
    return [str(corr.columns[i]) for i in dendro["leaves"]]


def diversification_ratio(returns: pd.DataFrame, weights: Mapping[str, float]) -> float:
    """分散化比率 = 加权平均波动 / 组合波动。>1 表示分散化起了作用。"""
    cols = list(weights)
    w = np.array([float(weights[c]) for c in cols])
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError("权重之和必须为 1")
    vol = returns[cols].std(ddof=1).to_numpy()
    portfolio_vol = float(np.sqrt(w @ returns[cols].cov().to_numpy() @ w))
    if portfolio_vol == 0:
        return float("nan")
    return float((w @ vol) / portfolio_vol)


def component_var(returns: pd.DataFrame, weights: Mapping[str, float], level: float = 0.95) -> pd.Series:
    """成分 VaR：把组合风险按权重与协方差分解到每个标的。

    采用正态近似下的欧拉分解（``w_i * (Σw)_i / σ_p * z``），
    因此**在厚尾数据上会低估尾部贡献**——这一点必须标注，
    教学上更适合与"历史法增量 VaR"并排对比。
    """
    cols = list(weights)
    w = np.array([float(weights[c]) for c in cols])
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError("权重之和必须为 1")
    cov = returns[cols].cov().to_numpy()
    sigma = float(np.sqrt(w @ cov @ w))
    if sigma == 0:
        return pd.Series(np.nan, index=cols, name="component_var")
    z = float(stats.norm.ppf(level))
    contrib = w * (cov @ w) / sigma * z
    return pd.Series(contrib, index=cols, name="component_var")


def marginal_impact(
    base_returns: pd.DataFrame,
    base_weights: Mapping[str, float],
    candidate: pd.Series,
    candidate_weight: float,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    """把一个新标的按给定权重加入组合，评估边际影响。

    Returns
    -------
    dict
        含加入前后的年化波动、VaR/CVaR、与现有组合的相关性等差异，
        以及 ``risk_contribution_share``（新资产占组合总风险的比例）。
        决策不应只看收益提升，也要看**风险贡献是否与权重相称**。

    Notes
    -----
    这是 M5「加入新板块」的第一个版本：只做"加进去会怎样"，
    不做优化求解（求解在 ``core.optimize`` 中实现）。
    """
    if not 0.0 <= candidate_weight < 1.0:
        raise ValueError("candidate_weight 必须在 [0, 1) 之间")
    cols = list(base_weights)
    w0 = np.array([float(base_weights[c]) for c in cols])
    if not np.isclose(w0.sum(), 1.0, atol=1e-8):
        raise ValueError("base_weights 之和必须为 1")

    frame = base_returns[cols].copy()
    frame["__candidate__"] = candidate
    frame = frame.dropna()
    if len(frame) < 30:
        raise ValueError(f"对齐后样本仅 {len(frame)} 条，不足以评估边际影响")

    new_w = np.append(w0 * (1.0 - candidate_weight), candidate_weight)
    cov = frame.cov().to_numpy()
    sigma0 = float(np.sqrt(w0 @ cov[: len(cols), : len(cols)] @ w0))
    sigma1 = float(np.sqrt(new_w @ cov @ new_w))

    def portfolio_of(weights: np.ndarray) -> pd.Series:
        return pd.Series(frame.to_numpy() @ weights, index=frame.index, name="port")

    # base 组合只含原标的列（frame 里还多一列候选资产）
    p0 = pd.Series(frame[cols].to_numpy() @ w0, index=frame.index, name="port")
    p1 = portfolio_of(new_w)

    def tail(series: pd.Series) -> tuple[float, float]:
        q = float(series.quantile(0.05))
        tail_values = series[series <= q]
        return -q, float(-tail_values.mean())

    var0, cvar0 = tail(p0)
    var1, cvar1 = tail(p1)
    corr = float(p0.corr(frame["__candidate__"]))
    z = float(stats.norm.ppf(0.95))
    marginal = float(cov[-1, :] @ new_w / sigma1 * z) if sigma1 > 0 else float("nan")
    risk_share = float(new_w[-1] * marginal / (sigma1 * z)) if sigma1 > 0 and z != 0 else float("nan")

    return {
        "candidate_weight": float(candidate_weight),
        "correlation_to_portfolio": corr,
        "volatility_before": sigma0 * np.sqrt(TRADING_DAYS_PER_YEAR),
        "volatility_after": sigma1 * np.sqrt(TRADING_DAYS_PER_YEAR),
        "volatility_change": (sigma1 - sigma0) * np.sqrt(TRADING_DAYS_PER_YEAR),
        "var95_before": var0,
        "var95_after": var1,
        "var95_change": var1 - var0,
        "cvar95_before": cvar0,
        "cvar95_after": cvar1,
        "cvar95_change": cvar1 - cvar0,
        "risk_contribution_share": risk_share,
        "weight_vs_risk_share": float(candidate_weight - risk_share) if np.isfinite(risk_share) else float("nan"),
        "annualized_return_before": float((1.0 + p0).prod() ** (TRADING_DAYS_PER_YEAR / len(p0)) - 1.0),
        "annualized_return_after": float((1.0 + p1).prod() ** (TRADING_DAYS_PER_YEAR / len(p1)) - 1.0),
        "rf_annual_used": rf_annual,
    }
