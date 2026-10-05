"""收益风险指标：纯函数，无 I/O。

每个指标都必须能回答三个问题：**怎么算的、说明什么、什么时候会骗人**。
后者写在各自 docstring 的 ``Caveats`` 里，并应同步出现在网站的解释卡片上。

符号约定
--------
- ``VaR`` / ``CVaR`` 一律返回**正数表示损失**（例如 0.032 表示"在给定置信水平下，
  单期损失不超过 3.2%"），便于展示；不要与收益的负号混用。
- 所有年化默认按 ``252`` 个交易日换算，调用方可以覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DrawdownInfo:
    """一次最大回撤的完整描述。"""

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    depth: float
    """回撤幅度，负数（例如 -0.35 表示最大回撤 35%）。"""
    duration_days: int
    """从高点到低点的天数。"""
    recovery_days: int | None
    """从低点回到前高的天数；未回到前高则为 None。"""

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None


@dataclass(frozen=True)
class VarResult:
    """单期风险价值与条件风险价值。"""

    level: float
    method: str
    var: float
    cvar: float
    n_obs: int


# --------------------------------------------------------------------------- #
# 收益与波动
# --------------------------------------------------------------------------- #
def annualized_return(nav: pd.Series) -> float:
    """几何年化收益：``(末值/首值)^(1/年数) - 1``。

    Caveats
    -------
    用自然日折算年数，而不是按行数除以 252——后者在停牌或缺失较多时会失真。
    几何年化低于算术平均收益，两者不可互换（这是最常见的教学误区之一）。
    """
    if len(nav) < 2:
        raise ValueError("净值序列至少需要两个点")
    first, last = float(nav.iloc[0]), float(nav.iloc[-1])
    if first <= 0 or last <= 0:
        raise ValueError("净值必须为正")
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        raise ValueError("区间长度必须大于 0 天")
    return (last / first) ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化波动率：日收益标准差 × √252。

    Caveats
    -------
    波动率把上涨和下跌同等对待；对定投这类"持续买入"的策略，
    下行波动（Sortino 的分母）往往比总波动更有信息量。
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_return_from_returns(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """用**几何**方式从收益序列年化（与 ``annualized_return(nav)`` 等价但更稳健）。"""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    growth = float((1.0 + clean).prod())
    if growth <= 0:
        return float("nan")
    years = len(clean) / periods_per_year
    return growth ** (1.0 / years) - 1.0


# --------------------------------------------------------------------------- #
# 风险调整后收益
# --------------------------------------------------------------------------- #
def arithmetic_annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """算术平均年化：``日收益均值 × 252``。

    Caveats
    -------
    它**不是**你实际拿到的年化收益。算术平均忽略复利与波动，
    在波动存在时系统性高于几何年化，差额近似为 ``σ²/2``（波动拖累）。
    基金宣传页上那个"平均年化收益"常常就是它——这正是本站要纠正的认知偏差之一。
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean() * periods_per_year)


def volatility_drag(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """波动拖累：``算术年化 − 几何年化``，理论上近似 ``σ²/2``。

    这是"为什么波动会吃掉长期收益"的直接度量。波动越大，两者差距越大——
    所以拿算术平均去估算长期复利，会系统性地高估。
    """
    arithmetic = arithmetic_annualized_return(returns, periods_per_year)
    geometric = annualized_return_from_returns(returns, periods_per_year)
    if not np.isfinite(arithmetic) or not np.isfinite(geometric):
        return float("nan")
    return float(arithmetic - geometric)


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """夏普比率：每单位波动的超额收益。

    Caveats
    -------
    分子分母都依赖"无风险利率取多少"。国内常用 1 年期国债或 3 个月 shibor，
    两者在低利率与高利率环境下会让同一个组合的夏普差出可观幅度——
    所以页面上必须显示本次计算所用的 rf 值。
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_period
    sd = float(excess.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """索提诺比率：只用下行波动作为风险度量。

    Caveats
    -------
    只惩罚低于"无风险利率"的收益，不惩罚低于均值的收益；
    在样本中下行日很少时数值会极度放大（分母接近 0），需同时展示样本量。
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_period
    downside = excess[excess < 0.0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else float("nan")
    dd = float(np.sqrt((downside**2).mean()))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def calmar_ratio(nav: pd.Series) -> float:
    """卡玛比率：年化收益 / 最大回撤幅度。"""
    info = max_drawdown(nav)
    if info.depth == 0:
        return float("nan")
    return annualized_return(nav) / abs(info.depth)


# --------------------------------------------------------------------------- #
# 回撤
# --------------------------------------------------------------------------- #
def drawdown_series(nav: pd.Series) -> pd.Series:
    """水下曲线：当前净值相对历史最高点的跌幅（≤ 0）。"""
    if nav.empty:
        raise ValueError("净值序列为空")
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(nav: pd.Series) -> DrawdownInfo:
    """最大回撤及其高点、低点、修复时间。

    Caveats
    -------
    最大回撤是**单一历史路径**上的极值，对样本区间极度敏感：
    起点往前挪一年可能完全不同。因此页面上应同时给出"回撤最深的前 N 段"，
    而不是只报一个数字。
    """
    if len(nav) < 2:
        raise ValueError("净值序列至少需要两个点")
    dd = drawdown_series(nav)
    trough_date = dd.idxmin()
    depth = float(dd.loc[trough_date])
    peak_date = nav.loc[:trough_date].idxmax()

    after = nav.loc[trough_date:]
    peak_level = float(nav.loc[peak_date])
    recovered = after[after >= peak_level]
    recovery_date = recovered.index[0] if len(recovered) else None

    return DrawdownInfo(
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        depth=depth,
        duration_days=int((trough_date - peak_date).days),
        recovery_days=int((recovery_date - trough_date).days) if recovery_date is not None else None,
    )


def top_drawdowns(nav: pd.Series, top_n: int = 5) -> list[DrawdownInfo]:
    """回撤最深的前 N 段（彼此不重叠）。

    做法：找到最深的一段后，跳过它的区间再找下一段。这是一个近似但可解释的算法，
    比"取局部极值"更稳定。
    """
    if top_n < 1:
        raise ValueError("top_n 必须 ≥ 1")
    results: list[DrawdownInfo] = []
    remaining = nav.copy()
    for _ in range(top_n):
        if len(remaining) < 2:
            break
        info = max_drawdown(remaining)
        if info.depth >= 0:
            break
        results.append(info)
        end = info.recovery_date if info.recovery_date is not None else remaining.index[-1]
        remaining = remaining.loc[end:]
        # 去掉与刚取区间同起点的重复段
        if info.recovery_date is None and len(remaining) <= 1:
            break
    return results


# --------------------------------------------------------------------------- #
# 尾部风险
# --------------------------------------------------------------------------- #
def var_cvar(returns: pd.Series, level: float = 0.95, method: str = "historical") -> VarResult:
    """单期 VaR / CVaR（返回正数表示损失）。

    Parameters
    ----------
    level
        置信水平，例如 0.95。
    method
        ``"historical"``：直接用历史分位，不假设分布；
        ``"parametric"``：假设正态，用均值与标准差解析计算。

    Caveats
    -------
    历史法在样本不足或分布突变时会低估尾部风险；参数法在 A 股这种尖峰厚尾
    分布上会系统性低估极端损失。教学场景应**两种都展示**，让差异自己说话。
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level 必须在 (0.5, 1) 之间")
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("收益序列为空，无法计算 VaR")

    if method == "historical":
        q = float(clean.quantile(1.0 - level))
        tail = clean[clean <= q]
        cvar = float(-tail.mean()) if len(tail) else float("nan")
    elif method == "parametric":
        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        z = float(stats.norm.ppf(1.0 - level))
        q = mu + sigma * z
        cvar = float(-(mu - sigma * stats.norm.pdf(z) / (1.0 - level)))
    else:
        raise ValueError(f"未知 method：{method!r}（可选 historical / parametric）")

    return VarResult(level=level, method=method, var=float(-q), cvar=cvar, n_obs=int(len(clean)))


def skewness(returns: pd.Series) -> float:
    """偏度：<0 表示左尾更长，即大跌比大涨更容易出现。"""
    clean = returns.dropna()
    return float(clean.skew()) if len(clean) >= 3 else float("nan")


def excess_kurtosis(returns: pd.Series) -> float:
    """超额峰度：>0 表示尖峰厚尾，极端值比正态分布更常见。"""
    clean = returns.dropna()
    return float(clean.kurt()) if len(clean) >= 4 else float("nan")


# --------------------------------------------------------------------------- #
# 相对基准
# --------------------------------------------------------------------------- #
def beta_alpha(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> tuple[float, float, float]:
    """对基准做线性回归，返回 ``(beta, alpha_annualized, r_squared)``。

    Caveats
    -------
    Beta 是**事后**统计量：它衡量的是历史共同波动幅度，不代表未来一致。
    用不同基准（沪深300 vs 中证全指）算出的 Beta 会不同，必须显示所用基准。
    """
    df = pd.concat([returns.rename("p"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), float("nan")
    cov = float(df["p"].cov(df["b"]))
    var = float(df["b"].var(ddof=1))
    if var == 0:
        return float("nan"), float("nan"), float("nan")
    beta = cov / var
    alpha_period = float(df["p"].mean() - beta * df["b"].mean())
    alpha_annual = (1.0 + alpha_period) ** periods_per_year - 1.0
    corr = float(df["p"].corr(df["b"]))
    return beta, float(alpha_annual), corr**2


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """跟踪误差：组合与基准收益差的年化标准差。"""
    df = pd.concat([returns.rename("p"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    active = df["p"] - df["b"]
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summary(
    nav: pd.Series,
    returns: pd.Series | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int | str | None]:
    """一次性给出全套指标，供页面与缓存层直接消费。

    ``returns`` **必须是单列**（组合收益）。传入多资产面板是最容易犯的量纲错误：
    ``DataFrame.quantile()`` 会返回一行 Series，指标会算出毫无意义的结果或直接报错。
    因此这里显式拒绝，而不是让它静默算出一个数。
    """
    if returns is None:
        returns = nav.pct_change().dropna()
        returns.name = "returns"
    elif isinstance(returns, pd.DataFrame):
        raise TypeError(
            "summary 的 returns 必须是单列 Series（组合收益）；"
            f"收到 {returns.shape[1]} 列的面板。多资产请先用 core.returns.portfolio_returns 合成。"
        )
    info = max_drawdown(nav)
    hist = var_cvar(returns, level=0.95, method="historical")
    para = var_cvar(returns, level=0.95, method="parametric")
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": round((nav.index[-1] - nav.index[0]).days / 365.25, 3),
        "n_obs": int(len(returns.dropna())),
        "total_return": total_return(nav),
        "annualized_return": annualized_return(nav),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "arithmetic_annualized_return": arithmetic_annualized_return(returns, periods_per_year),
        "volatility_drag": volatility_drag(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf_annual=rf_annual, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, rf_annual=rf_annual, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(nav),
        "max_drawdown": info.depth,
        "max_drawdown_peak": str(info.peak_date.date()),
        "max_drawdown_trough": str(info.trough_date.date()),
        "max_drawdown_recovery": str(info.recovery_date.date()) if info.recovery_date is not None else None,
        "max_drawdown_duration_days": info.duration_days,
        "max_drawdown_recovery_days": info.recovery_days,
        "var_95_historical": hist.var,
        "cvar_95_historical": hist.cvar,
        "var_95_parametric": para.var,
        "cvar_95_parametric": para.cvar,
        "skewness": skewness(returns),
        "excess_kurtosis": excess_kurtosis(returns),
        "rf_annual_used": rf_annual,
    }


def total_return(nav: pd.Series) -> float:
    """区间累计收益（放在此处以便 ``summary`` 单点导入）。"""
    if len(nav) < 2:
        raise ValueError("净值序列至少需要两个点")
    first, last = float(nav.iloc[0]), float(nav.iloc[-1])
    if first <= 0:
        raise ValueError("净值首值必须为正")
    return last / first - 1.0
