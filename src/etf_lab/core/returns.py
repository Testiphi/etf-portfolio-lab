"""收益序列与净值曲线：纯函数，无 I/O。

约定
----
- 价格/净值一律为 ``pandas`` 对象，索引是升序 ``DatetimeIndex``；
- ``DataFrame`` 的列是标的代码，行是交易日；
- 收益率有两种口径，全项目**不得混用**：
  ``log`` = 对数收益 ``ln(P_t / P_{t-1})``，``simple`` = 简单收益 ``P_t / P_{t-1} - 1``。
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

Method = Literal["log", "simple"]
TRADING_DAYS_PER_YEAR = 252


def _validate_ascending(obj: pd.Series | pd.DataFrame, name: str) -> None:
    """校验索引是升序 DatetimeIndex；不合法就报错，不猜测。"""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(f"{name} 的索引必须是 DatetimeIndex，实际是 {type(obj.index).__name__}")
    if not obj.index.is_monotonic_increasing:
        raise ValueError(f"{name} 的索引必须升序，请先排序而不是让本函数猜")
    if obj.index.has_duplicates:
        duplicates = obj.index[obj.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"{name} 的索引存在重复日期，例如 {duplicates}")


def to_returns(prices: pd.Series | pd.DataFrame, method: Method = "log") -> pd.Series | pd.DataFrame:
    """价格序列 → 收益序列（首行因无前值而产生 NaN，被丢弃）。

    Parameters
    ----------
    prices
        价格/净值序列或面板。必须为正数。
    method
        ``"log"`` 或 ``"simple"``。

    Notes
    -----
    非正价格会直接报错而不是替换成 0：价格 ≤ 0 通常意味着数据源有问题，
    静默处理会把数据错误伪装成正常结果。
    """
    if method not in ("log", "simple"):
        raise ValueError(f"method 只能是 'log' 或 'simple'，收到 {method!r}")
    _validate_ascending(prices, "prices")
    if (prices <= 0).to_numpy().any():
        bad = prices.index[(prices <= 0).to_numpy().any(axis=1)] if isinstance(prices, pd.DataFrame) else prices.index[prices <= 0]
        raise ValueError(f"价格必须为正，发现非正值，例如 {list(bad[:5])}")

    if method == "log":
        out = np.log(prices / prices.shift(1))
    else:
        out = prices.pct_change()
    return out.iloc[1:]


def to_simple(returns: pd.Series | pd.DataFrame, method: Method = "log") -> pd.Series | pd.DataFrame:
    """把收益序列统一转成简单收益。"""
    if method == "log":
        return np.expm1(returns)
    if method == "simple":
        return returns
    raise ValueError(f"method 只能是 'log' 或 'simple'，收到 {method!r}")


def portfolio_returns(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    method: Method = "log",
) -> pd.Series:
    """加权合成组合收益。

    这里采用的是**每日再平衡（daily rebalanced）**假设：权重每天回到目标值。
    这是一个必须写在页面上的简化，真实的买入持有组合权重会随时间漂移。
    缺失值不填充：任一标的当日缺失，该日组合收益为 NaN。
    """
    if isinstance(returns, pd.Series):
        result = returns.copy()
        if weights is not None:
            if len(weights) != 1:
                raise ValueError("单列收益只接受一个权重")
            result = result * float(next(iter(weights.values())))
        result.name = "portfolio"
        return result

    if weights is None:
        raise ValueError("多列收益必须提供 weights")
    missing = set(weights) - set(returns.columns)
    if missing:
        raise KeyError(f"weights 中存在收益面板里没有的标的：{sorted(missing)}")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"权重之和必须为 1，实际为 {total!r}")

    simple = to_simple(returns[list(weights)], method=method)
    w = pd.Series({k: float(v) for k, v in weights.items()})
    out = simple.mul(w, axis=1).sum(axis=1, min_count=len(w))
    # 出口口径必须与入口一致：log 进 log 出，simple 进 simple 出。
    # 早期版本这里漏了回转，导致 nav_curve 对已经变成简单收益的序列又做了一次 expm1。
    if method == "log":
        out = np.log1p(out)
    out.name = "portfolio"
    return out


def nav_curve(
    returns: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    method: Method = "log",
    base: float = 1.0,
) -> pd.Series:
    """收益序列 → 净值曲线。

    **口径警告**：返回序列的**首点已经包含第一个收益**（与 pyfolio 的
    ``cum_returns`` 一致），因此 ``nav.iloc[0] == base * (1 + r_1)``，
    此时的 ``base`` 对应的是"第一个收益发生之前"的净值。
    若要一条首点等于 ``base`` 的曲线（报告里更直观），请用 :func:`nav_from_prices`。
    """
    if returns.empty:
        raise ValueError("收益序列为空，无法构造净值曲线")
    port = portfolio_returns(returns, weights=weights, method=method).dropna()
    if port.empty:
        raise ValueError("合成后的收益序列全为缺失，请检查数据对齐")
    simple = to_simple(port, method=method)
    nav = base * (1.0 + simple).cumprod()
    nav.name = "nav"
    return nav


def nav_from_prices(
    prices: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    method: Method = "simple",
    base: float = 1.0,
) -> pd.Series:
    """价格面板 → 净值曲线，**首点等于 ``base``**。

    这是报告层应当使用的入口：因为价格比收益多一个观测点，
    首点可以直接作为起点，于是 ``metrics.total_return(nav)``
    与"区间累计收益"完全一致，不会漏掉第一期的涨跌。
    """
    _validate_ascending(prices, "prices")
    if len(prices) < 2:
        raise ValueError("价格序列至少需要两个点")
    rets = to_returns(prices, method=method)
    if isinstance(rets, pd.Series):
        simple = to_simple(rets, method=method)
        if weights:
            if len(weights) != 1:
                raise ValueError("单列价格只接受一个权重")
            simple = simple * float(next(iter(weights.values())))
    else:
        simple = to_simple(portfolio_returns(rets, weights=weights, method=method), method=method)
    growth = (1.0 + simple).cumprod()
    curve = pd.concat([pd.Series([base], index=prices.index[:1], dtype=float), base * growth])
    curve.name = "nav"
    return curve


def apply_annual_fee(
    returns: pd.Series | pd.DataFrame,
    annual_fee: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    method: Method = "log",
) -> pd.Series | pd.DataFrame:
    """按年费率逐期扣减收益（ETF 的管理费+托管费）。

    教学要点：年费率看似很小，但在十年以上的复利里是确定性的负收益，
    而标的收益是不确定的——所以它常常是"结论翻转项"。
    """
    if annual_fee < 0:
        raise ValueError("annual_fee 不能为负")
    if annual_fee == 0:
        return returns.copy()
    daily_simple = (1.0 - annual_fee) ** (1.0 / periods_per_year) - 1.0
    simple = to_simple(returns, method=method)
    adjusted = (1.0 + simple) * (1.0 + daily_simple) - 1.0
    if method == "log":
        return np.log1p(adjusted)
    return adjusted


def total_return(nav: pd.Series) -> float:
    """区间累计收益（净值末值 / 首值 − 1）。"""
    if len(nav) < 2:
        raise ValueError("净值序列至少需要两个点")
    first, last = float(nav.iloc[0]), float(nav.iloc[-1])
    if first <= 0:
        raise ValueError("净值首值必须为正")
    return last / first - 1.0


def years_elapsed(index: pd.DatetimeIndex) -> float:
    """按自然日折算年数——用于年化，而不是简单按行数除。"""
    if len(index) < 2:
        raise ValueError("至少需要两个日期")
    return (index[-1] - index[0]).days / 365.25
