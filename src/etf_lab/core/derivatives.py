"""期权定价与 Greeks：Black-Scholes 解析解、有限差分交叉校验、隐含波动率反解。

为什么这个模块需要"双路校验"
----------------------------
Greeks 是最容易写出"看起来对但差一点"的东西：单位混用（vega 是按 1.0 波动率还是 1 个
百分点？theta 是按年还是按天？）不会报错，只会让读者得出相反的结论。
所以本模块的每一个解析解都配一个**有限差分**实现，测试逐项比对——
解析公式与数值差分必须一致，单位约定写在函数签名与 docstring 里。

数据可得性的诚实说明
--------------------
**隐含波动率需要期权行情，本项目尚未接入。** 因此页面上展示的是用**历史波动率**
构造的期限结构，以及在这些波动率假设下的**理论** Greeks。
它是"如果按过去一年的波动定价，保险要花多少钱"，不是市场报价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.0
OptionType = str  # "call" / "put"


@dataclass(frozen=True)
class Greeks:
    """一个期权头寸的价格与 Greeks。

    单位约定（**混用单位是这套东西最常见的错误来源**）：

    - ``delta``：标的价格变动 1 个单位时的价格变动
    - ``gamma``：标的价格变动 1 个单位时 delta 的变动
    - ``vega``：波动率变动 **1 个百分点**时的价格变动
    - ``theta``：时间过去 **1 个自然日**时的价格变动（买入期权通常为负）
    - ``rho``：利率变动 **1 个百分点**时的价格变动
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def _check_inputs(S: float, K: float, T: float, sigma: float, option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type 只能是 'call' 或 'put'，收到 {option_type!r}")
    if S <= 0 or K <= 0:
        raise ValueError("标的价格与行权价必须为正")
    if T < 0:
        raise ValueError("到期时间不能为负")
    if sigma < 0:
        raise ValueError("波动率不能为负")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return float(d1), float(d1 - vol_sqrt_t)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call", q: float = 0.0) -> float:
    """Black-Scholes 欧式期权价格（连续分红率 ``q``）。"""
    _check_inputs(S, K, T, sigma, option_type)
    if T == 0 or sigma == 0:
        # 退化情形：只剩内在价值（按远期口径折算行权价的现值）
        forward_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        intrinsic = max(forward_intrinsic, 0.0) if option_type == "call" else max(-forward_intrinsic, 0.0)
        return float(intrinsic)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return float(S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
    return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> Greeks:
    """解析解价格与 Greeks（单位约定见 :class:`Greeks`）。"""
    _check_inputs(S, K, T, sigma, option_type)
    price = bs_price(S, K, T, r, sigma, option_type, q)
    if T == 0 or sigma == 0:
        return Greeks(price=price, delta=float("nan"), gamma=float("nan"), vega=float("nan"), theta=float("nan"), rho=float("nan"))

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    phi = stats.norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        delta = disc_q * stats.norm.cdf(d1)
        theta_annual = -S * disc_q * phi * sigma / (2 * np.sqrt(T)) - r * K * disc_r * stats.norm.cdf(d2) + q * S * disc_q * stats.norm.cdf(d1)
        rho_annual = K * T * disc_r * stats.norm.cdf(d2)
    else:
        delta = disc_q * (stats.norm.cdf(d1) - 1.0)
        theta_annual = -S * disc_q * phi * sigma / (2 * np.sqrt(T)) + r * K * disc_r * stats.norm.cdf(-d2) - q * S * disc_q * stats.norm.cdf(-d1)
        rho_annual = -K * T * disc_r * stats.norm.cdf(-d2)

    gamma = disc_q * phi / (S * sigma * np.sqrt(T))
    vega_per_unit = S * disc_q * phi * np.sqrt(T)
    return Greeks(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega_per_unit * 0.01),
        theta=float(theta_annual / CALENDAR_DAYS_PER_YEAR),
        rho=float(rho_annual * 0.01),
    )


def finite_difference_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    rel_step: float = 1e-4,
) -> Greeks:
    """用中心差分算同一组 Greeks，用于与解析解交叉校验。

    单位约定与 :func:`bs_greeks` 保持一致（vega 按 1 个百分点、theta 按自然日、rho 按 1 个百分点），
    否则"两路校验"会因为单位不同而永远对不上。
    """

    def price(**kwargs: float) -> float:
        args = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}
        args.update(kwargs)
        return bs_price(args["S"], args["K"], args["T"], args["r"], args["sigma"], option_type, args["q"])

    h_s = S * rel_step
    h_sigma = max(sigma * rel_step, 1e-6)
    h_t = max(T * rel_step, 1e-6)
    h_r = max(abs(r) * rel_step, 1e-6)

    delta = (price(S=S + h_s) - price(S=S - h_s)) / (2 * h_s)
    gamma = (price(S=S + h_s) - 2 * price() + price(S=S - h_s)) / (h_s**2)
    vega = (price(sigma=sigma + h_sigma) - price(sigma=sigma - h_sigma)) / (2 * h_sigma) * 0.01
    # 时间前进 → T 变小，所以对 T 的差分取负
    theta = -(price(T=T + h_t) - price(T=T - h_t)) / (2 * h_t) / CALENDAR_DAYS_PER_YEAR
    rho = (price(r=r + h_r) - price(r=r - h_r)) / (2 * h_r) * 0.01

    return Greeks(price=price(), delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    lower: float = 1e-4,
    upper: float = 5.0,
) -> float:
    """用 Brent 法反解隐含波动率；无解时返回 ``nan``（不抛异常，"无解"本身是结论）。"""
    _check_inputs(S, K, T, 1.0, option_type)
    if T <= 0:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type, q) - target_price

    low, high = objective(lower), objective(upper)
    if not (low <= 0 <= high or high <= 0 <= low):
        return float("nan")
    try:
        return float(optimize.brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def historical_volatility(returns: pd.Series, window: int = 252, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """最近 ``window`` 个交易日的年化历史波动率。"""
    clean = returns.dropna()
    if len(clean) < max(20, window // 5):
        return float("nan")
    tail = clean.iloc[-window:]
    return float(tail.std(ddof=1) * np.sqrt(periods_per_year))


def volatility_term_structure(
    returns: pd.Series,
    windows: Sequence[int] = (21, 63, 126, 252, 504),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[int, float]:
    """历史波动率的"期限结构"：不同回看窗口下的年化波动率。

    **它不等同于隐含波动率的期限结构**：前者是过去已实现的波动，
    后者是市场对未来波动的定价。市场恐慌时两者会显著背离，
    这正是期权"在最需要保险时最贵"的来源。本项目暂无期权行情，故只能给前者。
    """
    return {int(w): historical_volatility(returns, int(w), periods_per_year) for w in windows}


def protection_table(
    spot: float,
    vol_by_tenor: Mapping[float, float],
    moneyness: Sequence[float] = (0.85, 0.90, 0.95, 1.00, 1.05),
    tenors: Sequence[float] = (1 / 12, 3 / 12, 6 / 12, 1.0),
    r: float = 0.02,
    q: float = 0.0,
    fallback_vol: float = 0.20,
) -> list[dict[str, float]]:
    """买入认沽期权的成本表——把"给持仓买保险"变成可读的数字。

    ``vol_by_tenor`` 用期限（年）索引波动率；缺失时用 ``fallback_vol``。
    ``moneyness`` 为行权价 / 现价：0.90 表示"跌到九成才赔付"的深度价外保护。
    """
    rows: list[dict[str, float]] = []
    for tenor in tenors:
        vol = float(vol_by_tenor.get(tenor, fallback_vol) or fallback_vol)
        if not np.isfinite(vol) or vol <= 0:
            vol = fallback_vol
        for ratio in moneyness:
            strike = spot * float(ratio)
            greeks = bs_greeks(spot, strike, tenor, r, vol, "put", q)
            rows.append(
                {
                    "tenor_years": float(tenor),
                    "tenor_months": round(float(tenor) * 12, 1),
                    "moneyness": float(ratio),
                    "strike": float(strike),
                    "sigma": vol,
                    "cost_pct": float(greeks.price / spot),
                    "annualized_cost_pct": float(greeks.price / spot / max(tenor, 1e-9)),
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "vega": greeks.vega,
                    "theta": greeks.theta,
                }
            )
    return rows
