"""利率环境与久期估计的对照测试（不访问网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.core import rates


def _curve(values: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    """构造长表曲线：``{code: [每日收益率]}``。"""
    rows = []
    for code, series in values.items():
        dates = pd.date_range(start, periods=len(series), freq="B")
        for date, value in zip(dates, series):
            rows.append({"date": date, "code": code, "tenor": code, "yield": value})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 利率环境
# --------------------------------------------------------------------------- #
def test_risk_free_is_returned_as_a_decimal_not_percent() -> None:
    """数据源以百分数存储，但接口必须返回小数。

    这个单位错误会把无风险利率放大 100 倍（rf=1.22，相当于每年 122%），
    进而让所有夏普/索提诺数字失效。它当初被数值警告"实数被复数截断"暴露出来——
    因为 rf>100% 再叠加负收益率就会出现 (1+rf)<0 的分数次幂。
    """
    curve = _curve({"CN1Y": [1.20, 1.22]})
    env = rates.rate_environment(curve)
    assert env.risk_free == pytest.approx(0.0122)


def test_rate_environment_reads_risk_free_from_the_curve() -> None:
    n = 300
    curve = _curve(
        {
            "CN1Y": list(np.linspace(1.8, 1.3, n)),
            "CN5Y": list(np.linspace(2.4, 1.6, n)),
            "CN10Y": list(np.linspace(2.9, 1.85, n)),
        }
    )
    env = rates.rate_environment(curve)

    assert env.source == "curve"
    assert env.tenor_used == "CN1Y"
    # 数据是百分数 1.3，返回应当是小数 0.013
    assert env.risk_free == pytest.approx(0.013, abs=1e-9)
    assert env.slope_10y_1y == pytest.approx((1.85 - 1.3) / 100.0, abs=1e-9)
    # 一年前的曲线也要取到，用于展示变化
    assert env.curve_last_year["CN10Y"] > env.curve["CN10Y"]
    assert env.curve["CN10Y"] < 0.05, "曲线值应当是小数而非百分数"


def test_rate_environment_falls_back_and_says_so() -> None:
    env = rates.rate_environment(pd.DataFrame(), fallback=0.025)
    assert env.source == "assumption"
    assert env.risk_free == pytest.approx(0.025)
    assert env.curve == {}
    assert env.slope_10y_1y is None


def test_rate_environment_looks_back_when_latest_day_is_missing() -> None:
    """最新日期缺 1 年期数据时，应回看若干天取最后一个有效值，而不是退回假设值。"""
    n = 60
    values = np.linspace(1.6, 1.4, n)
    curve = _curve({"CN1Y": list(values)})
    curve.loc[curve.index[-5:], "yield"] = np.nan  # 最近 5 天缺失

    env = rates.rate_environment(curve, lookback_days=15)

    assert env.source == "curve", "不该因为最新一天缺数据就退回假设值"
    # 最后一个有效值是倒数第 6 天那个（不是 1.4——1.4 属于被抹掉的最后一天）；注意百分数转小数
    assert env.risk_free == pytest.approx(values[-6] / 100.0, abs=1e-9)


def test_yield_changes_are_in_basis_points() -> None:
    curve = _curve({"CN5Y": [2.00, 2.05, 2.03]})
    changes = rates.yield_changes_bp(curve, "CN5Y")
    assert len(changes) == 2
    assert changes.iloc[0] == pytest.approx(5.0)   # +0.05% = +5bp
    assert changes.iloc[1] == pytest.approx(-2.0)  # −0.02% = −2bp


def test_yield_changes_empty_for_unknown_tenor() -> None:
    assert rates.yield_changes_bp(_curve({"CN5Y": [2.0, 2.1]}), "CN30Y").empty


# --------------------------------------------------------------------------- #
# 久期估计
# --------------------------------------------------------------------------- #
def _bond_returns_from_yields(changes_bp: np.ndarray, duration: float, noise: float = 0.0, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = -duration * changes_bp / 10000.0 + rng.normal(0, noise, len(changes_bp))
    return pd.Series(returns, name="r")


def _rw_yields(n: int, sd: float = 0.03, seed: int = 1, start: float = 2.0) -> np.ndarray:
    """收益率水平的随机游走（``sd`` 以百分点计，0.03 ≈ 每日 3bp）。"""
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.normal(0, sd, n))


def _driven_returns(curve: pd.DataFrame, code: str, duration: float, noise: float = 0.0, seed: int = 3) -> pd.Series:
    """构造"由某条曲线驱动"的收益序列，**索引与收益率变动严格同日对齐**。

    这里必须直接用收益率变动对齐日期，而不是先造价格再 ``pct_change()``：
    后者会再错开一天（差分一次已经去掉首日，pct_change 又去掉一次），
    结果是把 r_t 与 Δy_{t−1} 做回归——相关性直接塌成噪声。
    我最初就是这么写的，导致"久期 5 年"被估成 −0.78。
    """
    levels = curve[curve["code"] == code].set_index("date")["yield"].sort_index()
    changes = (levels.diff().dropna() * 100.0).to_numpy()
    values = -duration * changes / 10000.0
    if noise:
        rng = np.random.default_rng(seed)
        values = values + rng.normal(0, noise, len(changes))
    return pd.Series(values, index=levels.index[1:], name="r")


def test_estimate_duration_recovers_a_known_duration() -> None:
    """构造"久期 5 年"的收益序列，回归应当把 5 年精确找回来（无噪声时 R²=1）。"""
    curve = _curve({"CN5Y": list(_rw_yields(400))})
    returns = _driven_returns(curve, "CN5Y", duration=5.0)

    estimate = rates.estimate_duration("511010", returns, curve)
    assert estimate is not None
    assert estimate.code == "CN5Y"
    assert estimate.duration == pytest.approx(5.0, rel=0.02)
    assert estimate.r_squared > 0.99
    assert estimate.reliable


def test_estimate_duration_with_noise_still_close() -> None:
    curve = _curve({"CN5Y": list(_rw_yields(600))})
    returns = _driven_returns(curve, "CN5Y", duration=4.2, noise=2e-4)
    estimate = rates.estimate_duration("511010", returns, curve)
    assert estimate is not None
    assert estimate.duration == pytest.approx(4.2, rel=0.25)
    assert 0.5 < estimate.r_squared < 1.0


def test_estimate_duration_picks_the_tenor_with_the_best_fit() -> None:
    """价格由 5 年期驱动、10 年期只是独立噪声 —— 应当选 5 年，并如实报告各候选 R²。"""
    n = 500
    curve = _curve({"CN5Y": list(_rw_yields(n, sd=0.03, seed=11)), "CN10Y": list(_rw_yields(n, sd=0.05, seed=99, start=2.6))})
    returns = _driven_returns(curve, "CN5Y", duration=4.0)

    estimate = rates.estimate_duration("511010", returns, curve)
    assert estimate is not None
    assert estimate.code == "CN5Y"
    assert estimate.candidates["CN5Y"] > estimate.candidates["CN10Y"]
    assert estimate.candidates["CN5Y"] > 0.99


def test_estimate_duration_flags_low_r_squared() -> None:
    """价格与利率无关时，久期数字没有意义——必须给出警告而不是照样输出。"""
    n = 300
    rng = np.random.default_rng(5)
    curve = _curve({"CN5Y": list(2.0 + np.cumsum(rng.normal(0, 0.01, n)))})
    price = pd.Series(
        (1 + rng.normal(0, 0.002, n)).cumprod(),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    estimate = rates.estimate_duration("999999", price.pct_change().dropna(), curve)
    assert estimate is not None
    assert estimate.r_squared < 0.2
    assert not estimate.reliable
    assert estimate.warning is not None and "R²" in estimate.warning


def test_estimate_duration_returns_none_for_short_sample() -> None:
    curve = _curve({"CN5Y": list(2.0 + np.linspace(0, 0.1, 30))})
    price = pd.Series(np.linspace(100, 101, 30), index=pd.date_range("2020-01-01", periods=30, freq="B"))
    assert rates.estimate_duration("511010", price.pct_change().dropna(), curve, min_obs=120) is None


def test_duration_units_are_documented_by_a_rule_of_thumb() -> None:
    """久期 5 年、收益率上行 100bp → 价格约跌 5%（教科书的拇指法则）。"""
    curve = _curve({"CN5Y": list(_rw_yields(500, sd=0.03, seed=7))})
    returns = _driven_returns(curve, "CN5Y", duration=5.0)

    estimate = rates.estimate_duration("511010", returns, curve)
    assert estimate is not None
    assert estimate.duration == pytest.approx(5.0, rel=0.02)
    # 用估计出的久期做冲击：100bp × 5 年 = 约 5%
    rows = rates.rate_scenarios({"511010": estimate.duration}, {"511010": 1.0}, shocks_bp=(100,))
    assert rows[0]["bond_sleeve_impact"] == pytest.approx(-0.05, rel=0.05)


# --------------------------------------------------------------------------- #
# 利率冲击情景
# --------------------------------------------------------------------------- #
def test_rate_scenarios_hand_computed() -> None:
    """30% 仓位、久期 5 年，收益率上行 100bp → 组合影响 = 0.3 × (−5) × 1% = −1.5%。"""
    rows = rates.rate_scenarios({"511010": 5.0}, {"511010": 0.3, "510300": 0.7}, shocks_bp=(100,))
    row = rows[0]
    assert row["portfolio_impact"] == pytest.approx(-0.015, rel=1e-9)
    assert row["bond_weight"] == pytest.approx(0.3)
    assert row["bond_sleeve_impact"] == pytest.approx(-0.05, rel=1e-9)


def test_rate_scenarios_signs_and_ordering() -> None:
    rows = rates.rate_scenarios({"A": 4.0}, {"A": 0.5}, shocks_bp=(100, 50, -50, -100))
    by_shock = {row["shock_bp"]: row for row in rows}
    assert by_shock[100]["portfolio_impact"] < by_shock[50]["portfolio_impact"] < 0
    assert by_shock[-100]["portfolio_impact"] > by_shock[-50]["portfolio_impact"] > 0


def test_rate_scenarios_ignores_holdings_without_duration() -> None:
    rows = rates.rate_scenarios({"A": 4.0}, {"A": 0.25, "STOCK": 0.75}, shocks_bp=(100,))
    assert rows[0]["bond_weight"] == pytest.approx(0.25)
    assert set(rows[0]["per_holding"]) == {"A"}


def test_rate_scenarios_handles_zero_bond_weight() -> None:
    rows = rates.rate_scenarios({}, {}, shocks_bp=(100,))
    assert rows[0]["portfolio_impact"] == 0.0
    assert rows[0]["bond_sleeve_impact"] is None
