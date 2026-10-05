"""期权定价与 Greeks 的对照测试（不访问网络）。

这套测试的重点是**双路校验**：每个解析解都与有限差分比对。
Greeks 最容易出的错是单位混用（vega 按 1.0 还是 1 个百分点、theta 按年还是按天），
这种错不会抛异常，只会让结论反过来——所以必须两路对上才算对。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.core import derivatives

# 教科书算例：S=K=100, T=1, r=5%, sigma=20%
S0, K0, T0, R0, SIGMA0 = 100.0, 100.0, 1.0, 0.05, 0.20


def test_bs_price_matches_textbook_values() -> None:
    call = derivatives.bs_price(S0, K0, T0, R0, SIGMA0, "call")
    put = derivatives.bs_price(S0, K0, T0, R0, SIGMA0, "put")
    assert call == pytest.approx(10.4506, abs=0.01)
    assert put == pytest.approx(5.5735, abs=0.01)


def test_put_call_parity_holds_exactly() -> None:
    """看涨-看跌平价关系是精确恒等式，任何实现错误都会在这里暴露。"""
    for strike in (80.0, 100.0, 120.0):
        call = derivatives.bs_price(S0, strike, T0, R0, SIGMA0, "call")
        put = derivatives.bs_price(S0, strike, T0, R0, SIGMA0, "put")
        assert call - put == pytest.approx(S0 - strike * np.exp(-R0 * T0), abs=1e-10)


@pytest.mark.parametrize(
    "case",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, "call"),
        (100.0, 90.0, 0.5, 0.03, 0.35, "put"),
        (4.43, 4.20, 0.25, 0.02, 0.18, "put"),
        (100.0, 110.0, 2.0, 0.04, 0.15, "call"),
    ],
)
def test_analytic_greeks_match_finite_difference(case: tuple) -> None:
    S, K, T, r, sigma, option_type = case
    analytic = derivatives.bs_greeks(S, K, T, r, sigma, option_type)
    numeric = derivatives.finite_difference_greeks(S, K, T, r, sigma, option_type)

    assert analytic.price == pytest.approx(numeric.price, rel=1e-9)
    assert analytic.delta == pytest.approx(numeric.delta, rel=1e-4, abs=1e-6)
    assert analytic.gamma == pytest.approx(numeric.gamma, rel=1e-3, abs=1e-6)
    assert analytic.vega == pytest.approx(numeric.vega, rel=1e-4, abs=1e-8)
    assert analytic.theta == pytest.approx(numeric.theta, rel=1e-3, abs=1e-8)
    assert analytic.rho == pytest.approx(numeric.rho, rel=1e-4, abs=1e-8)


def test_vega_and_theta_units_are_as_documented() -> None:
    """单位约定要能被验证：vega 按 1 个百分点、theta 按自然日。

    容差说明：这里用**有限步长**（σ 从 20% 跳到 21%、时间少一天）去比解析导数，
    因此会带一点二阶残差（vega 的残差来自 volga ≈ 1.2e-3 相对量级）。
    解析解与差分的**精确**一致性由 ``test_analytic_greeks_match_finite_difference``
    用极小步长（rel_step=1e-4）来保证；这一条只验证"数量级与方向对得上"。
    若单位写错（例如 vega 忘了除 100），这里会差 100 倍，一眼就能看出来。
    """
    greeks = derivatives.bs_greeks(S0, K0, T0, R0, SIGMA0, "call")
    up = derivatives.bs_price(S0, K0, T0, R0, SIGMA0 + 0.01, "call")
    assert up - greeks.price == pytest.approx(greeks.vega, rel=5e-3)
    less = derivatives.bs_price(S0, K0, T0 - 1 / 365, R0, SIGMA0, "call")
    assert less - greeks.price == pytest.approx(greeks.theta, rel=1e-2)


def test_gamma_is_largest_at_the_money() -> None:
    """Gamma 在平值附近最大——这是"平值期权最难对冲"的量化来源。"""
    atm = derivatives.bs_greeks(S0, 100.0, 0.25, R0, SIGMA0).gamma
    otm = derivatives.bs_greeks(S0, 130.0, 0.25, R0, SIGMA0).gamma
    assert atm > otm
    assert atm > 0 and otm > 0


def test_vectorised_delta_gamma_matches_scalar_implementation() -> None:
    """向量化 delta/gamma 必须与标量 ``bs_greeks`` 逐点一致。

    复制模拟为了性能改用向量化 Greeks（标量版跑 4000 条路径要十几分钟），
    而"向量化实现与标量实现分叉"正是这类优化最典型的隐患，所以必须钉住。
    """
    spots = np.array([2.0, 3.5, 4.43, 5.0, 8.0])
    for option_type in ("call", "put"):
        delta, gamma = derivatives.bs_delta_gamma_array(spots, 4.5, 0.75, 0.02, 0.18, option_type)
        for index, spot in enumerate(spots):
            scalar = derivatives.bs_greeks(float(spot), 4.5, 0.75, 0.02, 0.18, option_type)
            assert delta[index] == pytest.approx(scalar.delta, rel=1e-12)
            assert gamma[index] == pytest.approx(scalar.gamma, rel=1e-12)


def test_vectorised_delta_gamma_degenerate_cases() -> None:
    """到期或零波动时，向量化实现也要给出阶跃 delta 与零 gamma（不能出 NaN）。"""
    spots = np.array([3.0, 4.5, 6.0])
    delta, gamma = derivatives.bs_delta_gamma_array(spots, 4.5, 0.0, 0.02, 0.18, "call")
    assert np.allclose(delta, [0.0, 0.0, 1.0])
    assert np.allclose(gamma, 0.0)
    put_delta, _ = derivatives.bs_delta_gamma_array(spots, 4.5, 0.0, 0.02, 0.18, "put")
    assert np.allclose(put_delta, [-1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        derivatives.bs_delta_gamma_array(spots, 4.5, 1.0, 0.02, 0.18, "straddle")


def test_vectorised_price_matches_scalar_implementation() -> None:
    """向量化定价必须与标量 ``bs_price`` 逐点一致（含退化情形）。"""
    spots = np.array([2.0, 3.5, 4.43, 5.0, 8.0])
    for option_type in ("call", "put"):
        array = derivatives.bs_price_array(spots, 4.5, 0.75, 0.02, 0.18, option_type)
        for index, spot in enumerate(spots):
            scalar = derivatives.bs_price(float(spot), 4.5, 0.75, 0.02, 0.18, option_type)
            assert array[index] == pytest.approx(scalar, rel=1e-12)
    # 到期与零波动：只剩远期内在价值 max(S − K, 0)
    expired = derivatives.bs_price_array(spots, 4.5, 0.0, 0.02, 0.18, "call")
    assert np.allclose(expired, [0.0, 0.0, 0.0, 0.5, 3.5])
    zero_vol = derivatives.bs_price_array(spots, 4.5, 1.0, 0.0, 0.0, "call")
    assert np.allclose(zero_vol, [0.0, 0.0, 0.0, 0.5, 3.5])


def test_long_option_has_bounded_delta_and_negative_theta() -> None:
    call = derivatives.bs_greeks(S0, K0, T0, R0, SIGMA0, "call")
    put = derivatives.bs_greeks(S0, K0, T0, R0, SIGMA0, "put")
    assert 0.0 < call.delta < 1.0
    assert -1.0 < put.delta < 0.0
    assert call.theta < 0 and put.theta < 0
    # 看涨与看跌的 delta 之差恒为 1（无分红时）
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-12)


def test_expired_or_zero_vol_falls_back_to_intrinsic() -> None:
    assert derivatives.bs_price(120.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(20.0)
    assert derivatives.bs_price(80.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)
    assert derivatives.bs_price(80.0, 100.0, 0.0, 0.05, 0.2, "put") == pytest.approx(20.0)
    # 波动率为 0 时，只剩行权价现值的远期内在价值
    expected = max(100.0 - 100.0 * np.exp(-0.05), 0.0)
    assert derivatives.bs_price(100.0, 100.0, 1.0, 0.05, 0.0, "call") == pytest.approx(expected)


def test_implied_vol_round_trip() -> None:
    for sigma_true in (0.08, 0.15, 0.30, 0.60):
        for option_type in ("call", "put"):
            price = derivatives.bs_price(S0, 105.0, 0.5, R0, sigma_true, option_type)
            recovered = derivatives.implied_vol(price, S0, 105.0, 0.5, R0, option_type)
            assert recovered == pytest.approx(sigma_true, rel=1e-8)


def test_implied_vol_returns_nan_when_unattainable() -> None:
    """价格低于内在价值时无解——返回 nan 而不是抛异常（"无解"本身是结论）。"""
    intrinsic = S0 - 105.0 * np.exp(-R0 * 0.5)
    assert np.isnan(derivatives.implied_vol(intrinsic - 5.0, S0, 105.0, 0.5, R0, "call"))


def test_historical_volatility_on_known_series() -> None:
    rng = np.random.default_rng(7)
    daily = 0.01
    returns = pd.Series(rng.normal(0, daily, 5000))
    # 目标年化 0.01×√252 ≈ 15.87%，样本量足够时应当接近
    assert derivatives.historical_volatility(returns, window=252) == pytest.approx(daily * np.sqrt(252), rel=0.15)


def test_volatility_term_structure_reports_all_windows() -> None:
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0, 0.012, 1200))
    term = derivatives.volatility_term_structure(returns, windows=(21, 63, 252, 504))
    assert set(term) == {21, 63, 252, 504}
    assert all(np.isfinite(v) for v in term.values())


def test_historical_volatility_nan_when_sample_too_short() -> None:
    returns = pd.Series([0.01] * 10)
    assert np.isnan(derivatives.historical_volatility(returns, window=252))


def test_protection_table_behaves_economically() -> None:
    """经济直觉必须成立：越深度价外越便宜、期限越长越贵、平值 Delta 绝对值最大。"""
    table = derivatives.protection_table(
        spot=1.0,
        vol_by_tenor={1 / 12: 0.20, 3 / 12: 0.20, 6 / 12: 0.20, 1.0: 0.20},
        moneyness=(0.85, 0.90, 0.95, 1.00),
        tenors=(1 / 12, 1.0),
    )
    one_month = {row["moneyness"]: row for row in table if row["tenor_years"] == pytest.approx(1 / 12)}
    one_year = {row["moneyness"]: row for row in table if row["tenor_years"] == pytest.approx(1.0)}

    # 行权价越低（越价外）越便宜
    assert one_month[0.85]["cost_pct"] < one_month[0.90]["cost_pct"] < one_month[0.95]["cost_pct"] < one_month[1.00]["cost_pct"]
    # 同样行权价，一年期比一个月期贵
    assert one_year[0.95]["cost_pct"] > one_month[0.95]["cost_pct"]
    # 平值 put 的 Delta 绝对值最大（约 -0.5）
    assert abs(one_month[1.00]["delta"]) > abs(one_month[0.85]["delta"])
    assert -1.0 < one_month[1.00]["delta"] < 0.0
    # 年化成本必须被算出来，否则"每年花多少"这个最有用的数字就没了
    assert one_year[0.95]["annualized_cost_pct"] == pytest.approx(one_year[0.95]["cost_pct"], rel=1e-9)


def test_protection_table_falls_back_when_vol_missing() -> None:
    table = derivatives.protection_table(spot=1.0, vol_by_tenor={}, fallback_vol=0.25)
    assert all(row["sigma"] == pytest.approx(0.25) for row in table)
