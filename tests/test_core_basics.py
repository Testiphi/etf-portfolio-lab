"""`core/` 的数值对照测试。

原则：**每个测试都必须检验一个独立可推导的结论**，而不是把实现再抄一遍。
凡是"看起来对但差一点"的实现（XIRR、回撤区间、成分 VaR），都要有手算基准
或数学性质校验（例如欧拉分解的加总性质、NPV 归零）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etf_lab.core import correlation, dca, metrics, returns


# --------------------------------------------------------------------------- #
# 收益与净值
# --------------------------------------------------------------------------- #
def _daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def test_to_returns_simple_and_log_are_consistent() -> None:
    prices = pd.Series([100.0, 110.0, 99.0], index=_daily_index(3))
    simple = returns.to_returns(prices, method="simple")
    log = returns.to_returns(prices, method="log")

    assert simple.iloc[0] == pytest.approx(0.10)
    assert simple.iloc[1] == pytest.approx(-0.10)
    # 两种口径必须能互相转换
    np.testing.assert_allclose(np.expm1(log.to_numpy()), simple.to_numpy(), atol=1e-12)


def test_to_returns_rejects_non_positive_price() -> None:
    prices = pd.Series([100.0, 0.0, 50.0], index=_daily_index(3))
    with pytest.raises(ValueError):
        returns.to_returns(prices)


def test_to_returns_rejects_unsorted_index() -> None:
    prices = pd.Series([100.0, 110.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))
    with pytest.raises(ValueError):
        returns.to_returns(prices)


def test_nav_from_prices_recovers_total_return() -> None:
    prices = pd.DataFrame(
        {"a": [100.0, 110.0, 121.0], "b": [50.0, 55.0, 60.5]},
        index=_daily_index(3),
    )
    nav = returns.nav_from_prices(prices, weights={"a": 0.5, "b": 0.5})
    # 两个标的每日涨幅完全相同 → 组合净值应与单标的同步
    assert nav.iloc[0] == pytest.approx(1.0)
    assert nav.iloc[-1] == pytest.approx(1.21, rel=1e-12)
    assert len(nav) == len(prices)
    assert returns.total_return(nav) == pytest.approx(0.21, rel=1e-12)


def test_nav_curve_first_point_already_includes_first_return() -> None:
    """记录 ``nav_curve`` 的口径：首点是 base×(1+r₁)，不是 base。"""
    prices = pd.Series([100.0, 110.0, 121.0], index=_daily_index(3))
    rets = returns.to_returns(prices, method="simple")
    nav = returns.nav_curve(rets, method="simple")
    assert nav.iloc[0] == pytest.approx(1.10)
    assert nav.iloc[-1] == pytest.approx(1.21)


def test_portfolio_weights_must_sum_to_one() -> None:
    rets = pd.DataFrame({"a": [0.01], "b": [0.02]}, index=_daily_index(1))
    with pytest.raises(ValueError):
        returns.portfolio_returns(rets, weights={"a": 0.5, "b": 0.6}, method="simple")


def test_annual_fee_compounds_to_the_stated_rate() -> None:
    n = 252
    zero = pd.Series([0.0] * n, index=_daily_index(n))
    adjusted = returns.apply_annual_fee(zero, annual_fee=0.01, method="log")
    total = float(np.expm1(adjusted.sum()))
    assert total == pytest.approx(-0.01, abs=2e-4)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def test_annualized_return_matches_definition() -> None:
    nav = pd.Series([1.0, 1.21], index=pd.DatetimeIndex(["2018-01-01", "2020-01-01"]))
    days = (nav.index[-1] - nav.index[0]).days
    expected = 1.21 ** (365.25 / days) - 1.0
    assert metrics.annualized_return(nav) == pytest.approx(expected, rel=1e-12)


def test_max_drawdown_hand_computed() -> None:
    values = [100.0, 120.0, 90.0, 130.0, 60.0, 70.0]
    nav = pd.Series(values, index=_daily_index(6))
    info = metrics.max_drawdown(nav)

    assert info.depth == pytest.approx(60.0 / 130.0 - 1.0)  # -53.846%
    assert info.peak_date == nav.index[3]
    assert info.trough_date == nav.index[4]
    assert info.recovery_date is None  # 之后没有回到 130
    assert not info.recovered


def test_max_drawdown_recovery_detected() -> None:
    nav = pd.Series([100.0, 80.0, 130.0], index=_daily_index(3))
    info = metrics.max_drawdown(nav)
    assert info.depth == pytest.approx(-0.20)
    assert info.recovery_date == nav.index[2]
    assert info.recovered


def test_top_drawdowns_are_ordered_and_non_overlapping() -> None:
    values = [100.0, 60.0, 100.0, 50.0, 100.0]
    nav = pd.Series(values, index=_daily_index(5))
    segments = metrics.top_drawdowns(nav, top_n=2)
    assert len(segments) >= 1
    depths = [s.depth for s in segments]
    assert depths == sorted(depths)  # 第一段最深


def test_var_cvar_hand_computed() -> None:
    values = [-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    rets = pd.Series(values, index=_daily_index(len(values)))
    result = metrics.var_cvar(rets, level=0.95, method="historical")

    quantile = float(rets.quantile(0.05))
    assert result.var == pytest.approx(-quantile, rel=1e-12)
    # 尾部只有 -0.10 一个观测（<= 分位点）
    assert result.cvar == pytest.approx(0.10, rel=1e-12)
    assert result.n_obs == 10


def test_var_parametric_matches_normal_formula() -> None:
    rng = np.random.default_rng(20260101)
    rets = pd.Series(rng.normal(0.0005, 0.012, 4000), index=_daily_index(4000))
    result = metrics.var_cvar(rets, level=0.95, method="parametric")
    expected = -(rets.mean() + rets.std(ddof=1) * stats.norm.ppf(0.05))
    assert result.var == pytest.approx(expected, rel=1e-10)


def test_sharpe_zero_volatility_returns_nan() -> None:
    # 用精确的 0.0：像 0.001 这样的十进制小数在浮点下相减会留下残差，
    # 标准差不会恰好为 0，反而测不到"零波动"这条分支。
    rets = pd.Series([0.0] * 100, index=_daily_index(100))
    assert np.isnan(metrics.sharpe_ratio(rets))


def test_sharpe_equals_period_sharpe_times_sqrt_ppy() -> None:
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0004, 0.01, 1000), index=_daily_index(1000))
    period_sharpe = float(rets.mean() / rets.std(ddof=1))
    assert metrics.sharpe_ratio(rets, periods_per_year=252) == pytest.approx(period_sharpe * np.sqrt(252), rel=1e-12)


def test_sharpe_uses_supplied_risk_free_rate() -> None:
    rng = np.random.default_rng(17)
    rets = pd.Series(rng.normal(0.0004, 0.01, 800), index=_daily_index(800))
    with_rf = metrics.sharpe_ratio(rets, rf_annual=0.03)
    without_rf = metrics.sharpe_ratio(rets, rf_annual=0.0)
    # rf 取 3% 时应更低——这正是"页面必须显示所用 rf"的原因
    assert with_rf < without_rf


def test_summary_rejects_multi_asset_returns() -> None:
    """把多资产面板当组合收益传进来，是最容易犯的量纲错误，必须显式拒绝。"""
    nav = pd.Series([1.0, 1.1, 0.99, 1.2], index=_daily_index(4))
    panel = pd.DataFrame({"a": [0.01, -0.02, 0.03], "b": [0.02, 0.01, -0.01]}, index=_daily_index(3))
    with pytest.raises(TypeError, match="单列 Series"):
        metrics.summary(nav, panel)


def test_summary_reports_units_and_dates() -> None:
    nav = pd.Series([1.0, 1.1, 0.99, 1.2], index=_daily_index(4))
    out = metrics.summary(nav, rf_annual=0.02)
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "var_95_historical",
        "cvar_95_parametric",
        "rf_annual_used",
    ):
        assert key in out
    assert out["rf_annual_used"] == 0.02
    assert out["n_obs"] == 3


# --------------------------------------------------------------------------- #
# XIRR 与定投
# --------------------------------------------------------------------------- #
def test_xirr_known_ten_percent() -> None:
    flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2020-12-31"), 1100.0)]
    # 2020 是闰年，1/1 → 12/31 共 365 天
    assert (flows[1][0] - flows[0][0]).days == 365
    assert dca.xirr(flows) == pytest.approx(0.10, rel=1e-9)


def test_xirr_root_zeroes_npv() -> None:
    flows = [
        (pd.Timestamp("2020-01-01"), -1000.0),
        (pd.Timestamp("2020-04-01"), -1000.0),
        (pd.Timestamp("2021-02-15"), -500.0),
        (pd.Timestamp("2022-06-30"), 3200.0),
    ]
    rate = dca.xirr(flows)
    assert np.isfinite(rate)
    assert abs(dca.xnpv(rate, flows)) < 1e-6


def test_xirr_returns_nan_without_sign_change() -> None:
    flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2020-06-01"), -500.0)]
    assert np.isnan(dca.xirr(flows))


def test_contribution_dates_monthly_picks_first_trading_day() -> None:
    index = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-15", "2020-02-03", "2020-02-20", "2020-03-02"]
    )
    plan = dca.DcaPlan(amount=1000.0, freq="monthly")
    picked = dca.contribution_dates(index, plan)
    assert picked == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")]


def test_contribution_dates_monthly_honours_day_offset() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-15", "2020-02-03", "2020-02-20"])
    plan = dca.DcaPlan(amount=1000.0, freq="monthly", day=2)
    picked = dca.contribution_dates(index, plan)
    assert picked == [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-02-20")]


FREE = dca.CostModel(commission_rate=0.0, min_commission=0.0, slippage_bps=0.0, lot_size=100)


def test_dca_flat_price_has_zero_xirr() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2020-03-02"])
    prices = pd.Series([1.0, 1.0, 1.0], index=index)
    result = dca.simulate(dca.DcaPlan(amount=1000.0, freq="monthly"), prices, FREE)

    assert result.invested_total == pytest.approx(3000.0)
    assert result.final_value == pytest.approx(3000.0)
    assert result.shares.iloc[-1] == pytest.approx(3000.0)
    assert result.cash.iloc[-1] == pytest.approx(0.0)
    assert result.xirr == pytest.approx(0.0, abs=1e-9)


def test_dca_rising_price_hand_computed() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2020-03-02"])
    prices = pd.Series([1.0, 2.0, 3.0], index=index)
    result = dca.simulate(dca.DcaPlan(amount=1000.0, freq="monthly"), prices, FREE)

    # 手算：t1 买 1000 份；t2 买 500 份；t3 买 300 份（剩 100 元现金）
    assert result.shares.iloc[-1] == pytest.approx(1800.0)
    assert result.cash.iloc[-1] == pytest.approx(100.0)
    assert result.final_value == pytest.approx(5500.0)
    assert result.invested_total == pytest.approx(3000.0)


def test_dca_time_weighted_and_money_weighted_differ() -> None:
    """教学核心结论：同一段行情，"总收益÷总投入"与 XIRR 会给出不同答案。

    构造：价格在 23 个月里恒为 1.0，最后一个月跳到 2.0（一次性翻倍）。
    总收益÷总投入 = 47000/24000 − 1 ≈ 95.8%，而 XIRR 按每笔钱的实际在场时间折算，
    结果明显更低——这正是定投收益率最容易算错的地方。
    """
    index = pd.date_range("2020-01-02", periods=24, freq="MS")
    prices = pd.Series([1.0] * 23 + [2.0], index=index)
    result = dca.simulate(dca.DcaPlan(amount=1000.0, freq="monthly"), prices, FREE)

    simple = result.metrics["simple_return_on_invested"]
    assert result.invested_total == pytest.approx(24000.0)
    assert result.final_value == pytest.approx(47000.0)
    assert simple == pytest.approx(47000.0 / 24000.0 - 1.0, rel=1e-12)
    assert np.isfinite(result.xirr)
    assert abs(simple - result.xirr) > 0.02


def test_dca_lot_size_leaves_cash_when_amount_too_small() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-02-03"])
    prices = pd.Series([1.0, 1.0], index=index)
    result = dca.simulate(dca.DcaPlan(amount=50.0, freq="monthly"), prices, FREE)
    assert result.shares.iloc[-1] == pytest.approx(0.0)
    assert result.final_value == pytest.approx(100.0)  # 未买成，全部留现金


def test_dca_commission_reduces_shares() -> None:
    index = pd.DatetimeIndex(["2020-01-02"])
    prices = pd.Series([1.0], index=index)
    costly = dca.CostModel(commission_rate=0.01, min_commission=0.0, slippage_bps=0.0, lot_size=100)
    result = dca.simulate(dca.DcaPlan(amount=1000.0, freq="monthly"), prices, costly)
    # 1000 元本来可买 10 手，含 1% 佣金后只能买 9 手
    assert result.shares.iloc[-1] == pytest.approx(900.0)


def test_dca_rejects_missing_prices() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-02-03"])
    prices = pd.Series([1.0, np.nan], index=index)
    with pytest.raises(ValueError):
        dca.simulate(dca.DcaPlan(amount=1000.0), prices, FREE)


# --------------------------------------------------------------------------- #
# 相关性与边际影响
# --------------------------------------------------------------------------- #
def test_correlation_matrix_extremes() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(0, 0.01, 300)
    panel = pd.DataFrame(
        {"a": base, "same": base, "opposite": -base},
        index=_daily_index(300),
    )
    corr = correlation.correlation_matrix(panel, min_obs=10)
    assert corr.loc["a", "same"] == pytest.approx(1.0)
    assert corr.loc["a", "opposite"] == pytest.approx(-1.0)


def test_component_var_sums_to_portfolio_var() -> None:
    """欧拉分解的性质：各成分 VaR 之和应等于组合 VaR。"""
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        rng.normal(0, 0.01, size=(500, 3)),
        columns=["a", "b", "c"],
        index=_daily_index(500),
    )
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    comp = correlation.component_var(panel, weights, level=0.95)

    w = np.array([weights[c] for c in panel.columns])
    sigma = float(np.sqrt(w @ panel.cov().to_numpy() @ w))
    expected = stats.norm.ppf(0.95) * sigma
    assert comp.sum() == pytest.approx(expected, rel=1e-9)


def test_marginal_impact_duplicate_asset_keeps_volatility() -> None:
    rng = np.random.default_rng(5)
    panel = pd.DataFrame(
        rng.normal(0, 0.01, size=(400, 2)),
        columns=["a", "b"],
        index=_daily_index(400),
    )
    weights = {"a": 0.5, "b": 0.5}
    # 候选资产与现组合完全一致 → 加入后波动率不应变化
    candidate = panel.mul([0.5, 0.5]).sum(axis=1)
    out = correlation.marginal_impact(panel, weights, candidate, candidate_weight=0.2)
    assert out["volatility_after"] == pytest.approx(out["volatility_before"], rel=1e-9)
    assert out["correlation_to_portfolio"] == pytest.approx(1.0, rel=1e-9)
    assert out["volatility_change"] == pytest.approx(0.0, abs=1e-12)


def test_diversification_ratio_rewards_imperfect_correlation() -> None:
    rng = np.random.default_rng(9)
    panel = pd.DataFrame(
        rng.normal(0, 0.01, size=(600, 2)),
        columns=["a", "b"],
        index=_daily_index(600),
    )
    ratio = correlation.diversification_ratio(panel, {"a": 0.5, "b": 0.5})
    assert ratio > 1.0


def test_cluster_order_is_a_permutation() -> None:
    rng = np.random.default_rng(13)
    panel = pd.DataFrame(
        rng.normal(0, 0.01, size=(300, 4)),
        columns=["a", "b", "c", "d"],
        index=_daily_index(300),
    )
    corr = correlation.correlation_matrix(panel, min_obs=10)
    order = correlation.cluster_order(corr)
    assert sorted(order) == sorted(corr.columns)
