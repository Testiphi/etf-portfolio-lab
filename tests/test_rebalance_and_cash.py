"""再平衡规则与现金/外币合成资产的对照测试（不访问网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.core import returns, synthetic


def _prices(values: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(next(iter(values.values()))), freq="B")
    return pd.DataFrame(values, index=index).astype(float)


# --------------------------------------------------------------------------- #
# 再平衡：每日必须与旧行为逐点等价（这是"默认不变"的根据）
# --------------------------------------------------------------------------- #
def test_daily_policy_reproduces_the_old_behaviour_exactly() -> None:
    """``policy="daily"`` 必须与旧的 ``portfolio_returns`` 逐点一致。

    否则这次改动会悄无声息地改掉全站所有历史数字。
    """
    prices = _prices({"AAA": [1.0, 1.02, 1.05, 0.98, 1.10, 1.12], "BBB": [2.0, 2.01, 2.03, 2.02, 2.05, 2.06]})
    weights = {"AAA": 0.6, "BBB": 0.4}
    new = returns.rebalanced_returns(prices, weights, policy="daily")
    old = returns.to_simple(
        returns.portfolio_returns(returns.to_returns(prices), weights=weights), method="log"
    )
    assert np.allclose(new.to_numpy(), old.to_numpy(), rtol=1e-12, atol=1e-14)


def test_never_policy_lets_weights_drift() -> None:
    """买入持有：某个标的翻倍后，它的权重应当明显高于目标。"""
    prices = _prices({"AAA": [1.0, 1.5, 2.0, 2.5, 3.0], "BBB": [1.0, 1.0, 1.0, 1.0, 1.0]})
    weights = {"AAA": 0.5, "BBB": 0.5}
    drift = returns.rebalanced_returns(prices, weights, policy="never")
    daily = returns.rebalanced_returns(prices, weights, policy="daily")
    # 买入持有的累计收益应当高于每日再平衡（赢家在组合里权重越来越大）
    assert float((1 + drift).prod()) > float((1 + daily).prod())


def test_monthly_policy_lands_between_daily_and_never() -> None:
    """月度再平衡的累计收益应当落在"每日"与"从不"之间（漂移被部分放回）。"""
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(3)
    trend = np.linspace(1.0, 2.2, len(index))
    prices = pd.DataFrame(
        {"AAA": trend * (1 + rng.normal(0, 0.01, len(index))).cumprod(),
         "BBB": np.linspace(1.0, 1.25, len(index)) * (1 + rng.normal(0, 0.004, len(index))).cumprod()},
        index=index,
    ).abs()
    weights = {"AAA": 0.6, "BBB": 0.4}
    totals = {
        policy: float((1 + returns.rebalanced_returns(prices, weights, policy=policy)).prod())
        for policy in ("daily", "monthly", "never")
    }
    assert totals["daily"] <= totals["monthly"] <= totals["never"] or totals["never"] <= totals["monthly"] <= totals["daily"]


def test_threshold_policy_reacts_to_divergence() -> None:
    """阈值策略：偏离越大触发越多；阈值设得很松时就更接近"买入持有"。"""
    index = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        {"AAA": (1 + rng.normal(0.002, 0.03, len(index))).cumprod(),
         "BBB": (1 + rng.normal(0.0005, 0.005, len(index))).cumprod()},
        index=index,
    ).abs()
    weights = {"AAA": 0.5, "BBB": 0.5}
    tight = returns.rebalanced_returns(prices, weights, policy="threshold", threshold=0.005)
    loose = returns.rebalanced_returns(prices, weights, policy="threshold", threshold=0.5)
    never = returns.rebalanced_returns(prices, weights, policy="never")
    assert np.allclose(loose.to_numpy(), never.to_numpy()), "阈值极松时应当等价于买入持有"
    assert not np.allclose(tight.to_numpy(), never.to_numpy())


def test_rebalance_cost_reduces_returns() -> None:
    """扣成本必须让收益变低，且成本越高收益越低。"""
    prices = _prices({"AAA": [1.0, 1.2, 0.9, 1.3, 0.8, 1.4], "BBB": [1.0, 0.9, 1.1, 0.85, 1.2, 0.95]})
    weights = {"AAA": 0.5, "BBB": 0.5}
    free = float((1 + returns.rebalanced_returns(prices, weights, policy="daily")).prod())
    cheap = float((1 + returns.rebalanced_returns(prices, weights, policy="daily", cost_bps=10)).prod())
    pricey = float((1 + returns.rebalanced_returns(prices, weights, policy="daily", cost_bps=50)).prod())
    assert pricey < cheap < free


def test_rebalance_rejects_missing_values_and_bad_policy() -> None:
    prices = _prices({"AAA": [1.0, 1.1, 1.2], "BBB": [1.0, np.nan, 1.0]})
    with pytest.raises(ValueError, match="缺失值"):
        returns.rebalanced_returns(prices, {"AAA": 0.5, "BBB": 0.5})
    clean = _prices({"AAA": [1.0, 1.1, 1.2]})
    with pytest.raises(ValueError, match="未知再平衡规则"):
        returns.rebalanced_returns(clean, {"AAA": 1.0}, policy="fortnightly")


def test_policy_labels_cover_every_policy() -> None:
    assert set(returns.REBALANCE_POLICIES) == {"daily", "monthly", "quarterly", "annually", "never", "threshold"}
    assert all(returns.REBALANCE_POLICIES[key] for key in returns.REBALANCE_POLICIES)


def test_nav_from_prices_exposes_the_policy() -> None:
    prices = _prices({"AAA": [1.0, 1.3, 1.6, 1.9], "BBB": [1.0, 1.0, 1.0, 1.0]})
    weights = {"AAA": 0.5, "BBB": 0.5}
    daily = returns.nav_from_prices(prices, weights=weights, policy="daily")
    never = returns.nav_from_prices(prices, weights=weights, policy="never")
    assert daily.iloc[0] == pytest.approx(1.0) and never.iloc[0] == pytest.approx(1.0)
    assert never.iloc[-1] > daily.iloc[-1]


# --------------------------------------------------------------------------- #
# 现金与美元
# --------------------------------------------------------------------------- #
def _curve(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"date": dates, "code": "CN1Y", "tenor": "1年", "yield": values})


def _fx(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"pair": "USDCNY", "date": dates, "close": values})


def test_cash_price_compounds_at_the_short_rate() -> None:
    """现金按年化 2.52% 计息一年，应当涨约 2.52%（日频复利）。"""
    index = pd.date_range("2020-01-01", periods=253, freq="B")
    curve = _curve([2.52] * 253)
    price = synthetic.cash_price_series(curve, index)
    assert len(price) == 253
    assert price.iloc[-1] / price.iloc[0] == pytest.approx(1.0252, rel=0.002)
    assert price.is_monotonic_increasing, "现金价格必须单调（无回撤）"
    assert price.pct_change().std() < 1e-6, "现金不该有波动"


def test_cash_uses_last_published_rate_between_publications() -> None:
    """曲线只在部分日子发布时，沿用最近一次利率——这是计息的定义，不是填充数据。"""
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    sparse = _curve([2.0, 2.0], start="2020-01-01")  # 只在第 1、2 天有
    price = synthetic.cash_price_series(sparse, index)
    assert len(price) == 10
    assert price.is_monotonic_increasing


def test_cash_rejects_window_starting_before_the_data() -> None:
    """区间早于利率数据时必须报错，而不是外推。"""
    index = pd.date_range("2019-01-01", periods=10, freq="B")
    curve = _curve([2.0] * 10, start="2020-01-01")
    with pytest.raises(synthetic.SyntheticError, match="早于数据起点"):
        synthetic.cash_price_series(curve, index)


def test_usd_price_tracks_fx_and_optional_interest() -> None:
    index = pd.date_range("2020-01-01", periods=253, freq="B")
    # 汇率从 7.0 升到 7.7（人民币贬值 10%）
    rates = list(np.linspace(7.0, 7.7, 253))
    fx = _fx(rates)
    zero = synthetic.usd_price_series(fx, index, annual_rate=0.0)
    assert zero.iloc[-1] / zero.iloc[0] == pytest.approx(7.7 / 7.0, rel=1e-9)
    with_interest = synthetic.usd_price_series(fx, index, annual_rate=0.05)
    assert with_interest.iloc[-1] / with_interest.iloc[0] == pytest.approx((7.7 / 7.0) * 1.05, rel=0.002)
    assert with_interest.iloc[-1] > zero.iloc[-1]


def test_usd_default_has_no_interest_assumption() -> None:
    """默认必须是不生息——不能把当前的美元利率套用到 2012 年那种低利率时期。"""
    import inspect

    signature = inspect.signature(synthetic.usd_price_series)
    assert signature.parameters["annual_rate"].default == 0.0


def test_synthetic_prices_builds_only_requested_symbols() -> None:
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    frame = synthetic.synthetic_prices(
        index, symbols=[synthetic.CASH_SYMBOL, synthetic.USD_SYMBOL],
        curve=_curve([2.0] * 20), fx=_fx([7.0] * 20),
    )
    assert list(frame.columns) == [synthetic.CASH_SYMBOL, synthetic.USD_SYMBOL]
    assert len(frame) == len(index)
    with pytest.raises(synthetic.SyntheticError, match="未知的合成资产"):
        synthetic.synthetic_prices(index, symbols=["GOLD"], curve=_curve([2.0] * 20))


def test_synthetic_meta_marks_cash_and_usd() -> None:
    meta = synthetic.synthetic_meta([synthetic.CASH_SYMBOL, synthetic.USD_SYMBOL])
    assert set(meta["symbol"]) == {"CASH", "USD"}
    assert set(meta["asset_class"]) == {"cash", "fx_cash"}
    assert bool(meta.loc[meta["symbol"] == "USD", "is_cross_border"].iloc[0]) is True
    assert bool(meta.loc[meta["symbol"] == "CASH", "is_cross_border"].iloc[0]) is False


def test_cash_visible_in_a_portfolio_reduces_volatility() -> None:
    """现金的实际作用：降低波动。用同一条链算一遍，确认它真的起作用。"""
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    rng = np.random.default_rng(7)
    equity = pd.Series((1 + rng.normal(0.0003, 0.02, len(index))).cumprod(), index=index)
    cash = synthetic.cash_price_series(_curve([2.0] * len(index)), index)
    prices = pd.DataFrame({"AAA": equity, "CASH": cash})
    risky = returns.nav_from_prices(prices, weights={"AAA": 1.0})
    mixed = returns.nav_from_prices(prices, weights={"AAA": 0.5, "CASH": 0.5})
    assert mixed.pct_change().std() < risky.pct_change().std() * 0.6
