"""第 4 层（引擎已有、界面暴露）与现金/美元的集成测试。

重点守住三件事：
1. **界面暴露的参数必须是引擎真的读的参数**——手写一个引擎不认识的参数，
   用户会以为它生效了，其实什么也没发生（定投模式选择器就出过这个问题）；
2. **现金/美元进入组合后，敞口回归要跳过它们、但组合 beta 不能重新归一化**——
   后者会把"持有 50% 现金"的组合 beta 高估回 1.0；
3. **再平衡规则要真的改变结果**，否则又是一个摆设。
"""

from __future__ import annotations

import pytest

from etf_lab import universe
from etf_lab.core import dca as dca_core
from etf_lab.core import returns as returns_core
from etf_lab.core import synthetic as synthetic_core
from etf_lab.data import repo
from etf_lab.reports import static_site

from test_render_structure import _seed_db, _spec


@pytest.fixture()
def connection(tmp_path):
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    yield con
    con.close()


# --------------------------------------------------------------------------- #
# 参数表与引擎的一致性
# --------------------------------------------------------------------------- #
def test_param_specs_only_expose_parameters_the_engine_reads() -> None:
    """界面的参数表不能出现引擎不读的参数。

    引擎真正 ``params.get`` 的名字是固定的；界面多写一个，用户就以为它生效了，
    而实际上是静默无效——这正是"定投模式选择器是摆设"那类 bug 的温床。
    """
    engine_params = {
        "return_window",
        "target_vol",
        "min_mult",
        "max_mult",
        "growth",
        "allow_sell",
        "take_profit",
        "take_fraction",
        "cooldown_days",
    }
    exposed = {spec[0] for specs in dca_core.PARAM_SPECS.values() for spec in specs}
    assert exposed <= engine_params, f"界面暴露了引擎不读的参数：{sorted(exposed - engine_params)}"
    # 参数表必须覆盖每一种模式，否则某些模式就没有任何可调参数
    assert set(dca_core.PARAM_SPECS) == set(dca_core.MODES)
    assert set(dca_core.FREQ_LABELS) == {"daily", "weekly", "monthly", "quarterly"}


def test_every_param_spec_has_sane_bounds() -> None:
    for mode, specs in dca_core.PARAM_SPECS.items():
        for name, default, low, high, step, label in specs:
            assert low <= default <= high, f"{mode}.{name} 默认值不在区间内"
            assert step > 0, f"{mode}.{name} 步长必须为正"
            assert label, f"{mode}.{name} 缺少界面标签"


# --------------------------------------------------------------------------- #
# 再平衡：必须真的改变结果
# --------------------------------------------------------------------------- #
def test_rebalance_policy_changes_the_numbers(connection) -> None:
    daily = static_site.compute_preset(connection, _spec("rb_daily"))
    monthly = static_site.compute_preset(
        connection, _spec("rb_monthly", rebalance={"policy": "monthly", "cost_bps": 10.0})
    )
    assert daily["rebalance"]["policy"] == "daily"
    assert monthly["rebalance"]["policy"] == "monthly"
    # 两只以上标的、且走势不同，规则不同 → 结果必须不同
    assert daily["metrics"]["annualized_return"] != pytest.approx(
        monthly["metrics"]["annualized_return"], abs=1e-9
    )


def test_rebalance_block_covers_every_policy(connection) -> None:
    result = static_site.compute_preset(connection, _spec("rb_all"))
    rows = result["rebalance"]["rows"]
    assert {row["policy"] for row in rows} == set(returns_core.REBALANCE_POLICIES)
    assert sum(1 for row in rows if row["is_current"]) == 1
    assert result["rebalance"]["best_policy"] in returns_core.REBALANCE_POLICIES


def test_rebalance_cost_is_reflected(connection) -> None:
    """同规则下，成本越高年化越低——否则成本参数是个摆设。"""
    free = static_site.compute_preset(connection, _spec("cost0", rebalance={"policy": "monthly"}))
    pricey = static_site.compute_preset(
        connection, _spec("cost30", rebalance={"policy": "monthly", "cost_bps": 30.0})
    )
    assert pricey["metrics"]["annualized_return"] < free["metrics"]["annualized_return"]


# --------------------------------------------------------------------------- #
# 现金与美元
# --------------------------------------------------------------------------- #
def test_cash_enters_the_portfolio_and_reduces_risk(connection) -> None:
    equity = static_site.compute_preset(connection, _spec("only_equity", include_bond=False))
    mixed = static_site.compute_preset(
        connection, _spec("with_cash", include_bond=False, weights={"AAA": 0.5, "CASH": 0.5})
    )
    assert "CASH" in mixed["labels"] and "现金" in mixed["labels"]["CASH"]
    assert mixed["composition"]["by_asset_class"]["cash"] == pytest.approx(0.5)
    assert mixed["composition"]["has_cash"] is True
    assert mixed["metrics"]["annualized_volatility"] < equity["metrics"]["annualized_volatility"] * 0.8


def test_exposure_matrix_skips_zero_vol_assets_without_disturbing_the_portfolio_row() -> None:
    """现金被跳过的是**逐标的回归**，组合那一行不受影响。

    这里我一开始想错了一次：以为"把现金从面板剔掉会让组合 beta 被高估"，并写了
    "期望 beta≈0.5"的断言——结果实现给出 1.0，测试当场证伪。真实情况是
    RBSA 的 beta 为**归一化到和为 1 的相对权重**，而且给组合加一条常数序列
    不改变相关系数，所以现金在不在面板里，组合 beta 都一样。
    跳过的意义只是**不要打印一行无意义的「CASH beta」**。
    """
    import numpy as np
    import pandas as pd

    from etf_lab.core import exposure as exposure_mod

    rng = np.random.default_rng(5)
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    factor = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    asset = factor * 1.0 + pd.Series(rng.normal(0, 0.0005, len(index)), index=index)
    cash = pd.Series(0.0002, index=index)
    panel = pd.DataFrame({"AAA": asset, "CASH": cash})
    factors = pd.DataFrame({"MKT": factor})

    matrix = exposure_mod.exposure_matrix(
        panel, {"AAA": 0.5, "CASH": 0.5}, factors, min_obs=60, skip_symbols=["CASH"]
    )
    rows = {row["key"]: row for row in matrix["rows"]}
    assert rows["CASH"].get("skipped"), "现金应当被标记为跳过，并给出原因"
    assert rows["AAA"]["betas"]["MKT"] == pytest.approx(1.0, abs=0.15)
    portfolio = rows["__portfolio__"]
    assert sum(portfolio["betas"].values()) == pytest.approx(1.0, abs=0.02), (
        "RBSA 的 beta 归一化到和为 1，组合行也不例外"
    )
    # 顺带钉住一个**容易误读**的事实：约束回归（Σβ=1）无法表达组合的波动尺度，
    # 所以单因子、半仓现金的组合 R² 会接近 0。**beta 不是风险倍数**，
    # 现金的效果只体现在波动率与回撤上，不在 beta 里。
    assert portfolio["r_squared"] < 0.5, "约束回归捕捉不到尺度差异，这里本就不该有高 R²"


def test_exposure_matrix_without_skip_still_runs_but_reports_a_cash_row() -> None:
    """不传 skip 时现金会照常进入回归——这正是要避免的：一行没有意义的敞口。"""
    import numpy as np
    import pandas as pd

    from etf_lab.core import exposure as exposure_mod

    rng = np.random.default_rng(9)
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    factor = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    panel = pd.DataFrame({"AAA": factor + rng.normal(0, 0.0005, len(index)), "CASH": 0.0002}, index=index)
    matrix = exposure_mod.exposure_matrix(panel, {"AAA": 0.5, "CASH": 0.5}, pd.DataFrame({"MKT": factor}), min_obs=60)
    rows = {row["key"]: row for row in matrix["rows"]}
    assert "skipped" not in rows["CASH"], "不传 skip_symbols 时现金会被照常回归（对照组）"


def test_usd_requires_fx_data_and_says_so() -> None:
    """没有汇率数据时必须给出可执行的提示，而不是拿假设值顶替。"""
    with pytest.raises(synthetic_core.SyntheticError, match="汇率"):
        synthetic_core.required_start([synthetic_core.USD_SYMBOL], fx=None)


def test_required_start_takes_the_latest_of_all_synthetic_inputs() -> None:
    import pandas as pd

    curve = pd.DataFrame(
        {"date": pd.date_range("2015-01-05", periods=5, freq="B"), "code": "CN1Y", "yield": [2.0] * 5}
    )
    fx = pd.DataFrame(
        {"pair": "USDCNY", "date": pd.date_range("2012-01-04", periods=5, freq="B"), "close": [6.3] * 5}
    )
    start = synthetic_core.required_start(
        [synthetic_core.CASH_SYMBOL, synthetic_core.USD_SYMBOL], curve=curve, fx=fx
    )
    # 现金要求更晚的起点（2015），所以整组必须从 2015 起
    assert str(start.date()) == "2015-01-05"


def test_synthetic_asset_reports_late_start_clearly() -> None:
    with pytest.raises(synthetic_core.SyntheticError, match="收益率曲线"):
        synthetic_core.required_start([synthetic_core.CASH_SYMBOL], curve=None)


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #
def test_dca_with_non_first_day_no_longer_crashes() -> None:
    """``day`` 大于 1 时定投统计不能崩。

    这是暴露 ``day`` 参数时才撞出来的**既有缺陷**：定投市值曲线在第一笔投入之前是 0，
    而 ``metrics.summary`` 要求净值首值为正，于是"第一笔不在序列第一天"就直接抛错。
    以前界面不提供 ``day``，所以这个分支从没被走到——**把选项暴露出来，
    会把潜伏的缺陷一起暴露出来**，这本身是好事。
    """
    import numpy as np
    import pandas as pd

    from etf_lab.core import dca as dca_core

    index = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(3)
    nav = pd.Series((1 + rng.normal(0.0003, 0.01, len(index))).cumprod(), index=index)

    for freq, day in (("monthly", None), ("monthly", 3), ("quarterly", 5), ("weekly", 2)):
        plan = dca_core.DcaPlan(amount=1000.0, freq=freq, day=day, mode="fixed")
        result = dca_core.simulate(plan, nav)
        assert result.final_value > 0
        assert np.isfinite(result.xirr)
        # 统计块要么有内容、要么为空，但**不能因为曲线前导 0 而抛错**
        assert isinstance(result.metrics, dict)


def test_dca_value_curve_before_first_contribution_is_not_treated_as_nav() -> None:
    """第一笔投入之前，市值曲线为 0——这段不能参与净值统计。"""
    import numpy as np
    import pandas as pd

    from etf_lab.core import dca as dca_core

    index = pd.date_range("2020-01-01", periods=200, freq="B")
    nav = pd.Series(np.linspace(1.0, 1.3, len(index)), index=index)
    plan = dca_core.DcaPlan(amount=1000.0, freq="monthly", day=10, mode="fixed")
    result = dca_core.simulate(plan, nav)
    assert float(result.value.iloc[0]) == 0.0, "第一笔投入之前市值应当为 0"
    assert float(result.invested_curve.iloc[0]) == 0.0
    assert float(result.value.iloc[-1]) > 0


def test_result_exposes_cash_and_rebalance_sections(connection) -> None:
    result = static_site.compute_preset(connection, _spec("sections"))
    assert "rebalance" in result and "rebalance_options" in result
    assert "cash" in result and "usd_annual_rate" in result["cash"]
    assert result["cash"]["cash_tenor"] == "CN1Y"
    # 界面要从这里取规则清单，不能手写
    assert set(result["rebalance_options"]) == set(returns_core.REBALANCE_POLICIES)


def test_synthetic_labels_use_the_same_chain_as_real_assets(connection) -> None:
    """合成资产的人读标签必须与真实 ETF 走同一条链（名称（代码 · 板块））。"""
    result = static_site.compute_preset(
        connection, _spec("labels", include_bond=False, weights={"AAA": 0.7, "CASH": 0.3})
    )
    assert result["labels"]["CASH"] == f"{universe.symbol_label('CASH', '人民币现金', 'cash')}"
    assert "人民币现金" in result["labels"]["CASH"]
