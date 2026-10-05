"""Delta-Gamma 复制模拟的对照测试（不访问网络）。

这个模块有**闭式对照**可用，所以测试不能只测"跑得通"：

* Boyle–Emanuel：离散对冲误差的标准差 ∝ √Δt
* 模拟误差标准差应与解析预期 ``√(Σ½Γ²σ⁴S⁴Δt²)`` 同量级

这两条是数值正确性的硬证据。相反，如果只断言"误差 > 0"，实现错了也照样通过。
"""

from __future__ import annotations

import numpy as np
import pytest

from etf_lab.core import derivatives, hedge


def _plan(**overrides) -> hedge.ReplicationPlan:
    base = dict(
        spot=4.0,
        sigma=0.20,
        r=0.012,
        tenor_years=1.0,
        strike_ratio=0.95,
        target_ratio=0.5,
        rebalance_days=21,
        cost_bps=5.0,
    )
    base.update(overrides)
    return hedge.ReplicationPlan(**base)


# --------------------------------------------------------------------------- #
# 目标头寸
# --------------------------------------------------------------------------- #
def test_protection_units_matches_the_target_delta() -> None:
    """名义量必须让组合的初始 delta 正好等于目标比例。"""
    plan = _plan()
    units = 1.0
    notional = hedge.protection_units(plan, units)
    greeks = derivatives.bs_greeks(plan.spot, plan.strike, plan.tenor_years, plan.r, plan.sigma, "put")
    total_delta = units + notional * greeks.delta
    assert total_delta == pytest.approx(plan.target_ratio * units, rel=1e-12)


@pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 0.75])
def test_protection_units_grows_as_target_delta_falls(target: float) -> None:
    """保护越强（目标 delta 越小），需要的认沽名义量越大。"""
    notional = hedge.protection_units(_plan(target_ratio=target))
    assert notional > 0
    assert notional > hedge.protection_units(_plan(target_ratio=min(target + 0.2, 0.95)))


def test_no_hedge_means_zero_notional() -> None:
    assert hedge.protection_units(_plan(target_ratio=1.0)) == pytest.approx(0.0, abs=1e-12)


def test_target_value_at_expiry_is_the_put_payoff() -> None:
    plan = _plan()
    notional = 2.0
    assert hedge.target_value(np.array([plan.spot]), plan, 1.0, notional)[0] == pytest.approx(plan.spot)
    below = plan.strike - 0.5
    expected = below + notional * 0.5
    assert hedge.target_value(np.array([below]), plan, 1.0, notional)[0] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 路径与单路径复制
# --------------------------------------------------------------------------- #
def test_gbm_paths_shape_and_seed_reproducibility() -> None:
    first = hedge.gbm_paths(4.0, 0.012, 0.2, 1.0, 100, seed=1)
    second = hedge.gbm_paths(4.0, 0.012, 0.2, 1.0, 100, seed=1)
    assert first.shape == (100, 253)
    assert np.array_equal(first, second)
    assert np.all(first > 0)
    assert np.allclose(first[:, 0], 4.0)


def test_gbm_paths_have_expected_annualised_volatility() -> None:
    paths = hedge.gbm_paths(1.0, 0.0, 0.25, 2.0, 4000, seed=5)
    logs = np.log(paths[:, -1])
    assert logs.std(ddof=1) / np.sqrt(2.0) == pytest.approx(0.25, rel=0.05)


def test_continuous_rebalancing_error_is_small() -> None:
    """每天再平衡应非常接近连续对冲：误差远小于不复制的情况。"""
    plan = _plan(rebalance_days=1)
    paths = hedge.gbm_paths(plan.spot, plan.r, plan.sigma, plan.tenor_years, 500, seed=3)
    daily = hedge.run_replication(paths, plan)
    coarse_plan = _plan(rebalance_days=126)
    coarse = hedge.run_replication(paths, coarse_plan)
    assert daily.error_std < coarse.error_std / 3
    assert abs(daily.error_mean) < 0.02


def test_no_hedge_at_all_reproduces_target_exactly() -> None:
    """目标比例 = 1 时不需要任何交易：误差为 0，也不该被收建仓成本。

    复制组合从"已经持有标的"出发；若从零买入，就会对"买入自己已持有的 ETF"
    也收一笔钱——那不是对冲成本。
    """
    plan = _plan(target_ratio=1.0)
    paths = hedge.gbm_paths(plan.spot, plan.r, plan.sigma, plan.tenor_years, 50, seed=9)
    run = hedge.run_replication(paths, plan)
    assert run.error_std == pytest.approx(0.0, abs=1e-12)
    assert run.mean_cost == pytest.approx(0.0, abs=1e-12)
    assert run.error_mean == pytest.approx(0.0, abs=1e-12)


def test_costs_scale_linearly_with_cost_bps() -> None:
    """成本必须与 cost_bps 严格成正比（交易路径不依赖成本水平）。

    ``cost_sensitivity`` 正是靠这条性质做缩放而不重跑模拟的，所以要钉死。
    """
    plan = _plan(rebalance_days=21)
    paths = hedge.gbm_paths(plan.spot, plan.r, plan.sigma, plan.tenor_years, 200, seed=11)
    cheap = hedge.run_replication(paths, plan)
    expensive_plan = _plan(rebalance_days=21, cost_bps=plan.cost_bps * 4)
    expensive = hedge.run_replication(paths, expensive_plan)
    assert expensive.mean_cost == pytest.approx(cheap.mean_cost * 4, rel=1e-9)
    # 毛误差（把成本加回去）应当**几乎**与成本水平无关，但不必完全无关：
    # 成本从现金里扣、现金按无风险利率复利，所以早付的成本会少赚一点利息。
    # 实测这个残留约 6e-5（相对），因此容差取 1e-3——比它宽 15 倍，
    # 仍能挡住"成本污染了误差离散度"这类真问题（那会是几十个百分点的偏差）。
    residual = abs(expensive.error_std - cheap.error_std) / cheap.error_std
    assert residual < 1e-3, f"成本对误差离散度的影响过大：{residual:.2e}"
    assert expensive.error_std == pytest.approx(cheap.error_std, rel=1e-3)
    # 毛误差**均值**同样受这个利息效应影响，但要用绝对容差：
    # 成本差 3 倍（约 27bp/年）只让毛误差均值移动约 2bp（0.00002），
    # 而毛误差均值本身只有 1bp 量级——相对比较会把 0.2bp 的效应放大成 15%，
    # 看起来像 bug，其实经济上可以忽略。
    mean_shift = abs(expensive.error_mean - cheap.error_mean)
    assert mean_shift < 5e-5, f"成本对毛误差均值的影响过大：{mean_shift:.2e}"


# --------------------------------------------------------------------------- #
# 标度律：这是本模块最重要的数值证据
# --------------------------------------------------------------------------- #
def test_error_scales_as_sqrt_of_rebalance_interval() -> None:
    """Boyle–Emanuel：离散对冲误差的标准差 ∝ √Δt。

    间隔缩短到 1/4，误差标准差应降到约 1/2。
    """
    paths = hedge.gbm_paths(4.0, 0.012, 0.2, 2.0, 4000, seed=20260101)
    coarse_plan = _plan(tenor_years=2.0, rebalance_days=84)
    fine_plan = _plan(tenor_years=2.0, rebalance_days=21)
    coarse = hedge.run_replication(paths, coarse_plan)
    fine = hedge.run_replication(paths, fine_plan)
    ratio = fine.error_std / coarse.error_std
    assert ratio == pytest.approx(np.sqrt(0.25), rel=0.15)


def test_error_grows_as_interval_grows_but_cost_falls() -> None:
    """两个幂律方向相反：误差随间隔上升，成本随间隔下降——最优频率因此存在。"""
    paths = hedge.gbm_paths(4.0, 0.012, 0.2, 1.0, 1500, seed=13)
    dense = hedge.run_replication(paths, _plan(rebalance_days=5))
    sparse = hedge.run_replication(paths, _plan(rebalance_days=63))
    assert sparse.error_std > dense.error_std
    assert sparse.mean_cost < dense.mean_cost
    assert sparse.mean_trades < dense.mean_trades


def test_simulated_error_matches_analytic_prediction_in_magnitude() -> None:
    """模拟误差标准差与解析预期 √(Σ½Γ²σ⁴S⁴Δt²) 应当同量级。

    容差放到 2 倍是诚实的：Boyle–Emanuel 是忽略高阶项的近似，
    且它在区间**起点**评估 Γ。但比值若偏离到 2 倍以上，就说明实现或标度错了。
    """
    paths = hedge.gbm_paths(4.0, 0.012, 0.2, 1.0, 4000, seed=17)
    for days in (5, 21, 63):
        run = hedge.run_replication(paths, _plan(rebalance_days=days))
        assert 0.5 < run.scale_ratio < 2.0, f"间隔 {days} 日的解析/模拟比值异常：{run.scale_ratio}"


def test_gross_error_and_cost_are_reported_separately() -> None:
    """毛误差与成本必须分开报，且净结果 = 毛误差 − 成本。

    混在一起有两个后果：把成本重复计算一次；以及误差的**离散度**会随成本假设变化
    （成本逐路径不同），让"复制误差"这个数字失去意义。
    """
    paths = hedge.gbm_paths(4.0, 0.012, 0.2, 1.0, 2000, seed=19)
    run = hedge.run_replication(paths, _plan(rebalance_days=21))
    assert run.net_error_mean == pytest.approx(run.error_mean - run.mean_cost, rel=1e-12)
    assert run.mean_cost > 0
    # 纯粹的复制偏差应当远小于成本量级，或至少同量级——不该被成本淹没
    assert abs(run.error_mean) < 10 * run.mean_cost


# --------------------------------------------------------------------------- #
# 成本敏感度
# --------------------------------------------------------------------------- #
def test_cost_sensitivity_shifts_the_optimal_frequency() -> None:
    """成本越高，最优再平衡频率越稀疏——这是"最优频率"这句话的全部意义。"""
    analysis = hedge.analyse(_plan(cost_bps=5.0), n_paths=600, rebalance_grid=(1, 5, 21, 126))
    sensitivity = analysis.sensitivity
    cheap = next(row for row in sensitivity if row["cost_bps"] == 2.0)
    pricey = next(row for row in sensitivity if row["cost_bps"] == 50.0)
    assert pricey["optimal_rebalance_days"] >= cheap["optimal_rebalance_days"]
    assert pricey["optimal_mean_cost"] > cheap["optimal_mean_cost"]


def test_cost_sensitivity_uses_linear_rescaling() -> None:
    """缩放后的成本必须等于按比例放大后的真实成本。"""
    runs = [hedge.ReplicationRun(rebalance_days=21, error_mean=0.0, error_std=0.01, error_p5=-0.02,
                                 error_p95=0.02, predicted_std=0.009, mean_cost=0.001, mean_trades=12.0, n_paths=1000)]
    rows = hedge.cost_sensitivity(runs, base_cost_bps=5.0, cost_grid=(5.0, 25.0))
    assert rows[0]["optimal_mean_cost"] == pytest.approx(0.001)
    assert rows[1]["optimal_mean_cost"] == pytest.approx(0.005)


# --------------------------------------------------------------------------- #
# 泰勒分解的适用边界
# --------------------------------------------------------------------------- #
def test_taylor_residual_grows_with_step_and_shock() -> None:
    """二阶近似的残差随步长与冲击幅度上升——这是 Delta-Gamma-Theta 的适用边界。"""
    rows = hedge.taylor_residuals(_plan(), horizons=(1, 63), shocks=(0.01, 0.10))
    small = next(row for row in rows if row["days"] == 1 and row["shock"] == 0.01)
    large = next(row for row in rows if row["days"] == 63 and row["shock"] == 0.10)
    assert small["mean_abs_residual_pct"] < large["mean_abs_residual_pct"]
    # 小步长小冲击下近似应当几乎精确
    assert small["mean_abs_residual_pct"] < 0.01


def test_taylor_breaks_down_for_extreme_shocks() -> None:
    """极端冲击下二阶近似会给出超过期权本身价格的变动——必须如实暴露这一点。

    期权价格有下界（0）与上界（K），而二次多项式没有；
    所以"用 Greeks 估算极端行情损失"这个常见做法在大幅冲击下是危险的。
    """
    rows = hedge.taylor_residuals(_plan(), horizons=(1,), shocks=(0.50,))
    assert rows
    assert rows[0]["max_abs_residual_pct"] > 0.5


# --------------------------------------------------------------------------- #
# 整体分析
# --------------------------------------------------------------------------- #
def test_analyse_produces_all_sections_and_is_json_safe() -> None:
    import json

    result = hedge.analyse(_plan(), n_paths=800, rebalance_grid=(1, 21, 126))
    payload = json.dumps(result.as_dict(), ensure_ascii=False)
    assert "NaN" not in payload
    assert len(result.runs) == 3
    assert result.best is not None
    assert result.premium["premium_pct_of_portfolio"] > 0
    assert len(result.taylor) > 0
    assert len(result.sensitivity) > 0
    assert result.plan["notional_units"] > 0


def test_analyse_marks_the_best_frequency_by_total_burden() -> None:
    result = hedge.analyse(_plan(), n_paths=800, rebalance_grid=(1, 5, 21, 63))
    best = min(result.runs, key=lambda row: row["total_burden"])
    assert result.best["rebalance_days"] == best["rebalance_days"]


def test_analyse_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        hedge.gbm_paths(1.0, 0.0, 0.2, 0.0, 10)
    with pytest.raises(ValueError):
        hedge.gbm_paths(1.0, 0.0, 0.2, 1.0, 0)
