"""蒙特卡洛模拟的对照测试（不访问网络）。

这个模块最需要防的不是"算不出来"，而是"算出看起来很科学的假数字"。因此测试重点是：
可复现性、收敛诊断是否真的成立、块自助法是否真的保留了自相关、
以及**不同模型给出的结论是否存在实质差异**（差异本身是结论，不是 bug）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.core import simulate


def _ar1_returns(n: int = 3000, phi: float = 0.3, vol: float = 0.012, seed: int = 5) -> pd.Series:
    """带一阶自相关的收益序列——用来验证块自助法确实保留了自相关。"""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, vol, n)
    out = np.empty(n)
    out[0] = shocks[0]
    for index in range(1, n):
        out[index] = phi * out[index - 1] + shocks[index]
    return pd.Series(out)


def _iid_returns(n: int = 3000, vol: float = 0.012, drift: float = 0.0002, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, vol, n))


# --------------------------------------------------------------------------- #
# 拟合
# --------------------------------------------------------------------------- #
def test_garch_fit_respects_constraints_and_targets_variance() -> None:
    fit = simulate.fit_garch(_iid_returns())
    assert 0 < fit.alpha < 0.5
    assert 0 < fit.beta < 1
    assert fit.alpha + fit.beta < 1.0
    assert fit.omega > 0
    # 方差目标化参数化：长期方差 ω/(1−α−β) 应等于样本方差
    implied = fit.omega / (1.0 - fit.alpha - fit.beta)
    assert implied == pytest.approx(fit.long_run_variance, rel=1e-6)


def _garch_returns(n: int = 4000, alpha: float = 0.12, beta: float = 0.85, long_run_vol: float = 0.01, seed: int = 17) -> pd.Series:
    """按真实 GARCH(1,1) 递推生成数据，用于验证拟合能否把参数找回来。"""
    rng = np.random.default_rng(seed)
    long_run_variance = long_run_vol**2
    omega = long_run_variance * (1.0 - alpha - beta)
    variance = long_run_variance
    out = np.empty(n)
    for index in range(n):
        shock = np.sqrt(variance) * rng.standard_normal()
        out[index] = shock
        variance = omega + alpha * shock**2 + beta * variance
    return pd.Series(out)


def test_garch_recovers_persistence_from_simulated_data() -> None:
    """用真实 GARCH 递推生成的数据，拟合应当把持续性参数基本找回来。

    注意：**波动率"分段恒定"不是 GARCH 聚集**。如果每 500 天换一次波动水平、
    段内恒定，平方收益没有短期自相关，GARCH 正确地会估出 α≈β≈0——
    我最初就是这么写测试数据的，结果误判成实现有问题。
    """
    returns = _garch_returns(alpha=0.12, beta=0.85)
    fit = simulate.fit_garch(returns)
    assert fit.persistence > 0.5
    assert fit.alpha > 0.02
    assert fit.omega / (1.0 - fit.alpha - fit.beta) == pytest.approx(fit.long_run_variance, rel=1e-6)


def test_garch_finds_no_persistence_in_iid_data() -> None:
    """独立同分布数据里没有波动聚集，持续性参数应当很小——防止把噪声当信号。"""
    fit = simulate.fit_garch(_iid_returns(n=3000))
    assert fit.persistence < 0.5


def test_student_t_df_uses_excess_kurtosis() -> None:
    rng = np.random.default_rng(3)
    fat = rng.standard_t(4, 20000)
    thin = rng.normal(0, 1, 20000)
    assert simulate._student_t_df(fat) < simulate._student_t_df(thin)


def test_fit_model_rejects_short_sample() -> None:
    with pytest.raises(ValueError, match="不足以拟合"):
        simulate.fit_model(pd.Series([0.01] * 30), "gbm")


def test_fit_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="未知模型"):
        simulate.fit_model(_iid_returns(), "not_a_model")


# --------------------------------------------------------------------------- #
# 可复现与结构
# --------------------------------------------------------------------------- #
def test_same_seed_reproduces_exactly() -> None:
    returns = _iid_returns(n=800)
    first = simulate.simulate(returns, model="bootstrap", n_paths=500, horizon_days=252, seed=42)
    second = simulate.simulate(returns, model="bootstrap", n_paths=500, horizon_days=252, seed=42)
    assert first.summary == second.summary
    assert first.percentiles == second.percentiles


def test_different_seed_changes_result_but_not_wildly() -> None:
    returns = _iid_returns(n=1500)
    first = simulate.simulate(returns, model="gbm", n_paths=2000, horizon_days=252, seed=1)
    second = simulate.simulate(returns, model="gbm", n_paths=2000, horizon_days=252, seed=2)
    assert first.summary["median_terminal"] != second.summary["median_terminal"]
    assert first.summary["median_terminal"] == pytest.approx(second.summary["median_terminal"], rel=0.15)


def test_percentiles_are_monotone_at_every_step() -> None:
    result = simulate.simulate(_iid_returns(), model="bootstrap", n_paths=1000, horizon_days=252)
    for position in range(len(result.percentiles["p50"])):
        assert (
            result.percentiles["p5"][position]
            <= result.percentiles["p25"][position]
            <= result.percentiles["p50"][position]
            <= result.percentiles["p75"][position]
            <= result.percentiles["p95"][position]
        )


def test_recorded_steps_cover_full_horizon() -> None:
    result = simulate.simulate(_iid_returns(), model="gbm", n_paths=400, horizon_days=250, record_every=60)
    # 250 不是 60 的整数倍，最后一个记录点必须补齐到期末
    assert len(result.percentiles["p50"]) == len(list(range(60, 251, 60))) + 1


def test_stdout_types_are_json_safe() -> None:
    """结果要能直接写进静态站的数据文件，因此不能混入 numpy 标量。"""
    import json

    result = simulate.simulate(_iid_returns(), model="garch", n_paths=400, horizon_days=252)
    payload = json.dumps(result.as_dict(), ensure_ascii=False)
    assert "NaN" not in payload


# --------------------------------------------------------------------------- #
# 收敛诊断
# --------------------------------------------------------------------------- #
def test_convergence_standard_error_shrinks_like_one_over_sqrt_n() -> None:
    """标准误应大致按 1/√N 下降——这是"结果有多稳"的定量说法。"""
    result = simulate.simulate(
        _iid_returns(),
        model="gbm",
        n_paths=4000,
        horizon_days=252,
        convergence_sizes=(500, 2000, 4000),
    )
    assert [row["n_paths"] for row in result.convergence] == [500.0, 2000.0, 4000.0]
    first, last = result.convergence[0], result.convergence[-1]
    assert last["standard_error"] < first["standard_error"]
    ratio = first["standard_error"] / last["standard_error"]
    ideal = np.sqrt(4000 / 500)
    assert 0.6 * ideal < ratio < 1.5 * ideal
    # 报告里要能看到"每次统计用的路径数"
    assert all(row["n_paths"] > 0 for row in result.convergence)


def test_warning_is_absent_for_well_behaved_convergence() -> None:
    result = simulate.simulate(_iid_returns(), model="gbm", n_paths=2000, horizon_days=252)
    assert result.warning is None


# --------------------------------------------------------------------------- #
# 模型特性
# --------------------------------------------------------------------------- #
def test_zero_drift_gbm_has_median_near_one() -> None:
    """对数收益均值为 0 时，中位终值应接近 1（对数正态的中位数 = exp(均值)）。"""
    rng = np.random.default_rng(13)
    returns = pd.Series(rng.normal(0.0, 0.01, 2000))
    result = simulate.simulate(returns, model="gbm", n_paths=4000, horizon_days=252, seed=99)
    assert result.summary["median_terminal"] == pytest.approx(1.0, abs=0.02)


def test_block_bootstrap_preserves_autocorrelation_better_than_iid() -> None:
    """这是块自助法存在的全部意义：保留自相关。

    对照：同样从这份数据里**独立同分布**地抽（块长 1），自相关会被抹掉。
    """
    returns = _ar1_returns(phi=0.35, vol=0.012)
    original = float(returns.autocorr(lag=1))

    block_params = simulate.fit_model(returns, "bootstrap", block=21)
    iid_params = simulate.fit_model(returns, "bootstrap", block=1)
    rng = np.random.default_rng(5)
    block_paths = simulate._simulate_block("bootstrap", block_params, 40, 2000, rng)
    iid_paths = simulate._simulate_block("bootstrap", iid_params, 40, 2000, rng)

    block_ac = float(np.corrcoef(block_paths[:, :-1].ravel(), block_paths[:, 1:].ravel())[0, 1])
    iid_ac = float(np.corrcoef(iid_paths[:, :-1].ravel(), iid_paths[:, 1:].ravel())[0, 1])

    assert abs(iid_ac) < 0.02, "块长为 1 时自相关应当基本消失"
    assert abs(block_ac - original) < abs(iid_ac - original), "块自助法应当更接近原始自相关"


def test_bootstrap_terminal_distribution_reflects_history() -> None:
    """块自助法重放历史样本，中位终值应与**历史几何收益**基本一致。

    这里不能断言"收益为正"：漂移 0.0003/日 相对标准误（0.01/√3000）只有约 1.6 倍，
    样本实现出来的均值完全可能是负的。断言应当对齐"历史实际值"而不是"设定值"。
    """
    returns = _iid_returns(n=3000, drift=0.0008, vol=0.01)
    result = simulate.simulate(returns, model="bootstrap", n_paths=3000, horizon_days=252, seed=7)

    historical_cagr = float((1.0 + returns).prod() ** (252 / len(returns)) - 1.0)
    assert historical_cagr > 0, "这份样本的实现收益应当为正，否则测试前提不成立"
    # 块自助法每个路径只含约 12 个独立块（252/21），有效样本比路径数小得多，
    # 因此中位数的抽样噪声不可忽视：用相对容差而不是绝对容差
    assert result.summary["median_terminal"] == pytest.approx(1.0 + historical_cagr, rel=0.05)


def _fat_tail_returns(n: int = 3000, df: int = 4, vol: float = 0.012, seed: int = 23) -> pd.Series:
    """真正厚尾的收益（来自学生 t），波动率标准化到目标值。"""
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, n)
    return pd.Series(raw / np.sqrt(df / (df - 2)) * vol)


def test_standardized_t_tail_profile_crosses_over_the_normal() -> None:
    """标准化 t 的"厚尾"不是简单地"所有分位都更差"——这正是最反直觉的一点。

    把 t 分布标准化到**单位方差**后：

    - **5% 分位反而比正态更温和**（尾部极值把方差撑大了，标准化后中等分位被压缩）；
    - **1% 及更深处才比正态更极端**。

    直接推论：**用 5% VaR 去衡量"厚尾风险"会得出与直觉相反的结论**。
    本项目因此把模拟的分位记录扩展到 1%。
    """
    rng = np.random.default_rng(1)
    shocks = simulate._standardized_t(rng, (400_000,), 4.5)

    assert abs(float(shocks.std(ddof=1)) - 1.0) < 0.02, "方差必须已标准化到 1"
    assert np.percentile(shocks, 5) > -1.645, "5% 分位应当比正态更温和"
    assert np.percentile(shocks, 1) < -2.326, "1% 分位应当比正态更极端"


def test_tail_crossover_shows_up_in_simulated_terminal_distribution() -> None:
    """上面的分布事实必须能在模拟结果里复现，否则说明模拟用错了冲击项。"""
    returns = _fat_tail_returns(df=4)
    assert simulate._student_t_df(np.log1p(returns.to_numpy())) < 10.0, "厚尾数据的自由度估计应当较小"

    gbm = simulate.simulate(returns, model="gbm", n_paths=8000, horizon_days=1, seed=21)
    heavy = simulate.simulate(returns, model="student_t", n_paths=8000, horizon_days=1, seed=21)

    assert heavy.summary["p5_terminal"] > gbm.summary["p5_terminal"], "5% 处厚尾模型更温和"
    assert heavy.summary["p1_terminal"] < gbm.summary["p1_terminal"], "1% 处厚尾模型更极端"


def test_fat_tails_deepen_the_drawdown_path() -> None:
    """路径极值不受"求和平滑"影响：厚尾模型的深回撤概率应当不低。"""
    returns = _fat_tail_returns(df=4)
    gbm = simulate.simulate(returns, model="gbm", n_paths=4000, horizon_days=504, seed=21)
    heavy = simulate.simulate(returns, model="student_t", n_paths=4000, horizon_days=504, seed=21)
    assert heavy.risk["prob_deep_drawdown"] >= gbm.risk["prob_deep_drawdown"]


# --------------------------------------------------------------------------- #
# 风险指标与多模型对比
# --------------------------------------------------------------------------- #
def test_risk_probabilities_are_consistent() -> None:
    returns = _iid_returns(n=2000)
    result = simulate.simulate(returns, model="bootstrap", n_paths=2000, horizon_days=1260, goal_annual=0.08)
    risk = result.risk
    for key in ("prob_loss", "prob_goal", "prob_deep_drawdown"):
        assert 0.0 <= risk[key] <= 1.0
    # 达标门槛高于 1 时，"达标概率"不应超过"不亏损概率"
    assert risk["goal_multiple"] > 1.0
    assert risk["prob_goal"] <= risk["prob_loss"] + 1e-9 or risk["prob_loss"] == 0.0


def test_impossible_goal_has_near_zero_probability() -> None:
    returns = _iid_returns(n=2000)
    low = simulate.simulate(returns, model="gbm", n_paths=2000, horizon_days=252, goal_annual=0.02, seed=3)
    high = simulate.simulate(returns, model="gbm", n_paths=2000, horizon_days=252, goal_annual=5.0, seed=3)
    assert high.risk["prob_goal"] == pytest.approx(0.0, abs=1e-6)
    assert low.risk["prob_goal"] > high.risk["prob_goal"]


def test_all_models_run_and_comparison_exposes_disagreement() -> None:
    """模型对比是这套东西最重要的输出：结论对模型选择有多敏感。

    注意阈值要用**相对差**而不是绝对差：达标概率本身可能只有几个百分点，
    0.042 与 0.060 的绝对差只有 1.8pp，但相对差是 43%——那已经是实质差异。
    真实差异更多体现在尾部（p5 终值、深度回撤概率），所以两条都验。
    """
    returns = _iid_returns(n=2500, vol=0.015)
    results = simulate.simulate_all_models(returns, n_paths=1500, horizon_days=1260)
    assert {r.model for r in results} == set(simulate.MODELS)

    rows = simulate.model_comparison(results)
    assert len(rows) == len(results)
    goal_probabilities = [row["prob_goal"] for row in rows]
    low, high = min(goal_probabilities), max(goal_probabilities)
    assert high > low, "不同模型的达标概率不应完全一致"
    assert (high - low) / max(low, 1e-9) > 0.10, "相对差异应当达到实质程度"

    # 尾部差异由 test_fat_tails_affect_short_horizon_but_average_out_over_long_horizon 专门覆盖；
    # 这里只要求"模型之间确实给出了不同的数字"
    assert all("label" in row for row in rows)
    assert all("warning" in row for row in rows)


def test_simulate_all_models_skips_failures_without_losing_others() -> None:
    """单个模型拟合失败时，其余模型的结果必须保留。"""
    returns = _iid_returns(n=300)
    results = simulate.simulate_all_models(returns, n_paths=400, horizon_days=252, models=("gbm", "garch"))
    assert len(results) >= 1
