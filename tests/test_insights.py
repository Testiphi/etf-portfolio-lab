"""洞察规则引擎与风险/收益分解的对照测试（不访问网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.reports import insights, static_site


def _panel(seed: int = 3, n: int = 600, cols: tuple[str, ...] = ("a", "b", "c")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(n, len(cols))),
        columns=list(cols),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


# --------------------------------------------------------------------------- #
# 收益归因与风险贡献
# --------------------------------------------------------------------------- #
def test_arithmetic_contribution_sums_to_daily_return_sum() -> None:
    """算术贡献的各项之和 = 各日组合收益之和（这条是精确的）。

    注意两种精度：``return_contribution_sum`` 由未舍入的值求和后再舍入，因此可以严格对齐；
    而逐项展示用的 ``return_contribution`` 为控制 JSON 体积舍入到 8 位小数，
    把它们相加会带 ~1e-8 的舍入噪声——这正是归因面板必须单独给出"加总"数值的原因。
    """
    panel = _panel()
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    risk = static_site._risk_contribution(panel, weights)

    expected = float((panel.mul(pd.Series(weights), axis=1)).sum(axis=1).sum())
    # 返回值统一舍入到 8 位小数（控制 JSON 体积），因此对齐精度按 1e-8 而非 1e-9
    assert risk["return_contribution_sum"] == pytest.approx(expected, abs=1e-8)
    assert sum(risk["return_contribution"].values()) == pytest.approx(expected, abs=1e-7)


def test_arithmetic_contribution_differs_from_compounded_return() -> None:
    """归因之和**不等于**复利后的累计收益——这个差额必须如实展示，不能假装精确。

    这正是归因的固有难点：算术贡献加总与复利累计之间差一个交叉项。
    """
    panel = _panel(seed=11)
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    risk = static_site._risk_contribution(panel, weights)

    portfolio_returns = panel.mul(pd.Series(weights), axis=1).sum(axis=1)
    compounded = float((1.0 + portfolio_returns).prod() - 1.0)
    assert risk["return_contribution_sum"] != pytest.approx(compounded, rel=1e-6)
    # 量级应当接近，否则说明分解写错了
    assert abs(risk["return_contribution_sum"] - compounded) < 0.05


def test_component_var_shares_sum_to_one() -> None:
    panel = _panel(seed=5)
    weights = {"a": 0.4, "b": 0.4, "c": 0.2}
    risk = static_site._risk_contribution(panel, weights)
    shares = [v for v in risk["component_var_share"].values() if v is not None]
    assert sum(shares) == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# 洞察规则
# --------------------------------------------------------------------------- #
def _base_result() -> dict:
    return {
        "weights": {"a": 0.5, "b": 0.5},
        "risk_contribution": {"component_var_share": {"a": 0.5, "b": 0.5}},
        "diagnostics": {"max_corr_pair": {"a": "a", "b": "b", "rho": 0.3}, "n_obs": 3000, "adjustment_events": 0},
        "metrics": {
            "max_drawdown": -0.10,
            "max_drawdown_recovery_days": 60,
            "var_95_historical": 0.01,
            "var_95_parametric": 0.0105,
            "skewness": 0.0,
        },
        "dca": {"fixed": {"naive_annualized_return": 0.05, "xirr": 0.052}},
        "n_obs": 3000,
        "max_drawdown_info": {"depth": -0.10},
        "time_weighted_annualized": 0.07,
    }


def test_no_insight_on_well_balanced_portfolio() -> None:
    assert insights.evaluate(_base_result()) == []


def test_risk_contribution_gap_triggers() -> None:
    result = _base_result()
    result["weights"] = {"a": 0.5, "b": 0.5}
    result["risk_contribution"]["component_var_share"] = {"a": 0.8, "b": 0.2}
    found = insights.evaluate(result)
    keys = [i.key for i in found]
    assert "risk_contribution_gap" in keys
    gap = next(i for i in found if i.key == "risk_contribution_gap")
    assert "80%" in gap.title and gap.card == "risk_contribution"


def test_high_correlation_triggers() -> None:
    result = _base_result()
    result["diagnostics"]["max_corr_pair"] = {"a": "510300", "b": "510500", "rho": 0.95}
    found = insights.evaluate(result)
    assert any(i.key == "high_correlation" and i.card == "correlation" for i in found)


def test_dca_algorithm_gap_triggers_and_is_informational() -> None:
    result = _base_result()
    result["dca"]["fixed"] = {"naive_annualized_return": 0.037, "xirr": 0.069}
    found = insights.evaluate(result)
    item = next((i for i in found if i.key == "dca_gap"), None)
    assert item is not None
    assert item.level == "info"
    assert "3.70%" in item.title and "6.90%" in item.title


def test_slow_recovery_and_small_sample_trigger() -> None:
    result = _base_result()
    result["max_drawdown_info"] = {"depth": -0.55}
    result["metrics"]["max_drawdown"] = -0.55
    result["metrics"]["max_drawdown_recovery_days"] = 1300
    result["n_obs"] = 120
    result["diagnostics"]["n_obs"] = 120
    keys = [i.key for i in insights.evaluate(result)]
    assert "slow_recovery" in keys
    assert "small_sample" in keys


def test_var_method_divergence_and_left_tail() -> None:
    result = _base_result()
    result["metrics"]["var_95_historical"] = 0.02
    result["metrics"]["var_95_parametric"] = 0.028
    result["metrics"]["skewness"] = -0.8
    keys = [i.key for i in insights.evaluate(result)]
    assert "var_method_gap" in keys
    assert "left_tail" in keys


def test_return_concentration_uses_positive_contributions() -> None:
    result = _base_result()
    result["diagnostics"]["concentration"] = {"symbol": "a", "return_share": 0.91, "basis": "positive"}
    found = insights.evaluate(result)
    item = next((i for i in found if i.key == "return_concentration"), None)
    assert item is not None and "91%" in item.title


def test_warn_insights_come_first() -> None:
    result = _base_result()
    result["risk_contribution"]["component_var_share"] = {"a": 0.9, "b": 0.1}
    result["dca"]["fixed"] = {"naive_annualized_return": 0.01, "xirr": 0.06}
    found = insights.evaluate(result)
    levels = [i.level for i in found]
    assert levels == sorted(levels, key=lambda x: {"warn": 0, "info": 1}.get(x, 9))


def test_evaluate_is_robust_to_empty_result() -> None:
    """缺字段时应当少一条提醒，而不是让整页崩掉。"""
    assert insights.evaluate({}) == []
    assert insights.evaluate({"weights": {"a": 1.0}}) == []


# --------------------------------------------------------------------------- #
# 解锁机制
# --------------------------------------------------------------------------- #
def test_unlocks_require_the_right_portfolio_structure() -> None:
    items = {i["key"]: i for i in insights.evaluate_unlocks({"has_bond": True})}
    assert items["duration"]["triggered"] is True
    assert items["fx_exposure"]["triggered"] is False
    assert items["exposure_matrix"]["triggered"] is True  # always 类模块

    items2 = {i["key"]: i for i in insights.evaluate_unlocks({"has_cross_border": True})}
    assert items2["fx_exposure"]["triggered"] is True
    assert items2["duration"]["triggered"] is False


def test_unlocks_flag_unimplemented_modules_honestly() -> None:
    """组合结构满足但功能未上线时，必须标成"已触发·待接入"，不能假装可用。"""
    items = {i["key"]: i for i in insights.evaluate_unlocks({"has_cross_border": True})}
    fx = items["fx_exposure"]
    assert fx["triggered"] is True
    assert fx["implemented"] is False
    assert fx["pending"]


def test_unlock_catalog_cards_exist_in_teaching_content() -> None:
    """每个解锁模块引用的知识卡片都必须真实存在，否则点开会报错。"""
    from etf_lab.content import teaching

    for item in insights.UNLOCK_CATALOG:
        assert item["card"] in teaching.CARDS, item["key"]


def test_every_insight_card_exists_in_teaching_content() -> None:
    from etf_lab.content import teaching

    result = _base_result()
    result["risk_contribution"]["component_var_share"] = {"a": 0.9, "b": 0.1}
    result["diagnostics"]["max_corr_pair"] = {"a": "a", "b": "b", "rho": 0.99}
    result["diagnostics"]["adjustment_events"] = 12
    result["diagnostics"]["adjustment_symbols"] = ["a"]
    result["diagnostics"]["concentration"] = {"symbol": "a", "return_share": 0.95}
    result["metrics"]["var_95_parametric"] = 0.05
    result["metrics"]["skewness"] = -1.2
    result["n_obs"] = 100
    result["diagnostics"]["n_obs"] = 100
    result["max_drawdown_info"] = {"depth": -0.5}
    result["metrics"]["max_drawdown_recovery_days"] = 900
    result["dca"]["fixed"] = {"naive_annualized_return": 0.01, "xirr": 0.09}
    for item in insights.evaluate(result):
        assert item.card in teaching.CARDS, item.card
