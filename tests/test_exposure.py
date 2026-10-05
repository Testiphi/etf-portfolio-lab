"""RBSA 风格分析的对照测试（不访问网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.core import exposure


def _factors(seed: int = 5, n: int = 800, cols: tuple[str, ...] = ("F1", "F2", "F3")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(n, len(cols))),
        columns=list(cols),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )


def test_rbsa_recovers_exact_loadings() -> None:
    """y = 0.5·F1 + 0.3·F2 + 0.2·F3（合计为 1）时，回归应当把系数原样还回来。"""
    factors = _factors()
    y = factors["F1"] * 0.5 + factors["F2"] * 0.3 + factors["F3"] * 0.2

    result = exposure.rbsa(y, factors, min_obs=100)

    betas = dict(zip(result.factors, result.betas))
    assert betas["F1"] == pytest.approx(0.5, abs=1e-4)
    assert betas["F2"] == pytest.approx(0.3, abs=1e-4)
    assert betas["F3"] == pytest.approx(0.2, abs=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.alpha_annual == pytest.approx(0.0, abs=1e-4)
    assert result.tracking_error == pytest.approx(0.0, abs=1e-6)


def test_rbsa_sum_constraint_forces_normalization_and_costs_r_squared() -> None:
    """真实载荷合计不为 1 时，强制「合计为 1」会整体压低系数，并留下残差。

    构造：w = (0.5, 0.3, 0.2) 合计为 1，但把它放大 1.3 倍作为 y。
    此时真实的生成载荷 (0.65, 0.39, 0.26) 合计 1.3，约束必须把它们压回合计 1，
    于是拟合无法完美 → R² 明显小于 1。**这正是 R² 必须和 beta 一起看的原因**：
    约束给了系数「权重」的含义，代价是当组合并非满仓由这些因子构成时必然有残差。
    """
    factors = _factors(seed=13)
    y = (factors["F1"] * 0.5 + factors["F2"] * 0.3 + factors["F3"] * 0.2) * 1.3

    result = exposure.rbsa(y, factors, min_obs=100)
    betas = dict(zip(result.factors, result.betas))
    unconstrained = dict(zip(result.factors, result.unconstrained_betas))

    assert sum(result.betas) == pytest.approx(1.0, abs=1e-8)
    assert result.r_squared < 0.99
    # 相对大小关系必须保留
    assert betas["F1"] > betas["F2"] > betas["F3"]
    # 约束把每个系数都整体压低（从合计 1.3 压到合计 1）
    assert all(betas[key] < unconstrained[key] for key in betas)


def test_rbsa_betas_are_non_negative_and_sum_to_one() -> None:
    factors = _factors(seed=9)
    y = factors["F1"] * 2.0 + factors["F2"] * (-1.0) + factors["F3"] * 0.2

    result = exposure.rbsa(y, factors, non_negative=True, sum_to_one=True, min_obs=100)

    assert sum(result.betas) == pytest.approx(1.0, abs=1e-8)
    assert all(b >= -1e-12 for b in result.betas)
    # 无约束解里 F2 是负的，约束后必须被压到 0
    assert dict(zip(result.factors, result.unconstrained_betas))["F2"] < 0
    assert dict(zip(result.factors, result.betas))["F2"] == pytest.approx(0.0, abs=1e-6)


def test_rbsa_unconstrained_differs_from_constrained() -> None:
    """约束前后的差异本身就是教学内容，必须两个都留下。"""
    factors = _factors(seed=11)
    y = factors["F1"] * 1.4 - factors["F2"] * 0.4
    result = exposure.rbsa(y, factors, min_obs=100)
    assert result.constrained is True
    assert result.betas != result.unconstrained_betas


def test_rbsa_rejects_small_sample() -> None:
    factors = _factors(n=40)
    y = factors["F1"] * 0.5 + factors["F2"] * 0.5
    with pytest.raises(ValueError, match="少于 min_obs"):
        exposure.rbsa(y, factors, min_obs=120)


def test_rbsa_warns_on_collinear_factors() -> None:
    """两个因子几乎相同时条件数会爆掉，必须给出警告而不是端出精确系数。"""
    factors = _factors(seed=3, cols=("F1", "F2"))
    rng = np.random.default_rng(4)
    factors["F2"] = factors["F1"] + rng.normal(0, 1e-6, len(factors))
    y = factors["F1"] * 0.5 + factors["F2"] * 0.5

    result = exposure.rbsa(y, factors, min_obs=100)
    assert result.condition_number > exposure.CONDITION_WARNING
    assert result.collinearity_warning is not None
    assert "条件数" in result.collinearity_warning
    # 共线性警告与解释力警告是两件独立的事，低 R² 的警告不能被它盖掉
    assert result.warning is None


def test_rbsa_warns_on_low_r_squared() -> None:
    factors = _factors(seed=21, cols=("F1",))
    rng = np.random.default_rng(22)
    y = pd.Series(rng.normal(0, 0.01, len(factors)), index=factors.index)

    result = exposure.rbsa(y, factors, min_obs=100)
    assert result.r_squared < 0.5
    assert result.warning is not None
    assert "R²" in result.warning


def test_exposure_matrix_has_portfolio_row_and_marks_failures() -> None:
    factors = _factors(seed=31, cols=("F1", "F2"))
    rng = np.random.default_rng(32)
    panel = pd.DataFrame(
        {
            "AAA": factors["F1"] * 0.8 + factors["F2"] * 0.2 + rng.normal(0, 1e-5, len(factors)),
            "BBB": rng.normal(0, 0.01, len(factors)),
        },
        index=factors.index,
    )

    matrix = exposure.exposure_matrix(panel, {"AAA": 0.5, "BBB": 0.5}, factors, min_obs=100)

    keys = [row["key"] for row in matrix["rows"]]
    assert keys[-1] == "__portfolio__"
    assert set(matrix["factors"]) == {"F1", "F2"}
    aaa = next(row for row in matrix["rows"] if row["key"] == "AAA")
    assert aaa["betas"]["F1"] == pytest.approx(0.8, abs=1e-3)
    assert aaa["r_squared"] > 0.99
    bbb = next(row for row in matrix["rows"] if row["key"] == "BBB")
    assert bbb["r_squared"] < 0.1


def test_exposure_matrix_reports_short_sample_instead_of_silent_blank() -> None:
    factors = _factors(seed=41, cols=("F1",), n=800)
    panel = pd.DataFrame({"AAA": factors["F1"] * 1.0}, index=factors.index)
    matrix = exposure.exposure_matrix(panel, {"AAA": 1.0}, factors, min_obs=900)
    row = matrix["rows"][0]
    assert "error" in row and "少于 min_obs" in row["error"]


def test_exposure_matrix_dedupes_collinearity_warning_and_reports_low_r2() -> None:
    """共线性只说一次；解释力不足（R² 低）逐行说。

    实测中黄金 ETF 对 A 股因子 R² 只有 0.007，却会得到一个 β=0.95 的"国债敞口"——
    这种数字必须被明确标注为不可靠，否则矩阵看着越精确越误导。
    """
    factors = _factors(seed=51, cols=("F1", "F2"))
    rng = np.random.default_rng(52)
    # F1 与 F2 高度相关 → 触发共线性警告
    factors["F2"] = factors["F1"] + rng.normal(0, 1e-6, len(factors))
    panel = pd.DataFrame(
        {
            "TRACKS": factors["F1"] * 0.5 + factors["F2"] * 0.5,
            "NOISE": rng.normal(0, 0.01, len(factors)),  # 与因子无关 → R² 极低
        },
        index=factors.index,
    )
    matrix = exposure.exposure_matrix(panel, {"TRACKS": 0.5, "NOISE": 0.5}, factors, min_obs=100)

    collinearity = [w for w in matrix["warnings"] if "条件数" in w]
    assert len(collinearity) == 1, "共线性警告应当只出现一次"
    assert any(w.startswith("NOISE：") and "R²" in w for w in matrix["warnings"])
