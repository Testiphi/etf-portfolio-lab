"""RBSA：收益法风格分析（Returns-Based Style Analysis，Sharpe 1992）。

为什么要用它
------------
要算"风险敞口"最直接的办法是穿透指数成分股，但免费源缺少可靠的历史时点成分名单，
幸存者偏差还会让结果失真。RBSA 换了条路：**不猜持仓里有什么，而是看你实际跟随了谁**——
用一组因子指数的收益去回归组合收益，回归系数就是敞口。

三个必须写在页面上的诚实说明
----------------------------
1. **这是近似**，不是真实持仓穿透；它衡量的是"收益共同波动"而非"持有什么"。
2. **R² 是这套分析的质量指标**：R² 低说明这组因子解释不了你的组合，
   这时beta 数字再好看也没有意义。
3. **多重共线性会稀释 beta**：因子指数之间高度相关时（沪深300/中证500/中证全指常常如此），
   单个 beta 不稳定。因此这里额外报告**条件数**，并在过高时明确警告，
   而不是端出一组看起来很精确的系数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

TRADING_DAYS_PER_YEAR = 252
DEFAULT_MIN_OBS = 120
"""少于约半年样本就不做回归：beta 会完全被噪声主导。"""
CONDITION_WARNING = 30.0
"""设计矩阵条件数超过该值 → beta 对样本扰动敏感，需明确提示。"""


@dataclass(frozen=True)
class RbsaResult:
    """一次风格分析的结果。"""

    factors: tuple[str, ...]
    betas: tuple[float, ...]
    """受约束（非负、和为 1）的敞口系数。"""
    unconstrained_betas: tuple[float, ...]
    """无约束 OLS 系数，用于对照"约束把结论改了多少"。"""
    alpha_annual: float
    """残差均值年化：因子解释不了的那部分收益。"""
    r_squared: float
    tracking_error: float
    n_obs: int
    condition_number: float
    constrained: bool

    def as_dict(self, factor_names: Sequence[str] | None = None) -> dict[str, Any]:
        names = list(factor_names or self.factors)
        return {
            "betas": {name: round(float(b), 6) for name, b in zip(names, self.betas)},
            "unconstrained_betas": {name: round(float(b), 6) for name, b in zip(names, self.unconstrained_betas)},
            "alpha_annual": round(float(self.alpha_annual), 6),
            "r_squared": round(float(self.r_squared), 6),
            "tracking_error": round(float(self.tracking_error), 6),
            "n_obs": int(self.n_obs),
            "condition_number": round(float(self.condition_number), 3),
            "constrained": self.constrained,
            "warning": self.warning,
        }

    @property
    def warning(self) -> str | None:
        """**解释力**不足的警告。这是比共线性更要紧的问题：
        R² 低意味着 β 本身没有意义，此时端出精确到小数点后三位的系数是误导。
        """
        if self.r_squared < 0.5:
            return f"R² 仅 {self.r_squared:.2f}：这组因子解释不了它的收益，敞口数字参考价值有限"
        return None

    @property
    def collinearity_warning(self) -> str | None:
        """因子之间高度相关的警告（整块矩阵共用一个，因此不逐行重复）。"""
        if self.condition_number > CONDITION_WARNING:
            return (
                f"因子之间高度相关（条件数 {self.condition_number:.1f}），"
                "单个 β 对样本区间敏感，请结合 R² 一起看"
            )
        return None


def rbsa(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    non_negative: bool = True,
    sum_to_one: bool = True,
    min_obs: int = DEFAULT_MIN_OBS,
) -> RbsaResult:
    """对组合收益做受约束回归，得到因子敞口。

    Parameters
    ----------
    non_negative, sum_to_one
        加上"敞口非负且合计为 1"的约束。这两条约束不是数学必需，而是**经济先验**：
        指数 ETF 组合不会做空因子，敞口也没有理由长期合计偏离 1。
        约束同时显著缓解多重共线性造成的系数漂移。
    """
    if factor_returns.empty:
        raise ValueError("因子收益面板为空")

    joined = pd.concat([portfolio_returns.rename("__p__"), factor_returns], axis=1, join="inner").dropna()
    if len(joined) < min_obs:
        raise ValueError(f"对齐后样本仅 {len(joined)} 条，少于 min_obs={min_obs}，不做回归")

    y = joined["__p__"].to_numpy(dtype=float)
    factor_names = [c for c in joined.columns if c != "__p__"]
    X = joined[factor_names].to_numpy(dtype=float)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("收益数据中存在非有限值")

    condition_number = float(np.linalg.cond(X))

    # 无约束 OLS（含截距）——只作为对照，不用于结论
    design = np.column_stack([np.ones(len(X)), X])
    ols, *_ = np.linalg.lstsq(design, y, rcond=None)
    unconstrained = tuple(float(v) for v in ols[1:])

    if non_negative or sum_to_one:
        x0 = np.full(X.shape[1], 1.0 / X.shape[1])

        def objective(beta: np.ndarray) -> float:
            residual = y - X @ beta
            return float(residual @ residual)

        constraints = []
        if sum_to_one:
            constraints.append({"type": "eq", "fun": lambda beta: float(np.sum(beta) - 1.0)})
        bounds = [(0.0, 1.0)] * X.shape[1] if non_negative else [(None, None)] * X.shape[1]
        solution = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not solution.success:
            raise ValueError(f"受约束回归未收敛：{solution.message}")
        betas = tuple(float(v) for v in solution.x)
        if sum_to_one:
            betas = tuple(v / sum(betas) for v in betas) if sum(betas) != 0 else betas
    else:
        betas = unconstrained

    beta_array = np.asarray(betas, dtype=float)
    residual = y - X @ beta_array
    ss_res = float(residual @ residual)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha_annual = float((1.0 + residual.mean()) ** TRADING_DAYS_PER_YEAR - 1.0)
    tracking_error = float(residual.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return RbsaResult(
        factors=tuple(factor_names),
        betas=betas,
        unconstrained_betas=unconstrained,
        alpha_annual=alpha_annual,
        r_squared=float(r_squared),
        tracking_error=tracking_error,
        n_obs=int(len(joined)),
        condition_number=condition_number,
        constrained=bool(non_negative or sum_to_one),
    )


def exposure_matrix(
    panel: pd.DataFrame,
    portfolio_weights: Mapping[str, float],
    factor_returns: pd.DataFrame,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    skip_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    """对**每只标的**与**整个组合**各做一次 RBSA，得到敞口矩阵。

    返回结构直接可渲染：行是标的（最后一行是组合），列是因子。
    解释不了的标的（样本不足）会被标记出来，而不是悄悄留空。

    ``skip_symbols`` 用于**近常数资产**（现金、外币现金）：以它们为被解释变量做 RBSA
    会退化成噪声，并打印出一行经济上无意义的敞口。注意**组合那一行不受影响**——
    RBSA 的 beta 是归一化到和为 1 的相对权重，且给组合加常数序列不改变相关系数，
    所以"跳过现金会修正组合 beta"是错的；跳过的意义只是不打印无意义的行。
    """
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    skip_set = {str(symbol) for symbol in skip_symbols}
    collinearity_reported = False

    def _register(result: RbsaResult, label: str) -> None:
        nonlocal collinearity_reported
        # 共线性是整块矩阵的共同问题，只说一次，避免四行重复同样的话
        if not collinearity_reported and result.collinearity_warning:
            warnings.append(result.collinearity_warning)
            collinearity_reported = True
        if result.warning:
            warnings.append(f"{label}：{result.warning}")

    for symbol in panel.columns:
        if str(symbol) in skip_set:
            rows.append(
                {
                    "key": str(symbol),
                    "name": str(symbol),
                    "skipped": "零波动资产（现金类）没有可估的因子敞口，按定义为 0",
                }
            )
            continue
        try:
            result = rbsa(panel[symbol].dropna(), factor_returns, min_obs=min_obs)
        except ValueError as exc:
            rows.append({"key": str(symbol), "name": str(symbol), "error": str(exc)})
            continue
        rows.append({"key": str(symbol), "name": str(symbol), **result.as_dict(result.factors)})
        _register(result, str(symbol))

    # 组合层：按权重合成（每日再平衡），与页面上其它指标保持一致的口径
    weight_series = pd.Series({k: float(v) for k, v in portfolio_weights.items() if k in panel.columns})
    if not weight_series.empty:
        total = float(weight_series.sum())
        weight_series = weight_series / total
        portfolio_returns = panel[list(weight_series.index)].mul(weight_series, axis=1).sum(axis=1, min_count=len(weight_series))
        try:
            result = rbsa(portfolio_returns.dropna(), factor_returns, min_obs=min_obs)
            rows.append({"key": "__portfolio__", "name": "组合", **result.as_dict(result.factors)})
            _register(result, "组合")
        except ValueError as exc:
            rows.append({"key": "__portfolio__", "name": "组合", "error": str(exc)})

    factor_names = list(factor_returns.columns)
    return {
        "factors": factor_names,
        "rows": rows,
        "warnings": warnings,
        "note": "敞口 = 用因子指数收益对组合收益做受约束回归（非负、合计 1）得到的系数；R² 是解释力，条件数反映因子共线性",
    }
