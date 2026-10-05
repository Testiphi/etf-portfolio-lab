"""蒙特卡洛模拟：把"一个点估计"变成"一个分布"，并强制暴露它的假设。

这个模块最容易变成"看起来很科学的编数字"，所以有三条硬性设计：

1. **模型不唯一，必须能对比**。同一个组合在 GBM、块自助法、Student-t、GARCH 下
   算出的达标概率可以差很多——这个差异本身就是最重要的结论。
   页面必须把多个模型并排展示，而不是挑一个好看的。
2. **必须给收敛诊断**。路径数不够时，结果自己就在抖；
   报告标准误随路径数的下降，让人看见"这个数字有多稳"。
3. **随机必须可复现**。所有函数强制传 ``seed``，同一个种子结果逐位一致。

关于 GARCH
----------
``arch`` 这类库在本项目的部署路线里装不上（Pyodide 只有平台专用二进制 wheel，
没有纯 Python wheel），所以这里**自己实现 GARCH(1,1) 的极大似然估计**，
并用**方差目标化**参数化（``ω = σ̄²(1-α-β)``）把待估参数降到两个——
这不仅更稳，也让长期方差天然与样本一致。

诚实边界：**它只能复现你假设的分布**。用历史数据拟合出来的模型，
永远模拟不出历史里没有出现过的极端事件（这是所有回测与模拟共同的边界）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS_PER_YEAR = 252
MODELS = ("bootstrap", "gbm", "student_t", "garch")
MODEL_LABELS = {
    "bootstrap": "块自助法（保留自相关与厚尾）",
    "gbm": "几何布朗运动（对数正态）",
    "student_t": "Student-t 独立同分布（厚尾）",
    "garch": "GARCH(1,1)（波动聚集）",
}

DEFAULT_BLOCK = 21
"""块自助法的块长（约一个月）：太短保留了自相关，太长会反复重放同一段历史。"""


@dataclass(frozen=True)
class GarchFit:
    """GARCH(1,1) 的拟合结果（方差目标化参数化）。"""

    omega: float
    alpha: float
    beta: float
    long_run_variance: float
    nu: float
    """标准化学生 t 的自由度（矩估计，用在模拟的冲击项上）。"""
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


@dataclass(frozen=True)
class SimulationResult:
    """一次蒙特卡洛的结果（全部为可 JSON 序列化的普通类型）。"""

    model: str
    label: str
    n_paths: int
    horizon_days: int
    record_every: int
    seed: int
    percentiles: dict[str, list[float]]
    """各记录时点的净值分位带，用于扇形图。"""
    summary: dict[str, float]
    risk: dict[str, float]
    convergence: list[dict[str, float]]
    histogram: dict[str, list[float]] = field(default_factory=dict)
    """终值分布的直方图（counts / edges），用于画分布而不是只给几个分位。"""
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def warning(self) -> str | None:
        """收敛不达标时给出警告，而不是让人相信一个还在抖的数字。"""
        if len(self.convergence) < 2:
            return None
        first, last = self.convergence[0], self.convergence[-1]
        if first["standard_error"] <= 0 or last["standard_error"] <= 0:
            return None
        ratio = first["standard_error"] / last["standard_error"]
        expected = np.sqrt(last["n_paths"] / first["n_paths"])
        if ratio < expected * 0.75:
            return (
                f"标准误下降慢于 1/√N（实际 {ratio:.2f}×，理想 {expected:.2f}×）："
                "结果对路径数仍敏感，请提高路径数再看"
            )
        return None


# --------------------------------------------------------------------------- #
# 参数拟合
# --------------------------------------------------------------------------- #
def _log_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 60:
        raise ValueError(f"样本仅 {len(values)} 个观测，不足以拟合模拟模型（至少 60）")
    return np.log1p(values)


def _student_t_df(log_returns: np.ndarray) -> float:
    """用超额峰度做矩估计得到自由度：df = 6/峰度 + 4，并夹在合理区间内。"""
    excess = float(stats.kurtosis(log_returns, fisher=True))
    if not np.isfinite(excess) or excess <= 0.1:
        return 30.0
    return float(np.clip(6.0 / excess + 4.0, 3.5, 60.0))


def fit_garch(returns: pd.Series | np.ndarray, *, max_iter: int = 200) -> GarchFit:
    """GARCH(1,1) 的极大似然估计（方差目标化，只估 alpha 与 beta）。

    σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁，其中 ω = σ̄²(1−α−β) 保证长期方差等于样本方差。
    """
    log_ret = _log_returns(returns)
    demeaned = log_ret - log_ret.mean()
    long_run_variance = float(np.var(demeaned, ddof=1))
    if long_run_variance <= 0:
        raise ValueError("收益方差为 0，无法拟合 GARCH")

    def negative_loglik(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        omega = long_run_variance * (1.0 - alpha - beta)
        if omega <= 0:
            return 1e12
        variance = np.empty_like(demeaned)
        variance[0] = long_run_variance
        for index in range(1, len(demeaned)):
            variance[index] = omega + alpha * demeaned[index - 1] ** 2 + beta * variance[index - 1]
        variance = np.maximum(variance, 1e-12)
        return float(0.5 * np.sum(np.log(variance) + demeaned**2 / variance))

    result = optimize.minimize(
        negative_loglik,
        x0=np.array([0.08, 0.88]),
        method="SLSQP",
        bounds=[(1e-6, 0.5), (1e-6, 0.999)],
        constraints=[{"type": "ineq", "fun": lambda p: 0.999 - (p[0] + p[1])}],
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    alpha, beta = float(result.x[0]), float(result.x[1])
    return GarchFit(
        omega=long_run_variance * (1.0 - alpha - beta),
        alpha=alpha,
        beta=beta,
        long_run_variance=long_run_variance,
        nu=_student_t_df(demeaned),
        loglik=float(-result.fun),
    )


# --------------------------------------------------------------------------- #
# 单块收益模拟（返回 (n_paths, horizon) 的简单收益）
# --------------------------------------------------------------------------- #
def _standardized_t(rng: np.random.Generator, size: tuple[int, ...], nu: float) -> np.ndarray:
    """均值 0、方差 1 的学生 t 冲击项。"""
    raw = rng.standard_t(nu, size=size)
    return raw / np.sqrt(nu / (nu - 2.0))


def _block_bootstrap(rng: np.random.Generator, sample: np.ndarray, n: int, horizon: int, block: int) -> np.ndarray:
    length = len(sample)
    blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, length, size=(n, blocks))
    offsets = np.arange(block)
    # 环形取样：走到末尾就回到开头，避免丢掉样本尾部
    indices = (starts[:, :, None] + offsets[None, None, :]) % length
    return sample[indices].reshape(n, blocks * block)[:, :horizon]


def _simulate_block(
    model: str,
    params: Mapping[str, Any],
    n: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成 ``(n, horizon)`` 的简单收益矩阵。"""
    if model == "bootstrap":
        return _block_bootstrap(rng, params["sample"], n, horizon, params["block"])

    if model == "gbm":
        shocks = rng.standard_normal((n, horizon))
        return np.expm1(params["mu"] + params["sigma"] * shocks)

    if model == "student_t":
        shocks = _standardized_t(rng, (n, horizon), params["nu"])
        return np.expm1(params["mu"] + params["sigma"] * shocks)

    if model == "garch":
        alpha, beta, omega = params["alpha"], params["beta"], params["omega"]
        nu, mu = params["nu"], params["mu"]
        shocks = _standardized_t(rng, (n, horizon), nu)
        variance = np.full(n, params["long_run_variance"])
        out = np.empty((n, horizon))
        for step in range(horizon):
            shock = np.sqrt(variance) * shocks[:, step]
            out[:, step] = np.expm1(mu + shock)
            # GARCH 的核心：这一期的大波动会抬高下一期的方差
            variance = omega + alpha * shock**2 + beta * variance
        return out

    raise ValueError(f"未知模型：{model!r}")


def fit_model(returns: pd.Series | np.ndarray, model: str, *, block: int = DEFAULT_BLOCK) -> dict[str, Any]:
    """按模型拟合参数。"""
    if model not in MODELS:
        raise ValueError(f"未知模型：{model!r}；可选 {MODELS}")
    log_ret = _log_returns(returns)
    if model == "bootstrap":
        return {"sample": np.expm1(log_ret), "block": int(block)}
    mu, sigma = float(log_ret.mean()), float(log_ret.std(ddof=1))
    if model == "gbm":
        return {"mu": mu, "sigma": sigma}
    if model == "student_t":
        return {"mu": mu, "sigma": sigma, "nu": _student_t_df(log_ret - log_ret.mean())}
    fit = fit_garch(returns)
    return {
        "omega": fit.omega,
        "alpha": fit.alpha,
        "beta": fit.beta,
        "nu": fit.nu,
        "mu": mu,
        "long_run_variance": fit.long_run_variance,
        "persistence": fit.persistence,
        "loglik": fit.loglik,
    }


# --------------------------------------------------------------------------- #
# 主模拟
# --------------------------------------------------------------------------- #
def simulate(
    returns: pd.Series | np.ndarray,
    *,
    model: str = "bootstrap",
    n_paths: int = 4000,
    horizon_days: int = 1260,
    seed: int = 20260101,
    record_every: int = 21,
    chunk: int = 500,
    goal_annual: float = 0.08,
    deep_drawdown: float = -0.30,
    block: int = DEFAULT_BLOCK,
    convergence_sizes: Sequence[int] = (250, 500, 1000, 2000, 4000),
) -> SimulationResult:
    """模拟组合未来的净值路径分布。

    Parameters
    ----------
    horizon_days
        模拟长度（交易日）。默认 1260 ≈ 5 年。
    record_every
        每隔多少步记录一次分位（默认 21 ≈ 一个月）。完整记录 4000×1260 个浮点
        既占内存也没必要——扇形图按月画完全够用。
    goal_annual
        目标年化收益；达标概率 = 终值 ≥ 期初 × (1+goal)^年数 的路径占比。
    deep_drawdown
        深度回撤阈值；给出"途中触及该回撤"的概率（比只看终值更贴近真实体验）。
    """
    if n_paths <= 0 or horizon_days <= 0:
        raise ValueError("n_paths 与 horizon_days 必须为正")
    params = fit_model(returns, model, block=block)
    rng = np.random.default_rng(seed)

    record_steps = list(range(record_every, horizon_days + 1, record_every))
    if not record_steps or record_steps[-1] != horizon_days:
        record_steps.append(horizon_days)
    record_index = {step: position for position, step in enumerate(record_steps)}

    growth_records: list[np.ndarray] = []
    terminal = np.empty(n_paths, dtype=float)
    max_drawdown = np.empty(n_paths, dtype=float)
    convergence: list[dict[str, float]] = []
    pending_sizes = sorted(size for size in convergence_sizes if size <= n_paths)

    done = 0
    while done < n_paths:
        size = min(chunk, n_paths - done)
        block_returns = _simulate_block(model, params, size, horizon_days, rng)
        growth = np.cumprod(1.0 + block_returns, axis=1)
        running_max = np.maximum.accumulate(np.concatenate([np.ones((size, 1)), growth], axis=1), axis=1)
        drawdown = np.concatenate([np.ones((size, 1)), growth], axis=1) / running_max - 1.0
        growth_records.append(growth[:, [step - 1 for step in record_steps]])
        terminal[done : done + size] = growth[:, -1]
        max_drawdown[done : done + size] = drawdown.min(axis=1)
        done += size

        while pending_sizes and done >= pending_sizes[0]:
            used = pending_sizes.pop(0)
            sample = terminal[:done]
            convergence.append(
                {
                    "n_paths": float(used),
                    "mean_terminal": float(sample.mean()),
                    "standard_error": float(sample.std(ddof=1) / np.sqrt(used)),
                    "median_terminal": float(np.median(sample)),
                }
            )

    growth_all = np.concatenate(growth_records, axis=0)
    # 记录到 1% 分位：厚尾与正态的差别在 5% 处可能反而更小，
    # 只在 1% 及更深处才体现出来（见 tests 里对标准化 t 的检验）
    percentile_levels = (1, 5, 25, 50, 75, 95, 99)
    percentiles = {
        f"p{level}": [round(float(value), 6) for value in np.percentile(growth_all, level, axis=0)]
        for level in percentile_levels
    }

    years = horizon_days / TRADING_DAYS_PER_YEAR
    goal_multiple = float((1.0 + goal_annual) ** years)
    terminal_return = terminal - 1.0
    q05 = float(np.percentile(terminal_return, 5))
    var95 = -q05
    cvar95 = -float(terminal_return[terminal_return <= q05].mean())

    summary = {
        "mean_terminal": round(float(terminal.mean()), 6),
        "median_terminal": round(float(np.median(terminal)), 6),
        "std_terminal": round(float(terminal.std(ddof=1)), 6),
        "p1_terminal": round(float(np.percentile(terminal, 1)), 6),
        "p5_terminal": round(float(np.percentile(terminal, 5)), 6),
        "p95_terminal": round(float(np.percentile(terminal, 95)), 6),
        "p99_terminal": round(float(np.percentile(terminal, 99)), 6),
        "mean_cagr": round(float(np.median(terminal) ** (1.0 / years) - 1.0), 6),
        "var95_terminal": round(var95, 6),
        "cvar95_terminal": round(cvar95, 6),
        "median_max_drawdown": round(float(np.median(max_drawdown)), 6),
        "p95_max_drawdown": round(float(np.percentile(max_drawdown, 5)), 6),
    }
    counts, edges = np.histogram(terminal, bins=40)
    histogram = {
        "counts": [int(value) for value in counts],
        "edges": [round(float(value), 6) for value in edges],
    }
    risk = {
        "prob_loss": round(float((terminal < 1.0).mean()), 6),
        "prob_goal": round(float((terminal >= goal_multiple).mean()), 6),
        "goal_annual": goal_annual,
        "goal_multiple": round(goal_multiple, 6),
        "prob_deep_drawdown": round(float((max_drawdown <= deep_drawdown).mean()), 6),
        "deep_drawdown": deep_drawdown,
    }
    return SimulationResult(
        model=model,
        label=MODEL_LABELS.get(model, model),
        n_paths=n_paths,
        horizon_days=horizon_days,
        record_every=record_every,
        seed=seed,
        percentiles=percentiles,
        summary=summary,
        risk=risk,
        convergence=convergence,
        histogram=histogram,
        params={k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in params.items() if k != "sample"},
    )


def simulate_all_models(
    returns: pd.Series | np.ndarray,
    *,
    n_paths: int = 4000,
    horizon_days: int = 1260,
    seed: int = 20260101,
    goal_annual: float = 0.08,
    deep_drawdown: float = -0.30,
    models: Sequence[str] = MODELS,
) -> list[SimulationResult]:
    """同一组合在多个模型下各跑一次——**模型之间的差异本身就是结论**。"""
    out: list[SimulationResult] = []
    for offset, model in enumerate(models):
        try:
            out.append(
                simulate(
                    returns,
                    model=model,
                    n_paths=n_paths,
                    horizon_days=horizon_days,
                    seed=seed + offset,
                    goal_annual=goal_annual,
                    deep_drawdown=deep_drawdown,
                )
            )
        except Exception:  # noqa: BLE001 - 单个模型失败不应让整组结果消失
            continue
    return out


def model_comparison(results: Sequence[SimulationResult]) -> list[dict[str, Any]]:
    """把多模型结果并排成一张表，突出"结论对模型有多敏感"。"""
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "model": result.model,
                "label": result.label,
                "prob_goal": result.risk["prob_goal"],
                "prob_loss": result.risk["prob_loss"],
                "prob_deep_drawdown": result.risk["prob_deep_drawdown"],
                "median_terminal": result.summary["median_terminal"],
                "p5_terminal": result.summary["p5_terminal"],
                "p95_max_drawdown": result.summary["p95_max_drawdown"],
                "warning": result.warning,
            }
        )
    return rows
