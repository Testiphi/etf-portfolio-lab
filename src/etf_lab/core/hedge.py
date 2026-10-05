"""Delta-Gamma 复制与再平衡模拟。

这个模块回答一个很实际的问题
--------------------------
**如果我的 ETF 没有上市期权，只能靠调整仓位来"近似买保险"，那这个近似有多准、要花多少钱？**

做法是标准的自融资复制实验：目标是复制"持有标的 + 买入认沽"的到期收益，
但不买期权，而是持仓位的 delta 随标的与时间变化而调整标的仓位。
复制组合与目标收益之间的差额，就是**离散再平衡的误差**，它由 Gamma 驱动。

为什么这个实验值得做
--------------------
它有一个**闭式对照**：Boyle–Emanuel（1980）给出离散对冲误差
``Var ≈ Σ ½Γ²σ⁴S⁴Δt²``，于是

* 误差标准差 ∝ **√Δt**（再平衡越密，误差按平方根下降）
* 交易成本 ∝ **1/Δt**（再平衡越密，成本线性上升）

两者反向，因此**存在最优再平衡频率**。这不是拍脑袋的经验值，
而是两个幂律的交叉点——这也是本模块最想让人看到的结论。

诚实边界
--------
1. **没有期权行情**，认沽价格用 Black-Scholes + 历史波动率计算，
   所以这是"理论定价下的复制实验"，不是市场报价下的实测。
   真实世界里隐含波动率远高于历史波动率（恐慌时尤其），实际成本会更高。
2. 标的按**几何布朗运动**生成。真实收益有厚尾与波动聚集，
   因此**误差的尾部会被低估**；这一点在蒙特卡洛面板里已经单独量化过。
3. 成本模型只有按名义量计的线性交易成本，不含买卖价差随波动扩大、冲击成本等。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from etf_lab.core import derivatives

TRADING_DAYS = 252
DEFAULT_REBALANCE_GRID: tuple[int, ...] = (1, 5, 10, 21, 63, 126)
"""再平衡间隔（交易日）：从每天到约半年一次。"""


@dataclass(frozen=True)
class ReplicationPlan:
    """一次复制实验的设定。"""

    spot: float
    sigma: float
    r: float
    tenor_years: float = 1.0
    strike_ratio: float = 0.95
    target_ratio: float = 0.5
    """目标下行参与比例：1.0 = 完全不对冲，0.5 = 下跌只承担一半，0.0 = 完全对冲。"""
    rebalance_days: int = 21
    """再平衡间隔（交易日）。"""
    cost_bps: float = 5.0
    """单边交易成本（基点，按成交名义额计）。"""
    q: float = 0.0

    @property
    def strike(self) -> float:
        return self.spot * self.strike_ratio


@dataclass(frozen=True)
class ReplicationRun:
    """某个再平衡频率下的一组结果。"""

    rebalance_days: int
    error_mean: float
    error_std: float
    error_p5: float
    error_p95: float
    predicted_std: float
    mean_cost: float
    mean_trades: float
    n_paths: int

    @property
    def scale_ratio(self) -> float:
        """模拟误差标准差 ÷ 解析预期。应接近 1，否则说明公式常数或实现有问题。"""
        if self.predicted_std <= 0:
            return float("nan")
        return self.error_std / self.predicted_std

    @property
    def net_error_mean(self) -> float:
        """扣掉交易成本后的净结果：``毛误差均值 − 成本``。

        毛误差（``error_mean``）是纯粹复制偏差；``mean_cost`` 是确定付出的成本。
        两者必须分开报——混在一起会把成本重复计算一次，也会让误差的离散度
        随成本假设变化。
        """
        return self.error_mean - self.mean_cost

    @property
    def total_burden(self) -> float:
        """误差标准差 + 累计成本。

        **这是一个选择，不是定理**：把不可消除的复制误差与确定付出的交易成本
        按 1:1 相加，等价于"我既怕误差也怕成本，且同样怕"。换一个风险厌恶系数
        最优频率就会移动，所以页面必须把这个口径写出来。
        """
        return self.error_std + self.mean_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "rebalance_days": int(self.rebalance_days),
            "error_mean": round(float(self.error_mean), 8),
            "error_std": round(float(self.error_std), 8),
            "error_p5": round(float(self.error_p5), 8),
            "error_p95": round(float(self.error_p95), 8),
            "predicted_std": round(float(self.predicted_std), 8),
            "scale_ratio": None if not np.isfinite(self.scale_ratio) else round(float(self.scale_ratio), 4),
            "mean_cost": round(float(self.mean_cost), 8),
            "net_error_mean": round(float(self.net_error_mean), 8),
            "mean_trades": round(float(self.mean_trades), 3),
            "total_burden": round(float(self.total_burden), 8),
            "n_paths": int(self.n_paths),
        }


def cost_sensitivity(
    runs: Sequence[ReplicationRun],
    *,
    base_cost_bps: float,
    cost_grid: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 50.0),
) -> list[dict[str, Any]]:
    """最优再平衡频率对交易成本假设的敏感度。

    成本在模型里是**线性**的（交易路径不依赖成本水平，成本只从现金里扣），
    因此可以直接按比例缩放，不必重跑模拟——这一点在页面注释里要写明，
    否则看起来像"跑了五次"。

    结论通常不是"某个频率最优"，而是"**最优频率随成本水平移动**"：
    低成本下越密越好，成本一高就迅速偏向稀疏。把这一点藏起来只给一个数字，
    等于把最重要的信息删掉。
    """
    out: list[dict[str, Any]] = []
    for cost in cost_grid:
        scale = float(cost) / base_cost_bps if base_cost_bps > 0 else 1.0
        burdens = [run.error_std + run.mean_cost * scale for run in runs]
        best_index = int(np.argmin(burdens))
        out.append(
            {
                "cost_bps": round(float(cost), 4),
                "optimal_rebalance_days": int(runs[best_index].rebalance_days),
                "optimal_total_burden": round(float(burdens[best_index]), 8),
                "optimal_error_std": round(float(runs[best_index].error_std), 8),
                "optimal_mean_cost": round(float(runs[best_index].mean_cost * scale), 8),
            }
        )
    return out


@dataclass(frozen=True)
class ReplicationResult:
    plan: dict[str, Any]
    runs: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    premium: dict[str, Any] = field(default_factory=dict)
    taylor: list[dict[str, Any]] = field(default_factory=list)
    sensitivity: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 目标头寸
# --------------------------------------------------------------------------- #
def protection_units(plan: ReplicationPlan, units: float = 1.0) -> float:
    """要让总 delta 等于 ``target_ratio × units``，需要多少单位的认沽名义量。

    总 delta = units + 名义量 × Δ_put，令其等于 ``target_ratio × units``：
    ``名义量 = units × (target_ratio − 1) / Δ_put``。
    因为 Δ_put < 0 且 target_ratio ≤ 1，名义量为正（买入认沽）。
    """
    greeks = derivatives.bs_greeks(
        plan.spot, plan.strike, plan.tenor_years, plan.r, plan.sigma, "put", plan.q
    )
    if not np.isfinite(greeks.delta) or greeks.delta >= 0:
        raise ValueError(f"认沽 delta 异常：{greeks.delta}")
    return units * (plan.target_ratio - 1.0) / greeks.delta


def target_value(path: np.ndarray, plan: ReplicationPlan, units: float, notional: float) -> np.ndarray:
    """目标头寸（标的 + 认沽）在给定标的价格下的价值。"""
    terminal = np.maximum(plan.strike - path, 0.0)
    return units * path + notional * terminal


def initial_value(plan: ReplicationPlan, units: float, notional: float) -> float:
    greeks = derivatives.bs_greeks(
        plan.spot, plan.strike, plan.tenor_years, plan.r, plan.sigma, "put", plan.q
    )
    return units * plan.spot + notional * greeks.price


# --------------------------------------------------------------------------- #
# 路径生成
# --------------------------------------------------------------------------- #
def gbm_paths(
    spot: float,
    mu: float,
    sigma: float,
    tenor_years: float,
    n_paths: int,
    *,
    steps_per_year: int = TRADING_DAYS,
    seed: int = 20260101,
) -> np.ndarray:
    """生成几何布朗运动路径，形状 ``(n_paths, steps + 1)``（含起点）。"""
    if n_paths <= 0 or tenor_years <= 0:
        raise ValueError("n_paths 与 tenor_years 必须为正")
    steps = max(1, int(round(tenor_years * steps_per_year)))
    dt = tenor_years / steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return spot * np.exp(log_paths)


# --------------------------------------------------------------------------- #
# 复制引擎（向量化：按时间步推进，一次处理全部路径）
# --------------------------------------------------------------------------- #
def _replication_engine(
    paths: np.ndarray,
    plan: ReplicationPlan,
    units: float,
    notional: float,
    dt: float | None = None,
) -> dict[str, Any]:
    """自融资 delta 复制，向量化实现。

    **为什么必须向量化**：逐路径循环要在每条路径上调用 253 次标量 Black-Scholes，
    4000 条路径 × 一个频率就跑不完（实测 >10 分钟）。改成按时间步推进、
    一步处理全部路径后，同样的工作量降到亚秒级。

    复制组合：持有 ``units_held`` 单位标的 + 现金；每隔 ``rebalance_days`` 天把
    持仓调整到目标 delta。目标是到期复现 ``units·S_T + notional·max(K−S_T,0)``。

    解析预期按**每个再平衡区间累加一次**（``½Γ²σ⁴S⁴Δt²``）——
    写成每个交易日累加会让预测值放大 ``再平衡间隔`` 倍。
    """
    matrix = np.asarray(paths, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    n_paths, n_cols = matrix.shape
    steps = n_cols - 1
    if steps <= 0:
        raise ValueError("路径太短")
    step_dt = dt if dt is not None else plan.tenor_years / steps
    cost_rate = plan.cost_bps / 10000.0

    def desired(step: int) -> tuple[np.ndarray, np.ndarray]:
        remaining = max(plan.tenor_years - step * step_dt, 0.0)
        delta, gamma = derivatives.bs_delta_gamma_array(
            matrix[:, step], plan.strike, remaining, plan.r, plan.sigma, "put", plan.q
        )
        return units + notional * delta, notional * gamma

    initial = initial_value(plan, units, notional)
    # 复制组合从"**已经持有** units 单位标的"出发，只做 delta 调整。
    # 若从零买入，target_ratio=1（完全不对冲）也会被收一笔建仓成本——
    # 而那笔"买入自己已持有的 ETF"根本不是对冲成本。
    cash = np.full(n_paths, initial - units * matrix[:, 0], dtype=float)  # 预留的权利金
    total_cost = np.zeros(n_paths, dtype=float)
    trades = np.zeros(n_paths, dtype=float)
    predicted_variance = np.zeros(n_paths, dtype=float)

    units_held, _ = desired(0)
    traded = units_held - units
    cash -= traded * matrix[:, 0]
    cost = cost_rate * np.abs(traded * matrix[:, 0])
    cash -= cost
    total_cost += cost
    trades += (np.abs(traded) > 0).astype(float)

    step = 0
    while step < steps:
        interval = min(plan.rebalance_days, steps - step)
        _, gamma = desired(step)
        predicted_variance += (
            0.5 * gamma**2 * plan.sigma**4 * matrix[:, step] ** 4 * (interval * step_dt) ** 2
        )
        cash *= (1.0 + plan.r * step_dt) ** interval
        target_step = step + interval
        if target_step < steps:
            new_units, _ = desired(target_step)
            traded = new_units - units_held
            cash -= traded * matrix[:, target_step]
            cost = cost_rate * np.abs(traded * matrix[:, target_step])
            cash -= cost
            total_cost += cost
            trades += (np.abs(traded) > 0).astype(float)
            units_held = new_units
        step = target_step

    terminal = units_held * matrix[:, -1] + cash
    target = units * matrix[:, -1] + notional * np.maximum(plan.strike - matrix[:, -1], 0.0)
    net_error = terminal - target
    return {
        # 毛误差 = 把交易成本加回去：它才是**纯粹的复制误差**。
        # 若不加回去，成本会污染误差的离散度（成本逐路径不同），
        # 于是"误差标准差"会随成本假设变化——那既不直观也没意义。
        "error": net_error + total_cost,
        "net_error": net_error,
        "cost": total_cost,
        "trades": trades,
        "predicted_std": np.sqrt(predicted_variance),
        "initial": float(initial),
    }


def replicate_one(
    path: np.ndarray,
    plan: ReplicationPlan,
    *,
    units: float = 1.0,
    notional: float | None = None,
    dt: float | None = None,
) -> dict[str, float]:
    """在一条价格路径上做复制，返回误差与成本（引擎的单路径包装）。"""
    if notional is None:
        notional = protection_units(plan, units)
    outcome = _replication_engine(np.asarray(path, dtype=float), plan, units, notional, dt)
    return {
        "error": float(outcome["error"][0]),
        "cost": float(outcome["cost"][0]),
        "trades": float(outcome["trades"][0]),
        "predicted_std": float(outcome["predicted_std"][0]),
        "initial": float(outcome["initial"]),
    }


def run_replication(
    paths: np.ndarray,
    plan: ReplicationPlan,
    *,
    units: float = 1.0,
    notional: float | None = None,
) -> ReplicationRun:
    """在一组路径上跑同一个再平衡频率，汇总误差与成本。"""
    if notional is None:
        notional = protection_units(plan, units)
    outcome = _replication_engine(np.asarray(paths, dtype=float), plan, units, notional)
    initial = float(outcome["initial"])
    error_array = np.asarray(outcome["error"], dtype=float) / initial
    cost_array = np.asarray(outcome["cost"], dtype=float) / initial
    return ReplicationRun(
        rebalance_days=int(plan.rebalance_days),
        error_mean=float(error_array.mean()),
        error_std=float(error_array.std(ddof=1)) if len(error_array) > 1 else float("nan"),
        error_p5=float(np.percentile(error_array, 5)),
        error_p95=float(np.percentile(error_array, 95)),
        predicted_std=float(np.mean(np.asarray(outcome["predicted_std"], dtype=float) / initial)),
        mean_cost=float(cost_array.mean()),
        mean_trades=float(np.mean(outcome["trades"])),
        n_paths=int(len(error_array)),
    )


# --------------------------------------------------------------------------- #
# 泰勒分解：验证 Delta-Gamma-Theta 近似的适用边界
# --------------------------------------------------------------------------- #
def taylor_residuals(
    plan: ReplicationPlan,
    *,
    horizons: Sequence[int] = (1, 5, 21, 63),
    shocks: Sequence[float] = (0.01, 0.03, 0.10),
    samples: int = 4000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """把真实价格变化与 Delta-Gamma-Theta 近似比较，看二阶近似何时失效。

    对每个（步长, 冲击幅度）组合：抽取 ``dS``，比较
    ``ΔP_full = P(S+dS, T−Δt) − P(S,T)`` 与
    ``ΔP_泰勒 = Δ·dS + ½Γ·dS² + Θ·Δt``，报告残差占期权价格的比例。
    """
    rng = np.random.default_rng(seed)
    base = derivatives.bs_greeks(plan.spot, plan.strike, plan.tenor_years, plan.r, plan.sigma, "put", plan.q)
    out: list[dict[str, Any]] = []
    for days in horizons:
        dt_years = days / TRADING_DAYS
        remaining = plan.tenor_years - dt_years
        if remaining <= 0:
            continue
        for shock in shocks:
            level = plan.spot * shock
            draws = rng.normal(0.0, 1.0, samples) * level
            moved = plan.spot + draws
            mask = moved > 0
            good = moved[mask]
            if len(good) == 0:
                continue
            # 向量化重估：标量循环在这里要跑 4.8 万次，是构建耗时的大头
            full = derivatives.bs_price_array(good, plan.strike, remaining, plan.r, plan.sigma, "put", plan.q) - base.price
            approx = base.delta * draws[mask] + 0.5 * base.gamma * draws[mask] ** 2 + base.theta * days
            residual = full - approx
            out.append(
                {
                    "days": int(days),
                    "shock": float(shock),
                    "mean_abs_residual_pct": float(np.mean(np.abs(residual)) / base.price),
                    "max_abs_residual_pct": float(np.max(np.abs(residual)) / base.price),
                    "mean_abs_move_pct": float(np.mean(np.abs(full)) / base.price),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# 主分析
# --------------------------------------------------------------------------- #
def analyse(
    plan: ReplicationPlan,
    *,
    n_paths: int = 4000,
    rebalance_grid: Sequence[int] = DEFAULT_REBALANCE_GRID,
    units: float = 1.0,
    mu: float | None = None,
    seed: int = 20260101,
) -> ReplicationResult:
    """跑完整分析：频率权衡、解析校验、与直接买认沽的成本对比、泰勒边界。"""
    drift = plan.r if mu is None else mu
    paths = gbm_paths(plan.spot, drift, plan.sigma, plan.tenor_years, n_paths, seed=seed)
    notional = protection_units(plan, units)

    runs: list[ReplicationRun] = []
    for days in rebalance_grid:
        step_plan = ReplicationPlan(**{**plan.__dict__, "rebalance_days": int(days)})
        runs.append(run_replication(paths, step_plan, units=units, notional=notional))

    best = min(runs, key=lambda run: run.total_burden)
    premium = derivatives.bs_greeks(
        plan.spot, plan.strike, plan.tenor_years, plan.r, plan.sigma, "put", plan.q
    )
    premium_ratio = notional * premium.price / (units * plan.spot)

    note = (
        f"目标：让组合的初始 delta 等于标的仓位的 {plan.target_ratio:.0%}"
        f"（即下跌只承担 {plan.target_ratio:.0%}），行权价为期初的 {plan.strike_ratio:.0%}，"
        f"期限 {plan.tenor_years:.2f} 年。标的按几何布朗运动生成、波动率取 {plan.sigma:.1%}，"
        "认沽按 Black-Scholes 理论定价（**没有期权行情**，实际市场成本会更高）。"
        "「总负担」= 复制误差标准差 + 累计交易成本，这个 1:1 权重是人为选择，不是定理。"
    )
    return ReplicationResult(
        plan={
            "spot": round(float(plan.spot), 6),
            "sigma": round(float(plan.sigma), 6),
            "r": round(float(plan.r), 6),
            "tenor_years": round(float(plan.tenor_years), 4),
            "strike": round(float(plan.strike), 6),
            "strike_ratio": round(float(plan.strike_ratio), 4),
            "target_ratio": round(float(plan.target_ratio), 4),
            "cost_bps": round(float(plan.cost_bps), 4),
            "notional_units": round(float(notional), 6),
            "n_paths": int(n_paths),
            "seed": int(seed),
        },
        runs=[run.as_dict() for run in runs],
        best=best.as_dict(),
        premium={
            "bs_price_ratio": round(float(premium.price / plan.spot), 6),
            "premium_pct_of_portfolio": round(float(premium_ratio), 6),
            "delta": round(float(premium.delta), 6),
            "gamma": round(float(premium.gamma), 8),
            "vega": round(float(premium.vega), 8),
            "note": "直接买入认沽的权利金（一次性、确定付出）vs 动态复制的交易成本（持续、随频率上升）",
        },
        taylor=taylor_residuals(plan, seed=seed),
        sensitivity=cost_sensitivity(runs, base_cost_bps=plan.cost_bps),
        note=note,
    )
