"""利率与久期：把收益率曲线变成可用的风险参数。

两件事
------
1. **无风险利率不再靠假设**。夏普 / 索提诺 / 卡玛都要减无风险利率，
   此前页面写死 2%——那是个不诚实的简化。现在从收益率曲线取。
2. **久期用回归反推**。A 股场内投资者看不到债券 ETF 的持仓与久期，
   但可以看它的**收益对收益率变动的敏感度**：``r ≈ α − D·Δy``，
   回归系数乘回去就是修正久期。这是数据可得条件下的正确做法，
   也顺带给出 R²——R² 低就说明这只 ETF 的价格不是被利率驱动的，
   此时久期数字没有意义（必须先看 R²，再看久期）。

单位约定（混用单位是这类计算最常见的错误）
------------------------------------------
- 收益率变动以**基点（bp）** 为单位进入回归
- 修正久期 ``D`` 的定义是 ``ΔP/P = −D × Δy``，其中 ``Δy`` 用小数
- 因此若回归系数是"每 1bp 的价格变动"，则 ``D = −β × 10000``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_FALLBACK_RF = 0.02
"""没有曲线数据时的兜底假设；页面上会明确标注这是假设值。"""
RISK_FREE_TENOR = "CN1Y"
"""无风险利率默认用 1 年期国债——本数据源里最短的期限，
也是夏普比率常用的短期无风险代理。严格做法应用 3 个月国债或 Shibor。"""
CANDIDATE_TENORS: tuple[str, ...] = ("CN3Y", "CN5Y", "CN7Y", "CN10Y")
"""久期回归的候选期限：债券 ETF 的久期通常落在 3~10 年区间。"""
MIN_OBS = 120


@dataclass(frozen=True)
class RateEnvironment:
    """当前利率环境。"""

    as_of: str
    risk_free: float
    source: str
    """``curve`` = 来自收益率曲线；``assumption`` = 数据缺失时的兜底假设。"""
    tenor_used: str
    curve: dict[str, float]
    curve_last_year: dict[str, float]
    slope_10y_1y: float | None
    """10 年减 1 年的期限利差——曲线陡峭度的常用度量。"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "risk_free": round(float(self.risk_free), 6),
            "source": self.source,
            "tenor_used": self.tenor_used,
            "curve": {k: round(float(v), 6) for k, v in self.curve.items()},
            "curve_last_year": {k: round(float(v), 6) for k, v in self.curve_last_year.items()},
            "slope_10y_1y": None if self.slope_10y_1y is None else round(float(self.slope_10y_1y), 6),
        }


@dataclass(frozen=True)
class DurationEstimate:
    """一只债券标的的久期估计。"""

    symbol: str
    code: str
    """选中的收益率期限。"""
    duration: float
    """修正久期（年）。"""
    beta_per_bp: float
    r_squared: float
    n_obs: int
    candidates: dict[str, float] = field(default_factory=dict)
    """各候选期限的 R²，供人工核对"到底该对哪条曲线"。"""

    @property
    def reliable(self) -> bool:
        return self.r_squared >= 0.2 and self.n_obs >= MIN_OBS

    @property
    def warning(self) -> str | None:
        if self.n_obs < MIN_OBS:
            return f"样本仅 {self.n_obs} 天，久期估计不稳定"
        if self.r_squared < 0.2:
            return f"R² 仅 {self.r_squared:.2f}：该标的价格主要不是被利率驱动的，久期数字参考价值有限"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "code": self.code,
            "duration": round(float(self.duration), 4),
            "beta_per_bp": round(float(self.beta_per_bp), 8),
            "r_squared": round(float(self.r_squared), 6),
            "n_obs": int(self.n_obs),
            "candidates": {k: round(float(v), 6) for k, v in self.candidates.items()},
            "reliable": self.reliable,
            "warning": self.warning,
        }


def rate_environment(
    curve: pd.DataFrame,
    *,
    tenor: str = RISK_FREE_TENOR,
    fallback: float = DEFAULT_FALLBACK_RF,
    lookback_days: int = 15,
) -> RateEnvironment:
    """从收益率曲线取当前无风险利率与曲线形态。

    **单位约定**：数据源以**百分数**存储（``1.6822`` 表示 1.68%），
    而本函数一律返回**小数**（``0.016822``）——因为下游要用它去减收益率、
    做复利折算，混用单位会把无风险利率放大 100 倍（实测踩过：
    rf 变成 1.22 相当于每年 122%，所有夏普数字随之失效）。

    Parameters
    ----------
    lookback_days
        取最近这些天内最后一个有效值——避免因为"今天还没更新"就退回假设值。
    """
    if curve is None or curve.empty:
        return RateEnvironment(
            as_of="—",
            risk_free=fallback,
            source="assumption",
            tenor_used=tenor,
            curve={},
            curve_last_year={},
            slope_10y_1y=None,
        )

    frame = curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["yield"] = pd.to_numeric(frame["yield"], errors="coerce") / 100.0  # 百分数 → 小数
    latest_date = frame["date"].max()
    recent = frame[frame["date"] >= latest_date - pd.Timedelta(days=lookback_days)]

    pivot = frame.pivot_table(index="date", columns="code", values="yield")
    recent_pivot = recent.pivot_table(index="date", columns="code", values="yield").dropna(how="all")
    if recent_pivot.empty:
        current: dict[str, float] = {}
    else:
        # 必须**逐列向前取值**再取最后一行：如果最新那天某个期限还没更新（NaN），
        # 直接取 last row 会让整行空掉，于是无风险利率莫名其妙退回假设值。
        current = {str(code): float(value) for code, value in recent_pivot.ffill().iloc[-1].dropna().items()}

    year_ago = pivot[pivot.index <= latest_date - pd.Timedelta(days=365)]
    last_year = {str(code): float(value) for code, value in (year_ago.iloc[-1].dropna().items() if not year_ago.empty else [])}

    risk_free = current.get(tenor)
    source = "curve" if risk_free is not None else "assumption"
    if risk_free is None:
        risk_free = fallback

    slope = None
    if "CN10Y" in current and "CN1Y" in current:
        slope = current["CN10Y"] - current["CN1Y"]

    return RateEnvironment(
        as_of=str(latest_date.date()),
        risk_free=float(risk_free),
        source=source,
        tenor_used=tenor,
        curve=current,
        curve_last_year=last_year,
        slope_10y_1y=slope,
    )


def yield_changes_bp(curve: pd.DataFrame, code: str) -> pd.Series:
    """某个期限的日度收益率变动（基点）。"""
    frame = curve[curve["code"] == code].copy()
    if frame.empty:
        return pd.Series(dtype=float, name=f"{code}_change_bp")
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["yield"].sort_index()
    changes = series.diff() * 100.0  # 1% = 100bp
    changes.name = f"{code}_change_bp"
    return changes.dropna()


def estimate_duration(
    symbol: str,
    price_returns: pd.Series,
    curve: pd.DataFrame,
    *,
    codes: Sequence[str] = CANDIDATE_TENORS,
    min_obs: int = MIN_OBS,
) -> DurationEstimate | None:
    """用收益对收益率变动的回归反推修正久期；样本不足时返回 ``None``。

    会在候选期限里挑 |相关系数| 最大的那一条，并把全部候选的 R² 一并返回——
    "该对哪条曲线"本身是需要人工判断的事，不该藏在代码里。
    """
    returns = price_returns.dropna()
    if returns.empty:
        return None
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    best: DurationEstimate | None = None
    candidates: dict[str, float] = {}
    for code in codes:
        changes = yield_changes_bp(curve, code)
        if changes.empty:
            continue
        joined = pd.concat([returns.rename("r"), changes.rename("dy")], axis=1, join="inner").dropna()
        if len(joined) < min_obs:
            candidates[code] = float("nan")
            continue
        x = joined["dy"].to_numpy(dtype=float)
        y = joined["r"].to_numpy(dtype=float)
        variance = float(np.var(x, ddof=1))
        if variance == 0:
            continue
        beta = float(np.cov(x, y, ddof=1)[0, 1] / variance)
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        r_squared = correlation**2 if np.isfinite(correlation) else float("nan")
        candidates[code] = r_squared
        estimate = DurationEstimate(
            symbol=symbol,
            code=code,
            # β 的单位是"每 1bp 的价格变动比例"，因此 D = −β × 10000
            duration=-beta * 10000.0,
            beta_per_bp=beta,
            r_squared=r_squared,
            n_obs=int(len(joined)),
            candidates={},
        )
        if best is None or (np.isfinite(r_squared) and r_squared > (best.r_squared if np.isfinite(best.r_squared) else -1)):
            best = estimate

    if best is None:
        return None
    return DurationEstimate(
        symbol=best.symbol,
        code=best.code,
        duration=best.duration,
        beta_per_bp=best.beta_per_bp,
        r_squared=best.r_squared,
        n_obs=best.n_obs,
        candidates={k: v for k, v in candidates.items() if np.isfinite(v)},
    )


def rate_scenarios(
    durations: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    shocks_bp: Sequence[float] = (25, 50, 100, -25, -50, -100),
) -> list[dict[str, Any]]:
    """利率冲击情景表：收益率平行移动 X 个基点时，组合受多大影响。

    只对**有久期估计的标的**求和；其余资产的利率敏感性不建模（页面上要写明）。
    """
    rows: list[dict[str, Any]] = []
    bond_weight = float(sum(weights.get(symbol, 0.0) for symbol in durations))
    for shock in shocks_bp:
        per_holding = {
            symbol: float(weights.get(symbol, 0.0)) * (-float(duration)) * (float(shock) / 10000.0)
            for symbol, duration in durations.items()
        }
        total = float(sum(per_holding.values()))
        rows.append(
            {
                "shock_bp": float(shock),
                "portfolio_impact": total,
                "bond_sleeve_impact": (total / bond_weight) if bond_weight > 0 else None,
                "bond_weight": bond_weight,
                "per_holding": {k: round(v, 8) for k, v in per_holding.items()},
            }
        )
    return rows
