"""定投（DCA）模拟与 XIRR：纯函数，无 I/O。

为什么定投要用 XIRR 而不是简单的"总收益 ÷ 投入本金"
----------------------------------------------------
定投的每一笔钱在场时间不同，用总收益除以总投入会**系统性高估**年化收益
（后期投入的钱几乎没参与上涨却摊薄了分母）。XIRR 按每笔现金流的实际
持有天数折现，才是定投的真实年化。这是本站最重要的教学点之一。

支持的模式
----------
- ``fixed``：每期固定金额；
- ``value_avg``：价值平均（目标市值路径 − 当前市值 = 本期投入）；
- ``target_vol``：目标波动率（按已实现波动反向缩放投入金额）；
- ``take_profit``：定投 + 达标止盈（触发后卖出部分份额转现金）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from scipy import optimize

from etf_lab.core import metrics

TRADING_DAYS_PER_YEAR = 252

MODES: tuple[str, ...] = ("fixed", "value_avg", "target_vol", "take_profit")
"""引擎支持的全部定投模式。**界面必须从这里取**，别再手写子集——
``compute_preset`` 曾硬编码只跑前两种，于是实验室里选了别的模式也毫无效果。"""
MODE_LABELS: dict[str, str] = {
    "fixed": "固定金额",
    "value_avg": "价值平均",
    "target_vol": "目标波动率",
    "take_profit": "达标止盈",
}


# --------------------------------------------------------------------------- #
# 参数对象
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostModel:
    """交易成本。ETF 免印花税，所以这里没有印花税项；佣金通常有最低值。"""

    commission_rate: float = 0.00025
    """佣金费率，默认万分之 2.5。"""
    min_commission: float = 0.0
    slippage_bps: float = 2.0
    """买卖价差/冲击成本，单位基点（1 bp = 0.01%）。"""
    lot_size: int = 100
    """ETF 场内最小交易单位（1 手 = 100 份）。"""

    def commission(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(amount * self.commission_rate, self.min_commission)


@dataclass(frozen=True)
class DcaPlan:
    """定投计划。

    ``day`` 的含义随频率变化：月度/季度表示"当月的第几个交易日"（1 起），
    周度表示"星期几"（0 = 周一）；为 None 时取该周期首个交易日。
    """

    amount: float
    freq: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"
    day: int | None = None
    mode: Literal["fixed", "value_avg", "target_vol", "take_profit"] = "fixed"
    params: Mapping[str, float] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("每期投入金额必须为正")
        if self.freq not in ("daily", "weekly", "monthly", "quarterly"):
            raise ValueError(f"未知频率：{self.freq!r}")
        if self.mode not in ("fixed", "value_avg", "target_vol", "take_profit"):
            raise ValueError(f"未知模式：{self.mode!r}")


@dataclass(frozen=True)
class DcaResult:
    """一次定投模拟的完整结果。"""

    value: pd.Series
    """组合每日市值（现金 + 持仓市值）。"""
    shares: pd.Series
    cash: pd.Series
    invested_curve: pd.Series
    """累计投入本金曲线——它的形状就是"定投"与"一次性买入"的差别所在。"""
    trades: pd.DataFrame
    invested_total: float
    final_value: float
    xirr: float
    metrics: Mapping[str, float | int | str | None]


# --------------------------------------------------------------------------- #
# XIRR
# --------------------------------------------------------------------------- #
def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    """按实际天数折现的净现值。"""
    if rate <= -1.0:
        return float("inf")
    t0 = cashflows[0][0]
    total = 0.0
    for date, amount in cashflows:
        years = (date - t0).days / 365.0
        total += amount / (1.0 + rate) ** years
    return total


def xirr(cashflows: list[tuple[pd.Timestamp, float]], guess: float = 0.1) -> float:
    """用实际天数求解内部收益率。

    先网格扫描找到符号变化区间，再用 Brent 法求根；找不到区间时退回 Newton 法。
    现金流必须同时包含正负值，否则无解。

    Returns
    -------
    float
        年化内部收益率；无解时返回 ``nan``（不抛异常，因为"无解"本身是有效结论）。
    """
    if len(cashflows) < 2:
        raise ValueError("至少需要两笔现金流")
    flows = sorted(cashflows, key=lambda item: item[0])
    amounts = [amount for _, amount in flows]
    if not (any(a > 0 for a in amounts) and any(a < 0 for a in amounts)):
        return float("nan")

    def f(rate: float) -> float:
        return xnpv(rate, flows)

    grid = np.concatenate([np.linspace(-0.9999, -0.5, 60), np.linspace(-0.5, 1.0, 300), np.linspace(1.0, 20.0, 200)])
    previous_rate = float(grid[0])
    previous_value = f(previous_rate)
    for rate in grid[1:]:
        rate = float(rate)
        value = f(rate)
        if np.isfinite(previous_value) and np.isfinite(value) and previous_value * value <= 0:
            try:
                return float(optimize.brentq(f, previous_rate, rate, xtol=1e-12, rtol=1e-12, maxiter=200))
            except (ValueError, RuntimeError):
                break
        previous_rate, previous_value = rate, value

    try:
        root = optimize.newton(f, guess, maxiter=200, tol=1e-10)
        return float(root) if np.isfinite(root) and root > -1.0 else float("nan")
    except (RuntimeError, ValueError, OverflowError):
        return float("nan")


# --------------------------------------------------------------------------- #
# 定投日期
# --------------------------------------------------------------------------- #
def contribution_dates(index: pd.DatetimeIndex, plan: DcaPlan) -> list[pd.Timestamp]:
    """按计划的频率与日期规则，从交易日索引中挑出定投日。"""
    if len(index) == 0:
        return []
    if plan.freq == "daily":
        return list(index)

    if plan.freq == "monthly":
        periods = index.to_period("M")
    elif plan.freq == "quarterly":
        periods = index.to_period("Q")
    else:  # weekly
        periods = index.to_period("W")

    frame = pd.DataFrame({"pos": np.arange(len(index)), "period": periods}, index=index)
    picked: list[pd.Timestamp] = []
    for _, group in frame.groupby("period", sort=True):
        if plan.freq == "weekly" and plan.day is not None:
            weekday = int(plan.day)
            match = [ts for ts in group.index if ts.weekday() == weekday]
            chosen = match[0] if match else (group.index[0] if weekday >= group.index[0].weekday() else None)
            if chosen is None:
                continue
        elif plan.day is not None:
            offset = min(int(plan.day) - 1, len(group) - 1)
            chosen = group.index[offset]
        else:
            chosen = group.index[0]
        picked.append(chosen)
    return sorted(picked)


# --------------------------------------------------------------------------- #
# 模拟
# --------------------------------------------------------------------------- #
def _buy(shares: float, cash: float, amount: float, exec_price: float, cost: CostModel) -> tuple[float, float, float, float]:
    """用给定金额买入，返回 ``(新增份额, 剩余现金, 成交金额, 手续费)``。

    先按毛额估算可买手数，再剔除"成交金额 + 佣金超过可用金额"的手数——
    保证不会超支，也不会出现零股。
    """
    lot = cost.lot_size
    lots = int((amount / exec_price) // lot)
    while lots > 0:
        gross = lots * lot * exec_price
        if gross + cost.commission(gross) <= amount + 1e-9:
            break
        lots -= 1
    if lots <= 0:
        return 0.0, cash + amount, 0.0, 0.0
    gross = lots * lot * exec_price
    fee = cost.commission(gross)
    return float(lots * lot), cash + amount - gross - fee, gross, fee


def _sell(shares: float, cash: float, fraction: float, exec_price: float, cost: CostModel) -> tuple[float, float, float, float]:
    """按比例卖出，返回 ``(卖出份额, 剩余现金, 成交金额, 手续费)``。"""
    lot = cost.lot_size
    lots = int((shares * fraction) // lot)
    if lots <= 0:
        return 0.0, cash, 0.0, 0.0
    sold = float(lots * lot)
    gross = sold * exec_price
    fee = cost.commission(gross)
    return sold, cash + gross - fee, gross, fee


def simulate(
    plan: DcaPlan,
    prices: pd.Series,
    cost: CostModel | None = None,
    rf_annual: float = 0.0,
) -> DcaResult:
    """按日推进的定投模拟。

    Parameters
    ----------
    plan
        定投计划。
    prices
        **已完成复权**的 ETF 价格/净值序列，升序 ``DatetimeIndex``。
        现金部分不计息（这是一个明确的简化，需在页面标注）。
    cost
        交易成本模型。

    Returns
    -------
    DcaResult
        含每日市值、份额、现金、累计投入、交易明细与 XIRR。

    Notes
    -----
    顺序假设：**先记定投，再按当日收盘价成交，最后按收盘价估值**。
    所有成交价都使用当日收盘价并叠加滑点，不使用未来信息。
    """
    cost = cost or CostModel()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices 的索引必须是 DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices 的索引必须升序")
    if prices.isna().any():
        raise ValueError("prices 中存在缺失值，请先显式处理（本项目不允许静默填充）")
    if (prices <= 0).any():
        raise ValueError("prices 必须为正")

    price = prices.astype(float)
    contrib = set(contribution_dates(price.index, plan))
    params = dict(plan.params)

    return_window = int(params.get("return_window", 60))
    target_vol = float(params.get("target_vol", 0.15))
    min_mult = float(params.get("min_mult", 0.25))
    max_mult = float(params.get("max_mult", 2.0))
    growth = float(params.get("growth", 0.002))
    allow_sell = bool(params.get("allow_sell", False))
    tp_threshold = float(params.get("take_profit", 0.30))
    tp_fraction = float(params.get("take_fraction", 0.5))
    cooldown_days = int(params.get("cooldown_days", 20))

    simple_returns = price.pct_change()

    shares = 0.0
    cash = 0.0
    n_contrib = 0
    invested_total = 0.0
    last_sell_date: pd.Timestamp | None = None

    cashflows: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []
    values: list[float] = []
    share_series: list[float] = []
    cash_series: list[float] = []
    invested_series: list[float] = []

    for i, (date, px) in enumerate(price.items()):
        if date in contrib:
            amount = plan.amount
            current_value = cash + shares * float(px)

            if plan.mode == "value_avg":
                n_contrib_target = n_contrib + 1
                target = plan.amount * n_contrib_target * (1.0 + growth) ** (n_contrib_target - 1)
                desired = target - current_value
                amount = float(np.clip(desired, -max_mult * plan.amount, max_mult * plan.amount))
                if amount < 0 and not allow_sell:
                    amount = 0.0
            elif plan.mode == "target_vol":
                window = simple_returns.iloc[max(0, i - return_window) : i].dropna()
                realized = float(window.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(window) > 5 else target_vol
                scale = target_vol / realized if realized > 0 else 1.0
                amount = plan.amount * float(np.clip(scale, min_mult, max_mult))
            elif plan.mode == "take_profit":
                amount = plan.amount
            n_contrib += 1

            if amount > 0:
                bought, cash, gross, fee = _buy(shares, cash, amount, float(px) * (1.0 + cost.slippage_bps / 1e4), cost)
                shares += bought
                invested_total += amount
                cashflows.append((date, -amount))
                trades.append(
                    {
                        "date": date,
                        "action": "buy",
                        "amount": amount,
                        "shares": bought,
                        "price": float(px),
                        "gross": gross,
                        "fee": fee,
                    }
                )
            elif amount < 0 and allow_sell:
                sold, cash, gross, fee = _sell(shares, cash, 1.0, float(px) * (1.0 - cost.slippage_bps / 1e4), cost)
                sale_amount = min(-amount, gross)
                # 只卖出本次所需金额对应的份额，多余部分不卖
                if sold > 0:
                    ratio = sale_amount / gross if gross > 0 else 0.0
                    lots = int((sold * ratio) // cost.lot_size)
                    if lots > 0:
                        sold_final = float(lots * cost.lot_size)
                        gross_final = sold_final * float(px) * (1.0 - cost.slippage_bps / 1e4)
                        fee_final = cost.commission(gross_final)
                        shares -= sold_final
                        cash += gross_final - fee_final
                        trades.append(
                            {
                                "date": date,
                                "action": "sell",
                                "amount": -(gross_final - fee_final),
                                "shares": -sold_final,
                                "price": float(px),
                                "gross": gross_final,
                                "fee": fee_final,
                            }
                        )

            # 止盈规则（与 mode 无关，只要参数给出就走）
            if plan.mode == "take_profit" or params.get("take_profit") is not None:
                value_now = cash + shares * float(px)
                gain = (value_now - invested_total) / invested_total if invested_total > 0 else 0.0
                on_cooldown = last_sell_date is not None and (date - last_sell_date).days < cooldown_days
                if gain >= tp_threshold and not on_cooldown and shares > 0:
                    sold, cash, gross, fee = _sell(shares, cash, tp_fraction, float(px) * (1.0 - cost.slippage_bps / 1e4), cost)
                    if sold > 0:
                        shares -= sold
                        last_sell_date = date
                        trades.append(
                            {
                                "date": date,
                                "action": "take_profit",
                                "amount": -(gross - fee),
                                "shares": -sold,
                                "price": float(px),
                                "gross": gross,
                                "fee": fee,
                            }
                        )

        values.append(cash + shares * float(px))
        share_series.append(shares)
        cash_series.append(cash)
        invested_series.append(invested_total)

    value_curve = pd.Series(values, index=price.index, name="value")
    final_value = float(value_curve.iloc[-1])
    cashflows.append((price.index[-1], final_value))

    result = DcaResult(
        value=value_curve,
        shares=pd.Series(share_series, index=price.index, name="shares"),
        cash=pd.Series(cash_series, index=price.index, name="cash"),
        invested_curve=pd.Series(invested_series, index=price.index, name="invested"),
        trades=pd.DataFrame(trades),
        invested_total=float(invested_total),
        final_value=final_value,
        xirr=xirr(cashflows),
        metrics={},
    )

    stats = metrics.summary(value_curve, rf_annual=rf_annual) if len(value_curve) >= 2 else {}
    stats.update(
        {
            "invested_total": result.invested_total,
            "final_value": final_value,
            "profit": final_value - result.invested_total,
            "simple_return_on_invested": (final_value / result.invested_total - 1.0) if result.invested_total > 0 else float("nan"),
            "xirr": result.xirr,
            "n_contributions": n_contrib,
            "total_fees": float(result.trades["fee"].sum()) if not result.trades.empty else 0.0,
        }
    )
    return DcaResult(
        value=result.value,
        shares=result.shares,
        cash=result.cash,
        invested_curve=result.invested_curve,
        trades=result.trades,
        invested_total=result.invested_total,
        final_value=result.final_value,
        xirr=result.xirr,
        metrics=stats,
    )
