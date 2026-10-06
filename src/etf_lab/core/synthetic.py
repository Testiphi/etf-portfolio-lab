"""现金与外币现金：把"不生息的仓位"与"外币仓位"做成可参与组合计算的合成资产。

为什么做成"价格序列"而不是到处加特例
------------------------------------
组合计算链（收益 → 净值 → 指标 → 归因 → 蒙特卡洛 → 复制实验）全部消费**价格面板**。
把现金做成一条价格序列，它就能原样流过整条链，不必在任何一处写 `if symbol == "CASH"`。
本模块只负责"造出这条价格序列"，其余一概不管。

口径说明（都会显示在页面上）
----------------------------
* **人民币现金**：按国债收益率曲线的短端（默认 1 年期）逐日计息。
  曲线按交易日发布，遇到没有发布的日子沿用最近一次利率——这不是"填充缺失数据"，
  而是计息工具本身的定义：利率在下次调整前不变。
* **美元现金（人民币视角）**：收益 = 汇率变动 + 可选的美元利率假设。
  **默认不含美元利息**：能拿到的美债利率历史只有近 4 年，不足以支撑跨周期计息；
  与其用假设值冒充历史，不如不叠加，并明确告知这会**低估**持有美元的收益。
  需要计入的用户可以自己填一个年化假设——把假设交回给人的手上，而不是藏在默认值里。

**必须注意**：含现金会使样本区间收缩到"现金数据可用"的起点之后
（人民币现金的利率从 2015 年起、美元汇率从 2012 年起）。
本模块不填充、不插值，而是由调用方收缩区间并把新的起点显示出来。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASH_SYMBOL = "CASH"
USD_SYMBOL = "USD"
PERIODS_PER_YEAR = 252

SYNTHETIC_ASSETS: dict[str, dict[str, str]] = {
    CASH_SYMBOL: {"name": "人民币现金", "asset_class": "cash", "note": "按国债曲线短端计息，无波动"},
    USD_SYMBOL: {"name": "美元现金", "asset_class": "fx_cash", "note": "汇率变动 +（可选）美元利率假设"},
}


class SyntheticError(ValueError):
    """合成资产无法构造（通常是数据区间不覆盖）。"""


def _align_to_index(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """把低频序列对齐到价格日：用最近一次已发布的值（前向取值）。

    这里的前向取值是**工具定义**，不是数据清洗：利率与汇率在下次发布前保持不变。
    但区间起点早于数据时**不猜**——直接报错，由调用方收缩区间。
    """
    if values.empty:
        raise SyntheticError("数据为空，无法构造合成资产")
    merged = values.reindex(values.index.union(index)).ffill().reindex(index)
    if bool(merged.isna().to_numpy().any()):
        raise SyntheticError(
            f"区间起点早于数据起点（数据从 {values.index.min().date()} 开始），"
            "无法外推——请收缩样本区间"
        )
    return merged


def cash_price_series(
    curve: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    tenor: str = "CN1Y",
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """人民币现金的价格序列：按短端利率逐日累计。

    现金的"价格"就是"一元钱连本带息长到多少"，因此它单调、无回撤、无波动。
    它的作用是**降低组合波动**，而不是提供超额收益——这一点必须在页面上说清。
    """
    if curve is None or curve.empty:
        raise SyntheticError("没有国债收益率曲线数据，无法构造人民币现金")
    subset = curve[curve["code"] == tenor].copy()
    if subset.empty:
        raise SyntheticError(f"收益率曲线里没有 {tenor}，无法构造人民币现金")
    subset["date"] = pd.to_datetime(subset["date"])
    rates = subset.set_index("date")["yield"].sort_index() / 100.0
    aligned = _align_to_index(rates, index)
    daily = (1.0 + aligned) ** (1.0 / periods_per_year) - 1.0
    price = (1.0 + daily).cumprod()
    price.name = CASH_SYMBOL
    return price


def usd_price_series(
    fx: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    pair: str = "USDCNY",
    annual_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """美元现金（人民币视角）的价格序列 = 汇率变动 ×（1 + 年化利率）的累计。

    ``annual_rate=0`` 表示**不生息**（默认）——因为美元利率的历史数据不足，
    叠加一个当前的利率假设会系统性高估 2012–2021 年（那时美元利率接近 0）。
    """
    if fx is None or fx.empty:
        raise SyntheticError("没有汇率数据，无法构造美元现金")
    subset = fx[fx["pair"] == pair].copy()
    if subset.empty:
        raise SyntheticError(f"汇率表里没有 {pair}，无法构造美元现金")
    subset["date"] = pd.to_datetime(subset["date"])
    rates = subset.set_index("date")["close"].sort_index()
    aligned = _align_to_index(rates, index)
    growth = aligned / float(aligned.iloc[0])
    steps = np.arange(len(index), dtype=float)
    interest = (1.0 + float(annual_rate)) ** (steps / periods_per_year)
    price = growth * interest
    price.name = USD_SYMBOL
    return price


def required_start(
    symbols: list[str],
    *,
    curve: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
    cash_tenor: str = "CN1Y",
) -> pd.Timestamp | None:
    """构造这些合成资产所需的最早日期。

    用途是让调用方**把样本区间收缩到数据可用处**，而不是等构造时抛错。
    含现金会把样本推到利率曲线的起点（本数据源是 2015 年），
    含美元推到汇率起点（2012 年）——这是必须显示给用户的事实，不是内部细节。

    缺数据时**明确报错**并给出可执行的下一步，而不是静默降级成假设值。
    """
    starts: list[pd.Timestamp] = []
    for symbol in symbols:
        if symbol == CASH_SYMBOL:
            if curve is None or curve.empty:
                raise SyntheticError(
                    "没有国债收益率曲线数据，无法构造人民币现金；"
                    "先运行：python -m etf_lab.cli fetch --preset macro"
                )
            subset = curve[curve["code"] == cash_tenor]
            if subset.empty:
                raise SyntheticError(f"收益率曲线里没有 {cash_tenor}，无法构造人民币现金")
            starts.append(pd.to_datetime(subset["date"]).min())
        elif symbol == USD_SYMBOL:
            if fx is None or fx.empty:
                raise SyntheticError(
                    "没有汇率数据，无法构造美元现金；先运行：python -m etf_lab.cli fetch --preset fx"
                )
            starts.append(pd.to_datetime(fx["date"]).min())
    if not starts:
        return None
    return max(starts)


def synthetic_prices(
    index: pd.DatetimeIndex,
    *,
    symbols: list[str],
    curve: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
    usd_annual_rate: float = 0.0,
    cash_tenor: str = "CN1Y",
) -> pd.DataFrame:
    """按需要构造合成资产价格面板（只包含请求的符号）。"""
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol == CASH_SYMBOL:
            columns[symbol] = cash_price_series(curve, index, tenor=cash_tenor)
        elif symbol == USD_SYMBOL:
            columns[symbol] = usd_price_series(fx, index, annual_rate=usd_annual_rate)
        else:
            raise SyntheticError(f"未知的合成资产：{symbol!r}；已支持 {sorted(SYNTHETIC_ASSETS)}")
    if not columns:
        return pd.DataFrame(index=index)
    return pd.DataFrame(columns, index=index)


def synthetic_meta(symbols: list[str]) -> pd.DataFrame:
    """合成资产的元数据行，让标签/板块识别与真实 ETF 走同一条路。"""
    rows = []
    for symbol in symbols:
        info = SYNTHETIC_ASSETS.get(symbol)
        if info is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": info["name"],
                "asset_class": info["asset_class"],
                "underlying_index": None,
                "t_plus": 0,
                "is_cross_border": symbol == USD_SYMBOL,
            }
        )
    return pd.DataFrame(rows)
