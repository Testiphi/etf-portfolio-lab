"""示例组合定义。

这三个组合是静态站（路线 A）与实算应用（路线 C）共用的演示对象。
每一组都刻意对应一个**教学问题**，而不是随便挑几只 ETF：

- ``three_asset``  股债金三分法：分散化到底降了什么风险？
- ``broad_plus_industry``  宽基打底 + 行业卫星：加一个板块的边际贡献有多大？
- ``cross_border``  加入跨境资产：收益里有多少是汇率给的？

注意：这些组合是**教学示例**，不构成任何推荐。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PortfolioSpec:
    """一个待测组合。"""

    key: str
    name: str
    question: str
    """这个组合要回答的教学问题（会显示在页面标题下方）。"""
    weights: Mapping[str, float]
    dca: Mapping[str, Any] = field(default_factory=dict)
    rebalance: Mapping[str, Any] = field(default_factory=dict)
    """再平衡规则：``{"policy": daily/monthly/quarterly/annually/never/threshold,
    "threshold": 0.05, "cost_bps": 0.0}``。
    空字典表示沿用历史行为（每日再平衡、零成本）——**默认不能悄悄改变已有数字**。"""
    cash: Mapping[str, Any] = field(default_factory=dict)
    """合成资产参数：``{"usd_annual_rate": 0.0, "cash_tenor": "CN1Y"}``。
    美元默认**不生息**：美债利率历史只有近 4 年，叠加当前利率会系统性高估早期年份。"""
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_DCA = {"amount": 2000.0, "freq": "monthly", "mode": "fixed", "day": None}


PRESETS: tuple[PortfolioSpec, ...] = (
    PortfolioSpec(
        key="three_asset",
        name="股债金三分法",
        question="把资金分成股票、债券、黄金三份，究竟降低了哪一种风险？",
        weights={"510300": 0.5, "511010": 0.3, "518880": 0.2},
        dca=_DEFAULT_DCA,
        caveat="债券 ETF 的价格受利率影响，久期越长波动越大；黄金不生息，长期收益取决于买入时点。",
    ),
    PortfolioSpec(
        key="broad_plus_industry",
        name="宽基打底 + 行业卫星",
        question="往宽基组合里加入一个行业板块，收益提升是否与风险提升相称？",
        weights={"510300": 0.6, "510500": 0.2, "159915": 0.2},
        dca=_DEFAULT_DCA,
        caveat="行业与主题 ETF 上市时间普遍较短，样本不足会让「年化收益」看起来比实际更确定。",
    ),
    PortfolioSpec(
        key="cross_border",
        name="加入跨境资产",
        question="跨境 ETF 的收益里，有多少来自标的指数、多少来自汇率？",
        weights={"510300": 0.5, "513100": 0.3, "159920": 0.2},
        dca=_DEFAULT_DCA,
        caveat="跨境 ETF 存在溢价、汇率与时区差异，其溢价回落会造成与指数无关的亏损。",
    ),
)

PRESETS_BY_KEY: dict[str, PortfolioSpec] = {p.key: p for p in PRESETS}


def get(key: str) -> PortfolioSpec:
    if key not in PRESETS_BY_KEY:
        raise KeyError(f"未知示例组合：{key!r}；可选 {sorted(PRESETS_BY_KEY)}")
    return PRESETS_BY_KEY[key]
