"""历史情节（战役复盘）的定义。

为什么要有这一层
----------------
知识点挂在目录里没人看，挂在一个**有情节的具体处境**上就有人看。
这里的每个情节都是真实的 A 股市场阶段，问题直接指向"如果你当时持有这个组合会怎样"，
术语则从当时真实发生的事里自然带出来——而不是先列术语再找例子。

诚实边界
--------
* 情节里的数字全部来自本地数据，算不出来就标"未覆盖"，不编。
* 部分术语（CDS、深度价外期权）属于**境外或银行间市场工具**，本组合无法直接使用；
  这类卡片会带 ``scope="concept_only"`` 标记，页面上以徽章形式写明"不在你的可投范围"。
  把小说里的工具当成能买的东西，是这类内容最容易犯的错。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    """一个可重放的历史情节。"""

    key: str
    title: str
    start: str
    end: str
    hook: str
    """一句提问式钩子——先制造"我那时会怎样"的好奇。"""
    what_happened: str
    """当时真实发生了什么，机制是什么。"""
    terms: tuple[str, ...]
    """从这段行情里自然带出的术语卡 key。"""
    benchmark: str = "000300"


EPISODES: tuple[Episode, ...] = (
    Episode(
        key="gfc_2008",
        title="2008：从 6124 到 1664",
        start="2008-01-02",
        end="2008-10-31",
        hook="如果你的持仓经历过 2008 年，你要等多久才能回本？",
        what_happened=(
            "起点是次贷。美国房价见顶后，按揭违约上升，挂钩这些按揭的证券（MBS/CDO）价格崩塌。"
            "真正把危机放大的是杠杆与信用衍生品：机构用很低的保证金持有巨额次贷敞口，"
            "一旦价格下跌就要补保证金，被迫抛售，价格进一步下跌——这就是**去杠杆螺旋**。"
            "A 股同步下跌，沪深300 在这段区间腰斩再腰斩。"
        ),
        terms=("leverage_unwind", "credit_spread_cds", "liquidity_spiral"),
    ),
    Episode(
        key="leverage_2015",
        title="2015：杠杆牛与强平",
        start="2015-06-12",
        end="2016-01-28",
        hook="如果你的组合在 2015-06-12 是满仓的，接下来 7 个月会发生什么？",
        what_happened=(
            "这一轮上涨的核心燃料是场外配资与两融。指数越涨，杠杆资金越多；"
            "当价格开始下跌，触及平仓线就被强制卖出，卖压又压低价格、触发更多平仓。"
            "**杠杆不会创造风险，它只是把风险集中到同一个时刻释放**。"
            "沪深300 从高点回撤超过 40%，创业板相关指数跌幅更大。"
        ),
        terms=("leverage_unwind", "liquidity_spiral", "max_drawdown"),
    ),
    Episode(
        key="covid_2020",
        title="2020：疫情冲击与波动率",
        start="2020-01-20",
        end="2020-03-23",
        hook="一个月内跌掉三成的时候，期权从「贵」变成「更贵」——为什么？",
        what_happened=(
            "疫情把不确定性一次性拉满：企业现金流中断、油价暴跌、美元流动性紧张。"
            "这段时间全球**隐含波动率**急剧抬升，期权价格随之暴涨——"
            "这正是小说里「买保险」能在危机中赚钱的原因，但代价是平时持续付出的权利金。"
            "沪深300 在一个月内快速回撤，随后又用几个月修复。"
        ),
        terms=("implied_volatility", "volatility", "liquidity_spiral"),
    ),
    Episode(
        key="liquidity_2024_02",
        title="2024 年 2 月：小盘股的流动性螺旋",
        start="2024-01-02",
        end="2024-02-05",
        hook="同样的市场，为什么你的宽基组合只跌一点，小盘指数却在几天内崩掉？",
        what_happened=(
            "挂钩小盘指数的雪球产品集中敲入、量化策略在极窄的股票池里同向交易、"
            "叠加流动性收缩，形成了一次典型的**流动性螺旋**："
            "越跌越要卖，越卖越跌，而接盘的人消失。"
            "这段行情最能说明「指数跌了多少」这件事对**不同市值、不同结构**的资产完全不一样。"
        ),
        terms=("liquidity_spiral", "max_drawdown", "correlation"),
    ),
    Episode(
        key="policy_2024_09",
        title="2024 年 9-10 月：政策行情的急涨急跌",
        start="2024-09-24",
        end="2024-10-09",
        hook="十个交易日里涨 20% 再跌 16%，定投和一次性买入的差别有多大？",
        what_happened=(
            "政策转向点燃情绪，资金集中涌入，创业板相关 ETF 单日涨幅触及 20% 涨停；"
            "几天后情绪退潮又快速回落。这类行情对**定投**特别友好（买在下跌途中），"
            "对**追高的一次性买入**特别残酷——同一段行情，两种买法的结果可以完全相反。"
        ),
        terms=("xirr", "arithmetic_vs_geometric", "calmar"),
    ),
    Episode(
        key="tariff_2025_04",
        title="2025 年 4 月：关税冲击",
        start="2025-04-01",
        end="2025-04-30",
        hook="外部冲击来的时候，你组合里的哪一块最先反应？",
        what_happened=(
            "关税落地引发全球风险资产同步下跌，跨境 ETF 还要额外承受汇率与溢价的双重变化。"
            "这类冲击最能检验一件事：你持有的资产**是不是真的分散**——"
            "如果它们在同一天一起跌，那分散化只是账面上的。"
        ),
        terms=("premium_discount", "correlation", "diversification_ratio"),
    ),
)

EPISODES_BY_KEY: dict[str, Episode] = {e.key: e for e in EPISODES}
