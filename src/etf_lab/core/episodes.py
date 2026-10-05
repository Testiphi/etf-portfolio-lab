"""历史情节重放：把**你现在的组合**放回真实的历史阶段里。

这是"边看自己持仓边学东西"的实现方式：不是先给你一份术语表，
而是先把真实发生过的行情在你的组合上重算一遍，让数字先把问题提出来，
术语再从"当时到底发生了什么"里自然带出来。

诚实边界
--------
标的上市时间决定能覆盖到哪：多数 A 股 ETF 是 2012 年之后才有的，
所以 2008 那段只能用**标的指数**做市场背景，并明确标注组合未覆盖。
**算不出来就标未覆盖，不编数字。**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from etf_lab.content.episodes import Episode
from etf_lab.core import metrics

MIN_WINDOW_OBS = 5
"""窗口内交易日少于该数量就判定为"太短，不给数字"。

这里只算区间收益、回撤与极值单日，**不涉及夏普/相关性**，所以十天左右的窗口是合法的：
「2024 年 9-10 月政策行情」本身就是十来个交易日的故事，不该因为样本短就被丢掉。
"""

TRADING_DAYS = 252


@dataclass(frozen=True)
class WindowStats:
    """一段区间内的表现。"""

    total_return: float
    max_drawdown: float
    worst_day: float
    best_day: float
    recovery_days: int | None
    n_obs: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in payload.items()}


def window_stats(levels: pd.Series, start: pd.Timestamp, end: pd.Timestamp, min_obs: int = MIN_WINDOW_OBS) -> WindowStats | None:
    """区间收益、回撤、极值单日与修复天数；样本不足返回 ``None``。"""
    if levels is None or levels.empty:
        return None
    window = levels.loc[(levels.index >= start) & (levels.index <= end)].dropna()
    if len(window) < min_obs:
        return None
    returns = window.pct_change().dropna()
    info = metrics.max_drawdown(window)
    return WindowStats(
        total_return=float(window.iloc[-1] / window.iloc[0] - 1.0),
        max_drawdown=float(info.depth),
        worst_day=float(returns.min()) if not returns.empty else float("nan"),
        best_day=float(returns.max()) if not returns.empty else float("nan"),
        recovery_days=info.recovery_days,
        n_obs=int(len(window)),
    )


def replay(
    nav: pd.Series,
    benchmark: pd.Series | None,
    episodes: Sequence[Episode],
    *,
    min_obs: int = MIN_WINDOW_OBS,
) -> list[dict[str, Any]]:
    """逐个情节重放，返回可直接渲染的结构。

    ``nav`` 是组合净值（其起点由成分标的的上市时间决定），
    ``benchmark`` 是市场基准指数点位（历史上可以追溯得更久）。
    """
    out: list[dict[str, Any]] = []
    data_start = nav.index[0] if not nav.empty else None
    data_end = nav.index[-1] if not nav.empty else None

    for episode in episodes:
        start = pd.Timestamp(episode.start)
        end = pd.Timestamp(episode.end)

        portfolio_stats = window_stats(nav, start, end, min_obs)
        market_stats = window_stats(benchmark, start, end, min_obs) if benchmark is not None else None

        if data_start is None or data_end is None or end < data_start or start > data_end:
            coverage = "none"
            portfolio_stats = None
        elif start < data_start:
            coverage = "partial"
        elif portfolio_stats is None:
            # 数据覆盖了这个窗口，但窗口内交易日太少，不足以给出统计
            coverage = "too_short"
        else:
            coverage = "full"

        out.append(
            {
                "key": episode.key,
                "title": episode.title,
                "start": episode.start,
                "end": episode.end,
                "hook": episode.hook,
                "what_happened": episode.what_happened,
                "terms": list(episode.terms),
                "coverage": coverage,
                "portfolio": portfolio_stats.as_dict() if portfolio_stats else None,
                "market": market_stats.as_dict() if market_stats else None,
                "benchmark": episode.benchmark,
            }
        )
    return out


COVERAGE_LABELS = {
    "full": "组合全覆盖",
    "partial": "组合部分覆盖（起点晚于情节开始）",
    "too_short": "窗口内交易日太少，不给统计数字",
    "none": "当时的 ETF 尚未上市，只能用指数看市场背景",
}
