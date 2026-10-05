"""历史情节重放的对照测试（不访问网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_lab.content.episodes import EPISODES, Episode
from etf_lab.core import episodes as replay_mod


def _levels(values: list[float], start: str = "2008-01-02") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"), dtype=float)


EPISODE = Episode(
    key="t",
    title="测试情节",
    start="2008-01-02",
    end="2008-02-29",
    hook="钩子",
    what_happened="发生了什么",
    terms=("max_drawdown",),
)


def test_window_stats_hand_computed() -> None:
    # 100 → 120 → 90：区间收益 -10%，最大回撤从 120 到 90 = -25%
    levels = _levels([100.0] + [110.0] * 10 + [120.0] + [100.0] * 10 + [90.0] + [95.0] * 10)
    stats = replay_mod.window_stats(levels, pd.Timestamp("2008-01-02"), pd.Timestamp("2008-03-31"))
    assert stats is not None
    assert stats.total_return == pytest.approx(95.0 / 100.0 - 1.0)
    assert stats.max_drawdown == pytest.approx(-0.25)
    assert stats.n_obs == len(levels)


def test_window_stats_returns_none_for_short_window() -> None:
    """窗口内交易日少于 MIN_WINDOW_OBS（当前为 5）就不给数字。

    阈值定得低是刻意的：这里只算区间收益/回撤/极值单日，不涉及夏普或相关性，
    所以十来个交易日的窗口（如 2024 年 9-10 月那段）是合法且有意义的。
    """
    levels = _levels([100.0] * 3)
    assert replay_mod.window_stats(levels, pd.Timestamp("2008-01-02"), pd.Timestamp("2008-12-31")) is None
    assert replay_mod.MIN_WINDOW_OBS == 5


def test_replay_marks_uncovered_episode_and_still_reports_market() -> None:
    """组合的 ETF 当时还没上市 → 组合标"未覆盖"，但市场背景仍要用指数给出来。"""
    nav = _levels([1.0] * 40, start="2015-01-05")
    benchmark = _levels([1000.0] * 10 + [800.0] * 10 + [700.0] * 10 + [600.0] * 10, start="2008-01-02")

    result = replay_mod.replay(nav, benchmark, [EPISODE])

    assert result[0]["coverage"] == "none"
    assert result[0]["portfolio"] is None
    assert result[0]["market"] is not None
    assert result[0]["market"]["total_return"] == pytest.approx(600.0 / 1000.0 - 1.0)


def test_replay_marks_partial_coverage_when_start_precedes_data() -> None:
    nav = _levels([1.0] * 30, start="2008-02-01")  # 晚于情节起点，但仍在窗口内
    result = replay_mod.replay(nav, None, [EPISODE])
    assert result[0]["coverage"] == "partial"
    assert result[0]["portfolio"] is not None


def test_replay_marks_full_coverage() -> None:
    # 需要足够长以覆盖到 2008 年：40 个交易日只到 2007-02，会被正确判为未覆盖
    nav = _levels([1.0] * 300, start="2007-01-02")
    result = replay_mod.replay(nav, nav, [EPISODE])
    assert result[0]["coverage"] == "full"
    assert result[0]["portfolio"] is not None
    assert result[0]["market"] is not None


def test_replay_carries_narrative_fields() -> None:
    nav = _levels([1.0] * 40, start="2007-01-02")
    item = replay_mod.replay(nav, None, [EPISODE])[0]
    assert item["title"] == "测试情节"
    assert item["hook"] == "钩子"
    assert item["what_happened"] == "发生了什么"
    assert item["terms"] == ["max_drawdown"]


def test_builtin_episodes_are_well_formed() -> None:
    """内置情节必须都有钩子、机制说明与至少一个术语，且术语卡真实存在。"""
    from etf_lab.content import teaching

    assert len(EPISODES) >= 5
    keys = [e.key for e in EPISODES]
    assert len(keys) == len(set(keys))
    for episode in EPISODES:
        assert episode.hook.strip()
        assert len(episode.what_happened) > 40
        assert episode.terms, f"{episode.key} 没有关联术语"
        assert episode.start < episode.end
        for term in episode.terms:
            assert term in teaching.CARDS, f"{episode.key} 引用了不存在的卡片 {term}"


def test_replay_marks_too_short_window() -> None:
    """数据覆盖了窗口，但窗口内交易日太少 → 标 too_short，而不是自相矛盾地报"全覆盖但无数字"。"""
    short = Episode(
        key="s",
        title="很短的情节",
        start="2008-01-02",
        end="2008-01-04",
        hook="钩子",
        what_happened="很短",
        terms=("max_drawdown",),
    )
    nav = _levels([1.0] * 300, start="2007-01-02")
    item = replay_mod.replay(nav, nav, [short])[0]
    assert item["coverage"] == "too_short"
    assert item["portfolio"] is None
    assert item["market"] is None


def test_coverage_labels_cover_all_states() -> None:
    assert set(replay_mod.COVERAGE_LABELS) == {"full", "partial", "too_short", "none"}
