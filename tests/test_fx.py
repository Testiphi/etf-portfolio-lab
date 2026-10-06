"""汇率解析的对照测试（不访问网络）。

重点钉住三件实测踩过的事：
1. **不能按下标取表**——ajax 片段只有一张表，整页有两张；
2. 源站以「元/100 外币」计价，必须除以 100；
3. 最新几天可能是 ``--`` 占位，必须剔除，不能当 0 也不能让整页失败。
"""

from __future__ import annotations

import pandas as pd
import pytest

from etf_lab.etl import fx

HEADER = ["日期", "中行汇买价(元)", "中行钞买价(元)", "中行钞卖价/汇卖价", "央行中间价", "中行折算价"]


def _html(rows: list[list[str]], *, nav_table: bool = False) -> str:
    data = pd.DataFrame(rows, columns=HEADER).to_html(index=False)
    if not nav_table:
        return data
    nav = pd.DataFrame([["财经首页", "股票", "基金"]]).to_html(index=False)
    return nav + data


def test_parse_picks_the_table_with_the_parity_column() -> None:
    """整页有两张表时必须找到含央行中间价的那张，而不是取下标 1。"""
    html = _html([["2025-12-31", "697.62", "697.62", "700.56", "702.88", "702.88"]], nav_table=True)
    frame = fx.parse_fx_html(html, "USDCNY")
    assert len(frame) == 1
    assert frame["close"].iloc[0] == pytest.approx(7.0288)


def test_parse_divides_by_one_hundred() -> None:
    """源站以「元/100 外币」计价，忘记除 100 会把汇率放大 100 倍。"""
    html = _html([["2012-12-31", "628.00", "628.00", "631.00", "628.55", "628.55"]])
    frame = fx.parse_fx_html(html, "USDCNY")
    assert frame["close"].iloc[0] == pytest.approx(6.2855)
    assert frame["close"].iloc[0] < 10, "汇率应当在 6~8 的量级"


def test_parse_drops_placeholder_rows() -> None:
    """`--` 占位必须被剔除，且不影响同一页里的有效行。"""
    html = _html(
        [
            ["2026-10-06", "--", "--", "--", "--", "--"],
            ["2026-10-05", "700.00", "700.00", "703.00", "702.50", "702.50"],
        ]
    )
    frame = fx.parse_fx_html(html, "USDCNY")
    assert len(frame) == 1
    assert frame["close"].iloc[0] == pytest.approx(7.025)


def test_parse_raises_on_all_placeholder_page() -> None:
    """整页都是占位符要明确报错，而不是返回空表让人以为"没数据就没数据"。"""
    html = _html([["2026-10-06", "--", "--", "--", "--", "--"]])
    with pytest.raises(fx.FxError, match="有效汇率记录"):
        fx.parse_fx_html(html, "USDCNY")


def test_parse_raises_when_structure_changes() -> None:
    """页面结构变了要报错，不要静默取到导航表当数据。"""
    nav = pd.DataFrame([["财经首页", "股票", "基金"]]).to_html(index=False)
    with pytest.raises(fx.FxError, match="央行中间价"):
        fx.parse_fx_html(nav, "USDCNY")


def test_parse_sorts_by_date_and_keeps_pair() -> None:
    html = _html(
        [
            ["2025-12-31", "697.62", "697.62", "700.56", "702.88", "702.88"],
            ["2025-11-04", "711.70", "711.70", "714.69", "708.85", "708.85"],
        ]
    )
    frame = fx.parse_fx_html(html, "USDCNY")
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2025-11-04", "2025-12-31"]
    assert set(frame["pair"]) == {"USDCNY"}


def test_unknown_pair_rejected() -> None:
    with pytest.raises(fx.FxError, match="未知货币对"):
        fx.fetch_pair("XXXCNY", start="2024-01-01", end="2024-02-01")


def test_page_count_reads_pagination_marker() -> None:
    html = '<a class="page" href="#">1</a><a class="page" href="#">2</a><a class="page" href="#">7</a>'
    assert fx._page_count(html) == 7
    assert fx._page_count("<html>无分页</html>") == 1
