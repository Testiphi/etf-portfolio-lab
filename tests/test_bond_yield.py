"""国债收益率曲线解析的对照测试（不访问网络）。

重点是那两个**实测踩出来的结构特征**：页面里有多张表；数据表里堆叠了多条曲线。
用合成 HTML 把这两条钉死，避免接口改版或过滤写错时静默串行。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from etf_lab.etl import bond_yield

HEADER = ["曲线名称", "日期", "3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]


def _chinabond_html(curves: list[tuple[str, list[list[str]]]]) -> str:
    """构造与真实页面同构的 HTML：第 1 张表是占位，第 2 张才是数据表。"""
    rows: list[list[str]] = []
    for name, values in curves:
        for row in values:
            rows.append([name, *row])
    data = pd.DataFrame(rows, columns=HEADER)
    filler = pd.DataFrame([["占位", "表"]])
    return filler.to_html(index=False) + data.to_html(index=False)


def _row(date: str, base: float) -> list[str]:
    return [date] + [f"{base + i * 0.01:.4f}" for i in range(8)]


def test_parse_chinabond_filters_to_the_government_curve() -> None:
    """数据表里堆叠了多条曲线，必须只取国债那条。

    实测：1 年窗口 745 行 ≈ 3 条曲线 × 248 个交易日；
    不过滤就会把同一天的不同曲线塞进同一日期。
    """
    html = _chinabond_html(
        [
            ("中债国开债收益率曲线", [_row("2025-01-02", 1.9), _row("2025-01-03", 1.91)]),
            (bond_yield.CURVE_NAME, [_row("2025-01-02", 1.1), _row("2025-01-03", 1.12)]),
            ("中债地方政府债收益率曲线", [_row("2025-01-02", 2.0)]),
        ]
    )
    frame = bond_yield.parse_chinabond_html(html)

    assert set(frame["code"]) == {"CN3M", "CN6M", "CN1Y", "CN3Y", "CN5Y", "CN7Y", "CN10Y", "CN30Y"}
    # 每个期限每天只应有一行（过滤生效，没有把三条曲线混在一起）
    assert len(frame) == 8 * 2
    assert frame.groupby(["code", "date"]).size().max() == 1
    # 取的是国债曲线（基数 1.1 附近），不是国开债（1.9）
    assert frame.loc[frame["code"] == "CN10Y", "yield"].min() < 1.2


def test_parse_chinabond_reports_missing_curve_with_available_names() -> None:
    """找不到国债曲线时要报出页面上实际有哪些曲线，而不是含糊失败。"""
    html = _chinabond_html([("中债国开债收益率曲线", [_row("2025-01-02", 1.9)])])
    with pytest.raises(bond_yield.BondYieldError, match="未找到"):
        bond_yield.parse_chinabond_html(html)


def test_parse_chinabond_rejects_page_without_second_table() -> None:
    only_one = pd.DataFrame([["a", "b"]]).to_html(index=False)
    with pytest.raises(bond_yield.BondYieldError, match="表格数不足"):
        bond_yield.parse_chinabond_html(only_one)


def test_parse_chinabond_skips_non_numeric_yields() -> None:
    html = _chinabond_html([(bond_yield.CURVE_NAME, [_row("2025-01-02", 1.1), ["2025-01-03", "--", "--", "--", "--", "--", "--", "--", "--"]])])
    frame = bond_yield.parse_chinabond_html(html)
    assert len(frame) == 8  # 只有第一天有效
    assert frame["date"].nunique() == 1


def test_parse_sina_payload_maps_tenor_and_types() -> None:
    payload = {"result": {"data": [{"d": "2025-01-02", "c": "1.6500"}, {"d": "2025-01-03", "c": "1.6700"}]}}
    frame = bond_yield.parse_sina_payload(payload, "CN10Y")
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2025-01-02", "2025-01-03"]
    assert frame["yield"].iloc[1] == pytest.approx(1.67)
    assert set(frame["code"]) == {"CN10Y"}
    assert set(frame["tenor"]) == {"10年"}


def test_parse_sina_payload_rejects_empty_and_bad_shape() -> None:
    with pytest.raises(bond_yield.BondYieldError, match="为空"):
        bond_yield.parse_sina_payload({"result": {"data": []}}, "CN10Y")
    with pytest.raises(bond_yield.BondYieldError, match="字段异常"):
        bond_yield.parse_sina_payload({"result": {"data": [{"x": 1}]}}, "CN10Y")


def test_windows_never_exceed_one_year_and_stay_contiguous() -> None:
    """中债接口要求区间小于一年，分窗必须既不超过上限、也不留空隙。"""
    start, end = dt.date(2015, 1, 1), dt.date(2026, 10, 5)
    windows = bond_yield._windows(start, end)
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for window_start, window_stop in windows:
        assert (window_stop - window_start).days <= bond_yield.WINDOW_DAYS
        assert window_stop >= window_start
    for (_, stop), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == stop + dt.timedelta(days=1)


def test_tenor_maps_are_consistent() -> None:
    """期限标签、代码、年数三张映射表必须互相对齐，否则久期会算错单位。"""
    assert set(bond_yield.TENOR_CODES.values()) == set(bond_yield.TENOR_YEARS)
    for label, code in bond_yield.TENOR_CODES.items():
        assert bond_yield.TENOR_YEARS[code] > 0
    assert bond_yield.TENOR_CODES["1年"] == "CN1Y"
    assert bond_yield.TENOR_YEARS["CN1Y"] == 1.0
    assert set(bond_yield.SINA_SYMBOLS) <= set(bond_yield.TENOR_YEARS)
