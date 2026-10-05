"""基金净值采集与折溢价计算的对照测试（不访问网络）。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from etf_lab.etl import fund_nav


def _payload(rows: list[tuple[str, str, str]], err_code: int = 0) -> dict:
    return {
        "ErrCode": err_code,
        "Data": {
            "SHORTNAME": "沪深300ETF华泰柏瑞",
            "LSJZList": [
                {"FSRQ": date, "DWJZ": nav, "LJJZ": acc, "JZZZL": "0.35"}
                for date, nav, acc in rows
            ],
        },
    }


def test_parse_lsjz_payload_sorts_and_types() -> None:
    frame = fund_nav.parse_lsjz_payload(
        _payload([("2024-01-03", "3.5000", "3.9000"), ("2024-01-02", "3.4800", "3.8800")])
    )
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert frame["nav"].iloc[1] == pytest.approx(3.5)
    assert frame["acc_nav"].iloc[1] == pytest.approx(3.9)
    assert frame["daily_change_pct"].iloc[0] == pytest.approx(0.35)


def test_parse_lsjz_payload_skips_rows_without_nav() -> None:
    """只有分红信息、没有净值的行要跳过——而不是当成 0 净值（那会造成 -100% 的假暴跌）。"""
    payload = _payload([("2024-01-03", "--", "--"), ("2024-01-02", "3.4800", "3.8800")])
    frame = fund_nav.parse_lsjz_payload(payload)
    assert len(frame) == 1
    assert frame["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_parse_lsjz_payload_rejects_error_code() -> None:
    with pytest.raises(fund_nav.FundNavError, match="ErrCode"):
        fund_nav.parse_lsjz_payload(_payload([], err_code=1))


def test_parse_lsjz_payload_rejects_empty_list() -> None:
    with pytest.raises(fund_nav.FundNavError, match="LSJZList 为空"):
        fund_nav.parse_lsjz_payload(_payload([]))


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _PagingSession:
    """按 pageIndex 返回不同页的假 Session。"""

    trust_env = True

    def __init__(self, pages: list[dict]) -> None:
        self._pages = pages
        self.calls: list[int] = []

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001, ANN003
        page = int((params or {}).get("pageIndex", 1))
        self.calls.append(page)
        index = min(page - 1, len(self._pages) - 1)
        return _FakeResponse(self._pages[index])

    def close(self) -> None:
        return None


def test_fetch_nav_history_paginates_until_page_not_full() -> None:
    """第二页记录数少于 page_size 即认为到底，应当停止而不是继续请求。"""
    page1 = _payload([("2024-03-01", "3.60", "4.00"), ("2024-02-01", "3.55", "3.95")])
    page2 = _payload([("2024-01-02", "3.48", "3.88")])
    session = _PagingSession([page1, page2])

    result = fund_nav.fetch_nav_history("510300", start="2010-01-01", session=session, page_size=2, pause=0.0)

    assert session.calls == [1, 2]
    assert len(result.frame) == 3
    assert result.source == "eastmoney-lsjz"


def test_fetch_nav_history_stops_when_covering_start_date() -> None:
    page1 = _payload([("2024-03-01", "3.60", "4.00"), ("2024-02-01", "3.55", "3.95")])
    session = _PagingSession([page1, _payload([("2024-01-02", "3.48", "3.88")])])

    result = fund_nav.fetch_nav_history("510300", start="2024-02-15", session=session, page_size=2, pause=0.0)

    # 第一页最早日期已早于 start，无需翻第二页
    assert session.calls == [1]
    assert len(result.frame) == 1  # 只保留 start 之后的记录
    assert result.frame["date"].iloc[0] == pd.Timestamp("2024-03-01")


def test_premium_discount_hand_computed() -> None:
    price = pd.Series([1.02, 0.99], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    nav = pd.Series([1.00, 1.00], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    pd_series = fund_nav.premium_discount(price, nav)
    assert pd_series.iloc[0] == pytest.approx(0.02)
    assert pd_series.iloc[1] == pytest.approx(-0.01)


def test_premium_discount_inner_joins_dates() -> None:
    price = pd.Series([1.02], index=pd.to_datetime(["2024-01-02"]))
    nav = pd.Series([1.00], index=pd.to_datetime(["2024-01-05"]))
    assert fund_nav.premium_discount(price, nav).empty
