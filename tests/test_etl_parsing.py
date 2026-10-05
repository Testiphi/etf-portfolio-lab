"""采集层解析与分页的对照测试（**不访问网络**）。

数据层的静默错误最贵：字段顺序错一位、分页漏一段，产出的收益曲线看着完全正常，
但结论是错的。所以这里用合成数据把所有解析约定钉死。
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from etf_lab.etl import fetch, sohu, tencent


# --------------------------------------------------------------------------- #
# 腾讯：符号、分页、解析
# --------------------------------------------------------------------------- #
def test_tencent_symbol_for_market_prefix() -> None:
    assert tencent.symbol_for("510300", "etf") == "sh510300"
    assert tencent.symbol_for("159915", "etf") == "sz159915"
    assert tencent.symbol_for("518880", "etf") == "sh518880"
    assert tencent.symbol_for("000300", "index") == "sh000300"
    assert tencent.symbol_for("399006", "index") == "sz399006"


def test_tencent_symbol_for_rejects_unknown() -> None:
    with pytest.raises(tencent.TencentError):
        tencent.symbol_for("999999", "etf")
    with pytest.raises(tencent.TencentError):
        tencent.symbol_for("123456", "index")


def test_plan_windows_cover_range_without_gap_or_overlap() -> None:
    start = dt.date(2010, 1, 1)
    end = dt.date(2026, 10, 5)
    windows = tencent.plan_windows(start, end, years=2)

    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (_, stop), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == stop + dt.timedelta(days=1)
    for window_start, window_stop in windows:
        assert window_stop >= window_start
        # 每个窗口都不超过 2 年
        assert window_stop <= window_start.replace(year=window_start.year + 2)


def test_parse_tencent_payload_uses_index_two_as_close() -> None:
    payload = {
        "data": {
            "sh510300": {
                "qfqday": [
                    ["2024-01-03", "3.20", "3.25", "3.30", "3.18", "1000"],
                    ["2024-01-02", "3.22", "3.173", "3.222", "3.171", "2000", "3269676679.0"],
                ],
                "qt": {"sh510300": ["1", "沪深300ETF"]},
            }
        }
    }
    name, frame = tencent.parse_kline_payload(payload, "sh510300", "qfq")

    assert name == "沪深300ETF"
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    # 收盘价必须是下标 2（腾讯口径），而不是最高价或开盘价
    assert frame.loc[0, "close"] == pytest.approx(3.173)
    assert frame.loc[0, "open"] == pytest.approx(3.22)
    assert frame.loc[0, "high"] == pytest.approx(3.222)
    assert frame.loc[0, "amount"] == pytest.approx(3269676679.0)
    assert pd.isna(frame.loc[1, "amount"])


def test_parse_tencent_payload_falls_back_to_day_key() -> None:
    payload = {"data": {"sh510300": {"day": [["2024-01-02", "1", "2", "3", "0.5", "10"]]}}}
    _, frame = tencent.parse_kline_payload(payload, "sh510300", "raw")
    assert frame.loc[0, "close"] == pytest.approx(2.0)


def test_parse_tencent_payload_rejects_short_row() -> None:
    payload = {"data": {"sh510300": {"qfqday": [["2024-01-02", "3.2", "3.3"]]}}}
    with pytest.raises(tencent.TencentError):
        tencent.parse_kline_payload(payload, "sh510300", "qfq")


def test_parse_tencent_payload_rejects_missing_symbol() -> None:
    with pytest.raises(tencent.TencentError):
        tencent.parse_kline_payload({"data": {"sz000001": {}}}, "sh510300", "qfq")


def test_fetch_daily_refuses_window_that_looks_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """窗口满额时必须报错，而不是接受一段可能缺失的行情。"""
    # 用真实交易日序列生成日期：手工拼 "2024-MM-DD" 会造出 13 月这种非法日期，
    # 结果测试失败的原因会变成"日期解析错误"而不是我们要验证的截断保护。
    dates = pd.date_range("2020-01-01", periods=tencent.MAX_BARS_PER_REQUEST, freq="B").strftime("%Y-%m-%d")
    rows = [[day, "1", "2", "3", "0.5", "10"] for day in dates]

    class FakeResponse:
        status_code = 200
        text = '{"data": {}}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": {"sh510300": {"qfqday": rows}}}

    class FakeSession:
        trust_env = True

        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return FakeResponse()

    with pytest.raises(tencent.TencentError, match="疑似被截断"):
        tencent.fetch_daily(
            "510300",
            start="2024-01-01",
            end="2024-06-30",
            adjust="qfq",
            session=FakeSession(),  # type: ignore[arg-type]
            pause=0.0,
        )


def test_parse_tencent_payload_allows_empty_window() -> None:
    """上市前的窗口返回空是**合法**的，不能让整只标的抓取失败。"""
    payload = {"data": {"sh512100": {"day": [], "mx_price": None}}}
    name, frame = tencent.parse_kline_payload(payload, "sh512100", "qfq", allow_empty=True)
    assert name == "sh512100"
    assert frame.empty
    with pytest.raises(tencent.TencentError):
        tencent.parse_kline_payload(payload, "sh512100", "qfq", allow_empty=False)


class _FakeResponse:
    def __init__(self, payload: dict, text: str = "", status_code: int = 200) -> None:
        self._payload = payload
        self.text = text or json.dumps(payload, ensure_ascii=False)
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _SequenceSession:
    """按调用次序返回不同响应的假 Session。"""

    trust_env = True

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self.calls = 0

    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _FakeResponse(payload)


def _payload(rows: list[list[str]]) -> dict:
    return {"data": {"sh512100": {"qfqday": rows, "qt": {"sh512100": ["1", "中证1000ETF"]}}}}


def test_fetch_daily_skips_windows_before_listing() -> None:
    """这只 ETF 上市晚：第一个窗口（2018-2019）为空，第二个窗口（2020-2021）有数据。

    这正是本项目真实踩过的坑——早期版本把空窗口当错误，导致所有 2016 年后上市的
    ETF 全部抓取失败，而报表只会显示"不可用"，非常容易被误判成数据源问题。
    """
    session = _SequenceSession(
        [
            _payload([]),
            _payload([["2020-01-02", "1.0", "1.1", "1.2", "0.9", "100"]]),
        ]
    )
    result = tencent.fetch_daily(
        "512100",
        kind="etf",
        start="2018-01-01",
        end="2021-06-30",
        adjust="qfq",
        session=session,  # type: ignore[arg-type]
        pause=0.0,
    )
    assert len(result.frame) == 1
    assert str(result.frame["date"].iloc[0].date()) == "2020-01-02"
    assert session.calls == 2


def test_fetch_daily_detects_anti_bot_challenge_page() -> None:
    """反爬挑战页必须被明确识别。

    否则它会被当成"该标的没有数据"——那会让整段历史**静默消失**，
    而报表上只会显示某只 ETF "不可用"，极难排查。腾讯 WAF 实测返回的正是
    HTTP 501 + 一段 JS 跳转的 HTML。
    """
    challenge = '<!DOCTYPE html><html><head><script>var i=location.href;window.location.href="https://wa"</script></head></html>'

    class ChallengeSession:
        trust_env = True

        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _FakeResponse({}, text=challenge, status_code=501)

    with pytest.raises(tencent.TencentError, match="反爬拦截"):
        tencent.fetch_daily(
            "510300",
            start="2024-01-01",
            end="2024-06-30",
            adjust="qfq",
            session=ChallengeSession(),  # type: ignore[arg-type]
            pause=0.0,
            max_retries=1,
        )


def test_fetch_daily_raises_when_every_window_is_empty() -> None:
    session = _SequenceSession([_payload([])])
    with pytest.raises(tencent.TencentError, match="所有窗口都没有数据"):
        tencent.fetch_daily(
            "512100",
            kind="etf",
            start="2018-01-01",
            end="2021-06-30",
            adjust="qfq",
            session=session,  # type: ignore[arg-type]
            pause=0.0,
        )


# --------------------------------------------------------------------------- #
# 搜狐：符号与解析
# --------------------------------------------------------------------------- #
def test_sohu_symbol_for() -> None:
    assert sohu.symbol_for("510300", "etf") == "cn_510300"
    assert sohu.symbol_for("159915", "etf") == "cn_159915"
    assert sohu.symbol_for("000300", "index") == "cn_000300"


def test_parse_sohu_payload_sorts_descending_input() -> None:
    payload = [
        {
            "code": "cn_510300",
            "status": 0,
            "hq": [
                ["2024-01-03", "3.30", "3.28", "0.05", "1.5%", "3.25", "3.32", "1000", "3300", "0.5%"],
                ["2024-01-02", "3.20", "3.173", "-0.05", "-1.4%", "3.17", "3.22", "2000", "6400", "0.6%"],
            ],
        }
    ]
    name, frame = sohu.parse_his_payload(payload, "510300")

    assert name == "cn_510300"
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert frame.loc[0, "close"] == pytest.approx(3.173)  # 收盘价在下标 2
    assert frame.loc[0, "low"] == pytest.approx(3.17)
    assert frame.loc[0, "high"] == pytest.approx(3.22)


def test_parse_sohu_payload_rejects_service_error() -> None:
    with pytest.raises(sohu.SohuError):
        sohu.parse_his_payload({"__ERROR": 3}, "510300")


# --------------------------------------------------------------------------- #
# 合并与交叉校验
# --------------------------------------------------------------------------- #
def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": [3.2, 3.3, 3.4],
            "high": [3.3, 3.4, 3.5],
            "low": [3.1, 3.2, 3.3],
            "close": [3.0, 3.0, 3.0],
            "volume": [100.0, 200.0, 300.0],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    qfq = pd.DataFrame({"date": dates, "close": [2.7, 2.85, 3.0]})
    return raw, qfq


def test_merge_adjusted_computes_adj_factor() -> None:
    raw, qfq = _frames()
    merged = fetch._merge_adjusted(raw, qfq, "510300")

    assert list(merged["symbol"]) == ["510300"] * 3
    # 越早的日期前复权价被压得越低 → 复权因子越小
    assert merged["adj_factor"].is_monotonic_increasing
    assert merged.loc[2, "adj_factor"] == pytest.approx(1.0)
    assert merged.loc[0, "adj_factor"] == pytest.approx(0.9)


def test_merge_adjusted_rejects_disjoint_dates() -> None:
    raw, qfq = _frames()
    qfq = qfq.assign(date=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04"]))
    with pytest.raises(RuntimeError):
        fetch._merge_adjusted(raw, qfq, "510300")


def test_verify_against_sohu_flags_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, _ = _frames()
    tampered = raw.assign(close=[3.0, 3.0, 3.5])  # 最后一天差 16.7%

    def fake_fetch_daily(*args, **kwargs):  # noqa: ANN002, ANN003
        return sohu.SohuResult(code="510300", name="cn_510300", frame=tampered)

    monkeypatch.setattr(fetch.sohu, "fetch_daily", fake_fetch_daily)
    quality = fetch.verify_against_sohu("510300", raw, "2024-01-01", "2024-12-31", session=None)  # type: ignore[arg-type]

    assert quality["checked"] is True
    assert quality["overlap"] == 3
    assert "warning" in quality
    assert quality["max_rel_diff"] > fetch.VERIFY_TOLERANCE


def test_verify_against_sohu_passes_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, _ = _frames()

    def fake_fetch_daily(*args, **kwargs):  # noqa: ANN002, ANN003
        return sohu.SohuResult(code="510300", name="cn_510300", frame=raw.copy())

    monkeypatch.setattr(fetch.sohu, "fetch_daily", fake_fetch_daily)
    quality = fetch.verify_against_sohu("510300", raw, "2024-01-01", "2024-12-31", session=None)  # type: ignore[arg-type]

    assert quality["checked"] is True
    assert quality["max_rel_diff"] == pytest.approx(0.0)
    assert "warning" not in quality


def test_verify_against_sohu_records_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, _ = _frames()

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise sohu.SohuError("接口不可用")

    monkeypatch.setattr(fetch.sohu, "fetch_daily", boom)
    quality = fetch.verify_against_sohu("510300", raw, "2024-01-01", "2024-12-31", session=None)  # type: ignore[arg-type]

    assert quality["checked"] is False
    assert "接口不可用" in str(quality["error"])
