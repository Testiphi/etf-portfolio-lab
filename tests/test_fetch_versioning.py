"""数据版本记账的不变量测试（不访问网络）。

**为什么值得单独立一条测试**
---------------------------
应用的缓存键是「组合 | 数据版本」。版本不变，就会把**旧数据算出的结果**当成新的用——
页面照常显示、数字却是过期的，而且**没有任何报错**。

记账原本只在 CLI 里做，于是任何直接调用 ``fetch_*`` 的脚本都会静默漏掉
（本项目的作者自己就这么漏过一次：手工补采了 3 只 ETF，版本号却没动）。
所以现在把记账收进 ETL 自身，并由这条测试守住。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etf_lab.data import repo
from etf_lab.data import audit
from etf_lab.etl import fetch


def _canned_pair(code, kind, start, end, session):
    """伪造一次行情响应，让测试不碰网络。"""
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [1.0] * 6,
            "high": [1.1] * 6,
            "low": [0.9] * 6,
            "close": [1.0, 1.02, 0.98, 1.03, 1.01, 1.04],
            "volume": [100.0] * 6,
            "amount": [100.0] * 6,
        }
    )
    return frame.copy(), frame.copy(), f"测试{code}", "canned"


def test_successful_etf_fetch_bumps_data_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "_fetch_pair", _canned_pair)
    con = repo.connect(tmp_path / "lab.duckdb")
    try:
        before = repo.latest_data_version(con)
        reports = fetch.fetch_etf_prices(con, ["999999"], start="2020-01-01", verify=False)
        after = repo.latest_data_version(con)

        assert any(report.ok for report in reports), "伪响应应当成功入库"
        assert after != before, "成功写入后数据版本必须递增"
        assert repo.table_counts(con)["etf_price"] == 6
    finally:
        con.close()


def test_failed_fetch_does_not_bump_data_version(tmp_path: Path, monkeypatch) -> None:
    """一项都没成功时不该动版本——否则版本号会变成噪音，缓存也会被无谓清空。"""

    def broken(*args, **kwargs):
        raise RuntimeError("模拟接口故障")

    monkeypatch.setattr(fetch, "_fetch_pair", broken)
    con = repo.connect(tmp_path / "lab.duckdb")
    try:
        before = repo.latest_data_version(con)
        reports = fetch.fetch_etf_prices(con, ["999999"], start="2020-01-01", verify=False)
        after = repo.latest_data_version(con)

        assert not any(report.ok for report in reports)
        assert after == before, "全部失败时版本不应变化"
    finally:
        con.close()


def test_version_note_says_what_was_written(tmp_path: Path, monkeypatch) -> None:
    """版本备注要写清是哪一类数据——排查"数字为什么变了"时全靠它。"""
    monkeypatch.setattr(fetch, "_fetch_pair", _canned_pair)
    con = repo.connect(tmp_path / "lab.duckdb")
    try:
        fetch.fetch_etf_prices(con, ["999999"], start="2020-01-01", verify=False)
        notes = con.execute("SELECT source, notes FROM data_version ORDER BY updated_at DESC").fetchall()
        joined = " ".join(f"{source} {note}" for source, note in notes)
        assert "etf_price" in joined
    finally:
        con.close()


@pytest.mark.parametrize("quality,status", [
    (None, "本次未执行校验"),
    ({"checked": False, "error": "offline"}, "校验未完成"),
    ({"checked": True, "overlap": 6, "max_rel_diff": 0.0, "tolerance": .001}, "重叠区间一致（阈值内）"),
    ({"checked": True, "overlap": 6, "max_rel_diff": .1, "tolerance": .001}, "重叠区间存在差异"),
    ({"checked": True, "overlap": 6, "max_rel_diff": float("nan"), "tolerance": .001}, "校验结果不完整"),
])
def test_audit_distinguishes_outcomes_and_detects_changed_prices(monkeypatch, quality, status):
    monkeypatch.setattr(fetch, "_fetch_pair", _canned_pair)
    monkeypatch.setattr(fetch, "verify_against_sohu", lambda *args: quality)
    con = repo.connect(":memory:")
    try:
        fetch.fetch_etf_prices(con, ["999999"])
        assert audit.etf_checks(con)[0]["status"] == status
        con.execute("UPDATE etf_price SET close_adj = close_adj * 2 WHERE symbol = '999999'")
        assert audit.etf_checks(con)[0]["status"] == "库存已变化，记录过期"
    finally:
        con.close()


def test_failed_audit_is_persisted_without_changing_prices_or_version(monkeypatch):
    monkeypatch.setattr(fetch, "_fetch_pair", _canned_pair)
    con = repo.connect(":memory:")
    try:
        fetch.fetch_etf_prices(con, ["999999"], verify=False)
        version = repo.latest_data_version(con)
        fingerprint = audit.etf_fingerprint(con, "999999")
        def broken(*args):
            raise RuntimeError("offline")
        monkeypatch.setattr(fetch, "_fetch_pair", broken)
        fetch.fetch_etf_prices(con, ["999999"])
        assert repo.latest_data_version(con) == version
        assert audit.etf_fingerprint(con, "999999") == fingerprint
        assert audit.etf_checks(con)[0]["status"] == "最近采集失败"
        assert con.execute("SELECT COUNT(*) FROM fetch_audit").fetchone()[0] == 2
    finally:
        con.close()


def test_old_schema_remains_readable_without_audit_table():
    con = repo.connect(":memory:")
    try:
        con.execute("DROP TABLE fetch_audit")
        con.execute("INSERT INTO etf_price (symbol, date, close) VALUES ('AAA', '2020-01-01', 1)")
        assert audit.etf_checks(con)[0]["status"] == "未保存校验记录"
    finally:
        con.close()


def test_data_versions_are_distinct_even_at_the_same_clock_time(monkeypatch):
    original = repo.dt.datetime
    class Frozen(original):
        @classmethod
        def now(cls, tz=None):
            return original(2026, 1, 1)
    monkeypatch.setattr(repo.dt, "datetime", Frozen)
    con = repo.connect(":memory:")
    try:
        versions = [repo.log_data_version(con, "test") for _ in range(3)]
        assert len(set(versions)) == 3
        assert repo.latest_data_version(con) == versions[-1]
        assert con.execute("SELECT COUNT(*) FROM data_version").fetchone()[0] == 3
    finally:
        con.close()


def test_verification_counts_only_finite_overlapping_prices(monkeypatch):
    from types import SimpleNamespace
    raw, _, _, _ = _canned_pair("x", "etf", None, None, None)
    reference = raw.copy()
    reference.loc[0, "close"] = float("nan")
    reference.loc[1, "close"] = 0
    monkeypatch.setattr(fetch.sohu, "fetch_daily", lambda *a, **k: SimpleNamespace(frame=reference))
    result = fetch.verify_against_sohu("x", raw, None, None, None)
    assert result["overlap"] == 4
    assert result["first"] == str(raw.loc[2, "date"].date())
    assert result["max_rel_diff"] == 0
