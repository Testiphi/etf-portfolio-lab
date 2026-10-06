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

from etf_lab.data import repo
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
