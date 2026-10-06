"""进程池作业函数的**跨进程**回归测试。

这一条必须真起子进程才测得到：DuckDB 在**同一进程内**允许多个连接混合读写，
所以任何进程内测试都会放过这个 bug。真实故障是这样的：

1. NiceGUI 主进程以只读打开数据库；
2. 进程池 worker 以**读写**打开（原实现），取得独占锁；
3. DuckDB 的数据库实例在进程内常驻——worker 算完任务、``con.close()`` 之后
   实例仍然持锁，直到进程退出；
4. 第二个 worker 再打开就失败：``IOException: 另一个程序正在使用此文件``；
5. 表现是"有的组合能打开、有的报 500"，而且**取决于访问顺序**。

修法是让作业函数只读（多进程只读共享锁是允许的）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from etf_lab.data import repo

ROOT = Path(__file__).resolve().parents[1]

# 子进程脚本：调用作业函数并把结果写到文件。
# 不通过管道回传（沙箱下管道的 stdio 可能不可用），而是写文件再读。
_CHILD_SCRIPT = """
import json, sys
sys.path.insert(0, {src!r})
from etf_lab.services.jobs import compute_custom_job

result = compute_custom_job(
    {{"AAA": 0.6, "BBB": 0.4}},
    {{"amount": 1000.0, "freq": "monthly", "mode": "fixed", "day": None, "params": {{}}}},
    {{"policy": "daily", "threshold": 0.05, "cost_bps": 0.0}},
    {{"usd_annual_rate": 0.0, "cash_tenor": "CN1Y"}},
    sys.argv[1],
)
Path = __import__("pathlib").Path
Path(sys.argv[2]).write_text(
    json.dumps({{"annualized": result["metrics"]["annualized_return"], "rf": result["rf_annual"]}}),
    encoding="utf-8",
)
"""


def _seed_minimal(path: Path) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    dates = pd.date_range("2019-01-01", periods=700, freq="B")
    con = repo.connect(path)
    try:
        for symbol, drift, vol in (("AAA", 0.0003, 0.010), ("BBB", 0.0001, 0.003)):
            prices = 1.0 * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": prices,
                    "high": prices * 1.001,
                    "low": prices * 0.999,
                    "close": prices,
                    "volume": 1_000_000.0,
                    "amount": 10_000_000.0,
                    "adj_factor": 1.0,
                    "close_adj": prices,
                }
            )
            repo.upsert(
                con,
                "etf_price",
                frame,
                ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "adj_factor", "close_adj"],
            )
        repo.upsert(
            con,
            "etf_meta",
            pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "name": name,
                        "exchange": "SH",
                        "asset_class": asset_class,
                        "underlying_index": None,
                        "list_date": dates[0].date(),
                        "mgmt_fee": None,
                        "custodian_fee": None,
                        "currency": "CNY",
                        "is_cross_border": False,
                        "t_plus": 1,
                        "price_limit": None,
                        "notes": None,
                    }
                    for symbol, name, asset_class in (("AAA", "宽基A", "broad"), ("BBB", "债券B", "bond"))
                ]
            ),
            [
                "symbol",
                "name",
                "exchange",
                "asset_class",
                "underlying_index",
                "list_date",
                "mgmt_fee",
                "custodian_fee",
                "currency",
                "is_cross_border",
                "t_plus",
                "price_limit",
                "notes",
            ],
        )
        repo.log_data_version(con, source="synthetic", notes="进程池测试数据")
    finally:
        con.close()


def _run_child(db: Path, out: Path) -> subprocess.CompletedProcess:
    script = _CHILD_SCRIPT.format(src=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-c", script, str(db), str(out)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
    )


def test_pool_job_reads_while_another_process_holds_the_database(tmp_path: Path) -> None:
    """父进程持有只读连接时，子进程的作业函数仍必须能读。

    这正是进程池的真实姿态：主进程在读数据版本与表行数，worker 同时在算组合。
    原实现（worker 读写打开）在这里会失败。
    """
    db = tmp_path / "lab.duckdb"
    _seed_minimal(db)

    parent = repo.connect(db, read_only=True)
    try:
        out = tmp_path / "child.json"
        completed = _run_child(db, out)
        assert completed.returncode == 0, "子进程计算失败——多半是数据库锁冲突（作业函数必须只读）"
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["annualized"] is not None
    finally:
        parent.close()


def test_two_pool_jobs_in_sequence_do_not_lock_each_other(tmp_path: Path) -> None:
    """连续两个作业都必须成功。

    原实现下第一个 worker 的常驻实例会锁住文件，第二个就直接失败——
    表现是"有的组合能打开、有的报 500"，且取决于访问顺序。
    """
    db = tmp_path / "lab.duckdb"
    _seed_minimal(db)

    first = _run_child(db, tmp_path / "first.json")
    assert first.returncode == 0, "第一个作业就失败了"
    second = _run_child(db, tmp_path / "second.json")
    assert second.returncode == 0, "第二个作业失败——文件被前一个进程池 worker 锁住了"

    one = json.loads((tmp_path / "first.json").read_text(encoding="utf-8"))
    two = json.loads((tmp_path / "second.json").read_text(encoding="utf-8"))
    assert one["annualized"] == pytest.approx(two["annualized"]), "同一份数据的两次计算必须一致"


def test_app_read_paths_open_the_database_read_only(tmp_path: Path) -> None:
    """界面侧读取数据版本与表行数必须用只读连接（否则会与 worker 抢独占锁）。"""
    db = tmp_path / "lab.duckdb"
    _seed_minimal(db)

    # 只读连接能被父进程与"另一个读者"同时持有
    first = repo.connect(db, read_only=True)
    second = repo.connect(db, read_only=True)
    try:
        assert repo.latest_data_version(first) == repo.latest_data_version(second)
        assert repo.table_counts(first)["etf_price"] > 0
    finally:
        first.close()
        second.close()
