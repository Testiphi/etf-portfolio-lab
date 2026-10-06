"""可放进进程池执行的作业函数。

NiceGUI 的 ``run.cpu_bound`` 用 pickle 把函数与参数送到子进程，因此有一条硬性契约
（见官方 FAQ）：**必须是模块级自由函数，参数只能是简单可 pickle 的对象，
不能访问类属性、闭包或 UI 状态**。这里的每个函数都遵守这条契约：
入参是字符串/数字/字典，出参是纯字典。

**连接必须只读**（``read_only=True``）——这一条是实测踩出来的，
不写清楚下次一定还会踩：

* DuckDB 里只要有**任何进程**以读写方式打开文件，就取独占锁；
* 更麻烦的是，**数据库实例在进程内是常驻的**：进程池的 worker 算完一个任务后，
  即使 ``con.close()``，该实例仍持有文件锁，直到进程退出；
* 于是"第一个 worker 算完了、第二个 worker 打开失败"，
  报错是 ``IOException: 另一个程序正在使用此文件``；
* 而**多个进程以只读方式打开同一文件是允许的**（共享锁）。

计算只读数据、结果缓存在主进程的内存字典里，因此作业函数没有任何写需求。
只有 ETL 采集（``cli.py fetch``）才需要读写，而那是独立进程。
"""

from __future__ import annotations

from typing import Any, Mapping

from etf_lab.presets import PortfolioSpec


def compute_preset_job(spec_key: str, db_path: str | None = None, rf_annual: float | None = None) -> dict[str, Any]:
    """计算一个预设组合（子进程安全）。"""
    from etf_lab.data import repo
    from etf_lab.presets import get
    from etf_lab.reports.static_site import compute_preset

    con = repo.connect(db_path, read_only=True)
    try:
        return compute_preset(con, get(spec_key), rf_annual=rf_annual)
    finally:
        con.close()


def compute_custom_job(
    weights: Mapping[str, float],
    dca: Mapping[str, Any] | None = None,
    rebalance: Mapping[str, Any] | None = None,
    cash: Mapping[str, Any] | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """计算用户自定义权重的组合：复用与示例组合**完全相同**的计算路径。

    这一点很重要——实验室页面与示例页面如果走两套代码，数字就会不一致，
    那时"教学"反而变成了误导。

    参数用字典而不是一长串位置参数：选项会越来越多（定投 9 个参数、再平衡 3 个、
    现金 2 个），位置参数一多没人记得住顺序，而 pickle 传字典没有任何代价。
    """
    from etf_lab.data import repo
    from etf_lab.reports.static_site import compute_preset

    dca_payload = dict(dca or {})
    spec = PortfolioSpec(
        key="custom",
        name="自定义组合",
        question="你调整权重后，风险与收益各自变成了什么？",
        weights={k: float(v) for k, v in weights.items() if float(v) > 0},
        dca={
            "amount": float(dca_payload.get("amount") or 2000.0),
            "freq": str(dca_payload.get("freq") or "monthly"),
            "mode": str(dca_payload.get("mode") or "fixed"),
            "day": dca_payload.get("day"),
            "params": dict(dca_payload.get("params") or {}),
        },
        rebalance=dict(rebalance or {}),
        cash=dict(cash or {}),
    )
    con = repo.connect(db_path, read_only=True)
    try:
        return compute_preset(con, spec)
    finally:
        con.close()
