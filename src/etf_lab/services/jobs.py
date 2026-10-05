"""可放进进程池执行的作业函数。

NiceGUI 的 ``run.cpu_bound`` 用 pickle 把函数与参数送到子进程，因此有一条硬性契约
（见官方 FAQ）：**必须是模块级自由函数，参数只能是简单可 pickle 的对象，
不能访问类属性、闭包或 UI 状态**。这里的每个函数都遵守这条契约：
入参是字符串/数字/字典，出参是纯字典。

把作业函数单独放在一个模块里还有一个好处：它可以脱离界面直接用 pytest 调用。
"""

from __future__ import annotations

from typing import Any, Mapping

from etf_lab.presets import PortfolioSpec


def compute_preset_job(spec_key: str, db_path: str | None = None, rf_annual: float = 0.02) -> dict[str, Any]:
    """计算一个预设组合（子进程安全）。"""
    from etf_lab.data import repo
    from etf_lab.presets import get
    from etf_lab.reports.static_site import compute_preset

    con = repo.connect(db_path)
    try:
        return compute_preset(con, get(spec_key), rf_annual=rf_annual)
    finally:
        con.close()


def compute_custom_job(
    weights: Mapping[str, float],
    dca_amount: float = 2000.0,
    dca_mode: str = "fixed",
    db_path: str | None = None,
    rf_annual: float = 0.02,
) -> dict[str, Any]:
    """计算用户自定义权重的组合：复用与示例组合**完全相同**的计算路径。

    这一点很重要——实验室页面与示例页面如果走两套代码，数字就会不一致，
    那时"教学"反而变成了误导。
    """
    from etf_lab.data import repo
    from etf_lab.reports.static_site import compute_preset

    spec = PortfolioSpec(
        key="custom",
        name="自定义组合",
        question="你调整权重后，风险与收益各自变成了什么？",
        weights={k: float(v) for k, v in weights.items() if float(v) > 0},
        dca={"amount": float(dca_amount), "freq": "monthly", "mode": dca_mode, "day": None},
    )
    con = repo.connect(db_path)
    try:
        return compute_preset(con, spec, rf_annual=rf_annual)
    finally:
        con.close()
