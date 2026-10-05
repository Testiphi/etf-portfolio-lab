"""进程池调用封装：把重计算挪出事件循环。

为什么需要它
------------
NiceGUI 是单进程 asyncio 应用，任何同步的 CPU 密集计算都会**阻塞所有在线用户**的
界面响应。官方 FAQ 明确要求这类计算走 ``run.cpu_bound``。

但进程池也有已知的脆弱点（进程池可能被一个坏任务打爆、v3.0.0 上曾整个失效），
所以这里封装了一层：**优先走进程池，失败则退回主进程计算并明确告知**，
而不是让页面直接报错或静默卡死。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class JobOutcome:
    """作业执行结果，附带"它是怎么算出来的"这一信息。"""

    value: Any
    via: str
    """``cpu_bound`` = 子进程；``inline`` = 主进程；``inline_fallback`` = 子进程失败后回退。"""
    note: str | None = None


async def run_heavy(function: Callable[..., Any], *args: Any, **kwargs: Any) -> JobOutcome:
    """优先在子进程执行，失败时在主进程执行。

    ``function`` 必须是模块级自由函数，且参数可 pickle（见 ``services/jobs.py``）。
    """
    try:
        from nicegui import run as nicegui_run

        value = await nicegui_run.cpu_bound(function, *args, **kwargs)
        return JobOutcome(value=value, via="cpu_bound")
    except Exception as exc:  # noqa: BLE001 - 进程池问题不应让页面不可用
        logger.warning("cpu_bound 执行失败，回退到主进程：%s", exc)
        value = function(*args, **kwargs)
        return JobOutcome(
            value=value,
            via="inline_fallback",
            note=f"子进程执行失败，已在主进程完成（{type(exc).__name__}）。长时间计算可能影响其他访问者。",
        )
