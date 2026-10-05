"""`etf_lab.core`：纯计算区。

硬性契约（全项目不可违反）：
- 本包内**不读数据库、不写文件、不 import UI**；
- 随机过程必须显式接收 ``seed`` 参数，保证可复现；
- 不静默填充缺失值：缺数据就保留 ``NaN`` 并让调用方显式处理；
- 每个数值结论都应能被 ``tests/`` 下的对照测试复现。
"""

from __future__ import annotations

__all__ = ["returns", "metrics", "dca", "correlation"]
