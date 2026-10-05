"""探测候选 ETF 代码在腾讯接口上是否可用，并给出实际上市日期。

用途：``universe.ETF_PRESET`` 里有 3 只标的抓取失败，需要换成**已验证可用**的代码，
而不是继续猜。脚本只打印结果，不修改任何文件。
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from etf_lab.etl import tencent  # noqa: E402

# 候选：中证1000、红利、长期国债三类，各给几个常见代码
CANDIDATES: dict[str, list[str]] = {
    "中证1000ETF": ["512100", "159845", "560010", "159633"],
    "红利ETF": ["515180", "510880", "512890", "159905"],
    "长期国债ETF": ["511260", "511090", "511020", "159972"],
    "其它宽基": ["588000", "159949", "512880", "512480"],
}


def main() -> int:
    print(f"{'类别':<12}{'代码':<10}{'可用':<6}{'名称':<22}{'起始':<12}{'结束':<12}行数")
    with tencent.requests.Session() as session:  # type: ignore[attr-defined]
        session.trust_env = False
        for category, codes in CANDIDATES.items():
            for code in codes:
                try:
                    result = tencent.fetch_daily(code, "etf", "2010-01-01", None, "qfq", session=session, pause=0.0)
                    frame = result.frame
                    print(
                        f"{category:<12}{code:<10}{'是':<6}{result.name:<22}"
                        f"{str(frame['date'].min().date()):<12}{str(frame['date'].max().date()):<12}{len(frame)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"{category:<12}{code:<10}{'否':<6}{type(exc).__name__:<22}{str(exc)[:40]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
