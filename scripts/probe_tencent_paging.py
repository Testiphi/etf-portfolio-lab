"""探测腾讯 K 线是否支持"按区间分页"，以拼出完整的前复权长历史。

已知：``count`` 上限约 800-1000 根，一次性要 6000 根会返回空。
若"区间 + 不超过上限的 count"可用，就能通过串联若干窗口得到 2012 年至今的
前复权日线——这是目前唯一被验证可用的前复权长历史路径。

只探测、只打印，不写库、不写文件。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

WINDOWS = [
    ("2010-01-01", "2012-12-31", "800"),
    ("2013-01-01", "2015-12-31", "800"),
    ("2016-01-01", "2018-12-31", "800"),
    ("2019-01-01", "2021-12-31", "800"),
    ("2022-01-01", "2024-12-31", "800"),
    ("2025-01-01", "2026-10-05", "800"),
    ("2010-01-01", "2026-10-05", "800"),
]


def main() -> int:
    session = requests.Session()
    session.trust_env = False
    results: list[dict[str, Any]] = []

    for start, end, count in WINDOWS:
        entry: dict[str, Any] = {"window": f"{start}~{end}", "count": count}
        begin = time.perf_counter()
        try:
            response = session.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": f"sh510300,day,{start},{end},{count},qfq"},
                headers=UA,
                timeout=25,
            )
            response.raise_for_status()
            node = (response.json().get("data") or {}).get("sh510300") or {}
            rows = node.get("qfqday") or node.get("day") or []
            if not rows:
                entry.update(ok=False, error=f"空结果；键={sorted(node.keys())}")
            else:
                entry.update(
                    ok=True,
                    rows=len(rows),
                    first=rows[0][0],
                    last=rows[-1][0],
                    first_close=rows[0][2],
                    last_close=rows[-1][2],
                )
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:150]}")
        entry["seconds"] = round(time.perf_counter() - begin, 2)
        results.append(entry)
        time.sleep(0.4)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    ok = [r for r in results if r.get("ok")]
    print(f"\n可用 {len(ok)}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
