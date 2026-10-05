"""区分两种失败：**窗口本来就没数据** vs **被数据源限流**。

两者都会让请求失败，但处理方式完全相反：
- 前者应当跳过该窗口并继续（ETF 上市前的区间就是空的）；
- 后者必须退避等待，且绝不能把"没拿到"当成"没有数据"——那会造成静默的历史缺失。

判据：同样一个"确定有数据"的近期窗口，如果也失败，说明是限流；
只有"上市前的空窗口"失败，而"上市后的窗口"正常，才说明是空区间问题。

只探测、只打印。
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")

import requests  # noqa: E402

URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

CASES = [
    ("510300 近期（必有数据）", "sh510300,day,2024-01-01,2024-06-30,200,qfq"),
    ("510300 2010-2012（上市前+上市初）", "sh510300,day,2010-01-01,2012-12-31,200,qfq"),
    ("510300 2010-2011（完全在上市前）", "sh510300,day,2010-01-01,2011-12-31,200,qfq"),
    ("588000 2020-2022（上市后有数据）", "sh588000,day,2020-01-01,2022-12-31,200,qfq"),
    ("588000 2010-2011（完全在上市前）", "sh588000,day,2010-01-01,2011-12-31,200,qfq"),
]


def main() -> int:
    session = requests.Session()
    session.trust_env = False
    results = []
    for index, (name, param) in enumerate(CASES):
        entry: dict[str, object] = {"case": name, "param": param}
        started = time.perf_counter()
        try:
            response = session.get(URL, params={"param": param}, headers=HEADERS, timeout=25)
            entry["status"] = response.status_code
            if response.status_code == 200:
                node = (response.json().get("data") or {}).get(param.split(",")[0]) or {}
                rows = node.get("qfqday") or node.get("day") or []
                entry.update(ok=True, rows=len(rows), keys=sorted(node.keys()))
            else:
                entry.update(ok=False, body=response.text[:150])
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:120]}")
        entry["seconds"] = round(time.perf_counter() - started, 2)
        results.append(entry)
        time.sleep(1.5)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
