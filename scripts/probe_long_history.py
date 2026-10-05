"""长历史行情源探测：为多源故障转移挑选可用的"历史主源"。

背景：2026-10 实测 ``push2his.eastmoney.com`` 在突发请求后被断开，
腾讯 ``fqkline`` 的短区间可用但长区间返回空。本项目需要 2010 年以来的日线，
所以必须逐个验证长历史接口的参数形式与实际上限。

只探测、只打印，不写数据库、不写文件。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # 国内行情源不需要代理；本机代理掉线会让失败原因被掩盖
    return session


def _tencent(session: requests.Session, param: str) -> dict[str, Any]:
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": param},
        headers=UA,
        timeout=25,
    )
    response.raise_for_status()
    node = (response.json().get("data") or {}).get("sh510300") or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise ValueError(f"空结果；返回键={sorted(node.keys())}")
    return {"rows": len(rows), "first": rows[0][0], "last": rows[-1][0]}


def tencent_count_320(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,,,320,qfq")


def tencent_count_800(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,,,800,qfq")


def tencent_count_2000(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,,,2000,qfq")


def tencent_count_6000(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,,,6000,qfq")


def tencent_range_2010_2020(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,2010-01-01,2020-12-31,3000,qfq")


def tencent_noadjust_long(session: requests.Session) -> dict[str, Any]:
    return _tencent(session, "sh510300,day,2010-01-01,2026-10-05,6000")


def sohu_his(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://q.stock.sohu.com/hisHq",
        params={
            "code": "cn_510300",
            "start": "20100101",
            "end": "20261005",
            "stat": "1",
            "order": "D",
            "period": "d",
            "rt": "json",
        },
        headers={**UA, "Referer": "https://q.stock.sohu.com/"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"返回异常：{response.text[:120]}")
    entry = payload[0]
    rows = entry.get("hq") or []
    if not rows:
        raise ValueError(f"hq 为空；键={sorted(entry.keys())}；status={entry.get('status')}")
    return {"rows": len(rows), "first": rows[-1][0], "last": rows[0][0], "columns": len(rows[0])}


def netease_csv(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://quotes.money.163.com/service/chddata.html",
        params={
            "code": "0510300",
            "start": "20100101",
            "end": "20261005",
            "fields": "TCLOSE;HIGH;LOW;TOPEN;VOTURNOVER;VATURNOVER",
        },
        headers={**UA, "Referer": "https://quotes.money.163.com/"},
        timeout=30,
    )
    response.raise_for_status()
    lines = [line for line in response.text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"CSV 行数不足：{len(lines)}；前 120 字符：{response.text[:120]}")
    header = lines[0].split(",")
    return {"rows": len(lines) - 1, "columns": len(header), "sample_first": lines[-1][:80], "sample_last": lines[1][:80]}


PROBES: list[tuple[str, Callable[[requests.Session], dict[str, Any]]]] = [
    ("腾讯 count=320", tencent_count_320),
    ("腾讯 count=800", tencent_count_800),
    ("腾讯 count=2000", tencent_count_2000),
    ("腾讯 count=6000", tencent_count_6000),
    ("腾讯 区间 2010-2020 count=3000", tencent_range_2010_2020),
    ("腾讯 不复权 2010-2026 count=6000", tencent_noadjust_long),
    ("搜狐 hisHq 全历史", sohu_his),
    ("网易 chddata CSV 全历史", netease_csv),
]


def main() -> int:
    session = _session()
    results: list[dict[str, Any]] = []
    for name, probe in PROBES:
        entry: dict[str, Any] = {"name": name}
        start = time.perf_counter()
        try:
            entry.update(ok=True, **probe(session))
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:180]}")
        entry["seconds"] = round(time.perf_counter() - start, 2)
        results.append(entry)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n可用 {sum(1 for r in results if r.get('ok'))}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
