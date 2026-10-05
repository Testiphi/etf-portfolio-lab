"""多源 K 线接口探测：找出哪些公开接口在**当前网络环境**下真的可用。

为什么需要它
------------
2026-10 实测：``push2his.eastmoney.com``（东财历史主机）在突发请求后被对端
直接断开（TCP 通、HTTP 被 reset），而同域的 ``push2.eastmoney.com`` 实时接口正常。
这说明单一数据源不可靠，采集层必须做**故障转移**。

本脚本只探测、只打印，不写数据库、不写文件。

用法::

    python scripts/probe_kline_sources.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}

START = "2024-01-02"
END = "2024-03-01"
LONG_START = "2010-01-01"


def _session() -> requests.Session:
    """统一禁用环境代理。

    本机系统代理是 127.0.0.1:10808，代理客户端掉线后所有请求都会失败；
    而国内行情源本来就不需要代理，显式绕过才能让失败原因可归因。
    """
    session = requests.Session()
    session.trust_env = False
    return session


def probe_eastmoney_push2his(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params={
            "secid": "1.510300",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",
            "beg": "20240101",
            "end": "20240301",
        },
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise ValueError("klines 为空")
    return {
        "rows": len(klines),
        "name": data.get("name"),
        "first": klines[0].split(",")[0],
        "last": klines[-1].split(",")[0],
        "fields": len(klines[0].split(",")),
    }


def probe_eastmoney_push2(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://push2.eastmoney.com/api/qt/stock/kline/get",
        params={
            "secid": "1.510300",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",
            "beg": "20240101",
            "end": "20240301",
        },
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    text = response.text
    data = response.json().get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise ValueError(f"klines 为空，前 120 字符：{text[:120]}")
    return {"rows": len(klines), "first": klines[0].split(",")[0], "last": klines[-1].split(",")[0]}


def probe_tencent(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"sh510300,day,{START},{END},300,qfq"},
        headers={"User-Agent": HEADERS["User-Agent"]},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    node = (payload.get("data") or {}).get("sh510300") or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise ValueError(f"未取到 K 线，返回键：{sorted(node.keys())}；原始前 120 字符：{response.text[:120]}")
    return {"rows": len(rows), "first": rows[0][0], "last": rows[-1][0], "keys": sorted(node.keys())}


def probe_tencent_long(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"sh510300,day,{LONG_START},2026-10-05,5000,qfq"},
        headers={"User-Agent": HEADERS["User-Agent"]},
        timeout=25,
    )
    response.raise_for_status()
    node = (response.json().get("data") or {}).get("sh510300") or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise ValueError("未取到长历史 K 线")
    return {"rows": len(rows), "first": rows[0][0], "last": rows[-1][0]}


def probe_sina_kline(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketDataService.getKLineData",
        params={"symbol": "sh510300", "scale": "240", "ma": "no", "datalen": "60"},
        headers={"User-Agent": HEADERS["User-Agent"], "Referer": "https://finance.sina.com.cn/"},
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"返回不是列表或为空：{response.text[:120]}")
    return {"rows": len(rows), "first": rows[0].get("day"), "last": rows[-1].get("day")}


def probe_sina_snapshot(session: requests.Session) -> dict[str, Any]:
    response = session.get(
        "https://hq.sinajs.cn/list=sh510300",
        headers={"User-Agent": HEADERS["User-Agent"], "Referer": "https://finance.sina.com.cn/"},
        timeout=20,
    )
    response.raise_for_status()
    if "hq_str_sh510300" not in response.text:
        raise ValueError("快照字段缺失")
    return {"rows": 1, "preview": response.text[:60]}


PROBES: list[tuple[str, Callable[[requests.Session], dict[str, Any]]]] = [
    ("东财 push2his 历史（原主源）", probe_eastmoney_push2his),
    ("东财 push2 实时主机 K 线", probe_eastmoney_push2),
    ("腾讯 fqkline（区间 + 前复权）", probe_tencent),
    ("腾讯 fqkline 长历史", probe_tencent_long),
    ("新浪 K 线", probe_sina_kline),
    ("新浪快照", probe_sina_snapshot),
]


def main() -> int:
    session = _session()
    results: list[dict[str, Any]] = []
    for name, probe in PROBES:
        entry: dict[str, Any] = {"name": name}
        start = time.perf_counter()
        try:
            entry.update(ok=True, **probe(session))
        except Exception as exc:  # noqa: BLE001 - 探测脚本要把失败原因全部打出来
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        entry["seconds"] = round(time.perf_counter() - start, 2)
        results.append(entry)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    ok = [r for r in results if r.get("ok")]
    print(f"\n可用 {len(ok)}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
