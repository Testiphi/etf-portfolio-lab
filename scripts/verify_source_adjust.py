"""校验数据源的复权口径是否一致。

这是本项目最关键的数据校验：**前复权价与未复权价算出的收益差别很大**
（分红除权那天的"下跌"其实不是亏损）。如果两条来源的口径不同，
把它们拼在一起就会凭空制造出一段不存在的收益。

只用公开接口做对比，不写库、不写文件。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CHECK_DATES = ["2024-01-02", "2024-06-03", "2025-01-02", "2026-09-30"]


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def sohu_history(session: requests.Session) -> tuple[list[str], dict[str, list[str]]]:
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
    entry = response.json()[0]
    stat = [str(c) for c in entry.get("stat", [])]
    rows = entry.get("hq") or []
    by_date = {str(row[0]): [str(cell) for cell in row] for row in rows}
    return stat, by_date


def tencent_qfq(session: requests.Session) -> dict[str, list[str]]:
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": "sh510300,day,,,800,qfq"},
        headers=UA,
        timeout=25,
    )
    response.raise_for_status()
    node = (response.json().get("data") or {}).get("sh510300") or {}
    rows = node.get("qfqday") or node.get("day") or []
    return {str(row[0]): [str(cell) for cell in row] for row in rows}


def tencent_raw(session: requests.Session) -> dict[str, list[str]]:
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": "sh510300,day,,,800,"},
        headers=UA,
        timeout=25,
    )
    response.raise_for_status()
    node = (response.json().get("data") or {}).get("sh510300") or {}
    rows = node.get("day") or []
    return {str(row[0]): [str(cell) for cell in row] for row in rows}


def main() -> int:
    session = _session()
    sohu_stat, sohu = sohu_history(session)
    qfq = tencent_qfq(session)
    raw = tencent_raw(session)

    print("搜狐 stat 列名：", json.dumps(sohu_stat, ensure_ascii=False))
    print("腾讯 qfq 行数：", len(qfq), " 腾讯未复权行数：", len(raw), " 搜狐行数：", len(sohu))
    print()

    header = f"{'日期':<12}{'搜狐行':<60}{'腾讯qfq收':>10}{'腾讯原始收':>12}"
    print(header)
    for date in CHECK_DATES:
        line = f"{date:<12}{','.join(sohu.get(date, ['-']))[:58]:<60}{qfq.get(date, ['-'])[2] if date in qfq else '-':>10}{raw.get(date, ['-'])[2] if date in raw else '-':>12}"
        print(line)

    # 用重叠区间的收盘价做比值，判断搜狐的口径更接近哪一个
    print("\n重叠区间比值（腾讯qfq / 搜狐 与 腾讯原始 / 搜狐 的中位数，越接近 1 越可能是同一口径）：")
    ratios_qfq: list[float] = []
    ratios_raw: list[float] = []
    for date, row in sohu.items():
        if len(row) < 3:
            continue
        try:
            sohu_close = float(row[2])
        except ValueError:
            continue
        if date in qfq and sohu_close:
            try:
                ratios_qfq.append(float(qfq[date][2]) / sohu_close)
            except (ValueError, ZeroDivisionError):
                pass
        if date in raw and sohu_close:
            try:
                ratios_raw.append(float(raw[date][2]) / sohu_close)
            except (ValueError, ZeroDivisionError):
                pass

    def median(values: list[float]) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    print(
        json.dumps(
            {
                "overlap_with_qfq": len(ratios_qfq),
                "median_tencent_qfq_over_sohu": round(median(ratios_qfq), 6),
                "overlap_with_raw": len(ratios_raw),
                "median_tencent_raw_over_sohu": round(median(ratios_raw), 6),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
