"""搜狐行情接口客户端（**独立校验源**）。

它的价值不在于"再多一个源"，而在于提供**独立口径的历史数据**：
2026-10 实测，搜狐 ``hisHq`` 一次可返回 2012 年至今约 3465 行日线，
且其收盘价与腾讯的**未复权**价完全一致（比值中位数 1.000000），
而与前复权价相差约 5%。因此它可以用来交叉验证原始价是否正确，
也能在腾讯不可用时兜底。

**注意**：搜狐返回的是未复权价，不能直接用于计算收益（分红除权那天的下跌
不是亏损）。它只用于原始价校验与兜底。

接口::

    GET https://q.stock.sohu.com/hisHq?code=cn_510300&start=YYYYMMDD&end=YYYYMMDD
        &stat=1&order=D&period=d&rt=json

行结构（``order=D`` 时为倒序）：
``[日期, 开, 收, 涨跌额, 涨跌幅, 最低, 最高, 成交量, 成交额, 换手率]``，
**收盘价在下标 2**。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import requests

BASE_URL = "https://q.stock.sohu.com/hisHq"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://q.stock.sohu.com/",
}


class SohuError(RuntimeError):
    """接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class SohuResult:
    code: str
    name: str
    frame: pd.DataFrame
    source: str = "sohu"


def symbol_for(code: str, kind: Literal["etf", "index"] = "etf") -> str:
    """搜狐用 ``cn_`` 前缀表示 A 股标的。"""
    code = str(code).strip()
    if kind == "etf":
        if code.startswith(("5", "1")):
            return f"cn_{code}"
        raise SohuError(f"无法判断 ETF {code!r} 的交易所")
    if code.startswith(("000", "399", "801", "999")):
        return f"cn_{code}"
    raise SohuError(f"无法判断指数 {code!r} 的交易所")


def parse_his_payload(payload: Any, code: str) -> tuple[str, pd.DataFrame]:
    """解析搜狐返回（``hq`` 为倒序数组）。"""
    if not isinstance(payload, list) or not payload:
        raise SohuError(f"返回不是非空数组：{str(payload)[:120]}")
    entry = payload[0]
    if not isinstance(entry, dict):
        raise SohuError(f"数组元素不是对象：{str(entry)[:120]}")
    rows = entry.get("hq")
    if not isinstance(rows, list) or not rows:
        raise SohuError(f"hq 为空；status={entry.get('status')}，键={sorted(entry.keys())}")

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            raise SohuError(f"行结构异常（需要至少 8 列）：{row!r}")
        records.append(
            {
                "date": row[0],
                "open": row[1],
                "close": row[2],
                "low": row[5],
                "high": row[6],
                "volume": row[7],
                "amount": row[8] if len(row) > 8 else None,
            }
        )
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise SohuError(f"{code} 存在无法解析的日期")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return str(entry.get("code", code)), frame


def fetch_daily(
    code: str,
    kind: Literal["etf", "index"] = "etf",
    start: str | dt.date = "2010-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> SohuResult:
    """抓取未复权日线（一次请求覆盖整段历史）。

    搜狐会限流并返回 503，因此内置退避重试；调用方在批量比对时仍应留出间隔。
    """
    symbol = symbol_for(code, kind)
    start_arg = pd.Timestamp(start).strftime("%Y%m%d")
    end_arg = pd.Timestamp(end or dt.date.today()).strftime("%Y%m%d")

    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False

    last_error: Exception | None = None
    payload: Any = None
    try:
        for attempt in range(retries):
            try:
                response = client.get(
                    BASE_URL,
                    params={
                        "code": symbol,
                        "start": start_arg,
                        "end": end_arg,
                        "stat": "1",
                        "order": "D",
                        "period": "d",
                        "rt": "json",
                    },
                    headers=HEADERS,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001 - 限流要退避重试
                last_error = exc
                time.sleep(1.2 * (attempt + 1))
        if payload is None:
            raise SohuError(f"{code} 请求失败（已重试 {retries} 次）：{last_error}")
    finally:
        if own_session:
            client.close()

    name, frame = parse_his_payload(payload, code)
    return SohuResult(code=code, name=name, frame=frame)
