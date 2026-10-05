"""国债收益率曲线采集（中债官方主源 + 新浪兜底）。

为什么需要它
------------
1. **无风险利率不能再靠假设**。夏普、索提诺、卡玛都要减无风险利率，
   此前页面用的是"教学场景固定 2%"，那是个不诚实的简化。
2. **久期分析的基础**。有了收益率曲线，才能用"债券 ETF 收益对收益率变动的回归"
   反推它的修正久期，进而做利率冲击情景表——这是 A 股场内投资者唯一能做的利率风险度量。

两个源的实测差异
----------------
+----------------------+--------------------------------+--------------------------------+
| 源                   | 形态                            | 实测                            |
+======================+================================+================================+
| 中债 yield.chinabond | HTML 表格，**区间必须小于一年**  | 200 OK；1 年窗口 745 行         |
| 新浪 bond.finance    | JSON，每个期限一个请求           | 200 OK；CN10YT 1000 行          |
+----------------------+--------------------------------+--------------------------------+

中债源有两个必须处理的结构特征（都是实测踩出来的）：

1. 页面里有多张表，数据在第 2 张（``index=1``）；
2. 那张表把**多条曲线堆叠在一起**（国债 / 国开债 / 地方债，745 行 ÷ 248 个交易日 ≈ 3 条），
   必须按「曲线名称」过滤，否则同一日期会被塞进三行不同曲线的数据。
"""

from __future__ import annotations

import datetime as dt
import io
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests

CHINABOND_URL = "https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery"
SINA_URL = "https://bond.finance.sina.com.cn/hq/gb/daily"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://yield.chinabond.com.cn/",
}
CURVE_NAME = "中债国债收益率曲线"
"""中债表里堆叠多条曲线，只取国债那条。"""

TENOR_CODES: dict[str, str] = {
    "3月": "CN3M",
    "6月": "CN6M",
    "1年": "CN1Y",
    "2年": "CN2Y",
    "3年": "CN3Y",
    "5年": "CN5Y",
    "7年": "CN7Y",
    "10年": "CN10Y",
    "30年": "CN30Y",
}
TENOR_YEARS: dict[str, float] = {
    "CN3M": 0.25,
    "CN6M": 0.5,
    "CN1Y": 1.0,
    "CN2Y": 2.0,
    "CN3Y": 3.0,
    "CN5Y": 5.0,
    "CN7Y": 7.0,
    "CN10Y": 10.0,
    "CN30Y": 30.0,
}
SINA_SYMBOLS: dict[str, str] = {
    "CN1Y": "CN1YT",
    "CN2Y": "CN2YT",
    "CN3Y": "CN3YT",
    "CN5Y": "CN5YT",
    "CN7Y": "CN7YT",
    "CN10Y": "CN10YT",
    "CN30Y": "CN30YT",
}
WINDOW_DAYS = 360
"""中债接口要求查询区间小于一年，因此按 360 天分窗。"""
MIN_REQUEST_INTERVAL = 0.8
_last_request_at = 0.0


class BondYieldError(RuntimeError):
    """接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class CurveResult:
    frame: pd.DataFrame
    """长表：date / code / tenor / yield。"""
    source: str


def _throttle(min_interval: float = MIN_REQUEST_INTERVAL) -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_at = time.monotonic()


def parse_chinabond_html(html_text: str) -> pd.DataFrame:
    """解析中债 historyQuery 的 HTML → 长表 ``(date, code, tenor, yield)``。

    步骤：取第 2 张表 → 按曲线名称过滤出国债曲线 → 宽表转长表 → 数值化。
    任何一步不满足预期都**明确报错**，不做"猜哪一列是收益率"这种事。
    """
    tables = pd.read_html(io.StringIO(html_text.replace("&nbsp", "")), header=0)
    if len(tables) < 2:
        raise BondYieldError(f"页面表格数不足（{len(tables)}），接口结构可能已变")
    table = tables[1]
    if "曲线名称" not in table.columns or "日期" not in table.columns:
        raise BondYieldError(f"表结构异常，实际列：{list(table.columns)[:12]}")

    matched = table[table["曲线名称"].astype(str).str.strip() == CURVE_NAME]
    if matched.empty:
        names = sorted(set(table["曲线名称"].astype(str).str.strip()))
        raise BondYieldError(f"未找到「{CURVE_NAME}」，页面中的曲线有：{names}")

    tenor_columns = [column for column in matched.columns if str(column) in TENOR_CODES]
    if not tenor_columns:
        raise BondYieldError(f"未找到任何期限列，实际列：{list(matched.columns)}")

    long = matched.melt(
        id_vars=["日期"],
        value_vars=tenor_columns,
        var_name="tenor_label",
        value_name="yield",
    )
    long["date"] = pd.to_datetime(long["日期"], errors="coerce")
    long["code"] = long["tenor_label"].map(TENOR_CODES)
    long["tenor"] = long["tenor_label"].astype(str)
    long["yield"] = pd.to_numeric(long["yield"], errors="coerce")
    long = long.dropna(subset=["date", "yield", "code"])
    if long.empty:
        raise BondYieldError("过滤后没有任何有效收益率记录")
    return long[["date", "code", "tenor", "yield"]].sort_values(["code", "date"]).reset_index(drop=True)


def parse_sina_payload(payload: dict[str, Any], code: str) -> pd.DataFrame:
    """解析新浪单期限返回 → 长表。"""
    rows = ((payload.get("result") or {}).get("data")) or []
    if not rows:
        raise BondYieldError(f"新浪 {code} 返回为空")
    frame = pd.DataFrame(rows)
    if "d" not in frame.columns or "c" not in frame.columns:
        raise BondYieldError(f"新浪返回字段异常：{list(frame.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["d"], errors="coerce"),
            "code": code,
            "tenor": {value: key for key, value in TENOR_CODES.items()}.get(code, code),
            "yield": pd.to_numeric(frame["c"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "yield"])
    if out.empty:
        raise BondYieldError(f"新浪 {code} 解析后无有效记录")
    return out.sort_values("date").reset_index(drop=True)


def _windows(start: dt.date, end: dt.date, days: int = WINDOW_DAYS) -> list[tuple[dt.date, dt.date]]:
    out: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + dt.timedelta(days=days), end)
        out.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return out


def fetch_chinabond(
    start: str | dt.date = "2015-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """按小于一年的窗口分页拉取中债国债收益率曲线。"""
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end or dt.date.today()).date()
    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False

    frames: list[pd.DataFrame] = []
    try:
        for window_start, window_end in _windows(start_date, end_date):
            _throttle()
            response = client.get(
                CHINABOND_URL,
                params={
                    "startDate": window_start.isoformat(),
                    "endDate": window_end.isoformat(),
                    "gjqx": "0",
                    "qxId": "ycqx",
                    "locale": "cn_ZH",
                },
                headers=HEADERS,
                timeout=40,
            )
            response.raise_for_status()
            frames.append(parse_chinabond_html(response.text))
    finally:
        if own_session:
            client.close()

    if not frames:
        raise BondYieldError("中债接口没有返回任何数据")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    return merged.sort_values(["code", "date"]).reset_index(drop=True)


def fetch_sina(codes: Iterable[str] = ("CN1Y", "CN3Y", "CN5Y", "CN10Y", "CN30Y"), session: requests.Session | None = None) -> pd.DataFrame:
    """新浪兜底：逐个期限请求（每个期限约返回最近 1000 个交易日）。"""
    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False
    frames: list[pd.DataFrame] = []
    try:
        for code in codes:
            symbol = SINA_SYMBOLS.get(code)
            if symbol is None:
                continue
            _throttle()
            response = client.get(SINA_URL, params={"symbol": symbol}, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)
            response.raise_for_status()
            frames.append(parse_sina_payload(response.json(), code))
    finally:
        if own_session:
            client.close()
    if not frames:
        raise BondYieldError("新浪没有返回任何期限的数据")
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)


def fetch_curve(
    start: str | dt.date = "2015-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
) -> CurveResult:
    """主源失败时自动落到兜底源；两个都失败则抛错（不静默返回空表）。"""
    try:
        return CurveResult(frame=fetch_chinabond(start, end, session), source="chinabond")
    except Exception as primary_error:  # noqa: BLE001 - 主源失败要落到兜底源
        try:
            return CurveResult(frame=fetch_sina(session=session), source="sina")
        except Exception as fallback_error:  # noqa: BLE001
            raise BondYieldError(f"主源与兜底源均失败：{primary_error}；兜底：{fallback_error}") from primary_error
