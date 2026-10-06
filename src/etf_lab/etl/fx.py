"""汇率采集：新浪中行牌价历史，取**央行中间价**。

为什么用央行中间价，而不是市场即期汇率或中行牌价
------------------------------------------------
* 中行牌价分「汇买价 / 钞买价 / 汇卖价 / 钞卖价」，**含买卖价差**。
  用买入价算收益，等于把价差当成汇率波动，会系统性低估持有外币的收益。
* 央行中间价是官方按日发布的参考价、**无买卖价差**，适合度量"汇率变动本身"。
* 它也是跨境资产收益拆分的标准口径：跨境 ETF 的收益 ≈ 标的涨跌 + 汇率变动 + 跟踪误差。

实测细节（都是踩出来的）
------------------------
* 源站以「元/100 外币」计价（如 ``702.88`` 表示 7.0288 元/美元），**必须除以 100**；
* 一页 50 行，需要按 `page` 分页；**必须带 ``call_type=ajax``**，否则返回整页 HTML
  （含导航表），表格下标就变了——这个坑让第一版探测全部报越界；
* 最新几天中间价可能是 ``--`` 占位，必须剔除，不能当 0。
"""

from __future__ import annotations

import datetime as dt
import io
import re
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

FX_URL = "http://biz.finance.sina.com.cn/forex/forex.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
RATE_COLUMN = "央行中间价"
"""取央行中间价，而不是任一牌价列。"""

PAIR_CODES: dict[str, str] = {
    "USDCNY": "USD",
    "HKDCNY": "HKD",
    "EURCNY": "EUR",
    "JPYCNY": "JPY",
    "GBPCNY": "GBP",
    "AUDCNY": "AUD",
}
"""内部 pair 名 → 源站 money_code。日元等按 100 单位计价，同样除以 100。"""

MIN_REQUEST_INTERVAL = 0.8
PAGE_ROWS = 50
_last_request_at = 0.0


class FxError(RuntimeError):
    """接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class FxResult:
    frame: pd.DataFrame
    source: str


def _throttle(min_interval: float = MIN_REQUEST_INTERVAL) -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_at = time.monotonic()


def _normalize_header(frame: pd.DataFrame) -> pd.DataFrame:
    """统一表头形态。

    真实页面**没有 ``<thead>``**，``read_html`` 会把表头当第一行数据（列名变成 0,1,2…）；
    而某些页面/夹具带 ``<thead>``，``read_html`` 已经把表头当列名。
    两种形态都得认，否则同一份解析代码在一处通、在另一处把首行数据吃掉。
    """
    columns = [str(column) for column in frame.columns]
    if "日期" in columns and RATE_COLUMN in columns:
        return frame.reset_index(drop=True)
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.iloc[0]]
    return frame.iloc[1:].reset_index(drop=True)


def _pick_rate_table(html: str) -> pd.DataFrame:
    """在所有表里找出含「央行中间价」的那张（ajax 片段一张、整页两张，不能按下标取）。"""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError as exc:
        raise FxError(f"页面里没有可解析的表格：{exc}") from exc
    for table in tables:
        first_row = [str(value) for value in table.iloc[0].tolist()] if len(table) else []
        columns = [str(column) for column in table.columns]
        if any(RATE_COLUMN in value for value in first_row + columns):
            return _normalize_header(table)
    raise FxError("未找到含央行中间价的表格，页面结构可能已变")


def parse_fx_html(html: str, pair: str = "USDCNY") -> pd.DataFrame:
    """解析一页牌价 → ``(pair, date, close)``，close 为「元/外币」。"""
    frame = _pick_rate_table(html)
    if "日期" not in frame.columns or RATE_COLUMN not in frame.columns:
        raise FxError(f"表结构异常，实际列：{list(frame.columns)}")
    out = pd.DataFrame(
        {
            "pair": pair,
            "date": pd.to_datetime(frame["日期"], errors="coerce"),
            # 源站是「元/100 外币」，且可能用 '--' 占位
            "close": pd.to_numeric(frame[RATE_COLUMN], errors="coerce") / 100.0,
        }
    )
    out = out.dropna(subset=["date", "close"])
    if out.empty:
        raise FxError("解析后没有任何有效汇率记录（可能整页都是占位符）")
    return out.sort_values("date").reset_index(drop=True)


def _page_count(html: str) -> int:
    pages = [int(value) for value in re.findall(r'class="page"[^>]*>(\d+)<', html)]
    return max(pages) if pages else 1


def fetch_pair(
    pair: str = "USDCNY",
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """抓一个货币对的完整历史（按年分窗 + 逐页翻）。"""
    code = PAIR_CODES.get(pair)
    if code is None:
        raise FxError(f"未知货币对 {pair!r}；已支持 {sorted(PAIR_CODES)}")
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end or dt.date.today()).date()
    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False

    frames: list[pd.DataFrame] = []
    try:
        year = start_date.year
        while year <= end_date.year:
            window_start = max(start_date, dt.date(year, 1, 1))
            window_end = min(end_date, dt.date(year, 12, 31))
            base_params = {
                "startdate": window_start.isoformat(),
                "enddate": window_end.isoformat(),
                "money_code": code,
                "type": "0",
                # 必须带 call_type=ajax：否则返回整页 HTML，表格下标会变
                "call_type": "ajax",
            }
            _throttle()
            first = client.get(FX_URL, params={**base_params, "page": "1"}, headers=HEADERS, timeout=30)
            first.raise_for_status()
            first.encoding = "gbk"
            total_pages = _page_count(first.text)
            frames.append(parse_fx_html(first.text, pair))
            for page in range(2, total_pages + 1):
                _throttle()
                response = client.get(FX_URL, params={**base_params, "page": str(page)}, headers=HEADERS, timeout=30)
                response.raise_for_status()
                response.encoding = "gbk"
                frames.append(parse_fx_html(response.text, pair))
            year += 1
    finally:
        if own_session:
            client.close()

    if not frames:
        raise FxError("没有取到任何汇率数据")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(["pair", "date"])
    return merged.sort_values("date").reset_index(drop=True)


def fetch_rates(
    pairs: Iterable[str] = ("USDCNY",),
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
) -> FxResult:
    """抓多个货币对；任一失败就整体报错（不静默返回部分数据）。"""
    frames = [fetch_pair(pair, start, end, session) for pair in pairs]
    return FxResult(frame=pd.concat(frames, ignore_index=True), source="sina-boc-parity")
