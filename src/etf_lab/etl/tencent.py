"""腾讯行情接口客户端（当前**主**数据源）。

2026-10 实测结论（见 ``scripts/probe_*.py``）：

- ``push2his.eastmoney.com`` 在突发请求后会被对端直接断开（TCP 通、HTTP 被 reset）；
- 腾讯 ``fqkline`` 短区间可用，``count`` 上限约 800 根，**一次性要 6000 根会返回空**；
- 但"区间 + count≤800"完全可用，因此按 2 年窗口分页即可拼出完整长历史；
- 腾讯同时提供前复权（``qfq``）与原始价两种口径，便于落库时算 ``adj_factor``。

接口::

    GET https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=<symbol>,day,<start>,<end>,<count>,qfq

``param`` 各段：代码、周期、起始日、结束日、根数、复权方式（``qfq`` / 空）。
返回 ``data.<symbol>.qfqday`` 或 ``data.<symbol>.day``，每行形如
``[日期, 开, 收, 高, 低, 成交量, ...]``——**收盘价在下标 2**，这一点与东财不同，
必须有测试守住。
"""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd
import requests

BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 单次请求的根数上限（实测 800 稳定，超过会返回空）
MAX_BARS_PER_REQUEST = 800
# 分页窗口长度（年）。A 股一年约 245 个交易日，3 年约 735 根，仍在上限内。
WINDOW_YEARS = 3
# 两次请求之间的最小间隔（秒）——**这是硬性要求**：实测连续快速请求会被
# 腾讯 WAF 判定为爬虫，之后所有请求返回 HTTP 501 + JS 挑战页。
MIN_REQUEST_INTERVAL = 1.0

_last_request_at = 0.0


def _throttle(min_interval: float = MIN_REQUEST_INTERVAL) -> None:
    """全局限速：保证任意两次请求之间至少间隔 ``min_interval`` 秒（含随机抖动）。"""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed + random.uniform(0.0, 0.3))
    _last_request_at = time.monotonic()


def _looks_like_challenge(text: str) -> bool:
    """识别反爬挑战页（返回 HTML 而不是 JSON）。"""
    head = text.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")

Adjust = Literal["qfq", "raw"]


class TencentError(RuntimeError):
    """接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class DailyResult:
    """一次日线抓取的结果。"""

    code: str
    symbol: str
    name: str
    adjust: Adjust
    frame: pd.DataFrame
    """列：date, open, high, low, close, volume, amount。"""
    source: str = "tencent"

    @property
    def start(self) -> str:
        return str(self.frame["date"].min().date())

    @property
    def end(self) -> str:
        return str(self.frame["date"].max().date())


def symbol_for(code: str, kind: Literal["etf", "index"] = "etf") -> str:
    """代码 → 腾讯符号（``sh`` / ``sz`` 前缀，港澳美股指数原样使用）。

    只用公开稳定的编码约定，判不出来就报错——猜错会静默返回**另一只标的**的数据，
    这是采集层最危险的失败模式。

    离岸指数（``hkHSI`` 恒生、``hkHSTECH`` 恒生科技等）在腾讯接口里**自带市场前缀**，
    不能加 ``sh``/``sz``，否则会取到别的标的或直接空数据。
    """
    code = str(code).strip()
    if code.startswith(("hk", "us")):
        # 离岸代码自带市场前缀；实测美股指数（usINX/usIXIC/usDJI）这个接口没有日线，
        # 港股指数（hkHSI/hkHSTECH）可用——可用性必须实测，不能假设。
        return code
    if kind == "etf":
        if code.startswith("5"):
            return f"sh{code}"
        if code.startswith("1"):
            return f"sz{code}"
        raise TencentError(f"无法判断 ETF {code!r} 的交易所（应以 5 或 1 开头）")
    if code.startswith("399"):
        return f"sz{code}"
    if code.startswith(("000", "801", "999")):
        return f"sh{code}"
    raise TencentError(f"无法判断指数 {code!r} 的交易所")


def plan_windows(start: dt.date, end: dt.date, years: int = WINDOW_YEARS) -> list[tuple[dt.date, dt.date]]:
    """把区间切成若干不超过 ``years`` 年的窗口。"""
    if start > end:
        raise ValueError("start 不能晚于 end")
    windows: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        try:
            stop = cursor.replace(year=cursor.year + years) - dt.timedelta(days=1)
        except ValueError:  # 2-29 之类
            stop = cursor.replace(year=cursor.year + years, day=28) - dt.timedelta(days=1)
        if stop > end:
            stop = end
        windows.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return windows


def parse_kline_payload(
    payload: dict[str, Any], symbol: str, adjust: Adjust, allow_empty: bool = False
) -> tuple[str, pd.DataFrame]:
    """解析腾讯返回，返回 ``(名称, DataFrame)``。

    收盘价取下标 2（腾讯为 ``日期, 开, 收, 高, 低, 量``）。字段校验用**断言式**失败，
    不做"猜一猜哪一列是收盘价"这种事。

    Parameters
    ----------
    allow_empty
        窗口内没有任何数据（例如该 ETF 尚未上市）时，返回空 DataFrame 而不是报错。
        这是**必需的**：按 2 年窗口分页抓取时，上市前的窗口本来就是空的，
        把它当成错误会让所有 2016 年后上市的 ETF 全部抓取失败——本项目真的踩过这个坑。
        判据是"整个区间都为空"才算失败，由 :func:`fetch_daily` 负责。
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        raise TencentError(f"返回缺少 data 字段：{str(payload)[:120]}")
    node = data.get(symbol)
    if not isinstance(node, dict):
        raise TencentError(f"返回缺少 {symbol} 节点，实际键：{sorted(data.keys())}")

    key = "qfqday" if adjust == "qfq" else "day"
    rows = node.get(key) or node.get("day") or node.get("qfqday") or []
    name = symbol
    qt = node.get("qt")
    if isinstance(qt, dict) and symbol in qt and isinstance(qt[symbol], list) and len(qt[symbol]) > 1:
        name = str(qt[symbol][1])

    if not rows:
        if allow_empty:
            empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
            return name, empty
        raise TencentError(f"{symbol} 的 {key} 为空（键：{sorted(node.keys())}）")

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise TencentError(f"{symbol} 的 K 线行结构异常（需要至少 6 列）：{row!r}")
        records.append(
            {
                "date": row[0],
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": row[5],
                "amount": row[6] if len(row) > 6 else None,
            }
        )

    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        bad = frame[frame["date"].isna()].head(3).to_dict("records")
        raise TencentError(f"{symbol} 存在无法解析的日期：{bad}")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return name, frame


def _request(session: requests.Session, symbol: str, start: dt.date, end: dt.date, adjust: Adjust, count: int) -> dict[str, Any]:
    _throttle()
    suffix = "qfq" if adjust == "qfq" else ""
    param = f"{symbol},day,{start.isoformat()},{end.isoformat()},{count},{suffix}"
    response = session.get(BASE_URL, params={"param": param}, headers=HEADERS, timeout=25)
    text = response.text
    if _looks_like_challenge(text):
        raise TencentError(
            f"请求被反爬拦截（HTTP {response.status_code}，返回 HTML 挑战页）。"
            "请降低请求频率并等待一段时间；不要把它误判成『该标的没有数据』。"
        )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise TencentError(f"返回不是 JSON（前 120 字符）：{text[:120]}") from exc


def fetch_daily(
    code: str,
    kind: Literal["etf", "index"] = "etf",
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
    adjust: Adjust = "qfq",
    session: requests.Session | None = None,
    pause: float = 0.2,
    max_retries: int = 3,
) -> DailyResult:
    """抓取完整日线（自动按窗口分页）。

    Notes
    -----
    分页是必需的：实测一次请求超过约 800 根会返回空。若某个窗口恰好返回满额，
    说明可能被截断，此处会报错而不是**静默丢掉中间的行情**——少一段行情会
    直接改变回测结论。
    """
    symbol = symbol_for(code, kind)
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date() if end is not None else dt.date.today()
    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False

    chunks: list[pd.DataFrame] = []
    name = symbol
    try:
        for index, (window_start, window_end) in enumerate(plan_windows(start_date, end_date)):
            last_error: Exception | None = None
            payload: dict[str, Any] | None = None
            for attempt in range(max_retries):
                try:
                    payload = _request(client, symbol, window_start, window_end, adjust, MAX_BARS_PER_REQUEST)
                    break
                except Exception as exc:  # noqa: BLE001 - 网络抖动要重试
                    last_error = exc
                    time.sleep(0.5 * (attempt + 1))
            if payload is None:
                raise TencentError(f"{symbol} {window_start}~{window_end} 请求失败：{last_error}")

            name, frame = parse_kline_payload(payload, symbol, adjust, allow_empty=True)
            if len(frame) >= MAX_BARS_PER_REQUEST:
                raise TencentError(
                    f"{symbol} {window_start}~{window_end} 返回满额 {len(frame)} 根，疑似被截断；"
                    "请缩短 WINDOW_YEARS 而不是接受一段缺失的行情"
                )
            if not frame.empty:
                chunks.append(frame)
            if pause and index < len(plan_windows(start_date, end_date)) - 1:
                time.sleep(pause)
    finally:
        if own_session:
            client.close()

    if not chunks:
        raise TencentError(
            f"{symbol} 在 {start_date}~{end_date} 的所有窗口都没有数据（标的可能尚未上市或代码有误）"
        )
    merged = pd.concat(chunks, ignore_index=True).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return DailyResult(code=code, symbol=symbol, name=name, adjust=adjust, frame=merged)


def fetch_daily_batch(
    codes: Sequence[str],
    kind: Literal["etf", "index"] = "etf",
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
    adjust: Adjust = "qfq",
    pause: float = 0.3,
) -> list[DailyResult | Exception]:
    """顺序抓取多个标的；单个失败返回异常对象而不中断整批。

    刻意复用同一个 ``Session`` 并加小停顿，避免像本项目第一次那样
    因突发请求被对端限流。
    """
    out: list[DailyResult | Exception] = []
    with requests.Session() as session:
        session.trust_env = False
        for index, code in enumerate(codes):
            try:
                out.append(
                    fetch_daily(code, kind=kind, start=start, end=end, adjust=adjust, session=session, pause=pause)
                )
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
            if index < len(codes) - 1:
                time.sleep(pause)
    return out
