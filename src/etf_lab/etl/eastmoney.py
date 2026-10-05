"""东方财富公开行情接口的极薄客户端。

为什么不用 akshare 作为主数据路径
--------------------------------
1. akshare 依赖 ``py_mini_racer``（内含 V8 引擎二进制），杀毒软件会误报并**直接杀进程**
   ——本项目实际就被卡巴斯基中断过一次，且是在跑采集的时候；
2. 路线 B（Pyodide 浏览器内计算）本来就无法使用抓取型库，数据必须预烘焙，
   所以主数据路径越靠近"纯 HTTP + 纯 Python"越好；
3. 少一层依赖 = 少一处会因为上游改版而静默出错的地方。

本模块只做三件事：拼请求、解析、返回规范化的 DataFrame。
**不写数据库、不写文件**——落库由 ``etl/fetch.py`` 负责。

接口说明（东财 K 线）
---------------------
``GET https://push2his.eastmoney.com/api/qt/stock/kline/get``

- ``secid``：市场前缀 + 代码，``1.`` = 上交所，``0.`` = 深交所
- ``klt``：101 = 日线
- ``fqt``：0 = 不复权，1 = 前复权，2 = 后复权
- ``fields2``：f51 日期, f52 开, f53 收, f54 高, f55 低, f56 成交量, f57 成交额,
  f58 振幅, f59 涨跌幅, f60 涨跌额, f61 换手率

返回的 ``data.klines`` 是逗号分隔的字符串数组，字段顺序与 ``fields2`` 一一对应。
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd
import requests

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

FIELDS1 = "f1,f2,f3,f4,f5,f6"
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}

Adjust = Literal["none", "qfq", "hfq"]
_FQT = {"none": 0, "qfq": 1, "hfq": 2}


class EastmoneyError(RuntimeError):
    """接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class KlineResult:
    """一次 K 线请求的结果。"""

    code: str
    secid: str
    name: str
    adjust: Adjust
    frame: pd.DataFrame


def secid_for(code: str, kind: Literal["etf", "index"] = "etf") -> str:
    """代码 → ``secid``。

    规则只用公开且稳定的编码约定，不做模糊猜测：
    - ETF：``5`` 开头在上交所，``1`` 开头在深交所；
    - 指数：``399`` 开头是深证/国证系列（深交所），其余 ``000`` 开头为中证/上证系列（上交所）。
    无法判定时直接报错，而不是猜一个默认值——猜错会静默返回另一只标的的数据。
    """
    code = str(code).strip()
    if kind == "etf":
        if code.startswith("5"):
            return f"1.{code}"
        if code.startswith("1"):
            return f"0.{code}"
        raise EastmoneyError(f"无法判断 ETF {code!r} 的交易所（应以 5 或 1 开头）")
    if code.startswith("399"):
        return f"0.{code}"
    if code.startswith(("000", "880", "999")):
        return f"1.{code}"
    raise EastmoneyError(f"无法判断指数 {code!r} 的交易所")


def _date_arg(value: str | dt.date | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def fetch_kline(
    code: str,
    kind: Literal["etf", "index"] = "etf",
    start: str | dt.date = "2010-01-01",
    end: str | dt.date | None = None,
    adjust: Adjust = "qfq",
    session: requests.Session | None = None,
    retries: int = 3,
    timeout: float = 25.0,
) -> KlineResult:
    """抓取单只标的的日线。

    Parameters
    ----------
    adjust
        ``none`` 返回原始价，``qfq`` 返回前复权价，``hfq`` 返回后复权价。
        本项目**只用 ``qfq`` 计算收益**，``none`` 仅用于还原 ``adj_factor``。

    Raises
    ------
    EastmoneyError
        重试后仍失败，或返回内容为空 / 字段数不符。
    """
    secid = secid_for(code, kind)
    params = {
        "secid": secid,
        "fields1": FIELDS1,
        "fields2": FIELDS2,
        "klt": "101",
        "fqt": str(_FQT[adjust]),
        "beg": _date_arg(start),
        "end": _date_arg(end or dt.date.today()),
        "rtntype": "6",
    }
    client = session or requests
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = client.get(KLINE_URL, params=params, headers=_HEADERS, timeout=timeout)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            break
        except Exception as exc:  # noqa: BLE001 - 网络层失败要重试，重试后仍失败再抛
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    else:
        raise EastmoneyError(f"{code} 请求失败（已重试 {retries} 次）：{last_error}")

    data = payload.get("data")
    if not data:
        raise EastmoneyError(f"{code} 返回 data 为空（可能代码不存在或已退市）：{payload.get('rc')}")
    klines = data.get("klines") or []
    if not klines:
        raise EastmoneyError(f"{code} 返回的 klines 为空")

    rows = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) != len(KLINE_COLUMNS):
            raise EastmoneyError(f"{code} 的 K 线字段数为 {len(parts)}，预期 {len(KLINE_COLUMNS)}：{line!r}")
        rows.append(parts)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in KLINE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)

    return KlineResult(code=code, secid=secid, name=str(data.get("name", code)), adjust=adjust, frame=frame)


def fetch_kline_batch(
    codes: Sequence[str],
    kind: Literal["etf", "index"] = "etf",
    start: str | dt.date = "2010-01-01",
    end: str | dt.date | None = None,
    adjust: Adjust = "qfq",
    pause: float = 0.15,
) -> list[KlineResult | Exception]:
    """顺序抓取多个标的，复用连接；单个失败返回异常对象而不中断整批。

    返回列表里混着结果与异常是刻意的：采集层需要"部分成功"的记录，
    而不是被一个坏代码整批拖垮。
    """
    out: list[KlineResult | Exception] = []
    with requests.Session() as session:
        for index, code in enumerate(codes):
            try:
                out.append(fetch_kline(code, kind=kind, start=start, end=end, adjust=adjust, session=session))
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
            if pause and index < len(codes) - 1:
                time.sleep(pause)
    return out
