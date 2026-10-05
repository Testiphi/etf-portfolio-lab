"""基金净值采集（东财 lsjz 主源 + 新浪 API 兜底）。

为什么要接基金净值
------------------
此前只有 ETF 的**二级市场行情**，缺基金公司口径的**单位净值 / 累计净值**。
补上之后能多做两件事：

1. **折溢价率** = 二级市场收盘价 / 单位净值 − 1。ETF 会溢价或折价交易，
   买入溢价高的 ETF 相当于多付钱；溢价回落会造成与标的指数无关的亏损。
2. **净值增长 vs 价格收益的对照**，把跟踪效果与折溢价分离开看。

接口来源
--------
主源与新源兜底的结构参考了作者另一个项目 `Testiphi/etf-tracker`
（东财 `/f10/lsjz` 主源 + 新浪基金 API 兜底 + 多源交叉校验），
在此基础上改为"整段历史分页拉取 + 落库"，并统一走本项目的限速与异常约定。

**口径提醒**：中国的"累计净值"通常是 ``单位净值 + 历史累计分红``（简单累加，
不考虑分红再投资），因此它**不等于**分红再投资的总收益指数。
它的用途是看分红发生的时间与金额，不能直接当作总收益序列与前复权价比较。
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

LSJZ_URL = "https://api.fund.eastmoney.com/f10/lsjz"
SINA_NAV_URL = "https://stock.finance.sina.com.cn/fundinfo/api/openapi.php/CaihuiFundInfoService.getNav"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": "https://fund.eastmoney.com/",
}
MIN_REQUEST_INTERVAL = 0.6
_last_request_at = 0.0


class FundNavError(RuntimeError):
    """净值接口不可用或返回结构与预期不符。"""


@dataclass(frozen=True)
class NavResult:
    code: str
    name: str
    frame: pd.DataFrame
    """列：date, nav（单位净值）, acc_nav（累计净值）, daily_change_pct。"""
    source: str


def _throttle(min_interval: float = MIN_REQUEST_INTERVAL) -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_at = time.monotonic()


def parse_lsjz_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """解析东财 lsjz 返回。

    字段：``FSRQ``(日期) / ``DWJZ``(单位净值) / ``LJJZ``(累计净值) / ``JZZZL``(日增长率%)。
    ``ErrCode != 0`` 或列表为空时抛错，绝不返回空表让调用方误以为"这只基金没有净值"。
    """
    if payload.get("ErrCode") not in (0, None):
        raise FundNavError(f"接口返回错误：ErrCode={payload.get('ErrCode')} ErrMsg={payload.get('ErrMsg')}")
    records = (payload.get("Data") or {}).get("LSJZList") or []
    if not records:
        raise FundNavError("LSJZList 为空（可能代码不存在或未上市）")

    rows = []
    for item in records:
        nav = item.get("DWJZ")
        if nav in (None, "", "--"):
            continue  # 少数日期只有分红信息没有净值，跳过而不是填 0
        rows.append(
            {
                "date": pd.to_datetime(item.get("FSRQ"), errors="coerce"),
                "nav": pd.to_numeric(nav, errors="coerce"),
                "acc_nav": pd.to_numeric(item.get("LJJZ"), errors="coerce"),
                "daily_change_pct": pd.to_numeric(item.get("JZZZL"), errors="coerce"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FundNavError("解析后没有任何有效净值记录")
    if frame["date"].isna().any():
        raise FundNavError("存在无法解析的日期")
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_nav_history(
    code: str,
    start: str | dt.date = "2022-01-01",
    end: str | dt.date | None = None,
    session: requests.Session | None = None,
    page_size: int = 20,
    max_pages: int = 400,
    pause: float = 0.3,
) -> NavResult:
    """分页拉取净值历史（东财为主源，失败回落新浪）。

    **服务端硬限制每页 20 条**（实测 ``pageSize`` 传 50/100/200 也只返回 20，传 500 返回 0），
    因此这里靠 ``pageIndex`` 翻页，停止条件是"取到 start 之前"或"某一页不满"，
    并用"最早日期是否持续前移"来识别 pageIndex 失效，避免死循环。

    净值只用于算折溢价与观察分红，**不参与收益计算**，所以默认窗口只需几年。
    """
    own_session = session is None
    client = session or requests.Session()
    if own_session:
        client.trust_env = False

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(dt.date.today())
    frames: list[pd.DataFrame] = []
    source = "eastmoney-lsjz"
    name = code
    try:
        first_len: int | None = None
        previous_min: pd.Timestamp | None = None
        for page in range(1, max_pages + 1):
            _throttle()
            response = client.get(
                LSJZ_URL,
                params={"fundCode": code, "pageIndex": page, "pageSize": page_size},
                headers=HEADERS,
                timeout=25,
            )
            text = response.text
            if text.lstrip().startswith("<"):
                raise FundNavError(f"返回 HTML 而非 JSON（可能被拦截）：{text[:100]}")
            response.raise_for_status()
            payload = response.json()
            frame = parse_lsjz_payload(payload)
            name = str((payload.get("Data") or {}).get("SHORTNAME") or name)
            frames.append(frame)
            current_min = frame["date"].min()

            if first_len is None:
                first_len = len(frame)
            if current_min <= start_ts:
                break
            if len(frame) < first_len:  # 最后一页
                break
            if previous_min is not None and current_min >= previous_min:
                raise FundNavError(
                    f"{code} 第 {page} 页没有前移（pageIndex 可能未生效），停止以免死循环"
                )
            previous_min = current_min
            if page == max_pages:
                raise FundNavError(f"{code} 翻页超过 {max_pages} 页仍未覆盖到 {start_ts.date()}")
            if pause:
                time.sleep(pause)
    except Exception as exc:  # noqa: BLE001 - 主源失败后尝试兜底源
        frames = []
        source = "sina-fund-api"
        try:
            result = _fetch_sina(code, start_ts, end_ts, client)
            return result
        except Exception as fallback_exc:  # noqa: BLE001
            raise FundNavError(f"主源与兜底源均失败：{exc}；兜底：{fallback_exc}") from exc
    finally:
        if own_session:
            client.close()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)]
    merged = merged.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if merged.empty:
        raise FundNavError(f"{code} 在 {start_ts.date()}~{end_ts.date()} 没有净值记录")
    return NavResult(code=code, name=name, frame=merged, source=source)


def _fetch_sina(code: str, start: pd.Timestamp, end: pd.Timestamp, client: requests.Session) -> NavResult:
    """新浪基金净值 API 兜底（一次最多若干页，只取区间内数据）。"""
    frames: list[pd.DataFrame] = []
    name = code
    for page in range(1, 40):
        _throttle()
        response = client.get(
            SINA_NAV_URL,
            params={"callback": "jQuery", "fund": code, "page": page, "num": 50, "sort": "nav_date", "asc": "desc"},
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=25,
        )
        response.raise_for_status()
        match = re.search(r"jQuery\((.+)\)", response.text, re.DOTALL)
        if not match:
            raise FundNavError("新浪返回格式不可识别")
        payload = json.loads(match.group(1))
        data = ((payload.get("result") or {}).get("data")) or []
        if not data:
            break
        rows = []
        for item in data:
            rows.append(
                {
                    "date": pd.to_datetime(item.get("nav_date"), errors="coerce"),
                    "nav": pd.to_numeric(item.get("nav"), errors="coerce"),
                    "acc_nav": pd.to_numeric(item.get("accumulated_nav"), errors="coerce"),
                    "daily_change_pct": pd.to_numeric(item.get("daily_profit"), errors="coerce"),
                }
            )
        frame = pd.DataFrame(rows).dropna(subset=["date", "nav"])
        frames.append(frame)
        if frame["date"].min() <= start:
            break
    if not frames:
        raise FundNavError("新浪也没有返回净值数据")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[(merged["date"] >= start) & (merged["date"] <= end)].sort_values("date").drop_duplicates("date")
    return NavResult(code=code, name=name, frame=merged.reset_index(drop=True), source="sina-fund-api")


def premium_discount(price: pd.Series, nav: pd.Series) -> pd.Series:
    """折溢价率 = 市场价 / 单位净值 − 1。

    正值表示溢价（你多付了钱），负值表示折价。跨境 ETF 在额度紧张时溢价会明显放大，
    且溢价回落造成的亏损与标的指数涨跌无关——这是跨境品种特有的风险来源。
    """
    joined = pd.concat([price.rename("price"), nav.rename("nav")], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float, name="premium_discount")
    out = joined["price"] / joined["nav"] - 1.0
    out.name = "premium_discount"
    return out
