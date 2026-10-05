"""数据采集编排：多源抓取 → 一致性校验 → 写入本地 DuckDB。

源策略（2026-10 实测结论，见 ``scripts/probe_*.py``）
----------------------------------------------------
+----------------------+------------------+--------------------------------+
| 源                   | 口径             | 用途                           |
+======================+==================+================================+
| 腾讯 fqkline         | 前复权 / 原始    | **主源**（区间分页取长历史）   |
| 搜狐 hisHq           | 原始（未复权）   | **独立校验源**（一次取全历史） |
| 东财 push2his        | 前复权 / 原始    | 备用（会被限流，实测会拒连）   |
+----------------------+------------------+--------------------------------+

核心纪律
--------
1. **双口径并存**：同时落未复权价与前复权价，并写 ``adj_factor``，
   让任何结论都能回溯到"当时用的是哪个价"。
2. **交叉校验**：主源与校验源在重叠日期上的收盘价差异超过阈值就报警——
   宁可显示"数据可疑"，也不要静默产出一条看着很漂亮的收益曲线。
3. **不静默填充、不静默截断**：抓不到就记 ``ok=False``，窗口满额就报错。
4. **绕过代理**：国内行情源不需要代理；本机代理掉线时，继承代理会让
   失败原因变得极难排查（实测表现是 ProxyError 而不是"源不可用"）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from etf_lab.data import repo
from etf_lab.etl import eastmoney, sohu, tencent
from etf_lab.universe import ALL_INDEX_NAMES, ETF_PRESET, INDEX_PRESET, T_PLUS_BY_CLASS

# 主源与校验源在重叠区间的收盘价相对差异超过此阈值即报警
VERIFY_TOLERANCE = 0.001


@dataclass
class FetchReport:
    """单次抓取的结果记录。"""

    target: str
    key: str
    ok: bool
    name: str | None = None
    rows: int = 0
    start: str | None = None
    end: str | None = None
    source: str | None = None
    quality: dict[str, Any] | None = field(default=None)
    error: str | None = None


def report_to_dicts(reports: Sequence[FetchReport]) -> list[dict[str, Any]]:
    return [asdict(r) for r in reports]


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


# --------------------------------------------------------------------------- #
# 主源 + 备用源
# --------------------------------------------------------------------------- #
def _fetch_pair(code: str, kind: str, start: str | dt.date, end: str | dt.date | None, session: requests.Session):
    """抓一只标的的（原始价, 前复权价），返回 ``(frame, name, source)``。

    主源腾讯，失败后回落到东财；两者都失败则抛异常，由调用方记录。
    """
    errors: list[str] = []
    try:
        qfq = tencent.fetch_daily(code, kind=kind, start=start, end=end, adjust="qfq", session=session)
        raw = tencent.fetch_daily(code, kind=kind, start=start, end=end, adjust="raw", session=session)
        return raw.frame, qfq.frame, qfq.name, "tencent"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tencent: {type(exc).__name__}: {exc}")

    try:
        raw_em = eastmoney.fetch_kline(code, kind=kind, start=start, end=end, adjust="none", session=session)
        qfq_em = eastmoney.fetch_kline(code, kind=kind, start=start, end=end, adjust="qfq", session=session)
        return raw_em.frame, qfq_em.frame, raw_em.name, "eastmoney"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {type(exc).__name__}: {exc}")

    raise RuntimeError("所有数据源均失败 → " + " | ".join(errors))


def _merge_adjusted(raw: pd.DataFrame, qfq: pd.DataFrame, code: str) -> pd.DataFrame:
    """把原始价与前复权价按日期合并，并算 ``adj_factor``。"""
    merged = raw[["date", "open", "high", "low", "close", "volume", "amount"]].merge(
        qfq[["date", "close"]].rename(columns={"close": "close_adj"}),
        on="date",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError(f"{code} 原始价与前复权价没有交集日期")
    merged = merged.copy()
    merged["symbol"] = code
    merged["adj_factor"] = merged["close_adj"] / merged["close"]
    return merged


def verify_against_sohu(
    code: str,
    raw_frame: pd.DataFrame,
    start: str | dt.date,
    end: str | dt.date | None,
    session: requests.Session,
) -> dict[str, Any]:
    """用搜狐的未复权价交叉校验主源的原始价。

    这是本项目最重要的数据质量闸门：如果两个独立来源的原始收盘价对不上，
    那么后面所有的收益、回撤、对冲成本都建立在错误的价格上。
    """
    result: dict[str, Any] = {"source": "sohu", "checked": False}
    try:
        reference = sohu.fetch_daily(code, start=start, end=end, session=session)
    except Exception as exc:  # noqa: BLE001 - 校验失败不算抓取失败，但要如实记录
        result["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return result

    merged = raw_frame[["date", "close"]].merge(
        reference.frame[["date", "close"]], on="date", how="inner", suffixes=("_main", "_sohu")
    )
    if merged.empty:
        result["error"] = "与校验源没有重叠日期"
        return result

    relative = ((merged["close_main"] - merged["close_sohu"]).abs() / merged["close_sohu"]).dropna()
    result.update(
        checked=True,
        overlap=int(len(merged)),
        max_rel_diff=float(relative.max()),
        median_rel_diff=float(relative.median()),
        first=str(merged["date"].min().date()),
        last=str(merged["date"].max().date()),
    )
    if not relative.empty and float(relative.max()) > VERIFY_TOLERANCE:
        worst = merged.loc[relative.idxmax()]
        result["warning"] = (
            f"与校验源最大差异 {float(relative.max()):.4%}（{worst['date'].date()}："
            f"主源 {worst['close_main']} vs 校验源 {worst['close_sohu']}）"
        )
    return result


# --------------------------------------------------------------------------- #
# ETF
# --------------------------------------------------------------------------- #
def fetch_etf_prices(
    con,
    symbols: Iterable[str] | None = None,
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
    verify: bool = True,
) -> list[FetchReport]:
    """抓取 ETF 日线，写入 ``etf_price`` 与 ``etf_meta``。"""
    codes = list(symbols or ETF_PRESET)
    reports: list[FetchReport] = []
    meta_rows: list[dict[str, Any]] = []

    with _session() as session:
        for code in codes:
            try:
                raw, qfq, name, source = _fetch_pair(code, "etf", start, end, session)
                merged = _merge_adjusted(raw, qfq, code)
                rows = repo.upsert(
                    con,
                    "etf_price",
                    merged,
                    ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "adj_factor", "close_adj"],
                )
                quality = verify_against_sohu(code, raw, start, end, session) if verify else None

                meta = ETF_PRESET.get(code, {})
                asset_class = str(meta.get("asset_class", "unknown"))
                meta_rows.append(
                    {
                        "symbol": code,
                        "name": name,
                        "exchange": "SH" if code.startswith("5") else "SZ",
                        "asset_class": asset_class,
                        "underlying_index": meta.get("underlying_index"),
                        "list_date": merged["date"].min().date(),
                        "mgmt_fee": None,
                        "custodian_fee": None,
                        "currency": "CNY",
                        "is_cross_border": asset_class == "cross_border",
                        "t_plus": T_PLUS_BY_CLASS.get(asset_class),
                        "price_limit": None,
                        "notes": "T+0/T+1 为公开常见分类；费率与涨跌幅限制待核实，不写猜测值",
                    }
                )
                reports.append(
                    FetchReport(
                        target="etf_price",
                        key=code,
                        ok=True,
                        name=name,
                        rows=rows,
                        start=str(merged["date"].min().date()),
                        end=str(merged["date"].max().date()),
                        source=source,
                        quality=quality,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 采集层必须记录失败并继续
                reports.append(FetchReport(target="etf_price", key=code, ok=False, error=f"{type(exc).__name__}: {exc}"))

    if meta_rows:
        repo.upsert(
            con,
            "etf_meta",
            pd.DataFrame(meta_rows),
            [
                "symbol",
                "name",
                "exchange",
                "asset_class",
                "underlying_index",
                "list_date",
                "mgmt_fee",
                "custodian_fee",
                "currency",
                "is_cross_border",
                "t_plus",
                "price_limit",
                "notes",
            ],
        )
    return reports


# --------------------------------------------------------------------------- #
# 指数
# --------------------------------------------------------------------------- #
def fetch_index_prices(
    con,
    index_codes: Iterable[str] | None = None,
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
) -> list[FetchReport]:
    """抓取指数日线（指数不复权，用于补 ETF 上市时间过短的样本不足）。"""
    codes = list(index_codes or INDEX_PRESET)
    reports: list[FetchReport] = []

    with _session() as session:
        for code in codes:
            try:
                result = tencent.fetch_daily(code, kind="index", start=start, end=end, adjust="raw", session=session)
                frame = pd.DataFrame(
                    {
                        "index_code": code,
                        "name": ALL_INDEX_NAMES.get(code, result.name),
                        "date": result.frame["date"],
                        "close": result.frame["close"],
                    }
                )
                rows = repo.upsert(con, "index_price", frame, ["index_code", "name", "date", "close"])
                reports.append(
                    FetchReport(
                        target="index_price",
                        key=code,
                        ok=True,
                        name=ALL_INDEX_NAMES.get(code, result.name),
                        rows=rows,
                        start=str(frame["date"].min().date()),
                        end=str(frame["date"].max().date()),
                        source="tencent",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reports.append(FetchReport(target="index_price", key=code, ok=False, error=f"{type(exc).__name__}: {exc}"))
    return reports


# --------------------------------------------------------------------------- #
# 基金净值（折溢价与分红的来源）
# --------------------------------------------------------------------------- #
def fetch_fund_nav(
    con,
    symbols: Iterable[str] | None = None,
    start: str | dt.date = "2012-01-01",
    end: str | dt.date | None = None,
) -> list[FetchReport]:
    """抓取基金单位净值/累计净值，写入 ``fund_nav``。

    有了它才能算**折溢价率**（市场价/单位净值 − 1），把"买贵了"这件事从收益里分离出来。
    接口结构参考作者另一个项目 etf-tracker（东财主源 + 新浪兜底）。
    """
    from etf_lab.etl import fund_nav as fund_nav_client

    codes = list(symbols or ETF_PRESET)
    reports: list[FetchReport] = []
    with _session() as session:
        for code in codes:
            try:
                result = fund_nav_client.fetch_nav_history(code, start=start, end=end, session=session)
                frame = result.frame.copy()
                frame["symbol"] = code
                frame["dividend"] = None
                frame["daily_change"] = frame["daily_change_pct"]
                rows = repo.upsert(
                    con,
                    "fund_nav",
                    frame,
                    ["symbol", "date", "nav", "acc_nav", "dividend", "daily_change"],
                )
                reports.append(
                    FetchReport(
                        target="fund_nav",
                        key=code,
                        ok=True,
                        name=result.name,
                        rows=rows,
                        start=str(frame["date"].min().date()),
                        end=str(frame["date"].max().date()),
                        source=result.source,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reports.append(FetchReport(target="fund_nav", key=code, ok=False, error=f"{type(exc).__name__}: {exc}"))
    return reports


# --------------------------------------------------------------------------- #
# 国债收益率曲线
# --------------------------------------------------------------------------- #
def fetch_bond_yields(
    con,
    start: str | dt.date = "2015-01-01",
    end: str | dt.date | None = None,
) -> list[FetchReport]:
    """抓取国债收益率曲线，写入 ``bond_yield``。

    它的用途有两个：**无风险利率不再靠假设**（夏普等指标都要减它），
    以及用"债券 ETF 收益对收益率变动的回归"反推久期。
    """
    from etf_lab.etl import bond_yield as bond_yield_client

    try:
        result = bond_yield_client.fetch_curve(start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        return [FetchReport(target="bond_yield", key="curve", ok=False, error=f"{type(exc).__name__}: {exc}")]

    frame = result.frame.copy()
    rows = repo.upsert(con, "bond_yield", frame, ["date", "code", "tenor", "yield"])
    codes = sorted(frame["code"].unique())
    return [
        FetchReport(
            target="bond_yield",
            key="curve",
            ok=True,
            name=f"{len(codes)} 个期限：{', '.join(codes)}",
            rows=rows,
            start=str(frame["date"].min().date()),
            end=str(frame["date"].max().date()),
            source=result.source,
        )
    ]


# --------------------------------------------------------------------------- #
# 数据源探针
# --------------------------------------------------------------------------- #
PENDING_SOURCES: tuple[str, ...] = (
    "future_daily（股指期货基差/展期）：尚未接入",
    "option_daily（ETF 期权与隐含波动率）：尚未接入",
    "fx_rate（汇率）：跨境 ETF 的汇率贡献待接入",
)
def probe_sources() -> list[dict[str, Any]]:
    """探测各源在当前网络环境下是否可用（M0 证伪步骤）。"""
    with _session() as session:
        cases: list[tuple[str, Any]] = [
            ("腾讯 ETF 前复权（主源）", lambda: tencent.fetch_daily("510300", "etf", "2024-01-01", "2024-06-30", "qfq", session=session)),
            ("腾讯 ETF 原始价", lambda: tencent.fetch_daily("510300", "etf", "2024-01-01", "2024-06-30", "raw", session=session)),
            ("腾讯 指数 沪深300", lambda: tencent.fetch_daily("000300", "index", "2024-01-01", "2024-06-30", "raw", session=session)),
            ("腾讯 指数 创业板指", lambda: tencent.fetch_daily("399006", "index", "2024-01-01", "2024-06-30", "raw", session=session)),
            ("腾讯 债券 ETF", lambda: tencent.fetch_daily("511010", "etf", "2024-01-01", "2024-06-30", "qfq", session=session)),
            ("腾讯 黄金 ETF", lambda: tencent.fetch_daily("518880", "etf", "2024-01-01", "2024-06-30", "qfq", session=session)),
            ("腾讯 跨境 ETF", lambda: tencent.fetch_daily("513100", "etf", "2024-01-01", "2024-06-30", "qfq", session=session)),
            ("搜狐 未复权（校验源）", lambda: sohu.fetch_daily("510300", "etf", "2024-01-01", "2024-06-30", session=session)),
            ("东财 历史（备用源）", lambda: eastmoney.fetch_kline("510300", "etf", "2024-01-01", "2024-06-30", "qfq", session=session)),
        ]
        results: list[dict[str, Any]] = []
        for name, call in cases:
            entry: dict[str, Any] = {"name": name}
            try:
                outcome = call()
                frame = outcome.frame
                entry.update(
                    ok=True,
                    rows=len(frame),
                    first=str(frame["date"].min().date()),
                    last=str(frame["date"].max().date()),
                    source_name=getattr(outcome, "name", None),
                )
            except Exception as exc:  # noqa: BLE001
                entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:180]}")
            results.append(entry)

    for pending in PENDING_SOURCES:
        results.append({"name": pending, "ok": False, "error": "not_implemented"})
    return results
