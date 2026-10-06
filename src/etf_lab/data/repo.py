"""数据层：DuckDB 本地仓的读写。

这是**唯一**允许触碰数据库的模块。``core/`` 一律不读库，由本层取数后把
DataFrame 传进去——这样核心计算可以脱离数据库单独测试，也让缓存键只依赖
``data_version`` 而不依赖连接状态。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DEFAULT_DB_PATH = Path("data") / "lab.duckdb"
DEFAULT_USERS_DB_PATH = Path("data") / "users.duckdb"
"""用户库：账号与已保存的组合。**与行情库分开**，理由见 data/users_schema.sql。"""
USERS_SCHEMA_PATH = Path(__file__).with_name("users_schema.sql")


def connect(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """打开（必要时创建）本地数据仓。"""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        init_schema(con)
    return con


def connect_users(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """打开**用户库**（账号与已保存的组合），它是独立于行情库的第二个文件。

    为什么独立
    ----------
    行情库必须能被多个进程**只读**共享（进程池并发计算），而保存组合需要写。
    DuckDB 的写锁是文件级且进程内常驻的，两者放同一文件必然互相锁死。
    详见 ``data/users_schema.sql`` 顶部的说明。
    """
    path = Path(db_path) if db_path is not None else DEFAULT_USERS_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(USERS_SCHEMA_PATH.read_text(encoding="utf-8"))
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """建表（幂等）。"""
    con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def upsert(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame, columns: Sequence[str]) -> int:
    """按主键写入或覆盖。

    显式传 ``columns`` 而不是依赖顺序，避免上游 DataFrame 列序变化时静默错位。
    这是数据层最常见的静默错误来源之一。
    """
    if frame.empty:
        return 0
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"写入 {table} 缺少列：{missing}；实际列为 {list(frame.columns)}")
    payload = frame.loc[:, list(columns)].copy()
    con.register("_payload", payload)
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _payload")
    finally:
        con.unregister("_payload")
    return len(payload)


def log_data_version(con: duckdb.DuckDBPyConnection, source: str, notes: str = "") -> str:
    """记录一次数据更新，返回版本号（同时用于缓存键）。"""
    version = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    con.execute(
        "INSERT OR REPLACE INTO data_version VALUES (?, ?, ?, ?)",
        [version, dt.datetime.now(), source, notes],
    )
    return version


def latest_data_version(con: duckdb.DuckDBPyConnection) -> str:
    """最近一次数据版本；没有任何数据时返回 ``'empty'``（缓存键仍可用）。"""
    row = con.execute("SELECT version FROM data_version ORDER BY updated_at DESC LIMIT 1").fetchone()
    return str(row[0]) if row else "empty"


def read_price_panel(
    con: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    field: str = "close_adj",
) -> pd.DataFrame:
    """读取多个标的的价格面板（行=日期，列=代码）。

    缺失值**保留为 NaN**：不做前向填充。某只 ETF 尚未上市的那段时间本来就没有价格，
    把它填成别的价格会制造出根本不存在的收益。
    """
    if field not in ("open", "high", "low", "close", "close_adj", "adj_factor"):
        raise ValueError(f"不支持的字段：{field}")
    codes = list(symbols)
    if not codes:
        raise ValueError("symbols 不能为空")

    clauses = [f"symbol IN ({', '.join(['?'] * len(codes))})"]
    params: list[object] = list(codes)
    if start is not None:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        clauses.append("date <= ?")
        params.append(pd.Timestamp(end).date())

    sql = f"SELECT symbol, date, {field} AS value FROM etf_price WHERE {' AND '.join(clauses)}"
    frame = con.execute(sql, params).df()
    if frame.empty:
        return pd.DataFrame()
    panel = frame.pivot(index="date", columns="symbol", values="value").sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel


def read_index_panel(
    con: duckdb.DuckDBPyConnection,
    index_codes: Iterable[str],
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
) -> pd.DataFrame:
    """读取指数收盘价面板。"""
    codes = list(index_codes)
    if not codes:
        raise ValueError("index_codes 不能为空")
    clauses = [f"index_code IN ({', '.join(['?'] * len(codes))})"]
    params: list[object] = list(codes)
    if start is not None:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        clauses.append("date <= ?")
        params.append(pd.Timestamp(end).date())

    sql = f"SELECT index_code, date, close FROM index_price WHERE {' AND '.join(clauses)}"
    frame = con.execute(sql, params).df()
    if frame.empty:
        return pd.DataFrame()
    panel = frame.pivot(index="date", columns="index_code", values="close").sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel


def read_etf_meta(con: duckdb.DuckDBPyConnection, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """读取标的属性（资产类别、名称、跟踪指数等）。

    仪表盘要用资产类别来判断"配出了什么结构"（例如是否含跨境/债券），
    因此这是解锁机制的数据来源。

    ``symbols=None`` 表示取全部——自定义实验室要列出**所有**已采集的标的，
    这样新采集的标的会自动出现，而不需要改代码。
    """
    if symbols is None:
        return con.execute(
            "SELECT symbol, name, asset_class, underlying_index, t_plus, is_cross_border "
            "FROM etf_meta ORDER BY asset_class, symbol"
        ).df()
    codes = list(symbols)
    if not codes:
        return pd.DataFrame()
    sql = (
        "SELECT symbol, name, asset_class, underlying_index, t_plus, is_cross_border "
        f"FROM etf_meta WHERE symbol IN ({', '.join(['?'] * len(codes))})"
    )
    return con.execute(sql, codes).df()


def read_nav_panel(
    con: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
) -> pd.DataFrame:
    """读取**单位净值**面板（行=日期，列=代码）。

    折溢价率必须用未复权市场价与单位净值比较——前复权价已被分红调整过，
    拿它算折溢价会把历史分红误算成折价。
    """
    codes = list(symbols)
    if not codes:
        raise ValueError("symbols 不能为空")
    clauses = [f"symbol IN ({', '.join(['?'] * len(codes))})"]
    params: list[object] = list(codes)
    if start is not None:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        clauses.append("date <= ?")
        params.append(pd.Timestamp(end).date())

    sql = f"SELECT symbol, date, nav FROM fund_nav WHERE {' AND '.join(clauses)}"
    frame = con.execute(sql, params).df()
    if frame.empty:
        return pd.DataFrame()
    panel = frame.pivot(index="date", columns="symbol", values="nav").sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel


def read_bond_yield(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """读取国债收益率曲线长表（date / code / tenor / yield）。"""
    frame = con.execute("SELECT date, code, tenor, yield FROM bond_yield WHERE yield IS NOT NULL ORDER BY date").df()
    if frame.empty:
        return pd.DataFrame(columns=["date", "code", "tenor", "yield"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """各表行数——用于页面上的"数据底座"展示与断供排查。"""
    tables = [
        "etf_meta",
        "etf_price",
        "index_price",
        "fund_nav",
        "fx_rate",
        "bond_yield",
        "future_daily",
        "option_daily",
        "result_cache",
        "preset_portfolios",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def cache_get(con: duckdb.DuckDBPyConnection, cache_key: str) -> str | None:
    row = con.execute("SELECT payload FROM result_cache WHERE cache_key = ?", [cache_key]).fetchone()
    return str(row[0]) if row else None


def cache_put(con: duckdb.DuckDBPyConnection, cache_key: str, module: str, payload: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO result_cache VALUES (?, ?, ?, ?, ?)",
        [cache_key, module, payload, dt.datetime.now(), len(payload.encode("utf-8"))],
    )
