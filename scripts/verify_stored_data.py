"""对**已入库**的数据做独立交叉校验与复权因子诊断。

为什么必须做这一步
------------------
抓取成功不等于数据正确。入库后的价格要再过三关：

1. **独立源比对**：本库的未复权收盘价 vs 搜狐的未复权收盘价，逐日比对；
2. **复权因子的形态**：真正的复权因子是一个**阶梯函数**（只在分红/份额折算日跳变），
   平时保持不变。但两个价格序列都只保留三位小数，所以逐日算出的比值带有
   约 0.1%~0.5% 的四舍五入噪声——因此这里检查的是"跳变是否集中在少数日子"，
   而不是"序列是否严格单调"。**推论：adj_factor 只能用于展示与回溯，
   绝不能作为收益计算的输入**（收益一律用 close_adj）。
3. **极端单日涨跌**：阈值必须按标的区分——创业板/科创板相关 ETF 是 20%，
   主板相关是 10%。用统一阈值会把 2024 年 9-10 月的真实行情误报成数据错误。

只读、只打印，不修改数据库。
"""

from __future__ import annotations

import sys
import time

import duckdb
import pandas as pd

sys.path.insert(0, "src")

import requests  # noqa: E402

from etf_lab.etl import sohu  # noqa: E402
from etf_lab.universe import ETF_PRESET  # noqa: E402

DB_PATH = "data/lab.duckdb"
# 真正的复权跳变应显著大于四舍五入噪声
STEP_THRESHOLD = 0.01
# 主板/一般 ETF 与 创业板/科创板 ETF 的涨跌幅限制
LIMIT_DEFAULT = 0.10
LIMIT_20PCT = 0.20
TWENTY_PCT_UNDERLYINGS = {"399006", "000688", "399005"}


def _price_limit(symbol: str) -> float:
    underlying = str(ETF_PRESET.get(symbol, {}).get("underlying_index") or "")
    return LIMIT_20PCT if underlying in TWENTY_PCT_UNDERLYINGS else LIMIT_DEFAULT


def _fetch_reference(symbol: str, session: requests.Session, retries: int = 4) -> pd.DataFrame:
    """搜狐接口会限流（503），必须退避重试。"""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return sohu.fetch_daily(symbol, kind="etf", start="2012-01-01", session=session).frame
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"搜狐接口连续失败：{last}")


def diagnose_factor(frame: pd.DataFrame) -> dict[str, object]:
    """把复权因子的形态拆成"跳变"与"噪声"两部分。"""
    ratio = frame["close_adj"] / frame["close"]
    change = ratio.pct_change().dropna()
    steps = change[change.abs() > STEP_THRESHOLD]
    noise = change[change.abs() <= STEP_THRESHOLD]
    return {
        "first": float(ratio.iloc[0]),
        "last": float(ratio.iloc[-1]),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "step_days": int(len(steps)),
        "largest_step": float(steps.abs().max()) if len(steps) else 0.0,
        "noise_p95": float(noise.abs().quantile(0.95)) if len(noise) else 0.0,
    }


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    con = duckdb.connect(DB_PATH, read_only=True)

    symbols = [row[0] for row in con.execute("SELECT DISTINCT symbol FROM etf_price ORDER BY symbol").fetchall()]
    print(f"库内 ETF：{symbols}\n")

    print("=== 1) 复权因子形态（跳变 vs 四舍五入噪声）===")
    factor_rows = []
    for symbol in symbols:
        frame = con.execute(
            "SELECT date, close, close_adj FROM etf_price WHERE symbol = ? ORDER BY date", [symbol]
        ).df()
        frame["date"] = pd.to_datetime(frame["date"])
        stats = diagnose_factor(frame)
        factor_rows.append({"symbol": symbol, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}})
    print(pd.DataFrame(factor_rows).to_string(index=False))
    print(
        "\n说明：noise_p95 是"被认为只是四舍五入"的那部分日变化的 95 分位；"
        "step_days 是超过 1% 的跳变天数（对应真实的分红/份额折算日）。\n"
        "注意 513100 的因子低至 0.1996（约 1/5）、510500 的因子高至 2.94（约 3 倍），"
        "这种量级只能来自**份额折算（拆分/合并）**而不是分红——这正是前复权必须处理的另一类事件。\n"
    )

    print("=== 2) 与独立源（搜狐，未复权）逐日比对 ===")
    session = requests.Session()
    session.trust_env = False  # 国内源不需要代理；代理掉线会让失败原因被掩盖
    rows = []
    for index, symbol in enumerate(symbols):
        try:
            reference = _fetch_reference(symbol, session)
        except Exception as exc:  # noqa: BLE001
            rows.append({"symbol": symbol, "overlap": 0, "max_rel_diff": None, "median_rel_diff": None, "note": str(exc)[:70]})
            continue
        mine = con.execute("SELECT date, close FROM etf_price WHERE symbol = ?", [symbol]).df()
        mine["date"] = pd.to_datetime(mine["date"])
        merged = mine.merge(reference[["date", "close"]], on="date", how="inner", suffixes=("_db", "_ref"))
        if merged.empty:
            rows.append({"symbol": symbol, "overlap": 0, "max_rel_diff": None, "median_rel_diff": None, "note": "无重叠日期"})
            continue
        rel = ((merged["close_db"] - merged["close_ref"]).abs() / merged["close_ref"]).dropna()
        worst = merged.loc[rel.idxmax()]
        rows.append(
            {
                "symbol": symbol,
                "overlap": len(merged),
                "max_rel_diff": round(float(rel.max()), 6),
                "median_rel_diff": round(float(rel.median()), 6),
                "worst_date": str(worst["date"].date()),
                "db_close": float(worst["close_db"]),
                "ref_close": float(worst["close_ref"]),
                "note": "通过" if float(rel.max()) <= 0.01 else "需人工确认",
            }
        )
        if index < len(symbols) - 1:
            time.sleep(1.2)  # 搜狐会限流，必须留间隔
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=== 3) 极端单日涨跌（按标的实际涨跌幅限制判定）===")
    flagged = 0
    for symbol in symbols:
        limit = _price_limit(symbol)
        frame = con.execute("SELECT date, close_adj FROM etf_price WHERE symbol = ? ORDER BY date", [symbol]).df()
        frame["date"] = pd.to_datetime(frame["date"])
        move = frame["close_adj"].pct_change()
        extreme = frame.loc[move.abs() > limit, ["date", "close_adj"]].assign(move=move[move.abs() > limit])
        print(f"{symbol}: 限制 ±{limit:.0%}，超限 {len(extreme)} 天")
        if not extreme.empty:
            flagged += len(extreme)
            print(extreme.head(4).to_string(index=False))
    if flagged == 0:
        print("未发现超过各自涨跌幅限制的单日涨跌")
    return 0


if __name__ == "__main__":
    sys.exit(main())
