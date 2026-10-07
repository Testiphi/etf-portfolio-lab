"""持久化采集证据；历史校验只能证明当次重叠区间，不能外推。"""

import datetime as dt
import hashlib
import json
import math
from uuid import uuid4


def etf_fingerprint(con, symbol: str) -> str:
    rows = con.execute(
        "SELECT date, close, close_adj FROM etf_price WHERE symbol = ? ORDER BY date", [symbol]
    ).fetchall()
    payload = json.dumps(rows, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_reports(con, reports, data_version: str | None) -> None:
    for report in reports:
        fingerprint = None
        if report["target"] == "etf_price" and report["ok"]:
            fingerprint = etf_fingerprint(con, report["key"])
        con.execute(
            "INSERT INTO fetch_audit VALUES (?, ?, ?, ?, ?, ?, ?)",
            [uuid4().hex, dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), data_version,
             report["target"], report["key"], json.dumps(report, ensure_ascii=False), fingerprint],
        )


def etf_checks(con) -> list[dict]:
    """兼容尚无审计表的旧只读库；失败、跳过和过期均不标为通过。"""
    symbols = {row[0] for row in con.execute("SELECT DISTINCT symbol FROM etf_price").fetchall()}
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'fetch_audit'"
    ).fetchone()
    latest = {}
    if exists:
        records = con.execute(
            "SELECT symbol, recorded_at, report_json, snapshot_sha256 FROM fetch_audit "
            "WHERE target = 'etf_price' "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY recorded_at DESC, audit_id DESC) = 1"
        ).fetchall()
        latest = {symbol: (when, json.loads(payload), fingerprint) for symbol, when, payload, fingerprint in records}
    rows = []
    for symbol in sorted(symbols | latest.keys()):
        row = {"symbol": symbol, "status": "未保存校验记录", "source": None,
               "recorded_at": None, "first": None, "last": None, "overlap": None, "max_rel_diff": None}
        if symbol in latest:
            when, report, fingerprint = latest[symbol]
            q = report.get("quality") or {}
            row.update(source=report.get("source"), recorded_at=str(when),
                       first=q.get("first"), last=q.get("last"), overlap=q.get("overlap"),
                       max_rel_diff=q.get("max_rel_diff"))
            if not report.get("ok"):
                row["status"] = "最近采集失败"
            elif fingerprint != etf_fingerprint(con, symbol):
                row["status"] = "库存已变化，记录过期"
            elif not q:
                row["status"] = "本次未执行校验"
            elif not q.get("checked") or not q.get("overlap"):
                row["status"] = "校验未完成"
            elif not all(isinstance(q.get(k), (int, float)) and math.isfinite(q[k])
                         for k in ("max_rel_diff", "tolerance")):
                row["status"] = "校验结果不完整"
            elif q.get("warning") or q["max_rel_diff"] > q["tolerance"]:
                row["status"] = "重叠区间存在差异"
            else:
                row["status"] = "重叠区间一致（阈值内）"
        rows.append(row)
    return rows
