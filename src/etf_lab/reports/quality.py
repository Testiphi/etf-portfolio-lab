"""数据覆盖与来源展示；未知状态始终显式保留。"""

from typing import Any, Mapping, Sequence

import pandas as pd

from etf_lab.reports import theme


def coverage_summary(panel: pd.DataFrame, aligned: pd.DataFrame) -> dict[str, Any]:
    """日期分母为请求区间内已观测日期的并集，不推断交易日历。"""
    if panel.empty:
        return {"observed_dates": 0, "used_dates": len(aligned), "excluded_dates": None, "assets": []}
    assets = []
    for symbol in panel.columns:
        valid = panel[symbol].dropna()
        assets.append({
            "symbol": symbol, "start": str(valid.index.min().date()) if len(valid) else None,
            "end": str(valid.index.max().date()) if len(valid) else None,
            "missing": int(panel[symbol].isna().sum()),
            "missing_ratio": float(panel[symbol].isna().mean()),
        })
    return {"observed_dates": len(panel), "used_dates": len(aligned),
            "excluded_dates": len(panel.index.difference(aligned.index)), "assets": assets}


def coverage_html(result: Mapping[str, Any]) -> str:
    coverage = result.get("data_coverage") or {}
    if not coverage.get("observed_dates"):
        return '<p class="note">没有可汇总的实物标的价格面板；合成资产覆盖以实际计算区间为准。</p>'
    labels = result.get("labels") or {}
    rows = "".join(
        f'<tr><td>{theme.esc(labels.get(r["symbol"], r["symbol"]))}</td>'
        f'<td>{theme.esc(r["start"])}</td><td>{theme.esc(r["end"])}</td>'
        f'<td>{r["missing"]}（{theme.pct(r["missing_ratio"])}）</td></tr>'
        for r in coverage["assets"]
    )
    return (
        f'<p class="note">已观测日期并集 {coverage["observed_dates"]} 天 → 实际样本 '
        f'{coverage["used_dates"]} 天；排除 {coverage["excluded_dates"]} 天。'
        '排除可能来自标的缺失、上市时间差异或合成资产区间限制。</p>'
        '<table><thead><tr><th>标的</th><th>首个有效日期</th><th>最新有效日期</th>'
        f'<th>并集内缺失</th></tr></thead><tbody>{rows}</tbody></table>'
        '<p class="note">缺失比例以所选标的在请求区间内的观测日期并集为分母，含上市前空缺；'
        '不是交易日缺报率，也不能发现所有标的同时缺报的日期。不填充缺失价格。</p>'
    )


def dataset_html(rows: Sequence[Mapping[str, Any]] | None) -> str:
    if rows is None:
        return '<p class="note">未提供数据质量快照。</p>'
    body = "".join(
        f'<tr><td>{theme.esc(r["label"])}</td><td>{r["rows"]:,}</td>'
        f'<td>{theme.esc(r["start"] or "—")}</td><td>{theme.esc(r["end"] or "—")}</td>'
        f'<td>{theme.pct(r["missing_ratio"])}</td>'
        f'<td>{"有入库记录" if r["rows"] else "无入库记录"}</td>'
        f'<td>{theme.esc(r["source"])}</td></tr>' for r in rows
    )
    return (
        '<table><thead><tr><th>数据集</th><th>行数</th><th>起始日期</th><th>最新日期</th>'
        '<th>关键值缺失</th><th>库存状态</th><th>采集配置</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        '<p class="note">关键值缺失 = 现有记录中价格、净值或收益率为空或非有限值的比例；'
        '不含未入库日期，不能据此断言数据完整。范围为整表覆盖，不代表每只标的都有同样长的历史。</p>'
        '<p class="note">采集配置不是逐条历史来源证明；最新日期也不等于已更新到最近交易日。</p>'
        + audit_html(next((r.get("checks", []) for r in rows if r["table"] == "etf_price"), []))
    )


def audit_html(checks: Sequence[Mapping[str, Any]]) -> str:
    if not checks:
        return '<p class="note">未保存 ETF 逐标的校验记录。</p>'
    body = "".join(
        f'<tr><td>{theme.esc(r["symbol"])}</td><td>{theme.esc(r["status"])}</td>'
        f'<td>{theme.esc(r["source"] or "—")}</td><td>{theme.esc(r["recorded_at"] or "—")}</td>'
        f'<td>{theme.esc(r["first"] or "—")} ~ {theme.esc(r["last"] or "—")}</td>'
        f'<td>{theme.esc(r["overlap"] if r["overlap"] is not None else "—")}</td>'
        f'<td>{theme.pct(r["max_rel_diff"], digits=4)}</td></tr>' for r in checks
    )
    return (
        '<h3>ETF 最近采集与交叉校验</h3><table><thead><tr><th>标的</th><th>状态</th>'
        '<th>当次行情源</th><th>记录时间（UTC）</th><th>校验覆盖区间</th><th>有效重叠日</th>'
        f'<th>最大相对差异</th></tr></thead><tbody>{body}</tbody></table>'
        '<p class="note">仅比较当次采集与搜狐重叠日期的未复权收盘价；一致不代表整段历史、前复权价或收益计算已验证。'
        '库存价格指纹变化后记录标为过期。未保存、未执行、未完成与失败均不能视为通过；失败不清除旧行情。</p>'
    )
