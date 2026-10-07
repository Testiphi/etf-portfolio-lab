"""静态站文件编排与导出；计算和 HTML 渲染仍由 static_site 提供。"""

from __future__ import annotations

import json
import shutil
from contextlib import closing
from pathlib import Path

import pandas as pd

from etf_lab.data import repo
from etf_lab.presets import PRESETS
from etf_lab.reports import manifest, static_site as site, theme

def _copy_plotly_js(out_dir: Path) -> Path:
    """把 plotly.min.js 复制一份到 assets/，各页面共享（每页内联会让站点膨胀到几十 MB）。"""
    import plotly

    source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not source.exists():  # pragma: no cover - 依赖包结构变化时给出明确指引
        raise RuntimeError(f"未找到 plotly.min.js：{source}；请确认 plotly 版本")
    target_dir = out_dir / "assets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "plotly.min.js"
    shutil.copyfile(source, target)
    return target


def export_preset_prices(con, out_dir: Path) -> Path | None:
    """导出示例组合用到的价格序列（供路线 B 的 Pyodide 页面直接吃）。"""
    symbols: list[str] = []
    for spec in PRESETS:
        symbols.extend(s for s in spec.weights if s not in symbols)
    panel = repo.read_price_panel(con, symbols, field="close_adj")
    if panel.empty:
        return None
    payload = {
        "data_version": repo.latest_data_version(con),
        "field": "close_adj",
        "note": "前复权收盘价；日期为交易日，缺失表示该标的当日无数据（不做填充）",
        "dates": [str(idx.date()) for idx in panel.index],
        "series": {c: [None if pd.isna(v) else round(float(v), 4) for v in panel[c]] for c in panel.columns},
    }
    target_dir = out_dir / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "preset_prices.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def build(out_dir: str | Path = "docs", db_path: str | Path | None = None, rf_annual: float | None = None) -> Path:
    """生成整站，返回输出目录。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with closing(repo.connect(db_path, read_only=True)) as con:
        counts = repo.table_counts(con)
        data_quality = repo.data_quality_summary(con)
        data_version = repo.latest_data_version(con)
        if counts.get("etf_price", 0) == 0:
            raise RuntimeError("本地数据仓还没有行情数据，请先运行：python -m etf_lab.cli fetch --preset core")

        _copy_plotly_js(out)
        (out / "assets" / "style.css").write_text(theme.STYLE, encoding="utf-8")

        results: list[dict[str, Any]] = []
        dashboards: dict[str, str] = {}
        figs_by_key: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for spec in PRESETS:
            try:
                result = site.compute_preset(con, spec, rf_annual=rf_annual)
                figs: dict[str, Any] = {}
                # 仪表盘只渲染一次，首屏与独立页共用；图表数据同时被收集起来写文件
                dashboards[spec.key] = site.render_dashboard(result, prefix=f"{spec.key}-", figs=figs)
                figs_by_key[spec.key] = figs
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - 单个组合作不出来不应让整站失败
                failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")

        # 图表 JSON 外置：数据文件按档位懒加载且可被缓存
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for key, figs in figs_by_key.items():
            (data_dir / f"{key}.figs.js").write_text(theme.figure_data_js(figs), encoding="utf-8")
        (out / "assets" / "lab.js").write_text(theme.LAB_JS, encoding="utf-8")

        (out / "index.html").write_text(
            site.render_index(results, dashboards=dashboards, counts=counts, data_version=data_version, data_quality=data_quality), encoding="utf-8"
        )
        for result in results:
            key = str(result["key"])
            (out / f"{key}.html").write_text(
                site.render_preset_page(result, dashboard=dashboards[key]), encoding="utf-8"
            )
        (out / "concepts.html").write_text(site.render_concepts(), encoding="utf-8")
        (out / "about.html").write_text(site.render_about(counts=counts, data_version=data_version, data_quality=data_quality, results=results), encoding="utf-8")
        (out / ".nojekyll").write_text("", encoding="utf-8")
        export_preset_prices(con, out)

        manifest.write_manifest(
            out, db_path=Path(db_path) if db_path is not None else repo.DEFAULT_DB_PATH,
            data_version=data_version, specs=PRESETS, results=results, failures=failures,
        )

        if failures:
            print("以下组合未能生成（已跳过，未做填充）：")
            for line in failures:
                print(f"  - {line}")
        return out
