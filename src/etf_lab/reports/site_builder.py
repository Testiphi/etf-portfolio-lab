"""静态站文件编排与导出；计算和 HTML 渲染仍由 static_site 提供。"""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import closing
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

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


def _build_into(out_dir: str | Path, db_path: str | Path | None, rf_annual: float | None) -> Path:
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
            except Exception as exc:  # noqa: BLE001 - 收集失败后统一终止发布
                failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")

        if failures:
            raise RuntimeError("构建未发布，以下组合失败：\n" + "\n".join(failures))
        if not results:
            raise RuntimeError("构建未发布：没有可生成的组合")

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

        return out


class _PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        for name in ("href", "src", "data-src"):
            if values.get(name):
                self.links.append(values[name])


def validate_bundle(staged: Path) -> None:
    """发布前验证本地链接与图表文件，失败不触碰原站点。"""
    staged = staged.resolve()
    metadata = json.loads((staged / "build-manifest.json").read_text(encoding="utf-8"))
    if not metadata.get("reports") or metadata.get("failed_presets"):
        raise RuntimeError("构建清单没有完整成功的报告")
    for name in ("index.html", "about.html", "concepts.html", "assets/lab.js", "assets/style.css", "assets/plotly.min.js"):
        if not (staged / name).is_file():
            raise RuntimeError(f"构建缺少文件：{name}")
    for result in metadata["reports"]:
        key = result["key"]
        for name in (f"{key}.html", f"data/{key}.figs.js"):
            target = (staged / name).resolve()
            if not target.is_relative_to(staged) or not target.is_file():
                raise RuntimeError(f"组合缺少页面或图表数据：{key}")
    for page in staged.glob("*.html"):
        parser = _PageLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        if len(parser.ids) != len(set(parser.ids)):
            raise RuntimeError(f"页面存在重复 ID：{page.name}")
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            if not url.path:
                if url.fragment and unquote(url.fragment) not in parser.ids:
                    raise RuntimeError(f"页内导航目标不存在：{page.name} → {link}")
                continue
            target = (page.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(staged) or not target.is_file():
                raise RuntimeError(f"本地资源不存在或越界：{page.name} → {link}")


class PublicationRecoveryError(RuntimeError):
    """恢复也失败时保留临时备份，交由操作者处理。"""


def _publish(staged: Path, out: Path, backup: Path) -> None:
    artifacts = sorted(path.relative_to(staged) for path in staged.rglob("*") if path.is_file())
    # 先检查全部目标，防止目录链接将写入引向其它位置。
    for relative in artifacts:
        target = out / relative
        if target.resolve() != target or (target.exists() and not target.is_file()):
            raise RuntimeError(f"拒绝覆盖目录或链接：{target}")
    # 写入前备份所有要替换的文件；无关文件保持原样。
    for relative in artifacts:
        target = out / relative
        if target.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    changed = []
    try:
        for relative in artifacts:
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged / relative).replace(target)
            changed.append(relative)
    except BaseException as error:
        recovery_errors = []
        for relative in reversed(changed):
            target, saved = out / relative, backup / relative
            try:
                if saved.exists():
                    saved.replace(target)
                else:
                    target.unlink()
            except OSError as recovery_error:
                recovery_errors.append(str(recovery_error))
        if recovery_errors:
            raise PublicationRecoveryError(f"发布及恢复失败，备份保留在 {backup}") from error
        if isinstance(error, OSError):
            raise RuntimeError("发布失败，已恢复原有文件") from error
        raise


def build(out_dir: str | Path = "docs", db_path: str | Path | None = None, rf_annual: float | None = None) -> Path:
    out = Path(out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".etf-build-", dir=out.parent)).resolve()
    # 临时目录只允许在明确的输出父目录内清理。
    if work.parent != out.parent or not work.name.startswith(".etf-build-"):
        raise RuntimeError("临时构建目录不在预期位置")
    keep_backup = False
    try:
        staged = work / "site"
        _build_into(staged, db_path, rf_annual)
        validate_bundle(staged)
        _publish(staged, out, work / "backup")
    except PublicationRecoveryError:
        keep_backup = True
        raise
    finally:
        if not keep_backup:
            shutil.rmtree(work)
    return Path(out_dir)
