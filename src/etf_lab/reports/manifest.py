"""记录报告的输入与运行环境，不收集本地路径或账号信息。"""

import datetime as dt
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from etf_lab import __version__


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digest(root: Path) -> str:
    """按相对路径与文本内容散列，忽略平台换行差异。"""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".js", ".sql"}:
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    return digest.hexdigest()


def write_manifest(out: Path, *, db_path: Path, data_version: str, specs, results, failures) -> Path:
    dependencies = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "duckdb", "plotly", "pyyaml", "requests", "nicegui"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    payload = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_version": __version__,
        "source_sha256": source_digest(Path(__file__).parents[1]),
        "data_version": data_version,
        "database_sha256": file_digest(db_path),
        "requirements_lock_sha256": file_digest(Path("requirements.lock")),
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": dependencies,
        "requested_presets": [spec.to_dict() for spec in specs],
        "reports": [{
            "key": r["key"], "start": r["start"], "end": r["end"], "n_obs": r["n_obs"],
            "rf_annual": r["rf_annual"], "rf_source": r["rf_source"],
            "monte_carlo": (r.get("monte_carlo") or {}).get("params"),
            "hedge": (r.get("hedge") or {}).get("plan"),
        } for r in results],
        # 不发布异常文本，避免异常包含本地路径；详情留在构建终端。
        "failed_presets": [str(line).split(":", 1)[0] for line in failures],
        "note": "完整复现需要相同源代码、依赖、参数和行情库快照；本清单不包含行情库。源码摘要按相对路径与标准化换行后的文本计算。",
    }
    target = out / "build-manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
