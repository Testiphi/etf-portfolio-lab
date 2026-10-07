"""保证计算、资源或发布失败时不会把半套报告当作新站点。"""

import json
from pathlib import Path

import pytest

from etf_lab.data import repo
from etf_lab.reports import site_builder as builder


def _bundle(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "about.html", "concepts.html", "one.html"):
        (root / name).write_text('<div id="top"></div><a href="#top">Top</a><script src="assets/lab.js"></script>')
    (root / "assets").mkdir()
    for name in ("lab.js", "style.css", "plotly.min.js"):
        (root / "assets" / name).write_text("test")
    (root / "data").mkdir()
    (root / "data/one.figs.js").write_text("test")
    (root / "build-manifest.json").write_text(json.dumps({"reports": [{"key": "one"}], "failed_presets": []}))


def test_failed_computation_keeps_existing_site(tmp_path, monkeypatch):
    db = tmp_path / "lab.duckdb"
    con = repo.connect(db)
    con.execute("INSERT INTO etf_price (symbol, date, close_adj) VALUES ('AAA', '2020-01-01', 1)")
    con.close()
    out = tmp_path / "docs"
    out.mkdir()
    (out / "index.html").write_text("previous report")
    def broken(*args, **kwargs):
        raise ValueError("bad sample")
    monkeypatch.setattr(builder.site, "compute_preset", broken)
    with pytest.raises(RuntimeError, match="构建未发布"):
        builder.build(out, db)
    assert (out / "index.html").read_text() == "previous report"
    assert len(list(out.iterdir())) == 1
    assert not list(tmp_path.glob(".etf-build-*"))
    # 构建异常后连接也应关闭，不阻止下一次采集写入。
    con = repo.connect(db)
    con.close()


@pytest.mark.parametrize("html", [
    '<div id="same"></div><div id="same"></div>',
    '<a href="#missing">Missing anchor</a>',
    '<script src="assets/missing.js"></script>',
    '<a href="../outside.html">Outside</a>',
])
def test_bundle_rejects_broken_links_and_duplicate_ids(tmp_path, html):
    _bundle(tmp_path / "site")
    (tmp_path / "site/index.html").write_text(html)
    with pytest.raises(RuntimeError):
        builder.validate_bundle(tmp_path / "site")


def test_successful_publish_preserves_unrelated_files(tmp_path, monkeypatch):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "CNAME").write_text("example.org")
    monkeypatch.setattr(builder, "_build_into", lambda staged, *_: _bundle(staged))
    builder.build(out)
    assert (out / "CNAME").read_text() == "example.org"
    assert (out / "one.html").is_file()
    assert not list(tmp_path.glob(".etf-build-*"))


def test_publication_io_failure_restores_replaced_and_removes_new_files(tmp_path, monkeypatch):
    staged, out, backup = (tmp_path / name for name in ("stage", "docs", "backup"))
    staged.mkdir(); out.mkdir()
    for name in ("0new.html", "a.html", "b.html"):
        (staged / name).write_text("new")
    for name in ("a.html", "b.html"):
        (out / name).write_text("old")
    original = Path.replace
    def fail_second(self, target):
        if self == staged / "b.html":
            raise OSError("disk error")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(RuntimeError, match="已恢复"):
        builder._publish(staged, out, backup)
    assert (out / "a.html").read_text() == (out / "b.html").read_text() == "old"
    assert not (out / "0new.html").exists()


def test_failed_recovery_preserves_backup(tmp_path, monkeypatch):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "about.html").write_text("old about")
    monkeypatch.setattr(builder, "_build_into", lambda staged, *_: _bundle(staged))
    original = Path.replace
    def fail_publish_and_recovery(self, target):
        if self.name == "one.html" or "backup" in self.parts:
            raise OSError("disk error")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", fail_publish_and_recovery)
    with pytest.raises(builder.PublicationRecoveryError, match="备份保留"):
        builder.build(out)
    work = next(tmp_path.glob(".etf-build-*"))
    assert (work / "backup/about.html").read_text() == "old about"
