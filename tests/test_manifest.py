import json

from etf_lab.presets import PortfolioSpec
from etf_lab.reports import manifest


def test_source_digest_ignores_platform_newlines_but_tracks_resources(tmp_path):
    source = tmp_path / "lab.js"
    source.write_bytes(b"a\r\nb\r\n")
    first = manifest.source_digest(tmp_path)
    source.write_bytes(b"a\nb\n")
    assert manifest.source_digest(tmp_path) == first
    source.write_bytes(b"changed\n")
    assert manifest.source_digest(tmp_path) != first


def test_manifest_records_effective_inputs_without_local_paths(tmp_path):
    db = tmp_path / "private-path.duckdb"
    db.write_bytes(b"test snapshot")
    target = manifest.write_manifest(
        tmp_path, db_path=db, data_version="test",
        specs=[PortfolioSpec(key="one", name="test", question="why", weights={"AAA": 1.0})],
        results=[{"key": "one", "start": "2020-01-01", "end": "2021-01-01", "n_obs": 200,
                  "rf_annual": .03, "rf_source": "override", "monte_carlo": {"params": {"seed": 7}}}],
        failures=["two: RuntimeError: C:/private/path"],
    )
    text = target.read_text(encoding="utf-8")
    value = json.loads(text)
    assert value["database_sha256"] == manifest.file_digest(db)
    assert value["reports"][0]["monte_carlo"]["seed"] == 7
    assert value["failed_presets"] == ["two"]
    assert "private" not in text
