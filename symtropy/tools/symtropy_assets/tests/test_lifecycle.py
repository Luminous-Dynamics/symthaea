import pytest
import os
import sqlite3
import sys
import yaml
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from registry import create_registry
from registry_manager import ingest_manifest
from review_manager import review_asset
from exporters.export_pack import generate_export_reports
import paths

@pytest.fixture
def clean_registry(tmp_path, monkeypatch):
    registry_path = str(tmp_path / "assets.sqlite")
    create_registry(registry_path)
    # Patch paths for the test
    monkeypatch.setattr(paths, "get_registry_path", lambda *args, **kwargs: registry_path)
    return registry_path

def test_export_filters_quarantined_assets(clean_registry, tmp_path):
    # Setup test assets
    assets = [
        ("test.approved", "APPROVED_CC0"),
        ("test.quarantined", "QUARANTINE_REVIEW")
    ]

    for aid, status in assets:
        manifest = {"id": aid, "title": aid, "source": {"source_url": "http://test.com"}, "license": {"id": "CC0-1.0"}, "ai": {"provenance_state": "not_ai"}}
        m_path = tmp_path / f"{aid}.yaml"
        with open(m_path, "w") as f: yaml.dump(manifest, f)
        ingest_manifest(str(m_path), status)

    export_dir = tmp_path / "export"
    generate_export_reports("test_pack", str(export_dir))

    # Verify: only approved asset in attribution
    with open(export_dir / "ATTRIBUTION.md", "r") as f:
        content = f.read()
        assert "test.approved" in content
        assert "test.quarantined" not in content

def test_real_file_export(clean_registry, tmp_path, monkeypatch):
    # Setup dummy source file
    raw_vault = tmp_path / "raw_vault"
    raw_vault.mkdir()
    asset_file = raw_vault / "test.glb"
    asset_file.write_text("dummy glb content")

    # Setup paths override
    monkeypatch.setattr(paths, "get_asset_root", lambda *args: str(tmp_path))
    monkeypatch.setattr(paths, "get_registry_path", lambda *args: str(tmp_path / "assets.sqlite"))

    # Ingest
    manifest = {
        "id": "test.export.real",
        "title": "Real Export",
        "source": {"source_url": "http://test.com"},
        "license": {"id": "CC0-1.0"},
        "ai": {"provenance_state": "not_ai"},
        "files": [{"role": "source", "path": "raw_vault/test.glb"}]
    }
    m_path = tmp_path / "manifest.yaml"
    with open(m_path, "w") as f: yaml.dump(manifest, f)

    ingest_manifest(str(m_path), "APPROVED_CC0")

    # Export
    export_dir = tmp_path / "export"
    generate_export_reports("test_pack", str(export_dir))

    # Verify
    assert (export_dir / "models" / "test.glb").exists()
    assert (export_dir / "asset_index.json").exists()
    with open(export_dir / "asset_index.json", "r") as f:
        idx = json.load(f)
        assert idx[0]["id"] == "test.export.real"

def test_review_workflow(clean_registry, monkeypatch):
    # Setup paths override
    monkeypatch.setattr(paths, "get_registry_path", lambda *args, **kwargs: clean_registry)

    # Ingest a quarantined asset
    manifest = {"id": "test.q", "title": "Q", "source": {"source_url": "http://test.com"}, "license": {"id": "CC0-1.0"}, "ai": {"provenance_state": "unknown_ai_provenance"}}
    m_path = "tmp_q.yaml"
    with open(m_path, "w") as f: yaml.dump(manifest, f)
    ingest_manifest(m_path, "QUARANTINE_REVIEW")
    os.remove(m_path)

    # Approve it
    review_asset("test.q", "APPROVED_CC0", "test_user", "Clear after review")

    # Verify status change
    conn = sqlite3.connect(clean_registry)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM assets WHERE id = 'test.q'")
    assert cursor.fetchone()[0] == "APPROVED_CC0"
    conn.close()
