import pytest
import os
import sqlite3
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from registry import create_registry
from registry_manager import ingest_manifest
from paths import get_registry_path

@pytest.fixture
def clean_registry(tmp_path):
    registry_path = str(tmp_path / "assets.sqlite")
    create_registry(registry_path)
    return registry_path

def test_registry_initializes_with_meta(clean_registry):
    conn = sqlite3.connect(clean_registry)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'")
    assert cursor.fetchone()[0] == "1"
    conn.close()

def test_ingestion_writes_provenance(clean_registry, tmp_path, monkeypatch):
    manifest = {"id": "test.prov", "title": "P", "source": {"source_url": "http://t.com"}, "license": {"id": "CC0-1.0"}, "ai": {"provenance_state": "not_ai"}}
    m_path = tmp_path / "m.yaml"
    with open(m_path, "w") as f: yaml.dump(manifest, f)

    import paths
    monkeypatch.setattr(paths, "get_registry_path", lambda *args, **kwargs: clean_registry)
    ingest_manifest(str(m_path), "APPROVED_CC0")

    conn = sqlite3.connect(clean_registry)
    cursor = conn.cursor()
    cursor.execute("SELECT event_type FROM provenance_events WHERE asset_id = 'test.prov'")
    assert cursor.fetchone()[0] == "INGESTION"
    conn.close()

def test_duplicate_id_updates(clean_registry, tmp_path, monkeypatch):
    manifest = {"id": "test.dup", "title": "First", "source": {"source_url": "http://t.com"}, "license": {"id": "CC0-1.0"}, "ai": {"provenance_state": "not_ai"}}
    m_path = tmp_path / "m.yaml"
    with open(m_path, "w") as f: yaml.dump(manifest, f)

    import paths
    monkeypatch.setattr(paths, "get_registry_path", lambda *args, **kwargs: clean_registry)
    ingest_manifest(str(m_path), "APPROVED_CC0")

    manifest["title"] = "Updated"
    with open(m_path, "w") as f: yaml.dump(manifest, f)
    ingest_manifest(str(m_path), "APPROVED_CC0")

    conn = sqlite3.connect(clean_registry)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM assets WHERE id = 'test.dup'")
    assert cursor.fetchone()[0] == "Updated"
    conn.close()

def test_ingestion_preserves_source_and_license_metadata(clean_registry, tmp_path, monkeypatch):
    manifest = {
        "id": "test.metadata",
        "title": "Metadata",
        "type": "model",
        "source": {
            "source_name": "Kenney",
            "source_url": "https://kenney.nl/assets/example",
            "creator": "Kenney",
            "acquisition_method": "public_asset_page_manifest",
        },
        "license": {
            "id": "CC0-1.0",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        "ai": {"provenance_state": "not_ai"},
    }
    m_path = tmp_path / "metadata.yaml"
    with open(m_path, "w") as f:
        yaml.dump(manifest, f)

    import paths
    monkeypatch.setattr(paths, "get_registry_path", lambda *args, **kwargs: clean_registry)
    ingest_manifest(str(m_path), "APPROVED_CC0")

    conn = sqlite3.connect(clean_registry)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT type, source_name, creator, license_url, acquisition_method
        FROM assets
        WHERE id = 'test.metadata'
    """)
    assert cursor.fetchone() == (
        "model",
        "Kenney",
        "Kenney",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "public_asset_page_manifest",
    )
    conn.close()
