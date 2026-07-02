import pytest
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from license_gate import validate_manifest
from sources.adapter import KenneyAdapter, PolyHavenAdapter, QuaterniusAdapter, AmbientCGAdapter

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def test_manifest_validation():
    test_cases = [
        ("cc0_valid.yaml", "APPROVED_CC0"),
        ("cc_by_valid.yaml", "APPROVED_ATTRIBUTION_REQUIRED"),
        ("nc_rejected.yaml", "REJECTED"),
        ("nd_rejected.yaml", "REJECTED"),
        ("unknown_ai_quarantine.yaml", "QUARANTINE_REVIEW"),
        ("missing_source_rejected.yaml", "REJECTED"),
        ("missing_id_rejected.yaml", "REJECTED"),
        ("cc_by_no_creator_quarantine.yaml", "QUARANTINE_REVIEW"),
    ]

    for filename, expected_status in test_cases:
        path = os.path.join(FIXTURES_DIR, filename)
        status, _ = validate_manifest(path)
        assert status == expected_status, f"Failed for {filename}: expected {expected_status}, got {status}"


def test_kenney_adapter_generates_cc0_manifest(tmp_path):
    adapter = KenneyAdapter(str(tmp_path))
    manifest = adapter.fetch_manifest("Tiny Dungeon")
    manifest_path = tmp_path / "kenney.yaml"
    manifest_path.write_text(yaml.dump(manifest))

    status, reason = validate_manifest(str(manifest_path))

    assert status == "APPROVED_CC0", reason
    assert manifest["id"] == "kenney.tiny-dungeon"
    assert manifest["source"]["source_name"] == "Kenney"
    assert manifest["source"]["source_url"] == "https://kenney.nl/assets/tiny-dungeon"
    assert manifest["license"]["id"] == "CC0-1.0"
    assert manifest["license"]["commercial_allowed"] is True


def test_kenney_adapter_rejects_empty_identifier(tmp_path):
    adapter = KenneyAdapter(str(tmp_path))

    with pytest.raises(ValueError):
        adapter.fetch_manifest(" !!! ")


def test_polyhaven_adapter_generates_cc0_manifest(tmp_path):
    # Real asset ID, validated end-to-end (download -> Blender normalize ->
    # ingest -> convert -> export) during this review session.
    adapter = PolyHavenAdapter(str(tmp_path))
    manifest = adapter.fetch_manifest("ClassicNightstand_01")
    manifest_path = tmp_path / "polyhaven.yaml"
    manifest_path.write_text(yaml.dump(manifest))

    status, reason = validate_manifest(str(manifest_path))

    assert status == "APPROVED_CC0", reason
    assert manifest["id"] == "polyhaven.ClassicNightstand_01"
    assert manifest["source"]["source_name"] == "Poly Haven"
    assert manifest["source"]["source_url"] == "https://polyhaven.com/a/ClassicNightstand_01"
    assert manifest["license"]["id"] == "CC0-1.0"
    assert manifest["license"]["commercial_allowed"] is True


def test_polyhaven_adapter_preserves_case():
    adapter = PolyHavenAdapter("/tmp")
    # Poly Haven IDs are exact-case and load-bearing for API/download URLs —
    # must NOT be lowercased the way KenneyAdapter slugifies its identifiers.
    manifest = adapter.fetch_manifest("ClassicNightstand_01")
    assert "classicnightstand" not in manifest["source"]["source_url"].lower() or \
        manifest["source"]["source_url"] == "https://polyhaven.com/a/ClassicNightstand_01"
    assert manifest["source"]["source_url"] == "https://polyhaven.com/a/ClassicNightstand_01"


def test_polyhaven_adapter_rejects_invalid_identifier(tmp_path):
    adapter = PolyHavenAdapter(str(tmp_path))

    with pytest.raises(ValueError):
        adapter.fetch_manifest("not a valid id!")


def test_polyhaven_adapter_rejects_invalid_asset_type(tmp_path):
    adapter = PolyHavenAdapter(str(tmp_path))

    with pytest.raises(ValueError):
        adapter.fetch_manifest("ClassicNightstand_01", asset_type="not_a_real_type")


def test_quaternius_adapter_generates_cc0_manifest(tmp_path):
    # Real pack, confirmed CC0 by fetching the actual pack page.
    adapter = QuaterniusAdapter(str(tmp_path))
    manifest = adapter.fetch_manifest("Downtown City MegaKit")
    manifest_path = tmp_path / "quaternius.yaml"
    manifest_path.write_text(yaml.dump(manifest))

    status, reason = validate_manifest(str(manifest_path))

    assert status == "APPROVED_CC0", reason
    assert manifest["id"] == "quaternius.downtowncitymegakit"
    assert manifest["source"]["source_name"] == "Quaternius"
    assert manifest["source"]["source_url"] == "https://quaternius.com/packs/downtowncitymegakit.html"
    assert manifest["title"] == "Downtown City MegaKit"
    assert manifest["license"]["id"] == "CC0-1.0"


def test_quaternius_adapter_normalizes_already_concatenated_id(tmp_path):
    adapter = QuaterniusAdapter(str(tmp_path))
    # Passing the bare site identifier directly should work identically.
    manifest = adapter.fetch_manifest("downtowncitymegakit")
    assert manifest["id"] == "quaternius.downtowncitymegakit"
    assert manifest["source"]["source_url"] == "https://quaternius.com/packs/downtowncitymegakit.html"


def test_quaternius_adapter_rejects_empty_identifier(tmp_path):
    adapter = QuaterniusAdapter(str(tmp_path))

    with pytest.raises(ValueError):
        adapter.fetch_manifest(" !!! ")


def test_ambientcg_adapter_generates_cc0_manifest(tmp_path):
    # Real asset ID, confirmed CC0 by fetching the actual asset page.
    adapter = AmbientCGAdapter(str(tmp_path))
    manifest = adapter.fetch_manifest("Tiles141")
    manifest_path = tmp_path / "ambientcg.yaml"
    manifest_path.write_text(yaml.dump(manifest))

    status, reason = validate_manifest(str(manifest_path))

    assert status == "APPROVED_CC0", reason
    assert manifest["id"] == "ambientcg.Tiles141"
    assert manifest["source"]["source_name"] == "ambientCG"
    assert manifest["source"]["source_url"] == "https://ambientcg.com/a/Tiles141"
    assert manifest["type"] == "material"
    assert manifest["license"]["id"] == "CC0-1.0"


def test_ambientcg_adapter_rejects_invalid_identifier(tmp_path):
    adapter = AmbientCGAdapter(str(tmp_path))

    with pytest.raises(ValueError):
        adapter.fetch_manifest("Tiles 141")  # spaces not allowed, only exact-case alphanumeric
