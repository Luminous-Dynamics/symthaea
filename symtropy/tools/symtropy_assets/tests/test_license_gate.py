import pytest
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from license_gate import validate_manifest
from sources.adapter import KenneyAdapter

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
