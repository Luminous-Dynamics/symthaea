import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from style_gate import validate_style

def test_style_approved():
    manifest = {
        "id": "test.style.001",
        "style": {
            "biome": "wetland_mycelial",
            "material_family": "living_infrastructure",
            "emissive_role": "biospheric_signal"
        }
    }
    report = validate_style(manifest)
    assert report["style_status"] == "STYLE_APPROVED"
    assert len(report["warnings"]) == 0

def test_style_quarantine_unknown_biome():
    manifest = {
        "id": "test.style.002",
        "style": {
            "biome": "unknown_land",
        }
    }
    report = validate_style(manifest)
    assert report["style_status"] == "STYLE_QUARANTINE"
    assert "Unknown biome: unknown_land" in report["warnings"]

def test_style_warning_unusual_material():
    manifest = {
        "id": "test.style.003",
        "style": {
            "biome": "wetland_mycelial",
            "material_family": "hazard_boundary"
        }
    }
    report = validate_style(manifest)
    assert report["style_status"] == "STYLE_WARNINGS"
    assert "Material hazard_boundary unusual for biome wetland_mycelial" in report["warnings"]

def test_style_warning_unrecognized_emissive():
    manifest = {
        "id": "test.style.004",
        "style": {
            "biome": "wetland_mycelial",
            "emissive_role": "party_lights"
        }
    }
    report = validate_style(manifest)
    assert report["style_status"] == "STYLE_WARNINGS"
    assert "Unrecognized emissive role: party_lights" in report["warnings"]
