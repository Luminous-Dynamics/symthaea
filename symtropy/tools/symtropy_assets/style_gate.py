import json
import os

# Biome and Style Data as per Spec v0.1
BIOMES = {
    "wetland_mycelial": {
        "palette": "wetland_mycelial",
        "materials": ["living_infrastructure", "biospheric_tissue"]
    },
    "mist_forest": {
        "palette": "mist_forest",
        "materials": ["biospheric_tissue"]
    },
    "desert_spore": {
        "palette": "desert_spore",
        "materials": ["archive_matter", "hazard_boundary", "habitat_shell"]
    },
    "ocean_reef_mind": {
        "palette": "ocean_reef_mind",
        "materials": ["biospheric_tissue", "governance_ritual_surface"]
    },
    "orbital_commons": {
        "palette": "orbital_commons",
        "materials": ["habitat_shell", "robotics_care_machine", "ui_holographic"]
    },
    "subterranean_archive": {
        "palette": "subterranean_archive",
        "materials": ["archive_matter", "governance_ritual_surface"]
    },
    "civic_commons": {
        "palette": "civic_commons",
        "materials": ["living_infrastructure", "governance_ritual_surface"]
    },
    "robotics_field_ops": {
        "palette": "robotics_field_ops",
        "materials": ["robotics_care_machine"]
    },
    "red_bloom_hazard": {
        "palette": "red_bloom_hazard",
        "materials": ["hazard_boundary", "biospheric_tissue"]
    },
    "ice_shell_ocean": {
        "palette": "ice_shell_ocean",
        "materials": ["habitat_shell", "biospheric_tissue"]
    }
}

EMISSIVE_ROLES = [
    "care_repair", "consent_permission", "trust_civic", "memory_ancestry",
    "biospheric_signal", "machine_active", "warning_boundary", "quarantine_hazard",
    "grief_ritual", "celebration_festival"
]

def generate_style_report(asset_id, biome, material_family, emissive_role, corruption_overlay=False, civic_decal=False):
    # Validation logic
    warnings = []
    status = "STYLE_APPROVED"

    if not biome:
        status = "STYLE_QUARANTINE"
        warnings.append("Missing biome assignment.")
    elif biome not in BIOMES:
        status = "STYLE_QUARANTINE"
        warnings.append(f"Unknown biome: {biome}")
    else:
        if material_family and material_family not in BIOMES[biome]["materials"]:
            status = "STYLE_WARNINGS"
            warnings.append(f"Material {material_family} unusual for biome {biome}")

    if emissive_role and emissive_role not in EMISSIVE_ROLES:
        status = "STYLE_WARNINGS"
        warnings.append(f"Unrecognized emissive role: {emissive_role}")

    # Semantic Rules
    if corruption_overlay and biome == "orbital_commons":
        status = "STYLE_WARNINGS"
        warnings.append("NULL corruption overlay unusual for orbital_commons biome.")

    if civic_decal and material_family == "biospheric_tissue":
        status = "STYLE_WARNINGS"
        warnings.append("Civic decals unusual on biospheric_tissue.")

    report = {
        "asset_id": asset_id,
        "biome": biome,
        "material_family": material_family,
        "emissive_role": emissive_role,
        "corruption_overlay": corruption_overlay,
        "civic_decal": civic_decal,
        "style_status": status,
        "warnings": warnings,
        "notes": "Style audit completed."
    }

    return report

def validate_style(manifest):
    style_data = manifest.get("style", {})
    return generate_style_report(
        manifest.get("id"),
        style_data.get("biome"),
        style_data.get("material_family"),
        style_data.get("emissive_role"),
        corruption_overlay=style_data.get("corruption_overlay", False),
        civic_decal=style_data.get("civic_decal", False)
    )
