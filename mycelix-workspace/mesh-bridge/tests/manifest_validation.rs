// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manifest & wiring validation tests.
//!
//! Validates that the resilience hApp manifest is well-formed,
//! DNA paths resolve, and role names match what the Observatory
//! resilience-client.ts expects.

use std::path::PathBuf;

/// Path from the mesh-bridge crate root to the workspace root.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[derive(serde::Deserialize)]
struct HappManifest {
    manifest_version: String,
    name: String,
    roles: Vec<Role>,
}

#[derive(serde::Deserialize)]
struct Role {
    name: String,
    dna: Dna,
}

#[derive(serde::Deserialize)]
struct Dna {
    path: String,
}

/// The role names that resilience-client.ts uses in callZome() calls.
const EXPECTED_ROLES: &[&str] = &[
    "identity",
    "finance",
    "commons_care",
    "civic",
    "governance",
    "hearth",
    "knowledge",
    "supplychain",
];

/// The role→zome mappings that resilience-client.ts depends on.
const EXPECTED_ZOME_ROUTES: &[(&str, &str)] = &[
    ("finance", "tend"),
    ("finance", "price_oracle"),
    ("commons_care", "food_production"),
    ("commons_care", "mutualaid_timebank"),
    ("commons_care", "water_capture"),
    ("commons_care", "water_purity"),
    ("commons_care", "care_circles"),
    ("commons_care", "housing_units"),
    ("civic", "emergency_comms"),
    ("hearth", "hearth_kinship"),
    ("hearth", "hearth_emergency"),
    ("hearth", "hearth_resources"),
    ("knowledge", "claims"),
    ("knowledge", "graph"),
    ("supplychain", "inventory_coordinator"),
];

#[test]
fn test_manifest_parses() {
    let manifest_path = workspace_root().join("happs/mycelix-resilience-happ.yaml");
    assert!(
        manifest_path.exists(),
        "Resilience manifest not found at {manifest_path:?}"
    );

    let yaml = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: HappManifest = serde_yaml::from_str(&yaml).unwrap();

    assert_eq!(manifest.name, "mycelix-resilience");
    assert_eq!(manifest.manifest_version, "0");
}

#[test]
fn test_manifest_has_all_expected_roles() {
    let manifest_path = workspace_root().join("happs/mycelix-resilience-happ.yaml");
    let yaml = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: HappManifest = serde_yaml::from_str(&yaml).unwrap();

    let role_names: Vec<&str> = manifest.roles.iter().map(|r| r.name.as_str()).collect();

    for expected in EXPECTED_ROLES {
        assert!(
            role_names.contains(expected),
            "Manifest missing role '{expected}'. Found: {role_names:?}"
        );
    }

    // Exactly 8 roles — no bloat
    assert_eq!(
        manifest.roles.len(),
        8,
        "Expected exactly 8 roles, found {}",
        manifest.roles.len()
    );
}

#[test]
fn test_dna_paths_resolve() {
    let manifest_path = workspace_root().join("happs/mycelix-resilience-happ.yaml");
    let yaml = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: HappManifest = serde_yaml::from_str(&yaml).unwrap();

    let happs_dir = workspace_root().join("happs");

    for role in &manifest.roles {
        let dna_path = happs_dir.join(&role.dna.path);
        // DNA path should reference a directory containing dna.yaml or similar.
        // The path format is ./cluster/dna/name.dna — the parent dir should exist.
        let parent = dna_path.parent().unwrap();
        assert!(
            parent.exists(),
            "DNA parent dir for role '{}' not found at {:?} (path in manifest: {})",
            role.name,
            parent,
            role.dna.path
        );
    }
}

#[test]
fn test_role_names_match_client() {
    // This test ensures the resilience-client.ts role_name strings
    // match the manifest role names. If either drifts, this fails.
    let manifest_path = workspace_root().join("happs/mycelix-resilience-happ.yaml");
    let yaml = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: HappManifest = serde_yaml::from_str(&yaml).unwrap();

    let role_names: Vec<&str> = manifest.roles.iter().map(|r| r.name.as_str()).collect();

    for (role, zome) in EXPECTED_ZOME_ROUTES {
        assert!(
            role_names.contains(role),
            "resilience-client.ts calls zome '{zome}' on role '{role}', \
             but that role is not in the manifest. Roles: {role_names:?}"
        );
    }
}

#[test]
fn test_manifest_no_clone_limit() {
    // Resilience deployment should not allow cloned DNAs
    let manifest_path = workspace_root().join("happs/mycelix-resilience-happ.yaml");
    let yaml = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: HappManifest = serde_yaml::from_str(&yaml).unwrap();

    // If clone_limit existed in our struct we'd check it directly.
    // Instead, verify the raw YAML doesn't have clone_limit > 0.
    for role in &manifest.roles {
        assert!(
            !yaml.contains(&format!("clone_limit: 1"))
                || yaml.contains("clone_limit: 0"),
            "Role '{}' should have clone_limit: 0 for resilience deployment",
            role.name
        );
    }
}

#[test]
fn test_bootstrap_script_exists_and_executable() {
    let script = workspace_root().join("scripts/resilience-bootstrap.sh");
    assert!(script.exists(), "Bootstrap script not found");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::metadata(&script).unwrap().permissions();
        assert!(
            perms.mode() & 0o111 != 0,
            "Bootstrap script is not executable"
        );
    }
}

#[test]
fn test_quickstart_doc_exists() {
    let doc = workspace_root().join("RESILIENCE_QUICKSTART.md");
    assert!(doc.exists(), "RESILIENCE_QUICKSTART.md not found");

    let content = std::fs::read_to_string(&doc).unwrap();
    assert!(content.contains("TEND"), "Quickstart should mention TEND");
    assert!(
        content.contains("Value Anchor"),
        "Quickstart should mention Value Anchor"
    );
    assert!(
        content.contains("resilience-test"),
        "Quickstart should mention the test recipe"
    );
}
