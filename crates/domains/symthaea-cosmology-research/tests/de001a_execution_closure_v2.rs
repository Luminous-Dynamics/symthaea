// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_cosmology_research::identity::{GitObjectId, Sha256Digest};

const CLOSURE: &str = include_str!("../references/de001a_execution_closure_v2.json");

fn closure() -> Value {
    serde_json::from_str(CLOSURE).expect("v2 execution closure must remain valid JSON")
}

#[test]
fn v2_is_implemented_but_still_unqualified() {
    let value = closure();
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["status"], "closure-implementation-frozen-unqualified");
    assert_eq!(value["execution_authority"], false);
    assert_eq!(value["parent_closure"], "DE-001A-EXECUTION-CLOSURE-v1");
}

#[test]
fn every_external_source_has_sha256_and_sri_identity() {
    let value = closure();
    let sources = value["external_sources"].as_array().unwrap();
    assert_eq!(sources.len(), 4);

    for source in sources {
        GitObjectId::parse(source["git_commit"].as_str().unwrap()).unwrap();
        Sha256Digest::parse(source["sha256"].as_str().unwrap()).unwrap();
        let sri = source["nix_sri"].as_str().unwrap();
        assert!(sri.starts_with("sha256-"));
        assert!(sri.len() > "sha256-".len());
    }
}

#[test]
fn selected_source_hashes_are_frozen_exactly() {
    let value = closure();
    let sources = value["external_sources"].as_array().unwrap();
    let hash_for = |name: &str| {
        sources
            .iter()
            .find(|source| source["name"] == name)
            .and_then(|source| source["sha256"].as_str())
            .unwrap()
    };

    assert_eq!(hash_for("cobaya"), "8f1061d6347427f08380e1e0c0b766d695d3978b5439fb0b1cc1a7002152d9c8");
    assert_eq!(hash_for("camb"), "9856202a5c05570256e52377b20431891c7b08b2e9c334e141fd08d2a085516f");
    assert_eq!(hash_for("getdist"), "1bd69c9748891fa1dc516e2474b30b3afa083d2975531c09c4377ae30e8636ac");
    assert_eq!(hash_for("py-bobyqa"), "f698848e372fa0625fb9fd3a7a8b4f557804d7858b23a45d8187ffaea6341e33");
}

#[test]
fn only_execution_qualification_and_realized_identity_remain_blocked() {
    let value = closure();
    let blockers: Vec<&str> = value["blockers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        blockers,
        vec![
            "nix-environment-check-not-yet-qualified",
            "realized-closure-identity-not-yet-recorded"
        ]
    );
}

#[test]
fn workflow_dependencies_are_commit_pinned() {
    let value = closure();
    let workflow = &value["qualification_workflow"];
    for key in ["checkout_commit", "nix_installer_commit", "artifact_upload_commit"] {
        GitObjectId::parse(workflow[key].as_str().unwrap()).unwrap();
    }
}

#[test]
fn promotion_still_requires_a_realized_qualified_closure() {
    let value = closure();
    let rule = &value["promotion_rule"];
    assert_eq!(rule["required_status"], "closure-realized-qualified");
    assert_eq!(rule["required_execution_authority"], true);
    assert_eq!(rule["requires_environment_check_pass"], true);
    assert_eq!(rule["requires_realized_store_path"], true);
    assert_eq!(rule["requires_realized_nar_hash"], true);
    assert_eq!(rule["requires_zero_blockers"], true);
}
