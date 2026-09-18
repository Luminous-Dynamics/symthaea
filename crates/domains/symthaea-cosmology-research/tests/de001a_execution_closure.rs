// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use std::collections::BTreeSet;
use symthaea_cosmology_research::identity::GitObjectId;

const CLOSURE: &str = include_str!("../references/de001a_execution_closure_v1.json");

fn closure() -> Value {
    serde_json::from_str(CLOSURE).expect("execution closure manifest must remain valid JSON")
}

#[test]
fn closure_is_explicitly_non_executable_while_incomplete() {
    let value = closure();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["closure_id"], "DE-001A-EXECUTION-CLOSURE-v1");
    assert_eq!(value["status"], "closure-preregistered-incomplete");
    assert_eq!(value["execution_authority"], false);
    assert_eq!(value["claim_policy"], "reproduction-only");
    assert_eq!(
        value["parent_reference"]["repository_commit"],
        "7c1668ed64caab6dbcfde981c928c61d6da30061"
    );
}

#[test]
fn root_nix_identity_is_frozen() {
    let value = closure();
    assert_eq!(value["nix"]["root_nixpkgs_input"], "nixpkgs_2");
    assert_eq!(
        value["nix"]["nixpkgs_revision"],
        "9ae611a455b90cf061d8f332b977e387bda8e1ca"
    );
    assert_eq!(
        value["nix"]["nixpkgs_nar_hash"],
        "sha256-md8WlXOlfnIeHeOScMTTHFyf2d6iaTwPl2apR5EQ3P4="
    );
    assert_eq!(
        value["nix"]["flake_lock_git_blob"],
        "e71fcd4a705040f1bf55c8f2611fefec760a9188"
    );
}

#[test]
fn external_source_versions_and_commits_are_frozen() {
    let value = closure();
    let pins = value["external_source_pins"]
        .as_array()
        .expect("external_source_pins must be an array");

    let expected = [
        ("cobaya", "3.6.2", "899f30a49f85de610dac321e91a1af50018e56aa"),
        ("camb", "1.6.6", "3ef0272d6f7ba1231128872e56e6d4c12af8267b"),
        ("getdist", "1.7.4", "f8d5fb7f39927c199dbfa1bf87eb4fac6fe7c206"),
        ("py-bobyqa", "1.4.1", "3a3bd50732a5695a0a434cba4c1fde01c0204e08"),
    ];

    for (name, version, commit) in expected {
        let pin = pins
            .iter()
            .find(|pin| pin["name"] == name)
            .expect("required source pin missing");
        assert_eq!(pin["version"], version);
        assert_eq!(pin["commit"], commit);
        GitObjectId::parse(commit).expect("source commit must be a valid Git object id");
    }
}

#[test]
fn every_missing_fixed_output_hash_has_a_named_blocker() {
    let value = closure();
    let pins = value["external_source_pins"].as_array().unwrap();
    let blockers: BTreeSet<&str> = value["blockers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|blocker| blocker.as_str().expect("blocker must be a string"))
        .collect();

    let mut unresolved = 0;
    for pin in pins {
        if pin["fixed_output_hash"].is_null() {
            unresolved += 1;
            let name = pin["name"].as_str().unwrap();
            let expected_blocker = match name {
                "cobaya" => "cobaya-source-fixed-output-hash",
                "camb" => "camb-source-fixed-output-hash",
                "getdist" => "getdist-source-fixed-output-hash",
                "py-bobyqa" => "pybobyqa-source-fixed-output-hash",
                other => panic!("unexpected unresolved source {other}"),
            };
            assert!(blockers.contains(expected_blocker));
        }
    }

    assert_eq!(unresolved, 4);
    assert!(!value["execution_authority"].as_bool().unwrap());
}

#[test]
fn scientific_execution_is_network_closed_and_install_free() {
    let value = closure();
    assert_eq!(
        value["network_policy"]["scientific_execution"],
        "network-disabled"
    );
    assert_eq!(
        value["network_policy"]["runtime_package_installation"],
        "forbidden"
    );
    assert_eq!(
        value["network_policy"]["realization"],
        "fixed-output-fetches-only"
    );
}

#[test]
fn promotion_requires_zero_blockers_and_a_realized_qualified_closure() {
    let value = closure();
    let rule = &value["promotion_rule"];
    assert_eq!(rule["required_status"], "closure-realized-qualified");
    assert_eq!(rule["required_execution_authority"], true);
    assert_eq!(rule["requires_zero_blockers"], true);
    assert_eq!(rule["requires_all_external_fixed_output_hashes"], true);
    assert_eq!(rule["requires_version_assertions"], true);
    assert_eq!(
        rule["requires_network_disabled_scientific_execution"],
        true
    );
}

#[test]
fn independent_lane_is_not_misrepresented_as_the_desi_internal_stack() {
    let value = closure();
    assert_eq!(value["lane"]["name"], "independent-public-likelihood");
    let description = value["lane"]["description"].as_str().unwrap();
    assert!(description.contains("public Cobaya bao.desi_dr2"));
    assert!(description.contains("rather than DESI's internal"));
}
