// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;

const CONTRACT: &str = include_str!(
    "../references/de001a_a2ei_executor_implementation_contract_v1.json"
);
const SOURCE: &str = include_str!("../src/bin/de001a-a2ei-contract.rs");

fn parsed() -> Value {
    serde_json::from_str(CONTRACT).expect("valid A2EI contract JSON")
}

#[test]
fn a2ei_is_contract_only_and_grants_no_execution_authority() {
    let value = parsed();
    assert_eq!(
        value["protocol"].as_str(),
        Some("DE-001A2EI-EXECUTOR-IMPLEMENTATION-CONTRACT-v1")
    );
    assert_eq!(value["status"].as_str(), Some("preregistered-contract-only"));
    assert_eq!(value["scientific_claim"].as_str(), Some("NONE"));

    let promotion = &value["promotion_boundary"];
    for key in [
        "contract_consistency_establishes_executor_implementation_qualification",
        "executor_implementation_qualified",
        "optimizer_execution_authorized",
        "a2_execution_authorized",
        "real_optimizer_execution_authorized",
        "a2q_execution_authorized",
    ] {
        assert_eq!(promotion[key].as_bool(), Some(false), "{key}");
    }
    assert_eq!(promotion["scientific_claim"].as_str(), Some("NONE"));

    assert!(SOURCE.contains("\"executor_implementation_qualified\": false"));
    assert!(SOURCE.contains("\"real_optimizer_execution_authorized\": false"));
}

#[test]
fn a2ei_freezes_single_invocation_and_no_retry() {
    let value = parsed();
    let behavior = &value["behavior_theorem"];
    assert_eq!(
        behavior["optimizer_invocations_per_executor_process"].as_u64(),
        Some(1)
    );
    assert_eq!(
        behavior["preflight_optimizer_invocation_allowed"].as_bool(),
        Some(false)
    );
    assert_eq!(
        behavior["dry_run_optimizer_invocation_allowed"].as_bool(),
        Some(false)
    );
    assert_eq!(behavior["automatic_retry_allowed"].as_bool(), Some(false));
    assert_eq!(
        behavior["retry_requires_new_authorization_receipt"].as_bool(),
        Some(true)
    );
    assert_eq!(
        behavior["executor_may_classify_reproduction"].as_bool(),
        Some(false)
    );
    assert_eq!(
        behavior["executor_may_embed_historical_reproduction_tolerances"].as_bool(),
        Some(false)
    );
}

#[test]
fn a2ei_freezes_fail_closed_result_tree_confinement() {
    let value = parsed();
    let fs = &value["filesystem_confinement"];
    for key in [
        "result_root_must_be_real_directory",
        "manifest_path_components_must_be_normal_only",
        "intermediate_components_must_be_directories",
        "terminal_entries_must_be_regular_files",
    ] {
        assert_eq!(fs[key].as_bool(), Some(true), "{key}");
    }
    for key in [
        "result_root_symlink_allowed",
        "curdir_component_allowed",
        "parent_component_allowed",
        "root_or_prefix_component_allowed",
        "intermediate_symlink_allowed",
        "terminal_symlink_allowed",
        "duplicate_roles_allowed",
        "duplicate_paths_allowed",
    ] {
        assert_eq!(fs[key].as_bool(), Some(false), "{key}");
    }

    let fixtures = value["fixture_matrix"].as_array().expect("fixture array");
    for required in [
        "final-file-symlink",
        "intermediate-directory-symlink",
        "result-root-symlink",
        "curdir-path-alias",
        "parent-path-escape",
    ] {
        assert!(
            fixtures.iter().any(|entry| entry.as_str() == Some(required)),
            "missing fixture {required}"
        );
    }
}

#[test]
fn a2ei_requires_enforceable_runtime_isolation() {
    let value = parsed();
    let isolation = &value["runtime_isolation"];
    assert_eq!(isolation["network_allowed"].as_bool(), Some(false));
    assert_eq!(
        isolation["runtime_package_installation_allowed"].as_bool(),
        Some(false)
    );
    assert_eq!(
        isolation["enforceable_network_isolation_required"].as_bool(),
        Some(true)
    );
    assert_eq!(
        isolation["policy_flag_only_is_sufficient"].as_bool(),
        Some(false)
    );
    assert_eq!(
        isolation["real_execution_authority_if_isolation_unproven"].as_bool(),
        Some(false)
    );
}

#[test]
fn a2ei_raw_result_must_bind_exact_qualified_executor() {
    let value = parsed();
    let binding = &value["required_raw_result_binding"];
    for key in [
        "must_bind_a2ei_qualification_receipt_sha256",
        "must_bind_executor_source_head",
        "must_bind_executor_source_tree",
        "must_bind_executor_executable_sha256",
        "must_bind_execution_contract_receipt_sha256",
        "must_bind_authorization_receipt_sha256",
        "must_bind_scientific_subject_head_tree",
        "must_bind_effective_sampler_sha256",
        "must_bind_sampled_coordinate_order",
    ] {
        assert_eq!(binding[key].as_bool(), Some(true), "{key}");
    }
    assert_eq!(binding["reproduction_verdict"].as_str(), Some("UNASSESSED"));
    assert_eq!(binding["scientific_claim"].as_str(), Some("NONE"));
}
