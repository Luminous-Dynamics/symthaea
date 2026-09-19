// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;

const SPEC: &str = include_str!("../references/de001a_a2_authorization_v1.json");

#[test]
fn a2_authorization_requires_agreement_and_exact_method_provenance() {
    let value: Value = serde_json::from_str(SPEC).expect("A2 authorization spec must parse");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["protocol"],
        "DE-001A2-EXECUTION-AUTHORIZATION-v1"
    );
    assert_eq!(value["status"], "preregistered-authorization-firewall");
    assert_eq!(value["scientific_claim"], "NONE");
    assert_eq!(
        value["authority"],
        "exact-optimizer-execution-authorization-only"
    );

    let a1x = &value["required_a1x"];
    assert_eq!(a1x["protocol"], "DE-001A1X-CROSS-IMPLEMENTATION-v1");
    assert_eq!(a1x["capsule_protocol"], "DE-001A1X-COMPARISON-CAPSULE-v1");
    assert_eq!(a1x["verdict"], "AGREE");
    assert_eq!(a1x["comparison_exit_code"], "0");
    assert_eq!(a1x["a2_execution_authorized"], false);

    let a2m = &value["required_a2m"];
    assert_eq!(a2m["protocol"], "DE-001A2M-OPTIMIZER-METHOD-BINDING-v1");
    assert_eq!(a2m["capsule_protocol"], "DE-001A2M-QUALIFICATION-CAPSULE-v1");
    assert_eq!(a2m["verdict"], "PASS");
    assert_eq!(a2m["optimizer_execution_authorized"], false);
    assert_eq!(a2m["a2_execution_authorized"], false);

    let lineage = &value["lineage"];
    for key in [
        "a1x_subject_head_must_equal_a2r_subject_head",
        "a1x_subject_tree_must_equal_a2r_subject_tree",
        "a2m_a2r_subject_head_must_equal_a2r_subject_head",
        "a2m_receipt_must_match_a2m_capsule_hash",
        "a2m_a0_a2r_a2g_hashes_must_match_capsule",
        "a2r_receipt_hash_must_match_a2m_binding",
        "effective_sampler_hash_must_match_a2m_capsule",
    ] {
        assert_eq!(lineage[key], true, "{key} must remain true");
    }
}

#[test]
fn a2_authorization_is_exact_and_does_not_authorize_migration() {
    let value: Value = serde_json::from_str(SPEC).expect("A2 authorization spec must parse");
    let budget = &value["execution_budget"];
    assert_eq!(budget["optimizer_invocations_per_executor_process"], 1);
    assert_eq!(budget["exact_effective_sampler_sha256_required"], true);
    assert_eq!(budget["exact_sampled_coordinate_order_required"], true);

    for key in [
        "backend_substitution_allowed",
        "method_substitution_allowed",
        "option_addition_allowed",
        "option_deletion_allowed",
        "option_rewrite_allowed",
        "coordinate_addition_allowed",
        "coordinate_deletion_allowed",
        "coordinate_reordering_allowed",
        "normalization_or_migration_authorized",
        "network_allowed",
        "runtime_package_installation_allowed",
    ] {
        assert_eq!(budget[key], false, "{key} must remain false");
    }

    let semantics = &value["authorization_semantics"];
    assert_eq!(semantics["stateful_global_single_use_claimed"], false);
    assert_eq!(semantics["executor_must_bind_authorization_receipt_sha256"], true);
    assert_eq!(semantics["executor_must_record_actual_optimizer_invocation_count"], true);
    assert_eq!(semantics["executor_must_refuse_invocation_count_other_than_one"], true);
    assert_eq!(semantics["executor_must_bind_exact_effective_sampler_sha256"], true);
    assert_eq!(semantics["executor_must_bind_exact_sampled_coordinate_order"], true);

    let promotion = &value["promotion"];
    assert_eq!(promotion["optimizer_execution_authorized"], true);
    assert_eq!(promotion["a2_execution_authorized"], true);
    assert_eq!(promotion["scientific_claim"], "NONE");
}
