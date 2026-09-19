// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;

const SPEC: &str = include_str!("../references/de001a_a2e_execution_contract_v1.json");

fn nested<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}

#[test]
fn a2e_contract_freezes_raw_execution_without_reproduction_judgment() {
    let spec: Value = serde_json::from_str(SPEC).expect("valid A2E specification JSON");
    assert_eq!(
        spec["protocol"],
        "DE-001A2E-EXECUTION-CONTRACT-v1"
    );
    assert_eq!(spec["status"], "preregistered-executor-contract-only");
    assert_eq!(spec["scientific_claim"], "NONE");
    assert_eq!(spec["authority"], "raw-optimizer-execution-contract-only");

    let auth = nested(&spec, "required_authorization");
    assert_eq!(auth["verdict"], "PASS");
    assert_eq!(auth["optimizer_execution_authorized"], true);
    assert_eq!(auth["a2_execution_authorized"], true);
    assert_eq!(auth["normalization_or_migration_authorized"], false);
    assert_eq!(auth["optimizer_invocation_budget"], 1);

    let policy = nested(&spec, "execution_policy");
    assert_eq!(policy["optimizer_invocations_per_executor_process"], 1);
    assert_eq!(policy["automatic_retry_allowed"], false);
    assert_eq!(policy["preflight_optimizer_invocation_allowed"], false);
    assert_eq!(policy["dry_run_optimizer_invocation_allowed"], false);
    assert_eq!(policy["retry_requires_new_authorization_receipt"], true);
    assert_eq!(policy["normalization_or_migration_allowed"], false);
    assert_eq!(policy["network_allowed"], false);
    assert_eq!(policy["runtime_package_installation_allowed"], false);

    let raw = nested(&spec, "required_raw_result");
    assert_eq!(raw["protocol"], "DE-001A2E-RAW-OPTIMIZER-EXECUTION-v1");
    assert_eq!(raw["reproduction_verdict"], "UNASSESSED");
    assert_eq!(raw["must_record_actual_optimizer_invocation_count"], true);
    assert_eq!(raw["must_record_process_exit_code"], true);
    assert_eq!(raw["must_record_result_file_manifest"], true);

    let boundary = nested(&spec, "interpretation_boundary");
    assert_eq!(boundary["executor_may_classify_reproduction"], false);
    assert_eq!(boundary["executor_may_change_historical_tolerances"], false);
    assert_eq!(boundary["executor_may_select_success_metric"], false);
    assert_eq!(boundary["a2q_execution_authorized"], false);
}
