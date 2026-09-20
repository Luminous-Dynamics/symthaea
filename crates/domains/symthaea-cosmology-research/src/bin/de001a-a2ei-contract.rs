// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A2EI executor-implementation qualification contract consistency.
//!
//! This binary performs no cosmology, likelihood evaluation, sampler
//! construction, minimization, optimizer execution, or executor qualification.
//! It validates only the preregistered contract that a future exact executor
//! implementation must satisfy before it can receive real A2 execution authority.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CONTRACT: &str = include_str!(
    "../../references/de001a_a2ei_executor_implementation_contract_v1.json"
);
const PROTOCOL: &str = "DE-001A2EI-EXECUTOR-IMPLEMENTATION-CONTRACT-v1";
const AUTHORITY: &str =
    "raw-optimizer-executor-implementation-qualification-contract-only";
const RECEIPT_PROTOCOL: &str = "DE-001A2EI-CONTRACT-CONSISTENCY-v1";
const RECEIPT_AUTHORITY: &str = "contract-consistency-only";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn nested<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing nested field: {key}"))
}

fn expect_str(value: &Value, key: &str, expected: &str) -> Result<(), String> {
    let observed = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))?;
    if observed != expected {
        return Err(format!("field {key} drifted: expected {expected:?}, observed {observed:?}"));
    }
    Ok(())
}

fn expect_bool(value: &Value, key: &str, expected: bool) -> Result<(), String> {
    let observed = value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean field: {key}"))?;
    if observed != expected {
        return Err(format!("field {key} drifted: expected {expected}, observed {observed}"));
    }
    Ok(())
}

fn expect_u64(value: &Value, key: &str, expected: u64) -> Result<(), String> {
    let observed = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or non-u64 field: {key}"))?;
    if observed != expected {
        return Err(format!("field {key} drifted: expected {expected}, observed {observed}"));
    }
    Ok(())
}

fn expect_bool_keys(value: &Value, keys: &[&str], expected: bool) -> Result<(), String> {
    for key in keys {
        expect_bool(value, key, expected)?;
    }
    Ok(())
}

fn validate_contract(contract: &Value) -> Result<(), String> {
    if contract.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("schema_version must remain 1".into());
    }
    expect_str(contract, "protocol", PROTOCOL)?;
    expect_str(contract, "status", "preregistered-contract-only")?;
    expect_str(contract, "scientific_claim", "NONE")?;
    expect_str(contract, "authority", AUTHORITY)?;

    let upstream = nested(contract, "required_upstream")?;
    expect_str(
        upstream,
        "qualification_capsule_protocol",
        "DE-001A2E-CONTRACT-QUALIFICATION-CAPSULE-v1",
    )?;
    expect_str(
        upstream,
        "execution_contract_protocol",
        "DE-001A2E-EXECUTION-CONTRACT-v1",
    )?;
    expect_str(upstream, "execution_contract_verdict", "PASS")?;
    expect_str(
        upstream,
        "execution_contract_authority",
        "raw-optimizer-execution-contract-only",
    )?;
    expect_str(
        upstream,
        "raw_result_protocol",
        "DE-001A2E-RAW-OPTIMIZER-EXECUTION-v1",
    )?;
    expect_bool(upstream, "executor_implementation_authorized", false)?;
    expect_str(upstream, "reproduction_verdict", "UNASSESSED")?;
    expect_u64(upstream, "optimizer_invocation_budget", 1)?;
    expect_bool_keys(
        upstream,
        &[
            "automatic_retry_allowed",
            "normalization_or_migration_allowed",
            "network_allowed",
            "runtime_package_installation_allowed",
        ],
        false,
    )?;

    let identity = nested(contract, "exact_executor_identity")?;
    expect_bool_keys(
        identity,
        &[
            "source_head_required",
            "source_tree_required",
            "content_sha256_root_required",
            "release_executable_sha256_required",
            "release_executable_size_required",
            "cargo_lock_sha256_required",
            "rust_toolchain_identity_required",
            "runtime_closure_identity_required",
            "execution_contract_sha256_required",
            "raw_result_protocol_identity_required",
            "command_construction_identity_required",
            "postflight_implementation_identity_required",
        ],
        true,
    )?;

    let behavior = nested(contract, "behavior_theorem")?;
    expect_u64(behavior, "optimizer_invocations_per_executor_process", 1)?;
    expect_bool_keys(
        behavior,
        &[
            "retry_requires_new_authorization_receipt",
            "successful_fixture_exit_maps_to_executed",
            "nonzero_fixture_exit_maps_to_execution_error",
            "invalid_local_evidence_maps_to_invalid",
            "command_argv_digest_must_bind_exact_launched_argv",
            "stdout_digest_must_bind_exact_captured_bytes",
            "stderr_digest_must_bind_exact_captured_bytes",
            "result_manifest_must_bind_exact_role_path_size_sha256",
            "postflight_source_identity_must_equal_preflight",
            "postflight_executable_sha256_must_equal_preflight",
        ],
        true,
    )?;
    expect_bool_keys(
        behavior,
        &[
            "preflight_optimizer_invocation_allowed",
            "dry_run_optimizer_invocation_allowed",
            "automatic_retry_allowed",
            "executor_may_classify_reproduction",
            "executor_may_embed_historical_reproduction_tolerances",
            "executor_may_select_scientific_success_metric",
        ],
        false,
    )?;

    let expected_fixtures = [
        "success-single-invocation",
        "nonzero-exit-no-retry",
        "invalid-local-evidence",
        "attempted-second-invocation",
        "argv-digest-divergence",
        "stdout-digest-divergence",
        "stderr-digest-divergence",
        "executable-postflight-mutation",
        "source-postflight-mutation",
        "duplicate-manifest-role",
        "duplicate-manifest-path",
        "final-file-symlink",
        "intermediate-directory-symlink",
        "result-root-symlink",
        "curdir-path-alias",
        "parent-path-escape",
        "forbidden-reproduction-verdict",
        "execution-contract-hash-mismatch",
        "authorization-sampler-coordinate-lineage-mismatch",
    ];
    let fixtures = nested(contract, "fixture_matrix")?
        .as_array()
        .ok_or_else(|| "fixture_matrix must be an array".to_owned())?;
    if fixtures.len() != expected_fixtures.len()
        || fixtures
            .iter()
            .zip(expected_fixtures)
            .any(|(actual, expected)| actual.as_str() != Some(expected))
    {
        return Err("fixture_matrix drifted from the preregistered order/set".into());
    }

    let filesystem = nested(contract, "filesystem_confinement")?;
    expect_bool_keys(
        filesystem,
        &[
            "result_root_must_be_real_directory",
            "manifest_path_components_must_be_normal_only",
            "intermediate_components_must_be_directories",
            "terminal_entries_must_be_regular_files",
        ],
        true,
    )?;
    expect_bool_keys(
        filesystem,
        &[
            "result_root_symlink_allowed",
            "curdir_component_allowed",
            "parent_component_allowed",
            "root_or_prefix_component_allowed",
            "intermediate_symlink_allowed",
            "terminal_symlink_allowed",
            "duplicate_roles_allowed",
            "duplicate_paths_allowed",
        ],
        false,
    )?;

    let isolation = nested(contract, "runtime_isolation")?;
    expect_bool(isolation, "network_allowed", false)?;
    expect_bool(isolation, "runtime_package_installation_allowed", false)?;
    expect_bool(isolation, "enforceable_network_isolation_required", true)?;
    expect_bool(isolation, "policy_flag_only_is_sufficient", false)?;
    expect_str(
        isolation,
        "preferred_mechanism",
        "nix-sandbox-or-equivalent-fail-closed-isolation",
    )?;
    expect_bool(isolation, "real_execution_authority_if_isolation_unproven", false)?;

    let raw = nested(contract, "required_raw_result_binding")?;
    expect_bool_keys(
        raw,
        &[
            "must_bind_a2ei_qualification_receipt_sha256",
            "must_bind_executor_source_head",
            "must_bind_executor_source_tree",
            "must_bind_executor_executable_sha256",
            "must_bind_execution_contract_receipt_sha256",
            "must_bind_authorization_receipt_sha256",
            "must_bind_scientific_subject_head_tree",
            "must_bind_effective_sampler_sha256",
            "must_bind_sampled_coordinate_order",
        ],
        true,
    )?;
    expect_str(raw, "reproduction_verdict", "UNASSESSED")?;
    expect_str(raw, "scientific_claim", "NONE")?;

    let promotion = nested(contract, "promotion_boundary")?;
    expect_bool_keys(
        promotion,
        &[
            "contract_consistency_establishes_executor_implementation_qualification",
            "executor_implementation_qualified",
            "optimizer_execution_authorized",
            "a2_execution_authorized",
            "real_optimizer_execution_authorized",
            "a2q_execution_authorized",
        ],
        false,
    )?;
    expect_str(promotion, "scientific_claim", "NONE")?;
    Ok(())
}

fn receipt(verdict: &str, error: Option<String>) -> Value {
    json!({
        "protocol": RECEIPT_PROTOCOL,
        "verdict": verdict,
        "scientific_claim": "NONE",
        "authority": RECEIPT_AUTHORITY,
        "contract_protocol": PROTOCOL,
        "contract_sha256": sha256_hex(CONTRACT.as_bytes()),
        "executor_implementation_qualified": false,
        "optimizer_execution_authorized": false,
        "a2_execution_authorized": false,
        "real_optimizer_execution_authorized": false,
        "a2q_execution_authorized": false,
        "error": error,
    })
}

fn main() {
    let result = serde_json::from_str::<Value>(CONTRACT)
        .map_err(|error| format!("invalid A2EI contract JSON: {error}"))
        .and_then(|contract| validate_contract(&contract));

    match result {
        Ok(()) => println!("{}", receipt("PASS", None)),
        Err(error) => {
            println!("{}", receipt("INVALID", Some(error)));
            std::process::exit(2);
        }
    }
}
