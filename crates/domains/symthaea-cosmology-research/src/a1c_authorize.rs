// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1C fixed-point execution authorization firewall.
//!
//! This program performs no cosmology and executes no Python. It combines a
//! qualified A1Q evidence bundle, qualified reusable A1E environment evidence,
//! and a valid A1C contract receipt into a narrowly scoped authorization for
//! exactly one fixed-point likelihood call in one executor process.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A1C-EXECUTION-AUTHORIZATION-v1";
const AUTHORITY: &str = "fixed-point-execution-authorization-only";
const A1Q_PROTOCOL: &str = "DE-001A1Q-BUNDLE-INTEGRITY-v1";
const A1E_PROTOCOL: &str = "DE-001A1E-ENVIRONMENT-EVIDENCE-v1";
const A1C_CONTRACT_PROTOCOL: &str = "DE-001A1C-CONTRACT-CONSISTENCY-v1";
const A1C_MANIFEST_PROTOCOL: &str = "DE-001A1C-COBAYA-FIXED-POINT-v1";
const DEFINITION_PROTOCOL: &str = "DE-001A-ENVIRONMENT-DEFINITION-v1";
const NIX_VERSION: &str = "nix (Nix) 2.34.7";
const A1C_MANIFEST_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/references/de001a_a1c_cobaya_fixed_point_v1.json";
const AUTH_SPEC_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/references/de001a_a1c_authorization_v1.json";
const FLAKE_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/nix/flake.nix";
const LOCK_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/nix/flake.lock";
const CLOSURE_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/references/de001a_execution_closure_v2.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationSpec {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    authority: String,
    status: String,
    required_a1q: RequiredA1q,
    required_a1e: RequiredA1e,
    required_a1c_contract: RequiredA1cContract,
    execution_budget: ExecutionBudget,
    authorization_semantics: AuthorizationSemantics,
    promotion: Promotion,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequiredA1q {
    protocol: String,
    verdict: String,
    authority: String,
    accepted_reproduction_verdicts: Vec<String>,
    subject_scope: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequiredA1e {
    protocol: String,
    verdict: String,
    authority: String,
    environment_reuse_authorized: bool,
    a1c_execution_authorized: bool,
    environment_scope: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequiredA1cContract {
    protocol: String,
    verdict: String,
    authority: String,
    execution_authorized: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionBudget {
    likelihood_calls: u64,
    sampler_allowed: bool,
    minimizer_allowed: bool,
    optimization_allowed: bool,
    parameter_mutation_allowed: bool,
    network_allowed: bool,
    camb_allowed: bool,
    runtime_package_installation_allowed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationSemantics {
    scope: String,
    stateful_global_single_use_claimed: bool,
    executor_must_bind_authorization_receipt_sha256: bool,
    executor_must_record_actual_likelihood_call_count: bool,
    executor_must_refuse_call_count_other_than_one: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Promotion {
    scientific_claim: String,
    permits: String,
    does_not_permit: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a1c_execution_authorized: bool,
    subject_head: String,
    subject_tree: String,
    authorization_spec_sha256: String,
    a1c_manifest_sha256: String,
    a1q_receipt_sha256: String,
    a1e_receipt_sha256: String,
    a1c_contract_receipt_sha256: String,
    a1q_reproduction_verdict: String,
    environment_definition_sha256: String,
    likelihood_call_budget: u64,
    sampler_allowed: bool,
    minimizer_allowed: bool,
    optimization_allowed: bool,
    parameter_mutation_allowed: bool,
    network_allowed: bool,
    camb_allowed: bool,
    runtime_package_installation_allowed: bool,
    authorization_scope: String,
    stateful_global_single_use_claimed: bool,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a1c_execution_authorized: bool,
    error: String,
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > MAX_JSON_BYTES {
        return Err(format!("{}: file exceeds {MAX_JSON_BYTES} bytes", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn run_text(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn repository_root_and_subject() -> Result<(PathBuf, String, String), String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if root.is_empty() || head.is_empty() || tree.is_empty() {
        return Err("git returned an empty repository identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
}

fn require_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))
}

fn require_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean field: {key}"))
}

fn require_exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let required: BTreeSet<_> = expected.iter().copied().collect();
    if actual != required {
        return Err(format!("{label} field set differs from the frozen schema"));
    }
    Ok(())
}

fn compute_environment_definition(root: &Path) -> Result<String, String> {
    let flake_nix = read_regular_file(&root.join(FLAKE_RELATIVE))?;
    let flake_lock = read_regular_file(&root.join(LOCK_RELATIVE))?;
    let closure = read_regular_file(&root.join(CLOSURE_RELATIVE))?;
    let definition = format!(
        "protocol={DEFINITION_PROTOCOL}\nplatform=x86_64-linux\nnix_version={NIX_VERSION}\nflake_nix_sha256={}\nflake_lock_sha256={}\nclosure_manifest_sha256={}\n",
        sha256_hex(&flake_nix),
        sha256_hex(&flake_lock),
        sha256_hex(&closure)
    );
    Ok(sha256_hex(definition.as_bytes()))
}

fn validate_spec(spec: &AuthorizationSpec) -> Result<(), String> {
    let expected_negative_permissions = [
        "sampling",
        "minimization",
        "optimization",
        "parameter search",
        "A1X promotion",
        "A2 optimizer reproduction",
        "cosmological interpretation",
    ];
    if spec.schema_version != 1
        || spec.protocol != PROTOCOL
        || spec.scientific_claim != "NONE"
        || spec.authority != AUTHORITY
        || spec.status != "preregistered-authorization-firewall"
        || spec.required_a1q.protocol != A1Q_PROTOCOL
        || spec.required_a1q.verdict != "PASS"
        || spec.required_a1q.authority != "evidence-bundle-integrity-only"
        || spec.required_a1q.accepted_reproduction_verdicts != ["PASS", "NEGATIVE"]
        || spec.required_a1q.subject_scope != "current-exact-head-and-tree"
        || spec.required_a1e.protocol != A1E_PROTOCOL
        || spec.required_a1e.verdict != "PASS"
        || spec.required_a1e.authority != "environment-evidence-reuse-only"
        || !spec.required_a1e.environment_reuse_authorized
        || spec.required_a1e.a1c_execution_authorized
        || spec.required_a1e.environment_scope != "current-environment-definition"
        || spec.required_a1c_contract.protocol != A1C_CONTRACT_PROTOCOL
        || spec.required_a1c_contract.verdict != "PASS"
        || spec.required_a1c_contract.authority != "contract-consistency-only"
        || spec.required_a1c_contract.execution_authorized
        || spec.execution_budget.likelihood_calls != 1
        || spec.execution_budget.sampler_allowed
        || spec.execution_budget.minimizer_allowed
        || spec.execution_budget.optimization_allowed
        || spec.execution_budget.parameter_mutation_allowed
        || spec.execution_budget.network_allowed
        || spec.execution_budget.camb_allowed
        || spec.execution_budget.runtime_package_installation_allowed
        || spec.authorization_semantics.scope != "one-fixed-point-executor-process"
        || spec.authorization_semantics.stateful_global_single_use_claimed
        || !spec.authorization_semantics.executor_must_bind_authorization_receipt_sha256
        || !spec.authorization_semantics.executor_must_record_actual_likelihood_call_count
        || !spec.authorization_semantics.executor_must_refuse_call_count_other_than_one
        || spec.promotion.scientific_claim != "NONE"
        || spec.promotion.permits != "A1C fixed-point execution only"
        || spec.promotion.does_not_permit.iter().map(String::as_str).collect::<Vec<_>>()
            != expected_negative_permissions
    {
        return Err("authorization specification differs from the frozen firewall".into());
    }
    Ok(())
}

fn accepted_reproduction_verdict(value: &str) -> bool {
    matches!(value, "PASS" | "NEGATIVE")
}

fn validate_a1q(value: &Value, head: &str, tree: &str) -> Result<String, String> {
    require_exact_keys(
        value,
        &[
            "protocol", "verdict", "scientific_claim", "authority", "bundle_sha256",
            "subject_head", "subject_tree", "reproduction_verdict", "cargo_lock_sha256",
            "binary_sha256", "a0_nar_hash", "a0_manifest_sha256", "a1n_spec_sha256",
            "a0_receipt_sha256", "a1p_receipt_sha256", "a1r_primary_receipt_sha256",
            "a1r_refined_receipt_sha256", "a1n_receipt_sha256",
        ],
        "A1Q receipt",
    )?;
    if require_str(value, "protocol")? != A1Q_PROTOCOL
        || require_str(value, "verdict")? != "PASS"
        || require_str(value, "scientific_claim")? != "NONE"
        || require_str(value, "authority")? != "evidence-bundle-integrity-only"
        || require_str(value, "subject_head")? != head
        || require_str(value, "subject_tree")? != tree
    {
        return Err("A1Q receipt does not qualify the current exact subject".into());
    }
    let reproduction = require_str(value, "reproduction_verdict")?;
    if !accepted_reproduction_verdict(reproduction) {
        return Err("A1Q reproduction verdict is neither PASS nor NEGATIVE".into());
    }
    Ok(reproduction.to_owned())
}

fn validate_a1e(value: &Value, environment_definition: &str) -> Result<(), String> {
    require_exact_keys(
        value,
        &[
            "protocol", "verdict", "scientific_claim", "authority",
            "environment_reuse_authorized", "a1c_execution_authorized",
            "qualification_receipt_sha256", "versions_sha256",
            "current_environment_definition_sha256", "qualifying_source_head",
            "realized_environment_store_path", "realized_environment_nar_hash",
            "realized_environment_closure_size", "nix_version",
        ],
        "A1E receipt",
    )?;
    if require_str(value, "protocol")? != A1E_PROTOCOL
        || require_str(value, "verdict")? != "PASS"
        || require_str(value, "scientific_claim")? != "NONE"
        || require_str(value, "authority")? != "environment-evidence-reuse-only"
        || !require_bool(value, "environment_reuse_authorized")?
        || require_bool(value, "a1c_execution_authorized")?
        || require_str(value, "current_environment_definition_sha256")? != environment_definition
        || require_str(value, "nix_version")? != NIX_VERSION
    {
        return Err("A1E receipt does not authorize reuse of the current environment".into());
    }
    Ok(())
}

fn validate_a1c_contract(value: &Value, manifest_sha256: &str) -> Result<(), String> {
    require_exact_keys(
        value,
        &[
            "protocol", "verdict", "scientific_claim", "authority", "execution_authorized",
            "a1c_manifest_sha256", "a1r_manifest_sha256", "cobaya_version",
            "cobaya_source_commit", "subject",
        ],
        "A1C contract receipt",
    )?;
    if require_str(value, "protocol")? != A1C_CONTRACT_PROTOCOL
        || require_str(value, "verdict")? != "PASS"
        || require_str(value, "scientific_claim")? != "NONE"
        || require_str(value, "authority")? != "contract-consistency-only"
        || require_bool(value, "execution_authorized")?
        || require_str(value, "a1c_manifest_sha256")? != manifest_sha256
        || require_str(value, "cobaya_version")? != "3.6.2"
        || require_str(value, "cobaya_source_commit")? != "899f30a49f85de610dac321e91a1af50018e56aa"
    {
        return Err("A1C contract receipt does not match the frozen current A1C subject".into());
    }
    Ok(())
}

fn validate_a1c_manifest(value: &Value) -> Result<(), String> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(1)
        || require_str(value, "protocol")? != A1C_MANIFEST_PROTOCOL
        || require_str(value, "scientific_claim")? != "NONE"
        || require_str(value, "authority")? != "released-likelihood-fixed-point-reproduction-only"
        || require_str(value, "status")? != "preregistered-contract-only"
    {
        return Err("current A1C manifest identity is invalid".into());
    }
    Ok(())
}

fn execute(a1q_path: &Path, a1e_path: &Path, a1c_contract_path: &Path) -> Result<Receipt, String> {
    let (root, head, tree) = repository_root_and_subject()?;
    let spec_bytes = read_regular_file(&root.join(AUTH_SPEC_RELATIVE))?;
    let manifest_bytes = read_regular_file(&root.join(A1C_MANIFEST_RELATIVE))?;
    let a1q_bytes = read_regular_file(a1q_path)?;
    let a1e_bytes = read_regular_file(a1e_path)?;
    let a1c_contract_bytes = read_regular_file(a1c_contract_path)?;

    let spec: AuthorizationSpec = serde_json::from_slice(&spec_bytes)
        .map_err(|error| format!("invalid authorization specification JSON: {error}"))?;
    validate_spec(&spec)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid A1C manifest JSON: {error}"))?;
    validate_a1c_manifest(&manifest)?;
    let a1q: Value = serde_json::from_slice(&a1q_bytes)
        .map_err(|error| format!("invalid A1Q receipt JSON: {error}"))?;
    let a1e: Value = serde_json::from_slice(&a1e_bytes)
        .map_err(|error| format!("invalid A1E receipt JSON: {error}"))?;
    let a1c_contract: Value = serde_json::from_slice(&a1c_contract_bytes)
        .map_err(|error| format!("invalid A1C contract receipt JSON: {error}"))?;

    let environment_definition = compute_environment_definition(&root)?;
    let reproduction_verdict = validate_a1q(&a1q, &head, &tree)?;
    validate_a1e(&a1e, &environment_definition)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    validate_a1c_contract(&a1c_contract, &manifest_sha256)?;

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        a1c_execution_authorized: true,
        subject_head: head,
        subject_tree: tree,
        authorization_spec_sha256: sha256_hex(&spec_bytes),
        a1c_manifest_sha256: manifest_sha256,
        a1q_receipt_sha256: sha256_hex(&a1q_bytes),
        a1e_receipt_sha256: sha256_hex(&a1e_bytes),
        a1c_contract_receipt_sha256: sha256_hex(&a1c_contract_bytes),
        a1q_reproduction_verdict: reproduction_verdict,
        environment_definition_sha256: environment_definition,
        likelihood_call_budget: spec.execution_budget.likelihood_calls,
        sampler_allowed: false,
        minimizer_allowed: false,
        optimization_allowed: false,
        parameter_mutation_allowed: false,
        network_allowed: false,
        camb_allowed: false,
        runtime_package_installation_allowed: false,
        authorization_scope: spec.authorization_semantics.scope,
        stateful_global_single_use_claimed: false,
    })
}

fn usage(program: &str) {
    eprintln!("usage: {program} <a1q-receipt.json> <a1e-receipt.json> <a1c-contract-receipt.json>");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        usage(args.first().map(String::as_str).unwrap_or("de001a-a1c-authorize"));
        std::process::exit(2);
    }

    let result = execute(Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]));
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match result {
        Ok(receipt) => {
            serde_json::to_writer(&mut out, &receipt).expect("serialize authorization receipt");
            writeln!(out).expect("write newline");
        }
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                a1c_execution_authorized: false,
                error,
            };
            serde_json::to_writer(&mut out, &receipt).expect("serialize invalid receipt");
            writeln!(out).expect("write newline");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_and_negative_are_both_eligible_for_independent_a1c_diagnosis() {
        assert!(accepted_reproduction_verdict("PASS"));
        assert!(accepted_reproduction_verdict("NEGATIVE"));
        assert!(!accepted_reproduction_verdict("INVALID"));
        assert!(!accepted_reproduction_verdict("INDETERMINATE"));
    }

    #[test]
    fn frozen_budget_never_claims_global_single_use() {
        let spec: AuthorizationSpec = serde_json::from_str(include_str!(
            "../references/de001a_a1c_authorization_v1.json"
        ))
        .expect("authorization specification must parse");
        validate_spec(&spec).expect("frozen authorization specification must validate");
        assert_eq!(spec.execution_budget.likelihood_calls, 1);
        assert!(!spec.authorization_semantics.stateful_global_single_use_claimed);
    }
}
