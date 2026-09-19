// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A2 exact optimizer execution authorization firewall.
//!
//! This program performs no cosmology, likelihood evaluation, sampler
//! construction, minimization, or optimization. It combines qualified A1X
//! implementation agreement with qualified A2M optimizer provenance and emits
//! a narrowly scoped authorization for one exact optimizer invocation.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A2-EXECUTION-AUTHORIZATION-v1";
const AUTHORITY: &str = "exact-optimizer-execution-authorization-only";
const A1X_PROTOCOL: &str = "DE-001A1X-CROSS-IMPLEMENTATION-v1";
const A1X_CAPSULE_PROTOCOL: &str = "DE-001A1X-COMPARISON-CAPSULE-v1";
const A2M_PROTOCOL: &str = "DE-001A2M-OPTIMIZER-METHOD-BINDING-v1";
const A2M_CAPSULE_PROTOCOL: &str = "DE-001A2M-QUALIFICATION-CAPSULE-v1";
const A2R_CAPSULE_PROTOCOL: &str = "DE-001A2R-QUALIFICATION-CAPSULE-v1";
const SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a2_authorization_v1.json";

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    optimizer_execution_authorized: bool,
    a2_execution_authorized: bool,
    normalization_or_migration_authorized: bool,
    authorizer_head: String,
    authorizer_tree: String,
    scientific_subject_head: String,
    scientific_subject_tree: String,
    authorization_spec_sha256: String,
    a1x_receipt_sha256: String,
    a1x_capsule_sha256: String,
    a2m_receipt_sha256: String,
    a2m_capsule_sha256: String,
    a2r_qualification_capsule_sha256: String,
    effective_sampler_sha256: String,
    sampled_coordinate_order: Vec<String>,
    optimizer_invocation_budget: u64,
    exact_effective_sampler_required: bool,
    exact_sampled_coordinate_order_required: bool,
    backend_substitution_allowed: bool,
    method_substitution_allowed: bool,
    option_rewrite_allowed: bool,
    network_allowed: bool,
    runtime_package_installation_allowed: bool,
    authorization_scope: &'static str,
    stateful_global_single_use_claimed: bool,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    optimizer_execution_authorized: bool,
    a2_execution_authorized: bool,
    error: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
        return Err(format!("{}: exceeds {MAX_JSON_BYTES} bytes", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn read_json(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let bytes = read_regular_file(path)?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid {label} JSON: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((bytes, value))
}

fn field_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))
}

fn field_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean field: {key}"))
}

fn nested<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing nested field: {key}"))
}

fn nested_str<'a>(value: &'a Value, object: &str, key: &str) -> Result<&'a str, String> {
    field_str(nested(value, object)?, key)
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

fn authorizer_identity() -> Result<(PathBuf, String, String), String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if root.is_empty() || !is_git_oid(&head) || !is_git_oid(&tree) {
        return Err("invalid authorizer Git identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
}

fn validate_spec(spec: &Value) -> Result<(), String> {
    if spec.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(spec, "protocol")? != PROTOCOL
        || field_str(spec, "status")? != "preregistered-authorization-firewall"
        || field_str(spec, "scientific_claim")? != "NONE"
        || field_str(spec, "authority")? != AUTHORITY
    {
        return Err("A2 authorization specification identity drifted".into());
    }

    let a1x = nested(spec, "required_a1x")?;
    if field_str(a1x, "protocol")? != A1X_PROTOCOL
        || field_str(a1x, "capsule_protocol")? != A1X_CAPSULE_PROTOCOL
        || field_str(a1x, "verdict")? != "AGREE"
        || field_str(a1x, "comparison_exit_code")? != "0"
        || field_bool(a1x, "a2_execution_authorized")?
    {
        return Err("A1X authorization prerequisite drifted".into());
    }

    let a2m = nested(spec, "required_a2m")?;
    if field_str(a2m, "protocol")? != A2M_PROTOCOL
        || field_str(a2m, "capsule_protocol")? != A2M_CAPSULE_PROTOCOL
        || field_str(a2m, "verdict")? != "PASS"
        || field_bool(a2m, "optimizer_execution_authorized")?
        || field_bool(a2m, "a2_execution_authorized")?
    {
        return Err("A2M authorization prerequisite drifted".into());
    }

    let a2r = nested(spec, "required_a2r_qualification")?;
    if field_str(a2r, "protocol")? != A2R_CAPSULE_PROTOCOL
        || field_str(a2r, "authority")? != "optimizer-parameter-role-binding-only"
        || field_bool(a2r, "a2_execution_authorized")?
    {
        return Err("A2R authorization prerequisite drifted".into());
    }

    let lineage = nested(spec, "lineage")?;
    for key in [
        "a1x_subject_head_must_equal_a2r_subject_head",
        "a1x_subject_tree_must_equal_a2r_subject_tree",
        "a2m_a2r_subject_head_must_equal_a2r_subject_head",
        "a2m_receipt_must_match_a2m_capsule_hash",
        "a2m_a0_a2r_a2g_hashes_must_match_capsule",
        "a2r_receipt_hash_must_match_a2m_binding",
        "effective_sampler_hash_must_match_a2m_capsule",
    ] {
        if !field_bool(lineage, key)? {
            return Err(format!("lineage requirement {key} is not frozen true"));
        }
    }

    let budget = nested(spec, "execution_budget")?;
    if budget
        .get("optimizer_invocations_per_executor_process")
        .and_then(Value::as_u64)
        != Some(1)
        || !field_bool(budget, "exact_effective_sampler_sha256_required")?
        || !field_bool(budget, "exact_sampled_coordinate_order_required")?
        || field_bool(budget, "backend_substitution_allowed")?
        || field_bool(budget, "method_substitution_allowed")?
        || field_bool(budget, "option_addition_allowed")?
        || field_bool(budget, "option_deletion_allowed")?
        || field_bool(budget, "option_rewrite_allowed")?
        || field_bool(budget, "coordinate_addition_allowed")?
        || field_bool(budget, "coordinate_deletion_allowed")?
        || field_bool(budget, "coordinate_reordering_allowed")?
        || field_bool(budget, "normalization_or_migration_authorized")?
        || field_bool(budget, "network_allowed")?
        || field_bool(budget, "runtime_package_installation_allowed")?
    {
        return Err("A2 execution budget drifted".into());
    }

    let promotion = nested(spec, "promotion")?;
    if !field_bool(promotion, "optimizer_execution_authorized")?
        || !field_bool(promotion, "a2_execution_authorized")?
        || field_str(promotion, "scientific_claim")? != "NONE"
    {
        return Err("A2 authorization promotion boundary drifted".into());
    }
    Ok(())
}

fn sampled_coordinates(value: &Value) -> Result<Vec<String>, String> {
    let array = value
        .get("sampled_coordinate_order")
        .and_then(Value::as_array)
        .ok_or_else(|| "A2M sampled_coordinate_order is not an array".to_owned())?;
    if array.is_empty() {
        return Err("A2M sampled coordinate order is empty".into());
    }
    let mut output = Vec::with_capacity(array.len());
    let mut unique = BTreeSet::new();
    for entry in array {
        let name = entry
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| "A2M sampled coordinate contains an invalid name".to_owned())?;
        if !unique.insert(name.to_owned()) {
            return Err("A2M sampled coordinate order contains duplicates".into());
        }
        output.push(name.to_owned());
    }
    Ok(output)
}

fn validate_a1x(
    receipt_bytes: &[u8],
    receipt: &Value,
    capsule_bytes: &[u8],
    capsule: &Value,
) -> Result<(String, String), String> {
    if field_str(receipt, "protocol")? != A1X_PROTOCOL
        || field_str(receipt, "verdict")? != "AGREE"
        || field_str(receipt, "scientific_claim")? != "NONE"
        || field_str(receipt, "authority")? != "cross-implementation-fixed-point-agreement-only"
        || field_bool(receipt, "a2_execution_authorized")?
    {
        return Err("A1X receipt is not qualified AGREE evidence".into());
    }
    let head = field_str(receipt, "subject_head")?.to_owned();
    let tree = field_str(receipt, "subject_tree")?.to_owned();
    if !is_git_oid(&head) || !is_git_oid(&tree) {
        return Err("A1X subject HEAD/TREE is malformed".into());
    }
    if let Some(reasons) = receipt.get("disagreement_reasons").and_then(Value::as_array) {
        if !reasons.is_empty() {
            return Err("A1X AGREE receipt contains disagreement reasons".into());
        }
    }

    if field_str(capsule, "protocol")? != A1X_CAPSULE_PROTOCOL
        || field_str(capsule, "scientific_claim")? != "NONE"
        || field_str(capsule, "authority")? != "cross-implementation-fixed-point-agreement-only"
        || field_str(capsule, "comparison_verdict")? != "AGREE"
        || field_str(capsule, "subject_head")? != head
        || nested_str(capsule, "workflow", "comparison_exit_code")? != "0"
        || nested_str(capsule, "workflow", "comparison_outcome")? != "success"
        || nested_str(capsule, "workflow", "postflight_outcome")? != "success"
    {
        return Err("A1X comparison capsule is not an AGREE qualification".into());
    }
    let actual = sha256_hex(receipt_bytes);
    if nested_str(capsule, "receipt_sha256", "a1x")? != actual {
        return Err("A1X capsule does not bind the supplied A1X receipt".into());
    }
    if capsule_bytes.is_empty() {
        return Err("A1X capsule is empty".into());
    }
    Ok((head, tree))
}

fn validate_a2m(
    receipt_bytes: &[u8],
    receipt: &Value,
    capsule_bytes: &[u8],
    capsule: &Value,
    subject_head: &str,
) -> Result<(String, Vec<String>, String, String, String), String> {
    if field_str(receipt, "protocol")? != A2M_PROTOCOL
        || field_str(receipt, "verdict")? != "PASS"
        || field_str(receipt, "scientific_claim")? != "NONE"
        || field_str(receipt, "authority")? != "optimizer-method-options-binding-only"
        || field_bool(receipt, "optimizer_execution_authorized")?
        || field_bool(receipt, "a2_execution_authorized")?
    {
        return Err("A2M receipt is not qualified provenance evidence".into());
    }
    let sampler_sha = field_str(receipt, "effective_sampler_sha256")?.to_owned();
    if !is_sha256(&sampler_sha) {
        return Err("A2M effective sampler SHA-256 is malformed".into());
    }
    let coordinates = sampled_coordinates(receipt)?;
    let a0_sha = field_str(receipt, "a0_receipt_sha256")?.to_owned();
    let a2r_sha = field_str(receipt, "a2r_receipt_sha256")?.to_owned();
    let a2g_sha = field_str(receipt, "a2g_receipt_sha256")?.to_owned();
    for value in [&a0_sha, &a2r_sha, &a2g_sha] {
        if !is_sha256(value) {
            return Err("A2M transitive receipt SHA-256 is malformed".into());
        }
    }

    if field_str(capsule, "protocol")? != A2M_CAPSULE_PROTOCOL
        || field_str(capsule, "scientific_claim")? != "NONE"
        || field_str(capsule, "authority")? != "optimizer-method-options-binding-only"
        || field_bool(capsule, "optimizer_execution_authorized")?
        || field_bool(capsule, "a2_execution_authorized")?
        || field_str(capsule, "a2r_subject_head")? != subject_head
        || field_str(capsule, "effective_sampler_sha256")? != sampler_sha
        || nested_str(capsule, "workflow", "software_outcome")? != "success"
        || nested_str(capsule, "workflow", "a1e_outcome")? != "success"
        || nested_str(capsule, "workflow", "a0_artifacts_outcome")? != "success"
        || nested_str(capsule, "workflow", "assemble_outcome")? != "success"
        || nested_str(capsule, "workflow", "a2m_outcome")? != "success"
        || nested_str(capsule, "workflow", "postflight_outcome")? != "success"
    {
        return Err("A2M qualification capsule does not qualify this subject".into());
    }
    if nested_str(capsule, "receipt_sha256", "a2m")? != sha256_hex(receipt_bytes)
        || nested_str(capsule, "receipt_sha256", "a0")? != a0_sha
        || nested_str(capsule, "receipt_sha256", "a2r")? != a2r_sha
        || nested_str(capsule, "receipt_sha256", "a2g")? != a2g_sha
    {
        return Err("A2M qualification capsule receipt hashes do not match".into());
    }
    if capsule_bytes.is_empty() {
        return Err("A2M qualification capsule is empty".into());
    }
    Ok((sampler_sha, coordinates, a0_sha, a2r_sha, a2g_sha))
}

fn validate_a2r_capsule(
    capsule: &Value,
    subject_head: &str,
    subject_tree: &str,
    a2r_sha256: &str,
    a2m_capsule: &Value,
) -> Result<(), String> {
    if field_str(capsule, "protocol")? != A2R_CAPSULE_PROTOCOL
        || field_str(capsule, "scientific_claim")? != "NONE"
        || field_str(capsule, "authority")? != "optimizer-parameter-role-binding-only"
        || field_bool(capsule, "a2_execution_authorized")?
        || field_str(capsule, "subject_head")? != subject_head
        || field_str(capsule, "subject_tree")? != subject_tree
        || nested_str(capsule, "receipt_sha256", "a2r")? != a2r_sha256
        || nested_str(capsule, "workflow", "a2r_outcome")? != "success"
        || nested_str(capsule, "workflow", "postflight_outcome")? != "success"
    {
        return Err("A2R qualification capsule does not bind the A1X subject".into());
    }
    if field_str(a2m_capsule, "a2r_subject_head")? != subject_head {
        return Err("A2M and A2R subject heads differ".into());
    }
    if let (Ok(a2r_run), Ok(bound_run)) = (
        nested_str(capsule, "workflow", "run_id"),
        field_str(a2m_capsule, "a2r_run_id"),
    ) {
        if a2r_run != bound_run {
            return Err("A2M capsule selects a different A2R run".into());
        }
    }
    Ok(())
}

fn execute(
    a1x_receipt_path: &Path,
    a1x_capsule_path: &Path,
    a2m_receipt_path: &Path,
    a2m_capsule_path: &Path,
    a2r_capsule_path: &Path,
) -> Result<Receipt, String> {
    let (root, authorizer_head, authorizer_tree) = authorizer_identity()?;
    let spec_bytes = read_regular_file(&root.join(SPEC_RELATIVE))?;
    let spec: Value = serde_json::from_slice(&spec_bytes)
        .map_err(|error| format!("invalid A2 authorization spec JSON: {error}"))?;
    validate_spec(&spec)?;

    let (a1x_bytes, a1x) = read_json(a1x_receipt_path, "A1X receipt")?;
    let (a1x_capsule_bytes, a1x_capsule) = read_json(a1x_capsule_path, "A1X capsule")?;
    let (subject_head, subject_tree) =
        validate_a1x(&a1x_bytes, &a1x, &a1x_capsule_bytes, &a1x_capsule)?;

    let (a2m_bytes, a2m) = read_json(a2m_receipt_path, "A2M receipt")?;
    let (a2m_capsule_bytes, a2m_capsule) = read_json(a2m_capsule_path, "A2M capsule")?;
    let (sampler_sha, coordinates, _a0_sha, a2r_sha, _a2g_sha) = validate_a2m(
        &a2m_bytes,
        &a2m,
        &a2m_capsule_bytes,
        &a2m_capsule,
        &subject_head,
    )?;

    let (a2r_capsule_bytes, a2r_capsule) =
        read_json(a2r_capsule_path, "A2R qualification capsule")?;
    validate_a2r_capsule(
        &a2r_capsule,
        &subject_head,
        &subject_tree,
        &a2r_sha,
        &a2m_capsule,
    )?;

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        optimizer_execution_authorized: true,
        a2_execution_authorized: true,
        normalization_or_migration_authorized: false,
        authorizer_head,
        authorizer_tree,
        scientific_subject_head: subject_head,
        scientific_subject_tree: subject_tree,
        authorization_spec_sha256: sha256_hex(&spec_bytes),
        a1x_receipt_sha256: sha256_hex(&a1x_bytes),
        a1x_capsule_sha256: sha256_hex(&a1x_capsule_bytes),
        a2m_receipt_sha256: sha256_hex(&a2m_bytes),
        a2m_capsule_sha256: sha256_hex(&a2m_capsule_bytes),
        a2r_qualification_capsule_sha256: sha256_hex(&a2r_capsule_bytes),
        effective_sampler_sha256: sampler_sha,
        sampled_coordinate_order: coordinates,
        optimizer_invocation_budget: 1,
        exact_effective_sampler_required: true,
        exact_sampled_coordinate_order_required: true,
        backend_substitution_allowed: false,
        method_substitution_allowed: false,
        option_rewrite_allowed: false,
        network_allowed: false,
        runtime_package_installation_allowed: false,
        authorization_scope: "one-exact-optimizer-invocation-per-executor-process",
        stateful_global_single_use_claimed: false,
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer(&mut lock, value)
        .map_err(|error| format!("failed to serialize receipt: {error}"))?;
    lock.write_all(b"\n")
        .map_err(|error| format!("failed to write receipt: {error}"))
}

fn main() {
    let args: Vec<_> = env::args_os().collect();
    let result = if args.len() == 6 {
        execute(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
        )
    } else {
        Err("usage: de001a-a2-authorize <a1x.json> <a1x-capsule.json> <a2m.json> <a2m-capsule.json> <a2r-qualification-capsule.json>".into())
    };

    match result {
        Ok(receipt) => {
            if let Err(error) = write_json(&receipt) {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
        Err(error) => {
            let invalid = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                optimizer_execution_authorized: false,
                a2_execution_authorized: false,
                error,
            };
            if let Err(write_error) = write_json(&invalid) {
                eprintln!("{write_error}");
            }
            std::process::exit(2);
        }
    }
}
