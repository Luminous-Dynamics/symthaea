// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A2Q optimizer reproduction result qualification.
//!
//! This program performs no optimization. It validates one raw A2E execution
//! against the frozen A2 authorization/contract lineage and then applies only
//! the reproduction criteria that were frozen in the DE-001A known-answer
//! manifest before any A2 optimizer output existed.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RESULT_FILE_BYTES: u64 = 64 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A2Q-RESULT-QUALIFICATION-v1";
const AUTHORITY: &str = "optimizer-reproduction-qualification-only";
const RAW_PROTOCOL: &str = "DE-001A2E-RAW-OPTIMIZER-EXECUTION-v1";
const RAW_AUTHORITY: &str = "exact-optimizer-raw-execution-only";
const A2E_PROTOCOL: &str = "DE-001A2E-EXECUTION-CONTRACT-v1";
const A2E_AUTHORITY: &str = "raw-optimizer-execution-contract-only";
const AUTH_PROTOCOL: &str = "DE-001A2-EXECUTION-AUTHORIZATION-v1";
const AUTH_AUTHORITY: &str = "exact-optimizer-execution-authorization-only";
const TARGET_ID: &str = "DE-001A-DESI-DR2-BAO-FLAT-LCDM-v1";
const REFERENCE_BESTFIT_SHA256: &str =
    "bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358";
const SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a2q_result_qualification_v1.json";
const TARGET_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_desi_dr2_bao_flat_lcdm_v1.json";

#[derive(Debug, Clone)]
struct FrozenCriterion {
    statistic: &'static str,
    reference_value: f64,
    max_absolute_delta: f64,
}

const CRITERIA: [FrozenCriterion; 3] = [
    FrozenCriterion {
        statistic: "chi2__BAO",
        reference_value: 10.282299,
        max_absolute_delta: 0.01,
    },
    FrozenCriterion {
        statistic: "omegam",
        reference_value: 0.29717936,
        max_absolute_delta: 0.001,
    },
    FrozenCriterion {
        statistic: "hrdrag",
        reference_value: 101.54786,
        max_absolute_delta: 0.1,
    },
];

#[derive(Debug, Serialize)]
struct CriterionReceipt {
    statistic: &'static str,
    reference_value: f64,
    observed_value: f64,
    max_absolute_delta: f64,
    absolute_delta: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    reproduction_verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    qualifier_head: String,
    qualifier_tree: String,
    scientific_subject_head: String,
    scientific_subject_tree: String,
    qualification_spec_sha256: String,
    target_manifest_sha256: String,
    reference_bestfit_sha256: String,
    raw_execution_sha256: String,
    execution_contract_receipt_sha256: String,
    authorization_receipt_sha256: String,
    effective_sampler_sha256: String,
    sampled_coordinate_order: Vec<String>,
    execution_state: String,
    optimizer_invocation_count: u64,
    process_exit_code: i64,
    result_file_manifest_sha256: String,
    criteria: Vec<CriterionReceipt>,
    a3_execution_authorized: bool,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    reproduction_verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a3_execution_authorized: bool,
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

fn read_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!("{}: exceeds byte limit {max_bytes}", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn read_json(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let bytes = read_regular_file(path, MAX_JSON_BYTES)?;
    let value = serde_json::from_slice::<Value>(&bytes)
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

fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or non-u64 field: {key}"))
}

fn field_i64(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing or non-i64 field: {key}"))
}

fn field_f64(value: &Value, key: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or non-number field: {key}"))?;
    if !number.is_finite() {
        return Err(format!("field {key} must be finite"));
    }
    Ok(number)
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

fn repository_identity() -> Result<(PathBuf, String, String), String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if root.is_empty() || !is_git_oid(&head) || !is_git_oid(&tree) {
        return Err("invalid A2Q Git identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
}

fn ordered_strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    let array = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array field: {key}"))?;
    if array.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    let mut result = Vec::with_capacity(array.len());
    let mut seen = BTreeSet::new();
    for item in array {
        let text = item
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{key} contains a non-string or empty item"))?;
        if !seen.insert(text) {
            return Err(format!("{key} contains duplicate item {text:?}"));
        }
        result.push(text.to_owned());
    }
    Ok(result)
}

fn validate_spec(spec: &Value) -> Result<(), String> {
    if spec.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(spec, "protocol")? != PROTOCOL
        || field_str(spec, "status")? != "preregistered-before-optimizer-output"
        || field_str(spec, "scientific_claim")? != "NONE"
        || field_str(spec, "authority")? != AUTHORITY
    {
        return Err("A2Q specification identity mismatch".into());
    }
    let criteria = spec
        .get("frozen_reproduction_criteria")
        .and_then(Value::as_array)
        .ok_or_else(|| "A2Q specification missing criteria".to_owned())?;
    if criteria.len() != CRITERIA.len() {
        return Err("A2Q specification criterion count changed".into());
    }
    for expected in &CRITERIA {
        let matches: Vec<_> = criteria
            .iter()
            .filter(|criterion| criterion.get("statistic").and_then(Value::as_str) == Some(expected.statistic))
            .collect();
        if matches.len() != 1 {
            return Err(format!("A2Q specification must contain one {} criterion", expected.statistic));
        }
        let criterion = matches[0];
        if field_f64(criterion, "reference_value")?.to_bits() != expected.reference_value.to_bits()
            || field_f64(criterion, "max_absolute_delta")?.to_bits()
                != expected.max_absolute_delta.to_bits()
        {
            return Err(format!("A2Q specification drifted for {}", expected.statistic));
        }
    }
    Ok(())
}

fn validate_target_manifest(target: &Value) -> Result<(), String> {
    if target.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(target, "target_id")? != TARGET_ID
        || field_str(target, "status")? != "reference-frozen-not-executable"
        || field_str(target, "claim_policy")? != "reproduction-only"
        || field_str(target, "model")? != "flat-lambda-cdm"
        || field_str(target, "dataset")? != "desi-dr2-bao-all"
    {
        return Err("known-answer target identity mismatch".into());
    }

    let artifacts = target
        .get("official_reference_artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| "target missing official_reference_artifacts".to_owned())?;
    let bestfit: Vec<_> = artifacts
        .iter()
        .filter(|artifact| artifact.get("role").and_then(Value::as_str) == Some("reference-bestfit-text"))
        .collect();
    if bestfit.len() != 1
        || field_str(bestfit[0], "sha256")? != REFERENCE_BESTFIT_SHA256
    {
        return Err("target reference-bestfit identity mismatch".into());
    }

    let criteria = target
        .get("criteria")
        .and_then(Value::as_array)
        .ok_or_else(|| "target missing criteria".to_owned())?;
    for expected in &CRITERIA {
        let matches: Vec<_> = criteria
            .iter()
            .filter(|criterion| {
                criterion.get("subgate").and_then(Value::as_str) == Some("DE-001A2")
                    && criterion.get("statistic").and_then(Value::as_str) == Some(expected.statistic)
            })
            .collect();
        if matches.len() != 1 {
            return Err(format!("target must contain one DE-001A2 {} criterion", expected.statistic));
        }
        let criterion = matches[0];
        if field_f64(criterion, "reference_value")?.to_bits() != expected.reference_value.to_bits()
            || field_f64(criterion, "max_absolute_delta")?.to_bits()
                != expected.max_absolute_delta.to_bits()
        {
            return Err(format!("target DE-001A2 criterion drifted for {}", expected.statistic));
        }
    }
    Ok(())
}

fn validate_authorization(bytes: &[u8], value: &Value) -> Result<(String, String, String, Vec<String>), String> {
    if field_str(value, "protocol")? != AUTH_PROTOCOL
        || field_str(value, "verdict")? != "PASS"
        || field_str(value, "scientific_claim")? != "NONE"
        || field_str(value, "authority")? != AUTH_AUTHORITY
        || !field_bool(value, "optimizer_execution_authorized")?
        || !field_bool(value, "a2_execution_authorized")?
        || field_bool(value, "normalization_or_migration_authorized")?
        || field_u64(value, "optimizer_invocation_budget")? != 1
    {
        return Err("A2 authorization is not an exact-execution PASS".into());
    }
    let head = field_str(value, "scientific_subject_head")?.to_owned();
    let tree = field_str(value, "scientific_subject_tree")?.to_owned();
    let sampler = field_str(value, "effective_sampler_sha256")?.to_owned();
    let coordinates = ordered_strings(value, "sampled_coordinate_order")?;
    if !is_git_oid(&head) || !is_git_oid(&tree) || !is_sha256(&sampler) {
        return Err("A2 authorization contains invalid identities".into());
    }
    if sha256_hex(bytes).is_empty() {
        return Err("unreachable authorization hash failure".into());
    }
    Ok((head, tree, sampler, coordinates))
}

fn validate_a2e_contract(
    bytes: &[u8],
    value: &Value,
    authorization_sha256: &str,
    subject_head: &str,
    subject_tree: &str,
    sampler: &str,
    coordinates: &[String],
) -> Result<(), String> {
    if field_str(value, "protocol")? != A2E_PROTOCOL
        || field_str(value, "verdict")? != "PASS"
        || field_str(value, "scientific_claim")? != "NONE"
        || field_str(value, "authority")? != A2E_AUTHORITY
        || !field_bool(value, "optimizer_execution_contract_valid")?
        || field_bool(value, "executor_implementation_authorized")?
        || field_str(value, "reproduction_verdict")? != "UNASSESSED"
        || field_bool(value, "a2q_execution_authorized")?
        || field_u64(value, "optimizer_invocation_budget")? != 1
        || field_str(value, "required_raw_result_protocol")? != RAW_PROTOCOL
        || field_bool(value, "automatic_retry_allowed")?
        || field_bool(value, "normalization_or_migration_allowed")?
        || field_bool(value, "network_allowed")?
        || field_bool(value, "runtime_package_installation_allowed")?
    {
        return Err("A2E contract receipt is not a frozen PASS".into());
    }
    if field_str(value, "authorization_receipt_sha256")? != authorization_sha256
        || field_str(value, "scientific_subject_head")? != subject_head
        || field_str(value, "scientific_subject_tree")? != subject_tree
        || field_str(value, "effective_sampler_sha256")? != sampler
        || ordered_strings(value, "sampled_coordinate_order")? != coordinates
    {
        return Err("A2E contract receipt does not bind the supplied authorization".into());
    }
    if sha256_hex(bytes).is_empty() {
        return Err("unreachable A2E hash failure".into());
    }
    Ok(())
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err("result manifest path must be non-empty and relative".into());
    }
    for component in path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            _ => return Err(format!("unsafe result manifest path: {value}")),
        }
    }
    Ok(path.to_owned())
}

fn validate_result_manifest(
    value: &Value,
    result_root: &Path,
    require_bestfit: bool,
) -> Result<(String, Option<Vec<u8>>), String> {
    let entries = value
        .get("result_file_manifest")
        .and_then(Value::as_array)
        .ok_or_else(|| "raw result missing result_file_manifest".to_owned())?;
    let manifest_bytes = serde_json::to_vec(entries)
        .map_err(|error| format!("failed to canonicalize result manifest: {error}"))?;
    let mut roles = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut bestfit = None;

    for entry in entries {
        let role = field_str(entry, "role")?;
        let relative = field_str(entry, "path")?;
        let expected_sha = field_str(entry, "sha256")?;
        let expected_size = field_u64(entry, "size")?;
        if !is_sha256(expected_sha) || !roles.insert(role) || !paths.insert(relative) {
            return Err("result manifest contains invalid or duplicate role/path/hash".into());
        }
        let relative_path = safe_relative_path(relative)?;
        let full_path = result_root.join(relative_path);
        let bytes = read_regular_file(&full_path, MAX_RESULT_FILE_BYTES)?;
        if bytes.len() as u64 != expected_size || sha256_hex(&bytes) != expected_sha {
            return Err(format!("result file identity mismatch for role {role}"));
        }
        if role == "optimizer-bestfit-text" {
            bestfit = Some(bytes);
        }
    }

    if require_bestfit && bestfit.is_none() {
        return Err("EXECUTED raw result requires optimizer-bestfit-text".into());
    }
    if !require_bestfit && bestfit.is_some() {
        return Err("EXECUTION_ERROR must not present optimizer-bestfit-text as completed output".into());
    }
    Ok((sha256_hex(&manifest_bytes), bestfit))
}

fn parse_bestfit_table(bytes: &[u8]) -> Result<BTreeMap<String, f64>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("best-fit text is not UTF-8: {error}"))?;
    let lines: Vec<_> = text.lines().collect();
    let required: BTreeSet<_> = CRITERIA.iter().map(|criterion| criterion.statistic).collect();
    let mut header_index = None;
    let mut headers = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.trim_start_matches('#').split_whitespace().collect();
        let field_set: BTreeSet<_> = fields.iter().copied().collect();
        if required.is_subset(&field_set) {
            if field_set.len() != fields.len() {
                return Err("best-fit header contains duplicate field names".into());
            }
            header_index = Some(index);
            headers = fields;
            break;
        }
    }

    let header_index = header_index.ok_or_else(|| {
        "best-fit text has no header containing chi2__BAO, omegam, and hrdrag".to_owned()
    })?;
    let row = lines
        .iter()
        .skip(header_index + 1)
        .map(|line| line.trim())
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or_else(|| "best-fit table has no data row after header".to_owned())?;
    let values: Vec<_> = row.split_whitespace().collect();
    if values.len() != headers.len() {
        return Err("best-fit row/header width mismatch".into());
    }
    let mut parsed = BTreeMap::new();
    for (field, token) in headers.into_iter().zip(values) {
        let number: f64 = token
            .parse()
            .map_err(|_| format!("best-fit field {field:?} is not numeric"))?;
        if !number.is_finite() || parsed.insert(field.to_owned(), number).is_some() {
            return Err(format!("invalid or duplicate best-fit field {field:?}"));
        }
    }
    for criterion in &CRITERIA {
        if !parsed.contains_key(criterion.statistic) {
            return Err(format!("best-fit table missing {}", criterion.statistic));
        }
    }
    Ok(parsed)
}

fn validate_raw_common(
    raw: &Value,
    raw_bytes: &[u8],
    a2e_sha256: &str,
    authorization_sha256: &str,
    subject_head: &str,
    subject_tree: &str,
    sampler: &str,
    coordinates: &[String],
) -> Result<(String, u64, i64), String> {
    if field_str(raw, "protocol")? != RAW_PROTOCOL
        || field_str(raw, "scientific_claim")? != "NONE"
        || field_str(raw, "authority")? != RAW_AUTHORITY
        || field_str(raw, "reproduction_verdict")? != "UNASSESSED"
        || field_str(raw, "authorization_receipt_sha256")? != authorization_sha256
        || field_str(raw, "execution_contract_receipt_sha256")? != a2e_sha256
        || field_str(raw, "scientific_subject_head")? != subject_head
        || field_str(raw, "scientific_subject_tree")? != subject_tree
        || field_str(raw, "effective_sampler_sha256")? != sampler
        || ordered_strings(raw, "sampled_coordinate_order")? != coordinates
    {
        return Err("raw execution does not bind the qualified A2 lineage".into());
    }
    for key in ["command_argv_sha256", "stdout_sha256", "stderr_sha256"] {
        if !is_sha256(field_str(raw, key)?) {
            return Err(format!("raw execution field {key} is not SHA-256"));
        }
    }
    let postflight = raw
        .get("postflight")
        .ok_or_else(|| "raw execution missing postflight".to_owned())?;
    if !field_bool(postflight, "immutable")?
        || !is_git_oid(field_str(postflight, "executor_head")?)
        || !is_git_oid(field_str(postflight, "executor_tree")?)
    {
        return Err("raw execution postflight identity is invalid".into());
    }
    let state = field_str(raw, "execution_state")?.to_owned();
    let invocations = field_u64(raw, "optimizer_invocation_count")?;
    let exit_code = field_i64(raw, "process_exit_code")?;
    if sha256_hex(raw_bytes).is_empty() {
        return Err("unreachable raw execution hash failure".into());
    }
    Ok((state, invocations, exit_code))
}

fn execute(
    raw_path: &Path,
    a2e_path: &Path,
    authorization_path: &Path,
    reference_bestfit_path: &Path,
    result_root: &Path,
) -> Result<Receipt, String> {
    let (root, qualifier_head, qualifier_tree) = repository_identity()?;
    let spec_bytes = read_regular_file(&root.join(SPEC_RELATIVE), MAX_JSON_BYTES)?;
    let spec: Value = serde_json::from_slice(&spec_bytes)
        .map_err(|error| format!("invalid A2Q specification JSON: {error}"))?;
    validate_spec(&spec)?;

    let target_bytes = read_regular_file(&root.join(TARGET_RELATIVE), MAX_JSON_BYTES)?;
    let target: Value = serde_json::from_slice(&target_bytes)
        .map_err(|error| format!("invalid known-answer target JSON: {error}"))?;
    validate_target_manifest(&target)?;

    let reference_bytes = read_regular_file(reference_bestfit_path, MAX_RESULT_FILE_BYTES)?;
    if sha256_hex(&reference_bytes) != REFERENCE_BESTFIT_SHA256 {
        return Err("reference best-fit bytes do not match frozen target".into());
    }
    let reference_values = parse_bestfit_table(&reference_bytes)?;
    for criterion in &CRITERIA {
        let observed = *reference_values
            .get(criterion.statistic)
            .ok_or_else(|| format!("reference best-fit missing {}", criterion.statistic))?;
        if observed.to_bits() != criterion.reference_value.to_bits() {
            return Err(format!("reference best-fit value disagrees with frozen {} target", criterion.statistic));
        }
    }

    let (authorization_bytes, authorization) = read_json(authorization_path, "A2 authorization")?;
    let authorization_sha = sha256_hex(&authorization_bytes);
    let (subject_head, subject_tree, sampler, coordinates) =
        validate_authorization(&authorization_bytes, &authorization)?;

    let (a2e_bytes, a2e) = read_json(a2e_path, "A2E contract receipt")?;
    let a2e_sha = sha256_hex(&a2e_bytes);
    validate_a2e_contract(
        &a2e_bytes,
        &a2e,
        &authorization_sha,
        &subject_head,
        &subject_tree,
        &sampler,
        &coordinates,
    )?;

    let (raw_bytes, raw) = read_json(raw_path, "A2E raw execution")?;
    let (state, invocations, exit_code) = validate_raw_common(
        &raw,
        &raw_bytes,
        &a2e_sha,
        &authorization_sha,
        &subject_head,
        &subject_tree,
        &sampler,
        &coordinates,
    )?;

    if state == "INVALID" {
        return Err("raw executor classified its own evidence INVALID".into());
    }
    if invocations != 1 {
        return Err("A2Q requires exactly one optimizer invocation".into());
    }

    let (manifest_sha, candidate_bytes) = match state.as_str() {
        "EXECUTED" => {
            if exit_code != 0 {
                return Err("EXECUTED raw result must have process_exit_code=0".into());
            }
            validate_result_manifest(&raw, result_root, true)?
        }
        "EXECUTION_ERROR" => {
            if exit_code == 0 {
                return Err("EXECUTION_ERROR raw result must have nonzero process_exit_code".into());
            }
            validate_result_manifest(&raw, result_root, false)?
        }
        other => return Err(format!("unsupported raw execution state {other:?}")),
    };

    let (reproduction_verdict, criteria) = if state == "EXECUTION_ERROR" {
        ("UNASSESSED", Vec::new())
    } else {
        let candidate = parse_bestfit_table(
            candidate_bytes
                .as_deref()
                .ok_or_else(|| "missing candidate best-fit bytes".to_owned())?,
        )?;
        let mut receipts = Vec::with_capacity(CRITERIA.len());
        let mut all_pass = true;
        for criterion in &CRITERIA {
            let observed = *candidate
                .get(criterion.statistic)
                .ok_or_else(|| format!("candidate best-fit missing {}", criterion.statistic))?;
            let delta = (observed - criterion.reference_value).abs();
            let passed = delta <= criterion.max_absolute_delta;
            all_pass &= passed;
            receipts.push(CriterionReceipt {
                statistic: criterion.statistic,
                reference_value: criterion.reference_value,
                observed_value: observed,
                max_absolute_delta: criterion.max_absolute_delta,
                absolute_delta: delta,
                passed,
            });
        }
        (if all_pass { "PASS" } else { "NEGATIVE" }, receipts)
    };

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        reproduction_verdict,
        scientific_claim: "NONE",
        authority: AUTHORITY,
        qualifier_head,
        qualifier_tree,
        scientific_subject_head: subject_head,
        scientific_subject_tree: subject_tree,
        qualification_spec_sha256: sha256_hex(&spec_bytes),
        target_manifest_sha256: sha256_hex(&target_bytes),
        reference_bestfit_sha256: REFERENCE_BESTFIT_SHA256.to_owned(),
        raw_execution_sha256: sha256_hex(&raw_bytes),
        execution_contract_receipt_sha256: a2e_sha,
        authorization_receipt_sha256: authorization_sha,
        effective_sampler_sha256: sampler,
        sampled_coordinate_order: coordinates,
        execution_state: state,
        optimizer_invocation_count: invocations,
        process_exit_code: exit_code,
        result_file_manifest_sha256: manifest_sha,
        criteria,
        a3_execution_authorized: false,
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer(&mut lock, value)
        .map_err(|error| format!("failed to serialize A2Q receipt: {error}"))?;
    lock.write_all(b"\n")
        .map_err(|error| format!("failed to write A2Q receipt: {error}"))
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
        Err("usage: de001a-a2q-qualify <raw-execution.json> <a2e-contract.json> <a2-authorization.json> <reference-bestfit.txt> <result-root>".into())
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
                reproduction_verdict: "UNASSESSED",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                a3_execution_authorized: false,
                error,
            };
            if let Err(write_error) = write_json(&invalid) {
                eprintln!("{write_error}");
            }
            std::process::exit(2);
        }
    }
}
