//! Fail-closed structural validator for scientific benchmark refresh manifests.
//!
//! This validates the declared manifest contract and cross-field invariants.
//! It does not execute a generator, authenticate a runner, prove that oracle
//! data were kept secret, or validate scientific truth.

use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MANIFEST_SCHEMA: &str = "symthaea.science_bench_refresh_manifest.v1";
pub const REPORT_SCHEMA: &str = "symthaea.science_bench_refresh_validation_report.v1";
pub const TASK_SET_SCHEMA: &str = "symthaea.science_bench_refresh_task_set.v1";
const WEIGHT_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationManifestV1 {
    pub schema: String,
    pub generation_id: String,
    pub purpose: String,
    pub lane: String,
    pub generator: GeneratorV1,
    pub evaluator: EvaluatorV1,
    pub protocol_commitment_sha256: String,
    pub seed_commitment_sha256: String,
    pub task_set_commitment_sha256: String,
    pub expected_task_count: u64,
    pub task_family_mixture: Vec<TaskFamilyV1>,
    pub policy_freeze: PolicyFreezeV1,
    pub budget: BudgetV1,
    pub analysis_plan_sha256: String,
    pub post_run_reveal_policy: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorV1 {
    pub repository: String,
    pub revision: String,
    pub configuration_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorV1 {
    pub version: String,
    pub source_sha256: String,
    pub oracle_separation: String,
}


#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSetArtifactV1 {
    pub schema: String,
    pub generation_id: String,
    pub tasks: Vec<TaskSetRecordV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSetRecordV1 {
    pub task_id: String,
    pub family_id: String,
    pub policy_input: serde_json::Value,
    pub evaluator_only: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskFamilyV1 {
    pub family_id: String,
    pub weight: f64,
    pub planned_task_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFreezeV1 {
    pub subject_id: String,
    pub source_or_policy_sha256: String,
    pub prompt_sha256: String,
    pub configuration_sha256: String,
    pub tool_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetV1 {
    pub max_observations: u64,
    pub max_experiments: u64,
    pub max_tool_calls: u64,
    pub max_total_tokens: u64,
    pub max_wall_clock_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReportV1 {
    pub schema: String,
    pub manifest_sha256: String,
    pub valid: bool,
    pub manifest_declares_scoring_eligible: bool,
    pub manifest_declares_sealed_generalization_profile: bool,
    pub expected_task_count: u64,
    pub planned_task_count: u64,
    pub task_family_weight_sum: f64,
    pub issues: Vec<String>,
    pub admission_reason: String,
}

#[derive(Debug, Serialize)]
struct ProtocolCommitContextV1<'a> {
    schema: &'a str,
    generation_id: &'a str,
    purpose: &'a str,
    lane: &'a str,
    generator: &'a GeneratorV1,
    evaluator: &'a EvaluatorV1,
    expected_task_count: u64,
    task_family_mixture: &'a [TaskFamilyV1],
    policy_freeze: &'a PolicyFreezeV1,
    budget: &'a BudgetV1,
    analysis_plan_sha256: &'a str,
    post_run_reveal_policy: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct RevealVerificationReportV1 {
    pub schema: String,
    pub manifest_sha256: String,
    pub generation_id: String,
    pub protocol_commitment_expected: String,
    pub protocol_commitment_recomputed: String,
    pub seed_commitment_expected: String,
    pub seed_commitment_recomputed: String,
    pub seed_commitment_matches: bool,
    pub task_set_commitment_expected: String,
    pub task_set_commitment_recomputed: String,
    pub task_set_commitment_matches: bool,
    pub task_set_contract_valid: bool,
    pub seed_bytes: u64,
    pub task_set_bytes: u64,
    pub verified: bool,
    pub issues: Vec<String>,
}

pub const REVEAL_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_reveal_verification.v1";
const COMMITMENT_PREFIX: &[u8] = b"symthaea.science_bench_refresh.commitment.v1\0";
const MIN_SEED_BYTES: usize = 32;

fn parse_manifest_for_commitment(bytes: &[u8]) -> Result<GenerationManifestV1, String> {
    serde_json::from_slice(bytes)
        .map_err(|error| format!("manifest JSON does not match the strict v1 shape: {error}"))
}

fn protocol_context(manifest: &GenerationManifestV1) -> ProtocolCommitContextV1<'_> {
    ProtocolCommitContextV1 {
        schema: &manifest.schema,
        generation_id: &manifest.generation_id,
        purpose: &manifest.purpose,
        lane: &manifest.lane,
        generator: &manifest.generator,
        evaluator: &manifest.evaluator,
        expected_task_count: manifest.expected_task_count,
        task_family_mixture: &manifest.task_family_mixture,
        policy_freeze: &manifest.policy_freeze,
        budget: &manifest.budget,
        analysis_plan_sha256: &manifest.analysis_plan_sha256,
        post_run_reveal_policy: &manifest.post_run_reveal_policy,
    }
}

fn protocol_commitment_for_manifest(
    manifest: &GenerationManifestV1,
) -> Result<String, String> {
    let context = serde_json::to_vec(&protocol_context(manifest))
        .map_err(|error| format!("cannot serialize protocol commitment context: {error}"))?;
    let context_len = u64::try_from(context.len())
        .map_err(|_| "protocol context exceeds u64 byte length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(COMMITMENT_PREFIX);
    hasher.update(b"protocol.v1\0");
    hasher.update(context_len.to_be_bytes());
    hasher.update(&context);
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn artifact_commitment_for(
    kind: &str,
    protocol_commitment: &str,
    artifact_bytes: &[u8],
) -> Result<String, String> {
    let kind_domain: &[u8] = match kind {
        "seed" => b"seed.v1\0",
        "task-set" => b"task-set.v1\0",
        other => return Err(format!("unsupported commitment kind: {other}")),
    };
    if !is_lower_hex(protocol_commitment, 64) {
        return Err("protocol commitment must be 64 lowercase hexadecimal characters".to_owned());
    }
    let context = protocol_commitment.as_bytes();
    let context_len = u64::try_from(context.len())
        .map_err(|_| "protocol commitment exceeds u64 byte length".to_owned())?;
    let payload_len = u64::try_from(artifact_bytes.len())
        .map_err(|_| "artifact exceeds u64 byte length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(COMMITMENT_PREFIX);
    hasher.update(kind_domain);
    hasher.update(context_len.to_be_bytes());
    hasher.update(context);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(artifact_bytes);
    Ok(hex_digest(&hasher.finalize()))
}

/// Compute the commitment to all frozen non-artifact manifest fields.
/// Commitment fields themselves are excluded to avoid circularity.
pub fn compute_protocol_commitment(manifest_bytes: &[u8]) -> Result<String, String> {
    let manifest = parse_manifest_for_commitment(manifest_bytes)?;
    protocol_commitment_for_manifest(&manifest)
}

/// Compute an artifact commitment bound to the protocol context encoded in a
/// manifest. This deliberately ignores the manifest's declared digest fields
/// and recomputes the context from the rest of the manifest.
pub fn compute_artifact_commitment(
    manifest_bytes: &[u8],
    kind: &str,
    artifact_bytes: &[u8],
) -> Result<String, String> {
    if kind == "seed" && artifact_bytes.len() < MIN_SEED_BYTES {
        return Err(format!(
            "seed artifact must contain at least {MIN_SEED_BYTES} bytes of entropy"
        ));
    }
    if kind == "task-set" && artifact_bytes.is_empty() {
        return Err("task-set artifact must not be empty".to_owned());
    }
    let protocol_commitment = compute_protocol_commitment(manifest_bytes)?;
    artifact_commitment_for(kind, &protocol_commitment, artifact_bytes)
}


fn scan_policy_input_for_oracle_fields(
    value: &serde_json::Value,
    path: &str,
    issues: &mut Vec<String>,
) {
    const FORBIDDEN_KEYS: &[&str] = &[
        "answer_key",
        "evaluator_only",
        "ground_truth",
        "hidden_law",
        "hidden_target",
        "oracle_answer",
        "true_equation",
        "target_equation",
    ];
    match value {
        serde_json::Value::Object(fields) => {
            for (key, nested) in fields {
                let normalized = key.to_ascii_lowercase();
                if FORBIDDEN_KEYS.contains(&normalized.as_str()) {
                    issues.push(format!(
                        "policy_input contains forbidden oracle field at {path}.{key}"
                    ));
                }
                scan_policy_input_for_oracle_fields(nested, &format!("{path}.{key}"), issues);
            }
        }
        serde_json::Value::Array(items) => {
            for (index, nested) in items.iter().enumerate() {
                scan_policy_input_for_oracle_fields(nested, &format!("{path}[{index}]"), issues);
            }
        }
        _ => {}
    }
}

fn validate_task_set_contract(
    manifest: &GenerationManifestV1,
    task_set_bytes: &[u8],
) -> Vec<String> {
    let task_set: TaskSetArtifactV1 = match serde_json::from_slice(task_set_bytes) {
        Ok(value) => value,
        Err(error) => {
            return vec![format!("task-set artifact does not match the strict v1 shape: {error}")];
        }
    };
    let mut issues = Vec::new();
    if task_set.schema != TASK_SET_SCHEMA {
        issues.push(format!("task-set schema must be {TASK_SET_SCHEMA}"));
    }
    if task_set.generation_id != manifest.generation_id {
        issues.push("task-set generation_id does not match the frozen manifest".to_owned());
    }
    if u64::try_from(task_set.tasks.len()).ok() != Some(manifest.expected_task_count) {
        issues.push(format!(
            "task-set record count does not match expected_task_count={}",
            manifest.expected_task_count
        ));
    }

    let mut expected_by_family = std::collections::BTreeMap::<String, u64>::new();
    for family in &manifest.task_family_mixture {
        expected_by_family.insert(family.family_id.clone(), family.planned_task_count);
    }
    let mut observed_by_family = std::collections::BTreeMap::<String, u64>::new();
    let mut task_ids = BTreeSet::new();

    for (index, task) in task_set.tasks.iter().enumerate() {
        let prefix = format!("tasks[{index}]");
        if task.task_id.trim().is_empty() {
            issues.push(format!("{prefix}.task_id must not be empty"));
        } else if !task_ids.insert(task.task_id.as_str()) {
            issues.push(format!("duplicate task_id: {}", task.task_id));
        }
        if !expected_by_family.contains_key(&task.family_id) {
            issues.push(format!("{prefix} has undeclared family_id {}", task.family_id));
        }
        let observed = observed_by_family.entry(task.family_id.clone()).or_default();
        match observed.checked_add(1) {
            Some(total) => *observed = total,
            None => issues.push(format!("task count overflows for family {}", task.family_id)),
        }
        if !task.policy_input.is_object() {
            issues.push(format!("{prefix}.policy_input must be a JSON object"));
        } else {
            scan_policy_input_for_oracle_fields(
                &task.policy_input,
                &format!("{prefix}.policy_input"),
                &mut issues,
            );
        }
        if !task.evaluator_only.is_object() {
            issues.push(format!("{prefix}.evaluator_only must be a JSON object"));
        }
    }

    for (family_id, expected_count) in expected_by_family {
        let observed_count = observed_by_family.get(&family_id).copied().unwrap_or(0);
        if observed_count != expected_count {
            issues.push(format!(
                "task-set family count mismatch for {family_id}: expected {expected_count}, observed {observed_count}"
            ));
        }
    }
    issues
}

/// Verify a post-run reveal against the exact precommitted protocol, seed and
/// task-set bytes. This establishes byte-level commitment consistency only;
/// it cannot prove the runner enforced secrecy or executed the declared policy.
pub fn verify_reveal(
    manifest_bytes: &[u8],
    seed_bytes: &[u8],
    task_set_bytes: &[u8],
) -> Result<RevealVerificationReportV1, String> {
    let manifest: GenerationManifestV1 = parse_manifest_for_commitment(manifest_bytes)?;
    let validation = validate_manifest_bytes(manifest_bytes)?;
    if !validation.valid {
        return Err(format!(
            "manifest failed structural validation: {}",
            validation.issues.join("; ")
        ));
    }
    if manifest.purpose != "qualified_run" || manifest.lane != "sealed_refresh" {
        return Err(
            "reveal verification requires purpose=qualified_run and lane=sealed_refresh"
                .to_owned(),
        );
    }
    if seed_bytes.len() < MIN_SEED_BYTES {
        return Err(format!(
            "revealed seed must contain at least {MIN_SEED_BYTES} bytes"
        ));
    }
    if task_set_bytes.is_empty() {
        return Err("revealed task-set artifact must not be empty".to_owned());
    }

    let task_set_issues = validate_task_set_contract(&manifest, task_set_bytes);
    let task_set_contract_valid = task_set_issues.is_empty();
    let protocol_computed = protocol_commitment_for_manifest(&manifest)?;
    let seed_computed = artifact_commitment_for("seed", &protocol_computed, seed_bytes)?;
    let task_set_computed =
        artifact_commitment_for("task-set", &protocol_computed, task_set_bytes)?;
    let seed_matches = seed_computed == manifest.seed_commitment_sha256;
    let task_set_matches = task_set_computed == manifest.task_set_commitment_sha256;
    let mut issues = task_set_issues;
    if !seed_matches {
        issues.push("revealed seed does not match the manifest commitment".to_owned());
    }
    if !task_set_matches {
        issues.push("revealed task-set bytes do not match the manifest commitment".to_owned());
    }
    let verified = seed_matches && task_set_matches && task_set_contract_valid;

    Ok(RevealVerificationReportV1 {
        schema: REVEAL_REPORT_SCHEMA.to_owned(),
        manifest_sha256: sha256_hex(manifest_bytes),
        generation_id: manifest.generation_id,
        protocol_commitment_expected: manifest.protocol_commitment_sha256,
        protocol_commitment_recomputed: protocol_computed,
        seed_commitment_expected: manifest.seed_commitment_sha256,
        seed_commitment_recomputed: seed_computed,
        seed_commitment_matches: seed_matches,
        task_set_commitment_expected: manifest.task_set_commitment_sha256,
        task_set_commitment_recomputed: task_set_computed,
        task_set_commitment_matches: task_set_matches,
        task_set_contract_valid,
        seed_bytes: u64::try_from(seed_bytes.len())
            .map_err(|_| "seed artifact exceeds u64 byte length".to_owned())?,
        task_set_bytes: u64::try_from(task_set_bytes.len())
            .map_err(|_| "task-set artifact exceeds u64 byte length".to_owned())?,
        verified,
        issues,
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_digest(&Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_zero_digest(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte == b'0')
}

fn check_sha256(
    field: &str,
    value: &str,
    require_nonzero: bool,
    issues: &mut Vec<String>,
) {
    if !is_lower_hex(value, 64) {
        issues.push(format!("{field} must be 64 lowercase hexadecimal characters"));
    } else if require_nonzero && is_zero_digest(value) {
        issues.push(format!("{field} is an all-zero placeholder in a qualified run"));
    }
}

fn check_commit_sha(
    field: &str,
    value: &str,
    require_nonzero: bool,
    issues: &mut Vec<String>,
) {
    if !is_lower_hex(value, 40) {
        issues.push(format!("{field} must be a 40-character lowercase Git commit SHA"));
    } else if require_nonzero && value.bytes().all(|byte| byte == b'0') {
        issues.push(format!("{field} is an all-zero placeholder in a qualified run"));
    }
}

pub fn validate_manifest_bytes(bytes: &[u8]) -> Result<ValidationReportV1, String> {
    let manifest: GenerationManifestV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("manifest JSON does not match the strict v1 shape: {error}"))?;
    let mut issues = Vec::new();
    let is_qualified = manifest.purpose == "qualified_run";

    if manifest.schema != MANIFEST_SCHEMA {
        issues.push(format!(
            "schema must be {MANIFEST_SCHEMA}, got {}",
            manifest.schema
        ));
    }
    if manifest.generation_id.trim().is_empty() {
        issues.push("generation_id must not be empty".to_owned());
    }
    if !matches!(
        manifest.purpose.as_str(),
        "protocol_fixture_not_for_scoring" | "qualified_run"
    ) {
        issues.push("purpose must be protocol_fixture_not_for_scoring or qualified_run".to_owned());
    }
    if !matches!(
        manifest.lane.as_str(),
        "public_development" | "immutable_anchor" | "sealed_refresh" | "reproducible_open"
    ) {
        issues.push("lane is not a supported evaluation lane".to_owned());
    }
    if manifest.generator.repository.trim().is_empty() {
        issues.push("generator.repository must not be empty".to_owned());
    }
    if manifest.generator.revision.trim().is_empty() {
        issues.push("generator.revision must not be empty".to_owned());
    }
    if manifest.evaluator.version.trim().is_empty() {
        issues.push("evaluator.version must not be empty".to_owned());
    }
    if manifest.policy_freeze.subject_id.trim().is_empty() {
        issues.push("policy_freeze.subject_id must not be empty".to_owned());
    }

    let post_run_reveal_ok = match manifest.lane.as_str() {
        "sealed_refresh" => {
            manifest.post_run_reveal_policy == "reveal_seed_and_task_bytes_after_submission"
        }
        "public_development" | "immutable_anchor" | "reproducible_open" => {
            manifest.post_run_reveal_policy == "open_from_start_not_sealed"
        }
        _ => false,
    };
    if !post_run_reveal_ok {
        issues.push(
            "post_run_reveal_policy is incompatible with the declared lane".to_owned(),
        );
    }

    if !matches!(
        manifest.evaluator.oracle_separation.as_str(),
        "not_qualified_example" | "independent_evaluator_boundary_verified"
    ) {
        issues.push("evaluator.oracle_separation is not a supported value".to_owned());
    }
    if is_qualified
        && manifest.evaluator.oracle_separation != "independent_evaluator_boundary_verified"
    {
        issues.push(
            "qualified_run requires an independently verified evaluator/oracle boundary"
                .to_owned(),
        );
    }

    check_sha256(
        "generator.configuration_sha256",
        &manifest.generator.configuration_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "evaluator.source_sha256",
        &manifest.evaluator.source_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "protocol_commitment_sha256",
        &manifest.protocol_commitment_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "seed_commitment_sha256",
        &manifest.seed_commitment_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "task_set_commitment_sha256",
        &manifest.task_set_commitment_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "policy_freeze.source_or_policy_sha256",
        &manifest.policy_freeze.source_or_policy_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "policy_freeze.prompt_sha256",
        &manifest.policy_freeze.prompt_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "policy_freeze.configuration_sha256",
        &manifest.policy_freeze.configuration_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "policy_freeze.tool_manifest_sha256",
        &manifest.policy_freeze.tool_manifest_sha256,
        is_qualified,
        &mut issues,
    );
    check_sha256(
        "analysis_plan_sha256",
        &manifest.analysis_plan_sha256,
        is_qualified,
        &mut issues,
    );
    if is_qualified {
        check_commit_sha(
            "generator.revision",
            &manifest.generator.revision,
            true,
            &mut issues,
        );
    }
    if is_qualified && manifest.lane == "public_development" {
        issues.push("public_development cannot be marked as a qualified scored run".to_owned());
    }

    if manifest.expected_task_count == 0 {
        issues.push("expected_task_count must be greater than zero".to_owned());
    }
    if manifest.task_family_mixture.is_empty() {
        issues.push("task_family_mixture must not be empty".to_owned());
    }

    let mut family_ids = BTreeSet::new();
    let mut planned_task_count = 0_u64;
    let mut task_count_overflow = false;
    let mut weight_sum = 0.0_f64;
    for (index, family) in manifest.task_family_mixture.iter().enumerate() {
        let prefix = format!("task_family_mixture[{index}]");
        if family.family_id.trim().is_empty() {
            issues.push(format!("{prefix}.family_id must not be empty"));
        }
        if !family_ids.insert(family.family_id.as_str()) {
            issues.push(format!("duplicate task-family ID: {}", family.family_id));
        }
        if !family.weight.is_finite() || family.weight <= 0.0 || family.weight > 1.0 {
            issues.push(format!("{prefix}.weight must be finite and in (0, 1]"));
        } else {
            weight_sum += family.weight;
        }
        if family.planned_task_count == 0 {
            issues.push(format!("{prefix}.planned_task_count must be greater than zero"));
        }
        match planned_task_count.checked_add(family.planned_task_count) {
            Some(total) => planned_task_count = total,
            None => {
                task_count_overflow = true;
                issues.push("sum of planned task counts overflows u64".to_owned());
            }
        }
    }
    if !task_count_overflow && planned_task_count != manifest.expected_task_count {
        issues.push(format!(
            "task denominator mismatch: expected_task_count={}, planned_task_count={planned_task_count}",
            manifest.expected_task_count
        ));
    }
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > WEIGHT_TOLERANCE {
        issues.push(format!(
            "task-family weights must sum to 1 within {WEIGHT_TOLERANCE}; got {weight_sum}"
        ));
    }

    if manifest.budget.max_wall_clock_seconds == 0 {
        issues.push("budget.max_wall_clock_seconds must be greater than zero".to_owned());
    }

    if is_qualified || !is_zero_digest(&manifest.protocol_commitment_sha256) {
        match protocol_commitment_for_manifest(&manifest) {
            Ok(computed) if computed != manifest.protocol_commitment_sha256 => {
                issues.push(
                    "protocol_commitment_sha256 does not match the frozen manifest context"
                        .to_owned(),
                );
            }
            Ok(_) => {}
            Err(error) => issues.push(format!("cannot recompute protocol commitment: {error}")),
        }
    }

    let valid = issues.is_empty();
    let manifest_declares_scoring_eligible = valid
        && is_qualified
        && matches!(
            manifest.lane.as_str(),
            "immutable_anchor" | "sealed_refresh" | "reproducible_open"
        );
    let manifest_declares_sealed_generalization_profile =
        manifest_declares_scoring_eligible && manifest.lane == "sealed_refresh";
    let admission_reason = if !valid {
        "manifest failed one or more validation checks".to_owned()
    } else if !is_qualified {
        "protocol fixture is explicitly not admissible for scoring".to_owned()
    } else if manifest.lane == "public_development" {
        "public development tasks are not confirmatory scoring evidence".to_owned()
    } else if manifest_declares_sealed_generalization_profile {
        concat!(
            "manifest declares a sealed-refresh profile; ",
            "this validator cannot establish actual runner isolation"
        )
        .to_owned()
    } else if manifest.lane == "reproducible_open" {
        concat!(
            "scoring may be reported as open/reproducible, but this lane is ",
            "not sealed generalization evidence"
        )
        .to_owned()
    } else if manifest.lane == "immutable_anchor" {
        concat!(
            "eligible for the frozen anchor lane, subject to execution ",
            "and evaluator evidence"
        )
        .to_owned()
    } else {
        "lane is not eligible for scoring".to_owned()
    };

    Ok(ValidationReportV1 {
        schema: REPORT_SCHEMA.to_owned(),
        manifest_sha256: sha256_hex(bytes),
        valid,
        manifest_declares_scoring_eligible,
        manifest_declares_sealed_generalization_profile,
        expected_task_count: manifest.expected_task_count,
        planned_task_count,
        task_family_weight_sum: weight_sum,
        issues,
        admission_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn digest(letter: char) -> String {
        std::iter::repeat_n(letter, 64).collect()
    }

    fn valid_qualified_manifest(lane: &str) -> Value {
        let reveal_policy = if lane == "sealed_refresh" {
            "reveal_seed_and_task_bytes_after_submission"
        } else {
            "open_from_start_not_sealed"
        };
        let mut value = json!({
            "schema": MANIFEST_SCHEMA,
            "generation_id": "test-generation-0001",
            "purpose": "qualified_run",
            "lane": lane,
            "generator": {
                "repository": "https://example.invalid/generator",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "configuration_sha256": digest('a')
            },
            "evaluator": {
                "version": "test-evaluator-v1",
                "source_sha256": digest('c'),
                "oracle_separation": "independent_evaluator_boundary_verified"
            },
            "protocol_commitment_sha256": digest('0'),
            "seed_commitment_sha256": digest('0'),
            "task_set_commitment_sha256": digest('0'),
            "expected_task_count": 2,
            "task_family_mixture": [
                {"family_id": "family-a", "weight": 0.5, "planned_task_count": 1},
                {"family_id": "family-b", "weight": 0.5, "planned_task_count": 1}
            ],
            "policy_freeze": {
                "subject_id": "frozen-test-policy",
                "source_or_policy_sha256": digest('1'),
                "prompt_sha256": digest('2'),
                "configuration_sha256": digest('3'),
                "tool_manifest_sha256": digest('4')
            },
            "budget": {
                "max_observations": 100,
                "max_experiments": 20,
                "max_tool_calls": 10,
                "max_total_tokens": 50000,
                "max_wall_clock_seconds": 3600
            },
            "analysis_plan_sha256": digest('5'),
            "post_run_reveal_policy": reveal_policy
        });
        let seed = test_seed();
        let task_set = test_task_set();
        let raw = serde_json::to_vec(&value).expect("manifest serializes");
        let protocol = compute_protocol_commitment(&raw).expect("protocol commitment computes");
        value["protocol_commitment_sha256"] = json!(protocol);
        let raw = serde_json::to_vec(&value).expect("manifest serializes");
        value["seed_commitment_sha256"] = json!(
            compute_artifact_commitment(&raw, "seed", &seed).expect("seed commitment computes")
        );
        value["task_set_commitment_sha256"] = json!(
            compute_artifact_commitment(&raw, "task-set", &task_set)
                .expect("task-set commitment computes")
        );
        value
    }

    fn test_seed() -> Vec<u8> {
        vec![0x5a; 32]
    }

    fn test_task_set() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": TASK_SET_SCHEMA,
            "generation_id": "test-generation-0001",
            "tasks": [
                {
                    "task_id": "task-0001",
                    "family_id": "family-a",
                    "policy_input": {"observations": [{"x": 1.0, "y": 2.0}]},
                    "evaluator_only": {"expected_law": "y = 2*x"}
                },
                {
                    "task_id": "task-0002",
                    "family_id": "family-b",
                    "policy_input": {"observations": [{"x": 3.0, "y": 6.0}]},
                    "evaluator_only": {"expected_law": "y = 2*x"}
                }
            ]
        }))
        .expect("task-set serializes")
    }


    fn validate_value(value: &Value) -> Result<ValidationReportV1, String> {
        validate_manifest_bytes(&serde_json::to_vec(value).expect("JSON serializes"))
    }

    #[test]
    fn example_fixture_is_valid_but_never_score_admissible() {
        let report = validate_manifest_bytes(include_bytes!(
            "../fixtures/generation-manifest.example.json"
        ))
        .expect("example fixture has the v1 shape");
        assert!(report.valid, "{:?}", report.issues);
        assert!(!report.manifest_declares_scoring_eligible);
        assert!(!report.manifest_declares_sealed_generalization_profile);
        assert_eq!(report.expected_task_count, 8);
        assert_eq!(report.planned_task_count, 8);
        assert!((report.task_family_weight_sum - 1.0).abs() < WEIGHT_TOLERANCE);
    }

    #[test]
    fn valid_synthetic_sealed_manifest_is_scoring_admissible_only_at_contract_level() {
        let report = validate_value(&valid_qualified_manifest("sealed_refresh")).unwrap();
        assert!(report.valid, "{:?}", report.issues);
        assert!(report.manifest_declares_scoring_eligible);
        assert!(report.manifest_declares_sealed_generalization_profile);
        assert!(report
            .admission_reason
            .contains("cannot establish actual runner isolation"));
    }

    #[test]
    fn denominator_mismatch_fails_closed() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["expected_task_count"] = json!(3);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(!report.manifest_declares_scoring_eligible);
        assert!(report.issues.iter().any(|issue| issue.contains("denominator mismatch")));
    }

    #[test]
    fn family_weights_must_sum_to_one() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["task_family_mixture"][0]["weight"] = json!(0.2);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("weights must sum to 1")));
    }

    #[test]
    fn duplicate_family_ids_are_rejected() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["task_family_mixture"][1]["family_id"] = json!("family-a");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate task-family ID")));
    }

    #[test]
    fn unpinned_generator_revision_is_rejected_for_qualified_runs() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["generator"]["revision"] = json!("main");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("40-character lowercase Git commit SHA")));
    }

    #[test]
    fn placeholder_digests_are_rejected_for_qualified_runs() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["seed_commitment_sha256"] = json!(digest('0'));
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("all-zero placeholder")));
    }

    #[test]
    fn unverified_oracle_boundary_is_rejected_for_qualified_runs() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["evaluator"]["oracle_separation"] = json!("not_qualified_example");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(!report.manifest_declares_scoring_eligible);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("independently verified evaluator/oracle boundary")));
    }

    #[test]
    fn sealed_lane_rejects_open_from_start_policy() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["post_run_reveal_policy"] = json!("open_from_start_not_sealed");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("incompatible with the declared lane")));
    }

    #[test]
    fn public_development_is_not_scoring_admissible() {
        let report = validate_value(&valid_qualified_manifest("public_development")).unwrap();
        assert!(!report.valid); // Qualified public-development artifacts are a contract mismatch.
        assert!(!report.manifest_declares_scoring_eligible);
    }

    #[test]
    fn reproducible_open_is_not_sealed_generalization_evidence() {
        let report = validate_value(&valid_qualified_manifest("reproducible_open")).unwrap();
        assert!(report.valid, "{:?}", report.issues);
        assert!(report.manifest_declares_scoring_eligible);
        assert!(!report.manifest_declares_sealed_generalization_profile);
        assert!(report.admission_reason.contains("not sealed generalization evidence"));
    }

    #[test]
    fn unknown_top_level_oracle_field_is_rejected() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["evaluator_seed_hex"] = json!("deadbeef");
        let error = validate_value(&value).expect_err("unknown field must be denied");
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn wrong_schema_version_is_rejected() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["schema"] = json!("symthaea.science_bench_refresh_manifest.v0");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("schema must be")));
    }

    #[test]
    fn uppercase_hex_digests_are_rejected() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["analysis_plan_sha256"] = json!(digest('A'));
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("lowercase hexadecimal")));
    }

    #[test]
    fn zero_wall_clock_budget_is_rejected() {
        let mut value = valid_qualified_manifest("sealed_refresh");
        value["budget"]["max_wall_clock_seconds"] = json!(0);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("max_wall_clock_seconds")));
    }

    #[test]
    fn raw_manifest_digest_changes_when_input_bytes_change() {
        let value = valid_qualified_manifest("sealed_refresh");
        let compact = serde_json::to_vec(&value).unwrap();
        let pretty = serde_json::to_vec_pretty(&value).unwrap();
        let first = validate_manifest_bytes(&compact).unwrap();
        let second = validate_manifest_bytes(&pretty).unwrap();
        assert!(first.valid && second.valid);
        assert_ne!(first.manifest_sha256, second.manifest_sha256);
    }

    #[test]
    fn schema_document_is_valid_json_and_matches_manifest_version() {
        let schema: Value = serde_json::from_str(include_str!(
            "../schema/generation-manifest.schema.json"
        ))
        .expect("schema document must be valid JSON");
        assert_eq!(
            schema["properties"]["schema"]["const"].as_str(),
            Some(MANIFEST_SCHEMA)
        );
        assert!(schema["required"]
            .as_array()
            .expect("required fields array")
            .iter()
            .any(|field| field.as_str() == Some("expected_task_count")));
        assert!(schema["required"]
            .as_array()
            .expect("required fields array")
            .iter()
            .any(|field| field.as_str() == Some("protocol_commitment_sha256")));
    }


    #[test]
    fn schema_required_and_property_sets_match_for_strict_objects() {
        use std::collections::BTreeSet;

        fn property_names(value: &Value) -> BTreeSet<String> {
            value
                .as_object()
                .expect("schema properties object")
                .keys()
                .cloned()
                .collect()
        }

        fn required_names(value: &Value) -> BTreeSet<String> {
            value["required"]
                .as_array()
                .expect("schema required array")
                .iter()
                .map(|field| field.as_str().expect("field name").to_owned())
                .collect()
        }

        let schema: Value = serde_json::from_str(include_str!(
            "../schema/generation-manifest.schema.json"
        ))
        .expect("valid JSON Schema document");
        assert_eq!(
            property_names(&schema["properties"]),
            required_names(&schema)
        );

        for path in [
            &schema["properties"]["generator"],
            &schema["properties"]["evaluator"],
            &schema["properties"]["policy_freeze"],
            &schema["properties"]["budget"],
            &schema["properties"]["task_family_mixture"]["items"],
        ] {
            assert_eq!(
                property_names(&path["properties"]),
                required_names(path),
                "nested object property/required mismatch"
            );
        }
    }

    #[test]
    fn public_task_set_fixture_matches_placeholder_manifest_plan() {
        let manifest: GenerationManifestV1 = serde_json::from_slice(include_bytes!(
            "../fixtures/generation-manifest.example.json"
        ))
        .expect("placeholder manifest shape");
        let issues = validate_task_set_contract(
            &manifest,
            include_bytes!("../fixtures/task-set.example.json"),
        );
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn task_set_schema_is_valid_and_matches_strict_record_shape() {
        use std::collections::BTreeSet;

        fn names(value: &Value) -> BTreeSet<String> {
            value["properties"]
                .as_object()
                .expect("properties object")
                .keys()
                .cloned()
                .collect()
        }

        fn required(value: &Value) -> BTreeSet<String> {
            value["required"]
                .as_array()
                .expect("required array")
                .iter()
                .map(|item| item.as_str().expect("field name").to_owned())
                .collect()
        }

        let schema: Value = serde_json::from_str(include_str!(
            "../schema/task-set.schema.json"
        ))
        .expect("valid task-set JSON Schema");
        assert_eq!(names(&schema), required(&schema));
        let record = &schema["properties"]["tasks"]["items"];
        assert_eq!(names(record), required(record));
        assert_eq!(
            schema["properties"]["schema"]["const"].as_str(),
            Some(TASK_SET_SCHEMA)
        );
    }

    #[test]
    fn sha256_implementation_matches_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }


    #[test]
    fn revealed_seed_and_task_set_verify_against_exact_commitments() {
        let manifest = valid_qualified_manifest("sealed_refresh");
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        let report = verify_reveal(&manifest_bytes, &test_seed(), &test_task_set()).unwrap();
        assert!(report.verified, "{:?}", report.issues);
        assert!(report.seed_commitment_matches);
        assert!(report.task_set_commitment_matches);
        assert_eq!(report.seed_bytes, 32);
        assert_eq!(report.task_set_bytes, u64::try_from(test_task_set().len()).unwrap());
    }

    #[test]
    fn changed_seed_is_reported_as_commitment_mismatch() {
        let manifest = valid_qualified_manifest("sealed_refresh");
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let mut seed = test_seed();
        seed[0] ^= 0xff;
        let report = verify_reveal(&manifest_bytes, &seed, &test_task_set()).unwrap();
        assert!(!report.verified);
        assert!(!report.seed_commitment_matches);
        assert!(report.task_set_commitment_matches);
        assert!(report.issues.iter().any(|issue| issue.contains("seed does not match")));
    }

    #[test]
    fn changed_task_set_is_reported_as_commitment_mismatch() {
        let manifest = valid_qualified_manifest("sealed_refresh");
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let mut task_set = test_task_set();
        task_set.push(b' ');
        let report = verify_reveal(&manifest_bytes, &test_seed(), &task_set).unwrap();
        assert!(!report.verified);
        assert!(report.seed_commitment_matches);
        assert!(!report.task_set_commitment_matches);
        assert!(report.issues.iter().any(|issue| issue.contains("task-set bytes do not match")));
    }

    #[test]
    fn protocol_context_change_invalidates_frozen_commitment() {
        let mut manifest = valid_qualified_manifest("sealed_refresh");
        manifest["budget"]["max_total_tokens"] = json!(50001);
        let report = validate_value(&manifest).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("does not match the frozen manifest context")));
    }

    #[test]
    fn artifact_commitments_are_domain_separated() {
        let manifest = valid_qualified_manifest("sealed_refresh");
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let same_bytes = vec![0x44; 32];
        let seed_hash =
            compute_artifact_commitment(&manifest_bytes, "seed", &same_bytes).unwrap();
        let task_hash =
            compute_artifact_commitment(&manifest_bytes, "task-set", &same_bytes).unwrap();
        assert_ne!(seed_hash, task_hash);
    }

    #[test]
    fn short_seed_is_rejected_before_commitment() {
        let manifest = valid_qualified_manifest("sealed_refresh");
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let error = compute_artifact_commitment(&manifest_bytes, "seed", b"short")
            .expect_err("short seed must be rejected");
        assert!(error.contains("at least 32 bytes"));
    }

    #[test]
    fn reveal_verification_refuses_open_or_unqualified_manifests() {
        let manifest = valid_qualified_manifest("reproducible_open");
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let error = verify_reveal(&manifest_bytes, &test_seed(), &test_task_set())
            .expect_err("open benchmark lane must not claim sealed reveal verification");
        assert!(error.contains("lane=sealed_refresh"));
    }

    #[test]
    fn task_set_with_correct_digest_but_wrong_denominator_is_not_verified() {
        let mut manifest = valid_qualified_manifest("sealed_refresh");
        let malformed = serde_json::to_vec(&json!({
            "schema": TASK_SET_SCHEMA,
            "generation_id": "test-generation-0001",
            "tasks": [{
                "task_id": "task-0001",
                "family_id": "family-a",
                "policy_input": {"observations": []},
                "evaluator_only": {"expected_law": "y = 2*x"}
            }]
        }))
        .unwrap();
        let initial_manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let commitment =
            compute_artifact_commitment(&initial_manifest_bytes, "task-set", &malformed).unwrap();
        manifest["task_set_commitment_sha256"] = json!(commitment);
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let report = verify_reveal(&manifest_bytes, &test_seed(), &malformed).unwrap();
        assert!(report.task_set_commitment_matches);
        assert!(!report.task_set_contract_valid);
        assert!(!report.verified);
        assert!(report.issues.iter().any(|issue| issue.contains("record count")));
    }

    #[test]
    fn committed_task_set_with_nested_oracle_field_is_rejected() {
        let mut manifest = valid_qualified_manifest("sealed_refresh");
        let mut task_set: Value = serde_json::from_slice(&test_task_set()).unwrap();
        task_set["tasks"][0]["policy_input"]["nested"] =
            json!({"hidden_law": "y = 2*x"});
        let task_bytes = serde_json::to_vec(&task_set).unwrap();
        let initial_manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let commitment =
            compute_artifact_commitment(&initial_manifest_bytes, "task-set", &task_bytes).unwrap();
        manifest["task_set_commitment_sha256"] = json!(commitment);
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let report = verify_reveal(&manifest_bytes, &test_seed(), &task_bytes).unwrap();
        assert!(report.task_set_commitment_matches);
        assert!(!report.task_set_contract_valid);
        assert!(!report.verified);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("forbidden oracle field")));
    }

    #[test]
    fn duplicate_task_ids_are_rejected_even_when_bytes_are_committed() {
        let mut manifest = valid_qualified_manifest("sealed_refresh");
        let mut task_set: Value = serde_json::from_slice(&test_task_set()).unwrap();
        task_set["tasks"][1]["task_id"] = json!("task-0001");
        let task_bytes = serde_json::to_vec(&task_set).unwrap();
        let initial_manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        manifest["task_set_commitment_sha256"] = json!(
            compute_artifact_commitment(&initial_manifest_bytes, "task-set", &task_bytes).unwrap()
        );
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let report = verify_reveal(&manifest_bytes, &test_seed(), &task_bytes).unwrap();
        assert!(report.task_set_commitment_matches);
        assert!(!report.task_set_contract_valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate task_id")));
    }

    #[test]
    fn malformed_json_is_rejected() {
        let error = validate_manifest_bytes(b"{not-json")
            .expect_err("malformed JSON must not produce a report");
        assert!(error.contains("strict v1 shape"));
    }

}
