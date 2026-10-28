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
    pub seed_commitment_sha256: String,
    pub task_set_commitment_sha256: String,
    pub expected_task_count: u64,
    pub task_family_mixture: Vec<TaskFamilyV1>,
    pub policy_freeze: PolicyFreezeV1,
    pub budget: BudgetV1,
    pub analysis_plan_sha256: String,
    pub post_run_reveal_policy: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorV1 {
    pub repository: String,
    pub revision: String,
    pub configuration_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorV1 {
    pub version: String,
    pub source_sha256: String,
    pub oracle_separation: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskFamilyV1 {
    pub family_id: String,
    pub weight: f64,
    pub planned_task_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFreezeV1 {
    pub subject_id: String,
    pub source_or_policy_sha256: String,
    pub prompt_sha256: String,
    pub configuration_sha256: String,
    pub tool_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
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

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
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
        "manifest declares a sealed-refresh profile; this validator cannot establish actual runner isolation".to_owned()
    } else if manifest.lane == "reproducible_open" {
        "scoring may be reported as open/reproducible, but this lane is not sealed generalization evidence".to_owned()
    } else if manifest.lane == "immutable_anchor" {
        "eligible for the frozen anchor lane, subject to execution and evaluator evidence".to_owned()
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
        json!({
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
            "seed_commitment_sha256": digest('d'),
            "task_set_commitment_sha256": digest('e'),
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
        })
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
        assert!(report.issues.iter().any(|issue| issue.contains("40-character lowercase Git commit SHA")));
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
    }

    #[test]
    fn malformed_json_is_rejected() {
        let error = validate_manifest_bytes(b"{not-json")
            .expect_err("malformed JSON must not produce a report");
        assert!(error.contains("strict v1 shape"));
    }

}
