//! Judge-as-measurement-instrument reliability audit.
//!
//! This module scores declared verdict behavior against evaluator-only references.
//! It does not authenticate remote judge identity, prove calls executed, or turn
//! a commitment digest into evidence of truth. Qualified runs need independent
//! pre-run anchoring and auditable call/adjudication receipts.

use crate::BUILT_GENERATOR_SOURCE_REVISION;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const JUDGE_RELIABILITY_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_reliability_plan.v1";
pub const JUDGE_RELIABILITY_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_reliability_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.judge_reliability.v1\0";
const PLAN_DOMAIN: &[u8] = b"plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"report.v1\0";
const MAX_ITEMS: usize = 100_000;
const MAX_TRIALS: usize = 500_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeAuditPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl JudgeAuditPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeReferenceLabelV1 {
    Pass,
    Fail,
    NotEvaluable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgePerturbationKindV1 {
    Baseline,
    PromptVariant,
    OrderSwap,
    Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgePresentationOrderV1 {
    AThenB,
    BThenA,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeTerminalStatusV1 {
    Completed,
    Abstained,
    Error,
    Missing,
    InvalidReceipt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeVerdictV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeIdentityV1 {
    pub provider: String,
    pub model_id: String,
    pub model_revision: String,
    pub api_version: Option<String>,
    pub prompt_sha256: String,
    pub rubric_sha256: String,
    pub settings_sha256: String,
    pub evaluator_source_revision: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeReferencePolicyV1 {
    pub label_set_sha256: String,
    pub protocol_sha256: String,
    pub source: String,
    pub blinded_from_judge: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditItemV1 {
    pub case_id: String,
    pub contract_family: String,
    pub difficulty: String,
    pub reference_label: JudgeReferenceLabelV1,
    pub reference_evidence_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditTrialV1 {
    pub trial_id: String,
    pub case_id: String,
    pub perturbation_kind: JudgePerturbationKindV1,
    pub variant_id: String,
    pub replicate_index: u32,
    pub presentation_order: JudgePresentationOrderV1,
    pub prompt_sha256: String,
    pub expected_invariant_to_baseline: bool,
}

/// Typed plan payload. The final commitment field is intentionally excluded.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeReliabilityPlanFieldsV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub audit_id: String,
    pub judge: JudgeIdentityV1,
    pub reference: JudgeReferencePolicyV1,
    pub perturbation_policy_sha256: String,
    pub scoring_rules_sha256: String,
    pub items: Vec<JudgeAuditItemV1>,
    pub trials: Vec<JudgeAuditTrialV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeReliabilityPlanV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub audit_id: String,
    pub judge: JudgeIdentityV1,
    pub reference: JudgeReferencePolicyV1,
    pub perturbation_policy_sha256: String,
    pub scoring_rules_sha256: String,
    pub items: Vec<JudgeAuditItemV1>,
    pub trials: Vec<JudgeAuditTrialV1>,
    pub plan_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditRecordV1 {
    pub trial_id: String,
    pub case_id: String,
    pub terminal_status: JudgeTerminalStatusV1,
    pub verdict: Option<JudgeVerdictV1>,
    /// Probability in millionths. Null means it was not recorded.
    /// Probability that this verdict is correct, in millionths.
    pub confidence_millionths: Option<u32>,
    pub request_receipt_sha256: Option<String>,
    pub response_artifact_sha256: Option<String>,
    pub evaluator_record_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeReliabilityReportFieldsV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub judge_source_revision: String,
    pub scoring_rules_sha256: String,
    pub records: Vec<JudgeAuditRecordV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeReliabilityReportV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub judge_source_revision: String,
    pub scoring_rules_sha256: String,
    pub records: Vec<JudgeAuditRecordV1>,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgePerturbationMetricV1 {
    pub kind: JudgePerturbationKindV1,
    pub planned_comparisons: u64,
    pub comparable_comparisons: u64,
    pub flip_count: u64,
    pub incomplete_comparisons: u64,
    pub flip_rate_millionths: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeReliabilityStratumV1 {
    pub dimension: String,
    pub value: String,
    pub planned_baseline_items: u64,
    pub reference_evaluable_items: u64,
    pub completed_baseline_verdicts: u64,
    pub correct_count: u64,
    pub incorrect_count: u64,
    pub accuracy_millionths: Option<u32>,
    pub coverage_millionths: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeReliabilityScorecardV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub report_sha256: String,
    pub report_commitment_sha256: String,
    pub planned_item_count: u64,
    pub planned_trial_count: u64,
    pub observed_record_count: u64,
    pub missing_record_count: u64,
    pub completed_trial_count: u64,
    pub abstained_trial_count: u64,
    pub error_trial_count: u64,
    pub missing_trial_count: u64,
    pub invalid_receipt_count: u64,
    pub reference_evaluable_item_count: u64,
    pub baseline_completed_verdict_count: u64,
    pub baseline_correct_count: u64,
    pub baseline_incorrect_count: u64,
    pub baseline_accuracy_millionths: Option<u32>,
    pub baseline_coverage_millionths: Option<u32>,
    pub baseline_confidence_recorded_count: u64,
    pub baseline_brier_score_millionths: Option<u32>,
    pub baseline_expected_calibration_error_millionths: Option<u32>,
    pub baseline_sensitivity_millionths: Option<u32>,
    pub baseline_specificity_millionths: Option<u32>,
    pub baseline_balanced_accuracy_millionths: Option<u32>,
    pub trusted_item_count: u64,
    pub trusted_verdict_rate_millionths: Option<u32>,
    pub perturbation_metrics: Vec<JudgePerturbationMetricV1>,
    pub strata: Vec<JudgeReliabilityStratumV1>,
    pub fixed_benchmark_only: bool,
    pub generalized_accuracy_estimated: bool,
    pub judge_identity_authenticated: bool,
    pub external_anchor_required: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgePlanVerificationV1 {
    pub schema: String,
    pub valid: bool,
    pub structural_valid: bool,
    pub plan_commitment_matches: bool,
    pub plan_sha256: String,
    pub commitment_expected: String,
    pub commitment_recomputed: String,
    pub audit_id: String,
    pub planned_item_count: u64,
    pub planned_trial_count: u64,
    pub issues: Vec<String>,
    pub scoring_eligible_declared: bool,
    pub judge_identity_authenticated: bool,
    pub external_anchor_required: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeReportVerificationV1 {
    pub schema: String,
    pub valid: bool,
    pub plan_binding_valid: bool,
    pub report_commitment_matches: bool,
    pub planned_trial_count: u64,
    pub observed_record_count: u64,
    pub missing_record_count: u64,
    pub issues: Vec<String>,
    pub scorecard: Option<JudgeReliabilityScorecardV1>,
}

#[derive(Debug, Clone, Serialize)]
struct JudgeReportCommitmentPayloadV1<'a> {
    schema: &'a str,
    purpose: JudgeAuditPurposeV1,
    plan_sha256: &'a str,
    plan_commitment_sha256: &'a str,
    judge_source_revision: &'a str,
    scoring_rules_sha256: &'a str,
    records: &'a [JudgeAuditRecordV1],
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(value, "{byte:02x}");
    }
    value
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn framed_digest(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len =
        u64::try_from(domain.len()).map_err(|_| "digest domain exceeds u64".to_owned())?;
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| "digest payload exceeds u64".to_owned())?;
    let mut hash = Sha256::new();
    hash.update(HASH_PREFIX);
    hash.update(domain_len.to_be_bytes());
    hash.update(domain);
    hash.update(payload_len.to_be_bytes());
    hash.update(payload);
    Ok(hex_digest(&hash.finalize()))
}

fn fields_from_plan(plan: &JudgeReliabilityPlanV1) -> JudgeReliabilityPlanFieldsV1 {
    JudgeReliabilityPlanFieldsV1 {
        schema: plan.schema.clone(),
        purpose: plan.purpose,
        audit_id: plan.audit_id.clone(),
        judge: plan.judge.clone(),
        reference: plan.reference.clone(),
        perturbation_policy_sha256: plan.perturbation_policy_sha256.clone(),
        scoring_rules_sha256: plan.scoring_rules_sha256.clone(),
        items: plan.items.clone(),
        trials: plan.trials.clone(),
    }
}

fn validate_plan_fields(fields: &JudgeReliabilityPlanFieldsV1) -> Vec<String> {
    let mut issues = Vec::new();
    if fields.schema != JUDGE_RELIABILITY_PLAN_SCHEMA {
        issues.push("unsupported judge reliability plan schema".to_owned());
    }
    if fields.audit_id.trim().is_empty() || fields.audit_id.len() > 128 {
        issues.push("audit_id must be non-empty and at most 128 bytes".to_owned());
    }
    if fields.items.is_empty() || fields.items.len() > MAX_ITEMS {
        issues.push(format!("items must contain 1..={MAX_ITEMS} entries"));
    }
    if fields.trials.is_empty() || fields.trials.len() > MAX_TRIALS {
        issues.push(format!("trials must contain 1..={MAX_TRIALS} entries"));
    }
    let allow_zero = !fields.purpose.qualified();
    if fields.judge.provider.trim().is_empty()
        || fields.judge.model_id.trim().is_empty()
        || fields.judge.model_revision.trim().is_empty()
    {
        issues.push("judge provider, model id and model revision must be non-empty".to_owned());
    }
    if ![
        "deterministic_programmatic",
        "dual_human_adjudicated",
        "expert_adjudicated",
        "mixed_adjudicated",
    ]
    .contains(&fields.reference.source.as_str())
    {
        issues.push("unsupported reference label source".to_owned());
    }
    for (label, value) in [
        ("judge prompt", fields.judge.prompt_sha256.as_str()),
        ("judge rubric", fields.judge.rubric_sha256.as_str()),
        ("judge settings", fields.judge.settings_sha256.as_str()),
        (
            "reference label set",
            fields.reference.label_set_sha256.as_str(),
        ),
        (
            "reference protocol",
            fields.reference.protocol_sha256.as_str(),
        ),
        (
            "perturbation policy",
            fields.perturbation_policy_sha256.as_str(),
        ),
        ("scoring rules", fields.scoring_rules_sha256.as_str()),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!(
                "{label} digest is invalid or a forbidden placeholder"
            ));
        }
    }
    if fields.purpose.qualified() {
        if !valid_revision(&fields.judge.evaluator_source_revision) {
            issues.push("qualified audit requires a full lower-case judge source SHA".to_owned());
        } else if fields.judge.evaluator_source_revision != BUILT_GENERATOR_SOURCE_REVISION {
            issues
                .push("judge evaluator source revision does not match inspected build".to_owned());
        }
        if !fields.reference.blinded_from_judge {
            issues.push(
                "qualified audit requires reference labels blinded from the judge".to_owned(),
            );
        }
    } else if fields.judge.evaluator_source_revision.trim().is_empty() {
        issues.push("example-only judge revision marker must be non-empty".to_owned());
    }

    let mut item_map = BTreeMap::new();
    for item in &fields.items {
        if item.case_id.trim().is_empty() || item.case_id.len() > 128 {
            issues.push("case_id must be non-empty and at most 128 bytes".to_owned());
        }
        if item_map.insert(item.case_id.as_str(), item).is_some() {
            issues.push(format!("duplicate case_id {}", item.case_id));
        }
        if ![
            "update_compliance",
            "conflict_disclosure",
            "decoy_invariance",
            "deletion_calibration",
            "permutation_invariance",
            "evidence_force_monotonicity",
            "claim_evidence_completeness",
            "tool_workflow",
            "other",
        ]
        .contains(&item.contract_family.as_str())
        {
            issues.push(format!(
                "case {} has unsupported contract family",
                item.case_id
            ));
        }
        if !["easy", "medium", "hard", "adversarial"].contains(&item.difficulty.as_str()) {
            issues.push(format!("case {} has unsupported difficulty", item.case_id));
        }
        if !valid_digest(&item.reference_evidence_sha256, allow_zero) {
            issues.push(format!(
                "case {} has invalid or placeholder reference evidence digest",
                item.case_id
            ));
        }
    }

    let mut trial_ids = BTreeSet::new();
    let mut trial_cells = BTreeSet::new();
    let mut per_item: BTreeMap<&str, Vec<&JudgeAuditTrialV1>> = BTreeMap::new();
    for trial in &fields.trials {
        if trial.trial_id.trim().is_empty() || trial.trial_id.len() > 160 {
            issues.push("trial_id must be non-empty and at most 160 bytes".to_owned());
        }
        if trial.variant_id.trim().is_empty() || trial.variant_id.len() > 128 {
            issues.push(format!(
                "trial {} has an invalid variant_id",
                trial.trial_id
            ));
        }
        if !trial_ids.insert(trial.trial_id.as_str()) {
            issues.push(format!("duplicate trial_id {}", trial.trial_id));
        }
        let cell = (
            trial.case_id.clone(),
            trial.perturbation_kind,
            trial.variant_id.clone(),
            trial.replicate_index,
        );
        if !trial_cells.insert(cell) {
            issues.push(format!(
                "duplicate perturbation cell for trial {}",
                trial.trial_id
            ));
        }
        if !item_map.contains_key(trial.case_id.as_str()) {
            issues.push(format!(
                "trial {} references an unknown case",
                trial.trial_id
            ));
        }
        if !valid_digest(&trial.prompt_sha256, allow_zero) {
            issues.push(format!(
                "trial {} has an invalid prompt digest",
                trial.trial_id
            ));
        }
        let expected_invariant = trial.perturbation_kind != JudgePerturbationKindV1::Baseline;
        if trial.expected_invariant_to_baseline != expected_invariant {
            issues.push(format!(
                "trial {} has an inconsistent invariant-to-baseline flag",
                trial.trial_id
            ));
        }
        if let Some(item_trials) = per_item.get_mut(trial.case_id.as_str()) {
            item_trials.push(trial);
        } else {
            per_item.insert(trial.case_id.as_str(), vec![trial]);
        }
    }

    for item in &fields.items {
        let trials = per_item
            .get(item.case_id.as_str())
            .cloned()
            .unwrap_or_default();
        let baselines: Vec<_> = trials
            .iter()
            .copied()
            .filter(|trial| trial.perturbation_kind == JudgePerturbationKindV1::Baseline)
            .collect();
        if baselines.len() != 1 {
            issues.push(format!(
                "case {} must have exactly one baseline trial",
                item.case_id
            ));
            continue;
        }
        let baseline = baselines[0];
        if baseline.prompt_sha256 != fields.judge.prompt_sha256 {
            issues.push(format!(
                "case {} baseline prompt digest does not match judge prompt",
                item.case_id
            ));
        }
        if baseline.presentation_order != JudgePresentationOrderV1::AThenB {
            issues.push(format!(
                "case {} baseline must use a_then_b order",
                item.case_id
            ));
        }
        if baseline.expected_invariant_to_baseline || baseline.replicate_index != 0 {
            issues.push(format!(
                "case {} baseline trial metadata is invalid",
                item.case_id
            ));
        }
        for kind in [
            JudgePerturbationKindV1::PromptVariant,
            JudgePerturbationKindV1::OrderSwap,
            JudgePerturbationKindV1::Repeat,
        ] {
            let matching: Vec<_> = trials
                .iter()
                .copied()
                .filter(|trial| trial.perturbation_kind == kind)
                .collect();
            if matching.is_empty() {
                issues.push(format!(
                    "case {} is missing required {:?} trial",
                    item.case_id, kind
                ));
            }
            for trial in matching {
                match kind {
                    JudgePerturbationKindV1::PromptVariant => {
                        if trial.prompt_sha256 == baseline.prompt_sha256 {
                            issues.push(format!(
                                "case {} prompt variant reuses baseline prompt digest",
                                item.case_id
                            ));
                        }
                        if trial.presentation_order != baseline.presentation_order {
                            issues.push(format!(
                                "case {} prompt variant changes presentation order",
                                item.case_id
                            ));
                        }
                    }
                    JudgePerturbationKindV1::OrderSwap => {
                        if trial.prompt_sha256 != baseline.prompt_sha256
                            || trial.presentation_order != JudgePresentationOrderV1::BThenA
                        {
                            issues.push(format!(
                                "case {} order swap must preserve prompt and reverse order",
                                item.case_id
                            ));
                        }
                    }
                    JudgePerturbationKindV1::Repeat => {
                        if trial.prompt_sha256 != baseline.prompt_sha256
                            || trial.presentation_order != baseline.presentation_order
                            || trial.replicate_index == 0
                        {
                            issues.push(format!(
                                "case {} repeat must reuse baseline config and use replicate > 0",
                                item.case_id
                            ));
                        }
                    }
                    JudgePerturbationKindV1::Baseline => {}
                }
            }
        }
    }
    issues
}

fn require_keys(bytes: &[u8], full: bool, report: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("audit JSON is malformed: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "audit artifact must be a JSON object".to_owned())?;
    let keys: &[&str] = if report {
        if full {
            &[
                "schema",
                "purpose",
                "plan_sha256",
                "plan_commitment_sha256",
                "judge_source_revision",
                "scoring_rules_sha256",
                "records",
                "report_commitment_sha256",
            ]
        } else {
            &[
                "schema",
                "purpose",
                "plan_sha256",
                "plan_commitment_sha256",
                "judge_source_revision",
                "scoring_rules_sha256",
                "records",
            ]
        }
    } else if full {
        &[
            "schema",
            "purpose",
            "audit_id",
            "judge",
            "reference",
            "perturbation_policy_sha256",
            "scoring_rules_sha256",
            "items",
            "trials",
            "plan_commitment_sha256",
        ]
    } else {
        &[
            "schema",
            "purpose",
            "audit_id",
            "judge",
            "reference",
            "perturbation_policy_sha256",
            "scoring_rules_sha256",
            "items",
            "trials",
        ]
    };
    for key in keys {
        if !object.contains_key(*key) {
            return Err(format!("audit artifact is missing required field {key}"));
        }
    }
    if report {
        let records = value["records"]
            .as_array()
            .ok_or_else(|| "judge report records must be an array".to_owned())?;
        const RECORD_KEYS: &[&str] = &[
            "trial_id",
            "case_id",
            "terminal_status",
            "verdict",
            "confidence_millionths",
            "request_receipt_sha256",
            "response_artifact_sha256",
            "evaluator_record_sha256",
        ];
        for (index, record) in records.iter().enumerate() {
            let record_object = record
                .as_object()
                .ok_or_else(|| format!("judge report record {index} must be an object"))?;
            for key in RECORD_KEYS {
                if !record_object.contains_key(*key) {
                    return Err(format!(
                        "judge report record {index} is missing required field {key}"
                    ));
                }
            }
        }
    } else {
        let judge = value["judge"]
            .as_object()
            .ok_or_else(|| "judge plan judge must be an object".to_owned())?;
        for key in [
            "provider",
            "model_id",
            "model_revision",
            "api_version",
            "prompt_sha256",
            "rubric_sha256",
            "settings_sha256",
            "evaluator_source_revision",
        ] {
            if !judge.contains_key(key) {
                return Err(format!("judge identity is missing required field {key}"));
            }
        }
        let reference = value["reference"]
            .as_object()
            .ok_or_else(|| "judge plan reference must be an object".to_owned())?;
        for key in [
            "label_set_sha256",
            "protocol_sha256",
            "source",
            "blinded_from_judge",
        ] {
            if !reference.contains_key(key) {
                return Err(format!("judge reference is missing required field {key}"));
            }
        }
    }
    Ok(())
}

pub fn compute_judge_reliability_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    require_keys(fields_bytes, false, false)?;
    let fields: JudgeReliabilityPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("judge plan fields are invalid: {error}"))?;
    let issues = validate_plan_fields(&fields);
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize judge plan commitment: {error}"))?;
    framed_digest(PLAN_DOMAIN, &payload)
}

pub fn verify_judge_reliability_plan(plan_bytes: &[u8]) -> Result<JudgePlanVerificationV1, String> {
    require_keys(plan_bytes, true, false)?;
    let plan: JudgeReliabilityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("judge plan is invalid: {error}"))?;
    let fields = fields_from_plan(&plan);
    let mut issues = validate_plan_fields(&fields);
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize plan payload: {error}"))?;
    let recomputed = framed_digest(PLAN_DOMAIN, &payload)?;
    let commitment_matches = plan.plan_commitment_sha256 == recomputed
        && valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified());
    if !commitment_matches {
        issues.push("judge plan commitment does not match typed fields".to_owned());
    }
    let structural_valid = issues
        .iter()
        .all(|issue| issue != "judge plan commitment does not match typed fields");
    Ok(JudgePlanVerificationV1 {
        schema: "symthaea.science_bench_refresh_judge_plan_verification.v1".to_owned(),
        valid: structural_valid && commitment_matches && plan.purpose.qualified(),
        structural_valid,
        plan_commitment_matches: commitment_matches,
        plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
        commitment_expected: plan.plan_commitment_sha256.clone(),
        commitment_recomputed: recomputed,
        audit_id: plan.audit_id,
        planned_item_count: plan.items.len() as u64,
        planned_trial_count: plan.trials.len() as u64,
        issues,
        scoring_eligible_declared: plan.purpose.qualified(),
        judge_identity_authenticated: false,
        external_anchor_required: true,
    })
}

fn report_fields_from_report(report: &JudgeReliabilityReportV1) -> JudgeReliabilityReportFieldsV1 {
    JudgeReliabilityReportFieldsV1 {
        schema: report.schema.clone(),
        purpose: report.purpose,
        plan_sha256: report.plan_sha256.clone(),
        plan_commitment_sha256: report.plan_commitment_sha256.clone(),
        judge_source_revision: report.judge_source_revision.clone(),
        scoring_rules_sha256: report.scoring_rules_sha256.clone(),
        records: report.records.clone(),
    }
}

fn validate_record_semantics(record: &JudgeAuditRecordV1, allow_zero: bool) -> Result<(), String> {
    if !valid_digest(&record.evaluator_record_sha256, allow_zero) {
        return Err(format!(
            "trial {} has invalid evaluator record digest",
            record.trial_id
        ));
    }
    if let Some(confidence) = record.confidence_millionths {
        if confidence > 1_000_000 {
            return Err(format!(
                "trial {} confidence is outside 0..=1e6",
                record.trial_id
            ));
        }
    }
    let valid_optional_digest = |value: &Option<String>| {
        value
            .as_ref()
            .is_none_or(|digest| valid_digest(digest, allow_zero))
    };
    if !valid_optional_digest(&record.request_receipt_sha256)
        || !valid_optional_digest(&record.response_artifact_sha256)
    {
        return Err(format!(
            "trial {} has an invalid artifact digest",
            record.trial_id
        ));
    }
    match record.terminal_status {
        JudgeTerminalStatusV1::Completed => {
            if record.verdict.is_none()
                || record.request_receipt_sha256.is_none()
                || record.response_artifact_sha256.is_none()
            {
                return Err(format!(
                    "completed trial {} requires verdict, request receipt and response artifact",
                    record.trial_id
                ));
            }
        }
        JudgeTerminalStatusV1::Abstained => {
            if record.verdict.is_some() || record.response_artifact_sha256.is_none() {
                return Err(format!(
                    "abstained trial {} requires null verdict and response artifact",
                    record.trial_id
                ));
            }
        }
        JudgeTerminalStatusV1::Error
        | JudgeTerminalStatusV1::Missing
        | JudgeTerminalStatusV1::InvalidReceipt => {
            if record.verdict.is_some() {
                return Err(format!(
                    "non-completed trial {} must not carry a binary verdict",
                    record.trial_id
                ));
            }
        }
    }
    Ok(())
}

fn report_commitment(fields: &JudgeReliabilityReportFieldsV1) -> Result<String, String> {
    let payload = JudgeReportCommitmentPayloadV1 {
        schema: &fields.schema,
        purpose: fields.purpose,
        plan_sha256: &fields.plan_sha256,
        plan_commitment_sha256: &fields.plan_commitment_sha256,
        judge_source_revision: &fields.judge_source_revision,
        scoring_rules_sha256: &fields.scoring_rules_sha256,
        records: &fields.records,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize judge report commitment: {error}"))?;
    framed_digest(REPORT_DOMAIN, &bytes)
}

fn validate_report_fields(
    plan: &JudgeReliabilityPlanV1,
    plan_bytes: &[u8],
    fields: &JudgeReliabilityReportFieldsV1,
) -> Result<Vec<String>, String> {
    let mut issues = Vec::new();
    let expected_plan_sha = hex_digest(&Sha256::digest(plan_bytes));
    if fields.schema != JUDGE_RELIABILITY_REPORT_SCHEMA {
        issues.push("unsupported judge reliability report schema".to_owned());
    }
    if fields.purpose != plan.purpose {
        issues.push("report purpose does not match plan".to_owned());
    }
    if fields.plan_sha256 != expected_plan_sha {
        issues.push("report does not bind exact plan bytes".to_owned());
    }
    if fields.plan_commitment_sha256 != plan.plan_commitment_sha256 {
        issues.push("report plan commitment does not match plan".to_owned());
    }
    if fields.judge_source_revision != plan.judge.evaluator_source_revision {
        issues.push("report judge source revision does not match plan".to_owned());
    }
    if fields.scoring_rules_sha256 != plan.scoring_rules_sha256 {
        issues.push("report scoring-rules digest does not match plan".to_owned());
    }
    if fields.records.len() != plan.trials.len() {
        issues.push(format!(
            "report record count {} differs from planned trial count {}",
            fields.records.len(),
            plan.trials.len()
        ));
    }
    let allow_zero = !fields.purpose.qualified();
    let planned: BTreeMap<&str, &JudgeAuditTrialV1> = plan
        .trials
        .iter()
        .map(|trial| (trial.trial_id.as_str(), trial))
        .collect();
    let mut seen = BTreeSet::new();
    for record in &fields.records {
        if !seen.insert(record.trial_id.as_str()) {
            issues.push(format!("duplicate record for trial {}", record.trial_id));
            continue;
        }
        match planned.get(record.trial_id.as_str()) {
            Some(trial) if trial.case_id == record.case_id => {}
            Some(_) => issues.push(format!("record {} has mismatched case_id", record.trial_id)),
            None => issues.push(format!("unplanned trial record {}", record.trial_id)),
        }
        if let Err(issue) = validate_record_semantics(record, allow_zero) {
            issues.push(issue);
        }
    }
    for trial in &plan.trials {
        if !seen.contains(trial.trial_id.as_str()) {
            issues.push(format!(
                "missing terminal record for trial {}",
                trial.trial_id
            ));
        }
    }
    Ok(issues)
}

pub fn compute_judge_reliability_report_commitment(
    plan_bytes: &[u8],
    fields_bytes: &[u8],
) -> Result<String, String> {
    require_keys(fields_bytes, false, true)?;
    let plan_check = verify_judge_reliability_plan(plan_bytes)?;
    if !plan_check.valid {
        return Err("cannot commit report against invalid/non-qualified plan".to_owned());
    }
    let plan: JudgeReliabilityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("judge plan is invalid: {error}"))?;
    let fields: JudgeReliabilityReportFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("judge report fields are invalid: {error}"))?;
    let issues = validate_report_fields(&plan, plan_bytes, &fields)?;
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    report_commitment(&fields)
}

fn ratio_millionths(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        return None;
    }
    let rounded =
        (u128::from(numerator) * 1_000_000 + u128::from(denominator) / 2) / u128::from(denominator);
    Some(rounded.min(1_000_000) as u32)
}

fn item_is_trusted(
    item: &JudgeAuditItemV1,
    plan: &JudgeReliabilityPlanV1,
    records: &BTreeMap<&str, &JudgeAuditRecordV1>,
) -> bool {
    if item.reference_label == JudgeReferenceLabelV1::NotEvaluable {
        return false;
    }
    let baseline_trial = match plan.trials.iter().find(|trial| {
        trial.case_id == item.case_id
            && trial.perturbation_kind == JudgePerturbationKindV1::Baseline
    }) {
        Some(trial) => trial,
        None => return false,
    };
    let baseline = match records.get(baseline_trial.trial_id.as_str()) {
        Some(record)
            if record.terminal_status == JudgeTerminalStatusV1::Completed
                && record.verdict.is_some() =>
        {
            *record
        }
        _ => return false,
    };
    let target = match item.reference_label {
        JudgeReferenceLabelV1::Pass => JudgeVerdictV1::Pass,
        JudgeReferenceLabelV1::Fail => JudgeVerdictV1::Fail,
        JudgeReferenceLabelV1::NotEvaluable => return false,
    };
    if baseline.verdict != Some(target) {
        return false;
    }
    let invariance_trials: Vec<_> = plan
        .trials
        .iter()
        .filter(|trial| trial.case_id == item.case_id && trial.expected_invariant_to_baseline)
        .collect();
    if invariance_trials.is_empty() {
        return false;
    }
    invariance_trials.iter().all(|trial| {
        records.get(trial.trial_id.as_str()).is_some_and(|record| {
            record.terminal_status == JudgeTerminalStatusV1::Completed
                && record.verdict == Some(target)
        })
    })
}

fn perturbation_metrics(
    plan: &JudgeReliabilityPlanV1,
    records: &BTreeMap<&str, &JudgeAuditRecordV1>,
) -> Vec<JudgePerturbationMetricV1> {
    [
        JudgePerturbationKindV1::PromptVariant,
        JudgePerturbationKindV1::OrderSwap,
        JudgePerturbationKindV1::Repeat,
    ]
    .into_iter()
    .map(|kind| {
        let trials: Vec<_> = plan
            .trials
            .iter()
            .filter(|trial| trial.perturbation_kind == kind && trial.expected_invariant_to_baseline)
            .collect();
        let mut comparable = 0_u64;
        let mut flips = 0_u64;
        for trial in &trials {
            let baseline_trial = match plan.trials.iter().find(|base| {
                base.case_id == trial.case_id
                    && base.perturbation_kind == JudgePerturbationKindV1::Baseline
            }) {
                Some(value) => value,
                None => continue,
            };
            let pair = (
                records.get(baseline_trial.trial_id.as_str()),
                records.get(trial.trial_id.as_str()),
            );
            if let (Some(base), Some(candidate)) = pair {
                if base.terminal_status == JudgeTerminalStatusV1::Completed
                    && candidate.terminal_status == JudgeTerminalStatusV1::Completed
                {
                    if let (Some(base_verdict), Some(candidate_verdict)) =
                        (base.verdict, candidate.verdict)
                    {
                        comparable += 1;
                        if base_verdict != candidate_verdict {
                            flips += 1;
                        }
                    }
                }
            }
        }
        let planned = trials.len() as u64;
        JudgePerturbationMetricV1 {
            kind,
            planned_comparisons: planned,
            comparable_comparisons: comparable,
            flip_count: flips,
            incomplete_comparisons: planned.saturating_sub(comparable),
            flip_rate_millionths: ratio_millionths(flips, comparable),
        }
    })
    .collect()
}

fn baseline_strata(
    plan: &JudgeReliabilityPlanV1,
    records: &BTreeMap<&str, &JudgeAuditRecordV1>,
) -> Vec<JudgeReliabilityStratumV1> {
    #[derive(Default)]
    struct Counts {
        planned: u64,
        reference_evaluable: u64,
        completed: u64,
        correct: u64,
        incorrect: u64,
    }
    let mut counts: BTreeMap<(String, String), Counts> = BTreeMap::new();
    for item in &plan.items {
        let baseline_trial = plan.trials.iter().find(|trial| {
            trial.case_id == item.case_id
                && trial.perturbation_kind == JudgePerturbationKindV1::Baseline
        });
        for (dimension, value) in [
            ("contract_family".to_owned(), item.contract_family.clone()),
            ("difficulty".to_owned(), item.difficulty.clone()),
        ] {
            let entry = counts.entry((dimension, value)).or_default();
            entry.planned += 1;
            if item.reference_label == JudgeReferenceLabelV1::NotEvaluable {
                continue;
            }
            entry.reference_evaluable += 1;
            if let Some(record) =
                baseline_trial.and_then(|trial| records.get(trial.trial_id.as_str()).copied())
            {
                if record.terminal_status == JudgeTerminalStatusV1::Completed {
                    if let Some(verdict) = record.verdict {
                        entry.completed += 1;
                        let correct = match item.reference_label {
                            JudgeReferenceLabelV1::Pass => verdict == JudgeVerdictV1::Pass,
                            JudgeReferenceLabelV1::Fail => verdict == JudgeVerdictV1::Fail,
                            JudgeReferenceLabelV1::NotEvaluable => false,
                        };
                        if correct {
                            entry.correct += 1;
                        } else {
                            entry.incorrect += 1;
                        }
                    }
                }
            }
        }
    }
    counts
        .into_iter()
        .map(|((dimension, value), count)| JudgeReliabilityStratumV1 {
            dimension,
            value,
            planned_baseline_items: count.planned,
            reference_evaluable_items: count.reference_evaluable,
            completed_baseline_verdicts: count.completed,
            correct_count: count.correct,
            incorrect_count: count.incorrect,
            accuracy_millionths: ratio_millionths(count.correct, count.completed),
            coverage_millionths: ratio_millionths(count.completed, count.reference_evaluable),
        })
        .collect()
}

fn baseline_calibration(
    plan: &JudgeReliabilityPlanV1,
    records: &BTreeMap<&str, &JudgeAuditRecordV1>,
) -> (u64, Option<u32>, Option<u32>) {
    let mut count = 0_u64;
    let mut squared_error_sum = 0_u128;
    // Each bin stores (sample count, confidence sum, correct verdict count).
    let mut bins = [(0_u64, 0_u64, 0_u64); 10];
    for item in &plan.items {
        if item.reference_label == JudgeReferenceLabelV1::NotEvaluable {
            continue;
        }
        let baseline_trial = plan.trials.iter().find(|trial| {
            trial.case_id == item.case_id
                && trial.perturbation_kind == JudgePerturbationKindV1::Baseline
        });
        let Some(record) =
            baseline_trial.and_then(|trial| records.get(trial.trial_id.as_str()).copied())
        else {
            continue;
        };
        if record.terminal_status != JudgeTerminalStatusV1::Completed {
            continue;
        }
        let (Some(verdict), Some(confidence)) = (record.verdict, record.confidence_millionths)
        else {
            continue;
        };
        let correct = matches!(
            (item.reference_label, verdict),
            (JudgeReferenceLabelV1::Pass, JudgeVerdictV1::Pass)
                | (JudgeReferenceLabelV1::Fail, JudgeVerdictV1::Fail)
        );
        let target = if correct { 1_000_000_i64 } else { 0_i64 };
        let error = i64::from(confidence) - target;
        squared_error_sum += u128::from(error.unsigned_abs()).pow(2);
        let bin_index = ((u64::from(confidence) * 10) / 1_000_000).min(9) as usize;
        bins[bin_index].0 += 1;
        bins[bin_index].1 += u64::from(confidence);
        if correct {
            bins[bin_index].2 += 1;
        }
        count += 1;
    }
    if count == 0 {
        return (0, None, None);
    }
    let brier_denominator = u128::from(count) * 1_000_000;
    let brier =
        ((squared_error_sum + brier_denominator / 2) / brier_denominator).min(1_000_000) as u32;
    let mut weighted_calibration_error = 0_u128;
    for (bin_count, confidence_sum, correct_count) in bins {
        if bin_count == 0 {
            continue;
        }
        let mean_confidence = (confidence_sum + bin_count / 2) / bin_count;
        let observed_accuracy = (correct_count * 1_000_000 + bin_count / 2) / bin_count;
        let gap = mean_confidence.abs_diff(observed_accuracy);
        weighted_calibration_error += u128::from(bin_count) * u128::from(gap);
    }
    let ece = ((weighted_calibration_error + u128::from(count) / 2) / u128::from(count))
        .min(1_000_000) as u32;
    (count, Some(brier), Some(ece))
}

fn make_scorecard(
    plan_bytes: &[u8],
    report_bytes: &[u8],
    plan: &JudgeReliabilityPlanV1,
    report: &JudgeReliabilityReportV1,
) -> JudgeReliabilityScorecardV1 {
    let records: BTreeMap<&str, &JudgeAuditRecordV1> = report
        .records
        .iter()
        .map(|record| (record.trial_id.as_str(), record))
        .collect();
    let mut completed = 0_u64;
    let mut abstained = 0_u64;
    let mut errors = 0_u64;
    let mut missing_trials = 0_u64;
    let mut invalid_receipts = 0_u64;
    for record in &report.records {
        match record.terminal_status {
            JudgeTerminalStatusV1::Completed => completed += 1,
            JudgeTerminalStatusV1::Abstained => abstained += 1,
            JudgeTerminalStatusV1::Error => errors += 1,
            JudgeTerminalStatusV1::Missing => missing_trials += 1,
            JudgeTerminalStatusV1::InvalidReceipt => invalid_receipts += 1,
        }
    }
    let mut baseline_evaluable = 0_u64;
    let mut baseline_completed = 0_u64;
    let mut baseline_correct = 0_u64;
    let mut tp = 0_u64;
    let mut fn_count = 0_u64;
    let mut tn = 0_u64;
    let mut fp = 0_u64;
    let mut trusted = 0_u64;
    for item in &plan.items {
        let evaluable = item.reference_label != JudgeReferenceLabelV1::NotEvaluable;
        if evaluable {
            baseline_evaluable += 1;
        }
        let baseline_trial = plan.trials.iter().find(|trial| {
            trial.case_id == item.case_id
                && trial.perturbation_kind == JudgePerturbationKindV1::Baseline
        });
        let baseline_record =
            baseline_trial.and_then(|trial| records.get(trial.trial_id.as_str()).copied());
        if let Some(record) = baseline_record {
            if evaluable
                && record.terminal_status == JudgeTerminalStatusV1::Completed
                && record.verdict.is_some()
            {
                baseline_completed += 1;
                let verdict = record.verdict.expect("checked above");
                match (item.reference_label, verdict) {
                    (JudgeReferenceLabelV1::Pass, JudgeVerdictV1::Pass) => {
                        baseline_correct += 1;
                        tp += 1;
                    }
                    (JudgeReferenceLabelV1::Pass, JudgeVerdictV1::Fail) => fn_count += 1,
                    (JudgeReferenceLabelV1::Fail, JudgeVerdictV1::Fail) => {
                        baseline_correct += 1;
                        tn += 1;
                    }
                    (JudgeReferenceLabelV1::Fail, JudgeVerdictV1::Pass) => fp += 1,
                    (JudgeReferenceLabelV1::NotEvaluable, _) => {}
                }
            }
        }
        if item_is_trusted(item, plan, &records) {
            trusted += 1;
        }
    }
    let baseline_incorrect = baseline_completed.saturating_sub(baseline_correct);
    let sensitivity = ratio_millionths(tp, tp + fn_count);
    let specificity = ratio_millionths(tn, tn + fp);
    let balanced = match (sensitivity, specificity) {
        (Some(left), Some(right)) => Some((left + right) / 2),
        _ => None,
    };
    let (confidence_count, brier, expected_calibration_error) =
        baseline_calibration(plan, &records);
    JudgeReliabilityScorecardV1 {
        schema: "symthaea.science_bench_refresh_judge_reliability_scorecard.v1".to_owned(),
        plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
        plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
        report_sha256: hex_digest(&Sha256::digest(report_bytes)),
        report_commitment_sha256: report.report_commitment_sha256.clone(),
        planned_item_count: plan.items.len() as u64,
        planned_trial_count: plan.trials.len() as u64,
        observed_record_count: report.records.len() as u64,
        missing_record_count: plan
            .trials
            .iter()
            .filter(|trial| !records.contains_key(trial.trial_id.as_str()))
            .count() as u64,
        completed_trial_count: completed,
        abstained_trial_count: abstained,
        error_trial_count: errors,
        missing_trial_count: missing_trials,
        invalid_receipt_count: invalid_receipts,
        reference_evaluable_item_count: baseline_evaluable,
        baseline_completed_verdict_count: baseline_completed,
        baseline_correct_count: baseline_correct,
        baseline_incorrect_count: baseline_incorrect,
        baseline_accuracy_millionths: ratio_millionths(baseline_correct, baseline_completed),
        baseline_coverage_millionths: ratio_millionths(baseline_completed, baseline_evaluable),
        baseline_confidence_recorded_count: confidence_count,
        baseline_brier_score_millionths: brier,
        baseline_expected_calibration_error_millionths: expected_calibration_error,
        baseline_sensitivity_millionths: sensitivity,
        baseline_specificity_millionths: specificity,
        baseline_balanced_accuracy_millionths: balanced,
        trusted_item_count: trusted,
        trusted_verdict_rate_millionths: ratio_millionths(trusted, baseline_evaluable),
        perturbation_metrics: perturbation_metrics(plan, &records),
        strata: baseline_strata(plan, &records),
        fixed_benchmark_only: true,
        generalized_accuracy_estimated: false,
        judge_identity_authenticated: false,
        external_anchor_required: true,
        claim_boundary: concat!(
            "A digest commits to declared bytes; it does not authenticate judge identity, ",
            "call execution, adjudicator independence, truth or generalization."
        )
        .to_owned(),
    }
}

/// Report validity concerns receipt integrity; a valid report can contain
/// systematically wrong verdicts. The scorecard exposes accuracy and stability
/// separately, and the trusted-verdict rate requires both correctness and
/// invariance under every predeclared prompt/order/repeat perturbation.
pub fn verify_judge_reliability_report(
    plan_bytes: &[u8],
    report_bytes: &[u8],
) -> Result<JudgeReportVerificationV1, String> {
    require_keys(plan_bytes, true, false)?;
    require_keys(report_bytes, true, true)?;
    let plan_verification = verify_judge_reliability_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err("cannot verify judge report against invalid/non-qualified plan".to_owned());
    }
    let plan: JudgeReliabilityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("judge plan is invalid: {error}"))?;
    let report: JudgeReliabilityReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("judge report is invalid: {error}"))?;
    let fields = JudgeReliabilityReportFieldsV1 {
        schema: report.schema.clone(),
        purpose: report.purpose,
        plan_sha256: report.plan_sha256.clone(),
        plan_commitment_sha256: report.plan_commitment_sha256.clone(),
        judge_source_revision: report.judge_source_revision.clone(),
        scoring_rules_sha256: report.scoring_rules_sha256.clone(),
        records: report.records.clone(),
    };
    let issues = validate_report_fields(&plan, plan_bytes, &fields)?;
    let recomputed = report_commitment(&fields)?;
    let commitment_matches = report.report_commitment_sha256 == recomputed
        && valid_digest(
            &report.report_commitment_sha256,
            !report.purpose.qualified(),
        );
    let mut final_issues = issues;
    if !commitment_matches {
        final_issues.push("judge report commitment does not match typed fields".to_owned());
    }
    let valid = final_issues.is_empty() && commitment_matches && report.purpose.qualified();
    let expected_plan_sha = hex_digest(&Sha256::digest(plan_bytes));
    let plan_binding_valid = report.schema == JUDGE_RELIABILITY_REPORT_SCHEMA
        && report.purpose == plan.purpose
        && report.plan_sha256 == expected_plan_sha
        && report.plan_commitment_sha256 == plan.plan_commitment_sha256
        && report.judge_source_revision == plan.judge.evaluator_source_revision
        && report.scoring_rules_sha256 == plan.scoring_rules_sha256;
    let report_trial_ids: BTreeSet<&str> = report
        .records
        .iter()
        .map(|record| record.trial_id.as_str())
        .collect();
    let missing = plan
        .trials
        .iter()
        .filter(|trial| !report_trial_ids.contains(trial.trial_id.as_str()))
        .count() as u64;
    let scorecard = Some(make_scorecard(plan_bytes, report_bytes, &plan, &report));
    Ok(JudgeReportVerificationV1 {
        schema: "symthaea.science_bench_refresh_judge_report_verification.v1".to_owned(),
        valid,
        plan_binding_valid,
        report_commitment_matches: commitment_matches,
        planned_trial_count: plan.trials.len() as u64,
        observed_record_count: report.records.len() as u64,
        missing_record_count: missing,
        issues: final_issues,
        scorecard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(char: char) -> String {
        std::iter::repeat_n(char, 64).collect()
    }

    fn plan_fields() -> JudgeReliabilityPlanFieldsV1 {
        let source = BUILT_GENERATOR_SOURCE_REVISION.to_owned();
        let judge = JudgeIdentityV1 {
            provider: "synthetic".to_owned(),
            model_id: "synthetic-judge".to_owned(),
            model_revision: "judge-test-v1".to_owned(),
            api_version: Some("fixture".to_owned()),
            prompt_sha256: digest('a'),
            rubric_sha256: digest('b'),
            settings_sha256: digest('c'),
            evaluator_source_revision: source,
        };
        let reference = JudgeReferencePolicyV1 {
            label_set_sha256: digest('d'),
            protocol_sha256: digest('e'),
            source: "deterministic_programmatic".to_owned(),
            blinded_from_judge: true,
        };
        let mut items = Vec::new();
        let mut trials = Vec::new();
        for (case_id, label) in [
            ("stable-correct", JudgeReferenceLabelV1::Pass),
            ("stable-wrong", JudgeReferenceLabelV1::Fail),
        ] {
            items.push(JudgeAuditItemV1 {
                case_id: case_id.to_owned(),
                contract_family: "conflict_disclosure".to_owned(),
                difficulty: "easy".to_owned(),
                reference_label: label,
                reference_evidence_sha256: digest('f'),
            });
            let variants = [
                (
                    "baseline",
                    JudgePerturbationKindV1::Baseline,
                    0,
                    JudgePresentationOrderV1::AThenB,
                    digest('a'),
                    false,
                ),
                (
                    "prompt",
                    JudgePerturbationKindV1::PromptVariant,
                    0,
                    JudgePresentationOrderV1::AThenB,
                    digest('9'),
                    true,
                ),
                (
                    "order",
                    JudgePerturbationKindV1::OrderSwap,
                    0,
                    JudgePresentationOrderV1::BThenA,
                    digest('a'),
                    true,
                ),
                (
                    "repeat",
                    JudgePerturbationKindV1::Repeat,
                    1,
                    JudgePresentationOrderV1::AThenB,
                    digest('a'),
                    true,
                ),
            ];
            for (
                suffix,
                kind,
                replicate_index,
                order,
                prompt_sha256,
                expected_invariant_to_baseline,
            ) in variants
            {
                trials.push(JudgeAuditTrialV1 {
                    trial_id: format!("{case_id}-{suffix}"),
                    case_id: case_id.to_owned(),
                    perturbation_kind: kind,
                    variant_id: suffix.to_owned(),
                    replicate_index,
                    presentation_order: order,
                    prompt_sha256,
                    expected_invariant_to_baseline,
                });
            }
        }
        JudgeReliabilityPlanFieldsV1 {
            schema: JUDGE_RELIABILITY_PLAN_SCHEMA.to_owned(),
            purpose: JudgeAuditPurposeV1::QualifiedRun,
            audit_id: "unit-test-audit".to_owned(),
            judge,
            reference,
            perturbation_policy_sha256: digest('7'),
            scoring_rules_sha256: digest('8'),
            items,
            trials,
        }
    }

    fn committed_plan() -> Vec<u8> {
        let fields = plan_fields();
        let bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_judge_reliability_plan_commitment(&bytes).unwrap();
        let plan = JudgeReliabilityPlanV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            audit_id: fields.audit_id,
            judge: fields.judge,
            reference: fields.reference,
            perturbation_policy_sha256: fields.perturbation_policy_sha256,
            scoring_rules_sha256: fields.scoring_rules_sha256,
            items: fields.items,
            trials: fields.trials,
            plan_commitment_sha256: commitment,
        };
        serde_json::to_vec(&plan).unwrap()
    }

    fn report_fields(plan_bytes: &[u8], always_wrong: bool) -> JudgeReliabilityReportFieldsV1 {
        let plan: JudgeReliabilityPlanV1 = serde_json::from_slice(plan_bytes).unwrap();
        let mut records = Vec::new();
        for trial in &plan.trials {
            let item = plan
                .items
                .iter()
                .find(|item| item.case_id == trial.case_id)
                .unwrap();
            let verdict = if always_wrong {
                match item.reference_label {
                    JudgeReferenceLabelV1::Pass => JudgeVerdictV1::Fail,
                    JudgeReferenceLabelV1::Fail => JudgeVerdictV1::Pass,
                    JudgeReferenceLabelV1::NotEvaluable => JudgeVerdictV1::Pass,
                }
            } else {
                match item.reference_label {
                    JudgeReferenceLabelV1::Pass => JudgeVerdictV1::Pass,
                    JudgeReferenceLabelV1::Fail => JudgeVerdictV1::Fail,
                    JudgeReferenceLabelV1::NotEvaluable => JudgeVerdictV1::Pass,
                }
            };
            records.push(JudgeAuditRecordV1 {
                trial_id: trial.trial_id.clone(),
                case_id: trial.case_id.clone(),
                terminal_status: JudgeTerminalStatusV1::Completed,
                verdict: Some(verdict),
                confidence_millionths: Some(900_000),
                request_receipt_sha256: Some(digest('1')),
                response_artifact_sha256: Some(digest('2')),
                evaluator_record_sha256: digest('3'),
            });
        }
        JudgeReliabilityReportFieldsV1 {
            schema: JUDGE_RELIABILITY_REPORT_SCHEMA.to_owned(),
            purpose: JudgeAuditPurposeV1::QualifiedRun,
            plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
            plan_commitment_sha256: plan.plan_commitment_sha256,
            judge_source_revision: plan.judge.evaluator_source_revision,
            scoring_rules_sha256: plan.scoring_rules_sha256,
            records,
        }
    }

    fn committed_report(plan_bytes: &[u8], always_wrong: bool) -> Vec<u8> {
        let fields = report_fields(plan_bytes, always_wrong);
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment =
            compute_judge_reliability_report_commitment(plan_bytes, &fields_bytes).unwrap();
        let report = JudgeReliabilityReportV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            plan_sha256: fields.plan_sha256,
            plan_commitment_sha256: fields.plan_commitment_sha256,
            judge_source_revision: fields.judge_source_revision,
            scoring_rules_sha256: fields.scoring_rules_sha256,
            records: fields.records,
            report_commitment_sha256: commitment,
        };
        serde_json::to_vec(&report).unwrap()
    }

    #[test]
    fn valid_plan_binds_exact_build_revision() {
        let bytes = committed_plan();
        let result = verify_judge_reliability_plan(&bytes).unwrap();
        assert!(result.valid, "{:?}", result.issues);
        assert_eq!(result.planned_item_count, 2);
        assert_eq!(result.planned_trial_count, 8);
        assert!(!result.judge_identity_authenticated);
    }

    #[test]
    fn always_wrong_but_stable_judge_is_not_trusted() {
        let plan = committed_plan();
        let report = committed_report(&plan, true);
        let result = verify_judge_reliability_report(&plan, &report).unwrap();
        assert!(result.valid, "{:?}", result.issues);
        let scorecard = result.scorecard.unwrap();
        assert_eq!(scorecard.baseline_accuracy_millionths, Some(0));
        assert_eq!(scorecard.trusted_verdict_rate_millionths, Some(0));
        assert_eq!(scorecard.baseline_brier_score_millionths, Some(810_000));
        assert_eq!(
            scorecard.baseline_expected_calibration_error_millionths,
            Some(900_000)
        );
        assert_eq!(
            scorecard
                .perturbation_metrics
                .iter()
                .map(|metric| metric.flip_count)
                .sum::<u64>(),
            0
        );
    }

    #[test]
    fn perfect_stable_judge_has_full_trusted_verdict_rate() {
        let plan = committed_plan();
        let report = committed_report(&plan, false);
        let result = verify_judge_reliability_report(&plan, &report).unwrap();
        assert!(result.valid, "{:?}", result.issues);
        let scorecard = result.scorecard.unwrap();
        assert_eq!(scorecard.baseline_accuracy_millionths, Some(1_000_000));
        assert_eq!(scorecard.trusted_verdict_rate_millionths, Some(1_000_000));
        assert_eq!(scorecard.baseline_brier_score_millionths, Some(10_000));
        assert_eq!(
            scorecard.baseline_expected_calibration_error_millionths,
            Some(100_000)
        );
    }

    #[test]
    fn missing_baseline_confidence_keeps_calibration_null() {
        let plan = committed_plan();
        let mut fields = report_fields(&plan, false);
        for record in &mut fields.records {
            if record.trial_id.ends_with("-baseline") {
                record.confidence_millionths = None;
            }
        }
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_judge_reliability_report_commitment(&plan, &fields_bytes).unwrap();
        let report = JudgeReliabilityReportV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            plan_sha256: fields.plan_sha256,
            plan_commitment_sha256: fields.plan_commitment_sha256,
            judge_source_revision: fields.judge_source_revision,
            scoring_rules_sha256: fields.scoring_rules_sha256,
            records: fields.records,
            report_commitment_sha256: commitment,
        };
        let result =
            verify_judge_reliability_report(&plan, &serde_json::to_vec(&report).unwrap()).unwrap();
        assert!(result.valid, "{:?}", result.issues);
        let scorecard = result.scorecard.unwrap();
        assert_eq!(scorecard.baseline_confidence_recorded_count, 0);
        assert_eq!(scorecard.baseline_brier_score_millionths, None);
        assert_eq!(
            scorecard.baseline_expected_calibration_error_millionths,
            None
        );
    }

    #[test]
    fn duplicate_records_fail_closed() {
        let plan = committed_plan();
        let mut fields = report_fields(&plan, false);
        fields.records[1].trial_id = fields.records[0].trial_id.clone();
        assert!(
            compute_judge_reliability_report_commitment(
                &plan,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_perturbation_cell_is_rejected() {
        let mut fields = plan_fields();
        let mut duplicate = fields.trials[3].clone();
        duplicate.trial_id = "stable-correct-repeat-duplicate".to_owned();
        fields.trials.push(duplicate);
        assert!(
            compute_judge_reliability_plan_commitment(&serde_json::to_vec(&fields).unwrap())
                .is_err()
        );
    }

    #[test]
    fn omitted_optional_receipt_fields_are_rejected() {
        let plan = committed_plan();
        let fields = report_fields(&plan, false);
        let mut value: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&fields).unwrap()).unwrap();
        value["records"][0]
            .as_object_mut()
            .unwrap()
            .remove("confidence_millionths");
        assert!(
            compute_judge_reliability_report_commitment(
                &plan,
                &serde_json::to_vec(&value).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn missing_record_count_uses_ids_not_length_difference() {
        let plan = committed_plan();
        let mut report: JudgeReliabilityReportV1 =
            serde_json::from_slice(&committed_report(&plan, false)).unwrap();
        report.records.pop();
        report.records[0].trial_id = report.records[1].trial_id.clone();
        let result =
            verify_judge_reliability_report(&plan, &serde_json::to_vec(&report).unwrap()).unwrap();
        assert!(!result.valid);
        assert_eq!(result.missing_record_count, 2);
        assert_eq!(result.scorecard.unwrap().missing_record_count, 2);
    }

    #[test]
    fn report_is_bound_to_exact_plan_bytes() {
        let plan = committed_plan();
        let fields = report_fields(&plan, false);
        let mut changed_plan = plan.clone();
        changed_plan.push(b'\n');
        assert!(
            compute_judge_reliability_report_commitment(
                &changed_plan,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn prompt_variant_must_change_prompt_digest() {
        let mut fields = plan_fields();
        fields.trials[1].prompt_sha256 = fields.judge.prompt_sha256.clone();
        assert!(
            compute_judge_reliability_plan_commitment(&serde_json::to_vec(&fields).unwrap())
                .is_err()
        );
    }
}
