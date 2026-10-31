//! Preregistered evaluator repeatability, intervention-bias and validity audit.
//!
//! This module audits the evaluator rather than the evaluated model. A valid
//! audit receipt may demonstrate that the judge is unstable or biased; receipt
//! validity is deliberately distinct from evaluator quality. Hashes establish
//! byte binding only, not trusted timing, runner execution, adjudicator
//! independence, or truth.

use crate::BUILT_GENERATOR_SOURCE_REVISION;
use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const JUDGE_ROBUSTNESS_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_robustness_plan.v1";
pub const JUDGE_ROBUSTNESS_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_robustness_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.judge_robustness.v1\0";
const PLAN_DOMAIN: &[u8] = b"plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"report.v1\0";
const MAX_CASES: usize = 100_000;
const MAX_REPEATS: u32 = 1_000;

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
pub enum JudgeAuditFamilyV1 {
    Repeatability,
    PositionSwap,
    FormattingInvariance,
    ConfidenceStyleInvariance,
    AuthorityCueInvariance,
    SemanticSensitivityControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeExpectedRelationV1 {
    SemanticDecisionInvariant,
    SemanticDecisionMustChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrderV1 {
    CandidateAFirst,
    CandidateBFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticDecisionV1 {
    CandidateA,
    CandidateB,
    Tie,
    Abstain,
}

impl SemanticDecisionV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::CandidateA => "candidate_a",
            Self::CandidateB => "candidate_b",
            Self::Tie => "tie",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawJudgeDecisionV1 {
    Left,
    Right,
    Tie,
    Abstain,
}

impl RawJudgeDecisionV1 {
    fn normalize(self, order: CandidateOrderV1) -> SemanticDecisionV1 {
        match (self, order) {
            (Self::Left, CandidateOrderV1::CandidateAFirst)
            | (Self::Right, CandidateOrderV1::CandidateBFirst) => SemanticDecisionV1::CandidateA,
            (Self::Right, CandidateOrderV1::CandidateAFirst)
            | (Self::Left, CandidateOrderV1::CandidateBFirst) => SemanticDecisionV1::CandidateB,
            (Self::Tie, _) => SemanticDecisionV1::Tie,
            (Self::Abstain, _) => SemanticDecisionV1::Abstain,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeAuditConditionV1 {
    Baseline,
    Variant,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditCaseV1 {
    pub case_id: String,
    pub family: JudgeAuditFamilyV1,
    pub expected_relation: JudgeExpectedRelationV1,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub baseline_order: CandidateOrderV1,
    pub variant_order: CandidateOrderV1,
    /// Digest of the exact semantic enum label, e.g. SHA-256 of bytes "candidate_a".
    /// The reference decision remains hidden until the report is revealed.
    pub reference_baseline_decision_sha256: String,
    pub reference_variant_decision_sha256: String,
}

/// Typed plan payload omits only the plan commitment. Field order is part of v1.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeRobustnessPlanFieldsV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub audit_set_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub decision_policy_sha256: String,
    pub repeats_per_condition: u32,
    pub cases: Vec<JudgeAuditCaseV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeRobustnessPlanV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub audit_set_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub decision_policy_sha256: String,
    pub repeats_per_condition: u32,
    pub cases: Vec<JudgeAuditCaseV1>,
    pub plan_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditObservationV1 {
    pub condition: JudgeAuditConditionV1,
    pub repeat_index: u32,
    pub input_sha256: String,
    pub candidate_order: CandidateOrderV1,
    pub raw_decision: RawJudgeDecisionV1,
    pub judge_receipt_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeAuditCaseOutcomeV1 {
    pub case_id: String,
    pub reference_baseline_decision: SemanticDecisionV1,
    pub reference_variant_decision: SemanticDecisionV1,
    pub observations: Vec<JudgeAuditObservationV1>,
}

/// Typed report payload omits only report_commitment_sha256.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeRobustnessReportFieldsV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub decision_policy_sha256: String,
    pub blind_to_system_identity: bool,
    pub cases: Vec<JudgeAuditCaseOutcomeV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeRobustnessReportV1 {
    pub schema: String,
    pub purpose: JudgeAuditPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub decision_policy_sha256: String,
    pub blind_to_system_identity: bool,
    pub cases: Vec<JudgeAuditCaseOutcomeV1>,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeAuditFamilySummaryV1 {
    pub family: JudgeAuditFamilyV1,
    pub planned_cases: u64,
    pub observed_cases: u64,
    pub planned_observations: u64,
    pub observed_observations: u64,
    pub repeat_disagreement_pairs: u64,
    pub repeat_comparable_pairs: u64,
    pub repeat_disagreement_rate_millionths: Option<u32>,
    pub relation_violation_count: u64,
    pub relation_comparison_count: u64,
    pub relation_violation_rate_millionths: Option<u32>,
    pub gold_agreement_count: u64,
    pub gold_comparison_count: u64,
    pub gold_agreement_rate_millionths: Option<u32>,
    pub abstention_count: u64,
    pub left_slot_pair_count: u64,
    pub right_slot_pair_count: u64,
    pub slot_following_pair_count: u64,
    pub slot_comparison_count: u64,
    pub slot_following_rate_millionths: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeRobustnessScorecardV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub report_sha256: String,
    pub report_commitment_sha256: String,
    pub planned_case_count: u64,
    pub observed_case_count: u64,
    pub missing_case_count: u64,
    pub planned_observation_count: u64,
    pub observed_observation_count: u64,
    pub missing_observation_count: u64,
    pub blind_to_system_identity: bool,
    pub family_summaries: Vec<JudgeAuditFamilySummaryV1>,
    pub fixed_audit_only: bool,
    pub generalized_accuracy_estimated: bool,
    pub runner_execution_proven: bool,
    pub external_anchor_required: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeRobustnessPlanVerificationV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_expected: String,
    pub plan_commitment_recomputed: String,
    pub plan_commitment_matches: bool,
    pub structural_valid: bool,
    pub valid: bool,
    pub audit_set_id: String,
    pub planned_case_count: u64,
    pub planned_observation_count: u64,
    pub issues: Vec<String>,
    pub scoring_eligible_declared: bool,
    pub external_anchor_required: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeRobustnessReportVerificationV1 {
    pub schema: String,
    pub valid: bool,
    pub plan_binding_valid: bool,
    pub report_commitment_matches: bool,
    pub planned_case_count: u64,
    pub observed_case_count: u64,
    pub missing_case_count: u64,
    pub planned_observation_count: u64,
    pub observed_observation_count: u64,
    pub missing_observation_count: u64,
    pub issues: Vec<String>,
    pub scorecard: Option<JudgeRobustnessScorecardV1>,
}

#[derive(Debug, Clone, Serialize)]
struct ReportCommitmentPayloadV1<'a> {
    schema: &'a str,
    purpose: JudgeAuditPurposeV1,
    plan_sha256: &'a str,
    plan_commitment_sha256: &'a str,
    judge_source_revision: &'a str,
    judge_prompt_sha256: &'a str,
    rubric_sha256: &'a str,
    decision_policy_sha256: &'a str,
    blind_to_system_identity: bool,
    cases: &'a [JudgeAuditCaseOutcomeV1],
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
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

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len =
        u64::try_from(domain.len()).map_err(|_| "hash domain exceeds u64 length".to_owned())?;
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| "hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn rate_millionths(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        return None;
    }
    let numerator = u128::from(numerator) * 1_000_000 + u128::from(denominator) / 2;
    Some((numerator / u128::from(denominator)) as u32)
}

fn hash_reference_decision(decision: SemanticDecisionV1) -> String {
    sha256_hex(decision.as_str().as_bytes())
}

fn expected_relation(family: JudgeAuditFamilyV1) -> JudgeExpectedRelationV1 {
    match family {
        JudgeAuditFamilyV1::SemanticSensitivityControl => {
            JudgeExpectedRelationV1::SemanticDecisionMustChange
        }
        _ => JudgeExpectedRelationV1::SemanticDecisionInvariant,
    }
}

fn validate_plan_fields(fields: &JudgeRobustnessPlanFieldsV1) -> Vec<String> {
    let mut issues = Vec::new();
    if fields.schema != JUDGE_ROBUSTNESS_PLAN_SCHEMA {
        issues.push("unsupported judge robustness plan schema".to_owned());
    }
    if fields.audit_set_id.trim().is_empty() || fields.audit_set_id.len() > 128 {
        issues.push("audit_set_id must be non-empty and at most 128 bytes".to_owned());
    }
    if fields.cases.is_empty() || fields.cases.len() > MAX_CASES {
        issues.push(format!("cases must contain 1..={MAX_CASES} items"));
    }
    if fields.repeats_per_condition < 2 || fields.repeats_per_condition > MAX_REPEATS {
        issues.push(format!("repeats_per_condition must be 2..={MAX_REPEATS}"));
    }

    let allow_zero = !fields.purpose.qualified();
    for (name, value) in [
        ("judge_prompt_sha256", fields.judge_prompt_sha256.as_str()),
        ("rubric_sha256", fields.rubric_sha256.as_str()),
        (
            "decision_policy_sha256",
            fields.decision_policy_sha256.as_str(),
        ),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{name} must be a valid lower-case SHA-256 digest"));
        }
    }
    if fields.purpose.qualified() {
        if !valid_revision(&fields.judge_source_revision) {
            issues.push(
                "qualified audit requires a non-zero lower-case judge revision SHA".to_owned(),
            );
        } else if fields.judge_source_revision != BUILT_GENERATOR_SOURCE_REVISION {
            issues
                .push("judge source revision differs from the inspected build revision".to_owned());
        }
    } else if fields.judge_source_revision.trim().is_empty() {
        issues.push("example-only judge source revision marker must be non-empty".to_owned());
    }

    let mut seen = BTreeSet::new();
    for case in &fields.cases {
        if case.case_id.trim().is_empty() || case.case_id.len() > 128 {
            issues.push("case_id must be non-empty and at most 128 bytes".to_owned());
        }
        if !seen.insert(case.case_id.as_str()) {
            issues.push(format!("duplicate judge audit case_id {}", case.case_id));
        }
        if case.expected_relation != expected_relation(case.family) {
            issues.push(format!(
                "case {} has relation incompatible with family {:?}",
                case.case_id, case.family
            ));
        }
        for (name, value) in [
            ("baseline_input_sha256", case.baseline_input_sha256.as_str()),
            ("variant_input_sha256", case.variant_input_sha256.as_str()),
            (
                "reference_baseline_decision_sha256",
                case.reference_baseline_decision_sha256.as_str(),
            ),
            (
                "reference_variant_decision_sha256",
                case.reference_variant_decision_sha256.as_str(),
            ),
        ] {
            if !valid_digest(value, allow_zero) {
                issues.push(format!(
                    "case {} {name} is not an eligible digest",
                    case.case_id
                ));
            }
        }

        match case.family {
            JudgeAuditFamilyV1::Repeatability => {
                if fields.purpose.qualified()
                    && (case.baseline_input_sha256 != case.variant_input_sha256
                        || case.baseline_order != case.variant_order
                        || case.reference_baseline_decision_sha256
                            != case.reference_variant_decision_sha256)
                {
                    issues.push(format!(
                        "repeatability case {} must repeat identical input, order and reference",
                        case.case_id
                    ));
                }
            }
            JudgeAuditFamilyV1::PositionSwap => {
                if case.baseline_order == case.variant_order {
                    issues.push(format!(
                        "position-swap case {} must swap candidate order",
                        case.case_id
                    ));
                }
                if fields.purpose.qualified()
                    && (case.baseline_input_sha256 == case.variant_input_sha256
                        || case.reference_baseline_decision_sha256
                            != case.reference_variant_decision_sha256)
                {
                    issues.push(format!(
                        "position-swap case {} must change input bytes but preserve reference decision",
                        case.case_id
                    ));
                }
            }
            JudgeAuditFamilyV1::FormattingInvariance
            | JudgeAuditFamilyV1::ConfidenceStyleInvariance
            | JudgeAuditFamilyV1::AuthorityCueInvariance => {
                if case.baseline_order != case.variant_order {
                    issues.push(format!(
                        "invariance case {} must keep candidate order unchanged",
                        case.case_id
                    ));
                }
                if fields.purpose.qualified()
                    && (case.baseline_input_sha256 == case.variant_input_sha256
                        || case.reference_baseline_decision_sha256
                            != case.reference_variant_decision_sha256)
                {
                    issues.push(format!(
                        "invariance case {} must change input bytes but preserve reference decision",
                        case.case_id
                    ));
                }
            }
            JudgeAuditFamilyV1::SemanticSensitivityControl => {
                if case.baseline_order != case.variant_order {
                    issues.push(format!(
                        "semantic-sensitivity case {} must keep candidate order unchanged",
                        case.case_id
                    ));
                }
                if fields.purpose.qualified()
                    && (case.baseline_input_sha256 == case.variant_input_sha256
                        || case.reference_baseline_decision_sha256
                            == case.reference_variant_decision_sha256)
                {
                    issues.push(format!(
                        "semantic-sensitivity case {} must change input and reference decision",
                        case.case_id
                    ));
                }
            }
        }
    }
    issues
}

fn fields_from_plan(plan: &JudgeRobustnessPlanV1) -> JudgeRobustnessPlanFieldsV1 {
    JudgeRobustnessPlanFieldsV1 {
        schema: plan.schema.clone(),
        purpose: plan.purpose,
        audit_set_id: plan.audit_set_id.clone(),
        judge_source_revision: plan.judge_source_revision.clone(),
        judge_prompt_sha256: plan.judge_prompt_sha256.clone(),
        rubric_sha256: plan.rubric_sha256.clone(),
        decision_policy_sha256: plan.decision_policy_sha256.clone(),
        repeats_per_condition: plan.repeats_per_condition,
        cases: plan.cases.clone(),
    }
}

fn validate_plan_shape(bytes: &[u8], full: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("judge robustness plan JSON malformed: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "judge robustness plan must be a JSON object".to_owned())?;
    let mut required = vec![
        "schema",
        "purpose",
        "audit_set_id",
        "judge_source_revision",
        "judge_prompt_sha256",
        "rubric_sha256",
        "decision_policy_sha256",
        "repeats_per_condition",
        "cases",
    ];
    if full {
        required.push("plan_commitment_sha256");
    }
    for key in required {
        if !object.contains_key(key) {
            return Err(format!("judge robustness plan missing field {key}"));
        }
    }
    Ok(())
}

fn validate_report_shape(bytes: &[u8], full: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("judge robustness report JSON malformed: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "judge robustness report must be a JSON object".to_owned())?;
    let mut required = vec![
        "schema",
        "purpose",
        "plan_sha256",
        "plan_commitment_sha256",
        "judge_source_revision",
        "judge_prompt_sha256",
        "rubric_sha256",
        "decision_policy_sha256",
        "blind_to_system_identity",
        "cases",
    ];
    if full {
        required.push("report_commitment_sha256");
    }
    for key in required {
        if !object.contains_key(key) {
            return Err(format!("judge robustness report missing field {key}"));
        }
    }
    let cases = value["cases"]
        .as_array()
        .ok_or_else(|| "judge robustness report cases must be an array".to_owned())?;
    for (index, case) in cases.iter().enumerate() {
        let case_object = case
            .as_object()
            .ok_or_else(|| format!("judge outcome {index} must be an object"))?;
        for key in [
            "case_id",
            "reference_baseline_decision",
            "reference_variant_decision",
            "observations",
        ] {
            if !case_object.contains_key(key) {
                return Err(format!("judge outcome {index} missing field {key}"));
            }
        }
        let observations = case["observations"]
            .as_array()
            .ok_or_else(|| format!("judge outcome {index} observations must be an array"))?;
        for (observation_index, observation) in observations.iter().enumerate() {
            let observation_object = observation.as_object().ok_or_else(|| {
                format!("judge observation {index}/{observation_index} must be an object")
            })?;
            for key in [
                "condition",
                "repeat_index",
                "input_sha256",
                "candidate_order",
                "raw_decision",
                "judge_receipt_sha256",
            ] {
                if !observation_object.contains_key(key) {
                    return Err(format!(
                        "judge observation {index}/{observation_index} missing field {key}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Compute a typed plan commitment from a fields file without the commitment.
pub fn compute_judge_robustness_plan_commitment(bytes: &[u8]) -> Result<String, String> {
    validate_plan_shape(bytes, false)?;
    let fields: JudgeRobustnessPlanFieldsV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("judge robustness plan fields invalid: {error}"))?;
    let issues = validate_plan_fields(&fields);
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize judge plan fields: {error}"))?;
    digest_domain_framed(PLAN_DOMAIN, &payload)
}

/// Verify plan structure, typed commitment and exact evaluator build revision.
pub fn verify_judge_robustness_plan(
    bytes: &[u8],
) -> Result<JudgeRobustnessPlanVerificationV1, String> {
    validate_plan_shape(bytes, true)?;
    let plan: JudgeRobustnessPlanV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("judge robustness plan invalid: {error}"))?;
    let fields = fields_from_plan(&plan);
    let mut issues = validate_plan_fields(&fields);
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize judge plan payload: {error}"))?;
    let recomputed = digest_domain_framed(PLAN_DOMAIN, &payload)?;
    let matches = plan.plan_commitment_sha256 == recomputed
        && valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified());
    if !matches {
        issues.push("judge plan commitment mismatch".to_owned());
    }
    let structural_valid = validate_plan_fields(&fields).is_empty();
    let planned_observations = u64::try_from(plan.cases.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(plan.repeats_per_condition))
        .saturating_mul(2);
    Ok(JudgeRobustnessPlanVerificationV1 {
        schema: "symthaea.science_bench_refresh_judge_plan_verification.v1".to_owned(),
        plan_sha256: sha256_hex(bytes),
        plan_commitment_expected: plan.plan_commitment_sha256.clone(),
        plan_commitment_recomputed: recomputed,
        plan_commitment_matches: matches,
        structural_valid,
        valid: structural_valid && matches && plan.purpose.qualified(),
        audit_set_id: plan.audit_set_id,
        planned_case_count: plan.cases.len() as u64,
        planned_observation_count: planned_observations,
        issues,
        scoring_eligible_declared: plan.purpose.qualified(),
        external_anchor_required: true,
    })
}

fn report_fields_from_report(report: &JudgeRobustnessReportV1) -> JudgeRobustnessReportFieldsV1 {
    JudgeRobustnessReportFieldsV1 {
        schema: report.schema.clone(),
        purpose: report.purpose,
        plan_sha256: report.plan_sha256.clone(),
        plan_commitment_sha256: report.plan_commitment_sha256.clone(),
        judge_source_revision: report.judge_source_revision.clone(),
        judge_prompt_sha256: report.judge_prompt_sha256.clone(),
        rubric_sha256: report.rubric_sha256.clone(),
        decision_policy_sha256: report.decision_policy_sha256.clone(),
        blind_to_system_identity: report.blind_to_system_identity,
        cases: report.cases.clone(),
    }
}

fn validate_report_fields(
    plan_bytes: &[u8],
    fields: &JudgeRobustnessReportFieldsV1,
) -> Result<(JudgeRobustnessPlanV1, Vec<String>), String> {
    let plan_verification = verify_judge_robustness_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err(
            "cannot verify judge report against an invalid or example-only plan".to_owned(),
        );
    }
    let plan: JudgeRobustnessPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("judge robustness plan invalid: {error}"))?;
    let mut issues = Vec::new();
    if fields.schema != JUDGE_ROBUSTNESS_REPORT_SCHEMA {
        issues.push("unsupported judge robustness report schema".to_owned());
    }
    if fields.purpose != plan.purpose {
        issues.push("judge report purpose differs from plan".to_owned());
    }
    if fields.plan_sha256 != sha256_hex(plan_bytes) {
        issues.push("judge report does not bind exact plan bytes".to_owned());
    }
    if fields.plan_commitment_sha256 != plan.plan_commitment_sha256 {
        issues.push("judge report plan commitment differs from plan".to_owned());
    }
    if fields.judge_source_revision != plan.judge_source_revision {
        issues.push("judge report source revision differs from plan".to_owned());
    }
    if fields.judge_prompt_sha256 != plan.judge_prompt_sha256
        || fields.rubric_sha256 != plan.rubric_sha256
        || fields.decision_policy_sha256 != plan.decision_policy_sha256
    {
        issues.push("judge report prompt/rubric/decision policy differs from plan".to_owned());
    }
    if fields.purpose.qualified() && !fields.blind_to_system_identity {
        issues.push("qualified judge audit must declare identity-blinded evaluation".to_owned());
    }
    if fields.cases.len() != plan.cases.len() {
        issues.push(format!(
            "judge report case count {} differs from plan count {}",
            fields.cases.len(),
            plan.cases.len()
        ));
    }
    let plans: BTreeMap<&str, &JudgeAuditCaseV1> = plan
        .cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    let mut seen_cases = BTreeSet::new();
    for outcome in &fields.cases {
        if !seen_cases.insert(outcome.case_id.as_str()) {
            issues.push(format!("duplicate judge audit outcome {}", outcome.case_id));
            continue;
        }
        let Some(case) = plans.get(outcome.case_id.as_str()) else {
            issues.push(format!("unplanned judge audit outcome {}", outcome.case_id));
            continue;
        };
        if hash_reference_decision(outcome.reference_baseline_decision)
            != case.reference_baseline_decision_sha256
        {
            issues.push(format!(
                "case {} baseline reference decision does not match preregistered digest",
                case.case_id
            ));
        }
        if hash_reference_decision(outcome.reference_variant_decision)
            != case.reference_variant_decision_sha256
        {
            issues.push(format!(
                "case {} variant reference decision does not match preregistered digest",
                case.case_id
            ));
        }
        match case.expected_relation {
            JudgeExpectedRelationV1::SemanticDecisionInvariant => {
                if outcome.reference_baseline_decision != outcome.reference_variant_decision {
                    issues.push(format!(
                        "case {} violates its invariant reference contract",
                        case.case_id
                    ));
                }
            }
            JudgeExpectedRelationV1::SemanticDecisionMustChange => {
                if outcome.reference_baseline_decision == outcome.reference_variant_decision {
                    issues.push(format!(
                        "case {} violates its semantic-change reference contract",
                        case.case_id
                    ));
                }
            }
        }

        let expected_observation_count =
            usize::try_from(plan.repeats_per_condition).unwrap_or(usize::MAX) * 2;
        if outcome.observations.len() != expected_observation_count {
            issues.push(format!(
                "case {} has {} observations; expected {}",
                case.case_id,
                outcome.observations.len(),
                expected_observation_count
            ));
        }
        let mut seen_observations = BTreeSet::new();
        for observation in &outcome.observations {
            if observation.repeat_index >= plan.repeats_per_condition {
                issues.push(format!(
                    "case {} has out-of-range repeat index",
                    case.case_id
                ));
            }
            let key = (observation.condition, observation.repeat_index);
            if !seen_observations.insert(key) {
                issues.push(format!(
                    "case {} has duplicate condition/repeat observation",
                    case.case_id
                ));
            }
            let (expected_input, expected_order) = match observation.condition {
                JudgeAuditConditionV1::Baseline => {
                    (&case.baseline_input_sha256, case.baseline_order)
                }
                JudgeAuditConditionV1::Variant => (&case.variant_input_sha256, case.variant_order),
            };
            if observation.input_sha256 != *expected_input {
                issues.push(format!(
                    "case {} observation input digest does not match its condition",
                    case.case_id
                ));
            }
            if observation.candidate_order != expected_order {
                issues.push(format!(
                    "case {} observation candidate order does not match its condition",
                    case.case_id
                ));
            }
            if !valid_digest(&observation.input_sha256, !fields.purpose.qualified()) {
                issues.push(format!(
                    "case {} has invalid observation input digest",
                    case.case_id
                ));
            }
            if !valid_digest(
                &observation.judge_receipt_sha256,
                !fields.purpose.qualified(),
            ) {
                issues.push(format!(
                    "case {} has invalid judge receipt digest",
                    case.case_id
                ));
            }
        }
        for condition in [
            JudgeAuditConditionV1::Baseline,
            JudgeAuditConditionV1::Variant,
        ] {
            for repeat_index in 0..plan.repeats_per_condition {
                if !seen_observations.contains(&(condition, repeat_index)) {
                    issues.push(format!(
                        "case {} missing {:?} repeat {}",
                        case.case_id, condition, repeat_index
                    ));
                }
            }
        }
    }
    for case in &plan.cases {
        if !seen_cases.contains(case.case_id.as_str()) {
            issues.push(format!("missing judge audit outcome for {}", case.case_id));
        }
    }
    // Avoid imposing any quality threshold here: an internally sound report may
    // document the judge's failure to comply with the expected relation.
    Ok((plan, issues))
}

fn report_commitment(fields: &JudgeRobustnessReportFieldsV1) -> Result<String, String> {
    let payload = ReportCommitmentPayloadV1 {
        schema: &fields.schema,
        purpose: fields.purpose,
        plan_sha256: &fields.plan_sha256,
        plan_commitment_sha256: &fields.plan_commitment_sha256,
        judge_source_revision: &fields.judge_source_revision,
        judge_prompt_sha256: &fields.judge_prompt_sha256,
        rubric_sha256: &fields.rubric_sha256,
        decision_policy_sha256: &fields.decision_policy_sha256,
        blind_to_system_identity: fields.blind_to_system_identity,
        cases: &fields.cases,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize judge report commitment: {error}"))?;
    digest_domain_framed(REPORT_DOMAIN, &bytes)
}

/// Compute a typed report commitment after complete case/observation validation.
pub fn compute_judge_robustness_report_commitment(
    plan_bytes: &[u8],
    fields_bytes: &[u8],
) -> Result<String, String> {
    validate_report_shape(fields_bytes, false)?;
    let fields: JudgeRobustnessReportFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("judge report fields invalid: {error}"))?;
    let (_, issues) = validate_report_fields(plan_bytes, &fields)?;
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    report_commitment(&fields)
}

#[derive(Default)]
struct FamilyAccumulator {
    planned_cases: u64,
    observed_cases: u64,
    planned_observations: u64,
    observed_observations: u64,
    repeat_disagreement_pairs: u64,
    repeat_comparable_pairs: u64,
    relation_violation_count: u64,
    relation_comparison_count: u64,
    gold_agreement_count: u64,
    gold_comparison_count: u64,
    abstention_count: u64,
    left_slot_pair_count: u64,
    right_slot_pair_count: u64,
    slot_following_pair_count: u64,
    slot_comparison_count: u64,
}

fn normalized_by_condition(
    outcome: &JudgeAuditCaseOutcomeV1,
) -> BTreeMap<(JudgeAuditConditionV1, u32), SemanticDecisionV1> {
    outcome
        .observations
        .iter()
        .map(|observation| {
            (
                (observation.condition, observation.repeat_index),
                observation
                    .raw_decision
                    .normalize(observation.candidate_order),
            )
        })
        .collect()
}

fn family_summaries(
    plan: &JudgeRobustnessPlanV1,
    report: &JudgeRobustnessReportV1,
) -> Vec<JudgeAuditFamilySummaryV1> {
    let cases_by_id: BTreeMap<&str, &JudgeAuditCaseOutcomeV1> = report
        .cases
        .iter()
        .map(|outcome| (outcome.case_id.as_str(), outcome))
        .collect();
    let mut accumulators = BTreeMap::<JudgeAuditFamilyV1, FamilyAccumulator>::new();

    for case in &plan.cases {
        let accumulator = accumulators.entry(case.family).or_default();
        accumulator.planned_cases += 1;
        accumulator.planned_observations = accumulator
            .planned_observations
            .saturating_add(u64::from(plan.repeats_per_condition).saturating_mul(2));
        let Some(outcome) = cases_by_id.get(case.case_id.as_str()) else {
            continue;
        };
        accumulator.observed_cases += 1;
        accumulator.observed_observations = accumulator
            .observed_observations
            .saturating_add(outcome.observations.len() as u64);
        let normalized = normalized_by_condition(outcome);

        let decision_groups: Vec<Vec<SemanticDecisionV1>> =
            if case.family == JudgeAuditFamilyV1::Repeatability {
                // Both conditions are byte-identical for this family, so pool every
                // repeat into one same-input group and count cross-condition pairs too.
                vec![
                    outcome
                        .observations
                        .iter()
                        .map(|observation| {
                            observation
                                .raw_decision
                                .normalize(observation.candidate_order)
                        })
                        .collect::<Vec<_>>(),
                ]
            } else {
                [
                    JudgeAuditConditionV1::Baseline,
                    JudgeAuditConditionV1::Variant,
                ]
                .into_iter()
                .map(|condition| {
                    (0..plan.repeats_per_condition)
                        .filter_map(|repeat| normalized.get(&(condition, repeat)).copied())
                        .collect::<Vec<_>>()
                })
                .collect()
            };
        for decisions in decision_groups {
            for left_index in 0..decisions.len() {
                for right_index in (left_index + 1)..decisions.len() {
                    accumulator.repeat_comparable_pairs += 1;
                    if decisions[left_index] != decisions[right_index] {
                        accumulator.repeat_disagreement_pairs += 1;
                    }
                }
            }
        }

        for repeat in 0..plan.repeats_per_condition {
            let baseline = normalized.get(&(JudgeAuditConditionV1::Baseline, repeat));
            let variant = normalized.get(&(JudgeAuditConditionV1::Variant, repeat));
            let (Some(baseline), Some(variant)) = (baseline, variant) else {
                continue;
            };
            accumulator.relation_comparison_count += 1;
            let violated = match case.expected_relation {
                JudgeExpectedRelationV1::SemanticDecisionInvariant => baseline != variant,
                JudgeExpectedRelationV1::SemanticDecisionMustChange => baseline == variant,
            };
            if violated {
                accumulator.relation_violation_count += 1;
            }

            if case.family == JudgeAuditFamilyV1::PositionSwap {
                let baseline_raw = outcome.observations.iter().find(|o| {
                    o.condition == JudgeAuditConditionV1::Baseline && o.repeat_index == repeat
                });
                let variant_raw = outcome.observations.iter().find(|o| {
                    o.condition == JudgeAuditConditionV1::Variant && o.repeat_index == repeat
                });
                if let (Some(b), Some(v)) = (baseline_raw, variant_raw) {
                    let b_slot = match b.raw_decision {
                        RawJudgeDecisionV1::Left => Some(true),
                        RawJudgeDecisionV1::Right => Some(false),
                        RawJudgeDecisionV1::Tie | RawJudgeDecisionV1::Abstain => None,
                    };
                    let v_slot = match v.raw_decision {
                        RawJudgeDecisionV1::Left => Some(true),
                        RawJudgeDecisionV1::Right => Some(false),
                        RawJudgeDecisionV1::Tie | RawJudgeDecisionV1::Abstain => None,
                    };
                    if let (Some(b_left), Some(v_left)) = (b_slot, v_slot) {
                        accumulator.slot_comparison_count += 1;
                        if b_left == v_left {
                            accumulator.slot_following_pair_count += 1;
                            if b_left {
                                accumulator.left_slot_pair_count += 1;
                            } else {
                                accumulator.right_slot_pair_count += 1;
                            }
                        }
                    }
                }
            }
        }
        for observation in &outcome.observations {
            if observation.raw_decision == RawJudgeDecisionV1::Abstain {
                accumulator.abstention_count += 1;
            }
            let normalized = observation
                .raw_decision
                .normalize(observation.candidate_order);
            let reference = match observation.condition {
                JudgeAuditConditionV1::Baseline => outcome.reference_baseline_decision,
                JudgeAuditConditionV1::Variant => outcome.reference_variant_decision,
            };
            accumulator.gold_comparison_count += 1;
            if normalized == reference {
                accumulator.gold_agreement_count += 1;
            }
        }
    }

    accumulators
        .into_iter()
        .map(|(family, counts)| JudgeAuditFamilySummaryV1 {
            family,
            planned_cases: counts.planned_cases,
            observed_cases: counts.observed_cases,
            planned_observations: counts.planned_observations,
            observed_observations: counts.observed_observations,
            repeat_disagreement_pairs: counts.repeat_disagreement_pairs,
            repeat_comparable_pairs: counts.repeat_comparable_pairs,
            repeat_disagreement_rate_millionths: rate_millionths(
                counts.repeat_disagreement_pairs,
                counts.repeat_comparable_pairs,
            ),
            relation_violation_count: counts.relation_violation_count,
            relation_comparison_count: counts.relation_comparison_count,
            relation_violation_rate_millionths: rate_millionths(
                counts.relation_violation_count,
                counts.relation_comparison_count,
            ),
            gold_agreement_count: counts.gold_agreement_count,
            gold_comparison_count: counts.gold_comparison_count,
            gold_agreement_rate_millionths: rate_millionths(
                counts.gold_agreement_count,
                counts.gold_comparison_count,
            ),
            abstention_count: counts.abstention_count,
            left_slot_pair_count: counts.left_slot_pair_count,
            right_slot_pair_count: counts.right_slot_pair_count,
            slot_following_pair_count: counts.slot_following_pair_count,
            slot_comparison_count: counts.slot_comparison_count,
            slot_following_rate_millionths: rate_millionths(
                counts.slot_following_pair_count,
                counts.slot_comparison_count,
            ),
        })
        .collect()
}

/// Verify the full report. A valid report can have high bias/instability rates.
pub fn verify_judge_robustness_report(
    plan_bytes: &[u8],
    report_bytes: &[u8],
) -> Result<JudgeRobustnessReportVerificationV1, String> {
    validate_report_shape(report_bytes, true)?;
    let plan_verification = verify_judge_robustness_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err("cannot verify judge report against invalid or example-only plan".to_owned());
    }
    let report: JudgeRobustnessReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("judge robustness report invalid: {error}"))?;
    let fields = report_fields_from_report(&report);
    let (plan, mut issues) = validate_report_fields(plan_bytes, &fields)?;
    let commitment = report_commitment(&fields)?;
    let commitment_matches = report.report_commitment_sha256 == commitment
        && valid_digest(
            &report.report_commitment_sha256,
            !report.purpose.qualified(),
        );
    if !commitment_matches {
        issues.push("judge report commitment mismatch".to_owned());
    }
    let valid = issues.is_empty() && commitment_matches && report.purpose.qualified();
    let outcomes_by_id: BTreeMap<&str, &JudgeAuditCaseOutcomeV1> = report
        .cases
        .iter()
        .map(|outcome| (outcome.case_id.as_str(), outcome))
        .collect();
    let missing_case_count = plan
        .cases
        .iter()
        .filter(|case| !outcomes_by_id.contains_key(case.case_id.as_str()))
        .count() as u64;
    let planned_observation_count = u64::try_from(plan.cases.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(plan.repeats_per_condition))
        .saturating_mul(2);
    let observed_observation_count = report
        .cases
        .iter()
        .map(|outcome| outcome.observations.len() as u64)
        .fold(0_u64, u64::saturating_add);
    let missing_observation_count = planned_observation_count
        .saturating_sub(observed_observation_count.min(planned_observation_count));
    let scorecard = if valid {
        Some(JudgeRobustnessScorecardV1 {
            schema: "symthaea.science_bench_refresh_judge_robustness_scorecard.v1".to_owned(),
            plan_sha256: sha256_hex(plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            report_sha256: sha256_hex(report_bytes),
            report_commitment_sha256: report.report_commitment_sha256.clone(),
            planned_case_count: plan.cases.len() as u64,
            observed_case_count: report.cases.len() as u64,
            missing_case_count,
            planned_observation_count,
            observed_observation_count,
            missing_observation_count,
            blind_to_system_identity: report.blind_to_system_identity,
            family_summaries: family_summaries(&plan, &report),
            fixed_audit_only: true,
            generalized_accuracy_estimated: false,
            runner_execution_proven: false,
            external_anchor_required: true,
            claim_boundary: "Judge stability, intervention consistency and agreement with reference labels are distinct; a digest does not prove execution, independent adjudication, truth or generalization.".to_owned(),
        })
    } else {
        None
    };
    let plan_binding_valid = fields.plan_sha256 == sha256_hex(plan_bytes)
        && fields.plan_commitment_sha256 == plan.plan_commitment_sha256
        && fields.judge_source_revision == plan.judge_source_revision
        && fields.judge_prompt_sha256 == plan.judge_prompt_sha256
        && fields.rubric_sha256 == plan.rubric_sha256
        && fields.decision_policy_sha256 == plan.decision_policy_sha256;
    Ok(JudgeRobustnessReportVerificationV1 {
        schema: "symthaea.science_bench_refresh_judge_report_verification.v1".to_owned(),
        valid,
        plan_binding_valid,
        report_commitment_matches: commitment_matches,
        planned_case_count: plan.cases.len() as u64,
        observed_case_count: report.cases.len() as u64,
        missing_case_count,
        planned_observation_count,
        observed_observation_count,
        missing_observation_count,
        issues,
        scorecard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn test_cases() -> Vec<JudgeAuditCaseV1> {
        let make = |case_id: &str,
                    family: JudgeAuditFamilyV1,
                    baseline_input: &str,
                    variant_input: &str,
                    baseline_order: CandidateOrderV1,
                    variant_order: CandidateOrderV1,
                    baseline_gold: SemanticDecisionV1,
                    variant_gold: SemanticDecisionV1| {
            JudgeAuditCaseV1 {
                case_id: case_id.to_owned(),
                family,
                expected_relation: expected_relation(family),
                baseline_input_sha256: digest(baseline_input),
                variant_input_sha256: digest(variant_input),
                baseline_order,
                variant_order,
                reference_baseline_decision_sha256: digest(baseline_gold.as_str()),
                reference_variant_decision_sha256: digest(variant_gold.as_str()),
            }
        };
        vec![
            make(
                "repeat-1",
                JudgeAuditFamilyV1::Repeatability,
                "same-input",
                "same-input",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateAFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateA,
            ),
            make(
                "position-1",
                JudgeAuditFamilyV1::PositionSwap,
                "a-first",
                "b-first",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateBFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateA,
            ),
            make(
                "format-1",
                JudgeAuditFamilyV1::FormattingInvariance,
                "plain-format",
                "json-format",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateAFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateA,
            ),
            make(
                "confidence-1",
                JudgeAuditFamilyV1::ConfidenceStyleInvariance,
                "neutral-claim",
                "assertive-claim",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateAFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateA,
            ),
            make(
                "authority-1",
                JudgeAuditFamilyV1::AuthorityCueInvariance,
                "plain-source",
                "fabricated-authority-cue",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateAFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateA,
            ),
            make(
                "sensitivity-1",
                JudgeAuditFamilyV1::SemanticSensitivityControl,
                "evidence-supports-a",
                "evidence-supports-b",
                CandidateOrderV1::CandidateAFirst,
                CandidateOrderV1::CandidateAFirst,
                SemanticDecisionV1::CandidateA,
                SemanticDecisionV1::CandidateB,
            ),
        ]
    }

    fn plan_fields() -> JudgeRobustnessPlanFieldsV1 {
        JudgeRobustnessPlanFieldsV1 {
            schema: JUDGE_ROBUSTNESS_PLAN_SCHEMA.to_owned(),
            purpose: JudgeAuditPurposeV1::QualifiedRun,
            audit_set_id: "judge-audit-unit-test-v1".to_owned(),
            judge_source_revision: BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            judge_prompt_sha256: digest("judge-prompt"),
            rubric_sha256: digest("rubric"),
            decision_policy_sha256: digest("decision-policy"),
            repeats_per_condition: 3,
            cases: test_cases(),
        }
    }

    fn committed_plan_bytes() -> Vec<u8> {
        let fields = plan_fields();
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_judge_robustness_plan_commitment(&fields_bytes).unwrap();
        let plan = JudgeRobustnessPlanV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            audit_set_id: fields.audit_set_id,
            judge_source_revision: fields.judge_source_revision,
            judge_prompt_sha256: fields.judge_prompt_sha256,
            rubric_sha256: fields.rubric_sha256,
            decision_policy_sha256: fields.decision_policy_sha256,
            repeats_per_condition: fields.repeats_per_condition,
            cases: fields.cases,
            plan_commitment_sha256: commitment,
        };
        serde_json::to_vec(&plan).unwrap()
    }

    fn report_fields(plan_bytes: &[u8]) -> JudgeRobustnessReportFieldsV1 {
        let plan: JudgeRobustnessPlanV1 = serde_json::from_slice(plan_bytes).unwrap();
        let cases = plan
            .cases
            .iter()
            .map(|case| {
                let reference_baseline_decision = SemanticDecisionV1::CandidateA;
                let reference_variant_decision =
                    if case.family == JudgeAuditFamilyV1::SemanticSensitivityControl {
                        SemanticDecisionV1::CandidateB
                    } else {
                        SemanticDecisionV1::CandidateA
                    };
                let observations = [
                    JudgeAuditConditionV1::Baseline,
                    JudgeAuditConditionV1::Variant,
                ]
                .into_iter()
                .flat_map(|condition| {
                    (0..plan.repeats_per_condition).map(move |repeat_index| {
                        let is_variant = condition == JudgeAuditConditionV1::Variant;
                        let order = if is_variant {
                            case.variant_order
                        } else {
                            case.baseline_order
                        };
                        // The synthetic oracle is semantically consistent for every
                        // invariance case and changes its winner for the sensitivity control.
                        let semantic_winner = if is_variant {
                            reference_variant_decision
                        } else {
                            reference_baseline_decision
                        };
                        let raw_decision = match (semantic_winner, order) {
                            (
                                SemanticDecisionV1::CandidateA,
                                CandidateOrderV1::CandidateAFirst,
                            )
                            | (
                                SemanticDecisionV1::CandidateB,
                                CandidateOrderV1::CandidateBFirst,
                            ) => RawJudgeDecisionV1::Left,
                            (
                                SemanticDecisionV1::CandidateB,
                                CandidateOrderV1::CandidateAFirst,
                            )
                            | (
                                SemanticDecisionV1::CandidateA,
                                CandidateOrderV1::CandidateBFirst,
                            ) => RawJudgeDecisionV1::Right,
                            (SemanticDecisionV1::Tie, _) => RawJudgeDecisionV1::Tie,
                            (SemanticDecisionV1::Abstain, _) => RawJudgeDecisionV1::Abstain,
                        };
                        JudgeAuditObservationV1 {
                            condition,
                            repeat_index,
                            input_sha256: if is_variant {
                                case.variant_input_sha256.clone()
                            } else {
                                case.baseline_input_sha256.clone()
                            },
                            candidate_order: order,
                            raw_decision,
                            judge_receipt_sha256: digest(&format!(
                                "{}:{:?}:{}",
                                case.case_id, condition, repeat_index
                            )),
                        }
                    })
                })
                .collect();
                JudgeAuditCaseOutcomeV1 {
                    case_id: case.case_id.clone(),
                    reference_baseline_decision,
                    reference_variant_decision,
                    observations,
                }
            })
            .collect();
        JudgeRobustnessReportFieldsV1 {
            schema: JUDGE_ROBUSTNESS_REPORT_SCHEMA.to_owned(),
            purpose: JudgeAuditPurposeV1::QualifiedRun,
            plan_sha256: sha256_hex(plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256,
            judge_source_revision: plan.judge_source_revision,
            judge_prompt_sha256: plan.judge_prompt_sha256,
            rubric_sha256: plan.rubric_sha256,
            decision_policy_sha256: plan.decision_policy_sha256,
            blind_to_system_identity: true,
            cases,
        }
    }

    fn full_report_bytes(_plan_bytes: &[u8], fields: JudgeRobustnessReportFieldsV1) -> Vec<u8> {
        let commitment = report_commitment(&fields).unwrap();
        let report = JudgeRobustnessReportV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            plan_sha256: fields.plan_sha256,
            plan_commitment_sha256: fields.plan_commitment_sha256,
            judge_source_revision: fields.judge_source_revision,
            judge_prompt_sha256: fields.judge_prompt_sha256,
            rubric_sha256: fields.rubric_sha256,
            decision_policy_sha256: fields.decision_policy_sha256,
            blind_to_system_identity: fields.blind_to_system_identity,
            cases: fields.cases,
            report_commitment_sha256: commitment,
        };
        serde_json::to_vec(&report).unwrap()
    }

    #[test]
    fn committed_plan_verifies_at_exact_build_revision() {
        let bytes = committed_plan_bytes();
        let verification = verify_judge_robustness_plan(&bytes).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        assert_eq!(verification.planned_case_count, 6);
        assert_eq!(verification.planned_observation_count, 36);
        assert!(verification.external_anchor_required);
    }

    #[test]
    fn plan_rejects_position_swap_without_order_reversal() {
        let mut fields = plan_fields();
        fields.cases[1].variant_order = CandidateOrderV1::CandidateAFirst;
        assert!(
            compute_judge_robustness_plan_commitment(&serde_json::to_vec(&fields).unwrap())
                .is_err()
        );
    }

    #[test]
    fn plan_rejects_reference_change_in_invariance_family() {
        let mut fields = plan_fields();
        fields.cases[2].reference_variant_decision_sha256 =
            digest(SemanticDecisionV1::CandidateB.as_str());
        assert!(
            compute_judge_robustness_plan_commitment(&serde_json::to_vec(&fields).unwrap())
                .is_err()
        );
    }

    #[test]
    fn report_validity_is_separate_from_judge_quality() {
        let plan = committed_plan_bytes();
        let fields = report_fields(&plan);
        let report = full_report_bytes(&plan, fields);
        let verification = verify_judge_robustness_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        assert_eq!(verification.observed_observation_count, 36);
        let scorecard = verification.scorecard.unwrap();
        assert!(!scorecard.runner_execution_proven);
        assert!(scorecard.external_anchor_required);
    }

    #[test]
    fn report_rejects_duplicate_observation_and_missing_repeat() {
        let plan = committed_plan_bytes();
        let mut fields = report_fields(&plan);
        let case = &mut fields.cases[0];
        let first_repeat_index = case.observations[0].repeat_index;
        case.observations[1].repeat_index = first_repeat_index;
        assert!(
            compute_judge_robustness_report_commitment(
                &plan,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn report_commitment_rejects_posthoc_decision_mutation() {
        let plan = committed_plan_bytes();
        let fields = report_fields(&plan);
        let mut report: JudgeRobustnessReportV1 =
            serde_json::from_slice(&full_report_bytes(&plan, fields)).unwrap();
        report.cases[0].observations[0].raw_decision = RawJudgeDecisionV1::Right;
        let bytes = serde_json::to_vec(&report).unwrap();
        let verification = verify_judge_robustness_report(&plan, &bytes).unwrap();
        assert!(!verification.valid);
        assert!(!verification.report_commitment_matches);
    }

    #[test]
    fn stable_but_wrong_judge_can_be_validly_recorded() {
        let plan = committed_plan_bytes();
        let mut fields = report_fields(&plan);
        for observation in &mut fields.cases[2].observations {
            observation.raw_decision = RawJudgeDecisionV1::Right;
        }
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_judge_robustness_report_commitment(&plan, &fields_bytes).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&fields_bytes).unwrap();
        value["report_commitment_sha256"] = serde_json::json!(commitment);
        let report_bytes = serde_json::to_vec(&value).unwrap();
        let verification = verify_judge_robustness_report(&plan, &report_bytes).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        let summary = verification
            .scorecard
            .unwrap()
            .family_summaries
            .into_iter()
            .find(|summary| summary.family == JudgeAuditFamilyV1::FormattingInvariance)
            .unwrap();
        assert!(summary.gold_agreement_rate_millionths.unwrap() < 1_000_000);
    }
}
