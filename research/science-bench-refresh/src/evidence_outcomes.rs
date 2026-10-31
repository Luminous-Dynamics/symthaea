//! Evaluator-side outcomes for precommitted counterfactual evidence interventions.
//!
//! This module validates outcome receipts against the exact intervention plan.
//! It preserves every planned trial, keeps evaluator dimensions orthogonal, and
//! never claims that a digest authenticates the runner or that a label proves
//! scientific truth. Inference/calibration uncertainty is a separate analysis.

use crate::counterfactual_evidence::{
    EvidenceInterventionPlanV1, verify_evidence_intervention_plan,
};
use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EVIDENCE_OUTCOME_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_evidence_outcome_report.v1";
pub const EVIDENCE_OUTCOME_SCORECARD_SCHEMA: &str =
    "symthaea.science_bench_refresh_evidence_outcome_scorecard.v1";
const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.evidence_outcomes.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeTerminalStatusV1 {
    Completed,
    Contaminated,
    BudgetViolation,
    Timeout,
    Blocked,
    InvalidReceipt,
    RunnerError,
    NotStarted,
}

impl OutcomeTerminalStatusV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Contaminated => "contaminated",
            Self::BudgetViolation => "budget_violation",
            Self::Timeout => "timeout",
            Self::Blocked => "blocked",
            Self::InvalidReceipt => "invalid_receipt",
            Self::RunnerError => "runner_error",
            Self::NotStarted => "not_started",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeResponseStatusV1 {
    Answered,
    Abstained,
    NoResponse,
}

impl OutcomeResponseStatusV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Answered => "answered",
            Self::Abstained => "abstained",
            Self::NoResponse => "no_response",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCorrectnessV1 {
    Correct,
    Incorrect,
    Partial,
    NotEvaluated,
}

impl TaskCorrectnessV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Incorrect => "incorrect",
            Self::Partial => "partial",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceUseV1 {
    UsedRelevant,
    IgnoredRelevant,
    MisinterpretedRelevant,
    NoRelevantEvidence,
    NotEvaluated,
}

impl EvidenceUseV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::UsedRelevant => "used_relevant",
            Self::IgnoredRelevant => "ignored_relevant",
            Self::MisinterpretedRelevant => "misinterpreted_relevant",
            Self::NoRelevantEvidence => "no_relevant_evidence",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimSupportV1 {
    FullySupported,
    PartiallySupported,
    Unsupported,
    NotEvaluated,
}

impl ClaimSupportV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::FullySupported => "fully_supported",
            Self::PartiallySupported => "partially_supported",
            Self::Unsupported => "unsupported",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstentionQualityV1 {
    Appropriate,
    Inappropriate,
    NotApplicable,
    NotEvaluated,
}

impl AbstentionQualityV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Appropriate => "appropriate",
            Self::Inappropriate => "inappropriate",
            Self::NotApplicable => "not_applicable",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSourceV1 {
    ModelReported,
    Elicited,
    None,
}

impl ConfidenceSourceV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::ModelReported => "model_reported",
            Self::Elicited => "elicited",
            Self::None => "none",
        }
    }
}

/// Optional counters remain optional: absence is not reported as numeric zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeResourceUseV1 {
    pub wall_clock_ms: Option<u64>,
    pub tool_calls: Option<u64>,
    pub experiment_count: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Millionths of US dollars; null means unavailable, not zero.
    pub cost_micro_usd: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceOutcomeRecordV1 {
    pub pair_id: String,
    pub trial_id: String,
    pub condition: String,
    pub terminal_status: OutcomeTerminalStatusV1,
    pub response_status: OutcomeResponseStatusV1,
    /// Digest of the immutable runner receipt, if the trial was started.
    pub runner_receipt_sha256: Option<String>,
    /// Digest of the response artifact, never the raw text itself.
    pub response_artifact_sha256: Option<String>,
    /// Digest of the independent evaluator record that backs the labels below.
    pub evaluator_record_sha256: Option<String>,
    pub task_correctness: TaskCorrectnessV1,
    pub evidence_use: EvidenceUseV1,
    pub claim_support: ClaimSupportV1,
    pub abstention_quality: AbstentionQualityV1,
    /// Probability in millionths, 0..=1_000_000. Null means not collected.
    pub confidence_millionths: Option<u32>,
    pub confidence_source: ConfidenceSourceV1,
    pub resources: OutcomeResourceUseV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceOutcomeReportFieldsV1 {
    pub schema: String,
    /// SHA-256 of the exact plan bytes, not its reserialized object.
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub evaluator_revision_sha256: String,
    pub decision_rules_sha256: String,
    /// Exactly one ordered record for every plan trial, including failures.
    pub records: Vec<EvidenceOutcomeRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceOutcomeReportV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub evaluator_revision_sha256: String,
    pub decision_rules_sha256: String,
    pub records: Vec<EvidenceOutcomeRecordV1>,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CounterSummaryV1 {
    pub recorded_trials: u64,
    pub total_observed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConditionOutcomeSummaryV1 {
    pub condition: String,
    pub planned_trials: u64,
    pub terminal_status_counts: BTreeMap<String, u64>,
    pub response_status_counts: BTreeMap<String, u64>,
    pub correctness_counts: BTreeMap<String, u64>,
    pub evidence_use_counts: BTreeMap<String, u64>,
    pub claim_support_counts: BTreeMap<String, u64>,
    pub abstention_quality_counts: BTreeMap<String, u64>,
    pub confidence_source_counts: BTreeMap<String, u64>,
    pub confidence_recorded_trials: u64,
    pub confidence_missing_answered_trials: u64,
    pub wall_clock_ms: CounterSummaryV1,
    pub tool_calls: CounterSummaryV1,
    pub experiment_count: CounterSummaryV1,
    pub input_tokens: CounterSummaryV1,
    pub output_tokens: CounterSummaryV1,
    pub cost_micro_usd: CounterSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceOutcomeScorecardV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub report_sha256: String,
    pub report_commitment_sha256: String,
    pub planned_trial_count: u64,
    pub observed_record_count: u64,
    pub missing_trial_count: u64,
    pub terminal_status_counts: BTreeMap<String, u64>,
    pub condition_summaries: Vec<ConditionOutcomeSummaryV1>,
    pub external_anchor_required: bool,
    pub runner_execution_proven: bool,
    pub uncertainty_model: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceOutcomeVerificationReportV1 {
    pub schema: String,
    pub valid: bool,
    pub plan_binding_valid: bool,
    pub report_commitment_matches: bool,
    pub planned_trial_count: u64,
    pub observed_record_count: u64,
    pub missing_trial_count: u64,
    pub issues: Vec<String>,
    pub scorecard: Option<EvidenceOutcomeScorecardV1>,
}

#[derive(Debug, Clone, Serialize)]
struct OutcomeCommitmentPayloadV1<'a> {
    schema: &'a str,
    plan_sha256: &'a str,
    plan_commitment_sha256: &'a str,
    evaluator_revision_sha256: &'a str,
    decision_rules_sha256: &'a str,
    records: &'a [EvidenceOutcomeRecordV1],
}

fn require_keys(value: &serde_json::Value, keys: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    for key in keys {
        if !object.contains_key(*key) {
            return Err(format!("{label} is missing required field {key}"));
        }
    }
    Ok(())
}

/// Serde intentionally maps omitted Option fields to None; the receipt contract
/// distinguishes an explicit null from an omitted field, so enforce presence.
fn validate_json_shape(bytes: &[u8], full_report: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("outcome JSON is malformed: {error}"))?;
    let mut root_keys = vec![
        "schema",
        "plan_sha256",
        "plan_commitment_sha256",
        "evaluator_revision_sha256",
        "decision_rules_sha256",
        "records",
    ];
    if full_report {
        root_keys.push("report_commitment_sha256");
    }
    require_keys(&value, &root_keys, "outcome report")?;
    let records = value["records"]
        .as_array()
        .ok_or_else(|| "outcome report records must be a JSON array".to_owned())?;
    const RECORD_KEYS: &[&str] = &[
        "pair_id",
        "trial_id",
        "condition",
        "terminal_status",
        "response_status",
        "runner_receipt_sha256",
        "response_artifact_sha256",
        "evaluator_record_sha256",
        "task_correctness",
        "evidence_use",
        "claim_support",
        "abstention_quality",
        "confidence_millionths",
        "confidence_source",
        "resources",
    ];
    const RESOURCE_KEYS: &[&str] = &[
        "wall_clock_ms",
        "tool_calls",
        "experiment_count",
        "input_tokens",
        "output_tokens",
        "cost_micro_usd",
    ];
    for (index, record) in records.iter().enumerate() {
        let label = format!("records[{index}]");
        require_keys(record, RECORD_KEYS, &label)?;
        require_keys(
            &record["resources"],
            RESOURCE_KEYS,
            &format!("{label}.resources"),
        )?;
    }
    Ok(())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && !value.bytes().all(|byte| byte == b'0')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn commitment_from_fields(fields: &EvidenceOutcomeReportFieldsV1) -> Result<String, String> {
    let payload = OutcomeCommitmentPayloadV1 {
        schema: &fields.schema,
        plan_sha256: &fields.plan_sha256,
        plan_commitment_sha256: &fields.plan_commitment_sha256,
        evaluator_revision_sha256: &fields.evaluator_revision_sha256,
        decision_rules_sha256: &fields.decision_rules_sha256,
        records: &fields.records,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize outcome commitment payload: {error}"))?;
    digest_domain_framed(b"outcome_report.v1", &encoded)
}

fn fields_from_report(report: &EvidenceOutcomeReportV1) -> EvidenceOutcomeReportFieldsV1 {
    EvidenceOutcomeReportFieldsV1 {
        schema: report.schema.clone(),
        plan_sha256: report.plan_sha256.clone(),
        plan_commitment_sha256: report.plan_commitment_sha256.clone(),
        evaluator_revision_sha256: report.evaluator_revision_sha256.clone(),
        decision_rules_sha256: report.decision_rules_sha256.clone(),
        records: report.records.clone(),
    }
}

fn unscored(record: &EvidenceOutcomeRecordV1) -> bool {
    record.task_correctness == TaskCorrectnessV1::NotEvaluated
        && record.evidence_use == EvidenceUseV1::NotEvaluated
        && record.claim_support == ClaimSupportV1::NotEvaluated
        && record.abstention_quality == AbstentionQualityV1::NotEvaluated
        && record.confidence_millionths.is_none()
        && record.confidence_source == ConfidenceSourceV1::None
}

fn validate_record(record: &EvidenceOutcomeRecordV1, position: usize) -> Vec<String> {
    let mut issues = Vec::new();
    let label = format!("records[{position}]");
    if record.pair_id.trim().is_empty() || record.pair_id.len() > 128 {
        issues.push(format!("{label}.pair_id must be 1..=128 characters"));
    }
    if record.trial_id.trim().is_empty() || record.trial_id.len() > 160 {
        issues.push(format!("{label}.trial_id must be 1..=160 characters"));
    }
    if record.condition.trim().is_empty() || record.condition.len() > 128 {
        issues.push(format!("{label}.condition must be 1..=128 characters"));
    }
    for (name, digest) in [
        (
            "runner_receipt_sha256",
            record.runner_receipt_sha256.as_deref(),
        ),
        (
            "response_artifact_sha256",
            record.response_artifact_sha256.as_deref(),
        ),
        (
            "evaluator_record_sha256",
            record.evaluator_record_sha256.as_deref(),
        ),
    ] {
        if digest.is_some_and(|value| !valid_digest(value)) {
            issues.push(format!(
                "{label}.{name} must be a nonzero lowercase SHA-256 digest"
            ));
        }
    }
    if record
        .confidence_millionths
        .is_some_and(|value| value > 1_000_000)
    {
        issues.push(format!(
            "{label}.confidence_millionths must be between 0 and 1,000,000"
        ));
    }
    match (record.confidence_source, record.confidence_millionths) {
        (ConfidenceSourceV1::None, Some(_)) => {
            issues.push(format!("{label} has confidence but source=none"));
        }
        (ConfidenceSourceV1::ModelReported | ConfidenceSourceV1::Elicited, None) => {
            issues.push(format!(
                "{label} declares a confidence source but has no confidence value"
            ));
        }
        _ => {}
    }

    let executed = record.terminal_status != OutcomeTerminalStatusV1::NotStarted;
    if executed && record.runner_receipt_sha256.is_none() {
        issues.push(format!("{label} started but lacks a runner receipt digest"));
    }
    if !executed && record.runner_receipt_sha256.is_some() {
        issues.push(format!(
            "{label} is not_started but carries a runner receipt"
        ));
    }

    match record.terminal_status {
        OutcomeTerminalStatusV1::Completed => {
            if record.response_status == OutcomeResponseStatusV1::NoResponse {
                issues.push(format!("{label} completed but response_status=no_response"));
            }
            if record.response_artifact_sha256.is_none() {
                issues.push(format!(
                    "{label} completed but lacks a response artifact digest"
                ));
            }
            if record.evaluator_record_sha256.is_none() {
                issues.push(format!(
                    "{label} completed but lacks an evaluator record digest"
                ));
            }
            match record.response_status {
                OutcomeResponseStatusV1::Answered => {
                    if record.task_correctness == TaskCorrectnessV1::NotEvaluated
                        || record.evidence_use == EvidenceUseV1::NotEvaluated
                        || record.claim_support == ClaimSupportV1::NotEvaluated
                    {
                        issues.push(format!(
                            "{label} answered but has unevaluated required axes"
                        ));
                    }
                    if record.abstention_quality != AbstentionQualityV1::NotApplicable {
                        issues.push(format!(
                            "{label} answered but abstention_quality is not not_applicable"
                        ));
                    }
                }
                OutcomeResponseStatusV1::Abstained => {
                    if record.task_correctness != TaskCorrectnessV1::NotEvaluated
                        || record.claim_support != ClaimSupportV1::NotEvaluated
                    {
                        issues.push(format!(
                            "{label} abstained but claims answer-only outcome labels"
                        ));
                    }
                    if !matches!(
                        record.evidence_use,
                        EvidenceUseV1::UsedRelevant
                            | EvidenceUseV1::NoRelevantEvidence
                            | EvidenceUseV1::NotEvaluated
                    ) {
                        issues.push(format!(
                            "{label} abstained but claims ignored or misinterpreted evidence"
                        ));
                    }
                    if !matches!(
                        record.abstention_quality,
                        AbstentionQualityV1::Appropriate | AbstentionQualityV1::Inappropriate
                    ) {
                        issues.push(format!(
                            "{label} abstained but lacks an abstention-quality label"
                        ));
                    }
                    if record.confidence_millionths.is_some()
                        || record.confidence_source != ConfidenceSourceV1::None
                    {
                        issues.push(format!("{label} abstained but carries answer confidence"));
                    }
                }
                OutcomeResponseStatusV1::NoResponse => {}
            }
            if record.confidence_millionths.is_some()
                && record.response_status != OutcomeResponseStatusV1::Answered
            {
                issues.push(format!(
                    "{label} records confidence without an answered response"
                ));
            }
        }
        OutcomeTerminalStatusV1::Contaminated | OutcomeTerminalStatusV1::BudgetViolation => {
            if record.evaluator_record_sha256.is_none() {
                issues.push(format!("{label} lacks an evaluator disposition digest"));
            }
            if record.response_status == OutcomeResponseStatusV1::NoResponse {
                if record.response_artifact_sha256.is_some() {
                    issues.push(format!(
                        "{label} has a response artifact but response_status=no_response"
                    ));
                }
            } else if record.response_artifact_sha256.is_none() {
                issues.push(format!(
                    "{label} has a response status but lacks its response artifact"
                ));
            }
            if !unscored(record) {
                issues.push(format!(
                    "{label} is contaminated/budget-violating and must not carry scored outcomes"
                ));
            }
        }
        OutcomeTerminalStatusV1::Timeout
        | OutcomeTerminalStatusV1::Blocked
        | OutcomeTerminalStatusV1::InvalidReceipt
        | OutcomeTerminalStatusV1::RunnerError
        | OutcomeTerminalStatusV1::NotStarted => {
            if record.response_status != OutcomeResponseStatusV1::NoResponse {
                issues.push(format!(
                    "{label} has a non-completed terminal status with a response"
                ));
            }
            if record.evaluator_record_sha256.is_some() {
                issues.push(format!(
                    "{label} has an evaluator score record for an unscored terminal status"
                ));
            }
            if !unscored(record) {
                issues.push(format!(
                    "{label} has scored outcomes despite a non-completed terminal status"
                ));
            }
            if record.terminal_status == OutcomeTerminalStatusV1::NotStarted
                && (record.response_artifact_sha256.is_some()
                    || record.resources.wall_clock_ms.is_some()
                    || record.resources.tool_calls.is_some()
                    || record.resources.experiment_count.is_some()
                    || record.resources.input_tokens.is_some()
                    || record.resources.output_tokens.is_some()
                    || record.resources.cost_micro_usd.is_some())
            {
                issues.push(format!(
                    "{label} is not_started but carries response artifacts or resource observations"
                ));
            }
        }
    }
    issues
}

fn increment(map: &mut BTreeMap<String, u64>, key: &str) -> Result<(), String> {
    let entry = map.entry(key.to_owned()).or_default();
    *entry = entry
        .checked_add(1)
        .ok_or_else(|| format!("summary count overflow for {key}"))?;
    Ok(())
}

fn resource_summary(values: impl Iterator<Item = Option<u64>>) -> Result<CounterSummaryV1, String> {
    let mut summary = CounterSummaryV1 {
        recorded_trials: 0,
        total_observed: 0,
    };
    for value in values.flatten() {
        summary.recorded_trials = summary
            .recorded_trials
            .checked_add(1)
            .ok_or_else(|| "resource recorded-trial count overflow".to_owned())?;
        summary.total_observed = summary
            .total_observed
            .checked_add(value)
            .ok_or_else(|| "resource total overflow".to_owned())?;
    }
    Ok(summary)
}

fn scorecard(
    plan_bytes: &[u8],
    plan: &EvidenceInterventionPlanV1,
    report_bytes: &[u8],
    report: &EvidenceOutcomeReportV1,
) -> Result<EvidenceOutcomeScorecardV1, String> {
    let mut terminal_total = BTreeMap::new();
    let mut condition_summaries = Vec::new();
    for pair in &plan.pairs {
        for trial in &pair.trials {
            let record = report
                .records
                .iter()
                .find(|record| record.trial_id == trial.trial_id)
                .ok_or_else(|| {
                    format!("internal error: missing validated trial {}", trial.trial_id)
                })?;
            increment(&mut terminal_total, record.terminal_status.as_str())?;
        }
    }

    for condition in &plan.required_conditions {
        let selected = report
            .records
            .iter()
            .filter(|record| record.condition == *condition)
            .collect::<Vec<_>>();
        let mut terminal_status_counts = BTreeMap::new();
        let mut response_status_counts = BTreeMap::new();
        let mut correctness_counts = BTreeMap::new();
        let mut evidence_use_counts = BTreeMap::new();
        let mut claim_support_counts = BTreeMap::new();
        let mut abstention_quality_counts = BTreeMap::new();
        let mut confidence_source_counts = BTreeMap::new();
        let mut confidence_recorded_trials = 0u64;
        let mut confidence_missing_answered_trials = 0u64;
        for record in &selected {
            increment(&mut terminal_status_counts, record.terminal_status.as_str())?;
            increment(&mut response_status_counts, record.response_status.as_str())?;
            increment(&mut correctness_counts, record.task_correctness.as_str())?;
            increment(&mut evidence_use_counts, record.evidence_use.as_str())?;
            increment(&mut claim_support_counts, record.claim_support.as_str())?;
            increment(
                &mut abstention_quality_counts,
                record.abstention_quality.as_str(),
            )?;
            increment(
                &mut confidence_source_counts,
                record.confidence_source.as_str(),
            )?;
            if record.confidence_millionths.is_some() {
                confidence_recorded_trials = confidence_recorded_trials
                    .checked_add(1)
                    .ok_or_else(|| "confidence count overflow".to_owned())?;
            }
            if record.terminal_status == OutcomeTerminalStatusV1::Completed
                && record.response_status == OutcomeResponseStatusV1::Answered
                && record.confidence_millionths.is_none()
            {
                confidence_missing_answered_trials = confidence_missing_answered_trials
                    .checked_add(1)
                    .ok_or_else(|| "confidence missing count overflow".to_owned())?;
            }
        }
        condition_summaries.push(ConditionOutcomeSummaryV1 {
            condition: condition.clone(),
            planned_trials: u64::try_from(selected.len())
                .map_err(|_| "condition trial count exceeds u64".to_owned())?,
            terminal_status_counts,
            response_status_counts,
            correctness_counts,
            evidence_use_counts,
            claim_support_counts,
            abstention_quality_counts,
            confidence_source_counts,
            confidence_recorded_trials,
            confidence_missing_answered_trials,
            wall_clock_ms: resource_summary(selected.iter().map(|r| r.resources.wall_clock_ms))?,
            tool_calls: resource_summary(selected.iter().map(|r| r.resources.tool_calls))?,
            experiment_count: resource_summary(
                selected.iter().map(|r| r.resources.experiment_count),
            )?,
            input_tokens: resource_summary(selected.iter().map(|r| r.resources.input_tokens))?,
            output_tokens: resource_summary(selected.iter().map(|r| r.resources.output_tokens))?,
            cost_micro_usd: resource_summary(selected.iter().map(|r| r.resources.cost_micro_usd))?,
        });
    }

    Ok(EvidenceOutcomeScorecardV1 {
        schema: EVIDENCE_OUTCOME_SCORECARD_SCHEMA.to_owned(),
        plan_sha256: sha256_hex(plan_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
        report_sha256: sha256_hex(report_bytes),
        report_commitment_sha256: report.report_commitment_sha256.clone(),
        planned_trial_count: u64::try_from(report.records.len())
            .map_err(|_| "planned trial count exceeds u64".to_owned())?,
        observed_record_count: u64::try_from(report.records.len())
            .map_err(|_| "observed record count exceeds u64".to_owned())?,
        missing_trial_count: 0,
        terminal_status_counts: terminal_total,
        condition_summaries,
        external_anchor_required: true,
        runner_execution_proven: false,
        uncertainty_model: "not_estimated_by_structural_validator".to_owned(),
        claim_boundary: "descriptive labels; runner unverified; no generalization/SOTA claim"
            .to_owned(),
    })
}

fn validate_against_plan(
    plan_bytes: &[u8],
    fields: &EvidenceOutcomeReportFieldsV1,
) -> Result<(EvidenceInterventionPlanV1, Vec<String>), String> {
    let plan_verification = verify_evidence_intervention_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err(format!(
            "referenced plan is invalid: {}",
            plan_verification.issues.join("; ")
        ));
    }
    let plan: EvidenceInterventionPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("cannot parse verified intervention plan: {error}"))?;
    let mut issues = Vec::new();
    if fields.schema != EVIDENCE_OUTCOME_REPORT_SCHEMA {
        issues.push(format!("schema must be {EVIDENCE_OUTCOME_REPORT_SCHEMA}"));
    }
    if fields.plan_sha256 != sha256_hex(plan_bytes) {
        issues.push("plan_sha256 must bind the exact plan bytes".to_owned());
    }
    if fields.plan_commitment_sha256 != plan.plan_commitment_sha256 {
        issues.push("plan_commitment_sha256 does not match the validated plan".to_owned());
    }
    if fields.evaluator_revision_sha256 != plan.evaluator_revision_sha256 {
        issues.push("evaluator_revision_sha256 differs from the precommitted plan".to_owned());
    }
    if fields.decision_rules_sha256 != plan.decision_rules_sha256 {
        issues.push("decision_rules_sha256 differs from the precommitted plan".to_owned());
    }
    for (name, value) in [
        (
            "evaluator_revision_sha256",
            fields.evaluator_revision_sha256.as_str(),
        ),
        (
            "decision_rules_sha256",
            fields.decision_rules_sha256.as_str(),
        ),
    ] {
        if !valid_digest(value) {
            issues.push(format!("{name} must be a nonzero lowercase SHA-256 digest"));
        }
    }

    let expected = plan
        .pairs
        .iter()
        .flat_map(|pair| {
            pair.trials.iter().map(move |trial| {
                (
                    pair.pair_id.as_str(),
                    trial.trial_id.as_str(),
                    trial.condition.as_str(),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut seen_trials = BTreeSet::new();
    for (index, record) in fields.records.iter().enumerate() {
        issues.extend(validate_record(record, index));
        if !seen_trials.insert(record.trial_id.as_str()) {
            issues.push(format!(
                "duplicate trial_id in outcome report: {}",
                record.trial_id
            ));
        }
        if let Some((pair_id, trial_id, condition)) = expected.get(index).copied() {
            if record.pair_id != pair_id
                || record.trial_id != trial_id
                || record.condition != condition
            {
                issues.push(format!(
                    "records[{index}] does not match the planned pair, trial, and condition"
                ));
            }
        } else {
            issues.push(format!(
                "outcome report contains unknown extra record at index {index}"
            ));
        }
    }
    if fields.records.len() != expected.len() {
        issues.push(format!(
            "outcome report must preserve every planned trial: expected {}, observed {}",
            expected.len(),
            fields.records.len()
        ));
    }
    if seen_trials.len() != expected.len().min(fields.records.len()) {
        issues.push("outcome report trial identity coverage is not one-to-one".to_owned());
    }
    Ok((plan, issues))
}

/// Compute the typed, domain-separated report commitment after validating all
/// records against the plan. The resulting hash is not an execution attestation.
pub fn compute_evidence_outcome_report_commitment(
    plan_bytes: &[u8],
    fields_bytes: &[u8],
) -> Result<String, String> {
    validate_json_shape(fields_bytes, false)?;
    let fields: EvidenceOutcomeReportFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("outcome fields do not match strict v1 shape: {error}"))?;
    let (_, issues) = validate_against_plan(plan_bytes, &fields)?;
    if !issues.is_empty() {
        return Err(format!("outcome fields invalid: {}", issues.join("; ")));
    }
    commitment_from_fields(&fields)
}

/// Verify the exact plan binding, complete trial census, evaluator status
/// invariants and report commitment, then build a multidimensional scorecard.
pub fn verify_evidence_outcome_report(
    plan_bytes: &[u8],
    report_bytes: &[u8],
) -> Result<EvidenceOutcomeVerificationReportV1, String> {
    validate_json_shape(report_bytes, true)?;
    let report: EvidenceOutcomeReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("outcome report does not match strict v1 shape: {error}"))?;
    let fields = fields_from_report(&report);
    let (plan, mut issues) = validate_against_plan(plan_bytes, &fields)?;
    if !valid_digest(&report.report_commitment_sha256) {
        issues
            .push("report_commitment_sha256 must be a nonzero lowercase SHA-256 digest".to_owned());
    }
    let recomputed = commitment_from_fields(&fields)?;
    let commitment_matches = valid_digest(&report.report_commitment_sha256)
        && report.report_commitment_sha256 == recomputed;
    if !commitment_matches {
        issues.push("report_commitment_sha256 does not match the typed report contents".to_owned());
    }
    let plan_binding_valid = fields.plan_sha256 == sha256_hex(plan_bytes)
        && fields.plan_commitment_sha256 == plan.plan_commitment_sha256
        && verify_evidence_intervention_plan(plan_bytes)?.valid;
    let valid = issues.is_empty() && commitment_matches && plan_binding_valid;
    let expected_trial_ids = plan
        .pairs
        .iter()
        .flat_map(|pair| pair.trials.iter().map(|trial| trial.trial_id.as_str()))
        .collect::<BTreeSet<_>>();
    let observed_trial_ids = report
        .records
        .iter()
        .map(|record| record.trial_id.as_str())
        .collect::<BTreeSet<_>>();
    let expected_count = expected_trial_ids.len();
    let observed_count = report.records.len();
    let missing_count = expected_trial_ids.difference(&observed_trial_ids).count();
    let scorecard = if valid {
        Some(scorecard(plan_bytes, &plan, report_bytes, &report)?)
    } else {
        None
    };
    Ok(EvidenceOutcomeVerificationReportV1 {
        schema: "symthaea.science_bench_refresh_evidence_outcome_verification.v1".to_owned(),
        valid,
        plan_binding_valid,
        report_commitment_matches: commitment_matches,
        planned_trial_count: u64::try_from(expected_count).unwrap_or(u64::MAX),
        observed_record_count: u64::try_from(observed_count).unwrap_or(u64::MAX),
        missing_trial_count: u64::try_from(missing_count).unwrap_or(u64::MAX),
        issues,
        scorecard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual_evidence::{
        EvidenceInterventionPairV1, EvidenceInterventionPlanFieldsV1, EvidenceInterventionTrialV1,
    };
    use serde_json::json;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn intervention_plan() -> Vec<u8> {
        let conditions = [
            "natural_retrieval",
            "network_isolated",
            "gold_evidence_injected",
        ];
        let trials = conditions
            .iter()
            .enumerate()
            .map(|(index, condition)| EvidenceInterventionTrialV1 {
                trial_id: format!("trial-{index}"),
                condition: (*condition).to_owned(),
                policy_input_sha256: digest(char::from(b'a' + index as u8)),
                evidence_manifest_sha256: digest(char::from(b'd' + index as u8)),
                randomization_order: index as u32,
                seed_commitment_sha256: digest(char::from(b'7' + index as u8)),
            })
            .collect::<Vec<_>>();
        let fields = EvidenceInterventionPlanFieldsV1 {
            schema: crate::counterfactual_evidence::EVIDENCE_INTERVENTION_PLAN_SCHEMA.to_owned(),
            generation_id: "generation-test".to_owned(),
            plan_id: "plan-test".to_owned(),
            policy_revision_sha256: digest('1'),
            configuration_sha256: digest('2'),
            prompt_sha256: digest('3'),
            tool_manifest_sha256: digest('4'),
            budget_sha256: digest('5'),
            analysis_plan_sha256: digest('6'),
            evaluator_revision_sha256: digest('9'),
            decision_rules_sha256: digest('c'),
            randomization_commitment_sha256: digest('7'),
            required_conditions: conditions.iter().map(|value| (*value).to_owned()).collect(),
            pairs: vec![EvidenceInterventionPairV1 {
                pair_id: "pair-test".to_owned(),
                task_ref_sha256: digest('8'),
                task_family_id: "science".to_owned(),
                trials,
            }],
        };
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment =
            crate::counterfactual_evidence::compute_evidence_intervention_plan_commitment(
                &fields_bytes,
            )
            .unwrap();
        let mut full = serde_json::to_value(fields).unwrap();
        full["plan_commitment_sha256"] = json!(commitment);
        serde_json::to_vec(&full).unwrap()
    }

    fn outcome_record(
        pair_id: &str,
        trial_id: &str,
        condition: &str,
        index: usize,
    ) -> EvidenceOutcomeRecordV1 {
        EvidenceOutcomeRecordV1 {
            pair_id: pair_id.to_owned(),
            trial_id: trial_id.to_owned(),
            condition: condition.to_owned(),
            terminal_status: OutcomeTerminalStatusV1::Completed,
            response_status: OutcomeResponseStatusV1::Answered,
            runner_receipt_sha256: Some(digest(char::from(b'a' + index as u8))),
            response_artifact_sha256: Some(digest(char::from(b'd' + index as u8))),
            evaluator_record_sha256: Some(digest(char::from(b'1' + index as u8))),
            task_correctness: if index == 1 {
                TaskCorrectnessV1::Incorrect
            } else {
                TaskCorrectnessV1::Correct
            },
            evidence_use: EvidenceUseV1::UsedRelevant,
            claim_support: ClaimSupportV1::FullySupported,
            abstention_quality: AbstentionQualityV1::NotApplicable,
            confidence_millionths: Some(900_000),
            confidence_source: ConfidenceSourceV1::Elicited,
            resources: OutcomeResourceUseV1 {
                wall_clock_ms: Some(100 + index as u64),
                tool_calls: Some(index as u64),
                experiment_count: Some(2),
                input_tokens: Some(1_000),
                output_tokens: Some(100),
                cost_micro_usd: None,
            },
        }
    }

    fn fields(plan: &[u8]) -> EvidenceOutcomeReportFieldsV1 {
        let plan_obj: EvidenceInterventionPlanV1 = serde_json::from_slice(plan).unwrap();
        let records = plan_obj
            .pairs
            .iter()
            .flat_map(|pair| {
                pair.trials.iter().enumerate().map(move |(index, trial)| {
                    outcome_record(&pair.pair_id, &trial.trial_id, &trial.condition, index)
                })
            })
            .collect::<Vec<_>>();
        EvidenceOutcomeReportFieldsV1 {
            schema: EVIDENCE_OUTCOME_REPORT_SCHEMA.to_owned(),
            plan_sha256: sha256_hex(plan),
            plan_commitment_sha256: plan_obj.plan_commitment_sha256,
            evaluator_revision_sha256: plan_obj.evaluator_revision_sha256,
            decision_rules_sha256: plan_obj.decision_rules_sha256,
            records,
        }
    }

    fn full_report(fields: EvidenceOutcomeReportFieldsV1) -> Vec<u8> {
        let commitment = commitment_from_fields(&fields).unwrap();
        let mut value = serde_json::to_value(fields).unwrap();
        value["report_commitment_sha256"] = json!(commitment);
        serde_json::to_vec(&value).unwrap()
    }

    fn make_unstarted(record: &mut EvidenceOutcomeRecordV1) {
        record.terminal_status = OutcomeTerminalStatusV1::NotStarted;
        record.response_status = OutcomeResponseStatusV1::NoResponse;
        record.runner_receipt_sha256 = None;
        record.response_artifact_sha256 = None;
        record.evaluator_record_sha256 = None;
        record.task_correctness = TaskCorrectnessV1::NotEvaluated;
        record.evidence_use = EvidenceUseV1::NotEvaluated;
        record.claim_support = ClaimSupportV1::NotEvaluated;
        record.abstention_quality = AbstentionQualityV1::NotEvaluated;
        record.confidence_millionths = None;
        record.confidence_source = ConfidenceSourceV1::None;
        record.resources = OutcomeResourceUseV1 {
            wall_clock_ms: None,
            tool_calls: None,
            experiment_count: None,
            input_tokens: None,
            output_tokens: None,
            cost_micro_usd: None,
        };
    }

    #[test]
    fn accepts_complete_report_and_keeps_missing_cost_distinct_from_zero() {
        let plan = intervention_plan();
        let report = full_report(fields(&plan));
        let verification = verify_evidence_outcome_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        let scorecard = verification.scorecard.unwrap();
        assert_eq!(scorecard.planned_trial_count, 3);
        assert_eq!(scorecard.observed_record_count, 3);
        assert_eq!(scorecard.missing_trial_count, 0);
        assert_eq!(scorecard.condition_summaries.len(), 3);
        assert_eq!(
            scorecard.condition_summaries[0].confidence_recorded_trials,
            1
        );
        assert_eq!(
            scorecard.condition_summaries[0]
                .cost_micro_usd
                .recorded_trials,
            0
        );
        assert_eq!(
            scorecard.condition_summaries[0]
                .cost_micro_usd
                .total_observed,
            0
        );
        assert!(!scorecard.runner_execution_proven);
    }

    #[test]
    fn preserves_not_started_trial_and_does_not_impute_resource_zeroes() {
        let plan = intervention_plan();
        let mut fields = fields(&plan);
        make_unstarted(&mut fields.records[2]);
        let report = full_report(fields);
        let verification = verify_evidence_outcome_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        let scorecard = verification.scorecard.unwrap();
        assert_eq!(scorecard.planned_trial_count, 3);
        assert_eq!(scorecard.missing_trial_count, 0);
        assert_eq!(
            scorecard.terminal_status_counts.get("not_started"),
            Some(&1)
        );
        let last_condition = scorecard.condition_summaries.last().unwrap();
        assert_eq!(last_condition.wall_clock_ms.recorded_trials, 0);
    }

    #[test]
    fn preserves_partial_timeout_artifact_without_scoring_it() {
        let plan = intervention_plan();
        let mut fields = fields(&plan);
        let record = &mut fields.records[1];
        record.terminal_status = OutcomeTerminalStatusV1::Timeout;
        record.response_status = OutcomeResponseStatusV1::NoResponse;
        record.response_artifact_sha256 = Some(digest('e'));
        record.evaluator_record_sha256 = None;
        record.task_correctness = TaskCorrectnessV1::NotEvaluated;
        record.evidence_use = EvidenceUseV1::NotEvaluated;
        record.claim_support = ClaimSupportV1::NotEvaluated;
        record.abstention_quality = AbstentionQualityV1::NotEvaluated;
        record.confidence_millionths = None;
        record.confidence_source = ConfidenceSourceV1::None;

        let report = full_report(fields);
        let verification = verify_evidence_outcome_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        let scorecard = verification.scorecard.unwrap();
        assert_eq!(scorecard.planned_trial_count, 3);
        assert_eq!(scorecard.terminal_status_counts.get("timeout"), Some(&1));
        // The raw receipt retains the partial output digest while the failed
        // trial contributes no correctness/evidence-use score.
        let outcome: EvidenceOutcomeReportV1 = serde_json::from_slice(&report).unwrap();
        assert_eq!(
            outcome.records[1].response_artifact_sha256.as_deref(),
            Some(digest('e').as_str())
        );
        assert_eq!(
            outcome.records[1].task_correctness,
            TaskCorrectnessV1::NotEvaluated
        );
    }

    #[test]
    fn abstention_can_be_supported_by_relevant_but_conflicting_evidence() {
        let plan = intervention_plan();
        let mut fields = fields(&plan);
        let record = &mut fields.records[2];
        record.response_status = OutcomeResponseStatusV1::Abstained;
        record.task_correctness = TaskCorrectnessV1::NotEvaluated;
        record.evidence_use = EvidenceUseV1::UsedRelevant;
        record.claim_support = ClaimSupportV1::NotEvaluated;
        record.abstention_quality = AbstentionQualityV1::Appropriate;
        record.confidence_millionths = None;
        record.confidence_source = ConfidenceSourceV1::None;

        let report = full_report(fields);
        let verification = verify_evidence_outcome_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        let scorecard = verification.scorecard.unwrap();
        assert_eq!(scorecard.planned_trial_count, 3);
    }

    #[test]
    fn rejects_missing_duplicate_and_wrong_condition_records() {
        let plan = intervention_plan();

        let mut missing = fields(&plan);
        missing.records.pop();
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&missing).unwrap()
            )
            .is_err()
        );

        let mut duplicate = fields(&plan);
        duplicate.records[1] = duplicate.records[0].clone();
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&duplicate).unwrap()
            )
            .is_err()
        );
        let duplicate_report = full_report(duplicate);
        let duplicate_verification =
            verify_evidence_outcome_report(&plan, &duplicate_report).unwrap();
        assert!(!duplicate_verification.valid);
        assert_eq!(duplicate_verification.missing_trial_count, 1);

        let mut wrong_condition = fields(&plan);
        wrong_condition.records[0].condition = "source_contradiction".to_owned();
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&wrong_condition).unwrap()
            )
            .is_err()
        );

        // Evaluator revision and decision rules cannot be changed after the plan
        // commitment without invalidating outcome-report admission.
        let mut changed_evaluator = fields(&plan);
        changed_evaluator.evaluator_revision_sha256 = digest('1');
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&changed_evaluator).unwrap()
            )
            .is_err()
        );

        let mut changed_rules = fields(&plan);
        changed_rules.decision_rules_sha256 = digest('2');
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&changed_rules).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_mutated_report_commitment_without_emitting_a_scorecard() {
        let plan = intervention_plan();
        let mut report: serde_json::Value =
            serde_json::from_slice(&full_report(fields(&plan))).unwrap();
        report["report_commitment_sha256"] = json!(digest('0'));
        let verification =
            verify_evidence_outcome_report(&plan, &serde_json::to_vec(&report).unwrap()).unwrap();
        assert!(!verification.valid);
        assert!(!verification.report_commitment_matches);
        assert!(verification.scorecard.is_none());
    }

    #[test]
    fn contaminated_and_budget_violating_trials_cannot_receive_task_scores() {
        let plan = intervention_plan();
        let mut contaminated = fields(&plan);
        let row = &mut contaminated.records[0];
        row.terminal_status = OutcomeTerminalStatusV1::Contaminated;
        row.task_correctness = TaskCorrectnessV1::NotEvaluated;
        row.evidence_use = EvidenceUseV1::NotEvaluated;
        row.claim_support = ClaimSupportV1::NotEvaluated;
        row.abstention_quality = AbstentionQualityV1::NotEvaluated;
        row.confidence_millionths = None;
        row.confidence_source = ConfidenceSourceV1::None;
        let valid_report = full_report(contaminated.clone());
        let verification = verify_evidence_outcome_report(&plan, &valid_report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);

        contaminated.records[0].task_correctness = TaskCorrectnessV1::Correct;
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&contaminated).unwrap()
            )
            .is_err()
        );

        let mut over_budget = fields(&plan);
        over_budget.records[1].terminal_status = OutcomeTerminalStatusV1::BudgetViolation;
        over_budget.records[1].task_correctness = TaskCorrectnessV1::NotEvaluated;
        over_budget.records[1].evidence_use = EvidenceUseV1::NotEvaluated;
        over_budget.records[1].claim_support = ClaimSupportV1::NotEvaluated;
        over_budget.records[1].abstention_quality = AbstentionQualityV1::NotEvaluated;
        over_budget.records[1].confidence_millionths = None;
        over_budget.records[1].confidence_source = ConfidenceSourceV1::None;
        let report = full_report(over_budget.clone());
        let verification = verify_evidence_outcome_report(&plan, &report).unwrap();
        assert!(verification.valid, "{:?}", verification.issues);
        over_budget.records[1].task_correctness = TaskCorrectnessV1::Incorrect;
        assert!(
            compute_evidence_outcome_report_commitment(
                &plan,
                &serde_json::to_vec(&over_budget).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_confidence_outside_unit_interval_and_source_mismatch() {
        let plan = intervention_plan();
        let mut bad = fields(&plan);
        bad.records[0].confidence_millionths = Some(1_000_001);
        assert!(
            compute_evidence_outcome_report_commitment(&plan, &serde_json::to_vec(&bad).unwrap())
                .is_err()
        );

        bad = fields(&plan);
        bad.records[0].confidence_source = ConfidenceSourceV1::None;
        assert!(
            compute_evidence_outcome_report_commitment(&plan, &serde_json::to_vec(&bad).unwrap())
                .is_err()
        );
    }

    #[test]
    fn rejects_omitted_nullable_resource_field() {
        let plan = intervention_plan();
        let mut report: serde_json::Value =
            serde_json::from_slice(&full_report(fields(&plan))).unwrap();
        report["records"][0]["resources"]
            .as_object_mut()
            .unwrap()
            .remove("cost_micro_usd");
        assert!(
            verify_evidence_outcome_report(&plan, &serde_json::to_vec(&report).unwrap()).is_err()
        );
    }
}
