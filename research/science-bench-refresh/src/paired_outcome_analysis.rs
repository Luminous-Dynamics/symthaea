//! Preregistered, matched analysis of counterfactual evidence interventions.
//!
//! This module estimates fixed-benchmark within-pair contrasts only. It does not
//! infer generalized accuracy, authenticate execution, or collapse endpoints
//! into an evidence/trust scalar. Binary endpoints use an exact McNemar-style
//! conditional test; numerical resource deltas are descriptive only.

use crate::counterfactual_evidence::{EvidenceInterventionPlanV1, verify_evidence_intervention_plan};
use crate::evidence_outcomes::{
    ClaimSupportV1, EvidenceOutcomeReportV1, EvidenceUseV1, OutcomeResponseStatusV1,
    OutcomeTerminalStatusV1, TaskCorrectnessV1, verify_evidence_outcome_report,
};
use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PAIRED_ANALYSIS_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_paired_analysis_plan.v1";
pub const PAIRED_ANALYSIS_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_paired_analysis_report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisEndpointV1 {
    StrictCorrectness,
    FullySupportedClaim,
    RelevantEvidenceUse,
    ExpectedResponse,
    WallClockMs,
    ToolCalls,
    ExperimentCount,
    InputTokens,
    OutputTokens,
    CostMicroUsd,
}

impl AnalysisEndpointV1 {
    fn is_resource(self) -> bool {
        matches!(
            self,
            Self::WallClockMs
                | Self::ToolCalls
                | Self::ExperimentCount
                | Self::InputTokens
                | Self::OutputTokens
                | Self::CostMicroUsd
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionalHypothesisV1 {
    TreatmentHigher,
    TreatmentLower,
    NoDirectionalClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiplicityPolicyV1 {
    HolmAcrossBinaryContrasts,
    UnadjustedExploratory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedResponseV1 {
    Answered,
    Abstained,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedContrastV1 {
    pub contrast_id: String,
    pub treatment_condition: String,
    pub reference_condition: String,
    pub endpoint: AnalysisEndpointV1,
    pub directional_hypothesis: DirectionalHypothesisV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedResponseRuleV1 {
    pub pair_id: String,
    pub condition: String,
    pub expected_response: ExpectedResponseV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedAnalysisPlanV1 {
    pub schema: String,
    /// Full 40-character source commit SHA of the analyzer implementation.
    pub analysis_implementation_revision: String,
    /// Corrected only when explicitly planned before trial outcomes are examined.
    pub multiplicity_policy: MultiplicityPolicyV1,
    pub contrasts: Vec<PlannedContrastV1>,
    /// Required exactly once for each planned pair/condition if any contrast
    /// uses expected_response. This artifact is evaluator-side, never policy-visible.
    pub expected_responses: Vec<ExpectedResponseRuleV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairContrastResultV1 {
    pub pair_id: String,
    pub reference_value: Option<f64>,
    pub treatment_value: Option<f64>,
    pub treatment_minus_reference: Option<f64>,
    pub eligibility_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContrastAnalysisResultV1 {
    pub contrast_id: String,
    pub treatment_condition: String,
    pub reference_condition: String,
    pub endpoint: AnalysisEndpointV1,
    pub directional_hypothesis: DirectionalHypothesisV1,
    pub assigned_pair_count: u64,
    pub eligible_pair_count: u64,
    pub incomplete_pair_count: u64,
    pub ineligible_reason_counts: BTreeMap<String, u64>,
    /// Binary endpoint only; ties count among endpoint-eligible matched pairs.
    pub treatment_wins: Option<u64>,
    pub treatment_losses: Option<u64>,
    pub ties: Option<u64>,
    /// Mean of paired binary treatment-reference differences on eligible pairs.
    pub paired_risk_difference: Option<f64>,
    pub discordant_pair_count: Option<u64>,
    /// Exact two-sided conditional binomial p-value for equal discordant directions.
    pub exact_mcnemar_two_sided_p: Option<f64>,
    pub holm_adjusted_p: Option<f64>,
    /// Wilson 95% interval for treatment wins / all discordant pairs. This is
    /// conditional preference among discordances, not a risk-difference interval.
    pub discordant_win_rate: Option<f64>,
    pub discordant_win_rate_wilson_95_low: Option<f64>,
    pub discordant_win_rate_wilson_95_high: Option<f64>,
    /// Numerical resource endpoints only; descriptive, no p-value or CI.
    pub mean_resource_delta_treatment_minus_reference: Option<f64>,
    pub metric_complete_pair_count: Option<u64>,
    pub metric_incomplete_pair_count: Option<u64>,
    pub pair_results: Vec<PairContrastResultV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedAnalysisReportBodyV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub outcome_report_sha256: String,
    pub outcome_report_commitment_sha256: String,
    pub analysis_plan_sha256: String,
    pub analysis_implementation_revision: String,
    pub multiplicity_policy: MultiplicityPolicyV1,
    pub assigned_pair_count: u64,
    pub contrast_results: Vec<ContrastAnalysisResultV1>,
    pub fixed_benchmark_only: bool,
    pub generalized_accuracy_estimated: bool,
    pub external_anchor_required: bool,
    pub runner_execution_proven: bool,
    pub uncertainty_boundary: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedAnalysisReportV1 {
    pub body: PairedAnalysisReportBodyV1,
    /// SHA-256 of deterministic compact JSON serialization of the report body.
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedAnalysisVerificationV1 {
    pub schema: String,
    pub valid: bool,
    pub plan_binding_valid: bool,
    pub outcome_report_valid: bool,
    pub analysis_plan_hash_matches: bool,
    pub analysis_revision_matches: bool,
    pub issues: Vec<String>,
    pub report: Option<PairedAnalysisReportV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PairConditionKey {
    pair_id: String,
    condition: String,
}

fn valid_commit_sha(value: &str) -> bool {
    value.len() == 40
        && !value.bytes().all(|byte| byte == b'0')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn increment(map: &mut BTreeMap<String, u64>, key: &str) -> Result<(), String> {
    let count = map.entry(key.to_owned()).or_default();
    *count = count
        .checked_add(1)
        .ok_or_else(|| format!("count overflow for {key}"))?;
    Ok(())
}

fn expected_response_lookup(
    analysis: &PairedAnalysisPlanV1,
) -> BTreeMap<PairConditionKey, ExpectedResponseV1> {
    analysis
        .expected_responses
        .iter()
        .map(|rule| {
            (
                PairConditionKey {
                    pair_id: rule.pair_id.clone(),
                    condition: rule.condition.clone(),
                },
                rule.expected_response,
            )
        })
        .collect()
}

fn endpoint_value(
    record: &crate::evidence_outcomes::EvidenceOutcomeRecordV1,
    endpoint: AnalysisEndpointV1,
    expected_response: Option<ExpectedResponseV1>,
) -> Option<f64> {
    use AnalysisEndpointV1 as E;
    if endpoint.is_resource() {
        let value = match endpoint {
            E::WallClockMs => record.resources.wall_clock_ms,
            E::ToolCalls => record.resources.tool_calls,
            E::ExperimentCount => record.resources.experiment_count,
            E::InputTokens => record.resources.input_tokens,
            E::OutputTokens => record.resources.output_tokens,
            E::CostMicroUsd => record.resources.cost_micro_usd,
            _ => None,
        }?;
        return Some(value as f64);
    }

    if record.terminal_status != OutcomeTerminalStatusV1::Completed {
        return None;
    }
    match endpoint {
        E::StrictCorrectness => {
            if record.response_status != OutcomeResponseStatusV1::Answered {
                return None;
            }
            match record.task_correctness {
                TaskCorrectnessV1::Correct => Some(1.0),
                TaskCorrectnessV1::Incorrect | TaskCorrectnessV1::Partial => Some(0.0),
                TaskCorrectnessV1::NotEvaluated => None,
            }
        }
        E::FullySupportedClaim => {
            if record.response_status != OutcomeResponseStatusV1::Answered {
                return None;
            }
            match record.claim_support {
                ClaimSupportV1::FullySupported => Some(1.0),
                ClaimSupportV1::PartiallySupported | ClaimSupportV1::Unsupported => Some(0.0),
                ClaimSupportV1::NotEvaluated => None,
            }
        }
        E::RelevantEvidenceUse => {
            if record.response_status != OutcomeResponseStatusV1::Answered {
                return None;
            }
            match record.evidence_use {
                EvidenceUseV1::UsedRelevant => Some(1.0),
                EvidenceUseV1::IgnoredRelevant | EvidenceUseV1::MisinterpretedRelevant => {
                    Some(0.0)
                }
                EvidenceUseV1::NoRelevantEvidence | EvidenceUseV1::NotEvaluated => None,
            }
        }
        E::ExpectedResponse => {
            let expected = expected_response?;
            let matches = matches!(
                (expected, record.response_status),
                (ExpectedResponseV1::Answered, OutcomeResponseStatusV1::Answered)
                    | (ExpectedResponseV1::Abstained, OutcomeResponseStatusV1::Abstained)
            );
            Some(if matches { 1.0 } else { 0.0 })
        }
        E::WallClockMs
        | E::ToolCalls
        | E::ExperimentCount
        | E::InputTokens
        | E::OutputTokens
        | E::CostMicroUsd => None,
    }
}

fn wilson_interval(successes: u64, trials: u64) -> Option<(f64, f64, f64)> {
    if trials == 0 {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Some((p, (center - half).max(0.0), (center + half).min(1.0)))
}

/// Exact two-sided McNemar p-value, conditional on the discordant-pair count.
fn exact_mcnemar_p(wins: u64, losses: u64) -> f64 {
    let discordant = wins.saturating_add(losses);
    if discordant == 0 {
        return 1.0;
    }
    let k = wins.min(losses);
    let n = discordant as f64;
    let mut log_choose = 0.0_f64;
    for i in 1..=k {
        log_choose += ((discordant - k + i) as f64).ln() - (i as f64).ln();
    }
    let max_log_probability = log_choose - n * 2.0_f64.ln();
    let mut relative_tail_sum = 1.0_f64;
    let mut relative_mass = 1.0_f64;
    let mut i = k;
    while i > 0 {
        relative_mass *= (i as f64) / ((discordant - i + 1) as f64);
        relative_tail_sum += relative_mass;
        i -= 1;
    }
    (2.0 * max_log_probability.exp() * relative_tail_sum).min(1.0)
}

fn outcome_lookup<'a>(
    outcome: &'a EvidenceOutcomeReportV1,
) -> BTreeMap<PairConditionKey, &'a crate::evidence_outcomes::EvidenceOutcomeRecordV1> {
    outcome
        .records
        .iter()
        .map(|record| {
            (
                PairConditionKey {
                    pair_id: record.pair_id.clone(),
                    condition: record.condition.clone(),
                },
                record,
            )
        })
        .collect()
}

fn build_contrast(
    contrast: &PlannedContrastV1,
    pairs: &[crate::counterfactual_evidence::EvidenceInterventionPairV1],
    outcomes: &BTreeMap<PairConditionKey, &crate::evidence_outcomes::EvidenceOutcomeRecordV1>,
    expected: &BTreeMap<PairConditionKey, ExpectedResponseV1>,
) -> Result<ContrastAnalysisResultV1, String> {
    let mut eligible_pair_count = 0_u64;
    let mut incomplete_pair_count = 0_u64;
    let mut ineligible_reason_counts = BTreeMap::new();
    let mut wins = 0_u64;
    let mut losses = 0_u64;
    let mut ties = 0_u64;
    let mut resource_deltas = Vec::new();
    let mut pair_results = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let treatment_key = PairConditionKey {
            pair_id: pair.pair_id.clone(),
            condition: contrast.treatment_condition.clone(),
        };
        let reference_key = PairConditionKey {
            pair_id: pair.pair_id.clone(),
            condition: contrast.reference_condition.clone(),
        };
        let treatment_record = outcomes
            .get(&treatment_key)
            .ok_or_else(|| format!("missing treatment record for pair {}", pair.pair_id))?;
        let reference_record = outcomes
            .get(&reference_key)
            .ok_or_else(|| format!("missing reference record for pair {}", pair.pair_id))?;

        let treatment_value = endpoint_value(
            treatment_record,
            contrast.endpoint,
            expected.get(&treatment_key).copied(),
        );
        let reference_value = endpoint_value(
            reference_record,
            contrast.endpoint,
            expected.get(&reference_key).copied(),
        );
        let (delta, eligibility_reason) = match (reference_value, treatment_value) {
            (Some(reference), Some(treatment)) => {
                eligible_pair_count = eligible_pair_count
                    .checked_add(1)
                    .ok_or_else(|| "eligible pair count overflow".to_owned())?;
                let delta = treatment - reference;
                if !contrast.endpoint.is_resource() {
                    if delta > 0.0 {
                        wins += 1;
                    } else if delta < 0.0 {
                        losses += 1;
                    } else {
                        ties += 1;
                    }
                } else {
                    resource_deltas.push(delta);
                }
                (Some(delta), None)
            }
            (None, None) => {
                incomplete_pair_count += 1;
                increment(&mut ineligible_reason_counts, "both_endpoint_unavailable")?;
                (None, Some("both_endpoint_unavailable".to_owned()))
            }
            (None, Some(_)) => {
                incomplete_pair_count += 1;
                increment(&mut ineligible_reason_counts, "reference_endpoint_unavailable")?;
                (None, Some("reference_endpoint_unavailable".to_owned()))
            }
            (Some(_), None) => {
                incomplete_pair_count += 1;
                increment(&mut ineligible_reason_counts, "treatment_endpoint_unavailable")?;
                (None, Some("treatment_endpoint_unavailable".to_owned()))
            }
        };
        pair_results.push(PairContrastResultV1 {
            pair_id: pair.pair_id.clone(),
            reference_value,
            treatment_value,
            treatment_minus_reference: delta,
            eligibility_reason,
        });
    }

    let assigned = u64::try_from(pairs.len())
        .map_err(|_| "assigned pair count exceeds u64".to_owned())?;
    let is_resource = contrast.endpoint.is_resource();
    let (risk_difference, discordant, p_value, win_rate, win_low, win_high) = if is_resource {
        (None, None, None, None, None, None)
    } else {
        let difference = if eligible_pair_count == 0 {
            None
        } else {
            Some((wins as f64 - losses as f64) / eligible_pair_count as f64)
        };
        let discordant = wins.saturating_add(losses);
        let interval = wilson_interval(wins, discordant);
        (
            difference,
            Some(discordant),
            if eligible_pair_count == 0 {
                None
            } else {
                Some(exact_mcnemar_p(wins, losses))
            },
            interval.map(|value| value.0),
            interval.map(|value| value.1),
            interval.map(|value| value.2),
        )
    };
    let mean_resource_delta = if is_resource && !resource_deltas.is_empty() {
        Some(resource_deltas.iter().sum::<f64>() / resource_deltas.len() as f64)
    } else {
        None
    };
    Ok(ContrastAnalysisResultV1 {
        contrast_id: contrast.contrast_id.clone(),
        treatment_condition: contrast.treatment_condition.clone(),
        reference_condition: contrast.reference_condition.clone(),
        endpoint: contrast.endpoint,
        directional_hypothesis: contrast.directional_hypothesis,
        assigned_pair_count: assigned,
        eligible_pair_count,
        incomplete_pair_count,
        ineligible_reason_counts,
        treatment_wins: if is_resource { None } else { Some(wins) },
        treatment_losses: if is_resource { None } else { Some(losses) },
        ties: if is_resource { None } else { Some(ties) },
        paired_risk_difference: risk_difference,
        discordant_pair_count: discordant,
        exact_mcnemar_two_sided_p: p_value,
        holm_adjusted_p: None,
        discordant_win_rate: win_rate,
        discordant_win_rate_wilson_95_low: win_low,
        discordant_win_rate_wilson_95_high: win_high,
        mean_resource_delta_treatment_minus_reference: mean_resource_delta,
        metric_complete_pair_count: if is_resource { Some(eligible_pair_count) } else { None },
        metric_incomplete_pair_count: if is_resource { Some(incomplete_pair_count) } else { None },
        pair_results,
    })
}

fn apply_holm(results: &mut [ContrastAnalysisResultV1]) {
    let mut indexes = results
        .iter()
        .enumerate()
        .filter_map(|(index, result)| {
            result
                .exact_mcnemar_two_sided_p
                .map(|p| (index, p.clamp(0.0, 1.0)))
        })
        .collect::<Vec<_>>();
    indexes.sort_by(|left, right| left.1.total_cmp(&right.1));
    let m = indexes.len();
    let mut previous = 0.0_f64;
    for (rank, (index, p)) in indexes.into_iter().enumerate() {
        let adjusted = ((m - rank) as f64 * p).min(1.0).max(previous);
        results[index].holm_adjusted_p = Some(adjusted);
        previous = adjusted;
    }
}

fn validate_analysis_plan(
    plan: &EvidenceInterventionPlanV1,
    analysis: &PairedAnalysisPlanV1,
) -> Vec<String> {
    let mut issues = Vec::new();
    if analysis.schema != PAIRED_ANALYSIS_PLAN_SCHEMA {
        issues.push(format!("schema must be {PAIRED_ANALYSIS_PLAN_SCHEMA}"));
    }
    if !valid_commit_sha(&analysis.analysis_implementation_revision) {
        issues.push("analysis_implementation_revision must be a full lowercase Git SHA".to_owned());
    } else if analysis.analysis_implementation_revision != crate::BUILT_GENERATOR_SOURCE_REVISION {
        issues.push(
            "analysis implementation revision does not match the inspected build source revision"
                .to_owned(),
        );
    }
    if analysis.contrasts.is_empty() {
        issues.push("at least one contrast must be preregistered".to_owned());
    }
    let conditions = plan.required_conditions.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let pair_ids = plan.pairs.iter().map(|pair| pair.pair_id.as_str()).collect::<BTreeSet<_>>();
    let uses_expected_response = analysis
        .contrasts
        .iter()
        .any(|contrast| contrast.endpoint == AnalysisEndpointV1::ExpectedResponse);
    let mut contrast_ids = BTreeSet::new();
    for (index, contrast) in analysis.contrasts.iter().enumerate() {
        if contrast.contrast_id.trim().is_empty() || contrast.contrast_id.len() > 128 {
            issues.push(format!("contrasts[{index}].contrast_id must be 1..=128 characters"));
        }
        if !contrast_ids.insert(contrast.contrast_id.as_str()) {
            issues.push(format!("duplicate contrast_id: {}", contrast.contrast_id));
        }
        if contrast.treatment_condition == contrast.reference_condition {
            issues.push(format!(
                "contrast {} compares a condition with itself",
                contrast.contrast_id
            ));
        }
        if !conditions.contains(contrast.treatment_condition.as_str()) {
            issues.push(format!(
                "contrast {} uses an undeclared treatment condition",
                contrast.contrast_id
            ));
        }
        if !conditions.contains(contrast.reference_condition.as_str()) {
            issues.push(format!(
                "contrast {} uses an undeclared reference condition",
                contrast.contrast_id
            ));
        }
    }
    let expected = expected_response_lookup(analysis);
    let mut seen = BTreeSet::new();
    for (index, rule) in analysis.expected_responses.iter().enumerate() {
        if !pair_ids.contains(rule.pair_id.as_str()) {
            issues.push(format!("expected_responses[{index}] references an undeclared pair"));
        }
        if !conditions.contains(rule.condition.as_str()) {
            issues.push(format!("expected_responses[{index}] references an undeclared condition"));
        }
        let key = PairConditionKey {
            pair_id: rule.pair_id.clone(),
            condition: rule.condition.clone(),
        };
        if !seen.insert(key) {
            issues.push(format!(
                "duplicate expected response for pair {} condition {}",
                rule.pair_id, rule.condition
            ));
        }
    }
    if uses_expected_response {
        let expected_count = plan.pairs.len().saturating_mul(plan.required_conditions.len());
        if expected.len() != expected_count {
            issues.push(format!(
                "expected_response endpoint requires an expectation for every pair/condition: \
                 expected {expected_count}, observed {}",
                expected.len()
            ));
        }
        for pair in &plan.pairs {
            for condition in &plan.required_conditions {
                if !expected.contains_key(&PairConditionKey {
                    pair_id: pair.pair_id.clone(),
                    condition: condition.clone(),
                }) {
                    issues.push(format!(
                        "missing expected response for pair {} condition {}",
                        pair.pair_id, condition
                    ));
                }
            }
        }
    }
    issues
}

pub fn analyze_paired_evidence_outcomes(
    plan_bytes: &[u8],
    outcome_bytes: &[u8],
    analysis_plan_bytes: &[u8],
) -> Result<PairedAnalysisVerificationV1, String> {
    let plan_verification = verify_evidence_intervention_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err(format!(
            "intervention plan is invalid: {}",
            plan_verification.issues.join("; ")
        ));
    }
    let outcome_verification = verify_evidence_outcome_report(plan_bytes, outcome_bytes)?;
    let outcome_report: EvidenceOutcomeReportV1 = serde_json::from_slice(outcome_bytes)
        .map_err(|error| format!("cannot parse outcome report: {error}"))?;
    let plan: EvidenceInterventionPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("cannot parse verified intervention plan: {error}"))?;
    let analysis: PairedAnalysisPlanV1 = serde_json::from_slice(analysis_plan_bytes)
        .map_err(|error| format!("cannot parse paired analysis plan: {error}"))?;
    let analysis_plan_sha256 = sha256_hex(analysis_plan_bytes);
    let analysis_plan_hash_matches = analysis_plan_sha256 == plan.analysis_plan_sha256;
    let analysis_revision_matches = valid_commit_sha(&analysis.analysis_implementation_revision)
        && analysis.analysis_implementation_revision == crate::BUILT_GENERATOR_SOURCE_REVISION;
    let mut issues = validate_analysis_plan(&plan, &analysis);
    if !analysis_plan_hash_matches {
        issues.push(
            "analysis plan bytes do not match the digest frozen in the intervention plan"
                .to_owned(),
        );
    }
    if !outcome_verification.valid {
        issues.push("outcome report is invalid and cannot be analyzed".to_owned());
    }
    let plan_binding_valid = plan_verification.valid;
    let outcome_report_valid = outcome_verification.valid;
    let valid = plan_binding_valid
        && outcome_report_valid
        && analysis_plan_hash_matches
        && analysis_revision_matches
        && issues.is_empty();
    let report = if valid {
        let outcomes = outcome_lookup(&outcome_report);
        let expected = expected_response_lookup(&analysis);
        let mut contrasts = analysis
            .contrasts
            .iter()
            .map(|contrast| build_contrast(contrast, &plan.pairs, &outcomes, &expected))
            .collect::<Result<Vec<_>, _>>()?;
        if analysis.multiplicity_policy == MultiplicityPolicyV1::HolmAcrossBinaryContrasts {
            apply_holm(&mut contrasts);
        }
        let body = PairedAnalysisReportBodyV1 {
            schema: PAIRED_ANALYSIS_REPORT_SCHEMA.to_owned(),
            plan_sha256: sha256_hex(plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            outcome_report_sha256: sha256_hex(outcome_bytes),
            outcome_report_commitment_sha256: outcome_report.report_commitment_sha256.clone(),
            analysis_plan_sha256,
            analysis_implementation_revision: analysis.analysis_implementation_revision.clone(),
            multiplicity_policy: analysis.multiplicity_policy,
            assigned_pair_count: u64::try_from(plan.pairs.len())
                .map_err(|_| "assigned pair count exceeds u64".to_owned())?,
            contrast_results: contrasts,
            fixed_benchmark_only: true,
            generalized_accuracy_estimated: false,
            external_anchor_required: true,
            runner_execution_proven: false,
            uncertainty_boundary: "exact conditional McNemar tests for binary matched endpoints; \
                Wilson interval is for discordant-win probability only; resource deltas are \
                descriptive; no generalized-performance interval"
                .to_owned(),
            claim_boundary: "preregistered fixed-benchmark paired contrasts only; evaluator and \
                runner truth remain unauthenticated; no SOTA or generalized-performance claim"
                .to_owned(),
        };
        let bytes = serde_json::to_vec(&body)
            .map_err(|error| format!("cannot serialize paired analysis report: {error}"))?;
        Some(PairedAnalysisReportV1 {
            body,
            report_commitment_sha256: sha256_hex(&bytes),
        })
    } else {
        None
    };
    Ok(PairedAnalysisVerificationV1 {
        schema: "symthaea.science_bench_refresh_paired_analysis_verification.v1".to_owned(),
        valid,
        plan_binding_valid,
        outcome_report_valid,
        analysis_plan_hash_matches,
        analysis_revision_matches,
        issues,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_paired_discordance_test_behaves_sensibly() {
        assert_eq!(exact_mcnemar_p(0, 0), 1.0);
        assert!((exact_mcnemar_p(5, 0) - 0.0625).abs() < 1.0e-12);
        assert!((exact_mcnemar_p(6, 0) - 0.03125).abs() < 1.0e-12);
        assert_eq!(exact_mcnemar_p(3, 3), 1.0);
    }

    #[test]
    fn wilson_interval_is_bounded_and_symmetric_at_half() {
        let (rate, low, high) = wilson_interval(5, 10).unwrap();
        assert_eq!(rate, 0.5);
        assert!(low < rate && high > rate);
        assert!((low + high - 1.0).abs() < 1.0e-12);
        assert!(wilson_interval(0, 0).is_none());
    }

    #[test]
    fn holm_adjustment_is_monotone_and_bounded() {
        let mut results = vec![
            ContrastAnalysisResultV1 {
                contrast_id: "a".to_owned(),
                treatment_condition: "t".to_owned(),
                reference_condition: "r".to_owned(),
                endpoint: AnalysisEndpointV1::StrictCorrectness,
                directional_hypothesis: DirectionalHypothesisV1::TreatmentHigher,
                assigned_pair_count: 10,
                eligible_pair_count: 10,
                incomplete_pair_count: 0,
                ineligible_reason_counts: BTreeMap::new(),
                treatment_wins: Some(5),
                treatment_losses: Some(1),
                ties: Some(4),
                paired_risk_difference: Some(0.4),
                discordant_pair_count: Some(6),
                exact_mcnemar_two_sided_p: Some(0.03125),
                holm_adjusted_p: None,
                discordant_win_rate: Some(5.0 / 6.0),
                discordant_win_rate_wilson_95_low: Some(0.0),
                discordant_win_rate_wilson_95_high: Some(1.0),
                mean_resource_delta_treatment_minus_reference: None,
                metric_complete_pair_count: None,
                metric_incomplete_pair_count: None,
                pair_results: Vec::new(),
            },
            ContrastAnalysisResultV1 {
                contrast_id: "b".to_owned(),
                treatment_condition: "t".to_owned(),
                reference_condition: "r".to_owned(),
                endpoint: AnalysisEndpointV1::FullySupportedClaim,
                directional_hypothesis: DirectionalHypothesisV1::TreatmentHigher,
                assigned_pair_count: 10,
                eligible_pair_count: 10,
                incomplete_pair_count: 0,
                ineligible_reason_counts: BTreeMap::new(),
                treatment_wins: Some(2),
                treatment_losses: Some(1),
                ties: Some(7),
                paired_risk_difference: Some(0.1),
                discordant_pair_count: Some(3),
                exact_mcnemar_two_sided_p: Some(0.25),
                holm_adjusted_p: None,
                discordant_win_rate: Some(2.0 / 3.0),
                discordant_win_rate_wilson_95_low: Some(0.0),
                discordant_win_rate_wilson_95_high: Some(1.0),
                mean_resource_delta_treatment_minus_reference: None,
                metric_complete_pair_count: None,
                metric_incomplete_pair_count: None,
                pair_results: Vec::new(),
            },
        ];
        apply_holm(&mut results);
        let first = results[0].holm_adjusted_p.unwrap();
        let second = results[1].holm_adjusted_p.unwrap();
        assert!(first <= second);
        assert!(first >= 0.0 && second <= 1.0);
        assert!((first - 0.0625).abs() < 1.0e-12);
        assert!((second - 0.25).abs() < 1.0e-12);
    }
}
