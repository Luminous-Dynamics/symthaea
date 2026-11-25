//! Selective temporal grounding and reason-specific abstention audit.
//!
//! The module reuses the production temporal scorer for every case. It keeps
//! answer, explicit absence claims, ambiguity abstention and not-estimable
//! abstention distinct; reports risk/coverage and calibration without
//! collapsing them into a single quality score.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_grounding::{
    score_temporal_grounding, validate_temporal_ground_truth,
    validate_temporal_grounding_prediction, GroundingPredictionStateV1,
    GroundTruthStateV1, TemporalGroundTruthV1, TemporalGroundingOutcomeV1,
    TemporalGroundingPlanV1, TemporalGroundingPredictionV1, TemporalGroundingScoreV1,
};

const CASE_SCHEMA_V1: &str = "mel-selective-temporal-case-v1";
const BATCH_SCHEMA_V1: &str = "mel-selective-temporal-batch-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectiveDecisionV1 {
    Answer,
    ClaimAbsent,
    AbstainAmbiguous,
    AbstainNotEstimable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IdentifiabilityBasisV1 {
    ClearPositive,
    ClearAbsence,
    AnnotatorDisagreement,
    TaskAmbiguity,
    SignalInsufficient,
    UnsupportedQuery,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveInferenceCostV1 {
    /// Total model forwards used to produce the selected decision and confidence.
    pub forward_passes: u32,
    /// End-to-end selective-policy wall time for this exact query and runtime.
    pub selective_wall_time_ms: u64,
    /// Matched single-forward baseline wall time for the same query/audio/runtime.
    pub one_pass_baseline_wall_time_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveGroundingCaseV1 {
    pub schema: String,
    pub case_id: String,
    /// Stable source-work identity for source-disjoint reporting.
    pub source_lineage_id: String,
    /// Examples: clear, dense_polyphony, masked, cropped, ambiguous_boundary.
    pub evidence_stratum: String,
    /// Why the ground-truth disposition is identifiable or not identifiable.
    pub identifiability_basis: IdentifiabilityBasisV1,
    /// Frozen adjudication-policy identity; raw annotation records live in #13560.
    pub adjudication_profile_sha256: String,
    pub inference_cost: SelectiveInferenceCostV1,
    pub truth: TemporalGroundTruthV1,
    /// Prediction disposition after applying the selective decision.
    pub prediction: TemporalGroundingPredictionV1,
    /// Raw candidate that would be emitted with the abstention policy disabled.
    pub unconditional_prediction: TemporalGroundingPredictionV1,
    pub decision: SelectiveDecisionV1,
    /// Model's probability that its selected decision is correct, not confidence
    /// in a private reasoning trace. This remains a self-reported model output.
    pub decision_confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveGroundingBatchV1 {
    pub schema: String,
    pub evaluation_id: String,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub hardware_identity_sha256: String,
    pub plan: TemporalGroundingPlanV1,
    pub cases: Vec<SelectiveGroundingCaseV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectiveCaseResultV1 {
    pub case_id: String,
    pub source_lineage_id: String,
    pub evidence_stratum: String,
    pub identifiability_basis: IdentifiabilityBasisV1,
    pub truth_state: GroundTruthStateV1,
    pub decision: SelectiveDecisionV1,
    pub decision_confidence: f64,
    pub inference_cost: SelectiveInferenceCostV1,
    pub latency_multiplier: f64,
    pub decision_correct: bool,
    pub unconditional_answer_correct: bool,
    pub covered: bool,
    pub selective_risk_contribution: Option<f64>,
    pub answer_f1: Option<f64>,
    pub answer_recall: Option<f64>,
    pub count_gated_f1: Option<f64>,
    pub false_absence_claim: bool,
    pub reason_correct: Option<bool>,
    pub prediction: TemporalGroundingPredictionV1,
    pub unconditional_prediction: TemporalGroundingPredictionV1,
    pub score: TemporalGroundingScoreV1,
    pub unconditional_score: TemporalGroundingScoreV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationBinV1 {
    pub lower_inclusive: f64,
    pub upper_exclusive: f64,
    pub count: usize,
    pub mean_confidence: Option<f64>,
    pub observed_decision_accuracy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RiskCoveragePointV1 {
    pub coverage: f64,
    pub risk: f64,
    pub answered_cases: usize,
    pub errors: usize,
    pub minimum_confidence_included: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StratumSelectiveSummaryV1 {
    pub evidence_stratum: String,
    pub cases: usize,
    pub covered_cases: usize,
    pub coverage: Option<f64>,
    pub selective_risk: Option<f64>,
    pub false_absence_claims: usize,
    pub ambiguous_abstentions: usize,
    pub not_estimable_abstentions: usize,
    pub reason_correct_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdentifiabilityBasisSummaryV1 {
    pub basis: IdentifiabilityBasisV1,
    pub cases: usize,
    pub decision_correct: usize,
    pub decision_correct_rate: Option<f64>,
    pub covered_cases: usize,
    pub abstained_cases: usize,
    pub false_absence_claims: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectiveGroundingSummaryV1 {
    pub cases: usize,
    pub source_work_clusters: usize,
    pub covered_cases: usize,
    pub coverage: f64,
    pub selective_errors: usize,
    pub selective_risk: Option<f64>,
    pub selective_accuracy: Option<f64>,
    pub answer_cases: usize,
    pub absent_claim_cases: usize,
    pub abstain_ambiguous_cases: usize,
    pub abstain_not_estimable_cases: usize,
    pub false_absence_claims: usize,
    pub false_absence_claim_rate: f64,
    pub ambiguous_reason_correct: usize,
    pub ambiguous_reason_precision: Option<f64>,
    pub ambiguous_reason_recall: Option<f64>,
    pub not_estimable_reason_correct: usize,
    pub not_estimable_reason_precision: Option<f64>,
    pub not_estimable_reason_recall: Option<f64>,
    pub answer_present_cases: usize,
    pub answer_present_mean_f1: Option<f64>,
    pub answer_present_mean_recall: Option<f64>,
    pub answer_present_mean_count_gated_f1: Option<f64>,
    pub mean_forward_passes: Option<f64>,
    pub p95_forward_passes: Option<f64>,
    pub mean_selective_wall_time_ms: Option<f64>,
    pub p95_selective_wall_time_ms: Option<f64>,
    pub mean_latency_multiplier: Option<f64>,
    pub p95_latency_multiplier: Option<f64>,
    pub decision_confidence_brier: Option<f64>,
    pub decision_confidence_ece: Option<f64>,
    pub calibration_bins: Vec<CalibrationBinV1>,
    pub risk_coverage_curve: Vec<RiskCoveragePointV1>,
    pub area_under_risk_coverage_curve: Option<f64>,
    pub risk_at_80_percent_coverage: Option<f64>,
    pub max_coverage_at_risk_le_5_percent: Option<f64>,
    pub always_answer_coverage: f64,
    pub always_answer_risk: f64,
    /// Expected random-retention risk at the model's observed maximum coverage.
    pub random_abstention_baseline_risk: Option<f64>,
    pub random_abstention_baseline_aurc: Option<f64>,
    pub always_abstain_coverage: f64,
    pub always_abstain_risk: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectiveGroundingReportV1 {
    pub schema: &'static str,
    pub evaluation_id: String,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub hardware_identity_sha256: String,
    pub plan: TemporalGroundingPlanV1,
    pub summary: SelectiveGroundingSummaryV1,
    pub strata: Vec<StratumSelectiveSummaryV1>,
    pub identifiability_basis_summaries: Vec<IdentifiabilityBasisSummaryV1>,
    pub case_results: Vec<SelectiveCaseResultV1>,
    /// The harness validates digest syntax/identity binding, not authenticity.
    pub runtime_identity_authenticated: bool,
    pub hardware_identity_authenticated: bool,
    pub inference_cost_receipts_authenticated: bool,
    pub evidence_receipts_authenticated: bool,
    /// This evaluator has no access to private model reasoning.
    pub model_reasoning_audited: bool,
    /// A no-answer policy has zero coverage and is not treated as successful.
    pub always_abstain_is_a_product_baseline: bool,
}

pub fn audit_selective_batch_json(input: &str) -> Result<SelectiveGroundingReportV1, String> {
    let batch: SelectiveGroundingBatchV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid selective-grounding batch JSON: {error}"))?;
    audit_selective_batch(&batch)
}

pub fn audit_selective_batch(
    batch: &SelectiveGroundingBatchV1,
) -> Result<SelectiveGroundingReportV1, String> {
    if batch.schema != BATCH_SCHEMA_V1 {
        return Err(format!("unsupported selective-grounding batch schema: {}", batch.schema));
    }
    require_text(&batch.evaluation_id, "evaluation_id")?;
    validate_digest(&batch.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&batch.runtime_identity_sha256, "runtime_identity_sha256")?;
    validate_digest(&batch.decoding_config_sha256, "decoding_config_sha256")?;
    validate_digest(&batch.hardware_identity_sha256, "hardware_identity_sha256")?;
    if batch.cases.is_empty() {
        return Err("selective-grounding batch must contain at least one case".into());
    }

    let mut case_ids = BTreeSet::new();
    let mut results = Vec::with_capacity(batch.cases.len());
    for case in &batch.cases {
        results.push(score_case(case, &batch.plan, &mut case_ids)?);
    }

    let summary = summarize(&results);
    let strata = summarize_strata(&results);
    let identifiability_basis_summaries = summarize_identifiability_basis(&results);

    Ok(SelectiveGroundingReportV1 {
        schema: "mel-selective-temporal-report-v1",
        evaluation_id: batch.evaluation_id.clone(),
        model_identity_sha256: batch.model_identity_sha256.clone(),
        runtime_identity_sha256: batch.runtime_identity_sha256.clone(),
        decoding_config_sha256: batch.decoding_config_sha256.clone(),
        hardware_identity_sha256: batch.hardware_identity_sha256.clone(),
        plan: batch.plan.clone(),
        summary,
        strata,
        identifiability_basis_summaries,
        case_results: results,
        runtime_identity_authenticated: false,
        hardware_identity_authenticated: false,
        inference_cost_receipts_authenticated: false,
        evidence_receipts_authenticated: false,
        model_reasoning_audited: false,
        always_abstain_is_a_product_baseline: false,
    })
}

fn score_case(
    case: &SelectiveGroundingCaseV1,
    plan: &TemporalGroundingPlanV1,
    case_ids: &mut BTreeSet<String>,
) -> Result<SelectiveCaseResultV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!("unsupported case schema for {}: {}", case.case_id, case.schema));
    }
    require_text(&case.case_id, "case_id")?;
    require_text(&case.source_lineage_id, "source_lineage_id")?;
    require_text(&case.evidence_stratum, "evidence_stratum")?;
    validate_digest(&case.adjudication_profile_sha256, "adjudication_profile_sha256")?;
    if !case.decision_confidence.is_finite()
        || !(0.0..=1.0).contains(&case.decision_confidence)
    {
        return Err(format!("case {} has confidence outside [0,1]", case.case_id));
    }
    if case.inference_cost.forward_passes == 0
        || case.inference_cost.selective_wall_time_ms == 0
        || case.inference_cost.one_pass_baseline_wall_time_ms == 0
    {
        return Err(format!("case {} has invalid inference-cost measurements", case.case_id));
    }
    if !case_ids.insert(case.case_id.clone()) {
        return Err(format!("duplicate selective-grounding case_id: {}", case.case_id));
    }

    let basis_matches_truth = matches!(
        (case.truth.state, case.identifiability_basis),
        (GroundTruthStateV1::Present, IdentifiabilityBasisV1::ClearPositive)
            | (GroundTruthStateV1::Absent, IdentifiabilityBasisV1::ClearAbsence)
            | (
                GroundTruthStateV1::Ambiguous,
                IdentifiabilityBasisV1::AnnotatorDisagreement
                    | IdentifiabilityBasisV1::TaskAmbiguity
            )
            | (
                GroundTruthStateV1::NotEstimable,
                IdentifiabilityBasisV1::SignalInsufficient
                    | IdentifiabilityBasisV1::UnsupportedQuery
            )
    );
    if !basis_matches_truth {
        return Err(format!(
            "case {} identifiability basis {:?} contradicts adjudicated state {:?}",
            case.case_id, case.identifiability_basis, case.truth.state
        ));
    }

    let truth_issues = validate_temporal_ground_truth(&case.truth);
    if !truth_issues.is_empty() {
        return Err(format!("case {} has invalid ground truth: {truth_issues:?}", case.case_id));
    }
    let prediction_issues = validate_temporal_grounding_prediction(&case.prediction);
    if !prediction_issues.is_empty() {
        return Err(format!("case {} has invalid prediction: {prediction_issues:?}", case.case_id));
    }
    let unconditional_issues =
        validate_temporal_grounding_prediction(&case.unconditional_prediction);
    if !unconditional_issues.is_empty() {
        return Err(format!(
            "case {} has invalid unconditional prediction: {unconditional_issues:?}",
            case.case_id
        ));
    }
    if case.unconditional_prediction.query_id != case.truth.query_id
        || case.unconditional_prediction.artifact != case.truth.artifact
        || !matches!(
            &case.unconditional_prediction.state,
            GroundingPredictionStateV1::Occurrences { .. }
                | GroundingPredictionStateV1::QueryAbsent
        )
    {
        return Err(format!(
            "case {} unconditional prediction must be an answer/absence candidate for the exact query and artifact",
            case.case_id
        ));
    }

    match (&case.decision, &case.prediction.state) {
        (SelectiveDecisionV1::Answer, GroundingPredictionStateV1::Occurrences { .. })
        | (SelectiveDecisionV1::ClaimAbsent, GroundingPredictionStateV1::QueryAbsent)
        | (SelectiveDecisionV1::AbstainAmbiguous, GroundingPredictionStateV1::Ambiguous { .. })
        | (SelectiveDecisionV1::AbstainNotEstimable, GroundingPredictionStateV1::NotEstimable { .. }) => {}
        (decision, prediction) => {
            return Err(format!(
                "case {} decision {:?} contradicts prediction disposition {:?}",
                case.case_id, decision, prediction
            ));
        }
    }

    if matches!(case.decision, SelectiveDecisionV1::Answer | SelectiveDecisionV1::ClaimAbsent)
        && case.prediction != case.unconditional_prediction
    {
        return Err(format!(
            "case {} covered decision must use the same prediction as its unconditional candidate",
            case.case_id
        ));
    }
    let score = score_temporal_grounding(&case.truth, &case.prediction, plan)
        .map_err(|issues| format!("case {} rejected by production scorer: {issues:?}", case.case_id))?;
    let unconditional_score =
        score_temporal_grounding(&case.truth, &case.unconditional_prediction, plan)
            .map_err(|issues| format!("case {} unconditional candidate rejected by production scorer: {issues:?}", case.case_id))?;

    let reason_correct = match case.truth.state {
        GroundTruthStateV1::Ambiguous => {
            Some(case.decision == SelectiveDecisionV1::AbstainAmbiguous)
        }
        GroundTruthStateV1::NotEstimable => {
            Some(case.decision == SelectiveDecisionV1::AbstainNotEstimable)
        }
        _ => None,
    };
    let decision_correct = match case.truth.state {
        GroundTruthStateV1::Present => {
            case.decision == SelectiveDecisionV1::Answer && is_perfect_present(&score)
        }
        GroundTruthStateV1::Absent => {
            case.decision == SelectiveDecisionV1::ClaimAbsent
                && score.correct_absence_claim == Some(true)
        }
        GroundTruthStateV1::Ambiguous => {
            case.decision == SelectiveDecisionV1::AbstainAmbiguous
        }
        GroundTruthStateV1::NotEstimable => {
            case.decision == SelectiveDecisionV1::AbstainNotEstimable
        }
    };
    let unconditional_answer_correct = match case.truth.state {
        GroundTruthStateV1::Present => is_perfect_present(&unconditional_score),
        GroundTruthStateV1::Absent => {
            unconditional_score.correct_absence_claim == Some(true)
        }
        GroundTruthStateV1::Ambiguous | GroundTruthStateV1::NotEstimable => false,
    };
    let covered = matches!(
        case.decision,
        SelectiveDecisionV1::Answer | SelectiveDecisionV1::ClaimAbsent
    );
    let false_absence_claim = case.decision == SelectiveDecisionV1::ClaimAbsent
        && case.truth.state != GroundTruthStateV1::Absent;
    let answer_present = case.truth.state == GroundTruthStateV1::Present
        && case.decision == SelectiveDecisionV1::Answer;

    Ok(SelectiveCaseResultV1 {
        case_id: case.case_id.clone(),
        source_lineage_id: case.source_lineage_id.clone(),
        evidence_stratum: case.evidence_stratum.clone(),
        identifiability_basis: case.identifiability_basis,
        truth_state: case.truth.state,
        decision: case.decision,
        decision_confidence: case.decision_confidence,
        inference_cost: case.inference_cost.clone(),
        latency_multiplier: case.inference_cost.selective_wall_time_ms as f64
            / case.inference_cost.one_pass_baseline_wall_time_ms as f64,
        decision_correct,
        unconditional_answer_correct,
        covered,
        selective_risk_contribution: covered.then_some(if decision_correct { 0.0 } else { 1.0 }),
        answer_f1: answer_present.then_some(score.f1.unwrap_or(0.0)),
        answer_recall: answer_present.then_some(score.recall.unwrap_or(0.0)),
        count_gated_f1: answer_present.then_some(if score.exact_occurrence_count == Some(true) {
            score.f1.unwrap_or(0.0)
        } else if score.ground_truth_occurrences > 1 {
            0.0
        } else {
            score.f1.unwrap_or(0.0)
        }),
        false_absence_claim,
        reason_correct,
        prediction: case.prediction.clone(),
        unconditional_prediction: case.unconditional_prediction.clone(),
        score,
        unconditional_score,
    })
}

fn is_perfect_present(score: &TemporalGroundingScoreV1) -> bool {
    score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery
        && score.f1 == Some(1.0)
        && score.true_positives == score.ground_truth_occurrences
        && score.false_positives == 0
        && score.false_negatives == 0
        && score.exact_occurrence_count == Some(true)
}

fn summarize(results: &[SelectiveCaseResultV1]) -> SelectiveGroundingSummaryV1 {
    let cases = results.len();
    let covered: Vec<_> = results.iter().filter(|item| item.covered).collect();
    let selective_errors = covered.iter().filter(|item| !item.decision_correct).count();
    let answer_cases = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::Answer)
        .count();
    let absent_claim_cases = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::ClaimAbsent)
        .count();
    let abstain_ambiguous_cases = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::AbstainAmbiguous)
        .count();
    let abstain_not_estimable_cases = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::AbstainNotEstimable)
        .count();
    let false_absence_claims = results.iter().filter(|item| item.false_absence_claim).count();
    let non_absent_cases = results
        .iter()
        .filter(|item| item.truth_state != GroundTruthStateV1::Absent)
        .count();
    let forward_passes = results.iter()
        .map(|item| item.inference_cost.forward_passes as f64)
        .collect::<Vec<_>>();
    let selective_latency = results.iter()
        .map(|item| item.inference_cost.selective_wall_time_ms as f64)
        .collect::<Vec<_>>();
    let latency_multipliers = results.iter()
        .map(|item| item.latency_multiplier)
        .collect::<Vec<_>>();
    let decision_confidence_brier = mean(
        &results
            .iter()
            .map(|item| {
                let target = if item.decision_correct { 1.0 } else { 0.0 };
                (item.decision_confidence - target).powi(2)
            })
            .collect::<Vec<_>>(),
    );
    let calibration_bins = make_calibration_bins(results, 10);
    let decision_confidence_ece = ece(&calibration_bins, cases);

    let mut ranked: Vec<_> = covered.iter().copied().collect();
    ranked.sort_by(|left, right| {
        right
            .decision_confidence
            .total_cmp(&left.decision_confidence)
            .then_with(|| left.case_id.cmp(&right.case_id))
    });
    let risk_coverage_curve = build_risk_coverage_curve(&ranked, cases);
    let area_under_risk_coverage_curve = integrate_aurc(&risk_coverage_curve);
    let risk_at_80_percent_coverage = risk_at_coverage(&risk_coverage_curve, 0.8);
    let max_coverage_at_risk_le_5_percent = risk_coverage_curve
        .iter()
        .filter(|point| point.risk <= 0.05)
        .map(|point| point.coverage)
        .reduce(f64::max);

    let present_answer: Vec<_> = results
        .iter()
        .filter(|item| item.truth_state == GroundTruthStateV1::Present
            && item.decision == SelectiveDecisionV1::Answer)
        .collect();
    let present_scores: Vec<_> = present_answer.iter().filter_map(|item| item.answer_f1).collect();
    let present_recalls: Vec<_> = present_answer.iter().filter_map(|item| item.answer_recall).collect();
    let present_count_gated: Vec<_> = present_answer.iter().filter_map(|item| item.count_gated_f1).collect();

    let ambiguous_predictions = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::AbstainAmbiguous)
        .count();
    let ambiguous_correct = results
        .iter()
        .filter(|item| item.reason_correct == Some(true)
            && item.decision == SelectiveDecisionV1::AbstainAmbiguous)
        .count();
    let ambiguous_truth = results
        .iter()
        .filter(|item| item.truth_state == GroundTruthStateV1::Ambiguous)
        .count();

    let not_estimable_predictions = results
        .iter()
        .filter(|item| item.decision == SelectiveDecisionV1::AbstainNotEstimable)
        .count();
    let not_estimable_correct = results
        .iter()
        .filter(|item| item.reason_correct == Some(true)
            && item.decision == SelectiveDecisionV1::AbstainNotEstimable)
        .count();
    let not_estimable_truth = results
        .iter()
        .filter(|item| item.truth_state == GroundTruthStateV1::NotEstimable)
        .count();

    SelectiveGroundingSummaryV1 {
        cases,
        source_work_clusters: results.iter().map(|item| item.source_lineage_id.as_str())
            .collect::<BTreeSet<_>>().len(),
        covered_cases: covered.len(),
        coverage: covered.len() as f64 / cases as f64,
        selective_errors,
        selective_risk: rate(selective_errors, covered.len()),
        selective_accuracy: rate(covered.len() - selective_errors, covered.len()),
        answer_cases,
        absent_claim_cases,
        abstain_ambiguous_cases,
        abstain_not_estimable_cases,
        false_absence_claims,
        false_absence_claim_rate: rate(false_absence_claims, non_absent_cases).unwrap_or(0.0),
        ambiguous_reason_correct: ambiguous_correct,
        ambiguous_reason_precision: rate(ambiguous_correct, ambiguous_predictions),
        ambiguous_reason_recall: rate(ambiguous_correct, ambiguous_truth),
        not_estimable_reason_correct: not_estimable_correct,
        not_estimable_reason_precision: rate(not_estimable_correct, not_estimable_predictions),
        not_estimable_reason_recall: rate(not_estimable_correct, not_estimable_truth),
        answer_present_cases: present_answer.len(),
        answer_present_mean_f1: mean(&present_scores),
        answer_present_mean_recall: mean(&present_recalls),
        answer_present_mean_count_gated_f1: mean(&present_count_gated),
        mean_forward_passes: mean(&forward_passes),
        p95_forward_passes: percentile(&forward_passes, 0.95),
        mean_selective_wall_time_ms: mean(&selective_latency),
        p95_selective_wall_time_ms: percentile(&selective_latency, 0.95),
        mean_latency_multiplier: mean(&latency_multipliers),
        p95_latency_multiplier: percentile(&latency_multipliers, 0.95),
        decision_confidence_brier,
        decision_confidence_ece,
        calibration_bins,
        risk_coverage_curve,
        area_under_risk_coverage_curve,
        risk_at_80_percent_coverage,
        max_coverage_at_risk_le_5_percent,
        always_answer_coverage: 1.0,
        always_answer_risk: rate(
            results.iter().filter(|item| !item.unconditional_answer_correct).count(),
            cases,
        ).unwrap_or(1.0),
        random_abstention_baseline_risk: rate(
            results.iter().filter(|item| !item.unconditional_answer_correct).count(),
            cases,
        ),
        random_abstention_baseline_aurc: rate(
            results.iter().filter(|item| !item.unconditional_answer_correct).count(),
            cases,
        ),
        always_abstain_coverage: 0.0,
        always_abstain_risk: None,
    }
}

fn summarize_strata(results: &[SelectiveCaseResultV1]) -> Vec<StratumSelectiveSummaryV1> {
    let mut groups: BTreeMap<&str, Vec<&SelectiveCaseResultV1>> = BTreeMap::new();
    for result in results {
        groups.entry(result.evidence_stratum.as_str()).or_default().push(result);
    }
    groups
        .into_iter()
        .map(|(stratum, items)| {
            let covered: Vec<_> = items.iter().filter(|item| item.covered).collect();
            let errors = covered.iter().filter(|item| !item.decision_correct).count();
            let reasons: Vec<_> = items.iter().filter_map(|item| item.reason_correct).collect();
            StratumSelectiveSummaryV1 {
                evidence_stratum: stratum.to_owned(),
                cases: items.len(),
                covered_cases: covered.len(),
                coverage: rate(covered.len(), items.len()),
                selective_risk: rate(errors, covered.len()),
                false_absence_claims: items.iter().filter(|item| item.false_absence_claim).count(),
                ambiguous_abstentions: items
                    .iter()
                    .filter(|item| item.decision == SelectiveDecisionV1::AbstainAmbiguous)
                    .count(),
                not_estimable_abstentions: items
                    .iter()
                    .filter(|item| item.decision == SelectiveDecisionV1::AbstainNotEstimable)
                    .count(),
                reason_correct_rate: mean(&reasons.iter().map(|value| if *value { 1.0 } else { 0.0 }).collect::<Vec<_>>()),
            }
        })
        .collect()
}

fn summarize_identifiability_basis(
    results: &[SelectiveCaseResultV1],
) -> Vec<IdentifiabilityBasisSummaryV1> {
    let mut groups: BTreeMap<IdentifiabilityBasisV1, Vec<&SelectiveCaseResultV1>> =
        BTreeMap::new();
    for result in results {
        groups
            .entry(result.identifiability_basis)
            .or_default()
            .push(result);
    }
    groups
        .into_iter()
        .map(|(basis, items)| {
            let decision_correct = items.iter().filter(|item| item.decision_correct).count();
            let covered_cases = items.iter().filter(|item| item.covered).count();
            IdentifiabilityBasisSummaryV1 {
                basis,
                cases: items.len(),
                decision_correct,
                decision_correct_rate: rate(decision_correct, items.len()),
                covered_cases,
                abstained_cases: items.len() - covered_cases,
                false_absence_claims: items.iter().filter(|item| item.false_absence_claim).count(),
            }
        })
        .collect()
}

fn make_calibration_bins(
    results: &[SelectiveCaseResultV1],
    bin_count: usize,
) -> Vec<CalibrationBinV1> {
    (0..bin_count)
        .map(|index| {
            let lower = index as f64 / bin_count as f64;
            let upper = (index + 1) as f64 / bin_count as f64;
            let items: Vec<_> = results
                .iter()
                .filter(|item| {
                    item.decision_confidence >= lower
                        && (item.decision_confidence < upper
                            || (index + 1 == bin_count && item.decision_confidence <= upper))
                })
                .collect();
            CalibrationBinV1 {
                lower_inclusive: lower,
                upper_exclusive: upper,
                count: items.len(),
                mean_confidence: mean(&items.iter().map(|item| item.decision_confidence).collect::<Vec<_>>()),
                observed_decision_accuracy: rate(
                    items.iter().filter(|item| item.decision_correct).count(),
                    items.len(),
                ),
            }
        })
        .collect()
}

fn ece(bins: &[CalibrationBinV1], total: usize) -> Option<f64> {
    if total == 0 {
        return None;
    }
    Some(
        bins.iter()
            .filter(|bin| bin.count > 0)
            .map(|bin| {
                (bin.count as f64 / total as f64)
                    * (bin.mean_confidence.unwrap_or(0.0)
                        - bin.observed_decision_accuracy.unwrap_or(0.0))
                        .abs()
            })
            .sum(),
    )
}

fn build_risk_coverage_curve(
    ranked_covered: &[&SelectiveCaseResultV1],
    total_cases: usize,
) -> Vec<RiskCoveragePointV1> {
    if ranked_covered.is_empty() || total_cases == 0 {
        return Vec::new();
    }
    let mut points = vec![RiskCoveragePointV1 {
        coverage: 0.0,
        risk: 0.0,
        answered_cases: 0,
        errors: 0,
        minimum_confidence_included: 1.0,
    }];
    let mut answered = 0_usize;
    let mut errors = 0_usize;
    let mut index = 0_usize;
    while index < ranked_covered.len() {
        let confidence = ranked_covered[index].decision_confidence;
        while index < ranked_covered.len()
            && (ranked_covered[index].decision_confidence - confidence).abs() <= f64::EPSILON
        {
            answered += 1;
            if !ranked_covered[index].decision_correct {
                errors += 1;
            }
            index += 1;
        }
        points.push(RiskCoveragePointV1 {
            coverage: answered as f64 / total_cases as f64,
            risk: errors as f64 / answered as f64,
            answered_cases: answered,
            errors,
            minimum_confidence_included: confidence,
        });
    }
    points
}

fn integrate_aurc(points: &[RiskCoveragePointV1]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let mut area = 0.0;
    for pair in points.windows(2) {
        let width = pair[1].coverage - pair[0].coverage;
        area += width * (pair[0].risk + pair[1].risk) / 2.0;
    }
    Some(area)
}

fn risk_at_coverage(points: &[RiskCoveragePointV1], target: f64) -> Option<f64> {
    let last = points.last()?;
    if last.coverage + f64::EPSILON < target {
        return None;
    }
    if points.len() == 1 {
        return None;
    }
    for pair in points.windows(2) {
        if target >= pair[0].coverage && target <= pair[1].coverage {
            let width = pair[1].coverage - pair[0].coverage;
            if width <= f64::EPSILON {
                return Some(pair[1].risk);
            }
            let fraction = (target - pair[0].coverage) / width;
            return Some(pair[0].risk + fraction * (pair[1].risk - pair[0].risk));
        }
    }
    Some(last.risk)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::{
        AudioArtifactClockV1, FrameIntervalV1, GroundingPredictionStateV1,
        TEMPORAL_GROUNDING_EVALUATOR_V1, TEMPORAL_GROUNDING_SCHEMA_V1,
    };

    const AUDIO: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ANNOTATION: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const MODEL: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const RUNTIME: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DECODING: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const POLICY: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn artifact() -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: AUDIO.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 480_000,
        }
    }

    fn truth(state: GroundTruthStateV1, intervals: Vec<crate::temporal_grounding::FrameIntervalV1>, reason: Option<&str>) -> TemporalGroundTruthV1 {
        TemporalGroundTruthV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-piano".into(),
            query_text: "Every piano entrance".into(),
            artifact: artifact(),
            annotation_policy_version: "selective-adjudication-v1".into(),
            annotation_set_sha256: ANNOTATION.into(),
            annotator_count: 3,
            state,
            intervals,
            status_reason: reason.map(str::to_owned),
        }
    }

    fn prediction(state: GroundingPredictionStateV1) -> TemporalGroundingPredictionV1 {
        TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-piano".into(),
            artifact: artifact(),
            state,
        }
    }

    fn case(
        id: &str,
        state: GroundTruthStateV1,
        intervals: Vec<crate::temporal_grounding::FrameIntervalV1>,
        reason: Option<&str>,
        decision: SelectiveDecisionV1,
        predicted: GroundingPredictionStateV1,
        confidence: f64,
    ) -> SelectiveGroundingCaseV1 {
        let selected_prediction = prediction(predicted);
        let unconditional_prediction = if matches!(
            decision,
            SelectiveDecisionV1::Answer | SelectiveDecisionV1::ClaimAbsent
        ) {
            selected_prediction.clone()
        } else {
            prediction(GroundingPredictionStateV1::QueryAbsent)
        };
        SelectiveGroundingCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            case_id: id.into(),
            source_lineage_id: format!("lineage-{id}"),
            identifiability_basis: match state {
                GroundTruthStateV1::Present => IdentifiabilityBasisV1::ClearPositive,
                GroundTruthStateV1::Absent => IdentifiabilityBasisV1::ClearAbsence,
                GroundTruthStateV1::Ambiguous => IdentifiabilityBasisV1::TaskAmbiguity,
                GroundTruthStateV1::NotEstimable => IdentifiabilityBasisV1::SignalInsufficient,
            },
            evidence_stratum: match state {
                GroundTruthStateV1::Present => "clear_or_localized".into(),
                GroundTruthStateV1::Absent => "genuine_absence".into(),
                GroundTruthStateV1::Ambiguous => "ambiguous_boundary".into(),
                GroundTruthStateV1::NotEstimable => "not_estimable".into(),
            },
            adjudication_profile_sha256: POLICY.into(),
            inference_cost: SelectiveInferenceCostV1 {
                forward_passes: if confidence > 0.85 { 1 } else { 3 },
                selective_wall_time_ms: if confidence > 0.85 { 100 } else { 180 },
                one_pass_baseline_wall_time_ms: 100,
            },
            truth: truth(state, intervals, reason),
            prediction: selected_prediction,
            unconditional_prediction,
            decision,
            decision_confidence: confidence,
        }
    }

    fn batch(cases: Vec<SelectiveGroundingCaseV1>) -> SelectiveGroundingBatchV1 {
        SelectiveGroundingBatchV1 {
            schema: BATCH_SCHEMA_V1.into(),
            evaluation_id: "selective-fixture-v1".into(),
            model_identity_sha256: MODEL.into(),
            runtime_identity_sha256: RUNTIME.into(),
            decoding_config_sha256: DECODING.into(),
            hardware_identity_sha256: "1111111111111111111111111111111111111111111111111111111111111111".into(),
            plan: TemporalGroundingPlanV1 {
                evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
                minimum_interval_iou: 0.5,
            },
            cases,
        }
    }

    #[test]
    fn clear_present_answer_and_true_absence_are_covered_and_correct() {
        let present = case(
            "present",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            },
            0.95,
        );
        let absent = case(
            "absent",
            GroundTruthStateV1::Absent,
            vec![],
            None,
            SelectiveDecisionV1::ClaimAbsent,
            GroundingPredictionStateV1::QueryAbsent,
            0.9,
        );
        let report = audit_selective_batch(&batch(vec![present, absent])).unwrap();
        assert_eq!(report.summary.coverage, 1.0);
        assert_eq!(report.summary.selective_risk, Some(0.0));
        assert_eq!(report.summary.answer_cases, 1);
        assert_eq!(report.summary.absent_claim_cases, 1);
        assert_eq!(report.summary.false_absence_claims, 0);
    }

    #[test]
    fn ambiguous_and_not_estimable_require_their_own_abstention_reasons() {
        let ambiguous = case(
            "amb",
            GroundTruthStateV1::Ambiguous,
            vec![],
            Some("gradual entrance"),
            SelectiveDecisionV1::AbstainAmbiguous,
            GroundingPredictionStateV1::Ambiguous {
                candidate_intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 90, end_frame: 210 }],
                reason: "two plausible entrances".into(),
            },
            0.8,
        );
        let not_estimable = case(
            "unobs",
            GroundTruthStateV1::NotEstimable,
            vec![],
            Some("masking prevents distinction"),
            SelectiveDecisionV1::AbstainNotEstimable,
            GroundingPredictionStateV1::NotEstimable {
                reason: "insufficient signal".into(),
            },
            0.85,
        );
        let report = audit_selective_batch(&batch(vec![ambiguous, not_estimable])).unwrap();
        assert_eq!(report.summary.coverage, 0.0);
        assert_eq!(report.summary.ambiguous_reason_recall, Some(1.0));
        assert_eq!(report.summary.not_estimable_reason_recall, Some(1.0));
        assert_eq!(report.summary.selective_risk, None);
        assert_eq!(report.summary.always_abstain_coverage, 0.0);
    }

    #[test]
    fn false_absence_claim_on_present_or_unobservable_case_is_counted() {
        let wrong_present = case(
            "wrong-present",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::ClaimAbsent,
            GroundingPredictionStateV1::QueryAbsent,
            0.99,
        );
        let wrong_unobservable = case(
            "wrong-unobservable",
            GroundTruthStateV1::NotEstimable,
            vec![],
            Some("masked"),
            SelectiveDecisionV1::ClaimAbsent,
            GroundingPredictionStateV1::QueryAbsent,
            0.99,
        );
        let report = audit_selective_batch(&batch(vec![wrong_present, wrong_unobservable])).unwrap();
        assert_eq!(report.summary.false_absence_claims, 2);
        assert_eq!(report.summary.false_absence_claim_rate, 1.0);
        assert_eq!(report.summary.selective_risk, Some(1.0));
    }

    #[test]
    fn confident_wrong_answer_is_visible_in_risk_and_brier_score() {
        let wrong = case(
            "confident-wrong",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 300, end_frame: 400 }],
            },
            0.99,
        );
        let report = audit_selective_batch(&batch(vec![wrong])).unwrap();
        assert_eq!(report.summary.selective_risk, Some(1.0));
        assert!(report.summary.decision_confidence_brier.unwrap() > 0.9);
    }

    #[test]
    fn always_abstain_cannot_pass_as_success_because_coverage_is_zero() {
        let abstain = case(
            "abstain-clear",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::AbstainNotEstimable,
            GroundingPredictionStateV1::NotEstimable {
                reason: "model chose not to answer".into(),
            },
            0.9,
        );
        let report = audit_selective_batch(&batch(vec![abstain])).unwrap();
        assert_eq!(report.summary.coverage, 0.0);
        assert_eq!(report.summary.risk_coverage_curve.len(), 0);
        assert_eq!(report.summary.area_under_risk_coverage_curve, None);
    }

    #[test]
    fn logged_unconditional_candidate_can_be_correct_even_when_policy_abstains() {
        let mut abstain = case(
            "abstain-with-correct-candidate",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
            None,
            SelectiveDecisionV1::AbstainNotEstimable,
            GroundingPredictionStateV1::NotEstimable {
                reason: "policy abstained".into(),
            },
            0.6,
        );
        abstain.unconditional_prediction = prediction(
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
            },
        );
        let report = audit_selective_batch(&batch(vec![abstain])).unwrap();
        assert_eq!(report.summary.coverage, 0.0);
        assert_eq!(report.summary.always_answer_risk, 0.0);
        assert_eq!(
            report.case_results[0].unconditional_score.f1,
            Some(1.0)
        );
        assert_eq!(report.case_results[0].unconditional_answer_correct, true);
    }

    #[test]
    fn covered_decision_must_match_its_unconditional_candidate() {
        let mut selected = case(
            "mismatched-candidate",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
            },
            0.9,
        );
        selected.unconditional_prediction = prediction(GroundingPredictionStateV1::QueryAbsent);
        assert!(audit_selective_batch(&batch(vec![selected])).is_err());
    }

    #[test]
    fn always_answer_and_random_abstention_baselines_are_reported() {
        let right = case(
            "baseline-right",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            },
            0.9,
        );
        let abstain = case(
            "baseline-abstain",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::AbstainNotEstimable,
            GroundingPredictionStateV1::NotEstimable { reason: "not answered".into() },
            0.8,
        );
        let report = audit_selective_batch(&batch(vec![right, abstain])).unwrap();
        assert_eq!(report.summary.coverage, 0.5);
        assert_eq!(report.summary.always_answer_coverage, 1.0);
        assert_eq!(report.summary.always_answer_risk, 0.5);
        assert_eq!(report.summary.random_abstention_baseline_risk, Some(0.5));
        assert_eq!(report.summary.random_abstention_baseline_aurc, Some(0.5));
        assert_eq!(report.summary.always_abstain_coverage, 0.0);
        assert_eq!(report.summary.always_abstain_risk, None);
    }

    #[test]
    fn confidence_ordering_changes_risk_coverage_curve_and_aurc() {
        let right = case(
            "right",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            },
            0.95,
        );
        let wrong = case(
            "wrong",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 300, end_frame: 400 }],
            },
            0.1,
        );
        let report = audit_selective_batch(&batch(vec![right, wrong])).unwrap();
        assert_eq!(report.summary.risk_coverage_curve.len(), 3);
        assert_eq!(report.summary.risk_coverage_curve[1].risk, 0.0);
        assert_eq!(report.summary.risk_coverage_curve[2].risk, 0.5);
        assert!(report.summary.area_under_risk_coverage_curve.unwrap() < 0.5);
    }

    #[test]
    fn strict_batch_json_round_trips_and_rejects_unknown_fields() {
        let valid = batch(vec![case(
            "schema-roundtrip",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
            },
            0.9,
        )]);
        let encoded = serde_json::to_string(&valid).unwrap();
        assert!(audit_selective_batch_json(&encoded).is_ok());

        let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        value["unknown_batch_field"] = serde_json::json!(true);
        assert!(audit_selective_batch_json(&value.to_string()).is_err());

        let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        value["cases"][0]["unknown_case_field"] = serde_json::json!(true);
        assert!(audit_selective_batch_json(&value.to_string()).is_err());
    }

    #[test]
    fn inference_cost_measurements_must_be_positive() {
        let mut invalid = batch(vec![case(
            "zero-cost",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
            },
            0.9,
        )]);
        invalid.cases[0].inference_cost.forward_passes = 0;
        assert!(audit_selective_batch(&invalid).is_err());

        invalid.cases[0].inference_cost.forward_passes = 1;
        invalid.cases[0].inference_cost.one_pass_baseline_wall_time_ms = 0;
        assert!(audit_selective_batch(&invalid).is_err());
    }

    #[test]
    fn confidence_and_case_metadata_are_validated() {
        let mut invalid = batch(vec![case(
            "bad-confidence",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            },
            0.7,
        )]);
        invalid.cases[0].decision_confidence = f64::NAN;
        assert!(audit_selective_batch(&invalid).is_err());

        let mut invalid = batch(vec![case(
            "bad-binding",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::QueryAbsent,
            0.7,
        )]);
        assert!(audit_selective_batch(&invalid).is_err());
    }

    #[test]
    fn annotator_disagreement_and_task_ambiguity_remain_distinct_bases() {
        let mut disagreement = case(
            "disagreement",
            GroundTruthStateV1::Ambiguous,
            vec![],
            Some("annotators disagree on event identity"),
            SelectiveDecisionV1::AbstainAmbiguous,
            GroundingPredictionStateV1::Ambiguous {
                candidate_intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
                reason: "two annotator interpretations".into(),
            },
            0.8,
        );
        disagreement.identifiability_basis = IdentifiabilityBasisV1::AnnotatorDisagreement;

        let task_ambiguity = case(
            "task-ambiguity",
            GroundTruthStateV1::Ambiguous,
            vec![],
            Some("gradual entrance lacks a stable operational boundary"),
            SelectiveDecisionV1::AbstainAmbiguous,
            GroundingPredictionStateV1::Ambiguous {
                candidate_intervals: vec![crate::temporal_grounding::FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                }],
                reason: "several valid segmentations".into(),
            },
            0.8,
        );
        let report = audit_selective_batch(&batch(vec![disagreement, task_ambiguity])).unwrap();
        assert_eq!(report.identifiability_basis_summaries.len(), 2);
        assert_eq!(
            report.identifiability_basis_summaries[0].basis,
            IdentifiabilityBasisV1::AnnotatorDisagreement
        );
        assert_eq!(
            report.identifiability_basis_summaries[1].basis,
            IdentifiabilityBasisV1::TaskAmbiguity
        );
    }

    #[test]
    fn basis_cannot_relabel_signal_insufficiency_as_annotator_disagreement() {
        let mut invalid = case(
            "wrong-basis",
            GroundTruthStateV1::NotEstimable,
            vec![],
            Some("source masked"),
            SelectiveDecisionV1::AbstainNotEstimable,
            GroundingPredictionStateV1::NotEstimable {
                reason: "not enough signal".into(),
            },
            0.8,
        );
        invalid.identifiability_basis = IdentifiabilityBasisV1::AnnotatorDisagreement;
        assert!(audit_selective_batch(&batch(vec![invalid])).is_err());
    }

    #[test]
    fn source_lineage_clusters_are_counted_without_fabricating_population_intervals() {
        let mut first = case(
            "first",
            GroundTruthStateV1::Present,
            vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            None,
            SelectiveDecisionV1::Answer,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![crate::temporal_grounding::FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            },
            0.9,
        );
        let mut second = first.clone();
        second.case_id = "second".into();
        first.source_lineage_id = "same-work".into();
        second.source_lineage_id = "same-work".into();
        let report = audit_selective_batch(&batch(vec![first, second])).unwrap();
        assert_eq!(report.summary.source_work_clusters, 1);
    }
}
