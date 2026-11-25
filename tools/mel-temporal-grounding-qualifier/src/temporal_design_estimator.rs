//! Design-based finite-population estimation for the frozen two-stage audit.
//!
//! Replays the acquisition planner and requires the exact selection receipt.
//! Only probability-audit outcomes are accepted. Active-refinement labels are
//! never pooled into the estimator. The general HT variance estimator uses
//! first- and second-order ultimate-item inclusion probabilities; strata with
//! zero pairwise inclusion support report variance unavailable.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_acquisition::{
    plan_annotation_acquisition, AnnotationAcquisitionDatasetV1,
    AnnotationSelectionRecordV1,
    ProbabilityAuditStratumDesignV1, SelectionStreamV1,
};

const ESTIMATOR_SCHEMA_V1: &str = "mel-two-stage-design-estimator-input-v1";
const ESTIMATOR_REPORT_SCHEMA_V1: &str = "mel-two-stage-design-estimator-report-v1";
const ESTIMATOR_VERSION_V1: &str = "mel-two-stage-ht-estimator-v1";
const PPM_SCALE: u32 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedAuditOutcomeV1 {
    pub item_id: String,
    pub source_work_id_sha256: String,
    pub scoring_report_sha256: String,
    pub model_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    /// The same metric ID must be used for every sampled item.
    pub metric_id: String,
    /// Fixed-point outcome in [0, 1], scaled by 1,000,000.
    pub value_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignEstimatorInputV1 {
    pub schema: String,
    pub dataset: AnnotationAcquisitionDatasetV1,
    /// Must match the deterministic planner output exactly.
    /// Required for reviewed/audited runs. Null is permitted only for a
    /// fresh deterministic demonstration and is explicitly marked unverified.
    pub expected_selection_receipt_sha256: Option<String>,
    pub metric_id: String,
    pub model_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    /// Exactly one observed outcome per selected probability-audit item.
    pub observed_audit_outcomes: Vec<ObservedAuditOutcomeV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum VarianceUnavailableReasonV1 {
    FewerThanTwoSelectedSourceWorks,
    ZeroJointInclusionProbability,
    InsufficientWithinWorkSample,
    NonFiniteVarianceEstimate,
    NegativeVarianceEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StratumDesignEstimateV1 {
    pub source_stratum_id: String,
    pub eligible_source_works: usize,
    pub selected_source_works: usize,
    pub eligible_query_items: usize,
    pub selected_query_items: usize,
    pub estimated_total: f64,
    pub estimated_mean: f64,
    pub estimated_total_variance: Option<f64>,
    pub estimated_total_standard_error: Option<f64>,
    pub estimated_mean_variance: Option<f64>,
    pub variance_unavailable_reason: Option<VarianceUnavailableReasonV1>,
    pub estimated_source_work_total: f64,
    pub estimated_source_work_mean: f64,
    pub estimated_source_work_total_variance: Option<f64>,
    pub estimated_source_work_total_standard_error: Option<f64>,
    pub estimated_source_work_mean_variance: Option<f64>,
    pub source_work_variance_unavailable_reason: Option<VarianceUnavailableReasonV1>,
    pub source_work_fpc: Option<f64>,
    pub selected_source_work_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DesignEstimatorReportV1 {
    pub schema: &'static str,
    pub estimator_version: &'static str,
    pub dataset_id: String,
    pub metric_id: String,
    pub model_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    pub selection_receipt_sha256: String,
    pub candidate_manifest_sha256: String,
    pub normalized_config_sha256: String,
    pub observed_probability_audit_items: usize,
    pub eligible_probability_audit_items: usize,
    pub estimated_population_total: Option<f64>,
    pub estimated_population_mean: Option<f64>,
    pub estimated_population_total_variance: Option<f64>,
    pub estimated_population_total_standard_error: Option<f64>,
    pub estimated_population_mean_variance: Option<f64>,
    pub variance_unavailable_reasons: Vec<(String, VarianceUnavailableReasonV1)>,
    pub estimated_source_work_population_total: Option<f64>,
    pub estimated_source_work_population_mean: Option<f64>,
    pub estimated_source_work_population_total_variance: Option<f64>,
    pub estimated_source_work_population_total_standard_error: Option<f64>,
    pub estimated_source_work_population_mean_variance: Option<f64>,
    pub source_work_variance_unavailable_reasons: Vec<(String, VarianceUnavailableReasonV1)>,
    pub stratum_estimates: Vec<StratumDesignEstimateV1>,
    pub adaptive_active_stream_used: bool,
    pub exact_selection_receipt_replayed: bool,
    pub expected_selection_receipt_supplied: bool,
    pub selection_receipt_match_verified: bool,
    pub observed_outcomes_authenticated: bool,
    pub population_confidence_claim_permitted: bool,
}

#[derive(Debug, Clone)]
struct SampledItemV1<'a> {
    selection: &'a AnnotationSelectionRecordV1,
    outcome: f64,
    source_work_size: usize,
    first_order_probability: f64,
}

#[derive(Debug, Clone, Copy)]
enum EstimatorTargetV1 {
    QueryItem,
    SourceWorkMean,
}

pub fn estimate_design_based_metric_json(input: &str) -> Result<DesignEstimatorReportV1, String> {
    let input: DesignEstimatorInputV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid two-stage estimator input JSON: {error}"))?;
    estimate_design_based_metric(&input)
}

pub fn estimate_design_based_metric(
    input: &DesignEstimatorInputV1,
) -> Result<DesignEstimatorReportV1, String> {
    if input.schema != ESTIMATOR_SCHEMA_V1 {
        return Err(format!("unsupported estimator input schema: {}", input.schema));
    }
    require_text(&input.metric_id, "metric_id")?;
    require_text(&input.evaluator_version, "evaluator_version")?;
    validate_digest(&input.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&input.scoring_plan_sha256, "scoring_plan_sha256")?;
    if let Some(expected) = input.expected_selection_receipt_sha256.as_deref() {
        validate_digest(expected, "expected_selection_receipt_sha256")?;
    }

    let planned = plan_annotation_acquisition(&input.dataset)?;
    if input
        .expected_selection_receipt_sha256
        .as_deref()
        .is_some_and(|expected| planned.selection_receipt_sha256 != expected)
    {
        return Err("selection receipt digest does not match deterministic planner replay".into());
    }
    let selection_receipt_match_verified = input.expected_selection_receipt_sha256.is_some();
    if planned.probability_audit_is_adaptive || !planned.source_work_sets_are_disjoint {
        return Err("planner report does not satisfy frozen independent audit-stream invariants".into());
    }
    if planned.probability_audit.is_empty() {
        return Err("probability audit sample is empty".into());
    }

    let mut outcomes = BTreeMap::new();
    let mut scoring_receipts = BTreeSet::new();
    for outcome in &input.observed_audit_outcomes {
        require_text(&outcome.item_id, "observed_audit_outcomes.item_id")?;
        validate_digest(&outcome.source_work_id_sha256, "outcome.source_work_id_sha256")?;
        validate_digest(&outcome.scoring_report_sha256, "outcome.scoring_report_sha256")?;
        if !scoring_receipts.insert(outcome.scoring_report_sha256.as_str()) {
            return Err(format!("duplicate scoring-report receipt for item {}", outcome.item_id));
        }
        validate_digest(&outcome.model_identity_sha256, "outcome.model_identity_sha256")?;
        validate_digest(&outcome.scoring_plan_sha256, "outcome.scoring_plan_sha256")?;
        if outcome.model_identity_sha256 != input.model_identity_sha256
            || outcome.evaluator_version != input.evaluator_version
            || outcome.scoring_plan_sha256 != input.scoring_plan_sha256
        {
            return Err(format!(
                "item {} has a model/evaluator/scoring-plan identity mismatch",
                outcome.item_id
            ));
        }
        if outcome.metric_id != input.metric_id {
            return Err(format!(
                "item {} has metric ID {}, expected {}",
                outcome.item_id, outcome.metric_id, input.metric_id
            ));
        }
        if outcome.value_ppm > PPM_SCALE {
            return Err(format!(
                "item {} outcome is outside 0..=1,000,000",
                outcome.item_id
            ));
        }
        if outcomes.insert(outcome.item_id.as_str(), outcome).is_some() {
            return Err(format!("duplicate observed audit outcome for {}", outcome.item_id));
        }
    }

    let selected_ids: BTreeSet<_> = planned
        .probability_audit
        .iter()
        .map(|item| item.item_id.as_str())
        .collect();
    if outcomes.len() != selected_ids.len() {
        return Err(format!(
            "observed audit outcome count {} differs from selected audit item count {}",
            outcomes.len(),
            selected_ids.len()
        ));
    }
    for item_id in outcomes.keys() {
        if !selected_ids.contains(item_id) {
            return Err(format!(
                "outcome {} is not in the frozen probability-audit sample (active and unselected rows are forbidden)",
                item_id
            ));
        }
    }
    for selection in &planned.probability_audit {
        if selection.stream != SelectionStreamV1::ProbabilityAudit {
            return Err("planner probability-audit field contains a non-audit selection".into());
        }
        if !outcomes.contains_key(selection.item_id.as_str()) {
            return Err(format!("missing observed outcome for selected item {}", selection.item_id));
        }
    }

    let mut stratum_estimates = Vec::new();
    let mut variance_unavailable_reasons = Vec::new();
    let mut total_estimate = 0.0;
    let mut total_variance = 0.0;
    let mut total_variance_available = true;
    let mut source_work_total_estimate = 0.0;
    let mut source_work_total_variance = 0.0;
    let mut source_work_variance_available = true;
    let mut population_item_count = 0_u64;
    let mut population_source_work_count = 0_u64;
    let mut eligible_probability_audit_items = 0_usize;
    for design in &planned.probability_audit_design {
        eligible_probability_audit_items = eligible_probability_audit_items
            .checked_add(design.eligible_query_items)
            .ok_or_else(|| "eligible probability-audit item count overflow".to_string())?;
    }
    let mut variance_unavailable_reasons = Vec::new();
    let mut source_work_variance_unavailable_reasons = Vec::new();

    for design in &planned.probability_audit_design {
        if design.eligible_query_items == 0 {
            continue;
        }
        if design.selected_source_works == 0 || design.selected_query_items == 0 {
            return Err(format!(
                "stratum {} has eligible units but zero probability-audit inclusion; it cannot be silently assigned zero",
                design.source_stratum_id
            ));
        }
        let mut selected: Vec<_> = planned
            .probability_audit
            .iter()
            .filter(|selection| selection.source_stratum_id == design.source_stratum_id)
            .map(|selection| {
                let observed = outcomes
                    .get(selection.item_id.as_str())
                    .copied()
                    .ok_or_else(|| format!("missing selected outcome for {}", selection.item_id))?;
                if observed.source_work_id_sha256 != selection.source_work_id_sha256 {
                    return Err(format!(
                        "observed outcome {} is bound to a different source work than its planner selection",
                        selection.item_id
                    ));
                }
                let value = observed.value_ppm;
                let source_work_size = selection
                    .eligible_items_in_source_work
                    .filter(|count| *count > 0)
                    .ok_or_else(|| format!("selected item {} lacks a valid source-work size", selection.item_id))?;
                let pi = probability_to_f64(
                    selection.item_inclusion_probability.ok_or_else(|| {
                        format!("selected item {} lacks first-order inclusion probability", selection.item_id)
                    })?,
                )?;
                if pi <= 0.0 || pi > 1.0 {
                    return Err(format!("item {} has invalid inclusion probability {pi}", selection.item_id));
                }
                Ok(SampledItemV1 {
                    selection,
                    outcome: value as f64 / PPM_SCALE as f64,
                    source_work_size,
                    first_order_probability: pi,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        selected.sort_by(|left, right| left.selection.item_id.cmp(&right.selection.item_id));

        let estimate: f64 = selected
            .iter()
            .map(|item| item.outcome / item.first_order_probability)
            .sum();
        let mean_estimate = estimate / design.eligible_query_items as f64;
        let variance = estimate_stratum_variance(design, &selected, EstimatorTargetV1::QueryItem);
        let (variance_value, reason) = match variance {
            Ok(value) if value.is_finite() && value >= 0.0 => (Some(value), None),
            Ok(value) if !value.is_finite() => (
                None,
                Some(VarianceUnavailableReasonV1::NonFiniteVarianceEstimate),
            ),
            Ok(_) => (
                None,
                Some(VarianceUnavailableReasonV1::NegativeVarianceEstimate),
            ),
            Err(reason) => (None, Some(reason)),
        };
        if let Some(value) = variance_value {
            total_variance += value;
        } else {
            total_variance_available = false;
            variance_unavailable_reasons.push((design.source_stratum_id.clone(), reason.unwrap()));
        }
        let source_work_estimate: f64 = selected
            .iter()
            .map(|item| (item.outcome / item.source_work_size as f64) / item.first_order_probability)
            .sum();
        let source_work_mean_estimate =
            source_work_estimate / design.eligible_source_works as f64;
        let source_work_variance = estimate_stratum_variance(
            design,
            &selected,
            EstimatorTargetV1::SourceWorkMean,
        );
        let (source_work_variance_value, source_work_variance_reason) = match source_work_variance {
            Ok(value) if value.is_finite() && value >= 0.0 => (Some(value), None),
            Ok(value) if !value.is_finite() => (
                None,
                Some(VarianceUnavailableReasonV1::NonFiniteVarianceEstimate),
            ),
            Ok(_) => (
                None,
                Some(VarianceUnavailableReasonV1::NegativeVarianceEstimate),
            ),
            Err(reason) => (None, Some(reason)),
        };
        if let Some(value) = source_work_variance_value {
            source_work_total_variance += value;
        } else {
            source_work_variance_available = false;
            source_work_variance_unavailable_reasons
                .push((design.source_stratum_id.clone(), source_work_variance_reason.unwrap()));
        }

        total_estimate += estimate;
        source_work_total_estimate += source_work_estimate;
        population_item_count = population_item_count
            .checked_add(design.eligible_query_items as u64)
            .ok_or_else(|| "population query-item count overflow".to_string())?;
        population_source_work_count = population_source_work_count
            .checked_add(design.eligible_source_works as u64)
            .ok_or_else(|| "population source-work count overflow".to_string())?;
        let variance_standard_error = variance_value.map(f64::sqrt);
        let source_work_variance_standard_error = source_work_variance_value.map(f64::sqrt);
        let source_work_mean_variance = source_work_variance_value
            .map(|value| value / (design.eligible_source_works as f64).powi(2));
        let mean_variance = variance_value.map(|value| {
            value / (design.eligible_query_items as f64).powi(2)
        });
        let source_work_fpc = if design.eligible_source_works > 0 {
            Some(
                1.0 - design.selected_source_works as f64
                    / design.eligible_source_works as f64,
            )
        } else {
            None
        };

        stratum_estimates.push(StratumDesignEstimateV1 {
            source_stratum_id: design.source_stratum_id.clone(),
            eligible_source_works: design.eligible_source_works,
            selected_source_works: design.selected_source_works,
            eligible_query_items: design.eligible_query_items,
            selected_query_items: design.selected_query_items,
            estimated_total: estimate,
            estimated_mean: mean_estimate,
            estimated_total_variance: variance_value,
            estimated_total_standard_error: variance_standard_error,
            estimated_mean_variance: mean_variance,
            variance_unavailable_reason: reason,
            estimated_source_work_total: source_work_estimate,
            estimated_source_work_mean: source_work_mean_estimate,
            estimated_source_work_total_variance: source_work_variance_value,
            estimated_source_work_total_standard_error: source_work_variance_standard_error,
            estimated_source_work_mean_variance: source_work_mean_variance,
            source_work_variance_unavailable_reason: source_work_variance_reason,
            source_work_fpc,
            selected_source_work_ids: selected
                .iter()
                .map(|item| item.selection.source_work_id_sha256.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        });
    }

    if population_item_count == 0 {
        return Err("no eligible probability-audit query items exist in the finite population".into());
    }
    let population_mean = total_estimate / population_item_count as f64;
    let population_variance = total_variance_available.then_some(total_variance);
    let population_mean_variance =
        population_variance.map(|value| value / (population_item_count as f64).powi(2));
    let source_work_population_mean =
        source_work_total_estimate / population_source_work_count as f64;
    let source_work_population_variance =
        source_work_variance_available.then_some(source_work_total_variance);
    let source_work_population_mean_variance = source_work_population_variance
        .map(|value| value / (population_source_work_count as f64).powi(2));

    Ok(DesignEstimatorReportV1 {
        schema: ESTIMATOR_REPORT_SCHEMA_V1,
        estimator_version: ESTIMATOR_VERSION_V1,
        dataset_id: input.dataset.dataset_id.clone(),
        metric_id: input.metric_id.clone(),
        model_identity_sha256: input.model_identity_sha256.clone(),
        evaluator_version: input.evaluator_version.clone(),
        scoring_plan_sha256: input.scoring_plan_sha256.clone(),
        selection_receipt_sha256: planned.selection_receipt_sha256,
        candidate_manifest_sha256: planned.candidate_manifest_sha256,
        normalized_config_sha256: planned.normalized_config_sha256,
        observed_probability_audit_items: selected_ids.len(),
        eligible_probability_audit_items,
        estimated_population_total: Some(total_estimate),
        estimated_population_mean: Some(population_mean),
        estimated_population_total_variance: population_variance,
        estimated_population_total_standard_error: population_variance.map(f64::sqrt),
        estimated_population_mean_variance: population_mean_variance,
        variance_unavailable_reasons,
        estimated_source_work_population_total: Some(source_work_total_estimate),
        estimated_source_work_population_mean: Some(source_work_population_mean),
        estimated_source_work_population_total_variance: source_work_population_variance,
        estimated_source_work_population_total_standard_error: source_work_population_variance
            .map(f64::sqrt),
        estimated_source_work_population_mean_variance: source_work_population_mean_variance,
        source_work_variance_unavailable_reasons,
        stratum_estimates,
        adaptive_active_stream_used: false,
        exact_selection_receipt_replayed: true,
        expected_selection_receipt_supplied: input.expected_selection_receipt_sha256.is_some(),
        selection_receipt_match_verified,
        observed_outcomes_authenticated: false,
        population_confidence_claim_permitted: false,
    })
}

fn estimate_stratum_variance(
    design: &ProbabilityAuditStratumDesignV1,
    sample: &[SampledItemV1<'_>],
    target: EstimatorTargetV1,
) -> Result<f64, VarianceUnavailableReasonV1> {
    if design.selected_source_works < 2 && design.eligible_source_works > 1 {
        return Err(VarianceUnavailableReasonV1::FewerThanTwoSelectedSourceWorks);
    }
    let mut variance = 0.0;

    for left in sample {
        for right in sample {
            let left_selection = left.selection;
            let right_selection = right.selection;
            let pi_left = left.first_order_probability;
            let pi_right = right.first_order_probability;
            let pi_joint = if left_selection.item_id == right_selection.item_id {
                pi_left
            } else {
                second_order_inclusion_probability(left_selection, right_selection)?
            };
            if pi_joint <= 0.0 || pi_joint > 1.0 {
                return Err(VarianceUnavailableReasonV1::ZeroJointInclusionProbability);
            }
            let covariance_factor = (pi_joint - pi_left * pi_right) / pi_joint;
            let left_value = match target {
                EstimatorTargetV1::QueryItem => left.outcome,
                EstimatorTargetV1::SourceWorkMean => left.outcome / left.source_work_size as f64,
            };
            let right_value = match target {
                EstimatorTargetV1::QueryItem => right.outcome,
                EstimatorTargetV1::SourceWorkMean => right.outcome / right.source_work_size as f64,
            };
            variance += covariance_factor
                * (left_value / pi_left)
                * (right_value / pi_right);
        }
    }
    Ok(variance)
}

fn second_order_inclusion_probability(
    left: &AnnotationSelectionRecordV1,
    right: &AnnotationSelectionRecordV1,
) -> Result<f64, VarianceUnavailableReasonV1> {
    if left.source_stratum_id != right.source_stratum_id {
        // Independent stratified selections have zero covariance. The caller
        // estimates each stratum separately and never calls this cross-stratum.
        return Ok(0.0);
    }
    let works = left
        .eligible_source_works_in_stratum
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;
    let selected_works = left
        .selected_source_works_in_stratum
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;
    let left_n = left
        .eligible_items_in_source_work
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;
    let right_n = right
        .eligible_items_in_source_work
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;
    let left_m = left
        .selected_items_in_source_work
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;
    let right_m = right
        .selected_items_in_source_work
        .ok_or(VarianceUnavailableReasonV1::ZeroJointInclusionProbability)?;

    if works == 0 || selected_works == 0 {
        return Err(VarianceUnavailableReasonV1::ZeroJointInclusionProbability);
    }
    let source_probability = if left.source_work_id_sha256 == right.source_work_id_sha256 {
        if left_n <= 1 {
            return Err(VarianceUnavailableReasonV1::InsufficientWithinWorkSample);
        }
        if left_n != right_n || left_m != right_m {
            return Err(VarianceUnavailableReasonV1::ZeroJointInclusionProbability);
        }
        (selected_works as f64 / works as f64)
            * (left_m as f64 * (left_m - 1) as f64)
            / (left_n as f64 * (left_n - 1) as f64)
    } else {
        if works <= 1 || selected_works <= 1 {
            return Err(VarianceUnavailableReasonV1::FewerThanTwoSelectedSourceWorks);
        }
        let between_source_probability = (selected_works as f64
            * (selected_works - 1) as f64)
            / (works as f64 * (works - 1) as f64);
        let left_conditional = left_m as f64 / left_n as f64;
        let right_conditional = right_m as f64 / right_n as f64;
        between_source_probability * left_conditional * right_conditional
    };
    Ok(source_probability)
}

fn probability_to_f64(
    probability: crate::temporal_acquisition::ProbabilityFractionV1,
) -> Result<f64, String> {
    if probability.denominator == 0 || probability.numerator == 0 {
        return Err("inclusion probability has a zero numerator/denominator".into());
    }
    Ok(probability.numerator as f64 / probability.denominator as f64)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use crate::temporal_acquisition::{
        AnnotationAcquisitionCandidateV1, AnnotationCohortRoleV1,
        AnnotationStratumQuotaV1, AuthorizationStateV1,
    };

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn candidate(item: &str, work: char) -> AnnotationAcquisitionCandidateV1 {
        AnnotationAcquisitionCandidateV1 {
            item_id: item.into(),
            query_id: format!("query-{item}"),
            source_work_id_sha256: digest(work),
            recording_sha256: digest(work.to_ascii_lowercase()),
            source_stratum_id: "stratum-a".into(),
            task_family: "piano-entry".into(),
            purpose_scope: "confirmatory-evaluation".into(),
            model_grounding_report_sha256: digest('b'),
            inter_annotator_reliability_report_sha256: digest('c'),
            cohort_assignment_manifest_sha256: digest('d'),
            cohort_role: AnnotationCohortRoleV1::ConfirmatoryAudit,
            authorization_state: AuthorizationStateV1::Authorized,
            authorization_receipt_sha256: Some(digest('e')),
            current_annotation_count: 1,
            maximum_annotation_count: 3,
            annotation_cost_units: 1,
            model_error_ppm: 0,
            human_disagreement_ppm: 0,
            occurrence_count_disagreement_ppm: 0,
            prediction_coverage_ppm: 1_000_000,
            boundary_dispersion_ms: 0,
            hidden_repeat_control: false,
        }
    }

    fn dataset(seed: u64, source_work_quota: u32) -> AnnotationAcquisitionDatasetV1 {
        let mut candidates = Vec::new();
        for (work, values) in [
            ('a', ["a-1", "a-2"]),
            ('b', ["b-1", "b-2"]),
            ('c', ["c-1", "c-2"]),
        ] {
            candidates.push(candidate(values[0], work));
            candidates.push(candidate(values[1], work));
        }
        AnnotationAcquisitionDatasetV1 {
            dataset_id: "finite-population-three-work-oracle".into(),
            config: AnnotationAcquisitionConfigV1 {
                schema: "mel-annotation-acquisition-plan-v1".into(),
                planner_id: "mel-annotation-acquisition-v1".into(),
                seed,
                minimum_panel_count: 1,
                maximum_active_items_per_source_work: 1,
                audit_items_per_source_work: 2,
                total_budget_units: 20,
                active_budget_units: 0,
                probability_audit_budget_units: 20,
                model_error_threshold_ppm: 300_000,
                human_disagreement_threshold_ppm: 250_000,
                occurrence_count_disagreement_threshold_ppm: 100_000,
                minimum_prediction_coverage_ppm: 750_000,
                boundary_dispersion_threshold_ms: 80,
                stratum_quotas: vec![AnnotationStratumQuotaV1 {
                    source_stratum_id: "stratum-a".into(),
                    active_minimum_items: 0,
                    active_maximum_items: 0,
                    audit_source_work_count: source_work_quota,
                }],
            },
            candidates,
        }
    }

    fn value_for_item(item_id: &str) -> u32 {
        match item_id {
            "a-1" | "b-1" | "b-2" => PPM_SCALE,
            "a-2" | "c-1" | "c-2" => 0,
            _ => panic!("unexpected item ID"),
        }
    }

    fn estimator_input(dataset: AnnotationAcquisitionDatasetV1) -> DesignEstimatorInputV1 {
        let planned = plan_annotation_acquisition(&dataset).unwrap();
        let observed_audit_outcomes = planned
            .probability_audit
            .iter()
            .map(|selection| ObservedAuditOutcomeV1 {
                item_id: selection.item_id.clone(),
                source_work_id_sha256: selection.source_work_id_sha256.clone(),
                scoring_report_sha256: digest('3'),
                model_identity_sha256: digest('1'),
                evaluator_version: "mel-grounding-model-eval-v1".into(),
                scoring_plan_sha256: digest('2'),
                metric_id: "grounded-correct".into(),
                value_ppm: value_for_item(&selection.item_id),
            })
            .collect();
        DesignEstimatorInputV1 {
            schema: ESTIMATOR_SCHEMA_V1.into(),
            dataset,
            expected_selection_receipt_sha256: Some(planned.selection_receipt_sha256),
            metric_id: "grounded-correct".into(),
            model_identity_sha256: digest('1'),
            evaluator_version: "mel-grounding-model-eval-v1".into(),
            scoring_plan_sha256: digest('2'),
            observed_audit_outcomes,
        }
    }

    #[test]
    fn exhaustive_two_stage_oracle_recovers_total_and_design_variance() {
        let mut estimates = Vec::new();
        let mut variances = Vec::new();
        let mut selected_work_sets = BTreeSet::new();

        // The finite population has three works with totals [1, 2, 0].
        // K=2 works are sampled without replacement and both query items
        // inside each selected work are a within-work census. There are
        // exactly C(3,2)=3 possible work samples.
        for seed in 1_u64..=256 {
            let input = estimator_input(dataset(seed, 2));
            let report = estimate_design_based_metric(&input).unwrap();
            let work_set: BTreeSet<_> = report.stratum_estimates[0]
                .selected_source_work_ids
                .iter()
                .cloned()
                .collect();
            if selected_work_sets.insert(work_set) {
                estimates.push(report.estimated_population_total.unwrap());
                variances.push(
                    report
                        .estimated_population_total_variance
                        .expect("K=2 supports between-work variance"),
                );
            }
        }

        assert_eq!(selected_work_sets.len(), 3);
        assert_eq!(estimates.len(), 3);
        assert!((mean(&estimates) - 3.0).abs() < 1e-10);
        // Exact finite-population variance for PSU totals [1,2,0], W=3,K=2:
        // W^2 * (1-K/W) * S^2 / K = 1.5.
        assert!((mean(&variances) - 1.5).abs() < 1e-10);
        assert!(variances.iter().all(|variance| *variance >= 0.0));
    }

    #[test]
    fn unequal_work_sizes_use_distinct_within_work_inclusion_probabilities() {
        let mut data = dataset(101, 2);
        data.candidates = vec![
            candidate("a-1", 'a'),
            candidate("b-1", 'b'),
            candidate("b-2", 'b'),
            candidate("c-1", 'c'),
            candidate("c-2", 'c'),
            candidate("c-3", 'c'),
        ];
        let mut observed_same_work = false;
        let mut observed_different_size_works = false;

        for seed in 1_u64..=512 {
            data.config.seed = seed;
            let plan = plan_annotation_acquisition(&data).unwrap();
            for left_index in 0..plan.probability_audit.len() {
                for right_index in (left_index + 1)..plan.probability_audit.len() {
                    let left = &plan.probability_audit[left_index];
                    let right = &plan.probability_audit[right_index];
                    let joint = second_order_inclusion_probability(left, right).unwrap();
                    let works = left.eligible_source_works_in_stratum.unwrap() as f64;
                    let selected_works = left.selected_source_works_in_stratum.unwrap() as f64;
                    let left_n = left.eligible_items_in_source_work.unwrap() as f64;
                    let right_n = right.eligible_items_in_source_work.unwrap() as f64;
                    let left_m = left.selected_items_in_source_work.unwrap() as f64;
                    let right_m = right.selected_items_in_source_work.unwrap() as f64;

                    let expected = if left.source_work_id_sha256 == right.source_work_id_sha256 {
                        observed_same_work = true;
                        (selected_works / works)
                            * (left_m * (left_m - 1.0) / (left_n * (left_n - 1.0)))
                    } else {
                        if (left_n - right_n).abs() > f64::EPSILON {
                            observed_different_size_works = true;
                        }
                        (selected_works * (selected_works - 1.0) / (works * (works - 1.0)))
                            * (left_m / left_n)
                            * (right_m / right_n)
                    };
                    assert!((joint - expected).abs() < 1e-12);
                }
            }
            if observed_same_work && observed_different_size_works {
                break;
            }
        }
        assert!(observed_same_work);
        assert!(observed_different_size_works);
    }

    #[test]
    fn source_work_weighted_estimator_recovers_equal_work_target_and_variance() {
        let mut means = Vec::new();
        let mut variances = Vec::new();
        let mut selected_sets = BTreeSet::new();

        // Source-work means are [0.5, 1.0, 0.0]. Across every possible
        // two-of-three work sample, the HT source-work total is unbiased.
        for seed in 1_u64..=256 {
            let report = estimate_design_based_metric(&estimator_input(dataset(seed, 2))).unwrap();
            let selected: BTreeSet<_> = report.stratum_estimates[0]
                .selected_source_work_ids
                .iter()
                .cloned()
                .collect();
            if selected_sets.insert(selected) {
                means.push(report.estimated_source_work_population_mean.unwrap());
                variances.push(
                    report.estimated_source_work_population_total_variance
                        .expect("two selected source works support between-work variance"),
                );
            }
        }
        assert_eq!(selected_sets.len(), 3);
        assert!((mean(&means) - 0.5).abs() < 1e-10);
        // Exact HT total variance for PSU means [0.5, 1.0, 0.0], W=3,K=2.
        assert!((mean(&variances) - 0.375).abs() < 1e-10);
    }

    #[test]
    fn observed_source_work_and_scoring_identities_must_match_selected_sample() {
        let mut input = estimator_input(dataset(19, 2));
        input.observed_audit_outcomes[0].source_work_id_sha256 = digest('z');
        assert!(estimate_design_based_metric(&input).is_err());

        let mut input = estimator_input(dataset(19, 2));
        input.observed_audit_outcomes[0].model_identity_sha256 = digest('z');
        assert!(estimate_design_based_metric(&input).is_err());

        let mut input = estimator_input(dataset(19, 2));
        input.observed_audit_outcomes[0].scoring_plan_sha256 = digest('z');
        assert!(estimate_design_based_metric(&input).is_err());
    }

    #[test]
    fn one_selected_work_returns_point_estimate_but_withholds_between_work_variance() {
        let input = estimator_input(dataset(71, 1));
        let report = estimate_design_based_metric(&input).unwrap();
        assert!(report.estimated_population_total.is_some());
        assert!(report.estimated_population_mean.is_some());
        assert_eq!(report.estimated_population_total_variance, None);
        assert_eq!(
            report.stratum_estimates[0].variance_unavailable_reason,
            Some(VarianceUnavailableReasonV1::FewerThanTwoSelectedSourceWorks)
        );
        assert!(report.estimated_source_work_population_mean.is_some());
        assert_eq!(report.estimated_source_work_population_total_variance, None);
        assert_eq!(
            report.stratum_estimates[0].source_work_variance_unavailable_reason,
            Some(VarianceUnavailableReasonV1::FewerThanTwoSelectedSourceWorks)
        );
        assert!(!report.population_confidence_claim_permitted);
    }

    #[test]
    fn exact_receipt_and_exact_selected_outcome_set_are_required() {
        let mut input = estimator_input(dataset(19, 2));
        input.expected_selection_receipt_sha256 = Some(digest('f'));
        assert!(estimate_design_based_metric(&input).is_err());

        let mut input = estimator_input(dataset(19, 2));
        input.observed_audit_outcomes.pop();
        assert!(estimate_design_based_metric(&input).is_err());

        let mut input = estimator_input(dataset(19, 2));
        input.observed_audit_outcomes.push(ObservedAuditOutcomeV1 {
            item_id: "not-selected-active-item".into(),
            source_work_id_sha256: digest('a'),
            scoring_report_sha256: digest('3'),
            model_identity_sha256: digest('1'),
            evaluator_version: "mel-grounding-model-eval-v1".into(),
            scoring_plan_sha256: digest('2'),
            metric_id: "grounded-correct".into(),
            value_ppm: PPM_SCALE,
        });
        assert!(estimate_design_based_metric(&input).is_err());
    }

    #[test]
    fn duplicate_metric_rows_and_active_stream_data_are_rejected() {
        let mut input = estimator_input(dataset(23, 2));
        input.observed_audit_outcomes.push(input.observed_audit_outcomes[0].clone());
        assert!(estimate_design_based_metric(&input).is_err());

        let mut input = estimator_input(dataset(23, 2));
        input.observed_audit_outcomes[0].metric_id = "other-metric".into();
        assert!(estimate_design_based_metric(&input).is_err());
    }

    #[test]
    fn scoring_report_receipts_must_be_unique_per_audited_query_item() {
        let mut input = estimator_input(dataset(29, 2));
        input.observed_audit_outcomes[1].scoring_report_sha256 =
            input.observed_audit_outcomes[0].scoring_report_sha256.clone();
        assert!(estimate_design_based_metric(&input).is_err());
    }

    #[test]
    fn design_estimator_keeps_audit_and_population_claim_boundaries_explicit() {
        let input = estimator_input(dataset(31, 2));
        let report = estimate_design_based_metric(&input).unwrap();
        assert!(report.exact_selection_receipt_replayed);
        assert!(report.expected_selection_receipt_supplied);
        assert!(report.selection_receipt_match_verified);
        assert!(!report.adaptive_active_stream_used);
        assert!(!report.observed_outcomes_authenticated);
        assert!(!report.population_confidence_claim_permitted);
        assert_eq!(report.metric_id, "grounded-correct");
        assert_eq!(report.eligible_probability_audit_items, 6);
        assert!(report.estimated_source_work_population_mean.is_some());
        assert!(report.estimated_source_work_population_total_variance.is_some());
        assert!(report.estimated_source_work_population_total_variance.unwrap().is_finite());
    }
}
