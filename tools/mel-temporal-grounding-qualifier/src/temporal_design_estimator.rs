//! Design-based finite-population estimation for the frozen two-stage audit.
//!
//! Replays the acquisition planner and requires the exact selection receipt.
//! Only probability-audit outcomes are accepted. Active-refinement labels are
//! never pooled into the estimator. The general HT variance estimator uses
//! first- and second-order ultimate-item inclusion probabilities; strata with
//! zero pairwise inclusion support report variance unavailable.

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
    pub expected_selection_receipt_sha256: String,
    pub metric_id: String,
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
    pub source_work_fpc: Option<f64>,
    pub selected_source_work_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DesignEstimatorReportV1 {
    pub schema: &'static str,
    pub estimator_version: &'static str,
    pub dataset_id: String,
    pub metric_id: String,
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
    pub stratum_estimates: Vec<StratumDesignEstimateV1>,
    pub adaptive_active_stream_used: bool,
    pub exact_selection_receipt_replayed: bool,
    pub observed_outcomes_authenticated: bool,
    pub population_confidence_claim_permitted: bool,
}

#[derive(Debug, Clone)]
struct SampledItemV1<'a> {
    selection: &'a AnnotationSelectionRecordV1,
    outcome: f64,
    first_order_probability: f64,
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
    validate_digest(
        &input.expected_selection_receipt_sha256,
        "expected_selection_receipt_sha256",
    )?;

    let planned = plan_annotation_acquisition(&input.dataset)?;
    if planned.selection_receipt_sha256 != input.expected_selection_receipt_sha256 {
        return Err("selection receipt digest does not match deterministic planner replay".into());
    }
    if planned.probability_audit_is_adaptive || !planned.source_work_sets_are_disjoint {
        return Err("planner report does not satisfy frozen independent audit-stream invariants".into());
    }
    if planned.probability_audit.is_empty() {
        return Err("probability audit sample is empty".into());
    }

    let mut outcomes = BTreeMap::new();
    for outcome in &input.observed_audit_outcomes {
        require_text(&outcome.item_id, "observed_audit_outcomes.item_id")?;
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
        if outcomes.insert(outcome.item_id.as_str(), outcome.value_ppm).is_some() {
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
    let mut population_item_count = 0_u64;
    let eligible_probability_audit_items: usize = planned
        .probability_audit_design
        .iter()
        .map(|design| design.eligible_query_items)
        .sum();

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
                let value = outcomes
                    .get(selection.item_id.as_str())
                    .copied()
                    .ok_or_else(|| format!("missing selected outcome for {}", selection.item_id))?;
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
        let variance = estimate_stratum_variance(design, &selected);
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
        total_estimate += estimate;
        population_item_count = population_item_count
            .checked_add(design.eligible_query_items as u64)
            .ok_or_else(|| "population query-item count overflow".to_string())?;
        let variance_standard_error = variance_value.map(f64::sqrt);
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

    Ok(DesignEstimatorReportV1 {
        schema: ESTIMATOR_REPORT_SCHEMA_V1,
        estimator_version: ESTIMATOR_VERSION_V1,
        dataset_id: input.dataset.dataset_id.clone(),
        metric_id: input.metric_id.clone(),
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
        stratum_estimates,
        adaptive_active_stream_used: false,
        exact_selection_receipt_replayed: true,
        observed_outcomes_authenticated: false,
        population_confidence_claim_permitted: false,
    })
}

fn estimate_stratum_variance(
    design: &ProbabilityAuditStratumDesignV1,
    sample: &[SampledItemV1<'_>],
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
            variance += covariance_factor
                * (left.outcome / pi_left)
                * (right.outcome / pi_right);
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
