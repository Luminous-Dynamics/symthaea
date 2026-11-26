//! Design-weighted selective risk/coverage over the frozen two-stage audit.
//!
//! This adapter composes the existing source-balanced acquisition planner,
//! selective-grounding decision outputs, and the shared HT design-variance
//! primitive. It does not create a new sampling design or rescore model
//! predictions. Selection, execution and outcome identities remain
//! unauthenticated until their owning verifiers provide verified receipts.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::evidence_digest::canonical_json_sha256;
use crate::temporal_acquisition::{
    AnnotationAcquisitionDatasetV1, AnnotationSelectionRecordV1, ProbabilityAuditStratumDesignV1,
    SelectionStreamV1, plan_annotation_acquisition,
};
use crate::temporal_design_estimator::{
    EstimatorTargetV1, SampledItemV1, VarianceUnavailableReasonV1, estimate_stratum_variance,
    probability_to_f64,
};
use crate::temporal_selective::{IdentifiabilityBasisV1, SelectiveDecisionV1};

const INPUT_SCHEMA_V1: &str = "mel-selective-design-risk-input-v1";
const REPORT_SCHEMA_V1: &str = "mel-selective-design-risk-report-v1";
const THRESHOLD_SCHEMA_V1: &str = "mel-selective-threshold-plan-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenSelectiveThresholdPlanV1 {
    pub schema: String,
    /// Fixed-point confidence thresholds, in strictly increasing order.
    pub thresholds_ppm: Vec<u32>,
    /// V1 supports only confidence >= threshold.
    pub tie_rule: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveAuditOutcomeV1 {
    pub item_id: String,
    pub query_id: String,
    pub source_work_id_sha256: String,
    /// Digest of the selective-grounding case/report record that produced this row.
    pub selective_report_sha256: String,
    pub adjudication_profile_sha256: String,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub hardware_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    pub evidence_stratum: String,
    pub identifiability_basis: IdentifiabilityBasisV1,
    pub decision: SelectiveDecisionV1,
    pub decision_confidence_ppm: u32,
    /// Correctness of the actual selective decision, as scored by #13648.
    pub decision_correct: bool,
    /// Correctness of the unconditional candidate when abstention is disabled.
    pub unconditional_answer_correct: bool,
    pub false_absence_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveDesignRiskInputV1 {
    pub schema: String,
    pub dataset: AnnotationAcquisitionDatasetV1,
    /// Null is allowed for synthetic demonstration only.
    pub expected_selection_receipt_sha256: Option<String>,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub hardware_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    pub threshold_plan: FrozenSelectiveThresholdPlanV1,
    pub threshold_plan_sha256: String,
    /// Exactly one outcome per selected ProbabilityAudit item.
    pub observed_probability_audit_outcomes: Vec<SelectiveAuditOutcomeV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProportionIntervalV1 {
    pub estimate: f64,
    pub standard_error: Option<f64>,
    /// Normal-approximation 95% interval, clipped to the proportion domain.
    pub confidence_95_low: Option<f64>,
    pub confidence_95_high: Option<f64>,
    pub variance_unavailable_reason: Option<VarianceUnavailableReasonV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectiveRiskCoveragePointV1 {
    pub threshold_ppm: u32,
    pub sample_covered_items: usize,
    pub sample_covered_errors: usize,
    pub sample_false_absence_claims: usize,
    pub item_weighted_coverage: ProportionIntervalV1,
    pub item_weighted_selective_risk: Option<ProportionIntervalV1>,
    pub item_weighted_false_absence_rate: Option<ProportionIntervalV1>,
    pub source_work_weighted_coverage: ProportionIntervalV1,
    pub source_work_weighted_selective_risk: Option<ProportionIntervalV1>,
    pub source_work_weighted_false_absence_rate: Option<ProportionIntervalV1>,
    /// Conditional risk for random retention at the same coverage equals the
    /// unconditional candidate error rate, under retention independent of error.
    pub item_random_retention_risk_at_same_coverage: Option<ProportionIntervalV1>,
    pub source_work_random_retention_risk_at_same_coverage: Option<ProportionIntervalV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvidenceStratumSampleCountV1 {
    pub evidence_stratum: String,
    pub sampled_items: usize,
    pub covered_items: usize,
    pub covered_errors: usize,
    pub false_absence_claims: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectiveDesignRiskReportV1 {
    pub schema: &'static str,
    pub estimator_version: &'static str,
    pub dataset_id: String,
    pub metric_id: String,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub hardware_identity_sha256: String,
    pub evaluator_version: String,
    pub scoring_plan_sha256: String,
    pub threshold_plan_sha256: String,
    pub selection_receipt_sha256: String,
    pub candidate_manifest_sha256: String,
    pub normalized_config_sha256: String,
    pub eligible_source_works: usize,
    pub eligible_query_items: usize,
    pub selected_probability_audit_source_works: usize,
    pub selected_probability_audit_items: usize,
    pub minimum_item_inclusion_probability: f64,
    pub maximum_item_inclusion_probability: f64,
    pub sample_evidence_strata: Vec<EvidenceStratumSampleCountV1>,
    pub always_answer_item_weighted_risk: ProportionIntervalV1,
    pub always_answer_source_work_weighted_risk: ProportionIntervalV1,
    pub random_retention_baseline_is_unconditional_error: bool,
    pub risk_coverage: Vec<SelectiveRiskCoveragePointV1>,
    pub active_refinement_stream_used: bool,
    pub exact_selection_receipt_replayed: bool,
    pub selection_receipt_match_verified: bool,
    pub selective_outcomes_authenticated: bool,
    pub population_confidence_claim_permitted: bool,
    /// V1 reports threshold-specific intervals only; AURC covariance is deferred.
    pub aurc_confidence_interval_computed: bool,
}

#[derive(Debug, Clone)]
struct JoinedAuditItem<'a> {
    selection: &'a AnnotationSelectionRecordV1,
    outcome: &'a SelectiveAuditOutcomeV1,
    first_order_probability: f64,
    source_work_size: usize,
}

pub fn audit_selective_design_risk_json(
    input: &str,
) -> Result<SelectiveDesignRiskReportV1, String> {
    let input: SelectiveDesignRiskInputV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid selective design-risk input JSON: {error}"))?;
    audit_selective_design_risk(&input)
}

pub fn audit_selective_design_risk(
    input: &SelectiveDesignRiskInputV1,
) -> Result<SelectiveDesignRiskReportV1, String> {
    if input.schema != INPUT_SCHEMA_V1 {
        return Err(format!(
            "unsupported selective design-risk schema: {}",
            input.schema
        ));
    }
    validate_threshold_plan(&input.threshold_plan)?;
    validate_digest(&input.threshold_plan_sha256, "threshold_plan_sha256")?;
    let expected_threshold_digest = canonical_json_sha256(&input.threshold_plan)
        .map_err(|error| format!("failed to hash frozen threshold plan: {error}"))?;
    if input.threshold_plan_sha256 != expected_threshold_digest {
        return Err("frozen threshold-plan digest does not match its contents".into());
    }
    require_text(&input.dataset.dataset_id, "dataset.dataset_id")?;
    validate_digest(&input.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&input.runtime_identity_sha256, "runtime_identity_sha256")?;
    validate_digest(&input.decoding_config_sha256, "decoding_config_sha256")?;
    validate_digest(&input.hardware_identity_sha256, "hardware_identity_sha256")?;
    validate_digest(&input.scoring_plan_sha256, "scoring_plan_sha256")?;
    require_text(&input.evaluator_version, "evaluator_version")?;
    if let Some(expected) = input.expected_selection_receipt_sha256.as_deref() {
        validate_digest(expected, "expected_selection_receipt_sha256")?;
    }

    let planned = plan_annotation_acquisition(&input.dataset)?;
    if input
        .expected_selection_receipt_sha256
        .as_deref()
        .is_some_and(|expected| expected != planned.selection_receipt_sha256)
    {
        return Err(
            "expected selection receipt does not match deterministic planner replay".into(),
        );
    }
    if planned.probability_audit_is_adaptive || !planned.source_work_sets_are_disjoint {
        return Err("planner output is not a frozen, source-disjoint probability audit".into());
    }
    if planned.probability_audit.is_empty() {
        return Err("frozen probability-audit sample is empty".into());
    }

    let mut outcomes = BTreeMap::new();
    let mut report_digests = BTreeSet::new();
    for outcome in &input.observed_probability_audit_outcomes {
        validate_outcome_identity(input, outcome)?;
        if outcome.false_absence_claim
            && (outcome.decision != SelectiveDecisionV1::ClaimAbsent || outcome.decision_correct)
        {
            return Err(format!(
                "item {} has an inconsistent false-absence flag",
                outcome.item_id
            ));
        }
        if !report_digests.insert(outcome.selective_report_sha256.as_str()) {
            return Err(format!(
                "duplicate selective report receipt for item {}",
                outcome.item_id
            ));
        }
        if outcomes.insert(outcome.item_id.as_str(), outcome).is_some() {
            return Err(format!(
                "duplicate selective outcome for item {}",
                outcome.item_id
            ));
        }
    }

    let selected_ids: BTreeSet<_> = planned
        .probability_audit
        .iter()
        .map(|selection| selection.item_id.as_str())
        .collect();
    if outcomes.len() != selected_ids.len() {
        return Err(format!(
            "selective outcome count {} differs from selected audit item count {}",
            outcomes.len(),
            selected_ids.len()
        ));
    }
    if outcomes
        .keys()
        .any(|item_id| !selected_ids.contains(*item_id))
    {
        return Err("selective outcomes include an unselected or active-refinement item".into());
    }

    let mut joined = Vec::with_capacity(planned.probability_audit.len());
    for selection in &planned.probability_audit {
        if selection.stream != SelectionStreamV1::ProbabilityAudit {
            return Err("planner probability-audit selection contains a non-audit stream".into());
        }
        let outcome = outcomes
            .get(selection.item_id.as_str())
            .copied()
            .ok_or_else(|| format!("missing selective outcome for {}", selection.item_id))?;
        if outcome.query_id != selection.query_id
            || outcome.source_work_id_sha256 != selection.source_work_id_sha256
        {
            return Err(format!(
                "selective outcome {} does not bind to its exact planner query/source work",
                selection.item_id
            ));
        }
        let inclusion =
            probability_to_f64(selection.item_inclusion_probability.ok_or_else(|| {
                format!(
                    "selected item {} lacks inclusion probability",
                    selection.item_id
                )
            })?)?;
        if !(0.0..=1.0).contains(&inclusion) || inclusion == 0.0 {
            return Err(format!(
                "selected item {} has invalid inclusion probability",
                selection.item_id
            ));
        }
        let source_work_size = selection
            .eligible_items_in_source_work
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                format!(
                    "selected item {} lacks eligible source-work size",
                    selection.item_id
                )
            })?;
        joined.push(JoinedAuditItem {
            selection,
            outcome,
            first_order_probability: inclusion,
            source_work_size,
        });
    }

    let eligible_items = checked_sum(
        planned
            .probability_audit_design
            .iter()
            .map(|design| design.eligible_query_items),
        "eligible query-item population",
    )?;
    let eligible_works = checked_sum(
        planned
            .probability_audit_design
            .iter()
            .map(|design| design.eligible_source_works),
        "eligible source-work population",
    )?;
    if eligible_items == 0 || eligible_works == 0 {
        return Err("probability-audit finite population is empty".into());
    }
    for design in &planned.probability_audit_design {
        if design.eligible_query_items > 0 && design.selected_source_works == 0 {
            return Err(format!(
                "stratum {} has eligible items but zero audit inclusion",
                design.source_stratum_id
            ));
        }
    }

    let selected_works: BTreeSet<_> = joined
        .iter()
        .map(|item| item.selection.source_work_id_sha256.as_str())
        .collect();
    let minimum_pi = joined
        .iter()
        .map(|item| item.first_order_probability)
        .reduce(f64::min)
        .unwrap();
    let maximum_pi = joined
        .iter()
        .map(|item| item.first_order_probability)
        .reduce(f64::max)
        .unwrap();
    let sample_evidence_strata = summarize_sample_evidence_strata(&joined);

    let always_answer_error = make_samples(&joined, |item| {
        if item.outcome.unconditional_answer_correct {
            0.0
        } else {
            1.0
        }
    });
    let always_item_total = ht_total(&always_answer_error, EstimatorTargetV1::QueryItem);
    let always_work_total = ht_total(&always_answer_error, EstimatorTargetV1::SourceWorkMean);
    let always_item_variance = design_variance(
        &planned.probability_audit_design,
        &always_answer_error,
        EstimatorTargetV1::QueryItem,
    );
    let always_work_variance = design_variance(
        &planned.probability_audit_design,
        &always_answer_error,
        EstimatorTargetV1::SourceWorkMean,
    );
    let always_item_risk = always_item_total / eligible_items as f64;
    let always_work_risk = always_work_total / eligible_works as f64;
    let always_answer_item_weighted_risk = proportion_interval_from_total(
        always_item_risk,
        always_item_variance,
        eligible_items as f64,
    );
    let always_answer_source_work_weighted_risk = proportion_interval_from_total(
        always_work_risk,
        always_work_variance,
        eligible_works as f64,
    );

    let mut risk_coverage = Vec::with_capacity(input.threshold_plan.thresholds_ppm.len());
    for threshold in &input.threshold_plan.thresholds_ppm {
        let covered = make_samples(&joined, |item| {
            bool_value(is_covered(item.outcome, *threshold))
        });
        let errors = make_samples(&joined, |item| {
            bool_value(is_covered(item.outcome, *threshold) && !item.outcome.decision_correct)
        });
        let false_absences = make_samples(&joined, |item| {
            bool_value(is_covered(item.outcome, *threshold) && item.outcome.false_absence_claim)
        });

        let sample_covered_items = joined
            .iter()
            .filter(|item| is_covered(item.outcome, *threshold))
            .count();
        let sample_covered_errors = joined
            .iter()
            .filter(|item| is_covered(item.outcome, *threshold) && !item.outcome.decision_correct)
            .count();
        let sample_false_absence_claims = joined
            .iter()
            .filter(|item| is_covered(item.outcome, *threshold) && item.outcome.false_absence_claim)
            .count();

        let covered_item_total = ht_total(&covered, EstimatorTargetV1::QueryItem);
        let covered_work_total = ht_total(&covered, EstimatorTargetV1::SourceWorkMean);
        let error_item_total = ht_total(&errors, EstimatorTargetV1::QueryItem);
        let error_work_total = ht_total(&errors, EstimatorTargetV1::SourceWorkMean);
        let false_absence_item_total = ht_total(&false_absences, EstimatorTargetV1::QueryItem);
        let false_absence_work_total = ht_total(&false_absences, EstimatorTargetV1::SourceWorkMean);

        let coverage_item_variance = design_variance(
            &planned.probability_audit_design,
            &covered,
            EstimatorTargetV1::QueryItem,
        );
        let coverage_work_variance = design_variance(
            &planned.probability_audit_design,
            &covered,
            EstimatorTargetV1::SourceWorkMean,
        );

        let item_coverage = covered_item_total / eligible_items as f64;
        let work_coverage = covered_work_total / eligible_works as f64;
        let item_coverage_interval = proportion_interval_from_total(
            item_coverage,
            coverage_item_variance,
            eligible_items as f64,
        );
        let work_coverage_interval = proportion_interval_from_total(
            work_coverage,
            coverage_work_variance,
            eligible_works as f64,
        );

        let item_risk = (covered_item_total > 0.0).then_some(error_item_total / covered_item_total);
        let work_risk = (covered_work_total > 0.0).then_some(error_work_total / covered_work_total);
        let item_risk_interval = item_risk.map(|risk| {
            let linearized = make_samples(&joined, |item| {
                let c = bool_value(is_covered(item.outcome, *threshold));
                let e = bool_value(
                    is_covered(item.outcome, *threshold) && !item.outcome.decision_correct,
                );
                e - risk * c
            });
            let variance = design_variance(
                &planned.probability_audit_design,
                &linearized,
                EstimatorTargetV1::QueryItem,
            )
            .map(|value| value / covered_item_total.powi(2));
            ratio_interval(risk, variance)
        });
        let work_risk_interval = work_risk.map(|risk| {
            let linearized = make_samples(&joined, |item| {
                let c = bool_value(is_covered(item.outcome, *threshold));
                let e = bool_value(
                    is_covered(item.outcome, *threshold) && !item.outcome.decision_correct,
                );
                e - risk * c
            });
            let variance = design_variance(
                &planned.probability_audit_design,
                &linearized,
                EstimatorTargetV1::SourceWorkMean,
            )
            .map(|value| value / covered_work_total.powi(2));
            ratio_interval(risk, variance)
        });

        let item_false_absence_rate =
            (covered_item_total > 0.0).then_some(false_absence_item_total / covered_item_total);
        let work_false_absence_rate =
            (covered_work_total > 0.0).then_some(false_absence_work_total / covered_work_total);
        let item_false_absence_interval = item_false_absence_rate.map(|rate| {
            let linearized = make_samples(&joined, |item| {
                let c = bool_value(is_covered(item.outcome, *threshold));
                let f = bool_value(
                    is_covered(item.outcome, *threshold) && item.outcome.false_absence_claim,
                );
                f - rate * c
            });
            let variance = design_variance(
                &planned.probability_audit_design,
                &linearized,
                EstimatorTargetV1::QueryItem,
            )
            .map(|value| value / covered_item_total.powi(2));
            ratio_interval(rate, variance)
        });
        let work_false_absence_interval = work_false_absence_rate.map(|rate| {
            let linearized = make_samples(&joined, |item| {
                let c = bool_value(is_covered(item.outcome, *threshold));
                let f = bool_value(
                    is_covered(item.outcome, *threshold) && item.outcome.false_absence_claim,
                );
                f - rate * c
            });
            let variance = design_variance(
                &planned.probability_audit_design,
                &linearized,
                EstimatorTargetV1::SourceWorkMean,
            )
            .map(|value| value / covered_work_total.powi(2));
            ratio_interval(rate, variance)
        });

        risk_coverage.push(SelectiveRiskCoveragePointV1 {
            threshold_ppm: *threshold,
            sample_covered_items,
            sample_covered_errors,
            sample_false_absence_claims,
            item_weighted_coverage: item_coverage_interval,
            item_weighted_selective_risk: item_risk_interval,
            item_weighted_false_absence_rate: item_false_absence_interval,
            source_work_weighted_coverage: work_coverage_interval,
            source_work_weighted_selective_risk: work_risk_interval,
            source_work_weighted_false_absence_rate: work_false_absence_interval,
            item_random_retention_risk_at_same_coverage: Some(
                always_answer_item_weighted_risk.clone(),
            ),
            source_work_random_retention_risk_at_same_coverage: Some(
                always_answer_source_work_weighted_risk.clone(),
            ),
        });
    }

    Ok(SelectiveDesignRiskReportV1 {
        schema: REPORT_SCHEMA_V1,
        estimator_version: "mel-selective-design-risk-v1",
        dataset_id: input.dataset.dataset_id.clone(),
        metric_id: "selective-decision-error".into(),
        model_identity_sha256: input.model_identity_sha256.clone(),
        runtime_identity_sha256: input.runtime_identity_sha256.clone(),
        decoding_config_sha256: input.decoding_config_sha256.clone(),
        hardware_identity_sha256: input.hardware_identity_sha256.clone(),
        evaluator_version: input.evaluator_version.clone(),
        scoring_plan_sha256: input.scoring_plan_sha256.clone(),
        threshold_plan_sha256: input.threshold_plan_sha256.clone(),
        selection_receipt_sha256: planned.selection_receipt_sha256,
        candidate_manifest_sha256: planned.candidate_manifest_sha256,
        normalized_config_sha256: planned.normalized_config_sha256,
        eligible_source_works: eligible_works,
        eligible_query_items: eligible_items,
        selected_probability_audit_source_works: selected_works.len(),
        selected_probability_audit_items: joined.len(),
        minimum_item_inclusion_probability: minimum_pi,
        maximum_item_inclusion_probability: maximum_pi,
        sample_evidence_strata,
        always_answer_item_weighted_risk,
        always_answer_source_work_weighted_risk,
        random_retention_baseline_is_unconditional_error: true,
        risk_coverage,
        active_refinement_stream_used: false,
        exact_selection_receipt_replayed: true,
        selection_receipt_match_verified: input.expected_selection_receipt_sha256.is_some(),
        selective_outcomes_authenticated: false,
        population_confidence_claim_permitted: false,
        aurc_confidence_interval_computed: false,
    })
}

fn validate_outcome_identity(
    input: &SelectiveDesignRiskInputV1,
    outcome: &SelectiveAuditOutcomeV1,
) -> Result<(), String> {
    require_text(&outcome.item_id, "outcome.item_id")?;
    require_text(&outcome.query_id, "outcome.query_id")?;
    require_text(&outcome.evidence_stratum, "outcome.evidence_stratum")?;
    require_text(&outcome.evaluator_version, "outcome.evaluator_version")?;
    for (value, label) in [
        (&outcome.source_work_id_sha256, "source_work_id_sha256"),
        (&outcome.selective_report_sha256, "selective_report_sha256"),
        (
            &outcome.adjudication_profile_sha256,
            "adjudication_profile_sha256",
        ),
        (&outcome.model_identity_sha256, "model_identity_sha256"),
        (&outcome.runtime_identity_sha256, "runtime_identity_sha256"),
        (&outcome.decoding_config_sha256, "decoding_config_sha256"),
        (
            &outcome.hardware_identity_sha256,
            "hardware_identity_sha256",
        ),
        (&outcome.scoring_plan_sha256, "scoring_plan_sha256"),
    ] {
        validate_digest(value, label)?;
    }
    if outcome.decision_confidence_ppm > 1_000_000 {
        return Err(format!(
            "item {} confidence exceeds 1,000,000 ppm",
            outcome.item_id
        ));
    }
    if outcome.model_identity_sha256 != input.model_identity_sha256
        || outcome.runtime_identity_sha256 != input.runtime_identity_sha256
        || outcome.decoding_config_sha256 != input.decoding_config_sha256
        || outcome.hardware_identity_sha256 != input.hardware_identity_sha256
        || outcome.evaluator_version != input.evaluator_version
        || outcome.scoring_plan_sha256 != input.scoring_plan_sha256
    {
        return Err(format!(
            "item {} has a model/runtime/evaluator/plan mismatch",
            outcome.item_id
        ));
    }
    Ok(())
}

fn validate_threshold_plan(plan: &FrozenSelectiveThresholdPlanV1) -> Result<(), String> {
    if plan.schema != THRESHOLD_SCHEMA_V1 {
        return Err(format!(
            "unsupported selective threshold plan schema: {}",
            plan.schema
        ));
    }
    if plan.tie_rule != "greater_or_equal" {
        return Err("V1 threshold tie rule must be greater_or_equal".into());
    }
    if plan.thresholds_ppm.is_empty() || plan.thresholds_ppm.len() > 101 {
        return Err("threshold plan must contain 1..=101 thresholds".into());
    }
    let mut previous = None;
    for threshold in &plan.thresholds_ppm {
        if *threshold > 1_000_000 || previous.is_some_and(|prior| prior >= *threshold) {
            return Err(
                "thresholds must be unique, strictly increasing values in 0..=1,000,000".into(),
            );
        }
        previous = Some(*threshold);
    }
    Ok(())
}

fn bool_value(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn is_covered(outcome: &SelectiveAuditOutcomeV1, threshold_ppm: u32) -> bool {
    matches!(
        outcome.decision,
        SelectiveDecisionV1::Answer | SelectiveDecisionV1::ClaimAbsent
    ) && outcome.decision_confidence_ppm >= threshold_ppm
}

fn make_samples<'a, F>(joined: &'a [JoinedAuditItem<'a>], value: F) -> Vec<SampledItemV1<'a>>
where
    F: Fn(&JoinedAuditItem<'a>) -> f64,
{
    joined
        .iter()
        .map(|item| SampledItemV1 {
            selection: item.selection,
            outcome: value(item),
            source_work_size: item.source_work_size,
            first_order_probability: item.first_order_probability,
        })
        .collect()
}

fn ht_total(sample: &[SampledItemV1<'_>], target: EstimatorTargetV1) -> f64 {
    sample
        .iter()
        .map(|item| match target {
            EstimatorTargetV1::QueryItem => item.outcome / item.first_order_probability,
            EstimatorTargetV1::SourceWorkMean => {
                (item.outcome / item.source_work_size as f64) / item.first_order_probability
            }
        })
        .sum()
}

fn design_variance(
    designs: &[ProbabilityAuditStratumDesignV1],
    sample: &[SampledItemV1<'_>],
    target: EstimatorTargetV1,
) -> Result<f64, VarianceUnavailableReasonV1> {
    let mut total = 0.0;
    for design in designs {
        if design.eligible_query_items == 0 {
            continue;
        }
        if design.selected_source_works == 0 {
            return Err(VarianceUnavailableReasonV1::FewerThanTwoSelectedSourceWorks);
        }
        let stratum_sample: Vec<_> = sample
            .iter()
            .filter(|item| item.selection.source_stratum_id == design.source_stratum_id)
            .cloned()
            .collect();
        if stratum_sample.len() != design.selected_query_items {
            return Err(VarianceUnavailableReasonV1::ZeroJointInclusionProbability);
        }
        let variance = estimate_stratum_variance(design, &stratum_sample, target)?;
        if !variance.is_finite() {
            return Err(VarianceUnavailableReasonV1::NonFiniteVarianceEstimate);
        }
        if variance < 0.0 {
            return Err(VarianceUnavailableReasonV1::NegativeVarianceEstimate);
        }
        total += variance;
    }
    Ok(total)
}

fn proportion_interval_from_total(
    estimate: f64,
    total_variance: Result<f64, VarianceUnavailableReasonV1>,
    denominator: f64,
) -> ProportionIntervalV1 {
    match total_variance {
        Ok(variance) if denominator > 0.0 && variance.is_finite() && variance >= 0.0 => {
            let standard_error = variance.sqrt() / denominator;
            let margin = 1.96 * standard_error;
            ProportionIntervalV1 {
                estimate,
                standard_error: Some(standard_error),
                confidence_95_low: Some((estimate - margin).max(0.0)),
                confidence_95_high: Some((estimate + margin).min(1.0)),
                variance_unavailable_reason: None,
            }
        }
        Ok(_) => ProportionIntervalV1 {
            estimate,
            standard_error: None,
            confidence_95_low: None,
            confidence_95_high: None,
            variance_unavailable_reason: Some(
                VarianceUnavailableReasonV1::NonFiniteVarianceEstimate,
            ),
        },
        Err(reason) => ProportionIntervalV1 {
            estimate,
            standard_error: None,
            confidence_95_low: None,
            confidence_95_high: None,
            variance_unavailable_reason: Some(reason),
        },
    }
}

fn ratio_interval(
    estimate: f64,
    ratio_variance: Result<f64, VarianceUnavailableReasonV1>,
) -> ProportionIntervalV1 {
    match ratio_variance {
        Ok(variance) if variance.is_finite() && variance >= 0.0 => {
            let standard_error = variance.sqrt();
            let margin = 1.96 * standard_error;
            ProportionIntervalV1 {
                estimate,
                standard_error: Some(standard_error),
                confidence_95_low: Some((estimate - margin).max(0.0)),
                confidence_95_high: Some((estimate + margin).min(1.0)),
                variance_unavailable_reason: None,
            }
        }
        Ok(_) => ProportionIntervalV1 {
            estimate,
            standard_error: None,
            confidence_95_low: None,
            confidence_95_high: None,
            variance_unavailable_reason: Some(
                VarianceUnavailableReasonV1::NonFiniteVarianceEstimate,
            ),
        },
        Err(reason) => ProportionIntervalV1 {
            estimate,
            standard_error: None,
            confidence_95_low: None,
            confidence_95_high: None,
            variance_unavailable_reason: Some(reason),
        },
    }
}

fn summarize_sample_evidence_strata(
    joined: &[JoinedAuditItem<'_>],
) -> Vec<EvidenceStratumSampleCountV1> {
    let mut grouped: BTreeMap<&str, Vec<&JoinedAuditItem<'_>>> = BTreeMap::new();
    for item in joined {
        grouped
            .entry(item.outcome.evidence_stratum.as_str())
            .or_default()
            .push(item);
    }
    grouped
        .into_iter()
        .map(|(evidence_stratum, items)| {
            let covered: Vec<_> = items
                .iter()
                .filter(|item| is_covered(item.outcome, 0))
                .collect();
            EvidenceStratumSampleCountV1 {
                evidence_stratum: evidence_stratum.to_owned(),
                sampled_items: items.len(),
                covered_items: covered.len(),
                covered_errors: covered
                    .iter()
                    .filter(|item| !item.outcome.decision_correct)
                    .count(),
                false_absence_claims: covered
                    .iter()
                    .filter(|item| item.outcome.false_absence_claim)
                    .count(),
            }
        })
        .collect()
}

fn checked_sum<I>(values: I, label: &str) -> Result<usize, String>
where
    I: IntoIterator<Item = usize>,
{
    values.into_iter().try_fold(0_usize, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| format!("{label} count overflow"))
    })
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
    use crate::temporal_acquisition::{
        AnnotationAcquisitionCandidateV1, AnnotationAcquisitionConfigV1, AnnotationCohortRoleV1,
        AnnotationStratumQuotaV1, AuthorizationStateV1,
    };

    fn digest(char: char) -> String {
        char.to_string().repeat(64)
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

    fn dataset(
        seed: u64,
        source_works: u32,
        items_per_work: usize,
    ) -> AnnotationAcquisitionDatasetV1 {
        let mut candidates = Vec::new();
        for work_index in 0..source_works {
            let work = char::from(b'a' + u8::try_from(work_index).unwrap());
            for item_index in 0..items_per_work {
                let item_id = format!("{work}-{item_index}");
                candidates.push(candidate(&item_id, work));
            }
        }
        AnnotationAcquisitionDatasetV1 {
            dataset_id: "selective-risk-coverage-finite-population".into(),
            config: AnnotationAcquisitionConfigV1 {
                schema: "mel-annotation-acquisition-plan-v1".into(),
                planner_id: "mel-annotation-acquisition-v1".into(),
                seed,
                minimum_panel_count: 1,
                maximum_active_items_per_source_work: 1,
                audit_items_per_source_work: u32::try_from(items_per_work).unwrap(),
                total_budget_units: 10_000,
                active_budget_units: 0,
                probability_audit_budget_units: 10_000,
                model_error_threshold_ppm: 300_000,
                human_disagreement_threshold_ppm: 250_000,
                occurrence_count_disagreement_threshold_ppm: 100_000,
                minimum_prediction_coverage_ppm: 750_000,
                boundary_dispersion_threshold_ms: 80,
                stratum_quotas: vec![AnnotationStratumQuotaV1 {
                    source_stratum_id: "stratum-a".into(),
                    active_minimum_items: 0,
                    active_maximum_items: 0,
                    audit_source_work_count: source_works.saturating_sub(1),
                }],
            },
            candidates,
        }
    }

    fn outcome(
        item_id: &str,
        work: char,
        decision: SelectiveDecisionV1,
        confidence_ppm: u32,
        decision_correct: bool,
        unconditional_answer_correct: bool,
        false_absence_claim: bool,
    ) -> SelectiveAuditOutcomeV1 {
        let work = digest(work);
        SelectiveAuditOutcomeV1 {
            item_id: item_id.into(),
            query_id: format!("query-{item_id}"),
            source_work_id_sha256: work,
            selective_report_sha256: crate::evidence_digest::sha256_hex(item_id.as_bytes()),
            adjudication_profile_sha256: digest('3'),
            model_identity_sha256: digest('4'),
            runtime_identity_sha256: digest('5'),
            decoding_config_sha256: digest('6'),
            hardware_identity_sha256: digest('7'),
            evaluator_version: "mel-temporal-grounding-v1".into(),
            scoring_plan_sha256: digest('8'),
            evidence_stratum: "clear".into(),
            identifiability_basis: IdentifiabilityBasisV1::ClearPositive,
            decision,
            decision_confidence_ppm: confidence_ppm,
            decision_correct,
            unconditional_answer_correct,
            false_absence_claim,
        }
    }

    fn base_input(dataset: AnnotationAcquisitionDatasetV1) -> SelectiveDesignRiskInputV1 {
        let planned = plan_annotation_acquisition(&dataset).unwrap();
        let threshold_plan = FrozenSelectiveThresholdPlanV1 {
            schema: THRESHOLD_SCHEMA_V1.into(),
            thresholds_ppm: vec![0, 500_000, 800_000],
            tie_rule: "greater_or_equal".into(),
        };
        let threshold_plan_sha256 = canonical_json_sha256(&threshold_plan).unwrap();
        let observed_probability_audit_outcomes = planned
            .probability_audit
            .iter()
            .map(|selection| {
                outcome(
                    &selection.item_id,
                    selection.source_work_id_sha256.chars().next().unwrap(),
                    SelectiveDecisionV1::Answer,
                    900_000,
                    true,
                    true,
                    false,
                )
            })
            .collect();
        SelectiveDesignRiskInputV1 {
            schema: INPUT_SCHEMA_V1.into(),
            dataset,
            expected_selection_receipt_sha256: Some(planned.selection_receipt_sha256),
            model_identity_sha256: digest('4'),
            runtime_identity_sha256: digest('5'),
            decoding_config_sha256: digest('6'),
            hardware_identity_sha256: digest('7'),
            evaluator_version: "mel-temporal-grounding-v1".into(),
            scoring_plan_sha256: digest('8'),
            threshold_plan,
            threshold_plan_sha256,
            observed_probability_audit_outcomes,
        }
    }

    #[test]
    fn zero_covered_threshold_has_zero_coverage_and_undefined_risk() {
        let mut input = base_input(dataset(1, 3, 2));
        for outcome in &mut input.observed_probability_audit_outcomes {
            outcome.decision_confidence_ppm = 100;
        }
        input.threshold_plan.thresholds_ppm = vec![900_000];
        input.threshold_plan_sha256 = canonical_json_sha256(&input.threshold_plan).unwrap();
        let report = audit_selective_design_risk(&input).unwrap();
        let point = &report.risk_coverage[0];
        assert_eq!(point.sample_covered_items, 0);
        assert_eq!(point.item_weighted_coverage.estimate, 0.0);
        assert_eq!(point.item_weighted_selective_risk, None);
        assert_eq!(point.source_work_weighted_selective_risk, None);
    }

    #[test]
    fn false_absence_is_a_covered_error_and_remains_separate() {
        let mut input = base_input(dataset(5, 3, 2));
        input.observed_probability_audit_outcomes[0].decision = SelectiveDecisionV1::ClaimAbsent;
        input.observed_probability_audit_outcomes[0].decision_correct = false;
        input.observed_probability_audit_outcomes[0].unconditional_answer_correct = false;
        input.observed_probability_audit_outcomes[0].false_absence_claim = true;
        let report = audit_selective_design_risk(&input).unwrap();
        let point = &report.risk_coverage[0];
        assert_eq!(point.sample_false_absence_claims, 1);
        assert!(
            point
                .item_weighted_selective_risk
                .as_ref()
                .unwrap()
                .estimate
                > 0.0
        );
        assert!(
            point
                .item_weighted_false_absence_rate
                .as_ref()
                .unwrap()
                .estimate
                > 0.0
        );
    }

    #[test]
    fn wrong_receipt_source_model_or_threshold_plan_is_rejected() {
        let mut input = base_input(dataset(9, 3, 2));
        input.expected_selection_receipt_sha256 = Some(digest('f'));
        assert!(audit_selective_design_risk(&input).is_err());

        let mut input = base_input(dataset(9, 3, 2));
        input.observed_probability_audit_outcomes[0].source_work_id_sha256 = digest('f');
        assert!(audit_selective_design_risk(&input).is_err());

        let mut input = base_input(dataset(9, 3, 2));
        input.observed_probability_audit_outcomes[0].model_identity_sha256 = digest('f');
        assert!(audit_selective_design_risk(&input).is_err());

        let mut input = base_input(dataset(9, 3, 2));
        input.threshold_plan.thresholds_ppm.reverse();
        assert!(audit_selective_design_risk(&input).is_err());
    }

    #[test]
    fn active_and_unselected_outcomes_are_rejected() {
        let mut input = base_input(dataset(11, 3, 2));
        input.observed_probability_audit_outcomes[0].item_id = "not-selected".into();
        assert!(audit_selective_design_risk(&input).is_err());
    }

    #[test]
    fn ratio_variance_uses_existing_ht_pairwise_estimator_for_both_targets() {
        let input = base_input(dataset(15, 3, 2));
        let report = audit_selective_design_risk(&input).unwrap();
        let point = &report.risk_coverage[0];
        assert_eq!(
            point
                .item_weighted_selective_risk
                .as_ref()
                .unwrap()
                .estimate,
            0.0
        );
        assert_eq!(
            point
                .source_work_weighted_selective_risk
                .as_ref()
                .unwrap()
                .estimate,
            0.0
        );
        assert!(
            point
                .item_weighted_selective_risk
                .as_ref()
                .unwrap()
                .standard_error
                .is_some()
        );
        assert!(
            point
                .source_work_weighted_selective_risk
                .as_ref()
                .unwrap()
                .standard_error
                .is_some()
        );
    }

    #[test]
    fn item_and_work_weighted_curves_are_distinct_for_unequal_work_sizes() {
        let mut input = None;
        for seed in 1_u64..=256 {
            let mut data = dataset(seed, 3, 2);
            data.candidates.push(candidate("a-extra", 'a'));
            data.config.audit_items_per_source_work = 2;
            let planned = plan_annotation_acquisition(&data).unwrap();
            if planned
                .probability_audit
                .iter()
                .any(|selection| selection.source_work_id_sha256 == digest('a'))
            {
                input = Some(base_input(data));
                break;
            }
        }
        let mut input = input.expect("a deterministic seed should select the larger source work");
        for outcome in &mut input.observed_probability_audit_outcomes {
            if outcome.item_id.starts_with('a') {
                outcome.decision_correct = false;
                outcome.unconditional_answer_correct = false;
            }
        }
        let report = audit_selective_design_risk(&input).unwrap();
        let point = &report.risk_coverage[0];
        assert_ne!(
            point
                .item_weighted_selective_risk
                .as_ref()
                .unwrap()
                .estimate,
            point
                .source_work_weighted_selective_risk
                .as_ref()
                .unwrap()
                .estimate
        );
    }

    #[test]
    fn exhaustive_two_of_three_source_work_ratio_oracle_matches_sampling_variance() {
        let mut seen = BTreeSet::new();
        let mut risks = Vec::new();
        let mut risk_variances = Vec::new();

        for seed in 1_u64..=512 {
            let input = base_input(dataset(seed, 3, 2));
            let planned = plan_annotation_acquisition(&input.dataset).unwrap();
            let selected: BTreeSet<_> = planned
                .probability_audit
                .iter()
                .map(|selection| selection.source_work_id_sha256.clone())
                .collect();
            if selected.len() != 2 || !seen.insert(selected) {
                continue;
            }
            let mut input = input;
            for outcome in &mut input.observed_probability_audit_outcomes {
                if outcome.source_work_id_sha256 == digest('a') {
                    outcome.decision_correct = false;
                    outcome.unconditional_answer_correct = false;
                }
            }
            let report = audit_selective_design_risk(&input).unwrap();
            let point = &report.risk_coverage[0];
            risks.push(
                point
                    .item_weighted_selective_risk
                    .as_ref()
                    .unwrap()
                    .estimate,
            );
            risk_variances.push(
                point
                    .item_weighted_selective_risk
                    .as_ref()
                    .unwrap()
                    .standard_error
                    .unwrap()
                    .powi(2),
            );
            if seen.len() == 3 {
                break;
            }
        }

        assert_eq!(seen.len(), 3);
        let mean_risk = risks.iter().sum::<f64>() / risks.len() as f64;
        let mean_variance = risk_variances.iter().sum::<f64>() / risk_variances.len() as f64;
        assert!((mean_risk - (1.0 / 3.0)).abs() < 1e-10);
        assert!((mean_variance - (1.0 / 18.0)).abs() < 1e-10);
    }

    #[test]
    fn one_selected_work_from_a_larger_population_withholds_variance() {
        let mut data = dataset(21, 3, 2);
        data.config.stratum_quotas[0].audit_source_work_count = 1;
        let mut input = base_input(data);
        input.threshold_plan.thresholds_ppm = vec![0, 500_000];
        input.threshold_plan_sha256 = canonical_json_sha256(&input.threshold_plan).unwrap();
        let report = audit_selective_design_risk(&input).unwrap();
        assert!(
            report.risk_coverage[0]
                .item_weighted_coverage
                .variance_unavailable_reason
                .is_some()
        );
        assert!(
            report.risk_coverage[0]
                .item_weighted_selective_risk
                .as_ref()
                .unwrap()
                .variance_unavailable_reason
                .is_some()
        );
    }
}
