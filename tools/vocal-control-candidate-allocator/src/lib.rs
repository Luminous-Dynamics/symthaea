//! VCCA-001: conservative finite-candidate allocation for vocal controls.
//!
//! The allocator consumes an independently calibrated local response matrix
//! and a declared finite set of control candidates. It returns a minimum-effort
//! proposal only when the entire uncertainty envelope for every target and
//! protected metric satisfies its predeclared constraint.
//!
//! It does not render audio, authorize controls, infer acoustic measurements,
//! validate the response model outside its profile, or claim convergence.
//! Every selected proposal requires downstream capability authorization and
//! fresh post-render measurements.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const VOCAL_CONTROL_ALLOCATOR_SCHEMA_V1: &str = "vocal-control-allocator/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocatorError {
    pub message: String,
}

impl AllocatorError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for AllocatorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for AllocatorError {}

#[derive(Clone, Debug, PartialEq)]
pub struct VocalControlAllocatorSpecV1 {
    pub response_profile_sha256: String,
    pub response_matrix_sha256: String,
    pub frozen_context_sha256: String,
    pub calibration_receipt_sha256: String,
    pub maximum_weighted_effort: f64,
    pub controls: Vec<ControlBoundV1>,
    pub metrics: Vec<MetricConstraintV1>,
    pub sensitivities: Vec<SensitivityCellV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlBoundV1 {
    pub control_id: String,
    pub unit: String,
    pub supported: bool,
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    /// Maximum absolute change allowed in one proposed correction.
    pub trust_radius: f64,
    /// Normalization scale used for dimensionless intervention effort.
    pub effort_scale: f64,
    /// Positive weight for the normalized intervention effort.
    pub effort_weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricRoleV1 {
    Target,
    Protected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub role: MetricRoleV1,
    /// Target metrics require the entire predicted interval to lie in
    /// [accepted_delta_min, accepted_delta_max].
    pub accepted_delta_min: Option<f64>,
    pub accepted_delta_max: Option<f64>,
    /// Protected metrics require the entire predicted interval to lie in
    /// [-maximum_absolute_delta, +maximum_absolute_delta].
    pub maximum_absolute_delta: Option<f64>,
    /// Additive worst-case model-residual envelope for this metric.
    pub residual_uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityCellV1 {
    pub metric_id: String,
    pub metric_unit: String,
    pub control_id: String,
    pub control_unit: String,
    /// Expected metric change per one unit of the declared control.
    pub effect_per_control_unit: f64,
    /// Worst-case absolute coefficient uncertainty, in metric-units per
    /// control-unit, derived from calibration evidence.
    pub absolute_uncertainty_per_control_unit: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlDeltaV1 {
    pub control_id: String,
    pub unit: String,
    pub delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlCandidateV1 {
    pub candidate_id: String,
    /// Must contain exactly one entry for every declared control.
    pub deltas: Vec<ControlDeltaV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictedMetricIntervalV1 {
    pub metric_id: String,
    pub unit: String,
    pub role: MetricRoleV1,
    pub predicted_center: f64,
    pub uncertainty_radius: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub robustly_satisfied: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateRejectionV1 {
    UnsupportedControl { control_id: String },
    ControlOutsideBounds { control_id: String },
    ControlOutsideTrustRadius { control_id: String },
    WeightedEffortExceeded,
    TargetNotRobustlySatisfied { metric_id: String },
    ProtectedMetricGuardExceeded { metric_id: String },
    NumericOverflow,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvaluationV1 {
    pub candidate_id: String,
    pub weighted_effort: Option<f64>,
    pub feasible: bool,
    pub predicted_metrics: Vec<PredictedMetricIntervalV1>,
    pub rejections: Vec<CandidateRejectionV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationDispositionV1 {
    ProposalOnly,
    AbstainNoRobustCandidate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlProposalV1 {
    pub candidate_id: String,
    pub deltas: Vec<ControlDeltaV1>,
    pub weighted_effort: f64,
    pub response_profile_sha256: String,
    pub response_matrix_sha256: String,
    pub frozen_context_sha256: String,
    pub calibration_receipt_sha256: String,
    /// Always true: this crate never authorizes or executes renderer actions.
    pub proposal_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VocalControlAllocationReportV1 {
    pub schema: String,
    pub disposition: AllocationDispositionV1,
    pub selected_proposal: Option<ControlProposalV1>,
    pub candidate_count: usize,
    pub feasible_candidate_count: usize,
    pub candidate_evaluations: Vec<CandidateEvaluationV1>,
}

/// Select the minimum-effort candidate whose full uncertainty envelopes meet
/// all target and protected metric constraints.
///
/// The candidate set is finite and supplied by the caller. No continuous
/// optimum is claimed. If there is no robustly feasible candidate, the report
/// explicitly abstains and preserves the rejection reasons for every candidate.
pub fn allocate_vocal_control_candidate(
    spec: &VocalControlAllocatorSpecV1,
    candidates: &[ControlCandidateV1],
) -> Result<VocalControlAllocationReportV1, AllocatorError> {
    validate_spec(spec)?;

    if candidates.is_empty() {
        return Err(AllocatorError::new(
            "at least one predeclared control candidate is required",
        ));
    }

    let controls_by_id: BTreeMap<&str, &ControlBoundV1> = spec
        .controls
        .iter()
        .map(|control| (control.control_id.as_str(), control))
        .collect();
    let sensitivities = sensitivity_index(spec)?;
    let mut seen_candidates = BTreeSet::new();
    let mut evaluations = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        require_non_empty(&candidate.candidate_id, "candidate_id")?;
        if !seen_candidates.insert(candidate.candidate_id.as_str()) {
            return Err(AllocatorError::new(format!(
                "duplicate candidate_id: {}",
                candidate.candidate_id
            )));
        }
        let evaluation = evaluate_candidate(
            spec,
            candidate,
            &controls_by_id,
            &sensitivities,
        )?;
        evaluations.push(evaluation);
    }

    let feasible_candidate_count = evaluations.iter().filter(|result| result.feasible).count();

    let selected_index = evaluations
        .iter()
        .enumerate()
        .filter(|(_, result)| result.feasible)
        .min_by(|(_, left), (_, right)| {
            left.weighted_effort
                .expect("feasible candidate has an effort")
                .total_cmp(&right.weighted_effort.expect("feasible candidate has an effort"))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .map(|(index, _)| index);

    let (disposition, selected_proposal) = match selected_index {
        Some(index) => {
            let evaluation = &evaluations[index];
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.candidate_id == evaluation.candidate_id)
                .ok_or_else(|| AllocatorError::new("selected candidate disappeared"))?;
            let proposal = ControlProposalV1 {
                candidate_id: candidate.candidate_id.clone(),
                deltas: canonical_deltas(candidate, spec)?,
                weighted_effort: evaluation
                    .weighted_effort
                    .ok_or_else(|| AllocatorError::new("feasible candidate has no effort"))?,
                response_profile_sha256: spec.response_profile_sha256.clone(),
                response_matrix_sha256: spec.response_matrix_sha256.clone(),
                frozen_context_sha256: spec.frozen_context_sha256.clone(),
                calibration_receipt_sha256: spec.calibration_receipt_sha256.clone(),
                proposal_only: true,
            };
            (AllocationDispositionV1::ProposalOnly, Some(proposal))
        }
        None => (AllocationDispositionV1::AbstainNoRobustCandidate, None),
    };

    Ok(VocalControlAllocationReportV1 {
        schema: VOCAL_CONTROL_ALLOCATOR_SCHEMA_V1.to_owned(),
        disposition,
        selected_proposal,
        candidate_count: candidates.len(),
        feasible_candidate_count,
        candidate_evaluations: evaluations,
    })
}

fn validate_spec(spec: &VocalControlAllocatorSpecV1) -> Result<(), AllocatorError> {
    validate_digest(
        &spec.response_profile_sha256,
        "response_profile_sha256",
    )?;
    validate_digest(
        &spec.response_matrix_sha256,
        "response_matrix_sha256",
    )?;
    validate_digest(&spec.frozen_context_sha256, "frozen_context_sha256")?;
    validate_digest(
        &spec.calibration_receipt_sha256,
        "calibration_receipt_sha256",
    )?;

    if !spec.maximum_weighted_effort.is_finite() || spec.maximum_weighted_effort <= 0.0 {
        return Err(AllocatorError::new(
            "maximum_weighted_effort must be finite and positive",
        ));
    }
    if spec.controls.is_empty() {
        return Err(AllocatorError::new("at least one control is required"));
    }
    if spec.metrics.is_empty() {
        return Err(AllocatorError::new("at least one metric is required"));
    }
    if !spec.metrics.iter().any(|metric| metric.role == MetricRoleV1::Target) {
        return Err(AllocatorError::new(
            "at least one target metric is required",
        ));
    }
    if !spec.metrics.iter().any(|metric| metric.role == MetricRoleV1::Protected) {
        return Err(AllocatorError::new(
            "at least one protected metric is required",
        ));
    }

    let mut control_ids = BTreeSet::new();
    for control in &spec.controls {
        require_non_empty(&control.control_id, "control_id")?;
        require_non_empty(&control.unit, "control unit")?;
        if !control_ids.insert(control.control_id.as_str()) {
            return Err(AllocatorError::new(format!(
                "duplicate control_id: {}",
                control.control_id
            )));
        }
        if !control.minimum_delta.is_finite()
            || !control.maximum_delta.is_finite()
            || control.minimum_delta > control.maximum_delta
        {
            return Err(AllocatorError::new(format!(
                "control {} has invalid minimum/maximum bounds",
                control.control_id
            )));
        }
        if !control.trust_radius.is_finite()
            || control.trust_radius < 0.0
            || !control.effort_scale.is_finite()
            || control.effort_scale <= 0.0
            || !control.effort_weight.is_finite()
            || control.effort_weight <= 0.0
        {
            return Err(AllocatorError::new(format!(
                "control {} has invalid trust radius or effort weighting",
                control.control_id
            )));
        }
        if !control.supported && (control.minimum_delta != 0.0 || control.maximum_delta != 0.0) {
            return Err(AllocatorError::new(format!(
                "unsupported control {} must have zero-only bounds",
                control.control_id
            )));
        }
    }

    let mut metric_ids = BTreeSet::new();
    for metric in &spec.metrics {
        require_non_empty(&metric.metric_id, "metric_id")?;
        require_non_empty(&metric.unit, "metric unit")?;
        if !metric_ids.insert(metric.metric_id.as_str()) {
            return Err(AllocatorError::new(format!(
                "duplicate metric_id: {}",
                metric.metric_id
            )));
        }
        if !metric.residual_uncertainty.is_finite() || metric.residual_uncertainty < 0.0 {
            return Err(AllocatorError::new(format!(
                "metric {} has invalid model residual uncertainty",
                metric.metric_id
            )));
        }
        match metric.role {
            MetricRoleV1::Target => {
                let lower = metric.accepted_delta_min.ok_or_else(|| {
                    AllocatorError::new(format!(
                        "target metric {} is missing accepted_delta_min",
                        metric.metric_id
                    ))
                })?;
                let upper = metric.accepted_delta_max.ok_or_else(|| {
                    AllocatorError::new(format!(
                        "target metric {} is missing accepted_delta_max",
                        metric.metric_id
                    ))
                })?;
                if metric.maximum_absolute_delta.is_some()
                    || !lower.is_finite()
                    || !upper.is_finite()
                    || lower >= upper
                    || (lower <= 0.0 && upper >= 0.0)
                {
                    return Err(AllocatorError::new(format!(
                        "target metric {} requires a finite non-zero signed interval",
                        metric.metric_id
                    )));
                }
            }
            MetricRoleV1::Protected => {
                let tolerance = metric.maximum_absolute_delta.ok_or_else(|| {
                    AllocatorError::new(format!(
                        "protected metric {} is missing maximum_absolute_delta",
                        metric.metric_id
                    ))
                })?;
                if metric.accepted_delta_min.is_some()
                    || metric.accepted_delta_max.is_some()
                    || !tolerance.is_finite()
                    || tolerance < 0.0
                {
                    return Err(AllocatorError::new(format!(
                        "protected metric {} requires a finite non-negative absolute tolerance",
                        metric.metric_id
                    )));
                }
            }
        }
    }

    let expected_cells = spec
        .controls
        .len()
        .checked_mul(spec.metrics.len())
        .ok_or_else(|| AllocatorError::new("sensitivity matrix dimensions overflow"))?;
    if spec.sensitivities.len() != expected_cells {
        return Err(AllocatorError::new(format!(
            "sensitivity matrix requires exactly {expected_cells} cells, received {}",
            spec.sensitivities.len()
        )));
    }

    let expected_controls: BTreeMap<&str, &str> = spec
        .controls
        .iter()
        .map(|control| (control.control_id.as_str(), control.unit.as_str()))
        .collect();
    let expected_metrics: BTreeMap<&str, &str> = spec
        .metrics
        .iter()
        .map(|metric| (metric.metric_id.as_str(), metric.unit.as_str()))
        .collect();
    let mut seen_cells = BTreeSet::new();

    for cell in &spec.sensitivities {
        require_non_empty(&cell.metric_id, "sensitivity metric_id")?;
        require_non_empty(&cell.metric_unit, "sensitivity metric_unit")?;
        require_non_empty(&cell.control_id, "sensitivity control_id")?;
        require_non_empty(&cell.control_unit, "sensitivity control_unit")?;
        let expected_metric_unit = expected_metrics
            .get(cell.metric_id.as_str())
            .ok_or_else(|| {
                AllocatorError::new(format!(
                    "unexpected sensitivity metric {}",
                    cell.metric_id
                ))
            })?;
        let expected_control_unit = expected_controls
            .get(cell.control_id.as_str())
            .ok_or_else(|| {
                AllocatorError::new(format!(
                    "unexpected sensitivity control {}",
                    cell.control_id
                ))
            })?;
        if *expected_metric_unit != cell.metric_unit
            || *expected_control_unit != cell.control_unit
        {
            return Err(AllocatorError::new(format!(
                "unit mismatch in sensitivity cell {}/{}",
                cell.metric_id, cell.control_id
            )));
        }
        if !cell.effect_per_control_unit.is_finite()
            || !cell.absolute_uncertainty_per_control_unit.is_finite()
            || cell.absolute_uncertainty_per_control_unit < 0.0
        {
            return Err(AllocatorError::new(format!(
                "invalid coefficient or uncertainty in sensitivity cell {}/{}",
                cell.metric_id, cell.control_id
            )));
        }
        if !seen_cells.insert((cell.metric_id.as_str(), cell.control_id.as_str())) {
            return Err(AllocatorError::new(format!(
                "duplicate sensitivity cell {}/{}",
                cell.metric_id, cell.control_id
            )));
        }
    }

    Ok(())
}

fn sensitivity_index<'a>(
    spec: &'a VocalControlAllocatorSpecV1,
) -> Result<BTreeMap<(&'a str, &'a str), &'a SensitivityCellV1>, AllocatorError> {
    let mut index = BTreeMap::new();
    for cell in &spec.sensitivities {
        if index
            .insert(
                (cell.metric_id.as_str(), cell.control_id.as_str()),
                cell,
            )
            .is_some()
        {
            return Err(AllocatorError::new("duplicate sensitivity matrix cell"));
        }
    }
    Ok(index)
}

fn evaluate_candidate(
    spec: &VocalControlAllocatorSpecV1,
    candidate: &ControlCandidateV1,
    controls_by_id: &BTreeMap<&str, &ControlBoundV1>,
    sensitivities: &BTreeMap<(&str, &str), &SensitivityCellV1>,
) -> Result<CandidateEvaluationV1, AllocatorError> {
    let mut rejections = Vec::new();
    if candidate.deltas.len() != spec.controls.len() {
        return Err(AllocatorError::new(format!(
            "candidate {} requires exactly {} control deltas, received {}",
            candidate.candidate_id,
            spec.controls.len(),
            candidate.deltas.len()
        )));
    }

    let mut candidate_deltas = BTreeMap::new();
    for delta in &candidate.deltas {
        require_non_empty(&delta.control_id, "candidate control_id")?;
        require_non_empty(&delta.unit, "candidate control unit")?;
        if !delta.delta.is_finite() {
            return Err(AllocatorError::new(format!(
                "candidate {} has a non-finite delta for {}",
                candidate.candidate_id, delta.control_id
            )));
        }
        let control = controls_by_id
            .get(delta.control_id.as_str())
            .copied()
            .ok_or_else(|| {
                AllocatorError::new(format!(
                    "candidate {} contains unexpected control {}",
                    candidate.candidate_id, delta.control_id
                ))
            })?;
        if delta.unit != control.unit {
            return Err(AllocatorError::new(format!(
                "candidate {} control-unit mismatch for {}",
                candidate.candidate_id, delta.control_id
            )));
        }
        if candidate_deltas
            .insert(delta.control_id.as_str(), delta.delta)
            .is_some()
        {
            return Err(AllocatorError::new(format!(
                "candidate {} contains duplicate control {}",
                candidate.candidate_id, delta.control_id
            )));
        }
    }
    for control in &spec.controls {
        if !candidate_deltas.contains_key(control.control_id.as_str()) {
            return Err(AllocatorError::new(format!(
                "candidate {} is missing control {}",
                candidate.candidate_id, control.control_id
            )));
        }
    }

    let mut weighted_effort = 0.0f64;
    for control in &spec.controls {
        let delta = candidate_deltas[control.control_id.as_str()];
        if !control.supported && delta != 0.0 {
            rejections.push(CandidateRejectionV1::UnsupportedControl {
                control_id: control.control_id.clone(),
            });
        }
        if delta < control.minimum_delta || delta > control.maximum_delta {
            rejections.push(CandidateRejectionV1::ControlOutsideBounds {
                control_id: control.control_id.clone(),
            });
        }
        if delta.abs() > control.trust_radius {
            rejections.push(CandidateRejectionV1::ControlOutsideTrustRadius {
                control_id: control.control_id.clone(),
            });
        }
        let normalized = delta / control.effort_scale;
        let contribution = control.effort_weight * normalized * normalized;
        weighted_effort += contribution;
        if !contribution.is_finite() || !weighted_effort.is_finite() {
            rejections.push(CandidateRejectionV1::NumericOverflow);
            break;
        }
    }
    if weighted_effort.is_finite() && weighted_effort > spec.maximum_weighted_effort {
        rejections.push(CandidateRejectionV1::WeightedEffortExceeded);
    }

    let mut predicted_metrics = Vec::with_capacity(spec.metrics.len());
    for metric in &spec.metrics {
        let mut center = 0.0f64;
        let mut uncertainty = metric.residual_uncertainty;
        for control in &spec.controls {
            let delta = candidate_deltas[control.control_id.as_str()];
            let cell = sensitivities
                .get(&(metric.metric_id.as_str(), control.control_id.as_str()))
                .copied()
                .ok_or_else(|| AllocatorError::new("validated sensitivity cell disappeared"))?;
            center += cell.effect_per_control_unit * delta;
            uncertainty += cell.absolute_uncertainty_per_control_unit * delta.abs();
        }
        let lower = center - uncertainty;
        let upper = center + uncertainty;
        if !center.is_finite()
            || !uncertainty.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
        {
            rejections.push(CandidateRejectionV1::NumericOverflow);
            predicted_metrics.push(PredictedMetricIntervalV1 {
                metric_id: metric.metric_id.clone(),
                unit: metric.unit.clone(),
                role: metric.role,
                predicted_center: center,
                uncertainty_radius: uncertainty,
                lower_bound: lower,
                upper_bound: upper,
                robustly_satisfied: false,
            });
            continue;
        }

        let robustly_satisfied = match metric.role {
            MetricRoleV1::Target => {
                let accepted_min = metric.accepted_delta_min.ok_or_else(|| {
                    AllocatorError::new("validated target minimum disappeared")
                })?;
                let accepted_max = metric.accepted_delta_max.ok_or_else(|| {
                    AllocatorError::new("validated target maximum disappeared")
                })?;
                let passed = lower >= accepted_min && upper <= accepted_max;
                if !passed {
                    rejections.push(CandidateRejectionV1::TargetNotRobustlySatisfied {
                        metric_id: metric.metric_id.clone(),
                    });
                }
                passed
            }
            MetricRoleV1::Protected => {
                let tolerance = metric.maximum_absolute_delta.ok_or_else(|| {
                    AllocatorError::new("validated protected tolerance disappeared")
                })?;
                let passed = lower >= -tolerance && upper <= tolerance;
                if !passed {
                    rejections.push(CandidateRejectionV1::ProtectedMetricGuardExceeded {
                        metric_id: metric.metric_id.clone(),
                    });
                }
                passed
            }
        };
        predicted_metrics.push(PredictedMetricIntervalV1 {
            metric_id: metric.metric_id.clone(),
            unit: metric.unit.clone(),
            role: metric.role,
            predicted_center: center,
            uncertainty_radius: uncertainty,
            lower_bound: lower,
            upper_bound: upper,
            robustly_satisfied,
        });
    }

    let feasible = rejections.is_empty();
    Ok(CandidateEvaluationV1 {
        candidate_id: candidate.candidate_id.clone(),
        weighted_effort: weighted_effort.is_finite().then_some(weighted_effort),
        feasible,
        predicted_metrics,
        rejections,
    })
}

fn canonical_deltas(
    candidate: &ControlCandidateV1,
    spec: &VocalControlAllocatorSpecV1,
) -> Result<Vec<ControlDeltaV1>, AllocatorError> {
    let by_id: BTreeMap<&str, &ControlDeltaV1> = candidate
        .deltas
        .iter()
        .map(|delta| (delta.control_id.as_str(), delta))
        .collect();
    spec.controls
        .iter()
        .map(|control| {
            by_id
                .get(control.control_id.as_str())
                .copied()
                .cloned()
                .ok_or_else(|| AllocatorError::new("selected control delta disappeared"))
        })
        .collect()
}

fn validate_digest(value: &str, field: &str) -> Result<(), AllocatorError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if !valid {
        return Err(AllocatorError::new(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), AllocatorError> {
    if value.trim().is_empty() {
        Err(AllocatorError::new(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn spec() -> VocalControlAllocatorSpecV1 {
        VocalControlAllocatorSpecV1 {
            response_profile_sha256: digest('1'),
            response_matrix_sha256: digest('2'),
            frozen_context_sha256: digest('3'),
            calibration_receipt_sha256: digest('4'),
            maximum_weighted_effort: 1.0,
            controls: vec![
                ControlBoundV1 {
                    control_id: "pitch_offset".to_owned(),
                    unit: "semitones".to_owned(),
                    supported: true,
                    minimum_delta: -2.0,
                    maximum_delta: 2.0,
                    trust_radius: 1.0,
                    effort_scale: 1.0,
                    effort_weight: 1.0,
                },
                ControlBoundV1 {
                    control_id: "vibrato_depth".to_owned(),
                    unit: "cents".to_owned(),
                    supported: true,
                    minimum_delta: -10.0,
                    maximum_delta: 10.0,
                    trust_radius: 10.0,
                    effort_scale: 10.0,
                    effort_weight: 1.0,
                },
            ],
            metrics: vec![
                MetricConstraintV1 {
                    metric_id: "pitch_delta".to_owned(),
                    unit: "cents".to_owned(),
                    role: MetricRoleV1::Target,
                    accepted_delta_min: Some(90.0),
                    accepted_delta_max: Some(110.0),
                    maximum_absolute_delta: None,
                    residual_uncertainty: 2.0,
                },
                MetricConstraintV1 {
                    metric_id: "vibrato_change".to_owned(),
                    unit: "cents".to_owned(),
                    role: MetricRoleV1::Protected,
                    accepted_delta_min: None,
                    accepted_delta_max: None,
                    maximum_absolute_delta: Some(5.0),
                    residual_uncertainty: 0.5,
                },
            ],
            sensitivities: vec![
                SensitivityCellV1 {
                    metric_id: "pitch_delta".to_owned(),
                    metric_unit: "cents".to_owned(),
                    control_id: "pitch_offset".to_owned(),
                    control_unit: "semitones".to_owned(),
                    effect_per_control_unit: 100.0,
                    absolute_uncertainty_per_control_unit: 1.0,
                },
                SensitivityCellV1 {
                    metric_id: "pitch_delta".to_owned(),
                    metric_unit: "cents".to_owned(),
                    control_id: "vibrato_depth".to_owned(),
                    control_unit: "cents".to_owned(),
                    effect_per_control_unit: 0.0,
                    absolute_uncertainty_per_control_unit: 0.0,
                },
                SensitivityCellV1 {
                    metric_id: "vibrato_change".to_owned(),
                    metric_unit: "cents".to_owned(),
                    control_id: "pitch_offset".to_owned(),
                    control_unit: "semitones".to_owned(),
                    effect_per_control_unit: 0.2,
                    absolute_uncertainty_per_control_unit: 0.1,
                },
                SensitivityCellV1 {
                    metric_id: "vibrato_change".to_owned(),
                    metric_unit: "cents".to_owned(),
                    control_id: "vibrato_depth".to_owned(),
                    control_unit: "cents".to_owned(),
                    effect_per_control_unit: 1.0,
                    absolute_uncertainty_per_control_unit: 0.05,
                },
            ],
        }
    }

    fn candidate(
        id: &str,
        pitch_delta: f64,
        vibrato_delta: f64,
    ) -> ControlCandidateV1 {
        ControlCandidateV1 {
            candidate_id: id.to_owned(),
            deltas: vec![
                ControlDeltaV1 {
                    control_id: "pitch_offset".to_owned(),
                    unit: "semitones".to_owned(),
                    delta: pitch_delta,
                },
                ControlDeltaV1 {
                    control_id: "vibrato_depth".to_owned(),
                    unit: "cents".to_owned(),
                    delta: vibrato_delta,
                },
            ],
        }
    }

    #[test]
    fn chooses_minimum_effort_candidate_that_robustly_satisfies_all_metrics() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("too-large", 1.1, 0.0),
                candidate("minimum-safe", 1.0, 0.0),
                candidate("also-safe", 1.0, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(report.disposition, AllocationDispositionV1::ProposalOnly);
        assert_eq!(
            report.selected_proposal.as_ref().unwrap().candidate_id,
            "minimum-safe"
        );
        assert!(report.selected_proposal.as_ref().unwrap().proposal_only);
        assert_eq!(report.feasible_candidate_count, 2);
    }

    #[test]
    fn no_op_cannot_satisfy_a_nonzero_target() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[candidate("no-op", 0.0, 0.0)],
        )
        .unwrap();
        assert_eq!(
            report.disposition,
            AllocationDispositionV1::AbstainNoRobustCandidate
        );
        assert!(report.selected_proposal.is_none());
        assert!(report.candidate_evaluations[0].rejections.iter().any(|r| {
            matches!(r, CandidateRejectionV1::TargetNotRobustlySatisfied { .. })
        }));
    }

    #[test]
    fn uncertainty_crossing_target_boundary_rejects_candidate_even_when_center_is_in_band() {
        let mut spec = spec();
        spec.metrics[0].residual_uncertainty = 9.0;
        let report = allocate_vocal_control_candidate(
            &spec,
            &[candidate("boundary", 1.0, 0.0)],
        )
        .unwrap();
        let evaluation = &report.candidate_evaluations[0];
        assert!(!evaluation.feasible);
        assert_eq!(evaluation.predicted_metrics[0].predicted_center, 100.0);
        assert!(evaluation.predicted_metrics[0].lower_bound < 90.0);
    }

    #[test]
    fn protected_uncertainty_crossing_guard_rejects_candidate() {
        let mut spec = spec();
        spec.metrics[1].residual_uncertainty = 4.0;
        let report = allocate_vocal_control_candidate(
            &spec,
            &[candidate("guard-edge", 1.0, 0.0)],
        )
        .unwrap();
        assert!(!report.candidate_evaluations[0].feasible);
        assert!(report.candidate_evaluations[0].rejections.iter().any(|r| {
            matches!(r, CandidateRejectionV1::ProtectedMetricGuardExceeded { .. })
        }));
    }

    #[test]
    fn wrong_direction_response_fails_target_interval() {
        let mut spec = spec();
        let cell = spec
            .sensitivities
            .iter_mut()
            .find(|cell| {
                cell.metric_id == "pitch_delta" && cell.control_id == "pitch_offset"
            })
            .unwrap();
        cell.effect_per_control_unit = -100.0;
        let report = allocate_vocal_control_candidate(
            &spec,
            &[candidate("wrong-direction", 1.0, 0.0)],
        )
        .unwrap();
        assert!(!report.candidate_evaluations[0].feasible);
    }

    #[test]
    fn unsupported_control_must_remain_zero() {
        let mut spec = spec();
        spec.controls[1].supported = false;
        spec.controls[1].minimum_delta = 0.0;
        spec.controls[1].maximum_delta = 0.0;
        let report = allocate_vocal_control_candidate(
            &spec,
            &[candidate("uses-unsupported", 1.0, 1.0)],
        )
        .unwrap();
        assert!(!report.candidate_evaluations[0].feasible);
        assert!(report.candidate_evaluations[0].rejections.iter().any(|r| {
            matches!(r, CandidateRejectionV1::UnsupportedControl { .. })
        }));
    }

    #[test]
    fn control_bounds_and_trust_radius_are_enforced() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[candidate("outside-trust", 1.5, 0.0)],
        )
        .unwrap();
        let rejections = &report.candidate_evaluations[0].rejections;
        assert!(rejections.iter().any(|r| matches!(
            r, CandidateRejectionV1::ControlOutsideTrustRadius { .. }
        )));
    }

    #[test]
    fn weighted_effort_budget_is_hard() {
        let mut spec = spec();
        spec.maximum_weighted_effort = 0.5;
        let report = allocate_vocal_control_candidate(
            &spec,
            &[candidate("over-budget", 1.0, 0.0)],
        )
        .unwrap();
        assert!(!report.candidate_evaluations[0].feasible);
        assert!(report.candidate_evaluations[0].rejections.iter().any(|r| {
            matches!(r, CandidateRejectionV1::WeightedEffortExceeded)
        }));
    }

    #[test]
    fn incomplete_and_duplicate_sensitivity_cells_are_rejected() {
        let mut missing = spec();
        missing.sensitivities.pop();
        assert!(allocate_vocal_control_candidate(
            &missing,
            &[candidate("candidate", 1.0, 0.0)]
        )
        .is_err());

        let mut duplicate = spec();
        duplicate.sensitivities[1] = duplicate.sensitivities[0].clone();
        assert!(allocate_vocal_control_candidate(
            &duplicate,
            &[candidate("candidate", 1.0, 0.0)]
        )
        .is_err());
    }

    #[test]
    fn missing_extra_duplicate_or_wrong_unit_control_deltas_are_rejected() {
        let mut missing = candidate("missing", 1.0, 0.0);
        missing.deltas.pop();
        assert!(allocate_vocal_control_candidate(&spec(), &[missing]).is_err());

        let mut duplicate = candidate("duplicate", 1.0, 0.0);
        duplicate.deltas[1].control_id = "pitch_offset".to_owned();
        assert!(allocate_vocal_control_candidate(&spec(), &[duplicate]).is_err());

        let mut wrong_unit = candidate("wrong-unit", 1.0, 0.0);
        wrong_unit.deltas[0].unit = "Hz".to_owned();
        assert!(allocate_vocal_control_candidate(&spec(), &[wrong_unit]).is_err());

        let mut extra = candidate("extra", 1.0, 0.0);
        extra.deltas.push(ControlDeltaV1 {
            control_id: "global_identity".to_owned(),
            unit: "unit".to_owned(),
            delta: 0.0,
        });
        assert!(allocate_vocal_control_candidate(&spec(), &[extra]).is_err());
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        assert!(allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("same", 1.0, 0.0),
                candidate("same", 1.0, 1.0),
            ],
        )
        .is_err());
    }

    #[test]
    fn no_feasible_candidate_preserves_all_rejection_reasons() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("no-op", 0.0, 0.0),
                candidate("over-trust", 1.5, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(
            report.disposition,
            AllocationDispositionV1::AbstainNoRobustCandidate
        );
        assert_eq!(report.candidate_evaluations.len(), 2);
        assert!(report.candidate_evaluations.iter().all(|r| !r.rejections.is_empty()));
    }

    #[test]
    fn equal_effort_ties_are_broken_by_candidate_id() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("z-choice", 1.0, 0.0),
                candidate("a-choice", 1.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(
            report.selected_proposal.as_ref().unwrap().candidate_id,
            "a-choice"
        );
    }

    #[test]
    fn model_residual_uncertainty_is_included_in_interval() {
        let report = allocate_vocal_control_candidate(
            &spec(),
            &[candidate("candidate", 1.0, 0.0)],
        )
        .unwrap();
        let target = &report.candidate_evaluations[0].predicted_metrics[0];
        assert_eq!(target.predicted_center, 100.0);
        assert_eq!(target.uncertainty_radius, 3.0);
        assert_eq!(target.lower_bound, 97.0);
        assert_eq!(target.upper_bound, 103.0);
    }

    #[test]
    fn malformed_digests_and_nonfinite_specification_values_are_rejected() {
        let mut malformed_digest = spec();
        malformed_digest.response_matrix_sha256 = "bad".to_owned();
        assert!(allocate_vocal_control_candidate(
            &malformed_digest,
            &[candidate("candidate", 1.0, 0.0)]
        )
        .is_err());

        let mut nonfinite = spec();
        nonfinite.controls[0].effort_scale = f64::NAN;
        assert!(allocate_vocal_control_candidate(
            &nonfinite,
            &[candidate("candidate", 1.0, 0.0)]
        )
        .is_err());
    }

    #[test]
    fn deterministic_input_order_produces_same_candidate_selection() {
        let first = allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("b", 1.0, 0.0),
                candidate("a", 1.0, 0.0),
            ],
        )
        .unwrap();
        let second = allocate_vocal_control_candidate(
            &spec(),
            &[
                candidate("a", 1.0, 0.0),
                candidate("b", 1.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(
            first.selected_proposal.unwrap().candidate_id,
            second.selected_proposal.unwrap().candidate_id
        );
    }
}
