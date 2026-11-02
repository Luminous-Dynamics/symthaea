#![forbid(unsafe_code)]

//! Finite-cohort partial-identification bounds for selection-aware connectomics.
//!
//! This dependency-free kernel does not fit detectors, infer neural non-response from
//! detector output, authenticate assignment, or produce population confidence intervals.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmRole { Treatment, Control }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStatus {
    Detected,
    NotDetected,
    Unobserved,
    ClassifierInvalid,
    RegistrationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStatus {
    /// The preregistered source/target response-selection rule was met.
    Selected,
    /// The rule was evaluated and not met. This is not a ground-truth neural zero.
    NotSelected,
    /// The trace or selection decision was unavailable.
    Unobserved,
    /// The classifier could not return a valid decision.
    ClassifierInvalid,
    /// Registration/alignment failed, so selection could not be evaluated.
    RegistrationFailed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationUnit {
    pub unit_id: String,
    pub selection_status: SelectionStatus,
    /// Source-neuron response detector state, independent from target.
    pub source_response_status: ResponseStatus,
    /// Target-neuron response detector state, independent from source.
    pub target_response_status: ResponseStatus,
    /// A separate outcome defined for every planned unit. None means missing.
    /// Do not populate this from detector non-detection unless that detector output
    /// is itself the declared estimand and its measurement error is qualified.
    pub all_event_outcome: Option<f64>,
    /// A conditional kernel defined only among Selected units. Missing selected kernels
    /// are bounded, not zero-filled. Non-selected units are outside this estimand.
    pub conditional_kernel: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StudyArm {
    pub arm_id: String,
    pub role: ArmRole,
    pub units: Vec<ObservationUnit>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueBounds { pub lower: f64, pub upper: f64 }

/// Externally calibrated operating-characteristic ranges for one binary response detector.
/// These are treated as deterministic admissible ranges, not statistical confidence intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectorOperatingBounds {
    pub sensitivity: ValueBounds,
    pub specificity: ValueBounds,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StudyInput {
    pub protocol_id: String,
    pub estimand_id: String,
    pub selection_rule_id: String,
    pub all_event_outcome_definition: String,
    pub conditional_kernel_definition: String,
    pub all_event_outcome_bounds: ValueBounds,
    pub conditional_kernel_bounds: ValueBounds,
    pub arms: Vec<StudyArm>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericInterval { pub lower: f64, pub upper: f64 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStatus {
    PointIdentifiedForFiniteCohort,
    PartiallyIdentified,
    NoObservedMeasurements,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundedMeanResult {
    pub target_units: usize,
    pub observed_units: usize,
    pub missing_units: usize,
    pub observed_mean: Option<f64>,
    pub bounds: Option<NumericInterval>,
    pub status: BoundStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResponseStatusCounts {
    pub detected: usize,
    pub not_detected: usize,
    pub unobserved: usize,
    pub classifier_invalid: usize,
    pub registration_failed: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectionSummary {
    pub planned_units: usize,
    pub selected_units: usize,
    pub not_selected_units: usize,
    pub unobserved_units: usize,
    pub classifier_invalid_units: usize,
    pub registration_failed_units: usize,
    /// Selected / (Selected + evaluated NotSelected). Unknown statuses are excluded.
    pub selection_rate_among_evaluable: Option<f64>,
    /// Selected / all planned. Not a response probability when selection is unknown.
    pub selected_fraction_of_planned: f64,
    pub selection_evaluable_units: usize,
    pub selection_unknown_units: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArmReport {
    pub arm_id: String,
    pub role: ArmRole,
    pub selection: SelectionSummary,
    pub source_response_counts: ResponseStatusCounts,
    pub target_response_counts: ResponseStatusCounts,
    /// Bounded over every planned unit, regardless of response-selection status.
    pub all_event_outcome: BoundedMeanResult,
    /// Bounded only over selected units; non-selected units are not imputed.
    pub conditional_kernel_among_selected: BoundedMeanResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastClassification { PositiveOnly, NegativeOnly, IncludesZero }

#[derive(Debug, Clone, PartialEq)]
pub struct ArmContrastBounds {
    /// Treatment finite-cohort mean minus control finite-cohort mean; not automatically causal.
    pub bounds: NumericInterval,
    pub classification: ContrastClassification,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaimBoundary {
    pub causal_identification_verified: bool,
    pub sampling_uncertainty_included: bool,
    pub detector_error_corrected: bool,
    pub conditional_kernel_transport_claimed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StudyReport {
    pub report_version: &'static str,
    pub protocol_id: String,
    pub estimand_id: String,
    pub selection_rule_id: String,
    pub all_event_outcome_definition: String,
    pub conditional_kernel_definition: String,
    pub arms: Vec<ArmReport>,
    pub all_event_arm_contrast: ArmContrastBounds,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    EmptyIdentifier(&'static str),
    InvalidBounds(&'static str),
    WrongArmCount,
    DuplicateArmId(String),
    DuplicateArmRole,
    EmptyArm(String),
    DuplicateUnitId(String),
    NonFiniteValue { unit_id: String, field: &'static str },
    ValueOutsideBounds { unit_id: String, field: &'static str },
    KernelValueForNonSelectedUnit(String),
    SelectionStatusMismatch(String),
    InvalidProbability(&'static str),
    InfeasibleDetectorCalibration,
    NumericOverflow,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(x) => write!(f, "{x} must not be empty"),
            Self::InvalidBounds(x) => write!(f, "{x} bounds must be finite and ordered"),
            Self::WrongArmCount => write!(f, "exactly one treatment and one control arm are required"),
            Self::DuplicateArmId(x) => write!(f, "duplicate arm id: {x}"),
            Self::DuplicateArmRole => write!(f, "exactly one arm per role is required"),
            Self::EmptyArm(x) => write!(f, "arm {x} contains no planned units"),
            Self::DuplicateUnitId(x) => write!(f, "unit id is empty or not globally unique: {x}"),
            Self::NonFiniteValue { unit_id, field } => write!(f, "{field} for {unit_id} is not finite"),
            Self::ValueOutsideBounds { unit_id, field } => write!(f, "{field} for {unit_id} is outside its declared bounds"),
            Self::KernelValueForNonSelectedUnit(x) => write!(f, "non-selected or unevaluable unit {x} has a conditional kernel"),
            Self::SelectionStatusMismatch(x) => write!(f, "combined selection status does not match source/target response states for {x}"),
            Self::InvalidProbability(x) => write!(f, "{x} must be finite and in [0,1]"),
            Self::InfeasibleDetectorCalibration => write!(f, "observed detector-positive rate is incompatible with all declared sensitivity/specificity values"),
            Self::NumericOverflow => write!(f, "numeric overflow while computing bounds"),
        }
    }
}
impl std::error::Error for EvalError {}

/// Derive the combined source-and-target selection state without discarding either raw state.
///
/// A measured NotDetected status makes the conjunctive selection rule fail even if the
/// other detector is unknown; the unknown source/target status remains separately retained.
fn derive_selection_status(source: ResponseStatus, target: ResponseStatus) -> SelectionStatus {
    use ResponseStatus::*;
    if source == Detected && target == Detected {
        SelectionStatus::Selected
    } else if source == NotDetected || target == NotDetected {
        SelectionStatus::NotSelected
    } else if source == RegistrationFailed || target == RegistrationFailed {
        SelectionStatus::RegistrationFailed
    } else if source == ClassifierInvalid || target == ClassifierInvalid {
        SelectionStatus::ClassifierInvalid
    } else {
        SelectionStatus::Unobserved
    }
}

fn validate_bounds(b: ValueBounds, name: &'static str) -> Result<(), EvalError> {
    if !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper {
        return Err(EvalError::InvalidBounds(name));
    }
    Ok(())
}


fn validate_probability_bounds(bounds: ValueBounds, field: &'static str) -> Result<(), EvalError> {
    if !bounds.lower.is_finite() || !bounds.upper.is_finite()
        || bounds.lower < 0.0 || bounds.upper > 1.0 || bounds.lower > bounds.upper
    {
        return Err(EvalError::InvalidProbability(field));
    }
    Ok(())
}

/// Intersect [lower, upper] with coefficient * p <= rhs.
fn intersect_linear_leq(lower: &mut f64, upper: &mut f64, coefficient: f64, rhs: f64) -> bool {
    if coefficient > 0.0 {
        *upper = (*upper).min(rhs / coefficient);
    } else if coefficient < 0.0 {
        *lower = (*lower).max(rhs / coefficient);
    } else if rhs < 0.0 {
        return false;
    }
    *lower <= *upper
}

/// Partially identify latent binary response prevalence from an observed detector-positive
/// fraction and externally qualified sensitivity/specificity ranges.
///
/// q = Se*p + (1-Sp)*(1-p), where p is latent prevalence. Sensitivity and specificity
/// independently vary inside their declared rectangles. The result contains every p in
/// [0,1] compatible with at least one operating point. It does not assume source/target
/// detector independence; marginal intervals must never be multiplied into a joint rate.
///
/// The interval is conditional on the population represented by the observed fraction.
/// The caller's operating ranges are treated as fixed admissible ranges, not as estimated
/// confidence intervals. This function does not correct trace missingness or transport
/// detector calibration across animals/sessions.
pub fn latent_prevalence_bounds(
    observed_positive_fraction: f64,
    operating: DetectorOperatingBounds,
) -> Result<NumericInterval, EvalError> {
    if !observed_positive_fraction.is_finite()
        || !(0.0..=1.0).contains(&observed_positive_fraction)
    {
        return Err(EvalError::InvalidProbability("observed_positive_fraction"));
    }
    validate_probability_bounds(operating.sensitivity, "sensitivity")?;
    validate_probability_bounds(operating.specificity, "specificity")?;

    // For fixed p, the possible observed rates form:
    // q_min(p)=Se_L*p+(1-Sp_U)*(1-p)
    // q_max(p)=Se_U*p+(1-Sp_L)*(1-p).
    // Feasibility requires q_min(p) <= q <= q_max(p), two linear constraints in p.
    let q = observed_positive_fraction;
    let min_slope = operating.sensitivity.lower + operating.specificity.upper - 1.0;
    let min_rhs = q + operating.specificity.upper - 1.0;
    let max_slope = operating.sensitivity.upper + operating.specificity.lower - 1.0;
    let max_rhs = q + operating.specificity.lower - 1.0;

    let mut lower = 0.0;
    let mut upper = 1.0;
    if !intersect_linear_leq(&mut lower, &mut upper, min_slope, min_rhs)
        || !intersect_linear_leq(&mut lower, &mut upper, -max_slope, -max_rhs)
        || lower > upper
        || !lower.is_finite()
        || !upper.is_finite()
    {
        return Err(EvalError::InfeasibleDetectorCalibration);
    }
    Ok(NumericInterval { lower: lower.max(0.0), upper: upper.min(1.0) })
}

fn validate(input: &StudyInput) -> Result<(), EvalError> {
    for (value, name) in [
        (input.protocol_id.as_str(), "protocol_id"),
        (input.estimand_id.as_str(), "estimand_id"),
        (input.selection_rule_id.as_str(), "selection_rule_id"),
        (input.all_event_outcome_definition.as_str(), "all_event_outcome_definition"),
        (input.conditional_kernel_definition.as_str(), "conditional_kernel_definition"),
    ] {
        if value.trim().is_empty() { return Err(EvalError::EmptyIdentifier(name)); }
    }
    validate_bounds(input.all_event_outcome_bounds, "all_event_outcome")?;
    validate_bounds(input.conditional_kernel_bounds, "conditional_kernel")?;
    if input.arms.len() != 2 { return Err(EvalError::WrongArmCount); }

    let mut arm_ids = BTreeSet::new();
    let mut unit_ids = BTreeSet::new();
    let (mut treatments, mut controls) = (0usize, 0usize);
    for arm in &input.arms {
        if arm.arm_id.trim().is_empty() { return Err(EvalError::EmptyIdentifier("arm_id")); }
        if !arm_ids.insert(arm.arm_id.as_str()) {
            return Err(EvalError::DuplicateArmId(arm.arm_id.clone()));
        }
        match arm.role { ArmRole::Treatment => treatments += 1, ArmRole::Control => controls += 1 }
        if arm.units.is_empty() { return Err(EvalError::EmptyArm(arm.arm_id.clone())); }
        for unit in &arm.units {
            if derive_selection_status(unit.source_response_status, unit.target_response_status) != unit.selection_status {
                return Err(EvalError::SelectionStatusMismatch(unit.unit_id.clone()));
            }
            if unit.unit_id.trim().is_empty() || !unit_ids.insert(unit.unit_id.as_str()) {
                return Err(EvalError::DuplicateUnitId(unit.unit_id.clone()));
            }
            if let Some(v) = unit.all_event_outcome {
                if !v.is_finite() {
                    return Err(EvalError::NonFiniteValue { unit_id: unit.unit_id.clone(), field: "all_event_outcome" });
                }
                if v < input.all_event_outcome_bounds.lower || v > input.all_event_outcome_bounds.upper {
                    return Err(EvalError::ValueOutsideBounds { unit_id: unit.unit_id.clone(), field: "all_event_outcome" });
                }
            }
            if let Some(v) = unit.conditional_kernel {
                if !v.is_finite() {
                    return Err(EvalError::NonFiniteValue { unit_id: unit.unit_id.clone(), field: "conditional_kernel" });
                }
                if v < input.conditional_kernel_bounds.lower || v > input.conditional_kernel_bounds.upper {
                    return Err(EvalError::ValueOutsideBounds { unit_id: unit.unit_id.clone(), field: "conditional_kernel" });
                }
                if unit.selection_status != SelectionStatus::Selected {
                    return Err(EvalError::KernelValueForNonSelectedUnit(unit.unit_id.clone()));
                }
            }
        }
    }
    if treatments != 1 || controls != 1 { return Err(EvalError::DuplicateArmRole); }
    Ok(())
}

fn bounded_mean(
    target: usize, observed_sum: f64, observed: usize, range: ValueBounds,
) -> Result<BoundedMeanResult, EvalError> {
    if target == 0 {
        return Ok(BoundedMeanResult {
            target_units: 0, observed_units: 0, missing_units: 0, observed_mean: None,
            bounds: None, status: BoundStatus::NotEstimable,
        });
    }
    let missing = target.checked_sub(observed).ok_or(EvalError::NumericOverflow)?;
    let observed_mean = if observed == 0 { None } else { Some(observed_sum / observed as f64) };
    let lower = (observed_sum + missing as f64 * range.lower) / target as f64;
    let upper = (observed_sum + missing as f64 * range.upper) / target as f64;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(EvalError::NumericOverflow);
    }
    let status = if observed == 0 { BoundStatus::NoObservedMeasurements }
        else if missing == 0 { BoundStatus::PointIdentifiedForFiniteCohort }
        else { BoundStatus::PartiallyIdentified };
    Ok(BoundedMeanResult {
        target_units: target, observed_units: observed, missing_units: missing,
        observed_mean, bounds: Some(NumericInterval { lower, upper }), status,
    })
}

fn count_response_status(arm: &StudyArm, source: bool) -> ResponseStatusCounts {
    let mut counts = ResponseStatusCounts::default();
    for unit in &arm.units {
        let status = if source { unit.source_response_status } else { unit.target_response_status };
        match status {
            ResponseStatus::Detected => counts.detected += 1,
            ResponseStatus::NotDetected => counts.not_detected += 1,
            ResponseStatus::Unobserved => counts.unobserved += 1,
            ResponseStatus::ClassifierInvalid => counts.classifier_invalid += 1,
            ResponseStatus::RegistrationFailed => counts.registration_failed += 1,
        }
    }
    counts
}

fn evaluate_arm(arm: &StudyArm, input: &StudyInput) -> Result<ArmReport, EvalError> {
    let (mut selected, mut not_selected, mut unobserved) = (0usize, 0usize, 0usize);
    let (mut classifier_invalid, mut registration_failed) = (0usize, 0usize);
    let (mut all_sum, mut all_n, mut kernel_sum, mut kernel_n) = (0.0f64, 0usize, 0.0f64, 0usize);
    for unit in &arm.units {
        match unit.selection_status {
            SelectionStatus::Selected => selected += 1,
            SelectionStatus::NotSelected => not_selected += 1,
            SelectionStatus::Unobserved => unobserved += 1,
            SelectionStatus::ClassifierInvalid => classifier_invalid += 1,
            SelectionStatus::RegistrationFailed => registration_failed += 1,
        }
        if let Some(v) = unit.all_event_outcome { all_sum += v; all_n += 1; }
        if let Some(v) = unit.conditional_kernel { kernel_sum += v; kernel_n += 1; }
    }
    if !all_sum.is_finite() || !kernel_sum.is_finite() { return Err(EvalError::NumericOverflow); }
    let planned = arm.units.len();
    let evaluable = selected + not_selected;
    let unknown = unobserved + classifier_invalid + registration_failed;
    let selection = SelectionSummary {
        planned_units: planned,
        selected_units: selected,
        not_selected_units: not_selected,
        unobserved_units: unobserved,
        classifier_invalid_units: classifier_invalid,
        registration_failed_units: registration_failed,
        selection_rate_among_evaluable: if evaluable == 0 { None } else { Some(selected as f64 / evaluable as f64) },
        selected_fraction_of_planned: selected as f64 / planned as f64,
        selection_evaluable_units: evaluable,
        selection_unknown_units: unknown,
    };
    Ok(ArmReport {
        arm_id: arm.arm_id.clone(),
        role: arm.role,
        selection,
        source_response_counts: count_response_status(arm, true),
        target_response_counts: count_response_status(arm, false),
        all_event_outcome: bounded_mean(planned, all_sum, all_n, input.all_event_outcome_bounds)?,
        conditional_kernel_among_selected: bounded_mean(selected, kernel_sum, kernel_n, input.conditional_kernel_bounds)?,
    })
}

/// Compute worst-case finite-cohort bounds without imputing missing outcomes or kernels.
///
/// The all-event outcome must be defined for every planned unit. Its missing values are
/// bounded over the complete cohort. Conditional kernels use only Selected units. A
/// NotSelected status is not a neural zero; unevaluable selection is not a negative result.
/// The between-arm contrast is descriptive unless a separate design audit establishes
/// causal identification conditions.
pub fn evaluate(input: &StudyInput) -> Result<StudyReport, EvalError> {
    validate(input)?;
    let reports = input.arms.iter().map(|a| evaluate_arm(a, input)).collect::<Result<Vec<_>, _>>()?;
    let treatment = reports.iter().find(|a| a.role == ArmRole::Treatment).ok_or(EvalError::WrongArmCount)?;
    let control = reports.iter().find(|a| a.role == ArmRole::Control).ok_or(EvalError::WrongArmCount)?;
    let t = treatment.all_event_outcome.bounds.ok_or(EvalError::WrongArmCount)?;
    let c = control.all_event_outcome.bounds.ok_or(EvalError::WrongArmCount)?;
    let lower = t.lower - c.upper;
    let upper = t.upper - c.lower;
    if !lower.is_finite() || !upper.is_finite() || lower > upper { return Err(EvalError::NumericOverflow); }
    let classification = if lower > 0.0 { ContrastClassification::PositiveOnly }
        else if upper < 0.0 { ContrastClassification::NegativeOnly }
        else { ContrastClassification::IncludesZero };
    Ok(StudyReport {
        report_version: "connectome-selection-bounds-v1",
        protocol_id: input.protocol_id.clone(),
        estimand_id: input.estimand_id.clone(),
        selection_rule_id: input.selection_rule_id.clone(),
        all_event_outcome_definition: input.all_event_outcome_definition.clone(),
        conditional_kernel_definition: input.conditional_kernel_definition.clone(),
        arms: reports,
        all_event_arm_contrast: ArmContrastBounds { bounds: NumericInterval { lower, upper }, classification },
        claim_boundary: ClaimBoundary {
            causal_identification_verified: false,
            sampling_uncertainty_included: false,
            detector_error_corrected: false,
            conditional_kernel_transport_claimed: false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn u(id: &str, s: SelectionStatus, y: Option<f64>, k: Option<f64>) -> ObservationUnit {
        let (source_response_status, target_response_status) = match s {
            SelectionStatus::Selected => (ResponseStatus::Detected, ResponseStatus::Detected),
            SelectionStatus::NotSelected => (ResponseStatus::NotDetected, ResponseStatus::Detected),
            SelectionStatus::Unobserved => (ResponseStatus::Unobserved, ResponseStatus::Detected),
            SelectionStatus::ClassifierInvalid => (ResponseStatus::ClassifierInvalid, ResponseStatus::Detected),
            SelectionStatus::RegistrationFailed => (ResponseStatus::RegistrationFailed, ResponseStatus::Detected),
        };
        ObservationUnit { unit_id: id.to_string(), selection_status: s, source_response_status,
            target_response_status, all_event_outcome: y, conditional_kernel: k }
    }
    fn study() -> StudyInput {
        StudyInput {
            protocol_id: "synthetic-selection-bounds-v1".into(),
            estimand_id: "finite-cohort-all-event-functional-outcome".into(),
            selection_rule_id: "source-and-target-response-detected-v1".into(),
            all_event_outcome_definition: "Independent bounded endpoint in [0,1] for every planned unit".into(),
            conditional_kernel_definition: "Kernel in [-1,1] among selected units".into(),
            all_event_outcome_bounds: ValueBounds { lower: 0.0, upper: 1.0 },
            conditional_kernel_bounds: ValueBounds { lower: -1.0, upper: 1.0 },
            arms: vec![
                StudyArm { arm_id: "treatment".into(), role: ArmRole::Treatment, units: vec![
                    u("t1", SelectionStatus::Selected, Some(0.9), Some(0.8)),
                    u("t2", SelectionStatus::Selected, Some(0.8), Some(0.5)),
                    u("t3", SelectionStatus::Selected, Some(0.7), None),
                    u("t4", SelectionStatus::Selected, Some(0.6), Some(0.7)),
                    u("t5", SelectionStatus::NotSelected, None, None),
                    u("t6", SelectionStatus::Unobserved, None, None),
                ]},
                StudyArm { arm_id: "control".into(), role: ArmRole::Control, units: vec![
                    u("c1", SelectionStatus::Selected, Some(0.4), Some(0.1)),
                    u("c2", SelectionStatus::Selected, Some(0.3), Some(-0.2)),
                    u("c3", SelectionStatus::NotSelected, Some(0.5), None),
                    u("c4", SelectionStatus::NotSelected, Some(0.2), None),
                    u("c5", SelectionStatus::NotSelected, Some(0.4), None),
                    u("c6", SelectionStatus::ClassifierInvalid, None, None),
                ]},
            ],
        }
    }
    fn close(a: f64, b: f64) { assert!((a-b).abs() < 1e-12, "actual={a}, expected={b}"); }

    #[test]
    fn detector_calibration_bounds_match_analytic_known_answer() {
        let bounds = latent_prevalence_bounds(
            0.6,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.8, upper: 0.9 },
                specificity: ValueBounds { lower: 0.9, upper: 0.95 },
            },
        ).unwrap();
        close(bounds.lower, 0.625);
        close(bounds.upper, 11.0/15.0);
    }

    #[test]
    fn perfect_detector_identifies_observed_prevalence() {
        let bounds = latent_prevalence_bounds(
            0.73,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 1.0, upper: 1.0 },
                specificity: ValueBounds { lower: 1.0, upper: 1.0 },
            },
        ).unwrap();
        close(bounds.lower, 0.73);
        close(bounds.upper, 0.73);
    }

    #[test]
    fn uninformative_detector_does_not_create_false_precision() {
        let bounds = latent_prevalence_bounds(
            0.41,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.0, upper: 1.0 },
                specificity: ValueBounds { lower: 0.0, upper: 1.0 },
            },
        ).unwrap();
        close(bounds.lower, 0.0);
        close(bounds.upper, 1.0);
    }

    #[test]
    fn incompatible_detector_calibration_is_rejected() {
        let result = latent_prevalence_bounds(
            0.99,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.8, upper: 0.9 },
                specificity: ValueBounds { lower: 0.9, upper: 0.95 },
            },
        );
        assert_eq!(result.unwrap_err(), EvalError::InfeasibleDetectorCalibration);
    }

    #[test]
    fn invalid_detector_probability_ranges_fail_closed() {
        let result = latent_prevalence_bounds(
            0.4,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: -0.1, upper: 0.9 },
                specificity: ValueBounds { lower: 0.9, upper: 1.1 },
            },
        );
        assert_eq!(result.unwrap_err(), EvalError::InvalidProbability("sensitivity"));
        let result = latent_prevalence_bounds(
            f64::NAN,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.8, upper: 0.9 },
                specificity: ValueBounds { lower: 0.9, upper: 0.95 },
            },
        );
        assert_eq!(result.unwrap_err(), EvalError::InvalidProbability("observed_positive_fraction"));
    }

    #[test]
    fn wider_detector_uncertainty_cannot_narrow_prevalence_bounds() {
        let narrow = latent_prevalence_bounds(
            0.6,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.8, upper: 0.9 },
                specificity: ValueBounds { lower: 0.9, upper: 0.95 },
            },
        ).unwrap();
        let wide = latent_prevalence_bounds(
            0.6,
            DetectorOperatingBounds {
                sensitivity: ValueBounds { lower: 0.75, upper: 0.95 },
                specificity: ValueBounds { lower: 0.85, upper: 0.98 },
            },
        ).unwrap();
        assert!(wide.lower <= narrow.lower);
        assert!(wide.upper >= narrow.upper);
    }

    #[test]
    fn known_answer_manski_bounds_match_hand_calculation() {
        let r = evaluate(&study()).unwrap();
        let t = r.arms.iter().find(|a| a.role == ArmRole::Treatment).unwrap().all_event_outcome.bounds.unwrap();
        let c = r.arms.iter().find(|a| a.role == ArmRole::Control).unwrap().all_event_outcome.bounds.unwrap();
        close(t.lower, 0.5); close(t.upper, 5.0/6.0);
        close(c.lower, 0.3); close(c.upper, 2.8/6.0);
        close(r.all_event_arm_contrast.bounds.lower, 0.5 - 2.8/6.0);
        close(r.all_event_arm_contrast.bounds.upper, 5.0/6.0 - 0.3);
        assert_eq!(r.all_event_arm_contrast.classification, ContrastClassification::PositiveOnly);
    }
    #[test]
    fn conditional_kernel_uses_selected_denominator_only() {
        let r = evaluate(&study()).unwrap();
        let t = r.arms.iter().find(|a| a.role == ArmRole::Treatment).unwrap();
        let k = &t.conditional_kernel_among_selected;
        assert_eq!(k.target_units, 4); assert_eq!(k.observed_units, 3); assert_eq!(k.missing_units, 1);
        close(k.observed_mean.unwrap(), 2.0/3.0);
        close(k.bounds.unwrap().lower, 0.25); close(k.bounds.unwrap().upper, 0.75);
    }
    #[test]
    fn unknown_selection_is_not_counted_as_not_selected() {
        let r = evaluate(&study()).unwrap();
        let t = r.arms.iter().find(|a| a.role == ArmRole::Treatment).unwrap();
        assert_eq!(t.selection.selected_units, 4);
        assert_eq!(t.selection.not_selected_units, 1);
        assert_eq!(t.selection.unobserved_units, 1);
        assert_eq!(t.selection.selection_evaluable_units, 5);
        assert_eq!(t.selection.selection_unknown_units, 1);
        close(t.selection.selection_rate_among_evaluable.unwrap(), 0.8);
        close(t.selection.selected_fraction_of_planned, 4.0/6.0);
    }
    #[test]
    fn nonselected_unit_can_still_have_a_distinct_all_event_outcome() {
        let r = evaluate(&study()).unwrap();
        let c = r.arms.iter().find(|a| a.role == ArmRole::Control).unwrap();
        assert_eq!(c.all_event_outcome.observed_units, 5);
        assert_eq!(c.conditional_kernel_among_selected.target_units, 2);
        close(c.conditional_kernel_among_selected.bounds.unwrap().lower, -0.05);
    }
    #[test]
    fn no_selected_units_makes_conditional_kernel_not_estimable() {
        let mut x = study();
        for arm in &mut x.arms { for unit in &mut arm.units {
            unit.selection_status = SelectionStatus::NotSelected; unit.conditional_kernel = None;
        }}
        let r = evaluate(&x).unwrap();
        assert!(r.arms.iter().all(|a| a.conditional_kernel_among_selected.bounds.is_none()));
    }
    #[test]
    fn no_observed_outcomes_return_full_declared_range() {
        let mut x = study();
        for arm in &mut x.arms { for unit in &mut arm.units { unit.all_event_outcome = None; }}
        let r = evaluate(&x).unwrap();
        for arm in r.arms {
            let y = arm.all_event_outcome;
            assert_eq!(y.status, BoundStatus::NoObservedMeasurements);
            close(y.bounds.unwrap().lower, 0.0); close(y.bounds.unwrap().upper, 1.0);
        }
    }
    #[test]
    fn complete_outcomes_are_point_identified_for_finite_cohort() {
        let mut x = study();
        for arm in &mut x.arms { for unit in &mut arm.units { unit.all_event_outcome = Some(0.5); }}
        let r = evaluate(&x).unwrap();
        for arm in r.arms {
            assert_eq!(arm.all_event_outcome.status, BoundStatus::PointIdentifiedForFiniteCohort);
            close(arm.all_event_outcome.bounds.unwrap().lower, 0.5);
            close(arm.all_event_outcome.bounds.unwrap().upper, 0.5);
        }
    }
    #[test]
    fn adding_an_observed_outcome_shrinks_or_preserves_bounds() {
        let mut x = study();
        let before = evaluate(&x).unwrap();
        let b = before.arms[0].all_event_outcome.bounds.unwrap();
        let width_before = b.upper-b.lower;
        x.arms[0].units[4].all_event_outcome = Some(0.2);
        let after = evaluate(&x).unwrap();
        let a = after.arms[0].all_event_outcome.bounds.unwrap();
        assert!(a.upper-a.lower < width_before);
    }
    #[test]
    fn kernel_value_on_nonselected_unit_is_rejected() {
        let mut x = study(); x.arms[0].units[4].conditional_kernel = Some(0.0);
        assert_eq!(evaluate(&x).unwrap_err(), EvalError::KernelValueForNonSelectedUnit("t5".into()));
    }
    #[test]
    fn duplicate_unit_ids_across_arms_are_rejected() {
        let mut x = study(); x.arms[1].units[0].unit_id = "t1".into();
        assert!(matches!(evaluate(&x), Err(EvalError::DuplicateUnitId(_))));
    }
    #[test]
    fn malformed_bounds_and_out_of_range_values_fail_closed() {
        let mut x = study(); x.all_event_outcome_bounds = ValueBounds { lower: 2.0, upper: 1.0 };
        assert!(matches!(evaluate(&x), Err(EvalError::InvalidBounds(_))));
        let mut x = study(); x.arms[0].units[0].all_event_outcome = Some(1.1);
        assert!(matches!(evaluate(&x), Err(EvalError::ValueOutsideBounds { .. })));
    }
    #[test]
    fn wrong_arm_roles_are_rejected() {
        let mut x = study(); x.arms[1].role = ArmRole::Treatment;
        assert_eq!(evaluate(&x).unwrap_err(), EvalError::DuplicateArmRole);
    }
    #[test]
    fn source_and_target_detector_states_are_separately_reported() {
        let mut x = study();
        x.arms[0].units[4].source_response_status = ResponseStatus::NotDetected;
        x.arms[0].units[4].target_response_status = ResponseStatus::RegistrationFailed;
        let r = evaluate(&x).unwrap();
        let t = r.arms.iter().find(|a| a.role == ArmRole::Treatment).unwrap();
        assert_eq!(t.source_response_counts.not_detected, 1);
        assert_eq!(t.target_response_counts.registration_failed, 1);
        assert_eq!(t.selection.not_selected_units, 1);
        assert_eq!(t.selection.registration_failed_units, 0);
    }
    #[test]
    fn inconsistent_combined_selection_status_is_rejected() {
        let mut x = study();
        x.arms[0].units[0].target_response_status = ResponseStatus::NotDetected;
        assert_eq!(evaluate(&x).unwrap_err(), EvalError::SelectionStatusMismatch("t1".into()));
    }
    #[test]
    fn report_hard_codes_claim_boundaries() {
        let r = evaluate(&study()).unwrap();
        assert_eq!(r.report_version, "connectome-selection-bounds-v1");
        assert!(!r.claim_boundary.causal_identification_verified);
        assert!(!r.claim_boundary.sampling_uncertainty_included);
        assert!(!r.claim_boundary.detector_error_corrected);
        assert!(!r.claim_boundary.conditional_kernel_transport_claimed);
    }
}
