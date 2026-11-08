//! Evidence-bounded primitives for marine structural-health inspection decisions.
//!
//! This crate deliberately does not calculate fatigue life, corrosion kinetics,
//! structural strength, NDT indications, inspection intervals, or seaworthiness.
//! It evaluates only declared evidence gates and uncertainty intervals supplied
//! by competent engineering / inspection authorities.
//!
//! A positive result is scoped to one region, method, criterion, and evidence
//! record. It is never a whole-vessel structural acceptance.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MethodQualification {
    QualifiedForDeclaredScope,
    ResearchOnly,
    Unqualified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataQuality {
    Valid,
    Missing,
    Stale,
    DriftSuspected,
    Saturated,
    Unsynchronized,
    OutOfDomain,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationStatus {
    Current,
    Expired,
    DriftSuspected,
    Missing,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InspectionResult {
    NoIndication,
    Indication,
    Measurement {
        value: f64,
        expanded_uncertainty: f64,
    },
    Inconclusive,
    NotPerformed,
}

/// Acceptance interval supplied by the responsible structural / inspection
/// authority. None means that side of the interval is not applicable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptanceBounds {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

/// Exact inspection evidence required by the interval evaluator.
///
/// Region IDs must refer to the same immutable component/region identity and
/// revision. The coverage threshold is supplied by the qualified method owner;
/// this crate does not invent one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InspectionEvidence<'a> {
    pub expected_region_id: &'a str,
    pub observed_region_id: &'a str,
    pub expected_structure_revision: &'a str,
    pub observed_structure_revision: &'a str,
    pub acceptance_criterion_ref: Option<&'a str>,
    pub method_qualification: MethodQualification,
    pub data_quality: DataQuality,
    pub calibration: CalibrationStatus,
    pub coverage_fraction: f64,
    pub required_coverage_fraction: f64,
    pub result: InspectionResult,
    pub bounds: AcceptanceBounds,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndeterminateReason {
    DataNotValid,
    CalibrationNotCurrent,
    CoverageIncomplete,
    InspectionInconclusive,
    InspectionNotPerformed,
    InvalidCoverage,
    InvalidMeasurement,
    MeasurementUncertaintyCrossesCriterion,
    InvalidAcceptanceBounds,
    MissingAcceptanceBound,
    MissingAcceptanceCriterion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedReason {
    MethodNotQualified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceRejection {
    RegionIdentityMismatch,
    StructureRevisionMismatch,
}

/// A positive result applies only to the exact inspected scope. It is not a
/// declaration that the rest of the hull or the whole vessel is undamaged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InspectionAssessment {
    NoIndicationWithinDeclaredScope {
        coverage_fraction: f64,
    },
    MeasurementWithinDeclaredCriterion {
        interval_lower: f64,
        interval_upper: f64,
    },
    RejectedByCriterion {
        interval_lower: f64,
        interval_upper: f64,
    },
    InspectionRequired,
    Indeterminate(IndeterminateReason),
    Unsupported(UnsupportedReason),
    EvidenceRejected(EvidenceRejection),
}

fn finite_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_bounds(bounds: AcceptanceBounds) -> bool {
    match (bounds.lower, bounds.upper) {
        (Some(lower), Some(upper)) => {
            lower.is_finite() && upper.is_finite() && lower <= upper
        }
        (Some(lower), None) => lower.is_finite(),
        (None, Some(upper)) => upper.is_finite(),
        (None, None) => false,
    }
}

/// Evaluate one inspection record using conservative interval containment.
/// Invalid, stale, uncalibrated, unqualified or incompletely covered evidence
/// cannot produce a positive disposition.
pub fn evaluate_inspection(
    evidence: InspectionEvidence<'_>,
) -> InspectionAssessment {
    if evidence.expected_region_id != evidence.observed_region_id {
        return InspectionAssessment::EvidenceRejected(
            EvidenceRejection::RegionIdentityMismatch,
        );
    }

    if evidence.expected_structure_revision != evidence.observed_structure_revision {
        return InspectionAssessment::EvidenceRejected(
            EvidenceRejection::StructureRevisionMismatch,
        );
    }

    if evidence.method_qualification != MethodQualification::QualifiedForDeclaredScope {
        return InspectionAssessment::Unsupported(
            UnsupportedReason::MethodNotQualified,
        );
    }

    if evidence.data_quality != DataQuality::Valid {
        return InspectionAssessment::Indeterminate(
            IndeterminateReason::DataNotValid,
        );
    }

    if evidence.calibration != CalibrationStatus::Current {
        return InspectionAssessment::Indeterminate(
            IndeterminateReason::CalibrationNotCurrent,
        );
    }

    if !finite_unit_interval(evidence.coverage_fraction)
        || !finite_unit_interval(evidence.required_coverage_fraction)
    {
        return InspectionAssessment::Indeterminate(
            IndeterminateReason::InvalidCoverage,
        );
    }

    if evidence.coverage_fraction < evidence.required_coverage_fraction {
        return InspectionAssessment::Indeterminate(
            IndeterminateReason::CoverageIncomplete,
        );
    }

    if evidence
        .acceptance_criterion_ref
        .is_none_or(|criterion| criterion.trim().is_empty())
    {
        return InspectionAssessment::Indeterminate(
            IndeterminateReason::MissingAcceptanceCriterion,
        );
    }

    match evidence.result {
        InspectionResult::NoIndication => {
            InspectionAssessment::NoIndicationWithinDeclaredScope {
                coverage_fraction: evidence.coverage_fraction,
            }
        }
        InspectionResult::Indication => InspectionAssessment::InspectionRequired,
        InspectionResult::Inconclusive => InspectionAssessment::Indeterminate(
            IndeterminateReason::InspectionInconclusive,
        ),
        InspectionResult::NotPerformed => InspectionAssessment::Indeterminate(
            IndeterminateReason::InspectionNotPerformed,
        ),
        InspectionResult::Measurement {
            value,
            expanded_uncertainty,
        } => {
            if !value.is_finite()
                || !expanded_uncertainty.is_finite()
                || expanded_uncertainty < 0.0
            {
                return InspectionAssessment::Indeterminate(
                    IndeterminateReason::InvalidMeasurement,
                );
            }

            if !valid_bounds(evidence.bounds) {
                return InspectionAssessment::Indeterminate(
                    if evidence.bounds.lower.is_none()
                        && evidence.bounds.upper.is_none()
                    {
                        IndeterminateReason::MissingAcceptanceBound
                    } else {
                        IndeterminateReason::InvalidAcceptanceBounds
                    },
                );
            }

            let interval_lower = value - expanded_uncertainty;
            let interval_upper = value + expanded_uncertainty;

            let fully_inside = evidence
                .bounds
                .lower
                .is_none_or(|lower| interval_lower >= lower)
                && evidence
                    .bounds
                    .upper
                    .is_none_or(|upper| interval_upper <= upper);

            if fully_inside {
                return InspectionAssessment::MeasurementWithinDeclaredCriterion {
                    interval_lower,
                    interval_upper,
                };
            }

            let entirely_below = evidence
                .bounds
                .lower
                .is_some_and(|lower| interval_upper < lower);
            let entirely_above = evidence
                .bounds
                .upper
                .is_some_and(|upper| interval_lower > upper);

            if entirely_below || entirely_above {
                InspectionAssessment::RejectedByCriterion {
                    interval_lower,
                    interval_upper,
                }
            } else {
                InspectionAssessment::Indeterminate(
                    IndeterminateReason::MeasurementUncertaintyCrossesCriterion,
                )
            }
        }
    }
}

/// Declared expected-loss and acquisition-cost inputs for VoI. These are not
/// damage probabilities. The model owner must explicitly qualify the priors,
/// likelihoods, consequences and cost boundary before setting models_qualified.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiInputs {
    pub expected_loss_before_information: f64,
    pub expected_loss_after_information: f64,
    pub acquisition_and_downtime_cost: f64,
    pub models_qualified: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VoiResult {
    Calculated {
        evsi: f64,
        net_voi: f64,
    },
    Unknown,
    Invalid,
}

/// EVSI = expected loss before information - expected loss after information.
/// Net VoI = EVSI - acquisition and downtime cost.
///
/// Negative EVSI is an inconsistent input contract: with an optimal decision
/// rule, optional information should not increase expected decision loss before
/// acquisition cost.
pub fn calculate_voi(inputs: VoiInputs) -> VoiResult {
    if !inputs.models_qualified {
        return VoiResult::Unknown;
    }

    let values = [
        inputs.expected_loss_before_information,
        inputs.expected_loss_after_information,
        inputs.acquisition_and_downtime_cost,
    ];
    if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return VoiResult::Invalid;
    }

    let evsi =
        inputs.expected_loss_before_information - inputs.expected_loss_after_information;
    if evsi < 0.0 {
        return VoiResult::Invalid;
    }

    VoiResult::Calculated {
        evsi,
        net_voi: evsi - inputs.acquisition_and_downtime_cost,
    }
}

/// Advisory actions only. This enum never grants permission to operate,
/// inspect, repair, defer a survey, or return an asset to service.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdvisoryAction {
    ContinueExistingApprovedPlan,
    TargetedInspection,
    CompetentStructuralReview,
    AuthorityEscalation,
    NoRecommendation,
}

/// Mandatory inspection requirements dominate optional negative VoI. The
/// function cannot extend an authority deadline or change approved intervals.
pub fn recommend_action(
    assessment: InspectionAssessment,
    mandatory_inspection_due: bool,
    voi: VoiResult,
) -> AdvisoryAction {
    match assessment {
        InspectionAssessment::EvidenceRejected(_) => {
            AdvisoryAction::CompetentStructuralReview
        }
        InspectionAssessment::Unsupported(_) => AdvisoryAction::NoRecommendation,
        InspectionAssessment::Indeterminate(_) => {
            AdvisoryAction::CompetentStructuralReview
        }
        InspectionAssessment::InspectionRequired
        | InspectionAssessment::RejectedByCriterion { .. } => {
            AdvisoryAction::AuthorityEscalation
        }
        InspectionAssessment::NoIndicationWithinDeclaredScope { .. }
        | InspectionAssessment::MeasurementWithinDeclaredCriterion { .. } => {
            if mandatory_inspection_due {
                AdvisoryAction::TargetedInspection
            } else if matches!(
                voi,
                VoiResult::Calculated { net_voi, .. } if net_voi > 0.0
            ) {
                AdvisoryAction::TargetedInspection
            } else {
                AdvisoryAction::ContinueExistingApprovedPlan
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(result: InspectionResult) -> InspectionEvidence<'static> {
        InspectionEvidence {
            expected_region_id: "plate-01",
            observed_region_id: "plate-01",
            expected_structure_revision: "REV-1",
            observed_structure_revision: "REV-1",
            acceptance_criterion_ref: Some("criterion:plate-thickness-v1"),
            method_qualification: MethodQualification::QualifiedForDeclaredScope,
            data_quality: DataQuality::Valid,
            calibration: CalibrationStatus::Current,
            coverage_fraction: 0.99,
            required_coverage_fraction: 0.95,
            result,
            bounds: AcceptanceBounds {
                lower: Some(7.0),
                upper: None,
            },
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-9);
    }

    #[test]
    fn uncertainty_interval_fully_inside_is_within_scope() {
        match evaluate_inspection(evidence(InspectionResult::Measurement {
            value: 8.0,
            expanded_uncertainty: 0.2,
        })) {
            InspectionAssessment::MeasurementWithinDeclaredCriterion {
                interval_lower,
                interval_upper,
            } => {
                assert_close(interval_lower, 7.8);
                assert_close(interval_upper, 8.2);
            }
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn uncertainty_interval_crossing_limit_is_indeterminate() {
        assert_eq!(
            evaluate_inspection(evidence(InspectionResult::Measurement {
                value: 7.1,
                expanded_uncertainty: 0.3,
            })),
            InspectionAssessment::Indeterminate(
                IndeterminateReason::MeasurementUncertaintyCrossesCriterion
            )
        );
    }

    #[test]
    fn interval_entirely_below_limit_is_rejected() {
        match evaluate_inspection(evidence(InspectionResult::Measurement {
            value: 6.0,
            expanded_uncertainty: 0.2,
        })) {
            InspectionAssessment::RejectedByCriterion {
                interval_lower,
                interval_upper,
            } => {
                assert_close(interval_lower, 5.8);
                assert_close(interval_upper, 6.2);
            }
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn stale_calibration_cannot_produce_no_indication_pass() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.calibration = CalibrationStatus::Expired;
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::Indeterminate(
                IndeterminateReason::CalibrationNotCurrent
            )
        );
    }

    #[test]
    fn incomplete_coverage_cannot_produce_no_indication_pass() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.coverage_fraction = 0.65;
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::Indeterminate(
                IndeterminateReason::CoverageIncomplete
            )
        );
    }

    #[test]
    fn region_identity_mismatch_is_rejected() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.observed_region_id = "plate-02";
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::EvidenceRejected(
                EvidenceRejection::RegionIdentityMismatch
            )
        );
    }

    #[test]
    fn superseded_structure_revision_is_rejected() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.observed_structure_revision = "REV-0";
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::EvidenceRejected(
                EvidenceRejection::StructureRevisionMismatch
            )
        );
    }

    #[test]
    fn missing_acceptance_criterion_cannot_produce_a_clear_disposition() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.acceptance_criterion_ref = None;
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::Indeterminate(
                IndeterminateReason::MissingAcceptanceCriterion
            )
        );
    }

    #[test]
    fn research_only_method_cannot_produce_acceptance() {
        let mut input = evidence(InspectionResult::NoIndication);
        input.method_qualification = MethodQualification::ResearchOnly;
        assert_eq!(
            evaluate_inspection(input),
            InspectionAssessment::Unsupported(
                UnsupportedReason::MethodNotQualified
            )
        );
    }

    #[test]
    fn positive_voi_recomputes_exactly() {
        assert_eq!(
            calculate_voi(VoiInputs {
                expected_loss_before_information: 1000.0,
                expected_loss_after_information: 400.0,
                acquisition_and_downtime_cost: 100.0,
                models_qualified: true,
            }),
            VoiResult::Calculated {
                evsi: 600.0,
                net_voi: 500.0,
            }
        );
    }

    #[test]
    fn unqualified_voi_models_return_unknown() {
        assert_eq!(
            calculate_voi(VoiInputs {
                expected_loss_before_information: 1000.0,
                expected_loss_after_information: 400.0,
                acquisition_and_downtime_cost: 100.0,
                models_qualified: false,
            }),
            VoiResult::Unknown
        );
    }

    #[test]
    fn inconsistent_voi_losses_are_invalid() {
        assert_eq!(
            calculate_voi(VoiInputs {
                expected_loss_before_information: 400.0,
                expected_loss_after_information: 1000.0,
                acquisition_and_downtime_cost: 100.0,
                models_qualified: true,
            }),
            VoiResult::Invalid
        );
    }

    #[test]
    fn mandatory_inspection_is_not_deferred_by_negative_voi() {
        let assessment =
            InspectionAssessment::NoIndicationWithinDeclaredScope {
                coverage_fraction: 0.99,
            };
        assert_eq!(
            recommend_action(
                assessment,
                true,
                VoiResult::Calculated {
                    evsi: 10.0,
                    net_voi: -100.0,
                }
            ),
            AdvisoryAction::TargetedInspection
        );
    }

    #[test]
    fn no_indication_is_scoped_and_advisory() {
        let result = evaluate_inspection(evidence(InspectionResult::NoIndication));
        assert_eq!(
            result,
            InspectionAssessment::NoIndicationWithinDeclaredScope {
                coverage_fraction: 0.99,
            }
        );
        assert_eq!(
            recommend_action(result, false, VoiResult::Unknown),
            AdvisoryAction::ContinueExistingApprovedPlan
        );
    }
}
