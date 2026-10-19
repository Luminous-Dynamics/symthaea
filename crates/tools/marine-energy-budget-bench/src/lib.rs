//! Mission-equivalent total input-energy comparison for marine design experiments.
//!
//! V0 uses electrical input energy only. It checks arithmetic and declared comparability;
//! it is not a hydrodynamic model and does not establish physical validation.
//!
//! Uncertainty bounds are conservative absolute limits. Propulsion and auxiliary bounds
//! are added rather than combined statistically; no independence/cancellation assumption
//! is made.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EvidenceOrigin {
    Synthetic,
    Analytical,
    Cfd,
    Experimental,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssessmentState {
    Ready,
    Indeterminate,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonDisposition {
    RobustSavingWithinDeclaredBounds,
    RobustWorseWithinDeclaredBounds,
    Indeterminate,
    Incomparable,
    Invalid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MissionSpec {
    pub mission_id: String,
    pub conditions_digest: String,
    pub route_distance_m: f64,
    pub payload_kg: Option<f64>,
    pub target_speed_mps: f64,
    pub arrival_policy_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnergyCase {
    pub case_id: String,
    pub mission: MissionSpec,
    /// Identity of the electrical measurement point / complete energy boundary.
    pub energy_boundary_id: String,
    /// Identity of this candidate's frozen configuration (may differ from baseline).
    pub configuration_digest: Option<String>,
    /// Source record, measurement receipt, or model-run identity.
    pub evidence_ref: Option<String>,
    pub evidence_origin: Option<EvidenceOrigin>,
    /// Must identify the same declared absolute-bound construction for both cases.
    pub uncertainty_method_id: Option<String>,
    /// Electrical input energy consumed by propulsion, in Wh.
    pub propulsion_input_wh: f64,
    /// Conservative absolute uncertainty bound for propulsion input energy, in Wh.
    pub propulsion_uncertainty_wh: f64,
    /// Sum of non-propulsion electrical inputs, in Wh. Missing is unknown, not zero.
    pub auxiliary_input_wh: Option<f64>,
    /// Conservative absolute uncertainty bound for auxiliary input, in Wh.
    pub auxiliary_uncertainty_wh: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyBand {
    pub lower_wh: f64,
    pub nominal_wh: f64,
    pub upper_wh: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractionInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaseMetrics {
    pub propulsion_input_wh: f64,
    pub auxiliary_input_wh: f64,
    pub total_input_wh: f64,
    /// Sum of conservative absolute uncertainty bounds, in Wh.
    pub total_uncertainty_wh: f64,
    pub total_energy_band: EnergyBand,
    pub wh_per_km: f64,
    /// None when payload is absent or zero; it is never fabricated as a denominator.
    pub wh_per_tonne_km: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Finding {
    MissingCaseId { case_id: String },
    MissingMissionId { case_id: String },
    MissingConditionsDigest { case_id: String },
    MissingArrivalPolicyId { case_id: String },
    MissingEnergyBoundaryId { case_id: String },
    MissingConfigurationDigest { case_id: String },
    MissingEvidenceReference { case_id: String },
    MissingEvidenceOrigin { case_id: String },
    MissingUncertaintyMethod { case_id: String },
    MissingAuxiliaryEnergy { case_id: String },
    MissingAuxiliaryUncertainty { case_id: String },
    InvalidNonFinite { case_id: String, field: String },
    InvalidNegative { case_id: String, field: String },
    InvalidNonPositive { case_id: String, field: String },
    DerivedOverflow { case_id: String, field: String },
    MissionMismatch,
    ConditionsMismatch,
    DeliveredServiceMismatch,
    EnergyBoundaryMismatch,
    EvidenceOriginMismatch,
    UncertaintyMethodMismatch,
    BaselineLowerBoundNotPositive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaseAssessment {
    pub state: AssessmentState,
    pub metrics: Option<CaseMetrics>,
    pub findings: Vec<Finding>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub disposition: ComparisonDisposition,
    pub baseline_metrics: Option<CaseMetrics>,
    pub candidate_metrics: Option<CaseMetrics>,
    pub nominal_saving_fraction: Option<f64>,
    /// Conservative bound derived from the two declared total-energy bands.
    pub saving_fraction_interval: Option<FractionInterval>,
    pub findings: Vec<Finding>,
}

fn note_nonempty(
    value: &str,
    finding: Finding,
    findings: &mut Vec<Finding>,
) {
    if value.trim().is_empty() {
        findings.push(finding);
    }
}

fn check_nonnegative(
    case_id: &str,
    field: &str,
    value: f64,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
) -> bool {
    if !value.is_finite() {
        findings.push(Finding::InvalidNonFinite {
            case_id: case_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        false
    } else if value < 0.0 {
        findings.push(Finding::InvalidNegative {
            case_id: case_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        false
    } else {
        true
    }
}

fn check_positive(
    case_id: &str,
    field: &str,
    value: f64,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
) -> bool {
    if !value.is_finite() {
        findings.push(Finding::InvalidNonFinite {
            case_id: case_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        false
    } else if value <= 0.0 {
        findings.push(Finding::InvalidNonPositive {
            case_id: case_id.to_owned(),
            field: field.to_owned(),
        });
        *invalid = true;
        false
    } else {
        true
    }
}

fn sort_findings(findings: &mut Vec<Finding>) {
    findings.sort();
    findings.dedup();
}

/// Validate one case and calculate metrics where the numeric energy/service data
/// is complete. Missing metadata yields Indeterminate but does not hide computable
/// metrics; invalid numeric inputs yield Invalid and no metrics.
pub fn assess_case(case: &EnergyCase) -> CaseAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let case_id = case.case_id.as_str();

    note_nonempty(
        &case.case_id,
        Finding::MissingCaseId { case_id: case.case_id.clone() },
        &mut findings,
    );
    note_nonempty(
        &case.mission.mission_id,
        Finding::MissingMissionId { case_id: case.case_id.clone() },
        &mut findings,
    );
    note_nonempty(
        &case.mission.conditions_digest,
        Finding::MissingConditionsDigest { case_id: case.case_id.clone() },
        &mut findings,
    );
    note_nonempty(
        &case.mission.arrival_policy_id,
        Finding::MissingArrivalPolicyId { case_id: case.case_id.clone() },
        &mut findings,
    );
    note_nonempty(
        &case.energy_boundary_id,
        Finding::MissingEnergyBoundaryId { case_id: case.case_id.clone() },
        &mut findings,
    );

    if case.configuration_digest.as_deref().is_none_or(|s| s.trim().is_empty()) {
        findings.push(Finding::MissingConfigurationDigest { case_id: case.case_id.clone() });
    }
    if case.evidence_ref.as_deref().is_none_or(|s| s.trim().is_empty()) {
        findings.push(Finding::MissingEvidenceReference { case_id: case.case_id.clone() });
    }
    if case.evidence_origin.is_none() {
        findings.push(Finding::MissingEvidenceOrigin { case_id: case.case_id.clone() });
    }
    if case.uncertainty_method_id.as_deref().is_none_or(|s| s.trim().is_empty()) {
        findings.push(Finding::MissingUncertaintyMethod { case_id: case.case_id.clone() });
    }

    let distance_ok = check_positive(
        case_id,
        "mission.route_distance_m",
        case.mission.route_distance_m,
        &mut findings,
        &mut invalid,
    );
    let speed_ok = check_positive(
        case_id,
        "mission.target_speed_mps",
        case.mission.target_speed_mps,
        &mut findings,
        &mut invalid,
    );
    let payload_ok = case.mission.payload_kg.is_none_or(|payload| {
        check_nonnegative(case_id, "mission.payload_kg", payload, &mut findings, &mut invalid)
    });
    let propulsion_ok = check_nonnegative(
        case_id,
        "propulsion_input_wh",
        case.propulsion_input_wh,
        &mut findings,
        &mut invalid,
    );
    let propulsion_uncertainty_ok = check_nonnegative(
        case_id,
        "propulsion_uncertainty_wh",
        case.propulsion_uncertainty_wh,
        &mut findings,
        &mut invalid,
    );

    let auxiliary_ok = match case.auxiliary_input_wh {
        None => {
            findings.push(Finding::MissingAuxiliaryEnergy { case_id: case.case_id.clone() });
            false
        }
        Some(auxiliary) => check_nonnegative(
            case_id,
            "auxiliary_input_wh",
            auxiliary,
            &mut findings,
            &mut invalid,
        ),
    };
    let auxiliary_uncertainty_ok = match case.auxiliary_uncertainty_wh {
        None => {
            findings.push(Finding::MissingAuxiliaryUncertainty {
                case_id: case.case_id.clone(),
            });
            false
        }
        Some(uncertainty) => check_nonnegative(
            case_id,
            "auxiliary_uncertainty_wh",
            uncertainty,
            &mut findings,
            &mut invalid,
        ),
    };

    let numeric_ready = distance_ok
        && speed_ok
        && payload_ok
        && propulsion_ok
        && propulsion_uncertainty_ok
        && auxiliary_ok
        && auxiliary_uncertainty_ok;

    let metrics = if numeric_ready && !invalid {
        let auxiliary = case.auxiliary_input_wh.expect("numeric_ready implies auxiliary energy");
        let auxiliary_uncertainty = case
            .auxiliary_uncertainty_wh
            .expect("numeric_ready implies auxiliary uncertainty");
        let total = case.propulsion_input_wh + auxiliary;
        let uncertainty = case.propulsion_uncertainty_wh + auxiliary_uncertainty;
        let lower = (total - uncertainty).max(0.0);
        let upper = total + uncertainty;
        let distance_km = case.mission.route_distance_m / 1_000.0;
        let wh_per_km = total / distance_km;
        let wh_per_tonne_km = case.mission.payload_kg.and_then(|payload_kg| {
            if payload_kg > 0.0 {
                Some(total / (distance_km * (payload_kg / 1_000.0)))
            } else {
                None
            }
        });

        let derived = [
            ("total_input_wh", total),
            ("total_uncertainty_wh", uncertainty),
            ("total_energy_lower_wh", lower),
            ("total_energy_upper_wh", upper),
            ("wh_per_km", wh_per_km),
        ];
        for (field, value) in derived {
            if !value.is_finite() {
                findings.push(Finding::DerivedOverflow {
                    case_id: case.case_id.clone(),
                    field: field.to_owned(),
                });
                invalid = true;
            }
        }
        if let Some(value) = wh_per_tonne_km {
            if !value.is_finite() {
                findings.push(Finding::DerivedOverflow {
                    case_id: case.case_id.clone(),
                    field: "wh_per_tonne_km".to_owned(),
                });
                invalid = true;
            }
        }

        if invalid {
            None
        } else {
            Some(CaseMetrics {
                propulsion_input_wh: case.propulsion_input_wh,
                auxiliary_input_wh: auxiliary,
                total_input_wh: total,
                total_uncertainty_wh: uncertainty,
                total_energy_band: EnergyBand {
                    lower_wh: lower,
                    nominal_wh: total,
                    upper_wh: upper,
                },
                wh_per_km,
                wh_per_tonne_km,
            })
        }
    } else {
        None
    };

    sort_findings(&mut findings);
    let state = if invalid {
        AssessmentState::Invalid
    } else if !findings.is_empty() || metrics.is_none() {
        AssessmentState::Indeterminate
    } else {
        AssessmentState::Ready
    };

    CaseAssessment { state, metrics, findings }
}

fn incompatible_findings(baseline: &EnergyCase, candidate: &EnergyCase) -> Vec<Finding> {
    let mut findings = Vec::new();
    if baseline.mission.mission_id != candidate.mission.mission_id {
        findings.push(Finding::MissionMismatch);
    }
    if baseline.mission.conditions_digest != candidate.mission.conditions_digest {
        findings.push(Finding::ConditionsMismatch);
    }
    let service_matches = baseline.mission.route_distance_m == candidate.mission.route_distance_m
        && baseline.mission.payload_kg == candidate.mission.payload_kg
        && baseline.mission.target_speed_mps == candidate.mission.target_speed_mps
        && baseline.mission.arrival_policy_id == candidate.mission.arrival_policy_id;
    if !service_matches {
        findings.push(Finding::DeliveredServiceMismatch);
    }
    if baseline.energy_boundary_id != candidate.energy_boundary_id {
        findings.push(Finding::EnergyBoundaryMismatch);
    }
    if baseline.evidence_origin != candidate.evidence_origin {
        findings.push(Finding::EvidenceOriginMismatch);
    }
    if baseline.uncertainty_method_id != candidate.uncertainty_method_id {
        findings.push(Finding::UncertaintyMethodMismatch);
    }
    sort_findings(&mut findings);
    findings
}

/// Compare candidate input energy with baseline input energy under exact declared
/// mission/service/evidence-basis matching and conservative energy bounds.
pub fn compare(baseline: &EnergyCase, candidate: &EnergyCase) -> ComparisonReport {
    let baseline_assessment = assess_case(baseline);
    let candidate_assessment = assess_case(candidate);
    let baseline_metrics = baseline_assessment.metrics.clone();
    let candidate_metrics = candidate_assessment.metrics.clone();
    let mut findings = baseline_assessment.findings.clone();
    findings.extend(candidate_assessment.findings.clone());
    sort_findings(&mut findings);

    if baseline_assessment.state == AssessmentState::Invalid
        || candidate_assessment.state == AssessmentState::Invalid
    {
        return ComparisonReport {
            disposition: ComparisonDisposition::Invalid,
            baseline_metrics,
            candidate_metrics,
            nominal_saving_fraction: None,
            saving_fraction_interval: None,
            findings,
        };
    }

    if baseline_assessment.state == AssessmentState::Indeterminate
        || candidate_assessment.state == AssessmentState::Indeterminate
    {
        return ComparisonReport {
            disposition: ComparisonDisposition::Indeterminate,
            baseline_metrics,
            candidate_metrics,
            nominal_saving_fraction: None,
            saving_fraction_interval: None,
            findings,
        };
    }

    let mismatches = incompatible_findings(baseline, candidate);
    if !mismatches.is_empty() {
        return ComparisonReport {
            disposition: ComparisonDisposition::Incomparable,
            baseline_metrics,
            candidate_metrics,
            nominal_saving_fraction: None,
            saving_fraction_interval: None,
            findings: mismatches,
        };
    }

    let baseline_metrics = baseline_metrics.expect("ready case has metrics");
    let candidate_metrics = candidate_metrics.expect("ready case has metrics");
    let baseline_band = baseline_metrics.total_energy_band;
    let candidate_band = candidate_metrics.total_energy_band;

    if baseline_band.lower_wh <= 0.0 {
        findings.push(Finding::BaselineLowerBoundNotPositive);
        sort_findings(&mut findings);
        return ComparisonReport {
            disposition: ComparisonDisposition::Indeterminate,
            baseline_metrics: Some(baseline_metrics),
            candidate_metrics: Some(candidate_metrics),
            nominal_saving_fraction: None,
            saving_fraction_interval: None,
            findings,
        };
    }

    let nominal_saving =
        (baseline_band.nominal_wh - candidate_band.nominal_wh) / baseline_band.nominal_wh;
    let lower_saving = 1.0 - candidate_band.upper_wh / baseline_band.lower_wh;
    let upper_saving = 1.0 - candidate_band.lower_wh / baseline_band.upper_wh;

    if !nominal_saving.is_finite()
        || !lower_saving.is_finite()
        || !upper_saving.is_finite()
    {
        findings.push(Finding::DerivedOverflow {
            case_id: candidate.case_id.clone(),
            field: "relative_saving_fraction".to_owned(),
        });
        sort_findings(&mut findings);
        return ComparisonReport {
            disposition: ComparisonDisposition::Invalid,
            baseline_metrics: Some(baseline_metrics),
            candidate_metrics: Some(candidate_metrics),
            nominal_saving_fraction: None,
            saving_fraction_interval: None,
            findings,
        };
    }

    let saving_fraction_interval = FractionInterval {
        lower: lower_saving,
        upper: upper_saving,
    };
    let disposition = if lower_saving > 0.0 {
        ComparisonDisposition::RobustSavingWithinDeclaredBounds
    } else if upper_saving < 0.0 {
        ComparisonDisposition::RobustWorseWithinDeclaredBounds
    } else {
        ComparisonDisposition::Indeterminate
    };

    ComparisonReport {
        disposition,
        baseline_metrics: Some(baseline_metrics),
        candidate_metrics: Some(candidate_metrics),
        nominal_saving_fraction: Some(nominal_saving),
        saving_fraction_interval: Some(saving_fraction_interval),
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        case_id: &str,
        propulsion_wh: f64,
        propulsion_uncertainty_wh: f64,
        auxiliary_wh: Option<f64>,
        auxiliary_uncertainty_wh: Option<f64>,
    ) -> EnergyCase {
        EnergyCase {
            case_id: case_id.to_owned(),
            mission: MissionSpec {
                mission_id: "synthetic-auv-survey-v1".to_owned(),
                conditions_digest: "conditions-sha256:synthetic-001".to_owned(),
                route_distance_m: 10_000.0,
                payload_kg: Some(2_000.0),
                target_speed_mps: 1.0,
                arrival_policy_id: "target-speed-fixed-v1".to_owned(),
            },
            energy_boundary_id: "battery-electrical-input-v1".to_owned(),
            configuration_digest: Some(format!("config:{case_id}")),
            evidence_ref: Some(format!("synthetic-evidence:{case_id}")),
            evidence_origin: Some(EvidenceOrigin::Synthetic),
            uncertainty_method_id: Some("conservative-absolute-sum-v1".to_owned()),
            propulsion_input_wh: propulsion_wh,
            propulsion_uncertainty_wh,
            auxiliary_input_wh: auxiliary_wh,
            auxiliary_uncertainty_wh,
        }
    }

    fn baseline() -> EnergyCase {
        case("baseline", 1_000.0, 10.0, Some(100.0), Some(5.0))
    }

    fn improved_candidate() -> EnergyCase {
        case("candidate", 900.0, 10.0, Some(100.0), Some(5.0))
    }

    #[test]
    fn synthetic_known_answer_metrics_are_correct() {
        let result = assess_case(&baseline());
        assert_eq!(result.state, AssessmentState::Ready);
        let metrics = result.metrics.expect("complete case must have metrics");
        assert_eq!(metrics.total_input_wh, 1_100.0);
        assert_eq!(metrics.total_uncertainty_wh, 15.0);
        assert_eq!(metrics.total_energy_band.lower_wh, 1_085.0);
        assert_eq!(metrics.total_energy_band.upper_wh, 1_115.0);
        assert_eq!(metrics.wh_per_km, 110.0);
        assert_eq!(metrics.wh_per_tonne_km, Some(55.0));
    }

    #[test]
    fn robust_saving_requires_positive_conservative_lower_bound() {
        let result = compare(&baseline(), &improved_candidate());
        assert_eq!(
            result.disposition,
            ComparisonDisposition::RobustSavingWithinDeclaredBounds
        );
        let nominal = result.nominal_saving_fraction.expect("ready comparison");
        assert!((nominal - (100.0 / 1_100.0)).abs() < 1e-12);
        let band = result.saving_fraction_interval.expect("ready comparison");
        assert!(band.lower > 0.0);
        assert!(band.upper > band.lower);
        assert!((band.lower - (1.0 - 1_015.0 / 1_085.0)).abs() < 1e-12);
        assert!((band.upper - (1.0 - 985.0 / 1_115.0)).abs() < 1e-12);
    }

    #[test]
    fn auxiliary_energy_can_erase_a_propulsion_only_saving() {
        let candidate = case("candidate", 900.0, 10.0, Some(250.0), Some(5.0));
        let result = compare(&baseline(), &candidate);
        assert_eq!(
            result.disposition,
            ComparisonDisposition::RobustWorseWithinDeclaredBounds
        );
        assert_eq!(
            result.candidate_metrics.expect("complete candidate").total_input_wh,
            1_150.0
        );
    }

    #[test]
    fn overlapping_uncertainty_bands_are_indeterminate() {
        let baseline = case("baseline-wide", 1_000.0, 100.0, Some(100.0), Some(20.0));
        let candidate = case("candidate-wide", 900.0, 110.0, Some(100.0), Some(20.0));
        let result = compare(&baseline, &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
        let band = result.saving_fraction_interval.expect("complete arithmetic");
        assert!(band.lower <= 0.0);
        assert!(band.upper >= 0.0);
    }

    #[test]
    fn missing_auxiliary_energy_is_unknown_not_zero() {
        let candidate = case("candidate", 900.0, 10.0, None, Some(0.0));
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
        assert!(result.findings.contains(&Finding::MissingAuxiliaryEnergy {
            case_id: "candidate".to_owned()
        }));
        assert!(result.candidate_metrics.is_none());
    }

    #[test]
    fn invalid_auxiliary_uncertainty_is_detected_even_if_energy_is_missing() {
        let candidate = case("candidate", 900.0, 10.0, None, Some(f64::NAN));
        let result = assess_case(&candidate);
        assert_eq!(result.state, AssessmentState::Invalid);
        assert!(result.findings.iter().any(|finding| matches!(
            finding,
            Finding::InvalidNonFinite { case_id, field }
                if case_id == "candidate" && field == "auxiliary_uncertainty_wh"
        )));
    }

    #[test]
    fn missing_auxiliary_uncertainty_is_indeterminate() {
        let candidate = case("candidate", 900.0, 10.0, Some(100.0), None);
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
        assert!(result.findings.contains(&Finding::MissingAuxiliaryUncertainty {
            case_id: "candidate".to_owned()
        }));
    }

    #[test]
    fn condition_mismatch_is_incomparable() {
        let mut candidate = improved_candidate();
        candidate.mission.conditions_digest = "conditions-sha256:different".to_owned();
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.contains(&Finding::ConditionsMismatch));
        assert!(result.nominal_saving_fraction.is_none());
    }

    #[test]
    fn service_mismatch_is_incomparable() {
        let mut candidate = improved_candidate();
        candidate.mission.payload_kg = Some(1_500.0);
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.contains(&Finding::DeliveredServiceMismatch));
    }

    #[test]
    fn evidence_origin_mismatch_is_incomparable() {
        let mut candidate = improved_candidate();
        candidate.evidence_origin = Some(EvidenceOrigin::Cfd);
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.contains(&Finding::EvidenceOriginMismatch));
    }

    #[test]
    fn boundary_mismatch_is_incomparable() {
        let mut candidate = improved_candidate();
        candidate.energy_boundary_id = "shaft-output-v1".to_owned();
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.contains(&Finding::EnergyBoundaryMismatch));
    }

    #[test]
    fn missing_evidence_reference_prevents_positive_claim() {
        let mut candidate = improved_candidate();
        candidate.evidence_ref = None;
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
        assert!(result.findings.contains(&Finding::MissingEvidenceReference {
            case_id: "candidate".to_owned()
        }));
    }

    #[test]
    fn invalid_numeric_inputs_are_rejected() {
        let mut candidate = improved_candidate();
        candidate.propulsion_input_wh = f64::NAN;
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Invalid);
        assert!(result.findings.iter().any(|finding| matches!(
            finding,
            Finding::InvalidNonFinite { case_id, field }
                if case_id == "candidate" && field == "propulsion_input_wh"
        )));
    }

    #[test]
    fn negative_auxiliary_energy_is_invalid() {
        let candidate = case("candidate", 900.0, 10.0, Some(-1.0), Some(0.0));
        assert_eq!(assess_case(&candidate).state, AssessmentState::Invalid);
    }

    #[test]
    fn baseline_lower_energy_bound_must_be_positive() {
        let baseline = case("baseline-uncertain", 20.0, 50.0, Some(10.0), Some(50.0));
        let result = compare(&baseline, &improved_candidate());
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
        assert!(result.findings.contains(&Finding::BaselineLowerBoundNotPositive));
    }

    #[test]
    fn zero_distance_is_invalid() {
        let mut candidate = improved_candidate();
        candidate.mission.route_distance_m = 0.0;
        assert_eq!(assess_case(&candidate).state, AssessmentState::Invalid);
    }

    #[test]
    fn payloadless_missions_do_not_fabricate_tonne_km() {
        let mut case = baseline();
        case.mission.payload_kg = None;
        let result = assess_case(&case);
        assert_eq!(result.state, AssessmentState::Ready);
        assert_eq!(result.metrics.expect("complete case").wh_per_tonne_km, None);
    }

    #[test]
    fn uncertainty_method_mismatch_is_incomparable() {
        let mut candidate = improved_candidate();
        candidate.uncertainty_method_id = Some("different-bound-method".to_owned());
        let result = compare(&baseline(), &candidate);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.contains(&Finding::UncertaintyMethodMismatch));
    }
}
