// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Research contract only. This crate does not forecast individual mobility,
// determine service rights, certify destinations, or authorize relocation.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Epistemic class of an input or assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    Synthetic,
    Observed,
    Estimated,
    Modelled,
    Calibrated,
    Unknown,
}

/// How far the assessment has been qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentQualification {
    ContractOnly,
    ReferenceMatched,
    LocallyRetrospective,
    ProspectiveShadow,
    Unknown,
}

/// Mobility is represented as a scenario pathway, not a command or fate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathwayKind {
    InPlaceAdaptation,
    TemporaryDisplacement,
    VoluntaryMigration,
    CommunityLedPlannedRelocation,
    PostDisasterRecovery,
    ConstrainedImmobility,
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimateAttribution {
    NotClaimed,
    ObservedWithMethod,
    ModelledWithMethod,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionState {
    ScenarioOnly,
    CommunityDefined,
    HumanAuthorised,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaRole {
    Origin,
    Destination,
    OriginAndDestination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceDomain {
    DrinkingWater,
    Wastewater,
    Housing,
    Energy,
    HealthCare,
    Transport,
    Livelihoods,
    Ecology,
    Governance,
}

/// Area identity is an opaque, version-bound reference supplied by the spatial
/// owner. This crate does not interpret coordinates or infer boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioArea {
    pub id: String,
    pub role: AreaRole,
    pub boundary_revision: Option<String>,
}

/// A proposed or observed mobility pathway. A pathway remains an alternative
/// for comparison; it is never itself a recommendation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobilityPathway {
    pub id: String,
    pub kind: PathwayKind,
    pub origin_area_id: String,
    pub destination_area_id: Option<String>,
    pub moved_people: Option<u64>,
    pub climate_attribution: ClimateAttribution,
    pub attribution_method_id: Option<String>,
    pub decision_state: DecisionState,
    /// Must remain false. Human/community processes own decisions.
    pub is_recommendation: bool,
}

/// Arithmetic-only destination capacity input.
///
/// All quantities use the declared unit. Capacity adjustment may be negative.
/// Missing values, unknown evidence, or unknown qualification result in an
/// indeterminate outcome rather than a positive capacity claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityAssessment {
    pub id: String,
    pub pathway_id: String,
    pub area_id: String,
    pub domain: ServiceDomain,
    pub metric: String,
    pub unit: String,
    pub baseline_demand: Option<u64>,
    pub incremental_demand: Option<u64>,
    pub baseline_capacity: Option<u64>,
    pub capacity_adjustment: Option<i64>,
    pub evidence_class: EvidenceClass,
    pub qualification: AssessmentQualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityOutcome {
    Sufficient,
    Shortfall,
    Indeterminate,
}

/// The output deliberately says capacity sufficiency, not safe, suitable,
/// rights-compliant, or service-floor met.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityAssessmentResult {
    pub assessment_id: String,
    pub outcome: CapacityOutcome,
    pub total_demand: Option<u64>,
    pub available_capacity: Option<u64>,
    pub shortfall: Option<u64>,
    pub evidence_class: EvidenceClass,
    pub qualification: AssessmentQualification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyField(&'static str),
    DemandOverflow { assessment_id: String },
    CapacityOverflow { assessment_id: String },
    NegativeAvailableCapacity { assessment_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyField(&'static str),
    DuplicateAreaId(String),
    DuplicatePathwayId(String),
    DuplicateAssessmentId(String),
    DuplicateFloorAssessmentId(String),
    UnknownOriginArea { pathway_id: String, area_id: String },
    UnknownDestinationArea { pathway_id: String, area_id: String },
    OriginRoleMismatch { pathway_id: String, area_id: String },
    DestinationRoleMismatch { pathway_id: String, area_id: String },
    UnknownAssessmentPathway { assessment_id: String, pathway_id: String },
    UnknownAssessmentArea { assessment_id: String, area_id: String },
    AssessmentAreaNotInPathway { assessment_id: String, pathway_id: String, area_id: String },
    UnknownFloorAssessmentPathway { assessment_id: String, pathway_id: String },
    UnknownFloorAssessmentArea { assessment_id: String, area_id: String },
    FloorAssessmentAreaNotInPathway { assessment_id: String, pathway_id: String, area_id: String },
    ServiceFloor(ServiceFloorError),
    RecommendationAuthorityForbidden { pathway_id: String },
    AttributionMethodRequired { pathway_id: String },
    Evaluation(EvaluationError),
}

impl From<EvaluationError> for ContractError {
    fn from(value: EvaluationError) -> Self {
        Self::Evaluation(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MobilityScenarioSet {
    pub areas: Vec<ScenarioArea>,
    pub pathways: Vec<MobilityPathway>,
    pub assessments: Vec<CapacityAssessment>,
    pub floor_assessments: Vec<ServiceFloorAssessment>,
}

/// Evaluate a single declared capacity envelope.
///
/// This is arithmetic, not a hydraulic/network solver. Sufficient means only
/// that supplied aggregate capacity is at least supplied aggregate demand for
/// this assessment's unit and scope.
pub fn evaluate_capacity(
    input: &CapacityAssessment,
) -> Result<CapacityAssessmentResult, EvaluationError> {
    if input.id.trim().is_empty() {
        return Err(EvaluationError::EmptyField("assessment id"));
    }
    if input.unit.trim().is_empty() {
        return Err(EvaluationError::EmptyField("unit"));
    }
    if input.metric.trim().is_empty() {
        return Err(EvaluationError::EmptyField("metric"));
    }

    let indeterminate = || CapacityAssessmentResult {
        assessment_id: input.id.clone(),
        outcome: CapacityOutcome::Indeterminate,
        total_demand: None,
        available_capacity: None,
        shortfall: None,
        evidence_class: input.evidence_class,
        qualification: input.qualification,
    };

    if input.evidence_class == EvidenceClass::Unknown
        || input.qualification == AssessmentQualification::Unknown
    {
        return Ok(indeterminate());
    }

    let (
        Some(baseline_demand),
        Some(incremental_demand),
        Some(baseline_capacity),
        Some(capacity_adjustment),
    ) = (
        input.baseline_demand,
        input.incremental_demand,
        input.baseline_capacity,
        input.capacity_adjustment,
    )
    else {
        return Ok(indeterminate());
    };

    let total_demand = baseline_demand
        .checked_add(incremental_demand)
        .ok_or_else(|| EvaluationError::DemandOverflow {
            assessment_id: input.id.clone(),
        })?;

    let capacity = i128::from(baseline_capacity) + i128::from(capacity_adjustment);
    if capacity < 0 {
        return Err(EvaluationError::NegativeAvailableCapacity {
            assessment_id: input.id.clone(),
        });
    }
    if capacity > i128::from(u64::MAX) {
        return Err(EvaluationError::CapacityOverflow {
            assessment_id: input.id.clone(),
        });
    }
    let available_capacity = capacity as u64;
    let shortfall = total_demand.saturating_sub(available_capacity);
    let outcome = if shortfall == 0 {
        CapacityOutcome::Sufficient
    } else {
        CapacityOutcome::Shortfall
    };

    Ok(CapacityAssessmentResult {
        assessment_id: input.id.clone(),
        outcome,
        total_demand: Some(total_demand),
        available_capacity: Some(available_capacity),
        shortfall: Some(shortfall),
        evidence_class: input.evidence_class,
        qualification: input.qualification,
    })
}

impl MobilityScenarioSet {
    /// Validate identity and cross-reference integrity without promoting
    /// synthetic or incomplete evidence. This method does not assign
    /// probabilities or select a preferred pathway.
    pub fn validate(&self) -> Result<(), ContractError> {
        let mut areas = BTreeMap::new();
        for area in &self.areas {
            if area.id.trim().is_empty() {
                return Err(ContractError::EmptyField("area id"));
            }
            if areas.insert(area.id.as_str(), area.role).is_some() {
                return Err(ContractError::DuplicateAreaId(area.id.clone()));
            }
        }

        let mut pathways = BTreeSet::new();
        for pathway in &self.pathways {
            if pathway.id.trim().is_empty() || pathway.origin_area_id.trim().is_empty() {
                return Err(ContractError::EmptyField("pathway id or origin area id"));
            }
            if !pathways.insert(pathway.id.as_str()) {
                return Err(ContractError::DuplicatePathwayId(pathway.id.clone()));
            }
            if pathway.is_recommendation {
                return Err(ContractError::RecommendationAuthorityForbidden {
                    pathway_id: pathway.id.clone(),
                });
            }
            if matches!(
                pathway.climate_attribution,
                ClimateAttribution::ObservedWithMethod | ClimateAttribution::ModelledWithMethod
            ) && pathway
                .attribution_method_id
                .as_deref()
                .is_none_or(str::is_empty)
            {
                return Err(ContractError::AttributionMethodRequired {
                    pathway_id: pathway.id.clone(),
                });
            }

            let origin_role = areas.get(pathway.origin_area_id.as_str()).ok_or_else(|| {
                ContractError::UnknownOriginArea {
                    pathway_id: pathway.id.clone(),
                    area_id: pathway.origin_area_id.clone(),
                }
            })?;
            if !matches!(origin_role, AreaRole::Origin | AreaRole::OriginAndDestination) {
                return Err(ContractError::OriginRoleMismatch {
                    pathway_id: pathway.id.clone(),
                    area_id: pathway.origin_area_id.clone(),
                });
            }

            if let Some(destination_id) = pathway.destination_area_id.as_deref() {
                let destination_role = areas.get(destination_id).ok_or_else(|| {
                    ContractError::UnknownDestinationArea {
                        pathway_id: pathway.id.clone(),
                        area_id: destination_id.to_owned(),
                    }
                })?;
                if !matches!(
                    destination_role,
                    AreaRole::Destination | AreaRole::OriginAndDestination
                ) {
                    return Err(ContractError::DestinationRoleMismatch {
                        pathway_id: pathway.id.clone(),
                        area_id: destination_id.to_owned(),
                    });
                }
            }
        }

        let mut assessment_ids = BTreeSet::new();
        for assessment in &self.assessments {
            if assessment.id.trim().is_empty()
                || assessment.pathway_id.trim().is_empty()
                || assessment.area_id.trim().is_empty()
            {
                return Err(ContractError::EmptyField(
                    "assessment id, pathway id, or area id",
                ));
            }
            if !assessment_ids.insert(assessment.id.as_str()) {
                return Err(ContractError::DuplicateAssessmentId(assessment.id.clone()));
            }
            if !pathways.contains(assessment.pathway_id.as_str()) {
                return Err(ContractError::UnknownAssessmentPathway {
                    assessment_id: assessment.id.clone(),
                    pathway_id: assessment.pathway_id.clone(),
                });
            }
            if !areas.contains_key(assessment.area_id.as_str()) {
                return Err(ContractError::UnknownAssessmentArea {
                    assessment_id: assessment.id.clone(),
                    area_id: assessment.area_id.clone(),
                });
            }
            let pathway = self
                .pathways
                .iter()
                .find(|pathway| pathway.id == assessment.pathway_id)
                .expect("pathway reference was checked above");
            let area_is_in_pathway = assessment.area_id == pathway.origin_area_id
                || pathway.destination_area_id.as_deref() == Some(assessment.area_id.as_str());
            if !area_is_in_pathway {
                return Err(ContractError::AssessmentAreaNotInPathway {
                    assessment_id: assessment.id.clone(),
                    pathway_id: assessment.pathway_id.clone(),
                    area_id: assessment.area_id.clone(),
                });
            }
            // Arithmetic errors fail the contract; missing/unknown evidence
            // remains a valid, explicitly indeterminate assessment.
            let _ = evaluate_capacity(assessment)?;
        }

        let mut floor_assessment_ids = BTreeSet::new();
        for floor in &self.floor_assessments {
            if floor.id.trim().is_empty()
                || floor.pathway_id.trim().is_empty()
                || floor.area_id.trim().is_empty()
            {
                return Err(ContractError::EmptyField(
                    "floor assessment id, pathway id, or area id",
                ));
            }
            if !floor_assessment_ids.insert(floor.id.as_str()) {
                return Err(ContractError::DuplicateFloorAssessmentId(floor.id.clone()));
            }
            if !pathways.contains(floor.pathway_id.as_str()) {
                return Err(ContractError::UnknownFloorAssessmentPathway {
                    assessment_id: floor.id.clone(),
                    pathway_id: floor.pathway_id.clone(),
                });
            }
            if !areas.contains_key(floor.area_id.as_str()) {
                return Err(ContractError::UnknownFloorAssessmentArea {
                    assessment_id: floor.id.clone(),
                    area_id: floor.area_id.clone(),
                });
            }
            let pathway = self
                .pathways
                .iter()
                .find(|pathway| pathway.id == floor.pathway_id)
                .expect("floor pathway reference was checked above");
            let area_is_in_pathway = floor.area_id == pathway.origin_area_id
                || pathway.destination_area_id.as_deref() == Some(floor.area_id.as_str());
            if !area_is_in_pathway {
                return Err(ContractError::FloorAssessmentAreaNotInPathway {
                    assessment_id: floor.id.clone(),
                    pathway_id: floor.pathway_id.clone(),
                    area_id: floor.area_id.clone(),
                });
            }
            let _ = evaluate_service_floor(floor).map_err(ContractError::ServiceFloor)?;
        }
        Ok(())
    }
}


/// A locally authorized, version-bound minimum service requirement.
///
/// floor_authority_ref identifies the authority/profile that supplied the
/// threshold; this crate does not decide whether that authority is legally
/// valid. observation_ref identifies the admitted evidence used for the
/// interval. Missing references or incomplete coverage prohibit a positive
/// "meets floor" outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceFloorAssessment {
    pub id: String,
    pub pathway_id: String,
    pub area_id: String,
    pub service_scope_id: String,
    pub metric: String,
    pub unit: String,
    pub floor_profile_revision: Option<String>,
    pub floor_authority_ref: Option<String>,
    pub observation_ref: Option<String>,
    /// Inclusive observation valid-time window, Unix seconds.
    pub observation_valid_from_unix_seconds: Option<i64>,
    pub observation_valid_to_unix_seconds: Option<i64>,
    /// Evaluation clock and caller/authority-declared freshness budget.
    pub assessed_at_unix_seconds: Option<i64>,
    pub max_observation_age_seconds: Option<u64>,
    pub minimum_required: Option<u64>,
    pub observed_lower_bound: Option<u64>,
    pub observed_upper_bound: Option<u64>,
    /// Count of covered service units (e.g. metered connections or households).
    /// This is an aggregate count, not an individual identity or location.
    pub coverage_numerator: Option<u64>,
    pub coverage_denominator: Option<u64>,
    pub evidence_class: EvidenceClass,
    pub qualification: AssessmentQualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceFloorOutcome {
    MeetsFloor,
    BreachedFloor,
    /// A breach is observed in the covered subset, but full-scope coverage is
    /// incomplete; do not generalize it to the entire service scope.
    ObservedBreach,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceFloorResult {
    pub assessment_id: String,
    pub pathway_id: String,
    pub area_id: String,
    pub service_scope_id: String,
    pub observation_valid_from_unix_seconds: Option<i64>,
    pub observation_valid_to_unix_seconds: Option<i64>,
    pub assessed_at_unix_seconds: Option<i64>,
    pub max_observation_age_seconds: Option<u64>,
    pub outcome: ServiceFloorOutcome,
    pub minimum_required: Option<u64>,
    pub observed_lower_bound: Option<u64>,
    pub observed_upper_bound: Option<u64>,
    pub coverage_numerator: Option<u64>,
    pub coverage_denominator: Option<u64>,
    pub evidence_class: EvidenceClass,
    pub qualification: AssessmentQualification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceFloorError {
    EmptyField(&'static str),
    InvalidCoverage { assessment_id: String },
    InvertedObservationInterval { assessment_id: String },
    InvertedObservationWindow { assessment_id: String },
    ObservationAfterAssessment { assessment_id: String },
    InvalidFreshnessBudget { assessment_id: String },
}

/// Evaluate a non-compensatory service floor using conservative observation
/// bounds. Full coverage is required to conclude that a floor is met or that
/// the complete declared scope is below the floor. A breach observed in a
/// partial subset remains visible as ObservedBreach; positive claims never
/// generalize from partial coverage.
pub fn evaluate_service_floor(
    input: &ServiceFloorAssessment,
) -> Result<ServiceFloorResult, ServiceFloorError> {
    if input.id.trim().is_empty() {
        return Err(ServiceFloorError::EmptyField("assessment id"));
    }
    if input.service_scope_id.trim().is_empty() {
        return Err(ServiceFloorError::EmptyField("service scope id"));
    }
    if input.metric.trim().is_empty() {
        return Err(ServiceFloorError::EmptyField("metric"));
    }
    if input.unit.trim().is_empty() {
        return Err(ServiceFloorError::EmptyField("unit"));
    }

    let indeterminate = || ServiceFloorResult {
        assessment_id: input.id.clone(),
        pathway_id: input.pathway_id.clone(),
        area_id: input.area_id.clone(),
        service_scope_id: input.service_scope_id.clone(),
        observation_valid_from_unix_seconds: input.observation_valid_from_unix_seconds,
        observation_valid_to_unix_seconds: input.observation_valid_to_unix_seconds,
        assessed_at_unix_seconds: input.assessed_at_unix_seconds,
        max_observation_age_seconds: input.max_observation_age_seconds,
        outcome: ServiceFloorOutcome::Indeterminate,
        minimum_required: input.minimum_required,
        observed_lower_bound: input.observed_lower_bound,
        observed_upper_bound: input.observed_upper_bound,
        coverage_numerator: input.coverage_numerator,
        coverage_denominator: input.coverage_denominator,
        evidence_class: input.evidence_class,
        qualification: input.qualification,
    };

    if matches!(input.evidence_class, EvidenceClass::Unknown | EvidenceClass::Synthetic)
        || matches!(
            input.qualification,
            AssessmentQualification::Unknown | AssessmentQualification::ContractOnly
        )
        || input.floor_profile_revision.as_deref().is_none_or(|value| value.trim().is_empty())
        || input.floor_authority_ref.as_deref().is_none_or(|value| value.trim().is_empty())
        || input.observation_ref.as_deref().is_none_or(|value| value.trim().is_empty())
    {
        return Ok(indeterminate());
    }

    let (
        Some(observation_valid_from),
        Some(observation_valid_to),
        Some(assessed_at),
        Some(max_observation_age),
    ) = (
        input.observation_valid_from_unix_seconds,
        input.observation_valid_to_unix_seconds,
        input.assessed_at_unix_seconds,
        input.max_observation_age_seconds,
    ) else {
        return Ok(indeterminate());
    };

    if max_observation_age == 0 {
        return Err(ServiceFloorError::InvalidFreshnessBudget {
            assessment_id: input.id.clone(),
        });
    }
    if observation_valid_from > observation_valid_to {
        return Err(ServiceFloorError::InvertedObservationWindow {
            assessment_id: input.id.clone(),
        });
    }
    if observation_valid_to > assessed_at {
        return Err(ServiceFloorError::ObservationAfterAssessment {
            assessment_id: input.id.clone(),
        });
    }
    let observation_age_seconds =
        i128::from(assessed_at) - i128::from(observation_valid_to);
    if observation_age_seconds > i128::from(max_observation_age) {
        return Ok(indeterminate());
    }

    let (
        Some(minimum_required),
        Some(observed_lower_bound),
        Some(observed_upper_bound),
        Some(coverage_numerator),
        Some(coverage_denominator),
    ) = (
        input.minimum_required,
        input.observed_lower_bound,
        input.observed_upper_bound,
        input.coverage_numerator,
        input.coverage_denominator,
    )
    else {
        return Ok(indeterminate());
    };

    if coverage_denominator == 0
        || coverage_numerator == 0
        || coverage_numerator > coverage_denominator
    {
        return Err(ServiceFloorError::InvalidCoverage {
            assessment_id: input.id.clone(),
        });
    }
    if observed_lower_bound > observed_upper_bound {
        return Err(ServiceFloorError::InvertedObservationInterval {
            assessment_id: input.id.clone(),
        });
    }

    let full_coverage = coverage_numerator == coverage_denominator;
    let outcome = if full_coverage && observed_lower_bound >= minimum_required {
        ServiceFloorOutcome::MeetsFloor
    } else if full_coverage && observed_upper_bound < minimum_required {
        ServiceFloorOutcome::BreachedFloor
    } else if observed_upper_bound < minimum_required {
        ServiceFloorOutcome::ObservedBreach
    } else {
        ServiceFloorOutcome::Indeterminate
    };

    Ok(ServiceFloorResult {
        assessment_id: input.id.clone(),
        pathway_id: input.pathway_id.clone(),
        area_id: input.area_id.clone(),
        service_scope_id: input.service_scope_id.clone(),
        observation_valid_from_unix_seconds: Some(observation_valid_from),
        observation_valid_to_unix_seconds: Some(observation_valid_to),
        assessed_at_unix_seconds: Some(assessed_at),
        max_observation_age_seconds: Some(max_observation_age),
        outcome,
        minimum_required: Some(minimum_required),
        observed_lower_bound: Some(observed_lower_bound),
        observed_upper_bound: Some(observed_upper_bound),
        coverage_numerator: Some(coverage_numerator),
        coverage_denominator: Some(coverage_denominator),
        evidence_class: input.evidence_class,
        qualification: input.qualification,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assessment(
        id: &str,
        baseline_demand: Option<u64>,
        incremental_demand: Option<u64>,
        baseline_capacity: Option<u64>,
        capacity_adjustment: Option<i64>,
    ) -> CapacityAssessment {
        CapacityAssessment {
            id: id.to_owned(),
            pathway_id: "pathway-a".to_owned(),
            area_id: "area-a".to_owned(),
            domain: ServiceDomain::DrinkingWater,
            metric: "reliable_potable_delivery".to_owned(),
            unit: "L/day".to_owned(),
            baseline_demand,
            incremental_demand,
            baseline_capacity,
            capacity_adjustment,
            evidence_class: EvidenceClass::Synthetic,
            qualification: AssessmentQualification::ContractOnly,
        }
    }

    #[test]
    fn adaptation_fixture_has_capacity_surplus() {
        let input = assessment(
            "stay-water",
            Some(500_000),
            Some(0),
            Some(480_000),
            Some(40_000),
        );
        let result = evaluate_capacity(&input).expect("valid synthetic arithmetic");
        assert_eq!(result.outcome, CapacityOutcome::Sufficient);
        assert_eq!(result.total_demand, Some(500_000));
        assert_eq!(result.available_capacity, Some(520_000));
        assert_eq!(result.shortfall, Some(0));
        assert_eq!(result.evidence_class, EvidenceClass::Synthetic);
    }

    #[test]
    fn arrival_fixture_reports_water_capacity_shortfall() {
        let input = assessment(
            "arrival-water",
            Some(500_000),
            Some(50_000),
            Some(520_000),
            Some(0),
        );
        let result = evaluate_capacity(&input).expect("valid synthetic arithmetic");
        assert_eq!(result.outcome, CapacityOutcome::Shortfall);
        assert_eq!(result.total_demand, Some(550_000));
        assert_eq!(result.available_capacity, Some(520_000));
        assert_eq!(result.shortfall, Some(30_000));
    }

    #[test]
    fn exact_capacity_is_sufficient_not_a_shortfall() {
        let input = assessment("housing", Some(4_500), Some(0), Some(4_500), Some(0));
        let result = evaluate_capacity(&input).expect("valid synthetic arithmetic");
        assert_eq!(result.outcome, CapacityOutcome::Sufficient);
        assert_eq!(result.shortfall, Some(0));
    }

    #[test]
    fn unknown_evidence_never_becomes_a_positive_capacity_claim() {
        let mut input = assessment("unknown", Some(10), Some(0), Some(100), Some(0));
        input.evidence_class = EvidenceClass::Unknown;
        let result = evaluate_capacity(&input).expect("unknown evidence is indeterminate");
        assert_eq!(result.outcome, CapacityOutcome::Indeterminate);
        assert_eq!(result.available_capacity, None);
    }

    #[test]
    fn missing_capacity_input_is_indeterminate() {
        let input = assessment("missing", Some(10), Some(0), None, Some(0));
        let result = evaluate_capacity(&input).expect("missing input is indeterminate");
        assert_eq!(result.outcome, CapacityOutcome::Indeterminate);
        assert_eq!(result.total_demand, None);
    }

    #[test]
    fn negative_available_capacity_is_rejected() {
        let input = assessment("negative", Some(1), Some(0), Some(10), Some(-11));
        assert_eq!(
            evaluate_capacity(&input),
            Err(EvaluationError::NegativeAvailableCapacity {
                assessment_id: "negative".to_owned()
            })
        );
    }

    #[test]
    fn demand_overflow_is_rejected() {
        let input = assessment("overflow", Some(u64::MAX), Some(1), Some(10), Some(0));
        assert_eq!(
            evaluate_capacity(&input),
            Err(EvaluationError::DemandOverflow {
                assessment_id: "overflow".to_owned()
            })
        );
    }

    #[test]
    fn methodless_climate_attribution_is_rejected() {
        let set = MobilityScenarioSet {
            areas: vec![ScenarioArea {
                id: "origin".to_owned(),
                role: AreaRole::Origin,
                boundary_revision: None,
            }],
            pathways: vec![MobilityPathway {
                id: "path-a".to_owned(),
                kind: PathwayKind::PostDisasterRecovery,
                origin_area_id: "origin".to_owned(),
                destination_area_id: None,
                moved_people: Some(5),
                climate_attribution: ClimateAttribution::ObservedWithMethod,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: false,
            }],
            assessments: Vec::new(),
            floor_assessments: Vec::new(),
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::AttributionMethodRequired {
                pathway_id: "path-a".to_owned()
            })
        );
    }

    #[test]
    fn recommendation_authority_is_rejected() {
        let set = MobilityScenarioSet {
            areas: vec![ScenarioArea {
                id: "origin".to_owned(),
                role: AreaRole::Origin,
                boundary_revision: None,
            }],
            pathways: vec![MobilityPathway {
                id: "path-a".to_owned(),
                kind: PathwayKind::InPlaceAdaptation,
                origin_area_id: "origin".to_owned(),
                destination_area_id: None,
                moved_people: Some(0),
                climate_attribution: ClimateAttribution::NotClaimed,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: true,
            }],
            assessments: Vec::new(),
            floor_assessments: Vec::new(),
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::RecommendationAuthorityForbidden {
                pathway_id: "path-a".to_owned()
            })
        );
    }

    #[test]
    fn unknown_assessment_area_is_rejected() {
        let set = MobilityScenarioSet {
            areas: vec![ScenarioArea {
                id: "origin".to_owned(),
                role: AreaRole::Origin,
                boundary_revision: None,
            }],
            pathways: vec![MobilityPathway {
                id: "path-a".to_owned(),
                kind: PathwayKind::InPlaceAdaptation,
                origin_area_id: "origin".to_owned(),
                destination_area_id: None,
                moved_people: Some(0),
                climate_attribution: ClimateAttribution::NotClaimed,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: false,
            }],
            assessments: vec![assessment("a", Some(1), Some(0), Some(2), Some(0))],
            floor_assessments: Vec::new(),
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::UnknownAssessmentArea {
                assessment_id: "a".to_owned(),
                area_id: "area-a".to_owned()
            })
        );
    }
    #[test]
    fn assessment_area_must_belong_to_its_pathway() {
        let set = MobilityScenarioSet {
            areas: vec![
                ScenarioArea {
                    id: "origin".to_owned(),
                    role: AreaRole::Origin,
                    boundary_revision: None,
                },
                ScenarioArea {
                    id: "destination".to_owned(),
                    role: AreaRole::Destination,
                    boundary_revision: None,
                },
                ScenarioArea {
                    id: "unrelated".to_owned(),
                    role: AreaRole::Destination,
                    boundary_revision: None,
                },
            ],
            pathways: vec![MobilityPathway {
                id: "path-a".to_owned(),
                kind: PathwayKind::VoluntaryMigration,
                origin_area_id: "origin".to_owned(),
                destination_area_id: Some("destination".to_owned()),
                moved_people: Some(10),
                climate_attribution: ClimateAttribution::NotClaimed,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: false,
            }],
            assessments: vec![CapacityAssessment {
                id: "unrelated-assessment".to_owned(),
                pathway_id: "path-a".to_owned(),
                area_id: "unrelated".to_owned(),
                domain: ServiceDomain::DrinkingWater,
                metric: "reliable_potable_delivery".to_owned(),
                unit: "L/day".to_owned(),
                baseline_demand: Some(10),
                incremental_demand: Some(0),
                baseline_capacity: Some(10),
                capacity_adjustment: Some(0),
                evidence_class: EvidenceClass::Synthetic,
                qualification: AssessmentQualification::ContractOnly,
            }],
            floor_assessments: Vec::new(),
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::AssessmentAreaNotInPathway {
                assessment_id: "unrelated-assessment".to_owned(),
                pathway_id: "path-a".to_owned(),
                area_id: "unrelated".to_owned()
            })
        );
    }

    fn service_floor(
        coverage_numerator: Option<u64>,
        coverage_denominator: Option<u64>,
        lower: Option<u64>,
        upper: Option<u64>,
    ) -> ServiceFloorAssessment {
        ServiceFloorAssessment {
            id: "floor-water-area-a".to_owned(),
            pathway_id: "pathway-a".to_owned(),
            area_id: "area-a".to_owned(),
            service_scope_id: "community-cell-a".to_owned(),
            metric: "delivered_potable_water_per_person_day".to_owned(),
            unit: "L/person/day".to_owned(),
            floor_profile_revision: Some("local-profile-v1".to_owned()),
            floor_authority_ref: Some("authority-profile-ref".to_owned()),
            observation_ref: Some("observation-bundle-ref".to_owned()),
            observation_valid_from_unix_seconds: Some(1_000),
            observation_valid_to_unix_seconds: Some(1_100),
            assessed_at_unix_seconds: Some(1_200),
            max_observation_age_seconds: Some(300),
            minimum_required: Some(25),
            observed_lower_bound: lower,
            observed_upper_bound: upper,
            coverage_numerator,
            coverage_denominator,
            evidence_class: EvidenceClass::Observed,
            qualification: AssessmentQualification::ReferenceMatched,
        }
    }

    #[test]
    fn complete_coverage_and_lower_bound_above_floor_meets_floor() {
        let input = service_floor(Some(100), Some(100), Some(27), Some(31));
        let result = evaluate_service_floor(&input).expect("valid floor interval");
        assert_eq!(result.outcome, ServiceFloorOutcome::MeetsFloor);
    }

    #[test]
    fn complete_coverage_and_upper_bound_below_floor_confirms_breach() {
        let input = service_floor(Some(100), Some(100), Some(12), Some(20));
        let result = evaluate_service_floor(&input).expect("valid floor interval");
        assert_eq!(result.outcome, ServiceFloorOutcome::BreachedFloor);
    }

    #[test]
    fn partial_coverage_preserves_observed_breach_without_generalizing_scope() {
        let input = service_floor(Some(10), Some(100), Some(12), Some(20));
        let result = evaluate_service_floor(&input).expect("valid partial evidence");
        assert_eq!(result.outcome, ServiceFloorOutcome::ObservedBreach);
    }

    #[test]
    fn partial_coverage_never_proves_that_the_floor_is_met() {
        let input = service_floor(Some(10), Some(100), Some(27), Some(31));
        let result = evaluate_service_floor(&input).expect("valid partial evidence");
        assert_eq!(result.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn an_interval_crossing_the_floor_is_indeterminate() {
        let input = service_floor(Some(100), Some(100), Some(20), Some(30));
        let result = evaluate_service_floor(&input).expect("valid uncertain interval");
        assert_eq!(result.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn missing_floor_authority_or_observation_cannot_pass() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.floor_authority_ref = None;
        let first = evaluate_service_floor(&input).expect("missing authority is indeterminate");
        assert_eq!(first.outcome, ServiceFloorOutcome::Indeterminate);

        input.floor_authority_ref = Some("   ".to_owned());
        let whitespace = evaluate_service_floor(&input).expect("blank authority is indeterminate");
        assert_eq!(whitespace.outcome, ServiceFloorOutcome::Indeterminate);

        input.floor_authority_ref = Some("authority-profile-ref".to_owned());
        input.observation_ref = None;
        let second = evaluate_service_floor(&input).expect("missing observation is indeterminate");
        assert_eq!(second.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn unknown_evidence_cannot_pass_a_service_floor() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.evidence_class = EvidenceClass::Unknown;
        let result = evaluate_service_floor(&input).expect("unknown evidence is indeterminate");
        assert_eq!(result.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn missing_threshold_or_coverage_is_indeterminate() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.minimum_required = None;
        let no_floor = evaluate_service_floor(&input).expect("missing threshold is indeterminate");
        assert_eq!(no_floor.outcome, ServiceFloorOutcome::Indeterminate);

        input.minimum_required = Some(25);
        input.coverage_numerator = None;
        let no_coverage = evaluate_service_floor(&input).expect("missing coverage is indeterminate");
        assert_eq!(no_coverage.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn invalid_coverage_is_rejected() {
        let input = service_floor(Some(101), Some(100), Some(30), Some(35));
        assert_eq!(
            evaluate_service_floor(&input),
            Err(ServiceFloorError::InvalidCoverage {
                assessment_id: "floor-water-area-a".to_owned()
            })
        );
    }

    #[test]
    fn inverted_observation_interval_is_rejected() {
        let input = service_floor(Some(100), Some(100), Some(35), Some(30));
        assert_eq!(
            evaluate_service_floor(&input),
            Err(ServiceFloorError::InvertedObservationInterval {
                assessment_id: "floor-water-area-a".to_owned()
            })
        );
    }
 
    #[test]
    fn scenario_validation_binds_floor_assessment_to_pathway_and_area() {
        let set = MobilityScenarioSet {
            areas: vec![ScenarioArea {
                id: "area-a".to_owned(),
                role: AreaRole::Origin,
                boundary_revision: Some("boundary-v1".to_owned()),
            }],
            pathways: vec![MobilityPathway {
                id: "pathway-a".to_owned(),
                kind: PathwayKind::InPlaceAdaptation,
                origin_area_id: "area-a".to_owned(),
                destination_area_id: None,
                moved_people: Some(0),
                climate_attribution: ClimateAttribution::NotClaimed,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: false,
            }],
            assessments: Vec::new(),
            floor_assessments: vec![service_floor(
                Some(10),
                Some(100),
                Some(12),
                Some(20),
            )],
        };
        assert_eq!(set.validate(), Ok(()));
    }

    #[test]
    fn floor_assessment_cannot_be_attached_to_an_unrelated_area() {
        let mut floor = service_floor(Some(100), Some(100), Some(30), Some(35));
        floor.pathway_id = "path-a".to_owned();
        floor.area_id = "unrelated".to_owned();
        let set = MobilityScenarioSet {
            areas: vec![
                ScenarioArea {
                    id: "origin".to_owned(),
                    role: AreaRole::Origin,
                    boundary_revision: None,
                },
                ScenarioArea {
                    id: "destination".to_owned(),
                    role: AreaRole::Destination,
                    boundary_revision: None,
                },
                ScenarioArea {
                    id: "unrelated".to_owned(),
                    role: AreaRole::Destination,
                    boundary_revision: None,
                },
            ],
            pathways: vec![MobilityPathway {
                id: "path-a".to_owned(),
                kind: PathwayKind::VoluntaryMigration,
                origin_area_id: "origin".to_owned(),
                destination_area_id: Some("destination".to_owned()),
                moved_people: Some(5),
                climate_attribution: ClimateAttribution::NotClaimed,
                attribution_method_id: None,
                decision_state: DecisionState::ScenarioOnly,
                is_recommendation: false,
            }],
            assessments: Vec::new(),
            floor_assessments: vec![floor],
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::FloorAssessmentAreaNotInPathway {
                assessment_id: "floor-water-area-a".to_owned(),
                pathway_id: "path-a".to_owned(),
                area_id: "unrelated".to_owned(),
            })
        );
    }
 
    #[test]
    fn synthetic_or_contract_only_floor_evidence_never_becomes_a_real_floor_outcome() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.evidence_class = EvidenceClass::Synthetic;
        input.qualification = AssessmentQualification::ContractOnly;
        let result = evaluate_service_floor(&input).expect("synthetic floor evidence abstains");
        assert_eq!(result.outcome, ServiceFloorOutcome::Indeterminate);

        input.evidence_class = EvidenceClass::Observed;
        input.qualification = AssessmentQualification::ContractOnly;
        let contract_only =
            evaluate_service_floor(&input).expect("contract-only floor evidence abstains");
        assert_eq!(contract_only.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn stale_floor_observation_is_indeterminate() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.observation_valid_to_unix_seconds = Some(100);
        input.assessed_at_unix_seconds = Some(1_000);
        input.max_observation_age_seconds = Some(50);
        let result = evaluate_service_floor(&input).expect("stale evidence abstains");
        assert_eq!(result.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn missing_observation_window_or_freshness_budget_is_indeterminate() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.observation_valid_from_unix_seconds = None;
        let missing_window =
            evaluate_service_floor(&input).expect("missing valid time abstains");
        assert_eq!(missing_window.outcome, ServiceFloorOutcome::Indeterminate);

        input.observation_valid_from_unix_seconds = Some(1_000);
        input.max_observation_age_seconds = None;
        let missing_budget =
            evaluate_service_floor(&input).expect("missing freshness budget abstains");
        assert_eq!(missing_budget.outcome, ServiceFloorOutcome::Indeterminate);
    }

    #[test]
    fn observation_window_after_assessment_time_is_rejected() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.observation_valid_to_unix_seconds = Some(1_300);
        assert_eq!(
            evaluate_service_floor(&input),
            Err(ServiceFloorError::ObservationAfterAssessment {
                assessment_id: "floor-water-area-a".to_owned()
            })
        );
    }

    #[test]
    fn inverted_observation_valid_time_window_is_rejected() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.observation_valid_from_unix_seconds = Some(1_200);
        input.observation_valid_to_unix_seconds = Some(1_100);
        assert_eq!(
            evaluate_service_floor(&input),
            Err(ServiceFloorError::InvertedObservationWindow {
                assessment_id: "floor-water-area-a".to_owned()
            })
        );
    }

    #[test]
    fn zero_freshness_budget_is_rejected() {
        let mut input = service_floor(Some(100), Some(100), Some(30), Some(35));
        input.max_observation_age_seconds = Some(0);
        assert_eq!(
            evaluate_service_floor(&input),
            Err(ServiceFloorError::InvalidFreshnessBudget {
                assessment_id: "floor-water-area-a".to_owned()
            })
        );
    }

}
