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
    UnknownOriginArea { pathway_id: String, area_id: String },
    UnknownDestinationArea { pathway_id: String, area_id: String },
    OriginRoleMismatch { pathway_id: String, area_id: String },
    DestinationRoleMismatch { pathway_id: String, area_id: String },
    UnknownAssessmentPathway { assessment_id: String, pathway_id: String },
    UnknownAssessmentArea { assessment_id: String, area_id: String },
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
            // Arithmetic errors fail the contract; missing/unknown evidence
            // remains a valid, explicitly indeterminate assessment.
            let _ = evaluate_capacity(assessment)?;
        }
        Ok(())
    }
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
        };
        assert_eq!(
            set.validate(),
            Err(ContractError::UnknownAssessmentArea {
                assessment_id: "a".to_owned(),
                area_id: "area-a".to_owned()
            })
        );
    }
}
