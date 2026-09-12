// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-Pareto eligibility cohort for evidence-complete energy materials.
//!
//! This bridge does not compute Pareto ranks. It revalidates screening
//! assessments against an exact Tier-1 policy and emits only complete + feasible
//! generic discovery evaluations for a downstream Pareto adapter.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_discovery::{
    CandidateId, Constraint, ConstraintBound, Evaluation, Feasibility, Objective, Prediction,
};
use symthaea_energy_material_screening::{
    DimensionStatus, EnergyMaterialScreeningAssessment, EnergyMaterialScreeningPolicy,
    EvidenceCompleteness, EvidenceDimension, MetricContract, ScreeningError,
    CAPABILITY_CLASSIFICATION as SCREENING_CAPABILITY_CLASSIFICATION,
};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "PRE-PARETO ELIGIBILITY COHORT ONLY -- not evidence generation, Pareto superiority, scientific validation, material certification, synthesis authority, or deployment approval.";

const REPORT_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.pareto-cohort.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionFailure {
    pub dimension: EvidenceDimension,
    pub status: DimensionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionReason {
    IncompleteEvidence { unavailable: Vec<DimensionFailure> },
    HardConstraintInfeasible,
    UnknownFeasibility,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateExclusion {
    pub candidate_id: CandidateId,
    pub reason: ExclusionReason,
}

/// Canonical, candidate-id-sorted cohort report.
///
/// `eligible_evaluations` are still unranked (`pareto_rank == None`). The shared
/// discovery Pareto bridge is the intended downstream consumer once that stack
/// is integrated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoEligibilityCohort {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub policy_id: String,
    pub policy_sha256: String,
    pub input_candidate_count: usize,
    pub eligible_evaluations: Vec<Evaluation>,
    pub exclusions: Vec<CandidateExclusion>,
}

impl ParetoEligibilityCohort {
    pub fn sha256(&self) -> Result<String, CohortError> {
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn to_json_pretty(&self) -> Result<String, CohortError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[derive(Debug, Error)]
pub enum CohortError {
    #[error("pre-Pareto cohort cannot be empty")]
    EmptyCohort,
    #[error("duplicate screening assessment for candidate {0:?}")]
    DuplicateCandidate(String),
    #[error("candidate {candidate:?} assessment integrity mismatch: {reason}")]
    AssessmentIntegrity { candidate: String, reason: String },
    #[error(transparent)]
    Screening(#[from] ScreeningError),
    #[error("cohort serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Revalidate screening assessments and produce a canonical cohort that is safe
/// to hand to a downstream Pareto adapter.
///
/// The function deliberately does not mutate source assessments and does not
/// assign ranks. Incomplete and infeasible candidates remain visible as
/// explicit exclusions rather than disappearing from the evidence record.
pub fn build_pareto_eligibility_cohort(
    policy: &EnergyMaterialScreeningPolicy,
    assessments: &[EnergyMaterialScreeningAssessment],
) -> Result<ParetoEligibilityCohort, CohortError> {
    policy.validate()?;
    if assessments.is_empty() {
        return Err(CohortError::EmptyCohort);
    }

    let policy_sha256 = policy.sha256()?;
    let expected_objectives = canonical_objectives(policy);
    let expected_constraints = canonical_constraints(policy);
    let expected_contracts = canonical_contracts(policy);

    let mut ordered: Vec<&EnergyMaterialScreeningAssessment> = assessments.iter().collect();
    ordered.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let mut seen = BTreeSet::new();
    let mut eligible_evaluations = Vec::new();
    let mut exclusions = Vec::new();

    for assessment in ordered {
        if !seen.insert(assessment.candidate_id.clone()) {
            return Err(CohortError::DuplicateCandidate(
                assessment.candidate_id.0.clone(),
            ));
        }
        validate_assessment_binding(
            assessment,
            policy,
            &policy_sha256,
            &expected_contracts,
            &expected_objectives,
            &expected_constraints,
        )?;

        match assessment.completeness {
            EvidenceCompleteness::Incomplete => {
                let unavailable = assessment
                    .dimensions
                    .iter()
                    .filter(|dimension| dimension.status != DimensionStatus::Available)
                    .map(|dimension| DimensionFailure {
                        dimension: dimension.dimension,
                        status: dimension.status,
                    })
                    .collect();
                exclusions.push(CandidateExclusion {
                    candidate_id: assessment.candidate_id.clone(),
                    reason: ExclusionReason::IncompleteEvidence { unavailable },
                });
            }
            EvidenceCompleteness::Complete => {
                let evaluation = assessment.evaluation.as_ref().ok_or_else(|| {
                    integrity(
                        assessment,
                        "complete assessment is missing its generic evaluation",
                    )
                })?;
                match evaluation.feasibility().map_err(ScreeningError::from)? {
                    Feasibility::Feasible => eligible_evaluations.push(evaluation.clone()),
                    Feasibility::Infeasible => exclusions.push(CandidateExclusion {
                        candidate_id: assessment.candidate_id.clone(),
                        reason: ExclusionReason::HardConstraintInfeasible,
                    }),
                    Feasibility::Unknown => exclusions.push(CandidateExclusion {
                        candidate_id: assessment.candidate_id.clone(),
                        reason: ExclusionReason::UnknownFeasibility,
                    }),
                }
            }
        }
    }

    Ok(ParetoEligibilityCohort {
        schema: "symthaea.energy-material.pareto-eligibility-cohort.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        policy_id: policy.policy_id.clone(),
        policy_sha256,
        input_candidate_count: assessments.len(),
        eligible_evaluations,
        exclusions,
    })
}

fn validate_assessment_binding(
    assessment: &EnergyMaterialScreeningAssessment,
    policy: &EnergyMaterialScreeningPolicy,
    policy_sha256: &str,
    contracts: &[&MetricContract],
    expected_objectives: &[Objective],
    expected_constraints: &[Constraint],
) -> Result<(), CohortError> {
    CandidateId::new(assessment.candidate_id.0.clone()).map_err(ScreeningError::from)?;
    if assessment.policy_id != policy.policy_id || assessment.policy_sha256 != policy_sha256 {
        return Err(integrity(
            assessment,
            "policy id/digest does not match the supplied screening policy",
        ));
    }
    if assessment.capability_classification != SCREENING_CAPABILITY_CLASSIFICATION {
        return Err(integrity(
            assessment,
            "screening capability classification was altered",
        ));
    }
    if assessment.dimensions.len() != contracts.len() {
        return Err(integrity(
            assessment,
            "dimension count does not match the supplied screening policy",
        ));
    }

    let mut selected_predictions: Vec<Prediction> = Vec::with_capacity(contracts.len());
    let mut all_available = true;
    for (dimension, contract) in assessment.dimensions.iter().zip(contracts.iter()) {
        if dimension.dimension != contract.dimension
            || dimension.metric != contract.metric
            || dimension.unit != contract.unit
        {
            return Err(integrity(
                assessment,
                "dimension schema/order does not match canonical policy order",
            ));
        }
        match dimension.status {
            DimensionStatus::Available => {
                let prediction = dimension.selected_prediction.as_ref().ok_or_else(|| {
                    integrity(
                        assessment,
                        "available dimension is missing its selected prediction",
                    )
                })?;
                if prediction.metric != contract.metric || prediction.unit != contract.unit {
                    return Err(integrity(
                        assessment,
                        "selected prediction does not match dimension metric/unit",
                    ));
                }
                if prediction.fidelity.rank() < contract.minimum_fidelity.rank() {
                    return Err(integrity(
                        assessment,
                        "selected prediction is below the declared minimum fidelity",
                    ));
                }
                if !prediction.evidence.iter().any(|evidence| {
                    contract.accepted_evidence_kinds.contains(&evidence.kind)
                }) {
                    return Err(integrity(
                        assessment,
                        "selected prediction lacks an accepted evidence kind",
                    ));
                }
                prediction.validate().map_err(ScreeningError::from)?;
                selected_predictions.push(prediction.clone());
            }
            _ => {
                all_available = false;
                if dimension.selected_prediction.is_some() {
                    return Err(integrity(
                        assessment,
                        "unavailable dimension unexpectedly carries a selected prediction",
                    ));
                }
            }
        }
    }

    match assessment.completeness {
        EvidenceCompleteness::Complete if !all_available => Err(integrity(
            assessment,
            "assessment says complete but one or more dimensions are unavailable",
        )),
        EvidenceCompleteness::Incomplete if all_available => Err(integrity(
            assessment,
            "assessment says incomplete but every dimension is available",
        )),
        EvidenceCompleteness::Incomplete => {
            if assessment.evaluation.is_some() {
                return Err(integrity(
                    assessment,
                    "incomplete assessment must not carry a generic evaluation",
                ));
            }
            Ok(())
        }
        EvidenceCompleteness::Complete => {
            let evaluation = assessment.evaluation.as_ref().ok_or_else(|| {
                integrity(
                    assessment,
                    "complete assessment is missing its generic evaluation",
                )
            })?;
            evaluation.validate().map_err(ScreeningError::from)?;
            if evaluation.candidate_id != assessment.candidate_id {
                return Err(integrity(
                    assessment,
                    "evaluation candidate id does not match assessment candidate id",
                ));
            }
            if evaluation.pareto_rank.is_some() {
                return Err(integrity(
                    assessment,
                    "screening assessment already carries a Pareto rank",
                ));
            }
            if evaluation.objectives != expected_objectives {
                return Err(integrity(
                    assessment,
                    "evaluation objective schema/order does not match canonical policy",
                ));
            }
            if evaluation.constraints != expected_constraints {
                return Err(integrity(
                    assessment,
                    "evaluation constraints do not match canonical policy",
                ));
            }
            if evaluation.predictions != selected_predictions {
                return Err(integrity(
                    assessment,
                    "evaluation predictions do not equal canonical selected dimension evidence",
                ));
            }
            Ok(())
        }
    }
}

fn canonical_contracts(policy: &EnergyMaterialScreeningPolicy) -> Vec<&MetricContract> {
    let mut contracts: Vec<&MetricContract> = policy.contracts.iter().collect();
    contracts.sort_by_key(|contract| dimension_code(contract.dimension));
    contracts
}

fn canonical_objectives(policy: &EnergyMaterialScreeningPolicy) -> Vec<Objective> {
    canonical_contracts(policy)
        .into_iter()
        .map(MetricContract::objective)
        .collect()
}

fn canonical_constraints(policy: &EnergyMaterialScreeningPolicy) -> Vec<Constraint> {
    let mut constraints = policy.constraints.clone();
    constraints.sort_by(|left, right| {
        left.metric
            .cmp(&right.metric)
            .then_with(|| left.unit.cmp(&right.unit))
            .then_with(|| constraint_bound_key(left.bound).cmp(&constraint_bound_key(right.bound)))
    });
    constraints
}

fn dimension_code(dimension: EvidenceDimension) -> u8 {
    match dimension {
        EvidenceDimension::FunctionalPerformance => 0,
        EvidenceDimension::ThermodynamicStability => 1,
        EvidenceDimension::CriticalMaterialBurden => 2,
        EvidenceDimension::SupplyResilience => 3,
        EvidenceDimension::HumanEnvironmentalHazard => 4,
        EvidenceDimension::Circularity => 5,
        EvidenceDimension::Manufacturability => 6,
    }
}

fn constraint_bound_key(bound: ConstraintBound) -> (u8, u64, u64) {
    match bound {
        ConstraintBound::AtLeast(value) => (0, value.to_bits(), 0),
        ConstraintBound::AtMost(value) => (1, value.to_bits(), 0),
        ConstraintBound::Between { min, max } => (2, min.to_bits(), max.to_bits()),
    }
}

fn integrity(assessment: &EnergyMaterialScreeningAssessment, reason: impl Into<String>) -> CohortError {
    CohortError::AssessmentIntegrity {
        candidate: assessment.candidate_id.0.clone(),
        reason: reason.into(),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_discovery::{
        ConstraintBound, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance,
        ObjectiveDirection, Prediction, UncertaintyEstimate,
    };
    use symthaea_energy_material_screening::{
        EnergyMaterialEvidenceBundle, EvidenceDimension, MetricContract,
    };

    fn contract(
        dimension: EvidenceDimension,
        metric: &str,
        direction: ObjectiveDirection,
    ) -> MetricContract {
        MetricContract {
            dimension,
            metric: metric.into(),
            unit: "score".into(),
            direction,
            minimum_fidelity: FidelityLevel::Surrogate,
            accepted_evidence_kinds: vec![EvidenceKind::Dataset],
        }
    }

    fn policy() -> EnergyMaterialScreeningPolicy {
        EnergyMaterialScreeningPolicy {
            policy_id: "tier1-cohort-fixture".into(),
            contracts: vec![
                contract(
                    EvidenceDimension::FunctionalPerformance,
                    "functional",
                    ObjectiveDirection::Maximize,
                ),
                contract(
                    EvidenceDimension::ThermodynamicStability,
                    "stability",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::CriticalMaterialBurden,
                    "criticality",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::SupplyResilience,
                    "supply",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::HumanEnvironmentalHazard,
                    "hazard",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::Circularity,
                    "circularity",
                    ObjectiveDirection::Maximize,
                ),
                contract(
                    EvidenceDimension::Manufacturability,
                    "manufacturability",
                    ObjectiveDirection::Maximize,
                ),
            ],
            constraints: vec![Constraint {
                metric: "stability".into(),
                unit: "score".into(),
                bound: ConstraintBound::AtMost(0.5),
            }],
        }
    }

    fn prediction(metric: &str, value: f64) -> Prediction {
        Prediction {
            metric: metric.into(),
            value,
            unit: "score".into(),
            uncertainty: UncertaintyEstimate::new(0.2, 0.1).unwrap(),
            fidelity: FidelityLevel::ExternalSimulation,
            model: ModelProvenance::named("fixture").unwrap(),
            assumptions: vec![],
            evidence: vec![EvidenceRef {
                id: format!("e-{metric}"),
                kind: EvidenceKind::Dataset,
                uri: Some("https://example.invalid/evidence".into()),
                digest: None,
                note: None,
            }],
        }
    }

    fn bundle(id: &str, stability: f64) -> EnergyMaterialEvidenceBundle {
        EnergyMaterialEvidenceBundle {
            candidate_id: CandidateId::new(id).unwrap(),
            predictions: vec![
                prediction("functional", 0.9),
                prediction("stability", stability),
                prediction("criticality", 0.2),
                prediction("supply", 0.3),
                prediction("hazard", 0.1),
                prediction("circularity", 0.8),
                prediction("manufacturability", 0.7),
            ],
        }
    }

    #[test]
    fn cohort_retains_incomplete_and_infeasible_exclusions() {
        let policy = policy();
        let feasible = bundle("a-feasible", 0.2).assess(&policy).unwrap();
        let infeasible = bundle("b-infeasible", 0.8).assess(&policy).unwrap();
        let mut incomplete_bundle = bundle("c-incomplete", 0.2);
        incomplete_bundle
            .predictions
            .retain(|prediction| prediction.metric != "hazard");
        let incomplete = incomplete_bundle.assess(&policy).unwrap();

        let report = build_pareto_eligibility_cohort(
            &policy,
            &[incomplete, feasible, infeasible],
        )
        .unwrap();
        assert_eq!(report.eligible_evaluations.len(), 1);
        assert_eq!(report.eligible_evaluations[0].candidate_id.0, "a-feasible");
        assert_eq!(report.exclusions.len(), 2);
        assert!(report.eligible_evaluations.iter().all(|evaluation| evaluation.pareto_rank.is_none()));
    }

    #[test]
    fn input_order_does_not_change_report_identity() {
        let policy = policy();
        let a = bundle("a", 0.2).assess(&policy).unwrap();
        let b = bundle("b", 0.3).assess(&policy).unwrap();
        let first = build_pareto_eligibility_cohort(&policy, &[a.clone(), b.clone()]).unwrap();
        let second = build_pareto_eligibility_cohort(&policy, &[b, a]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }

    #[test]
    fn forged_policy_identity_fails_closed() {
        let policy = policy();
        let mut assessment = bundle("a", 0.2).assess(&policy).unwrap();
        assessment.policy_sha256 = "0".repeat(64);
        assert!(matches!(
            build_pareto_eligibility_cohort(&policy, &[assessment]),
            Err(CohortError::AssessmentIntegrity { .. })
        ));
    }

    #[test]
    fn preexisting_pareto_rank_is_rejected() {
        let policy = policy();
        let mut assessment = bundle("a", 0.2).assess(&policy).unwrap();
        assessment.evaluation.as_mut().unwrap().pareto_rank = Some(0);
        assert!(matches!(
            build_pareto_eligibility_cohort(&policy, &[assessment]),
            Err(CohortError::AssessmentIntegrity { .. })
        ));
    }

    #[test]
    fn tampered_objective_direction_is_rejected() {
        let policy = policy();
        let mut assessment = bundle("a", 0.2).assess(&policy).unwrap();
        assessment.evaluation.as_mut().unwrap().objectives[0].direction = ObjectiveDirection::Minimize;
        assert!(matches!(
            build_pareto_eligibility_cohort(&policy, &[assessment]),
            Err(CohortError::AssessmentIntegrity { .. })
        ));
    }

    #[test]
    fn duplicate_candidate_assessments_fail_closed() {
        let policy = policy();
        let assessment = bundle("same", 0.2).assess(&policy).unwrap();
        assert!(matches!(
            build_pareto_eligibility_cohort(&policy, &[assessment.clone(), assessment]),
            Err(CohortError::DuplicateCandidate(id)) if id == "same"
        ));
    }
}
