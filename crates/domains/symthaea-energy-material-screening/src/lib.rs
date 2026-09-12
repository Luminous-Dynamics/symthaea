// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-complete, multi-objective energy-material screening contracts.
//!
//! This crate defines what evidence must be present before an energy material
//! is eligible for downstream feasibility/Pareto comparison. It does not fetch
//! datasets, invent universal hazard/criticality scores, rank candidates, run
//! experiments, synthesize materials, or authorize real-world action.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use symthaea_discovery::{
    CandidateId, Constraint, DiscoveryError, Evaluation, EvidenceKind, Feasibility, FidelityLevel,
    Objective, ObjectiveDirection, Prediction,
};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "ENERGY-MATERIAL SCREENING CONTRACT ONLY -- evidence completeness and feasibility are not novelty, synthesis success, device validation, deployment approval, or physical authority.";

const POLICY_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material-screening.policy.v0\0";

/// Common metric names that have reasonably concrete physical interpretations.
///
/// The policy remains metric-pluggable; especially for hazard and supply, a
/// caller should choose a metric backed by an identified dataset/method rather
/// than treating these constants as a complete ontology.
pub mod metric {
    pub const BAND_GAP: &str = "band_gap";
    pub const ENERGY_ABOVE_HULL: &str = "energy_above_hull";
    pub const CRITICAL_MATERIAL_MASS_FRACTION: &str = "critical_material_mass_fraction";
    pub const SUPPLY_CONCENTRATION_HHI: &str = "supply_concentration_hhi";
    pub const RECYCLABILITY_FRACTION: &str = "recyclability_fraction";
    pub const MAXIMUM_PROCESS_TEMPERATURE: &str = "maximum_process_temperature";
    pub const SYNTHESIS_STEP_COUNT: &str = "synthesis_step_count";
}

pub mod unit {
    pub const EV: &str = "eV";
    pub const EV_PER_ATOM: &str = "eV/atom";
    pub const FRACTION: &str = "fraction";
    pub const KELVIN: &str = "K";
    pub const COUNT: &str = "count";
    pub const SCORE: &str = "score";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDimension {
    FunctionalPerformance,
    ThermodynamicStability,
    CriticalMaterialBurden,
    SupplyResilience,
    HumanEnvironmentalHazard,
    Circularity,
    Manufacturability,
}

impl EvidenceDimension {
    pub const ALL: [EvidenceDimension; 7] = [
        EvidenceDimension::FunctionalPerformance,
        EvidenceDimension::ThermodynamicStability,
        EvidenceDimension::CriticalMaterialBurden,
        EvidenceDimension::SupplyResilience,
        EvidenceDimension::HumanEnvironmentalHazard,
        EvidenceDimension::Circularity,
        EvidenceDimension::Manufacturability,
    ];

    const fn code(self) -> u8 {
        match self {
            EvidenceDimension::FunctionalPerformance => 0,
            EvidenceDimension::ThermodynamicStability => 1,
            EvidenceDimension::CriticalMaterialBurden => 2,
            EvidenceDimension::SupplyResilience => 3,
            EvidenceDimension::HumanEnvironmentalHazard => 4,
            EvidenceDimension::Circularity => 5,
            EvidenceDimension::Manufacturability => 6,
        }
    }
}

/// Exact evidence contract for one screening dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricContract {
    pub dimension: EvidenceDimension,
    pub metric: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
    pub minimum_fidelity: FidelityLevel,
    /// At least one evidence reference on the selected prediction must have an
    /// accepted kind. This list is policy, not a global ranking of evidence.
    pub accepted_evidence_kinds: Vec<EvidenceKind>,
}

impl MetricContract {
    pub fn validate(&self) -> Result<(), ScreeningError> {
        if self.metric.trim().is_empty() || self.unit.trim().is_empty() {
            return Err(ScreeningError::InvalidPolicy(
                "metric contracts require non-empty metric and unit".into(),
            ));
        }
        if let ObjectiveDirection::Target { value, tolerance } = self.direction {
            if !value.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
                return Err(ScreeningError::InvalidPolicy(
                    "target direction requires finite value and non-negative tolerance".into(),
                ));
            }
        }
        if self.accepted_evidence_kinds.is_empty() {
            return Err(ScreeningError::InvalidPolicy(format!(
                "dimension {:?} requires at least one accepted evidence kind",
                self.dimension
            )));
        }
        let mut seen = HashSet::new();
        for kind in &self.accepted_evidence_kinds {
            if !seen.insert(*kind) {
                return Err(ScreeningError::InvalidPolicy(format!(
                    "dimension {:?} repeats evidence kind {kind:?}",
                    self.dimension
                )));
            }
        }
        Ok(())
    }

    pub fn objective(&self) -> Objective {
        Objective {
            metric: self.metric.clone(),
            unit: self.unit.clone(),
            direction: self.direction,
        }
    }
}

/// A Tier-1 energy-material policy requires exactly one metric contract for
/// every canonical evidence dimension. There are no objective weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMaterialScreeningPolicy {
    pub policy_id: String,
    pub contracts: Vec<MetricContract>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
}

impl EnergyMaterialScreeningPolicy {
    pub fn validate(&self) -> Result<(), ScreeningError> {
        if self.policy_id.trim().is_empty() {
            return Err(ScreeningError::InvalidPolicy(
                "policy id cannot be empty".into(),
            ));
        }
        if self.contracts.len() != EvidenceDimension::ALL.len() {
            return Err(ScreeningError::InvalidPolicy(format!(
                "Tier-1 policy requires exactly {} dimension contracts, got {}",
                EvidenceDimension::ALL.len(),
                self.contracts.len()
            )));
        }

        let mut dimensions = HashSet::new();
        let mut metric_units = BTreeSet::new();
        for contract in &self.contracts {
            contract.validate()?;
            if !dimensions.insert(contract.dimension) {
                return Err(ScreeningError::InvalidPolicy(format!(
                    "duplicate dimension {:?}",
                    contract.dimension
                )));
            }
            if !metric_units.insert((contract.metric.clone(), contract.unit.clone())) {
                return Err(ScreeningError::InvalidPolicy(format!(
                    "one metric/unit pair cannot stand in for multiple evidence dimensions: {}/{}",
                    contract.metric, contract.unit
                )));
            }
        }
        for required in EvidenceDimension::ALL {
            if !dimensions.contains(&required) {
                return Err(ScreeningError::InvalidPolicy(format!(
                    "missing required evidence dimension {required:?}"
                )));
            }
        }

        for constraint in &self.constraints {
            constraint.validate()?;
            if !self.contracts.iter().any(|contract| {
                contract.metric == constraint.metric && contract.unit == constraint.unit
            }) {
                return Err(ScreeningError::InvalidPolicy(format!(
                    "constraint {}/{} must reference a declared dimension metric",
                    constraint.metric, constraint.unit
                )));
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ScreeningError> {
        self.validate()?;
        let mut contracts: Vec<&MetricContract> = self.contracts.iter().collect();
        contracts.sort_by_key(|contract| contract.dimension.code());
        let mut constraints: Vec<&Constraint> = self.constraints.iter().collect();
        constraints.sort_by(|left, right| {
            left.metric
                .cmp(&right.metric)
                .then_with(|| left.unit.cmp(&right.unit))
                .then_with(|| format!("{:?}", left.bound).cmp(&format!("{:?}", right.bound)))
        });

        let mut hasher = Sha256::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        update_text(&mut hasher, &self.policy_id);
        for contract in contracts {
            hasher.update([contract.dimension.code()]);
            update_text(&mut hasher, &contract.metric);
            update_text(&mut hasher, &contract.unit);
            hash_direction(&mut hasher, contract.direction);
            hasher.update([contract.minimum_fidelity.rank()]);
            let mut kinds = contract.accepted_evidence_kinds.clone();
            kinds.sort_by_key(|kind| evidence_kind_code(*kind));
            for kind in kinds {
                hasher.update([evidence_kind_code(kind)]);
            }
            hasher.update([0xff]);
        }
        for constraint in constraints {
            update_text(&mut hasher, &constraint.metric);
            update_text(&mut hasher, &constraint.unit);
            update_text(&mut hasher, &format!("{:?}", constraint.bound));
        }
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionStatus {
    Available,
    Missing,
    BelowMinimumFidelity,
    UnsupportedEvidenceKind,
    AmbiguousHighestFidelity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionAssessment {
    pub dimension: EvidenceDimension,
    pub metric: String,
    pub unit: String,
    pub status: DimensionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highest_observed_fidelity: Option<FidelityLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_prediction: Option<Prediction>,
    pub matching_prediction_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceCompleteness {
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMaterialScreeningAssessment {
    pub candidate_id: CandidateId,
    pub policy_id: String,
    pub policy_sha256: String,
    pub capability_classification: String,
    pub completeness: EvidenceCompleteness,
    pub dimensions: Vec<DimensionAssessment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<Evaluation>,
}

impl EnergyMaterialScreeningAssessment {
    pub fn feasibility(&self) -> Result<Option<Feasibility>, ScreeningError> {
        self.evaluation
            .as_ref()
            .map(Evaluation::feasibility)
            .transpose()
            .map_err(ScreeningError::Discovery)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMaterialEvidenceBundle {
    pub candidate_id: CandidateId,
    #[serde(default)]
    pub predictions: Vec<Prediction>,
}

impl EnergyMaterialEvidenceBundle {
    pub fn assess(
        &self,
        policy: &EnergyMaterialScreeningPolicy,
    ) -> Result<EnergyMaterialScreeningAssessment, ScreeningError> {
        policy.validate()?;
        CandidateId::new(self.candidate_id.0.clone())?;
        for prediction in &self.predictions {
            prediction.validate()?;
        }

        let mut contracts: Vec<&MetricContract> = policy.contracts.iter().collect();
        contracts.sort_by_key(|contract| contract.dimension.code());
        let mut dimensions = Vec::with_capacity(contracts.len());
        let mut selected_predictions = Vec::with_capacity(contracts.len());

        for contract in contracts {
            let assessment = self.assess_dimension(contract)?;
            if let Some(prediction) = &assessment.selected_prediction {
                selected_predictions.push(prediction.clone());
            }
            dimensions.push(assessment);
        }

        let complete = dimensions
            .iter()
            .all(|assessment| assessment.status == DimensionStatus::Available);
        let evaluation = if complete {
            let evaluation = Evaluation {
                candidate_id: self.candidate_id.clone(),
                objectives: policy.contracts.iter().map(MetricContract::objective).collect(),
                constraints: policy.constraints.clone(),
                predictions: selected_predictions,
                pareto_rank: None,
            };
            evaluation.validate()?;
            Some(evaluation)
        } else {
            None
        };

        Ok(EnergyMaterialScreeningAssessment {
            candidate_id: self.candidate_id.clone(),
            policy_id: policy.policy_id.clone(),
            policy_sha256: policy.sha256()?,
            capability_classification: CAPABILITY_CLASSIFICATION.to_owned(),
            completeness: if complete {
                EvidenceCompleteness::Complete
            } else {
                EvidenceCompleteness::Incomplete
            },
            dimensions,
            evaluation,
        })
    }

    fn assess_dimension(
        &self,
        contract: &MetricContract,
    ) -> Result<DimensionAssessment, ScreeningError> {
        let same_metric: Vec<&Prediction> = self
            .predictions
            .iter()
            .filter(|prediction| prediction.metric == contract.metric)
            .collect();
        if same_metric.is_empty() {
            return Ok(dimension_result(contract, DimensionStatus::Missing, None, None, 0));
        }

        let matching_unit: Vec<&Prediction> = same_metric
            .iter()
            .copied()
            .filter(|prediction| prediction.unit == contract.unit)
            .collect();
        if matching_unit.is_empty() {
            let mut found: Vec<String> = same_metric
                .iter()
                .map(|prediction| prediction.unit.clone())
                .collect();
            found.sort();
            found.dedup();
            return Err(ScreeningError::UnitMismatch {
                dimension: contract.dimension,
                metric: contract.metric.clone(),
                expected: contract.unit.clone(),
                found,
            });
        }

        let highest_observed = matching_unit
            .iter()
            .map(|prediction| prediction.fidelity)
            .max_by_key(|fidelity| fidelity.rank());
        let fidelity_eligible: Vec<&Prediction> = matching_unit
            .iter()
            .copied()
            .filter(|prediction| {
                prediction.fidelity.rank() >= contract.minimum_fidelity.rank()
            })
            .collect();
        if fidelity_eligible.is_empty() {
            return Ok(dimension_result(
                contract,
                DimensionStatus::BelowMinimumFidelity,
                highest_observed,
                None,
                matching_unit.len(),
            ));
        }

        let evidence_eligible: Vec<&Prediction> = fidelity_eligible
            .iter()
            .copied()
            .filter(|prediction| {
                prediction.evidence.iter().any(|evidence| {
                    contract.accepted_evidence_kinds.contains(&evidence.kind)
                })
            })
            .collect();
        if evidence_eligible.is_empty() {
            return Ok(dimension_result(
                contract,
                DimensionStatus::UnsupportedEvidenceKind,
                highest_observed,
                None,
                matching_unit.len(),
            ));
        }

        let max_rank = evidence_eligible
            .iter()
            .map(|prediction| prediction.fidelity.rank())
            .max()
            .expect("evidence_eligible is non-empty");
        let highest: Vec<&Prediction> = evidence_eligible
            .into_iter()
            .filter(|prediction| prediction.fidelity.rank() == max_rank)
            .collect();
        if highest.len() != 1 {
            return Ok(dimension_result(
                contract,
                DimensionStatus::AmbiguousHighestFidelity,
                Some(highest[0].fidelity),
                None,
                matching_unit.len(),
            ));
        }

        Ok(dimension_result(
            contract,
            DimensionStatus::Available,
            Some(highest[0].fidelity),
            Some(highest[0].clone()),
            matching_unit.len(),
        ))
    }
}

fn dimension_result(
    contract: &MetricContract,
    status: DimensionStatus,
    highest_observed_fidelity: Option<FidelityLevel>,
    selected_prediction: Option<Prediction>,
    matching_prediction_count: usize,
) -> DimensionAssessment {
    DimensionAssessment {
        dimension: contract.dimension,
        metric: contract.metric.clone(),
        unit: contract.unit.clone(),
        status,
        highest_observed_fidelity,
        selected_prediction,
        matching_prediction_count,
    }
}

#[derive(Debug, Error)]
pub enum ScreeningError {
    #[error("invalid energy-material screening policy: {0}")]
    InvalidPolicy(String),
    #[error(
        "unit mismatch for dimension {dimension:?} metric {metric:?}: expected {expected:?}, found {found:?}"
    )]
    UnitMismatch {
        dimension: EvidenceDimension,
        metric: String,
        expected: String,
        found: Vec<String>,
    },
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
}

fn evidence_kind_code(kind: EvidenceKind) -> u8 {
    match kind {
        EvidenceKind::Literature => 0,
        EvidenceKind::Dataset => 1,
        EvidenceKind::Heuristic => 2,
        EvidenceKind::SurrogateModel => 3,
        EvidenceKind::AnalyticalModel => 4,
        EvidenceKind::FirstPrinciplesSimulation => 5,
        EvidenceKind::ExternalSimulation => 6,
        EvidenceKind::Experiment => 7,
        EvidenceKind::IndependentReplication => 8,
        EvidenceKind::DeviceValidation => 9,
        EvidenceKind::FieldObservation => 10,
    }
}

fn hash_direction(hasher: &mut Sha256, direction: ObjectiveDirection) {
    match direction {
        ObjectiveDirection::Maximize => hasher.update([0]),
        ObjectiveDirection::Minimize => hasher.update([1]),
        ObjectiveDirection::Target { value, tolerance } => {
            hasher.update([2]);
            hasher.update(value.to_bits().to_le_bytes());
            hasher.update(tolerance.to_bits().to_le_bytes());
        }
    }
}

fn update_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
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
        ConstraintBound, EvidenceRef, ModelProvenance, UncertaintyEstimate,
    };

    fn contract(
        dimension: EvidenceDimension,
        metric: &str,
        unit: &str,
        direction: ObjectiveDirection,
    ) -> MetricContract {
        MetricContract {
            dimension,
            metric: metric.into(),
            unit: unit.into(),
            direction,
            minimum_fidelity: FidelityLevel::Dataset,
            accepted_evidence_kinds: vec![EvidenceKind::Dataset, EvidenceKind::Experiment],
        }
    }

    fn policy() -> EnergyMaterialScreeningPolicy {
        EnergyMaterialScreeningPolicy {
            policy_id: "tier1-fixture".into(),
            contracts: vec![
                contract(
                    EvidenceDimension::FunctionalPerformance,
                    "functional",
                    "score",
                    ObjectiveDirection::Maximize,
                ),
                contract(
                    EvidenceDimension::ThermodynamicStability,
                    "stability",
                    "eV/atom",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::CriticalMaterialBurden,
                    "critical_fraction",
                    "fraction",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::SupplyResilience,
                    "supply_risk",
                    "score",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::HumanEnvironmentalHazard,
                    "hazard",
                    "score",
                    ObjectiveDirection::Minimize,
                ),
                contract(
                    EvidenceDimension::Circularity,
                    "recyclability",
                    "fraction",
                    ObjectiveDirection::Maximize,
                ),
                contract(
                    EvidenceDimension::Manufacturability,
                    "manufacturability",
                    "score",
                    ObjectiveDirection::Maximize,
                ),
            ],
            constraints: vec![Constraint {
                metric: "stability".into(),
                unit: "eV/atom".into(),
                bound: ConstraintBound::AtMost(0.2),
            }],
        }
    }

    fn prediction(metric: &str, unit: &str, value: f64) -> Prediction {
        Prediction {
            metric: metric.into(),
            value,
            unit: unit.into(),
            uncertainty: UncertaintyEstimate::new(0.2, 0.1).unwrap(),
            fidelity: FidelityLevel::ExternalSimulation,
            model: ModelProvenance::named("fixture-model").unwrap(),
            assumptions: vec![],
            evidence: vec![EvidenceRef {
                id: format!("evidence-{metric}"),
                kind: EvidenceKind::Dataset,
                uri: Some("https://example.invalid/evidence".into()),
                digest: None,
                note: None,
            }],
        }
    }

    fn complete_bundle() -> EnergyMaterialEvidenceBundle {
        EnergyMaterialEvidenceBundle {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            predictions: vec![
                prediction("functional", "score", 0.9),
                prediction("stability", "eV/atom", 0.1),
                prediction("critical_fraction", "fraction", 0.2),
                prediction("supply_risk", "score", 0.3),
                prediction("hazard", "score", 0.1),
                prediction("recyclability", "fraction", 0.8),
                prediction("manufacturability", "score", 0.7),
            ],
        }
    }

    #[test]
    fn policy_requires_all_dimensions_and_distinct_metric_contracts() {
        let mut missing = policy();
        missing.contracts.pop();
        assert!(missing.validate().is_err());

        let mut duplicate_metric = policy();
        duplicate_metric.contracts[6].metric = duplicate_metric.contracts[5].metric.clone();
        duplicate_metric.contracts[6].unit = duplicate_metric.contracts[5].unit.clone();
        assert!(duplicate_metric.validate().is_err());
    }

    #[test]
    fn missing_dimension_stays_unknown_instead_of_becoming_zero() {
        let mut bundle = complete_bundle();
        bundle.predictions.retain(|prediction| prediction.metric != "hazard");
        let assessment = bundle.assess(&policy()).unwrap();
        assert_eq!(assessment.completeness, EvidenceCompleteness::Incomplete);
        assert!(assessment.evaluation.is_none());
        let hazard = assessment
            .dimensions
            .iter()
            .find(|dimension| dimension.dimension == EvidenceDimension::HumanEnvironmentalHazard)
            .unwrap();
        assert_eq!(hazard.status, DimensionStatus::Missing);
    }

    #[test]
    fn complete_evidence_produces_generic_evaluation_and_feasibility() {
        let assessment = complete_bundle().assess(&policy()).unwrap();
        assert_eq!(assessment.completeness, EvidenceCompleteness::Complete);
        assert_eq!(assessment.feasibility().unwrap(), Some(Feasibility::Feasible));
        let evaluation = assessment.evaluation.unwrap();
        assert_eq!(evaluation.objectives.len(), 7);
        assert_eq!(evaluation.predictions.len(), 7);
        assert_eq!(evaluation.pareto_rank, None);
    }

    #[test]
    fn violated_hard_constraint_is_not_confused_with_incomplete_evidence() {
        let mut bundle = complete_bundle();
        bundle
            .predictions
            .iter_mut()
            .find(|prediction| prediction.metric == "stability")
            .unwrap()
            .value = 0.4;
        let assessment = bundle.assess(&policy()).unwrap();
        assert_eq!(assessment.completeness, EvidenceCompleteness::Complete);
        assert_eq!(assessment.feasibility().unwrap(), Some(Feasibility::Infeasible));
    }

    #[test]
    fn low_fidelity_and_wrong_evidence_kind_do_not_satisfy_dimension() {
        let mut bundle = complete_bundle();
        let hazard = bundle
            .predictions
            .iter_mut()
            .find(|prediction| prediction.metric == "hazard")
            .unwrap();
        hazard.fidelity = FidelityLevel::Heuristic;
        let assessment = bundle.assess(&policy()).unwrap();
        let hazard_assessment = assessment
            .dimensions
            .iter()
            .find(|dimension| dimension.dimension == EvidenceDimension::HumanEnvironmentalHazard)
            .unwrap();
        assert_eq!(hazard_assessment.status, DimensionStatus::BelowMinimumFidelity);

        let mut bundle = complete_bundle();
        let hazard = bundle
            .predictions
            .iter_mut()
            .find(|prediction| prediction.metric == "hazard")
            .unwrap();
        hazard.evidence[0].kind = EvidenceKind::Heuristic;
        let assessment = bundle.assess(&policy()).unwrap();
        let hazard_assessment = assessment
            .dimensions
            .iter()
            .find(|dimension| dimension.dimension == EvidenceDimension::HumanEnvironmentalHazard)
            .unwrap();
        assert_eq!(hazard_assessment.status, DimensionStatus::UnsupportedEvidenceKind);
    }

    #[test]
    fn equal_highest_fidelity_predictions_are_ambiguous() {
        let mut bundle = complete_bundle();
        bundle.predictions.push(prediction("functional", "score", 0.8));
        let assessment = bundle.assess(&policy()).unwrap();
        let functional = assessment
            .dimensions
            .iter()
            .find(|dimension| dimension.dimension == EvidenceDimension::FunctionalPerformance)
            .unwrap();
        assert_eq!(functional.status, DimensionStatus::AmbiguousHighestFidelity);
        assert_eq!(assessment.completeness, EvidenceCompleteness::Incomplete);
    }

    #[test]
    fn policy_digest_is_order_independent_for_dimension_contracts() {
        let first = policy();
        let mut second = first.clone();
        second.contracts.reverse();
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }
}
