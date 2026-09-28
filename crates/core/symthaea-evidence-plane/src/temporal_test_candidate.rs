// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit candidate-test materialization for temporal scientific discrimination.
//!
//! This module packages an existing 002N temporal evidence-gap candidate into a
//! concrete, externally executable test specification. It deliberately does
//! not choose a winner, execute a test, emit an observation, or promote a
//! computational prediction into evidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

use super::temporal_evidence_gap::TemporalEvidenceGapPlan;
use symthaea_core::scientific_temporal_outcome_discrimination::TemporalOutcomeDiscrimination;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalTestCandidateSpec {
    pub candidate_id: String,
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub outcome_id: String,
    pub horizon_seconds: f32,
    pub advances_unmet_predicates: BTreeSet<String>,
    /// Caller-supplied identifier for the executable test specification.
    pub test_specification_id: String,
    /// Caller-supplied identifier for the measurement/observation definition.
    pub measurement_specification_id: String,
    /// Caller-supplied planning metadata; not an observed cost.
    pub estimated_cost: f64,
    /// Caller-supplied planning metadata; not an observed risk.
    pub pragmatic_risk: f64,
    /// Identity of the 002N candidate from which this object was materialized.
    pub source_candidate_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalTestCandidateSet {
    pub candidates: Vec<TemporalTestCandidateSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalTestCandidateError {
    EmptyCandidateId,
    DuplicateCandidateId,
    EmptyTestSpecificationId,
    EmptyMeasurementSpecificationId,
    EmptySourceCandidateId,
    EmptyModelId,
    EmptyLineage,
    EmptyOutcomeId,
    InvalidHorizon,
    NonFinitePlanningMetadata,
    UnknownSourceCandidate,
    SourceBindingMismatch,
    EmptyEvidencePredicates,
}

impl fmt::Display for TemporalTestCandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TemporalTestCandidateError {}

/// Return the canonical, unambiguous source identity for a temporal discrimination candidate.
pub fn source_candidate_id(candidate: &TemporalOutcomeDiscrimination) -> String {
    let fields = [
        candidate.left_model_id.as_str(),
        candidate.right_model_id.as_str(),
        candidate.left_lineage.as_str(),
        candidate.right_lineage.as_str(),
        candidate.outcome_id.as_str(),
    ];
    let mut identity = String::from("temporal-discrimination-v1");
    for field in fields {
        identity.push_str(&format!(":{}:{}", field.len(), field));
    }
    identity.push_str(&format!(":h:{:08x}", candidate.horizon_seconds.to_bits()));
    identity
}

/// Materialize caller-supplied executable test specifications from 002N gaps.
///
/// Every specification must bind exactly to an existing temporal discrimination
/// candidate. Input order is irrelevant; output is deterministically sorted.
/// No candidate is ranked or selected.
pub fn materialize(
    plan: &TemporalEvidenceGapPlan,
    specifications: &[TemporalTestCandidateSpecInput],
) -> Result<TemporalTestCandidateSet, TemporalTestCandidateError> {
    let mut sources = std::collections::BTreeMap::new();

    for assessment in &plan.assessments {
        for candidate in &assessment.candidates {
            sources.insert(source_candidate_id(candidate), (assessment, candidate));
        }
    }

    let mut seen = BTreeSet::new();
    let mut candidates = Vec::with_capacity(specifications.len());

    for input in specifications {
        validate_input(input)?;
        if !sources.contains_key(input.source_candidate_id.as_str()) {
            return Err(TemporalTestCandidateError::UnknownSourceCandidate);
        }
        if !seen.insert(input.candidate_id.clone()) {
            return Err(TemporalTestCandidateError::DuplicateCandidateId);
        }

        let (_, source) = sources
            .get(input.source_candidate_id.as_str())
            .expect("validated source candidate");

        if input.left_model_id != source.left_model_id
            || input.right_model_id != source.right_model_id
            || input.left_lineage != source.left_lineage
            || input.right_lineage != source.right_lineage
            || input.outcome_id != source.outcome_id
            || input.horizon_seconds.to_bits() != source.horizon_seconds.to_bits()
        {
            return Err(TemporalTestCandidateError::SourceBindingMismatch);
        }

        let advances_unmet_predicates = plan
            .assessments
            .iter()
            .find(|a| a.left_model_id == source.left_model_id
                && a.right_model_id == source.right_model_id
                && a.outcome_id == source.outcome_id
                && a.left_lineage == source.left_lineage
                && a.right_lineage == source.right_lineage)
            .map(|a| a.advances_unmet_predicates.clone())
            .unwrap_or_default();

        if advances_unmet_predicates.is_empty() {
            return Err(TemporalTestCandidateError::EmptyEvidencePredicates);
        }

        candidates.push(TemporalTestCandidateSpec {
            candidate_id: input.candidate_id.clone(),
            left_model_id: input.left_model_id.clone(),
            right_model_id: input.right_model_id.clone(),
            left_lineage: input.left_lineage.clone(),
            right_lineage: input.right_lineage.clone(),
            outcome_id: input.outcome_id.clone(),
            horizon_seconds: input.horizon_seconds,
            advances_unmet_predicates,
            test_specification_id: input.test_specification_id.clone(),
            measurement_specification_id: input.measurement_specification_id.clone(),
            estimated_cost: input.estimated_cost,
            pragmatic_risk: input.pragmatic_risk,
            source_candidate_id: input.source_candidate_id.clone(),
        });
    }

    candidates.sort_by(|a, b| {
        a.candidate_id
            .cmp(&b.candidate_id)
            .then_with(|| a.test_specification_id.cmp(&b.test_specification_id))
    });

    Ok(TemporalTestCandidateSet { candidates })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalTestCandidateSpecInput {
    pub candidate_id: String,
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub outcome_id: String,
    pub horizon_seconds: f32,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
    pub estimated_cost: f64,
    pub pragmatic_risk: f64,
    pub source_candidate_id: String,
}

fn validate_input(input: &TemporalTestCandidateSpecInput) -> Result<(), TemporalTestCandidateError> {
    if input.candidate_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyCandidateId);
    }
    if input.test_specification_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyTestSpecificationId);
    }
    if input.measurement_specification_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyMeasurementSpecificationId);
    }
    if input.source_candidate_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptySourceCandidateId);
    }
    if input.left_model_id.trim().is_empty() || input.right_model_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyModelId);
    }
    if input.left_lineage.trim().is_empty() || input.right_lineage.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyLineage);
    }
    if input.outcome_id.trim().is_empty() {
        return Err(TemporalTestCandidateError::EmptyOutcomeId);
    }
    if !input.horizon_seconds.is_finite() || input.horizon_seconds <= 0.0 {
        return Err(TemporalTestCandidateError::InvalidHorizon);
    }
    if !input.estimated_cost.is_finite() || !input.pragmatic_risk.is_finite() {
        return Err(TemporalTestCandidateError::NonFinitePlanningMetadata);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_evidence_gap::TemporalOutcomeEvidenceMapping;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_core::scientific_multihorizon_divergence::ScientificTrajectorySnapshot;
    use symthaea_core::scientific_outcome_likelihood::{OutcomeLikelihoodModel, ScientificOutcomePrototype};
    use symthaea_core::scientific_temporal_outcome_discrimination::discover;
    use std::collections::BTreeSet;

    fn hv(seed: u64) -> ContinuousHV { ContinuousHV::random(HDC_DIMENSION, seed) }

    fn plan() -> TemporalEvidenceGapPlan {
        let snapshots = vec![
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 1.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 2.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 1.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 2.0, predicted_state: hv(2) },
        ];
        let outcomes = vec![ScientificOutcomePrototype { outcome_id: "o1".into(), prototype: hv(2) }];
        let discrimination = discover(&snapshots, &outcomes, &OutcomeLikelihoodModel::default()).unwrap();
        let mappings = vec![TemporalOutcomeEvidenceMapping {
            outcome_id: "o1".into(),
            advances_predicates: ["replication".into()].into_iter().collect(),
        }];
        let gaps: BTreeSet<String> = ["replication".into()].into_iter().collect();
        crate::temporal_evidence_gap::plan(&discrimination, &mappings, &gaps).unwrap()
    }

    fn source_id(horizon: f32) -> String {
        { let p = plan(); let candidate = p.assessments.iter().flat_map(|a| a.candidates.iter()).find(|c| c.horizon_seconds == horizon).unwrap(); source_candidate_id(candidate) }
    }

    fn input(horizon: f32) -> TemporalTestCandidateSpecInput {
        TemporalTestCandidateSpecInput {
            candidate_id: format!("test-{horizon}"),
            left_model_id: "a".into(),
            right_model_id: "b".into(),
            left_lineage: "la".into(),
            right_lineage: "lb".into(),
            outcome_id: "o1".into(),
            horizon_seconds: horizon,
            test_specification_id: format!("test-spec-{horizon}"),
            measurement_specification_id: format!("measurement-{horizon}"),
            estimated_cost: 2.0,
            pragmatic_risk: 0.5,
            source_candidate_id: source_id(horizon),
        }
    }

    #[test]
    fn binds_exact_temporal_candidate() {
        let result = materialize(&plan(), &[input(2.0)]).unwrap();
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].source_candidate_id, source_id(2.0));
    }

    #[test]
    fn rejects_missing_source() {
        let mut value = input(2.0);
        value.source_candidate_id = "missing".into();
        assert!(matches!(materialize(&plan(), &[value]), Err(TemporalTestCandidateError::UnknownSourceCandidate)));
    }

    #[test]
    fn rejects_temporal_binding_mismatch() {
        let mut value = input(2.0);
        value.horizon_seconds = 1.0;
        assert!(matches!(materialize(&plan(), &[value]), Err(TemporalTestCandidateError::SourceBindingMismatch)));
    }

    #[test]
    fn preserves_competing_alternatives_without_ranking() {
        let result = materialize(&plan(), &[input(2.0), input(1.0)]).unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].candidate_id, "test-1");
        assert_eq!(result.candidates[1].candidate_id, "test-2");
    }

    #[test]
    fn rejects_non_finite_planning_metadata() {
        let mut value = input(2.0);
        value.estimated_cost = f64::NAN;
        assert!(matches!(materialize(&plan(), &[value]), Err(TemporalTestCandidateError::NonFinitePlanningMetadata)));
    }

    #[test]
    fn specification_order_is_invariant() {
        let a = materialize(&plan(), &[input(1.0), input(2.0)]).unwrap();
        let b = materialize(&plan(), &[input(2.0), input(1.0)]).unwrap();
        assert_eq!(a, b);
    }
}
