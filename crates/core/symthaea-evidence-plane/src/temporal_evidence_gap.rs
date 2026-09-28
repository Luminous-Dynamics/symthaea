// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal evidence-gap accounting for horizon-aligned computational discrimination.
//!
//! Preserves the time dimension of 002M. It identifies which explicit unmet
//! predicates are potentially advanced by a model-pair/outcome candidate and
//! records the horizons at which computational separation was observed.
//! It never turns these predictions into evidence.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_core::scientific_temporal_outcome_discrimination::{
    TemporalOutcomeDiscrimination, TemporalOutcomeDiscriminationSet,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalOutcomeEvidenceMapping {
    pub outcome_id: String,
    pub advances_predicates: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalEvidenceGapAssessment {
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub outcome_id: String,
    pub advances_unmet_predicates: BTreeSet<String>,
    pub horizons_seconds: Vec<f32>,
    pub first_horizon_seconds: f32,
    pub strongest_horizon_seconds: f32,
    pub strongest_likelihood_separation: f64,
    pub strongest_trajectory_divergence: f32,
    pub candidates: Vec<TemporalOutcomeDiscrimination>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalEvidenceGapPlan {
    pub assessments: Vec<TemporalEvidenceGapAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalEvidenceGapError {
    EmptyUnmetPredicates,
    EmptyMappingOutcomeId,
    DuplicateMappingOutcomeId,
    EmptyPredicate,
    UnknownOutcome(String),
    NonFiniteDiscrimination(String),
}

impl fmt::Display for TemporalEvidenceGapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for TemporalEvidenceGapError {}

pub fn plan(
    discrimination: &TemporalOutcomeDiscriminationSet,
    mappings: &[TemporalOutcomeEvidenceMapping],
    unmet_predicates: &BTreeSet<String>,
) -> Result<TemporalEvidenceGapPlan, TemporalEvidenceGapError> {
    if unmet_predicates.is_empty() {
        return Err(TemporalEvidenceGapError::EmptyUnmetPredicates);
    }

    let known_outcomes: BTreeSet<&str> =
        discrimination.candidates.iter().map(|c| c.outcome_id.as_str()).collect();
    let mut by_outcome: BTreeMap<&str, &TemporalOutcomeEvidenceMapping> = BTreeMap::new();

    for mapping in mappings {
        if mapping.outcome_id.trim().is_empty() {
            return Err(TemporalEvidenceGapError::EmptyMappingOutcomeId);
        }
        if !known_outcomes.contains(mapping.outcome_id.as_str()) {
            return Err(TemporalEvidenceGapError::UnknownOutcome(mapping.outcome_id.clone()));
        }
        if by_outcome.insert(mapping.outcome_id.as_str(), mapping).is_some() {
            return Err(TemporalEvidenceGapError::DuplicateMappingOutcomeId);
        }
        if mapping.advances_predicates.iter().any(|p| p.trim().is_empty()) {
            return Err(TemporalEvidenceGapError::EmptyPredicate);
        }
    }

    let mut groups: BTreeMap<(String, String, String), Vec<TemporalOutcomeDiscrimination>> =
        BTreeMap::new();

    for candidate in &discrimination.candidates {
        if !candidate.likelihood_separation.is_finite()
            || !candidate.trajectory_divergence.is_finite()
            || !candidate.horizon_seconds.is_finite()
        {
            return Err(TemporalEvidenceGapError::NonFiniteDiscrimination(
                candidate.outcome_id.clone(),
            ));
        }

        let Some(mapping) = by_outcome.get(candidate.outcome_id.as_str()) else {
            return Err(TemporalEvidenceGapError::UnknownOutcome(candidate.outcome_id.clone()));
        };
        if mapping.advances_predicates.intersection(unmet_predicates).next().is_none() {
            continue;
        }

        groups.entry((
            candidate.left_model_id.clone(),
            candidate.right_model_id.clone(),
            candidate.outcome_id.clone(),
        )).or_default().push(candidate.clone());
    }

    let mut assessments = Vec::new();
    for ((left_model_id, right_model_id, outcome_id), mut candidates) in groups {
        candidates.sort_by(|a, b| {
            a.horizon_seconds.total_cmp(&b.horizon_seconds)
                .then_with(|| b.likelihood_separation.total_cmp(&a.likelihood_separation))
                .then_with(|| b.trajectory_divergence.total_cmp(&a.trajectory_divergence))
        });

        let mapping = by_outcome.get(outcome_id.as_str()).expect("validated mapping");
        let covered: BTreeSet<String> = mapping.advances_predicates
            .intersection(unmet_predicates).cloned().collect();

        let first_horizon_seconds = candidates.first().expect("group cannot be empty").horizon_seconds;
        let strongest = candidates.iter().max_by(|a, b| {
            a.likelihood_separation.total_cmp(&b.likelihood_separation)
                .then_with(|| a.trajectory_divergence.total_cmp(&b.trajectory_divergence))
                .then_with(|| b.horizon_seconds.total_cmp(&a.horizon_seconds))
        }).expect("group cannot be empty");

        let horizons = candidates.iter().map(|c| c.horizon_seconds).collect();

        assessments.push(TemporalEvidenceGapAssessment {
            left_model_id,
            right_model_id,
            left_lineage: candidates[0].left_lineage.clone(),
            right_lineage: candidates[0].right_lineage.clone(),
            outcome_id,
            advances_unmet_predicates: covered,
            horizons_seconds: horizons,
            first_horizon_seconds,
            strongest_horizon_seconds: strongest.horizon_seconds,
            strongest_likelihood_separation: strongest.likelihood_separation,
            strongest_trajectory_divergence: strongest.trajectory_divergence,
            candidates,
        });
    }

    assessments.sort_by(|a, b| {
        a.first_horizon_seconds.total_cmp(&b.first_horizon_seconds)
            .then_with(|| a.outcome_id.cmp(&b.outcome_id))
            .then_with(|| a.left_model_id.cmp(&b.left_model_id))
            .then_with(|| a.right_model_id.cmp(&b.right_model_id))
    });

    Ok(TemporalEvidenceGapPlan { assessments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_core::scientific_multihorizon_divergence::ScientificTrajectorySnapshot;
    use symthaea_core::scientific_outcome_likelihood::{OutcomeLikelihoodModel, ScientificOutcomePrototype};
    use symthaea_core::scientific_temporal_outcome_discrimination::discover;

    fn hv(seed: u64) -> ContinuousHV { ContinuousHV::random(HDC_DIMENSION, seed) }

    fn discrimination() -> TemporalOutcomeDiscriminationSet {
        let snapshots = vec![
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 1.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 2.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 1.0, predicted_state: hv(1) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 2.0, predicted_state: hv(2) },
        ];
        let outcomes = vec![
            ScientificOutcomePrototype { outcome_id: "o1".into(), prototype: hv(1) },
            ScientificOutcomePrototype { outcome_id: "o2".into(), prototype: hv(2) },
        ];
        discover(&snapshots, &outcomes, &OutcomeLikelihoodModel::default()).unwrap()
    }

    fn mapping(id: &str, predicates: &[&str]) -> TemporalOutcomeEvidenceMapping {
        TemporalOutcomeEvidenceMapping {
            outcome_id: id.into(),
            advances_predicates: predicates.iter().map(|p| (*p).into()).collect(),
        }
    }

    fn all_mappings() -> Vec<TemporalOutcomeEvidenceMapping> {
        vec![mapping("o1", &["provenance"]), mapping("o2", &["replication"])]
    }

    #[test]
    fn preserves_all_qualifying_horizons() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        let result = plan(&discrimination(), &all_mappings(), &gaps).unwrap();
        let assessment = &result.assessments[0];
        assert_eq!(assessment.horizons_seconds, vec![1.0, 2.0]);
        assert_eq!(assessment.first_horizon_seconds, 1.0);
        assert_eq!(assessment.candidates.len(), 2);
    }

    #[test]
    fn strongest_horizon_is_descriptive_not_a_winner() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        let result = plan(&discrimination(), &all_mappings(), &gaps).unwrap();
        let assessment = &result.assessments[0];
        assert!(assessment.horizons_seconds.contains(&assessment.strongest_horizon_seconds));
        assert_eq!(assessment.candidates.len(), 2);
    }

    #[test]
    fn filters_outcomes_that_advance_no_unmet_predicate() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        let result = plan(&discrimination(), &all_mappings(), &gaps).unwrap();
        assert!(result.assessments.iter().all(|a| a.outcome_id == "o2"));
    }

    #[test]
    fn rejects_unknown_mapping() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        assert!(matches!(
            plan(&discrimination(), &[mapping("unknown", &["replication"])], &gaps),
            Err(TemporalEvidenceGapError::UnknownOutcome(_))
        ));
    }

    #[test]
    fn rejects_duplicate_mapping() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        let mut m = all_mappings();
        m.push(mapping("o2", &["replication"]));
        assert!(matches!(
            plan(&discrimination(), &m, &gaps),
            Err(TemporalEvidenceGapError::DuplicateMappingOutcomeId)
        ));
    }

    #[test]
    fn mapping_order_is_invariant() {
        let gaps = ["replication"].into_iter().map(String::from).collect();
        let a = plan(&discrimination(), &all_mappings(), &gaps).unwrap();
        let mut reversed = all_mappings();
        reversed.reverse();
        let b = plan(&discrimination(), &reversed, &gaps).unwrap();
        assert_eq!(a, b);
    }
}
