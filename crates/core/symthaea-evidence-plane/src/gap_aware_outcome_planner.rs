// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-gap-aware planning over computational outcome discrimination.
//!
//! This is the boundary between Symthaea's computational discrimination layer
//! and the evidence plane's explicit unmet-predicate accounting. It selects
//! already-declared candidates; it never creates observations or evidence.

use std::collections::BTreeSet;
use std::fmt;
use serde::{Deserialize, Serialize};
use symthaea_core::scientific_trajectory_outcome_discrimination::{DiscriminatingOutcomeSet, TrajectoryOutcomeDiscrimination};

/// An explicit evidence predicate that an outcome candidate may advance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeEvidenceMapping {
    pub outcome_id: String,
    pub advances_predicates: BTreeSet<String>,
}

/// Gap-aware assessment for one model-pair/outcome candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapAwareOutcomeAssessment {
    pub candidate: TrajectoryOutcomeDiscrimination,
    pub advances_unmet_predicates: BTreeSet<String>,
    pub gap_coverage: f64,
    pub eligible: bool,
}

/// Deterministically ranked candidates after evidence-gap filtering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapAwareOutcomePlan {
    pub assessments: Vec<GapAwareOutcomeAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapAwareOutcomePlannerError {
    EmptyUnmetPredicates,
    EmptyMappingOutcomeId,
    DuplicateMappingOutcomeId,
    EmptyPredicate(String),
    UnknownOutcome(String),
    NonFiniteLikelihoodSeparation(String),
}
impl fmt::Display for GapAwareOutcomePlannerError { fn fmt(&self, f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"{self:?}") } }
impl std::error::Error for GapAwareOutcomePlannerError {}

/// Rank explicit outcome candidates by unmet evidence-predicate coverage first,
/// then model-implied discrimination. Candidates advancing no unmet predicate
/// are excluded; no evidence or observation is emitted.
pub fn plan(
    discrimination: &DiscriminatingOutcomeSet,
    mappings: &[OutcomeEvidenceMapping],
    unmet_predicates: &BTreeSet<String>,
) -> Result<GapAwareOutcomePlan, GapAwareOutcomePlannerError> {
    if unmet_predicates.is_empty() { return Err(GapAwareOutcomePlannerError::EmptyUnmetPredicates); }
    let mut by_outcome = std::collections::BTreeMap::new();
    for mapping in mappings {
        if mapping.outcome_id.trim().is_empty() { return Err(GapAwareOutcomePlannerError::EmptyMappingOutcomeId); }
        if by_outcome.insert(mapping.outcome_id.as_str(), mapping).is_some() {
            return Err(GapAwareOutcomePlannerError::DuplicateMappingOutcomeId);
        }
        if let Some(p) = mapping.advances_predicates.iter().find(|p| p.trim().is_empty()) {
            return Err(GapAwareOutcomePlannerError::EmptyPredicate(p.clone()));
        }
    }
    let known_outcomes: BTreeSet<&str> =
        discrimination.candidates.iter().map(|c| c.outcome_id.as_str()).collect();
    if let Some(unknown) = mapping_by_outcome
        .keys()
        .find(|id| !known_outcomes.contains(**id))
    {
        return Err(GapAwareOutcomePlannerError::UnknownOutcome((*unknown).to_string()));
    }

    let mut assessments = Vec::new();
    for candidate in &discrimination.candidates {
        let Some(mapping) = by_outcome.get(candidate.outcome_id.as_str()) else {
            return Err(GapAwareOutcomePlannerError::UnknownOutcome(candidate.outcome_id.clone()));
        };
        if !candidate.likelihood_separation.is_finite() {
            return Err(GapAwareOutcomePlannerError::NonFiniteLikelihoodSeparation(candidate.outcome_id.clone()));
        }
        let covered: BTreeSet<String> = mapping.advances_predicates.intersection(unmet_predicates).cloned().collect();
        if covered.is_empty() { continue; }
        assessments.push(GapAwareOutcomeAssessment {
            candidate: candidate.clone(),
            gap_coverage: covered.len() as f64 / unmet_predicates.len() as f64,
            advances_unmet_predicates: covered,
            eligible: true,
        });
    }
    assessments.sort_by(|a,b|
        b.gap_coverage.total_cmp(&a.gap_coverage)
            .then_with(|| b.candidate.likelihood_separation.total_cmp(&a.candidate.likelihood_separation))
            .then_with(|| b.candidate.trajectory_divergence.total_cmp(&a.candidate.trajectory_divergence))
            .then_with(|| a.candidate.outcome_id.cmp(&b.candidate.outcome_id))
            .then_with(|| a.candidate.left_model_id.cmp(&b.candidate.left_model_id))
            .then_with(|| a.candidate.right_model_id.cmp(&b.candidate.right_model_id))
    );
    Ok(GapAwareOutcomePlan { assessments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::{ContinuousHV,HDC_DIMENSION};
    use symthaea_core::scientific_active_inference::ScientificTrajectoryForecast;
    use symthaea_core::scientific_outcome_likelihood::{OutcomeLikelihoodModel,ScientificOutcomePrototype};
    use symthaea_core::scientific_trajectory_outcome_discrimination::discover;
    fn hv(seed:u64)->ContinuousHV{ContinuousHV::random(HDC_DIMENSION,seed)}
    fn discrimination()->DiscriminatingOutcomeSet {
        let forecasts=vec![
            ScientificTrajectoryForecast{model_id:"a".into(),lineage:"la".into(),prior:0.5,horizon_seconds:10.0,predicted_state:hv(1),outcome_probabilities:vec![]},
            ScientificTrajectoryForecast{model_id:"b".into(),lineage:"lb".into(),prior:0.5,horizon_seconds:10.0,predicted_state:hv(2),outcome_probabilities:vec![]},
        ];
        let outcomes=vec![
            ScientificOutcomePrototype{outcome_id:"o1".into(),prototype:hv(1)},
            ScientificOutcomePrototype{outcome_id:"o2".into(),prototype:hv(2)},
        ];
        discover(&forecasts,&outcomes,&OutcomeLikelihoodModel::default()).unwrap()
    }
    fn map(id:&str, ps:&[&str])->OutcomeEvidenceMapping{
        OutcomeEvidenceMapping{outcome_id:id.into(),advances_predicates:ps.iter().map(|p|(*p).into()).collect()}
    }
    #[test] fn prioritizes_gap_coverage(){let gaps=["replication","provenance"].into_iter().map(String::from).collect();let r=plan(&discrimination(),&[map("o1",&["replication"]),map("o2",&["replication","provenance"])],&gaps).unwrap();assert_eq!(r.assessments[0].candidate.outcome_id,"o2");assert_eq!(r.assessments[0].gap_coverage,1.0);}
    #[test] fn drops_non_advancing_candidates(){let gaps=["replication"].into_iter().map(String::from).collect();let r=plan(&discrimination(),&[map("o1",&["provenance"]),map("o2",&["replication"])],&gaps).unwrap();assert_eq!(r.assessments.len(),1);}
    #[test] fn rejects_duplicate_mappings(){let gaps=["replication"].into_iter().map(String::from).collect();assert!(matches!(plan(&discrimination(),&[map("o1",&["replication"]),map("o1",&["replication"])],&gaps),Err(GapAwareOutcomePlannerError::DuplicateMappingOutcomeId)));}
    #[test] fn rejects_unknown_outcome(){let gaps=["replication"].into_iter().map(String::from).collect();assert!(matches!(plan(&discrimination(),&[map("o1",&["replication"]),map("o2",&["replication"]),map("unknown",&["replication"])],&gaps),Err(GapAwareOutcomePlannerError::UnknownOutcome(_))));}
    #[test] fn empty_predicates_are_rejected(){let gaps=["replication"].into_iter().map(String::from).collect();let mut m=map("o1",&[]);m.advances_predicates.insert("".into());assert!(matches!(plan(&discrimination(),&[m,map("o2",&["replication"])],&gaps),Err(GapAwareOutcomePlannerError::EmptyPredicate(_))));}
}
