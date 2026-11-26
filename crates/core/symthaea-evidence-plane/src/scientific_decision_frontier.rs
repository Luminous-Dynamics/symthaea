// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-objective decision frontier for scientific discovery planning.
//!
//! The frontier preserves trade-offs instead of collapsing heterogeneous
//! planning objectives into a scalar score. It is decision support only.

use std::fmt;
use serde::{Deserialize, Serialize};

/// Explicit dimensions for one already-declared scientific planning candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificDecisionCandidate {
    pub candidate_id: String,
    pub gap_coverage: f64,
    pub likelihood_separation: f64,
    pub trajectory_divergence: f64,
    pub eig_per_cost: f64,
    pub pragmatic_risk: f64,
    pub left_lineage: String,
    pub right_lineage: String,
}

/// Candidate plus the other candidates that strictly dominate it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoAssessment {
    pub candidate: ScientificDecisionCandidate,
    pub is_frontier: bool,
    pub dominated_by: Vec<String>,
}

/// Complete deterministic Pareto analysis. No scalar winner is selected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificDecisionFrontier {
    pub assessments: Vec<ParetoAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParetoFrontierError {
    EmptyCandidates,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    EmptyLineage,
    NonFiniteDimension(String),
}
impl fmt::Display for ParetoFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ParetoFrontierError {}

impl ScientificDecisionCandidate {
    fn validate(&self) -> Result<(), ParetoFrontierError> {
        if self.candidate_id.trim().is_empty() {
            return Err(ParetoFrontierError::EmptyCandidateId);
        }
        if self.left_lineage.trim().is_empty() || self.right_lineage.trim().is_empty() {
            return Err(ParetoFrontierError::EmptyLineage);
        }
        for (name, value) in [
            ("gap_coverage", self.gap_coverage),
            ("likelihood_separation", self.likelihood_separation),
            ("trajectory_divergence", self.trajectory_divergence),
            ("eig_per_cost", self.eig_per_cost),
            ("pragmatic_risk", self.pragmatic_risk),
        ] {
            if !value.is_finite() {
                return Err(ParetoFrontierError::NonFiniteDimension(
                    format!("{}:{}", self.candidate_id, name)
                ));
            }
        }
        Ok(())
    }

    /// Pareto dominance: at least as good on every dimension and strictly
    /// better on at least one. The first four dimensions are maximized;
    /// pragmatic risk is minimized.
    pub fn dominates(&self, other: &Self) -> bool {
        let ge_all =
            self.gap_coverage >= other.gap_coverage
            && self.likelihood_separation >= other.likelihood_separation
            && self.trajectory_divergence >= other.trajectory_divergence
            && self.eig_per_cost >= other.eig_per_cost
            && self.pragmatic_risk <= other.pragmatic_risk;
        let strict =
            self.gap_coverage > other.gap_coverage
            || self.likelihood_separation > other.likelihood_separation
            || self.trajectory_divergence > other.trajectory_divergence
            || self.eig_per_cost > other.eig_per_cost
            || self.pragmatic_risk < other.pragmatic_risk;
        ge_all && strict
    }
}

/// Compute the non-dominated set while retaining domination provenance for
/// every candidate. Shared model ancestry is metadata, not independence.
pub fn frontier(
    candidates: &[ScientificDecisionCandidate],
) -> Result<ScientificDecisionFrontier, ParetoFrontierError> {
    if candidates.is_empty() {
        return Err(ParetoFrontierError::EmptyCandidates);
    }
    for candidate in candidates {
        candidate.validate()?;
    }

    let mut seen = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(ParetoFrontierError::DuplicateCandidateId(
                candidate.candidate_id.clone()
            ));
        }
    }

    let mut assessments = candidates.iter().map(|candidate| {
        let mut dominated_by: Vec<String> = candidates.iter()
            .filter(|other| other.candidate_id != candidate.candidate_id && other.dominates(candidate))
            .map(|other| other.candidate_id.clone())
            .collect();
        dominated_by.sort();
        ParetoAssessment {
            candidate: candidate.clone(),
            is_frontier: dominated_by.is_empty(),
            dominated_by,
        }
    }).collect::<Vec<_>>();

    // Frontier membership is a set property, not a preference ordering.
    // Stable IDs only make exported artifacts reproducible.
    assessments.sort_by(|a, b| {
        b.is_frontier.cmp(&a.is_frontier)
            .then_with(|| a.candidate.candidate_id.cmp(&b.candidate.candidate_id))
    });

    Ok(ScientificDecisionFrontier { assessments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, gap: f64, separation: f64, divergence: f64, eig: f64, risk: f64) -> ScientificDecisionCandidate {
        ScientificDecisionCandidate {
            candidate_id: id.into(),
            gap_coverage: gap,
            likelihood_separation: separation,
            trajectory_divergence: divergence,
            eig_per_cost: eig,
            pragmatic_risk: risk,
            left_lineage: "left".into(),
            right_lineage: "right".into(),
        }
    }

    #[test]
    fn dominated_candidate_is_excluded_from_frontier_membership() {
        let result = frontier(&[
            candidate("a", 1.0, 1.0, 1.0, 1.0, 1.0),
            candidate("b", 0.5, 1.0, 1.0, 1.0, 1.0),
        ]).unwrap();
        assert!(result.assessments.iter().find(|a| a.candidate.candidate_id == "a").unwrap().is_frontier);
        let b = result.assessments.iter().find(|a| a.candidate.candidate_id == "b").unwrap();
        assert!(!b.is_frontier);
        assert_eq!(b.dominated_by, vec!["a"]);
    }

    #[test]
    fn incomparable_tradeoffs_remain_on_frontier() {
        let result = frontier(&[
            candidate("gap-heavy", 1.0, 0.2, 0.2, 0.2, 1.0),
            candidate("eig-heavy", 0.2, 1.0, 1.0, 1.0, 1.0),
        ]).unwrap();
        assert!(result.assessments.iter().all(|a| a.is_frontier));
    }

    #[test]
    fn equal_candidates_are_both_frontier_and_stably_ordered() {
        let result = frontier(&[
            candidate("b", 1.0, 1.0, 1.0, 1.0, 1.0),
            candidate("a", 1.0, 1.0, 1.0, 1.0, 1.0),
        ]).unwrap();
        assert_eq!(result.assessments.iter().map(|a| a.candidate.candidate_id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert!(result.assessments.iter().all(|a| a.dominated_by.is_empty()));
    }

    #[test]
    fn nonfinite_dimensions_are_rejected() {
        let mut a = candidate("a", 1.0, 1.0, 1.0, 1.0, 1.0);
        a.eig_per_cost = f64::NAN;
        assert!(matches!(frontier(&[a]), Err(ParetoFrontierError::NonFiniteDimension(_))));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        assert!(matches!(
            frontier(&[candidate("a",1.0,1.0,1.0,1.0,1.0), candidate("a",0.9,0.9,0.9,0.9,0.9)]),
            Err(ParetoFrontierError::DuplicateCandidateId(_))
        ));
    }

    #[test]
    fn shared_lineage_does_not_create_independence() {
        let mut a = candidate("a", 1.0, 0.8, 0.7, 0.6, 1.0);
        let mut b = candidate("b", 0.9, 0.7, 0.8, 0.5, 1.0);
        b.left_lineage = a.left_lineage.clone();
        b.right_lineage = a.right_lineage.clone();
        let result = frontier(&[a, b]).unwrap();
        assert!(result.assessments.iter().all(|x| x.is_frontier));
        assert_eq!(result.assessments[0].candidate.left_lineage, result.assessments[1].candidate.left_lineage);
    }
}
