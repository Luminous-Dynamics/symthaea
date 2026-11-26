// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit FEP-side admissibility constraints for the scientific Pareto frontier.
//!
//! This module filters an already-computed non-dominated set. It deliberately
//! does not collapse the frontier into a scalar score or select a winner.

use serde::{Deserialize, Serialize};
use std::fmt;
use symthaea_evidence_plane::scientific_decision_frontier::{ParetoAssessment, ScientificDecisionCandidate, ScientificDecisionFrontier};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScientificAdmissibilityPolicy {
    pub max_pragmatic_risk: Option<f64>,
    pub min_gap_coverage: Option<f64>,
    pub min_eig_per_cost: Option<f64>,
    pub min_likelihood_separation: Option<f64>,
    pub min_trajectory_divergence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificAdmissibilityResult {
    pub admissible: Vec<ParetoAssessment>,
    pub rejected_candidate_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificAdmissibilityError {
    InvalidThreshold(&'static str),
    NonFrontierCandidate(String),
}
impl fmt::Display for ScientificAdmissibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ScientificAdmissibilityError {}

impl Default for ScientificAdmissibilityPolicy {
    fn default() -> Self {
        Self {
            max_pragmatic_risk: None,
            min_gap_coverage: None,
            min_eig_per_cost: None,
            min_likelihood_separation: None,
            min_trajectory_divergence: None,
        }
    }
}

impl ScientificAdmissibilityPolicy {
    pub fn validate(&self) -> Result<(), ScientificAdmissibilityError> {
        if self.max_pragmatic_risk.is_some_and(|v| !v.is_finite()) { return Err(ScientificAdmissibilityError::InvalidThreshold("max_pragmatic_risk")); }
        if self.min_gap_coverage.is_some_and(|v| !v.is_finite()) { return Err(ScientificAdmissibilityError::InvalidThreshold("min_gap_coverage")); }
        if self.min_eig_per_cost.is_some_and(|v| !v.is_finite()) { return Err(ScientificAdmissibilityError::InvalidThreshold("min_eig_per_cost")); }
        if self.min_likelihood_separation.is_some_and(|v| !v.is_finite()) { return Err(ScientificAdmissibilityError::InvalidThreshold("min_likelihood_separation")); }
        if self.min_trajectory_divergence.is_some_and(|v| !v.is_finite()) { return Err(ScientificAdmissibilityError::InvalidThreshold("min_trajectory_divergence")); }
        Ok(())
    }

    fn accepts(&self, c: &ScientificDecisionCandidate) -> bool {
        self.max_pragmatic_risk.is_none_or(|v| c.pragmatic_risk <= v)
            && self.min_gap_coverage.is_none_or(|v| c.gap_coverage >= v)
            && self.min_eig_per_cost.is_none_or(|v| c.eig_per_cost >= v)
            && self.min_likelihood_separation.is_none_or(|v| c.likelihood_separation >= v)
            && self.min_trajectory_divergence.is_none_or(|v| c.trajectory_divergence >= v)
    }
}

/// Apply explicit planner constraints to the 002I frontier.
///
/// Only frontier members may enter the admissible set. Rejected members are
/// retained as IDs for auditability. Stable ID ordering is deterministic but
/// has no preference semantics.
pub fn admit(
    frontier: &ScientificDecisionFrontier,
    policy: &ScientificAdmissibilityPolicy,
) -> Result<ScientificAdmissibilityResult, ScientificAdmissibilityError> {
    policy.validate()?;
    let mut admissible = Vec::new();
    let mut rejected = Vec::new();
    for assessment in &frontier.assessments {
        if !assessment.is_frontier {
            return Err(ScientificAdmissibilityError::NonFrontierCandidate(
                assessment.candidate.candidate_id.clone()
            ));
        }
        if policy.accepts(&assessment.candidate) {
            admissible.push(assessment.clone());
        } else {
            rejected.push(assessment.candidate.candidate_id.clone());
        }
    }
    admissible.sort_by(|a,b| a.candidate.candidate_id.cmp(&b.candidate.candidate_id));
    rejected.sort();
    Ok(ScientificAdmissibilityResult { admissible, rejected_candidate_ids: rejected })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_plane::scientific_decision_frontier::frontier;

    fn c(id:&str,g:f64,e:f64,r:f64)->ScientificDecisionCandidate {
        ScientificDecisionCandidate {
            candidate_id:id.into(), gap_coverage:g, likelihood_separation:e,
            trajectory_divergence:e, eig_per_cost:e, pragmatic_risk:r,
            left_lineage:"l".into(), right_lineage:"r".into()
        }
    }

    #[test]
    fn risk_and_information_constraints_filter_without_ranking() {
        let f=frontier(&[c("a",1.0,0.8,1.0),c("b",0.8,1.0,0.5)]).unwrap();
        let p=ScientificAdmissibilityPolicy {
            max_pragmatic_risk:Some(0.75), min_eig_per_cost:Some(0.7),
            ..Default::default()
        };
        let r=admit(&f,&p).unwrap();
        assert_eq!(r.admissible.len(),1);
        assert_eq!(r.admissible[0].candidate.candidate_id,"b");
        assert_eq!(r.rejected_candidate_ids,vec!["a"]);
    }

    #[test]
    fn incomparable_frontier_members_can_both_survive() {
        let f=frontier(&[c("a",1.0,0.8,0.5),c("b",0.8,1.0,0.5)]).unwrap();
        let r=admit(&f,&ScientificAdmissibilityPolicy::default()).unwrap();
        assert_eq!(r.admissible.iter().map(|x|x.candidate.candidate_id.as_str()).collect::<Vec<_>>(),vec!["a","b"]);
    }

    #[test]
    fn nonfinite_threshold_rejected() {
        let p=ScientificAdmissibilityPolicy { min_eig_per_cost:Some(f64::NAN), ..Default::default() };
        let f=frontier(&[c("a",1.0,1.0,1.0)]).unwrap();
        assert!(matches!(admit(&f,&p),Err(ScientificAdmissibilityError::InvalidThreshold("min_eig_per_cost"))));
    }

    #[test]
    fn non_frontier_input_is_rejected() {
        let f=frontier(&[c("a",1.0,1.0,1.0),c("b",0.5,0.5,0.5)]).unwrap();
        assert!(matches!(admit(&f,&ScientificAdmissibilityPolicy::default()),Err(ScientificAdmissibilityError::NonFrontierCandidate("b"))));
    }
}
