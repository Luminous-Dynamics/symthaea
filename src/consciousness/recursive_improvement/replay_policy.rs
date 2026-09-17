// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded policy selection over historical replay worlds.
//!
//! This module does not rewrite Symthaea's model, evaluator, evidence rules, or
//! safety constraints. It only ranks already-constructed exploration policies
//! against replay scores and preserves the incumbent as an eligible candidate.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayPolicyScore {
    pub policy_id: String,
    pub replay_quality: f64,
    pub evaluation_cost: f64,
    pub parallelism_credit: f64,
}

impl ReplayPolicyScore {
    pub fn objective(&self, beta_cost: f64, beta_parallelism: f64) -> f64 {
        self.replay_quality - beta_cost * self.evaluation_cost
            + beta_parallelism * self.parallelism_credit
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicySelectionError {
    EmptyCandidates,
    IncumbentMissing(String),
    DuplicatePolicyId(String),
    InvalidObjectiveWeights,
    NonFiniteScore(String),
    NegativeEvaluationCost(String),
    NegativeParallelismCredit(String),
}

/// Select the best replay-scored policy while requiring the incumbent to remain
/// in the candidate set. This gives a narrow non-degradation guarantee only with
/// respect to the supplied replay objective and replay pool.
///
/// Exact ties preserve the incumbent. Candidate ordering therefore cannot create
/// policy churn without a strict objective improvement.
pub fn select_replay_policy<'a>(
    incumbent_id: &str,
    candidates: &'a [ReplayPolicyScore],
    beta_cost: f64,
    beta_parallelism: f64,
) -> Result<&'a ReplayPolicyScore, PolicySelectionError> {
    if candidates.is_empty() {
        return Err(PolicySelectionError::EmptyCandidates);
    }
    if !beta_cost.is_finite()
        || !beta_parallelism.is_finite()
        || beta_cost < 0.0
        || beta_parallelism < 0.0
    {
        return Err(PolicySelectionError::InvalidObjectiveWeights);
    }

    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.policy_id.as_str()) {
            return Err(PolicySelectionError::DuplicatePolicyId(
                candidate.policy_id.clone(),
            ));
        }
        if !candidate.replay_quality.is_finite()
            || !candidate.evaluation_cost.is_finite()
            || !candidate.parallelism_credit.is_finite()
        {
            return Err(PolicySelectionError::NonFiniteScore(
                candidate.policy_id.clone(),
            ));
        }
        if candidate.evaluation_cost < 0.0 {
            return Err(PolicySelectionError::NegativeEvaluationCost(
                candidate.policy_id.clone(),
            ));
        }
        if candidate.parallelism_credit < 0.0 {
            return Err(PolicySelectionError::NegativeParallelismCredit(
                candidate.policy_id.clone(),
            ));
        }
    }

    let incumbent = candidates
        .iter()
        .find(|candidate| candidate.policy_id == incumbent_id)
        .ok_or_else(|| PolicySelectionError::IncumbentMissing(incumbent_id.to_owned()))?;

    let mut best = incumbent;
    let mut best_objective = incumbent.objective(beta_cost, beta_parallelism);
    if !best_objective.is_finite() {
        return Err(PolicySelectionError::NonFiniteScore(
            incumbent.policy_id.clone(),
        ));
    }

    for candidate in candidates {
        if candidate.policy_id == incumbent_id {
            continue;
        }
        let objective = candidate.objective(beta_cost, beta_parallelism);
        if objective > best_objective {
            best = candidate;
            best_objective = objective;
        }
    }

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(id: &str, quality: f64, cost: f64) -> ReplayPolicyScore {
        ReplayPolicyScore {
            policy_id: id.into(),
            replay_quality: quality,
            evaluation_cost: cost,
            parallelism_credit: 0.0,
        }
    }

    #[test]
    fn incumbent_must_be_present() {
        let candidates = vec![score("new", 2.0, 1.0)];
        assert_eq!(
            select_replay_policy("old", &candidates, 0.1, 0.0),
            Err(PolicySelectionError::IncumbentMissing("old".into()))
        );
    }

    #[test]
    fn better_replay_objective_can_replace_incumbent() {
        let candidates = vec![score("old", 1.0, 1.0), score("new", 1.2, 1.0)];
        assert_eq!(
            select_replay_policy("old", &candidates, 0.1, 0.0)
                .unwrap()
                .policy_id,
            "new"
        );
    }

    #[test]
    fn exact_tie_preserves_incumbent_even_if_challenger_is_first() {
        let candidates = vec![score("new", 1.0, 1.0), score("old", 1.0, 1.0)];
        assert_eq!(
            select_replay_policy("old", &candidates, 0.1, 0.0)
                .unwrap()
                .policy_id,
            "old"
        );
    }

    #[test]
    fn duplicate_ids_and_negative_costs_are_rejected() {
        let duplicates = vec![score("old", 1.0, 1.0), score("old", 2.0, 1.0)];
        assert_eq!(
            select_replay_policy("old", &duplicates, 0.1, 0.0),
            Err(PolicySelectionError::DuplicatePolicyId("old".into()))
        );

        let negative = vec![score("old", 1.0, -1.0)];
        assert_eq!(
            select_replay_policy("old", &negative, 0.1, 0.0),
            Err(PolicySelectionError::NegativeEvaluationCost("old".into()))
        );
    }

    #[test]
    fn objective_weights_must_be_finite_and_non_negative() {
        let candidates = vec![score("old", 1.0, 1.0)];
        assert_eq!(
            select_replay_policy("old", &candidates, -0.1, 0.0),
            Err(PolicySelectionError::InvalidObjectiveWeights)
        );
        assert_eq!(
            select_replay_policy("old", &candidates, f64::NAN, 0.0),
            Err(PolicySelectionError::InvalidObjectiveWeights)
        );
    }
}
