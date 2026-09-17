// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded policy selection over historical replay worlds.
//!
//! This module does not rewrite Symthaea's model, evaluator, evidence rules, or
//! safety constraints. It only ranks already-constructed exploration policies
//! against replay scores and preserves the incumbent as an eligible candidate.

use serde::{Deserialize, Serialize};

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
    NonFiniteScore(String),
}

/// Select the best replay-scored policy while requiring the incumbent to remain
/// in the candidate set. This gives a narrow non-degradation guarantee only with
/// respect to the supplied replay objective and replay pool.
pub fn select_replay_policy<'a>(
    incumbent_id: &str,
    candidates: &'a [ReplayPolicyScore],
    beta_cost: f64,
    beta_parallelism: f64,
) -> Result<&'a ReplayPolicyScore, PolicySelectionError> {
    if candidates.is_empty() {
        return Err(PolicySelectionError::EmptyCandidates);
    }
    if !candidates.iter().any(|c| c.policy_id == incumbent_id) {
        return Err(PolicySelectionError::IncumbentMissing(
            incumbent_id.to_owned(),
        ));
    }

    let mut best: Option<(&ReplayPolicyScore, f64)> = None;
    for candidate in candidates {
        let objective = candidate.objective(beta_cost, beta_parallelism);
        if !objective.is_finite() {
            return Err(PolicySelectionError::NonFiniteScore(
                candidate.policy_id.clone(),
            ));
        }
        if best.map(|(_, score)| objective > score).unwrap_or(true) {
            best = Some((candidate, objective));
        }
    }

    Ok(best.expect("non-empty candidates checked above").0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incumbent_must_be_present() {
        let candidates = vec![ReplayPolicyScore {
            policy_id: "new".into(),
            replay_quality: 2.0,
            evaluation_cost: 1.0,
            parallelism_credit: 0.0,
        }];
        assert_eq!(
            select_replay_policy("old", &candidates, 0.1, 0.0),
            Err(PolicySelectionError::IncumbentMissing("old".into()))
        );
    }

    #[test]
    fn better_replay_objective_can_replace_incumbent() {
        let candidates = vec![
            ReplayPolicyScore {
                policy_id: "old".into(),
                replay_quality: 1.0,
                evaluation_cost: 1.0,
                parallelism_credit: 0.0,
            },
            ReplayPolicyScore {
                policy_id: "new".into(),
                replay_quality: 1.2,
                evaluation_cost: 1.0,
                parallelism_credit: 0.0,
            },
        ];
        assert_eq!(
            select_replay_policy("old", &candidates, 0.1, 0.0)
                .unwrap()
                .policy_id,
            "new"
        );
    }
}
