// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact replay over immutable observed experience.
//!
//! This module is intentionally incapable of inventing unseen transitions. If a
//! requested action was not observed from the selected state, replay fails closed.

use super::experience_tree::{ExperienceNode, ExperienceNodeId, ExperienceTree};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    MissingState(ExperienceNodeId),
    UnsupportedAction {
        state: ExperienceNodeId,
        action_digest: String,
    },
}

/// Read-only replay facade over an [`ExperienceTree`].
pub struct ExactReplayWorld<'a> {
    tree: &'a ExperienceTree,
}

impl<'a> ExactReplayWorld<'a> {
    pub fn new(tree: &'a ExperienceTree) -> Self {
        Self { tree }
    }

    /// Follow a previously observed action edge from `state`.
    ///
    /// No learned model or generator is consulted. Missing actions are explicit
    /// failures rather than invitations to synthesize an outcome.
    pub fn step(
        &self,
        state: ExperienceNodeId,
        action_digest: &str,
    ) -> Result<&'a ExperienceNode, ReplayError> {
        if self.tree.node(state).is_none() {
            return Err(ReplayError::MissingState(state));
        }

        for child_id in self.tree.children_of(state) {
            if let Some(child) = self.tree.node(*child_id)
                && child.action_digest == action_digest
            {
                return Ok(child);
            }
        }

        Err(ReplayError::UnsupportedAction {
            state,
            action_digest: action_digest.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::experience_tree::{
        ExperienceNode, ExperienceProvenance,
    };

    fn node(id: u64, parent: Option<u64>, action: &str) -> ExperienceNode {
        ExperienceNode {
            id,
            parent,
            world_state_digest: format!("state-{id}"),
            action_digest: action.into(),
            observation_digest: format!("obs-{id}"),
            realized_outcome_digest: format!("outcome-{id}"),
            prediction_error: None,
            utility: vec![1.0],
            compute_cost: 1.0,
            uncertainty: Some(0.1),
            model_version: "test-v1".into(),
            evidence_digest: format!("evidence-{id}"),
            provenance: ExperienceProvenance::Observed,
        }
    }

    #[test]
    fn replays_only_observed_edges() {
        let mut tree = ExperienceTree::new();
        tree.append(node(1, None, "root")).unwrap();
        tree.append(node(2, Some(1), "observed-action")).unwrap();

        let replay = ExactReplayWorld::new(&tree);
        assert_eq!(replay.step(1, "observed-action").unwrap().id, 2);
        assert_eq!(
            replay.step(1, "unseen-action"),
            Err(ReplayError::UnsupportedAction {
                state: 1,
                action_digest: "unseen-action".into(),
            })
        );
    }
}
