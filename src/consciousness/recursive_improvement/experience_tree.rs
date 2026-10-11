// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable causal experience trees for replay-grounded recursive improvement.
//!
//! An experience tree records only observed transitions. It is deliberately not a
//! predictive model: missing edges stay missing so exact replay cannot silently
//! manufacture evidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type ExperienceNodeId = u64;

/// Provenance class for a recorded transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperienceProvenance {
    /// Directly observed from an executed interaction.
    Observed,
    /// Imported from an externally verified evidence capsule.
    VerifiedImport,
}

/// One immutable observed transition in an experience tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperienceNode {
    pub id: ExperienceNodeId,
    pub parent: Option<ExperienceNodeId>,
    pub world_state_digest: String,
    pub action_digest: String,
    pub observation_digest: String,
    pub realized_outcome_digest: String,
    pub prediction_error: Option<f64>,
    pub utility: Vec<f64>,
    pub compute_cost: f64,
    pub uncertainty: Option<f64>,
    pub model_version: String,
    pub evidence_digest: String,
    pub provenance: ExperienceProvenance,
}

impl ExperienceNode {
    pub fn validate(&self) -> Result<(), ExperienceTreeError> {
        if self.world_state_digest.is_empty()
            || self.action_digest.is_empty()
            || self.observation_digest.is_empty()
            || self.realized_outcome_digest.is_empty()
            || self.model_version.is_empty()
            || self.evidence_digest.is_empty()
        {
            return Err(ExperienceTreeError::MissingRequiredField(self.id));
        }
        if !self.compute_cost.is_finite() || self.compute_cost < 0.0 {
            return Err(ExperienceTreeError::InvalidComputeCost(self.id));
        }
        if let Some(u) = self.uncertainty
            && (!u.is_finite() || !(0.0..=1.0).contains(&u))
        {
            return Err(ExperienceTreeError::InvalidUncertainty(self.id));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperienceTreeError {
    DuplicateNode(ExperienceNodeId),
    MissingParent(ExperienceNodeId),
    MissingRequiredField(ExperienceNodeId),
    InvalidComputeCost(ExperienceNodeId),
    InvalidUncertainty(ExperienceNodeId),
}

/// Append-only tree of observed experience.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperienceTree {
    nodes: BTreeMap<ExperienceNodeId, ExperienceNode>,
    children: BTreeMap<ExperienceNodeId, Vec<ExperienceNodeId>>,
}

impl ExperienceTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an observed node. Existing nodes are never overwritten.
    pub fn append(&mut self, node: ExperienceNode) -> Result<(), ExperienceTreeError> {
        node.validate()?;
        if self.nodes.contains_key(&node.id) {
            return Err(ExperienceTreeError::DuplicateNode(node.id));
        }
        if let Some(parent) = node.parent
            && !self.nodes.contains_key(&parent)
        {
            return Err(ExperienceTreeError::MissingParent(parent));
        }

        if let Some(parent) = node.parent {
            self.children.entry(parent).or_default().push(node.id);
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    pub fn node(&self, id: ExperienceNodeId) -> Option<&ExperienceNode> {
        self.nodes.get(&id)
    }

    pub fn children_of(&self, id: ExperienceNodeId) -> &[ExperienceNodeId] {
        self.children.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, parent: Option<u64>) -> ExperienceNode {
        ExperienceNode {
            id,
            parent,
            world_state_digest: format!("state-{id}"),
            action_digest: format!("action-{id}"),
            observation_digest: format!("obs-{id}"),
            realized_outcome_digest: format!("outcome-{id}"),
            prediction_error: Some(0.1),
            utility: vec![1.0, 0.5],
            compute_cost: 1.0,
            uncertainty: Some(0.2),
            model_version: "test-v1".into(),
            evidence_digest: format!("evidence-{id}"),
            provenance: ExperienceProvenance::Observed,
        }
    }

    #[test]
    fn append_is_parent_ordered_and_immutable() {
        let mut tree = ExperienceTree::new();
        assert_eq!(
            tree.append(node(2, Some(1))),
            Err(ExperienceTreeError::MissingParent(1))
        );

        tree.append(node(1, None)).unwrap();
        tree.append(node(2, Some(1))).unwrap();
        assert_eq!(tree.children_of(1), &[2]);
        assert_eq!(tree.len(), 2);

        assert_eq!(
            tree.append(node(2, Some(1))),
            Err(ExperienceTreeError::DuplicateNode(2))
        );
    }

    #[test]
    fn rejects_invalid_uncertainty() {
        let mut invalid = node(1, None);
        invalid.uncertainty = Some(1.5);
        assert_eq!(
            invalid.validate(),
            Err(ExperienceTreeError::InvalidUncertainty(1))
        );
    }
}
