// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Session-local cache feedback for SCIP semantic synchronization.
//!
//! A peer may legitimately evict semantic state after previously acknowledging
//! it. This module lets an authenticated session retract only the stale
//! possession claim so the existing transfer planner can recompute a safe
//! fallback. SCIP validates content addresses but does not authenticate, order,
//! or authorize feedback; those responsibilities remain with Xenia or another
//! surrounding session/transport layer.

use crate::protocol::require_content_hash;
use crate::session::{PeerSemanticInventory, SemanticCacheAck};
use crate::InterchangeError;
use serde::{Deserialize, Serialize};

/// Which exact prerequisite was missing when a transfer could not be resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCacheMissKind {
    /// A `SemanticReference` target was not present in the receiver cache.
    SemanticReferenceTarget,
    /// The exact base graph required by a `GraphDelta` was not present.
    GraphDeltaBase,
}

/// Feedback that an attempted exact semantic transfer referenced unavailable
/// cached state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticCacheMiss {
    pub semantic_hash: String,
    pub requirement: SemanticCacheMissKind,
}

impl SemanticCacheMiss {
    pub fn new(
        semantic_hash: impl Into<String>,
        requirement: SemanticCacheMissKind,
    ) -> Result<Self, InterchangeError> {
        let miss = Self {
            semantic_hash: semantic_hash.into(),
            requirement,
        };
        miss.validate()?;
        Ok(miss)
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        require_content_hash(&self.semantic_hash, "semantic cache miss")
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, InterchangeError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

/// Proactive notice that a peer no longer claims to possess one semantic hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticCacheRevoke {
    pub semantic_hash: String,
}

impl SemanticCacheRevoke {
    pub fn new(semantic_hash: impl Into<String>) -> Result<Self, InterchangeError> {
        let revoke = Self {
            semantic_hash: semantic_hash.into(),
        };
        revoke.validate()?;
        Ok(revoke)
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        require_content_hash(&self.semantic_hash, "semantic cache revocation")
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, InterchangeError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

/// Transport-neutral cache feedback accepted by a SCIP session.
///
/// The enum uses an explicit tagged shape so independent transports can inspect
/// the feedback kind without guessing an enum representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "body", rename_all = "snake_case")]
pub enum SemanticCacheFeedback {
    Ack(SemanticCacheAck),
    Miss(SemanticCacheMiss),
    Revoke(SemanticCacheRevoke),
}

impl SemanticCacheFeedback {
    pub fn validate(&self) -> Result<(), InterchangeError> {
        match self {
            Self::Ack(ack) => ack.validate(),
            Self::Miss(miss) => miss.validate(),
            Self::Revoke(revoke) => revoke.validate(),
        }
    }

    /// Deterministic bytes for transcript binding and size measurement.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, InterchangeError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

/// Effect of accepted feedback on the local record of peer possession.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryUpdate {
    Added,
    Removed,
    Unchanged,
}

/// Apply cache feedback after the surrounding session has authenticated,
/// freshness-checked, transcript-bound, and authorized it.
///
/// A miss or revocation removes only the named semantic hash. Repeated or
/// already-obsolete feedback is idempotent. Callers then rebuild their ordinary
/// `TransferPlanningInput`; no special recovery planner is required.
pub fn apply_cache_feedback(
    inventory: &mut PeerSemanticInventory,
    feedback: &SemanticCacheFeedback,
) -> Result<InventoryUpdate, InterchangeError> {
    feedback.validate()?;
    match feedback {
        SemanticCacheFeedback::Ack(ack) => {
            if inventory.record_ack(ack)? {
                Ok(InventoryUpdate::Added)
            } else {
                Ok(InventoryUpdate::Unchanged)
            }
        }
        SemanticCacheFeedback::Miss(miss) => {
            if inventory.revoke(&miss.semantic_hash)? {
                Ok(InventoryUpdate::Removed)
            } else {
                Ok(InventoryUpdate::Unchanged)
            }
        }
        SemanticCacheFeedback::Revoke(revoke) => {
            if inventory.revoke(&revoke.semantic_hash)? {
                Ok(InventoryUpdate::Removed)
            } else {
                Ok(InventoryUpdate::Unchanged)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GraphDelta, PeerCapabilities, SemanticTransferMode, TransferPolicy,
        build_grounded_transfer_input, graph_semantic_hash, negotiate, plan_transfer,
    };
    use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};

    fn graph() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "reactor".into(),
                    kind: ConceptKind::Object,
                    label: Some("R-17".into()),
                    grounded_by: vec!["obs-r17".into()],
                    confidence: 0.95,
                },
                ConceptNode {
                    id: "sensor".into(),
                    kind: ConceptKind::Object,
                    label: Some("S-22".into()),
                    grounded_by: vec!["obs-s22".into()],
                    confidence: 0.91,
                },
            ],
            edges: vec![ConceptEdge {
                source: "sensor".into(),
                relation: "measures".into(),
                target: "reactor".into(),
                evidence_ids: vec!["ev-1".into()],
                confidence: 0.9,
            }],
        }
    }

    fn target_graph() -> GroundedConceptGraph {
        let mut target = graph();
        target.nodes[0].confidence = 0.82;
        target
    }

    fn ack(graph: &GroundedConceptGraph) -> SemanticCacheFeedback {
        SemanticCacheFeedback::Ack(SemanticCacheAck::from_graph(graph).unwrap())
    }

    #[test]
    fn malformed_miss_and_revoke_are_rejected() {
        assert!(
            SemanticCacheMiss::new(
                "bad",
                SemanticCacheMissKind::SemanticReferenceTarget,
            )
            .is_err()
        );
        assert!(SemanticCacheRevoke::new("also-bad").is_err());
    }

    #[test]
    fn feedback_updates_are_idempotent() {
        let graph = graph();
        let hash = graph_semantic_hash(&graph).unwrap();
        let mut inventory = PeerSemanticInventory::default();

        assert_eq!(
            apply_cache_feedback(&mut inventory, &ack(&graph)).unwrap(),
            InventoryUpdate::Added
        );
        assert_eq!(
            apply_cache_feedback(&mut inventory, &ack(&graph)).unwrap(),
            InventoryUpdate::Unchanged
        );

        let revoke = SemanticCacheFeedback::Revoke(SemanticCacheRevoke::new(hash).unwrap());
        assert_eq!(
            apply_cache_feedback(&mut inventory, &revoke).unwrap(),
            InventoryUpdate::Removed
        );
        assert_eq!(
            apply_cache_feedback(&mut inventory, &revoke).unwrap(),
            InventoryUpdate::Unchanged
        );
    }

    #[test]
    fn reference_miss_removes_only_target_claim() {
        let base = graph();
        let target = target_graph();
        let base_hash = graph_semantic_hash(&base).unwrap();
        let target_hash = graph_semantic_hash(&target).unwrap();
        let delta = GraphDelta::between(&base, &target).unwrap();
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let mut inventory = PeerSemanticInventory::default();
        apply_cache_feedback(&mut inventory, &ack(&base)).unwrap();
        apply_cache_feedback(&mut inventory, &ack(&target)).unwrap();

        let before = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        assert!(before.semantic_reference_bytes.is_some());
        assert!(before.graph_delta_bytes.is_some());

        let miss = SemanticCacheFeedback::Miss(
            SemanticCacheMiss::new(
                target_hash.clone(),
                SemanticCacheMissKind::SemanticReferenceTarget,
            )
            .unwrap(),
        );
        assert_eq!(
            apply_cache_feedback(&mut inventory, &miss).unwrap(),
            InventoryUpdate::Removed
        );
        assert!(!inventory.contains(&target_hash));
        assert!(inventory.contains(&base_hash));

        let after = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(after.semantic_reference_bytes, None);
        assert!(after.graph_delta_bytes.is_some());
    }

    #[test]
    fn delta_base_miss_falls_back_to_full_grounded_graph() {
        let base = graph();
        let target = target_graph();
        let base_hash = graph_semantic_hash(&base).unwrap();
        let delta = GraphDelta::between(&base, &target).unwrap();
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let mut inventory = PeerSemanticInventory::default();
        apply_cache_feedback(&mut inventory, &ack(&base)).unwrap();

        let before = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        assert!(before.graph_delta_bytes.is_some());

        let miss = SemanticCacheFeedback::Miss(
            SemanticCacheMiss::new(base_hash, SemanticCacheMissKind::GraphDeltaBase).unwrap(),
        );
        apply_cache_feedback(&mut inventory, &miss).unwrap();

        let after = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(after.semantic_reference_bytes, None);
        assert_eq!(after.graph_delta_bytes, None);
        assert_eq!(
            plan_transfer(&after, TransferPolicy::default())
                .unwrap()
                .semantic,
            SemanticTransferMode::GroundedGraph
        );
    }

    #[test]
    fn reack_after_miss_restores_reference_eligibility() {
        let target = target_graph();
        let target_hash = graph_semantic_hash(&target).unwrap();
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let mut inventory = PeerSemanticInventory::default();
        apply_cache_feedback(&mut inventory, &ack(&target)).unwrap();

        let miss = SemanticCacheFeedback::Miss(
            SemanticCacheMiss::new(
                target_hash,
                SemanticCacheMissKind::SemanticReferenceTarget,
            )
            .unwrap(),
        );
        apply_cache_feedback(&mut inventory, &miss).unwrap();
        assert_eq!(
            build_grounded_transfer_input(&session, &inventory, &target, None, None, vec![])
                .unwrap()
                .semantic_reference_bytes,
            None
        );

        assert_eq!(
            apply_cache_feedback(&mut inventory, &ack(&target)).unwrap(),
            InventoryUpdate::Added
        );
        assert!(
            build_grounded_transfer_input(&session, &inventory, &target, None, None, vec![])
                .unwrap()
                .semantic_reference_bytes
                .is_some()
        );
    }

    #[test]
    fn feedback_bytes_are_deterministic_and_tagged() {
        let feedback = SemanticCacheFeedback::Miss(
            SemanticCacheMiss::new(
                "a".repeat(64),
                SemanticCacheMissKind::SemanticReferenceTarget,
            )
            .unwrap(),
        );
        let first = feedback.canonical_bytes().unwrap();
        let second = feedback.canonical_bytes().unwrap();
        assert_eq!(first, second);
        let encoded = String::from_utf8(first).unwrap();
        assert!(encoded.contains("\"kind\":\"miss\""));
        assert!(encoded.contains("\"semantic_reference_target\""));
    }
}
