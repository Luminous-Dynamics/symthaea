// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed peer possession state for safe SCIP transfer planning.
//!
//! This module records what a peer has acknowledged caching so callers do not
//! have to manually translate that session state into optimistic reference or
//! delta candidates. A [`SemanticCacheAck`] is deliberately **not** an
//! authentication primitive: Xenia or another transport/session layer must
//! authenticate and transcript-bind an acknowledgement before it is recorded.

use crate::protocol::require_content_hash;
use crate::{
    GraphDelta, InterchangeError, InterchangeRepresentation, NegotiatedSession,
    ProjectionCandidate, SemanticReference, TransferPlanningInput, canonical_graph_bytes,
    graph_semantic_hash,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_communication::GroundedConceptGraph;

pub const MAX_PEER_SEMANTIC_INVENTORY_ENTRIES: usize = 4_096;

/// Transport-neutral acknowledgement that a peer possesses one exact canonical
/// grounded semantic object.
///
/// The hash is meaningful only after the surrounding session authenticates who
/// sent the acknowledgement. SCIP validates the content address but does not
/// authenticate the peer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticCacheAck {
    pub semantic_hash: String,
}

impl SemanticCacheAck {
    pub fn new(semantic_hash: impl Into<String>) -> Result<Self, InterchangeError> {
        let ack = Self {
            semantic_hash: semantic_hash.into(),
        };
        ack.validate()?;
        Ok(ack)
    }

    pub fn from_graph(graph: &GroundedConceptGraph) -> Result<Self, InterchangeError> {
        Self::new(graph_semantic_hash(graph)?)
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        require_content_hash(&self.semantic_hash, "semantic cache acknowledgement")
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, InterchangeError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

/// Bounded session-local record of canonical semantic hashes a peer has
/// acknowledged possessing.
///
/// This is deliberately not serializable as authoritative state. Populate it
/// only from acknowledgements that the surrounding transport/session layer has
/// already authenticated and accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerSemanticInventory {
    max_entries: usize,
    semantic_hashes: BTreeSet<String>,
}

impl Default for PeerSemanticInventory {
    fn default() -> Self {
        Self {
            max_entries: MAX_PEER_SEMANTIC_INVENTORY_ENTRIES,
            semantic_hashes: BTreeSet::new(),
        }
    }
}

impl PeerSemanticInventory {
    pub fn with_limit(max_entries: usize) -> Result<Self, InterchangeError> {
        if max_entries == 0 || max_entries > MAX_PEER_SEMANTIC_INVENTORY_ENTRIES {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "peer semantic inventory limit {max_entries} is outside 1..={MAX_PEER_SEMANTIC_INVENTORY_ENTRIES}"
            )));
        }
        Ok(Self {
            max_entries,
            semantic_hashes: BTreeSet::new(),
        })
    }

    /// Record an acknowledgement only after the caller has authenticated it at
    /// the session/transport boundary. Returns `false` for an already-known hash.
    pub fn record_ack(&mut self, ack: &SemanticCacheAck) -> Result<bool, InterchangeError> {
        ack.validate()?;
        if self.semantic_hashes.contains(&ack.semantic_hash) {
            return Ok(false);
        }
        if self.semantic_hashes.len() >= self.max_entries {
            return Err(InterchangeError::ResourceLimitExceeded(format!(
                "peer semantic inventory reached its {} entry limit",
                self.max_entries
            )));
        }
        Ok(self.semantic_hashes.insert(ack.semantic_hash.clone()))
    }

    pub fn contains(&self, semantic_hash: &str) -> bool {
        self.semantic_hashes.contains(semantic_hash)
    }

    pub fn revoke(&mut self, semantic_hash: &str) -> Result<bool, InterchangeError> {
        require_content_hash(semantic_hash, "semantic inventory revocation")?;
        Ok(self.semantic_hashes.remove(semantic_hash))
    }

    pub fn clear(&mut self) {
        self.semantic_hashes.clear();
    }

    pub fn len(&self) -> usize {
        self.semantic_hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.semantic_hashes.is_empty()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

/// Build the low-level transfer-planner input from negotiated session state and
/// authenticated peer possession state.
///
/// The builder fails if the session has no exact grounded representation. It
/// exposes a semantic-reference candidate only when the peer inventory contains
/// the exact target hash, and a graph-delta candidate only when exact deltas were
/// negotiated and the peer inventory contains the delta's exact base hash.
/// Human-text and HDC projection candidates are likewise removed when those
/// representations were not negotiated.
pub fn build_grounded_transfer_input(
    session: &NegotiatedSession,
    peer_inventory: &PeerSemanticInventory,
    target_graph: &GroundedConceptGraph,
    delta: Option<&GraphDelta>,
    human_text_bytes: Option<usize>,
    projection_candidates: Vec<ProjectionCandidate>,
) -> Result<TransferPlanningInput, InterchangeError> {
    let shares = |representation: &InterchangeRepresentation| {
        session.shared_representations.contains(representation)
    };
    let shares_exact_grounded = shares(&InterchangeRepresentation::GroundedGraph)
        || shares(&InterchangeRepresentation::StructuredJson);
    if !shares_exact_grounded {
        return Err(InterchangeError::NegotiationFailed);
    }

    let target_semantic_hash = graph_semantic_hash(target_graph)?;
    let grounded_graph_bytes = canonical_graph_bytes(target_graph)?.len();

    let semantic_reference_bytes =
        if session.semantic_references && peer_inventory.contains(&target_semantic_hash) {
            Some(
                serde_json::to_vec(&SemanticReference {
                    semantic_hash: target_semantic_hash.clone(),
                })?
                .len(),
            )
        } else {
            None
        };

    let graph_delta_bytes = if let Some(delta) = delta {
        require_content_hash(&delta.base_semantic_hash, "graph delta base semantic hash")?;
        require_content_hash(
            &delta.target_semantic_hash,
            "graph delta target semantic hash",
        )?;
        if delta.target_semantic_hash != target_semantic_hash {
            return Err(InterchangeError::InvalidDelta(
                "graph delta target does not match transfer target".into(),
            ));
        }
        if session.exact_graph_deltas && peer_inventory.contains(&delta.base_semantic_hash) {
            Some(delta.estimated_wire_bytes()?)
        } else {
            None
        }
    } else {
        None
    };

    let human_text_bytes =
        human_text_bytes.filter(|_| shares(&InterchangeRepresentation::HumanText));
    let projection_candidates =
        if session.hdc_profile.is_some() && shares(&InterchangeRepresentation::Hdc) {
            projection_candidates
        } else {
            Vec::new()
        };

    Ok(TransferPlanningInput {
        semantic_reference_bytes,
        graph_delta_bytes,
        grounded_graph_bytes,
        human_text_bytes,
        projection_candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HdcWireEncoding, PeerCapabilities, ProjectionCandidate, SemanticTransferMode,
        TransferPolicy, negotiate, plan_transfer,
    };
    use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode};

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

    fn acknowledge(
        inventory: &mut PeerSemanticInventory,
        graph: &GroundedConceptGraph,
    ) -> Result<(), InterchangeError> {
        let ack = SemanticCacheAck::from_graph(graph)?;
        inventory.record_ack(&ack)?;
        Ok(())
    }

    #[test]
    fn semantic_cache_ack_rejects_malformed_hash() {
        assert!(SemanticCacheAck::new("not-a-content-hash").is_err());
    }

    #[test]
    fn peer_inventory_is_bounded_and_duplicate_acks_are_idempotent() {
        let mut inventory = PeerSemanticInventory::with_limit(1).unwrap();
        let first = SemanticCacheAck::new("a".repeat(64)).unwrap();
        let second = SemanticCacheAck::new("b".repeat(64)).unwrap();

        assert!(inventory.record_ack(&first).unwrap());
        assert!(!inventory.record_ack(&first).unwrap());
        assert!(matches!(
            inventory.record_ack(&second),
            Err(InterchangeError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    fn no_ack_exposes_only_full_grounded_semantics() {
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let target = target_graph();
        let delta = GraphDelta::between(&graph(), &target).unwrap();
        let input = build_grounded_transfer_input(
            &session,
            &PeerSemanticInventory::default(),
            &target,
            Some(&delta),
            Some(10),
            vec![],
        )
        .unwrap();

        assert_eq!(input.semantic_reference_bytes, None);
        assert_eq!(input.graph_delta_bytes, None);
        let plan = plan_transfer(&input, TransferPolicy::default()).unwrap();
        assert_eq!(plan.semantic, SemanticTransferMode::GroundedGraph);
    }

    #[test]
    fn acknowledgements_enable_candidates_without_overriding_size_planning() {
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let base = graph();
        let target = target_graph();
        let delta = GraphDelta::between(&base, &target).unwrap();
        let mut inventory = PeerSemanticInventory::default();
        acknowledge(&mut inventory, &base).unwrap();

        let delta_input = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        let delta_bytes = delta_input.graph_delta_bytes.unwrap();
        assert_eq!(delta_input.semantic_reference_bytes, None);

        let delta_plan = plan_transfer(&delta_input, TransferPolicy::default()).unwrap();
        let expected_delta_mode = if delta_bytes <= delta_input.grounded_graph_bytes {
            SemanticTransferMode::GraphDelta
        } else {
            SemanticTransferMode::GroundedGraph
        };
        assert_eq!(delta_plan.semantic, expected_delta_mode);
        assert_eq!(
            delta_plan.semantic_bytes,
            delta_bytes.min(delta_input.grounded_graph_bytes)
        );

        acknowledge(&mut inventory, &target).unwrap();
        let reference_input = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![],
        )
        .unwrap();
        let reference_bytes = reference_input.semantic_reference_bytes.unwrap();
        let graph_delta_bytes = reference_input.graph_delta_bytes.unwrap();
        let expected_reference_mode = [
            (SemanticTransferMode::SemanticReference, reference_bytes),
            (SemanticTransferMode::GraphDelta, graph_delta_bytes),
            (
                SemanticTransferMode::GroundedGraph,
                reference_input.grounded_graph_bytes,
            ),
        ]
        .into_iter()
        .min_by_key(|(_, bytes)| *bytes)
        .unwrap();
        let reference_plan = plan_transfer(&reference_input, TransferPolicy::default()).unwrap();
        assert_eq!(reference_plan.semantic, expected_reference_mode.0);
        assert_eq!(reference_plan.semantic_bytes, expected_reference_mode.1);
    }

    #[test]
    fn mismatched_delta_target_is_rejected_before_planning() {
        let session = negotiate(
            &PeerCapabilities::structured_only(),
            &PeerCapabilities::structured_only(),
        )
        .unwrap();
        let base = graph();
        let mut other_target = target_graph();
        other_target.nodes[1].confidence = 0.55;
        let delta = GraphDelta::between(&base, &other_target).unwrap();
        let mut inventory = PeerSemanticInventory::default();
        acknowledge(&mut inventory, &base).unwrap();

        assert!(matches!(
            build_grounded_transfer_input(
                &session,
                &inventory,
                &target_graph(),
                Some(&delta),
                None,
                vec![],
            ),
            Err(InterchangeError::InvalidDelta(_))
        ));
    }

    #[test]
    fn hdc_preferred_session_can_use_grounded_delta_when_base_is_acknowledged() {
        let caps = PeerCapabilities::symthaea_default();
        let session = negotiate(&caps, &caps).unwrap();
        let base = graph();
        let target = target_graph();
        let delta = GraphDelta::between(&base, &target).unwrap();
        let mut inventory = PeerSemanticInventory::default();
        acknowledge(&mut inventory, &base).unwrap();

        let input = build_grounded_transfer_input(
            &session,
            &inventory,
            &target,
            Some(&delta),
            None,
            vec![ProjectionCandidate {
                encoding: HdcWireEncoding::Q8SymmetricV1,
                bytes: 16_384,
                cosine_similarity: 0.999,
                exact: false,
            }],
        )
        .unwrap();
        assert!(input.graph_delta_bytes.is_some());
        assert_eq!(input.projection_candidates.len(), 1);
    }

    #[test]
    fn hdc_only_session_cannot_build_grounded_transfer_input() {
        let mut caps = PeerCapabilities::symthaea_default();
        caps.representations = vec![InterchangeRepresentation::Hdc];
        let session = negotiate(&caps, &caps).unwrap();

        assert!(matches!(
            build_grounded_transfer_input(
                &session,
                &PeerSemanticInventory::default(),
                &target_graph(),
                None,
                None,
                vec![],
            ),
            Err(InterchangeError::NegotiationFailed)
        ));
    }

    #[test]
    fn unnegotiated_text_and_projection_candidates_are_removed() {
        let mut caps = PeerCapabilities::structured_only();
        caps.representations = vec![InterchangeRepresentation::GroundedGraph];
        let session = negotiate(&caps, &caps).unwrap();
        let input = build_grounded_transfer_input(
            &session,
            &PeerSemanticInventory::default(),
            &target_graph(),
            None,
            Some(7),
            vec![ProjectionCandidate {
                encoding: HdcWireEncoding::Q8SymmetricV1,
                bytes: 16_384,
                cosine_similarity: 0.999,
                exact: false,
            }],
        )
        .unwrap();

        assert_eq!(input.human_text_bytes, None);
        assert!(input.projection_candidates.is_empty());
    }
}
