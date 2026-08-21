// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed semantic graph synchronization for SCIP.
//!
//! A shared SCIP HDC profile is deterministic, so peers do not need to transmit
//! a dense HDC vector every time cognitive state changes. They can synchronize
//! the canonical [`GroundedConceptGraph`] and reconstruct the projection locally.

use crate::{InterchangeError, graph_semantic_hash, validate_graph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_communication::{ConceptEdge, ConceptNode, GroundedConceptGraph, content_hash};

/// Exact edit set from one grounded semantic graph to another.
///
/// Both endpoints are content-addressed. Applying a delta to any graph other
/// than `base_semantic_hash` fails, and the reconstructed graph must hash to
/// `target_semantic_hash` before it is accepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphDelta {
    pub base_semantic_hash: String,
    pub target_semantic_hash: String,
    pub remove_nodes: Vec<String>,
    pub upsert_nodes: Vec<ConceptNode>,
    pub remove_edges: Vec<String>,
    pub add_edges: Vec<ConceptEdge>,
}

impl GraphDelta {
    pub fn between(
        base: &GroundedConceptGraph,
        target: &GroundedConceptGraph,
    ) -> Result<Self, InterchangeError> {
        validate_graph(base)?;
        validate_graph(target)?;

        let base_nodes: BTreeMap<String, ConceptNode> = base
            .nodes
            .iter()
            .cloned()
            .map(|node| (node.id.clone(), node))
            .collect();
        let target_nodes: BTreeMap<String, ConceptNode> = target
            .nodes
            .iter()
            .cloned()
            .map(|node| (node.id.clone(), node))
            .collect();

        let mut remove_nodes = base_nodes
            .keys()
            .filter(|id| !target_nodes.contains_key(*id))
            .cloned()
            .collect::<Vec<_>>();
        let mut upsert_nodes = target_nodes
            .iter()
            .filter_map(|(id, node)| match base_nodes.get(id) {
                Some(existing) if existing == node => None,
                _ => Some(node.clone()),
            })
            .collect::<Vec<_>>();

        let base_edges = edge_map(base)?;
        let target_edges = edge_map(target)?;
        let mut remove_edges = base_edges
            .keys()
            .filter(|hash| !target_edges.contains_key(*hash))
            .cloned()
            .collect::<Vec<_>>();
        let mut add_edges = target_edges
            .iter()
            .filter(|(hash, _)| !base_edges.contains_key(*hash))
            .map(|(_, edge)| edge.clone())
            .collect::<Vec<_>>();

        remove_nodes.sort();
        upsert_nodes.sort_by(|a, b| a.id.cmp(&b.id));
        remove_edges.sort();
        sort_edges(&mut add_edges)?;

        Ok(Self {
            base_semantic_hash: graph_semantic_hash(base)?,
            target_semantic_hash: graph_semantic_hash(target)?,
            remove_nodes,
            upsert_nodes,
            remove_edges,
            add_edges,
        })
    }

    pub fn apply(
        &self,
        base: &GroundedConceptGraph,
    ) -> Result<GroundedConceptGraph, InterchangeError> {
        validate_graph(base)?;
        if graph_semantic_hash(base)? != self.base_semantic_hash {
            return Err(InterchangeError::InvalidDelta(
                "graph delta base semantic hash mismatch".into(),
            ));
        }

        ensure_unique(&self.remove_nodes, "duplicate node removal")?;
        ensure_unique(&self.remove_edges, "duplicate edge removal")?;

        let mut nodes: BTreeMap<String, ConceptNode> = base
            .nodes
            .iter()
            .cloned()
            .map(|node| (node.id.clone(), node))
            .collect();

        for id in &self.remove_nodes {
            if nodes.remove(id).is_none() {
                return Err(InterchangeError::InvalidDelta(format!(
                    "graph delta removes unknown node {id}"
                )));
            }
        }

        let mut seen_upserts = BTreeSet::new();
        for node in &self.upsert_nodes {
            if !seen_upserts.insert(node.id.as_str()) {
                return Err(InterchangeError::InvalidDelta(format!(
                    "duplicate upsert for node {}",
                    node.id
                )));
            }
            nodes.insert(node.id.clone(), node.clone());
        }

        let mut edges = edge_map(base)?;
        for hash in &self.remove_edges {
            if edges.remove(hash).is_none() {
                return Err(InterchangeError::InvalidDelta(format!(
                    "graph delta removes unknown edge {hash}"
                )));
            }
        }
        for edge in &self.add_edges {
            let hash = edge_semantic_hash(edge)?;
            if edges.insert(hash.clone(), edge.clone()).is_some() {
                return Err(InterchangeError::InvalidDelta(format!(
                    "graph delta adds duplicate edge {hash}"
                )));
            }
        }

        let mut graph = GroundedConceptGraph {
            nodes: nodes.into_values().collect(),
            edges: edges.into_values().collect(),
        };
        graph.nodes.sort_by(|a, b| a.id.cmp(&b.id));
        sort_edges(&mut graph.edges)?;
        validate_graph(&graph)?;

        if graph_semantic_hash(&graph)? != self.target_semantic_hash {
            return Err(InterchangeError::InvalidDelta(
                "graph delta target semantic hash mismatch".into(),
            ));
        }
        Ok(graph)
    }

    /// Deterministic bytes suitable for wire-size measurement and hashing.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, InterchangeError> {
        let mut canonical = self.clone();
        canonical.remove_nodes.sort();
        canonical.upsert_nodes.sort_by(|a, b| a.id.cmp(&b.id));
        canonical.remove_edges.sort();
        sort_edges(&mut canonical.add_edges)?;
        Ok(serde_json::to_vec(&canonical)?)
    }

    pub fn estimated_wire_bytes(&self) -> Result<usize, InterchangeError> {
        Ok(self.canonical_bytes()?.len())
    }
}

/// Canonical bytes for one semantic edge. Evidence ordering is non-semantic.
pub fn canonical_edge_bytes(edge: &ConceptEdge) -> Result<Vec<u8>, InterchangeError> {
    let mut canonical = edge.clone();
    canonical.evidence_ids.sort();
    Ok(serde_json::to_vec(&canonical)?)
}

/// Content address used by graph deltas to remove exact semantic edges.
pub fn edge_semantic_hash(edge: &ConceptEdge) -> Result<String, InterchangeError> {
    Ok(content_hash(&canonical_edge_bytes(edge)?))
}

fn edge_map(
    graph: &GroundedConceptGraph,
) -> Result<BTreeMap<String, ConceptEdge>, InterchangeError> {
    let mut map = BTreeMap::new();
    for edge in &graph.edges {
        let hash = edge_semantic_hash(edge)?;
        if map.insert(hash.clone(), edge.clone()).is_some() {
            return Err(InterchangeError::InvalidGraph(format!(
                "duplicate semantic edge {hash}"
            )));
        }
    }
    Ok(map)
}

fn sort_edges(edges: &mut [ConceptEdge]) -> Result<(), InterchangeError> {
    let mut keyed = edges
        .iter()
        .cloned()
        .map(|edge| Ok((edge_semantic_hash(&edge)?, edge)))
        .collect::<Result<Vec<_>, InterchangeError>>()?;
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    for (slot, (_, edge)) in edges.iter_mut().zip(keyed) {
        *slot = edge;
    }
    Ok(())
}

fn ensure_unique(values: &[String], message: &str) -> Result<(), InterchangeError> {
    let mut seen = BTreeSet::new();
    if values.iter().any(|value| !seen.insert(value.as_str())) {
        return Err(InterchangeError::InvalidDelta(message.into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_communication::{ConceptKind, ConceptNode};

    fn graph() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "alice".into(),
                    kind: ConceptKind::Agent,
                    label: Some("Alice".into()),
                    grounded_by: vec!["obs-a".into()],
                    confidence: 0.9,
                },
                ConceptNode {
                    id: "reactor".into(),
                    kind: ConceptKind::Object,
                    label: Some("Reactor".into()),
                    grounded_by: vec!["obs-r".into()],
                    confidence: 0.95,
                },
            ],
            edges: vec![ConceptEdge {
                source: "alice".into(),
                relation: "observes".into(),
                target: "reactor".into(),
                evidence_ids: vec!["ev-1".into()],
                confidence: 0.8,
            }],
        }
    }

    #[test]
    fn graph_delta_round_trips_by_semantic_hash() {
        let base = graph();
        let mut target = base.clone();
        target.nodes[1].confidence = 0.85;
        target.nodes.push(ConceptNode {
            id: "sensor".into(),
            kind: ConceptKind::Object,
            label: Some("S17".into()),
            grounded_by: vec!["obs-s17".into()],
            confidence: 0.99,
        });
        target.edges.push(ConceptEdge {
            source: "sensor".into(),
            relation: "measures".into(),
            target: "reactor".into(),
            evidence_ids: vec!["ev-2".into()],
            confidence: 0.93,
        });

        let delta = GraphDelta::between(&base, &target).unwrap();
        let reconstructed = delta.apply(&base).unwrap();
        assert_eq!(
            graph_semantic_hash(&reconstructed).unwrap(),
            graph_semantic_hash(&target).unwrap()
        );
    }

    #[test]
    fn graph_delta_rejects_wrong_base() {
        let base = graph();
        let mut target = base.clone();
        target.nodes[0].confidence = 0.7;
        let delta = GraphDelta::between(&base, &target).unwrap();

        let mut wrong = base;
        wrong.nodes[1].confidence = 0.1;
        assert!(delta.apply(&wrong).is_err());
    }

    #[test]
    fn edge_hash_ignores_evidence_order() {
        let mut edge = graph().edges.remove(0);
        edge.evidence_ids = vec!["b".into(), "a".into()];
        let first = edge_semantic_hash(&edge).unwrap();
        edge.evidence_ids.reverse();
        assert_eq!(first, edge_semantic_hash(&edge).unwrap());
    }
}
