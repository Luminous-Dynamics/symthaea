//! Deterministic N0 interlingua preservation measurements.
//!
//! These helpers compare grounded concept graphs without claiming semantic
//! understanding. They deliberately separate structural preservation from
//! transport integrity and from any future neural decoder quality.

use crate::{content_hash, ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const INTERLINGUA_BENCHMARK_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterlinguaPerturbation {
    Exact,
    ReorderedCollections,
    RenamedIdentifiers,
    MissingEdge,
    DuplicateEdge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterlinguaMetrics {
    pub schema_version: u32,
    pub expected_graph_hash: String,
    pub observed_graph_hash: String,
    pub node_precision: f64,
    pub node_recall: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub confidence_mae: f64,
    pub structural_equivalence: bool,
    pub expected_bytes: usize,
    pub observed_bytes: usize,
}

impl InterlinguaMetrics {
    pub fn validates(&self) -> bool {
        self.schema_version == INTERLINGUA_BENCHMARK_SCHEMA_VERSION
            && self.node_precision.is_finite()
            && self.node_recall.is_finite()
            && self.edge_precision.is_finite()
            && self.edge_recall.is_finite()
            && self.confidence_mae.is_finite()
            && (0.0..=1.0).contains(&self.node_precision)
            && (0.0..=1.0).contains(&self.node_recall)
            && (0.0..=1.0).contains(&self.edge_precision)
            && (0.0..=1.0).contains(&self.edge_recall)
            && self.confidence_mae >= 0.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterlinguaBenchmarkCase {
    pub case_id: String,
    pub seed: u64,
    pub perturbation: InterlinguaPerturbation,
    pub metrics: InterlinguaMetrics,
}

pub fn graph_hash(graph: &GroundedConceptGraph) -> Result<String, String> {
    let bytes = serde_json::to_vec(graph).map_err(|error| error.to_string())?;
    Ok(content_hash(&bytes))
}

/// Compare graphs by grounded semantic structure, not collection order or
/// lexical labels. Node identifiers are treated as transport-local names.
pub fn compare_graphs(
    expected: &GroundedConceptGraph,
    observed: &GroundedConceptGraph,
) -> Result<InterlinguaMetrics, String> {
    let expected_nodes = canonical_nodes(expected);
    let observed_nodes = canonical_nodes(observed);
    let expected_edges = canonical_edges(expected);
    let observed_edges = canonical_edges(observed);

    let node_intersection = expected_nodes
        .keys()
        .filter(|key| observed_nodes.contains_key(*key))
        .count();
    let edge_intersection = expected_edges
        .iter()
        .filter(|key| observed_edges.binary_search(key).is_ok())
        .count();

    let node_precision = ratio(node_intersection, observed_nodes.len());
    let node_recall = ratio(node_intersection, expected_nodes.len());
    let edge_precision = ratio(edge_intersection, observed_edges.len());
    let edge_recall = ratio(edge_intersection, expected_edges.len());

    let confidence_mae = confidence_mae(&expected_nodes, &observed_nodes);
    let structural_equivalence = expected_nodes == observed_nodes && expected_edges == observed_edges;

    let expected_bytes = serde_json::to_vec(expected)
        .map_err(|error| error.to_string())?
        .len();
    let observed_bytes = serde_json::to_vec(observed)
        .map_err(|error| error.to_string())?
        .len();

    let metrics = InterlinguaMetrics {
        schema_version: INTERLINGUA_BENCHMARK_SCHEMA_VERSION,
        expected_graph_hash: graph_hash(expected)?,
        observed_graph_hash: graph_hash(observed)?,
        node_precision,
        node_recall,
        edge_precision,
        edge_recall,
        confidence_mae,
        structural_equivalence,
        expected_bytes,
        observed_bytes,
    };

    if !metrics.validates() {
        return Err("interlingua metrics are invalid".into());
    }
    Ok(metrics)
}

pub fn reorder_collections(graph: &GroundedConceptGraph) -> GroundedConceptGraph {
    let mut reordered = graph.clone();
    reordered.nodes.reverse();
    reordered.edges.reverse();
    reordered
}

pub fn rename_identifiers(graph: &GroundedConceptGraph, prefix: &str) -> GroundedConceptGraph {
    let mapping = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.clone(), format!("{prefix}{index}")))
        .collect::<BTreeMap<_, _>>();

    let mut renamed = graph.clone();
    for node in &mut renamed.nodes {
        node.id = mapping
            .get(&node.id)
            .cloned()
            .unwrap_or_else(|| node.id.clone());
    }
    for edge in &mut renamed.edges {
        if let Some(source) = mapping.get(&edge.source) {
            edge.source = source.clone();
        }
        if let Some(target) = mapping.get(&edge.target) {
            edge.target = target.clone();
        }
    }
    renamed
}

pub fn drop_last_edge(graph: &GroundedConceptGraph) -> GroundedConceptGraph {
    let mut reduced = graph.clone();
    reduced.edges.pop();
    reduced
}

pub fn duplicate_last_edge(graph: &GroundedConceptGraph) -> GroundedConceptGraph {
    let mut duplicated = graph.clone();
    if let Some(edge) = duplicated.edges.last().cloned() {
        duplicated.edges.push(edge);
    }
    duplicated
}

fn canonical_nodes(graph: &GroundedConceptGraph) -> BTreeMap<String, (ConceptKind, Vec<String>, f32)> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let mut grounded = node.grounded_by.clone();
            grounded.sort();
            (
                canonical_node_key(node),
                (node.kind.clone(), grounded, node.confidence),
            )
        })
        .collect()
}

fn canonical_edges(graph: &GroundedConceptGraph) -> Vec<(String, String, String)> {
    let mut edges = graph
        .edges
        .iter()
        .map(|edge| {
            let source_key = graph
                .nodes
                .iter()
                .find(|node| node.id == edge.source)
                .map(canonical_node_key)
                .unwrap_or_else(|| edge.source.clone());
            let target_key = graph
                .nodes
                .iter()
                .find(|node| node.id == edge.target)
                .map(canonical_node_key)
                .unwrap_or_else(|| edge.target.clone());
            (source_key, edge.relation.clone(), target_key)
        })
        .collect::<Vec<_>>();
    edges.sort();
    edges
}

fn canonical_node_key(node: &ConceptNode) -> String {
    let mut grounded = node.grounded_by.clone();
    grounded.sort();
    let kind = serde_json::to_string(&node.kind).unwrap_or_else(|_| format!("{:?}", node.kind));
    format!("{kind}|{}", grounded.join(","))
}

fn confidence_mae(
    expected: &BTreeMap<String, (ConceptKind, Vec<String>, f32)>,
    observed: &BTreeMap<String, (ConceptKind, Vec<String>, f32)>,
) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for (key, (_, _, expected_confidence)) in expected {
        if let Some((_, _, observed_confidence)) = observed.get(key) {
            total += (*expected_confidence as f64 - *observed_confidence as f64).abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        if numerator == 0 { 1.0 } else { 0.0 }
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> GroundedConceptGraph {
        GroundedConceptGraph {
            nodes: vec![
                ConceptNode {
                    id: "agent".into(),
                    kind: ConceptKind::Agent,
                    label: Some("sender".into()),
                    grounded_by: vec!["obs-1".into()],
                    confidence: 0.9,
                },
                ConceptNode {
                    id: "event".into(),
                    kind: ConceptKind::Event,
                    label: Some("approach".into()),
                    grounded_by: vec!["obs-2".into()],
                    confidence: 0.8,
                },
                ConceptNode {
                    id: "object".into(),
                    kind: ConceptKind::Object,
                    label: Some("target".into()),
                    grounded_by: vec!["obs-3".into()],
                    confidence: 0.7,
                },
            ],
            edges: vec![
                ConceptEdge {
                    source: "agent".into(),
                    relation: "initiates".into(),
                    target: "event".into(),
                    evidence_ids: vec!["obs-2".into()],
                    confidence: 0.8,
                },
                ConceptEdge {
                    source: "event".into(),
                    relation: "targets".into(),
                    target: "object".into(),
                    evidence_ids: vec!["obs-3".into()],
                    confidence: 0.7,
                },
            ],
        }
    }

    #[test]
    fn collection_order_does_not_change_structural_equivalence() {
        let expected = fixture();
        let observed = reorder_collections(&expected);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.structural_equivalence);
        assert_eq!(metrics.node_recall, 1.0);
        assert_eq!(metrics.edge_recall, 1.0);
    }

    #[test]
    fn identifier_renaming_preserves_grounded_structure() {
        let expected = fixture();
        let observed = rename_identifiers(&expected, "node-");
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.structural_equivalence);
    }

    #[test]
    fn missing_edge_lowers_recall_without_fabricating_equivalence() {
        let expected = fixture();
        let observed = drop_last_edge(&expected);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(!metrics.structural_equivalence);
        assert!(metrics.edge_recall < 1.0);
    }

    #[test]
    fn duplicate_edge_does_not_hide_precision_loss() {
        let expected = fixture();
        let observed = duplicate_last_edge(&expected);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(!metrics.structural_equivalence);
        assert!(metrics.edge_precision < 1.0);
    }
}
