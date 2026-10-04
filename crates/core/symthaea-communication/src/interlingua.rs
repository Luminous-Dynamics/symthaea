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
    DuplicateNode,
    DuplicateEdge,
    Relabeled,
    ConfidenceDrift,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterlinguaMetrics {
    pub schema_version: u32,
    pub expected_graph_hash: String,
    pub observed_graph_hash: String,
    pub expected_structural_hash: String,
    pub observed_structural_hash: String,
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

/// Content hash of the canonical grounded structure. Collection ordering,
/// transport-local identifiers, and lexical labels do not affect this value.
/// Confidence is intentionally excluded because it is measured separately.
pub fn structural_hash(graph: &GroundedConceptGraph) -> Result<String, String> {
    let canonical = (
        canonical_node_multiset_keys(graph),
        canonical_edges(graph),
    );
    let bytes = serde_json::to_vec(&canonical).map_err(|error| error.to_string())?;
    Ok(content_hash(&bytes))
}

/// Compare graphs by grounded structure, not collection order or lexical labels.
/// Node identifiers are treated as transport-local names.
pub fn compare_graphs(
    expected: &GroundedConceptGraph,
    observed: &GroundedConceptGraph,
) -> Result<InterlinguaMetrics, String> {
    let expected_nodes = canonical_nodes(expected);
    let observed_nodes = canonical_nodes(observed);
    let expected_edges = canonical_edges(expected);
    let observed_edges = canonical_edges(observed);

    let node_intersection = expected_nodes
        .iter()
        .map(|(key, expected_confidences)| {
            observed_nodes
                .get(key)
                .map(|observed_confidences| {
                    expected_confidences
                        .len()
                        .min(observed_confidences.len())
                })
                .unwrap_or(0)
        })
        .sum::<usize>();
    let edge_intersection = multiset_intersection_len(&expected_edges, &observed_edges);

    let metrics = InterlinguaMetrics {
        schema_version: INTERLINGUA_BENCHMARK_SCHEMA_VERSION,
        expected_graph_hash: graph_hash(expected)?,
        observed_graph_hash: graph_hash(observed)?,
        expected_structural_hash: structural_hash(expected)?,
        observed_structural_hash: structural_hash(observed)?,
        node_precision: ratio(node_intersection, observed.nodes.len()),
        node_recall: ratio(node_intersection, expected.nodes.len()),
        edge_precision: ratio(edge_intersection, observed_edges.len()),
        edge_recall: ratio(edge_intersection, expected_edges.len()),
        confidence_mae: confidence_mae(&expected_nodes, &observed_nodes),
        structural_equivalence: node_multisets_equal(&expected_nodes, &observed_nodes)
            && multiset_edges_equal(&expected_edges, &observed_edges),
        expected_bytes: serde_json::to_vec(expected)
            .map_err(|error| error.to_string())?
            .len(),
        observed_bytes: serde_json::to_vec(observed)
            .map_err(|error| error.to_string())?
            .len(),
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

pub fn relabel_nodes(graph: &GroundedConceptGraph, suffix: &str) -> GroundedConceptGraph {
    let mut relabeled = graph.clone();
    for node in &mut relabeled.nodes {
        node.label = node.label.as_ref().map(|label| format!("{label}{suffix}"));
    }
    relabeled
}

pub fn adjust_confidence(graph: &GroundedConceptGraph, delta: f32) -> GroundedConceptGraph {
    let mut adjusted = graph.clone();
    for node in &mut adjusted.nodes {
        node.confidence = (node.confidence + delta).clamp(0.0, 1.0);
    }
    adjusted
}

pub fn duplicate_last_node(graph: &GroundedConceptGraph) -> GroundedConceptGraph {
    let mut duplicated = graph.clone();
    if let Some(node) = duplicated.nodes.last().cloned() {
        duplicated.nodes.push(node);
    }
    duplicated
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

fn canonical_node_multiset_keys(graph: &GroundedConceptGraph) -> Vec<String> {
    let mut keys = graph
        .nodes
        .iter()
        .map(canonical_node_key)
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn canonical_nodes(graph: &GroundedConceptGraph) -> BTreeMap<String, Vec<f32>> {
    let mut nodes = BTreeMap::<String, Vec<f32>>::new();
    for node in &graph.nodes {
        nodes
            .entry(canonical_node_key(node))
            .or_default()
            .push(node.confidence);
    }
    for confidences in nodes.values_mut() {
        confidences.sort_by(f32::total_cmp);
    }
    nodes
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

fn node_multisets_equal(
    expected: &BTreeMap<String, Vec<f32>>,
    observed: &BTreeMap<String, Vec<f32>>,
) -> bool {
    expected
        .iter()
        .all(|(key, expected_confidences)| {
            observed
                .get(key)
                .map(|observed_confidences| observed_confidences.len() == expected_confidences.len())
                .unwrap_or(false)
        })
        && expected.len() == observed.len()
}

fn multiset_edges_equal(
    expected: &[(String, String, String)],
    observed: &[(String, String, String)],
) -> bool {
    expected == observed
}

fn multiset_intersection_len<T: Ord>(expected: &[T], observed: &[T]) -> usize {
    let mut expected_index = 0;
    let mut observed_index = 0;
    let mut intersection = 0;

    while expected_index < expected.len() && observed_index < observed.len() {
        match expected[expected_index].cmp(&observed[observed_index]) {
            std::cmp::Ordering::Less => expected_index += 1,
            std::cmp::Ordering::Greater => observed_index += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                expected_index += 1;
                observed_index += 1;
            }
        }
    }

    intersection
}

fn canonical_node_key(node: &ConceptNode) -> String {
    let mut grounded = node.grounded_by.clone();
    grounded.sort();
    let kind =
        serde_json::to_string(&node.kind).unwrap_or_else(|_| format!("{:?}", node.kind));
    format!("{kind}|{}", grounded.join(","))
}

fn confidence_mae(expected: &BTreeMap<String, Vec<f32>>, observed: &BTreeMap<String, Vec<f32>>) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for (key, expected_confidences) in expected {
        if let Some(observed_confidences) = observed.get(key) {
            for (expected_confidence, observed_confidence) in
                expected_confidences.iter().zip(observed_confidences)
            {
                total += (*expected_confidence as f64 - *observed_confidence as f64).abs();
                count += 1;
            }
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
        assert_eq!(
            metrics.expected_structural_hash,
            metrics.observed_structural_hash
        );
        assert_eq!(metrics.edge_recall, 1.0);
    }

    #[test]
    fn identifier_renaming_preserves_grounded_structure() {
        let expected = fixture();
        let observed = rename_identifiers(&expected, "node-");
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.structural_equivalence);
        assert_eq!(
            metrics.expected_structural_hash,
            metrics.observed_structural_hash
        );
    }

    #[test]
    fn structural_hash_detects_duplicate_node() {
        let expected = fixture();
        let observed = duplicate_last_node(&expected);
        let expected_hash = structural_hash(&expected).unwrap();
        let observed_hash = structural_hash(&observed).unwrap();
        assert_ne!(expected_hash, observed_hash);
    }

    #[test]
    fn structural_hash_detects_missing_edge() {
        let expected = fixture();
        let observed = drop_last_edge(&expected);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert_ne!(
            metrics.expected_structural_hash,
            metrics.observed_structural_hash
        );
    }

    #[test]
    fn lexical_relabeling_is_structurally_neutral() {
        let expected = fixture();
        let observed = relabel_nodes(&expected, " (paraphrase)");
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.structural_equivalence);
    }

    #[test]
    fn structural_hash_ignores_confidence() {
        let expected = fixture();
        let observed = adjust_confidence(&expected, 0.05);
        assert_eq!(
            structural_hash(&expected).unwrap(),
            structural_hash(&observed).unwrap()
        );
    }

    #[test]
    fn confidence_drift_is_reported_separately_from_structure() {
        let expected = fixture();
        let observed = adjust_confidence(&expected, 0.05);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.structural_equivalence);
        assert!(metrics.confidence_mae > 0.0);
    }

    #[test]
    fn duplicate_node_does_not_hide_precision_loss() {
        let expected = fixture();
        let observed = duplicate_last_node(&expected);
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(!metrics.structural_equivalence);
        assert!(metrics.node_precision < 1.0);
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
    fn duplicate_expected_edge_does_not_exceed_metric_bounds() {
        let mut expected = fixture();
        if let Some(edge) = expected.edges.last().cloned() {
            expected.edges.push(edge);
        }
        let observed = fixture();
        let metrics = compare_graphs(&expected, &observed).unwrap();
        assert!(metrics.edge_recall < 1.0);
        assert!((0.0..=1.0).contains(&metrics.edge_precision));
        assert!((0.0..=1.0).contains(&metrics.edge_recall));
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
