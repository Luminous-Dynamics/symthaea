// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Claim Inference Engine
//!
//! Automatically derives new knowledge from claim relationships.
//! Implements transitive inference, contradiction detection,
//! and missing link suggestions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

use crate::claims::ClaimRelationType;

/// A node in the claim graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimNode {
    pub id: Uuid,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
}

/// An edge in the claim graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub relation_type: ClaimRelationType,
    pub strength: f64,
    pub created_at: DateTime<Utc>,
}

/// The claim knowledge graph
#[derive(Debug, Clone, Default)]
pub struct ClaimGraph {
    nodes: HashMap<Uuid, ClaimNode>,
    edges: Vec<ClaimEdge>,
    /// Adjacency list: node -> [(neighbor, edge_index)]
    adjacency: HashMap<Uuid, Vec<(Uuid, usize)>>,
    /// Reverse adjacency: node -> [(source, edge_index)]
    reverse_adjacency: HashMap<Uuid, Vec<(Uuid, usize)>>,
}

impl ClaimGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a claim node
    pub fn add_node(&mut self, node: ClaimNode) {
        let id = node.id;
        self.nodes.insert(id, node);
        self.adjacency.entry(id).or_default();
        self.reverse_adjacency.entry(id).or_default();
    }

    /// Add a relationship edge
    pub fn add_edge(&mut self, edge: ClaimEdge) {
        let idx = self.edges.len();
        self.adjacency
            .entry(edge.from)
            .or_default()
            .push((edge.to, idx));
        self.reverse_adjacency
            .entry(edge.to)
            .or_default()
            .push((edge.from, idx));
        self.edges.push(edge);
    }

    /// Get all outgoing edges from a node
    pub fn outgoing(&self, node: Uuid) -> Vec<&ClaimEdge> {
        self.adjacency
            .get(&node)
            .map(|neighbors| neighbors.iter().map(|(_, idx)| &self.edges[*idx]).collect())
            .unwrap_or_default()
    }

    /// Get all incoming edges to a node
    pub fn incoming(&self, node: Uuid) -> Vec<&ClaimEdge> {
        self.reverse_adjacency
            .get(&node)
            .map(|sources| sources.iter().map(|(_, idx)| &self.edges[*idx]).collect())
            .unwrap_or_default()
    }

    /// Get node by ID
    pub fn get_node(&self, id: Uuid) -> Option<&ClaimNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &ClaimNode> {
        self.nodes.values()
    }

    /// Get all edges
    pub fn edges(&self) -> &[ClaimEdge] {
        &self.edges
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// An inferred relationship between claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredRelation {
    pub from: Uuid,
    pub to: Uuid,
    pub relation_type: ClaimRelationType,
    pub inferred_strength: f64,
    /// The path through which this was inferred
    pub inference_path: Vec<Uuid>,
    /// Confidence in the inference itself
    pub inference_confidence: f64,
    /// Rule that generated this inference
    pub inference_rule: InferenceRule,
}

/// Rules for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceRule {
    /// A supports B, B supports C => A transitively supports C
    TransitiveSupport,
    /// A refutes B, B supports C => A indirectly undermines C
    RefutationPropagation,
    /// A depends on B, B is retracted => A is undermined
    DependencyChain,
    /// A and B both support C strongly => A and B are likely related
    CommonTarget,
    /// A cites B, B cites C => A may benefit from citing C
    CitationChain,
}

impl InferenceRule {
    /// Decay factor for this rule type
    pub fn decay_factor(&self) -> f64 {
        match self {
            Self::TransitiveSupport => 0.8,
            Self::RefutationPropagation => 0.7,
            Self::DependencyChain => 0.9,
            Self::CommonTarget => 0.6,
            Self::CitationChain => 0.5,
        }
    }
}

/// A detected contradiction in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionReport {
    pub claim_a: Uuid,
    pub claim_b: Uuid,
    pub contradiction_type: ContradictionType,
    pub severity: f64,
    pub evidence_path: Vec<Uuid>,
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContradictionType {
    /// Direct refutation relationship
    DirectRefutation,
    /// A supports X, B refutes X
    IndirectConflict,
    /// Circular dependency
    CircularDependency,
    /// Mutual exclusion (A depends on NOT B, B depends on NOT A)
    MutualExclusion,
    /// Inconsistent support (A supports B and refutes B)
    InconsistentRelation,
}

/// A suggested missing relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedRelation {
    pub from: Uuid,
    pub to: Uuid,
    pub suggested_type: ClaimRelationType,
    pub confidence: f64,
    pub reasoning: SuggestionReasoning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionReasoning {
    /// Both cite same sources
    SharedCitations,
    /// Similar verification patterns
    SimilarVerifiers,
    /// Topological proximity
    GraphProximity,
    /// Semantic similarity (would need external input)
    SemanticSimilarity,
    /// Common dependents
    CommonDependents,
}

/// The inference engine
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    /// Maximum path length for transitive inference
    pub max_path_length: usize,
    /// Minimum strength to consider for inference
    pub min_strength_threshold: f64,
    /// Decay factor per hop
    pub hop_decay: f64,
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self {
            max_path_length: 4,
            min_strength_threshold: 0.3,
            hop_decay: 0.8,
        }
    }
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Derive transitive support relationships
    pub fn derive_transitive_support(&self, graph: &ClaimGraph) -> Vec<InferredRelation> {
        let mut inferred = Vec::new();
        let mut visited_pairs: HashSet<(Uuid, Uuid)> = HashSet::new();

        // For each node, BFS to find transitively supported claims
        for start_node in graph.nodes() {
            let mut queue: VecDeque<(Uuid, Vec<Uuid>, f64)> = VecDeque::new();
            queue.push_back((start_node.id, vec![start_node.id], 1.0));

            let mut seen: HashSet<Uuid> = HashSet::new();
            seen.insert(start_node.id);

            while let Some((current, path, strength)) = queue.pop_front() {
                if path.len() > self.max_path_length {
                    continue;
                }

                for edge in graph.outgoing(current) {
                    if edge.relation_type != ClaimRelationType::Supports {
                        continue;
                    }

                    let new_strength = strength * edge.strength * self.hop_decay;
                    if new_strength < self.min_strength_threshold {
                        continue;
                    }

                    let target = edge.to;

                    // Skip direct relationships (those already exist)
                    if path.len() == 1 {
                        if !seen.contains(&target) {
                            seen.insert(target);
                            let mut new_path = path.clone();
                            new_path.push(target);
                            queue.push_back((target, new_path, new_strength));
                        }
                        continue;
                    }

                    // Found a transitive relationship
                    let pair = (start_node.id, target);
                    if !visited_pairs.contains(&pair) && start_node.id != target {
                        visited_pairs.insert(pair);

                        let mut full_path = path.clone();
                        full_path.push(target);

                        inferred.push(InferredRelation {
                            from: start_node.id,
                            to: target,
                            relation_type: ClaimRelationType::Supports,
                            inferred_strength: new_strength,
                            inference_path: full_path.clone(),
                            inference_confidence: new_strength * 0.9,
                            inference_rule: InferenceRule::TransitiveSupport,
                        });
                    }

                    if !seen.contains(&target) {
                        seen.insert(target);
                        let mut new_path = path.clone();
                        new_path.push(target);
                        queue.push_back((target, new_path, new_strength));
                    }
                }
            }
        }

        inferred
    }

    /// Detect contradictions in the graph
    pub fn detect_contradictions(&self, graph: &ClaimGraph) -> Vec<ContradictionReport> {
        let mut contradictions = Vec::new();

        // Check for direct refutations
        for edge in graph.edges() {
            if edge.relation_type == ClaimRelationType::Refutes {
                contradictions.push(ContradictionReport {
                    claim_a: edge.from,
                    claim_b: edge.to,
                    contradiction_type: ContradictionType::DirectRefutation,
                    severity: edge.strength,
                    evidence_path: vec![edge.from, edge.to],
                    detected_at: Utc::now(),
                });
            }
        }

        // Check for inconsistent relations (A both supports and refutes B)
        let mut relation_map: HashMap<(Uuid, Uuid), Vec<ClaimRelationType>> = HashMap::new();
        for edge in graph.edges() {
            relation_map
                .entry((edge.from, edge.to))
                .or_default()
                .push(edge.relation_type);
        }

        for ((from, to), relations) in &relation_map {
            let has_support = relations.contains(&ClaimRelationType::Supports);
            let has_refute = relations.contains(&ClaimRelationType::Refutes);

            if has_support && has_refute {
                contradictions.push(ContradictionReport {
                    claim_a: *from,
                    claim_b: *to,
                    contradiction_type: ContradictionType::InconsistentRelation,
                    severity: 1.0,
                    evidence_path: vec![*from, *to],
                    detected_at: Utc::now(),
                });
            }
        }

        // Check for indirect conflicts (A supports X, B refutes X)
        for node in graph.nodes() {
            let incoming = graph.incoming(node.id);
            let supporters: Vec<Uuid> = incoming
                .iter()
                .filter(|e| e.relation_type == ClaimRelationType::Supports)
                .map(|e| e.from)
                .collect();
            let refuters: Vec<Uuid> = incoming
                .iter()
                .filter(|e| e.relation_type == ClaimRelationType::Refutes)
                .map(|e| e.from)
                .collect();

            for supporter in &supporters {
                for refuter in &refuters {
                    if supporter != refuter {
                        contradictions.push(ContradictionReport {
                            claim_a: *supporter,
                            claim_b: *refuter,
                            contradiction_type: ContradictionType::IndirectConflict,
                            severity: 0.7,
                            evidence_path: vec![*supporter, node.id, *refuter],
                            detected_at: Utc::now(),
                        });
                    }
                }
            }
        }

        // Check for circular dependencies
        for node in graph.nodes() {
            if let Some(cycle) = self.find_cycle(graph, node.id) {
                contradictions.push(ContradictionReport {
                    claim_a: cycle[0],
                    claim_b: *cycle.last().unwrap_or(&cycle[0]),
                    contradiction_type: ContradictionType::CircularDependency,
                    severity: 0.8,
                    evidence_path: cycle,
                    detected_at: Utc::now(),
                });
            }
        }

        contradictions
    }

    /// Find a cycle starting from a node (DependsOn edges only)
    fn find_cycle(&self, graph: &ClaimGraph, start: Uuid) -> Option<Vec<Uuid>> {
        let mut visited: HashSet<Uuid> = HashSet::new();
        let mut path: Vec<Uuid> = Vec::new();
        let mut stack: Vec<(Uuid, bool)> = vec![(start, false)];

        while let Some((node, processed)) = stack.pop() {
            if processed {
                path.pop();
                continue;
            }

            if visited.contains(&node) {
                if path.contains(&node) {
                    // Found a cycle
                    let cycle_start = path.iter().position(|&n| n == node).unwrap();
                    let mut cycle: Vec<Uuid> = path[cycle_start..].to_vec();
                    cycle.push(node);
                    return Some(cycle);
                }
                continue;
            }

            visited.insert(node);
            path.push(node);
            stack.push((node, true)); // Mark for backtrack

            for edge in graph.outgoing(node) {
                if edge.relation_type == ClaimRelationType::DependsOn {
                    stack.push((edge.to, false));
                }
            }
        }

        None
    }

    /// Suggest missing relationships based on graph structure
    pub fn suggest_missing_links(&self, graph: &ClaimGraph) -> Vec<SuggestedRelation> {
        let mut suggestions = Vec::new();

        // Find claims that cite the same sources (SharedCitations)
        let mut citation_targets: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for edge in graph.edges() {
            if edge.relation_type == ClaimRelationType::Cites {
                citation_targets.entry(edge.to).or_default().push(edge.from);
            }
        }

        for (_target, citers) in &citation_targets {
            if citers.len() >= 2 {
                for i in 0..citers.len() {
                    for j in (i + 1)..citers.len() {
                        let a = citers[i];
                        let b = citers[j];

                        // Check if they already have a relationship
                        let has_relation = graph
                            .outgoing(a)
                            .iter()
                            .any(|e| e.to == b)
                            || graph.outgoing(b).iter().any(|e| e.to == a);

                        if !has_relation {
                            suggestions.push(SuggestedRelation {
                                from: a,
                                to: b,
                                suggested_type: ClaimRelationType::Supports,
                                confidence: 0.5,
                                reasoning: SuggestionReasoning::SharedCitations,
                            });
                        }
                    }
                }
            }
        }

        // Find claims with common dependents (CommonDependents)
        let mut dependent_sources: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for edge in graph.edges() {
            if edge.relation_type == ClaimRelationType::DependsOn {
                dependent_sources.entry(edge.from).or_default().push(edge.to);
            }
        }

        // Claims that are depended upon by the same claims
        let mut depended_by: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for edge in graph.edges() {
            if edge.relation_type == ClaimRelationType::DependsOn {
                depended_by.entry(edge.to).or_default().push(edge.from);
            }
        }

        for (_dependent, sources) in &dependent_sources {
            if sources.len() >= 2 {
                for i in 0..sources.len() {
                    for j in (i + 1)..sources.len() {
                        let a = sources[i];
                        let b = sources[j];

                        let has_relation = graph
                            .outgoing(a)
                            .iter()
                            .any(|e| e.to == b)
                            || graph.outgoing(b).iter().any(|e| e.to == a);

                        if !has_relation {
                            suggestions.push(SuggestedRelation {
                                from: a,
                                to: b,
                                suggested_type: ClaimRelationType::Supports,
                                confidence: 0.4,
                                reasoning: SuggestionReasoning::CommonDependents,
                            });
                        }
                    }
                }
            }
        }

        // Deduplicate suggestions
        suggestions.sort_by(|a, b| {
            (a.from, a.to)
                .partial_cmp(&(b.from, b.to))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.dedup_by(|a, b| a.from == b.from && a.to == b.to);

        suggestions
    }

    /// Get summary statistics about inferences
    pub fn inference_summary(&self, graph: &ClaimGraph) -> InferenceSummary {
        let transitive = self.derive_transitive_support(graph);
        let contradictions = self.detect_contradictions(graph);
        let suggestions = self.suggest_missing_links(graph);

        InferenceSummary {
            total_nodes: graph.node_count(),
            total_edges: graph.edge_count(),
            transitive_relations: transitive.len(),
            contradictions_found: contradictions.len(),
            suggested_links: suggestions.len(),
            graph_density: if graph.node_count() > 1 {
                graph.edge_count() as f64
                    / (graph.node_count() * (graph.node_count() - 1)) as f64
            } else {
                0.0
            },
        }
    }
}

/// Summary of inference analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSummary {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub transitive_relations: usize,
    pub contradictions_found: usize,
    pub suggested_links: usize,
    pub graph_density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_node(id: Uuid) -> ClaimNode {
        ClaimNode {
            id,
            confidence: 0.8,
            created_at: Utc::now(),
        }
    }

    fn create_edge(from: Uuid, to: Uuid, relation: ClaimRelationType) -> ClaimEdge {
        ClaimEdge {
            from,
            to,
            relation_type: relation,
            strength: 0.9,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_transitive_support() {
        let mut graph = ClaimGraph::new();

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        graph.add_node(create_node(a));
        graph.add_node(create_node(b));
        graph.add_node(create_node(c));

        // A supports B, B supports C
        graph.add_edge(create_edge(a, b, ClaimRelationType::Supports));
        graph.add_edge(create_edge(b, c, ClaimRelationType::Supports));

        let engine = InferenceEngine::new();
        let inferred = engine.derive_transitive_support(&graph);

        // Should infer A transitively supports C
        assert!(inferred.iter().any(|r| r.from == a && r.to == c));
    }

    #[test]
    fn test_detect_direct_refutation() {
        let mut graph = ClaimGraph::new();

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        graph.add_node(create_node(a));
        graph.add_node(create_node(b));

        graph.add_edge(create_edge(a, b, ClaimRelationType::Refutes));

        let engine = InferenceEngine::new();
        let contradictions = engine.detect_contradictions(&graph);

        assert!(!contradictions.is_empty());
        assert!(contradictions
            .iter()
            .any(|c| c.contradiction_type == ContradictionType::DirectRefutation));
    }

    #[test]
    fn test_detect_inconsistent_relation() {
        let mut graph = ClaimGraph::new();

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        graph.add_node(create_node(a));
        graph.add_node(create_node(b));

        // A both supports and refutes B - inconsistent!
        graph.add_edge(create_edge(a, b, ClaimRelationType::Supports));
        graph.add_edge(create_edge(a, b, ClaimRelationType::Refutes));

        let engine = InferenceEngine::new();
        let contradictions = engine.detect_contradictions(&graph);

        assert!(contradictions
            .iter()
            .any(|c| c.contradiction_type == ContradictionType::InconsistentRelation));
    }

    #[test]
    fn test_suggest_shared_citations() {
        let mut graph = ClaimGraph::new();

        let source = Uuid::new_v4();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        graph.add_node(create_node(source));
        graph.add_node(create_node(a));
        graph.add_node(create_node(b));

        // Both A and B cite Source
        graph.add_edge(create_edge(a, source, ClaimRelationType::Cites));
        graph.add_edge(create_edge(b, source, ClaimRelationType::Cites));

        let engine = InferenceEngine::new();
        let suggestions = engine.suggest_missing_links(&graph);

        // Should suggest A and B are related
        assert!(suggestions
            .iter()
            .any(|s| (s.from == a && s.to == b) || (s.from == b && s.to == a)));
    }

    #[test]
    fn test_inference_summary() {
        let mut graph = ClaimGraph::new();

        for _ in 0..5 {
            graph.add_node(create_node(Uuid::new_v4()));
        }

        let engine = InferenceEngine::new();
        let summary = engine.inference_summary(&graph);

        assert_eq!(summary.total_nodes, 5);
        assert_eq!(summary.total_edges, 0);
    }
}
