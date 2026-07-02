// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Citation Graph Analysis
//!
//! Implements PageRank-style importance scoring for claims based on their
//! citation network, enabling identification of foundational vs derivative work.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Citation importance metrics for a claim
#[derive(Debug, Clone)]
pub struct CitationMetrics {
    /// PageRank-style importance score (0.0-1.0)
    pub importance_score: f64,
    /// Number of incoming citations
    pub citation_count: usize,
    /// Number of outgoing references
    pub reference_count: usize,
    /// H-index for this claim's citation network
    pub h_index: usize,
    /// Whether this is a foundational claim (high citations, low references)
    pub is_foundational: bool,
    /// Citation depth (max distance to any citing claim)
    pub citation_depth: usize,
    /// Normalized influence score
    pub influence_score: f64,
}

impl Default for CitationMetrics {
    fn default() -> Self {
        Self {
            importance_score: 0.0,
            citation_count: 0,
            reference_count: 0,
            h_index: 0,
            is_foundational: false,
            citation_depth: 0,
            influence_score: 0.0,
        }
    }
}

/// A citation edge in the graph
#[derive(Debug, Clone)]
pub struct Citation {
    /// The citing claim
    pub citing_claim: Uuid,
    /// The cited claim
    pub cited_claim: Uuid,
    /// Citation strength/weight (0.0-1.0)
    pub strength: f64,
    /// Citation context/type
    pub citation_type: CitationType,
    /// When the citation was made
    pub timestamp: i64,
}

/// Types of citations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CitationType {
    /// Direct supporting evidence
    Supporting,
    /// Extends or builds upon
    Extending,
    /// Contradicts or challenges
    Contradicting,
    /// Neutral reference for context
    Contextual,
    /// Methodological reference
    Methodological,
    /// Data source
    DataSource,
}

impl CitationType {
    /// Get weight modifier for this citation type
    pub fn weight_modifier(&self) -> f64 {
        match self {
            CitationType::Supporting => 1.0,
            CitationType::Extending => 0.9,
            CitationType::DataSource => 0.85,
            CitationType::Methodological => 0.8,
            CitationType::Contextual => 0.5,
            CitationType::Contradicting => 0.3, // Still counts but less
        }
    }
}

/// Citation graph for PageRank analysis
#[derive(Debug, Clone)]
pub struct CitationGraph {
    /// All claims in the graph
    claims: HashSet<Uuid>,
    /// Citations (citing -> cited)
    citations: Vec<Citation>,
    /// Outgoing citations per claim
    outgoing: HashMap<Uuid, Vec<usize>>,
    /// Incoming citations per claim
    incoming: HashMap<Uuid, Vec<usize>>,
}

impl CitationGraph {
    /// Create a new empty citation graph
    pub fn new() -> Self {
        Self {
            claims: HashSet::new(),
            citations: Vec::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        }
    }

    /// Add a claim to the graph
    pub fn add_claim(&mut self, claim_id: Uuid) {
        self.claims.insert(claim_id);
        self.outgoing.entry(claim_id).or_default();
        self.incoming.entry(claim_id).or_default();
    }

    /// Add a citation between claims
    pub fn add_citation(&mut self, citation: Citation) {
        // Ensure both claims exist
        self.add_claim(citation.citing_claim);
        self.add_claim(citation.cited_claim);

        let idx = self.citations.len();
        self.citations.push(citation.clone());

        self.outgoing
            .entry(citation.citing_claim)
            .or_default()
            .push(idx);
        self.incoming
            .entry(citation.cited_claim)
            .or_default()
            .push(idx);
    }

    /// Get number of claims in the graph
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    /// Get number of citations in the graph
    pub fn citation_count(&self) -> usize {
        self.citations.len()
    }

    /// Get all citations for a claim (as cited)
    pub fn get_incoming_citations(&self, claim_id: Uuid) -> Vec<&Citation> {
        self.incoming
            .get(&claim_id)
            .map(|indices| indices.iter().map(|&i| &self.citations[i]).collect())
            .unwrap_or_default()
    }

    /// Get all references from a claim (as citing)
    pub fn get_outgoing_citations(&self, claim_id: Uuid) -> Vec<&Citation> {
        self.outgoing
            .get(&claim_id)
            .map(|indices| indices.iter().map(|&i| &self.citations[i]).collect())
            .unwrap_or_default()
    }
}

impl Default for CitationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for PageRank computation
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (typically 0.85)
    pub damping_factor: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Whether to weight by citation type
    pub use_citation_weights: bool,
    /// Time decay factor for older citations
    pub time_decay_factor: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            use_citation_weights: true,
            time_decay_factor: 0.0, // No decay by default
        }
    }
}

/// Citation analysis engine
pub struct CitationAnalyzer {
    config: PageRankConfig,
}

impl CitationAnalyzer {
    /// Create a new citation analyzer with default config
    pub fn new() -> Self {
        Self {
            config: PageRankConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: PageRankConfig) -> Self {
        Self { config }
    }

    /// Compute PageRank scores for all claims
    pub fn compute_pagerank(&self, graph: &CitationGraph) -> HashMap<Uuid, f64> {
        let n = graph.claim_count();
        if n == 0 {
            return HashMap::new();
        }

        let claims: Vec<Uuid> = graph.claims.iter().cloned().collect();
        let initial_score = 1.0 / n as f64;

        // Initialize scores
        let mut scores: HashMap<Uuid, f64> = claims.iter().map(|&id| (id, initial_score)).collect();

        // Iterative PageRank computation
        for _ in 0..self.config.max_iterations {
            let mut new_scores: HashMap<Uuid, f64> = HashMap::new();
            let mut max_diff = 0.0f64;

            for &claim_id in &claims {
                let incoming = graph.get_incoming_citations(claim_id);

                // Sum of weighted contributions from citing claims
                let mut sum = 0.0;
                for citation in incoming {
                    let citing_score = scores.get(&citation.citing_claim).copied().unwrap_or(0.0);
                    let out_count = graph
                        .outgoing
                        .get(&citation.citing_claim)
                        .map(|v| v.len())
                        .unwrap_or(1)
                        .max(1) as f64;

                    let weight = if self.config.use_citation_weights {
                        citation.strength * citation.citation_type.weight_modifier()
                    } else {
                        1.0
                    };

                    sum += (citing_score / out_count) * weight;
                }

                // Apply damping factor
                let new_score =
                    (1.0 - self.config.damping_factor) / n as f64 + self.config.damping_factor * sum;

                let old_score = scores.get(&claim_id).copied().unwrap_or(0.0);
                max_diff = max_diff.max((new_score - old_score).abs());
                new_scores.insert(claim_id, new_score);
            }

            scores = new_scores;

            // Check convergence
            if max_diff < self.config.convergence_threshold {
                break;
            }
        }

        // Normalize scores to 0-1 range
        let max_score = scores.values().cloned().fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }

        scores
    }

    /// Compute H-index for a claim based on its citation network
    pub fn compute_h_index(&self, graph: &CitationGraph, claim_id: Uuid) -> usize {
        let incoming = graph.get_incoming_citations(claim_id);
        if incoming.is_empty() {
            return 0;
        }

        // Get citation counts for each citing claim
        let mut citation_counts: Vec<usize> = incoming
            .iter()
            .map(|c| graph.get_incoming_citations(c.citing_claim).len())
            .collect();

        citation_counts.sort_by(|a, b| b.cmp(a));

        // Find h-index
        let mut h = 0;
        for (i, &count) in citation_counts.iter().enumerate() {
            if count >= i + 1 {
                h = i + 1;
            } else {
                break;
            }
        }

        h
    }

    /// Compute citation depth (max path length from any citing claim)
    pub fn compute_citation_depth(&self, graph: &CitationGraph, claim_id: Uuid) -> usize {
        let mut visited = HashSet::new();
        let mut max_depth = 0;

        fn dfs(
            graph: &CitationGraph,
            claim_id: Uuid,
            visited: &mut HashSet<Uuid>,
            current_depth: usize,
            max_depth: &mut usize,
        ) {
            if visited.contains(&claim_id) {
                return;
            }
            visited.insert(claim_id);
            *max_depth = (*max_depth).max(current_depth);

            for citation in graph.get_incoming_citations(claim_id) {
                dfs(
                    graph,
                    citation.citing_claim,
                    visited,
                    current_depth + 1,
                    max_depth,
                );
            }
        }

        dfs(graph, claim_id, &mut visited, 0, &mut max_depth);
        max_depth
    }

    /// Compute full metrics for a claim
    pub fn compute_metrics(
        &self,
        graph: &CitationGraph,
        claim_id: Uuid,
        pagerank_scores: &HashMap<Uuid, f64>,
    ) -> CitationMetrics {
        let incoming = graph.get_incoming_citations(claim_id);
        let outgoing = graph.get_outgoing_citations(claim_id);

        let citation_count = incoming.len();
        let reference_count = outgoing.len();

        // Foundational = many citations, few references
        let is_foundational = citation_count > 5 && reference_count <= 2;

        let importance_score = pagerank_scores.get(&claim_id).copied().unwrap_or(0.0);
        let h_index = self.compute_h_index(graph, claim_id);
        let citation_depth = self.compute_citation_depth(graph, claim_id);

        // Influence score combines multiple factors
        let influence_score = if citation_count > 0 {
            let avg_strength: f64 =
                incoming.iter().map(|c| c.strength).sum::<f64>() / citation_count as f64;
            (importance_score * 0.4)
                + (avg_strength * 0.3)
                + ((h_index as f64 / 10.0).min(1.0) * 0.2)
                + ((citation_depth as f64 / 5.0).min(1.0) * 0.1)
        } else {
            0.0
        };

        CitationMetrics {
            importance_score,
            citation_count,
            reference_count,
            h_index,
            is_foundational,
            citation_depth,
            influence_score,
        }
    }

    /// Find foundational claims in the graph
    pub fn find_foundational_claims(&self, graph: &CitationGraph) -> Vec<(Uuid, CitationMetrics)> {
        let pagerank = self.compute_pagerank(graph);

        let mut results: Vec<(Uuid, CitationMetrics)> = graph
            .claims
            .iter()
            .map(|&id| {
                let metrics = self.compute_metrics(graph, id, &pagerank);
                (id, metrics)
            })
            .filter(|(_, m)| m.is_foundational || m.importance_score > 0.7)
            .collect();

        results.sort_by(|a, b| {
            b.1.importance_score
                .partial_cmp(&a.1.importance_score)
                .unwrap()
        });

        results
    }

    /// Find claims with high influence but few direct citations (hidden gems)
    pub fn find_hidden_gems(&self, graph: &CitationGraph) -> Vec<(Uuid, CitationMetrics)> {
        let pagerank = self.compute_pagerank(graph);

        let mut results: Vec<(Uuid, CitationMetrics)> = graph
            .claims
            .iter()
            .map(|&id| {
                let metrics = self.compute_metrics(graph, id, &pagerank);
                (id, metrics)
            })
            .filter(|(_, m)| {
                m.citation_count < 5 && m.citation_depth > 2 && m.importance_score > 0.3
            })
            .collect();

        results.sort_by(|a, b| {
            b.1.influence_score
                .partial_cmp(&a.1.influence_score)
                .unwrap()
        });

        results
    }

    /// Detect citation clusters (groups of mutually citing claims)
    pub fn detect_clusters(&self, graph: &CitationGraph, min_size: usize) -> Vec<Vec<Uuid>> {
        let mut clusters: Vec<Vec<Uuid>> = Vec::new();
        let mut visited: HashSet<Uuid> = HashSet::new();

        for &claim_id in &graph.claims {
            if visited.contains(&claim_id) {
                continue;
            }

            // BFS to find connected component
            let mut cluster = Vec::new();
            let mut queue = vec![claim_id];

            while let Some(current) = queue.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);
                cluster.push(current);

                // Add connected claims
                for citation in graph.get_incoming_citations(current) {
                    if !visited.contains(&citation.citing_claim) {
                        queue.push(citation.citing_claim);
                    }
                }
                for citation in graph.get_outgoing_citations(current) {
                    if !visited.contains(&citation.cited_claim) {
                        queue.push(citation.cited_claim);
                    }
                }
            }

            if cluster.len() >= min_size {
                clusters.push(cluster);
            }
        }

        clusters.sort_by(|a, b| b.len().cmp(&a.len()));
        clusters
    }
}

impl Default for CitationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CitationGraph {
        let mut graph = CitationGraph::new();

        // Create a citation network:
        // A <- B <- D
        // A <- C <- D
        // A is foundational (cited by B and C)
        // D is derivative (cites B and C)

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();

        graph.add_claim(a);
        graph.add_claim(b);
        graph.add_claim(c);
        graph.add_claim(d);

        // B cites A
        graph.add_citation(Citation {
            citing_claim: b,
            cited_claim: a,
            strength: 0.9,
            citation_type: CitationType::Supporting,
            timestamp: 1000,
        });

        // C cites A
        graph.add_citation(Citation {
            citing_claim: c,
            cited_claim: a,
            strength: 0.8,
            citation_type: CitationType::Extending,
            timestamp: 2000,
        });

        // D cites B
        graph.add_citation(Citation {
            citing_claim: d,
            cited_claim: b,
            strength: 0.7,
            citation_type: CitationType::Supporting,
            timestamp: 3000,
        });

        // D cites C
        graph.add_citation(Citation {
            citing_claim: d,
            cited_claim: c,
            strength: 0.6,
            citation_type: CitationType::Contextual,
            timestamp: 3000,
        });

        graph
    }

    #[test]
    fn test_graph_construction() {
        let graph = create_test_graph();
        assert_eq!(graph.claim_count(), 4);
        assert_eq!(graph.citation_count(), 4);
    }

    #[test]
    fn test_pagerank_computation() {
        let graph = create_test_graph();
        let analyzer = CitationAnalyzer::new();

        let scores = analyzer.compute_pagerank(&graph);

        // Should have scores for all claims
        assert_eq!(scores.len(), 4);

        // All scores should be between 0 and 1
        for score in scores.values() {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_citation_metrics() {
        let graph = create_test_graph();
        let analyzer = CitationAnalyzer::new();
        let pagerank = analyzer.compute_pagerank(&graph);

        // Find the claim with most incoming citations
        let mut best_claim = None;
        let mut best_count = 0;

        for &claim_id in &graph.claims {
            let metrics = analyzer.compute_metrics(&graph, claim_id, &pagerank);
            if metrics.citation_count > best_count {
                best_count = metrics.citation_count;
                best_claim = Some((claim_id, metrics));
            }
        }

        assert!(best_claim.is_some());
        let (_, metrics) = best_claim.unwrap();
        assert_eq!(metrics.citation_count, 2); // A has 2 citations
    }

    #[test]
    fn test_cluster_detection() {
        let graph = create_test_graph();
        let analyzer = CitationAnalyzer::new();

        let clusters = analyzer.detect_clusters(&graph, 2);

        // All claims should be in one cluster (connected)
        assert!(!clusters.is_empty());
        assert_eq!(clusters[0].len(), 4);
    }

    #[test]
    fn test_citation_depth() {
        let graph = create_test_graph();
        let analyzer = CitationAnalyzer::new();

        // Find claim A (the one with 2 incoming citations)
        for &claim_id in &graph.claims {
            let incoming = graph.get_incoming_citations(claim_id);
            if incoming.len() == 2 {
                let depth = analyzer.compute_citation_depth(&graph, claim_id);
                // A has depth 2 (D -> B -> A or D -> C -> A)
                assert!(depth >= 1);
                break;
            }
        }
    }
}
