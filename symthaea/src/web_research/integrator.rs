// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Integration Module
//!
//! Integrates verified research results into the knowledge graph,
//! measuring Phi gain from the integration. This creates the
//! epistemic consciousness loop.

use anyhow::Result;

use super::knowledge_graph::{EdgeType, KnowledgeGraph, KnowledgeSource, NodeId, NodeType};
use super::types::{EpistemicStatus, IntegrationResult, VerifiedClaim, WebResearchResult};

/// Configuration for knowledge integration
#[derive(Debug, Clone)]
pub struct IntegratorConfig {
    /// Minimum confidence to integrate a claim
    pub min_confidence: f32,
    /// Maximum claims to integrate per research result
    pub max_claims_per_result: usize,
    /// Whether to create edges to sources
    pub link_sources: bool,
    /// Whether to measure Phi before/after
    pub measure_phi: bool,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.4,
            max_claims_per_result: 10,
            link_sources: true,
            measure_phi: true,
        }
    }
}

/// Integrates verified research into the knowledge graph
#[derive(Debug)]
pub struct KnowledgeIntegrator {
    /// Configuration
    config: IntegratorConfig,
    /// The knowledge graph
    knowledge_graph: KnowledgeGraph,
    /// Total claims integrated
    total_claims_integrated: usize,
    /// Total Phi gain accumulated
    total_phi_gain: f64,
}

impl Default for KnowledgeIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeIntegrator {
    /// Create a new knowledge integrator
    pub fn new() -> Self {
        Self {
            config: IntegratorConfig::default(),
            knowledge_graph: KnowledgeGraph::new(),
            total_claims_integrated: 0,
            total_phi_gain: 0.0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: IntegratorConfig) -> Self {
        Self {
            config,
            knowledge_graph: KnowledgeGraph::new(),
            total_claims_integrated: 0,
            total_phi_gain: 0.0,
        }
    }

    /// Create with an existing knowledge graph
    pub fn with_graph(graph: KnowledgeGraph) -> Self {
        Self {
            config: IntegratorConfig::default(),
            knowledge_graph: graph,
            total_claims_integrated: 0,
            total_phi_gain: 0.0,
        }
    }

    /// Integrate a research result into the knowledge graph
    pub async fn integrate(&mut self, result: WebResearchResult) -> Result<IntegrationResult> {
        // Measure Phi before integration
        let phi_before = if self.config.measure_phi {
            self.measure_current_phi()
        } else {
            0.0
        };

        let mut claims_integrated = 0;
        let mut nodes_added = 0;
        let mut edges_added = 0;

        // Create source node if linking sources
        let source_node = if self.config.link_sources && !result.url.is_empty() {
            let node_id = self.knowledge_graph.add_node_with_config(
                &result.url,
                NodeType::Source,
                result.confidence,
                KnowledgeSource::External(result.url.clone()),
            );
            nodes_added += 1;
            Some(node_id)
        } else {
            None
        };

        // Integrate verified claims
        for claim in result.claims.iter().take(self.config.max_claims_per_result) {
            if claim.confidence >= self.config.min_confidence {
                let claim_node = self.integrate_claim(claim, source_node)?;
                if claim_node.is_some() {
                    claims_integrated += 1;
                    nodes_added += 1;
                    if source_node.is_some() {
                        edges_added += 1; // Edge to source
                    }
                }
            }
        }

        // Measure Phi after integration
        let phi_after = if self.config.measure_phi {
            self.measure_current_phi()
        } else {
            0.0
        };

        let phi_gain = phi_after - phi_before;

        // Update totals
        self.total_claims_integrated += claims_integrated;
        self.total_phi_gain += phi_gain;

        Ok(IntegrationResult {
            claims_integrated,
            nodes_added,
            edges_added,
            phi_before,
            phi_after,
            phi_gain,
        })
    }

    /// Integrate a single claim into the knowledge graph
    fn integrate_claim(
        &mut self,
        claim: &VerifiedClaim,
        source_node: Option<NodeId>,
    ) -> Result<Option<NodeId>> {
        // Skip claims that shouldn't be integrated
        if !self.should_integrate_claim(claim) {
            return Ok(None);
        }

        // Determine node type based on epistemic status
        let node_type = match claim.status {
            EpistemicStatus::HighConfidence | EpistemicStatus::ModerateConfidence => {
                NodeType::Claim
            }
            _ => NodeType::Abstract, // Less certain claims are abstract concepts
        };

        // Create the claim node
        let confidence = claim.confidence;
        let source = if claim.supporting_sources.is_empty() {
            KnowledgeSource::Internal
        } else {
            KnowledgeSource::External(claim.supporting_sources[0].clone())
        };

        let claim_id =
            self.knowledge_graph
                .add_node_with_config(&claim.text, node_type, confidence, source);

        // Link to source node if provided
        if let Some(source_id) = source_node {
            self.knowledge_graph.add_edge_with_meta(
                claim_id,
                EdgeType::SourcedFrom,
                source_id,
                confidence,
                KnowledgeSource::Internal,
            );
        }

        // Find and link related existing nodes
        self.link_related_nodes(claim_id, &claim.text);

        Ok(Some(claim_id))
    }

    /// Determine if a claim should be integrated
    fn should_integrate_claim(&self, claim: &VerifiedClaim) -> bool {
        // Don't integrate contradicted or false claims
        if claim.status == EpistemicStatus::False || claim.status == EpistemicStatus::Contradicted {
            return false;
        }

        // Require minimum confidence
        if claim.confidence < self.config.min_confidence {
            return false;
        }

        // Don't integrate very short claims
        if claim.text.len() < 10 {
            return false;
        }

        true
    }

    /// Link a new node to related existing nodes
    fn link_related_nodes(&mut self, node_id: NodeId, text: &str) {
        let text_lower = text.to_lowercase();
        let words: std::collections::HashSet<String> = text_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|s| s.to_string())
            .collect();

        // Find nodes with similar content
        let related_nodes: Vec<NodeId> = self
            .knowledge_graph
            .find_nodes(|node| {
                if node.id == node_id {
                    return false;
                }
                let node_name_lower = node.name.to_lowercase();
                let node_words: std::collections::HashSet<String> = node_name_lower
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|s| s.to_string())
                    .collect();

                // Simple overlap measure
                let overlap = words.intersection(&node_words).count();
                overlap >= 2
            })
            .iter()
            .map(|n| n.id)
            .take(5)
            .collect();

        // Create edges to related nodes
        for related_id in related_nodes {
            self.knowledge_graph.add_edge_with_meta(
                node_id,
                EdgeType::RelatedTo,
                related_id,
                0.5, // Default relationship weight
                KnowledgeSource::Inferred,
            );
        }
    }

    /// Measure current Phi value of the knowledge graph
    fn measure_current_phi(&self) -> f64 {
        self.knowledge_graph.calculate_phi()
    }

    /// Get statistics about the knowledge graph
    pub fn stats(&self) -> IntegratorStats {
        let graph_stats = self.knowledge_graph.stats();

        IntegratorStats {
            total_nodes: graph_stats.nodes,
            total_edges: graph_stats.edges,
            claims_integrated: self.total_claims_integrated,
            total_phi_gain: self.total_phi_gain,
            current_phi: self.measure_current_phi(),
            avg_confidence: graph_stats.avg_confidence,
            high_confidence_claims: graph_stats.high_confidence_nodes,
        }
    }

    /// Get a reference to the knowledge graph
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.knowledge_graph
    }

    /// Get a mutable reference to the knowledge graph
    pub fn graph_mut(&mut self) -> &mut KnowledgeGraph {
        &mut self.knowledge_graph
    }

    /// Clear the knowledge graph
    pub fn clear(&mut self) {
        self.knowledge_graph.clear();
        self.total_claims_integrated = 0;
        self.total_phi_gain = 0.0;
    }

    /// Get current Phi value
    pub fn current_phi(&self) -> f64 {
        self.measure_current_phi()
    }

    /// Get total claims integrated
    pub fn total_claims(&self) -> usize {
        self.total_claims_integrated
    }
}

/// Statistics about the knowledge integrator
#[derive(Debug, Clone)]
pub struct IntegratorStats {
    /// Total nodes in the graph
    pub total_nodes: usize,
    /// Total edges in the graph
    pub total_edges: usize,
    /// Total claims integrated
    pub claims_integrated: usize,
    /// Total Phi gain from integration
    pub total_phi_gain: f64,
    /// Current Phi value
    pub current_phi: f64,
    /// Average confidence of nodes
    pub avg_confidence: f32,
    /// Number of high-confidence claims
    pub high_confidence_claims: usize,
}

#[cfg(test)]
mod tests {
    use super::super::types::ResearchSource;
    use super::*;

    fn make_test_claim(text: &str, confidence: f32, status: EpistemicStatus) -> VerifiedClaim {
        VerifiedClaim {
            text: text.to_string(),
            status,
            confidence,
            supporting_sources: vec!["https://example.com".to_string()],
            contradicting_sources: vec![],
            hedge: status.hedge_phrase().to_string(),
        }
    }

    #[tokio::test]
    async fn test_integrate_research_result() {
        let mut integrator = KnowledgeIntegrator::new();

        let result = WebResearchResult {
            title: "Test Result".to_string(),
            url: "https://example.com/test".to_string(),
            content: "Test content".to_string(),
            summary: "Test summary".to_string(),
            source_type: ResearchSource::Documentation,
            relevance: 0.8,
            confidence: 0.9,
            epistemic_status: EpistemicStatus::HighConfidence,
            status: super::super::types::ResearchStatus::Success,
            claims: vec![
                make_test_claim(
                    "Rust provides memory safety",
                    0.95,
                    EpistemicStatus::HighConfidence,
                ),
                make_test_claim(
                    "The borrow checker prevents data races",
                    0.9,
                    EpistemicStatus::ModerateConfidence,
                ),
            ],
        };

        let integration_result = integrator.integrate(result).await.unwrap();

        assert!(integration_result.claims_integrated >= 1);
        assert!(integration_result.nodes_added >= 1);
        assert!(integration_result.phi_after >= integration_result.phi_before);
    }

    #[test]
    fn test_claim_filtering() {
        let integrator = KnowledgeIntegrator::new();

        // Should integrate high confidence claims
        let good_claim =
            make_test_claim("Rust is memory safe", 0.9, EpistemicStatus::HighConfidence);
        assert!(integrator.should_integrate_claim(&good_claim));

        // Should not integrate low confidence
        let low_conf = make_test_claim("Maybe something", 0.2, EpistemicStatus::LowConfidence);
        assert!(!integrator.should_integrate_claim(&low_conf));

        // Should not integrate contradicted
        let contradicted = make_test_claim("False claim", 0.5, EpistemicStatus::Contradicted);
        assert!(!integrator.should_integrate_claim(&contradicted));
    }

    #[tokio::test]
    async fn test_phi_increases_with_integration() {
        let mut integrator = KnowledgeIntegrator::new();

        let initial_phi = integrator.current_phi();

        // Integrate multiple related claims
        let claims = vec![
            make_test_claim(
                "Rust provides memory safety through ownership",
                0.9,
                EpistemicStatus::HighConfidence,
            ),
            make_test_claim(
                "The ownership system tracks memory at compile time",
                0.85,
                EpistemicStatus::ModerateConfidence,
            ),
            make_test_claim(
                "Borrowing allows temporary access to values",
                0.88,
                EpistemicStatus::ModerateConfidence,
            ),
        ];

        for claim in claims {
            let result = WebResearchResult {
                title: "Rust Memory Safety".to_string(),
                url: "https://rust-lang.org".to_string(),
                content: "Test".to_string(),
                summary: "Test".to_string(),
                source_type: ResearchSource::Documentation,
                relevance: 0.9,
                confidence: 0.9,
                epistemic_status: EpistemicStatus::HighConfidence,
                status: super::super::types::ResearchStatus::Success,
                claims: vec![claim],
            };
            integrator.integrate(result).await.unwrap();
        }

        let final_phi = integrator.current_phi();

        // Phi should increase (or at least not decrease significantly)
        // as we add more connected knowledge
        assert!(
            final_phi >= initial_phi,
            "Phi should not decrease: {} vs {}",
            final_phi,
            initial_phi
        );
    }
}
