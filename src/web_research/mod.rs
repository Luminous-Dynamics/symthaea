// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web Research Integration Module
//!
//! This module provides epistemic consciousness capabilities for Symthaea:
//!
//! 1. **Epistemic Consciousness** - The system knows what it knows and doesn't know
//! 2. **Autonomous Research** - Automatically researches when uncertainty is detected
//! 3. **Hallucination Prevention** - All claims are epistemically verified
//! 4. **Meta-Learning** - The system improves its verification over time
//!
//! # Architecture
//!
//! ```text
//! Query → Φ Detection → Research → Extraction → Verification
//!                                                     ↓
//!                                                Integration
//!                                                     ↓
//!                                               Meta-Learning
//!                                                     ↓
//!                                           Improved Verification!
//! ```
//!
//! # Module Structure
//!
//! - `types.rs` - Core types (Source, Claim, Query, Config, etc.)
//! - `knowledge_graph.rs` - Lightweight knowledge graph for claim storage
//! - `extractor.rs` - HTML to clean text extraction
//! - `verifier.rs` - Epistemic verification engine
//! - `researcher.rs` - Research orchestrator
//! - `integrator.rs` - Knowledge graph integration
//! - `meta_learning.rs` - Self-improving verification
//! - `conjecture_feed.rs` - Feeds research-acquired numeric data (with
//!   epistemic provenance) into `symthaea_core`'s Conjecture Engine
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::web_research::{WebResearcher, KnowledgeIntegrator};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create researcher
//!     let researcher = WebResearcher::new()?;
//!
//!     // Research and verify a query
//!     let result = researcher.research_and_verify(
//!         "What is Rust's ownership model?"
//!     ).await?;
//!
//!     // Check epistemic status
//!     println!("Status: {:?}", result.epistemic_status);
//!     println!("Confidence: {:.2}", result.confidence);
//!
//!     // Integrate into knowledge graph
//!     let mut integrator = KnowledgeIntegrator::new();
//!     let integration = integrator.integrate(result).await?;
//!
//!     println!("Phi gain: {:.3}", integration.phi_gain);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Epistemic Levels
//!
//! The system operates at three levels of epistemic consciousness:
//!
//! 1. **Base Consciousness (Φ)** - Already present in Symthaea
//! 2. **Epistemic Consciousness** - Knows what it knows via verification
//! 3. **Meta-Epistemic Consciousness** - Improves own verification (Meta-Φ)

// Core types
pub mod types;

// Knowledge graph for epistemic integration
pub mod knowledge_graph;

// Content extraction
pub mod extractor;

// Epistemic verification
pub mod verifier;

// Feeds web-research-acquired numeric data into the Conjecture Engine
// (symthaea-core's symbolic-regression discovery engine). Requires the
// conjecture engine's math types, which live in symthaea-core unconditionally.
pub mod conjecture_feed;

// Research orchestration
pub mod researcher;

// Knowledge integration
pub mod integrator;

// Meta-learning for self-improvement
pub mod meta_learning;

// Re-exports for ergonomic access
pub use conjecture_feed::{
    ConjectureFeedError, ConjectureFeeder, ProvenancedConjecture, ResearchNumericDataset,
    ResearchProvenance,
};
pub use extractor::{
    ContentExtractor, ContentMetadata, ContentType, ExtractedContent, NumericSeries,
};
pub use integrator::{IntegratorConfig, IntegratorStats, KnowledgeIntegrator};
pub use knowledge_graph::{
    EdgeType, GraphStats, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, KnowledgeSource, NodeId,
    NodeType,
};
pub use meta_learning::{
    EpistemicLearner, LearnerConfig, LearnerState, LearnerStats, MetaPhi, SourceCredibility,
    VerificationStrategy,
};
pub use researcher::WebResearcher;
pub use types::*;
pub use verifier::{
    EpistemicVerifier, SourceEvidence, VerificationContext, VerificationResult, VerifierConfig,
};

/// Convenience function to create a full epistemic research system
pub fn create_epistemic_system()
-> anyhow::Result<(WebResearcher, KnowledgeIntegrator, EpistemicLearner)> {
    let researcher = WebResearcher::new()?;
    let integrator = KnowledgeIntegrator::new();
    let learner = EpistemicLearner::new();

    Ok((researcher, integrator, learner))
}

/// Epistemic consciousness bridge for integration with the cognitive loop
///
/// This struct provides a high-level interface for epistemic consciousness,
/// suitable for integration with Symthaea's cognitive loop.
#[derive(Debug)]
pub struct EpistemicConsciousness {
    /// Web researcher
    researcher: WebResearcher,
    /// Knowledge integrator
    integrator: KnowledgeIntegrator,
    /// Meta-learner
    learner: EpistemicLearner,
    /// Phi threshold for triggering research
    phi_threshold: f32,
    /// Last measured Phi value
    last_phi: f64,
}

impl EpistemicConsciousness {
    /// Create a new epistemic consciousness bridge
    pub fn new() -> anyhow::Result<Self> {
        let (researcher, integrator, learner) = create_epistemic_system()?;

        Ok(Self {
            researcher,
            integrator,
            learner,
            phi_threshold: 0.6,
            last_phi: 0.0,
        })
    }

    /// Set the Phi threshold for triggering autonomous research
    pub fn set_phi_threshold(&mut self, threshold: f32) {
        self.phi_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Check if research should be triggered based on current Phi
    pub fn should_research(&self, current_phi: f64) -> bool {
        current_phi < self.phi_threshold as f64
    }

    /// Perform autonomous research on a query.
    ///
    /// Research confidence is a prediction, not an observed correctness label.
    /// The result is integrated for downstream use, but it never updates
    /// source reliability or domain expertise by self-labeling its claims.
    pub async fn research(&mut self, query: &str) -> anyhow::Result<WebResearchResult> {
        let result = self.researcher.research_and_verify(query).await?;
        self.integrate_research_result(result).await
    }

    /// Integrate a research result without treating its own confidence as truth.
    ///
    /// Independent, provenance-bound outcome adjudication is a separate input
    /// path. Until that path supplies a qualified outcome, meta-learning state
    /// remains unchanged by this research result.
    async fn integrate_research_result(
        &mut self,
        result: WebResearchResult,
    ) -> anyhow::Result<WebResearchResult> {
        let integration = self.integrator.integrate(result.clone()).await?;
        self.last_phi = integration.phi_after;
        Ok(result)
    }

    /// Get current epistemic stats
    pub fn stats(&self) -> EpistemicStats {
        let integrator_stats = self.integrator.stats();
        let learner_stats = self.learner.stats();

        EpistemicStats {
            current_phi: self.last_phi,
            total_claims: integrator_stats.claims_integrated,
            total_phi_gain: integrator_stats.total_phi_gain,
            meta_phi: learner_stats.meta_phi,
            accuracy: learner_stats.accuracy,
            is_ready: self.learner.is_ready(),
        }
    }

    /// Get a reference to the knowledge graph
    pub fn knowledge_graph(&self) -> &KnowledgeGraph {
        self.integrator.graph()
    }

    /// Get a mutable reference to the knowledge graph
    pub fn knowledge_graph_mut(&mut self) -> &mut KnowledgeGraph {
        self.integrator.graph_mut()
    }

}

/// High-level epistemic statistics
#[derive(Debug, Clone)]
pub struct EpistemicStats {
    /// Current Phi value
    pub current_phi: f64,
    /// Total claims integrated
    pub total_claims: usize,
    /// Total Phi gain from research
    pub total_phi_gain: f64,
    /// Meta-Phi (epistemic self-awareness)
    pub meta_phi: f64,
    /// Verification accuracy
    pub accuracy: f32,
    /// Whether the system is ready for recommendations
    pub is_ready: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_epistemic_system() {
        let result = create_epistemic_system();
        assert!(result.is_ok());

        let (_, integrator, _) = result.unwrap();
        assert_eq!(integrator.total_claims(), 0);
    }

    #[test]
    fn test_epistemic_consciousness_creation() {
        let ec = EpistemicConsciousness::new();
        assert!(ec.is_ok());

        let ec = ec.unwrap();
        assert!(!ec.should_research(0.8)); // High Phi, no research needed
        assert!(ec.should_research(0.3)); // Low Phi, research needed
    }

    #[tokio::test]
    async fn research_confidence_does_not_self_label_meta_learning_outcomes() {
        let mut consciousness = EpistemicConsciousness::new().unwrap();
        let result = WebResearchResult {
            title: "Test result".to_string(),
            url: "https://aggregate.example/result".to_string(),
            content: "Source-backed test content".to_string(),
            summary: "A deliberately high-confidence test claim".to_string(),
            source_type: ResearchSource::Documentation,
            relevance: 0.99,
            confidence: 0.99,
            epistemic_status: EpistemicStatus::HighConfidence,
            status: ResearchStatus::Success,
            claims: vec![VerifiedClaim::verified(
                "A deliberately high-confidence claim for regression testing",
                vec!["https://claim-source.example/evidence".to_string()],
            )],
        };

        let integrated = consciousness
            .integrate_research_result(result.clone())
            .await
            .unwrap();

        assert_eq!(integrated.claims.len(), 1);
        assert_eq!(integrated.claims[0].confidence, 0.9);
        assert_eq!(consciousness.integrator.total_claims(), 1);
        assert_eq!(consciousness.learner.stats().total_outcomes, 0);
        assert_eq!(consciousness.learner.stats().accuracy, 0.5);
    }

    #[test]
    fn test_epistemic_types() {
        // Test that all types are properly exported
        let status = EpistemicStatus::HighConfidence;
        assert_eq!(status.confidence_score(), 0.95);
        assert!(!status.hedge_phrase().is_empty());

        let source = ResearchSource::Documentation;
        assert_eq!(source.label(), "Documentation");
    }
}
