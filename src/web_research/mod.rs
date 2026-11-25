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

    /// Perform autonomous research on a query
    pub async fn research(&mut self, query: &str) -> anyhow::Result<WebResearchResult> {
        let result = self.researcher.research_and_verify(query).await?;

        // Integrate the result
        let integration = self.integrator.integrate(result.clone()).await?;

        // Update last Phi
        self.last_phi = integration.phi_after;

        // Do not feed the local verifier's own confidence/status back as
        // correctness labels. Meta-learning requires independent ground truth;
        // callers can supply that later via record_adjudicated_outcome().
        Ok(result)
    }

    /// Record a verification outcome only after independent adjudication.
    ///
    /// The authority/evidence fields make the provenance requirement explicit,
    /// but this method does not authenticate the authority or resolve the
    /// referenced evidence. Callers must use the existing human-review or
    /// independent-evaluator process; never derive `was_correct` from model
    /// confidence or the local `EpistemicVerifier` result.
    pub fn record_adjudicated_outcome(
        &mut self,
        adjudicated: AdjudicatedVerificationOutcome,
    ) -> anyhow::Result<()> {
        let outcome = &adjudicated.outcome;
        if outcome.claim.trim().is_empty() {
            anyhow::bail!("adjudicated verification claim must be non-empty");
        }
        if outcome.source_domain.trim().is_empty() {
            anyhow::bail!("adjudicated verification source domain must be non-empty");
        }
        if outcome.domain.trim().is_empty() {
            anyhow::bail!("adjudicated verification topic domain must be non-empty");
        }
        if !outcome.predicted_confidence.is_finite()
            || !(0.0..=1.0).contains(&outcome.predicted_confidence)
        {
            anyhow::bail!("adjudicated verification confidence must be finite and in [0, 1]");
        }
        if adjudicated.authority_id.trim().is_empty() {
            anyhow::bail!("adjudication authority ID must be non-empty");
        }
        if adjudicated.evidence_reference.trim().is_empty() {
            anyhow::bail!("adjudication evidence reference must be non-empty");
        }

        self.learner.record_outcome(adjudicated.outcome)
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

    /// Extract domain from URL
    fn extract_domain(&self, url: &str) -> String {
        url.split("://")
            .nth(1)
            .unwrap_or("")
            .split('/')
            .next()
            .unwrap_or("")
            .to_string()
    }

    /// Detect topic domain from query
    fn detect_domain(&self, query: &str) -> String {
        let lower = query.to_lowercase();

        if lower.contains("programming")
            || lower.contains("code")
            || lower.contains("rust")
            || lower.contains("python")
        {
            return "programming".to_string();
        }

        if lower.contains("nixos") || lower.contains("nix") || lower.contains("linux") {
            return "systems".to_string();
        }

        if lower.contains("science") || lower.contains("research") {
            return "science".to_string();
        }

        "general".to_string()
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

    fn test_consciousness() -> EpistemicConsciousness {
        let (researcher, integrator, learner) = create_epistemic_system().unwrap();
        EpistemicConsciousness {
            researcher,
            integrator,
            learner,
            phi_threshold: 0.6,
            last_phi: 0.0,
        }
    }

    fn adjudicated_outcome() -> AdjudicatedVerificationOutcome {
        AdjudicatedVerificationOutcome {
            outcome: VerificationOutcome {
                claim: "The test result is independently reproducible".to_owned(),
                source_domain: "example.org".to_owned(),
                source_type: ResearchSource::Academic,
                was_correct: true,
                predicted_confidence: 0.9,
                domain: "science".to_owned(),
            },
            adjudication_kind: AdjudicationKind::IndependentEvaluator,
            authority_id: "independent-evaluator-test".to_owned(),
            evidence_reference: "urn:example:receipt:sha256:0123456789abcdef".to_owned(),
        }
    }

    #[test]
    fn meta_learning_accepts_only_complete_adjudication_records() {
        let mut consciousness = test_consciousness();
        assert_eq!(consciousness.learner.stats().total_outcomes, 0);

        let mut missing_authority = adjudicated_outcome();
        missing_authority.authority_id.clear();
        assert!(consciousness
            .record_adjudicated_outcome(missing_authority)
            .is_err());

        let mut missing_evidence = adjudicated_outcome();
        missing_evidence.evidence_reference.clear();
        assert!(consciousness
            .record_adjudicated_outcome(missing_evidence)
            .is_err());

        let mut invalid_confidence = adjudicated_outcome();
        invalid_confidence.outcome.predicted_confidence = f32::NAN;
        assert!(consciousness
            .record_adjudicated_outcome(invalid_confidence)
            .is_err());

        let mut empty_claim = adjudicated_outcome();
        empty_claim.outcome.claim = "   ".to_owned();
        assert!(consciousness
            .record_adjudicated_outcome(empty_claim)
            .is_err());

        assert_eq!(consciousness.learner.stats().total_outcomes, 0);
    }

    #[test]
    fn independently_adjudicated_outcome_updates_the_existing_learner() {
        let mut consciousness = test_consciousness();
        consciousness
            .record_adjudicated_outcome(adjudicated_outcome())
            .unwrap();

        assert_eq!(consciousness.learner.stats().total_outcomes, 1);
        assert_eq!(
            consciousness
                .learner
                .get_source_credibility("example.org")
                .unwrap()
                .outcome_count,
            1
        );
    }

    #[test]
    fn test_epistemic_types() {
        // Test that all types are properly exported
        let status = EpistemicStatus::HighConfidence;
        assert_eq!(status.confidence_score(), 0.95);
        assert!(!status.hedge_phrase().is_empty());

        let source = ResearchSource::Documentation;
        assert_eq!(source.label(), "Documentation");

        let adjudication = AdjudicationKind::IndependentEvaluator;
        assert_eq!(adjudication, AdjudicationKind::IndependentEvaluator);
    }
}
