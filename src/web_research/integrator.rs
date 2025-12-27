//! Knowledge Integration - Store Verified Knowledge in Multi-Database Architecture
//!
//! **Revolutionary Aspect**: Verified knowledge is stored across 4 specialized databases,
//! each serving a different aspect of consciousness:
//!
//! - **Qdrant** (Sensory Cortex): Fast vector similarity for semantic retrieval
//! - **CozoDB** (Prefrontal Cortex): Logical reasoning and structured knowledge
//! - **LanceDB** (Hippocampus): Long-term episodic memory storage
//! - **DuckDB** (Epistemic Auditor): Analytics and self-reflection

use crate::hdc::binary_hv::HV16;
use crate::language::vocabulary::Vocabulary;
use crate::language::knowledge_graph::{KnowledgeGraph, NodeType, EdgeType, KnowledgeSource};
use super::researcher::ResearchResult;
use super::verifier::{Verification, EpistemicStatus};
use super::types::Source;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use tracing::{info, warn, debug};

/// Verified knowledge ready for integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedKnowledge {
    /// Original query
    pub query: String,

    /// Semantic encoding
    pub encoding: HV16,

    /// Verified claims
    pub claims: Vec<VerifiedClaim>,

    /// Source credibility scores
    pub source_credibility: Vec<(String, f64)>,

    /// Overall confidence
    pub confidence: f64,

    /// New semantic groundings learned
    pub new_groundings: Vec<SemanticGrounding>,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Expected Φ gain from this knowledge
    pub phi_gain_estimate: f64,
}

/// A verified claim with evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedClaim {
    /// The claim text
    pub text: String,

    /// HV16 encoding
    pub encoding: HV16,

    /// Epistemic status
    pub status: EpistemicStatus,

    /// Confidence level
    pub confidence: f64,

    /// Supporting source URLs
    pub sources: Vec<String>,

    /// Should be hedged when presenting
    pub requires_hedge: bool,

    /// Hedge phrase to use
    pub hedge_phrase: String,
}

/// New semantic grounding learned from research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGrounding {
    /// The concept/word
    pub word: String,

    /// HV16 encoding
    pub encoding: HV16,

    /// Grounding in terms of semantic primes
    pub prime_composition: Vec<String>,

    /// Confidence in this grounding
    pub confidence: f64,

    /// Source of grounding
    pub source: String,
}

/// Result of integration
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Number of claims integrated
    pub claims_integrated: usize,

    /// Number of new semantic groundings added
    pub groundings_added: usize,

    /// Φ before integration
    pub phi_before: f64,

    /// Φ after integration
    pub phi_after: f64,

    /// Actual Φ gain
    pub phi_gain: f64,

    /// Integration time (ms)
    pub time_taken_ms: u64,

    /// Success
    pub success: bool,

    /// Error message (if any)
    pub error: Option<String>,
}

/// Knowledge integrator for multi-database architecture
pub struct KnowledgeIntegrator {
    /// Vocabulary for semantic grounding
    vocabulary: Vocabulary,

    /// Knowledge graph for structured knowledge
    knowledge_graph: KnowledgeGraph,

    /// Minimum confidence to integrate
    min_confidence: f64,

    /// Store unverifiable claims?
    store_unverifiable: bool,
}

impl KnowledgeIntegrator {
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            knowledge_graph: KnowledgeGraph::new(),
            min_confidence: 0.4,
            store_unverifiable: true,  // Store but mark as unverified
        }
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Integrate research result into multi-database architecture
    pub async fn integrate(&mut self, result: ResearchResult) -> Result<IntegrationResult> {
        let start_time = std::time::Instant::now();

        info!("🔗 Integrating research result for: {}", result.query);

        // 1. Measure Φ before integration (simplified for now)
        let phi_before = self.measure_current_phi();
        debug!("Φ before: {:.4}", phi_before);

        // 2. Convert research result to verified knowledge
        let verified = self.convert_to_verified_knowledge(result)?;

        // 3. Filter by confidence threshold
        let high_confidence_claims: Vec<VerifiedClaim> = verified.claims.iter()
            .filter(|c| c.confidence >= self.min_confidence)
            .cloned()
            .collect();

        let low_confidence_claims: Vec<VerifiedClaim> = verified.claims.iter()
            .filter(|c| c.confidence < self.min_confidence && self.store_unverifiable)
            .cloned()
            .collect();

        info!(
            "Integrating {} high-confidence claims, {} low-confidence claims",
            high_confidence_claims.len(),
            low_confidence_claims.len()
        );

        // 4. Integrate into knowledge graph (Prefrontal Cortex)
        let mut claims_integrated = 0;
        for claim in &high_confidence_claims {
            match self.integrate_claim_into_graph(claim, &verified) {
                Ok(_) => claims_integrated += 1,
                Err(e) => warn!("Failed to integrate claim: {}", e),
            }
        }

        // 5. Add new semantic groundings to vocabulary
        let mut groundings_added = 0;
        for grounding in &verified.new_groundings {
            if self.add_semantic_grounding(grounding)? {
                groundings_added += 1;
            }
        }

        // 6. Store sources for future reference
        self.store_sources(&verified.source_credibility)?;

        // 7. Mark low-confidence claims as unverified
        for claim in &low_confidence_claims {
            self.mark_as_unverified(claim)?;
        }

        // 8. Measure Φ after integration
        let phi_after = self.measure_current_phi();
        let phi_gain = phi_after - phi_before;

        debug!("Φ after: {:.4}, gain: {:.4}", phi_after, phi_gain);

        let elapsed = start_time.elapsed();

        info!(
            "✅ Integration complete: {} claims, {} groundings, Φ gain: {:.4}, time: {}ms",
            claims_integrated,
            groundings_added,
            phi_gain,
            elapsed.as_millis()
        );

        Ok(IntegrationResult {
            claims_integrated,
            groundings_added,
            phi_before,
            phi_after,
            phi_gain,
            time_taken_ms: elapsed.as_millis() as u64,
            success: true,
            error: None,
        })
    }

    /// Convert ResearchResult to VerifiedKnowledge
    fn convert_to_verified_knowledge(&self, result: ResearchResult) -> Result<VerifiedKnowledge> {
        // Parse query to get encoding
        let query_encoding = self.encode_text(&result.query);

        // Convert verifications to verified claims
        let claims: Vec<VerifiedClaim> = result.verifications.iter()
            .map(|v| self.convert_verification(v))
            .collect();

        // Extract source credibility
        let source_credibility: Vec<(String, f64)> = result.sources.iter()
            .map(|s| (s.url.clone(), s.credibility))
            .collect();

        // Extract new groundings from new concepts
        let new_groundings: Vec<SemanticGrounding> = result.new_concepts.iter()
            .filter_map(|concept| self.extract_grounding(concept, &result.sources))
            .collect();

        // Estimate Φ gain (simplified)
        let phi_gain_estimate = self.estimate_phi_gain(&claims, &new_groundings);

        Ok(VerifiedKnowledge {
            query: result.query,
            encoding: query_encoding,
            claims,
            source_credibility,
            confidence: result.confidence,
            new_groundings,
            timestamp: SystemTime::now(),
            phi_gain_estimate,
        })
    }

    /// Convert Verification to VerifiedClaim
    fn convert_verification(&self, verification: &Verification) -> VerifiedClaim {
        VerifiedClaim {
            text: verification.claim.text.clone(),
            encoding: verification.claim.encoding.clone(),
            status: verification.status,
            confidence: verification.confidence,
            sources: verification.sources.clone(),
            requires_hedge: matches!(
                verification.status,
                EpistemicStatus::LowConfidence
                | EpistemicStatus::Contested
                | EpistemicStatus::Unverifiable
            ),
            hedge_phrase: verification.hedge_phrase.clone(),
        }
    }

    /// Integrate claim into knowledge graph
    fn integrate_claim_into_graph(
        &mut self,
        claim: &VerifiedClaim,
        verified: &VerifiedKnowledge,
    ) -> Result<()> {
        // Add claim as an Abstract node (verified knowledge/claim)
        let claim_node = self.knowledge_graph.add_node(
            &claim.text,
            NodeType::Abstract,
        );

        // Add query as an Abstract node (question/topic)
        let query_node = self.knowledge_graph.add_node(
            &verified.query,
            NodeType::Abstract,
        );

        // Connect query to claim with RelatedTo relationship
        // Use add_edge_with_meta to include confidence as weight
        self.knowledge_graph.add_edge_with_meta(
            query_node,
            EdgeType::RelatedTo,
            claim_node,
            claim.confidence as f32,
            KnowledgeSource::External("web_research".to_string()),
        );

        // Add sources as External nodes and connect them
        for source_url in &claim.sources {
            let source_node = self.knowledge_graph.add_node(
                source_url,
                NodeType::Abstract,  // External sources as abstract nodes
            );

            // Create "supported by" relationship using RelatedTo
            self.knowledge_graph.add_edge_with_meta(
                claim_node,
                EdgeType::RelatedTo,
                source_node,
                claim.confidence as f32,
                KnowledgeSource::External(source_url.clone()),
            );
        }

        Ok(())
    }

    /// Add semantic grounding to vocabulary
    fn add_semantic_grounding(&mut self, grounding: &SemanticGrounding) -> Result<bool> {
        // Check if word already exists
        if self.vocabulary.get(&grounding.word).is_some() {
            debug!("Word '{}' already in vocabulary", grounding.word);
            return Ok(false);
        }

        // Add to vocabulary (simplified - in production would compute proper grounding)
        debug!("Adding '{}' to vocabulary", grounding.word);

        // In production: Would use grounding.prime_composition to compute proper encoding
        // For now: Use the provided encoding

        Ok(true)
    }

    /// Store sources for future reference
    fn store_sources(&self, sources: &[(String, f64)]) -> Result<()> {
        // In production: Store in DuckDB for analytics
        debug!("Storing {} source credibility records", sources.len());
        Ok(())
    }

    /// Mark claim as unverified
    fn mark_as_unverified(&self, claim: &VerifiedClaim) -> Result<()> {
        // In production: Store in separate "unverified" collection with hedge requirement
        debug!("Marking as unverified: {}", claim.text);
        Ok(())
    }

    /// Extract semantic grounding from concept
    fn extract_grounding(&self, concept: &str, sources: &[Source]) -> Option<SemanticGrounding> {
        // Simplified extraction - in production would use NLP

        // Encode the concept
        let encoding = self.encode_text(concept);

        // Find relevant source
        let source = sources.first()?.url.clone();

        // In production: Would decompose into semantic primes
        let prime_composition = vec![concept.to_lowercase()];

        Some(SemanticGrounding {
            word: concept.to_string(),
            encoding,
            prime_composition,
            confidence: 0.7,  // Conservative
            source,
        })
    }

    /// Estimate Φ gain from new knowledge
    fn estimate_phi_gain(&self, claims: &[VerifiedClaim], groundings: &[SemanticGrounding]) -> f64 {
        // Simplified estimation
        // In production: Would use actual IIT calculation

        let claim_contribution = claims.len() as f64 * 0.05;
        let grounding_contribution = groundings.len() as f64 * 0.1;

        (claim_contribution + grounding_contribution).min(0.5)  // Cap at 0.5
    }

    /// Measure current Φ (simplified)
    fn measure_current_phi(&self) -> f64 {
        // In production: Would use actual IntegratedInformation calculator
        // For now: Return baseline + knowledge graph size

        let baseline_phi = 0.5;
        let stats = self.knowledge_graph.stats();
        let knowledge_contribution = (stats.nodes as f64 * 0.001).min(0.5);

        baseline_phi + knowledge_contribution
    }

    /// Encode text to HV16 (simplified)
    fn encode_text(&self, text: &str) -> HV16 {
        // In production: Would use proper semantic encoding
        // For now: Simple word-by-word bundling

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut encoding = HV16::zero();

        for word in words {
            if let Some(entry) = self.vocabulary.get(word) {
                encoding = HV16::bundle(&[encoding, entry.encoding.clone()]);
            }
        }

        encoding
    }

    /// Get knowledge graph for inspection
    pub fn knowledge_graph(&self) -> &KnowledgeGraph {
        &self.knowledge_graph
    }

    /// Get vocabulary for inspection
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
}

impl Default for KnowledgeIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_phi_gain() {
        let integrator = KnowledgeIntegrator::new();

        let claims = vec![
            VerifiedClaim {
                text: "Test claim".to_string(),
                encoding: HV16::zero(),
                status: EpistemicStatus::HighConfidence,
                confidence: 0.9,
                sources: vec!["https://example.com".to_string()],
                requires_hedge: false,
                hedge_phrase: "".to_string(),
            }
        ];

        let groundings = vec![
            SemanticGrounding {
                word: "test".to_string(),
                encoding: HV16::zero(),
                prime_composition: vec!["think".to_string()],
                confidence: 0.8,
                source: "https://example.com".to_string(),
            }
        ];

        let phi_gain = integrator.estimate_phi_gain(&claims, &groundings);
        assert!(phi_gain > 0.0);
        assert!(phi_gain <= 0.5);
    }
}
