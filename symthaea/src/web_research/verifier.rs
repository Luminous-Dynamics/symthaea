// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Verification Module
//!
//! Verifies claims against multiple sources and assigns epistemic status.
//! This is the core of hallucination prevention - every claim must be
//! epistemically verified before being presented.

use super::types::{EpistemicStatus, ResearchSource, VerifiedClaim, WebResearchResult};
use std::collections::HashMap;

/// Configuration for the epistemic verifier
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Minimum sources needed for high confidence
    pub min_sources_high_confidence: usize,
    /// Minimum sources needed for moderate confidence
    pub min_sources_moderate: usize,
    /// Weight for source agreement
    pub agreement_weight: f32,
    /// Weight for source credibility
    pub credibility_weight: f32,
    /// Contradiction penalty
    pub contradiction_penalty: f32,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            min_sources_high_confidence: 3,
            min_sources_moderate: 2,
            agreement_weight: 0.6,
            credibility_weight: 0.4,
            contradiction_penalty: 0.3,
        }
    }
}

/// Verification context containing source information
#[derive(Debug, Clone)]
pub struct VerificationContext {
    /// All sources consulted
    pub sources: Vec<SourceEvidence>,
    /// Original query that led to this verification
    pub query: String,
    /// Domain of the claim (e.g., "programming", "science")
    pub domain: String,
}

/// Evidence from a single source
#[derive(Debug, Clone)]
pub struct SourceEvidence {
    /// URL of the source
    pub url: String,
    /// Source domain (e.g., "rust-lang.org")
    pub domain: String,
    /// Source type
    pub source_type: ResearchSource,
    /// Relevant text from the source
    pub relevant_text: String,
    /// Does this source support the claim?
    pub supports: Option<bool>,
    /// Similarity score to the claim (0.0 - 1.0)
    pub similarity: f32,
    /// Credibility score of the source (0.0 - 1.0)
    pub credibility: f32,
}

/// Result of verifying a claim
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// The claim that was verified
    pub claim: String,
    /// Epistemic status assigned
    pub status: EpistemicStatus,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Hedge phrase to use when presenting
    pub hedge: String,
    /// Supporting sources
    pub supporting_sources: Vec<String>,
    /// Contradicting sources
    pub contradicting_sources: Vec<String>,
    /// Verification reasoning
    pub reasoning: String,
}

/// Epistemic verifier for claims
#[derive(Debug, Clone)]
pub struct EpistemicVerifier {
    /// Configuration
    config: VerifierConfig,
    /// Domain-specific credibility scores
    domain_credibility: HashMap<String, f32>,
    /// Source type credibility multipliers
    source_type_credibility: HashMap<ResearchSource, f32>,
}

impl Default for EpistemicVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicVerifier {
    /// Create a new epistemic verifier
    pub fn new() -> Self {
        let mut domain_credibility = HashMap::new();
        // Known high-credibility domains
        domain_credibility.insert("rust-lang.org".to_string(), 0.95);
        domain_credibility.insert("doc.rust-lang.org".to_string(), 0.95);
        domain_credibility.insert("docs.python.org".to_string(), 0.95);
        domain_credibility.insert("developer.mozilla.org".to_string(), 0.9);
        domain_credibility.insert("en.wikipedia.org".to_string(), 0.75);
        domain_credibility.insert("stackoverflow.com".to_string(), 0.7);
        domain_credibility.insert("arxiv.org".to_string(), 0.85);
        domain_credibility.insert("github.com".to_string(), 0.7);
        domain_credibility.insert("nixos.org".to_string(), 0.9);

        let mut source_type_credibility = HashMap::new();
        source_type_credibility.insert(ResearchSource::Documentation, 0.9);
        source_type_credibility.insert(ResearchSource::Academic, 0.85);
        source_type_credibility.insert(ResearchSource::Web, 0.6);

        Self {
            config: VerifierConfig::default(),
            domain_credibility,
            source_type_credibility,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VerifierConfig) -> Self {
        let mut verifier = Self::new();
        verifier.config = config;
        verifier
    }

    /// Verify a claim against collected evidence
    pub fn verify_claim(&self, claim: &str, context: &VerificationContext) -> VerificationResult {
        if context.sources.is_empty() {
            return VerificationResult {
                claim: claim.to_string(),
                status: EpistemicStatus::InsufficientEvidence,
                confidence: 0.2,
                hedge: EpistemicStatus::InsufficientEvidence
                    .hedge_phrase()
                    .to_string(),
                supporting_sources: vec![],
                contradicting_sources: vec![],
                reasoning: "No sources available for verification".to_string(),
            };
        }

        // Count supporting and contradicting sources
        let mut supporting: Vec<String> = Vec::new();
        let mut contradicting: Vec<String> = Vec::new();
        let mut total_support_score = 0.0f32;
        let mut total_contradict_score = 0.0f32;

        for source in &context.sources {
            let credibility = self.get_source_credibility(source);

            match source.supports {
                Some(true) => {
                    supporting.push(source.url.clone());
                    total_support_score += credibility * source.similarity;
                }
                Some(false) => {
                    contradicting.push(source.url.clone());
                    total_contradict_score += credibility * source.similarity;
                }
                None => {
                    // Neutral/unclear - slight support if similarity is high
                    if source.similarity > 0.7 {
                        total_support_score += credibility * source.similarity * 0.3;
                    }
                }
            }
        }

        // Determine epistemic status
        let (status, confidence, reasoning) = self.determine_status(
            &supporting,
            &contradicting,
            total_support_score,
            total_contradict_score,
        );

        VerificationResult {
            claim: claim.to_string(),
            status,
            confidence,
            hedge: status.hedge_phrase().to_string(),
            supporting_sources: supporting,
            contradicting_sources: contradicting,
            reasoning,
        }
    }

    /// Verify a claim from research results
    pub fn verify_from_results(&self, claim: &str, results: &[WebResearchResult]) -> VerifiedClaim {
        // Build verification context from results
        let sources: Vec<SourceEvidence> = results
            .iter()
            .map(|r| SourceEvidence {
                url: r.url.clone(),
                domain: self.extract_domain(&r.url),
                source_type: r.source_type,
                relevant_text: r.summary.clone(),
                supports: Some(r.confidence > 0.5),
                similarity: r.relevance,
                credibility: r.confidence,
            })
            .collect();

        let context = VerificationContext {
            sources,
            query: claim.to_string(),
            domain: String::new(),
        };

        let result = self.verify_claim(claim, &context);

        VerifiedClaim {
            text: result.claim,
            status: result.status,
            confidence: result.confidence,
            supporting_sources: result.supporting_sources,
            contradicting_sources: result.contradicting_sources,
            hedge: result.hedge,
        }
    }

    /// Get credibility score for a source
    fn get_source_credibility(&self, source: &SourceEvidence) -> f32 {
        // Start with domain credibility
        let domain_cred = self
            .domain_credibility
            .get(&source.domain)
            .copied()
            .unwrap_or(0.5);

        // Apply source type multiplier
        let type_mult = self
            .source_type_credibility
            .get(&source.source_type)
            .copied()
            .unwrap_or(0.6);

        // Combine (weighted average)
        (domain_cred * self.config.credibility_weight
            + source.credibility * self.config.agreement_weight)
            * type_mult
    }

    /// Determine epistemic status from evidence
    fn determine_status(
        &self,
        supporting: &[String],
        contradicting: &[String],
        support_score: f32,
        contradict_score: f32,
    ) -> (EpistemicStatus, f32, String) {
        let support_count = supporting.len();
        let contradict_count = contradicting.len();

        // Check for contradictions first
        if contradict_count > 0 && contradict_score > support_score * 0.7 {
            if contradict_score > support_score {
                return (
                    EpistemicStatus::Contradicted,
                    0.3 + (contradict_score - support_score).min(0.3),
                    format!(
                        "Claim contradicted by {} source(s); {} supporting",
                        contradict_count, support_count
                    ),
                );
            }
            return (
                EpistemicStatus::LowConfidence,
                0.4,
                format!(
                    "Mixed evidence: {} supporting, {} contradicting",
                    support_count, contradict_count
                ),
            );
        }

        // Evaluate support level
        if support_count >= self.config.min_sources_high_confidence {
            let confidence = (0.85 + support_score.min(0.15)).min(0.99);
            return (
                EpistemicStatus::HighConfidence,
                confidence,
                format!("Strongly supported by {} reliable sources", support_count),
            );
        }

        if support_count >= self.config.min_sources_moderate {
            let confidence = 0.6 + (support_score * 0.2).min(0.2);
            return (
                EpistemicStatus::ModerateConfidence,
                confidence,
                format!("Supported by {} sources", support_count),
            );
        }

        if support_count >= 1 {
            let confidence = 0.35 + (support_score * 0.15).min(0.15);
            return (
                EpistemicStatus::LowConfidence,
                confidence,
                format!("Limited support from {} source(s)", support_count),
            );
        }

        (
            EpistemicStatus::InsufficientEvidence,
            0.2,
            "No supporting sources found".to_string(),
        )
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

    /// Update domain credibility based on verification outcomes
    pub fn update_domain_credibility(&mut self, domain: &str, outcome_correct: bool) {
        let current = self.domain_credibility.get(domain).copied().unwrap_or(0.5);
        let adjustment = if outcome_correct { 0.02 } else { -0.05 };
        let new_value = (current + adjustment).clamp(0.1, 0.99);
        self.domain_credibility
            .insert(domain.to_string(), new_value);
    }

    /// Get current domain credibility
    pub fn get_domain_credibility(&self, domain: &str) -> f32 {
        self.domain_credibility.get(domain).copied().unwrap_or(0.5)
    }

    /// Add a known credible domain
    pub fn add_credible_domain(&mut self, domain: String, credibility: f32) {
        self.domain_credibility
            .insert(domain, credibility.clamp(0.0, 1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_with_no_sources() {
        let verifier = EpistemicVerifier::new();
        let context = VerificationContext {
            sources: vec![],
            query: "test".to_string(),
            domain: "test".to_string(),
        };

        let result = verifier.verify_claim("Rust is memory safe", &context);
        assert_eq!(result.status, EpistemicStatus::InsufficientEvidence);
    }

    #[test]
    fn test_verify_with_supporting_sources() {
        let verifier = EpistemicVerifier::new();
        let context = VerificationContext {
            sources: vec![
                SourceEvidence {
                    url: "https://rust-lang.org/safety".to_string(),
                    domain: "rust-lang.org".to_string(),
                    source_type: ResearchSource::Documentation,
                    relevant_text: "Rust provides memory safety".to_string(),
                    supports: Some(true),
                    similarity: 0.9,
                    credibility: 0.95,
                },
                SourceEvidence {
                    url: "https://doc.rust-lang.org/book".to_string(),
                    domain: "doc.rust-lang.org".to_string(),
                    source_type: ResearchSource::Documentation,
                    relevant_text: "Memory safety without GC".to_string(),
                    supports: Some(true),
                    similarity: 0.85,
                    credibility: 0.95,
                },
                SourceEvidence {
                    url: "https://en.wikipedia.org/wiki/Rust".to_string(),
                    domain: "en.wikipedia.org".to_string(),
                    source_type: ResearchSource::Web,
                    relevant_text: "Rust guarantees memory safety".to_string(),
                    supports: Some(true),
                    similarity: 0.8,
                    credibility: 0.75,
                },
            ],
            query: "Is Rust memory safe?".to_string(),
            domain: "programming".to_string(),
        };

        let result = verifier.verify_claim("Rust provides memory safety", &context);
        assert_eq!(result.status, EpistemicStatus::HighConfidence);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_verify_with_contradiction() {
        let verifier = EpistemicVerifier::new();
        let context = VerificationContext {
            sources: vec![
                SourceEvidence {
                    url: "https://source1.com".to_string(),
                    domain: "source1.com".to_string(),
                    source_type: ResearchSource::Web,
                    relevant_text: "The sky is blue".to_string(),
                    supports: Some(true),
                    similarity: 0.9,
                    credibility: 0.7,
                },
                SourceEvidence {
                    url: "https://source2.com".to_string(),
                    domain: "source2.com".to_string(),
                    source_type: ResearchSource::Web,
                    relevant_text: "The sky is green".to_string(),
                    supports: Some(false),
                    similarity: 0.9,
                    credibility: 0.8,
                },
            ],
            query: "What color is the sky?".to_string(),
            domain: "general".to_string(),
        };

        let result = verifier.verify_claim("The sky is blue", &context);
        // Should be low confidence or contradicted due to conflicting sources
        assert!(result.confidence < 0.7);
    }
}
