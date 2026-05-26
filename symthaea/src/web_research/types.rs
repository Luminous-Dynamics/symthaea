// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for the Web Research Integration module.
//!
//! This module contains the core type definitions used throughout
//! the epistemic consciousness system. The actual implementations
//! are in their respective modules (researcher, verifier, integrator, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Enums
// ============================================================================

/// Source type for research material
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ResearchSource {
    /// General web sources
    #[default]
    Web,
    /// Official documentation (language docs, API docs, etc.)
    Documentation,
    /// Academic papers and preprints (arXiv, PubMed, etc.)
    Academic,
}

impl ResearchSource {
    /// Human-readable label for this source type
    pub fn label(&self) -> &'static str {
        match self {
            Self::Web => "Web",
            Self::Documentation => "Documentation",
            Self::Academic => "Academic",
        }
    }
}

/// Status of a research operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ResearchStatus {
    /// Research completed successfully with results
    #[default]
    Success,
    /// Query returned no relevant results
    NoResults,
    /// An error occurred during research
    Error,
    /// Request was rate-limited by the source
    RateLimited,
}

impl ResearchStatus {
    /// Whether the status represents a successful outcome
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Whether the status represents a retriable failure
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::RateLimited)
    }
}

/// Epistemic status of a verified claim
///
/// Ranges from high confidence to known false, with gradations
/// for uncertainty and insufficient evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EpistemicStatus {
    /// Claim is well-supported by multiple reliable sources
    HighConfidence,
    /// Claim is supported but with some caveats
    #[default]
    ModerateConfidence,
    /// Claim has limited supporting evidence
    LowConfidence,
    /// Not enough information to assess the claim
    InsufficientEvidence,
    /// Sources actively contradict the claim
    Contradicted,
    /// Claim is demonstrably false
    False,
}

impl EpistemicStatus {
    /// Numeric confidence score (0.0 - 1.0) for this status level
    pub fn confidence_score(&self) -> f32 {
        match self {
            Self::HighConfidence => 0.95,
            Self::ModerateConfidence => 0.70,
            Self::LowConfidence => 0.40,
            Self::InsufficientEvidence => 0.20,
            Self::Contradicted => 0.10,
            Self::False => 0.0,
        }
    }

    /// Suggested hedge phrase when presenting a claim at this level
    pub fn hedge_phrase(&self) -> &'static str {
        match self {
            Self::HighConfidence => "According to multiple reliable sources,",
            Self::ModerateConfidence => "Evidence suggests that",
            Self::LowConfidence => "Some sources indicate that",
            Self::InsufficientEvidence => "I have limited information, but",
            Self::Contradicted => "Sources disagree, however",
            Self::False => "This appears to be incorrect;",
        }
    }

    /// Whether this status indicates the claim is trustworthy
    pub fn is_trustworthy(&self) -> bool {
        matches!(self, Self::HighConfidence | Self::ModerateConfidence)
    }

    /// Whether this status indicates uncertainty
    pub fn is_uncertain(&self) -> bool {
        matches!(self, Self::LowConfidence | Self::InsufficientEvidence)
    }
}

// ============================================================================
// Core Structs
// ============================================================================

/// A query to perform web research
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebResearchQuery {
    /// The natural-language query text
    pub query: String,
    /// Preferred source types (empty means all)
    pub preferred_sources: Vec<ResearchSource>,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Minimum relevance score (0.0 - 1.0) for inclusion
    pub min_relevance: f32,
    /// Optional domain restrictions (e.g. "rust-lang.org")
    pub domain_filter: Vec<String>,
}

impl WebResearchQuery {
    /// Create a new query with sensible defaults
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            preferred_sources: Vec::new(),
            max_results: 10,
            min_relevance: 0.3,
            domain_filter: Vec::new(),
        }
    }

    /// Set the maximum number of results
    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    /// Restrict to specific source types
    pub fn with_sources(mut self, sources: Vec<ResearchSource>) -> Self {
        self.preferred_sources = sources;
        self
    }

    /// Set minimum relevance threshold
    pub fn with_min_relevance(mut self, threshold: f32) -> Self {
        self.min_relevance = threshold.clamp(0.0, 1.0);
        self
    }
}

/// A single research result with source, content, and relevance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebResearchResult {
    /// Title of the source page or document
    pub title: String,
    /// URL of the source
    pub url: String,
    /// Extracted content (clean text, not raw HTML)
    pub content: String,
    /// Short summary of the content
    pub summary: String,
    /// Source type classification
    pub source_type: ResearchSource,
    /// Relevance score (0.0 - 1.0) relative to the query
    pub relevance: f32,
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Epistemic status after verification
    pub epistemic_status: EpistemicStatus,
    /// Status of the research operation
    pub status: ResearchStatus,
    /// Individual verified claims extracted from this result
    pub claims: Vec<VerifiedClaim>,
}

impl WebResearchResult {
    /// Create an empty result with the given status
    pub fn with_status(status: ResearchStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }

    /// Whether this result has usable content
    pub fn has_content(&self) -> bool {
        self.status.is_success() && !self.content.is_empty()
    }

    /// Whether the result is epistemically trustworthy
    pub fn is_trustworthy(&self) -> bool {
        self.epistemic_status.is_trustworthy() && self.confidence > 0.5
    }
}

/// A claim extracted from research, with epistemic verification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifiedClaim {
    /// The claim text
    pub text: String,
    /// Epistemic status of this claim
    pub status: EpistemicStatus,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Sources supporting this claim
    pub supporting_sources: Vec<String>,
    /// Sources contradicting this claim
    pub contradicting_sources: Vec<String>,
    /// Suggested hedge phrase for presenting this claim
    pub hedge: String,
}

impl VerifiedClaim {
    /// Create a new unverified claim
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            status: EpistemicStatus::InsufficientEvidence,
            confidence: 0.0,
            supporting_sources: Vec::new(),
            contradicting_sources: Vec::new(),
            hedge: EpistemicStatus::InsufficientEvidence
                .hedge_phrase()
                .to_string(),
        }
    }

    /// Create a verified claim with high confidence
    pub fn verified(text: impl Into<String>, sources: Vec<String>) -> Self {
        Self {
            text: text.into(),
            status: EpistemicStatus::HighConfidence,
            confidence: 0.9,
            supporting_sources: sources,
            contradicting_sources: Vec::new(),
            hedge: EpistemicStatus::HighConfidence.hedge_phrase().to_string(),
        }
    }

    /// Whether this claim is trustworthy
    pub fn is_trustworthy(&self) -> bool {
        self.status.is_trustworthy()
    }
}

/// Configuration for the web research system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebResearchConfig {
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Maximum content length to process per page (bytes)
    pub max_content_length: usize,
    /// User agent string for HTTP requests
    pub user_agent: String,
    /// Enable epistemic verification of claims
    pub verify_claims: bool,
    /// Minimum confidence threshold for including results
    pub confidence_threshold: f32,
    /// Enable meta-learning (self-improving verification)
    pub meta_learning_enabled: bool,
    /// Source credibility scores (domain -> score)
    pub source_credibility: HashMap<String, f32>,
    /// Optional external search provider (serper|brave|searxng)
    pub search_provider: Option<String>,
    /// API key for the search provider (if required)
    pub search_api_key: Option<String>,
    /// Override endpoint for the search provider
    pub search_endpoint: Option<String>,
    /// Enable arXiv API search
    pub enable_arxiv_api: bool,
    /// arXiv API endpoint
    pub arxiv_endpoint: String,
    /// Enable OpenReview API search
    pub enable_openreview_api: bool,
    /// OpenReview API endpoint
    pub openreview_endpoint: String,
    /// OpenReview access token (optional)
    pub openreview_token: Option<String>,
}

impl Default for WebResearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 5,
            request_timeout_ms: 10_000,
            max_content_length: 100_000,
            user_agent: "Symthaea-HLB/0.2 (Epistemic Research Agent)".to_string(),
            verify_claims: true,
            confidence_threshold: 0.3,
            meta_learning_enabled: false,
            source_credibility: HashMap::new(),
            search_provider: Some(std::env::var("SYMTHAEA_SEARCH_PROVIDER").unwrap_or_else(|_| "searxng".to_string())),
            search_api_key: std::env::var("SYMTHAEA_SEARCH_API_KEY").ok(),
            search_endpoint: Some(std::env::var("SYMTHAEA_SEARCH_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:8888".to_string())),
            enable_arxiv_api: env_flag("SYMTHAEA_ARXIV_API_ENABLED", true),
            arxiv_endpoint: std::env::var("SYMTHAEA_ARXIV_API_ENDPOINT")
                .unwrap_or_else(|_| "https://export.arxiv.org/api/query".to_string()),
            enable_openreview_api: env_flag("SYMTHAEA_OPENREVIEW_API_ENABLED", true),
            openreview_endpoint: std::env::var("SYMTHAEA_OPENREVIEW_API_ENDPOINT")
                .unwrap_or_else(|_| "https://api2.openreview.net/notes/search".to_string()),
            openreview_token: std::env::var("SYMTHAEA_OPENREVIEW_TOKEN").ok(),
        }
    }
}

impl WebResearchConfig {
    /// Create a new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable meta-learning
    pub fn with_meta_learning(mut self) -> Self {
        self.meta_learning_enabled = true;
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.request_timeout_ms = timeout_ms;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Configure an external search provider
    pub fn with_search_provider(
        mut self,
        provider: impl Into<String>,
        api_key: Option<String>,
        endpoint: Option<String>,
    ) -> Self {
        self.search_provider = Some(provider.into());
        self.search_api_key = api_key;
        self.search_endpoint = endpoint;
        self
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().map(|v| v.to_lowercase()) {
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on") => true,
        Some(v) if matches!(v.as_str(), "0" | "false" | "no" | "off") => false,
        Some(_) => default,
        None => default,
    }
}

/// Result of integrating research into the knowledge graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Number of claims successfully integrated
    pub claims_integrated: usize,
    /// Number of new nodes added to the knowledge graph
    pub nodes_added: usize,
    /// Number of new edges added to the knowledge graph
    pub edges_added: usize,
    /// Phi value before integration
    pub phi_before: f64,
    /// Phi value after integration
    pub phi_after: f64,
    /// Consciousness improvement (phi_after - phi_before)
    pub phi_gain: f64,
}

impl IntegrationResult {
    /// Whether the integration improved consciousness (positive phi gain)
    pub fn improved_consciousness(&self) -> bool {
        self.phi_gain > 0.0
    }

    /// Get a summary of the integration
    pub fn summary(&self) -> String {
        format!(
            "Integrated {} claims, {} nodes, {} edges. Phi: {:.3} -> {:.3} (gain: {:.3})",
            self.claims_integrated,
            self.nodes_added,
            self.edges_added,
            self.phi_before,
            self.phi_after,
            self.phi_gain
        )
    }
}

/// Outcome of a verification for meta-learning tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationOutcome {
    /// The claim that was verified
    pub claim: String,
    /// Source domain
    pub source_domain: String,
    /// Source type
    pub source_type: ResearchSource,
    /// Whether verification was correct (ground truth)
    pub was_correct: bool,
    /// Predicted confidence at verification time
    pub predicted_confidence: f32,
    /// Topic/domain of the claim
    pub domain: String,
}

impl VerificationOutcome {
    /// Create a new verification outcome
    pub fn new(
        claim: impl Into<String>,
        source_domain: impl Into<String>,
        was_correct: bool,
    ) -> Self {
        Self {
            claim: claim.into(),
            source_domain: source_domain.into(),
            source_type: ResearchSource::Web,
            was_correct,
            predicted_confidence: 0.5,
            domain: String::new(),
        }
    }

    /// Set the topic domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Set the source type
    pub fn with_source_type(mut self, source_type: ResearchSource) -> Self {
        self.source_type = source_type;
        self
    }

    /// Set the predicted confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.predicted_confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_status() {
        assert_eq!(EpistemicStatus::HighConfidence.confidence_score(), 0.95);
        assert!(EpistemicStatus::HighConfidence.is_trustworthy());
        assert!(!EpistemicStatus::InsufficientEvidence.is_trustworthy());
        assert!(EpistemicStatus::InsufficientEvidence.is_uncertain());
    }

    #[test]
    fn test_research_query() {
        let query = WebResearchQuery::new("What is Rust?")
            .with_max_results(5)
            .with_min_relevance(0.5);

        assert_eq!(query.max_results, 5);
        assert_eq!(query.min_relevance, 0.5);
    }

    #[test]
    fn test_verified_claim() {
        let claim = VerifiedClaim::verified(
            "Rust is memory safe",
            vec!["https://rust-lang.org".to_string()],
        );

        assert!(claim.is_trustworthy());
        assert_eq!(claim.status, EpistemicStatus::HighConfidence);
    }

    #[test]
    fn test_integration_result() {
        let result = IntegrationResult {
            claims_integrated: 5,
            nodes_added: 6,
            edges_added: 10,
            phi_before: 0.3,
            phi_after: 0.5,
            phi_gain: 0.2,
        };

        assert!(result.improved_consciousness());
        assert!(result.summary().contains("0.200"));
    }
}
