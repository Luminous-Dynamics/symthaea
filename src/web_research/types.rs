//! Core types for web research and epistemic verification

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A source from web research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// URL of the source
    pub url: String,

    /// Title of the page/article
    pub title: String,

    /// Extracted text content
    pub content: String,

    /// Publication date (if available)
    pub published_date: Option<SystemTime>,

    /// Author (if available)
    pub author: Option<String>,

    /// Domain authority/credibility score (0.0-1.0)
    pub credibility: f64,

    /// Semantic encoding of content
    pub encoding: HV16,

    /// When this was fetched
    pub fetch_timestamp: SystemTime,
}

/// A search query with semantic expansion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Original query text
    pub original: String,

    /// Semantically expanded queries
    pub expansions: Vec<String>,

    /// HV16 encoding of query intent
    pub intent_encoding: HV16,

    /// Priority (higher = more important)
    pub priority: f64,
}

/// Research plan guided by consciousness gradient
#[derive(Debug, Clone)]
pub struct ResearchPlan {
    /// Main query to research
    pub query: String,

    /// Decomposed sub-questions
    pub sub_questions: Vec<String>,

    /// Expected Φ gain from answering
    pub expected_phi_gain: f64,

    /// Verification level required
    pub verification_level: VerificationLevel,

    /// Maximum sources to fetch
    pub max_sources: usize,

    /// Timeout for research
    pub timeout_seconds: u64,
}

/// Verification level for epistemic checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// Minimal checking (single source OK)
    Minimal,

    /// Standard verification (multiple sources)
    Standard,

    /// Rigorous verification (cross-reference + contradiction check)
    Rigorous,

    /// Academic-level verification (peer-reviewed sources preferred)
    Academic,
}

/// A claim extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    /// The claim text
    pub text: String,

    /// HV16 encoding
    pub encoding: HV16,

    /// Subject of claim
    pub subject: String,

    /// Predicate/assertion
    pub predicate: String,

    /// Object (if applicable)
    pub object: Option<String>,

    /// Confidence in extraction (0.0-1.0)
    pub extraction_confidence: f64,
}

/// Ranking of sources by relevance
#[derive(Debug, Clone)]
pub struct SourceRanking {
    /// Source
    pub source: Source,

    /// Relevance score (0.0-1.0)
    pub relevance: f64,

    /// Credibility score (0.0-1.0)
    pub credibility: f64,

    /// Recency score (0.0-1.0)
    pub recency: f64,

    /// Combined score
    pub combined_score: f64,
}

impl SourceRanking {
    /// Calculate combined score from components
    pub fn calculate_score(
        relevance: f64,
        credibility: f64,
        recency: f64,
    ) -> f64 {
        // Weighted combination
        (relevance * 0.6) + (credibility * 0.3) + (recency * 0.1)
    }
}

/// Relevance score breakdown
#[derive(Debug, Clone)]
pub struct RelevanceScore {
    /// Semantic similarity to query
    pub semantic_similarity: f64,

    /// Keyword overlap
    pub keyword_overlap: f64,

    /// Topical relevance
    pub topical_relevance: f64,

    /// Combined relevance
    pub combined: f64,
}

impl RelevanceScore {
    pub fn new(semantic: f64, keyword: f64, topical: f64) -> Self {
        let combined = (semantic * 0.5) + (keyword * 0.3) + (topical * 0.2);
        Self {
            semantic_similarity: semantic,
            keyword_overlap: keyword,
            topical_relevance: topical,
            combined,
        }
    }
}
