// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG (Distributed Knowledge Graph) - Verifiable Claims with Confidence Scoring
//!
//! The DKG stores verifiable claims (triples) with provenance tracking and
//! dynamic confidence calculation. Each claim's confidence is computed from:
//!
//! 1. **Attestation count** - More attesters = higher confidence (diminishing returns)
//! 2. **Reputation weighting** - High-rep attesters contribute more (reputation² voting)
//! 3. **Source quality** - Peer-reviewed > news > social media
//! 4. **Time decay** - Older claims decay in confidence
//! 5. **Consistency** - Contradicting claims reduce confidence
//!
//! # Security Assumptions
//!
//! This module assumes:
//! - **Signature Verification**: All attestations MUST have valid Ed25519 signatures
//! - **Key Security**: Private signing keys are never exposed to untrusted code
//! - **Identity Binding**: Attester IDs are cryptographically bound to their keys
//! - **Randomness**: System RNG provides cryptographically secure randomness
//!
//! ## Threat Model
//!
//! - Adversary can submit false claims (detected via attestation confidence)
//! - Adversary can attempt Sybil attacks (mitigated by reputation² voting)
//! - Byzantine nodes may deviate from attestation protocol
//! - Adversary cannot forge attestation signatures without private keys
//!
//! ## Limitations
//!
//! - Confidence scores are probabilistic, not guarantees of truth
//! - Initial "cold start" period with few attestations has lower accuracy
//! - Time-based rate limiting (60s default) may delay legitimate updates
//! - Attestation signatures must be verified before trusting claims
//!
//! ## Security Best Practices
//!
//! 1. ALWAYS use `add_verified()` to add attestations (validates signatures)
//! 2. Use `compute_triple_commitment()` for cryptographic binding to triples
//! 3. Implement proper key management for attester signing keys
//! 4. Monitor for unusual attestation patterns (potential Sybil attacks)
//!
//! # Epistemic Classification
//!
//! Claims are classified by epistemic status:
//! - **E (Empirical)**: Verifiable through observation/experiment
//! - **N (Normative)**: Value judgments, should-statements
//! - **M (Metaphysical)**: Unfalsifiable claims about reality
//!
//! This classification affects how confidence is calculated and displayed.
//!
//! # Genesis Simulation
//!
//! The "Sky Color" scenario demonstrates the Truth Engine:
//! - Alice claims "The sky is blue" (truth)
//! - Mallory claims "The sky is green" (lie)
//! - Bob endorses Alice's claim
//! - The Truth Engine favors Alice's claim due to social consensus
//!
//! # Integration with MATL
//!
//! DKG confidence scoring integrates with MATL's reputation system:
//! - K-Vector dimensions inform attester reputation weighting
//! - RB-BFT consensus protects against coordinated misinformation
//! - PoGQ validation scores contribute to claim confidence

pub mod attestation;
pub mod confidence;
pub mod contradictions;
pub mod discovery_engine;
pub mod phi_integration;
pub mod phi_query_router;
pub mod propagation_engine;
pub mod query;
pub mod query_index;

pub use attestation::{
    Attestation, AttestationCounts, AttestationError, AttestationSet, AttestationType,
    InMemoryKeyRegistry, PublicKeyRegistry,
};
pub use confidence::{
    calculate_confidence, meets_threshold, ConfidenceFactors, ConfidenceInput, ConfidenceScore,
    ConfidenceThresholds,
};
pub use contradictions::{
    Contradiction, ContradictionAnalysis, ContradictionConfig, ContradictionDetector,
    ContradictionType,
};
pub use discovery_engine::{
    DiscoveryEngine, DiscoveryEngineConfig, DiscoveryResult, DiscoveryStats, DiscoverySuggestion,
    GapType, KnowledgeGap, PatternMatch, RegisterTripleInput, SemanticCluster, SuggestionType,
};
pub use phi_integration::{
    apply_consciousness_to_confidence, apply_phi_to_confidence, coherence_allows_operation,
    phi_to_confidence_boost, CoherenceState, ConsciousnessMetrics, PhiConfidenceFactors,
};
pub use phi_query_router::{
    deduct_query_cost, CoherenceWeightedResult, PhiQuery, PhiQueryResult, PhiQueryRouter,
    QueryCostDeduction, QueryRestriction, QueryRouterConfig, QueryRoutingDecision, QueryType,
};
pub use propagation_engine::{
    ConfidencePropagation, InferenceResult, InferenceRule, PropagationEngine,
    PropagationEngineConfig, PropagationNode, PropagationPath, PropagationResult, PropagationStats,
};
pub use query::{
    query_by_predicate, query_by_subject, query_triples, AgentKreditBalance, KreditQueryExecutor,
    KreditQueryResult, ObjectFilter, QueryCostPreview, QueryFilter, WeightedQueryResult,
};
pub use query_index::{
    CachedTripleIndex, IndexStats, PaginatedResult, QueryCache, QueryCacheStats, TripleIndex,
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Epistemic classification of claims
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum EpistemicType {
    /// Empirical: Verifiable through observation/experiment
    #[default]
    Empirical,
    /// Normative: Value judgments, prescriptive statements
    Normative,
    /// Metaphysical: Unfalsifiable claims about fundamental reality
    Metaphysical,
}

/// URI reference for predicates and sources
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct URI {
    /// The URL string
    pub url: String,
}

impl URI {
    /// Create a new URI
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Extract domain from URL for source quality scoring
    pub fn domain(&self) -> &str {
        if let Some(domain_start) = self.url.find("://") {
            let after_protocol = &self.url[domain_start + 3..];
            if let Some(domain_end) = after_protocol.find('/') {
                &after_protocol[..domain_end]
            } else {
                after_protocol
            }
        } else {
            &self.url
        }
    }
}

impl From<&str> for URI {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Value types for triple objects
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum TripleValue {
    /// String value
    String(String),
    /// Floating-point number
    Float(f64),
    /// Integer value
    Integer(i64),
    /// Boolean value
    Boolean(bool),
    /// Reference to another entity (by hash)
    Reference(String),
}

impl TripleValue {
    /// Get as string reference if this is a String variant
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as float, converting integers if necessary
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// A verifiable triple representing a knowledge claim
///
/// Triples follow the RDF-like structure: (subject, predicate, object)
/// with additional metadata for provenance and confidence calculation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct VerifiableTriple {
    /// The subject of the claim (entity identifier)
    pub subject: String,
    /// The predicate (relationship/property URI)
    pub predicate: URI,
    /// The object (value being claimed)
    pub object: TripleValue,
    /// Epistemic classification
    pub epistemic_type: EpistemicType,
    /// Domain for domain-specific confidence adjustments
    pub domain: Option<String>,
    /// Source URIs supporting this claim
    pub sources: Vec<URI>,
    /// Unix timestamp when claim was created
    pub created_at: u64,
    /// Original creator's identifier
    pub creator: String,
}

impl VerifiableTriple {
    /// Create a new verifiable triple
    pub fn new(subject: impl Into<String>, predicate: impl Into<URI>, object: TripleValue) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object,
            epistemic_type: EpistemicType::default(),
            domain: None,
            sources: Vec::new(),
            created_at: 0,
            creator: String::new(),
        }
    }

    /// Builder: set epistemic type
    pub fn with_epistemic_type(mut self, t: EpistemicType) -> Self {
        self.epistemic_type = t;
        self
    }

    /// Builder: set domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Builder: add source
    pub fn with_source(mut self, source: impl Into<URI>) -> Self {
        self.sources.push(source.into());
        self
    }

    /// Builder: set creation time
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.created_at = ts;
        self
    }

    /// Builder: set creator
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = creator.into();
        self
    }

    /// Validate the triple has required fields
    pub fn validate(&self) -> Result<(), String> {
        if self.subject.is_empty() {
            return Err("Subject cannot be empty".into());
        }
        if self.subject.len() > 256 {
            return Err("Subject too long (max 256 bytes)".into());
        }
        if self.predicate.url.is_empty() {
            return Err("Predicate cannot be empty".into());
        }
        if self.predicate.url.len() > 512 {
            return Err("Predicate URI too long (max 512 bytes)".into());
        }
        if self.sources.len() > 20 {
            return Err("Too many sources (max 20)".into());
        }
        Ok(())
    }
}

/// Stored triple with computed metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct StoredTriple {
    /// The underlying verifiable triple
    pub triple: VerifiableTriple,
    /// Hash of the triple (for references)
    pub hash: String,
    /// Number of attestations received
    pub attestation_count: u32,
    /// Last computed confidence score
    pub cached_confidence: f64,
    /// When confidence was last computed
    pub confidence_computed_at: u64,
}

impl StoredTriple {
    /// Create a new stored triple
    pub fn new(triple: VerifiableTriple, hash: String) -> Self {
        Self {
            triple,
            hash,
            attestation_count: 0,
            cached_confidence: 0.0,
            confidence_computed_at: 0,
        }
    }

    /// Check if cached confidence is stale (older than 1 hour)
    pub fn is_confidence_stale(&self, current_time: u64) -> bool {
        current_time.saturating_sub(self.confidence_computed_at) > 3600
    }
}

/// Configuration for DKG operations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct DKGConfig {
    /// Maximum sources per triple
    pub max_sources: usize,
    /// Maximum attestations to track per triple
    pub max_attestations: usize,
    /// Confidence cache TTL in seconds
    pub confidence_cache_ttl: u64,
    /// Minimum confidence to consider a claim "verified"
    pub verification_threshold: f64,
    /// Time decay half-life in days (domain-specific)
    pub default_half_life_days: u64,
}

impl Default for DKGConfig {
    fn default() -> Self {
        Self {
            max_sources: 20,
            max_attestations: 100,
            confidence_cache_ttl: 3600,
            verification_threshold: 0.75,
            default_half_life_days: 365,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_domain_extraction() {
        let uri = URI::new("https://arxiv.org/abs/1234");
        assert_eq!(uri.domain(), "arxiv.org");

        let uri2 = URI::new("http://example.com/path/to/resource");
        assert_eq!(uri2.domain(), "example.com");

        let uri3 = URI::new("simple-string");
        assert_eq!(uri3.domain(), "simple-string");
    }

    #[test]
    fn test_triple_validation() {
        let valid = VerifiableTriple::new(
            "subject123",
            "predicate:type",
            TripleValue::String("value".into()),
        );
        assert!(valid.validate().is_ok());

        let empty_subject =
            VerifiableTriple::new("", "predicate:type", TripleValue::String("value".into()));
        assert!(empty_subject.validate().is_err());

        let empty_predicate =
            VerifiableTriple::new("subject", "", TripleValue::String("value".into()));
        assert!(empty_predicate.validate().is_err());
    }

    #[test]
    fn test_triple_builder() {
        let triple = VerifiableTriple::new(
            "gradient:abc123",
            "pogq:quality_score",
            TripleValue::Float(0.95),
        )
        .with_epistemic_type(EpistemicType::Empirical)
        .with_domain("federated_learning")
        .with_source("https://arxiv.org/paper/123")
        .with_creator("validator:alice")
        .with_timestamp(1700000000);

        assert_eq!(triple.subject, "gradient:abc123");
        assert_eq!(triple.epistemic_type, EpistemicType::Empirical);
        assert_eq!(triple.domain, Some("federated_learning".into()));
        assert_eq!(triple.sources.len(), 1);
        assert_eq!(triple.creator, "validator:alice");
        assert_eq!(triple.created_at, 1700000000);
    }

    #[test]
    fn test_triple_value_conversions() {
        let str_val = TripleValue::String("hello".into());
        assert_eq!(str_val.as_string(), Some("hello"));
        assert_eq!(str_val.as_float(), None);

        let float_val = TripleValue::Float(3.14);
        assert_eq!(float_val.as_string(), None);
        assert!((float_val.as_float().unwrap() - 3.14).abs() < 0.001);

        let int_val = TripleValue::Integer(42);
        assert!((int_val.as_float().unwrap() - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_stored_triple_staleness() {
        let triple = VerifiableTriple::new("s", "p", TripleValue::Boolean(true));
        let mut stored = StoredTriple::new(triple, "hash123".into());
        stored.confidence_computed_at = 1000;

        // Not stale if within 1 hour
        assert!(!stored.is_confidence_stale(2000));

        // Stale if over 1 hour (3600 seconds)
        assert!(stored.is_confidence_stale(5000));
    }
}
