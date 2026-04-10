// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared types for Symthaea Prism.
//!
//! Prism is a consciousness-aware pure Rust browser with epistemic search,
//! powered by Symthaea.

use serde::{Deserialize, Serialize};

/// Privacy zone classification for web content.
///
/// Default is `Local`. Content must be explicitly upgraded to `Public`
/// via E3+/E4 epistemic classification or explicit user consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum ContentZone {
    /// Never encoded, never shared. Banks, email, corporate portals.
    Private,
    /// HDC encoded for personal semantic memory. Never broadcast to DHT.
    #[default]
    Local,
    /// Encoded and offered to DHT with E/N/M classification.
    Public,
}


/// NRC-inspired safety level (Nuclear Regulatory Commission graduated response).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyLevel {
    /// Normal operation.
    Green = 0,
    /// Heightened awareness — monitor, reduce learning rate.
    Yellow = 1,
    /// Active intervention — restrict output, reduce capabilities.
    Orange = 2,
    /// Emergency halt — block content, minimal response.
    Red = 3,
}

impl PartialOrd for SafetyLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SafetyLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

/// Threat categories detected by the reflex arc or immune system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatType {
    /// Prompt injection attempts ("ignore previous instructions").
    Adversarial,
    /// Harmful content ("exploit", "weapon").
    Harmful,
    /// Deception attempts ("pretend", "roleplay").
    Deceptive,
    /// Credential/boundary probing ("password", "SSN").
    Boundary,
    /// High non-alphanumeric ratio (< 30% alpha chars).
    Incoherent,
    /// Input exceeding size limits.
    Overload,
    /// Unclassified anomaly.
    Unknown,
}

// Re-export unified epistemic types from the shared crate.
// Previously defined locally as EmpiricalLevel only; now includes full E/N/M.
pub use mycelix_claim_types::{
    EmpiricalLevel, NormativeLevel, MaterialityLevel,
    ClaimType, EpistemicClassification, UnifiedClaim,
};

/// A search result from the epistemic knowledge engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The claim text.
    pub content: String,
    /// Original source URL(s).
    pub sources: Vec<String>,
    /// Empirical evidence level (E0-E4).
    pub empirical_level: EmpiricalLevel,
    /// Normative consensus level (N0-N3).
    pub normative_level: NormativeLevel,
    /// Materiality / real-world impact (M0-M3).
    pub materiality_level: MaterialityLevel,
    /// HDC cosine similarity to query (0.0-1.0).
    pub query_similarity: f32,
    /// Author reputation from ConsciousnessProfile (0.0-1.0).
    pub author_reputation: f32,
    /// Age in days since claim creation.
    pub age_days: u32,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl SearchResult {
    /// Composite ranking score.
    /// Weights: 0.4 relevance + 0.3 epistemic + 0.2 trust + 0.1 freshness.
    pub fn rank_score(&self) -> f32 {
        let freshness = 1.0 / (1.0 + self.age_days as f32 / 365.0);
        0.4 * self.query_similarity
            + 0.3 * self.empirical_level.as_f32()
            + 0.2 * self.author_reputation
            + 0.1 * freshness
    }
}

/// Node participation tier in the P2P search network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeTier {
    /// Browser users. Query + submit. No persistent storage for others.
    Light,
    /// Community operators. Serve LSH shards. TEND-incentivized. K-vector >= 0.4.
    Relay,
    /// Institutions/dedicated. Full index replicas. K-vector >= 0.6.
    Archive,
}

/// Unique identifier for a browser tab.
pub type TabId = u32;

/// Unique identifier for a search query.
pub type QueryId = u64;
