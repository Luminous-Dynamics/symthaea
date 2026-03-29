// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Claims
//!
//! Structured claims with epistemic classification and evidence.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use super::cube::{EmpiricalLevel, EpistemicClassification, MaterialityLevel, NormativeLevel};

/// A claim with epistemic classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpistemicClaim {
    /// The content of the claim
    pub content: String,

    /// Empirical classification
    pub empirical: EmpiricalLevel,

    /// Normative classification
    pub normative: NormativeLevel,

    /// Materiality classification
    pub materiality: MaterialityLevel,

    /// Supporting evidence (hashes, URLs, etc.)
    pub evidence: Vec<String>,

    /// Issuer identifier (DID, pubkey, etc.)
    pub issuer: String,

    /// Unix timestamp of creation
    pub timestamp: u64,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl EpistemicClaim {
    /// Create a new epistemic claim
    pub fn new(
        content: impl Into<String>,
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            content: content.into(),
            empirical,
            normative,
            materiality,
            evidence: Vec::new(),
            issuer: String::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            metadata: None,
        }
    }

    /// Get the full classification
    pub fn classification(&self) -> EpistemicClassification {
        EpistemicClassification::new(self.empirical, self.normative, self.materiality)
    }

    /// Check if claim meets minimum standards
    pub fn meets_standard(&self, min_e: EmpiricalLevel, min_n: NormativeLevel) -> bool {
        self.empirical.meets_minimum(min_e) && self.normative.meets_minimum(min_n)
    }

    /// Check if claim should be retained based on materiality and time
    pub fn should_retain(&self, current_time: u64) -> bool {
        match self.materiality {
            MaterialityLevel::M0Ephemeral => false,
            MaterialityLevel::M1Temporal => true, // Caller decides based on state
            MaterialityLevel::M2Persistent => {
                // Retain for suggested period
                if let Some(days) = self.materiality.suggested_retention_days() {
                    let retention_secs = days * 86400;
                    current_time - self.timestamp < retention_secs
                } else {
                    true
                }
            }
            MaterialityLevel::M3Foundational => true,
        }
    }

    /// Add evidence to the claim
    pub fn add_evidence(&mut self, evidence: impl Into<String>) {
        self.evidence.push(evidence.into());
    }

    /// Set the issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get age in seconds
    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.timestamp)
    }

    /// Get classification code string
    pub fn code(&self) -> String {
        self.classification().to_code()
    }
}

/// Builder for epistemic claims
pub struct ClaimBuilder {
    content: String,
    empirical: EmpiricalLevel,
    normative: NormativeLevel,
    materiality: MaterialityLevel,
    evidence: Vec<String>,
    issuer: String,
    metadata: Option<serde_json::Value>,
}

impl ClaimBuilder {
    /// Start building a claim
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            empirical: EmpiricalLevel::E0Null,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M0Ephemeral,
            evidence: Vec::new(),
            issuer: String::new(),
            metadata: None,
        }
    }

    /// Set empirical level
    pub fn empirical(mut self, level: EmpiricalLevel) -> Self {
        self.empirical = level;
        self
    }

    /// Set normative level
    pub fn normative(mut self, level: NormativeLevel) -> Self {
        self.normative = level;
        self
    }

    /// Set materiality level
    pub fn materiality(mut self, level: MaterialityLevel) -> Self {
        self.materiality = level;
        self
    }

    /// Add evidence
    pub fn evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    /// Set issuer
    pub fn issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build the claim
    pub fn build(self) -> EpistemicClaim {
        let mut claim = EpistemicClaim::new(
            self.content,
            self.empirical,
            self.normative,
            self.materiality,
        );
        claim.evidence = self.evidence;
        claim.issuer = self.issuer;
        claim.metadata = self.metadata;
        claim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_claim() {
        let claim = EpistemicClaim::new(
            "Test claim",
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        );

        assert_eq!(claim.content, "Test claim");
        assert_eq!(claim.empirical, EmpiricalLevel::E2PrivateVerify);
        assert_eq!(claim.code(), "E2-N1-M1");
    }

    #[test]
    fn test_meets_standard() {
        let claim = EpistemicClaim::new(
            "Signed document",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        assert!(claim.meets_standard(EmpiricalLevel::E2PrivateVerify, NormativeLevel::N1Communal));

        assert!(!claim.meets_standard(EmpiricalLevel::E4PublicRepro, NormativeLevel::N1Communal));
    }

    #[test]
    fn test_builder() {
        let claim = ClaimBuilder::new("Built claim")
            .empirical(EmpiricalLevel::E3Cryptographic)
            .normative(NormativeLevel::N2Network)
            .materiality(MaterialityLevel::M2Persistent)
            .issuer("did:example:123")
            .evidence("hash:abc123")
            .build();

        assert_eq!(claim.content, "Built claim");
        assert_eq!(claim.issuer, "did:example:123");
        assert_eq!(claim.evidence.len(), 1);
    }

    #[test]
    fn test_add_evidence() {
        let mut claim = EpistemicClaim::new(
            "Claim with evidence",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        claim.add_evidence("sig:abc");
        claim.add_evidence("hash:def");

        assert_eq!(claim.evidence.len(), 2);
    }
}
