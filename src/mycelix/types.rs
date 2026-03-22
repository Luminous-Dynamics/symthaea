// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for Symthaea-Mycelix integration.
//!
//! These types bridge Symthaea's consciousness model with Mycelix's
//! epistemic classification system (E/N/M).
//!
//! The canonical definitions for the Epistemic Cube (Empirical,
//! Normative, Materiality) live in the `mycelix-sdk` crate. The enums
//! in this module are a thin, Symthaea-focused veneer with explicit
//! conversion implementations to and from `mycelix_sdk::epistemic`
//! when the `mycelix` feature is enabled.

use std::time::{Duration, SystemTime};

/// Empirical Level - How verifiable is this claim?
///
/// Corresponds to Mycelix UESS E-levels:
/// - E0: Subjective experience (no external verification possible)
/// - E1: Witnessed/Testimonial (trusted party attests)
/// - E2: Privately verifiable (holder can prove to verifier)
/// - E3: Cryptographically verifiable (ZK proofs, signatures)
/// - E4: Publicly reproducible (anyone can independently verify)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum EmpiricalLevel {
    /// E0: Subjective - Cannot be externally verified
    /// Example: "I feel this is true"
    Subjective = 0,

    /// E1: Testimonial - Trusted party attests
    /// Example: "Expert X said this"
    Testimonial = 1,

    /// E2: Privately Verifiable - Holder can prove to verifier
    /// Example: "I can show you my credentials"
    PrivatelyVerifiable = 2,

    /// E3: Cryptographically Verifiable - ZK/digital signatures
    /// Example: "Here's a cryptographic proof"
    CryptographicallyVerifiable = 3,

    /// E4: Publicly Reproducible - Anyone can verify independently
    /// Example: "Run this experiment yourself"
    PubliclyReproducible = 4,
}

impl EmpiricalLevel {
    /// Create from Φ (consciousness integration) value
    ///
    /// Higher Φ indicates more integrated/coherent reasoning,
    /// which correlates with higher verifiability.
    pub fn from_phi(phi: f32, has_reproducibility: bool) -> Self {
        match phi {
            p if p >= 0.4 && has_reproducibility => Self::PubliclyReproducible,
            p if p >= 0.3 => Self::CryptographicallyVerifiable,
            p if p >= 0.2 => Self::PrivatelyVerifiable,
            p if p >= 0.1 => Self::Testimonial,
            _ => Self::Subjective,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Subjective => "Subjective experience, not externally verifiable",
            Self::Testimonial => "Attested by trusted party",
            Self::PrivatelyVerifiable => "Can be privately verified by holder",
            Self::CryptographicallyVerifiable => "Cryptographically provable",
            Self::PubliclyReproducible => "Independently reproducible by anyone",
        }
    }
}

/// Normative Level - Who should have access to this?
///
/// Corresponds to Mycelix UESS N-levels:
/// - N0: Internal only (never leaves the agent)
/// - N1: Agent-to-agent (trusted network)
/// - N2: Network-wide (full DHT)
/// - N3: Foundational/Axiomatic (maximum distribution)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum NormativeLevel {
    /// N0: Internal - Never leaves the agent
    /// Example: Private thoughts, personal secrets
    Internal = 0,

    /// N1: Agent - Shared with trusted peers only
    /// Example: Private messages, personal reputation
    Agent = 1,

    /// N2: Network - Shared across the full network
    /// Example: Public claims, shared knowledge
    Network = 2,

    /// N3: Foundational - Maximum distribution, axiomatic truths
    /// Example: Mathematical axioms, physical laws
    Foundational = 3,
}

impl NormativeLevel {
    /// Create from workspace scope
    pub fn from_scope(scope: WorkspaceScope) -> Self {
        match scope {
            WorkspaceScope::Internal => Self::Internal,
            WorkspaceScope::Local => Self::Agent,
            WorkspaceScope::Network => Self::Network,
            WorkspaceScope::Universal => Self::Foundational,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Internal => "Internal only, never shared",
            Self::Agent => "Shared with trusted peers",
            Self::Network => "Network-wide distribution",
            Self::Foundational => "Universal, axiomatic truth",
        }
    }
}

/// Materiality Level - How long should this persist?
///
/// Corresponds to Mycelix UESS M-levels:
/// - M0: Ephemeral (session-only)
/// - M1: Short-term (hours to days)
/// - M2: Medium-term (months to years)
/// - M3: Permanent (with cooling period)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MaterialityLevel {
    /// M0: Ephemeral - Session only, no persistence
    Ephemeral = 0,

    /// M1: ShortTerm - Hours to days
    ShortTerm = 1,

    /// M2: MediumTerm - Months to years
    MediumTerm = 2,

    /// M3: Permanent - Long-term with cooling period
    Permanent = 3,
}

impl MaterialityLevel {
    /// Create from importance score (0.0 - 1.0)
    pub fn from_importance(importance: f32) -> Self {
        match importance {
            i if i >= 0.75 => Self::Permanent,
            i if i >= 0.50 => Self::MediumTerm,
            i if i >= 0.25 => Self::ShortTerm,
            _ => Self::Ephemeral,
        }
    }

    /// Get suggested TTL
    pub fn suggested_ttl(&self) -> Option<Duration> {
        match self {
            Self::Ephemeral => Some(Duration::from_secs(3600)), // 1 hour
            Self::ShortTerm => Some(Duration::from_secs(86400 * 7)), // 1 week
            Self::MediumTerm => Some(Duration::from_secs(86400 * 365)), // 1 year
            Self::Permanent => None,                            // No expiry
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ephemeral => "Session-only, expires within hours",
            Self::ShortTerm => "Short-term, expires within days",
            Self::MediumTerm => "Medium-term, persists for months",
            Self::Permanent => "Permanent storage with cooling period",
        }
    }
}

// ============================================================================
// Interop with canonical Mycelix Epistemic types
// ============================================================================

#[cfg(feature = "mycelix_sdk")]
impl From<mycelix_sdk::epistemic::EmpiricalLevel> for EmpiricalLevel {
    fn from(level: mycelix_sdk::epistemic::EmpiricalLevel) -> Self {
        use mycelix_sdk::epistemic::EmpiricalLevel as E;
        match level {
            E::E0Null => EmpiricalLevel::Subjective,
            E::E1Testimonial => EmpiricalLevel::Testimonial,
            E::E2PrivateVerify => EmpiricalLevel::PrivatelyVerifiable,
            E::E3Cryptographic => EmpiricalLevel::CryptographicallyVerifiable,
            E::E4PublicRepro => EmpiricalLevel::PubliclyReproducible,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<EmpiricalLevel> for mycelix_sdk::epistemic::EmpiricalLevel {
    fn from(level: EmpiricalLevel) -> Self {
        use mycelix_sdk::epistemic::EmpiricalLevel as E;
        match level {
            EmpiricalLevel::Subjective => E::E0Null,
            EmpiricalLevel::Testimonial => E::E1Testimonial,
            EmpiricalLevel::PrivatelyVerifiable => E::E2PrivateVerify,
            EmpiricalLevel::CryptographicallyVerifiable => E::E3Cryptographic,
            EmpiricalLevel::PubliclyReproducible => E::E4PublicRepro,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<mycelix_sdk::epistemic::NormativeLevel> for NormativeLevel {
    fn from(level: mycelix_sdk::epistemic::NormativeLevel) -> Self {
        use mycelix_sdk::epistemic::NormativeLevel as N;
        match level {
            N::N0Personal => NormativeLevel::Internal,
            N::N1Communal => NormativeLevel::Agent,
            N::N2Network => NormativeLevel::Network,
            N::N3Axiomatic => NormativeLevel::Foundational,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<NormativeLevel> for mycelix_sdk::epistemic::NormativeLevel {
    fn from(level: NormativeLevel) -> Self {
        use mycelix_sdk::epistemic::NormativeLevel as N;
        match level {
            NormativeLevel::Internal => N::N0Personal,
            NormativeLevel::Agent => N::N1Communal,
            NormativeLevel::Network => N::N2Network,
            NormativeLevel::Foundational => N::N3Axiomatic,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<mycelix_sdk::epistemic::MaterialityLevel> for MaterialityLevel {
    fn from(level: mycelix_sdk::epistemic::MaterialityLevel) -> Self {
        use mycelix_sdk::epistemic::MaterialityLevel as M;
        match level {
            M::M0Ephemeral => MaterialityLevel::Ephemeral,
            M::M1Temporal => MaterialityLevel::ShortTerm,
            M::M2Persistent => MaterialityLevel::MediumTerm,
            M::M3Foundational => MaterialityLevel::Permanent,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<MaterialityLevel> for mycelix_sdk::epistemic::MaterialityLevel {
    fn from(level: MaterialityLevel) -> Self {
        use mycelix_sdk::epistemic::MaterialityLevel as M;
        match level {
            MaterialityLevel::Ephemeral => M::M0Ephemeral,
            MaterialityLevel::ShortTerm => M::M1Temporal,
            MaterialityLevel::MediumTerm => M::M2Persistent,
            MaterialityLevel::Permanent => M::M3Foundational,
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<mycelix_sdk::epistemic::EpistemicClassification> for EpistemicClassification {
    fn from(class: mycelix_sdk::epistemic::EpistemicClassification) -> Self {
        EpistemicClassification {
            empirical: EmpiricalLevel::from(class.empirical),
            normative: NormativeLevel::from(class.normative),
            materiality: MaterialityLevel::from(class.materiality),
        }
    }
}

#[cfg(feature = "mycelix_sdk")]
impl From<EpistemicClassification> for mycelix_sdk::epistemic::EpistemicClassification {
    fn from(class: EpistemicClassification) -> Self {
        mycelix_sdk::epistemic::EpistemicClassification::new(
            class.empirical.into(),
            class.normative.into(),
            class.materiality.into(),
        )
    }
}

/// Workspace scope for normative level mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceScope {
    /// Internal processing only
    Internal,
    /// Local agent context
    Local,
    /// Network-wide scope
    Network,
    /// Universal/foundational
    Universal,
}

/// Complete epistemic classification (E/N/M)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicClassification {
    /// Empirical level (verifiability)
    pub empirical: EmpiricalLevel,
    /// Normative level (access scope)
    pub normative: NormativeLevel,
    /// Materiality level (persistence)
    pub materiality: MaterialityLevel,
}

impl EpistemicClassification {
    /// Create a new classification
    pub fn new(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
        }
    }

    /// Get the classification code (e.g., "E2/N1/M2")
    pub fn code(&self) -> String {
        format!(
            "E{}/N{}/M{}",
            self.empirical as u8, self.normative as u8, self.materiality as u8
        )
    }

    /// Parse from code string
    pub fn from_code(code: &str) -> Option<Self> {
        let parts: Vec<&str> = code.split('/').collect();
        if parts.len() != 3 {
            return None;
        }

        let e = parts[0].strip_prefix('E')?.parse::<u8>().ok()?;
        let n = parts[1].strip_prefix('N')?.parse::<u8>().ok()?;
        let m = parts[2].strip_prefix('M')?.parse::<u8>().ok()?;

        Some(Self {
            empirical: match e {
                0 => EmpiricalLevel::Subjective,
                1 => EmpiricalLevel::Testimonial,
                2 => EmpiricalLevel::PrivatelyVerifiable,
                3 => EmpiricalLevel::CryptographicallyVerifiable,
                4 => EmpiricalLevel::PubliclyReproducible,
                _ => return None,
            },
            normative: match n {
                0 => NormativeLevel::Internal,
                1 => NormativeLevel::Agent,
                2 => NormativeLevel::Network,
                3 => NormativeLevel::Foundational,
                _ => return None,
            },
            materiality: match m {
                0 => MaterialityLevel::Ephemeral,
                1 => MaterialityLevel::ShortTerm,
                2 => MaterialityLevel::MediumTerm,
                3 => MaterialityLevel::Permanent,
                _ => return None,
            },
        })
    }
}

/// Evidence supporting a claim
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Unique identifier
    pub id: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Content or reference
    pub content: String,
    /// Strength of evidence (0.0 - 1.0)
    pub strength: f32,
    /// When was this evidence collected?
    pub timestamp: SystemTime,
    /// Source agent (if applicable)
    pub source: Option<String>,
}

/// Types of evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceType {
    /// Direct observation
    Observation,
    /// Testimonial from trusted party
    Testimony,
    /// Logical derivation
    Derivation,
    /// Cryptographic proof
    CryptographicProof,
    /// Experimental result
    Experiment,
    /// Consensus from multiple sources
    Consensus,
}

/// A conscious output with epistemic classification
#[derive(Debug, Clone)]
pub struct ClassifiedConsciousOutput {
    /// The output content
    pub content: String,

    /// HDC semantic embedding (from Symthaea)
    pub semantic_embedding: Option<Vec<f32>>,

    /// Φ (phi) value - consciousness integration measure
    pub phi: f32,

    /// Epistemic classification (E/N/M)
    pub classification: EpistemicClassification,

    /// Supporting evidence
    pub evidence: Vec<Evidence>,

    /// Confidence in this output (0.0 - 1.0)
    pub confidence: f32,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Source workspace
    pub workspace_scope: WorkspaceScope,

    /// Whether this output is reproducible
    pub is_reproducible: bool,

    /// Importance score (for M-level)
    pub importance: f32,
}

impl ClassifiedConsciousOutput {
    /// Create a new classified output
    pub fn new(
        content: String,
        phi: f32,
        workspace_scope: WorkspaceScope,
        importance: f32,
        is_reproducible: bool,
    ) -> Self {
        let classification = EpistemicClassification {
            empirical: EmpiricalLevel::from_phi(phi, is_reproducible),
            normative: NormativeLevel::from_scope(workspace_scope),
            materiality: MaterialityLevel::from_importance(importance),
        };

        Self {
            content,
            semantic_embedding: None,
            phi,
            classification,
            evidence: Vec::new(),
            confidence: phi, // Default confidence to Φ
            timestamp: SystemTime::now(),
            workspace_scope,
            is_reproducible,
            importance,
        }
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Set semantic embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.semantic_embedding = Some(embedding);
        self
    }

    /// Recalculate classification based on current state
    pub fn reclassify(&mut self) {
        self.classification = EpistemicClassification {
            empirical: EmpiricalLevel::from_phi(self.phi, self.is_reproducible),
            normative: NormativeLevel::from_scope(self.workspace_scope),
            materiality: MaterialityLevel::from_importance(self.importance),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_from_phi() {
        assert_eq!(
            EmpiricalLevel::from_phi(0.05, false),
            EmpiricalLevel::Subjective
        );
        assert_eq!(
            EmpiricalLevel::from_phi(0.15, false),
            EmpiricalLevel::Testimonial
        );
        assert_eq!(
            EmpiricalLevel::from_phi(0.25, false),
            EmpiricalLevel::PrivatelyVerifiable
        );
        assert_eq!(
            EmpiricalLevel::from_phi(0.35, false),
            EmpiricalLevel::CryptographicallyVerifiable
        );
        assert_eq!(
            EmpiricalLevel::from_phi(0.45, true),
            EmpiricalLevel::PubliclyReproducible
        );
        assert_eq!(
            EmpiricalLevel::from_phi(0.45, false),
            EmpiricalLevel::CryptographicallyVerifiable
        );
    }

    #[test]
    fn test_classification_code() {
        let class = EpistemicClassification::new(
            EmpiricalLevel::CryptographicallyVerifiable,
            NormativeLevel::Network,
            MaterialityLevel::MediumTerm,
        );
        assert_eq!(class.code(), "E3/N2/M2");

        let parsed = EpistemicClassification::from_code("E3/N2/M2").unwrap();
        assert_eq!(
            parsed.empirical,
            EmpiricalLevel::CryptographicallyVerifiable
        );
        assert_eq!(parsed.normative, NormativeLevel::Network);
        assert_eq!(parsed.materiality, MaterialityLevel::MediumTerm);
    }

    #[test]
    fn test_materiality_ttl() {
        assert!(MaterialityLevel::Ephemeral.suggested_ttl().unwrap() < Duration::from_secs(86400));
        assert!(
            MaterialityLevel::ShortTerm.suggested_ttl().unwrap() <= Duration::from_secs(86400 * 7)
        );
        assert!(MaterialityLevel::Permanent.suggested_ttl().is_none());
    }

    #[cfg(feature = "mycelix_sdk")]
    #[test]
    fn test_epistemic_level_roundtrip_with_mycelix_sdk() {
        use mycelix_sdk::epistemic as m;

        // Empirical
        let empirical_levels = [
            EmpiricalLevel::Subjective,
            EmpiricalLevel::Testimonial,
            EmpiricalLevel::PrivatelyVerifiable,
            EmpiricalLevel::CryptographicallyVerifiable,
            EmpiricalLevel::PubliclyReproducible,
        ];
        for lvl in empirical_levels {
            let canonical: m::EmpiricalLevel = lvl.into();
            let back: EmpiricalLevel = canonical.into();
            assert_eq!(back, lvl);
        }

        // Normative
        let normative_levels = [
            NormativeLevel::Internal,
            NormativeLevel::Agent,
            NormativeLevel::Network,
            NormativeLevel::Foundational,
        ];
        for lvl in normative_levels {
            let canonical: m::NormativeLevel = lvl.into();
            let back: NormativeLevel = canonical.into();
            assert_eq!(back, lvl);
        }

        // Materiality
        let materiality_levels = [
            MaterialityLevel::Ephemeral,
            MaterialityLevel::ShortTerm,
            MaterialityLevel::MediumTerm,
            MaterialityLevel::Permanent,
        ];
        for lvl in materiality_levels {
            let canonical: m::MaterialityLevel = lvl.into();
            let back: MaterialityLevel = canonical.into();
            assert_eq!(back, lvl);
        }

        // Classification
        let class = EpistemicClassification::new(
            EmpiricalLevel::CryptographicallyVerifiable,
            NormativeLevel::Network,
            MaterialityLevel::MediumTerm,
        );
        let canonical: m::EpistemicClassification = class.clone().into();
        let back: EpistemicClassification = canonical.into();
        assert_eq!(back.empirical, class.empirical);
        assert_eq!(back.normative, class.normative);
        assert_eq!(back.materiality, class.materiality);
    }
}
