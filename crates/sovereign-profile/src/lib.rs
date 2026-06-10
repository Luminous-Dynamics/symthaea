// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # 8-Dimensional Sovereign Profile
//!
//! Anti-tyranny civic identity for the Mycelix governance system.
//!
//! When you reduce a human being to a single number — like a credit score or
//! social credit rating — you create a system that is infinitely gamifiable
//! and inherently oppressive. The 8D Sovereign Profile maps identity as a
//! holographic, multi-faceted geometry rather than a flat number.
//!
//! ## The 8 Axes of Civic Identity
//!
//! 1. **Epistemic Integrity** — truth/claim verification history
//! 2. **Thermodynamic Yield** — physical energy contribution to local commons
//! 3. **Network Resilience** — node uptime and bandwidth provision
//! 4. **Economic Velocity** — compliance with anti-hoarding demurrage
//! 5. **Civic Participation** — sortition jury duty, voting, dispute resolution
//! 6. **Stewardship & Care** — verified physical labor on the commons
//! 7. **Semantic Resonance** — entanglement with community consensus
//! 8. **Domain Competence** — peer-verified specialization
//!
//! Each dimension is directly measurable from a primary source (smart meter,
//! node log, transaction ledger, jury record, telemetry shim). Multiple
//! pathways to citizenship are respected — a reclusive solar engineer and a
//! social mediator both pass the threshold through completely different
//! dimensional profiles.
//!
//! ## Privacy
//!
//! The profile never leaves the local device. ZKP circuits prove the
//! *aggregate* weight exceeds a threshold without revealing individual
//! dimensions. The network knows you are a net-positive citizen but has
//! no idea *how* you achieved that.

pub mod collectors;
pub mod compat;
pub mod decay;
#[cfg(feature = "hdc")]
pub mod hdc;
pub mod i18n;
pub mod weights;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The 8 axes of civic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SovereignDimension {
    /// D0: Truth/claim verification history.
    EpistemicIntegrity,
    /// D1: Physical energy contribution to local commons.
    ThermodynamicYield,
    /// D2: Node uptime and bandwidth provision.
    NetworkResilience,
    /// D3: Compliance with anti-hoarding demurrage.
    EconomicVelocity,
    /// D4: Sortition jury duty, voting, dispute resolution.
    CivicParticipation,
    /// D5: Verified physical labor on the commons.
    StewardshipCare,
    /// D6: Entanglement with community consensus.
    SemanticResonance,
    /// D7: Peer-verified specialization.
    DomainCompetence,
}

impl SovereignDimension {
    /// All 8 dimensions in canonical order.
    pub const ALL: [Self; 8] = [
        Self::EpistemicIntegrity,
        Self::ThermodynamicYield,
        Self::NetworkResilience,
        Self::EconomicVelocity,
        Self::CivicParticipation,
        Self::StewardshipCare,
        Self::SemanticResonance,
        Self::DomainCompetence,
    ];

    /// Index (0-7) for array-based access.
    pub fn index(&self) -> usize {
        match self {
            Self::EpistemicIntegrity => 0,
            Self::ThermodynamicYield => 1,
            Self::NetworkResilience => 2,
            Self::EconomicVelocity => 3,
            Self::CivicParticipation => 4,
            Self::StewardshipCare => 5,
            Self::SemanticResonance => 6,
            Self::DomainCompetence => 7,
        }
    }

    /// i18n key for this dimension.
    pub fn key(&self) -> &'static str {
        match self {
            Self::EpistemicIntegrity => "epistemic_integrity",
            Self::ThermodynamicYield => "thermodynamic_yield",
            Self::NetworkResilience => "network_resilience",
            Self::EconomicVelocity => "economic_velocity",
            Self::CivicParticipation => "civic_participation",
            Self::StewardshipCare => "stewardship_care",
            Self::SemanticResonance => "semantic_resonance",
            Self::DomainCompetence => "domain_competence",
        }
    }
}

// ---------------------------------------------------------------------------
// SovereignProfile
// ---------------------------------------------------------------------------

/// 8-dimensional sovereign civic profile.
///
/// Each dimension is normalized to [0.0, 1.0]. Values outside this range
/// are clamped during scoring. NaN and Inf are treated as 0.0.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SovereignProfile {
    /// D0: Truth/claim verification history.
    pub epistemic_integrity: f64,
    /// D1: Physical energy contribution to local commons.
    pub thermodynamic_yield: f64,
    /// D2: Node uptime and bandwidth provision.
    pub network_resilience: f64,
    /// D3: Compliance with anti-hoarding demurrage.
    pub economic_velocity: f64,
    /// D4: Sortition jury duty, voting, dispute resolution.
    pub civic_participation: f64,
    /// D5: Verified physical labor on the commons.
    pub stewardship_care: f64,
    /// D6: Entanglement with community consensus.
    pub semantic_resonance: f64,
    /// D7: Peer-verified specialization.
    pub domain_competence: f64,
}

impl SovereignProfile {
    /// Create a profile with all dimensions at zero (Observer).
    pub fn zero() -> Self {
        Self {
            epistemic_integrity: 0.0,
            thermodynamic_yield: 0.0,
            network_resilience: 0.0,
            economic_velocity: 0.0,
            civic_participation: 0.0,
            stewardship_care: 0.0,
            semantic_resonance: 0.0,
            domain_competence: 0.0,
        }
    }

    /// Get a dimension value by enum variant.
    pub fn get(&self, dim: SovereignDimension) -> f64 {
        match dim {
            SovereignDimension::EpistemicIntegrity => self.epistemic_integrity,
            SovereignDimension::ThermodynamicYield => self.thermodynamic_yield,
            SovereignDimension::NetworkResilience => self.network_resilience,
            SovereignDimension::EconomicVelocity => self.economic_velocity,
            SovereignDimension::CivicParticipation => self.civic_participation,
            SovereignDimension::StewardshipCare => self.stewardship_care,
            SovereignDimension::SemanticResonance => self.semantic_resonance,
            SovereignDimension::DomainCompetence => self.domain_competence,
        }
    }

    /// Set a dimension value by enum variant.
    pub fn set(&mut self, dim: SovereignDimension, value: f64) {
        let target = match dim {
            SovereignDimension::EpistemicIntegrity => &mut self.epistemic_integrity,
            SovereignDimension::ThermodynamicYield => &mut self.thermodynamic_yield,
            SovereignDimension::NetworkResilience => &mut self.network_resilience,
            SovereignDimension::EconomicVelocity => &mut self.economic_velocity,
            SovereignDimension::CivicParticipation => &mut self.civic_participation,
            SovereignDimension::StewardshipCare => &mut self.stewardship_care,
            SovereignDimension::SemanticResonance => &mut self.semantic_resonance,
            SovereignDimension::DomainCompetence => &mut self.domain_competence,
        };
        *target = value;
    }

    /// All 8 dimension values as an array in canonical order.
    pub fn as_array(&self) -> [f64; 8] {
        [
            self.epistemic_integrity,
            self.thermodynamic_yield,
            self.network_resilience,
            self.economic_velocity,
            self.civic_participation,
            self.stewardship_care,
            self.semantic_resonance,
            self.domain_competence,
        ]
    }

    /// Create from an array of 8 values in canonical order.
    pub fn from_array(values: [f64; 8]) -> Self {
        Self {
            epistemic_integrity: values[0],
            thermodynamic_yield: values[1],
            network_resilience: values[2],
            economic_velocity: values[3],
            civic_participation: values[4],
            stewardship_care: values[5],
            semantic_resonance: values[6],
            domain_competence: values[7],
        }
    }

    /// Sanitize a single dimension value: NaN/Inf → 0.0, then clamp [0.0, 1.0].
    fn sanitize(v: f64) -> f64 {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Combined score using the given dimension weights.
    ///
    /// Each dimension is sanitized (NaN/Inf → 0.0, clamped [0,1]) then
    /// multiplied by its weight. The result is clamped to [0.0, 1.0].
    pub fn combined_score(&self, weights: &weights::DimensionWeights) -> f64 {
        let dims = self.as_array();
        let w = &weights.weights;
        let mut score = 0.0;
        for i in 0..8 {
            score += Self::sanitize(dims[i]) * w[i];
        }
        score.clamp(0.0, 1.0)
    }

    /// Derive the civic tier from this profile using the given weights.
    pub fn tier(&self, weights: &weights::DimensionWeights) -> CivicTier {
        CivicTier::from_score(self.combined_score(weights))
    }

    /// Check whether this profile meets a civic requirement.
    pub fn meets_requirement(
        &self,
        requirement: &CivicRequirement,
        weights: &weights::DimensionWeights,
    ) -> bool {
        let tier = self.tier(weights);
        if tier < requirement.min_tier {
            return false;
        }
        for &(dim, min_value) in &requirement.min_dimensions {
            if Self::sanitize(self.get(dim)) < min_value {
                return false;
            }
        }
        true
    }
}

impl Default for SovereignProfile {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// CivicTier
// ---------------------------------------------------------------------------

/// Progressive civic tier derived from the combined 8D score.
///
/// Higher tiers grant greater governance capabilities. The tier ordering
/// is total and monotonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CivicTier {
    /// Read-only access. No governance participation.
    Observer,
    /// Basic proposals, commenting.
    Participant,
    /// Voting rights.
    Citizen,
    /// Constitutional amendments, budget decisions.
    Steward,
    /// Emergency powers, system administration.
    Guardian,
}

impl CivicTier {
    /// Minimum combined score for each tier.
    pub fn min_score(&self) -> f64 {
        match self {
            Self::Observer => 0.0,
            Self::Participant => 0.3,
            Self::Citizen => 0.4,
            Self::Steward => 0.6,
            Self::Guardian => 0.8,
        }
    }

    /// Derive tier from a combined score.
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Guardian
        } else if score >= 0.6 {
            Self::Steward
        } else if score >= 0.4 {
            Self::Citizen
        } else if score >= 0.3 {
            Self::Participant
        } else {
            Self::Observer
        }
    }

    /// Vote weight in basis points (1 bp = 0.01%).
    pub fn vote_weight_bp(&self) -> u32 {
        match self {
            Self::Observer => 0,
            Self::Participant => 5_000,
            Self::Citizen => 7_500,
            Self::Steward => 10_000,
            Self::Guardian => 10_000,
        }
    }

    /// i18n key for this tier.
    pub fn key(&self) -> &'static str {
        match self {
            Self::Observer => "observer",
            Self::Participant => "participant",
            Self::Citizen => "citizen",
            Self::Steward => "steward",
            Self::Guardian => "guardian",
        }
    }

    /// English label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Observer",
            Self::Participant => "Participant",
            Self::Citizen => "Citizen",
            Self::Steward => "Steward",
            Self::Guardian => "Guardian",
        }
    }

    /// CSS class suffix for UI styling.
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Observer => "observer",
            Self::Participant => "participant",
            Self::Citizen => "citizen",
            Self::Steward => "steward",
            Self::Guardian => "guardian",
        }
    }
}

// ---------------------------------------------------------------------------
// CivicRequirement
// ---------------------------------------------------------------------------

/// A governance requirement specifying minimum tier and optional per-dimension minimums.
///
/// An agent meets the requirement when:
/// 1. Their derived tier >= `min_tier`
/// 2. For each (dimension, threshold) in `min_dimensions`, the agent's
///    sanitized dimension value >= threshold
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CivicRequirement {
    pub min_tier: CivicTier,
    pub min_dimensions: Vec<(SovereignDimension, f64)>,
}

/// Requirement for basic civic participation (viewing proposals, commenting).
pub fn civic_requirement_basic() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Participant,
        min_dimensions: vec![],
    }
}

/// Requirement for submitting proposals.
pub fn civic_requirement_proposal() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Participant,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    }
}

/// Requirement for casting votes.
pub fn civic_requirement_voting() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    }
}

/// Requirement for constitutional changes (amendments, budget).
pub fn civic_requirement_constitutional() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Steward,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.5),
            (SovereignDimension::CivicParticipation, 0.3),
        ],
    }
}

/// Requirement for guardian-level operations (emergency powers).
pub fn civic_requirement_guardian() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Guardian,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.7),
            (SovereignDimension::CivicParticipation, 0.5),
        ],
    }
}

// ---------------------------------------------------------------------------
// SovereignCredential
// ---------------------------------------------------------------------------

/// A time-limited, cryptographically-bound civic credential.
///
/// Issued by the identity bridge after gathering all 8 dimensions from
/// their source clusters. Evaluated locally by pure functions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SovereignCredential {
    /// Decentralized identifier (e.g. "did:mycelix:uCAES...")
    pub did: String,
    /// The 8-dimensional profile at issuance time.
    pub profile: SovereignProfile,
    /// Derived tier at issuance.
    pub tier: CivicTier,
    /// Microseconds since epoch when issued.
    pub issued_at: u64,
    /// Microseconds since epoch when this credential expires.
    pub expires_at: u64,
    /// DID of the issuing bridge.
    pub issuer: String,
    /// Extensible key-value pairs for future dimensions or metadata.
    pub extensions: Vec<(String, Vec<u8>)>,
}
