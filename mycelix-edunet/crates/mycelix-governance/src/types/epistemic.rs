// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic tier classification for proposal validation
//!
//! Based on the Layered Epistemic Model (LEM), this module provides truth confidence
//! scoring for governance proposals. Higher tiers require stronger evidence and validation.

use serde::{Deserialize, Serialize};

/// Epistemic tier - represents truth confidence level
///
/// The epistemic tier determines how much evidence and validation a proposal has received.
/// This integrates with the reputation-weighted voting system to ensure high-stakes decisions
/// require stronger evidence.
///
/// ## Tier Progression
///
/// Proposals typically start at E1 and can be elevated through community validation:
///
/// ```text
/// E0 → E1 → E2 → E3 → E4
/// ↑    ↑    ↑    ↑    ↑
/// No   Self  Peer  Community  Public
/// evidence    verified  consensus  reproducible
/// ```
///
/// ## Integration with Voting
///
/// - **E0-E1**: Requires higher quorum (40%+) due to low confidence
/// - **E2**: Standard quorum (33%)
/// - **E3**: Reduced quorum (25%) - community has validated
/// - **E4**: Minimum quorum (20%) - highest confidence
///
/// ## Example
///
/// ```rust
/// use mycelix_governance::types::EpistemicTier;
///
/// let proposal_tier = EpistemicTier::E2PeerVerified;
/// assert_eq!(proposal_tier.min_quorum(), 0.33);
/// assert_eq!(proposal_tier.confidence_multiplier(), 1.5);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Copy)]
pub enum EpistemicTier {
    /// E0: Null - No evidence or unverified claim
    ///
    /// Used for:
    /// - Placeholder proposals
    /// - Retracted claims
    /// - Proposals lacking any supporting evidence
    ///
    /// Voting Impact:
    /// - Requires 40%+ quorum
    /// - No confidence multiplier (1.0x)
    E0Null,

    /// E1: Personal Testimony - Single agent's claim
    ///
    /// Used for:
    /// - Initial proposal submission
    /// - Personal observations
    /// - Anecdotal evidence
    ///
    /// Voting Impact:
    /// - Requires 40% quorum
    /// - Minimal confidence multiplier (1.1x)
    ///
    /// Example: "I noticed the FL model isn't converging well"
    E1PersonalTestimony,

    /// E2: Peer Verified - Multiple agents confirm independently
    ///
    /// Used for:
    /// - Proposals verified by 2+ domain experts
    /// - Independently reproducible observations
    /// - Technical claims with evidence
    ///
    /// Voting Impact:
    /// - Standard 33% quorum
    /// - Moderate confidence multiplier (1.5x)
    ///
    /// Example: "Three ML engineers confirmed the convergence issue"
    E2PeerVerified,

    /// E3: Community Consensus - Broad agreement across community
    ///
    /// Used for:
    /// - Proposals with 60%+ pre-vote support
    /// - Well-documented technical improvements
    /// - Widely accepted facts
    ///
    /// Voting Impact:
    /// - Reduced 25% quorum
    /// - High confidence multiplier (2.0x)
    ///
    /// Example: "80% of educators agree current curriculum needs updating"
    E3CommunityCensensus,

    /// E4: Publicly Reproducible - Objectively verifiable truth
    ///
    /// Used for:
    /// - Mathematical proofs
    /// - Reproducible experimental results
    /// - Open-source code improvements with benchmarks
    ///
    /// Voting Impact:
    /// - Minimum 20% quorum
    /// - Maximum confidence multiplier (3.0x)
    ///
    /// Example: "New algorithm shows 40% speed improvement in public benchmarks"
    E4PubliclyReproducible,
}

impl EpistemicTier {
    /// Get minimum quorum required for this epistemic tier
    ///
    /// Returns a value between 0.0 and 1.0 representing the fraction of
    /// eligible voters that must participate for the vote to be valid.
    pub fn min_quorum(&self) -> f64 {
        match self {
            EpistemicTier::E0Null => 0.40,
            EpistemicTier::E1PersonalTestimony => 0.40,
            EpistemicTier::E2PeerVerified => 0.33,
            EpistemicTier::E3CommunityCensensus => 0.25,
            EpistemicTier::E4PubliclyReproducible => 0.20,
        }
    }

    /// Get confidence multiplier for reputation weighting
    ///
    /// Higher tiers amplify the effective reputation of voters who support
    /// well-evidenced proposals, incentivizing thorough research.
    pub fn confidence_multiplier(&self) -> f64 {
        match self {
            EpistemicTier::E0Null => 1.0,
            EpistemicTier::E1PersonalTestimony => 1.1,
            EpistemicTier::E2PeerVerified => 1.5,
            EpistemicTier::E3CommunityCensensus => 2.0,
            EpistemicTier::E4PubliclyReproducible => 3.0,
        }
    }

    /// Get human-readable description of this tier
    pub fn description(&self) -> &'static str {
        match self {
            EpistemicTier::E0Null => "No evidence or verification",
            EpistemicTier::E1PersonalTestimony => "Single agent's claim",
            EpistemicTier::E2PeerVerified => "Independently verified by peers",
            EpistemicTier::E3CommunityCensensus => "Broad community agreement",
            EpistemicTier::E4PubliclyReproducible => "Objectively verifiable truth",
        }
    }

    /// Check if this tier requires peer review before voting
    pub fn requires_peer_review(&self) -> bool {
        matches!(
            self,
            EpistemicTier::E2PeerVerified
                | EpistemicTier::E3CommunityCensensus
                | EpistemicTier::E4PubliclyReproducible
        )
    }

    /// Get recommended voting period in hours for this tier
    pub fn recommended_voting_period_hours(&self) -> i64 {
        match self {
            EpistemicTier::E0Null => 168,              // 7 days - needs discussion
            EpistemicTier::E1PersonalTestimony => 168, // 7 days - needs validation
            EpistemicTier::E2PeerVerified => 120,      // 5 days - standard
            EpistemicTier::E3CommunityCensensus => 72, // 3 days - well-supported
            EpistemicTier::E4PubliclyReproducible => 48, // 2 days - clear evidence
        }
    }
}

impl Default for EpistemicTier {
    /// Default to E1 (Personal Testimony) for new proposals
    fn default() -> Self {
        EpistemicTier::E1PersonalTestimony
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_ordering() {
        assert!(EpistemicTier::E0Null < EpistemicTier::E1PersonalTestimony);
        assert!(EpistemicTier::E1PersonalTestimony < EpistemicTier::E2PeerVerified);
        assert!(EpistemicTier::E2PeerVerified < EpistemicTier::E3CommunityCensensus);
        assert!(EpistemicTier::E3CommunityCensensus < EpistemicTier::E4PubliclyReproducible);
    }

    #[test]
    fn test_quorum_requirements() {
        assert_eq!(EpistemicTier::E0Null.min_quorum(), 0.40);
        assert_eq!(EpistemicTier::E1PersonalTestimony.min_quorum(), 0.40);
        assert_eq!(EpistemicTier::E2PeerVerified.min_quorum(), 0.33);
        assert_eq!(EpistemicTier::E3CommunityCensensus.min_quorum(), 0.25);
        assert_eq!(EpistemicTier::E4PubliclyReproducible.min_quorum(), 0.20);

        // Higher tiers require lower quorum
        assert!(EpistemicTier::E4PubliclyReproducible.min_quorum() < EpistemicTier::E0Null.min_quorum());
    }

    #[test]
    fn test_confidence_multipliers() {
        assert_eq!(EpistemicTier::E0Null.confidence_multiplier(), 1.0);
        assert_eq!(EpistemicTier::E1PersonalTestimony.confidence_multiplier(), 1.1);
        assert_eq!(EpistemicTier::E2PeerVerified.confidence_multiplier(), 1.5);
        assert_eq!(EpistemicTier::E3CommunityCensensus.confidence_multiplier(), 2.0);
        assert_eq!(EpistemicTier::E4PubliclyReproducible.confidence_multiplier(), 3.0);

        // Higher tiers have higher confidence
        assert!(EpistemicTier::E4PubliclyReproducible.confidence_multiplier() > EpistemicTier::E0Null.confidence_multiplier());
    }

    #[test]
    fn test_peer_review_requirements() {
        assert!(!EpistemicTier::E0Null.requires_peer_review());
        assert!(!EpistemicTier::E1PersonalTestimony.requires_peer_review());
        assert!(EpistemicTier::E2PeerVerified.requires_peer_review());
        assert!(EpistemicTier::E3CommunityCensensus.requires_peer_review());
        assert!(EpistemicTier::E4PubliclyReproducible.requires_peer_review());
    }

    #[test]
    fn test_voting_period_recommendations() {
        // Lower tiers need more time for discussion
        assert!(EpistemicTier::E0Null.recommended_voting_period_hours() > EpistemicTier::E4PubliclyReproducible.recommended_voting_period_hours());

        assert_eq!(EpistemicTier::E4PubliclyReproducible.recommended_voting_period_hours(), 48);
        assert_eq!(EpistemicTier::E3CommunityCensensus.recommended_voting_period_hours(), 72);
    }

    #[test]
    fn test_default_tier() {
        let default_tier = EpistemicTier::default();
        assert_eq!(default_tier, EpistemicTier::E1PersonalTestimony);
    }

    #[test]
    fn test_descriptions() {
        assert!(!EpistemicTier::E0Null.description().is_empty());
        assert!(!EpistemicTier::E4PubliclyReproducible.description().is_empty());
    }
}
