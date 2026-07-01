// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness tier definitions.
//!
//! Well-known consciousness gate thresholds used across Mycelix clusters
//! for civic-tier gating (voting, proposals, constitutional authority).
//!
//! Note: the full `mycelix-zkp-core` crate also provides real Winterfell
//! STARK range proofs over these tiers (`ConsciousnessProofRequest` /
//! `ConsciousnessProofResult`); that machinery depends on the
//! `backend-winterfell` feature and private proof-system crates not
//! vendored here. This standalone subset provides the `CivicTier` enum
//! itself, which is all symthaea's own code needs.

use serde::{Deserialize, Serialize};

/// Well-known consciousness gate thresholds (from bridge-common).
pub mod thresholds {
    /// Basic participation (Finance pledge, Attribution)
    pub const BASIC: f64 = 0.2;
    /// Proposal submission (Governance propose, Finance TEND matching)
    pub const PROPOSAL: f64 = 0.3;
    /// Voting rights (Governance vote)
    pub const VOTING: f64 = 0.4;
    /// Constitutional authority (Governance constitutional amendment)
    pub const CONSTITUTIONAL: f64 = 0.6;
}

/// Consciousness tier names for human-readable output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CivicTier {
    Observer,    // < 0.2
    Participant, // >= 0.2
    Citizen,     // >= 0.3
    Steward,     // >= 0.4
    Guardian,    // >= 0.6
}

impl CivicTier {
    /// Minimum Phi threshold for this tier.
    pub fn min_phi(&self) -> f64 {
        match self {
            CivicTier::Observer => 0.0,
            CivicTier::Participant => thresholds::BASIC,
            CivicTier::Citizen => thresholds::PROPOSAL,
            CivicTier::Steward => thresholds::VOTING,
            CivicTier::Guardian => thresholds::CONSTITUTIONAL,
        }
    }

    /// Convert Phi threshold to Q16.16 fixed-point scaled to [0, 10000].
    pub fn threshold_scaled(&self) -> u64 {
        (self.min_phi() * 10000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_phi_ordering() {
        assert!(CivicTier::Observer.min_phi() < CivicTier::Participant.min_phi());
        assert!(CivicTier::Participant.min_phi() < CivicTier::Citizen.min_phi());
        assert!(CivicTier::Citizen.min_phi() < CivicTier::Steward.min_phi());
        assert!(CivicTier::Steward.min_phi() < CivicTier::Guardian.min_phi());
    }
}
