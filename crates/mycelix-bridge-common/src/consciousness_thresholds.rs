// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical consciousness threshold configuration — single source of truth.
//!
//! All Mycelix components import these values instead of hardcoding their own.
//! FL core, governance, and personal cluster all reference this module for
//! consistent consciousness-based gating decisions.
//!
//! ## Three Distinct Consciousness Metrics
//!
//! 1. **Integration (Symthaea)**: SpectralConnectivity / Fiedler value from PhiEngine
//!    (NOT true IIT Phi — SpectralConnectivity λ₂ has r≈-0.14 vs Exact tier)
//! 2. **Gradient Coherence (FL)**: Proxy score from gradient statistics
//! 3. **Consciousness Level (Governance)**: Agent's attested consciousness level for voting
//!
//! The thresholds here apply to concepts 1 and 3. Concept 2 uses its own
//! coherence-specific thresholds but imports the FL veto/dampen/boost values.

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Canonical consciousness threshold configuration.
///
/// All Mycelix components should import these values rather than
/// hardcoding their own constants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsciousnessThresholds {
    // FL Byzantine weight adjustment
    /// Below this: exclude gradient entirely (default 0.1)
    pub fl_veto: f32,
    /// Below this: reduce weight (default 0.3)
    pub fl_dampen: f32,
    /// Above this: increase weight (default 0.6)
    pub fl_boost: f32,
    /// Multiplier when dampened (default 0.3)
    pub fl_dampen_factor: f32,
    /// Multiplier when boosted (default 1.5)
    pub fl_boost_factor: f32,

    // Governance action gates
    /// Basic participation threshold (default 0.2)
    pub consciousness_gate_basic: f64,
    /// Proposal submission threshold (default 0.3)
    pub consciousness_gate_proposal: f64,
    /// Voting threshold (default 0.4)
    pub consciousness_gate_voting: f64,
    /// Constitutional change threshold (default 0.6)
    pub consciousness_gate_constitutional: f64,

    // Allocation gating (Conscious Allocation Network)
    /// Pledge submission threshold (default 0.2, Participant tier)
    pub consciousness_gate_pledge: f64,
    /// Running matching threshold (default 0.3, Citizen tier)
    pub consciousness_gate_matching: f64,
    /// Impact reporting threshold (default 0.2, Participant tier)
    pub consciousness_gate_impact: f64,

    // Bootstrap gating (cold-start communities)
    /// Maximum community size for bootstrap eligibility (default 5)
    pub bootstrap_community_threshold: u32,
    /// TTL for bootstrap credentials in microseconds (default 1 hour = 3_600_000_000)
    pub bootstrap_ttl_us: u64,
    /// Minimum identity score for bootstrap eligibility (default 0.25 = Basic MFA)
    pub bootstrap_min_identity: f64,
}

impl Default for ConsciousnessThresholds {
    fn default() -> Self {
        Self {
            fl_veto: 0.1,
            fl_dampen: 0.3,
            fl_boost: 0.6,
            fl_dampen_factor: 0.3,
            fl_boost_factor: 1.5,
            consciousness_gate_basic: 0.2,
            consciousness_gate_proposal: 0.3,
            consciousness_gate_voting: 0.4,
            consciousness_gate_constitutional: 0.6,
            consciousness_gate_pledge: 0.2,
            consciousness_gate_matching: 0.3,
            consciousness_gate_impact: 0.2,
            bootstrap_community_threshold: BOOTSTRAP_COMMUNITY_THRESHOLD,
            bootstrap_ttl_us: BOOTSTRAP_TTL_US,
            bootstrap_min_identity: BOOTSTRAP_MIN_IDENTITY,
        }
    }
}

/// Maximum community member count for bootstrap eligibility.
pub const BOOTSTRAP_COMMUNITY_THRESHOLD: u32 = 5;

/// TTL for bootstrap credentials: 15 minutes in microseconds.
///
/// Reduced from 1 hour to minimize the TOCTOU window where a bootstrap
/// credential remains valid after the community grows past the bootstrap
/// threshold (5 members). A 15-minute window is sufficient for initial
/// setup while limiting exposure.
pub const BOOTSTRAP_TTL_US: u64 = 900_000_000;

/// Minimum identity score required for bootstrap eligibility (Basic MFA = 0.25).
pub const BOOTSTRAP_MIN_IDENTITY: f64 = 0.25;

/// Backward-compatible type alias.
pub type PhiThresholds = ConsciousnessThresholds;

/// Global canonical thresholds, lazily initialized.
static CONSCIOUSNESS_THRESHOLDS: LazyLock<ConsciousnessThresholds> =
    LazyLock::new(ConsciousnessThresholds::default);

/// Get the canonical consciousness thresholds.
///
/// Returns a static reference to the default thresholds. All Mycelix
/// components should use this function rather than defining their own constants.
pub fn consciousness_thresholds() -> &'static ConsciousnessThresholds {
    &CONSCIOUSNESS_THRESHOLDS
}

/// Backward-compatible function alias.
pub fn phi_thresholds() -> &'static ConsciousnessThresholds {
    consciousness_thresholds()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thresholds_are_consistent() {
        let t = ConsciousnessThresholds::default();
        assert!(t.fl_veto < t.fl_dampen);
        assert!(t.fl_dampen < t.fl_boost);
        assert!(t.consciousness_gate_basic < t.consciousness_gate_proposal);
        assert!(t.consciousness_gate_proposal < t.consciousness_gate_voting);
        assert!(t.consciousness_gate_voting < t.consciousness_gate_constitutional);
    }

    #[test]
    fn consciousness_thresholds_returns_defaults() {
        let t = consciousness_thresholds();
        assert_eq!(t.fl_veto, 0.1);
        assert_eq!(t.fl_dampen, 0.3);
        assert_eq!(t.fl_boost, 0.6);
        assert_eq!(t.fl_dampen_factor, 0.3);
        assert_eq!(t.fl_boost_factor, 1.5);
        assert_eq!(t.consciousness_gate_basic, 0.2);
        assert_eq!(t.consciousness_gate_proposal, 0.3);
        assert_eq!(t.consciousness_gate_voting, 0.4);
        assert_eq!(t.consciousness_gate_constitutional, 0.6);
    }

    #[test]
    fn allocation_thresholds_defaults() {
        let t = consciousness_thresholds();
        assert_eq!(t.consciousness_gate_pledge, 0.2);
        assert_eq!(t.consciousness_gate_matching, 0.3);
        assert_eq!(t.consciousness_gate_impact, 0.2);
    }

    #[test]
    fn allocation_thresholds_consistent_with_governance() {
        let t = ConsciousnessThresholds::default();
        // Pledge uses basic tier (Participant)
        assert_eq!(t.consciousness_gate_pledge, t.consciousness_gate_basic);
        // Matching uses proposal tier (Citizen)
        assert_eq!(t.consciousness_gate_matching, t.consciousness_gate_proposal);
        // Impact uses basic tier (Participant)
        assert_eq!(t.consciousness_gate_impact, t.consciousness_gate_basic);
    }

    #[test]
    fn bootstrap_thresholds_defaults() {
        let t = ConsciousnessThresholds::default();
        assert_eq!(t.bootstrap_community_threshold, 5);
        assert_eq!(t.bootstrap_ttl_us, 900_000_000); // 15 minutes
        assert_eq!(t.bootstrap_min_identity, 0.25);
    }

    #[test]
    fn backward_compat_alias_works() {
        let t = phi_thresholds();
        assert_eq!(t.fl_veto, 0.1);
    }

    #[test]
    fn serde_roundtrip() {
        let t = ConsciousnessThresholds::default();
        let json = serde_json::to_string(&t).unwrap();
        let t2: ConsciousnessThresholds = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
