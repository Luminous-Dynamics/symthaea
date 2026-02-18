//! Canonical Phi threshold configuration — single source of truth.
//!
//! All Mycelix components import these values instead of hardcoding their own.
//! FL core, governance, and personal cluster all reference this module for
//! consistent Phi-based gating decisions.
//!
//! ## Three Distinct "Phi" Concepts
//!
//! 1. **IIT Phi (Symthaea)**: True integrated information from PhiEngine
//! 2. **Gradient Coherence (FL)**: Proxy score from gradient statistics
//! 3. **Governance Phi**: Agent's attested consciousness level for voting
//!
//! The thresholds here apply to concepts 1 and 3. Concept 2 uses its own
//! coherence-specific thresholds but imports the FL veto/dampen/boost values.

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Canonical Phi threshold configuration.
///
/// All Mycelix components should import these values rather than
/// hardcoding their own constants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhiThresholds {
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
    pub gov_basic: f64,
    /// Proposal submission threshold (default 0.3)
    pub gov_proposal: f64,
    /// Voting threshold (default 0.4)
    pub gov_voting: f64,
    /// Constitutional change threshold (default 0.6)
    pub gov_constitutional: f64,
}

impl Default for PhiThresholds {
    fn default() -> Self {
        Self {
            fl_veto: 0.1,
            fl_dampen: 0.3,
            fl_boost: 0.6,
            fl_dampen_factor: 0.3,
            fl_boost_factor: 1.5,
            gov_basic: 0.2,
            gov_proposal: 0.3,
            gov_voting: 0.4,
            gov_constitutional: 0.6,
        }
    }
}

/// Global canonical thresholds, lazily initialized.
static PHI_THRESHOLDS: LazyLock<PhiThresholds> = LazyLock::new(PhiThresholds::default);

/// Get the canonical Phi thresholds.
///
/// Returns a static reference to the default thresholds. All Mycelix
/// components should use this function rather than defining their own constants.
pub fn phi_thresholds() -> &'static PhiThresholds {
    &PHI_THRESHOLDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thresholds_are_consistent() {
        let t = PhiThresholds::default();
        // FL thresholds are ordered: veto < dampen < boost
        assert!(t.fl_veto < t.fl_dampen);
        assert!(t.fl_dampen < t.fl_boost);
        // Governance thresholds are ordered: basic < proposal < voting < constitutional
        assert!(t.gov_basic < t.gov_proposal);
        assert!(t.gov_proposal < t.gov_voting);
        assert!(t.gov_voting < t.gov_constitutional);
    }

    #[test]
    fn phi_thresholds_returns_defaults() {
        let t = phi_thresholds();
        assert_eq!(t.fl_veto, 0.1);
        assert_eq!(t.fl_dampen, 0.3);
        assert_eq!(t.fl_boost, 0.6);
        assert_eq!(t.fl_dampen_factor, 0.3);
        assert_eq!(t.fl_boost_factor, 1.5);
        assert_eq!(t.gov_basic, 0.2);
        assert_eq!(t.gov_proposal, 0.3);
        assert_eq!(t.gov_voting, 0.4);
        assert_eq!(t.gov_constitutional, 0.6);
    }

    #[test]
    fn serde_roundtrip() {
        let t = PhiThresholds::default();
        let json = serde_json::to_string(&t).unwrap();
        let t2: PhiThresholds = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
