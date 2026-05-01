// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Backward compatibility with the legacy 4D ConsciousnessProfile.
//!
//! The old 4D profile (identity, reputation, community, engagement) maps
//! to the new 8D profile by splitting each old dimension across two new ones.
//! The reverse projection (8D → 4D) is lossy but preserves tier derivation
//! for any profile that was created via the forward conversion.
//!
//! ## Mapping (4D → 8D)
//!
//! | Old Dimension | → New Dimensions |
//! |---------------|-----------------|
//! | identity (0.25) | epistemic_integrity (0.5×), network_resilience (0.5×) |
//! | reputation (0.25) | economic_velocity (0.5×), stewardship_care (0.5×) |
//! | community (0.30) | civic_participation (0.5×), semantic_resonance (0.5×) |
//! | engagement (0.20) | thermodynamic_yield (0.5×), domain_competence (0.5×) |
//!
//! ## Mapping (8D → 4D)
//!
//! | New Dimensions | → Old Dimension |
//! |---------------|----------------|
//! | epistemic_integrity + network_resilience | identity = avg |
//! | economic_velocity + stewardship_care | reputation = avg |
//! | civic_participation + semantic_resonance | community = avg |
//! | thermodynamic_yield + domain_competence | engagement = avg |

use crate::{CivicTier, SovereignProfile};

/// Legacy 4D consciousness profile (for backward compatibility).
///
/// This is a minimal replica of `mycelix_bridge_common::ConsciousnessProfile`.
/// The canonical version lives in bridge-common; this exists so that
/// `sovereign-profile` can provide `From` conversions without depending on HDK.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LegacyProfile {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

impl LegacyProfile {
    /// Combined score using the legacy 4D weights (25/25/30/20).
    pub fn combined_score(&self) -> f64 {
        let i = sanitize(self.identity);
        let r = sanitize(self.reputation);
        let c = sanitize(self.community);
        let e = sanitize(self.engagement);
        (i * 0.25 + r * 0.25 + c * 0.30 + e * 0.20).clamp(0.0, 1.0)
    }
}

/// Legacy 5-tier consciousness enum (for backward compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LegacyTier {
    Observer,
    Participant,
    Citizen,
    Steward,
    Guardian,
}

// ---------------------------------------------------------------------------
// Conversions: Legacy → Sovereign
// ---------------------------------------------------------------------------

impl From<LegacyProfile> for SovereignProfile {
    fn from(old: LegacyProfile) -> Self {
        let i = sanitize(old.identity);
        let r = sanitize(old.reputation);
        let c = sanitize(old.community);
        let e = sanitize(old.engagement);

        Self {
            epistemic_integrity: i,
            thermodynamic_yield: e,
            network_resilience: i,
            economic_velocity: r,
            civic_participation: c,
            stewardship_care: r,
            semantic_resonance: c,
            domain_competence: e,
        }
    }
}

impl From<LegacyTier> for CivicTier {
    fn from(old: LegacyTier) -> Self {
        match old {
            LegacyTier::Observer => CivicTier::Observer,
            LegacyTier::Participant => CivicTier::Participant,
            LegacyTier::Citizen => CivicTier::Citizen,
            LegacyTier::Steward => CivicTier::Steward,
            LegacyTier::Guardian => CivicTier::Guardian,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions: Sovereign → Legacy (lossy)
// ---------------------------------------------------------------------------

impl From<SovereignProfile> for LegacyProfile {
    fn from(new: SovereignProfile) -> Self {
        Self {
            identity: (sanitize(new.epistemic_integrity) + sanitize(new.network_resilience)) / 2.0,
            reputation: (sanitize(new.economic_velocity) + sanitize(new.stewardship_care)) / 2.0,
            community: (sanitize(new.civic_participation) + sanitize(new.semantic_resonance)) / 2.0,
            engagement: (sanitize(new.thermodynamic_yield) + sanitize(new.domain_competence)) / 2.0,
        }
    }
}

impl From<CivicTier> for LegacyTier {
    fn from(new: CivicTier) -> Self {
        match new {
            CivicTier::Observer => LegacyTier::Observer,
            CivicTier::Participant => LegacyTier::Participant,
            CivicTier::Citizen => LegacyTier::Citizen,
            CivicTier::Steward => LegacyTier::Steward,
            CivicTier::Guardian => LegacyTier::Guardian,
        }
    }
}

fn sanitize(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
