// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 8D Sovereign Profile provider for Leptos frontends.
//!
//! Provides the reactive `SovereignProfileState` that drives TierGate,
//! vote weights, and civic action gating across all Mycelix cluster UIs.
//!
//! ## The 8 Axes
//!
//! 1. Epistemic Integrity — truth/claim verification
//! 2. Thermodynamic Yield — energy contribution
//! 3. Network Resilience — node uptime
//! 4. Economic Velocity — anti-hoarding compliance
//! 5. Civic Participation — governance engagement
//! 6. Stewardship & Care — commons maintenance
//! 7. Semantic Resonance — community alignment
//! 8. Domain Competence — peer-verified expertise
//!
//! ## Backward Compatibility
//!
//! `ConsciousnessProfile` is a type alias for backward-compat imports.
//! The old 4D `combined_score()` is available via `legacy_combined_score()`.

use leptos::prelude::*;
use personal_leptos_types::TrustTier;
pub use sovereign_profile::{
    i18n::{dimension_label, DIMENSION_LABELS},
    weights::DimensionWeights,
    CivicTier, SovereignDimension, SovereignProfile,
};

/// Backward-compatible alias — code that imported ConsciousnessProfile
/// will continue to compile, but new code should use SovereignProfile.
pub type ConsciousnessProfile = SovereignProfile;

/// Combined score using default governance weights.
///
/// Convenience wrapper for frontends that don't need custom weights.
pub fn combined_score(profile: &SovereignProfile) -> f64 {
    profile.combined_score(&DimensionWeights::governance())
}

/// Derive the civic tier from a profile using governance weights.
pub fn civic_tier(profile: &SovereignProfile) -> CivicTier {
    profile.tier(&DimensionWeights::governance())
}

/// Map CivicTier to the legacy TrustTier enum used by personal-leptos-types.
pub fn trust_tier_from_civic(tier: CivicTier) -> TrustTier {
    match tier {
        CivicTier::Observer => TrustTier::Observer,
        CivicTier::Participant => TrustTier::Basic,
        CivicTier::Citizen => TrustTier::Standard,
        CivicTier::Steward => TrustTier::Elevated,
        CivicTier::Guardian => TrustTier::Guardian,
    }
}

// ---------------------------------------------------------------------------
// Reactive state
// ---------------------------------------------------------------------------

/// Reactive sovereign profile state for Leptos context.
#[derive(Clone)]
pub struct ConsciousnessState {
    pub profile: ReadSignal<SovereignProfile>,
    pub tier: ReadSignal<TrustTier>,
    pub set_profile: WriteSignal<SovereignProfile>,
}

/// Initialize the sovereign profile provider with default values.
///
/// Provides `ConsciousnessState` via Leptos context. All 8 dimensions
/// start at 0.5 (mid-range) until refreshed from the conductor.
pub fn provide_consciousness_context() -> ConsciousnessState {
    let default_profile = SovereignProfile {
        epistemic_integrity: 0.5,
        thermodynamic_yield: 0.5,
        network_resilience: 0.5,
        economic_velocity: 0.5,
        civic_participation: 0.5,
        stewardship_care: 0.5,
        semantic_resonance: 0.5,
        domain_competence: 0.5,
    };

    let (profile, set_profile) = signal(default_profile);
    let tier = Memo::new(move |_| trust_tier_from_civic(civic_tier(&profile.get())));
    let (tier_signal, set_tier) = signal(TrustTier::Standard);

    Effect::new(move |_| {
        set_tier.set(tier.get());
    });

    let state = ConsciousnessState {
        profile,
        tier: tier_signal,
        set_profile,
    };

    provide_context(state.clone());
    state
}

pub fn use_consciousness() -> ConsciousnessState {
    expect_context::<ConsciousnessState>()
}

/// Attempt to refresh the sovereign profile from the conductor.
///
/// Tries the new 8D `get_sovereign_credential` endpoint first. If that
/// fails (identity bridge not yet upgraded), falls back to the legacy 4D
/// `get_consciousness_profile` and maps to 8D via From conversion.
pub fn refresh_consciousness_from_conductor(
    state: &ConsciousnessState,
    hc: &crate::holochain_provider::HolochainCtx,
) {
    let set_profile = state.set_profile;
    let hc = hc.clone();

    wasm_bindgen_futures::spawn_local(async move {
        // Try 8D sovereign credential first
        let result_8d: Result<ProfileWire, String> = hc
            .call_zome(
                "identity",
                "identity_bridge",
                "get_sovereign_credential",
                &(),
            )
            .await;

        match result_8d {
            Ok(wire) => {
                let profile = wire.to_sovereign();
                let tier = civic_tier(&profile);
                set_profile.set(profile);
                web_sys::console::log_1(
                    &format!(
                        "[Sovereign] 8D profile from conductor: tier={}",
                        tier.label()
                    )
                    .into(),
                );
                return;
            }
            Err(_) => {
                // 8D endpoint not available — try legacy 4D
            }
        }

        // Fallback: legacy 4D consciousness profile → 8D mapping
        let result_4d: Result<ProfileWire, String> = hc
            .call_zome(
                "identity",
                "identity_bridge",
                "get_consciousness_credential",
                &(),
            )
            .await;

        match result_4d {
            Ok(wire) => {
                let profile = wire.to_sovereign();
                let tier = civic_tier(&profile);
                set_profile.set(profile);
                web_sys::console::log_1(
                    &format!(
                        "[Sovereign] 4D fallback from conductor: tier={}",
                        tier.label()
                    )
                    .into(),
                );
            }
            Err(_) => {
                // No conductor — keep defaults
            }
        }
    });
}

/// Wire type for deserializing from the conductor.
///
/// Accepts both 4D legacy and 8D sovereign formats via serde defaults.
#[derive(Clone, Debug, serde::Deserialize)]
struct ProfileWire {
    // Legacy 4D fields (used during transition)
    #[serde(default)]
    identity: f64,
    #[serde(default)]
    reputation: f64,
    #[serde(default)]
    community: f64,
    #[serde(default)]
    engagement: f64,
    // New 8D fields (take precedence when present)
    #[serde(default)]
    epistemic_integrity: Option<f64>,
    #[serde(default)]
    thermodynamic_yield: Option<f64>,
    #[serde(default)]
    network_resilience: Option<f64>,
    #[serde(default)]
    economic_velocity: Option<f64>,
    #[serde(default)]
    civic_participation: Option<f64>,
    #[serde(default)]
    stewardship_care: Option<f64>,
    #[serde(default)]
    semantic_resonance: Option<f64>,
    #[serde(default)]
    domain_competence: Option<f64>,
}

impl ProfileWire {
    /// Convert to SovereignProfile, preferring 8D fields when available.
    fn to_sovereign(&self) -> SovereignProfile {
        SovereignProfile {
            epistemic_integrity: self.epistemic_integrity.unwrap_or(self.identity),
            thermodynamic_yield: self.thermodynamic_yield.unwrap_or(self.engagement),
            network_resilience: self.network_resilience.unwrap_or(self.identity),
            economic_velocity: self.economic_velocity.unwrap_or(self.reputation),
            civic_participation: self.civic_participation.unwrap_or(self.community),
            stewardship_care: self.stewardship_care.unwrap_or(self.reputation),
            semantic_resonance: self.semantic_resonance.unwrap_or(self.community),
            domain_competence: self.domain_competence.unwrap_or(self.engagement),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_8d(v: f64) -> SovereignProfile {
        SovereignProfile::from_array([v; 8])
    }

    #[test]
    fn zero_profile_is_observer() {
        assert_eq!(civic_tier(&SovereignProfile::zero()), CivicTier::Observer);
    }

    #[test]
    fn full_profile_is_guardian() {
        assert_eq!(civic_tier(&profile_8d(1.0)), CivicTier::Guardian);
    }

    #[test]
    fn mid_profile_combined_score() {
        let score = combined_score(&profile_8d(0.5));
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn civic_tier_maps_to_trust_tier() {
        assert_eq!(
            trust_tier_from_civic(CivicTier::Observer),
            TrustTier::Observer
        );
        assert_eq!(
            trust_tier_from_civic(CivicTier::Participant),
            TrustTier::Basic
        );
        assert_eq!(
            trust_tier_from_civic(CivicTier::Citizen),
            TrustTier::Standard
        );
        assert_eq!(
            trust_tier_from_civic(CivicTier::Steward),
            TrustTier::Elevated
        );
        assert_eq!(
            trust_tier_from_civic(CivicTier::Guardian),
            TrustTier::Guardian
        );
    }

    #[test]
    fn governance_weights_are_normalized() {
        assert!(DimensionWeights::governance().is_normalized());
    }

    #[test]
    fn single_dimension_contributions() {
        let weights = DimensionWeights::governance();
        for dim in SovereignDimension::ALL {
            let mut p = SovereignProfile::zero();
            p.set(dim, 1.0);
            let score = p.combined_score(&weights);
            assert!(
                score > 0.0 && score <= 1.0,
                "Dimension {:?} should contribute positively, got {score}",
                dim
            );
        }
    }

    #[test]
    fn wire_legacy_4d_converts_to_8d() {
        let wire = ProfileWire {
            identity: 0.7,
            reputation: 0.5,
            community: 0.9,
            engagement: 0.3,
            epistemic_integrity: None,
            thermodynamic_yield: None,
            network_resilience: None,
            economic_velocity: None,
            civic_participation: None,
            stewardship_care: None,
            semantic_resonance: None,
            domain_competence: None,
        };
        let p = wire.to_sovereign();
        assert!((p.epistemic_integrity - 0.7).abs() < 1e-10); // from identity
        assert!((p.civic_participation - 0.9).abs() < 1e-10); // from community
        assert!((p.thermodynamic_yield - 0.3).abs() < 1e-10); // from engagement
    }

    #[test]
    fn wire_8d_fields_take_precedence() {
        let wire = ProfileWire {
            identity: 0.7,
            reputation: 0.5,
            community: 0.9,
            engagement: 0.3,
            epistemic_integrity: Some(0.99),
            thermodynamic_yield: None,
            network_resilience: None,
            economic_velocity: None,
            civic_participation: Some(0.11),
            stewardship_care: None,
            semantic_resonance: None,
            domain_competence: None,
        };
        let p = wire.to_sovereign();
        assert!((p.epistemic_integrity - 0.99).abs() < 1e-10); // 8D takes precedence
        assert!((p.civic_participation - 0.11).abs() < 1e-10); // 8D takes precedence
        assert!((p.thermodynamic_yield - 0.3).abs() < 1e-10); // falls back to legacy
    }
}
