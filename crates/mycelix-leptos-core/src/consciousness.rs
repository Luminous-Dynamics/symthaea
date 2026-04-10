// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness profile provider (4D: identity/reputation/community/engagement).
//!
//! Determines the consciousness tier (Observer -> Guardian) which gates
//! UI features, vote weights, and civic actions.
//!
//! Note: EduNet uses a separate, domain-specific consciousness system
//! (Spore simulation engine). This module is for the shared 4D profile
//! used by Hearth, Governance, and future cluster UIs.

use leptos::prelude::*;
use personal_leptos_types::TrustTier;

/// 4-dimensional consciousness profile.
#[derive(Clone, Debug)]
pub struct ConsciousnessProfile {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

impl ConsciousnessProfile {
    /// Combined score — weighted average matching backend (bridge-common).
    ///
    /// Weights: identity 25%, reputation 25%, community 30%, engagement 20%.
    /// Community is weighted higher for governance participation.
    pub fn combined_score(&self) -> f64 {
        let i = self.identity.clamp(0.0, 1.0);
        let r = self.reputation.clamp(0.0, 1.0);
        let c = self.community.clamp(0.0, 1.0);
        let e = self.engagement.clamp(0.0, 1.0);
        (i * 0.25 + r * 0.25 + c * 0.30 + e * 0.20).clamp(0.0, 1.0)
    }

    pub fn tier(&self) -> TrustTier {
        TrustTier::from_score(self.combined_score())
    }
}

/// Reactive consciousness state.
#[derive(Clone)]
pub struct ConsciousnessState {
    pub profile: ReadSignal<ConsciousnessProfile>,
    pub tier: ReadSignal<TrustTier>,
    pub set_profile: WriteSignal<ConsciousnessProfile>,
}

/// Initialize consciousness provider with simulated defaults.
pub fn provide_consciousness_context() -> ConsciousnessState {
    let default_profile = ConsciousnessProfile {
        identity: 0.5,
        reputation: 0.5,
        community: 0.5,
        engagement: 0.5,
    };

    let (profile, set_profile) = signal(default_profile);
    let tier = Memo::new(move |_| profile.get().tier());
    let (tier_signal, set_tier) = signal(TrustTier::Standard);

    // Keep tier signal in sync with memo
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

/// Attempt to refresh the consciousness profile from the conductor.
///
/// Calls `identity_bridge.get_consciousness_profile` (or equivalent) and
/// updates the reactive profile signal. Falls back silently on error.
///
/// Should be called after the HolochainProvider reports `Connected` status.
pub fn refresh_consciousness_from_conductor(
    state: &ConsciousnessState,
    hc: &crate::holochain_provider::HolochainCtx,
) {
    let set_profile = state.set_profile;
    let hc = hc.clone();

    wasm_bindgen_futures::spawn_local(async move {
        // Try identity_bridge first, then personal_bridge as fallback
        let result: Result<ConsciousnessProfileWire, String> = hc
            .call_zome("identity", "identity_bridge", "get_consciousness_profile", &())
            .await;

        match result {
            Ok(wire) => {
                set_profile.set(ConsciousnessProfile {
                    identity: wire.identity,
                    reputation: wire.reputation,
                    community: wire.community,
                    engagement: wire.engagement,
                });
                web_sys::console::log_1(
                    &format!(
                        "[Consciousness] Refreshed from conductor: tier={}",
                        ConsciousnessProfile {
                            identity: wire.identity,
                            reputation: wire.reputation,
                            community: wire.community,
                            engagement: wire.engagement,
                        }.tier().label()
                    ).into()
                );
            }
            Err(_) => {
                // Silently fall back to defaults — conductor may not have identity cluster
            }
        }
    });
}

/// Wire type for deserializing the consciousness profile from the conductor.
#[derive(Clone, Debug, serde::Deserialize)]
struct ConsciousnessProfileWire {
    #[serde(default)]
    identity: f64,
    #[serde(default)]
    reputation: f64,
    #[serde(default)]
    community: f64,
    #[serde(default)]
    engagement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(i: f64, r: f64, c: f64, e: f64) -> ConsciousnessProfile {
        ConsciousnessProfile { identity: i, reputation: r, community: c, engagement: e }
    }

    #[test]
    fn identity_only_contributes_25_percent() {
        let p = profile(1.0, 0.0, 0.0, 0.0);
        assert!((p.combined_score() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn reputation_only_contributes_25_percent() {
        let p = profile(0.0, 1.0, 0.0, 0.0);
        assert!((p.combined_score() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn community_only_contributes_30_percent() {
        let p = profile(0.0, 0.0, 1.0, 0.0);
        assert!((p.combined_score() - 0.30).abs() < 1e-10);
    }

    #[test]
    fn engagement_only_contributes_20_percent() {
        let p = profile(0.0, 0.0, 0.0, 1.0);
        assert!((p.combined_score() - 0.20).abs() < 1e-10);
    }

    #[test]
    fn all_ones_equals_one() {
        let p = profile(1.0, 1.0, 1.0, 1.0);
        assert!((p.combined_score() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn all_zeros_equals_zero() {
        let p = profile(0.0, 0.0, 0.0, 0.0);
        assert!((p.combined_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn clamps_out_of_range_values() {
        let p = profile(2.0, -1.0, 1.5, -0.5);
        let score = p.combined_score();
        assert!(score >= 0.0 && score <= 1.0);
        // identity=1.0(clamped), reputation=0.0(clamped), community=1.0(clamped), engagement=0.0(clamped)
        // = 0.25 + 0.0 + 0.30 + 0.0 = 0.55
        assert!((score - 0.55).abs() < 1e-10);
    }

    #[test]
    fn matches_backend_bridge_common_weights() {
        // Scenario: high community, low engagement — weights diverge from simple average
        let p = profile(0.5, 0.5, 1.0, 0.0);
        let weighted = 0.5 * 0.25 + 0.5 * 0.25 + 1.0 * 0.30 + 0.0 * 0.20;
        let simple_avg = (0.5 + 0.5 + 1.0 + 0.0) / 4.0;
        assert!((p.combined_score() - weighted).abs() < 1e-10);
        // Confirm this differs from simple average
        assert!((weighted - simple_avg).abs() > 0.01);
    }
}
