// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness profile provider (4D: identity/reputation/community/engagement).
//!
//! Determines the consciousness tier (Observer → Guardian) which gates
//! UI features and modulates vote weights.

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
    /// Combined score (simple average for now).
    pub fn combined_score(&self) -> f64 {
        (self.identity + self.reputation + self.community + self.engagement) / 4.0
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
/// Will be wired to real Spore engine in Phase 6.
pub fn provide_consciousness_context() -> ConsciousnessState {
    let default_profile = ConsciousnessProfile {
        identity: 0.5,
        reputation: 0.5,
        community: 0.5,
        engagement: 0.5,
    };

    let (profile, set_profile) = signal(default_profile);
    let tier = Memo::new(move |_| profile.get().tier());
    let (tier_signal, _set_tier) = signal(TrustTier::Standard);

    // Keep tier signal in sync with memo
    Effect::new(move |_| {
        _set_tier.set(tier.get());
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
