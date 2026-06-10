// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Trust Perception
//!
//! Converts MATL trust scores from the Mycelix network into
//! hyperdimensional semantic vectors for the Symthaea brain.
//!
//! "Feeling the trust of the community."

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// A raw signal from the social organism (Mycelix)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialTrustSignal {
    pub agent_did: String,
    pub trust_score: f64, // 0.0 to 1.0
    pub is_anomalous: bool,
}

/// The percept of trust, encoded in the brain's language
#[derive(Debug, Clone)]
pub struct TrustPercept {
    pub raw: SocialTrustSignal,
    pub vector: ContinuousHV,
}

/// Hash a string to a u64 seed for deterministic vector generation
fn hash_to_seed(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

impl TrustPercept {
    /// Encode a trust signal into a hypervector
    pub fn encode(signal: SocialTrustSignal) -> Self {
        // 1. Generate base vector for the Agent ID (Identity)
        // Uses hash-based seed for deterministic mapping
        let identity_seed = hash_to_seed(&signal.agent_did);
        let identity_vec = ContinuousHV::random(HDC_DIMENSION, identity_seed);

        // 2. Generate concept vectors for "Trust" vs "Distrust"
        // Fixed seeds for consistent concept vectors across all uses
        const TRUST_SEED: u64 = 0x5452555354; // "TRUST" in hex
        const DISTRUST_SEED: u64 = 0x4449535452; // "DISTR" in hex
        let trust_concept = ContinuousHV::random(HDC_DIMENSION, TRUST_SEED);
        let distrust_concept = ContinuousHV::random(HDC_DIMENSION, DISTRUST_SEED);

        // 3. Modulate based on score (0.0-1.0)
        // High score (1.0) -> closer to Trust Concept
        // Low score (0.0) -> closer to Distrust Concept
        // We use binding (element-wise multiplication) to create associations:
        // If score > 0.8: Bind Identity ⊗ Trust
        // If score < 0.3: Bind Identity ⊗ Distrust
        // Else: Identity (neutral)

        let sentiment_vec = if signal.trust_score > 0.8 {
            identity_vec.bind(&trust_concept)
        } else if signal.trust_score < 0.3 {
            identity_vec.bind(&distrust_concept)
        } else {
            identity_vec.clone()
        };

        Self {
            raw: signal,
            vector: sentiment_vec,
        }
    }
}
