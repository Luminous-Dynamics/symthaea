// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Epistemic Decay configuration for a specific ecological feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    pub feature_name: String,
    /// Half-life in hours. After this time, the claim's empirical
    /// confidence drops by 50% or one tier.
    pub half_life_hours: f64,
}

impl DecayConfig {
    /// Civilizational Defaults (Fallback if no LARP Pact exists)
    pub fn default_for(feature: &str) -> Self {
        let half_life = match feature {
            "FloodExtent" => 48.0,       // 2 days - water is fast
            "VegetationHealth" => 720.0, // 30 days - seasonal changes
            "Deforestation" => 8760.0,   // 1 year - trees grow slow
            "SoilMoisture" => 168.0,     // 7 days
            _ => 2160.0,                 // 3 months default
        };

        Self {
            feature_name: feature.to_string(),
            half_life_hours: half_life,
        }
    }
}

/// Calculate the current Epistemic Tier based on age and λ-decay.
pub fn calculate_decayed_tier(
    original_tier: u8,
    observed_at: DateTime<Utc>,
    config: &DecayConfig,
) -> u8 {
    let age = Utc::now() - observed_at;
    let age_hours = age.num_seconds() as f64 / 3600.0;

    // Exponential decay: T(t) = T0 * exp(-lambda * t)
    // lambda = ln(2) / half_life
    let lambda = 0.693147 / config.half_life_hours;
    let decay_factor = (-lambda * age_hours).exp();

    let decayed_value = original_tier as f64 * decay_factor;
    decayed_value.round() as u8
}
