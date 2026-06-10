// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dimension collectors — normalization math for the 8D Sovereign Profile.
//!
//! Each dimension is fed by a source cluster that provides raw data.
//! This module contains the pure normalization functions that convert
//! raw metrics into [0.0, 1.0] scores. The actual cross-cluster calls
//! live in the identity bridge coordinator (HDK-dependent).
//!
//! ## Design Principles
//!
//! 1. **Conservative defaults:** Missing data → 0.0 (never grants unearned access)
//! 2. **Temporal decay:** Recent activity weights more than old activity
//! 3. **Capped normalization:** Scores saturate at 1.0 (no benefit to gaming beyond max)
//! 4. **Cluster independence:** Each collector can fail without affecting others

use crate::SovereignDimension;

// ============================================================================
// Raw data structures (passed from cluster zomes to collectors)
// ============================================================================

/// Raw data for a single dimension, provided by the source cluster.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DimensionInput {
    /// Count of qualifying activities (e.g., validated claims, votes cast, care sessions).
    pub activity_count: u64,
    /// Baseline for saturation (e.g., 50 claims for epistemic, 20 votes for civic).
    /// Score = min(1.0, activity_count / baseline).
    pub baseline: u64,
    /// Quality ratio [0.0, 1.0] (e.g., validation ratio, efficiency, settlement speed).
    pub quality: f64,
    /// Days since last qualifying activity (for recency weighting).
    pub days_since_last: f64,
    /// Recency decay half-life in days (dimension-specific).
    pub recency_half_life_days: f64,
}

/// Collected raw data for all 8 dimensions.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CollectedDimensions {
    pub epistemic_integrity: DimensionInput,
    pub thermodynamic_yield: DimensionInput,
    pub network_resilience: DimensionInput,
    pub economic_velocity: DimensionInput,
    pub civic_participation: DimensionInput,
    pub stewardship_care: DimensionInput,
    pub semantic_resonance: DimensionInput,
    pub domain_competence: DimensionInput,
}

impl CollectedDimensions {
    /// Get the input for a specific dimension.
    pub fn get(&self, dim: SovereignDimension) -> &DimensionInput {
        match dim {
            SovereignDimension::EpistemicIntegrity => &self.epistemic_integrity,
            SovereignDimension::ThermodynamicYield => &self.thermodynamic_yield,
            SovereignDimension::NetworkResilience => &self.network_resilience,
            SovereignDimension::EconomicVelocity => &self.economic_velocity,
            SovereignDimension::CivicParticipation => &self.civic_participation,
            SovereignDimension::StewardshipCare => &self.stewardship_care,
            SovereignDimension::SemanticResonance => &self.semantic_resonance,
            SovereignDimension::DomainCompetence => &self.domain_competence,
        }
    }
}

// ============================================================================
// Normalization
// ============================================================================

/// Normalize a dimension input to [0.0, 1.0].
///
/// Formula: `score = saturation × quality × recency_weight`
///
/// Where:
/// - `saturation = min(1.0, activity_count / baseline)` — caps at full
/// - `quality` — cluster-provided [0,1] ratio
/// - `recency_weight = 2^(-days_since_last / half_life)` — exponential decay
///
/// Returns 0.0 if baseline is 0 or inputs are non-finite.
pub fn normalize(input: &DimensionInput) -> f64 {
    if input.baseline == 0 {
        return 0.0;
    }

    // Saturation: how many activities relative to baseline
    let saturation = (input.activity_count as f64 / input.baseline as f64).min(1.0);

    // Quality: cluster-provided ratio, clamped
    let quality = if input.quality.is_finite() {
        input.quality.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Recency: exponential decay based on time since last activity
    let recency = if input.recency_half_life_days > 0.0 && input.days_since_last >= 0.0 {
        (-input.days_since_last * core::f64::consts::LN_2 / input.recency_half_life_days).exp()
    } else {
        1.0 // No decay configured or just active
    };

    (saturation * quality * recency).clamp(0.0, 1.0)
}

/// Normalize all 8 dimensions and return a SovereignProfile.
pub fn normalize_all(collected: &CollectedDimensions) -> crate::SovereignProfile {
    crate::SovereignProfile {
        epistemic_integrity: normalize(&collected.epistemic_integrity),
        thermodynamic_yield: normalize(&collected.thermodynamic_yield),
        network_resilience: normalize(&collected.network_resilience),
        economic_velocity: normalize(&collected.economic_velocity),
        civic_participation: normalize(&collected.civic_participation),
        stewardship_care: normalize(&collected.stewardship_care),
        semantic_resonance: normalize(&collected.semantic_resonance),
        domain_competence: normalize(&collected.domain_competence),
    }
}

// ============================================================================
// Dimension-specific defaults (baselines and half-lives)
// ============================================================================

/// Default baselines and half-lives for each dimension.
///
/// Communities can override these via governance. These defaults represent
/// a moderate-activity community.
pub fn default_input(dim: SovereignDimension) -> DimensionInput {
    match dim {
        SovereignDimension::EpistemicIntegrity => DimensionInput {
            baseline: 50,                 // 50 validated claims for saturation
            recency_half_life_days: 90.0, // knowledge contribution decays over 3 months
            ..Default::default()
        },
        SovereignDimension::ThermodynamicYield => DimensionInput {
            baseline: 30,                 // 30 verified energy contributions
            recency_half_life_days: 30.0, // energy is real-time, decays fast
            ..Default::default()
        },
        SovereignDimension::NetworkResilience => DimensionInput {
            baseline: 720,                // 720 hours (~30 days) of uptime
            recency_half_life_days: 14.0, // infrastructure needs constant maintenance
            ..Default::default()
        },
        SovereignDimension::EconomicVelocity => DimensionInput {
            baseline: 50,                 // 50 TEND exchanges
            recency_half_life_days: 60.0, // economic activity over 2 months
            ..Default::default()
        },
        SovereignDimension::CivicParticipation => DimensionInput {
            baseline: 20,                  // 20 governance actions (votes + proposals + jury)
            recency_half_life_days: 180.0, // civic engagement decays slowly
            ..Default::default()
        },
        SovereignDimension::StewardshipCare => DimensionInput {
            baseline: 30,                 // 30 care sessions or maintenance tasks
            recency_half_life_days: 60.0, // care work over 2 months
            ..Default::default()
        },
        SovereignDimension::SemanticResonance => DimensionInput {
            baseline: 1,                  // 1.0 = full alignment (cosine similarity)
            recency_half_life_days: 30.0, // community alignment shifts monthly
            ..Default::default()
        },
        SovereignDimension::DomainCompetence => DimensionInput {
            baseline: 3,                   // 3 living credentials for saturation
            recency_half_life_days: 365.0, // expertise decays over a year
            ..Default::default()
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn input(count: u64, baseline: u64, quality: f64, days: f64, half_life: f64) -> DimensionInput {
        DimensionInput {
            activity_count: count,
            baseline,
            quality,
            days_since_last: days,
            recency_half_life_days: half_life,
        }
    }

    #[test]
    fn normalize_full_saturation_full_quality_recent() {
        let score = normalize(&input(100, 50, 1.0, 0.0, 30.0));
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_half_saturation() {
        let score = normalize(&input(25, 50, 1.0, 0.0, 30.0));
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn normalize_half_quality() {
        let score = normalize(&input(50, 50, 0.5, 0.0, 30.0));
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn normalize_recency_decay_at_half_life() {
        let score = normalize(&input(50, 50, 1.0, 30.0, 30.0));
        // At half-life, recency = 0.5
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn normalize_zero_activity_is_zero() {
        let score = normalize(&input(0, 50, 1.0, 0.0, 30.0));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn normalize_zero_baseline_is_zero() {
        let score = normalize(&input(100, 0, 1.0, 0.0, 30.0));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn normalize_zero_quality_is_zero() {
        let score = normalize(&input(50, 50, 0.0, 0.0, 30.0));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn normalize_nan_quality_is_zero() {
        let score = normalize(&input(50, 50, f64::NAN, 0.0, 30.0));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn normalize_capped_at_one() {
        let score = normalize(&input(200, 50, 1.0, 0.0, 30.0));
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_all_produces_valid_profile() {
        let mut collected = CollectedDimensions::default();
        collected.epistemic_integrity = input(40, 50, 0.9, 5.0, 90.0);
        collected.civic_participation = input(15, 20, 0.8, 10.0, 180.0);
        collected.domain_competence = input(2, 3, 0.95, 30.0, 365.0);

        let profile = normalize_all(&collected);
        assert!(profile.epistemic_integrity > 0.0);
        assert!(profile.civic_participation > 0.0);
        assert!(profile.domain_competence > 0.0);
        // Dimensions with no input should be 0.0
        assert_eq!(profile.thermodynamic_yield, 0.0);
        assert_eq!(profile.network_resilience, 0.0);
    }

    #[test]
    fn default_inputs_have_nonzero_baselines() {
        for dim in SovereignDimension::ALL {
            let def = default_input(dim);
            assert!(def.baseline > 0, "Dimension {:?} has zero baseline", dim);
            assert!(
                def.recency_half_life_days > 0.0,
                "Dimension {:?} has zero half-life",
                dim
            );
        }
    }

    #[test]
    fn conservative_defaults_start_at_zero() {
        // With default inputs (no activity), all scores should be 0.0
        let collected = CollectedDimensions::default();
        let profile = normalize_all(&collected);
        for dim in SovereignDimension::ALL {
            assert_eq!(
                profile.get(dim),
                0.0,
                "Dimension {:?} should default to 0.0",
                dim
            );
        }
    }
}
