// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parameterized exponential decay for the 8D Sovereign Profile.
//!
//! Civic standing is not permanent. Without ongoing engagement, all
//! dimensions decay toward zero following an exponential curve:
//!
//! ```text
//! S_decayed = S_raw × e^(-λ × Δt)
//! ```
//!
//! where λ is the community-configurable decay rate and Δt is elapsed
//! time since last verified interaction.
//!
//! ## Ostrom Principle 2: Rules Matched to Local Conditions
//!
//! Different communities operate on different temporal frequencies:
//! - Emergency pods need rapid decay (λ_max) — only active responders govern
//! - Land trusts need slow decay (λ_min) — stewards aren't penalized for a sabbatical
//!
//! The decay rate λ is tuneable via Steward-tier governance. The existence
//! of decay is constitutional and cannot be disabled.
//!
//! ## Constitutional Bounds (Immutable)
//!
//! - `LAMBDA_MIN` = 0.001 per day — half-life ≈ 693 days (~23 months)
//! - `LAMBDA_MAX` = 0.020 per day — half-life ≈ 35 days (~5 weeks)
//! - λ = 0 is mathematically impossible (prevents permanent oligarchy)
//! - `RAMP_DAYS_MIN` = 30 — minimum transition period for λ changes
//!
//! ## Anti-Grief Protection
//!
//! When governance votes to change λ, the transition is **gradual**:
//!
//! ```text
//! λ_effective(t) = λ_old + (λ_new - λ_old) × min(1, (t - t_vote) / T_ramp)
//! ```
//!
//! This prevents a single governance vote from instantly dropping half
//! the population below their civic tier threshold.
//!
//! ## ZKP Compatibility
//!
//! The decay formula is designed for STARK circuit proving:
//! - Public inputs: λ, t_now, threshold
//! - Private inputs: S_raw, t_last
//! - Proof: S_raw × e^(-λ × Δt) ≥ threshold
//!
//! The network learns only that the citizen passes — not their raw score,
//! last activity timestamp, or which dimensions are strong.

use crate::SovereignProfile;

// ============================================================================
// Constitutional constants (immutable — no governance can change these)
// ============================================================================

/// Minimum decay rate per day. Half-life ≈ 693 days (~23 months).
/// Protects long-term stewards from unfair penalization during sabbaticals.
pub const LAMBDA_MIN: f64 = 0.001;

/// Maximum decay rate per day. Half-life ≈ 35 days (~5 weeks).
/// Ensures only active participants retain governance power in fast-paced
/// communities (disaster relief, emergency response).
pub const LAMBDA_MAX: f64 = 0.020;

/// Minimum ramp period (days) for any λ transition.
/// Prevents sudden disenfranchisement from a single governance vote.
pub const RAMP_DAYS_MIN: u32 = 30;

/// Grace period (days) after a λ change announcement.
/// Citizens whose score would drop below their current tier get notified
/// and have this many days to re-engage before the new decay takes effect.
pub const GRACE_PERIOD_DAYS: u32 = 30;

/// Microseconds per day (constant for timestamp math).
const MICROS_PER_DAY: f64 = 86_400_000_000.0;

// ============================================================================
// DecayConfig — community-configurable parameters
// ============================================================================

/// Community-configurable decay parameters.
///
/// Modified via Steward-tier governance. Constitutional bounds are enforced
/// at construction time — invalid configurations are rejected.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecayConfig {
    /// Current decay rate λ (per day). Range: [LAMBDA_MIN, LAMBDA_MAX].
    pub lambda: f64,
    /// Ramp period for λ transitions (days). Minimum: RAMP_DAYS_MIN.
    pub ramp_days: u32,
    /// Previous λ value (for gradual transition). Set to `lambda` when no
    /// transition is in progress.
    pub lambda_previous: f64,
    /// Microsecond timestamp when the current λ was approved by governance.
    pub lambda_changed_at_us: u64,
}

impl DecayConfig {
    /// Create a new decay config with constitutional bound enforcement.
    ///
    /// Returns `None` if λ is outside [LAMBDA_MIN, LAMBDA_MAX] or
    /// ramp_days < RAMP_DAYS_MIN.
    pub fn new(lambda: f64, ramp_days: u32) -> Option<Self> {
        if lambda < LAMBDA_MIN || lambda > LAMBDA_MAX {
            return None;
        }
        if ramp_days < RAMP_DAYS_MIN {
            return None;
        }
        Some(Self {
            lambda,
            ramp_days,
            lambda_previous: lambda,
            lambda_changed_at_us: 0,
        })
    }

    /// Default configuration: moderate decay (half-life ≈ 347 days).
    pub fn default_governance() -> Self {
        Self {
            lambda: 0.002,
            ramp_days: 30,
            lambda_previous: 0.002,
            lambda_changed_at_us: 0,
        }
    }

    /// Emergency pod configuration: rapid decay (half-life ≈ 35 days).
    pub fn emergency_pod() -> Self {
        Self {
            lambda: LAMBDA_MAX,
            ramp_days: RAMP_DAYS_MIN,
            lambda_previous: LAMBDA_MAX,
            lambda_changed_at_us: 0,
        }
    }

    /// Land trust configuration: slow decay (half-life ≈ 693 days).
    pub fn land_trust() -> Self {
        Self {
            lambda: LAMBDA_MIN,
            ramp_days: 90, // 3 months — land trusts move slowly
            lambda_previous: LAMBDA_MIN,
            lambda_changed_at_us: 0,
        }
    }

    /// Propose a new λ value via governance. Returns the updated config
    /// with the transition ramp initiated. Returns `None` if the new λ
    /// violates constitutional bounds.
    pub fn propose_lambda_change(&self, new_lambda: f64, vote_timestamp_us: u64) -> Option<Self> {
        if new_lambda < LAMBDA_MIN || new_lambda > LAMBDA_MAX {
            return None;
        }
        Some(Self {
            lambda: new_lambda,
            ramp_days: self.ramp_days,
            lambda_previous: self.effective_lambda(vote_timestamp_us),
            lambda_changed_at_us: vote_timestamp_us,
        })
    }

    /// Compute the effective λ at a given timestamp, accounting for the
    /// gradual ramp transition.
    ///
    /// During the ramp period, λ interpolates linearly from λ_previous
    /// to λ_new. After the ramp completes, returns the target λ.
    pub fn effective_lambda(&self, now_us: u64) -> f64 {
        // No transition in progress: previous == target
        if (self.lambda - self.lambda_previous).abs() < 1e-15 {
            return self.lambda.clamp(LAMBDA_MIN, LAMBDA_MAX);
        }
        if now_us <= self.lambda_changed_at_us {
            return self.lambda_previous.clamp(LAMBDA_MIN, LAMBDA_MAX);
        }

        let elapsed_days = (now_us - self.lambda_changed_at_us) as f64 / MICROS_PER_DAY;
        let ramp_progress = (elapsed_days / self.ramp_days as f64).min(1.0);

        let effective = self.lambda_previous + (self.lambda - self.lambda_previous) * ramp_progress;

        // Constitutional clamp (belt + suspenders)
        effective.clamp(LAMBDA_MIN, LAMBDA_MAX)
    }

    /// Half-life in days for the current effective λ.
    /// Half-life = ln(2) / λ
    pub fn half_life_days(&self, now_us: u64) -> f64 {
        let lambda = self.effective_lambda(now_us);
        if lambda > 0.0 {
            core::f64::consts::LN_2 / lambda
        } else {
            f64::INFINITY
        }
    }

    /// Check if a λ transition is currently in progress.
    pub fn is_transitioning(&self, now_us: u64) -> bool {
        if (self.lambda - self.lambda_previous).abs() < 1e-15 {
            return false;
        }
        if now_us <= self.lambda_changed_at_us {
            return true; // transition hasn't started yet
        }
        let elapsed_days = (now_us - self.lambda_changed_at_us) as f64 / MICROS_PER_DAY;
        elapsed_days < self.ramp_days as f64
    }

    /// Ramp progress as a fraction [0.0, 1.0]. Returns 1.0 if no transition
    /// is in progress.
    pub fn ramp_progress(&self, now_us: u64) -> f64 {
        if (self.lambda - self.lambda_previous).abs() < 1e-15 {
            return 1.0;
        }
        if now_us <= self.lambda_changed_at_us {
            return 0.0;
        }
        let elapsed_days = (now_us - self.lambda_changed_at_us) as f64 / MICROS_PER_DAY;
        (elapsed_days / self.ramp_days as f64).min(1.0)
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self::default_governance()
    }
}

// ============================================================================
// Decay computation
// ============================================================================

/// Apply exponential time-decay to a single dimension score.
///
/// ```text
/// S_decayed = S_raw × e^(-λ × Δt_days)
/// ```
///
/// Returns 0.0 if Δt is negative or λ is non-finite.
pub fn decay_score(raw_score: f64, lambda: f64, elapsed_days: f64) -> f64 {
    if !lambda.is_finite() || !elapsed_days.is_finite() || elapsed_days < 0.0 {
        return 0.0;
    }
    let decayed = raw_score * (-lambda * elapsed_days).exp();
    decayed.clamp(0.0, 1.0)
}

/// Apply time-decay to all 8 dimensions of a sovereign profile.
///
/// Each dimension decays independently using the same λ and Δt.
/// The `last_interaction_us` is the timestamp of the agent's most recent
/// verified civic interaction (any dimension).
pub fn apply_decay(
    profile: &SovereignProfile,
    last_interaction_us: u64,
    now_us: u64,
    config: &DecayConfig,
) -> SovereignProfile {
    let lambda = config.effective_lambda(now_us);
    let elapsed_days = if now_us > last_interaction_us {
        (now_us - last_interaction_us) as f64 / MICROS_PER_DAY
    } else {
        0.0
    };

    let raw = profile.as_array();
    let decayed: [f64; 8] = core::array::from_fn(|i| decay_score(raw[i], lambda, elapsed_days));
    SovereignProfile::from_array(decayed)
}

/// Predict when a profile will drop below a combined score threshold.
///
/// Returns the number of days until `combined_score(decayed_profile) < threshold`,
/// assuming no further civic interaction. Returns `f64::INFINITY` if the profile
/// is already below the threshold or would never cross it.
pub fn days_until_threshold(
    profile: &SovereignProfile,
    weights: &crate::weights::DimensionWeights,
    config: &DecayConfig,
    now_us: u64,
    threshold: f64,
) -> f64 {
    let current_score = profile.combined_score(weights);
    if current_score <= threshold {
        return 0.0;
    }
    if threshold <= 0.0 {
        return f64::INFINITY;
    }

    let lambda = config.effective_lambda(now_us);
    if lambda <= 0.0 {
        return f64::INFINITY;
    }

    // S × e^(-λt) = threshold  →  t = -ln(threshold/S) / λ
    let ratio = threshold / current_score;
    if ratio >= 1.0 {
        return 0.0;
    }
    -(ratio.ln()) / lambda
}

/// Identify agents who will drop below a tier threshold within the grace period.
///
/// Returns true if the agent's decayed score at (now + GRACE_PERIOD_DAYS) would
/// be below the given threshold. Used for proactive notifications.
pub fn needs_grace_notification(
    profile: &SovereignProfile,
    weights: &crate::weights::DimensionWeights,
    config: &DecayConfig,
    now_us: u64,
    threshold: f64,
) -> bool {
    let days = days_until_threshold(profile, weights, config, now_us, threshold);
    days < GRACE_PERIOD_DAYS as f64 && days > 0.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::DimensionWeights;

    const DAY_US: u64 = 86_400_000_000;

    // --- Constitutional bounds ---

    #[test]
    fn lambda_zero_rejected() {
        assert!(DecayConfig::new(0.0, 30).is_none());
    }

    #[test]
    fn lambda_negative_rejected() {
        assert!(DecayConfig::new(-0.001, 30).is_none());
    }

    #[test]
    fn lambda_below_min_rejected() {
        assert!(DecayConfig::new(LAMBDA_MIN - 0.0001, 30).is_none());
    }

    #[test]
    fn lambda_above_max_rejected() {
        assert!(DecayConfig::new(LAMBDA_MAX + 0.001, 30).is_none());
    }

    #[test]
    fn ramp_below_minimum_rejected() {
        assert!(DecayConfig::new(0.005, RAMP_DAYS_MIN - 1).is_none());
    }

    #[test]
    fn lambda_at_min_accepted() {
        assert!(DecayConfig::new(LAMBDA_MIN, 30).is_some());
    }

    #[test]
    fn lambda_at_max_accepted() {
        assert!(DecayConfig::new(LAMBDA_MAX, 30).is_some());
    }

    #[test]
    fn propose_change_to_zero_rejected() {
        let config = DecayConfig::default_governance();
        assert!(config.propose_lambda_change(0.0, 1_000_000).is_none());
    }

    // --- Decay math ---

    #[test]
    fn no_decay_at_zero_elapsed() {
        let score = decay_score(0.8, 0.005, 0.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn decay_reduces_score_over_time() {
        let score_0 = decay_score(1.0, 0.005, 0.0);
        let score_30 = decay_score(1.0, 0.005, 30.0);
        let score_100 = decay_score(1.0, 0.005, 100.0);
        assert!(score_0 > score_30);
        assert!(score_30 > score_100);
        assert!(score_100 > 0.0);
    }

    #[test]
    fn half_life_math_is_correct() {
        // At t = half_life, score should be ~0.5
        let lambda = 0.002;
        let half_life = core::f64::consts::LN_2 / lambda; // ≈ 346.57 days
        let score = decay_score(1.0, lambda, half_life);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn decay_never_goes_negative() {
        let score = decay_score(1.0, LAMBDA_MAX, 10_000.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn decay_clamps_to_one() {
        // Negative elapsed shouldn't boost
        let score = decay_score(0.8, 0.005, -10.0);
        assert_eq!(score, 0.0); // negative elapsed → 0.0 (safety)
    }

    // --- Profile-level decay ---

    #[test]
    fn apply_decay_reduces_all_dimensions() {
        let profile = SovereignProfile::from_array([0.9; 8]);
        let config = DecayConfig::default_governance();
        let now = 100 * DAY_US;
        let last = 0;

        let decayed = apply_decay(&profile, last, now, &config);
        for dim in crate::SovereignDimension::ALL {
            assert!(
                decayed.get(dim) < 0.9,
                "Dimension {:?} should decay from 0.9, got {}",
                dim,
                decayed.get(dim)
            );
            assert!(
                decayed.get(dim) > 0.0,
                "Dimension {:?} should not decay to zero in 100 days",
                dim
            );
        }
    }

    #[test]
    fn no_decay_when_just_interacted() {
        let profile = SovereignProfile::from_array([0.7; 8]);
        let config = DecayConfig::default_governance();
        let now = 1000 * DAY_US;

        let decayed = apply_decay(&profile, now, now, &config);
        for dim in crate::SovereignDimension::ALL {
            assert!((decayed.get(dim) - 0.7).abs() < 1e-10);
        }
    }

    // --- Ramp transition ---

    #[test]
    fn effective_lambda_before_ramp_returns_previous() {
        let config = DecayConfig {
            lambda: 0.010,
            ramp_days: 30,
            lambda_previous: 0.002,
            lambda_changed_at_us: 100 * DAY_US,
        };
        // Before the vote
        let effective = config.effective_lambda(50 * DAY_US);
        assert!((effective - 0.002).abs() < 1e-10);
    }

    #[test]
    fn effective_lambda_at_ramp_midpoint_is_interpolated() {
        let config = DecayConfig {
            lambda: 0.010,
            ramp_days: 30,
            lambda_previous: 0.002,
            lambda_changed_at_us: 0,
        };
        // 15 days in = 50% ramp
        let effective = config.effective_lambda(15 * DAY_US);
        let expected = 0.002 + (0.010 - 0.002) * 0.5;
        assert!((effective - expected).abs() < 1e-6);
    }

    #[test]
    fn effective_lambda_after_ramp_returns_target() {
        let config = DecayConfig {
            lambda: 0.010,
            ramp_days: 30,
            lambda_previous: 0.002,
            lambda_changed_at_us: 0,
        };
        // 60 days in = past ramp
        let effective = config.effective_lambda(60 * DAY_US);
        assert!((effective - 0.010).abs() < 1e-10);
    }

    #[test]
    fn effective_lambda_clamped_during_ramp() {
        // Edge case: previous was at max, new is at min
        let config = DecayConfig {
            lambda: LAMBDA_MIN,
            ramp_days: 30,
            lambda_previous: LAMBDA_MAX,
            lambda_changed_at_us: 0,
        };
        let effective = config.effective_lambda(15 * DAY_US);
        assert!(effective >= LAMBDA_MIN);
        assert!(effective <= LAMBDA_MAX);
    }

    #[test]
    fn is_transitioning_during_ramp() {
        let config = DecayConfig {
            lambda: 0.010,
            ramp_days: 30,
            lambda_previous: 0.002,
            lambda_changed_at_us: 0,
        };
        assert!(config.is_transitioning(15 * DAY_US));
        assert!(!config.is_transitioning(31 * DAY_US));
    }

    // --- Half-life ---

    #[test]
    fn half_life_default_governance() {
        let config = DecayConfig::default_governance();
        let hl = config.half_life_days(0);
        // λ=0.002 → half-life = ln(2)/0.002 ≈ 346.57 days
        assert!((hl - 346.57).abs() < 1.0);
    }

    #[test]
    fn half_life_emergency_pod() {
        let config = DecayConfig::emergency_pod();
        let hl = config.half_life_days(0);
        // λ=0.020 → half-life = ln(2)/0.020 ≈ 34.66 days
        assert!((hl - 34.66).abs() < 1.0);
    }

    #[test]
    fn half_life_land_trust() {
        let config = DecayConfig::land_trust();
        let hl = config.half_life_days(0);
        // λ=0.001 → half-life = ln(2)/0.001 ≈ 693.15 days
        assert!((hl - 693.15).abs() < 1.0);
    }

    // --- Threshold prediction ---

    #[test]
    fn days_until_threshold_positive_when_above() {
        let profile = SovereignProfile::from_array([0.8; 8]);
        let weights = DimensionWeights::equal();
        let config = DecayConfig::default_governance();
        let threshold = 0.4;

        let days = days_until_threshold(&profile, &weights, &config, 0, threshold);
        assert!(days > 0.0);
        assert!(days < 1000.0);
        // At λ=0.002: time to drop from 0.8 to 0.4 = -ln(0.5)/0.002 ≈ 346.6 days
        assert!((days - 346.6).abs() < 1.0);
    }

    #[test]
    fn days_until_threshold_zero_when_already_below() {
        let profile = SovereignProfile::from_array([0.2; 8]);
        let weights = DimensionWeights::equal();
        let config = DecayConfig::default_governance();

        let days = days_until_threshold(&profile, &weights, &config, 0, 0.4);
        assert_eq!(days, 0.0);
    }

    // --- Grace notification ---

    #[test]
    fn grace_notification_triggers_when_close_to_threshold() {
        let profile = SovereignProfile::from_array([0.42; 8]);
        let weights = DimensionWeights::equal();
        let config = DecayConfig::default_governance();
        let threshold = 0.4;

        // Score is 0.42, threshold is 0.40 — will drop below within ~25 days
        let needs = needs_grace_notification(&profile, &weights, &config, 0, threshold);
        assert!(
            needs,
            "Should trigger grace notification when close to threshold"
        );
    }

    #[test]
    fn grace_notification_does_not_trigger_when_far_above() {
        let profile = SovereignProfile::from_array([0.9; 8]);
        let weights = DimensionWeights::equal();
        let config = DecayConfig::default_governance();

        let needs = needs_grace_notification(&profile, &weights, &config, 0, 0.4);
        assert!(!needs, "Should not trigger when far above threshold");
    }

    // --- Preset configs ---

    #[test]
    fn all_presets_have_valid_bounds() {
        let configs = [
            DecayConfig::default_governance(),
            DecayConfig::emergency_pod(),
            DecayConfig::land_trust(),
        ];
        for config in &configs {
            assert!(
                config.lambda >= LAMBDA_MIN,
                "λ below min: {}",
                config.lambda
            );
            assert!(
                config.lambda <= LAMBDA_MAX,
                "λ above max: {}",
                config.lambda
            );
            assert!(
                config.ramp_days >= RAMP_DAYS_MIN,
                "ramp below min: {}",
                config.ramp_days
            );
        }
    }
}
