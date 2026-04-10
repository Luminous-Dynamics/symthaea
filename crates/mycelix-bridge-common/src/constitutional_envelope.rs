// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Constitutional Envelope — Immutable Governance Invariants
//!
//! The hard floor that no community can override, regardless of which
//! scoring model they adopt.  These invariants are mathematically enforced
//! before any score reaches governance gating.
//!
//! ## Invariants
//!
//! 1. **Decay must exist**: λ > 0.  Prevents permanent power hoarding.
//! 2. **No single dimension > 50%**: Prevents gaming via one metric.
//! 3. **Sybil maturation floor**: New identities have reduced weight for ≥ 72h.
//! 4. **Tier thresholds immutable**: The 5-tier structure cannot be changed.
//!
//! ## Design Rationale
//!
//! If a community can vote to set decay to zero, they reinvent the 1602
//! architecture — permanent, non-expiring governance power severed from
//! ongoing contribution.  The Constitutional Envelope makes this
//! mathematically impossible while leaving everything else configurable.
//!
//! Grounded in Ostrom (1990): certain rules are "constitutional immutability
//! for core rights" while operational rules remain community-tunable.

use serde::{Deserialize, Serialize};

use crate::consciousness_profile::ConsciousnessTier;

// ============================================================================
// Constitutional Constants — NEVER change these
// ============================================================================

/// Minimum decay rate (lambda per day).  Corresponds to a half-life of
/// ~693 days.  Communities can increase but never decrease below this.
pub const LAMBDA_MIN: f64 = 0.001;

/// Maximum decay rate (lambda per day).  Corresponds to a half-life of
/// ~35 days.  Prevents communities from making governance inaccessible
/// to anyone who takes a brief break.
pub const LAMBDA_MAX: f64 = 0.020;

/// Maximum weight any single dimension can have in a scoring model.
/// Prevents gaming by farming a single metric.  Forces broad civic
/// contribution across multiple dimensions.
pub const MAX_DIMENSION_WEIGHT: f64 = 0.50;

/// Minimum maturation period for new identities (hours).
/// Communities can increase (e.g., 168h = 1 week) but never decrease.
pub const MIN_MATURATION_HOURS: u64 = 72;

/// The immutable tier thresholds.  These define the 5-tier governance
/// structure and cannot be changed by any community.
///
/// Index: 0=Observer, 1=Participant, 2=Citizen, 3=Steward, 4=Guardian
pub const TIER_THRESHOLDS: [f64; 5] = [0.0, 0.3, 0.4, 0.6, 0.8];

/// Minimum number of dimensions a scoring model must have.
/// A 1D model is trivially gameable.
pub const MIN_DIMENSIONS: usize = 2;

/// Maximum number of dimensions a scoring model can have.
/// Prevents complexity explosion in ZK circuits.
pub const MAX_DIMENSIONS: usize = 16;

/// Weight sum tolerance.  Weights must sum to 1.0 ± this epsilon.
pub const WEIGHT_SUM_TOLERANCE: f64 = 1e-6;

// ============================================================================
// Violation Types
// ============================================================================

/// A constitutional invariant violation.  These are hard errors — the
/// system MUST reject the offending model or parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstitutionalViolation {
    /// Decay rate λ is below the constitutional minimum.
    DecayTooLow {
        lambda: f64,
        minimum: f64,
    },
    /// Decay rate λ exceeds the constitutional maximum.
    DecayTooHigh {
        lambda: f64,
        maximum: f64,
    },
    /// A dimension weight exceeds the constitutional maximum.
    DimensionWeightExceeds {
        dimension_index: usize,
        weight: f64,
        maximum: f64,
    },
    /// Dimension weights do not sum to 1.0 (± tolerance).
    WeightSumInvalid {
        sum: f64,
        expected: f64,
    },
    /// Sybil maturation period is below the constitutional minimum.
    MaturationTooShort {
        hours: u64,
        minimum: u64,
    },
    /// Too few dimensions for meaningful governance.
    TooFewDimensions {
        count: usize,
        minimum: usize,
    },
    /// Too many dimensions (ZK circuit complexity bound).
    TooManyDimensions {
        count: usize,
        maximum: usize,
    },
    /// A dimension weight is negative or non-finite.
    InvalidWeight {
        dimension_index: usize,
        weight: f64,
    },
    /// Lambda is NaN or infinite.
    InvalidLambda {
        lambda: f64,
    },
}

impl core::fmt::Display for ConstitutionalViolation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DecayTooLow { lambda, minimum } => {
                write!(f, "Decay rate {lambda} < constitutional minimum {minimum}")
            }
            Self::DecayTooHigh { lambda, maximum } => {
                write!(f, "Decay rate {lambda} > constitutional maximum {maximum}")
            }
            Self::DimensionWeightExceeds {
                dimension_index,
                weight,
                maximum,
            } => write!(
                f,
                "Dimension {dimension_index} weight {weight} > constitutional maximum {maximum}"
            ),
            Self::WeightSumInvalid { sum, expected } => {
                write!(f, "Weight sum {sum} != expected {expected}")
            }
            Self::MaturationTooShort { hours, minimum } => {
                write!(
                    f,
                    "Maturation period {hours}h < constitutional minimum {minimum}h"
                )
            }
            Self::TooFewDimensions { count, minimum } => {
                write!(f, "{count} dimensions < constitutional minimum {minimum}")
            }
            Self::TooManyDimensions { count, maximum } => {
                write!(f, "{count} dimensions > constitutional maximum {maximum}")
            }
            Self::InvalidWeight {
                dimension_index,
                weight,
            } => write!(f, "Dimension {dimension_index} has invalid weight {weight}"),
            Self::InvalidLambda { lambda } => {
                write!(f, "Lambda {lambda} is not a finite positive number")
            }
        }
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate a scoring model's parameters against constitutional invariants.
///
/// Returns `Ok(())` if all invariants are satisfied, or `Err` with the
/// first violation found.  Call this before activating any scoring model.
///
/// # Arguments
/// * `weights` - Per-dimension weights (must sum to 1.0, each ≤ 0.50)
/// * `lambda` - Decay rate per day (must be in [LAMBDA_MIN, LAMBDA_MAX])
/// * `maturation_hours` - Sybil maturation period (must be ≥ MIN_MATURATION_HOURS)
pub fn validate_model(
    weights: &[f64],
    lambda: f64,
    maturation_hours: u64,
) -> Result<(), ConstitutionalViolation> {
    // Invariant 0: Lambda must be finite and positive
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(ConstitutionalViolation::InvalidLambda { lambda });
    }

    // Invariant 1: Decay must exist (λ ≥ LAMBDA_MIN)
    if lambda < LAMBDA_MIN {
        return Err(ConstitutionalViolation::DecayTooLow {
            lambda,
            minimum: LAMBDA_MIN,
        });
    }

    // Also check upper bound
    if lambda > LAMBDA_MAX {
        return Err(ConstitutionalViolation::DecayTooHigh {
            lambda,
            maximum: LAMBDA_MAX,
        });
    }

    // Dimension count bounds
    if weights.len() < MIN_DIMENSIONS {
        return Err(ConstitutionalViolation::TooFewDimensions {
            count: weights.len(),
            minimum: MIN_DIMENSIONS,
        });
    }
    if weights.len() > MAX_DIMENSIONS {
        return Err(ConstitutionalViolation::TooManyDimensions {
            count: weights.len(),
            maximum: MAX_DIMENSIONS,
        });
    }

    // Invariant 2: No single dimension > 50%
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(ConstitutionalViolation::InvalidWeight {
                dimension_index: i,
                weight: w,
            });
        }
        if w > MAX_DIMENSION_WEIGHT {
            return Err(ConstitutionalViolation::DimensionWeightExceeds {
                dimension_index: i,
                weight: w,
                maximum: MAX_DIMENSION_WEIGHT,
            });
        }
    }

    // Weights must sum to 1.0
    let sum: f64 = weights.iter().sum();
    if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        return Err(ConstitutionalViolation::WeightSumInvalid {
            sum,
            expected: 1.0,
        });
    }

    // Invariant 3: Sybil maturation floor
    if maturation_hours < MIN_MATURATION_HOURS {
        return Err(ConstitutionalViolation::MaturationTooShort {
            hours: maturation_hours,
            minimum: MIN_MATURATION_HOURS,
        });
    }

    Ok(())
}

/// Map a combined score to a governance tier using the immutable thresholds.
///
/// This is the ONLY function that performs score→tier mapping.  The
/// thresholds are constitutional constants that cannot be changed by
/// any community governance action.
///
/// Invariant 4: Tier thresholds are immutable.
pub fn score_to_tier(score: f64) -> ConsciousnessTier {
    let s = if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    };

    if s >= TIER_THRESHOLDS[4] {
        ConsciousnessTier::Guardian
    } else if s >= TIER_THRESHOLDS[3] {
        ConsciousnessTier::Steward
    } else if s >= TIER_THRESHOLDS[2] {
        ConsciousnessTier::Citizen
    } else if s >= TIER_THRESHOLDS[1] {
        ConsciousnessTier::Participant
    } else {
        ConsciousnessTier::Observer
    }
}

/// Validate a computed score — sanitize NaN/Inf, clamp to [0, 1].
///
/// This runs AFTER a scoring model computes a raw score, ensuring
/// the output is always a valid probability regardless of model bugs.
pub fn sanitize_score(raw: f64) -> f64 {
    if raw.is_finite() {
        raw.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute the decay multiplier for a given lambda and elapsed time.
///
/// Returns M = e^(-λ × Δt) where Δt is in days.
///
/// This is the pre-computed multiplier that should be passed to the
/// Miden ZK circuit as a private input (avoiding in-circuit exponentiation).
pub fn decay_multiplier(lambda: f64, elapsed_days: f64) -> f64 {
    if !lambda.is_finite() || !elapsed_days.is_finite() || elapsed_days < 0.0 {
        return 0.0;
    }
    let m = (-lambda * elapsed_days).exp();
    if m.is_finite() {
        m.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Apply decay to a score: S_decayed = S_raw × e^(-λ × Δt)
///
/// Convenience wrapper around `decay_multiplier`.
pub fn apply_decay(score: f64, lambda: f64, elapsed_days: f64) -> f64 {
    sanitize_score(score * decay_multiplier(lambda, elapsed_days))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Invariant 1: Decay must exist ----

    #[test]
    fn valid_model_passes() {
        let weights = &[0.25, 0.25, 0.30, 0.20]; // canonical 4D
        assert!(validate_model(weights, 0.002, 72).is_ok());
    }

    #[test]
    fn zero_lambda_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, 0.0, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::InvalidLambda { .. })
        ));
    }

    #[test]
    fn negative_lambda_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, -0.001, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::InvalidLambda { .. })
        ));
    }

    #[test]
    fn lambda_below_minimum_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, 0.0005, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::DecayTooLow { .. })
        ));
    }

    #[test]
    fn lambda_above_maximum_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, 0.05, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::DecayTooHigh { .. })
        ));
    }

    #[test]
    fn lambda_at_minimum_accepted() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, LAMBDA_MIN, 72).is_ok());
    }

    #[test]
    fn lambda_at_maximum_accepted() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, LAMBDA_MAX, 72).is_ok());
    }

    #[test]
    fn nan_lambda_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, f64::NAN, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::InvalidLambda { .. })
        ));
    }

    // ---- Invariant 2: No single dimension > 50% ----

    #[test]
    fn weight_exceeding_50_percent_rejected() {
        let weights = &[0.60, 0.40]; // 60% > 50%
        let result = validate_model(weights, 0.002, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::DimensionWeightExceeds { .. })
        ));
    }

    #[test]
    fn weight_at_50_percent_accepted() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, 0.002, 72).is_ok());
    }

    #[test]
    fn negative_weight_rejected() {
        let weights = &[0.50, 0.50, -0.10, 0.10];
        let result = validate_model(weights, 0.002, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::InvalidWeight { .. })
        ));
    }

    #[test]
    fn nan_weight_rejected() {
        let weights = &[0.50, f64::NAN, 0.50];
        let result = validate_model(weights, 0.002, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::InvalidWeight { .. })
        ));
    }

    #[test]
    fn weights_not_summing_to_one_rejected() {
        let weights = &[0.30, 0.30, 0.30]; // sums to 0.90
        let result = validate_model(weights, 0.002, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::WeightSumInvalid { .. })
        ));
    }

    #[test]
    fn canonical_4d_weights_pass() {
        let weights = &[0.25, 0.25, 0.30, 0.20]; // I/R/C/E
        assert!(validate_model(weights, 0.002, 72).is_ok());
    }

    #[test]
    fn sovereign_8d_governance_weights_pass() {
        // From sovereign-profile governance preset
        let weights = &[0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10];
        assert!(validate_model(weights, 0.002, 72).is_ok());
    }

    #[test]
    fn minimal_2d_model_passes() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, 0.002, 72).is_ok());
    }

    #[test]
    fn single_dimension_rejected() {
        let weights = &[1.0]; // 100% > 50% AND < MIN_DIMENSIONS
        let result = validate_model(weights, 0.002, 72);
        assert!(result.is_err());
    }

    #[test]
    fn seventeen_dimensions_rejected() {
        let weights: Vec<f64> = (0..17).map(|_| 1.0 / 17.0).collect();
        let result = validate_model(&weights, 0.002, 72);
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::TooManyDimensions { .. })
        ));
    }

    // ---- Invariant 3: Sybil maturation floor ----

    #[test]
    fn maturation_below_minimum_rejected() {
        let weights = &[0.50, 0.50];
        let result = validate_model(weights, 0.002, 48); // 48h < 72h
        assert!(matches!(
            result,
            Err(ConstitutionalViolation::MaturationTooShort { .. })
        ));
    }

    #[test]
    fn maturation_at_minimum_accepted() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, 0.002, MIN_MATURATION_HOURS).is_ok());
    }

    #[test]
    fn maturation_above_minimum_accepted() {
        let weights = &[0.50, 0.50];
        assert!(validate_model(weights, 0.002, 168).is_ok()); // 1 week
    }

    // ---- Invariant 4: Tier thresholds immutable ----

    #[test]
    fn tier_thresholds_are_monotonic() {
        for pair in TIER_THRESHOLDS.windows(2) {
            assert!(
                pair[0] < pair[1],
                "Tier thresholds must be strictly increasing: {} >= {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn score_to_tier_matches_canonical() {
        assert_eq!(score_to_tier(0.0), ConsciousnessTier::Observer);
        assert_eq!(score_to_tier(0.29), ConsciousnessTier::Observer);
        assert_eq!(score_to_tier(0.3), ConsciousnessTier::Participant);
        assert_eq!(score_to_tier(0.39), ConsciousnessTier::Participant);
        assert_eq!(score_to_tier(0.4), ConsciousnessTier::Citizen);
        assert_eq!(score_to_tier(0.59), ConsciousnessTier::Citizen);
        assert_eq!(score_to_tier(0.6), ConsciousnessTier::Steward);
        assert_eq!(score_to_tier(0.79), ConsciousnessTier::Steward);
        assert_eq!(score_to_tier(0.8), ConsciousnessTier::Guardian);
        assert_eq!(score_to_tier(1.0), ConsciousnessTier::Guardian);
    }

    #[test]
    fn score_to_tier_handles_nan() {
        assert_eq!(score_to_tier(f64::NAN), ConsciousnessTier::Observer);
    }

    #[test]
    fn score_to_tier_handles_infinity() {
        // Non-finite values are sanitized to 0.0 → Observer
        assert_eq!(score_to_tier(f64::INFINITY), ConsciousnessTier::Observer);
        assert_eq!(
            score_to_tier(f64::NEG_INFINITY),
            ConsciousnessTier::Observer
        );
    }

    #[test]
    fn score_to_tier_clamps_above_one() {
        assert_eq!(score_to_tier(1.5), ConsciousnessTier::Guardian);
    }

    #[test]
    fn score_to_tier_clamps_below_zero() {
        assert_eq!(score_to_tier(-0.5), ConsciousnessTier::Observer);
    }

    // ---- Decay multiplier ----

    #[test]
    fn decay_multiplier_at_zero_elapsed() {
        let m = decay_multiplier(0.002, 0.0);
        assert!((m - 1.0).abs() < 1e-10, "No elapsed time → no decay");
    }

    #[test]
    fn decay_multiplier_decreases_over_time() {
        let m1 = decay_multiplier(0.002, 30.0);
        let m2 = decay_multiplier(0.002, 60.0);
        assert!(m2 < m1, "Decay should increase with time");
        assert!(m1 < 1.0, "30-day decay should reduce below 1.0");
    }

    #[test]
    fn decay_multiplier_higher_lambda_decays_faster() {
        let m_slow = decay_multiplier(0.001, 30.0);
        let m_fast = decay_multiplier(0.010, 30.0);
        assert!(
            m_fast < m_slow,
            "Higher lambda should decay faster: fast={}, slow={}",
            m_fast,
            m_slow
        );
    }

    #[test]
    fn decay_multiplier_handles_nan() {
        assert_eq!(decay_multiplier(f64::NAN, 30.0), 0.0);
        assert_eq!(decay_multiplier(0.002, f64::NAN), 0.0);
    }

    #[test]
    fn decay_multiplier_handles_negative_time() {
        assert_eq!(decay_multiplier(0.002, -10.0), 0.0);
    }

    #[test]
    fn apply_decay_full_score_30_days() {
        let decayed = apply_decay(1.0, 0.002, 30.0);
        // e^(-0.002 * 30) = e^(-0.06) ≈ 0.9418
        assert!(
            decayed > 0.93 && decayed < 0.95,
            "Score after 30 days at lambda=0.002 should be ~0.94, got {}",
            decayed
        );
    }

    #[test]
    fn apply_decay_respects_lambda_min_half_life() {
        // At LAMBDA_MIN (0.001), half-life = ln(2)/0.001 ≈ 693 days
        let decayed = apply_decay(1.0, LAMBDA_MIN, 693.0);
        assert!(
            decayed > 0.45 && decayed < 0.55,
            "At lambda_min half-life, score should be ~0.5, got {}",
            decayed
        );
    }

    #[test]
    fn apply_decay_respects_lambda_max_half_life() {
        // At LAMBDA_MAX (0.020), half-life = ln(2)/0.020 ≈ 35 days
        let decayed = apply_decay(1.0, LAMBDA_MAX, 35.0);
        assert!(
            decayed > 0.45 && decayed < 0.55,
            "At lambda_max half-life, score should be ~0.5, got {}",
            decayed
        );
    }

    // ---- Violation Display ----

    #[test]
    fn violation_display_is_readable() {
        let v = ConstitutionalViolation::DecayTooLow {
            lambda: 0.0005,
            minimum: 0.001,
        };
        let s = format!("{}", v);
        assert!(s.contains("0.0005"));
        assert!(s.contains("0.001"));
    }

    // ---- Constitutional constants are consistent ----

    #[test]
    fn lambda_min_less_than_max() {
        assert!(LAMBDA_MIN < LAMBDA_MAX);
    }

    #[test]
    fn tier_thresholds_match_consciousness_tier() {
        // Verify our immutable thresholds match the canonical tier definitions
        assert_eq!(
            score_to_tier(TIER_THRESHOLDS[0]),
            ConsciousnessTier::Observer
        );
        assert_eq!(
            score_to_tier(TIER_THRESHOLDS[1]),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            score_to_tier(TIER_THRESHOLDS[2]),
            ConsciousnessTier::Citizen
        );
        assert_eq!(
            score_to_tier(TIER_THRESHOLDS[3]),
            ConsciousnessTier::Steward
        );
        assert_eq!(
            score_to_tier(TIER_THRESHOLDS[4]),
            ConsciousnessTier::Guardian
        );
    }

    #[test]
    fn max_dimension_weight_prevents_single_axis_domination() {
        // With MAX_DIMENSION_WEIGHT = 0.50, any model needs at least 2 dimensions
        // each carrying meaningful weight.
        assert!(MAX_DIMENSION_WEIGHT <= 0.50);
    }
}
