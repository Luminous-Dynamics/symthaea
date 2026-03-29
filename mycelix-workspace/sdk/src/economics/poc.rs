// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MYCEL Score Calculation
//!
//! Replaces Proof of Contribution with 4-component MYCEL scoring:
//!
//! | Component     | Weight | Source |
//! |--------------|--------|--------|
//! | Participation | 40%    | Transaction activity, governance voting, commons engagement |
//! | Recognition   | 20%    | Weighted recognition events from other members |
//! | Validation    | 20%    | Quality of work as validator/contributor |
//! | Longevity     | 20%    | Time active, capped at 24 months to max |
//!
//! Constitutional: MYCEL is non-transferable.

use serde::{Deserialize, Serialize};

/// MYCEL score with 4-component breakdown (0.0 - 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelScore {
    /// Participation component (40% weight)
    pub participation: f64,
    /// Recognition component (20% weight)
    pub recognition: f64,
    /// Validation component (20% weight)
    pub validation: f64,
    /// Longevity component (20% weight, capped at 24 months)
    pub longevity: f64,
    /// Composite MYCEL score (0.0 - 1.0)
    pub composite: f64,
}

impl MycelScore {
    /// Create apprentice score (starting state)
    pub fn apprentice() -> Self {
        Self {
            participation: 0.1,
            recognition: 0.0,
            validation: 0.0,
            longevity: 0.0,
            composite: 0.1,
        }
    }
}

/// Individual MYCEL component types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MycelComponent {
    /// Participation component (tx activity, governance, commons)
    Participation,
    /// Recognition component (peer recognition scores)
    Recognition,
    /// Validation component (quality ratings, validator performance)
    Validation,
    /// Longevity component (time-based, capped at 24 months)
    Longevity,
}

/// Input data for MYCEL score calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelCalculation {
    /// Participation score (0.0-1.0): tx activity, governance, commons
    pub participation: f64,
    /// Recognition score (0.0-1.0): from recognition module
    pub recognition: f64,
    /// Validation score (0.0-1.0): quality ratings, validator performance
    pub validation: f64,
    /// Active months (capped at 24 for max longevity)
    pub active_months: u32,
}

/// Calculate the composite MYCEL score from 4 components.
///
/// Weights: Participation 40%, Recognition 20%, Validation 20%, Longevity 20%
pub fn calculate_mycel_score(calc: &MycelCalculation) -> MycelScore {
    let participation = calc.participation.clamp(0.0, 1.0);
    let recognition = calc.recognition.clamp(0.0, 1.0);
    let validation = calc.validation.clamp(0.0, 1.0);

    // Longevity: linear ramp to 1.0 at 24 months, then capped
    let longevity = (calc.active_months as f64 / 24.0).min(1.0);

    let composite =
        participation * 0.40 + recognition * 0.20 + validation * 0.20 + longevity * 0.20;

    MycelScore {
        participation,
        recognition,
        validation,
        longevity,
        composite: composite.clamp(0.0, 1.0),
    }
}

/// Jubilee normalization (every 4 years):
/// `new_mycel = 0.3 + (current - 0.3) * 0.8`
///
/// Compresses toward mean without resetting.
/// Active members recover quickly; inactive members drift toward newcomer level.
pub fn jubilee_normalize(current: f64) -> f64 {
    let normalized = 0.3 + (current - 0.3) * 0.8;
    normalized.clamp(0.0, 1.0)
}

/// Apply 5% annual passive decay for non-participation
pub fn passive_decay(current: f64) -> f64 {
    let decayed = current * 0.95;
    decayed.max(0.0)
}

/// Anti-gaming detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingDetection {
    /// Sybil cluster suspicion score
    pub sybil_score: f64,
    /// Burst activity detection
    pub burst_detected: bool,
    /// Wash trading suspicion
    pub wash_trade_score: f64,
    /// Recommendation
    pub recommendation: GamingRecommendation,
}

/// Recommendation from anti-gaming detection analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GamingRecommendation {
    /// No action needed
    Clear,
    /// Increased monitoring
    Watch,
    /// Manual review required
    Review,
    /// Score adjustment applied
    Penalized,
}

/// Detect potential gaming behavior in MYCEL metrics
pub fn detect_gaming(
    current: &MycelCalculation,
    historical: &[MycelCalculation],
) -> GamingDetection {
    let mut sybil_score = 0.0;
    let mut burst_detected = false;
    let mut wash_trade_score = 0.0;

    // Check for sudden spikes in participation
    if let Some(prev) = historical.last() {
        let participation_delta = (current.participation - prev.participation).abs();
        if participation_delta > 0.5 {
            burst_detected = true;
            sybil_score += 0.3;
        }
    }

    // High recognition without participation is suspicious
    if current.recognition > 0.8 && current.participation < 0.3 {
        sybil_score += 0.4;
        wash_trade_score += 0.3;
    }

    let recommendation = if sybil_score > 0.6 {
        GamingRecommendation::Review
    } else if sybil_score > 0.3 || burst_detected {
        GamingRecommendation::Watch
    } else {
        GamingRecommendation::Clear
    };

    GamingDetection {
        sybil_score,
        burst_detected,
        wash_trade_score,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mycel_score_calculation() {
        let calc = MycelCalculation {
            participation: 0.8,
            recognition: 0.6,
            validation: 0.7,
            active_months: 12,
        };

        let score = calculate_mycel_score(&calc);

        // participation: 0.8 * 0.4 = 0.32
        // recognition: 0.6 * 0.2 = 0.12
        // validation: 0.7 * 0.2 = 0.14
        // longevity: 0.5 * 0.2 = 0.10  (12/24 = 0.5)
        // total: 0.68
        assert!((score.composite - 0.68).abs() < 0.01);
        assert!((score.longevity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_longevity_caps_at_24_months() {
        let calc = MycelCalculation {
            participation: 1.0,
            recognition: 1.0,
            validation: 1.0,
            active_months: 48, // 4 years, but capped at 24 months
        };

        let score = calculate_mycel_score(&calc);
        assert!((score.longevity - 1.0).abs() < 0.01);
        // All max: 0.4 + 0.2 + 0.2 + 0.2 = 1.0
        assert!((score.composite - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_jubilee_normalization() {
        // Steward at 0.8 → 0.3 + 0.5 * 0.8 = 0.7
        assert!((jubilee_normalize(0.8) - 0.7).abs() < 0.001);

        // Newcomer at 0.3 → stays at 0.3
        assert!((jubilee_normalize(0.3) - 0.3).abs() < 0.001);

        // Low score at 0.1 → 0.3 + (-0.2) * 0.8 = 0.14
        assert!((jubilee_normalize(0.1) - 0.14).abs() < 0.001);
    }

    #[test]
    fn test_gaming_detection() {
        let current = MycelCalculation {
            participation: 0.1, // Low participation
            recognition: 0.9,   // But high recognition — suspicious
            validation: 0.5,
            active_months: 3,
        };

        let detection = detect_gaming(&current, &[]);
        assert!(detection.sybil_score > 0.3);
        assert!(matches!(
            detection.recommendation,
            GamingRecommendation::Watch | GamingRecommendation::Review
        ));
    }
}
