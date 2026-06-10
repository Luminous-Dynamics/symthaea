// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decomposition prediction for waste materials using sigmoid-based decay models.
//!
//! Predicts material degradation over time with O(1) computation per query.
//! Temperature follows the Q10 rule (rate doubles per 10 C), moisture follows a
//! Gaussian optimum centered at 55%, and oxygen modulates aerobic decomposition.

use crate::waste_encoder::WasteCategory;
use serde::{Deserialize, Serialize};

/// Environmental conditions affecting decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConditions {
    /// Temperature in degrees Celsius.
    pub temperature_c: f64,
    /// Moisture percentage (0-100).
    pub moisture_pct: f64,
    /// Oxygen percentage (0-21 typical).
    pub oxygen_pct: f64,
    /// Initial carbon-to-nitrogen ratio (typical 20-40 for composting).
    pub initial_cn_ratio: f64,
}

/// Result of a decomposition prediction at a specific time point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionPrediction {
    /// Number of days elapsed since start.
    pub days_elapsed: f64,
    /// Percentage of material decomposed (0-100).
    pub decomposition_pct: f64,
    /// Remaining structural strength as fraction (0-1).
    pub remaining_strength: f64,
}

/// Predicts decomposition timelines for waste materials.
///
/// Uses a sigmoid decay model where the rate constant depends on material type
/// and environmental conditions. Each prediction is O(1) — just evaluates the
/// sigmoid at the requested time point.
pub struct DecompositionPredictor;

impl DecompositionPredictor {
    /// Create a new predictor.
    pub fn new() -> Self {
        Self
    }

    /// Predict decomposition state at `days` elapsed for the given material and conditions.
    pub fn predict(
        &self,
        category: WasteCategory,
        conditions: &DecompositionConditions,
        days: f64,
    ) -> DecompositionPrediction {
        let t_half_base = Self::base_half_life(category);
        let rate_modifier = Self::environmental_rate_modifier(conditions);

        // Effective half-life adjusted by environment
        let t_half = if rate_modifier > 0.0 {
            t_half_base / rate_modifier
        } else {
            f64::MAX
        };

        // Sigmoid steepness parameter: k = 4 / t_half gives ~95% decomposition at 2*t_half
        let k = if t_half < f64::MAX && t_half > 0.0 {
            4.0 / t_half
        } else {
            0.0
        };

        // Sigmoid: decomposition_pct = 100 / (1 + exp(-k * (days - t_half)))
        let exponent = -k * (days - t_half);
        let decomposition_pct = if exponent > 500.0 {
            // Prevent overflow — nearly 0% decomposed
            0.0
        } else if exponent < -500.0 {
            // Nearly 100% decomposed
            100.0
        } else {
            100.0 / (1.0 + exponent.exp())
        };

        let decomposition_pct = decomposition_pct.clamp(0.0, 100.0);
        let remaining_strength = 1.0 - decomposition_pct / 100.0;

        DecompositionPrediction {
            days_elapsed: days,
            decomposition_pct,
            remaining_strength,
        }
    }

    /// Base half-life in days for each waste category (time to 50% decomposition
    /// under reference conditions: 25 C, 55% moisture, 21% O2).
    fn base_half_life(category: WasteCategory) -> f64 {
        match category {
            WasteCategory::Organic => 30.0,   // food waste
            WasteCategory::Paper => 60.0,     // office paper
            WasteCategory::Cardboard => 90.0, // corrugated
            WasteCategory::Wood => 365.0,     // untreated lumber
            WasteCategory::Textiles => 180.0, // cotton/natural fibers
            WasteCategory::PlasticPET
            | WasteCategory::PlasticHDPE
            | WasteCategory::PlasticPP
            | WasteCategory::PlasticPS
            | WasteCategory::PlasticOther => 36_500.0, // ~100 years
            WasteCategory::Aluminum => 73_000.0, // ~200 years
            WasteCategory::Steel => 18_250.0, // ~50 years (rusting)
            WasteCategory::GlassClear | WasteCategory::GlassColored => 365_000.0, // ~1000 years
            WasteCategory::Rubber => 36_500.0, // ~100 years
            WasteCategory::Electronic => 36_500.0, // ~100 years
            WasteCategory::Hazardous => 18_250.0, // varies widely
            WasteCategory::Mixed => 3_650.0,  // ~10 years average
        }
    }

    /// Environmental rate modifier combining temperature, moisture, and oxygen effects.
    fn environmental_rate_modifier(conditions: &DecompositionConditions) -> f64 {
        let temp_factor = Self::temperature_factor(conditions.temperature_c);
        let moisture_factor = Self::moisture_factor(conditions.moisture_pct);
        let oxygen_factor = Self::oxygen_factor(conditions.oxygen_pct);
        let cn_factor = Self::cn_ratio_factor(conditions.initial_cn_ratio);

        temp_factor * moisture_factor * oxygen_factor * cn_factor
    }

    /// Q10 rule: rate doubles for every 10 C increase above reference (25 C).
    fn temperature_factor(temp_c: f64) -> f64 {
        let factor = 2.0_f64.powf((temp_c - 25.0) / 10.0);
        factor.max(0.01) // floor to avoid zero/negative
    }

    /// Gaussian moisture optimum centered at 55%, sigma = 15%.
    fn moisture_factor(moisture_pct: f64) -> f64 {
        let optimal = 55.0;
        let sigma = 15.0;
        let diff = moisture_pct - optimal;
        let factor = (-0.5 * (diff / sigma).powi(2)).exp();
        factor.max(0.01)
    }

    /// Oxygen factor: aerobic decomposition needs O2, anaerobic is much slower.
    fn oxygen_factor(oxygen_pct: f64) -> f64 {
        if oxygen_pct >= 15.0 {
            1.0
        } else if oxygen_pct > 0.0 {
            0.2 + 0.8 * (oxygen_pct / 15.0) // linear ramp from anaerobic (0.2x) to aerobic (1.0x)
        } else {
            0.2 // anaerobic baseline
        }
    }

    /// C:N ratio factor: optimal composting at 25-30.
    fn cn_ratio_factor(cn_ratio: f64) -> f64 {
        if cn_ratio >= 20.0 && cn_ratio <= 35.0 {
            1.0
        } else if cn_ratio < 20.0 {
            0.7 + 0.3 * (cn_ratio / 20.0).min(1.0)
        } else {
            // High C:N slows decomposition
            0.5 + 0.5 * (35.0 / cn_ratio).min(1.0)
        }
    }
}

impl Default for DecompositionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_conditions() -> DecompositionConditions {
        DecompositionConditions {
            temperature_c: 25.0,
            moisture_pct: 55.0,
            oxygen_pct: 21.0,
            initial_cn_ratio: 30.0,
        }
    }

    #[test]
    fn test_organic_faster_than_plastic() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let organic = predictor.predict(WasteCategory::Organic, &cond, 365.0);
        let plastic = predictor.predict(WasteCategory::PlasticPET, &cond, 365.0);
        assert!(
            organic.decomposition_pct > plastic.decomposition_pct,
            "Organic ({:.1}%) should decompose faster than plastic ({:.1}%) after 1 year",
            organic.decomposition_pct,
            plastic.decomposition_pct
        );
    }

    #[test]
    fn test_higher_temperature_faster() {
        let predictor = DecompositionPredictor::new();
        let cold = DecompositionConditions {
            temperature_c: 10.0,
            ..standard_conditions()
        };
        let hot = DecompositionConditions {
            temperature_c: 40.0,
            ..standard_conditions()
        };
        let dec_cold = predictor.predict(WasteCategory::Organic, &cold, 30.0);
        let dec_hot = predictor.predict(WasteCategory::Organic, &hot, 30.0);
        assert!(
            dec_hot.decomposition_pct > dec_cold.decomposition_pct,
            "Hot ({:.1}%) should decompose faster than cold ({:.1}%)",
            dec_hot.decomposition_pct,
            dec_cold.decomposition_pct
        );
    }

    #[test]
    fn test_optimal_moisture_maximum_rate() {
        let predictor = DecompositionPredictor::new();
        let optimal = standard_conditions(); // 55% moisture
        let dry = DecompositionConditions {
            moisture_pct: 10.0,
            ..standard_conditions()
        };
        let wet = DecompositionConditions {
            moisture_pct: 95.0,
            ..standard_conditions()
        };
        let dec_opt = predictor.predict(WasteCategory::Organic, &optimal, 30.0);
        let dec_dry = predictor.predict(WasteCategory::Organic, &dry, 30.0);
        let dec_wet = predictor.predict(WasteCategory::Organic, &wet, 30.0);
        assert!(
            dec_opt.decomposition_pct >= dec_dry.decomposition_pct,
            "Optimal moisture ({:.1}%) >= dry ({:.1}%)",
            dec_opt.decomposition_pct,
            dec_dry.decomposition_pct
        );
        assert!(
            dec_opt.decomposition_pct >= dec_wet.decomposition_pct,
            "Optimal moisture ({:.1}%) >= wet ({:.1}%)",
            dec_opt.decomposition_pct,
            dec_wet.decomposition_pct
        );
    }

    #[test]
    fn test_decomposition_bounded() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        for &cat in &WasteCategory::ALL {
            for &days in &[0.0, 1.0, 100.0, 10000.0, 1_000_000.0] {
                let pred = predictor.predict(cat, &cond, days);
                assert!(
                    pred.decomposition_pct >= 0.0 && pred.decomposition_pct <= 100.0,
                    "{:?} at {} days: decomposition {:.2}% out of bounds",
                    cat,
                    days,
                    pred.decomposition_pct
                );
            }
        }
    }

    #[test]
    fn test_remaining_strength_complement() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let pred = predictor.predict(WasteCategory::Wood, &cond, 200.0);
        let expected = 1.0 - pred.decomposition_pct / 100.0;
        assert!(
            (pred.remaining_strength - expected).abs() < 1e-10,
            "Remaining strength ({}) should equal 1 - decomposition_pct/100 ({})",
            pred.remaining_strength,
            expected
        );
    }

    #[test]
    fn test_organic_nearly_decomposed_at_1000_days() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let pred = predictor.predict(WasteCategory::Organic, &cond, 1000.0);
        assert!(
            pred.decomposition_pct > 95.0,
            "Organic should be >95% decomposed after 1000 days, got {:.1}%",
            pred.decomposition_pct
        );
    }

    #[test]
    fn test_plastic_barely_decomposes_at_1000_days() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let pred = predictor.predict(WasteCategory::PlasticPET, &cond, 1000.0);
        assert!(
            pred.decomposition_pct < 5.0,
            "Plastic should be <5% decomposed after 1000 days, got {:.1}%",
            pred.decomposition_pct
        );
    }

    #[test]
    fn test_zero_days_minimal_decomposition() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        for &cat in &WasteCategory::ALL {
            let pred = predictor.predict(cat, &cond, 0.0);
            // At t=0, sigmoid gives ~50% at half-life, but for zero days elapsed
            // (which is always < t_half for any material), decomposition should be < 50%
            assert!(
                pred.decomposition_pct < 50.0,
                "{:?} at 0 days should be <50% decomposed, got {:.1}%",
                cat,
                pred.decomposition_pct
            );
        }
    }

    #[test]
    fn test_paper_decomposes_faster_than_wood() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let paper = predictor.predict(WasteCategory::Paper, &cond, 180.0);
        let wood = predictor.predict(WasteCategory::Wood, &cond, 180.0);
        assert!(
            paper.decomposition_pct > wood.decomposition_pct,
            "Paper ({:.1}%) should decompose faster than wood ({:.1}%) at 180 days",
            paper.decomposition_pct,
            wood.decomposition_pct
        );
    }

    #[test]
    fn test_days_elapsed_preserved() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let pred = predictor.predict(WasteCategory::Organic, &cond, 42.5);
        assert_eq!(pred.days_elapsed, 42.5);
    }

    #[test]
    fn test_glass_extremely_slow() {
        let predictor = DecompositionPredictor::new();
        let cond = standard_conditions();
        let pred = predictor.predict(WasteCategory::GlassClear, &cond, 36500.0); // 100 years
        assert!(
            pred.decomposition_pct < 10.0,
            "Glass should be <10% decomposed after 100 years, got {:.1}%",
            pred.decomposition_pct
        );
    }

    #[test]
    fn test_low_oxygen_slows_decomposition() {
        let predictor = DecompositionPredictor::new();
        let aerobic = standard_conditions();
        let anaerobic = DecompositionConditions {
            oxygen_pct: 0.0,
            ..standard_conditions()
        };
        let dec_aero = predictor.predict(WasteCategory::Organic, &aerobic, 30.0);
        let dec_anaero = predictor.predict(WasteCategory::Organic, &anaerobic, 30.0);
        assert!(
            dec_aero.decomposition_pct > dec_anaero.decomposition_pct,
            "Aerobic ({:.1}%) should be faster than anaerobic ({:.1}%)",
            dec_aero.decomposition_pct,
            dec_anaero.decomposition_pct
        );
    }
}
