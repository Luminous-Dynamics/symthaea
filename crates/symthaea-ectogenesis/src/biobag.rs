// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BioBag amniotic fluid management.
//!
//! Inspired by the CHOP (Children's Hospital of Philadelphia) BioBag system,
//! this module models the closed amniotic fluid circuit that maintains
//! temperature, pH, volume, and electrolyte balance throughout gestation.

use serde::{Deserialize, Serialize};

use crate::types::{AmnioticFluid, GestationalWeek};

/// Amniotic fluid composition controller (CHOP BioBag inspired).
///
/// Manages target fluid parameters that change throughout gestation
/// and provides quality scoring for current conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioBag {
    /// Current fluid state.
    pub fluid: AmnioticFluid,
    /// Target fluid state for current gestational week.
    pub target_fluid: AmnioticFluid,
}

impl BioBag {
    /// Create a new BioBag configured for the given gestational week.
    pub fn new(initial_week: GestationalWeek) -> Self {
        let target = Self::target_fluid_for_week(initial_week);
        Self {
            fluid: target.clone(),
            target_fluid: target,
        }
    }

    /// Compute target amniotic fluid volume (mL) by gestational week.
    ///
    /// Volume rises from ~30 mL at week 10 to ~800 mL at week 32-36,
    /// then declines slightly toward term.
    pub fn target_volume(week: GestationalWeek) -> f64 {
        let w = week.0 as f64;
        if w < 8.0 {
            return 10.0 + w * 2.5;
        }
        if w <= 33.0 {
            // Rising phase
            30.0 + (w - 8.0) * 30.0
        } else {
            // Plateau and slight decline
            let peak = 30.0 + 25.0 * 30.0; // ~780 mL
            peak - (w - 33.0) * 20.0
        }
    }

    /// Compute target fluid parameters for a given week.
    fn target_fluid_for_week(week: GestationalWeek) -> AmnioticFluid {
        AmnioticFluid {
            temperature_celsius: 37.0,
            ph: 7.1 + (week.0 as f64 * 0.003), // Slight rise with gestation
            volume_ml: Self::target_volume(week),
            electrolyte_balance: 1.0,
            protein_concentration: 200.0 + (week.0 as f64 * 5.0),
        }
    }

    /// Adjust the BioBag targets for the current gestational week.
    pub fn adjust(&mut self, week: GestationalWeek) {
        self.target_fluid = Self::target_fluid_for_week(week);

        // Fluid gradually approaches target (simple exponential smoothing)
        let alpha = 0.3;
        self.fluid.temperature_celsius +=
            alpha * (self.target_fluid.temperature_celsius - self.fluid.temperature_celsius);
        self.fluid.ph += alpha * (self.target_fluid.ph - self.fluid.ph);
        self.fluid.volume_ml += alpha * (self.target_fluid.volume_ml - self.fluid.volume_ml);
        self.fluid.electrolyte_balance +=
            alpha * (self.target_fluid.electrolyte_balance - self.fluid.electrolyte_balance);
        self.fluid.protein_concentration +=
            alpha * (self.target_fluid.protein_concentration - self.fluid.protein_concentration);
    }

    /// Compute a quality score (0-1) for the current fluid state.
    ///
    /// 1.0 = perfect match to targets, 0.0 = critical deviation.
    pub fn quality_score(&self) -> f64 {
        let temp_err =
            (self.fluid.temperature_celsius - self.target_fluid.temperature_celsius).abs() / 2.0;
        let ph_err = (self.fluid.ph - self.target_fluid.ph).abs() / 0.5;
        let vol_err = if self.target_fluid.volume_ml > 0.0 {
            (self.fluid.volume_ml - self.target_fluid.volume_ml).abs() / self.target_fluid.volume_ml
        } else {
            0.0
        };
        let elec_err =
            (self.fluid.electrolyte_balance - self.target_fluid.electrolyte_balance).abs();

        let total_err = (temp_err + ph_err + vol_err + elec_err) / 4.0;
        (1.0 - total_err).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biobag_new() {
        let bag = BioBag::new(GestationalWeek::new(10));
        assert!((bag.fluid.temperature_celsius - 37.0).abs() < 0.01);
        assert!(bag.fluid.volume_ml > 0.0);
    }

    #[test]
    fn test_target_volume_rises_then_declines() {
        let v10 = BioBag::target_volume(GestationalWeek::new(10));
        let v20 = BioBag::target_volume(GestationalWeek::new(20));
        let v32 = BioBag::target_volume(GestationalWeek::new(32));
        let v40 = BioBag::target_volume(GestationalWeek::new(40));

        assert!(v10 < v20, "Volume should rise: v10={v10}, v20={v20}");
        assert!(v20 < v32, "Volume should rise: v20={v20}, v32={v32}");
        assert!(
            v40 < v32,
            "Volume should decline after peak: v40={v40}, v32={v32}"
        );
    }

    #[test]
    fn test_quality_score_perfect() {
        let bag = BioBag::new(GestationalWeek::new(20));
        let score = bag.quality_score();
        assert!(
            (score - 1.0).abs() < 0.01,
            "New bag should have perfect quality, got {score}"
        );
    }

    #[test]
    fn test_quality_score_degrades_with_temp_deviation() {
        let mut bag = BioBag::new(GestationalWeek::new(20));
        bag.fluid.temperature_celsius = 39.0; // Fever-level
        let score = bag.quality_score();
        assert!(
            score < 0.8,
            "Temperature deviation should lower quality, got {score}"
        );
    }

    #[test]
    fn test_adjust_converges() {
        let mut bag = BioBag::new(GestationalWeek::new(10));
        bag.fluid.temperature_celsius = 35.0; // Too cold

        // Adjust multiple times
        for _ in 0..20 {
            bag.adjust(GestationalWeek::new(10));
        }

        assert!(
            (bag.fluid.temperature_celsius - 37.0).abs() < 0.1,
            "Should converge to target: {}",
            bag.fluid.temperature_celsius
        );
    }

    #[test]
    fn test_biobag_serde_roundtrip() {
        let bag = BioBag::new(GestationalWeek::new(25));
        let json = serde_json::to_string(&bag).unwrap();
        let deser: BioBag = serde_json::from_str(&json).unwrap();
        assert!((deser.fluid.temperature_celsius - bag.fluid.temperature_celsius).abs() < 1e-9);
    }
}
