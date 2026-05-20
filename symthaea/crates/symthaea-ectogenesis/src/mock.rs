// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mock womb for testing and simulation.
//!
//! Provides a configurable mock that can advance week-by-week through
//! gestation, optionally injecting complications (growth restriction,
//! tachycardia, etc.) at a configurable rate.

use rand::Rng;

use crate::types::{FetalMetrics, GestationalWeek};

/// Mock artificial womb for simulation and testing.
#[derive(Debug, Clone)]
pub struct MockWomb {
    /// Current gestational week.
    pub week: GestationalWeek,
    /// Current fetal metrics.
    pub metrics: FetalMetrics,
    /// Whether random complications are enabled.
    pub complications_enabled: bool,
    /// Probability of a complication per week (0-1).
    pub complication_rate: f64,
}

impl MockWomb {
    /// Create a healthy mock womb starting at a given week.
    pub fn healthy_at(week: u32) -> Self {
        let gw = GestationalWeek::new(week);
        Self {
            week: gw,
            metrics: FetalMetrics::normative(gw),
            complications_enabled: false,
            complication_rate: 0.0,
        }
    }

    /// Create a healthy mock womb starting at week 0.
    pub fn healthy() -> Self {
        Self::healthy_at(0)
    }

    /// Create a mock womb with complications enabled.
    pub fn with_complications(rate: f64) -> Self {
        Self {
            complications_enabled: true,
            complication_rate: rate.clamp(0.0, 1.0),
            ..Self::healthy()
        }
    }

    /// Advance the womb by one week, updating metrics.
    ///
    /// If complications are enabled, may randomly perturb metrics.
    pub fn advance_week(&mut self, rng: &mut impl Rng) {
        let new_week = GestationalWeek::new(self.week.0 + 1);
        self.week = new_week;
        self.metrics = FetalMetrics::normative(new_week);

        if self.complications_enabled && rng.r#gen::<f64>() < self.complication_rate {
            self.inject_complication(rng);
        }
    }

    /// Run the simulation to term (week 40), collecting metrics at each week.
    pub fn run_to_term(&mut self, rng: &mut impl Rng) -> Vec<FetalMetrics> {
        let mut history = Vec::new();
        history.push(self.metrics.clone());

        while self.week.0 < 40 {
            self.advance_week(rng);
            history.push(self.metrics.clone());
        }

        history
    }

    /// Inject a random complication into the current metrics.
    fn inject_complication(&mut self, rng: &mut impl Rng) {
        let complication_type = rng.gen_range(0..4);
        match complication_type {
            0 => {
                // Growth restriction
                self.metrics.weight_grams *= 0.7;
            }
            1 => {
                // Tachycardia
                self.metrics.heart_rate_bpm *= 1.3;
            }
            2 => {
                // Reduced movement
                self.metrics.movement_score *= 0.3;
            }
            _ => {
                // Delayed lung maturity
                self.metrics.lung_maturity *= 0.5;
            }
        }
    }
}

impl Default for MockWomb {
    fn default() -> Self {
        Self::healthy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    type TestRng = rand::rngs::StdRng;

    fn test_rng() -> TestRng {
        TestRng::seed_from_u64(42)
    }

    #[test]
    fn test_healthy_womb_starts_at_zero() {
        let womb = MockWomb::healthy();
        assert_eq!(womb.week.0, 0);
        assert!(!womb.complications_enabled);
    }

    #[test]
    fn test_healthy_at_specific_week() {
        let womb = MockWomb::healthy_at(20);
        assert_eq!(womb.week.0, 20);
        assert!(womb.metrics.weight_grams > 10.0);
    }

    #[test]
    fn test_advance_week() {
        let mut womb = MockWomb::healthy_at(10);
        let mut rng = test_rng();
        womb.advance_week(&mut rng);
        assert_eq!(womb.week.0, 11);
    }

    #[test]
    fn test_advance_clamped_at_40() {
        let mut womb = MockWomb::healthy_at(39);
        let mut rng = test_rng();
        womb.advance_week(&mut rng);
        assert_eq!(womb.week.0, 40);
        // One more should stay at 40
        womb.advance_week(&mut rng);
        assert_eq!(womb.week.0, 40);
    }

    #[test]
    fn test_run_to_term() {
        let mut womb = MockWomb::healthy();
        let mut rng = test_rng();
        let history = womb.run_to_term(&mut rng);
        // Should have 41 entries (weeks 0-40 inclusive)
        assert_eq!(history.len(), 41);
        assert_eq!(history.last().unwrap().week.0, 40);
    }

    #[test]
    fn test_run_to_term_weight_increases() {
        let mut womb = MockWomb::healthy();
        let mut rng = test_rng();
        let history = womb.run_to_term(&mut rng);
        // Weight should generally increase (normative trajectory)
        let early_weight = history[10].weight_grams;
        let late_weight = history[35].weight_grams;
        assert!(
            late_weight > early_weight,
            "Weight should increase over gestation"
        );
    }

    #[test]
    fn test_complications_alter_metrics() {
        let mut rng = test_rng();
        let mut womb = MockWomb::with_complications(1.0); // 100% complication rate
        womb.week = GestationalWeek::new(20);
        womb.metrics = FetalMetrics::normative(GestationalWeek::new(20));
        let normal_weight = womb.metrics.weight_grams;
        let normal_hr = womb.metrics.heart_rate_bpm;
        let normal_movement = womb.metrics.movement_score;
        let normal_lung = womb.metrics.lung_maturity;

        // Advance many times and check that at least one metric was perturbed
        let mut any_changed = false;
        for _ in 0..20 {
            womb.week = GestationalWeek::new(20);
            womb.metrics = FetalMetrics::normative(GestationalWeek::new(20));
            womb.advance_week(&mut rng);
            // The metrics get reset to normative at week 21 then complication applied
            let m21_normative = FetalMetrics::normative(GestationalWeek::new(21));
            if (womb.metrics.weight_grams - m21_normative.weight_grams).abs() > 0.1
                || (womb.metrics.heart_rate_bpm - m21_normative.heart_rate_bpm).abs() > 0.1
                || (womb.metrics.movement_score - m21_normative.movement_score).abs() > 0.01
                || (womb.metrics.lung_maturity - m21_normative.lung_maturity).abs() > 0.01
            {
                any_changed = true;
                break;
            }
        }
        assert!(
            any_changed,
            "Complications should alter at least one metric"
        );
        // Suppress unused variable warnings
        let _ = (normal_weight, normal_hr, normal_movement, normal_lung);
    }
}
