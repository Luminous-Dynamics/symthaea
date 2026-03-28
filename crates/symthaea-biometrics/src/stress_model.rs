// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Player stress model: allostatic load accumulation and recovery.
//!
//! Adapted from two validated sources:
//! - `symthaea-psych-bench/allostatic_stress.rs` (McEwen 1998 cortisol model)
//! - `mycelix-multiworld-sim/needs.rs` (thermodynamic vice model)
//!
//! The key dynamics:
//! - Sustained arousal > 0.6 → cortisol proxy rises
//! - Cortisol > 0.4 → allostatic load accumulates
//! - Arousal < 0.3 → natural decay (recovery)
//! - Load > 0.8 → burnout: DA and 5-HT baselines depress

use crate::input_telemetry::StressVector;

// ─── Constants (McEwen 1998, adapted for real-time) ─────────────────────────

/// Cortisol accumulation rate when arousal > threshold.
/// At arousal=0.9 (excess=0.3): +0.06 * 0.3 = +0.018/s.
const CORTISOL_ACCUMULATION_RATE: f32 = 0.06;

/// Arousal threshold for cortisol production.
const CORTISOL_AROUSAL_THRESHOLD: f32 = 0.6;

/// Cortisol natural decay rate per second.
const CORTISOL_DECAY_RATE: f32 = 0.005;

/// Allostatic load accumulation rate when cortisol > threshold.
/// At cortisol=0.6 (excess=0.2): +0.02 * 0.2 = +0.004/s.
const ALLOSTATIC_ACCUMULATION_RATE: f32 = 0.02;

/// Cortisol threshold for allostatic accumulation.
const ALLOSTATIC_CORTISOL_THRESHOLD: f32 = 0.4;

/// Allostatic load natural decay rate per second (when arousal is low).
const ALLOSTATIC_DECAY_RATE: f32 = 0.002;

/// Arousal threshold below which recovery occurs.
const RECOVERY_AROUSAL_THRESHOLD: f32 = 0.3;

/// Allostatic load threshold for burnout regime.
const BURNOUT_THRESHOLD: f32 = 0.8;

/// Baseline depression rate under burnout (per second).
const BURNOUT_DEPRESSION_RATE: f32 = 0.001;

/// Floor for depressed baselines under burnout.
const BURNOUT_BASELINE_FLOOR: f32 = 0.35;

/// Consecutive low-stress seconds needed to start active recovery.
const RECOVERY_ONSET_SECS: f32 = 5.0;

/// Active recovery rate (faster than natural decay).
const ACTIVE_RECOVERY_RATE: f32 = 0.006;

// ─── Model ──────────────────────────────────────────────────────────────────

/// Tracks the player's cumulative physiological stress state.
///
/// Models the HPA axis (hypothalamic-pituitary-adrenal) stress response:
/// sustained arousal → cortisol → allostatic load → burnout.
///
/// # Reference
/// McEwen, B.S. (1998). Protective and damaging effects of stress mediators.
/// *New England Journal of Medicine*, 338(3), 171-179.
#[derive(Debug, Clone)]
pub struct PlayerStressModel {
    /// Cumulative allostatic load [0, 1].
    pub allostatic_load: f32,
    /// Cortisol proxy [0, 1] — driven by sustained arousal.
    pub cortisol_proxy: f32,
    /// Dopamine baseline [0, 1] — depresses under chronic stress.
    pub da_baseline: f32,
    /// Serotonin baseline [0, 1] — depresses under chronic stress.
    pub sht_baseline: f32,
    /// Consecutive seconds of low arousal (for recovery onset).
    pub calm_duration_secs: f32,
}

impl PlayerStressModel {
    /// Create a new model in resting state.
    pub fn new() -> Self {
        Self {
            allostatic_load: 0.0,
            cortisol_proxy: 0.0,
            da_baseline: 0.5,
            sht_baseline: 0.5,
            calm_duration_secs: 0.0,
        }
    }

    /// Advance the stress model by one time step.
    ///
    /// Call this at a regular cadence (e.g., 10-60 Hz) with the current
    /// stress vector and delta time in seconds.
    pub fn tick(&mut self, stress: &StressVector, dt_secs: f32) {
        // Cortisol: rises with sustained high arousal
        if stress.arousal > CORTISOL_AROUSAL_THRESHOLD {
            self.cortisol_proxy += CORTISOL_ACCUMULATION_RATE
                * (stress.arousal - CORTISOL_AROUSAL_THRESHOLD)
                * dt_secs;
        }
        // Natural cortisol decay
        self.cortisol_proxy -= CORTISOL_DECAY_RATE * dt_secs;
        self.cortisol_proxy = self.cortisol_proxy.clamp(0.0, 1.0);

        // Allostatic load: accumulates when cortisol exceeds threshold
        if self.cortisol_proxy > ALLOSTATIC_CORTISOL_THRESHOLD {
            self.allostatic_load += ALLOSTATIC_ACCUMULATION_RATE
                * (self.cortisol_proxy - ALLOSTATIC_CORTISOL_THRESHOLD)
                * dt_secs;
        }

        // Recovery tracking
        if stress.arousal < RECOVERY_AROUSAL_THRESHOLD {
            self.calm_duration_secs += dt_secs;

            // Natural decay
            self.allostatic_load -= ALLOSTATIC_DECAY_RATE * dt_secs;

            // Active recovery after sustained calm
            if self.calm_duration_secs > RECOVERY_ONSET_SECS {
                self.allostatic_load -= ACTIVE_RECOVERY_RATE * dt_secs;
            }
        } else {
            self.calm_duration_secs = 0.0;
        }
        self.allostatic_load = self.allostatic_load.clamp(0.0, 1.0);

        // Burnout regime: chronic high load depresses monoamine baselines
        if self.allostatic_load > BURNOUT_THRESHOLD {
            if self.da_baseline > BURNOUT_BASELINE_FLOOR {
                self.da_baseline -= BURNOUT_DEPRESSION_RATE * dt_secs;
                self.da_baseline = self.da_baseline.max(BURNOUT_BASELINE_FLOOR);
            }
            if self.sht_baseline > BURNOUT_BASELINE_FLOOR {
                self.sht_baseline -= BURNOUT_DEPRESSION_RATE * dt_secs;
                self.sht_baseline = self.sht_baseline.max(BURNOUT_BASELINE_FLOOR);
            }
        } else {
            // Gradual baseline recovery when not in burnout
            if self.da_baseline < 0.5 {
                self.da_baseline += BURNOUT_DEPRESSION_RATE * 0.5 * dt_secs;
                self.da_baseline = self.da_baseline.min(0.5);
            }
            if self.sht_baseline < 0.5 {
                self.sht_baseline += BURNOUT_DEPRESSION_RATE * 0.5 * dt_secs;
                self.sht_baseline = self.sht_baseline.min(0.5);
            }
        }
    }

    /// Whether the player is in burnout regime (allostatic load > 0.8).
    pub fn is_burnout(&self) -> bool {
        self.allostatic_load > BURNOUT_THRESHOLD
    }

    /// Reset to resting state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for PlayerStressModel {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_model_at_rest() {
        let model = PlayerStressModel::new();
        assert_eq!(model.allostatic_load, 0.0);
        assert_eq!(model.cortisol_proxy, 0.0);
        assert!(!model.is_burnout());
    }

    #[test]
    fn high_arousal_accumulates_cortisol() {
        let mut model = PlayerStressModel::new();
        let stress = StressVector {
            arousal: 0.9,
            valence: -0.5,
            dominance: 0.3,
        };
        // 30 seconds of high arousal at 10Hz
        for _ in 0..300 {
            model.tick(&stress, 0.1);
        }
        assert!(
            model.cortisol_proxy > 0.05,
            "cortisol should rise with sustained arousal, got {}",
            model.cortisol_proxy
        );
    }

    #[test]
    fn sustained_stress_accumulates_allostatic_load() {
        let mut model = PlayerStressModel::new();
        let stress = StressVector {
            arousal: 0.9,
            valence: -0.8,
            dominance: 0.1,
        };
        // 120 seconds of extreme stress at 10Hz
        for _ in 0..1200 {
            model.tick(&stress, 0.1);
        }
        assert!(
            model.allostatic_load > 0.01,
            "allostatic load should accumulate, got {}",
            model.allostatic_load
        );
    }

    #[test]
    fn calm_causes_recovery() {
        let mut model = PlayerStressModel::new();
        // First: stress up for 120 seconds
        let high = StressVector {
            arousal: 0.9,
            valence: -0.5,
            dominance: 0.3,
        };
        for _ in 0..1200 {
            model.tick(&high, 0.1);
        }
        let peak_load = model.allostatic_load;

        // Then: calm down for 120 seconds (cortisol must decay below threshold first)
        let low = StressVector {
            arousal: 0.1,
            valence: 0.3,
            dominance: 0.2,
        };
        for _ in 0..1200 {
            model.tick(&low, 0.1);
        }
        assert!(
            model.allostatic_load < peak_load,
            "load should decrease during calm: peak={}, now={}",
            peak_load,
            model.allostatic_load
        );
    }

    #[test]
    fn burnout_depresses_baselines() {
        let mut model = PlayerStressModel::new();
        // Force into burnout
        model.allostatic_load = 0.9;
        let stress = StressVector {
            arousal: 0.8,
            valence: -0.7,
            dominance: 0.0,
        };
        for _ in 0..100 {
            model.tick(&stress, 0.1);
        }
        assert!(model.is_burnout());
        assert!(
            model.da_baseline < 0.5,
            "DA baseline should depress under burnout, got {}",
            model.da_baseline
        );
        assert!(
            model.sht_baseline < 0.5,
            "5-HT baseline should depress under burnout, got {}",
            model.sht_baseline
        );
    }

    #[test]
    fn allostatic_load_bounded() {
        let mut model = PlayerStressModel::new();
        let extreme = StressVector {
            arousal: 1.0,
            valence: -1.0,
            dominance: 0.0,
        };
        // Extreme stress for a very long time
        for _ in 0..10000 {
            model.tick(&extreme, 0.1);
        }
        assert!(model.allostatic_load <= 1.0);
        assert!(model.cortisol_proxy <= 1.0);
        assert!(model.da_baseline >= BURNOUT_BASELINE_FLOOR);
        assert!(model.sht_baseline >= BURNOUT_BASELINE_FLOOR);
    }

    #[test]
    fn zero_arousal_no_accumulation() {
        let mut model = PlayerStressModel::new();
        let calm = StressVector {
            arousal: 0.0,
            valence: 0.5,
            dominance: 0.3,
        };
        for _ in 0..1000 {
            model.tick(&calm, 0.1);
        }
        assert_eq!(model.allostatic_load, 0.0);
        assert_eq!(model.cortisol_proxy, 0.0);
    }
}
