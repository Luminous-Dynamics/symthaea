// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime threshold overrides for cognitive loop parameters.
//!
//! Allows evolved threshold parameters (from `ThresholdPhenotype`) to be
//! applied at runtime without modifying the `pub const` defaults. Each field
//! is `Option<T>` — `None` means use the compile-time default from `thresholds/`.
//!
//! Usage: `service.threshold_overrides.apply_from_phenotype(&phenotype);`
//! Then in hot loops: `let scale = overrides.fep_surprise_scale.unwrap_or(FEP_SURPRISE_SCALE);`

use serde::{Deserialize, Serialize};

/// Runtime threshold overrides. All `None` = use compile-time defaults.
/// Populated from `ThresholdPhenotype` via `apply_from_phenotype()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThresholdOverrides {
    // Learning
    pub fep_surprise_scale: Option<f32>,
    pub fep_lr_decay: Option<f32>,

    // Consciousness
    pub dream_base_interval: Option<u64>,
    pub dream_min_interval: Option<u64>,

    // Neuromodulation
    pub neuromod_d2_baseline: Option<f64>,
    pub neuromod_ne_phasic_threshold: Option<f32>,
    pub neuromod_arousal_ema_decay: Option<f32>,
    pub homeostasis_recalibrate_high: Option<f32>,
    pub homeostasis_recalibrate_low: Option<f32>,
    pub neuromod_ema_alpha: Option<f32>,

    // Drives
    pub frustration_dampen_threshold: Option<f64>,
    pub engagement_low_threshold: Option<f64>,
    pub flow_exploration_increment: Option<f32>,
    pub coherence_low: Option<f32>,

    // Feedback
    pub arousal_trap_threshold: Option<f32>,
    pub self_model_weight_high: Option<f32>,
    pub homeostasis_pull_cruise: Option<f32>,
    pub confidence_crash_threshold: Option<f64>,
}

impl ThresholdOverrides {
    /// Apply all thresholds from an evolved phenotype.
    #[cfg(feature = "neuroevolution")]
    pub fn apply_from_phenotype(&mut self, p: &symthaea_neuroevolution::ThresholdPhenotype) {
        self.fep_surprise_scale = Some(p.fep_surprise_scale);
        self.fep_lr_decay = Some(p.fep_lr_decay);
        self.dream_base_interval = Some(p.dream_base_interval);
        self.dream_min_interval = Some(p.dream_min_interval);
        self.neuromod_d2_baseline = Some(p.neuromod_d2_baseline);
        self.neuromod_ne_phasic_threshold = Some(p.neuromod_ne_phasic_threshold);
        self.neuromod_arousal_ema_decay = Some(p.neuromod_arousal_ema_decay);
        self.homeostasis_recalibrate_high = Some(p.homeostasis_recalibrate_high);
        self.homeostasis_recalibrate_low = Some(p.homeostasis_recalibrate_low);
        self.neuromod_ema_alpha = Some(p.neuromod_ema_alpha);
        self.frustration_dampen_threshold = Some(p.frustration_dampen_threshold);
        self.engagement_low_threshold = Some(p.engagement_low_threshold);
        self.flow_exploration_increment = Some(p.flow_exploration_increment);
        self.coherence_low = Some(p.coherence_low);
        self.arousal_trap_threshold = Some(p.arousal_trap_threshold);
        self.self_model_weight_high = Some(p.self_model_weight_high);
        self.homeostasis_pull_cruise = Some(p.homeostasis_pull_cruise);
        self.confidence_crash_threshold = Some(p.confidence_crash_threshold);
    }

    /// Apply blended thresholds from governance gate result.
    #[cfg(feature = "neuroevolution")]
    pub fn apply_blended(&mut self, blended: &symthaea_neuroevolution::ThresholdPhenotype) {
        self.apply_from_phenotype(blended);
    }

    /// Clear all overrides (revert to compile-time defaults).
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Count how many thresholds are overridden.
    pub fn active_count(&self) -> usize {
        let mut n = 0;
        if self.fep_surprise_scale.is_some() {
            n += 1;
        }
        if self.fep_lr_decay.is_some() {
            n += 1;
        }
        if self.dream_base_interval.is_some() {
            n += 1;
        }
        if self.dream_min_interval.is_some() {
            n += 1;
        }
        if self.neuromod_d2_baseline.is_some() {
            n += 1;
        }
        if self.neuromod_ne_phasic_threshold.is_some() {
            n += 1;
        }
        if self.neuromod_arousal_ema_decay.is_some() {
            n += 1;
        }
        if self.homeostasis_recalibrate_high.is_some() {
            n += 1;
        }
        if self.homeostasis_recalibrate_low.is_some() {
            n += 1;
        }
        if self.neuromod_ema_alpha.is_some() {
            n += 1;
        }
        if self.frustration_dampen_threshold.is_some() {
            n += 1;
        }
        if self.engagement_low_threshold.is_some() {
            n += 1;
        }
        if self.flow_exploration_increment.is_some() {
            n += 1;
        }
        if self.coherence_low.is_some() {
            n += 1;
        }
        if self.arousal_trap_threshold.is_some() {
            n += 1;
        }
        if self.self_model_weight_high.is_some() {
            n += 1;
        }
        if self.homeostasis_pull_cruise.is_some() {
            n += 1;
        }
        if self.confidence_crash_threshold.is_some() {
            n += 1;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_all_none() {
        let o = ThresholdOverrides::default();
        assert_eq!(o.active_count(), 0);
    }

    #[test]
    fn test_clear_resets() {
        let mut o = ThresholdOverrides::default();
        o.fep_surprise_scale = Some(5.0);
        o.dream_base_interval = Some(200);
        assert_eq!(o.active_count(), 2);
        o.clear();
        assert_eq!(o.active_count(), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE ACCESSORS (use compile-time defaults as fallback)
// ═══════════════════════════════════════════════════════════════════════════════

impl ThresholdOverrides {
    /// FEP surprise → LR boost scale. Fallback: `thresholds::FEP_SURPRISE_SCALE`.
    pub fn fep_surprise_scale(&self) -> f32 {
        self.fep_surprise_scale
            .unwrap_or(super::thresholds::FEP_SURPRISE_SCALE)
    }
    /// FEP per-cycle LR decay. Fallback: `thresholds::FEP_LR_DECAY`.
    pub fn fep_lr_decay(&self) -> f32 {
        self.fep_lr_decay.unwrap_or(super::thresholds::FEP_LR_DECAY)
    }
    /// Dream consolidation base interval. Fallback: `thresholds::DREAM_BASE_INTERVAL`.
    pub fn dream_base_interval(&self) -> u64 {
        self.dream_base_interval
            .unwrap_or(super::thresholds::DREAM_BASE_INTERVAL)
    }
    /// Dream minimum interval. Fallback: `thresholds::DREAM_MIN_INTERVAL`.
    pub fn dream_min_interval(&self) -> u64 {
        self.dream_min_interval
            .unwrap_or(super::thresholds::DREAM_MIN_INTERVAL)
    }
    /// D2 flexibility baseline. Fallback: `thresholds::NEUROMOD_D2_FLEXIBILITY_BASELINE`.
    pub fn neuromod_d2_baseline(&self) -> f64 {
        self.neuromod_d2_baseline
            .unwrap_or(super::thresholds::NEUROMOD_D2_FLEXIBILITY_BASELINE)
    }
    /// NE phasic burst threshold. Fallback: `thresholds::NEUROMOD_NE_PHASIC_THRESHOLD`.
    pub fn neuromod_ne_phasic_threshold(&self) -> f32 {
        self.neuromod_ne_phasic_threshold
            .unwrap_or(super::thresholds::NEUROMOD_NE_PHASIC_THRESHOLD)
    }
    /// Arousal EMA decay. Fallback: `thresholds::NEUROMOD_AROUSAL_EMA_DECAY`.
    pub fn neuromod_arousal_ema_decay(&self) -> f32 {
        self.neuromod_arousal_ema_decay
            .unwrap_or(super::thresholds::NEUROMOD_AROUSAL_EMA_DECAY)
    }
    /// Homeostasis recalibration ceiling. Fallback: `thresholds::HOMEOSTASIS_RECALIBRATE_HIGH`.
    pub fn homeostasis_recalibrate_high(&self) -> f32 {
        self.homeostasis_recalibrate_high
            .unwrap_or(super::thresholds::HOMEOSTASIS_RECALIBRATE_HIGH)
    }
    /// Homeostasis recalibration floor. Fallback: `thresholds::HOMEOSTASIS_RECALIBRATE_LOW`.
    pub fn homeostasis_recalibrate_low(&self) -> f32 {
        self.homeostasis_recalibrate_low
            .unwrap_or(super::thresholds::HOMEOSTASIS_RECALIBRATE_LOW)
    }
    /// Neuromodulator EMA alpha. Fallback: `thresholds::NEUROMOD_EMA_ALPHA`.
    pub fn neuromod_ema_alpha(&self) -> f32 {
        self.neuromod_ema_alpha
            .unwrap_or(super::thresholds::NEUROMOD_EMA_ALPHA)
    }
    /// Frustration dampening threshold. Fallback: `thresholds::FRUSTRATION_DAMPEN_THRESHOLD`.
    pub fn frustration_dampen_threshold(&self) -> f64 {
        self.frustration_dampen_threshold
            .unwrap_or(super::thresholds::FRUSTRATION_DAMPEN_THRESHOLD)
    }
    /// Low engagement threshold. Fallback: `thresholds::ENGAGEMENT_LOW_THRESHOLD`.
    pub fn engagement_low_threshold(&self) -> f64 {
        self.engagement_low_threshold
            .unwrap_or(super::thresholds::ENGAGEMENT_LOW_THRESHOLD)
    }
    /// Flow exploration increment. Fallback: `thresholds::FLOW_EXPLORATION_INCREMENT`.
    pub fn flow_exploration_increment(&self) -> f32 {
        self.flow_exploration_increment
            .unwrap_or(super::thresholds::FLOW_EXPLORATION_INCREMENT)
    }
    /// Coherence low boundary. Fallback: `thresholds::COHERENCE_LOW`.
    pub fn coherence_low(&self) -> f32 {
        self.coherence_low
            .unwrap_or(super::thresholds::COHERENCE_LOW)
    }
    /// Arousal trap threshold. Fallback: `thresholds::AROUSAL_TRAP_DETECT_THRESHOLD`.
    pub fn arousal_trap_threshold(&self) -> f32 {
        self.arousal_trap_threshold
            .unwrap_or(super::thresholds::AROUSAL_TRAP_DETECT_THRESHOLD)
    }
    /// Self-model weight high threshold. Fallback: `thresholds::SELF_MODEL_WEIGHT_HIGH_THRESHOLD`.
    pub fn self_model_weight_high(&self) -> f32 {
        self.self_model_weight_high
            .unwrap_or(super::thresholds::SELF_MODEL_WEIGHT_HIGH_THRESHOLD)
    }
    /// Homeostasis pull in cruise. Fallback: `thresholds::HOMEOSTASIS_PULL_CRUISE`.
    pub fn homeostasis_pull_cruise(&self) -> f32 {
        self.homeostasis_pull_cruise
            .unwrap_or(super::thresholds::HOMEOSTASIS_PULL_CRUISE)
    }
    /// Confidence crash threshold. Fallback: `thresholds::CONFIDENCE_CRASH_THRESHOLD`.
    pub fn confidence_crash_threshold(&self) -> f64 {
        self.confidence_crash_threshold
            .unwrap_or(super::thresholds::CONFIDENCE_CRASH_THRESHOLD)
    }
}
