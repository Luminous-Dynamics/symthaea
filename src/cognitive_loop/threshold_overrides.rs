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

use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use symthaea_types::ThresholdOverrideValues;

/// Runtime threshold overrides. All `None` = use compile-time defaults.
/// Populated from `ThresholdPhenotype` via `apply_from_phenotype()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ThresholdOverrides(pub ThresholdOverrideValues);

impl Deref for ThresholdOverrides {
    type Target = ThresholdOverrideValues;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ThresholdOverrides {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
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
        self.0.clear();
    }

    /// Load threshold overrides from a JSON file. This is the file format
    /// written by the CLS threshold-phenotype promotion path (Tier 1.2,
    /// `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`) -
    /// `scripts/cls_promote_candidate.sh` copies a gated candidate
    /// `ThresholdPhenotype` JSON to the path this reads. `ThresholdPhenotype`
    /// (from `symthaea_neuroevolution`) and `ThresholdOverrides` declare the
    /// same 18 fields with the same names/types (modulo `Option<T>`), so a
    /// `ThresholdPhenotype` JSON deserializes here with every field becoming
    /// `Some(..)` - no conversion step needed. See
    /// `cls_evolution_harness.rs` module docs and
    /// `test_threshold_phenotype_json_loads_as_overrides` below for the
    /// contract this relies on.
    ///
    /// Fail-closed: a missing, unreadable, or unparseable file NEVER panics -
    /// it logs a warning (missing file is silent, since "no promoted
    /// phenotype yet" is the common/default case) and falls back to
    /// `Self::default()`, i.e. compile-time defaults.
    pub fn load_from_file(path: &std::path::Path) -> Self {
        Self(ThresholdOverrideValues::load_from_file(path))
    }

    /// Construct from the `SYMTHAEA_THRESHOLD_OVERRIDES_PATH` env var, if
    /// set. This is the only place production code reads a promoted CLS
    /// threshold phenotype - no live consumer hot-reloads this file; the env
    /// var is read once, at `CognitiveLoopService` construction time. Falls
    /// back to `Self::default()` if the env var is unset/empty, or via
    /// `load_from_file`'s fail-closed behavior if it's set but the file is
    /// missing/invalid. Never panics.
    pub fn from_env() -> Self {
        Self(ThresholdOverrideValues::from_env())
    }

    /// Count how many thresholds are overridden.
    pub fn active_count(&self) -> usize {
        self.0.active_count()
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
    fn test_load_from_file_missing_falls_back_to_default() {
        let path = std::path::Path::new("/nonexistent/path/does-not-exist-12345.json");
        let o = ThresholdOverrides::load_from_file(path);
        assert_eq!(o.active_count(), 0);
    }

    #[test]
    fn test_load_from_file_invalid_json_falls_back_to_default() {
        let dir =
            std::env::temp_dir().join(format!("threshold_overrides_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "not valid json{{{").unwrap();

        let o = ThresholdOverrides::load_from_file(&path);
        assert_eq!(
            o.active_count(),
            0,
            "invalid JSON must fall back to defaults, never panic"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_from_file_valid_populates_overrides() {
        let dir = std::env::temp_dir().join(format!(
            "threshold_overrides_test_valid_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overrides.json");
        let mut o = ThresholdOverrides::default();
        o.fep_surprise_scale = Some(4.2);
        o.homeostasis_recalibrate_high = Some(1.2);
        std::fs::write(&path, serde_json::to_string(&o).unwrap()).unwrap();

        let loaded = ThresholdOverrides::load_from_file(&path);
        assert_eq!(loaded.fep_surprise_scale, Some(4.2));
        assert_eq!(loaded.homeostasis_recalibrate_high, Some(1.2));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_env_unset_is_default() {
        // Ensure clean env for this test (best-effort; std::env::var is
        // process-global, but no other test in this module sets this var).
        // SAFETY: This test clears a single process-wide test variable before
        // reading it and no other test in this module mutates this variable.
        unsafe { std::env::remove_var("SYMTHAEA_THRESHOLD_OVERRIDES_PATH") };
        let o = ThresholdOverrides::from_env();
        assert_eq!(o.active_count(), 0);
    }

    /// Pins the contract `cls_evolution_harness.rs` and `load_from_file` rely
    /// on: a `ThresholdPhenotype` (the type `evolve_cls`/the promotion gate
    /// actually produce) serializes to JSON that deserializes cleanly into a
    /// `ThresholdOverrides` with every field populated as `Some(..)` matching
    /// the phenotype's value. If this ever breaks (e.g. a field renamed on
    /// one side but not the other), a promoted phenotype would silently load
    /// as a partial/empty override set instead of failing loudly - this test
    /// exists so that drift is caught here first.
    #[cfg(feature = "neuroevolution")]
    #[test]
    fn test_threshold_phenotype_json_loads_as_overrides() {
        let pheno = symthaea_neuroevolution::ThresholdPhenotype {
            fep_surprise_scale: 4.5,
            fep_lr_decay: 0.93,
            dream_base_interval: 123,
            dream_min_interval: 45,
            neuromod_d2_baseline: 0.61,
            neuromod_ne_phasic_threshold: 0.25,
            neuromod_arousal_ema_decay: 0.88,
            homeostasis_recalibrate_high: 1.21,
            homeostasis_recalibrate_low: 0.79,
            neuromod_ema_alpha: 0.07,
            frustration_dampen_threshold: 0.55,
            engagement_low_threshold: 0.28,
            flow_exploration_increment: 0.06,
            coherence_low: 0.31,
            arousal_trap_threshold: 0.77,
            self_model_weight_high: 0.69,
            homeostasis_pull_cruise: 1.6,
            confidence_crash_threshold: 0.29,
        };

        let json = serde_json::to_string(&pheno).expect("phenotype serializes");
        let overrides: ThresholdOverrides =
            serde_json::from_str(&json).expect("phenotype JSON must deserialize as overrides");

        assert_eq!(overrides.active_count(), 18, "every field must be Some(..)");
        assert_eq!(overrides.fep_surprise_scale, Some(pheno.fep_surprise_scale));
        assert_eq!(overrides.fep_lr_decay, Some(pheno.fep_lr_decay));
        assert_eq!(
            overrides.dream_base_interval,
            Some(pheno.dream_base_interval)
        );
        assert_eq!(overrides.dream_min_interval, Some(pheno.dream_min_interval));
        assert_eq!(
            overrides.neuromod_d2_baseline,
            Some(pheno.neuromod_d2_baseline)
        );
        assert_eq!(
            overrides.neuromod_ne_phasic_threshold,
            Some(pheno.neuromod_ne_phasic_threshold)
        );
        assert_eq!(
            overrides.neuromod_arousal_ema_decay,
            Some(pheno.neuromod_arousal_ema_decay)
        );
        assert_eq!(
            overrides.homeostasis_recalibrate_high,
            Some(pheno.homeostasis_recalibrate_high)
        );
        assert_eq!(
            overrides.homeostasis_recalibrate_low,
            Some(pheno.homeostasis_recalibrate_low)
        );
        assert_eq!(overrides.neuromod_ema_alpha, Some(pheno.neuromod_ema_alpha));
        assert_eq!(
            overrides.frustration_dampen_threshold,
            Some(pheno.frustration_dampen_threshold)
        );
        assert_eq!(
            overrides.engagement_low_threshold,
            Some(pheno.engagement_low_threshold)
        );
        assert_eq!(
            overrides.flow_exploration_increment,
            Some(pheno.flow_exploration_increment)
        );
        assert_eq!(overrides.coherence_low, Some(pheno.coherence_low));
        assert_eq!(
            overrides.arousal_trap_threshold,
            Some(pheno.arousal_trap_threshold)
        );
        assert_eq!(
            overrides.self_model_weight_high,
            Some(pheno.self_model_weight_high)
        );
        assert_eq!(
            overrides.homeostasis_pull_cruise,
            Some(pheno.homeostasis_pull_cruise)
        );
        assert_eq!(
            overrides.confidence_crash_threshold,
            Some(pheno.confidence_crash_threshold)
        );
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
