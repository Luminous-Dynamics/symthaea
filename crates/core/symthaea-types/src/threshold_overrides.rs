// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Serializable runtime threshold overrides shared across Symthaea crates.

use serde::{Deserialize, Serialize};

/// Environment variable pointing at a promoted threshold override JSON file.
pub const THRESHOLD_OVERRIDES_PATH_ENV: &str = "SYMTHAEA_THRESHOLD_OVERRIDES_PATH";

/// Runtime threshold override values. All `None` means use caller defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThresholdOverrideValues {
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

impl ThresholdOverrideValues {
    /// Clear all overrides.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Load threshold overrides from a JSON file.
    ///
    /// Fail-closed: a missing, unreadable, or unparseable file never panics.
    /// Missing files are silent because "no promoted phenotype yet" is the
    /// common default; unreadable or invalid files log a warning and fall back
    /// to an all-`None` default.
    pub fn load_from_file(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<Self>(&contents) {
                Ok(overrides) => {
                    tracing::info!(
                        path = %path.display(),
                        active_count = overrides.active_count(),
                        "Loaded promoted threshold overrides"
                    );
                    overrides
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to parse promoted threshold overrides; falling back to compile-time defaults"
                    );
                    Self::default()
                }
            },
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Could not read promoted threshold overrides; falling back to compile-time defaults"
                    );
                }
                Self::default()
            }
        }
    }

    /// Construct from [`THRESHOLD_OVERRIDES_PATH_ENV`], if set.
    pub fn from_env() -> Self {
        match std::env::var(THRESHOLD_OVERRIDES_PATH_ENV) {
            Ok(path) if !path.trim().is_empty() => {
                Self::load_from_file(std::path::Path::new(path.trim()))
            }
            _ => Self::default(),
        }
    }

    /// Count how many thresholds are overridden.
    pub fn active_count(&self) -> usize {
        [
            self.fep_surprise_scale.is_some(),
            self.fep_lr_decay.is_some(),
            self.dream_base_interval.is_some(),
            self.dream_min_interval.is_some(),
            self.neuromod_d2_baseline.is_some(),
            self.neuromod_ne_phasic_threshold.is_some(),
            self.neuromod_arousal_ema_decay.is_some(),
            self.homeostasis_recalibrate_high.is_some(),
            self.homeostasis_recalibrate_low.is_some(),
            self.neuromod_ema_alpha.is_some(),
            self.frustration_dampen_threshold.is_some(),
            self.engagement_low_threshold.is_some(),
            self.flow_exploration_increment.is_some(),
            self.coherence_low.is_some(),
            self.arousal_trap_threshold.is_some(),
            self.self_model_weight_high.is_some(),
            self.homeostasis_pull_cruise.is_some(),
            self.confidence_crash_threshold.is_some(),
        ]
        .into_iter()
        .filter(|is_active| *is_active)
        .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_none() {
        let overrides = ThresholdOverrideValues::default();
        assert_eq!(overrides.active_count(), 0);
    }

    #[test]
    fn load_from_file_missing_falls_back_to_default() {
        let path = std::path::Path::new("/nonexistent/path/does-not-exist-12345.json");
        let overrides = ThresholdOverrideValues::load_from_file(path);
        assert_eq!(overrides.active_count(), 0);
    }

    #[test]
    fn load_from_file_invalid_json_falls_back_to_default() {
        let dir = std::env::temp_dir().join(format!(
            "threshold_override_values_invalid_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "not valid json{{{").unwrap();

        let overrides = ThresholdOverrideValues::load_from_file(&path);
        assert_eq!(overrides.active_count(), 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_from_file_valid_populates_overrides() {
        let dir = std::env::temp_dir().join(format!(
            "threshold_override_values_valid_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overrides.json");
        let mut overrides = ThresholdOverrideValues::default();
        overrides.fep_surprise_scale = Some(4.2);
        overrides.homeostasis_recalibrate_high = Some(1.2);
        std::fs::write(&path, serde_json::to_string(&overrides).unwrap()).unwrap();

        let loaded = ThresholdOverrideValues::load_from_file(&path);
        assert_eq!(loaded.fep_surprise_scale, Some(4.2));
        assert_eq!(loaded.homeostasis_recalibrate_high, Some(1.2));

        std::fs::remove_dir_all(&dir).ok();
    }
}
