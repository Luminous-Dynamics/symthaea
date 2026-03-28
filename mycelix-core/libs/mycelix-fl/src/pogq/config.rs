// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Configuration for PoGQ v4.1 Enhanced and Adaptive PoGQ.
//!
//! All Gen-4 parameters are represented with their paper-specified defaults.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PoGQ v4.1 Enhanced Configuration
// ---------------------------------------------------------------------------

/// Configuration for PoGQ v4.1 Enhanced.
///
/// Covers all Gen-4 components: Mondrian conformal, adaptive lambda,
/// direction prefilter, Winsorized dispersion, EMA smoothing, warm-up
/// quota, and hysteresis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoGQv41Config {
    // -- EMA smoothing --
    /// EMA smoothing factor (weight for historical scores).
    /// Higher values produce more stable but slower-reacting scores.
    /// Default: 0.85
    pub beta: f64,

    // -- Warm-up --
    /// Number of rounds before quarantine enforcement begins.
    /// During warm-up, only egregious violations trigger quarantine.
    /// Default: 3
    pub warm_up_rounds: usize,

    // -- Hysteresis (prevent flapping) --
    /// Consecutive violations required to enter quarantine.
    /// Default: 2
    pub k_quarantine: usize,

    /// Consecutive clears required to release from quarantine.
    /// Default: 3
    pub m_release: usize,

    // -- Hybrid scoring weights --
    /// Weight of cosine-similarity (direction) component in hybrid score.
    /// `hybrid = direction_weight * direction + magnitude_weight * magnitude`
    /// This is the *default* lambda when adaptive lambda has insufficient history.
    /// Default: 0.6
    pub direction_weight: f64,

    /// Weight of magnitude (z-score of L2 norm) component in hybrid score.
    /// Default: 0.4
    pub magnitude_weight: f64,

    // -- Adaptive lambda --
    /// Minimum adaptive lambda (floor for direction weight).
    /// Default: 0.3
    pub lambda_min: f64,

    /// Maximum adaptive lambda (ceiling for direction weight).
    /// Default: 0.9
    pub lambda_max: f64,

    /// SNR coefficient `a` in `lambda = sigmoid(a * SNR + b)`.
    /// Default: 2.0
    pub lambda_a: f64,

    /// SNR offset `b` in `lambda = sigmoid(a * SNR + b)`.
    /// Default: -1.0
    pub lambda_b: f64,

    /// Minimum number of gradient history entries before adaptive lambda
    /// is used. Below this, [`direction_weight`](Self::direction_weight) is used.
    /// Default: 5
    pub lambda_min_history: usize,

    // -- Egregious cap --
    /// Percentile for the egregious violation cap during warm-up.
    /// Scores below this percentile of historical scores are flagged even
    /// during the warm-up grace period.
    /// Default: 0.999 (p99.9)
    pub egregious_percentile: f64,

    // -- Mondrian conformal --
    /// Target false-positive rate for Mondrian conformal buckets.
    /// Default: 0.10
    pub mondrian_alpha: f64,

    /// Minimum number of calibration samples for a Mondrian bucket to
    /// get its own threshold. Below this, hierarchical backoff is used.
    /// Default: 10
    pub mondrian_min_bucket_size: usize,

    // -- Winsorized dispersion --
    /// Fraction to clip from each tail when computing Winsorized dispersion.
    /// Default: 0.05
    pub winsorize_percentile: f64,

    // -- Direction prefilter --
    /// Enable the direction prefilter (immediate rejection of opposite-direction
    /// gradients where cosine similarity < direction_prefilter_threshold).
    /// Default: true
    pub enable_direction_prefilter: bool,

    /// Cosine similarity threshold for the direction prefilter.
    /// Gradients with cosine similarity below this value vs. the centroid
    /// are flagged immediately. Default: -0.5
    pub direction_prefilter_threshold: f64,

    // -- PCA projection --
    /// Enable PCA-based cosine similarity (requires external PCA fitting).
    /// When false, full-space cosine similarity is used.
    /// Default: false (PCA not available in pure Rust yet)
    pub enable_pca: bool,

    /// Number of PCA components for projected cosine similarity.
    /// Default: 32
    pub pca_components: usize,

    // -- Gradient history --
    /// Maximum number of gradient norms retained for SNR computation.
    /// Default: 20
    pub gradient_history_window: usize,

    // -- Score history --
    /// Maximum number of historical scores retained per class for
    /// Mondrian z-score normalization.
    /// Default: 10
    pub class_score_window: usize,
}

impl Default for PoGQv41Config {
    fn default() -> Self {
        Self {
            beta: 0.85,
            warm_up_rounds: 3,
            k_quarantine: 2,
            m_release: 3,
            direction_weight: 0.6,
            magnitude_weight: 0.4,
            lambda_min: 0.3,
            lambda_max: 0.9,
            lambda_a: 2.0,
            lambda_b: -1.0,
            lambda_min_history: 5,
            egregious_percentile: 0.999,
            mondrian_alpha: 0.10,
            mondrian_min_bucket_size: 10,
            winsorize_percentile: 0.05,
            enable_direction_prefilter: true,
            direction_prefilter_threshold: -0.5,
            enable_pca: false,
            pca_components: 32,
            gradient_history_window: 20,
            class_score_window: 10,
        }
    }
}

impl PoGQv41Config {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.beta < 0.0 || self.beta > 1.0 {
            return Err(format!("beta must be in [0, 1], got {}", self.beta));
        }
        if self.direction_weight + self.magnitude_weight < 1e-12 {
            return Err("direction_weight + magnitude_weight must be > 0".into());
        }
        if self.lambda_min > self.lambda_max {
            return Err(format!(
                "lambda_min ({}) must be <= lambda_max ({})",
                self.lambda_min, self.lambda_max
            ));
        }
        if self.egregious_percentile < 0.0 || self.egregious_percentile > 1.0 {
            return Err(format!(
                "egregious_percentile must be in [0, 1], got {}",
                self.egregious_percentile
            ));
        }
        if self.mondrian_alpha <= 0.0 || self.mondrian_alpha >= 1.0 {
            return Err(format!(
                "mondrian_alpha must be in (0, 1), got {}",
                self.mondrian_alpha
            ));
        }
        if self.winsorize_percentile < 0.0 || self.winsorize_percentile >= 0.5 {
            return Err(format!(
                "winsorize_percentile must be in [0, 0.5), got {}",
                self.winsorize_percentile
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Adaptive PoGQ Configuration
// ---------------------------------------------------------------------------

/// Configuration for Adaptive PoGQ (per-node thresholds).
///
/// Extends PoGQ v4.1 with per-node adaptive thresholds that accommodate
/// non-IID data distributions. Instead of a single global threshold,
/// each node gets `tau_i = max(tau_min, mu_i - k * sigma_i)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptivePoGQConfig {
    /// Minimum global threshold (safety floor).
    /// No node's adaptive threshold can go below this.
    /// Default: 0.1
    pub tau_min: f64,

    /// Initial global threshold used before a node has enough history.
    /// Default: 0.3
    pub tau_global: f64,

    /// Confidence level `k` in `tau_i = mu_i - k * sigma_i`.
    /// `k=2.0` accepts ~95% of honest behavior (Gaussian assumption).
    /// Default: 2.0
    pub confidence_level: f64,

    /// Anomaly detection sensitivity `k_anomaly` for z-score detection.
    /// `k=3.0` flags ~0.3% of honest behavior (stricter than threshold).
    /// Default: 3.0
    pub anomaly_sensitivity: f64,

    /// Minimum number of historical observations before switching from
    /// the global threshold to the adaptive per-node threshold.
    /// Default: 5
    pub min_history_size: usize,

    /// Maximum number of historical observations kept per node.
    /// Default: 20
    pub max_history_size: usize,

    /// Enable gradient-norm anomaly detection (|z_norm| > anomaly_sensitivity).
    /// Default: true
    pub enable_norm_anomaly: bool,

    /// Enable quality z-score anomaly detection (z_quality < -anomaly_sensitivity).
    /// Default: true
    pub enable_quality_anomaly: bool,
}

impl Default for AdaptivePoGQConfig {
    fn default() -> Self {
        Self {
            tau_min: 0.1,
            tau_global: 0.3,
            confidence_level: 2.0,
            anomaly_sensitivity: 3.0,
            min_history_size: 5,
            max_history_size: 20,
            enable_norm_anomaly: true,
            enable_quality_anomaly: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pogq_config_default() {
        let cfg = PoGQv41Config::default();
        assert!((cfg.beta - 0.85).abs() < 1e-9);
        assert_eq!(cfg.warm_up_rounds, 3);
        assert_eq!(cfg.k_quarantine, 2);
        assert_eq!(cfg.m_release, 3);
        assert!((cfg.direction_weight - 0.6).abs() < 1e-9);
        assert!((cfg.magnitude_weight - 0.4).abs() < 1e-9);
        assert!((cfg.lambda_min - 0.3).abs() < 1e-9);
        assert!((cfg.lambda_max - 0.9).abs() < 1e-9);
        assert!((cfg.egregious_percentile - 0.999).abs() < 1e-9);
        assert!((cfg.mondrian_alpha - 0.10).abs() < 1e-9);
        assert!((cfg.winsorize_percentile - 0.05).abs() < 1e-9);
        assert!(!cfg.enable_pca);
        assert_eq!(cfg.pca_components, 32);
    }

    #[test]
    fn test_pogq_config_validate_ok() {
        let cfg = PoGQv41Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pogq_config_validate_bad_beta() {
        let mut cfg = PoGQv41Config::default();
        cfg.beta = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pogq_config_validate_bad_lambda_order() {
        let mut cfg = PoGQv41Config::default();
        cfg.lambda_min = 0.95;
        cfg.lambda_max = 0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pogq_config_serde_roundtrip() {
        let cfg = PoGQv41Config::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: PoGQv41Config = serde_json::from_str(&json).unwrap();
        assert!((cfg.beta - cfg2.beta).abs() < 1e-12);
        assert_eq!(cfg.warm_up_rounds, cfg2.warm_up_rounds);
        assert_eq!(cfg.k_quarantine, cfg2.k_quarantine);
    }

    #[test]
    fn test_adaptive_config_default() {
        let cfg = AdaptivePoGQConfig::default();
        assert!((cfg.tau_min - 0.1).abs() < 1e-9);
        assert!((cfg.tau_global - 0.3).abs() < 1e-9);
        assert!((cfg.confidence_level - 2.0).abs() < 1e-9);
        assert!((cfg.anomaly_sensitivity - 3.0).abs() < 1e-9);
        assert_eq!(cfg.min_history_size, 5);
        assert_eq!(cfg.max_history_size, 20);
        assert!(cfg.enable_norm_anomaly);
        assert!(cfg.enable_quality_anomaly);
    }

    #[test]
    fn test_adaptive_config_serde_roundtrip() {
        let cfg = AdaptivePoGQConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: AdaptivePoGQConfig = serde_json::from_str(&json).unwrap();
        assert!((cfg.tau_min - cfg2.tau_min).abs() < 1e-12);
        assert!((cfg.confidence_level - cfg2.confidence_level).abs() < 1e-12);
    }
}
