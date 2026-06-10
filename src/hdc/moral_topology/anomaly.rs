// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Anomaly detection and adaptive thresholds for moral topology.

use super::*;

/// Running statistics for adaptive anomaly threshold self-tuning.
///
/// Uses Welford's online algorithm for numerically stable mean/variance.
/// The system "learns" what drift and FE levels are normal and adjusts
/// thresholds to `mean + sigma_factor * std_dev`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAnomalyState {
    /// EMA of observed moral drift values.
    pub(crate) drift_ema: f64,
    /// EMA of squared drift deviations (for variance).
    pub(crate) drift_var_ema: f64,
    /// EMA of observed free energy values.
    pub(crate) fe_ema: f64,
    /// EMA of squared FE deviations (for variance).
    pub(crate) fe_var_ema: f64,
    /// Number of observations processed.
    pub(crate) observations: usize,
    /// Current effective drift threshold (adaptive).
    pub effective_drift_threshold: f64,
    /// Current effective FE sigma multiplier (adaptive).
    pub effective_fe_sigma: f64,
}

impl Default for AdaptiveAnomalyState {
    fn default() -> Self {
        Self {
            drift_ema: 0.0,
            drift_var_ema: 0.0,
            fe_ema: 0.0,
            fe_var_ema: 0.0,
            observations: 0,
            effective_drift_threshold: 0.25,
            effective_fe_sigma: 2.0,
        }
    }
}

impl AdaptiveAnomalyState {
    /// Update running statistics with a new observation.
    ///
    /// Returns true when warmup is complete and adaptive thresholds are active.
    pub fn observe(&mut self, drift: f64, free_energy: f64, config: &MoralAnomalyConfig) -> bool {
        // Guard: non-finite inputs would corrupt EMA state permanently
        let drift = if drift.is_finite() { drift } else { 0.0 };
        let free_energy = if free_energy.is_finite() {
            free_energy
        } else {
            0.0
        };

        let alpha = config.adaptive_alpha;
        self.observations += 1;

        // EMA update for drift
        self.drift_ema = alpha * drift + (1.0 - alpha) * self.drift_ema;
        let drift_dev = (drift - self.drift_ema).powi(2);
        self.drift_var_ema = alpha * drift_dev + (1.0 - alpha) * self.drift_var_ema;

        // EMA update for free energy
        self.fe_ema = alpha * free_energy + (1.0 - alpha) * self.fe_ema;
        let fe_dev = (free_energy - self.fe_ema).powi(2);
        self.fe_var_ema = alpha * fe_dev + (1.0 - alpha) * self.fe_var_ema;

        // Only activate after warmup
        if self.observations < config.adaptive_warmup {
            return false;
        }

        // Compute adaptive drift threshold: mean + sigma_factor * std_dev
        let drift_std = self.drift_var_ema.sqrt();
        let adaptive_drift = self.drift_ema + config.adaptive_sigma_factor * drift_std;
        // Clamp: never below 50% of the user's configured baseline (prevents
        // adaptive loosening from silencing detection entirely), and never
        // above 0.8 (never become completely insensitive).
        let floor = (config.drift_alert_threshold * 0.5).max(0.05);
        self.effective_drift_threshold = adaptive_drift.clamp(floor, 0.8);

        // Compute adaptive FE sigma: scale based on observed FE variance.
        // If FE variance is naturally high, use a higher multiplier (less sensitive).
        // If FE variance is naturally low, tighten sensitivity.
        let fe_std = self.fe_var_ema.sqrt();
        let fe_cv = if self.fe_ema.abs() > 1e-9 {
            fe_std / self.fe_ema.abs()
        } else {
            0.0
        };
        // CV < 0.1 → tight (1.5σ), CV > 0.5 → loose (3.0σ)
        let adaptive_fe_sigma = 1.5 + fe_cv.clamp(0.0, 0.5) * 3.0;
        self.effective_fe_sigma = adaptive_fe_sigma.clamp(1.5, 3.5);

        true
    }

    /// Number of observations processed so far.
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Current drift EMA (mean observed drift).
    pub fn drift_mean(&self) -> f64 {
        self.drift_ema
    }

    /// Current drift standard deviation estimate.
    pub fn drift_std(&self) -> f64 {
        self.drift_var_ema.sqrt()
    }
}

/// Report of detected moral trajectory anomalies.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MoralAnomalyReport {
    /// Dominant harmony axis flipped since last evaluation.
    pub value_inversion: bool,
    /// Free energy jumped >2σ from recent rolling mean.
    pub free_energy_spike: bool,
    /// β₀ increased (more disconnected components).
    pub fragmentation_increase: bool,
    /// `moral_drift(20) > 0.25`.
    pub drift_alert: bool,
    /// Compartmentalized trajectory convergence detected.
    ///
    /// Fires when individually benign requests form an emergent cluster whose
    /// aggregate topology signals hazardous convergence:
    /// - Anomalous pairwise similarity increase (unrelated topics clustering)
    /// - Harmony entropy decline (narrowing moral focus)
    /// - Flourishing deficit (care/consent axes suppressed)
    ///
    /// This catches the "unknowingly building a nuke" pattern: each component
    /// looks benign in isolation, but the trajectory's aggregate shape reveals
    /// the adversary's compartmentalized plan.
    pub trajectory_convergence: bool,
    /// Trajectory convergence severity in \[0.0, 1.0\].
    ///
    /// Computed as the mean of three normalized signals:
    /// similarity_anomaly, entropy_decline, flourishing_deficit.
    /// 0.0 = no convergence; 1.0 = all three signals maximally triggered.
    pub convergence_severity: f64,
    /// Name of matched hazard signature template (e.g. "weaponization"), if any.
    pub matched_hazard: Option<String>,
    /// Sustained moral overconfidence detected.
    #[serde(default)]
    pub moral_hubris: bool,
    /// Composite anomaly score in \[0.0, 1.0\].
    pub anomaly_score: f64,
}

impl MoralTopology {
    /// Detect anomalies by comparing `current_summary` against trajectory history.
    ///
    /// Thresholds and weights are drawn from `self.anomaly_config`.
    /// When `adaptive_enabled`, thresholds are dynamically tuned from experience.
    pub fn detect_anomalies(
        &mut self,
        current_summary: &MoralTopologySummary,
    ) -> MoralAnomalyReport {
        // Compare against the PREVIOUS analyze() result, not last_summary
        // (which analyze() already overwrote to match current_summary).
        let prev = &self.prev_summary;
        let ac = self.anomaly_config.clone();

        // Resolve effective thresholds: adaptive or static
        let (drift_threshold, fe_sigma) = if ac.adaptive_enabled {
            (
                self.adaptive_state.effective_drift_threshold,
                self.adaptive_state.effective_fe_sigma,
            )
        } else {
            (ac.drift_alert_threshold, ac.fe_sigma_multiplier)
        };

        // Value inversion: dominant harmony axis changed.
        // Require >= 4 scenarios so the PGA dominant axis is statistically meaningful.
        let value_inversion =
            prev.scenario_count >= 4 && current_summary.dominant_harmony != prev.dominant_harmony;

        // Free energy spike: > fe_sigma × σ from rolling mean.
        // Guard against NaN/infinity in moral_free_energy.
        let fe_spike = {
            let points: Vec<_> = self.trajectory.iter().collect();
            let fe_current = if current_summary.moral_free_energy.is_finite() {
                current_summary.moral_free_energy
            } else {
                0.0 // treat NaN/infinity as prior (no spike)
            };
            if points.len() >= 4 {
                let mean_fe =
                    points.iter().map(|p| p.free_energy).sum::<f64>() / points.len() as f64;
                let var = points
                    .iter()
                    .map(|p| (p.free_energy - mean_fe).powi(2))
                    .sum::<f64>()
                    / points.len() as f64;
                let sigma = var.sqrt().max(1e-9);
                (fe_current - mean_fe).abs() > fe_sigma * sigma
            } else {
                false
            }
        };

        // Fragmentation: β₀ increased (more disconnected components).
        // Require >= 4 scenarios so β₀ reflects actual topology, not noise.
        let fragmentation_increase =
            prev.scenario_count >= 4 && current_summary.beta_0 > prev.beta_0;

        // Drift alert
        let drift = self.moral_drift(20);
        let drift_alert = drift > drift_threshold;

        // Trajectory convergence: compartmentalized adversarial detection
        let convergence_report = self.detect_trajectory_convergence();
        let trajectory_convergence = convergence_report.convergence_detected;
        let convergence_severity = convergence_report.severity;

        // Moral hubris: sustained high-coherence low-variance state
        let moral_hubris = if ac.hubris_enabled {
            let max_entropy = (N_HARMONIES as f64).ln();
            let normalized_entropy =
                if max_entropy > 0.0 && current_summary.harmony_entropy.is_finite() {
                    current_summary.harmony_entropy / max_entropy
                } else {
                    1.0
                };
            let coherence_proxy = if current_summary.moral_free_energy.is_finite() {
                (1.0 - current_summary.moral_free_energy.min(2.0) / 2.0).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if coherence_proxy > ac.hubris_coherence_threshold
                && normalized_entropy < ac.hubris_max_variance
            {
                self.hubris_streak += 1;
            } else {
                self.hubris_streak = 0;
            }
            self.hubris_streak >= ac.hubris_min_streak
        } else {
            false
        };

        // Composite score: weighted sum clamped to [0, 1]
        let raw = (value_inversion as u8 as f64) * ac.weight_value_inversion
            + (fe_spike as u8 as f64) * ac.weight_fe_spike
            + (fragmentation_increase as u8 as f64) * ac.weight_fragmentation
            + (drift_alert as u8 as f64) * ac.weight_drift
            + (trajectory_convergence as u8 as f64) * ac.weight_convergence
            + (moral_hubris as u8 as f64) * ac.weight_hubris;
        let anomaly_score = raw.clamp(0.0, 1.0);

        // Feed observation to adaptive state if enabled
        if ac.adaptive_enabled {
            self.adaptive_state
                .observe(drift, current_summary.moral_free_energy, &ac);
        }

        MoralAnomalyReport {
            value_inversion,
            free_energy_spike: fe_spike,
            fragmentation_increase,
            drift_alert,
            trajectory_convergence,
            convergence_severity,
            matched_hazard: convergence_report.matched_hazard.clone(),
            moral_hubris,
            anomaly_score,
        }
    }

    /// Access the adaptive anomaly state (for telemetry/debugging).
    pub fn adaptive_state(&self) -> &AdaptiveAnomalyState {
        &self.adaptive_state
    }

    /// Moral drift: L2 distance between mean of first half and second half
    /// of the last `lookback` trajectory points. Higher → greater drift.
    pub fn moral_drift(&self, lookback: usize) -> f64 {
        let points: Vec<_> = self.trajectory.iter().rev().take(lookback).collect();
        if points.len() < 4 {
            return 0.0;
        }
        let mid = points.len() / 2;
        let mean_half = |slice: &[&MoralTrajectoryPoint]| -> [f64; N_HARMONIES] {
            let mut m = [0.0; N_HARMONIES];
            for p in slice {
                for i in 0..N_HARMONIES {
                    m[i] += p.coordinates[i];
                }
            }
            let n = slice.len() as f64;
            for v in &mut m {
                *v /= n;
            }
            m
        };
        let first_half = mean_half(&points[mid..]);
        let second_half = mean_half(&points[..mid]);
        let mut dist_sq = 0.0;
        for i in 0..N_HARMONIES {
            let d = first_half[i] - second_half[i];
            dist_sq += d * d;
        }
        dist_sq.sqrt()
    }
}
