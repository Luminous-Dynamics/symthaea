// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A snapshot of the 7 canonical Waterfall metrics at one instant in time.
//!
//! All values are `f64` normalized where possible:
//! - `phi` — integration index (raw, typically 0.0–10.0+)
//! - `fep_prediction_error` — free energy prediction error (raw ≥ 0)
//! - `workspace_activation` — GWT broadcast magnitude [0, ∞)
//! - `hot_confidence` — HOT self-model confidence [0, 1]
//! - `anomaly_score` — computed anomaly index [0, 1]
//! - `memory_pressure` — total uncertainty (raw, normalized by dim)
//! - `mip_instability` — MIP normalized instability [0, 1]

use std::time::{SystemTime, UNIX_EPOCH};

/// A timestamped snapshot of all 7 canonical waterfall metrics.
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    /// Unix timestamp (seconds, fractional).
    pub timestamp: f64,
    /// Monotonic frame counter.
    pub frame_id: u64,

    // ── The 7 canonical metrics ────────────────────────────────────────────
    /// IIT integration index (from phi-oracle). Raw value.
    /// Typical range: 0.0–5.0. Higher = more integrated.
    pub phi: f64,

    /// FEP prediction error magnitude. Raw value ≥ 0.
    /// Higher = more surprise / model mismatch.
    pub fep_prediction_error: f64,

    /// GWT workspace broadcast magnitude.
    /// 0.0 = no active broadcast. Higher = stronger attention focus.
    pub workspace_activation: f64,

    /// HOT self-model confidence [0, 1].
    /// 0 = complete uncertainty about internal state.
    pub hot_confidence: f64,

    /// Anomaly score [0, 1].
    /// Computed as: divergence between phi trend and prediction_error trend.
    /// High = something has changed that the model didn't predict.
    pub anomaly_score: f64,

    /// Memory pressure — total belief uncertainty.
    /// Raw value from HiddenState::total_uncertainty(). Higher = more uncertain.
    pub memory_pressure: f64,

    /// MIP instability — normalized MIP index from phi-oracle.
    /// 0 = stable partition. 1 = maximum instability.
    pub mip_instability: f64,

    // ── Metadata ──────────────────────────────────────────────────────────
    /// True if this snapshot was taken during a known perturbation event.
    pub is_perturbation_event: bool,

    /// Human-readable description of any active event (for Chronicle).
    pub event_description: Option<String>,

    /// Source system tags (which subsystems contributed to this snapshot).
    pub source_tags: Vec<String>,
}

impl MetricSnapshot {
    /// Create a snapshot with all metrics explicitly specified.
    pub fn new(
        frame_id: u64,
        phi: f64,
        fep_prediction_error: f64,
        workspace_activation: f64,
        hot_confidence: f64,
        memory_pressure: f64,
        mip_instability: f64,
    ) -> Self {
        let anomaly_score = Self::compute_anomaly(phi, fep_prediction_error, mip_instability);

        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(frame_id as f64),
            frame_id,
            phi,
            fep_prediction_error,
            workspace_activation,
            hot_confidence,
            anomaly_score,
            memory_pressure,
            mip_instability,
            is_perturbation_event: false,
            event_description: None,
            source_tags: vec![
                "fep".to_string(),
                "phi_oracle".to_string(),
                "workspace".to_string(),
            ],
        }
    }

    /// Create a zero/baseline snapshot for testing or warmup.
    ///
    /// Represents a cold-start system with neutral (0.5) confidence —
    /// not a perfectly certain system. All signal metrics are zero.
    pub fn baseline(frame_id: u64) -> Self {
        Self::new(frame_id, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    }

    /// Mark this snapshot as a perturbation event.
    pub fn with_event(mut self, description: impl Into<String>) -> Self {
        self.is_perturbation_event = true;
        self.event_description = Some(description.into());
        self
    }

    /// Compute the anomaly score from the three most diagnostic metrics.
    ///
    /// Formula: weighted combination of:
    /// - phi collapse (phi near zero when it was high = anomaly)
    /// - prediction error spike (high prediction_error = surprise)
    /// - mip instability
    ///
    /// Result is clamped to [0, 1].
    pub fn compute_anomaly(phi: f64, prediction_error: f64, mip_instability: f64) -> f64 {
        // Normalize prediction error (assume typical range 0–5)
        let pred_err_norm = (prediction_error / 5.0).min(1.0);
        // Weight: prediction error is the most acute signal
        let score = 0.4 * pred_err_norm + 0.3 * mip_instability + 0.3 * (1.0 - phi.min(5.0) / 5.0);
        score.clamp(0.0, 1.0)
    }

    /// Frame confidence: high when anomaly is low and hot_confidence is high.
    pub fn frame_confidence(&self) -> f64 {
        // If anomaly is high, confidence in the frame's data quality drops
        let base = self.hot_confidence;
        let anomaly_penalty = self.anomaly_score * 0.5;
        (base - anomaly_penalty).clamp(0.05, 1.0)
    }

    /// True if this snapshot looks suspiciously uniform (false-green risk).
    ///
    /// False-green pattern: the machine reports everything as healthy and perfectly stable.
    /// The key signals are ALL four of:
    /// - `hot_confidence > 0.99` (machine claims perfect self-knowledge)
    /// - `fep_prediction_error < 0.01` (zero surprise — too good)
    /// - `mip_instability < 0.01` (zero structural instability — too stable)
    /// - `memory_pressure < 0.01` (zero uncertainty — physically impossible in real systems)
    ///
    /// Note: phi is NOT checked here — a false-green machine can report any phi value.
    /// The anomaly_score is also NOT checked — it's a derived quantity that may be
    /// non-zero even when the raw metrics look artificially clean.
    pub fn is_false_green_suspect(&self) -> bool {
        self.hot_confidence > 0.99
            && self.fep_prediction_error < 0.01
            && self.mip_instability < 0.01
            && self.memory_pressure < 0.01
    }

    /// Produce anomaly tags for the ProjectionFrame.
    pub fn anomaly_tags(&self) -> Vec<String> {
        let mut tags = vec![];
        if self.anomaly_score > 0.7 {
            tags.push("high_anomaly".to_string());
        }
        if self.fep_prediction_error > 2.0 {
            tags.push("prediction_error_spike".to_string());
        }
        if self.mip_instability > 0.6 {
            tags.push("mip_instability".to_string());
        }
        if self.phi < 0.1 && self.workspace_activation > 0.5 {
            tags.push("integration_collapse_with_active_workspace".to_string());
        }
        if self.is_false_green_suspect() {
            tags.push("false_green_suspect".to_string());
        }
        if self.is_perturbation_event {
            tags.push("perturbation_event".to_string());
        }
        tags
    }

    /// Convert to a scalar metrics map for ProjectionFrame.
    pub fn to_scalar_map(&self) -> std::collections::HashMap<String, f64> {
        let mut map = std::collections::HashMap::new();
        map.insert("phi".to_string(), self.phi);
        map.insert(
            "fep_prediction_error".to_string(),
            self.fep_prediction_error,
        );
        map.insert(
            "workspace_activation".to_string(),
            self.workspace_activation,
        );
        map.insert("hot_confidence".to_string(), self.hot_confidence);
        map.insert("anomaly_score".to_string(), self.anomaly_score);
        map.insert("memory_pressure".to_string(), self.memory_pressure);
        map.insert("mip_instability".to_string(), self.mip_instability);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_snapshot_is_not_anomalous() {
        let s = MetricSnapshot::baseline(0);
        assert!(s.anomaly_score < 0.4);
        assert!(!s.is_false_green_suspect());
    }

    #[test]
    fn high_prediction_error_raises_anomaly() {
        let s = MetricSnapshot::new(1, 2.0, 4.5, 0.5, 0.8, 0.5, 0.2);
        assert!(
            s.anomaly_score > 0.3,
            "high pred error should raise anomaly: {}",
            s.anomaly_score
        );
        assert!(
            s.anomaly_tags()
                .iter()
                .any(|t| t == "prediction_error_spike")
        );
    }

    #[test]
    fn false_green_detected() {
        let s = MetricSnapshot::new(2, 1.0, 0.001, 0.5, 0.999, 0.001, 0.001);
        assert!(
            s.is_false_green_suspect(),
            "suspiciously perfect metrics should be flagged"
        );
        assert!(s.anomaly_tags().iter().any(|t| t == "false_green_suspect"));
    }

    #[test]
    fn frame_confidence_drops_under_high_anomaly() {
        let s = MetricSnapshot::new(3, 0.0, 5.0, 0.0, 0.5, 2.0, 1.0);
        assert!(s.frame_confidence() < 0.5);
    }

    #[test]
    fn scalar_map_has_all_seven_keys() {
        let s = MetricSnapshot::baseline(0);
        let map = s.to_scalar_map();
        for key in &[
            "phi",
            "fep_prediction_error",
            "workspace_activation",
            "hot_confidence",
            "anomaly_score",
            "memory_pressure",
            "mip_instability",
        ] {
            assert!(map.contains_key(*key), "missing key: {key}");
        }
    }
}
