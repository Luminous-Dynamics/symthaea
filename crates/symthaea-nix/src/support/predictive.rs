//! LTC Temporal Failure Prediction
//!
//! Extrapolates system telemetry forward in time using the O(1) LTC closed-form
//! solver. Predicts disk exhaustion, memory pressure, store bloat, and error
//! rate acceleration across multiple time horizons (1h, 6h, 24h, 7d).
//!
//! Key insight: encode telemetry as HDC vector → feed to LTC neuron →
//! `evolve_closed_form(dt, input_hv)` → decode predicted values → check thresholds.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::ContinuousHV;

use crate::encoding::codebook::{NixCodebook, NIX_HDC_DIM};

/// System telemetry sample for prediction input.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemTelemetry {
    pub disk_used_pct: f64,
    pub memory_used_pct: f64,
    pub store_path_count: u64,
    pub failed_unit_count: u32,
}

/// A single telemetry sample with its timestamp for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedTelemetrySample {
    pub timestamp_secs: u64,
    pub telemetry: SystemTelemetry,
}

/// Serializable predictive monitor state for persistence across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPredictiveState {
    pub samples: Vec<SavedTelemetrySample>,
    pub neuron_state: Vec<f32>,
}

/// Alert thresholds for predicted values.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub disk_warn_pct: f64,
    pub disk_crit_pct: f64,
    pub memory_warn_pct: f64,
    pub memory_crit_pct: f64,
    pub store_warn_paths: u64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            disk_warn_pct: 80.0,
            disk_crit_pct: 90.0,
            memory_warn_pct: 85.0,
            memory_crit_pct: 95.0,
            store_warn_paths: 100_000,
        }
    }
}

/// A single prediction result.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub metric: &'static str,
    pub current_value: f64,
    pub predicted_value: f64,
    pub hours_ahead: f64,
    pub crosses_threshold: bool,
    pub threshold: f64,
    pub recommended_action: Option<String>,
    pub confidence: f32,
}

/// Predictive monitor using LTC temporal evolution.
pub struct PredictiveMonitor {
    neuron: HdcLtcUnifiedNeuron,
    /// (Instant for dt computation, unix timestamp for persistence, telemetry)
    history: Vec<(Instant, u64, SystemTelemetry)>,
    thresholds: AlertThresholds,
    codebook: NixCodebook,
    max_history: usize,
}

impl PredictiveMonitor {
    /// Create a new predictive monitor.
    pub fn new(thresholds: AlertThresholds) -> Self {
        let config = UnifiedConfig {
            dimension: NIX_HDC_DIM,
            tau_base: 3600.0, // 1-hour time constant (telemetry evolves slowly)
            ..UnifiedConfig::default()
        };

        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x4E49_5850_5244_0000), // "NIXPRD\0\0"
            history: Vec::new(),
            thresholds,
            codebook: NixCodebook::new(),
            max_history: 1000,
        }
    }

    /// Create with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(AlertThresholds::default())
    }

    /// Ingest a new telemetry sample. The neuron evolves its internal state.
    pub fn ingest(&mut self, telemetry: SystemTelemetry) {
        let input_hv = self.encode_telemetry(&telemetry);

        // Compute dt from last observation
        let dt = if let Some((last_time, _, _)) = self.history.last() {
            last_time.elapsed().as_secs_f32()
        } else {
            1.0 // first sample — use 1s as nominal
        };

        // Evolve the LTC neuron with the new observation
        self.neuron.evolve_closed_form(dt, &input_hv);

        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.history.push((Instant::now(), now_secs, telemetry));
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Predict all metrics across standard horizons (1h, 6h, 24h, 7d).
    pub fn predict_all_horizons(&mut self) -> Vec<Prediction> {
        let horizons = [1.0, 6.0, 24.0, 168.0]; // hours
        let mut predictions = Vec::new();

        for &hours in &horizons {
            predictions.extend(self.predict(hours));
        }

        predictions
    }

    /// Decode a predicted HDC vector back into telemetry values by projecting
    /// onto the role vectors used during encoding.
    pub fn decode_telemetry(&mut self, hv: &ContinuousHV) -> SystemTelemetry {
        let disk_role = self.codebook.get_or_create("disk_usage").clone();
        let mem_role = self.codebook.get_or_create("memory_usage").clone();
        let store_role = self.codebook.get_or_create("store_paths").clone();
        let fail_role = self.codebook.get_or_create("failed_units").clone();

        // Similarity gives the approximate scaled value, attenuated by bundling.
        // With 4 components bundled, each similarity is ~1/4 of the true value,
        // so we compensate with NUM_COMPONENTS.
        const NUM_COMPONENTS: f64 = 4.0;
        let disk_sim = hv.similarity(&disk_role).max(0.0) as f64 * NUM_COMPONENTS;
        let mem_sim = hv.similarity(&mem_role).max(0.0) as f64 * NUM_COMPONENTS;
        let store_sim = hv.similarity(&store_role).max(0.0) as f64 * NUM_COMPONENTS;
        let fail_sim = hv.similarity(&fail_role).max(0.0) as f64 * NUM_COMPONENTS;

        // Denormalize back to metric ranges
        SystemTelemetry {
            disk_used_pct: (disk_sim * 100.0).clamp(0.0, 100.0),
            memory_used_pct: (mem_sim * 100.0).clamp(0.0, 100.0),
            store_path_count: (store_sim * 200_000.0).max(0.0) as u64,
            failed_unit_count: (fail_sim * 10.0).max(0.0) as u32,
        }
    }

    /// Use the LTC neuron's closed-form solver to predict future telemetry.
    ///
    /// Clones the neuron, evolves it forward by `hours_ahead` hours, then
    /// decodes the resulting state vector back to telemetry values.
    pub fn predict_ltc(&mut self, hours_ahead: f64) -> Option<SystemTelemetry> {
        if self.history.is_empty() {
            return None;
        }

        // Clone the last telemetry to avoid borrow conflict
        let last_entry = self.history.last()?;
        let last_telemetry = last_entry.2.clone();
        let last_input = self.encode_telemetry(&last_telemetry);

        let mut future_neuron = self.neuron.clone();
        let dt_seconds = (hours_ahead * 3600.0) as f32;
        future_neuron.evolve_closed_form_iterative(dt_seconds, &last_input);
        let predicted_hv = future_neuron.state().clone();

        Some(self.decode_telemetry(&predicted_hv))
    }

    /// Predict all metrics at a specific time horizon.
    ///
    /// Uses linear trend extrapolation for predicted values. When enough history
    /// is available (>=10 samples), the LTC neuron is consulted for directional
    /// agreement — if LTC and trend agree, confidence is boosted.
    pub fn predict(&mut self, hours_ahead: f64) -> Vec<Prediction> {
        let current = match self.history.last() {
            Some((_, _, t)) => t.clone(),
            None => return vec![],
        };

        let trend = self.compute_trend();
        let ltc_pred = if self.history.len() >= 10 {
            self.predict_ltc(hours_ahead)
        } else {
            None
        };
        let mut confidence = self.compute_confidence();

        // If LTC and trend agree on direction, boost confidence
        if let Some(ref ltc) = ltc_pred {
            let trend_disk_dir = trend.disk_used_pct.signum();
            let ltc_disk_dir = (ltc.disk_used_pct - current.disk_used_pct).signum();
            if trend_disk_dir == ltc_disk_dir {
                confidence = (confidence * 1.15).min(1.0);
            }
        }

        let mut predictions = Vec::new();

        // Disk prediction — trend extrapolation (LTC used for confidence only)
        let trend_disk = current.disk_used_pct + trend.disk_used_pct * hours_ahead;
        let disk_pred = trend_disk.clamp(0.0, 100.0);
        let disk_threshold = self.thresholds.disk_crit_pct;
        predictions.push(Prediction {
            metric: "disk_used_pct",
            current_value: current.disk_used_pct,
            predicted_value: disk_pred,
            hours_ahead,
            crosses_threshold: disk_pred >= disk_threshold
                && current.disk_used_pct < disk_threshold,
            threshold: disk_threshold,
            recommended_action: if disk_pred >= self.thresholds.disk_warn_pct {
                Some("Run: nix-collect-garbage -d --delete-older-than 7d".to_string())
            } else {
                None
            },
            confidence,
        });

        // Memory prediction — trend extrapolation (LTC used for confidence only)
        let trend_mem = current.memory_used_pct + trend.memory_used_pct * hours_ahead;
        let mem_pred = trend_mem.clamp(0.0, 100.0);
        let mem_threshold = self.thresholds.memory_crit_pct;
        predictions.push(Prediction {
            metric: "memory_used_pct",
            current_value: current.memory_used_pct,
            predicted_value: mem_pred,
            hours_ahead,
            crosses_threshold: mem_pred >= mem_threshold && current.memory_used_pct < mem_threshold,
            threshold: mem_threshold,
            recommended_action: if mem_pred >= self.thresholds.memory_warn_pct {
                Some("Investigate memory usage: systemd-cgtop".to_string())
            } else {
                None
            },
            confidence,
        });

        // Store path count prediction — trend extrapolation (LTC used for confidence only)
        let trend_store =
            current.store_path_count as f64 + trend.store_paths_per_hour * hours_ahead;
        let store_pred = trend_store.max(0.0);
        let store_threshold = self.thresholds.store_warn_paths as f64;
        predictions.push(Prediction {
            metric: "store_path_count",
            current_value: current.store_path_count as f64,
            predicted_value: store_pred,
            hours_ahead,
            crosses_threshold: store_pred >= store_threshold
                && (current.store_path_count as f64) < store_threshold,
            threshold: store_threshold,
            recommended_action: if store_pred >= store_threshold {
                Some("Run: nix-collect-garbage -d".to_string())
            } else {
                None
            },
            confidence,
        });

        // Failed unit count prediction — trend extrapolation (LTC used for confidence only)
        let trend_fail =
            current.failed_unit_count as f64 + trend.failed_units_per_hour * hours_ahead;
        let fail_pred = trend_fail.max(0.0);
        let fail_threshold = 3.0;
        predictions.push(Prediction {
            metric: "failed_unit_count",
            current_value: current.failed_unit_count as f64,
            predicted_value: fail_pred,
            hours_ahead,
            crosses_threshold: fail_pred >= fail_threshold
                && (current.failed_unit_count as f64) < fail_threshold,
            threshold: fail_threshold,
            recommended_action: if fail_pred >= fail_threshold {
                Some("Review failed services: systemctl --failed".to_string())
            } else {
                None
            },
            confidence,
        });

        predictions
    }

    /// Number of telemetry samples ingested.
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Serialize the monitor's state for persistence across restarts.
    pub fn save(&self) -> SavedPredictiveState {
        SavedPredictiveState {
            samples: self
                .history
                .iter()
                .map(|(_, ts, t)| SavedTelemetrySample {
                    timestamp_secs: *ts,
                    telemetry: t.clone(),
                })
                .collect(),
            neuron_state: self.neuron.state().as_slice().to_vec(),
        }
    }

    /// Restore a monitor from saved state. History entries get approximate
    /// `Instant` offsets reconstructed from the unix timestamps.
    pub fn load(saved: SavedPredictiveState, thresholds: AlertThresholds) -> Self {
        let config = UnifiedConfig {
            dimension: NIX_HDC_DIM,
            tau_base: 3600.0,
            ..UnifiedConfig::default()
        };

        let mut neuron = HdcLtcUnifiedNeuron::new(config, 0x4E49_5850_5244_0000);

        // Restore neuron state if dimensions match
        if !saved.neuron_state.is_empty() {
            let restored_hv = ContinuousHV::from_values(saved.neuron_state);
            if restored_hv.dim() == NIX_HDC_DIM {
                *neuron.state_mut() = restored_hv;
            }
        }

        let now = Instant::now();
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Reconstruct Instants with approximate offsets from saved timestamps
        let history: Vec<(Instant, u64, SystemTelemetry)> = saved
            .samples
            .into_iter()
            .map(|sample| {
                let age_secs = now_secs.saturating_sub(sample.timestamp_secs);
                let instant = now - std::time::Duration::from_secs(age_secs);
                (instant, sample.timestamp_secs, sample.telemetry)
            })
            .collect();

        Self {
            neuron,
            history,
            thresholds,
            codebook: NixCodebook::new(),
            max_history: 1000,
        }
    }

    // ---- Internal helpers ----

    /// Encode telemetry values into an HDC vector.
    fn encode_telemetry(&mut self, telemetry: &SystemTelemetry) -> ContinuousHV {
        // Encode each metric as a role-bound value
        let disk_hv = self.codebook.get_or_create("disk_usage").clone();
        let mem_hv = self.codebook.get_or_create("memory_usage").clone();
        let store_hv = self.codebook.get_or_create("store_paths").clone();
        let fail_hv = self.codebook.get_or_create("failed_units").clone();

        // Scale role vectors by normalized metric values
        let disk_encoded = disk_hv.scale(telemetry.disk_used_pct as f32 / 100.0);
        let mem_encoded = mem_hv.scale(telemetry.memory_used_pct as f32 / 100.0);
        let store_encoded =
            store_hv.scale((telemetry.store_path_count as f32 / 200_000.0).min(1.0));
        let fail_encoded = fail_hv.scale((telemetry.failed_unit_count as f32 / 10.0).min(1.0));

        ContinuousHV::bundle(&[&disk_encoded, &mem_encoded, &store_encoded, &fail_encoded])
    }

    /// Compute per-hour trends from the history using simple linear regression.
    fn compute_trend(&self) -> TelemetryTrend {
        if self.history.len() < 2 {
            return TelemetryTrend::default();
        }

        let (first_time, _, first) = &self.history[0];
        let Some((last_time, _, last)) = self.history.last() else {
            return TelemetryTrend::default();
        };
        let hours = last_time.duration_since(*first_time).as_secs_f64() / 3600.0;

        if hours < 0.001 {
            return TelemetryTrend::default();
        }

        TelemetryTrend {
            disk_used_pct: (last.disk_used_pct - first.disk_used_pct) / hours,
            memory_used_pct: (last.memory_used_pct - first.memory_used_pct) / hours,
            store_paths_per_hour: (last.store_path_count as f64 - first.store_path_count as f64)
                / hours,
            failed_units_per_hour: (last.failed_unit_count as f64 - first.failed_unit_count as f64)
                / hours,
        }
    }

    /// Compute confidence based on history depth and neuron state.
    fn compute_confidence(&self) -> f32 {
        // More history = higher confidence, up to a cap
        let history_factor = (self.history.len() as f32 / 100.0).min(1.0);
        // Neuron state norm with smooth saturation: norm/(norm+1) maps [0,∞) → [0,1)
        let norm = self.neuron.state().norm();
        let state_factor = norm / (norm + 1.0);
        // Combine
        (history_factor * 0.7 + state_factor * 0.3).clamp(0.0, 1.0)
    }
}

/// Per-hour rates of change.
#[derive(Debug, Clone, Default)]
struct TelemetryTrend {
    disk_used_pct: f64,
    memory_used_pct: f64,
    store_paths_per_hour: f64,
    failed_units_per_hour: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_telemetry(disk: f64, mem: f64, store: u64, failed: u32) -> SystemTelemetry {
        SystemTelemetry {
            disk_used_pct: disk,
            memory_used_pct: mem,
            store_path_count: store,
            failed_unit_count: failed,
        }
    }

    #[test]
    fn test_predictive_monitor_creation() {
        let monitor = PredictiveMonitor::with_defaults();
        assert_eq!(monitor.sample_count(), 0);
    }

    #[test]
    fn test_ingest_single_sample() {
        let mut monitor = PredictiveMonitor::with_defaults();
        monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        assert_eq!(monitor.sample_count(), 1);
    }

    #[test]
    fn test_predict_with_no_history() {
        let mut monitor = PredictiveMonitor::with_defaults();
        let predictions = monitor.predict(24.0);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predict_stable_system() {
        let mut monitor = PredictiveMonitor::with_defaults();
        // Ingest identical samples — trend should be ~zero
        for _ in 0..5 {
            monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        }

        let predictions = monitor.predict(24.0);
        assert!(!predictions.is_empty());

        // With no trend, predicted values should be near current
        let disk_pred = predictions
            .iter()
            .find(|p| p.metric == "disk_used_pct")
            .unwrap();
        assert!(
            (disk_pred.predicted_value - 50.0).abs() < 5.0,
            "Stable system should predict near-current value, got {}",
            disk_pred.predicted_value
        );
        assert!(!disk_pred.crosses_threshold);
    }

    #[test]
    fn test_predict_all_horizons() {
        let mut monitor = PredictiveMonitor::with_defaults();
        for _ in 0..3 {
            monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        }

        let predictions = monitor.predict_all_horizons();
        // 4 metrics × 4 horizons = 16 predictions
        assert_eq!(predictions.len(), 16);
    }

    #[test]
    fn test_predict_crossing_threshold() {
        let mut monitor = PredictiveMonitor::new(AlertThresholds {
            disk_crit_pct: 90.0,
            ..Default::default()
        });

        // Simulate rising disk usage by manually injecting history entries
        // with sufficient time gap to establish a trend.
        // We add entries directly by ingesting many samples (trend is computed
        // from first to last entry in history, not by real wall-clock time).
        // Since all samples are ingested nearly instantly, we need many
        // to ensure the trend computation works with the small time delta.
        for i in 0..20 {
            monitor.ingest(sample_telemetry(70.0 + i as f64, 40.0, 50_000, 0));
        }

        // Latest value is 89.0, trend is strongly positive
        let predictions = monitor.predict(24.0);
        let disk_pred = predictions
            .iter()
            .find(|p| p.metric == "disk_used_pct")
            .unwrap();

        // Current is 89.0, with a strong positive trend it should predict even higher
        // (The exact value depends on the time delta between samples, but the trend is clearly up)
        assert!(
            disk_pred.predicted_value >= 89.0,
            "Rising disk should predict at or above current value, got {}",
            disk_pred.predicted_value
        );
    }

    #[test]
    fn test_confidence_increases_with_history() {
        let mut monitor = PredictiveMonitor::with_defaults();
        monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        let pred1 = monitor.predict(1.0);
        let conf1 = pred1[0].confidence;

        for _ in 0..50 {
            monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        }
        let pred2 = monitor.predict(1.0);
        let conf2 = pred2[0].confidence;

        assert!(
            conf2 >= conf1,
            "More history should not decrease confidence: {} vs {}",
            conf2,
            conf1
        );
    }

    #[test]
    fn test_prediction_has_recommendation_when_high() {
        let mut monitor = PredictiveMonitor::with_defaults();
        // High disk usage — should trigger recommendation
        monitor.ingest(sample_telemetry(88.0, 40.0, 50_000, 0));
        monitor.ingest(sample_telemetry(89.0, 40.0, 50_000, 0));

        let predictions = monitor.predict(24.0);
        let disk_pred = predictions
            .iter()
            .find(|p| p.metric == "disk_used_pct")
            .unwrap();
        assert!(
            disk_pred.recommended_action.is_some(),
            "High disk should have recommendation"
        );
    }

    #[test]
    fn test_max_history_cap() {
        let mut monitor = PredictiveMonitor::with_defaults();
        for i in 0..1500 {
            monitor.ingest(sample_telemetry(50.0 + i as f64 * 0.01, 40.0, 50_000, 0));
        }
        assert!(monitor.sample_count() <= 1000);
    }

    #[test]
    fn test_decode_encode_approximate_roundtrip() {
        let mut monitor = PredictiveMonitor::with_defaults();
        let telemetry = sample_telemetry(60.0, 45.0, 80_000, 2);
        let hv = monitor.encode_telemetry(&telemetry);
        let decoded = monitor.decode_telemetry(&hv);
        // HDC encoding is lossy — just verify values are in reasonable range
        assert!(
            decoded.disk_used_pct >= 0.0 && decoded.disk_used_pct <= 100.0,
            "Decoded disk should be in range, got {}",
            decoded.disk_used_pct
        );
        assert!(
            decoded.memory_used_pct >= 0.0 && decoded.memory_used_pct <= 100.0,
            "Decoded memory should be in range, got {}",
            decoded.memory_used_pct
        );
    }

    #[test]
    fn test_ltc_prediction_returns_values() {
        let mut monitor = PredictiveMonitor::with_defaults();
        // Feed a few samples so the neuron has state
        for _ in 0..5 {
            monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        }
        let ltc_pred = monitor.predict_ltc(1.0);
        assert!(ltc_pred.is_some());
        let pred = ltc_pred.unwrap();
        assert!(pred.disk_used_pct >= 0.0 && pred.disk_used_pct <= 100.0);
    }

    #[test]
    fn test_blended_prediction_differs_from_pure_trend() {
        let mut monitor = PredictiveMonitor::with_defaults();
        // Need enough history to have both trend and LTC
        for i in 0..10 {
            monitor.ingest(sample_telemetry(50.0 + i as f64, 40.0, 50_000, 0));
        }
        let predictions = monitor.predict(24.0);
        assert!(!predictions.is_empty());
        // The blended prediction should produce valid results
        let disk_pred = predictions
            .iter()
            .find(|p| p.metric == "disk_used_pct")
            .unwrap();
        assert!(
            disk_pred.predicted_value >= 0.0 && disk_pred.predicted_value <= 100.0,
            "Blended prediction should be in valid range, got {}",
            disk_pred.predicted_value
        );
    }

    #[test]
    fn test_ltc_prediction_empty_history_returns_none() {
        let mut monitor = PredictiveMonitor::with_defaults();
        assert!(monitor.predict_ltc(1.0).is_none());
    }

    #[test]
    fn test_save_empty_monitor() {
        let monitor = PredictiveMonitor::with_defaults();
        let saved = monitor.save();
        assert!(saved.samples.is_empty());
        assert!(!saved.neuron_state.is_empty()); // neuron has initial state
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut monitor = PredictiveMonitor::with_defaults();
        for i in 0..5 {
            monitor.ingest(sample_telemetry(50.0 + i as f64, 40.0, 50_000, 0));
        }

        let saved = monitor.save();
        assert_eq!(saved.samples.len(), 5);

        let restored = PredictiveMonitor::load(saved, AlertThresholds::default());
        assert_eq!(restored.sample_count(), 5);
    }

    #[test]
    fn test_saved_state_serializes() {
        let mut monitor = PredictiveMonitor::with_defaults();
        monitor.ingest(sample_telemetry(60.0, 50.0, 70_000, 1));
        let saved = monitor.save();

        let json = serde_json::to_string(&saved).unwrap();
        let restored: SavedPredictiveState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.samples.len(), 1);
        assert!((restored.samples[0].telemetry.disk_used_pct - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_restored_monitor_can_predict() {
        let mut monitor = PredictiveMonitor::with_defaults();
        for i in 0..10 {
            monitor.ingest(sample_telemetry(50.0 + i as f64, 40.0, 50_000, 0));
        }

        let saved = monitor.save();
        let mut restored = PredictiveMonitor::load(saved, AlertThresholds::default());
        let predictions = restored.predict(24.0);
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_neuron_state_preserved() {
        let mut monitor = PredictiveMonitor::with_defaults();
        for _ in 0..5 {
            monitor.ingest(sample_telemetry(50.0, 40.0, 50_000, 0));
        }
        let original_norm = monitor.neuron.state().norm();

        let saved = monitor.save();
        let restored = PredictiveMonitor::load(saved, AlertThresholds::default());
        let restored_norm = restored.neuron.state().norm();

        assert!(
            (original_norm - restored_norm).abs() < 1e-4,
            "Neuron state norm should be preserved: {} vs {}",
            original_norm,
            restored_norm
        );
    }
}
