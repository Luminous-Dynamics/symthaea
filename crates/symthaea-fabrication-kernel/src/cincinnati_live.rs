//! Live Cincinnati monitoring — sensor polling and anomaly detection.
//!
//! Ingests sensor readings from a Cincinnati large-area additive
//! manufacturing system, maintains per-channel baselines via exponential
//! moving average (EMA), and generates anomaly alerts when Z-scores
//! exceed configurable thresholds. Sensor state can be mapped to a
//! [`ManufacturingReading`] for integration with the HDC prediction
//! pipeline.

use crate::manufacturing::ManufacturingReading;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ── Constants ───────────────────────────────────────────────────────────

/// Default poll interval in milliseconds.
const DEFAULT_POLL_INTERVAL_MS: u64 = 250;

/// Default anomaly Z-score threshold.
const DEFAULT_ANOMALY_THRESHOLD: f32 = 3.0;

/// Maximum number of readings retained in the rolling history window.
const MAX_HISTORY_LEN: usize = 2048;

/// EMA smoothing factor for baseline learning.
const BASELINE_EMA_ALPHA: f64 = 0.05;

/// Minimum number of readings before anomaly detection activates per channel.
const MIN_READINGS_FOR_DETECTION: usize = 10;

// ── Config ──────────────────────────────────────────────────────────────

/// Configuration for the Cincinnati live monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiMonitorConfig {
    /// Sensor polling interval in milliseconds.
    pub poll_interval_ms: u64,
    /// Z-score threshold above which an anomaly alert fires.
    pub anomaly_threshold: f32,
    /// Names of sensor channels to monitor.
    pub sensor_channels: Vec<String>,
}

impl Default for CincinnatiMonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            anomaly_threshold: DEFAULT_ANOMALY_THRESHOLD,
            sensor_channels: vec![
                "nozzle_temp".into(),
                "bed_temp".into(),
                "extrusion_rate".into(),
                "layer_height".into(),
                "vibration".into(),
            ],
        }
    }
}

// ── Sensor Reading ──────────────────────────────────────────────────────

/// A single sensor sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Channel name (must match one of `CincinnatiMonitorConfig.sensor_channels`).
    pub channel: String,
    /// Measured value (units depend on channel).
    pub value: f64,
    /// Timestamp in milliseconds since epoch (or monotonic origin).
    pub timestamp_ms: u64,
}

// ── Anomaly Types ───────────────────────────────────────────────────────

/// Classification of detected anomalies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Rapid temperature excursion.
    TemperatureSpike,
    /// Inter-layer adhesion failure signature.
    LayerDelamination,
    /// Extrusion flow rate deviation.
    ExtrusionAnomaly,
    /// Mechanical vibration beyond tolerance.
    VibrationExcess,
    /// Application-defined anomaly.
    Custom(String),
}

/// Alert generated when a sensor reading exceeds the anomaly threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    /// Which sensor channel triggered the alert.
    pub channel: String,
    /// Severity in [0.0, 1.0] — higher is worse.
    pub severity: f32,
    /// Absolute Z-score of the triggering reading.
    pub z_score: f64,
    /// Classification of the anomaly.
    pub anomaly_type: AnomalyType,
}

// ── Per-Channel Statistics ──────────────────────────────────────────────

/// Running statistics for a single sensor channel.
#[derive(Debug, Clone)]
struct ChannelStats {
    /// EMA baseline (mean estimate).
    mean: f64,
    /// EMA variance estimate (Welford-style, single-pass).
    variance: f64,
    /// Number of readings ingested.
    count: usize,
}

impl ChannelStats {
    fn new(initial_value: f64) -> Self {
        Self {
            mean: initial_value,
            variance: 0.0,
            count: 1,
        }
    }

    /// Update the running mean and variance with a new observation.
    fn update(&mut self, value: f64) {
        self.count += 1;
        let diff = value - self.mean;
        self.mean += BASELINE_EMA_ALPHA * diff;
        // Exponentially-weighted variance update.
        self.variance =
            (1.0 - BASELINE_EMA_ALPHA) * (self.variance + BASELINE_EMA_ALPHA * diff * diff);
    }

    /// Standard deviation (floored to avoid division by zero).
    fn stddev(&self) -> f64 {
        self.variance.sqrt().max(1e-12)
    }

    /// Compute the Z-score for a given value.
    fn z_score(&self, value: f64) -> f64 {
        (value - self.mean).abs() / self.stddev()
    }

    /// Whether we have enough data for reliable anomaly detection.
    fn ready(&self) -> bool {
        self.count >= MIN_READINGS_FOR_DETECTION
    }
}

// ── Monitor ─────────────────────────────────────────────────────────────

/// Live Cincinnati additive manufacturing monitor.
///
/// Maintains per-channel baselines and a rolling window of recent
/// readings. Each ingested reading is checked against the learned
/// baseline; if the Z-score exceeds the configured threshold, an
/// [`AnomalyAlert`] is returned.
pub struct CincinnatiMonitor {
    /// Monitor configuration.
    config: CincinnatiMonitorConfig,
    /// Learned baselines per channel.
    baseline: HashMap<String, f64>,
    /// Rolling window of recent readings (all channels interleaved).
    reading_history: VecDeque<SensorReading>,
    /// Per-channel running statistics.
    channel_stats: HashMap<String, ChannelStats>,
    /// Total anomaly alerts generated.
    pub anomaly_count: u64,
}

impl CincinnatiMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: CincinnatiMonitorConfig) -> Self {
        Self {
            baseline: HashMap::with_capacity(config.sensor_channels.len()),
            reading_history: VecDeque::with_capacity(MAX_HISTORY_LEN),
            channel_stats: HashMap::with_capacity(config.sensor_channels.len()),
            anomaly_count: 0,
            config,
        }
    }

    /// Ingest a sensor reading: update baseline, check for anomalies,
    /// and store in the rolling history.
    ///
    /// Returns `Some(AnomalyAlert)` if the reading exceeds the
    /// configured threshold.
    pub fn ingest_reading(&mut self, reading: SensorReading) -> Option<AnomalyAlert> {
        // Update baseline and stats.
        self.update_baseline(&reading.channel, reading.value);

        // Check for anomaly before pushing to history.
        let alert = self.detect_anomaly(&reading);
        if alert.is_some() {
            self.anomaly_count += 1;
        }

        // Push to rolling window.
        if self.reading_history.len() >= MAX_HISTORY_LEN {
            self.reading_history.pop_front();
        }
        self.reading_history.push_back(reading);

        alert
    }

    /// Detect whether a reading constitutes an anomaly based on
    /// Z-score deviation from the learned baseline.
    pub fn detect_anomaly(&self, reading: &SensorReading) -> Option<AnomalyAlert> {
        let stats = self.channel_stats.get(&reading.channel)?;

        // Don't alert until we have enough data for a reliable baseline.
        if !stats.ready() {
            return None;
        }

        let z = stats.z_score(reading.value);
        if z < self.config.anomaly_threshold as f64 {
            return None;
        }

        // Severity: maps Z-score range [threshold, threshold+6] → [0.0, 1.0].
        let threshold = self.config.anomaly_threshold as f64;
        let severity = ((z - threshold) / 6.0).clamp(0.0, 1.0) as f32;

        let anomaly_type = classify_anomaly(&reading.channel, z);

        Some(AnomalyAlert {
            channel: reading.channel.clone(),
            severity,
            z_score: z,
            anomaly_type,
        })
    }

    /// Update the EMA baseline for a channel.
    pub fn update_baseline(&mut self, channel: &str, value: f64) {
        match self.channel_stats.get_mut(channel) {
            Some(stats) => {
                stats.update(value);
                self.baseline.insert(channel.to_string(), stats.mean);
            }
            None => {
                let stats = ChannelStats::new(value);
                self.baseline.insert(channel.to_string(), value);
                self.channel_stats.insert(channel.to_string(), stats);
            }
        }
    }

    /// Map current sensor state to a [`ManufacturingReading`] for the
    /// HDC prediction pipeline.
    ///
    /// Channel mapping (by convention):
    /// - `"extrusion_rate"` → tolerance (extrusion consistency ≈ dimensional accuracy)
    /// - `"layer_height"` → surface_quality (layer uniformity)
    /// - `"nozzle_temp"` → throughput (thermal stability enables speed)
    /// - `"bed_temp"` → energy_cost (bed heating dominates energy budget)
    ///
    /// Missing channels default to 0.5.
    pub fn to_manufacturing_reading(&self) -> ManufacturingReading {
        let get = |ch: &str| -> f64 {
            self.baseline.get(ch).copied().unwrap_or(0.5)
        };

        // Normalize to [0, 1] — assume baselines are already in
        // reasonable ranges; clamp for safety.
        let tolerance = get("extrusion_rate").clamp(0.0, 1.0);
        let surface_quality = get("layer_height").clamp(0.0, 1.0);
        let throughput = get("nozzle_temp").clamp(0.0, 1.0);
        let energy_cost = get("bed_temp").clamp(0.0, 1.0);

        ManufacturingReading {
            tolerance,
            surface_quality,
            throughput,
            energy_cost,
        }
    }

    /// Number of readings in the rolling history window.
    pub fn history_len(&self) -> usize {
        self.reading_history.len()
    }

    /// Current baseline value for a channel, if known.
    pub fn baseline_for(&self, channel: &str) -> Option<f64> {
        self.baseline.get(channel).copied()
    }

    /// Number of distinct channels with learned baselines.
    pub fn tracked_channels(&self) -> usize {
        self.channel_stats.len()
    }

    /// Access the monitor configuration.
    pub fn config(&self) -> &CincinnatiMonitorConfig {
        &self.config
    }

    /// Access the full reading history.
    pub fn reading_history(&self) -> &VecDeque<SensorReading> {
        &self.reading_history
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Classify an anomaly based on channel name and Z-score magnitude.
fn classify_anomaly(channel: &str, _z_score: f64) -> AnomalyType {
    match channel {
        c if c.contains("temp") => AnomalyType::TemperatureSpike,
        c if c.contains("layer") => AnomalyType::LayerDelamination,
        c if c.contains("extrusion") => AnomalyType::ExtrusionAnomaly,
        c if c.contains("vibration") => AnomalyType::VibrationExcess,
        _ => AnomalyType::Custom(format!("anomaly on channel '{channel}'")),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_monitor() -> CincinnatiMonitor {
        CincinnatiMonitor::new(CincinnatiMonitorConfig::default())
    }

    fn reading(channel: &str, value: f64, ts: u64) -> SensorReading {
        SensorReading {
            channel: channel.to_string(),
            value,
            timestamp_ms: ts,
        }
    }

    #[test]
    fn test_baseline_learning_single_channel() {
        let mut mon = default_monitor();
        for i in 0..20 {
            mon.ingest_reading(reading("nozzle_temp", 200.0, i));
        }
        let baseline = mon.baseline_for("nozzle_temp").unwrap();
        // After 20 identical readings, baseline should converge to 200.0.
        assert!(
            (baseline - 200.0).abs() < 1.0,
            "baseline={baseline}, expected ~200"
        );
    }

    #[test]
    fn test_baseline_learning_multiple_channels() {
        let mut mon = default_monitor();
        for i in 0..20 {
            mon.ingest_reading(reading("nozzle_temp", 200.0, i));
            mon.ingest_reading(reading("bed_temp", 60.0, i));
        }
        assert_eq!(mon.tracked_channels(), 2);
        assert!(mon.baseline_for("nozzle_temp").is_some());
        assert!(mon.baseline_for("bed_temp").is_some());
    }

    #[test]
    fn test_no_anomaly_on_stable_readings() {
        let mut mon = default_monitor();
        // Feed stable readings — no anomalies should fire.
        for i in 0..50 {
            let alert = mon.ingest_reading(reading("extrusion_rate", 0.8, i));
            assert!(alert.is_none(), "unexpected anomaly at reading {i}");
        }
        assert_eq!(mon.anomaly_count, 0);
    }

    #[test]
    fn test_anomaly_detection_spike() {
        let mut mon = default_monitor();
        // Build a stable baseline.
        for i in 0..30 {
            mon.ingest_reading(reading("nozzle_temp", 200.0, i));
        }
        // Inject a massive spike.
        let alert = mon.ingest_reading(reading("nozzle_temp", 500.0, 30));
        assert!(alert.is_some(), "expected anomaly alert on spike");

        let alert = alert.unwrap();
        assert_eq!(alert.channel, "nozzle_temp");
        assert!(alert.z_score > 3.0);
        assert_eq!(alert.anomaly_type, AnomalyType::TemperatureSpike);
        assert!(alert.severity >= 0.0 && alert.severity <= 1.0);
        assert_eq!(mon.anomaly_count, 1);
    }

    #[test]
    fn test_z_score_threshold_boundary() {
        let mut mon = CincinnatiMonitor::new(CincinnatiMonitorConfig {
            anomaly_threshold: 2.0,
            ..Default::default()
        });
        // Build baseline at 100.0 with low variance.
        for i in 0..30 {
            mon.ingest_reading(reading("vibration", 100.0, i));
        }

        // Small deviation — should NOT trigger with low threshold.
        let stats = mon.channel_stats.get("vibration").unwrap();
        let small_dev = stats.mean + stats.stddev() * 1.5;
        let alert = mon.detect_anomaly(&reading("vibration", small_dev, 31));
        assert!(alert.is_none(), "1.5σ should not trigger at threshold=2.0");

        // Larger deviation — SHOULD trigger.
        let big_dev = stats.mean + stats.stddev() * 2.5;
        let alert = mon.detect_anomaly(&reading("vibration", big_dev, 32));
        assert!(alert.is_some(), "2.5σ should trigger at threshold=2.0");
    }

    #[test]
    fn test_no_anomaly_before_min_readings() {
        let mut mon = default_monitor();
        // Only 5 readings — below MIN_READINGS_FOR_DETECTION.
        for i in 0..5 {
            mon.ingest_reading(reading("extrusion_rate", 0.5, i));
        }
        // Even a huge spike should not alert.
        let alert = mon.ingest_reading(reading("extrusion_rate", 100.0, 5));
        assert!(
            alert.is_none(),
            "should not alert before MIN_READINGS_FOR_DETECTION"
        );
    }

    #[test]
    fn test_reading_history_cap() {
        let mut mon = default_monitor();
        for i in 0..(MAX_HISTORY_LEN + 100) {
            mon.ingest_reading(reading("nozzle_temp", 200.0, i as u64));
        }
        assert!(
            mon.history_len() <= MAX_HISTORY_LEN,
            "history_len={} exceeds cap={}",
            mon.history_len(),
            MAX_HISTORY_LEN
        );
    }

    #[test]
    fn test_to_manufacturing_reading_defaults() {
        let mon = default_monitor();
        let mr = mon.to_manufacturing_reading();
        // No baselines learned — all channels default to 0.5.
        assert_eq!(mr.tolerance, 0.5);
        assert_eq!(mr.surface_quality, 0.5);
        assert_eq!(mr.throughput, 0.5);
        assert_eq!(mr.energy_cost, 0.5);
    }

    #[test]
    fn test_to_manufacturing_reading_with_baselines() {
        let mut mon = default_monitor();
        // Feed known values.
        for i in 0..30 {
            mon.ingest_reading(reading("extrusion_rate", 0.9, i));
            mon.ingest_reading(reading("layer_height", 0.85, i));
            mon.ingest_reading(reading("nozzle_temp", 0.7, i));
            mon.ingest_reading(reading("bed_temp", 0.3, i));
        }
        let mr = mon.to_manufacturing_reading();
        assert!((mr.tolerance - 0.9).abs() < 0.05, "tolerance={}", mr.tolerance);
        assert!(
            (mr.surface_quality - 0.85).abs() < 0.05,
            "surface_quality={}",
            mr.surface_quality
        );
        assert!((mr.throughput - 0.7).abs() < 0.05, "throughput={}", mr.throughput);
        assert!((mr.energy_cost - 0.3).abs() < 0.05, "energy_cost={}", mr.energy_cost);
    }

    #[test]
    fn test_anomaly_type_classification() {
        assert_eq!(classify_anomaly("nozzle_temp", 5.0), AnomalyType::TemperatureSpike);
        assert_eq!(classify_anomaly("bed_temp", 4.0), AnomalyType::TemperatureSpike);
        assert_eq!(classify_anomaly("layer_height", 3.5), AnomalyType::LayerDelamination);
        assert_eq!(classify_anomaly("extrusion_rate", 4.0), AnomalyType::ExtrusionAnomaly);
        assert_eq!(classify_anomaly("vibration", 3.0), AnomalyType::VibrationExcess);
        assert!(matches!(
            classify_anomaly("custom_sensor", 3.0),
            AnomalyType::Custom(_)
        ));
    }

    #[test]
    fn test_severity_scaling() {
        let mut mon = default_monitor();
        // Build tight baseline.
        for i in 0..30 {
            mon.ingest_reading(reading("nozzle_temp", 200.0, i));
        }
        // Moderate spike.
        let alert_mod = mon.detect_anomaly(&reading("nozzle_temp", 250.0, 30));
        // Extreme spike.
        let alert_ext = mon.detect_anomaly(&reading("nozzle_temp", 900.0, 31));

        if let (Some(a1), Some(a2)) = (alert_mod, alert_ext) {
            assert!(
                a2.severity >= a1.severity,
                "extreme spike severity ({}) should >= moderate ({})",
                a2.severity,
                a1.severity
            );
        }
    }

    #[test]
    fn test_config_accessors() {
        let config = CincinnatiMonitorConfig {
            poll_interval_ms: 500,
            anomaly_threshold: 4.0,
            sensor_channels: vec!["ch1".into(), "ch2".into()],
        };
        let mon = CincinnatiMonitor::new(config);
        assert_eq!(mon.config().poll_interval_ms, 500);
        assert_eq!(mon.config().anomaly_threshold, 4.0);
        assert_eq!(mon.config().sensor_channels.len(), 2);
    }

    #[test]
    fn test_channel_stats_z_score_symmetric() {
        let mut stats = ChannelStats::new(100.0);
        for _ in 0..50 {
            stats.update(100.0);
        }
        // Positive and negative deviations should produce the same Z-score.
        let z_pos = stats.z_score(105.0);
        let z_neg = stats.z_score(95.0);
        assert!(
            (z_pos - z_neg).abs() < 1e-6,
            "z_pos={z_pos}, z_neg={z_neg} should be equal"
        );
    }
}
