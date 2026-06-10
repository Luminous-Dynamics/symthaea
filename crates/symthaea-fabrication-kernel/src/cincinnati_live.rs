// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cincinnati in-situ monitoring for additive manufacturing.
//!
//! Processes real-time sensor readings from a Cincinnati BAAM (Big Area
//! Additive Manufacturing) or similar large-format printer, detects
//! anomalies via online Welford statistics + z-score thresholding, and
//! bridges to the HDC manufacturing digital twin.

use crate::manufacturing::ManufacturingReading;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Maximum number of sensor readings retained in the history ring buffer.
const HISTORY_CAP: usize = 2048;

/// EMA smoothing factor for baseline updates.
const BASELINE_EMA_ALPHA: f64 = 0.05;

/// Minimum number of readings before anomaly detection activates.
const MIN_READINGS_FOR_DETECTION: u64 = 10;

// ── Configuration ────────────────────────────────────────────────────────

/// Configuration for the Cincinnati in-situ monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiMonitorConfig {
    /// Polling interval in milliseconds.
    pub poll_interval_ms: u64,
    /// Z-score threshold above which a reading is flagged as anomalous.
    pub anomaly_threshold: f32,
    /// Named sensor channels to monitor.
    pub sensor_channels: Vec<String>,
}

impl Default for CincinnatiMonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 100,
            anomaly_threshold: 3.0,
            sensor_channels: vec![
                "temperature_nozzle".into(),
                "temperature_bed".into(),
                "vibration_x".into(),
                "extrusion_pressure".into(),
            ],
        }
    }
}

// ── Sensor reading ───────────────────────────────────────────────────────

/// A single sensor reading from one channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Channel name (e.g., "temperature_nozzle").
    pub channel: String,
    /// Measured value.
    pub value: f64,
    /// Timestamp in milliseconds since epoch or session start.
    pub timestamp_ms: u64,
}

// ── Anomaly types ────────────────────────────────────────────────────────

/// Classification of detected anomalies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    Warping,
    /// Unexpected temperature spike or drop.
    TemperatureSpike,
    /// Inter-layer adhesion failure signature.
    LayerDelamination,
    /// Extrusion flow irregularity.
    ExtrusionAnomaly,
    /// Mechanical vibration exceeding normal range.
    VibrationExcess,
    /// Unclassified anomaly from a custom channel.
    Custom(String),
}

/// An alert raised when a sensor reading exceeds the anomaly threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    /// The channel that triggered the alert.
    pub channel: String,
    /// Severity score in [0.0, 1.0].
    pub severity: f32,
    /// Computed z-score of the anomalous reading.
    pub z_score: f64,
    /// Classification of the anomaly.
    pub anomaly_type: AnomalyType,
}

// ── Online statistics (Welford) ──────────────────────────────────────────

/// Online (streaming) statistics using Welford's algorithm.
///
/// Tracks count, running mean, and running M2 (sum of squared deviations)
/// for numerically stable variance computation without storing all readings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    /// Number of readings ingested.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running M2 (sum of squared deviations from the current mean).
    pub m2: f64,
}

impl ChannelStats {
    /// Create a new empty statistics tracker.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update statistics with a new value (Welford's online algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (returns 0.0 if fewer than 2 samples).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self::new()
    }
}

// ── Monitor ──────────────────────────────────────────────────────────────

/// Cincinnati in-situ monitor: ingests sensor readings, maintains online
/// statistics per channel, detects anomalies via z-score, and bridges
/// to the HDC manufacturing twin.
#[derive(Debug, Clone)]
pub struct CincinnatiMonitor {
    /// Monitor configuration.
    config: CincinnatiMonitorConfig,
    /// EMA baselines per channel (smoothed expected value).
    baselines: HashMap<String, f64>,
    /// Online statistics per channel (Welford).
    channel_stats: HashMap<String, ChannelStats>,
    /// Ring buffer of recent readings (capped at `HISTORY_CAP`).
    history: VecDeque<SensorReading>,
    /// Total number of anomalies detected.
    anomaly_count: u64,
}

impl CincinnatiMonitor {
    /// Create a new monitor from configuration.
    pub fn new(config: CincinnatiMonitorConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
            channel_stats: HashMap::new(),
            history: VecDeque::new(),
            anomaly_count: 0,
        }
    }

    /// Ingest a sensor reading: update stats, check for anomaly, store in history.
    ///
    /// Returns `Some(AnomalyAlert)` if the reading is anomalous, `None` otherwise.
    pub fn ingest_reading(&mut self, reading: SensorReading) -> Option<AnomalyAlert> {
        // Update online statistics.
        let stats = self
            .channel_stats
            .entry(reading.channel.clone())
            .or_default();
        stats.update(reading.value);

        // Update EMA baseline.
        self.update_baseline(&reading.channel, reading.value);

        // Detect anomaly.
        let alert = self.detect_anomaly(&reading);
        if alert.is_some() {
            self.anomaly_count += 1;
        }

        // Store in history (ring buffer).
        self.history.push_back(reading);
        if self.history.len() > HISTORY_CAP {
            self.history.pop_front();
        }

        alert
    }

    /// Detect whether a reading is anomalous based on z-score.
    ///
    /// Returns `None` if fewer than `MIN_READINGS_FOR_DETECTION` samples
    /// have been observed for the channel (insufficient data for reliable
    /// z-score computation).
    pub fn detect_anomaly(&self, reading: &SensorReading) -> Option<AnomalyAlert> {
        let stats = self.channel_stats.get(&reading.channel)?;

        // Guard: need minimum readings for reliable statistics.
        if stats.count < MIN_READINGS_FOR_DETECTION {
            return None;
        }

        let std_dev = stats.std_dev();
        if std_dev < 1e-12 {
            // All readings identical — any deviation is anomalous, but we
            // can't compute a meaningful z-score. Skip.
            return None;
        }

        let z_score = (reading.value - stats.mean()) / std_dev;
        let threshold = self.config.anomaly_threshold as f64;

        if z_score.abs() > threshold {
            let anomaly_type = classify_channel(&reading.channel);
            let severity =
                (z_score.abs() as f32 / (self.config.anomaly_threshold * 2.0)).clamp(0.0, 1.0);

            Some(AnomalyAlert {
                channel: reading.channel.clone(),
                severity,
                z_score,
                anomaly_type,
            })
        } else {
            None
        }
    }

    /// Update the EMA baseline for a channel.
    pub fn update_baseline(&mut self, channel: &str, value: f64) {
        let entry = self.baselines.entry(channel.to_string()).or_insert(value);
        *entry = *entry * (1.0 - BASELINE_EMA_ALPHA) + value * BASELINE_EMA_ALPHA;
    }

    /// Convert current baselines into a `ManufacturingReading`.
    ///
    /// Maps baseline values to the [0, 1] range expected by the manufacturing
    /// digital twin. Channels not present default to 0.5 (neutral).
    pub fn to_manufacturing_reading(&self) -> ManufacturingReading {
        let get = |key: &str| -> f64 {
            self.baselines
                .get(key)
                .copied()
                .unwrap_or(0.5)
                .clamp(0.0, 1.0)
        };

        ManufacturingReading {
            tolerance: get("tolerance"),
            surface_quality: get("surface_quality"),
            throughput: get("throughput"),
            energy_cost: get("energy_cost"),
        }
    }

    /// Total number of anomalies detected since creation.
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Simulate a real-time hardware failure (e.g., extruder slip detected via acoustic entropy).
    pub fn simulate_acoustic_failure(&self) -> AnomalyAlert {
        AnomalyAlert {
            channel: "acoustic_emission".into(),
            anomaly_type: AnomalyType::Warping, // Proxy for delamination
            severity: 0.95,
            z_score: 15.0,
        }
    }

    /// Current history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Access the monitor configuration.
    pub fn config(&self) -> &CincinnatiMonitorConfig {
        &self.config
    }

    /// Access per-channel statistics.
    pub fn channel_stats(&self) -> &HashMap<String, ChannelStats> {
        &self.channel_stats
    }

    /// Access current baselines.
    pub fn baselines(&self) -> &HashMap<String, f64> {
        &self.baselines
    }
}

// ── Anomaly classification ───────────────────────────────────────────────

/// Classify an anomaly type based on the channel name.
fn classify_channel(channel: &str) -> AnomalyType {
    let lower = channel.to_lowercase();
    if lower.contains("temperature") {
        AnomalyType::TemperatureSpike
    } else if lower.contains("vibration") {
        AnomalyType::VibrationExcess
    } else if lower.contains("extrusion") {
        AnomalyType::ExtrusionAnomaly
    } else if lower.contains("delamination") {
        AnomalyType::LayerDelamination
    } else {
        AnomalyType::Custom(channel.to_string())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CincinnatiMonitorConfig {
        CincinnatiMonitorConfig {
            poll_interval_ms: 100,
            anomaly_threshold: 3.0,
            sensor_channels: vec!["temperature_nozzle".into()],
        }
    }

    fn make_reading(channel: &str, value: f64, ts: u64) -> SensorReading {
        SensorReading {
            channel: channel.to_string(),
            value,
            timestamp_ms: ts,
        }
    }

    /// Feed `n` stable readings to build up statistics.
    fn feed_stable(monitor: &mut CincinnatiMonitor, channel: &str, value: f64, n: u64) {
        for i in 0..n {
            let r = make_reading(channel, value + (i as f64 * 0.001), i * 100);
            let alert = monitor.ingest_reading(r);
            assert!(
                alert.is_none(),
                "stable reading {} should not trigger alert",
                i
            );
        }
    }

    #[test]
    fn baseline_learning() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        // First reading initializes the baseline.
        monitor.update_baseline("temp", 200.0);
        let b1 = monitor.baselines["temp"];
        assert!(
            (b1 - 200.0).abs() < 1e-6,
            "first baseline should be the initial value"
        );

        // Subsequent readings apply EMA smoothing.
        monitor.update_baseline("temp", 210.0);
        let b2 = monitor.baselines["temp"];
        // EMA: 200 * 0.95 + 210 * 0.05 = 190 + 10.5 = 200.5
        assert!(
            (b2 - 200.5).abs() < 1e-6,
            "EMA baseline should be ~200.5, got {}",
            b2
        );
    }

    #[test]
    fn stable_readings_no_false_alarm() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        // Feed 50 readings with very small variance.
        for i in 0..50 {
            let r = make_reading("temperature_nozzle", 210.0 + (i as f64 * 0.01), i * 100);
            let alert = monitor.ingest_reading(r);
            assert!(
                alert.is_none(),
                "stable reading {} triggered false alarm",
                i
            );
        }
        assert_eq!(monitor.anomaly_count(), 0);
    }

    #[test]
    fn spike_detection() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        // Build baseline with stable readings.
        feed_stable(&mut monitor, "temperature_nozzle", 210.0, 20);

        // Inject a massive spike.
        let spike = make_reading("temperature_nozzle", 500.0, 5000);
        let alert = monitor.ingest_reading(spike);
        assert!(alert.is_some(), "massive spike should trigger an alert");
        let alert = alert.unwrap();
        assert_eq!(alert.anomaly_type, AnomalyType::TemperatureSpike);
        assert!(alert.z_score > 3.0, "z_score should exceed threshold");
        assert!(alert.severity > 0.0, "severity should be positive");
    }

    #[test]
    fn z_score_threshold_boundary() {
        let mut monitor = CincinnatiMonitor::new(CincinnatiMonitorConfig {
            anomaly_threshold: 2.0,
            ..default_config()
        });

        // Feed identical readings to get zero variance, then slightly different
        // ones to build real variance.
        for i in 0..30 {
            // Alternate between 100.0 and 101.0 to create std_dev ~ 0.5.
            let val = if i % 2 == 0 { 100.0 } else { 101.0 };
            monitor.ingest_reading(make_reading("sensor", val, i * 100));
        }

        let stats = &monitor.channel_stats["sensor"];
        let sd = stats.std_dev();
        assert!(sd > 0.0, "should have nonzero std dev");

        // A reading at mean + 1.5 * sd should NOT trigger (threshold = 2.0).
        let below = make_reading("sensor", stats.mean() + 1.5 * sd, 10000);
        let alert = monitor.detect_anomaly(&below);
        assert!(alert.is_none(), "below-threshold reading should not alert");

        // A reading at mean + 3.0 * sd SHOULD trigger.
        let above = make_reading("sensor", stats.mean() + 3.0 * sd, 10001);
        let alert = monitor.detect_anomaly(&above);
        assert!(alert.is_some(), "above-threshold reading should alert");
    }

    #[test]
    fn min_readings_guard() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        // Feed only 5 readings (below MIN_READINGS_FOR_DETECTION = 10).
        for i in 0..5 {
            monitor.ingest_reading(make_reading("ch", 100.0, i * 100));
        }

        // Even a wild outlier should not trigger because we lack data.
        let spike = make_reading("ch", 99999.0, 9999);
        let alert = monitor.detect_anomaly(&spike);
        assert!(
            alert.is_none(),
            "should not detect anomaly with fewer than {} readings",
            MIN_READINGS_FOR_DETECTION
        );
    }

    #[test]
    fn history_cap() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        for i in 0..(HISTORY_CAP + 500) as u64 {
            monitor.ingest_reading(make_reading("ch", 100.0, i));
        }
        assert_eq!(
            monitor.history_len(),
            HISTORY_CAP,
            "history should be capped at {}",
            HISTORY_CAP
        );
    }

    #[test]
    fn to_manufacturing_reading_defaults() {
        let monitor = CincinnatiMonitor::new(default_config());
        let reading = monitor.to_manufacturing_reading();
        // No baselines set — everything should default to 0.5.
        assert!(
            (reading.tolerance - 0.5).abs() < 1e-6,
            "default tolerance={}",
            reading.tolerance
        );
        assert!(
            (reading.surface_quality - 0.5).abs() < 1e-6,
            "default surface_quality={}",
            reading.surface_quality
        );
        assert!(
            (reading.throughput - 0.5).abs() < 1e-6,
            "default throughput={}",
            reading.throughput
        );
        assert!(
            (reading.energy_cost - 0.5).abs() < 1e-6,
            "default energy_cost={}",
            reading.energy_cost
        );
    }

    #[test]
    fn to_manufacturing_reading_with_baselines() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        monitor.update_baseline("tolerance", 0.95);
        monitor.update_baseline("surface_quality", 0.8);
        monitor.update_baseline("throughput", 0.7);
        monitor.update_baseline("energy_cost", 0.3);

        let reading = monitor.to_manufacturing_reading();
        assert!(
            (reading.tolerance - 0.95).abs() < 1e-6,
            "tolerance={}",
            reading.tolerance
        );
        assert!(
            (reading.surface_quality - 0.8).abs() < 1e-6,
            "surface_quality={}",
            reading.surface_quality
        );
        assert!(
            (reading.throughput - 0.7).abs() < 1e-6,
            "throughput={}",
            reading.throughput
        );
        assert!(
            (reading.energy_cost - 0.3).abs() < 1e-6,
            "energy_cost={}",
            reading.energy_cost
        );
    }

    #[test]
    fn anomaly_classification() {
        assert_eq!(
            classify_channel("temperature_nozzle"),
            AnomalyType::TemperatureSpike
        );
        assert_eq!(
            classify_channel("TEMPERATURE_BED"),
            AnomalyType::TemperatureSpike
        );
        assert_eq!(
            classify_channel("vibration_x"),
            AnomalyType::VibrationExcess
        );
        assert_eq!(
            classify_channel("extrusion_pressure"),
            AnomalyType::ExtrusionAnomaly
        );
        assert_eq!(
            classify_channel("delamination_sensor"),
            AnomalyType::LayerDelamination
        );
        assert_eq!(
            classify_channel("humidity"),
            AnomalyType::Custom("humidity".into())
        );
    }

    #[test]
    fn severity_scaling() {
        // Severity = (|z_score| / (threshold * 2)).clamp(0, 1)
        // With threshold=3.0: z=3.0 → severity = 3/6 = 0.5
        //                      z=6.0 → severity = 6/6 = 1.0
        //                      z=9.0 → severity = 9/6 = 1.0 (clamped)
        let mut monitor = CincinnatiMonitor::new(default_config());
        feed_stable(&mut monitor, "temperature_nozzle", 100.0, 30);

        let stats = &monitor.channel_stats["temperature_nozzle"];
        let sd = stats.std_dev();
        let mean = stats.mean();

        // Spike at exactly 3 * sd above mean.
        let reading_3sd = make_reading("temperature_nozzle", mean + 3.5 * sd, 99999);
        if let Some(alert) = monitor.detect_anomaly(&reading_3sd) {
            assert!(
                alert.severity > 0.0 && alert.severity <= 1.0,
                "severity should be in (0, 1], got {}",
                alert.severity
            );
        }

        // Very large spike should clamp severity to 1.0.
        let reading_huge = make_reading("temperature_nozzle", mean + 100.0 * sd, 99999);
        if let Some(alert) = monitor.detect_anomaly(&reading_huge) {
            assert!(
                (alert.severity - 1.0).abs() < 1e-4,
                "very large spike severity should be ~1.0, got {}",
                alert.severity
            );
        }
    }

    #[test]
    fn welford_stats_correctness() {
        let mut stats = ChannelStats::new();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for &v in &values {
            stats.update(v);
        }
        assert_eq!(stats.count, 8);
        // Mean = 40/8 = 5.0
        assert!(
            (stats.mean() - 5.0).abs() < 1e-10,
            "mean={}, expected 5.0",
            stats.mean()
        );
        // Population variance = 4.0
        assert!(
            (stats.variance() - 4.0).abs() < 1e-10,
            "variance={}, expected 4.0",
            stats.variance()
        );
        // Std dev = 2.0
        assert!(
            (stats.std_dev() - 2.0).abs() < 1e-10,
            "std_dev={}, expected 2.0",
            stats.std_dev()
        );
    }

    #[test]
    fn multiple_channels_independent() {
        let mut monitor = CincinnatiMonitor::new(default_config());
        feed_stable(&mut monitor, "temperature_nozzle", 210.0, 20);
        feed_stable(&mut monitor, "vibration_x", 0.5, 20);

        assert_eq!(monitor.channel_stats().len(), 2);
        let temp_mean = monitor.channel_stats["temperature_nozzle"].mean();
        let vib_mean = monitor.channel_stats["vibration_x"].mean();
        assert!(
            (temp_mean - 210.0).abs() < 1.0,
            "temp mean ~210, got {}",
            temp_mean
        );
        assert!(
            (vib_mean - 0.5).abs() < 0.1,
            "vib mean ~0.5, got {}",
            vib_mean
        );
    }
}
