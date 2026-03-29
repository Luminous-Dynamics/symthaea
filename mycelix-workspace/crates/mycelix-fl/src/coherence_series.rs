// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/phi_series.rs at commit feat/fl-consolidation
//! Coherence Time-Series Tracking for Coherence Evolution
//!
//! **This tracks gradient coherence over time, NOT IIT Phi.**
//!
//! WASM-compatible time-series analysis of coherence measurements over federated
//! learning rounds, enabling:
//!
//! - Trend detection (rising, falling, stable, volatile)
//! - Anomaly detection for sudden drops (potential attacks)
//! - Statistical analysis over rolling windows
//! - JSON serialization for external consumption

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for coherence time-series tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceTimeSeriesConfig {
    /// Rolling window size for statistics calculation
    pub window_size: usize,
    /// Standard deviations threshold for anomaly detection
    pub anomaly_threshold: f32,
    /// Whether to store full history or only window
    pub store_full_history: bool,
}

impl Default for CoherenceTimeSeriesConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            anomaly_threshold: 2.5,
            store_full_history: true,
        }
    }
}

impl CoherenceTimeSeriesConfig {
    /// Create config with custom window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Create config with custom anomaly threshold
    pub fn with_anomaly_threshold(mut self, threshold: f32) -> Self {
        self.anomaly_threshold = threshold;
        self
    }

    /// Enable or disable full history storage
    pub fn with_full_history(mut self, store: bool) -> Self {
        self.store_full_history = store;
        self
    }
}

/// A single coherence measurement data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceDataPoint {
    /// Training round number
    pub round: u64,
    /// Timestamp (caller-provided, e.g. unix millis from host)
    pub timestamp: i64,
    /// Measured coherence value
    pub coherence_value: f32,
    /// Epistemic confidence in the measurement
    pub epistemic_confidence: f32,
    /// Number of Byzantine nodes detected this round
    pub byzantine_count: usize,
    /// Total number of nodes in the round
    pub node_count: usize,
    /// Optional metadata (layer-wise coherence, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, f32>>,
}

impl CoherenceDataPoint {
    /// Create a new data point with a caller-provided timestamp.
    pub fn new(
        round: u64,
        timestamp: i64,
        coherence_value: f32,
        epistemic_confidence: f32,
        byzantine_count: usize,
        node_count: usize,
    ) -> Self {
        Self {
            round,
            timestamp,
            coherence_value,
            epistemic_confidence,
            byzantine_count,
            node_count,
            metadata: None,
        }
    }

    /// Add metadata to the data point
    pub fn with_metadata(mut self, metadata: HashMap<String, f32>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Trend direction of coherence over time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoherenceTrend {
    /// Coherence is increasing (healthy learning)
    Rising { slope: f32 },
    /// Coherence is decreasing (potential attack/degradation)
    Falling { slope: f32 },
    /// Coherence is stable
    Stable { variance: f32 },
    /// High variance, unstable system
    Volatile { variance: f32 },
}

impl CoherenceTrend {
    /// Check if the trend indicates healthy learning
    pub fn is_healthy(&self) -> bool {
        matches!(self, CoherenceTrend::Rising { .. } | CoherenceTrend::Stable { .. })
    }

    /// Check if the trend indicates potential issues
    pub fn is_concerning(&self) -> bool {
        matches!(self, CoherenceTrend::Falling { .. } | CoherenceTrend::Volatile { .. })
    }

    /// Get the slope if applicable
    pub fn slope(&self) -> Option<f32> {
        match self {
            CoherenceTrend::Rising { slope } | CoherenceTrend::Falling { slope } => Some(*slope),
            _ => None,
        }
    }

    /// Get the variance if applicable
    pub fn variance(&self) -> Option<f32> {
        match self {
            CoherenceTrend::Stable { variance } | CoherenceTrend::Volatile { variance } => {
                Some(*variance)
            }
            _ => None,
        }
    }
}

/// Severity levels for anomalies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AnomalySeverity {
    /// Create severity from z-score
    pub fn from_z_score(z_score: f32) -> Self {
        let abs_z = z_score.abs();
        if abs_z >= 4.0 {
            AnomalySeverity::Critical
        } else if abs_z >= 3.0 {
            AnomalySeverity::High
        } else if abs_z >= 2.5 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
}

/// Detected anomaly in coherence measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceAnomaly {
    /// Round where anomaly was detected
    pub round: u64,
    /// Expected coherence value based on history
    pub expected_coherence: f32,
    /// Actual measured coherence value
    pub actual_coherence: f32,
    /// Z-score of the deviation
    pub z_score: f32,
    /// Severity of the anomaly
    pub severity: AnomalySeverity,
    /// Possible cause of the anomaly
    pub possible_cause: String,
}

impl CoherenceAnomaly {
    /// Create a new anomaly
    pub fn new(
        round: u64,
        expected_coherence: f32,
        actual_coherence: f32,
        z_score: f32,
        possible_cause: String,
    ) -> Self {
        Self {
            round,
            expected_coherence,
            actual_coherence,
            z_score,
            severity: AnomalySeverity::from_z_score(z_score),
            possible_cause,
        }
    }

    /// Check if this is a drop (negative deviation)
    pub fn is_drop(&self) -> bool {
        self.actual_coherence < self.expected_coherence
    }

    /// Check if this is a spike (positive deviation)
    pub fn is_spike(&self) -> bool {
        self.actual_coherence > self.expected_coherence
    }
}

/// Statistics computed over a window of coherence measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceStatistics {
    /// Mean coherence value over the window
    pub mean: f32,
    /// Standard deviation of coherence
    pub std_dev: f32,
    /// Minimum coherence value in window
    pub min: f32,
    /// Maximum coherence value in window
    pub max: f32,
    /// Trend slope (positive = increasing)
    pub trend_slope: f32,
    /// Autocorrelation with previous values (lag-1)
    pub autocorrelation: f32,
    /// Number of data points in the calculation
    pub sample_count: usize,
}

impl CoherenceStatistics {
    /// Create empty statistics
    pub fn empty() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            trend_slope: 0.0,
            autocorrelation: 0.0,
            sample_count: 0,
        }
    }

    /// Check if statistics are valid (have enough data)
    pub fn is_valid(&self) -> bool {
        self.sample_count >= 2
    }
}

/// Time-series tracker for coherence measurements.
#[derive(Debug)]
pub struct CoherenceTimeSeries {
    config: CoherenceTimeSeriesConfig,
    history: Vec<CoherenceDataPoint>,
    cached_stats: Option<CoherenceStatistics>,
    cached_trend: Option<CoherenceTrend>,
}

impl CoherenceTimeSeries {
    /// Create a new coherence time-series tracker with the given configuration.
    pub fn new(config: CoherenceTimeSeriesConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            cached_stats: None,
            cached_trend: None,
        }
    }

    /// Record a new coherence measurement.
    pub fn record(
        &mut self,
        round: u64,
        timestamp: i64,
        coherence_value: f32,
        epistemic_confidence: f32,
        byzantine_count: usize,
        node_count: usize,
    ) {
        let data_point = CoherenceDataPoint::new(
            round,
            timestamp,
            coherence_value,
            epistemic_confidence,
            byzantine_count,
            node_count,
        );
        self.add_data_point(data_point);
    }

    /// Record a new coherence measurement with metadata.
    pub fn record_with_metadata(
        &mut self,
        round: u64,
        timestamp: i64,
        coherence_value: f32,
        epistemic_confidence: f32,
        byzantine_count: usize,
        node_count: usize,
        metadata: HashMap<String, f32>,
    ) {
        let data_point = CoherenceDataPoint::new(
            round,
            timestamp,
            coherence_value,
            epistemic_confidence,
            byzantine_count,
            node_count,
        )
        .with_metadata(metadata);
        self.add_data_point(data_point);
    }

    /// Add a data point and manage history.
    fn add_data_point(&mut self, data_point: CoherenceDataPoint) {
        self.history.push(data_point);

        if !self.config.store_full_history && self.history.len() > self.config.window_size {
            let excess = self.history.len() - self.config.window_size;
            self.history.drain(0..excess);
        }

        // Invalidate caches
        self.cached_stats = None;
        self.cached_trend = None;
    }

    /// Get the current (most recent) coherence value.
    pub fn get_current_coherence(&self) -> Option<f32> {
        self.history.last().map(|dp| dp.coherence_value)
    }

    /// Get the current data point.
    pub fn get_current(&self) -> Option<&CoherenceDataPoint> {
        self.history.last()
    }

    /// Get the number of recorded data points.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if the series is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get the window of data used for statistics.
    fn get_window(&self) -> &[CoherenceDataPoint] {
        let n = self.history.len();
        let start = n.saturating_sub(self.config.window_size);
        &self.history[start..]
    }

    /// Calculate statistics over the window.
    pub fn get_statistics(&mut self) -> CoherenceStatistics {
        if let Some(ref stats) = self.cached_stats {
            return stats.clone();
        }

        let window = self.get_window();
        if window.is_empty() {
            return CoherenceStatistics::empty();
        }

        let values: Vec<f32> = window.iter().map(|dp| dp.coherence_value).collect();
        let n = values.len();

        let mean = values.iter().sum::<f32>() / n as f32;

        let variance = if n > 1 {
            values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / (n - 1) as f32
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let trend_slope = calculate_slope(&values);
        let autocorrelation = calculate_autocorrelation(&values);

        let stats = CoherenceStatistics {
            mean,
            std_dev,
            min,
            max,
            trend_slope,
            autocorrelation,
            sample_count: n,
        };

        self.cached_stats = Some(stats.clone());
        stats
    }

    /// Determine the current trend.
    pub fn get_trend(&mut self) -> CoherenceTrend {
        if let Some(ref trend) = self.cached_trend {
            return trend.clone();
        }

        let stats = self.get_statistics();

        if stats.sample_count < 3 {
            let trend = CoherenceTrend::Stable { variance: 0.0 };
            self.cached_trend = Some(trend.clone());
            return trend;
        }

        let cv = if stats.mean.abs() > 1e-10 {
            stats.std_dev / stats.mean.abs()
        } else {
            0.0
        };

        const SLOPE_THRESHOLD: f32 = 0.001;
        const VOLATILITY_THRESHOLD: f32 = 0.3;

        let trend = if cv > VOLATILITY_THRESHOLD {
            CoherenceTrend::Volatile {
                variance: stats.std_dev.powi(2),
            }
        } else if stats.trend_slope > SLOPE_THRESHOLD {
            CoherenceTrend::Rising {
                slope: stats.trend_slope,
            }
        } else if stats.trend_slope < -SLOPE_THRESHOLD {
            CoherenceTrend::Falling {
                slope: stats.trend_slope,
            }
        } else {
            CoherenceTrend::Stable {
                variance: stats.std_dev.powi(2),
            }
        };

        self.cached_trend = Some(trend.clone());
        trend
    }

    /// Detect anomalies in the most recent measurement.
    pub fn detect_anomaly(&mut self) -> Option<CoherenceAnomaly> {
        if self.history.len() < 5 {
            return None;
        }

        let stats = self.get_statistics();

        if stats.std_dev < 1e-10 {
            return None;
        }

        let current = self.history.last()?;
        let z_score = (current.coherence_value - stats.mean) / stats.std_dev;

        if z_score.abs() >= self.config.anomaly_threshold {
            let possible_cause = self.diagnose_anomaly_cause(current, z_score);

            Some(CoherenceAnomaly::new(
                current.round,
                stats.mean,
                current.coherence_value,
                z_score,
                possible_cause,
            ))
        } else {
            None
        }
    }

    /// Diagnose the possible cause of an anomaly.
    fn diagnose_anomaly_cause(&self, current: &CoherenceDataPoint, z_score: f32) -> String {
        let mut causes = Vec::new();

        if current.byzantine_count > 0 && z_score < 0.0 {
            causes.push(format!(
                "Byzantine activity ({} nodes flagged)",
                current.byzantine_count
            ));
        }

        if let Some(prev) = self.history.iter().rev().nth(1) {
            if current.node_count != prev.node_count {
                let diff = current.node_count as i64 - prev.node_count as i64;
                if diff < 0 {
                    causes.push(format!("{} nodes dropped", -diff));
                } else {
                    causes.push(format!("{} new nodes joined", diff));
                }
            }
        }

        if current.epistemic_confidence < 0.5 {
            causes.push("Low epistemic confidence".to_string());
        }

        if z_score < -3.0 {
            causes.push("Possible coordinated attack".to_string());
        }

        if causes.is_empty() {
            if z_score > 0.0 {
                "Unexpected improvement".to_string()
            } else {
                "Unknown cause - requires investigation".to_string()
            }
        } else {
            causes.join("; ")
        }
    }

    /// Get recent history.
    pub fn get_history(&self, last_n: usize) -> Vec<CoherenceDataPoint> {
        let n = self.history.len();
        let start = n.saturating_sub(last_n);
        self.history[start..].to_vec()
    }

    /// Get full history.
    pub fn get_full_history(&self) -> &[CoherenceDataPoint] {
        &self.history
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.history).unwrap_or_else(|_| "[]".to_string())
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.cached_stats = None;
        self.cached_trend = None;
    }

    /// Get the configuration.
    pub fn config(&self) -> &CoherenceTimeSeriesConfig {
        &self.config
    }
}

/// Calculate the slope of values using linear regression.
fn calculate_slope(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }

    let n_f = n as f32;
    let x_mean = (n_f - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f32>() / n_f;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f32;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Calculate autocorrelation at lag 1.
fn calculate_autocorrelation(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 3 {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / n as f32;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;

    for i in 1..n {
        numerator += (values[i] - mean) * (values[i - 1] - mean);
    }

    for &v in values {
        denominator += (v - mean).powi(2);
    }

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_rising_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 + 0.002 * i as f32).collect()
    }

    fn create_falling_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.8 - 0.003 * i as f32).collect()
    }

    fn create_stable_data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.6 + 0.001 * (i as f32 * 0.1).sin())
            .collect()
    }

    fn create_volatile_data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 + 0.2 * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    #[test]
    fn test_config_builder() {
        let config = CoherenceTimeSeriesConfig::default()
            .with_window_size(50)
            .with_anomaly_threshold(3.0)
            .with_full_history(false);

        assert_eq!(config.window_size, 50);
        assert_eq!(config.anomaly_threshold, 3.0);
        assert!(!config.store_full_history);
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut series = CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default());

        series.record(1, 0, 0.5, 0.9, 0, 10);
        series.record(2, 0, 0.55, 0.92, 1, 10);
        series.record(3, 0, 0.6, 0.88, 0, 10);

        assert_eq!(series.len(), 3);
        assert_eq!(series.get_current_coherence(), Some(0.6));

        let history = series.get_history(2);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].round, 2);
        assert_eq!(history[1].round, 3);
    }

    #[test]
    fn test_trend_detection_rising() {
        let mut series =
            CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default().with_window_size(20));

        let values = create_rising_data(20);
        for (i, &v) in values.iter().enumerate() {
            series.record(i as u64, 0, v, 0.9, 0, 10);
        }

        let trend = series.get_trend();
        match trend {
            CoherenceTrend::Rising { slope } => {
                assert!(slope > 0.0, "Slope should be positive for rising trend");
            }
            other => panic!("Expected Rising trend, got {:?}", other),
        }
    }

    #[test]
    fn test_trend_detection_falling() {
        let mut series =
            CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default().with_window_size(20));

        let values = create_falling_data(20);
        for (i, &v) in values.iter().enumerate() {
            series.record(i as u64, 0, v, 0.9, 0, 10);
        }

        let trend = series.get_trend();
        match trend {
            CoherenceTrend::Falling { slope } => {
                assert!(slope < 0.0, "Slope should be negative for falling trend");
            }
            other => panic!("Expected Falling trend, got {:?}", other),
        }
    }

    #[test]
    fn test_trend_detection_stable() {
        let mut series =
            CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default().with_window_size(20));

        let values = create_stable_data(20);
        for (i, &v) in values.iter().enumerate() {
            series.record(i as u64, 0, v, 0.9, 0, 10);
        }

        let trend = series.get_trend();
        match trend {
            CoherenceTrend::Stable { variance } => {
                assert!(variance >= 0.0, "Variance should be non-negative");
            }
            other => panic!("Expected Stable trend, got {:?}", other),
        }
    }

    #[test]
    fn test_trend_detection_volatile() {
        let mut series =
            CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default().with_window_size(20));

        let values = create_volatile_data(20);
        for (i, &v) in values.iter().enumerate() {
            series.record(i as u64, 0, v, 0.9, 0, 10);
        }

        let trend = series.get_trend();
        match trend {
            CoherenceTrend::Volatile { variance } => {
                assert!(
                    variance > 0.0,
                    "Variance should be positive for volatile data"
                );
            }
            other => panic!("Expected Volatile trend, got {:?}", other),
        }
    }

    #[test]
    fn test_anomaly_detection_drop() {
        let mut series = CoherenceTimeSeries::new(
            CoherenceTimeSeriesConfig::default()
                .with_window_size(20)
                .with_anomaly_threshold(2.0),
        );

        for i in 0..15 {
            series.record(i, 0, 0.7, 0.9, 0, 10);
        }

        // Inject anomaly (sudden drop)
        series.record(15, 0, 0.2, 0.5, 3, 10);

        let anomaly = series.detect_anomaly();
        assert!(anomaly.is_some(), "Should detect the anomaly");

        let anomaly = anomaly.unwrap();
        assert_eq!(anomaly.round, 15);
        assert!(anomaly.is_drop(), "Should be a drop anomaly");
        assert!(anomaly.z_score < 0.0, "Z-score should be negative for drop");
        assert!(
            anomaly.severity >= AnomalySeverity::Medium,
            "Severity should be at least Medium"
        );
    }

    #[test]
    fn test_anomaly_detection_spike() {
        let mut series = CoherenceTimeSeries::new(
            CoherenceTimeSeriesConfig::default()
                .with_window_size(20)
                .with_anomaly_threshold(2.0),
        );

        for i in 0..15 {
            series.record(i, 0, 0.3, 0.9, 0, 10);
        }

        // Inject anomaly (sudden spike)
        series.record(15, 0, 0.9, 0.95, 0, 10);

        let anomaly = series.detect_anomaly();
        assert!(anomaly.is_some(), "Should detect the anomaly");

        let anomaly = anomaly.unwrap();
        assert!(anomaly.is_spike(), "Should be a spike anomaly");
        assert!(anomaly.z_score > 0.0);
    }

    #[test]
    fn test_no_false_positives() {
        let mut series = CoherenceTimeSeries::new(
            CoherenceTimeSeriesConfig::default()
                .with_window_size(20)
                .with_anomaly_threshold(2.5),
        );

        for i in 0..20 {
            let v = 0.6 + 0.02 * ((i as f32) * 0.3).sin();
            series.record(i, 0, v, 0.9, 0, 10);
        }

        let anomaly = series.detect_anomaly();
        assert!(anomaly.is_none(), "Should not detect anomaly in normal data");
    }

    #[test]
    fn test_statistics_calculation() {
        let mut series =
            CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default().with_window_size(10));

        let values = [0.5, 0.6, 0.7, 0.55, 0.65, 0.58, 0.62, 0.68, 0.72, 0.60];
        for (i, &v) in values.iter().enumerate() {
            series.record(i as u64, 0, v, 0.9, 0, 10);
        }

        let stats = series.get_statistics();

        let expected_mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        assert!(
            (stats.mean - expected_mean).abs() < 1e-5,
            "Mean should match"
        );

        assert_eq!(stats.min, 0.5);
        assert_eq!(stats.max, 0.72);
        assert_eq!(stats.sample_count, 10);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_rolling_window_behavior() {
        let config = CoherenceTimeSeriesConfig::default()
            .with_window_size(5)
            .with_full_history(false);
        let mut series = CoherenceTimeSeries::new(config);

        for i in 0..10 {
            series.record(i, 0, 0.5 + 0.01 * i as f32, 0.9, 0, 10);
        }

        assert_eq!(series.len(), 5);
        assert_eq!(series.get_full_history()[0].round, 5);
    }

    #[test]
    fn test_full_history_mode() {
        let config = CoherenceTimeSeriesConfig::default()
            .with_window_size(5)
            .with_full_history(true);
        let mut series = CoherenceTimeSeries::new(config);

        for i in 0..10 {
            series.record(i, 0, 0.5, 0.9, 0, 10);
        }

        assert_eq!(series.len(), 10);
    }

    #[test]
    fn test_serialization() {
        let mut series = CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default());

        series.record(1, 100, 0.5, 0.9, 0, 10);
        series.record(2, 200, 0.55, 0.92, 1, 10);

        let json = series.to_json();
        assert!(json.contains("round"), "JSON should contain round");
        assert!(
            json.contains("coherence_value"),
            "JSON should contain coherence_value"
        );
    }

    #[test]
    fn test_trend_helpers() {
        let rising = CoherenceTrend::Rising { slope: 0.01 };
        assert!(rising.is_healthy());
        assert!(!rising.is_concerning());
        assert_eq!(rising.slope(), Some(0.01));
        assert_eq!(rising.variance(), None);

        let falling = CoherenceTrend::Falling { slope: -0.02 };
        assert!(!falling.is_healthy());
        assert!(falling.is_concerning());
        assert_eq!(falling.slope(), Some(-0.02));

        let volatile = CoherenceTrend::Volatile { variance: 0.05 };
        assert!(volatile.is_concerning());
        assert_eq!(volatile.variance(), Some(0.05));
    }

    #[test]
    fn test_anomaly_severity_from_z_score() {
        assert_eq!(AnomalySeverity::from_z_score(1.5), AnomalySeverity::Low);
        assert_eq!(AnomalySeverity::from_z_score(2.6), AnomalySeverity::Medium);
        assert_eq!(AnomalySeverity::from_z_score(3.5), AnomalySeverity::High);
        assert_eq!(AnomalySeverity::from_z_score(4.5), AnomalySeverity::Critical);
        assert_eq!(
            AnomalySeverity::from_z_score(-4.5),
            AnomalySeverity::Critical
        );
    }

    #[test]
    fn test_empty_series() {
        let mut series = CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default());

        assert!(series.is_empty());
        assert_eq!(series.len(), 0);
        assert_eq!(series.get_current_coherence(), None);
        assert!(series.detect_anomaly().is_none());

        let stats = series.get_statistics();
        assert_eq!(stats.sample_count, 0);
    }

    #[test]
    fn test_record_with_metadata() {
        let mut series = CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default());

        let mut metadata = HashMap::new();
        metadata.insert("layer_0".to_string(), 0.4);
        metadata.insert("layer_1".to_string(), 0.6);

        series.record_with_metadata(1, 100, 0.5, 0.9, 0, 10, metadata);

        let current = series.get_current().unwrap();
        assert!(current.metadata.is_some());

        let meta = current.metadata.as_ref().unwrap();
        assert_eq!(meta.get("layer_0"), Some(&0.4));
        assert_eq!(meta.get("layer_1"), Some(&0.6));
    }

    #[test]
    fn test_clear() {
        let mut series = CoherenceTimeSeries::new(CoherenceTimeSeriesConfig::default());

        series.record(1, 0, 0.5, 0.9, 0, 10);
        series.record(2, 0, 0.55, 0.92, 1, 10);

        assert_eq!(series.len(), 2);

        series.clear();

        assert!(series.is_empty());
        assert_eq!(series.len(), 0);
    }
}
