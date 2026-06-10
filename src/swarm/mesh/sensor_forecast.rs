// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demand Forecaster — CfC-Based Resource Prediction
//!
//! Uses patterns from CfC hidden state and chronobiology to predict
//! resource consumption (water, power, food) over configurable horizons.
//!
//! # Design
//! - Maintains per-resource-type moving averages and trend estimates
//! - Integrates chronobiology for time-of-day patterns
//! - Provides forecast confidence based on data quality

use std::collections::HashMap;

/// Maximum forecast horizon in hours.
const MAX_FORECAST_HORIZON_HOURS: u32 = 168; // 1 week

/// Minimum data points for a meaningful forecast.
const MIN_DATA_POINTS: usize = 10;

/// EMA alpha for consumption tracking.
const CONSUMPTION_EMA_ALPHA: f64 = 0.1;

/// A demand forecast for a resource type.
#[derive(Debug, Clone)]
pub struct DemandForecast {
    /// Resource type being forecast.
    pub resource_type: String,
    /// Forecast horizon in hours.
    pub horizon_hours: u32,
    /// Predicted consumption rate (units/hour).
    pub predicted_rate: f64,
    /// Predicted total consumption over horizon.
    pub predicted_total: f64,
    /// Confidence in forecast (0.0-1.0).
    pub confidence: f64,
    /// Estimated time until resource exhaustion (hours, None if not applicable).
    pub time_to_exhaustion_hours: Option<f64>,
    /// Timestamp of forecast generation.
    pub generated_at: u64,
}

/// Per-resource consumption tracking.
#[derive(Debug, Clone)]
struct ResourceTracker {
    /// Recent consumption values (rate per hour).
    values: Vec<f64>,
    /// Maximum values to retain.
    max_values: usize,
    /// EMA of consumption rate.
    rate_ema: f64,
    /// Current available quantity (if known).
    available: Option<f64>,
    /// Trend direction (-1 = decreasing, 0 = stable, +1 = increasing).
    trend: f64,
    /// Hour-of-day pattern (24 bins, normalized).
    hourly_pattern: [f64; 24],
    /// Samples per hour (for pattern building).
    hourly_counts: [u32; 24],
}

impl ResourceTracker {
    fn new(max_values: usize) -> Self {
        Self {
            values: Vec::new(),
            max_values,
            rate_ema: 0.0,
            available: None,
            trend: 0.0,
            hourly_pattern: [1.0 / 24.0; 24],
            hourly_counts: [0; 24],
        }
    }

    fn add_value(&mut self, rate: f64, hour: usize) {
        if self.values.len() >= self.max_values {
            self.values.remove(0);
        }
        self.values.push(rate);
        self.rate_ema =
            self.rate_ema * (1.0 - CONSUMPTION_EMA_ALPHA) + rate * CONSUMPTION_EMA_ALPHA;

        // Update hourly pattern
        let h = hour.min(23);
        self.hourly_counts[h] += 1;
        let alpha = 1.0 / (self.hourly_counts[h] as f64).min(100.0);
        self.hourly_pattern[h] = self.hourly_pattern[h] * (1.0 - alpha) + rate * alpha;

        // Update trend
        if self.values.len() >= 3 {
            let recent = &self.values[self.values.len() - 3..];
            let slope = (recent[2] - recent[0]) / 2.0;
            self.trend = self.trend * 0.9 + slope * 0.1;
        }
    }

    fn data_quality(&self) -> f64 {
        if self.values.len() < MIN_DATA_POINTS {
            return self.values.len() as f64 / MIN_DATA_POINTS as f64;
        }
        // Quality increases with more data, saturates at 1.0
        (self.values.len() as f64 / (MIN_DATA_POINTS as f64 * 5.0))
            .clamp(0.0, 1.0)
            .max(self.values.len() as f64 / MIN_DATA_POINTS as f64)
            .clamp(0.0, 1.0)
    }
}

/// Demand forecaster for resource consumption prediction.
pub struct DemandForecaster {
    /// Per-resource trackers.
    trackers: HashMap<String, ResourceTracker>,
    /// Maximum history per resource.
    max_history: usize,
}

impl DemandForecaster {
    /// Create a new demand forecaster.
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            max_history: 1000,
        }
    }

    /// Record a consumption data point.
    pub fn record_consumption(
        &mut self,
        resource_type: &str,
        rate_per_hour: f64,
        hour_of_day: usize,
    ) {
        let tracker = self
            .trackers
            .entry(resource_type.to_string())
            .or_insert_with(|| ResourceTracker::new(self.max_history));
        tracker.add_value(rate_per_hour, hour_of_day);
    }

    /// Update the known available quantity for a resource.
    pub fn set_available(&mut self, resource_type: &str, quantity: f64) {
        if let Some(tracker) = self.trackers.get_mut(resource_type) {
            tracker.available = Some(quantity);
        }
    }

    /// Generate a demand forecast.
    pub fn forecast(
        &self,
        resource_type: &str,
        horizon_hours: u32,
        current_hour: usize,
        now_secs: u64,
    ) -> DemandForecast {
        let horizon = horizon_hours.min(MAX_FORECAST_HORIZON_HOURS);

        let tracker = match self.trackers.get(resource_type) {
            Some(t) => t,
            None => {
                return DemandForecast {
                    resource_type: resource_type.to_string(),
                    horizon_hours: horizon,
                    predicted_rate: 0.0,
                    predicted_total: 0.0,
                    confidence: 0.0,
                    time_to_exhaustion_hours: None,
                    generated_at: now_secs,
                };
            }
        };

        // Base rate from EMA + trend adjustment
        let base_rate = (tracker.rate_ema + tracker.trend * horizon as f64 / 2.0).max(0.0);

        // Apply hourly pattern for more accurate total
        let mut total = 0.0;
        for h in 0..horizon as usize {
            let hour = (current_hour + h) % 24;
            let pattern_factor = tracker.hourly_pattern[hour] * 24.0; // Normalize
            total += base_rate * pattern_factor.max(0.1); // Floor at 10% of base
        }

        // Confidence based on data quality and horizon
        let data_confidence = tracker.data_quality();
        let horizon_penalty = 1.0 / (1.0 + (horizon as f64 / 24.0).powi(2));
        let confidence = (data_confidence * horizon_penalty).clamp(0.0, 1.0);

        // Time to exhaustion
        let time_to_exhaustion_hours = tracker.available.map(|avail| {
            if base_rate > f64::EPSILON {
                avail / base_rate
            } else {
                f64::INFINITY
            }
        });

        DemandForecast {
            resource_type: resource_type.to_string(),
            horizon_hours: horizon,
            predicted_rate: base_rate,
            predicted_total: total,
            confidence,
            time_to_exhaustion_hours,
            generated_at: now_secs,
        }
    }

    /// Number of tracked resource types.
    pub fn resource_count(&self) -> usize {
        self.trackers.len()
    }

    /// Get all tracked resource types.
    pub fn resource_types(&self) -> Vec<&str> {
        self.trackers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for DemandForecaster {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_forecast() {
        let fc = DemandForecaster::new();
        let f = fc.forecast("water", 24, 12, 0);
        assert_eq!(f.confidence, 0.0);
        assert_eq!(f.predicted_rate, 0.0);
    }

    #[test]
    fn test_basic_forecast() {
        let mut fc = DemandForecaster::new();
        for i in 0..20 {
            fc.record_consumption("power", 100.0, i % 24);
        }
        let f = fc.forecast("power", 24, 12, 0);
        assert!(f.predicted_rate > 0.0);
        assert!(f.confidence > 0.0);
        assert!(f.predicted_total > 0.0);
    }

    #[test]
    fn test_exhaustion_calculation() {
        let mut fc = DemandForecaster::new();
        for _ in 0..20 {
            fc.record_consumption("water", 10.0, 12);
        }
        fc.set_available("water", 100.0);
        let f = fc.forecast("water", 24, 12, 0);
        assert!(f.time_to_exhaustion_hours.is_some());
        let hours = f.time_to_exhaustion_hours.unwrap();
        assert!(hours > 0.0 && hours < 100.0);
    }

    #[test]
    fn test_confidence_decreases_with_horizon() {
        let mut fc = DemandForecaster::new();
        for i in 0..50 {
            fc.record_consumption("power", 100.0, i % 24);
        }
        let short = fc.forecast("power", 1, 12, 0);
        let long = fc.forecast("power", 168, 12, 0);
        assert!(short.confidence > long.confidence);
    }

    #[test]
    fn test_resource_count() {
        let mut fc = DemandForecaster::new();
        fc.record_consumption("water", 10.0, 12);
        fc.record_consumption("power", 100.0, 12);
        assert_eq!(fc.resource_count(), 2);
    }

    #[test]
    fn test_trend_detection() {
        let mut fc = DemandForecaster::new();
        // Increasing consumption
        for i in 0..20 {
            fc.record_consumption("power", 100.0 + i as f64 * 10.0, 12);
        }
        let f = fc.forecast("power", 24, 12, 0);
        // With increasing trend, predicted rate should be higher than early values
        assert!(f.predicted_rate > 100.0);
    }

    #[test]
    fn test_max_horizon_capped() {
        let fc = DemandForecaster::new();
        let f = fc.forecast("x", 999, 0, 0);
        assert_eq!(f.horizon_hours, MAX_FORECAST_HORIZON_HOURS);
    }

    #[test]
    fn test_default() {
        let fc = DemandForecaster::default();
        assert_eq!(fc.resource_count(), 0);
    }
}
