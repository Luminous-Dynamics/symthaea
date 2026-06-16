// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability stub module for symthaea-core
//!
//! Provides minimal types to allow integrated_information to compile
//! without the full observability infrastructure.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

pub mod types {
    //! Observability types (stubs)

    /// Stub trait for observation events
    pub trait ObservationEvent: Send + Sync {}

    /// Stub implementation for any type
    impl<T: Send + Sync> ObservationEvent for T {}
}

/// Shared observer handle using RwLock for interior mutability
pub type SharedObserver = Arc<RwLock<dyn Observer + Send + Sync>>;

/// Observer trait for consciousness metrics
pub trait Observer: Send + Sync {
    /// Record an observation
    fn observe(&self, event: &dyn types::ObservationEvent);

    /// Record a Φ measurement event
    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<(), String>;
}

/// No-op observer implementation
pub struct NoOpObserver;

impl Observer for NoOpObserver {
    fn observe(&self, _event: &dyn types::ObservationEvent) {
        // No-op
    }

    fn record_phi_measurement(&mut self, _event: PhiMeasurementEvent) -> Result<(), String> {
        // No-op - silently accept measurements
        Ok(())
    }
}

/// Create a shared no-op observer
pub fn no_op_observer() -> SharedObserver {
    Arc::new(RwLock::new(NoOpObserver))
}

/// Φ (Phi) measurement event for observability
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhiMeasurementEvent {
    /// Timestamp of the measurement
    pub timestamp: DateTime<Utc>,

    /// The Φ value (integrated information)
    pub phi: f64,

    /// Detailed Φ components
    pub components: PhiComponents,

    /// Temporal continuity with previous measurements
    pub temporal_continuity: f64,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Detailed Φ components based on IIT theory
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhiComponents {
    /// Core Φ value (minimum info partition loss)
    pub integration: f64,

    /// How strongly components bind (MIP info loss)
    pub binding: f64,

    /// Global workspace information (total system info)
    pub workspace: f64,

    /// Selective integration (component distinctiveness)
    pub attention: f64,

    /// Self-referential processing (temporal continuity)
    pub recursion: f64,

    /// Processing efficiency (normalized Φ)
    pub efficacy: f64,

    /// Accumulated information (historical Φ average)
    pub knowledge: f64,
}

// ---------------------------------------------------------------------------
// Real Metrics Collector
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Time-series data point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: String,
}

/// Consciousness state transition event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTransition {
    pub timestamp: DateTime<Utc>,
    pub from_phi: f64,
    pub to_phi: f64,
    pub from_level: f64,
    pub to_level: f64,
    pub trigger: String,
}

/// Metrics snapshot for export
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub phi_history: Vec<DataPoint>,
    pub consciousness_levels: Vec<DataPoint>,
    pub state_transitions: Vec<StateTransition>,
    pub operation_counts: HashMap<String, u64>,
    pub total_observations: u64,
    pub uptime_seconds: f64,
}

/// Real metrics collector that records consciousness telemetry
pub struct MetricsCollector {
    phi_history: Vec<DataPoint>,
    consciousness_levels: Vec<DataPoint>,
    state_transitions: Vec<StateTransition>,
    operation_counts: HashMap<String, u64>,
    total_observations: u64,
    start_time: DateTime<Utc>,
    max_history: usize,
    last_phi: f64,
    last_consciousness_level: f64,
}

impl MetricsCollector {
    /// Create a new `MetricsCollector` with a default history limit of 10 000.
    pub fn new() -> Self {
        Self::with_max_history(10_000)
    }

    /// Create a new `MetricsCollector` with a custom history limit.
    pub fn with_max_history(max: usize) -> Self {
        Self {
            phi_history: Vec::new(),
            consciousness_levels: Vec::new(),
            state_transitions: Vec::new(),
            operation_counts: HashMap::new(),
            total_observations: 0,
            start_time: Utc::now(),
            max_history: max,
            last_phi: 0.0,
            last_consciousness_level: 0.0,
        }
    }

    /// Record a consciousness level reading.
    pub fn record_consciousness_level(&mut self, level: f64) {
        let dp = DataPoint {
            timestamp: Utc::now(),
            value: level,
            label: "consciousness_level".into(),
        };
        self.consciousness_levels.push(dp);
        if self.consciousness_levels.len() > self.max_history {
            self.consciousness_levels.remove(0);
        }
        self.last_consciousness_level = level;
    }

    /// Increment the counter for `operation`.
    pub fn record_operation(&mut self, operation: &str) {
        *self
            .operation_counts
            .entry(operation.to_string())
            .or_insert(0) += 1;
    }

    /// Export the current state as an immutable snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = Utc::now()
            .signed_duration_since(self.start_time)
            .num_milliseconds() as f64
            / 1000.0;
        MetricsSnapshot {
            timestamp: Utc::now(),
            phi_history: self.phi_history.clone(),
            consciousness_levels: self.consciousness_levels.clone(),
            state_transitions: self.state_transitions.clone(),
            operation_counts: self.operation_counts.clone(),
            total_observations: self.total_observations,
            uptime_seconds: uptime,
        }
    }

    /// Average Δφ (delta-phi) over the most recent `window` measurements.
    ///
    /// Returns 0.0 when fewer than 2 data-points exist in the window.
    pub fn phi_trend(&self, window: usize) -> f64 {
        let len = self.phi_history.len();
        if len < 2 {
            return 0.0;
        }
        let start = len.saturating_sub(window);
        let slice = &self.phi_history[start..];
        if slice.len() < 2 {
            return 0.0;
        }
        let deltas: Vec<f64> = slice.windows(2).map(|w| w[1].value - w[0].value).collect();
        deltas.iter().sum::<f64>() / deltas.len() as f64
    }

    /// Average φ over the most recent `window` measurements.
    pub fn average_phi(&self, window: usize) -> f64 {
        let len = self.phi_history.len();
        if len == 0 {
            return 0.0;
        }
        let start = len.saturating_sub(window);
        let slice = &self.phi_history[start..];
        slice.iter().map(|dp| dp.value).sum::<f64>() / slice.len() as f64
    }

    /// Peak φ across the entire recorded history.
    pub fn peak_phi(&self) -> f64 {
        self.phi_history
            .iter()
            .map(|dp| dp.value)
            .fold(0.0_f64, f64::max)
    }

    /// Number of detected state transitions.
    pub fn transition_count(&self) -> usize {
        self.state_transitions.len()
    }

    /// Clear all collected data and reset counters.
    pub fn clear(&mut self) {
        self.phi_history.clear();
        self.consciousness_levels.clear();
        self.state_transitions.clear();
        self.operation_counts.clear();
        self.total_observations = 0;
        self.start_time = Utc::now();
        self.last_phi = 0.0;
        self.last_consciousness_level = 0.0;
    }
}

impl Observer for MetricsCollector {
    fn observe(&self, _event: &dyn types::ObservationEvent) {
        // MetricsCollector focuses on phi measurements; generic observe is a no-op.
    }

    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<(), String> {
        let phi = event.phi;
        let dp = DataPoint {
            timestamp: event.timestamp,
            value: phi,
            label: "phi".into(),
        };
        self.phi_history.push(dp);
        if self.phi_history.len() > self.max_history {
            self.phi_history.remove(0);
        }
        self.total_observations += 1;

        // Detect state transition when Δφ > 0.1
        let delta = (phi - self.last_phi).abs();
        if delta > 0.1 && self.total_observations > 1 {
            let transition = StateTransition {
                timestamp: event.timestamp,
                from_phi: self.last_phi,
                to_phi: phi,
                from_level: self.last_consciousness_level,
                to_level: self.last_consciousness_level,
                trigger: format!("phi_delta_{delta:.4}"),
            };
            self.state_transitions.push(transition);
            if self.state_transitions.len() > self.max_history {
                self.state_transitions.remove(0);
            }
        }

        self.last_phi = phi;
        Ok(())
    }
}

/// Create a shared `MetricsCollector` wrapped in `Arc<RwLock<..>>`.
pub fn metrics_collector() -> SharedObserver {
    Arc::new(RwLock::new(MetricsCollector::new()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a `PhiMeasurementEvent` with a given phi value.
    fn phi_event(phi: f64) -> PhiMeasurementEvent {
        PhiMeasurementEvent {
            timestamp: Utc::now(),
            phi,
            components: PhiComponents::default(),
            temporal_continuity: 1.0,
            metadata: None,
        }
    }

    #[test]
    fn test_collector_creation() {
        let c = MetricsCollector::new();
        assert_eq!(c.total_observations, 0);
        assert_eq!(c.phi_history.len(), 0);
        assert_eq!(c.max_history, 10_000);
        assert_eq!(c.last_phi, 0.0);

        let c2 = MetricsCollector::with_max_history(5);
        assert_eq!(c2.max_history, 5);
    }

    #[test]
    fn test_record_phi_measurement() {
        let mut c = MetricsCollector::new();
        c.record_phi_measurement(phi_event(0.42)).unwrap();
        assert_eq!(c.total_observations, 1);
        assert_eq!(c.phi_history.len(), 1);
        assert!((c.phi_history[0].value - 0.42).abs() < 1e-12);
        assert_eq!(c.last_phi, 0.42);
    }

    #[test]
    fn test_phi_trend_calculation() {
        let mut c = MetricsCollector::new();
        // 0.1, 0.2, 0.3, 0.4 => deltas: +0.1, +0.1, +0.1 => avg = 0.1
        for v in [0.1, 0.2, 0.3, 0.4] {
            c.record_phi_measurement(phi_event(v)).unwrap();
        }
        let trend = c.phi_trend(10);
        assert!((trend - 0.1).abs() < 1e-9, "trend was {trend}");

        // Window of 2 uses last 2 points => delta 0.3->0.4 = 0.1
        let trend2 = c.phi_trend(2);
        assert!((trend2 - 0.1).abs() < 1e-9, "trend2 was {trend2}");

        // Empty collector
        let empty = MetricsCollector::new();
        assert_eq!(empty.phi_trend(5), 0.0);
    }

    #[test]
    fn test_state_transition_detection() {
        let mut c = MetricsCollector::new();
        // First measurement — no transition (no prior)
        c.record_phi_measurement(phi_event(0.5)).unwrap();
        assert_eq!(c.transition_count(), 0);

        // Small change — no transition
        c.record_phi_measurement(phi_event(0.55)).unwrap();
        assert_eq!(c.transition_count(), 0);

        // Large change — triggers transition
        c.record_phi_measurement(phi_event(0.9)).unwrap();
        assert_eq!(c.transition_count(), 1);
        assert!((c.state_transitions[0].from_phi - 0.55).abs() < 1e-12);
        assert!((c.state_transitions[0].to_phi - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_operation_counting() {
        let mut c = MetricsCollector::new();
        c.record_operation("bind");
        c.record_operation("bind");
        c.record_operation("permute");
        assert_eq!(c.operation_counts["bind"], 2);
        assert_eq!(c.operation_counts["permute"], 1);
    }

    #[test]
    fn test_snapshot_export() {
        let mut c = MetricsCollector::new();
        c.record_phi_measurement(phi_event(0.7)).unwrap();
        c.record_operation("bundle");
        c.record_consciousness_level(0.8);

        let snap = c.snapshot();
        assert_eq!(snap.phi_history.len(), 1);
        assert_eq!(snap.consciousness_levels.len(), 1);
        assert_eq!(snap.total_observations, 1);
        assert_eq!(snap.operation_counts["bundle"], 1);
        assert!(snap.uptime_seconds >= 0.0);
    }

    #[test]
    fn test_history_limit() {
        let mut c = MetricsCollector::with_max_history(3);
        for i in 0..5 {
            c.record_phi_measurement(phi_event(i as f64)).unwrap();
        }
        // Only the last 3 entries survive
        assert_eq!(c.phi_history.len(), 3);
        assert!((c.phi_history[0].value - 2.0).abs() < 1e-12);
        assert!((c.phi_history[2].value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_average_phi() {
        let mut c = MetricsCollector::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            c.record_phi_measurement(phi_event(v)).unwrap();
        }
        // All 5: avg = 3.0
        assert!((c.average_phi(10) - 3.0).abs() < 1e-9);
        // Last 3: avg of 3,4,5 = 4.0
        assert!((c.average_phi(3) - 4.0).abs() < 1e-9);
        // Empty
        let e = MetricsCollector::new();
        assert_eq!(e.average_phi(5), 0.0);
    }

    #[test]
    fn test_peak_phi() {
        let mut c = MetricsCollector::new();
        for v in [0.1, 0.9, 0.5, 0.3] {
            c.record_phi_measurement(phi_event(v)).unwrap();
        }
        assert!((c.peak_phi() - 0.9).abs() < 1e-12);
        // Empty => 0.0
        let e = MetricsCollector::new();
        assert_eq!(e.peak_phi(), 0.0);
    }

    #[test]
    fn test_clear() {
        let mut c = MetricsCollector::new();
        c.record_phi_measurement(phi_event(0.5)).unwrap();
        c.record_operation("test_op");
        c.record_consciousness_level(0.7);
        c.clear();
        assert_eq!(c.total_observations, 0);
        assert_eq!(c.phi_history.len(), 0);
        assert_eq!(c.consciousness_levels.len(), 0);
        assert_eq!(c.state_transitions.len(), 0);
        assert!(c.operation_counts.is_empty());
        assert_eq!(c.last_phi, 0.0);
        assert_eq!(c.last_consciousness_level, 0.0);
    }

    #[test]
    fn test_shared_observer_factory() {
        let shared = metrics_collector();
        {
            let mut writer = shared.write().unwrap();
            let result = writer.record_phi_measurement(phi_event(0.6));
            assert!(result.is_ok());
        }
        {
            let reader = shared.read().unwrap();
            reader.observe(&42_u64);
        }
    }
}
