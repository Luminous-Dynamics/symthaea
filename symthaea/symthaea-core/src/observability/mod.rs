//! Observability stub module for symthaea-core
//!
//! Provides minimal types to allow integrated_information to compile
//! without the full observability infrastructure.

use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

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
