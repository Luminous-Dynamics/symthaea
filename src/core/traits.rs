//! # Consciousness API Traits
//!
//! Unified trait interfaces for consciousness measurement, state management,
//! and updates. This provides a stable abstraction layer for the 6+ Φ
//! implementations across the codebase.
//!
//! ## Design Goals
//!
//! 1. **Unify Measurement**: Single trait for all Φ calculation methods
//! 2. **State Abstraction**: Common interface for consciousness state
//! 3. **Observable**: Built-in support for telemetry and tracing
//! 4. **Theory-Agnostic**: Works with IIT, GWT, HOT, FEP, etc.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::core::traits::{ConsciousnessMetric, ConsciousnessState};
//!
//! fn measure_system<M: ConsciousnessMetric>(metric: &M, state: &[ContinuousHV]) -> f64 {
//!     let result = metric.measure(state);
//!     println!("Φ = {:.4} (theory: {})", result.value, result.theory_basis);
//!     result.value
//! }
//! ```

use std::fmt::Debug;

/// Result of a consciousness measurement
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// The measured value (typically Φ, but theory-dependent)
    pub value: f64,

    /// Confidence in the measurement (0.0 to 1.0)
    pub confidence: f32,

    /// Theoretical basis (e.g., "IIT 3.0", "GWT", "HOT", "FEP")
    pub theory_basis: &'static str,

    /// Number of components measured
    pub n_components: usize,

    /// Computation time in microseconds
    pub compute_time_us: u64,
}

impl MeasurementResult {
    /// Create a new measurement result
    pub fn new(value: f64, theory: &'static str, n_components: usize) -> Self {
        Self {
            value,
            confidence: 1.0,
            theory_basis: theory,
            n_components,
            compute_time_us: 0,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set computation time
    pub fn with_time(mut self, time_us: u64) -> Self {
        self.compute_time_us = time_us;
        self
    }
}

/// Unified trait for consciousness measurement
///
/// Implementations may use different theoretical frameworks:
/// - **IIT**: Integrated Information Theory (Φ)
/// - **GWT**: Global Workspace Theory (workspace access)
/// - **HOT**: Higher-Order Thought (meta-awareness)
/// - **FEP**: Free Energy Principle (prediction error)
///
/// All implementations return a standardized `MeasurementResult`.
pub trait ConsciousnessMetric: Send + Sync {
    /// The input type for measurement (typically hypervector collections)
    type Input: ?Sized;

    /// Measure consciousness level for the given input
    fn measure(&self, input: &Self::Input) -> MeasurementResult;

    /// Get the theoretical basis for this metric
    fn theory_basis(&self) -> &'static str;

    /// Get the computational complexity class
    fn complexity(&self) -> Complexity;

    /// Whether this metric supports caching
    fn supports_caching(&self) -> bool {
        false
    }
}

/// Computational complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    /// O(1) - constant time
    Constant,
    /// O(log n) - logarithmic
    Logarithmic,
    /// O(n) - linear
    Linear,
    /// O(n log n) - linearithmic
    Linearithmic,
    /// O(n²) - quadratic
    Quadratic,
    /// O(n³) - cubic
    Cubic,
    /// O(2^n) - exponential
    Exponential,
}

impl Complexity {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Complexity::Constant => "O(1) - instant",
            Complexity::Logarithmic => "O(log n) - very fast",
            Complexity::Linear => "O(n) - fast",
            Complexity::Linearithmic => "O(n log n) - moderate",
            Complexity::Quadratic => "O(n²) - slow for large n",
            Complexity::Cubic => "O(n³) - very slow for large n",
            Complexity::Exponential => "O(2^n) - only for small n",
        }
    }
}

/// Snapshot of consciousness state
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Current Φ value
    pub phi: f64,

    /// Global workspace contents summary
    pub workspace_summary: String,

    /// Active attention focus
    pub attention_focus: Option<String>,

    /// Meta-awareness level (0.0 to 1.0)
    pub meta_awareness: f64,

    /// Whether the system is considered conscious
    pub is_conscious: bool,
}

impl Default for StateSnapshot {
    fn default() -> Self {
        Self {
            phi: 0.0,
            workspace_summary: String::new(),
            attention_focus: None,
            meta_awareness: 0.0,
            is_conscious: false,
        }
    }
}

/// Trait for consciousness state management
///
/// Provides read access to the current consciousness state without
/// exposing internal implementation details.
pub trait ConsciousnessState: Send + Sync {
    /// Get current Φ measurement
    fn phi(&self) -> f64;

    /// Get a snapshot of the current state
    fn snapshot(&self) -> StateSnapshot;

    /// Whether the system is currently conscious (Φ > threshold)
    fn is_conscious(&self) -> bool {
        self.phi() > 0.5
    }

    /// Get the global workspace contents (if applicable)
    fn workspace_contents(&self) -> Option<String> {
        None
    }
}

/// Trait for updating consciousness state
///
/// Provides mutable operations for consciousness state transitions.
pub trait ConsciousnessUpdater: ConsciousnessState {
    /// Input type for updates
    type Input;

    /// Output type from updates
    type Output;

    /// Error type for failed updates
    type Error: std::error::Error;

    /// Process input and update consciousness state
    fn update(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// Reset to initial state
    fn reset(&mut self);

    /// Record a state transition for learning/analysis
    fn record_transition(&mut self, input: &Self::Input, output: &Self::Output) {
        // Default: no-op (override for learning systems)
        let _ = (input, output);
    }
}

/// Trait for consciousness observability
///
/// Enables telemetry, tracing, and debugging of consciousness processes.
pub trait ConsciousnessObserver: Send + Sync {
    /// Called when Φ is measured
    fn on_phi_measured(&self, phi: f64, method: &str);

    /// Called when consciousness state changes
    fn on_state_change(&self, old_phi: f64, new_phi: f64);

    /// Called when an anomaly is detected
    fn on_anomaly(&self, description: &str, severity: f64);

    /// Called when workspace contents change
    fn on_workspace_update(&self, contents: &str);
}

/// Null observer that does nothing (for when observability is disabled)
pub struct NullObserver;

impl ConsciousnessObserver for NullObserver {
    fn on_phi_measured(&self, _phi: f64, _method: &str) {}
    fn on_state_change(&self, _old_phi: f64, _new_phi: f64) {}
    fn on_anomaly(&self, _description: &str, _severity: f64) {}
    fn on_workspace_update(&self, _contents: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_result() {
        let result = MeasurementResult::new(0.75, "IIT 3.0", 8)
            .with_confidence(0.95)
            .with_time(1500);

        assert!((result.value - 0.75).abs() < 1e-10);
        assert_eq!(result.theory_basis, "IIT 3.0");
        assert_eq!(result.n_components, 8);
        assert!((result.confidence - 0.95).abs() < 1e-10);
        assert_eq!(result.compute_time_us, 1500);
    }

    #[test]
    fn test_complexity_description() {
        assert!(Complexity::Exponential.description().contains("small n"));
        assert!(Complexity::Constant.description().contains("instant"));
    }

    #[test]
    fn test_state_snapshot_default() {
        let snapshot = StateSnapshot::default();
        assert!(!snapshot.is_conscious);
        assert!(snapshot.phi < 0.01);
    }
}
