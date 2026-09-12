// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness-Relevant Measurement API Traits
//!
//! Unified trait interfaces for theory-dependent consciousness-relevant measurement,
//! state management, and updates. These interfaces expose computational proxies and
//! experimental indicators; they do **not** establish phenomenal consciousness.
//!
//! ## Design Goals
//!
//! 1. **Unify Measurement**: Single trait shape for multiple theory-dependent metrics
//! 2. **State Abstraction**: Common interface for consciousness-relevant state
//! 3. **Observable**: Built-in support for telemetry and tracing
//! 4. **Theory-Agnostic**: Works with IIT, GWT, HOT, FEP, etc.
//! 5. **Epistemically Honest**: Proxy thresholds are never named or documented as proof
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::core::traits::{ConsciousnessMetric, ConsciousnessState};
//!
//! fn measure_system<M: ConsciousnessMetric>(metric: &M, state: &M::Input) -> f64
//! where
//!     M::Input: ?Sized,
//! {
//!     let result = metric.measure(state);
//!     println!("proxy = {:.4} (theory: {})", result.value, result.theory_basis);
//!     result.value
//! }
//! ```

use std::fmt::Debug;

/// Result of a consciousness-relevant measurement.
///
/// `value` and `confidence` characterize the measurement under the named
/// theoretical/experimental model. They are not probabilities that the measured
/// system is phenomenally conscious.
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// The measured proxy value (often Φ, but theory-dependent).
    pub value: f64,

    /// Confidence in the measurement procedure/result, not P(conscious).
    pub confidence: f32,

    /// Theoretical basis (e.g., "IIT 3.0", "GWT", "HOT", "FEP").
    pub theory_basis: &'static str,

    /// Number of components measured.
    pub n_components: usize,

    /// Computation time in microseconds.
    pub compute_time_us: u64,
}

impl MeasurementResult {
    /// Create a new measurement result.
    pub fn new(value: f64, theory: &'static str, n_components: usize) -> Self {
        Self {
            value,
            confidence: 1.0,
            theory_basis: theory,
            n_components,
            compute_time_us: 0,
        }
    }

    /// Set confidence in the measurement procedure/result.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set computation time.
    pub fn with_time(mut self, time_us: u64) -> Self {
        self.compute_time_us = time_us;
        self
    }
}

/// Unified trait for consciousness-relevant measurement.
///
/// Implementations may use different theoretical frameworks:
/// - **IIT**: Integrated Information Theory (Φ)
/// - **GWT**: Global Workspace Theory (workspace access)
/// - **HOT**: Higher-Order Thought (meta-representation)
/// - **FEP**: Free Energy Principle (prediction-related dynamics)
///
/// All implementations return a standardized `MeasurementResult`. Callers must
/// preserve the theory basis and must not treat a threshold crossing as proof of
/// phenomenal consciousness or moral-patient status.
pub trait ConsciousnessMetric: Send + Sync {
    /// The input type for measurement (typically hypervector collections).
    type Input: ?Sized;

    /// Measure the theory-dependent proxy for the given input.
    fn measure(&self, input: &Self::Input) -> MeasurementResult;

    /// Get the theoretical basis for this metric.
    fn theory_basis(&self) -> &'static str;

    /// Get the computational complexity class.
    fn complexity(&self) -> Complexity;

    /// Whether this metric supports caching.
    fn supports_caching(&self) -> bool {
        false
    }
}

/// Computational complexity classification.
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
    /// Human-readable description.
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

/// Snapshot of consciousness-relevant state.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Current Φ proxy value.
    pub phi: f64,

    /// Global workspace contents summary.
    pub workspace_summary: String,

    /// Active attention focus.
    pub attention_focus: Option<String>,

    /// Meta-awareness proxy level (0.0 to 1.0).
    pub meta_awareness: f64,

    /// Legacy compatibility flag indicating that the implementation's configured
    /// consciousness proxy threshold was crossed. This is **not** an ontological
    /// consciousness verdict and must not be used as a moral-status decision.
    pub is_conscious: bool,
}

impl StateSnapshot {
    /// Epistemically explicit accessor for the legacy threshold flag.
    pub fn consciousness_proxy_active(&self) -> bool {
        self.is_conscious
    }
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

/// Trait for consciousness-relevant state management.
///
/// Provides read access to theory-dependent proxy state without exposing internal
/// implementation details. Threshold helpers report a configured proxy condition;
/// they do not prove phenomenal consciousness.
pub trait ConsciousnessState: Send + Sync {
    /// Get current Φ proxy measurement.
    fn phi(&self) -> f64;

    /// Get a snapshot of the current state.
    fn snapshot(&self) -> StateSnapshot;

    /// Whether the implementation's legacy proxy threshold is crossed.
    fn consciousness_proxy_active(&self) -> bool {
        self.phi() > 0.5
    }

    /// Legacy alias retained for source compatibility.
    ///
    /// Do not use this method as evidence of phenomenal consciousness or moral
    /// patienthood. New code should call `consciousness_proxy_active()` and preserve
    /// the underlying metric/theory in evidence records.
    #[deprecated(
        since = "0.1.0",
        note = "ambiguous ontological name; use consciousness_proxy_active() and preserve theory/provenance"
    )]
    fn is_conscious(&self) -> bool {
        self.consciousness_proxy_active()
    }

    /// Get the global workspace contents (if applicable).
    fn workspace_contents(&self) -> Option<String> {
        None
    }
}

/// Trait for updating consciousness-relevant state.
///
/// Provides mutable operations for state transitions.
pub trait ConsciousnessUpdater: ConsciousnessState {
    /// Input type for updates.
    type Input;

    /// Output type from updates.
    type Output;

    /// Error type for failed updates.
    type Error: std::error::Error;

    /// Process input and update consciousness-relevant state.
    fn update(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Record a state transition for learning/analysis.
    fn record_transition(&mut self, input: &Self::Input, output: &Self::Output) {
        let _ = (input, output);
    }
}

/// Trait for consciousness-relevant observability.
///
/// Enables telemetry, tracing, and debugging of proxy measurements and state.
pub trait ConsciousnessObserver: Send + Sync {
    /// Called when Φ is measured.
    fn on_phi_measured(&self, phi: f64, method: &str);

    /// Called when consciousness-relevant state changes.
    fn on_state_change(&self, old_phi: f64, new_phi: f64);

    /// Called when an anomaly is detected.
    fn on_anomaly(&self, description: &str, severity: f64);

    /// Called when workspace contents change.
    fn on_workspace_update(&self, contents: &str);
}

/// Null observer that does nothing (for when observability is disabled).
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
        assert!(!snapshot.consciousness_proxy_active());
        assert!(snapshot.phi < 0.01);
    }

    struct ProxyState(f64);

    impl ConsciousnessState for ProxyState {
        fn phi(&self) -> f64 {
            self.0
        }

        fn snapshot(&self) -> StateSnapshot {
            StateSnapshot {
                phi: self.0,
                is_conscious: self.consciousness_proxy_active(),
                ..StateSnapshot::default()
            }
        }
    }

    #[test]
    fn proxy_threshold_is_named_as_proxy_not_ontology() {
        assert!(!ProxyState(0.49).consciousness_proxy_active());
        assert!(ProxyState(0.51).consciousness_proxy_active());
    }
}
