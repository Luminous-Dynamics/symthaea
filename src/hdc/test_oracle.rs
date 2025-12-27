//! Test Oracle for Deterministic Consciousness Measurements
//!
//! This module provides mock consciousness measurements for testing.
//! It solves the circular dependency problem where tests need consciousness
//! metrics, but consciousness metrics require complex calculations.
//!
//! # The Problem
//!
//! Many tests fail because:
//! 1. They need Φ (integrated information) measurements
//! 2. Φ calculation is O(2^n) and times out
//! 3. Even if fast, Φ is non-deterministic (depends on random seeds)
//!
//! # The Solution
//!
//! The Test Oracle provides:
//! - Deterministic consciousness values for testing
//! - Configurable mock behaviors
//! - No external dependencies
//!
//! # Usage
//!
//! ```rust
//! use symthaea::hdc::test_oracle::TestOracle;
//!
//! // Create oracle with default settings
//! let oracle = TestOracle::default();
//!
//! // Get deterministic Φ
//! let phi = oracle.phi(5); // Returns 0.5 for 5 components
//!
//! // Use in tests
//! #[cfg(test)]
//! mod tests {
//!     use super::*;
//!     use symthaea::hdc::test_oracle::TestOracle;
//!
//!     #[test]
//!     fn test_something_needing_phi() {
//!         let oracle = TestOracle::default();
//!         let phi = oracle.phi(10);
//!         assert!(phi > 0.0);
//!     }
//! }
//! ```

use std::collections::HashMap;

// ============================================================================
// TEST ORACLE
// ============================================================================

/// Deterministic consciousness oracle for testing
///
/// Provides predictable, repeatable values for consciousness metrics.
#[derive(Debug, Clone)]
pub struct TestOracle {
    /// Base Φ value (default: 0.5)
    pub base_phi: f64,

    /// Φ scaling factor per component (default: 0.05)
    pub phi_per_component: f64,

    /// Maximum Φ (default: 0.95)
    pub max_phi: f64,

    /// Base free energy (default: 0.3)
    pub base_free_energy: f64,

    /// Override values for specific contexts
    pub overrides: HashMap<String, f64>,

    /// Whether to track calls for verification
    pub track_calls: bool,

    /// Call history (if tracking enabled)
    call_history: Vec<OracleCall>,
}

/// Record of an oracle call
#[derive(Debug, Clone)]
pub struct OracleCall {
    /// Method called
    pub method: String,
    /// Arguments (as string for simplicity)
    pub args: String,
    /// Value returned
    pub value: f64,
}

impl Default for TestOracle {
    fn default() -> Self {
        Self {
            base_phi: 0.5,
            phi_per_component: 0.05,
            max_phi: 0.95,
            base_free_energy: 0.3,
            overrides: HashMap::new(),
            track_calls: false,
            call_history: Vec::new(),
        }
    }
}

impl TestOracle {
    /// Create a new test oracle with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create oracle with custom base Φ
    pub fn with_base_phi(base_phi: f64) -> Self {
        Self {
            base_phi,
            ..Default::default()
        }
    }

    /// Create oracle that tracks all calls
    pub fn with_tracking() -> Self {
        Self {
            track_calls: true,
            ..Default::default()
        }
    }

    /// Set an override value for a specific context
    pub fn override_phi(&mut self, context: &str, value: f64) {
        self.overrides.insert(context.to_string(), value);
    }

    /// Get deterministic Φ for a given number of components
    ///
    /// Formula: min(base_phi + phi_per_component * (n - 1), max_phi)
    pub fn phi(&mut self, num_components: usize) -> f64 {
        let value = if num_components < 2 {
            0.0
        } else {
            let scaled = self.base_phi + self.phi_per_component * (num_components - 1) as f64;
            scaled.min(self.max_phi)
        };

        if self.track_calls {
            self.call_history.push(OracleCall {
                method: "phi".to_string(),
                args: format!("num_components={}", num_components),
                value,
            });
        }

        value
    }

    /// Get Φ with context override support
    pub fn phi_with_context(&mut self, num_components: usize, context: &str) -> f64 {
        if let Some(&override_value) = self.overrides.get(context) {
            if self.track_calls {
                self.call_history.push(OracleCall {
                    method: "phi_with_context".to_string(),
                    args: format!("context={}", context),
                    value: override_value,
                });
            }
            return override_value;
        }

        self.phi(num_components)
    }

    /// Get deterministic free energy
    ///
    /// Free energy inversely correlates with Φ for most systems
    pub fn free_energy(&mut self, num_components: usize) -> f64 {
        let phi = self.phi(num_components);
        let value = self.base_free_energy * (1.0 - phi);

        if self.track_calls {
            self.call_history.push(OracleCall {
                method: "free_energy".to_string(),
                args: format!("num_components={}", num_components),
                value,
            });
        }

        value
    }

    /// Get deterministic consciousness level description
    pub fn consciousness_level(&mut self, num_components: usize) -> ConsciousnessLevel {
        let phi = self.phi(num_components);

        let level = if phi < 0.2 {
            ConsciousnessLevel::Unconscious
        } else if phi < 0.4 {
            ConsciousnessLevel::Minimal
        } else if phi < 0.6 {
            ConsciousnessLevel::Moderate
        } else if phi < 0.8 {
            ConsciousnessLevel::High
        } else {
            ConsciousnessLevel::Full
        };

        if self.track_calls {
            self.call_history.push(OracleCall {
                method: "consciousness_level".to_string(),
                args: format!("num_components={}", num_components),
                value: phi,
            });
        }

        level
    }

    /// Get call history (if tracking enabled)
    pub fn call_history(&self) -> &[OracleCall] {
        &self.call_history
    }

    /// Clear call history
    pub fn clear_history(&mut self) {
        self.call_history.clear();
    }

    /// Get number of calls made
    pub fn call_count(&self) -> usize {
        self.call_history.len()
    }
}

/// Consciousness levels for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessLevel {
    /// Φ < 0.2
    Unconscious,
    /// 0.2 ≤ Φ < 0.4
    Minimal,
    /// 0.4 ≤ Φ < 0.6
    Moderate,
    /// 0.6 ≤ Φ < 0.8
    High,
    /// Φ ≥ 0.8
    Full,
}

impl ConsciousnessLevel {
    /// Get description string
    pub fn description(&self) -> &'static str {
        match self {
            ConsciousnessLevel::Unconscious => "Unconscious (Φ < 0.2)",
            ConsciousnessLevel::Minimal => "Minimal consciousness (0.2 ≤ Φ < 0.4)",
            ConsciousnessLevel::Moderate => "Moderate consciousness (0.4 ≤ Φ < 0.6)",
            ConsciousnessLevel::High => "High consciousness (0.6 ≤ Φ < 0.8)",
            ConsciousnessLevel::Full => "Full consciousness (Φ ≥ 0.8)",
        }
    }

    /// Get numeric range
    pub fn phi_range(&self) -> (f64, f64) {
        match self {
            ConsciousnessLevel::Unconscious => (0.0, 0.2),
            ConsciousnessLevel::Minimal => (0.2, 0.4),
            ConsciousnessLevel::Moderate => (0.4, 0.6),
            ConsciousnessLevel::High => (0.6, 0.8),
            ConsciousnessLevel::Full => (0.8, 1.0),
        }
    }
}

// ============================================================================
// GLOBAL TEST ORACLE (for easy integration)
// ============================================================================

use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Global test oracle instance
static GLOBAL_ORACLE: Lazy<Mutex<TestOracle>> = Lazy::new(|| {
    Mutex::new(TestOracle::default())
});

/// Get Φ from global oracle (for simple test usage)
pub fn mock_phi(num_components: usize) -> f64 {
    GLOBAL_ORACLE.lock().unwrap().phi(num_components)
}

/// Get free energy from global oracle
pub fn mock_free_energy(num_components: usize) -> f64 {
    GLOBAL_ORACLE.lock().unwrap().free_energy(num_components)
}

/// Get consciousness level from global oracle
pub fn mock_consciousness_level(num_components: usize) -> ConsciousnessLevel {
    GLOBAL_ORACLE.lock().unwrap().consciousness_level(num_components)
}

/// Reset global oracle to defaults
pub fn reset_global_oracle() {
    *GLOBAL_ORACLE.lock().unwrap() = TestOracle::default();
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_oracle() {
        let mut oracle = TestOracle::default();

        assert_eq!(oracle.phi(0), 0.0);
        assert_eq!(oracle.phi(1), 0.0);
        assert!(oracle.phi(2) > 0.0);
    }

    #[test]
    fn test_phi_scales_with_components() {
        let mut oracle = TestOracle::default();

        let phi_2 = oracle.phi(2);
        let phi_5 = oracle.phi(5);
        let phi_10 = oracle.phi(10);

        assert!(phi_5 > phi_2);
        assert!(phi_10 > phi_5);
    }

    #[test]
    fn test_phi_capped_at_max() {
        let mut oracle = TestOracle::default();

        let phi_100 = oracle.phi(100);

        assert!(phi_100 <= oracle.max_phi);
    }

    #[test]
    fn test_deterministic() {
        let mut oracle1 = TestOracle::default();
        let mut oracle2 = TestOracle::default();

        assert_eq!(oracle1.phi(5), oracle2.phi(5));
        assert_eq!(oracle1.phi(10), oracle2.phi(10));
    }

    #[test]
    fn test_overrides() {
        let mut oracle = TestOracle::default();
        oracle.override_phi("special_context", 0.99);

        let normal = oracle.phi(5);
        let overridden = oracle.phi_with_context(5, "special_context");

        assert!(normal < 0.99);
        assert_eq!(overridden, 0.99);
    }

    #[test]
    fn test_free_energy() {
        let mut oracle = TestOracle::default();

        let fe = oracle.free_energy(5);

        assert!(fe >= 0.0);
        assert!(fe <= oracle.base_free_energy);
    }

    #[test]
    fn test_consciousness_levels() {
        let mut oracle = TestOracle::new();

        // Very low components → low Φ → unconscious
        oracle.base_phi = 0.1;
        oracle.phi_per_component = 0.01;
        assert_eq!(oracle.consciousness_level(2), ConsciousnessLevel::Unconscious);

        // High base → high consciousness
        oracle.base_phi = 0.7;
        assert_eq!(oracle.consciousness_level(5), ConsciousnessLevel::High);
    }

    #[test]
    fn test_call_tracking() {
        let mut oracle = TestOracle::with_tracking();

        oracle.phi(5);
        oracle.phi(10);
        oracle.free_energy(3);

        assert_eq!(oracle.call_count(), 4); // phi(3) called internally by free_energy
        assert_eq!(oracle.call_history()[0].method, "phi");
    }

    #[test]
    fn test_global_oracle() {
        reset_global_oracle();

        let phi = mock_phi(5);
        let fe = mock_free_energy(5);
        let level = mock_consciousness_level(5);

        assert!(phi > 0.0);
        assert!(fe >= 0.0);
        // phi(5) = 0.5 + 0.05 * 4 = 0.7, which is High (0.6 <= phi < 0.8)
        assert_eq!(level, ConsciousnessLevel::High);
    }

    #[test]
    fn test_level_descriptions() {
        assert!(ConsciousnessLevel::Full.description().contains("Full"));
        assert!(ConsciousnessLevel::Unconscious.description().contains("Unconscious"));
    }

    #[test]
    fn test_level_ranges() {
        let (low, high) = ConsciousnessLevel::Moderate.phi_range();
        assert_eq!(low, 0.4);
        assert_eq!(high, 0.6);
    }
}
