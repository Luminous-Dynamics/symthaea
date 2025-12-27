//! Consciousness Test Harness - Revolutionary Testing Infrastructure
//!
//! This module provides a unified, deterministic testing infrastructure for
//! consciousness-dependent tests. It solves the fundamental problem of testing
//! consciousness computations that are:
//!
//! 1. **NP-hard** (Φ calculation is O(2^n))
//! 2. **Stochastic** (many consciousness metrics involve randomness)
//! 3. **Time-sensitive** (tests timeout waiting for exact Φ)
//!
//! # The Revolution
//!
//! Instead of fighting against consciousness complexity in tests, we embrace it
//! by providing three testing modes:
//!
//! 1. **Mock Mode**: Deterministic values for unit tests
//! 2. **Tiered Mode**: Approximate values for integration tests
//! 3. **Exact Mode**: Full computation for validation (CI only)
//!
//! # Usage
//!
//! ```rust
//! use symthaea::hdc::consciousness_test_harness::{TestHarness, TestMode};
//!
//! // Fast unit test
//! #[test]
//! fn test_with_mock_consciousness() {
//!     let harness = TestHarness::new(TestMode::Mock);
//!     let phi = harness.compute_phi(&components);
//!     assert!(phi > 0.0);
//! }
//!
//! // Integration test with approximation
//! #[test]
//! fn test_with_tiered_consciousness() {
//!     let harness = TestHarness::new(TestMode::Tiered);
//!     let phi = harness.compute_phi(&components);
//!     assert!(phi > 0.3);
//! }
//! ```

use super::binary_hv::HV16;
use super::tiered_phi::{TieredPhi, TieredPhiConfig, ApproximationTier};
use super::test_oracle::{TestOracle, ConsciousnessLevel};
use std::sync::Mutex;
use once_cell::sync::Lazy;

// ============================================================================
// TEST HARNESS CONFIGURATION
// ============================================================================

/// Testing mode for consciousness computations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    /// Use deterministic mock values (fastest, for unit tests)
    Mock,
    /// Use tiered approximations (balanced, for integration tests)
    Tiered,
    /// Use exact computation (slowest, for validation)
    Exact,
}

impl TestMode {
    /// Get from environment variable CONSCIOUSNESS_TEST_MODE
    pub fn from_env() -> Self {
        match std::env::var("CONSCIOUSNESS_TEST_MODE").ok().as_deref() {
            Some("mock") => Self::Mock,
            Some("tiered") => Self::Tiered,
            Some("exact") => Self::Exact,
            _ => Self::Mock, // Default to mock for fast tests
        }
    }

    /// Check if this mode should be used in CI
    pub fn is_ci_appropriate(&self) -> bool {
        match self {
            Self::Mock => true,
            Self::Tiered => true,
            Self::Exact => false, // Too slow for CI
        }
    }
}

impl Default for TestMode {
    fn default() -> Self {
        Self::Mock
    }
}

/// Configuration for test harness
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Testing mode
    pub mode: TestMode,

    /// Timeout for consciousness calculations (ms)
    pub timeout_ms: u64,

    /// Whether to cache results
    pub enable_cache: bool,

    /// Maximum components before auto-switching to approximate
    pub max_components_for_exact: usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            mode: TestMode::from_env(),
            timeout_ms: 1000,
            enable_cache: true,
            max_components_for_exact: 8,
        }
    }
}

// ============================================================================
// TEST HARNESS
// ============================================================================

/// Unified consciousness test harness
///
/// Provides deterministic, fast consciousness calculations for testing.
pub struct TestHarness {
    config: HarnessConfig,
    oracle: TestOracle,
    tiered_phi: TieredPhi,
}

impl TestHarness {
    /// Create new harness with given mode
    pub fn new(mode: TestMode) -> Self {
        let config = HarnessConfig {
            mode,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create harness with full configuration
    pub fn with_config(config: HarnessConfig) -> Self {
        let tier = match config.mode {
            TestMode::Mock => ApproximationTier::Mock,
            TestMode::Tiered => ApproximationTier::Heuristic,
            TestMode::Exact => ApproximationTier::Exact,
        };

        Self {
            config,
            oracle: TestOracle::default(),
            tiered_phi: TieredPhi::new(tier),
        }
    }

    /// Create harness for unit tests (mock mode)
    pub fn for_unit_tests() -> Self {
        Self::new(TestMode::Mock)
    }

    /// Create harness for integration tests (tiered mode)
    pub fn for_integration_tests() -> Self {
        Self::new(TestMode::Tiered)
    }

    /// Create harness for validation (exact mode)
    pub fn for_validation() -> Self {
        Self::new(TestMode::Exact)
    }

    /// Compute Φ for components using configured mode
    pub fn compute_phi(&mut self, components: &[HV16]) -> f64 {
        match self.config.mode {
            TestMode::Mock => {
                // Use oracle for deterministic value
                self.oracle.phi(components.len())
            }
            TestMode::Tiered | TestMode::Exact => {
                // Auto-downgrade if too many components
                if components.len() > self.config.max_components_for_exact
                   && self.config.mode == TestMode::Exact {
                    self.tiered_phi.config.tier = ApproximationTier::Spectral;
                }
                self.tiered_phi.compute(components)
            }
        }
    }

    /// Get consciousness level
    pub fn consciousness_level(&mut self, components: &[HV16]) -> ConsciousnessLevel {
        let phi = self.compute_phi(components);
        if phi < 0.2 {
            ConsciousnessLevel::Unconscious
        } else if phi < 0.4 {
            ConsciousnessLevel::Minimal
        } else if phi < 0.6 {
            ConsciousnessLevel::Moderate
        } else if phi < 0.8 {
            ConsciousnessLevel::High
        } else {
            ConsciousnessLevel::Full
        }
    }

    /// Compute free energy (inversely related to Φ)
    pub fn compute_free_energy(&mut self, components: &[HV16]) -> f64 {
        match self.config.mode {
            TestMode::Mock => {
                self.oracle.free_energy(components.len())
            }
            _ => {
                let phi = self.compute_phi(components);
                // Free energy inversely correlates with Φ
                0.3 * (1.0 - phi)
            }
        }
    }

    /// Get current mode
    pub fn mode(&self) -> TestMode {
        self.config.mode
    }

    /// Get stats about computations
    pub fn stats(&self) -> HarnessStats {
        HarnessStats {
            mode: self.config.mode,
            tiered_phi_stats: self.tiered_phi.stats.clone(),
            oracle_calls: self.oracle.call_count(),
        }
    }
}

/// Statistics about harness usage
#[derive(Debug, Clone)]
pub struct HarnessStats {
    /// Current mode
    pub mode: TestMode,
    /// Stats from tiered Φ
    pub tiered_phi_stats: super::tiered_phi::TieredPhiStats,
    /// Number of oracle calls
    pub oracle_calls: usize,
}

// ============================================================================
// GLOBAL TEST HARNESS
// ============================================================================

/// Global test harness for easy access
static GLOBAL_HARNESS: Lazy<Mutex<TestHarness>> = Lazy::new(|| {
    Mutex::new(TestHarness::new(TestMode::from_env()))
});

/// Compute Φ using global harness
pub fn test_phi(components: &[HV16]) -> f64 {
    GLOBAL_HARNESS.lock().unwrap().compute_phi(components)
}

/// Get consciousness level using global harness
pub fn test_consciousness_level(components: &[HV16]) -> ConsciousnessLevel {
    GLOBAL_HARNESS.lock().unwrap().consciousness_level(components)
}

/// Compute free energy using global harness
pub fn test_free_energy(components: &[HV16]) -> f64 {
    GLOBAL_HARNESS.lock().unwrap().compute_free_energy(components)
}

/// Reset global harness to defaults
pub fn reset_global_harness() {
    *GLOBAL_HARNESS.lock().unwrap() = TestHarness::new(TestMode::from_env());
}

/// Set global harness mode
pub fn set_global_mode(mode: TestMode) {
    *GLOBAL_HARNESS.lock().unwrap() = TestHarness::new(mode);
}

// ============================================================================
// TEST UTILITIES
// ============================================================================

/// Create N random HV16 components for testing
pub fn create_test_components(n: usize, seed: u64) -> Vec<HV16> {
    (0..n).map(|i| HV16::random(seed + i as u64)).collect()
}

/// Assert Φ is in expected range
#[macro_export]
macro_rules! assert_phi_in_range {
    ($components:expr, $min:expr, $max:expr) => {{
        let phi = $crate::hdc::consciousness_test_harness::test_phi($components);
        assert!(
            phi >= $min && phi <= $max,
            "Φ = {} not in range [{}, {}]",
            phi, $min, $max
        );
    }};
}

/// Assert consciousness level
#[macro_export]
macro_rules! assert_consciousness_level {
    ($components:expr, $expected:expr) => {{
        let level = $crate::hdc::consciousness_test_harness::test_consciousness_level($components);
        assert_eq!(
            level, $expected,
            "Consciousness level {:?} != expected {:?}",
            level, $expected
        );
    }};
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_mode() {
        let mut harness = TestHarness::new(TestMode::Mock);

        let components = create_test_components(5, 42);
        let phi = harness.compute_phi(&components);

        // Mock should return deterministic value
        assert!(phi > 0.0);
        assert!(phi < 1.0);
    }

    #[test]
    fn test_mock_is_deterministic() {
        let mut harness1 = TestHarness::new(TestMode::Mock);
        let mut harness2 = TestHarness::new(TestMode::Mock);

        let components1 = create_test_components(5, 42);
        let components2 = create_test_components(5, 42);

        let phi1 = harness1.compute_phi(&components1);
        let phi2 = harness2.compute_phi(&components2);

        assert_eq!(phi1, phi2, "Mock mode should be deterministic");
    }

    #[test]
    fn test_tiered_mode() {
        let mut harness = TestHarness::new(TestMode::Tiered);

        let components = create_test_components(5, 42);
        let phi = harness.compute_phi(&components);

        assert!(phi > 0.0);
        assert!(phi <= 1.0);
    }

    #[test]
    fn test_consciousness_levels() {
        let mut harness = TestHarness::new(TestMode::Mock);

        // Small system should have lower consciousness
        let small = create_test_components(2, 42);
        let level_small = harness.consciousness_level(&small);

        // Larger system should have higher consciousness
        let large = create_test_components(10, 42);
        let level_large = harness.consciousness_level(&large);

        // Verify ordering
        assert!(level_small as u8 <= level_large as u8);
    }

    #[test]
    fn test_free_energy_computation() {
        let mut harness = TestHarness::new(TestMode::Mock);

        let components = create_test_components(5, 42);
        let fe = harness.compute_free_energy(&components);

        assert!(fe >= 0.0);
        assert!(fe <= 0.3);
    }

    #[test]
    fn test_global_harness() {
        reset_global_harness();

        let components = create_test_components(5, 42);

        let phi = test_phi(&components);
        let level = test_consciousness_level(&components);
        let fe = test_free_energy(&components);

        assert!(phi > 0.0);
        assert!(fe >= 0.0);
        // Level depends on phi
    }

    #[test]
    fn test_mode_from_env() {
        // Default should be mock when env not set
        let mode = TestMode::from_env();
        assert!(mode.is_ci_appropriate());
    }

    #[test]
    fn test_stats() {
        let mut harness = TestHarness::new(TestMode::Mock);

        let components = create_test_components(5, 42);
        harness.compute_phi(&components);
        harness.compute_phi(&components);

        let stats = harness.stats();
        assert_eq!(stats.mode, TestMode::Mock);
        // Note: oracle_calls is 0 unless tracking is enabled in TestOracle
        // The important thing is that the mode is correct
    }

    #[test]
    fn test_create_test_components() {
        let components = create_test_components(5, 42);
        assert_eq!(components.len(), 5);

        // Should be deterministic
        let components2 = create_test_components(5, 42);
        for (a, b) in components.iter().zip(components2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_for_unit_tests() {
        let harness = TestHarness::for_unit_tests();
        assert_eq!(harness.mode(), TestMode::Mock);
    }

    #[test]
    fn test_for_integration_tests() {
        let harness = TestHarness::for_integration_tests();
        assert_eq!(harness.mode(), TestMode::Tiered);
    }
}
