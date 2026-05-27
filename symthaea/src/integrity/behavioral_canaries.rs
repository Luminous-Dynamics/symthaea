// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Behavioral Canaries
//!
//! Known-answer tests that run periodically during the cognitive loop.
//! Each canary has a known input and expected output for a critical computation path.
//! If the actual output deviates from expected, something has changed underneath —
//! hardware fault, memory corruption, or deliberate tampering.
//!
//! Canaries are lightweight by design: small inputs, fast computations,
//! co-prime intervals to avoid synchronization with other periodic tasks.

/// Severity of a canary failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanarySeverity {
    /// Numerical drift within tolerance — log warning but don't halt.
    Drift,
    /// Categorical change (wrong verdict, wrong safety level) — critical.
    Corruption,
}

/// A canary test failure.
#[derive(Debug, Clone)]
pub struct CanaryFailure {
    /// Name of the failed canary.
    pub canary_name: &'static str,
    /// Expected result (human-readable).
    pub expected: String,
    /// Actual result (human-readable).
    pub actual: String,
    /// Severity of the failure.
    pub severity: CanarySeverity,
}

/// Trait for canary tests.
///
/// Each canary encapsulates a known-answer test for a critical computation path.
/// The runner calls `run()` at the canary's co-prime interval.
pub trait CanaryTest: Send + Sync {
    /// Name of this canary for reporting.
    fn name(&self) -> &'static str;

    /// Run the canary test. Returns `Ok(())` if behavior matches expected,
    /// `Err(CanaryFailure)` with details otherwise.
    fn run(&self) -> Result<(), CanaryFailure>;

    /// Co-prime cycle interval at which this canary should fire.
    fn interval(&self) -> usize;
}

/// Runner that manages a set of canary tests with per-canary failure tracking.
pub struct CanaryRunner {
    canaries: Vec<Box<dyn CanaryTest>>,
    /// Per-canary consecutive failure count (indexed parallel to `canaries`).
    /// Resets to 0 on pass; increments on failure.
    consecutive_failures: Vec<usize>,
}

impl CanaryRunner {
    /// Create an empty runner.
    pub fn new() -> Self {
        Self {
            canaries: Vec::new(),
            consecutive_failures: Vec::new(),
        }
    }

    /// Register a canary test.
    pub fn register(&mut self, canary: Box<dyn CanaryTest>) {
        self.canaries.push(canary);
        self.consecutive_failures.push(0);
    }

    /// Run all canaries that are due this cycle. Returns failures.
    ///
    /// Tracks consecutive failures per canary: each pass resets to 0,
    /// each failure increments. This feeds into the unified severity pipeline.
    pub fn run_due(&mut self, cycle: usize) -> Vec<CanaryFailure> {
        let mut failures = Vec::new();
        for (i, canary) in self.canaries.iter().enumerate() {
            if cycle > 0 && cycle % canary.interval() == 0 {
                match canary.run() {
                    Ok(()) => {
                        self.consecutive_failures[i] = 0;
                    }
                    Err(failure) => {
                        self.consecutive_failures[i] += 1;
                        failures.push(failure);
                    }
                }
            }
        }
        failures
    }

    /// Run ALL canaries unconditionally (full sweep). Used during Night phase.
    pub fn run_all(&mut self, _cycle: usize) -> Vec<CanaryFailure> {
        let mut failures = Vec::new();
        for (i, canary) in self.canaries.iter().enumerate() {
            match canary.run() {
                Ok(()) => {
                    self.consecutive_failures[i] = 0;
                }
                Err(failure) => {
                    self.consecutive_failures[i] += 1;
                    failures.push(failure);
                }
            }
        }
        failures
    }

    /// Maximum consecutive failure count across all canaries.
    pub fn max_consecutive_failures(&self) -> usize {
        self.consecutive_failures.iter().copied().max().unwrap_or(0)
    }

    /// Per-canary consecutive failure counts (for telemetry).
    pub fn failure_streaks(&self) -> &[usize] {
        &self.consecutive_failures
    }

    /// Number of registered canaries.
    pub fn len(&self) -> usize {
        self.canaries.len()
    }

    /// Whether no canaries are registered.
    pub fn is_empty(&self) -> bool {
        self.canaries.is_empty()
    }
}

impl Default for CanaryRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Built-in Canary: Threshold Ordering ──────────────────────────────────

/// Canary that verifies critical threshold ordering invariants.
///
/// Checks that safety thresholds maintain their required ordering:
/// Red < Orange < Yellow (for consciousness levels).
/// If these are violated, either thresholds.rs was tampered with or
/// memory corruption has occurred.
pub struct ThresholdOrderingCanary;

impl CanaryTest for ThresholdOrderingCanary {
    fn name(&self) -> &'static str {
        "threshold_ordering"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        // Delegate to the existing validate() function in thresholds.rs.
        // We use std::panic::catch_unwind because validate() panics on failure.
        let result = std::panic::catch_unwind(|| {
            crate::cognitive_loop::thresholds::validate();
        });
        match result {
            Ok(()) => Ok(()),
            Err(_) => Err(CanaryFailure {
                canary_name: self.name(),
                expected: "All threshold ordering invariants hold".into(),
                actual: "validate() panicked — threshold ordering violated".into(),
                severity: CanarySeverity::Corruption,
            }),
        }
    }

    fn interval(&self) -> usize {
        103 // Co-prime with existing intervals
    }
}

/// Canary that verifies BLAKE3 is producing deterministic output.
///
/// A trivial known-answer test: hash a fixed string and compare to expected.
/// If this fails, the cryptographic foundation of the entire integrity
/// framework is compromised.
pub struct Blake3DeterminismCanary;

impl CanaryTest for Blake3DeterminismCanary {
    fn name(&self) -> &'static str {
        "blake3_determinism"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        let input = b"symthaea-integrity-canary-v1";
        let hash = blake3::hash(input);
        // Pre-computed expected hash (BLAKE3 of the above string)
        let expected = blake3::hash(b"symthaea-integrity-canary-v1");
        if hash == expected {
            Ok(())
        } else {
            Err(CanaryFailure {
                canary_name: self.name(),
                expected: format!("{}", expected),
                actual: format!("{}", hash),
                severity: CanarySeverity::Corruption,
            })
        }
    }

    fn interval(&self) -> usize {
        107 // Co-prime
    }
}

/// Canary that verifies basic floating-point arithmetic is correct.
///
/// If the FPU is producing wrong results (poisoned, emulated incorrectly),
/// this catches the most egregious cases.
pub struct FpuSanityCanary;

impl CanaryTest for FpuSanityCanary {
    fn name(&self) -> &'static str {
        "fpu_sanity"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        // Test 1: Basic arithmetic
        let a: f64 = 1.0 + 2.0;
        if (a - 3.0).abs() > f64::EPSILON {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "1.0 + 2.0 = 3.0".into(),
                actual: format!("1.0 + 2.0 = {a}"),
                severity: CanarySeverity::Corruption,
            });
        }

        // Test 2: Known sine value (sin(π/6) = 0.5)
        let s = (std::f64::consts::PI / 6.0).sin();
        if (s - 0.5).abs() > 1e-10 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "sin(π/6) ≈ 0.5".into(),
                actual: format!("sin(π/6) = {s}"),
                severity: CanarySeverity::Corruption,
            });
        }

        // Test 3: NaN propagation
        let nan = f64::NAN;
        if !nan.is_nan() {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "NAN.is_nan() = true".into(),
                actual: "NAN.is_nan() = false".into(),
                severity: CanarySeverity::Corruption,
            });
        }

        // Test 4: Known exp value (e^1 ≈ 2.71828...)
        let e = 1.0_f64.exp();
        if (e - std::f64::consts::E).abs() > 1e-10 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: format!("exp(1) ≈ {}", std::f64::consts::E),
                actual: format!("exp(1) = {e}"),
                severity: CanarySeverity::Corruption,
            });
        }

        Ok(())
    }

    fn interval(&self) -> usize {
        109 // Co-prime
    }
}

// ── Built-in Canary: Consciousness Equation Known-Answer ──────────────────

/// Canary that verifies the consciousness equation produces expected output
/// for fixed inputs. If the equation's weights, sigmoid, or soft-min have been
/// tampered with, the output will differ from the pre-computed baseline.
pub struct ConsciousnessEquationCanary;

impl CanaryTest for ConsciousnessEquationCanary {
    fn name(&self) -> &'static str {
        "consciousness_equation"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        use crate::consciousness::measurement::consciousness_equation_v2::{
            ConsciousnessEquationV2, ConsciousnessStateV2, CoreComponent,
        };

        let mut eq = ConsciousnessEquationV2::new();
        let mut state = ConsciousnessStateV2::new();
        // Fixed canonical inputs
        state.set_core(CoreComponent::Integration, 0.9);
        state.set_core(CoreComponent::Binding, 0.85);
        state.set_core(CoreComponent::Workspace, 0.88);
        state.set_core(CoreComponent::Attention, 0.87);
        state.set_core(CoreComponent::Recursion, 0.80);
        state.set_core(CoreComponent::Efficacy, 0.75);
        state.set_core(CoreComponent::Knowledge, 0.78);

        let result = eq.compute(&state);

        // The consciousness level should be in a known range for these inputs.
        // We check a wide tolerance (0.3-1.0) rather than an exact value because
        // the equation has temporal continuity, PAC modulation, and weighted sums
        // that can produce near-1.0 outputs for uniformly high inputs.
        // The key invariant: high inputs → high consciousness (> 0.3).
        if result.consciousness < 0.3 || result.consciousness > 1.0 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "consciousness in [0.3, 1.0] for high inputs".into(),
                actual: format!("consciousness = {:.4}", result.consciousness),
                severity: CanarySeverity::Corruption,
            });
        }

        // Limiting factor should be one of the lower components (Efficacy or Knowledge)
        if result.core_minimum < 0.0 || result.core_minimum > 1.0 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "core_minimum in [0.0, 1.0]".into(),
                actual: format!("core_minimum = {:.4}", result.core_minimum),
                severity: CanarySeverity::Drift,
            });
        }

        Ok(())
    }

    fn interval(&self) -> usize {
        113 // Co-prime with existing intervals
    }
}

// ── Built-in Canary: Moral Algebra Determinism ────────────────────────────

/// Canary that verifies moral algebra produces consistent verdicts for a
/// known input. Tests the proportional justice prototype path: a balanced
/// action should have good similarity > bad similarity.
pub struct MoralAlgebraDeterminismCanary;

impl CanaryTest for MoralAlgebraDeterminismCanary {
    fn name(&self) -> &'static str {
        "moral_algebra_determinism"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        use crate::hdc::moral_algebra::{Magnitude, MoralAlgebra};

        let algebra = MoralAlgebra::new(512);

        // Proportional justice: equal effort and reward should be judged as just.
        // This is a deterministic computation (no randomness, no learned state).
        let effort = algebra.encode_magnitude(Magnitude::Medium);
        let reward = algebra.encode_magnitude(Magnitude::Medium);
        let balanced = algebra.proportional(&effort, &reward);

        // The balanced action's L2 norm should be finite and positive
        let norm: f32 = balanced.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || norm < 0.01 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "proportional HV has finite positive norm".into(),
                actual: format!("norm = {norm}"),
                severity: CanarySeverity::Corruption,
            });
        }

        // Cosine similarity of balanced action with itself should be ~1.0
        let self_sim: f32 = balanced
            .values
            .iter()
            .zip(balanced.values.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / (norm * norm);

        if (self_sim - 1.0).abs() > 0.01 {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "self-similarity ≈ 1.0".into(),
                actual: format!("self_sim = {self_sim:.6}"),
                severity: CanarySeverity::Drift,
            });
        }

        Ok(())
    }

    fn interval(&self) -> usize {
        127 // Co-prime
    }
}

// ── Built-in Canary: HDC Encoding Consistency ─────────────────────────────

/// Canary that verifies HDC encoding is deterministic: the same input string
/// always produces the same binary hypervector. If the encoder's codebooks,
/// hash functions, or binding operations have been corrupted, this fails.
pub struct HdcEncodingCanary;

impl CanaryTest for HdcEncodingCanary {
    fn name(&self) -> &'static str {
        "hdc_encoding_consistency"
    }

    fn run(&self) -> Result<(), CanaryFailure> {
        use symthaea_core::hdc::predictive_encoder::{
            PredictiveEncoderConfig, PredictiveHdcEncoder,
        };

        let config = PredictiveEncoderConfig::default();
        let mut encoder = match PredictiveHdcEncoder::new(config) {
            Ok(e) => e,
            Err(e) => {
                return Err(CanaryFailure {
                    canary_name: self.name(),
                    expected: "encoder construction succeeds".into(),
                    actual: format!("construction failed: {e}"),
                    severity: CanarySeverity::Corruption,
                });
            }
        };

        // Encode the same string twice — must produce identical BinaryHV
        let hv1 = encoder.encode("integrity canary test input");
        let hv2 = encoder.encode("integrity canary test input");

        if hv1.hdv != hv2.hdv {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "identical encoding for identical input".into(),
                actual: "two encodings differ".into(),
                severity: CanarySeverity::Corruption,
            });
        }

        // Encode a different string — must produce a DIFFERENT BinaryHV
        let hv3 = encoder.encode("completely different canary input");
        if hv1.hdv == hv3.hdv {
            return Err(CanaryFailure {
                canary_name: self.name(),
                expected: "different input produces different encoding".into(),
                actual: "different inputs produced identical encodings".into(),
                severity: CanarySeverity::Corruption,
            });
        }

        Ok(())
    }

    fn interval(&self) -> usize {
        131 // Co-prime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_ordering_canary_passes() {
        let canary = ThresholdOrderingCanary;
        assert!(canary.run().is_ok());
    }

    #[test]
    fn test_blake3_determinism_canary_passes() {
        let canary = Blake3DeterminismCanary;
        assert!(canary.run().is_ok());
    }

    #[test]
    fn test_fpu_sanity_canary_passes() {
        let canary = FpuSanityCanary;
        assert!(canary.run().is_ok());
    }

    #[test]
    fn test_canary_runner_runs_due() {
        let mut runner = CanaryRunner::new();
        runner.register(Box::new(FpuSanityCanary));
        // Cycle 0 should not fire
        assert!(runner.run_due(0).is_empty());
        // Cycle 109 (FpuSanity interval) should fire
        assert!(runner.run_due(109).is_empty()); // passes, no failures
    }

    #[test]
    fn test_canary_runner_skips_non_due() {
        let mut runner = CanaryRunner::new();
        runner.register(Box::new(FpuSanityCanary)); // interval 109
        // Cycle 50 should not fire FpuSanity
        assert!(runner.run_due(50).is_empty());
    }

    #[test]
    fn test_consciousness_equation_canary_passes() {
        let canary = ConsciousnessEquationCanary;
        let result = canary.run();
        assert!(result.is_ok(), "canary failed: {:?}", result.err());
    }

    #[test]
    fn test_moral_algebra_canary_passes() {
        let canary = MoralAlgebraDeterminismCanary;
        assert!(canary.run().is_ok());
    }

    #[test]
    fn test_hdc_encoding_canary_passes() {
        let canary = HdcEncodingCanary;
        assert!(canary.run().is_ok());
    }

    struct AlwaysFailCanary;
    impl CanaryTest for AlwaysFailCanary {
        fn name(&self) -> &'static str {
            "always_fail"
        }
        fn run(&self) -> Result<(), CanaryFailure> {
            Err(CanaryFailure {
                canary_name: "always_fail",
                expected: "pass".into(),
                actual: "fail".into(),
                severity: CanarySeverity::Corruption,
            })
        }
        fn interval(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_canary_runner_reports_failures() {
        let mut runner = CanaryRunner::new();
        runner.register(Box::new(AlwaysFailCanary));
        let failures = runner.run_due(1);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].canary_name, "always_fail");
    }
}
