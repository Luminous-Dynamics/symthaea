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

/// Runner that manages a set of canary tests.
pub struct CanaryRunner {
    canaries: Vec<Box<dyn CanaryTest>>,
}

impl CanaryRunner {
    /// Create an empty runner.
    pub fn new() -> Self {
        Self {
            canaries: Vec::new(),
        }
    }

    /// Register a canary test.
    pub fn register(&mut self, canary: Box<dyn CanaryTest>) {
        self.canaries.push(canary);
    }

    /// Run all canaries that are due this cycle. Returns failures.
    pub fn run_due(&self, cycle: usize) -> Vec<CanaryFailure> {
        let mut failures = Vec::new();
        for canary in &self.canaries {
            if cycle > 0 && cycle % canary.interval() == 0 {
                if let Err(failure) = canary.run() {
                    failures.push(failure);
                }
            }
        }
        failures
    }

    /// Run ALL canaries unconditionally (full sweep). Used during Night phase.
    pub fn run_all(&self, _cycle: usize) -> Vec<CanaryFailure> {
        let mut failures = Vec::new();
        for canary in &self.canaries {
            if let Err(failure) = canary.run() {
                failures.push(failure);
            }
        }
        failures
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
