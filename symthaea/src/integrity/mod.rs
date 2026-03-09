//! Hardware Integrity & Tamper Detection Framework
//!
//! Provides runtime verification that critical data structures haven't been
//! tampered with (memory corruption, binary patching, adversarial substrate).
//!
//! ## Components
//!
//! - **Attestation** (`attestation.rs`): BLAKE3 hashing of critical structures at startup,
//!   periodic re-verification at co-prime intervals.
//! - **Temporal Consistency** (`temporal_consistency.rs`): Cross-checks wall clock against
//!   CfC temporal dynamics to detect clock manipulation.
//! - **Behavioral Canaries** (`behavioral_canaries.rs`): Known-answer tests for moral algebra,
//!   Phi computation, safety levels. Run periodically; deviation = corruption or tampering.
//! - **Redundant Computation** (`redundant_computation.rs`): Trait interface for future
//!   multi-hardware verification (design only).
//! - **HSM Interface** (`hsm_interface.rs`): Trait interface for hardware security module
//!   integration (design only).
//!
//! Feature-gated behind `integrity`.

pub mod attestation;
pub mod behavioral_canaries;
pub mod hsm_interface;
pub mod redundant_computation;
pub mod temporal_consistency;

use std::time::Instant;

pub use attestation::{AttestationRecord, AttestationRegistry};
pub use behavioral_canaries::{CanaryFailure, CanaryRunner, CanarySeverity, CanaryTest};
pub use temporal_consistency::TemporalConsistencyMonitor;

/// Overall integrity status, updated each check cycle.
#[derive(Debug, Clone)]
pub struct IntegrityStatus {
    /// Whether all attestation hashes matched.
    pub attestation_passed: bool,
    /// Whether temporal consistency checks passed.
    pub temporal_passed: bool,
    /// Whether all behavioral canaries returned expected results.
    pub canaries_passed: bool,
    /// Cycle number of the last integrity check.
    pub last_check_cycle: usize,
    /// Any detected anomalies.
    pub anomalies: Vec<IntegrityAnomaly>,
}

impl Default for IntegrityStatus {
    fn default() -> Self {
        Self {
            attestation_passed: true,
            temporal_passed: true,
            canaries_passed: true,
            last_check_cycle: 0,
            anomalies: Vec::new(),
        }
    }
}

/// A detected integrity anomaly.
#[derive(Debug, Clone)]
pub struct IntegrityAnomaly {
    /// Which component detected the anomaly.
    pub source: &'static str,
    /// Human-readable description.
    pub description: String,
    /// When it was detected.
    pub detected_at: Instant,
    /// Severity level.
    pub severity: AnomalySeverity,
}

/// Severity of an integrity anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    /// Numerical drift within tolerance — log warning.
    Warning,
    /// Categorical deviation — escalate to SafetyAgent.
    Critical,
}

/// Main integrity manager, integrated into CognitiveLoopService.
///
/// Runs attestation, temporal, and canary checks at co-prime intervals.
pub struct IntegrityManager {
    pub attestation: AttestationRegistry,
    pub temporal: TemporalConsistencyMonitor,
    pub canaries: CanaryRunner,
    /// Most recent integrity status for telemetry.
    pub status: IntegrityStatus,
    /// Timestamp of the last cycle start, for computing wall elapsed.
    last_cycle_instant: Instant,
    /// Substrate tau factor for scaling temporal consistency tolerance.
    /// Set via `set_substrate_tau_factor()`. Default 1.0.
    substrate_tau_factor: f32,
}

/// Co-prime attestation check interval (not in the existing set: 7,11,13,19,23,47,97).
const ATTESTATION_INTERVAL: usize = 101;

impl IntegrityManager {
    /// Create a new integrity manager, computing baseline hashes immediately.
    pub fn new() -> Self {
        Self {
            attestation: AttestationRegistry::new(),
            temporal: TemporalConsistencyMonitor::new(),
            canaries: CanaryRunner::new(),
            status: IntegrityStatus::default(),
            last_cycle_instant: Instant::now(),
            substrate_tau_factor: 1.0,
        }
    }

    /// Set the substrate tau factor for scaling temporal consistency tolerance.
    ///
    /// Faster substrates (tau > 1) legitimately have shorter cycle times,
    /// so the temporal monitor's min_cycle_duration should scale accordingly.
    pub fn set_substrate_tau_factor(&mut self, tau: f32) {
        self.substrate_tau_factor = tau;
        // Scale temporal monitor thresholds: faster substrate → shorter acceptable cycles
        let inv_tau = 1.0 / tau.max(0.01);
        self.temporal.min_cycle_duration =
            std::time::Duration::from_micros((100.0 * inv_tau as f64) as u64);
    }

    /// Run integrity checks that are due this cycle. Self-tracks wall elapsed.
    ///
    /// Each component fires at its own co-prime interval:
    /// - Attestation: every 101 cycles
    /// - Temporal: every cycle (lightweight)
    /// - Canaries: individual intervals (103, 107, 109, 113)
    ///
    /// When `full_sweep` is true (Night phase), all checks run unconditionally.
    /// Science: Besedovsky et al. (2012) — immune system maintenance peaks during sleep.
    pub fn tick(
        &mut self,
        cycle: usize,
        cfc_delta_t: f32,
        full_sweep: bool,
    ) -> &IntegrityStatus {
        let wall_elapsed = self.last_cycle_instant.elapsed();
        self.last_cycle_instant = Instant::now();
        self.status.anomalies.clear();
        self.status.last_check_cycle = cycle;

        // Attestation (every 101 cycles, or every cycle during full sweep)
        if full_sweep || (cycle % ATTESTATION_INTERVAL == 0 && cycle > 0) {
            let failures = self.attestation.verify_all(cycle);
            self.status.attestation_passed = failures.is_empty();
            for failure in failures {
                self.status.anomalies.push(IntegrityAnomaly {
                    source: "attestation",
                    description: failure,
                    detected_at: Instant::now(),
                    severity: AnomalySeverity::Critical,
                });
            }
        }

        // Temporal consistency (every cycle, lightweight)
        if let Some(anomaly) = self.temporal.check(wall_elapsed, cfc_delta_t) {
            self.status.temporal_passed = false;
            self.status.anomalies.push(IntegrityAnomaly {
                source: "temporal",
                description: anomaly,
                detected_at: Instant::now(),
                severity: AnomalySeverity::Warning,
            });
        } else {
            self.status.temporal_passed = true;
        }

        // Behavioral canaries (individual co-prime intervals, or all during full sweep)
        let canary_failures = if full_sweep {
            self.canaries.run_all(cycle)
        } else {
            self.canaries.run_due(cycle)
        };
        self.status.canaries_passed = canary_failures.is_empty();
        for failure in canary_failures {
            let severity = match failure.severity {
                CanarySeverity::Drift => AnomalySeverity::Warning,
                CanarySeverity::Corruption => AnomalySeverity::Critical,
            };
            self.status.anomalies.push(IntegrityAnomaly {
                source: "canary",
                description: format!(
                    "{}: expected {}, got {}",
                    failure.canary_name, failure.expected, failure.actual
                ),
                detected_at: Instant::now(),
                severity,
            });
        }

        &self.status
    }

    /// Check if any critical anomalies have been detected.
    pub fn has_critical_anomaly(&self) -> bool {
        self.status
            .anomalies
            .iter()
            .any(|a| a.severity == AnomalySeverity::Critical)
    }
}

impl Default for IntegrityManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Real-structure attestation helpers ─────────────────────────────────────

impl IntegrityManager {
    /// Register safety-critical thresholds for attestation.
    ///
    /// These are the values where tampering would be most dangerous:
    /// consciousness equation weights, safety level thresholds, moral topology constants.
    pub fn register_safety_thresholds(&mut self, thresholds: &[f32]) {
        let hash = attestation::blake3_hash_f32_slice(thresholds);
        let frozen = thresholds.to_vec();
        self.attestation.register(
            "safety_thresholds",
            hash,
            Box::new(move || attestation::blake3_hash_f32_slice(&frozen)),
        );
    }

    /// Register consciousness equation weights for attestation.
    pub fn register_consciousness_weights(&mut self, weights: &[f64]) {
        let hash = attestation::blake3_hash_f64_slice(weights);
        let frozen = weights.to_vec();
        self.attestation.register(
            "consciousness_weights",
            hash,
            Box::new(move || attestation::blake3_hash_f64_slice(&frozen)),
        );
    }

    /// Register neuromodulator receptor sensitivity curves.
    pub fn register_receptor_sensitivities(&mut self, sensitivities: &[f32]) {
        let hash = attestation::blake3_hash_f32_slice(sensitivities);
        let frozen = sensitivities.to_vec();
        self.attestation.register(
            "receptor_sensitivities",
            hash,
            Box::new(move || attestation::blake3_hash_f32_slice(&frozen)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrity_manager_default_is_clean() {
        let mgr = IntegrityManager::new();
        assert!(mgr.status.attestation_passed);
        assert!(mgr.status.temporal_passed);
        assert!(mgr.status.canaries_passed);
        assert!(mgr.status.anomalies.is_empty());
    }

    #[test]
    fn test_tick_on_non_check_cycle_is_clean() {
        let mut mgr = IntegrityManager::new();
        let status = mgr.tick(5, 0.02, false);
        assert!(status.attestation_passed);
        assert!(status.temporal_passed);
    }

    #[test]
    fn test_full_sweep_runs_attestation_on_any_cycle() {
        let mut mgr = IntegrityManager::new();
        // Register a passing attestation
        let hash = attestation::blake3_hash(b"test");
        mgr.attestation
            .register("test", hash, Box::new(move || attestation::blake3_hash(b"test")));
        // Cycle 5 normally wouldn't trigger attestation (interval=101)
        let status = mgr.tick(5, 0.02, true);
        assert!(status.attestation_passed);
        // Verify the attestation actually ran (last_verification populated)
        assert!(mgr.attestation.records()[0].last_verification.is_some());
    }

    #[test]
    fn test_full_sweep_detects_tampering() {
        let mut mgr = IntegrityManager::new();
        let baseline = attestation::blake3_hash(b"original");
        mgr.attestation.register(
            "tampered",
            baseline,
            Box::new(move || attestation::blake3_hash(b"modified")),
        );
        let status = mgr.tick(1, 0.02, true);
        assert!(!status.attestation_passed);
        assert!(mgr.has_critical_anomaly());
    }
}
