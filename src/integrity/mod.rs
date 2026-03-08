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
        }
    }

    /// Run integrity checks that are due this cycle.
    ///
    /// Each component fires at its own co-prime interval:
    /// - Attestation: every 101 cycles
    /// - Temporal: every cycle (lightweight)
    /// - Canaries: individual intervals (103, 107, 109, 113)
    pub fn tick(
        &mut self,
        cycle: usize,
        wall_elapsed: std::time::Duration,
        cfc_delta_t: f32,
    ) -> &IntegrityStatus {
        self.status.anomalies.clear();
        self.status.last_check_cycle = cycle;

        // Attestation (every 101 cycles)
        if cycle % ATTESTATION_INTERVAL == 0 && cycle > 0 {
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

        // Behavioral canaries (individual co-prime intervals)
        let canary_failures = self.canaries.run_due(cycle);
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
        let status = mgr.tick(5, std::time::Duration::from_millis(20), 0.02);
        assert!(status.attestation_passed);
        assert!(status.temporal_passed);
    }
}
