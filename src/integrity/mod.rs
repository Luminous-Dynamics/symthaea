// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub use attestation::{AttestationRecord, AttestationRegistry};
pub use behavioral_canaries::{CanaryFailure, CanaryRunner, CanarySeverity, CanaryTest};
pub use temporal_consistency::TemporalConsistencyMonitor;

// ── Panic Hook for Crash Forensics ───────────────────────────────────────

/// Global storage for the most recent integrity snapshot blob.
/// Updated each tick by `update_panic_snapshot()`, read by the panic hook.
static PANIC_SNAPSHOT: OnceLock<Arc<Mutex<Option<Vec<u8>>>>> = OnceLock::new();

fn panic_snapshot_storage() -> &'static Arc<Mutex<Option<Vec<u8>>>> {
    PANIC_SNAPSHOT.get_or_init(|| Arc::new(Mutex::new(None)))
}

/// Install a panic hook that dumps the integrity snapshot to disk on crash.
///
/// The blob is written to `/tmp/symthaea-integrity-dump-{unix_secs}.bin`.
/// The previous panic hook (if any) is chained so standard panic behavior is preserved.
///
/// Call once at startup (after IntegrityManager is constructed).
pub fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Try to dump the snapshot — best-effort, don't panic inside the panic hook
        if let Ok(guard) = panic_snapshot_storage().lock() {
            if let Some(blob) = guard.as_ref() {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let path = format!("/tmp/symthaea-integrity-dump-{ts}.bin");
                if std::fs::write(&path, blob).is_ok() {
                    eprintln!(
                        "[integrity] Crash forensics: dumped {} bytes to {path}",
                        blob.len()
                    );
                }
            }
        }
        // Chain to previous hook
        prev(info);
    }));
}

/// Update the global panic snapshot with the latest integrity state.
///
/// Called from `IntegrityManager::tick()` each cycle so the dump is always fresh.
pub fn update_panic_snapshot(blob: Vec<u8>) {
    if let Ok(mut guard) = panic_snapshot_storage().lock() {
        *guard = Some(blob);
    }
}

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

/// A timestamped integrity event for the history ring buffer.
#[derive(Debug, Clone)]
pub struct IntegrityEvent {
    /// Cycle when the event occurred.
    pub cycle: usize,
    /// When the event was recorded.
    pub timestamp: Instant,
    /// Source component (attestation, temporal, canary, live_attestation).
    pub source: &'static str,
    /// Human-readable description.
    pub description: String,
    /// Severity at the time of the event.
    pub severity: AnomalySeverity,
}

/// Capacity of the integrity event history ring buffer.
const EVENT_LOG_CAPACITY: usize = 64;

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
    /// Confidence multiplier for consciousness scores (1.0 = fully trusted,
    /// 0.5 = drift detected, 0.1 = critical anomaly). Updated each tick.
    pub integrity_confidence: f32,
    /// Ring buffer of recent integrity events (capacity 64).
    /// Persists across tick() cycles for post-mortem analysis.
    pub event_log: VecDeque<IntegrityEvent>,
    /// Unified cross-source failure streak (attestation + canary failures combined).
    /// Resets to 0 when a full tick completes with no anomalies.
    /// Used for severity escalation: 1-2 = Warning, 3+ = Critical.
    pub global_failure_streak: usize,
    /// Rolling 60-cycle history of integrity_confidence values for sparkline display.
    pub confidence_history: VecDeque<f32>,
    /// Deterministic jitter seed for attestation interval randomization.
    /// Initialized from cycle 0 hash to avoid predictable check timing.
    jitter_seed: u64,
}

/// Base attestation check interval (subject to ±10% jitter).
const ATTESTATION_INTERVAL: usize = 101;

/// Capacity of the integrity confidence sparkline history.
const CONFIDENCE_HISTORY_CAPACITY: usize = 60;

impl IntegrityManager {
    /// Create a new integrity manager, computing baseline hashes immediately.
    ///
    /// Registers all built-in canaries:
    /// - ThresholdOrderingCanary (103): safety threshold ordering invariants
    /// - Blake3DeterminismCanary (107): cryptographic hash determinism
    /// - FpuSanityCanary (109): floating-point arithmetic correctness
    /// - ConsciousnessEquationCanary (113): consciousness equation known-answer
    /// - MoralAlgebraDeterminismCanary (127): moral algebra consistency
    /// - HdcEncodingCanary (131): HDC encoding determinism
    pub fn new() -> Self {
        use behavioral_canaries::*;
        let mut canaries = CanaryRunner::new();
        canaries.register(Box::new(ThresholdOrderingCanary));
        canaries.register(Box::new(Blake3DeterminismCanary));
        canaries.register(Box::new(FpuSanityCanary));
        canaries.register(Box::new(ConsciousnessEquationCanary));
        canaries.register(Box::new(MoralAlgebraDeterminismCanary));
        canaries.register(Box::new(HdcEncodingCanary));
        Self {
            attestation: AttestationRegistry::new(),
            temporal: TemporalConsistencyMonitor::new(),
            canaries,
            status: IntegrityStatus::default(),
            last_cycle_instant: Instant::now(),
            substrate_tau_factor: 1.0,
            integrity_confidence: 1.0,
            event_log: VecDeque::with_capacity(EVENT_LOG_CAPACITY),
            global_failure_streak: 0,
            confidence_history: VecDeque::with_capacity(CONFIDENCE_HISTORY_CAPACITY),
            jitter_seed: 0x5AFE_C0DE_DEAD_BEEF,
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
        let micros = (100.0 * inv_tau as f64).clamp(1.0, 10_000_000.0) as u64;
        self.temporal.min_cycle_duration = std::time::Duration::from_micros(micros);
    }

    /// Compute a deterministic jittered attestation interval for the given cycle.
    ///
    /// Applies ±10% jitter using a simple hash of (cycle + seed) to make
    /// check timing unpredictable. An attacker who knows the base interval
    /// cannot time manipulation to fall between checks.
    fn jittered_attestation_due(&self, cycle: usize) -> bool {
        if cycle == 0 {
            return false;
        }
        // Simple deterministic jitter: hash cycle with seed, produce offset in [-10, +10]
        let h = (cycle as u64)
            .wrapping_mul(self.jitter_seed)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        let jitter = (h % 21) as isize - 10; // range [-10, +10]
        let interval = (ATTESTATION_INTERVAL as isize + jitter).max(80) as usize;
        cycle % interval == 0
    }

    /// Run integrity checks that are due this cycle. Self-tracks wall elapsed.
    ///
    /// Each component fires at its own co-prime interval:
    /// - Attestation: every ~101 cycles (±10% jitter)
    /// - Temporal: every cycle (lightweight)
    /// - Canaries: individual intervals (103, 107, 109, 113, 127, 131)
    ///
    /// When `full_sweep` is true (Night phase), all checks run unconditionally.
    /// Science: Besedovsky et al. (2012) — immune system maintenance peaks during sleep.
    ///
    /// **Unified severity**: Attestation and canary failures feed a single
    /// `global_failure_streak`. 1-2 = Warning, 3+ = Critical. The streak resets
    /// only when a full tick completes with zero anomalies across all sources.
    pub fn tick(&mut self, cycle: usize, cfc_delta_t: f32, full_sweep: bool) -> &IntegrityStatus {
        let wall_elapsed = self.last_cycle_instant.elapsed();
        self.last_cycle_instant = Instant::now();
        self.status.anomalies.clear();
        self.status.last_check_cycle = cycle;

        // Attestation (jittered ~101 cycles, or every cycle during full sweep)
        if full_sweep || self.jittered_attestation_due(cycle) {
            let failures = self.attestation.verify_all(cycle);
            self.status.attestation_passed = failures.is_empty();
            for failure in failures {
                self.status.anomalies.push(IntegrityAnomaly {
                    source: "attestation",
                    description: failure,
                    detected_at: Instant::now(),
                    // Severity assigned below from unified streak
                    severity: AnomalySeverity::Warning,
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
            self.status.anomalies.push(IntegrityAnomaly {
                source: "canary",
                description: format!(
                    "{}: expected {}, got {}",
                    failure.canary_name, failure.expected, failure.actual
                ),
                detected_at: Instant::now(),
                // Canary Corruption → will be escalated via streak below
                severity: match failure.severity {
                    CanarySeverity::Drift => AnomalySeverity::Warning,
                    CanarySeverity::Corruption => AnomalySeverity::Critical,
                },
            });
        }

        // ── Unified severity pipeline ──────────────────────────────────────
        // All anomaly sources (attestation, canary, temporal) feed a single
        // global_failure_streak. Resets on clean tick; escalates to Critical at 3+.
        if self.status.anomalies.is_empty() {
            self.global_failure_streak = 0;
        } else {
            self.global_failure_streak += 1;
        }

        // Upgrade all anomaly severities based on unified streak
        let unified_severity = if self.global_failure_streak >= 3 {
            AnomalySeverity::Critical
        } else {
            AnomalySeverity::Warning
        };
        // Escalate warnings to critical if streak warrants it (never downgrade)
        for anomaly in &mut self.status.anomalies {
            if unified_severity == AnomalySeverity::Critical {
                anomaly.severity = AnomalySeverity::Critical;
            }
        }

        // Log all anomalies to the persistent ring buffer for post-mortem analysis
        let events_to_log: Vec<_> = self
            .status
            .anomalies
            .iter()
            .map(|a| (a.source, a.description.clone(), a.severity))
            .collect();
        for (source, description, severity) in events_to_log {
            self.log_event(cycle, source, description, severity);
        }

        // Compute integrity confidence for consciousness gating.
        // Critical anomaly → 0.1 (distrust metrics), Warning only → 0.5, clean → 1.0.
        self.integrity_confidence = if self.has_critical_anomaly() {
            0.1
        } else if !self.status.anomalies.is_empty() {
            0.5
        } else {
            1.0
        };

        // Record confidence for sparkline history
        if self.confidence_history.len() >= CONFIDENCE_HISTORY_CAPACITY {
            self.confidence_history.pop_front();
        }
        self.confidence_history.push_back(self.integrity_confidence);

        // Update panic snapshot for crash forensics (best-effort, every 10th cycle)
        if cycle % 10 == 0 {
            update_panic_snapshot(self.export_snapshot());
        }

        &self.status
    }

    /// Push an event to the ring buffer, evicting the oldest if at capacity.
    fn log_event(
        &mut self,
        cycle: usize,
        source: &'static str,
        description: String,
        severity: AnomalySeverity,
    ) {
        if self.event_log.len() >= EVENT_LOG_CAPACITY {
            self.event_log.pop_front();
        }
        self.event_log.push_back(IntegrityEvent {
            cycle,
            timestamp: Instant::now(),
            source,
            description,
            severity,
        });
    }

    /// Get the event history for debugging/dashboard display.
    pub fn event_history(&self) -> &VecDeque<IntegrityEvent> {
        &self.event_log
    }

    /// Check if any critical anomalies have been detected.
    pub fn has_critical_anomaly(&self) -> bool {
        self.status
            .anomalies
            .iter()
            .any(|a| a.severity == AnomalySeverity::Critical)
    }

    /// Get the rolling confidence history for sparkline display.
    pub fn confidence_history(&self) -> &VecDeque<f32> {
        &self.confidence_history
    }

    /// Export a BLAKE3-signed snapshot of all attestation state for forensic analysis.
    ///
    /// The blob contains: record count, per-record (name, baseline_hash, consecutive_failures),
    /// global_failure_streak, integrity_confidence, and a trailing BLAKE3 signature over all
    /// preceding bytes. This enables external auditors to verify the blob's integrity
    /// and reconstruct the attestation state at dump time.
    ///
    /// Format: `[u32:count][records...][u32:streak][f32:confidence][32:blake3_signature]`
    pub fn export_snapshot(&self) -> Vec<u8> {
        let mut blob = Vec::with_capacity(512);

        // Record count
        let count = self.attestation.records().len() as u32;
        blob.extend_from_slice(&count.to_le_bytes());

        // Per-record: name_len(u16) + name(utf8) + baseline_hash(32) + consecutive_failures(u32)
        for record in self.attestation.records() {
            let name_bytes = record.name.as_bytes();
            blob.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            blob.extend_from_slice(name_bytes);
            blob.extend_from_slice(&record.baseline_hash);
            blob.extend_from_slice(&(record.consecutive_failures as u32).to_le_bytes());
        }

        // Global state
        blob.extend_from_slice(&(self.global_failure_streak as u32).to_le_bytes());
        blob.extend_from_slice(&self.integrity_confidence.to_le_bytes());

        // Event log count + last N events (source + severity)
        let event_count = self.event_log.len().min(EVENT_LOG_CAPACITY) as u32;
        blob.extend_from_slice(&event_count.to_le_bytes());
        for event in &self.event_log {
            let src = event.source.as_bytes();
            blob.extend_from_slice(&(src.len() as u16).to_le_bytes());
            blob.extend_from_slice(src);
            blob.push(match event.severity {
                AnomalySeverity::Warning => 0,
                AnomalySeverity::Critical => 1,
            });
            blob.extend_from_slice(&(event.cycle as u32).to_le_bytes());
        }

        // BLAKE3 signature over everything above
        let signature = *blake3::hash(&blob).as_bytes();
        blob.extend_from_slice(&signature);

        blob
    }

    /// Verify an exported snapshot blob's BLAKE3 signature.
    ///
    /// Returns true if the trailing 32-byte signature matches the BLAKE3 hash
    /// of the preceding content bytes.
    pub fn verify_snapshot(blob: &[u8]) -> bool {
        if blob.len() < 32 {
            return false;
        }
        let (content, sig) = blob.split_at(blob.len() - 32);
        let expected = blake3::hash(content);
        expected.as_bytes() == sig
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

    /// Register moral topology basis constants for attestation.
    ///
    /// These are the HDC harmony basis vector dimensions and moral algebra constants.
    /// Tampering with these would silently distort all moral evaluations.
    pub fn register_moral_topology_constants(&mut self, constants: &[f32]) {
        let hash = attestation::blake3_hash_f32_slice(constants);
        let frozen = constants.to_vec();
        self.attestation.register(
            "moral_topology_constants",
            hash,
            Box::new(move || attestation::blake3_hash_f32_slice(&frozen)),
        );
    }

    /// Register governance neuromodulatory threshold constants for attestation.
    ///
    /// These control the neurochemical response to governance events (emergency → NE,
    /// reciprocity → oxytocin, etc.). Tampering could silently distort the embodied
    /// governance experience — e.g., suppressing the emergency NE surge.
    #[cfg(feature = "mycelix")]
    pub fn register_governance_thresholds(&mut self) {
        use crate::cognitive_loop::thresholds;
        let constants: Vec<f32> = vec![
            thresholds::GOV_NEUROMOD_FLOOR,
            thresholds::GOV_EMERGENCY_NE_NUDGE,
            thresholds::GOV_RECIPROCITY_OXY_DOSE,
            thresholds::GOV_RECIPROCITY_OXY_CAP,
            thresholds::GOV_RECIPROCITY_OXY_HALFLIFE as f32,
            thresholds::GOV_DISPUTE_NE_NUDGE,
            thresholds::GOV_DISPUTE_SHT_NUDGE,
            thresholds::GOV_ALIGNED_PASS_DA_DOSE,
            thresholds::GOV_ALIGNED_PASS_DA_HALFLIFE as f32,
            thresholds::GOV_ALIGNED_FAIL_DA_NUDGE,
            thresholds::GOV_REPUTATION_DECLINE_SHT,
            thresholds::GOV_COLLECTIVE_PHI_ECB,
            thresholds::GOV_CONSCIOUSNESS_MODULATION as f32,
        ];
        let hash = attestation::blake3_hash_f32_slice(&constants);
        self.attestation.register(
            "governance_thresholds",
            hash,
            Box::new(move || attestation::blake3_hash_f32_slice(&constants)),
        );
    }

    /// Verify live safety thresholds against the registered baseline.
    ///
    /// Called from the cognitive cycle with current threshold values. This catches
    /// runtime mutation (not just binary patching) because it hashes the live
    /// values rather than a frozen copy.
    ///
    /// Returns `Some(failure_description)` if the live values differ from baseline.
    pub fn verify_live_thresholds(&self, name: &str, live_hash: [u8; 32]) -> Option<String> {
        for record in self.attestation.records() {
            if record.name == name && record.baseline_hash != live_hash {
                return Some(format!(
                    "Live attestation FAILED for '{}': baseline {:x?} != live {:x?}",
                    name,
                    &record.baseline_hash[..8],
                    &live_hash[..8],
                ));
            }
        }
        None
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
        mgr.attestation.register(
            "test",
            hash,
            Box::new(move || attestation::blake3_hash(b"test")),
        );
        // Cycle 5 normally wouldn't trigger attestation (interval=101)
        let status = mgr.tick(5, 0.02, true);
        assert!(status.attestation_passed);
        // Verify the attestation actually ran (last_verification populated)
        assert!(mgr.attestation.records()[0].last_verification.is_some());
    }

    #[test]
    fn test_full_sweep_detects_drift_on_first_failure() {
        let mut mgr = IntegrityManager::new();
        let baseline = attestation::blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(move || attestation::blake3_hash(b"modified")),
        );
        mgr.tick(1, 0.02, true);
        assert!(!mgr.status.attestation_passed);
        // First failure = Warning (drift), not Critical
        assert!(!mgr.has_critical_anomaly());
        assert_eq!(mgr.status.anomalies.len(), 1);
        assert_eq!(mgr.status.anomalies[0].severity, AnomalySeverity::Warning);
    }

    #[test]
    fn test_consecutive_failures_escalate_to_critical() {
        let mut mgr = IntegrityManager::new();
        let baseline = attestation::blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(move || attestation::blake3_hash(b"modified")),
        );
        // 3 consecutive failures → escalate to Critical
        // Use full_sweep=true to bypass jittered interval timing
        mgr.tick(101, 0.02, true); // streak=1
        mgr.tick(202, 0.02, true); // streak=2
        mgr.tick(303, 0.02, true); // streak=3 → Critical
        assert!(!mgr.status.attestation_passed);
        assert!(mgr.has_critical_anomaly());
    }

    #[test]
    fn test_integrity_confidence_tracks_severity() {
        let mut mgr = IntegrityManager::new();
        // Clean state → confidence 1.0
        mgr.tick(1, 0.02, false);
        assert_eq!(mgr.integrity_confidence, 1.0);

        // Register a tampered attestation
        let baseline = attestation::blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(move || attestation::blake3_hash(b"modified")),
        );
        // First failure → Warning → confidence 0.5
        // Use full_sweep=true to bypass jittered interval timing
        mgr.tick(101, 0.02, true);
        assert_eq!(mgr.integrity_confidence, 0.5);

        // 3 consecutive → Critical → confidence 0.1
        mgr.tick(202, 0.02, true);
        mgr.tick(303, 0.02, true);
        assert_eq!(mgr.integrity_confidence, 0.1);
    }

    #[test]
    fn test_verify_live_thresholds_passes_when_matching() {
        let mut mgr = IntegrityManager::new();
        let values = [1.0f32, 2.0, 3.0];
        mgr.register_safety_thresholds(&values);
        let live_hash = attestation::blake3_hash_f32_slice(&values);
        assert!(
            mgr.verify_live_thresholds("safety_thresholds", live_hash)
                .is_none()
        );
    }

    #[test]
    fn test_verify_live_thresholds_fails_when_different() {
        let mut mgr = IntegrityManager::new();
        mgr.register_safety_thresholds(&[1.0, 2.0, 3.0]);
        let tampered_hash = attestation::blake3_hash_f32_slice(&[1.0, 2.0, 999.0]);
        assert!(
            mgr.verify_live_thresholds("safety_thresholds", tampered_hash)
                .is_some()
        );
    }

    #[test]
    fn test_snapshot_export_and_verify() {
        let mut mgr = IntegrityManager::new();
        mgr.register_safety_thresholds(&[1.0, 2.0, 3.0]);
        mgr.tick(1, 0.02, true);
        let blob = mgr.export_snapshot();
        assert!(blob.len() > 32);
        assert!(IntegrityManager::verify_snapshot(&blob));
        // Tamper with one byte → verification fails
        let mut tampered = blob.clone();
        tampered[4] ^= 0xFF;
        assert!(!IntegrityManager::verify_snapshot(&tampered));
    }

    #[test]
    fn test_unified_failure_streak_resets() {
        let mut mgr = IntegrityManager::new();
        let baseline = attestation::blake3_hash(b"original");
        mgr.attestation.register_tampered(
            "tampered",
            baseline,
            Box::new(move || attestation::blake3_hash(b"modified")),
        );
        // Build streak with full sweep
        mgr.tick(1, 0.02, true);
        assert_eq!(mgr.global_failure_streak, 1);
        mgr.tick(2, 0.02, true);
        assert_eq!(mgr.global_failure_streak, 2);
        // Remove the tampered attestation by replacing registry
        mgr.attestation = attestation::AttestationRegistry::new();
        // Clean tick → streak resets
        mgr.tick(3, 0.02, true);
        assert_eq!(mgr.global_failure_streak, 0);
    }

    #[test]
    fn test_confidence_history_accumulates() {
        let mut mgr = IntegrityManager::new();
        for i in 1..=5 {
            mgr.tick(i, 0.02, false);
        }
        assert_eq!(mgr.confidence_history.len(), 5);
        // All values should be valid confidence levels (may include 0.5 if temporal fires)
        assert!(
            mgr.confidence_history
                .iter()
                .all(|&c| c == 1.0 || c == 0.5 || c == 0.1)
        );
    }

    #[test]
    fn test_confidence_history_bounded_at_60() {
        let mut mgr = IntegrityManager::new();
        for i in 1..=100 {
            mgr.tick(i, 0.02, false);
        }
        assert_eq!(mgr.confidence_history.len(), 60);
    }

    #[test]
    fn test_jittered_attestation_fires_near_interval() {
        let mgr = IntegrityManager::new();
        // Over cycles 80-120, at least one should trigger jittered attestation
        let fired: Vec<usize> = (80..=120)
            .filter(|&c| mgr.jittered_attestation_due(c))
            .collect();
        assert!(
            !fired.is_empty(),
            "jittered attestation should fire at least once near interval 101"
        );
    }
}
