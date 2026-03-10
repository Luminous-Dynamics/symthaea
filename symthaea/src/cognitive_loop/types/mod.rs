//! Public types returned by the cognitive loop.
//!
//! Decomposed from a monolithic types.rs into thematic sub-modules.
//! All public APIs are preserved via re-exports.

mod carryover;
mod output;
mod scheduling;
mod telemetry;

pub use output::*;
pub use scheduling::*;
pub use telemetry::*;

// Re-export crate-visible types
pub(crate) use carryover::{ConsciousnessCache, CycleCarryover, QualityMetrics};
// Used by test modules — gate to suppress unused-import warnings in lib builds
#[cfg(test)]
pub(crate) use carryover::{LearningState, UrgencyState};
pub(crate) use scheduling::CycleState;

// ── Substrate Telemetry ─────────────────────────────────────────────────────

/// Substrate telemetry snapshot returned by `SubstrateManager::telemetry()`.
///
/// Groups all substrate-related fields into a single struct for assignment
/// to `CycleMetadata.substrate` via `metadata.substrate = self.substrate_manager.telemetry()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SubstrateTelemetry {
    /// Effective substrate feasibility [0,1] used in consciousness equation.
    /// Legacy field — identical to `substrate_effective_feasibility`.
    pub substrate_feasibility: f64,
    /// Describes a substrate transition that occurred during this cycle (if any).
    pub substrate_transition: Option<String>,
    /// Raw substrate feasibility before validation overlay (0.0-1.0).
    pub substrate_feasibility_raw: f64,
    /// Honest evidence confidence for current substrate (0.0-0.95).
    pub substrate_honest_confidence: f64,
    /// Effective feasibility after validation overlay blending (0.0-1.0).
    pub substrate_effective_feasibility: f64,
    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    #[serde(default = "default_one_f32_substrate")]
    pub substrate_tau_factor: f32,
    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    #[serde(default)]
    pub substrate_scale_pressure: f32,
    /// Per-region feasibility breakdown (empty when per-region not configured).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_region_feasibility: Vec<(String, f32)>,
    /// HDC encoding noise fraction applied this cycle [0.0, 0.1].
    /// Non-zero when substrate encoding noise is enabled and scale_pressure < 0.
    #[serde(default)]
    pub substrate_encoding_noise: f32,
}

fn default_one_f32_substrate() -> f32 {
    1.0
}

// ── Integrity Telemetry ───────────────────────────────────────────────────

/// Integrity telemetry snapshot from IntegrityManager.
///
/// Reports tamper detection status: attestation, temporal consistency,
/// behavioral canaries. Feature-gated behind `integrity`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IntegrityTelemetry {
    /// Whether all BLAKE3 attestation hashes matched.
    pub attestation_passed: bool,
    /// Whether temporal consistency (wall clock vs CfC delta_t) passed.
    pub temporal_passed: bool,
    /// Whether all behavioral canaries returned expected results.
    pub canaries_passed: bool,
    /// Number of anomalies detected this cycle.
    pub anomaly_count: usize,
    /// Whether any anomaly has Critical severity.
    pub has_critical: bool,
    /// Cycle number of the last integrity check.
    pub last_check_cycle: usize,
    /// Consciousness confidence multiplier (1.0 = trusted, 0.5 = drift, 0.1 = critical).
    #[serde(default = "default_integrity_confidence")]
    pub integrity_confidence: f32,
    /// Per-attestation detail: (name, passed, consecutive_failures).
    /// Empty when no attestation check ran this cycle.
    #[serde(default)]
    pub attestation_details: Vec<AttestationDetail>,
    /// Unified cross-source failure streak (attestation + canary).
    /// 1-2 = Warning, 3+ = Critical. Resets on clean tick.
    #[serde(default)]
    pub global_failure_streak: usize,
    /// Rolling 60-cycle history of integrity_confidence values for sparkline display.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub confidence_history: Vec<f32>,
}

/// Per-attestation telemetry entry for dashboard visibility.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AttestationDetail {
    /// Attestation name (e.g., "safety_thresholds").
    pub name: String,
    /// Whether this attestation passed on last check.
    pub passed: bool,
    /// Number of consecutive failures (0 = healthy).
    pub consecutive_failures: usize,
}

fn default_integrity_confidence() -> f32 {
    1.0
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
