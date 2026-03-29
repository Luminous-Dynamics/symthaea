// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public types returned by the cognitive loop.
//!
//! Decomposed from a monolithic types.rs into thematic sub-modules.
//! All public APIs are preserved via re-exports.

pub mod carryover;
mod output;
mod scheduling;
pub(crate) mod telemetry;

pub use output::*;
pub use scheduling::*;
pub use telemetry::*;

// Re-export crate-visible types
pub use carryover::{ConsciousnessCache, CycleCarryover, QualityMetrics};
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
    /// Total energy spent so far (joules). Monotonically increasing.
    #[serde(default)]
    pub total_energy_spent: f64,
    /// Energy spent this cycle (joules, speed-adjusted via tau_factor).
    #[serde(default)]
    pub energy_this_cycle: f64,
    /// Energy throughput multiplier (ratio of bio energy to substrate energy).
    #[serde(default = "default_one_f32_substrate")]
    pub energy_throughput_multiplier: f32,
    /// Effective HDC/CfC dimensionality fraction [0.1, 1.0].
    /// 1.0 for substrates at or above biological scale.
    #[serde(default = "default_one_f32_substrate")]
    pub effective_dim_fraction: f32,
    /// Number of substrate transitions recorded so far.
    #[serde(default)]
    pub transition_count: usize,
}

fn default_one_f32_substrate() -> f32 {
    1.0
}

// ── Thermal Telemetry ─────────────────────────────────────────────────────

/// Thermal telemetry snapshot from ThermalBridge.
///
/// Reports platform thermal state and its effect on CfC tau modulation.
/// Populated each cycle from `thermal_bridge.signals()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ThermalTelemetry {
    /// Current platform thermal level (0=Nominal, 1=Fair, 2=Serious, 3=Critical, 4=Emergency).
    #[serde(default)]
    pub thermal_level: u8,
    /// EMA-smoothed CfC tau factor from thermal state [1.0, 2.5].
    /// Multiplied into delta_t as the 10th modulation factor.
    #[serde(default = "default_one_f64_thermal")]
    pub thermal_tau_factor: f64,
    /// Whether the thermal bridge recommends a consciousness profile downgrade.
    #[serde(default)]
    pub should_reduce_profile: bool,
    /// Recommended frequency cap from thermal state (None = no override).
    #[serde(default)]
    pub target_frequency_override: Option<f32>,
}

fn default_one_f64_thermal() -> f64 {
    1.0
}

// ── Integrity Telemetry ───────────────────────────────────────────────────

/// Integrity telemetry snapshot from IntegrityManager.
///
/// Reports tamper detection status: attestation, temporal consistency,
/// behavioral canaries. Feature-gated behind `integrity`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// When the `integrity` feature is off, IntegrityTelemetry defaults to "all pass, fully trusted".
/// This prevents downstream consumers (Pulse, SafetyAgent) from seeing uninitialized false-alarm data.
impl Default for IntegrityTelemetry {
    fn default() -> Self {
        Self {
            attestation_passed: true,
            temporal_passed: true,
            canaries_passed: true,
            anomaly_count: 0,
            has_critical: false,
            last_check_cycle: 0,
            integrity_confidence: 1.0,
            attestation_details: Vec::new(),
            global_failure_streak: 0,
            confidence_history: Vec::new(),
        }
    }
}

fn default_integrity_confidence() -> f32 {
    1.0
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
