// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared embodiment types for consciousness-coupled motor control.
//!
//! These types are the canonical definitions used by the cognitive loop,
//! all platform crates (flight, vehicle, helicopter, manipulator, AUV),
//! and the hardware abstraction layer (HAL).

use serde::{Deserialize, Serialize};

use crate::hdc::ContinuousHV;

// ── Motor Safety Level ─────────────────────────────────────────────────────

/// NRC-inspired safety level for motor output gating.
///
/// Maps to consciousness level (Phi) thresholds:
/// - Green (Phi > 0.6): full authority
/// - Yellow (0.3–0.6): reduced speed/force
/// - Orange (0.1–0.3): retreat to safe pose
/// - Red (< 0.1): emergency stop, power cut
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MotorSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

impl MotorSafetyLevel {
    /// Determine safety level from current Phi value.
    ///
    /// Non-finite inputs (NaN, Inf, -Inf) map to `Red` (emergency mode).
    ///
    /// IMPORTANT: `Red` does NOT mean "zero motor force". It means
    /// "execute the platform's SafeFallback behavior" — which for a
    /// vehicle means emergency braking, for an exoskeleton means locking
    /// in standing position, for a helicopter means autorotation, etc.
    /// See the [`SafeFallback`] trait for platform-specific guarantees.
    pub fn from_phi(phi: f64) -> Self {
        if !phi.is_finite() {
            return Self::Red;
        }
        if phi > 0.6 {
            Self::Green
        } else if phi > 0.3 {
            Self::Yellow
        } else if phi > 0.1 {
            Self::Orange
        } else {
            Self::Red
        }
    }

    /// Motor gain multiplier for consciousness-modulated commanded force.
    ///
    /// This is applied to COMMANDED torques only. Platforms that implement
    /// [`SafeFallback`] produce an overriding command at Red that is NOT
    /// scaled by this gain — the safe fallback must execute at full authority
    /// to guarantee minimum safe behavior (emergency braking, lock, etc.).
    ///
    /// - Green: 1.0 (full commanded authority)
    /// - Yellow: 0.6 (reduced)
    /// - Orange: 0.3 (compliant)
    /// - Red: 0.0 for commanded torques (but SafeFallback executes at 1.0)
    pub fn motor_gain(&self) -> f32 {
        match self {
            Self::Green => 1.0,
            Self::Yellow => 0.6,
            Self::Orange => 0.3,
            Self::Red => 0.0,
        }
    }
}

// ── Embodiment Result ──────────────────────────────────────────────────────

/// Result of a single embodiment step.
#[derive(Debug, Clone)]
pub struct EmbodimentResult {
    pub num_actuators: usize,
    pub control_effort: f32,
    pub success: bool,
    pub prediction_error: f32,
    pub safety_level: MotorSafetyLevel,
    /// Epistemic grounding level: 0=Sensorimotor, 1=Temporal, 2=Social.
    /// Uses u8 to keep symthaea-core free of epistemic-types dependency.
    pub epistemic_grounding: u8,
    /// Observation confidence derived from prediction error (low error = high confidence).
    pub observation_confidence: f32,
}

// ── Embodiment Telemetry ───────────────────────────────────────────────────

/// Telemetry from the embodiment subsystem, exposed in CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbodimentTelemetry {
    pub total_steps: u64,
    pub control_effort: f32,
    pub prediction_error: f32,
    pub safety_level: String,
    pub platform: String,
    pub num_actuators: usize,
    /// Human-readable epistemic grounding level (e.g. "Sensorimotor", "Temporal", "Social").
    pub epistemic_grounding: String,
    /// Observation confidence derived from prediction error.
    pub observation_confidence: f32,
    /// Platform-specific telemetry bytes (serialized by each platform).
    /// Empty by default; platforms override `platform_telemetry_bytes()` to populate.
    #[serde(default)]
    pub platform_specific: Vec<u8>,
}

// ── Epistemic Grounding Constants ─────────────────────────────────────────

/// Sensorimotor grounding: direct physical sensor data.
pub const GROUNDING_SENSORIMOTOR: u8 = 0;
/// Temporal grounding: derived from temporal patterns / prediction.
pub const GROUNDING_TEMPORAL: u8 = 1;
/// Social grounding: derived from social/swarm interaction.
pub const GROUNDING_SOCIAL: u8 = 2;

/// Convert a grounding level u8 to a human-readable string.
pub fn grounding_label(level: u8) -> &'static str {
    match level {
        0 => "Sensorimotor",
        1 => "Temporal",
        2 => "Social",
        _ => "Unknown",
    }
}

// ── Grounding Estimator ──────────────────────────────────────────────

/// Estimates epistemic grounding level from prediction error trends
/// and optional peer interaction data.
///
/// - **Sensorimotor** (default): Raw proprioceptive data, no temporal model.
/// - **Temporal**: Prediction errors are consistently decreasing and low — the
///   platform has learned a temporal model of its body dynamics.
/// - **Social**: Prediction incorporates multi-agent data (mesh swarm peers).
#[derive(Debug, Clone)]
pub struct GroundingEstimator {
    /// Rolling window of prediction errors (most recent at end).
    window: Vec<f32>,
    /// Maximum window size.
    capacity: usize,
}

impl GroundingEstimator {
    /// Create with a 32-sample rolling window.
    pub fn new() -> Self {
        Self {
            window: Vec::with_capacity(32),
            capacity: 32,
        }
    }

    /// Update with the latest prediction error and return the estimated grounding level.
    ///
    /// `peer_count`: number of mesh/swarm peers contributing to perception.
    /// Pass `None` or `Some(0)` for platforms without social interaction.
    pub fn estimate(&mut self, prediction_error: f32, peer_count: Option<usize>) -> u8 {
        // Update rolling window
        if self.window.len() >= self.capacity {
            self.window.remove(0);
        }
        self.window.push(prediction_error);

        // Social grounding: requires peers and a stable temporal model
        if let Some(peers) = peer_count
            && peers > 0
            && self.window.len() >= 16
        {
            let mean = self.window.iter().sum::<f32>() / self.window.len() as f32;
            if mean < 0.3 {
                return GROUNDING_SOCIAL;
            }
        }

        // Temporal grounding: full window, mean < 0.3, and trend is decreasing
        if self.window.len() >= self.capacity {
            let mean = self.window.iter().sum::<f32>() / self.window.len() as f32;
            if mean < 0.3 {
                // Check trend: compare first half mean to second half mean
                let half = self.capacity / 2;
                let first_half: f32 = self.window[..half].iter().sum::<f32>() / half as f32;
                let second_half: f32 = self.window[half..].iter().sum::<f32>() / half as f32;
                if second_half <= first_half {
                    return GROUNDING_TEMPORAL;
                }
            }
        }

        GROUNDING_SENSORIMOTOR
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

impl Default for GroundingEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute observation confidence from prediction error.
///
/// Low prediction error implies high confidence in the observation.
/// Returns a value in \[0.0, 1.0\].
pub fn grounding_from_prediction_error(prediction_error: f32) -> f32 {
    1.0 - prediction_error.clamp(0.0, 1.0)
}

// ── Embodiment Platform ────────────────────────────────────────────────────

/// Which embodiment platform to use.
///
/// Each variant (except `None` and `CareProvider`) maps to a robotics crate
/// behind a feature flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbodimentPlatform {
    #[default]
    None,
    Humanoid,
    Quadrotor,
    Vehicle,
    Helicopter,
    Auv,
    Manipulator,
    /// Lower-limb exoskeleton — shared authority with human user.
    Exoskeleton,
    /// Surgical robot — sub-mm RCM-constrained precision.
    Surgical,
    /// Orbital servicing arm — zero-g dual-body dynamics.
    Orbital,
    /// Legged quadruped — CPG-modulated locomotion.
    Quadruped,
    /// Subterranean scout / boring platform — digging, spoil, thermal load.
    Subterranean,
    /// Stationary infrastructure agent — hub, storage, microgrid, routing.
    Infrastructure,
    /// Recovery / recycling platform — salvage, teardown, materials sorting.
    Scavenger,
    /// Ecological stewardship platform — soil, water, light, crop tending.
    Agribot,
    /// Interspecies sanctuary platform — animal right-of-way and distress sensing.
    Biota,
    /// Habitat homeostasis platform — atmosphere, HVAC, circadian support.
    Clime,
    /// Virtual caregiver agent — no physical embodiment.
    CareProvider,
    /// Browser agent — web pages as sensory environment via CDP.
    Browser,
    /// Phone screen — ADB-controlled Android device as sensory environment.
    Phone,
    /// Desktop screen — mouse/keyboard control via xdotool/ydotool.
    Desktop,
}

// ── Agent Identity ────────────────────────────────────────────────────────

/// Platform-agnostic agent identity for embodied components.
///
/// In Holochain contexts this maps to a uhCAk... Ed25519 public key.
/// In standalone Symthaea it is a blake3 hash of the genesis seed.
/// This keeps `symthaea-core` free of Holochain dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentIdentity(pub String);

impl AgentIdentity {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Safe Fallback Trait ────────────────────────────────────────────────────

/// Safe fallback behaviors for consciousness degradation.
///
/// # The Core Safety Principle
///
/// When consciousness degrades, a machine does not lose its strength —
/// it loses its goal-directed authority. The machine's MINIMUM-SAFE
/// behavior must execute regardless of consciousness state.
///
/// This trait defines what "minimum-safe behavior" means for each platform:
///
/// | Platform | Red-tier Safe Fallback |
/// |----------|------------------------|
/// | Manipulator | Hold current pose (gravity compensation) |
/// | Helicopter | Autorotation descent |
/// | Quadrotor | Controlled emergency landing |
/// | **Vehicle** | **Maximum brake + hazards + lane hold** |
/// | **Exoskeleton** | **Lock legs in standing position** |
/// | AUV | Buoyancy ascent to surface |
/// | Quadruped | Controlled sit/lie-down |
/// | Surgical | Freeze in place, retract cautery |
///
/// # Design Principle
///
/// `motor_gain=0` at Red is the DEFAULT behavior that requires no
/// platform-specific safety knowledge — it's safe for manipulators
/// and stationary arms. But for platforms carrying humans, moving at
/// speed, or operating in environments where "no force" means "crash",
/// the platform MUST implement this trait and override the step() logic
/// to execute the safe fallback instead of the zero command.
///
/// # Invariants
///
/// 1. `safe_fallback_active()` returns true iff safety_level is Red or Orange
/// 2. `safe_fallback_priority()` returns a numeric priority (higher = more critical)
/// 3. The safe fallback command executes at FULL authority, not scaled by motor_gain
/// 4. Safe fallback must NEVER require consciousness to execute (it's reflexive)
pub trait SafeFallback {
    /// Platform name for telemetry.
    fn platform_name(&self) -> &'static str;

    /// Whether the safe fallback is currently active (at Red/Orange safety).
    fn safe_fallback_active(&self) -> bool {
        matches!(
            self.current_safety_level(),
            MotorSafetyLevel::Red | MotorSafetyLevel::Orange
        )
    }

    /// Current safety level driving fallback behavior.
    fn current_safety_level(&self) -> MotorSafetyLevel;

    /// Priority of this platform's safe fallback:
    /// - 10: Critical (human-carrying, airborne, moving)
    /// - 5: High (grasping, in-contact)
    /// - 1: Low (stationary, contained)
    fn safe_fallback_priority(&self) -> u8;

    /// Human-readable description of the safe fallback behavior.
    fn safe_fallback_description(&self) -> &'static str;

    /// Latency in simulation cycles from Red trigger to fallback active.
    /// MUST be 1 for human-carrying platforms.
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

// ── Embodiment Bridge Trait ────────────────────────────────────────────────

/// Platform-agnostic embodiment bridge trait.
///
/// Any robotics platform that wants to close the proprioceptive loop
/// with the cognitive loop must implement this trait.
pub trait EmbodimentBridge: Send + Sync {
    /// Translate thought → motor commands, step physics, return result.
    /// `phi` is the current consciousness level for safety gating.
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult;

    /// Apply active inference motor reflexes (Channel 1/3) based on prediction error.
    ///
    /// Prediction error (surprise) can trigger reflexive motor actions to
    /// minimize that surprise by changing the physical world (active inference).
    fn apply_active_inference(&mut self, error: &crate::hdc::unified_hv::ContinuousHV) {
        // Default implementation is a no-op; platforms override to generate impulses.
        let _ = error;
    }

    /// Encode the current body state into a proprioceptive 16,384D HV.
    fn encode_perception(&mut self) -> ContinuousHV;

    /// Reset body to default state.
    fn reset(&mut self);

    /// Current safety level.
    fn safety_level(&self) -> MotorSafetyLevel;

    /// Override safety level from SafetyAgent. Merged with phi via max().
    fn set_safety_override(&mut self, level: MotorSafetyLevel);

    /// Clear external safety override.
    fn clear_safety_override(&mut self);

    /// Platform identifier.
    fn platform(&self) -> EmbodimentPlatform;

    /// Number of actuators.
    fn num_actuators(&self) -> usize;
    /// Return current health/reliability for each actuator [0.0, 1.0].
    fn actuator_health(&self) -> Vec<f32> {
        vec![1.0; self.num_actuators()]
    }

    /// Return the last encoded perception HV.
    fn last_perception_hv(&self) -> Option<ContinuousHV> {
        None
    }

    /// Return current sensorimotor accuracy [0.0, 1.0].
    fn sensorimotor_accuracy(&self) -> f32 {
        1.0
    }

    /// Total steps executed.
    fn total_steps(&self) -> usize;

    /// Get telemetry for CycleMetadata.
    fn telemetry(&self) -> EmbodimentTelemetry;

    /// Platform-agnostic agent identity for trust and coordination.
    ///
    /// Default returns a placeholder; platform crates should override with
    /// a deterministic identity derived from their genesis seed.
    fn agent_identity(&self) -> AgentIdentity {
        AgentIdentity::new("unset")
    }

    /// Physical reliability score (0.0–1.0) for reputation/trust gating.
    ///
    /// 1.0 = fully reliable sensor/actuator chain. Platforms with degraded
    /// hardware should report lower values.
    fn physical_reliability(&self) -> f64 {
        1.0
    }

    /// Apply moral gate from the ethics engine.
    ///
    /// Platforms opt in by overriding this method. The default is a no-op.
    /// Safety-critical platforms (manipulator, surgical) should override
    /// to enforce: Blocked → Red, Caution → Yellow cap, consent → Orange,
    /// ahimsa → Red.
    fn apply_moral_gate(&mut self, _gate: MoralGateInput) {}

    /// Platform-specific telemetry bytes for dispatch zome integration.
    ///
    /// Default returns empty vec. Override to serialize platform-specific
    /// sensor data (e.g., joint angles, chemical readings, rotor RPM).
    fn platform_telemetry_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
}

// ── Moral Gate Input ─────────────────────────────────────────────────

/// Ethics engine output forwarded to the embodiment bridge.
///
/// Uses u8 for `verdict` to keep symthaea-core free of ethics-types dependency:
/// - 0 = Safe (no constraint)
/// - 1 = Caution (cap at Yellow)
/// - 2 = Blocked (force Red)
#[derive(Debug, Clone, Copy)]
pub struct MoralGateInput {
    /// Ethical verdict: 0=Safe, 1=Caution, 2=Blocked.
    pub verdict: u8,
    /// True if consent was violated (force Orange or stricter).
    pub consent_violation: bool,
    /// True if ahimsa (non-harm) principle was violated (force Red).
    pub ahimsa_violated: bool,
}

impl MoralGateInput {
    pub const VERDICT_SAFE: u8 = 0;
    pub const VERDICT_CAUTION: u8 = 1;
    pub const VERDICT_BLOCKED: u8 = 2;
}

// ── Platform Plugin System ───────────────────────────────────────────────

/// Factory + metadata for a robotics platform.
///
/// Each platform crate exports a `pub struct XPlugin;` implementing this trait.
/// The cognitive loop's constructor collects all enabled plugins into a
/// `PlatformRegistry` and dispatches bridge creation by `EmbodimentPlatform`.
///
/// This reduces "add a platform" to 3 steps:
/// 1. Add variant to `EmbodimentPlatform`
/// 2. Add feature flag + optional dep to Cargo.toml
/// 3. Register the plugin in the constructor's `build_registry()`
pub trait PlatformPlugin: Send + Sync {
    /// Which platform this plugin provides.
    fn platform(&self) -> EmbodimentPlatform;

    /// Feature name for diagnostics (e.g., "helicopter", "exoskeleton").
    fn feature_name(&self) -> &'static str;

    /// Number of actuators for this platform.
    fn num_actuators(&self) -> usize;
    /// Return current health/reliability for each actuator [0.0, 1.0].
    fn actuator_health(&self) -> Vec<f32> {
        vec![1.0; self.num_actuators()]
    }

    /// Return the last encoded perception HV.
    fn last_perception_hv(&self) -> Option<ContinuousHV> {
        None
    }

    /// Return current sensorimotor accuracy [0.0, 1.0].
    fn sensorimotor_accuracy(&self) -> f32 {
        1.0
    }

    /// Create a new embodiment bridge from a genesis seed.
    fn create_bridge(&self, genesis: &crate::genesis::GenesisSeed) -> Box<dyn EmbodimentBridge>;
}

/// Compile-time platform registry.
///
/// Collects `PlatformPlugin` implementations registered by feature-gated
/// platform crates. The constructor calls `create_bridge()` on the matching
/// plugin to instantiate the correct embodiment.
pub struct PlatformRegistry {
    plugins: Vec<Box<dyn PlatformPlugin>>,
}

impl PlatformRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register a platform plugin.
    pub fn register(&mut self, plugin: Box<dyn PlatformPlugin>) {
        self.plugins.push(plugin);
    }

    /// Create an embodiment bridge for the given platform.
    /// Returns `None` if no plugin is registered for that platform.
    pub fn create_bridge(
        &self,
        platform: EmbodimentPlatform,
        genesis: &crate::genesis::GenesisSeed,
    ) -> Option<Box<dyn EmbodimentBridge>> {
        self.plugins
            .iter()
            .find(|p| p.platform() == platform)
            .map(|p| p.create_bridge(genesis))
    }

    /// List all registered platforms.
    pub fn registered_platforms(&self) -> Vec<EmbodimentPlatform> {
        self.plugins.iter().map(|p| p.platform()).collect()
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for PlatformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_phi_thresholds() {
        assert_eq!(MotorSafetyLevel::from_phi(0.0), MotorSafetyLevel::Red);
        assert_eq!(MotorSafetyLevel::from_phi(0.1), MotorSafetyLevel::Red);
        assert_eq!(MotorSafetyLevel::from_phi(0.2), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.3), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.5), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.6), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.7), MotorSafetyLevel::Green);
        assert_eq!(MotorSafetyLevel::from_phi(1.0), MotorSafetyLevel::Green);
    }

    #[test]
    fn test_motor_gain_values() {
        assert_eq!(MotorSafetyLevel::Green.motor_gain(), 1.0);
        assert_eq!(MotorSafetyLevel::Yellow.motor_gain(), 0.6);
        assert_eq!(MotorSafetyLevel::Orange.motor_gain(), 0.3);
        assert_eq!(MotorSafetyLevel::Red.motor_gain(), 0.0);
    }

    #[test]
    fn test_ordering() {
        assert!(MotorSafetyLevel::Green < MotorSafetyLevel::Yellow);
        assert!(MotorSafetyLevel::Yellow < MotorSafetyLevel::Orange);
        assert!(MotorSafetyLevel::Orange < MotorSafetyLevel::Red);
    }

    #[test]
    fn test_grounding_from_prediction_error() {
        // Zero error = full confidence
        assert!((grounding_from_prediction_error(0.0) - 1.0).abs() < f32::EPSILON);
        // Full error = zero confidence
        assert!((grounding_from_prediction_error(1.0) - 0.0).abs() < f32::EPSILON);
        // Mid-range
        assert!((grounding_from_prediction_error(0.3) - 0.7).abs() < 1e-6);
        // Clamping: values outside [0,1] are clamped
        assert!((grounding_from_prediction_error(-0.5) - 1.0).abs() < f32::EPSILON);
        assert!((grounding_from_prediction_error(1.5) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_phi_non_finite_returns_red() {
        assert_eq!(MotorSafetyLevel::from_phi(f64::NAN), MotorSafetyLevel::Red);
        assert_eq!(
            MotorSafetyLevel::from_phi(f64::INFINITY),
            MotorSafetyLevel::Red
        );
        assert_eq!(
            MotorSafetyLevel::from_phi(f64::NEG_INFINITY),
            MotorSafetyLevel::Red
        );
    }

    #[test]
    fn test_from_phi_negative_returns_red() {
        assert_eq!(MotorSafetyLevel::from_phi(-1.0), MotorSafetyLevel::Red);
        assert_eq!(MotorSafetyLevel::from_phi(-100.0), MotorSafetyLevel::Red);
    }

    #[test]
    fn test_grounding_labels() {
        assert_eq!(grounding_label(GROUNDING_SENSORIMOTOR), "Sensorimotor");
        assert_eq!(grounding_label(GROUNDING_TEMPORAL), "Temporal");
        assert_eq!(grounding_label(GROUNDING_SOCIAL), "Social");
        assert_eq!(grounding_label(42), "Unknown");
    }

    #[test]
    fn test_grounding_estimator_starts_sensorimotor() {
        let mut est = GroundingEstimator::new();
        // Not enough samples yet
        assert_eq!(est.estimate(0.1, None), GROUNDING_SENSORIMOTOR);
    }

    #[test]
    fn test_grounding_estimator_temporal_after_stable_predictions() {
        let mut est = GroundingEstimator::new();
        // Fill window with low, decreasing prediction errors
        for i in 0..32 {
            let pe = 0.2 - (i as f32 * 0.005); // decreasing from 0.2 to 0.045
            est.estimate(pe, None);
        }
        let level = est.estimate(0.05, None);
        assert_eq!(level, GROUNDING_TEMPORAL);
    }

    #[test]
    fn test_grounding_estimator_social_with_peers() {
        let mut est = GroundingEstimator::new();
        // Fill 16+ samples with low PE and provide peers
        for _ in 0..20 {
            est.estimate(0.1, Some(3));
        }
        let level = est.estimate(0.1, Some(3));
        assert_eq!(level, GROUNDING_SOCIAL);
    }

    #[test]
    fn test_grounding_estimator_no_social_without_peers() {
        let mut est = GroundingEstimator::new();
        for _ in 0..20 {
            est.estimate(0.1, None);
        }
        let level = est.estimate(0.1, None);
        // Without peers, should be temporal at best (if window is full)
        assert_ne!(level, GROUNDING_SOCIAL);
    }

    #[test]
    fn test_grounding_estimator_reset() {
        let mut est = GroundingEstimator::new();
        for _ in 0..32 {
            est.estimate(0.1, None);
        }
        est.reset();
        assert_eq!(est.estimate(0.5, None), GROUNDING_SENSORIMOTOR);
    }
}
