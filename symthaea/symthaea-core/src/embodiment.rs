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
    pub fn from_phi(phi: f64) -> Self {
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

    /// Motor gain multiplier for this safety level.
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
    /// Virtual caregiver agent — no physical embodiment.
    CareProvider,
    /// Browser agent — web pages as sensory environment via CDP.
    Browser,
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

// ── Embodiment Bridge Trait ────────────────────────────────────────────────

/// Platform-agnostic embodiment bridge trait.
///
/// Any robotics platform that wants to close the proprioceptive loop
/// with the cognitive loop must implement this trait.
pub trait EmbodimentBridge: Send + Sync {
    /// Translate thought → motor commands, step physics, return result.
    /// `phi` is the current consciousness level for safety gating.
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult;

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
    fn test_grounding_labels() {
        assert_eq!(grounding_label(GROUNDING_SENSORIMOTOR), "Sensorimotor");
        assert_eq!(grounding_label(GROUNDING_TEMPORAL), "Temporal");
        assert_eq!(grounding_label(GROUNDING_SOCIAL), "Social");
        assert_eq!(grounding_label(42), "Unknown");
    }
}
