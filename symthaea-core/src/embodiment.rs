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
}
