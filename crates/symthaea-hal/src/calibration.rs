// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Joint calibration: torque → angle → pulse µs mapping.
//!
//! Each joint has a [`JointCalibration`] that defines the mapping from
//! the cognitive loop's normalized torque (−1..+1) through joint angle
//! (degrees) to servo pulse width (µs). A [`CalibrationProfile`] holds
//! all 21 joints and supports JSON persistence.

use serde::{Deserialize, Serialize};
use symthaea_humanoid::types::{JOINT_NAMES, NUM_ACTUATORS};

use crate::error::{HalError, HalResult};

// ============================================================================
// PER-JOINT CALIBRATION
// ============================================================================

/// Calibration parameters for a single joint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointCalibration {
    /// Joint name (matches `JOINT_NAMES`).
    pub name: String,
    /// Minimum servo pulse width in µs (corresponds to `angle_min`).
    pub pulse_min_us: u16,
    /// Maximum servo pulse width in µs (corresponds to `angle_max`).
    pub pulse_max_us: u16,
    /// Minimum mechanical angle in degrees.
    pub angle_min_deg: f32,
    /// Maximum mechanical angle in degrees.
    pub angle_max_deg: f32,
    /// Whether the servo is reversed (torque +1 → angle_min instead of angle_max).
    pub reversed: bool,
}

impl JointCalibration {
    /// Create a calibration with standard hobby servo defaults.
    ///
    /// - Pulse: 500–2500 µs
    /// - Angle: −90° to +90°
    /// - Not reversed
    pub fn default_for(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pulse_min_us: 500,
            pulse_max_us: 2500,
            angle_min_deg: -90.0,
            angle_max_deg: 90.0,
            reversed: false,
        }
    }

    /// Map a normalized torque (−1..+1) to a target angle in degrees.
    ///
    /// Linearly maps: −1 → `angle_min`, +1 → `angle_max` (or reversed).
    pub fn torque_to_angle(&self, torque: f32) -> f32 {
        let t = torque.clamp(-1.0, 1.0);
        let t = if self.reversed { -t } else { t };
        let mid = (self.angle_min_deg + self.angle_max_deg) / 2.0;
        let half_range = (self.angle_max_deg - self.angle_min_deg) / 2.0;
        mid + t * half_range
    }

    /// Map an angle in degrees to a pulse width in µs.
    ///
    /// Linearly interpolates between `pulse_min` (at `angle_min`) and
    /// `pulse_max` (at `angle_max`), clamped to the pulse range.
    pub fn angle_to_pulse_us(&self, angle_deg: f32) -> u16 {
        let range = self.angle_max_deg - self.angle_min_deg;
        if range.abs() < f32::EPSILON {
            return (self.pulse_min_us + self.pulse_max_us) / 2;
        }
        let fraction = (angle_deg - self.angle_min_deg) / range;
        let fraction = fraction.clamp(0.0, 1.0);
        let pulse = self.pulse_min_us as f32
            + fraction * (self.pulse_max_us as f32 - self.pulse_min_us as f32);
        (pulse.round() as u16).clamp(self.pulse_min_us, self.pulse_max_us)
    }

    /// Convenience: torque → angle → pulse in one step.
    pub fn torque_to_pulse_us(&self, torque: f32) -> u16 {
        let angle = self.torque_to_angle(torque);
        self.angle_to_pulse_us(angle)
    }

    /// The center (neutral) pulse width in µs.
    pub fn center_pulse_us(&self) -> u16 {
        (self.pulse_min_us + self.pulse_max_us) / 2
    }
}

// ============================================================================
// CALIBRATION PROFILE (21 JOINTS)
// ============================================================================

/// Calibration profile for all 21 humanoid joints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationProfile {
    /// Per-joint calibrations (indexed by joint number, same order as `JOINT_NAMES`).
    pub joints: Vec<JointCalibration>,
}

impl CalibrationProfile {
    /// Create a default profile with standard servo parameters for all 21 joints.
    pub fn default_21() -> Self {
        let joints = JOINT_NAMES
            .iter()
            .map(|name| JointCalibration::default_for(name))
            .collect();
        Self { joints }
    }

    /// Load a profile from a JSON string.
    pub fn from_json(json: &str) -> HalResult<Self> {
        let profile: Self =
            serde_json::from_str(json).map_err(|e| HalError::Calibration(e.to_string()))?;
        if profile.joints.len() != NUM_ACTUATORS {
            return Err(HalError::Calibration(format!(
                "expected {} joints, got {}",
                NUM_ACTUATORS,
                profile.joints.len()
            )));
        }
        Ok(profile)
    }

    /// Serialize the profile to a JSON string.
    pub fn to_json(&self) -> HalResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| HalError::Calibration(e.to_string()))
    }

    /// Load a profile from a JSON file.
    pub fn load(path: &std::path::Path) -> HalResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| HalError::Calibration(format!("read {}: {}", path.display(), e)))?;
        Self::from_json(&json)
    }

    /// Save the profile to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> HalResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| HalError::Calibration(format!("write {}: {}", path.display(), e)))
    }

    /// Get calibration for a joint by index.
    pub fn joint(&self, index: usize) -> Option<&JointCalibration> {
        self.joints.get(index)
    }

    /// Get mutable calibration for a joint by index.
    pub fn joint_mut(&mut self, index: usize) -> Option<&mut JointCalibration> {
        self.joints.get_mut(index)
    }

    /// Convert torques to pulse widths (µs).
    pub fn torques_to_pulses(&self, torques: &[f32]) -> [u16; NUM_ACTUATORS] {
        let mut pulses = [1500u16; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS.min(torques.len()) {
            pulses[i] = self.joints[i].torque_to_pulse_us(torques[i]);
        }
        pulses
    }
}

impl Default for CalibrationProfile {
    fn default() -> Self {
        Self::default_21()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_calibration_center() {
        let cal = JointCalibration::default_for("test");
        // Torque 0 → angle 0° → center pulse
        let pulse = cal.torque_to_pulse_us(0.0);
        assert_eq!(pulse, 1500);
    }

    #[test]
    fn test_joint_calibration_extremes() {
        let cal = JointCalibration::default_for("test");
        // Torque -1 → angle_min (-90°) → pulse_min (500)
        assert_eq!(cal.torque_to_pulse_us(-1.0), 500);
        // Torque +1 → angle_max (+90°) → pulse_max (2500)
        assert_eq!(cal.torque_to_pulse_us(1.0), 2500);
    }

    #[test]
    fn test_joint_calibration_reversed() {
        let mut cal = JointCalibration::default_for("test");
        cal.reversed = true;
        // Torque +1 reversed → angle_min → pulse_min
        assert_eq!(cal.torque_to_pulse_us(1.0), 500);
        // Torque -1 reversed → angle_max → pulse_max
        assert_eq!(cal.torque_to_pulse_us(-1.0), 2500);
    }

    #[test]
    fn test_joint_calibration_clamps() {
        let cal = JointCalibration::default_for("test");
        // Out-of-range torques clamp to ±1
        assert_eq!(cal.torque_to_pulse_us(5.0), 2500);
        assert_eq!(cal.torque_to_pulse_us(-5.0), 500);
    }

    #[test]
    fn test_torque_to_angle() {
        let cal = JointCalibration::default_for("test");
        let angle = cal.torque_to_angle(0.5);
        // 0.5 → 0 + 0.5 * 90 = 45°
        assert!((angle - 45.0).abs() < 0.01);
    }

    #[test]
    fn test_angle_to_pulse() {
        let cal = JointCalibration::default_for("test");
        // -90° → 500, 0° → 1500, +90° → 2500
        assert_eq!(cal.angle_to_pulse_us(-90.0), 500);
        assert_eq!(cal.angle_to_pulse_us(0.0), 1500);
        assert_eq!(cal.angle_to_pulse_us(90.0), 2500);
    }

    #[test]
    fn test_profile_default_21() {
        let profile = CalibrationProfile::default_21();
        assert_eq!(profile.joints.len(), NUM_ACTUATORS);
        assert_eq!(profile.joints[0].name, "abdomen_y");
        assert_eq!(profile.joints[20].name, "left_elbow");
    }

    #[test]
    fn test_profile_json_roundtrip() {
        let profile = CalibrationProfile::default_21();
        let json = profile.to_json().unwrap();
        let restored = CalibrationProfile::from_json(&json).unwrap();
        assert_eq!(restored.joints.len(), NUM_ACTUATORS);
        assert_eq!(restored.joints[0].name, profile.joints[0].name);
    }

    #[test]
    fn test_profile_wrong_count_rejected() {
        let mut profile = CalibrationProfile::default_21();
        profile.joints.pop(); // now 20 joints
        let json = serde_json::to_string(&profile).unwrap();
        let result = CalibrationProfile::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_torques_to_pulses() {
        let profile = CalibrationProfile::default_21();
        let torques = [0.0f32; NUM_ACTUATORS];
        let pulses = profile.torques_to_pulses(&torques);
        for &p in &pulses {
            assert_eq!(p, 1500);
        }
    }

    #[test]
    fn test_center_pulse() {
        let cal = JointCalibration::default_for("test");
        assert_eq!(cal.center_pulse_us(), 1500);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn torque_to_pulse_always_in_range(torque in -10.0f32..10.0) {
            let cal = JointCalibration::default_for("test");
            let pulse = cal.torque_to_pulse_us(torque);
            prop_assert!(pulse >= cal.pulse_min_us, "pulse {} < min {}", pulse, cal.pulse_min_us);
            prop_assert!(pulse <= cal.pulse_max_us, "pulse {} > max {}", pulse, cal.pulse_max_us);
        }

        #[test]
        fn torque_to_pulse_monotonic(a in -1.0f32..1.0, b in -1.0f32..1.0) {
            let cal = JointCalibration::default_for("test");
            let pa = cal.torque_to_pulse_us(a);
            let pb = cal.torque_to_pulse_us(b);
            if a <= b {
                prop_assert!(pa <= pb, "monotonicity: t={} → p={}, t={} → p={}", a, pa, b, pb);
            } else {
                prop_assert!(pa >= pb, "monotonicity: t={} → p={}, t={} → p={}", a, pa, b, pb);
            }
        }

        #[test]
        fn extreme_torques_clamp_to_endpoints(torque in prop::num::f32::ANY) {
            if torque.is_finite() {
                let cal = JointCalibration::default_for("test");
                let pulse = cal.torque_to_pulse_us(torque);
                prop_assert!(pulse >= cal.pulse_min_us);
                prop_assert!(pulse <= cal.pulse_max_us);
            }
        }
    }
}
