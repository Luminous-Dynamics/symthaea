// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-gated workspace safety for industrial manipulator.
//!
//! | Level | Phi | Force Cap | Speed Cap | Workspace |
//! |-------|-----|-----------|-----------|-----------|
//! | Green | >0.6 | Full (87N) | Full | 100% |
//! | Yellow | 0.3–0.6 | 20N | 50% | Inner 80% |
//! | Orange | 0.1–0.3 | 0N | 20% | Retreat to home |
//! | Red | <0.1 | Power cut | 0 | Gravity-compensated hold |

use symthaea_core::embodiment::MotorSafetyLevel;

/// Runtime limits derived from the canonical embodiment safety tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManipulatorSafetyLimits {
    pub max_force_n: f64,
    pub normalized_torque_cap: f32,
    pub max_joint_speed_rad_s: f64,
    pub workspace_fraction: f64,
    pub retreat_required: bool,
}

impl ManipulatorSafetyLimits {
    pub fn for_level(level: MotorSafetyLevel) -> Self {
        match level {
            MotorSafetyLevel::Green => Self {
                max_force_n: 87.0,
                normalized_torque_cap: 1.0,
                max_joint_speed_rad_s: 2.5,
                workspace_fraction: 1.0,
                retreat_required: false,
            },
            MotorSafetyLevel::Yellow => Self {
                max_force_n: 20.0,
                normalized_torque_cap: 0.5,
                max_joint_speed_rad_s: 1.0,
                workspace_fraction: 0.8,
                retreat_required: false,
            },
            MotorSafetyLevel::Orange => Self {
                max_force_n: 0.0,
                normalized_torque_cap: 0.2,
                max_joint_speed_rad_s: 0.4,
                workspace_fraction: 1.0,
                retreat_required: true,
            },
            MotorSafetyLevel::Red => Self {
                max_force_n: 0.0,
                normalized_torque_cap: 0.0,
                max_joint_speed_rad_s: 0.0,
                workspace_fraction: 1.0,
                retreat_required: false,
            },
        }
    }
}

/// Workspace boundary: sphere centered on base with given radius.
#[derive(Debug, Clone)]
pub struct WorkspaceBoundary {
    /// Maximum reach radius (meters).
    pub max_radius: f64,
    /// Minimum height (meters, prevents floor collision).
    pub min_height: f64,
    /// Human proximity zones: [(center_x, center_y, center_z, radius)].
    pub human_zones: Vec<[f64; 4]>,
}

impl Default for WorkspaceBoundary {
    fn default() -> Self {
        Self {
            max_radius: 0.855, // Panda max reach
            min_height: 0.05,  // 5cm above table
            human_zones: Vec::new(),
        }
    }
}

impl WorkspaceBoundary {
    /// Check if a position is within the safe workspace.
    pub fn is_within(&self, pos: &[f64; 3], safety: MotorSafetyLevel) -> bool {
        let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let max = self.max_radius * ManipulatorSafetyLimits::for_level(safety).workspace_fraction;

        if radius > max {
            return false;
        }
        if pos[2] < self.min_height {
            return false;
        }

        // Check human proximity zones
        for zone in &self.human_zones {
            let dx = pos[0] - zone[0];
            let dy = pos[1] - zone[1];
            let dz = pos[2] - zone[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < zone[3] {
                return false;
            }
        }

        true
    }

    /// Distance to nearest boundary violation (positive = safe, negative = violation).
    pub fn clearance(&self, pos: &[f64; 3], safety: MotorSafetyLevel) -> f64 {
        let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let max = self.max_radius * ManipulatorSafetyLimits::for_level(safety).workspace_fraction;

        let radius_clearance = max - radius;
        let height_clearance = pos[2] - self.min_height;

        let mut min_human_clearance = f64::MAX;
        for zone in &self.human_zones {
            let dx = pos[0] - zone[0];
            let dy = pos[1] - zone[1];
            let dz = pos[2] - zone[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            min_human_clearance = min_human_clearance.min(dist - zone[3]);
        }

        radius_clearance
            .min(height_clearance)
            .min(min_human_clearance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_limits() {
        assert_eq!(
            ManipulatorSafetyLimits::for_level(MotorSafetyLevel::Green).max_force_n,
            87.0
        );
        assert_eq!(
            ManipulatorSafetyLimits::for_level(MotorSafetyLevel::Red).max_force_n,
            0.0
        );
        assert!(
            ManipulatorSafetyLimits::for_level(MotorSafetyLevel::Yellow).max_force_n
                < ManipulatorSafetyLimits::for_level(MotorSafetyLevel::Green).max_force_n
        );
    }

    #[test]
    fn test_workspace_boundary() {
        let boundary = WorkspaceBoundary::default();
        let inside = [0.3, 0.0, 0.3];
        let outside = [1.0, 1.0, 0.3]; // Beyond reach
        assert!(boundary.is_within(&inside, MotorSafetyLevel::Green));
        assert!(!boundary.is_within(&outside, MotorSafetyLevel::Green));
    }

    #[test]
    fn test_workspace_shrinks_with_safety() {
        let boundary = WorkspaceBoundary::default();
        let edge = [0.7, 0.0, 0.3]; // Near max reach
        assert!(boundary.is_within(&edge, MotorSafetyLevel::Green));
        assert!(!boundary.is_within(&edge, MotorSafetyLevel::Yellow)); // 80% workspace
    }

    #[test]
    fn test_human_proximity_zone() {
        let boundary = WorkspaceBoundary {
            human_zones: vec![[0.3, 0.0, 0.3, 0.2]], // Human at (0.3,0,0.3) radius 0.2m
            ..Default::default()
        };
        let near_human = [0.35, 0.0, 0.3];
        let far_from_human = [0.0, 0.5, 0.3];
        assert!(!boundary.is_within(&near_human, MotorSafetyLevel::Green));
        assert!(boundary.is_within(&far_from_human, MotorSafetyLevel::Green));
    }

    #[test]
    fn test_clearance() {
        let boundary = WorkspaceBoundary::default();
        let center = [0.0, 0.0, 0.3];
        assert!(boundary.clearance(&center, MotorSafetyLevel::Green) > 0.0);
        let outside = [1.0, 0.0, 0.3];
        assert!(boundary.clearance(&outside, MotorSafetyLevel::Green) < 0.0);
    }

    #[test]
    fn test_min_height() {
        let boundary = WorkspaceBoundary::default();
        let below = [0.3, 0.0, 0.01]; // Below 5cm
        assert!(!boundary.is_within(&below, MotorSafetyLevel::Green));
    }
}
