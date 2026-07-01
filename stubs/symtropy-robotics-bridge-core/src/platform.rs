// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Platform types for Symthaea robotics platforms.

/// Robotic platform type -- maps to symthaea's EmbodimentPlatform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PlatformType {
    /// Quadrotor drone (4 actuators).
    Quadrotor,
    /// Autonomous car (3 actuators: steering, throttle, brake).
    Vehicle,
    /// Bipedal humanoid (64 DOF FullSpine core).
    Humanoid,
    /// Autonomous underwater vehicle (8 thrusters).
    Auv,
    /// SAR helicopter (6 DOF with rotor dynamics).
    Helicopter,
    /// Industrial robot arm (7+1 DOF).
    Manipulator,
    /// Surgical robot (8 actuators: cart-arm + precision tool).
    Surgical,
    /// Orbital platform (7 thrusters for attitude + translation).
    Orbital,
    /// Quadruped (4 legs x 3 joints = 12 DOF).
    Quadruped,
    /// Full-frame exoskeleton (6 powered joints, human-worn).
    Exoskeleton,
}

impl Default for PlatformType {
    fn default() -> Self {
        Self::Humanoid
    }
}

impl PlatformType {
    /// Number of actuators for this platform.
    pub fn num_actuators(&self) -> usize {
        match self {
            Self::Quadrotor => 4,
            Self::Vehicle => 3,
            Self::Humanoid => 64,
            Self::Auv => 8,
            Self::Helicopter => 6,
            Self::Manipulator => 8,
            Self::Surgical => 8,
            Self::Orbital => 7,
            Self::Quadruped => 12,
            Self::Exoskeleton => 6,
        }
    }

    /// Default mass in kg.
    pub fn default_mass(&self) -> f64 {
        match self {
            Self::Quadrotor => 0.027,
            Self::Vehicle => 1500.0,
            Self::Humanoid => 95.0,
            Self::Auv => 50.0,
            Self::Helicopter => 2500.0,
            Self::Manipulator => 20.0,
            Self::Surgical => 50.0,
            Self::Orbital => 500.0,
            Self::Quadruped => 12.0,
            Self::Exoskeleton => 25.0,
        }
    }

    /// Default collider radius (bounding sphere) in meters.
    pub fn default_radius(&self) -> f64 {
        match self {
            Self::Quadrotor => 0.1,
            Self::Vehicle => 2.5,
            Self::Humanoid => 0.85,
            Self::Auv => 1.0,
            Self::Helicopter => 5.0,
            Self::Manipulator => 1.0,
            Self::Surgical => 1.5,
            Self::Orbital => 5.0,
            Self::Quadruped => 0.5,
            Self::Exoskeleton => 1.0,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Quadrotor => "Quadrotor",
            Self::Vehicle => "Vehicle",
            Self::Humanoid => "Humanoid",
            Self::Auv => "AUV",
            Self::Helicopter => "Helicopter",
            Self::Manipulator => "Manipulator",
            Self::Surgical => "Surgical",
            Self::Orbital => "Orbital",
            Self::Quadruped => "Quadruped",
            Self::Exoskeleton => "Exoskeleton",
        }
    }

    /// All platform variants (for iteration).
    pub const ALL: [PlatformType; 10] = [
        Self::Quadrotor,
        Self::Vehicle,
        Self::Humanoid,
        Self::Auv,
        Self::Helicopter,
        Self::Manipulator,
        Self::Surgical,
        Self::Orbital,
        Self::Quadruped,
        Self::Exoskeleton,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_platforms_have_actuators() {
        for p in PlatformType::ALL {
            assert!(p.num_actuators() > 0);
            assert!(p.default_mass() > 0.0);
            assert!(p.default_radius() > 0.0);
            assert!(!p.name().is_empty());
        }
    }
}
