// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Platform types for Symthaea robotics platforms.

/// Robotic platform type — maps to symthaea's EmbodimentPlatform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlatformType {
    /// Quadrotor drone (4 actuators).
    Quadrotor,
    /// Autonomous car (3 actuators: steering, throttle, brake).
    Vehicle,
    /// Bipedal humanoid (21 DOF).
    Humanoid,
    /// Autonomous underwater vehicle (8 thrusters).
    Auv,
    /// SAR helicopter (6 DOF with rotor dynamics).
    Helicopter,
    /// Industrial robot arm (7+1 DOF).
    Manipulator,
}

impl PlatformType {
    /// Number of actuators for this platform.
    pub fn num_actuators(&self) -> usize {
        match self {
            Self::Quadrotor => 4,
            Self::Vehicle => 3,
            Self::Humanoid => 21,
            Self::Auv => 8,
            Self::Helicopter => 6,
            Self::Manipulator => 8,
        }
    }

    /// Default mass in kg.
    pub fn default_mass(&self) -> f64 {
        match self {
            Self::Quadrotor => 0.027,  // Crazyflie
            Self::Vehicle => 1500.0,   // Car
            Self::Humanoid => 70.0,    // Adult
            Self::Auv => 50.0,        // REMUS-class
            Self::Helicopter => 2500.0, // SAR helicopter
            Self::Manipulator => 20.0, // Robot arm
        }
    }

    /// Default collider radius (bounding sphere) in meters.
    pub fn default_radius(&self) -> f64 {
        match self {
            Self::Quadrotor => 0.1,
            Self::Vehicle => 2.5,
            Self::Humanoid => 0.5,
            Self::Auv => 1.0,
            Self::Helicopter => 5.0,
            Self::Manipulator => 1.0,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_platforms_have_actuators() {
        let platforms = [
            PlatformType::Quadrotor,
            PlatformType::Vehicle,
            PlatformType::Humanoid,
            PlatformType::Auv,
            PlatformType::Helicopter,
            PlatformType::Manipulator,
        ];
        for p in platforms {
            assert!(p.num_actuators() > 0, "{:?} has 0 actuators", p);
            assert!(p.default_mass() > 0.0);
            assert!(p.default_radius() > 0.0);
        }
    }
}
