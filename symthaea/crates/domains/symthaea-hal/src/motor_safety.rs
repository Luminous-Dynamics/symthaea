// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Re-export canonical motor safety types from symthaea-core.
//!
//! All robotics embodiment bridges (flight, vehicle, manipulator, helicopter, AUV)
//! import `MotorSafetyLevel` through this module or directly from `symthaea_core::embodiment`.
//! Both paths resolve to the same Rust type — no cross-crate E0308 mismatches.

pub use symthaea_core::embodiment::MotorSafetyLevel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_phi_thresholds() {
        assert_eq!(MotorSafetyLevel::from_phi(0.8), MotorSafetyLevel::Green);
        assert_eq!(MotorSafetyLevel::from_phi(0.6), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.5), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.3), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.2), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.1), MotorSafetyLevel::Red);
        assert_eq!(MotorSafetyLevel::from_phi(0.05), MotorSafetyLevel::Red);
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
