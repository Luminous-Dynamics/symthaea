// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared motor safety level for consciousness-gated motor authority.
//!
//! All robotics embodiment bridges (flight, vehicle, manipulator, helicopter, AUV)
//! use the same 4-tier safety model derived from Phi (consciousness integration).
//! This module provides the canonical definition so each crate can import rather
//! than duplicate.

/// Safety level for consciousness-gated motor authority.
///
/// Derived from Phi via [`MotorSafetyLevel::from_phi`]. Higher Phi grants more
/// motor authority; low Phi progressively restricts actuator output down to full
/// shutdown at Red.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MotorSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

impl MotorSafetyLevel {
    /// Map a Phi (consciousness integration) value to a safety tier.
    ///
    /// | Phi range | Level |
    /// |-----------|-------|
    /// | > 0.6     | Green |
    /// | 0.3 - 0.6 | Yellow |
    /// | 0.1 - 0.3 | Orange |
    /// | < 0.1     | Red |
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

    /// Multiplicative gain applied to motor commands at this safety level.
    ///
    /// Green = full authority (1.0), Red = motors off (0.0).
    pub fn motor_gain(&self) -> f32 {
        match self {
            Self::Green => 1.0,
            Self::Yellow => 0.6,
            Self::Orange => 0.3,
            Self::Red => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_phi_thresholds() {
        assert_eq!(MotorSafetyLevel::from_phi(0.8), MotorSafetyLevel::Green);
        assert_eq!(MotorSafetyLevel::from_phi(0.6), MotorSafetyLevel::Yellow); // boundary: not >0.6
        assert_eq!(MotorSafetyLevel::from_phi(0.5), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.3), MotorSafetyLevel::Orange); // boundary: not >0.3
        assert_eq!(MotorSafetyLevel::from_phi(0.2), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.1), MotorSafetyLevel::Red); // boundary: not >0.1
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
