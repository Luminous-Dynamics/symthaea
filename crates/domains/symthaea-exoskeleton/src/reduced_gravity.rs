// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-gravity environment model for full-body exosuit research.
//!
//! This module removes *new* space-exosuit calculations from implicit Earth-g
//! assumptions. It is intentionally a simulation/reference layer; values here
//! are not hardware qualification data.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

pub const EARTH_GRAVITY_M_S2: f64 = 9.80665;
pub const MOON_GRAVITY_M_S2: f64 = 1.62;
pub const MARS_GRAVITY_M_S2: f64 = 3.71;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GravityEnvironment {
    pub gravity_m_s2: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl GravityEnvironment {
    pub const fn earth_reference() -> Self {
        Self {
            gravity_m_s2: EARTH_GRAVITY_M_S2,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub const fn lunar_reference() -> Self {
        Self {
            gravity_m_s2: MOON_GRAVITY_M_S2,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub const fn mars_reference() -> Self {
        Self {
            gravity_m_s2: MARS_GRAVITY_M_S2,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn custom(gravity_m_s2: f64, evidence: ExosuitEvidenceLevel) -> Self {
        Self {
            gravity_m_s2,
            evidence,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.gravity_m_s2.is_finite() && self.gravity_m_s2 >= 0.0
    }

    pub fn gravity_fraction_of_earth(&self) -> f64 {
        self.gravity_m_s2 / EARTH_GRAVITY_M_S2
    }

    pub fn body_weight_n(&self, mass_kg: f64) -> Option<f64> {
        if !self.is_valid() || !mass_kg.is_finite() || mass_kg < 0.0 {
            return None;
        }
        Some(mass_kg * self.gravity_m_s2)
    }

    /// Additional downward-equivalent load required to reach a target gravity
    /// for a given supported mass. This is a *loading target*, not a claim that
    /// joint resistance perfectly reproduces gravity physiology.
    pub fn additional_load_to_target_n(
        &self,
        mass_kg: f64,
        target_gravity_m_s2: f64,
    ) -> Option<f64> {
        if !self.is_valid()
            || !mass_kg.is_finite()
            || mass_kg < 0.0
            || !target_gravity_m_s2.is_finite()
            || target_gravity_m_s2 < 0.0
        {
            return None;
        }
        Some((mass_kg * (target_gravity_m_s2 - self.gravity_m_s2)).max(0.0))
    }
}

/// Reference gravity suite used by SX-002 regression tests and later
/// full-frame integration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReducedGravitySuite {
    pub earth: GravityEnvironment,
    pub moon: GravityEnvironment,
    pub mars: GravityEnvironment,
}

impl Default for ReducedGravitySuite {
    fn default() -> Self {
        Self {
            earth: GravityEnvironment::earth_reference(),
            moon: GravityEnvironment::lunar_reference(),
            mars: GravityEnvironment::mars_reference(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_gravity_order_is_earth_mars_moon() {
        let s = ReducedGravitySuite::default();
        assert!(s.earth.gravity_m_s2 > s.mars.gravity_m_s2);
        assert!(s.mars.gravity_m_s2 > s.moon.gravity_m_s2);
    }

    #[test]
    fn same_mass_has_expected_reduced_weight_order() {
        let mass = 80.0;
        let s = ReducedGravitySuite::default();
        let earth = s.earth.body_weight_n(mass).unwrap();
        let mars = s.mars.body_weight_n(mass).unwrap();
        let moon = s.moon.body_weight_n(mass).unwrap();
        assert!(earth > mars && mars > moon);
        assert!((moon / earth - MOON_GRAVITY_M_S2 / EARTH_GRAVITY_M_S2).abs() < 1e-12);
    }

    #[test]
    fn lunar_workout_can_request_missing_earth_loading() {
        let moon = GravityEnvironment::lunar_reference();
        let missing = moon
            .additional_load_to_target_n(80.0, EARTH_GRAVITY_M_S2)
            .unwrap();
        let expected = 80.0 * (EARTH_GRAVITY_M_S2 - MOON_GRAVITY_M_S2);
        assert!((missing - expected).abs() < 1e-9);
    }

    #[test]
    fn malformed_gravity_fails_closed() {
        let invalid = GravityEnvironment::custom(f64::NAN, ExosuitEvidenceLevel::Simulation);
        assert!(!invalid.is_valid());
        assert_eq!(invalid.body_weight_n(80.0), None);
    }
}
