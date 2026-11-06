// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure, validated Landauer erasure-heat arithmetic.
//!
//! The caller must supply the physical Boltzmann constant in J/K. The result
//! is a lower bound only for the explicitly declared number of logically
//! irreversible bit erasures at the declared positive bath temperature. It is
//! not a general energy cost for FLOPs or arbitrary information processing.

/// A finite SI energy value expressed in joules.
///
/// This unit-bearing wrapper keeps physical energy distinct from dimensionless
/// model costs, information quantities and raw operation counts.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PhysicalEnergyJoules(f64);

impl PhysicalEnergyJoules {
    /// Construct a finite energy value in joules.
    pub fn new(value_j: f64) -> Option<Self> {
        value_j.is_finite().then_some(Self(value_j))
    }

    /// Return the numeric value in joules.
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Compute the Landauer heat floor as a typed joule value for declared bit erasures.
///
/// Returns None for negative/non-finite bit counts, non-positive/non-finite
/// temperatures, an invalid Boltzmann constant, or arithmetic overflow.
pub fn erasure_heat_floor_j(
    bits_erased: f64,
    temperature_k: f64,
    boltzmann_j_per_k: f64,
) -> Option<PhysicalEnergyJoules> {
    if !bits_erased.is_finite()
        || bits_erased < 0.0
        || !temperature_k.is_finite()
        || temperature_k <= 0.0
        || !boltzmann_j_per_k.is_finite()
        || boltzmann_j_per_k <= 0.0
    {
        return None;
    }

    let heat_j = boltzmann_j_per_k * temperature_k * std::f64::consts::LN_2 * bits_erased;
    PhysicalEnergyJoules::new(heat_j)
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_BOLTZMANN: f64 = 1.380_649e-23;

    #[test]
    fn one_bit_reference_matches_300k_and_310k() {
        let at_300k = erasure_heat_floor_j(1.0, 300.0, K_BOLTZMANN).unwrap().get();
        let at_310k = erasure_heat_floor_j(1.0, 310.0, K_BOLTZMANN).unwrap().get();

        assert!((at_300k - 2.870_978_885_078_724e-21).abs() < 1e-35);
        assert!((at_310k - 2.966_678_181_248_014_7e-21).abs() < 1e-35);
    }

    #[test]
    fn floor_scales_linearly_with_declared_erased_bits_and_temperature() {
        let one_bit = erasure_heat_floor_j(1.0, 300.0, K_BOLTZMANN).unwrap().get();
        let two_bits = erasure_heat_floor_j(2.0, 300.0, K_BOLTZMANN).unwrap().get();
        let warmer = erasure_heat_floor_j(1.0, 310.0, K_BOLTZMANN).unwrap().get();

        assert!((two_bits - 2.0 * one_bit).abs() < 1e-35);
        assert!((warmer / one_bit - 310.0 / 300.0).abs() < 1e-14);
        assert_eq!(erasure_heat_floor_j(0.0, 300.0, K_BOLTZMANN).unwrap().get(), 0.0);
    }

    #[test]
    fn physical_energy_joules_rejects_non_finite_values() {
        assert!(PhysicalEnergyJoules::new(f64::NAN).is_none());
        assert!(PhysicalEnergyJoules::new(f64::INFINITY).is_none());
        assert!(PhysicalEnergyJoules::new(f64::NEG_INFINITY).is_none());
        assert_eq!(PhysicalEnergyJoules::new(1.25).unwrap().get(), 1.25);
    }

    #[test]
    fn invalid_inputs_and_overflow_are_rejected() {
        assert_eq!(erasure_heat_floor_j(-1.0, 300.0, K_BOLTZMANN), None);
        assert_eq!(erasure_heat_floor_j(f64::NAN, 300.0, K_BOLTZMANN), None);
        assert_eq!(erasure_heat_floor_j(1.0, 0.0, K_BOLTZMANN), None);
        assert_eq!(erasure_heat_floor_j(1.0, -300.0, K_BOLTZMANN), None);
        assert_eq!(erasure_heat_floor_j(1.0, f64::INFINITY, K_BOLTZMANN), None);
        assert_eq!(erasure_heat_floor_j(1.0, 300.0, 0.0), None);
        assert_eq!(erasure_heat_floor_j(1.0, 300.0, f64::NAN), None);
        assert_eq!(erasure_heat_floor_j(f64::MAX, f64::MAX, K_BOLTZMANN), None);
    }
}
