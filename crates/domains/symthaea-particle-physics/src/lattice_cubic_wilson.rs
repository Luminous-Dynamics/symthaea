// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cubic-orbit averaging for already-qualified off-axis mixed Wilson loops.
//!
//! One representative displacement is expanded to every unique signed spatial
//! permutation. Each orientation is measured by LQCD-020M and then averaged
//! with equal weight. This module does not redefine spatial transport.

use std::collections::BTreeSet;

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_off_axis_wilson::{
    OffAxisWilsonError, average_off_axis_mixed_wilson_loop,
};

pub const CUBIC_SIGNED_PERMUTATION_ORBIT_ID: &str = "cubic_signed_permutation_orbit_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicOrbitWilsonMeasurement {
    pub value: f64,
    pub orbit_size: usize,
    pub paths_per_orientation: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CubicOrbitWilsonError {
    OffAxis(OffAxisWilsonError),
    ZeroDisplacement,
    ComponentNegationOverflow(i32),
    InvalidOrientationBudget(usize),
    OrbitBudgetExceeded {
        required: usize,
        budget: usize,
    },
    PathCountMismatch {
        expected: usize,
        actual: usize,
        orientation: [i32; 3],
    },
    NonFiniteAverage,
}

impl From<OffAxisWilsonError> for CubicOrbitWilsonError {
    fn from(value: OffAxisWilsonError) -> Self {
        Self::OffAxis(value)
    }
}

fn signed(value: i32, negative: bool) -> Result<i32, CubicOrbitWilsonError> {
    if !negative {
        return Ok(value);
    }
    value
        .checked_neg()
        .ok_or(CubicOrbitWilsonError::ComponentNegationOverflow(value))
}

/// Return the deterministic lexicographically sorted unique signed-permutation orbit.
pub fn cubic_signed_permutation_orbit(
    representative: [i32; 3],
) -> Result<Vec<[i32; 3]>, CubicOrbitWilsonError> {
    if representative == [0, 0, 0] {
        return Err(CubicOrbitWilsonError::ZeroDisplacement);
    }

    let [a, b, c] = representative;
    let permutations = [
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ];
    let mut orbit = BTreeSet::new();
    for permutation in permutations {
        for mask in 0u8..8 {
            orbit.insert([
                signed(permutation[0], mask & 1 != 0)?,
                signed(permutation[1], mask & 2 != 0)?,
                signed(permutation[2], mask & 4 != 0)?,
            ]);
        }
    }
    Ok(orbit.into_iter().collect())
}

/// Average one off-axis mixed Wilson observable over the full cubic orbit.
///
/// `max_orientations` bounds orbit expansion. `max_paths_per_orientation` is
/// passed directly to the qualified LQCD-020M shortest-path measurement.
pub fn average_cubic_orbit_mixed_wilson_loop(
    original: &WilsonGaugeField,
    spatial_operator: &WilsonGaugeField,
    representative: [i32; 3],
    temporal_extent: usize,
    max_orientations: usize,
    max_paths_per_orientation: usize,
) -> Result<CubicOrbitWilsonMeasurement, CubicOrbitWilsonError> {
    if max_orientations == 0 {
        return Err(CubicOrbitWilsonError::InvalidOrientationBudget(0));
    }
    let orbit = cubic_signed_permutation_orbit(representative)?;
    if orbit.len() > max_orientations {
        return Err(CubicOrbitWilsonError::OrbitBudgetExceeded {
            required: orbit.len(),
            budget: max_orientations,
        });
    }

    let mut sum = 0.0;
    let mut paths_per_orientation = None;
    for orientation in &orbit {
        let measured = average_off_axis_mixed_wilson_loop(
            original,
            spatial_operator,
            *orientation,
            temporal_extent,
            max_paths_per_orientation,
        )?;
        if let Some(expected) = paths_per_orientation {
            if measured.path_count != expected {
                return Err(CubicOrbitWilsonError::PathCountMismatch {
                    expected,
                    actual: measured.path_count,
                    orientation: *orientation,
                });
            }
        } else {
            paths_per_orientation = Some(measured.path_count);
        }
        sum += measured.value;
    }

    let value = sum / orbit.len() as f64;
    if !value.is_finite() {
        return Err(CubicOrbitWilsonError::NonFiniteAverage);
    }
    Ok(CubicOrbitWilsonMeasurement {
        value,
        orbit_size: orbit.len(),
        paths_per_orientation: paths_per_orientation.unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_spatial_smearing::{SpatialApeConfig, spatial_ape_smear};

    #[test]
    fn reproduces_independent_lqcd_020t_orbit_geometry() {
        let fixtures = [
            ([1, 0, 0], 6, 1),
            ([2, 0, 0], 6, 1),
            ([3, 0, 0], 6, 1),
            ([1, 1, 0], 12, 2),
            ([1, 1, 1], 8, 6),
            ([2, 1, 0], 24, 3),
            ([2, 2, 0], 12, 6),
            ([3, 3, 0], 12, 20),
            ([2, 2, 2], 8, 90),
            ([3, 3, 3], 8, 1680),
            ([4, 2, 0], 24, 15),
            ([6, 3, 0], 24, 84),
        ];
        for (representative, orbit_size, path_count) in fixtures {
            let orbit = cubic_signed_permutation_orbit(representative).unwrap();
            assert_eq!(orbit.len(), orbit_size);
            let magnitude = representative.map(|component| component.unsigned_abs() as usize);
            let n = magnitude.iter().sum::<usize>();
            let factorial = |value: usize| (1..=value).product::<usize>().max(1);
            let count = factorial(n)
                / (factorial(magnitude[0]) * factorial(magnitude[1]) * factorial(magnitude[2]));
            assert_eq!(count, path_count);
        }
    }

    #[test]
    fn aliases_canonicalize_identically() {
        assert_eq!(
            cubic_signed_permutation_orbit([2, 1, 0]).unwrap(),
            cubic_signed_permutation_orbit([-1, 0, 2]).unwrap()
        );
        assert_eq!(
            cubic_signed_permutation_orbit([1, 1, 1]).unwrap(),
            cubic_signed_permutation_orbit([-1, 1, -1]).unwrap()
        );
    }

    #[test]
    fn identity_field_orbit_average_is_one() {
        let original = WilsonGaugeField::identity([4, 4, 4, 4]).unwrap();
        let operator = spatial_ape_smear(&original, &SpatialApeConfig::default()).unwrap();
        for representative in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]] {
            let measured = average_cubic_orbit_mixed_wilson_loop(
                &original,
                &operator,
                representative,
                1,
                48,
                32,
            )
            .unwrap();
            assert!((measured.value - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn work_budgets_fail_closed() {
        let original = WilsonGaugeField::identity([4, 4, 4, 4]).unwrap();
        let operator = spatial_ape_smear(&original, &SpatialApeConfig::default()).unwrap();
        assert!(matches!(
            average_cubic_orbit_mixed_wilson_loop(&original, &operator, [2, 1, 0], 1, 23, 8),
            Err(CubicOrbitWilsonError::OrbitBudgetExceeded { required: 24, budget: 23 })
        ));
        assert!(matches!(
            average_cubic_orbit_mixed_wilson_loop(&original, &operator, [1, 1, 1], 1, 8, 5),
            Err(CubicOrbitWilsonError::OffAxis(
                OffAxisWilsonError::PathBudgetExceeded(5)
            ))
        ));
    }

    #[test]
    fn invalid_orbit_inputs_fail_closed() {
        assert_eq!(
            cubic_signed_permutation_orbit([0, 0, 0]),
            Err(CubicOrbitWilsonError::ZeroDisplacement)
        );
        assert!(matches!(
            cubic_signed_permutation_orbit([i32::MIN, 0, 0]),
            Err(CubicOrbitWilsonError::ComponentNegationOverflow(i32::MIN))
        ));
    }
}
