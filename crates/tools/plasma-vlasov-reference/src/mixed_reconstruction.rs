//! Coefficient-space synthesis for the pinned mixed Hermite-Legendre basis.
//!
//! Inputs are real-space coefficient fields. Each coefficient array is
//! row-major (mode, x) and the output distribution arrays are row-major
//! (x, v). Fourier-space coefficient ordering and inverse FFT are deliberately
//! outside this module; an upstream file adapter must first convert Ck/Bk to
//! real-space coefficient fields using the source FFT convention.
//!
//! This module returns the Hermite bulk, Legendre residual, and their sum
//! separately. The Legendre residual may be signed. No positivity clipping or
//! conservation correction is performed.

use crate::mixed_basis::{
    MixedBasisError, asymmetric_hermite_basis_values, legendre_window_basis_values,
};

/// Which coefficient family contained invalid data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientFamily {
    Hermite,
    Legendre,
}

/// Invalid coefficient shape, value or reconstructed distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedReconstructionError {
    EmptySpatialGrid,
    EmptyVelocityGrid,
    NoRepresentedModes,
    CoefficientShape {
        family: CoefficientFamily,
        expected: usize,
        actual: usize,
    },
    NonFiniteCoefficient {
        family: CoefficientFamily,
        index: usize,
    },
    NonFiniteReconstruction {
        spatial_index: usize,
        velocity_index: usize,
    },
    ShapeOverflow,
    AllocationFailed {
        elements: usize,
    },
    Basis(MixedBasisError),
}

impl From<MixedBasisError> for MixedReconstructionError {
    fn from(value: MixedBasisError) -> Self {
        Self::Basis(value)
    }
}

/// Reconstructed mixed distribution and its two basis contributions.
///
/// Each array is row-major with shape (spatial_count, velocity_count).
#[derive(Debug, Clone, PartialEq)]
pub struct MixedDistribution {
    pub spatial_count: usize,
    pub velocity_count: usize,
    hermite_bulk: Vec<f64>,
    legendre_residual: Vec<f64>,
    total: Vec<f64>,
}

impl MixedDistribution {
    /// Hermite bulk value at (x_index, velocity_index).
    pub fn hermite_at(&self, x_index: usize, velocity_index: usize) -> Option<f64> {
        self.get(&self.hermite_bulk, x_index, velocity_index)
    }

    /// Legendre residual value at (x_index, velocity_index).
    pub fn legendre_at(&self, x_index: usize, velocity_index: usize) -> Option<f64> {
        self.get(&self.legendre_residual, x_index, velocity_index)
    }

    /// Total physical distribution at (x_index, velocity_index).
    pub fn total_at(&self, x_index: usize, velocity_index: usize) -> Option<f64> {
        self.get(&self.total, x_index, velocity_index)
    }

    /// Borrow the Hermite bulk in row-major (x, v) layout.
    pub fn hermite_bulk(&self) -> &[f64] {
        &self.hermite_bulk
    }

    /// Borrow the Legendre residual in row-major (x, v) layout.
    pub fn legendre_residual(&self) -> &[f64] {
        &self.legendre_residual
    }

    /// Borrow the total distribution in row-major (x, v) layout.
    pub fn total(&self) -> &[f64] {
        &self.total
    }

    fn get(&self, values: &[f64], x_index: usize, velocity_index: usize) -> Option<f64> {
        if x_index >= self.spatial_count || velocity_index >= self.velocity_count {
            return None;
        }
        values
            .get(x_index * self.velocity_count + velocity_index)
            .copied()
    }
}

/// Coefficient fields in row-major (mode, x) layout.
///
/// Hermite and Legendre coefficients are stored separately. Either family may
/// be omitted with a zero mode count and an empty slice. The nonempty family
/// still needs exactly mode_count * spatial_count values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedCoefficientFields<'a> {
    pub spatial_count: usize,
    pub hermite_modes: usize,
    pub hermite_coefficients: &'a [f64],
    pub legendre_modes: usize,
    pub legendre_coefficients: &'a [f64],
}

/// Physical velocity nodes and basis-coordinate configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedVelocityGrid<'a> {
    pub velocities: &'a [f64],
    pub shift: f64,
    pub scale: f64,
    pub velocity_min: f64,
    pub velocity_max: f64,
}

/// Reconstruct a real-space mixed Hermite-Legendre distribution.
///
/// Input coefficient arrays are row-major (mode, x); output arrays are
/// row-major (x, v). Either the Hermite or Legendre family may be omitted.
/// Coefficients must already be in physical space after any source-specific
/// inverse FFT. This function does not interpret FFT ordering or normalization,
/// perform positivity clipping, or apply a conservation correction.
pub fn reconstruct_mixed_distribution(
    coefficients: MixedCoefficientFields<'_>,
    grid: MixedVelocityGrid<'_>,
) -> Result<MixedDistribution, MixedReconstructionError> {
    let spatial_count = coefficients.spatial_count;
    let velocities = grid.velocities;
    let hermite_modes = coefficients.hermite_modes;
    let hermite_coefficients = coefficients.hermite_coefficients;
    let legendre_modes = coefficients.legendre_modes;
    let legendre_coefficients = coefficients.legendre_coefficients;
    let shift = grid.shift;
    let scale = grid.scale;
    let velocity_min = grid.velocity_min;
    let velocity_max = grid.velocity_max;
    if spatial_count == 0 {
        return Err(MixedReconstructionError::EmptySpatialGrid);
    }
    if velocities.is_empty() {
        return Err(MixedReconstructionError::EmptyVelocityGrid);
    }
    if hermite_modes == 0 && legendre_modes == 0 {
        return Err(MixedReconstructionError::NoRepresentedModes);
    }

    let hermite_elements = hermite_modes
        .checked_mul(spatial_count)
        .ok_or(MixedReconstructionError::ShapeOverflow)?;
    let legendre_elements = legendre_modes
        .checked_mul(spatial_count)
        .ok_or(MixedReconstructionError::ShapeOverflow)?;

    if hermite_coefficients.len() != hermite_elements {
        return Err(MixedReconstructionError::CoefficientShape {
            family: CoefficientFamily::Hermite,
            expected: hermite_elements,
            actual: hermite_coefficients.len(),
        });
    }
    if legendre_coefficients.len() != legendre_elements {
        return Err(MixedReconstructionError::CoefficientShape {
            family: CoefficientFamily::Legendre,
            expected: legendre_elements,
            actual: legendre_coefficients.len(),
        });
    }

    for (index, coefficient) in hermite_coefficients.iter().copied().enumerate() {
        if !coefficient.is_finite() {
            return Err(MixedReconstructionError::NonFiniteCoefficient {
                family: CoefficientFamily::Hermite,
                index,
            });
        }
    }
    for (index, coefficient) in legendre_coefficients.iter().copied().enumerate() {
        if !coefficient.is_finite() {
            return Err(MixedReconstructionError::NonFiniteCoefficient {
                family: CoefficientFamily::Legendre,
                index,
            });
        }
    }

    let hermite_basis = if hermite_modes > 0 {
        Some(asymmetric_hermite_basis_values(
            hermite_modes,
            velocities,
            shift,
            scale,
        )?)
    } else {
        None
    };
    let legendre_basis = if legendre_modes > 0 {
        Some(legendre_window_basis_values(
            legendre_modes,
            velocities,
            velocity_min,
            velocity_max,
        )?)
    } else {
        None
    };

    let velocity_count = velocities.len();
    let output_elements = spatial_count
        .checked_mul(velocity_count)
        .ok_or(MixedReconstructionError::ShapeOverflow)?;
    let mut hermite_bulk = allocate_output(output_elements)?;
    let mut legendre_residual = allocate_output(output_elements)?;
    let mut total = allocate_output(output_elements)?;

    for x_index in 0..spatial_count {
        for velocity_index in 0..velocity_count {
            let mut hermite_value = 0.0;
            if let Some(basis) = &hermite_basis {
                for mode in 0..hermite_modes {
                    hermite_value += hermite_coefficients[mode * spatial_count + x_index]
                        * basis
                            .get(mode, velocity_index)
                            .expect("Hermite basis dimensions match the loop bounds");
                }
            }

            let mut legendre_value = 0.0;
            if let Some(basis) = &legendre_basis {
                for mode in 0..legendre_modes {
                    legendre_value += legendre_coefficients[mode * spatial_count + x_index]
                        * basis
                            .get(mode, velocity_index)
                            .expect("Legendre basis dimensions match the loop bounds");
                }
            }

            let combined = hermite_value + legendre_value;
            if !hermite_value.is_finite() || !legendre_value.is_finite() || !combined.is_finite() {
                return Err(MixedReconstructionError::NonFiniteReconstruction {
                    spatial_index: x_index,
                    velocity_index,
                });
            }

            let output_index = x_index * velocity_count + velocity_index;
            hermite_bulk[output_index] = hermite_value;
            legendre_residual[output_index] = legendre_value;
            total[output_index] = combined;
        }
    }

    Ok(MixedDistribution {
        spatial_count,
        velocity_count,
        hermite_bulk,
        legendre_residual,
        total,
    })
}

fn allocate_output(elements: usize) -> Result<Vec<f64>, MixedReconstructionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| MixedReconstructionError::AllocationFailed { elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields<'a>(
        spatial_count: usize,
        hermite_modes: usize,
        hermite_coefficients: &'a [f64],
        legendre_modes: usize,
        legendre_coefficients: &'a [f64],
    ) -> MixedCoefficientFields<'a> {
        MixedCoefficientFields {
            spatial_count,
            hermite_modes,
            hermite_coefficients,
            legendre_modes,
            legendre_coefficients,
        }
    }

    fn grid<'a>(
        velocities: &'a [f64],
        shift: f64,
        scale: f64,
        velocity_min: f64,
        velocity_max: f64,
    ) -> MixedVelocityGrid<'a> {
        MixedVelocityGrid {
            velocities,
            shift,
            scale,
            velocity_min,
            velocity_max,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn reconstructs_hermite_bulk_and_legendre_residual_separately() {
        let velocities = [-2.0, 0.0, 1.0, 2.0];
        let distribution = reconstruct_mixed_distribution(
            fields(2, 1, &[2.0, 3.0], 1, &[0.5, -0.25]),
            grid(&velocities, 0.0, 1.0, -1.0, 1.0),
        )
        .unwrap();

        let psi_zero_at_zero = std::f64::consts::PI.sqrt().recip();
        let psi_zero_at_two = (-4.0_f64).exp() * psi_zero_at_zero;

        close(
            distribution.hermite_at(0, 1).unwrap(),
            2.0 * psi_zero_at_zero,
            1.0e-15,
        );
        close(distribution.legendre_at(0, 1).unwrap(), 0.5, 0.0);
        close(
            distribution.total_at(0, 1).unwrap(),
            2.0 * psi_zero_at_zero + 0.5,
            1.0e-15,
        );
        close(
            distribution.hermite_at(1, 2).unwrap(),
            3.0 * std::f64::consts::PI.sqrt().recip() * (-1.0_f64).exp(),
            1.0e-15,
        );
        close(distribution.legendre_at(1, 2).unwrap(), -0.25, 0.0);
        close(
            distribution.total_at(1, 2).unwrap(),
            3.0 * std::f64::consts::PI.sqrt().recip() * (-1.0_f64).exp() - 0.25,
            1.0e-15,
        );
        close(distribution.legendre_at(0, 0).unwrap(), 0.0, 0.0);
        close(
            distribution.total_at(0, 0).unwrap(),
            2.0 * psi_zero_at_two,
            1.0e-15,
        );
        close(distribution.legendre_at(1, 3).unwrap(), 0.0, 0.0);
    }

    #[test]
    fn pure_hermite_reconstruction_allows_zero_legendre_modes() {
        let velocities = [0.0, 0.5];
        let distribution = reconstruct_mixed_distribution(
            fields(1, 1, &[2.0], 0, &[]),
            grid(&velocities, 0.0, 1.0, -1.0, 1.0),
        )
        .unwrap();

        assert_eq!(distribution.legendre_residual(), &[0.0, 0.0]);
        assert_eq!(distribution.total(), distribution.hermite_bulk());
    }

    #[test]
    fn pure_legendre_reconstruction_allows_zero_hermite_modes() {
        let velocities = [-2.0, 0.0, 2.0];
        let distribution = reconstruct_mixed_distribution(
            fields(1, 0, &[], 1, &[0.75]),
            grid(&velocities, f64::NAN, 0.0, -1.0, 1.0),
        )
        .unwrap();

        assert_eq!(distribution.hermite_bulk(), &[0.0, 0.0, 0.0]);
        assert_eq!(distribution.legendre_residual(), &[0.0, 0.75, 0.0]);
        assert_eq!(distribution.total(), distribution.legendre_residual());
    }

    #[test]
    fn signed_legendre_residual_is_preserved_without_clipping() {
        let distribution = reconstruct_mixed_distribution(
            fields(1, 1, &[0.0], 1, &[-2.5]),
            grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
        )
        .unwrap();

        assert_eq!(distribution.legendre_at(0, 0), Some(-2.5));
        assert_eq!(distribution.total_at(0, 0), Some(-2.5));
    }

    #[test]
    fn reconstruction_requires_exact_coefficient_shapes() {
        let error = reconstruct_mixed_distribution(
            fields(2, 1, &[1.0], 1, &[0.5, 0.5]),
            grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
        );
        assert_eq!(
            error,
            Err(MixedReconstructionError::CoefficientShape {
                family: CoefficientFamily::Hermite,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn reconstruction_rejects_non_finite_coefficients() {
        let hermite_error = reconstruct_mixed_distribution(
            fields(1, 1, &[f64::NAN], 0, &[]),
            grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
        );
        assert_eq!(
            hermite_error,
            Err(MixedReconstructionError::NonFiniteCoefficient {
                family: CoefficientFamily::Hermite,
                index: 0,
            })
        );

        let legendre_error = reconstruct_mixed_distribution(
            fields(1, 0, &[], 1, &[f64::INFINITY]),
            grid(&[0.0], f64::NAN, 0.0, -1.0, 1.0),
        );
        assert_eq!(
            legendre_error,
            Err(MixedReconstructionError::NonFiniteCoefficient {
                family: CoefficientFamily::Legendre,
                index: 0,
            })
        );
    }

    #[test]
    fn reconstruction_detects_non_finite_output() {
        let distribution = reconstruct_mixed_distribution(
            fields(1, 2, &[1.7e308, 1.7e308], 1, &[1.7e308]),
            grid(&[1.0], 0.0, 1.0, -2.0, 2.0),
        );
        assert!(matches!(
            distribution,
            Err(MixedReconstructionError::NonFiniteReconstruction {
                spatial_index: 0,
                velocity_index: 0
            })
        ));
    }

    #[test]
    fn reconstruction_rejects_empty_grids_and_no_basis() {
        assert_eq!(
            reconstruct_mixed_distribution(
                fields(0, 1, &[1.0], 0, &[]),
                grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
            ),
            Err(MixedReconstructionError::EmptySpatialGrid)
        );
        assert_eq!(
            reconstruct_mixed_distribution(
                fields(1, 1, &[1.0], 0, &[]),
                grid(&[], 0.0, 1.0, -1.0, 1.0),
            ),
            Err(MixedReconstructionError::EmptyVelocityGrid)
        );
        assert_eq!(
            reconstruct_mixed_distribution(
                fields(1, 0, &[], 0, &[]),
                grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
            ),
            Err(MixedReconstructionError::NoRepresentedModes)
        );
    }

    #[test]
    fn reconstruction_rejects_shape_overflow_before_allocating() {
        assert_eq!(
            reconstruct_mixed_distribution(
                fields(usize::MAX, 2, &[], 0, &[]),
                grid(&[0.0], 0.0, 1.0, -1.0, 1.0),
            ),
            Err(MixedReconstructionError::ShapeOverflow)
        );
    }

    #[test]
    fn reconstruction_checks_all_output_shapes_and_accessors() {
        let distribution = reconstruct_mixed_distribution(
            fields(2, 1, &[1.0, 2.0], 0, &[]),
            grid(&[0.0, 1.0, 2.0], 0.0, 1.0, -1.0, 1.0),
        )
        .unwrap();

        assert_eq!(distribution.spatial_count, 2);
        assert_eq!(distribution.velocity_count, 3);
        assert_eq!(distribution.hermite_bulk().len(), 6);
        assert_eq!(distribution.total().len(), 6);
        assert_eq!(distribution.total_at(1, 2), distribution.hermite_at(1, 2));
        assert_eq!(distribution.total_at(2, 0), None);
        assert_eq!(distribution.total_at(0, 3), None);
    }
}
