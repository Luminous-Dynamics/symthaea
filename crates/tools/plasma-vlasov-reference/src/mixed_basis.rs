//! Source-derived mixed Hermite-Legendre velocity basis evaluators.
//!
//! Recurrences are pinned to ergodicio/adept at commit
//! 0c85ce0d3d66abab51212191c2644f58a31a49cd, file
//! adept/_hermite_legendre_1d/vector_field.py.
//!
//! This module evaluates basis functions only. It does not construct a
//! Vlasov-Poisson field, evolve coefficients, couple the bases, or make
//! positivity/conservation claims.

/// Row-major basis values with shape (mode_count, velocity_count).
#[derive(Debug, Clone, PartialEq)]
pub struct BasisValues {
    pub mode_count: usize,
    pub velocity_count: usize,
    values: Vec<f64>,
}

impl BasisValues {
    /// Return a basis value, or None for an out-of-range index.
    pub fn get(&self, mode: usize, velocity_index: usize) -> Option<f64> {
        if mode >= self.mode_count || velocity_index >= self.velocity_count {
            return None;
        }
        self.values
            .get(mode * self.velocity_count + velocity_index)
            .copied()
    }

    /// Borrow the row-major backing array.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
}

/// Invalid basis configuration or input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedBasisError {
    EmptyModeCount,
    EmptyVelocityGrid,
    InvalidShift,
    InvalidScale,
    InvalidLegendreWindow,
    NonFiniteVelocity { index: usize },
    NonFiniteHermiteCoordinate { index: usize },
    ShapeOverflow,
    AllocationFailed { elements: usize },
}

fn allocate_basis(
    mode_count: usize,
    velocity_count: usize,
) -> Result<BasisValues, MixedBasisError> {
    if mode_count == 0 {
        return Err(MixedBasisError::EmptyModeCount);
    }
    if velocity_count == 0 {
        return Err(MixedBasisError::EmptyVelocityGrid);
    }
    let elements = mode_count
        .checked_mul(velocity_count)
        .ok_or(MixedBasisError::ShapeOverflow)?;

    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| MixedBasisError::AllocationFailed { elements })?;
    values.resize(elements, 0.0);

    Ok(BasisValues {
        mode_count,
        velocity_count,
        values,
    })
}

/// Evaluate the asymmetrically weighted Hermite basis used by ADEPT.
///
/// For z = (v - shift) / scale, with scale > 0:
///
/// psi_0 = pi^(-1/2) exp(-z^2)
/// psi_1 = sqrt(2) z psi_0
/// psi_(n+1) = z sqrt(2/(n+1)) psi_n - sqrt(n/(n+1)) psi_(n-1)
///
/// The result is row-major with shape (mode_count, velocities.len()).
/// The normalized recurrence avoids constructing unnormalized Hermite
/// polynomials and factorials, which overflow at high order.
pub fn asymmetric_hermite_basis_values(
    mode_count: usize,
    velocities: &[f64],
    shift: f64,
    scale: f64,
) -> Result<BasisValues, MixedBasisError> {
    if !shift.is_finite() {
        return Err(MixedBasisError::InvalidShift);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(MixedBasisError::InvalidScale);
    }
    let mut basis = allocate_basis(mode_count, velocities.len())?;

    for (velocity_index, velocity) in velocities.iter().copied().enumerate() {
        if !velocity.is_finite() {
            return Err(MixedBasisError::NonFiniteVelocity {
                index: velocity_index,
            });
        }
        let z = (velocity - shift) / scale;
        if !z.is_finite() {
            return Err(MixedBasisError::NonFiniteHermiteCoordinate {
                index: velocity_index,
            });
        }

        let column = velocity_index;
        let psi_zero = std::f64::consts::PI.sqrt().recip() * (-(z * z)).exp();
        basis.values[column] = psi_zero;

        if mode_count > 1 {
            basis.values[basis.velocity_count + column] = std::f64::consts::SQRT_2 * z * psi_zero;
        }

        for mode in 1..mode_count.saturating_sub(1) {
            let previous = basis.values[(mode - 1) * basis.velocity_count + column];
            let current = basis.values[mode * basis.velocity_count + column];
            let next = z * (2.0 / (mode + 1) as f64).sqrt() * current
                - (mode as f64 / (mode + 1) as f64).sqrt() * previous;
            basis.values[(mode + 1) * basis.velocity_count + column] = next;
        }
    }

    Ok(basis)
}

/// Evaluate the normalized Legendre basis on a finite velocity interval.
///
/// The computational coordinate is s = 2 * (v - v_a) / (v_b - v_a) - 1,
/// algebraically equivalent to the source expression while avoiding the
/// intermediate sum v_a + v_b. The recurrence is P_0 = 1, P_1 = s, and
///
/// P_(m+1) = ((2m+1) s P_m - m P_(m-1)) / (m+1)
///
/// The returned basis is sqrt(2m+1) * P_m(s) inside the closed interval
/// [v_a, v_b] and exactly zero outside it. The array is row-major.
pub fn legendre_window_basis_values(
    mode_count: usize,
    velocities: &[f64],
    v_a: f64,
    v_b: f64,
) -> Result<BasisValues, MixedBasisError> {
    let width = v_b - v_a;
    if !v_a.is_finite() || !v_b.is_finite() || !width.is_finite() || width <= 0.0 {
        return Err(MixedBasisError::InvalidLegendreWindow);
    }

    let mut basis = allocate_basis(mode_count, velocities.len())?;

    for (velocity_index, velocity) in velocities.iter().copied().enumerate() {
        if !velocity.is_finite() {
            return Err(MixedBasisError::NonFiniteVelocity {
                index: velocity_index,
            });
        }
        if velocity < v_a || velocity > v_b {
            continue;
        }

        let s = 2.0 * ((velocity - v_a) / width) - 1.0;
        let column = velocity_index;
        basis.values[column] = 1.0;

        if mode_count > 1 {
            basis.values[basis.velocity_count + column] = s * 3.0_f64.sqrt();
        }

        for mode in 1..mode_count.saturating_sub(1) {
            let previous = basis.values[(mode - 1) * basis.velocity_count + column]
                / (2.0 * (mode - 1) as f64 + 1.0).sqrt();
            let current = basis.values[mode * basis.velocity_count + column]
                / (2.0 * mode as f64 + 1.0).sqrt();
            let next_polynomial = ((2.0 * mode as f64 + 1.0) * s * current
                - mode as f64 * previous)
                / (mode + 1) as f64;
            basis.values[(mode + 1) * basis.velocity_count + column] =
                (2.0 * (mode + 1) as f64 + 1.0).sqrt() * next_polynomial;
        }
    }

    Ok(basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn hermite_basis_matches_known_values_at_zero_coordinate() {
        let basis = asymmetric_hermite_basis_values(4, &[2.0], 2.0, 0.75).unwrap();
        let psi_zero = std::f64::consts::PI.sqrt().recip();

        close(basis.get(0, 0).unwrap(), psi_zero, 1.0e-15);
        close(basis.get(1, 0).unwrap(), 0.0, 0.0);
        close(
            basis.get(2, 0).unwrap(),
            -psi_zero / 2.0_f64.sqrt(),
            1.0e-15,
        );
        close(basis.get(3, 0).unwrap(), 0.0, 0.0);
    }

    #[test]
    fn hermite_basis_matches_known_values_at_unit_coordinate() {
        let basis = asymmetric_hermite_basis_values(3, &[1.0], 0.0, 1.0).unwrap();
        let psi_zero = (-1.0_f64).exp() / std::f64::consts::PI.sqrt();

        close(basis.get(0, 0).unwrap(), psi_zero, 1.0e-15);
        close(basis.get(1, 0).unwrap(), 2.0_f64.sqrt() * psi_zero, 1.0e-15);
        close(basis.get(2, 0).unwrap(), psi_zero / 2.0_f64.sqrt(), 1.0e-15);
    }

    #[test]
    fn hermite_basis_shape_and_row_major_access_are_explicit() {
        let basis = asymmetric_hermite_basis_values(3, &[-1.0, 0.0, 1.0], 0.0, 1.0).unwrap();

        assert_eq!(basis.mode_count, 3);
        assert_eq!(basis.velocity_count, 3);
        assert_eq!(basis.as_slice().len(), 9);
        assert_eq!(basis.get(1, 1), Some(0.0));
        assert_eq!(basis.get(3, 0), None);
        assert_eq!(basis.get(0, 3), None);
    }

    #[test]
    fn hermite_recurrence_remains_finite_at_high_order() {
        let velocities = [-5.0, -2.0, 0.0, 1.0, 4.0, 5.0];
        let basis = asymmetric_hermite_basis_values(256, &velocities, 0.0, 1.0).unwrap();

        assert!(basis.as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn legendre_basis_matches_midpoint_and_endpoint_values() {
        let basis = legendre_window_basis_values(3, &[-1.0, 0.0, 1.0], -1.0, 1.0).unwrap();

        close(basis.get(0, 1).unwrap(), 1.0, 0.0);
        close(basis.get(1, 1).unwrap(), 0.0, 0.0);
        close(basis.get(2, 1).unwrap(), -5.0_f64.sqrt() / 2.0, 1.0e-15);
        close(basis.get(1, 0).unwrap(), -3.0_f64.sqrt(), 1.0e-15);
        close(basis.get(2, 0).unwrap(), 5.0_f64.sqrt(), 1.0e-15);
        close(basis.get(1, 2).unwrap(), 3.0_f64.sqrt(), 1.0e-15);
        close(basis.get(2, 2).unwrap(), 5.0_f64.sqrt(), 1.0e-15);
    }

    #[test]
    fn legendre_basis_is_exactly_zero_outside_its_velocity_window() {
        let basis =
            legendre_window_basis_values(6, &[-2.0, -1.0, 0.0, 1.0, 2.0], -1.0, 1.0).unwrap();

        for mode in 0..basis.mode_count {
            assert_eq!(basis.get(mode, 0), Some(0.0));
            assert_eq!(basis.get(mode, 4), Some(0.0));
        }
        assert_ne!(basis.get(0, 1), Some(0.0));
        assert_ne!(basis.get(0, 3), Some(0.0));
    }

    #[test]
    fn legendre_interval_scaling_is_affine_invariant() {
        let normalized =
            legendre_window_basis_values(5, &[-1.0, -0.5, 0.0, 0.5, 1.0], -1.0, 1.0).unwrap();
        let shifted_scaled =
            legendre_window_basis_values(5, &[2.0, 3.0, 4.0, 5.0, 6.0], 2.0, 6.0).unwrap();

        assert_eq!(normalized, shifted_scaled);
    }

    #[test]
    fn hermite_invalid_frame_and_velocity_inputs_are_rejected() {
        assert_eq!(
            asymmetric_hermite_basis_values(2, &[0.0], f64::NAN, 1.0),
            Err(MixedBasisError::InvalidShift)
        );
        assert_eq!(
            asymmetric_hermite_basis_values(2, &[0.0], 0.0, 0.0),
            Err(MixedBasisError::InvalidScale)
        );
        assert_eq!(
            asymmetric_hermite_basis_values(2, &[f64::INFINITY], 0.0, 1.0),
            Err(MixedBasisError::NonFiniteVelocity { index: 0 })
        );
    }

    #[test]
    fn legendre_invalid_window_and_velocity_inputs_are_rejected() {
        assert_eq!(
            legendre_window_basis_values(2, &[0.0], 1.0, 1.0),
            Err(MixedBasisError::InvalidLegendreWindow)
        );
        assert_eq!(
            legendre_window_basis_values(2, &[f64::NAN], -1.0, 1.0),
            Err(MixedBasisError::NonFiniteVelocity { index: 0 })
        );
    }

    #[test]
    fn empty_shapes_are_rejected() {
        assert_eq!(
            asymmetric_hermite_basis_values(0, &[0.0], 0.0, 1.0),
            Err(MixedBasisError::EmptyModeCount)
        );
        assert_eq!(
            legendre_window_basis_values(2, &[], -1.0, 1.0),
            Err(MixedBasisError::EmptyVelocityGrid)
        );
    }

    #[test]
    fn basis_shape_overflow_is_rejected_before_allocation() {
        assert_eq!(
            asymmetric_hermite_basis_values(usize::MAX, &[0.0, 1.0], 0.0, 1.0),
            Err(MixedBasisError::ShapeOverflow)
        );
    }
}
