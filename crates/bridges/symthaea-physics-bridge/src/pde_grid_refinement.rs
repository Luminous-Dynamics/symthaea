// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! One-dimensional arbitrary-resolution wave operator for grid-refinement checks.
//!
//! This module varies the number of interior grid points independently of the
//! number of physical dimensions. It uses a unit interval, fixed-zero Dirichlet
//! boundaries, wave speed c = 1, and spacing h = 1/(n+1). The state layout is
//! [u_0, ..., u_(n-1), v_0, ..., v_(n-1)].
//!
//! This is a semi-discrete spatial operator and energy oracle. It does not yet
//! provide a full space-time convergence study or claim continuum-level proof.

/// Return the number of interior sites encoded by the state length.
pub fn interior_points_for_state_len(state_len: usize) -> Option<usize> {
    if state_len < 2 || state_len % 2 != 0 {
        return None;
    }
    Some(state_len / 2)
}

/// Grid spacing for a positive number of interior points on the unit interval.
pub fn grid_spacing(interior_points: usize) -> Option<f64> {
    if interior_points == 0 {
        None
    } else {
        Some(1.0 / (interior_points as f64 + 1.0))
    }
}

/// Fundamental-mode frequency of the semi-discrete centered-difference operator.
pub fn discrete_fundamental_frequency(interior_points: usize) -> Option<f64> {
    let h = grid_spacing(interior_points)?;
    Some(2.0 * (std::f64::consts::PI * h / 2.0).sin() / h)
}

/// Relative error between the discrete fundamental frequency and the continuum value π.
pub fn fundamental_frequency_relative_error(interior_points: usize) -> Option<f64> {
    let discrete = discrete_fundamental_frequency(interior_points)?;
    Some((discrete - std::f64::consts::PI).abs() / std::f64::consts::PI)
}

/// Evaluate the 1-D semi-discrete scalar-wave RHS with fixed-zero boundaries.
///
/// Returns None for empty, odd-length, or otherwise malformed states.
pub fn wave_1d_rhs(state: &[f64]) -> Option<Vec<f64>> {
    let n = interior_points_for_state_len(state.len())?;
    let h = grid_spacing(n)?;
    let (u, v) = state.split_at(n);
    let inv_h_squared = 1.0 / (h * h);
    let mut derivative = vec![0.0; state.len()];
    derivative[..n].copy_from_slice(v);

    for i in 0..n {
        let left = if i == 0 { 0.0 } else { u[i - 1] };
        let right = if i + 1 == n { 0.0 } else { u[i + 1] };
        derivative[n + i] = (left - 2.0 * u[i] + right) * inv_h_squared;
    }
    Some(derivative)
}

/// Semi-discrete Hamiltonian with the continuum-compatible quadrature scaling.
///
/// H = h/2 sum(v_i^2) + 1/(2h) sum((u_(i+1)-u_i)^2), including boundary edges.
pub fn wave_1d_energy(state: &[f64]) -> Option<f64> {
    let n = interior_points_for_state_len(state.len())?;
    let h = grid_spacing(n)?;
    let (u, v) = state.split_at(n);

    let kinetic = 0.5 * h * v.iter().map(|value| value * value).sum::<f64>();
    let mut edge_difference_sum = u[0] * u[0];
    for i in 1..n {
        let difference = u[i] - u[i - 1];
        edge_difference_sum += difference * difference;
    }
    edge_difference_sum += u[n - 1] * u[n - 1];

    Some(kinetic + 0.5 * edge_difference_sum / h)
}

/// Analytically evaluate the directional derivative of H along the semi-discrete flow.
pub fn wave_1d_energy_derivative(state: &[f64]) -> Option<f64> {
    let n = interior_points_for_state_len(state.len())?;
    let h = grid_spacing(n)?;
    let (u, v) = state.split_at(n);
    let flow = wave_1d_rhs(state)?;

    let kinetic_derivative = h * (0..n).map(|i| v[i] * flow[n + i]).sum::<f64>();

    // Boundary velocities are zero. Each edge contributes
    // (u_right-u_left)(v_right-v_left)/h.
    let mut edge_derivative_sum = u[0] * v[0];
    for i in 1..n {
        edge_derivative_sum += (u[i] - u[i - 1]) * (v[i] - v[i - 1]);
    }
    edge_derivative_sum += u[n - 1] * v[n - 1];

    Some(kinetic_derivative + edge_derivative_sum / h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smooth_state(interior_points: usize) -> Vec<f64> {
        let h = grid_spacing(interior_points).unwrap();
        let mut state = vec![0.0; 2 * interior_points];
        for i in 0..interior_points {
            let x = (i + 1) as f64 * h;
            state[i] = (std::f64::consts::PI * x).sin();
            state[interior_points + i] = 0.3 * (2.0 * std::f64::consts::PI * x).cos();
        }
        state
    }

    #[test]
    fn state_validation_accepts_arbitrary_positive_resolutions() {
        for n in [1, 2, 3, 8, 17, 64] {
            assert_eq!(interior_points_for_state_len(2 * n), Some(n));
            assert!(grid_spacing(n).unwrap().is_finite());
        }
        for malformed in [0, 1, 3, 5, 11] {
            assert_eq!(interior_points_for_state_len(malformed), None);
        }
        assert_eq!(grid_spacing(0), None);
        assert_eq!(discrete_fundamental_frequency(0), None);
    }

    #[test]
    fn fundamental_sine_mode_is_an_eigenvector_of_the_discrete_operator() {
        for n in [1, 2, 4, 8, 16, 32] {
            let h = grid_spacing(n).unwrap();
            let omega = discrete_fundamental_frequency(n).unwrap();
            let mut state = vec![0.0; 2 * n];
            for i in 0..n {
                state[i] =
                    (std::f64::consts::PI * (i + 1) as f64 * h).sin();
            }

            let rhs = wave_1d_rhs(&state).unwrap();
            for i in 0..n {
                let expected = -omega * omega * state[i];
                assert!(
                    (rhs[n + i] - expected).abs() < 1e-10 * (1.0 + omega * omega),
                    "n={n}, i={i}, got={}, expected={expected}",
                    rhs[n + i]
                );
            }
        }
    }

    #[test]
    fn fundamental_frequency_converges_at_second_order() {
        let error_16 = fundamental_frequency_relative_error(16).unwrap();
        let error_32 = fundamental_frequency_relative_error(32).unwrap();
        let observed_order = (error_16 / error_32).log2();
        assert!(
            observed_order > 1.8 && observed_order < 2.1,
            "observed order={observed_order:.4}, errors=({error_16:.3e}, {error_32:.3e})"
        );
        assert!(fundamental_frequency_relative_error(32).unwrap()
            < fundamental_frequency_relative_error(16).unwrap());
        assert!(fundamental_frequency_relative_error(64).unwrap()
            < fundamental_frequency_relative_error(32).unwrap());
    }

    #[test]
    fn semi_discrete_energy_derivative_cancels_across_resolutions() {
        for n in [1, 2, 3, 8, 16, 32, 64] {
            let state = smooth_state(n);
            let derivative = wave_1d_energy_derivative(&state).unwrap();
            assert!(
                derivative.abs() < 1e-10,
                "n={n}, dH/dt={derivative:.3e}"
            );
            let energy = wave_1d_energy(&state).unwrap();
            assert!(energy.is_finite() && energy >= 0.0);
        }
    }

    #[test]
    fn independent_finite_difference_energy_derivative_is_small() {
        let epsilon = 1e-6;
        for n in [1, 2, 4, 8, 16] {
            let state = smooth_state(n);
            let flow = wave_1d_rhs(&state).unwrap();
            let plus: Vec<f64> = state
                .iter()
                .zip(&flow)
                .map(|(x, dx)| x + epsilon * dx)
                .collect();
            let minus: Vec<f64> = state
                .iter()
                .zip(&flow)
                .map(|(x, dx)| x - epsilon * dx)
                .collect();
            let finite_difference =
                (wave_1d_energy(&plus).unwrap() - wave_1d_energy(&minus).unwrap())
                    / (2.0 * epsilon);
            assert!(
                finite_difference.abs() < 1e-7,
                "n={n}, finite-difference dH/dt={finite_difference:.3e}"
            );
        }
    }
}
