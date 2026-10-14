// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A dimension-general, finite-difference scalar-wave benchmark.
//!
//! This module varies **physical spatial dimension**, not HDC vector width
//! and not merely the number of ODE state variables. In spatial dimension
//! `d`, it uses a tiny hypercubic lattice with two free sites along every
//! axis (`2^d` sites total), fixed-zero Dirichlet boundary values, and
//! `c = h = 1`:
//!
//! `üᵢ = Σₐ (u_{i xor 2ᵃ} - 2uᵢ)`.
//!
//! The binary index encodes each site's coordinate in `{0, 1}^d`; toggling
//! bit `a` selects the adjacent interior site along axis `a`. Each site also
//! touches one fixed-zero boundary face per axis. The Hamiltonian is
//!
//! `H = 1/2 Σᵢ vᵢ² + 1/2 Σₐ [Σ_{i: bitₐ(i)=0}(uᵢ-u_{i xor 2ᵃ})² + Σᵢ uᵢ²]`.
//!
//! Its time derivative cancels exactly under the semidiscrete dynamics.
//! For `d = 1`, this reduces to the existing two-interior-point wave chain
//! in `pde_wave_stage_a.rs`; `d = 2` and above genuinely add spatial axes.
//!
//! Scope is intentionally narrow: this is a scalar linear wave equation on
//! a minimal lattice, not a relativistic field solver and not evidence for
//! extra physical dimensions. It provides a known-answer benchmark for
//! dimension sweeps and for the existing autonomous-invariant discovery
//! pipeline. Discovery success must be measured separately from truth-energy
//! conservation; a correct hand-derived oracle is not a discovery result.

/// Return the physical spatial dimension for a valid state length.
///
/// State layout is `[u_0, ..., u_(N-1), v_0, ..., v_(N-1)]`, with
/// `N = 2^d` free lattice sites. Returns `None` for malformed or
/// unsupported state lengths.
pub fn spatial_dimension_for_state_len(state_len: usize) -> Option<usize> {
    if state_len % 2 != 0 {
        return None;
    }
    let sites = state_len / 2;
    if sites < 2 || !sites.is_power_of_two() {
        return None;
    }
    Some(sites.trailing_zeros() as usize)
}

/// Evaluate the first-order RHS for the hypercubic wave benchmark.
///
/// Panics when `state.len()` does not encode `2^d` lattice sites with
/// `d >= 1`. This mirrors the infallible function-pointer signature used by
/// the autonomous-invariant discovery API; callers that accept external
/// state lengths should validate with `spatial_dimension_for_state_len`.
pub fn hypercubic_wave_rhs(state: &[f64], _t: f64) -> Vec<f64> {
    let sites = state.len() / 2;
    let dimensions = spatial_dimension_for_state_len(state.len())
        .expect("hypercubic wave state must have 2 * 2^d entries for d >= 1");
    let (u, v) = state.split_at(sites);
    let mut derivative = vec![0.0; state.len()];

    for i in 0..sites {
        derivative[i] = v[i];
        let mut laplacian = 0.0;
        for axis in 0..dimensions {
            let neighbor = i ^ (1usize << axis);
            laplacian += u[neighbor] - 2.0 * u[i];
        }
        derivative[sites + i] = laplacian;
    }
    derivative
}

/// Evaluate the positive semidefinite discrete Hamiltonian.
///
/// Uses one contribution per interior bond and one contribution per
/// site/axis for the adjacent fixed-zero boundary. Energy is dimensionless
/// under this benchmark's `c = h = 1` normalization.
pub fn hypercubic_wave_energy(state: &[f64]) -> f64 {
    let sites = state.len() / 2;
    let dimensions = spatial_dimension_for_state_len(state.len())
        .expect("hypercubic wave state must have 2 * 2^d entries for d >= 1");
    let (u, v) = state.split_at(sites);

    let kinetic: f64 = v.iter().map(|velocity| 0.5 * velocity * velocity).sum();
    let mut potential = 0.0;
    for i in 0..sites {
        for axis in 0..dimensions {
            let bit = 1usize << axis;
            // Count each interior bond exactly once.
            if i & bit == 0 {
                let difference = u[i] - u[i | bit];
                potential += 0.5 * difference * difference;
            }
            // Each site has one fixed-zero boundary face per axis.
            potential += 0.5 * u[i] * u[i];
        }
    }
    kinetic + potential
}

/// Compute dH/dt from the Hamiltonian's bond and boundary terms.
///
/// This independent derivative calculation is useful for a stringent
/// invariant oracle: it does not estimate conservation from a time
/// integrator's energy drift.
pub fn hypercubic_wave_energy_derivative(state: &[f64]) -> f64 {
    let sites = state.len() / 2;
    let dimensions = spatial_dimension_for_state_len(state.len())
        .expect("hypercubic wave state must have 2 * 2^d entries for d >= 1");
    let (u, v) = state.split_at(sites);
    let rhs = hypercubic_wave_rhs(state, 0.0);

    // d/dt of the kinetic term: Σᵢ vᵢ aᵢ.
    let mut derivative: f64 = (0..sites).map(|i| v[i] * rhs[sites + i]).sum();

    // d/dt of each interior-bond and fixed-boundary contribution.
    for i in 0..sites {
        for axis in 0..dimensions {
            let bit = 1usize << axis;
            derivative += u[i] * v[i];
            if i & bit == 0 {
                let neighbor = i | bit;
                derivative += (u[i] - u[neighbor]) * (v[i] - v[neighbor]);
            }
        }
    }
    derivative
}

/// Integrate a state with fixed-step classical RK4.
///
/// The trajectory stores the state before each step, matching
/// `pde_wave_stage_a::wave_trajectory`. `dt` must be finite and positive.
pub fn hypercubic_wave_trajectory(
    initial_state: &[f64],
    steps: usize,
    dt: f64,
) -> Vec<Vec<f64>> {
    assert!(
        spatial_dimension_for_state_len(initial_state.len()).is_some(),
        "initial state must have 2 * 2^d entries for d >= 1"
    );
    assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");

    let mut state = initial_state.to_vec();
    let mut trajectory = Vec::with_capacity(steps);
    for _ in 0..steps {
        trajectory.push(state.clone());
        let k1 = hypercubic_wave_rhs(&state, 0.0);
        let k2 = hypercubic_wave_rhs(&add_scaled(&state, &k1, 0.5 * dt), 0.0);
        let k3 = hypercubic_wave_rhs(&add_scaled(&state, &k2, 0.5 * dt), 0.0);
        let k4 = hypercubic_wave_rhs(&add_scaled(&state, &k3, dt), 0.0);
        for i in 0..state.len() {
            state[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    trajectory
}

fn add_scaled(state: &[f64], derivative: &[f64], scale: f64) -> Vec<f64> {
    state
        .iter()
        .zip(derivative)
        .map(|(value, slope)| value + scale * slope)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_length_encodes_spatial_dimension_not_hdc_width() {
        for spatial_dimensions in 1..=6 {
            let state_len = 1usize << (spatial_dimensions + 1);
            assert_eq!(
                spatial_dimension_for_state_len(state_len),
                Some(spatial_dimensions),
                "state_len={state_len}"
            );
        }
    }

    #[test]
    fn malformed_state_lengths_are_rejected() {
        for state_len in [0, 1, 2, 3, 6, 10, 12, 24] {
            assert_eq!(spatial_dimension_for_state_len(state_len), None);
        }
    }

    #[test]
    fn one_dimensional_rhs_matches_the_existing_wave_benchmark() {
        use crate::pde_wave_stage_a::wave_rhs;

        let state = [0.7, -0.2, 0.4, -0.6];
        let actual = hypercubic_wave_rhs(&state, 0.0);
        let expected = wave_rhs(&state, 0.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn energy_derivative_cancels_in_spatial_dimensions_one_through_six() {
        for dimensions in 1..=6 {
            let sites = 1usize << dimensions;
            let mut state = vec![0.0; 2 * sites];
            for i in 0..sites {
                state[i] = ((i + 1) as f64 * 0.37).sin();
                state[sites + i] = 0.3 * ((i + 1) as f64 * 0.61).cos();
            }
            let derivative = hypercubic_wave_energy_derivative(&state);
            assert!(
                derivative.abs() < 1e-12,
                "d={dimensions}, dH/dt={derivative:.3e}"
            );
            assert!(hypercubic_wave_energy(&state).is_finite());
            assert!(hypercubic_wave_energy(&state) >= 0.0);
        }
    }

    #[test]
    fn one_dimensional_hamiltonian_matches_stage_a_energy_expression() {
        use crate::pde_wave_stage_a::wave_energy_truth;

        let state = [0.7, -0.2, 0.4, -0.6];
        let names = ["u1", "u2", "v1", "v2"];
        let assignments: Vec<(&str, f64)> = names.into_iter().zip(state).collect();
        let stage_a_energy = wave_energy_truth().eval(&assignments);
        let hypercubic_energy = hypercubic_wave_energy(&state);

        assert!(
            (stage_a_energy - hypercubic_energy).abs() < 1e-12,
            "Stage A energy={stage_a_energy}, hypercubic energy={hypercubic_energy}"
        );
    }

    #[test]
    fn finite_difference_directional_derivative_of_energy_is_zero_in_dimensions_one_through_six() {
        // Independent check: estimate ∇H · f(u,v) directly from the energy
        // function, without calling hypercubic_wave_energy_derivative.
        let epsilon = 1e-6;
        for dimensions in 1..=6 {
            let sites = 1usize << dimensions;
            let mut state = vec![0.0; 2 * sites];
            for i in 0..sites {
                state[i] = ((i + 1) as f64 * 0.37).sin();
                state[sites + i] = 0.3 * ((i + 1) as f64 * 0.61).cos();
            }
            let flow = hypercubic_wave_rhs(&state, 0.0);
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
            let directional_derivative =
                (hypercubic_wave_energy(&plus) - hypercubic_wave_energy(&minus))
                    / (2.0 * epsilon);

            assert!(
                directional_derivative.abs() < 1e-8,
                "d={dimensions}, finite-difference dH/dt={directional_derivative:.3e}"
            );
        }
    }

    #[test]
    fn rk4_energy_drift_remains_small_through_six_spatial_dimensions() {
        for dimensions in 1..=6 {
            let sites = 1usize << dimensions;
            let mut initial = vec![0.0; 2 * sites];
            for i in 0..sites {
                initial[i] = ((i + 1) as f64 * 0.41).sin();
                initial[sites + i] = 0.2 * ((i + 1) as f64 * 0.29).cos();
            }
            let initial_energy = hypercubic_wave_energy(&initial);
            let trajectory = hypercubic_wave_trajectory(&initial, 2_000, 0.001);
            let max_relative_drift = trajectory
                .iter()
                .map(|state| {
                    (hypercubic_wave_energy(state) - initial_energy).abs()
                        / initial_energy.max(f64::MIN_POSITIVE)
                })
                .fold(0.0_f64, f64::max);
            assert!(
                max_relative_drift < 1e-8,
                "d={dimensions}, max relative RK4 energy drift={max_relative_drift:.3e}"
            );
        }
    }
}
