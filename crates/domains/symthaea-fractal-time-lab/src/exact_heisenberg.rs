// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small-system kicked anisotropic-Heisenberg Floquet reference.
//!
//! Floquet convention (hbar=1, period=1):
//!
//! U_F = exp[-i H_int] exp[-i theta/2 sum_i X_i]
//! H_int = sum_(i,j) (Jx XX + Jy YY + Jz ZZ) + sum_i h_i Z_i.
//!
//! The two-site XYZ gate is applied analytically. Overlapping graph edges do
//! not commute, so the interaction exponential uses a symmetric second-order
//! Suzuki-Trotter decomposition over greedy edge-color matchings and the
//! on-site field group. The trotter_slices field controls the discretization.
//!
//! This is not exact evolution of the full interacting graph unless all
//! interaction groups commute (e.g. the Ising Jx=Jy=0 limit). It is a
//! convergence-controlled small-system reference for model-family studies,
//! not a thermodynamic-limit solver or a claim to reproduce a specific
//! hardware experiment without matching its published protocol.

use crate::exact_floquet::FloquetError;
use rustfft::num_complex::Complex;
use std::collections::BTreeSet;
/// An anisotropic spin-1/2 XYZ interaction on an undirected edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeisenbergEdge {
    pub first: usize,
    pub second: usize,
    pub j_x: f64,
    pub j_y: f64,
    pub j_z: f64,
}

/// Finite-graph kicked anisotropic-Heisenberg simulator.
///
/// The maximum exact state-vector size is 20 spins (2^20 complex amplitudes).
/// The interaction evolution is second-order Suzuki-Trotter with an explicit
/// number of subdivisions per Floquet period.
#[derive(Clone, Debug)]
pub struct KickedHeisenbergFloquet {
    n_spins: usize,
    edges: Vec<HeisenbergEdge>,
    edge_colors: Vec<Vec<usize>>,
    fields: Vec<f64>,
    pulse_angle: f64,
    trotter_slices: usize,
    amplitudes: Vec<Complex<f64>>,
}

impl KickedHeisenbergFloquet {
    pub const MAX_SPINS: usize = 20;
    pub const MAX_TROTTER_SLICES: usize = 1024;

    /// Construct an arbitrary simple graph with XYZ interactions.
    ///
    /// Edges are canonicalized to ascending endpoint order. Duplicate
    /// undirected edges, self-edges, invalid endpoints, non-finite couplings,
    /// non-finite fields, and invalid Trotter slice counts are rejected.
    /// The initial state is the all-up computational basis state |00...0>.
    pub fn new_graph(
        n_spins: usize,
        edges: Vec<HeisenbergEdge>,
        fields: Vec<f64>,
        pulse_angle: f64,
        trotter_slices: usize,
    ) -> Result<Self, FloquetError> {
        if !(1..=Self::MAX_SPINS).contains(&n_spins) {
            return Err(FloquetError::InvalidSpinCount { n_spins });
        }
        if fields.len() != n_spins {
            return Err(FloquetError::FieldCount {
                expected: n_spins,
                actual: fields.len(),
            });
        }
        if !pulse_angle.is_finite() {
            return Err(FloquetError::NonFiniteParameter {
                name: "pulse_angle",
                index: 0,
            });
        }
        if trotter_slices == 0 || trotter_slices > Self::MAX_TROTTER_SLICES {
            return Err(FloquetError::InvalidTrotterSlices {
                slices: trotter_slices,
                maximum: Self::MAX_TROTTER_SLICES,
            });
        }
        for (index, value) in fields.iter().enumerate() {
            if !value.is_finite() {
                return Err(FloquetError::NonFiniteParameter {
                    name: "fields",
                    index,
                });
            }
        }

        let mut canonical_edges = Vec::with_capacity(edges.len());
        let mut seen_edges = BTreeSet::new();
        for (edge_index, edge) in edges.into_iter().enumerate() {
            if edge.first >= n_spins || edge.second >= n_spins || edge.first == edge.second {
                return Err(FloquetError::InvalidEdge {
                    edge_index,
                    first: edge.first,
                    second: edge.second,
                    n_spins,
                });
            }
            for (name, value) in [("j_x", edge.j_x), ("j_y", edge.j_y), ("j_z", edge.j_z)] {
                if !value.is_finite() {
                    return Err(FloquetError::NonFiniteParameter {
                        name,
                        index: edge_index,
                    });
                }
            }

            let first = edge.first.min(edge.second);
            let second = edge.first.max(edge.second);
            if !seen_edges.insert((first, second)) {
                return Err(FloquetError::DuplicateEdge { first, second });
            }
            canonical_edges.push(HeisenbergEdge {
                first,
                second,
                j_x: edge.j_x,
                j_y: edge.j_y,
                j_z: edge.j_z,
            });
        }

        let edge_colors = color_edges(&canonical_edges);
        let dimension = 1_usize
            .checked_shl(n_spins as u32)
            .ok_or(FloquetError::DimensionOverflow)?;
        let mut amplitudes = vec![Complex::new(0.0, 0.0); dimension];
        amplitudes[0] = Complex::new(1.0, 0.0);

        Ok(Self {
            n_spins,
            edges: canonical_edges,
            edge_colors,
            fields,
            pulse_angle,
            trotter_slices,
            amplitudes,
        })
    }

    /// Construct an open-boundary rectangular square-lattice interaction graph.
    ///
    /// Spin index is y * width + x. Horizontal and vertical edges can have
    /// different anisotropic XYZ couplings, supplied as [Jx, Jy, Jz].
    pub fn new_square_lattice(
        width: usize,
        height: usize,
        horizontal_xyz: [f64; 3],
        vertical_xyz: [f64; 3],
        fields: Vec<f64>,
        pulse_angle: f64,
        trotter_slices: usize,
    ) -> Result<Self, FloquetError> {
        let n_spins = width
            .checked_mul(height)
            .ok_or(FloquetError::DimensionOverflow)?;
        if !(1..=Self::MAX_SPINS).contains(&n_spins) {
            return Err(FloquetError::InvalidSpinCount { n_spins });
        }

        let mut edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let site = y * width + x;
                if x + 1 < width {
                    edges.push(HeisenbergEdge {
                        first: site,
                        second: site + 1,
                        j_x: horizontal_xyz[0],
                        j_y: horizontal_xyz[1],
                        j_z: horizontal_xyz[2],
                    });
                }
                if y + 1 < height {
                    edges.push(HeisenbergEdge {
                        first: site,
                        second: site + width,
                        j_x: vertical_xyz[0],
                        j_y: vertical_xyz[1],
                        j_z: vertical_xyz[2],
                    });
                }
            }
        }

        Self::new_graph(n_spins, edges, fields, pulse_angle, trotter_slices)
    }

    pub fn n_spins(&self) -> usize {
        self.n_spins
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edge_color_count(&self) -> usize {
        self.edge_colors.len()
    }

    pub fn trotter_slices(&self) -> usize {
        self.trotter_slices
    }

    pub fn edges(&self) -> &[HeisenbergEdge] {
        &self.edges
    }

    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// Read-only state-vector amplitudes in computational basis order.
    pub fn amplitudes(&self) -> &[Complex<f64>] {
        &self.amplitudes
    }

    /// Reset to a computational-basis product state.
    pub fn set_initial_basis(&mut self, basis: usize) -> Result<(), FloquetError> {
        if basis >= self.dimension() {
            return Err(FloquetError::InitialBasisOutOfRange {
                basis,
                dimension: self.dimension(),
            });
        }
        self.amplitudes.fill(Complex::new(0.0, 0.0));
        self.amplitudes[basis] = Complex::new(1.0, 0.0);
        Ok(())
    }

    /// Set an arbitrary finite complex state, normalizing it to unit norm.
    ///
    /// Basis index bits use the same little-endian spin ordering as
    /// `set_initial_basis`. A failed update leaves the current state unchanged.
    pub fn set_initial_state(
        &mut self,
        amplitudes: &[Complex<f64>],
    ) -> Result<(), FloquetError> {
        if amplitudes.len() != self.dimension() {
            return Err(FloquetError::InitialStateDimension {
                expected: self.dimension(),
                actual: amplitudes.len(),
            });
        }
        for (index, amplitude) in amplitudes.iter().enumerate() {
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(FloquetError::NonFiniteAmplitude { index });
            }
        }
        let norm_squared: f64 = amplitudes.iter().map(|amplitude| amplitude.norm_sqr()).sum();
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(FloquetError::InvalidInitialStateNorm);
        }
        let inverse_norm = norm_squared.sqrt().recip();
        let normalized: Vec<Complex<f64>> = amplitudes
            .iter()
            .map(|amplitude| *amplitude * inverse_norm)
            .collect();
        self.amplitudes = normalized;
        Ok(())
    }

    /// Apply one Floquet period using the declared symmetric Trotter schedule.
    pub fn step(&mut self) {
        self.apply_global_x_rotation();
        let slice_duration = 1.0 / self.trotter_slices as f64;
        for _ in 0..self.trotter_slices {
            self.apply_interaction_strang(slice_duration);
        }
    }

    /// State-vector norm squared; every gate in the schedule is unitary.
    pub fn norm_squared(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .sum()
    }

    /// Mean z magnetization normalized to [-1,1].
    pub fn magnetization(&self) -> f64 {
        let mut expectation = 0.0;
        for (basis, amplitude) in self.amplitudes.iter().enumerate() {
            let probability = amplitude.norm_sqr();
            let mut magnetization = 0.0;
            for spin in 0..self.n_spins {
                let z = if basis & (1 << spin) == 0 { 1.0 } else { -1.0 };
                magnetization += z;
            }
            expectation += probability * magnetization / self.n_spins as f64;
        }
        expectation
    }

    /// Stroboscopic magnetization including the initial sample at period zero.
    pub fn stroboscopic_magnetization(&mut self, periods: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(periods + 1);
        signal.push(self.magnetization());
        for _ in 0..periods {
            self.step();
            signal.push(self.magnetization());
        }
        signal
    }

    fn apply_global_x_rotation(&mut self) {
        let cosine = (0.5 * self.pulse_angle).cos();
        let sine = (0.5 * self.pulse_angle).sin();
        let minus_i_sine = Complex::new(0.0, -sine);
        let dimension = self.dimension();

        for spin in 0..self.n_spins {
            let mask = 1_usize << spin;
            for basis in 0..dimension {
                if basis & mask == 0 {
                    let partner = basis | mask;
                    let a = self.amplitudes[basis];
                    let b = self.amplitudes[partner];
                    self.amplitudes[basis] = a * cosine + b * minus_i_sine;
                    self.amplitudes[partner] = b * cosine + a * minus_i_sine;
                }
            }
        }
    }

    /// Symmetric second-order splitting of H_fields + sum_color H_color.
    ///
    /// Every color is a matching, so all two-site gates within a color act on
    /// disjoint spins and commute. The on-site Z fields commute with each other.
    fn apply_interaction_strang(&mut self, duration: f64) {
        if self.edge_colors.is_empty() {
            self.apply_field_group(duration);
            return;
        }

        self.apply_field_group(0.5 * duration);
        for color_index in 0..(self.edge_colors.len() - 1) {
            self.apply_edge_color(color_index, 0.5 * duration);
        }
        let final_color = self.edge_colors.len() - 1;
        self.apply_edge_color(final_color, duration);
        for color_index in (0..final_color).rev() {
            self.apply_edge_color(color_index, 0.5 * duration);
        }
        self.apply_field_group(0.5 * duration);
    }

    fn apply_field_group(&mut self, duration: f64) {
        if duration == 0.0 {
            return;
        }
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            let mut energy = 0.0;
            for spin in 0..self.n_spins {
                let z = if basis & (1 << spin) == 0 { 1.0 } else { -1.0 };
                energy += self.fields[spin] * z;
            }
            *amplitude *= Complex::from_polar(1.0, -duration * energy);
        }
    }

    fn apply_edge_color(&mut self, color_index: usize, duration: f64) {
        if duration == 0.0 {
            return;
        }
        // Copy the small edge-index schedule before mutating amplitudes.
        // This preserves the graph-color order while ending the immutable
        // borrow of self.edge_colors before apply_xyz_edge borrows self mutably.
        let edge_indices = self.edge_colors[color_index].clone();
        for edge_index in edge_indices {
            self.apply_xyz_edge(self.edges[edge_index], duration);
        }
    }

    /// Exact analytic exponential of one two-spin XYZ interaction.
    ///
    /// XX, YY and ZZ commute on a single edge. The |00>,|11> block has
    /// eigenphase Jz and coupling Jx-Jy; the |01>,|10> block has eigenphase
    /// -Jz and coupling Jx+Jy.
    fn apply_xyz_edge(&mut self, edge: HeisenbergEdge, duration: f64) {
        let first_mask = 1_usize << edge.first;
        let second_mask = 1_usize << edge.second;
        let even_cos = ((edge.j_x - edge.j_y) * duration).cos();
        let even_sin = ((edge.j_x - edge.j_y) * duration).sin();
        let even_phase = Complex::from_polar(1.0, -edge.j_z * duration);
        let odd_cos = ((edge.j_x + edge.j_y) * duration).cos();
        let odd_sin = ((edge.j_x + edge.j_y) * duration).sin();
        let odd_phase = Complex::from_polar(1.0, edge.j_z * duration);
        let even_off_diagonal = Complex::new(0.0, -even_sin);
        let odd_off_diagonal = Complex::new(0.0, -odd_sin);

        for basis in 0..self.dimension() {
            let first_is_zero = basis & first_mask == 0;
            let second_is_zero = basis & second_mask == 0;

            if first_is_zero && second_is_zero {
                let partner = basis | first_mask | second_mask;
                let a = self.amplitudes[basis];
                let b = self.amplitudes[partner];
                self.amplitudes[basis] = even_phase * (a * even_cos + b * even_off_diagonal);
                self.amplitudes[partner] = even_phase * (b * even_cos + a * even_off_diagonal);
            } else if first_is_zero && !second_is_zero {
                let partner = (basis | first_mask) & !second_mask;
                let a = self.amplitudes[basis];
                let b = self.amplitudes[partner];
                self.amplitudes[basis] = odd_phase * (a * odd_cos + b * odd_off_diagonal);
                self.amplitudes[partner] = odd_phase * (b * odd_cos + a * odd_off_diagonal);
            }
        }
    }
}

/// Greedy edge coloring into matchings. Each color contains no shared endpoint.
fn color_edges(edges: &[HeisenbergEdge]) -> Vec<Vec<usize>> {
    let mut colors: Vec<Vec<usize>> = Vec::new();
    let mut occupied: Vec<BTreeSet<usize>> = Vec::new();

    for (edge_index, edge) in edges.iter().enumerate() {
        let selected_color = occupied
            .iter()
            .position(|sites| !sites.contains(&edge.first) && !sites.contains(&edge.second));
        let color_index = selected_color.unwrap_or_else(|| {
            colors.push(Vec::new());
            occupied.push(BTreeSet::new());
            colors.len() - 1
        });
        colors[color_index].push(edge_index);
        occupied[color_index].insert(edge.first);
        occupied[color_index].insert(edge.second);
    }
    colors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact_floquet::{IsingEdge, KickedIsingFloquet};
    use std::f64::consts::PI;

    fn dense_hamiltonian_action(
        state: &[Complex<f64>],
        n_spins: usize,
        edges: &[HeisenbergEdge],
        fields: &[f64],
    ) -> Vec<Complex<f64>> {
        let mut output = vec![Complex::new(0.0, 0.0); state.len()];
        for (basis, amplitude) in state.iter().enumerate() {
            let mut diagonal_energy = 0.0;
            for spin in 0..n_spins {
                let z = if basis & (1 << spin) == 0 { 1.0 } else { -1.0 };
                diagonal_energy += fields[spin] * z;
            }
            output[basis] += *amplitude * diagonal_energy;

            for edge in edges {
                let first_is_one = basis & (1 << edge.first) != 0;
                let second_is_one = basis & (1 << edge.second) != 0;
                let z_first = if first_is_one { -1.0 } else { 1.0 };
                let z_second = if second_is_one { -1.0 } else { 1.0 };
                diagonal_energy = edge.j_z * z_first * z_second;
                output[basis] += *amplitude * diagonal_energy;

                let yy_sign = if first_is_one == second_is_one {
                    -1.0
                } else {
                    1.0
                };
                let partner = basis ^ (1 << edge.first) ^ (1 << edge.second);
                output[basis] += state[partner] * (edge.j_x + edge.j_y * yy_sign);
            }
        }
        output
    }

    fn dense_x_pulse(
        state: &[Complex<f64>],
        n_spins: usize,
        pulse_angle: f64,
    ) -> Vec<Complex<f64>> {
        let mut amplitudes = state.to_vec();
        let cosine = (pulse_angle / 2.0).cos();
        let minus_i_sine = Complex::new(0.0, -(pulse_angle / 2.0).sin());
        for spin in 0..n_spins {
            let mask = 1_usize << spin;
            let previous = amplitudes.clone();
            for basis in 0..amplitudes.len() {
                if basis & mask == 0 {
                    let partner = basis | mask;
                    amplitudes[basis] = previous[basis] * cosine + previous[partner] * minus_i_sine;
                    amplitudes[partner] =
                        previous[partner] * cosine + previous[basis] * minus_i_sine;
                }
            }
        }
        amplitudes
    }

    fn dense_exact_interaction_exponential(
        initial: &[Complex<f64>],
        n_spins: usize,
        edges: &[HeisenbergEdge],
        fields: &[f64],
    ) -> Vec<Complex<f64>> {
        // Taylor action for exp(-iH) on a small system. The chosen test
        // Hamiltonian has modest norm; 64 terms make truncation negligible.
        let mut result = initial.to_vec();
        let mut term = initial.to_vec();
        for order in 1..=64 {
            let acted = dense_hamiltonian_action(&term, n_spins, edges, fields);
            let coefficient = Complex::new(0.0, -1.0 / order as f64);
            let mut term_norm_squared = 0.0;
            for index in 0..term.len() {
                term[index] = acted[index] * coefficient;
                result[index] += term[index];
                term_norm_squared += term[index].norm_sqr();
            }
            if term_norm_squared.sqrt() < 1.0e-16 {
                break;
            }
        }
        result
    }

    fn dense_floquet_step_reference(
        n_spins: usize,
        edges: &[HeisenbergEdge],
        fields: &[f64],
        pulse_angle: f64,
    ) -> Vec<Complex<f64>> {
        let mut initial = vec![Complex::new(0.0, 0.0); 1_usize << n_spins];
        initial[0] = Complex::new(1.0, 0.0);
        let after_pulse = dense_x_pulse(&initial, n_spins, pulse_angle);
        dense_exact_interaction_exponential(&after_pulse, n_spins, edges, fields)
    }

    #[test]
    fn one_edge_xyz_gate_matches_closed_form() {
        let j_x = 0.31;
        let j_y = 0.11;
        let j_z = 0.23;
        let mut model = KickedHeisenbergFloquet::new_graph(
            2,
            vec![HeisenbergEdge {
                first: 0,
                second: 1,
                j_x,
                j_y,
                j_z,
            }],
            vec![0.0; 2],
            0.0,
            1,
        )
        .unwrap();
        model.step();

        let expected_00 = Complex::from_polar(1.0, -j_z) * (j_x - j_y).cos();
        let expected_11 = Complex::from_polar(1.0, -j_z) * Complex::new(0.0, -(j_x - j_y).sin());
        assert!((model.amplitudes()[0] - expected_00).norm() < 1e-12);
        assert!((model.amplitudes()[3] - expected_11).norm() < 1e-12);
        assert!(model.amplitudes()[1].norm() < 1e-12);
        assert!(model.amplitudes()[2].norm() < 1e-12);
    }

    #[test]
    fn anisotropic_solver_reduces_to_exact_kicked_ising_when_jx_jy_are_zero() {
        let ising_edges = vec![
            IsingEdge {
                first: 0,
                second: 1,
                coupling: 0.27,
            },
            IsingEdge {
                first: 1,
                second: 2,
                coupling: -0.19,
            },
        ];
        let heisenberg_edges: Vec<HeisenbergEdge> = ising_edges
            .iter()
            .map(|edge| HeisenbergEdge {
                first: edge.first,
                second: edge.second,
                j_x: 0.0,
                j_y: 0.0,
                j_z: edge.coupling,
            })
            .collect();
        let fields = vec![0.13, -0.08, 0.21];
        let mut ising =
            KickedIsingFloquet::new_graph(3, ising_edges, fields.clone(), PI + 0.07).unwrap();
        let mut heisenberg =
            KickedHeisenbergFloquet::new_graph(3, heisenberg_edges, fields, PI + 0.07, 1).unwrap();

        for _ in 0..12 {
            ising.step();
            heisenberg.step();
            for (left, right) in ising.amplitudes().iter().zip(heisenberg.amplitudes()) {
                assert!((*left - *right).norm() < 1e-11);
            }
        }
    }

    #[test]
    fn arbitrary_complex_initial_state_is_normalized_and_evolves_unitarily() {
        let mut model = KickedHeisenbergFloquet::new_square_lattice(
            2,
            2,
            [0.27, 0.13, 0.41],
            [0.19, 0.31, 0.35],
            vec![0.01, -0.02, 0.03, -0.04],
            PI + 0.09,
            4,
        )
        .unwrap();
        let amplitudes = [
            Complex::new(1.0, 0.0),
            Complex::new(0.5, -0.5),
            Complex::new(-0.25, 0.75),
            Complex::new(0.0, 1.0),
            Complex::new(0.2, -0.3),
            Complex::new(-0.6, 0.1),
            Complex::new(0.4, 0.2),
            Complex::new(0.1, 0.9),
            Complex::new(-0.2, 0.2),
            Complex::new(0.3, -0.1),
            Complex::new(0.8, 0.4),
            Complex::new(-0.1, -0.5),
            Complex::new(0.2, 0.1),
            Complex::new(-0.4, 0.2),
            Complex::new(0.6, -0.3),
            Complex::new(0.1, 0.1),
        ];
        model.set_initial_state(&amplitudes).unwrap();
        assert!((model.norm_squared() - 1.0).abs() < 1.0e-12);
        for _ in 0..12 {
            model.step();
            assert!((model.norm_squared() - 1.0).abs() < 1.0e-10);
        }
    }

    #[test]
    fn invalid_arbitrary_initial_state_is_rejected_without_mutation() {
        let mut model = KickedHeisenbergFloquet::new_graph(
            2,
            vec![HeisenbergEdge {
                first: 0,
                second: 1,
                j_x: 0.2,
                j_y: 0.3,
                j_z: 0.4,
            }],
            vec![0.0; 2],
            PI,
            2,
        )
        .unwrap();
        let original = model.amplitudes().to_vec();
        assert!(matches!(
            model.set_initial_state(&[Complex::new(1.0, 0.0)]),
            Err(FloquetError::InitialStateDimension { .. })
        ));
        assert_eq!(
            model.set_initial_state(&[Complex::new(0.0, 0.0); 4]),
            Err(FloquetError::InvalidInitialStateNorm)
        );
        let mut non_finite = [Complex::new(0.0, 0.0); 4];
        non_finite[2] = Complex::new(0.0, f64::INFINITY);
        assert_eq!(
            model.set_initial_state(&non_finite),
            Err(FloquetError::NonFiniteAmplitude { index: 2 })
        );
        assert_eq!(model.amplitudes(), original.as_slice());
    }

    #[test]
    fn graph_trotter_evolution_preserves_norm() {
        let mut model = KickedHeisenbergFloquet::new_square_lattice(
            3,
            2,
            [0.37, 0.21, 0.43],
            [0.19, 0.33, 0.28],
            vec![0.01, -0.02, 0.03, -0.04, 0.02, -0.01],
            PI + 0.08,
            8,
        )
        .unwrap();
        assert_eq!(model.edge_count(), 7);
        assert!(model.edge_color_count() >= 2);

        for _ in 0..20 {
            model.step();
            assert!((model.norm_squared() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn second_order_trotter_converges_against_independent_dense_exponential() {
        let edges = vec![
            HeisenbergEdge {
                first: 0,
                second: 1,
                j_x: 0.11,
                j_y: 0.17,
                j_z: 0.13,
            },
            HeisenbergEdge {
                first: 1,
                second: 2,
                j_x: 0.09,
                j_y: 0.14,
                j_z: 0.18,
            },
        ];
        let fields = vec![0.03, -0.04, 0.02];
        let pulse_angle = 0.41;
        let reference = dense_floquet_step_reference(3, &edges, &fields, pulse_angle);

        let mut errors = Vec::new();
        for slices in [1, 2, 4, 8, 16] {
            let mut candidate = KickedHeisenbergFloquet::new_graph(
                3,
                edges.clone(),
                fields.clone(),
                pulse_angle,
                slices,
            )
            .unwrap();
            candidate.step();
            let error = candidate
                .amplitudes()
                .iter()
                .zip(reference.iter())
                .map(|(left, right)| (*left - *right).norm_sqr())
                .sum::<f64>()
                .sqrt();
            errors.push(error);
        }
        assert!(
            errors.windows(2).all(|pair| pair[1] < pair[0]),
            "{errors:?}"
        );
        let refinement_ratios: Vec<f64> = errors.windows(2).map(|pair| pair[0] / pair[1]).collect();
        assert!(
            refinement_ratios
                .iter()
                .all(|ratio| (3.5..=4.5).contains(ratio)),
            "second-order refinement ratios={refinement_ratios:?}"
        );
    }

    #[test]
    fn square_lattice_constructor_and_matching_colors_are_valid() {
        let model = KickedHeisenbergFloquet::new_square_lattice(
            2,
            2,
            [0.3, 0.2, 0.1],
            [0.4, 0.1, 0.2],
            vec![0.0; 4],
            PI,
            4,
        )
        .unwrap();
        assert_eq!(model.edge_count(), 4);
        for color in &model.edge_colors {
            let mut endpoints = BTreeSet::new();
            for &edge_index in color {
                let edge = model.edges[edge_index];
                assert!(endpoints.insert(edge.first));
                assert!(endpoints.insert(edge.second));
            }
        }
    }

    #[test]
    fn global_spin_inversion_maps_all_up_magnetization_to_all_down() {
        let mut all_up = KickedHeisenbergFloquet::new_square_lattice(
            2,
            2,
            [0.27, 0.13, 0.41],
            [0.19, 0.31, 0.35],
            vec![0.0; 4],
            PI + 0.09,
            8,
        )
        .unwrap();
        let mut all_down = KickedHeisenbergFloquet::new_square_lattice(
            2,
            2,
            [0.27, 0.13, 0.41],
            [0.19, 0.31, 0.35],
            vec![0.0; 4],
            PI + 0.09,
            8,
        )
        .unwrap();
        all_down.set_initial_basis((1 << 4) - 1).unwrap();

        let up_signal = all_up.stroboscopic_magnetization(24);
        let down_signal = all_down.stroboscopic_magnetization(24);
        for (up, down) in up_signal.iter().zip(down_signal.iter()) {
            assert!((up + down).abs() < 1e-10);
        }
    }

    #[test]
    fn malformed_graphs_and_trotter_parameters_fail_closed() {
        assert!(matches!(
            KickedHeisenbergFloquet::new_graph(2, vec![], vec![0.0; 2], PI, 0),
            Err(FloquetError::InvalidTrotterSlices { .. })
        ));
        assert!(matches!(
            KickedHeisenbergFloquet::new_graph(
                2,
                vec![
                    HeisenbergEdge {
                        first: 0,
                        second: 1,
                        j_x: 0.1,
                        j_y: 0.2,
                        j_z: 0.3,
                    },
                    HeisenbergEdge {
                        first: 1,
                        second: 0,
                        j_x: 0.1,
                        j_y: 0.2,
                        j_z: 0.3,
                    },
                ],
                vec![0.0; 2],
                PI,
                1,
            ),
            Err(FloquetError::DuplicateEdge { .. })
        ));
        assert!(matches!(
            KickedHeisenbergFloquet::new_graph(2, vec![], vec![0.0; 2], f64::NAN, 1,),
            Err(FloquetError::NonFiniteParameter { .. })
        ));
    }
}
