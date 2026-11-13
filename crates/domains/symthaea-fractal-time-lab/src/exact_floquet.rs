// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact state-vector reference evolution for a finite kicked-Ising graph.
//!
//! One Floquet period applies a global x rotation followed by a diagonal
//! Ising/longitudinal-field phase on an explicit graph:
//!
//! U_F = exp(-i [sum_(i,j) J_ij Z_i Z_j + sum_i h_i Z_i])
//!       exp(-i theta/2 sum_i X_i).
//!
//! This is exact finite-dimensional state-vector evolution (up to f64
//! arithmetic), not an MPS approximation. It is an oracle for small systems
//! and for testing time-crystal observables. An ideal pi pulse produces a
//! period-two response even without interactions; therefore that response
//! alone is NOT evidence of a discrete time-crystalline phase.
//!
//! Open boundary conditions and hbar=1, period=1 units are used.
//! Complexity is O(N_sites * 2^N_sites) per Floquet period and O(2^N_sites)
//! memory.

use rustfft::num_complex::Complex;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Errors constructing an exact finite-dimensional Floquet system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FloquetError {
    InvalidSpinCount {
        n_spins: usize,
    },
    DimensionOverflow,
    CouplingCount {
        expected: usize,
        actual: usize,
    },
    FieldCount {
        expected: usize,
        actual: usize,
    },
    NonFiniteParameter {
        name: &'static str,
        index: usize,
    },
    InitialBasisOutOfRange {
        basis: usize,
        dimension: usize,
    },
    InitialStateDimension {
        expected: usize,
        actual: usize,
    },
    NonFiniteAmplitude {
        index: usize,
    },
    InvalidInitialStateNorm,
    InvalidEdge {
        edge_index: usize,
        first: usize,
        second: usize,
        n_spins: usize,
    },
    DuplicateEdge {
        first: usize,
        second: usize,
    },
    InvalidTrotterSlices {
        slices: usize,
        maximum: usize,
    },
    InvalidProbeFrequency,
}

impl fmt::Display for FloquetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpinCount { n_spins } => {
                write!(f, "n_spins must be in 1..=20, got {n_spins}")
            }
            Self::DimensionOverflow => write!(f, "state-vector dimension overflow"),
            Self::CouplingCount { expected, actual } => {
                write!(f, "expected {expected} Ising couplings, got {actual}")
            }
            Self::FieldCount { expected, actual } => {
                write!(f, "expected {expected} longitudinal fields, got {actual}")
            }
            Self::NonFiniteParameter { name, index } => {
                write!(f, "{name}[{index}] is not finite")
            }
            Self::InitialBasisOutOfRange { basis, dimension } => {
                write!(
                    f,
                    "basis index {basis} is outside state dimension {dimension}"
                )
            }
            Self::InitialStateDimension { expected, actual } => {
                write!(
                    f,
                    "initial state has {actual} amplitudes; expected {expected}"
                )
            }
            Self::NonFiniteAmplitude { index } => {
                write!(f, "initial state amplitude {index} is not finite")
            }
            Self::InvalidInitialStateNorm => {
                write!(f, "initial state must have a finite, nonzero norm")
            }
            Self::InvalidEdge {
                edge_index,
                first,
                second,
                n_spins,
            } => {
                write!(
                    f,
                    "edge {edge_index} ({first}, {second}) is invalid for {n_spins} spins"
                )
            }
            Self::DuplicateEdge { first, second } => {
                write!(f, "duplicate Ising edge ({first}, {second})")
            }
            Self::InvalidTrotterSlices { slices, maximum } => {
                write!(
                    f,
                    "Trotter slice count must be in 1..={maximum}, got {slices}"
                )
            }
            Self::InvalidProbeFrequency => {
                write!(
                    f,
                    "AC probe frequency must be finite and in [0, 0.5] cycles per drive period"
                )
            }
        }
    }
}

impl Error for FloquetError {}

/// A ZZ interaction edge in the kicked-Ising graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsingEdge {
    pub first: usize,
    pub second: usize,
    pub coupling: f64,
}

/// Exact state-vector simulator for a finite kicked-Ising interaction graph.
#[derive(Clone, Debug)]
pub struct KickedIsingFloquet {
    n_spins: usize,
    edges: Vec<IsingEdge>,
    fields: Vec<f64>,
    pulse_angle: f64,
    amplitudes: Vec<Complex<f64>>,
}

impl KickedIsingFloquet {
    /// Construct an open-boundary nearest-neighbour chain in the all-up
    /// computational-basis state |00...0>.
    ///
    /// couplings[i] multiplies Z_i Z_(i+1); fields[i] multiplies Z_i.
    /// The maximum size is capped at 20 spins to bound exact-state memory.
    pub fn new(
        n_spins: usize,
        couplings: Vec<f64>,
        fields: Vec<f64>,
        pulse_angle: f64,
    ) -> Result<Self, FloquetError> {
        if !(1..=20).contains(&n_spins) {
            return Err(FloquetError::InvalidSpinCount { n_spins });
        }
        let expected_couplings = n_spins - 1;
        if couplings.len() != expected_couplings {
            return Err(FloquetError::CouplingCount {
                expected: expected_couplings,
                actual: couplings.len(),
            });
        }
        let edges = couplings
            .into_iter()
            .enumerate()
            .map(|(index, coupling)| IsingEdge {
                first: index,
                second: index + 1,
                coupling,
            })
            .collect();
        Self::new_graph(n_spins, edges, fields, pulse_angle)
    }

    /// Construct an exact kicked-Ising simulator on an arbitrary simple graph.
    ///
    /// Edge endpoints are zero-based spin indices. Duplicate undirected
    /// edges, self-edges, out-of-range indices and non-finite couplings are
    /// rejected. The initial state is all-up |00...0>.
    pub fn new_graph(
        n_spins: usize,
        edges: Vec<IsingEdge>,
        fields: Vec<f64>,
        pulse_angle: f64,
    ) -> Result<Self, FloquetError> {
        if !(1..=20).contains(&n_spins) {
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
        for (index, value) in fields.iter().enumerate() {
            if !value.is_finite() {
                return Err(FloquetError::NonFiniteParameter {
                    name: "fields",
                    index,
                });
            }
        }

        let mut seen_edges = BTreeSet::new();
        for (edge_index, edge) in edges.iter().enumerate() {
            if edge.first >= n_spins || edge.second >= n_spins || edge.first == edge.second {
                return Err(FloquetError::InvalidEdge {
                    edge_index,
                    first: edge.first,
                    second: edge.second,
                    n_spins,
                });
            }
            if !edge.coupling.is_finite() {
                return Err(FloquetError::NonFiniteParameter {
                    name: "edge_coupling",
                    index: edge_index,
                });
            }
            let canonical = (edge.first.min(edge.second), edge.first.max(edge.second));
            if !seen_edges.insert(canonical) {
                return Err(FloquetError::DuplicateEdge {
                    first: canonical.0,
                    second: canonical.1,
                });
            }
        }

        let dimension = 1_usize
            .checked_shl(n_spins as u32)
            .ok_or(FloquetError::DimensionOverflow)?;
        let mut amplitudes = vec![Complex::new(0.0, 0.0); dimension];
        amplitudes[0] = Complex::new(1.0, 0.0);

        Ok(Self {
            n_spins,
            edges,
            fields,
            pulse_angle,
            amplitudes,
        })
    }

    /// Construct an open-boundary rectangular square-lattice Ising model.
    /// Spin index is y * width + x. Horizontal and vertical couplings may differ.
    pub fn new_square_lattice(
        width: usize,
        height: usize,
        horizontal_coupling: f64,
        vertical_coupling: f64,
        fields: Vec<f64>,
        pulse_angle: f64,
    ) -> Result<Self, FloquetError> {
        let n_spins = width
            .checked_mul(height)
            .ok_or(FloquetError::DimensionOverflow)?;
        if !(1..=20).contains(&n_spins) {
            return Err(FloquetError::InvalidSpinCount { n_spins });
        }
        let mut edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let site = y * width + x;
                if x + 1 < width {
                    edges.push(IsingEdge {
                        first: site,
                        second: site + 1,
                        coupling: horizontal_coupling,
                    });
                }
                if y + 1 < height {
                    edges.push(IsingEdge {
                        first: site,
                        second: site + width,
                        coupling: vertical_coupling,
                    });
                }
            }
        }
        Self::new_graph(n_spins, edges, fields, pulse_angle)
    }

    pub fn n_spins(&self) -> usize {
        self.n_spins
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[IsingEdge] {
        &self.edges
    }

    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    pub fn pulse_angle(&self) -> f64 {
        self.pulse_angle
    }

    /// Read-only amplitudes in computational basis order (bit 0 is spin 0).
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
    pub fn set_initial_state(&mut self, amplitudes: &[Complex<f64>]) -> Result<(), FloquetError> {
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
        let norm_squared: f64 = amplitudes
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .sum();
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

    /// Apply one Floquet period exactly to the state vector.
    pub fn step(&mut self) {
        self.apply_global_x_rotation();
        self.apply_diagonal_ising_phase();
    }

    /// Apply one Floquet period with a sampled uniform longitudinal AC probe.
    ///
    /// The probe is a piecewise-constant longitudinal field kick coupled to
    /// `sum_i Z_i` after the global X rotation:
    ///
    /// `U_n = exp[-i(H_Ising + A cos(phi_n) sum_i Z_i)] exp[-i theta/2 sum_i X_i]`.
    ///
    /// This is an exact state-vector update for the declared sampled Floquet
    /// map. It is not a continuous-time simulation of a dipolar diamond sensor.
    pub fn step_with_uniform_ac_probe(
        &mut self,
        amplitude: f64,
        phase_rad: f64,
    ) -> Result<(), FloquetError> {
        if !amplitude.is_finite() {
            return Err(FloquetError::NonFiniteParameter {
                name: "probe_amplitude",
                index: 0,
            });
        }
        if !phase_rad.is_finite() {
            return Err(FloquetError::NonFiniteParameter {
                name: "probe_phase_rad",
                index: 0,
            });
        }
        self.apply_global_x_rotation();
        self.apply_diagonal_ising_phase_with_uniform_probe(amplitude * phase_rad.cos());
        Ok(())
    }

    /// Stroboscopic magnetization under a sampled longitudinal AC probe.
    ///
    /// `frequency_cycles_per_drive_period` is restricted to the Nyquist
    /// interval [0, 0.5] for one probe sample per Floquet period. The probe
    /// phase is sampled at each period midpoint. This is a deterministic
    /// piecewise-constant-kick model, not a calibrated continuous sensor.
    pub fn stroboscopic_magnetization_with_uniform_ac_probe(
        &mut self,
        periods: usize,
        amplitude: f64,
        frequency_cycles_per_drive_period: f64,
        initial_phase_rad: f64,
    ) -> Result<Vec<f64>, FloquetError> {
        if !frequency_cycles_per_drive_period.is_finite()
            || !(0.0..=0.5).contains(&frequency_cycles_per_drive_period)
        {
            return Err(FloquetError::InvalidProbeFrequency);
        }
        if !amplitude.is_finite() {
            return Err(FloquetError::NonFiniteParameter {
                name: "probe_amplitude",
                index: 0,
            });
        }
        if !initial_phase_rad.is_finite() {
            return Err(FloquetError::NonFiniteParameter {
                name: "probe_initial_phase_rad",
                index: 0,
            });
        }

        let mut signal = Vec::with_capacity(periods + 1);
        signal.push(self.magnetization());
        for period in 0..periods {
            let midpoint = period as f64 + 0.5;
            let phase = 2.0 * std::f64::consts::PI * frequency_cycles_per_drive_period * midpoint
                + initial_phase_rad;
            self.step_with_uniform_ac_probe(amplitude, phase)?;
            signal.push(self.magnetization());
        }
        Ok(signal)
    }

    /// Norm squared; unitary evolution must preserve this at numerical precision.
    pub fn norm_squared(&self) -> f64 {
        self.amplitudes.iter().map(|amp| amp.norm_sqr()).sum()
    }

    /// Mean z magnetization, normalized to [-1, 1].
    pub fn magnetization(&self) -> f64 {
        let mut expectation = 0.0;
        for (basis, amplitude) in self.amplitudes.iter().enumerate() {
            let probability = amplitude.norm_sqr();
            let mut magnet = 0.0;
            for spin in 0..self.n_spins {
                let z = if (basis & (1 << spin)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                magnet += z;
            }
            expectation += probability * magnet / self.n_spins as f64;
        }
        expectation
    }

    /// Stroboscopic magnetization including t=0 and one sample after each
    /// subsequent Floquet period. The returned length is periods + 1.
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

        // Every pair differs only by the selected spin bit. The two-state
        // update is the exact matrix [[cos,-i sin],[-i sin,cos]].
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

    fn apply_diagonal_ising_phase(&mut self) {
        self.apply_diagonal_ising_phase_with_uniform_probe(0.0);
    }

    fn apply_diagonal_ising_phase_with_uniform_probe(&mut self, uniform_probe_field: f64) {
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            let mut energy = 0.0;
            for spin in 0..self.n_spins {
                let z = if (basis & (1 << spin)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                energy += (self.fields[spin] + uniform_probe_field) * z;
            }
            for edge in &self.edges {
                let z_first = if (basis & (1 << edge.first)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                let z_second = if (basis & (1 << edge.second)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                energy += edge.coupling * z_first * z_second;
            }
            *amplitude *= Complex::from_polar(1.0, -energy);
        }
    }
}

/// Finite-window period-doubling statistics for a stroboscopic signal.
///
/// These are signal diagnostics only. A large subharmonic response is not
/// proof of a discrete time-crystalline phase: an ideal pi pulse gives an
/// exact two-cycle even at zero interaction. Physical phase claims require
/// interaction dependence, phase rigidity, finite-size and lifetime studies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubharmonicResponse {
    /// Mean of (-1)^n m_n over the selected measurement window.
    pub mean_demodulated_order: f64,
    /// Mean absolute demodulated order; distinguishes cancellation from a
    /// stable signed order parameter.
    pub mean_absolute_order: f64,
    /// |mean_demodulated_order| / mean_absolute_order, in [0, 1].
    /// This is a finite-window sign-coherence statistic, not a DTC phase
    /// rigidity proof.
    pub sign_coherence: f64,
    /// Number of samples used.
    pub samples: usize,
}

/// Analyze a stroboscopic magnetization trace beginning at sample index start.
///
/// Sample zero is the initial state and sample n is after n Floquet periods.
/// None is returned for an invalid start, fewer than two samples or any
/// non-finite value.
pub fn analyze_subharmonic_response(signal: &[f64], start: usize) -> Option<SubharmonicResponse> {
    if start >= signal.len() || signal.len() - start < 2 {
        return None;
    }
    if signal.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let mut signed_sum = 0.0;
    let mut absolute_sum = 0.0;
    let mut samples = 0_usize;
    for (period, magnetization) in signal.iter().enumerate().skip(start) {
        let demodulated = if period % 2 == 0 {
            *magnetization
        } else {
            -*magnetization
        };
        signed_sum += demodulated;
        absolute_sum += demodulated.abs();
        samples += 1;
    }

    let mean_demodulated_order = signed_sum / samples as f64;
    let mean_absolute_order = absolute_sum / samples as f64;
    let sign_coherence = if mean_absolute_order > f64::EPSILON {
        (mean_demodulated_order.abs() / mean_absolute_order).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Some(SubharmonicResponse {
        mean_demodulated_order,
        mean_absolute_order,
        sign_coherence,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn arbitrary_complex_initial_state_is_normalized_and_evolves_unitarily() {
        let mut model =
            KickedIsingFloquet::new(2, vec![0.31], vec![0.11, -0.23], PI + 0.08).unwrap();
        let amplitudes = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(-0.5, 0.25),
            Complex::new(0.0, 1.5),
        ];
        model.set_initial_state(&amplitudes).unwrap();
        assert!((model.norm_squared() - 1.0).abs() < 1.0e-12);
        for _ in 0..24 {
            model.step();
            assert!((model.norm_squared() - 1.0).abs() < 1.0e-11);
        }
    }

    #[test]
    fn invalid_arbitrary_initial_state_is_rejected_without_mutation() {
        let mut model = KickedIsingFloquet::new(2, vec![0.2], vec![0.0; 2], PI).unwrap();
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
        non_finite[2] = Complex::new(f64::NAN, 0.0);
        assert_eq!(
            model.set_initial_state(&non_finite),
            Err(FloquetError::NonFiniteAmplitude { index: 2 })
        );
        assert_eq!(model.amplitudes(), original.as_slice());
    }

    #[test]
    fn exact_evolution_preserves_norm() {
        let mut model = KickedIsingFloquet::new(
            6,
            vec![0.31, -0.27, 0.42, 0.18, -0.36],
            vec![0.11, -0.23, 0.07, 0.19, -0.13, 0.29],
            PI + 0.08,
        )
        .unwrap();

        for _ in 0..32 {
            model.step();
            assert!((model.norm_squared() - 1.0).abs() < 1e-11);
        }
    }

    #[test]
    fn ideal_pi_pulse_is_a_period_two_reference_even_without_interactions() {
        let mut model = KickedIsingFloquet::new(4, vec![0.0; 3], vec![0.0; 4], PI).unwrap();
        let signal = model.stroboscopic_magnetization(8);
        for (step, magnetization) in signal.iter().enumerate() {
            let expected = if step % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                (magnetization - expected).abs() < 1e-12,
                "step={step}, magnetization={magnetization}"
            );
        }
    }

    #[test]
    fn two_spin_step_matches_closed_form_amplitudes() {
        let coupling = 0.37;
        let h0 = 0.11;
        let h1 = -0.23;
        let theta: f64 = 0.7;
        let mut model = KickedIsingFloquet::new(2, vec![coupling], vec![h0, h1], theta).unwrap();
        model.step();

        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let initial = [
            Complex::new(c * c, 0.0),
            Complex::new(0.0, -c * s),
            Complex::new(0.0, -c * s),
            Complex::new(-s * s, 0.0),
        ];
        // Basis index bit 0 addresses spin 0 and bit 1 addresses spin 1.
        // A set bit has Z=-1, so the |01> and |10> field terms carry
        // -h0+h1 and +h0-h1 respectively.
        let energies = [
            coupling + h0 + h1,
            -coupling - h0 + h1,
            -coupling + h0 - h1,
            coupling - h0 - h1,
        ];
        for basis in 0..4 {
            let expected = initial[basis] * Complex::from_polar(1.0, -energies[basis]);
            let error = (model.amplitudes()[basis] - expected).norm();
            assert!(
                error < 1e-12,
                "basis={basis}, observed={:?}, expected={expected:?}, error={error:e}",
                model.amplitudes()[basis]
            );
        }
    }

    #[test]
    fn single_spin_pulse_error_drifts_without_interaction_locking() {
        let epsilon = 0.12;
        let mut model = KickedIsingFloquet::new(1, vec![], vec![0.0], PI + epsilon).unwrap();
        let signal = model.stroboscopic_magnetization(12);
        let demodulated_first = signal[1] * -1.0;
        let demodulated_last = signal[12];
        assert!(
            (demodulated_first - demodulated_last).abs() > 0.5,
            "negative control unexpectedly phase-locked"
        );
    }

    #[test]
    fn subharmonic_metric_distinguishes_period_two_trace_and_phase_slip() {
        let mut ideal = KickedIsingFloquet::new(4, vec![0.0; 3], vec![0.0; 4], PI).unwrap();
        let ideal_signal = ideal.stroboscopic_magnetization(100);
        let ideal_response = analyze_subharmonic_response(&ideal_signal, 20).unwrap();
        assert!((ideal_response.mean_demodulated_order - 1.0).abs() < 1e-12);
        assert!((ideal_response.sign_coherence - 1.0).abs() < 1e-12);

        let mut unlocked =
            KickedIsingFloquet::new(8, vec![0.0; 7], vec![0.0; 8], PI + 0.1).unwrap();
        let unlocked_signal = unlocked.stroboscopic_magnetization(100);
        let unlocked_response = analyze_subharmonic_response(&unlocked_signal, 20).unwrap();

        let mut interacting =
            KickedIsingFloquet::new(8, vec![0.5; 7], vec![0.0; 8], PI + 0.1).unwrap();
        let interacting_signal = interacting.stroboscopic_magnetization(100);
        let interacting_response = analyze_subharmonic_response(&interacting_signal, 20).unwrap();

        assert!(
            interacting_response.mean_demodulated_order
                > unlocked_response.mean_demodulated_order + 0.5
        );
        assert!(
            interacting_response.mean_absolute_order > unlocked_response.mean_absolute_order + 0.25,
            "interaction should preserve a larger subharmonic envelope: interacting={:.6}, unlocked={:.6}",
            interacting_response.mean_absolute_order,
            unlocked_response.mean_absolute_order
        );
        assert!(
            interacting_response.sign_coherence > unlocked_response.sign_coherence + 0.5,
            "interaction should improve demodulated sign coherence: interacting={:.6}, unlocked={:.6}",
            interacting_response.sign_coherence,
            unlocked_response.sign_coherence
        );
    }

    #[test]
    fn graph_constructor_matches_chain_constructor() {
        let mut chain = KickedIsingFloquet::new(
            4,
            vec![0.3, 0.7, -0.2],
            vec![0.1, -0.2, 0.05, 0.17],
            PI + 0.06,
        )
        .unwrap();
        let mut graph = KickedIsingFloquet::new_graph(
            4,
            vec![
                IsingEdge {
                    first: 0,
                    second: 1,
                    coupling: 0.3,
                },
                IsingEdge {
                    first: 1,
                    second: 2,
                    coupling: 0.7,
                },
                IsingEdge {
                    first: 2,
                    second: 3,
                    coupling: -0.2,
                },
            ],
            vec![0.1, -0.2, 0.05, 0.17],
            PI + 0.06,
        )
        .unwrap();

        chain.step();
        graph.step();
        for (left, right) in chain.amplitudes().iter().zip(graph.amplitudes()) {
            assert!((*left - *right).norm() < 1e-12);
        }
    }

    #[test]
    fn square_lattice_kicked_ising_has_interaction_stabilized_subharmonic_response() {
        let mut noninteracting =
            KickedIsingFloquet::new_square_lattice(2, 2, 0.0, 0.0, vec![0.0; 4], PI + 0.1).unwrap();
        let mut interacting =
            KickedIsingFloquet::new_square_lattice(2, 2, 0.5, 0.5, vec![0.0; 4], PI + 0.1).unwrap();
        assert_eq!(interacting.edge_count(), 4);

        let unlocked_signal = noninteracting.stroboscopic_magnetization(100);
        let locked_signal = interacting.stroboscopic_magnetization(100);
        let unlocked = analyze_subharmonic_response(&unlocked_signal, 20).unwrap();
        let locked = analyze_subharmonic_response(&locked_signal, 20).unwrap();

        assert!(locked.mean_demodulated_order > unlocked.mean_demodulated_order + 0.5);
        assert!(locked.sign_coherence > unlocked.sign_coherence);
    }

    #[test]
    fn graph_constructor_rejects_duplicate_and_invalid_edges() {
        assert!(matches!(
            KickedIsingFloquet::new_graph(
                3,
                vec![
                    IsingEdge {
                        first: 0,
                        second: 1,
                        coupling: 0.5
                    },
                    IsingEdge {
                        first: 1,
                        second: 0,
                        coupling: 0.5
                    },
                ],
                vec![0.0; 3],
                PI,
            ),
            Err(FloquetError::DuplicateEdge { .. })
        ));
        assert!(matches!(
            KickedIsingFloquet::new_graph(
                3,
                vec![IsingEdge {
                    first: 0,
                    second: 3,
                    coupling: 0.5
                }],
                vec![0.0; 3],
                PI,
            ),
            Err(FloquetError::InvalidEdge { .. })
        ));
    }

    #[test]
    fn subharmonic_metric_rejects_invalid_windows_and_non_finite_data() {
        assert!(analyze_subharmonic_response(&[1.0], 0).is_none());
        assert!(analyze_subharmonic_response(&[1.0, -1.0], 2).is_none());
        assert!(analyze_subharmonic_response(&[1.0, f64::NAN], 0).is_none());
    }

    #[test]
    fn invalid_model_dimensions_and_parameters_are_rejected() {
        assert!(matches!(
            KickedIsingFloquet::new(0, vec![], vec![], PI),
            Err(FloquetError::InvalidSpinCount { .. })
        ));
        assert!(matches!(
            KickedIsingFloquet::new(2, vec![], vec![0.0, 0.0], PI),
            Err(FloquetError::CouplingCount { .. })
        ));
        assert!(matches!(
            KickedIsingFloquet::new(1, vec![], vec![f64::NAN], PI),
            Err(FloquetError::NonFiniteParameter { .. })
        ));
    }
}
