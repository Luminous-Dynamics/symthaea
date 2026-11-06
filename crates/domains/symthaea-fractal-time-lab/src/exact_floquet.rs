// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact state-vector reference evolution for a finite kicked-Ising spin chain.
//!
//! One Floquet period applies a global x rotation followed by a diagonal
//! nearest-neighbour Ising/longitudinal-field phase:
//!
//! U_F = exp(-i [sum_i J_i Z_i Z_(i+1) + sum_i h_i Z_i])
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
use std::error::Error;
use std::fmt;

/// Errors constructing an exact finite-dimensional Floquet system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FloquetError {
    InvalidSpinCount { n_spins: usize },
    DimensionOverflow,
    CouplingCount { expected: usize, actual: usize },
    FieldCount { expected: usize, actual: usize },
    NonFiniteParameter { name: &'static str, index: usize },
    InitialBasisOutOfRange { basis: usize, dimension: usize },
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
                write!(f, "basis index {basis} is outside state dimension {dimension}")
            }
        }
    }
}

impl Error for FloquetError {}

/// Exact state-vector simulator for a finite open-boundary kicked-Ising chain.
#[derive(Clone, Debug)]
pub struct KickedIsingFloquet {
    n_spins: usize,
    couplings: Vec<f64>,
    fields: Vec<f64>,
    pulse_angle: f64,
    amplitudes: Vec<Complex<f64>>,
}

impl KickedIsingFloquet {
    /// Construct an all-up product state |00...0>.
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
        for (index, value) in couplings.iter().enumerate() {
            if !value.is_finite() {
                return Err(FloquetError::NonFiniteParameter {
                    name: "couplings",
                    index,
                });
            }
        }
        for (index, value) in fields.iter().enumerate() {
            if !value.is_finite() {
                return Err(FloquetError::NonFiniteParameter {
                    name: "fields",
                    index,
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
            couplings,
            fields,
            pulse_angle,
            amplitudes,
        })
    }

    pub fn n_spins(&self) -> usize {
        self.n_spins
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

    /// Apply one Floquet period exactly to the state vector.
    pub fn step(&mut self) {
        self.apply_global_x_rotation();
        self.apply_diagonal_ising_phase();
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
                let z = if (basis & (1 << spin)) == 0 { 1.0 } else { -1.0 };
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
        for (basis, amplitude) in self.amplitudes.iter_mut().enumerate() {
            let mut energy = 0.0;
            for spin in 0..self.n_spins {
                let z = if (basis & (1 << spin)) == 0 { 1.0 } else { -1.0 };
                energy += self.fields[spin] * z;
                if spin + 1 < self.n_spins {
                    let next_z = if (basis & (1 << (spin + 1))) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    energy += self.couplings[spin] * z * next_z;
                }
            }
            *amplitude *= Complex::from_polar(1.0, -energy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
        let mut model =
            KickedIsingFloquet::new(4, vec![0.0; 3], vec![0.0; 4], PI).unwrap();
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
        let mut model =
            KickedIsingFloquet::new(2, vec![coupling], vec![h0, h1], theta).unwrap();
        model.step();

        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let initial = [
            Complex::new(c * c, 0.0),
            Complex::new(0.0, -c * s),
            Complex::new(0.0, -c * s),
            Complex::new(-s * s, 0.0),
        ];
        let energies = [
            coupling + h0 + h1,
            -coupling + h0 - h1,
            -coupling - h0 + h1,
            coupling - h0 - h1,
        ];
        for basis in 0..4 {
            let expected = initial[basis] * Complex::from_polar(1.0, -energies[basis]);
            assert!((model.amplitudes()[basis] - expected).norm() < 1e-12);
        }
    }

    #[test]
    fn single_spin_pulse_error_drifts_without_interaction_locking() {
        let epsilon = 0.12;
        let mut model =
            KickedIsingFloquet::new(1, vec![], vec![0.0], PI + epsilon).unwrap();
        let signal = model.stroboscopic_magnetization(12);
        let demodulated_first = signal[1] * -1.0;
        let demodulated_last = signal[12];
        assert!(
            (demodulated_first - demodulated_last).abs() > 0.5,
            "negative control unexpectedly phase-locked"
        );
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
