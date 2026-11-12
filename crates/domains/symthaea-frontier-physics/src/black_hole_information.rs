// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-dimensional quantum-information primitives for black-hole information
//! experiments. This is a quantum-information laboratory, not a model of
//! quantum gravity. Qubit 0 is the most-significant bit.
//!
//! Dense work is explicitly bounded. States are not silently renormalized and
//! density matrices are never projected onto the positive-semidefinite cone.
//! For H=A+iB Hermitian, [[A,-B],[B,A]] is a real symmetric embedding whose
//! eigenvalues are the eigenvalues of H, each repeated twice.

use core::fmt;
use num_complex::Complex64;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;

pub const MAX_DENSE_REDUCED_DIMENSION: usize = 64;
/// Match the existing state-vector circuit ceiling; larger states require a tensor-network backend.
pub const MAX_DENSE_STATE_QUBITS: usize = 20;
/// Upper product dimension for the O(m*n) harmonic-number Page oracle.
pub const MAX_PAGE_REFERENCE_TOTAL_DIMENSION: usize = 1 << MAX_DENSE_STATE_QUBITS;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumInformationError {
    InvalidDimension(usize),
    DimensionOverflow,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    DenseDimensionLimit {
        requested: usize,
        maximum: usize,
    },
    DenseStateQubitLimit {
        requested: usize,
        maximum: usize,
    },
    PageReferenceDimensionLimit {
        requested: usize,
        maximum: usize,
    },
    InvalidPageEntropy(f64),
    InvalidMutualInformation {
        value: f64,
        tolerance: f64,
    },
    InvalidSquaredCommutator {
        value: f64,
        tolerance: f64,
    },
    InformationIdentityError {
        residual: f64,
        tolerance: f64,
    },
    InvalidEnsembleSampleCount(usize),
    InvalidCircuitTopology,
    InvalidSubsystemSize {
        requested: usize,
        maximum: usize,
    },
    InvalidExcitationSector {
        excitations: usize,
        number_of_qubits: usize,
    },
    EvaporationRegisterLimit {
        requested: usize,
        maximum: usize,
    },
    UnitaryConstructionFailure,
    InvalidTolerance(f64),
    NonFiniteState {
        index: usize,
    },
    NonFiniteMatrix {
        row: usize,
        column: usize,
    },
    InvalidQubit {
        qubit: usize,
        register_size: usize,
    },
    DuplicateQubit(usize),
    NormalizationError {
        norm_squared: f64,
        tolerance: f64,
    },
    NonHermitian {
        residual: f64,
        tolerance: f64,
    },
    TraceError {
        real: f64,
        imaginary: f64,
        tolerance: f64,
    },
    NonPositive {
        minimum_eigenvalue: f64,
        tolerance: f64,
    },
    EigensolverNonconvergence {
        sweeps: usize,
        residual: f64,
    },
}

impl fmt::Display for QuantumInformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(d) => write!(f, "invalid Hilbert-space dimension {d}"),
            Self::DimensionOverflow => write!(f, "Hilbert-space dimension overflow"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::DenseDimensionLimit { requested, maximum } => {
                write!(f, "dense dimension {requested} exceeds limit {maximum}")
            }
            Self::DenseStateQubitLimit { requested, maximum } => write!(
                f,
                "dense state with {requested} qubits exceeds limit {maximum}"
            ),
            Self::PageReferenceDimensionLimit { requested, maximum } => write!(
                f,
                "Page reference dimension product {requested} exceeds limit {maximum}"
            ),
            Self::InvalidPageEntropy(value) => {
                write!(f, "Page reference entropy is invalid: {value}")
            }
            Self::InvalidMutualInformation { value, tolerance } => write!(
                f,
                "mutual information {value} is below the allowed numerical tolerance {tolerance}"
            ),
            Self::InvalidSquaredCommutator { value, tolerance } => write!(
                f,
                "normalized squared commutator {value} is outside [0,2] beyond tolerance {tolerance}"
            ),
            Self::InformationIdentityError {
                residual,
                tolerance,
            } => write!(
                f,
                "information complementarity residual {residual} exceeds {tolerance}"
            ),
            Self::InvalidEnsembleSampleCount(count) => {
                write!(f, "Haar ensemble needs at least two samples, got {count}")
            }
            Self::InvalidCircuitTopology => {
                write!(f, "circuit topology requires at least two qubits")
            }
            Self::InvalidSubsystemSize { requested, maximum } => write!(
                f,
                "subsystem size {requested} exceeds total register size {maximum}"
            ),
            Self::InvalidExcitationSector {
                excitations,
                number_of_qubits,
            } => write!(
                f,
                "excitation sector {excitations} is invalid for {number_of_qubits} qubits"
            ),
            Self::EvaporationRegisterLimit { requested, maximum } => write!(
                f,
                "sequential evaporation register has {requested} qubits; dense complementary diagnostics support at most {maximum}"
            ),
            Self::UnitaryConstructionFailure => write!(
                f,
                "failed to construct a numerically independent Haar unitary basis"
            ),
            Self::InvalidTolerance(t) => write!(f, "invalid tolerance {t}"),
            Self::NonFiniteState { index } => {
                write!(f, "non-finite state amplitude at index {index}")
            }
            Self::NonFiniteMatrix { row, column } => {
                write!(f, "non-finite matrix entry ({row}, {column})")
            }
            Self::InvalidQubit {
                qubit,
                register_size,
            } => write!(
                f,
                "qubit {qubit} is outside a {register_size}-qubit register"
            ),
            Self::DuplicateQubit(q) => write!(f, "qubit {q} appears more than once"),
            Self::NormalizationError {
                norm_squared,
                tolerance,
            } => write!(
                f,
                "state norm squared {norm_squared} differs from one beyond {tolerance}"
            ),
            Self::NonHermitian {
                residual,
                tolerance,
            } => write!(f, "Hermiticity residual {residual} exceeds {tolerance}"),
            Self::TraceError {
                real,
                imaginary,
                tolerance,
            } => write!(
                f,
                "trace ({real}, {imaginary}) differs from one beyond {tolerance}"
            ),
            Self::NonPositive {
                minimum_eigenvalue,
                tolerance,
            } => write!(
                f,
                "minimum eigenvalue {minimum_eigenvalue} is below -{tolerance}"
            ),
            Self::EigensolverNonconvergence { sweeps, residual } => write!(
                f,
                "Jacobi eigensolver did not converge after {sweeps} sweeps (residual {residual})"
            ),
        }
    }
}
impl std::error::Error for QuantumInformationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct DensityMatrix {
    dimension: usize,
    entries: Vec<Complex64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DensityDiagnostics {
    pub trace: Complex64,
    pub hermiticity_residual: f64,
    pub purity: f64,
    pub eigenvalues: Vec<f64>,
    pub minimum_eigenvalue: f64,
}

impl DensityMatrix {
    /// Construct row-major storage. Physical invariants are checked by
    /// diagnostics and entropy methods; they are never silently repaired.
    pub fn from_entries(
        dimension: usize,
        entries: Vec<Complex64>,
    ) -> Result<Self, QuantumInformationError> {
        if dimension == 0 {
            return Err(QuantumInformationError::InvalidDimension(dimension));
        }
        if dimension > MAX_DENSE_REDUCED_DIMENSION {
            return Err(QuantumInformationError::DenseDimensionLimit {
                requested: dimension,
                maximum: MAX_DENSE_REDUCED_DIMENSION,
            });
        }
        let expected = dimension
            .checked_mul(dimension)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        if entries.len() != expected {
            return Err(QuantumInformationError::DimensionMismatch {
                expected,
                actual: entries.len(),
            });
        }
        for row in 0..dimension {
            for column in 0..dimension {
                let x = entries[row * dimension + column];
                if !x.re.is_finite() || !x.im.is_finite() {
                    return Err(QuantumInformationError::NonFiniteMatrix { row, column });
                }
            }
        }
        Ok(Self { dimension, entries })
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }

    pub fn get(&self, row: usize, column: usize) -> Option<Complex64> {
        if row < self.dimension && column < self.dimension {
            Some(self.entries[row * self.dimension + column])
        } else {
            None
        }
    }

    /// Partial trace of a normalized pure state onto retained qubits.
    /// Output basis order follows retained_qubits; environment bits follow
    /// ascending original qubit order. No state renormalization is applied.
    pub fn from_pure_state(
        amplitudes: &[Complex64],
        number_of_qubits: usize,
        retained_qubits: &[usize],
        normalization_tolerance: f64,
    ) -> Result<Self, QuantumInformationError> {
        validate_tolerance(normalization_tolerance)?;
        if number_of_qubits > MAX_DENSE_STATE_QUBITS {
            return Err(QuantumInformationError::DenseStateQubitLimit {
                requested: number_of_qubits,
                maximum: MAX_DENSE_STATE_QUBITS,
            });
        }
        if number_of_qubits >= usize::BITS as usize {
            return Err(QuantumInformationError::DimensionOverflow);
        }
        let full_dimension = 1usize << number_of_qubits;
        if amplitudes.len() != full_dimension {
            return Err(QuantumInformationError::DimensionMismatch {
                expected: full_dimension,
                actual: amplitudes.len(),
            });
        }
        let mut norm_squared = 0.0;
        for (index, a) in amplitudes.iter().enumerate() {
            if !a.re.is_finite() || !a.im.is_finite() {
                return Err(QuantumInformationError::NonFiniteState { index });
            }
            norm_squared += a.norm_sqr();
        }
        if !norm_squared.is_finite() {
            return Err(QuantumInformationError::NonFiniteState {
                index: amplitudes.len(),
            });
        }
        if (norm_squared - 1.0).abs() > normalization_tolerance {
            return Err(QuantumInformationError::NormalizationError {
                norm_squared,
                tolerance: normalization_tolerance,
            });
        }

        let mut retained = vec![false; number_of_qubits];
        for &qubit in retained_qubits {
            if qubit >= number_of_qubits {
                return Err(QuantumInformationError::InvalidQubit {
                    qubit,
                    register_size: number_of_qubits,
                });
            }
            if retained[qubit] {
                return Err(QuantumInformationError::DuplicateQubit(qubit));
            }
            retained[qubit] = true;
        }
        if retained_qubits.len() >= usize::BITS as usize {
            return Err(QuantumInformationError::DimensionOverflow);
        }
        let reduced_dimension = 1usize << retained_qubits.len();
        if reduced_dimension > MAX_DENSE_REDUCED_DIMENSION {
            return Err(QuantumInformationError::DenseDimensionLimit {
                requested: reduced_dimension,
                maximum: MAX_DENSE_REDUCED_DIMENSION,
            });
        }
        let environment_qubits = number_of_qubits - retained_qubits.len();
        if environment_qubits >= usize::BITS as usize {
            return Err(QuantumInformationError::DimensionOverflow);
        }
        let environment_dimension = 1usize << environment_qubits;

        // Regroup amplitudes as psi[environment, retained], then form
        // rho_A = sum_e |psi_e><psi_e|.
        let mut blocks =
            vec![vec![Complex64::new(0.0, 0.0); reduced_dimension]; environment_dimension];
        for (basis_index, amplitude) in amplitudes.iter().copied().enumerate() {
            let mut a_index = 0usize;
            for &qubit in retained_qubits {
                let bit = (basis_index >> (number_of_qubits - 1 - qubit)) & 1;
                a_index = (a_index << 1) | bit;
            }
            let mut e_index = 0usize;
            for qubit in 0..number_of_qubits {
                if !retained[qubit] {
                    let bit = (basis_index >> (number_of_qubits - 1 - qubit)) & 1;
                    e_index = (e_index << 1) | bit;
                }
            }
            blocks[e_index][a_index] = amplitude;
        }
        let mut entries = vec![Complex64::new(0.0, 0.0); reduced_dimension * reduced_dimension];
        for row in 0..reduced_dimension {
            for column in 0..reduced_dimension {
                let mut value = Complex64::new(0.0, 0.0);
                for block in &blocks {
                    value += block[row] * block[column].conj();
                }
                entries[row * reduced_dimension + column] = value;
            }
        }
        Self::from_entries(reduced_dimension, entries)
    }

    pub fn trace(&self) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension {
            result += self.entries[i * self.dimension + i];
        }
        result
    }

    pub fn hermiticity_residual(&self) -> f64 {
        let mut residual: f64 = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                residual = residual.max(
                    (self.entries[i * self.dimension + j]
                        - self.entries[j * self.dimension + i].conj())
                    .norm(),
                );
            }
        }
        residual
    }

    pub fn spectrum(&self, tolerance: f64) -> Result<Vec<f64>, QuantumInformationError> {
        Ok(self.checked_diagnostics(tolerance)?.eigenvalues)
    }

    pub fn diagnostics(
        &self,
        tolerance: f64,
    ) -> Result<DensityDiagnostics, QuantumInformationError> {
        self.checked_diagnostics(tolerance)
    }

    /// Von Neumann entropy in nats. Eigenvalues in the permitted roundoff
    /// interval [-tolerance, 0] contribute zero; larger negative values reject.
    pub fn von_neumann_entropy(&self, tolerance: f64) -> Result<f64, QuantumInformationError> {
        Ok(self
            .spectrum(tolerance)?
            .into_iter()
            .filter(|p| *p > 0.0)
            .map(|p| -p * p.ln())
            .sum())
    }

    /// Second Renyi entropy in nats.
    pub fn renyi_2_entropy(&self, tolerance: f64) -> Result<f64, QuantumInformationError> {
        let d = self.checked_diagnostics(tolerance)?;
        if d.purity <= 0.0 || !d.purity.is_finite() {
            return Err(QuantumInformationError::NonPositive {
                minimum_eigenvalue: d.minimum_eigenvalue,
                tolerance,
            });
        }
        Ok(-d.purity.ln())
    }

    /// Third Renyi entropy in nats.
    pub fn renyi_3_entropy(&self, tolerance: f64) -> Result<f64, QuantumInformationError> {
        self.checked_diagnostics(tolerance)?;
        let n = self.dimension;
        let mut square = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    square[i * n + j] += self.entries[i * n + k] * self.entries[k * n + j];
                }
            }
        }
        let mut trace_cube = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                trace_cube += square[i * n + j] * self.entries[j * n + i];
            }
        }
        if !trace_cube.re.is_finite()
            || !trace_cube.im.is_finite()
            || trace_cube.re <= 0.0
            || trace_cube.im.abs() > tolerance
        {
            return Err(QuantumInformationError::TraceError {
                real: trace_cube.re,
                imaginary: trace_cube.im,
                tolerance,
            });
        }
        Ok(-0.5 * trace_cube.re.ln())
    }

    fn checked_diagnostics(
        &self,
        tolerance: f64,
    ) -> Result<DensityDiagnostics, QuantumInformationError> {
        validate_tolerance(tolerance)?;
        let trace = self.trace();
        if (trace.re - 1.0).abs() > tolerance || trace.im.abs() > tolerance {
            return Err(QuantumInformationError::TraceError {
                real: trace.re,
                imaginary: trace.im,
                tolerance,
            });
        }
        let hermiticity_residual = self.hermiticity_residual();
        if hermiticity_residual > tolerance {
            return Err(QuantumInformationError::NonHermitian {
                residual: hermiticity_residual,
                tolerance,
            });
        }
        let eigenvalues = hermitian_eigenvalues(self, (tolerance * 0.1).max(1e-14))?;
        let minimum_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if minimum_eigenvalue < -tolerance {
            return Err(QuantumInformationError::NonPositive {
                minimum_eigenvalue,
                tolerance,
            });
        }
        let purity: f64 = self.entries.iter().map(|x| x.norm_sqr()).sum();
        if !purity.is_finite() {
            return Err(QuantumInformationError::NonFiniteMatrix { row: 0, column: 0 });
        }
        Ok(DensityDiagnostics {
            trace,
            hermiticity_residual,
            purity,
            eigenvalues,
            minimum_eigenvalue,
        })
    }
}

/// Exact Page theorem benchmark for a Haar-random pure state on
/// C^m tensor C^n, returned in nats.
///
/// E[S_A] = H_(m*n) - H_max(m,n) - (min(m,n)-1)/(2*max(m,n)).
///
/// This is an ensemble-average reference for finite-dimensional random pure
/// states. It is not an evaporation Hamiltonian, a local-circuit result, or a
/// gravitational path-integral calculation. The bounded O(m*n) harmonic sum
/// makes the numerical cost explicit and deterministic.
pub fn page_average_entropy_nats(
    subsystem_a_dimension: usize,
    subsystem_b_dimension: usize,
) -> Result<f64, QuantumInformationError> {
    if subsystem_a_dimension == 0 {
        return Err(QuantumInformationError::InvalidDimension(0));
    }
    if subsystem_b_dimension == 0 {
        return Err(QuantumInformationError::InvalidDimension(0));
    }
    let total_dimension = subsystem_a_dimension
        .checked_mul(subsystem_b_dimension)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    if total_dimension > MAX_PAGE_REFERENCE_TOTAL_DIMENSION {
        return Err(QuantumInformationError::PageReferenceDimensionLimit {
            requested: total_dimension,
            maximum: MAX_PAGE_REFERENCE_TOTAL_DIMENSION,
        });
    }
    let smaller = subsystem_a_dimension.min(subsystem_b_dimension);
    let larger = subsystem_a_dimension.max(subsystem_b_dimension);
    let entropy = harmonic_number(total_dimension)
        - harmonic_number(larger)
        - (smaller - 1) as f64 / (2.0 * larger as f64);
    if entropy.is_finite() && entropy >= 0.0 {
        Ok(entropy)
    } else {
        Err(QuantumInformationError::InvalidPageEntropy(entropy))
    }
}

/// Exact Haar-average entropy for a pure state uniformly sampled inside a
/// fixed total-excitation (U(1) charge) sector.
///
/// For N qubits, total excitation K, and a subsystem of A qubits, the
/// fixed-charge Hilbert space decomposes into blocks
/// H_A(q) tensor H_B(K-q), with dimensions
/// m_q = C(A,q), n_q = C(N-A,K-q). The reduced state is block diagonal;
/// its block weights follow a Dirichlet distribution with parameters
/// d_q = m_q*n_q. Thus
///
/// E[S_A] = H_D - sum_q (d_q/D) H_(d_q)
///          + sum_q (d_q/D) Page(m_q,n_q),
///
/// where D=C(N,K), H_j is the j-th harmonic number, and Page(m,n) is the
/// finite-dimensional complex-Haar Page average. This is the symmetry-sector
/// equilibrium oracle, not the unconstrained Page theorem and not a claim
/// that a particular local circuit has mixed to that equilibrium.
pub fn symmetry_resolved_page_average_entropy_nats(
    number_of_qubits: usize,
    total_excitations: usize,
    subsystem_a_qubits: usize,
) -> Result<f64, QuantumInformationError> {
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    if subsystem_a_qubits > number_of_qubits {
        return Err(QuantumInformationError::InvalidSubsystemSize {
            requested: subsystem_a_qubits,
            maximum: number_of_qubits,
        });
    }
    if total_excitations > number_of_qubits {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: total_excitations,
            number_of_qubits,
        });
    }

    let subsystem_b_qubits = number_of_qubits - subsystem_a_qubits;
    let total_dimension = binomial_dimension(number_of_qubits, total_excitations)?;
    if total_dimension > MAX_PAGE_REFERENCE_TOTAL_DIMENSION {
        return Err(QuantumInformationError::PageReferenceDimensionLimit {
            requested: total_dimension,
            maximum: MAX_PAGE_REFERENCE_TOTAL_DIMENSION,
        });
    }

    let minimum_a_charge = total_excitations.saturating_sub(subsystem_b_qubits);
    let maximum_a_charge = total_excitations.min(subsystem_a_qubits);
    let mut block_dimension_sum = 0usize;
    let mut weighted_block_harmonic = 0.0;
    let mut weighted_page_entropy = 0.0;

    for a_charge in minimum_a_charge..=maximum_a_charge {
        let dimension_a = binomial_dimension(subsystem_a_qubits, a_charge)?;
        let dimension_b = binomial_dimension(subsystem_b_qubits, total_excitations - a_charge)?;
        let block_dimension = dimension_a
            .checked_mul(dimension_b)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        block_dimension_sum = block_dimension_sum
            .checked_add(block_dimension)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        let weight = block_dimension as f64 / total_dimension as f64;
        weighted_block_harmonic += weight * harmonic_number(block_dimension);
        weighted_page_entropy += weight * page_average_entropy_nats(dimension_a, dimension_b)?;
    }

    if block_dimension_sum != total_dimension {
        return Err(QuantumInformationError::DimensionMismatch {
            expected: total_dimension,
            actual: block_dimension_sum,
        });
    }

    let entropy =
        harmonic_number(total_dimension) - weighted_block_harmonic + weighted_page_entropy;
    if entropy.is_finite() && entropy >= 0.0 {
        Ok(entropy)
    } else {
        Err(QuantumInformationError::InvalidPageEntropy(entropy))
    }
}

/// Exact integer binomial coefficient with checked intermediate arithmetic.
fn binomial_dimension(n: usize, k: usize) -> Result<usize, QuantumInformationError> {
    if k > n {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: k,
            number_of_qubits: n,
        });
    }
    let k = k.min(n - k);
    let mut value = 1usize;
    for index in 1..=k {
        value = value
            .checked_mul(n - k + index)
            .ok_or(QuantumInformationError::DimensionOverflow)?
            / index;
    }
    Ok(value)
}

/// Topology for finite-depth random two-qubit scrambling circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomCircuitTopology {
    /// Nearest-neighbour 1D brickwork; even and odd bonds alternate.
    Brickwork1D,
    /// A fresh random disjoint matching of qubit pairs at every layer.
    AllToAllMatching,
}

/// One depth's reproducible random-circuit observables.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomCircuitEntropySample {
    pub number_of_qubits: usize,
    pub depth: usize,
    pub seed: u64,
    pub topology: RandomCircuitTopology,
    pub two_qubit_gate_count: usize,
    pub state_norm_squared: f64,
    pub subsystem_entropy_nats: f64,
    pub subsystem_purity: f64,
}

/// Ensemble statistics over independent circuit seeds at one declared depth.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomCircuitEntropySummary {
    pub number_of_qubits: usize,
    pub depth: usize,
    pub topology: RandomCircuitTopology,
    pub sample_count: usize,
    pub first_seed: u64,
    pub mean_entropy_nats: f64,
    pub sample_variance_nats_squared: f64,
    pub standard_error_nats: f64,
}

/// Generate a Haar-distributed 4x4 complex unitary by Gram-Schmidt
/// orthonormalization of iid complex Gaussian columns (Ginibre construction).
fn haar_unitary_4(rng: &mut ChaCha20Rng) -> Result<[[Complex64; 4]; 4], QuantumInformationError> {
    let mut columns = [[Complex64::new(0.0, 0.0); 4]; 4];
    for column in 0..4 {
        let mut accepted = false;
        for _attempt in 0..8 {
            let mut vector = [Complex64::new(0.0, 0.0); 4];
            for entry in &mut vector {
                let u1 = rng.gen_range(0.0_f64..1.0_f64).max(f64::MIN_POSITIVE);
                let u2 = rng.gen_range(0.0_f64..1.0_f64);
                let radius = (-2.0 * u1.ln()).sqrt();
                let phase = core::f64::consts::TAU * u2;
                *entry = Complex64::new(radius * phase.cos(), radius * phase.sin());
            }
            for previous in 0..column {
                let mut overlap = Complex64::new(0.0, 0.0);
                for row in 0..4 {
                    overlap += columns[previous][row].conj() * vector[row];
                }
                for row in 0..4 {
                    vector[row] -= columns[previous][row] * overlap;
                }
            }
            let norm_squared: f64 = vector.iter().map(|value| value.norm_sqr()).sum();
            if norm_squared.is_finite() && norm_squared > 1e-24 {
                let inverse_norm = norm_squared.sqrt().recip();
                for row in 0..4 {
                    columns[column][row] = vector[row] * inverse_norm;
                }
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(QuantumInformationError::UnitaryConstructionFailure);
        }
    }
    let mut unitary = [[Complex64::new(0.0, 0.0); 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            unitary[row][column] = columns[column][row];
        }
    }
    Ok(unitary)
}

/// Apply a two-qubit gate in basis order |00>, |01>, |10>, |11>.
/// The register convention is MSB-first, matching DensityMatrix partial trace.
fn apply_two_qubit_unitary(
    amplitudes: &mut [Complex64],
    number_of_qubits: usize,
    first_qubit: usize,
    second_qubit: usize,
    unitary: &[[Complex64; 4]; 4],
) -> Result<(), QuantumInformationError> {
    if first_qubit >= number_of_qubits
        || second_qubit >= number_of_qubits
        || first_qubit == second_qubit
    {
        return Err(QuantumInformationError::InvalidQubit {
            qubit: first_qubit.max(second_qubit),
            register_size: number_of_qubits,
        });
    }
    let expected = 1usize
        .checked_shl(number_of_qubits as u32)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    if amplitudes.len() != expected {
        return Err(QuantumInformationError::DimensionMismatch {
            expected,
            actual: amplitudes.len(),
        });
    }
    let first_bit = number_of_qubits - 1 - first_qubit;
    let second_bit = number_of_qubits - 1 - second_qubit;
    let first_mask = 1usize << first_bit;
    let second_mask = 1usize << second_bit;
    for base in 0..expected {
        if base & first_mask != 0 || base & second_mask != 0 {
            continue;
        }
        let indices = [
            base,
            base | second_mask,
            base | first_mask,
            base | first_mask | second_mask,
        ];
        let input = [
            amplitudes[indices[0]],
            amplitudes[indices[1]],
            amplitudes[indices[2]],
            amplitudes[indices[3]],
        ];
        for row in 0..4 {
            let mut value = Complex64::new(0.0, 0.0);
            for column in 0..4 {
                value += unitary[row][column] * input[column];
            }
            amplitudes[indices[row]] = value;
        }
    }
    Ok(())
}

/// Simulate one seeded random-circuit sample and return subsystem entropy.
/// This is a finite-depth scrambling experiment, not an evaporation model.
pub fn simulate_random_circuit_entropy(
    number_of_qubits: usize,
    depth: usize,
    topology: RandomCircuitTopology,
    retained_qubits: &[usize],
    seed: u64,
    tolerance: f64,
) -> Result<RandomCircuitEntropySample, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if number_of_qubits < 2 {
        return Err(QuantumInformationError::InvalidCircuitTopology);
    }
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    let dimension = 1usize << number_of_qubits;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
    amplitudes[0] = Complex64::new(1.0, 0.0);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut gate_count = 0usize;

    for layer in 0..depth {
        let pairs = match topology {
            RandomCircuitTopology::Brickwork1D => {
                let parity = layer % 2;
                (parity..(number_of_qubits - 1))
                    .step_by(2)
                    .map(|first| (first, first + 1))
                    .collect::<Vec<_>>()
            }
            RandomCircuitTopology::AllToAllMatching => {
                let mut qubits = (0..number_of_qubits).collect::<Vec<_>>();
                qubits.shuffle(&mut rng);
                qubits
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect::<Vec<_>>()
            }
        };
        for (first, second) in pairs {
            let unitary = haar_unitary_4(&mut rng)?;
            apply_two_qubit_unitary(&mut amplitudes, number_of_qubits, first, second, &unitary)?;
            gate_count += 1;
        }
    }

    let norm_squared: f64 = amplitudes
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .sum();
    if !norm_squared.is_finite() {
        return Err(QuantumInformationError::NonFiniteState { index: dimension });
    }
    if (norm_squared - 1.0).abs() > tolerance {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared,
            tolerance,
        });
    }
    let reduced =
        DensityMatrix::from_pure_state(&amplitudes, number_of_qubits, retained_qubits, tolerance)?;
    let entropy = reduced.von_neumann_entropy(tolerance)?;
    let purity = reduced.diagnostics(tolerance)?.purity;
    Ok(RandomCircuitEntropySample {
        number_of_qubits,
        depth,
        seed,
        topology,
        two_qubit_gate_count: gate_count,
        state_norm_squared: norm_squared,
        subsystem_entropy_nats: entropy,
        subsystem_purity: purity,
    })
}

/// Run an independent-seed finite-depth circuit ensemble at a fixed topology
/// and depth. This reports sampling variance; it does not assume Haar equilibrium.
pub fn sample_random_circuit_entropy_ensemble(
    number_of_qubits: usize,
    depth: usize,
    topology: RandomCircuitTopology,
    retained_qubits: &[usize],
    sample_count: usize,
    first_seed: u64,
    tolerance: f64,
) -> Result<RandomCircuitEntropySummary, QuantumInformationError> {
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    for sample_index in 0..sample_count {
        let seed = first_seed.wrapping_add(sample_index as u64);
        let sample = simulate_random_circuit_entropy(
            number_of_qubits,
            depth,
            topology,
            retained_qubits,
            seed,
            tolerance,
        )?;
        let delta = sample.subsystem_entropy_nats - mean;
        mean += delta / (sample_index + 1) as f64;
        sum_squared_deviations += delta * (sample.subsystem_entropy_nats - mean);
    }
    let variance = sum_squared_deviations / (sample_count - 1) as f64;
    Ok(RandomCircuitEntropySummary {
        number_of_qubits,
        depth,
        topology,
        sample_count,
        first_seed,
        mean_entropy_nats: mean,
        sample_variance_nats_squared: variance,
        standard_error_nats: (variance / sample_count as f64).sqrt(),
    })
}

/// One sample of a finite-depth U(1)-conserving circuit started from a
/// fixed-charge computational-basis product state.
#[derive(Debug, Clone, PartialEq)]
pub struct U1RandomCircuitEntropySample {
    pub number_of_qubits: usize,
    pub total_excitations: usize,
    pub subsystem_a_qubits: usize,
    pub depth: usize,
    pub seed: u64,
    pub topology: RandomCircuitTopology,
    pub two_qubit_gate_count: usize,
    pub state_norm_squared: f64,
    pub total_excitation_expectation: f64,
    pub total_excitation_variance: f64,
    pub subsystem_entropy_nats: f64,
    pub subsystem_purity: f64,
    pub symmetry_resolved_page_mean_nats: f64,
    /// Positive means the finite-depth sample is below its symmetry-sector
    /// Haar equilibrium mean; this is not a mixing-time estimate by itself.
    pub entropy_gap_to_equilibrium_nats: f64,
}

/// Independent-seed statistics for one fixed U(1)-conserving circuit depth.
#[derive(Debug, Clone, PartialEq)]
pub struct U1RandomCircuitEntropySummary {
    pub number_of_qubits: usize,
    pub total_excitations: usize,
    pub subsystem_a_qubits: usize,
    pub depth: usize,
    pub topology: RandomCircuitTopology,
    pub sample_count: usize,
    pub first_seed: u64,
    pub mean_entropy_nats: f64,
    pub sample_variance_nats_squared: f64,
    pub standard_error_nats: f64,
    pub symmetry_resolved_page_mean_nats: f64,
    pub mean_entropy_gap_to_equilibrium_nats: f64,
}

/// Evolve a low-entanglement product state with a finite-depth random circuit
/// whose two-qubit gates exactly preserve the equal-gap excitation number.
///
/// The initial state is the computational-basis state with the first K qubits
/// occupied and the rest empty. Each gate is Haar-random in U(1) x U(2) x U(1),
/// embedded block-diagonally in {|00>,|01>,|10>,|11>}. Brickwork uses alternating
/// nearest-neighbour bonds; all-to-all uses a fresh random disjoint matching per
/// layer. The equilibrium reference is the exact Haar average restricted to
/// the same (N,K) sector, not the unconstrained Page theorem.
///
/// This is a unitary scrambling benchmark, not a sequential-emission protocol,
/// Hawking radiation model, or gravitational evaporation calculation.
pub fn simulate_u1_conserving_random_circuit_entropy(
    number_of_qubits: usize,
    total_excitations: usize,
    depth: usize,
    topology: RandomCircuitTopology,
    retained_qubits: &[usize],
    seed: u64,
    tolerance: f64,
) -> Result<U1RandomCircuitEntropySample, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if number_of_qubits < 2 {
        return Err(QuantumInformationError::InvalidCircuitTopology);
    }
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    if total_excitations > number_of_qubits {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: total_excitations,
            number_of_qubits,
        });
    }

    let state_dimension = 1usize << number_of_qubits;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); state_dimension];
    let mut initial_basis = 0usize;
    for qubit in 0..total_excitations {
        initial_basis |= 1usize << (number_of_qubits - 1 - qubit);
    }
    amplitudes[initial_basis] = Complex64::new(1.0, 0.0);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut gate_count = 0usize;
    for layer in 0..depth {
        let pairs = match topology {
            RandomCircuitTopology::Brickwork1D => {
                let parity = layer % 2;
                (parity..(number_of_qubits - 1))
                    .step_by(2)
                    .map(|first| (first, first + 1))
                    .collect::<Vec<_>>()
            }
            RandomCircuitTopology::AllToAllMatching => {
                let mut qubits = (0..number_of_qubits).collect::<Vec<_>>();
                qubits.shuffle(&mut rng);
                qubits
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect::<Vec<_>>()
            }
        };
        for (first, second) in pairs {
            let unitary = u1_conserving_unitary_4(&mut rng)?;
            apply_two_qubit_unitary(&mut amplitudes, number_of_qubits, first, second, &unitary)?;
            gate_count += 1;
        }
    }

    let norm_squared: f64 = amplitudes
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .sum();
    if !norm_squared.is_finite() {
        return Err(QuantumInformationError::NonFiniteState {
            index: state_dimension,
        });
    }
    if (norm_squared - 1.0).abs() > tolerance {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared,
            tolerance,
        });
    }
    let (charge_expectation, charge_variance) = total_excitation_moments(&amplitudes);
    if (charge_expectation - total_excitations as f64).abs() > tolerance
        || charge_variance > tolerance
    {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared: charge_expectation,
            tolerance,
        });
    }

    let reduced =
        DensityMatrix::from_pure_state(&amplitudes, number_of_qubits, retained_qubits, tolerance)?;
    let entropy = reduced.von_neumann_entropy(tolerance)?;
    let purity = reduced.diagnostics(tolerance)?.purity;
    let page_mean = symmetry_resolved_page_average_entropy_nats(
        number_of_qubits,
        total_excitations,
        retained_qubits.len(),
    )?;
    Ok(U1RandomCircuitEntropySample {
        number_of_qubits,
        total_excitations,
        subsystem_a_qubits: retained_qubits.len(),
        depth,
        seed,
        topology,
        two_qubit_gate_count: gate_count,
        state_norm_squared: norm_squared,
        total_excitation_expectation: charge_expectation,
        total_excitation_variance: charge_variance,
        subsystem_entropy_nats: entropy,
        subsystem_purity: purity,
        symmetry_resolved_page_mean_nats: page_mean,
        entropy_gap_to_equilibrium_nats: page_mean - entropy,
    })
}

/// Run independent fixed-charge circuit realizations at one declared depth.
pub fn sample_u1_conserving_random_circuit_entropy_ensemble(
    number_of_qubits: usize,
    total_excitations: usize,
    depth: usize,
    topology: RandomCircuitTopology,
    retained_qubits: &[usize],
    sample_count: usize,
    first_seed: u64,
    tolerance: f64,
) -> Result<U1RandomCircuitEntropySummary, QuantumInformationError> {
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    let mut page_mean = None;
    for sample_index in 0..sample_count {
        let seed = first_seed.wrapping_add(sample_index as u64);
        let sample = simulate_u1_conserving_random_circuit_entropy(
            number_of_qubits,
            total_excitations,
            depth,
            topology,
            retained_qubits,
            seed,
            tolerance,
        )?;
        let delta = sample.subsystem_entropy_nats - mean;
        mean += delta / (sample_index + 1) as f64;
        sum_squared_deviations += delta * (sample.subsystem_entropy_nats - mean);
        page_mean = Some(sample.symmetry_resolved_page_mean_nats);
    }
    let variance = sum_squared_deviations / (sample_count - 1) as f64;
    let page_mean = page_mean.unwrap_or(0.0);
    Ok(U1RandomCircuitEntropySummary {
        number_of_qubits,
        total_excitations,
        subsystem_a_qubits: retained_qubits.len(),
        depth,
        topology,
        sample_count,
        first_seed,
        mean_entropy_nats: mean,
        sample_variance_nats_squared: variance,
        standard_error_nats: (variance / sample_count as f64).sqrt(),
        symmetry_resolved_page_mean_nats: page_mean,
        mean_entropy_gap_to_equilibrium_nats: page_mean - mean,
    })
}

#[cfg(test)]
mod u1_random_circuit_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn depth_zero_product_state_has_zero_entropy_and_declared_charge() {
        let sample = simulate_u1_conserving_random_circuit_entropy(
            8,
            4,
            0,
            RandomCircuitTopology::Brickwork1D,
            &[0, 1, 2, 3],
            0xD3F7,
            TOLERANCE,
        )
        .unwrap();
        assert_eq!(sample.two_qubit_gate_count, 0);
        assert!(sample.subsystem_entropy_nats.abs() < TOLERANCE);
        assert!((sample.total_excitation_expectation - 4.0).abs() < TOLERANCE);
        assert!(sample.total_excitation_variance < TOLERANCE);
        assert!((sample.state_norm_squared - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn fixed_charge_brickwork_and_all_to_all_preserve_charge_and_norm() {
        for topology in [
            RandomCircuitTopology::Brickwork1D,
            RandomCircuitTopology::AllToAllMatching,
        ] {
            let sample = simulate_u1_conserving_random_circuit_entropy(
                8,
                4,
                12,
                topology,
                &[0, 1, 2, 3],
                0xB1A5 + topology as u64,
                TOLERANCE,
            )
            .unwrap();
            assert!((sample.state_norm_squared - 1.0).abs() < TOLERANCE);
            assert!((sample.total_excitation_expectation - 4.0).abs() < TOLERANCE);
            assert!(sample.total_excitation_variance < TOLERANCE);
            assert!(sample.subsystem_entropy_nats >= -TOLERANCE);
            let maximal_bipartite_entropy = ((1usize << 4).min(1usize << 4) as f64).ln();
            assert!(sample.subsystem_entropy_nats <= maximal_bipartite_entropy + 1e-9);
        }
    }

    #[test]
    fn fixed_charge_circuit_sample_is_reproducible() {
        let first = simulate_u1_conserving_random_circuit_entropy(
            8,
            4,
            8,
            RandomCircuitTopology::AllToAllMatching,
            &[0, 1, 2, 3],
            0xA110,
            TOLERANCE,
        )
        .unwrap();
        let repeated = simulate_u1_conserving_random_circuit_entropy(
            8,
            4,
            8,
            RandomCircuitTopology::AllToAllMatching,
            &[0, 1, 2, 3],
            0xA110,
            TOLERANCE,
        )
        .unwrap();
        assert_eq!(first, repeated);
    }

    #[test]
    fn fixed_charge_circuit_ensemble_reports_uncertainty_and_correct_equilibrium() {
        let summary = sample_u1_conserving_random_circuit_entropy_ensemble(
            8,
            4,
            8,
            RandomCircuitTopology::Brickwork1D,
            &[0, 1, 2, 3],
            12,
            0xC1AC,
            TOLERANCE,
        )
        .unwrap();
        assert_eq!(summary.sample_count, 12);
        assert!(summary.sample_variance_nats_squared >= 0.0);
        assert!(summary.standard_error_nats >= 0.0);
        assert!(
            (summary.mean_entropy_gap_to_equilibrium_nats
                - (summary.symmetry_resolved_page_mean_nats - summary.mean_entropy_nats))
                .abs()
                < TOLERANCE
        );
    }

    #[test]
    fn fixed_charge_circuit_rejects_invalid_sector_and_empty_subsystem_overflow() {
        assert!(matches!(
            simulate_u1_conserving_random_circuit_entropy(
                8,
                9,
                2,
                RandomCircuitTopology::Brickwork1D,
                &[0, 1],
                1,
                TOLERANCE,
            ),
            Err(QuantumInformationError::InvalidExcitationSector { .. })
        ));
        assert!(matches!(
            simulate_u1_conserving_random_circuit_entropy(
                8,
                4,
                2,
                RandomCircuitTopology::Brickwork1D,
                &[0, 1, 2, 3, 4, 5, 6],
                1,
                TOLERANCE,
            ),
            Err(QuantumInformationError::DenseDimensionLimit { .. })
        ));
    }
}

/// Summary of a seeded Haar-random pure-state Page-entropy ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct HaarPageEnsembleSummary {
    pub subsystem_a_dimension: usize,
    pub subsystem_b_dimension: usize,
    pub sample_count: usize,
    pub seed: u64,
    pub mean_von_neumann_entropy_nats: f64,
    pub sample_variance_nats_squared: f64,
    pub standard_error_nats: f64,
    /// Normal-approximation interval: mean plus/minus 1.96 standard errors.
    pub confidence_95_approx_nats: [f64; 2],
    pub page_theorem_mean_nats: f64,
    pub normalized_mean_deviation: f64,
}

/// Sample Haar-distributed bipartite pure states using normalized iid complex
/// Gaussian amplitudes and a seeded ChaCha20 stream, then measure S(rho_A).
///
/// The requested subsystem sizes are expressed as qubit counts, so dimensions
/// are powers of two. Subsystem A must fit the dense reduced-state ceiling and
/// the total must fit the existing 20-qubit state-vector ceiling. The reported
/// interval is a normal approximation; it is not a theorem-level confidence
/// guarantee for small sample counts.
///
/// This reproduces the finite-dimensional Haar/Page reference ensemble only.
/// It does not implement local scramblers, sequential evaporation, JT gravity,
/// replica wormholes or an island formula.
pub fn sample_haar_page_entropy_ensemble(
    subsystem_a_qubits: usize,
    subsystem_b_qubits: usize,
    sample_count: usize,
    seed: u64,
    tolerance: f64,
) -> Result<HaarPageEnsembleSummary, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let total_qubits = subsystem_a_qubits
        .checked_add(subsystem_b_qubits)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    if total_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: total_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    if subsystem_a_qubits >= usize::BITS as usize || subsystem_b_qubits >= usize::BITS as usize {
        return Err(QuantumInformationError::DimensionOverflow);
    }
    let dimension_a = 1usize << subsystem_a_qubits;
    let dimension_b = 1usize << subsystem_b_qubits;
    if dimension_a > MAX_DENSE_REDUCED_DIMENSION {
        return Err(QuantumInformationError::DenseDimensionLimit {
            requested: dimension_a,
            maximum: MAX_DENSE_REDUCED_DIMENSION,
        });
    }
    let state_dimension = 1usize << total_qubits;
    let retained_qubits = (0..subsystem_a_qubits).collect::<Vec<_>>();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;

    for sample_index in 0..sample_count {
        let mut amplitudes = Vec::with_capacity(state_dimension);
        while amplitudes.len() < state_dimension {
            // Box-Muller produces independent standard-normal real components;
            // the resulting complex Gaussian is rotationally invariant.
            let u1 = rng.gen_range(0.0_f64..1.0_f64).max(f64::MIN_POSITIVE);
            let u2 = rng.gen_range(0.0_f64..1.0_f64);
            let radius = (-2.0 * u1.ln()).sqrt();
            let phase = core::f64::consts::TAU * u2;
            amplitudes.push(Complex64::new(radius * phase.cos(), radius * phase.sin()));
        }

        // Normalization is part of Haar-state construction, not a repair of a
        // user-supplied state or an evolved numerical trajectory.
        let norm_squared: f64 = amplitudes
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .sum();
        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            return Err(QuantumInformationError::NonFiniteState {
                index: sample_index,
            });
        }
        let inverse_norm = norm_squared.sqrt().recip();
        for amplitude in &mut amplitudes {
            *amplitude *= inverse_norm;
        }

        let reduced =
            DensityMatrix::from_pure_state(&amplitudes, total_qubits, &retained_qubits, tolerance)?;
        let entropy = reduced.von_neumann_entropy(tolerance)?;
        let delta = entropy - mean;
        mean += delta / (sample_index + 1) as f64;
        sum_squared_deviations += delta * (entropy - mean);
    }

    let sample_variance = sum_squared_deviations / (sample_count - 1) as f64;
    let standard_error = (sample_variance / sample_count as f64).sqrt();
    let page_mean = page_average_entropy_nats(dimension_a, dimension_b)?;
    let normalized_deviation = if standard_error > 0.0 {
        (mean - page_mean) / standard_error
    } else if mean == page_mean {
        0.0
    } else {
        f64::INFINITY.copysign(mean - page_mean)
    };
    Ok(HaarPageEnsembleSummary {
        subsystem_a_dimension: dimension_a,
        subsystem_b_dimension: dimension_b,
        sample_count,
        seed,
        mean_von_neumann_entropy_nats: mean,
        sample_variance_nats_squared: sample_variance,
        standard_error_nats: standard_error,
        confidence_95_approx_nats: [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        page_theorem_mean_nats: page_mean,
        normalized_mean_deviation: normalized_deviation,
    })
}

/// One layer of a 1D U(1)-conserving domain-wall transport experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct U1DomainWallTransportSample {
    pub layer: usize,
    pub density_profile: Vec<f64>,
    /// Left-to-right expected excitation current for bond (i,i+1).
    pub bond_currents: Vec<f64>,
    pub total_excitation_expectation: f64,
    pub total_excitation_variance: f64,
    pub left_half_excitation_expectation: f64,
    pub domain_wall_imbalance: f64,
    pub cumulative_cut_transport: f64,
    pub continuity_max_residual: f64,
    pub full_state_norm_squared: f64,
    pub half_chain_entropy_nats: f64,
    pub half_chain_purity: f64,
    pub two_qubit_gate_count_this_layer: usize,
}

/// Full trajectory for a 1D brickwork domain-wall quench.
#[derive(Debug, Clone, PartialEq)]
pub struct U1DomainWallTransportResult {
    pub number_of_qubits: usize,
    pub initial_left_excitations: usize,
    pub seed: u64,
    pub total_two_qubit_gates: usize,
    pub samples: Vec<U1DomainWallTransportSample>,
}

/// Simulate charge diffusion and entanglement growth from a domain-wall product
/// state under nearest-neighbour U(1)-conserving random gates.
///
/// The initial state is |1...10...0> with initial_left_excitations occupied
/// sites at the left edge. The standard benchmark uses an even register and
/// half filling. Each layer applies a brickwork matching of independent Haar
/// U(1) x U(2) x U(1) gates. The routine records site occupations, bond
/// currents, local continuity residuals, left-half charge, domain-wall
/// imbalance, integrated cut transport, and half-chain entropy/purity.
///
/// This is a finite-dimensional closed spin-chain transport experiment, not
/// Hawking emission or a black-hole evaporation model.
pub fn simulate_u1_domain_wall_transport(
    number_of_qubits: usize,
    initial_left_excitations: usize,
    depth: usize,
    seed: u64,
    tolerance: f64,
) -> Result<U1DomainWallTransportResult, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if number_of_qubits < 2 || number_of_qubits % 2 != 0 {
        return Err(QuantumInformationError::InvalidCircuitTopology);
    }
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    if initial_left_excitations != number_of_qubits / 2 {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: initial_left_excitations,
            number_of_qubits,
        });
    }
    let half = number_of_qubits / 2;
    if (1usize << half) > MAX_DENSE_REDUCED_DIMENSION {
        return Err(QuantumInformationError::DenseDimensionLimit {
            requested: 1usize << half,
            maximum: MAX_DENSE_REDUCED_DIMENSION,
        });
    }

    let dimension = 1usize << number_of_qubits;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
    let mut initial_basis = 0usize;
    for qubit in 0..initial_left_excitations {
        initial_basis |= 1usize << (number_of_qubits - 1 - qubit);
    }
    amplitudes[initial_basis] = Complex64::new(1.0, 0.0);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(depth + 1);
    let mut total_two_qubit_gates = 0usize;
    let initial_left_charge = initial_left_excitations as f64;
    let mut cumulative_cut_transport = 0.0;

    for layer in 0..=depth {
        let density_before = (0..number_of_qubits)
            .map(|qubit| qubit_excitation_expectation(&amplitudes, number_of_qubits, qubit))
            .collect::<Vec<_>>();
        let mut currents = vec![0.0; number_of_qubits - 1];
        let mut gates_this_layer = 0usize;
        if layer > 0 {
            let parity = (layer - 1) % 2;
            for first in (parity..(number_of_qubits - 1)).step_by(2) {
                let second = first + 1;
                let before_left = density_before[first];
                let unitary = u1_conserving_unitary_4(&mut rng)?;
                apply_two_qubit_unitary(
                    &mut amplitudes,
                    number_of_qubits,
                    first,
                    second,
                    &unitary,
                )?;
                let after_left = qubit_excitation_expectation(&amplitudes, number_of_qubits, first);
                currents[first] = before_left - after_left;
                gates_this_layer += 1;
                total_two_qubit_gates += 1;
            }
        }

        let density_profile = (0..number_of_qubits)
            .map(|qubit| qubit_excitation_expectation(&amplitudes, number_of_qubits, qubit))
            .collect::<Vec<_>>();
        let left_charge: f64 = density_profile[..half].iter().sum();
        let right_charge: f64 = density_profile[half..].iter().sum();
        let total_charge: f64 = density_profile.iter().sum();
        let domain_wall_imbalance = left_charge - right_charge;
        cumulative_cut_transport += currents[half - 1];
        let continuity_max_residual = (0..number_of_qubits)
            .map(|site| {
                let incoming = if site == 0 { 0.0 } else { currents[site - 1] };
                let outgoing = if site + 1 == number_of_qubits {
                    0.0
                } else {
                    currents[site]
                };
                let actual_change = density_profile[site] - density_before[site];
                (actual_change - (incoming - outgoing)).abs()
            })
            .fold(0.0_f64, f64::max);

        let norm_squared: f64 = amplitudes.iter().map(|value| value.norm_sqr()).sum();
        if !norm_squared.is_finite() {
            return Err(QuantumInformationError::NonFiniteState { index: dimension });
        }
        if (norm_squared - 1.0).abs() > tolerance {
            return Err(QuantumInformationError::NormalizationError {
                norm_squared,
                tolerance,
            });
        }
        let (charge_expectation, charge_variance) = total_excitation_moments(&amplitudes);
        if (charge_expectation - initial_left_charge).abs() > tolerance
            || charge_variance > tolerance
            || (total_charge - initial_left_charge).abs() > tolerance
        {
            return Err(QuantumInformationError::NormalizationError {
                norm_squared: charge_expectation,
                tolerance,
            });
        }
        let retained = (0..half).collect::<Vec<_>>();
        let reduced =
            DensityMatrix::from_pure_state(&amplitudes, number_of_qubits, &retained, tolerance)?;
        let entropy = reduced.von_neumann_entropy(tolerance)?;
        let purity = reduced.diagnostics(tolerance)?.purity;
        samples.push(U1DomainWallTransportSample {
            layer,
            density_profile,
            bond_currents: currents,
            total_excitation_expectation: charge_expectation,
            total_excitation_variance: charge_variance,
            left_half_excitation_expectation: left_charge,
            domain_wall_imbalance,
            cumulative_cut_transport,
            continuity_max_residual,
            full_state_norm_squared: norm_squared,
            half_chain_entropy_nats: entropy,
            half_chain_purity: purity,
            two_qubit_gate_count_this_layer: gates_this_layer,
        });
    }

    Ok(U1DomainWallTransportResult {
        number_of_qubits,
        initial_left_excitations,
        seed,
        total_two_qubit_gates,
        samples,
    })
}

/// Expected occupation of one site in the computational basis.
fn qubit_excitation_expectation(
    amplitudes: &[Complex64],
    number_of_qubits: usize,
    qubit: usize,
) -> f64 {
    let mask = 1usize << (number_of_qubits - 1 - qubit);
    amplitudes
        .iter()
        .enumerate()
        .filter(|(basis, _)| *basis & mask != 0)
        .map(|(_, amplitude)| amplitude.norm_sqr())
        .sum()
}

#[cfg(test)]
mod u1_domain_wall_transport_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn depth_zero_domain_wall_has_exact_density_zero_entropy_and_charge() {
        let result = simulate_u1_domain_wall_transport(8, 4, 0, 0xD0A11, TOLERANCE).unwrap();
        assert_eq!(result.samples.len(), 1);
        let sample = &result.samples[0];
        assert_eq!(
            sample.density_profile,
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert!(sample.half_chain_entropy_nats.abs() < TOLERANCE);
        assert!((sample.total_excitation_expectation - 4.0).abs() < TOLERANCE);
        assert!(sample.total_excitation_variance < TOLERANCE);
        assert!(sample.continuity_max_residual < TOLERANCE);
        assert_eq!(result.total_two_qubit_gates, 0);
    }

    #[test]
    fn domain_wall_layers_preserve_charge_and_report_current_continuity() {
        let result = simulate_u1_domain_wall_transport(8, 4, 12, 0xD1FF, TOLERANCE).unwrap();
        assert_eq!(result.samples.len(), 13);
        for sample in &result.samples {
            assert!((sample.total_excitation_expectation - 4.0).abs() < TOLERANCE);
            assert!(sample.total_excitation_variance < TOLERANCE);
            assert!((sample.full_state_norm_squared - 1.0).abs() < TOLERANCE);
            assert!(sample.continuity_max_residual < TOLERANCE);
            assert!(sample.half_chain_entropy_nats >= -TOLERANCE);
            assert!((0.0..=1.0).contains(&sample.half_chain_purity));
            assert!(
                (sample.left_half_excitation_expectation + sample.cumulative_cut_transport - 4.0)
                    .abs()
                    < 1e-8
            );
        }
        let final_sample = result.samples.last().unwrap();
        assert!(
            (final_sample.left_half_excitation_expectation - 4.0).abs() > 1e-8,
            "domain-wall benchmark seed should produce nonzero transport"
        );
        assert!(result.total_two_qubit_gates > 0);
    }

    #[test]
    fn domain_wall_trajectory_is_seed_reproducible() {
        let a = simulate_u1_domain_wall_transport(8, 4, 6, 0x51DE, TOLERANCE).unwrap();
        let b = simulate_u1_domain_wall_transport(8, 4, 6, 0x51DE, TOLERANCE).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn domain_wall_rejects_non_half_filling_odd_and_unsupported_inputs() {
        assert!(matches!(
            simulate_u1_domain_wall_transport(8, 3, 4, 1, TOLERANCE),
            Err(QuantumInformationError::InvalidExcitationSector { .. })
        ));
        assert!(matches!(
            simulate_u1_domain_wall_transport(7, 3, 4, 1, TOLERANCE),
            Err(QuantumInformationError::InvalidCircuitTopology)
        ));
        assert!(matches!(
            simulate_u1_domain_wall_transport(22, 11, 4, 1, TOLERANCE),
            Err(QuantumInformationError::DenseStateQubitLimit { .. })
        ));
    }
}

/// Mean and standard error for one layer of an independent-seed domain-wall run.
#[derive(Debug, Clone, PartialEq)]
pub struct U1DomainWallTransportLayerSummary {
    pub layer: usize,
    pub sample_count: usize,
    pub mean_density_profile: Vec<f64>,
    pub density_profile_standard_error: Vec<f64>,
    pub mean_bond_currents: Vec<f64>,
    pub bond_current_standard_error: Vec<f64>,
    pub mean_domain_wall_imbalance: f64,
    pub domain_wall_imbalance_standard_error: f64,
    pub mean_cumulative_cut_transport: f64,
    pub cumulative_cut_transport_standard_error: f64,
    pub mean_half_chain_entropy_nats: f64,
    pub half_chain_entropy_standard_error: f64,
    pub mean_half_chain_purity: f64,
    pub maximum_continuity_residual: f64,
    pub maximum_norm_drift: f64,
    pub maximum_charge_drift: f64,
    pub maximum_charge_variance: f64,
}

/// Reproducible independent-seed domain-wall transport ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct U1DomainWallTransportEnsembleSummary {
    pub number_of_qubits: usize,
    pub initial_left_excitations: usize,
    pub depth: usize,
    pub sample_count: usize,
    pub first_seed: u64,
    pub layers: Vec<U1DomainWallTransportLayerSummary>,
}

/// Run independent ChaCha20-seeded domain-wall quenches and report per-layer
/// means, standard errors, continuity, norm and charge qualification.
pub fn sample_u1_domain_wall_transport_ensemble(
    number_of_qubits: usize,
    initial_left_excitations: usize,
    depth: usize,
    sample_count: usize,
    first_seed: u64,
    tolerance: f64,
) -> Result<U1DomainWallTransportEnsembleSummary, QuantumInformationError> {
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let mut trajectories = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        trajectories.push(simulate_u1_domain_wall_transport(
            number_of_qubits,
            initial_left_excitations,
            depth,
            first_seed.wrapping_add(sample_index as u64),
            tolerance,
        )?);
    }

    let mut layers = Vec::with_capacity(depth + 1);
    for layer_index in 0..=depth {
        let qubits = number_of_qubits;
        let bonds = number_of_qubits - 1;
        let mut density_mean = vec![0.0; qubits];
        let mut density_m2 = vec![0.0; qubits];
        let mut current_mean = vec![0.0; bonds];
        let mut current_m2 = vec![0.0; bonds];
        let mut imbalance_mean = 0.0;
        let mut imbalance_m2 = 0.0;
        let mut cut_mean = 0.0;
        let mut cut_m2 = 0.0;
        let mut entropy_mean = 0.0;
        let mut entropy_m2 = 0.0;
        let mut purity_mean = 0.0;
        let mut maximum_continuity_residual: f64 = 0.0;
        let mut maximum_norm_drift: f64 = 0.0;
        let mut maximum_charge_drift: f64 = 0.0;
        let mut maximum_charge_variance: f64 = 0.0;

        for (sample_index, trajectory) in trajectories.iter().enumerate() {
            let sample = &trajectory.samples[layer_index];
            let count = (sample_index + 1) as f64;
            for site in 0..qubits {
                let delta = sample.density_profile[site] - density_mean[site];
                density_mean[site] += delta / count;
                density_m2[site] += delta * (sample.density_profile[site] - density_mean[site]);
            }
            for bond in 0..bonds {
                let delta = sample.bond_currents[bond] - current_mean[bond];
                current_mean[bond] += delta / count;
                current_m2[bond] += delta * (sample.bond_currents[bond] - current_mean[bond]);
            }
            let delta_imbalance = sample.domain_wall_imbalance - imbalance_mean;
            imbalance_mean += delta_imbalance / count;
            imbalance_m2 += delta_imbalance * (sample.domain_wall_imbalance - imbalance_mean);

            let delta_cut = sample.cumulative_cut_transport - cut_mean;
            cut_mean += delta_cut / count;
            cut_m2 += delta_cut * (sample.cumulative_cut_transport - cut_mean);

            let delta_entropy = sample.half_chain_entropy_nats - entropy_mean;
            entropy_mean += delta_entropy / count;
            entropy_m2 += delta_entropy * (sample.half_chain_entropy_nats - entropy_mean);
            purity_mean += (sample.half_chain_purity - purity_mean) / count;

            maximum_continuity_residual =
                maximum_continuity_residual.max(sample.continuity_max_residual);
            maximum_norm_drift =
                maximum_norm_drift.max((sample.full_state_norm_squared - 1.0).abs());
            maximum_charge_drift = maximum_charge_drift
                .max((sample.total_excitation_expectation - initial_left_excitations as f64).abs());
            maximum_charge_variance = maximum_charge_variance.max(sample.total_excitation_variance);
        }

        let denominator = (sample_count - 1) as f64;
        let inverse_sqrt_count = 1.0 / (sample_count as f64).sqrt();
        let density_standard_error = density_m2
            .iter()
            .map(|m2| (m2 / denominator).sqrt() * inverse_sqrt_count)
            .collect();
        let bond_current_standard_error = current_m2
            .iter()
            .map(|m2| (m2 / denominator).sqrt() * inverse_sqrt_count)
            .collect();
        layers.push(U1DomainWallTransportLayerSummary {
            layer: layer_index,
            sample_count,
            mean_density_profile: density_mean,
            density_profile_standard_error,
            mean_bond_currents: current_mean,
            bond_current_standard_error,
            mean_domain_wall_imbalance: imbalance_mean,
            domain_wall_imbalance_standard_error: (imbalance_m2 / denominator).sqrt()
                * inverse_sqrt_count,
            mean_cumulative_cut_transport: cut_mean,
            cumulative_cut_transport_standard_error: (cut_m2 / denominator).sqrt()
                * inverse_sqrt_count,
            mean_half_chain_entropy_nats: entropy_mean,
            half_chain_entropy_standard_error: (entropy_m2 / denominator).sqrt()
                * inverse_sqrt_count,
            mean_half_chain_purity: purity_mean,
            maximum_continuity_residual,
            maximum_norm_drift,
            maximum_charge_drift,
            maximum_charge_variance,
        });
    }

    Ok(U1DomainWallTransportEnsembleSummary {
        number_of_qubits,
        initial_left_excitations,
        depth,
        sample_count,
        first_seed,
        layers,
    })
}

#[cfg(test)]
mod u1_domain_wall_transport_ensemble_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn domain_wall_ensemble_reports_profile_transport_entropy_and_uncertainty() {
        let summary =
            sample_u1_domain_wall_transport_ensemble(8, 4, 6, 8, 0xE115, TOLERANCE).unwrap();
        assert_eq!(summary.layers.len(), 7);
        for layer in &summary.layers {
            assert_eq!(layer.sample_count, 8);
            assert_eq!(layer.mean_density_profile.len(), 8);
            assert_eq!(layer.density_profile_standard_error.len(), 8);
            assert_eq!(layer.mean_bond_currents.len(), 7);
            assert_eq!(layer.bond_current_standard_error.len(), 7);
            assert!(layer.maximum_continuity_residual < TOLERANCE);
            assert!(layer.maximum_norm_drift < TOLERANCE);
            assert!(layer.maximum_charge_drift < TOLERANCE);
            assert!(layer.maximum_charge_variance < TOLERANCE);
            assert!(layer.domain_wall_imbalance_standard_error >= 0.0);
            assert!(layer.half_chain_entropy_standard_error >= 0.0);
            assert!((layer.mean_density_profile.iter().sum::<f64>() - 4.0).abs() < 1e-9);
            assert!(
                (layer.mean_cumulative_cut_transport
                    + layer.mean_density_profile[..4].iter().sum::<f64>()
                    - 4.0)
                    .abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn domain_wall_ensemble_is_reproducible_for_a_frozen_seed_schedule() {
        let a = sample_u1_domain_wall_transport_ensemble(8, 4, 4, 4, 0xE115, TOLERANCE).unwrap();
        let b = sample_u1_domain_wall_transport_ensemble(8, 4, 4, 4, 0xE115, TOLERANCE).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn domain_wall_ensemble_rejects_single_sample() {
        assert!(matches!(
            sample_u1_domain_wall_transport_ensemble(8, 4, 2, 1, 5, TOLERANCE),
            Err(QuantumInformationError::InvalidEnsembleSampleCount(1))
        ));
    }
}

/// One-step, exactly enumerable Hayden-Preskill-style decoupling diagnostic.
///
/// Q is a reference maximally entangled with message M; the initial black-hole
/// factor B is maximally entangled with early radiation R. A seeded Haar U(4)
/// scrambler acts on M and B. The two output qubits are labelled outgoing E
/// and remaining black-hole B'. The accessible output is A = R tensor E.
///
/// This computes a decoupling diagnostic only. It does not construct or run a
/// recovery decoder, nor does this four-qubit model represent a macroscopic
/// black hole or a gravitational evaporation process.
#[derive(Debug, Clone, PartialEq)]
pub struct HaydenPreskillDecouplingSample {
    pub seed: u64,
    pub state_norm_squared: f64,
    pub reference_entropy_nats: f64,
    pub remaining_black_hole_entropy_nats: f64,
    pub reference_black_hole_joint_entropy_nats: f64,
    pub reference_remaining_black_hole_mutual_information_nats: f64,
    pub accessible_radiation_entropy_nats: f64,
    pub reference_accessible_radiation_joint_entropy_nats: f64,
    pub reference_accessible_radiation_mutual_information_nats: f64,
    /// For a pure Q B' A state, I(Q:B') + I(Q:A) = 2 S(Q).
    pub information_complementarity_residual_nats: f64,
}

/// Single seeded one-qubit-message/one-qubit-black-hole decoupling experiment.
pub fn simulate_small_hayden_preskill_decoupling(
    seed: u64,
    tolerance: f64,
) -> Result<HaydenPreskillDecouplingSample, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    const NUMBER_OF_QUBITS: usize = 4;
    // Register order is Q, M, B, R (MSB-first). Both pairs start in Bell states.
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); 1 << NUMBER_OF_QUBITS];
    for message_bit in 0..2 {
        for black_hole_bit in 0..2 {
            let q = message_bit;
            let m = message_bit;
            let b = black_hole_bit;
            let r = black_hole_bit;
            let basis = (q << 3) | (m << 2) | (b << 1) | r;
            amplitudes[basis] = Complex64::new(0.5, 0.0);
        }
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let scrambler = haar_unitary_4(&mut rng)?;
    apply_two_qubit_unitary(
        &mut amplitudes,
        NUMBER_OF_QUBITS,
        1, // message M
        2, // initial black-hole register B
        &scrambler,
    )?;

    let norm_squared: f64 = amplitudes
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .sum();
    if !norm_squared.is_finite() {
        return Err(QuantumInformationError::NonFiniteState {
            index: amplitudes.len(),
        });
    }
    if (norm_squared - 1.0).abs() > tolerance {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared,
            tolerance,
        });
    }

    let entropy_of = |retained: &[usize]| -> Result<f64, QuantumInformationError> {
        DensityMatrix::from_pure_state(&amplitudes, NUMBER_OF_QUBITS, retained, tolerance)?
            .von_neumann_entropy(tolerance)
    };

    let reference_entropy = entropy_of(&[0])?;
    let remaining_black_hole_entropy = entropy_of(&[2])?;
    let reference_black_hole_joint_entropy = entropy_of(&[0, 2])?;
    let accessible_radiation_entropy = entropy_of(&[1, 3])?;
    let reference_accessible_radiation_joint_entropy = entropy_of(&[0, 1, 3])?;

    let reference_black_hole_mutual_information =
        reference_entropy + remaining_black_hole_entropy - reference_black_hole_joint_entropy;
    let reference_accessible_radiation_mutual_information = reference_entropy
        + accessible_radiation_entropy
        - reference_accessible_radiation_joint_entropy;
    let information_complementarity_residual = (reference_black_hole_mutual_information
        + reference_accessible_radiation_mutual_information
        - 2.0 * reference_entropy)
        .abs();

    for value in [
        reference_black_hole_mutual_information,
        reference_accessible_radiation_mutual_information,
    ] {
        if !value.is_finite() || value < -tolerance {
            return Err(QuantumInformationError::InvalidMutualInformation { value, tolerance });
        }
    }
    if information_complementarity_residual > tolerance * 10.0 {
        return Err(QuantumInformationError::InformationIdentityError {
            residual: information_complementarity_residual,
            tolerance: tolerance * 10.0,
        });
    }

    Ok(HaydenPreskillDecouplingSample {
        seed,
        state_norm_squared: norm_squared,
        reference_entropy_nats: reference_entropy,
        remaining_black_hole_entropy_nats: remaining_black_hole_entropy,
        reference_black_hole_joint_entropy_nats: reference_black_hole_joint_entropy,
        reference_remaining_black_hole_mutual_information_nats:
            reference_black_hole_mutual_information.max(0.0),
        accessible_radiation_entropy_nats: accessible_radiation_entropy,
        reference_accessible_radiation_joint_entropy_nats:
            reference_accessible_radiation_joint_entropy,
        reference_accessible_radiation_mutual_information_nats:
            reference_accessible_radiation_mutual_information.max(0.0),
        information_complementarity_residual_nats: information_complementarity_residual,
    })
}

/// Independent-seed ensemble statistics for the one-step decoupling diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct HaydenPreskillDecouplingSummary {
    pub sample_count: usize,
    pub first_seed: u64,
    pub mean_reference_black_hole_mutual_information_nats: f64,
    pub standard_error_reference_black_hole_mutual_information_nats: f64,
    pub mean_reference_accessible_radiation_mutual_information_nats: f64,
    pub standard_error_reference_accessible_radiation_mutual_information_nats: f64,
    pub maximum_information_complementarity_residual_nats: f64,
}

/// Sample independent two-qubit scramblers and preserve per-ensemble uncertainty.
pub fn sample_small_hayden_preskill_decoupling_ensemble(
    sample_count: usize,
    first_seed: u64,
    tolerance: f64,
) -> Result<HaydenPreskillDecouplingSummary, QuantumInformationError> {
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let mut mean_qb = 0.0;
    let mut m2_qb = 0.0;
    let mut mean_qa = 0.0;
    let mut m2_qa = 0.0;
    let mut maximum_complementarity_residual: f64 = 0.0;
    for sample_index in 0..sample_count {
        let sample = simulate_small_hayden_preskill_decoupling(
            first_seed.wrapping_add(sample_index as u64),
            tolerance,
        )?;
        let count = (sample_index + 1) as f64;
        let delta_qb = sample.reference_remaining_black_hole_mutual_information_nats - mean_qb;
        mean_qb += delta_qb / count;
        m2_qb +=
            delta_qb * (sample.reference_remaining_black_hole_mutual_information_nats - mean_qb);
        let delta_qa = sample.reference_accessible_radiation_mutual_information_nats - mean_qa;
        mean_qa += delta_qa / count;
        m2_qa +=
            delta_qa * (sample.reference_accessible_radiation_mutual_information_nats - mean_qa);
        maximum_complementarity_residual =
            maximum_complementarity_residual.max(sample.information_complementarity_residual_nats);
    }
    let denominator = (sample_count - 1) as f64;
    Ok(HaydenPreskillDecouplingSummary {
        sample_count,
        first_seed,
        mean_reference_black_hole_mutual_information_nats: mean_qb,
        standard_error_reference_black_hole_mutual_information_nats: (m2_qb
            / denominator
            / sample_count as f64)
            .sqrt(),
        mean_reference_accessible_radiation_mutual_information_nats: mean_qa,
        standard_error_reference_accessible_radiation_mutual_information_nats: (m2_qa
            / denominator
            / sample_count as f64)
            .sqrt(),
        maximum_information_complementarity_residual_nats: maximum_complementarity_residual,
    })
}

#[cfg(test)]
mod hayden_preskill_decoupling_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn one_step_hayden_preskill_toy_preserves_reference_entropy_and_information_balance() {
        let sample = simulate_small_hayden_preskill_decoupling(0x48A7, TOLERANCE).unwrap();
        assert!((sample.reference_entropy_nats - core::f64::consts::LN_2).abs() < TOLERANCE);
        assert!((sample.state_norm_squared - 1.0).abs() < TOLERANCE);
        assert!(sample.reference_remaining_black_hole_mutual_information_nats >= -TOLERANCE);
        assert!(sample.reference_accessible_radiation_mutual_information_nats >= -TOLERANCE);
        assert!(
            (sample.reference_black_hole_joint_entropy_nats
                - sample.accessible_radiation_entropy_nats)
                .abs()
                < TOLERANCE
        );
        assert!(sample.information_complementarity_residual_nats < TOLERANCE);
    }

    #[test]
    fn one_step_hayden_preskill_toy_is_seed_reproducible() {
        let first = simulate_small_hayden_preskill_decoupling(0x48A7, TOLERANCE).unwrap();
        let second = simulate_small_hayden_preskill_decoupling(0x48A7, TOLERANCE).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn one_step_hayden_preskill_ensemble_reports_uncertainty_and_complementarity() {
        let summary =
            sample_small_hayden_preskill_decoupling_ensemble(32, 0x4850, TOLERANCE).unwrap();
        assert_eq!(summary.sample_count, 32);
        assert!(summary.standard_error_reference_black_hole_mutual_information_nats >= 0.0);
        assert!(
            summary.standard_error_reference_accessible_radiation_mutual_information_nats >= 0.0
        );
        assert!(summary.maximum_information_complementarity_residual_nats < TOLERANCE);
    }

    #[test]
    fn information_failures_have_specific_error_messages() {
        let mutual_information = QuantumInformationError::InvalidMutualInformation {
            value: -0.01,
            tolerance: 1e-10,
        };
        let identity = QuantumInformationError::InformationIdentityError {
            residual: 0.02,
            tolerance: 1e-9,
        };
        assert!(
            mutual_information
                .to_string()
                .contains("mutual information")
        );
        assert!(
            identity
                .to_string()
                .contains("information complementarity residual")
        );
    }

    #[test]
    fn one_step_hayden_preskill_ensemble_rejects_single_sample() {
        assert!(matches!(
            sample_small_hayden_preskill_decoupling_ensemble(1, 1, TOLERANCE),
            Err(QuantumInformationError::InvalidEnsembleSampleCount(1))
        ));
    }
}

/// Operator-spreading profile under a declared U(1)-conserving brickwork circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct U1OperatorSpreadingSample {
    pub number_of_qubits: usize,
    pub source_qubit: usize,
    pub depth: usize,
    pub seed: u64,
    pub two_qubit_gate_count: usize,
    /// Normalized squared commutator with local Z probes, one per site.
    pub squared_commutator_z: Vec<f64>,
    /// Normalized squared commutator with local X probes, one per site.
    pub squared_commutator_x: Vec<f64>,
    pub operator_hilbert_schmidt_norm_squared: f64,
}

/// Evolve a local X operator by forward unitary conjugation and measure its
/// commutator profile against local X and Z probes.
///
/// The declared operator convention is O <- U O U^dagger for each brickwork
/// gate in circuit application order. The normalized squared commutator is
/// ||[O,V]||_F^2/(2*2^N), which is zero for commuting probes and equals two
/// for anticommuting Pauli operators. The gate family preserves total U(1)
/// excitation number. This is an operator-influence diagnostic, not a decoder.
pub fn simulate_u1_operator_spreading(
    number_of_qubits: usize,
    source_qubit: usize,
    depth: usize,
    seed: u64,
    tolerance: f64,
) -> Result<U1OperatorSpreadingSample, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if number_of_qubits < 2 || number_of_qubits > 8 {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: 8,
        });
    }
    if source_qubit >= number_of_qubits {
        return Err(QuantumInformationError::InvalidQubit {
            qubit: source_qubit,
            register_size: number_of_qubits,
        });
    }

    let dimension = 1usize << number_of_qubits;
    let mut operator = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let source_mask = 1usize << (number_of_qubits - 1 - source_qubit);
    for basis in 0..dimension {
        operator[basis * dimension + (basis ^ source_mask)] = Complex64::new(1.0, 0.0);
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut gate_count = 0usize;
    for layer in 0..depth {
        let parity = layer % 2;
        for first in (parity..(number_of_qubits - 1)).step_by(2) {
            let gate = u1_conserving_unitary_4(&mut rng)?;
            conjugate_operator_by_two_qubit_unitary(
                &mut operator,
                number_of_qubits,
                first,
                first + 1,
                &gate,
            )?;
            gate_count += 1;
        }
    }

    let norm_squared =
        operator.iter().map(|value| value.norm_sqr()).sum::<f64>() / dimension as f64;
    if !norm_squared.is_finite() || (norm_squared - 1.0).abs() > tolerance * 10.0 {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared,
            tolerance: tolerance * 10.0,
        });
    }

    let mut squared_commutator_z = Vec::with_capacity(number_of_qubits);
    let mut squared_commutator_x = Vec::with_capacity(number_of_qubits);
    for qubit in 0..number_of_qubits {
        let mask = 1usize << (number_of_qubits - 1 - qubit);
        let mut z_commutator_norm = 0.0;
        let mut x_commutator_norm = 0.0;
        for row in 0..dimension {
            let z_row = if row & mask == 0 { 1.0 } else { -1.0 };
            let flipped_row = row ^ mask;
            for column in 0..dimension {
                let z_column = if column & mask == 0 { 1.0 } else { -1.0 };
                let entry = operator[row * dimension + column];
                z_commutator_norm += (entry * (z_column - z_row)).norm_sqr();

                let right_x = operator[row * dimension + (column ^ mask)];
                let left_x = operator[flipped_row * dimension + column];
                x_commutator_norm += (right_x - left_x).norm_sqr();
            }
        }
        squared_commutator_z.push(z_commutator_norm / (2.0 * dimension as f64));
        squared_commutator_x.push(x_commutator_norm / (2.0 * dimension as f64));
    }

    if squared_commutator_z
        .iter()
        .chain(squared_commutator_x.iter())
        .any(|value| !value.is_finite() || *value < -tolerance || *value > 2.0 + tolerance)
    {
        return Err(QuantumInformationError::InvalidSquaredCommutator {
            value: squared_commutator_z
                .iter()
                .chain(squared_commutator_x.iter())
                .copied()
                .find(|value| !value.is_finite() || *value < -tolerance || *value > 2.0 + tolerance)
                .unwrap_or(f64::NAN),
            tolerance,
        });
    }

    Ok(U1OperatorSpreadingSample {
        number_of_qubits,
        source_qubit,
        depth,
        seed,
        two_qubit_gate_count: gate_count,
        squared_commutator_z,
        squared_commutator_x,
        operator_hilbert_schmidt_norm_squared: norm_squared,
    })
}

/// Independent-seed statistics for U(1)-conserving operator spreading.
#[derive(Debug, Clone, PartialEq)]
pub struct U1OperatorSpreadingEnsembleSummary {
    pub number_of_qubits: usize,
    pub source_qubit: usize,
    pub depth: usize,
    pub sample_count: usize,
    pub first_seed: u64,
    pub mean_squared_commutator_z: Vec<f64>,
    pub standard_error_squared_commutator_z: Vec<f64>,
    pub mean_squared_commutator_x: Vec<f64>,
    pub standard_error_squared_commutator_x: Vec<f64>,
    pub maximum_operator_norm_drift: f64,
}

/// Sample independent operator-spreading circuits and report per-site
/// squared-commutator means and standard errors.
pub fn sample_u1_operator_spreading_ensemble(
    number_of_qubits: usize,
    source_qubit: usize,
    depth: usize,
    sample_count: usize,
    first_seed: u64,
    tolerance: f64,
) -> Result<U1OperatorSpreadingEnsembleSummary, QuantumInformationError> {
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    let mut mean_z = vec![0.0; number_of_qubits];
    let mut m2_z = vec![0.0; number_of_qubits];
    let mut mean_x = vec![0.0; number_of_qubits];
    let mut m2_x = vec![0.0; number_of_qubits];
    let mut maximum_norm_drift: f64 = 0.0;

    for sample_index in 0..sample_count {
        let sample = simulate_u1_operator_spreading(
            number_of_qubits,
            source_qubit,
            depth,
            first_seed.wrapping_add(sample_index as u64),
            tolerance,
        )?;
        let count = (sample_index + 1) as f64;
        for site in 0..number_of_qubits {
            let delta_z = sample.squared_commutator_z[site] - mean_z[site];
            mean_z[site] += delta_z / count;
            m2_z[site] += delta_z * (sample.squared_commutator_z[site] - mean_z[site]);

            let delta_x = sample.squared_commutator_x[site] - mean_x[site];
            mean_x[site] += delta_x / count;
            m2_x[site] += delta_x * (sample.squared_commutator_x[site] - mean_x[site]);
        }
        maximum_norm_drift =
            maximum_norm_drift.max((sample.operator_hilbert_schmidt_norm_squared - 1.0).abs());
    }
    let denominator = (sample_count - 1) as f64;
    let inverse_sqrt_count = 1.0 / (sample_count as f64).sqrt();
    let standard_error_z = m2_z
        .iter()
        .map(|m2| (m2 / denominator).sqrt() * inverse_sqrt_count)
        .collect();
    let standard_error_x = m2_x
        .iter()
        .map(|m2| (m2 / denominator).sqrt() * inverse_sqrt_count)
        .collect();

    Ok(U1OperatorSpreadingEnsembleSummary {
        number_of_qubits,
        source_qubit,
        depth,
        sample_count,
        first_seed,
        mean_squared_commutator_z: mean_z,
        standard_error_squared_commutator_z: standard_error_z,
        mean_squared_commutator_x: mean_x,
        standard_error_squared_commutator_x: standard_error_x,
        maximum_operator_norm_drift: maximum_norm_drift,
    })
}

#[cfg(test)]
mod u1_operator_spreading_ensemble_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn operator_spreading_ensemble_reports_per_site_uncertainty_and_norm() {
        let summary = sample_u1_operator_spreading_ensemble(6, 2, 6, 8, 0x0A70, TOLERANCE).unwrap();
        assert_eq!(summary.sample_count, 8);
        assert_eq!(summary.mean_squared_commutator_z.len(), 6);
        assert_eq!(summary.standard_error_squared_commutator_z.len(), 6);
        assert_eq!(summary.mean_squared_commutator_x.len(), 6);
        assert_eq!(summary.standard_error_squared_commutator_x.len(), 6);
        assert!(summary.maximum_operator_norm_drift < TOLERANCE);
        assert!(
            summary
                .standard_error_squared_commutator_z
                .iter()
                .chain(summary.standard_error_squared_commutator_x.iter())
                .all(|value| *value >= 0.0)
        );
    }

    #[test]
    fn operator_spreading_ensemble_is_seed_reproducible() {
        let a = sample_u1_operator_spreading_ensemble(6, 2, 4, 4, 0x0A70, TOLERANCE).unwrap();
        let b = sample_u1_operator_spreading_ensemble(6, 2, 4, 4, 0x0A70, TOLERANCE).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn operator_spreading_ensemble_rejects_single_sample() {
        assert!(matches!(
            sample_u1_operator_spreading_ensemble(6, 2, 4, 1, 0x0A70, TOLERANCE),
            Err(QuantumInformationError::InvalidEnsembleSampleCount(1))
        ));
    }
}

/// Apply O <- U O U^dagger to an operator stored as a row-major d-by-d matrix.
/// Rows and columns are grouped independently by the two acted-on tensor bits.
fn conjugate_operator_by_two_qubit_unitary(
    operator: &mut [Complex64],
    number_of_qubits: usize,
    first_qubit: usize,
    second_qubit: usize,
    unitary: &[[Complex64; 4]; 4],
) -> Result<(), QuantumInformationError> {
    if first_qubit >= number_of_qubits
        || second_qubit >= number_of_qubits
        || first_qubit == second_qubit
    {
        return Err(QuantumInformationError::InvalidQubit {
            qubit: first_qubit.max(second_qubit),
            register_size: number_of_qubits,
        });
    }
    let dimension = 1usize
        .checked_shl(number_of_qubits as u32)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    let expected = dimension
        .checked_mul(dimension)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    if operator.len() != expected {
        return Err(QuantumInformationError::DimensionMismatch {
            expected,
            actual: operator.len(),
        });
    }
    let first_mask = 1usize << (number_of_qubits - 1 - first_qubit);
    let second_mask = 1usize << (number_of_qubits - 1 - second_qubit);
    let dagger = core::array::from_fn::<_, 4, _>(|row| {
        core::array::from_fn::<_, 4, _>(|column| unitary[column][row].conj())
    });

    for row_base in 0..dimension {
        if row_base & first_mask != 0 || row_base & second_mask != 0 {
            continue;
        }
        let row_indices = [
            row_base,
            row_base | second_mask,
            row_base | first_mask,
            row_base | first_mask | second_mask,
        ];
        for column_base in 0..dimension {
            if column_base & first_mask != 0 || column_base & second_mask != 0 {
                continue;
            }
            let column_indices = [
                column_base,
                column_base | second_mask,
                column_base | first_mask,
                column_base | first_mask | second_mask,
            ];
            let mut block = [[Complex64::new(0.0, 0.0); 4]; 4];
            for row in 0..4 {
                for column in 0..4 {
                    block[row][column] =
                        operator[row_indices[row] * dimension + column_indices[column]];
                }
            }
            let mut left_product = [[Complex64::new(0.0, 0.0); 4]; 4];
            for row in 0..4 {
                for column in 0..4 {
                    for inner in 0..4 {
                        left_product[row][column] += unitary[row][inner] * block[inner][column];
                    }
                }
            }
            let mut transformed = [[Complex64::new(0.0, 0.0); 4]; 4];
            for row in 0..4 {
                for column in 0..4 {
                    for inner in 0..4 {
                        transformed[row][column] +=
                            left_product[row][inner] * dagger[inner][column];
                    }
                }
            }
            for row in 0..4 {
                for column in 0..4 {
                    operator[row_indices[row] * dimension + column_indices[column]] =
                        transformed[row][column];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod u1_operator_spreading_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn depth_zero_pauli_operator_has_exact_local_commutator_profile() {
        let sample = simulate_u1_operator_spreading(6, 2, 0, 0x0F70, TOLERANCE).unwrap();
        assert_eq!(sample.two_qubit_gate_count, 0);
        assert!((sample.operator_hilbert_schmidt_norm_squared - 1.0).abs() < TOLERANCE);
        for qubit in 0..6 {
            let expected_z = if qubit == 2 { 2.0 } else { 0.0 };
            assert!((sample.squared_commutator_z[qubit] - expected_z).abs() < TOLERANCE);
            assert!(sample.squared_commutator_x[qubit].abs() < TOLERANCE);
        }
    }

    #[test]
    fn u1_operator_spreading_preserves_operator_norm_and_commutator_bounds() {
        let sample = simulate_u1_operator_spreading(6, 2, 8, 0x07C0, TOLERANCE).unwrap();
        assert!((sample.operator_hilbert_schmidt_norm_squared - 1.0).abs() < TOLERANCE);
        assert!(
            sample
                .squared_commutator_z
                .iter()
                .chain(sample.squared_commutator_x.iter())
                .all(|value| *value >= -TOLERANCE && *value <= 2.0 + TOLERANCE)
        );
        assert!(
            sample
                .squared_commutator_z
                .iter()
                .any(|value| *value > 1e-8)
        );
    }

    #[test]
    fn u1_operator_spreading_is_seed_reproducible_and_rejects_oversized_dense_operator() {
        let a = simulate_u1_operator_spreading(6, 2, 5, 0x07C0, TOLERANCE).unwrap();
        let b = simulate_u1_operator_spreading(6, 2, 5, 0x07C0, TOLERANCE).unwrap();
        assert_eq!(a, b);
        assert!(matches!(
            simulate_u1_operator_spreading(9, 2, 2, 1, TOLERANCE),
            Err(QuantumInformationError::DenseStateQubitLimit { .. })
        ));
    }
}

/// Sampling statistics for Haar-random states inside a fixed U(1) charge sector.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryResolvedPageEnsembleSummary {
    pub number_of_qubits: usize,
    pub total_excitations: usize,
    pub subsystem_a_qubits: usize,
    pub sample_count: usize,
    pub seed: u64,
    pub mean_von_neumann_entropy_nats: f64,
    pub sample_variance_nats_squared: f64,
    pub standard_error_nats: f64,
    /// Normal-approximation interval, not a finite-sample coverage guarantee.
    pub confidence_95_approx_nats: [f64; 2],
    pub symmetry_resolved_page_mean_nats: f64,
    pub normalized_mean_deviation: f64,
}

/// Sample Haar-random pure states from one fixed-excitation subspace and
/// compare their reduced entropy with the exact symmetry-resolved Page oracle.
/// The random complex-Gaussian amplitudes are populated only on computational
/// basis states whose Hamming weight equals total_excitations.
pub fn sample_symmetry_resolved_page_entropy_ensemble(
    number_of_qubits: usize,
    total_excitations: usize,
    subsystem_a_qubits: usize,
    sample_count: usize,
    seed: u64,
    tolerance: f64,
) -> Result<SymmetryResolvedPageEnsembleSummary, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if sample_count < 2 {
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(
            sample_count,
        ));
    }
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    if subsystem_a_qubits > number_of_qubits {
        return Err(QuantumInformationError::InvalidSubsystemSize {
            requested: subsystem_a_qubits,
            maximum: number_of_qubits,
        });
    }
    if total_excitations > number_of_qubits {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: total_excitations,
            number_of_qubits,
        });
    }
    let subsystem_dimension = 1usize << subsystem_a_qubits;
    if subsystem_dimension > MAX_DENSE_REDUCED_DIMENSION {
        return Err(QuantumInformationError::DenseDimensionLimit {
            requested: subsystem_dimension,
            maximum: MAX_DENSE_REDUCED_DIMENSION,
        });
    }

    // Calculate the exact oracle first; this also checks the fixed-sector
    // Hilbert-space resource ceiling before allocating any sample state.
    let exact_page_mean = symmetry_resolved_page_average_entropy_nats(
        number_of_qubits,
        total_excitations,
        subsystem_a_qubits,
    )?;
    let state_dimension = 1usize << number_of_qubits;
    let retained_qubits = (0..subsystem_a_qubits).collect::<Vec<_>>();
    let sector_basis = (0..state_dimension)
        .filter(|basis| basis.count_ones() as usize == total_excitations)
        .collect::<Vec<_>>();
    if sector_basis.is_empty() {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: total_excitations,
            number_of_qubits,
        });
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    for sample_index in 0..sample_count {
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); state_dimension];
        for &basis in &sector_basis {
            amplitudes[basis] = complex_standard_normal(&mut rng);
        }
        let norm_squared: f64 = amplitudes
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .sum();
        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            return Err(QuantumInformationError::NonFiniteState {
                index: sample_index,
            });
        }
        let inverse_norm = norm_squared.sqrt().recip();
        for amplitude in &mut amplitudes {
            *amplitude *= inverse_norm;
        }

        let reduced = DensityMatrix::from_pure_state(
            &amplitudes,
            number_of_qubits,
            &retained_qubits,
            tolerance,
        )?;
        let entropy = reduced.von_neumann_entropy(tolerance)?;
        let delta = entropy - mean;
        mean += delta / (sample_index + 1) as f64;
        sum_squared_deviations += delta * (entropy - mean);
    }

    let sample_variance = sum_squared_deviations / (sample_count - 1) as f64;
    let standard_error = (sample_variance / sample_count as f64).sqrt();
    let normalized_deviation = if standard_error > 0.0 {
        (mean - exact_page_mean) / standard_error
    } else if mean == exact_page_mean {
        0.0
    } else {
        f64::INFINITY.copysign(mean - exact_page_mean)
    };
    Ok(SymmetryResolvedPageEnsembleSummary {
        number_of_qubits,
        total_excitations,
        subsystem_a_qubits,
        sample_count,
        seed,
        mean_von_neumann_entropy_nats: mean,
        sample_variance_nats_squared: sample_variance,
        standard_error_nats: standard_error,
        confidence_95_approx_nats: [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        symmetry_resolved_page_mean_nats: exact_page_mean,
        normalized_mean_deviation: normalized_deviation,
    })
}

#[cfg(test)]
mod symmetry_resolved_page_sampler_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn fixed_charge_haar_sampler_is_seeded_and_uses_matching_sector_oracle() {
        let first =
            sample_symmetry_resolved_page_entropy_ensemble(8, 4, 4, 256, 0xC4A26E, TOLERANCE)
                .unwrap();
        let repeated =
            sample_symmetry_resolved_page_entropy_ensemble(8, 4, 4, 256, 0xC4A26E, TOLERANCE)
                .unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.total_excitations, 4);
        assert!(first.sample_variance_nats_squared > 0.0);
        assert!(first.standard_error_nats > 0.0);
        assert!(first.normalized_mean_deviation.abs() < 6.0);
        assert!(
            (first.mean_von_neumann_entropy_nats - first.symmetry_resolved_page_mean_nats).abs()
                < 0.10
        );
    }

    #[test]
    fn fixed_charge_haar_sampler_has_zero_entropy_in_extremal_sectors() {
        for total_excitations in [0, 4] {
            let sample = sample_symmetry_resolved_page_entropy_ensemble(
                4,
                total_excitations,
                2,
                16,
                0xE100 + total_excitations as u64,
                TOLERANCE,
            )
            .unwrap();
            assert!(sample.mean_von_neumann_entropy_nats.abs() < TOLERANCE);
            assert!(sample.symmetry_resolved_page_mean_nats.abs() < TOLERANCE);
            assert!(sample.sample_variance_nats_squared.abs() < 1e-24);
        }
    }

    #[test]
    fn fixed_charge_haar_sampler_rejects_invalid_sample_and_register_inputs() {
        assert!(matches!(
            sample_symmetry_resolved_page_entropy_ensemble(8, 4, 4, 1, 7, TOLERANCE),
            Err(QuantumInformationError::InvalidEnsembleSampleCount(1))
        ));
        assert!(matches!(
            sample_symmetry_resolved_page_entropy_ensemble(8, 9, 4, 8, 7, TOLERANCE),
            Err(QuantumInformationError::InvalidExcitationSector { .. })
        ));
        assert!(matches!(
            sample_symmetry_resolved_page_entropy_ensemble(8, 4, 7, 8, 7, TOLERANCE),
            Err(QuantumInformationError::DenseDimensionLimit { .. })
        ));
    }
}

/// Compensated harmonic sum H_n = sum_(k=1)^n 1/k.
fn harmonic_number(n: usize) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for k in 1..=n {
        let adjusted = 1.0 / k as f64 - compensation;
        let next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn validate_tolerance(t: f64) -> Result<(), QuantumInformationError> {
    if !t.is_finite() || t <= 0.0 {
        return Err(QuantumInformationError::InvalidTolerance(t));
    }
    Ok(())
}

/// Real symmetric embedding of a Hermitian matrix; duplicate eigenvalues are
/// paired after sorting to recover the physical spectrum.
fn hermitian_eigenvalues(
    matrix: &DensityMatrix,
    tolerance: f64,
) -> Result<Vec<f64>, QuantumInformationError> {
    let n = matrix.dimension;
    let m = n
        .checked_mul(2)
        .ok_or(QuantumInformationError::DimensionOverflow)?;
    let mut real = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..n {
            let z = matrix.entries[i * n + j];
            real[i * m + j] = z.re;
            real[i * m + n + j] = -z.im;
            real[(n + i) * m + j] = z.im;
            real[(n + i) * m + n + j] = z.re;
        }
    }
    let mut values = jacobi_eigenvalues(&mut real, m, tolerance)?;
    values.sort_by(f64::total_cmp);
    Ok(values
        .chunks_exact(2)
        .map(|p| 0.5 * (p[0] + p[1]))
        .collect())
}

/// Cyclic Jacobi sweeps have O(n^3) work per sweep, avoiding a full O(n^2)
/// maximum-pivot search before each individual rotation.
fn jacobi_eigenvalues(
    a: &mut [f64],
    n: usize,
    tol: f64,
) -> Result<Vec<f64>, QuantumInformationError> {
    const MAX_SWEEPS: usize = 80;
    let mut residual = f64::INFINITY;
    for _ in 0..MAX_SWEEPS {
        let mut scale: f64 = 1.0;
        residual = 0.0;
        for i in 0..n {
            scale = scale.max(a[i * n + i].abs());
            for j in (i + 1)..n {
                residual = residual.max(a[i * n + j].abs());
            }
        }
        if residual <= tol * scale {
            let mut values = (0..n).map(|i| a[i * n + i]).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            return Ok(values);
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= tol * scale / n as f64 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let sign = if tau < 0.0 { -1.0 } else { 1.0 };
                let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    let np = c * akp - s * akq;
                    let nq = s * akp + c * akq;
                    a[k * n + p] = np;
                    a[p * n + k] = np;
                    a[k * n + q] = nq;
                    a[q * n + k] = nq;
                }
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
            }
        }
    }
    Err(QuantumInformationError::EigensolverNonconvergence {
        sweeps: MAX_SWEEPS,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1e-10;

    #[test]
    fn product_state_reduces_to_a_pure_state() {
        let mut state = vec![Complex64::new(0.0, 0.0); 4];
        state[1] = Complex64::new(1.0, 0.0); // |01>
        let rho = DensityMatrix::from_pure_state(&state, 2, &[0], TOL).unwrap();
        assert!(rho.von_neumann_entropy(TOL).unwrap().abs() < 1e-12);
        assert!(rho.renyi_2_entropy(TOL).unwrap().abs() < 1e-12);
    }

    #[test]
    fn bell_pair_has_one_bit_of_entanglement() {
        let a = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![Complex64::new(0.0, 0.0); 4];
        state[0] = Complex64::new(a, 0.0);
        state[3] = Complex64::new(a, 0.0);
        let rho = DensityMatrix::from_pure_state(&state, 2, &[0], TOL).unwrap();
        assert!((rho.von_neumann_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
        assert!((rho.renyi_2_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
        assert!((rho.renyi_3_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
        assert!((rho.diagnostics(TOL).unwrap().purity - 0.5).abs() < 1e-12);
    }

    #[test]
    fn complex_phase_does_not_change_bell_pair_entropy() {
        let a = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![Complex64::new(0.0, 0.0); 4];
        state[0] = Complex64::new(a, 0.0);
        state[3] = Complex64::new(0.0, a);
        let rho = DensityMatrix::from_pure_state(&state, 2, &[1], TOL).unwrap();
        assert!((rho.von_neumann_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
        assert!(rho.hermiticity_residual() < 1e-14);
    }

    #[test]
    fn ghz_reduction_has_two_equal_nonzero_eigenvalues() {
        let a = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[0] = Complex64::new(a, 0.0);
        state[7] = Complex64::new(a, 0.0);
        let rho = DensityMatrix::from_pure_state(&state, 3, &[0, 1], TOL).unwrap();
        let e = rho.spectrum(TOL).unwrap();
        assert!(e[0].abs() < 1e-10 && e[1].abs() < 1e-10);
        assert!((e[2] - 0.5).abs() < 1e-10 && (e[3] - 0.5).abs() < 1e-10);
        assert!((rho.von_neumann_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn complementary_pure_state_reductions_have_equal_entropy() {
        let a = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[0] = Complex64::new(a, 0.0);
        state[7] = Complex64::new(a, 0.0);
        let left = DensityMatrix::from_pure_state(&state, 3, &[0], TOL).unwrap();
        let right = DensityMatrix::from_pure_state(&state, 3, &[1, 2], TOL).unwrap();
        assert!(
            (left.von_neumann_entropy(TOL).unwrap() - right.von_neumann_entropy(TOL).unwrap())
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn non_normalized_state_is_rejected_without_repair() {
        let state = vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)];
        assert!(matches!(
            DensityMatrix::from_pure_state(&state, 1, &[0], TOL),
            Err(QuantumInformationError::NormalizationError { .. })
        ));
    }

    #[test]
    fn duplicate_qubits_are_rejected() {
        let state = vec![Complex64::new(0.5, 0.0); 4];
        assert!(matches!(
            DensityMatrix::from_pure_state(&state, 2, &[0, 0], TOL),
            Err(QuantumInformationError::DuplicateQubit(0))
        ));
    }

    #[test]
    fn page_haar_average_matches_analytic_small_dimensions() {
        close_page(page_average_entropy_nats(1, 8).unwrap(), 0.0, 1e-14);
        close_page(page_average_entropy_nats(2, 2).unwrap(), 1.0 / 3.0, 1e-14);
        close_page(
            page_average_entropy_nats(2, 4).unwrap(),
            0.5095238095238095,
            1e-14,
        );
    }

    #[test]
    fn page_haar_average_is_symmetric_and_bounded_by_log_smaller_dimension() {
        for (m, n) in [(2, 8), (4, 16), (32, 32), (128, 64)] {
            let forward = page_average_entropy_nats(m, n).unwrap();
            let reverse = page_average_entropy_nats(n, m).unwrap();
            assert!((forward - reverse).abs() < 1e-13);
            assert!(forward <= (m.min(n) as f64).ln() + 1e-13);
            assert!(forward >= 0.0);
        }
    }

    #[test]
    fn haar_two_qubit_unitary_is_unitary_to_roundoff() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xA11CE);
        let unitary = haar_unitary_4(&mut rng).unwrap();
        for row in 0..4 {
            for column in 0..4 {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..4 {
                    dot += unitary[k][row].conj() * unitary[k][column];
                }
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((dot.re - expected).abs() < 1e-12);
                assert!(dot.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn zero_depth_random_circuit_leaves_product_state_unentangled() {
        for topology in [
            RandomCircuitTopology::Brickwork1D,
            RandomCircuitTopology::AllToAllMatching,
        ] {
            let sample =
                simulate_random_circuit_entropy(6, 0, topology, &[0, 1, 2], 42, TOL).unwrap();
            assert_eq!(sample.two_qubit_gate_count, 0);
            assert!((sample.state_norm_squared - 1.0).abs() < 1e-14);
            assert!(sample.subsystem_entropy_nats.abs() < 1e-12);
            assert!((sample.subsystem_purity - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_brickwork_circuit_is_seed_reproducible_and_norm_preserving() {
        let first = simulate_random_circuit_entropy(
            6,
            6,
            RandomCircuitTopology::Brickwork1D,
            &[0, 1, 2],
            0xB12C,
            TOL,
        )
        .unwrap();
        let repeated = simulate_random_circuit_entropy(
            6,
            6,
            RandomCircuitTopology::Brickwork1D,
            &[0, 1, 2],
            0xB12C,
            TOL,
        )
        .unwrap();
        assert_eq!(first, repeated);
        assert!((first.state_norm_squared - 1.0).abs() < 1e-12);
        assert!(first.subsystem_entropy_nats >= 0.0);
        assert!(first.subsystem_entropy_nats <= 3.0_f64.ln() + 1e-10);
        assert!(first.subsystem_purity > 0.0 && first.subsystem_purity <= 1.0 + 1e-10);
    }

    #[test]
    fn local_and_all_to_all_circuit_ensembles_report_seeded_statistics() {
        let local = sample_random_circuit_entropy_ensemble(
            6,
            4,
            RandomCircuitTopology::Brickwork1D,
            &[0, 1, 2],
            16,
            100,
            TOL,
        )
        .unwrap();
        let global = sample_random_circuit_entropy_ensemble(
            6,
            4,
            RandomCircuitTopology::AllToAllMatching,
            &[0, 1, 2],
            16,
            100,
            TOL,
        )
        .unwrap();
        assert_eq!(local.sample_count, 16);
        assert_eq!(global.sample_count, 16);
        assert!(local.sample_variance_nats_squared >= 0.0);
        assert!(global.sample_variance_nats_squared >= 0.0);
        assert!(local.standard_error_nats >= 0.0);
        assert!(global.standard_error_nats >= 0.0);
    }

    #[test]
    fn haar_page_ensemble_is_seed_reproducible_and_reports_uncertainty() {
        let first = sample_haar_page_entropy_ensemble(1, 3, 256, 0x5eed, TOL).unwrap();
        let repeated = sample_haar_page_entropy_ensemble(1, 3, 256, 0x5eed, TOL).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.subsystem_a_dimension, 2);
        assert_eq!(first.subsystem_b_dimension, 8);
        assert!(first.sample_variance_nats_squared > 0.0);
        assert!(first.standard_error_nats > 0.0);
        assert!(first.confidence_95_approx_nats[0] < first.confidence_95_approx_nats[1]);
        assert!((first.mean_von_neumann_entropy_nats - first.page_theorem_mean_nats).abs() < 0.10);
        assert!(first.normalized_mean_deviation.abs() < 6.0);
    }

    #[test]
    fn haar_ensemble_one_dimensional_subsystem_has_zero_entropy() {
        let result = sample_haar_page_entropy_ensemble(0, 3, 32, 7, TOL).unwrap();
        assert_eq!(result.page_theorem_mean_nats, 0.0);
        assert!(result.mean_von_neumann_entropy_nats.abs() < 1e-12);
        assert!(result.sample_variance_nats_squared.abs() < 1e-24);
    }

    #[test]
    fn haar_ensemble_rejects_insufficient_samples_and_oversized_registers() {
        assert!(matches!(
            sample_haar_page_entropy_ensemble(1, 2, 1, 7, TOL),
            Err(QuantumInformationError::InvalidEnsembleSampleCount(1))
        ));
        assert!(matches!(
            sample_haar_page_entropy_ensemble(1, MAX_DENSE_STATE_QUBITS, 2, 7, TOL),
            Err(QuantumInformationError::DenseStateQubitLimit { .. })
        ));
    }

    #[test]
    fn page_reference_rejects_invalid_or_oversized_dimensions() {
        assert!(matches!(
            page_average_entropy_nats(0, 4),
            Err(QuantumInformationError::InvalidDimension(0))
        ));
        assert!(matches!(
            page_average_entropy_nats(MAX_PAGE_REFERENCE_TOTAL_DIMENSION + 1, 1),
            Err(QuantumInformationError::PageReferenceDimensionLimit { .. })
        ));
    }

    fn close_page(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn dense_state_backend_rejects_above_existing_circuit_qubit_ceiling() {
        assert!(matches!(
            DensityMatrix::from_pure_state(&[], MAX_DENSE_STATE_QUBITS + 1, &[], TOL),
            Err(QuantumInformationError::DenseStateQubitLimit { .. })
        ));
    }

    #[test]
    fn negative_spectrum_is_rejected_instead_of_projected() {
        let rho = DensityMatrix::from_entries(
            2,
            vec![
                Complex64::new(1.1, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.1, 0.0),
            ],
        )
        .unwrap();
        assert!(matches!(
            rho.spectrum(TOL),
            Err(QuantumInformationError::NonPositive { .. })
        ));
    }
}

/// One checkpoint in a closed, charge-conserving sequential-emission toy model.
/// The global state remains pure; emitted tensor factors are never acted on
/// after their emission step. Entropies are evaluated on both sides of the
/// B|R cut while each side fits the dense reduced-state ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialEvaporationSample {
    pub emission_count: usize,
    pub newly_emitted_qubit: Option<usize>,
    /// Static tensor-factor labels still belonging to the active black-hole register.
    pub black_hole_qubits: Vec<usize>,
    /// Static tensor-factor labels collected as radiation, in emission order.
    pub radiation_qubits: Vec<usize>,
    pub radiation_entropy_nats: f64,
    pub black_hole_entropy_nats: f64,
    pub entropy_complementarity_residual_nats: f64,
    pub radiation_purity: f64,
    pub black_hole_purity: f64,
    pub state_norm_squared: f64,
    /// Expectation of N = sum_i |1><1|_i for equal-gap qubits.
    pub total_excitation_expectation: f64,
    pub total_excitation_variance: f64,
    /// Expected excitation number carried by each side of the B|R partition.
    pub radiation_excitation_expectation: f64,
    pub black_hole_excitation_expectation: f64,
    /// Absolute mismatch in the additive excitation ledger N = N_R + N_B.
    pub energy_partition_residual: f64,
    /// Occupation expectation of the factor emitted at this step, if any.
    pub newly_emitted_excitation_expectation: Option<f64>,
    pub two_qubit_gates_this_step: usize,
    pub total_two_qubit_gates: usize,
}

/// Reproducible finite-register sequential-emission experiment.
///
/// This is a U(1)-conserving qubit-chain toy model, not Hawking radiation:
/// the initial state is Haar-random inside a fixed excitation sector, a
/// charge-preserving two-qubit unitary acts on the active boundary pair, and
/// the first active tensor factor is then frozen into R. Later gates act only
/// on the remaining B factors. The equal-gap Hamiltonian is H = omega * N.
/// No gravitational dynamics, redshift, backreaction, or QES is represented.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialEvaporationResult {
    pub number_of_qubits: usize,
    pub initial_excitations: usize,
    pub seed: u64,
    pub total_two_qubit_gates: usize,
    pub samples: Vec<SequentialEvaporationSample>,
}

/// Simulate a sequential B -> B' tensor-factor transfer with exact U(1)
/// symmetry in every two-qubit gate.
///
/// The model is deliberately limited to 12 qubits so both complementary
/// reduced density matrices can be computed directly (dimension <= 64) at
/// every nontrivial cut. Use a tensor-network backend for larger registers.
pub fn simulate_u1_conserving_sequential_evaporation(
    number_of_qubits: usize,
    initial_excitations: usize,
    seed: u64,
    tolerance: f64,
) -> Result<SequentialEvaporationResult, QuantumInformationError> {
    validate_tolerance(tolerance)?;
    if number_of_qubits < 2 {
        return Err(QuantumInformationError::InvalidCircuitTopology);
    }
    if number_of_qubits > MAX_DENSE_STATE_QUBITS {
        return Err(QuantumInformationError::DenseStateQubitLimit {
            requested: number_of_qubits,
            maximum: MAX_DENSE_STATE_QUBITS,
        });
    }
    const MAX_COMPLEMENTARY_EVAPORATION_QUBITS: usize = 12;
    if number_of_qubits > MAX_COMPLEMENTARY_EVAPORATION_QUBITS {
        return Err(QuantumInformationError::EvaporationRegisterLimit {
            requested: number_of_qubits,
            maximum: MAX_COMPLEMENTARY_EVAPORATION_QUBITS,
        });
    }
    if initial_excitations > number_of_qubits {
        return Err(QuantumInformationError::InvalidExcitationSector {
            excitations: initial_excitations,
            number_of_qubits,
        });
    }

    let dimension = 1usize << number_of_qubits;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); dimension];

    // A normalized complex-Gaussian vector restricted to one Hamming-weight
    // sector is Haar distributed on that fixed-charge subspace.
    for (basis, amplitude) in amplitudes.iter_mut().enumerate() {
        if basis.count_ones() as usize == initial_excitations {
            *amplitude = complex_standard_normal(&mut rng);
        }
    }
    let initial_norm_squared: f64 = amplitudes.iter().map(|value| value.norm_sqr()).sum();
    if !initial_norm_squared.is_finite() || initial_norm_squared <= 0.0 {
        return Err(QuantumInformationError::NonFiniteState { index: dimension });
    }
    let inverse_initial_norm = initial_norm_squared.sqrt().recip();
    for amplitude in &mut amplitudes {
        *amplitude *= inverse_initial_norm;
    }

    let mut black_hole_qubits = (0..number_of_qubits).collect::<Vec<_>>();
    let mut radiation_qubits = Vec::with_capacity(number_of_qubits);
    let mut samples = Vec::with_capacity(number_of_qubits + 1);
    let mut total_two_qubit_gates = 0usize;

    for emission_count in 0..=number_of_qubits {
        let mut newly_emitted_qubit = None;
        let mut gates_this_step = 0usize;

        if emission_count > 0 {
            // Interact only the active boundary pair. The outgoing factor is
            // then removed from the active set before the next step.
            if black_hole_qubits.len() >= 2 {
                let gate = u1_conserving_unitary_4(&mut rng)?;
                apply_two_qubit_unitary(
                    &mut amplitudes,
                    number_of_qubits,
                    black_hole_qubits[0],
                    black_hole_qubits[1],
                    &gate,
                )?;
                gates_this_step = 1;
                total_two_qubit_gates += 1;
            }
            let emitted = black_hole_qubits.remove(0);
            radiation_qubits.push(emitted);
            newly_emitted_qubit = Some(emitted);
        }

        let state_norm_squared: f64 = amplitudes.iter().map(|value| value.norm_sqr()).sum();
        if !state_norm_squared.is_finite() {
            return Err(QuantumInformationError::NonFiniteState { index: dimension });
        }
        if (state_norm_squared - 1.0).abs() > tolerance {
            return Err(QuantumInformationError::NormalizationError {
                norm_squared: state_norm_squared,
                tolerance,
            });
        }

        let (radiation_entropy_nats, black_hole_entropy_nats, radiation_purity, black_hole_purity) =
            if radiation_qubits.is_empty() || black_hole_qubits.is_empty() {
                // The unpartitioned global state is pure at the two endpoints.
                (0.0, 0.0, 1.0, 1.0)
            } else {
                let radiation = DensityMatrix::from_pure_state(
                    &amplitudes,
                    number_of_qubits,
                    &radiation_qubits,
                    tolerance,
                )?;
                let black_hole = DensityMatrix::from_pure_state(
                    &amplitudes,
                    number_of_qubits,
                    &black_hole_qubits,
                    tolerance,
                )?;
                (
                    radiation.von_neumann_entropy(tolerance)?,
                    black_hole.von_neumann_entropy(tolerance)?,
                    radiation.diagnostics(tolerance)?.purity,
                    black_hole.diagnostics(tolerance)?.purity,
                )
            };

        let (total_excitation_expectation, total_excitation_variance) =
            total_excitation_moments(&amplitudes);
        let radiation_excitation_expectation =
            subsystem_excitation_expectation(&amplitudes, number_of_qubits, &radiation_qubits);
        let black_hole_excitation_expectation =
            subsystem_excitation_expectation(&amplitudes, number_of_qubits, &black_hole_qubits);
        let energy_partition_residual = (total_excitation_expectation
            - radiation_excitation_expectation
            - black_hole_excitation_expectation)
            .abs();
        let newly_emitted_excitation_expectation = newly_emitted_qubit
            .map(|qubit| subsystem_excitation_expectation(&amplitudes, number_of_qubits, &[qubit]));
        samples.push(SequentialEvaporationSample {
            emission_count,
            newly_emitted_qubit,
            black_hole_qubits: black_hole_qubits.clone(),
            radiation_qubits: radiation_qubits.clone(),
            radiation_entropy_nats,
            black_hole_entropy_nats,
            entropy_complementarity_residual_nats: (radiation_entropy_nats
                - black_hole_entropy_nats)
                .abs(),
            radiation_purity,
            black_hole_purity,
            state_norm_squared,
            total_excitation_expectation,
            total_excitation_variance,
            radiation_excitation_expectation,
            black_hole_excitation_expectation,
            energy_partition_residual,
            newly_emitted_excitation_expectation,
            two_qubit_gates_this_step: gates_this_step,
            total_two_qubit_gates,
        });
    }

    Ok(SequentialEvaporationResult {
        number_of_qubits,
        initial_excitations,
        seed,
        total_two_qubit_gates,
        samples,
    })
}

/// Draw a unit-variance circular complex Gaussian from two independent
/// standard normal variates. Overall scale cancels in Haar normalization.
fn complex_standard_normal(rng: &mut ChaCha20Rng) -> Complex64 {
    let u1 = rng.gen_range(0.0_f64..1.0_f64).max(f64::MIN_POSITIVE);
    let u2 = rng.gen_range(0.0_f64..1.0_f64);
    let radius = (-2.0 * u1.ln()).sqrt();
    let phase = core::f64::consts::TAU * u2;
    Complex64::new(radius * phase.cos(), radius * phase.sin())
}

/// Haar-random element of U(1) x U(2) x U(1), embedded in the two-qubit
/// computational basis. It commutes exactly with N_pair = n_1 + n_2.
fn u1_conserving_unitary_4(
    rng: &mut ChaCha20Rng,
) -> Result<[[Complex64; 4]; 4], QuantumInformationError> {
    let mut columns = [[Complex64::new(0.0, 0.0); 2]; 2];
    for column in 0..2 {
        let mut accepted = false;
        for _attempt in 0..8 {
            let mut vector = [complex_standard_normal(rng), complex_standard_normal(rng)];
            if column == 1 {
                let overlap = columns[0][0].conj() * vector[0] + columns[0][1].conj() * vector[1];
                vector[0] -= columns[0][0] * overlap;
                vector[1] -= columns[0][1] * overlap;
            }
            let norm_squared = vector[0].norm_sqr() + vector[1].norm_sqr();
            if norm_squared.is_finite() && norm_squared > 1e-24 {
                let inverse_norm = norm_squared.sqrt().recip();
                columns[column][0] = vector[0] * inverse_norm;
                columns[column][1] = vector[1] * inverse_norm;
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(QuantumInformationError::UnitaryConstructionFailure);
        }
    }

    let mut unitary = [[Complex64::new(0.0, 0.0); 4]; 4];
    for sector_state in 0..2 {
        let phase = core::f64::consts::TAU * rng.gen_range(0.0_f64..1.0_f64);
        let value = Complex64::new(phase.cos(), phase.sin());
        if sector_state == 0 {
            unitary[0][0] = value;
        } else {
            unitary[3][3] = value;
        }
    }
    // The one-excitation sector is spanned by |01>, |10>.
    for row in 0..2 {
        for column in 0..2 {
            unitary[row + 1][column + 1] = columns[column][row];
        }
    }
    Ok(unitary)
}

/// Expected number of excitations in a selected tensor-factor subsystem.
fn subsystem_excitation_expectation(
    amplitudes: &[Complex64],
    number_of_qubits: usize,
    subsystem_qubits: &[usize],
) -> f64 {
    let masks = subsystem_qubits
        .iter()
        .map(|qubit| 1usize << (number_of_qubits - 1 - qubit))
        .collect::<Vec<_>>();
    amplitudes
        .iter()
        .enumerate()
        .map(|(basis, amplitude)| {
            amplitude.norm_sqr() * masks.iter().filter(|mask| basis & **mask != 0).count() as f64
        })
        .sum()
}

/// Expected excitation number and variance of the computational-basis
/// number operator. A fixed-sector initial state should retain zero variance.
fn total_excitation_moments(amplitudes: &[Complex64]) -> (f64, f64) {
    let mut mean = 0.0;
    let mut second_moment = 0.0;
    for (basis, amplitude) in amplitudes.iter().enumerate() {
        let probability = amplitude.norm_sqr();
        let excitations = basis.count_ones() as f64;
        mean += probability * excitations;
        second_moment += probability * excitations * excitations;
    }
    (mean, (second_moment - mean * mean).max(0.0))
}

#[cfg(test)]
mod sequential_evaporation_tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn u1_gate_is_unitary_and_has_no_cross_charge_matrix_elements() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0A5E);
        let gate = u1_conserving_unitary_4(&mut rng).unwrap();
        for row in 0usize..4 {
            for column in 0usize..4 {
                let row_charge = row.count_ones() as usize;
                let column_charge = column.count_ones() as usize;
                if row_charge != column_charge {
                    assert!(gate[row][column].norm() < 1e-14);
                }
            }
        }
        for left in 0..4 {
            for right in 0..4 {
                let mut overlap = Complex64::new(0.0, 0.0);
                for row in 0..4 {
                    overlap += gate[row][left].conj() * gate[row][right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((overlap.re - expected).abs() < 1e-12);
                assert!(overlap.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sequential_emission_preserves_charge_norm_and_complementary_entropy() {
        let result =
            simulate_u1_conserving_sequential_evaporation(8, 4, 0xE_AA0, TOLERANCE).unwrap();
        assert_eq!(result.samples.len(), 9);
        assert_eq!(result.total_two_qubit_gates, 7);
        for (step, sample) in result.samples.iter().enumerate() {
            assert_eq!(sample.emission_count, step);
            assert_eq!(
                sample.black_hole_qubits.len() + sample.radiation_qubits.len(),
                8,
            );
            assert!((sample.state_norm_squared - 1.0).abs() < TOLERANCE);
            assert!((sample.total_excitation_expectation - 4.0).abs() < TOLERANCE);
            assert!(sample.total_excitation_variance < TOLERANCE);
            assert!(
                (sample.radiation_excitation_expectation
                    + sample.black_hole_excitation_expectation
                    - 4.0)
                    .abs()
                    < TOLERANCE
            );
            assert!(sample.energy_partition_residual < TOLERANCE);
            assert_eq!(
                sample.newly_emitted_excitation_expectation.is_some(),
                step > 0,
            );
            if let Some(occupation) = sample.newly_emitted_excitation_expectation {
                assert!((0.0..=1.0).contains(&occupation));
            }
            assert!(sample.entropy_complementarity_residual_nats < TOLERANCE);
            assert!(sample.radiation_entropy_nats >= -TOLERANCE);
            assert!(sample.black_hole_entropy_nats >= -TOLERANCE);
        }
    }

    #[test]
    fn emitted_qubits_are_a_monotonic_disjoint_prefix_and_gate_budget_is_exact() {
        let result =
            simulate_u1_conserving_sequential_evaporation(7, 3, 0x51A7E, TOLERANCE).unwrap();
        for (step, sample) in result.samples.iter().enumerate() {
            assert_eq!(sample.radiation_qubits.len(), step);
            assert_eq!(sample.black_hole_qubits.len(), 7 - step);
            let mut partition = vec![false; 7];
            for &qubit in &sample.radiation_qubits {
                assert!(
                    !partition[qubit],
                    "radiation contains a duplicate qubit {qubit}"
                );
                partition[qubit] = true;
            }
            for &qubit in &sample.black_hole_qubits {
                assert!(
                    !partition[qubit],
                    "an emitted qubit re-entered the active register"
                );
                partition[qubit] = true;
            }
            assert!(partition.into_iter().all(|present| present));
            assert_eq!(
                sample.two_qubit_gates_this_step,
                if step > 0 && step < 7 { 1 } else { 0 },
            );
        }
        assert_eq!(
            result.samples.last().unwrap().radiation_qubits,
            (0..7).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn vacuum_and_fully_filled_sectors_remain_unentangled() {
        for excitations in [0, 6] {
            let result = simulate_u1_conserving_sequential_evaporation(
                6,
                excitations,
                0xF00D + excitations as u64,
                TOLERANCE,
            )
            .unwrap();
            for sample in result.samples {
                assert!(sample.radiation_entropy_nats.abs() < TOLERANCE);
                assert!(sample.black_hole_entropy_nats.abs() < TOLERANCE);
                assert!((sample.radiation_purity - 1.0).abs() < TOLERANCE);
                assert!((sample.black_hole_purity - 1.0).abs() < TOLERANCE);
            }
        }
    }

    #[test]
    fn sequential_model_rejects_unsupported_register_or_charge_sector() {
        assert!(matches!(
            simulate_u1_conserving_sequential_evaporation(13, 6, 1, TOLERANCE),
            Err(QuantumInformationError::EvaporationRegisterLimit { .. })
        ));
        assert!(matches!(
            simulate_u1_conserving_sequential_evaporation(8, 9, 1, TOLERANCE),
            Err(QuantumInformationError::InvalidExcitationSector { .. })
        ));
    }

    #[test]
    fn sequential_model_is_reproducible_for_a_frozen_seed() {
        let first =
            simulate_u1_conserving_sequential_evaporation(8, 3, 0xDEC0DE, TOLERANCE).unwrap();
        let repeated =
            simulate_u1_conserving_sequential_evaporation(8, 3, 0xDEC0DE, TOLERANCE).unwrap();
        assert_eq!(first, repeated);
    }
}

#[cfg(test)]
mod symmetry_resolved_page_tests {
    use super::*;

    #[test]
    fn fixed_charge_page_reference_has_exact_small_sector_answers() {
        assert!(
            (symmetry_resolved_page_average_entropy_nats(2, 1, 1).unwrap() - 0.5).abs() < 1e-14
        );
        assert!(
            (symmetry_resolved_page_average_entropy_nats(4, 1, 2).unwrap() - 7.0 / 12.0).abs()
                < 1e-14
        );
        assert!(
            (symmetry_resolved_page_average_entropy_nats(4, 2, 2).unwrap() - 77.0 / 60.0).abs()
                < 1e-13
        );
    }

    #[test]
    fn fixed_charge_page_reference_is_complement_symmetric() {
        for (number_of_qubits, total_excitations, subsystem) in
            [(8, 4, 1), (8, 4, 3), (10, 5, 4), (12, 6, 5)]
        {
            let left = symmetry_resolved_page_average_entropy_nats(
                number_of_qubits,
                total_excitations,
                subsystem,
            )
            .unwrap();
            let right = symmetry_resolved_page_average_entropy_nats(
                number_of_qubits,
                total_excitations,
                number_of_qubits - subsystem,
            )
            .unwrap();
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[test]
    fn fixed_charge_page_reference_vanishes_for_trivial_sectors_and_cuts() {
        for (number_of_qubits, total_excitations, subsystem) in
            [(8, 0, 4), (8, 8, 4), (8, 4, 0), (8, 4, 8)]
        {
            assert!(
                symmetry_resolved_page_average_entropy_nats(
                    number_of_qubits,
                    total_excitations,
                    subsystem,
                )
                .unwrap()
                .abs()
                    < 1e-13
            );
        }
    }

    #[test]
    fn fixed_charge_page_reference_rejects_invalid_domain_inputs() {
        assert!(matches!(
            symmetry_resolved_page_average_entropy_nats(8, 9, 4),
            Err(QuantumInformationError::InvalidExcitationSector { .. })
        ));
        assert!(matches!(
            symmetry_resolved_page_average_entropy_nats(8, 4, 9),
            Err(QuantumInformationError::InvalidSubsystemSize { .. })
        ));
    }
}
