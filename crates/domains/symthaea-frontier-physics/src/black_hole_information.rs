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
    DimensionMismatch { expected: usize, actual: usize },
    DenseDimensionLimit { requested: usize, maximum: usize },
    DenseStateQubitLimit { requested: usize, maximum: usize },
    PageReferenceDimensionLimit { requested: usize, maximum: usize },
    InvalidPageEntropy(f64),
    InvalidEnsembleSampleCount(usize),
    InvalidCircuitTopology,
    UnitaryConstructionFailure,
    InvalidTolerance(f64),
    NonFiniteState { index: usize },
    NonFiniteMatrix { row: usize, column: usize },
    InvalidQubit { qubit: usize, register_size: usize },
    DuplicateQubit(usize),
    NormalizationError { norm_squared: f64, tolerance: f64 },
    NonHermitian { residual: f64, tolerance: f64 },
    TraceError { real: f64, imaginary: f64, tolerance: f64 },
    NonPositive { minimum_eigenvalue: f64, tolerance: f64 },
    EigensolverNonconvergence { sweeps: usize, residual: f64 },
}

impl fmt::Display for QuantumInformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(d) => write!(f, "invalid Hilbert-space dimension {d}"),
            Self::DimensionOverflow => write!(f, "Hilbert-space dimension overflow"),
            Self::DimensionMismatch { expected, actual } => write!(f, "dimension mismatch: expected {expected}, got {actual}"),
            Self::DenseDimensionLimit { requested, maximum } => write!(f, "dense dimension {requested} exceeds limit {maximum}"),
            Self::DenseStateQubitLimit { requested, maximum } => write!(f, "dense state with {requested} qubits exceeds limit {maximum}"),
            Self::PageReferenceDimensionLimit { requested, maximum } => write!(f, "Page reference dimension product {requested} exceeds limit {maximum}"),
            Self::InvalidPageEntropy(value) => write!(f, "Page reference entropy is invalid: {value}"),
            Self::InvalidEnsembleSampleCount(count) => write!(f, "Haar ensemble needs at least two samples, got {count}"),
            Self::InvalidCircuitTopology => write!(f, "circuit topology requires at least two qubits"),
            Self::UnitaryConstructionFailure => write!(f, "failed to construct a numerically independent Haar unitary basis"),
            Self::InvalidTolerance(t) => write!(f, "invalid tolerance {t}"),
            Self::NonFiniteState { index } => write!(f, "non-finite state amplitude at index {index}"),
            Self::NonFiniteMatrix { row, column } => write!(f, "non-finite matrix entry ({row}, {column})"),
            Self::InvalidQubit { qubit, register_size } => write!(f, "qubit {qubit} is outside a {register_size}-qubit register"),
            Self::DuplicateQubit(q) => write!(f, "qubit {q} appears more than once"),
            Self::NormalizationError { norm_squared, tolerance } => write!(f, "state norm squared {norm_squared} differs from one beyond {tolerance}"),
            Self::NonHermitian { residual, tolerance } => write!(f, "Hermiticity residual {residual} exceeds {tolerance}"),
            Self::TraceError { real, imaginary, tolerance } => write!(f, "trace ({real}, {imaginary}) differs from one beyond {tolerance}"),
            Self::NonPositive { minimum_eigenvalue, tolerance } => write!(f, "minimum eigenvalue {minimum_eigenvalue} is below -{tolerance}"),
            Self::EigensolverNonconvergence { sweeps, residual } => write!(f, "Jacobi eigensolver did not converge after {sweeps} sweeps (residual {residual})"),
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
    pub fn from_entries(dimension: usize, entries: Vec<Complex64>) -> Result<Self, QuantumInformationError> {
        if dimension == 0 { return Err(QuantumInformationError::InvalidDimension(dimension)); }
        if dimension > MAX_DENSE_REDUCED_DIMENSION {
            return Err(QuantumInformationError::DenseDimensionLimit { requested: dimension, maximum: MAX_DENSE_REDUCED_DIMENSION });
        }
        let expected = dimension.checked_mul(dimension).ok_or(QuantumInformationError::DimensionOverflow)?;
        if entries.len() != expected {
            return Err(QuantumInformationError::DimensionMismatch { expected, actual: entries.len() });
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

    pub const fn dimension(&self) -> usize { self.dimension }
    pub fn entries(&self) -> &[Complex64] { &self.entries }

    pub fn get(&self, row: usize, column: usize) -> Option<Complex64> {
        if row < self.dimension && column < self.dimension {
            Some(self.entries[row * self.dimension + column])
        } else { None }
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
        if number_of_qubits >= usize::BITS as usize { return Err(QuantumInformationError::DimensionOverflow); }
        let full_dimension = 1usize << number_of_qubits;
        if amplitudes.len() != full_dimension {
            return Err(QuantumInformationError::DimensionMismatch { expected: full_dimension, actual: amplitudes.len() });
        }
        let mut norm_squared = 0.0;
        for (index, a) in amplitudes.iter().enumerate() {
            if !a.re.is_finite() || !a.im.is_finite() { return Err(QuantumInformationError::NonFiniteState { index }); }
            norm_squared += a.norm_sqr();
        }
        if !norm_squared.is_finite() { return Err(QuantumInformationError::NonFiniteState { index: amplitudes.len() }); }
        if (norm_squared - 1.0).abs() > normalization_tolerance {
            return Err(QuantumInformationError::NormalizationError { norm_squared, tolerance: normalization_tolerance });
        }

        let mut retained = vec![false; number_of_qubits];
        for &qubit in retained_qubits {
            if qubit >= number_of_qubits {
                return Err(QuantumInformationError::InvalidQubit { qubit, register_size: number_of_qubits });
            }
            if retained[qubit] { return Err(QuantumInformationError::DuplicateQubit(qubit)); }
            retained[qubit] = true;
        }
        if retained_qubits.len() >= usize::BITS as usize { return Err(QuantumInformationError::DimensionOverflow); }
        let reduced_dimension = 1usize << retained_qubits.len();
        if reduced_dimension > MAX_DENSE_REDUCED_DIMENSION {
            return Err(QuantumInformationError::DenseDimensionLimit { requested: reduced_dimension, maximum: MAX_DENSE_REDUCED_DIMENSION });
        }
        let environment_qubits = number_of_qubits - retained_qubits.len();
        if environment_qubits >= usize::BITS as usize { return Err(QuantumInformationError::DimensionOverflow); }
        let environment_dimension = 1usize << environment_qubits;

        // Regroup amplitudes as psi[environment, retained], then form
        // rho_A = sum_e |psi_e><psi_e|.
        let mut blocks = vec![vec![Complex64::new(0.0, 0.0); reduced_dimension]; environment_dimension];
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
                for block in &blocks { value += block[row] * block[column].conj(); }
                entries[row * reduced_dimension + column] = value;
            }
        }
        Self::from_entries(reduced_dimension, entries)
    }

    pub fn trace(&self) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..self.dimension { result += self.entries[i * self.dimension + i]; }
        result
    }

    pub fn hermiticity_residual(&self) -> f64 {
        let mut residual: f64 = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                residual = residual.max((self.entries[i * self.dimension + j] - self.entries[j * self.dimension + i].conj()).norm());
            }
        }
        residual
    }

    pub fn spectrum(&self, tolerance: f64) -> Result<Vec<f64>, QuantumInformationError> {
        Ok(self.checked_diagnostics(tolerance)?.eigenvalues)
    }

    pub fn diagnostics(&self, tolerance: f64) -> Result<DensityDiagnostics, QuantumInformationError> {
        self.checked_diagnostics(tolerance)
    }

    /// Von Neumann entropy in nats. Eigenvalues in the permitted roundoff
    /// interval [-tolerance, 0] contribute zero; larger negative values reject.
    pub fn von_neumann_entropy(&self, tolerance: f64) -> Result<f64, QuantumInformationError> {
        Ok(self.spectrum(tolerance)?.into_iter().filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum())
    }

    /// Second Renyi entropy in nats.
    pub fn renyi_2_entropy(&self, tolerance: f64) -> Result<f64, QuantumInformationError> {
        let d = self.checked_diagnostics(tolerance)?;
        if d.purity <= 0.0 || !d.purity.is_finite() {
            return Err(QuantumInformationError::NonPositive { minimum_eigenvalue: d.minimum_eigenvalue, tolerance });
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
                for k in 0..n { square[i * n + j] += self.entries[i * n + k] * self.entries[k * n + j]; }
            }
        }
        let mut trace_cube = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n { trace_cube += square[i * n + j] * self.entries[j * n + i]; }
        }
        if !trace_cube.re.is_finite() || !trace_cube.im.is_finite() || trace_cube.re <= 0.0 || trace_cube.im.abs() > tolerance {
            return Err(QuantumInformationError::TraceError { real: trace_cube.re, imaginary: trace_cube.im, tolerance });
        }
        Ok(-0.5 * trace_cube.re.ln())
    }

    fn checked_diagnostics(&self, tolerance: f64) -> Result<DensityDiagnostics, QuantumInformationError> {
        validate_tolerance(tolerance)?;
        let trace = self.trace();
        if (trace.re - 1.0).abs() > tolerance || trace.im.abs() > tolerance {
            return Err(QuantumInformationError::TraceError { real: trace.re, imaginary: trace.im, tolerance });
        }
        let hermiticity_residual = self.hermiticity_residual();
        if hermiticity_residual > tolerance {
            return Err(QuantumInformationError::NonHermitian { residual: hermiticity_residual, tolerance });
        }
        let eigenvalues = hermitian_eigenvalues(self, (tolerance * 0.1).max(1e-14))?;
        let minimum_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if minimum_eigenvalue < -tolerance {
            return Err(QuantumInformationError::NonPositive { minimum_eigenvalue, tolerance });
        }
        let purity: f64 = self.entries.iter().map(|x| x.norm_sqr()).sum();
        if !purity.is_finite() { return Err(QuantumInformationError::NonFiniteMatrix { row: 0, column: 0 }); }
        Ok(DensityDiagnostics { trace, hermiticity_residual, purity, eigenvalues, minimum_eigenvalue })
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
fn haar_unitary_4(
    rng: &mut ChaCha20Rng,
) -> Result<[[Complex64; 4]; 4], QuantumInformationError> {
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
    if first_qubit >= number_of_qubits || second_qubit >= number_of_qubits
        || first_qubit == second_qubit
    {
        return Err(QuantumInformationError::InvalidQubit {
            qubit: first_qubit.max(second_qubit),
            register_size: number_of_qubits,
        });
    }
    let expected = 1usize.checked_shl(number_of_qubits as u32)
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
                qubits.chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect::<Vec<_>>()
            }
        };
        for (first, second) in pairs {
            let unitary = haar_unitary_4(&mut rng)?;
            apply_two_qubit_unitary(
                &mut amplitudes,
                number_of_qubits,
                first,
                second,
                &unitary,
            )?;
            gate_count += 1;
        }
    }

    let norm_squared: f64 = amplitudes.iter().map(|amplitude| amplitude.norm_sqr()).sum();
    if !norm_squared.is_finite() {
        return Err(QuantumInformationError::NonFiniteState { index: dimension });
    }
    if (norm_squared - 1.0).abs() > tolerance {
        return Err(QuantumInformationError::NormalizationError {
            norm_squared,
            tolerance,
        });
    }
    let reduced = DensityMatrix::from_pure_state(
        &amplitudes,
        number_of_qubits,
        retained_qubits,
        tolerance,
    )?;
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
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(sample_count));
    }
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    for sample_index in 0..sample_count {
        let seed = first_seed.wrapping_add(sample_index as u64);
        let sample = simulate_random_circuit_entropy(
            number_of_qubits, depth, topology, retained_qubits, seed, tolerance,
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
        return Err(QuantumInformationError::InvalidEnsembleSampleCount(sample_count));
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
    if subsystem_a_qubits >= usize::BITS as usize
        || subsystem_b_qubits >= usize::BITS as usize
    {
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
        let norm_squared: f64 = amplitudes.iter().map(|amplitude| amplitude.norm_sqr()).sum();
        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            return Err(QuantumInformationError::NonFiniteState { index: sample_index });
        }
        let inverse_norm = norm_squared.sqrt().recip();
        for amplitude in &mut amplitudes {
            *amplitude *= inverse_norm;
        }

        let reduced = DensityMatrix::from_pure_state(
            &amplitudes,
            total_qubits,
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
        confidence_95_approx_nats: [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        page_theorem_mean_nats: page_mean,
        normalized_mean_deviation: normalized_deviation,
    })
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
    if !t.is_finite() || t <= 0.0 { return Err(QuantumInformationError::InvalidTolerance(t)); }
    Ok(())
}

/// Real symmetric embedding of a Hermitian matrix; duplicate eigenvalues are
/// paired after sorting to recover the physical spectrum.
fn hermitian_eigenvalues(matrix: &DensityMatrix, tolerance: f64) -> Result<Vec<f64>, QuantumInformationError> {
    let n = matrix.dimension;
    let m = n.checked_mul(2).ok_or(QuantumInformationError::DimensionOverflow)?;
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
    Ok(values.chunks_exact(2).map(|p| 0.5 * (p[0] + p[1])).collect())
}

/// Cyclic Jacobi sweeps have O(n^3) work per sweep, avoiding a full O(n^2)
/// maximum-pivot search before each individual rotation.
fn jacobi_eigenvalues(a: &mut [f64], n: usize, tol: f64) -> Result<Vec<f64>, QuantumInformationError> {
    const MAX_SWEEPS: usize = 80;
    let mut residual: f64;
    for _ in 0..MAX_SWEEPS {
        let mut scale: f64 = 1.0;
        residual = 0.0;
        for i in 0..n {
            scale = scale.max(a[i * n + i].abs());
            for j in (i + 1)..n { residual = residual.max(a[i * n + j].abs()); }
        }
        if residual <= tol * scale {
            let mut values = (0..n).map(|i| a[i * n + i]).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            return Ok(values);
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= tol * scale / n as f64 { continue; }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let sign = if tau < 0.0 { -1.0 } else { 1.0 };
                let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    if k == p || k == q { continue; }
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    let np = c * akp - s * akq;
                    let nq = s * akp + c * akq;
                    a[k * n + p] = np; a[p * n + k] = np;
                    a[k * n + q] = nq; a[q * n + k] = nq;
                }
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0; a[q * n + p] = 0.0;
            }
        }
    }
    Err(QuantumInformationError::EigensolverNonconvergence { sweeps: MAX_SWEEPS, residual })
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
        state[0] = Complex64::new(a, 0.0); state[3] = Complex64::new(a, 0.0);
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
        state[0] = Complex64::new(a, 0.0); state[3] = Complex64::new(0.0, a);
        let rho = DensityMatrix::from_pure_state(&state, 2, &[1], TOL).unwrap();
        assert!((rho.von_neumann_entropy(TOL).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
        assert!(rho.hermiticity_residual() < 1e-14);
    }

    #[test]
    fn ghz_reduction_has_two_equal_nonzero_eigenvalues() {
        let a = 1.0 / 2.0_f64.sqrt();
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[0] = Complex64::new(a, 0.0); state[7] = Complex64::new(a, 0.0);
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
        assert!((left.von_neumann_entropy(TOL).unwrap()
            - right.von_neumann_entropy(TOL).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn non_normalized_state_is_rejected_without_repair() {
        let state = vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)];
        assert!(matches!(DensityMatrix::from_pure_state(&state, 1, &[0], TOL),
            Err(QuantumInformationError::NormalizationError { .. })));
    }

    #[test]
    fn duplicate_qubits_are_rejected() {
        let state = vec![Complex64::new(0.5, 0.0); 4];
        assert!(matches!(DensityMatrix::from_pure_state(&state, 2, &[0, 0], TOL),
            Err(QuantumInformationError::DuplicateQubit(0))));
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
    fn random_brickwork_circuit_is_seed_reproducible_and_norm_preserving() {
        let first = simulate_random_circuit_entropy(
            6, 6, RandomCircuitTopology::Brickwork1D, &[0, 1, 2], 0xB12C, TOL,
        ).unwrap();
        let repeated = simulate_random_circuit_entropy(
            6, 6, RandomCircuitTopology::Brickwork1D, &[0, 1, 2], 0xB12C, TOL,
        ).unwrap();
        assert_eq!(first, repeated);
        assert!((first.state_norm_squared - 1.0).abs() < 1e-12);
        assert!(first.subsystem_entropy_nats >= 0.0);
        assert!(first.subsystem_entropy_nats <= 3.0_f64.ln() + 1e-10);
        assert!(first.subsystem_purity > 0.0 && first.subsystem_purity <= 1.0 + 1e-10);
    }

    #[test]
    fn local_and_all_to_all_circuit_ensembles_report_seeded_statistics() {
        let local = sample_random_circuit_entropy_ensemble(
            6, 4, RandomCircuitTopology::Brickwork1D, &[0, 1, 2], 16, 100, TOL,
        ).unwrap();
        let global = sample_random_circuit_entropy_ensemble(
            6, 4, RandomCircuitTopology::AllToAllMatching, &[0, 1, 2], 16, 100, TOL,
        ).unwrap();
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
        let rho = DensityMatrix::from_entries(2, vec![
            Complex64::new(1.1, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-0.1, 0.0),
        ]).unwrap();
        assert!(matches!(rho.spectrum(TOL), Err(QuantumInformationError::NonPositive { .. })));
    }
}
