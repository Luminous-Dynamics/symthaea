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

pub const MAX_DENSE_REDUCED_DIMENSION: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumInformationError {
    InvalidDimension(usize),
    DimensionOverflow,
    DimensionMismatch { expected: usize, actual: usize },
    DenseDimensionLimit { requested: usize, maximum: usize },
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

    /// Von Neumann entropy in nats.
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
    let mut residual = f64::INFINITY;
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
    fn negative_spectrum_is_rejected_instead_of_projected() {
        let rho = DensityMatrix::from_entries(2, vec![
            Complex64::new(1.1, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-0.1, 0.0),
        ]).unwrap();
        assert!(matches!(rho.spectrum(TOL), Err(QuantumInformationError::NonPositive { .. })));
    }
}
