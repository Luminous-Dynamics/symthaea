// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generalized eigenvalue solver for the Roothaan-Hall equation: FC = SCε.
//!
//! Uses **canonical orthogonalization** (not Cholesky) to transform to a standard
//! eigenvalue problem. This is critical because:
//!
//! 1. Cholesky requires S to be strictly positive definite, but basis sets with
//!    near-linear dependence can have eigenvalues of S approaching zero.
//! 2. Canonical orthogonalization diagonalizes S, discards eigenvectors below a
//!    threshold (default 1e-6), and builds the transformation matrix X from the
//!    survivors. This is numerically stable even with large basis sets.
//!
//! The procedure:
//! 1. Diagonalize S: S = UsU^T (eigendecompose)
//! 2. Discard eigenvalues s_i < threshold
//! 3. Form X = U * s^{-1/2} (surviving columns only)
//! 4. Transform: F' = X^T F X
//! 5. Solve standard eigenproblem: F'C' = C'ε
//! 6. Back-transform: C = X C'
//!
//! Reference: Szabo & Ostlund, "Modern Quantum Chemistry", Section 3.4.5.

use crate::constants::CANONICAL_ORTH_THRESHOLD;

/// Result of the generalized eigenvalue problem FC = SCε.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenResult {
    /// Eigenvalues (orbital energies) in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (MO coefficients) as column-major: C[μ, i] = coefficients[μ * n_mo + i]
    /// where μ indexes AOs and i indexes MOs.
    pub coefficients: Vec<f64>,
    /// Number of basis functions retained after linear dependence removal.
    pub n_independent: usize,
    /// Number of basis functions discarded.
    pub n_discarded: usize,
    /// The orthogonalization matrix X (n_basis × n_independent).
    pub orthogonalization_matrix: Vec<f64>,
}

/// Compute the canonical orthogonalization matrix X from overlap matrix S.
///
/// Returns (X, n_independent, n_discarded) where X is n_basis × n_independent.
pub fn canonical_orthogonalization(s_matrix: &[f64], n: usize) -> (Vec<f64>, usize, usize) {
    canonical_orthogonalization_with_threshold(s_matrix, n, CANONICAL_ORTH_THRESHOLD)
}

/// Compute canonical orthogonalization with custom threshold.
pub fn canonical_orthogonalization_with_threshold(
    s_matrix: &[f64],
    n: usize,
    threshold: f64,
) -> (Vec<f64>, usize, usize) {
    // Step 1: Diagonalize S using Jacobi eigendecomposition
    let (eigenvalues, eigenvectors) = symmetric_eigen(s_matrix, n);

    // Step 2: Find surviving eigenvectors (eigenvalue >= threshold)
    let mut survivors: Vec<usize> = Vec::new();
    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev >= threshold {
            survivors.push(i);
        }
    }

    let n_independent = survivors.len();
    let n_discarded = n - n_independent;

    // Step 3: Build X = U * s^{-1/2} for surviving columns
    // X[μ, k] = U[μ, survivors[k]] / sqrt(eigenvalues[survivors[k]])
    let mut x = vec![0.0; n * n_independent];
    for (k, &idx) in survivors.iter().enumerate() {
        let inv_sqrt = 1.0 / eigenvalues[idx].sqrt();
        for mu in 0..n {
            x[mu * n_independent + k] = eigenvectors[mu * n + idx] * inv_sqrt;
        }
    }

    (x, n_independent, n_discarded)
}

/// Solve the generalized eigenvalue problem FC = SCε.
///
/// Takes the Fock matrix F and the pre-computed orthogonalization matrix X.
/// Returns eigenvalues and eigenvectors in the original AO basis.
pub fn solve_generalized_eigen(
    f_matrix: &[f64],
    x_matrix: &[f64],
    n_basis: usize,
    n_independent: usize,
) -> GeneralizedEigenResult {
    // Step 1: F' = X^T F X (n_independent × n_independent)
    let f_prime = xtax(x_matrix, f_matrix, n_basis, n_independent);

    // Step 2: Diagonalize F'
    let (eigenvalues, eigvecs_prime) = symmetric_eigen(&f_prime, n_independent);

    // Step 3: Sort by eigenvalue (ascending)
    let mut indices: Vec<usize> = (0..n_independent).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

    // Step 4: Back-transform C = X * C'
    // C[μ, i] = Σ_k X[μ, k] * C'[k, sorted_i]
    let mut coefficients = vec![0.0; n_basis * n_independent];
    for (i, &orig_i) in indices.iter().enumerate() {
        for mu in 0..n_basis {
            let mut sum = 0.0;
            for k in 0..n_independent {
                sum += x_matrix[mu * n_independent + k] * eigvecs_prime[k * n_independent + orig_i];
            }
            coefficients[mu * n_independent + i] = sum;
        }
    }

    GeneralizedEigenResult {
        eigenvalues: sorted_eigenvalues,
        coefficients,
        n_independent,
        n_discarded: n_basis - n_independent,
        orthogonalization_matrix: x_matrix.to_vec(),
    }
}

/// Maximum normalized residual accepted for the original generalized equation FC = SCε.
pub const CHECKED_GENERALIZED_EIGENPAIR_RESIDUAL_LIMIT: f64 = 1e-8;

/// Checked result for FC = SCε. The inner eigensolver diagnostics refer to F' = Xᵀ F X;
/// the generalized residual is recomputed independently in the original AO basis.
#[derive(Debug, Clone)]
pub struct CheckedGeneralizedEigenResult {
    pub generalized: GeneralizedEigenResult,
    pub eigensolver: CheckedSymmetricEigenResult,
    pub max_relative_generalized_eigenpair_residual: f64,
}

/// Numerical receipt aggregated over every successful generalized eigensolve in an SCF run.
///
/// Residuals are normalized dimensionless maxima. This receipt is separate from the SCF
/// energy/density convergence flag: it shows that each diagonalization met its own
/// numerical contracts, but does not imply that the SCF fixed point converged or that
/// the model/basis is chemically accurate.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfSolverDiagnostics {
    /// Count of successfully checked generalized eigen solves (initial guess plus SCF steps).
    pub generalized_eigensolve_count: usize,
    /// Total Jacobi rotations across successful transformed-Fock eigensolves.
    pub total_jacobi_rotations: usize,
    /// Largest absolute off-diagonal entry seen in any transformed Fock matrix (Hartree).
    pub max_transformed_off_diagonal_hartree: f64,
    /// Maximum residual for the transformed symmetric eigenproblem across all solves.
    pub max_transformed_eigenpair_residual: f64,
    /// Maximum (C'^T C' - I) elementwise residual across all solves.
    pub max_eigenvector_orthogonality_residual: f64,
    /// Maximum residual for (F C = S C \epsilon) in the original AO basis.
    pub max_generalized_eigenpair_residual: f64,
    /// Residual for (X^T S X - I) from checked canonical orthogonalization.
    pub overlap_orthogonality_residual: f64,
}

impl ScfSolverDiagnostics {
    pub(crate) fn from_overlap_residual(overlap_orthogonality_residual: f64) -> Self {
        Self {
            generalized_eigensolve_count: 0,
            total_jacobi_rotations: 0,
            max_transformed_off_diagonal_hartree: 0.0,
            max_transformed_eigenpair_residual: 0.0,
            max_eigenvector_orthogonality_residual: 0.0,
            max_generalized_eigenpair_residual: 0.0,
            overlap_orthogonality_residual,
        }
    }

    pub(crate) fn record_generalized_solve(&mut self, result: &CheckedGeneralizedEigenResult) {
        self.generalized_eigensolve_count = self.generalized_eigensolve_count.saturating_add(1);
        self.total_jacobi_rotations = self
            .total_jacobi_rotations
            .saturating_add(result.eigensolver.iterations);
        self.max_transformed_off_diagonal_hartree = self
            .max_transformed_off_diagonal_hartree
            .max(result.eigensolver.max_off_diagonal);
        self.max_transformed_eigenpair_residual = self
            .max_transformed_eigenpair_residual
            .max(result.eigensolver.max_relative_eigenpair_residual);
        self.max_eigenvector_orthogonality_residual = self
            .max_eigenvector_orthogonality_residual
            .max(result.eigensolver.max_orthogonality_residual);
        self.max_generalized_eigenpair_residual = self
            .max_generalized_eigenpair_residual
            .max(result.max_relative_generalized_eigenpair_residual);
    }
}

/// Failure from validating or solving the generalized eigenvalue problem.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckedGeneralizedEigenError {
    ZeroBasisDimension,
    ZeroIndependentDimension,
    IndependentDimensionExceedsBasis,
    MatrixLengthMismatch {
        matrix: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        matrix: &'static str,
        index: usize,
    },
    NonsymmetricFockMatrix {
        row: usize,
        column: usize,
        normalized_difference: f64,
    },
    NonsymmetricOverlapMatrix {
        row: usize,
        column: usize,
        normalized_difference: f64,
    },
    OrthogonalizationResidualExceeded {
        residual: f64,
        limit: f64,
    },
    NonFiniteTransformedMatrix { index: usize },
    NonFiniteIntermediate,
    Eigen(SymmetricEigenError),
    EigensolverNotConverged(CheckedSymmetricEigenResult),
    GeneralizedResidualExceeded {
        residual: f64,
        limit: f64,
    },
}

impl std::fmt::Display for CheckedGeneralizedEigenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBasisDimension => write!(f, "generalized eigenproblem has zero basis dimension"),
            Self::ZeroIndependentDimension => write!(f, "generalized eigenproblem has zero retained dimension"),
            Self::IndependentDimensionExceedsBasis => {
                write!(f, "retained dimension exceeds basis dimension")
            }
            Self::MatrixLengthMismatch { matrix, expected, actual } => write!(
                f,
                "{matrix} matrix expected {expected} entries, received {actual}"
            ),
            Self::NonFiniteInput { matrix, index } => {
                write!(f, "{matrix} input entry {index} is non-finite")
            }
            Self::NonsymmetricFockMatrix { row, column, normalized_difference } => write!(
                f,
                "Fock matrix is not symmetric at ({row}, {column}); normalized difference is {normalized_difference}"
            ),
            Self::NonsymmetricOverlapMatrix {
                row,
                column,
                normalized_difference,
            } => write!(
                f,
                "overlap matrix is not symmetric at ({row}, {column}); normalized difference is {normalized_difference}"
            ),
            Self::OrthogonalizationResidualExceeded { residual, limit } => write!(
                f,
                "Xᵀ S X residual {residual} exceeds limit {limit}"
            ),
            Self::NonFiniteTransformedMatrix { index } => {
                write!(f, "transformed Fock matrix entry {index} is non-finite")
            }
            Self::NonFiniteIntermediate => {
                write!(f, "generalized eigensolver produced a non-finite intermediate")
            }
            Self::Eigen(error) => write!(f, "transformed Fock eigensolver rejected input: {error}"),
            Self::EigensolverNotConverged(result) => write!(
                f,
                "transformed Fock eigensolver did not satisfy convergence contracts: {:?}",
                result.stopping_reason
            ),
            Self::GeneralizedResidualExceeded { residual, limit } => write!(
                f,
                "generalized eigenpair residual {residual} exceeds limit {limit}"
            ),
        }
    }
}

impl std::error::Error for CheckedGeneralizedEigenError {}

/// Fail-closed error returned by RHF/UHF checked numerical paths.
#[derive(Debug, Clone, PartialEq)]
pub enum ScfSolveError {
    EmptyMolecule,
    ElectronCountOverflow,
    UnsupportedMultiplicity {
        method: &'static str,
        multiplicity: u32,
    },
    InvalidMultiplicity {
        multiplicity: u32,
    },
    InconsistentElectronicState {
        electron_count: usize,
        multiplicity: u32,
    },
    InvalidCharge {
        charge: i32,
        nuclear_charge: i64,
    },
    NoElectrons,
    CanonicalOrthogonalization(CanonicalOrthogonalizationError),
    GeneralizedEigen(CheckedGeneralizedEigenError),
    NonFiniteEnergy {
        stage: &'static str,
    },
}

impl std::fmt::Display for ScfSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMolecule => write!(f, "molecular SCF requires at least one atom"),
            Self::ElectronCountOverflow => {
                write!(f, "electron count cannot be represented on this target")
            }
            Self::UnsupportedMultiplicity { method, multiplicity } => write!(
                f,
                "{method} does not support multiplicity {multiplicity}"
            ),
            Self::InvalidMultiplicity { multiplicity } => write!(
                f,
                "multiplicity {multiplicity} is invalid; multiplicity must be at least one"
            ),
            Self::InconsistentElectronicState { electron_count, multiplicity } => write!(
                f,
                "{electron_count} electrons are inconsistent with multiplicity {multiplicity}"
            ),
            Self::InvalidCharge { charge, nuclear_charge } => write!(
                f,
                "molecule charge {charge} exceeds total nuclear charge {nuclear_charge}"
            ),
            Self::NoElectrons => {
                write!(f, "zero-electron systems are unsupported by the molecular SCF solver")
            }
            Self::CanonicalOrthogonalization(error) => {
                write!(f, "checked canonical orthogonalization failed: {error}")
            }
            Self::GeneralizedEigen(error) => {
                write!(f, "checked generalized eigensolve failed: {error}")
            }
            Self::NonFiniteEnergy { stage } => {
                write!(f, "SCF energy became non-finite at {stage}")
            }
        }
    }
}

impl std::error::Error for ScfSolveError {}

impl From<CanonicalOrthogonalizationError> for ScfSolveError {
    fn from(value: CanonicalOrthogonalizationError) -> Self {
        Self::CanonicalOrthogonalization(value)
    }
}

impl From<CheckedGeneralizedEigenError> for ScfSolveError {
    fn from(value: CheckedGeneralizedEigenError) -> Self {
        Self::GeneralizedEigen(value)
    }
}

/// Solve FC = SCε using only checked numeric primitives.
///
/// The caller must provide an orthogonalization matrix produced by
/// `canonical_orthogonalization_checked`. This function verifies the matrix
/// shapes/finiteness, Fock symmetry, transformed eigensolver status, and the
/// original AO-space generalized eigenpair residual. An iteration limit or
/// any failed contract returns an error; no result with a false convergence
/// status is exposed as a successful generalized eigensolution.
pub fn solve_generalized_eigen_checked(
    f_matrix: &[f64],
    s_matrix: &[f64],
    x_matrix: &[f64],
    n_basis: usize,
    n_independent: usize,
    max_iterations: usize,
) -> Result<CheckedGeneralizedEigenResult, CheckedGeneralizedEigenError> {
    if n_basis == 0 {
        return Err(CheckedGeneralizedEigenError::ZeroBasisDimension);
    }
    if n_independent == 0 {
        return Err(CheckedGeneralizedEigenError::ZeroIndependentDimension);
    }
    if n_independent > n_basis {
        return Err(CheckedGeneralizedEigenError::IndependentDimensionExceedsBasis);
    }
    let square_len = n_basis
        .checked_mul(n_basis)
        .ok_or(CheckedGeneralizedEigenError::NonFiniteIntermediate)?;
    let transform_len = n_basis
        .checked_mul(n_independent)
        .ok_or(CheckedGeneralizedEigenError::NonFiniteIntermediate)?;
    for (name, matrix, expected) in [
        ("Fock", f_matrix, square_len),
        ("overlap", s_matrix, square_len),
        ("orthogonalization", x_matrix, transform_len),
    ] {
        if matrix.len() != expected {
            return Err(CheckedGeneralizedEigenError::MatrixLengthMismatch {
                matrix: name,
                expected,
                actual: matrix.len(),
            });
        }
        if let Some(index) = matrix.iter().position(|value| !value.is_finite()) {
            return Err(CheckedGeneralizedEigenError::NonFiniteInput {
                matrix: name,
                index,
            });
        }
    }

    let fock_scale = f_matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let fock_scale = if fock_scale == 0.0 { 1.0 } else { fock_scale };
    for row in 0..n_basis {
        for column in (row + 1)..n_basis {
            let normalized_difference = (
                f_matrix[row * n_basis + column] / fock_scale
                    - f_matrix[column * n_basis + row] / fock_scale
            )
            .abs();
            if normalized_difference > CHECKED_SYMMETRY_RELATIVE_TOLERANCE {
                return Err(CheckedGeneralizedEigenError::NonsymmetricFockMatrix {
                    row,
                    column,
                    normalized_difference,
                });
            }
        }
    }

    let overlap_scale = s_matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let overlap_scale = if overlap_scale == 0.0 { 1.0 } else { overlap_scale };
    for row in 0..n_basis {
        for column in (row + 1)..n_basis {
            let normalized_difference = (
                s_matrix[row * n_basis + column] / overlap_scale
                    - s_matrix[column * n_basis + row] / overlap_scale
            )
            .abs();
            if normalized_difference > CHECKED_SYMMETRY_RELATIVE_TOLERANCE {
                return Err(CheckedGeneralizedEigenError::NonsymmetricOverlapMatrix {
                    row,
                    column,
                    normalized_difference,
                });
            }
        }
    }
    let orthogonality_residual =
        max_overlap_orthogonality_residual(s_matrix, x_matrix, n_basis, n_independent);
    if !orthogonality_residual.is_finite() {
        return Err(CheckedGeneralizedEigenError::NonFiniteIntermediate);
    }
    if orthogonality_residual > CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT {
        return Err(CheckedGeneralizedEigenError::OrthogonalizationResidualExceeded {
            residual: orthogonality_residual,
            limit: CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT,
        });
    }

    let f_prime = xtax(x_matrix, f_matrix, n_basis, n_independent);
    if let Some(index) = f_prime.iter().position(|value| !value.is_finite()) {
        return Err(CheckedGeneralizedEigenError::NonFiniteTransformedMatrix { index });
    }
    let eigensolver = symmetric_eigen_checked(&f_prime, n_independent, max_iterations)
        .map_err(CheckedGeneralizedEigenError::Eigen)?;
    if !eigensolver.converged {
        return Err(CheckedGeneralizedEigenError::EigensolverNotConverged(eigensolver));
    }

    let mut indices: Vec<usize> = (0..n_independent).collect();
    indices.sort_by(|&a, &b| eigensolver.eigenvalues[a].total_cmp(&eigensolver.eigenvalues[b]));
    let sorted_eigenvalues: Vec<f64> = indices
        .iter()
        .map(|&index| eigensolver.eigenvalues[index])
        .collect();
    let mut coefficients = vec![0.0; transform_len];
    for (target_column, &source_column) in indices.iter().enumerate() {
        for row in 0..n_basis {
            let mut value = 0.0;
            for k in 0..n_independent {
                value += x_matrix[row * n_independent + k]
                    * eigensolver.eigenvectors[k * n_independent + source_column];
            }
            coefficients[row * n_independent + target_column] = value;
        }
    }
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(CheckedGeneralizedEigenError::NonFiniteIntermediate);
    }

    let overlap_scale = s_matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let max_energy = sorted_eigenvalues
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let rhs_scale = overlap_scale * max_energy;
    if !rhs_scale.is_finite() {
        return Err(CheckedGeneralizedEigenError::NonFiniteIntermediate);
    }
    let residual_scale = fock_scale.max(rhs_scale);
    let residual_scale = if residual_scale == 0.0 { 1.0 } else { residual_scale };
    let mut max_residual = 0.0_f64;
    for column in 0..n_independent {
        let energy = sorted_eigenvalues[column];
        for row in 0..n_basis {
            let mut fc = 0.0_f64;
            let mut sc = 0.0_f64;
            for k in 0..n_basis {
                fc += f_matrix[row * n_basis + k] * coefficients[k * n_independent + column];
                sc += s_matrix[row * n_basis + k] * coefficients[k * n_independent + column];
            }
            let residual = fc - energy * sc;
            if !fc.is_finite() || !sc.is_finite() || !residual.is_finite() {
                return Err(CheckedGeneralizedEigenError::NonFiniteIntermediate);
            }
            max_residual = max_residual.max(residual.abs() / residual_scale);
        }
    }
    if !max_residual.is_finite() {
        return Err(CheckedGeneralizedEigenError::NonFiniteIntermediate);
    }
    if max_residual > CHECKED_GENERALIZED_EIGENPAIR_RESIDUAL_LIMIT {
        return Err(CheckedGeneralizedEigenError::GeneralizedResidualExceeded {
            residual: max_residual,
            limit: CHECKED_GENERALIZED_EIGENPAIR_RESIDUAL_LIMIT,
        });
    }

    Ok(CheckedGeneralizedEigenResult {
        generalized: GeneralizedEigenResult {
            eigenvalues: sorted_eigenvalues,
            coefficients,
            n_independent,
            n_discarded: n_basis - n_independent,
            orthogonalization_matrix: x_matrix.to_vec(),
        },
        eigensolver,
        max_relative_generalized_eigenpair_residual: max_residual,
    })
}

// ── Internal linear algebra (pure f64, no HDC overhead for inner loops) ─────

/// Relative off-diagonal tolerance used by the checked Jacobi path.
pub const CHECKED_JACOBI_RELATIVE_TOLERANCE: f64 = 1e-14;
/// Maximum normalized eigenpair residual accepted by the checked path.
pub const CHECKED_EIGENPAIR_RESIDUAL_LIMIT: f64 = 1e-10;
/// Maximum eigenvector orthogonality residual accepted by the checked path.
pub const CHECKED_EIGENVECTOR_ORTHOGONALITY_LIMIT: f64 = 1e-10;
/// Maximum element-wise residual in Xᵀ S X - I accepted by checked orthogonalization.
pub const CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT: f64 = 1e-8;
/// Relative tolerance for checking that a supplied matrix is symmetric.
pub const CHECKED_SYMMETRY_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Why the checked Jacobi path stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenStoppingReason {
    /// Off-diagonal and independently measured residual contracts passed.
    Converged,
    /// The configured rotation budget was exhausted before the off-diagonal criterion passed.
    IterationLimit,
    /// Off-diagonal convergence was reached, but a residual contract failed.
    ResidualContractFailed,
}

/// Diagnostics from a checked symmetric eigendecomposition.
///
/// Eigenvectors are row-major with eigenvectors in columns:
/// `eigenvectors[row * n + column]`. Eigenvalues use the same column index.
/// A non-converged result is diagnostic only and must not be used as a valid
/// eigensolution by SCF or other numerical callers.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckedSymmetricEigenResult {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub stopping_reason: EigenStoppingReason,
    pub max_off_diagonal: f64,
    pub max_relative_eigenpair_residual: f64,
    pub max_orthogonality_residual: f64,
}

/// Input or intermediate failure from the checked symmetric eigensolver.
#[derive(Debug, Clone, PartialEq)]
pub enum SymmetricEigenError {
    ZeroDimension,
    DimensionOverflow,
    MatrixLengthMismatch { expected: usize, actual: usize },
    NonFiniteInput { index: usize },
    NonFiniteIntermediate,
    NonsymmetricInput {
        row: usize,
        column: usize,
        normalized_difference: f64,
    },
}

impl std::fmt::Display for SymmetricEigenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "symmetric eigensolver dimension must be positive"),
            Self::DimensionOverflow => {
                write!(f, "symmetric eigensolver matrix dimension overflowed")
            }
            Self::MatrixLengthMismatch { expected, actual } => write!(
                f,
                "symmetric eigensolver expected {expected} matrix entries, received {actual}"
            ),
            Self::NonFiniteInput { index } => {
                write!(f, "symmetric eigensolver input entry {index} is non-finite")
            }
            Self::NonFiniteIntermediate => {
                write!(f, "symmetric eigensolver produced a non-finite intermediate")
            }
            Self::NonsymmetricInput {
                row,
                column,
                normalized_difference,
            } => write!(
                f,
                "matrix is not symmetric at ({row}, {column}); normalized difference is {normalized_difference}"
            ),
        }
    }
}

impl std::error::Error for SymmetricEigenError {}

/// Checked Jacobi eigendecomposition for a finite symmetric row-major matrix.
///
/// The function validates dimensions, finiteness, and symmetry before rotating.
/// It reports a non-converged result when the rotation budget is exhausted or
/// the independently calculated eigenpair/orthogonality residuals exceed their
/// fixed contracts. A returned `Ok` is not synonymous with convergence:
/// callers must require `result.converged`.
///
/// `max_iterations` is a count of Jacobi rotations. Zero is accepted for
/// deterministic exhaustion tests and returns a non-converged result when
/// the matrix has material off-diagonal terms.
pub fn symmetric_eigen_checked(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
) -> Result<CheckedSymmetricEigenResult, SymmetricEigenError> {
    if n == 0 {
        return Err(SymmetricEigenError::ZeroDimension);
    }
    let expected = n.checked_mul(n).ok_or(SymmetricEigenError::DimensionOverflow)?;
    if matrix.len() != expected {
        return Err(SymmetricEigenError::MatrixLengthMismatch {
            expected,
            actual: matrix.len(),
        });
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        let index = matrix.iter().position(|value| !value.is_finite()).unwrap_or(0);
        return Err(SymmetricEigenError::NonFiniteInput { index });
    }

    let matrix_scale = matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    // Keep relative checks meaningful for tiny matrices while avoiding division
    // by zero for an exactly-zero matrix.
    let matrix_scale = if matrix_scale == 0.0 { 1.0 } else { matrix_scale };
    for row in 0..n {
        for column in (row + 1)..n {
            let normalized_difference =
                (matrix[row * n + column] / matrix_scale
                    - matrix[column * n + row] / matrix_scale)
                    .abs();
            if normalized_difference > CHECKED_SYMMETRY_RELATIVE_TOLERANCE {
                return Err(SymmetricEigenError::NonsymmetricInput {
                    row,
                    column,
                    normalized_difference,
                });
            }
        }
    }

    let mut a = matrix.to_vec();
    let mut eigenvectors = vec![0.0; expected];
    for i in 0..n {
        eigenvectors[i * n + i] = 1.0;
    }

    let mut iterations = 0;
    let (mut max_off_diagonal, mut p, mut q) = max_off_diagonal_entry(&a, n);
    while max_off_diagonal / matrix_scale > CHECKED_JACOBI_RELATIVE_TOLERANCE
        && iterations < max_iterations
    {
        // A scaled atan2 avoids the quadrant loss of atan(y / x) and keeps
        // 2*apq/app-aqq intermediate arithmetic bounded for large finite inputs.
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        // apq is nonzero in this loop, so the scale remains positive even
        // for subnormal matrices; flooring at MIN_POSITIVE would distort them.
        let rotation_scale = app.abs().max(aqq.abs()).max(apq.abs());
        let theta = 0.5
            * (2.0 * (apq / rotation_scale)).atan2(
                app / rotation_scale - aqq / rotation_scale,
            );
        let cosine = theta.cos();
        let sine = theta.sin();

        let mut next = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                next[i * n + p] = cosine * aip + sine * aiq;
                next[p * n + i] = next[i * n + p];
                next[i * n + q] = -sine * aip + cosine * aiq;
                next[q * n + i] = next[i * n + q];
            }
        }
        next[p * n + p] =
            cosine * cosine * app + 2.0 * cosine * sine * apq + sine * sine * aqq;
        next[q * n + q] =
            sine * sine * app - 2.0 * cosine * sine * apq + cosine * cosine * aqq;
        next[p * n + q] = 0.0;
        next[q * n + p] = 0.0;

        for i in 0..n {
            let vip = eigenvectors[i * n + p];
            let viq = eigenvectors[i * n + q];
            eigenvectors[i * n + p] = cosine * vip + sine * viq;
            eigenvectors[i * n + q] = -sine * vip + cosine * viq;
        }
        if next.iter().any(|value| !value.is_finite())
            || eigenvectors.iter().any(|value| !value.is_finite())
        {
            return Err(SymmetricEigenError::NonFiniteIntermediate);
        }
        a = next;
        iterations += 1;
        (max_off_diagonal, p, q) = max_off_diagonal_entry(&a, n);
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(SymmetricEigenError::NonFiniteIntermediate);
    }

    let max_relative_eigenpair_residual =
        max_relative_eigenpair_residual(matrix, &eigenvalues, &eigenvectors, n, matrix_scale);
    let max_orthogonality_residual = max_eigenvector_orthogonality_residual(&eigenvectors, n);
    if !max_relative_eigenpair_residual.is_finite() || !max_orthogonality_residual.is_finite() {
        return Err(SymmetricEigenError::NonFiniteIntermediate);
    }

    let off_diagonal_converged =
        max_off_diagonal / matrix_scale <= CHECKED_JACOBI_RELATIVE_TOLERANCE;
    let residuals_converged = max_relative_eigenpair_residual <= CHECKED_EIGENPAIR_RESIDUAL_LIMIT
        && max_orthogonality_residual <= CHECKED_EIGENVECTOR_ORTHOGONALITY_LIMIT;
    let stopping_reason = if !off_diagonal_converged {
        EigenStoppingReason::IterationLimit
    } else if !residuals_converged {
        EigenStoppingReason::ResidualContractFailed
    } else {
        EigenStoppingReason::Converged
    };

    Ok(CheckedSymmetricEigenResult {
        eigenvalues,
        eigenvectors,
        converged: stopping_reason == EigenStoppingReason::Converged,
        iterations,
        stopping_reason,
        max_off_diagonal,
        max_relative_eigenpair_residual,
        max_orthogonality_residual,
    })
}

fn max_off_diagonal_entry(matrix: &[f64], n: usize) -> (f64, usize, usize) {
    let mut maximum = 0.0_f64;
    let mut p = 0;
    let mut q = 0;
    for row in 0..n {
        for column in (row + 1)..n {
            let value = matrix[row * n + column].abs();
            if value > maximum {
                maximum = value;
                p = row;
                q = column;
            }
        }
    }
    (maximum, p, q)
}

fn max_relative_eigenpair_residual(
    matrix: &[f64],
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    n: usize,
    matrix_scale: f64,
) -> f64 {
    let mut maximum = 0.0_f64;
    for column in 0..n {
        let eigenvalue = eigenvalues[column] / matrix_scale;
        for row in 0..n {
            let mut product = 0.0;
            for k in 0..n {
                product += (matrix[row * n + k] / matrix_scale)
                    * eigenvectors[k * n + column];
            }
            let residual = (product - eigenvalue * eigenvectors[row * n + column]).abs();
            maximum = maximum.max(residual);
        }
    }
    maximum
}

fn max_eigenvector_orthogonality_residual(eigenvectors: &[f64], n: usize) -> f64 {
    let mut maximum = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for row in 0..n {
                dot += eigenvectors[row * n + i] * eigenvectors[row * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            maximum = maximum.max((dot - expected).abs());
        }
    }
    maximum
}

/// Checked result of canonical orthogonalization of an overlap matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckedCanonicalOrthogonalization {
    /// Row-major transformation X (n_basis × n_independent).
    pub transformation: Vec<f64>,
    pub n_independent: usize,
    pub n_discarded: usize,
    pub orthogonality_residual: f64,
    /// Eigensolver diagnostics for the original overlap matrix.
    pub eigensolver: CheckedSymmetricEigenResult,
}

/// Failure while validating a checked canonical-orthogonalization result.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalOrthogonalizationError {
    InvalidThreshold,
    Eigen(SymmetricEigenError),
    EigensolverNotConverged(CheckedSymmetricEigenResult),
    NegativeOverlapEigenvalue {
        index: usize,
        eigenvalue: f64,
        threshold: f64,
    },
    NoIndependentBasisFunctions,
    NonFiniteTransform,
    OrthogonalityResidualExceeded { residual: f64, limit: f64 },
}

impl std::fmt::Display for CanonicalOrthogonalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreshold => write!(
                f,
                "canonical-orthogonalization threshold must be finite and positive"
            ),
            Self::Eigen(error) => write!(f, "overlap eigensolver rejected the matrix: {error}"),
            Self::EigensolverNotConverged(result) => write!(
                f,
                "overlap eigensolver did not satisfy convergence contracts: {:?}",
                result.stopping_reason
            ),
            Self::NegativeOverlapEigenvalue {
                index,
                eigenvalue,
                threshold,
            } => write!(
                f,
                "overlap eigenvalue {index} is materially negative ({eigenvalue}); \
                 minimum accepted is -{threshold}"
            ),
            Self::NoIndependentBasisFunctions => {
                write!(f, "overlap matrix has no independent basis functions")
            }
            Self::NonFiniteTransform => {
                write!(f, "canonical orthogonalization produced non-finite values")
            }
            Self::OrthogonalityResidualExceeded { residual, limit } => write!(
                f,
                "Xᵀ S X residual {residual} exceeds the accepted limit {limit}"
            ),
        }
    }
}

impl std::error::Error for CanonicalOrthogonalizationError {}

/// Checked canonical orthogonalization that refuses an unqualified eigensolve.
///
/// The legacy `canonical_orthogonalization` wrapper is retained for compatibility,
/// but this path requires a converged overlap eigensolve and independently checks
/// `Xᵀ S X ≈ I` before returning a transformation.
pub fn canonical_orthogonalization_checked(
    s_matrix: &[f64],
    n: usize,
    threshold: f64,
    max_iterations: usize,
) -> Result<CheckedCanonicalOrthogonalization, CanonicalOrthogonalizationError> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(CanonicalOrthogonalizationError::InvalidThreshold);
    }
    let eigensolver = symmetric_eigen_checked(s_matrix, n, max_iterations)
        .map_err(CanonicalOrthogonalizationError::Eigen)?;
    if !eigensolver.converged {
        return Err(CanonicalOrthogonalizationError::EigensolverNotConverged(eigensolver));
    }

    // Tiny negative modes within the configured rank threshold can arise from
    // floating-point roundoff in an almost-dependent basis. A materially
    // negative overlap eigenvalue, however, is evidence of an invalid S matrix
    // or failed integral construction and must not be silently discarded.
    if let Some((index, &eigenvalue)) = eigensolver
        .eigenvalues
        .iter()
        .enumerate()
        .find(|(_, value)| **value < -threshold)
    {
        return Err(CanonicalOrthogonalizationError::NegativeOverlapEigenvalue {
            index,
            eigenvalue,
            threshold,
        });
    }

    let survivors: Vec<usize> = eigensolver
        .eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value >= threshold).then_some(index))
        .collect();
    let n_independent = survivors.len();
    if n_independent == 0 {
        return Err(CanonicalOrthogonalizationError::NoIndependentBasisFunctions);
    }
    let n_discarded = n - n_independent;
    let mut transformation = vec![0.0; n * n_independent];
    for (target_column, &source_column) in survivors.iter().enumerate() {
        let inverse_sqrt = 1.0 / eigensolver.eigenvalues[source_column].sqrt();
        for row in 0..n {
            transformation[row * n_independent + target_column] =
                eigensolver.eigenvectors[row * n + source_column] * inverse_sqrt;
        }
    }
    if transformation.iter().any(|value| !value.is_finite()) {
        return Err(CanonicalOrthogonalizationError::NonFiniteTransform);
    }
    let residual = max_overlap_orthogonality_residual(
        s_matrix,
        &transformation,
        n,
        n_independent,
    );
    if !residual.is_finite() {
        return Err(CanonicalOrthogonalizationError::NonFiniteTransform);
    }
    if residual > CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT {
        return Err(CanonicalOrthogonalizationError::OrthogonalityResidualExceeded {
            residual,
            limit: CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT,
        });
    }

    Ok(CheckedCanonicalOrthogonalization {
        transformation,
        n_independent,
        n_discarded,
        orthogonality_residual: residual,
        eigensolver,
    })
}

fn max_overlap_orthogonality_residual(
    overlap: &[f64],
    transformation: &[f64],
    n_basis: usize,
    n_independent: usize,
) -> f64 {
    let mut maximum = 0.0_f64;
    for i in 0..n_independent {
        for j in 0..n_independent {
            let mut value = 0.0;
            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    value += transformation[mu * n_independent + i]
                        * overlap[mu * n_basis + nu]
                        * transformation[nu * n_independent + j];
                }
            }
            if !value.is_finite() {
                return f64::INFINITY;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            maximum = maximum.max((value - expected).abs());
        }
    }
    maximum
}

/// Symmetric eigendecomposition via Jacobi rotations.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-major.
/// Eigenvalues are returned in the order the Jacobi sweep converges them
/// (NOT pre-sorted) -- callers that need sorted output must sort themselves
/// (see `canonical_orthogonalize`/`generalized_eigen_fock` above for the
/// established pattern).
///
/// `pub(crate)` since Phase Q4 (2026-07-17): reused by `vibrational.rs` to
/// diagonalize the mass-weighted Hessian for normal-mode analysis.
pub(crate) fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    // Initialize V = I
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = 1e-14;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to A
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Apply rotation to V
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Compute X^T * A * X where X is n×m and A is n×n, result is m×m.
fn xtax(x: &[f64], a: &[f64], n: usize, m: usize) -> Vec<f64> {
    // First: AX = A * X (n×m)
    let mut ax = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * x[k * m + j];
            }
            ax[i * m + j] = sum;
        }
    }

    // Then: X^T * AX (m×m)
    let mut result = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[k * m + i] * ax[k * m + j];
            }
            result[i * m + j] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_eigen_identity() {
        // Eigenvalues of identity should be all 1.0
        let n = 3;
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let (eigenvalues, _) = symmetric_eigen(&eye, n);
        for &ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let n = 3;
        let mut diag = vec![0.0; n * n];
        diag[0] = 3.0;
        diag[4] = 1.0;
        diag[8] = 2.0;
        let (eigenvalues, _) = symmetric_eigen(&diag, n);

        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonical_orthogonalization_identity() {
        // For S = I, X should be I and nothing discarded
        let n = 3;
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let (x, n_ind, n_disc) = canonical_orthogonalization(&eye, n);
        assert_eq!(n_ind, 3);
        assert_eq!(n_disc, 0);
        assert_eq!(x.len(), 9);
    }

    #[test]
    fn test_canonical_orthogonalization_discards_linear_dependence() {
        // S with one near-zero eigenvalue
        let n = 2;
        // S = [[1, 0.9999999], [0.9999999, 1]] → eigenvalues ~2.0 and ~1e-7
        let s = vec![1.0, 0.9999999, 0.9999999, 1.0];
        let (x, n_ind, n_disc) = canonical_orthogonalization(&s, n);

        // Should discard the near-zero eigenvector
        assert_eq!(n_ind, 1, "Should keep 1 of 2 functions");
        assert_eq!(n_disc, 1, "Should discard 1 function");
        assert_eq!(x.len(), 2); // n * n_independent = 2 * 1
    }

    #[test]
    fn test_generalized_eigen_known_problem() {
        // Simple 2x2: F = [[2, 1], [1, 2]], S = [[1, 0], [0, 1]]
        // Standard eigenvalues: 1 and 3
        let n = 2;
        let f = vec![2.0, 1.0, 1.0, 2.0];
        let s = vec![1.0, 0.0, 0.0, 1.0];

        let (x, n_ind, _) = canonical_orthogonalization(&s, n);
        let result = solve_generalized_eigen(&f, &x, n, n_ind);

        assert_eq!(result.eigenvalues.len(), 2);
        let mut evals = result.eigenvalues.clone();
        evals.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (evals[0] - 1.0).abs() < 1e-8,
            "First eigenvalue: {}, expected 1.0",
            evals[0]
        );
        assert!(
            (evals[1] - 3.0).abs() < 1e-8,
            "Second eigenvalue: {}, expected 3.0",
            evals[1]
        );
    }

    #[test]
    fn checked_generalized_eigensolver_verifies_original_ao_equation() {
        let f = [2.0, 1.0, 1.0, 2.0];
        let s = [1.0, 0.0, 0.0, 1.0];
        let x = [1.0, 0.0, 0.0, 1.0];
        let result = solve_generalized_eigen_checked(&f, &s, &x, 2, 2, 10).unwrap();
        assert_eq!(result.generalized.eigenvalues, vec![1.0, 3.0]);
        assert!(result.eigensolver.converged);
        assert!(
            result.max_relative_generalized_eigenpair_residual
                <= CHECKED_GENERALIZED_EIGENPAIR_RESIDUAL_LIMIT
        );
        assert!(
            (result.generalized.coefficients[0] - 2.0_f64.sqrt().recip()).abs() < 1e-12
                || (result.generalized.coefficients[0] + 2.0_f64.sqrt().recip()).abs() < 1e-12
        );
    }

    #[test]
    fn checked_generalized_eigensolver_rejects_unconverged_and_invalid_inputs() {
        let f = [2.0, 0.5, 0.5, 2.0];
        let s = [1.0, 0.0, 0.0, 1.0];
        let x = [1.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            solve_generalized_eigen_checked(&f, &s, &x, 2, 2, 0),
            Err(CheckedGeneralizedEigenError::EigensolverNotConverged(_))
        ));

        assert!(matches!(
            solve_generalized_eigen_checked(&[1.0, 0.5, 0.1, 1.0], &s, &x, 2, 2, 10),
            Err(CheckedGeneralizedEigenError::NonsymmetricFockMatrix { .. })
        ));

        let invalid_s = [1.0, 0.0, 0.0, 2.0];
        assert!(matches!(
            solve_generalized_eigen_checked(&s, &invalid_s, &x, 2, 2, 10),
            Err(CheckedGeneralizedEigenError::OrthogonalizationResidualExceeded { .. })
        ));
    }

    #[test]
    fn checked_eigensolver_accepts_one_by_one_diagonal_and_degenerate_matrices() {
        let one = symmetric_eigen_checked(&[3.25], 1, 0).unwrap();
        assert!(one.converged);
        assert_eq!(one.eigenvalues, vec![3.25]);
        assert_eq!(one.iterations, 0);

        let diagonal = symmetric_eigen_checked(&[3.0, 0.0, 0.0, 1.0], 2, 0).unwrap();
        assert!(diagonal.converged);
        assert_eq!(diagonal.stopping_reason, EigenStoppingReason::Converged);
        assert_eq!(diagonal.max_off_diagonal, 0.0);

        let degenerate = symmetric_eigen_checked(
            &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            3,
            0,
        )
        .unwrap();
        assert!(degenerate.converged);
        assert_eq!(degenerate.eigenvalues, vec![2.0, 2.0, 2.0]);
        assert!(degenerate.max_orthogonality_residual <= CHECKED_EIGENVECTOR_ORTHOGONALITY_LIMIT);
    }

    #[test]
    fn checked_eigensolver_reports_rotation_limit_without_false_convergence() {
        let matrix = [2.0, 0.5, 0.5, 2.0];
        let result = symmetric_eigen_checked(&matrix, 2, 0).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.stopping_reason, EigenStoppingReason::IterationLimit);

        let converged = symmetric_eigen_checked(&matrix, 2, 10).unwrap();
        assert!(converged.converged, "{converged:?}");
        let mut eigenvalues = converged.eigenvalues.clone();
        eigenvalues.sort_by(f64::total_cmp);
        assert!((eigenvalues[0] - 1.5).abs() < 1e-12);
        assert!((eigenvalues[1] - 2.5).abs() < 1e-12);
        assert!(converged.max_relative_eigenpair_residual <= CHECKED_EIGENPAIR_RESIDUAL_LIMIT);
        assert!(
            converged.max_orthogonality_residual
                <= CHECKED_EIGENVECTOR_ORTHOGONALITY_LIMIT
        );

        // Relative scaling must remain correct below f64::MIN_POSITIVE.
        let subnormal = [1e-310, 0.5e-310, 0.5e-310, 1e-310];
        let tiny = symmetric_eigen_checked(&subnormal, 2, 10).unwrap();
        assert!(tiny.converged, "{tiny:?}");
        let mut tiny_values = tiny.eigenvalues.clone();
        tiny_values.sort_by(f64::total_cmp);
        assert!((tiny_values[0] / 1e-310 - 0.5).abs() < 1e-10);
        assert!((tiny_values[1] / 1e-310 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn checked_eigensolver_rejects_malformed_matrices() {
        assert_eq!(
            symmetric_eigen_checked(&[], 0, 10).unwrap_err(),
            SymmetricEigenError::ZeroDimension
        );
        assert!(matches!(
            symmetric_eigen_checked(&[1.0, 0.0, 0.0], 2, 10),
            Err(SymmetricEigenError::MatrixLengthMismatch { .. })
        ));
        assert_eq!(
            symmetric_eigen_checked(&[1.0, f64::NAN, f64::NAN, 1.0], 2, 10).unwrap_err(),
            SymmetricEigenError::NonFiniteInput { index: 1 }
        );
        assert!(matches!(
            symmetric_eigen_checked(&[1.0, 0.5, 0.1, 1.0], 2, 10),
            Err(SymmetricEigenError::NonsymmetricInput { .. })
        ));
    }

    #[test]
    fn checked_canonical_orthogonalization_enforces_overlap_identity() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let result = canonical_orthogonalization_checked(&identity, 2, 1e-6, 0).unwrap();
        assert_eq!(result.n_independent, 2);
        assert_eq!(result.n_discarded, 0);
        assert!(result.eigensolver.converged);
        assert!(result.orthogonality_residual <= CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT);

        let near_dependent = [1.0, 0.9999999, 0.9999999, 1.0];
        let reduced =
            canonical_orthogonalization_checked(&near_dependent, 2, 1e-6, 100).unwrap();
        assert_eq!(reduced.n_independent, 1);
        assert_eq!(reduced.n_discarded, 1);
        assert!(reduced.orthogonality_residual <= CHECKED_ORTHOGONALIZATION_RESIDUAL_LIMIT);
    }

    #[test]
    fn checked_canonical_orthogonalization_rejects_materially_negative_overlap_modes() {
        let indefinite = [1.0, 0.0, 0.0, -1e-3];
        assert!(matches!(
            canonical_orthogonalization_checked(&indefinite, 2, 1e-6, 0),
            Err(CanonicalOrthogonalizationError::NegativeOverlapEigenvalue {
                index: 1,
                eigenvalue,
                threshold,
            }) if (eigenvalue + 1e-3).abs() < 1e-15 && (threshold - 1e-6).abs() < 1e-15
        ));

        // A small negative mode within the configured cutoff is numerical-rank
        // noise and is discarded, provided the retained transform satisfies Xᵀ S X ≈ I.
        let tiny_negative = [1.0, 0.0, 0.0, -1e-8];
        let result = canonical_orthogonalization_checked(&tiny_negative, 2, 1e-6, 0).unwrap();
        assert_eq!(result.n_independent, 1);
        assert_eq!(result.n_discarded, 1);
    }

    #[test]
    fn checked_canonical_orthogonalization_rejects_unconverged_or_malformed_inputs() {
        let difficult = [2.0, 0.5, 0.5, 2.0];
        assert!(matches!(
            canonical_orthogonalization_checked(&difficult, 2, 1e-6, 0),
            Err(CanonicalOrthogonalizationError::EigensolverNotConverged(_))
        ));
        assert!(matches!(
            canonical_orthogonalization_checked(&[1.0, 0.5, 0.1, 1.0], 2, 1e-6, 10),
            Err(CanonicalOrthogonalizationError::Eigen(
                SymmetricEigenError::NonsymmetricInput { .. }
            ))
        ));
        assert_eq!(
            canonical_orthogonalization_checked(&[1.0], 1, 0.0, 10).unwrap_err(),
            CanonicalOrthogonalizationError::InvalidThreshold
        );
    }

    #[test]
    fn test_generalized_eigen_with_overlap() {
        // F = [[1, 0.5], [0.5, 2]], S = [[1, 0.3], [0.3, 1]]
        // Reference: eigenvalues from Wolfram Alpha
        let n = 2;
        let f = vec![1.0, 0.5, 0.5, 2.0];
        let s = vec![1.0, 0.3, 0.3, 1.0];

        let (x, n_ind, _) = canonical_orthogonalization(&s, n);
        let result = solve_generalized_eigen(&f, &x, n, n_ind);

        assert_eq!(result.n_independent, 2);
        assert_eq!(result.n_discarded, 0);

        // Verify FC = SCε by checking residuals
        for i in 0..n_ind {
            let eps = result.eigenvalues[i];
            for mu in 0..n {
                let mut fc = 0.0;
                let mut sc = 0.0;
                for nu in 0..n {
                    fc += f[mu * n + nu] * result.coefficients[nu * n_ind + i];
                    sc += s[mu * n + nu] * result.coefficients[nu * n_ind + i];
                }
                let residual = (fc - eps * sc).abs();
                assert!(
                    residual < 1e-8,
                    "Residual at ({}, {}) = {} (eigenvalue={})",
                    mu,
                    i,
                    residual,
                    eps,
                );
            }
        }
    }
}
