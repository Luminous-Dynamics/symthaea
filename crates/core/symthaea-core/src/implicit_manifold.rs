// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed implicit-manifold primitives for bounded analytic constraints.
//!
//! V1 deliberately starts with Euclidean ambient spaces and equality constraints
//! with analytic Jacobians. Projection convergence, constraint satisfaction,
//! local regularity and global manifold existence are distinct claims. A failed
//! local solve is never promoted to proof that no valid point exists.

use blake3::Hasher;
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

use crate::state_space::{EuclideanSpace, StateSpace, StateValidity};

/// Maximum Euclidean ambient dimension accepted by the V1 kernel.
pub const MAX_IMPLICIT_AMBIENT_DIMENSION: usize = 64;
/// Maximum equality-constraint codimension accepted by the V1 kernel.
pub const MAX_IMPLICIT_CODIMENSION: usize = 16;

/// Exact identity-bearing description of one equality-constraint family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintProfile {
    kind: String,
    parameters: Vec<u8>,
}

impl ConstraintProfile {
    /// Construct an extension constraint profile.
    ///
    /// `kind` should be stable and versioned when semantics change. `parameters`
    /// must contain every value that changes residual/Jacobian/domain semantics.
    pub fn new(kind: impl Into<String>, parameters: Vec<u8>) -> Self {
        Self {
            kind: kind.into(),
            parameters,
        }
    }

    /// Stable constraint-family identifier.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Opaque exact semantic parameters.
    pub fn parameters(&self) -> &[u8] {
        &self.parameters
    }

    /// Deterministic BLAKE3 identity for this exact constraint profile.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-constraint-profile-v1\0");
        update_len_prefixed(&mut hasher, self.kind.as_bytes());
        update_len_prefixed(&mut hasher, &self.parameters);
        *hasher.finalize().as_bytes()
    }
}

fn update_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Domain qualification for evaluating one constraint at one ambient state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainStatus {
    /// The point is inside the declared evaluation domain.
    Inside,
    /// The point lies outside the declared evaluation domain.
    Outside { reason: String },
}

/// Analytic equality constraint `F(x) = 0` over a bounded Euclidean ambient space.
///
/// Jacobians are returned row-major with exact shape
/// `codimension × ambient_dimension`.
pub trait EqualityConstraint {
    /// Exact semantic profile.
    fn profile(&self) -> &ConstraintProfile;
    /// Ambient coordinate dimension.
    fn ambient_dimension(&self) -> usize;
    /// Number of independent equality equations advertised by the profile.
    fn codimension(&self) -> usize;
    /// Evaluate `F(x)`.
    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError>;
    /// Evaluate the analytic Jacobian `dF/dx`, row-major.
    fn jacobian(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError>;

    /// Qualify the point for residual/Jacobian evaluation.
    fn domain_status(&self, _state: &[f64]) -> Result<DomainStatus, ImplicitManifoldError> {
        Ok(DomainStatus::Inside)
    }
}

/// Exact numerical policy for local normal projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionPolicy {
    residual_tolerance: f64,
    rank_tolerance: f64,
    step_tolerance: f64,
    max_step_norm: f64,
    max_iterations: usize,
}

impl ProjectionPolicy {
    /// Construct a fully explicit projection policy.
    pub fn new(
        residual_tolerance: f64,
        rank_tolerance: f64,
        step_tolerance: f64,
        max_step_norm: f64,
        max_iterations: usize,
    ) -> Result<Self, ImplicitManifoldError> {
        for (name, value) in [
            ("residual_tolerance", residual_tolerance),
            ("rank_tolerance", rank_tolerance),
            ("step_tolerance", step_tolerance),
            ("max_step_norm", max_step_norm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ImplicitManifoldError::InvalidPolicy {
                    reason: format!("{name} must be finite and > 0, got {value}"),
                });
            }
        }
        if max_iterations == 0 {
            return Err(ImplicitManifoldError::InvalidPolicy {
                reason: "max_iterations must be > 0".to_string(),
            });
        }
        Ok(Self {
            residual_tolerance,
            rank_tolerance,
            step_tolerance,
            max_step_norm,
            max_iterations,
        })
    }

    /// Conservative software-reference policy for analytic fixtures.
    pub fn reference_v1() -> Self {
        Self::new(1e-10, 1e-12, 1e-13, 1.0, 32)
            .expect("reference projection policy is statically valid")
    }

    /// Residual norm required for local constraint satisfaction.
    pub fn residual_tolerance(&self) -> f64 {
        self.residual_tolerance
    }

    /// Absolute modified-Gram-Schmidt row-rank threshold.
    pub fn rank_tolerance(&self) -> f64 {
        self.rank_tolerance
    }

    /// Step norm below which a non-satisfied solve is classified as stagnated.
    pub fn step_tolerance(&self) -> f64 {
        self.step_tolerance
    }

    /// Maximum Euclidean correction norm per iteration.
    pub fn max_step_norm(&self) -> f64 {
        self.max_step_norm
    }

    /// Hard iteration budget.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    fn update_hash(&self, hasher: &mut Hasher) {
        hasher.update(&self.residual_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.rank_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.step_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.max_step_norm.to_bits().to_le_bytes());
        hasher.update(&(self.max_iterations as u64).to_le_bytes());
    }
}

/// Identity-bearing constrained-manifold problem profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplicitManifoldProfile {
    ambient_dimension: usize,
    codimension: usize,
    identity: [u8; 32],
}

impl ImplicitManifoldProfile {
    /// Ambient coordinate dimension.
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Declared equality codimension.
    pub fn codimension(&self) -> usize {
        self.codimension
    }

    /// Regular-point intrinsic dimension `ambient - codimension`.
    ///
    /// This is only a declared regular-case dimension. Rank loss at a point is
    /// reported explicitly and does not inherit this value as a theorem.
    pub fn declared_regular_dimension(&self) -> usize {
        self.ambient_dimension - self.codimension
    }

    /// Exact BLAKE3 identity binding ambient, constraint and solver policy.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Typed failures in the implicit-manifold kernel itself.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ImplicitManifoldError {
    /// A declared dimension is unsupported or inconsistent.
    #[error("invalid implicit-manifold dimensions: {reason}")]
    InvalidDimensions { reason: String },
    /// A numerical policy is invalid.
    #[error("invalid projection policy: {reason}")]
    InvalidPolicy { reason: String },
    /// Input vector dimension does not match the ambient space.
    #[error("state dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// Constraint output shape violates its advertised dimensions.
    #[error("constraint {kind} returned {actual} {output}, expected {expected}")]
    ConstraintShape {
        kind: String,
        output: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Non-finite data was encountered.
    #[error("non-finite value during {stage}")]
    NonFinite { stage: &'static str },
    /// A point that must lie on the manifold does not meet the residual tolerance.
    #[error("state is not on the declared manifold: residual norm {residual_norm}")]
    StateNotOnManifold { residual_norm: f64 },
    /// A state is unsuitable for tangent construction for another explicit reason.
    #[error("state is not qualified for tangent construction: {reason}")]
    StateNotQualified { reason: String },
    /// Local Jacobian row rank is below the declared codimension.
    #[error("constraint Jacobian rank {rank} is below required codimension {required}")]
    RankDeficient { rank: usize, required: usize },
    /// An orthonormal tangent complement could not be completed numerically.
    #[error("tangent basis incomplete: expected {expected} vectors, built {actual}")]
    TangentBasisIncomplete { expected: usize, actual: usize },
}

/// Constraint validity without promoting local numerical failure into impossibility.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintValidity {
    /// The point satisfies the declared equality residual tolerance.
    Valid { residual_norm: f64 },
    /// Ambient coordinates are structurally invalid.
    AmbientInvalid(StateValidity),
    /// The point is outside the constraint's evaluation domain.
    OutsideDomain { reason: String },
    /// Residual is finite but above tolerance.
    ResidualExceeded {
        residual_norm: f64,
        tolerance: f64,
    },
    /// Constraint evaluation itself failed.
    EvaluationFailed { reason: String },
}

impl ConstraintValidity {
    /// True only for explicitly qualified local constraint satisfaction.
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }
}

/// Explicit outcome of one bounded local projection attempt.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionOutcome {
    /// Input met the residual tolerance and the local Jacobian has full row rank.
    AlreadySatisfied { residual_norm: f64 },
    /// Projection converged after at least one correction step.
    Projected {
        iterations: usize,
        residual_norm: f64,
    },
    /// Solver stagnated while residual remained above tolerance.
    NoConvergence {
        iterations: usize,
        residual_norm: f64,
        reason: String,
    },
    /// Jacobian rank was insufficient for the advertised codimension.
    RankDeficient {
        rank: usize,
        required: usize,
        residual_norm: f64,
    },
    /// Constraint domain rejected the current/candidate point.
    OutsideDomain { reason: String },
    /// Non-finite arithmetic occurred during an otherwise well-shaped solve.
    NonFinite { stage: &'static str },
    /// The explicit iteration budget was exhausted.
    BudgetExceeded {
        iterations: usize,
        residual_norm: f64,
    },
}

/// Projection result retains the final local iterate regardless of outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionResult {
    /// Final local iterate. This is not necessarily constraint-valid or regular.
    pub state: Vec<f64>,
    /// Explicit bounded-solver outcome.
    pub outcome: ProjectionOutcome,
}

/// Orthonormal tangent basis at one qualified regular point.
#[derive(Clone, Debug, PartialEq)]
pub struct TangentBasis {
    /// Ambient-coordinate basis vectors.
    pub vectors: Vec<Vec<f64>>,
    /// Rank of the constraint Jacobian used to construct this basis.
    pub constraint_rank: usize,
}

/// Euclidean implicit equality manifold with a profile-bound local solver.
#[derive(Clone, Debug)]
pub struct ImplicitManifold<C> {
    ambient: EuclideanSpace,
    constraint: C,
    policy: ProjectionPolicy,
    profile: ImplicitManifoldProfile,
}

impl<C> ImplicitManifold<C>
where
    C: EqualityConstraint,
{
    /// Construct a bounded implicit-manifold problem.
    pub fn new(constraint: C, policy: ProjectionPolicy) -> Result<Self, ImplicitManifoldError> {
        let ambient_dimension = constraint.ambient_dimension();
        let codimension = constraint.codimension();
        if ambient_dimension == 0 || ambient_dimension > MAX_IMPLICIT_AMBIENT_DIMENSION {
            return Err(ImplicitManifoldError::InvalidDimensions {
                reason: format!(
                    "ambient dimension must be in 1..={MAX_IMPLICIT_AMBIENT_DIMENSION}, got {ambient_dimension}"
                ),
            });
        }
        if codimension == 0
            || codimension > MAX_IMPLICIT_CODIMENSION
            || codimension > ambient_dimension
        {
            return Err(ImplicitManifoldError::InvalidDimensions {
                reason: format!(
                    "codimension must be in 1..=min({MAX_IMPLICIT_CODIMENSION}, ambient), got {codimension}"
                ),
            });
        }

        let ambient = EuclideanSpace::new(ambient_dimension);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-implicit-manifold-profile-v1\0");
        hasher.update(&ambient.profile().identity());
        hasher.update(&constraint.profile().identity());
        policy.update_hash(&mut hasher);
        let identity = *hasher.finalize().as_bytes();

        Ok(Self {
            ambient,
            constraint,
            policy,
            profile: ImplicitManifoldProfile {
                ambient_dimension,
                codimension,
                identity,
            },
        })
    }

    /// Exact problem profile.
    pub fn profile(&self) -> &ImplicitManifoldProfile {
        &self.profile
    }

    /// Ambient Euclidean state space.
    pub fn ambient(&self) -> &EuclideanSpace {
        &self.ambient
    }

    /// Analytic constraint implementation.
    pub fn constraint(&self) -> &C {
        &self.constraint
    }

    /// Exact local projection policy.
    pub fn projection_policy(&self) -> ProjectionPolicy {
        self.policy
    }

    /// Qualify one ambient state against the declared equality tolerance.
    ///
    /// This establishes constraint satisfaction only. Use [`Self::tangent_basis`]
    /// or [`Self::project`] when local regularity is also required.
    pub fn validate_state(&self, state: &[f64]) -> ConstraintValidity {
        let owned = state.to_vec();
        let ambient_validity = self.ambient.validate_state(&owned);
        if !ambient_validity.is_valid() {
            return ConstraintValidity::AmbientInvalid(ambient_validity);
        }

        match self.constraint.domain_status(state) {
            Ok(DomainStatus::Inside) => {}
            Ok(DomainStatus::Outside { reason }) => {
                return ConstraintValidity::OutsideDomain { reason };
            }
            Err(error) => {
                return ConstraintValidity::EvaluationFailed {
                    reason: error.to_string(),
                };
            }
        }

        match self.residual_checked(state) {
            Ok(residual) => {
                let residual_norm = stable_norm(&residual);
                if !residual_norm.is_finite() {
                    ConstraintValidity::EvaluationFailed {
                        reason: "residual norm became non-finite".to_string(),
                    }
                } else if residual_norm <= self.policy.residual_tolerance {
                    ConstraintValidity::Valid { residual_norm }
                } else {
                    ConstraintValidity::ResidualExceeded {
                        residual_norm,
                        tolerance: self.policy.residual_tolerance,
                    }
                }
            }
            Err(error) => ConstraintValidity::EvaluationFailed {
                reason: error.to_string(),
            },
        }
    }

    /// Project one ambient state with a bounded normal-space Newton iteration.
    ///
    /// The correction solves `J J^T λ = -F`, then applies `δ = J^T λ`.
    /// Rank is checked independently before every success classification.
    pub fn project(&self, state: &[f64]) -> Result<ProjectionResult, ImplicitManifoldError> {
        if state.len() != self.profile.ambient_dimension {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: self.profile.ambient_dimension,
                actual: state.len(),
            });
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Ok(ProjectionResult {
                state: state.to_vec(),
                outcome: ProjectionOutcome::NonFinite {
                    stage: "ambient state",
                },
            });
        }

        let mut current = state.to_vec();
        match self.constraint.domain_status(&current) {
            Ok(DomainStatus::Inside) => {}
            Ok(DomainStatus::Outside { reason }) => {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::OutsideDomain { reason },
                });
            }
            Err(error) => return projection_evaluation_error(error, current),
        }

        let mut residual = match self.residual_checked(&current) {
            Ok(residual) => residual,
            Err(error) => return projection_evaluation_error(error, current),
        };
        let mut residual_norm = stable_norm(&residual);
        if !residual_norm.is_finite() {
            return Ok(ProjectionResult {
                state: current,
                outcome: ProjectionOutcome::NonFinite {
                    stage: "initial residual norm",
                },
            });
        }

        let m = self.profile.codimension;
        let n = self.profile.ambient_dimension;
        if residual_norm <= self.policy.residual_tolerance {
            return self.classify_satisfied(current, residual_norm);
        }

        for iteration in 1..=self.policy.max_iterations {
            let jacobian = match self.jacobian_checked(&current) {
                Ok(jacobian) => jacobian,
                Err(error) => return projection_evaluation_error(error, current),
            };
            let row_basis = match orthonormal_row_basis(
                &jacobian,
                m,
                n,
                self.policy.rank_tolerance,
            ) {
                Ok(basis) => basis,
                Err(error) => return projection_evaluation_error(error, current),
            };
            let rank = row_basis.len();
            if rank < m {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::RankDeficient {
                        rank,
                        required: m,
                        residual_norm,
                    },
                });
            }

            let j = DMatrix::from_row_slice(m, n, &jacobian);
            let gram = &j * j.transpose();
            let rhs = -DVector::from_vec(residual.clone());
            let Some(multiplier) = gram.lu().solve(&rhs) else {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::NoConvergence {
                        iterations: iteration,
                        residual_norm,
                        reason: "full-rank precheck passed but normal-equation solve failed"
                            .to_string(),
                    },
                });
            };
            let mut delta = j.transpose() * multiplier;
            let mut step_norm = delta.norm();
            if !step_norm.is_finite() || delta.iter().any(|value| !value.is_finite()) {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::NonFinite {
                        stage: "projection correction",
                    },
                });
            }
            if step_norm > self.policy.max_step_norm {
                let scale = self.policy.max_step_norm / step_norm;
                delta *= scale;
                step_norm = self.policy.max_step_norm;
            }
            if step_norm <= self.policy.step_tolerance {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::NoConvergence {
                        iterations: iteration,
                        residual_norm,
                        reason: "correction step stagnated above residual tolerance".to_string(),
                    },
                });
            }

            for index in 0..n {
                current[index] += delta[index];
            }
            if current.iter().any(|value| !value.is_finite()) {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::NonFinite {
                        stage: "candidate state",
                    },
                });
            }
            match self.constraint.domain_status(&current) {
                Ok(DomainStatus::Inside) => {}
                Ok(DomainStatus::Outside { reason }) => {
                    return Ok(ProjectionResult {
                        state: current,
                        outcome: ProjectionOutcome::OutsideDomain { reason },
                    });
                }
                Err(error) => return projection_evaluation_error(error, current),
            }

            residual = match self.residual_checked(&current) {
                Ok(residual) => residual,
                Err(error) => return projection_evaluation_error(error, current),
            };
            residual_norm = stable_norm(&residual);
            if !residual_norm.is_finite() {
                return Ok(ProjectionResult {
                    state: current,
                    outcome: ProjectionOutcome::NonFinite {
                        stage: "updated residual norm",
                    },
                });
            }
            if residual_norm <= self.policy.residual_tolerance {
                // Rank was full at the pre-step point, but regularity is local;
                // re-check at the converged point before declaring success.
                let classified = self.classify_satisfied(current, residual_norm)?;
                return Ok(match classified.outcome {
                    ProjectionOutcome::AlreadySatisfied { residual_norm } => ProjectionResult {
                        state: classified.state,
                        outcome: ProjectionOutcome::Projected {
                            iterations: iteration,
                            residual_norm,
                        },
                    },
                    _ => classified,
                });
            }
        }

        Ok(ProjectionResult {
            state: current,
            outcome: ProjectionOutcome::BudgetExceeded {
                iterations: self.policy.max_iterations,
                residual_norm,
            },
        })
    }

    /// Construct an orthonormal tangent basis at a qualified regular point.
    pub fn tangent_basis(&self, state: &[f64]) -> Result<TangentBasis, ImplicitManifoldError> {
        self.require_ambient(state)?;
        match self.validate_state(state) {
            ConstraintValidity::Valid { .. } => {}
            ConstraintValidity::ResidualExceeded { residual_norm, .. } => {
                return Err(ImplicitManifoldError::StateNotOnManifold { residual_norm });
            }
            other => {
                return Err(ImplicitManifoldError::StateNotQualified {
                    reason: format!("{other:?}"),
                });
            }
        }

        let m = self.profile.codimension;
        let n = self.profile.ambient_dimension;
        let jacobian = self.jacobian_checked(state)?;
        let normal_basis =
            orthonormal_row_basis(&jacobian, m, n, self.policy.rank_tolerance)?;
        if normal_basis.len() < m {
            return Err(ImplicitManifoldError::RankDeficient {
                rank: normal_basis.len(),
                required: m,
            });
        }

        let target = n - m;
        let mut tangents: Vec<Vec<f64>> = Vec::with_capacity(target);
        for axis in 0..n {
            if tangents.len() == target {
                break;
            }
            let mut candidate = vec![0.0; n];
            candidate[axis] = 1.0;
            remove_components(&mut candidate, &normal_basis);
            remove_components(&mut candidate, &tangents);
            let norm = stable_norm(&candidate);
            if norm > self.policy.rank_tolerance && norm.is_finite() {
                for value in &mut candidate {
                    *value /= norm;
                }
                tangents.push(candidate);
            }
        }

        if tangents.len() != target {
            return Err(ImplicitManifoldError::TangentBasisIncomplete {
                expected: target,
                actual: tangents.len(),
            });
        }
        Ok(TangentBasis {
            vectors: tangents,
            constraint_rank: normal_basis.len(),
        })
    }

    fn classify_satisfied(
        &self,
        state: Vec<f64>,
        residual_norm: f64,
    ) -> Result<ProjectionResult, ImplicitManifoldError> {
        let jacobian = match self.jacobian_checked(&state) {
            Ok(jacobian) => jacobian,
            Err(error) => return projection_evaluation_error(error, state),
        };
        let row_basis = match orthonormal_row_basis(
            &jacobian,
            self.profile.codimension,
            self.profile.ambient_dimension,
            self.policy.rank_tolerance,
        ) {
            Ok(basis) => basis,
            Err(error) => return projection_evaluation_error(error, state),
        };
        let rank = row_basis.len();
        if rank < self.profile.codimension {
            Ok(ProjectionResult {
                state,
                outcome: ProjectionOutcome::RankDeficient {
                    rank,
                    required: self.profile.codimension,
                    residual_norm,
                },
            })
        } else {
            Ok(ProjectionResult {
                state,
                outcome: ProjectionOutcome::AlreadySatisfied { residual_norm },
            })
        }
    }

    fn require_ambient(&self, state: &[f64]) -> Result<(), ImplicitManifoldError> {
        if state.len() != self.profile.ambient_dimension {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: self.profile.ambient_dimension,
                actual: state.len(),
            });
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(ImplicitManifoldError::NonFinite {
                stage: "ambient state",
            });
        }
        Ok(())
    }

    fn residual_checked(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        let residual = self.constraint.residual(state)?;
        if residual.len() != self.profile.codimension {
            return Err(ImplicitManifoldError::ConstraintShape {
                kind: self.constraint.profile().kind().to_string(),
                output: "residual values",
                expected: self.profile.codimension,
                actual: residual.len(),
            });
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(ImplicitManifoldError::NonFinite { stage: "residual" });
        }
        Ok(residual)
    }

    fn jacobian_checked(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        let jacobian = self.constraint.jacobian(state)?;
        let expected = self.profile.codimension * self.profile.ambient_dimension;
        if jacobian.len() != expected {
            return Err(ImplicitManifoldError::ConstraintShape {
                kind: self.constraint.profile().kind().to_string(),
                output: "Jacobian values",
                expected,
                actual: jacobian.len(),
            });
        }
        if jacobian.iter().any(|value| !value.is_finite()) {
            return Err(ImplicitManifoldError::NonFinite { stage: "Jacobian" });
        }
        Ok(jacobian)
    }
}

fn projection_evaluation_error(
    error: ImplicitManifoldError,
    state: Vec<f64>,
) -> Result<ProjectionResult, ImplicitManifoldError> {
    match error {
        ImplicitManifoldError::NonFinite { stage } => Ok(ProjectionResult {
            state,
            outcome: ProjectionOutcome::NonFinite { stage },
        }),
        other => Err(other),
    }
}

fn stable_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn remove_components(vector: &mut [f64], basis: &[Vec<f64>]) {
    for direction in basis {
        let coefficient = dot(vector, direction);
        for (value, component) in vector.iter_mut().zip(direction) {
            *value -= coefficient * component;
        }
    }
}

fn orthonormal_row_basis(
    jacobian: &[f64],
    rows: usize,
    columns: usize,
    rank_tolerance: f64,
) -> Result<Vec<Vec<f64>>, ImplicitManifoldError> {
    if jacobian.len() != rows * columns {
        return Err(ImplicitManifoldError::ConstraintShape {
            kind: "internal-jacobian".to_string(),
            output: "row-major values",
            expected: rows * columns,
            actual: jacobian.len(),
        });
    }
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * columns;
        let mut candidate = jacobian[start..start + columns].to_vec();
        remove_components(&mut candidate, &basis);
        let norm = stable_norm(&candidate);
        if !norm.is_finite() {
            return Err(ImplicitManifoldError::NonFinite {
                stage: "Jacobian rank basis",
            });
        }
        if norm > rank_tolerance {
            for value in &mut candidate {
                *value /= norm;
            }
            basis.push(candidate);
        }
    }
    Ok(basis)
}

/// Analytic sphere constraint `||x||² - r² = 0` in `R^n`.
#[derive(Clone, Debug)]
pub struct SphereConstraint {
    dimension: usize,
    radius: f64,
    profile: ConstraintProfile,
}

impl SphereConstraint {
    /// Construct an `S^(n-1)` equality constraint.
    pub fn new(dimension: usize, radius: f64) -> Result<Self, ImplicitManifoldError> {
        if dimension < 2 || dimension > MAX_IMPLICIT_AMBIENT_DIMENSION {
            return Err(ImplicitManifoldError::InvalidDimensions {
                reason: format!(
                    "sphere ambient dimension must be in 2..={MAX_IMPLICIT_AMBIENT_DIMENSION}"
                ),
            });
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(ImplicitManifoldError::InvalidPolicy {
                reason: format!("sphere radius must be finite and > 0, got {radius}"),
            });
        }
        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&(dimension as u64).to_le_bytes());
        parameters.extend_from_slice(&radius.to_bits().to_le_bytes());
        Ok(Self {
            dimension,
            radius,
            profile: ConstraintProfile::new("sphere-equality-v1", parameters),
        })
    }

    /// Sphere radius.
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl EqualityConstraint for SphereConstraint {
    fn profile(&self) -> &ConstraintProfile {
        &self.profile
    }

    fn ambient_dimension(&self) -> usize {
        self.dimension
    }

    fn codimension(&self) -> usize {
        1
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        if state.len() != self.dimension {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        let squared = state.iter().map(|value| value * value).sum::<f64>();
        let value = squared - self.radius * self.radius;
        if value.is_finite() {
            Ok(vec![value])
        } else {
            Err(ImplicitManifoldError::NonFinite {
                stage: "sphere residual",
            })
        }
    }

    fn jacobian(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        if state.len() != self.dimension {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        let result: Vec<f64> = state.iter().map(|value| 2.0 * value).collect();
        if result.iter().all(|value| value.is_finite()) {
            Ok(result)
        } else {
            Err(ImplicitManifoldError::NonFinite {
                stage: "sphere Jacobian",
            })
        }
    }
}

/// Polynomial torus constraint in `R³`.
///
/// For major radius `R` and minor radius `r`:
/// `(x²+y²+z²+R²-r²)² - 4R²(x²+y²) = 0`.
#[derive(Clone, Debug)]
pub struct TorusConstraint {
    major_radius: f64,
    minor_radius: f64,
    profile: ConstraintProfile,
}

impl TorusConstraint {
    /// Construct a regular ring torus with `R > r > 0`.
    pub fn new(major_radius: f64, minor_radius: f64) -> Result<Self, ImplicitManifoldError> {
        if !major_radius.is_finite()
            || !minor_radius.is_finite()
            || minor_radius <= 0.0
            || major_radius <= minor_radius
        {
            return Err(ImplicitManifoldError::InvalidPolicy {
                reason: format!(
                    "torus radii must satisfy finite R > r > 0, got R={major_radius}, r={minor_radius}"
                ),
            });
        }
        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&major_radius.to_bits().to_le_bytes());
        parameters.extend_from_slice(&minor_radius.to_bits().to_le_bytes());
        Ok(Self {
            major_radius,
            minor_radius,
            profile: ConstraintProfile::new("ring-torus-polynomial-v1", parameters),
        })
    }

    /// Major radius `R`.
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Minor radius `r`.
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }
}

impl EqualityConstraint for TorusConstraint {
    fn profile(&self) -> &ConstraintProfile {
        &self.profile
    }

    fn ambient_dimension(&self) -> usize {
        3
    }

    fn codimension(&self) -> usize {
        1
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        if state.len() != 3 {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: 3,
                actual: state.len(),
            });
        }
        let [x, y, z] = [state[0], state[1], state[2]];
        let major_sq = self.major_radius * self.major_radius;
        let minor_sq = self.minor_radius * self.minor_radius;
        let s = x * x + y * y + z * z + major_sq - minor_sq;
        let value = s * s - 4.0 * major_sq * (x * x + y * y);
        if value.is_finite() {
            Ok(vec![value])
        } else {
            Err(ImplicitManifoldError::NonFinite {
                stage: "torus residual",
            })
        }
    }

    fn jacobian(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
        if state.len() != 3 {
            return Err(ImplicitManifoldError::DimensionMismatch {
                expected: 3,
                actual: state.len(),
            });
        }
        let [x, y, z] = [state[0], state[1], state[2]];
        let major_sq = self.major_radius * self.major_radius;
        let minor_sq = self.minor_radius * self.minor_radius;
        let s = x * x + y * y + z * z + major_sq - minor_sq;
        let result = vec![
            4.0 * x * (s - 2.0 * major_sq),
            4.0 * y * (s - 2.0 * major_sq),
            4.0 * z * s,
        ];
        if result.iter().all(|value| value.is_finite()) {
            Ok(result)
        } else {
            Err(ImplicitManifoldError::NonFinite {
                stage: "torus Jacobian",
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct SingularSatisfiedConstraint {
        profile: ConstraintProfile,
    }

    impl SingularSatisfiedConstraint {
        fn new() -> Self {
            Self {
                profile: ConstraintProfile::new("singular-squared-origin-fixture-v1", Vec::new()),
            }
        }
    }

    impl EqualityConstraint for SingularSatisfiedConstraint {
        fn profile(&self) -> &ConstraintProfile {
            &self.profile
        }

        fn ambient_dimension(&self) -> usize {
            1
        }

        fn codimension(&self) -> usize {
            1
        }

        fn residual(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
            Ok(vec![state[0] * state[0]])
        }

        fn jacobian(&self, state: &[f64]) -> Result<Vec<f64>, ImplicitManifoldError> {
            Ok(vec![2.0 * state[0]])
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn sphere_projection_and_already_satisfied_are_distinct() {
        let manifold = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();

        let already = manifold.project(&[1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            already.outcome,
            ProjectionOutcome::AlreadySatisfied { .. }
        ));

        let projected = manifold.project(&[2.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            projected.outcome,
            ProjectionOutcome::Projected { .. }
        ));
        close(projected.state[0], 1.0, 1e-9);
        assert!(manifold.validate_state(&projected.state).is_valid());
    }

    #[test]
    fn satisfied_singular_point_is_not_reported_regular() {
        let manifold = ImplicitManifold::new(
            SingularSatisfiedConstraint::new(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        assert!(manifold.validate_state(&[0.0]).is_valid());
        let projection = manifold.project(&[0.0]).unwrap();
        assert!(matches!(
            projection.outcome,
            ProjectionOutcome::RankDeficient {
                rank: 0,
                required: 1,
                ..
            }
        ));
    }

    #[test]
    fn sphere_tangent_basis_is_orthogonal_to_radial_normal() {
        let manifold = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        let point = [1.0, 0.0, 0.0];
        let basis = manifold.tangent_basis(&point).unwrap();
        assert_eq!(basis.vectors.len(), 2);
        for tangent in &basis.vectors {
            close(dot(&point, tangent), 0.0, 1e-12);
            close(stable_norm(tangent), 1.0, 1e-12);
        }
    }

    #[test]
    fn sphere_origin_reports_rank_deficiency_not_impossibility() {
        let manifold = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        let result = manifold.project(&[0.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            result.outcome,
            ProjectionOutcome::RankDeficient {
                rank: 0,
                required: 1,
                ..
            }
        ));
        assert!(matches!(
            manifold.validate_state(&result.state),
            ConstraintValidity::ResidualExceeded { .. }
        ));
    }

    #[test]
    fn non_finite_projection_input_is_an_outcome_not_a_geometry_claim() {
        let manifold = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        let result = manifold.project(&[f64::NAN, 0.0, 0.0]).unwrap();
        assert!(matches!(
            result.outcome,
            ProjectionOutcome::NonFinite {
                stage: "ambient state"
            }
        ));
    }

    #[test]
    fn torus_known_points_and_projection_are_qualified() {
        let manifold = ImplicitManifold::new(
            TorusConstraint::new(2.0, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        assert!(manifold.validate_state(&[3.0, 0.0, 0.0]).is_valid());
        assert!(!manifold.validate_state(&[0.0, 0.0, 0.0]).is_valid());

        let projected = manifold.project(&[3.2, 0.0, 0.0]).unwrap();
        assert!(matches!(
            projected.outcome,
            ProjectionOutcome::Projected { .. }
        ));
        close(projected.state[0], 3.0, 1e-8);
        assert!(manifold.validate_state(&projected.state).is_valid());
    }

    #[test]
    fn profile_identity_binds_constraint_and_solver_policy() {
        let sphere_one = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        let sphere_two = ImplicitManifold::new(
            SphereConstraint::new(3, 2.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap();
        let changed_policy = ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::new(1e-8, 1e-12, 1e-13, 1.0, 32).unwrap(),
        )
        .unwrap();
        assert_ne!(sphere_one.profile().identity(), sphere_two.profile().identity());
        assert_ne!(
            sphere_one.profile().identity(),
            changed_policy.profile().identity()
        );
    }
}
