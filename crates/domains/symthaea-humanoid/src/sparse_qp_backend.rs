// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pluggable sparse-QP backend boundary.
//!
//! The default implementation keeps the deterministic in-crate active-set
//! solver. Production deployments can implement this trait with a fixed-
//! sparsity OSQP, qpOASES, ProxQP, or hardware-specific backend while retaining
//! the same validation, warm-start, residual, and deadline gates above it.
//!
//! [`ProcessSparseQpBackend`] is a concrete interoperability adapter. It speaks
//! a versioned JSON protocol to an external solver executable and independently
//! revalidates every returned solution. It intentionally launches one process
//! per solve, so it is suitable for oracle generation, integration testing, and
//! solver qualification—not a hard-real-time control loop.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::csc_qp::canonical_qp_structure_fingerprint;
use crate::equality_qp::{
    DenseEqualityQpSolver, DenseEqualityQuadraticProgram, EqualityQpSolution,
    EqualityQpSolverConfig,
};

pub const SPARSE_QP_WIRE_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct SparseQpBackendSolution {
    pub solution: EqualityQpSolution,
    pub backend_id: String,
    pub symbolic_pattern_reused: bool,
}

pub trait SparseQpBackend: Send + Sync {
    fn backend_id(&self) -> &str;

    fn solve(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        structure_fingerprint: u64,
        warm_start: Option<&[f64]>,
    ) -> Option<SparseQpBackendSolution>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQpWireRequest {
    pub protocol_version: u32,
    pub request_id: u64,
    pub requested_backend_id: String,
    pub structure_fingerprint: u64,
    pub problem: DenseEqualityQuadraticProgram,
    pub warm_start: Option<Vec<f64>>,
}

impl SparseQpWireRequest {
    pub fn validate(&self) -> bool {
        self.protocol_version == SPARSE_QP_WIRE_PROTOCOL_VERSION
            && self.request_id != 0
            && !self.requested_backend_id.trim().is_empty()
            && self.structure_fingerprint == canonical_qp_structure_fingerprint(&self.problem)
            && self.problem.validate()
            && self
                .warm_start
                .as_ref()
                .map(|values| {
                    values.len() == self.problem.diagonal_hessian.len()
                        && values.iter().all(|value| value.is_finite())
                })
                .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQpWireResponse {
    pub protocol_version: u32,
    pub request_id: u64,
    pub backend_id: String,
    pub symbolic_pattern_reused: bool,
    pub solution: Option<EqualityQpSolution>,
    pub error: Option<String>,
}

impl SparseQpWireResponse {
    pub fn success(
        request_id: u64,
        backend_id: impl Into<String>,
        symbolic_pattern_reused: bool,
        solution: EqualityQpSolution,
    ) -> Self {
        Self {
            protocol_version: SPARSE_QP_WIRE_PROTOCOL_VERSION,
            request_id,
            backend_id: backend_id.into(),
            symbolic_pattern_reused,
            solution: Some(solution),
            error: None,
        }
    }

    pub fn failure(
        request_id: u64,
        backend_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: SPARSE_QP_WIRE_PROTOCOL_VERSION,
            request_id,
            backend_id: backend_id.into(),
            symbolic_pattern_reused: false,
            solution: None,
            error: Some(error.into()),
        }
    }
}

pub struct DeterministicActiveSetBackend {
    solver: DenseEqualityQpSolver,
}

impl DeterministicActiveSetBackend {
    pub fn new(config: EqualityQpSolverConfig) -> Self {
        Self {
            solver: DenseEqualityQpSolver::with_config(config),
        }
    }
}

impl SparseQpBackend for DeterministicActiveSetBackend {
    fn backend_id(&self) -> &str {
        "symthaea-dense-active-set-v2"
    }

    fn solve(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        _structure_fingerprint: u64,
        warm_start: Option<&[f64]>,
    ) -> Option<SparseQpBackendSolution> {
        let symbolic_pattern_reused = warm_start.is_some();
        Some(SparseQpBackendSolution {
            solution: self.solver.solve_with_warm_start(problem, warm_start)?,
            backend_id: self.backend_id().to_string(),
            symbolic_pattern_reused,
        })
    }
}

/// One-shot external solver adapter using a versioned JSON request/response.
///
/// The child executable receives exactly one request on stdin and must emit
/// exactly one JSON response on stdout. Stderr is inherited for diagnostics.
/// Returned values are revalidated against the original QP before admission.
#[derive(Debug, Clone)]
pub struct ProcessSparseQpBackend {
    executable: PathBuf,
    arguments: Vec<String>,
    backend_id: String,
    maximum_response_bytes: usize,
    solution_tolerance: f64,
}

impl ProcessSparseQpBackend {
    pub fn new(executable: impl Into<PathBuf>, backend_id: impl Into<String>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            backend_id: backend_id.into(),
            maximum_response_bytes: 8 * 1024 * 1024,
            solution_tolerance: 1.0e-6,
        }
    }

    pub fn with_arguments(mut self, arguments: impl IntoIterator<Item = String>) -> Self {
        self.arguments = arguments.into_iter().collect();
        self
    }

    pub fn with_maximum_response_bytes(mut self, maximum_response_bytes: usize) -> Self {
        self.maximum_response_bytes = maximum_response_bytes.max(1024);
        self
    }

    pub fn with_solution_tolerance(mut self, solution_tolerance: f64) -> Self {
        if solution_tolerance.is_finite() && solution_tolerance >= 0.0 {
            self.solution_tolerance = solution_tolerance;
        }
        self
    }

    pub fn executable(&self) -> &Path {
        &self.executable
    }

    fn solve_external(&self, request: &SparseQpWireRequest) -> Option<SparseQpWireResponse> {
        if !request.validate()
            || self.backend_id.trim().is_empty()
            || self.executable.as_os_str().is_empty()
        {
            return None;
        }
        let payload = serde_json::to_vec(request).ok()?;
        let mut child = Command::new(&self.executable)
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .ok()?;
        {
            let mut stdin = child.stdin.take()?;
            stdin.write_all(&payload).ok()?;
            stdin.write_all(b"\n").ok()?;
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success()
            || output.stdout.is_empty()
            || output.stdout.len() > self.maximum_response_bytes
        {
            return None;
        }
        serde_json::from_slice(&output.stdout).ok()
    }
}

impl SparseQpBackend for ProcessSparseQpBackend {
    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn solve(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        structure_fingerprint: u64,
        warm_start: Option<&[f64]>,
    ) -> Option<SparseQpBackendSolution> {
        let request_id = structure_fingerprint.max(1);
        let request = SparseQpWireRequest {
            protocol_version: SPARSE_QP_WIRE_PROTOCOL_VERSION,
            request_id,
            requested_backend_id: self.backend_id.clone(),
            structure_fingerprint,
            problem: problem.clone(),
            warm_start: warm_start.map(ToOwned::to_owned),
        };
        let response = self.solve_external(&request)?;
        if response.protocol_version != SPARSE_QP_WIRE_PROTOCOL_VERSION
            || response.request_id != request_id
            || response.backend_id != self.backend_id
            || response.error.is_some()
        {
            return None;
        }
        let solution = response.solution?;
        if !validate_external_solution(problem, &solution, self.solution_tolerance) {
            return None;
        }
        Some(SparseQpBackendSolution {
            solution,
            backend_id: response.backend_id,
            symbolic_pattern_reused: response.symbolic_pattern_reused,
        })
    }
}

/// Independently validate an externally produced solution.
pub fn validate_external_solution(
    problem: &DenseEqualityQuadraticProgram,
    solution: &EqualityQpSolution,
    tolerance: f64,
) -> bool {
    if !problem.validate() || !tolerance.is_finite() || tolerance < 0.0 {
        return false;
    }
    let variables = problem.diagonal_hessian.len();
    if solution.values.len() != variables
        || solution.values.iter().any(|value| !value.is_finite())
        || solution.multipliers.iter().any(|value| !value.is_finite())
        || !solution.maximum_equality_residual.is_finite()
        || !solution.maximum_bound_violation.is_finite()
        || !solution.objective.is_finite()
    {
        return false;
    }
    let maximum_equality_residual = problem
        .equality_matrix
        .iter()
        .zip(problem.equality_target.iter())
        .map(|(row, target)| {
            (row.iter()
                .zip(solution.values.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
                - target)
                .abs()
        })
        .fold(0.0, f64::max);
    let maximum_bound_violation = (0..variables)
        .map(|index| {
            (problem.lower_bounds[index] - solution.values[index])
                .max(solution.values[index] - problem.upper_bounds[index])
                .max(0.0)
        })
        .fold(0.0, f64::max);
    let objective = (0..variables)
        .map(|index| {
            0.5 * problem.diagonal_hessian[index] * solution.values[index].powi(2)
                + problem.linear_term[index] * solution.values[index]
        })
        .sum::<f64>();
    let scale = 1.0 + objective.abs();
    let metadata_consistent = (solution.maximum_equality_residual - maximum_equality_residual)
        .abs()
        <= tolerance.max(1.0e-10)
        && (solution.maximum_bound_violation - maximum_bound_violation).abs()
            <= tolerance.max(1.0e-10)
        && (solution.objective - objective).abs() <= tolerance.max(1.0e-10) * scale;
    let feasible = maximum_equality_residual <= tolerance && maximum_bound_violation <= tolerance;
    metadata_consistent && solution.converged == feasible
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> DenseEqualityQuadraticProgram {
        DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![0.0, 0.0],
            lower_bounds: vec![-1.0, -1.0],
            upper_bounds: vec![1.0, 1.0],
            equality_matrix: vec![vec![1.0, 1.0]],
            equality_target: vec![0.0],
        }
    }

    #[test]
    fn deterministic_backend_reports_warm_pattern_reuse() {
        let problem = problem();
        let backend = DeterministicActiveSetBackend::new(EqualityQpSolverConfig::default());
        let result = backend.solve(&problem, 7, Some(&[0.0, 0.0])).unwrap();
        assert!(result.solution.converged);
        assert!(result.symbolic_pattern_reused);
        assert!(!result.backend_id.is_empty());
    }

    #[test]
    fn external_solution_metadata_is_recomputed() {
        let problem = problem();
        let solution = DenseEqualityQpSolver::new().solve(&problem).unwrap();
        assert!(validate_external_solution(&problem, &solution, 1.0e-7));
        let mut tampered = solution;
        tampered.objective += 1.0;
        assert!(!validate_external_solution(&problem, &tampered, 1.0e-7));
    }

    #[test]
    fn wire_request_rejects_invalid_warm_start() {
        let request = SparseQpWireRequest {
            protocol_version: SPARSE_QP_WIRE_PROTOCOL_VERSION,
            request_id: 1,
            requested_backend_id: "worker".to_string(),
            structure_fingerprint: canonical_qp_structure_fingerprint(&problem()),
            problem: problem(),
            warm_start: Some(vec![0.0]),
        };
        assert!(!request.validate());
    }
}
