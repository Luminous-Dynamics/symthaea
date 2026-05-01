#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Math Service — Cognitive Loop Math Dispatcher
//!
//! Unified dispatcher that routes math queries from the cognitive loop
//! to Phase 1-3 solvers (linear algebra, root finding, quadrature,
//! statistics, optimization, FFT, logic, constraint satisfaction).
//!
//! Receives `MathIntent`, classifies problem type, routes to solver,
//! multi-path verifies where possible, and returns `MathResponse`
//! with HDC encoding, Phi, and epistemic confidence.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::computational_geometry::{GeometryEngine, Point2D, Polygon};
use symthaea_core::hdc::constraint_solver::{CSPSolver, CSP};
use symthaea_core::hdc::differential_equations::DifferentialEquationsEngine;
use symthaea_core::hdc::fft::FftEngine;
use symthaea_core::hdc::graph_theory::Graph;
use symthaea_core::hdc::linear_algebra::{HdcMatrix, HdcVector, LinearAlgebraEngine};
use symthaea_core::hdc::logic_engine::{LogicEngine, Proposition};
use symthaea_core::hdc::optimization::OptimizationEngine;
use symthaea_core::hdc::primitive_system::seed_from_name;
use symthaea_core::hdc::quadrature::QuadratureEngine;
use symthaea_core::hdc::root_finding::RootFindingEngine;
use symthaea_core::hdc::statistics;

// Symbolic calculus for exact integration and derivative-assisted root finding
use symthaea_core::hdc::arithmetic_engine::Polynomial;
use symthaea_core::hdc::calculus::{SymbolicDifferentiator, SymbolicIntegrator};

use super::thresholds::{
    MATH_DEFAULT_TELEMETRY_CONFIDENCE, MATH_DEFAULT_TELEMETRY_PHI,
    MATH_INTEGRATION_UNVERIFIED_CONFIDENCE, MATH_INTEGRATION_VERIFIED_CONFIDENCE,
    MATH_LINEAR_UNVERIFIED_CONFIDENCE, MATH_LINEAR_VERIFIED_CONFIDENCE, MATH_MULTIPATH_PHI_BOOST,
    MATH_OPTIMIZATION_CONVERGED_CONFIDENCE, MATH_OPTIMIZATION_FAILED_CONFIDENCE,
    MATH_ROOT_FINDING_CONVERGED_CONFIDENCE, MATH_ROOT_FINDING_FAILED_CONFIDENCE,
    MATH_ROOT_FINDING_VERIFIED_CONFIDENCE, MATH_STATISTICS_CONFIDENCE,
    MATH_STATISTICS_PHI_BASELINE, MATH_SYMBOLIC_EXACT_CONFIDENCE, MATH_SYMBOLIC_EXACT_PHI_BOOST,
    MATH_SYMBOLIC_NUMERIC_AGREEMENT_TOL,
};

// ─── Types ───────────────────────────────────────────────────────────────────

/// Classification of a math problem type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MathProblemType {
    /// Solve Ax = b
    LinearSystem,
    /// Find roots of f(x) = 0
    RootFinding,
    /// Compute ∫f(x)dx
    Integration,
    /// Eigenvalues/SVD
    MatrixAnalysis,
    /// Statistics/probability
    Statistics,
    /// Optimization
    Optimization,
    /// FFT/signal analysis
    SignalAnalysis,
    /// Propositional/FOL logic
    Logic,
    /// Constraint satisfaction
    ConstraintSatisfaction,
    /// Geometry (convex hull, point-in-polygon, etc.)
    Geometry,
    /// Graph theory (shortest path, MST, coloring)
    GraphTheory,
    /// Differential equations (IVP, BVP, PDE)
    DifferentialEquation,
    /// Chemistry (stoichiometry, thermochemistry, kinetics)
    Chemistry,
    /// Proof construction (tactic-based theorem proving)
    Proof,
    /// General arithmetic
    Arithmetic,
    /// Unknown
    Unknown,
}

impl MathProblemType {
    /// Zero-allocation string conversion.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LinearSystem => "LinearSystem",
            Self::RootFinding => "RootFinding",
            Self::Integration => "Integration",
            Self::MatrixAnalysis => "MatrixAnalysis",
            Self::Statistics => "Statistics",
            Self::Optimization => "Optimization",
            Self::SignalAnalysis => "SignalAnalysis",
            Self::Logic => "Logic",
            Self::ConstraintSatisfaction => "ConstraintSatisfaction",
            Self::Geometry => "Geometry",
            Self::GraphTheory => "GraphTheory",
            Self::DifferentialEquation => "DifferentialEquation",
            Self::Chemistry => "Chemistry",
            Self::Proof => "Proof",
            Self::Arithmetic => "Arithmetic",
            Self::Unknown => "Unknown",
        }
    }
}

/// Intent from the language system (math plugin)
#[derive(Debug, Clone)]
pub struct MathIntent {
    /// Raw text of the math query
    pub text: String,
    /// Classified problem type
    pub problem_type: MathProblemType,
    /// Extracted numerical parameters
    pub parameters: Vec<f64>,
    /// HDC encoding of the intent
    pub encoding: BinaryHV,
}

/// Response from the math service
#[derive(Debug, Clone)]
pub struct MathResponse {
    /// Human-readable answer
    pub answer: String,
    /// Numerical result (if applicable)
    pub numerical_result: Option<f64>,
    /// Vector result (if applicable)
    pub vector_result: Option<Vec<f64>>,
    /// HDC encoding of the result
    pub encoding: BinaryHV,
    /// Phi from the computation
    pub phi: f64,
    /// Epistemic confidence (0-1)
    pub confidence: f64,
    /// Whether multi-path verification was performed
    pub multipath_verified: bool,
    /// Problem type that was solved
    pub problem_type: MathProblemType,
    /// Epistemic annotation: what we don't know
    pub epistemic_caveat: Option<String>,
    /// Error bound on numerical result (if applicable)
    pub error_bound: Option<f64>,
}

/// A stored math episode for analogical retrieval
#[derive(Debug, Clone)]
pub struct MathEpisode {
    /// Problem encoding
    pub problem_encoding: BinaryHV,
    /// Solution encoding
    pub solution_encoding: BinaryHV,
    /// Problem type
    pub problem_type: MathProblemType,
    /// Phi achieved
    pub phi: f64,
    /// Brief description
    pub description: String,
    /// Method used to solve (Phase 7c)
    pub method: String,
    /// Timestamp (cycle count) when solved (Phase 7c)
    pub timestamp: u64,
}

// ─── Phase 7a: Phi-Ranked Solution ─────────────────────────────────────────

/// A solution ranked by Phi (information integration measure).
///
/// When multiple solution paths exist for a math problem, each path
/// produces a different Phi score. Higher Phi indicates more elegant,
/// more integrated reasoning — shorter proofs with more verification
/// paths that agree with other methods.
///
/// Science: Dehaene (2007) — mathematical beauty correlates with
/// neural integration; Rota (1997) — elegance as information compression.
#[derive(Debug, Clone)]
pub struct PhiRankedSolution {
    /// The computed result.
    pub result: f64,
    /// Method name used for this solution path.
    pub method: String,
    /// Phi score for this solution (integration measure).
    pub phi: f64,
    /// Number of proof/computation steps.
    pub proof_steps: usize,
    /// Number of independent verification paths that confirmed this result.
    pub verification_paths: usize,
}

// ─── Phase 7c: Mathematical Memory ─────────────────────────────────────────

/// HDC-encoded mathematical memory for analogical retrieval (Phase 7c).
///
/// Stores solved problems as HDC-encoded episodes. When new problems
/// arrive, searches for similar past problems and transfers solution
/// strategies (analogical reasoning).
///
/// Science: Gentner (1983) — structure-mapping theory of analogy;
/// Hofstadter & Sander (2013) — analogy as core of cognition.
pub struct MathMemory {
    /// HDC-encoded problem-solution pairs.
    episodes: Vec<MathEpisode>,
    /// Maximum episodes to retain.
    capacity: usize,
}

impl MathMemory {
    /// Create a new mathematical memory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::new(),
            capacity,
        }
    }

    /// Find the most similar past problem by HDC cosine similarity.
    pub fn recall(&self, query: &BinaryHV) -> Option<&MathEpisode> {
        if self.episodes.is_empty() {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, ep) in self.episodes.iter().enumerate() {
            let sim = query.similarity(&ep.problem_encoding) as f64;
            if sim > 0.1 {
                if let Some((_, best_sim)) = best {
                    if sim > best_sim {
                        best = Some((i, sim));
                    }
                } else {
                    best = Some((i, sim));
                }
            }
        }
        best.map(|(i, _)| &self.episodes[i])
    }

    /// Find the top-k most similar episodes.
    pub fn recall_top_k(&self, query: &BinaryHV, k: usize) -> Vec<&MathEpisode> {
        let mut scored: Vec<(usize, f64)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| (i, query.similarity(&ep.problem_encoding) as f64))
            .filter(|(_, sim)| *sim > 0.1)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .take(k)
            .map(|(i, _)| &self.episodes[*i])
            .collect()
    }

    /// Store a new solved problem. Evicts lowest-Phi episode if at capacity.
    pub fn remember(&mut self, episode: MathEpisode) {
        if self.episodes.len() >= self.capacity {
            if let Some(min_idx) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.phi
                        .partial_cmp(&b.1.phi)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if episode.phi > self.episodes[min_idx].phi {
                    self.episodes.swap_remove(min_idx);
                } else {
                    return;
                }
            }
        }
        self.episodes.push(episode);
    }

    /// Transfer: suggest method from most similar past problem.
    pub fn suggest_method(&self, problem_encoding: &BinaryHV) -> Option<String> {
        self.recall(problem_encoding).map(|ep| ep.method.clone())
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// All stored episodes (read-only).
    pub fn episodes(&self) -> &[MathEpisode] {
        &self.episodes
    }
}

impl Default for MathMemory {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Telemetry from the math service
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MathServiceTelemetry {
    /// Total problems solved
    pub problems_solved: usize,
    /// Problems by type
    pub by_type: std::collections::HashMap<String, usize>,
    /// Multi-path verification rate
    pub verification_rate: f64,
    /// Average Phi across all solutions
    pub average_phi: f64,
    /// Total Phi accumulated
    pub total_phi: f64,
    /// Average confidence across all solutions (0.0–1.0)
    pub average_confidence: f64,
    /// Total confidence accumulated (for running average computation)
    pub total_confidence: f64,
}

// ─── Math Service ────────────────────────────────────────────────────────────

/// The cognitive loop's unified math dispatcher.
///
/// Routes mathematical queries to the appropriate solver engine,
/// performs multi-path verification where possible, and tracks
/// telemetry for consciousness coupling.
pub struct MathService {
    /// Linear algebra engine (stateful, tracks stats)
    linalg: LinearAlgebraEngine,
    /// Service telemetry
    telemetry: MathServiceTelemetry,
    /// Mathematical memory: stored episodes for analogical retrieval (Phase 7c)
    memory: Vec<MathEpisode>,
    /// Maximum memory capacity
    memory_capacity: usize,
}

/// Compute determinant of an n×n row-major matrix using LU decomposition.
fn lu_determinant(data: &[f64], n: usize) -> f64 {
    let mut mat: Vec<f64> = data.to_vec();
    let mut det = 1.0_f64;
    for col in 0..n {
        // Partial pivoting: find row with max abs value in this column
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return 0.0; // Singular matrix
        }
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            det *= -1.0;
        }
        det *= mat[col * n + col];
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let v = mat[col * n + k];
                mat[row * n + k] -= factor * v;
            }
        }
    }
    det
}

impl MathService {
    /// Create a new math service
    pub fn new() -> Self {
        Self {
            linalg: LinearAlgebraEngine::new(),
            telemetry: MathServiceTelemetry::default(),
            memory: Vec::new(),
            memory_capacity: 256,
        }
    }

    /// Classify a math problem from text keywords
    pub fn classify_problem(text: &str) -> MathProblemType {
        let lower = text.to_lowercase();

        // DE check must precede generic "solve equation" since DE texts contain both keywords
        if lower.contains("differential equation")
            || lower.contains("ode")
            || lower.contains("pde")
            || lower.contains("initial value")
            || lower.contains("boundary value")
        {
            return MathProblemType::DifferentialEquation;
        }
        if lower.contains("solve") && (lower.contains("system") || lower.contains("equation")) {
            if lower.contains("matrix") || lower.contains("linear system") {
                return MathProblemType::LinearSystem;
            }
            return MathProblemType::RootFinding;
        }
        if lower.contains("eigenvalue") || lower.contains("svd") || lower.contains("determinant") {
            return MathProblemType::MatrixAnalysis;
        }
        if lower.contains("integra") || lower.contains("∫") {
            return MathProblemType::Integration;
        }
        if lower.contains("root") || lower.contains("zero") {
            return MathProblemType::RootFinding;
        }
        if lower.contains("mean")
            || lower.contains("variance")
            || lower.contains("probabilit")
            || lower.contains("distribut")
            || lower.contains("bayesian")
            || lower.contains("hypothesis")
            || lower.contains("regression")
        {
            return MathProblemType::Statistics;
        }
        if lower.contains("optimize") || lower.contains("minimi") || lower.contains("maximi") {
            return MathProblemType::Optimization;
        }
        if lower.contains("fft") || lower.contains("fourier") || lower.contains("spectrum") {
            return MathProblemType::SignalAnalysis;
        }
        if lower.contains("prove")
            || lower.contains("theorem")
            || lower.contains("show that")
            || lower.contains("by induction")
            || lower.contains("by contradiction")
        {
            return MathProblemType::Proof;
        }
        if lower.contains("satisf")
            || lower.contains("tautolog")
            || lower.contains("logic")
            || lower.contains("proposition")
        {
            return MathProblemType::Logic;
        }
        if lower.contains("constraint")
            || lower.contains("n-queen")
            || lower.contains("sudoku")
            || lower.contains("color")
        {
            return MathProblemType::ConstraintSatisfaction;
        }
        if lower.contains("convex hull")
            || lower.contains("polygon")
            || lower.contains("geometry")
            || lower.contains("triangle")
            || lower.contains("area")
        {
            return MathProblemType::Geometry;
        }
        if lower.contains("graph")
            || lower.contains("shortest path")
            || lower.contains("spanning tree")
            || lower.contains("bfs")
            || lower.contains("dfs")
            || lower.contains("topological")
        {
            return MathProblemType::GraphTheory;
        }
        if lower.contains("molar mass")
            || lower.contains("stoichiom")
            || lower.contains("enthalpy")
            || lower.contains("gibbs")
            || lower.contains("equilibrium constant")
            || lower.contains("arrhenius")
            || lower.contains("reaction rate")
            || lower.contains("thermochem")
            || lower.contains("combustion")
            || lower.contains("hess")
        {
            return MathProblemType::Chemistry;
        }
        if lower.contains("add")
            || lower.contains("multiply")
            || lower.contains("subtract")
            || lower.contains("divide")
            || lower.contains("factorial")
        {
            return MathProblemType::Arithmetic;
        }

        MathProblemType::Unknown
    }

    /// Solve a linear system Ax = b
    pub fn solve_linear_system(
        &mut self,
        a_data: &[f64],
        rows: usize,
        cols: usize,
        b_data: &[f64],
    ) -> MathResponse {
        let a = HdcMatrix::new(a_data.to_vec(), rows, cols);
        let b = HdcVector::new(b_data.to_vec());
        let (x, result) = self.linalg.solve(&a, &b);

        let answer = format!(
            "Solution: [{}]",
            x.data
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );

        self.record_solve(MathProblemType::LinearSystem, result.phi);

        let response = MathResponse {
            answer,
            numerical_result: None,
            vector_result: Some(x.data),
            encoding: result.encoding,
            phi: result.phi,
            confidence: if result.verified {
                MATH_LINEAR_VERIFIED_CONFIDENCE
            } else {
                MATH_LINEAR_UNVERIFIED_CONFIDENCE
            },
            multipath_verified: result.verified,
            problem_type: MathProblemType::LinearSystem,
            epistemic_caveat: if !result.verified {
                Some("Solution not multi-path verified; condition number may be high".into())
            } else {
                None
            },
            error_bound: None,
        };

        self.store_episode(&response, "linear_system");
        response
    }

    /// Compute descriptive statistics
    pub fn compute_statistics(&mut self, data: &[f64]) -> MathResponse {
        if data.is_empty() {
            return MathResponse {
                answer: "n=0, no data".to_string(),
                numerical_result: Some(0.0),
                vector_result: None,
                encoding: BinaryHV::random(seed_from_name("STATS_EMPTY")),
                phi: 0.0,
                confidence: 0.0,
                multipath_verified: false,
                problem_type: MathProblemType::Statistics,
                epistemic_caveat: Some("Empty dataset".to_string()),
                error_bound: None,
            };
        }
        let m = statistics::mean(data);
        let v = statistics::variance(data);
        let s = statistics::std_dev(data);
        let med = statistics::median(data);
        let (q1, _, q3) = statistics::quartiles(data);
        let sk = statistics::skewness(data);
        let ku = statistics::kurtosis(data);

        let answer = format!(
            "n={}, mean={:.4}, var={:.4}, std={:.4}, median={:.4}, Q1={:.4}, Q3={:.4}, skew={:.4}, kurt={:.4}",
            data.len(), m, v, s, med, q1, q3, sk, ku
        );

        let phi = MATH_STATISTICS_PHI_BASELINE;
        self.record_solve(MathProblemType::Statistics, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!("STATS_{}", m.to_bits())));

        let response = MathResponse {
            answer,
            numerical_result: Some(m),
            vector_result: None,
            encoding,
            phi,
            confidence: MATH_STATISTICS_CONFIDENCE,
            multipath_verified: false,
            problem_type: MathProblemType::Statistics,
            epistemic_caveat: if data.len() < 30 {
                Some(format!(
                    "Small sample (n={}); estimates may be unstable",
                    data.len()
                ))
            } else {
                None
            },
            error_bound: Some(s / (data.len() as f64).sqrt()), // standard error of mean
        };

        self.store_episode(&response, "statistics");
        response
    }

    /// Perform linear regression
    pub fn linear_regression(&mut self, x: &[f64], y: &[f64]) -> MathResponse {
        let result = statistics::linear_regression(x, y);

        let answer = format!(
            "y = {:.4} + {:.4}x, R² = {:.4}, residual_std = {:.4}",
            result.intercept, result.slope, result.r_squared, result.residual_std
        );

        self.record_solve(MathProblemType::Statistics, result.phi);

        let encoding = BinaryHV::random(seed_from_name(&format!(
            "REGR_{}_{}",
            result.slope.to_bits(),
            result.intercept.to_bits()
        )));

        let response = MathResponse {
            answer,
            numerical_result: Some(result.r_squared),
            vector_result: Some(vec![result.intercept, result.slope]),
            encoding,
            phi: result.phi,
            confidence: result.r_squared,
            multipath_verified: false,
            problem_type: MathProblemType::Statistics,
            epistemic_caveat: if result.r_squared < 0.5 {
                Some(format!(
                    "Weak fit (R²={:.3}); linear model may be inappropriate",
                    result.r_squared
                ))
            } else {
                None
            },
            error_bound: Some(result.residual_std),
        };

        self.store_episode(&response, "regression");
        response
    }

    /// Check if a proposition is a tautology
    pub fn check_tautology(&mut self, prop: &Proposition) -> MathResponse {
        let is_taut = LogicEngine::is_tautology(prop);
        let tt = LogicEngine::truth_table(prop);

        let answer = if is_taut {
            format!(
                "{} is a TAUTOLOGY ({}/{} rows true)",
                prop,
                tt.satisfying_count,
                tt.rows.len()
            )
        } else if tt.is_contradiction {
            format!(
                "{} is a CONTRADICTION (0/{} rows true)",
                prop,
                tt.rows.len()
            )
        } else {
            format!(
                "{} is CONTINGENT ({}/{} rows true)",
                prop,
                tt.satisfying_count,
                tt.rows.len()
            )
        };

        let phi = tt.phi;
        self.record_solve(MathProblemType::Logic, phi);

        let response = MathResponse {
            answer,
            numerical_result: if tt.rows.is_empty() {
                Some(0.0)
            } else {
                Some(tt.satisfying_count as f64 / tt.rows.len() as f64)
            },
            vector_result: None,
            encoding: prop.encode(),
            phi,
            confidence: 1.0,
            multipath_verified: false,
            problem_type: MathProblemType::Logic,
            epistemic_caveat: None, // Truth tables are exhaustive
            error_bound: None,
        };

        self.store_episode(&response, "tautology_check");
        response
    }

    /// Solve a SAT problem
    pub fn solve_sat(&mut self, prop: &Proposition) -> MathResponse {
        let (result, proof) = LogicEngine::dpll_sat(prop);

        let answer = match &result {
            Some(assignment) => {
                let vars: Vec<String> = assignment
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect();
                format!("SATISFIABLE: {{{}}}", vars.join(", "))
            }
            None => "UNSATISFIABLE".to_string(),
        };

        self.record_solve(MathProblemType::Logic, proof.phi);

        let response = MathResponse {
            answer,
            numerical_result: Some(if result.is_some() { 1.0 } else { 0.0 }),
            vector_result: None,
            encoding: prop.encode(),
            phi: proof.phi,
            confidence: 1.0,
            multipath_verified: false,
            problem_type: MathProblemType::Logic,
            epistemic_caveat: None,
            error_bound: None,
        };

        self.store_episode(&response, "sat_solve");
        response
    }

    /// Find a root of f(x) = 0 using Brent's method, multi-path verified with bisection
    pub fn find_root<F: Fn(f64) -> f64>(&mut self, f: &F, a: f64, b: f64) -> MathResponse {
        let tol = 1e-10;
        let brent_result = RootFindingEngine::brent(f, a, b, tol);
        let bisect_result = RootFindingEngine::bisection(f, a, b, tol);

        // Multi-path: if both converge and agree within tolerance, verified
        let agreement = (brent_result.root - bisect_result.root).abs();
        let multipath_verified =
            brent_result.converged && bisect_result.converged && agreement < 1e-6;

        // Use Brent as primary (faster convergence), boost phi if verified
        let phi = if multipath_verified {
            (brent_result.phi + bisect_result.phi) / 2.0 * MATH_MULTIPATH_PHI_BOOST
        } else {
            brent_result.phi
        };
        self.record_solve(MathProblemType::RootFinding, phi);
        self.telemetry.verification_rate = {
            let total = self.telemetry.problems_solved as f64;
            let prev = self.telemetry.verification_rate * (total - 1.0).max(0.0);
            (prev + if multipath_verified { 1.0 } else { 0.0 }) / total
        };

        let response = MathResponse {
            answer: format!(
                "Root: {:.10} (residual: {:.2e})",
                brent_result.root, brent_result.residual
            ),
            numerical_result: Some(brent_result.root),
            vector_result: None,
            encoding: brent_result.encoding,
            phi,
            confidence: if multipath_verified {
                MATH_ROOT_FINDING_VERIFIED_CONFIDENCE
            } else if brent_result.converged {
                MATH_ROOT_FINDING_CONVERGED_CONFIDENCE
            } else {
                MATH_ROOT_FINDING_FAILED_CONFIDENCE
            },
            multipath_verified,
            problem_type: MathProblemType::RootFinding,
            epistemic_caveat: if !brent_result.converged {
                Some("Root finding did not converge within tolerance".into())
            } else {
                None
            },
            error_bound: Some(brent_result.residual.abs()),
        };
        self.store_episode(&response, "root_finding");
        response
    }

    /// Compute a definite integral ∫[a,b] f(x) dx, multi-path verified (Simpson vs Gauss-Legendre)
    pub fn integrate<F: Fn(f64) -> f64>(&mut self, f: &F, a: f64, b: f64) -> MathResponse {
        let simpson = QuadratureEngine::adaptive_simpson(f, a, b, 1e-10);
        let gauss = QuadratureEngine::gauss_legendre(f, a, b, 10);

        // Multi-path: if both methods agree within tolerance, verified
        let agreement = (simpson.value - gauss.value).abs();
        let multipath_verified = agreement < 1e-6;

        let phi = if multipath_verified {
            (simpson.phi + gauss.phi) / 2.0 * MATH_MULTIPATH_PHI_BOOST
        } else {
            simpson.phi
        };
        self.record_solve(MathProblemType::Integration, phi);
        self.telemetry.verification_rate = {
            let total = self.telemetry.problems_solved as f64;
            let prev = self.telemetry.verification_rate * (total - 1.0).max(0.0);
            (prev + if multipath_verified { 1.0 } else { 0.0 }) / total
        };

        let response = MathResponse {
            answer: format!("∫[{:.4},{:.4}] = {:.10}", a, b, simpson.value),
            numerical_result: Some(simpson.value),
            vector_result: None,
            encoding: simpson.encoding,
            phi,
            confidence: if multipath_verified {
                MATH_INTEGRATION_VERIFIED_CONFIDENCE
            } else {
                MATH_INTEGRATION_UNVERIFIED_CONFIDENCE
            },
            multipath_verified,
            problem_type: MathProblemType::Integration,
            epistemic_caveat: if !multipath_verified {
                Some(format!(
                    "Simpson vs Gauss-Legendre disagreement: {:.2e}",
                    agreement
                ))
            } else {
                None
            },
            error_bound: simpson.error_estimate,
        };
        self.store_episode(&response, "integration");
        response
    }

    /// Compute a definite integral of a polynomial using exact symbolic arithmetic.
    ///
    /// Returns the exact rational result (numerator/denominator) with perfect confidence.
    /// Cross-validates against numeric quadrature for triple-path verification.
    ///
    /// Science: Rota (1997) — closed-form solutions maximally compress computation,
    /// producing higher Phi than numeric approximation.
    pub fn integrate_symbolic(
        &mut self,
        poly: &Polynomial,
        lower: i64,
        upper: i64,
    ) -> MathResponse {
        // Symbolic: exact rational arithmetic
        let (num, den) = SymbolicIntegrator::definite_integral_exact(poly, lower, upper);
        let symbolic_value = num as f64 / den as f64;

        // Cross-validate with numeric quadrature for triple-path verification
        let coeffs = poly.coefficients();
        let numeric_f = move |x: f64| -> f64 {
            coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| c as f64 * x.powi(i as i32))
                .sum()
        };
        let simpson =
            QuadratureEngine::adaptive_simpson(&numeric_f, lower as f64, upper as f64, 1e-10);
        let gauss = QuadratureEngine::gauss_legendre(&numeric_f, lower as f64, upper as f64, 10);

        // Triple-path: symbolic vs Simpson vs Gauss-Legendre
        let sym_simp_agree =
            (symbolic_value - simpson.value).abs() < MATH_SYMBOLIC_NUMERIC_AGREEMENT_TOL;
        let sym_gauss_agree =
            (symbolic_value - gauss.value).abs() < MATH_SYMBOLIC_NUMERIC_AGREEMENT_TOL;
        let all_agree = sym_simp_agree && sym_gauss_agree;

        // Symbolic exact gets highest Phi; triple-path agreement boosts further
        let base_phi = (simpson.phi + gauss.phi) / 2.0;
        let phi = if all_agree {
            base_phi * MATH_SYMBOLIC_EXACT_PHI_BOOST
        } else {
            base_phi * MATH_MULTIPATH_PHI_BOOST
        };
        self.record_solve(MathProblemType::Integration, phi);
        self.telemetry.verification_rate = {
            let total = self.telemetry.problems_solved as f64;
            let prev = self.telemetry.verification_rate * (total - 1.0).max(0.0);
            (prev + 1.0) / total // Symbolic always counts as verified
        };

        let answer = if den == 1 {
            format!("∫[{},{}] = {} (exact)", lower, upper, num)
        } else {
            format!(
                "∫[{},{}] = {}/{} = {:.10} (exact rational)",
                lower, upper, num, den, symbolic_value
            )
        };

        let response = MathResponse {
            answer,
            numerical_result: Some(symbolic_value),
            vector_result: None,
            encoding: simpson.encoding,
            phi,
            confidence: MATH_SYMBOLIC_EXACT_CONFIDENCE,
            multipath_verified: true,
            problem_type: MathProblemType::Integration,
            epistemic_caveat: if !all_agree {
                Some(format!(
                    "Symbolic-numeric disagreement: Δ_simpson={:.2e}, Δ_gauss={:.2e}",
                    (symbolic_value - simpson.value).abs(),
                    (symbolic_value - gauss.value).abs(),
                ))
            } else {
                None
            },
            error_bound: Some(0.0), // Exact solution has zero error
        };
        self.store_episode(&response, "symbolic_integration");
        response
    }

    /// Integrate with symbolic-first dispatch: tries exact polynomial integration,
    /// falls back to numeric quadrature for non-polynomial functions.
    ///
    /// When `poly` is `Some`, attempts symbolic integration first. If the bounds
    /// are near-integer, uses exact rational arithmetic. Otherwise falls back
    /// to numeric integration.
    pub fn integrate_with_symbolic_fallback<F: Fn(f64) -> f64>(
        &mut self,
        f: &F,
        a: f64,
        b: f64,
        poly: Option<&Polynomial>,
    ) -> MathResponse {
        // Try symbolic path if polynomial representation is available
        if let Some(poly) = poly {
            let a_int = a.round() as i64;
            let b_int = b.round() as i64;
            // Only use symbolic if bounds are exactly integer (no precision loss)
            if (a - a_int as f64).abs() < 1e-12 && (b - b_int as f64).abs() < 1e-12 {
                return self.integrate_symbolic(poly, a_int, b_int);
            }
        }
        // Fall back to numeric
        self.integrate(f, a, b)
    }

    /// Find a root of f(x) = 0 using Newton-Raphson with a symbolic derivative,
    /// multi-path verified against Brent's method.
    ///
    /// When the derivative is computed symbolically (exact), Newton-Raphson
    /// achieves quadratic convergence without finite-difference error.
    /// Cross-validated with Brent for multi-path verification.
    pub fn find_root_with_derivative<F, G>(
        &mut self,
        f: &F,
        df: &G,
        x0: f64,
        a: f64,
        b: f64,
    ) -> MathResponse
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let tol = 1e-10;

        // Primary: Newton-Raphson with exact symbolic derivative
        let newton_result = RootFindingEngine::newton_raphson(f, df, x0, tol);

        // Cross-validate: Brent's method (bracket-based, no derivative needed)
        let brent_result = RootFindingEngine::brent(f, a, b, tol);

        // Multi-path: Newton (exact derivative) vs Brent (bracket)
        let agreement = (newton_result.root - brent_result.root).abs();
        let multipath_verified =
            newton_result.converged && brent_result.converged && agreement < 1e-6;

        // Newton with symbolic derivative gets Phi boost for exact derivative
        let phi = if multipath_verified {
            (newton_result.phi + brent_result.phi) / 2.0 * MATH_SYMBOLIC_EXACT_PHI_BOOST
        } else if newton_result.converged {
            newton_result.phi * MATH_MULTIPATH_PHI_BOOST
        } else {
            brent_result.phi
        };
        self.record_solve(MathProblemType::RootFinding, phi);
        self.telemetry.verification_rate = {
            let total = self.telemetry.problems_solved as f64;
            let prev = self.telemetry.verification_rate * (total - 1.0).max(0.0);
            (prev + if multipath_verified { 1.0 } else { 0.0 }) / total
        };

        // Use Newton as primary (quadratic convergence), fall back to Brent
        let primary = if newton_result.converged {
            &newton_result
        } else {
            &brent_result
        };

        let response = MathResponse {
            answer: format!(
                "Root: {:.10} (residual: {:.2e}, method: {})",
                primary.root,
                primary.residual,
                if newton_result.converged {
                    "Newton-Raphson (symbolic derivative)"
                } else {
                    "Brent (fallback)"
                }
            ),
            numerical_result: Some(primary.root),
            vector_result: None,
            encoding: primary.encoding.clone(),
            phi,
            confidence: if multipath_verified {
                MATH_SYMBOLIC_EXACT_CONFIDENCE
            } else if primary.converged {
                MATH_ROOT_FINDING_CONVERGED_CONFIDENCE
            } else {
                MATH_ROOT_FINDING_FAILED_CONFIDENCE
            },
            multipath_verified,
            problem_type: MathProblemType::RootFinding,
            epistemic_caveat: if !primary.converged {
                Some("Root finding did not converge within tolerance".into())
            } else if !multipath_verified && newton_result.converged && brent_result.converged {
                Some(format!("Newton vs Brent disagreement: {:.2e}", agreement))
            } else {
                None
            },
            error_bound: Some(primary.residual.abs()),
        };
        self.store_episode(&response, "symbolic_root_finding");
        response
    }

    /// Minimize an objective function f(x) using Nelder-Mead
    pub fn optimize<F: Fn(&[f64]) -> f64>(&mut self, f: &F, initial: &[f64]) -> MathResponse {
        let result = OptimizationEngine::nelder_mead(f, initial, 1.0, 1e-8);
        let phi = result.phi;
        self.record_solve(MathProblemType::Optimization, phi);

        let answer = format!(
            "Optimum: [{}] → f = {:.8}",
            result
                .x
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(", "),
            result.fx
        );

        let response = MathResponse {
            answer,
            numerical_result: Some(result.fx),
            vector_result: Some(result.x),
            encoding: result.encoding,
            phi,
            confidence: if result.converged {
                MATH_OPTIMIZATION_CONVERGED_CONFIDENCE
            } else {
                MATH_OPTIMIZATION_FAILED_CONFIDENCE
            },
            multipath_verified: false,
            problem_type: MathProblemType::Optimization,
            epistemic_caveat: if !result.converged {
                Some("Optimization did not converge; result may be a local minimum".into())
            } else {
                None
            },
            error_bound: None,
        };
        self.store_episode(&response, "optimization");
        response
    }

    /// Compute FFT of a real signal
    pub fn compute_fft(&mut self, signal: &[f64]) -> MathResponse {
        let result = FftEngine::fft(signal);
        let spectrum = result.power_spectrum();
        let phi = result.phi;
        self.record_solve(MathProblemType::SignalAnalysis, phi);

        let response = MathResponse {
            answer: format!(
                "FFT: {} points, dominant freq bin: {}",
                signal.len(),
                spectrum
                    .iter()
                    .enumerate()
                    .skip(1)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            ),
            numerical_result: Some(spectrum.iter().sum::<f64>()),
            vector_result: Some(spectrum),
            encoding: result.encoding,
            phi,
            confidence: 0.99,
            multipath_verified: false,
            problem_type: MathProblemType::SignalAnalysis,
            epistemic_caveat: if signal.len() & (signal.len() - 1) != 0 {
                Some("Input length not a power of 2; zero-padded internally".into())
            } else {
                None
            },
            error_bound: None,
        };
        self.store_episode(&response, "fft");
        response
    }

    /// Solve a constraint satisfaction problem
    pub fn solve_csp(&mut self, csp: &CSP) -> MathResponse {
        let result = CSPSolver::solve(csp);
        let phi = result.phi;
        self.record_solve(MathProblemType::ConstraintSatisfaction, phi);

        let answer = if result.solved {
            let assignment: Vec<String> = result
                .solution
                .iter()
                .flat_map(|s| s.iter())
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("SATISFIABLE: {{{}}}", assignment.join(", "))
        } else {
            "UNSATISFIABLE".to_string()
        };

        let response = MathResponse {
            answer,
            numerical_result: Some(if result.solved { 1.0 } else { 0.0 }),
            vector_result: None,
            encoding: result.encoding,
            phi,
            confidence: 1.0,
            multipath_verified: false,
            problem_type: MathProblemType::ConstraintSatisfaction,
            epistemic_caveat: if result.backtracks > 1000 {
                Some(format!("Heavy search: {} backtracks", result.backtracks))
            } else {
                None
            },
            error_bound: None,
        };
        self.store_episode(&response, "csp");
        response
    }

    /// Compute convex hull of 2D points
    pub fn convex_hull(&mut self, points: &[(f64, f64)]) -> MathResponse {
        let pts: Vec<Point2D> = points.iter().map(|&(x, y)| Point2D { x, y }).collect();
        let hull = GeometryEngine::convex_hull(&pts);

        let phi = 0.3 + 0.01 * hull.len() as f64;
        self.record_solve(MathProblemType::Geometry, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!("HULL_{}", hull.len())));

        let response = MathResponse {
            answer: format!(
                "Convex hull: {} vertices from {} input points",
                hull.len(),
                points.len()
            ),
            numerical_result: Some(hull.len() as f64),
            vector_result: Some(hull.iter().flat_map(|p| vec![p.x, p.y]).collect()),
            encoding,
            phi,
            confidence: 0.99,
            multipath_verified: false,
            problem_type: MathProblemType::Geometry,
            epistemic_caveat: None,
            error_bound: None,
        };
        self.store_episode(&response, "convex_hull");
        response
    }

    /// Find shortest path in a weighted graph using Dijkstra's algorithm
    pub fn shortest_path(
        &mut self,
        n: usize,
        edges: &[(usize, usize, f64)],
        source: usize,
    ) -> MathResponse {
        let mut graph = Graph::new(n, false);
        for &(u, v, w) in edges {
            graph.add_weighted_edge(u, v, w);
        }
        let (distances, _predecessors) = graph.dijkstra(source);

        let phi = 0.35 + 0.005 * n as f64;
        self.record_solve(MathProblemType::GraphTheory, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!("DIJKSTRA_{}", n)));

        let response = MathResponse {
            answer: format!(
                "Dijkstra from node {}: reachable {} of {} nodes",
                source,
                distances.iter().filter(|d| d.is_finite()).count(),
                n,
            ),
            numerical_result: None,
            vector_result: Some(distances),
            encoding,
            phi,
            confidence: 0.99,
            multipath_verified: false,
            problem_type: MathProblemType::GraphTheory,
            epistemic_caveat: None,
            error_bound: None,
        };
        self.store_episode(&response, "shortest_path");
        response
    }

    /// Solve an initial value problem (ODE system)
    pub fn solve_ode(
        &mut self,
        f: fn(f64, &[f64]) -> Vec<f64>,
        t_span: (f64, f64),
        y0: &[f64],
        n_steps: usize,
    ) -> MathResponse {
        use symthaea_core::hdc::differential_equations::ODESystem;
        let system = ODESystem { f, dim: y0.len() };
        let result =
            DifferentialEquationsEngine::solve_ivp(&system, t_span.0, t_span.1, y0, n_steps);
        let final_state = result.final_state().to_vec();

        let phi = 0.4 + 0.01 * result.steps as f64;
        self.record_solve(MathProblemType::DifferentialEquation, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!(
            "ODE_{}_{}",
            y0.len(),
            result.steps
        )));

        let response = MathResponse {
            answer: format!(
                "ODE solved: {} steps, final state: [{}]",
                result.steps,
                final_state
                    .iter()
                    .map(|v| format!("{:.6}", v))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            numerical_result: Some(result.t_end()),
            vector_result: Some(final_state),
            encoding,
            phi,
            confidence: 0.9,
            multipath_verified: false,
            problem_type: MathProblemType::DifferentialEquation,
            epistemic_caveat: Some("Numerical ODE solution; accuracy depends on step size".into()),
            error_bound: Some((t_span.1 - t_span.0) / n_steps as f64),
        };
        self.store_episode(&response, "ode_solve");
        response
    }

    /// Compute chemistry: molar mass, Hess's law, Gibbs, Arrhenius.
    ///
    /// Extracts a formula or compound name from the query text and computes
    /// molar mass as a baseline. More specific queries (enthalpy, Gibbs, kinetics)
    /// are dispatched to the appropriate chemistry engine function.
    pub fn compute_chemistry(&mut self, text: &str) -> MathResponse {
        use symthaea_core::hdc::chemistry::{
            all_elements, gibbs_free_energy, hess_law, molar_mass, thermochemical_database,
        };

        let lower = text.to_lowercase();
        let elements = all_elements();
        let db = thermochemical_database();

        // Try to extract a chemical formula from the text (e.g., "H2O", "NaCl", "CH4")
        let formula = extract_formula(text);

        let (answer, numerical, phi) = if lower.contains("gibbs") || lower.contains("spontan") {
            // Gibbs free energy query — needs ΔH and ΔS from thermochemical DB
            if let Some(ref f) = formula {
                let phase_formula = format!("{}(g)", f); // try gas phase
                let dh = db
                    .iter()
                    .find(|d| d.formula == phase_formula)
                    .map(|d| d.delta_hf_kj_mol);
                let ds = db
                    .iter()
                    .find(|d| d.formula == phase_formula)
                    .map(|d| d.delta_sf_j_mol_k);
                if let (Some(dh), Some(ds)) = (dh, ds) {
                    let dg = gibbs_free_energy(dh, 298.15, ds);
                    (
                        format!("ΔG°({}) = {:.2} kJ/mol at 298.15 K", f, dg),
                        Some(dg),
                        0.7,
                    )
                } else {
                    (format!("No thermochemical data for {}", f), None, 0.2)
                }
            } else {
                ("No chemical formula found in query".into(), None, 0.1)
            }
        } else if lower.contains("molar mass") || lower.contains("molecular weight") {
            if let Some(ref f) = formula {
                match molar_mass(f, &elements) {
                    Ok(mm) => (format!("M({}) = {:.3} g/mol", f, mm), Some(mm), 0.8),
                    Err(e) => (format!("Error: {}", e), None, 0.1),
                }
            } else {
                ("No chemical formula found in query".into(), None, 0.1)
            }
        } else if lower.contains("enthalpy")
            || lower.contains("hess")
            || lower.contains("combustion")
        {
            // Default: try to find ΔH°f for the formula
            if let Some(ref f) = formula {
                let phase_formula = format!("{}(g)", f);
                if let Some(entry) = db
                    .iter()
                    .find(|d| d.formula == phase_formula || d.formula == *f)
                {
                    (
                        format!(
                            "ΔH°f({}) = {:.3} kJ/mol",
                            entry.formula, entry.delta_hf_kj_mol
                        ),
                        Some(entry.delta_hf_kj_mol),
                        0.7,
                    )
                } else {
                    (format!("No enthalpy data for {}", f), None, 0.2)
                }
            } else {
                ("No chemical formula found in query".into(), None, 0.1)
            }
        } else {
            // Default: compute molar mass if formula found
            if let Some(ref f) = formula {
                match molar_mass(f, &elements) {
                    Ok(mm) => (format!("M({}) = {:.3} g/mol", f, mm), Some(mm), 0.6),
                    Err(e) => (format!("Chemistry query: {}", e), None, 0.1),
                }
            } else {
                (
                    "Chemistry query received but no formula extracted".into(),
                    None,
                    0.1,
                )
            }
        };

        self.record_solve(MathProblemType::Chemistry, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!(
            "CHEM_{}",
            formula.as_deref().unwrap_or("unknown")
        )));

        let response = MathResponse {
            answer,
            numerical_result: numerical,
            vector_result: None,
            encoding,
            phi,
            confidence: if numerical.is_some() { 0.9 } else { 0.3 },
            multipath_verified: false,
            problem_type: MathProblemType::Chemistry,
            epistemic_caveat: Some("Values from NIST-JANAF thermochemical tables".into()),
            error_bound: None,
        };
        self.store_episode(&response, "chemistry");
        response
    }

    /// Attempt to construct a proof for a mathematical statement.
    ///
    /// Uses the TacticProver from symthaea-core to search for a proof via
    /// automated tactic application (intro, assumption, ring, omega, norm_num,
    /// simp, contradiction, induction). Returns a proof script if found.
    pub fn construct_proof(&mut self, conjecture: &str) -> MathResponse {
        use symthaea_core::hdc::tactics::{Expr, Goal, TacticProver};

        // Parse the conjecture into a Goal and attempt automated proof search.
        let goal = Goal::new(Expr::Var(conjecture.to_string()));
        let prover = TacticProver::new(50); // max 50 steps depth
        let result = prover.prove(&goal);

        let (answer, phi) = if let Some(script) = &result {
            (
                format!(
                    "Proof found ({} steps):\n{}",
                    script.len(),
                    script.join("\n")
                ),
                0.8 + 0.01 * script.len().min(20) as f64,
            )
        } else {
            ("No proof found within search depth".to_string(), 0.2)
        };
        let found = result.is_some();

        self.record_solve(MathProblemType::Proof, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!(
            "PROOF_{}",
            &conjecture[..conjecture.len().min(32)]
        )));

        let response = MathResponse {
            answer,
            numerical_result: None,
            vector_result: None,
            encoding,
            phi,
            confidence: if found { 0.95 } else { 0.1 },
            multipath_verified: found,
            problem_type: MathProblemType::Proof,
            epistemic_caveat: Some("Tactic-based proof search; depth-limited".into()),
            error_bound: None,
        };
        self.store_episode(&response, "proof");
        response
    }

    /// Get service telemetry
    pub fn telemetry(&self) -> &MathServiceTelemetry {
        &self.telemetry
    }

    // ─── Phase 7: Consciousness-Coupled Methods ─────────────────────────

    /// Search mathematical memory for similar past problems (Phase 7c).
    /// Returns up to `k` most similar episodes by HDC cosine similarity.
    pub fn recall_similar(&self, query: &BinaryHV, k: usize) -> Vec<&MathEpisode> {
        let mut scored: Vec<(usize, f64)> = self
            .memory
            .iter()
            .enumerate()
            .map(|(i, ep)| (i, query.similarity(&ep.problem_encoding) as f64))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .take(k)
            .filter(|(_, sim)| *sim > 0.1)
            .map(|(i, _)| &self.memory[*i])
            .collect()
    }

    /// Find root using Phi-guided proof search (Phase 7a).
    ///
    /// Tries Brent, bisection, and Newton-Raphson (when derivative is available),
    /// then selects the result with the highest Phi. Multi-path agreement between
    /// the top-2 results further boosts Phi by 20%.
    ///
    /// Science: Dehaene (2007) — mathematical elegance correlates with neural
    /// integration; Rota (1997) — beauty in mathematics as information compression.
    pub fn find_root_phi_guided<F: Fn(f64) -> f64>(
        &mut self,
        f: &F,
        a: f64,
        b: f64,
    ) -> MathResponse {
        let tol = 1e-10;
        let brent = RootFindingEngine::brent(f, a, b, tol);
        let bisect = RootFindingEngine::bisection(f, a, b, tol);
        // Numerical derivative for Newton-Raphson (central difference)
        let df = |x: f64| {
            let h = 1e-8;
            (f(x + h) - f(x - h)) / (2.0 * h)
        };
        let newton = RootFindingEngine::newton_raphson(f, &df, (a + b) / 2.0, tol);

        // Collect converged results — store scalars only, index into encodings array.
        // This avoids cloning BinaryHV (16,384-bit) for each solver; only the winner is cloned.
        let encodings = [brent.encoding, bisect.encoding, newton.encoding];
        let mut candidates: Vec<(f64, f64, f64, usize)> = Vec::new(); // (root, phi, residual, encoding_idx)
        if brent.converged {
            candidates.push((brent.root, brent.phi, brent.residual, 0));
        }
        if bisect.converged {
            candidates.push((bisect.root, bisect.phi, bisect.residual, 1));
        }
        if newton.converged {
            candidates.push((newton.root, newton.phi, newton.residual, 2));
        }

        // If no method converged, fall back to Brent
        if candidates.is_empty() {
            return self.find_root(f, a, b);
        }

        // Sort by Phi descending — prefer the most elegant solution
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = &candidates[0];

        // Multi-path: check if top-2 agree → boost Phi
        let multipath_verified =
            candidates.len() >= 2 && (candidates[0].0 - candidates[1].0).abs() < 1e-6;
        let phi = if multipath_verified {
            best.1 * MATH_MULTIPATH_PHI_BOOST
        } else {
            best.1
        };

        self.record_solve(MathProblemType::RootFinding, phi);

        let response = MathResponse {
            answer: format!(
                "Root: {:.10} (residual: {:.2e}, {} methods converged, Phi-guided)",
                best.0,
                best.2,
                candidates.len()
            ),
            numerical_result: Some(best.0),
            vector_result: None,
            encoding: encodings[best.3].clone(),
            phi,
            confidence: if multipath_verified {
                0.99
            } else if candidates.len() >= 2 {
                0.95
            } else {
                0.9
            },
            multipath_verified,
            problem_type: MathProblemType::RootFinding,
            epistemic_caveat: None,
            error_bound: Some(best.2.abs()),
        };
        self.store_episode(&response, "root_phi_guided");
        response
    }

    /// Rank multiple candidate solutions by Phi (Phase 7a: prefer elegant solutions).
    pub fn rank_by_phi(responses: &[MathResponse]) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = responses
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.phi))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Get mathematical memory contents
    pub fn memory(&self) -> &[MathEpisode] {
        &self.memory
    }

    /// Get memory utilization (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        self.memory.len() as f64 / self.memory_capacity as f64
    }

    /// Solve a root-finding problem with Phi-ranked multi-method search (Phase 7a).
    ///
    /// Runs bisection, Brent, and Newton-Raphson in parallel, then ranks
    /// solutions by Phi. Higher Phi = fewer steps + more agreement.
    pub fn solve_with_phi_ranking<F: Fn(f64) -> f64>(
        &mut self,
        f: &F,
        a: f64,
        b: f64,
    ) -> Vec<PhiRankedSolution> {
        let tol = 1e-10;

        let brent = RootFindingEngine::brent(f, a, b, tol);
        let bisect = RootFindingEngine::bisection(f, a, b, tol);
        let df = |x: f64| {
            let h = 1e-8;
            (f(x + h) - f(x - h)) / (2.0 * h)
        };
        let newton = RootFindingEngine::newton_raphson(f, &df, (a + b) / 2.0, tol);

        struct RawCandidate {
            root: f64,
            iterations: usize,
            converged: bool,
            method: &'static str,
        }

        let mut raw = Vec::new();
        if brent.converged {
            raw.push(RawCandidate {
                root: brent.root,
                iterations: brent.iterations,
                converged: true,
                method: "brent",
            });
        }
        if bisect.converged {
            raw.push(RawCandidate {
                root: bisect.root,
                iterations: bisect.iterations,
                converged: true,
                method: "bisection",
            });
        }
        if newton.converged {
            raw.push(RawCandidate {
                root: newton.root,
                iterations: newton.iterations,
                converged: true,
                method: "newton_raphson",
            });
        }

        if raw.is_empty() {
            return Vec::new();
        }

        let mut solutions: Vec<PhiRankedSolution> = raw
            .iter()
            .map(|candidate| {
                let agreeing = raw
                    .iter()
                    .filter(|other| (other.root - candidate.root).abs() < 1e-6)
                    .count();
                let agreement_bonus = if agreeing > 1 {
                    1.0 + 0.2 * (agreeing - 1) as f64
                } else {
                    1.0
                };
                let proof_steps = candidate.iterations.max(1);
                let verification_paths = agreeing;
                let phi = agreement_bonus * (1.0 / proof_steps as f64) * verification_paths as f64;

                PhiRankedSolution {
                    result: candidate.root,
                    method: candidate.method.to_string(),
                    phi,
                    proof_steps,
                    verification_paths,
                }
            })
            .collect();

        solutions.sort_by(|a, b| {
            b.phi
                .partial_cmp(&a.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(best) = solutions.first() {
            self.record_solve(MathProblemType::RootFinding, best.phi);
        }

        solutions
    }

    /// Get a structured MathMemory view (Phase 7c).
    pub fn math_memory(&self) -> MathMemory {
        let mut mm = MathMemory::new(self.memory_capacity);
        for ep in &self.memory {
            mm.episodes.push(ep.clone());
        }
        mm
    }

    /// Suggest a method for a new problem based on past experience (Phase 7c).
    pub fn suggest_method_for(&self, problem_text: &str) -> Option<String> {
        let query = BinaryHV::random(seed_from_name(&format!("QUERY_{}", problem_text)));
        let mm = self.math_memory();
        mm.suggest_method(&query)
    }

    /// Compute the determinant of an n×n matrix (row-major layout).
    ///
    /// Uses LU decomposition with partial pivoting. Returns a `MathResponse`
    /// with `numerical_result` set to the determinant value.
    ///
    /// Science: Gaussian elimination with pivoting (Golub & Van Loan 1996).
    pub fn matrix_determinant(&mut self, data: &[f64], n: usize) -> MathResponse {
        assert_eq!(
            data.len(),
            n * n,
            "matrix_determinant: data.len() must equal n*n"
        );
        let det = lu_determinant(data, n);
        let response = MathResponse {
            answer: format!("det = {det:.6}"),
            numerical_result: Some(det),
            vector_result: None,
            encoding: BinaryHV::random(42),
            phi: MATH_DEFAULT_TELEMETRY_PHI,
            confidence: MATH_DEFAULT_TELEMETRY_CONFIDENCE,
            multipath_verified: false,
            problem_type: MathProblemType::MatrixAnalysis,
            epistemic_caveat: None,
            error_bound: None,
        };
        self.store_episode(&response, &format!("matrix_determinant(n={n})"));
        self.telemetry.problems_solved += 1;
        response
    }

    // ─── Internal ────────────────────────────────────────────────────────

    /// Store a solved problem as an HDC-encoded episode for future analogical retrieval
    fn store_episode(&mut self, response: &MathResponse, description: &str) {
        // Track confidence for telemetry
        self.record_confidence(response.confidence);

        if self.memory.len() >= self.memory_capacity {
            // Evict lowest-Phi episode
            if let Some(min_idx) = self
                .memory
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.phi
                        .partial_cmp(&b.1.phi)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if response.phi > self.memory[min_idx].phi {
                    self.memory.swap_remove(min_idx);
                } else {
                    return; // New episode isn't worth storing
                }
            }
        }

        let problem_encoding = BinaryHV::random(seed_from_name(&format!(
            "PROB_{}_{}",
            description, self.telemetry.problems_solved
        )));

        self.memory.push(MathEpisode {
            problem_encoding,
            solution_encoding: response.encoding.clone(),
            problem_type: response.problem_type,
            phi: response.phi,
            description: description.to_string(),
            method: format!("{:?}", response.problem_type),
            timestamp: self.telemetry.problems_solved as u64,
        });
    }

    fn record_solve(&mut self, problem_type: MathProblemType, phi: f64) {
        self.telemetry.problems_solved += 1;
        self.telemetry.total_phi += phi;
        self.telemetry.average_phi =
            self.telemetry.total_phi / self.telemetry.problems_solved as f64;

        let type_name = problem_type.as_str();
        *self.telemetry.by_type.entry(type_name.into()).or_insert(0) += 1;
    }

    /// Track confidence from a solved response for telemetry.
    fn record_confidence(&mut self, confidence: f64) {
        self.telemetry.total_confidence += confidence;
        self.telemetry.average_confidence =
            self.telemetry.total_confidence / self.telemetry.problems_solved.max(1) as f64;
    }
}

/// Extract a chemical formula from free text (e.g., "H2O", "NaCl", "CH4").
/// Looks for patterns of uppercase letter followed by optional lowercase + digits.
fn extract_formula(text: &str) -> Option<String> {
    // Match patterns like H2O, NaCl, CH3COOH, Ca(OH)2
    let mut best: Option<String> = None;
    for word in text.split_whitespace() {
        // Strip punctuation from word boundaries
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '(' && c != ')');
        if clean.len() < 2 {
            continue;
        }
        // Must start with uppercase and contain at least one lowercase or digit
        let first = clean.chars().next().unwrap_or(' ');
        if !first.is_ascii_uppercase() {
            continue;
        }
        // Check if it looks like a formula: uppercase + (lowercase|digit|parentheses)
        let has_chem_pattern = clean
            .chars()
            .skip(1)
            .any(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
        let has_non_alpha = clean.chars().any(|c| c.is_ascii_digit());
        let all_valid = clean
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '(' || c == ')');
        if all_valid && (has_chem_pattern || has_non_alpha) {
            // Prefer longer formulas (more specific)
            if best.as_ref().map_or(true, |b| clean.len() > b.len()) {
                best = Some(clean.to_string());
            }
        }
    }
    best
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_problem() {
        assert_eq!(
            MathService::classify_problem("solve the linear system Ax=b"),
            MathProblemType::LinearSystem
        );
        assert_eq!(
            MathService::classify_problem("find the roots of x^2 - 4"),
            MathProblemType::RootFinding
        );
        assert_eq!(
            MathService::classify_problem("integrate sin(x) from 0 to pi"),
            MathProblemType::Integration
        );
        assert_eq!(
            MathService::classify_problem("compute the eigenvalues"),
            MathProblemType::MatrixAnalysis
        );
        assert_eq!(
            MathService::classify_problem("what is the mean and variance"),
            MathProblemType::Statistics
        );
        assert_eq!(
            MathService::classify_problem("optimize the function"),
            MathProblemType::Optimization
        );
        assert_eq!(
            MathService::classify_problem("compute the FFT"),
            MathProblemType::SignalAnalysis
        );
        assert_eq!(
            MathService::classify_problem("prove this proposition is a tautology"),
            MathProblemType::Logic
        );
        assert_eq!(
            MathService::classify_problem("solve the N-Queens constraint problem"),
            MathProblemType::ConstraintSatisfaction
        );
    }

    #[test]
    fn test_solve_linear_system() {
        let mut service = MathService::new();
        // x + 2y = 5, 3x + 4y = 11 → x=1, y=2
        let response = service.solve_linear_system(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 11.0]);
        let x = response.vector_result.unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
        assert!(response.multipath_verified);
        assert!(response.phi > 0.0);
    }

    #[test]
    fn test_compute_statistics() {
        let mut service = MathService::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let response = service.compute_statistics(&data);
        assert!((response.numerical_result.unwrap() - 3.0).abs() < 1e-6);
        assert!(response.answer.contains("mean=3.0000"));
    }

    #[test]
    fn test_linear_regression() {
        let mut service = MathService::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let response = service.linear_regression(&x, &y);
        let coeffs = response.vector_result.unwrap();
        assert!((coeffs[0] - 1.0).abs() < 1e-6, "intercept = {}", coeffs[0]);
        assert!((coeffs[1] - 2.0).abs() < 1e-6, "slope = {}", coeffs[1]);
        assert!((response.numerical_result.unwrap() - 1.0).abs() < 1e-6); // R²
    }

    #[test]
    fn test_check_tautology() {
        let mut service = MathService::new();
        // P ∨ ¬P
        let p = Proposition::atom("P");
        let formula = p.clone().or(p.not());
        let response = service.check_tautology(&formula);
        assert!(response.answer.contains("TAUTOLOGY"));
    }

    #[test]
    fn test_solve_sat() {
        let mut service = MathService::new();
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.and(q);
        let response = service.solve_sat(&formula);
        assert!(response.answer.contains("SATISFIABLE"));
    }

    #[test]
    fn test_telemetry() {
        let mut service = MathService::new();
        let data = vec![1.0, 2.0, 3.0];
        service.compute_statistics(&data);
        service.compute_statistics(&data);
        assert_eq!(service.telemetry().problems_solved, 2);
        assert!(service.telemetry().average_phi > 0.0);
    }

    #[test]
    fn test_epistemic_caveat_small_sample() {
        let mut service = MathService::new();
        let data = vec![1.0, 2.0, 3.0]; // n=3, should trigger small sample caveat
        let response = service.compute_statistics(&data);
        assert!(response.epistemic_caveat.is_some());
        assert!(response.error_bound.is_some());
    }

    #[test]
    fn test_mathematical_memory() {
        let mut service = MathService::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        service.compute_statistics(&data);
        assert_eq!(service.memory().len(), 1);

        // Store more and recall
        service.compute_statistics(&[10.0, 20.0, 30.0]);
        assert_eq!(service.memory().len(), 2);

        let query = BinaryHV::random(seed_from_name("PROB_statistics_1"));
        let similar = service.recall_similar(&query, 3);
        assert!(!similar.is_empty());
    }

    #[test]
    fn test_rank_by_phi() {
        let r1 = MathResponse {
            answer: "a".into(),
            numerical_result: None,
            vector_result: None,
            encoding: BinaryHV::random(1),
            phi: 0.5,
            confidence: 0.9,
            multipath_verified: false,
            problem_type: MathProblemType::Arithmetic,
            epistemic_caveat: None,
            error_bound: None,
        };
        let r2 = MathResponse {
            phi: 0.9,
            ..r1.clone()
        };
        let r3 = MathResponse {
            phi: 0.1,
            ..r1.clone()
        };
        let ranked = MathService::rank_by_phi(&[r1, r2, r3]);
        assert_eq!(ranked[0].0, 1); // r2 has highest phi
        assert_eq!(ranked[2].0, 2); // r3 has lowest phi
    }

    // ── Integration tests: full pipeline ──────────────────────────────

    #[test]
    fn test_classify_new_problem_types() {
        assert_eq!(
            MathService::classify_problem("compute the convex hull of these points"),
            MathProblemType::Geometry
        );
        assert_eq!(
            MathService::classify_problem("find the shortest path in this graph"),
            MathProblemType::GraphTheory
        );
        assert_eq!(
            MathService::classify_problem("solve this differential equation with initial value"),
            MathProblemType::DifferentialEquation
        );
    }

    #[test]
    fn test_full_pipeline_classify_and_solve() {
        let mut service = MathService::new();

        // 1. Classify: "solve the linear system matrix A with b"
        let problem_type =
            MathService::classify_problem("solve the linear system Ax=b with matrix");
        assert_eq!(problem_type, MathProblemType::LinearSystem);

        // 2. Route to solver: x + y = 3, x - y = 1 → x=2, y=1
        let response = service.solve_linear_system(&[1.0, 1.0, 1.0, -1.0], 2, 2, &[3.0, 1.0]);

        // 3. Verify response has all consciousness metadata
        assert!(response.phi > 0.0, "Phi should be positive");
        assert!(response.confidence > 0.0, "Confidence should be positive");
        assert_eq!(response.problem_type, MathProblemType::LinearSystem);

        let x = response.vector_result.unwrap();
        assert!(
            (x[0] - 2.0).abs() < 1e-6,
            "x[0] should be 2.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 1.0).abs() < 1e-6,
            "x[1] should be 1.0, got {}",
            x[1]
        );

        // 4. Verify telemetry was updated
        assert_eq!(service.telemetry().problems_solved, 1);
        assert!(service.telemetry().average_phi > 0.0);

        // 5. Verify episode stored in memory
        assert_eq!(service.memory().len(), 1);
        assert_eq!(
            service.memory()[0].problem_type,
            MathProblemType::LinearSystem
        );
    }

    #[test]
    fn test_pipeline_root_finding() {
        let mut service = MathService::new();

        // Find root of x² - 4 = 0 in [0, 3] → should find x=2
        let response = service.find_root(&|x: f64| x * x - 4.0, 0.0, 3.0);
        assert!(response.confidence > 0.5);
        let root = response.numerical_result.unwrap();
        assert!(
            (root - 2.0).abs() < 1e-8,
            "Root should be 2.0, got {}",
            root
        );
    }

    #[test]
    fn test_pipeline_integration() {
        let mut service = MathService::new();

        // ∫₀^π sin(x) dx = 2
        let response = service.integrate(&|x: f64| x.sin(), 0.0, std::f64::consts::PI);
        let val = response.numerical_result.unwrap();
        assert!(
            (val - 2.0).abs() < 1e-6,
            "Integral should be 2.0, got {}",
            val
        );
    }

    #[test]
    fn test_pipeline_optimization() {
        let mut service = MathService::new();

        // Minimize (x-3)² + (y-4)² → optimum at [3, 4]
        let response = service.optimize(
            &|x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2),
            &[0.0, 0.0],
        );
        let x = response.vector_result.unwrap();
        assert!(
            (x[0] - 3.0).abs() < 0.1,
            "x should be near 3.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 4.0).abs() < 0.1,
            "y should be near 4.0, got {}",
            x[1]
        );
    }

    #[test]
    fn test_pipeline_fft() {
        let mut service = MathService::new();
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin())
            .collect();
        let response = service.compute_fft(&signal);
        assert!(response.vector_result.is_some());
        assert!(response.phi > 0.0);
    }

    #[test]
    fn test_pipeline_convex_hull() {
        let mut service = MathService::new();
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, 0.3)];
        let response = service.convex_hull(&points);
        // 4 points, hull should have 3 vertices (interior point excluded)
        let n_hull = response.numerical_result.unwrap() as usize;
        assert!(
            n_hull >= 3 && n_hull <= 4,
            "Hull should have 3-4 vertices, got {}",
            n_hull
        );
    }

    #[test]
    fn test_pipeline_shortest_path() {
        let mut service = MathService::new();
        // Triangle: 0→1 (weight 1), 1→2 (weight 2), 0→2 (weight 10)
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)];
        let response = service.shortest_path(3, &edges, 0);
        let dists = response.vector_result.unwrap();
        assert!((dists[0] - 0.0).abs() < 1e-10);
        assert!((dists[1] - 1.0).abs() < 1e-10);
        assert!(
            (dists[2] - 3.0).abs() < 1e-10,
            "Path 0→1→2 costs 3, got {}",
            dists[2]
        );
    }

    #[test]
    fn test_pipeline_ode() {
        let mut service = MathService::new();
        // dy/dt = -y, y(0) = 1 → y(1) = e^{-1} ≈ 0.368
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let response = service.solve_ode(decay, (0.0, 1.0), &[1.0], 1000);
        let y_final = response.vector_result.unwrap();
        let expected = (-1.0f64).exp();
        assert!(
            (y_final[0] - expected).abs() < 0.01,
            "y(1) should be ~{:.4}, got {:.4}",
            expected,
            y_final[0]
        );
    }

    #[test]
    fn test_multi_problem_telemetry() {
        let mut service = MathService::new();

        // Solve multiple problem types
        service.compute_statistics(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        service.solve_linear_system(&[1.0, 0.0, 0.0, 1.0], 2, 2, &[3.0, 4.0]);
        service.find_root(&|x: f64| x - 1.0, 0.0, 2.0);

        let telem = service.telemetry();
        assert_eq!(telem.problems_solved, 3);
        assert!(telem.by_type.contains_key("Statistics"));
        assert!(telem.by_type.contains_key("LinearSystem"));
        assert!(telem.by_type.contains_key("RootFinding"));
        assert!(telem.average_phi > 0.0);

        // Memory should have 3 episodes
        assert_eq!(service.memory().len(), 3);
    }

    #[test]
    fn test_phi_guided_root_finding() {
        let mut service = MathService::new();

        // x² - 4 = 0 in [0, 3] → root at x=2
        let response = service.find_root_phi_guided(&|x: f64| x * x - 4.0, 0.0, 3.0);
        let root = response.numerical_result.unwrap();
        assert!(
            (root - 2.0).abs() < 1e-8,
            "Root should be 2.0, got {}",
            root
        );
        assert!(response.answer.contains("Phi-guided"));
        assert!(response.confidence >= 0.9);
        // With 3 methods converging to the same root, should be multi-path verified
        assert!(response.multipath_verified, "3 methods → should verify");
        // Phi should be boosted by multi-path agreement
        assert!(response.phi > 0.0);
    }

    #[test]
    fn test_phi_guided_selects_highest_phi() {
        let mut service = MathService::new();

        // sin(x) = 0 in [2, 4] → root at π ≈ 3.14159
        let response = service.find_root_phi_guided(&|x: f64| x.sin(), 2.0, 4.0);
        let root = response.numerical_result.unwrap();
        assert!(
            (root - std::f64::consts::PI).abs() < 1e-6,
            "Root should be π, got {}",
            root
        );
        assert!(response.answer.contains("methods converged"));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 7a: Phi-Ranked Proof Search Tests
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_phi_ranking_prefers_elegant() {
        let mut service = MathService::new();
        let solutions = service.solve_with_phi_ranking(&|x: f64| x * x - 4.0, 0.0, 3.0);
        assert!(!solutions.is_empty());
        for pair in solutions.windows(2) {
            assert!(pair[0].phi >= pair[1].phi);
        }
        let best = &solutions[0];
        assert!((best.result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_method_agreement_bonus() {
        let mut service = MathService::new();
        let solutions = service.solve_with_phi_ranking(&|x: f64| x - 1.0, 0.0, 2.0);
        for sol in &solutions {
            assert!((sol.result - 1.0).abs() < 1e-6);
            assert!(sol.verification_paths > 1);
        }
    }

    #[test]
    fn test_single_method_fallback() {
        let mut service = MathService::new();
        let solutions =
            service.solve_with_phi_ranking(&|x: f64| (100.0 * (x - 1.0)).tanh(), 0.5, 1.5);
        assert!(!solutions.is_empty());
        let best = &solutions[0];
        assert!((best.result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_phi_ranking_empty_problem() {
        let mut service = MathService::new();
        let solutions = service.solve_with_phi_ranking(&|x: f64| x * x + 1.0, 2.0, 3.0);
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_proof_step_penalty() {
        let mut service = MathService::new();
        let solutions = service.solve_with_phi_ranking(&|x: f64| x - 5.0, 0.0, 10.0);
        assert!(!solutions.is_empty());
        for sol in &solutions {
            assert!(sol.proof_steps >= 1);
            assert!(sol.phi > 0.0);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 7c: Mathematical Memory Tests
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_math_memory_recall() {
        let mut mm = MathMemory::new(10);
        let encoding = BinaryHV::random(seed_from_name("test_problem_1"));
        mm.remember(MathEpisode {
            problem_encoding: encoding.clone(),
            solution_encoding: BinaryHV::random(seed_from_name("test_solution_1")),
            problem_type: MathProblemType::RootFinding,
            phi: 0.5,
            description: "root finding test".to_string(),
            method: "brent".to_string(),
            timestamp: 1,
        });
        let recalled = mm.recall(&encoding);
        assert!(recalled.is_some());
        assert_eq!(recalled.unwrap().description, "root finding test");
    }

    #[test]
    fn test_math_memory_capacity_limit() {
        let mut mm = MathMemory::new(3);
        for i in 0..4 {
            mm.remember(MathEpisode {
                problem_encoding: BinaryHV::random(seed_from_name(&format!("cap_prob_{}", i))),
                solution_encoding: BinaryHV::random(seed_from_name(&format!("cap_sol_{}", i))),
                problem_type: MathProblemType::Arithmetic,
                phi: i as f64 * 0.1 + 0.1,
                description: format!("episode_{}", i),
                method: "test".to_string(),
                timestamp: i as u64,
            });
        }
        assert!(mm.len() <= 3);
    }

    #[test]
    fn test_method_suggestion() {
        let mut mm = MathMemory::new(10);
        let encoding = BinaryHV::random(seed_from_name("suggest_prob"));
        mm.remember(MathEpisode {
            problem_encoding: encoding.clone(),
            solution_encoding: BinaryHV::random(42),
            problem_type: MathProblemType::RootFinding,
            phi: 0.8,
            description: "root test".to_string(),
            method: "newton_raphson".to_string(),
            timestamp: 1,
        });
        let suggestion = mm.suggest_method(&encoding);
        assert_eq!(suggestion, Some("newton_raphson".to_string()));
    }

    #[test]
    fn test_empty_memory_returns_none() {
        let mm = MathMemory::new(10);
        let query = BinaryHV::random(42);
        assert!(mm.recall(&query).is_none());
        assert!(mm.suggest_method(&query).is_none());
        assert!(mm.is_empty());
        assert_eq!(mm.len(), 0);
    }
}
