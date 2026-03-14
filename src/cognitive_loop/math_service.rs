#![allow(dead_code)]

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
    /// General arithmetic
    Arithmetic,
    /// Unknown
    Unknown,
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
        if lower.contains("satisf")
            || lower.contains("tautolog")
            || lower.contains("prove")
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

        // Phase 7a: Phi-guided verification via residual check.
        // Compute ||Ax - b||₂ to independently verify the solution.
        // Science: Golub & Van Loan (1996) — backward error analysis.
        let residual_norm = {
            let mut residual = 0.0;
            for i in 0..rows {
                let mut ax_i = 0.0;
                for j in 0..cols {
                    ax_i += a_data[i * cols + j] * x.data[j];
                }
                residual += (ax_i - b_data[i]).powi(2);
            }
            residual.sqrt()
        };
        let residual_verified = residual_norm < 1e-8;

        // Multi-path: engine's internal verification + our residual check
        let multipath_verified = result.verified || residual_verified;
        let phi = if multipath_verified {
            result.phi * 1.2 // Boost for verified solutions
        } else {
            result.phi
        };

        let answer = format!(
            "Solution: [{}] (residual: {:.2e})",
            x.data
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(", "),
            residual_norm
        );

        self.record_solve(MathProblemType::LinearSystem, phi);

        let response = MathResponse {
            answer,
            numerical_result: None,
            vector_result: Some(x.data),
            encoding: result.encoding,
            phi,
            confidence: if multipath_verified { 0.99 } else { 0.5 },
            multipath_verified,
            problem_type: MathProblemType::LinearSystem,
            epistemic_caveat: if !multipath_verified {
                Some(format!(
                    "Solution not verified; residual norm: {:.2e}",
                    residual_norm
                ))
            } else {
                None
            },
            error_bound: Some(residual_norm),
        };

        self.store_episode(&response, "linear_system");
        response
    }

    /// Compute descriptive statistics
    pub fn compute_statistics(&mut self, data: &[f64]) -> MathResponse {
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

        let phi = 0.3;
        self.record_solve(MathProblemType::Statistics, phi);

        let encoding = BinaryHV::random(seed_from_name(&format!("STATS_{}", m.to_bits())));

        let response = MathResponse {
            answer,
            numerical_result: Some(m),
            vector_result: None,
            encoding,
            phi,
            confidence: 0.99,
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
            numerical_result: Some(tt.satisfying_count as f64 / tt.rows.len() as f64),
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
            (brent_result.phi + bisect_result.phi) / 2.0 * 1.2
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
                0.99
            } else if brent_result.converged {
                0.9
            } else {
                0.3
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
            (simpson.phi + gauss.phi) / 2.0 * 1.2
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
            confidence: if multipath_verified { 0.99 } else { 0.9 },
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
            confidence: if result.converged { 0.9 } else { 0.4 },
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

        // Collect converged results with their Phi
        let mut candidates: Vec<(f64, f64, f64, bool, symthaea_core::hdc::binary_hv::BinaryHV)> =
            Vec::new(); // (root, phi, residual, converged, encoding)
        if brent.converged {
            candidates.push((
                brent.root,
                brent.phi,
                brent.residual,
                true,
                brent.encoding.clone(),
            ));
        }
        if bisect.converged {
            candidates.push((
                bisect.root,
                bisect.phi,
                bisect.residual,
                true,
                bisect.encoding.clone(),
            ));
        }
        if newton.converged {
            candidates.push((
                newton.root,
                newton.phi,
                newton.residual,
                true,
                newton.encoding.clone(),
            ));
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
            best.1 * 1.2
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
            encoding: best.4.clone(),
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

    /// Transfer solution strategy from a recalled episode (Phase 7c).
    ///
    /// Given a classified problem type and a recalled episode, determine if the
    /// recalled episode suggests a better approach. Returns `Some(transferred_type)`
    /// when the recalled type differs and has high confidence from past success.
    ///
    /// Science: Gentner (1983) — structure-mapping theory of analogical reasoning;
    ///          Holyoak & Thagard (1995) — analogical constraint satisfaction.
    pub fn transfer_strategy(
        &self,
        classified: MathProblemType,
        query: &BinaryHV,
    ) -> Option<MathProblemType> {
        let recalled = self.recall_similar(query, 1);
        let ep = recalled.first()?;
        let sim = query.similarity(&ep.problem_encoding) as f64;

        // Only transfer if similarity is high and the recalled type differs
        if sim > 0.6 && ep.problem_type != classified && ep.phi > 0.3 {
            Some(ep.problem_type)
        } else {
            None
        }
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

    /// Compute the determinant of an n×n matrix (row-major layout).
    ///
    /// Uses LU decomposition with partial pivoting. Returns a `MathResponse`
    /// with `numerical_result` set to the determinant value.
    ///
    /// Science: Gaussian elimination with pivoting (Golub & Van Loan 1996).
    pub fn matrix_determinant(&mut self, data: &[f64], n: usize) -> MathResponse {
        assert_eq!(data.len(), n * n, "matrix_determinant: data.len() must equal n*n");
        let det = lu_determinant(data, n);
        let response = MathResponse {
            answer: format!("det = {det:.6}"),
            numerical_result: Some(det),
            vector_result: None,
            encoding: BinaryHV::random(42),
            phi: 0.4,
            confidence: 0.9,
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
        });
    }

    fn record_solve(&mut self, problem_type: MathProblemType, phi: f64) {
        self.telemetry.problems_solved += 1;
        self.telemetry.total_phi += phi;
        self.telemetry.average_phi =
            self.telemetry.total_phi / self.telemetry.problems_solved as f64;

        let type_name = format!("{:?}", problem_type);
        *self.telemetry.by_type.entry(type_name).or_insert(0) += 1;
    }
}

// ─── Expression Parsing ─────────────────────────────────────────────────────

/// Parse a simple mathematical expression string into a callable closure.
///
/// Supports basic forms: `x^N`, `N*x`, `sin(x)`, `cos(x)`, `exp(x)`, `x - C`.
/// Returns `None` for expressions that cannot be parsed.
///
/// Basis: lightweight symbolic dispatch for cognitive loop math routing.
pub fn parse_expression(input: &str) -> Option<Box<dyn Fn(f64) -> f64>> {
    let s = input.trim().to_lowercase();

    // x^N  (e.g. "x^2 - 4")
    if let Some(rest) = s.strip_prefix("x^") {
        if let Some((exp_str, tail)) = rest.split_once(|c: char| c == ' ' || c == '-' || c == '+')
        {
            if let Ok(exp) = exp_str.trim().parse::<f64>() {
                let tail = tail.trim();
                let offset: f64 = tail.parse().unwrap_or(0.0);
                // Determine sign: if we split on '-', offset is subtracted
                if rest.contains('-') {
                    return Some(Box::new(move |x: f64| x.powf(exp) - offset.abs()));
                } else {
                    return Some(Box::new(move |x: f64| x.powf(exp) + offset));
                }
            }
        }
        // Plain x^N
        if let Ok(exp) = rest.trim().parse::<f64>() {
            return Some(Box::new(move |x: f64| x.powf(exp)));
        }
    }

    // sin(x)
    if s.contains("sin") {
        return Some(Box::new(|x: f64| x.sin()));
    }
    // cos(x)
    if s.contains("cos") {
        return Some(Box::new(|x: f64| x.cos()));
    }
    // exp(x)
    if s.contains("exp") {
        return Some(Box::new(|x: f64| x.exp()));
    }

    // x * x or x*x
    if s == "x * x" || s == "x*x" {
        return Some(Box::new(|x: f64| x * x));
    }

    // Plain "x"
    if s == "x" {
        return Some(Box::new(|x: f64| x));
    }

    None
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

    #[test]
    fn test_linear_system_residual_verification() {
        let mut service = MathService::new();
        // Solve: [2, 1; 1, 3] x = [5, 10] → x = [1, 3]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let response = service.solve_linear_system(&a, 2, 2, &b);
        assert!(response.multipath_verified, "should be residual-verified");
        assert!(response.phi > 0.0);
        assert!(response.error_bound.is_some());
        let residual = response.error_bound.unwrap();
        assert!(residual < 1e-8, "residual should be tiny, got {}", residual);
        assert!(response.answer.contains("residual"));
    }

    #[test]
    fn test_strategy_transfer_from_memory() {
        let mut service = MathService::new();
        // Solve several root-finding problems to build memory
        for _ in 0..5 {
            service.find_root_phi_guided(&|x: f64| x * x - 4.0, 0.0, 3.0);
        }
        // Verify memory has entries
        assert!(!service.memory().is_empty());
        // Query with a random encoding — won't match strongly
        let query = BinaryHV::random(12345);
        let transfer = service.transfer_strategy(MathProblemType::Unknown, &query);
        // Can't guarantee a match with random query, but API shouldn't panic
        let _ = transfer;
    }

    #[test]
    #[cfg(feature = "scientific_method")]
    fn test_scientific_method_full_cycle() {
        use crate::scientific_method::ScientificMethodEngine;

        let mut engine = ScientificMethodEngine::new();
        let mut math = MathService::new();

        // Hypothesize about data
        let hid = engine.hypothesize("Data mean is approximately 10", 0.5);
        let data = vec![9.5, 10.2, 10.1, 9.8, 10.4];
        let result = engine.numerical_experiment(&mut math, hid, &data, 0.5);
        assert!(result.is_some());

        // Update beliefs
        engine.update_beliefs();

        let report = engine.generate_report();
        assert_eq!(report.total_experiments, 1);
        assert!(report.total_phi > 0.0);
    }
}
