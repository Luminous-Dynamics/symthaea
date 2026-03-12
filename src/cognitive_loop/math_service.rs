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
use symthaea_core::hdc::constraint_solver::{CSPSolver, CSP};
use symthaea_core::hdc::fft::FftEngine;
use symthaea_core::hdc::linear_algebra::{HdcMatrix, HdcVector, LinearAlgebraEngine};
use symthaea_core::hdc::logic_engine::{LogicEngine, Proposition};
use symthaea_core::hdc::optimization::OptimizationEngine;
use symthaea_core::hdc::primitive_system::seed_from_name;
use symthaea_core::hdc::quadrature::QuadratureEngine;
use symthaea_core::hdc::root_finding::RootFindingEngine;
use symthaea_core::hdc::statistics;

// ─── Types ───────────────────────────────────────────────────────────────────

/// Classification of a math problem type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
        if lower.contains("mean") || lower.contains("variance") || lower.contains("probabilit")
            || lower.contains("distribut") || lower.contains("bayesian") || lower.contains("hypothesis")
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
        if lower.contains("satisf") || lower.contains("tautolog") || lower.contains("prove")
            || lower.contains("logic") || lower.contains("proposition")
        {
            return MathProblemType::Logic;
        }
        if lower.contains("constraint") || lower.contains("n-queen") || lower.contains("sudoku")
            || lower.contains("color")
        {
            return MathProblemType::ConstraintSatisfaction;
        }
        if lower.contains("add") || lower.contains("multiply") || lower.contains("subtract")
            || lower.contains("divide") || lower.contains("factorial")
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
            confidence: if result.verified { 0.95 } else { 0.5 },
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
                Some(format!("Small sample (n={}); estimates may be unstable", data.len()))
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
            "REGR_{}_{}", result.slope.to_bits(), result.intercept.to_bits()
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
                Some(format!("Weak fit (R²={:.3}); linear model may be inappropriate", result.r_squared))
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
            format!("{} is a TAUTOLOGY ({}/{} rows true)", prop, tt.satisfying_count, tt.rows.len())
        } else if tt.is_contradiction {
            format!("{} is a CONTRADICTION (0/{} rows true)", prop, tt.rows.len())
        } else {
            format!("{} is CONTINGENT ({}/{} rows true)", prop, tt.satisfying_count, tt.rows.len())
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

    // ─── Internal ────────────────────────────────────────────────────────

    /// Store a solved problem as an HDC-encoded episode for future analogical retrieval
    fn store_episode(&mut self, response: &MathResponse, description: &str) {
        if self.memory.len() >= self.memory_capacity {
            // Evict lowest-Phi episode
            if let Some(min_idx) = self
                .memory
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.phi.partial_cmp(&b.1.phi).unwrap_or(std::cmp::Ordering::Equal))
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
            description,
            self.telemetry.problems_solved
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
        let response = service.solve_linear_system(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &[5.0, 11.0],
        );
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
        let r2 = MathResponse { phi: 0.9, ..r1.clone() };
        let r3 = MathResponse { phi: 0.1, ..r1.clone() };
        let ranked = MathService::rank_by_phi(&[r1, r2, r3]);
        assert_eq!(ranked[0].0, 1); // r2 has highest phi
        assert_eq!(ranked[2].0, 2); // r3 has lowest phi
    }
}
