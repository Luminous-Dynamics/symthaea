// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Solver-Correctness Gate.
//!
//! An *honesty gate* for the mathematics suite. Several benchmarks in this
//! module (constraint_puzzles, logical_deduction, proof_construction,
//! statistical_inference) score **HDC structural discriminability** — how well
//! a hypervector encoding separates valid from invalid structure — rather than
//! whether the system actually *computes the right answer*. Those are different
//! questions, and a high HDC-discriminability score can coexist with a solver
//! that never runs.
//!
//! This gate closes that ambiguity. For each domain that has a real underlying
//! engine in `symthaea-core`, it invokes the **actual solver / decision
//! procedure** on problems with known ground-truth answers and reports true
//! pass/fail accuracy:
//!
//! 1. **N-Queens** — runs `CSPSolver::solve_n_queens(n)` (real AC-3 + backtracking)
//!    for n ∈ {4,6,8} and *verifies* the returned placement is conflict-free.
//! 2. **Argument validity** — decides propositional-argument validity with
//!    `LogicEngine::is_satisfiable`: an argument is valid iff
//!    `(premises ∧ ¬conclusion)` is UNSAT. Uses the real DPLL SAT solver.
//! 3. **Tautology / contradiction** — a formula F is a tautology iff `¬F` is
//!    UNSAT and a contradiction iff `F` is UNSAT. Both decided by DPLL.
//! 4. **Descriptive statistics** — `statistics::{mean, variance}` against
//!    datasets with closed-form mean/variance.
//!
//! Unlike the HDC-scored siblings, a correct engine here scores ~1.0 — that is
//! the point: this benchmark proves the solvers work end-to-end. A regression
//! that broke a solver would drop these metrics far below the HDC-similarity
//! scores, exposing a fault the discriminability benchmarks would mask.
//!
//! Baseline note: there is no human paradigm here — this measures *machine*
//! solver correctness, so metrics are expected near 1.0 for a healthy build.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};

use symthaea_core::hdc::constraint_solver::CSPSolver;
use symthaea_core::hdc::logic_engine::{LogicEngine, Proposition};
use symthaea_core::hdc::statistics;

/// Solver-correctness gate for the mathematics suite.
pub struct SolverCorrectnessGateBenchmark;

/// Verify an N-Queens placement: `sol[row]` = column of the queen in that row.
/// Valid iff no two queens share a column or a diagonal (rows are distinct by
/// construction).
fn n_queens_placement_valid(sol: &[i64]) -> bool {
    let n = sol.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (ci, cj) = (sol[i], sol[j]);
            if ci == cj {
                return false; // same column
            }
            if (ci - cj).abs() == (i as i64 - j as i64).abs() {
                return false; // same diagonal
            }
        }
    }
    true
}

/// Fraction of {4,6,8}-Queens the real CSP solver places conflict-free.
fn n_queens_solve_rate() -> f64 {
    let sizes = [4usize, 6, 8];
    let mut solved = 0usize;
    for &n in &sizes {
        if let Some(sol) = CSPSolver::solve_n_queens(n) {
            if sol.len() == n && n_queens_placement_valid(&sol) {
                solved += 1;
            }
        }
    }
    solved as f64 / sizes.len() as f64
}

/// True iff `prop` is a tautology (its negation is unsatisfiable).
fn is_tautology(prop: &Proposition) -> bool {
    !LogicEngine::is_satisfiable(&prop.clone().not())
}

/// True iff `prop` is a contradiction (unsatisfiable).
fn is_contradiction(prop: &Proposition) -> bool {
    !LogicEngine::is_satisfiable(prop)
}

/// Decide whether `premises ⊢ conclusion` is a valid entailment via SAT:
/// valid iff `(⋀ premises) ∧ ¬conclusion` is unsatisfiable.
fn argument_valid(premises: &[Proposition], conclusion: &Proposition) -> bool {
    let mut conj = conclusion.clone().not();
    for p in premises {
        conj = conj.and(p.clone());
    }
    !LogicEngine::is_satisfiable(&conj)
}

/// Accuracy of the DPLL decision procedure on a fixed battery of valid and
/// invalid propositional arguments with known labels.
fn argument_validity_accuracy() -> f64 {
    let p = || Proposition::atom("P");
    let q = || Proposition::atom("Q");
    let r = || Proposition::atom("R");

    // (premises, conclusion, expected_valid)
    let cases: Vec<(Vec<Proposition>, Proposition, bool)> = vec![
        // Modus ponens: P→Q, P ⊢ Q  (valid)
        (vec![p().implies(q()), p()], q(), true),
        // Modus tollens: P→Q, ¬Q ⊢ ¬P  (valid)
        (vec![p().implies(q()), q().not()], p().not(), true),
        // Hypothetical syllogism: P→Q, Q→R ⊢ P→R  (valid)
        (
            vec![p().implies(q()), q().implies(r())],
            p().implies(r()),
            true,
        ),
        // Disjunctive syllogism: P∨Q, ¬P ⊢ Q  (valid)
        (vec![p().or(q()), p().not()], q(), true),
        // Affirming the consequent: P→Q, Q ⊢ P  (invalid)
        (vec![p().implies(q()), q()], p(), false),
        // Denying the antecedent: P→Q, ¬P ⊢ ¬Q  (invalid)
        (vec![p().implies(q()), p().not()], q().not(), false),
    ];

    let mut correct = 0usize;
    for (premises, conclusion, expected) in &cases {
        if argument_valid(premises, conclusion) == *expected {
            correct += 1;
        }
    }
    correct as f64 / cases.len() as f64
}

/// Accuracy of tautology/contradiction detection on a fixed labelled battery.
fn tautology_detection_accuracy() -> f64 {
    let p = || Proposition::atom("P");
    let q = || Proposition::atom("Q");
    let r = || Proposition::atom("R");

    // (formula, is_tautology, is_contradiction)
    let cases: Vec<(Proposition, bool, bool)> = vec![
        // Law of excluded middle: P ∨ ¬P — tautology
        (p().or(p().not()), true, false),
        // Chained implication tautology: (P→Q)→((Q→R)→(P→R))
        (
            p().implies(q())
                .implies(q().implies(r()).implies(p().implies(r()))),
            true,
            false,
        ),
        // Non-contradiction: ¬(P ∧ ¬P) — tautology
        (p().and(p().not()).not(), true, false),
        // P ∧ ¬P — contradiction
        (p().and(p().not()), false, true),
        // P → Q — contingent (neither)
        (p().implies(q()), false, false),
        // P — contingent (neither)
        (p(), false, false),
    ];

    let mut correct = 0usize;
    let total = cases.len() * 2; // two independent decisions per case
    for (formula, taut, contra) in &cases {
        if is_tautology(formula) == *taut {
            correct += 1;
        }
        if is_contradiction(formula) == *contra {
            correct += 1;
        }
    }
    correct as f64 / total as f64
}

/// Accuracy of descriptive statistics against closed-form mean/variance.
///
/// Uses arithmetic sequences whose population mean and variance have exact
/// closed forms, so any deviation is a real solver fault, not sampling noise.
fn descriptive_stats_accuracy() -> f64 {
    // Data 0..n-1: mean = (n-1)/2, population variance = (n^2 - 1)/12.
    let batteries = [10usize, 25, 100];
    let mut checks = 0usize;
    let mut correct = 0usize;

    for &n in &batteries {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let true_mean = (n as f64 - 1.0) / 2.0;
        let true_var = ((n * n) as f64 - 1.0) / 12.0;

        let got_mean = statistics::mean(&data);
        // symthaea-core `variance` is the population variance (÷ N).
        let got_var = statistics::variance(&data);

        checks += 2;
        if (got_mean - true_mean).abs() <= 1e-9 {
            correct += 1;
        }
        // Relative tolerance for variance (larger magnitudes).
        if (got_var - true_var).abs() <= 1e-6 * true_var.max(1.0) {
            correct += 1;
        }
    }
    correct as f64 / checks as f64
}

impl SolverCorrectnessGateBenchmark {
    /// One trial evaluates every gate once. Because these are exact decision
    /// procedures, results are deterministic across trials — a healthy build
    /// yields identical (near-1.0) values every trial. That determinism is the
    /// desired property of a correctness gate.
    fn run_gates(&self) -> [f64; 5] {
        let queens = n_queens_solve_rate();
        let args = argument_validity_accuracy();
        let taut = tautology_detection_accuracy();
        let stats = descriptive_stats_accuracy();
        let overall = (queens + args + taut + stats) / 4.0;
        [queens, args, taut, stats, overall]
    }
}

impl PsychBenchmark for SolverCorrectnessGateBenchmark {
    fn name(&self) -> &str {
        "Mathematics::SolverCorrectnessGate"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Solver Correctness Gate (machine ground-truth)",
            citation: "Luminous Dynamics (2026), Phase 0 grounding audit",
            year: 2026,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut queens = Vec::with_capacity(config.trials_per_condition);
        let mut args = Vec::with_capacity(config.trials_per_condition);
        let mut taut = Vec::with_capacity(config.trials_per_condition);
        let mut stats = Vec::with_capacity(config.trials_per_condition);
        let mut overall = Vec::with_capacity(config.trials_per_condition);

        for _ in 0..config.trials_per_condition.max(1) {
            let g = self.run_gates();
            queens.push(g[0]);
            args.push(g[1]);
            taut.push(g[2]);
            stats.push(g[3]);
            overall.push(g[4]);
        }

        result.insert("n_queens_solve_rate", MetricValue::from_samples(&queens));
        result.insert(
            "argument_validity_accuracy",
            MetricValue::from_samples(&args),
        );
        result.insert(
            "tautology_detection_accuracy",
            MetricValue::from_samples(&taut),
        );
        result.insert(
            "descriptive_stats_accuracy",
            MetricValue::from_samples(&stats),
        );
        result.insert("solver_gate_overall", MetricValue::from_samples(&overall));

        result.conditions = 4; // queens, arguments, tautology, statistics
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn gate_runs_and_emits_all_metrics() {
        let result = SolverCorrectnessGateBenchmark.run(&test_config());
        for key in [
            "n_queens_solve_rate",
            "argument_validity_accuracy",
            "tautology_detection_accuracy",
            "descriptive_stats_accuracy",
            "solver_gate_overall",
        ] {
            assert!(result.metrics.contains_key(key), "missing metric {key}");
            assert!(result.metrics[key].mean.is_finite());
        }
    }

    #[test]
    fn real_csp_solver_places_queens_conflict_free() {
        // The whole point of the gate: the real solver actually solves it.
        assert_eq!(n_queens_solve_rate(), 1.0);
    }

    #[test]
    fn dpll_decides_argument_validity_perfectly() {
        assert_eq!(argument_validity_accuracy(), 1.0);
    }

    #[test]
    fn dpll_decides_tautology_and_contradiction_perfectly() {
        assert_eq!(tautology_detection_accuracy(), 1.0);
    }

    #[test]
    fn descriptive_stats_match_closed_form() {
        assert_eq!(descriptive_stats_accuracy(), 1.0);
    }

    #[test]
    fn placement_validator_rejects_attacks() {
        // Two queens on the same diagonal: rows 0,1 cols 0,1.
        assert!(!n_queens_placement_valid(&[0, 1]));
        // Same column.
        assert!(!n_queens_placement_valid(&[2, 2]));
        // A known valid 4-queens solution: cols [1,3,0,2].
        assert!(n_queens_placement_valid(&[1, 3, 0, 2]));
    }
}
