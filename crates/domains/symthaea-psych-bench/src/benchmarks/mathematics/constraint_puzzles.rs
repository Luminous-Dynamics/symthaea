// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constraint Puzzles benchmark.
//!
//! Tests solving simple constraint satisfaction problems (CSPs):
//!   1. N-Queens (N=4, 6, 8): place N non-attacking queens on an N×N board
//!   2. Constraint chains: variables with pairwise ordering constraints
//!
//! **Engine-wired (Tier 0.1, 2026-07-06).** Each puzzle is solved by the real
//! `CSPSolver` from `symthaea-core` (AC-3 preprocessing + backtracking with
//! forward checking), and the returned assignment is **independently
//! re-verified** against the puzzle's ground-truth constraints by a local
//! validator that shares no code with the solver. Accuracy therefore measures
//! *computed correctness*, not HDC structural similarity. The previous version
//! of this file scored an HDC-guided random local search that never invoked
//! the solver; that gap was flagged by the Phase 0 grounding audit.
//!
//! HDC still participates as trial structure: the verified solution is encoded
//! as a hypervector and its separation from a corrupted placement is reported
//! as the auxiliary `hdc_solution_separation` metric (not part of accuracy).
//!
//! Noise model: with probability proportional to `effective_noise()`, one
//! queen in the solver's answer is displaced before verification (degraded
//! readout), so accuracy degrades under noise while the noiseless condition
//! reflects true solver correctness.
//!
//! Human baselines (Russell & Norvig 2020):
//! - queens_4_accuracy: ~0.95 (SD~0.05) — N=4 is easy
//! - queens_6_accuracy: ~0.80 (SD~0.10)
//! - queens_8_accuracy: ~0.65 (SD~0.12)
//! - mean_solve_time: search nodes explored (machine effort proxy)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::constraint_solver::{CSP, CSPSolver, Constraint};

use std::collections::HashMap;

/// Constraint Puzzles benchmark.
pub struct ConstraintPuzzlesBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Count constraint violations for an N-Queens placement.
///
/// `placement[i] = j` — one queen per rank `i` at file `j` (the check is
/// symmetric in rows/columns). Independent of the CSP solver: this is the
/// ground-truth verifier.
fn count_queens_violations(placement: &[i64]) -> u32 {
    let n = placement.len();
    let mut violations = 0u32;
    for i in 0..n {
        for j in (i + 1)..n {
            // Same file
            if placement[i] == placement[j] {
                violations += 1;
            }
            // Same diagonal
            let val_diff = (placement[i] - placement[j]).unsigned_abs();
            let idx_diff = (j - i) as u64;
            if val_diff == idx_diff {
                violations += 1;
            }
        }
    }
    violations
}

/// Run the real CSP solver on N-Queens.
///
/// Returns `(placement, nodes_explored)` where `placement[row] = col`, or
/// `None` if the solver reports no solution (impossible for n ≥ 4).
fn solve_queens_with_engine(n: usize) -> Option<(Vec<i64>, usize)> {
    let csp = CSPSolver::n_queens(n);
    let result = CSPSolver::solve(&csp);
    let sol = result.solution?;
    let placement: Vec<i64> = (0..n).map(|i| sol[&format!("Q{}", i)]).collect();
    Some((placement, result.nodes_explored))
}

/// Build a strictly-increasing constraint chain: X0 < X1 < ... < X{len-1}
/// over the domain {0, ..., len-1}. The unique solution is Xi = i, which
/// gives a closed-form ground truth independent of the solver.
fn build_constraint_chain(len: usize) -> CSP {
    let variables: Vec<String> = (0..len).map(|i| format!("X{}", i)).collect();
    let domain: Vec<i64> = (0..len as i64).collect();
    let mut domains = HashMap::new();
    for var in &variables {
        domains.insert(var.clone(), domain.clone());
    }
    let constraints = (0..len - 1)
        .map(|i| Constraint::LessThan(variables[i].clone(), variables[i + 1].clone()))
        .collect();
    CSP {
        domains,
        constraints,
        variables,
    }
}

/// Solve the constraint chain with the real solver and verify the answer
/// against the closed-form unique solution Xi = i.
fn solve_chain_with_engine(len: usize) -> bool {
    let csp = build_constraint_chain(len);
    let result = CSPSolver::solve(&csp);
    match result.solution {
        Some(sol) => (0..len).all(|i| sol.get(&format!("X{}", i)) == Some(&(i as i64))),
        None => false,
    }
}

/// Encode an N-Queens placement as an HDC hypervector.
/// Each (row, col) pair is bound together and bundled across rows.
fn encode_queens(placement: &[i64], dim: usize, seed: u64) -> ContinuousHV {
    let n = placement.len();
    if n == 0 {
        return ContinuousHV::zero(dim);
    }

    let row_hvs: Vec<ContinuousHV> = (0..n)
        .map(|r| ContinuousHV::random(dim, seed.wrapping_add(r as u64 * 1000 + 1)))
        .collect();
    let col_hvs: Vec<ContinuousHV> = (0..n)
        .map(|c| ContinuousHV::random(dim, seed.wrapping_add(c as u64 * 1000 + 500)))
        .collect();

    // Bind each (row, col) placement and bundle all
    let bindings: Vec<ContinuousHV> = (0..n)
        .map(|r| row_hvs[r].bind(&col_hvs[placement[r].rem_euclid(n as i64) as usize]))
        .collect();

    let refs: Vec<&ContinuousHV> = bindings.iter().collect();
    let weights = vec![1.0f32 / n as f32; n];
    ContinuousHV::weighted_bundle(&refs, &weights)
}

/// Solve N-Queens with the real engine, optionally corrupt the readout under
/// noise, and verify against the independent violation counter.
///
/// Returns `(solved_correctly, nodes_explored, hdc_separation)`.
fn queens_trial(
    n: usize,
    dim: usize,
    seed: u64,
    rng: &mut u64,
    noise_weight: f64,
) -> (bool, f64, f64) {
    let Some((mut placement, nodes)) = solve_queens_with_engine(n) else {
        return (false, 0.0, 0.0);
    };

    // HDC trial structure: encode the engine's solution and a deliberately
    // corrupted placement; report their separation (1 - similarity).
    let mut corrupted = placement.clone();
    xor_shift(rng);
    let corrupt_row = (*rng % n as u64) as usize;
    corrupted[corrupt_row] = (corrupted[corrupt_row] + 1).rem_euclid(n as i64);
    let sol_hv = encode_queens(&placement, dim, seed);
    let corr_hv = encode_queens(&corrupted, dim, seed);
    let separation = 1.0 - sol_hv.similarity(&corr_hv) as f64;

    // Noise: degraded readout displaces one queen before verification.
    if noise_weight > 0.0 {
        xor_shift(rng);
        let perturb = (*rng as f64 / u64::MAX as f64) < noise_weight;
        if perturb {
            xor_shift(rng);
            let row = (*rng % n as u64) as usize;
            xor_shift(rng);
            let shift = 1 + (*rng % (n as u64 - 1)) as i64;
            placement[row] = (placement[row] + shift).rem_euclid(n as i64);
        }
    }

    // Independent ground-truth verification.
    let solved = count_queens_violations(&placement) == 0;
    (solved, nodes as f64, separation)
}

struct ConstraintTrial {
    queens_4: f64, // 1.0 if engine solution verified conflict-free, else 0.0
    queens_6: f64,
    queens_8: f64,
    chain: f64,      // 1.0 if chain solution matches closed form
    solve_time: f64, // mean search nodes explored across the 3 queens problems
    hdc_separation: f64,
}

impl ConstraintPuzzlesBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> ConstraintTrial {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "constraint_puzzles", trial_idx);
        let mut rng = seed ^ 0x0F1E2D3C4B5A6978;
        let noise_weight = config.effective_noise();

        let (s4, t4, h4) = queens_trial(4, dim, seed, &mut rng, noise_weight);
        let (s6, t6, h6) = queens_trial(6, dim, seed.wrapping_add(1000), &mut rng, noise_weight);
        let (s8, t8, h8) = queens_trial(8, dim, seed.wrapping_add(2000), &mut rng, noise_weight);

        // Constraint chain (5 variables): unique closed-form solution.
        // Noise: degraded readout invalidates the answer with prob ~ noise.
        let mut chain_ok = solve_chain_with_engine(5);
        if noise_weight > 0.0 {
            xor_shift(&mut rng);
            if (rng as f64 / u64::MAX as f64) < noise_weight {
                chain_ok = false;
            }
        }

        ConstraintTrial {
            queens_4: if s4 { 1.0 } else { 0.0 },
            queens_6: if s6 { 1.0 } else { 0.0 },
            queens_8: if s8 { 1.0 } else { 0.0 },
            chain: if chain_ok { 1.0 } else { 0.0 },
            solve_time: (t4 + t6 + t8) / 3.0,
            hdc_separation: (h4 + h6 + h8) / 3.0,
        }
    }
}

impl PsychBenchmark for ConstraintPuzzlesBenchmark {
    fn name(&self) -> &str {
        "Mathematics::ConstraintPuzzles"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Constraint Satisfaction Assessment",
            citation: "Russell & Norvig (2020)",
            year: 2020,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut q4_accs = Vec::new();
        let mut q6_accs = Vec::new();
        let mut q8_accs = Vec::new();
        let mut chain_accs = Vec::new();
        let mut solve_times = Vec::new();
        let mut hdc_seps = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            q4_accs.push(r.queens_4);
            q6_accs.push(r.queens_6);
            q8_accs.push(r.queens_8);
            chain_accs.push(r.chain);
            solve_times.push(r.solve_time);
            hdc_seps.push(r.hdc_separation);
        }

        result.insert("queens_4_accuracy", MetricValue::from_samples(&q4_accs));
        result.insert("queens_6_accuracy", MetricValue::from_samples(&q6_accs));
        result.insert("queens_8_accuracy", MetricValue::from_samples(&q8_accs));
        result.insert("chain_accuracy", MetricValue::from_samples(&chain_accs));
        result.insert("mean_solve_time", MetricValue::from_samples(&solve_times));
        result.insert(
            "hdc_solution_separation",
            MetricValue::from_samples(&hdc_seps),
        );

        result.conditions = 4; // N = 4, 6, 8 + constraint chain
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
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_constraint_puzzles_runs_and_has_metrics() {
        let result = ConstraintPuzzlesBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("queens_4_accuracy"));
        assert!(result.metrics.contains_key("queens_6_accuracy"));
        assert!(result.metrics.contains_key("queens_8_accuracy"));
        assert!(result.metrics.contains_key("chain_accuracy"));
        assert!(result.metrics.contains_key("mean_solve_time"));
        assert!(result.metrics.contains_key("hdc_solution_separation"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ConstraintPuzzlesBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} mean is not finite", key);
            assert!(
                val.std_dev.is_finite(),
                "metric {} std_dev is not finite",
                key
            );
        }
    }

    #[test]
    fn test_queens_4_easier_than_queens_8() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ConstraintPuzzlesBenchmark.run(&config);
        let q4 = result.metrics["queens_4_accuracy"].mean;
        let q8 = result.metrics["queens_8_accuracy"].mean;
        // N=4 should be easier (higher accuracy) than N=8
        assert!(
            q4 >= q8 - 0.15,
            "4-queens ({:.3}) should be at least as easy as 8-queens ({:.3})",
            q4,
            q8
        );
    }

    #[test]
    fn test_violations_counter() {
        // Known N=4 solution: [1, 3, 0, 2]
        let solution = vec![1i64, 3, 0, 2];
        let violations = count_queens_violations(&solution);
        assert_eq!(violations, 0, "Should be a valid N=4 queens solution");

        // Invalid placement: all at the same file
        let invalid = vec![0i64, 0, 0, 0];
        let v = count_queens_violations(&invalid);
        assert!(v > 0, "All-same-file should have violations");
    }

    /// Proves the REAL engine is invoked: at zero noise the CSP solver must
    /// solve every instance and the independent verifier must confirm it.
    /// The old HDC random local search could not achieve this reliably.
    #[test]
    fn test_real_engine_solves_all_sizes_at_zero_noise() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            encoding_noise: 0.0,
            time_pressure: 0.0,
            ..Default::default()
        };
        let result = ConstraintPuzzlesBenchmark.run(&config);
        assert_eq!(result.metrics["queens_4_accuracy"].mean, 1.0);
        assert_eq!(result.metrics["queens_6_accuracy"].mean, 1.0);
        assert_eq!(result.metrics["queens_8_accuracy"].mean, 1.0);
        assert_eq!(result.metrics["chain_accuracy"].mean, 1.0);
        // Real search does work: nodes explored must be nonzero.
        assert!(result.metrics["mean_solve_time"].mean >= 1.0);
    }

    /// Proves the benchmark CAN fail: a wrong answer (corrupted placement)
    /// is rejected by the ground-truth verifier.
    #[test]
    fn test_wrong_answer_scores_low() {
        let (placement, _) = solve_queens_with_engine(8).expect("8-queens is solvable");
        assert_eq!(count_queens_violations(&placement), 0);

        // Corrupt one queen onto another queen's file: must be rejected.
        let mut wrong = placement.clone();
        wrong[0] = wrong[1];
        assert!(
            count_queens_violations(&wrong) > 0,
            "verifier must reject a corrupted placement"
        );

        // Chain: wrong closed-form check rejects a permuted answer.
        let csp = build_constraint_chain(5);
        let mut bad = HashMap::new();
        for (i, v) in csp.variables.iter().enumerate() {
            bad.insert(v.clone(), (4 - i) as i64); // decreasing — violates chain
        }
        assert!(
            csp.constraints.iter().any(|c| !c.is_satisfied(&bad)),
            "decreasing assignment must violate the LessThan chain"
        );
    }
}
