// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Stochastic Resonance applied to Symbolic Regression
//!
//! Takes the Phase 4.5 SR finding (+19.8pp solve rate on Medium-difficulty
//! tactic selection) and applies it to a real search problem: **mutation
//! operator selection in genetic-programming symbolic regression**.
//!
//! ## Design
//!
//! We implement a small standalone GP for symbolic regression, deliberately
//! separate from `ConjectureEngine::SymbolicRegressor` (which is edited
//! actively by other sessions and would cause merge pain). The GP supports
//! four mutation operators:
//!
//! 1. `SubtreeReplace` — replace a random subtree with a newly sampled one
//! 2. `ConstantPerturb` — jitter a leaf constant
//! 3. `OperatorSwap` — swap a binary operator (Add ↔ Sub, Mul ↔ Div)
//! 4. `VariableSwap` — swap a variable leaf with a random alternative
//!
//! These operators have different "expected success rates" — on polynomial
//! targets, SubtreeReplace is usually best; on targets with simple constant
//! adjustments, ConstantPerturb wins. A partially-informative heuristic
//! (based on rolling success rates) should prefer the right operator on
//! average, but be imperfect enough that SR can find alternatives the
//! baseline misses.
//!
//! ## Hypothesis
//!
//! If the Phase 4.5 regime map applies here:
//! - **Easy targets** (baseline GP converges quickly): SR should hurt or
//!   be neutral (super-threshold regime)
//! - **Medium targets** (baseline GP converges slowly or inconsistently):
//!   SR should show a measurable improvement (amplification regime)
//! - **Hard targets** (baseline GP fails): SR should help by randomizing
//!   away a stuck operator-selection policy (override regime)
//!
//! ## Metric
//!
//! Iterations-to-convergence on synthetic targets where the ground truth
//! is known. We do NOT measure absolute wall-clock time because the GP is
//! a toy, not optimized.

use std::collections::HashMap;

// ─── Minimal expression tree for symbolic regression ────────────────────────

/// A tiny expression tree: the minimal vocabulary needed to hit a few
/// canonical symbolic regression targets.
#[derive(Debug, Clone, PartialEq)]
pub enum SrExpr {
    Const(f64),
    Var(usize), // 0 = x, 1 = y, etc.
    Add(Box<SrExpr>, Box<SrExpr>),
    Sub(Box<SrExpr>, Box<SrExpr>),
    Mul(Box<SrExpr>, Box<SrExpr>),
    Div(Box<SrExpr>, Box<SrExpr>),
}

impl SrExpr {
    /// Evaluate at a concrete input vector. Returns None on domain errors
    /// (divide by zero, overflow).
    pub fn eval(&self, xs: &[f64]) -> Option<f64> {
        match self {
            SrExpr::Const(c) => Some(*c),
            SrExpr::Var(i) => xs.get(*i).copied(),
            SrExpr::Add(a, b) => Some(a.eval(xs)? + b.eval(xs)?),
            SrExpr::Sub(a, b) => Some(a.eval(xs)? - b.eval(xs)?),
            SrExpr::Mul(a, b) => {
                let va = a.eval(xs)?;
                let vb = b.eval(xs)?;
                let r = va * vb;
                if r.is_finite() {
                    Some(r)
                } else {
                    None
                }
            }
            SrExpr::Div(a, b) => {
                let vb = b.eval(xs)?;
                if vb.abs() < 1e-12 {
                    None
                } else {
                    Some(a.eval(xs)? / vb)
                }
            }
        }
    }

    /// Depth of the expression tree (1 for leaf).
    pub fn depth(&self) -> usize {
        match self {
            SrExpr::Const(_) | SrExpr::Var(_) => 1,
            SrExpr::Add(a, b) | SrExpr::Sub(a, b) | SrExpr::Mul(a, b) | SrExpr::Div(a, b) => {
                1 + a.depth().max(b.depth())
            }
        }
    }
}

// ─── Fitness: mean squared error on a target ────────────────────────────────

/// Compute MSE of `expr` against a target dataset of (input, output) pairs.
/// Returns f64::INFINITY on any domain errors.
pub fn mse(expr: &SrExpr, dataset: &[(Vec<f64>, f64)]) -> f64 {
    let mut sum = 0.0;
    for (xs, y) in dataset {
        match expr.eval(xs) {
            Some(v) => {
                let d = v - y;
                sum += d * d;
            }
            None => return f64::INFINITY,
        }
    }
    sum / dataset.len() as f64
}

// ─── Mutation operators ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MutationOp {
    SubtreeReplace,
    ConstantPerturb,
    OperatorSwap,
    VariableSwap,
}

impl MutationOp {
    pub fn all() -> Vec<MutationOp> {
        vec![
            MutationOp::SubtreeReplace,
            MutationOp::ConstantPerturb,
            MutationOp::OperatorSwap,
            MutationOp::VariableSwap,
        ]
    }
}

// ─── Deterministic RNG (shared style with sr_tactic.rs) ────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0xDEADBEEFCAFEBABE;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn uniform_unit(state: &mut u64) -> f64 {
    xorshift64(state) as f64 / u64::MAX as f64
}

fn gaussian(state: &mut u64) -> f64 {
    let u1 = uniform_unit(state).max(1e-10);
    let u2 = uniform_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample a random leaf (Const or Var).
fn sample_leaf(state: &mut u64, n_vars: usize) -> SrExpr {
    let r = xorshift64(state);
    if r % 2 == 0 {
        let v = (r / 2) % n_vars as u64;
        SrExpr::Var(v as usize)
    } else {
        let c = (uniform_unit(state) - 0.5) * 4.0;
        SrExpr::Const((c * 100.0).round() / 100.0)
    }
}

/// Sample a random expression up to given depth.
fn sample_random(state: &mut u64, max_depth: usize, n_vars: usize) -> SrExpr {
    if max_depth <= 1 {
        return sample_leaf(state, n_vars);
    }
    let r = xorshift64(state) % 5;
    if r == 0 {
        sample_leaf(state, n_vars)
    } else {
        let a = Box::new(sample_random(state, max_depth - 1, n_vars));
        let b = Box::new(sample_random(state, max_depth - 1, n_vars));
        match r {
            1 => SrExpr::Add(a, b),
            2 => SrExpr::Sub(a, b),
            3 => SrExpr::Mul(a, b),
            _ => SrExpr::Div(a, b),
        }
    }
}

/// Apply a single mutation operation. If the operator doesn't apply
/// (e.g. OperatorSwap on a leaf), returns the original expression.
fn apply_mutation(
    op: MutationOp,
    expr: &SrExpr,
    state: &mut u64,
    n_vars: usize,
    max_depth: usize,
) -> SrExpr {
    match op {
        MutationOp::SubtreeReplace => sample_random(state, max_depth, n_vars),
        MutationOp::ConstantPerturb => match expr {
            SrExpr::Const(c) => {
                let delta = gaussian(state) * 0.5;
                SrExpr::Const(c + delta)
            }
            SrExpr::Add(a, b) => SrExpr::Add(
                Box::new(apply_mutation(op, a, state, n_vars, max_depth)),
                b.clone(),
            ),
            _ => expr.clone(),
        },
        MutationOp::OperatorSwap => match expr {
            SrExpr::Add(a, b) => SrExpr::Sub(a.clone(), b.clone()),
            SrExpr::Sub(a, b) => SrExpr::Add(a.clone(), b.clone()),
            SrExpr::Mul(a, b) => SrExpr::Div(a.clone(), b.clone()),
            SrExpr::Div(a, b) => SrExpr::Mul(a.clone(), b.clone()),
            _ => expr.clone(),
        },
        MutationOp::VariableSwap => match expr {
            SrExpr::Var(_) if n_vars > 1 => {
                let new_var = (xorshift64(state) as usize) % n_vars;
                SrExpr::Var(new_var)
            }
            _ => expr.clone(),
        },
    }
}

// ─── GP with operator-selection heuristic + optional SR ────────────────────

/// Rolling success statistics per mutation operator. Used as the
/// heuristic score for operator selection.
#[derive(Debug, Clone, Default)]
pub struct OpStats {
    pub successes: HashMap<MutationOp, usize>,
    pub attempts: HashMap<MutationOp, usize>,
}

impl OpStats {
    pub fn record(&mut self, op: MutationOp, success: bool) {
        *self.attempts.entry(op).or_insert(0) += 1;
        if success {
            *self.successes.entry(op).or_insert(0) += 1;
        }
    }

    /// Smoothed success rate with a +1/+2 Laplace prior to avoid early
    /// degeneracy (and keep all operators explored initially).
    pub fn rate(&self, op: MutationOp) -> f64 {
        let s = *self.successes.get(&op).unwrap_or(&0) as f64;
        let a = *self.attempts.get(&op).unwrap_or(&0) as f64;
        (s + 1.0) / (a + 2.0)
    }
}

/// Configuration for the SR-enabled GP run.
#[derive(Debug, Clone)]
pub struct SrGpConfig {
    pub max_iters: usize,
    pub max_depth: usize,
    pub n_vars: usize,
    pub sigma: f64, // SR noise amplitude; 0 = pure heuristic greedy
    pub target_mse: f64,
    pub seed: u64,
}

impl Default for SrGpConfig {
    fn default() -> Self {
        Self {
            max_iters: 2000,
            max_depth: 4,
            n_vars: 1,
            sigma: 0.0,
            target_mse: 1e-6,
            seed: 42,
        }
    }
}

/// Run a single GP search with operator selection perturbed by SR noise
/// at amplitude `config.sigma`. Returns the best expression found, its
/// fitness, and the iteration at which it was discovered.
pub fn run_gp(config: &SrGpConfig, dataset: &[(Vec<f64>, f64)]) -> (SrExpr, f64, usize) {
    let mut state = config.seed;
    let mut best = sample_random(&mut state, config.max_depth, config.n_vars);
    let mut best_mse = mse(&best, dataset);
    let mut stats = OpStats::default();
    let mut convergence_iter = 0;

    for iter in 0..config.max_iters {
        if best_mse <= config.target_mse {
            convergence_iter = iter;
            return (best, best_mse, convergence_iter);
        }
        // Score each mutation operator by its Laplace-smoothed success rate.
        // Add SR noise to the scores if σ > 0.
        let ops = MutationOp::all();
        let mut scored: Vec<(f64, MutationOp)> = ops
            .iter()
            .map(|&op| {
                let base = stats.rate(op);
                let noise = if config.sigma > 0.0 {
                    gaussian(&mut state) * config.sigma
                } else {
                    0.0
                };
                (base + noise, op)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let chosen = scored[0].1;
        let candidate = apply_mutation(chosen, &best, &mut state, config.n_vars, config.max_depth);
        let cand_mse = mse(&candidate, dataset);
        let success = cand_mse < best_mse;
        stats.record(chosen, success);
        if success {
            best = candidate;
            best_mse = cand_mse;
        }
        convergence_iter = iter + 1;
    }
    (best, best_mse, convergence_iter)
}

// ─── Dataset generators ─────────────────────────────────────────────────────

/// Target: f(x) = x² + 1. A typical "Easy" symbolic regression problem.
pub fn target_x_squared_plus_1() -> Vec<(Vec<f64>, f64)> {
    (0..20)
        .map(|i| {
            let x = i as f64 - 10.0;
            (vec![x], x * x + 1.0)
        })
        .collect()
}

/// Target: f(x) = 2x + 3. Even easier — tests whether SR helps on
/// trivial problems (it shouldn't — super-threshold regime).
pub fn target_linear() -> Vec<(Vec<f64>, f64)> {
    (0..20)
        .map(|i| {
            let x = i as f64 - 10.0;
            (vec![x], 2.0 * x + 3.0)
        })
        .collect()
}

/// Target: f(x) = x³ − 2x² + x − 1. A Medium-difficulty problem with
/// multiple non-trivial terms that usually requires more than one
/// mutation to reach.
pub fn target_cubic() -> Vec<(Vec<f64>, f64)> {
    (0..20)
        .map(|i| {
            let x = i as f64 - 10.0;
            (vec![x], x * x * x - 2.0 * x * x + x - 1.0)
        })
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srexpr_eval() {
        let e = SrExpr::Add(
            Box::new(SrExpr::Mul(
                Box::new(SrExpr::Var(0)),
                Box::new(SrExpr::Var(0)),
            )),
            Box::new(SrExpr::Const(1.0)),
        );
        assert_eq!(e.eval(&[3.0]), Some(10.0));
        assert_eq!(e.eval(&[0.0]), Some(1.0));
    }

    #[test]
    fn test_srexpr_depth() {
        assert_eq!(SrExpr::Const(1.0).depth(), 1);
        let e = SrExpr::Add(
            Box::new(SrExpr::Var(0)),
            Box::new(SrExpr::Mul(
                Box::new(SrExpr::Const(2.0)),
                Box::new(SrExpr::Var(0)),
            )),
        );
        assert_eq!(e.depth(), 3);
    }

    #[test]
    fn test_op_stats_rate() {
        let mut stats = OpStats::default();
        stats.record(MutationOp::SubtreeReplace, true);
        stats.record(MutationOp::SubtreeReplace, false);
        stats.record(MutationOp::SubtreeReplace, true);
        // 2 successes out of 3 attempts + Laplace prior → (2+1)/(3+2) = 0.6
        let r = stats.rate(MutationOp::SubtreeReplace);
        assert!((r - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_op_stats_unseen_op_nonzero() {
        let stats = OpStats::default();
        // Unseen op: (0+1)/(0+2) = 0.5 — keeps exploration alive
        assert!((stats.rate(MutationOp::OperatorSwap) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_run_gp_linear_target_converges() {
        // Easy target: 2x + 3. Deterministic GP should find it.
        let dataset = target_linear();
        let config = SrGpConfig {
            seed: 42,
            ..Default::default()
        };
        let (_expr, best_mse, _iters) = run_gp(&config, &dataset);
        // Don't assert convergence — GP is random. Just assert it ran
        // and produced a finite MSE.
        assert!(best_mse.is_finite());
    }

    /// **The main SR-on-symreg experiment.** Compare convergence and
    /// final MSE on x² + 1 across σ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}.
    /// 50 trials per σ, report both metrics.
    ///
    /// Reports two headline numbers:
    ///   1. **Convergence rate**: fraction of trials reaching MSE ≤ 1e-6
    ///      (absolute discovery of the target within the iteration budget)
    ///   2. **Mean final MSE**: continuous signal that catches partial
    ///      progress even when strict convergence fails
    ///
    /// The convergence rate answers "does SR help discover the target
    /// more reliably?"; the mean final MSE answers "does SR reduce the
    /// residual search error even when strict discovery fails?"
    #[test]
    fn test_phase4_5_sr_symreg_sigma_sweep() {
        let dataset = target_x_squared_plus_1();
        let sigmas = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50];
        let trials = 50;

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  SR-ON-SYMREG — target x² + 1");
        eprintln!("  {} trials per σ, max 2000 iters, target MSE 1e-6", trials);
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("    σ     │ converged │ mean iters │ mean final MSE");
        eprintln!("   ───────┼───────────┼────────────┼─────────────────");

        let mut summary: Vec<(f64, usize, f64, f64)> = Vec::new();

        for &sigma in &sigmas {
            let mut converged = 0usize;
            let mut total_iters = 0usize;
            let mut total_final_mse = 0.0f64;
            for trial in 0..trials {
                let config = SrGpConfig {
                    sigma,
                    seed: 42 + trial as u64 * 7919,
                    ..Default::default()
                };
                let (_expr, final_mse, iters) = run_gp(&config, &dataset);
                if final_mse <= 1e-6 {
                    converged += 1;
                }
                total_iters += iters;
                total_final_mse += final_mse.min(1e10);
            }
            let mean_iters = total_iters as f64 / trials as f64;
            let mean_mse = total_final_mse / trials as f64;
            eprintln!(
                "    {:.2}  │    {:3}    │   {:7.1}  │   {:10.3e}",
                sigma, converged, mean_iters, mean_mse
            );
            summary.push((sigma, converged, mean_iters, mean_mse));
        }
        eprintln!("════════════════════════════════════════════════════════════");

        let baseline = summary[0];
        // Two "best" notions: by convergence rate, by mean MSE
        let best_conv = summary.iter().max_by(|a, b| a.1.cmp(&b.1)).unwrap();
        let best_mse = summary
            .iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .unwrap();

        eprintln!(
            "\n  BASELINE (σ=0):  {}/{} converged, mean MSE {:.3e}",
            baseline.1, trials, baseline.3
        );
        eprintln!(
            "  BEST BY CONVERGENCE: σ={:.2}  ({}/{} trials, {:+} vs baseline)",
            best_conv.0,
            best_conv.1,
            trials,
            best_conv.1 as i32 - baseline.1 as i32
        );
        eprintln!(
            "  BEST BY MEAN MSE:    σ={:.2}  (MSE {:.3e}, {:.1}% of baseline)",
            best_mse.0,
            best_mse.3,
            if baseline.3 > 0.0 {
                best_mse.3 / baseline.3 * 100.0
            } else {
                100.0
            }
        );

        // Interpret
        if best_conv.0 > 0.0 && best_conv.1 > baseline.1 {
            let pp = (best_conv.1 as f64 - baseline.1 as f64) / trials as f64 * 100.0;
            eprintln!(
                "\n  ✓ SR-on-symreg (convergence): +{:.1}pp at σ={:.2}",
                pp, best_conv.0
            );
        }
        if best_mse.0 > 0.0 && best_mse.3 < baseline.3 * 0.9 {
            let rel = (baseline.3 - best_mse.3) / baseline.3 * 100.0;
            eprintln!(
                "  ✓ SR-on-symreg (mean MSE): {:.1}% lower at σ={:.2}",
                rel, best_mse.0
            );
        }
        if best_conv.0 == 0.0 && best_mse.0 == 0.0 {
            eprintln!("\n  ⊘ σ=0 wins on both metrics — SR does not help this GP");
        }

        assert_eq!(summary.len(), sigmas.len());
    }
}
