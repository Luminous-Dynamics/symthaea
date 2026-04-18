// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Invariant-discovery benchmark — measures what Symthaea's
//! `SymbolicRegressor` actually recovers on univariate sequences with
//! known closed forms.
//!
//! # Context
//!
//! The miniF2F / Lean-bridge arc (Phase 1-5 + ingest baseline) measures
//! a *static tactic compiler*. It doesn't exercise Symthaea's cognitive
//! machinery — no HDC, no CfC, no active inference. The Ramanujan arc
//! (Sessions 15-31) built the `conjecture_engine` / `SymbolicRegressor`
//! GP that *does* live inside Symthaea's hdc/ tree, but ended with an
//! explicit "do-not-reopen" close-out after hitting a structural
//! ceiling on Kepler multi-invariant discovery.
//!
//! What's never been measured: how well the existing GP recovers
//! closed forms on *simpler* single-variable sequences. The
//! `observe_X()` functions in `conjecture_engine.rs` give us a pile
//! of ground-truth sequences — this harness feeds a curated set to
//! `SymbolicRegressor::fit()`, measures test-split MSE, and reports a
//! scorecard.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p symthaea-core --release --example invariant_discovery_bench
//! ```
//!
//! Takes ~2-5 min depending on which sequences you enable. CSV to
//! stdout, summary to stderr.

use std::time::Instant;

use symthaea_core::hdc::conjecture_engine::{
    observe_balmer_series, observe_bell_numbers, observe_blackbody_peak, observe_catalan,
    observe_central_binomial_limit, observe_derangement_ratio, observe_fibonacci_ratios,
    observe_hydrogen_energy_levels, observe_kepler_third_law, observe_partitions,
    observe_prime_counting, observe_quantum_harmonic_oscillator, observe_stefan_boltzmann, Expr,
    ObservedSequence, RegressorConfig, SymbolicRegressor,
};

/// One benchmark row: (sequence, expected-shape description).
/// `expected_shape` is informational — the regressor is never told
/// what to find. It's logged in the CSV so a human reader can
/// compare against the discovered formula.
struct Problem {
    name: &'static str,
    expected_shape: &'static str,
    seq: ObservedSequence,
}

fn main() {
    let problems = vec![
        // ─── Physics — first principles-grade ground truths ──────────
        Problem {
            name: "hydrogen_energy_levels",
            expected_shape: "E(n) = -13.6 / n^2",
            seq: observe_hydrogen_energy_levels(30),
        },
        Problem {
            name: "quantum_harmonic_oscillator",
            expected_shape: "E(n) = n + 0.5",
            seq: observe_quantum_harmonic_oscillator(30),
        },
        Problem {
            name: "balmer_series",
            expected_shape: "1/λ = R*(1/4 - 1/n^2)",
            seq: observe_balmer_series(30),
        },
        Problem {
            name: "kepler_third_law",
            expected_shape: "T^2 = a^3 (i.e. T = a^1.5)",
            seq: observe_kepler_third_law(30),
        },
        Problem {
            name: "stefan_boltzmann",
            expected_shape: "P = σ * T^4",
            seq: observe_stefan_boltzmann(30),
        },
        Problem {
            name: "blackbody_peak",
            expected_shape: "λ_peak = b / T (Wien's law)",
            seq: observe_blackbody_peak(30),
        },
        // ─── Combinatorics — varying difficulty ───────────────────────
        Problem {
            name: "partitions",
            expected_shape: "p(n) ~ exp(π√(2n/3)) / (4n√3)  (Hardy-Ramanujan)",
            seq: observe_partitions(30),
        },
        Problem {
            name: "bell_numbers",
            expected_shape: "B(n) ~ (n/W(n))^n * e^((n/W(n))-n-1) / sqrt(…)",
            seq: observe_bell_numbers(15),
        },
        Problem {
            name: "catalan",
            expected_shape: "C(n) = (2n)! / ((n+1)! n!), ~ 4^n / n^1.5",
            seq: observe_catalan(20),
        },
        Problem {
            name: "central_binomial_limit",
            expected_shape: "C(2n,n) * n^0.5 / 4^n → 1/√π (constant)",
            seq: observe_central_binomial_limit(30),
        },
        Problem {
            name: "derangement_ratio",
            expected_shape: "!n / n! → 1/e ≈ 0.3679 (constant)",
            seq: observe_derangement_ratio(15),
        },
        Problem {
            name: "fibonacci_ratios",
            expected_shape: "F(n+1)/F(n) → φ = (1+√5)/2 ≈ 1.618 (constant)",
            seq: observe_fibonacci_ratios(30),
        },
        // ─── Number theory ─────────────────────────────────────────────
        Problem {
            name: "prime_counting",
            expected_shape: "π(n) ~ n / ln(n)",
            seq: observe_prime_counting(100),
        },
    ];

    eprintln!("Invariant-discovery benchmark");
    eprintln!("  regressor: SymbolicRegressor, default config (pop=200 gen=100 seed=42)");
    eprintln!("  measure: test-split MSE of top-1 conjecture");
    eprintln!("  problems: {}", problems.len());
    eprintln!();

    println!(
        "name,n_points,expected_shape,top_formula,train_mse,test_mse,test_rel_err,complexity,verdict,elapsed_ms"
    );

    let mut closed = 0usize;
    let mut total = 0usize;
    let start_all = Instant::now();

    for p in &problems {
        total += 1;
        let (train, test) = p.seq.train_test_split();
        if train.len() < 3 || test.is_empty() {
            eprintln!(
                "  {}: skipped (n={}, train={}, test={})",
                p.name,
                p.seq.data.len(),
                train.len(),
                test.len()
            );
            continue;
        }

        let config = RegressorConfig {
            seed: 42,
            population_size: 200,
            generations: 100,
            max_depth: 5,
            max_complexity: 20,
            ..RegressorConfig::default()
        };
        let mut regressor = SymbolicRegressor::new(config);
        let start = Instant::now();
        let results = regressor.fit(&p.seq, 3);
        let elapsed_ms = start.elapsed().as_millis();

        let best = results.into_iter().next();
        let (train_mse, test_mse, test_rel_err, complexity, formula_str) = match best.as_ref() {
            Some(c) => {
                let test_mse = compute_mse_on(&c.formula, &test);
                let test_rel_err = relative_error(&c.formula, &test);
                (
                    c.training_mse,
                    test_mse,
                    test_rel_err,
                    c.complexity,
                    c.formula_str.clone(),
                )
            }
            None => (
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                0,
                String::from("(none)"),
            ),
        };

        // Pass criterion: relative RMS error on test ≤ 5%. Relative
        // handles the fact that e.g. Stefan-Boltzmann has T^4 values
        // spanning 8+ orders of magnitude — absolute MSE is
        // meaningless, relative is what a physicist cares about.
        let closed_flag = test_rel_err.is_finite() && test_rel_err < 0.05;
        let verdict = if closed_flag {
            closed += 1;
            "closed"
        } else if test_rel_err.is_finite() && test_rel_err < 0.50 {
            "partial"
        } else {
            "missed"
        };

        println!(
            "{},{},\"{}\",\"{}\",{:.4e},{:.4e},{:.4e},{},{},{}",
            p.name,
            p.seq.data.len(),
            csv_escape(p.expected_shape),
            csv_escape(&formula_str),
            train_mse,
            test_mse,
            test_rel_err,
            complexity,
            verdict,
            elapsed_ms,
        );
        eprintln!(
            "  {:<32} {:>8} {:>12.2e}  {}",
            p.name, complexity, test_rel_err, verdict
        );
    }

    let total_elapsed_s = start_all.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("━━━ Scorecard ━━━");
    eprintln!(
        "  closed (test_rel_err < 5%):  {closed:3} / {total}  ({:.1}%)",
        100.0 * closed as f64 / total as f64
    );
    eprintln!("  total elapsed:               {total_elapsed_s:.1}s");
    eprintln!();
    eprintln!("  This number measures what SymbolicRegressor recovers on");
    eprintln!("  univariate sequences with known closed forms. Independent");
    eprintln!("  of the Lean bridge — exercises the cognitive-adjacent GP.");
}

fn compute_mse_on(expr: &Expr, data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for (x, y) in data {
        let pred = expr.eval(&[("n", *x)]);
        if !pred.is_finite() {
            return f64::INFINITY;
        }
        let diff = pred - y;
        sum += diff * diff;
        count += 1.0;
    }
    sum / count
}

fn relative_error(expr: &Expr, data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return f64::INFINITY;
    }
    // RMS relative error. For y≈0 we fall back to an absolute tolerance
    // so a zero reference doesn't blow up to infinity spuriously.
    let mut sq_sum = 0.0;
    let mut count = 0.0;
    for (x, y) in data {
        let pred = expr.eval(&[("n", *x)]);
        if !pred.is_finite() {
            return f64::INFINITY;
        }
        let diff = pred - y;
        let denom = y.abs().max(1e-12);
        let rel = diff / denom;
        sq_sum += rel * rel;
        count += 1.0;
    }
    (sq_sum / count).sqrt()
}

fn csv_escape(s: &str) -> String {
    s.replace('"', "\"\"").replace('\n', " ")
}
