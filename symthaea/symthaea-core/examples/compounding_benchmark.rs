// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Multi-Level Compounding Benchmark
//!
//! Tests the foundational architectural claim: does abstract_thought's
//! feedback loop **compound across iterations**, or does it saturate
//! after one level?
//!
//! ## Experimental design
//!
//! ```
//! Level 0 — Warmup on simple sequences        → extract M₁
//! Level 1 — Run on diverse physics targets
//!           Observe + reflect on engine       → extract M₂
//!           M₂ should strictly contain M₁ + new abstractions
//! Level 2 — Run on harder physics targets
//!           Compare 3 conditions:
//!             - Cold (no macros)
//!             - Primed with M₁ (first-gen only)
//!             - Primed with M₂ (first + second gen)
//!           Compounding = primed_M₂ > primed_M₁ on level 2
//! ```
//!
//! ## Outcomes
//!
//! - **Accumulation**: |M₂| > |M₁|, M₂ wins on level 2 → architecture compounds
//! - **Saturation**: |M₂| ≈ |M₁|, both primed tie → one-shot ceiling
//! - **Drift**: M₂ loses M₁ members, both primed lose to cold → overfit failure
//!
//! ## Run
//!
//! ```sh
//! cargo run --release -p symthaea-core --example compounding_benchmark \
//!   --features abstract_thought
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use symthaea_core::hdc::conjecture_engine::{
    observe_balmer_series, observe_fibonacci_ratios, observe_hydrogen_energy_levels,
    observe_inverse_square_law, observe_kepler_third_law, observe_partitions,
    observe_quantum_harmonic_oscillator, observe_relativistic_kinetic_energy,
    observe_stefan_boltzmann, ConjectureEngine, Expr, MathDomain, ObservedSequence,
    RegressorConfig, SymbolicRegressor,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

const GENERATIONS: usize = 12;
const POP_SIZE: usize = 150;
const SEEDS: &[u64] = &[42, 1337, 2718, 7919, 31415, 65537, 271828];

// ═══════════════════════════════════════════════════════════════════════════
// WARMUP SEQUENCES (Level 0)
// ═══════════════════════════════════════════════════════════════════════════

fn squares_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n).map(|n| (n as f64, (n * n) as f64)).collect();
    ObservedSequence::new("squares(n)", MathDomain::NumberTheory, data)
}

fn cubes_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n).map(|n| (n as f64, (n * n * n) as f64)).collect();
    ObservedSequence::new("cubes(n)", MathDomain::NumberTheory, data)
}

fn triangular_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, (n * (n + 1)) as f64 / 2.0))
        .collect();
    ObservedSequence::new("triangular(n)", MathDomain::Combinatorics, data)
}

fn harmonic_sequence(max_n: usize) -> ObservedSequence {
    let mut sum = 0.0;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| {
            sum += 1.0 / n as f64;
            (n as f64, sum)
        })
        .collect();
    ObservedSequence::new("harmonic(n)", MathDomain::NumberTheory, data)
}

fn kinetic_energy_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n).map(|n| (n as f64, 0.5 * (n * n) as f64)).collect();
    ObservedSequence::new("kinetic_energy(n)", MathDomain::Physics, data)
}

/// Custom: damped exponential target. Exercises the `Exp` primitive —
/// we want Level 1 to produce conjectures containing exp(k*n) subtrees
/// so M₂ can potentially extract them.
fn damped_exponential_sequence(n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=n)
        .map(|i| (i as f64, (-0.1 * i as f64).exp()))
        .collect();
    ObservedSequence::new("damped_exp(t)", MathDomain::Physics, data)
}

/// Custom: logistic growth. Contains nested (1 + exp(...)) structure.
fn logistic_sequence(n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=n)
        .map(|i| {
            let x = i as f64 - (n as f64) / 2.0;
            (i as f64, 1.0 / (1.0 + (-0.3 * x).exp()))
        })
        .collect();
    ObservedSequence::new("logistic(n)", MathDomain::Biology, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct RunMetrics {
    final_mse: f64,
    final_complexity: usize,
    wall_time_ms: u128,
    best_formula: String,
    macro_usage: HashMap<String, u64>,
}

#[derive(Debug, Clone)]
struct ConditionStats {
    label: &'static str,
    mean_final_mse: f64,
    median_final_mse: f64,
    mean_complexity: f64,
    sample_best_formula: String,
}

impl ConditionStats {
    fn from_runs(label: &'static str, runs: &[RunMetrics]) -> Self {
        let n = runs.len() as f64;
        let mean_final_mse = runs.iter().map(|r| r.final_mse).sum::<f64>() / n;
        let mut mses: Vec<f64> = runs.iter().map(|r| r.final_mse).collect();
        mses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_final_mse = mses[mses.len() / 2];
        let mean_complexity = runs.iter().map(|r| r.final_complexity as f64).sum::<f64>() / n;
        let sample_best_formula = runs
            .iter()
            .min_by(|a, b| a.final_mse.partial_cmp(&b.final_mse).unwrap())
            .map(|r| r.best_formula.clone())
            .unwrap_or_default();
        Self {
            label,
            mean_final_mse,
            median_final_mse,
            mean_complexity,
            sample_best_formula,
        }
    }
}

fn run_once(
    target: &ObservedSequence,
    seed: u64,
    macros: &[Expr],
    cold: bool,
) -> RunMetrics {
    let config = RegressorConfig {
        population_size: POP_SIZE,
        generations: GENERATIONS,
        max_depth: 5,
        max_complexity: 20,
        lambda: 0.001,
        tournament_size: 5,
        mutation_rate: 0.3,
        seed,
        disable_macro_seeds: cold,
    };
    let mut regressor = SymbolicRegressor::new(config);
    if !cold {
        regressor.set_seed_macros(macros.to_vec());
    }

    let start = Instant::now();
    let results = regressor.fit(target, 3);
    let elapsed = start.elapsed().as_millis();

    let best = results.into_iter().next();
    let (final_mse, final_complexity, formula_str) = match best {
        Some(c) => (c.training_mse, c.complexity, c.formula_str),
        None => (f64::INFINITY, 0, String::from("(none)")),
    };

    let macro_usage: HashMap<String, u64> = regressor.macro_usage().clone();

    RunMetrics {
        final_mse,
        final_complexity,
        wall_time_ms: elapsed,
        best_formula: formula_str,
        macro_usage,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Level Compounding Benchmark                               ║");
    println!("║  Does abstract_thought compound across iterations or saturate?   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config: {} gens, pop {}, {} seeds per condition", GENERATIONS, POP_SIZE, SEEDS.len());
    println!("  Levels: 0 (warmup) → 1 (extract M₂) → 2 (compare M₁ vs M₂)");

    let prims = PrimitiveSystem::new();

    // ════════════════════════════════════════════════════════════════
    // LEVEL 0: Warmup on simple sequences → extract M₁
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Level 0: Warmup (simple sequences → M₁) ━━━");
    let mut engine = ConjectureEngine::new();
    engine.enable_abstract_thought();

    engine.observe(squares_sequence(25));
    engine.observe(cubes_sequence(25));
    engine.observe(triangular_sequence(25));
    engine.observe(harmonic_sequence(25));
    engine.observe(observe_fibonacci_ratios(25));
    engine.observe(kinetic_energy_sequence(25));

    let t0 = Instant::now();
    engine.generate_conjectures(3);
    engine.verify_numerical();
    engine.reflect(&prims);
    let level0_time = t0.elapsed();

    let m1: Vec<Expr> = engine
        .macro_operators()
        .iter()
        .map(|op| op.template.clone())
        .collect();
    let m1_canonical: HashSet<String> = engine
        .macro_operators()
        .iter()
        .map(|op| format!("{}", op.template))
        .collect();

    println!("  Warmup took {:.2}s", level0_time.as_secs_f64());
    println!("  Conjectures generated: {}", engine.conjectures.len());
    println!("  |M₁| = {} macros:", m1.len());
    for (i, t) in m1.iter().enumerate() {
        println!("    {}. {}", i + 1, t);
    }

    if m1.is_empty() {
        println!("\n  ⚠ No macros extracted at Level 0 — aborting. Check thresholds.");
        return;
    }

    // ════════════════════════════════════════════════════════════════
    // LEVEL 1: Diverse physics targets → feed into engine → extract M₂
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Level 1: Diverse physics targets (M₁ injected, produces M₂) ━━━");
    println!("  Sequences are chosen for structural diversity so new macros");
    println!("  can potentially be extracted beyond what Level 0 produced.");

    // Add non-trivial sequences to the SAME engine so reflect() sees their conjectures.
    engine.observe(observe_partitions(22));           // Hardy-Ramanujan: exp(√n)/n
    engine.observe(observe_balmer_series(12));        // rational: 1/(A - B/n²)
    engine.observe(observe_hydrogen_energy_levels(20)); // -c/n²
    engine.observe(observe_relativistic_kinetic_energy(20)); // transcendental
    engine.observe(damped_exponential_sequence(20));  // exp(-kt)
    engine.observe(logistic_sequence(20));            // 1/(1+exp(-kx))

    let t1 = Instant::now();
    engine.generate_conjectures(3);
    engine.verify_numerical();
    engine.reflect(&prims);
    let level1_time = t1.elapsed();

    let m2: Vec<Expr> = engine
        .macro_operators()
        .iter()
        .map(|op| op.template.clone())
        .collect();
    let m2_canonical: HashSet<String> = engine
        .macro_operators()
        .iter()
        .map(|op| format!("{}", op.template))
        .collect();

    println!("  Level 1 extraction took {:.2}s", level1_time.as_secs_f64());
    println!("  Total conjectures in pool: {}", engine.conjectures.len());
    println!("  |M₂| = {} macros:", m2.len());
    for (i, t) in m2.iter().enumerate() {
        let is_new = !m1_canonical.contains(&format!("{}", t));
        let marker = if is_new { "🆕" } else { "  " };
        println!("    {} {}. {}", marker, i + 1, t);
    }

    let added: Vec<String> = m2_canonical.difference(&m1_canonical).cloned().collect();
    let dropped: Vec<String> = m1_canonical.difference(&m2_canonical).cloned().collect();
    let kept: usize = m1_canonical.intersection(&m2_canonical).count();

    println!();
    println!("  Pool evolution M₁ → M₂:");
    println!("    Kept from M₁: {}", kept);
    println!("    Dropped:      {} {:?}", dropped.len(), dropped);
    println!("    Added (new):  {} {:?}", added.len(), added);

    // ════════════════════════════════════════════════════════════════
    // LEVEL 2: Harder targets — compare cold / M₁ / M₂
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Level 2: Harder targets — 3-way comparison (cold / M₁ / M₂) ━━━");

    // Level 2 targets chosen to be structurally varied
    let level2_targets: Vec<(&str, ObservedSequence)> = vec![
        ("Stefan-Boltzmann (P ∝ T^4)",      observe_stefan_boltzmann(15)),
        ("Kepler 3rd law (T ∝ r^1.5)",      observe_kepler_third_law(15)),
        ("Inverse square (F ∝ 1/r²)",       observe_inverse_square_law(20)),
        ("Quantum HO (E = n + 1/2)",        observe_quantum_harmonic_oscillator(20)),
    ];

    #[allow(clippy::type_complexity)]
    let mut level2_results: Vec<(
        &str,
        ConditionStats, // cold
        ConditionStats, // M₁
        ConditionStats, // M₂
        Vec<RunMetrics>, // raw M₂ runs for macro usage
    )> = Vec::new();

    for (name, target) in &level2_targets {
        println!("\n  Target: {}", name);

        let cold_runs: Vec<RunMetrics> = SEEDS.iter().map(|&s| run_once(target, s, &[], true)).collect();
        let m1_runs: Vec<RunMetrics> = SEEDS.iter().map(|&s| run_once(target, s, &m1, false)).collect();
        let m2_runs: Vec<RunMetrics> = SEEDS.iter().map(|&s| run_once(target, s, &m2, false)).collect();

        let cold_stats = ConditionStats::from_runs("cold", &cold_runs);
        let m1_stats = ConditionStats::from_runs("M₁  ", &m1_runs);
        let m2_stats = ConditionStats::from_runs("M₂  ", &m2_runs);

        println!("    cold: mean_mse={:.3e}  median_mse={:.3e}",
            cold_stats.mean_final_mse, cold_stats.median_final_mse);
        println!("    M₁  : mean_mse={:.3e}  median_mse={:.3e}",
            m1_stats.mean_final_mse, m1_stats.median_final_mse);
        println!("    M₂  : mean_mse={:.3e}  median_mse={:.3e}",
            m2_stats.mean_final_mse, m2_stats.median_final_mse);
        println!("    sample (cold): {}", cold_stats.sample_best_formula);
        println!("    sample (M₂  ): {}", m2_stats.sample_best_formula);

        level2_results.push((name, cold_stats, m1_stats, m2_stats, m2_runs));
    }

    // ════════════════════════════════════════════════════════════════
    // COMPOUNDING VERDICT
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Summary Table: Cold vs M₁ vs M₂ ━━━");
    println!("  {:<32}  {:>12}  {:>12}  {:>12}  {}",
        "target", "cold_mse", "M₁_mse", "M₂_mse", "verdict");
    println!("  {}", "─".repeat(85));

    let mut m2_strict_wins = 0;     // M₂ beats M₁ AND cold
    let mut m1_equivalent = 0;       // M₁ ≈ M₂ (no compounding)
    let mut m2_regressions = 0;      // M₂ worse than M₁ (drift!)
    let mut noise_floor = 0;
    let mut cold_dominates = 0;

    for (name, cold, m1s, m2s, _) in &level2_results {
        let max_mse = cold
            .mean_final_mse
            .max(m1s.mean_final_mse)
            .max(m2s.mean_final_mse);

        let verdict = if max_mse < 1e-6 {
            noise_floor += 1;
            "noise (all ≈ 0)"
        } else if cold.mean_final_mse < m1s.mean_final_mse.min(m2s.mean_final_mse) * 0.95 {
            cold_dominates += 1;
            "cold wins"
        } else if (m2s.mean_final_mse - m1s.mean_final_mse).abs()
            / m1s.mean_final_mse.max(1e-12) < 0.05
        {
            m1_equivalent += 1;
            "≈ M₁ = M₂ (saturation)"
        } else if m2s.mean_final_mse < m1s.mean_final_mse * 0.95 {
            m2_strict_wins += 1;
            "✓ M₂ > M₁ (compound)"
        } else {
            m2_regressions += 1;
            "⚠ M₂ < M₁ (drift!)"
        };

        println!("  {:<32}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {}",
            name, cold.mean_final_mse, m1s.mean_final_mse, m2s.mean_final_mse, verdict);
    }

    println!("\n  Compounding verdict:");
    println!("    Strict compound wins (M₂ > M₁): {} / {}", m2_strict_wins, level2_results.len());
    println!("    Saturation (M₁ ≈ M₂):           {} / {}", m1_equivalent, level2_results.len());
    println!("    Drift (M₂ < M₁):                {} / {}", m2_regressions, level2_results.len());
    println!("    Noise-floor ties:               {} / {}", noise_floor, level2_results.len());
    println!("    Cold dominates (macros hurt):   {} / {}", cold_dominates, level2_results.len());

    println!("\n  Pool growth:");
    println!("    |M₁| = {}", m1.len());
    println!("    |M₂| = {}  (added: {}, dropped: {}, kept: {})",
        m2.len(), added.len(), dropped.len(), kept);

    println!("\n  Honest interpretation:");
    if added.is_empty() {
        println!("    ⚠ SATURATION: M₂ = M₁. Level 1 produced no new extractable");
        println!("      abstractions. Either the normalization is too aggressive,");
        println!("      or Level 1 conjectures don't contain novel recurring shapes.");
        println!("      This is the one-shot ceiling hypothesis confirmed.");
    } else if m2_strict_wins > 0 && m2_regressions == 0 {
        println!("    ✓ COMPOUNDING: M₂ strictly contains new macros ({}), and", added.len());
        println!("      on {} target(s) M₂ beats M₁. The architecture accumulates", m2_strict_wins);
        println!("      useful abstractions across iterations.");
    } else if m2_regressions > 0 && m2_strict_wins == 0 {
        println!("    ⚠ DRIFT: M₂ added {} macros but is WORSE than M₁ on {} target(s).", added.len(), m2_regressions);
        println!("      The new macros are overfit / pollute the population. This is");
        println!("      the extraction-failure mode I was worried about.");
    } else {
        println!("    ± MIXED: M₂ added {} macros. Some targets benefit ({}), some don't.", added.len(), m2_strict_wins);
        println!("      Partial compounding — specific M₁→M₂ pairs transfer.");
    }

    // ════════════════════════════════════════════════════════════════
    // SAFEGUARD: Scan M₂ for overfit constants
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Safeguard: overfit-constant scan of M₂ ━━━");
    println!("  Any Const other than 0, 1, 2 suggests target-specific leakage.");
    let mut suspicious_constants = Vec::new();
    for (i, t) in m2.iter().enumerate() {
        let bad_consts = collect_nonsemantic_constants(t);
        if !bad_consts.is_empty() {
            suspicious_constants.push((i, t.clone(), bad_consts));
        }
    }
    if suspicious_constants.is_empty() {
        println!("  ✓ All M₂ macro templates contain only semantic constants (0, 1, 2).");
        println!("    The normalization pipeline successfully stripped target-specific values.");
    } else {
        println!("  ⚠ Found {} M₂ macro(s) with non-semantic constants:", suspicious_constants.len());
        for (i, t, cs) in &suspicious_constants {
            println!("    {}. {} (constants: {:?})", i + 1, t, cs);
        }
        println!("    This is evidence of overfit leakage into M₂.");
    }
}

/// Collect all constant values in an expression that are NOT semantic
/// (i.e., not 0, 1, or 2). Used to detect overfit leakage.
fn collect_nonsemantic_constants(expr: &Expr) -> Vec<f64> {
    let mut result = Vec::new();
    collect_nonsemantic_inner(expr, &mut result);
    result
}

fn collect_nonsemantic_inner(expr: &Expr, out: &mut Vec<f64>) {
    match expr {
        Expr::Const(c) => {
            let is_semantic = (*c - 0.0).abs() < 1e-12
                || (*c - 1.0).abs() < 1e-12
                || (*c - 2.0).abs() < 1e-12;
            if !is_semantic {
                out.push(*c);
            }
        }
        Expr::Var(_) => {}
        Expr::BinOp(_, l, r) => {
            collect_nonsemantic_inner(l, out);
            collect_nonsemantic_inner(r, out);
        }
        Expr::Func(_, arg) => collect_nonsemantic_inner(arg, out),
        Expr::Sum(body, _) => collect_nonsemantic_inner(body, out),
    }
}
