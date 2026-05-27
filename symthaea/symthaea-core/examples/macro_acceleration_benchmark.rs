// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Macro Acceleration Benchmark
//!
//! Tests whether macro-operators learned from simple sequences accelerate
//! GP discovery on harder physics targets. This is the empirical answer
//! to "does the abstract_thought feedback loop actually compound?"
//!
//! ## Experimental design
//!
//! 1. **Warmup**: run the engine on simple sequences (squares, cubes, harmonic,
//!    triangular, kinetic energy) to extract macro-operators via the
//!    abstract_thought module. These macros become the "experience" for round 2.
//!
//! 2. **Cold vs Primed**: for each physics target:
//!    - **Cold**: fresh `SymbolicRegressor` with `disable_macro_seeds = true`
//!    - **Primed**: same config but `set_seed_macros()` called with warmup macros
//!    Repeated across 3 RNG seeds to average out noise.
//!
//! 3. **Metrics** (per run):
//!    - `final_mse`: best fitness at last generation
//!    - `gens_to_epsilon`: first generation where fitness < ε (or `None` if never)
//!    - `final_complexity`: AST node count of best formula
//!    - `wall_time_ms`
//!
//! 4. **Output**: table comparing cold vs primed across targets + summary
//!    of which macros (if any) helped most.
//!
//! ## Run
//!
//! ```sh
//! cargo run -p symthaea-core --example macro_acceleration_benchmark \
//!   --features abstract_thought --release
//! ```

use std::time::Instant;

use std::collections::HashMap;

use symthaea_core::hdc::abstract_thought::expr_signature;
use symthaea_core::hdc::abstract_thought::macro_quality::{
    MacroQualityReport, MacroQualityThresholds, evaluate_common_metrics, maybe_enforce,
    print_report,
};
use symthaea_core::hdc::conjecture_engine::{
    ConjectureEngine, Expr, MathDomain, ObservedSequence, RegressorConfig, SeedSpecializationStats,
    SymbolicRegressor, observe_fibonacci_ratios, observe_hydrogen_energy_levels,
    observe_inverse_square_law, observe_kepler_third_law, observe_quantum_harmonic_oscillator,
    observe_relativistic_kinetic_energy, observe_stefan_boltzmann,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

// ═══════════════════════════════════════════════════════════════════════════
// EXPERIMENT CONFIG
// ═══════════════════════════════════════════════════════════════════════════

/// Generation budget per run. Deliberately tight — we want "race to solve"
/// semantics, not "what's your asymptotic MSE". Cold GP is strong enough that
/// with the default 100 generations it crushes most physics targets,
/// leaving no observable window for macro acceleration.
const GENERATIONS: usize = 8;

/// Population size. Smaller than default to further tighten the search.
const POP_SIZE: usize = 100;

/// RNG seeds to average over (5 runs per condition for tighter error bars).
const SEEDS: &[u64] = &[42, 1337, 2718, 7919, 31415];

/// Epsilon for "solved" threshold on RAW MSE (not fitness — fitness includes
/// Occam penalty which makes 1e-3 structurally unreachable).
const EPSILON: f64 = 1e-2;

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLE WARMUP SEQUENCES (from the abstract_thought_demo)
// ═══════════════════════════════════════════════════════════════════════════

fn squares_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n).map(|n| (n as f64, (n * n) as f64)).collect();
    ObservedSequence::new("squares(n)", MathDomain::NumberTheory, data)
}

fn cubes_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, (n * n * n) as f64))
        .collect();
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
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, 0.5 * (n * n) as f64))
        .collect();
    ObservedSequence::new("kinetic_energy(n)", MathDomain::Physics, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct RunMetrics {
    final_mse: f64,
    /// First generation where best fitness fell below EPSILON. None = never.
    gens_to_epsilon: Option<usize>,
    final_complexity: usize,
    wall_time_ms: u128,
    best_formula: String,
    /// Per-macro appearance counts in this run's top-k formulas.
    /// Key: macro canonical string. Value: count across top-k.
    macro_usage: HashMap<String, u64>,
    seed_specialization: SeedSpecializationStats,
}

impl RunMetrics {
    fn from_run(
        fitness_history: &[f64],
        final_mse: f64,
        final_complexity: usize,
        wall_time_ms: u128,
        best_formula: String,
        macro_usage: HashMap<String, u64>,
        seed_specialization: SeedSpecializationStats,
    ) -> Self {
        // Fitness = MSE + lambda * complexity, with lambda=0.001.
        // For ranking "when did we get a good fit" we approximate raw MSE
        // by subtracting a conservative complexity penalty (~max 0.02).
        // This is approximate but adequate for "did we solve it by r#gen K".
        let gens_to_epsilon = fitness_history
            .iter()
            .position(|&f| (f - 0.02).max(0.0) < EPSILON);
        Self {
            final_mse,
            gens_to_epsilon,
            final_complexity,
            wall_time_ms,
            best_formula,
            macro_usage,
            seed_specialization,
        }
    }
}

/// Aggregate stats across multiple runs of the same condition.
#[derive(Debug, Clone)]
struct ConditionStats {
    label: &'static str,
    mean_final_mse: f64,
    median_gens_to_epsilon: Option<f64>,
    solved_count: usize,
    total_runs: usize,
    mean_wall_time_ms: u128,
    mean_complexity: f64,
    sample_formula: String,
}

impl ConditionStats {
    fn from_runs(label: &'static str, runs: &[RunMetrics]) -> Self {
        let n = runs.len() as f64;
        let mean_final_mse = runs.iter().map(|r| r.final_mse).sum::<f64>() / n;
        let solved: Vec<usize> = runs.iter().filter_map(|r| r.gens_to_epsilon).collect();
        let median_gens = if !solved.is_empty() {
            let mut s = solved.clone();
            s.sort_unstable();
            Some(s[s.len() / 2] as f64)
        } else {
            None
        };
        let mean_wall = (runs.iter().map(|r| r.wall_time_ms).sum::<u128>() as f64 / n) as u128;
        let mean_complexity = runs.iter().map(|r| r.final_complexity as f64).sum::<f64>() / n;
        let sample = runs
            .iter()
            .min_by(|a, b| a.final_mse.partial_cmp(&b.final_mse).unwrap())
            .map(|r| r.best_formula.clone())
            .unwrap_or_default();
        Self {
            label,
            mean_final_mse,
            median_gens_to_epsilon: median_gens,
            solved_count: solved.len(),
            total_runs: runs.len(),
            mean_wall_time_ms: mean_wall,
            mean_complexity,
            sample_formula: sample,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ActivationSummary {
    used_macros: usize,
    total_macros: usize,
    total_hits: u64,
    activation_precision: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// RUN HARNESS
// ═══════════════════════════════════════════════════════════════════════════

/// Run `SymbolicRegressor` once on a target and collect metrics.
fn run_once(target: &ObservedSequence, seed: u64, macros: &[Expr], cold: bool) -> RunMetrics {
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
        ..Default::default()
    };
    let mut regressor = SymbolicRegressor::new(config);
    if !cold {
        regressor.set_seed_macros(macros.to_vec());
    }

    let start = Instant::now();
    let results = regressor.fit(target, 3); // top-3 for better macro scan coverage
    let elapsed = start.elapsed().as_millis();

    let best = results.into_iter().next();
    let (final_mse, final_complexity, formula_str) = match best {
        Some(c) => (c.training_mse, c.complexity, c.formula_str),
        None => (f64::INFINITY, 0, String::from("(none)")),
    };

    let macro_usage: HashMap<String, u64> = regressor.macro_usage().clone();
    let seed_specialization = regressor.seed_specialization_stats().clone();

    RunMetrics::from_run(
        regressor.fitness_history(),
        final_mse,
        final_complexity,
        elapsed,
        formula_str,
        macro_usage,
        seed_specialization,
    )
}

fn print_macro_pool_snapshot(engine: &ConjectureEngine, label: &str) {
    let Some(metrics) = engine.macro_pool_metrics() else {
        return;
    };

    println!("\n  Macro pool quality ({label}):");
    println!(
        "    cycle={}  operators={}  candidates={}  promoted={}  pruned={}  survival={:.0}%",
        metrics.cycle,
        metrics.total_operators,
        metrics.total_candidates,
        metrics.total_promoted,
        metrics.total_pruned,
        metrics.survival_rate * 100.0
    );
    println!(
        "    tiers: formal={}  recurrent={}  quarantined={}",
        metrics.formal_operators, metrics.recurrent_operators, metrics.quarantined_operators
    );
    println!(
        "    engine-side usage: active_precision={:.0}%  mature_precision={:.0}%  avg_usage={:.2}  avg_sources={:.2}",
        metrics.active_precision * 100.0,
        metrics.mature_precision * 100.0,
        metrics.avg_usage_count,
        metrics.avg_source_count
    );

    if metrics.signature_stats.is_empty() {
        println!("    signature coverage: (none)");
    } else {
        let mut signature_stats = metrics.signature_stats;
        signature_stats.sort_by(|a, b| {
            b.used_operator_count
                .cmp(&a.used_operator_count)
                .then_with(|| b.total_usage_count.cmp(&a.total_usage_count))
                .then_with(|| b.operator_count.cmp(&a.operator_count))
                .then_with(|| a.signature.cmp(&b.signature))
        });

        println!("    top signatures:");
        for stat in signature_stats.iter().take(5) {
            println!(
                "      · {:<18} ops={} used={} hits={}",
                stat.signature,
                stat.operator_count,
                stat.used_operator_count,
                stat.total_usage_count
            );
        }
    }

    println!("    macro metadata:");
    for op in engine.macro_operators().iter().take(8) {
        let parent = op
            .parent_formulas
            .first()
            .map(|s| s.as_str())
            .unwrap_or("(none)");
        println!(
            "      · {} | sig={} tier={:?} sources={} usage={} arity={} cycle={} parent={}",
            op.template,
            op.signature,
            op.promotion_tier,
            op.source_count,
            op.usage_count,
            op.arity,
            op.created_at,
            parent
        );
    }
}

fn print_benchmark_macro_activation(
    macros: &[Expr],
    all_results: &[(&str, ConditionStats, ConditionStats, Vec<RunMetrics>)],
) -> ActivationSummary {
    let mut usage_by_macro: HashMap<String, u64> =
        macros.iter().map(|expr| (format!("{}", expr), 0)).collect();
    let mut usage_by_signature: HashMap<String, (usize, u64)> = HashMap::new();

    for (_, _, _, primed_runs) in all_results {
        for run in primed_runs {
            for (canonical, count) in &run.macro_usage {
                if let Some(total) = usage_by_macro.get_mut(canonical) {
                    *total += *count;
                }
            }
        }
    }

    let mut ranked_macros: Vec<(String, String, u64)> = macros
        .iter()
        .map(|expr| {
            let canonical = format!("{}", expr);
            let signature = expr_signature(expr);
            let usage = usage_by_macro.get(&canonical).copied().unwrap_or(0);
            (canonical, signature, usage)
        })
        .collect();
    ranked_macros.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    for (_, signature, usage) in &ranked_macros {
        let entry = usage_by_signature
            .entry(signature.clone())
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += *usage;
    }

    let used_macros = ranked_macros
        .iter()
        .filter(|(_, _, usage)| *usage > 0)
        .count();
    let total_usage_count: u64 = ranked_macros.iter().map(|(_, _, usage)| usage).sum();
    let active_precision = if ranked_macros.is_empty() {
        0.0
    } else {
        used_macros as f64 / ranked_macros.len() as f64
    };

    let mut ranked_signatures: Vec<(String, usize, u64)> = usage_by_signature
        .into_iter()
        .map(|(signature, (operator_count, total_usage_count))| {
            (signature, operator_count, total_usage_count)
        })
        .collect();
    ranked_signatures.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("\n  Seeded macro activation across primed benchmark runs:");
    println!(
        "    used_macros={} / {}  activation_precision={:.0}%  total_hits={}",
        used_macros,
        ranked_macros.len(),
        active_precision * 100.0,
        total_usage_count
    );

    println!("    top activated macros:");
    for (canonical, signature, usage) in ranked_macros.iter().take(5) {
        println!("      · {:<18} hits={:<4} {}", signature, usage, canonical);
    }

    if !ranked_signatures.is_empty() {
        println!("    signature hit-rate:");
        for (signature, operator_count, total_usage_count) in ranked_signatures.iter().take(5) {
            println!(
                "      · {:<18} ops={} hits={}",
                signature, operator_count, total_usage_count
            );
        }
    }

    let mut scored = 0usize;
    let mut seeded = 0usize;
    let mut elapsed_ms = 0u128;
    let mut exact = 0usize;
    for (_, _, _, runs) in all_results {
        for run in runs {
            scored += run.seed_specialization.variants_scored;
            seeded += run.seed_specialization.variants_seeded;
            elapsed_ms += run.seed_specialization.elapsed_ms;
            if run.seed_specialization.exact_fit_found {
                exact += 1;
            }
        }
    }
    println!("    specialization budget:");
    println!(
        "      variants_scored={} variants_seeded={} exact_fit_runs={} elapsed={}ms",
        scored, seeded, exact, elapsed_ms
    );

    ActivationSummary {
        used_macros,
        total_macros: ranked_macros.len(),
        total_hits: total_usage_count,
        activation_precision: active_precision,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Macro Acceleration Benchmark                                  ║");
    println!("║  Does abstract_thought's feedback loop actually compound?      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config: {} gens, pop {}, {} seeds per condition, ε < {:.0e}",
        GENERATIONS,
        POP_SIZE,
        SEEDS.len(),
        EPSILON
    );

    // ────────────────────────────────────────────────────────────────
    // PHASE 1: Warmup — extract macros from simple sequences
    // ────────────────────────────────────────────────────────────────
    println!("\n━━━ Phase 1: Warmup (extract macros from simple sequences) ━━━");
    let prims = PrimitiveSystem::new();
    let mut warmup_engine = ConjectureEngine::new();
    warmup_engine.enable_abstract_thought();

    warmup_engine.observe(squares_sequence(25));
    warmup_engine.observe(cubes_sequence(25));
    warmup_engine.observe(triangular_sequence(25));
    warmup_engine.observe(harmonic_sequence(25));
    warmup_engine.observe(kinetic_energy_sequence(25));
    warmup_engine.observe(observe_fibonacci_ratios(25));

    let warmup_start = Instant::now();
    warmup_engine.generate_conjectures(3);
    warmup_engine.verify_numerical();
    warmup_engine.reflect(&prims);
    let warmup_elapsed = warmup_start.elapsed();

    let macros: Vec<Expr> = warmup_engine
        .macro_operators()
        .iter()
        .map(|op| op.template.clone())
        .collect();

    println!("  Warmup ran in {:.2}s", warmup_elapsed.as_secs_f64());
    println!(
        "  Conjectures generated: {}",
        warmup_engine.conjectures.len()
    );
    println!("  Macros extracted: {}", macros.len());
    print_macro_pool_snapshot(&warmup_engine, "after warmup reflection");
    // Print the raw template (what the GP actually sees) — `op.canonical`
    // is further-collapsed for display purposes and can be misleading.
    for op in warmup_engine.macro_operators() {
        println!(
            "    · {}  (canonical label: {}, from {} conjectures)",
            op.template,
            op.canonical,
            op.source_conjectures.len()
        );
    }

    if macros.is_empty() {
        println!("\n  ⚠ No macros extracted — benchmark cannot compare conditions.");
        println!("    Check dynamic_grammar thresholds or warmup sequences.");
        return;
    }

    // ────────────────────────────────────────────────────────────────
    // PHASE 2: Physics targets — cold vs primed
    // ────────────────────────────────────────────────────────────────
    //
    // The last two targets are trig-structured — added specifically to test
    // whether the non-polynomial macros extracted by surgery round 1 (e.g.
    // `sin((1/n))`) actually transfer to function-containing targets.
    // If primed beats cold on these AND the sin/cos macros appear in winning
    // formulas, we have causal evidence that the non-polynomial extraction
    // pipeline produced real value, not just prettier pool counts.

    // Custom target: 1 / (sin(0.1*n) + 2) — has reciprocal-of-sin structure
    // that could benefit from the sin((1/n)) macro via GP recombination.
    let recip_sin_data: Vec<(f64, f64)> = (1..=20)
        .map(|i| {
            let n = i as f64;
            let v = 1.0 / ((0.1 * n).sin() + 2.0);
            (n, v)
        })
        .collect();
    let recip_sin_target =
        ObservedSequence::new("recip_sin(n)", MathDomain::Physics, recip_sin_data);

    // Custom target: simple sine wave sin(0.3*n) — tests whether the sin()
    // primitive, once in the macro pool, accelerates direct sine discovery.
    let sine_wave_data: Vec<(f64, f64)> = (1..=20)
        .map(|i| {
            let n = i as f64;
            (n, (0.3 * n).sin())
        })
        .collect();
    let sine_wave_target =
        ObservedSequence::new("sine_wave(n)", MathDomain::Physics, sine_wave_data);

    let targets: Vec<(&str, ObservedSequence)> = vec![
        ("Kepler 3rd law (T ∝ r^1.5)", observe_kepler_third_law(15)),
        ("Stefan-Boltzmann (P ∝ T^4)", observe_stefan_boltzmann(15)),
        ("Hydrogen E (E ∝ -1/n²)", observe_hydrogen_energy_levels(20)),
        (
            "Quantum HO (E = n + 1/2)",
            observe_quantum_harmonic_oscillator(20),
        ),
        ("Inverse square (F ∝ 1/r²)", observe_inverse_square_law(20)),
        (
            "Relativistic KE (γ − 1)",
            observe_relativistic_kinetic_energy(20),
        ),
        ("Reciprocal sine 1/(sin(0.1n)+2)", recip_sin_target),
        ("Sine wave sin(0.3n)", sine_wave_target),
    ];

    println!(
        "\n━━━ Phase 2: Cold vs Primed on {} physics targets ━━━",
        targets.len()
    );

    // Store raw runs so we can report per-target macro usage later
    let mut all_results: Vec<(&str, ConditionStats, ConditionStats, Vec<RunMetrics>)> = Vec::new();

    for (name, target) in &targets {
        println!("\n  Target: {}", name);

        // Cold runs (no macros)
        let cold_runs: Vec<RunMetrics> = SEEDS
            .iter()
            .map(|&s| run_once(target, s, &[], true))
            .collect();

        // Primed runs (with warmup macros)
        let primed_runs: Vec<RunMetrics> = SEEDS
            .iter()
            .map(|&s| run_once(target, s, &macros, false))
            .collect();

        let cold_stats = ConditionStats::from_runs("cold", &cold_runs);
        let primed_stats = ConditionStats::from_runs("primed", &primed_runs);

        println!(
            "    cold:    mse={:.3e}  solved {}/{}  median_gens={}  wall={}ms",
            cold_stats.mean_final_mse,
            cold_stats.solved_count,
            cold_stats.total_runs,
            cold_stats
                .median_gens_to_epsilon
                .map(|g| format!("{:.0}", g))
                .unwrap_or_else(|| "—".to_string()),
            cold_stats.mean_wall_time_ms
        );
        println!(
            "    primed:  mse={:.3e}  solved {}/{}  median_gens={}  wall={}ms",
            primed_stats.mean_final_mse,
            primed_stats.solved_count,
            primed_stats.total_runs,
            primed_stats
                .median_gens_to_epsilon
                .map(|g| format!("{:.0}", g))
                .unwrap_or_else(|| "—".to_string()),
            primed_stats.mean_wall_time_ms
        );
        println!("    sample (cold):   {}", cold_stats.sample_formula);
        println!("    sample (primed): {}", primed_stats.sample_formula);

        all_results.push((name, cold_stats, primed_stats, primed_runs));
    }

    // ────────────────────────────────────────────────────────────────
    // PHASE 3: Summary table
    // ────────────────────────────────────────────────────────────────
    println!("\n━━━ Summary ━━━");
    println!(
        "  {:<32}  {:>12}  {:>12}  {:>12}",
        "target", "cold_mse", "primed_mse", "Δ_mse"
    );
    println!("  {}", "─".repeat(72));

    // Smarter classification: distinguish "both solved" (uninformative tie)
    // from "both failed" (uninformative tie) from "meaningful primed win"
    // from "noise-floor difference".
    let mut uninformative_ties = 0; // both solved or both garbage at similar MSE
    let mut meaningful_primed_wins = 0; // primed found a meaningfully better fit
    let mut meaningful_cold_wins = 0; // cold found a meaningfully better fit
    let mut noise_floor = 0; // both effectively zero

    for (name, cold, primed, _) in &all_results {
        let delta = primed.mean_final_mse - cold.mean_final_mse;
        let max_mse = cold.mean_final_mse.max(primed.mean_final_mse);

        let (category, marker) = if max_mse < 1e-6 {
            noise_floor += 1;
            ("noise", "  ≈≈")
        } else if delta.abs() / cold.mean_final_mse.max(1e-12) < 0.05 {
            uninformative_ties += 1;
            ("tie", "  ≈ ")
        } else if delta < 0.0 {
            meaningful_primed_wins += 1;
            ("primed+", "  ✓P")
        } else {
            meaningful_cold_wins += 1;
            ("cold+", "  ✗C")
        };
        let _ = category;
        println!(
            "  {:<32}  {:>12.3e}  {:>12.3e}  {:>12.3e} {}",
            name, cold.mean_final_mse, primed.mean_final_mse, delta, marker
        );
    }

    println!("\n  Verdict:");
    println!(
        "    Meaningful primed wins: {} / {}",
        meaningful_primed_wins,
        all_results.len()
    );
    println!(
        "    Meaningful cold wins:   {} / {}",
        meaningful_cold_wins,
        all_results.len()
    );
    println!(
        "    Noise-floor ties (both ≈ 0): {} / {}",
        noise_floor,
        all_results.len()
    );
    println!(
        "    Uninformative ties (similar non-zero):  {} / {}",
        uninformative_ties,
        all_results.len()
    );
    let informative = meaningful_primed_wins + meaningful_cold_wins;
    println!(
        "    Informative comparisons: {} / {}",
        informative,
        all_results.len()
    );

    // Per-condition solved rates
    let cold_solved_total: usize = all_results.iter().map(|(_, c, _, _)| c.solved_count).sum();
    let primed_solved_total: usize = all_results.iter().map(|(_, _, p, _)| p.solved_count).sum();
    let total_runs = all_results.len() * SEEDS.len();
    println!(
        "\n    Solved rate (MSE < {:.0e} within {} gens):",
        EPSILON, GENERATIONS
    );
    println!(
        "      Cold:   {}/{} ({:.0}%)",
        cold_solved_total,
        total_runs,
        100.0 * cold_solved_total as f64 / total_runs as f64
    );
    println!(
        "      Primed: {}/{} ({:.0}%)",
        primed_solved_total,
        total_runs,
        100.0 * primed_solved_total as f64 / total_runs as f64
    );

    println!("\n  Honest interpretation:");
    if informative == 0 {
        println!("    ⚠ No informative comparisons. Every target either solved instantly");
        println!("      (noise-floor tie) or both conditions missed by similar amounts.");
        println!("      Benchmark inconclusive — need harder targets OR tighter budget.");
    } else if meaningful_primed_wins > 0 && meaningful_cold_wins == 0 {
        println!(
            "    ✓ Primed wins on every informative comparison ({}).",
            meaningful_primed_wins
        );
        println!("    The feedback loop compounds: macros from simple warmup sequences");
        println!("    transferred to harder physics targets that the cold GP couldn't");
        println!("    find in the generation budget. This is the hypothesis confirmed.");
    } else if meaningful_cold_wins > 0 && meaningful_primed_wins == 0 {
        println!(
            "    ✗ Cold wins on every informative comparison ({}).",
            meaningful_cold_wins
        );
        println!("    Macro injection HURTS — warmup macros are wasting population");
        println!("    slots on the target domain. The extracted abstractions don't");
        println!("    transfer.");
    } else {
        println!(
            "    ± Mixed results: primed wins {}, cold wins {}.",
            meaningful_primed_wins, meaningful_cold_wins
        );
        println!("    Macros help on some targets (those whose structure overlaps with");
        println!("    warmup patterns) and hurt on others. Partial compounding — the");
        println!("    question becomes WHICH abstractions transfer, not WHETHER they do.");
    }

    // ────────────────────────────────────────────────────────────────
    // PHASE 4: Per-target macro usage — CAUSAL ANALYSIS
    // ────────────────────────────────────────────────────────────────
    // This is the causal upgrade: for each target, we look at which of the
    // warmup-extracted macros actually appeared as subtrees in the primed
    // runs' top-k formulas. If the (1/n) macro dominates hydrogen runs but
    // nothing else, we have a causal link, not just a correlation.
    println!("\n━━━ Macro Usage (Causal Analysis) ━━━");
    println!("  For each target, count how often each warmup macro appeared");
    println!(
        "  as a subtree in the top-3 primed formulas across {} RNG seeds.",
        SEEDS.len()
    );
    println!(
        "  Max possible per macro per target = {} (seeds × top-k).",
        SEEDS.len() * 3
    );
    println!();

    // Collect all unique macro keys used
    let mut macro_keys: Vec<String> = macros.iter().map(|m| format!("{}", m)).collect();
    macro_keys.sort();

    // Print header
    print!("  {:<32}", "target");
    for key in &macro_keys {
        // Truncate long keys for column width
        let short: String = key.chars().take(14).collect();
        print!(" {:>14}", short);
    }
    println!();
    println!("  {}", "─".repeat(32 + macro_keys.len() * 15));

    // Aggregate usage across primed runs per target
    for (name, _cold, _primed, primed_runs) in &all_results {
        // Sum usage counts for each macro across all runs for this target
        let mut target_usage: HashMap<String, u64> = HashMap::new();
        for key in &macro_keys {
            target_usage.insert(key.clone(), 0);
        }
        for run in primed_runs {
            for (key, count) in &run.macro_usage {
                if let Some(v) = target_usage.get_mut(key) {
                    *v += count;
                }
            }
        }
        // Print row
        let short_name: String = name.chars().take(30).collect();
        print!("  {:<32}", short_name);
        for key in &macro_keys {
            let count = target_usage.get(key).copied().unwrap_or(0);
            print!(" {:>14}", count);
        }
        println!();
    }

    println!("\n  Reading this table:");
    println!("    · Non-zero counts = macro subtree appeared in primed top-k formulas");
    println!("    · Zero counts = macro didn't contribute structurally to this target");
    println!("    · If primed beats cold AND non-zero usage on same row → causal link");
    println!("    · If primed beats cold BUT all zeros → primed won for other reasons");

    let activation = print_benchmark_macro_activation(&macros, &all_results);

    println!("\n━━━ Macro Pool Quality Gates ━━━");
    let metrics = warmup_engine.macro_pool_metrics();
    let mut report = MacroQualityReport::new();
    report.push(
        "macros_extracted",
        !macros.is_empty(),
        format!("macros={}", macros.len()),
    );
    for gate in
        evaluate_common_metrics(metrics.as_ref(), &MacroQualityThresholds::one_dimensional()).gates
    {
        if gate.name != "has_used_macros" {
            report.gates.push(gate);
        }
    }
    report.push(
        "activation_precision_min",
        activation.activation_precision >= 0.50,
        format!(
            "used={}/{} hits={} precision={:.0}%",
            activation.used_macros,
            activation.total_macros,
            activation.total_hits,
            activation.activation_precision * 100.0
        ),
    );
    report.push(
        "primed_not_worse_overall",
        meaningful_primed_wins >= meaningful_cold_wins,
        format!(
            "primed_wins={} cold_wins={}",
            meaningful_primed_wins, meaningful_cold_wins
        ),
    );
    report.push(
        "cold_dominance_bounded",
        meaningful_cold_wins <= 1,
        format!("cold_wins={}", meaningful_cold_wins),
    );
    print_report(&report, "overall_macro_acceleration_quality");
    maybe_enforce(&report);
}
