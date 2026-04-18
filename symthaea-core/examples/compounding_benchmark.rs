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

use symthaea_core::hdc::abstract_thought::macro_quality::{
    evaluate_common_metrics, maybe_enforce, nonsemantic_constants, print_report,
    MacroQualityReport, MacroQualityThresholds,
};
use symthaea_core::hdc::conjecture_engine::{
    observe_balmer_series, observe_fibonacci_ratios, observe_hydrogen_energy_levels,
    observe_inverse_square_law, observe_kepler_third_law, observe_partitions,
    observe_quantum_harmonic_oscillator, observe_relativistic_kinetic_energy,
    observe_stefan_boltzmann, ConjectureEngine, ConjectureStatus, Expr, MathDomain,
    ObservedSequence, RegressorConfig, SeedSpecializationStats, SymbolicRegressor,
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

/// STAGE 1 SPIKE: 1D gravitational-kernel training target.
///
/// `f(n) = 1 / sqrt(n² + 1)` is the restriction of the 2D Newtonian
/// potential `1 / sqrt(x² + y²)` along the line y=1. It presents the
/// target shape — `Func(Sqrt, Add(Pow(Var, 2), Const))` wrapped in a
/// reciprocal — **cleanly** to the symbolic regressor.
///
/// This is the minimal experimental probe for Ceiling B (nonlocal spatial
/// primitives). If the extraction pipeline is going to produce a
/// distance-kernel macro from anywhere, it must produce it from this
/// target — because no other training target in the benchmark contains
/// that shape at all. The test is: after reflect(), does M₁ contain
/// a subtree structurally matching `sqrt((n^2) + C)` or `1/sqrt((n^2) + C)`?
///
/// - If YES: Ceiling B is breakable for 1D distance kernels via the
///   current pipeline. Next question becomes: does this macro propagate
///   across distinct training targets as a reusable abstraction?
/// - If NO: We learn exactly which filter rejects it (GP fit failure,
///   normalization collapse, or occurrence-count threshold), which
///   pins down whether the ceiling lives in SymbolicRegressor or in
///   the MetaHDC extraction logic.
fn distance_kernel_1d_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|i| {
            let n = i as f64;
            (n, 1.0 / (n * n + 1.0).sqrt())
        })
        .collect();
    ObservedSequence::new("distance_kernel_1d(n)", MathDomain::Physics, data)
}

/// STAGE 1 CURRICULUM TRANSFER TEST: unseen distance-kernel variant.
///
/// Target: `f(n) = 1 / sqrt(n² + 4)`. Same structural class as the Level 0
/// training target `1 / sqrt(n² + 1)`, but with a different offset constant.
/// The macro extracted from Level 0 has a generalized `n^placeholder` and
/// (after constant tuning during GP) a placeholder offset — so an M₁-primed
/// run should specialize it to the new constant faster than a cold run
/// that has to assemble `sqrt(Add(Pow, Const))` from leaves.
///
/// Success criterion: `M₁_mse < cold_mse` by a meaningful margin. If the
/// macro-primed run converges to the correct shape faster, curriculum
/// transfer works for the distance-kernel class. If the two tie, the
/// macro is present but not being meaningfully re-specialized (possible
/// sign that the `n^1` canonicalization dropped the exponent information).
fn distance_kernel_variant_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|i| {
            let n = i as f64;
            (n, 1.0 / (n * n + 4.0).sqrt())
        })
        .collect();
    ObservedSequence::new("distance_kernel_variant(n)", MathDomain::Physics, data)
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
    seed_specialization: SeedSpecializationStats,
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
    let seed_specialization = regressor.seed_specialization_stats().clone();

    RunMetrics {
        final_mse,
        final_complexity,
        wall_time_ms: elapsed,
        best_formula: formula_str,
        macro_usage,
        seed_specialization,
    }
}

fn print_macro_pool_snapshot(engine: &ConjectureEngine, label: &str) {
    let Some(metrics) = engine.macro_pool_metrics() else {
        return;
    };

    println!("\n  ▼ Macro pool quality ({label})");
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
        "    usage: active_precision={:.0}%  mature_precision={:.0}%  avg_usage={:.2}  avg_sources={:.2}",
        metrics.active_precision * 100.0,
        metrics.mature_precision * 100.0,
        metrics.avg_usage_count,
        metrics.avg_source_count
    );

    if metrics.signature_stats.is_empty() {
        println!("    top signatures: (none)");
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

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Level Compounding Benchmark                               ║");
    println!("║  Does abstract_thought compound across iterations or saturate?   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Config: {} gens, pop {}, {} seeds per condition",
        GENERATIONS,
        POP_SIZE,
        SEEDS.len()
    );
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
    // Stage 1.5 spike: 1D distance kernel regression probe. A persistent
    // training target that exercises the transcendental-distance primitive
    // path: `f(n) = 1/sqrt(n² + 1)`. With the hierarchical fitness ranking
    // and the sqrt seed template in the Polynomial branch of
    // `build_template_library`, the GP finds the exact fit and the
    // extraction pipeline promotes `1/sqrt((n^k)+c)` shape macros to M₁.
    // If this regresses in future, it means either the seed template was
    // dropped, the hierarchical sort was broken, or normalization collapsed
    // the shape — all of which we want to catch loudly.
    engine.observe(distance_kernel_1d_sequence(25));

    let t0 = Instant::now();
    engine.generate_conjectures(3);
    engine.verify_numerical();
    engine.verify_formal(30); // Enables fast-track promotion via strong verification
    engine.reflect(&prims);
    let level0_time = t0.elapsed();

    // Stage 1.5 regression probe: verify the hierarchical-fitness +
    // sqrt-seed-template fix is still in place. The GP should produce at
    // least one conjecture containing `sqrt` with near-perfect MSE for the
    // distance_kernel_1d target. If the sqrt count is zero, either the
    // template seeding was removed or the Occam fitness is again
    // suppressing perfect fits.
    let (dk_total, dk_sqrt, dk_best_mse) = {
        let mut total = 0usize;
        let mut sqrt_count = 0usize;
        let mut best_mse = f64::MAX;
        for c in engine.conjectures.iter() {
            if c.source == "distance_kernel_1d(n)" {
                total += 1;
                let s = format!("{}", c.formula);
                if s.contains("sqrt") {
                    sqrt_count += 1;
                }
                if c.training_mse < best_mse {
                    best_mse = c.training_mse;
                }
            }
        }
        (total, sqrt_count, best_mse)
    };
    println!(
        "\n  ▼ distance_kernel probe: {} conjectures, {} contain sqrt, best MSE = {:.3e}",
        dk_total, dk_sqrt, dk_best_mse
    );
    if dk_sqrt == 0 {
        println!("    ⚠ REGRESSION: no sqrt-family conjectures — hierarchical fitness or template may have drifted");
    } else {
        println!("    ✓ distance-kernel primitive reachable");
    }

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
    print_macro_pool_snapshot(&engine, "after Level 0 reflection");
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
    engine.observe(observe_partitions(22)); // Hardy-Ramanujan: exp(√n)/n
    engine.observe(observe_balmer_series(12)); // rational: 1/(A - B/n²)
    engine.observe(observe_hydrogen_energy_levels(20)); // -c/n²
    engine.observe(observe_relativistic_kinetic_energy(20)); // transcendental
    engine.observe(damped_exponential_sequence(20)); // exp(-kt)
    engine.observe(logistic_sequence(20)); // 1/(1+exp(-kx))

    let t1 = Instant::now();
    engine.generate_conjectures(3);
    engine.verify_numerical();
    engine.verify_formal(30); // Enables fast-track promotion via strong verification
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

    println!(
        "  Level 1 extraction took {:.2}s",
        level1_time.as_secs_f64()
    );
    println!("  Total conjectures in pool: {}", engine.conjectures.len());

    // Diagnostic: how many of the Level 1 conjectures are strongly verified?
    let mut formally = 0;
    let mut symbolically = 0;
    let mut tested_strong = 0; // test_mse < 1e-6
    let mut tested_weak = 0; // test_mse in [1e-6, 1e-3)
    let mut other = 0;
    let mut best_mse = f64::INFINITY;
    for c in &engine.conjectures {
        match &c.status {
            ConjectureStatus::FormallyVerified { .. } => formally += 1,
            ConjectureStatus::SymbolicallyChecked => symbolically += 1,
            ConjectureStatus::NumericallyTested { test_mse } => {
                if *test_mse < 1e-6 {
                    tested_strong += 1;
                } else if *test_mse < 1e-3 {
                    tested_weak += 1;
                } else {
                    other += 1;
                }
                if *test_mse < best_mse {
                    best_mse = *test_mse;
                }
            }
            _ => other += 1,
        }
    }
    println!("  Verification breakdown:");
    println!("    FormallyVerified:                      {}", formally);
    println!(
        "    SymbolicallyChecked:                   {}",
        symbolically
    );
    println!(
        "    NumericallyTested (test_mse < 1e-6):   {}",
        tested_strong
    );
    println!("    NumericallyTested (1e-6 ≤ mse < 1e-3): {}", tested_weak);
    println!("    Other (Proposed, Refuted, etc.):       {}", other);
    println!(
        "    Strong-eligible for fast-track: {}",
        formally + symbolically + tested_strong
    );
    println!("    Best test_mse observed: {:.3e}", best_mse);
    println!("  |M₂| = {} macros:", m2.len());
    print_macro_pool_snapshot(&engine, "after Level 1 reflection");
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
        ("Stefan-Boltzmann (P ∝ T^4)", observe_stefan_boltzmann(15)),
        ("Kepler 3rd law (T ∝ r^1.5)", observe_kepler_third_law(15)),
        ("Inverse square (F ∝ 1/r²)", observe_inverse_square_law(20)),
        (
            "Quantum HO (E = n + 1/2)",
            observe_quantum_harmonic_oscillator(20),
        ),
        // Stage 1 curriculum transfer probe: different-constant distance
        // kernel. Cold should struggle to assemble `sqrt(Add(Pow, Const))`
        // from leaves; M₁/M₂ should benefit from the promoted distance-
        // kernel macros. Success = M₁_mse materially < cold_mse.
        (
            "Distance kernel variant (1/√(n²+4))",
            distance_kernel_variant_sequence(20),
        ),
    ];

    #[allow(clippy::type_complexity)]
    let mut level2_results: Vec<(
        &str,
        ConditionStats,  // cold
        ConditionStats,  // M₁
        ConditionStats,  // M₂
        Vec<RunMetrics>, // raw M₂ runs for macro usage
    )> = Vec::new();

    for (name, target) in &level2_targets {
        println!("\n  Target: {}", name);

        let cold_runs: Vec<RunMetrics> = SEEDS
            .iter()
            .map(|&s| run_once(target, s, &[], true))
            .collect();
        let m1_runs: Vec<RunMetrics> = SEEDS
            .iter()
            .map(|&s| run_once(target, s, &m1, false))
            .collect();
        let m2_runs: Vec<RunMetrics> = SEEDS
            .iter()
            .map(|&s| run_once(target, s, &m2, false))
            .collect();

        let cold_stats = ConditionStats::from_runs("cold", &cold_runs);
        let m1_stats = ConditionStats::from_runs("M₁  ", &m1_runs);
        let m2_stats = ConditionStats::from_runs("M₂  ", &m2_runs);

        println!(
            "    cold: mean_mse={:.3e}  median_mse={:.3e}",
            cold_stats.mean_final_mse, cold_stats.median_final_mse
        );
        println!(
            "    M₁  : mean_mse={:.3e}  median_mse={:.3e}",
            m1_stats.mean_final_mse, m1_stats.median_final_mse
        );
        println!(
            "    M₂  : mean_mse={:.3e}  median_mse={:.3e}",
            m2_stats.mean_final_mse, m2_stats.median_final_mse
        );
        println!("    sample (cold): {}", cold_stats.sample_best_formula);
        println!("    sample (M₂  ): {}", m2_stats.sample_best_formula);

        level2_results.push((name, cold_stats, m1_stats, m2_stats, m2_runs));
    }

    // ════════════════════════════════════════════════════════════════
    // COMPOUNDING VERDICT
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Summary Table: Cold vs M₁ vs M₂ ━━━");
    println!(
        "  {:<32}  {:>12}  {:>12}  {:>12}  {}",
        "target", "cold_mse", "M₁_mse", "M₂_mse", "verdict"
    );
    println!("  {}", "─".repeat(85));

    let mut m2_strict_wins = 0; // M₂ beats M₁ AND cold
    let mut m1_equivalent = 0; // M₁ ≈ M₂ (no compounding)
    let mut m2_regressions = 0; // M₂ worse than M₁ (drift!)
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
        } else if (m2s.mean_final_mse - m1s.mean_final_mse).abs() / m1s.mean_final_mse.max(1e-12)
            < 0.05
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

        println!(
            "  {:<32}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {}",
            name, cold.mean_final_mse, m1s.mean_final_mse, m2s.mean_final_mse, verdict
        );
    }

    println!("\n  Compounding verdict:");
    println!(
        "    Strict compound wins (M₂ > M₁): {} / {}",
        m2_strict_wins,
        level2_results.len()
    );
    println!(
        "    Saturation (M₁ ≈ M₂):           {} / {}",
        m1_equivalent,
        level2_results.len()
    );
    println!(
        "    Drift (M₂ < M₁):                {} / {}",
        m2_regressions,
        level2_results.len()
    );
    println!(
        "    Noise-floor ties:               {} / {}",
        noise_floor,
        level2_results.len()
    );
    println!(
        "    Cold dominates (macros hurt):   {} / {}",
        cold_dominates,
        level2_results.len()
    );

    println!("\n  Pool growth:");
    println!("    |M₁| = {}", m1.len());
    println!(
        "    |M₂| = {}  (added: {}, dropped: {}, kept: {})",
        m2.len(),
        added.len(),
        dropped.len(),
        kept
    );
    let mut specialization_scored = 0usize;
    let mut specialization_seeded = 0usize;
    let mut specialization_elapsed_ms = 0u128;
    let mut specialization_exact_runs = 0usize;
    for (_, _, _, _, m2_runs) in &level2_results {
        for run in m2_runs {
            specialization_scored += run.seed_specialization.variants_scored;
            specialization_seeded += run.seed_specialization.variants_seeded;
            specialization_elapsed_ms += run.seed_specialization.elapsed_ms;
            if run.seed_specialization.exact_fit_found {
                specialization_exact_runs += 1;
            }
        }
    }
    println!("  Specialization budget use (M₂ runs):");
    println!(
        "    variants_scored={} variants_seeded={} exact_fit_runs={} elapsed={}ms",
        specialization_scored,
        specialization_seeded,
        specialization_exact_runs,
        specialization_elapsed_ms
    );

    println!("\n  Honest interpretation:");
    if added.is_empty() {
        println!("    ⚠ SATURATION: M₂ = M₁. Level 1 produced no new extractable");
        println!("      abstractions. Either the normalization is too aggressive,");
        println!("      or Level 1 conjectures don't contain novel recurring shapes.");
        println!("      This is the one-shot ceiling hypothesis confirmed.");
    } else if m2_strict_wins > 0 && m2_regressions == 0 {
        println!(
            "    ✓ COMPOUNDING: M₂ strictly contains new macros ({}), and",
            added.len()
        );
        println!(
            "      on {} target(s) M₂ beats M₁. The architecture accumulates",
            m2_strict_wins
        );
        println!("      useful abstractions across iterations.");
    } else if m2_regressions > 0 && m2_strict_wins == 0 {
        println!(
            "    ⚠ DRIFT: M₂ added {} macros but is WORSE than M₁ on {} target(s).",
            added.len(),
            m2_regressions
        );
        println!("      The new macros are overfit / pollute the population. This is");
        println!("      the extraction-failure mode I was worried about.");
    } else {
        println!(
            "    ± MIXED: M₂ added {} macros. Some targets benefit ({}), some don't.",
            added.len(),
            m2_strict_wins
        );
        println!("      Partial compounding — specific M₁→M₂ pairs transfer.");
    }

    // ════════════════════════════════════════════════════════════════
    // SAFEGUARD: Scan M₂ for overfit constants
    // ════════════════════════════════════════════════════════════════
    println!("\n━━━ Safeguard: overfit-constant scan of M₂ ━━━");
    println!("  Any non-structural Const other than 0, 1, 2 suggests target-specific leakage.");
    let mut suspicious_constants = Vec::new();
    for (i, t) in m2.iter().enumerate() {
        let bad_consts = nonsemantic_constants(t);
        if !bad_consts.is_empty() {
            suspicious_constants.push((i, t.clone(), bad_consts));
        }
    }
    if suspicious_constants.is_empty() {
        println!("  ✓ All M₂ macro templates contain only semantic constants (0, 1, 2).");
        println!("    The normalization pipeline successfully stripped target-specific values.");
    } else {
        println!(
            "  ⚠ Found {} M₂ macro(s) with non-semantic constants:",
            suspicious_constants.len()
        );
        for (i, t, cs) in &suspicious_constants {
            println!("    {}. {} (constants: {:?})", i + 1, t, cs);
        }
        println!("    This is evidence of overfit leakage into M₂.");
    }

    println!("\n━━━ Macro Pool Quality Gates ━━━");
    let metrics = engine.macro_pool_metrics();
    let mut report = MacroQualityReport::new();
    report.push(
        "distance_kernel_reachable",
        dk_sqrt > 0 && dk_best_mse < 1e-8,
        format!("sqrt_count={} best_mse={:.3e}", dk_sqrt, dk_best_mse),
    );
    for gate in
        evaluate_common_metrics(metrics.as_ref(), &MacroQualityThresholds::one_dimensional()).gates
    {
        report.gates.push(gate);
    }
    report.push(
        "no_nonsemantic_constants",
        suspicious_constants.is_empty(),
        format!("flagged={}", suspicious_constants.len()),
    );
    report.push(
        "no_m2_drift",
        m2_regressions == 0,
        format!("drift_targets={}", m2_regressions),
    );
    report.push(
        "cold_dominance_bounded",
        cold_dominates <= 1,
        format!("cold_dominates={}", cold_dominates),
    );
    print_report(&report, "overall_macro_pool_quality");
    maybe_enforce(&report);
}
