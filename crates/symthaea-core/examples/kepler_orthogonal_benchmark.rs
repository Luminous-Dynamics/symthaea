//! Session 29: two-stage Kepler discovery with gradient orthogonality.
//!
//! Stage 1: run the S27 preset on Kepler with the usual priors.
//! Expected (from S26): 5/5 seeds recover angular momentum L.
//!
//! Stage 2: run the same pipeline, but now pass the S1-discovered L
//! as a `known_invariant` and enable orthogonality penalty.
//! Expected: energy-like expressions (containing `v²` AND `sqrt` AND
//! `x² + y²`) appear in the returned top-10 because L variants are
//! penalized for gradient-parallelism with ∇L.
//!
//! The Session 28 diagnostic proved L dominates because:
//!   - L variance: ~1e-29 (machine epsilon)
//!   - inv_r variance: ~1e-15 (integration error of 1/r)
//!   - 14-order-of-magnitude gap pushes non-L candidates to rank ~46k
//!
//! With orthogonality penalty, every L-like expression (L·π, L+L,
//! exp(L), etc.) has gradient parallel to ∇L in state space and gets
//! fitness multiplied by the penalty factor. This should unshadow
//! the energy/inv_r family.
//!
//! Success criterion: ≥ 1 seed finds an energy-like or 1/r-containing
//! expression in the top-10 in stage 2 (vs 0/5 in stage 1).

use symthaea_core::hdc::conjecture_engine::{
    AutonomousInvariant, BinOp, Expr, RegressorConfig, UnaryFn, compose_top_k_invariants,
    discover_invariants_autonomous_with_seed_templates,
};

const SEEDS: &[u64] = &[42, 1337, 2718, 7919, 31415];
const T_MAX: f64 = 6.0;
const DT: f64 = 0.003;
const POP_SIZE: usize = 300;
const GENERATIONS: usize = 100;

fn kepler_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, vx, vy) = (s[0], s[1], s[2], s[3]);
    let r2 = x * x + y * y;
    if r2 < 1e-12 {
        return vec![vx, vy, 0.0, 0.0];
    }
    let r3 = r2 * r2.sqrt();
    vec![vx, vy, -x / r3, -y / r3]
}

fn var(name: &str) -> Expr {
    Expr::Var(name.into())
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))
}
fn add(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
}
fn sub(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b))
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b))
}
fn pow(a: Expr, e: f64) -> Expr {
    Expr::BinOp(BinOp::Pow, Box::new(a), Box::new(Expr::Const(e)))
}

fn kepler_priors() -> Vec<Expr> {
    let x = || var("x");
    let y = || var("y");
    let vx = || var("vx");
    let vy = || var("vy");
    let r2 = add(pow(x(), 2.0), pow(y(), 2.0));
    let v2 = add(pow(vx(), 2.0), pow(vy(), 2.0));
    let ang_mom = sub(mul(x(), vy()), mul(y(), vx()));
    let inv_r = div(
        Expr::Const(1.0),
        Expr::Func(UnaryFn::Sqrt, Box::new(add(pow(x(), 2.0), pow(y(), 2.0)))),
    );
    vec![ang_mom, r2, v2, inv_r]
}

/// Canonical angular-momentum expression for use as a known_invariant.
fn ang_mom_canonical() -> Expr {
    sub(mul(var("x"), var("vy")), mul(var("y"), var("vx")))
}

fn run_stage1(seed: u64, priors: &[Expr]) -> Vec<AutonomousInvariant> {
    let config = RegressorConfig {
        seed,
        population_size: POP_SIZE,
        generations: GENERATIONS,
        max_depth: 6,
        max_complexity: 24,
        lambda: 0.0005,
        mutation_rate: 0.35,
        ..RegressorConfig::for_autonomous_discovery()
    };
    discover_invariants_autonomous_with_seed_templates(
        kepler_rhs,
        &[1.0, 0.0, 0.0, 1.0],
        &["x", "y", "vx", "vy"],
        None,
        &config,
        T_MAX,
        DT,
        priors,
    )
}

fn run_stage2(seed: u64, priors: &[Expr], known: Vec<Expr>) -> Vec<AutonomousInvariant> {
    run_with_ic(seed, priors, known, [1.0, 0.0, 0.0, 1.0])
}

/// Stage 3: same pipeline as stage 2, but on an ECCENTRIC orbit.
///
/// Initial condition `[1.5, 0, 0, 0.6]` gives a bound elliptical orbit
/// with eccentricity ≈ 0.46 (vs stage 2's e = 0 circular orbit). On a
/// circular orbit, `r` and `v` stay near-constant, so single primitives
/// like `1/r` are themselves near-conserved — no pressure to compose.
/// On an eccentric orbit, `r` varies between perihelion and aphelion,
/// `1/r` varies substantially, and only the true Hamiltonian
/// `E = 0.5v² − 1/r` remains constant by physics.
///
/// If energy appears in stage 3 but not stage 2, the "composition
/// ceiling" identified after S29 was a circular-orbit artifact, not
/// a real mechanism problem.
fn run_stage3(seed: u64, priors: &[Expr], known: Vec<Expr>) -> Vec<AutonomousInvariant> {
    run_with_ic(seed, priors, known, [1.5, 0.0, 0.0, 0.6])
}

fn run_with_ic(
    seed: u64,
    priors: &[Expr],
    known: Vec<Expr>,
    ic: [f64; 4],
) -> Vec<AutonomousInvariant> {
    run_full(seed, priors, known, ic, false)
}

/// Session 30: orthogonality + Lie-derivative fitness.
fn run_stage4(seed: u64, priors: &[Expr], known: Vec<Expr>) -> Vec<AutonomousInvariant> {
    run_full(seed, priors, known, [1.5, 0.0, 0.0, 0.6], true)
}

fn run_full(
    seed: u64,
    priors: &[Expr],
    known: Vec<Expr>,
    ic: [f64; 4],
    lie: bool,
) -> Vec<AutonomousInvariant> {
    let config = RegressorConfig {
        seed,
        population_size: POP_SIZE,
        generations: GENERATIONS,
        max_depth: 6,
        max_complexity: 24,
        lambda: 0.0005,
        mutation_rate: 0.35,
        orthogonality_penalty: 1000.0,
        orthogonality_threshold: 0.9,
        known_invariants: known,
        use_lie_fitness: lie,
        ..RegressorConfig::for_autonomous_discovery()
    };
    discover_invariants_autonomous_with_seed_templates(
        kepler_rhs,
        &ic,
        &["x", "y", "vx", "vy"],
        None,
        &config,
        T_MAX,
        DT,
        priors,
    )
}

fn classify(f: &str) -> (bool, bool, bool) {
    let has_ang_mom = f.contains("(x * vy)") && f.contains("(y * vx)") && f.contains("-");
    let has_energy_like = (f.contains("(vx ^ 2)") || f.contains("(vy ^ 2)"))
        && f.contains("sqrt")
        && (f.contains("(x ^ 2)") || f.contains("(y ^ 2)"));
    let has_inv_r = f.contains("(1 / sqrt") && f.contains("(x ^ 2)") && f.contains("(y ^ 2)");
    (has_ang_mom, has_energy_like, has_inv_r)
}

fn report_stage(label: &str, runs: &[(u64, Vec<AutonomousInvariant>)]) -> (usize, usize, usize) {
    let mut any_ang_mom = 0;
    let mut any_energy = 0;
    let mut any_inv_r = 0;
    for (seed, invs) in runs {
        let mut has_ang_mom = false;
        let mut has_energy = false;
        let mut has_inv_r = false;
        for inv in invs {
            let (am, en, ir) = classify(&inv.formula_str);
            has_ang_mom |= am;
            has_energy |= en;
            has_inv_r |= ir;
        }
        if has_ang_mom {
            any_ang_mom += 1;
        }
        if has_energy {
            any_energy += 1;
        }
        if has_inv_r {
            any_inv_r += 1;
        }
        let best = invs
            .iter()
            .min_by(|a, b| {
                a.variance
                    .partial_cmp(&b.variance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|b| b.clone());
        if let Some(b) = best {
            let (am, en, ir) = classify(&b.formula_str);
            let tag = if en {
                "✓ ENERGY-LIKE"
            } else if am {
                "✓ ang-mom"
            } else if ir {
                "~ 1/r"
            } else {
                "— other"
            };
            println!(
                "  [{:>6}] seed={:<6} best_var={:.3e} pool=[am={} en={} ir={}]  best_tag={}  formula={}",
                label, seed, b.variance, has_ang_mom, has_energy, has_inv_r, tag, b.formula_str
            );
        }
    }
    (any_ang_mom, any_energy, any_inv_r)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Session 29: two-stage Kepler with gradient orthogonality    ║");
    println!("║  Does penalizing ∇L-parallel candidates unshadow energy/1/r? ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "\nConfig: {} seeds, pop={}, r#gen={}, t_max={}, dt={}",
        SEEDS.len(),
        POP_SIZE,
        GENERATIONS,
        T_MAX,
        DT
    );

    let priors = kepler_priors();
    println!("\nPriors used in both stages:");
    for p in &priors {
        println!("  · {}", p);
    }

    println!("\n━━━ Stage 1: S27 preset (no orthogonality penalty) ━━━");
    let stage1_runs: Vec<(u64, Vec<AutonomousInvariant>)> = SEEDS
        .iter()
        .map(|&seed| (seed, run_stage1(seed, &priors)))
        .collect();
    let (s1_am, s1_en, s1_ir) = report_stage("stage1", &stage1_runs);

    println!("\n━━━ Stage 2: orth ON + CIRCULAR orbit [1, 0, 0, 1] ━━━");
    let known = vec![ang_mom_canonical()];
    let stage2_runs: Vec<(u64, Vec<AutonomousInvariant>)> = SEEDS
        .iter()
        .map(|&seed| (seed, run_stage2(seed, &priors, known.clone())))
        .collect();
    let (s2_am, s2_en, s2_ir) = report_stage("stage2", &stage2_runs);

    println!("\n━━━ Stage 3: orth ON + ECCENTRIC orbit [1.5, 0, 0, 0.6], e≈0.46 ━━━");
    println!("  Tests whether 'composition ceiling' is real or circular-orbit artifact.");
    println!("  On circular orbits, 1/r is itself near-conserved (r=const).");
    println!("  On eccentric orbits, 1/r varies — only true energy is near-conserved.");
    let stage3_runs: Vec<(u64, Vec<AutonomousInvariant>)> = SEEDS
        .iter()
        .map(|&seed| (seed, run_stage3(seed, &priors, known.clone())))
        .collect();
    let (s3_am, s3_en, s3_ir) = report_stage("stage3", &stage3_runs);

    println!("\n━━━ Stage 4: orth ON + ECCENTRIC + LIE-DERIVATIVE FITNESS (Session 30) ━━━");
    println!("  Fitness = variance of ∇E·f(state) along trajectory, not variance of E.");
    println!("  Physics-correct: L_f E = 0 identically for true conservation laws.");
    println!("  1D near-constant accidents like y^6 have L_f = 6y⁵·vy, nonzero → rejected.");
    let stage4_runs: Vec<(u64, Vec<AutonomousInvariant>)> = SEEDS
        .iter()
        .map(|&seed| (seed, run_stage4(seed, &priors, known.clone())))
        .collect();
    let (s4_am, s4_en, s4_ir) = report_stage("stage4", &stage4_runs);

    println!("\n━━━ Pool-wide discovery accounting ━━━");
    println!(
        "  Stage 1 — seeds with any ang_mom in top-10:    {} / {}",
        s1_am,
        SEEDS.len()
    );
    println!(
        "  Stage 1 — seeds with any energy-like in top-10: {} / {}",
        s1_en,
        SEEDS.len()
    );
    println!(
        "  Stage 1 — seeds with any 1/r in top-10:        {} / {}",
        s1_ir,
        SEEDS.len()
    );
    println!();
    println!(
        "  Stage 2 — seeds with any ang_mom in top-10:    {} / {}",
        s2_am,
        SEEDS.len()
    );
    println!(
        "  Stage 2 — seeds with any energy-like in top-10: {} / {}",
        s2_en,
        SEEDS.len()
    );
    println!(
        "  Stage 2 — seeds with any 1/r in top-10:        {} / {}",
        s2_ir,
        SEEDS.len()
    );
    println!();
    println!(
        "  Stage 3 — seeds with any ang_mom in top-10:     {} / {}",
        s3_am,
        SEEDS.len()
    );
    println!(
        "  Stage 3 — seeds with any energy-like in top-10: {} / {}",
        s3_en,
        SEEDS.len()
    );
    println!(
        "  Stage 3 — seeds with any 1/r in top-10:         {} / {}",
        s3_ir,
        SEEDS.len()
    );
    println!();
    println!(
        "  Stage 4 — seeds with any ang_mom in top-10:     {} / {}",
        s4_am,
        SEEDS.len()
    );
    println!(
        "  Stage 4 — seeds with any energy-like in top-10: {} / {}",
        s4_en,
        SEEDS.len()
    );
    println!(
        "  Stage 4 — seeds with any 1/r in top-10:         {} / {}",
        s4_ir,
        SEEDS.len()
    );

    println!("\n━━━ Stage 5: compose_top_k postproc on stage 4 invariants (Session 31) ━━━");
    println!("  Exhaustive pairwise composition of top-5 survivors with 4 ops × 6 scales.");
    println!("  Physics target: E = 0.5·v² − 1/r. Success = composite contains both");
    println!("  (vx²+vy²) AND 1/sqrt(x²+y²) substructure with Lie variance < 1e-4.");
    let stage4_ic: [f64; 4] = [1.5, 0.0, 0.0, 0.6];
    let var_names = ["x", "y", "vx", "vy"];
    let mut s5_both = 0usize;
    let mut s5_lowvar = 0usize;
    let mut s5_energy_like = 0usize;
    for (seed, invs) in &stage4_runs {
        let composite =
            compose_top_k_invariants(invs, kepler_rhs, &stage4_ic, &var_names, T_MAX, DT, 5);
        match composite {
            Some(c) => {
                let has_v2 =
                    c.formula_str.contains("(vx ^ 2)") && c.formula_str.contains("(vy ^ 2)");
                let has_inv_r = c.formula_str.contains("(1 / sqrt")
                    && c.formula_str.contains("(x ^ 2)")
                    && c.formula_str.contains("(y ^ 2)");
                let both = has_v2 && has_inv_r;
                let (_am, energy_like, _ir) = classify(&c.formula_str);
                if both {
                    s5_both += 1;
                }
                if c.variance < 1e-4 {
                    s5_lowvar += 1;
                }
                if energy_like {
                    s5_energy_like += 1;
                }
                let tag = if energy_like {
                    "✓ ENERGY-LIKE"
                } else if both {
                    "✓ both subexprs"
                } else {
                    "— other"
                };
                println!(
                    "  [stage5] seed={:<6} lie_var={:.3e} both_v2_and_inv_r={}  tag={}  formula={}",
                    seed, c.variance, both, tag, c.formula_str
                );
            }
            None => {
                println!(
                    "  [stage5] seed={:<6} no pairwise composite beat the single-term baseline",
                    seed
                );
            }
        }
    }
    println!(
        "\n  Stage 5 — composites containing BOTH v² AND 1/sqrt(r²): {} / {}",
        s5_both,
        SEEDS.len()
    );
    println!(
        "  Stage 5 — composites with Lie variance < 1e-4:          {} / {}",
        s5_lowvar,
        SEEDS.len()
    );
    println!(
        "  Stage 5 — composites matching ENERGY-LIKE classifier:   {} / {}",
        s5_energy_like,
        SEEDS.len()
    );

    println!("\n━━━ Verdict ━━━");
    if s5_both >= 3 && s5_lowvar >= 3 {
        println!("  ✓✓ CEILING 4 CLOSED VIA POSTPROC (S31).");
        println!("    Pairwise composition of stage-4 top-K recovers an energy-family");
        println!("    invariant with both v² and 1/r subexpressions in ≥3/5 seeds, at");
        println!("    Lie variance < 1e-4. The arc is closed: GP finds components,");
        println!("    postproc composes them.");
    } else if s5_both >= 1 {
        println!("  ± PARTIAL POSTPROC (S31): some seeds compose v²+1/r, not reliable.");
        println!("    Mechanism is mechanically valid but fragile. Needs either richer");
        println!("    scale grid, op coverage, or upstream discovery improvements.");
    } else if s4_en >= 3 {
        println!("  ✓✓ ENERGY EMERGES WITH LIE-DERIVATIVE FITNESS (no postproc needed).");
    } else if s4_en >= 1 || s4_ir >= 3 {
        println!("  ± PARTIAL ON LIE: some energy/1/r visible with Lie fitness.");
        println!("    Postproc did not find a v²+1/r composite — pieces may not");
        println!("    co-occur in the same seed's top-K.");
    } else if s3_en >= 1 {
        println!("  ± ECCENTRIC SIGNAL: composition partially works on varied orbits.");
    } else {
        println!("  ✗ NO ENERGY: Lie fitness + postproc insufficient. Deeper gap remains.");
    }
}
