//! Session 18: does Kepler macro priming accelerate PCR3BP autonomous
//! discovery?
//!
//! A/B benchmark over N seeds: runs `discover_invariants_autonomous_with_seed_templates`
//! on the Planar Circular Restricted Three-Body Problem (PCR3BP) in two conditions:
//!
//! - **cold**: no `extra_seed_templates`
//! - **primed**: seeded with Kepler-derived macros (angular momentum
//!   `x*vy - y*vx`, squared radius `x²+y²`, squared speed `vx²+vy²`,
//!   inverse distance `1/sqrt(x²+y²)`)
//!
//! For each seed we record the best (lowest-variance) invariant the GP
//! returns. We then compare the two distributions.
//!
//! Context: Session 16's curriculum probe already locked in that the
//! bridge *propagates* macros across two autonomous runs on the same
//! engine (commit `e175630f4b`). What it didn't show is whether
//! priming actually *accelerates* downstream discovery. This binary
//! measures that effect.
//!
//! Expected failure modes:
//! - **Kepler priors don't help PCR3BP**: PCR3BP's Jacobi integral
//!   has structure (`(x²+y²) − (vx²+vy²) + 2·[(1-μ)/r₁ + μ/r₂]`) that
//!   the Kepler macros only partially match — `(x²+y²)` appears
//!   directly, `(vx²+vy²)` appears with opposite sign, but
//!   `(1-μ)/r₁ + μ/r₂` is genuinely out-of-distribution.
//! - **Noise dominates**: with only 5 seeds, any effect smaller than
//!   ~2× in variance is hard to distinguish from random fluctuation.
//!
//! This is a deliberate smoke-scale benchmark, not a publication-grade
//! result. If it shows clear signal, Session 19 can scale it up.

use symthaea_core::hdc::conjecture_engine::{
    discover_invariants_autonomous_with_seed_templates, AutonomousInvariant, BinOp, Expr,
    RegressorConfig, UnaryFn,
};

const SEEDS: &[u64] = &[42, 1337, 2718, 7919, 31415];
const T_MAX: f64 = 6.0;
const DT: f64 = 0.003;
const MU: f64 = 0.01215; // Earth-Moon mass ratio
const POP_SIZE: usize = 120;
const GENERATIONS: usize = 30;

fn pcr3bp_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, vx, vy) = (s[0], s[1], s[2], s[3]);
    let dx1 = x + MU;
    let dx2 = x - 1.0 + MU;
    let r1_sq = dx1 * dx1 + y * y;
    let r2_sq = dx2 * dx2 + y * y;
    if r1_sq < 1e-12 || r2_sq < 1e-12 {
        return vec![vx, vy, 0.0, 0.0];
    }
    let r1_3 = r1_sq * r1_sq.sqrt();
    let r2_3 = r2_sq * r2_sq.sqrt();
    let ax = 2.0 * vy + x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3;
    let ay = -2.0 * vx + y - (1.0 - MU) * y / r1_3 - MU * y / r2_3;
    vec![vx, vy, ax, ay]
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
    let r2 = || add(pow(x(), 2.0), pow(y(), 2.0));
    let v2 = || add(pow(vx(), 2.0), pow(vy(), 2.0));
    let ang_mom = sub(mul(x(), vy()), mul(y(), vx()));
    let inv_r = div(
        Expr::Const(1.0),
        Expr::Func(UnaryFn::Sqrt, Box::new(r2())),
    );
    vec![ang_mom, r2(), v2(), inv_r]
}

/// "Cheat" priors — the exact Jacobi-integral subtrees for PCR3BP.
///
/// The Jacobi integral is `C_J = (x²+y²) + 2·[(1-μ)/r₁ + μ/r₂] − (vx²+vy²)`
/// where `r₁ = sqrt((x+μ)²+y²)` and `r₂ = sqrt((x-1+μ)²+y²)`.
///
/// We hand it the displaced-origin distance kernels explicitly. If the
/// GP can't assemble the full Jacobi integral even with these priors,
/// the crossover/fitness machinery itself — not priming content — is
/// the Ceiling-4 bottleneck.
fn jacobi_cheat_priors() -> Vec<Expr> {
    let mu = Expr::Const(MU);
    let one_minus_mu = Expr::Const(1.0 - MU);
    let mu_minus_one = Expr::Const(MU - 1.0);
    let x = || var("x");
    let y = || var("y");
    let vx = || var("vx");
    let vy = || var("vy");
    let r2_local = || add(pow(x(), 2.0), pow(y(), 2.0));
    let v2_local = || add(pow(vx(), 2.0), pow(vy(), 2.0));
    // r₁² = (x+μ)² + y²
    let r1_sq = || add(pow(add(x(), mu.clone()), 2.0), pow(y(), 2.0));
    // r₂² = (x+μ-1)² + y²    = (x - (1-μ))² + y²
    let r2_sq = || add(pow(add(x(), mu_minus_one.clone()), 2.0), pow(y(), 2.0));
    let inv_r1 = div(
        Expr::Const(1.0),
        Expr::Func(UnaryFn::Sqrt, Box::new(r1_sq())),
    );
    let inv_r2 = div(
        Expr::Const(1.0),
        Expr::Func(UnaryFn::Sqrt, Box::new(r2_sq())),
    );
    // The nonlocal combination as a single prior — removes the
    // requirement that crossover assemble the two terms from scratch.
    let nonlocal = add(
        mul(one_minus_mu.clone(), inv_r1.clone()),
        mul(mu.clone(), inv_r2.clone()),
    );
    // The local quasi-Jacobi skeleton `(x²+y²) − (vx²+vy²)` — exact
    // form (no factor of 2 on the nonlocal yet; that's a constant the
    // constant-optimizer can find).
    let quasi_local = sub(r2_local(), v2_local());
    vec![
        inv_r1,
        inv_r2,
        nonlocal,
        quasi_local,
        r2_local(),
        v2_local(),
    ]
}

fn run_one(seed: u64, priors: &[Expr], exclude_trig: bool) -> Vec<AutonomousInvariant> {
    let config = RegressorConfig {
        seed,
        population_size: POP_SIZE,
        generations: GENERATIONS,
        max_depth: 5,
        max_complexity: 18,
        lambda: 0.0005,
        mutation_rate: 0.35,
        exclude_trig,
        ..RegressorConfig::default()
    };
    discover_invariants_autonomous_with_seed_templates(
        pcr3bp_rhs,
        &[0.8, 0.1, 0.05, 0.3],
        &["x", "y", "vx", "vy"],
        None,
        &config,
        T_MAX,
        DT,
        priors,
    )
}

#[derive(Debug, Clone)]
struct SeedResult {
    seed: u64,
    best_variance: f64,
    best_complexity: usize,
    best_formula: String,
    count_nontrivial: usize,
}

fn summarize(label: &str, seed: u64, results: &[AutonomousInvariant]) -> SeedResult {
    let best = results
        .iter()
        .min_by(|a, b| a.variance.partial_cmp(&b.variance).unwrap_or(std::cmp::Ordering::Equal));
    let count_nontrivial = results
        .iter()
        .filter(|inv| inv.variance.is_finite() && inv.variance < 1e-2 && inv.complexity >= 3)
        .count();
    let (variance, complexity, formula) = match best {
        Some(b) => (b.variance, b.complexity, b.formula_str.clone()),
        None => (f64::INFINITY, 0, "<none>".to_string()),
    };
    println!(
        "  [{:>6}] seed={:<6} best_variance={:.3e} complexity={} nontrivial={} formula={}",
        label, seed, variance, complexity, count_nontrivial, formula
    );
    SeedResult {
        seed,
        best_variance: variance,
        best_complexity: complexity,
        best_formula: formula,
        count_nontrivial,
    }
}

fn aggregate(label: &str, runs: &[SeedResult]) {
    let finite_vars: Vec<f64> = runs
        .iter()
        .filter(|r| r.best_variance.is_finite())
        .map(|r| r.best_variance)
        .collect();
    let mean_var = finite_vars.iter().sum::<f64>() / finite_vars.len().max(1) as f64;
    let median_var = {
        let mut v = finite_vars.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.get(v.len() / 2).copied().unwrap_or(f64::INFINITY)
    };
    let mean_cplx =
        runs.iter().map(|r| r.best_complexity as f64).sum::<f64>() / runs.len().max(1) as f64;
    let total_nontrivial: usize = runs.iter().map(|r| r.count_nontrivial).sum();
    println!(
        "  {} summary: mean_var={:.3e}  median_var={:.3e}  mean_cplx={:.1}  total_nontrivial={}",
        label, mean_var, median_var, mean_cplx, total_nontrivial
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Session 18: PCR3BP priming benchmark                        ║");
    println!("║  Does Kepler-derived macro priming accelerate autonomous     ║");
    println!("║  discovery on PCR3BP?                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "\nConfig: {} seeds, pop={}, gen={}, t_max={}, dt={}\n",
        SEEDS.len(),
        POP_SIZE,
        GENERATIONS,
        T_MAX,
        DT
    );

    let priors = kepler_priors();
    let cheat = jacobi_cheat_priors();
    println!("Kepler priors used in primed condition:");
    for p in &priors {
        println!("  · {}", p);
    }
    println!("\nJacobi cheat priors (Session 20):");
    for p in &cheat {
        println!("  · {}", p);
    }
    println!();

    println!("━━━ Cold (no priors, trig allowed) ━━━");
    let cold: Vec<SeedResult> = SEEDS
        .iter()
        .map(|&seed| summarize("cold", seed, &run_one(seed, &[], false)))
        .collect();
    aggregate("cold            ", &cold);

    println!("\n━━━ Primed (Kepler priors, trig allowed) ━━━");
    let primed: Vec<SeedResult> = SEEDS
        .iter()
        .map(|&seed| summarize("primed", seed, &run_one(seed, &priors, false)))
        .collect();
    aggregate("primed          ", &primed);

    println!("\n━━━ Cold + no-trig (Session 19 ablation) ━━━");
    let cold_notrig: Vec<SeedResult> = SEEDS
        .iter()
        .map(|&seed| summarize("cold-NT", seed, &run_one(seed, &[], true)))
        .collect();
    aggregate("cold + no-trig  ", &cold_notrig);

    println!("\n━━━ Primed + no-trig (Session 19 ablation) ━━━");
    let primed_notrig: Vec<SeedResult> = SEEDS
        .iter()
        .map(|&seed| summarize("prim-NT", seed, &run_one(seed, &priors, true)))
        .collect();
    aggregate("primed + no-trig", &primed_notrig);

    println!("\n━━━ Jacobi cheat priors + no-trig (Session 20 decisive test) ━━━");
    let cheat_notrig: Vec<SeedResult> = SEEDS
        .iter()
        .map(|&seed| summarize("cheat-NT", seed, &run_one(seed, &cheat, true)))
        .collect();
    aggregate("cheat + no-trig ", &cheat_notrig);

    println!("\n━━━ Head-to-head per seed (primed vs cold, trig allowed) ━━━");
    head_to_head("cold vs primed", &cold, &primed);

    println!(
        "\n━━━ Head-to-head per seed (primed vs cold, trig DISABLED — Session 19) ━━━"
    );
    head_to_head("cold-NT vs primed-NT", &cold_notrig, &primed_notrig);

    println!("\n━━━ Head-to-head per seed (cheat vs cold, trig DISABLED — Session 20) ━━━");
    head_to_head("cold-NT vs cheat-NT", &cold_notrig, &cheat_notrig);

    println!("\n━━━ Degeneracy accounting: formulas containing cos/sin ━━━");
    for (label, runs) in [
        ("cold           ", &cold),
        ("primed         ", &primed),
        ("cold+no-trig   ", &cold_notrig),
        ("primed+no-trig ", &primed_notrig),
        ("cheat+no-trig  ", &cheat_notrig),
    ] {
        let trig_count = runs
            .iter()
            .filter(|r| r.best_formula.contains("cos") || r.best_formula.contains("sin"))
            .count();
        println!(
            "  {}: {}/{} best-formula trig-degenerate",
            label,
            trig_count,
            runs.len()
        );
    }

    println!("\n━━━ Jacobi-family accounting: formulas containing displaced sqrt kernels ━━━");
    // Looking for any trace of the injected `1/sqrt((x+MU)²+y²)` or
    // `1/sqrt((x+MU-1)²+y²)` shapes. If zero appear in `cheat-NT` even
    // after we seeded them directly, Ceiling 4 is mechanism-limited,
    // not priming-limited.
    for (label, runs) in [
        ("cold+no-trig   ", &cold_notrig),
        ("primed+no-trig ", &primed_notrig),
        ("cheat+no-trig  ", &cheat_notrig),
    ] {
        let jacobi_count = runs
            .iter()
            .filter(|r| {
                let f = &r.best_formula;
                f.contains("sqrt") && (f.contains("x + 0.01215") || f.contains("x + -0.987"))
            })
            .count();
        println!(
            "  {}: {}/{} best-formula contains displaced-origin sqrt",
            label,
            jacobi_count,
            runs.len()
        );
    }
}

fn head_to_head(label: &str, a: &[SeedResult], b: &[SeedResult]) {
    let mut b_wins = 0;
    let mut a_wins = 0;
    let mut ties = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.seed, y.seed);
        let ratio = y.best_variance / x.best_variance.max(1e-30);
        let verdict = if y.best_variance < x.best_variance * 0.5 {
            b_wins += 1;
            "✓ B wins (>2× better)"
        } else if x.best_variance < y.best_variance * 0.5 {
            a_wins += 1;
            "✗ A wins (>2× better)"
        } else {
            ties += 1;
            "— effectively tied"
        };
        println!(
            "  seed={:<6} A={:.2e}  B={:.2e}  ratio={:.2}  {}",
            x.seed, x.best_variance, y.best_variance, ratio, verdict
        );
    }
    println!(
        "  {}: A_wins={} B_wins={} ties={} / {}",
        label,
        a_wins,
        b_wins,
        ties,
        a.len()
    );
}
