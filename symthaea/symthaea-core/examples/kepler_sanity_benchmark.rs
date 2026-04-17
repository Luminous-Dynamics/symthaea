//! Session 26: control experiment for the 11-session Ceiling-4 arc.
//!
//! Runs the Session-25 "best" pipeline (diverse-5 fitness + pinned
//! priors + prior-composition-0.15 + fragment-bonus-0.5) on Kepler
//! instead of PCR3BP. Kepler is the *friendly* case — a textbook
//! 4D system with two known conservation laws:
//!
//!   Angular momentum: L = x·vy − y·vx
//!   Energy:            E = ½·(vx² + vy²) − 1/r       where r = √(x²+y²)
//!
//! We seed the GP with the exact pieces — `ang_mom`, `(x²+y²)`,
//! `(vx²+vy²)`, `1/sqrt(x²+y²)` — and check whether the pipeline
//! recovers `E` via composition (0.5·v² combined with −1/r) or at
//! minimum preserves the seeded `ang_mom` as the best invariant.
//!
//! Expected outcomes:
//! - **Best case**: the GP composes 0.5·v² − 1/r into the energy
//!   expression. Then we've demonstrated the S24-S25 machinery
//!   closes the in-distribution case and PCR3BP's failure is a
//!   problem-difficulty result, not a mechanism flaw.
//! - **Passable case**: ang_mom survives as best invariant with
//!   near-zero variance. Mechanism is fine; composition still needs
//!   work but in-distribution priming is preserved.
//! - **Failure case**: neither energy nor ang_mom survive. The
//!   machinery has a deeper flaw worth investigating.

use symthaea_core::hdc::conjecture_engine::{
    discover_invariants_autonomous_with_seed_templates, AutonomousInvariant, BinOp, Expr,
    RegressorConfig, UnaryFn,
};

const SEEDS: &[u64] = &[42, 1337, 2718, 7919, 31415];
const T_MAX: f64 = 6.0;
const DT: f64 = 0.003;
const POP_SIZE: usize = 300;
const GENERATIONS: usize = 100;

/// Kepler 4D: planet orbiting central body at origin, unit GM.
///   ẍ = -x / r³, ÿ = -y / r³
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

fn run_kepler(seed: u64, priors: &[Expr]) -> Vec<AutonomousInvariant> {
    // Session 27: uses the `for_autonomous_discovery()` preset that
    // bundles the S19 (exclude_trig), S21 (diverse_trajectory),
    // S24 (prior_composition), and S25 (prior_fragment_bonus)
    // defaults validated by this very benchmark. Callers override
    // only problem-specific fields (pop, gens, depth, complexity).
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
        &[1.0, 0.0, 0.0, 1.0], // circular-ish initial condition
        &["x", "y", "vx", "vy"],
        None,
        &config,
        T_MAX,
        DT,
        priors,
    )
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Session 26: Kepler sanity benchmark (S25 pipeline on easy)  ║");
    println!("║  Does the full arc's machinery recover Kepler's invariants?  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "\nConfig: {} seeds, pop={}, gen={}, t_max={}, dt={}",
        SEEDS.len(),
        POP_SIZE,
        GENERATIONS,
        T_MAX,
        DT
    );
    println!("Pipeline: diverse-5 + pinned + composition-0.15 + fragment-bonus-0.5 + no-trig\n");

    let priors = kepler_priors();
    println!("Kepler priors used:");
    for p in &priors {
        println!("  · {}", p);
    }
    println!();

    println!("━━━ Best per seed ━━━");
    let mut found_ang_mom = 0;
    let mut found_energy_like = 0;
    let mut found_inv_r = 0;
    for &seed in SEEDS {
        let invs = run_kepler(seed, &priors);
        let best = invs
            .iter()
            .min_by(|a, b| {
                a.variance
                    .partial_cmp(&b.variance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|b| b.clone());

        match best {
            Some(b) => {
                let f = &b.formula_str;
                let has_ang_mom =
                    f.contains("(x * vy)") && f.contains("(y * vx)") && f.contains("-");
                let has_energy_like = (f.contains("(vx ^ 2)") || f.contains("(vy ^ 2)"))
                    && f.contains("sqrt")
                    && (f.contains("(x ^ 2)") || f.contains("(y ^ 2)"));
                let has_inv_r =
                    f.contains("(1 / sqrt") && f.contains("(x ^ 2)") && f.contains("(y ^ 2)");
                if has_ang_mom {
                    found_ang_mom += 1;
                }
                if has_energy_like {
                    found_energy_like += 1;
                }
                if has_inv_r {
                    found_inv_r += 1;
                }
                let tag = if has_energy_like {
                    "✓ ENERGY-LIKE"
                } else if has_ang_mom {
                    "✓ angular momentum"
                } else if has_inv_r {
                    "~ partial (1/r)"
                } else {
                    "— other"
                };
                println!(
                    "  seed={:<6} var={:.3e} cplx={:<2} {}  formula={}",
                    seed, b.variance, b.complexity, tag, f
                );
            }
            None => {
                println!("  seed={:<6} <no invariants returned>", seed);
            }
        }
    }

    println!("\n━━━ Summary ━━━");
    println!(
        "  Angular momentum recognized (x*vy - y*vx):  {} / {}",
        found_ang_mom,
        SEEDS.len()
    );
    println!(
        "  Energy-like (v² AND sqrt AND r²):           {} / {}",
        found_energy_like,
        SEEDS.len()
    );
    println!(
        "  Inverse-r containing (1/sqrt + x²+y²):      {} / {}",
        found_inv_r,
        SEEDS.len()
    );

    println!("\n━━━ Verdict ━━━");
    if found_energy_like >= 3 {
        println!("  ✓ PIPELINE VALIDATED: the full arc's machinery recovers Kepler energy.");
        println!("    PCR3BP failure is a problem-difficulty result, not a mechanism flaw.");
    } else if found_ang_mom >= 3 || found_inv_r >= 3 {
        println!("  ✓ PARTIAL VALIDATION: Kepler priors preserved, composition limited.");
        println!("    In-distribution priming works; full composition still needs S27+.");
    } else {
        println!("  ✗ PIPELINE FLAW: Kepler invariants not recovered even in-distribution.");
        println!("    Investigate fitness/selection chain before scaling to harder problems.");
    }
}
