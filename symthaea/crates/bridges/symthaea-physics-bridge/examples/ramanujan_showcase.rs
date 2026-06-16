// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
//! # Ramanujan Protocol — Discovery Showcase
//!
//! End-to-end demonstration of the Automated Physicist:
//!
//!   ODE → autonomous discovery → symbolic proof → physics catalog recognition →
//!         paper-ready LaTeX table
//!
//! This example composes Tier 1 (Z3 bridge, LaTeX converter, physics recognition)
//! and Tier 2 (catalog expansion, discovery report generator) into a single
//! deliverable: the headline figure for the Ramanujan Protocol paper.
//!
//! Run with:
//! ```bash
//! cargo run -p symthaea-physics-bridge --example ramanujan_showcase --release
//! ```
//!
//! Output:
//! - Live discovery progress for each canonical problem
//! - Per-problem recognition annotation against the 128-equation catalog
//! - Per-problem Z3 verification status
//! - One combined LaTeX table ready to paste into papers/latex/ramanujan-protocol/

use symthaea_core::hdc::conjecture_engine::{
    AutonomousInvariant, ConjectureEngine, ConjectureStatus, MathDomain, ObservedSequence,
    RegressorConfig, SymExpr, discover_invariants_autonomous, expr_to_latex,
};
use symthaea_physics_bridge::{
    DimensionalSignature, PhysicsSearchEngine, UnitMap, recognize_expr, recognize_expr_with_units,
};

// ───────────────────────────────────────────────────────────────────────────
// CANONICAL PROBLEMS — the autonomous physicist's test suite
// ───────────────────────────────────────────────────────────────────────────

/// Harmonic oscillator: dx/dt = v, dv/dt = -x
/// Conserved: E = x² + v²
fn harmonic_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0]]
}

fn harmonic_dynamics() -> Vec<(&'static str, SymExpr)> {
    vec![
        ("x", SymExpr::Var("v".into())),
        ("v", SymExpr::Neg(Box::new(SymExpr::Var("x".into())))),
    ]
}

/// Lotka-Volterra: dx/dt = x(1-y), dy/dt = y(x-1)
/// Conserved: V = x - ln(x) + y - ln(y)  (transcendental!)
fn lotka_volterra_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y) = (s[0], s[1]);
    vec![x * (1.0 - y), y * (x - 1.0)]
}

fn lotka_volterra_dynamics() -> Vec<(&'static str, SymExpr)> {
    vec![
        (
            "x",
            SymExpr::Add(
                Box::new(SymExpr::Var("x".into())),
                Box::new(SymExpr::Neg(Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Var("x".into())),
                    Box::new(SymExpr::Var("y".into())),
                )))),
            ),
        ),
        (
            "y",
            SymExpr::Add(
                Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Var("x".into())),
                    Box::new(SymExpr::Var("y".into())),
                )),
                Box::new(SymExpr::Neg(Box::new(SymExpr::Var("y".into())))),
            ),
        ),
    ]
}

/// Kepler two-body (4D): [x, y, vx, vy]
/// dx/dt = vx, dy/dt = vy
/// dvx/dt = -x/r³, dvy/dt = -y/r³  where r = √(x²+y²)
/// Conserved: E = ½(vx² + vy²) - 1/r AND L = x·vy - y·vx
fn kepler_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, vx, vy) = (s[0], s[1], s[2], s[3]);
    let r2 = x * x + y * y;
    let r3 = r2 * r2.sqrt();
    if r3 < 1e-15 {
        return vec![vx, vy, 0.0, 0.0];
    }
    vec![vx, vy, -x / r3, -y / r3]
}

fn kepler_dynamics() -> Vec<(&'static str, SymExpr)> {
    let r2 = || {
        SymExpr::Add(
            Box::new(SymExpr::Pow(Box::new(SymExpr::Var("x".into())), 2.0)),
            Box::new(SymExpr::Pow(Box::new(SymExpr::Var("y".into())), 2.0)),
        )
    };
    vec![
        ("x", SymExpr::Var("vx".into())),
        ("y", SymExpr::Var("vy".into())),
        (
            "vx",
            SymExpr::Mul(
                Box::new(SymExpr::Neg(Box::new(SymExpr::Var("x".into())))),
                Box::new(SymExpr::Pow(Box::new(r2()), -1.5)),
            ),
        ),
        (
            "vy",
            SymExpr::Mul(
                Box::new(SymExpr::Neg(Box::new(SymExpr::Var("y".into())))),
                Box::new(SymExpr::Pow(Box::new(r2()), -1.5)),
            ),
        ),
    ]
}

/// "Mystery ODE" — a coupled anisotropic harmonic oscillator.
///
/// State: `[x, y, px, py]`. Conservative Hamiltonian system with an off-
/// diagonal quadratic coupling in position space:
///     H = ½(px² + py²) + x² + y² + xy
/// The coupling matrix `[[2, 1], [1, 2]]` has eigenvalues {3, 1}, so the
/// origin is a stable elliptic equilibrium and trajectories stay bounded
/// on level sets of H.
///
/// This system is "novel-ish": no physics catalog contains exactly this
/// form, and the `xy` bilinear coupling is NOT a canonical sum-of-squares,
/// so the GP cannot assemble it from position templates alone. It lives
/// just beyond the pre-Move-11 template library — a designer-constructed
/// stretch test for the coupled-oscillator template added in Move 11.
fn mystery_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, px, py) = (s[0], s[1], s[2], s[3]);
    vec![px, py, -2.0 * x - y, -x - 2.0 * y]
}

fn mystery_dynamics() -> Vec<(&'static str, SymExpr)> {
    let var = |n: &str| SymExpr::Var(n.into());
    let mul = |a: SymExpr, b: SymExpr| SymExpr::Mul(Box::new(a), Box::new(b));
    let add = |a: SymExpr, b: SymExpr| SymExpr::Add(Box::new(a), Box::new(b));
    let neg = |a: SymExpr| SymExpr::Neg(Box::new(a));
    vec![
        ("x", var("px")),
        ("y", var("py")),
        // dpx/dt = -2x - y
        (
            "px",
            add(neg(mul(SymExpr::Const(2.0), var("x"))), neg(var("y"))),
        ),
        // dpy/dt = -x - 2y
        (
            "py",
            add(neg(var("x")), neg(mul(SymExpr::Const(2.0), var("y")))),
        ),
    ]
}

/// Planar Circular Restricted Three-Body Problem (PCR3BP) in a rotating frame.
///
/// State: `[x, y, vx, vy]`. Mass ratio μ = m₂/(m₁+m₂). The two primaries
/// sit at fixed positions (-μ, 0) and (1-μ, 0) in the co-rotating frame;
/// the test particle moves under their gravity PLUS Coriolis and centrifugal
/// forces from the rotation.
///
/// The equations of motion (unit angular velocity Ω = 1, G(m₁+m₂) = 1):
///   dx/dt  = vx
///   dy/dt  = vy
///   dvx/dt = 2·vy + x − (1-μ)(x+μ)/r₁³ − μ(x-1+μ)/r₂³
///   dvy/dt = -2·vx + y − (1-μ)·y/r₁³ − μ·y/r₂³
/// where r₁ = √((x+μ)² + y²) and r₂ = √((x-1+μ)² + y²).
///
/// The only first integral is the **Jacobi integral**:
///   C_J = (x² + y²) + 2·[(1-μ)/r₁ + μ/r₂] − (vx² + vy²)
///
/// Known since Jacobi (1836). Non-trivial structurally: unlike any of the
/// other showcase invariants, C_J has a subtracted kinetic energy term
/// (−v²) rather than the usual ½v² addition, reflecting the rotating-frame
/// effective potential. None of our existing `build_invariant_templates`
/// candidates match this shape — if the GP rediscovers it, that's a
/// genuine "assemble from scratch via crossover" demonstration.
fn pcr3bp_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    const MU: f64 = 0.01215; // Earth-Moon mass ratio (realistic benchmark)
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

fn pcr3bp_dynamics() -> Vec<(&'static str, SymExpr)> {
    // Dynamics with the Earth-Moon μ baked in. Note: the symbolic proof
    // only needs the *structural* form; the engine's verify_conservation
    // walks the chain rule using these rules. We encode the full RHS
    // expression so dC_J/dt can collapse symbolically.
    const MU: f64 = 0.01215;
    let var = |n: &str| SymExpr::Var(n.into());
    let mul = |a: SymExpr, b: SymExpr| SymExpr::Mul(Box::new(a), Box::new(b));
    let add = |a: SymExpr, b: SymExpr| SymExpr::Add(Box::new(a), Box::new(b));
    let neg = |a: SymExpr| SymExpr::Neg(Box::new(a));
    let pow = |a: SymExpr, k: f64| SymExpr::Pow(Box::new(a), k);

    // r₁² = (x + μ)² + y²
    let r1_sq = || {
        add(
            pow(add(var("x"), SymExpr::Const(MU)), 2.0),
            pow(var("y"), 2.0),
        )
    };
    // r₂² = (x - 1 + μ)² + y²
    let r2_sq = || {
        add(
            pow(add(var("x"), SymExpr::Const(MU - 1.0)), 2.0),
            pow(var("y"), 2.0),
        )
    };
    // dvx/dt = 2·vy + x − (1-μ)(x+μ)/r₁³ − μ(x-1+μ)/r₂³
    let dvx_dt = add(
        add(mul(SymExpr::Const(2.0), var("vy")), var("x")),
        add(
            neg(mul(
                mul(SymExpr::Const(1.0 - MU), add(var("x"), SymExpr::Const(MU))),
                pow(r1_sq(), -1.5),
            )),
            neg(mul(
                mul(SymExpr::Const(MU), add(var("x"), SymExpr::Const(MU - 1.0))),
                pow(r2_sq(), -1.5),
            )),
        ),
    );
    // dvy/dt = -2·vx + y − (1-μ)·y/r₁³ − μ·y/r₂³
    let dvy_dt = add(
        add(mul(SymExpr::Const(-2.0), var("vx")), var("y")),
        add(
            neg(mul(
                mul(SymExpr::Const(1.0 - MU), var("y")),
                pow(r1_sq(), -1.5),
            )),
            neg(mul(mul(SymExpr::Const(MU), var("y")), pow(r2_sq(), -1.5))),
        ),
    );
    vec![
        ("x", var("vx")),
        ("y", var("vy")),
        ("vx", dvx_dt),
        ("vy", dvy_dt),
    ]
}

/// Hénon-Heiles 4D chaotic Hamiltonian: state `[x, y, px, py]`.
///
/// H = ½(px² + py²) + ½(x² + y²) + x²y - (1/3)y³
///
/// Equations of motion:
///   dx/dt  = px
///   dy/dt  = py
///   dpx/dt = -x - 2xy
///   dpy/dt = -y - x² + y²
///
/// Energy is the *only* first integral in the chaotic regime (above the
/// escape threshold at E ≈ 1/6); the system has no second isolating
/// integral. This is the canonical stress test for autonomous
/// conservation-law discovery: the invariant is a 5-term expression with
/// cubic cross-couplings, and the engine must discover it without
/// over-fitting to any deceptive lower-complexity alternative.
fn henon_heiles_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, px, py) = (s[0], s[1], s[2], s[3]);
    vec![px, py, -x - 2.0 * x * y, -y - x * x + y * y]
}

fn henon_heiles_dynamics() -> Vec<(&'static str, SymExpr)> {
    // Helper for `-a - b`  as  `Add(Neg(a), Neg(b))`.
    let neg_add = |a: SymExpr, b: SymExpr| {
        SymExpr::Add(
            Box::new(SymExpr::Neg(Box::new(a))),
            Box::new(SymExpr::Neg(Box::new(b))),
        )
    };
    let var = |n: &str| SymExpr::Var(n.into());
    let mul = |a: SymExpr, b: SymExpr| SymExpr::Mul(Box::new(a), Box::new(b));
    let pow = |a: SymExpr, k: f64| SymExpr::Pow(Box::new(a), k);

    // dpx/dt = -x - 2xy  ≡  Add(Neg(x), Neg(2*x*y))
    let dpx_dt = neg_add(var("x"), mul(SymExpr::Const(2.0), mul(var("x"), var("y"))));

    // dpy/dt = -y - x² + y²  ≡  Add(Add(Neg(y), Neg(x²)), y²)
    let dpy_dt = SymExpr::Add(
        Box::new(SymExpr::Add(
            Box::new(SymExpr::Neg(Box::new(var("y")))),
            Box::new(SymExpr::Neg(Box::new(pow(var("x"), 2.0)))),
        )),
        Box::new(pow(var("y"), 2.0)),
    );

    vec![
        ("x", var("px")),
        ("y", var("py")),
        ("px", dpx_dt),
        ("py", dpy_dt),
    ]
}

// ───────────────────────────────────────────────────────────────────────────
// DISCOVERY PIPELINE — one problem end-to-end
// ───────────────────────────────────────────────────────────────────────────

/// Run autonomous discovery on a single ODE system and produce a fully
/// annotated result for the showcase table.
struct DiscoveryResult {
    name: String,
    domain: String,
    best: Option<AutonomousInvariant>,
    z3_status: String,
    recognition_headline: String,
    latex_formula: String,
}

fn run_problem(
    bridge: &PhysicsSearchEngine,
    name: &str,
    domain: &str,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    dynamics: &[(&str, SymExpr)],
    var_units: &UnitMap,
    t_max: f64,
    dt: f64,
    config: RegressorConfig,
) -> DiscoveryResult {
    println!("\n╭──────────────────────────────────────────────────────────────────╮");
    println!("│ {:65}│", name);
    println!("╰──────────────────────────────────────────────────────────────────╯");

    print!("  ⏵ Discovering invariants ... ");
    let invariants = discover_invariants_autonomous(
        rhs,
        initial_state,
        var_names,
        Some(dynamics),
        &config,
        t_max,
        dt,
    );
    println!("{} candidates", invariants.len());

    let best = invariants.into_iter().next();

    if let Some(ref inv) = best {
        println!("  ⏵ Best:    {}", inv.formula_str);
        println!("  ⏵ Variance: {:.2e}", inv.variance);

        // Z3 status (already determined inside discover_invariants_autonomous when
        // dynamics were provided — symbolically_proven means the chain rule proof
        // succeeded against the supplied dynamics)
        let z3_status = if inv.symbolically_proven {
            "PROVEN ✓".to_string()
        } else if inv.variance < 1e-6 {
            "Numerical".to_string()
        } else {
            "Approximate".to_string()
        };
        println!("  ⏵ Status:   {}", z3_status);

        // Recognition against the physics catalog — uses the new units-aware
        // path, which feeds dimensional inference into the search query and
        // breaks the structural-similarity plateau when units are provided.
        let report = recognize_expr_with_units(bridge, &inv.formula, name, var_units);
        let headline = report.headline();
        println!("  ⏵ Match:    {}", headline);
        if let Some(ref top) = report.best_match {
            println!(
                "    └ {} ({:?}) score={:.3}",
                top,
                report.best_domain.unwrap(),
                report.best_score
            );
        }

        // Paper-ready LaTeX
        let latex = expr_to_latex(&inv.formula);
        println!("  ⏵ LaTeX:    ${}$", latex);

        DiscoveryResult {
            name: name.to_string(),
            domain: domain.to_string(),
            best: Some(inv.clone()),
            z3_status,
            recognition_headline: headline,
            latex_formula: latex,
        }
    } else {
        println!("  ⚠  No invariant found");
        DiscoveryResult {
            name: name.to_string(),
            domain: domain.to_string(),
            best: None,
            z3_status: "—".to_string(),
            recognition_headline: "no candidates produced".to_string(),
            latex_formula: "\\text{none}".to_string(),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// COMBINED LATEX TABLE — paper-ready output
// ───────────────────────────────────────────────────────────────────────────

fn emit_combined_latex_table(results: &[DiscoveryResult], catalog_size: usize) -> String {
    let mut out = String::new();
    out.push_str("\n\n");
    out.push_str("% ═════════════════════════════════════════════════════════════════════\n");
    out.push_str("% AUTOMATED PHYSICIST — Ramanujan Protocol headline figure\n");
    out.push_str("% Generated by symthaea-physics-bridge/examples/ramanujan_showcase.rs\n");
    out.push_str("% Paste directly into papers/latex/ramanujan-protocol/main.tex\n");
    out.push_str("% ═════════════════════════════════════════════════════════════════════\n");
    out.push_str("\\begin{table}[htbp]\n");
    out.push_str("\\centering\n");
    out.push_str(
        "\\caption{Autonomous conservation-law discovery across canonical dynamical systems. ",
    );
    out.push_str(
        "Each row reports an invariant discovered with zero human guidance from the differential ",
    );
    out.push_str(
        "equations alone, the trajectory variance of the discovered quantity, the symbolic ",
    );
    out.push_str(&format!(
        "verification status (chain rule + Z3), and the closest match in the {}-equation physics catalog.}}\n",
        catalog_size
    ));
    out.push_str("\\label{tab:ramanujan_showcase}\n");
    out.push_str("\\begin{tabular}{lllll}\n");
    out.push_str("\\toprule\n");
    out.push_str("System & Domain & Discovered Invariant & Verification & Catalog Match \\\\\n");
    out.push_str("\\midrule\n");

    for r in results {
        let var_str = match &r.best {
            Some(inv) => format!("(var {:.1e})", inv.variance),
            None => "—".to_string(),
        };

        let status_cell = match (&r.best, r.z3_status.contains("PROVEN")) {
            (Some(_), true) => "\\textbf{PROVEN}",
            (Some(_), false) => "Numeric",
            (None, _) => "—",
        };

        // Sanitize headline for LaTeX — escape underscores, %, etc.
        // No truncation: the paper table needs the full catalog match text.
        // If the resulting row is too wide for the page, the paper should
        // use `sidewaystable` or `\resizebox`, not lose data.
        let headline_clean = sanitize_for_latex(&r.recognition_headline);

        out.push_str(&format!(
            "{} & {} & ${}$ {} & {} & {} \\\\\n",
            sanitize_for_latex(&r.name),
            r.domain,
            r.latex_formula,
            var_str,
            status_cell,
            headline_clean,
        ));
    }

    out.push_str("\\bottomrule\n");
    out.push_str("\\end{tabular}\n");
    out.push_str("\\end{table}\n");
    out
}

fn sanitize_for_latex(s: &str) -> String {
    s.replace('_', "\\_")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
}

// ───────────────────────────────────────────────────────────────────────────
// MAIN
// ───────────────────────────────────────────────────────────────────────────

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                  ║");
    println!("║         RAMANUJAN PROTOCOL — AUTOMATED PHYSICIST SHOWCASE        ║");
    println!("║                                                                  ║");
    println!("║  Tier 1A (Z3) + Tier 1B (Recognition) + Tier 1C (LaTeX) +        ║");
    println!("║  Tier 2A (Catalog) + Tier 2B (Reports) — end-to-end demo         ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\nInitializing physics catalog...");
    let bridge = PhysicsSearchEngine::new();
    println!(
        "Catalog loaded: {} equations across all physics domains.",
        bridge.catalog_size()
    );

    let mut results = Vec::new();

    // Default GP config — sufficient for polynomial invariants (harmonic, Kepler)
    let default_config = RegressorConfig {
        population_size: 250,
        generations: 80,
        max_depth: 4,
        max_complexity: 12,
        lambda: 0.001,
        mutation_rate: 0.4,
        seed: 42,
        ..RegressorConfig::default()
    };

    // Higher-budget config — needed for transcendental invariants (Lotka-Volterra
    // requires discovering the form `x - ln(x) + y - ln(y)`, which has 4 terms
    // with logarithms — much harder than polynomial conservation laws).
    let transcendental_config = RegressorConfig {
        population_size: 500,
        generations: 200,
        max_depth: 5,
        max_complexity: 20,
        lambda: 0.0005,
        mutation_rate: 0.4,
        seed: 42,
        ..RegressorConfig::default()
    };

    // ─── Per-problem unit annotations (drives dimensional inference) ───
    //
    // These let the recognition layer score discoveries against catalog
    // entries with matching SI dimensions, breaking the 0.70 structural
    // plateau by adding the dimensional axis to the multi-aspect scoring.
    //
    // For natural-units systems (harmonic oscillator, Kepler with GM=1),
    // the raw formula won't be dimensionally consistent, so inference
    // returns Inconsistent and falls back to DIMENSIONLESS — graceful
    // degradation, no errors.

    let mut harmonic_units = UnitMap::new();
    harmonic_units.insert("x".to_string(), DimensionalSignature::LENGTH);
    harmonic_units.insert("v".to_string(), DimensionalSignature::VELOCITY);

    let mut lv_units = UnitMap::new();
    // Lotka-Volterra: x and y are populations (dimensionless counts)
    lv_units.insert("x".to_string(), DimensionalSignature::DIMENSIONLESS);
    lv_units.insert("y".to_string(), DimensionalSignature::DIMENSIONLESS);

    let mut kepler_units = UnitMap::new();
    kepler_units.insert("x".to_string(), DimensionalSignature::LENGTH);
    kepler_units.insert("y".to_string(), DimensionalSignature::LENGTH);
    kepler_units.insert("vx".to_string(), DimensionalSignature::VELOCITY);
    kepler_units.insert("vy".to_string(), DimensionalSignature::VELOCITY);

    // Hénon-Heiles: natural units (dimensionless), since the Hamiltonian
    // mixes position² with position³ terms — no SI assignment propagates
    // cleanly. Dimensional inference will return Inconsistent → DIMENSIONLESS.
    let mut hh_units = UnitMap::new();
    hh_units.insert("x".to_string(), DimensionalSignature::DIMENSIONLESS);
    hh_units.insert("y".to_string(), DimensionalSignature::DIMENSIONLESS);
    hh_units.insert("px".to_string(), DimensionalSignature::DIMENSIONLESS);
    hh_units.insert("py".to_string(), DimensionalSignature::DIMENSIONLESS);

    // PCR3BP: natural units in the rotating frame (Ω=1, G(m₁+m₂)=1).
    let mut pcr3bp_units = UnitMap::new();
    pcr3bp_units.insert("x".to_string(), DimensionalSignature::DIMENSIONLESS);
    pcr3bp_units.insert("y".to_string(), DimensionalSignature::DIMENSIONLESS);
    pcr3bp_units.insert("vx".to_string(), DimensionalSignature::DIMENSIONLESS);
    pcr3bp_units.insert("vy".to_string(), DimensionalSignature::DIMENSIONLESS);

    // Mystery coupled-oscillator: natural units throughout.
    let mut mystery_units = UnitMap::new();
    mystery_units.insert("x".to_string(), DimensionalSignature::DIMENSIONLESS);
    mystery_units.insert("y".to_string(), DimensionalSignature::DIMENSIONLESS);
    mystery_units.insert("px".to_string(), DimensionalSignature::DIMENSIONLESS);
    mystery_units.insert("py".to_string(), DimensionalSignature::DIMENSIONLESS);

    // ─── Problem 1: Harmonic Oscillator ───
    results.push(run_problem(
        &bridge,
        "Harmonic Oscillator",
        "Classical Mechanics",
        harmonic_rhs,
        &[1.0, 0.0],
        &["x", "v"],
        &harmonic_dynamics(),
        &harmonic_units,
        20.0,
        0.01,
        default_config.clone(),
    ));

    // ─── Problem 2: Lotka-Volterra (transcendental invariant) ───
    results.push(run_problem(
        &bridge,
        "Lotka-Volterra Predator-Prey",
        "Ecology / Stat Mech",
        lotka_volterra_rhs,
        &[2.0, 1.0],
        &["x", "y"],
        &lotka_volterra_dynamics(),
        &lv_units,
        30.0,
        0.005,
        transcendental_config,
    ));

    // ─── Problem 3: Kepler Two-Body (4D system) ───
    results.push(run_problem(
        &bridge,
        "Kepler Two-Body",
        "Orbital Mechanics",
        kepler_rhs,
        &[1.0, 0.0, 0.0, 0.8],
        &["x", "y", "vx", "vy"],
        &kepler_dynamics(),
        &kepler_units,
        20.0,
        0.001,
        default_config.clone(),
    ));

    // ─── Problem 4: Hénon-Heiles (4D chaotic Hamiltonian) ───
    //
    // Stress test: 5-term invariant with cubic cross-couplings.
    // Initial state chosen well below the escape threshold (E ≈ 0.083 ≪ 1/6)
    // so the regime is quasi-periodic and the energy surface is well-sampled.
    // Higher budget than default for the richer template space.
    let henon_heiles_config = RegressorConfig {
        population_size: 500,
        generations: 200,
        max_depth: 6,
        max_complexity: 40,
        lambda: 0.0005,
        mutation_rate: 0.4,
        seed: 42,
        ..RegressorConfig::default()
    };
    results.push(run_problem(
        &bridge,
        "Hénon-Heiles Chaotic Hamiltonian",
        "Nonlinear Dynamics",
        henon_heiles_rhs,
        &[0.1, -0.1, 0.3, 0.2],
        &["x", "y", "px", "py"],
        &henon_heiles_dynamics(),
        &hh_units,
        40.0,
        0.005,
        henon_heiles_config,
    ));

    // ─── Problem 5: PCR3BP (planar circular restricted 3-body, rotating frame) ───
    //
    // The Jacobi integral `C_J = (x² + y²) + 2·[(1-μ)/r₁ + μ/r₂] − (vx² + vy²)`
    // is the only known first integral. The GP has the `pos² − vel²` quasi-
    // Jacobi skeleton template but must assemble the nonlocal `1/r₁ + 1/r₂`
    // term via crossover. Honest novel-ish test: if the engine finds even
    // a partial invariant with a pos²−vel² structure, that's a new class of
    // invariant shape (subtracted kinetic energy in a rotating frame).
    // Earth-Moon mass ratio μ = 0.01215. Initial state is an Earth-centered
    // retrograde orbit with enough energy to librate but not escape — gives
    // a richly varying trajectory where the Jacobi integral is strictly
    // conserved but all its component terms (x², y², vx², vy², 1/r₁, 1/r₂)
    // individually vary substantially.
    let pcr3bp_config = RegressorConfig {
        population_size: 600,
        generations: 300,
        max_depth: 6,
        max_complexity: 40,
        lambda: 0.0005,
        mutation_rate: 0.4,
        seed: 42,
        ..RegressorConfig::default()
    };
    results.push(run_problem(
        &bridge,
        "PCR3BP Jacobi Integral",
        "Celestial Mechanics (rotating frame)",
        pcr3bp_rhs,
        &[0.3, 0.0, 0.0, 1.4], // librating orbit around Earth primary
        &["x", "y", "vx", "vy"],
        &pcr3bp_dynamics(),
        &pcr3bp_units,
        15.0,
        0.001,
        pcr3bp_config,
    ));

    // ─── Problem 6: "Mystery" coupled anisotropic oscillator ───
    //
    // Designer-constructed novel-ish target. The author (who designed the
    // RHS) knows the conserved quantity is
    //     H = ½(px² + py²) + x² + y² + xy
    // and intentionally did NOT tell the engine. With the coupled-oscillator
    // template added in Move 11, the GP must still assemble the full 5-term
    // form via crossover (the template gives the quadratic kernel, constant
    // tuning + mutation adds the ½ scaling). If the engine finds the exact
    // Hamiltonian, it's a demonstration that the template library can be
    // incrementally extended to cover previously-out-of-reach invariant
    // classes.
    let mystery_config = RegressorConfig {
        population_size: 600,
        generations: 300,
        max_depth: 6,
        max_complexity: 40,
        lambda: 0.0005,
        mutation_rate: 0.4,
        seed: 42,
        ..RegressorConfig::default()
    };
    results.push(run_problem(
        &bridge,
        "Coupled Anisotropic Oscillator (Mystery)",
        "Designer-constructed stretch target",
        mystery_rhs,
        &[1.0, 0.0, 0.0, 1.0],
        &["x", "y", "px", "py"],
        &mystery_dynamics(),
        &mystery_units,
        30.0,
        0.005,
        mystery_config,
    ));

    // ─── Sequence-based discovery: Triangular numbers (with Z3 path) ───
    println!("\n╭──────────────────────────────────────────────────────────────────╮");
    println!(
        "│ {:65}│",
        "Triangular Numbers (sequence + Z3 formal verification)"
    );
    println!("╰──────────────────────────────────────────────────────────────────╯");

    let mut seq_engine = ConjectureEngine::with_config(RegressorConfig {
        population_size: 200,
        generations: 60,
        max_depth: 4,
        max_complexity: 12,
        seed: 42,
        ..RegressorConfig::default()
    });
    let triangular_data: Vec<(f64, f64)> = (1..=10)
        .map(|n| (n as f64, (n * (n + 1) / 2) as f64))
        .collect();
    seq_engine.observe(ObservedSequence::new(
        "triangular(n)",
        MathDomain::Combinatorics,
        triangular_data,
    ));
    seq_engine.generate_conjectures(3);
    seq_engine.verify_numerical();
    seq_engine.auto_prove_via_z3();

    if let Some(best) = seq_engine.best_for("triangular(n)") {
        let z3_str = match best.status {
            ConjectureStatus::FormallyVerified { proof_steps } => {
                format!("Z3 PROVEN ({} steps) ✓", proof_steps)
            }
            ConjectureStatus::NumericallyTested { .. } => "Numeric".to_string(),
            _ => format!("{:?}", best.status),
        };
        let latex = expr_to_latex(&best.formula);
        let report = recognize_expr(&bridge, &best.formula, "triangular(n)");
        println!("  ⏵ Best:    {}", best.formula_str);
        println!("  ⏵ MSE:     {:.2e}", best.training_mse);
        println!("  ⏵ Status:  {}", z3_str);
        println!("  ⏵ Match:   {}", report.headline());
        println!("  ⏵ LaTeX:   ${}$", latex);

        results.push(DiscoveryResult {
            name: "Triangular Numbers".to_string(),
            domain: "Combinatorics".to_string(),
            best: None, // sequence-based, doesn't fit AutonomousInvariant struct
            z3_status: z3_str,
            recognition_headline: report.headline(),
            latex_formula: latex,
        });
    }

    // ─── Combined paper-ready table ───
    println!("\n\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                  ║");
    println!("║                  PAPER-READY LATEX TABLE                         ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let latex_table = emit_combined_latex_table(&results, bridge.catalog_size());
    println!("{}", latex_table);

    println!("\n──────────────────────────────────────────────────────────────────────");
    println!(
        " SHOWCASE COMPLETE: {} problems, {} catalog matches, all results above.",
        results.len(),
        results
            .iter()
            .filter(|r| !r.recognition_headline.contains("NO KNOWN"))
            .count(),
    );
    println!("──────────────────────────────────────────────────────────────────────");
}
