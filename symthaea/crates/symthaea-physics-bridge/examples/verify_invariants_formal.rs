// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `verify_invariants_formal` — genuine Z3 formal verification of the
//! Ramanujan showcase's conservation laws, with SMT-LIB2 witness files
//! persisted to disk for independent re-verification.
//!
//! ## Why this is a separate binary from `ramanujan_showcase`
//!
//! The showcase's "PROVEN" tag is set by
//! `conjecture_engine::verify_conservation_symbolic`, which does:
//!   (a) symbolic chain-rule derivation of dE/dt
//!   (b) numerical residual check at 6 sample trajectory points
//! This is **strong evidence** (a polynomial that evaluates to zero at 6
//! generic points almost certainly is zero), but it is NOT a formal Z3
//! UNSAT proof. This binary closes that gap: for each of the committed
//! invariants, it hand-builds the SMT-LIB2 obligation ∃x: dE/dt ≠ 0,
//! feeds it to Z3, captures the `unsat` certificate, and writes the query
//! to `papers/ramanujan/proofs/*.smt2` so an independent verifier with any
//! SMT-LIB2-compliant solver can re-check.
//!
//! ## Run
//!
//! ```bash
//! # from repo root, inside `nix develop`:
//! cargo run -p symthaea-physics-bridge --example verify_invariants_formal
//! ```
//!
//! Emits CSV to stdout and writes witness files to
//! `papers/ramanujan/proofs/` (overridable via `SYMTHAEA_Z3_DUMP_DIR`).
//!
//! This binary is run by `papers/ramanujan/reproduce.sh --verify-proofs`
//! as the independent-witness step.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use symthaea_core::hdc::conjecture_engine::SymExpr;

// ───────────────────────────────────────────────────────────────────────────
// SymExpr → SMT-LIB2 serializer (local to this binary).
//
// Supports the polynomial + reciprocal-power subset that's in-scope for
// Z3's QF_NRA. Returns None for transcendentals (sin/cos/ln/exp) — those
// invariants will be reported as "out_of_scope" in the CSV and not
// committed as formal proofs. This matches the honesty discipline: we
// don't claim formal verification for Lotka-Volterra's log-of-x invariant.
// ───────────────────────────────────────────────────────────────────────────

fn sym_to_smt(expr: &SymExpr) -> Option<String> {
    match expr {
        SymExpr::Var(name) => Some(name.clone()),
        SymExpr::Const(c) => {
            if *c == c.trunc() && c.abs() < 1e18 {
                Some(format!("{}", *c as i64))
            } else {
                Some(format!("{}", c))
            }
        }
        SymExpr::Add(a, b) => Some(format!("(+ {} {})", sym_to_smt(a)?, sym_to_smt(b)?)),
        SymExpr::Mul(a, b) => Some(format!("(* {} {})", sym_to_smt(a)?, sym_to_smt(b)?)),
        SymExpr::Neg(a) => Some(format!("(- {})", sym_to_smt(a)?)),
        SymExpr::Pow(base, exp) => {
            if *exp != exp.trunc() || *exp < 0.0 || *exp > 16.0 {
                return None;
            }
            let n = *exp as u32;
            let s = sym_to_smt(base)?;
            if n == 0 {
                Some("1".to_string())
            } else if n == 1 {
                Some(s)
            } else {
                let mut out = s.clone();
                for _ in 1..n {
                    out = format!("(* {} {})", s, out);
                }
                Some(out)
            }
        }
        // Out-of-scope for QF_NRA formal verification.
        SymExpr::Div(_, _) | SymExpr::Log(_) | SymExpr::Sin(_) | SymExpr::Cos(_) => None,
    }
}

fn variables_in(expr: &SymExpr, out: &mut Vec<String>) {
    match expr {
        SymExpr::Var(n) => {
            if !out.contains(n) {
                out.push(n.clone());
            }
        }
        SymExpr::Const(_) => {}
        SymExpr::Add(a, b) | SymExpr::Mul(a, b) => {
            variables_in(a, out);
            variables_in(b, out);
        }
        SymExpr::Neg(a) => variables_in(a, out),
        SymExpr::Pow(base, _) => variables_in(base, out),
        SymExpr::Div(a, b) => {
            variables_in(a, out);
            variables_in(b, out);
        }
        SymExpr::Log(a) | SymExpr::Sin(a) | SymExpr::Cos(a) => variables_in(a, out),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The problem definitions. These match the committed showcase outputs.
// ───────────────────────────────────────────────────────────────────────────

struct Problem {
    name: &'static str,
    description: &'static str,
    energy: SymExpr,
    dynamics: Vec<(&'static str, SymExpr)>,
    /// If `false`, skip Z3 verification — the invariant uses transcendentals
    /// or operators outside QF_NRA.
    formally_verifiable: bool,
}

fn v(name: &str) -> SymExpr {
    SymExpr::Var(name.to_string())
}
fn c(x: f64) -> SymExpr {
    SymExpr::Const(x)
}
fn add(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::Mul(Box::new(a), Box::new(b))
}
fn neg(a: SymExpr) -> SymExpr {
    SymExpr::Neg(Box::new(a))
}
fn pow(a: SymExpr, n: f64) -> SymExpr {
    SymExpr::Pow(Box::new(a), n)
}

fn problems() -> Vec<Problem> {
    vec![
        Problem {
            name: "harmonic_oscillator",
            description: "E = x² + v²; dx/dt = v, dv/dt = -x",
            energy: add(pow(v("x"), 2.0), pow(v("v"), 2.0)),
            dynamics: vec![("x", v("v")), ("v", neg(v("x")))],
            formally_verifiable: true,
        },
        Problem {
            name: "kepler_angular_momentum",
            description: "L = x·vy - y·vx (Kepler two-body)",
            energy: add(mul(v("x"), v("vy")), neg(mul(v("y"), v("vx")))),
            dynamics: vec![
                ("x", v("vx")),
                ("y", v("vy")),
                // For the angular-momentum proof, the gravitational terms
                // in dvx/dt = -x/r³ and dvy/dt = -y/r³ cancel algebraically
                // because L's derivative is x·(-y/r³) - y·(-x/r³) = 0. We
                // encode the central-force structure as dvx/dt = -x·f(r)
                // and dvy/dt = -y·f(r) where f(r) is a single symbol `fr`;
                // this captures the essential property and stays polynomial.
                ("vx", neg(mul(v("x"), v("fr")))),
                ("vy", neg(mul(v("y"), v("fr")))),
            ],
            formally_verifiable: true,
        },
        Problem {
            // Multiply through by 6 to avoid fractional constants (1/2 and
            // 1/3 are not exact in f64, which makes Z3's exact arithmetic
            // read the serialized decimal as non-zero). The conserved quantity
            // 6H is still conserved iff H is. See VERIFY.md section
            // "Why rescaling is sound".
            name: "henon_heiles_6H",
            description: "6H = 3(px² + py²) + 3(x² + y²) + 6x²y − 2y³ (Hénon-Heiles, ×6 to avoid f64/rational mismatch)",
            energy: add(
                add(
                    mul(c(3.0), add(pow(v("px"), 2.0), pow(v("py"), 2.0))),
                    mul(c(3.0), add(pow(v("x"), 2.0), pow(v("y"), 2.0))),
                ),
                add(
                    mul(c(6.0), mul(pow(v("x"), 2.0), v("y"))),
                    neg(mul(c(2.0), pow(v("y"), 3.0))),
                ),
            ),
            dynamics: vec![
                ("x", v("px")),
                ("y", v("py")),
                ("px", neg(add(v("x"), mul(c(2.0), mul(v("x"), v("y")))))),
                (
                    "py",
                    neg(add(v("y"), add(pow(v("x"), 2.0), neg(pow(v("y"), 2.0))))),
                ),
            ],
            formally_verifiable: true,
        },
        Problem {
            name: "mystery_ode",
            description: "H = ½(px² + py²) + x² + y² + xy (anisotropic coupled oscillator)",
            energy: add(
                mul(c(0.5), add(pow(v("px"), 2.0), pow(v("py"), 2.0))),
                add(
                    add(pow(v("x"), 2.0), pow(v("y"), 2.0)),
                    mul(v("x"), v("y")),
                ),
            ),
            dynamics: vec![
                ("x", v("px")),
                ("y", v("py")),
                ("px", neg(add(mul(c(2.0), v("x")), v("y")))),
                ("py", neg(add(mul(c(2.0), v("y")), v("x")))),
            ],
            formally_verifiable: true,
        },
        Problem {
            name: "lotka_volterra",
            description: "V = x − ln(x) + y − ln(y) — transcendental, not Z3-verifiable",
            energy: v("x"), // placeholder — not used when formally_verifiable=false
            dynamics: vec![],
            formally_verifiable: false,
        },
        // ─── Additional polynomial invariants (Tier B extensions) ───────
        Problem {
            // Duffing oscillator (unforced, conservative):
            //   dx/dt = v, dv/dt = -x - x^3
            // Energy: E = ½v² + ½x² + ¼x⁴  →  ×4 for integer coefs:
            //   4E = 2v² + 2x² + x⁴
            name: "duffing_4E",
            description: "4E = 2v² + 2x² + x⁴ (Duffing oscillator, conservative; ×4 for integer coefs)",
            energy: add(
                add(mul(c(2.0), pow(v("v"), 2.0)), mul(c(2.0), pow(v("x"), 2.0))),
                pow(v("x"), 4.0),
            ),
            dynamics: vec![
                ("x", v("v")),
                ("v", neg(add(v("x"), pow(v("x"), 3.0)))),
            ],
            formally_verifiable: true,
        },
        Problem {
            // Quartic anharmonic oscillator:
            //   dx/dt = v, dv/dt = -x³
            // Energy: E = ½v² + ¼x⁴  →  ×4: 4E = 2v² + x⁴
            name: "quartic_anharmonic_4E",
            description: "4E = 2v² + x⁴ (quartic anharmonic oscillator; ×4 for integer coefs)",
            energy: add(mul(c(2.0), pow(v("v"), 2.0)), pow(v("x"), 4.0)),
            dynamics: vec![("x", v("v")), ("v", neg(pow(v("x"), 3.0)))],
            formally_verifiable: true,
        },
        Problem {
            // 2D isotropic harmonic oscillator:
            //   dx/dt = vx, dy/dt = vy
            //   dvx/dt = -x, dvy/dt = -y
            // Energy: E = ½(vx² + vy²) + ½(x² + y²)  →  ×2: 2E = vx² + vy² + x² + y²
            name: "isotropic_2d_energy",
            description: "2E = vx² + vy² + x² + y² (2D isotropic harmonic; ×2)",
            energy: add(
                add(pow(v("vx"), 2.0), pow(v("vy"), 2.0)),
                add(pow(v("x"), 2.0), pow(v("y"), 2.0)),
            ),
            dynamics: vec![
                ("x", v("vx")),
                ("y", v("vy")),
                ("vx", neg(v("x"))),
                ("vy", neg(v("y"))),
            ],
            formally_verifiable: true,
        },
        Problem {
            // Same 2D isotropic harmonic oscillator, but the angular-momentum
            // invariant L = x·vy − y·vx. Demonstrates a second independent
            // conservation law beyond energy.
            name: "isotropic_2d_angular_momentum",
            description: "L = x·vy − y·vx (2D isotropic harmonic oscillator)",
            energy: add(mul(v("x"), v("vy")), neg(mul(v("y"), v("vx")))),
            dynamics: vec![
                ("x", v("vx")),
                ("y", v("vy")),
                ("vx", neg(v("x"))),
                ("vy", neg(v("y"))),
            ],
            formally_verifiable: true,
        },
        Problem {
            // Linear coupled oscillators (two modes with coupling constant k=1):
            //   dx1/dt = v1, dx2/dt = v2
            //   dv1/dt = -2x1 + x2, dv2/dt = -2x2 + x1
            // Energy: E = ½(v1² + v2²) + x1² + x2² − x1·x2  →  ×2:
            //   2E = v1² + v2² + 2x1² + 2x2² − 2·x1·x2
            name: "linear_coupled_2E",
            description: "2E = v1² + v2² + 2x1² + 2x2² − 2·x1·x2 (linear coupled oscillators, k=1; ×2)",
            energy: add(
                add(pow(v("v1"), 2.0), pow(v("v2"), 2.0)),
                add(
                    add(mul(c(2.0), pow(v("x1"), 2.0)), mul(c(2.0), pow(v("x2"), 2.0))),
                    neg(mul(c(2.0), mul(v("x1"), v("x2")))),
                ),
            ),
            dynamics: vec![
                ("x1", v("v1")),
                ("x2", v("v2")),
                ("v1", add(neg(mul(c(2.0), v("x1"))), v("x2"))),
                ("v2", add(neg(mul(c(2.0), v("x2"))), v("x1"))),
            ],
            formally_verifiable: true,
        },
    ]
}

// ───────────────────────────────────────────────────────────────────────────
// Build the SMT-LIB2 obligation for dE/dt ≠ 0.
// If Z3 returns `unsat`, the invariant is formally conserved.
// ───────────────────────────────────────────────────────────────────────────

fn build_obligation(problem: &Problem) -> Option<String> {
    // dE/dt = Σ (∂E/∂x_i) * (dx_i/dt)
    let mut total_deriv = SymExpr::Const(0.0);
    for (var_name, dvar_dt) in &problem.dynamics {
        let partial = problem.energy.diff(var_name).simplify();
        total_deriv = add(total_deriv, mul(partial, dvar_dt.clone()));
    }
    let total_deriv = total_deriv.simplify();

    let mut vars: Vec<String> = Vec::new();
    variables_in(&total_deriv, &mut vars);
    variables_in(&problem.energy, &mut vars);
    vars.sort();
    vars.dedup();

    let deriv_smt = sym_to_smt(&total_deriv)?;
    let energy_smt = sym_to_smt(&problem.energy)?;

    let mut out = String::new();
    out.push_str("(set-logic QF_NRA)\n");
    out.push_str(&format!(
        "; Ramanujan Protocol formal-verification obligation\n"
    ));
    out.push_str(&format!("; Problem: {}\n", problem.description));
    out.push_str(&format!("; Invariant (Lean-ready): {}\n", energy_smt));
    out.push_str(&format!("; Claim: dE/dt = 0 identically.\n"));
    out.push_str(&format!(
        "; Z3 query: ∃{} : dE/dt ≠ 0 (expected UNSAT).\n",
        vars.join(", ")
    ));
    out.push('\n');
    for var in &vars {
        out.push_str(&format!("(declare-const {} Real)\n", var));
    }
    out.push('\n');
    out.push_str(&format!("(assert (not (= {} 0)))\n", deriv_smt));
    out.push_str("(check-sat)\n");
    Some(out)
}

fn run_z3(smt: &str) -> Result<String, String> {
    let z3 = env::var("Z3_PATH").unwrap_or_else(|_| "z3".to_string());
    let mut child = Command::new(&z3)
        .arg("-in")
        .arg("-T:10")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn z3 failed: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(smt.as_bytes())
            .map_err(|e| format!("stdin write failed: {}", e))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait failed: {}", e))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn dump_dir() -> PathBuf {
    env::var("SYMTHAEA_Z3_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("papers/ramanujan/proofs"))
}

fn main() -> ExitCode {
    let dir = dump_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("# mkdir {} failed: {}", dir.display(), e);
        return ExitCode::from(2);
    }

    println!("problem,smt_file,bytes,z3_result,formally_proven,note");

    let mut proven = 0usize;
    let mut skipped_transcendental = 0usize;
    let mut z3_unavailable = 0usize;
    let mut sat_or_unknown = 0usize;

    for p in problems() {
        if !p.formally_verifiable {
            skipped_transcendental += 1;
            let note = "transcendental_out_of_QF_NRA";
            println!("{},—,0,skipped,false,{}", p.name, note);
            eprintln!("# {} — {}  [skipped: {}]", p.name, p.description, note);
            continue;
        }

        let obligation = match build_obligation(&p) {
            Some(s) => s,
            None => {
                sat_or_unknown += 1;
                println!("{},—,0,build_failed,false,serializer_rejected", p.name);
                continue;
            }
        };

        let path = dir.join(format!("{}.smt2", p.name));
        if let Err(e) = fs::write(&path, &obligation) {
            eprintln!("# write {} failed: {}", path.display(), e);
            continue;
        }

        match run_z3(&obligation) {
            Ok(result) => {
                let trimmed = result.trim();
                // Z3 can emit multiple lines; the last line is typically
                // the (check-sat) answer.
                let last_line = trimmed.lines().last().unwrap_or(trimmed);
                let formally_proven = last_line == "unsat";
                if formally_proven {
                    proven += 1;
                } else {
                    sat_or_unknown += 1;
                }
                println!(
                    "{},{},{},{},{},{}",
                    p.name,
                    path.display(),
                    obligation.len(),
                    last_line,
                    formally_proven,
                    p.description.replace(',', ";")
                );
            }
            Err(e) => {
                z3_unavailable += 1;
                println!(
                    "{},{},{},z3_error,false,{}",
                    p.name,
                    path.display(),
                    obligation.len(),
                    e.replace(',', ";")
                );
            }
        }
    }

    eprintln!(
        "# summary: {} formally proven | {} skipped (transcendental) | {} sat/unknown | {} z3 errors",
        proven, skipped_transcendental, sat_or_unknown, z3_unavailable
    );

    if z3_unavailable > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
