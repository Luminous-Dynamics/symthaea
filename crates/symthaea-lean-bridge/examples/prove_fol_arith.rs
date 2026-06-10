// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `prove_fol_arith` — Phase 2 arithmetic theorem emission.
//!
//! Builds a suite of hand-crafted `FolFormulaExt` fixtures, emits one
//! Lean 4 `.lean` file per fixture via `fol_ext_bridge::render_fol_ext_file`,
//! and (optionally, when `LAKE_ENV=1`) invokes `lake env lean` for external
//! verification.
//!
//! ## Expected scope
//!
//! - `RatLit` rendering: `((1 : ℝ) / (3 : ℝ))` — no f64 rounding.
//! - Fragment detection: QF_LIA/LRA/NIA/NRA auto-selected per fixture.
//! - Tactic emission: `omega` for linear integer, `linarith` for linear
//!   real, `nlinarith` for non-linear.
//!
//! ## Running
//!
//! ```bash
//! # emit only (no Lean verification; fast):
//! cargo run -p symthaea-lean-bridge --example prove_fol_arith
//!
//! # emit + verify via Lake (requires lean-proofs/phase2/ Mathlib setup):
//! LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_fol_arith
//! ```
//!
//! Output directory: `proofs/fol_arith/`, overridable via `FOL_ARITH_OUT_DIR`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use symthaea_core::hdc::fol_ext_smt::detect_fragment;
use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};
use symthaea_lean_bridge::fol_ext_bridge::render_fol_ext_file;

fn v(name: &str) -> Term {
    Term::var(name)
}

struct Fixture {
    name: &'static str,
    description: &'static str,
    goal: FolFormulaExt,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        // ─── Linear, quantified reflexivity ───────────────────────────────
        Fixture {
            name: "refl_real",
            description: "∀ x : ℝ, x = x",
            goal: FolFormulaExt::forall("x", NumericType::Real, FolFormulaExt::eq(v("x"), v("x"))),
        },
        Fixture {
            name: "refl_int",
            description: "∀ n : ℤ, n = n",
            goal: FolFormulaExt::forall("n", NumericType::Int, FolFormulaExt::eq(v("n"), v("n"))),
        },
        // ─── Linear successor / monotonicity ──────────────────────────────
        Fixture {
            name: "succ_gt_self_int",
            description: "∀ n : ℤ, n < n + 1",
            goal: FolFormulaExt::forall(
                "n",
                NumericType::Int,
                FolFormulaExt::lt(v("n"), v("n").add(Term::IntLit(1))),
            ),
        },
        Fixture {
            name: "nat_nonneg",
            description: "∀ n : ℕ, 0 ≤ n",
            goal: FolFormulaExt::forall(
                "n",
                NumericType::Nat,
                FolFormulaExt::le(Term::IntLit(0), v("n")),
            ),
        },
        Fixture {
            name: "add_comm_real",
            description: "∀ a b : ℝ, a + b = b + a",
            goal: FolFormulaExt::forall(
                "a",
                NumericType::Real,
                FolFormulaExt::forall(
                    "b",
                    NumericType::Real,
                    FolFormulaExt::eq(v("a").add(v("b")), v("b").add(v("a"))),
                ),
            ),
        },
        Fixture {
            name: "trichotomy_real",
            description: "∀ x y : ℝ, x = y ∨ x < y ∨ y < x",
            goal: FolFormulaExt::forall(
                "x",
                NumericType::Real,
                FolFormulaExt::forall(
                    "y",
                    NumericType::Real,
                    FolFormulaExt::eq(v("x"), v("y"))
                        .or(FolFormulaExt::lt(v("x"), v("y")))
                        .or(FolFormulaExt::lt(v("y"), v("x"))),
                ),
            ),
        },
        // ─── Non-linear (nlinarith territory) ─────────────────────────────
        Fixture {
            name: "square_nonneg",
            description: "∀ x : ℝ, 0 ≤ x * x",
            goal: FolFormulaExt::forall(
                "x",
                NumericType::Real,
                FolFormulaExt::le(Term::IntLit(0), v("x").mul(v("x"))),
            ),
        },
        Fixture {
            name: "square_nonneg_pow",
            description: "∀ x : ℝ, 0 ≤ x² (using Pow)",
            goal: FolFormulaExt::forall(
                "x",
                NumericType::Real,
                FolFormulaExt::le(Term::IntLit(0), v("x").pow(2)),
            ),
        },
        // ─── Implications ─────────────────────────────────────────────────
        Fixture {
            name: "pos_implies_nonneg",
            description: "∀ x : ℝ, 0 < x → 0 ≤ x",
            goal: FolFormulaExt::forall(
                "x",
                NumericType::Real,
                FolFormulaExt::lt(Term::IntLit(0), v("x"))
                    .implies(FolFormulaExt::le(Term::IntLit(0), v("x"))),
            ),
        },
        // ─── Exact rational (RatLit pathway) ──────────────────────────────
        Fixture {
            name: "three_times_third_eq_one",
            description: "3 · (1/3) = 1  (RatLit exactness; closes without rescale)",
            goal: FolFormulaExt::eq(Term::IntLit(3).mul(Term::rat(1, 3)), Term::IntLit(1)),
        },
        // ─── Algebraic identities (stretch `nlinarith`) ───────────────────
        Fixture {
            // (a + b)² = a² + 2ab + b²
            name: "square_of_sum",
            description: "∀ a b : ℝ, (a + b)² = a² + 2·a·b + b²",
            goal: FolFormulaExt::forall(
                "a",
                NumericType::Real,
                FolFormulaExt::forall(
                    "b",
                    NumericType::Real,
                    FolFormulaExt::eq(
                        v("a").add(v("b")).pow(2),
                        v("a")
                            .pow(2)
                            .add(Term::IntLit(2).mul(v("a")).mul(v("b")))
                            .add(v("b").pow(2)),
                    ),
                ),
            ),
        },
        Fixture {
            // AM-GM for one variable degenerate: 2·a ≤ 1 + a² ⇔ (a - 1)² ≥ 0
            name: "am_gm_onevar",
            description: "∀ a : ℝ, 2·a ≤ 1 + a²  (equivalent to (a-1)² ≥ 0)",
            goal: FolFormulaExt::forall(
                "a",
                NumericType::Real,
                FolFormulaExt::le(
                    Term::IntLit(2).mul(v("a")),
                    Term::IntLit(1).add(v("a").pow(2)),
                ),
            ),
        },
        Fixture {
            // Sum of squares is non-negative
            name: "sum_of_squares_nonneg",
            description: "∀ a b : ℝ, 0 ≤ a² + b²",
            goal: FolFormulaExt::forall(
                "a",
                NumericType::Real,
                FolFormulaExt::forall(
                    "b",
                    NumericType::Real,
                    FolFormulaExt::le(Term::IntLit(0), v("a").pow(2).add(v("b").pow(2))),
                ),
            ),
        },
        Fixture {
            // 2xy ≤ x² + y² (variance ≥ 0 identity rearranged)
            name: "two_xy_le_sum_of_squares",
            description: "∀ x y : ℝ, 2·x·y ≤ x² + y²",
            goal: FolFormulaExt::forall(
                "x",
                NumericType::Real,
                FolFormulaExt::forall(
                    "y",
                    NumericType::Real,
                    FolFormulaExt::le(
                        Term::IntLit(2).mul(v("x")).mul(v("y")),
                        v("x").pow(2).add(v("y").pow(2)),
                    ),
                ),
            ),
        },
        // ─── IMO-class stress test ────────────────────────────────────────
        Fixture {
            // Cauchy-Schwarz (2-variable form): (a·x + b·y)² ≤ (a²+b²)(x²+y²)
            // Key witness: (a·y - b·x)² ≥ 0 expands to a²y² - 2abxy + b²x² ≥ 0,
            // which rearranges to the Cauchy-Schwarz inequality.
            // The cascade should close this via nlinarith with sq_nonneg hints
            // on pairwise differences (a·y - b·x).
            name: "cauchy_schwarz_2var",
            description: "∀ a b x y : ℝ, (a·x + b·y)² ≤ (a²+b²)(x²+y²) — Cauchy-Schwarz",
            goal: FolFormulaExt::forall(
                "a",
                NumericType::Real,
                FolFormulaExt::forall(
                    "b",
                    NumericType::Real,
                    FolFormulaExt::forall(
                        "x",
                        NumericType::Real,
                        FolFormulaExt::forall(
                            "y",
                            NumericType::Real,
                            FolFormulaExt::le(
                                v("a").mul(v("x")).add(v("b").mul(v("y"))).pow(2),
                                v("a")
                                    .pow(2)
                                    .add(v("b").pow(2))
                                    .mul(v("x").pow(2).add(v("y").pow(2))),
                            ),
                        ),
                    ),
                ),
            ),
        },
    ]
}

fn output_dir() -> PathBuf {
    env::var("FOL_ARITH_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("proofs/fol_arith"))
}

fn lake_env_available() -> bool {
    if env::var("LAKE_ENV").ok().as_deref() != Some("1") {
        return false;
    }
    Command::new("lake")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn verify_with_lake(lean_file: &PathBuf, project_dir: &PathBuf) -> Result<bool, String> {
    // Lake's `env lean` resolves the target path relative to the current
    // working directory, but we cd into `project_dir` for Mathlib
    // resolution. Canonicalize the lean file path to an absolute form so
    // the cd-into-project doesn't break path lookup.
    let abs_lean_file = fs::canonicalize(lean_file)
        .map_err(|e| format!("canonicalize {} failed: {}", lean_file.display(), e))?;
    let out = Command::new("lake")
        .arg("env")
        .arg("lean")
        .arg(&abs_lean_file)
        .current_dir(project_dir)
        .output()
        .map_err(|e| format!("spawn lake failed: {}", e))?;
    Ok(out.status.success())
}

fn main() -> ExitCode {
    let out_dir = output_dir();
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("# mkdir {} failed: {}", out_dir.display(), e);
        return ExitCode::from(2);
    }

    let use_lake = lake_env_available();
    let project_dir = PathBuf::from("lean-proofs/phase2");

    println!("fixture,file,fragment,tactic,bytes,lake_check,note");

    let mut lake_accepted = 0usize;
    let mut lake_rejected = 0usize;
    let mut emitted = 0usize;

    for fx in fixtures() {
        let file_contents = render_fol_ext_file(fx.name, &fx.goal);
        let file_path = out_dir.join(format!("{}.lean", fx.name));
        let bytes = file_contents.len();

        if let Err(e) = fs::write(&file_path, &file_contents) {
            eprintln!("# write failed for {}: {}", file_path.display(), e);
            continue;
        }
        emitted += 1;

        let fragment = detect_fragment(&fx.goal);
        let tactic = fragment.suggested_lean_tactic();

        let lake_label = if use_lake {
            match verify_with_lake(&file_path, &project_dir) {
                Ok(true) => {
                    lake_accepted += 1;
                    "accepted"
                }
                Ok(false) => {
                    lake_rejected += 1;
                    "rejected"
                }
                Err(_) => "lake_error",
            }
        } else {
            "skipped"
        };

        println!(
            "{},{},{},{},{},{},{}",
            fx.name,
            file_path.display(),
            fragment.logic_name(),
            tactic,
            bytes,
            lake_label,
            fx.description.replace(',', ";")
        );
        eprintln!("# {} — {}", fx.name, fx.description);
    }

    eprintln!(
        "# summary: {} emitted | lake accepted={} rejected={} | mode={}",
        emitted,
        lake_accepted,
        lake_rejected,
        if use_lake { "verify" } else { "emit-only" }
    );
    if !use_lake {
        eprintln!("# Set LAKE_ENV=1 and ensure lean-proofs/phase2/ has Mathlib");
        eprintln!("# resolved to enable external verification. See the README.");
    }

    if use_lake && lake_rejected > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
