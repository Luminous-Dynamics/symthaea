// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `prove_minif2f_curated` — Phase 3 Option (b) honest measurement of the
//! Phase 2 FolFormulaExt → Mathlib-tactic cascade against a hand-curated
//! subset of the public miniF2F-v2 corpus.
//!
//! ## Why this harness
//!
//! Phase 2 W1-W4 closed 14/14 = 100% on hand-*crafted* arithmetic fixtures
//! (see `prove_fol_arith.rs`). That's a training-set number — the cascade
//! was iterated until those 14 fixtures closed. The honest question is:
//! **does the same cascade close problems it has never seen, drawn from
//! the real miniF2F-v2 algebra subset?**
//!
//! This harness answers that question on a **curated-and-hand-translated**
//! subset. The translation is manual (no Lean parser yet — that's Phase 4
//! option (c)), so the fixture pool is small (~35 problems) but every
//! fixture is a verbatim translation of a real miniF2F-v2 theorem.
//!
//! ## Scope
//!
//! Each fixture is drawn from `data/benchmarks/minif2f/MiniF2F/{Valid,Test}/`
//! and labeled with the `mathd_algebra_*` or `mathd_numbertheory_*` name of
//! the upstream file. We only included problems that:
//!
//! - Use only ℝ, ℤ, ℕ numeric types (no ℂ, ZMod, Equiv, NNReal, …)
//! - Use only `Eq`, `Lt`, `Le`, `+`, `-`, `*`, `/`, integer `^n` (no
//!   `Real.sqrt`, `Real.log`, `Real.sin/cos`, fractional exponents, ...)
//! - Don't involve function abstraction (`∀ x, f x = …` where `f` is a
//!   bound variable), `abs`, `Finset`, or modular arithmetic (our AST has
//!   no `mod`).
//!
//! Under those constraints the in-scope subset of the 488-file corpus
//! is roughly 30-50 problems. Each fixture records the source filename
//! and the verbatim Lean statement for cross-reference.
//!
//! ## Running
//!
//! ```bash
//! cargo run -p symthaea-lean-bridge --example prove_minif2f_curated
//!
//! # With Lake verification (slower — resolves Mathlib):
//! LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_minif2f_curated
//! ```
//!
//! Output:
//! - stdout: CSV with one row per fixture (fixture,source,fragment,tactic,
//!   bytes,lake_check,z3_result,notes)
//! - stderr: summary counts
//! - files: one `.lean` per fixture in `proofs/minif2f_curated/`

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::Duration;

use symthaea_core::hdc::conjecture_engine::detect_z3_path;
use symthaea_core::hdc::fol_ext_smt::{detect_fragment, encode_as_query};
use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt as F, NumericType, Term};
use symthaea_lean_bridge::fol_ext_bridge::render_fol_ext_file;

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn v(name: &str) -> Term {
    Term::var(name)
}

fn i(n: i64) -> Term {
    Term::IntLit(n)
}

fn rat(p: i64, q: i64) -> Term {
    Term::rat(p, q)
}

/// Wrap a body in `∀ x : T, body` for each (name, ty) pair, right-to-left.
fn forall_all(binders: &[(&str, NumericType)], body: F) -> F {
    let mut out = body;
    for (name, ty) in binders.iter().rev() {
        out = F::forall(name, *ty, out);
    }
    out
}

/// Chain a list of hypotheses into `h1 → h2 → … → goal`.
fn implies_chain(hyps: Vec<F>, goal: F) -> F {
    let mut out = goal;
    for h in hyps.into_iter().rev() {
        out = h.implies(out);
    }
    out
}

struct Fixture {
    name: &'static str,
    source: &'static str,
    original: &'static str,
    category: Category,
    goal: F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Category {
    LinearReal,
    LinearInt,
    PolynomialIdentity,
    PolynomialInequality,
    PolynomialSystem,
    ClosedFormRational,
    NumberTheoryInt,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::LinearReal => "linear_real",
            Category::LinearInt => "linear_int",
            Category::PolynomialIdentity => "polynomial_identity",
            Category::PolynomialInequality => "polynomial_inequality",
            Category::PolynomialSystem => "polynomial_system",
            Category::ClosedFormRational => "closed_form_rational",
            Category::NumberTheoryInt => "numbertheory_int",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Fixtures — hand-translated from the miniF2F-v2 corpus
// ────────────────────────────────────────────────────────────────────────
//
// Convention: each fixture name matches the upstream file name so the
// translation can be cross-checked by diffing against
// `data/benchmarks/minif2f/MiniF2F/{Valid,Test}/<name>.lean`.

fn fixtures() -> Vec<Fixture> {
    let r = NumericType::Real;
    let z = NumericType::Int;
    let n = NumericType::Nat;

    vec![
        // ═══ Linear arithmetic over ℝ (linarith target) ═══════════════
        Fixture {
            name: "mathd_algebra_109",
            source: "Valid/mathd_algebra_109.lean",
            original: "(a b : ℝ) (h₀ : 3a + 2b = 12) (h₁ : a = 4) : b = 0",
            category: Category::LinearReal,
            goal: forall_all(
                &[("a", r), ("b", r)],
                implies_chain(
                    vec![
                        F::eq(i(3).mul(v("a")).add(i(2).mul(v("b"))), i(12)),
                        F::eq(v("a"), i(4)),
                    ],
                    F::eq(v("b"), i(0)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_119",
            source: "Valid/mathd_algebra_119.lean",
            original: "(d e : ℝ) (h₀ : 2d = 17e - 8) (h₁ : 2e = d - 9) : e = 2",
            category: Category::LinearReal,
            goal: forall_all(
                &[("d", r), ("e", r)],
                implies_chain(
                    vec![
                        F::eq(i(2).mul(v("d")), i(17).mul(v("e")).sub(i(8))),
                        F::eq(i(2).mul(v("e")), v("d").sub(i(9))),
                    ],
                    F::eq(v("e"), i(2)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_126",
            source: "Valid/mathd_algebra_126.lean",
            original: "(x y : ℝ) (h₀ : 2*3 = x - 9) (h₁ : 2*-5 = y + 1) : x = 15 ∧ y = -11",
            category: Category::LinearReal,
            goal: forall_all(
                &[("x", r), ("y", r)],
                implies_chain(
                    vec![
                        F::eq(i(2).mul(i(3)), v("x").sub(i(9))),
                        F::eq(i(2).mul(i(-5)), v("y").add(i(1))),
                    ],
                    F::eq(v("x"), i(15)).and(F::eq(v("y"), i(-11))),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_142",
            source: "Test/mathd_algebra_142.lean",
            original: "(m b : ℝ) (h₀ : 7m + b = -1) (h₁ : -m + b = 7) : m + b = 5",
            category: Category::LinearReal,
            goal: forall_all(
                &[("m", r), ("b", r)],
                implies_chain(
                    vec![
                        F::eq(i(7).mul(v("m")).add(v("b")), i(-1)),
                        F::eq(v("m").neg().add(v("b")), i(7)),
                    ],
                    F::eq(v("m").add(v("b")), i(5)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_160",
            source: "Test/mathd_algebra_160.lean",
            original: "(n x : ℝ) (h₀ : n + x = 97) (h₁ : n + 5x = 265) : n + 2x = 139",
            category: Category::LinearReal,
            goal: forall_all(
                &[("n", r), ("x", r)],
                implies_chain(
                    vec![
                        F::eq(v("n").add(v("x")), i(97)),
                        F::eq(v("n").add(i(5).mul(v("x"))), i(265)),
                    ],
                    F::eq(v("n").add(i(2).mul(v("x"))), i(139)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_329",
            source: "Test/mathd_algebra_329.lean",
            original: "(x y : ℝ) (h₀ : 3y = x) (h₁ : 2x + 5y = 11) : x + y = 4",
            category: Category::LinearReal,
            goal: forall_all(
                &[("x", r), ("y", r)],
                implies_chain(
                    vec![
                        F::eq(i(3).mul(v("y")), v("x")),
                        F::eq(i(2).mul(v("x")).add(i(5).mul(v("y"))), i(11)),
                    ],
                    F::eq(v("x").add(v("y")), i(4)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_354",
            source: "Test/mathd_algebra_354.lean",
            original: "(a d : ℝ) (h₀ : a + 6d = 30) (h₁ : a + 10d = 60) : a + 20d = 135",
            category: Category::LinearReal,
            goal: forall_all(
                &[("a", r), ("d", r)],
                implies_chain(
                    vec![
                        F::eq(v("a").add(i(6).mul(v("d"))), i(30)),
                        F::eq(v("a").add(i(10).mul(v("d"))), i(60)),
                    ],
                    F::eq(v("a").add(i(20).mul(v("d"))), i(135)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_359",
            source: "Test/mathd_algebra_359.lean",
            original: "(y : ℝ) (h₀ : y + 6 + y = 2*12) : y = 9",
            category: Category::LinearReal,
            goal: F::forall(
                "y",
                r,
                F::eq(v("y").add(i(6)).add(v("y")), i(2).mul(i(12))).implies(F::eq(v("y"), i(9))),
            ),
        },
        Fixture {
            name: "mathd_algebra_388",
            source: "Test/mathd_algebra_388.lean",
            original: "(x y z : ℝ) (h₀ : 3x+4y-12z = 10) (h₁ : -2x-3y+9z = -4) : x = 14",
            category: Category::LinearReal,
            goal: forall_all(
                &[("x", r), ("y", r), ("z", r)],
                implies_chain(
                    vec![
                        F::eq(
                            i(3).mul(v("x"))
                                .add(i(4).mul(v("y")))
                                .sub(i(12).mul(v("z"))),
                            i(10),
                        ),
                        F::eq(
                            i(-2)
                                .mul(v("x"))
                                .sub(i(3).mul(v("y")))
                                .add(i(9).mul(v("z"))),
                            i(-4),
                        ),
                    ],
                    F::eq(v("x"), i(14)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_455",
            source: "Valid/mathd_algebra_455.lean",
            original: "(x : ℝ) (h₀ : 2*(2*(2*(2*x))) = 48) : x = 3",
            category: Category::LinearReal,
            goal: F::forall(
                "x",
                r,
                F::eq(i(2).mul(i(2).mul(i(2).mul(i(2).mul(v("x"))))), i(48))
                    .implies(F::eq(v("x"), i(3))),
            ),
        },
        Fixture {
            name: "mathd_algebra_51",
            source: "Valid/mathd_algebra_51.lean",
            original: "(a b : ℝ) (h₀ : 0<a ∧ 0<b) (h₁ : a+b=35) (h₂ : a = 2/5 * b) : b - a = 15",
            category: Category::LinearReal,
            goal: forall_all(
                &[("a", r), ("b", r)],
                implies_chain(
                    vec![
                        F::lt(i(0), v("a")),
                        F::lt(i(0), v("b")),
                        F::eq(v("a").add(v("b")), i(35)),
                        F::eq(v("a"), rat(2, 5).mul(v("b"))),
                    ],
                    F::eq(v("b").sub(v("a")), i(15)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_24",
            source: "Test/mathd_algebra_24.lean",
            original: "(x : ℝ) (h₀ : x/50 = 40) : x = 2000",
            category: Category::LinearReal,
            goal: F::forall(
                "x",
                r,
                F::eq(v("x").div(i(50)), i(40)).implies(F::eq(v("x"), i(2000))),
            ),
        },
        // ═══ Closed-form rational (norm_num / ring target) ═══════════
        Fixture {
            name: "mathd_algebra_190",
            source: "Valid/mathd_algebra_190.lean",
            original: "((3:ℝ)/8 + 7/8) / (4/5) = 25/16",
            category: Category::ClosedFormRational,
            goal: F::eq(rat(3, 8).add(rat(7, 8)).div(rat(4, 5)), rat(25, 16)),
        },
        Fixture {
            name: "mathd_algebra_462",
            source: "Valid/mathd_algebra_462.lean",
            original: "((1:ℚ)/2 + 1/3) * (1/2 - 1/3) = 5/36 (translated as ℝ)",
            category: Category::ClosedFormRational,
            goal: F::eq(
                rat(1, 2).add(rat(1, 3)).mul(rat(1, 2).sub(rat(1, 3))),
                rat(5, 36),
            ),
        },
        Fixture {
            name: "mathd_algebra_304",
            source: "Test/mathd_algebra_304.lean",
            original: "91^2 = 8281 (ℕ; translated as ℤ)",
            category: Category::ClosedFormRational,
            goal: F::eq(i(91).pow(2), i(8281)),
        },
        Fixture {
            name: "mathd_algebra_104",
            source: "Valid/mathd_algebra_104.lean",
            original: "(x : ℝ) (h₀ : 125/8 = x/12) : x = 375/2",
            category: Category::ClosedFormRational,
            goal: F::forall(
                "x",
                r,
                F::eq(rat(125, 8), v("x").div(i(12))).implies(F::eq(v("x"), rat(375, 2))),
            ),
        },
        Fixture {
            name: "mathd_algebra_55",
            source: "Valid/mathd_algebra_55.lean",
            original:
                "(q p : ℝ) (h₀ : q = 2-4+6-8+10-12+14) (h₁ : p = 3-6+9-12+15-18+21) : q/p = 2/3",
            category: Category::ClosedFormRational,
            goal: forall_all(
                &[("q", r), ("p", r)],
                implies_chain(
                    vec![
                        F::eq(
                            v("q"),
                            i(2).sub(i(4))
                                .add(i(6))
                                .sub(i(8))
                                .add(i(10))
                                .sub(i(12))
                                .add(i(14)),
                        ),
                        F::eq(
                            v("p"),
                            i(3).sub(i(6))
                                .add(i(9))
                                .sub(i(12))
                                .add(i(15))
                                .sub(i(18))
                                .add(i(21)),
                        ),
                    ],
                    F::eq(v("q").div(v("p")), rat(2, 3)),
                ),
            ),
        },
        // ═══ Polynomial identities (ring / nlinarith target) ═════════
        Fixture {
            name: "mathd_algebra_176",
            source: "Test/mathd_algebra_176.lean",
            original: "(x : ℝ) : (x+1)^2 * x = x^3 + 2x^2 + x",
            category: Category::PolynomialIdentity,
            goal: F::forall(
                "x",
                r,
                F::eq(
                    v("x").add(i(1)).pow(2).mul(v("x")),
                    v("x").pow(3).add(i(2).mul(v("x").pow(2))).add(v("x")),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_107",
            source: "Test/mathd_algebra_107.lean",
            original: "(x y : ℝ) (h₀ : x²+8x+y²-6y = 0) : (x+4)² + (y-3)² = 5²",
            category: Category::PolynomialIdentity,
            goal: forall_all(
                &[("x", r), ("y", r)],
                F::eq(
                    v("x")
                        .pow(2)
                        .add(i(8).mul(v("x")))
                        .add(v("y").pow(2))
                        .sub(i(6).mul(v("y"))),
                    i(0),
                )
                .implies(F::eq(
                    v("x").add(i(4)).pow(2).add(v("y").sub(i(3)).pow(2)),
                    i(5).pow(2),
                )),
            ),
        },
        Fixture {
            name: "mathd_algebra_568",
            source: "Valid/mathd_algebra_568.lean",
            original: "(a : ℝ) : (a-1)(a+1)(a+2) - (a-2)(a+1) = a^3 + a^2",
            category: Category::PolynomialIdentity,
            goal: F::forall(
                "a",
                r,
                F::eq(
                    v("a")
                        .sub(i(1))
                        .mul(v("a").add(i(1)))
                        .mul(v("a").add(i(2)))
                        .sub(v("a").sub(i(2)).mul(v("a").add(i(1)))),
                    v("a").pow(3).add(v("a").pow(2)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_182",
            source: "Valid/mathd_algebra_182.lean",
            original: "7 * (3y + 2) = 21y + 14 (original ℂ; translated as ℝ identity)",
            category: Category::PolynomialIdentity,
            goal: F::forall(
                "y",
                r,
                F::eq(
                    i(7).mul(i(3).mul(v("y")).add(i(2))),
                    i(21).mul(v("y")).add(i(14)),
                ),
            ),
        },
        // ═══ Polynomial inequalities (nlinarith target) ══════════════
        Fixture {
            name: "mathd_algebra_101",
            source: "Valid/mathd_algebra_101.lean",
            original: "(x : ℝ) (h₀ : x^2 - 5x - 4 ≤ 10) : x ≥ -2 ∧ x ≤ 7",
            category: Category::PolynomialInequality,
            goal: F::forall(
                "x",
                r,
                F::le(v("x").pow(2).sub(i(5).mul(v("x"))).sub(i(4)), i(10))
                    .implies(F::le(i(-2), v("x")).and(F::le(v("x"), i(7)))),
            ),
        },
        Fixture {
            name: "mathd_algebra_113",
            source: "Test/mathd_algebra_113.lean",
            original: "(x : ℝ) : x² - 14x + 3 ≥ 7² - 14*7 + 3",
            category: Category::PolynomialInequality,
            goal: F::forall(
                "x",
                r,
                F::le(
                    i(7).pow(2).sub(i(14).mul(i(7))).add(i(3)),
                    v("x").pow(2).sub(i(14).mul(v("x"))).add(i(3)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_410",
            source: "Valid/mathd_algebra_410.lean",
            original: "(x y : ℝ) (h₀ : y = x² - 6x + 13) : 4 ≤ y",
            category: Category::PolynomialInequality,
            goal: forall_all(
                &[("x", r), ("y", r)],
                F::eq(v("y"), v("x").pow(2).sub(i(6).mul(v("x"))).add(i(13)))
                    .implies(F::le(i(4), v("y"))),
            ),
        },
        // ═══ Nonlinear systems (nlinarith target) ════════════════════
        Fixture {
            name: "mathd_algebra_37",
            source: "Valid/mathd_algebra_37.lean",
            original: "(x y : ℝ) (h₀ : x+y=7) (h₁ : 3x+y=45) : x² - y² = 217",
            category: Category::PolynomialSystem,
            goal: forall_all(
                &[("x", r), ("y", r)],
                implies_chain(
                    vec![
                        F::eq(v("x").add(v("y")), i(7)),
                        F::eq(i(3).mul(v("x")).add(v("y")), i(45)),
                    ],
                    F::eq(v("x").pow(2).sub(v("y").pow(2)), i(217)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_141",
            source: "Test/mathd_algebra_141.lean",
            original: "(a b : ℝ) (h₁ : ab=180) (h₂ : 2(a+b)=54) : a² + b² = 369",
            category: Category::PolynomialSystem,
            goal: forall_all(
                &[("a", r), ("b", r)],
                implies_chain(
                    vec![
                        F::eq(v("a").mul(v("b")), i(180)),
                        F::eq(i(2).mul(v("a").add(v("b"))), i(54)),
                    ],
                    F::eq(v("a").pow(2).add(v("b").pow(2)), i(369)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_234",
            source: "Valid/mathd_algebra_234.lean",
            original: "(d : ℝ) (h₀ : 27/125 * d = 9/25) : 3/5 * d^3 = 25/9",
            category: Category::PolynomialSystem,
            goal: F::forall(
                "d",
                r,
                F::eq(rat(27, 125).mul(v("d")), rat(9, 25))
                    .implies(F::eq(rat(3, 5).mul(v("d").pow(3)), rat(25, 9))),
            ),
        },
        Fixture {
            name: "mathd_algebra_338",
            source: "Test/mathd_algebra_338.lean",
            original: "(a b c : ℝ) 3a+b+c=-3, a+3b+c=9, a+b+3c=19 → abc=-56",
            category: Category::PolynomialSystem,
            goal: forall_all(
                &[("a", r), ("b", r), ("c", r)],
                implies_chain(
                    vec![
                        F::eq(i(3).mul(v("a")).add(v("b")).add(v("c")), i(-3)),
                        F::eq(v("a").add(i(3).mul(v("b"))).add(v("c")), i(9)),
                        F::eq(v("a").add(v("b")).add(i(3).mul(v("c"))), i(19)),
                    ],
                    F::eq(v("a").mul(v("b")).mul(v("c")), i(-56)),
                ),
            ),
        },
        // ═══ Number theory over ℤ/ℕ (omega / nlinarith target) ═══════
        Fixture {
            name: "mathd_numbertheory_136",
            source: "Valid/mathd_numbertheory_136.lean",
            original: "(n : ℕ) (h₀ : 123n + 17 = 39500) : n = 321",
            category: Category::NumberTheoryInt,
            goal: F::forall(
                "n",
                n,
                F::eq(i(123).mul(v("n")).add(i(17)), i(39500)).implies(F::eq(v("n"), i(321))),
            ),
        },
        Fixture {
            name: "mathd_numbertheory_326",
            source: "Valid/mathd_numbertheory_326.lean",
            original: "(n : ℤ) (h₀ : (n-1)*n*(n+1) = 720) : n + 1 = 10",
            category: Category::NumberTheoryInt,
            // Note: cubic constraint uniquely determines n ∈ {-10, 9}, so
            // the conclusion `n + 1 = 10` isn't globally true; the upstream
            // statement is equivalent to "among Int roots of n(n-1)(n+1)=720,
            // the positive one is 9". Z3 should still classify this as
            // sat (there exists n = -10 where n+1 = -9 ≠ 10) — this is a
            // known weakness of the theorem, not the pipeline. We include
            // it to measure the failure mode honestly.
            goal: F::forall(
                "n",
                z,
                F::eq(v("n").sub(i(1)).mul(v("n")).mul(v("n").add(i(1))), i(720))
                    .implies(F::eq(v("n").add(i(1)), i(10))),
            ),
        },
        Fixture {
            name: "mathd_numbertheory_48",
            source: "Valid/mathd_numbertheory_48.lean",
            original: "(b : ℕ) (h₀ : 0 < b) (h₁ : 3b² + 2b + 1 = 57) : b = 4",
            category: Category::NumberTheoryInt,
            goal: F::forall(
                "b",
                n,
                implies_chain(
                    vec![
                        F::lt(i(0), v("b")),
                        F::eq(
                            i(3).mul(v("b").pow(2)).add(i(2).mul(v("b"))).add(i(1)),
                            i(57),
                        ),
                    ],
                    F::eq(v("b"), i(4)),
                ),
            ),
        },
        Fixture {
            name: "mathd_algebra_123",
            source: "Valid/mathd_algebra_123.lean",
            original: "(a b : ℕ) (0<a ∧ 0<b) (a+b=20) (a=3b) : a - b = 10",
            category: Category::NumberTheoryInt,
            // ℕ subtraction is truncated, but omega handles this correctly
            // when we can derive a ≥ b from the hypotheses (here a = 3b ≥ b).
            goal: forall_all(
                &[("a", n), ("b", n)],
                implies_chain(
                    vec![
                        F::lt(i(0), v("a")),
                        F::lt(i(0), v("b")),
                        F::eq(v("a").add(v("b")), i(20)),
                        F::eq(v("a"), i(3).mul(v("b"))),
                    ],
                    F::eq(v("a").sub(v("b")), i(10)),
                ),
            ),
        },
    ]
}

// ────────────────────────────────────────────────────────────────────────
// Z3 invocation
// ────────────────────────────────────────────────────────────────────────

/// Run Z3 on the SMT-LIB2 encoding of `goal` with a short timeout. Returns
/// one of "unsat" / "sat" / "unknown" / "timeout" / "z3_missing" / "error".
fn z3_check(goal: &F, timeout: Duration) -> String {
    let Some(z3_path) = detect_z3_path() else {
        return "z3_missing".to_string();
    };

    let smt = encode_as_query(goal);

    let mut child = match Command::new(&z3_path)
        .args(["-in", "-smt2", "-T:10"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return "error".to_string(),
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if stdin.write_all(smt.as_bytes()).is_err() {
            let _ = child.kill();
            return "error".to_string();
        }
    }

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return "timeout".to_string();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return "error".to_string(),
        }
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => return "error".to_string(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap_or("").trim();
    match first_line {
        "unsat" => "unsat".to_string(),
        "sat" => "sat".to_string(),
        "unknown" => "unknown".to_string(),
        other if other.is_empty() => "error".to_string(),
        other => other.to_string(),
    }
}

// ────────────────────────────────────────────────────────────────────────
// Lake invocation
// ────────────────────────────────────────────────────────────────────────

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

fn verify_with_lake(lean_file: &Path, project_dir: &Path) -> Result<bool, String> {
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

// ────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let out_dir = env::var("MINIF2F_CURATED_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("proofs/minif2f_curated"));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("# mkdir {} failed: {}", out_dir.display(), e);
        return ExitCode::from(2);
    }

    let use_lake = lake_env_available();
    let project_dir = PathBuf::from("lean-proofs/phase2");
    let z3_timeout = Duration::from_secs(10);

    println!("fixture,source,category,fragment,tactic,bytes,z3_result,lake_check,note");

    let all = fixtures();
    let total = all.len();
    let mut lake_accepted = 0usize;
    let mut lake_rejected = 0usize;
    let mut z3_unsat = 0usize;
    let mut z3_sat = 0usize;
    let mut z3_unknown = 0usize;
    let mut z3_timeout_count = 0usize;

    for fx in &all {
        let file_contents = render_fol_ext_file(fx.name, &fx.goal);
        let file_path = out_dir.join(format!("{}.lean", fx.name));
        let bytes = file_contents.len();

        if let Err(e) = fs::write(&file_path, &file_contents) {
            eprintln!("# write failed for {}: {}", file_path.display(), e);
            continue;
        }

        let fragment = detect_fragment(&fx.goal);
        let tactic = fragment.suggested_lean_tactic();

        // Z3 check — always attempted if z3 is on PATH.
        let z3_result = z3_check(&fx.goal, z3_timeout);
        match z3_result.as_str() {
            "unsat" => z3_unsat += 1,
            "sat" => z3_sat += 1,
            "unknown" => z3_unknown += 1,
            "timeout" => z3_timeout_count += 1,
            _ => {}
        }

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

        // CSV note field: squash commas and newlines so a naive CSV parser
        // still partitions cleanly.
        let note = fx.original.replace(',', ";").replace('\n', " ");
        println!(
            "{},{},{},{},{},{},{},{},{}",
            fx.name,
            fx.source,
            fx.category.label(),
            fragment.logic_name(),
            tactic,
            bytes,
            z3_result,
            lake_label,
            note,
        );
    }

    let mode = if use_lake { "verify" } else { "emit-only" };
    eprintln!(
        "# summary: total={} z3[unsat={} sat={} unknown={} timeout={}] lake[accepted={} rejected={}] mode={}",
        total, z3_unsat, z3_sat, z3_unknown, z3_timeout_count, lake_accepted, lake_rejected, mode
    );
    if use_lake {
        let accept_pct = (lake_accepted as f64 / total as f64) * 100.0;
        eprintln!(
            "# Phase 3 (b) accept rate: {}/{} = {:.1}% (target: 15-30%)",
            lake_accepted, total, accept_pct
        );
    } else {
        eprintln!("# Set LAKE_ENV=1 to measure lake accept rate — this run emitted only.");
    }

    // Honest measurement: exit 0 regardless. The rate IS the result; we
    // don't want CI to bounce on individual problem rejections.
    ExitCode::SUCCESS
}
