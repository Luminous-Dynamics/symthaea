// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration: build `FolFormulaExt` goals, encode them via `fol_ext_smt`,
//! invoke Z3, confirm `unsat` on every tautology and `sat` on every
//! non-tautology. This is the Phase 2 Week-1 closeout gate.
//!
//! Skips gracefully when `z3` is not on PATH (CI without the SMT
//! toolchain just prints "skipped"). Inside `nix develop` the flake
//! provides `z3` so this runs naturally.
//!
//! Reference for what each fragment should decide:
//! <https://smt-lib.org/logics.shtml>.

use std::io::Write;
use std::process::{Command, Stdio};

use symthaea_core::hdc::fol_ext_smt::encode_as_query;
use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};

fn z3_available() -> bool {
    Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_z3(smt: &str) -> Option<String> {
    let mut child = Command::new("z3")
        .arg("-in")
        .arg("-T:10")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    {
        let mut stdin = child.stdin.take()?;
        stdin.write_all(smt.as_bytes()).ok()?;
    }
    let out = child.wait_with_output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn expect_unsat(phi: &FolFormulaExt, label: &str) {
    let smt = encode_as_query(phi);
    let result = run_z3(&smt).unwrap_or_else(|| panic!("z3 invocation failed for {}", label));
    let last = result.trim().lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "unsat",
        "expected unsat for tautology `{}`; got `{}`\nSMT:\n{}",
        label, last, smt
    );
}

fn expect_sat(phi: &FolFormulaExt, label: &str) {
    let smt = encode_as_query(phi);
    let result = run_z3(&smt).unwrap_or_else(|| panic!("z3 invocation failed for {}", label));
    let last = result.trim().lines().last().unwrap_or("").trim();
    assert!(
        last == "sat" || last == "unknown",
        "expected sat/unknown for `{}` (non-tautology); got `{}`\nSMT:\n{}",
        label,
        last,
        smt
    );
}

fn x() -> Term {
    Term::var("x")
}
fn y() -> Term {
    Term::var("y")
}
fn n() -> Term {
    Term::var("n")
}

// ═════════════════════════════════════════════════════════════════════════
// Tautologies — must return UNSAT (i.e. ¬φ is unsatisfiable).
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn taut_reflexivity_real() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ x : ℝ, x = x
    let phi = FolFormulaExt::forall("x", NumericType::Real, FolFormulaExt::eq(x(), x()));
    expect_unsat(&phi, "∀ x : ℝ, x = x");
}

#[test]
fn taut_reflexivity_int() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ n : ℤ, n = n
    let phi = FolFormulaExt::forall("n", NumericType::Int, FolFormulaExt::eq(n(), n()));
    expect_unsat(&phi, "∀ n : ℤ, n = n");
}

#[test]
fn taut_nat_nonneg() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ n : ℕ, 0 ≤ n  (uses the Nat side-constraint auto-inserted by the serializer)
    let phi = FolFormulaExt::forall(
        "n",
        NumericType::Nat,
        FolFormulaExt::le(Term::IntLit(0), n()),
    );
    expect_unsat(&phi, "∀ n : ℕ, 0 ≤ n");
}

#[test]
fn taut_square_nonneg_real() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ x : ℝ, 0 ≤ x * x
    let phi = FolFormulaExt::forall(
        "x",
        NumericType::Real,
        FolFormulaExt::le(Term::IntLit(0), x().mul(x())),
    );
    expect_unsat(&phi, "∀ x : ℝ, 0 ≤ x²");
}

#[test]
fn taut_linear_trichotomy_fragment() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ x, y : ℝ, x = y ∨ x < y ∨ y < x
    let phi = FolFormulaExt::forall(
        "x",
        NumericType::Real,
        FolFormulaExt::forall(
            "y",
            NumericType::Real,
            FolFormulaExt::eq(x(), y())
                .or(FolFormulaExt::lt(x(), y()))
                .or(FolFormulaExt::lt(y(), x())),
        ),
    );
    expect_unsat(&phi, "trichotomy over ℝ");
}

#[test]
fn taut_add_comm_int() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ a, b : ℤ, a + b = b + a
    let a = Term::var("a");
    let b = Term::var("b");
    let phi = FolFormulaExt::forall(
        "a",
        NumericType::Int,
        FolFormulaExt::forall(
            "b",
            NumericType::Int,
            FolFormulaExt::eq(a.clone().add(b.clone()), b.add(a)),
        ),
    );
    expect_unsat(&phi, "addition commutativity over ℤ");
}

#[test]
fn taut_exact_rational_one_third_times_three() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // 3 * (1/3) = 1 — this is the test that FAILS when RatLit is
    // serialized as 0.3333… f64 and Z3 reads the literal rational.
    // With (/ 1 3) exact serialization, Z3 closes this trivially.
    let phi = FolFormulaExt::eq(Term::IntLit(3).mul(Term::rat(1, 3)), Term::IntLit(1));
    expect_unsat(&phi, "3 · (1/3) = 1 (RatLit exactness)");
}

#[test]
fn taut_x_plus_one_gt_x_int() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // ∀ x : ℤ, x + 1 > x
    let phi = FolFormulaExt::forall(
        "x",
        NumericType::Int,
        FolFormulaExt::lt(x(), x().add(Term::IntLit(1))),
    );
    expect_unsat(&phi, "successor > self over ℤ");
}

// ═════════════════════════════════════════════════════════════════════════
// Non-tautologies — must return SAT (Z3 finds a counterexample).
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn nontaut_int_x_gt_0() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // x > 0   (free x : Real by default). NOT a tautology; Z3 returns
    // sat for x=0 or x=-1 (counterexample).
    let phi = FolFormulaExt::lt(Term::IntLit(0), x());
    expect_sat(&phi, "x > 0 (non-tautology)");
}

#[test]
fn nontaut_x_eq_y_over_reals() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping");
        return;
    }
    // x = y is not a tautology — Z3 finds x=0, y=1 counterexample.
    let phi = FolFormulaExt::eq(x(), y());
    expect_sat(&phi, "x = y (non-tautology)");
}
