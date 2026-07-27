// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lean-verified arithmetic gate — the *correctness* counterpart to the
//! shape-only unit tests in `fol_ext_bridge.rs`.
//!
//! The unit tests assert the emitted Lean *contains* the right tactic
//! (`omega`, `nlinarith`, …) and no `sorry`. They do NOT prove the proof
//! actually type-checks. This gate closes that gap: it renders a battery of
//! known-true arithmetic goals and runs each through the real Lean 4 toolchain
//! (`runner::check_with_lean4`), asserting Lean *accepts* it.
//!
//! ## Opt-in — never breaks a Lean-less CI
//!
//! The arithmetic files emit `import Mathlib.Tactic`, so a bare `lean` without
//! a Mathlib-resolved environment would *reject* them. To avoid false failures,
//! the assertion runs only when the operator explicitly signals a
//! Mathlib-resolved Lean is available:
//!
//! ```bash
//! # inside a Lake project with Mathlib on LEAN_PATH:
//! SYMTHAEA_LEAN_MATHLIB_GATE=1 cargo test -p symthaea-lean-bridge \
//!     --test lean_verified_arith_gate
//! ```
//!
//! Without that env var — or if `lean` is not installed — the gate prints a
//! skip notice and passes as a no-op. This is intentional: a toolchain-gated
//! correctness check should skip, not fail, where its toolchain is absent (see
//! `docs/minif2f-v2-scope.md`, "Concrete next step to close Phase 1").

use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};
use symthaea_lean_bridge::fol_ext_bridge::render_fol_ext_file;
use symthaea_lean_bridge::runner::{CheckOutcome, check_with_lean4};

/// A known-true arithmetic goal and the fragment/tactic it should route to.
struct GateCase {
    theorem: &'static str,
    build: fn() -> FolFormulaExt,
}

fn battery() -> Vec<GateCase> {
    vec![
        // ∀ n : ℤ, n < n + 1        → LIA → omega
        GateCase {
            theorem: "gate_succ_gt",
            build: || {
                FolFormulaExt::forall(
                    "n",
                    NumericType::Int,
                    FolFormulaExt::lt(Term::var("n"), Term::var("n").add(Term::IntLit(1))),
                )
            },
        },
        // ∀ x : ℝ, 0 ≤ x * x        → NRA → nlinarith [sq_nonneg]
        GateCase {
            theorem: "gate_sq_nonneg",
            build: || {
                FolFormulaExt::forall(
                    "x",
                    NumericType::Real,
                    FolFormulaExt::le(Term::IntLit(0), Term::var("x").mul(Term::var("x"))),
                )
            },
        },
        // ∀ x : ℝ, 0 ≤ x → x ≤ x + 1 → LRA → linarith
        GateCase {
            theorem: "gate_mono_succ",
            build: || {
                FolFormulaExt::forall(
                    "x",
                    NumericType::Real,
                    FolFormulaExt::le(Term::IntLit(0), Term::var("x")).implies(FolFormulaExt::le(
                        Term::var("x"),
                        Term::var("x").add(Term::IntLit(1)),
                    )),
                )
            },
        },
        // ∀ x : ℝ, x + 0 = x         → ring / simp
        GateCase {
            theorem: "gate_add_zero",
            build: || {
                FolFormulaExt::forall(
                    "x",
                    NumericType::Real,
                    FolFormulaExt::eq(Term::var("x").add(Term::IntLit(0)), Term::var("x")),
                )
            },
        },
    ]
}

/// True when the operator has signalled a Mathlib-resolved Lean is present.
fn mathlib_gate_enabled() -> bool {
    std::env::var("SYMTHAEA_LEAN_MATHLIB_GATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[test]
fn arithmetic_goals_are_lean_accepted() {
    // Every emitted file must at minimum be sorry-free — that is checkable
    // without Lean and always runs.
    let cases = battery();
    let mut rendered: Vec<(String, String)> = Vec::new();
    for c in &cases {
        let phi = (c.build)();
        let file = render_fol_ext_file(c.theorem, &phi);
        assert!(
            !file.contains("sorry"),
            "{} emitted a sorry-tagged (unproved) proof:\n{}",
            c.theorem,
            file
        );
        assert!(
            file.contains("import Mathlib.Tactic"),
            "{} arithmetic file missing Mathlib import",
            c.theorem
        );
        rendered.push((c.theorem.to_string(), file));
    }

    if !mathlib_gate_enabled() {
        eprintln!(
            "[lean-gate] SYMTHAEA_LEAN_MATHLIB_GATE not set — emitted {} sorry-free \
             arithmetic proofs but skipping external Lean verification. Set the env \
             var inside a Mathlib-resolved Lake project to enforce acceptance.",
            rendered.len()
        );
        return;
    }

    // Gate enabled: actually run each file through Lean.
    //
    // The directory name includes this process's PID so a local user can't
    // pre-stage a symlink at a guessable, stable path before the test runs
    // (a prior fixed `symthaea_lean_gate` name was guessable on a shared
    // multi-tenant host). File writes below additionally use
    // `create_new` (O_EXCL) rather than `fs::write`, so even a
    // same-PID-guessed pre-existing path/symlink causes a hard failure
    // instead of a silent write-through.
    let dir = std::env::temp_dir().join(format!("symthaea_lean_gate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let mut failures: Vec<String> = Vec::new();
    for (name, file) in &rendered {
        let path = dir.join(format!("{name}.lean"));
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create lean file (fails closed if the path is unexpectedly pre-occupied)");
            f.write_all(file.as_bytes()).expect("write lean file");
        }
        match check_with_lean4(&path) {
            CheckOutcome::Accepted => {}
            CheckOutcome::LeanNotInstalled => {
                eprintln!(
                    "[lean-gate] SYMTHAEA_LEAN_MATHLIB_GATE=1 but `lean` not found on \
                     PATH (or LEAN_PATH_BIN) — cannot verify; skipping."
                );
                return;
            }
            CheckOutcome::Rejected(err) => {
                failures.push(format!("{name}: REJECTED\n{err}"));
            }
            CheckOutcome::ProcessError(err) => {
                failures.push(format!("{name}: process error: {err}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Lean rejected {} of {} arithmetic proofs:\n\n{}",
        failures.len(),
        rendered.len(),
        failures.join("\n\n")
    );
}
