// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration: formal mathematical verification of HDC-LTC stability
//! invariants using the Z3 solver bridge.
//!
//! Specifically, verifies that the state update $\|h(t)\| \leq 5.0$ holds.

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

#[test]
fn test_ltc_stability_invariants() {
    if !z3_available() {
        eprintln!("z3 not installed; skipping stability invariant test");
        return;
    }

    // We verify the analytical invariant for state updates:
    // Given state update:
    //   h_i(t+dt) = (1 - \sigma_i) * h_i(t) + \sigma_i * x_inf_i
    //
    // Where:
    //   0 <= \sigma_i <= 1
    //   -1 <= x_inf_i <= 1  (bounded by tanh / sigmoid activation times gate in [0, 1])
    //
    // We want to verify that if the previous state is bounded:
    //   -5.0 <= h_i(t) <= 5.0
    //
    // Then the updated state remains bounded:
    //   -5.0 <= h_i(t+dt) <= 5.0
    //
    // We formulate this as an implication:
    //   (0 <= sigma <= 1) -> (-1 <= x_inf <= 1) -> (-5.0 <= h <= 5.0) ->
    //     (-5.0 <= (1 - sigma) * h + sigma * x_inf <= 5.0)

    let h = Term::var("h");
    let sigma = Term::var("sigma");
    let x_inf = Term::var("x_inf");

    let updated = Term::IntLit(1)
        .sub(sigma.clone())
        .mul(h.clone())
        .add(sigma.clone().mul(x_inf.clone()));

    // Hypotheses
    let h1 = FolFormulaExt::le(Term::real(0.0), sigma.clone());
    let h2 = FolFormulaExt::le(sigma.clone(), Term::real(1.0));
    let h3 = FolFormulaExt::le(Term::real(-1.0), x_inf.clone());
    let h4 = FolFormulaExt::le(x_inf.clone(), Term::real(1.0));
    let h5 = FolFormulaExt::le(Term::real(-5.0), h.clone());
    let h6 = FolFormulaExt::le(h.clone(), Term::real(5.0));

    // Goal: -5.0 <= updated <= 5.0
    let goal_lower = FolFormulaExt::le(Term::real(-5.0), updated.clone());
    let goal_upper = FolFormulaExt::le(updated, Term::real(5.0));
    let goal = goal_lower.and(goal_upper);

    // Build the full formula: forall h, sigma, x_inf: Real, (h1 & h2 & h3 & h4 & h5 & h6) -> goal
    let phi = FolFormulaExt::forall(
        "h",
        NumericType::Real,
        FolFormulaExt::forall(
            "sigma",
            NumericType::Real,
            FolFormulaExt::forall(
                "x_inf",
                NumericType::Real,
                h1.and(h2).and(h3).and(h4).and(h5).and(h6).implies(goal),
            ),
        ),
    );

    let smt = encode_as_query(&phi);
    let result = run_z3(&smt).expect("z3 execution failed");
    let last = result.trim().lines().last().unwrap_or("").trim();

    assert_eq!(
        last, "unsat",
        "Verification of LTC stability invariant failed! Got: {}\nSMT query:\n{}",
        last, smt
    );
    println!("LTC stability invariant successfully verified by Z3!");
}
