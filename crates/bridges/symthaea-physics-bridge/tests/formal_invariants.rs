// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration test: run `verify_invariants_formal` end-to-end and assert
//! every polynomial invariant returns `unsat`.
//!
//! Skips automatically when:
//! - `z3` is not on PATH (CI without the SMT toolchain → `skipped`, not failure)
//! - `FORMAL_INVARIANT_TESTS` env var is set to `0` (explicit opt-out)
//!
//! Runs ~2s under warm cargo cache once Z3 is available. Targets the Phase 1
//! paper's Tier B claims: if this test fails, the paper is wrong.

use std::process::Command;

fn z3_available() -> bool {
    Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn opt_out() -> bool {
    std::env::var("FORMAL_INVARIANT_TESTS").as_deref() == Ok("0")
}

#[test]
fn all_polynomial_invariants_formally_proven() {
    if opt_out() {
        eprintln!("FORMAL_INVARIANT_TESTS=0 set; skipping.");
        return;
    }
    if !z3_available() {
        eprintln!("z3 not on PATH; skipping formal-invariant verification.");
        eprintln!("To run this test, enter `nix develop` (the flake adds z3 + lean4).");
        return;
    }

    // Build + run the verify_invariants_formal example. It emits CSV to
    // stdout with a `formally_proven` column per row.
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "-p",
            "symthaea-physics-bridge",
            "--example",
            "verify_invariants_formal",
        ])
        .output()
        .expect("failed to invoke cargo run");

    assert!(
        output.status.success(),
        "verify_invariants_formal example exited non-zero:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines.len() >= 2,
        "expected CSV header + data rows, got: {}",
        stdout
    );

    let header = lines[0];
    assert!(
        header.starts_with("problem,"),
        "unexpected header: {}",
        header
    );
    let proven_col = header
        .split(',')
        .position(|c| c == "formally_proven")
        .expect("formally_proven column not present");

    let mut proven_count = 0usize;
    let mut skipped_count = 0usize;
    let mut rejected: Vec<String> = Vec::new();

    for row in &lines[1..] {
        if row.starts_with('#') || row.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = row.split(',').collect();
        if cells.len() <= proven_col {
            continue;
        }
        let name = cells[0];
        let proven = cells[proven_col];
        match proven {
            "true" => proven_count += 1,
            // The verifier declares transcendental invariants (Lotka-Volterra)
            // out-of-scope with `skipped,false`. That's honest and expected.
            "false" if cells.get(3) == Some(&"skipped") => skipped_count += 1,
            _ => rejected.push(format!("{}={}", name, proven)),
        }
    }

    assert!(
        rejected.is_empty(),
        "some invariants failed formal verification: {:?}",
        rejected
    );
    assert!(
        proven_count >= 9,
        "expected ≥9 formally-proven invariants, got {} (skipped={})",
        proven_count,
        skipped_count
    );
    eprintln!(
        "formal_invariants test: {} proven, {} skipped (transcendental, honest)",
        proven_count, skipped_count
    );
}
