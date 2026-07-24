// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Audit a real Lean `#print axioms` output file against an axiom policy.
//!
//! Usage:
//!   audit_print_axioms <lean-output-file> [constitutional|classical]
//!
//! Reads the captured output of `lean file.lean` (which must contain a
//! `#print axioms <thm>` command), applies the axiom-provenance gate, prints a
//! verdict, and exits 0 (clean) or 1 (violations). This is the live end-to-end
//! bridge from real Lean to [`symthaea_proof_audit`].

use std::process::ExitCode;
use symthaea_proof_audit::{AxiomPolicy, AxiomReport, audit};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: audit_print_axioms <lean-output-file> [constitutional|classical]");
        return ExitCode::FAILURE;
    };
    let policy = match args.get(2).map(String::as_str) {
        Some("classical") => AxiomPolicy::classical(),
        _ => AxiomPolicy::constitutional(),
    };

    let output = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let report = match AxiomReport::parse(&output) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid Lean evidence: {error}");
            return ExitCode::FAILURE;
        }
    };
    let verdict = audit(&report, &policy);

    if verdict.is_clean() {
        println!(
            "CLEAN  '{}' — reduces to the declared base (axioms: {:?})",
            report.theorem, report.axioms
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "REJECT '{}' — {} violation(s): {:?}",
            report.theorem,
            verdict.violations.len(),
            verdict.violations
        );
        ExitCode::FAILURE
    }
}
