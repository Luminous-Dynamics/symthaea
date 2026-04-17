// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `prove_minif2f_v2` — honest scope-signal harness against the public
//! miniF2F-v2 Lean 4 corpus.
//!
//! **Read this first:** see `docs/minif2f-v2-scope.md`. Symthaea's Phase 1
//! Lean bridge targets classical propositional / FOL tautologies. miniF2F-v2
//! is almost entirely real-arithmetic and number-theoretic algebra. The
//! expected accept rate at Phase 1 is **near zero** and is a scope signal,
//! not a quality signal.
//!
//! ## What this harness does
//!
//! 1. Reads `MINIF2F_V2_DIR` and walks every `.lean` file.
//! 2. For each file, runs `lean <file>` to measure how many miniF2F
//!    statements even type-check as-is (baseline: if the upstream file is
//!    `theorem foo … := by sorry`, Lean accepts it with a warning, so this
//!    is mostly a sanity check that the environment works).
//! 3. Counts files in which our bridge could conceivably participate — i.e.
//!    files whose theorem body uses only `∧ ∨ → ¬ ⊤ ⊥` connectives over
//!    propositional variables. This is a STRICT upper bound on our Phase 1
//!    accept rate.
//! 4. Emits a CSV to stdout and a summary line to stderr.
//!
//! The filter in step 3 is intentionally crude (string-based, not a Lean
//! parser). It is designed to err on the side of *overcounting* in-scope
//! problems, so that if the number still comes out tiny (which it will),
//! the message is unambiguous.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use symthaea_lean_bridge::runner::{check_with_lean4, CheckOutcome};

fn data_dir() -> Option<PathBuf> {
    env::var("MINIF2F_V2_DIR").ok().map(PathBuf::from)
}

/// Very rough test: does the file's body contain only propositional-looking
/// content? Heuristic tokens that disqualify a problem from our Phase 1
/// scope: `ℝ`, `ℕ`, `ℤ`, `Nat`, `Int`, `Real`, `=`, `*`, `+`, `^`, digits,
/// `Finset`, `∑`, `∏`, `Nat.`, etc.
fn looks_propositional_only(source: &str) -> bool {
    let disqualifiers = [
        "ℝ", "ℕ", "ℤ", "ℚ", "Nat", "Int", "Real", "Rat", "Finset", "∑", "∏", " = ", "≤", "≥", "<",
        ">", " + ", " * ", " - ", " / ", "^",
    ];
    for d in &disqualifiers {
        if source.contains(d) {
            return false;
        }
    }
    // Must mention at least one `Prop` or `theorem ... : P …` form.
    source.contains("Prop") || source.contains("theorem")
}

fn walk_lean_files(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_lean_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("lean") {
            out.push(path);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    println!("file,path,size_bytes,appears_propositional,lean_accepted,notes");

    let dir = match data_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "# MINIF2F_V2_DIR not set. This harness is an honest scope \
                 test — see docs/minif2f-v2-scope.md for the expected outcome \
                 and the Phase 2 path. Emitting an empty CSV and exiting 0 so \
                 downstream tooling can distinguish 'no data' from 'panic'."
            );
            return ExitCode::SUCCESS;
        }
    };

    if !dir.exists() {
        eprintln!("# MINIF2F_V2_DIR={} does not exist.", dir.display());
        return ExitCode::SUCCESS;
    }

    let mut files: Vec<PathBuf> = Vec::new();
    if let Err(e) = walk_lean_files(&dir, &mut files) {
        eprintln!("# walk failed: {}", e);
        return ExitCode::from(2);
    }
    files.sort();

    let mut total = 0usize;
    let mut in_scope_candidates = 0usize;
    let mut lean_accepted = 0usize;
    let mut lean_rejected = 0usize;
    let mut lean_skipped = 0usize;

    let do_check = env::var("LEAN_CHECK").is_ok();

    for path in &files {
        total += 1;
        let source = fs::read_to_string(path).unwrap_or_default();
        let appears_prop = looks_propositional_only(&source);
        if appears_prop {
            in_scope_candidates += 1;
        }

        let (label, note) = if do_check {
            match check_with_lean4(path) {
                CheckOutcome::Accepted => {
                    lean_accepted += 1;
                    ("accepted", "")
                }
                CheckOutcome::Rejected(_) => {
                    lean_rejected += 1;
                    ("rejected", "")
                }
                CheckOutcome::LeanNotInstalled => {
                    lean_skipped += 1;
                    ("lean_not_installed", "")
                }
                CheckOutcome::ProcessError(_) => {
                    lean_rejected += 1;
                    ("process_error", "")
                }
            }
        } else {
            lean_skipped += 1;
            ("skipped", "")
        };

        println!(
            "{},{},{},{},{},{}",
            path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            path.display(),
            source.len(),
            appears_prop,
            label,
            note
        );
    }

    eprintln!(
        "# summary: {} files | {} heuristically in scope (≤ strict upper bound \
         for Phase 1) | accepted={} rejected={} skipped={}",
        total, in_scope_candidates, lean_accepted, lean_rejected, lean_skipped
    );
    eprintln!(
        "# Phase 1 accept rate is architecturally bounded. See \
         docs/minif2f-v2-scope.md for the honest scope analysis and the \
         Phase 2 plan to reach 15-30%."
    );

    ExitCode::SUCCESS
}
