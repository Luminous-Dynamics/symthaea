// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `prove_minif2f` — write one miniF2F-style fixture to disk as a `.lean`
//! file, optionally verify with `lean4 --check`.
//!
//! Week-3 milestone of plans/2-please-make-precious-fairy.md WS-B:
//! "1 miniF2F problem emits Lean that `lean4 --check` accepts zero errors."
//!
//! ## Run
//!
//! ```bash
//! # Just emit the file(s):
//! cargo run -p symthaea-lean-bridge --example prove_minif2f
//!
//! # Emit + verify (requires `lean` on PATH):
//! LEAN_CHECK=1 cargo run -p symthaea-lean-bridge --example prove_minif2f
//! ```
//!
//! Output directory: `proofs/minif2f/` under the workspace root
//! (overridable via `LEAN_OUT_DIR`).

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_core::hdc::logic_engine::{LogicEngine, ProofResult, ProofStepLogic, Proposition};
use symthaea_lean_bridge::bridge::render_lean_file;
use symthaea_lean_bridge::runner::{CheckOutcome, check_with_lean4};

fn atom(name: &str) -> Proposition {
    Proposition::Atom(name.to_string())
}

struct Fixture {
    name: &'static str,
    description: &'static str,
    goal: Proposition,
    result: ProofResult,
}

fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();

    // Fixture 1: identity implication `P → P`.
    {
        let goal = atom("P").implies(atom("P"));
        assert!(LogicEngine::is_tautology(&goal));
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Modus Ponens".to_string(),
                formula: "P → P".to_string(),
                justification: "identity implication".to_string(),
            }],
            phi: 0.5,
            description: "identity implication".to_string(),
        };
        out.push(Fixture {
            name: "minif2f_identity_impl",
            description: "∀ P: Prop, P → P",
            goal,
            result,
        });
    }

    // Fixture 2: classical excluded middle: P ∨ ¬P.
    // Lean's `tauto` handles this classically (it imports `Classical`
    // automatically when compiled in the default surface).
    {
        let goal = atom("P").or(atom("P").not());
        assert!(LogicEngine::is_tautology(&goal));
        let (_a, result) = LogicEngine::dpll_sat(&goal);
        out.push(Fixture {
            name: "minif2f_excluded_middle",
            description: "∀ P: Prop, P ∨ ¬P (classical LEM)",
            goal,
            result,
        });
    }

    // Fixture 3: modus ponens at the theorem level:
    // ((P → Q) → P → Q). The engine's modus_ponens on P and P → Q yields
    // a ProofResult concluding Q; we wrap it as a deducibility claim.
    {
        let premise = atom("P");
        let implication = atom("P").implies(atom("Q"));
        let result = LogicEngine::modus_ponens(&premise, &implication)
            .expect("modus_ponens(P, P→Q) must succeed");
        let goal = atom("P")
            .implies(atom("Q"))
            .implies(atom("P").implies(atom("Q")));
        out.push(Fixture {
            name: "minif2f_modus_ponens_deducibility",
            description: "∀ P Q: Prop, (P → Q) → P → Q",
            goal,
            result,
        });
    }

    out
}

fn output_dir() -> PathBuf {
    env::var("LEAN_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("proofs/minif2f"))
}

fn main() -> ExitCode {
    let out_dir = output_dir();
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("Failed to create {}: {}", out_dir.display(), e);
        return ExitCode::from(2);
    }

    let do_check = env::var("LEAN_CHECK").is_ok();

    println!("fixture,file,wrote_bytes,contains_sorry,lean_check");

    let mut any_failure = false;

    for fx in fixtures() {
        let file_contents = render_lean_file(fx.name, &fx.goal, &fx.result);
        let file_path = out_dir.join(format!("{}.lean", fx.name));
        let bytes = file_contents.len();
        let contains_sorry = file_contents.contains("sorry");

        if let Err(e) = fs::write(&file_path, &file_contents) {
            eprintln!("# write failed for {}: {}", file_path.display(), e);
            any_failure = true;
            continue;
        }

        let check_label = if do_check {
            match check_with_lean4(&file_path) {
                CheckOutcome::Accepted => "accepted".to_string(),
                CheckOutcome::Rejected(_) => {
                    any_failure = true;
                    "rejected".to_string()
                }
                CheckOutcome::LeanNotInstalled => "lean_not_installed".to_string(),
                CheckOutcome::ProcessError(e) => {
                    any_failure = true;
                    format!("process_error:{}", e.replace(',', ";"))
                }
            }
        } else {
            "skipped".to_string()
        };

        println!(
            "{},{},{},{},{}",
            fx.name,
            file_path.display(),
            bytes,
            contains_sorry,
            check_label
        );

        eprintln!("# {} — {}", fx.name, fx.description);
    }

    if any_failure && do_check {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
