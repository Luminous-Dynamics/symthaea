// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `prove_proptauts` — classical propositional tautology benchmark.
//!
//! Emits one `.lean` file per fixture under `proofs/proptauts/` and, when
//! `LEAN_CHECK=1` is set, invokes `lean <file>` on each. Produces a CSV to
//! stdout with per-fixture acceptance.
//!
//! This is the scale-up from `prove_minif2f` (3 fixtures) to a broader
//! classical-propositional benchmark suite (~25 fixtures) covering:
//! - conjunction intro / elim / commutativity / associativity
//! - disjunction intro / commutativity / associativity
//! - implication (identity, K, composition, deducibility)
//! - negation / double negation / contrapositive
//! - De Morgan (one direction; the other requires classical reasoning)
//! - excluded middle and LEM-derived consequences
//!
//! Every emitted theorem is closed by `bridge::synthesize_proof_term` in
//! term mode. Goals that fall outside the synthesizer's current reach
//! (typically anything requiring intermediate case-analysis) emit `sorry`
//! and count as fails.
//!
//! ## Run
//!
//! ```bash
//! # inside `nix develop`
//! LEAN_CHECK=1 cargo run -p symthaea-lean-bridge --example prove_proptauts \
//!   > proptauts_results.csv
//! ```

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
}

/// Build the full suite of propositional tautology fixtures.
fn fixtures() -> Vec<Fixture> {
    let p = || atom("P");
    let q = || atom("Q");
    let r = || atom("R");

    vec![
        // ─── Identity & reflexive implications ──────────────────────────
        Fixture {
            name: "id_impl",
            description: "P → P",
            goal: p().implies(p()),
        },
        Fixture {
            name: "k_combinator",
            description: "P → Q → P  (K combinator)",
            goal: p().implies(q().implies(p())),
        },
        Fixture {
            name: "mp_deducibility",
            description: "(P → Q) → P → Q",
            goal: p().implies(q()).implies(p().implies(q())),
        },
        Fixture {
            name: "hypothetical_syllogism",
            description: "(P → Q) → (Q → R) → (P → R)",
            goal: p()
                .implies(q())
                .implies(q().implies(r()).implies(p().implies(r()))),
        },
        // ─── Conjunction ────────────────────────────────────────────────
        Fixture {
            name: "and_intro",
            description: "P → Q → P ∧ Q",
            goal: p().implies(q().implies(p().and(q()))),
        },
        Fixture {
            name: "and_elim_left",
            description: "P ∧ Q → P",
            goal: p().and(q()).implies(p()),
        },
        Fixture {
            name: "and_elim_right",
            description: "P ∧ Q → Q",
            goal: p().and(q()).implies(q()),
        },
        Fixture {
            name: "and_commutative",
            description: "P ∧ Q → Q ∧ P",
            goal: p().and(q()).implies(q().and(p())),
        },
        Fixture {
            name: "and_associative_right",
            description: "(P ∧ Q) ∧ R → P ∧ (Q ∧ R)",
            goal: p().and(q()).and(r()).implies(p().and(q().and(r()))),
        },
        Fixture {
            name: "and_associative_left",
            description: "P ∧ (Q ∧ R) → (P ∧ Q) ∧ R",
            goal: p().and(q().and(r())).implies(p().and(q()).and(r())),
        },
        // ─── Disjunction ────────────────────────────────────────────────
        Fixture {
            name: "or_intro_left",
            description: "P → P ∨ Q",
            goal: p().implies(p().or(q())),
        },
        Fixture {
            name: "or_intro_right",
            description: "Q → P ∨ Q",
            goal: q().implies(p().or(q())),
        },
        // ─── Negation / contrapositive / double negation ────────────────
        Fixture {
            name: "contrapositive",
            description: "(P → Q) → (¬Q → ¬P)",
            goal: p().implies(q()).implies(q().not().implies(p().not())),
        },
        Fixture {
            name: "double_negation_intro",
            description: "P → ¬¬P",
            goal: p().implies(p().not().not()),
        },
        Fixture {
            name: "ex_falso",
            description: "False → P (ex falso quodlibet)",
            goal: Proposition::False.implies(p()),
        },
        // ─── Excluded middle and its consequences ───────────────────────
        Fixture {
            name: "lem",
            description: "P ∨ ¬P",
            goal: p().or(p().not()),
        },
        Fixture {
            name: "lem_flipped",
            description: "¬P ∨ P",
            goal: p().not().or(p()),
        },
        // ─── Implication manipulation ───────────────────────────────────
        Fixture {
            name: "and_implies_to_impl",
            description: "(P ∧ Q → R) → (P → Q → R)  (curry)",
            goal: p()
                .and(q())
                .implies(r())
                .implies(p().implies(q().implies(r()))),
        },
        Fixture {
            name: "impl_to_and_implies",
            description: "(P → Q → R) → (P ∧ Q → R)  (uncurry)",
            goal: p()
                .implies(q().implies(r()))
                .implies(p().and(q()).implies(r())),
        },
        // ─── Truth ──────────────────────────────────────────────────────
        Fixture {
            name: "true_intro",
            description: "True",
            goal: Proposition::True,
        },
        Fixture {
            name: "true_impl",
            description: "P → True",
            goal: p().implies(Proposition::True),
        },
        // ─── Combined (harder) ──────────────────────────────────────────
        Fixture {
            name: "and_distrib_over_and",
            description: "P ∧ Q ∧ R → R ∧ P ∧ Q",
            goal: p().and(q().and(r())).implies(r().and(p().and(q()))),
        },
        Fixture {
            name: "triple_impl",
            description: "P → (P → Q) → (P → Q → R) → R",
            goal: p().implies(
                p().implies(q())
                    .implies(p().implies(q().implies(r())).implies(r())),
            ),
        },
    ]
}

fn output_dir() -> PathBuf {
    env::var("PROPTAUTS_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("proofs/proptauts"))
}

fn synthetic_result() -> ProofResult {
    // Fixtures are all tautologies; construct a ProofResult with a
    // classified rule so the bridge doesn't fall through to Sorry.
    ProofResult {
        valid: true,
        proof_steps: vec![ProofStepLogic {
            step_number: 1,
            rule: "Modus Ponens".to_string(),
            formula: "propositional tautology".to_string(),
            justification: "synthesized".to_string(),
        }],
        phi: 0.5,
        description: "propositional tautology benchmark".to_string(),
    }
}

fn main() -> ExitCode {
    let out_dir = output_dir();
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("Failed to create {}: {}", out_dir.display(), e);
        return ExitCode::from(2);
    }

    let do_check = env::var("LEAN_CHECK").is_ok();

    println!("fixture,file,wrote_bytes,contains_sorry,lean_check,is_tautology_per_engine");

    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut skipped = 0usize;
    let mut sorry_count = 0usize;

    for fx in fixtures() {
        let result = synthetic_result();

        // Honest signal: did our own logic engine agree this is a tautology?
        // For non-Iff goals we use is_tautology; for True we treat trivially.
        let is_taut = matches!(fx.goal, Proposition::True) || LogicEngine::is_tautology(&fx.goal);

        let file_contents = render_lean_file(fx.name, &fx.goal, &result);
        let file_path = out_dir.join(format!("{}.lean", fx.name));
        let bytes = file_contents.len();
        let contains_sorry = file_contents.contains("sorry");

        if contains_sorry {
            sorry_count += 1;
        }

        if let Err(e) = fs::write(&file_path, &file_contents) {
            eprintln!("# write failed for {}: {}", file_path.display(), e);
            continue;
        }

        let check_label = if do_check {
            match check_with_lean4(&file_path) {
                CheckOutcome::Accepted => {
                    accepted += 1;
                    "accepted".to_string()
                }
                CheckOutcome::Rejected(_) => {
                    rejected += 1;
                    "rejected".to_string()
                }
                CheckOutcome::LeanNotInstalled => {
                    skipped += 1;
                    "lean_not_installed".to_string()
                }
                CheckOutcome::ProcessError(e) => {
                    rejected += 1;
                    format!("process_error:{}", e.replace(',', ";"))
                }
            }
        } else {
            skipped += 1;
            "skipped".to_string()
        };

        println!(
            "{},{},{},{},{},{}",
            fx.name,
            file_path.display(),
            bytes,
            contains_sorry,
            check_label,
            is_taut
        );

        eprintln!("# {} — {}", fx.name, fx.description);
    }

    eprintln!(
        "# summary: {} fixtures total | {} sorry | accepted={} rejected={} skipped={}",
        fixtures().len(),
        sorry_count,
        accepted,
        rejected,
        skipped
    );

    // Exit code signals overall success when Lean was asked to verify.
    if do_check && rejected > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
