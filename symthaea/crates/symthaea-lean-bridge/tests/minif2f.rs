// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration test: end-to-end round-trip through the bridge.
//!
//! Week-2 milestone from plans/2-please-make-precious-fairy.md WS-B:
//! exercise `LogicEngine` → `ProofResult` → `LeanProofScript` with at
//! least one real FOL proposition and verify the emitted Lean syntax
//! is structurally well-formed.
//!
//! This test does NOT invoke `lean4 --check` — that's the Week-3
//! milestone (requires Lean toolchain installed). Here we assert
//! properties of the emitted text that are necessary conditions for
//! Lean acceptance.

use symthaea_core::hdc::logic_engine::{LogicEngine, ProofResult, ProofStepLogic, Proposition};
use symthaea_lean_bridge::bridge::{
    atoms_in_prop, proof_result_to_lean, prop_to_lean_term, render_lean_file,
};
use symthaea_lean_bridge::tactic::LeanTactic;

fn atom(name: &str) -> Proposition {
    Proposition::Atom(name.to_string())
}

/// Fixture 1: the identity implication `P → P` is a classical tautology.
/// The bridge should emit a Lean file that closes with `tauto`.
#[test]
fn identity_implication_emits_tauto() {
    let goal = atom("P").implies(atom("P"));

    // Confirm this really is a tautology according to the logic engine.
    assert!(
        LogicEngine::is_tautology(&goal),
        "P → P must be a tautology"
    );

    // Synthesize a minimal ProofResult tagged with a classically-sound rule.
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

    let file = render_lean_file("t_identity", &goal, &result);

    // Lean surface-syntax requirements:
    assert!(file.contains("variable (P : Prop)"), "must declare P");
    assert!(
        file.contains("theorem t_identity"),
        "must emit theorem name"
    );
    assert!(file.contains("(P → P)"), "must emit the implication");
    assert!(
        file.contains("fun ") && file.contains(" => "),
        "identity implication must use term-mode proof (core Lean 4)"
    );
    assert!(!file.contains("tauto"), "must not use Mathlib `tauto`");
    assert!(
        !file.contains("sorry"),
        "valid tautology must not emit sorry"
    );
}

/// Fixture 2: real modus_ponens invocation.
/// Exercise `LogicEngine::modus_ponens` and feed the result through the bridge.
#[test]
fn real_modus_ponens_through_bridge() {
    let premise = atom("P");
    let implication = atom("P").implies(atom("Q"));
    let result = LogicEngine::modus_ponens(&premise, &implication)
        .expect("modus_ponens on P and P → Q must succeed");

    // The engine says the proof concludes Q. Our Lean theorem states the
    // deducibility chain: (P → Q) → P → Q.
    let goal = atom("P")
        .implies(atom("Q"))
        .implies(atom("P").implies(atom("Q")));
    let file = render_lean_file("t_mp", &goal, &result);

    assert!(file.contains("variable (P : Prop)"));
    assert!(file.contains("variable (Q : Prop)"));
    assert!(file.contains("theorem t_mp"));
    // Synthesizer closes this with a term-mode lambda.
    assert!(
        file.contains("fun ") && file.contains(" => "),
        "reflexive implication chain must close in core Lean 4"
    );
    assert!(!file.contains("sorry"));
}

/// Fixture 3: term translation correctness for a compound formula.
#[test]
fn compound_formula_round_trips_to_lean_term() {
    // (P ∧ Q) → (P ∨ Q)
    let goal = atom("P").and(atom("Q")).implies(atom("P").or(atom("Q")));
    let term = prop_to_lean_term(&goal);
    let rendered = term.to_lean();

    assert_eq!(rendered, "((P ∧ Q) → (P ∨ Q))");
    assert_eq!(atoms_in_prop(&goal), vec!["P", "Q"]);
}

/// Fixture 4: invalid proof must produce a sorry-tagged script.
/// The external-verify gate counts `sorry` as failure, so this is the
/// honest signal for "engine couldn't close this".
#[test]
fn invalid_proof_is_honest_about_failure() {
    let goal = atom("P");
    let result = ProofResult {
        valid: false,
        proof_steps: vec![],
        phi: 0.0,
        description: "engine gave up".to_string(),
    };

    let script = proof_result_to_lean("t_fail", &goal, &result);
    assert!(script.contains_sorry());
    assert!(matches!(script.tactics.first(), Some(LeanTactic::Sorry)));
}

/// Fixture 5: DPLL-driven proof survives the bridge.
/// `dpll_sat` produces a ProofResult with `Unit Propagation` / `KB Clause`
/// steps, which the bridge classifies for `tauto`.
#[test]
fn dpll_proof_through_bridge() {
    // P is satisfiable; the ProofResult marks the attempt as valid
    // regardless of satisfiability (the result reports *whether a
    // proof/assignment was produced*, not logical validity).
    let prop = atom("P").or(atom("P").not());
    let (_assignment, result) = LogicEngine::dpll_sat(&prop);

    // dpll_sat always returns some ProofResult; only validity drives the
    // bridge's decision to emit `tauto` vs `sorry`.
    let goal = prop;
    let script = proof_result_to_lean("t_lem", &goal, &result);
    let rendered = script.to_lean();

    // Whatever the validity, the emitted string is syntactically a Lean
    // theorem block.
    assert!(rendered.starts_with("theorem t_lem"));
    assert!(rendered.contains(":= by"));
}
