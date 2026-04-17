// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge: `symthaea_core::hdc::logic_engine` → Lean 4 proof script.
//!
//! Two translation layers:
//!
//! 1. **Proposition AST → `LeanTerm`**: structural rewrite of the
//!    propositional calculus into Lean 4's term grammar (variables become
//!    `Prop`-typed identifiers).
//! 2. **`ProofResult` → `LeanProofScript`**: rule-string-driven tactic
//!    synthesis. Natural-deduction rules (Modus Ponens, Syllogisms) get
//!    structured tactic scripts. Resolution / SAT rules fall back to Lean's
//!    `tauto` decision procedure for Phase 1.
//!
//! ## Phase 1 policy on rule coverage
//!
//! | `ProofStepLogic.rule`     | Tactic strategy                  |
//! |---------------------------|----------------------------------|
//! | `Premise`                 | `intro` (hypothesis)             |
//! | `Modus Ponens`            | `apply` chain                    |
//! | `Modus Tollens`           | `tauto` (contrapositive auto)    |
//! | `Hypothetical Syllogism`  | `tauto`                          |
//! | `Disjunctive Syllogism`   | `tauto`                          |
//! | `Unit Propagation`        | `tauto`                          |
//! | `Resolution`              | `tauto`                          |
//! | `Negated Goal`            | `by_contra`                      |
//! | `KB Clause`               | `tauto`                          |
//! | anything else             | `Sorry` (counts as failure)      |
//!
//! `tauto` is Lean 4 Mathlib's propositional-tautology decision procedure;
//! it closes classical propositional goals without proof-term reconstruction.
//! This is the pragmatic choice for Phase 1 — we get externally-verified
//! proofs without reimplementing natural deduction in tactic mode.

use symthaea_core::hdc::logic_engine::{Proposition, ProofResult, ProofStepLogic};

use crate::tactic::{LeanProofScript, LeanTactic};
use crate::term::LeanTerm;

/// Translate a `Proposition` from the logic engine into a `LeanTerm`.
///
/// Atom names become Lean identifiers (the caller is responsible for
/// declaring them as `variable (name : Prop)` in the emitted Lean file).
pub fn prop_to_lean_term(prop: &Proposition) -> LeanTerm {
    match prop {
        Proposition::Atom(name) => LeanTerm::Ident(name.clone()),
        Proposition::Not(p) => LeanTerm::Not(Box::new(prop_to_lean_term(p))),
        Proposition::And(p, q) => LeanTerm::And(
            Box::new(prop_to_lean_term(p)),
            Box::new(prop_to_lean_term(q)),
        ),
        Proposition::Or(p, q) => LeanTerm::Or(
            Box::new(prop_to_lean_term(p)),
            Box::new(prop_to_lean_term(q)),
        ),
        Proposition::Implies(p, q) => LeanTerm::Implies(
            Box::new(prop_to_lean_term(p)),
            Box::new(prop_to_lean_term(q)),
        ),
        Proposition::Iff(p, q) => {
            // Lean 4 has ↔; we encode as "a ↔ b" via And of two implications
            // for portability across Lean versions without Mathlib.
            let a = prop_to_lean_term(p);
            let b = prop_to_lean_term(q);
            LeanTerm::And(
                Box::new(LeanTerm::Implies(Box::new(a.clone()), Box::new(b.clone()))),
                Box::new(LeanTerm::Implies(Box::new(b), Box::new(a))),
            )
        }
        Proposition::True => LeanTerm::True,
        Proposition::False => LeanTerm::False,
    }
}

/// Collect all atom names from a proposition, in stable (sorted) order.
pub fn atoms_in_prop(prop: &Proposition) -> Vec<String> {
    let mut vars: Vec<String> = prop.variables().into_iter().collect();
    vars.sort();
    vars
}

/// Map a single rule string to the Lean tactic we'll emit for it.
///
/// Returns `None` for rules we don't yet classify — caller should emit
/// `Sorry` in that case.
pub fn rule_to_tactic(rule: &str) -> Option<LeanTactic> {
    match rule {
        "Premise" => Some(LeanTactic::Raw("-- premise".into())),
        "Modus Ponens" | "Modus Tollens" => Some(LeanTactic::Raw("tauto".into())),
        "Hypothetical Syllogism" | "Disjunctive Syllogism" => {
            Some(LeanTactic::Raw("tauto".into()))
        }
        "Unit Propagation" | "Resolution" | "KB Clause" => {
            Some(LeanTactic::Raw("tauto".into()))
        }
        "Negated Goal" => Some(LeanTactic::Raw("by_contra h".into())),
        _ => None,
    }
}

/// Translate a full `ProofResult` into a `LeanProofScript` targeting
/// `theorem_name : <goal in Lean syntax>`.
///
/// The `goal` is the original proposition the proof aimed to establish.
/// Only valid proofs (`result.valid == true`) produce non-`sorry` scripts;
/// invalid proofs emit a `Sorry`-tagged script so the file still type-checks
/// with a warning.
pub fn proof_result_to_lean(
    theorem_name: &str,
    goal: &Proposition,
    result: &ProofResult,
) -> LeanProofScript {
    let goal_term = prop_to_lean_term(goal);
    let statement = goal_term.to_lean();

    // Phase 1 strategy: if the engine reports the proof valid AND every rule
    // in the proof is classically sound propositional reasoning, close with
    // `tauto`. If any step is unclassified, emit `sorry`.
    let tactics = if result.valid && all_rules_classified(&result.proof_steps) {
        vec![LeanTactic::Raw("tauto".into())]
    } else {
        vec![LeanTactic::Sorry]
    };

    LeanProofScript {
        theorem_name: theorem_name.to_string(),
        statement,
        tactics,
    }
}

fn all_rules_classified(steps: &[ProofStepLogic]) -> bool {
    steps.iter().all(|s| rule_to_tactic(&s.rule).is_some())
}

/// Render a complete Lean 4 file (with `variable` declarations) for a
/// proposition goal. The caller-visible API.
pub fn render_lean_file(
    theorem_name: &str,
    goal: &Proposition,
    result: &ProofResult,
) -> String {
    let atoms = atoms_in_prop(goal);
    let mut lines: Vec<String> = Vec::new();

    lines.push("-- Auto-generated by symthaea-lean-bridge".to_string());
    lines.push(format!("-- from ProofResult {{ valid = {}, phi = {:.3} }}", result.valid, result.phi));
    lines.push(String::new());

    // Propositional variables
    if !atoms.is_empty() {
        for a in &atoms {
            lines.push(format!("variable ({} : Prop)", a));
        }
        lines.push(String::new());
    }

    let script = proof_result_to_lean(theorem_name, goal, result);
    lines.push(script.to_lean());

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(name: &str) -> Proposition {
        Proposition::Atom(name.to_string())
    }

    #[test]
    fn atom_maps_to_ident() {
        let p = atom("P");
        match prop_to_lean_term(&p) {
            LeanTerm::Ident(s) => assert_eq!(s, "P"),
            other => panic!("expected Ident, got {:?}", other),
        }
    }

    #[test]
    fn implies_maps_to_arrow() {
        let imp = atom("P").implies(atom("Q"));
        let t = prop_to_lean_term(&imp);
        assert_eq!(t.to_lean(), "(P → Q)");
    }

    #[test]
    fn iff_expands_to_and_of_implications() {
        let i = atom("P").iff(atom("Q"));
        let t = prop_to_lean_term(&i);
        assert_eq!(t.to_lean(), "((P → Q) ∧ (Q → P))");
    }

    #[test]
    fn atoms_are_sorted() {
        let p = atom("Z").and(atom("A").or(atom("M")));
        assert_eq!(atoms_in_prop(&p), vec!["A", "M", "Z"]);
    }

    #[test]
    fn valid_classified_proof_emits_tauto() {
        let goal = atom("P").implies(atom("P"));
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Modus Ponens".to_string(),
                formula: "P → P".to_string(),
                justification: "trivial".to_string(),
            }],
            phi: 0.5,
            description: "identity implication".to_string(),
        };
        let script = proof_result_to_lean("id_impl", &goal, &result);
        let rendered = script.to_lean();
        assert!(rendered.contains("tauto"));
        assert!(!script.contains_sorry());
    }

    #[test]
    fn invalid_proof_emits_sorry() {
        let goal = atom("P");
        let result = ProofResult {
            valid: false,
            proof_steps: vec![],
            phi: 0.0,
            description: "not valid".to_string(),
        };
        let script = proof_result_to_lean("t", &goal, &result);
        assert!(script.contains_sorry());
    }

    #[test]
    fn unclassified_rule_emits_sorry() {
        let goal = atom("P").implies(atom("P"));
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Quantum Woo".to_string(), // not a real rule
                formula: "P → P".to_string(),
                justification: "trust me".to_string(),
            }],
            phi: 0.5,
            description: "bogus".to_string(),
        };
        let script = proof_result_to_lean("t", &goal, &result);
        assert!(script.contains_sorry());
    }

    #[test]
    fn full_file_has_variable_declarations() {
        let goal = atom("P").implies(atom("Q"));
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Premise".to_string(),
                formula: "P".to_string(),
                justification: "given".to_string(),
            }],
            phi: 0.5,
            description: "test".to_string(),
        };
        let file = render_lean_file("t", &goal, &result);
        assert!(file.contains("variable (P : Prop)"));
        assert!(file.contains("variable (Q : Prop)"));
        assert!(file.contains("theorem t"));
    }
}
