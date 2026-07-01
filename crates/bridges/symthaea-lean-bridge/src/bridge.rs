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

use symthaea_core::hdc::logic_engine::{ProofResult, ProofStepLogic, Proposition};

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
        "Hypothetical Syllogism" | "Disjunctive Syllogism" => Some(LeanTactic::Raw("tauto".into())),
        "Unit Propagation" | "Resolution" | "KB Clause" => Some(LeanTactic::Raw("tauto".into())),
        "Negated Goal" => Some(LeanTactic::Raw("by_contra h".into())),
        _ => None,
    }
}

/// Synthesize a closed Lean 4 proof term for a propositional goal, given a
/// context of named hypotheses. Returns `None` if no term is found within the
/// depth budget.
///
/// This is a small classical propositional proof search that handles:
/// - Identity / assumption lookup
/// - `True.intro`
/// - Conjunction intro (⟨a, b⟩) and projection (`h.1`, `h.2`)
/// - Disjunction intro (`Or.inl`, `Or.inr`) and disjunction elim
/// - Implication intro (`fun h => _`) and assumption chaining
/// - Excluded middle via `Classical.em` for `A ∨ ¬A`
/// - Reductio: derive `False` from `A` and `¬A` in context
///
/// The search is bounded by `depth`; call with `depth = 16` for Phase 1
/// propositional fixtures. All emitted terms type-check in core Lean 4 without
/// Mathlib imports.
/// Walk a hypothesis and record every `And`-projection path along with the
/// proposition at that path. For `h : A ∧ (B ∧ C)`, records:
/// `(h, A ∧ (B ∧ C))`, `(h.1, A)`, `(h.2, B ∧ C)`, `(h.2.1, B)`, `(h.2.2, C)`.
fn collect_projections<'a>(
    name: &str,
    prop: &'a Proposition,
    out: &mut Vec<(String, &'a Proposition)>,
) {
    out.push((name.to_string(), prop));
    if let Proposition::And(l, r) = prop {
        collect_projections(&format!("{}.1", name), l, out);
        collect_projections(&format!("{}.2", name), r, out);
    }
}

/// Peel an implication chain `A₁ → A₂ → … → Aₙ → G` and, if `G == goal`,
/// attempt to synthesize each Aᵢ as a proof term. Returns the list of
/// argument terms in order. Returns `None` if the chain's final conclusion
/// doesn't equal the goal, or if any Aᵢ is unprovable within the depth budget.
fn try_curry_apply(
    prop: &Proposition,
    goal: &Proposition,
    hypotheses: &[(String, Proposition)],
    depth: u32,
) -> Option<Vec<String>> {
    let mut cursor = prop;
    let mut args_to_prove: Vec<&Proposition> = Vec::new();
    while let Proposition::Implies(a, b) = cursor {
        args_to_prove.push(a);
        cursor = b;
    }
    // Must have consumed at least one arrow and landed exactly on the goal.
    if args_to_prove.is_empty() || cursor != goal {
        return None;
    }
    let mut arg_terms: Vec<String> = Vec::new();
    for a in &args_to_prove {
        let t = synthesize_proof_term(a, hypotheses, depth.saturating_sub(1))?;
        arg_terms.push(t);
    }
    Some(arg_terms)
}

pub fn synthesize_proof_term(
    goal: &Proposition,
    hypotheses: &[(String, Proposition)],
    depth: u32,
) -> Option<String> {
    if depth == 0 {
        return None;
    }

    // 1. True is always provable.
    if matches!(goal, Proposition::True) {
        return Some("True.intro".to_string());
    }

    // 2. Assumption lookup: is the goal already in context?
    for (name, h) in hypotheses {
        if h == goal {
            return Some(name.clone());
        }
    }

    // 3. Projection from conjunction hypotheses, including nested
    //    projection. Expand every hypothesis `h : A₁ ∧ A₂ ∧ …` into all
    //    reachable sub-projections `h.i.j.k…` and look for the goal.
    let mut projections: Vec<(String, &Proposition)> = Vec::new();
    for (name, h) in hypotheses {
        collect_projections(name, h, &mut projections);
    }
    for (expr, prop) in &projections {
        if *prop == goal {
            return Some(expr.clone());
        }
    }

    // 3b. False-elim: any hypothesis of type False (reachable via
    //     projection) closes any goal via `False.elim`.
    for (expr, prop) in &projections {
        if matches!(prop, Proposition::False) {
            return Some(format!("(False.elim {})", expr));
        }
    }

    // 4. Curried modus ponens: for each projection
    //    `h : A₁ → A₂ → … → Aₙ → G` whose final conclusion G equals the goal,
    //    try to synthesize each Aᵢ and emit `h a₁ a₂ … aₙ`. This subsumes
    //    single-arg MP and handles uncurrying patterns like
    //    `(P → Q → R) → P ∧ Q → R`.
    for (expr, prop) in &projections {
        if let Some(arg_terms) = try_curry_apply(prop, goal, hypotheses, depth - 1) {
            let mut s = expr.clone();
            for a in &arg_terms {
                s = format!("({} {})", s, a);
            }
            return Some(s);
        }
    }

    // 5. Goal-directed rules.
    match goal {
        Proposition::And(a, b) => {
            let ta = synthesize_proof_term(a, hypotheses, depth - 1)?;
            let tb = synthesize_proof_term(b, hypotheses, depth - 1)?;
            Some(format!("⟨{}, {}⟩", ta, tb))
        }
        Proposition::Or(a, b) => {
            // Excluded-middle shortcut: A ∨ ¬A or ¬A ∨ A.
            if let Proposition::Not(na) = b.as_ref() {
                if a.as_ref() == na.as_ref() {
                    let a_term = prop_to_lean_term(a);
                    return Some(format!("Classical.em {}", a_term.to_lean()));
                }
            }
            if let Proposition::Not(na) = a.as_ref() {
                if na.as_ref() == b.as_ref() {
                    let a_term = prop_to_lean_term(b);
                    return Some(format!("(Classical.em {}).symm", a_term.to_lean()));
                }
            }
            // Try the left side first.
            if let Some(ta) = synthesize_proof_term(a, hypotheses, depth - 1) {
                return Some(format!("Or.inl {}", ta));
            }
            if let Some(tb) = synthesize_proof_term(b, hypotheses, depth - 1) {
                return Some(format!("Or.inr {}", tb));
            }
            None
        }
        Proposition::Implies(a, b) => {
            // Introduce hypothesis. Use a fresh name to avoid clashes.
            let fresh = format!("h{}", hypotheses.len());
            let mut extended = hypotheses.to_vec();
            extended.push((fresh.clone(), (**a).clone()));
            let body = synthesize_proof_term(b, &extended, depth - 1)?;
            Some(format!("(fun {} => {})", fresh, body))
        }
        Proposition::Not(a) => {
            // ¬A is A → False; recurse as implication.
            let imp = Proposition::Implies(a.clone(), Box::new(Proposition::False));
            synthesize_proof_term(&imp, hypotheses, depth - 1)
        }
        Proposition::False => {
            // Find contradictory pair A and ¬A in context.
            for (nname, nh) in hypotheses {
                if let Proposition::Not(inner) = nh {
                    if let Some(ht) = synthesize_proof_term(inner, hypotheses, depth - 1) {
                        return Some(format!("({} {})", nname, ht));
                    }
                }
            }
            None
        }
        Proposition::Iff(a, b) => {
            // Encode A ↔ B as ⟨A → B, B → A⟩ (matches prop_to_lean_term).
            let ab = Proposition::Implies(a.clone(), b.clone());
            let ba = Proposition::Implies(b.clone(), a.clone());
            let tab = synthesize_proof_term(&ab, hypotheses, depth - 1)?;
            let tba = synthesize_proof_term(&ba, hypotheses, depth - 1)?;
            Some(format!("⟨{}, {}⟩", tab, tba))
        }
        _ => None,
    }
}

/// Pattern-match on a `Proposition` goal and emit a tactic block that closes
/// it using **core Lean 4 only** (no Mathlib dependency).
///
/// Phase 1 strategy: synthesize a closed proof term via
/// [`synthesize_proof_term`] and emit `exact <term>`. For the goals that don't
/// fit the propositional fragment (e.g. FOL with free variables), returns
/// `None` and the caller emits `Sorry`.
pub fn tactics_for_goal(goal: &Proposition) -> Option<Vec<LeanTactic>> {
    let term = synthesize_proof_term(goal, &[], 16)?;
    Some(vec![LeanTactic::Raw(format!("exact {}", term))])
}

/// Translate a full `ProofResult` into a `LeanProofScript` targeting
/// `theorem_name : <goal in Lean syntax>`.
///
/// The `goal` is the original proposition the proof aimed to establish.
/// Only valid proofs (`result.valid == true`) produce non-`sorry` scripts;
/// invalid proofs emit a `Sorry`-tagged script so the file still type-checks
/// with a warning.
///
/// The tactic block is chosen by [`tactics_for_goal`]. If the goal doesn't
/// match any Phase-1 pattern, we emit `Sorry` — counted as a failure in the
/// external-verify gate (which is the honest signal).
pub fn proof_result_to_lean(
    theorem_name: &str,
    goal: &Proposition,
    result: &ProofResult,
) -> LeanProofScript {
    let goal_term = prop_to_lean_term(goal);
    let statement = goal_term.to_lean();

    let tactics = if result.valid && all_rules_classified(&result.proof_steps) {
        tactics_for_goal(goal).unwrap_or_else(|| vec![LeanTactic::Sorry])
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
pub fn render_lean_file(theorem_name: &str, goal: &Proposition, result: &ProofResult) -> String {
    let atoms = atoms_in_prop(goal);
    let mut lines: Vec<String> = Vec::new();

    lines.push("-- Auto-generated by symthaea-lean-bridge".to_string());
    lines.push(format!(
        "-- from ProofResult {{ valid = {}, phi = {:.3} }}",
        result.valid, result.phi
    ));
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
    fn valid_classified_identity_uses_term_mode() {
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
        // Identity should close in term mode, not with tauto.
        // Synthesizer uses fresh names like `h0`; assert shape, not name.
        assert!(rendered.contains("fun ") && rendered.contains(" => "));
        assert!(!rendered.contains("tauto"));
        assert!(!script.contains_sorry());
    }

    #[test]
    fn excluded_middle_uses_classical_em() {
        let goal = atom("P").or(atom("P").not());
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Modus Ponens".to_string(),
                formula: "P ∨ ¬P".to_string(),
                justification: "LEM".to_string(),
            }],
            phi: 0.5,
            description: "LEM".to_string(),
        };
        let script = proof_result_to_lean("lem", &goal, &result);
        assert!(script.to_lean().contains("Classical.em"));
        assert!(!script.contains_sorry());
    }

    #[test]
    fn nested_implication_uses_intros_assumption() {
        let goal = atom("P")
            .implies(atom("Q"))
            .implies(atom("P").implies(atom("Q")));
        let result = ProofResult {
            valid: true,
            proof_steps: vec![ProofStepLogic {
                step_number: 1,
                rule: "Modus Ponens".to_string(),
                formula: "(P → Q) → P → Q".to_string(),
                justification: "deducibility".to_string(),
            }],
            phi: 0.5,
            description: "MP deducibility".to_string(),
        };
        let script = proof_result_to_lean("mp_ded", &goal, &result);
        let rendered = script.to_lean();
        // Synthesizer closes this as nested `fun`: (fun h0 => h0) whose type
        // is the reflexive implication.
        assert!(rendered.contains("fun ") && rendered.contains(" => "));
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
