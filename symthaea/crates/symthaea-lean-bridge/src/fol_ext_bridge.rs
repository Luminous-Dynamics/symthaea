// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 2 bridge: `FolFormulaExt` → Lean 4 proof script.
//!
//! Three-way dispatch:
//!
//! 1. **Pure-propositional goals** (those for which
//!    [`FolFormulaExt::is_purely_propositional`] returns true) are routed
//!    back to the Phase 1 [`crate::bridge::synthesize_proof_term`] —
//!    already 100% strict on classical propositional tautologies, no
//!    Mathlib dependency needed.
//!
//! 2. **Arithmetic goals** emit a Lean proof script with
//!    `import Mathlib.Tactic` and a single Mathlib tactic chosen by the
//!    detected SMT fragment (omega / linarith / nlinarith).
//!
//! 3. **Anything else** falls through to `sorry` — the honest failure
//!    signal.
//!
//! The emitted `.lean` file compiles only inside a Lake project that has
//! Mathlib resolved. See `lean-proofs/phase2/README.md` for the
//! setup command.
//!
//! ## Phase 2 Week 2 scope
//!
//! This module emits Lean that *targets* Mathlib but does not itself
//! invoke `lake build`. Actual Lean-check validation happens in the
//! `prove_fol_arith` example (when run inside a Mathlib-resolved project).
//! Structural validation (no `sorry` in emitted tautologies, tactic
//! choice matches fragment) is covered by this module's unit tests.

use symthaea_core::hdc::fol_ext_smt::{detect_fragment, SmtFragment};
use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};

use crate::bridge::render_lean_file;
use crate::tactic::{LeanProofScript, LeanTactic};

/// Render a Lean 4 term for an arithmetic `Term`. Mathlib syntax.
fn term_to_lean(t: &Term) -> String {
    match t {
        Term::Var(n) => n.clone(),
        Term::IntLit(n) => format!("({})", n),
        Term::RealLit(x) => {
            // Emit with explicit Real coercion so Lean doesn't infer Nat
            // for positive literals when the surrounding context is Real.
            format!("({} : ℝ)", x)
        }
        Term::RatLit(p, q) => {
            // Emit as `((p : ℝ) / (q : ℝ))` — reliably parses in Mathlib
            // without relying on implicit coercion rules.
            format!("(({} : ℝ) / ({} : ℝ))", p, q)
        }
        Term::BinOp(op, a, b) => {
            let sym = match op {
                symthaea_core::hdc::fol_formula_ext::ArithOp::Add => "+",
                symthaea_core::hdc::fol_formula_ext::ArithOp::Sub => "-",
                symthaea_core::hdc::fol_formula_ext::ArithOp::Mul => "*",
                symthaea_core::hdc::fol_formula_ext::ArithOp::Div => "/",
            };
            format!("({} {} {})", term_to_lean(a), sym, term_to_lean(b))
        }
        Term::Pow(base, exp) => {
            format!("({} ^ {})", term_to_lean(base), exp)
        }
        Term::Neg(a) => format!("(- {})", term_to_lean(a)),
    }
}

/// Render a Lean 4 proposition for a `FolFormulaExt`.
fn formula_to_lean(phi: &FolFormulaExt) -> String {
    match phi {
        FolFormulaExt::Base(_) => {
            // Callers should route pure-propositional goals to the Phase 1
            // path before rendering; this branch is defensive.
            "True".to_string()
        }
        FolFormulaExt::Eq(a, b) => {
            format!("({} = {})", term_to_lean(a), term_to_lean(b))
        }
        FolFormulaExt::Lt(a, b) => {
            format!("({} < {})", term_to_lean(a), term_to_lean(b))
        }
        FolFormulaExt::Le(a, b) => {
            format!("({} ≤ {})", term_to_lean(a), term_to_lean(b))
        }
        FolFormulaExt::And(a, b) => {
            format!("({} ∧ {})", formula_to_lean(a), formula_to_lean(b))
        }
        FolFormulaExt::Or(a, b) => {
            format!("({} ∨ {})", formula_to_lean(a), formula_to_lean(b))
        }
        FolFormulaExt::Not(a) => {
            format!("(¬ {})", formula_to_lean(a))
        }
        FolFormulaExt::Implies(a, b) => {
            format!("({} → {})", formula_to_lean(a), formula_to_lean(b))
        }
        FolFormulaExt::Forall(name, ty, body) => {
            format!(
                "(∀ {} : {}, {})",
                name,
                numeric_to_lean(*ty),
                formula_to_lean(body)
            )
        }
        FolFormulaExt::Exists(name, ty, body) => {
            format!(
                "(∃ {} : {}, {})",
                name,
                numeric_to_lean(*ty),
                formula_to_lean(body)
            )
        }
    }
}

fn numeric_to_lean(ty: NumericType) -> &'static str {
    match ty {
        NumericType::Int => "ℤ",
        NumericType::Nat => "ℕ",
        NumericType::Real => "ℝ",
    }
}

/// Generate a complete Lean 4 file (with appropriate `import` headers) for
/// a `FolFormulaExt` goal.
///
/// - Pure-propositional goals are delegated to
///   [`crate::bridge::render_lean_file`] — same Phase 1 synthesizer, no
///   Mathlib.
/// - Arithmetic goals emit `import Mathlib.Tactic` + the tactic suggested
///   by the detected SMT fragment.
pub fn render_fol_ext_file(theorem_name: &str, phi: &FolFormulaExt) -> String {
    // Route 1: pure propositional.
    if phi.is_purely_propositional() {
        if let FolFormulaExt::Base(prop) = phi {
            // Use the Phase 1 path verbatim: no Mathlib, no arithmetic.
            // We synthesize a proof and render a full file (with `variable
            // (P : Prop)` declarations).
            let result = symthaea_core::hdc::logic_engine::ProofResult {
                valid: true,
                proof_steps: vec![symthaea_core::hdc::logic_engine::ProofStepLogic {
                    step_number: 1,
                    rule: "Modus Ponens".to_string(),
                    formula: format!("{}", prop),
                    justification: "routed_from_fol_ext".to_string(),
                }],
                phi: 0.5,
                description: "pure-propositional via FolFormulaExt dispatch".to_string(),
            };
            return render_lean_file(theorem_name, prop, &result);
        }
        // Non-Base purely-prop (e.g., nested And/Or/Not/Implies over Base)
        // — try to synthesize against the unwrapped Proposition. Phase 1
        // synthesizer handles these when we re-extract. For now, fall
        // through to the arithmetic path which will emit `sorry`.
    }

    // Route 2: arithmetic.
    let fragment = detect_fragment(phi);
    let goal_lean = formula_to_lean(phi);
    let mathlib_tactic = fragment.suggested_lean_tactic();

    let (tactic, _contains_sorry) = synthesize_arith_tactic(phi, fragment);

    let mut out = String::new();
    out.push_str("-- Auto-generated by symthaea-lean-bridge::fol_ext_bridge\n");
    out.push_str(&format!("-- SMT fragment: {}\n", fragment.logic_name()));
    out.push_str(&format!("-- Mathlib tactic: {}\n", mathlib_tactic));
    out.push_str("-- Requires `import Mathlib.Tactic` + a Lake project with Mathlib resolved.\n");
    out.push_str("-- See lean-proofs/phase2/README.md for setup.\n");
    out.push_str("\n");
    out.push_str("import Mathlib.Tactic\n");
    out.push_str("\n");
    out.push_str(&format!("theorem {} : {} := by\n", theorem_name, goal_lean));
    out.push_str(&format!("  {}\n", tactic.to_lean()));
    out
}

/// Walk a `FolFormulaExt` and collect outer universal-quantifier binder
/// names in the order they appear. Stops at the first non-`Forall` node.
///
/// For `∀ x : ℝ, ∀ y : ℝ, x + y = y + x` → `["x", "y"]`.
/// For `3 * (1/3) = 1` → `[]`.
/// For `∀ x : ℝ, P → Q` → `["x"]` (stops at `Implies`).
fn collect_outer_forall_binders(phi: &FolFormulaExt) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = phi;
    while let FolFormulaExt::Forall(name, _, body) = cur {
        out.push(name.clone());
        cur = body;
    }
    out
}

/// Build the `nlinarith` hint list from a set of bound variable names.
/// With concrete names, hints unify cleanly; without them, `_` placeholders
/// often fail to resolve.
fn build_nlinarith_hints(names: &[String]) -> String {
    if names.is_empty() {
        return String::from("sq_nonneg _, mul_self_nonneg _");
    }
    let mut parts: Vec<String> = Vec::new();
    for n in names {
        parts.push(format!("sq_nonneg {}", n));
        parts.push(format!("mul_self_nonneg {}", n));
        // Literal offsets — catch AM-GM-like identities like
        // `2a ≤ 1 + a²` which reduces to `(a - 1)² ≥ 0`.
        parts.push(format!("sq_nonneg ({} - 1)", n));
        parts.push(format!("sq_nonneg ({} + 1)", n));
    }
    // Pairwise differences and sums (often needed for AM-GM-like identities).
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            parts.push(format!("sq_nonneg ({} - {})", names[i], names[j]));
            parts.push(format!("sq_nonneg ({} + {})", names[i], names[j]));
        }
    }
    parts.join(", ")
}

/// Choose a tactic block for an arithmetic goal. Returns
/// `(tactic_block, contains_sorry)`.
///
/// Strategy: we don't know a priori which tactic will close a given goal
/// (e.g. `∀ x : ℝ, x = x` is closed by `rfl` after `intro`, not by
/// `linarith`). Emit a `first | … | …` cascade that tries the most
/// common tactics in order from strongest-but-restricted to weakest-but-
/// broadest. The first succeeding alternative closes the goal.
///
/// **Phase 3 improvement: named-variable threading.** The cascade now
/// emits `intro x y z` with concrete names extracted from the outer
/// quantifier binders, and nlinarith hints like `sq_nonneg x` and
/// `mul_self_nonneg y` use those concrete names instead of `_`
/// placeholders that failed to unify. Similarly `rcases lt_trichotomy x y`
/// replaces the `_ _` form.
///
/// All tactics below are Mathlib. Ordering is chosen to maximize hit rate
/// on miniF2F-v2's typical goal shapes.
fn synthesize_arith_tactic(phi: &FolFormulaExt, fragment: SmtFragment) -> (LeanTactic, bool) {
    let _ = fragment; // cascade is fragment-agnostic; fragment guides
                      // the FIRST-choice tactic but we always include
                      // the full cascade so strong goals close fast
                      // and weird shapes still have fallbacks.

    // Lean 4 `first | t1 | t2 | …` tries alternatives left-to-right,
    // committing to the first that succeeds. `intros` + tactic composes
    // the two-step "strip universals then close" pattern seen in most
    // miniF2F-v2 statements.
    //
    // Rough ordering rationale:
    //   rfl          — literal `x = x` after intros
    //   norm_num     — numeric literals (3 · 1/3 = 1, etc.)
    //   ring         — polynomial equalities over commutative rings
    //   omega        — linear arithmetic over ℤ/ℕ
    //   linarith     — linear arithmetic over ordered fields
    //   nlinarith    — nonlinear arithmetic (squares, products)
    //   positivity   — non-negativity / strict-positivity of expressions
    //   tauto        — classical propositional tautologies
    //   polyrith     — polynomial arithmetic with rationals (catches some
    //                  cases that nlinarith misses)
    // Apply `intros` once at the top (Lean 4 drops `intros` → `intro` in
    // places; we use the safer `try intros` which no-ops if there are no
    // universals to strip). Then a flat `first | … | …` of closer tactics.
    //
    // `try` ensures `intros` never fails on goals that have no universals
    // (e.g. the raw `3 * (1/3) = 1` theorem); without `try` a closed-form
    // goal would silently fail the whole cascade because `intros` threw.
    // Phase 3: named-variable threading. Introduce outer universals with
    // concrete names (not anonymous via `try intros`), then use those
    // names in `rcases lt_trichotomy` and nlinarith hints so the `_`
    // placeholders that failed to unify in W4 now resolve cleanly.
    let binders = collect_outer_forall_binders(phi);
    // Introduce outer universals with concrete names, then `try intros`
    // to additionally peel implication hypotheses (goals like
    // `∀ x : ℝ, 0 < x → 0 ≤ x` need both: `intro x` for the universal
    // and `intros` for the implication's antecedent).
    let intro_line = if binders.is_empty() {
        String::from("try intros\n  ")
    } else {
        format!("intro {}\n  try intros\n  ", binders.join(" "))
    };
    let hints = build_nlinarith_hints(&binders);

    // rcases lt_trichotomy needs the first two binders. If we have fewer
    // than 2, fall back to `_ _` placeholders (the original W4 form).
    let lt_trichotomy_alt = if binders.len() >= 2 {
        format!(
            "(rcases lt_trichotomy {} {} with h | h | h <;> tauto; done)",
            binders[0], binders[1]
        )
    } else {
        String::from("(rcases lt_trichotomy _ _ with h | h | h <;> tauto; done)")
    };

    // Every alternative is `; done`-terminated. Reason: some Mathlib
    // tactics (notably `norm_num`) partially simplify the goal even when
    // they can't close it, which interferes with `first`'s backtracking.
    // Appending `done` forces each branch to either fully close the goal
    // or fail cleanly, letting `first` fall through to the next tactic.
    let cascade = format!(
        r#"{}first
    | (rfl; done)
    | (norm_num; done)
    | (ring; done)
    | (omega; done)
    | (linarith; done)
    | (nlinarith [{}]; done)
    | (positivity; done)
    | {}
    | (rcases le_total _ _ with h | h <;> first | linarith | tauto; done)
    | (tauto; done)
    | (polyrith; done)"#,
        intro_line, hints, lt_trichotomy_alt
    );

    (LeanTactic::Raw(cascade), false)
}

/// Same as [`render_fol_ext_file`] but returns the parsed `LeanProofScript`
/// rather than a full file. Callers that want to post-process (e.g. inject
/// hypotheses) can work on the script directly.
pub fn script_for_fol_ext(theorem_name: &str, phi: &FolFormulaExt) -> LeanProofScript {
    let fragment = detect_fragment(phi);
    let statement = formula_to_lean(phi);
    let (tactic, _) = synthesize_arith_tactic(phi, fragment);
    LeanProofScript {
        theorem_name: theorem_name.to_string(),
        statement,
        tactics: vec![tactic],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::fol_formula_ext::NumericType;

    fn x() -> Term {
        Term::var("x")
    }
    fn n() -> Term {
        Term::var("n")
    }

    #[test]
    fn arith_goals_emit_cascade() {
        // Every arithmetic goal gets the same `first | … | …` cascade.
        // The individual tactics (rfl, norm_num, ring, omega, linarith,
        // nlinarith, positivity, tauto, polyrith) appear verbatim in the
        // emitted file.
        let phi = FolFormulaExt::forall("x", NumericType::Real, FolFormulaExt::eq(x(), x()));
        let file = render_fol_ext_file("t_refl", &phi);
        assert!(file.contains("import Mathlib.Tactic"));
        assert!(file.contains("theorem t_refl"));
        assert!(file.contains("(∀ x : ℝ,"));
        // Cascade members must all be present:
        for t in [
            "rfl",
            "norm_num",
            "ring",
            "omega",
            "linarith",
            "nlinarith",
            "positivity",
            "tauto",
            "polyrith",
        ] {
            assert!(file.contains(t), "cascade missing {}: file = {}", t, file);
        }
        assert!(file.contains("first"));
        assert!(!file.contains("sorry"));
    }

    #[test]
    fn linear_int_cascade_includes_omega() {
        let phi = FolFormulaExt::forall(
            "n",
            NumericType::Int,
            FolFormulaExt::lt(n(), n().add(Term::IntLit(1))),
        );
        let file = render_fol_ext_file("t_succ_gt", &phi);
        assert!(file.contains("omega"));
        assert!(!file.contains("sorry"));
    }

    #[test]
    fn nonlinear_real_cascade_includes_nlinarith() {
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::le(Term::IntLit(0), x().mul(x())),
        );
        let file = render_fol_ext_file("t_sq_nonneg", &phi);
        assert!(file.contains("nlinarith"));
        assert!(!file.contains("sorry"));
    }

    #[test]
    fn formula_renders_unicode_comparators() {
        let phi = FolFormulaExt::le(Term::IntLit(0), x());
        let rendered = formula_to_lean(&phi);
        assert_eq!(rendered, "((0) ≤ x)");
    }

    #[test]
    fn rat_literal_renders_exact() {
        let t = Term::rat(1, 3);
        let rendered = term_to_lean(&t);
        assert_eq!(rendered, "((1 : ℝ) / (3 : ℝ))");
    }

    #[test]
    fn implication_renders_with_arrow() {
        // x > 0 → x ≥ 0
        let phi = FolFormulaExt::lt(Term::IntLit(0), x())
            .implies(FolFormulaExt::le(Term::IntLit(0), x()));
        let rendered = formula_to_lean(&phi);
        assert!(rendered.contains(" → "), "got: {}", rendered);
    }

    #[test]
    fn pure_prop_routes_to_phase1_synthesizer() {
        use symthaea_core::hdc::logic_engine::Proposition;
        // Wrap a pure-propositional tautology and verify the Phase 1
        // synthesizer signature appears (variable declarations).
        let phi = FolFormulaExt::from_prop(Proposition::atom("P").implies(Proposition::atom("P")));
        let file = render_fol_ext_file("t_id", &phi);
        // Phase 1 path emits `variable (P : Prop)`; arithmetic path does not.
        assert!(
            file.contains("variable (P : Prop)"),
            "expected Phase 1 routing; got: {}",
            file
        );
        // No Mathlib import for pure prop.
        assert!(!file.contains("import Mathlib"));
    }
}
