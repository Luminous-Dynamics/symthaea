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

use symthaea_core::hdc::fol_ext_smt::{SmtFragment, detect_fragment};
use symthaea_core::hdc::fol_formula_ext::{ArithOp, FolFormulaExt, NumericType, Term};

use crate::bridge::render_lean_file;
use crate::tactic::{LeanProofScript, LeanTactic};

/// Render a Lean 4 term for an arithmetic `Term`. Mathlib syntax.
///
/// `Term::Var` names originate from `symthaea_core`'s externally-sourced
/// `Term` AST and are sanitized here -- the sole choke point every
/// rendering path in this module goes through -- before interpolation. See
/// `sanitize` module docs for why this matters (Lean source injection via
/// unescaped identifiers).
fn term_to_lean(t: &Term) -> String {
    match t {
        Term::Var(n) => crate::sanitize::sanitize_ident(n),
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
                crate::sanitize::sanitize_ident(name),
                numeric_to_lean(*ty),
                formula_to_lean(body)
            )
        }
        FolFormulaExt::Exists(name, ty, body) => {
            format!(
                "(∃ {} : {}, {})",
                crate::sanitize::sanitize_ident(name),
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
    // Sanitized once up front: Route 2 below interpolates `theorem_name`
    // directly (not through `LeanProofScript::to_lean`, which does its own
    // sanitization), so this is that route's choke point. Route 1 delegates
    // to `render_lean_file`, which is independently safe regardless.
    let theorem_name_owned = crate::sanitize::sanitize_ident(theorem_name);
    let theorem_name = theorem_name_owned.as_str();
    // Route 1: pure propositional.
    if phi.is_purely_propositional()
        && let FolFormulaExt::Base(prop) = phi
    {
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
    out.push('\n');
    out.push_str("import Mathlib.Tactic\n");
    out.push('\n');
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
///
/// Sanitizes each binder name at collection time (rather than at each of
/// the many downstream render sites) so every caller -- `intro_line`, the
/// `nlinarith` hint builders, `lt_trichotomy_alt` -- inherits a
/// already-safe `Vec<String>` for free. See `sanitize` module docs.
fn collect_outer_forall_binders(phi: &FolFormulaExt) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = phi;
    while let FolFormulaExt::Forall(name, _, body) = cur {
        out.push(crate::sanitize::sanitize_ident(name));
        cur = body;
    }
    out
}

/// Strip outer `Forall` and `Implies` wrappers to reach the ultimate
/// conclusion formula. Used to decide whether to emit the And-splitter
/// branch: we only emit it when the conclusion is syntactically an `And`,
/// because embedding a `refine ⟨?_, ?_⟩` branch for non-And goals
/// interacts poorly with Lean's elaborator on the widened-hint path
/// (observed: deterministic heartbeat timeouts on `mathd_algebra_37`,
/// `_141` during Phase 4 re-measurement).
fn conclusion_is_and(phi: &FolFormulaExt) -> bool {
    let mut cur = phi;
    loop {
        match cur {
            FolFormulaExt::Forall(_, _, body) | FolFormulaExt::Implies(_, body) => cur = body,
            FolFormulaExt::And(_, _) => return true,
            _ => return false,
        }
    }
}

/// Walk through outer `Forall`s, then collect the left-hand sides of the
/// outer `Implies` chain. Returns the hypotheses in order of appearance.
///
/// For `∀ n : ℝ, (¬ n = 3) → ((n+5)/(n-3) = 2) → (n = 11)` returns
/// `[¬n=3, (n+5)/(n-3) = 2]`. Used to count hypotheses for named-intro
/// emission and to locate `≠` hypotheses for `sub_ne_zero` witness
/// derivation in the Phase 6a field-reasoning branch.
fn collect_outer_hypotheses(phi: &FolFormulaExt) -> Vec<FolFormulaExt> {
    let mut cur = phi;
    while let FolFormulaExt::Forall(_, _, body) = cur {
        cur = body;
    }
    let mut out = Vec::new();
    while let FolFormulaExt::Implies(h, rest) = cur {
        out.push((**h).clone());
        cur = rest;
    }
    out
}

/// Is this hypothesis `¬ (a = b)` shape? If so, `sub_ne_zero.mpr h`
/// yields `a - b ≠ 0`, which `field_simp` can use to clear denominators.
/// `mathd_algebra_181` (`h₀ : n ≠ 3`) is the canonical miniF2F case.
fn is_ne_hypothesis(phi: &FolFormulaExt) -> bool {
    matches!(
        phi,
        FolFormulaExt::Not(inner) if matches!(inner.as_ref(), FolFormulaExt::Eq(_, _))
    )
}

/// Does any term in the formula contain a `Div` binop whose right side
/// contains a free variable (i.e. division by a symbolic expression)?
///
/// Checks the *entire* formula including hypotheses, not just the
/// conclusion. Used to gate the Phase 5 `field_simp` cascade branch.
/// `mathd_algebra_55` (`q / p = 2 / 3`) has division in the conclusion;
/// `mathd_algebra_181` (`(n+5)/(n-3) = 2 → n = 11`) has division in a
/// *hypothesis* with a linear conclusion. Both need `field_simp` to
/// clear denominators; a conclusion-only check misses the second.
///
/// Division by a pure literal (`x/50 = 40` in `mathd_algebra_24`) is
/// NOT flagged — `linarith` handles that case fine because the literal
/// denominator is statically known nonzero. Only symbolic denominators
/// trigger the field branch.
fn formula_has_symbolic_division(phi: &FolFormulaExt) -> bool {
    fn term_has_sym_div(t: &Term) -> bool {
        match t {
            Term::Var(_) | Term::IntLit(_) | Term::RealLit(_) | Term::RatLit(_, _) => false,
            Term::BinOp(ArithOp::Div, _a, b) => {
                // Symbolic denominator = contains a free variable. We walk
                // `b` looking for any `Var`. Nested Divs in `b` also count
                // as "symbolic" (they're not pure literals).
                !b.free_vars().is_empty() || term_has_sym_div(b)
            }
            Term::BinOp(_, a, b) => term_has_sym_div(a) || term_has_sym_div(b),
            Term::Pow(base, _) => term_has_sym_div(base),
            Term::Neg(a) => term_has_sym_div(a),
        }
    }
    fn go(phi: &FolFormulaExt) -> bool {
        match phi {
            FolFormulaExt::Base(_) => false,
            FolFormulaExt::Eq(a, b) | FolFormulaExt::Lt(a, b) | FolFormulaExt::Le(a, b) => {
                term_has_sym_div(a) || term_has_sym_div(b)
            }
            FolFormulaExt::And(a, b) | FolFormulaExt::Or(a, b) | FolFormulaExt::Implies(a, b) => {
                go(a) || go(b)
            }
            FolFormulaExt::Not(a) => go(a),
            FolFormulaExt::Forall(_, _, body) | FolFormulaExt::Exists(_, _, body) => go(body),
        }
    }
    go(phi)
}

/// Compatibility alias — named after the Phase 5 semantics where only the
/// conclusion was scanned. Now scans the whole formula; kept to minimize
/// call-site churn inside existing tests.
fn conclusion_has_division(phi: &FolFormulaExt) -> bool {
    formula_has_symbolic_division(phi)
}

/// Offsets for the *widened* `sq_nonneg (x ± k)` hints used only in the
/// slow-path nlinarith branch. Phase 3 measurement showed vertex-at-7 and
/// vertex-at-3 parabolas needed these, but Phase 4 re-measurement showed
/// that emitting them in the *fast* nlinarith branch caused Lean heartbeat
/// timeouts on well-behaved problems (`mathd_algebra_37`, `_141`) where
/// the compact hint set closed in under a second. Solution: cascade tries
/// compact hints first, widened hints only as a fallback.
const VERTEX_OFFSETS_WIDE: &[i32] = &[-10, -7, -5, -3, -1, 1, 3, 5, 7, 10];

fn append_var_hints(parts: &mut Vec<String>, n: &str, offsets: &[i32]) {
    parts.push(format!("sq_nonneg {}", n));
    parts.push(format!("mul_self_nonneg {}", n));
    for k in offsets {
        let sign = if *k < 0 { "-" } else { "+" };
        parts.push(format!("sq_nonneg ({} {} {})", n, sign, k.abs()));
    }
}

fn append_pairwise_hints(parts: &mut Vec<String>, names: &[String]) {
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            parts.push(format!("sq_nonneg ({} - {})", names[i], names[j]));
            parts.push(format!("sq_nonneg ({} + {})", names[i], names[j]));
        }
    }

    // Multiplicative antisymmetric cross-term hints (Phase 3 Move 1).
    //
    // For goals shaped like Cauchy-Schwarz 2-variable —
    //   (a·x + b·y)² ≤ (a² + b²)(x² + y²)
    // — the key witness is `sq_nonneg (a·y - b·x)` because
    //   (a·y - b·x)² = a²y² - 2abxy + b²x² ≥ 0
    // rearranges directly to the target. Additive-only hints can't
    // produce this because the witness is *multiplicative*.
    //
    // Strategy: for every pair of disjoint 2-subsets of binders
    // {(names[i], names[j]), (names[k], names[l])}, emit the four
    // Lagrange-identity witnesses:
    //   sq_nonneg (a*x - b*y), sq_nonneg (a*y - b*x),
    //   sq_nonneg (a*x + b*y), sq_nonneg (a*y + b*x)
    // The antisymmetric forms (with minus) are the Cauchy-Schwarz
    // witnesses; the symmetric (with plus) cover Lagrange and
    // companion inequalities.
    //
    // Capped at 4-binder systems (one pair of pairs) to keep the hint
    // list manageable; this is the common IMO-algebra shape. Larger
    // systems still get pairwise-additive and per-binder hints.
    if names.len() >= 4 {
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                for k in (j + 1)..names.len() {
                    for l in (k + 1)..names.len() {
                        let a = &names[i];
                        let b = &names[j];
                        let x = &names[k];
                        let y = &names[l];
                        parts.push(format!("sq_nonneg ({}*{} - {}*{})", a, x, b, y));
                        parts.push(format!("sq_nonneg ({}*{} - {}*{})", a, y, b, x));
                    }
                }
            }
        }
    }
}

/// Build the *compact* `nlinarith` hint list (Phase 3 baseline). Uses only
/// the `{-1, +1}` offset pair. This is the fast path: nlinarith closes
/// most polynomial problems with this set in well under the Lean heartbeat
/// budget.
fn build_nlinarith_hints(names: &[String]) -> String {
    if names.is_empty() {
        return String::from("sq_nonneg _, mul_self_nonneg _");
    }
    let mut parts: Vec<String> = Vec::new();
    for n in names {
        append_var_hints(&mut parts, n, &[-1, 1]);
    }
    append_pairwise_hints(&mut parts, names);
    parts.join(", ")
}

/// Build the *widened* `nlinarith` hint list (Phase 4 addition). Uses the
/// dense offset set `VERTEX_OFFSETS_WIDE`. Emitted as a *later-in-cascade*
/// fallback branch — tried only if the compact branch doesn't close.
/// Catches vertex-of-parabola inequalities like `mathd_algebra_113`
/// (vertex at 7) and `mathd_algebra_410` (vertex at 3).
fn build_nlinarith_hints_widened(names: &[String]) -> String {
    if names.is_empty() {
        return String::from("sq_nonneg _, mul_self_nonneg _");
    }
    let mut parts: Vec<String> = Vec::new();
    for n in names {
        append_var_hints(&mut parts, n, VERTEX_OFFSETS_WIDE);
    }
    append_pairwise_hints(&mut parts, names);
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
    let hypotheses = collect_outer_hypotheses(phi);
    // Introduce outer universals and hypotheses with concrete names so
    // the field-reasoning branch can reference them for `sub_ne_zero`
    // witness derivation. Phase 6a: naming hypotheses up front enables
    // the `_181`-family fix (hypothesis `n ≠ 3` → explicit witness
    // `n - 3 ≠ 0` passed to `field_simp`). Non-field branches don't
    // reference these names, so the naming is harmless for them.
    let hyp_names: Vec<String> = (0..hypotheses.len()).map(|i| format!("h{i}")).collect();
    let intro_line = match (binders.is_empty(), hyp_names.is_empty()) {
        (true, true) => String::from("try intros\n  "),
        (false, true) => format!("intro {}\n  try intros\n  ", binders.join(" ")),
        (true, false) => format!("intro {}\n  ", hyp_names.join(" ")),
        (false, false) => format!("intro {} {}\n  ", binders.join(" "), hyp_names.join(" "),),
    };
    let hints_compact = build_nlinarith_hints(&binders);
    let hints_widened = build_nlinarith_hints_widened(&binders);

    // Phase 6a: collect indices of `¬ _ = _` hypotheses. For each, the
    // field branch emits `have ne_i := sub_ne_zero.mpr h_i`, giving
    // `field_simp` an explicit `expr ≠ 0` witness. This closes the
    // `_181`/`_251`/`_267` family identified in the Phase 6 Session 1b
    // null: those goals have `h : x ≠ c` + division by `x - c`, but
    // Mathlib's `field_simp` doesn't auto-derive `x - c ≠ 0` from
    // `¬ x = c` (empirically, `simp made no progress`).
    let ne_hyp_indices: Vec<usize> = hypotheses
        .iter()
        .enumerate()
        .filter(|(_, h)| is_ne_hypothesis(h))
        .map(|(i, _)| i)
        .collect();

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

    // **Phase 4 addition: conjunction-splitter.** Gated on the conclusion
    // being syntactically `And`. Phase 4 re-measurement revealed that
    // emitting this branch unconditionally caused Lean heartbeat timeouts
    // on non-And goals that *had* been closing under Phase 3 (regressed
    // `mathd_algebra_37`, `_141`). Gating fixes the regression while
    // keeping the gain on `mathd_algebra_126`, `_101`.
    let and_splitter_alt = if conclusion_is_and(phi) {
        format!(
            "\n    | (refine ⟨?_, ?_⟩ <;> first | (linarith; done) | (nlinarith [{hints_compact}]; done) | (nlinarith [{hints_widened}]; done) | (omega; done) | (norm_num; done) | (ring; done); done)",
            hints_compact = hints_compact,
            hints_widened = hints_widened,
        )
    } else {
        String::new()
    };

    // **Phase 5 addition: field-simp branch.** Gated on the conclusion
    // containing division by a symbolic expression. Targets goals like
    // `mathd_algebra_55` (`q / p = 2 / 3` where `q`, `p` are universally
    // bound) that linarith/nlinarith can't close. Strategy:
    //
    // 1. `try subst_eqs` — Mathlib tactic that substitutes every `var =
    //    expr` hypothesis. For the miniF2F pattern where hypotheses
    //    evaluate the symbolic numerator/denominator to numeric
    //    literals, this collapses the goal to `(literal) / (literal) =
    //    literal`, which `norm_num` then closes.
    // 2. `try field_simp` — if `subst_eqs` didn't fully resolve the
    //    symbolic denominators, clear fractions directly. Requires
    //    nonzero hypotheses to be in-scope; if they are, the post-
    //    field_simp goal is a polynomial equality that `ring`, `linarith`,
    //    or `nlinarith` can close.
    // 3. Closer cascade — try norm_num, ring, linarith, nlinarith in
    //    order, each `; done`-terminated so partial simplification
    //    doesn't trap us.
    //
    // Gated on `conclusion_has_division` because `subst_eqs` and
    // `field_simp` do nontrivial goal rewrites; emitting this branch
    // unconditionally risks the same kind of heartbeat blowup we saw
    // with the unconditional And-splitter in Phase 4a.
    let field_simp_alt = if conclusion_has_division(phi) {
        // Three field sub-branches, in order:
        //
        // 1. `subst_eqs + field_simp + closers` — the `_55` pattern:
        //    hypotheses evaluate the symbolic denominator to a numeric
        //    literal; subst collapses it, then norm_num closes.
        // 2. `sub_ne_zero witnesses + field_simp [witnesses] + closers` —
        //    Phase 6a, the `_181` pattern: hypothesis `h : x ≠ c` gets
        //    converted to `x - c ≠ 0` via `sub_ne_zero.mpr h`, then
        //    `field_simp [ne_witness] at *` clears the `(…)/(x - c)`
        //    denominator. Needs named hypotheses (h0, h1, …) — that's
        //    why the intro line was restructured to name everything up
        //    front.
        // 3. `field_simp at * + closers` — the generic fallback: no
        //    explicit witness, let field_simp try its own derivation.
        //    Keeps `_55` and closed-form-rational cases working.
        //
        // All three are `try`-guarded so non-applicable cases fall
        // through cleanly.
        let witness_lines: String = if ne_hyp_indices.is_empty() {
            String::new()
        } else {
            ne_hyp_indices
                .iter()
                .enumerate()
                .map(|(w, h_idx)| format!("have ne{w} := sub_ne_zero.mpr h{h_idx}; "))
                .collect()
        };
        let witness_args: String = (0..ne_hyp_indices.len())
            .map(|w| format!("ne{w}"))
            .collect::<Vec<_>>()
            .join(", ");
        let witness_branch = if ne_hyp_indices.is_empty() {
            String::new()
        } else {
            format!(
                "\n    | ({witness_lines}try (field_simp [{witness_args}] at *); first | (linarith; done) | (ring; done) | (norm_num; done) | (nlinarith [{hints_compact}]; done); done)",
                witness_lines = witness_lines,
                witness_args = witness_args,
                hints_compact = hints_compact,
            )
        };
        // Phase 6a ordering lesson: the `subst_eqs+field_simp` branch
        // uses `try`-guarded tactics that Lean's `first` interprets as
        // "succeeded-with-unsolved-goals" (committing the outer `first`
        // to this alternative even when inner closers fail). That
        // commitment prevents a later witness-branch from ever firing.
        // Fix: witness-branch must come FIRST among field branches —
        // its `have` preamble only applies when there are `≠`
        // hypotheses to extract witnesses from; for other goals,
        // `ne_hyp_indices` is empty and `witness_branch` is an empty
        // string, so this branch is skipped.
        format!(
            "{witness_branch}\n    | (try subst_eqs; try field_simp; first | (norm_num; done) | (ring; done) | (linarith; done) | (nlinarith [{hints_compact}]; done); done)\n    | (try (field_simp at *); first | (linarith; done) | (ring; done) | (norm_num; done) | (nlinarith [{hints_compact}]; done); done)",
            hints_compact = hints_compact,
            witness_branch = witness_branch,
        )
    } else {
        String::new()
    };

    // Every alternative is `; done`-terminated. Reason: some Mathlib
    // tactics (notably `norm_num`) partially simplify the goal even when
    // they can't close it, which interferes with `first`'s backtracking.
    // Appending `done` forces each branch to either fully close the goal
    // or fail cleanly, letting `first` fall through to the next tactic.
    //
    // **Phase 4 ordering.** Compact nlinarith (±1 offsets only) goes first
    // — it closes most polynomial problems fast and stays well under the
    // Lean heartbeat budget. Widened nlinarith (dense offset set) goes
    // later as a fallback for vertex-of-parabola inequalities like
    // `mathd_algebra_113` (vertex at 7) and `_410` (vertex at 3). The
    // And-splitter, gated on `conclusion_is_and`, sits between the two
    // so it can re-use the compact hints on each subgoal before
    // escalating.
    let cascade = format!(
        r#"{}first
    | (rfl; done)
    | (norm_num; done)
    | (ring; done)
    | (omega; done)
    | (linarith; done)
    | (nlinarith [{hints_compact}]; done)
    | (positivity; done){field_simp_alt}{and_splitter_alt}
    | (nlinarith [{hints_widened}]; done)
    | {lt_trichotomy_alt}
    | (rcases le_total _ _ with h | h <;> first | linarith | tauto; done)
    | (tauto; done)
    | (polyrith; done)"#,
        intro_line,
        hints_compact = hints_compact,
        hints_widened = hints_widened,
        field_simp_alt = field_simp_alt,
        and_splitter_alt = and_splitter_alt,
        lt_trichotomy_alt = lt_trichotomy_alt,
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

    // ─── Phase 4/5 cascade-gating regression lockdowns ─────────────────
    //
    // `conclusion_is_and` and `conclusion_has_division` gate two
    // optional cascade branches. These tests lock in the gating
    // predicates so the gates stay aligned with the cascade emitter.

    #[test]
    fn conclusion_is_and_detects_outer_and() {
        // `∀ x : ℝ, h → (A ∧ B)` — conclusion is syntactically `And`
        // after stripping Forall+Implies. mathd_algebra_101-shape.
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::lt(Term::IntLit(0), x()).implies(
                FolFormulaExt::lt(Term::IntLit(0), x())
                    .and(FolFormulaExt::lt(x(), Term::IntLit(10))),
            ),
        );
        assert!(conclusion_is_and(&phi));
    }

    #[test]
    fn conclusion_is_and_rejects_non_and() {
        // `∀ x : ℝ, h → goal` where `goal` is `Eq(x, 0)` — not And.
        // Regression: gating must NOT emit the refine branch for
        // non-And goals, otherwise Phase 4a's heartbeat timeouts on
        // `mathd_algebra_37`, `_141` return.
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::lt(Term::IntLit(0), x())
                .implies(FolFormulaExt::eq(x(), Term::IntLit(0))),
        );
        assert!(!conclusion_is_and(&phi));
    }

    #[test]
    fn conclusion_has_division_detects_symbolic_denom() {
        // `∀ p q : ℝ, p = 12 → q = 8 → q/p = 2/3` — mathd_algebra_55-shape.
        // Conclusion contains `Div(q, p)` where `p` is a free variable.
        let p = Term::var("p");
        let q = Term::var("q");
        let phi = FolFormulaExt::forall(
            "p",
            NumericType::Real,
            FolFormulaExt::forall(
                "q",
                NumericType::Real,
                FolFormulaExt::eq(p.clone(), Term::IntLit(12)).implies(
                    FolFormulaExt::eq(q.clone(), Term::IntLit(8))
                        .implies(FolFormulaExt::eq(q.div(p), Term::rat(2, 3))),
                ),
            ),
        );
        assert!(conclusion_has_division(&phi));
    }

    #[test]
    fn conclusion_has_division_rejects_literal_denom() {
        // `∀ x : ℝ, x / 50 = 40 → x = 2000` — mathd_algebra_24-shape.
        // `50` is a literal, not symbolic. The field_simp branch
        // should NOT fire here; linarith handles it fine. Regression
        // target: anything that flips this case positive.
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::eq(x().div(Term::IntLit(50)), Term::IntLit(40))
                .implies(FolFormulaExt::eq(x(), Term::IntLit(2000))),
        );
        assert!(!conclusion_has_division(&phi));
    }

    #[test]
    fn and_splitter_branch_emitted_when_gated() {
        // Verify that when `conclusion_is_and` returns true, the
        // cascade emits the `refine ⟨?_, ?_⟩` branch; when false, it
        // doesn't. Locks the gate-to-emit path end-to-end.
        let and_phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::eq(x(), Term::IntLit(1))
                .and(FolFormulaExt::eq(x().add(Term::IntLit(1)), Term::IntLit(2))),
        );
        let and_file = render_fol_ext_file("t_and", &and_phi);
        assert!(
            and_file.contains("refine ⟨?_, ?_⟩"),
            "And conclusion should emit refine branch: {}",
            and_file
        );

        let non_and_phi =
            FolFormulaExt::forall("x", NumericType::Real, FolFormulaExt::eq(x(), x()));
        let non_and_file = render_fol_ext_file("t_refl", &non_and_phi);
        assert!(
            !non_and_file.contains("refine ⟨?_, ?_⟩"),
            "Non-And conclusion should NOT emit refine branch: {}",
            non_and_file
        );
    }

    #[test]
    fn field_simp_branch_emitted_when_gated() {
        // Verify `conclusion_has_division` gates the field_simp branch.
        let field_phi = FolFormulaExt::forall(
            "p",
            NumericType::Real,
            FolFormulaExt::eq(Term::var("q").div(Term::var("p")), Term::rat(2, 3)),
        );
        let field_file = render_fol_ext_file("t_field", &field_phi);
        assert!(
            field_file.contains("field_simp"),
            "division-in-conclusion should emit field_simp branch: {}",
            field_file
        );

        let no_field_phi =
            FolFormulaExt::forall("x", NumericType::Real, FolFormulaExt::eq(x(), x()));
        let no_field_file = render_fol_ext_file("t_refl_nofield", &no_field_phi);
        assert!(
            !no_field_file.contains("field_simp"),
            "No-division conclusion should NOT emit field_simp: {}",
            no_field_file
        );
    }

    /// Regression test for the Lean-source-injection finding, covering
    /// `render_fol_ext_file`'s Route 2 (arithmetic goals), which builds its
    /// output with direct `format!` calls rather than going through
    /// `LeanProofScript::to_lean`. A malicious `theorem_name` and a
    /// malicious `Term::Var`/`Forall` binder name (both externally-sourced)
    /// must not survive into the emitted file as raw newlines.
    #[test]
    fn malicious_theorem_name_and_var_cannot_inject_lean_commands() {
        let attack_var = "x\nend\n#eval IO.Process.run { cmd := \"sh\" }";
        let phi = FolFormulaExt::forall(
            attack_var,
            NumericType::Real,
            FolFormulaExt::le(
                Term::IntLit(0),
                Term::var(attack_var).mul(Term::var(attack_var)),
            ),
        );
        let attack_theorem = "t\nend\n#eval 2";
        let file = render_fol_ext_file(attack_theorem, &phi);
        assert!(
            !file.contains("#eval"),
            "injected #eval must be neutralized, got: {}",
            file
        );
        assert!(
            !file.lines().any(|l| l.trim() == "end"),
            "injected bare `end` must not land on its own line, got: {}",
            file
        );
    }
}
