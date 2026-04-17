// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # SMT-LIB2 serializer for `FolFormulaExt`
//!
//! Emits self-contained SMT-LIB2 obligations in the appropriate logic
//! fragment (`QF_LIA`, `QF_LRA`, `QF_NIA`, `QF_NRA`, `LIA`, `LRA`, `NIA`,
//! `NRA`). Feeds straight into Z3 for decision — same subprocess pattern
//! as `verify_invariants_formal`, no external dependencies beyond `z3`
//! on PATH.
//!
//! ## Fragment selection
//!
//! | Quantifiers | Integral | Non-linear | Fragment |
//! |-------------|----------|------------|----------|
//! | no          | yes      | no         | `QF_LIA` |
//! | no          | yes      | yes        | `QF_NIA` |
//! | no          | no       | no         | `QF_LRA` |
//! | no          | no       | yes        | `QF_NRA` |
//! | yes         | yes      | no         | `LIA`    |
//! | yes         | yes      | yes        | `NIA`    |
//! | yes         | no       | no         | `LRA`    |
//! | yes         | no       | yes        | `NRA`    |
//!
//! "Integral" = every literal is an `IntLit` and every free-var type is
//! `Int` / `Nat` (i.e. no `RealLit`, `RatLit`, or `Real` bindings).

use crate::hdc::fol_formula_ext::{
    ArithOp, FolFormulaExt, NumericType, Term,
};
use std::fmt::Write;

// ═════════════════════════════════════════════════════════════════════════
// Fragment detection
// ═════════════════════════════════════════════════════════════════════════

/// The SMT logic fragment best suited to a given formula. Drives
/// `(set-logic ...)` emission and tactic choice downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmtFragment {
    QfLia,
    QfLra,
    QfNia,
    QfNra,
    Lia,
    Lra,
    Nia,
    Nra,
}

impl SmtFragment {
    pub fn logic_name(self) -> &'static str {
        match self {
            SmtFragment::QfLia => "QF_LIA",
            SmtFragment::QfLra => "QF_LRA",
            SmtFragment::QfNia => "QF_NIA",
            SmtFragment::QfNra => "QF_NRA",
            SmtFragment::Lia => "LIA",
            SmtFragment::Lra => "LRA",
            SmtFragment::Nia => "NIA",
            SmtFragment::Nra => "NRA",
        }
    }

    /// Phase 2 target Lean tactic for this fragment. Used by the bridge
    /// when converting a Z3-closable goal into a Lean `.lean` file. All
    /// of these require Mathlib; emit an `import Mathlib.Tactic` header.
    pub fn suggested_lean_tactic(self) -> &'static str {
        match self {
            SmtFragment::QfLia | SmtFragment::Lia => "omega",
            SmtFragment::QfLra | SmtFragment::Lra => "linarith",
            SmtFragment::QfNia | SmtFragment::Nia => "omega_nat", // bounded heuristic
            SmtFragment::QfNra | SmtFragment::Nra => "nlinarith",
        }
    }
}

/// Determine the SMT logic fragment for a formula.
pub fn detect_fragment(phi: &FolFormulaExt) -> SmtFragment {
    let has_quantifier = contains_quantifier(phi);
    let is_integral = formula_is_integral(phi);
    let is_non_linear = formula_is_non_linear(phi);

    match (has_quantifier, is_integral, is_non_linear) {
        (false, true, false) => SmtFragment::QfLia,
        (false, true, true) => SmtFragment::QfNia,
        (false, false, false) => SmtFragment::QfLra,
        (false, false, true) => SmtFragment::QfNra,
        (true, true, false) => SmtFragment::Lia,
        (true, true, true) => SmtFragment::Nia,
        (true, false, false) => SmtFragment::Lra,
        (true, false, true) => SmtFragment::Nra,
    }
}

fn contains_quantifier(phi: &FolFormulaExt) -> bool {
    match phi {
        FolFormulaExt::Forall(_, _, _) | FolFormulaExt::Exists(_, _, _) => true,
        FolFormulaExt::Base(_) | FolFormulaExt::Eq(_, _)
        | FolFormulaExt::Lt(_, _) | FolFormulaExt::Le(_, _) => false,
        FolFormulaExt::Not(a) => contains_quantifier(a),
        FolFormulaExt::And(a, b)
        | FolFormulaExt::Or(a, b)
        | FolFormulaExt::Implies(a, b) => contains_quantifier(a) || contains_quantifier(b),
    }
}

fn formula_is_integral(phi: &FolFormulaExt) -> bool {
    match phi {
        FolFormulaExt::Base(_) => true,
        FolFormulaExt::Eq(a, b) | FolFormulaExt::Lt(a, b) | FolFormulaExt::Le(a, b) => {
            a.is_integral() && b.is_integral()
        }
        FolFormulaExt::Not(a) => formula_is_integral(a),
        FolFormulaExt::And(a, b)
        | FolFormulaExt::Or(a, b)
        | FolFormulaExt::Implies(a, b) => formula_is_integral(a) && formula_is_integral(b),
        FolFormulaExt::Forall(_, ty, body) | FolFormulaExt::Exists(_, ty, body) => {
            matches!(ty, NumericType::Int | NumericType::Nat) && formula_is_integral(body)
        }
    }
}

fn formula_is_non_linear(phi: &FolFormulaExt) -> bool {
    match phi {
        FolFormulaExt::Base(_) => false,
        FolFormulaExt::Eq(a, b) | FolFormulaExt::Lt(a, b) | FolFormulaExt::Le(a, b) => {
            a.is_non_linear() || b.is_non_linear()
        }
        FolFormulaExt::Not(a) => formula_is_non_linear(a),
        FolFormulaExt::And(a, b)
        | FolFormulaExt::Or(a, b)
        | FolFormulaExt::Implies(a, b) => formula_is_non_linear(a) || formula_is_non_linear(b),
        FolFormulaExt::Forall(_, _, body) | FolFormulaExt::Exists(_, _, body) => {
            formula_is_non_linear(body)
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Term → SMT-LIB2
// ═════════════════════════════════════════════════════════════════════════

pub fn term_to_smt(t: &Term) -> String {
    match t {
        Term::Var(n) => n.clone(),
        Term::IntLit(n) => {
            if *n >= 0 {
                format!("{}", n)
            } else {
                format!("(- {})", -n)
            }
        }
        Term::RealLit(x) => {
            if *x >= 0.0 {
                format!("{}", x)
            } else {
                format!("(- {})", -x)
            }
        }
        Term::RatLit(p, q) => {
            // Exact rational as (/ p q) — handles 1/3 that f64 cannot
            // represent exactly.
            let p_str = if *p >= 0 {
                format!("{}", p)
            } else {
                format!("(- {})", -p)
            };
            let q_str = if *q >= 0 {
                format!("{}", q)
            } else {
                format!("(- {})", -q)
            };
            format!("(/ {} {})", p_str, q_str)
        }
        Term::BinOp(op, a, b) => {
            format!("({} {} {})", op.symbol(), term_to_smt(a), term_to_smt(b))
        }
        Term::Pow(base, exp) => {
            // SMT-LIB has no native power; expand to repeated multiplication.
            let s = term_to_smt(base);
            match *exp {
                0 => "1".to_string(),
                1 => s,
                n => {
                    let mut out = s.clone();
                    for _ in 1..n {
                        out = format!("(* {} {})", s, out);
                    }
                    out
                }
            }
        }
        Term::Neg(a) => format!("(- {})", term_to_smt(a)),
    }
}

// ═════════════════════════════════════════════════════════════════════════
// FolFormulaExt → SMT-LIB2 body (assertion form)
// ═════════════════════════════════════════════════════════════════════════

/// Serialize a formula as an SMT-LIB2 assertion body. Does NOT emit
/// `(assert …)` or logic headers — those are added by `encode_as_query`.
pub fn formula_to_smt(phi: &FolFormulaExt) -> String {
    match phi {
        FolFormulaExt::Base(_) => {
            // Pure-propositional base can't be rendered here without a
            // propositional atom-map. Phase 2 does not need to emit Base
            // propositions to SMT — those are routed back to the Phase 1
            // synthesizer. Emit `true` as a no-op placeholder.
            "true".to_string()
        }
        FolFormulaExt::Eq(a, b) => {
            format!("(= {} {})", term_to_smt(a), term_to_smt(b))
        }
        FolFormulaExt::Lt(a, b) => {
            format!("(< {} {})", term_to_smt(a), term_to_smt(b))
        }
        FolFormulaExt::Le(a, b) => {
            format!("(<= {} {})", term_to_smt(a), term_to_smt(b))
        }
        FolFormulaExt::And(a, b) => {
            format!("(and {} {})", formula_to_smt(a), formula_to_smt(b))
        }
        FolFormulaExt::Or(a, b) => {
            format!("(or {} {})", formula_to_smt(a), formula_to_smt(b))
        }
        FolFormulaExt::Not(a) => {
            format!("(not {})", formula_to_smt(a))
        }
        FolFormulaExt::Implies(a, b) => {
            format!("(=> {} {})", formula_to_smt(a), formula_to_smt(b))
        }
        FolFormulaExt::Forall(name, ty, body) => {
            let body_str = match ty.constraint_for(name) {
                Some(c) => format!("(=> {} {})", c, formula_to_smt(body)),
                None => formula_to_smt(body),
            };
            format!(
                "(forall (({} {})) {})",
                name,
                ty.smt_sort(),
                body_str
            )
        }
        FolFormulaExt::Exists(name, ty, body) => {
            let body_str = match ty.constraint_for(name) {
                Some(c) => format!("(and {} {})", c, formula_to_smt(body)),
                None => formula_to_smt(body),
            };
            format!(
                "(exists (({} {})) {})",
                name,
                ty.smt_sort(),
                body_str
            )
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Full obligation builder
// ═════════════════════════════════════════════════════════════════════════

/// Build a complete SMT-LIB2 obligation that asserts `¬φ` and calls
/// `(check-sat)`. A return of `unsat` from Z3 is a formal proof of `φ`.
///
/// - Emits `(set-logic ...)` with the detected fragment.
/// - Declares every free arithmetic variable with the inferred sort.
/// - For `Nat`-typed variables, emits the `(>= n 0)` side-constraint.
/// - Asserts `(not φ)` and calls `(check-sat)`.
pub fn encode_as_query(phi: &FolFormulaExt) -> String {
    let fragment = detect_fragment(phi);
    let free = phi.free_arith_vars();

    let mut out = String::new();
    writeln!(out, "(set-logic {})", fragment.logic_name()).unwrap();
    writeln!(out, "; auto-generated by symthaea-core::hdc::fol_ext_smt").unwrap();
    writeln!(out, "; obligation: prove φ by asserting ¬φ and checking UNSAT").unwrap();
    writeln!(out).unwrap();
    for (name, ty) in &free {
        writeln!(out, "(declare-const {} {})", name, ty.smt_sort()).unwrap();
    }
    for (name, ty) in &free {
        if let Some(c) = ty.constraint_for(name) {
            writeln!(out, "(assert {})", c).unwrap();
        }
    }
    writeln!(out).unwrap();
    writeln!(out, "(assert (not {}))", formula_to_smt(phi)).unwrap();
    writeln!(out, "(check-sat)").unwrap();
    out
}

// ═════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> Term {
        Term::var("x")
    }
    fn y() -> Term {
        Term::var("y")
    }

    // ─── Term serialization ────────────────────────────────────────────

    #[test]
    fn pos_int_literal() {
        assert_eq!(term_to_smt(&Term::IntLit(5)), "5");
    }

    #[test]
    fn neg_int_literal() {
        assert_eq!(term_to_smt(&Term::IntLit(-3)), "(- 3)");
    }

    #[test]
    fn rat_literal_exact() {
        assert_eq!(term_to_smt(&Term::rat(1, 3)), "(/ 1 3)");
    }

    #[test]
    fn rat_literal_negative_num() {
        assert_eq!(term_to_smt(&Term::rat(-1, 3)), "(/ (- 1) 3)");
    }

    #[test]
    fn pow_zero_is_one() {
        assert_eq!(term_to_smt(&x().pow(0)), "1");
    }

    #[test]
    fn pow_one_is_base() {
        assert_eq!(term_to_smt(&x().pow(1)), "x");
    }

    #[test]
    fn pow_three_expands_to_mult() {
        // x^3 → (* x (* x x))
        let expanded = term_to_smt(&x().pow(3));
        assert!(expanded.contains("* x"));
        assert_eq!(expanded.matches("x").count(), 3);
    }

    #[test]
    fn linear_sum() {
        assert_eq!(
            term_to_smt(&x().add(y())),
            "(+ x y)"
        );
    }

    // ─── Fragment detection ────────────────────────────────────────────

    #[test]
    fn qf_lia_for_linear_int() {
        // 0 ≤ x + 1
        let phi = FolFormulaExt::le(Term::IntLit(0), x().add(Term::IntLit(1)));
        assert_eq!(detect_fragment(&phi), SmtFragment::QfLia);
    }

    #[test]
    fn qf_lra_for_linear_real() {
        // 0.5 < x
        let phi = FolFormulaExt::lt(Term::real(0.5), x());
        assert_eq!(detect_fragment(&phi), SmtFragment::QfLra);
    }

    #[test]
    fn qf_nia_for_nonlinear_int() {
        // x * y = 0 where both x, y are integer variables
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Int,
            FolFormulaExt::forall(
                "y",
                NumericType::Int,
                FolFormulaExt::eq(x().mul(y()), Term::IntLit(0)),
            ),
        );
        // This has forall → LIA quantified. But it's non-linear (x*y).
        assert_eq!(detect_fragment(&phi), SmtFragment::Nia);
    }

    #[test]
    fn qf_nra_for_nonlinear_real() {
        // x^2 ≥ 0 (no quantifier → QF_NRA)
        let phi = FolFormulaExt::le(Term::real(0.0), x().pow(2));
        assert_eq!(detect_fragment(&phi), SmtFragment::QfNra);
    }

    // ─── Full obligation encoding ──────────────────────────────────────

    #[test]
    fn encodes_reflexivity_of_equality() {
        // ∀ x : Real, x = x
        let phi = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::eq(x(), x()),
        );
        let smt = encode_as_query(&phi);
        assert!(smt.contains("(set-logic LRA)"));
        assert!(smt.contains("forall"));
        assert!(smt.contains("(= x x)"));
        assert!(smt.contains("(check-sat)"));
    }

    #[test]
    fn nat_binding_adds_constraint_under_forall() {
        // ∀ n : Nat, 0 ≤ n
        let phi = FolFormulaExt::forall(
            "n",
            NumericType::Nat,
            FolFormulaExt::le(Term::IntLit(0), Term::var("n")),
        );
        let smt = encode_as_query(&phi);
        // Under forall, the Nat constraint (≥ n 0) guards the body
        // via implication: (forall ((n Int)) (=> (>= n 0) (<= 0 n)))
        assert!(smt.contains("(=> (>= n 0)"), "got: {}", smt);
    }

    #[test]
    fn free_var_decls_emitted() {
        // x + 1 > x (free `x : Real` by default)
        let phi = FolFormulaExt::lt(x(), x().add(Term::IntLit(1)));
        let smt = encode_as_query(&phi);
        assert!(smt.contains("(declare-const x Real)"), "got: {}", smt);
    }

    #[test]
    fn negation_wraps_goal() {
        let phi = FolFormulaExt::eq(Term::IntLit(1), Term::IntLit(1));
        let smt = encode_as_query(&phi);
        assert!(
            smt.contains("(assert (not (= 1 1)))"),
            "got: {}",
            smt
        );
    }
}
