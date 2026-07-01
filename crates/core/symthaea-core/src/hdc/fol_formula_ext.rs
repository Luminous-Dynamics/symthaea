// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # FolFormulaExt: first-order logic with arithmetic
//!
//! Phase 2 extension to `Proposition` that adds the expressiveness needed
//! to represent miniF2F-style algebraic theorems. The design keeps
//! backward compatibility via a `Base(Proposition)` variant so every
//! Phase 1 fixture remains in scope without modification.
//!
//! ## What this module adds beyond `Proposition`
//!
//! - **Arithmetic terms** (`Term`) over ℤ, ℕ, ℝ: literals, variables,
//!   addition, subtraction, multiplication, integer powers, negation.
//! - **Equality and ordering** (`Eq`, `Lt`, `Le`) on terms.
//! - **Typed quantification** (`Forall`, `Exists`) with a `NumericType`
//!   binding so the SMT serializer can pick the right logic fragment
//!   (`Int` → `QF_LIA`/`LIA`, `Real` → `QF_LRA`/`LRA`, `Nat` → `Int` with
//!   a non-negativity guard).
//!
//! ## What this module does NOT add
//!
//! Transcendentals (sin, cos, log, exp), calculus (deriv, integral), sets
//! (Finset, Set), algebraic structures (Ring, Field), and number-theory
//! lemmas (gcd, prime) are all explicitly deferred to Phase 3+. See
//! `docs/phase2-algebraic-reasoning-plan.md` for the scope matrix.
//!
//! ## Companion modules
//!
//! - `fol_ext_smt.rs` — SMT-LIB2 serializer with fragment detection.
//! - `fol_ext_synth.rs` — proof-term synthesis for goals the arithmetic
//!   extension can close in-house before delegating to Z3 (Phase 2 week 3+).

use crate::hdc::logic_engine::Proposition;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

// ═════════════════════════════════════════════════════════════════════════
// Numeric types
// ═════════════════════════════════════════════════════════════════════════

/// Numeric type annotation for quantifier bindings and term literals.
///
/// `Nat` is represented as `Int` in SMT with an additional `(>= n 0)`
/// constraint; this avoids dragging in a distinct SMT logic for naturals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericType {
    Int,
    Real,
    Nat,
}

impl NumericType {
    /// The SMT sort name for this type.
    pub fn smt_sort(self) -> &'static str {
        match self {
            NumericType::Int | NumericType::Nat => "Int",
            NumericType::Real => "Real",
        }
    }

    /// If this type adds a side-constraint at declaration time (e.g. `Nat`
    /// requires `n ≥ 0`), return the SMT-LIB2 fragment for it.
    pub fn constraint_for(self, var: &str) -> Option<String> {
        match self {
            NumericType::Nat => Some(format!("(>= {} 0)", var)),
            _ => None,
        }
    }
}

impl fmt::Display for NumericType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericType::Int => write!(f, "ℤ"),
            NumericType::Real => write!(f, "ℝ"),
            NumericType::Nat => write!(f, "ℕ"),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Arithmetic operators
// ═════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    /// Division: semantically valid for `Real`. For `Int`/`Nat`, represents
    /// truncated division; higher layers should guard this appropriately.
    Div,
}

impl ArithOp {
    pub fn symbol(self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Arithmetic terms
// ═════════════════════════════════════════════════════════════════════════

/// Arithmetic term over ℤ/ℕ/ℝ. Serialization to SMT-LIB2 lives in
/// `fol_ext_smt.rs`; this enum is a pure AST.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Term {
    /// Free variable. Binding is via an enclosing `Forall`/`Exists` or
    /// declaration context.
    Var(String),
    /// Integer literal.
    IntLit(i64),
    /// Real literal. NB: precision is limited to f64; see the Phase 1 note
    /// on `1/3` serialization for the workaround (rational literal encoding
    /// arrives in Phase 2 week 2 via `RatLit`).
    RealLit(f64),
    /// Exact rational literal, emitted as `(/ p q)` in SMT-LIB2. Resolves
    /// the f64-precision issue documented in the Phase 1 Hénon-Heiles
    /// ×6 rescale.
    RatLit(i64, i64),
    /// Binary arithmetic.
    BinOp(ArithOp, Box<Term>, Box<Term>),
    /// Integer power `t^n` with non-negative integer exponent. Negative
    /// exponents → use `BinOp(Div, IntLit(1), Pow(t, |n|))`.
    Pow(Box<Term>, u32),
    /// Negation.
    Neg(Box<Term>),
}

impl Term {
    pub fn var(name: &str) -> Self {
        Term::Var(name.to_string())
    }
    pub fn int(n: i64) -> Self {
        Term::IntLit(n)
    }
    pub fn real(x: f64) -> Self {
        Term::RealLit(x)
    }
    pub fn rat(p: i64, q: i64) -> Self {
        assert!(q != 0, "denominator must be non-zero");
        Term::RatLit(p, q)
    }

    pub fn add(self, other: Term) -> Self {
        Term::BinOp(ArithOp::Add, Box::new(self), Box::new(other))
    }
    pub fn sub(self, other: Term) -> Self {
        Term::BinOp(ArithOp::Sub, Box::new(self), Box::new(other))
    }
    pub fn mul(self, other: Term) -> Self {
        Term::BinOp(ArithOp::Mul, Box::new(self), Box::new(other))
    }
    pub fn div(self, other: Term) -> Self {
        Term::BinOp(ArithOp::Div, Box::new(self), Box::new(other))
    }
    pub fn pow(self, exp: u32) -> Self {
        Term::Pow(Box::new(self), exp)
    }
    pub fn neg(self) -> Self {
        Term::Neg(Box::new(self))
    }

    /// Collect free variables in stable (sorted) order.
    pub fn free_vars(&self) -> Vec<String> {
        let mut s: BTreeSet<String> = BTreeSet::new();
        self.collect_free_vars(&mut s);
        s.into_iter().collect()
    }

    fn collect_free_vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Term::Var(n) => {
                out.insert(n.clone());
            }
            Term::IntLit(_) | Term::RealLit(_) | Term::RatLit(_, _) => {}
            Term::BinOp(_, a, b) => {
                a.collect_free_vars(out);
                b.collect_free_vars(out);
            }
            Term::Pow(base, _) => base.collect_free_vars(out),
            Term::Neg(a) => a.collect_free_vars(out),
        }
    }

    /// Does this term use any `Real` or `Rat` literal? Drives fragment
    /// detection (`QF_LIA` vs `QF_LRA`).
    pub fn is_integral(&self) -> bool {
        match self {
            Term::Var(_) | Term::IntLit(_) => true,
            Term::RealLit(_) | Term::RatLit(_, _) => false,
            Term::BinOp(_, a, b) => a.is_integral() && b.is_integral(),
            Term::Pow(base, _) => base.is_integral(),
            Term::Neg(a) => a.is_integral(),
        }
    }

    /// Is this term a "simple" leaf from the SMT-LIB linear-fragment
    /// perspective — i.e. a variable or a single numeric literal?
    /// Compound expressions (sums, products, powers, negations) are NOT
    /// simple. Used by `is_non_linear` to detect cases Z3's `QF_LRA`
    /// parser rejects even when mathematically linear.
    fn is_simple_leaf(&self) -> bool {
        matches!(
            self,
            Term::Var(_) | Term::IntLit(_) | Term::RealLit(_) | Term::RatLit(_, _)
        )
    }

    /// Does this term fall outside the linear SMT fragment? Rule is
    /// conservative for Z3's benefit:
    ///
    /// - `Mul`: non-linear if both sides contain variables (classic
    ///   `x * y` case), OR if both sides are compound expressions.
    ///   Z3's `QF_LRA` parser rejects `(* (+ a b) (+ c d))` even when
    ///   the arguments evaluate to constants, because the parser
    ///   doesn't do constant folding up front. `mathd_algebra_462`
    ///   (`(1/2 + 1/3) * (1/2 - 1/3) = 5/36`) hit exactly this case:
    ///   both operands are `Add`/`Sub` compounds with no free variables,
    ///   but Z3 still errored with "logic does not support nonlinear
    ///   arithmetic". Flagging `(compound * compound)` as non-linear
    ///   routes the obligation through `QF_NRA`, which Z3 accepts.
    /// - `Div`: non-linear if the denominator contains a free variable.
    /// - `Pow`: non-linear if the base contains a variable and the
    ///   exponent is ≥ 2.
    pub fn is_non_linear(&self) -> bool {
        match self {
            Term::Var(_) | Term::IntLit(_) | Term::RealLit(_) | Term::RatLit(_, _) => false,
            Term::BinOp(ArithOp::Mul, a, b) => {
                let a_has = !a.free_vars().is_empty();
                let b_has = !b.free_vars().is_empty();
                let both_vars = a_has && b_has;
                let both_compound = !a.is_simple_leaf() && !b.is_simple_leaf();
                both_vars || both_compound || a.is_non_linear() || b.is_non_linear()
            }
            Term::BinOp(ArithOp::Div, a, b) => {
                // division by a variable is non-linear
                !b.free_vars().is_empty() || a.is_non_linear() || b.is_non_linear()
            }
            Term::BinOp(_, a, b) => a.is_non_linear() || b.is_non_linear(),
            Term::Pow(base, exp) => {
                // x^n with n >= 2 and x variable is non-linear
                (*exp >= 2 && !base.free_vars().is_empty()) || base.is_non_linear()
            }
            Term::Neg(a) => a.is_non_linear(),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(n) => write!(f, "{}", n),
            Term::IntLit(n) => write!(f, "{}", n),
            Term::RealLit(x) => write!(f, "{}", x),
            Term::RatLit(p, q) => write!(f, "{}/{}", p, q),
            Term::BinOp(op, a, b) => write!(f, "({} {} {})", a, op.symbol(), b),
            Term::Pow(base, exp) => write!(f, "{}^{}", base, exp),
            Term::Neg(a) => write!(f, "-({})", a),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Extended FOL formula
// ═════════════════════════════════════════════════════════════════════════

/// Extended first-order logic formula: subsumes `Proposition` via `Base`
/// and adds arithmetic predicates and typed quantifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FolFormulaExt {
    /// Inherit every Phase 1 `Proposition`. No arithmetic here.
    Base(Proposition),

    /// Arithmetic equality.
    Eq(Term, Term),
    /// Strict ordering.
    Lt(Term, Term),
    /// Non-strict ordering.
    Le(Term, Term),

    /// Conjunction (over `FolFormulaExt`).
    And(Box<FolFormulaExt>, Box<FolFormulaExt>),
    /// Disjunction.
    Or(Box<FolFormulaExt>, Box<FolFormulaExt>),
    /// Negation.
    Not(Box<FolFormulaExt>),
    /// Implication.
    Implies(Box<FolFormulaExt>, Box<FolFormulaExt>),

    /// `∀ x : T, φ`. The `NumericType` determines the SMT sort and any
    /// non-negativity side-constraints (for `Nat`).
    Forall(String, NumericType, Box<FolFormulaExt>),
    /// `∃ x : T, φ`.
    Exists(String, NumericType, Box<FolFormulaExt>),
}

impl FolFormulaExt {
    /// Inject a `Proposition` as a pure-boolean `FolFormulaExt`.
    pub fn from_prop(p: Proposition) -> Self {
        FolFormulaExt::Base(p)
    }

    pub fn eq(lhs: Term, rhs: Term) -> Self {
        FolFormulaExt::Eq(lhs, rhs)
    }
    pub fn lt(lhs: Term, rhs: Term) -> Self {
        FolFormulaExt::Lt(lhs, rhs)
    }
    pub fn le(lhs: Term, rhs: Term) -> Self {
        FolFormulaExt::Le(lhs, rhs)
    }

    pub fn implies(self, rhs: FolFormulaExt) -> Self {
        FolFormulaExt::Implies(Box::new(self), Box::new(rhs))
    }
    pub fn and(self, rhs: FolFormulaExt) -> Self {
        FolFormulaExt::And(Box::new(self), Box::new(rhs))
    }
    pub fn or(self, rhs: FolFormulaExt) -> Self {
        FolFormulaExt::Or(Box::new(self), Box::new(rhs))
    }
    pub fn neg(self) -> Self {
        FolFormulaExt::Not(Box::new(self))
    }

    pub fn forall(var: &str, ty: NumericType, body: FolFormulaExt) -> Self {
        FolFormulaExt::Forall(var.to_string(), ty, Box::new(body))
    }
    pub fn exists(var: &str, ty: NumericType, body: FolFormulaExt) -> Self {
        FolFormulaExt::Exists(var.to_string(), ty, Box::new(body))
    }

    /// Collect the set of `(name, type)` pairs for every **free** variable
    /// that appears in an arithmetic subterm. Quantifier-bound variables
    /// are excluded. Used by the SMT serializer to emit `declare-const`.
    pub fn free_arith_vars(&self) -> Vec<(String, NumericType)> {
        let mut out: Vec<(String, NumericType)> = Vec::new();
        let mut bound: BTreeSet<String> = BTreeSet::new();
        // Default numeric type for unbound vars is Real — SMT handles
        // mixed Int/Real in polynomial contexts only in QF_NIRA, which we
        // don't target here. Callers that want Int-typed free vars should
        // wrap the formula in a Forall binding for that variable.
        self.walk_free_vars(&mut out, &mut bound, NumericType::Real);
        // dedup by name, preserving first-seen type
        let mut seen: BTreeSet<String> = BTreeSet::new();
        out.retain(|(n, _)| seen.insert(n.clone()));
        out
    }

    fn walk_free_vars(
        &self,
        out: &mut Vec<(String, NumericType)>,
        bound: &mut BTreeSet<String>,
        default_ty: NumericType,
    ) {
        match self {
            FolFormulaExt::Base(_) => { /* propositional atoms aren't arithmetic */ }
            FolFormulaExt::Eq(a, b) | FolFormulaExt::Lt(a, b) | FolFormulaExt::Le(a, b) => {
                for v in a.free_vars().into_iter().chain(b.free_vars()) {
                    if !bound.contains(&v) {
                        out.push((v, default_ty));
                    }
                }
            }
            FolFormulaExt::And(a, b) | FolFormulaExt::Or(a, b) | FolFormulaExt::Implies(a, b) => {
                a.walk_free_vars(out, bound, default_ty);
                b.walk_free_vars(out, bound, default_ty);
            }
            FolFormulaExt::Not(a) => a.walk_free_vars(out, bound, default_ty),
            FolFormulaExt::Forall(name, ty, body) | FolFormulaExt::Exists(name, ty, body) => {
                let fresh = bound.insert(name.clone());
                body.walk_free_vars(out, bound, *ty);
                if fresh {
                    bound.remove(name);
                }
            }
        }
    }

    /// Does this formula contain any arithmetic at all? If not, it's
    /// essentially a boxed `Proposition` and can be routed to the Phase 1
    /// synthesizer.
    pub fn is_purely_propositional(&self) -> bool {
        match self {
            FolFormulaExt::Base(_) => true,
            FolFormulaExt::Eq(_, _) | FolFormulaExt::Lt(_, _) | FolFormulaExt::Le(_, _) => false,
            FolFormulaExt::And(a, b) | FolFormulaExt::Or(a, b) | FolFormulaExt::Implies(a, b) => {
                a.is_purely_propositional() && b.is_purely_propositional()
            }
            FolFormulaExt::Not(a) => a.is_purely_propositional(),
            FolFormulaExt::Forall(_, _, _) | FolFormulaExt::Exists(_, _, _) => false,
        }
    }
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

    // ─── Term construction + display ────────────────────────────────────

    #[test]
    fn term_linear_sum() {
        let t = x().add(y()).add(Term::int(3));
        assert_eq!(format!("{}", t), "((x + y) + 3)");
        assert_eq!(t.free_vars(), vec!["x", "y"]);
    }

    #[test]
    fn term_power() {
        let t = x().pow(2).add(y().pow(2));
        assert_eq!(format!("{}", t), "(x^2 + y^2)");
    }

    #[test]
    fn term_rational_literal() {
        let t = Term::rat(1, 3);
        assert_eq!(format!("{}", t), "1/3");
    }

    // ─── is_integral / is_non_linear ────────────────────────────────────

    #[test]
    fn integral_term_detected() {
        let t = x().add(Term::int(5)).mul(y());
        assert!(t.is_integral());
    }

    #[test]
    fn real_literal_breaks_integrality() {
        let t = x().add(Term::real(0.5));
        assert!(!t.is_integral());
    }

    #[test]
    fn linear_in_both_vars_is_linear() {
        let t = x().add(y()); // x + y, linear
        assert!(!t.is_non_linear());
    }

    #[test]
    fn product_of_vars_is_non_linear() {
        let t = x().mul(y()); // x * y
        assert!(t.is_non_linear());
    }

    #[test]
    fn var_times_const_is_linear() {
        let t = x().mul(Term::int(3)); // 3 * x — linear
        assert!(!t.is_non_linear());
    }

    #[test]
    fn pow_n_ge_2_with_var_is_non_linear() {
        let t = x().pow(2); // x^2
        assert!(t.is_non_linear());
    }

    #[test]
    fn const_pow_is_linear() {
        let t = Term::int(5).pow(3); // 5^3, no vars
        assert!(!t.is_non_linear());
    }

    // ─── FolFormulaExt construction ─────────────────────────────────────

    #[test]
    fn trivial_equality_purely_prop_false() {
        let f = FolFormulaExt::eq(x(), x());
        assert!(!f.is_purely_propositional());
    }

    #[test]
    fn base_prop_is_purely_prop() {
        let f = FolFormulaExt::from_prop(Proposition::atom("P"));
        assert!(f.is_purely_propositional());
    }

    #[test]
    fn forall_bound_var_excluded_from_free() {
        // ∀ x : Int, x = x
        let f = FolFormulaExt::forall("x", NumericType::Int, FolFormulaExt::eq(x(), x()));
        let free = f.free_arith_vars();
        assert!(free.is_empty(), "x should be bound; got free={:?}", free);
    }

    #[test]
    fn nat_quantifier_constraint() {
        assert_eq!(
            NumericType::Nat.constraint_for("n"),
            Some("(>= n 0)".to_string())
        );
        assert_eq!(NumericType::Int.constraint_for("n"), None);
        assert_eq!(NumericType::Real.constraint_for("x"), None);
    }

    #[test]
    fn smt_sorts() {
        assert_eq!(NumericType::Int.smt_sort(), "Int");
        assert_eq!(NumericType::Nat.smt_sort(), "Int");
        assert_eq!(NumericType::Real.smt_sort(), "Real");
    }

    #[test]
    fn implication_over_arithmetic_not_purely_prop() {
        // (x > 0) → (x ≥ 0)
        let f = FolFormulaExt::lt(Term::int(0), x()).implies(FolFormulaExt::le(Term::int(0), x()));
        assert!(!f.is_purely_propositional());
    }
}
