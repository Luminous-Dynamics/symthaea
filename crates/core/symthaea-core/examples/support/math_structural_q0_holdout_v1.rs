// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen held-out Q0 structural-neighbor cases for MATH-EXP-001.
//!
//! IMPORTANT: this file intentionally contains no evaluator and does not alter
//! either representation. It freezes exact `FolFormulaExt` challenges before
//! reading measured similarities from the parent Q0 harness.

use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};
use symthaea_core::hdc::logic_engine::Proposition;

pub const HOLDOUT_ID: &str = "math-structural-q0-holdout-v1";

#[derive(Debug, Clone)]
pub struct HoldoutCandidate {
    pub id: &'static str,
    pub formula: FolFormulaExt,
}

#[derive(Debug, Clone)]
pub struct HoldoutCase {
    pub id: &'static str,
    pub positive_id: &'static str,
    pub structural_law: &'static str,
    pub query: FolFormulaExt,
    pub candidates: Vec<HoldoutCandidate>,
}

fn forall_real(var: &str, body: FolFormulaExt) -> FolFormulaExt {
    FolFormulaExt::forall(var, NumericType::Real, body)
}

fn forall_two_real(a: &str, b: &str, body: FolFormulaExt) -> FolFormulaExt {
    FolFormulaExt::forall(a, NumericType::Real, FolFormulaExt::forall(b, NumericType::Real, body))
}

fn subtraction_equation(var: &str, offset: i64, rhs: i64, ty: NumericType) -> FolFormulaExt {
    FolFormulaExt::forall(
        var,
        ty,
        FolFormulaExt::eq(Term::var(var).sub(Term::int(offset)), Term::int(rhs)),
    )
}

fn division_equation(var: &str, denominator: i64, rhs: i64) -> FolFormulaExt {
    forall_real(
        var,
        FolFormulaExt::eq(Term::var(var).div(Term::int(denominator)), Term::int(rhs)),
    )
}

fn square_equation(var: &str, rhs: i64, ty: NumericType) -> FolFormulaExt {
    FolFormulaExt::forall(
        var,
        ty,
        FolFormulaExt::eq(Term::var(var).pow(2), Term::int(rhs)),
    )
}

fn nonzero_formula(var: &str) -> FolFormulaExt {
    forall_real(
        var,
        FolFormulaExt::eq(Term::var(var), Term::int(0)).neg(),
    )
}

fn conjunction_formula(a: &str, b: &str, swapped: bool) -> FolFormulaExt {
    let eq = FolFormulaExt::eq(Term::var(a), Term::int(0));
    let le = FolFormulaExt::le(Term::var(b), Term::int(1));
    let body = if swapped { le.and(eq) } else { eq.and(le) };
    forall_two_real(a, b, body)
}

fn shadowing_formula(outer: &str, inner: &str, body_uses_inner: bool) -> FolFormulaExt {
    let selected = if body_uses_inner { inner } else { outer };
    FolFormulaExt::forall(
        outer,
        NumericType::Real,
        FolFormulaExt::exists(
            inner,
            NumericType::Real,
            FolFormulaExt::eq(Term::var(selected), Term::int(0)),
        ),
    )
}

fn proposition_shape(a: &str, b: &str, c: &str) -> FolFormulaExt {
    FolFormulaExt::from_prop(
        Proposition::atom(a).implies(Proposition::atom(b).and(Proposition::atom(c))),
    )
}

fn rational_offset(var: &str, numerator: i64, denominator: i64) -> FolFormulaExt {
    forall_real(
        var,
        FolFormulaExt::eq(
            Term::var(var).add(Term::rat(numerator, denominator)),
            Term::int(0),
        ),
    )
}

fn ordered_difference(a: &str, b: &str, swap: bool) -> FolFormulaExt {
    let (lhs, rhs) = if swap { (b, a) } else { (a, b) };
    forall_two_real(
        a,
        b,
        FolFormulaExt::lt(Term::var(lhs).sub(Term::var(rhs)), Term::int(0)),
    )
}

pub fn holdout_cases() -> Vec<HoldoutCase> {
    vec![
        HoldoutCase {
            id: "ordered-subtraction-direction",
            positive_id: "subtraction-positive",
            structural_law: "Alpha/literal-class changes preserve ordered subtraction; addition, operand reversal, and quantifier type do not.",
            query: subtraction_equation("q", 4, 9, NumericType::Real),
            candidates: vec![
                HoldoutCandidate {
                    id: "subtraction-positive",
                    formula: subtraction_equation("x", 2, 5, NumericType::Real),
                },
                HoldoutCandidate {
                    id: "addition-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::var("x").add(Term::int(2)), Term::int(5)),
                    ),
                },
                HoldoutCandidate {
                    id: "reversed-subtraction",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::int(2).sub(Term::var("x")), Term::int(5)),
                    ),
                },
                HoldoutCandidate {
                    id: "integer-type-distractor",
                    formula: subtraction_equation("x", 2, 5, NumericType::Int),
                },
            ],
        },
        HoldoutCase {
            id: "ordered-division-direction",
            positive_id: "division-positive",
            structural_law: "Division direction is structural; alpha/literal-class changes may transfer, reciprocal and multiplication forms may not collapse into it.",
            query: division_equation("x", 2, 5),
            candidates: vec![
                HoldoutCandidate {
                    id: "division-positive",
                    formula: division_equation("renamed", 3, 7),
                },
                HoldoutCandidate {
                    id: "reciprocal-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::int(2).div(Term::var("x")), Term::int(5)),
                    ),
                },
                HoldoutCandidate {
                    id: "multiplication-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::var("x").mul(Term::int(2)), Term::int(5)),
                    ),
                },
            ],
        },
        HoldoutCase {
            id: "power-vs-expanded-product",
            positive_id: "power-positive",
            structural_law: "The Q0 representation is syntactic: a square node should transfer to another square while exponent changes and expanded multiplication remain distinct until an explicit algebraic-normalization channel exists.",
            query: square_equation("x", 4, NumericType::Real),
            candidates: vec![
                HoldoutCandidate {
                    id: "power-positive",
                    formula: square_equation("y", 9, NumericType::Real),
                },
                HoldoutCandidate {
                    id: "cube-distractor",
                    formula: forall_real(
                        "y",
                        FolFormulaExt::eq(Term::var("y").pow(3), Term::int(9)),
                    ),
                },
                HoldoutCandidate {
                    id: "expanded-product-distractor",
                    formula: forall_real(
                        "y",
                        FolFormulaExt::eq(
                            Term::var("y").mul(Term::var("y")),
                            Term::int(9),
                        ),
                    ),
                },
                HoldoutCandidate {
                    id: "integer-type-distractor",
                    formula: square_equation("y", 9, NumericType::Int),
                },
            ],
        },
        HoldoutCase {
            id: "negation-scope",
            positive_id: "negation-positive",
            structural_law: "Alpha renaming preserves negation over equality; changing the negated relation, removing negation, or adding a second negation changes structure.",
            query: nonzero_formula("x"),
            candidates: vec![
                HoldoutCandidate {
                    id: "negation-positive",
                    formula: nonzero_formula("renamed"),
                },
                HoldoutCandidate {
                    id: "negated-order-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::le(Term::var("x"), Term::int(0)).neg(),
                    ),
                },
                HoldoutCandidate {
                    id: "double-negation-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::var("x"), Term::int(0)).neg().neg(),
                    ),
                },
                HoldoutCandidate {
                    id: "unnegated-distractor",
                    formula: forall_real(
                        "x",
                        FolFormulaExt::eq(Term::var("x"), Term::int(0)),
                    ),
                },
            ],
        },
        HoldoutCase {
            id: "commutative-formula-reorder-with-binding",
            positive_id: "and-reorder-positive",
            structural_law: "Reordering children of conjunction should preserve structure even when distinct bound variables occur in different children; co-reference changes must remain visible.",
            query: conjunction_formula("x", "y", false),
            candidates: vec![
                HoldoutCandidate {
                    id: "and-reorder-positive",
                    formula: conjunction_formula("alpha", "beta", true),
                },
                HoldoutCandidate {
                    id: "or-distractor",
                    formula: forall_two_real(
                        "alpha",
                        "beta",
                        FolFormulaExt::eq(Term::var("alpha"), Term::int(0)).or(
                            FolFormulaExt::le(Term::var("beta"), Term::int(1)),
                        ),
                    ),
                },
                HoldoutCandidate {
                    id: "coreference-distractor",
                    formula: forall_two_real(
                        "alpha",
                        "beta",
                        FolFormulaExt::eq(Term::var("alpha"), Term::int(0)).and(
                            FolFormulaExt::le(Term::var("alpha"), Term::int(1)),
                        ),
                    ),
                },
            ],
        },
        HoldoutCase {
            id: "quantifier-shadowing-binding-distance",
            positive_id: "shadowing-positive",
            structural_law: "Bound-variable spelling is irrelevant, but whether the body refers to the inner or outer binder is structural and must survive shadowing.",
            query: shadowing_formula("x", "x", true),
            candidates: vec![
                HoldoutCandidate {
                    id: "shadowing-positive",
                    formula: shadowing_formula("a", "b", true),
                },
                HoldoutCandidate {
                    id: "outer-reference-distractor",
                    formula: shadowing_formula("a", "b", false),
                },
                HoldoutCandidate {
                    id: "forall-inner-distractor",
                    formula: FolFormulaExt::forall(
                        "a",
                        NumericType::Real,
                        FolFormulaExt::forall(
                            "b",
                            NumericType::Real,
                            FolFormulaExt::eq(Term::var("b"), Term::int(0)),
                        ),
                    ),
                },
            ],
        },
        HoldoutCase {
            id: "propositional-nesting",
            positive_id: "proposition-positive",
            structural_law: "Canonical proposition-atom renaming preserves implication-over-conjunction shape; implication reversal and disjunction do not.",
            query: proposition_shape("p", "q", "r"),
            candidates: vec![
                HoldoutCandidate {
                    id: "proposition-positive",
                    formula: proposition_shape("a", "b", "c"),
                },
                HoldoutCandidate {
                    id: "reversed-implication-distractor",
                    formula: FolFormulaExt::from_prop(
                        Proposition::atom("b")
                            .and(Proposition::atom("c"))
                            .implies(Proposition::atom("a")),
                    ),
                },
                HoldoutCandidate {
                    id: "or-distractor",
                    formula: FolFormulaExt::from_prop(
                        Proposition::atom("a")
                            .implies(Proposition::atom("b").or(Proposition::atom("c"))),
                    ),
                },
            ],
        },
        HoldoutCase {
            id: "rational-sign-denominator-class",
            positive_id: "rational-positive",
            structural_law: "Positive non-unit rational offsets transfer across exact values; sign and denominator-class changes remain distinguishable.",
            query: rational_offset("x", 1, 2),
            candidates: vec![
                HoldoutCandidate {
                    id: "rational-positive",
                    formula: rational_offset("renamed", 3, 5),
                },
                HoldoutCandidate {
                    id: "negative-rational-distractor",
                    formula: rational_offset("x", -3, 5),
                },
                HoldoutCandidate {
                    id: "unit-denominator-distractor",
                    formula: rational_offset("x", 3, 1),
                },
            ],
        },
        HoldoutCase {
            id: "nested-ordered-difference",
            positive_id: "ordered-difference-positive",
            structural_law: "Alpha renaming preserves ordered subtraction inside a strict inequality; swapping operands or weakening the relation changes structure.",
            query: ordered_difference("x", "y", false),
            candidates: vec![
                HoldoutCandidate {
                    id: "ordered-difference-positive",
                    formula: ordered_difference("alpha", "beta", false),
                },
                HoldoutCandidate {
                    id: "swapped-operands-distractor",
                    formula: ordered_difference("alpha", "beta", true),
                },
                HoldoutCandidate {
                    id: "non-strict-relation-distractor",
                    formula: forall_two_real(
                        "alpha",
                        "beta",
                        FolFormulaExt::le(
                            Term::var("alpha").sub(Term::var("beta")),
                            Term::int(0),
                        ),
                    ),
                },
            ],
        },
    ]
}
