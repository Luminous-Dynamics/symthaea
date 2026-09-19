// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Focused MATH-REP-001B v2 regression harness.

mod adapter_v1 {
    include!("support/math_exact_polynomial_adapter_v1.rs");
}
mod adapter_v2 {
    include!("support/math_exact_polynomial_adapter_v2.rs");
}

use adapter_v2::{normalize_term_uniform_domain, AdapterErrorKind, NORMALIZER_ID};
use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

fn normal_v1(term: &Term, domain: NumericType) -> String {
    adapter_v1::normalize_term_uniform_domain(term, domain)
        .expect("v1 fixture should normalize")
        .canonical_serialization
}

fn normal_v2(term: &Term, domain: NumericType) -> String {
    normalize_term_uniform_domain(term, domain)
        .expect("v2 fixture should normalize")
        .canonical_serialization
}

fn main() {
    println!("normalizer_id,{NORMALIZER_ID}");
    println!("repair,active_variable_projection");

    let x = Term::var("x");
    let y = Term::var("y");
    let lhs = x.clone().add(y.clone()).sub(y);
    println!(
        "cancellation_matches_x,{}",
        normal_v2(&lhs, NumericType::Real) == normal_v2(&x, NumericType::Real)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_static_defect_is_reproduced_by_source_variable_identity() {
        let x = Term::var("x");
        let y = Term::var("y");
        let lhs = x.clone().add(y.clone()).sub(y);
        assert_ne!(
            normal_v1(&lhs, NumericType::Real),
            normal_v1(&x, NumericType::Real)
        );
    }

    #[test]
    fn v2_projects_cancelled_variables_from_identity() {
        let x = Term::var("x");
        let y = Term::var("y");
        let lhs = x.clone().add(y.clone()).sub(y);
        assert_eq!(
            normal_v2(&lhs, NumericType::Real),
            normal_v2(&x, NumericType::Real)
        );
    }

    #[test]
    fn v2_zero_product_matches_literal_zero() {
        let lhs = Term::int(0).mul(Term::var("x").add(Term::var("y")));
        assert_eq!(
            normal_v2(&lhs, NumericType::Int),
            normal_v2(&Term::int(0), NumericType::Int)
        );
    }

    #[test]
    fn v2_complete_cancellation_matches_zero() {
        let x = Term::var("x");
        assert_eq!(
            normal_v2(&x.clone().sub(x), NumericType::Real),
            normal_v2(&Term::int(0), NumericType::Real)
        );
    }

    #[test]
    fn v2_preserves_surviving_free_variable_names() {
        assert_ne!(
            normal_v2(&Term::var("x"), NumericType::Real),
            normal_v2(&Term::var("y"), NumericType::Real)
        );
    }

    #[test]
    fn v2_preserves_subtraction_direction() {
        let x = Term::var("x");
        let y = Term::var("y");
        assert_ne!(
            normal_v2(&x.clone().sub(y.clone()), NumericType::Real),
            normal_v2(&y.sub(x), NumericType::Real)
        );
    }

    #[test]
    fn v2_keeps_domain_separation() {
        let term = Term::var("x").add(Term::int(1));
        assert_ne!(
            normal_v2(&term, NumericType::Int),
            normal_v2(&term, NumericType::Real)
        );
    }

    #[test]
    fn v2_keeps_v1_exact_arithmetic_behavior() {
        let x = Term::var("x");
        assert_eq!(
            normal_v2(&x.clone().pow(2), NumericType::Real),
            normal_v2(&x.clone().mul(x), NumericType::Real)
        );
    }

    #[test]
    fn v2_keeps_v1_refusal_behavior() {
        let err = normalize_term_uniform_domain(
            &Term::var("x").div(Term::int(2)),
            NumericType::Real,
        )
        .expect_err("division remains outside the v2 fragment");
        assert_eq!(err.kind, AdapterErrorKind::NonPolynomialDivision);
    }

    #[test]
    fn v2_variable_list_contains_only_active_names() {
        let normalized = normalize_term_uniform_domain(
            &Term::var("x").add(Term::var("y")).sub(Term::var("y")),
            NumericType::Real,
        )
        .expect("fixture should normalize");
        assert_eq!(normalized.variables, vec!["x".to_string()]);
        assert_eq!(normalized.polynomial.nvars, 1);
    }
}
