// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Focused qualification harness for MATH-REP-001B.

mod adapter {
    include!("support/math_exact_polynomial_adapter_v1.rs");
}

use adapter::{
    normalize_term_uniform_domain, AdapterErrorKind, Disposition, MAX_AST_DEPTH, NORMALIZER_ID,
};
use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

fn normal(term: &Term, domain: NumericType) -> String {
    normalize_term_uniform_domain(term, domain)
        .expect("fixture should normalize")
        .canonical_serialization
}

fn main() {
    println!("normalizer_id,{NORMALIZER_ID}");
    println!("fixture,equal_normal_form");

    let x = Term::var("x");
    let pairs = [
        ("square_vs_product", x.clone().pow(2), x.clone().mul(x.clone())),
        (
            "double_vs_sum",
            Term::int(2).mul(x.clone()),
            x.clone().add(x.clone()),
        ),
        (
            "binomial_square",
            x.clone().add(Term::int(1)).pow(2),
            x.clone()
                .pow(2)
                .add(Term::int(2).mul(x.clone()))
                .add(Term::int(1)),
        ),
    ];

    for (name, lhs, rhs) in pairs {
        println!(
            "{name},{}",
            normal(&lhs, NumericType::Real) == normal(&rhs, NumericType::Real)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adapter::MAX_VARIABLES;

    #[test]
    fn square_and_product_share_exact_normal_form() {
        let x = Term::var("x");
        assert_eq!(
            normal(&x.clone().pow(2), NumericType::Real),
            normal(&x.clone().mul(x), NumericType::Real)
        );
    }

    #[test]
    fn doubled_variable_and_repeated_sum_share_exact_normal_form() {
        let x = Term::var("x");
        assert_eq!(
            normal(&Term::int(2).mul(x.clone()), NumericType::Real),
            normal(&x.clone().add(x), NumericType::Real)
        );
    }

    #[test]
    fn binomial_expansion_is_exact() {
        let x = Term::var("x");
        let compact = x.clone().add(Term::int(1)).pow(2);
        let expanded = x
            .clone()
            .pow(2)
            .add(Term::int(2).mul(x.clone()))
            .add(Term::int(1));
        assert_eq!(
            normal(&compact, NumericType::Real),
            normal(&expanded, NumericType::Real)
        );
    }

    #[test]
    fn commutative_reordering_is_canonical() {
        let x = Term::var("x");
        let y = Term::var("y");
        assert_eq!(
            normal(&x.clone().add(y.clone()), NumericType::Real),
            normal(&y.add(x), NumericType::Real)
        );
    }

    #[test]
    fn subtraction_direction_is_preserved() {
        let x = Term::var("x");
        let y = Term::var("y");
        assert_ne!(
            normal(&x.clone().sub(y.clone()), NumericType::Real),
            normal(&y.sub(x), NumericType::Real)
        );
    }

    #[test]
    fn free_variable_names_remain_part_of_identity() {
        assert_ne!(
            normal(&Term::var("x"), NumericType::Real),
            normal(&Term::var("y"), NumericType::Real)
        );
    }

    #[test]
    fn numeric_domains_do_not_collapse() {
        let term = Term::var("x").add(Term::int(1));
        let int = normal(&term, NumericType::Int);
        let nat = normal(&term, NumericType::Nat);
        let real = normal(&term, NumericType::Real);
        assert_ne!(int, nat);
        assert_ne!(int, real);
        assert_ne!(nat, real);
    }

    #[test]
    fn rational_coefficients_reduce_exactly() {
        let x = Term::var("x");
        assert_eq!(
            normal(&Term::rat(2, 4).mul(x.clone()), NumericType::Real),
            normal(&Term::rat(1, 2).mul(x), NumericType::Real)
        );
    }

    #[test]
    fn rational_literals_do_not_leak_into_int_domain() {
        let err = normalize_term_uniform_domain(&Term::rat(1, 2), NumericType::Int)
            .expect_err("non-integral rational must be outside the Int v1 fragment");
        assert_eq!(err.kind, AdapterErrorKind::RationalLiteralOutsideReal);
        assert_eq!(err.receipt_rejection_reason(), "DomainAmbiguity");
        assert_eq!(err.disposition(), Disposition::Unsupported);
    }

    #[test]
    fn inexact_real_literals_are_not_silently_rationalized() {
        let err = normalize_term_uniform_domain(&Term::real(0.5), NumericType::Real)
            .expect_err("f64 literal must not enter exact-polynomial v1");
        assert_eq!(err.kind, AdapterErrorKind::InexactRealLiteral);
    }

    #[test]
    fn division_is_not_normalized_in_v1() {
        let term = Term::var("x").div(Term::int(2));
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("division is outside exact-polynomial v1");
        assert_eq!(err.kind, AdapterErrorKind::NonPolynomialDivision);
        assert_eq!(err.receipt_rejection_reason(), "NonPolynomialDivision");
    }

    #[test]
    fn malformed_zero_denominator_is_rejected_without_panicking() {
        let term = Term::RatLit(1, 0);
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("malformed rational must be rejected");
        assert_eq!(err.kind, AdapterErrorKind::ZeroDenominator);
        assert_eq!(err.disposition(), Disposition::Rejected);
    }

    #[test]
    fn coefficient_overflow_is_non_conclusive() {
        let term = Term::int(i64::MAX).add(Term::int(i64::MAX));
        let err = normalize_term_uniform_domain(&term, NumericType::Int)
            .expect_err("out-of-envelope coefficient must not wrap");
        assert_eq!(err.kind, AdapterErrorKind::CoefficientOverflow);
        assert_eq!(err.receipt_rejection_reason(), "ResourceLimit");
    }

    #[test]
    fn exponent_overflow_is_non_conclusive() {
        let term = Term::var("x").pow(u32::MAX).mul(Term::var("x"));
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("exponent overflow must not wrap");
        assert_eq!(err.kind, AdapterErrorKind::ExponentOverflow);
        assert_eq!(err.receipt_rejection_reason(), "ResourceLimit");
    }

    #[test]
    fn variable_budget_is_explicit() {
        let mut term = Term::int(0);
        for index in 0..=MAX_VARIABLES {
            term = term.add(Term::var(&format!("v{index}")));
        }
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("variable budget must be enforced");
        assert_eq!(err.kind, AdapterErrorKind::TooManyVariables);
    }

    #[test]
    fn monomial_expansion_budget_is_explicit() {
        let mut term = Term::int(1);
        for index in 0..13 {
            let lhs = Term::var(&format!("a{index}"));
            let rhs = Term::var(&format!("b{index}"));
            term = term.mul(lhs.add(rhs));
        }
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("2^13 monomials must exceed the frozen v1 budget");
        assert_eq!(err.kind, AdapterErrorKind::TooManyMonomials);
        assert_eq!(err.receipt_rejection_reason(), "ResourceLimit");
    }

    #[test]
    fn depth_budget_is_explicit() {
        let mut term = Term::var("x");
        for _ in 0..=MAX_AST_DEPTH {
            term = term.neg();
        }
        let err = normalize_term_uniform_domain(&term, NumericType::Real)
            .expect_err("depth budget must be enforced");
        assert_eq!(err.kind, AdapterErrorKind::AstDepthLimit);
    }

    #[test]
    fn core_poly_storage_and_transform_trace_are_populated() {
        let x = Term::var("x");
        let normalized = normalize_term_uniform_domain(
            &x.clone().add(Term::int(1)).pow(2),
            NumericType::Real,
        )
        .expect("fixture should normalize");
        assert_eq!(normalized.domain, NumericType::Real);
        assert_eq!(normalized.variables, vec!["x".to_string()]);
        assert!(!normalized.polynomial.terms.is_empty());
        assert!(
            normalized
                .transformations
                .contains(&"ExpandNonnegativeIntegerPower")
        );
        assert!(normalized.transformations.contains(&"CollectLikeTerms"));
    }
}
