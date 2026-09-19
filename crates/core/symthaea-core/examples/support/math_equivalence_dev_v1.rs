// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preregistered development fixtures for MATH-REP-001C.

use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

use super::fixture_types::{ExpectedRelation, PairCase, RefusalCase};

pub const DEVELOPMENT_SET_ID: &str = "math-equivalence-retrieval-q0-dev-v1";

pub fn pair_cases() -> Vec<PairCase> {
    let x = Term::var("x");
    let y = Term::var("y");
    let z = Term::var("z");

    vec![
        PairCase { id: "DEV_EQ_square_product", lhs_domain: NumericType::Real, lhs: x.clone().pow(2), rhs_domain: NumericType::Real, rhs: x.clone().mul(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_double_sum", lhs_domain: NumericType::Real, lhs: x.clone().add(x.clone()), rhs_domain: NumericType::Real, rhs: Term::int(2).mul(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_binomial_square", lhs_domain: NumericType::Real, lhs: x.clone().add(Term::int(1)).pow(2), rhs_domain: NumericType::Real, rhs: x.clone().pow(2).add(Term::int(2).mul(x.clone())).add(Term::int(1)), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_difference_of_squares", lhs_domain: NumericType::Real, lhs: x.clone().sub(y.clone()).mul(x.clone().add(y.clone())), rhs_domain: NumericType::Real, rhs: x.clone().pow(2).sub(y.clone().pow(2)), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_factor_distribution", lhs_domain: NumericType::Int, lhs: x.clone().mul(y.clone().add(z.clone())), rhs_domain: NumericType::Int, rhs: x.clone().mul(y.clone()).add(x.clone().mul(z.clone())), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_negation_distribution", lhs_domain: NumericType::Real, lhs: x.clone().sub(y.clone()).neg(), rhs_domain: NumericType::Real, rhs: y.clone().sub(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_rational_collection", lhs_domain: NumericType::Real, lhs: Term::rat(1, 2).mul(x.clone()).add(Term::rat(1, 3).mul(x.clone())), rhs_domain: NumericType::Real, rhs: Term::rat(5, 6).mul(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_nested_power", lhs_domain: NumericType::Real, lhs: x.clone().pow(2).pow(3), rhs_domain: NumericType::Real, rhs: x.clone().pow(6), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_EQ_additive_cancellation", lhs_domain: NumericType::Int, lhs: x.clone().add(y.clone()).sub(y.clone()), rhs_domain: NumericType::Int, rhs: x.clone(), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "DEV_NE_subtraction_direction", lhs_domain: NumericType::Real, lhs: x.clone().sub(y.clone()), rhs_domain: NumericType::Real, rhs: y.clone().sub(x.clone()), expected: ExpectedRelation::DifferentNormalForm },
        PairCase { id: "DEV_NE_free_variable_identity", lhs_domain: NumericType::Real, lhs: x.clone(), rhs_domain: NumericType::Real, rhs: y.clone(), expected: ExpectedRelation::DifferentNormalForm },
        PairCase { id: "DEV_NE_domain_identity", lhs_domain: NumericType::Int, lhs: x.clone().add(Term::int(1)), rhs_domain: NumericType::Real, rhs: x.add(Term::int(1)), expected: ExpectedRelation::DifferentNormalForm },
    ]
}

pub fn refusal_cases() -> Vec<RefusalCase> {
    vec![
        RefusalCase { id: "DEV_REFUSE_inexact_real_literal", domain: NumericType::Real, term: Term::real(0.5), expected_disposition: "Unsupported", expected_receipt_reason: "OtherUnsupported" },
        RefusalCase { id: "DEV_REFUSE_constant_division", domain: NumericType::Real, term: Term::var("x").div(Term::int(2)), expected_disposition: "Unsupported", expected_receipt_reason: "NonPolynomialDivision" },
        RefusalCase { id: "DEV_REFUSE_variable_division", domain: NumericType::Real, term: Term::var("x").div(Term::var("y")), expected_disposition: "Unsupported", expected_receipt_reason: "NonPolynomialDivision" },
        RefusalCase { id: "DEV_REFUSE_zero_denominator", domain: NumericType::Real, term: Term::RatLit(1, 0), expected_disposition: "Rejected", expected_receipt_reason: "OtherUnsupported" },
        RefusalCase { id: "DEV_REFUSE_fractional_int_literal", domain: NumericType::Int, term: Term::rat(1, 2), expected_disposition: "Unsupported", expected_receipt_reason: "DomainAmbiguity" },
        RefusalCase { id: "DEV_REFUSE_coefficient_overflow", domain: NumericType::Int, term: Term::int(i64::MAX).add(Term::int(i64::MAX)), expected_disposition: "Unsupported", expected_receipt_reason: "ResourceLimit" },
    ]
}
