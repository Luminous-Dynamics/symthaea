// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preregistered, non-blinded holdout fixtures for MATH-REP-001C.
//!
//! Freeze before observing MATH-REP-001B outputs. Do not use these cases to
//! tune v1 after scores/results are observed.

use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

use super::fixture_types::{ExpectedRelation, PairCase, RefusalCase};

pub const HOLDOUT_SET_ID: &str = "math-equivalence-retrieval-q0-holdout-v1";

pub fn pair_cases() -> Vec<PairCase> {
    let x = Term::var("x");
    let y = Term::var("y");
    let z = Term::var("z");

    vec![
        PairCase { id: "HOLD_EQ_linear_distribution", lhs_domain: NumericType::Real, lhs: Term::int(2).mul(x.clone().add(y.clone())), rhs_domain: NumericType::Real, rhs: Term::int(2).mul(x.clone()).add(Term::int(2).mul(y.clone())), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_square_of_negative", lhs_domain: NumericType::Real, lhs: x.clone().neg().pow(2), rhs_domain: NumericType::Real, rhs: x.clone().pow(2), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_three_variable_reorder", lhs_domain: NumericType::Int, lhs: x.clone().add(y.clone()).add(z.clone()), rhs_domain: NumericType::Int, rhs: z.clone().add(x.clone()).add(y.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_rational_common_denominator", lhs_domain: NumericType::Real, lhs: Term::rat(1, 6).mul(x.clone()).add(Term::rat(1, 3).mul(x.clone())), rhs_domain: NumericType::Real, rhs: Term::rat(1, 2).mul(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_power_composition", lhs_domain: NumericType::Real, lhs: x.clone().pow(3).pow(2), rhs_domain: NumericType::Real, rhs: x.clone().pow(6), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_product_commutation", lhs_domain: NumericType::Real, lhs: x.clone().mul(y.clone()), rhs_domain: NumericType::Real, rhs: y.clone().mul(x.clone()), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_zero_product", lhs_domain: NumericType::Int, lhs: Term::int(0).mul(x.clone().add(y.clone())), rhs_domain: NumericType::Int, rhs: Term::int(0), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_zero_power", lhs_domain: NumericType::Real, lhs: x.clone().sub(y.clone()).pow(0), rhs_domain: NumericType::Real, rhs: Term::int(1), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_EQ_collect_three_terms", lhs_domain: NumericType::Real, lhs: Term::int(3).mul(x.clone()).add(Term::int(2).mul(y.clone())).sub(x.clone()), rhs_domain: NumericType::Real, rhs: Term::int(2).mul(x.clone()).add(Term::int(2).mul(y.clone())), expected: ExpectedRelation::SameNormalForm },
        PairCase { id: "HOLD_NE_product_vs_sum", lhs_domain: NumericType::Real, lhs: x.clone().mul(y.clone()), rhs_domain: NumericType::Real, rhs: x.clone().add(y.clone()), expected: ExpectedRelation::DifferentNormalForm },
        PairCase { id: "HOLD_NE_degree_change", lhs_domain: NumericType::Real, lhs: x.clone().pow(2), rhs_domain: NumericType::Real, rhs: x.clone().pow(3), expected: ExpectedRelation::DifferentNormalForm },
        PairCase { id: "HOLD_NE_domain_nat_real", lhs_domain: NumericType::Nat, lhs: x.clone().add(Term::int(1)), rhs_domain: NumericType::Real, rhs: x.clone().add(Term::int(1)), expected: ExpectedRelation::DifferentNormalForm },
        PairCase { id: "HOLD_NE_free_parameter_identity", lhs_domain: NumericType::Real, lhs: x.clone().add(z.clone()), rhs_domain: NumericType::Real, rhs: y.add(z), expected: ExpectedRelation::DifferentNormalForm },
    ]
}

pub fn refusal_cases() -> Vec<RefusalCase> {
    vec![
        RefusalCase { id: "HOLD_REFUSE_inexact_quarter", domain: NumericType::Real, term: Term::real(0.25), expected_disposition: "Unsupported", expected_receipt_reason: "OtherUnsupported" },
        RefusalCase { id: "HOLD_REFUSE_compound_constant_division", domain: NumericType::Real, term: Term::var("x").add(Term::int(1)).div(Term::int(2)), expected_disposition: "Unsupported", expected_receipt_reason: "NonPolynomialDivision" },
        RefusalCase { id: "HOLD_REFUSE_compound_variable_denominator", domain: NumericType::Real, term: Term::int(1).div(Term::var("x").add(Term::int(1))), expected_disposition: "Unsupported", expected_receipt_reason: "NonPolynomialDivision" },
        RefusalCase { id: "HOLD_REFUSE_fractional_nat_literal", domain: NumericType::Nat, term: Term::rat(3, 2), expected_disposition: "Unsupported", expected_receipt_reason: "DomainAmbiguity" },
        RefusalCase { id: "HOLD_REFUSE_negated_i64_min", domain: NumericType::Int, term: Term::int(i64::MIN).neg(), expected_disposition: "Unsupported", expected_receipt_reason: "ResourceLimit" },
        RefusalCase { id: "HOLD_REFUSE_exponent_overflow", domain: NumericType::Real, term: Term::var("x").pow(u32::MAX).mul(Term::var("x")), expected_disposition: "Unsupported", expected_receipt_reason: "ResourceLimit" },
    ]
}
