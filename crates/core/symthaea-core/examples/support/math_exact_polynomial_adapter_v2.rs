// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-REP-001B v2 canonical-identity repair.
//!
//! Arithmetic and fragment acceptance are inherited from v1. v2 changes only
//! canonical variable projection after algebraic reduction: source variables
//! absent from every surviving monomial are removed from normal-form identity.

use serde::Serialize;
use symthaea_core::hdc::fol_formula_ext::{NumericType, Term as AstTerm};
use symthaea_core::hdc::polynomial_algebra::{Monomial, Poly, Term as PolynomialTerm};

pub use super::adapter_v1::{
    AdapterError, AdapterErrorKind, Disposition, MAX_AST_DEPTH, MAX_MONOMIALS, MAX_VARIABLES,
};
use super::adapter_v1::{normalize_term_uniform_domain as normalize_v1, NormalizedPolynomial as V1};

pub const NORMALIZER_ID: &str = "symthaea-exact-polynomial-term-v2";

#[derive(Debug)]
pub struct NormalizedPolynomial {
    pub domain: NumericType,
    pub variables: Vec<String>,
    pub polynomial: Poly,
    pub canonical_serialization: String,
    pub transformations: Vec<&'static str>,
}

#[derive(Serialize)]
struct CanonicalPolynomialView<'a> {
    format: &'static str,
    normalizer_id: &'static str,
    domain: &'static str,
    variable_identity: &'static str,
    variables: &'a [String],
    monomial_order: &'static str,
    terms: Vec<CanonicalTermView<'a>>,
}

#[derive(Serialize)]
struct CanonicalTermView<'a> {
    numerator: i64,
    denominator: i64,
    exponents: &'a [u32],
}

pub fn normalize_term_uniform_domain(
    term: &AstTerm,
    domain: NumericType,
) -> Result<NormalizedPolynomial, AdapterError> {
    let v1 = normalize_v1(term, domain)?;
    project_active_variables(v1)
}

fn project_active_variables(v1: V1) -> Result<NormalizedPolynomial, AdapterError> {
    if v1.polynomial.nvars != v1.variables.len() {
        return Err(AdapterError {
            kind: AdapterErrorKind::InternalDimensionMismatch,
        });
    }

    let active_indices: Vec<usize> = (0..v1.variables.len())
        .filter(|index| {
            v1.polynomial.terms.iter().any(|term| {
                term.mono
                    .exponents
                    .get(*index)
                    .copied()
                    .unwrap_or_default()
                    > 0
            })
        })
        .collect();

    let variables: Vec<String> = active_indices
        .iter()
        .map(|index| v1.variables[*index].clone())
        .collect();

    let projected_terms: Result<Vec<PolynomialTerm>, AdapterError> = v1
        .polynomial
        .terms
        .iter()
        .map(|term| {
            let mut exponents = Vec::with_capacity(active_indices.len());
            for index in &active_indices {
                let exponent = term.mono.exponents.get(*index).copied().ok_or(AdapterError {
                    kind: AdapterErrorKind::InternalDimensionMismatch,
                })?;
                exponents.push(exponent);
            }
            Ok(PolynomialTerm {
                coeff: term.coeff,
                mono: Monomial { exponents },
            })
        })
        .collect();

    // Active projection cannot merge distinct monomials because every removed
    // dimension is zero in every surviving term. `Poly::from_terms` therefore
    // remains a storage/order boundary rather than an arithmetic step here.
    let polynomial = Poly::from_terms(projected_terms?, variables.len());
    let canonical_serialization = serialize_canonical(v1.domain, &variables, &polynomial);

    Ok(NormalizedPolynomial {
        domain: v1.domain,
        variables,
        polynomial,
        canonical_serialization,
        transformations: v1.transformations,
    })
}

fn serialize_canonical(domain: NumericType, variables: &[String], polynomial: &Poly) -> String {
    let terms = polynomial
        .terms
        .iter()
        .map(|term| CanonicalTermView {
            numerator: term.coeff.num,
            denominator: term.coeff.den,
            exponents: &term.mono.exponents,
        })
        .collect();

    let view = CanonicalPolynomialView {
        format: "symthaea-exact-polynomial-normal-form-v2",
        normalizer_id: NORMALIZER_ID,
        domain: domain_name(domain),
        variable_identity: "PreserveActiveFreeVariableNames",
        variables,
        monomial_order: "GrlexDescending",
        terms,
    };

    serde_json::to_string(&view).expect("canonical polynomial v2 view must serialize")
}

fn domain_name(domain: NumericType) -> &'static str {
    match domain {
        NumericType::Int => "Int",
        NumericType::Nat => "Nat",
        NumericType::Real => "Real",
    }
}
