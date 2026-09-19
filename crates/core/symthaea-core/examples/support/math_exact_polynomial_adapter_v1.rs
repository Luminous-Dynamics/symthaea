// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative exact-polynomial adapter for MATH-REP-001B.
//!
//! This file is example-local on purpose. It qualifies the conversion boundary
//! before changing the public `symthaea-core` module surface or runtime retrieval.

use std::collections::BTreeMap;

use serde::Serialize;
use symthaea_core::hdc::fol_formula_ext::{ArithOp, NumericType, Term as AstTerm};
use symthaea_core::hdc::polynomial_algebra::{
    Monomial, Poly, Rat, Term as PolynomialTerm,
};

pub const NORMALIZER_ID: &str = "symthaea-exact-polynomial-term-v1";
pub const MAX_VARIABLES: usize = 64;
pub const MAX_MONOMIALS: usize = 4096;
pub const MAX_AST_DEPTH: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    Unsupported,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterErrorKind {
    InexactRealLiteral,
    NonPolynomialDivision,
    ZeroDenominator,
    RationalLiteralOutsideReal,
    TooManyVariables,
    TooManyMonomials,
    AstDepthLimit,
    CoefficientOverflow,
    ExponentOverflow,
    InternalDimensionMismatch,
    MissingVariableIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterError {
    pub kind: AdapterErrorKind,
}

impl AdapterError {
    pub fn disposition(self) -> Disposition {
        match self.kind {
            AdapterErrorKind::ZeroDenominator
            | AdapterErrorKind::InternalDimensionMismatch
            | AdapterErrorKind::MissingVariableIndex => Disposition::Rejected,
            _ => Disposition::Unsupported,
        }
    }

    /// Maps this adapter-local diagnosis onto the closed reason vocabulary in
    /// `math-normalization-receipt-v1.schema.json`.
    pub fn receipt_rejection_reason(self) -> &'static str {
        match self.kind {
            AdapterErrorKind::NonPolynomialDivision => "NonPolynomialDivision",
            AdapterErrorKind::RationalLiteralOutsideReal => "DomainAmbiguity",
            AdapterErrorKind::TooManyVariables
            | AdapterErrorKind::TooManyMonomials
            | AdapterErrorKind::AstDepthLimit
            | AdapterErrorKind::CoefficientOverflow
            | AdapterErrorKind::ExponentOverflow => "ResourceLimit",
            AdapterErrorKind::InexactRealLiteral
            | AdapterErrorKind::ZeroDenominator
            | AdapterErrorKind::InternalDimensionMismatch
            | AdapterErrorKind::MissingVariableIndex => "OtherUnsupported",
        }
    }
}

#[derive(Debug)]
pub struct NormalizedPolynomial {
    pub domain: NumericType,
    pub variables: Vec<String>,
    pub polynomial: Poly,
    pub canonical_serialization: String,
    pub transformations: Vec<&'static str>,
}

#[derive(Debug, Clone, Default)]
struct TransformTrace {
    saw_additive: bool,
    saw_multiplicative: bool,
    saw_power: bool,
    distributed: bool,
}

impl TransformTrace {
    fn receipt_transformations(&self) -> Vec<&'static str> {
        let mut out = vec![
            "ReduceRationalCoefficients",
            "SortCommutativeTerms",
            "CollectLikeTerms",
            "NormalizeSigns",
        ];
        if self.saw_additive {
            out.push("FlattenAssociativeAdd");
        }
        if self.saw_multiplicative {
            out.push("FlattenAssociativeMul");
        }
        if self.saw_power {
            out.push("ExpandNonnegativeIntegerPower");
        }
        if self.distributed {
            out.push("DistributeMultiplication");
        }
        out
    }
}

#[derive(Debug, Clone)]
struct CheckedPolynomial {
    nvars: usize,
    terms: BTreeMap<Vec<u32>, Rat>,
}

impl CheckedPolynomial {
    fn zero(nvars: usize) -> Self {
        Self {
            nvars,
            terms: BTreeMap::new(),
        }
    }

    fn one(nvars: usize) -> Self {
        Self::constant(nvars, Rat { num: 1, den: 1 })
    }

    fn constant(nvars: usize, coeff: Rat) -> Self {
        let mut out = Self::zero(nvars);
        if !coeff.is_zero() {
            out.terms.insert(vec![0; nvars], coeff);
        }
        out
    }

    fn variable(nvars: usize, index: usize) -> Result<Self, AdapterError> {
        if index >= nvars {
            return Err(error(AdapterErrorKind::MissingVariableIndex));
        }
        let mut exponents = vec![0; nvars];
        exponents[index] = 1;
        let mut out = Self::zero(nvars);
        out.terms.insert(exponents, Rat { num: 1, den: 1 });
        Ok(out)
    }

    fn ensure_same_dimension(&self, other: &Self) -> Result<(), AdapterError> {
        if self.nvars != other.nvars {
            return Err(error(AdapterErrorKind::InternalDimensionMismatch));
        }
        Ok(())
    }

    fn add_term(&mut self, exponents: Vec<u32>, coeff: Rat) -> Result<(), AdapterError> {
        if exponents.len() != self.nvars {
            return Err(error(AdapterErrorKind::InternalDimensionMismatch));
        }
        if coeff.is_zero() {
            return Ok(());
        }

        if let Some(existing) = self.terms.get(&exponents).copied() {
            let combined = checked_rat_add(existing, coeff)?;
            if combined.is_zero() {
                self.terms.remove(&exponents);
            } else {
                self.terms.insert(exponents, combined);
            }
        } else {
            self.terms.insert(exponents, coeff);
            if self.terms.len() > MAX_MONOMIALS {
                return Err(error(AdapterErrorKind::TooManyMonomials));
            }
        }
        Ok(())
    }

    fn add(&self, other: &Self) -> Result<Self, AdapterError> {
        self.ensure_same_dimension(other)?;
        let mut out = self.clone();
        for (exponents, coeff) in &other.terms {
            out.add_term(exponents.clone(), *coeff)?;
        }
        Ok(out)
    }

    fn sub(&self, other: &Self) -> Result<Self, AdapterError> {
        self.ensure_same_dimension(other)?;
        let mut out = self.clone();
        for (exponents, coeff) in &other.terms {
            out.add_term(exponents.clone(), checked_rat_neg(*coeff)?)?;
        }
        Ok(out)
    }

    fn neg(&self) -> Result<Self, AdapterError> {
        let mut out = Self::zero(self.nvars);
        for (exponents, coeff) in &self.terms {
            out.add_term(exponents.clone(), checked_rat_neg(*coeff)?)?;
        }
        Ok(out)
    }

    fn mul(&self, other: &Self) -> Result<Self, AdapterError> {
        self.ensure_same_dimension(other)?;
        let mut out = Self::zero(self.nvars);
        for (lhs_exp, lhs_coeff) in &self.terms {
            for (rhs_exp, rhs_coeff) in &other.terms {
                let exponents = checked_exponent_add(lhs_exp, rhs_exp)?;
                let coeff = checked_rat_mul(*lhs_coeff, *rhs_coeff)?;
                out.add_term(exponents, coeff)?;
            }
        }
        Ok(out)
    }

    fn pow(&self, mut exponent: u32) -> Result<Self, AdapterError> {
        let mut result = Self::one(self.nvars);
        if exponent == 0 {
            return Ok(result);
        }

        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    fn into_core_poly(self) -> Poly {
        // `terms` is already unique by monomial, so `Poly::from_terms` only
        // supplies the repository's canonical monomial ordering/storage here;
        // it is not asked to perform unchecked coefficient arithmetic.
        let terms = self
            .terms
            .into_iter()
            .map(|(exponents, coeff)| PolynomialTerm {
                coeff,
                mono: Monomial { exponents },
            })
            .collect();
        Poly::from_terms(terms, self.nvars)
    }
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
    let variables = term.free_vars();
    if variables.len() > MAX_VARIABLES {
        return Err(error(AdapterErrorKind::TooManyVariables));
    }

    // Standalone terms have no binder context. Free variable names therefore
    // remain part of identity. Alpha-normalizing them here would make `x-y`
    // and an independently renamed/permuted `y-x` dangerously easy to conflate.
    let indices: BTreeMap<&str, usize> = variables
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();

    let mut trace = TransformTrace::default();
    let checked = build_checked(term, domain, &indices, 0, &mut trace)?;
    let polynomial = checked.into_core_poly();
    let canonical_serialization = serialize_canonical(domain, &variables, &polynomial);

    Ok(NormalizedPolynomial {
        domain,
        variables,
        polynomial,
        canonical_serialization,
        transformations: trace.receipt_transformations(),
    })
}

fn build_checked(
    term: &AstTerm,
    domain: NumericType,
    indices: &BTreeMap<&str, usize>,
    depth: usize,
    trace: &mut TransformTrace,
) -> Result<CheckedPolynomial, AdapterError> {
    if depth > MAX_AST_DEPTH {
        return Err(error(AdapterErrorKind::AstDepthLimit));
    }
    let nvars = indices.len();

    match term {
        AstTerm::Var(name) => {
            let index = indices
                .get(name.as_str())
                .copied()
                .ok_or_else(|| error(AdapterErrorKind::MissingVariableIndex))?;
            CheckedPolynomial::variable(nvars, index)
        }
        AstTerm::IntLit(value) => Ok(CheckedPolynomial::constant(
            nvars,
            reduce_i128(*value as i128, 1)?,
        )),
        AstTerm::RealLit(_) => Err(error(AdapterErrorKind::InexactRealLiteral)),
        AstTerm::RatLit(num, den) => {
            if *den == 0 {
                return Err(error(AdapterErrorKind::ZeroDenominator));
            }
            let rational = reduce_i128(*num as i128, *den as i128)?;
            if domain != NumericType::Real && rational.den != 1 {
                return Err(error(AdapterErrorKind::RationalLiteralOutsideReal));
            }
            Ok(CheckedPolynomial::constant(nvars, rational))
        }
        AstTerm::BinOp(ArithOp::Add, lhs, rhs) => {
            trace.saw_additive = true;
            let lhs = build_checked(lhs, domain, indices, depth + 1, trace)?;
            let rhs = build_checked(rhs, domain, indices, depth + 1, trace)?;
            lhs.add(&rhs)
        }
        AstTerm::BinOp(ArithOp::Sub, lhs, rhs) => {
            trace.saw_additive = true;
            let lhs = build_checked(lhs, domain, indices, depth + 1, trace)?;
            let rhs = build_checked(rhs, domain, indices, depth + 1, trace)?;
            lhs.sub(&rhs)
        }
        AstTerm::BinOp(ArithOp::Mul, lhs, rhs) => {
            trace.saw_multiplicative = true;
            let lhs = build_checked(lhs, domain, indices, depth + 1, trace)?;
            let rhs = build_checked(rhs, domain, indices, depth + 1, trace)?;
            if lhs.terms.len() > 1 || rhs.terms.len() > 1 {
                trace.distributed = true;
            }
            lhs.mul(&rhs)
        }
        AstTerm::BinOp(ArithOp::Div, _, _) => {
            Err(error(AdapterErrorKind::NonPolynomialDivision))
        }
        AstTerm::Pow(base, exponent) => {
            trace.saw_power = true;
            let base = build_checked(base, domain, indices, depth + 1, trace)?;
            base.pow(*exponent)
        }
        AstTerm::Neg(body) => {
            trace.saw_additive = true;
            let body = build_checked(body, domain, indices, depth + 1, trace)?;
            body.neg()
        }
    }
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
        format: "symthaea-exact-polynomial-normal-form-v1",
        normalizer_id: NORMALIZER_ID,
        domain: domain_name(domain),
        variable_identity: "PreserveFreeVariableNames",
        variables,
        monomial_order: "GrlexDescending",
        terms,
    };

    // This structure contains only finite integers, strings, arrays and
    // borrowed slices. `serde_json` has no fallible value class here.
    serde_json::to_string(&view).expect("canonical polynomial view must serialize")
}

fn domain_name(domain: NumericType) -> &'static str {
    match domain {
        NumericType::Int => "Int",
        NumericType::Nat => "Nat",
        NumericType::Real => "Real",
    }
}

fn error(kind: AdapterErrorKind) -> AdapterError {
    AdapterError { kind }
}

fn checked_exponent_add(lhs: &[u32], rhs: &[u32]) -> Result<Vec<u32>, AdapterError> {
    if lhs.len() != rhs.len() {
        return Err(error(AdapterErrorKind::InternalDimensionMismatch));
    }
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| {
            a.checked_add(*b)
                .ok_or_else(|| error(AdapterErrorKind::ExponentOverflow))
        })
        .collect()
}

fn checked_rat_add(lhs: Rat, rhs: Rat) -> Result<Rat, AdapterError> {
    let left = (lhs.num as i128)
        .checked_mul(rhs.den as i128)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    let right = (rhs.num as i128)
        .checked_mul(lhs.den as i128)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    let numerator = left
        .checked_add(right)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    let denominator = (lhs.den as i128)
        .checked_mul(rhs.den as i128)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    reduce_i128(numerator, denominator)
}

fn checked_rat_mul(lhs: Rat, rhs: Rat) -> Result<Rat, AdapterError> {
    // Cross-reduce before multiplication. This both lowers overflow pressure and
    // keeps the accepted i64 envelope much larger than naive product-first math.
    let g1 = gcd_u128(lhs.num.unsigned_abs() as u128, rhs.den as u128);
    let g2 = gcd_u128(rhs.num.unsigned_abs() as u128, lhs.den as u128);

    let lhs_num = (lhs.num as i128) / g1 as i128;
    let rhs_den = (rhs.den as i128) / g1 as i128;
    let rhs_num = (rhs.num as i128) / g2 as i128;
    let lhs_den = (lhs.den as i128) / g2 as i128;

    let numerator = lhs_num
        .checked_mul(rhs_num)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    let denominator = lhs_den
        .checked_mul(rhs_den)
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    reduce_i128(numerator, denominator)
}

fn checked_rat_neg(value: Rat) -> Result<Rat, AdapterError> {
    let numerator = (value.num as i128)
        .checked_neg()
        .ok_or_else(|| error(AdapterErrorKind::CoefficientOverflow))?;
    reduce_i128(numerator, value.den as i128)
}

fn reduce_i128(numerator: i128, denominator: i128) -> Result<Rat, AdapterError> {
    if denominator == 0 {
        return Err(error(AdapterErrorKind::ZeroDenominator));
    }
    if numerator == 0 {
        return Ok(Rat { num: 0, den: 1 });
    }

    let negative = (numerator < 0) ^ (denominator < 0);
    let numerator_abs = numerator.unsigned_abs();
    let denominator_abs = denominator.unsigned_abs();
    let gcd = gcd_u128(numerator_abs, denominator_abs);
    let reduced_num = numerator_abs / gcd;
    let reduced_den = denominator_abs / gcd;

    if reduced_den > i64::MAX as u128 {
        return Err(error(AdapterErrorKind::CoefficientOverflow));
    }

    let num = if negative {
        let min_magnitude = (i64::MAX as u128) + 1;
        if reduced_num == min_magnitude {
            i64::MIN
        } else if reduced_num <= i64::MAX as u128 {
            -(reduced_num as i64)
        } else {
            return Err(error(AdapterErrorKind::CoefficientOverflow));
        }
    } else if reduced_num <= i64::MAX as u128 {
        reduced_num as i64
    } else {
        return Err(error(AdapterErrorKind::CoefficientOverflow));
    };

    Ok(Rat {
        num,
        den: reduced_den as i64,
    })
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}
