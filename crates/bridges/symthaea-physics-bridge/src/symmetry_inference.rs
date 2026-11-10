// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symmetry Inference from Discovered Expressions
//!
//! Walks a ConjectureEngine [`Expr`] and heuristically detects Lie-group
//! symmetries so the recognition query can declare them. Without this, the
//! query hardcodes `SymmetryDescriptor::none()` and any catalog entry
//! claiming its true symmetry drops ~0.20 on the symmetry similarity axis.
//!
//! ## What it detects
//!
//! Two narrow but high-value patterns that cover the Ramanujan Protocol
//! showcase's canonical invariants:
//!
//! 1. **Equal-coefficient sum of squared variables** — `x² + y²`, `x² + y² + z²`, `px² + py² + pz²`
//!    → SO(n) where n = number of distinct squared variables. A shared finite nonzero scalar is allowed; unequal per-variable weights are rejected.
//!    Matches Harmonic Oscillator, kinetic energy, `r² = x² + y² + z²`.
//!
//! 2. **2D antisymmetric cross product** — `a·b - c·d` where `{a,c}` and
//!    `{b,d}` are disjoint pairs → SO(2).
//!    Matches `x·vy - y·vx` (angular momentum z-component).
//!
//! Everything else returns [`SymmetryDescriptor::none()`]. Notably, `x - ln(x)`,
//! polynomial + log combinations, and anything involving transcendentals is
//! NOT classified — we prefer to undersell symmetry than to claim false
//! rotational invariance on a transcendental like the Lotka-Volterra invariant.
//!
//! ## Beyond the two structural heuristics: numeric Lie-derivative fallback
//!
//! The two heuristics above are exact-shape pattern matches — fast, but blind
//! to anything that isn't literally a monomial sum-of-squares or cross-product
//! (e.g. `sqrt(x²+y²)`, `1/(x²+y²)`, or any other radially-symmetric function
//! of the same invariant). [`infer_symmetry_numeric`] closes that gap using the
//! Lie-theory machinery in `symthaea_core::hdc::lie_theory`: for a candidate
//! group action (an so(n) generator acting on some subset of the expression's
//! variables), the expression is invariant iff its Lie derivative along the
//! generator's flow vanishes everywhere — evaluated numerically (finite-
//! difference gradient, matching the same technique
//! `conjecture_engine::gp_support::lie_derivative_variance` already uses for
//! checking conservation along a *time* flow; here the flow is a *symmetry*
//! generator instead) at several sampled points rather than proven
//! symbolically. This is deliberately bounded, not a full search:
//!
//! - joint SO(n) acting on all of the expression's variables at once (n ≤ 6,
//!   for generator-count cost), and
//! - SO(2) acting on each pair of variables independently,
//!
//! which strictly covers the two structural heuristics' cases (any function of
//! `x²+y²+...`, not just the sum itself) while staying tractable. It does not
//! search arbitrary variable subsets/groupings, and it is a numerical check
//! (some sampled points, fixed tolerance) rather than a symbolic proof — treat
//! a positive result as strong evidence, not a certificate.
//!
//! ## Why the structural heuristics still run first
//!
//! They're exact and essentially free; the numeric fallback only runs when
//! they don't fire, keeping the common cases fast and precise.

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_core::hdc::lie_theory::{self, LieAlgebra};

use crate::types::{LieGroup, SymmetryDescriptor};

/// Fully specified numerical policy used by the bounded Lie-derivative fallback.
///
/// The profile is returned with every numerical candidate so a consumer can
/// distinguish a sampled heuristic from a symbolic identity or formal proof.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericLieDerivativeProfileV1 {
    pub seed: u64,
    pub samples_per_generator: usize,
    pub finite_difference_step: f64,
    pub normalized_squared_residual_tolerance: f64,
    pub minimum_gradient_squared: f64,
    pub sample_min_inclusive: f64,
    pub sample_max_exclusive: f64,
    pub maximum_joint_dimension: usize,
    pub pairwise_so2_search: bool,
}

impl Default for NumericLieDerivativeProfileV1 {
    fn default() -> Self {
        Self {
            seed: 0x9E3779B97F4A7C15,
            samples_per_generator: 8,
            finite_difference_step: 1e-5,
            normalized_squared_residual_tolerance: 1e-6,
            minimum_gradient_squared: 1e-8,
            sample_min_inclusive: 0.5,
            sample_max_exclusive: 3.0,
            maximum_joint_dimension: 6,
            pairwise_so2_search: true,
        }
    }
}

/// Resource-bounded exact-polynomial profile for symbolic Lie-derivative checks.
///
/// Float literals in the source AST are interpreted as their exact IEEE-754
/// dyadic rational values. Coefficient arithmetic is then exact within checked
/// `i128` numerator/denominator bounds. This establishes an algebraic identity
/// for the represented polynomial; it is not a Lean proof or a physical-units proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactPolynomialLieProfileV1 {
    pub maximum_variables: usize,
    pub maximum_terms: usize,
    pub maximum_term_products: usize,
    pub maximum_total_degree: usize,
    pub maximum_power: u32,
    pub maximum_dyadic_denominator_power: u32,
    pub pairwise_so2_search: bool,
}

impl Default for ExactPolynomialLieProfileV1 {
    fn default() -> Self {
        Self {
            maximum_variables: 6,
            maximum_terms: 4096,
            maximum_term_products: 65_536,
            maximum_total_degree: 48,
            maximum_power: 16,
            maximum_dyadic_denominator_power: 126,
            pairwise_so2_search: true,
        }
    }
}

/// Outcome of the bounded exact-polynomial parser and Lie-derivative search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactPolynomialSearchStatus {
    NotRequested,
    NotRunStructuralShortcut,
    IdentityFound,
    CompletedNoCandidate,
    SkippedTooFewVariables,
    UnsupportedExpression,
    ResourceLimitExceeded,
}

/// Structural pattern that produced a symmetry candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StructuralSymmetryPattern {
    /// A sum of distinct squared variables with one shared finite nonzero coefficient.
    EqualCoefficientSumOfSquares {
        variable_count: u8,
        shared_coefficient: f64,
    },
    /// A two-dimensional antisymmetric bilinear determinant with four distinct variables.
    TwoDimensionalAntisymmetricBilinear,
}

/// Evidence source for a candidate symmetry.
///
/// `StructuralPatternMatch` means the expression matched a narrowly specified
/// exact AST pattern; it is not a Lean proof or a claim about physical units.
/// `SampledLieDerivative` is empirical evidence over a finite, declared sample
/// domain and must never be promoted to a symbolic/formal proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SymmetryInferenceEvidence {
    StructuralPatternMatch { pattern: StructuralSymmetryPattern },
    ExactPolynomialLieDerivative {
        profile: ExactPolynomialLieProfileV1,
        acted_on_variables: Vec<String>,
        group: LieGroup,
        checked_generator_count: usize,
    },
    SampledLieDerivative {
        profile: NumericLieDerivativeProfileV1,
        acted_on_variables: Vec<String>,
        group: LieGroup,
    },
}

/// How far the symmetry-inference pipeline progressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymmetryInferenceSearchStatus {
    /// An exact-shape structural detector matched; numeric search was short-circuited.
    StructuralMatchShortCircuit,
    /// The bounded exact-polynomial Lie-derivative identity was zero for every searched generator.
    ExactPolynomialIdentityFound,
    /// The supported polynomial was checked exactly against the searched group families; none matched.
    ExactPolynomialSearchCompletedNoCandidate,
    /// The bounded numerical search ran to completion.
    NumericSearchCompleted,
    /// The expression had fewer than two free variables, so numerical rotation search was skipped.
    NumericSearchSkippedTooFewVariables,
}

/// Auditable result of symmetry inference.
///
/// A missing candidate means only that this bounded search found no candidate;
/// it is not evidence that the expression has no symmetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymmetryInferenceReport {
    pub candidate: Option<SymmetryDescriptor>,
    pub evidence: Option<SymmetryInferenceEvidence>,
    pub search_status: SymmetryInferenceSearchStatus,
    pub numeric_profile: NumericLieDerivativeProfileV1,
    pub exact_polynomial_profile: ExactPolynomialLieProfileV1,
    pub exact_polynomial_status: ExactPolynomialSearchStatus,
}

impl SymmetryInferenceReport {
    /// Compatibility projection for APIs that still consume only a descriptor.
    pub fn descriptor_or_none(&self) -> SymmetryDescriptor {
        self.candidate.clone().unwrap_or_else(SymmetryDescriptor::none)
    }
}
/// Infer a best-effort symmetry descriptor from a discovered expression.
///
/// This compatibility API intentionally discards evidence provenance. New
/// claim-bearing callers should use [`infer_symmetry_with_evidence`].
pub fn infer_symmetry(expr: &Expr) -> SymmetryDescriptor {
    infer_symmetry_with_evidence(expr).descriptor_or_none()
}

/// Infer a symmetry candidate while preserving the evidence source and search limits.
pub fn infer_symmetry_with_evidence(expr: &Expr) -> SymmetryInferenceReport {
    let numeric_profile = NumericLieDerivativeProfileV1::default();
    let exact_polynomial_profile = ExactPolynomialLieProfileV1::default();

    if let Some((n, shared_coefficient)) = detect_sum_of_squares(expr)
        && (2..=10).contains(&n)
    {
        return SymmetryInferenceReport {
            candidate: Some(SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(n)], false)),
            evidence: Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::EqualCoefficientSumOfSquares {
                    variable_count: n,
                    shared_coefficient,
                },
            }),
            search_status: SymmetryInferenceSearchStatus::StructuralMatchShortCircuit,
            numeric_profile,
            exact_polynomial_profile,
            exact_polynomial_status: ExactPolynomialSearchStatus::NotRunStructuralShortcut,
        };
    }

    if detect_2d_antisymmetric_cross(expr) {
        return SymmetryInferenceReport {
            candidate: Some(SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false)),
            evidence: Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::TwoDimensionalAntisymmetricBilinear,
            }),
            search_status: SymmetryInferenceSearchStatus::StructuralMatchShortCircuit,
            numeric_profile,
            exact_polynomial_profile,
            exact_polynomial_status: ExactPolynomialSearchStatus::NotRunStructuralShortcut,
        };
    }

    let mut var_names = Vec::new();
    collect_var_names(expr, &mut var_names);
    if var_names.len() < 2 {
        return SymmetryInferenceReport {
            candidate: None,
            evidence: None,
            search_status: SymmetryInferenceSearchStatus::NumericSearchSkippedTooFewVariables,
            numeric_profile,
            exact_polynomial_profile,
            exact_polynomial_status: ExactPolynomialSearchStatus::SkippedTooFewVariables,
        };
    }

    match exact_polynomial_symmetry_candidate(expr, &var_names, exact_polynomial_profile) {
        Ok(Some((group, acted_on_variables, checked_generator_count))) => {
            SymmetryInferenceReport {
                candidate: Some(SymmetryDescriptor::from_lie_groups(vec![group], false)),
                evidence: Some(SymmetryInferenceEvidence::ExactPolynomialLieDerivative {
                    profile: exact_polynomial_profile,
                    acted_on_variables,
                    group,
                    checked_generator_count,
                }),
                search_status: SymmetryInferenceSearchStatus::ExactPolynomialIdentityFound,
                numeric_profile,
                exact_polynomial_profile,
                exact_polynomial_status: ExactPolynomialSearchStatus::IdentityFound,
            }
        }
        Ok(None) => SymmetryInferenceReport {
            candidate: None,
            evidence: None,
            search_status: SymmetryInferenceSearchStatus::ExactPolynomialSearchCompletedNoCandidate,
            numeric_profile,
            exact_polynomial_profile,
            exact_polynomial_status: ExactPolynomialSearchStatus::CompletedNoCandidate,
        },
        Err(ExactPolynomialError::UnsupportedExpression) => infer_symmetry_numeric_with_profile(
            expr,
            numeric_profile,
            exact_polynomial_profile,
            ExactPolynomialSearchStatus::UnsupportedExpression,
        ),
        Err(ExactPolynomialError::ResourceLimitExceeded) => infer_symmetry_numeric_with_profile(
            expr,
            numeric_profile,
            exact_polynomial_profile,
            ExactPolynomialSearchStatus::ResourceLimitExceeded,
        ),
    }
}

/// The generic numeric fallback, with its finite search domain and tolerances in the result.
pub fn infer_symmetry_numeric_with_evidence(expr: &Expr) -> SymmetryInferenceReport {
    infer_symmetry_numeric_with_profile(
        expr,
        NumericLieDerivativeProfileV1::default(),
        ExactPolynomialLieProfileV1::default(),
        ExactPolynomialSearchStatus::NotRequested,
    )
}

/// Compatibility API for callers that need only a descriptor from the numeric fallback.
pub fn infer_symmetry_numeric(expr: &Expr) -> SymmetryDescriptor {
    infer_symmetry_numeric_with_evidence(expr).descriptor_or_none()
}

fn infer_symmetry_numeric_with_profile(
    expr: &Expr,
    profile: NumericLieDerivativeProfileV1,
    exact_polynomial_profile: ExactPolynomialLieProfileV1,
    exact_polynomial_status: ExactPolynomialSearchStatus,
) -> SymmetryInferenceReport {
    let mut var_names = Vec::new();
    collect_var_names(expr, &mut var_names);

    if var_names.len() < 2 {
        return SymmetryInferenceReport {
            candidate: None,
            evidence: None,
            search_status: SymmetryInferenceSearchStatus::NumericSearchSkippedTooFewVariables,
            numeric_profile: profile,
            exact_polynomial_profile,
            exact_polynomial_status,
        };
    }

    // Joint rotation of the first n variables (bounded by the declared profile).
    let n = var_names.len().min(profile.maximum_joint_dimension);
    let joint_algebra = lie_theory::so_n(n);
    let joint_indices: Vec<usize> = (0..n).collect();
    if is_lie_invariant(expr, &var_names, &joint_indices, &joint_algebra, profile) {
        let group = LieGroup::SO(n as u8);
        return SymmetryInferenceReport {
            candidate: Some(SymmetryDescriptor::from_lie_groups(vec![group], false)),
            evidence: Some(SymmetryInferenceEvidence::SampledLieDerivative {
                profile,
                acted_on_variables: var_names[..n].to_vec(),
                group,
            }),
            search_status: SymmetryInferenceSearchStatus::NumericSearchCompleted,
            numeric_profile: profile,
            exact_polynomial_profile,
            exact_polynomial_status,
        };
    }

    // Pairwise SO(2) search over every distinct variable pair when enabled.
    if profile.pairwise_so2_search {
        let so2 = lie_theory::so_n(2);
        for i in 0..var_names.len() {
            for j in (i + 1)..var_names.len() {
                if is_lie_invariant(expr, &var_names, &[i, j], &so2, profile) {
                    let group = LieGroup::SO(2);
                    return SymmetryInferenceReport {
                        candidate: Some(SymmetryDescriptor::from_lie_groups(vec![group], false)),
                        evidence: Some(SymmetryInferenceEvidence::SampledLieDerivative {
                            profile,
                            acted_on_variables: vec![var_names[i].clone(), var_names[j].clone()],
                            group,
                        }),
                        search_status: SymmetryInferenceSearchStatus::NumericSearchCompleted,
                        numeric_profile: profile,
                        exact_polynomial_profile,
                        exact_polynomial_status,
                    };
                }
            }
        }
    }

    SymmetryInferenceReport {
        candidate: None,
        evidence: None,
        search_status: SymmetryInferenceSearchStatus::NumericSearchCompleted,
        numeric_profile: profile,
        exact_polynomial_profile,
        exact_polynomial_status,
    }
}
/// Collect free variable names referenced by `expr`, first-seen order, deduplicated.
fn collect_var_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            collect_var_names(l, out);
            collect_var_names(r, out);
        }
        Expr::Func(_, arg) => collect_var_names(arg, out),
        Expr::Sum(body, var) => {
            collect_var_names(body, out);
            if !out.contains(var) {
                out.push(var.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactPolynomialError {
    UnsupportedExpression,
    ResourceLimitExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactRational {
    numerator: i128,
    denominator: i128,
}

impl ExactRational {
    const ZERO: Self = Self { numerator: 0, denominator: 1 };
    const ONE: Self = Self { numerator: 1, denominator: 1 };

    fn new(mut numerator: i128, mut denominator: i128) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        if numerator == 0 {
            return Some(Self::ZERO);
        }
        if denominator < 0 {
            numerator = numerator.checked_neg()?;
            denominator = denominator.checked_neg()?;
        }
        let divisor = exact_gcd(numerator.unsigned_abs(), denominator as u128);
        let divisor = i128::try_from(divisor).ok()?;
        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn from_f64(value: f64, maximum_denominator_power: u32) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::ZERO);
        }

        // Decode the exact binary rational represented by this IEEE-754 value.
        let bits = value.to_bits();
        let negative = (bits >> 63) != 0;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mut significand, mut binary_exponent) = if exponent_bits == 0 {
            (fraction, -1022 - 52)
        } else {
            (fraction | (1_u64 << 52), exponent_bits - 1023 - 52)
        };
        if significand == 0 {
            return Some(Self::ZERO);
        }

        let trailing_zeros = significand.trailing_zeros();
        significand >>= trailing_zeros;
        binary_exponent += trailing_zeros as i32;

        let magnitude = i128::from(significand);
        if binary_exponent >= 0 {
            let shift = u32::try_from(binary_exponent).ok()?;
            if shift > 126 {
                return None;
            }
            let scale = 1_i128.checked_shl(shift)?;
            let numerator = magnitude.checked_mul(scale)?;
            let numerator = if negative { numerator.checked_neg()? } else { numerator };
            Self::new(numerator, 1)
        } else {
            let denominator_power = u32::try_from(-binary_exponent).ok()?;
            if denominator_power > maximum_denominator_power || denominator_power > 126 {
                return None;
            }
            let denominator = 1_i128.checked_shl(denominator_power)?;
            let numerator = if negative { magnitude.checked_neg()? } else { magnitude };
            Self::new(numerator, denominator)
        }
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        let numerator = left.checked_add(right)?;
        let denominator = self.denominator.checked_mul(other.denominator)?;
        Self::new(numerator, denominator)
    }

    fn checked_mul(self, other: Self) -> Option<Self> {
        let numerator = self.numerator.checked_mul(other.numerator)?;
        let denominator = self.denominator.checked_mul(other.denominator)?;
        Self::new(numerator, denominator)
    }

    fn checked_neg(self) -> Option<Self> {
        Some(Self {
            numerator: self.numerator.checked_neg()?,
            denominator: self.denominator,
        })
    }

    fn is_zero(self) -> bool {
        self.numerator == 0
    }
}

fn exact_gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

type ExactPolynomial = BTreeMap<Vec<u8>, ExactRational>;

fn check_monomial(
    monomial: &[u8],
    profile: ExactPolynomialLieProfileV1,
) -> Result<(), ExactPolynomialError> {
    let degree: usize = monomial.iter().map(|power| usize::from(*power)).sum();
    if degree > profile.maximum_total_degree {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    Ok(())
}

fn polynomial_add_term(
    polynomial: &mut ExactPolynomial,
    monomial: Vec<u8>,
    coefficient: ExactRational,
    profile: ExactPolynomialLieProfileV1,
) -> Result<(), ExactPolynomialError> {
    check_monomial(&monomial, profile)?;
    if coefficient.is_zero() {
        return Ok(());
    }

    if let Some(existing) = polynomial.get(&monomial).copied() {
        let combined = existing
            .checked_add(coefficient)
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        if combined.is_zero() {
            polynomial.remove(&monomial);
        } else {
            polynomial.insert(monomial, combined);
        }
    } else {
        polynomial.insert(monomial, coefficient);
    }

    if polynomial.len() > profile.maximum_terms {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    Ok(())
}

fn polynomial_constant(
    dimension: usize,
    coefficient: ExactRational,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut polynomial = ExactPolynomial::new();
    polynomial_add_term(&mut polynomial, vec![0; dimension], coefficient, profile)?;
    Ok(polynomial)
}

fn polynomial_add(
    left: &ExactPolynomial,
    right: &ExactPolynomial,
    negate_right: bool,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut result = left.clone();
    for (monomial, coefficient) in right {
        let coefficient = if negate_right {
            coefficient.checked_neg().ok_or(ExactPolynomialError::ResourceLimitExceeded)?
        } else {
            *coefficient
        };
        polynomial_add_term(&mut result, monomial.clone(), coefficient, profile)?;
    }
    Ok(result)
}

fn polynomial_multiply(
    left: &ExactPolynomial,
    right: &ExactPolynomial,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut result = ExactPolynomial::new();
    let mut term_products = 0usize;
    for (left_monomial, left_coefficient) in left {
        for (right_monomial, right_coefficient) in right {
            term_products = term_products.saturating_add(1);
            if term_products > profile.maximum_term_products {
                return Err(ExactPolynomialError::ResourceLimitExceeded);
            }
            let mut monomial = Vec::with_capacity(left_monomial.len());
            for (left_power, right_power) in left_monomial.iter().zip(right_monomial) {
                monomial.push(
                    left_power
                        .checked_add(*right_power)
                        .ok_or(ExactPolynomialError::ResourceLimitExceeded)?,
                );
            }
            let coefficient = left_coefficient
                .checked_mul(*right_coefficient)
                .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
            polynomial_add_term(&mut result, monomial, coefficient, profile)?;
        }
    }
    Ok(result)
}

fn polynomial_power(
    base: &ExactPolynomial,
    exponent: u32,
    dimension: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    if exponent > profile.maximum_power {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    let mut result = polynomial_constant(dimension, ExactRational::ONE, profile)?;
    let mut factor = base.clone();
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = polynomial_multiply(&result, &factor, profile)?;
        }
        remaining >>= 1;
        if remaining > 0 {
            factor = polynomial_multiply(&factor, &factor, profile)?;
        }
    }
    Ok(result)
}

fn parse_exact_polynomial(
    expr: &Expr,
    variable_names: &[String],
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let dimension = variable_names.len();
    match expr {
        Expr::Const(value) => {
            let coefficient = ExactRational::from_f64(
                *value,
                profile.maximum_dyadic_denominator_power,
            )
            .ok_or(ExactPolynomialError::UnsupportedExpression)?;
            polynomial_constant(dimension, coefficient, profile)
        }
        Expr::Var(name) => {
            let index = variable_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or(ExactPolynomialError::UnsupportedExpression)?;
            let mut monomial = vec![0; dimension];
            monomial[index] = 1;
            let mut polynomial = ExactPolynomial::new();
            polynomial_add_term(&mut polynomial, monomial, ExactRational::ONE, profile)?;
            Ok(polynomial)
        }
        Expr::BinOp(BinOp::Add, left, right) => {
            let left = parse_exact_polynomial(left, variable_names, profile)?;
            let right = parse_exact_polynomial(right, variable_names, profile)?;
            polynomial_add(&left, &right, false, profile)
        }
        Expr::BinOp(BinOp::Sub, left, right) => {
            let left = parse_exact_polynomial(left, variable_names, profile)?;
            let right = parse_exact_polynomial(right, variable_names, profile)?;
            polynomial_add(&left, &right, true, profile)
        }
        Expr::BinOp(BinOp::Mul, left, right) => {
            let left = parse_exact_polynomial(left, variable_names, profile)?;
            let right = parse_exact_polynomial(right, variable_names, profile)?;
            polynomial_multiply(&left, &right, profile)
        }
        Expr::BinOp(BinOp::Pow, base, exponent) => {
            let Expr::Const(exponent) = exponent.as_ref() else {
                return Err(ExactPolynomialError::UnsupportedExpression);
            };
            if !exponent.is_finite() || exponent.fract() != 0.0 || *exponent < 0.0 {
                return Err(ExactPolynomialError::UnsupportedExpression);
            }
            let exponent = u32::try_from(*exponent as u64)
                .map_err(|_| ExactPolynomialError::ResourceLimitExceeded)?;
            let base = parse_exact_polynomial(base, variable_names, profile)?;
            polynomial_power(&base, exponent, dimension, profile)
        }
        Expr::BinOp(BinOp::Div, _, _) | Expr::Func(_, _) | Expr::Sum(_, _) => {
            Err(ExactPolynomialError::UnsupportedExpression)
        }
    }
}

fn polynomial_derivative(
    polynomial: &ExactPolynomial,
    variable_index: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut result = ExactPolynomial::new();
    for (monomial, coefficient) in polynomial {
        let exponent = monomial[variable_index];
        if exponent == 0 {
            continue;
        }
        let scaled = coefficient
            .checked_mul(ExactRational::new(i128::from(exponent), 1).unwrap())
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        let mut reduced = monomial.clone();
        reduced[variable_index] -= 1;
        polynomial_add_term(&mut result, reduced, scaled, profile)?;
    }
    Ok(result)
}

fn polynomial_multiply_variable(
    polynomial: &ExactPolynomial,
    variable_index: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut result = ExactPolynomial::new();
    for (monomial, coefficient) in polynomial {
        let mut multiplied = monomial.clone();
        multiplied[variable_index] = multiplied[variable_index]
            .checked_add(1)
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        polynomial_add_term(&mut result, multiplied, *coefficient, profile)?;
    }
    Ok(result)
}

/// Exact Lie derivative for the SO(n) generator with δx_i=-x_j, δx_j=x_i.
fn exact_polynomial_lie_derivative_is_zero(
    polynomial: &ExactPolynomial,
    i: usize,
    j: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<bool, ExactPolynomialError> {
    let derivative_i = polynomial_derivative(polynomial, i, profile)?;
    let derivative_j = polynomial_derivative(polynomial, j, profile)?;
    let term_i = polynomial_multiply_variable(&derivative_i, j, profile)?;
    let term_j = polynomial_multiply_variable(&derivative_j, i, profile)?;
    let residual = polynomial_add(&term_j, &term_i, true, profile)?;
    Ok(residual.is_empty())
}

fn exact_polynomial_symmetry_candidate(
    expr: &Expr,
    variable_names: &[String],
    profile: ExactPolynomialLieProfileV1,
) -> Result<Option<(LieGroup, Vec<String>, usize)>, ExactPolynomialError> {
    if variable_names.len() < 2 {
        return Ok(None);
    }
    if variable_names.len() > profile.maximum_variables {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    let polynomial = parse_exact_polynomial(expr, variable_names, profile)?;
    if polynomial
        .keys()
        .all(|monomial| monomial.iter().all(|exponent| *exponent == 0))
    {
        // Constant/zero expressions carry no useful state-dependent symmetry signal.
    let generator_count = dimension * (dimension - 1) / 2;

    let mut all_generators_invariant = true;
    for i in 0..dimension {
        for j in (i + 1)..dimension {
            if !exact_polynomial_lie_derivative_is_zero(&polynomial, i, j, profile)? {
                all_generators_invariant = false;
                break;
            }
        }
        if !all_generators_invariant {
            break;
        }
    }
    if all_generators_invariant {
        return Ok(Some((
            LieGroup::SO(dimension as u8),
            variable_names.to_vec(),
            generator_count,
        )));
    }

    if profile.pairwise_so2_search {
        for i in 0..dimension {
            for j in (i + 1)..dimension {
                if exact_polynomial_lie_derivative_is_zero(&polynomial, i, j, profile)? {
                    return Ok(Some((
                        LieGroup::SO(2),
                        vec![variable_names[i].clone(), variable_names[j].clone()],
                        1,
                    )));
                }
            }
        }
    }
    Ok(None)
}
/// Central-difference gradient of `expr` at `state`, w.r.t. `var_names` (same
/// binding order as `state`). Mirrors
/// `conjecture_engine::gp_support::fd_gradient`, reimplemented locally since
/// that helper is `pub(crate)` to symthaea-core.
fn fd_gradient(
    expr: &Expr,
    state: &[f64],
    var_names: &[String],
    profile: NumericLieDerivativeProfileV1,
) -> Vec<f64> {
    let eps = profile.finite_difference_step;
    let mut grad = Vec::with_capacity(var_names.len());
    for i in 0..var_names.len() {
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let bindings_plus: Vec<(&str, f64)> = var_names
            .iter()
            .map(|s| s.as_str())
            .zip(plus.iter().copied())
            .collect();
        let bindings_minus: Vec<(&str, f64)> = var_names
            .iter()
            .map(|s| s.as_str())
            .zip(minus.iter().copied())
            .collect();
        grad.push((expr.eval(&bindings_plus) - expr.eval(&bindings_minus)) / (2.0 * eps));
    }
    grad
}

/// Deterministic xorshift64 sampler — no external RNG dependency, reproducible
/// across runs (important for test stability).
fn sample_states(dim: usize, profile: NumericLieDerivativeProfileV1) -> Vec<Vec<f64>> {
    let mut state = profile.seed | 1;
    (0..profile.samples_per_generator)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let mut x = state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    state = x;
                    let unit = (x >> 11) as f64 / (1u64 << 53) as f64;
                    profile.sample_min_inclusive
                        + unit * (profile.sample_max_exclusive - profile.sample_min_inclusive)
                })
                .collect()
        })
        .collect()
}

/// Numerically test whether `expr` is invariant under the group generated by
/// `algebra`, acting on the variables at `active_indices` (all other
/// variables held fixed). Requires the normalized squared Lie derivative to
/// be below tolerance at every sampled point, for every generator — and
/// requires at least one sample to be "informative" (non-vanishing gradient
/// on the active subspace), so an expression that's merely locally flat at
/// every sampled point doesn't get misread as globally invariant.
fn is_lie_invariant(
    expr: &Expr,
    var_names: &[String],
    active_indices: &[usize],
    algebra: &LieAlgebra,
    profile: NumericLieDerivativeProfileV1,
) -> bool {
    let samples = sample_states(var_names.len(), profile);

    let mut informative = false;
    for generator in &algebra.basis {
        for state in &samples {
            let grad = fd_gradient(expr, state, var_names, profile);
            if grad.iter().any(|g| !g.is_finite()) {
                return false;
            }

            // flow[active_indices[gi]] = sum_gj generator[gi][gj] * state[active_indices[gj]]
            let mut flow = vec![0.0; var_names.len()];
            for (gi, &vi) in active_indices.iter().enumerate() {
                let mut acc = 0.0;
                for (gj, &vj) in active_indices.iter().enumerate() {
                    acc += generator[gi][gj] * state[vj];
                }
                flow[vi] = acc;
            }

            let lie: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();
            let grad_sq: f64 = active_indices.iter().map(|&vi| grad[vi] * grad[vi]).sum();
            if !lie.is_finite() || !grad_sq.is_finite() {
                return false;
            }
            if grad_sq < profile.minimum_gradient_squared {
                continue;
            }
            informative = true;
            if (lie * lie) / grad_sq > profile.normalized_squared_residual_tolerance {
                return false;
            }
        }
    }
    informative
}

/// Flatten an `Add` tree into a vector of leaf terms. Unlike the recognize
/// module's `flatten_add`, we stop at all non-Add boundaries *without*
/// descending into Sub — the two have different semantics for symmetry
/// inference: a `Sub` at the top level means the expression is not a simple
/// sum and shouldn't be treated as a sum-of-squares candidate.
fn flatten_add_only<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::BinOp(BinOp::Add, l, r) = expr {
        flatten_add_only(l, out);
        flatten_add_only(r, out);
    } else {
        out.push(expr);
    }
}

/// If expr is `c * (x₁² + ... + xₖ²)` for k ≥ 2 distinct variables and one
/// shared finite, nonzero coefficient c, return k. Unequal weights define an
/// ellipsoid rather than an SO(k)-invariant quadratic form and are rejected.
fn detect_sum_of_squares(expr: &Expr) -> Option<(u8, f64)> {
    let mut terms = Vec::new();
    flatten_add_only(expr, &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut seen = Vec::new();
    let mut common_coefficient: Option<f64> = None;
    for term in terms {
        let (var, coefficient) = squared_term(term)?;
        if !coefficient.is_finite() || coefficient == 0.0 {
            return None;
        }
        if let Some(common) = common_coefficient {
            // Coefficients are AST literals: require exact equality rather
            // than a tolerance that could promote an anisotropic form.
            if coefficient != common {
                return None;
            }
        } else {
            common_coefficient = Some(coefficient);
        }

        // Reject duplicates — they'd over-count the symmetry group dimension.
        if seen.contains(&var) {
            return None;
        }
        seen.push(var);
    }
    Some((seen.len() as u8, common_coefficient?))
}

/// Return (variable name, total scalar coefficient) for a squared variable.
/// Nested scalar multiplications are accumulated, and non-finite products fail.
fn squared_term(expr: &Expr) -> Option<(String, f64)> {
    match expr {
        Expr::BinOp(BinOp::Pow, base, exp) => {
            if let (Expr::Var(name), Expr::Const(k)) = (base.as_ref(), exp.as_ref())
                && (*k - 2.0).abs() < 1e-12
            {
                return Some((name.clone(), 1.0));
            }
            None
        }
        Expr::BinOp(BinOp::Mul, l, r) => {
            match (l.as_ref(), r.as_ref()) {
                (Expr::Const(coefficient), other) | (other, Expr::Const(coefficient)) => {
                    let (name, inner_coefficient) = squared_term(other)?;
                    let total = *coefficient * inner_coefficient;
                    total.is_finite().then_some((name, total))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Detect `a·b - c·d` where `{a,c}` and `{b,d}` are two distinct variable
/// pairs, modeling an antisymmetric cross product.
///
/// Example matches: `x*vy - y*vx`, `p*q - q*p` → SO(2). The match is
/// structural only; we don't verify that the variables form an actual
/// canonical pair (position/momentum), just that the shape is antisymmetric.
fn detect_2d_antisymmetric_cross(expr: &Expr) -> bool {
    let Expr::BinOp(BinOp::Sub, l, r) = expr else {
        return false;
    };
    let Expr::BinOp(BinOp::Mul, l1, l2) = l.as_ref() else {
        return false;
    };
    let Expr::BinOp(BinOp::Mul, r1, r2) = r.as_ref() else {
        return false;
    };
    let (l1v, l2v, r1v, r2v) = match (l1.as_ref(), l2.as_ref(), r1.as_ref(), r2.as_ref()) {
        (Expr::Var(a), Expr::Var(b), Expr::Var(c), Expr::Var(d)) => (a, b, c, d),
        _ => return false,
    };
    // {l1, r1} and {l2, r2} should each be a pair of distinct variables,
    // and the two pairs together must use 4 distinct names (e.g. x, vy, y, vx).
    let all = [l1v, l2v, r1v, r2v];
    let unique_count = all
        .iter()
        .enumerate()
        .filter(|(i, name)| !all[..*i].contains(name))
        .count();
    if unique_count != 4 {
        return false;
    }
    // The shape `a·b - c·d` is antisymmetric iff swapping `a↔c` and `b↔d`
    // negates the expression, which is automatic when {a,c} and {b,d} are
    // the two pairs of the cross product.
    true
}

// Silence unused-import warning when UnaryFn is referenced only in rustdoc.
#[allow(dead_code)]
fn _unused(_: UnaryFn) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow2(name: &str) -> Expr {
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(name.to_string())),
            Box::new(Expr::Const(2.0)),
        )
    }

    fn add(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(l), Box::new(r))
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    #[test]
    fn structural_symmetry_report_preserves_pattern_provenance() {
        let expr = add(pow2("x"), pow2("y"));
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(
            report.candidate.unwrap().lie_groups,
            vec![LieGroup::SO(2)]
        );
        assert_eq!(
            report.evidence,
            Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::EqualCoefficientSumOfSquares {
                    variable_count: 2,
                    shared_coefficient: 1.0,
                },
            })
        );
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::StructuralMatchShortCircuit
        );
    }

    #[test]
    fn numeric_symmetry_report_is_explicitly_sampled_not_proven() {
        let expr = sqrt_of(add(pow2("x"), pow2("y")));
        let report = infer_symmetry_numeric_with_evidence(&expr);
        assert_eq!(report.candidate.unwrap().lie_groups, vec![LieGroup::SO(2)]);
        let Some(SymmetryInferenceEvidence::SampledLieDerivative {
            profile,
            group,
            acted_on_variables,
        }) = report.evidence
        else {
            panic!("numeric fallback must preserve sampled evidence provenance");
        };
        assert_eq!(group, LieGroup::SO(2));
        assert_eq!(acted_on_variables, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(profile.samples_per_generator, 8);
        assert_eq!(profile.finite_difference_step, 1e-5);
        assert_eq!(profile.sample_min_inclusive, 0.5);
        assert_eq!(profile.sample_max_exclusive, 3.0);
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::NumericSearchCompleted
        );
    }

    #[test]
    fn near_isotropic_quadratic_candidate_remains_sampled_evidence() {
        // The 1e-5 anisotropy is below the declared sampled residual threshold.
        // It must not be represented as an exact structural pattern or formal proof.
        let expr = sqrt_of(add(
            pow2("x"),
            mul(Expr::Const(1.00001), pow2("y")),
        ));
        assert!(detect_sum_of_squares(&expr).is_none());
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(
            report.candidate.as_ref().unwrap().lie_groups,
            vec![LieGroup::SO(2)]
        );
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::SampledLieDerivative { .. })
        ));
    }
    #[test]
    fn no_candidate_report_does_not_claim_absence_of_symmetry() {
        let report = infer_symmetry_with_evidence(&mul(var("x"), var("y")));
        assert!(report.candidate.is_none());
        assert!(report.evidence.is_none());
        assert_eq!(report.search_status, SymmetryInferenceSearchStatus::ExactPolynomialSearchCompletedNoCandidate);
        assert_eq!(report.exact_polynomial_status, ExactPolynomialSearchStatus::CompletedNoCandidate);
    }

    #[test]
    fn numeric_search_skip_is_a_distinct_status() {
        let report = infer_symmetry_numeric_with_evidence(&pow2("x"));
        assert!(report.candidate.is_none());
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::NumericSearchSkippedTooFewVariables
        );
    }
    #[test]
    fn harmonic_oscillator_infers_so2() {
        let expr = add(pow2("x"), pow2("v"));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn three_d_sum_of_squares_infers_so3() {
        let expr = add(add(pow2("x"), pow2("y")), pow2("z"));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(3)]);
    }

    #[test]
    fn unequal_quadratic_weights_do_not_infer_rotational_symmetry() {
        // x² + 2y² is an ellipse, not an SO(2)-invariant radial form.
        let expr = add(
            mul(Expr::Const(1.0), pow2("x")),
            mul(Expr::Const(2.0), pow2("y")),
        );
        assert!(detect_sum_of_squares(&expr).is_none());
        assert!(infer_symmetry(&expr).lie_groups.is_empty());
    }

    #[test]
    fn equal_nonzero_quadratic_weights_preserve_rotational_symmetry() {
        let expr = add(
            mul(Expr::Const(2.5), pow2("x")),
            mul(Expr::Const(2.5), pow2("y")),
        );
        assert_eq!(detect_sum_of_squares(&expr), Some((2, 2.5)));
        assert_eq!(infer_symmetry(&expr).lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn negative_common_quadratic_weight_preserves_rotational_symmetry() {
        let expr = add(
            mul(Expr::Const(-3.0), pow2("x")),
            mul(Expr::Const(-3.0), pow2("y")),
        );
        assert_eq!(detect_sum_of_squares(&expr), Some((2, -3.0)));
        assert_eq!(infer_symmetry(&expr).lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn zero_quadratic_weight_does_not_create_rotational_symmetry() {
        let expr = add(
            mul(Expr::Const(0.0), pow2("x")),
            mul(Expr::Const(2.0), pow2("y")),
        );
        assert!(detect_sum_of_squares(&expr).is_none());
        assert!(infer_symmetry(&expr).lie_groups.is_empty());
    }

    #[test]
    fn angular_momentum_cartesian_infers_so2() {
        // x*vy - y*vx
        let expr = sub(mul(var("x"), var("vy")), mul(var("y"), var("vx")));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn lv_invariant_stays_none() {
        // (x - ln(x)) + (y - ln(y)) — transcendental, must NOT claim rotational symmetry.
        let ln = |name: &str| Expr::Func(UnaryFn::Log, Box::new(var(name)));
        let expr = add(sub(var("x"), ln("x")), sub(var("y"), ln("y")));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn duplicate_squared_var_rejected() {
        // x² + x² should not classify as SO(2) — duplicates indicate a
        // coefficient mismatch, not a rotational symmetry.
        let expr = add(pow2("x"), pow2("x"));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn mixed_power_rejected() {
        // x² + y³ — not a pure sum of squares, shouldn't trigger.
        let expr = add(
            pow2("x"),
            Expr::BinOp(BinOp::Pow, Box::new(var("y")), Box::new(Expr::Const(3.0))),
        );
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn single_squared_var_not_enough() {
        // Just `x²` is scale-symmetric, not rotational — don't claim SO(1).
        let sym = infer_symmetry(&pow2("x"));
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn same_pair_cross_rejected() {
        // x*y - y*x is zero, structurally degenerate; should not classify.
        let expr = sub(mul(var("x"), var("y")), mul(var("y"), var("x")));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    // ─── Numeric fallback: cases the structural heuristics CANNOT catch ───

    fn sqrt_of(e: Expr) -> Expr {
        Expr::Func(UnaryFn::Sqrt, Box::new(e))
    }

    #[test]
    fn radial_magnitude_infers_so2_via_numeric_fallback() {
        // sqrt(x^2 + y^2) is SO(2)-invariant, but is not a bare sum of
        // squares (it's sqrt of one) — the structural heuristic cannot match
        // this shape at all. This is the actual capability gain: any function
        // of the invariant, not just the invariant itself.
        let expr = sqrt_of(add(pow2("x"), pow2("y")));
        assert!(
            detect_sum_of_squares(&expr).is_none(),
            "sanity: structural heuristic must NOT match sqrt(..)"
        );
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn inverse_square_radial_infers_so3_via_numeric_fallback() {
        // 1/(x^2+y^2+z^2) — Kepler-potential-shaped, SO(3)-invariant, and
        // again not a literal sum-of-squares shape.
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(add(add(pow2("x"), pow2("y")), pow2("z"))),
        );
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(3)]);
    }

    #[test]
    fn non_invariant_product_stays_none_under_numeric_fallback() {
        // x*y is NOT rotationally invariant (it picks up x^2-y^2 cross terms
        // under a generic rotation) — must not false-positive.
        let expr = mul(var("x"), var("y"));
        let sym = infer_symmetry_numeric(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn lv_invariant_stays_none_under_numeric_fallback_too() {
        // Re-run the transcendental Lotka-Volterra case directly against the
        // numeric fallback (not just the combined infer_symmetry), since this
        // is exactly the shape a Lie-derivative search could plausibly
        // false-positive on if the tolerance were too loose.
        let ln = |name: &str| Expr::Func(UnaryFn::Log, Box::new(var(name)));
        let expr = add(sub(var("x"), ln("x")), sub(var("y"), ln("y")));
        let sym = infer_symmetry_numeric(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn single_variable_never_attempts_numeric_fallback() {
        // Fewer than 2 variables: no rotation is even well-defined.
        let sym = infer_symmetry_numeric(&pow2("x"));
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn four_var_sum_of_squares_infers_so4_via_numeric_fallback() {
        let expr = add(add(add(pow2("x"), pow2("y")), pow2("z")), pow2("w"));
        // Structural heuristic already covers this one (regression check),
        // but confirm the numeric path independently agrees.
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(4)]);
    }
}
