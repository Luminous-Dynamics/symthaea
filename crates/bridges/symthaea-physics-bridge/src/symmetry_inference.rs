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
//! ## Exact polynomial and sampled Lie-derivative lanes
//!
//! For bounded polynomial ASTs (constants, variables, addition, subtraction,
//! multiplication and non-negative integer powers), the exact lane normalizes
//! the expression into sparse monomials with exact rational coefficients and
//! checks every generator's Lie derivative. Finite `f64` literals are interpreted
//! as their exact IEEE-754 dyadic rationals. A zero residual is an exact algebraic
//! identity for that interpreted polynomial, not a Lean proof or a physical-units proof.
//!
//! Unsupported constructs and resource-limit outcomes fall through to the sampled
//! numeric lane, which preserves its seed, domain, finite-difference step and tolerances.
//! A sampled candidate is never promoted to exact or formal evidence.
//!
//! ## Exact checks for declared linear actions
//!
//! `check_exact_linear_action_invariance` is a separate verifier for an
//! already admitted `LinearGroupActionV1`. It computes the exact polynomial
//! Lie derivative for each supplied matrix generator, including non-standard
//! and conjugated coordinate representations. This verifies scalar-expression
//! invariance under the supplied infinitesimal generators; it does not discover
//! a group, establish Lie-algebra closure, or make a physical claim.
//!
//! ## Sampled numeric Lie-derivative fallback
//!
//! The sampled lane can detect radial expressions such as square roots and reciprocal
//! functions that are outside the polynomial fragment, but it remains finite evidence.
//!
//! The two structural heuristics above are exact-shape pattern matches — fast, but blind
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
//! - SO(k) actions on deterministic variable subsets of the first n variables
//!   (n ≤ 6), searched from largest k to smallest; and
//! - optionally, SO(2) actions involving variables beyond that bounded prefix.
//!
//! The subset budget and ordering are explicit in the profile. This can discover
//! a maximal SO(3) subgroup embedded in a larger expression with unacted
//! parameters, rather than reporting only the first SO(2) pair. It is still a
//! bounded search over standard coordinate-rotation representations, not an
//! exhaustive search over arbitrary Lie groups or nonlinear group actions.
//!
//! ## Why the structural heuristics still run first
//!
//! They're exact and essentially free; the numeric fallback only runs when
//! they don't fire, keeping the common cases fast and precise.

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_core::hdc::lie_theory::{self, LieAlgebra};

use crate::linear_group_action::AdmittedLinearGroupActionV1;
use crate::types::{LieGroup, SymmetryDescriptor};

/// Fully specified numerical policy used by the bounded Lie-derivative fallback.
///
/// The profile is returned with every numerical candidate so a consumer can
/// distinguish a sampled heuristic from a symbolic identity or formal proof.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericLieDerivativeProfileV2 {
    pub seed: u64,
    pub samples_per_generator: usize,
    pub finite_difference_step: f64,
    /// Maximum squared cosine between the expression gradient and group-orbit tangent.
    /// Dimensionless and invariant to nonzero expression and uniform coordinate rescaling.
    pub squared_cosine_tolerance: f64,
    pub sample_min_inclusive: f64,
    pub sample_max_exclusive: f64,
    pub maximum_joint_dimension: usize,
    /// Maximum variable subsets considered for joint SO(k) actions.
    pub maximum_group_subsets: usize,
    pub pairwise_so2_search: bool,
}

impl Default for NumericLieDerivativeProfileV2 {
    fn default() -> Self {
        Self {
            seed: 0x9E3779B97F4A7C15,
            samples_per_generator: 8,
            finite_difference_step: 1e-5,
            squared_cosine_tolerance: 1e-6,
            sample_min_inclusive: 0.5,
            sample_max_exclusive: 3.0,
            maximum_joint_dimension: 6,
            maximum_group_subsets: 64,
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
    /// Maximum exact polynomial term operations. The declared-linear-action
    /// checker consumes this budget cumulatively across all generators.
    pub maximum_term_products: usize,
    pub maximum_total_degree: usize,
    pub maximum_power: u32,
    pub maximum_dyadic_denominator_power: u32,
    /// Maximum variable subsets considered for exact SO(k) actions.
    pub maximum_group_subsets: usize,
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
            maximum_group_subsets: 64,
            pairwise_so2_search: true,
        }
    }
}

/// Typed failure from exact polynomial checking against an admitted linear action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearActionPolynomialErrorV1 {
    InvalidProfile,
    UnsupportedExpression,
    ResourceLimitExceeded,
}

/// Exact Lie-derivative result for one declared infinitesimal generator.
///
/// A residual term count of zero means the exact polynomial derivative is
/// identically zero under the declared IEEE-754-dyadic coefficient policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearActionGeneratorLieResultV1 {
    pub generator_id: String,
    pub residual_term_count: usize,
}

/// Exact polynomial invariance report for a supplied, metric-admitted linear action.
///
/// This qualifies invariance under each declared infinitesimal generator. It does
/// not prove that the generators are independent, form a closed Lie algebra,
/// integrate to a globally defined subgroup, or describe a physical symmetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionPolynomialReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub coordinate_names: Vec<String>,
    /// Variables present in the expression, followed by any declared action
    /// coordinates absent from the expression.
    pub expression_variables: Vec<String>,
    /// True when the parsed expression is constant in all included variables.
    pub constant_expression: bool,
    pub profile: ExactPolynomialLieProfileV1,
    pub generator_results: Vec<LinearActionGeneratorLieResultV1>,
}

impl LinearActionPolynomialReportV1 {
    /// Whether the exact Lie derivative is zero for every declared generator.
    pub fn invariant_under_every_generator(&self) -> bool {
        !self.generator_results.is_empty()
            && self
                .generator_results
                .iter()
                .all(|result| result.residual_term_count == 0)
    }
}

/// Failure from exact polynomial equivariance checks for a declared linear action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearActionVectorFieldErrorV1 {
    InvalidProfile,
    ComponentCountMismatch { expected: usize, actual: usize },
    UnsupportedExpression,
    ResourceLimitExceeded,
}

/// Exact component-wise infinitesimal equivariance result for one generator.
///
/// Zero residual terms in every component establish
/// D F(x) A x - A F(x) = 0 as an exact polynomial identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearActionVectorFieldGeneratorResultV1 {
    pub generator_id: String,
    pub component_residual_term_counts: Vec<usize>,
}

/// Exact polynomial vector-field equivariance report for a supplied action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionVectorFieldEquivarianceReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub coordinate_names: Vec<String>,
    pub expression_variables: Vec<String>,
    pub profile: ExactPolynomialLieProfileV1,
    pub generator_results: Vec<LinearActionVectorFieldGeneratorResultV1>,
}

impl LinearActionVectorFieldEquivarianceReportV1 {
    /// True only when every component residual is identically zero for every generator.
    pub fn equivariant_under_every_generator(&self) -> bool {
        !self.generator_results.is_empty()
            && self.generator_results.iter().all(|generator| {
                generator
                    .component_residual_term_counts
                    .iter()
                    .all(|term_count| *term_count == 0)
            })
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuralSymmetryPattern {
    /// A sum of distinct squared variables with one shared finite nonzero coefficient.
    EqualCoefficientSumOfSquares {
        variables: Vec<String>,
        shared_coefficient: f64,
    },
    /// A determinant invariant under one diagonal SO(2) action on two coordinate pairs.
    TwoDimensionalAntisymmetricBilinear {
        first_pair: [String; 2],
        second_pair: [String; 2],
    },
}

/// Observed diagnostics for an accepted sampled Lie-derivative candidate.
///
/// The maximum is taken over every informative generator/sample pair evaluated
/// by the declared profile. Each value is the squared cosine between the expression
/// gradient and the group-orbit tangent. It is not a confidence interval or a proof.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LieDerivativeSampleStatisticsV1 {
    pub sample_count_per_generator: usize,
    pub checked_generator_count: usize,
    pub informative_evaluation_count: usize,
    pub maximum_normalized_squared_residual: f64,
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
        profile: NumericLieDerivativeProfileV2,
        acted_on_variables: Vec<String>,
        group: LieGroup,
        statistics: LieDerivativeSampleStatisticsV1,
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
    /// The numerical subset search stopped before exhausting the candidate space.
    NumericSearchBudgetExceeded,
    /// The expression had fewer than two free variables, so numerical rotation search was skipped.
    NumericSearchSkippedTooFewVariables,
}

/// Mathematical object to which this report's symmetry claim applies.
///
/// The current inference entry point receives one scalar expression. Its result
/// is therefore only a claim that this expression is invariant under the
/// searched standard coordinate-rotation action. It is not a claim that a
/// vector field is equivariant, that the expression is conserved by an ODE,
/// or that a Lagrangian/Hamiltonian has a Noether symmetry or momentum map.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymmetryClaimScope {
    #[default]
    ScalarExpressionInvariant,
}

/// Auditable result of symmetry inference.///
/// A missing candidate means only that the recorded bounded search found no candidate;
/// it is not evidence that the expression has no symmetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymmetryInferenceReport {
    /// The precise mathematical object covered by this candidate.
    #[serde(default)]
    pub claim_scope: SymmetryClaimScope,
    pub candidate: Option<SymmetryDescriptor>,
    pub evidence: Option<SymmetryInferenceEvidence>,
    pub search_status: SymmetryInferenceSearchStatus,
    pub numeric_profile: NumericLieDerivativeProfileV2,
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
    let numeric_profile = NumericLieDerivativeProfileV2::default();
    let exact_polynomial_profile = ExactPolynomialLieProfileV1::default();

    if let Some((variables, shared_coefficient)) = detect_sum_of_squares(expr)
        && (2..=10).contains(&variables.len())
    {
        let dimension = variables.len() as u8;
        return SymmetryInferenceReport {
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
            candidate: Some(SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(dimension)], false)),
            evidence: Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::EqualCoefficientSumOfSquares {
                    variables,
                    shared_coefficient,
                },
            }),
            search_status: SymmetryInferenceSearchStatus::StructuralMatchShortCircuit,
            numeric_profile,
            exact_polynomial_profile,
            exact_polynomial_status: ExactPolynomialSearchStatus::NotRunStructuralShortcut,
        };
    }

    if let Some((first_pair, second_pair)) = detect_2d_antisymmetric_cross(expr) {
        return SymmetryInferenceReport {
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
            candidate: Some(SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false)),
            evidence: Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::TwoDimensionalAntisymmetricBilinear {
                    first_pair,
                    second_pair,
                },
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
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
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
                claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
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
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
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
        NumericLieDerivativeProfileV2::default(),
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
    profile: NumericLieDerivativeProfileV2,
    exact_polynomial_profile: ExactPolynomialLieProfileV1,
    exact_polynomial_status: ExactPolynomialSearchStatus,
) -> SymmetryInferenceReport {
    let mut var_names = Vec::new();
    collect_var_names(expr, &mut var_names);

    if var_names.len() < 2 {
        return SymmetryInferenceReport {
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
            candidate: None,
            evidence: None,
            search_status: SymmetryInferenceSearchStatus::NumericSearchSkippedTooFewVariables,
            numeric_profile: profile,
            exact_polynomial_profile,
            exact_polynomial_status,
        };
    }

    // Search deterministic variable subsets from largest to smallest. This finds
    // maximal SO(k) subgroups embedded in expressions with unacted parameters,
    // rather than downgrading every non-global result to the first SO(2) pair.
    let n = var_names.len().min(profile.maximum_joint_dimension);
    let subset_count = group_subset_count(n, profile.pairwise_so2_search);
    if subset_count > profile.maximum_group_subsets {
        return SymmetryInferenceReport {
            claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
            candidate: None,
            evidence: None,
            search_status: SymmetryInferenceSearchStatus::NumericSearchBudgetExceeded,
            numeric_profile: profile,
            exact_polynomial_profile,
            exact_polynomial_status,
        };
    }
    for subset_size in (2..=n).rev() {
        if subset_size == 2 && !profile.pairwise_so2_search {
            continue;
        }
        let algebra = lie_theory::so_n(subset_size);
        for active_indices in index_combinations(n, subset_size) {
            if let Some(statistics) = is_lie_invariant(expr, &var_names, &active_indices, &algebra, profile) {
                let group = LieGroup::SO(subset_size as u8);
                let acted_on_variables = active_indices.iter().map(|&i| var_names[i].clone()).collect();
                return SymmetryInferenceReport {
                    claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
                    candidate: Some(SymmetryDescriptor::from_lie_groups(vec![group], false)),
                    evidence: Some(SymmetryInferenceEvidence::SampledLieDerivative {
                        profile,
                        acted_on_variables,
                        group,
                        statistics,
                    }),
                    search_status: SymmetryInferenceSearchStatus::NumericSearchCompleted,
                    numeric_profile: profile,
                    exact_polynomial_profile,
                    exact_polynomial_status,
                };
            }
        }
    }

    // Preserve pairwise coverage beyond the bounded joint-search prefix, but
    // charge every additional pair against the same declared subset budget.
    if profile.pairwise_so2_search && var_names.len() > n {
        let so2 = lie_theory::so_n(2);
        let remaining_pair_budget = profile.maximum_group_subsets.saturating_sub(subset_count);
        let mut checked_extra_pairs = 0usize;
        for i in 0..var_names.len() {
            for j in (i + 1)..var_names.len() {
                if i < n && j < n {
                    continue;
                }
                if checked_extra_pairs >= remaining_pair_budget {
                    return SymmetryInferenceReport {
                        claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
                        candidate: None,
                        evidence: None,
                        search_status: SymmetryInferenceSearchStatus::NumericSearchBudgetExceeded,
                        numeric_profile: profile,
                        exact_polynomial_profile,
                        exact_polynomial_status,
                    };
                }
                checked_extra_pairs = checked_extra_pairs.saturating_add(1);
                if let Some(statistics) = is_lie_invariant(expr, &var_names, &[i, j], &so2, profile) {
                    let group = LieGroup::SO(2);
                    return SymmetryInferenceReport {
                        claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
                        candidate: Some(SymmetryDescriptor::from_lie_groups(vec![group], false)),
                        evidence: Some(SymmetryInferenceEvidence::SampledLieDerivative {
                            profile,
                            acted_on_variables: vec![var_names[i].clone(), var_names[j].clone()],
                            group,
                            statistics,
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
        claim_scope: SymmetryClaimScope::ScalarExpressionInvariant,
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

/// Number of group-action subsets in the bounded search (k >= 2, and optionally k = 2).
fn group_subset_count(dimension: usize, include_so2: bool) -> usize {
    if dimension < 2 {
        return 0;
    }
    let all_nonempty = (1usize << dimension).saturating_sub(1);
    let singletons = dimension;
    let pairs = dimension.saturating_mul(dimension.saturating_sub(1)) / 2;
    if include_so2 {
        all_nonempty.saturating_sub(singletons)
    } else {
        all_nonempty.saturating_sub(singletons).saturating_sub(pairs)
    }
}

/// Lexicographically ordered index combinations; caller chooses descending group size.
fn index_combinations(dimension: usize, subset_size: usize) -> Vec<Vec<usize>> {
    fn extend(
        dimension: usize,
        remaining: usize,
        next_index: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(current.clone());
            return;
        }
        let last_start = dimension.saturating_sub(remaining);
        for index in next_index..=last_start {
            current.push(index);
            extend(dimension, remaining - 1, index + 1, current, output);
            current.pop();
        }
    }

    let mut output = Vec::new();
    if subset_size <= dimension && subset_size >= 2 {
        extend(dimension, subset_size, 0, &mut Vec::new(), &mut output);
    }
    output
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
        // Use the least common denominator to avoid avoidable intermediate growth.
        let common = exact_gcd(self.denominator as u128, other.denominator as u128);
        let common = i128::try_from(common).ok()?;
        let left_scale = other.denominator / common;
        let right_scale = self.denominator / common;
        let left = self.numerator.checked_mul(left_scale)?;
        let right = other.numerator.checked_mul(right_scale)?;
        let numerator = left.checked_add(right)?;
        let denominator = self.denominator.checked_mul(left_scale)?;
        Self::new(numerator, denominator)
    }

    fn checked_mul(self, other: Self) -> Option<Self> {
        // Cross-cancel before multiplication to preserve the bounded exact domain.
        let cancel_left = exact_gcd(self.numerator.unsigned_abs(), other.denominator as u128);
        let cancel_right = exact_gcd(other.numerator.unsigned_abs(), self.denominator as u128);
        let cancel_left = i128::try_from(cancel_left).ok()?;
        let cancel_right = i128::try_from(cancel_right).ok()?;
        let left_numerator = self.numerator / cancel_left;
        let right_denominator = other.denominator / cancel_left;
        let right_numerator = other.numerator / cancel_right;
        let left_denominator = self.denominator / cancel_right;
        let numerator = left_numerator.checked_mul(right_numerator)?;
        let denominator = left_denominator.checked_mul(right_denominator)?;
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

fn polynomial_depends_on_variable(polynomial: &ExactPolynomial, variable_index: usize) -> bool {
    polynomial
        .keys()
        .any(|monomial| monomial.get(variable_index).is_some_and(|power| *power > 0))
}
fn charge_exact_polynomial_operations(
    operation_count: &mut usize,
    additional: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<(), ExactPolynomialError> {
    *operation_count = (*operation_count).saturating_add(additional);
    if *operation_count > profile.maximum_term_products {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    Ok(())
}

fn exact_polynomial_linear_action_lie_derivative(
    polynomial: &ExactPolynomial,
    variable_names: &[String],
    coordinate_names: &[String],
    generator: &crate::linear_group_action::LinearGeneratorV1,
    profile: ExactPolynomialLieProfileV1,
    operation_count: &mut usize,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut residual = ExactPolynomial::new();

    for (row_index, generator_row) in generator.matrix.iter().enumerate() {
        let row_variable = variable_names
            .iter()
            .position(|name| name == &coordinate_names[row_index])
            .ok_or(ExactPolynomialError::UnsupportedExpression)?;
        charge_exact_polynomial_operations(operation_count, polynomial.len(), profile)?;
        let derivative = polynomial_derivative(polynomial, row_variable, profile)?;

        for (column_index, matrix_entry) in generator_row.iter().enumerate() {
            let coefficient = ExactRational::from_f64(
                *matrix_entry,
                profile.maximum_dyadic_denominator_power,
            )
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
            if coefficient.is_zero() {
                continue;
            }
            let column_variable = variable_names
                .iter()
                .position(|name| name == &coordinate_names[column_index])
                .ok_or(ExactPolynomialError::UnsupportedExpression)?;
            charge_exact_polynomial_operations(operation_count, derivative.len(), profile)?;
            let term = polynomial_multiply_variable(&derivative, column_variable, profile)?;
            charge_exact_polynomial_operations(operation_count, term.len(), profile)?;
            for (monomial, term_coefficient) in &term {
                let scaled_coefficient = term_coefficient
                    .checked_mul(coefficient)
                    .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
                polynomial_add_term(
                    &mut residual,
                    monomial.clone(),
                    scaled_coefficient,
                    profile,
                )?;
            }
        }
    }

    Ok(residual)
}

/// Check exact polynomial invariance under every generator of an already admitted action.
///
/// The expression variables are collected in first-seen order. Declared action
/// coordinates not referenced by the expression are appended, so the matrix
/// representation and polynomial exponent vectors always share one explicit
/// variable ordering. Other expression variables remain fixed parameters.
pub fn check_exact_linear_action_invariance(
    expr: &Expr,
    action: &AdmittedLinearGroupActionV1,
    profile: ExactPolynomialLieProfileV1,
) -> Result<LinearActionPolynomialReportV1, LinearActionPolynomialErrorV1> {
    if profile.maximum_variables == 0
        || profile.maximum_terms == 0
        || profile.maximum_term_products == 0
        || profile.maximum_total_degree == 0
        || profile.maximum_power == 0
        || profile.maximum_dyadic_denominator_power > 126
    {
        return Err(LinearActionPolynomialErrorV1::InvalidProfile);
    }

    let declaration = action.action();
    let mut variable_names = Vec::new();
    collect_var_names(expr, &mut variable_names);
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    if variable_names.len() > profile.maximum_variables {
        return Err(LinearActionPolynomialErrorV1::ResourceLimitExceeded);
    }

    let polynomial = parse_exact_polynomial(expr, &variable_names, profile).map_err(|error| {
        match error {
            ExactPolynomialError::UnsupportedExpression => {
                LinearActionPolynomialErrorV1::UnsupportedExpression
            }
            ExactPolynomialError::ResourceLimitExceeded => {
                LinearActionPolynomialErrorV1::ResourceLimitExceeded
            }
        }
    })?;
    let constant_expression = polynomial
        .keys()
        .all(|monomial| monomial.iter().all(|exponent| *exponent == 0));

    let mut generator_results = Vec::with_capacity(declaration.generators.len());
    // One operation budget is shared across every coordinate and generator.
    let mut operation_count = 0usize;
    for generator in &declaration.generators {
        let residual = exact_polynomial_linear_action_lie_derivative(
            &polynomial,
            &variable_names,
            &declaration.coordinate_names,
            generator,
            profile,
            &mut operation_count,
        )
        .map_err(|error| match error {
            ExactPolynomialError::UnsupportedExpression => {
                LinearActionPolynomialErrorV1::UnsupportedExpression
            }
            ExactPolynomialError::ResourceLimitExceeded => {
                LinearActionPolynomialErrorV1::ResourceLimitExceeded
            }
        })?;
        generator_results.push(LinearActionGeneratorLieResultV1 {
            generator_id: generator.id.clone(),
            residual_term_count: residual.len(),
        });
    }

    Ok(LinearActionPolynomialReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        coordinate_names: declaration.coordinate_names.clone(),
        expression_variables: variable_names,
        constant_expression,
        profile,
        generator_results,
    })
}

fn exact_polynomial_vector_field_generator_residual(
    components: &[ExactPolynomial],
    variable_names: &[String],
    coordinate_names: &[String],
    generator: &crate::linear_group_action::LinearGeneratorV1,
    profile: ExactPolynomialLieProfileV1,
    operation_count: &mut usize,
) -> Result<Vec<ExactPolynomial>, ExactPolynomialError> {
    let mut component_residuals = Vec::with_capacity(coordinate_names.len());

    for (output_index, output_component) in components.iter().enumerate() {
        let mut residual = ExactPolynomial::new();

        // First term: D F(x) * (A x).
        for (action_row, coordinate_name) in coordinate_names.iter().enumerate() {
            let coordinate_index = variable_names
                .iter()
                .position(|name| name == coordinate_name)
                .ok_or(ExactPolynomialError::UnsupportedExpression)?;
            charge_exact_polynomial_operations(operation_count, output_component.len(), profile)?;
            let derivative =
                polynomial_derivative(output_component, coordinate_index, profile)?;

            for (state_column, matrix_entry) in
                generator.matrix[action_row].iter().enumerate()
            {
                let coefficient = ExactRational::from_f64(
                    *matrix_entry,
                    profile.maximum_dyadic_denominator_power,
                )
                .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
                if coefficient.is_zero() {
                    continue;
                }

                let column_variable = variable_names
                    .iter()
                    .position(|name| name == &coordinate_names[state_column])
                    .ok_or(ExactPolynomialError::UnsupportedExpression)?;
                let term =
                    polynomial_multiply_variable(&derivative, column_variable, profile)?;
                charge_exact_polynomial_operations(operation_count, term.len(), profile)?;
                for (monomial, term_coefficient) in &term {
                    let scaled_coefficient = term_coefficient
                        .checked_mul(coefficient)
                        .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
                    polynomial_add_term(
                        &mut residual,
                        monomial.clone(),
                        scaled_coefficient,
                        profile,
                    )?;
                }
            }
        }

        // Second term: subtract (A F(x))_output_index.
        for (field_index, matrix_entry) in
            generator.matrix[output_index].iter().enumerate()
        {
            let coefficient = ExactRational::from_f64(
                *matrix_entry,
                profile.maximum_dyadic_denominator_power,
            )
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
            if coefficient.is_zero() {
                continue;
            }
            charge_exact_polynomial_operations(
                operation_count,
                components[field_index].len(),
                profile,
            )?;
            for (monomial, term_coefficient) in &components[field_index] {
                let scaled_coefficient = term_coefficient
                    .checked_mul(coefficient)
                    .and_then(ExactRational::checked_neg)
                    .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
                polynomial_add_term(
                    &mut residual,
                    monomial.clone(),
                    scaled_coefficient,
                    profile,
                )?;
            }
        }

        component_residuals.push(residual);
    }

    Ok(component_residuals)
}

/// Check exact polynomial vector-field equivariance under every admitted generator.
///
/// For a linear action generated by A, the exact infinitesimal condition is
/// D F(x) A x - A F(x) = 0 component-wise. Vector-field components are ordered
/// exactly as the action's coordinate_names. Additional expression variables
/// are held fixed as parameters.
pub fn check_exact_linear_action_vector_field_equivariance(
    vector_field_components: &[Expr],
    action: &AdmittedLinearGroupActionV1,
    profile: ExactPolynomialLieProfileV1,
) -> Result<
    LinearActionVectorFieldEquivarianceReportV1,
    LinearActionVectorFieldErrorV1,
> {
    if profile.maximum_variables == 0
        || profile.maximum_terms == 0
        || profile.maximum_term_products == 0
        || profile.maximum_total_degree == 0
        || profile.maximum_power == 0
        || profile.maximum_dyadic_denominator_power > 126
    {
        return Err(LinearActionVectorFieldErrorV1::InvalidProfile);
    }

    let declaration = action.action();
    let dimension = declaration.coordinate_names.len();
    if vector_field_components.len() != dimension {
        return Err(LinearActionVectorFieldErrorV1::ComponentCountMismatch {
            expected: dimension,
            actual: vector_field_components.len(),
        });
    }

    let mut variable_names = Vec::new();
    for component in vector_field_components {
        collect_var_names(component, &mut variable_names);
    }
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    if variable_names.len() > profile.maximum_variables {
        return Err(LinearActionVectorFieldErrorV1::ResourceLimitExceeded);
    }

    let mut operation_count = 0usize;
    let mut components = Vec::with_capacity(vector_field_components.len());
    for component in vector_field_components {
        let polynomial = parse_exact_polynomial(component, &variable_names, profile).map_err(
            |error| match error {
                ExactPolynomialError::UnsupportedExpression => {
                    LinearActionVectorFieldErrorV1::UnsupportedExpression
                }
                ExactPolynomialError::ResourceLimitExceeded => {
                    LinearActionVectorFieldErrorV1::ResourceLimitExceeded
                }
            },
        )?;
        charge_exact_polynomial_operations(&mut operation_count, polynomial.len(), profile)
            .map_err(|_| LinearActionVectorFieldErrorV1::ResourceLimitExceeded)?;
        components.push(polynomial);
    }

    let mut generator_results = Vec::with_capacity(declaration.generators.len());
    for generator in &declaration.generators {
        let component_residuals = exact_polynomial_vector_field_generator_residual(
            &components,
            &variable_names,
            &declaration.coordinate_names,
            generator,
            profile,
            &mut operation_count,
        )
        .map_err(|error| match error {
            ExactPolynomialError::UnsupportedExpression => {
                LinearActionVectorFieldErrorV1::UnsupportedExpression
            }
            ExactPolynomialError::ResourceLimitExceeded => {
                LinearActionVectorFieldErrorV1::ResourceLimitExceeded
            }
        })?;
        generator_results.push(LinearActionVectorFieldGeneratorResultV1 {
            generator_id: generator.id.clone(),
            component_residual_term_counts: component_residuals
                .iter()
                .map(ExactPolynomial::len)
                .collect(),
        });
    }

    Ok(LinearActionVectorFieldEquivarianceReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        coordinate_names: declaration.coordinate_names.clone(),
        expression_variables: variable_names,
        profile,
        generator_results,
    })
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
        return Ok(None);
    }
    let dimension = variable_names.len();
    let subset_count = group_subset_count(dimension, profile.pairwise_so2_search);
    if subset_count > profile.maximum_group_subsets {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }

    // Maximal subgroup first: an expression may be SO(3)-invariant in
    // (x,y,z) while a fourth variable acts as a parameter or independent term.
    for subset_size in (2..=dimension).rev() {
        if subset_size == 2 && !profile.pairwise_so2_search {
            continue;
        }
        for active_indices in index_combinations(dimension, subset_size) {
            if !active_indices.iter().all(|&index| polynomial_depends_on_variable(&polynomial, index)) {
                continue;
            }

            let mut all_generators_invariant = true;
            for a in 0..active_indices.len() {
                for b in (a + 1)..active_indices.len() {
                    if !exact_polynomial_lie_derivative_is_zero(
                        &polynomial,
                        active_indices[a],
                        active_indices[b],
                        profile,
                    )? {
                        all_generators_invariant = false;
                        break;
                    }
                }
                if !all_generators_invariant {
                    break;
                }
            }

            if all_generators_invariant {
                let variables = active_indices.iter().map(|&index| variable_names[index].clone()).collect();
                let generator_count = subset_size * (subset_size - 1) / 2;
                return Ok(Some((LieGroup::SO(subset_size as u8), variables, generator_count)));
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
    profile: NumericLieDerivativeProfileV2,
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
fn sample_states(dim: usize, profile: NumericLieDerivativeProfileV2) -> Vec<Vec<f64>> {
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
/// be below tolerance at every sampled point, for every generator. The residual
/// is the squared cosine between the expression gradient and group-orbit tangent.
/// Unlike |L_A f|² / ||grad f||², it is dimensionless and does not depend on the
/// magnitude/units of the orbit vector. Samples with a zero gradient or zero
/// orbit tangent are uninformative and are skipped.
fn is_lie_invariant(
    expr: &Expr,
    var_names: &[String],
    active_indices: &[usize],
    algebra: &LieAlgebra,
    profile: NumericLieDerivativeProfileV2,
) -> Option<LieDerivativeSampleStatisticsV1> {
    let samples = sample_states(var_names.len(), profile);

    let mut informative_evaluation_count = 0usize;
    let mut maximum_normalized_squared_residual = 0.0_f64;
    for generator in &algebra.basis {
        for state in &samples {
            let grad = fd_gradient(expr, state, var_names, profile);
            if grad.iter().any(|g| !g.is_finite()) {
                return None;
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

            // Normalize both vectors before taking their dot product. The prior
            // |L_A f|² / ||grad f||² metric equals ||A x||² cos²(theta), so its
            // threshold depends on coordinate magnitude/units. hypot keeps norm
            // accumulation stable; the normalized dot product avoids forming an
            // unscaled Lie derivative or multiplying the two norms.
            let gradient_norm = active_indices
                .iter()
                .fold(0.0_f64, |norm, &vi| norm.hypot(grad[vi]));
            let flow_norm = active_indices
                .iter()
                .fold(0.0_f64, |norm, &vi| norm.hypot(flow[vi]));
            if !gradient_norm.is_finite() || !flow_norm.is_finite() {
                return None;
            }
            if gradient_norm == 0.0 || flow_norm == 0.0 {
                continue;
            }

            let normalized_directional_derivative: f64 = active_indices
                .iter()
                .map(|&vi| (grad[vi] / gradient_norm) * (flow[vi] / flow_norm))
                .sum();
            if !normalized_directional_derivative.is_finite() {
                return None;
            }
            let normalized_squared_residual =
                normalized_directional_derivative * normalized_directional_derivative;
            if !normalized_squared_residual.is_finite()
                || normalized_squared_residual > profile.squared_cosine_tolerance
            {
                return None;
            }

            informative_evaluation_count = informative_evaluation_count.saturating_add(1);
            maximum_normalized_squared_residual =
                maximum_normalized_squared_residual.max(normalized_squared_residual);
        }
    }

    (informative_evaluation_count > 0).then_some(LieDerivativeSampleStatisticsV1 {
        sample_count_per_generator: samples.len(),
        checked_generator_count: algebra.basis.len(),
        informative_evaluation_count,
        maximum_normalized_squared_residual,
    })
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
fn detect_sum_of_squares(expr: &Expr) -> Option<(Vec<String>, f64)> {
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
    Some((seen, common_coefficient?))
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
fn detect_2d_antisymmetric_cross(expr: &Expr) -> Option<([String; 2], [String; 2])> {
    let Expr::BinOp(BinOp::Sub, l, r) = expr else {
        return None;
    };
    let Expr::BinOp(BinOp::Mul, l1, l2) = l.as_ref() else {
        return None;
    };
    let Expr::BinOp(BinOp::Mul, r1, r2) = r.as_ref() else {
        return None;
    };
    let (a, b, c, d) = match (l1.as_ref(), l2.as_ref(), r1.as_ref(), r2.as_ref()) {
        (Expr::Var(a), Expr::Var(b), Expr::Var(c), Expr::Var(d)) => (a, b, c, d),
        _ => return None,
    };
    // Interpret a*b-c*d as det([[a,d],[c,b]]). Left SO(2) multiplication
    // rotates (a,c) and (d,b) by the same angle, preserving the determinant.
    let all = [a, b, c, d];
    let unique_count = all
        .iter()
        .enumerate()
        .filter(|(i, name)| !all[..*i].contains(name))
        .count();
    if unique_count != 4 {
        return None;
    }
    Some(([a.clone(), c.clone()], [d.clone(), b.clone()]))
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

    fn admitted_so2_action() -> AdmittedLinearGroupActionV1 {
        crate::linear_group_action::LinearGroupActionV1 {
            action_id: "exact-so2".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: vec![crate::linear_group_action::LinearGeneratorV1 {
                id: "J".to_string(),
                matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            }],
        }
        .try_admit(crate::linear_group_action::LinearGroupActionAdmissionProfileV1::default())
        .unwrap()
    }

    fn sheared_so2_action() -> AdmittedLinearGroupActionV1 {
        crate::linear_group_action::LinearGroupActionV1 {
            action_id: "exact-so2-sheared".to_string(),
            coordinate_basis_id: "XY".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, -1.0], vec![-1.0, 2.0]],
            generators: vec![crate::linear_group_action::LinearGeneratorV1 {
                id: "conjugated-J".to_string(),
                matrix: vec![vec![1.0, -2.0], vec![1.0, -1.0]],
            }],
        }
        .try_admit(crate::linear_group_action::LinearGroupActionAdmissionProfileV1::default())
        .unwrap()
    }

    #[test]
    fn exact_vector_field_equivariance_accepts_radial_damping() {
        let vector_field = vec![
            mul(Expr::Const(-1.0), var("x")),
            mul(Expr::Const(-1.0), var("y")),
        ];
        let report = check_exact_linear_action_vector_field_equivariance(
            &vector_field,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(
            report.generator_results[0].component_residual_term_counts,
            vec![0, 0]
        );
    }

    #[test]
    fn exact_vector_field_equivariance_accepts_the_rotation_generator_field() {
        let vector_field = vec![mul(Expr::Const(-1.0), var("y")), var("x")];
        let report = check_exact_linear_action_vector_field_equivariance(
            &vector_field,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(
            report.generator_results[0].component_residual_term_counts,
            vec![0, 0]
        );
    }

    #[test]
    fn exact_vector_field_equivariance_rejects_anisotropic_dynamics() {
        let vector_field = vec![
            mul(Expr::Const(-1.0), var("x")),
            mul(Expr::Const(-2.0), var("y")),
        ];
        let report = check_exact_linear_action_vector_field_equivariance(
            &vector_field,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(!report.equivariant_under_every_generator());
        assert!(
            report.generator_results[0]
                .component_residual_term_counts
                .iter()
                .all(|term_count| *term_count > 0)
        );
    }

    #[test]
    fn exact_vector_field_equivariance_is_covariant_under_sheared_coordinates() {
        // F(x,y)=Jx transforms as F'(X,Y)=D J D^-1 [X,Y]^T.
        let vector_field = vec![
            sub(var("X"), mul(Expr::Const(2.0), var("Y"))),
            sub(var("X"), var("Y")),
        ];
        let report = check_exact_linear_action_vector_field_equivariance(
            &vector_field,
            &sheared_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(
            report.generator_results[0].component_residual_term_counts,
            vec![0, 0]
        );
    }

    #[test]
    fn exact_vector_field_equivariance_keeps_unacted_parameters_fixed() {
        // F(x,y;z) = z J[x,y], with z a fixed parameter under the action.
        let vector_field = vec![
            mul(Expr::Const(-1.0), mul(var("z"), var("y"))),
            mul(var("z"), var("x")),
        ];
        let report = check_exact_linear_action_vector_field_equivariance(
            &vector_field,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(
            report.expression_variables,
            vec!["z".to_string(), "y".to_string(), "x".to_string()]
        );
    }

    #[test]
    fn exact_vector_field_equivariance_fails_closed_on_bad_shapes_and_limits() {
        let too_few_components = vec![var("x")];
        assert_eq!(
            check_exact_linear_action_vector_field_equivariance(
                &too_few_components,
                &admitted_so2_action(),
                ExactPolynomialLieProfileV1::default()
            ),
            Err(LinearActionVectorFieldErrorV1::ComponentCountMismatch {
                expected: 2,
                actual: 1
            })
        );

        let unsupported = vec![
            Expr::Func(UnaryFn::Sqrt, Box::new(pow2("x"))),
            var("y"),
        ];
        assert_eq!(
            check_exact_linear_action_vector_field_equivariance(
                &unsupported,
                &admitted_so2_action(),
                ExactPolynomialLieProfileV1::default()
            ),
            Err(LinearActionVectorFieldErrorV1::UnsupportedExpression)
        );

        let vector_field = vec![var("x"), var("y")];
        let mut bounded = ExactPolynomialLieProfileV1::default();
        bounded.maximum_term_products = 1;
        assert_eq!(
            check_exact_linear_action_vector_field_equivariance(
                &vector_field,
                &admitted_so2_action(),
                bounded
            ),
            Err(LinearActionVectorFieldErrorV1::ResourceLimitExceeded)
        );
    }

    #[test]
    fn exact_linear_action_checker_proves_radial_polynomial_invariance() {
        let expression = add(pow2("x"), pow2("y"));
        let report = check_exact_linear_action_invariance(
            &expression,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.invariant_under_every_generator());
        assert!(!report.constant_expression);
        assert_eq!(report.generator_results.len(), 1);
        assert_eq!(report.generator_results[0].generator_id, "J");
        assert_eq!(report.generator_results[0].residual_term_count, 0);
        assert_eq!(report.expression_variables, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn exact_linear_action_checker_rejects_anisotropic_polynomial() {
        let expression = add(pow2("x"), mul(Expr::Const(2.0), pow2("y")));
        let report = check_exact_linear_action_invariance(
            &expression,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(!report.invariant_under_every_generator());
        assert_eq!(report.generator_results[0].residual_term_count, 1);
    }

    #[test]
    fn exact_linear_action_checker_handles_conjugated_generator_and_metric() {
        // D = [[1, 1], [0, 1]], A' = D J D^-1, and G' = D^-T D^-1.
        // f(X,Y) = [X,Y] G' [X,Y]^T is exactly invariant under A'.
        let action = crate::linear_group_action::LinearGroupActionV1 {
            action_id: "exact-so2-sheared".to_string(),
            coordinate_basis_id: "XY".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, -1.0], vec![-1.0, 2.0]],
            generators: vec![crate::linear_group_action::LinearGeneratorV1 {
                id: "conjugated-J".to_string(),
                matrix: vec![vec![1.0, -2.0], vec![1.0, -1.0]],
            }],
        }
        .try_admit(crate::linear_group_action::LinearGroupActionAdmissionProfileV1::default())
        .unwrap();

        let expression = sub(
            add(pow2("X"), mul(Expr::Const(2.0), pow2("Y"))),
            mul(Expr::Const(2.0), mul(var("X"), var("Y"))),
        );
        let report = check_exact_linear_action_invariance(
            &expression,
            &action,
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.invariant_under_every_generator());
        assert_eq!(report.generator_results[0].residual_term_count, 0);
    }

    #[test]
    fn exact_linear_action_checker_keeps_unacted_variables_as_parameters() {
        let expression = mul(var("z"), add(pow2("x"), pow2("y")));
        let report = check_exact_linear_action_invariance(
            &expression,
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.invariant_under_every_generator());
        assert_eq!(
            report.expression_variables,
            vec!["z".to_string(), "x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn exact_linear_action_checker_distinguishes_constant_expressions() {
        let report = check_exact_linear_action_invariance(
            &Expr::Const(3.0),
            &admitted_so2_action(),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.constant_expression);
        assert!(report.invariant_under_every_generator());
    }

    #[test]
    fn exact_linear_action_checker_reports_unsupported_and_resource_limited_inputs() {
        let unsupported = sqrt_of(add(pow2("x"), pow2("y")));
        assert_eq!(
            check_exact_linear_action_invariance(
                &unsupported,
                &admitted_so2_action(),
                ExactPolynomialLieProfileV1::default(),
            ),
            Err(LinearActionPolynomialErrorV1::UnsupportedExpression)
        );

        let expression = add(pow2("x"), pow2("y"));
        let mut bounded = ExactPolynomialLieProfileV1::default();
        bounded.maximum_variables = 1;
        assert_eq!(
            check_exact_linear_action_invariance(&expression, &admitted_so2_action(), bounded),
            Err(LinearActionPolynomialErrorV1::ResourceLimitExceeded)
        );

        let mut invalid = ExactPolynomialLieProfileV1::default();
        invalid.maximum_terms = 0;
        assert_eq!(
            check_exact_linear_action_invariance(
                &expression,
                &admitted_so2_action(),
                invalid
            ),
            Err(LinearActionPolynomialErrorV1::InvalidProfile)
        );

        let mut operation_limited = ExactPolynomialLieProfileV1::default();
        operation_limited.maximum_term_products = 1;
        assert_eq!(
            check_exact_linear_action_invariance(
                &expression,
                &admitted_so2_action(),
                operation_limited
            ),
            Err(LinearActionPolynomialErrorV1::ResourceLimitExceeded)
        );

        let action_with_two_generators = crate::linear_group_action::LinearGroupActionV1 {
            action_id: "exact-so2-two-generator-budget".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: vec![
                crate::linear_group_action::LinearGeneratorV1 {
                    id: "J".to_string(),
                    matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
                },
                crate::linear_group_action::LinearGeneratorV1 {
                    id: "2J".to_string(),
                    matrix: vec![vec![0.0, -2.0], vec![2.0, 0.0]],
                },
            ],
        }
        .try_admit(crate::linear_group_action::LinearGroupActionAdmissionProfileV1::default())
        .unwrap();
        let mut cumulative_limited = ExactPolynomialLieProfileV1::default();
        cumulative_limited.maximum_term_products = 15;
        assert_eq!(
            check_exact_linear_action_invariance(
                &expression,
                &action_with_two_generators,
                cumulative_limited
            ),
            Err(LinearActionPolynomialErrorV1::ResourceLimitExceeded)
        );
    }

    #[test]
    fn structural_symmetry_report_preserves_pattern_provenance() {
        let expr = add(pow2("x"), pow2("y"));
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(report.claim_scope, SymmetryClaimScope::ScalarExpressionInvariant);
        assert_eq!(
            report.candidate.unwrap().lie_groups,
            vec![LieGroup::SO(2)]
        );
        assert_eq!(
            report.evidence,
            Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::EqualCoefficientSumOfSquares {
                    variables: vec!["x".to_string(), "y".to_string()],
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
        assert_eq!(report.claim_scope, SymmetryClaimScope::ScalarExpressionInvariant);
        assert_eq!(report.candidate.unwrap().lie_groups, vec![LieGroup::SO(2)]);
        let Some(SymmetryInferenceEvidence::SampledLieDerivative {
            profile,
            group,
            acted_on_variables,
            statistics,
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
        assert_eq!(statistics.sample_count_per_generator, 8);
        assert_eq!(statistics.checked_generator_count, 1);
        assert_eq!(statistics.informative_evaluation_count, 8);
        assert!(statistics.maximum_normalized_squared_residual <= profile.squared_cosine_tolerance);
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::NumericSearchCompleted
        );
    }

    fn numeric_profile_with_coordinate_scale(scale: f64) -> NumericLieDerivativeProfileV2 {
        let mut profile = NumericLieDerivativeProfileV2::default();
        profile.sample_min_inclusive *= scale;
        profile.sample_max_exclusive *= scale;
        profile.finite_difference_step *= scale;
        profile
    }

    #[test]
    fn sampled_symmetry_is_robust_to_expression_and_coordinate_scales() {
        // The numeric lane must not abstain merely because a valid expression
        // is represented at a very small or large amplitude. The old absolute
        // minimum-gradient cutoff rejected both extremes.
        let radial = sqrt_of(add(pow2("x"), pow2("y")));
        for amplitude in [5e-200, 5e200] {
            let scaled = mul(Expr::Const(amplitude), radial.clone());
            let report = infer_symmetry_numeric_with_evidence(&scaled);
            assert_eq!(
                report.candidate.as_ref().map(|candidate| candidate.lie_groups.clone()),
                Some(vec![LieGroup::SO(2)]),
                "radial symmetry should survive expression amplitude {amplitude}"
            );
        }

        // Coordinate units also matter. Let X = c*x, Y = c*y and express the
        // same anisotropic function as sqrt((X² + 2Y²) / c²). With c=1e-4,
        // the old residual ||A X||² cos²(theta) falls below 1e-6 on the frozen
        // sample box and can falsely accept SO(2), despite unchanged geometry.
        // The dimensionless squared-cosine residual remains above threshold.
        let anisotropic = add(pow2("x"), mul(Expr::Const(2.0), pow2("y")));
        let baseline = infer_symmetry_numeric_with_evidence(&sqrt_of(anisotropic.clone()));
        assert!(baseline.candidate.is_none());

        let coordinate_scale = 1e-4;
        let inverse_scale_squared = 1.0 / (coordinate_scale * coordinate_scale);
        let scaled_coordinates_expr = sqrt_of(mul(
            Expr::Const(inverse_scale_squared),
            anisotropic,
        ));
        let scaled_profile = numeric_profile_with_coordinate_scale(coordinate_scale);
        let scaled_report = infer_symmetry_numeric_with_profile(
            &scaled_coordinates_expr,
            scaled_profile,
            ExactPolynomialLieProfileV1::default(),
            ExactPolynomialSearchStatus::NotRequested,
        );
        assert!(scaled_report.candidate.is_none());
        assert!(scaled_report.evidence.is_none());
        assert_eq!(
            scaled_report.search_status,
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
        let Some(SymmetryInferenceEvidence::SampledLieDerivative { statistics, profile, .. }) = report.evidence else {
            panic!("near-isotropic candidate must remain sampled evidence");
        };
        assert!(statistics.maximum_normalized_squared_residual > 0.0);
        assert!(statistics.maximum_normalized_squared_residual <= profile.squared_cosine_tolerance);
        assert_eq!(statistics.sample_count_per_generator, profile.samples_per_generator);
        assert_eq!(
            report.exact_polynomial_status,
            ExactPolynomialSearchStatus::UnsupportedExpression
        );
    }
    #[test]
    fn quartic_radial_polynomial_has_exact_so2_lie_derivative_identity() {
        let radial_squared = add(pow2("x"), pow2("y"));
        let quartic = Expr::BinOp(
            BinOp::Pow,
            Box::new(radial_squared),
            Box::new(Expr::Const(2.0)),
        );
        // A dyadic coefficient tests the exact rational interpretation of f64 literals.
        let expr = mul(Expr::Const(0.5), quartic);
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(report.claim_scope, SymmetryClaimScope::ScalarExpressionInvariant);
        assert_eq!(
            report.candidate.as_ref().unwrap().lie_groups,
            vec![LieGroup::SO(2)]
        );
        let Some(SymmetryInferenceEvidence::ExactPolynomialLieDerivative {
            profile,
            acted_on_variables,
            group,
            checked_generator_count,
        }) = report.evidence
        else {
            panic!("supported radial polynomial should produce exact symbolic evidence");
        };
        assert_eq!(group, LieGroup::SO(2));
        assert_eq!(acted_on_variables, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(checked_generator_count, 1);
        assert_eq!(profile.maximum_terms, 4096);
        assert_eq!(profile.maximum_term_products, 65_536);
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::ExactPolynomialIdentityFound
        );
        assert_eq!(report.exact_polynomial_status, ExactPolynomialSearchStatus::IdentityFound);
    }

    #[test]
    fn exact_polynomial_search_finds_pairwise_rotation_with_an_unacted_parameter() {
        // z*(x²+y²) is invariant under SO(2) acting on x,y while z remains fixed.
        let expr = mul(
            var("z"),
            add(pow2("x"), pow2("y")),
        );
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(
            report.candidate.as_ref().unwrap().lie_groups,
            vec![LieGroup::SO(2)]
        );
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::ExactPolynomialLieDerivative {
                acted_on_variables,
                checked_generator_count: 1,
                ..
            }) if acted_on_variables == vec!["x".to_string(), "y".to_string()]
        ));
    }

    #[test]
    fn exact_search_finds_maximal_so3_subgroup_on_nonleading_coordinates() {
        // z is an unacted parameter; the SO(3) action is on x,y,w, not the first three AST variables.
        let radial = add(add(pow2("x"), pow2("y")), pow2("w"));
        let expr = mul(var("z"), radial);
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(
            report.candidate.as_ref().unwrap().lie_groups,
            vec![LieGroup::SO(3)]
        );
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::ExactPolynomialLieDerivative {
                acted_on_variables,
                checked_generator_count: 3,
                group: LieGroup::SO(3),
                ..
            }) if acted_on_variables == vec!["x".to_string(), "y".to_string(), "w".to_string()]
        ));
    }

    #[test]
    fn sampled_search_finds_maximal_so3_subgroup_on_nonleading_coordinates() {
        // sqrt(z*(x²+y²+w²)) is radially invariant in x,y,w for fixed z>0.
        let radial = add(add(pow2("x"), pow2("y")), pow2("w"));
        let expr = sqrt_of(mul(var("z"), radial));
        let report = infer_symmetry_numeric_with_evidence(&expr);
        assert_eq!(
            report.candidate.as_ref().unwrap().lie_groups,
            vec![LieGroup::SO(3)]
        );
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::SampledLieDerivative {
                acted_on_variables,
                group: LieGroup::SO(3),
                statistics: LieDerivativeSampleStatisticsV1 { checked_generator_count: 3, .. },
                ..
            }) if acted_on_variables == vec!["x".to_string(), "y".to_string(), "w".to_string()]
        ));
    }

    #[test]
    fn exact_polynomial_search_rejects_anisotropic_quadratic_with_no_numeric_promotion() {
        let expr = add(
            pow2("x"),
            mul(Expr::Const(1.00001), pow2("y")),
        );
        let report = infer_symmetry_with_evidence(&expr);
        assert!(report.candidate.is_none());
        assert!(report.evidence.is_none());
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::ExactPolynomialSearchCompletedNoCandidate
        );
        assert_eq!(report.exact_polynomial_status, ExactPolynomialSearchStatus::CompletedNoCandidate);
    }

    #[test]
    fn exact_polynomial_search_does_not_infer_rotation_on_unused_variables() {
        // The source AST mentions x and y, but both disappear after multiplying by zero.
        let expr = add(
            var("z"),
            add(
                mul(Expr::Const(0.0), var("x")),
                mul(Expr::Const(0.0), var("y")),
            ),
        );
        let report = infer_symmetry_with_evidence(&expr);
        assert!(report.candidate.is_none());
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::ExactPolynomialSearchCompletedNoCandidate
        );
    }
    #[test]
    fn exact_polynomial_search_uses_full_so3_generator_basis() {
        let radial_squared = add(add(pow2("x"), pow2("y")), pow2("z"));
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(radial_squared),
            Box::new(Expr::Const(2.0)),
        );
        let report = infer_symmetry_with_evidence(&expr);
        assert_eq!(report.candidate.unwrap().lie_groups, vec![LieGroup::SO(3)]);
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::ExactPolynomialLieDerivative {
                checked_generator_count: 3,
                group: LieGroup::SO(3),
                ..
            })
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
        assert_eq!(detect_sum_of_squares(&expr), Some((vec!["x".to_string(), "y".to_string()], 2.5)));
        assert_eq!(infer_symmetry(&expr).lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn negative_common_quadratic_weight_preserves_rotational_symmetry() {
        let expr = add(
            mul(Expr::Const(-3.0), pow2("x")),
            mul(Expr::Const(-3.0), pow2("y")),
        );
        assert_eq!(detect_sum_of_squares(&expr), Some((vec!["x".to_string(), "y".to_string()], -3.0)));
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
    fn antisymmetric_bilinear_report_binds_the_diagonal_rotation_action() {
        let expr = sub(mul(var("x"), var("vy")), mul(var("y"), var("vx")));
        let report = infer_symmetry_with_evidence(&expr);
        assert!(matches!(
            report.evidence,
            Some(SymmetryInferenceEvidence::StructuralPatternMatch {
                pattern: StructuralSymmetryPattern::TwoDimensionalAntisymmetricBilinear {
                    first_pair,
                    second_pair,
                },
            }) if first_pair == ["x".to_string(), "y".to_string()]
                && second_pair == ["vx".to_string(), "vy".to_string()]
        ));
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
    fn numeric_search_reports_budget_exhaustion_for_unsearched_extra_pairs() {
        let mut expr = var("x0");
        for index in 1..8 {
            expr = add(expr, var(&format!("x{index}")));
        }
        let report = infer_symmetry_numeric_with_evidence(&expr);
        assert!(report.candidate.is_none());
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::NumericSearchBudgetExceeded
        );
    }

    #[test]
    fn numeric_search_skips_pairwise_combinations_when_profile_disables_them() {
        let expr = sqrt_of(add(pow2("x"), pow2("y")));
        let mut profile = NumericLieDerivativeProfileV2::default();
        profile.pairwise_so2_search = false;
        let report = infer_symmetry_numeric_with_profile(
            &expr,
            profile,
            ExactPolynomialLieProfileV1::default(),
            ExactPolynomialSearchStatus::NotRequested,
        );
        assert!(report.candidate.is_none());
        assert_eq!(
            report.search_status,
            SymmetryInferenceSearchStatus::NumericSearchCompleted
        );
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
