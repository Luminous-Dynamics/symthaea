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
//! ## Declared-action metric-aware sampling
//!
//! `evaluate_metric_aware_linear_action_samples` accepts an admitted linear
//! action, explicit sample states in its declared coordinate basis, and fixed
//! parameter bindings. It uses forward-mode automatic differentiation for the
//! supported smooth AST fragment, then evaluates the dual/primal cosine after
//! Cholesky metric whitening. The exact states, action declarations, admission
//! diagnostics and versioned expression encoding are retained in the report.
//! Unsupported or nondifferentiable expressions abstain; invalid samples are
//! counted and cannot produce a positive invariance decision.
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

const HARD_MAX_METRIC_AWARE_ACTION_DIMENSION: usize = 32;
const HARD_MAX_METRIC_AWARE_ACTION_GENERATORS: usize = 128;
const HARD_MAX_METRIC_AWARE_EXPRESSION_VARIABLES: usize = 512;
const HARD_MAX_METRIC_AWARE_SAMPLES: usize = 4096;
const HARD_MAX_METRIC_AWARE_EXPRESSION_NODES: usize = 16_384;
const HARD_MAX_METRIC_AWARE_GRADIENT_WORK: usize = 20_000_000;
const HARD_MAX_METRIC_AWARE_GENERATOR_SAMPLE_PAIRS: usize = 65_536;

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
    /// Maximum monomial-pair multiplications per polynomial product operation.
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

/// Numerical policy for replayable metric-aware linear-action sampling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareLinearActionSampleProfileV1 {
    pub maximum_samples: usize,
    pub maximum_expression_nodes: usize,
    pub maximum_total_gradient_work: usize,
    /// Maximum number of (generator, sample) pairs evaluated for matrix actions.
    pub maximum_total_generator_sample_pairs: usize,
    pub minimum_informative_samples_per_generator: usize,
    /// Maximum squared cosine in metric-whitened coordinates.
    pub squared_cosine_tolerance: f64,
}

impl Default for MetricAwareLinearActionSampleProfileV1 {
    fn default() -> Self {
        Self {
            maximum_samples: 4096,
            maximum_expression_nodes: 16_384,
            maximum_total_gradient_work: 20_000_000,
            maximum_total_generator_sample_pairs: 65_536,
            minimum_informative_samples_per_generator: 1,
            squared_cosine_tolerance: 1e-8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricAwareLinearActionSampleErrorV1 {
    InvalidProfile,
    ActionDimensionLimitExceeded { actual: usize, maximum: usize },
    GeneratorCountLimitExceeded { actual: usize, maximum: usize },
    TooManySamples { actual: usize, maximum: usize },
    EmptySampleSet,
    SampleDimensionMismatch { sample_index: usize, expected: usize, actual: usize },
    NonFiniteSampleCoordinate { sample_index: usize, coordinate_index: usize },
    ExpressionTooComplex { actual: usize, maximum: usize },
    TotalGradientWorkExceeded { actual: usize, maximum: usize },
    GeneratorSampleBudgetExceeded { actual: usize, maximum: usize },
    EmptyFixedBindingName { index: usize },
    DuplicateFixedBinding { name: String },
    FixedBindingConflictsWithActionCoordinate { name: String },
    NonFiniteFixedBinding { name: String },
    MissingFixedBinding { name: String },
    UnsupportedExpression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareGeneratorSampleResultV1 {
    pub generator_id: String,
    pub sample_count: usize,
    pub informative_sample_count: usize,
    pub uninformative_sample_count: usize,
    pub invalid_sample_count: usize,
    pub maximum_squared_residual: f64,
    pub mean_squared_residual: f64,
}

/// Replayable sampled evidence for a declared action and scalar expression.
/// The effective and original declarations, exact expression tree, states and
/// fixed parameters are retained in the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareLinearActionSampleReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub original_action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub action_admission_profile: crate::linear_group_action::LinearGroupActionAdmissionProfileV1,
    pub action_admission_diagnostics: crate::linear_group_action::LinearGroupActionAdmissionDiagnosticsV1,
    pub coordinate_names: Vec<String>,
    pub coordinate_unit_ids: Vec<String>,
    pub expression_display: String,
    /// Structural AST representation with exact IEEE-754 bit patterns.
    pub expression_canonical: String,
    pub expression_complexity: usize,
    pub profile: MetricAwareLinearActionSampleProfileV1,
    pub fixed_bindings: Vec<(String, f64)>,
    pub sample_states: Vec<Vec<f64>>,
    pub generator_results: Vec<MetricAwareGeneratorSampleResultV1>,
}

impl MetricAwareLinearActionSampleReportV1 {
    /// True only when every generator has enough informative samples, no invalid
    /// evaluations, and a maximum residual within the declared threshold.
    pub fn invariant_under_every_generator(&self) -> bool {
        !self.generator_results.is_empty()
            && self.generator_results.iter().all(|result| {
                result.invalid_sample_count == 0
                    && result.informative_sample_count
                        >= self.profile.minimum_informative_samples_per_generator
                    && result.maximum_squared_residual
                        <= self.profile.squared_cosine_tolerance
            })
    }

    pub fn total_invalid_evaluations(&self) -> usize {
        self.generator_results
            .iter()
            .map(|result| result.invalid_sample_count)
            .sum()
    }

    pub fn total_informative_evaluations(&self) -> usize {
        self.generator_results
            .iter()
            .map(|result| result.informative_sample_count)
            .sum()
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
#[derive(Debug, Clone)]
struct DualValue {
    value: f64,
    gradient: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DualEvaluationError {
    Unsupported,
    Nondifferentiable,
    InvalidDomain,
    NonFinite,
}

fn checked_dual(value: f64, gradient: Vec<f64>) -> Result<DualValue, DualEvaluationError> {
    if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
        return Err(DualEvaluationError::NonFinite);
    }
    Ok(DualValue { value, gradient })
}

/// Forward-mode differentiation for the smooth, supported expression fragment.
/// Abs, Floor and Sum abstain rather than receiving an unjustified derivative.
fn eval_expression_with_gradient(
    expr: &Expr,
    variable_names: &[String],
    active_variable_indices: &[usize],
    values: &[f64],
) -> Result<DualValue, DualEvaluationError> {
    let dimension = variable_names.len();
    match expr {
        Expr::Var(name) => {
            let index = variable_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or(DualEvaluationError::Unsupported)?;
            let mut gradient = vec![0.0; dimension];
            if active_variable_indices.contains(&index) {
                gradient[index] = 1.0;
            }
            checked_dual(values[index], gradient)
        }
        Expr::Const(value) => checked_dual(*value, vec![0.0; dimension]),
        Expr::BinOp(operation, left_expr, right_expr) => {
            let left = eval_expression_with_gradient(
                left_expr,
                variable_names,
                active_variable_indices,
                values,
            )?;
            let right = eval_expression_with_gradient(
                right_expr,
                variable_names,
                active_variable_indices,
                values,
            )?;
            let mut gradient = vec![0.0; dimension];
            let value = match operation {
                BinOp::Add => {
                    for (out, (left_gradient, right_gradient)) in gradient
                        .iter_mut()
                        .zip(left.gradient.iter().zip(&right.gradient))
                    {
                        *out = left_gradient + right_gradient;
                    }
                    left.value + right.value
                }
                BinOp::Sub => {
                    for (out, (left_gradient, right_gradient)) in gradient
                        .iter_mut()
                        .zip(left.gradient.iter().zip(&right.gradient))
                    {
                        *out = left_gradient - right_gradient;
                    }
                    left.value - right.value
                }
                BinOp::Mul => {
                    for (out, (left_gradient, right_gradient)) in gradient
                        .iter_mut()
                        .zip(left.gradient.iter().zip(&right.gradient))
                    {
                        *out = left_gradient * right.value + left.value * right_gradient;
                    }
                    left.value * right.value
                }
                BinOp::Div => {
                    if right.value.abs() <= 1e-15 {
                        return Err(DualEvaluationError::InvalidDomain);
                    }
                    let denominator = right.value * right.value;
                    for (out, (left_gradient, right_gradient)) in gradient
                        .iter_mut()
                        .zip(left.gradient.iter().zip(&right.gradient))
                    {
                        *out = (left_gradient * right.value - left.value * right_gradient)
                            / denominator;
                    }
                    left.value / right.value
                }
                BinOp::Pow => {
                    let exponent_is_constant = right.gradient.iter().all(|entry| *entry == 0.0);
                    if left.value > 0.0 {
                        let value = left.value.powf(right.value);
                        let log_base = left.value.ln();
                        for (out, (left_gradient, right_gradient)) in gradient
                            .iter_mut()
                            .zip(left.gradient.iter().zip(&right.gradient))
                        {
                            *out = value
                                * (right.value * left_gradient / left.value
                                    + log_base * right_gradient);
                        }
                        value
                    } else if left.value < 0.0
                        && exponent_is_constant
                        && right.value.fract() == 0.0
                    {
                        let value = left.value.powf(right.value);
                        let derivative = right.value * left.value.powf(right.value - 1.0);
                        for (out, left_gradient) in gradient.iter_mut().zip(&left.gradient) {
                            *out = derivative * left_gradient;
                        }
                        value
                    } else if left.value == 0.0 && exponent_is_constant && right.value > 0.0 {
                        if right.value == 1.0 {
                            gradient.clone_from(&left.gradient);
                        } else if right.value > 1.0 {
                            gradient.fill(0.0);
                        } else {
                            return Err(DualEvaluationError::Nondifferentiable);
                        }
                        0.0_f64.powf(right.value)
                    } else {
                        return Err(DualEvaluationError::InvalidDomain);
                    }
                }
            };
            checked_dual(value, gradient)
        }
        Expr::Func(function, argument) => {
            let argument = eval_expression_with_gradient(
                argument,
                variable_names,
                active_variable_indices,
                values,
            )?;
            let mut gradient = vec![0.0; dimension];
            let value = match function {
                UnaryFn::Sqrt => {
                    if argument.value < 0.0 {
                        return Err(DualEvaluationError::InvalidDomain);
                    }
                    if argument.value == 0.0 {
                        return Err(DualEvaluationError::Nondifferentiable);
                    }
                    let value = argument.value.sqrt();
                    let derivative = 0.5 / value;
                    for (out, input) in gradient.iter_mut().zip(&argument.gradient) {
                        *out = derivative * input;
                    }
                    value
                }
                UnaryFn::Log => {
                    if argument.value <= 0.0 {
                        return Err(DualEvaluationError::InvalidDomain);
                    }
                    for (out, input) in gradient.iter_mut().zip(&argument.gradient) {
                        *out = input / argument.value;
                    }
                    argument.value.ln()
                }
                UnaryFn::Exp => {
                    let value = argument.value.exp();
                    for (out, input) in gradient.iter_mut().zip(&argument.gradient) {
                        *out = value * input;
                    }
                    value
                }
                UnaryFn::Sin => {
                    let derivative = argument.value.cos();
                    for (out, input) in gradient.iter_mut().zip(&argument.gradient) {
                        *out = derivative * input;
                    }
                    argument.value.sin()
                }
                UnaryFn::Cos => {
                    let derivative = -argument.value.sin();
                    for (out, input) in gradient.iter_mut().zip(&argument.gradient) {
                        *out = derivative * input;
                    }
                    argument.value.cos()
                }
                UnaryFn::Abs | UnaryFn::Floor => return Err(DualEvaluationError::Unsupported),
            };
            checked_dual(value, gradient)
        }
        Expr::Sum(_, _) => Err(DualEvaluationError::Unsupported),
    }
}

/// Return true only for the smooth AST subset supported by the metric-aware
/// forward-mode derivative. Reject unsupported syntax before fixed-binding
/// resolution so a summation or nondifferentiable node cannot be misreported
/// as a missing parameter.
fn metric_aware_expression_supported(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) | Expr::Const(_) => true,
        Expr::BinOp(_, left, right) => {
            metric_aware_expression_supported(left)
                && metric_aware_expression_supported(right)
        }
        Expr::Func(UnaryFn::Abs | UnaryFn::Floor, _) | Expr::Sum(_, _) => false,
        Expr::Func(_, argument) => metric_aware_expression_supported(argument),
    }
}

fn canonical_expression(expr: &Expr) -> String {
    format!("symthaea-expr-v1:{}", canonical_expression_node(expr))
}

fn canonical_expression_node(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => format!("V{}:{name}", name.len()),
        Expr::Const(value) => format!("C{:016x}", value.to_bits()),
        Expr::BinOp(operation, left, right) => {
            let tag = match operation {
                BinOp::Add => "ADD",
                BinOp::Sub => "SUB",
                BinOp::Mul => "MUL",
                BinOp::Div => "DIV",
                BinOp::Pow => "POW",
            };
            format!("B{tag}({},{})", canonical_expression_node(left), canonical_expression_node(right))
        }
        Expr::Func(function, argument) => {
            let tag = match function {
                UnaryFn::Sqrt => "SQRT",
                UnaryFn::Log => "LOG",
                UnaryFn::Exp => "EXP",
                UnaryFn::Sin => "SIN",
                UnaryFn::Cos => "COS",
                UnaryFn::Abs => "ABS",
                UnaryFn::Floor => "FLOOR",
            };
            format!("F{tag}({})", canonical_expression_node(argument))
        }
        Expr::Sum(body, variable) => {
            format!("SUM{}:{variable}({})", variable.len(), canonical_expression_node(body))
        }
    }
}

/// Solve L * result = rhs for a lower-triangular L using forward substitution.
fn solve_lower_triangular(lower: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let mut result = Vec::with_capacity(rhs.len());
    for (row_index, rhs_value) in rhs.iter().enumerate() {
        let correction = result
            .iter()
            .zip(lower[row_index].iter())
            .map(|(value, coefficient)| *value * *coefficient)
            .sum::<f64>();
        let pivot = lower[row_index][row_index];
        if !pivot.is_finite() || pivot <= 0.0 {
            return None;
        }
        let value = (*rhs_value - correction) / pivot;
        if !value.is_finite() {
            return None;
        }
        result.push(value);
    }
    Some(result)
}

fn stable_vector_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}

/// Evaluate replayable metric-aware residuals for arbitrary admitted linear actions.
///
/// The derivative is computed analytically by forward-mode AD on the supported
/// smooth AST fragment. For G = s L L^T, this evaluates the Euclidean cosine
/// between L^-1 grad(f) and L^T A x, exactly the metric-aware dual/primal cosine.
/// No explicit inverse of G or finite-difference step is used.
pub fn evaluate_metric_aware_linear_action_samples(
    expr: &Expr,
    action: &AdmittedLinearGroupActionV1,
    sample_states: &[Vec<f64>],
    fixed_bindings: &[(&str, f64)],
    profile: MetricAwareLinearActionSampleProfileV1,
) -> Result<MetricAwareLinearActionSampleReportV1, MetricAwareLinearActionSampleErrorV1> {
    if profile.maximum_samples == 0
        || profile.maximum_samples > HARD_MAX_METRIC_AWARE_SAMPLES
        || profile.maximum_expression_nodes == 0
        || profile.maximum_expression_nodes > HARD_MAX_METRIC_AWARE_EXPRESSION_NODES
        || profile.maximum_total_gradient_work == 0
        || profile.maximum_total_gradient_work > HARD_MAX_METRIC_AWARE_GRADIENT_WORK
        || profile.maximum_total_generator_sample_pairs == 0
        || profile.maximum_total_generator_sample_pairs
            > HARD_MAX_METRIC_AWARE_GENERATOR_SAMPLE_PAIRS
        || profile.minimum_informative_samples_per_generator == 0
        || profile.minimum_informative_samples_per_generator > profile.maximum_samples
        || !profile.squared_cosine_tolerance.is_finite()
        || !(0.0..=1.0).contains(&profile.squared_cosine_tolerance)
    {
        return Err(MetricAwareLinearActionSampleErrorV1::InvalidProfile);
    }
    if sample_states.is_empty() {
        return Err(MetricAwareLinearActionSampleErrorV1::EmptySampleSet);
    }
    if sample_states.len() > profile.maximum_samples {
        return Err(MetricAwareLinearActionSampleErrorV1::TooManySamples {
            actual: sample_states.len(),
            maximum: profile.maximum_samples,
        });
    }

    let declaration = action.action();
    let dimension = declaration.coordinate_names.len();
    if dimension > HARD_MAX_METRIC_AWARE_ACTION_DIMENSION {
        return Err(MetricAwareLinearActionSampleErrorV1::ActionDimensionLimitExceeded {
            actual: dimension,
            maximum: HARD_MAX_METRIC_AWARE_ACTION_DIMENSION,
        });
    }
    if declaration.generators.len() > HARD_MAX_METRIC_AWARE_ACTION_GENERATORS {
        return Err(MetricAwareLinearActionSampleErrorV1::GeneratorCountLimitExceeded {
            actual: declaration.generators.len(),
            maximum: HARD_MAX_METRIC_AWARE_ACTION_GENERATORS,
        });
    }
    let complexity = expr.complexity();
    if !metric_aware_expression_supported(expr) {
        return Err(MetricAwareLinearActionSampleErrorV1::UnsupportedExpression);
    }
    if complexity > profile.maximum_expression_nodes {
        return Err(MetricAwareLinearActionSampleErrorV1::ExpressionTooComplex {
            actual: complexity,
            maximum: profile.maximum_expression_nodes,
        });
    }

    for (sample_index, sample) in sample_states.iter().enumerate() {
        if sample.len() != dimension {
            return Err(MetricAwareLinearActionSampleErrorV1::SampleDimensionMismatch {
                sample_index,
                expected: dimension,
                actual: sample.len(),
            });
        }
        for (coordinate_index, value) in sample.iter().enumerate() {
            if !value.is_finite() {
                return Err(MetricAwareLinearActionSampleErrorV1::NonFiniteSampleCoordinate {
                    sample_index,
                    coordinate_index,
                });
            }
        }
    }

    let mut fixed_values = BTreeMap::new();
    for (index, (raw_name, value)) in fixed_bindings.iter().enumerate() {
        let name = *raw_name;
        if name.trim().is_empty() {
            return Err(MetricAwareLinearActionSampleErrorV1::EmptyFixedBindingName { index });
        }
        if declaration.coordinate_names.iter().any(|coordinate| coordinate.as_str() == name) {
            return Err(
                MetricAwareLinearActionSampleErrorV1::FixedBindingConflictsWithActionCoordinate {
                    name: name.to_string(),
                },
            );
        }
        if !value.is_finite() {
            return Err(MetricAwareLinearActionSampleErrorV1::NonFiniteFixedBinding {
                name: name.to_string(),
            });
        }
        if fixed_values.insert(name.to_string(), *value).is_some() {
            return Err(MetricAwareLinearActionSampleErrorV1::DuplicateFixedBinding {
                name: name.to_string(),
            });
        }
    }

    let mut variable_names = Vec::new();
    collect_var_names(expr, &mut variable_names);
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    for name in &variable_names {
        if !declaration.coordinate_names.contains(name)
            && !fixed_values.contains_key(name)
        {
            return Err(MetricAwareLinearActionSampleErrorV1::MissingFixedBinding {
                name: name.clone(),
            });
        }
    }
    let coordinate_indices: Vec<usize> = declaration
        .coordinate_names
        .iter()
        .map(|name| {
            variable_names
                .iter()
                .position(|candidate| candidate == name)
                .expect("action coordinates appended to variable order")
        })
        .collect();

    let total_work = sample_states
        .len()
        .checked_mul(declaration.generators.len())
        .and_then(|count| count.checked_mul(complexity))
        .and_then(|count| count.checked_mul(variable_names.len()))
        .ok_or(MetricAwareLinearActionSampleErrorV1::InvalidProfile)?;
    let generator_sample_pairs = sample_states
        .len()
        .checked_mul(declaration.generators.len())
        .ok_or(MetricAwareLinearActionSampleErrorV1::InvalidProfile)?;
    if generator_sample_pairs > profile.maximum_total_generator_sample_pairs {
        return Err(MetricAwareLinearActionSampleErrorV1::GeneratorSampleBudgetExceeded {
            actual: generator_sample_pairs,
            maximum: profile.maximum_total_generator_sample_pairs,
        });
    }
    if total_work > profile.maximum_total_gradient_work {
        return Err(
            MetricAwareLinearActionSampleErrorV1::TotalGradientWorkExceeded {
                actual: total_work,
                maximum: profile.maximum_total_gradient_work,
            },
        );
    }

    let fixed_bindings_owned: Vec<(String, f64)> = fixed_bindings
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect();
    let lower = action.scaled_metric_cholesky_lower();
    let mut generator_results = Vec::with_capacity(declaration.generators.len());

    for generator in &declaration.generators {
        let mut informative_sample_count = 0usize;
        let mut uninformative_sample_count = 0usize;
        let mut invalid_sample_count = 0usize;
        let mut maximum_squared_residual = 0.0_f64;
        let mut residual_sum = 0.0_f64;

        for sample in sample_states {
            let mut values = vec![0.0_f64; variable_names.len()];
            for (coordinate_index, variable_index) in coordinate_indices.iter().enumerate() {
                values[*variable_index] = sample[coordinate_index];
            }
            for (name, value) in &fixed_values {
                let variable_index = variable_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .expect("fixed expression variable retained in AST variable order");
                values[variable_index] = *value;
            }

            let dual = match eval_expression_with_gradient(
                expr,
                &variable_names,
                &coordinate_indices,
                &values,
            ) {
                Ok(result) => result,
                Err(DualEvaluationError::Unsupported) => {
                    return Err(MetricAwareLinearActionSampleErrorV1::UnsupportedExpression);
                }
                Err(
                    DualEvaluationError::Nondifferentiable
                    | DualEvaluationError::InvalidDomain
                    | DualEvaluationError::NonFinite,
                ) => {
                    invalid_sample_count = invalid_sample_count.saturating_add(1);
                    continue;
                }
            };

            let mut gradient = vec![0.0_f64; dimension];
            for (coordinate_index, variable_index) in coordinate_indices.iter().enumerate() {
                gradient[coordinate_index] = dual.gradient[*variable_index];
            }

            let mut flow = vec![0.0_f64; dimension];
            for (row_index, row) in generator.matrix.iter().enumerate() {
                for (column_index, coefficient) in row.iter().enumerate() {
                    flow[row_index] += coefficient * sample[column_index];
                }
            }
            if flow.iter().any(|value| !value.is_finite()) {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                continue;
            }

            let Some(whitened_gradient) = solve_lower_triangular(lower, &gradient) else {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                continue;
            };
            let mut whitened_flow = vec![0.0_f64; dimension];
            for (column_index, output) in whitened_flow.iter_mut().enumerate() {
                for (row_index, flow_value) in flow.iter().enumerate() {
                    *output += lower[row_index][column_index] * *flow_value;
                }
            }
            if whitened_flow.iter().any(|value| !value.is_finite()) {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                continue;
            }

            let gradient_norm = stable_vector_norm(&whitened_gradient);
            let flow_norm = stable_vector_norm(&whitened_flow);
            if !gradient_norm.is_finite() || !flow_norm.is_finite() {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                continue;
            }
            if gradient_norm == 0.0 || flow_norm == 0.0 {
                uninformative_sample_count = uninformative_sample_count.saturating_add(1);
                continue;
            }

            let directional_cosine = whitened_gradient
                .iter()
                .zip(&whitened_flow)
                .map(|(gradient_value, flow_value)| {
                    (gradient_value / gradient_norm) * (flow_value / flow_norm)
                })
                .sum::<f64>();
            if !directional_cosine.is_finite() {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                continue;
            }
            let squared_residual = (directional_cosine * directional_cosine).clamp(0.0, 1.0);
            informative_sample_count = informative_sample_count.saturating_add(1);
            maximum_squared_residual = maximum_squared_residual.max(squared_residual);
            residual_sum += squared_residual;
        }

        let mean_squared_residual = if informative_sample_count == 0 {
            0.0
        } else {
            residual_sum / informative_sample_count as f64
        };
        generator_results.push(MetricAwareGeneratorSampleResultV1 {
            generator_id: generator.id.clone(),
            sample_count: sample_states.len(),
            informative_sample_count,
            uninformative_sample_count,
            invalid_sample_count,
            maximum_squared_residual,
            mean_squared_residual,
        });
    }

    Ok(MetricAwareLinearActionSampleReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        action_declaration: declaration.clone(),
        original_action_declaration: action.declared_action().clone(),
        action_admission_profile: *action.profile(),
        action_admission_diagnostics: action.diagnostics().clone(),
        coordinate_names: declaration.coordinate_names.clone(),
        coordinate_unit_ids: declaration.coordinate_unit_ids.clone(),
        expression_display: expr.to_string(),
        expression_canonical: canonical_expression(expr),
        expression_complexity: complexity,
        profile,
        fixed_bindings: fixed_bindings_owned,
        sample_states: sample_states.to_vec(),
        generator_results,
    })
}


/// Numerical policy for infinitesimal equivariance of a vector field.
/// Explicit output-space representation for a vector field under the same
/// named infinitesimal generators as an admitted state-space action.
/// Output dimension, basis and units may differ from the state-space representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearVectorFieldOutputRepresentationV1 {
    pub representation_id: String,
    pub coordinate_basis_id: String,
    pub component_names: Vec<String>,
    pub component_unit_ids: Vec<String>,
    pub generators: Vec<crate::linear_group_action::LinearGeneratorV1>,
}

/// Typed errors from exact polynomial vector-field equivariance checking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactVectorFieldEquivarianceErrorV1 {
    InvalidProfile,
    EmptyOutputRepresentationId,
    EmptyOutputBasisId,
    NoOutputComponents,
    OutputComponentCountMismatch { expected: usize, actual: usize },
    EmptyOutputComponentName { index: usize },
    DuplicateOutputComponentName { name: String },
    EmptyOutputUnitId { index: usize },
    OutputGeneratorCountMismatch { expected: usize, actual: usize },
    MissingOutputGenerator { generator_id: String },
    DuplicateOutputGenerator { generator_id: String },
    OutputGeneratorRowCountMismatch { generator_id: String, expected: usize, actual: usize },
    OutputGeneratorColumnCountMismatch {
        generator_id: String, row: usize, expected: usize, actual: usize
    },
    NonFiniteOutputGeneratorEntry {
        generator_id: String, row: usize, column: usize
    },
    UnsupportedExpression,
    ResourceLimitExceeded,
}

/// Exact residual term counts for one paired state/output generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactVectorFieldGeneratorResultV1 {
    pub generator_id: String,
    pub component_residual_term_counts: Vec<usize>,
}

impl ExactVectorFieldGeneratorResultV1 {
    pub fn equivariant(&self) -> bool {
        self.component_residual_term_counts.iter().all(|count| *count == 0)
    }
}

/// Exact polynomial report for D F(x) A x - B F(x).
///
/// This establishes only an exact algebraic identity for the interpreted
/// polynomial components and bounded exact coefficient arithmetic. It does not
/// establish Lie-algebra representation compatibility, global group integration,
/// conservation, or physical qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactVectorFieldEquivarianceReportV1 {
    pub action_id: String,
    pub state_coordinate_basis_id: String,
    pub output_representation: LinearVectorFieldOutputRepresentationV1,
    pub state_action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub original_state_action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub state_action_admission_profile: crate::linear_group_action::LinearGroupActionAdmissionProfileV1,
    pub state_action_admission_diagnostics: crate::linear_group_action::LinearGroupActionAdmissionDiagnosticsV1,
    pub state_coordinate_names: Vec<String>,
    pub output_component_names: Vec<String>,
    pub expression_variables: Vec<String>,
    pub vector_field_canonical: Vec<String>,
    pub profile: ExactPolynomialLieProfileV1,
    pub generator_results: Vec<ExactVectorFieldGeneratorResultV1>,
}

impl ExactVectorFieldEquivarianceReportV1 {
    pub fn equivariant_under_every_generator(&self) -> bool {
        !self.generator_results.is_empty()
            && self.generator_results.iter().all(ExactVectorFieldGeneratorResultV1::equivariant)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareVectorFieldEquivarianceProfileV1 {
    pub maximum_samples: usize,
    pub maximum_expression_nodes: usize,
    pub maximum_total_gradient_work: usize,
    pub maximum_total_generator_sample_pairs: usize,
    pub minimum_informative_samples_per_generator: usize,
    /// Maximum relative metric-whitened infinitesimal equivariance defect.
    pub relative_equivariance_tolerance: f64,
}

impl Default for MetricAwareVectorFieldEquivarianceProfileV1 {
    fn default() -> Self {
        Self {
            maximum_samples: 4096,
            maximum_expression_nodes: 16_384,
            maximum_total_gradient_work: 20_000_000,
            maximum_total_generator_sample_pairs: 65_536,
            minimum_informative_samples_per_generator: 1,
            relative_equivariance_tolerance: 1.0e-8,
        }
    }
}

/// Fail-closed input errors for a sampled vector-field equivariance check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricAwareVectorFieldEquivarianceErrorV1 {
    InvalidProfile,
    EmptySampleSet,
    VectorFieldDimensionMismatch { expected: usize, actual: usize },
    ActionDimensionLimitExceeded { actual: usize, maximum: usize },
    GeneratorCountLimitExceeded { actual: usize, maximum: usize },
    TooManySamples { actual: usize, maximum: usize },
    SampleDimensionMismatch { sample_index: usize, expected: usize, actual: usize },
    NonFiniteSampleCoordinate { sample_index: usize, coordinate_index: usize },
    ExpressionTooComplex { actual: usize, maximum: usize },
    ExpressionVariableLimitExceeded { actual: usize, maximum: usize },
    TotalGradientWorkExceeded { actual: usize, maximum: usize },
    GeneratorSampleBudgetExceeded { actual: usize, maximum: usize },
    EmptyFixedBindingName { index: usize },
    DuplicateFixedBinding { name: String },
    FixedBindingConflictsWithActionCoordinate { name: String },
    NonFiniteFixedBinding { name: String },
    MissingFixedBinding { name: String },
    UnsupportedExpression,
}

/// Per-generator diagnostics for infinitesimal vector-field equivariance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareVectorFieldGeneratorResultV1 {
    pub generator_id: String,
    pub sample_count: usize,
    pub informative_sample_count: usize,
    pub uninformative_sample_count: usize,
    pub invalid_sample_count: usize,
    pub maximum_relative_defect: f64,
    pub mean_relative_defect: f64,
    /// One entry per supplied sample, in original order. None denotes an
    /// uninformative or invalid field evaluation.
    pub sample_relative_defects: Vec<Option<f64>>,
}

/// Replayable evidence for sampled infinitesimal equivariance of a vector field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAwareVectorFieldEquivarianceReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub original_action_declaration: crate::linear_group_action::LinearGroupActionV1,
    pub action_admission_profile: crate::linear_group_action::LinearGroupActionAdmissionProfileV1,
    pub action_admission_diagnostics: crate::linear_group_action::LinearGroupActionAdmissionDiagnosticsV1,
    pub coordinate_names: Vec<String>,
    pub coordinate_unit_ids: Vec<String>,
    pub vector_field_display: Vec<String>,
    pub vector_field_canonical: Vec<String>,
    pub total_expression_complexity: usize,
    pub profile: MetricAwareVectorFieldEquivarianceProfileV1,
    pub fixed_bindings: Vec<(String, f64)>,
    pub sample_states: Vec<Vec<f64>>,
    pub generator_results: Vec<MetricAwareVectorFieldGeneratorResultV1>,
}

impl MetricAwareVectorFieldEquivarianceReportV1 {
    /// True only when every generator has enough informative samples, no invalid
    /// evaluations, and a maximum relative equivariance defect within tolerance.
    pub fn equivariant_under_every_generator(&self) -> bool {
        !self.generator_results.is_empty()
            && self.generator_results.iter().all(|result| {
                result.invalid_sample_count == 0
                    && result.informative_sample_count
                        >= self.profile.minimum_informative_samples_per_generator
                    && result.maximum_relative_defect
                        <= self.profile.relative_equivariance_tolerance
            })
    }

    pub fn total_invalid_evaluations(&self) -> usize {
        self.generator_results
            .iter()
            .map(|result| result.invalid_sample_count)
            .sum()
    }

    pub fn total_informative_evaluations(&self) -> usize {
        self.generator_results
            .iter()
            .map(|result| result.informative_sample_count)
            .sum()
    }
}

struct EvaluatedVectorFieldSample {
    values: Vec<f64>,
    jacobian: Vec<Vec<f64>>,
}

/// Scale-safe relative defect between two metric-whitened vectors.
///
/// Returns None when the inputs are non-finite, Some(None) when both terms are
/// zero (uninformative), or Some(Some(ratio)) for a finite relative defect.
fn relative_whitened_vector_defect(left: &[f64], right: &[f64]) -> Option<Option<f64>> {
    if left.len() != right.len()
        || left.iter().chain(right.iter()).any(|value| !value.is_finite())
    {
        return None;
    }
    let scale = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return Some(None);
    }

    let normalized_left: Vec<f64> = left.iter().map(|value| *value / scale).collect();
    let normalized_right: Vec<f64> = right.iter().map(|value| *value / scale).collect();
    let left_norm = stable_vector_norm(&normalized_left);
    let right_norm = stable_vector_norm(&normalized_right);
    let denominator = left_norm + right_norm;
    if denominator == 0.0 || !denominator.is_finite() {
        return Some(None);
    }
    let difference_norm = normalized_left
        .iter()
        .zip(normalized_right.iter())
        .fold(0.0_f64, |norm, (left_value, right_value)| {
            norm.hypot(left_value - right_value)
        });
    let relative_defect = difference_norm / denominator;
    relative_defect.is_finite().then_some(Some(relative_defect))
}

fn charge_exact_vector_field_operations(
    operation_count: &mut usize,
    additional: usize,
    profile: ExactPolynomialLieProfileV1,
) -> Result<(), ExactPolynomialError> {
    *operation_count = (*operation_count)
        .checked_add(additional)
        .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
    if *operation_count > profile.maximum_term_products {
        return Err(ExactPolynomialError::ResourceLimitExceeded);
    }
    Ok(())
}

fn add_scaled_exact_polynomial(
    target: &mut ExactPolynomial,
    source: &ExactPolynomial,
    scale: ExactRational,
    profile: ExactPolynomialLieProfileV1,
    operation_count: &mut usize,
) -> Result<(), ExactPolynomialError> {
    for (monomial, coefficient) in source {
        charge_exact_vector_field_operations(operation_count, 1, profile)?;
        let scaled = coefficient
            .checked_mul(scale)
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        polynomial_add_term(target, monomial.clone(), scaled, profile)?;
    }
    Ok(())
}

fn exact_vector_field_component_residual(
    state_components: &[ExactPolynomial],
    output_index: usize,
    state_generator: &crate::linear_group_action::LinearGeneratorV1,
    output_generator: &crate::linear_group_action::LinearGeneratorV1,
    state_coordinate_names: &[String],
    variable_names: &[String],
    profile: ExactPolynomialLieProfileV1,
    operation_count: &mut usize,
) -> Result<ExactPolynomial, ExactPolynomialError> {
    let mut residual = ExactPolynomial::new();

    // Build the state-orbit derivative term: sum_j (dF_i/dx_j) * (A x)_j.
    for (row_index, generator_row) in state_generator.matrix.iter().enumerate() {
        let state_variable_index = variable_names
            .iter()
            .position(|name| name == &state_coordinate_names[row_index])
            .ok_or(ExactPolynomialError::UnsupportedExpression)?;
        let derivative = polynomial_derivative(
            &state_components[output_index],
            state_variable_index,
            profile,
        )?;
        for (column_index, matrix_entry) in generator_row.iter().enumerate() {
            let coefficient = ExactRational::from_f64(
                *matrix_entry,
                profile.maximum_dyadic_denominator_power,
            )
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
            if coefficient.is_zero() {
                continue;
            }
            let coordinate_variable_index = variable_names
                .iter()
                .position(|name| name == &state_coordinate_names[column_index])
                .ok_or(ExactPolynomialError::UnsupportedExpression)?;
            charge_exact_vector_field_operations(operation_count, derivative.len(), profile)?;
            let term = polynomial_multiply_variable(
                &derivative,
                coordinate_variable_index,
                profile,
            )?;
            add_scaled_exact_polynomial(
                &mut residual,
                &term,
                coefficient,
                profile,
                operation_count,
            )?;
        }
    }

    // Subtract the explicit output-orbit term: sum_j B_ij F_j(x).
    for (component_index, matrix_entry) in
        output_generator.matrix[output_index].iter().enumerate()
    {
        let coefficient = ExactRational::from_f64(
            *matrix_entry,
            profile.maximum_dyadic_denominator_power,
        )
        .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        if coefficient.is_zero() {
            continue;
        }
        let negative_coefficient = coefficient
            .checked_neg()
            .ok_or(ExactPolynomialError::ResourceLimitExceeded)?;
        add_scaled_exact_polynomial(
            &mut residual,
            &state_components[component_index],
            negative_coefficient,
            profile,
            operation_count,
        )?;
    }

    Ok(residual)
}

/// Check exact polynomial infinitesimal equivariance under explicitly paired
/// state and output representations.
///
/// For each paired generator (A, B), every output component is checked against
/// D F(x) A x - B F(x) = 0. State and output dimensions may differ. The output
/// generator IDs must pair one-to-one with the admitted state-action generator
/// IDs; no implicit B = A assumption is made.
///
/// All finite f64 polynomial coefficients and generator matrix entries are
/// interpreted as exact IEEE-754 dyadic rationals in the existing bounded
/// sparse-polynomial arithmetic. Variables outside the state action remain
/// symbolic parameters in the identity.
pub fn check_exact_vector_field_equivariance(
    vector_field: &[Expr],
    state_action: &AdmittedLinearGroupActionV1,
    output_representation: &LinearVectorFieldOutputRepresentationV1,
    profile: ExactPolynomialLieProfileV1,
) -> Result<ExactVectorFieldEquivarianceReportV1, ExactVectorFieldEquivarianceErrorV1> {
    if profile.maximum_variables == 0
        || profile.maximum_terms == 0
        || profile.maximum_term_products == 0
        || profile.maximum_total_degree == 0
        || profile.maximum_power == 0
        || profile.maximum_dyadic_denominator_power > 126
    {
        return Err(ExactVectorFieldEquivarianceErrorV1::InvalidProfile);
    }

    if output_representation.representation_id.trim().is_empty() {
        return Err(ExactVectorFieldEquivarianceErrorV1::EmptyOutputRepresentationId);
    }
    if output_representation.coordinate_basis_id.trim().is_empty() {
        return Err(ExactVectorFieldEquivarianceErrorV1::EmptyOutputBasisId);
    }

    let output_dimension = output_representation.component_names.len();
    if output_dimension == 0 {
        return Err(ExactVectorFieldEquivarianceErrorV1::NoOutputComponents);
    }
    if output_dimension > HARD_MAX_METRIC_AWARE_ACTION_DIMENSION {
        return Err(ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded);
    }
    if vector_field.len() != output_dimension {
        return Err(ExactVectorFieldEquivarianceErrorV1::OutputComponentCountMismatch {
            expected: output_dimension,
            actual: vector_field.len(),
        });
    }
    if output_representation.component_unit_ids.len() != output_dimension {
        return Err(ExactVectorFieldEquivarianceErrorV1::OutputComponentCountMismatch {
            expected: output_dimension,
            actual: output_representation.component_unit_ids.len(),
        });
    }

    let mut component_names = BTreeSet::new();
    for (index, name) in output_representation.component_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(ExactVectorFieldEquivarianceErrorV1::EmptyOutputComponentName {
                index,
            });
        }
        if !component_names.insert(name) {
            return Err(ExactVectorFieldEquivarianceErrorV1::DuplicateOutputComponentName {
                name: name.clone(),
            });
        }
    }
    for (index, unit_id) in output_representation.component_unit_ids.iter().enumerate() {
        if unit_id.trim().is_empty() {
            return Err(ExactVectorFieldEquivarianceErrorV1::EmptyOutputUnitId { index });
        }
    }

    let state_declaration = state_action.action();
    let state_dimension = state_declaration.coordinate_names.len();
    if state_dimension > HARD_MAX_METRIC_AWARE_ACTION_DIMENSION {
        return Err(ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded);
    }
    if state_declaration.generators.len() > HARD_MAX_METRIC_AWARE_ACTION_GENERATORS {
        return Err(ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded);
    }
    if output_representation.generators.len() != state_declaration.generators.len() {
        return Err(ExactVectorFieldEquivarianceErrorV1::OutputGeneratorCountMismatch {
            expected: state_declaration.generators.len(),
            actual: output_representation.generators.len(),
        });
    }

    let mut output_generator_by_id = BTreeMap::new();
    for generator in &output_representation.generators {
        if generator.id.trim().is_empty() {
            return Err(ExactVectorFieldEquivarianceErrorV1::MissingOutputGenerator {
                generator_id: generator.id.clone(),
            });
        }
        if output_generator_by_id
            .insert(generator.id.as_str(), generator)
            .is_some()
        {
            return Err(ExactVectorFieldEquivarianceErrorV1::DuplicateOutputGenerator {
                generator_id: generator.id.clone(),
            });
        }
        if generator.matrix.len() != output_dimension {
            return Err(ExactVectorFieldEquivarianceErrorV1::OutputGeneratorRowCountMismatch {
                generator_id: generator.id.clone(),
                expected: output_dimension,
                actual: generator.matrix.len(),
            });
        }
        for (row_index, row) in generator.matrix.iter().enumerate() {
            if row.len() != output_dimension {
                return Err(
                    ExactVectorFieldEquivarianceErrorV1::OutputGeneratorColumnCountMismatch {
                        generator_id: generator.id.clone(),
                        row: row_index,
                        expected: output_dimension,
                        actual: row.len(),
                    },
                );
            }
            for (column_index, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(
                        ExactVectorFieldEquivarianceErrorV1::NonFiniteOutputGeneratorEntry {
                            generator_id: generator.id.clone(),
                            row: row_index,
                            column: column_index,
                        },
                    );
                }
            }
        }
    }

    let mut variable_names = Vec::new();
    for component in vector_field {
        if !metric_aware_expression_supported(component) {
            return Err(ExactVectorFieldEquivarianceErrorV1::UnsupportedExpression);
        }
        collect_var_names(component, &mut variable_names);
    }
    for coordinate_name in &state_declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    if variable_names.len() > profile.maximum_variables
        || variable_names.len() > HARD_MAX_METRIC_AWARE_EXPRESSION_VARIABLES
    {
        return Err(ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded);
    }

    let mut state_components = Vec::with_capacity(output_dimension);
    for component in vector_field {
        let polynomial = parse_exact_polynomial(component, &variable_names, profile).map_err(
            |error| match error {
                ExactPolynomialError::UnsupportedExpression => {
                    ExactVectorFieldEquivarianceErrorV1::UnsupportedExpression
                }
                ExactPolynomialError::ResourceLimitExceeded => {
                    ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded
                }
            },
        )?;
        state_components.push(polynomial);
    }

    let mut operation_count = 0usize;
    let mut generator_results = Vec::with_capacity(state_declaration.generators.len());
    for state_generator in &state_declaration.generators {
        let output_generator = output_generator_by_id
            .get(state_generator.id.as_str())
            .copied()
            .ok_or_else(|| ExactVectorFieldEquivarianceErrorV1::MissingOutputGenerator {
                generator_id: state_generator.id.clone(),
            })?;

        let mut component_residual_term_counts = Vec::with_capacity(output_dimension);
        for output_index in 0..output_dimension {
            let residual = exact_vector_field_component_residual(
                &state_components,
                output_index,
                state_generator,
                output_generator,
                &state_declaration.coordinate_names,
                &variable_names,
                profile,
                &mut operation_count,
            )
            .map_err(|error| match error {
                ExactPolynomialError::UnsupportedExpression => {
                    ExactVectorFieldEquivarianceErrorV1::UnsupportedExpression
                }
                ExactPolynomialError::ResourceLimitExceeded => {
                    ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded
                }
            })?;
            component_residual_term_counts.push(residual.len());
        }

        generator_results.push(ExactVectorFieldGeneratorResultV1 {
            generator_id: state_generator.id.clone(),
            component_residual_term_counts,
        });
    }

    Ok(ExactVectorFieldEquivarianceReportV1 {
        action_id: state_declaration.action_id.clone(),
        state_coordinate_basis_id: state_declaration.coordinate_basis_id.clone(),
        output_representation: output_representation.clone(),
        state_action_declaration: state_declaration.clone(),
        original_state_action_declaration: state_action.declared_action().clone(),
        state_action_admission_profile: *state_action.profile(),
        state_action_admission_diagnostics: state_action.diagnostics().clone(),
        state_coordinate_names: state_declaration.coordinate_names.clone(),
        output_component_names: output_representation.component_names.clone(),
        expression_variables: variable_names,
        vector_field_canonical: vector_field.iter().map(canonical_expression).collect(),
        profile,
        generator_results,
    })
}

/// Evaluate a vector field's infinitesimal equivariance under every generator
/// of an explicitly admitted linear action.
///
/// For a linear action generated by A, equivariance requires
///
///     D F(x) * A x = A * F(x).
///
/// This function evaluates the defect at the caller-supplied states. It uses
/// forward-mode AD on the supported smooth expression fragment and measures the
/// vector defect in the declared metric. It does not establish finite group
/// equivariance, discrete-time step-map equivariance, conservation, or a physical
/// Noether map.
pub fn evaluate_metric_aware_vector_field_equivariance(
    vector_field: &[Expr],
    action: &AdmittedLinearGroupActionV1,
    sample_states: &[Vec<f64>],
    fixed_bindings: &[(&str, f64)],
    profile: MetricAwareVectorFieldEquivarianceProfileV1,
) -> Result<
    MetricAwareVectorFieldEquivarianceReportV1,
    MetricAwareVectorFieldEquivarianceErrorV1,
> {
    if profile.maximum_samples == 0
        || profile.maximum_samples > HARD_MAX_METRIC_AWARE_SAMPLES
        || profile.maximum_expression_nodes == 0
        || profile.maximum_expression_nodes > HARD_MAX_METRIC_AWARE_EXPRESSION_NODES
        || profile.maximum_total_gradient_work == 0
        || profile.maximum_total_gradient_work > HARD_MAX_METRIC_AWARE_GRADIENT_WORK
        || profile.maximum_total_generator_sample_pairs == 0
        || profile.maximum_total_generator_sample_pairs
            > HARD_MAX_METRIC_AWARE_GENERATOR_SAMPLE_PAIRS
        || profile.minimum_informative_samples_per_generator == 0
        || profile.minimum_informative_samples_per_generator > profile.maximum_samples
        || !profile.relative_equivariance_tolerance.is_finite()
        || !(0.0..=1.0).contains(&profile.relative_equivariance_tolerance)
    {
        return Err(MetricAwareVectorFieldEquivarianceErrorV1::InvalidProfile);
    }
    if sample_states.is_empty() {
        return Err(MetricAwareVectorFieldEquivarianceErrorV1::EmptySampleSet);
    }
    if sample_states.len() > profile.maximum_samples {
        return Err(MetricAwareVectorFieldEquivarianceErrorV1::TooManySamples {
            actual: sample_states.len(),
            maximum: profile.maximum_samples,
        });
    }

    let declaration = action.action();
    let dimension = declaration.coordinate_names.len();
    if vector_field.len() != dimension {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::VectorFieldDimensionMismatch {
                expected: dimension,
                actual: vector_field.len(),
            },
        );
    }
    if dimension > HARD_MAX_METRIC_AWARE_ACTION_DIMENSION {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::ActionDimensionLimitExceeded {
                actual: dimension,
                maximum: HARD_MAX_METRIC_AWARE_ACTION_DIMENSION,
            },
        );
    }
    if declaration.generators.len() > HARD_MAX_METRIC_AWARE_ACTION_GENERATORS {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::GeneratorCountLimitExceeded {
                actual: declaration.generators.len(),
                maximum: HARD_MAX_METRIC_AWARE_ACTION_GENERATORS,
            },
        );
    }

    let mut total_expression_complexity = 0usize;
    for expression in vector_field {
        if !metric_aware_expression_supported(expression) {
            return Err(MetricAwareVectorFieldEquivarianceErrorV1::UnsupportedExpression);
        }
        total_expression_complexity = total_expression_complexity
            .checked_add(expression.complexity())
            .ok_or(MetricAwareVectorFieldEquivarianceErrorV1::InvalidProfile)?;
    }
    if total_expression_complexity > profile.maximum_expression_nodes {
        return Err(MetricAwareVectorFieldEquivarianceErrorV1::ExpressionTooComplex {
            actual: total_expression_complexity,
            maximum: profile.maximum_expression_nodes,
        });
    }

    let mut variable_names = Vec::new();
    for expression in vector_field {
        collect_var_names(expression, &mut variable_names);
    }
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    if variable_names.len() > HARD_MAX_METRIC_AWARE_EXPRESSION_VARIABLES {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::ExpressionVariableLimitExceeded {
                actual: variable_names.len(),
                maximum: HARD_MAX_METRIC_AWARE_EXPRESSION_VARIABLES,
            },
        );
    }

    let mut fixed_values = BTreeMap::new();
    for (index, (raw_name, value)) in fixed_bindings.iter().enumerate() {
        let name = *raw_name;
        if name.trim().is_empty() {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::EmptyFixedBindingName { index },
            );
        }
        if declaration
            .coordinate_names
            .iter()
            .any(|coordinate| coordinate.as_str() == name)
        {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::FixedBindingConflictsWithActionCoordinate {
                    name: name.to_string(),
                },
            );
        }
        if !value.is_finite() {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::NonFiniteFixedBinding {
                    name: name.to_string(),
                },
            );
        }
        if fixed_values.insert(name.to_string(), *value).is_some() {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::DuplicateFixedBinding {
                    name: name.to_string(),
                },
            );
        }
    }
    for name in &variable_names {
        if !declaration.coordinate_names.contains(name) && !fixed_values.contains_key(name) {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::MissingFixedBinding {
                    name: name.clone(),
                },
            );
        }
    }

    let generator_sample_pairs = sample_states
        .len()
        .checked_mul(declaration.generators.len())
        .ok_or(MetricAwareVectorFieldEquivarianceErrorV1::InvalidProfile)?;
    if generator_sample_pairs > profile.maximum_total_generator_sample_pairs {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::GeneratorSampleBudgetExceeded {
                actual: generator_sample_pairs,
                maximum: profile.maximum_total_generator_sample_pairs,
            },
        );
    }

    let gradient_work = sample_states
        .len()
        .checked_mul(dimension)
        .and_then(|count| count.checked_mul(total_expression_complexity))
        .and_then(|count| count.checked_mul(variable_names.len()))
        .ok_or(MetricAwareVectorFieldEquivarianceErrorV1::InvalidProfile)?;
    if gradient_work > profile.maximum_total_gradient_work {
        return Err(
            MetricAwareVectorFieldEquivarianceErrorV1::TotalGradientWorkExceeded {
                actual: gradient_work,
                maximum: profile.maximum_total_gradient_work,
            },
        );
    }

    let coordinate_indices: Vec<usize> = declaration
        .coordinate_names
        .iter()
        .map(|name| {
            variable_names
                .iter()
                .position(|candidate| candidate == name)
                .expect("action coordinates appended to variable order")
        })
        .collect();

    // Evaluate F(x) and D F(x) once per sample, then reuse them for each generator.
    let mut evaluated_samples: Vec<Option<EvaluatedVectorFieldSample>> =
        Vec::with_capacity(sample_states.len());

    for (sample_index, sample) in sample_states.iter().enumerate() {
        if sample.len() != dimension {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::SampleDimensionMismatch {
                    sample_index,
                    expected: dimension,
                    actual: sample.len(),
                },
            );
        }
        if let Some(coordinate_index) = sample.iter().position(|value| !value.is_finite()) {
            return Err(
                MetricAwareVectorFieldEquivarianceErrorV1::NonFiniteSampleCoordinate {
                    sample_index,
                    coordinate_index,
                },
            );
        }

        let mut values = vec![0.0_f64; variable_names.len()];
        for (coordinate_index, variable_index) in coordinate_indices.iter().enumerate() {
            values[*variable_index] = sample[coordinate_index];
        }
        for (name, value) in &fixed_values {
            let variable_index = variable_names
                .iter()
                .position(|candidate| candidate == name)
                .expect("fixed expression variable retained in AST variable order");
            values[variable_index] = *value;
        }

        let mut field_values = Vec::with_capacity(dimension);
        let mut jacobian = vec![vec![0.0_f64; dimension]; dimension];
        let mut invalid = false;
        for (component_index, expression) in vector_field.iter().enumerate() {
            match eval_expression_with_gradient(
                expression,
                &variable_names,
                &coordinate_indices,
                &values,
            ) {
                Ok(dual) => {
                    field_values.push(dual.value);
                    for (coordinate_index, variable_index) in
                        coordinate_indices.iter().enumerate()
                    {
                        jacobian[component_index][coordinate_index] =
                            dual.gradient[*variable_index];
                    }
                }
                Err(DualEvaluationError::Unsupported) => {
                    return Err(
                        MetricAwareVectorFieldEquivarianceErrorV1::UnsupportedExpression,
                    );
                }
                Err(
                    DualEvaluationError::Nondifferentiable
                    | DualEvaluationError::InvalidDomain
                    | DualEvaluationError::NonFinite,
                ) => {
                    invalid = true;
                    break;
                }
            }
        }

        if invalid
            || field_values.iter().any(|value| !value.is_finite())
            || jacobian.iter().flatten().any(|value| !value.is_finite())
        {
            evaluated_samples.push(None);
        } else {
            evaluated_samples.push(Some(EvaluatedVectorFieldSample {
                values: field_values,
                jacobian,
            }));
        }
    }

    let lower = action.scaled_metric_cholesky_lower();
    let mut generator_results = Vec::with_capacity(declaration.generators.len());
    for generator in &declaration.generators {
        let mut informative_sample_count = 0usize;
        let mut uninformative_sample_count = 0usize;
        let mut invalid_sample_count = 0usize;
        let mut residual_sum = 0.0_f64;
        let mut maximum_relative_defect = 0.0_f64;
        let mut sample_relative_defects = Vec::with_capacity(sample_states.len());

        for (sample, evaluated) in sample_states.iter().zip(evaluated_samples.iter()) {
            let Some(evaluated) = evaluated else {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                sample_relative_defects.push(None);
                continue;
            };

            let mut orbit_tangent = vec![0.0_f64; dimension];
            let mut pushed_forward_field = vec![0.0_f64; dimension];
            let mut derivative_along_orbit = vec![0.0_f64; dimension];

            for (row_index, generator_row) in generator.matrix.iter().enumerate() {
                for (column_index, coefficient) in generator_row.iter().enumerate() {
                    orbit_tangent[row_index] += coefficient * sample[column_index];
                    pushed_forward_field[row_index] += coefficient * evaluated.values[column_index];
                }
            }
            for (row_index, jacobian_row) in evaluated.jacobian.iter().enumerate() {
                derivative_along_orbit[row_index] = jacobian_row
                    .iter()
                    .zip(orbit_tangent.iter())
                    .map(|(derivative, tangent)| derivative * tangent)
                    .sum::<f64>();
            }

            if orbit_tangent.iter().any(|value| !value.is_finite())
                || pushed_forward_field.iter().any(|value| !value.is_finite())
                || derivative_along_orbit.iter().any(|value| !value.is_finite())
            {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                sample_relative_defects.push(None);
                continue;
            }

            let mut whitened_derivative = vec![0.0_f64; dimension];
            let mut whitened_pushforward = vec![0.0_f64; dimension];
            for (column_index, output) in whitened_derivative.iter_mut().enumerate() {
                for (row_index, value) in derivative_along_orbit.iter().enumerate() {
                    *output += lower[row_index][column_index] * *value;
                }
            }
            for (column_index, output) in whitened_pushforward.iter_mut().enumerate() {
                for (row_index, value) in pushed_forward_field.iter().enumerate() {
                    *output += lower[row_index][column_index] * *value;
                }
            }

            if whitened_derivative.iter().any(|value| !value.is_finite())
                || whitened_pushforward.iter().any(|value| !value.is_finite())
            {
                invalid_sample_count = invalid_sample_count.saturating_add(1);
                sample_relative_defects.push(None);
                continue;
            }

            let relative_defect = match relative_whitened_vector_defect(
                &whitened_derivative,
                &whitened_pushforward,
            ) {
                None => {
                    invalid_sample_count = invalid_sample_count.saturating_add(1);
                    sample_relative_defects.push(None);
                    continue;
                }
                Some(None) => {
                    uninformative_sample_count = uninformative_sample_count.saturating_add(1);
                    sample_relative_defects.push(None);
                    continue;
                }
                Some(Some(value)) => value,
            };

            informative_sample_count = informative_sample_count.saturating_add(1);
            residual_sum += relative_defect;
            maximum_relative_defect = maximum_relative_defect.max(relative_defect);
            sample_relative_defects.push(Some(relative_defect));
        }

        let mean_relative_defect = if informative_sample_count == 0 {
            0.0
        } else {
            residual_sum / informative_sample_count as f64
        };
        generator_results.push(MetricAwareVectorFieldGeneratorResultV1 {
            generator_id: generator.id.clone(),
            sample_count: sample_states.len(),
            informative_sample_count,
            uninformative_sample_count,
            invalid_sample_count,
            maximum_relative_defect,
            mean_relative_defect,
            sample_relative_defects,
        });
    }

    Ok(MetricAwareVectorFieldEquivarianceReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        action_declaration: declaration.clone(),
        original_action_declaration: action.declared_action().clone(),
        action_admission_profile: *action.profile(),
        action_admission_diagnostics: action.diagnostics().clone(),
        coordinate_names: declaration.coordinate_names.clone(),
        coordinate_unit_ids: declaration.coordinate_unit_ids.clone(),
        vector_field_display: vector_field.iter().map(ToString::to_string).collect(),
        vector_field_canonical: vector_field.iter().map(canonical_expression).collect(),
        total_expression_complexity,
        profile,
        fixed_bindings: fixed_bindings
            .iter()
            .map(|(name, value)| ((*name).to_string(), *value))
            .collect(),
        sample_states: sample_states.to_vec(),
        generator_results,
    })
}

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

    fn metric_aware_so2_action() -> crate::linear_group_action::AdmittedLinearGroupActionV1 {
        crate::linear_group_action::LinearGroupActionV1 {
            action_id: "sample-so2".to_string(),
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

    fn metric_sample_states() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![2.0, -1.0],
            vec![-0.5, 1.5],
            vec![3.0, 0.25],
        ]
    }

    fn output_representation(
        representation_id: &str,
        component_names: &[&str],
        component_unit_ids: &[&str],
        generator_id: &str,
        matrix: Vec<Vec<f64>>,
    ) -> LinearVectorFieldOutputRepresentationV1 {
        LinearVectorFieldOutputRepresentationV1 {
            representation_id: representation_id.to_string(),
            coordinate_basis_id: "output-basis".to_string(),
            component_names: component_names.iter().map(|name| name.to_string()).collect(),
            component_unit_ids: component_unit_ids.iter().map(|unit| unit.to_string()).collect(),
            generators: vec![crate::linear_group_action::LinearGeneratorV1 {
                id: generator_id.to_string(),
                matrix,
            }],
        }
    }



    fn rotational_vector_field() -> Vec<Expr> {
        vec![
            mul(Expr::Const(-1.0), var("y")),
            var("x"),
        ]
    }

    fn radial_rotation_vector_field() -> Vec<Expr> {
        let radial_squared = add(pow2("x"), pow2("y"));
        let radial_scale = Expr::Func(UnaryFn::Sin, Box::new(radial_squared));
        vec![
            mul(Expr::Const(-1.0), mul(radial_scale.clone(), var("y"))),
            mul(radial_scale, var("x")),
        ]
    }

    #[test]
    fn exact_vector_field_equivariance_accepts_state_and_output_representation_match() {
        let report = check_exact_vector_field_equivariance(
            &rotational_vector_field(),
            &metric_aware_so2_action(),
            &output_representation(
                "same-state-output",
                &["vx", "vy"],
                &["m/s", "m/s"],
                "J",
                vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            ),
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(report.generator_results[0].component_residual_term_counts, vec![0, 0]);
        assert_eq!(report.output_representation.component_names, vec!["vx", "vy"]);
    }

    #[test]
    fn exact_vector_field_equivariance_accepts_twice_angle_complex_square_output() {
        // F(x,y) = (x^2-y^2, 2xy) transforms under B=2J when the
        // state transforms under A=J. This is a concrete A != B fixture.
        let vector_field = vec![
            sub(pow2("x"), pow2("y")),
            mul(Expr::Const(2.0), mul(var("x"), var("y"))),
        ];
        let output = output_representation(
            "weight-two-output",
            &["real", "imag"],
            &["1", "1"],
            "J",
            vec![vec![0.0, -2.0], vec![2.0, 0.0]],
        );
        let report = check_exact_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &output,
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(report.generator_results[0].component_residual_term_counts, vec![0, 0]);
    }

    #[test]
    fn exact_vector_field_equivariance_rejects_wrong_output_representation() {
        let vector_field = vec![
            sub(pow2("x"), pow2("y")),
            mul(Expr::Const(2.0), mul(var("x"), var("y"))),
        ];
        let wrong_output = output_representation(
            "wrong-weight-output",
            &["real", "imag"],
            &["1", "1"],
            "J",
            vec![vec![0.0, -1.0], vec![1.0, 0.0]],
        );
        let report = check_exact_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &wrong_output,
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(!report.equivariant_under_every_generator());
        assert!(report.generator_results[0]
            .component_residual_term_counts
            .iter()
            .any(|count| *count > 0));
    }

    #[test]
    fn exact_vector_field_equivariance_supports_scalar_output_with_distinct_dimension() {
        let scalar_output = vec![add(pow2("x"), pow2("y"))];
        let output = output_representation(
            "scalar-radial-output",
            &["radius_squared"],
            &["m^2"],
            "J",
            vec![vec![0.0]],
        );
        let report = check_exact_vector_field_equivariance(
            &scalar_output,
            &metric_aware_so2_action(),
            &output,
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(report.generator_results[0].component_residual_term_counts, vec![0]);
    }

    #[test]
    fn exact_vector_field_equivariance_keeps_unacted_parameters_symbolic() {
        let vector_field = vec![
            mul(var("rate"), sub(pow2("x"), pow2("y"))),
            mul(Expr::Const(2.0), mul(var("rate"), mul(var("x"), var("y")))),
        ];
        let output = output_representation(
            "weighted-complex-square",
            &["real", "imag"],
            &["1", "1"],
            "J",
            vec![vec![0.0, -2.0], vec![2.0, 0.0]],
        );
        let report = check_exact_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &output,
            ExactPolynomialLieProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert!(report.expression_variables.contains(&"rate".to_string()));
    }

    #[test]
    fn exact_vector_field_equivariance_rejects_bad_output_pairing_and_budget() {
        let vector_field = vec![var("x"), var("y")];
        let missing = output_representation(
            "missing-generator",
            &["vx", "vy"],
            &["1", "1"],
            "other-generator",
            vec![vec![0.0, -1.0], vec![1.0, 0.0]],
        );
        assert_eq!(
            check_exact_vector_field_equivariance(
                &vector_field,
                &metric_aware_so2_action(),
                &missing,
                ExactPolynomialLieProfileV1::default(),
            ),
            Err(ExactVectorFieldEquivarianceErrorV1::MissingOutputGenerator {
                generator_id: "J".to_string(),
            })
        );

        let output = output_representation(
            "budget-limited",
            &["vx", "vy"],
            &["1", "1"],
            "J",
            vec![vec![0.0, -1.0], vec![1.0, 0.0]],
        );
        let mut limited = ExactPolynomialLieProfileV1::default();
        limited.maximum_term_products = 1;
        assert_eq!(
            check_exact_vector_field_equivariance(
                &vector_field,
                &metric_aware_so2_action(),
                &output,
                limited,
            ),
            Err(ExactVectorFieldEquivarianceErrorV1::ResourceLimitExceeded)
        );
    }

    #[test]
    fn vector_field_equivariance_accepts_linear_and_radially_scaled_rotation_fields() {
        for vector_field in [
            rotational_vector_field(),
            radial_rotation_vector_field(),
        ] {
            let report = evaluate_metric_aware_vector_field_equivariance(
                &vector_field,
                &metric_aware_so2_action(),
                &metric_sample_states(),
                &[],
                MetricAwareVectorFieldEquivarianceProfileV1::default(),
            )
            .unwrap();

            assert!(report.equivariant_under_every_generator());
            assert_eq!(report.generator_results[0].invalid_sample_count, 0);
            assert_eq!(
                report.generator_results[0].informative_sample_count,
                metric_sample_states().len()
            );
            assert!(report.generator_results[0].maximum_relative_defect < 1.0e-14);
        }
    }

    #[test]
    fn vector_field_equivariance_rejects_anisotropic_dynamics() {
        let vector_field = vec![
            mul(Expr::Const(2.0), var("x")),
            var("y"),
        ];
        let report = evaluate_metric_aware_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();

        assert!(!report.equivariant_under_every_generator());
        assert!(report.generator_results[0].maximum_relative_defect > 1.0e-2);
    }

    #[test]
    fn vector_field_equivariance_is_covariant_under_nonorthogonal_coordinate_change() {
        // D = [[1,1],[0,1]], A' = D A D^-1, G' = D^-T G D^-1.
        // F = sin(x²+y²) Jx becomes F' = sin(X²-2XY+2Y²) A'X.
        let transform = nalgebra::Matrix2::new(1.0, 1.0, 0.0, 1.0);
        let inverse = transform.try_inverse().unwrap();
        let generator = nalgebra::Matrix2::new(0.0, -1.0, 1.0, 0.0);
        let transformed_generator = transform * generator * inverse;
        let transformed_metric = inverse.transpose() * inverse;
        let transformed_action = crate::linear_group_action::LinearGroupActionV1 {
            action_id: "sample-so2-sheared-vector-field".to_string(),
            coordinate_basis_id: "XY".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: transformed_metric
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect(),
            generators: vec![crate::linear_group_action::LinearGeneratorV1 {
                id: "J-prime".to_string(),
                matrix: transformed_generator
                    .row_iter()
                    .map(|row| row.iter().copied().collect())
                    .collect(),
            }],
        }
        .try_admit(crate::linear_group_action::LinearGroupActionAdmissionProfileV1::default())
        .unwrap();

        let base_states = metric_sample_states();
        let transformed_states: Vec<Vec<f64>> = base_states
            .iter()
            .map(|state| {
                let transformed = transform * nalgebra::Vector2::new(state[0], state[1]);
                vec![transformed[0], transformed[1]]
            })
            .collect();

        let base = evaluate_metric_aware_vector_field_equivariance(
            &radial_rotation_vector_field(),
            &metric_aware_so2_action(),
            &base_states,
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();

        let radial_form = sub(
            add(pow2("X"), mul(Expr::Const(2.0), pow2("Y"))),
            mul(Expr::Const(2.0), mul(var("X"), var("Y"))),
        );
        let radial_scale = Expr::Func(UnaryFn::Sin, Box::new(radial_form));
        let transformed_field = vec![
            mul(
                radial_scale.clone(),
                sub(var("X"), mul(Expr::Const(2.0), var("Y"))),
            ),
            mul(radial_scale, sub(var("X"), var("Y"))),
        ];
        let transformed = evaluate_metric_aware_vector_field_equivariance(
            &transformed_field,
            &transformed_action,
            &transformed_states,
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();

        assert!(base.equivariant_under_every_generator());
        assert!(transformed.equivariant_under_every_generator());
        assert!(
            (base.generator_results[0].maximum_relative_defect
                - transformed.generator_results[0].maximum_relative_defect)
                .abs()
                < 1.0e-12
        );
        for (base_sample, transformed_sample) in base.generator_results[0]
            .sample_relative_defects
            .iter()
            .zip(&transformed.generator_results[0].sample_relative_defects)
        {
            assert!((base_sample.unwrap() - transformed_sample.unwrap()).abs() < 1.0e-12);
        }
    }

    #[test]
    fn vector_field_equivariance_holds_with_fixed_parameters() {
        let vector_field = vec![
            mul(Expr::Const(-1.0), mul(var("rate"), var("y"))),
            mul(var("rate"), var("x")),
        ];
        let report = evaluate_metric_aware_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[("rate", 2.5)],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(report.fixed_bindings, vec![("rate".to_string(), 2.5)]);
    }

    #[test]
    fn vector_field_equivariance_abstains_on_zero_fields_and_invalid_domains() {
        let zero_field = vec![Expr::Const(0.0), Expr::Const(0.0)];
        let zero_report = evaluate_metric_aware_vector_field_equivariance(
            &zero_field,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();
        assert!(!zero_report.equivariant_under_every_generator());
        assert_eq!(
            zero_report.generator_results[0].uninformative_sample_count,
            metric_sample_states().len()
        );

        let invalid_field = vec![
            Expr::Func(UnaryFn::Sqrt, Box::new(var("x"))),
            var("y"),
        ];
        let invalid_report = evaluate_metric_aware_vector_field_equivariance(
            &invalid_field,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();
        assert!(!invalid_report.equivariant_under_every_generator());
        assert_eq!(invalid_report.generator_results[0].invalid_sample_count, 1);
    }

    #[test]
    fn vector_field_equivariance_relative_defect_is_stable_at_large_vector_scales() {
        let scale = 4.0e307;
        let vector_field = vec![
            mul(Expr::Const(scale), var("x")),
            mul(Expr::Const(scale), var("y")),
        ];
        let report = evaluate_metric_aware_vector_field_equivariance(
            &vector_field,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareVectorFieldEquivarianceProfileV1::default(),
        )
        .unwrap();

        assert!(report.equivariant_under_every_generator());
        assert_eq!(report.generator_results[0].invalid_sample_count, 0);
        assert!(report.generator_results[0].maximum_relative_defect < 1.0e-14);
    }

    #[test]
    fn vector_field_equivariance_rejects_unsupported_inputs_and_dimension_mismatches() {
        let unsupported = vec![
            Expr::Func(UnaryFn::Abs, Box::new(var("x"))),
            var("y"),
        ];
        assert_eq!(
            evaluate_metric_aware_vector_field_equivariance(
                &unsupported,
                &metric_aware_so2_action(),
                &metric_sample_states(),
                &[],
                MetricAwareVectorFieldEquivarianceProfileV1::default(),
            ),
            Err(MetricAwareVectorFieldEquivarianceErrorV1::UnsupportedExpression)
        );

        assert_eq!(
            evaluate_metric_aware_vector_field_equivariance(
                &[var("x")],
                &metric_aware_so2_action(),
                &metric_sample_states(),
                &[],
                MetricAwareVectorFieldEquivarianceProfileV1::default(),
            ),
            Err(MetricAwareVectorFieldEquivarianceErrorV1::VectorFieldDimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn metric_aware_sampler_accepts_exact_radial_invariant_and_retains_receipt() {
        let report = evaluate_metric_aware_linear_action_samples(
            &add(pow2("x"), pow2("y")),
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();

        assert!(report.invariant_under_every_generator());
        assert_eq!(report.sample_states, metric_sample_states());
        assert_eq!(report.coordinate_unit_ids, vec!["1".to_string(), "1".to_string()]);
        assert!(report.expression_canonical.starts_with("symthaea-expr-v1:BADD("));
        assert_eq!(report.action_declaration.action_id, "sample-so2");
        assert_eq!(report.original_action_declaration.action_id, "sample-so2");
        assert_eq!(report.total_invalid_evaluations(), 0);
        assert_eq!(report.total_informative_evaluations(), 4);
        assert!(report.generator_results[0].maximum_squared_residual < 1e-28);
    }

    #[test]
    fn analytic_and_directional_derivative_witnesses_agree_on_frozen_smooth_samples() {
        use crate::linear_action_sampling::{
            LinearActionSampleV1, LinearActionSamplingProfileV1, evaluate_linear_action_samples,
        };

        let samples = metric_sample_states();
        let directional_samples: Vec<LinearActionSampleV1> = samples
            .iter()
            .enumerate()
            .map(|(index, state)| LinearActionSampleV1 {
                sample_id: format!("dual-witness-{index}"),
                state: state.clone(),
                direction_basis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            })
            .collect();
        let mut directional_profile = LinearActionSamplingProfileV1::default();
        directional_profile.minimum_informative_samples = samples.len();
        directional_profile.maximum_samples = 16;
        directional_profile.squared_cosine_tolerance = 1e-8;

        let controls = [
            (
                Expr::Func(UnaryFn::Sqrt, Box::new(add(pow2("x"), pow2("y")))),
                true,
            ),
            (
                Expr::Func(
                    UnaryFn::Sqrt,
                    Box::new(add(pow2("x"), mul(Expr::Const(2.0), pow2("y")))),
                ),
                false,
            ),
            (
                Expr::Func(UnaryFn::Sin, Box::new(add(pow2("x"), pow2("y")))),
                true,
            ),
        ];

        for (expression, expected_invariant) in controls {
            let action = metric_aware_so2_action();
            let analytic = evaluate_metric_aware_linear_action_samples(
                &expression,
                &action,
                &samples,
                &[],
                MetricAwareLinearActionSampleProfileV1::default(),
            )
            .unwrap();
            let directional = evaluate_linear_action_samples(
                &expression,
                &action,
                &directional_samples,
                directional_profile,
            )
            .unwrap();

            assert_eq!(analytic.invariant_under_every_generator(), expected_invariant);
            assert_eq!(directional.invariant_within_tolerance(), expected_invariant);
            assert_eq!(
                analytic.generator_results[0].informative_sample_count,
                directional.generator_statistics[0].informative_sample_count
            );
            for (analytic_index, directional_residual) in directional.generator_statistics[0]
                .sample_residuals
                .iter()
                .enumerate()
            {
                let analytic_per_sample = if analytic_index < samples.len() {
                    // The analytic report contains aggregate per-generator
                    // statistics; the finite-difference path retains per-sample
                    // residuals. Re-run AD on this singleton to pair the witnesses.
                    let one_sample = vec![samples[analytic_index].clone()];
                    let one = evaluate_metric_aware_linear_action_samples(
                        &expression,
                        &action,
                        &one_sample,
                        &[],
                        MetricAwareLinearActionSampleProfileV1::default(),
                    )
                    .unwrap();
                    one.generator_results[0].maximum_squared_residual
                } else {
                    unreachable!("paired sample index is in range")
                };
                assert!(
                    (analytic_per_sample
                        - directional_residual.normalized_squared_residual.unwrap())
                        .abs()
                        < 2.0e-7,
                    "analytic and directional residuals disagree at sample {analytic_index}"
                );
            }
        }
    }

    #[test]
    fn metric_aware_sampler_supports_composed_smooth_unary_functions() {
        let radius = sqrt_of(add(pow2("x"), pow2("y")));
        let expression = Expr::Func(UnaryFn::Sin, Box::new(radius));
        let report = evaluate_metric_aware_linear_action_samples(
            &expression,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert!(report.invariant_under_every_generator());
        assert!(report.generator_results[0].maximum_squared_residual < 1e-26);
    }

    #[test]
    fn metric_aware_receipt_expression_identity_preserves_exact_constant_bits() {
        let first = Expr::BinOp(
            BinOp::Mul,
            Box::new(var("x")),
            Box::new(Expr::Const(1.0000001)),
        );
        let second = Expr::BinOp(
            BinOp::Mul,
            Box::new(var("x")),
            Box::new(Expr::Const(1.0000002)),
        );
        let first_report = evaluate_metric_aware_linear_action_samples(
            &first,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        let second_report = evaluate_metric_aware_linear_action_samples(
            &second,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert_eq!(first_report.expression_display, second_report.expression_display);
        assert_ne!(first_report.expression_canonical, second_report.expression_canonical);
    }

    #[test]
    fn metric_aware_sampler_rejects_anisotropic_expression() {
        let expression = add(pow2("x"), mul(Expr::Const(2.0), pow2("y")));
        let report = evaluate_metric_aware_linear_action_samples(
            &expression,
            &metric_aware_so2_action(),
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();

        assert!(!report.invariant_under_every_generator());
        assert!(report.generator_results[0].maximum_squared_residual > 1e-3);
    }

    #[test]
    fn metric_aware_sampler_is_covariant_under_nonorthogonal_coordinate_change() {
        // D = [[1,1],[0,1]], A' = D A D^-1, G' = D^-T D^-1.
        // The transformed scalar is f'(X,Y) = X² - 2XY + 2Y².
        let transformed_action = crate::linear_group_action::LinearGroupActionV1 {
            action_id: "sample-so2-sheared".to_string(),
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
        let base_samples = metric_sample_states();
        let transformed_samples: Vec<Vec<f64>> = base_samples
            .iter()
            .map(|state| vec![state[0] + state[1], state[1]])
            .collect();
        let base_report = evaluate_metric_aware_linear_action_samples(
            &add(pow2("x"), pow2("y")),
            &metric_aware_so2_action(),
            &base_samples,
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        let transformed_expression = sub(
            add(pow2("X"), mul(Expr::Const(2.0), pow2("Y"))),
            mul(Expr::Const(2.0), mul(var("X"), var("Y"))),
        );
        let transformed_report = evaluate_metric_aware_linear_action_samples(
            &transformed_expression,
            &transformed_action,
            &transformed_samples,
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();

        assert!(base_report.invariant_under_every_generator());
        assert!(transformed_report.invariant_under_every_generator());
        let base_residual = base_report.generator_results[0].maximum_squared_residual;
        let transformed_residual =
            transformed_report.generator_results[0].maximum_squared_residual;
        assert!(
            (base_residual - transformed_residual).abs() < 1e-24,
            "metric-aware residual changed under basis transform: {base_residual} vs {transformed_residual}"
        );

        // Repeat the metamorphic comparison for a non-invariant quadratic:
        // x² + 2y² transforms to X² - 2XY + 3Y².
        let base_anisotropic = add(pow2("x"), mul(Expr::Const(2.0), pow2("y")));
        let transformed_anisotropic = sub(
            add(pow2("X"), mul(Expr::Const(3.0), pow2("Y"))),
            mul(Expr::Const(2.0), mul(var("X"), var("Y"))),
        );
        let base_negative = evaluate_metric_aware_linear_action_samples(
            &base_anisotropic,
            &metric_aware_so2_action(),
            &base_samples,
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        let transformed_negative = evaluate_metric_aware_linear_action_samples(
            &transformed_anisotropic,
            &transformed_action,
            &transformed_samples,
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert!(!base_negative.invariant_under_every_generator());
        assert!(!transformed_negative.invariant_under_every_generator());
        let base_negative_residual =
            base_negative.generator_results[0].maximum_squared_residual;
        let transformed_negative_residual =
            transformed_negative.generator_results[0].maximum_squared_residual;
        assert!(
            (base_negative_residual - transformed_negative_residual).abs() < 1e-12,
            "negative-control residual changed under basis transform: {base_negative_residual} vs {transformed_negative_residual}"
        );
    }

    #[test]
    fn fixed_parameters_are_constant_when_differentiating_the_action_coordinates() {
        // z is a fixed parameter used as the exponent. It is not part of the
        // action and therefore its AD component must be zero in the directional
        // derivative contract, even though z appears in the AST.
        let action = metric_aware_so2_action();
        let expression = Expr::BinOp(
            BinOp::Pow,
            Box::new(var("x")),
            Box::new(var("z")),
        );
        let report = evaluate_metric_aware_linear_action_samples(
            &expression,
            &action,
            &[vec![-2.0, 1.0], vec![-1.5, 0.75], vec![-1.0, 2.0]],
            &[("z", 3.0)],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();

        assert_eq!(report.generator_results[0].invalid_sample_count, 0);
        assert_eq!(report.generator_results[0].informative_sample_count, 3);
        assert!(!report.invariant_under_every_generator());
    }

    #[test]
    fn metric_aware_sampler_supports_fixed_parameters_and_requires_them() {
        let expression = mul(var("z"), add(pow2("x"), pow2("y")));
        let action = metric_aware_so2_action();
        let missing = evaluate_metric_aware_linear_action_samples(
            &expression,
            &action,
            &metric_sample_states(),
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        );
        assert_eq!(
            missing,
            Err(MetricAwareLinearActionSampleErrorV1::MissingFixedBinding {
                name: "z".to_string()
            })
        );

        let report = evaluate_metric_aware_linear_action_samples(
            &expression,
            &action,
            &metric_sample_states(),
            &[("z", 3.0)],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert!(report.invariant_under_every_generator());
        assert_eq!(report.fixed_bindings, vec![("z".to_string(), 3.0)]);
    }

    #[test]
    fn metric_aware_sampler_abstains_on_invalid_and_uninformative_samples() {
        let expression = Expr::Func(UnaryFn::Sqrt, Box::new(var("x")));
        let report = evaluate_metric_aware_linear_action_samples(
            &expression,
            &metric_aware_so2_action(),
            &[vec![-1.0, 1.0], vec![1.0, 0.0]],
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert!(!report.invariant_under_every_generator());
        assert_eq!(report.generator_results[0].invalid_sample_count, 1);

        let radial = add(pow2("x"), pow2("y"));
        let report = evaluate_metric_aware_linear_action_samples(
            &radial,
            &metric_aware_so2_action(),
            &[vec![0.0, 0.0], vec![1.0, 0.0]],
            &[],
            MetricAwareLinearActionSampleProfileV1::default(),
        )
        .unwrap();
        assert_eq!(report.generator_results[0].uninformative_sample_count, 1);
        assert_eq!(report.generator_results[0].informative_sample_count, 1);
    }

    #[test]
    fn metric_aware_sampler_rejects_unsupported_nondifferentiable_ast() {
        let expression = Expr::Func(UnaryFn::Abs, Box::new(var("x")));
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &metric_sample_states(),
                &[],
                MetricAwareLinearActionSampleProfileV1::default(),
            ),
            Err(MetricAwareLinearActionSampleErrorV1::UnsupportedExpression)
        );
    }

    #[test]
    fn metric_aware_sampler_rejects_profiles_above_hard_resource_ceilings() {
        let expression = add(pow2("x"), pow2("y"));
        let states = metric_sample_states();

        let too_many_samples = MetricAwareLinearActionSampleProfileV1 {
            maximum_samples: HARD_MAX_METRIC_AWARE_SAMPLES + 1,
            ..MetricAwareLinearActionSampleProfileV1::default()
        };
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                too_many_samples
            ),
            Err(MetricAwareLinearActionSampleErrorV1::InvalidProfile)
        );

        let too_many_nodes = MetricAwareLinearActionSampleProfileV1 {
            maximum_expression_nodes: HARD_MAX_METRIC_AWARE_EXPRESSION_NODES + 1,
            ..MetricAwareLinearActionSampleProfileV1::default()
        };
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                too_many_nodes
            ),
            Err(MetricAwareLinearActionSampleErrorV1::InvalidProfile)
        );

        let too_much_work = MetricAwareLinearActionSampleProfileV1 {
            maximum_total_gradient_work: HARD_MAX_METRIC_AWARE_GRADIENT_WORK + 1,
            ..MetricAwareLinearActionSampleProfileV1::default()
        };
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                too_much_work
            ),
            Err(MetricAwareLinearActionSampleErrorV1::InvalidProfile)
        );

        let too_many_pairs = MetricAwareLinearActionSampleProfileV1 {
            maximum_total_generator_sample_pairs:
                HARD_MAX_METRIC_AWARE_GENERATOR_SAMPLE_PAIRS + 1,
            ..MetricAwareLinearActionSampleProfileV1::default()
        };
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                too_many_pairs
            ),
            Err(MetricAwareLinearActionSampleErrorV1::InvalidProfile)
        );
    }

    #[test]
    fn metric_aware_sampler_enforces_hard_action_dimension_and_generator_limits() {
        use crate::linear_group_action::{
            LinearGeneratorV1, LinearGroupActionAdmissionProfileV1, LinearGroupActionV1,
        };

        let dimension = HARD_MAX_METRIC_AWARE_ACTION_DIMENSION + 1;
        let mut generator = vec![vec![0.0_f64; dimension]; dimension];
        generator[0][1] = -1.0;
        generator[1][0] = 1.0;
        let large_dimension_action = LinearGroupActionV1 {
            action_id: "metric-sampler-over-dimension-cap".to_string(),
            coordinate_basis_id: "large-basis".to_string(),
            coordinate_names: (0..dimension).map(|index| format!("x{index}")).collect(),
            coordinate_unit_ids: vec!["1".to_string(); dimension],
            metric: (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|column| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
            generators: vec![LinearGeneratorV1 {
                id: "J".to_string(),
                matrix: generator,
            }],
        }
        .try_admit(LinearGroupActionAdmissionProfileV1 {
            maximum_dimension: dimension,
            ..LinearGroupActionAdmissionProfileV1::default()
        })
        .unwrap();
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &Expr::Var("x0".to_string()),
                &large_dimension_action,
                &[vec![1.0; dimension]],
                &[],
                MetricAwareLinearActionSampleProfileV1::default(),
            ),
            Err(MetricAwareLinearActionSampleErrorV1::ActionDimensionLimitExceeded {
                actual: dimension,
                maximum: HARD_MAX_METRIC_AWARE_ACTION_DIMENSION,
            })
        );

        let too_many_generators = LinearGroupActionV1 {
            action_id: "metric-sampler-over-generator-cap".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: (0..=HARD_MAX_METRIC_AWARE_ACTION_GENERATORS)
                .map(|index| LinearGeneratorV1 {
                    id: format!("J-{index}"),
                    matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
                })
                .collect(),
        }
        .try_admit(LinearGroupActionAdmissionProfileV1 {
            maximum_generators: HARD_MAX_METRIC_AWARE_ACTION_GENERATORS + 1,
            ..LinearGroupActionAdmissionProfileV1::default()
        })
        .unwrap();
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &add(pow2("x"), pow2("y")),
                &too_many_generators,
                &metric_sample_states(),
                &[],
                MetricAwareLinearActionSampleProfileV1::default(),
            ),
            Err(MetricAwareLinearActionSampleErrorV1::GeneratorCountLimitExceeded {
                actual: HARD_MAX_METRIC_AWARE_ACTION_GENERATORS + 1,
                maximum: HARD_MAX_METRIC_AWARE_ACTION_GENERATORS,
            })
        );
    }

    #[test]
    fn metric_aware_sampler_rejects_summation_before_fixed_binding_resolution() {
        let expression = Expr::Sum(Box::new(var("x")), "k".to_string());
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &metric_sample_states(),
                &[],
                MetricAwareLinearActionSampleProfileV1::default(),
            ),
            Err(MetricAwareLinearActionSampleErrorV1::UnsupportedExpression)
        );
    }

    #[test]
    fn metric_aware_sampler_enforces_sample_and_work_budgets() {
        let expression = add(pow2("x"), pow2("y"));
        let states = metric_sample_states();
        let mut too_few = MetricAwareLinearActionSampleProfileV1::default();
        too_few.maximum_samples = 2;
        assert_eq!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                too_few
            ),
            Err(MetricAwareLinearActionSampleErrorV1::TooManySamples {
                actual: 4,
                maximum: 2
            })
        );

        let mut low_work = MetricAwareLinearActionSampleProfileV1::default();
        low_work.maximum_total_gradient_work = 1;
        assert!(matches!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                low_work
            ),
            Err(MetricAwareLinearActionSampleErrorV1::TotalGradientWorkExceeded { .. })
        ));

        let mut low_pair_budget = MetricAwareLinearActionSampleProfileV1::default();
        low_pair_budget.maximum_total_generator_sample_pairs = 2;
        assert!(matches!(
            evaluate_metric_aware_linear_action_samples(
                &expression,
                &metric_aware_so2_action(),
                &states,
                &[],
                low_pair_budget
            ),
            Err(MetricAwareLinearActionSampleErrorV1::GeneratorSampleBudgetExceeded { .. })
        ));
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
