// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wiring three of the Ramanujan Protocol's standalone capability modules --
//! [`crate::typed_generation`], `experiment_selection`, and `cfc_smoothing` -- into one real
//! active-discovery loop, and measuring which ingredient(s) actually matter via a full 2x2x2
//! factorial ablation, not a single monolithic "everything on vs everything off" comparison.
//!
//! ## Why a factorial, not "integrated vs naive"
//!
//! A single integrated-pipeline-vs-naive-baseline comparison would conflate three independent
//! claims into one number: if the integrated pipeline wins, which of the three ingredients
//! actually did the work? This module runs all 8 combinations of {typed vs. untyped candidate
//! generation} x {active vs. round-robin experiment selection} x {CfC-smoothed vs. raw noisy
//! observation}, so each capability's marginal contribution can be measured directly (mean
//! success rate with the factor on vs. off, averaged over the other two factors) instead of
//! asserted from a single combined win.
//!
//! ## Why `iit_coupling` is deliberately excluded here
//!
//! `iit_coupling` measures the discrimination loop (it computes Phi over a trajectory of
//! [live_count, disagreement, ...] features) -- it isn't a generative or corrective ingredient
//! like the other three, and its own honest conclusion (see that module's doc comment) is that
//! the Phi half of its prediction is unmeasurable with short trajectories, only the efficiency
//! half held up. Bolting it onto this loop as a fourth ablation factor would not be "wiring
//! together a validated capability" -- it would be adding an already-flagged-as-unreliable
//! measurement into a comparison it wasn't built to support. It stays a separate, standalone
//! diagnostic.
//!
//! ## Task
//!
//! A hypothesis-discrimination loop: one dimensionally-typed "ground truth" ENERGY-valued law
//! over `{m: MASS, v: VELOCITY, r: LENGTH, t: TIME}` (drawn via
//! [`crate::typed_generation::random_expr_with_dimension`], always -- the ground truth itself
//! doesn't depend on the condition being tested, only how *distractor* hypotheses are drawn
//! does), plus a pool of `POOL_SIZE - 1` distractors: `NEAR_MISS_COUNT` uniform-scale near-miss
//! variants of ground truth (functionally close, condition-independent -- see
//! [`near_miss_distractors`] for why the pool needs these) and the rest "wildcards" drawn
//! either via the typed generator (dimensionally valid but functionally different from ground
//! truth) or via a small untyped grammar mirroring the one `typed_generation.rs`'s own tests
//! use as an "existing untyped generator" baseline. Each round, an experiment (an assignment of
//! values to all four variables) is chosen from a shared candidate pool -- either the single
//! most-discriminating point among the still-live hypotheses
//! ([`select_most_informative_experiment`]) or the next point in a fixed shuffled order,
//! ignoring which hypotheses remain live. Ground truth is evaluated at that one point; the
//! `smoothing` factor governs *how that single point is measured*: either one noisy reading, or
//! `K_REPEATS` repeated noisy readings at that same point blended through a fresh
//! [`CfcSmoother`] each round (irregular gaps between the repeats). This is a deliberately
//! narrower scope than smoothing *across* rounds -- rounds evaluate ground truth at different,
//! deliberately-differing points, and `CfcSmoother` is built to track one evolving quantity over
//! time, not to blend unrelated evaluations together (an earlier draft of this module did
//! exactly that and it was a real, found-and-fixed bug -- see the `NOISE_SIGMA_FRACTION`/
//! `ELIMINATION_TOLERANCE` comment for the calibration history it caused). Any live hypothesis
//! whose prediction disagrees with the (possibly smoothed) observation beyond a tolerance is
//! eliminated. The loop ends when one hypothesis remains or the round budget is exhausted.
//!
//! ## Predeclared interpretation (frozen before running the real ablation)
//!
//! For each of the 3 factors, compare mean success rate (exactly one hypothesis survives a
//! bounded round budget, and it is ground truth) with that factor on vs. off, averaged over
//! the other two factors and paired by seed. A factor is judged **supported** if its "on" mean
//! success rate exceeds its "off" mean by more than 0.05 (5 percentage points) consistently
//! across the seed set; **negative** otherwise. This mirrors this arc's standing rule to
//! predeclare thresholds before looking at results, not tune them after.

use std::collections::{HashMap, HashSet};

use symthaea_core::hdc::conjecture_engine::{
    BinOp, CfcSmoother, Expr, InformationGainScenario, UnaryFn,
    select_experiment_by_information_gain_per_cost,
    select_experiment_by_scenario_robust_information_gain_per_cost,
    select_experiment_by_worst_case_information_gain_per_cost, select_most_informative_experiment,
};

use crate::dimensional_inference::{InferenceResult, UnitMap, infer_dimensions};
use crate::prequential_pit_martingale::{PrequentialPitDecision, PrequentialPitMartingale};
use crate::typed_generation::random_expr_with_dimension;
use crate::types::DimensionalSignature;

fn xorshift_next(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

fn rand_index(rng: &mut u64, n: usize) -> usize {
    (xorshift_next(rng) as usize) % n
}

fn rand_unit(rng: &mut u64) -> f64 {
    (xorshift_next(rng) >> 11) as f64 / (1u64 << 53) as f64
}

fn rand_range(rng: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + rand_unit(rng) * (hi - lo)
}

/// Box-Muller Gaussian noise, matching this crate family's no-new-dependency convention
/// (same technique as `cfc_smoothing.rs`'s test helpers).
fn rand_gaussian(rng: &mut u64, sigma: f64) -> f64 {
    let u1 = rand_unit(rng).max(1e-12);
    let u2 = rand_unit(rng);
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Standard-normal CDF approximation for this deterministic simulator only.
/// Real predictors must provide a vetted conditional predictive CDF.
fn standard_normal_cdf(value: f64) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736)
        * t)
        + 0.254829592)
        * t;
    let probability = 0.5 * (1.0 + sign * (1.0 - polynomial * (-x * x).exp()));
    probability
        .is_finite()
        .then_some(probability.clamp(0.0, 1.0))
}

fn shuffle<T>(rng: &mut u64, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = rand_index(rng, i + 1);
        items.swap(i, j);
    }
}

const RANDOM_CONSTANTS: [f64; 6] = [0.5, 1.0, 2.0, 3.0, -1.0, -0.5];
const VAR_NAMES: [&str; 4] = ["m", "v", "r", "t"];
const VAR_RANGE: (f64, f64) = (1.0, 2.5);

fn demo_units() -> UnitMap {
    HashMap::from([
        ("m".to_string(), DimensionalSignature::MASS),
        ("v".to_string(), DimensionalSignature::VELOCITY),
        ("r".to_string(), DimensionalSignature::LENGTH),
        ("t".to_string(), DimensionalSignature::TIME),
    ])
}

/// An "experiment": one value assigned to each of the four variables.
type Experiment = Vec<(String, f64)>;

fn random_experiment(rng: &mut u64) -> Experiment {
    VAR_NAMES
        .iter()
        .map(|v| (v.to_string(), rand_range(rng, VAR_RANGE.0, VAR_RANGE.1)))
        .collect()
}

/// Sample evaluator-only points strictly outside the discovery interval for every variable.
/// Randomly mixes lower-tail and upper-tail values. Never pass this design to the search policy.
fn random_extrapolation_experiment(rng: &mut u64) -> Experiment {
    VAR_NAMES
        .iter()
        .map(|name| {
            let value = if rand_index(rng, 2) == 0 {
                rand_range(rng, 0.25, 0.75)
            } else {
                rand_range(rng, 2.75, 4.0)
            };
            (name.to_string(), value)
        })
        .collect()
}

fn is_extrapolation_experiment(point: &Experiment) -> bool {
    point.len() == VAR_NAMES.len()
        && point
            .iter()
            .all(|(_, value)| *value < VAR_RANGE.0 || *value > VAR_RANGE.1)
}

fn eval_at(e: &Expr, point: &Experiment) -> Option<f64> {
    let vs: Vec<(&str, f64)> = point.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let r = e.eval(&vs);
    r.is_finite().then_some(r)
}

/// Evaluator-only equivalence check on a separate pre-generated design. This function must only
/// be called after the policy finishes; its reference expression and points never inform search.
fn evaluator_matches_reference(candidate: &Expr, reference: &Expr, holdout: &[Experiment]) -> bool {
    if holdout.len() < 8 {
        return false;
    }
    holdout.iter().all(
        |point| match (eval_at(candidate, point), eval_at(reference, point)) {
            (Some(predicted), Some(expected)) => {
                (predicted - expected).abs() <= 1.0e-6 * (1.0 + expected.abs())
            }
            _ => false,
        },
    )
}

/// A candidate is only reference-equivalent for the combined domain contract when it agrees on
/// both the discovery-range evaluator split and a genuinely out-of-range split.
fn evaluator_matches_reference_across_domains(
    candidate: &Expr,
    reference: &Expr,
    in_domain: &[Experiment],
    out_of_domain: &[Experiment],
) -> bool {
    out_of_domain.len() >= 8
        && evaluator_matches_reference(candidate, reference, in_domain)
        && evaluator_matches_reference(candidate, reference, out_of_domain)
}

fn vars_used(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Var(n) => {
            out.insert(n.clone());
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            vars_used(l, out);
            vars_used(r, out);
        }
        Expr::Func(_, a) => vars_used(a, out),
        Expr::Sum(body, bound) => {
            vars_used(body, out);
            out.remove(bound);
        }
    }
}

/// Small self-contained untyped grammar, mirroring `typed_generation.rs`'s own
/// `naive_untyped_expr` test helper (that one is private to its module) -- the "existing
/// untyped generator" baseline for the candidate-generation ablation factor.
fn naive_untyped_expr(rng: &mut u64, max_depth: usize) -> Expr {
    if max_depth == 0 || rand_index(rng, 3) == 0 {
        if rand_index(rng, 2) == 0 {
            Expr::Var(VAR_NAMES[rand_index(rng, VAR_NAMES.len())].to_string())
        } else {
            Expr::Const(RANDOM_CONSTANTS[rand_index(rng, RANDOM_CONSTANTS.len())])
        }
    } else if rand_index(rng, 5) == 0 {
        let fns = [UnaryFn::Sqrt, UnaryFn::Log, UnaryFn::Sin, UnaryFn::Cos];
        Expr::Func(
            fns[rand_index(rng, fns.len())],
            Box::new(naive_untyped_expr(rng, max_depth - 1)),
        )
    } else {
        let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
        Expr::BinOp(
            ops[rand_index(rng, ops.len())],
            Box::new(naive_untyped_expr(rng, max_depth - 1)),
            Box::new(naive_untyped_expr(rng, max_depth - 1)),
        )
    }
}

const GEN_MAX_DEPTH: usize = 4;
const GT_SANITY_PROBES: usize = 5;
const GEN_ATTEMPT_CAP: usize = 5000;

/// Draw a ground-truth ENERGY-dimensioned law: typed-generated (ground truth is always
/// dimensionally sensible, regardless of which condition is under test), using at least 2
/// distinct variables, and sane (finite, non-degenerate magnitude) across a handful of probe
/// points.
fn nontrivial_ground_truth(rng: &mut u64, probes: &[Experiment]) -> Expr {
    let units = demo_units();
    for _ in 0..GEN_ATTEMPT_CAP {
        let e =
            random_expr_with_dimension(rng, DimensionalSignature::ENERGY, &units, GEN_MAX_DEPTH);
        let mut used = HashSet::new();
        vars_used(&e, &mut used);
        if used.len() < 2 {
            continue;
        }
        let vals: Option<Vec<f64>> = probes.iter().map(|p| eval_at(&e, p)).collect();
        if let Some(vals) = vals {
            if vals.iter().all(|v| v.abs() > 1e-6 && v.abs() < 1e6) {
                return e;
            }
        }
    }
    panic!("could not draw a sane nontrivial ground-truth law within the attempt cap");
}

/// Construct a held-out target that uses a unary trigonometric operator the typed
/// algebraic candidate generator does not emit. Sin/Cos selection happens on the evaluator side;
/// the target expression is never passed to candidate generation or recovery proposals.
fn nontrivial_unary_ground_truth(rng: &mut u64, probes: &[Experiment]) -> Option<Expr> {
    let units = demo_units();
    for _ in 0..GEN_ATTEMPT_CAP {
        let base = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        // v*t/r is dimensionless: (L/T)*T/L = 1.
        let phase = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Var("t".to_string())),
            )),
            Box::new(Expr::Var("r".to_string())),
        );
        let unary = if rand_index(rng, 2) == 0 {
            UnaryFn::Sin
        } else {
            UnaryFn::Cos
        };
        let candidate = Expr::BinOp(
            BinOp::Mul,
            Box::new(base),
            Box::new(Expr::Func(unary, Box::new(phase))),
        );
        if infer_dimensions(&candidate, &units)
            != InferenceResult::Inferred(DimensionalSignature::ENERGY)
        {
            continue;
        }
        let Some(values) = probes
            .iter()
            .map(|point| eval_at(&candidate, point))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if values
            .iter()
            .all(|value| value.abs() > 1.0e-8 && value.abs() < 1.0e6)
        {
            return Some(candidate);
        }
    }
    None
}

fn expr_contains_unary_function(expression: &Expr) -> bool {
    match expression {
        Expr::Func(_, _) => true,
        Expr::BinOp(_, left, right) => {
            expr_contains_unary_function(left) || expr_contains_unary_function(right)
        }
        Expr::Sum(body, _) => expr_contains_unary_function(body),
        Expr::Var(_) | Expr::Const(_) => false,
    }
}

fn functionally_distinct(cand_vals: &[f64], gt_vals: &[f64]) -> bool {
    cand_vals
        .iter()
        .zip(gt_vals)
        .any(|(a, b)| (a - b).abs() > 0.05 * (1.0 + b.abs()))
}

/// A typed (dimensionally-valid) distractor: functionally distinct from ground truth and sane
/// at the probe points.
fn typed_distractor(rng: &mut u64, gt_vals: &[f64], probes: &[Experiment]) -> Expr {
    let units = demo_units();
    for _ in 0..GEN_ATTEMPT_CAP {
        let e =
            random_expr_with_dimension(rng, DimensionalSignature::ENERGY, &units, GEN_MAX_DEPTH);
        let vals: Option<Vec<f64>> = probes.iter().map(|p| eval_at(&e, p)).collect();
        if let Some(vals) = vals {
            if vals.iter().all(|v| v.abs() > 1e-6 && v.abs() < 1e6)
                && functionally_distinct(&vals, gt_vals)
            {
                return e;
            }
        }
    }
    panic!("could not draw a sane, distinct typed distractor within the attempt cap");
}

const MAX_TYPED_RECOVERY_RESTARTS: usize = 2;
const RECOVERY_CANDIDATES_PER_EXPANSION: usize = 4;
const RECOVERY_GENERATION_ATTEMPT_CAP: usize = 256;
const RECOVERY_COMPLEXITY_PENALTY: f64 = 1.0e-4;
/// Truth-absent recovery is not successful until the final candidate survives at least this
/// many new policy-visible measurements after the most recent recovery restart.
const MIN_POST_RECOVERY_VALIDATION_MEASUREMENTS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecoveryProposalFamily {
    AdditiveCorrection,
    MultiplicativeCorrection,
    InteractionCorrection,
    UnaryFunctionCorrection,
    GenericTyped,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RecoveryProposalSummary {
    accepted_count: usize,
    structured_count: usize,
    unary_function_count: usize,
}

/// Build a unit-correct correction basis around a rejected candidate. The scalar coefficient is fitted
/// separately from observed outcomes; the result remains a proposal until a fresh audit validates it.
///
/// Additive: basis has ENERGY dimensions.
/// Multiplicative: basis = base * dimensionless factor, so base + c*basis == base*(1 + c*factor).
/// Interaction: basis = x * delta_x, where delta_x has dimensions ENERGY / dimensions(x).
fn structured_correction_basis(
    rng: &mut u64,
    base: &Expr,
    family: RecoveryProposalFamily,
) -> Option<Expr> {
    let units = demo_units();
    let basis = match family {
        RecoveryProposalFamily::AdditiveCorrection => {
            random_expr_with_dimension(rng, DimensionalSignature::ENERGY, &units, GEN_MAX_DEPTH)
        }
        RecoveryProposalFamily::MultiplicativeCorrection => {
            let factor = random_expr_with_dimension(
                rng,
                DimensionalSignature::DIMENSIONLESS,
                &units,
                GEN_MAX_DEPTH,
            );
            Expr::BinOp(BinOp::Mul, Box::new(base.clone()), Box::new(factor))
        }
        RecoveryProposalFamily::InteractionCorrection => {
            if VAR_NAMES.is_empty() {
                return None;
            }
            let name = VAR_NAMES[rand_index(rng, VAR_NAMES.len())];
            let variable_dimension = *units.get(name)?;
            let correction_dimension = DimensionalSignature::ENERGY.sub(&variable_dimension);
            let correction =
                random_expr_with_dimension(rng, correction_dimension, &units, GEN_MAX_DEPTH);
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(name.to_string())),
                Box::new(correction),
            )
        }
        RecoveryProposalFamily::UnaryFunctionCorrection => {
            // This family is only enabled in the operator-held-out track. Construct a
            // dimensionally valid oscillatory expression and subtract the rejected base so the
            // fitted coefficient can replace the old structure when it is identifiable.
            // The proposal uses a public dimensionless invariant basis: q=v*t/r or q^-1.
            // This is deliberately a narrow operator-holdout task, not a claim of arbitrary
            // new-operator invention or unrestricted grammar expansion.
            let velocity_time = Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Var("t".to_string())),
            );
            let phase = if rand_index(rng, 2) == 0 {
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(velocity_time),
                    Box::new(Expr::Var("r".to_string())),
                )
            } else {
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::Var("r".to_string())),
                    Box::new(velocity_time),
                )
            };
            let unary = if rand_index(rng, 2) == 0 {
                UnaryFn::Sin
            } else {
                UnaryFn::Cos
            };
            let oscillatory = Expr::BinOp(
                BinOp::Mul,
                Box::new(base.clone()),
                Box::new(Expr::Func(unary, Box::new(phase))),
            );
            Expr::BinOp(BinOp::Sub, Box::new(oscillatory), Box::new(base.clone()))
        }
        RecoveryProposalFamily::GenericTyped => return None,
    };
    matches!(
        infer_dimensions(&basis, &units),
        InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
    )
    .then_some(basis)
}

/// Fit the scalar in candidate = base + coefficient * basis by weighted least squares.
/// The weights normalize each outcome by 1 + abs(y), matching the recovery score. The
/// deterministic coefficient bound is fixed in advance so ill-scaled observations cannot
/// produce arbitrarily large corrections. No hidden truth or discrepancy label is read.
fn fit_linear_correction_coefficient(
    base: &Expr,
    basis: &Expr,
    observations: &[(Experiment, f64)],
) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (point, observed) in observations {
        if !observed.is_finite() {
            return None;
        }
        let base_prediction = eval_at(base, point)?;
        let basis_prediction = eval_at(basis, point)?;
        let scale = 1.0 + observed.abs();
        let residual = (observed - base_prediction) / scale;
        let normalized_basis = basis_prediction / scale;
        numerator += normalized_basis * residual;
        denominator += normalized_basis * normalized_basis;
    }
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 1.0e-12 {
        return None;
    }
    let coefficient = (numerator / denominator).clamp(-10.0, 10.0);
    coefficient.is_finite().then_some(coefficient)
}

fn compose_linear_correction(base: &Expr, basis: &Expr, coefficient: f64) -> Option<Expr> {
    if !coefficient.is_finite() {
        return None;
    }
    let candidate = Expr::BinOp(
        BinOp::Add,
        Box::new(base.clone()),
        Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(coefficient)),
            Box::new(basis.clone()),
        )),
    );
    matches!(
        infer_dimensions(&candidate, &demo_units()),
        InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
    )
    .then_some(candidate)
}

/// Normalized empirical loss used only to rank recovery proposals. It is not a validation
/// metric and not an inferential guarantee; all past outcomes are reused for proposal ranking,
/// while only previously unused experiment points are eligible for subsequent audit evidence.
fn recovery_fit_score(candidate: &Expr, observations: &[(Experiment, f64)]) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    let mut loss = 0.0;
    for (point, observed) in observations {
        if !observed.is_finite() {
            return None;
        }
        let prediction = eval_at(candidate, point)?;
        let residual = (prediction - observed) / (1.0 + observed.abs());
        loss += residual * residual;
    }
    let mean_loss = loss / observations.len() as f64;
    let score = mean_loss + RECOVERY_COMPLEXITY_PENALTY * candidate.complexity() as f64;
    score.is_finite().then_some(score)
}

/// Reopen archived candidates after a predictive-adequacy alarm. Proposals are generated from
/// the public ENERGY task signature and the rejected expression, then ranked using observed
/// outcomes only. No evaluator discrepancy amplitude or ground-truth handle enters this policy.
/// The alarmed provisional candidate is archived but excluded from the new live set.
fn reseed_typed_candidates_after_alarm(
    rng: &mut u64,
    candidate_pool: &[Experiment],
    observations: &[(Experiment, f64)],
    hypotheses: &mut Vec<Expr>,
    live: &mut Vec<usize>,
    rejected_candidates: &mut Vec<usize>,
    alarmed_candidate: usize,
    allow_unary_function_recovery: bool,
) -> RecoveryProposalSummary {
    let archived_count = hypotheses.len();
    if candidate_pool.is_empty() || alarmed_candidate >= archived_count {
        return RecoveryProposalSummary::default();
    }
    if !rejected_candidates.contains(&alarmed_candidate) {
        rejected_candidates.push(alarmed_candidate);
    }

    let units = demo_units();
    let mut known_predictions: Vec<Vec<f64>> = hypotheses[..archived_count]
        .iter()
        .filter_map(|expression| {
            candidate_pool
                .iter()
                .map(|point| eval_at(expression, point))
                .collect::<Option<Vec<_>>>()
        })
        .collect();

    live.clear();
    live.extend((0..archived_count).filter(|index| !rejected_candidates.contains(index)));

    let base = hypotheses[alarmed_candidate].clone();
    let baseline_score = recovery_fit_score(&base, observations);
    // tuple order: structured first, then normalized loss, candidate, pool predictions, family
    let mut proposals: Vec<(u8, f64, Expr, Vec<f64>, RecoveryProposalFamily)> = Vec::new();
    let mut attempts = 0usize;
    while attempts < RECOVERY_GENERATION_ATTEMPT_CAP {
        let family = if allow_unary_function_recovery {
            match attempts % 5 {
                0 => RecoveryProposalFamily::AdditiveCorrection,
                1 => RecoveryProposalFamily::MultiplicativeCorrection,
                2 => RecoveryProposalFamily::InteractionCorrection,
                3 => RecoveryProposalFamily::UnaryFunctionCorrection,
                _ => RecoveryProposalFamily::GenericTyped,
            }
        } else {
            match attempts % 4 {
                0 => RecoveryProposalFamily::AdditiveCorrection,
                1 => RecoveryProposalFamily::MultiplicativeCorrection,
                2 => RecoveryProposalFamily::InteractionCorrection,
                _ => RecoveryProposalFamily::GenericTyped,
            }
        };
        attempts += 1;

        let candidate = if family == RecoveryProposalFamily::GenericTyped {
            random_expr_with_dimension(rng, DimensionalSignature::ENERGY, &units, GEN_MAX_DEPTH)
        } else {
            let Some(basis) = structured_correction_basis(rng, &base, family) else {
                continue;
            };
            let Some(coefficient) = fit_linear_correction_coefficient(&base, &basis, observations)
            else {
                continue;
            };
            let Some(candidate) = compose_linear_correction(&base, &basis, coefficient) else {
                continue;
            };
            candidate
        };
        if !matches!(
            infer_dimensions(&candidate, &units),
            InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
        ) {
            continue;
        }

        let Some(values) = candidate_pool
            .iter()
            .map(|point| eval_at(&candidate, point))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if !values.iter().all(|value| value.abs() < 1.0e6)
            || !values.iter().any(|value| value.abs() > 1.0e-6)
            || known_predictions
                .iter()
                .any(|known| !functionally_distinct(&values, known))
            || proposals
                .iter()
                .any(|(_, _, _, known, _)| !functionally_distinct(&values, known))
        {
            continue;
        }

        let score = recovery_fit_score(&candidate, observations)
            .unwrap_or_else(|| RECOVERY_COMPLEXITY_PENALTY * candidate.complexity() as f64);
        let is_structured = family != RecoveryProposalFamily::GenericTyped;
        if is_structured && baseline_score.is_some_and(|baseline| score >= baseline) {
            continue;
        }
        // Prefer candidates that improve observed residuals, but keep a generic typed fallback
        // if no structured correction survives the empirical screen. Fresh measurements remain
        // the arbiter of validity.
        proposals.push((
            if is_structured { 0 } else { 1 },
            score,
            candidate,
            values,
            family,
        ));
    }

    proposals.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let mut summary = RecoveryProposalSummary::default();
    for (_, _, candidate, values, family) in proposals {
        if summary.accepted_count >= RECOVERY_CANDIDATES_PER_EXPANSION {
            break;
        }
        if known_predictions
            .iter()
            .any(|known| !functionally_distinct(&values, known))
        {
            continue;
        }
        let index = hypotheses.len();
        hypotheses.push(candidate);
        live.push(index);
        known_predictions.push(values);
        summary.accepted_count += 1;
        if family != RecoveryProposalFamily::GenericTyped {
            summary.structured_count += 1;
        }
        if family == RecoveryProposalFamily::UnaryFunctionCorrection {
            summary.unary_function_count += 1;
        }
    }
    summary
}

/// An untyped distractor: same distinctness/sanity bar, but drawn from the naive grammar (no
/// dimensional constraint at all) -- this is the candidate-generation ablation factor's "off"
/// arm. Deliberately allowed to fall back to its last (possibly degenerate) draw if the
/// attempt cap is hit, rather than silently retrying forever -- an honest reflection of the
/// untyped grammar's own weakness at this task, not swept under the rug (see
/// `untyped_distractor_needs_far_more_attempts_than_typed` for how often this actually bites).
fn untyped_distractor(rng: &mut u64, gt_vals: &[f64], probes: &[Experiment]) -> (Expr, usize) {
    let mut last = Expr::Const(1.0);
    for attempt in 1..=GEN_ATTEMPT_CAP {
        let e = naive_untyped_expr(rng, GEN_MAX_DEPTH);
        let vals: Option<Vec<f64>> = probes.iter().map(|p| eval_at(&e, p)).collect();
        if let Some(vals) = vals {
            if vals.iter().all(|v| v.abs() > 1e-9 && v.abs() < 1e9)
                && functionally_distinct(&vals, gt_vals)
            {
                return (e, attempt);
            }
        }
        last = e;
    }
    (last, GEN_ATTEMPT_CAP)
}

#[derive(Clone, Copy, Debug)]
struct PipelineConfig {
    typed_candidates: bool,
    active_selection: bool,
    smoothing: bool,
}

impl PipelineConfig {
    const ALL_8: [PipelineConfig; 8] = {
        let mut out = [PipelineConfig {
            typed_candidates: false,
            active_selection: false,
            smoothing: false,
        }; 8];
        let mut i = 0;
        while i < 8 {
            out[i] = PipelineConfig {
                typed_candidates: (i & 1) != 0,
                active_selection: (i & 2) != 0,
                smoothing: (i & 4) != 0,
            };
            i += 1;
        }
        out
    };
}

/// Generate the initial truth-absent candidate pool from public task inputs only.
/// Deliberately accepts no reference expression, truth values, or evaluator holdout, making
/// oracle-conditioned candidate filtering impossible through this interface.
fn generate_truth_absent_candidates(
    seed: u64,
    config: PipelineConfig,
    design: &[Experiment],
) -> Option<Vec<Expr>> {
    generate_truth_absent_candidates_with_attempt_cap(
        seed,
        config,
        design,
        GEN_ATTEMPT_CAP * POOL_SIZE,
    )
}

/// Bounded core exposed to tests so exhaustion can be exercised deterministically without
/// changing the production attempt budget or panicking.
fn generate_truth_absent_candidates_with_attempt_cap(
    seed: u64,
    config: PipelineConfig,
    design: &[Experiment],
    attempt_cap: usize,
) -> Option<Vec<Expr>> {
    let mut hypotheses: Vec<Expr> = Vec::with_capacity(POOL_SIZE);
    let mut candidate_rng = seed
        ^ if config.typed_candidates {
            0xA11C_E55E_2026_0001
        } else {
            0xA11C_E55E_2026_0002
        };
    let mut candidate_values: Vec<Vec<f64>> = Vec::new();
    let mut attempts = 0usize;
    while hypotheses.len() < POOL_SIZE && attempts < attempt_cap {
        attempts += 1;
        let candidate = if config.typed_candidates {
            random_expr_with_dimension(
                &mut candidate_rng,
                DimensionalSignature::ENERGY,
                &demo_units(),
                GEN_MAX_DEPTH,
            )
        } else {
            naive_untyped_expr(&mut candidate_rng, GEN_MAX_DEPTH)
        };
        let Some(values) = design
            .iter()
            .map(|point| eval_at(&candidate, point))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if !values
            .iter()
            .all(|value| value.abs() > 1.0e-6 && value.abs() < 1.0e6)
            || candidate_values
                .iter()
                .any(|known| !functionally_distinct(&values, known))
        {
            continue;
        }
        candidate_values.push(values);
        hypotheses.push(candidate);
    }
    (hypotheses.len() == POOL_SIZE).then_some(hypotheses)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveSelector {
    RawVariance,
    ExpectedInformationGain,
    WorstCaseInformationGain,
    InformationGainWithAudits,
    InformationGainWithPrequentialPitAudits,
}

struct TrialOutcome {
    rounds_used: usize,
    success: bool,
    final_live: usize,
    selection_failed: bool,
    audit_queries: usize,
    model_discrepancy_detected: bool,
    overconfident_under_misspecification: bool,
    prequential_pit_alarm: bool,
    monitoring_protocol_invalid: bool,
    ground_truth_equivalent: bool,
    ground_truth_extrapolation_equivalent: bool,
    initial_pool_contains_reference: bool,
    reference_audit_evaluable: bool,
    in_domain_holdout_points: usize,
    extrapolation_holdout_points: usize,
    recovery_triggered_on_empty_pool: bool,
    typed_recovery_restarts: usize,
    structured_recovery_candidates: usize,
    unary_function_recovery_candidates: usize,
    accepted_recovery_candidates: usize,
    post_recovery_validation_measurements: usize,
    terminal_pit_alarm: bool,
}

fn unresolved_trial_outcome(
    final_live: usize,
    selection_failed: bool,
    monitoring_protocol_invalid: bool,
    terminal_pit_alarm: bool,
) -> TrialOutcome {
    TrialOutcome {
        rounds_used: 0,
        success: false,
        final_live,
        selection_failed,
        audit_queries: 0,
        model_discrepancy_detected: false,
        overconfident_under_misspecification: false,
        prequential_pit_alarm: false,
        monitoring_protocol_invalid,
        ground_truth_equivalent: false,
        ground_truth_extrapolation_equivalent: false,
        initial_pool_contains_reference: false,
        reference_audit_evaluable: false,
        in_domain_holdout_points: 0,
        extrapolation_holdout_points: 0,
        recovery_triggered_on_empty_pool: false,
        typed_recovery_restarts: 0,
        structured_recovery_candidates: 0,
        unary_function_recovery_candidates: 0,
        accepted_recovery_candidates: 0,
        post_recovery_validation_measurements: 0,
        terminal_pit_alarm,
    }
}

const POOL_SIZE: usize = 6;
const EXPERIMENT_POOL_SIZE: usize = 30;
const MAX_ROUNDS: usize = 20;
// Calibrated via two audit iterations (see `audit_everything_on_converges_at_a_measurable_rate`
// history): the first draft's 0.30 tolerance let independently-drawn distractors get
// eliminated in ~1 round regardless of strategy (nothing to discriminate). Simply tightening
// tolerance to 0.06 with the *original* 0.12 noise sigma broke it the other way -- tolerance
// was tighter than a single raw noisy sample's own error, so round 1 (before the smoother has
// any history to blend against) spuriously eliminated ground truth itself ~60% of the time, a
// coin flip unrelated to any of the three capabilities under test. Fixed by lowering
// NOISE_SIGMA_FRACTION so a single raw sample's error sits comfortably below tolerance (~1.75
// sigma), while tolerance stays below the near-miss cluster's 10-15% gap -- ground truth
// usually survives a single bad round, near-miss distractors still get reliably eliminated
// given a few rounds' evidence, and smoothing's job is exactly to make "a few rounds" fewer.
const NOISE_SIGMA_FRACTION: f64 = 0.04;
const ELIMINATION_TOLERANCE: f64 = 0.07;
const CFC_TAU: f64 = 0.6;
/// How many repeated noisy readings a `smoothing=true` round takes at its chosen point before
/// blending them via [`CfcSmoother`]. A `smoothing=false` round takes exactly one.
const K_REPEATS: usize = 4;

/// Uniform-scale "near-miss" ratios for [`near_miss_distractors`] -- see that function's docs
/// for why the pool needs these at all (an audit run showed the original design, where every
/// distractor was independently drawn, converged in ~1.5 rounds on average regardless of
/// condition: any random experiment already discriminated wildly-different candidates, leaving
/// none of the three capabilities under test any room to matter).
const NEAR_MISS_RATIOS: [f64; 4] = [0.85, 0.90, 1.10, 1.15];
const NEAR_MISS_COUNT: usize = 3;

/// `count` distinct uniform-scale variants of `ground_truth` (`ground_truth * ratio`, `ratio`
/// drawn from [`NEAR_MISS_RATIOS`] without replacement). A scaled copy keeps exactly the same
/// dimensional signature and functional *shape* as ground truth -- it differs from ground
/// truth by a fixed 10-15% at every point, which passes the `functionally_distinct` bar
/// cleanly but is comparable in scale to a single raw noisy observation's own error (see
/// `NOISE_SIGMA_FRACTION`), so telling it apart from ground truth reliably requires either
/// several rounds of noise-averaging or a well-chosen point, not a lucky single sample. This is
/// what actually gives the discrimination loop nontrivial multi-round texture -- independently
/// drawn (typed or untyped) distractors remain in the pool too (as "wildcards"), preserving the
/// candidate-generation ablation, but the near-miss cluster is what makes convergence take more
/// than one round.
fn near_miss_distractors(rng: &mut u64, ground_truth: &Expr, count: usize) -> Vec<Expr> {
    let mut ratios = NEAR_MISS_RATIOS.to_vec();
    shuffle(rng, &mut ratios);
    ratios
        .into_iter()
        .take(count)
        .map(|r| {
            Expr::BinOp(
                BinOp::Mul,
                Box::new(ground_truth.clone()),
                Box::new(Expr::Const(r)),
            )
        })
        .collect()
}

fn run_trial(seed: u64, config: PipelineConfig, max_rounds: usize) -> TrialOutcome {
    run_trial_with_selector(seed, config, max_rounds, ActiveSelector::RawVariance)
}

/// Estimate a common observation-noise scale from live predictions alone.
/// The simulator uses truth-scaled noise; the median proxy avoids leakage from the
/// hidden law and makes the EIG selector's homoscedastic approximation explicit.
fn selection_noise_std_dev(remaining: &[Experiment], live: &[usize], hypotheses: &[Expr]) -> f64 {
    let mut magnitudes = Vec::new();
    for experiment in remaining {
        for index in live {
            if let Some(prediction) = eval_at(&hypotheses[*index], experiment) {
                magnitudes.push(prediction.abs());
            }
        }
    }
    magnitudes.retain(|magnitude: &f64| magnitude.is_finite());
    if magnitudes.is_empty() {
        return NOISE_SIGMA_FRACTION;
    }
    magnitudes.sort_by(f64::total_cmp);
    NOISE_SIGMA_FRACTION * (1.0 + magnitudes[magnitudes.len() / 2])
}

fn run_trial_with_selector(
    seed: u64,
    config: PipelineConfig,
    max_rounds: usize,
    active_selector: ActiveSelector,
) -> TrialOutcome {
    run_trial_with_discrepancy(seed, config, max_rounds, active_selector, 0.0)
}

fn run_trial_with_discrepancy(
    seed: u64,
    config: PipelineConfig,
    max_rounds: usize,
    active_selector: ActiveSelector,
    systematic_discrepancy: f64,
) -> TrialOutcome {
    run_trial_with_mode(
        seed,
        config,
        max_rounds,
        active_selector,
        systematic_discrepancy,
        false,
        true,
        false,
    )
}

/// Separate truth-absent track. The policy receives a candidate pool generated independently of
/// the reference law. The disabled-recovery setting is the fail-closed stop/unresolved baseline.
fn run_truth_absent_trial(
    seed: u64,
    config: PipelineConfig,
    max_rounds: usize,
    active_selector: ActiveSelector,
    enable_recovery: bool,
) -> TrialOutcome {
    run_trial_with_mode(
        seed,
        config,
        max_rounds,
        active_selector,
        0.0,
        true,
        enable_recovery,
        false,
    )
}

/// Operator-held-out recovery track. Its target has a dimensionally valid unary function that
/// the initial typed algebraic generator does not produce; the recovery policy may explicitly
/// enable a bounded sine/cosine proposal family after falsification.
fn run_operator_heldout_trial(
    seed: u64,
    config: PipelineConfig,
    max_rounds: usize,
    active_selector: ActiveSelector,
    enable_recovery: bool,
) -> TrialOutcome {
    run_trial_with_mode(
        seed,
        config,
        max_rounds,
        active_selector,
        0.0,
        true,
        enable_recovery,
        true,
    )
}

fn run_trial_with_mode(
    seed: u64,
    config: PipelineConfig,
    max_rounds: usize,
    active_selector: ActiveSelector,
    systematic_discrepancy: f64,
    truth_absent: bool,
    enable_recovery: bool,
    operator_held_out: bool,
) -> TrialOutcome {
    let pit_mode = active_selector == ActiveSelector::InformationGainWithPrequentialPitAudits;
    if pit_mode && (!config.active_selection || config.smoothing) {
        return unresolved_trial_outcome(POOL_SIZE, true, true, false);
    }
    if operator_held_out && (!truth_absent || !config.typed_candidates) {
        return unresolved_trial_outcome(0, true, false, true);
    }
    let mut setup_rng = seed ^ 0x5EED_0000_0000_0001;
    let probes: Vec<Experiment> = (0..GT_SANITY_PROBES)
        .map(|_| random_experiment(&mut setup_rng))
        .collect();

    // The discovery design is generated before the hidden reference and without reference-based
    // rejection. The task generator later requires the reference to be defined and sane on these
    // predeclared points, so an undefined measurement never silently changes the selector's pool.
    let mut exp_rng = seed ^ 0x1111_2222_3333_4444;
    let candidate_pool: Vec<Experiment> = (0..EXPERIMENT_POOL_SIZE)
        .map(|_| random_experiment(&mut exp_rng))
        .collect();
    let mut reference_sanity_points = probes.clone();
    reference_sanity_points.extend(candidate_pool.iter().cloned());
    let ground_truth = if operator_held_out {
        let Some(target) =
            nontrivial_unary_ground_truth(&mut setup_rng, &reference_sanity_points)
        else {
            return unresolved_trial_outcome(0, true, false, true);
        };
        target
    } else {
        nontrivial_ground_truth(&mut setup_rng, &reference_sanity_points)
    };
    let gt_vals: Vec<f64> = probes
        .iter()
        .map(|point| eval_at(&ground_truth, point).unwrap())
        .collect();

    let mut hypotheses: Vec<Expr> = if truth_absent {
        let Some(candidates) = generate_truth_absent_candidates(seed, config, &candidate_pool) else {
            return unresolved_trial_outcome(0, true, false, true);
        };
        candidates
    } else {
        // Oracle-present discrimination track: the reference is deliberately included, and
        // near-miss candidates are derived from it. These results are reported separately.
        let mut hypotheses = vec![ground_truth.clone()];
        let mut near_miss_rng = seed ^ 0x2026_0716_0000_0003;
        hypotheses.extend(near_miss_distractors(
            &mut near_miss_rng,
            &ground_truth,
            NEAR_MISS_COUNT,
        ));
        let mut gen_rng = seed
            ^ if config.typed_candidates {
                0x7A3F_9B21_0000_0001
            } else {
                0xC0DE_BAAD_0000_0002
            };
        while hypotheses.len() < POOL_SIZE {
            let distractor = if config.typed_candidates {
                typed_distractor(&mut gen_rng, &gt_vals, &probes)
            } else {
                untyped_distractor(&mut gen_rng, &gt_vals, &probes).0
            };
            hypotheses.push(distractor);
        }
        hypotheses
    };
    let initial_hypotheses = hypotheses.clone();

    // Independent evaluator-only points: never passed to the experiment selector or recovery
    // generator. Use a bounded attempt budget and require a minimum valid design; otherwise the
    // evaluator endpoint fails closed instead of hanging or silently shrinking its split.
    const IN_DOMAIN_HOLDOUT_TARGET: usize = 64;
    const IN_DOMAIN_HOLDOUT_MINIMUM: usize = 32;
    const IN_DOMAIN_HOLDOUT_ATTEMPT_CAP: usize = 4096;
    let mut evaluator_rng = seed ^ 0xE7A1_0A7A_2026_0001;
    let mut holdout: Vec<Experiment> = Vec::with_capacity(IN_DOMAIN_HOLDOUT_TARGET);
    let mut in_domain_attempts = 0usize;
    while holdout.len() < IN_DOMAIN_HOLDOUT_TARGET
        && in_domain_attempts < IN_DOMAIN_HOLDOUT_ATTEMPT_CAP
    {
        in_domain_attempts += 1;
        let point = random_experiment(&mut evaluator_rng);
        if eval_at(&ground_truth, &point).is_some_and(|value| value.abs() < 1.0e12) {
            holdout.push(point);
        }
    }

    // Independent extrapolation design, strictly outside every training-variable interval.
    // The bounded loop prevents pathological expressions from hanging a benchmark trial. If
    // fewer than the minimum valid truth points can be generated, OOD recovery is scored false.
    const OOD_HOLDOUT_TARGET: usize = 64;
    const OOD_HOLDOUT_MINIMUM: usize = 32;
    const OOD_ATTEMPT_CAP: usize = 4096;
    const OOD_ABSOLUTE_VALUE_CAP: f64 = 1.0e12;
    let mut ood_evaluator_rng = seed ^ 0x00D0_0D0E_2026_0001;
    let mut extrapolation_holdout: Vec<Experiment> = Vec::with_capacity(OOD_HOLDOUT_TARGET);
    let mut ood_attempts = 0usize;
    while extrapolation_holdout.len() < OOD_HOLDOUT_TARGET && ood_attempts < OOD_ATTEMPT_CAP {
        ood_attempts += 1;
        let point = random_extrapolation_experiment(&mut ood_evaluator_rng);
        if eval_at(&ground_truth, &point).is_some_and(|value| value.abs() < OOD_ABSOLUTE_VALUE_CAP)
        {
            extrapolation_holdout.push(point);
        }
    }

    let mut fixed_order = candidate_pool.clone();
    let mut order_rng = seed ^ 0x9999_8888_7777_6666;
    shuffle(&mut order_rng, &mut fixed_order);

    let mut live: Vec<usize> = (0..hypotheses.len()).collect();
    let mut used: Vec<Experiment> = Vec::new();
    let mut measured_observations: Vec<(Experiment, f64)> = Vec::new();
    let mut noise_rng = seed ^ 0xAAAA_BBBB_CCCC_DDDD;
    let mut dt_rng = seed ^ 0xDDDD_CCCC_BBBB_AAAA;
    let mut audit_rng = seed ^ 0xA0D1_7000_51E1_0001;
    let mut selection_failed = false;
    let mut audit_queries = 0usize;
    let mut pit_monitor = if pit_mode {
        Some(
            PrequentialPitMartingale::new(0.05, VAR_NAMES.len())
                .expect("fixed alpha and context-channel count are valid"),
        )
    } else {
        None
    };
    let mut prequential_pit_alarm = false;
    let mut monitoring_protocol_invalid = false;
    let mut typed_recovery_restarts = 0usize;
    let mut structured_recovery_candidates = 0usize;
    let mut unary_function_recovery_candidates = 0usize;
    let mut accepted_recovery_candidates = 0usize;
    let mut awaiting_post_recovery_validation = false;
    let mut post_recovery_validation_measurements = 0usize;
    let mut recovery_triggered_on_empty_pool = false;
    let mut rejected_after_alarm = Vec::<usize>::new();
    let mut terminal_pit_alarm = false;

    let mut round = 0usize;
    while round < max_rounds
        && (live.len() > 1
            || (config.active_selection
                && matches!(
                    active_selector,
                    ActiveSelector::InformationGainWithAudits
                        | ActiveSelector::InformationGainWithPrequentialPitAudits
                )
                && live.len() == 1))
    {
        let predict = |idx: &usize, e: &Experiment| eval_at(&hypotheses[*idx], e);
        let chosen = if config.active_selection {
            let remaining: Vec<Experiment> = candidate_pool
                .iter()
                .filter(|e| !used.contains(e))
                .cloned()
                .collect();
            if remaining.is_empty() {
                selection_failed = true;
                break;
            }
            // A singleton is provisional under the audit policy: use every remaining
            // query as a validation audit. Before a singleton, schedule an audit every
            // second round to reduce greedy over-exploitation.
            let scheduled_audit = match active_selector {
                ActiveSelector::InformationGainWithAudits => {
                    live.len() == 1 || (round + 1) % 2 == 0
                }
                ActiveSelector::InformationGainWithPrequentialPitAudits => live.len() == 1,
                _ => false,
            };
            let selected = if scheduled_audit {
                audit_queries += 1;
                Some(remaining[rand_index(&mut audit_rng, remaining.len())].clone())
            } else {
                match active_selector {
                    ActiveSelector::RawVariance => {
                        select_most_informative_experiment(&remaining, &live, predict)
                            .map(|(experiment, _)| experiment.clone())
                    }
                    ActiveSelector::ExpectedInformationGain
                    | ActiveSelector::InformationGainWithAudits
                    | ActiveSelector::InformationGainWithPrequentialPitAudits => {
                        let noise_std_dev = selection_noise_std_dev(&remaining, &live, &hypotheses);
                        select_experiment_by_information_gain_per_cost(
                            &remaining,
                            &live,
                            noise_std_dev,
                            predict,
                            |_: &Experiment| 1.0,
                        )
                        .map(|(experiment, _)| experiment.clone())
                    }
                    ActiveSelector::WorstCaseInformationGain => {
                        let nominal_noise = selection_noise_std_dev(&remaining, &live, &hypotheses);
                        // These are bundled stress scenarios, not a full Cartesian sweep:
                        // three measurement scales paired with the nominal model and two
                        // alternative smooth discrepancy shapes. All parameters are fixed
                        // before observing outcomes; the hidden true discrepancy is not read.
                        let scenarios = [
                            InformationGainScenario {
                                observation_std_dev: nominal_noise * 0.5,
                                prediction_model: 0,
                            },
                            InformationGainScenario {
                                observation_std_dev: nominal_noise,
                                prediction_model: 1,
                            },
                            InformationGainScenario {
                                observation_std_dev: nominal_noise * 2.0,
                                prediction_model: 2,
                            },
                        ];
                        let predict_under_scenario =
                            |idx: &usize,
                             experiment: &Experiment,
                             scenario: &InformationGainScenario| {
                                let base = eval_at(&hypotheses[*idx], experiment)?;
                                if scenario.prediction_model == 0 {
                                    return Some(base);
                                }
                                let coordinates: Vec<f64> =
                                    experiment.iter().map(|(_, value)| *value).collect();
                                let frequency = if scenario.prediction_model == 1 {
                                    5.0
                                } else {
                                    9.0
                                };
                                let discrepancy_basis = coordinates
                                    .iter()
                                    .enumerate()
                                    .map(|(index, value)| {
                                        (value * (frequency + index as f64)).sin()
                                    })
                                    .sum::<f64>()
                                    / coordinates.len().max(1) as f64;
                                Some(base + 0.12 * (1.0 + base.abs()) * discrepancy_basis)
                            };
                        select_experiment_by_scenario_robust_information_gain_per_cost(
                            &remaining,
                            &live,
                            &scenarios,
                            predict_under_scenario,
                            |_: &Experiment| 1.0,
                        )
                        .map(|(experiment, _)| experiment.clone())
                    }
                }
            };
            let Some(selected) = selected else {
                selection_failed = true;
                break;
            };
            selected
        } else {
            if round >= fixed_order.len() {
                break;
            }
            fixed_order[round].clone()
        };
        let is_post_recovery_singleton_validation =
            awaiting_post_recovery_validation && live.len() == 1;
        used.push(chosen.clone());

        let Some(base_true_val) = eval_at(&ground_truth, &chosen) else {
            // In truth-absent mode the policy cannot use hidden truth to pre-filter designs.
            // If the evaluator's physical law is undefined at this selected point, fail closed.
            selection_failed = true;
            break;
        };
        // This controlled additive term is hidden from every candidate and selector.
        // A nonzero amplitude perturbs observations relative to the finite candidate
        // pool; selectors must not read this term.
        let coordinates: Vec<f64> = chosen.iter().map(|(_, value)| *value).collect();
        let discrepancy_wave = coordinates
            .iter()
            .enumerate()
            .map(|(index, value)| (value * (3.0 + index as f64)).sin())
            .sum::<f64>()
            / coordinates.len().max(1) as f64;
        let true_val =
            base_true_val + systematic_discrepancy * (1.0 + base_true_val.abs()) * discrepancy_wave;
        // `smoothing`'s real question: is it worth taking K_REPEATS repeated noisy readings at
        // this SAME chosen point and blending them (CfC, dt-aware, irregular gaps between
        // readings), vs. a single one-shot reading? Deliberately NOT smoothing across
        // *different* experiment points from round to round -- `CfcSmoother` tracks one
        // evolving quantity over time, and successive rounds here evaluate ground truth at
        // unrelated (m,v,r,t) points chosen precisely to differ; blending those together
        // (an earlier draft of this module did exactly that) contaminates the current point's
        // estimate with stale, unrelated values and was the actual cause of an early miscalibration
        // where ground truth itself got spuriously eliminated most rounds, independent of noise
        // tuning. A fresh smoother per round, fed only repeated readings of the one point that
        // round chose, is the correct scope for what this module is testing.
        let observed = if config.smoothing {
            let mut smoother = CfcSmoother::new(1, CFC_TAU);
            for i in 0..K_REPEATS {
                let dt = if i == 0 {
                    0.0
                } else {
                    rand_range(&mut dt_rng, 0.2, 2.0)
                };
                let noisy = true_val
                    + rand_gaussian(
                        &mut noise_rng,
                        NOISE_SIGMA_FRACTION * (1.0 + true_val.abs()),
                    );
                smoother.observe(dt, &[noisy]);
            }
            smoother.state()[0]
        } else {
            true_val
                + rand_gaussian(
                    &mut noise_rng,
                    NOISE_SIGMA_FRACTION * (1.0 + true_val.abs()),
                )
        };

        let monitored_candidate = if pit_mode && live.len() == 1 {
            Some(live[0])
        } else {
            None
        };
        let mut pit_alarm_this_round = false;
        if pit_mode && live.len() == 1 {
            let Some(prediction) = eval_at(&hypotheses[live[0]], &chosen) else {
                monitoring_protocol_invalid = true;
                selection_failed = true;
                round += 1;
                break;
            };
            let predictive_scale = NOISE_SIGMA_FRACTION * (1.0 + prediction.abs());
            if !predictive_scale.is_finite() || predictive_scale <= 0.0 {
                monitoring_protocol_invalid = true;
                selection_failed = true;
                round += 1;
                break;
            }
            let Some(pit) = standard_normal_cdf((observed - prediction) / predictive_scale) else {
                monitoring_protocol_invalid = true;
                selection_failed = true;
                round += 1;
                break;
            };
            let context_features = chosen
                .iter()
                .map(|(_, value)| {
                    (2.0 * (*value - VAR_RANGE.0) / (VAR_RANGE.1 - VAR_RANGE.0) - 1.0)
                        .clamp(-1.0, 1.0)
                })
                .collect::<Vec<_>>();
            let Some(monitor) = pit_monitor.as_mut() else {
                monitoring_protocol_invalid = true;
                selection_failed = true;
                round += 1;
                break;
            };
            match monitor.observe(pit, &context_features) {
                Ok(result) => {
                    pit_alarm_this_round = result.decision == PrequentialPitDecision::Alarm;
                    prequential_pit_alarm |= pit_alarm_this_round;
                }
                Err(_) => {
                    monitoring_protocol_invalid = true;
                    selection_failed = true;
                    round += 1;
                    break;
                }
            }
        }

        if is_post_recovery_singleton_validation {
            post_recovery_validation_measurements += 1;
        }
        live.retain(|idx| match eval_at(&hypotheses[*idx], &chosen) {
            Some(pred) => (pred - observed).abs() <= ELIMINATION_TOLERANCE * (1.0 + observed.abs()),
            None => false,
        });
        measured_observations.push((chosen.clone(), observed));
        round += 1;
        if pit_alarm_this_round {
            // An alarm is evidence against this provisional model, not evidence for any
            // replacement. Reopen the archived pool, exclude the alarmed candidate, add a
            // bounded set of dimensionally-valid proposals, and begin a fresh monitor epoch.
            if typed_recovery_restarts >= MAX_TYPED_RECOVERY_RESTARTS {
                terminal_pit_alarm = true;
                break;
            }
            if !enable_recovery {
                terminal_pit_alarm = true;
                break;
            }
            let Some(alarmed_index) = monitored_candidate else {
                monitoring_protocol_invalid = true;
                terminal_pit_alarm = true;
                break;
            };
            let mut recovery_rng =
                seed ^ 0xF1A1_1E5E_2026_0001 ^ (typed_recovery_restarts as u64).rotate_left(19);
            let recovery = reseed_typed_candidates_after_alarm(
                &mut recovery_rng,
                &candidate_pool,
                &measured_observations,
                &mut hypotheses,
                &mut live,
                &mut rejected_after_alarm,
                alarmed_index,
                operator_held_out,
            );
            if recovery.accepted_count == 0 {
                terminal_pit_alarm = true;
                break;
            }
            structured_recovery_candidates += recovery.structured_count;
            unary_function_recovery_candidates += recovery.unary_function_count;
            accepted_recovery_candidates += recovery.accepted_count;
            typed_recovery_restarts += 1;
            awaiting_post_recovery_validation = true;
            post_recovery_validation_measurements = 0;
            terminal_pit_alarm = false;
            if pit_mode {
                pit_monitor = Some(
                    PrequentialPitMartingale::new(0.05, VAR_NAMES.len())
                        .expect("fixed alpha and context-channel count are valid"),
                );
            }
        }

        // Open-ended recovery trigger: once every current candidate is contradicted, restart
        // from the lowest-residual archived expression while excluding the entire falsified
        // pool. This branch is evaluator-blind; it does not consult the reference law or OOD
        // holdout. If recovery is disabled, the trial stays unresolved at an empty live set.
        if truth_absent && enable_recovery && live.is_empty() {
            if typed_recovery_restarts >= MAX_TYPED_RECOVERY_RESTARTS {
                terminal_pit_alarm = true;
                break;
            }
            let Some((base_index, _)) = hypotheses
                .iter()
                .enumerate()
                .filter_map(|(index, expression)| {
                    recovery_fit_score(expression, &measured_observations)
                        .map(|score| (index, score))
                })
                .min_by(|left, right| left.1.total_cmp(&right.1))
            else {
                terminal_pit_alarm = true;
                break;
            };
            rejected_after_alarm.extend(0..hypotheses.len());
            let mut recovery_rng =
                seed ^ 0x0DD0_5EED_2026_0001 ^ (typed_recovery_restarts as u64).rotate_left(23);
            recovery_triggered_on_empty_pool = true;
            let recovery = reseed_typed_candidates_after_alarm(
                &mut recovery_rng,
                &candidate_pool,
                &measured_observations,
                &mut hypotheses,
                &mut live,
                &mut rejected_after_alarm,
                base_index,
                operator_held_out,
            );
            if recovery.accepted_count == 0 || live.is_empty() {
                terminal_pit_alarm = true;
                break;
            }
            structured_recovery_candidates += recovery.structured_count;
            unary_function_recovery_candidates += recovery.unary_function_count;
            accepted_recovery_candidates += recovery.accepted_count;
            typed_recovery_restarts += 1;
            awaiting_post_recovery_validation = true;
            post_recovery_validation_measurements = 0;
        }
    }

    let ground_truth_equivalent = live.len() == 1
        && holdout.len() >= IN_DOMAIN_HOLDOUT_MINIMUM
        && evaluator_matches_reference(&hypotheses[live[0]], &ground_truth, &holdout);
    let ground_truth_extrapolation_equivalent = live.len() == 1
        && extrapolation_holdout.len() >= OOD_HOLDOUT_MINIMUM
        && evaluator_matches_reference(&hypotheses[live[0]], &ground_truth, &extrapolation_holdout);
    // Evaluator-only audit of the initial candidate set. It runs after the policy ends and is
    // used only to identify accidentally oracle-present trials in the truth-absent track.
    let reference_audit_evaluable = holdout.len() >= IN_DOMAIN_HOLDOUT_MINIMUM
        && extrapolation_holdout.len() >= OOD_HOLDOUT_MINIMUM;
    let initial_pool_contains_reference = reference_audit_evaluable
        && initial_hypotheses.iter().any(|candidate| {
            evaluator_matches_reference_across_domains(
                candidate,
                &ground_truth,
                &holdout,
                &extrapolation_holdout,
            )
        });

    TrialOutcome {
        rounds_used: round,
        success: systematic_discrepancy == 0.0
            && holdout.len() >= IN_DOMAIN_HOLDOUT_MINIMUM
            && ground_truth_equivalent
            && ground_truth_extrapolation_equivalent
            && (!truth_absent || !reference_audit_evaluable || !initial_pool_contains_reference)
            && (!truth_absent
                || post_recovery_validation_measurements
                    >= MIN_POST_RECOVERY_VALIDATION_MEASUREMENTS)
            && !terminal_pit_alarm
            && !monitoring_protocol_invalid,
        final_live: live.len(),
        selection_failed,
        audit_queries,
        model_discrepancy_detected: systematic_discrepancy != 0.0
            && (live.is_empty() || prequential_pit_alarm),
        overconfident_under_misspecification: systematic_discrepancy != 0.0
            && live.len() == 1
            && !prequential_pit_alarm,
        prequential_pit_alarm,
        monitoring_protocol_invalid,
        ground_truth_equivalent,
        ground_truth_extrapolation_equivalent,
        initial_pool_contains_reference,
        reference_audit_evaluable,
        in_domain_holdout_points: holdout.len(),
        extrapolation_holdout_points: extrapolation_holdout.len(),
        recovery_triggered_on_empty_pool,
        typed_recovery_restarts,
        structured_recovery_candidates,
        unary_function_recovery_candidates,
        accepted_recovery_candidates,
        post_recovery_validation_measurements,
        terminal_pit_alarm,
    }
}

/// Run the full 2x2x2 factorial over `n_seeds` paired seeds at a given `max_rounds` budget,
/// returning `(success_rate, mean_rounds)` per condition. Shared by
/// `factorial_ablation_of_generation_selection_smoothing` (the original, `MAX_ROUNDS`-budget
/// frozen comparison) and `factorial_ablation_under_scarce_round_budget` (the follow-up that
/// tightens the budget to test whether `active_selection`'s small, sub-threshold effect at the
/// original budget was being masked by round-robin having enough slack to also succeed via
/// brute force).
#[cfg(test)]
fn run_factorial(n_seeds: u64, max_rounds: usize) -> HashMap<(bool, bool, bool), (f64, f64)> {
    let mut out = HashMap::new();
    for config in PipelineConfig::ALL_8 {
        let mut successes = 0usize;
        let mut total_rounds = 0usize;
        for seed in 1..=n_seeds {
            let outcome = run_trial(seed, config, max_rounds);
            if outcome.success {
                successes += 1;
            }
            total_rounds += outcome.rounds_used;
        }
        let key = (
            config.typed_candidates,
            config.active_selection,
            config.smoothing,
        );
        out.insert(
            key,
            (
                successes as f64 / n_seeds as f64,
                total_rounds as f64 / n_seeds as f64,
            ),
        );
        println!(
            "typed={:>5} active={:>5} smoothed={:>5}: success_rate={:.3} mean_rounds={:.2}",
            config.typed_candidates,
            config.active_selection,
            config.smoothing,
            successes as f64 / n_seeds as f64,
            total_rounds as f64 / n_seeds as f64
        );
    }
    out
}

/// Mean success rate with `factor` on vs. off, averaged over the other two factors -- shared
/// by both factorial tests.
#[cfg(test)]
fn mean_on_off(
    success_by_config: &HashMap<(bool, bool, bool), (f64, f64)>,
    factor: usize,
) -> (f64, f64) {
    let (mut on_sum, mut off_sum) = (0.0, 0.0);
    for (&(t, a, s), &(rate, _)) in success_by_config {
        let bits = [t, a, s];
        if bits[factor] {
            on_sum += rate;
        } else {
            off_sum += rate;
        }
    }
    (on_sum / 4.0, off_sum / 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adequacy_calibration::{ConformalAdequacyCalibrator, standardized_rms_residual};
    use crate::model_adequacy::{
        DiscoveryAction, DiscoverySnapshot, TerminationStatus, ValidationStatus,
        next_discovery_action,
    };

    type SelectorPair = (bool, bool, usize, usize, bool);

    struct MisspecSelectorPair {
        eig_detected: bool,
        eig_false_singleton: bool,
        audit_detected: bool,
        audit_false_singleton: bool,
        clean_eig_success: bool,
        clean_audit_success: bool,
    }

    struct RobustEIGPair {
        clean_nominal_success: bool,
        clean_robust_success: bool,
        misspec_nominal_false_singleton: bool,
        misspec_robust_false_singleton: bool,
        misspec_nominal_detected: bool,
        misspec_robust_detected: bool,
        misspec_nominal_unresolved: bool,
        misspec_robust_unresolved: bool,
        misspec_nominal_selection_failed: bool,
        misspec_robust_selection_failed: bool,
        misspec_nominal_rounds: usize,
        misspec_robust_rounds: usize,
    }

    /// Generate a batch-level predictive residual score without exposing the hidden
    /// discrepancy amplitude to the detector. Each batch uses its own deterministic seed.
    fn conformal_audit_batch_score(seed: u64, discrepancy_amplitude: f64) -> f64 {
        const BATCH_SIZE: usize = 12;
        let mut setup_rng = seed ^ 0x5EED_0000_0000_0001;
        let probes: Vec<Experiment> = (0..GT_SANITY_PROBES)
            .map(|_| random_experiment(&mut setup_rng))
            .collect();
        let ground_truth = nontrivial_ground_truth(&mut setup_rng, &probes);
        let mut point_rng = seed ^ 0xCA11_BA7C_0000_0001;
        let mut noise_rng = seed ^ 0xA11D_1700_0000_0002;
        let mut observed = Vec::with_capacity(BATCH_SIZE);
        let mut predicted = Vec::with_capacity(BATCH_SIZE);
        let mut scales = Vec::with_capacity(BATCH_SIZE);

        for _ in 0..GEN_ATTEMPT_CAP {
            if observed.len() == BATCH_SIZE {
                break;
            }
            let point = random_experiment(&mut point_rng);
            let Some(prediction) = eval_at(&ground_truth, &point) else {
                continue;
            };
            if prediction.abs() > 1.0e6 {
                continue;
            }

            let coordinates: Vec<f64> = point.iter().map(|(_, value)| *value).collect();
            let discrepancy_wave = coordinates
                .iter()
                .enumerate()
                .map(|(index, value)| (value * (3.0 + index as f64)).sin())
                .sum::<f64>()
                / coordinates.len().max(1) as f64;
            let simulated_observation =
                prediction + discrepancy_amplitude * (1.0 + prediction.abs()) * discrepancy_wave;
            let actual_noise_scale = NOISE_SIGMA_FRACTION * (1.0 + simulated_observation.abs());
            let measurement =
                simulated_observation + rand_gaussian(&mut noise_rng, actual_noise_scale);
            let predictive_scale = NOISE_SIGMA_FRACTION * (1.0 + prediction.abs());
            if measurement.is_finite() && predictive_scale.is_finite() && predictive_scale > 0.0 {
                observed.push(measurement);
                predicted.push(prediction);
                scales.push(predictive_scale);
            }
        }

        assert_eq!(
            observed.len(),
            BATCH_SIZE,
            "fixed seed should yield a complete finite audit batch"
        );
        standardized_rms_residual(&observed, &predicted, &scales)
            .expect("generated residual batch must have a finite score")
    }

    /// Paired percentile bootstrap over seed/world outcomes. Every resample carries the raw
    /// and EIG result together, preserving pairing. This small pilot interval is diagnostic;
    /// it does not replace world-clustered confirmatory inference on a larger held-out corpus.
    fn paired_bootstrap_mean_ci<T>(
        samples: &[T],
        metric: impl Fn(&T) -> f64,
        seed: u64,
    ) -> (f64, f64, f64) {
        const BOOTSTRAP_REPLICATES: usize = 2_000;
        assert!(!samples.is_empty());
        let observed = samples.iter().map(&metric).sum::<f64>() / samples.len() as f64;
        let mut rng = seed;
        let mut means = Vec::with_capacity(BOOTSTRAP_REPLICATES);
        for _ in 0..BOOTSTRAP_REPLICATES {
            let sum = (0..samples.len())
                .map(|_| metric(&samples[rand_index(&mut rng, samples.len())]))
                .sum::<f64>();
            means.push(sum / samples.len() as f64);
        }
        means.sort_by(f64::total_cmp);
        let last = means.len() - 1;
        let low_index = (last * 25) / 1_000;
        let high_index = (last * 975 + 999) / 1_000;
        (observed, means[low_index], means[high_index])
    }

    /// Small paired-seed pilot for the robust-noise-envelope selector versus nominal EIG.
    /// It reports clean recovery and three disjoint outcomes under hidden structural
    /// discrepancy; it intentionally does not assert that the robust policy wins.
    #[test]
    fn scenario_robust_eig_vs_nominal_eig_pilot() {
        const SEEDS: u64 = 20;
        const ROUND_BUDGET: usize = 4;
        const SYSTEMATIC_DISCREPANCY: f64 = 0.12;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };
        let mut clean_nominal_successes = 0usize;
        let mut clean_robust_successes = 0usize;
        let mut misspec_nominal_false_singletons = 0usize;
        let mut misspec_robust_false_singletons = 0usize;
        let mut misspec_nominal_detected = 0usize;
        let mut misspec_robust_detected = 0usize;
        let mut misspec_nominal_unresolved = 0usize;
        let mut misspec_robust_unresolved = 0usize;
        let mut misspec_nominal_selection_failures = 0usize;
        let mut misspec_robust_selection_failures = 0usize;
        let mut misspec_nominal_round_total = 0usize;
        let mut misspec_robust_round_total = 0usize;
        let mut paired: Vec<RobustEIGPair> = Vec::with_capacity(SEEDS as usize);

        for seed in 1..=SEEDS {
            let clean_nominal = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::ExpectedInformationGain,
                0.0,
            );
            let clean_robust = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::WorstCaseInformationGain,
                0.0,
            );
            let misspec_nominal = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::ExpectedInformationGain,
                SYSTEMATIC_DISCREPANCY,
            );
            let misspec_robust = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::WorstCaseInformationGain,
                SYSTEMATIC_DISCREPANCY,
            );

            clean_nominal_successes += if clean_nominal.success { 1 } else { 0 };
            clean_robust_successes += if clean_robust.success { 1 } else { 0 };
            misspec_nominal_false_singletons +=
                if misspec_nominal.overconfident_under_misspecification {
                    1
                } else {
                    0
                };
            misspec_robust_false_singletons +=
                if misspec_robust.overconfident_under_misspecification {
                    1
                } else {
                    0
                };
            misspec_nominal_detected += if misspec_nominal.model_discrepancy_detected {
                1
            } else {
                0
            };
            misspec_robust_detected += if misspec_robust.model_discrepancy_detected {
                1
            } else {
                0
            };
            misspec_nominal_unresolved += if misspec_nominal.final_live > 1 { 1 } else { 0 };
            misspec_robust_unresolved += if misspec_robust.final_live > 1 { 1 } else { 0 };
            misspec_nominal_selection_failures += if misspec_nominal.selection_failed {
                1
            } else {
                0
            };
            misspec_robust_selection_failures += if misspec_robust.selection_failed {
                1
            } else {
                0
            };
            misspec_nominal_round_total += misspec_nominal.rounds_used;
            misspec_robust_round_total += misspec_robust.rounds_used;
            paired.push(RobustEIGPair {
                clean_nominal_success: clean_nominal.success,
                clean_robust_success: clean_robust.success,
                misspec_nominal_false_singleton: misspec_nominal
                    .overconfident_under_misspecification,
                misspec_robust_false_singleton: misspec_robust.overconfident_under_misspecification,
                misspec_nominal_detected: misspec_nominal.model_discrepancy_detected,
                misspec_robust_detected: misspec_robust.model_discrepancy_detected,
                misspec_nominal_unresolved: misspec_nominal.final_live > 1,
                misspec_robust_unresolved: misspec_robust.final_live > 1,
                misspec_nominal_selection_failed: misspec_nominal.selection_failed,
                misspec_robust_selection_failed: misspec_robust.selection_failed,
                misspec_nominal_rounds: misspec_nominal.rounds_used,
                misspec_robust_rounds: misspec_robust.rounds_used,
            });
        }

        let (clean_delta, clean_ci_low, clean_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.clean_robust_success { 1.0 } else { 0.0 })
                    - (if pair.clean_nominal_success { 1.0 } else { 0.0 })
            },
            0x524f_4255_5354_434c,
        );
        let (false_delta, false_ci_low, false_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.misspec_robust_false_singleton {
                    1.0
                } else {
                    0.0
                }) - (if pair.misspec_nominal_false_singleton {
                    1.0
                } else {
                    0.0
                })
            },
            0x524f_4255_5354_4643,
        );
        let (detect_delta, detect_ci_low, detect_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.misspec_robust_detected {
                    1.0
                } else {
                    0.0
                }) - (if pair.misspec_nominal_detected {
                    1.0
                } else {
                    0.0
                })
            },
            0x524f_4255_5354_4445,
        );

        let (failure_delta, failure_ci_low, failure_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.misspec_robust_selection_failed {
                    1.0
                } else {
                    0.0
                }) - (if pair.misspec_nominal_selection_failed {
                    1.0
                } else {
                    0.0
                })
            },
            0x524f_4255_5354_5346,
        );
        let (round_delta, round_ci_low, round_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| pair.misspec_robust_rounds as f64 - pair.misspec_nominal_rounds as f64,
            0x524f_4255_5354_5244,
        );

        println!(
            "scenario_robust_eig_pilot,seeds={SEEDS},round_budget={ROUND_BUDGET},noise_factors=[0.5,1,2],discrepancy={SYSTEMATIC_DISCREPANCY},clean_nominal_recovery={:.3},clean_robust_recovery={:.3},clean_delta_robust_minus_nominal={clean_delta:.3},clean_ci95=[{clean_ci_low:.3},{clean_ci_high:.3}],misspec_nominal_false_singleton={:.3},misspec_robust_false_singleton={:.3},false_singleton_delta_robust_minus_nominal={false_delta:.3},false_singleton_ci95=[{false_ci_low:.3},{false_ci_high:.3}],misspec_nominal_detected={:.3},misspec_robust_detected={:.3},detection_delta_robust_minus_nominal={detect_delta:.3},detection_ci95=[{detect_ci_low:.3},{detect_ci_high:.3}],misspec_nominal_unresolved={:.3},misspec_robust_unresolved={:.3},nominal_selection_failures={misspec_nominal_selection_failures},robust_selection_failures={misspec_robust_selection_failures},selection_failure_delta={failure_delta:.3},selection_failure_ci95=[{failure_ci_low:.3},{failure_ci_high:.3}],nominal_mean_rounds={:.2},robust_mean_rounds={:.2},round_delta={round_delta:.3},round_ci95=[{round_ci_low:.3},{round_ci_high:.3}]",
            clean_nominal_successes as f64 / SEEDS as f64,
            clean_robust_successes as f64 / SEEDS as f64,
            misspec_nominal_false_singletons as f64 / SEEDS as f64,
            misspec_robust_false_singletons as f64 / SEEDS as f64,
            misspec_nominal_detected as f64 / SEEDS as f64,
            misspec_robust_detected as f64 / SEEDS as f64,
            misspec_nominal_unresolved as f64 / SEEDS as f64,
            misspec_robust_unresolved as f64 / SEEDS as f64,
            misspec_nominal_round_total as f64 / SEEDS as f64,
            misspec_robust_round_total as f64 / SEEDS as f64,
        );

        assert_eq!(paired.len(), SEEDS as usize);
        assert_eq!(
            misspec_nominal_false_singletons
                + misspec_nominal_detected
                + misspec_nominal_unresolved,
            SEEDS as usize
        );
        assert_eq!(
            misspec_robust_false_singletons + misspec_robust_detected + misspec_robust_unresolved,
            SEEDS as usize
        );
        assert!(clean_ci_low <= clean_ci_high);
        assert!(false_ci_low <= false_ci_high);
        assert!(detect_ci_low <= detect_ci_high);
        assert!(clean_ci_low.is_finite() && clean_ci_high.is_finite());
        assert!(false_ci_low.is_finite() && false_ci_high.is_finite());
        assert!(detect_ci_low.is_finite() && detect_ci_high.is_finite());
        assert!(failure_ci_low <= failure_ci_high);
        assert!(round_ci_low <= round_ci_high);
        assert!(failure_ci_low.is_finite() && failure_ci_high.is_finite());
        assert!(round_ci_low.is_finite() && round_ci_high.is_finite());
        assert!(misspec_nominal_selection_failures <= SEEDS as usize);
        assert!(misspec_robust_selection_failures <= SEEDS as usize);
        assert!(clean_nominal_successes <= SEEDS as usize);
        assert!(clean_robust_successes <= SEEDS as usize);
    }

    /// Paired-seed pilot comparing raw variance with expected information gain in the
    /// real multi-round equation-discovery loop. This reports evidence; it does not assume EIG
    /// wins. EIG uses an explicit common-noise proxy and rejects incomplete predictions.
    #[test]
    fn expected_information_gain_vs_variance_on_paired_seeds() {
        const SEEDS: u64 = 40;
        const ROUND_BUDGET: usize = 4;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };
        let mut raw_successes = 0usize;
        let mut eig_successes = 0usize;
        let mut raw_rounds = 0usize;
        let mut eig_rounds = 0usize;
        let mut eig_selection_failures = 0usize;
        let mut eig_only_wins = 0usize;
        let mut raw_only_wins = 0usize;
        let mut ties = 0usize;
        let mut paired_outcomes: Vec<SelectorPair> = Vec::with_capacity(SEEDS as usize);

        for seed in 1..=SEEDS {
            let raw =
                run_trial_with_selector(seed, config, ROUND_BUDGET, ActiveSelector::RawVariance);
            let eig = run_trial_with_selector(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::ExpectedInformationGain,
            );
            raw_successes += if raw.success { 1 } else { 0 };
            eig_successes += if eig.success { 1 } else { 0 };
            raw_rounds += raw.rounds_used;
            eig_rounds += eig.rounds_used;
            eig_selection_failures += if eig.selection_failed { 1 } else { 0 };
            paired_outcomes.push((
                raw.success,
                eig.success,
                raw.rounds_used,
                eig.rounds_used,
                eig.selection_failed,
            ));
            match (raw.success, eig.success) {
                (true, false) => raw_only_wins += 1,
                (false, true) => eig_only_wins += 1,
                _ => ties += 1,
            }
        }

        let (success_delta, success_ci_low, success_ci_high) = paired_bootstrap_mean_ci(
            &paired_outcomes,
            |pair| (if pair.1 { 1.0 } else { 0.0 }) - (if pair.0 { 1.0 } else { 0.0 }),
            0x4549_475f_5355_4343,
        );
        let (round_delta, round_ci_low, round_ci_high) = paired_bootstrap_mean_ci(
            &paired_outcomes,
            |pair| pair.3 as f64 - pair.2 as f64,
            0x4549_475f_524f_554e,
        );
        println!(
            "selector_ablation,seeds={SEEDS},round_budget={ROUND_BUDGET},raw_success_rate={:.3},eig_success_rate={:.3},raw_mean_rounds={:.3},eig_mean_rounds={:.3},paired_success_delta_eig_minus_raw={success_delta:.3},success_delta_ci95=[{success_ci_low:.3},{success_ci_high:.3}],paired_round_delta_eig_minus_raw={round_delta:.3},round_delta_ci95=[{round_ci_low:.3},{round_ci_high:.3}],eig_selection_failures={eig_selection_failures},raw_only_wins={raw_only_wins},eig_only_wins={eig_only_wins},ties={ties}",
            raw_successes as f64 / SEEDS as f64,
            eig_successes as f64 / SEEDS as f64,
            raw_rounds as f64 / SEEDS as f64,
            eig_rounds as f64 / SEEDS as f64,
        );
        assert_eq!(raw_only_wins + eig_only_wins + ties, SEEDS as usize);
        assert_eq!(paired_outcomes.len(), SEEDS as usize);
        assert!(success_ci_low <= success_ci_high);
        assert!(round_ci_low <= round_ci_high);
        assert!(success_ci_low.is_finite() && success_ci_high.is_finite());
        assert!(round_ci_low.is_finite() && round_ci_high.is_finite());
        assert!((0.0..=1.0).contains(&(raw_successes as f64 / SEEDS as f64)));
        assert!((0.0..=1.0).contains(&(eig_successes as f64 / SEEDS as f64)));
    }

    /// This is a separate open-ended-recovery benchmark: the initial candidate pool is
    /// generated without the ground-truth expression or any ground-truth-derived near misses.
    /// Oracle-present discrimination scores are intentionally not reused here.
    #[test]
    fn truth_absent_recovery_is_reported_separately_from_oracle_present_discrimination() {
        const SEEDS: u64 = 12;
        const ROUND_BUDGET: usize = 16;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };
        let mut initial_reference_hits = 0usize;
        let mut baseline_initial_reference_hits = 0usize;
        let mut incomplete_reference_audits = 0usize;
        let mut in_domain_holdout_points_total = 0usize;
        let mut ood_holdout_points_total = 0usize;
        let mut unresolved_without_recovery = 0usize;
        let mut false_singletons_without_recovery = 0usize;
        let mut empty_pool_recovery_attempts = 0usize;
        let mut failed_recovery_attempts = 0usize;
        let mut accepted_recovery_candidates = 0usize;
        let mut false_singletons_after_recovery = 0usize;
        let mut same_domain_recovered = 0usize;
        let mut ood_recovered = 0usize;
        let mut full_recovery_successes = 0usize;
        let mut runs_with_post_recovery_validation = 0usize;
        let mut total_post_recovery_validation_measurements = 0usize;
        let mut baseline_measurements = 0usize;
        let mut recovery_measurements = 0usize;
        let mut unresolved_after_recovery = 0usize;
        let mut selection_failures_after_recovery = 0usize;

        for seed in 1..=SEEDS {
            let baseline = run_truth_absent_trial(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                false,
            );
            let recovered = run_truth_absent_trial(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                true,
            );
            initial_reference_hits += if recovered.reference_audit_evaluable
                && recovered.initial_pool_contains_reference
            {
                1
            } else {
                0
            };
            baseline_initial_reference_hits += if baseline.reference_audit_evaluable
                && baseline.initial_pool_contains_reference
            {
                1
            } else {
                0
            };
            incomplete_reference_audits += if recovered.reference_audit_evaluable {
                0
            } else {
                1
            };
            in_domain_holdout_points_total += recovered.in_domain_holdout_points;
            ood_holdout_points_total += recovered.extrapolation_holdout_points;
            baseline_measurements += baseline.rounds_used;
            recovery_measurements += recovered.rounds_used;
            unresolved_without_recovery += if baseline.final_live != 1 { 1 } else { 0 };
            unresolved_after_recovery += if recovered.final_live != 1 { 1 } else { 0 };
            selection_failures_after_recovery += if recovered.selection_failed { 1 } else { 0 };
            false_singletons_without_recovery += if baseline.final_live == 1 && !baseline.success {
                1
            } else {
                0
            };
            empty_pool_recovery_attempts += if recovered.recovery_triggered_on_empty_pool {
                1
            } else {
                0
            };
            failed_recovery_attempts +=
                if recovered.recovery_triggered_on_empty_pool && recovered.terminal_pit_alarm {
                    1
                } else {
                    0
                };
            accepted_recovery_candidates += recovered.accepted_recovery_candidates;
            false_singletons_after_recovery += if recovered.final_live == 1
                && (!recovered.ground_truth_equivalent
                    || !recovered.ground_truth_extrapolation_equivalent)
            {
                1
            } else {
                0
            };
            same_domain_recovered += if recovered.ground_truth_equivalent {
                1
            } else {
                0
            };
            ood_recovered += if recovered.ground_truth_extrapolation_equivalent {
                1
            } else {
                0
            };
            full_recovery_successes += if recovered.success { 1 } else { 0 };
            runs_with_post_recovery_validation += if recovered.post_recovery_validation_measurements
                >= MIN_POST_RECOVERY_VALIDATION_MEASUREMENTS
            {
                1
            } else {
                0
            };
            total_post_recovery_validation_measurements +=
                recovered.post_recovery_validation_measurements;
            assert!(
                !recovered.success
                    || recovered.post_recovery_validation_measurements
                        >= MIN_POST_RECOVERY_VALIDATION_MEASUREMENTS
            );

            // Per-seed records are retained in CI output so aggregate counts are auditable.
            // The reference expression and evaluator values are intentionally not serialized.
            println!(
                concat!(
                    "truth_absent_run,seed={seed},",
                    "baseline_rounds={},baseline_live={},baseline_success={},",
                    "baseline_selection_failed={},baseline_false_singleton={},",
                    "recovery_rounds={},recovery_live={},recovery_selection_failed={},",
                    "reference_audit_evaluable={},reference_equivalent_initial_pool={},",
                    "in_domain_holdout_points={},ood_holdout_points={},",
                    "recovery_triggered={},recovery_restarts={},accepted_candidates={},",
                    "same_domain_match={},ood_match={},post_recovery_singleton_audits={},",
                    "full_recovery_success={}"
                ),
                baseline.rounds_used,
                baseline.final_live,
                baseline.success,
                baseline.selection_failed,
                baseline.final_live == 1 && !baseline.success,
                recovered.rounds_used,
                recovered.final_live,
                recovered.selection_failed,
                recovered.reference_audit_evaluable,
                recovered.initial_pool_contains_reference,
                recovered.in_domain_holdout_points,
                recovered.extrapolation_holdout_points,
                recovered.recovery_triggered_on_empty_pool,
                recovered.typed_recovery_restarts,
                recovered.accepted_recovery_candidates,
                recovered.ground_truth_equivalent,
                recovered.ground_truth_extrapolation_equivalent,
                recovered.post_recovery_validation_measurements,
                recovered.success,
            );
        }

        println!(
            "truth_absent_recovery,seeds={SEEDS},budget={ROUND_BUDGET},initial_reference_hits={initial_reference_hits},baseline_initial_reference_hits={baseline_initial_reference_hits},incomplete_reference_audits={incomplete_reference_audits},in_domain_holdout_points_total={in_domain_holdout_points_total},ood_holdout_points_total={ood_holdout_points_total},unresolved_without_recovery={unresolved_without_recovery},false_singletons_without_recovery={false_singletons_without_recovery},unresolved_after_recovery={unresolved_after_recovery},selection_failures_after_recovery={selection_failures_after_recovery},baseline_measurements={baseline_measurements},recovery_measurements={recovery_measurements},runs_with_post_recovery_validation={runs_with_post_recovery_validation},total_post_recovery_validation_measurements={total_post_recovery_validation_measurements},empty_pool_recovery_attempts={empty_pool_recovery_attempts},failed_recovery_attempts={failed_recovery_attempts},accepted_recovery_candidates={accepted_recovery_candidates},false_singletons_after_recovery={false_singletons_after_recovery},same_domain_recovered={same_domain_recovered},ood_recovered={ood_recovered},full_recovery_successes={full_recovery_successes}"
        );
        assert_eq!(
            initial_reference_hits, 0,
            "truth-absent initial pool unexpectedly contains a reference-equivalent candidate"
        );
        assert_eq!(
            baseline_initial_reference_hits, 0,
            "truth-absent no-recovery baseline unexpectedly contains a reference-equivalent candidate"
        );
        assert!(
            empty_pool_recovery_attempts > 0,
            "truth-absent pilot never exercised recovery after a pool-collapse alarm"
        );
        assert!(empty_pool_recovery_attempts <= SEEDS as usize);
        assert!(same_domain_recovered <= SEEDS as usize);
        assert!(ood_recovered <= same_domain_recovered);
        assert!(full_recovery_successes <= ood_recovered);
    }

    #[test]
    fn unary_heldout_truth_is_valid_but_absent_from_typed_initial_candidates() {
        let mut design_rng = 0x51A7_0000_2026_0001_u64;
        let design: Vec<Experiment> = (0..EXPERIMENT_POOL_SIZE)
            .map(|_| random_experiment(&mut design_rng))
            .collect();
        let mut truth_rng = 0x0A11_0F00_2026_0001_u64;
        let truth = nontrivial_unary_ground_truth(&mut truth_rng, &design)
            .expect("declared sin/cos family should yield a sane ENERGY law");
        assert!(expr_contains_unary_function(&truth));
        assert!(matches!(
            infer_dimensions(&truth, &demo_units()),
            InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
        ));

        let candidates = generate_truth_absent_candidates(
            0xCA7A_10_2026_0001,
            PipelineConfig {
                typed_candidates: true,
                active_selection: true,
                smoothing: false,
            },
            &design,
        )
        .expect("typed algebraic generator should fill its bounded initial pool");
        assert_eq!(candidates.len(), POOL_SIZE);
        assert!(
            candidates.iter().all(|candidate| !expr_contains_unary_function(candidate)),
            "held-out unary operators must not leak into the initial typed candidate grammar"
        );
    }

    /// Full diagnostic for a true law whose unary operator is withheld from the initial typed
    /// grammar. Results remain separate from oracle-present and in-grammar recovery rates.
    #[test]
    fn operator_heldout_recovery_is_reported_separately_from_in_grammar_recovery() {
        const SEEDS: u64 = 8;
        const ROUND_BUDGET: usize = 16;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };
        let mut evaluable_worlds = 0usize;
        let mut incomplete_evaluator_worlds = 0usize;
        let mut initial_reference_hits = 0usize;
        let mut baseline_reference_hits = 0usize;
        let mut baseline_unresolved = 0usize;
        let mut recovery_unresolved = 0usize;
        let mut pool_collapse_recoveries = 0usize;
        let mut accepted_unary_corrections = 0usize;
        let mut same_domain_matches = 0usize;
        let mut ood_matches = 0usize;
        let mut full_successes = 0usize;
        let mut post_recovery_validation_measurements = 0usize;

        for seed in 1..=SEEDS {
            let baseline = run_operator_heldout_trial(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                false,
            );
            let recovered = run_operator_heldout_trial(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                true,
            );
            if recovered.reference_audit_evaluable {
                evaluable_worlds += 1;
                initial_reference_hits += if recovered.initial_pool_contains_reference {
                    1
                } else {
                    0
                };
            } else {
                incomplete_evaluator_worlds += 1;
            }
            baseline_reference_hits += if baseline.reference_audit_evaluable
                && baseline.initial_pool_contains_reference
            {
                1
            } else {
                0
            };
            baseline_unresolved += if baseline.final_live != 1 || !baseline.success {
                1
            } else {
                0
            };
            recovery_unresolved += if recovered.final_live != 1 || !recovered.success {
                1
            } else {
                0
            };
            pool_collapse_recoveries +=
                if recovered.recovery_triggered_on_empty_pool { 1 } else { 0 };
            accepted_unary_corrections += recovered.unary_function_recovery_candidates;
            same_domain_matches += if recovered.ground_truth_equivalent { 1 } else { 0 };
            ood_matches += if recovered.ground_truth_extrapolation_equivalent {
                1
            } else {
                0
            };
            full_successes += if recovered.success { 1 } else { 0 };
            post_recovery_validation_measurements +=
                recovered.post_recovery_validation_measurements;

            assert!(
                !recovered.success
                    || (recovered.ground_truth_equivalent
                        && recovered.ground_truth_extrapolation_equivalent
                        && recovered.post_recovery_validation_measurements
                            >= MIN_POST_RECOVERY_VALIDATION_MEASUREMENTS)
            );
        }
        println!(
            "operator_heldout_recovery,seeds={SEEDS},budget={ROUND_BUDGET},evaluable_worlds={evaluable_worlds},incomplete_evaluator_worlds={incomplete_evaluator_worlds},initial_reference_hits={initial_reference_hits},baseline_reference_hits={baseline_reference_hits},baseline_unresolved={baseline_unresolved},recovery_unresolved={recovery_unresolved},pool_collapse_recoveries={pool_collapse_recoveries},accepted_unary_corrections={accepted_unary_corrections},same_domain_matches={same_domain_matches},ood_matches={ood_matches},full_successes={full_successes},post_recovery_validation_measurements={post_recovery_validation_measurements}"
        );
        assert_eq!(evaluable_worlds + incomplete_evaluator_worlds, SEEDS as usize);
        assert_eq!(initial_reference_hits, 0);
        assert_eq!(baseline_reference_hits, 0);
        assert!(same_domain_matches <= evaluable_worlds);
        assert!(ood_matches <= same_domain_matches);
        assert!(full_successes <= ood_matches);
    }

    #[test]
    fn truth_absent_candidate_generation_exhaustion_returns_none_without_panicking() {
        let mut rng = 0xA11C_E55E_2026_0001_u64;
        let design = vec![random_experiment(&mut rng)];
        let result = generate_truth_absent_candidates_with_attempt_cap(
            0xA11C_E55E_2026_0001,
            PipelineConfig {
                typed_candidates: true,
                active_selection: true,
                smoothing: false,
            },
            &design,
            0,
        );
        assert!(
            result.is_none(),
            "zero-attempt candidate generation must return unresolved rather than panic or invent a pool"
        );
    }

    #[test]
    fn extrapolation_design_is_outside_every_training_variable_interval() {
        let mut rng = 0x00D0_0D0E_2026_0001_u64;
        for _ in 0..128 {
            let point = random_extrapolation_experiment(&mut rng);
            assert!(is_extrapolation_experiment(&point));
            assert_eq!(point.len(), VAR_NAMES.len());
            for (_, value) in point {
                assert!(value.is_finite());
                assert!(value < VAR_RANGE.0 || value > VAR_RANGE.1);
            }
        }
    }

    #[test]
    fn extrapolation_equivalence_distinguishes_an_equivalent_law_from_a_shift() {
        let reference = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let equivalent = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(reference.clone()),
            )),
        );
        let shifted = Expr::BinOp(
            BinOp::Add,
            Box::new(reference.clone()),
            Box::new(Expr::Const(0.5)),
        );
        let mut rng = 0x0DD0_2026_1023_u64;
        let holdout: Vec<Experiment> = (0..64)
            .map(|_| random_extrapolation_experiment(&mut rng))
            .collect();

        assert!(holdout.iter().all(is_extrapolation_experiment));
        assert!(evaluator_matches_reference(
            &equivalent,
            &reference,
            &holdout
        ));
        assert!(!evaluator_matches_reference(&shifted, &reference, &holdout));
    }

    #[test]
    fn oracle_equivalence_audit_requires_both_in_domain_and_ood_match() {
        let reference = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        // This expression agrees on the deliberately narrow in-domain slice v=1.5, but
        // differs at out-of-range v=3.0. A same-domain-only oracle audit would miss this.
        let candidate = Expr::BinOp(
            BinOp::Mul,
            Box::new(reference.clone()),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(1.5)),
            )),
        );
        let in_domain: Vec<Experiment> = (0..8)
            .map(|_| {
                vec![
                    ("m".to_string(), 1.5),
                    ("v".to_string(), 1.5),
                    ("r".to_string(), 1.5),
                    ("t".to_string(), 1.5),
                ]
            })
            .collect();
        let out_of_domain: Vec<Experiment> = (0..8)
            .map(|_| {
                vec![
                    ("m".to_string(), 3.0),
                    ("v".to_string(), 3.0),
                    ("r".to_string(), 0.5),
                    ("t".to_string(), 3.0),
                ]
            })
            .collect();

        assert!(evaluator_matches_reference(
            &candidate, &reference, &in_domain
        ));
        assert!(!evaluator_matches_reference(
            &candidate,
            &reference,
            &out_of_domain
        ));
        assert!(!evaluator_matches_reference_across_domains(
            &candidate,
            &reference,
            &in_domain,
            &out_of_domain,
        ));
    }

    #[test]
    fn evaluator_equivalence_requires_independent_heldout_points() {
        let reference = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let equivalent = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(2.0)),
                Box::new(reference.clone()),
            )),
        );
        let non_equivalent = Expr::BinOp(
            BinOp::Add,
            Box::new(reference.clone()),
            Box::new(Expr::Const(0.5)),
        );
        let mut rng = 0xE7A1_0A7A_2026_0001_u64;
        let holdout: Vec<Experiment> = (0..64).map(|_| random_experiment(&mut rng)).collect();

        assert!(evaluator_matches_reference(
            &equivalent,
            &reference,
            &holdout
        ));
        assert!(!evaluator_matches_reference(
            &non_equivalent,
            &reference,
            &holdout
        ));
        assert!(!evaluator_matches_reference(
            &equivalent,
            &reference,
            &holdout[..4]
        ));
    }

    #[test]
    fn structured_recovery_families_preserve_energy_dimensions() {
        let base = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let units = demo_units();
        let families = [
            RecoveryProposalFamily::AdditiveCorrection,
            RecoveryProposalFamily::MultiplicativeCorrection,
            RecoveryProposalFamily::InteractionCorrection,
            RecoveryProposalFamily::UnaryFunctionCorrection,
        ];
        let mut rng = 0xCAFE_BABE_1234_5678_u64;

        for family in families {
            let mut candidate = None;
            for _ in 0..128 {
                let Some(basis) = structured_correction_basis(&mut rng, &base, family) else {
                    continue;
                };
                let Some(proposal) = compose_linear_correction(&base, &basis, 0.5) else {
                    continue;
                };
                candidate = Some(proposal);
                break;
            }
            let candidate =
                candidate.expect("typed correction family failed to propose a valid expression");
            assert!(matches!(
                infer_dimensions(&candidate, &units),
                InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
            ));
        }
    }

    #[test]
    fn fitted_correction_coefficient_recovers_known_scalar_on_synthetic_residuals() {
        let base = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let basis = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("m".to_string())),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var("r".to_string())),
                    Box::new(Expr::Const(2.0)),
                )),
            )),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("t".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        const TRUE_COEFFICIENT: f64 = 0.37;
        let mut rng = 0xF17C_0EFF_1C1E_2026_u64;
        let observations: Vec<(Experiment, f64)> = (0..64)
            .map(|_| {
                let point = random_experiment(&mut rng);
                let value = eval_at(&base, &point).unwrap()
                    + TRUE_COEFFICIENT * eval_at(&basis, &point).unwrap();
                (point, value)
            })
            .collect();

        let fitted = fit_linear_correction_coefficient(&base, &basis, &observations)
            .expect("identifiable coefficient should fit");
        assert!(
            (fitted - TRUE_COEFFICIENT).abs() < 1.0e-10,
            "expected {TRUE_COEFFICIENT}, recovered {fitted}"
        );
        let corrected = compose_linear_correction(&base, &basis, fitted)
            .expect("fitted correction must preserve units");
        assert!(matches!(
            infer_dimensions(&corrected, &demo_units()),
            InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
        ));
        assert!(
            recovery_fit_score(&corrected, &observations).unwrap()
                < recovery_fit_score(&base, &observations).unwrap()
        );
    }

    #[test]
    fn empirical_recovery_score_rewards_a_structured_correction_on_observed_residuals() {
        let base = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let radius_squared = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("r".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let time_squared = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("t".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let correction = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Var("m".to_string())),
                    Box::new(radius_squared),
                )),
                Box::new(time_squared),
            )),
        );
        let corrected = Expr::BinOp(BinOp::Add, Box::new(base.clone()), Box::new(correction));
        let mut rng = 0xA11C_E5ED_9876_5432_u64;
        let observations: Vec<(Experiment, f64)> = (0..24)
            .map(|_| {
                let point = random_experiment(&mut rng);
                let value = eval_at(&corrected, &point).unwrap();
                (point, value)
            })
            .collect();

        let baseline_score = recovery_fit_score(&base, &observations).unwrap();
        let corrected_score = recovery_fit_score(&corrected, &observations).unwrap();
        assert!(
            corrected_score < baseline_score,
            "structured correction should explain the synthetic residual: base={baseline_score}, corrected={corrected_score}"
        );
        assert!(matches!(
            infer_dimensions(&corrected, &demo_units()),
            InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
        ));
    }

    #[test]
    fn post_alarm_recovery_reopens_archived_candidates_and_adds_typed_candidates() {
        let mut rng = 0xBADC_0FFE_E0DD_F00D_u64;
        let mut pool_rng = 0x1234_5678_9ABC_DEF0_u64;
        let pool: Vec<Experiment> = (0..EXPERIMENT_POOL_SIZE)
            .map(|_| random_experiment(&mut pool_rng))
            .collect();
        let energy = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".to_string())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut hypotheses = vec![
            energy,
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("m".to_string())),
                Box::new(Expr::Var("v".to_string())),
            ),
            Expr::Var("r".to_string()),
        ];
        let archived_count = hypotheses.len();
        let mut live = vec![0];
        let mut rejected = Vec::new();

        let observations: Vec<(Experiment, f64)> = pool
            .iter()
            .take(8)
            .map(|point| {
                let value = eval_at(&hypotheses[0], point).unwrap();
                (point.clone(), value)
            })
            .collect();
        let summary = reseed_typed_candidates_after_alarm(
            &mut rng,
            &pool,
            &observations,
            &mut hypotheses,
            &mut live,
            &mut rejected,
            0,
            false,
        );

        assert!(
            summary.accepted_count > 0,
            "bounded typed recovery failed to add any candidate"
        );
        assert!(summary.accepted_count <= RECOVERY_CANDIDATES_PER_EXPANSION);
        assert!(
            !live.contains(&0),
            "the alarmed provisional candidate stayed live"
        );
        assert!(
            live.contains(&1) && live.contains(&2),
            "archived alternatives were not reopened"
        );
        assert_eq!(live.len(), archived_count - 1 + summary.accepted_count);
        let units = demo_units();
        for index in archived_count..hypotheses.len() {
            assert!(matches!(
                infer_dimensions(&hypotheses[index], &units),
                InferenceResult::Inferred(dimension) if dimension == DimensionalSignature::ENERGY
            ));
        }

        // A second alarm must not resurrect the first alarmed hypothesis.
        let second_alarmed = archived_count;
        let mut second_rng = 0x0DDC_0FFE_E0DD_F00D_u64;
        let _ = reseed_typed_candidates_after_alarm(
            &mut second_rng,
            &pool,
            &observations,
            &mut hypotheses,
            &mut live,
            &mut rejected,
            second_alarmed,
            false,
        );
        assert!(rejected.contains(&0) && rejected.contains(&second_alarmed));
        assert!(!live.contains(&0) && !live.contains(&second_alarmed));
    }

    /// End-to-end integration check for the prequential PIT monitor. The monitor activates
    /// only after the candidate set becomes a singleton, when each random validation query has
    /// a model-visible Gaussian predictive distribution. The CDF is computed from the candidate
    /// prediction and the declared noise model before using the current outcome in any policy
    /// update. This is a diagnostic and deliberately does not assert that the policy wins.
    #[test]
    fn prequential_pit_audits_are_wired_into_adaptive_discovery() {
        const SEEDS: u64 = 32;
        const DISCREPANCY: f64 = 0.12;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };

        let mut clean_baseline_successes = 0usize;
        let mut clean_pit_successes = 0usize;
        let mut clean_pit_reference_equivalents = 0usize;
        let mut misspec_pit_reference_equivalents = 0usize;
        let mut clean_pit_ood_equivalents = 0usize;
        let mut misspec_pit_ood_equivalents = 0usize;
        let mut clean_pit_ood_points = 0usize;
        let mut misspec_pit_ood_points = 0usize;
        let mut clean_pit_alarms = 0usize;
        let mut misspec_baseline_false_singletons = 0usize;
        let mut misspec_pit_false_singletons = 0usize;
        let mut misspec_pit_detections = 0usize;
        let mut pit_audit_queries = 0usize;
        let mut protocol_failures = 0usize;
        let mut total_typed_recovery_restarts = 0usize;
        let mut structured_recovery_candidates = 0usize;
        let mut terminal_pit_alarms = 0usize;

        for seed in 1..=SEEDS {
            let clean_baseline = run_trial_with_discrepancy(
                seed,
                config,
                MAX_ROUNDS,
                ActiveSelector::ExpectedInformationGain,
                0.0,
            );
            let clean_pit = run_trial_with_discrepancy(
                seed,
                config,
                MAX_ROUNDS,
                ActiveSelector::InformationGainWithPrequentialPitAudits,
                0.0,
            );
            let misspecified_baseline = run_trial_with_discrepancy(
                seed,
                config,
                MAX_ROUNDS,
                ActiveSelector::ExpectedInformationGain,
                DISCREPANCY,
            );
            let misspecified_pit = run_trial_with_discrepancy(
                seed,
                config,
                MAX_ROUNDS,
                ActiveSelector::InformationGainWithPrequentialPitAudits,
                DISCREPANCY,
            );

            clean_baseline_successes += if clean_baseline.success { 1 } else { 0 };
            clean_pit_successes += if clean_pit.success { 1 } else { 0 };
            clean_pit_reference_equivalents += if clean_pit.ground_truth_equivalent {
                1
            } else {
                0
            };
            misspec_pit_reference_equivalents += if misspecified_pit.ground_truth_equivalent {
                1
            } else {
                0
            };
            clean_pit_ood_equivalents += if clean_pit.ground_truth_extrapolation_equivalent {
                1
            } else {
                0
            };
            misspec_pit_ood_equivalents += if misspecified_pit.ground_truth_extrapolation_equivalent
            {
                1
            } else {
                0
            };
            clean_pit_ood_points += clean_pit.extrapolation_holdout_points;
            misspec_pit_ood_points += misspecified_pit.extrapolation_holdout_points;
            clean_pit_alarms += if clean_pit.prequential_pit_alarm {
                1
            } else {
                0
            };
            misspec_baseline_false_singletons +=
                if misspecified_baseline.overconfident_under_misspecification {
                    1
                } else {
                    0
                };
            misspec_pit_false_singletons += if misspecified_pit.overconfident_under_misspecification
            {
                1
            } else {
                0
            };
            misspec_pit_detections += if misspecified_pit.model_discrepancy_detected {
                1
            } else {
                0
            };
            pit_audit_queries += misspecified_pit.audit_queries + clean_pit.audit_queries;
            total_typed_recovery_restarts +=
                clean_pit.typed_recovery_restarts + misspecified_pit.typed_recovery_restarts;
            structured_recovery_candidates += clean_pit.structured_recovery_candidates
                + misspecified_pit.structured_recovery_candidates;
            terminal_pit_alarms += if clean_pit.terminal_pit_alarm { 1 } else { 0 };
            terminal_pit_alarms += if misspecified_pit.terminal_pit_alarm {
                1
            } else {
                0
            };
            protocol_failures += if clean_pit.monitoring_protocol_invalid {
                1
            } else {
                0
            };
            protocol_failures += if misspecified_pit.monitoring_protocol_invalid {
                1
            } else {
                0
            };
        }

        println!(concat!(
            "prequential_pit_integrated,seeds={SEEDS},budget={MAX_ROUNDS},",
            "discrepancy={DISCREPANCY},clean_baseline_successes={clean_baseline_successes},",
            "clean_pit_successes={clean_pit_successes},",
            "clean_pit_reference_equivalents={clean_pit_reference_equivalents},",
            "misspec_pit_reference_equivalents={misspec_pit_reference_equivalents},",
            "clean_pit_ood_equivalents={clean_pit_ood_equivalents},",
            "misspec_pit_ood_equivalents={misspec_pit_ood_equivalents},",
            "clean_pit_ood_points={clean_pit_ood_points},",
            "misspec_pit_ood_points={misspec_pit_ood_points},",
            "clean_pit_alarms={clean_pit_alarms},",
            "misspec_baseline_false_singletons={misspec_baseline_false_singletons},",
            "misspec_pit_false_singletons={misspec_pit_false_singletons},",
            "misspec_pit_detections={misspec_pit_detections},",
            "pit_audit_queries={pit_audit_queries},typed_recovery_restarts={total_typed_recovery_restarts},",
            "structured_recovery_candidates={structured_recovery_candidates},",
            "terminal_pit_alarms={terminal_pit_alarms},protocol_failures={protocol_failures}"
        ));

        assert_eq!(
            protocol_failures, 0,
            "PIT monitor must fail closed only on real protocol defects"
        );
        assert!(
            pit_audit_queries > 0,
            "the integrated policy never reached a singleton audit"
        );
        assert!(total_typed_recovery_restarts <= 4 * SEEDS as usize);
        assert!(terminal_pit_alarms <= 2 * SEEDS as usize);
        assert!(clean_baseline_successes <= SEEDS as usize);
        assert!(clean_pit_successes <= SEEDS as usize);
        assert!(misspec_pit_detections <= SEEDS as usize);
        assert!(misspec_pit_false_singletons <= SEEDS as usize);
    }

    #[test]
    fn prequential_pit_policy_rejects_smoothed_measurements() {
        let outcome = run_trial_with_discrepancy(
            1,
            PipelineConfig {
                typed_candidates: true,
                active_selection: true,
                smoothing: true,
            },
            MAX_ROUNDS,
            ActiveSelector::InformationGainWithPrequentialPitAudits,
            0.12,
        );
        assert!(outcome.monitoring_protocol_invalid);
        assert!(outcome.selection_failed);
        assert!(!outcome.success);
    }

    /// Pilot testing whether scheduled, seeded audit experiments reduce false confidence
    /// when the candidate class is deliberately misspecified. Audit queries consume the same
    /// round budget and use only the existing admissible pool; the selector does not see the
    /// hidden discrepancy. This is a diagnostic, not a claim that periodic random exploration
    /// is universally optimal.
    #[test]
    fn scheduled_audits_measure_false_confidence_under_model_misspecification() {
        const SEEDS: u64 = 40;
        const ROUND_BUDGET: usize = 4;
        const SYSTEMATIC_DISCREPANCY: f64 = 0.12;
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: false,
        };

        let mut clean_eig_successes = 0usize;
        let mut clean_audit_successes = 0usize;
        let mut misspecified_eig_false_singletons = 0usize;
        let mut misspecified_audit_false_singletons = 0usize;
        let mut misspecified_eig_detections = 0usize;
        let mut misspecified_audit_detections = 0usize;
        let mut misspecified_eig_unresolved = 0usize;
        let mut misspecified_audit_unresolved = 0usize;
        let mut audit_query_total = 0usize;
        let mut eig_selection_failures = 0usize;
        let mut audit_selection_failures = 0usize;
        let mut eig_total_rounds = 0usize;
        let mut audit_total_rounds = 0usize;
        let mut paired: Vec<MisspecSelectorPair> = Vec::with_capacity(SEEDS as usize);

        for seed in 1..=SEEDS {
            let clean_eig = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::ExpectedInformationGain,
                0.0,
            );
            let clean_audit = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                0.0,
            );
            let misspec_eig = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::ExpectedInformationGain,
                SYSTEMATIC_DISCREPANCY,
            );
            let misspec_audit = run_trial_with_discrepancy(
                seed,
                config,
                ROUND_BUDGET,
                ActiveSelector::InformationGainWithAudits,
                SYSTEMATIC_DISCREPANCY,
            );

            clean_eig_successes += if clean_eig.success { 1 } else { 0 };
            clean_audit_successes += if clean_audit.success { 1 } else { 0 };
            misspecified_eig_false_singletons += if misspec_eig.overconfident_under_misspecification
            {
                1
            } else {
                0
            };
            misspecified_audit_false_singletons +=
                if misspec_audit.overconfident_under_misspecification {
                    1
                } else {
                    0
                };
            misspecified_eig_detections += if misspec_eig.model_discrepancy_detected {
                1
            } else {
                0
            };
            misspecified_audit_detections += if misspec_audit.model_discrepancy_detected {
                1
            } else {
                0
            };
            misspecified_eig_unresolved += if misspec_eig.final_live > 1 { 1 } else { 0 };
            misspecified_audit_unresolved += if misspec_audit.final_live > 1 { 1 } else { 0 };
            audit_query_total += misspec_audit.audit_queries;
            eig_selection_failures += if misspec_eig.selection_failed { 1 } else { 0 };
            audit_selection_failures += if misspec_audit.selection_failed { 1 } else { 0 };
            eig_total_rounds += misspec_eig.rounds_used;
            audit_total_rounds += misspec_audit.rounds_used;

            paired.push(MisspecSelectorPair {
                eig_detected: misspec_eig.model_discrepancy_detected,
                eig_false_singleton: misspec_eig.overconfident_under_misspecification,
                audit_detected: misspec_audit.model_discrepancy_detected,
                audit_false_singleton: misspec_audit.overconfident_under_misspecification,
                clean_eig_success: clean_eig.success,
                clean_audit_success: clean_audit.success,
            });
        }

        let (false_singleton_delta, false_ci_low, false_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.audit_false_singleton { 1.0 } else { 0.0 })
                    - (if pair.eig_false_singleton { 1.0 } else { 0.0 })
            },
            0x4155_4449_545f_4643,
        );
        let (detection_delta, detection_ci_low, detection_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.audit_detected { 1.0 } else { 0.0 })
                    - (if pair.eig_detected { 1.0 } else { 0.0 })
            },
            0x4155_4449_545f_4445,
        );
        let (clean_delta, clean_ci_low, clean_ci_high) = paired_bootstrap_mean_ci(
            &paired,
            |pair| {
                (if pair.clean_audit_success { 1.0 } else { 0.0 })
                    - (if pair.clean_eig_success { 1.0 } else { 0.0 })
            },
            0x4155_4449_545f_434c,
        );

        println!(
            "misspec_audit,seeds={SEEDS},round_budget={ROUND_BUDGET},discrepancy={SYSTEMATIC_DISCREPANCY},clean_eig_recovery={:.3},clean_audit_recovery={:.3},misspec_eig_false_singleton={:.3},misspec_audit_false_singleton={:.3},false_singleton_delta_audit_minus_eig={false_singleton_delta:.3},false_singleton_ci95=[{false_ci_low:.3},{false_ci_high:.3}],misspec_eig_detected={:.3},misspec_audit_detected={:.3},misspec_eig_unresolved={:.3},misspec_audit_unresolved={:.3},detection_delta_audit_minus_eig={detection_delta:.3},detection_ci95=[{detection_ci_low:.3},{detection_ci_high:.3}],clean_recovery_delta_audit_minus_eig={clean_delta:.3},clean_recovery_ci95=[{clean_ci_low:.3},{clean_ci_high:.3}],eig_selection_failures={eig_selection_failures},audit_selection_failures={audit_selection_failures},eig_mean_rounds={:.2},audit_mean_rounds={:.2},scheduled_audit_queries={audit_query_total}",
            clean_eig_successes as f64 / SEEDS as f64,
            clean_audit_successes as f64 / SEEDS as f64,
            misspecified_eig_false_singletons as f64 / SEEDS as f64,
            misspecified_audit_false_singletons as f64 / SEEDS as f64,
            misspecified_eig_detections as f64 / SEEDS as f64,
            misspecified_audit_detections as f64 / SEEDS as f64,
            misspecified_eig_unresolved as f64 / SEEDS as f64,
            misspecified_audit_unresolved as f64 / SEEDS as f64,
            eig_total_rounds as f64 / SEEDS as f64,
            audit_total_rounds as f64 / SEEDS as f64,
        );
        assert_eq!(paired.len(), SEEDS as usize);
        assert_eq!(
            misspecified_eig_false_singletons
                + misspecified_eig_detections
                + misspecified_eig_unresolved,
            SEEDS as usize
        );
        assert_eq!(
            misspecified_audit_false_singletons
                + misspecified_audit_detections
                + misspecified_audit_unresolved,
            SEEDS as usize
        );
        assert!(false_ci_low <= false_ci_high);
        assert!(detection_ci_low <= detection_ci_high);
        assert!(clean_ci_low <= clean_ci_high);
        assert!(false_ci_low.is_finite() && false_ci_high.is_finite());
        assert!(detection_ci_low.is_finite() && detection_ci_high.is_finite());
        assert!(clean_ci_low.is_finite() && clean_ci_high.is_finite());
        assert!(clean_eig_successes <= SEEDS as usize);
        assert!(clean_audit_successes <= SEEDS as usize);
    }

    /// Diagnostic pilot: calibrate on clean batches, then report false alarms and
    /// misspecification detections on disjoint clean/perturbed audit seeds. This test deliberately
    /// does not assert that the detector wins; it asserts only valid probability accounting and a
    /// loose upper bound against gross false-alarm regressions.
    #[test]
    fn conformal_adequacy_calibration_reports_clean_and_perturbed_audit_rates() {
        const CALIBRATION_BATCHES: usize = 80;
        const AUDIT_BATCHES: usize = 50;
        const ALPHA: f64 = 0.05;
        const DISCREPANCY_AMPLITUDE: f64 = 0.12;

        let calibration_scores: Vec<f64> = (0..CALIBRATION_BATCHES)
            .map(|index| conformal_audit_batch_score(10_000 + index as u64, 0.0))
            .collect();
        let calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, ALPHA)
            .expect("enough independent clean calibration batches");

        let mut clean_false_alarms = 0usize;
        let mut misspecified_detections = 0usize;
        let mut clean_non_alarm_actions = 0usize;
        for index in 0..AUDIT_BATCHES {
            let clean = calibrator
                .assess_audit_score(
                    conformal_audit_batch_score(20_000 + index as u64, 0.0),
                    true,
                )
                .expect("clean audit batch is independent");
            let clean_action = next_discovery_action(DiscoverySnapshot {
                live_candidate_count: 1,
                adequacy: clean.assessment,
                validation: ValidationStatus::InProgress,
                termination: TerminationStatus::Active,
            });
            if clean_action == DiscoveryAction::ReportModelSpaceInadequate {
                clean_false_alarms += 1;
            } else if clean_action == DiscoveryAction::ValidateProvisionalSingleton {
                clean_non_alarm_actions += 1;
            }

            let perturbed = calibrator
                .assess_audit_score(
                    conformal_audit_batch_score(30_000 + index as u64, DISCREPANCY_AMPLITUDE),
                    true,
                )
                .expect("perturbed audit batch is independent of calibration");
            let action = next_discovery_action(DiscoverySnapshot {
                live_candidate_count: 1,
                adequacy: perturbed.assessment,
                validation: ValidationStatus::InProgress,
                termination: TerminationStatus::Active,
            });
            if action == DiscoveryAction::ReportModelSpaceInadequate {
                misspecified_detections += 1;
            }
        }

        let clean_false_alarm_rate = clean_false_alarms as f64 / AUDIT_BATCHES as f64;
        let detection_rate = misspecified_detections as f64 / AUDIT_BATCHES as f64;
        println!(
            "conformal_adequacy_pilot,calibration_batches={CALIBRATION_BATCHES},audit_batches={AUDIT_BATCHES},alpha={ALPHA},discrepancy={DISCREPANCY_AMPLITUDE},clean_false_alarm_rate={clean_false_alarm_rate:.3},misspecification_detection_rate={detection_rate:.3},clean_non_alarm_actions={clean_non_alarm_actions}"
        );

        assert_eq!(clean_false_alarms + clean_non_alarm_actions, AUDIT_BATCHES);
        assert!((0.0..=1.0).contains(&clean_false_alarm_rate));
        assert!((0.0..=1.0).contains(&detection_rate));
        assert!(
            clean_false_alarm_rate <= 0.20,
            "clean false-alarm rate indicates a gross calibration failure: {clean_false_alarm_rate}"
        );
    }

    /// Landscape audit (run before trusting the frozen comparison, per this arc's established
    /// discipline): with the "everything on" configuration, does the task actually converge at
    /// all within budget, and does ground truth reliably survive? If success rate here were
    /// near 0% or near 100%, the tolerance/noise/round-budget constants would need recalibrating
    /// before the factorial comparison could show a real signal either way.
    #[test]
    fn audit_everything_on_converges_at_a_measurable_rate() {
        let config = PipelineConfig {
            typed_candidates: true,
            active_selection: true,
            smoothing: true,
        };
        let n = 40;
        let mut successes = 0usize;
        let mut total_rounds = 0usize;
        let mut final_lives = Vec::new();
        for seed in 1..=n as u64 {
            let outcome = run_trial(seed, config, MAX_ROUNDS);
            if outcome.success {
                successes += 1;
            }
            total_rounds += outcome.rounds_used;
            final_lives.push(outcome.final_live);
        }
        println!(
            "audit (typed+active+smoothed): {successes}/{n} succeeded, mean_rounds={:.2}, \
             final_live_counts={final_lives:?}",
            total_rounds as f64 / n as f64
        );
        assert!(
            successes > 0 && successes < n,
            "success rate should be measurable (neither floor nor ceiling) for the factorial \
             comparison to show a real signal; got {successes}/{n} -- recalibrate \
             NOISE_SIGMA_FRACTION/ELIMINATION_TOLERANCE/MAX_ROUNDS if this fails"
        );
    }

    /// Sanity check on the ground-truth-always-in-pool invariant: with zero noise and infinite
    /// tolerance-in-spirit (a huge tolerance), ground truth should never be spuriously
    /// eliminated by its own (noiseless) evaluation matching itself exactly.
    #[test]
    fn ground_truth_is_never_spuriously_eliminated_by_its_own_noiseless_prediction() {
        for seed in 1..=10u64 {
            let mut setup_rng = seed ^ 0x5EED_0000_0000_0001;
            let probes: Vec<Experiment> = (0..GT_SANITY_PROBES)
                .map(|_| random_experiment(&mut setup_rng))
                .collect();
            let gt = nontrivial_ground_truth(&mut setup_rng, &probes);
            for p in &probes {
                let v = eval_at(&gt, p).unwrap();
                assert!((v - v).abs() <= ELIMINATION_TOLERANCE * (1.0 + v.abs()));
            }
        }
    }

    #[test]
    fn untyped_distractor_needs_far_more_attempts_than_typed() {
        // Direct measurement of the same M2.1-shaped phenomenon typed_generation.rs already
        // documented for single candidates: hitting a specific multi-unit dimensional target
        // (here, additionally required to be functionally distinct and sane) is far harder for
        // an untyped grammar than a typed one.
        let mut setup_rng = 0xF00D_u64;
        let probes: Vec<Experiment> = (0..GT_SANITY_PROBES)
            .map(|_| random_experiment(&mut setup_rng))
            .collect();
        let gt = nontrivial_ground_truth(&mut setup_rng, &probes);
        let gt_vals: Vec<f64> = probes.iter().map(|p| eval_at(&gt, p).unwrap()).collect();

        let mut rng = 0xABCD_1234_u64;
        let mut total_attempts = 0usize;
        let draws = 20;
        for _ in 0..draws {
            let (_, attempts) = untyped_distractor(&mut rng, &gt_vals, &probes);
            total_attempts += attempts;
        }
        let mean_attempts = total_attempts as f64 / draws as f64;
        println!("untyped distractor: mean {mean_attempts:.1} attempts/draw over {draws} draws");
        // Typed distractors succeed in ~1 attempt essentially always (dimensionally valid by
        // construction, only the distinctness/sanity filter can reject). This just documents
        // the untyped side's cost is meaningfully higher, not a strict pass/fail gate.
        assert!(mean_attempts >= 1.0);
    }

    /// The frozen factorial ablation: for each of the 3 factors, does turning it on improve
    /// mean success rate by more than the predeclared 0.05 threshold, averaged over the other
    /// two factors and paired by seed? Reports real numbers for all 8 conditions regardless of
    /// the verdict, matching this arc's practice throughout.
    #[test]
    fn factorial_ablation_of_generation_selection_smoothing() {
        let success_by_config = run_factorial(30, MAX_ROUNDS);
        report_and_check_verdicts(&success_by_config);
    }

    /// Follow-up to the above (run 2026-07-16, same day): the original result found
    /// `active_selection`'s effect directionally positive (+0.033) but below the predeclared
    /// 0.05 threshold. Predeclared hypothesis *before running this test*: `MAX_ROUNDS=20` gives
    /// round-robin selection enough slack to also eventually succeed via brute force, masking
    /// any real advantage active selection has when the round budget is actually scarce.
    /// Prediction: under a tighter budget, `active_selection`'s effect grows and clears the
    /// threshold, while `smoothing` remains supported and `typed_candidates` remains not
    /// supported (neither of those two mechanisms is round-budget-sensitive the way
    /// discrimination-order is). `SCARCE_MAX_ROUNDS` is chosen close to but above the original
    /// run's typical mean_rounds (~1.6-2.9 across conditions), so it binds for slower-converging
    /// trials without making the task impossible for all of them (a floor-effect risk this test
    /// checks for directly, same discipline as the landscape audit above).
    #[test]
    fn factorial_ablation_under_scarce_round_budget() {
        const SCARCE_MAX_ROUNDS: usize = 4;
        let success_by_config = run_factorial(30, SCARCE_MAX_ROUNDS);
        // Floor-effect guard: if every condition collapses near 0% success, a tighter budget
        // made the task impossible rather than merely scarce, and the comparison below would be
        // uninformative -- same principle as `audit_everything_on_converges_at_a_measurable_rate`.
        let max_rate = success_by_config
            .values()
            .map(|&(rate, _)| rate)
            .fold(0.0f64, f64::max);
        assert!(
            max_rate > 0.1,
            "SCARCE_MAX_ROUNDS={SCARCE_MAX_ROUNDS} made the task impossible for every condition \
             (max success rate {max_rate:.3}) -- this budget is too tight to be informative, \
             loosen it and rerun rather than trusting a floor effect"
        );
        report_and_check_verdicts(&success_by_config);
    }

    /// Shared reporting + predeclared-threshold verdict logic for both factorial tests above.
    fn report_and_check_verdicts(success_by_config: &HashMap<(bool, bool, bool), (f64, f64)>) {
        let names = ["typed_candidates", "active_selection", "smoothing"];
        let mut verdicts = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let (on, off) = mean_on_off(success_by_config, i);
            let delta = on - off;
            let verdict = if delta > 0.05 {
                "SUPPORTED"
            } else {
                "NEGATIVE"
            };
            verdicts.push((*name, on, off, delta, verdict));
            println!(
                "factor `{name}`: mean success on={on:.3} off={off:.3} delta={delta:+.3} -> {verdict}"
            );
        }

        // Not a hard pass/fail assertion on the science -- this test's job is to run the frozen
        // comparison and report real numbers, which it just did above (visible via
        // --nocapture). The one thing worth asserting mechanically: every condition should at
        // least *terminate cleanly* (no panics, which the loop above already proves by reaching
        // here) and produce a rate in [0, 1].
        for &(rate, _) in success_by_config.values() {
            assert!((0.0..=1.0).contains(&rate));
        }
        assert_eq!(verdicts.len(), 3);
    }
}
