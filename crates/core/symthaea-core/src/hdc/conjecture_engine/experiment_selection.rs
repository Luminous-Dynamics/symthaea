// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FEP-flavored active experiment selection: given several live candidate
//! hypotheses that all still fit the data seen so far, choose which
//! experiment to run next by picking the one that maximally *discriminates*
//! between them.
//!
//! This is the second capability in the agreed longer sequence for the
//! Ramanujan Protocol (HDC → constrained physical reasoning → **this** →
//! CfC → IIT). The idea was scoped back when M2 (flux discovery) was still
//! open: "FEP for active experiment selection (choosing initial conditions
//! that maximally discriminate between surviving candidate hypotheses --
//! needs multiple live candidates to discriminate between, which the search
//! doesn't have yet)". M2's factorized search (`flux_discovery.rs`) and the
//! new `typed_generation` module (`symthaea-physics-bridge`) both now
//! routinely produce multiple structurally-diverse surviving candidates, so
//! this capability is buildable.
//!
//! ## Why not reuse `symthaea-fep::ExpectedFreeEnergyComputer` directly
//!
//! `symthaea-fep` already has a real, working expected-free-energy
//! computation (`free_energy.rs`), and its `epistemic_value` there is
//! exactly the right *concept* (uncertainty reduction), but not directly
//! reusable machinery: it computes the entropy of one continuous
//! `HiddenState` before and after a predicted transition under a single
//! `GenerativeModel` -- built for the cognitive loop's own perception-action
//! domain. What this module needs is different: not "how much does one
//! model's own uncertainty shrink," but "how much do *several independent
//! discrete symbolic hypotheses' predictions disagree* for a given
//! experiment" -- the classic optimal-experimental-design / query-by-
//! committee framing, which happens to be the multi-hypothesis analogue of
//! the same FEP epistemic-value idea. `epistemic_value` below is a new,
//! small, honestly-scoped implementation of that analogue, not a
//! reimplementation of `symthaea-fep`'s machinery.
//!
//! ## Design
//!
//! Deliberately generic over the hypothesis and experiment representations
//! (via a `predict` closure) rather than hardcoded to `Expr` -- this makes
//! it reusable for whatever the next discovery task looks like, not just
//! the closed M2 wave-chain problem. `None` and non-finite predictions are
//! excluded from the variance calculation. If fewer than two finite
//! predictions remain, the typed assessment reports insufficient support and
//! the selector will not rank the experiment as a zero-score tie.
//!
//! The score remains prediction variance (query-by-committee), not a Bayesian
//! posterior, calibrated expected information gain, or a cost-normalized
//! experiment utility. Decision-bearing callers should use
//! `assess_epistemic_value` instead of the compatibility scalar API when they
//! need to distinguish agreement, insufficient support, and unrepresentable
//! scores.

/// A typed result for scoring a candidate experiment against live hypotheses.
///
/// Keeping unsupported and unrepresentable scores distinct prevents callers
/// from mistaking an invalid experiment for a legitimate zero-disagreement one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpistemicAssessment {
    /// Finite, representable population variance over at least two predictions.
    Score {
        value: f64,
        valid_predictions: usize,
    },
    /// Fewer than two hypotheses produced finite predictions for this experiment.
    InsufficientValidPredictions { valid_predictions: usize },
    /// Predictions were finite, but the variance was not representable as f64.
    UnrepresentableScore { valid_predictions: usize },
}

/// Population variance (not sample variance). Scaling before reduction avoids
/// overflowing the mean when several large finite predictions share a sign.
fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let scale = values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return 0.0;
    }

    let normalized: Vec<f64> = values.iter().map(|value| value / scale).collect();
    let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
    let normalized_variance = normalized
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / normalized.len() as f64;

    // Multiplication order matters: normalized_variance * scale * scale can
    // retain a representable result when scale*scale alone would overflow.
    normalized_variance * scale * scale
}

/// Score a candidate experiment while preserving validity information.
///
/// Non-finite predictions are excluded from the comparison. Fewer than two
/// finite predictions return `InsufficientValidPredictions`; an overflowed
/// variance returns `UnrepresentableScore` instead of silently becoming a
/// high score or contaminating the candidate ranking.
pub fn assess_epistemic_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> EpistemicAssessment {
    let predictions: Vec<f64> = hypotheses
        .iter()
        .filter_map(|hypothesis| predict(hypothesis, experiment))
        .filter(|prediction| prediction.is_finite())
        .collect();

    if predictions.len() < 2 {
        return EpistemicAssessment::InsufficientValidPredictions {
            valid_predictions: predictions.len(),
        };
    }

    let valid_predictions = predictions.len();
    let value = variance(&predictions);
    if !value.is_finite() || value < 0.0 {
        EpistemicAssessment::UnrepresentableScore { valid_predictions }
    } else {
        EpistemicAssessment::Score {
            value,
            valid_predictions,
        }
    }
}

/// Compatibility score for one candidate experiment.
///
/// High population variance means candidate hypotheses disagree; zero means
/// either that the predictions agree, fewer than two finite predictions remain,
/// or the score is unrepresentable. Callers that need to distinguish those cases
/// should use [`assess_epistemic_value`] instead.
pub fn epistemic_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> f64 {
    match assess_epistemic_value(experiment, hypotheses, predict) {
        EpistemicAssessment::Score { value, .. } => value,
        EpistemicAssessment::InsufficientValidPredictions { .. }
        | EpistemicAssessment::UnrepresentableScore { .. } => 0.0,
    }
}

/// Select the highest-scoring candidate with finite, representable disagreement
/// across at least two hypotheses. Experiments with no valid comparison or zero
/// disagreement are omitted; returns `None` if no informative candidate remains.
/// Ties keep the first candidate, preserving deterministic input-order semantics.
pub fn select_most_informative_experiment<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<(&'a E, f64)> {
    let mut best: Option<(&'a E, f64)> = None;

    for candidate in candidates {
        let EpistemicAssessment::Score { value, .. } =
            assess_epistemic_value(candidate, hypotheses, predict)
        else {
            continue;
        };

        if value <= 0.0 {
            continue;
        }

        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((candidate, value));
        }
    }

    best
}

/// Number of deterministic quadrature points used to integrate expected information gain.
const INFORMATION_GAIN_QUADRATURE_STEPS: usize = 256;
const INFORMATION_GAIN_Z_LIMIT: f64 = 8.0;

/// Compute log(sum(exp(values))) without overflowing intermediate exponentials.
fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return maximum;
    }
    maximum
        + values
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f64>()
            .ln()
}

/// Limit of the uniform-prior information gain as the observation noise tends to zero.
/// Predictions that are exactly equal remain observationally indistinguishable.
fn noiseless_information_gain(predictions: &[f64]) -> f64 {
    let mut groups: Vec<(f64, usize)> = Vec::new();
    for &prediction in predictions {
        if let Some((_, count)) = groups.iter_mut().find(|(value, _)| *value == prediction) {
            *count += 1;
        } else {
            groups.push((prediction, 1));
        }
    }
    let total = predictions.len() as f64;
    groups
        .iter()
        .map(|(_, count)| {
            let probability = *count as f64 / total;
            -probability * probability.ln()
        })
        .sum()
}

/// Estimate expected information gain, in nats, from an experiment under a common
/// Gaussian observation-noise model and a uniform prior over live hypotheses.
///
/// This is a model-discrimination score, not raw prediction variance:
///
/// I(H; Y | experiment) = E[log(p(Y | H, experiment) / p(Y | experiment))].
///
/// A deterministic midpoint quadrature integrates standard-normal noise over ±8
/// standard deviations. Candidate scoring is comparable only when the supplied
/// observation_std_dev represents the same measurement process for all candidates.
/// To avoid silently rewarding a candidate whose predictions are undefined, every
/// hypothesis must return a finite prediction. None indicates invalid noise,
/// fewer than two hypotheses, or incomplete/non-finite predictions. Identical
/// predictions score zero; as noise decreases, distinguishable prediction groups
/// approach their categorical entropy (at most ln(hypotheses.len()) nats).
pub fn expected_information_gain_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    expected_information_gain_with_steps(
        experiment,
        hypotheses,
        observation_std_dev,
        INFORMATION_GAIN_QUADRATURE_STEPS,
        predict,
    )
}

fn expected_information_gain_with_steps<H, E>(
    experiment: &E,
    hypotheses: &[H],
    observation_std_dev: f64,
    quadrature_steps: usize,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    if quadrature_steps == 0
        || hypotheses.len() < 2
        || !observation_std_dev.is_finite()
        || observation_std_dev <= 0.0
    {
        return None;
    }

    let predictions: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| predict(hypothesis, experiment))
        .collect::<Option<Vec<_>>>()?;
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        return None;
    }

    // Scale means and noise by a shared finite magnitude before subtraction. This
    // avoids overflow for large, opposite-sign predictions while preserving ratios.
    let mean_scale = predictions
        .iter()
        .map(|prediction| prediction.abs())
        .fold(observation_std_dev, f64::max);
    let scaled_means: Vec<f64> = predictions
        .iter()
        .map(|prediction| prediction / mean_scale)
        .collect();
    let scaled_noise = observation_std_dev / mean_scale;
    if scaled_noise == 0.0 {
        return Some(noiseless_information_gain(&predictions));
    }

    let step_width = 2.0 * INFORMATION_GAIN_Z_LIMIT / quadrature_steps as f64;
    let mut quadrature = Vec::with_capacity(quadrature_steps);
    for step_index in 0..quadrature_steps {
        let z = -INFORMATION_GAIN_Z_LIMIT + (step_index as f64 + 0.5) * step_width;
        // The normalizing constant cancels when these density weights are normalized.
        quadrature.push((z, (-0.5 * z * z).exp() * step_width));
    }
    let weight_sum: f64 = quadrature.iter().map(|(_, weight)| weight).sum();
    let prior_log_probability = -(predictions.len() as f64).ln();
    let prior_probability = prior_log_probability.exp();
    let mut log_terms = Vec::with_capacity(predictions.len());
    let mut information_gain = 0.0;

    for true_mean in scaled_means.iter().copied() {
        for &(z, weight) in &quadrature {
            log_terms.clear();
            for other_mean in scaled_means.iter().copied() {
                let separation = (true_mean - other_mean) / scaled_noise;
                let log_likelihood_ratio = if separation == 0.0 {
                    0.0
                } else if !separation.is_finite() || separation.abs() > 1.0e6 {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * separation * separation - z * separation
                };
                log_terms.push(prior_log_probability + log_likelihood_ratio);
            }
            let log_mixture_ratio = log_sum_exp(&log_terms);
            // Pointwise information is -log of the prior-weighted likelihood ratio mixture.
            information_gain -= (weight / weight_sum) * prior_probability * log_mixture_ratio;
        }
    }

    if !information_gain.is_finite() {
        return None;
    }
    Some(information_gain.clamp(0.0, (predictions.len() as f64).ln()))
}

/// Expected information gain in nats under explicit (possibly unnormalized) prior weights.
///
/// Unlike the uniform-prior API, this function does not count duplicated hypotheses as
/// additional prior mass unless the caller assigns them additional weight. Zero-weight
/// hypotheses are ignored. The remaining finite, non-negative weights are normalized
/// internally and must leave at least two positive-mass hypotheses.
///
/// The prior is over the supplied hypothesis instances. Callers should aggregate
/// observationally equivalent hypotheses into equivalence classes when their multiplicity
/// is an artifact of candidate generation, and must not estimate prior weights from held-out
/// outcomes. Measurement noise remains an independently supplied Gaussian standard deviation.
pub fn expected_information_gain_nats_with_prior_weights<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    expected_information_gain_with_prior_weights_and_steps(
        experiment,
        hypotheses,
        prior_weights,
        observation_std_dev,
        INFORMATION_GAIN_QUADRATURE_STEPS,
        predict,
    )
}

fn expected_information_gain_with_prior_weights_and_steps<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    quadrature_steps: usize,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    if quadrature_steps == 0
        || hypotheses.len() < 2
        || hypotheses.len() != prior_weights.len()
        || !observation_std_dev.is_finite()
        || observation_std_dev <= 0.0
        || prior_weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }

    let prior_total: f64 = prior_weights.iter().sum();
    if !prior_total.is_finite() || prior_total <= 0.0 {
        return None;
    }
    let normalized: Vec<f64> = prior_weights.iter().map(|weight| weight / prior_total).collect();
    let mut weighted_predictions = Vec::new();
    for (hypothesis, prior_probability) in hypotheses.iter().zip(normalized.iter().copied()) {
        if prior_probability == 0.0 {
            continue;
        }
        let prediction = predict(hypothesis, experiment)?;
        if !prediction.is_finite() {
            return None;
        }
        weighted_predictions.push((prediction, prior_probability));
    }
    if weighted_predictions.len() < 2 {
        return None;
    }

    let prior_entropy = -weighted_predictions
        .iter()
        .map(|(_, probability)| probability * probability.ln())
        .sum::<f64>();
    if !prior_entropy.is_finite() || prior_entropy <= 0.0 {
        return None;
    }

    // Scale means and noise by one finite magnitude before subtraction to avoid
    // overflow for large, opposite-sign predictions while preserving ratios.
    let mean_scale = weighted_predictions
        .iter()
        .map(|(prediction, _)| prediction.abs())
        .fold(observation_std_dev, f64::max);
    let scaled: Vec<(f64, f64)> = weighted_predictions
        .iter()
        .map(|(prediction, probability)| (prediction / mean_scale, *probability))
        .collect();
    let scaled_noise = observation_std_dev / mean_scale;
    if scaled_noise == 0.0 {
        return Some(noiseless_information_gain_with_prior_weights(
            &weighted_predictions,
        )
        .clamp(0.0, prior_entropy));
    }

    let step_width = 2.0 * INFORMATION_GAIN_Z_LIMIT / quadrature_steps as f64;
    let mut quadrature = Vec::with_capacity(quadrature_steps);
    for step_index in 0..quadrature_steps {
        let z = -INFORMATION_GAIN_Z_LIMIT + (step_index as f64 + 0.5) * step_width;
        quadrature.push((z, (-0.5 * z * z).exp() * step_width));
    }
    let weight_sum: f64 = quadrature.iter().map(|(_, weight)| weight).sum();
    let mut log_terms = Vec::with_capacity(scaled.len());
    let mut information_gain = 0.0;

    for &(true_mean, true_probability) in &scaled {
        for &(z, quadrature_weight) in &quadrature {
            log_terms.clear();
            for &(other_mean, other_probability) in &scaled {
                let separation = (true_mean - other_mean) / scaled_noise;
                let log_likelihood_ratio = if separation == 0.0 {
                    0.0
                } else if !separation.is_finite() || separation.abs() > 1.0e6 {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * separation * separation - z * separation
                };
                log_terms.push(other_probability.ln() + log_likelihood_ratio);
            }
            let log_mixture_ratio = log_sum_exp(&log_terms);
            information_gain -= (quadrature_weight / weight_sum)
                * true_probability
                * log_mixture_ratio;
        }
    }

    if !information_gain.is_finite() {
        return None;
    }
    Some(information_gain.clamp(0.0, prior_entropy))
}

fn noiseless_information_gain_with_prior_weights(weighted_predictions: &[(f64, f64)]) -> f64 {
    let mut groups: Vec<(f64, f64)> = Vec::new();
    for &(prediction, probability) in weighted_predictions {
        if let Some((_, total_probability)) = groups
            .iter_mut()
            .find(|(existing, _)| *existing == prediction)
        {
            *total_probability += probability;
        } else {
            groups.push((prediction, probability));
        }
    }
    -groups
        .iter()
        .map(|(_, probability)| probability * probability.ln())
        .sum::<f64>()
}

/// Select an eligible experiment using prior-weighted, candidate-noise EIG per cost.
///
/// Prior weights are supplied by the caller and normalized by the EIG kernel. The
/// eligibility callback is evaluated before cost, noise or prediction callbacks.
/// Invalid priors, measurements or costs cause the candidate to be skipped; if no
/// eligible candidate retains positive information gain, the function abstains.
pub fn select_experiment_by_information_gain_per_cost_with_prior_weights_and_noise_and_eligibility<
    'a,
    H,
    E,
>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    eligible: impl Fn(&E) -> bool + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        if !eligible(experiment) {
            continue;
        }
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let noise = observation_std_dev(experiment);
        let Some(information_gain) = expected_information_gain_nats_with_prior_weights(
            experiment,
            hypotheses,
            prior_weights,
            noise,
            predict,
        ) else {
            continue;
        };
        let score = information_gain / experiment_cost;
        if score.is_finite()
            && score > 0.0
            && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Select the candidate maximizing expected information gain per unit cost when each
/// candidate has its own observation-noise standard deviation and prevalidated eligibility.
///
/// This is useful when candidate experiments use different instruments, fixtures, test
/// methods, or calibration states. The eligibility predicate is a hard gate: it is evaluated
/// before cost, measurement-noise, or hypothesis-prediction closures. An ineligible experiment
/// is never rescued by a high information score. The caller remains responsible for ensuring
/// eligibility was computed from an independently validated route, safety, calibration, and
/// resource policy; this function does not validate those records.
///
/// Costs must use one comparable unit (for example, validated labor-minutes or currency) and
/// each per-candidate standard deviation must describe the measurement process for that
/// experiment. Incomplete predictions, invalid noise, invalid cost, and zero information gain
/// are skipped. Returns None when no positive-information eligible experiment remains.
pub fn select_experiment_by_information_gain_per_cost_with_noise_and_eligibility<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    eligible: impl Fn(&E) -> bool + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        // Hard constraints precede every information/cost calculation.
        if !eligible(experiment) {
            continue;
        }
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let noise = observation_std_dev(experiment);
        let Some(information_gain) =
            expected_information_gain_nats(experiment, hypotheses, noise, predict)
        else {
            continue;
        };
        let score = information_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Select an eligible experiment by worst-case information gain across a finite
/// set of model-discrepancy scenarios, normalized by acquisition cost.
///
/// This is a conservative finite-set robustness baseline: every candidate must retain
/// strictly positive information gain under every declared scenario. The returned
/// score is the minimum scenario EIG (nats) divided by cost. It does not guarantee
/// robustness outside the supplied scenario set, and it is not the R-IDeA criterion.
///
/// Eligibility is evaluated before cost, noise, or prediction callbacks. A candidate
/// with an incomplete or non-finite prediction in any scenario is skipped rather than
/// being scored on only its favorable scenarios. Empty scenario sets, invalid noise,
/// invalid costs, and zero worst-case information gain fail closed with None.
///
/// The caller must ensure all scenarios share the same hypothesis meaning and prior,
/// and that all candidate predictions represent the same measurand/unit. Measurement
/// uncertainty is supplied independently by the metrology process.
/// Select an eligible experiment by the worst-case prior-weighted EIG per cost
/// across a finite, frozen set of model-discrepancy scenarios.
pub fn select_experiment_by_worst_case_information_gain_per_cost_with_prior_weights<
    'a,
    H,
    E,
    S,
>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    model_scenarios: &[S],
    eligible: impl Fn(&E) -> bool + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E, &S) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if model_scenarios.is_empty() {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        if !eligible(experiment) {
            continue;
        }
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let noise = observation_std_dev(experiment);
        if !noise.is_finite() || noise <= 0.0 {
            continue;
        }

        let mut worst_case_information_gain = f64::INFINITY;
        let mut complete = true;
        for scenario in model_scenarios {
            let Some(information_gain) = expected_information_gain_nats_with_prior_weights(
                experiment,
                hypotheses,
                prior_weights,
                noise,
                |hypothesis, candidate| predict(hypothesis, candidate, scenario),
            ) else {
                complete = false;
                break;
            };
            if !information_gain.is_finite() || information_gain <= 0.0 {
                complete = false;
                break;
            }
            worst_case_information_gain = worst_case_information_gain.min(information_gain);
        }
        if !complete {
            continue;
        }

        let score = worst_case_information_gain / experiment_cost;
        if score.is_finite()
            && score > 0.0
            && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

pub fn select_experiment_by_worst_case_information_gain_per_cost<'a, H, E, S>(
    candidates: &'a [E],
    hypotheses: &[H],
    model_scenarios: &[S],
    eligible: impl Fn(&E) -> bool + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E, &S) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if model_scenarios.is_empty() {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        // Hard constraints precede all information and cost calculations.
        if !eligible(experiment) {
            continue;
        }
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let noise = observation_std_dev(experiment);
        if !noise.is_finite() || noise <= 0.0 {
            continue;
        }

        let mut worst_case_information_gain = f64::INFINITY;
        let mut complete = true;
        for scenario in model_scenarios {
            let information_gain = expected_information_gain_nats(
                experiment,
                hypotheses,
                noise,
                |hypothesis, candidate| predict(hypothesis, candidate, scenario),
            );
            let Some(information_gain) = information_gain else {
                complete = false;
                break;
            };
            if !information_gain.is_finite() || information_gain <= 0.0 {
                complete = false;
                break;
            }
            worst_case_information_gain = worst_case_information_gain.min(information_gain);
        }
        if !complete {
            continue;
        }

        let score = worst_case_information_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Select an eligible experiment using a finite-scenario, representative and
/// de-amplifying acquisition proxy.
///
/// The base information term is the minimum prior-weighted EIG across every supplied
/// model-discrepancy scenario. It is then multiplied by two independently supplied,
/// unitless evidence scores:
///
/// - `representativeness` in [0, 1], measuring how well the candidate represents the
///   frozen deployment/production population;
/// - `amplification_risk` in [0, 1], estimating the chance/severity that this candidate
///   amplifies predictive or inferential error under the declared misspecification audit.
///
/// The final score is:
/// `min_s(EIG_s) * representativeness * (1 - amplification_risk) / cost`.
///
/// This is an inspectable engineering proxy inspired by the representative/informative/de-
/// amplifying principle; it is NOT an implementation or reproduction of the published
/// R-IDeA acquisition function and is NOT a safety certificate. The supplied scores must
/// come from frozen, independent validation evidence. They must not be generated by the
/// same candidate-ranking model or estimated from held-out outcomes. Unknown, non-finite
/// or out-of-range scores cause the candidate to be skipped.
///
/// Eligibility is a hard filter and runs before representativeness, risk, cost, noise,
/// or prediction callbacks. Every model scenario must provide complete predictions and
/// strictly positive EIG; one unsupported scenario rejects the candidate. The caller must
/// ensure a common measurand/unit and common hypothesis semantics across scenarios.
///
/// A missing/empty scenario set or no positive robust candidate returns None (abstention).
pub fn select_experiment_by_representative_deamplifying_worst_case_eig_per_cost<
    'a,
    H,
    E,
    S,
>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    model_scenarios: &[S],
    eligible: impl Fn(&E) -> bool + Copy,
    representativeness: impl Fn(&E) -> f64 + Copy,
    amplification_risk: impl Fn(&E) -> f64 + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E, &S) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if model_scenarios.is_empty() {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        // Hard safety/route/calibration/resource gate before reading any ranking features.
        if !eligible(experiment) {
            continue;
        }

        let representative = representativeness(experiment);
        let amplification = amplification_risk(experiment);
        if !representative.is_finite()
            || !(0.0..=1.0).contains(&representative)
            || !amplification.is_finite()
            || !(0.0..=1.0).contains(&amplification)
            || representative <= 0.0
            || amplification >= 1.0
        {
            continue;
        }

        let experiment_cost = cost(experiment);
        let noise = observation_std_dev(experiment);
        if !experiment_cost.is_finite()
            || experiment_cost <= 0.0
            || !noise.is_finite()
            || noise <= 0.0
        {
            continue;
        }

        let mut worst_case_information_gain = f64::INFINITY;
        let mut complete = true;
        for scenario in model_scenarios {
            let Some(information_gain) = expected_information_gain_nats_with_prior_weights(
                experiment,
                hypotheses,
                prior_weights,
                noise,
                |hypothesis, candidate| predict(hypothesis, candidate, scenario),
            ) else {
                complete = false;
                break;
            };
            if !information_gain.is_finite() || information_gain <= 0.0 {
                complete = false;
                break;
            }
            worst_case_information_gain = worst_case_information_gain.min(information_gain);
        }
        if !complete {
            continue;
        }

        let score =
            worst_case_information_gain * representative * (1.0 - amplification) / experiment_cost;
        if score.is_finite()
            && score > 0.0
            && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Select the candidate maximizing expected information gain per unit cost.
///
/// Candidates with non-positive/non-finite costs, incomplete predictions, or zero
/// information gain are skipped. The returned score is nats per cost unit. Strict
/// comparison preserves the first informative candidate when scores tie, making
/// selection deterministic. Returns `None` if no candidate is informative.
pub fn select_experiment_by_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
        candidates,
        hypotheses,
        |_| true,
        |_| observation_std_dev,
        predict,
        cost,
    )
}


/// Expected value of sample information (EVSI) for a finite engineering decision.
///
/// Utilities are supplied by the approved decision policy; larger values are better.
/// EVSI is the expected improvement in the best posterior expected utility after one
/// measurement, compared with the best action before measuring. It is therefore
/// decision-aware: information about distinctions that cannot change the preferred
/// action has zero (or near-zero) value, even when that information gain is large.
///
/// The prior may be unnormalized but must have finite non-negative entries and at
/// least two positive-mass hypotheses. Zero-weight hypotheses are ignored. Every
/// positive-mass hypothesis needs a finite prediction and finite utility for each
/// action. Measurement noise is an independent Gaussian standard deviation in the
/// same unit as the predictions. A deterministic midpoint quadrature integrates
/// over standard-normal outcomes across ±8 standard deviations.
///
/// The result is in the utility unit declared by the caller's action×hypothesis
/// matrix, not nats. Returns None for invalid priors, incomplete predictions,
/// invalid noise, empty action sets, or non-finite utilities. This ranking primitive
/// does not validate or authorize the utility policy.
pub fn expected_value_of_sample_information<H, E, A>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    actions: &[A],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64>,
    utility: impl Fn(&A, &H) -> f64,
) -> Option<f64> {
    expected_value_of_sample_information_with_steps(
        experiment,
        hypotheses,
        prior_weights,
        actions,
        observation_std_dev,
        INFORMATION_GAIN_QUADRATURE_STEPS,
        predict,
        utility,
    )
}

fn expected_value_of_sample_information_with_steps<H, E, A>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    actions: &[A],
    observation_std_dev: f64,
    quadrature_steps: usize,
    predict: impl Fn(&H, &E) -> Option<f64>,
    utility: impl Fn(&A, &H) -> f64,
) -> Option<f64> {
    if quadrature_steps == 0
        || hypotheses.len() < 2
        || hypotheses.len() != prior_weights.len()
        || actions.is_empty()
        || !observation_std_dev.is_finite()
        || observation_std_dev <= 0.0
        || prior_weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }

    // Normalize after max-scaling so finite but very large prior weights cannot
    // overflow their sum.
    let max_weight = prior_weights.iter().copied().fold(0.0_f64, f64::max);
    if max_weight <= 0.0 {
        return None;
    }
    let scaled_weights: Vec<f64> = prior_weights.iter().map(|weight| weight / max_weight).collect();
    let weight_sum: f64 = scaled_weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return None;
    }

    let mut active: Vec<(usize, f64, f64)> = Vec::new();
    for (index, weight) in scaled_weights.iter().copied().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let probability = weight / weight_sum;
        let prediction = predict(&hypotheses[index], experiment)?;
        if !prediction.is_finite() {
            return None;
        }
        active.push((index, probability, prediction));
    }
    if active.len() < 2 {
        return None;
    }

    let utilities: Vec<Vec<f64>> = actions
        .iter()
        .map(|action| {
            active
                .iter()
                .map(|(index, _, _)| utility(action, &hypotheses[*index]))
                .collect()
        })
        .collect();
    if utilities.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }

    let mut baseline = f64::NEG_INFINITY;
    for action_utilities in &utilities {
        let expected = active
            .iter()
            .zip(action_utilities.iter())
            .map(|((_, probability, _), value)| probability * value)
            .sum::<f64>();
        if !expected.is_finite() {
            return None;
        }
        baseline = baseline.max(expected);
    }

    // Scale means and noise together to prevent overflow in mean differences.
    let scale = active
        .iter()
        .map(|(_, _, prediction)| prediction.abs())
        .fold(observation_std_dev, f64::max);
    let means: Vec<f64> = active.iter().map(|(_, _, mean)| mean / scale).collect();
    let noise = observation_std_dev / scale;

    if noise == 0.0 {
        // In the noiseless limit, equal predicted values remain indistinguishable.
        let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for (index, mean) in means.iter().copied().enumerate() {
            if let Some((_, members)) = groups.iter_mut().find(|(value, _)| *value == mean) {
                members.push(index);
            } else {
                groups.push((mean, vec![index]));
            }
        }
        let mut expected_after = 0.0;
        for (_, members) in groups {
            let group_mass: f64 = members.iter().map(|i| active[*i].1).sum();
            let mut best_action = f64::NEG_INFINITY;
            for action_utilities in &utilities {
                let expected = members
                    .iter()
                    .map(|i| active[*i].1 * action_utilities[*i])
                    .sum::<f64>()
                    / group_mass;
                best_action = best_action.max(expected);
            }
            expected_after += group_mass * best_action;
        }
        let value = expected_after - baseline;
        return value.is_finite().then_some(value.max(0.0));
    }

    let step_width = 2.0 * INFORMATION_GAIN_Z_LIMIT / quadrature_steps as f64;
    let quadrature: Vec<(f64, f64)> = (0..quadrature_steps)
        .map(|step| {
            let z = -INFORMATION_GAIN_Z_LIMIT + (step as f64 + 0.5) * step_width;
            (z, (-0.5 * z * z).exp() * step_width)
        })
        .collect();
    let quadrature_weight_sum: f64 = quadrature.iter().map(|(_, weight)| weight).sum();
    if !quadrature_weight_sum.is_finite() || quadrature_weight_sum <= 0.0 {
        return None;
    }

    let mut expected_after = 0.0;
    let mut log_joint = Vec::with_capacity(active.len());
    for (true_index, (_, true_probability, true_mean)) in active.iter().enumerate() {
        let scaled_true_mean = means[true_index];
        for (z, raw_weight) in &quadrature {
            log_joint.clear();
            for (other_mean, (_, probability, _)) in means.iter().zip(active.iter()) {
                let separation = (scaled_true_mean - other_mean) / noise;
                let standardized = separation + z;
                let log_likelihood = if !standardized.is_finite() || standardized.abs() > 1.0e6 {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * standardized * standardized
                };
                log_joint.push(probability.ln() + log_likelihood);
            }
            let log_normalizer = log_sum_exp(&log_joint);
            if !log_normalizer.is_finite() {
                return None;
            }
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|log_value| (*log_value - log_normalizer).exp())
                .collect();

            let mut best_posterior_utility = f64::NEG_INFINITY;
            for action_utilities in &utilities {
                let value = posterior
                    .iter()
                    .zip(action_utilities.iter())
                    .map(|(probability, utility)| probability * utility)
                    .sum::<f64>();
                if !value.is_finite() {
                    return None;
                }
                best_posterior_utility = best_posterior_utility.max(value);
            }
            expected_after += true_probability
                * (*raw_weight / quadrature_weight_sum)
                * best_posterior_utility;
        }
        let _ = true_mean;
    }

    let evsi = expected_after - baseline;
    if !evsi.is_finite() {
        return None;
    }
    let utility_scale = utilities
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    // Finite quadrature can introduce tiny negative roundoff despite non-negative
    // theoretical EVSI; larger negative values indicate a numerical/model defect.
    if evsi < -1.0e-7 * utility_scale {
        return None;
    }
    Some(evsi.max(0.0))
}

/// Select an experiment by worst-case expected value of sample information (EVSI).
///
/// The robust score is:
/// min_s EVSI(experiment | discrepancy scenario s)
///   × representativeness × (1 - amplification_risk) / acquisition_cost.
///
/// Utility values are provided by an externally approved action×hypothesis policy.
/// Eligibility is a hard gate and is checked before representativeness, risk, cost,
/// noise, prediction or utility callbacks. The scenario set is finite and caller-owned;
/// this does not establish robustness outside the frozen scenario family. Returns None
/// when no admissible candidate has strictly positive worst-case decision value.
pub fn select_experiment_by_representative_deamplifying_worst_case_evsi_per_cost<
    'a,
    H,
    E,
    A,
    S,
>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    actions: &[A],
    model_scenarios: &[S],
    eligible: impl Fn(&E) -> bool + Copy,
    representativeness: impl Fn(&E) -> f64 + Copy,
    amplification_risk: impl Fn(&E) -> f64 + Copy,
    observation_std_dev: impl Fn(&E) -> f64 + Copy,
    predict: impl Fn(&H, &E, &S) -> Option<f64> + Copy,
    utility: impl Fn(&A, &H) -> f64 + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if model_scenarios.is_empty() {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        if !eligible(experiment) {
            continue;
        }
        let representative = representativeness(experiment);
        let amplification = amplification_risk(experiment);
        if !representative.is_finite()
            || !(0.0..=1.0).contains(&representative)
            || !amplification.is_finite()
            || !(0.0..=1.0).contains(&amplification)
            || representative <= 0.0
            || amplification >= 1.0
        {
            continue;
        }
        let experiment_cost = cost(experiment);
        let noise = observation_std_dev(experiment);
        if !experiment_cost.is_finite()
            || experiment_cost <= 0.0
            || !noise.is_finite()
            || noise <= 0.0
        {
            continue;
        }

        let mut worst_case_evsi = f64::INFINITY;
        let mut complete = true;
        for scenario in model_scenarios {
            let Some(value) = expected_value_of_sample_information(
                experiment,
                hypotheses,
                prior_weights,
                actions,
                noise,
                |hypothesis, candidate| predict(hypothesis, candidate, scenario),
                utility,
            ) else {
                complete = false;
                break;
            };
            if !value.is_finite() || value <= 0.0 {
                complete = false;
                break;
            }
            worst_case_evsi = worst_case_evsi.min(value);
        }
        if !complete {
            continue;
        }

        let score = worst_case_evsi * representative * (1.0 - amplification) / experiment_cost;
        if score.is_finite()
            && score > 0.0
            && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::conjecture_engine::Expr;

    #[test]
    fn variance_of_identical_values_is_zero() {
        assert_eq!(variance(&[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn variance_matches_hand_computation() {
        // [1, 2, 3]: mean=2, population variance = ((1)^2+(0)^2+(1)^2)/3 = 2/3
        let v = variance(&[1.0, 2.0, 3.0]);
        assert!((v - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn epistemic_value_is_zero_when_hypotheses_agree() {
        // Four synthetic "laws" that all happen to predict 0 at x=0.
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let v = epistemic_value(&0.0, &hypotheses, predict);
        assert!(v < 1e-9, "expected near-zero disagreement at x=0, got {v}");
    }

    #[test]
    fn epistemic_value_is_high_when_hypotheses_diverge() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let at_zero = epistemic_value(&0.0, &hypotheses, predict);
        let at_three = epistemic_value(&3.0, &hypotheses, predict);
        assert!(
            at_three > at_zero,
            "x=3 (predictions 3, 9, 6, sin(3)≈0.14) should disagree far more than x=0 \
             (all predict 0), got at_zero={at_zero}, at_three={at_three}"
        );
    }

    #[test]
    fn selector_avoids_the_degenerate_all_agree_point() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        // Candidate pool deliberately includes the degenerate x=0 point
        // alongside genuinely discriminating ones.
        let candidates = [0.0, -3.0, -1.0, 1.0, 3.0];
        let (chosen, value) = select_most_informative_experiment(&candidates, &hypotheses, predict)
            .expect("non-empty candidate pool");
        assert_ne!(
            *chosen, 0.0,
            "selector should not pick the point where every hypothesis agrees"
        );
        assert!(value > 0.0);
    }

    #[test]
    fn selector_returns_none_on_empty_candidate_pool() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let candidates: [f64; 0] = [];
        assert!(select_most_informative_experiment(&candidates, &hypotheses, predict).is_none());
    }

    #[test]
    fn none_predictions_are_excluded_not_treated_as_zero_disagreement() {
        // A hypothesis that can't produce a prediction for some experiments
        // (e.g. undefined there) should simply not count toward that
        // experiment's variance -- not silently contribute a 0.0 that could
        // suppress a real signal from the hypotheses that DO predict there.
        let hypotheses = vec!["always_valid", "invalid_at_zero"];
        let predict = |h: &&str, x: &f64| match *h {
            "always_valid" => Some(*x),
            "invalid_at_zero" if *x == 0.0 => None,
            "invalid_at_zero" => Some(*x * 10.0),
            _ => None,
        };
        // At x=0.0, only one hypothesis contributes a prediction -> variance 0.0
        // (not because they "agree", but because there's nothing to compare).
        let v = epistemic_value(&0.0, &hypotheses, predict);
        assert_eq!(v, 0.0);
        // At x=1.0, both contribute (1.0 vs 10.0) -> real disagreement.
        let v2 = epistemic_value(&1.0, &hypotheses, predict);
        assert!(v2 > 0.0);
    }

    /// Integration check: the same mechanism applied to actual
    /// `conjecture_engine::Expr` candidates (the Ramanujan Protocol's real
    /// hypothesis representation), not just closures -- confirms this is
    /// directly usable against the discovery engine's own candidate type,
    /// without depending on any specific domain (wave-chain or otherwise).
    #[test]
    fn information_gain_is_zero_when_all_hypotheses_predict_the_same_value() {
        let hypotheses = [0_i32, 1, 2, 3];
        let score = expected_information_gain_nats(&(), &hypotheses, 1.0, |_, _| Some(42.0))
            .expect("valid noise and complete predictions");
        assert!(
            score.abs() < 1.0e-12,
            "identical predictions should teach nothing: {score}"
        );
    }

    #[test]
    fn information_gain_prefers_balanced_discrimination_over_a_single_outlier() {
        // Raw variance heavily favors [0, 0, 0, 100], although one observation
        // can only identify one hypothesis against three indistinguishable ones.
        let outlier = [0.0, 0.0, 0.0, 100.0];
        let balanced = [0.0, 0.0, 10.0, 10.0];
        assert!(variance(&outlier) > variance(&balanced));
        let hypotheses = [0_usize, 1, 2, 3];
        let outlier_gain =
            expected_information_gain_nats(&0_usize, &hypotheses, 1.0, |hypothesis, _| {
                Some(outlier[*hypothesis])
            })
            .expect("complete finite predictions");
        let balanced_gain =
            expected_information_gain_nats(&0_usize, &hypotheses, 1.0, |hypothesis, _| {
                Some(balanced[*hypothesis])
            })
            .expect("complete finite predictions");
        assert!(
            balanced_gain > outlier_gain + 0.05,
            "balanced split should reduce hypothesis uncertainty more: outlier={outlier_gain}, balanced={balanced_gain}"
        );
        assert!(balanced_gain <= 4.0_f64.ln());
    }

    #[test]
    fn information_gain_decreases_as_measurement_noise_increases() {
        let hypotheses = [0.0_f64, 2.0];
        let predict = |hypothesis: &f64, _: &()| Some(*hypothesis);
        let low_noise = expected_information_gain_nats(&(), &hypotheses, 0.1, predict).unwrap();
        let medium_noise = expected_information_gain_nats(&(), &hypotheses, 1.0, predict).unwrap();
        let high_noise = expected_information_gain_nats(&(), &hypotheses, 10.0, predict).unwrap();
        assert!(low_noise > medium_noise);
        assert!(medium_noise > high_noise);
        for score in [low_noise, medium_noise, high_noise] {
            assert!(score.is_finite() && (0.0..=2.0_f64.ln()).contains(&score));
        }
    }

    #[test]
    fn information_gain_fails_closed_on_invalid_noise_or_incomplete_predictions() {
        let hypotheses = [0_i32, 1, 2];
        assert!(
            expected_information_gain_nats(&(), &hypotheses, 0.0, |h, _| Some(*h as f64)).is_none()
        );
        assert!(
            expected_information_gain_nats(&(), &hypotheses, f64::NAN, |h, _| Some(*h as f64))
                .is_none()
        );
        assert!(
            expected_information_gain_nats(&(), &hypotheses, 1.0, |h, _| {
                (*h != 1).then_some(*h as f64)
            })
            .is_none()
        );
        assert!(
            expected_information_gain_nats(&(), &[1_i32], 1.0, |h, _| Some(*h as f64)).is_none()
        );
    }

    #[derive(Clone, Copy)]
    struct RobustCandidate {
        id: &'static str,
        representative: f64,
        amplification_risk: f64,
        eligible: bool,
    }

    #[test]
    fn robust_representative_selector_deamplifies_unrepresentative_high_information_candidate() {
        let candidates = [
            RobustCandidate {
                id: "high-eig-unrepresentative",
                representative: 0.05,
                amplification_risk: 0.90,
                eligible: true,
            },
            RobustCandidate {
                id: "representative-low-risk",
                representative: 0.95,
                amplification_risk: 0.05,
                eligible: true,
            },
        ];
        let hypotheses = [0_usize, 1];
        let scenarios = [0_u8, 1_u8];
        let predict = |hypothesis: &usize, candidate: &RobustCandidate, _scenario: &u8| {
            let signal = if candidate.id == "high-eig-unrepresentative" { 10.0 } else { 2.0 };
            Some(*hypothesis as f64 * signal)
        };
        let (chosen, score) =
            select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
                &candidates,
                &hypotheses,
                &[0.5, 0.5],
                &scenarios,
                |candidate| candidate.eligible,
                |candidate| candidate.representative,
                |candidate| candidate.amplification_risk,
                |_| 1.0,
                predict,
                |_| 1.0,
            )
            .expect("at least one robust representative candidate");
        assert_eq!(chosen.id, "representative-low-risk");
        assert!(score.is_finite() && score > 0.0);
    }

    #[test]
    fn robust_representative_selector_requires_every_declared_discrepancy_scenario() {
        let candidates = [
            RobustCandidate {
                id: "scenario-brittle",
                representative: 1.0,
                amplification_risk: 0.0,
                eligible: true,
            },
            RobustCandidate {
                id: "scenario-stable",
                representative: 0.8,
                amplification_risk: 0.1,
                eligible: true,
            },
        ];
        let hypotheses = [0_usize, 1];
        let scenarios = [0_u8, 1_u8];
        let predict = |hypothesis: &usize, candidate: &RobustCandidate, scenario: &u8| {
            match (candidate.id, *scenario, *hypothesis) {
                ("scenario-brittle", 0, 0) => Some(0.0),
                ("scenario-brittle", 0, 1) => Some(8.0),
                ("scenario-brittle", 1, _) => Some(0.0),
                ("scenario-stable", _, 0) => Some(0.0),
                ("scenario-stable", _, 1) => Some(2.0),
                _ => None,
            }
        };
        let selected = select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
            &candidates,
            &hypotheses,
            &[0.5, 0.5],
            &scenarios,
            |candidate| candidate.eligible,
            |candidate| candidate.representative,
            |candidate| candidate.amplification_risk,
            |_| 1.0,
            predict,
            |_| 1.0,
        )
        .expect("stable candidate retains positive EIG in both scenarios");
        assert_eq!(selected.0.id, "scenario-stable");
    }

    #[test]
    fn robust_representative_selector_skips_unknown_scores_and_abstains() {
        let candidates = [
            RobustCandidate {
                id: "unknown-representativeness",
                representative: f64::NAN,
                amplification_risk: 0.0,
                eligible: true,
            },
            RobustCandidate {
                id: "unknown-amplification",
                representative: 0.9,
                amplification_risk: f64::NAN,
                eligible: true,
            },
            RobustCandidate {
                id: "out-of-range-risk",
                representative: 0.9,
                amplification_risk: 1.1,
                eligible: true,
            },
        ];
        let hypotheses = [0_usize, 1];
        let scenarios = [0_u8];
        let predict = |hypothesis: &usize, _: &RobustCandidate, _: &u8| {
            Some(*hypothesis as f64)
        };
        assert!(
            select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
                &candidates,
                &hypotheses,
                &[0.5, 0.5],
                &scenarios,
                |candidate| candidate.eligible,
                |candidate| candidate.representative,
                |candidate| candidate.amplification_risk,
                |_| 1.0,
                predict,
                |_| 1.0,
            )
            .is_none(),
            "unknown or invalid robustness metadata must cause abstention"
        );
    }

    #[test]
    fn robust_representative_selector_applies_eligibility_before_other_callbacks() {
        let candidates = [RobustCandidate {
            id: "blocked",
            representative: 1.0,
            amplification_risk: 0.0,
            eligible: false,
        }];
        let hypotheses = [0_usize, 1];
        let scenarios = [0_u8];
        let predict = |hypothesis: &usize, _: &RobustCandidate, _: &u8| {
            Some(*hypothesis as f64)
        };
        assert!(
            select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
                &candidates,
                &hypotheses,
                &[0.5, 0.5],
                &scenarios,
                |candidate| candidate.eligible,
                |_| panic!("ineligible candidate representativeness must not be queried"),
                |_| panic!("ineligible candidate risk must not be queried"),
                |_| panic!("ineligible candidate noise must not be queried"),
                predict,
                |_| panic!("ineligible candidate cost must not be queried"),
            )
            .is_none()
        );
    }

    struct ManufacturingAdapterCandidate {
        id: String,
        representative: f64,
        amplification_risk: f64,
        noise: f64,
        cost: f64,
        cost_unit: String,
        eligible: bool,
    }

    #[test]
    fn consumes_symtropy_manufacturing_adapter_and_selects_representative_low_risk_candidate() {
        const ADAPTER: &str =
            include_str!("fixtures/manufacturing-robustness-adapter-synthetic.csv");
        let mut lines = ADAPTER.lines();
        assert_eq!(
            lines.next(),
            Some("candidate_id,representativeness_score,amplification_risk,observation_std_uncertainty,acquisition_cost,cost_unit,eligible,eligibility_state,eligibility_receipt_sha256")
        );
        let candidates: Vec<ManufacturingAdapterCandidate> = lines
            .map(|line| {
                let fields: Vec<&str> = line.split(',').collect();
                assert_eq!(fields.len(), 9);
                assert_eq!(fields[7], "ELIGIBLE");
                assert_eq!(fields[8].len(), 64);
                let candidate = ManufacturingAdapterCandidate {
                    id: fields[0].to_string(),
                    representative: fields[1].parse().unwrap(),
                    amplification_risk: fields[2].parse().unwrap(),
                    noise: fields[3].parse().unwrap(),
                    cost: fields[4].parse().unwrap(),
                    cost_unit: fields[5].to_string(),
                    eligible: fields[6].parse().unwrap(),
                };
                assert!((0.0..=1.0).contains(&candidate.representative));
                assert!((0.0..=1.0).contains(&candidate.amplification_risk));
                assert!(candidate.noise.is_finite() && candidate.noise > 0.0);
                assert!(candidate.cost.is_finite() && candidate.cost > 0.0);
                candidate
            })
            .collect();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().all(|candidate| candidate.cost_unit == "USD_2026"));

        let hypotheses = [0_usize, 1];
        let discrepancy_scenarios = [0_u8, 1_u8];
        let predict = |hypothesis: &usize,
                       candidate: &ManufacturingAdapterCandidate,
                       scenario: &u8| {
            let signal = if candidate.id == "TX-CAND-OOD-002" { 10.0 } else { 2.0 };
            Some(*hypothesis as f64 * signal * (1.0 + f64::from(*scenario) * 0.1))
        };
        let (chosen, score) =
            select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
                &candidates,
                &hypotheses,
                &[0.5, 0.5],
                &discrepancy_scenarios,
                |candidate| candidate.eligible,
                |candidate| candidate.representative,
                |candidate| candidate.amplification_risk,
                |candidate| candidate.noise,
                predict,
                |candidate| candidate.cost,
            )
            .expect("the manufacturing adapter contains an eligible positive-information candidate");
        assert_eq!(chosen.id, "TX-CAND-REP-001");
        assert!(score.is_finite() && score > 0.0);
    }

    #[test]
    fn information_gain_selector_maximizes_information_per_cost_not_raw_information() {
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [1_u8, 2, 3];
        let predict = |hypothesis: &usize, experiment: &u8| match (*experiment, *hypothesis) {
            (1, 0 | 1) => Some(0.0),
            (1, 2 | 3) => Some(1.0),
            (2, 0 | 1) => Some(0.0),
            (2, 2 | 3) => Some(10.0),
            (3, 0) => Some(-100.0),
            (3, 1) => Some(-50.0),
            (3, 2) => Some(50.0),
            (3, 3) => Some(100.0),
            _ => None,
        };
        let low_cost_information =
            expected_information_gain_nats(&1, &hypotheses, 1.0, predict).unwrap();
        let high_cost_information =
            expected_information_gain_nats(&2, &hypotheses, 1.0, predict).unwrap();
        assert!(high_cost_information > low_cost_information);

        let (chosen, score) = select_experiment_by_information_gain_per_cost(
            &candidates,
            &hypotheses,
            1.0,
            predict,
            |experiment| match experiment {
                1 => 1.0,
                2 => 100.0,
                3 => 0.0, // Non-physical costs are rejected, even with high information.
                _ => f64::INFINITY,
            },
        )
        .expect("at least one valid candidate");
        assert_eq!(*chosen, 1);
        assert!(score.is_finite() && score > 0.0);
    }

    #[test]
    fn heteroscedastic_selector_prefers_a_cleaner_measurement_over_a_noisier_high_signal_one() {
        let hypotheses = [0_usize, 1];
        let candidates = ["high_signal_bad_meter", "low_signal_good_meter"];
        let predict = |hypothesis: &usize, candidate: &&str| match (*candidate, *hypothesis) {
            ("high_signal_bad_meter", 0) => Some(0.0),
            ("high_signal_bad_meter", 1) => Some(10.0),
            ("low_signal_good_meter", 0) => Some(0.0),
            ("low_signal_good_meter", 1) => Some(1.0),
            _ => None,
        };
        let noise = |candidate: &&str| match *candidate {
            "high_signal_bad_meter" => 100.0,
            "low_signal_good_meter" => 0.1,
            _ => f64::NAN,
        };
        let (chosen, score) =
            select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
                &candidates,
                &hypotheses,
                |_| true,
                noise,
                predict,
                |_| 1.0,
            )
            .expect("the calibrated candidate has positive information gain");
        assert_eq!(*chosen, "low_signal_good_meter");
        assert!(score.is_finite() && score > 0.0);

        // Under a shared low-noise assumption, the much larger nominal signal is
        // selected first. Candidate-specific metrology uncertainty reverses that rank.
        let (common_noise_choice, _) = select_experiment_by_information_gain_per_cost(
            &candidates,
            &hypotheses,
            0.1,
            predict,
            |_| 1.0,
        )
        .expect("both candidates are informative under the common-noise assumption");
        assert_eq!(*common_noise_choice, "high_signal_bad_meter");
    }

    #[test]
    fn hard_eligibility_gate_runs_before_cost_noise_or_prediction() {
        let candidates = ["blocked", "eligible"];
        let hypotheses = [0_usize, 1];
        let predict = |hypothesis: &usize, candidate: &&str| {
            assert_ne!(
                *candidate, "blocked",
                "blocked candidates must not be scored"
            );
            Some(*hypothesis as f64)
        };
        let cost = |candidate: &&str| {
            assert_ne!(
                *candidate, "blocked",
                "blocked candidate cost must not be queried"
            );
            1.0
        };
        let noise = |candidate: &&str| {
            assert_ne!(
                *candidate, "blocked",
                "blocked candidate noise must not be queried"
            );
            1.0
        };
        let selected = select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
            &candidates,
            &hypotheses,
            |candidate| *candidate == "eligible",
            noise,
            predict,
            cost,
        )
        .expect("one eligible informative candidate");
        assert_eq!(*selected.0, "eligible");

        assert!(
            select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
                &candidates,
                &hypotheses,
                |_| false,
                noise,
                predict,
                cost,
            )
            .is_none()
        );
    }

    #[test]
    fn heteroscedastic_selector_skips_invalid_noise_and_invalid_cost() {
        let candidates = [0_usize, 1, 2];
        let hypotheses = [0_usize, 1];
        let selected = select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
            &candidates,
            &hypotheses,
            |_| true,
            |candidate| match candidate {
                0 => f64::NAN,
                1 => 1.0,
                _ => 1.0,
            },
            |hypothesis, candidate| Some((*hypothesis * (*candidate + 1)) as f64),
            |candidate| if *candidate == 2 { f64::INFINITY } else { 1.0 },
        )
        .expect("candidate 1 is admissible and informative");
        assert_eq!(*selected.0, 1);

        assert!(
            select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
                &candidates,
                &hypotheses,
                |_| true,
                |_| f64::NAN,
                |hypothesis, candidate| Some((*hypothesis * (*candidate + 1)) as f64),
                |_| 1.0,
            )
            .is_none()
        );
    }

    #[test]
    fn weighted_eig_is_invariant_to_duplicate_hypotheses_with_split_prior_mass() {
        let hypotheses = [0_usize, 1];
        let two_hypothesis_gain = expected_information_gain_nats_with_prior_weights(
            &(),
            &hypotheses,
            &[0.5, 0.5],
            0.25,
            |hypothesis, _| Some(*hypothesis as f64),
        )
        .unwrap();

        let duplicated_hypotheses = [0_usize, 1, 2];
        let duplicated_gain = expected_information_gain_nats_with_prior_weights(
            &(),
            &duplicated_hypotheses,
            &[0.25, 0.25, 0.5],
            0.25,
            |hypothesis, _| Some(if *hypothesis == 2 { 1.0 } else { 0.0 }),
        )
        .unwrap();
        assert!((two_hypothesis_gain - duplicated_gain).abs() < 1.0e-10);
    }

    #[test]
    fn weighted_eig_respects_prior_entropy_and_asymmetric_hypothesis_mass() {
        let hypotheses = [0.0_f64, 2.0];
        let uniform = expected_information_gain_nats_with_prior_weights(
            &(),
            &hypotheses,
            &[1.0, 1.0],
            0.05,
            |hypothesis, _| Some(*hypothesis),
        )
        .unwrap();
        let skewed = expected_information_gain_nats_with_prior_weights(
            &(),
            &hypotheses,
            &[9.0, 1.0],
            0.05,
            |hypothesis, _| Some(*hypothesis),
        )
        .unwrap();
        let binary_entropy = -(0.9_f64 * 0.9_f64.ln() + 0.1_f64 * 0.1_f64.ln());
        assert!(uniform > skewed);
        assert!(skewed <= binary_entropy + 1.0e-12);
        assert!(skewed > 0.0);
    }

    #[test]
    fn weighted_eig_fails_closed_on_invalid_or_degenerate_priors() {
        let hypotheses = [0_usize, 1];
        let predict = |hypothesis: &usize, _| Some(*hypothesis as f64);
        assert!(expected_information_gain_nats_with_prior_weights(
            &(), &hypotheses, &[1.0], 1.0, predict
        )
        .is_none());
        assert!(expected_information_gain_nats_with_prior_weights(
            &(), &hypotheses, &[f64::NAN, 1.0], 1.0, predict
        )
        .is_none());
        assert!(expected_information_gain_nats_with_prior_weights(
            &(), &hypotheses, &[-0.1, 1.1], 1.0, predict
        )
        .is_none());
        assert!(expected_information_gain_nats_with_prior_weights(
            &(), &hypotheses, &[1.0, 0.0], 1.0, predict
        )
        .is_none());
        assert!(expected_information_gain_nats_with_prior_weights(
            &(), &hypotheses, &[0.0, 0.0], 1.0, predict
        )
        .is_none());
    }

    #[test]
    fn weighted_eig_zero_noise_limit_is_entropy_of_distinct_prediction_mass() {
        let hypotheses = [0_usize, 1, 2];
        let gain = expected_information_gain_with_prior_weights_and_steps(
            &(),
            &hypotheses,
            &[0.2, 0.3, 0.5],
            1.0e-300,
            INFORMATION_GAIN_QUADRATURE_STEPS,
            |hypothesis, _| Some(if *hypothesis < 2 { 7.0 } else { 9.0 }),
        )
        .unwrap();
        let expected = -(0.5_f64 * 0.5_f64.ln() + 0.5_f64 * 0.5_f64.ln());
        assert!((gain - expected).abs() < 1.0e-8);
    }

    #[test]
    fn prior_weighted_selector_is_invariant_to_duplicate_hypotheses_with_split_mass() {
        let candidates = ["candidate"];
        let two_hypotheses = [0_usize, 1];
        let two = select_experiment_by_information_gain_per_cost_with_prior_weights_and_noise_and_eligibility(
            &candidates,
            &two_hypotheses,
            &[0.5, 0.5],
            |_| true,
            |_| 0.25,
            |hypothesis, _| Some(*hypothesis as f64),
            |_| 1.0,
        )
        .unwrap();

        let three_hypotheses = [0_usize, 1, 2];
        let duplicated = select_experiment_by_information_gain_per_cost_with_prior_weights_and_noise_and_eligibility(
            &candidates,
            &three_hypotheses,
            &[0.25, 0.25, 0.5],
            |_| true,
            |_| 0.25,
            |hypothesis, _| Some(if *hypothesis == 2 { 1.0 } else { 0.0 }),
            |_| 1.0,
        )
        .unwrap();
        assert!((two.1 - duplicated.1).abs() < 1.0e-10);
    }

    #[test]
    fn prior_weighted_robust_selector_keeps_hard_eligibility_and_scenario_floor() {
        let hypotheses = [0_usize, 1];
        let candidates = ["blocked", "fragile", "robust"];
        let scenarios = ["nominal", "shift"];
        let selected = select_experiment_by_worst_case_information_gain_per_cost_with_prior_weights(
            &candidates,
            &hypotheses,
            &[0.5, 0.5],
            &scenarios,
            |candidate| *candidate != "blocked",
            |_| 0.1,
            |hypothesis, candidate, scenario| match (*candidate, *scenario, *hypothesis) {
                ("fragile", "nominal", 0) => Some(0.0),
                ("fragile", "nominal", 1) => Some(5.0),
                ("fragile", "shift", _) => Some(3.0),
                ("robust", _, 0) => Some(0.0),
                ("robust", _, 1) => Some(1.0),
                _ => None,
            },
            |candidate| {
                assert_ne!(*candidate, "blocked");
                1.0
            },
        )
        .expect("robust candidate has positive weighted information in all scenarios");
        assert_eq!(*selected.0, "robust");
    }

    #[test]
    fn robust_selector_rejects_nominally_informative_candidate_that_fails_under_shift() {
        let hypotheses = [0_usize, 1];
        let candidates = ["fragile", "robust"];
        let scenarios = ["nominal", "boundary_shift"];
        let predict = |hypothesis: &usize, candidate: &&str, scenario: &&str| match (
            *candidate,
            *scenario,
            *hypothesis,
        ) {
            ("fragile", "nominal", 0) => Some(0.0),
            ("fragile", "nominal", 1) => Some(5.0),
            ("fragile", "boundary_shift", _) => Some(2.0),
            ("robust", _, 0) => Some(0.0),
            ("robust", _, 1) => Some(1.0),
            _ => None,
        };
        let nominal = select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
            &candidates,
            &hypotheses,
            |_| true,
            |_| 0.1,
            |hypothesis, candidate| predict(hypothesis, candidate, &"nominal"),
            |_| 1.0,
        )
        .expect("nominal EIG has a positive-information candidate");
        assert_eq!(*nominal.0, "fragile");

        let robust = select_experiment_by_worst_case_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &scenarios,
            |_| true,
            |_| 0.1,
            predict,
            |_| 1.0,
        )
        .expect("robust candidate retains information in both scenarios");
        assert_eq!(*robust.0, "robust");
        assert!(robust.1.is_finite() && robust.1 > 0.0);
    }

    #[test]
    fn robust_selector_fails_closed_on_empty_scenario_set_or_incomplete_scenario() {
        let hypotheses = [0_usize, 1];
        let candidates = [0_usize, 1];
        assert!(
            select_experiment_by_worst_case_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[],
                |_| true,
                |_| 1.0,
                |hypothesis, candidate, _scenario: &usize| {
                    Some((*hypothesis * (*candidate + 1)) as f64)
                },
                |_| 1.0,
            )
            .is_none()
        );

        let scenarios = [0_usize, 1];
        let selected = select_experiment_by_worst_case_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &scenarios,
            |_| true,
            |_| 1.0,
            |hypothesis, candidate, scenario| {
                if *candidate == 0 && *scenario == 1 {
                    None
                } else {
                    Some((*hypothesis * (*candidate + 1)) as f64)
                }
            },
            |_| 1.0,
        );
        assert_eq!(selected.map(|(candidate, _)| *candidate), Some(1));
    }

    #[test]
    fn robust_selector_applies_eligibility_before_cost_noise_or_scenario_predictions() {
        let hypotheses = [0_usize, 1];
        let candidates = ["blocked", "eligible"];
        let scenarios = [0_usize, 1];
        let selected = select_experiment_by_worst_case_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &scenarios,
            |candidate| *candidate == "eligible",
            |candidate| {
                assert_ne!(*candidate, "blocked");
                1.0
            },
            |hypothesis, candidate, scenario| {
                assert_ne!(*candidate, "blocked");
                Some((*hypothesis + *scenario) as f64)
            },
            |candidate| {
                assert_ne!(*candidate, "blocked");
                1.0
            },
        )
        .expect("eligible candidate is informative in each scenario");
        assert_eq!(*selected.0, "eligible");
    }


    #[test]
    fn evsi_ignores_nuisance_information_that_cannot_change_the_decision() {
        let hypotheses = [0_usize, 1, 2, 3];
        let priors = [0.25, 0.25, 0.25, 0.25];
        let actions = ["ACCEPT_CLASS_A", "ACCEPT_CLASS_B"];
        let utility = |action: &&str, hypothesis: &usize| match (*action, *hypothesis) {
            ("ACCEPT_CLASS_A", 0 | 1) => 10.0,
            ("ACCEPT_CLASS_A", 2 | 3) => 0.0,
            ("ACCEPT_CLASS_B", 0 | 1) => 0.0,
            ("ACCEPT_CLASS_B", 2 | 3) => 10.0,
            _ => f64::NAN,
        };
        // This test separates within-decision-class nuisance hypotheses, but does
        // not reveal which action class is correct. It has information, not decision value.
        let nuisance = expected_value_of_sample_information(
            &"nuisance",
            &hypotheses,
            &priors,
            &actions,
            0.1,
            |h, _| Some(if *h % 2 == 0 { 0.0 } else { 4.0 }),
            utility,
        )
        .expect("finite nuisance measurement model");
        let decision_relevant = expected_value_of_sample_information(
            &"decision",
            &hypotheses,
            &priors,
            &actions,
            0.1,
            |h, _| Some(if *h < 2 { 0.0 } else { 4.0 }),
            utility,
        )
        .expect("finite decision-relevant measurement model");
        assert!(nuisance < 1.0e-6, "nuisance EVSI should be zero, got {nuisance}");
        assert!(decision_relevant > 4.9, "decision-relevant EVSI too small: {decision_relevant}");
    }

    #[test]
    fn evsi_is_invariant_to_representation_duplication_when_prior_mass_is_split() {
        let actions = ["CHOOSE_ZERO", "CHOOSE_ONE"];
        let utility = |action: &&str, hypothesis: &usize| match (*action, *hypothesis) {
            ("CHOOSE_ZERO", 0) => 1.0,
            ("CHOOSE_ZERO", 1) => 0.0,
            ("CHOOSE_ONE", 0) => 0.0,
            ("CHOOSE_ONE", 1) => 1.0,
            _ => f64::NAN,
        };
        let base_h = [0_usize, 1];
        let base = expected_value_of_sample_information(
            &(), &base_h, &[0.5, 0.5], &actions, 0.2,
            |h, _| Some(*h as f64 * 2.0), utility,
        ).unwrap();

        let duplicate_h = [0_usize, 0, 1];
        let duplicate = expected_value_of_sample_information(
            &(), &duplicate_h, &[0.25, 0.25, 0.5], &actions, 0.2,
            |h, _| Some(*h as f64 * 2.0), |action, h| utility(action, h),
        ).unwrap();
        assert!((base - duplicate).abs() < 1.0e-10);
    }

    #[test]
    fn robust_evsi_rejects_a_nominally_useful_measurement_that_fails_under_shift() {
        let hypotheses = [0_usize, 1];
        let priors = [0.5, 0.5];
        let actions = ["A", "B"];
        let utility = |action: &&str, hypothesis: &usize| match (*action, *hypothesis) {
            ("A", 0) | ("B", 1) => 10.0,
            _ => 0.0,
        };
        let candidates = ["fragile", "robust"];
        let scenarios = ["nominal", "boundary_shift"];
        let predict = |h: &usize, candidate: &&str, scenario: &&str| match (*candidate, *scenario, *h) {
            ("fragile", "nominal", 0) => Some(0.0),
            ("fragile", "nominal", 1) => Some(5.0),
            ("fragile", "boundary_shift", _) => Some(2.0),
            ("robust", _, 0) => Some(0.0),
            ("robust", _, 1) => Some(2.0),
            _ => None,
        };
        let nominal = expected_value_of_sample_information(
            &"fragile", &hypotheses, &priors, &actions, 0.1,
            |h, candidate| predict(h, candidate, &"nominal"), utility,
        ).unwrap();
        let robust = select_experiment_by_representative_deamplifying_worst_case_evsi_per_cost(
            &candidates, &hypotheses, &priors, &actions, &scenarios,
            |_| true, |_| 1.0, |_| 0.0, |_| 0.1,
            predict, utility, |_| 1.0,
        ).expect("robust candidate has decision value in every scenario");
        assert!(nominal > 4.9);
        assert_eq!(*robust.0, "robust");
        assert!(robust.1 > 4.9);
    }

    #[test]
    fn evsi_eligibility_gate_precedes_every_candidate_specific_callback() {
        let hypotheses = [0_usize, 1];
        let priors = [0.5, 0.5];
        let actions = ["A", "B"];
        let scenarios = [0_usize];
        let candidates = ["blocked", "allowed"];
        let utility = |action: &&str, hypothesis: &usize| {
            if *action == "A" { 1.0 - *hypothesis as f64 } else { *hypothesis as f64 }
        };
        let chosen = select_experiment_by_representative_deamplifying_worst_case_evsi_per_cost(
            &candidates, &hypotheses, &priors, &actions, &scenarios,
            |c| *c == "allowed",
            |c| { assert_ne!(*c, "blocked"); 1.0 },
            |c| { assert_ne!(*c, "blocked"); 0.0 },
            |c| { assert_ne!(*c, "blocked"); 0.1 },
            |h, c, _| { assert_ne!(*c, "blocked"); Some(*h as f64) },
            utility,
            |c| { assert_ne!(*c, "blocked"); 1.0 },
        ).expect("eligible informative candidate");
        assert_eq!(*chosen.0, "allowed");
        assert!(select_experiment_by_representative_deamplifying_worst_case_evsi_per_cost(
            &candidates, &hypotheses, &priors, &actions, &scenarios,
            |_| false, |_| 1.0, |_| 0.0, |_| 0.1,
            |h, c, _| { assert_ne!(*c, "blocked"); Some(*h as f64) },
            utility, |_| 1.0,
        ).is_none());
    }

    #[test]
    fn evsi_fails_closed_on_invalid_utility_prior_or_scenario() {
        let hypotheses = [0_usize, 1];
        let actions = ["A", "B"];
        assert!(expected_value_of_sample_information(
            &(), &hypotheses, &[0.5, 0.5], &actions, 1.0,
            |h, _| Some(*h as f64), |_, _| f64::NAN,
        ).is_none());
        assert!(expected_value_of_sample_information(
            &(), &hypotheses, &[0.0, 0.0], &actions, 1.0,
            |h, _| Some(*h as f64), |_, _| 1.0,
        ).is_none());
        assert!(expected_value_of_sample_information(
            &(), &hypotheses, &[0.5, 0.5], &[], 1.0,
            |h, _| Some(*h as f64), |_, _| 1.0,
        ).is_none());
    }

    #[test]
    fn information_gain_selector_returns_none_when_every_candidate_is_uninformative() {
        let hypotheses = [0_i32, 1, 2];
        let candidates = [0.0_f64, 1.0, 2.0];
        assert!(
            select_experiment_by_information_gain_per_cost(
                &candidates,
                &hypotheses,
                1.0,
                |_, _| Some(7.0),
                |_| 1.0,
            )
            .is_none()
        );
    }

    #[test]
    fn midpoint_quadrature_matches_a_refined_reference_on_stress_cases() {
        let prediction_cases = [
            vec![-3.0_f64, -0.2, 0.4, 4.0],
            vec![-0.01_f64, 0.0, 0.01, 0.02],
            vec![0.0_f64, 0.0, 1.0, 1.0],
            vec![0.0_f64, 0.1, 3.0, 10.0],
            vec![-1.0_f64, -0.5, 0.25, 1.5, 2.0],
        ];
        let noise_levels = [0.03_f64, 0.1, 0.5, 1.0, 3.0, 10.0];

        for predictions in &prediction_cases {
            for observation_std_dev in noise_levels {
                let estimate = expected_information_gain_with_steps(
                    &(),
                    predictions,
                    observation_std_dev,
                    INFORMATION_GAIN_QUADRATURE_STEPS,
                    |prediction, _| Some(*prediction),
                )
                .expect("valid finite prediction set");
                let refined = expected_information_gain_with_steps(
                    &(),
                    predictions,
                    observation_std_dev,
                    2_048,
                    |prediction, _| Some(*prediction),
                )
                .expect("valid finite prediction set");

                assert!(
                    (estimate - refined).abs() <= 1.0e-6,
                    "256-step estimate diverged from 2048-step reference: predictions={predictions:?}, noise={observation_std_dev}, estimate={estimate}, refined={refined}"
                );
                assert!((0.0..=(predictions.len() as f64).ln()).contains(&estimate));
            }
        }

        assert!(
            expected_information_gain_with_steps(&(), &[0.0_f64, 1.0], 1.0, 0, |prediction, _| {
                Some(*prediction)
            },)
            .is_none()
        );
    }

    #[test]
    fn information_gain_is_robust_to_extreme_finite_predictions() {
        let hypotheses = [-1.0e308_f64, 1.0e308_f64];
        let score = expected_information_gain_nats(&(), &hypotheses, 1.0, |h, _| Some(*h))
            .expect("finite means and noise");
        assert!((score - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn works_directly_against_expr_hypotheses() {
        use crate::hdc::conjecture_engine::BinOp;
        let var = |n: &str| Expr::Var(n.to_string());
        // Two candidate "laws": y = x, and y = x^2.
        let h1 = var("x");
        let h2 = Expr::BinOp(BinOp::Pow, Box::new(var("x")), Box::new(Expr::Const(2.0)));
        let hypotheses = vec![h1, h2];
        let predict = |h: &Expr, x: &f64| {
            let v = h.eval(&[("x", *x)]);
            v.is_finite().then_some(v)
        };
        // x=1: both predict 1 (agree). x=3: predict 3 vs 9 (disagree a lot).
        let candidates = [1.0, 3.0];
        let (chosen, _) = select_most_informative_experiment(&candidates, &hypotheses, predict)
            .expect("non-empty pool");
        assert_eq!(
            *chosen, 3.0,
            "should prefer the point where x vs x^2 diverge most"
        );
    }

    #[test]
    fn selector_skips_non_finite_and_under_supported_candidates() {
        let hypotheses = [0_i32, 1_i32, 2_i32];
        let predict = |hypothesis: &i32, experiment: &f64| {
            if *experiment == 0.0 {
                if *hypothesis == 0 {
                    Some(1.0)
                } else {
                    Some(f64::NAN)
                }
            } else if *experiment == 1.0 {
                match *hypothesis {
                    0 => Some(0.0),
                    1 => Some(10.0),
                    _ => Some(f64::INFINITY),
                }
            } else {
                None
            }
        };

        assert_eq!(
            assess_epistemic_value(&0.0, &hypotheses, predict),
            EpistemicAssessment::InsufficientValidPredictions {
                valid_predictions: 1
            }
        );

        let candidates = [0.0, 1.0];
        let (chosen, score) = select_most_informative_experiment(&candidates, &hypotheses, predict)
            .expect("the second candidate has two finite, disagreeing predictions");
        assert_eq!(*chosen, 1.0);
        assert!((score - 25.0).abs() < 1e-12);
        assert!(score.is_finite());
    }

    #[test]
    fn assessment_distinguishes_unrepresentable_variance() {
        let extreme = [-1.0e308_f64, 1.0e308_f64];
        assert_eq!(
            assess_epistemic_value(&(), &extreme, |value, _| Some(*value)),
            EpistemicAssessment::UnrepresentableScore {
                valid_predictions: 2
            }
        );

        // The scaled calculation still returns zero for equal large values,
        // instead of overflowing while summing the mean.
        let equal_large = [1.0e308_f64, 1.0e308_f64];
        assert_eq!(
            assess_epistemic_value(&(), &equal_large, |value, _| Some(*value)),
            EpistemicAssessment::Score {
                value: 0.0,
                valid_predictions: 2
            }
        );
    }

    #[test]
    fn selector_returns_none_when_no_candidate_has_positive_disagreement() {
        let hypotheses = [1_i32, 2_i32, 3_i32];
        let predict = |_: &i32, _: &f64| Some(4.0);
        let candidates = [0.0, 1.0, 2.0];
        assert!(select_most_informative_experiment(&candidates, &hypotheses, predict).is_none());
    }
}
