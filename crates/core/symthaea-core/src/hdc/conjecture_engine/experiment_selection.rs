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
    maximum + values.iter().map(|value| (*value - maximum).exp()).sum::<f64>().ln()
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

/// Estimate expected information gain under explicit prior weights, in nats.
///
/// Unlike the uniform-prior compatibility API, this function lets callers allocate prior mass
/// across structural families or validated equivalence classes, then split that mass across
/// representations within a family. Weights must be finite and non-negative, have one entry per
/// hypothesis, and have positive total mass. They are normalized internally.
///
/// This API does not infer semantic equivalence or deduplicate expressions: callers must assign
/// weights according to a documented prior policy. Every hypothesis must still produce a finite
/// prediction at the experiment, including zero-weight hypotheses, so invalid candidate behavior
/// is not silently hidden. The common Gaussian-noise assumption and quadrature method are the
/// same as in the uniform-prior API.
pub fn expected_information_gain_with_prior_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    if hypotheses.len() < 2
        || prior_weights.len() != hypotheses.len()
        || !observation_std_dev.is_finite()
        || observation_std_dev <= 0.0
    {
        return None;
    }

    if prior_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }
    let max_weight = prior_weights.iter().copied().fold(0.0_f64, f64::max);
    if max_weight <= 0.0 {
        return None;
    }
    let scaled_weights: Vec<f64> = prior_weights
        .iter()
        .map(|weight| weight / max_weight)
        .collect();
    let weight_sum = scaled_weights.iter().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return None;
    }
    let prior_probabilities: Vec<f64> = scaled_weights
        .iter()
        .map(|weight| weight / weight_sum)
        .collect();
    let prior_entropy = prior_probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();

    let predictions: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| predict(hypothesis, experiment))
        .collect::<Option<Vec<_>>>()?;
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        return None;
    }
    if prior_entropy == 0.0 {
        return Some(0.0);
    }

    // Scale means and noise by a shared finite magnitude before subtraction.
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
        return Some(noiseless_information_gain_with_priors(
            &predictions,
            &prior_probabilities,
        ));
    }

    let step_width = 2.0 * INFORMATION_GAIN_Z_LIMIT / INFORMATION_GAIN_QUADRATURE_STEPS as f64;
    let mut quadrature = Vec::with_capacity(INFORMATION_GAIN_QUADRATURE_STEPS);
    for step_index in 0..INFORMATION_GAIN_QUADRATURE_STEPS {
        let z =
            -INFORMATION_GAIN_Z_LIMIT + (step_index as f64 + 0.5) * step_width;
        quadrature.push((z, (-0.5 * z * z).exp() * step_width));
    }
    let weight_sum: f64 = quadrature.iter().map(|(_, weight)| weight).sum();
    let log_prior_probabilities: Vec<f64> = prior_probabilities
        .iter()
        .map(|probability| {
            if *probability > 0.0 {
                probability.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let mut log_terms = Vec::with_capacity(predictions.len());
    let mut information_gain = 0.0;

    for (index, true_mean) in scaled_means.iter().copied().enumerate() {
        let true_probability = prior_probabilities[index];
        if true_probability == 0.0 {
            continue;
        }
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
                let prior_index = log_terms.len();
                log_terms.push(log_prior_probabilities[prior_index] + log_likelihood_ratio);
            }
            let log_mixture_ratio = log_sum_exp(&log_terms);
            information_gain -=
                (weight / weight_sum) * true_probability * log_mixture_ratio;
        }
    }

    if !information_gain.is_finite() {
        return None;
    }
    Some(information_gain.clamp(0.0, prior_entropy))
}

fn noiseless_information_gain_with_priors(predictions: &[f64], priors: &[f64]) -> f64 {
    let mut groups: Vec<(f64, f64)> = Vec::new();
    for (prediction, prior) in predictions.iter().copied().zip(priors.iter().copied()) {
        if prior == 0.0 {
            continue;
        }
        if let Some((_, group_probability)) =
            groups.iter_mut().find(|(value, _)| *value == prediction)
        {
            *group_probability += prior;
        } else {
            groups.push((prediction, prior));
        }
    }
    groups
        .iter()
        .map(|(_, probability)| -probability * probability.ln())
        .sum()
}

/// Select the candidate maximizing weighted-prior expected information gain per unit cost.
///
/// The weights describe hypothesis prior mass, not arbitrary confidence scores. They must
/// remain consistent when representations are duplicated; split a family's mass across its
/// candidate instances or construct the weights at the structural-family level.
pub fn select_experiment_by_weighted_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(information_gain) = expected_information_gain_with_prior_nats(
            experiment,
            hypotheses,
            prior_weights,
            observation_std_dev,
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
    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(information_gain) =
            expected_information_gain_nats(experiment, hypotheses, observation_std_dev, predict)
        else {
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
        assert!(score.abs() < 1.0e-12, "identical predictions should teach nothing: {score}");
    }

    #[test]
    fn information_gain_prefers_balanced_discrimination_over_a_single_outlier() {
        // Raw variance heavily favors [0, 0, 0, 100], although one observation
        // can only identify one hypothesis against three indistinguishable ones.
        let outlier = [0.0, 0.0, 0.0, 100.0];
        let balanced = [0.0, 0.0, 10.0, 10.0];
        assert!(variance(&outlier) > variance(&balanced));
        let hypotheses = [0_usize, 1, 2, 3];
        let outlier_gain = expected_information_gain_nats(
            &0_usize,
            &hypotheses,
            1.0,
            |hypothesis, _| Some(outlier[*hypothesis]),
        )
        .expect("complete finite predictions");
        let balanced_gain = expected_information_gain_nats(
            &0_usize,
            &hypotheses,
            1.0,
            |hypothesis, _| Some(balanced[*hypothesis]),
        )
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
            expected_information_gain_nats(&(), &hypotheses, 0.0, |h, _| Some(*h as f64))
                .is_none()
        );
        assert!(
            expected_information_gain_nats(&(), &hypotheses, f64::NAN, |h, _| Some(*h as f64))
                .is_none()
        );
        assert!(expected_information_gain_nats(&(), &hypotheses, 1.0, |h, _| {
            (*h != 1).then_some(*h as f64)
        })
        .is_none());
        assert!(
            expected_information_gain_nats(&(), &[1_i32], 1.0, |h, _| Some(*h as f64)).is_none()
        );
    }

    #[test]
    fn information_gain_selector_maximizes_information_per_cost_not_raw_information() {
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [1_u8, 2, 3];
        let predict = |hypothesis: &usize, experiment: &u8| {
            match (*experiment, *hypothesis) {
                (1, 0 | 1) => Some(0.0),
                (1, 2 | 3) => Some(1.0),
                (2, 0 | 1) => Some(0.0),
                (2, 2 | 3) => Some(10.0),
                (3, 0) => Some(-100.0),
                (3, 1) => Some(-50.0),
                (3, 2) => Some(50.0),
                (3, 3) => Some(100.0),
                _ => None,
            }
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
            expected_information_gain_with_steps(
                &(),
                &[0.0_f64, 1.0],
                1.0,
                0,
                |prediction, _| Some(*prediction),
            )
            .is_none()
        );
    }

    #[test]
    fn weighted_information_gain_matches_uniform_api_for_equal_weights() {
        let hypotheses = [0.0_f64, 1.0, 3.0, 5.0];
        let priors = [1.0, 1.0, 1.0, 1.0];
        let uniform = expected_information_gain_nats(&(), &hypotheses, 0.7, |h, _| Some(*h))
            .expect("valid uniform score");
        let weighted = expected_information_gain_with_prior_nats(
            &(),
            &hypotheses,
            &priors,
            0.7,
            |h, _| Some(*h),
        )
        .expect("valid weighted score");
        assert!((uniform - weighted).abs() < 1e-10);
    }

    #[test]
    fn weighted_prior_is_invariant_to_duplicate_representation_when_mass_is_split() {
        let original_hypotheses = [0_usize, 1, 2, 3];
        let original_priors = [1.0, 1.0, 1.0, 1.0];
        let duplicated_hypotheses = [0_usize, 1, 2, 3, 0];
        // The duplicated instance splits the original family mass instead of increasing it.
        let duplicated_priors = [1.0, 2.0, 2.0, 2.0, 1.0];
        let prediction_table = [[0.0_f64, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 100.0]];
        let experiments = [0_usize, 1_usize];

        for experiment in experiments {
            let original = expected_information_gain_with_prior_nats(
                &experiment,
                &original_hypotheses,
                &original_priors,
                1.0,
                |hypothesis, experiment| Some(prediction_table[*experiment][*hypothesis]),
            )
            .expect("valid original prior");
            let duplicated = expected_information_gain_with_prior_nats(
                &experiment,
                &duplicated_hypotheses,
                &duplicated_priors,
                1.0,
                |hypothesis, experiment| Some(prediction_table[*experiment][*hypothesis]),
            )
            .expect("valid split family prior");
            assert!(
                (original - duplicated).abs() < 1.0e-6,
                "duplicating a representation must not change the prior mass of its family: original={original}, duplicated={duplicated}"
            );
        }

        let original_choice = select_experiment_by_weighted_information_gain_per_cost(
            &experiments,
            &original_hypotheses,
            &original_priors,
            1.0,
            |hypothesis, experiment| Some(prediction_table[*experiment][*hypothesis]),
            |_| 1.0,
        )
        .expect("an informative original experiment")
        .0;
        let duplicate_choice = select_experiment_by_weighted_information_gain_per_cost(
            &experiments,
            &duplicated_hypotheses,
            &duplicated_priors,
            1.0,
            |hypothesis, experiment| Some(prediction_table[*experiment][*hypothesis]),
            |_| 1.0,
        )
        .expect("an informative duplicate-invariant experiment")
        .0;
        assert_eq!(original_choice, duplicate_choice);
        assert_eq!(*original_choice, 0);
    }

    #[test]
    fn weighted_information_gain_rejects_invalid_priors_and_accepts_zero_mass() {
        let hypotheses = [0_i32, 1, 2];
        assert!(
            expected_information_gain_with_prior_nats(&(), &hypotheses, &[1.0, 2.0], 1.0, |h, _| {
                Some(*h as f64)
            })
            .is_none()
        );
        assert!(
            expected_information_gain_with_prior_nats(&(), &hypotheses, &[1.0, -1.0, 1.0], 1.0, |h, _| {
                Some(*h as f64)
            })
            .is_none()
        );
        assert!(
            expected_information_gain_with_prior_nats(&(), &hypotheses, &[0.0, 0.0, 0.0], 1.0, |h, _| {
                Some(*h as f64)
            })
            .is_none()
        );
        assert!(
            expected_information_gain_with_prior_nats(&(), &hypotheses, &[1.0, f64::NAN, 1.0], 1.0, |h, _| {
                Some(*h as f64)
            })
            .is_none()
        );
        assert_eq!(
            expected_information_gain_with_prior_nats(&(), &hypotheses, &[0.0, 0.0, 1.0], 1.0, |h, _| {
                Some(*h as f64)
            }),
            Some(0.0)
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
