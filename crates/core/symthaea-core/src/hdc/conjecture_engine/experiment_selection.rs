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

/// Normalize finite non-negative prior masses. At least two hypotheses must retain
/// positive mass so the result measures discrimination rather than a degenerate prior.
fn normalized_prior_probabilities(
    prior_weights: &[f64],
    hypothesis_count: usize,
) -> Option<Vec<f64>> {
    if hypothesis_count < 2
        || prior_weights.len() != hypothesis_count
        || prior_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }

    // Normalize after dividing by the largest mass. Summing finite inputs first can
    // overflow (e.g. [f64::MAX, f64::MAX, 1.0]) even though their ratios are valid.
    let maximum = prior_weights.iter().copied().fold(0.0_f64, f64::max);
    if maximum <= 0.0 {
        return None;
    }
    let scaled_weights: Vec<f64> = prior_weights
        .iter()
        .map(|weight| weight / maximum)
        .collect();
    let total = scaled_weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let probabilities: Vec<f64> = scaled_weights.iter().map(|weight| weight / total).collect();
    (probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .count()
        >= 2)
        .then_some(probabilities)
}

/// Zero-noise limit when observationally identical prediction means form a single group.
fn noiseless_weighted_information_gain(predictions: &[f64], prior: &[f64]) -> f64 {
    let mut groups: Vec<(f64, f64)> = Vec::new();
    for (&prediction, &probability) in predictions.iter().zip(prior) {
        if probability == 0.0 {
            continue;
        }
        if let Some((_, group_probability)) =
            groups.iter_mut().find(|(value, _)| *value == prediction)
        {
            *group_probability += probability;
        } else {
            groups.push((prediction, probability));
        }
    }
    groups
        .iter()
        .filter(|(_, probability)| *probability > 0.0)
        .map(|(_, probability)| -probability * probability.ln())
        .sum()
}

/// Estimate expected information gain under caller-supplied prior masses.
///
/// Prior masses are normalized internally; zero-mass hypotheses do not contribute to the
/// mixture, but predictions must still be complete and finite for every supplied hypothesis.
/// Splitting one hypothesis' prior mass across observationally identical duplicates leaves the
/// score unchanged. This avoids making the score depend on how many times a candidate was
/// inserted into the list, provided callers supply equivalent prior mass. It does not infer
/// calibrated posterior weights or establish structural equivalence between expressions.
///
/// Uses the same deterministic Gaussian midpoint quadrature as the uniform-prior API. Returns
/// None for a malformed prior, fewer than two positive-mass hypotheses, invalid noise, or
/// incomplete/non-finite predictions. Scores are in nats and bounded by the entropy of the
/// normalized prior over observationally distinguishable prediction groups.
pub fn expected_information_gain_weighted_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<f64> {
    if hypotheses.len() < 2 || !observation_std_dev.is_finite() || observation_std_dev <= 0.0 {
        return None;
    }
    let prior = normalized_prior_probabilities(prior_weights, hypotheses.len())?;
    let predictions: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| predict(hypothesis, experiment))
        .collect::<Option<Vec<_>>>()?;
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        return None;
    }

    let mean_scale = predictions
        .iter()
        .zip(&prior)
        .filter(|(_, probability)| **probability > 0.0)
        .map(|(prediction, _)| prediction.abs())
        .fold(observation_std_dev, f64::max);
    let scaled_means: Vec<f64> = predictions
        .iter()
        .zip(&prior)
        .map(|(prediction, probability)| {
            if *probability > 0.0 {
                prediction / mean_scale
            } else {
                0.0
            }
        })
        .collect();
    let scaled_noise = observation_std_dev / mean_scale;
    if scaled_noise == 0.0 {
        return Some(noiseless_weighted_information_gain(&predictions, &prior));
    }

    let step_width = 2.0 * INFORMATION_GAIN_Z_LIMIT / INFORMATION_GAIN_QUADRATURE_STEPS as f64;
    let mut quadrature = Vec::with_capacity(INFORMATION_GAIN_QUADRATURE_STEPS);
    for step_index in 0..INFORMATION_GAIN_QUADRATURE_STEPS {
        let z = -INFORMATION_GAIN_Z_LIMIT + (step_index as f64 + 0.5) * step_width;
        quadrature.push((z, (-0.5 * z * z).exp() * step_width));
    }
    let weight_sum: f64 = quadrature.iter().map(|(_, weight)| weight).sum();
    let prior_entropy = prior
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 || !prior_entropy.is_finite() {
        return None;
    }

    let prior_log_probabilities: Vec<f64> = prior
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
        let true_prior = prior[index];
        if true_prior == 0.0 {
            continue;
        }
        for &(z, weight) in &quadrature {
            log_terms.clear();
            for (other_index, other_mean) in scaled_means.iter().copied().enumerate() {
                let separation = (true_mean - other_mean) / scaled_noise;
                let log_likelihood_ratio = if separation == 0.0 {
                    0.0
                } else if !separation.is_finite() || separation.abs() > 1.0e6 {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * separation * separation - z * separation
                };
                log_terms.push(prior_log_probabilities[other_index] + log_likelihood_ratio);
            }
            let log_mixture_ratio = log_sum_exp(&log_terms);
            information_gain -= (weight / weight_sum) * true_prior * log_mixture_ratio;
        }
    }

    if !information_gain.is_finite() {
        return None;
    }
    Some(information_gain.clamp(0.0, prior_entropy))
}

/// Select by weighted expected information gain divided by positive finite experiment cost.
///
/// This opt-in API leaves the uniform-prior selector unchanged. Candidate predictions must be
/// complete, and invalid costs or non-informative candidates are skipped. Ties preserve input
/// order. The prior is supplied by the caller; it is not automatically calibrated by this code.
pub fn select_experiment_by_weighted_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    observation_std_dev: f64,
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if normalized_prior_probabilities(prior_weights, hypotheses.len()).is_none() {
        return None;
    }
    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(information_gain) = expected_information_gain_weighted_nats(
            experiment,
            hypotheses,
            prior_weights,
            observation_std_dev,
            predict,
        ) else {
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
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Compute conservative expected information gain over a finite set of plausible
/// observation-noise standard deviations.
///
/// Compute conservative expected information gain over a finite set of plausible
/// observation-noise standard deviations.
///
/// Under the shared additive Gaussian-noise model, increasing noise is a
/// post-processing channel, so exact mutual information cannot increase with
/// noise. Thus the worst case is the largest supplied standard deviation; this
/// computes that score once instead of repeating the quadrature for every
/// scenario. The finite set is still validated in full, and this is not a bound
/// over unlisted noise scales or protection against a misspecified hypothesis
/// class. An empty/invalid scenario set, incomplete predictions, or non-finite
/// prediction fails closed with `None`.
pub fn expected_information_gain_worst_case_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    observation_std_devs: &[f64],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<f64> {
    if observation_std_devs.is_empty()
        || observation_std_devs
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma <= 0.0)
    {
        return None;
    }

    let worst_case_std_dev = observation_std_devs.iter().copied().fold(0.0_f64, f64::max);
    expected_information_gain_nats(experiment, hypotheses, worst_case_std_dev, predict)
}

/// Select the experiment maximizing worst-case expected information gain per cost.
///
/// Each candidate is scored at the largest supplied observation-noise scale;
/// under the shared additive Gaussian-noise model, this realizes the exact
/// information-gain minimum across the finite set by data-processing monotonicity.
/// The score is divided by positive finite cost. This is conservative only with
/// respect to the caller-provided noise scales. It does not replace discrepancy
/// audits, posterior calibration, safety constraints, or out-of-model hypothesis
/// checks. Invalid scenario sets are rejected before ranking; candidates with
/// invalid predictions or costs are skipped. Input-order tie-breaking is stable.
pub fn select_experiment_by_worst_case_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    observation_std_devs: &[f64],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if observation_std_devs.is_empty()
        || observation_std_devs
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma <= 0.0)
    {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(worst_case_gain) = expected_information_gain_worst_case_nats(
            experiment,
            hypotheses,
            observation_std_devs,
            predict,
        ) else {
            continue;
        };
        let score = worst_case_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// One finite scenario for robust expected-information-gain evaluation.
///
/// `prediction_model` is an opaque, caller-defined scenario identifier passed to
/// the prediction closure. It can represent an alternative structural model,
/// omitted-term hypothesis, nuisance setting, or other plausible predictive world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InformationGainScenario {
    /// Standard deviation of the observation noise for this scenario.
    pub observation_std_dev: f64,
    /// Caller-defined predictive-model/discrepancy scenario identifier.
    pub prediction_model: usize,
}

/// Compute the minimum expected information gain across explicit joint scenarios
/// of observation noise and plausible predictive-model variants.
///
/// Unlike a noise-only envelope, scenarios may produce different predictions
/// for a given candidate experiment. This allows the acquisition rule to account
/// for a finite set of structural/discrepancy alternatives rather than only
/// choosing the largest Gaussian noise scale. The supplied scenario set is not
/// a guarantee over unlisted worlds, and the caller remains responsible for
/// defining defensible alternatives. Empty scenarios, invalid noise, or any
/// incomplete/non-finite scenario predictions fail closed with `None`.
pub fn expected_information_gain_over_scenarios_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    scenarios: &[InformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &InformationGainScenario) -> Option<f64> + Copy,
) -> Option<f64> {
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite() || scenario.observation_std_dev <= 0.0
        })
    {
        return None;
    }

    let mut worst_case = f64::INFINITY;
    for scenario in scenarios {
        let information_gain = expected_information_gain_nats(
            experiment,
            hypotheses,
            scenario.observation_std_dev,
            |hypothesis, experiment| predict_under_scenario(hypothesis, experiment, scenario),
        )?;
        worst_case = worst_case.min(information_gain);
    }

    (worst_case.is_finite() && worst_case >= 0.0).then_some(worst_case)
}

/// Select the candidate maximizing scenario-robust information gain per cost.
///
/// Each candidate is evaluated under every finite joint scenario, using that
/// scenario's noise scale and predictive model variant. The minimum information
/// gain is divided by positive finite cost. This is only robust to the supplied
/// scenario set; it is not a calibrated posterior, a formal uncertainty bound,
/// or a substitute for open-world audits. Invalid scenario sets fail closed,
/// invalid candidate costs/predictions are skipped, and input-order tie breaking
/// is deterministic.
pub fn select_experiment_by_scenario_robust_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    scenarios: &[InformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &InformationGainScenario) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite() || scenario.observation_std_dev <= 0.0
        })
    {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(worst_case_gain) = expected_information_gain_over_scenarios_nats(
            experiment,
            hypotheses,
            scenarios,
            predict_under_scenario,
        ) else {
            continue;
        };
        let score = worst_case_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// One finite scenario that includes its own hypothesis-prior masses plus observation
/// noise and a caller-defined predictive-model variant.
///
/// Prior masses are supplied, not estimated. Each scenario is evaluated as a separate
/// plausible belief/model/noise world; the robust envelope takes the minimum information gain.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorInformationGainScenario {
    /// Standard deviation of Gaussian observation noise for this scenario.
    pub observation_std_dev: f64,
    /// Caller-defined predictive-model or discrepancy scenario identifier.
    pub prediction_model: usize,
    /// Non-negative hypothesis prior masses; normalized by the evaluator.
    pub prior_weights: Vec<f64>,
}

/// Compute the minimum weighted EIG across explicit prior, predictive-model, and noise scenarios.
///
/// This treats prior uncertainty as part of the scenario set rather than holding one prior fixed.
/// The returned lower envelope applies only to the caller-supplied finite set: it is not a
/// distributionally robust guarantee over all possible priors or unlisted worlds.
pub fn expected_information_gain_over_prior_scenarios_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    scenarios: &[PriorInformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &PriorInformationGainScenario) -> Option<f64> + Copy,
) -> Option<f64> {
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite()
                || scenario.observation_std_dev <= 0.0
                || normalized_prior_probabilities(&scenario.prior_weights, hypotheses.len())
                    .is_none()
        })
    {
        return None;
    }

    let mut worst_case = f64::INFINITY;
    for scenario in scenarios {
        let information_gain = expected_information_gain_weighted_nats(
            experiment,
            hypotheses,
            &scenario.prior_weights,
            scenario.observation_std_dev,
            |hypothesis, experiment| predict_under_scenario(hypothesis, experiment, scenario),
        )?;
        worst_case = worst_case.min(information_gain);
    }

    (worst_case.is_finite() && worst_case >= 0.0).then_some(worst_case)
}

/// Select by minimum weighted EIG across explicit prior/model/noise scenarios, divided by cost.
///
/// Invalid scenario declarations reject the selection. A candidate with incomplete predictions
/// in any scenario or with an invalid cost is skipped. Ties preserve input order.
pub fn select_experiment_by_prior_robust_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    scenarios: &[PriorInformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &PriorInformationGainScenario) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite()
                || scenario.observation_std_dev <= 0.0
                || normalized_prior_probabilities(&scenario.prior_weights, hypotheses.len())
                    .is_none()
        })
    {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(worst_case_gain) = expected_information_gain_over_prior_scenarios_nats(
            experiment,
            hypotheses,
            scenarios,
            predict_under_scenario,
        ) else {
            continue;
        };
        let score = worst_case_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
        {
            best = Some((experiment, score));
        }
    }
    best
}

/// Compute the minimum weighted expected information gain across explicit joint
/// scenarios of observation noise and plausible predictive-model variants.
///
/// The caller-supplied prior is held fixed across scenarios; only the predictive
/// model and noise scale vary. This is a finite-scenario lower envelope, not a
/// guarantee over unlisted worlds or over uncertainty in the prior itself.
pub fn expected_information_gain_weighted_over_scenarios_nats<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prior_weights: &[f64],
    scenarios: &[InformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &InformationGainScenario) -> Option<f64> + Copy,
) -> Option<f64> {
    normalized_prior_probabilities(prior_weights, hypotheses.len())?;
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite() || scenario.observation_std_dev <= 0.0
        })
    {
        return None;
    }

    let mut worst_case = f64::INFINITY;
    for scenario in scenarios {
        let information_gain = expected_information_gain_weighted_nats(
            experiment,
            hypotheses,
            prior_weights,
            scenario.observation_std_dev,
            |hypothesis, experiment| predict_under_scenario(hypothesis, experiment, scenario),
        )?;
        worst_case = worst_case.min(information_gain);
    }

    (worst_case.is_finite() && worst_case >= 0.0).then_some(worst_case)
}

/// Select by minimum weighted expected information gain across the supplied scenarios,
/// divided by positive finite experiment cost.
///
/// The prior is supplied by the caller and remains fixed across scenarios. Candidate
/// predictions must be complete and finite for every scenario; invalid priors or scenario
/// declarations reject the selection, while invalid costs or candidate predictions skip
/// that candidate. Ties preserve input order.
pub fn select_experiment_by_weighted_scenario_robust_information_gain_per_cost<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    prior_weights: &[f64],
    scenarios: &[InformationGainScenario],
    predict_under_scenario: impl Fn(&H, &E, &InformationGainScenario) -> Option<f64> + Copy,
    cost: impl Fn(&E) -> f64 + Copy,
) -> Option<(&'a E, f64)> {
    normalized_prior_probabilities(prior_weights, hypotheses.len())?;
    if scenarios.is_empty()
        || scenarios.iter().any(|scenario| {
            !scenario.observation_std_dev.is_finite() || scenario.observation_std_dev <= 0.0
        })
    {
        return None;
    }

    let mut best: Option<(&'a E, f64)> = None;
    for experiment in candidates {
        let experiment_cost = cost(experiment);
        if !experiment_cost.is_finite() || experiment_cost <= 0.0 {
            continue;
        }
        let Some(worst_case_gain) = expected_information_gain_weighted_over_scenarios_nats(
            experiment,
            hypotheses,
            prior_weights,
            scenarios,
            predict_under_scenario,
        ) else {
            continue;
        };
        let score = worst_case_gain / experiment_cost;
        if score.is_finite() && score > 0.0 && best.is_none_or(|(_, best_score)| score > best_score)
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
    fn weighted_information_gain_matches_uniform_api_for_equal_masses() {
        let hypotheses = [0_usize, 1, 2, 3];
        let prediction = |hypothesis: &usize, _: &u8| Some([0.0_f64, 0.25, 1.5, 3.0][*hypothesis]);
        let uniform = expected_information_gain_nats(&0_u8, &hypotheses, 0.4, prediction).unwrap();
        let weighted = expected_information_gain_weighted_nats(
            &0_u8,
            &hypotheses,
            &[7.0, 7.0, 7.0, 7.0],
            0.4,
            prediction,
        )
        .unwrap();
        assert!((uniform - weighted).abs() <= 1.0e-12);
    }

    #[test]
    fn weighted_information_gain_is_invariant_to_splitting_duplicate_prior_mass() {
        let hypotheses = [0_usize, 1];
        let base = expected_information_gain_weighted_nats(
            &(),
            &hypotheses,
            &[0.5, 0.5],
            0.3,
            |hypothesis, _| Some([0.0_f64, 1.0][*hypothesis]),
        )
        .unwrap();

        let duplicated_hypotheses = [0_usize, 1, 2];
        let duplicated = expected_information_gain_weighted_nats(
            &(),
            &duplicated_hypotheses,
            &[0.25, 0.25, 0.5],
            0.3,
            |hypothesis, _| Some([0.0_f64, 0.0, 1.0][*hypothesis]),
        )
        .unwrap();
        assert!((base - duplicated).abs() <= 1.0e-12);
    }

    #[test]
    fn weighted_information_gain_rejects_malformed_priors_and_predictions() {
        let hypotheses = [0_i32, 1, 2];
        let prediction = |hypothesis: &i32, _: &()| Some(*hypothesis as f64);
        for weights in [
            &[][..],
            &[1.0, 1.0][..],
            &[1.0, -1.0, 1.0][..],
            &[1.0, f64::NAN, 1.0][..],
            &[0.0, 0.0, 0.0][..],
            &[1.0, 0.0, 0.0][..],
        ] {
            assert!(
                expected_information_gain_weighted_nats(&(), &hypotheses, weights, 1.0, prediction)
                    .is_none()
            );
        }
        assert!(
            expected_information_gain_weighted_nats(
                &(),
                &hypotheses,
                &[1.0, 1.0, 1.0],
                1.0,
                |hypothesis, _| if *hypothesis == 1 {
                    None
                } else {
                    Some(*hypothesis as f64)
                },
            )
            .is_none()
        );
    }

    #[test]
    fn weighted_prior_normalization_avoids_finite_sum_overflow() {
        let weights = [f64::MAX, f64::MAX, 1.0];
        let probabilities = normalized_prior_probabilities(&weights, weights.len())
            .expect("finite masses should normalize without summing them first");

        assert_eq!(probabilities[0], probabilities[1]);
        assert!((probabilities[0] - 0.5).abs() <= 1.0e-15);
        assert!(probabilities[2] > 0.0 && probabilities[2].is_finite());
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() <= 1.0e-15);
    }

    #[test]
    fn weighted_selector_respects_prior_mass_when_uniform_partitions_tie() {
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [0_u8, 1_u8];
        let predict = |hypothesis: &usize, candidate: &u8| {
            let values = match candidate {
                // Under the weighted prior, this partitions mass 0.8 versus 0.2.
                0 => [0.0_f64, 10.0, 0.0, 10.0],
                // This partitions the same hypotheses 0.5 versus 0.5.
                1 => [0.0_f64, 0.0, 10.0, 10.0],
                _ => return None,
            };
            Some(values[*hypothesis])
        };
        let uniform = select_experiment_by_information_gain_per_cost(
            &candidates,
            &hypotheses,
            0.05,
            predict,
            |_| 1.0,
        )
        .expect("both partitions are informative");
        assert_eq!(
            *uniform.0, 0,
            "uniform-prior ties retain first-candidate order"
        );

        let weighted = select_experiment_by_weighted_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &[0.4, 0.1, 0.4, 0.1],
            0.05,
            predict,
            |_| 1.0,
        )
        .expect("both weighted partitions retain positive information");
        assert_eq!(
            *weighted.0, 1,
            "weighted selector should prefer balanced prior mass"
        );
        assert!(weighted.1 > 0.0 && weighted.1.is_finite());
    }

    #[test]
    fn weighted_cost_selector_fails_closed_on_invalid_prior_and_cost() {
        let hypotheses = [0_i32, 1];
        let candidates = [0_u8, 1_u8];
        let prediction =
            |hypothesis: &i32, candidate: &u8| Some(*hypothesis as f64 * *candidate as f64);
        assert!(
            select_experiment_by_weighted_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, -1.0],
                1.0,
                prediction,
                |_| 1.0,
            )
            .is_none()
        );
        assert!(
            select_experiment_by_weighted_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, 1.0],
                1.0,
                prediction,
                |_| f64::NAN,
            )
            .is_none()
        );
    }

    #[test]
    fn scenario_robust_selector_accounts_for_prediction_discrepancy_not_only_noise() {
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [0_u8, 1_u8];
        let scenarios = [
            InformationGainScenario {
                observation_std_dev: 0.1,
                prediction_model: 0,
            },
            InformationGainScenario {
                observation_std_dev: 1.0,
                prediction_model: 7,
            },
        ];
        let predict = |hypothesis: &usize, candidate: &u8, scenario: &InformationGainScenario| {
            match (*candidate, scenario.prediction_model) {
                // Candidate 0 has one outlier in both predictive worlds and remains
                // somewhat informative under the noisier scenario.
                (0, 0 | 7) => Some([0.0_f64, 0.0, 0.0, 3.0][*hypothesis]),
                // Candidate 1 wins nominally, but one defensible alternative model
                // compresses its predicted outcomes to a near-agreement point.
                (1, 0) => Some([0.0_f64, 1.0, 2.0, 3.0][*hypothesis]),
                (1, 7) => Some([0.0_f64, 0.001, -0.001, 0.002][*hypothesis]),
                _ => None,
            }
        };

        let (nominal, _) = select_experiment_by_information_gain_per_cost(
            &candidates,
            &hypotheses,
            0.1,
            |hypothesis, candidate| {
                predict(
                    hypothesis,
                    candidate,
                    &InformationGainScenario {
                        observation_std_dev: 0.1,
                        prediction_model: 0,
                    },
                )
            },
            |_| 1.0,
        )
        .expect("nominal pool is informative");
        assert_eq!(*nominal, 1);

        let (robust, robust_score) =
            select_experiment_by_scenario_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &scenarios,
                predict,
                |_| 1.0,
            )
            .expect("candidate 0 has positive information under each scenario");
        assert_eq!(*robust, 0);
        assert!(robust_score.is_finite() && robust_score > 0.0);
    }

    #[test]
    fn scenario_robust_information_gain_is_permutation_invariant_and_cannot_increase_when_scenarios_are_added()
     {
        let hypotheses = [0_usize, 1, 2, 3];
        let base = [
            InformationGainScenario {
                observation_std_dev: 0.1,
                prediction_model: 0,
            },
            InformationGainScenario {
                observation_std_dev: 0.6,
                prediction_model: 1,
            },
        ];
        let reversed = [base[1], base[0]];
        let extended = [
            base[0],
            base[1],
            InformationGainScenario {
                observation_std_dev: 0.2,
                prediction_model: 2,
            },
        ];
        let predict = |hypothesis: &usize, _: &(), scenario: &InformationGainScenario| {
            let values = match scenario.prediction_model {
                0 => [0.0_f64, 1.0, 2.0, 3.0],
                1 => [0.0_f64, 0.2, 2.0, 3.5],
                2 => [0.0_f64, 0.001, 0.002, 0.003],
                _ => return None,
            };
            Some(values[*hypothesis])
        };

        let score = expected_information_gain_over_scenarios_nats(&(), &hypotheses, &base, predict)
            .expect("base scenarios are valid");
        let reversed_score =
            expected_information_gain_over_scenarios_nats(&(), &hypotheses, &reversed, predict)
                .expect("reordered scenarios are valid");
        let extended_score =
            expected_information_gain_over_scenarios_nats(&(), &hypotheses, &extended, predict)
                .expect("extended scenarios are valid");

        assert!((score - reversed_score).abs() <= 1.0e-12);
        assert!(
            extended_score <= score + 1.0e-12,
            "adding a plausible world must not increase a minimum-over-worlds score"
        );
    }

    #[test]
    fn weighted_scenario_robust_gain_equals_the_minimum_scenario_specific_weighted_gain() {
        let hypotheses = [0_usize, 1, 2, 3];
        let prior_weights = [0.4_f64, 0.1, 0.4, 0.1];
        let scenarios = [
            InformationGainScenario {
                observation_std_dev: 0.1,
                prediction_model: 0,
            },
            InformationGainScenario {
                observation_std_dev: 0.8,
                prediction_model: 1,
            },
        ];
        let predict = |hypothesis: &usize, _: &(), scenario: &InformationGainScenario| {
            let values = match scenario.prediction_model {
                0 => [0.0_f64, 1.0, 2.0, 3.0],
                1 => [0.0_f64, 0.0, 0.1, 2.0],
                _ => return None,
            };
            Some(values[*hypothesis])
        };
        let robust = expected_information_gain_weighted_over_scenarios_nats(
            &(),
            &hypotheses,
            &prior_weights,
            &scenarios,
            predict,
        )
        .expect("all declared scenarios are valid");
        let expected = scenarios
            .iter()
            .map(|scenario| {
                expected_information_gain_weighted_nats(
                    &(),
                    &hypotheses,
                    &prior_weights,
                    scenario.observation_std_dev,
                    |hypothesis, experiment| predict(hypothesis, experiment, scenario),
                )
                .unwrap()
            })
            .fold(f64::INFINITY, f64::min);
        assert!((robust - expected).abs() <= 1.0e-12);
    }

    #[test]
    fn weighted_scenario_robust_selector_avoids_candidate_that_fails_under_alternative_model() {
        let hypotheses = [0_usize, 1, 2, 3];
        let prior_weights = [0.4_f64, 0.1, 0.4, 0.1];
        let candidates = [0_u8, 1_u8];
        let scenarios = [
            InformationGainScenario {
                observation_std_dev: 0.1,
                prediction_model: 0,
            },
            InformationGainScenario {
                observation_std_dev: 1.0,
                prediction_model: 1,
            },
        ];
        let predict = |hypothesis: &usize, candidate: &u8, scenario: &InformationGainScenario| {
            let values = match (*candidate, scenario.prediction_model) {
                (0, 0 | 1) => [0.0_f64, 0.0, 0.0, 3.0],
                (1, 0) => [0.0_f64, 1.0, 2.0, 3.0],
                (1, 1) => [0.0_f64, 0.001, -0.001, 0.002],
                _ => return None,
            };
            Some(values[*hypothesis])
        };
        let nominal = select_experiment_by_weighted_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &prior_weights,
            0.1,
            |hypothesis, candidate| {
                predict(
                    hypothesis,
                    candidate,
                    &InformationGainScenario {
                        observation_std_dev: 0.1,
                        prediction_model: 0,
                    },
                )
            },
            |_| 1.0,
        )
        .expect("nominal candidates are informative");
        assert_eq!(*nominal.0, 1);

        let robust = select_experiment_by_weighted_scenario_robust_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &prior_weights,
            &scenarios,
            predict,
            |_| 1.0,
        )
        .expect("candidate zero retains information in both declared worlds");
        assert_eq!(*robust.0, 0);
        assert!(robust.1.is_finite() && robust.1 > 0.0);
    }

    #[test]
    fn weighted_scenario_robust_selector_rejects_invalid_priors_and_scenarios() {
        let hypotheses = [0_usize, 1];
        let candidates = [0_u8, 1_u8];
        let scenario = [InformationGainScenario {
            observation_std_dev: 0.5,
            prediction_model: 0,
        }];
        let predict = |hypothesis: &usize, candidate: &u8, _: &InformationGainScenario| {
            Some(*hypothesis as f64 * *candidate as f64)
        };
        assert!(
            select_experiment_by_weighted_scenario_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, f64::NAN],
                &scenario,
                predict,
                |_| 1.0,
            )
            .is_none()
        );
        assert!(
            select_experiment_by_weighted_scenario_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, 1.0],
                &[],
                predict,
                |_| 1.0,
            )
            .is_none()
        );
        assert!(
            select_experiment_by_weighted_scenario_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, 1.0],
                &scenario,
                predict,
                |_| f64::NAN,
            )
            .is_none()
        );
    }

    #[test]
    fn prior_robust_selector_handles_prior_shift_and_scenario_order() {
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [0_u8, 1_u8];
        let scenarios = [
            PriorInformationGainScenario {
                observation_std_dev: 0.05,
                prediction_model: 0,
                prior_weights: vec![0.49, 0.49, 0.01, 0.01],
            },
            PriorInformationGainScenario {
                observation_std_dev: 0.05,
                prediction_model: 0,
                prior_weights: vec![0.01, 0.01, 0.49, 0.49],
            },
        ];
        let predict = |hypothesis: &usize, candidate: &u8, _: &PriorInformationGainScenario| {
            let values = match candidate {
                // Separates the two halves evenly under either plausible prior.
                0 => [0.0_f64, 0.0, 5.0, 5.0],
                // Excellent under the first prior; almost uninformative under the second.
                1 => [0.0_f64, 5.0, 0.0, 0.0],
                _ => return None,
            };
            Some(values[*hypothesis])
        };

        let nominal = select_experiment_by_weighted_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &scenarios[0].prior_weights,
            scenarios[0].observation_std_dev,
            |hypothesis, candidate| predict(hypothesis, candidate, &scenarios[0]),
            |_| 1.0,
        )
        .expect("first prior makes one candidate strongly informative");
        assert_eq!(*nominal.0, 1);

        let robust = select_experiment_by_prior_robust_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &scenarios,
            predict,
            |_| 1.0,
        )
        .expect("each candidate has positive gain over the complete prior envelope");
        assert_eq!(*robust.0, 0);

        let base_envelope = expected_information_gain_over_prior_scenarios_nats(
            &0_u8,
            &hypotheses,
            &scenarios,
            predict,
        )
        .expect("base prior envelope is valid");
        let mut extended = scenarios.to_vec();
        extended.push(PriorInformationGainScenario {
            observation_std_dev: 0.05,
            prediction_model: 0,
            prior_weights: vec![0.99, 0.003, 0.003, 0.004],
        });
        let extended_envelope = expected_information_gain_over_prior_scenarios_nats(
            &0_u8,
            &hypotheses,
            &extended,
            predict,
        )
        .expect("extended prior envelope is valid");
        assert!(
            extended_envelope <= base_envelope + 1.0e-12,
            "adding a prior scenario cannot increase the minimum information gain"
        );

        let reversed = [scenarios[1].clone(), scenarios[0].clone()];
        let robust_reversed = select_experiment_by_prior_robust_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &reversed,
            predict,
            |_| 1.0,
        )
        .expect("reordered finite prior set remains valid");
        assert_eq!(robust.0, robust_reversed.0);
        assert!((robust.1 - robust_reversed.1).abs() <= 1.0e-12);
    }

    #[test]
    fn prior_robust_selector_rejects_invalid_prior_scenarios() {
        let hypotheses = [0_usize, 1];
        let candidates = [0_u8, 1_u8];
        let invalid = [PriorInformationGainScenario {
            observation_std_dev: 0.5,
            prediction_model: 0,
            prior_weights: vec![1.0, f64::NAN],
        }];
        assert!(
            select_experiment_by_prior_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &invalid,
                |hypothesis, candidate, _| Some(*hypothesis as f64 * *candidate as f64),
                |_| 1.0,
            )
            .is_none()
        );
        assert!(
            expected_information_gain_over_prior_scenarios_nats(
                &0_u8,
                &hypotheses,
                &[],
                |hypothesis, candidate, _| Some(*hypothesis as f64 * *candidate as f64),
            )
            .is_none()
        );
    }

    #[test]
    fn scenario_robust_selector_fails_closed_on_invalid_scenarios() {
        let hypotheses = [0_i32, 1];
        let candidates = [0_u8, 1_u8];
        assert!(
            expected_information_gain_over_scenarios_nats(
                &0_u8,
                &hypotheses,
                &[],
                |hypothesis, candidate, _| Some(*hypothesis as f64 * *candidate as f64),
            )
            .is_none()
        );
        assert!(
            select_experiment_by_scenario_robust_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[InformationGainScenario {
                    observation_std_dev: f64::NAN,
                    prediction_model: 0,
                }],
                |hypothesis, candidate, _| Some(*hypothesis as f64 * *candidate as f64),
                |_| 1.0,
            )
            .is_none()
        );
    }

    #[test]
    fn worst_case_information_gain_is_bounded_by_every_scenario() {
        let hypotheses = [0_i32, 1, 2, 3];
        let predictions = [0.0_f64, 0.0, 0.0, 3.0];
        let scenarios = [0.1_f64, 1.0, 5.0];
        let robust = expected_information_gain_worst_case_nats(
            &(),
            &hypotheses,
            &scenarios,
            |hypothesis, _| Some(predictions[*hypothesis as usize]),
        )
        .expect("valid finite scenarios");
        for sigma in scenarios {
            let scenario_score =
                expected_information_gain_nats(&(), &hypotheses, sigma, |hypothesis, _| {
                    Some(predictions[*hypothesis as usize])
                })
                .unwrap();
            assert!(
                robust <= scenario_score + 1.0e-12,
                "worst-case score {robust} exceeded scenario score {scenario_score} at sigma={sigma}"
            );
        }
    }

    #[test]
    fn robust_selector_can_trade_nominal_information_for_noise_envelope() {
        // Candidate 1 separates every hypothesis at low noise. Candidate 0 has
        // lower low-noise entropy, but higher signal variance and therefore better
        // worst-case information under the much noisier scenario.
        let hypotheses = [0_usize, 1, 2, 3];
        let candidates = [0_u8, 1_u8];
        let predict = |hypothesis: &usize, candidate: &u8| {
            let values = match candidate {
                0 => [0.0_f64, 0.0, 0.0, 3.0],
                1 => [0.0_f64, 1.0, 2.0, 3.0],
                _ => return None,
            };
            Some(values[*hypothesis])
        };
        let (nominal, nominal_score) = select_experiment_by_information_gain_per_cost(
            &candidates,
            &hypotheses,
            0.1,
            predict,
            |_| 1.0,
        )
        .expect("a low-noise candidate is informative");
        assert_eq!(*nominal, 1);

        let (robust, robust_score) = select_experiment_by_worst_case_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &[0.1, 5.0],
            predict,
            |_| 1.0,
        )
        .expect("both candidates have positive worst-case information");
        assert_eq!(*robust, 0);
        assert!(nominal_score > 0.0 && robust_score > 0.0);

        let (cost_aware, cost_score) = select_experiment_by_worst_case_information_gain_per_cost(
            &candidates,
            &hypotheses,
            &[0.1, 5.0],
            predict,
            |candidate| if *candidate == 0 { 100.0 } else { 1.0 },
        )
        .expect("at least one candidate has positive information per cost");
        assert_eq!(*cost_aware, 1);
        assert!(cost_score.is_finite() && cost_score > 0.0);
    }

    #[test]
    fn robust_selector_fails_closed_on_invalid_noise_envelope() {
        let hypotheses = [0_i32, 1];
        let candidates = [0.0_f64, 1.0];
        let predict = |hypothesis: &i32, experiment: &f64| Some(*hypothesis as f64 * *experiment);
        assert!(
            expected_information_gain_worst_case_nats(&0.0, &hypotheses, &[], predict).is_none()
        );
        assert!(
            expected_information_gain_worst_case_nats(&0.0, &hypotheses, &[1.0, f64::NAN], predict)
                .is_none()
        );
        assert!(
            select_experiment_by_worst_case_information_gain_per_cost(
                &candidates,
                &hypotheses,
                &[1.0, 0.0],
                predict,
                |_| 1.0
            )
            .is_none()
        );
    }

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
