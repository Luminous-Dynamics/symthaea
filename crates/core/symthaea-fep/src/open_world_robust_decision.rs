// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Open-world model-adequacy monitoring and finite-TV robust diagnostics.
//!
//! This module separates uncertainty within a registered hypothesis family
//! from evidence that the predictive family itself is inadequate.
//!
//! The adequacy monitor consumes prequential categorical predictions: each
//! distribution must be frozen before its corresponding observation. A
//! zero-support observation is an immediate adequacy failure. Passing the
//! configured score thresholds means only "no detected inadequacy", never
//! proof that the model family is correct.
//!
//! The robust diagnostic selector wraps nominal EVSI with a finite categorical
//! total-variation ambiguity set around each action's predictive outcome
//! distribution. It exactly maximizes expected posterior loss over the finite
//! outcome TV ball. Outcomes assigned zero nominal probability have no
//! in-family posterior; they receive the maximum declared task loss, a
//! conservative open-support penalty.
//!
//! This is a transparent finite-outcome baseline. It is not an implementation
//! of MMD/RoBAS, a general distributionally robust POMDP solver, or an
//! action-execution adapter.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionDiagnosticError, DecisionHypothesis, DiagnosticAction,
    evaluate_decision_focused_diagnostics,
};

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const RISK_TOLERANCE: f64 = 1e-10;

/// One prequential categorical prediction and its subsequently observed outcome.
///
/// The prediction must be generated from evidence available before the
/// observation. This type cannot prove temporal ordering; callers must bind
/// records to source, task, and observation digests in their surrounding receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialCategoricalObservation {
    pub predicted_outcome_probabilities: Vec<f64>,
    pub observed_outcome_index: usize,
}

/// Externally calibrated thresholds for a declared task and observation stream.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelAdequacyThresholds {
    pub minimum_observations: usize,
    /// Positive floor used to flag very-low-probability observations.
    pub minimum_predictive_probability: f64,
    pub maximum_mean_log_loss: f64,
    /// Multiclass Brier score is the sum over categories and lies in [0, 2].
    pub maximum_mean_brier_score: f64,
    pub maximum_low_probability_rate: f64,
}

/// Model-family adequacy state. No state asserts that a model is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAdequacyStatus {
    /// Not enough prequential observations to evaluate the frozen thresholds.
    InsufficientEvidence,
    /// The supplied thresholds detected no failure; adequacy is not proven.
    NoDetectedInadequacy,
    /// A support violation or a preregistered score threshold was crossed.
    Inadequate,
}

/// Prequential predictive adequacy report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAdequacyReport {
    pub status: ModelAdequacyStatus,
    pub observation_count: usize,
    pub zero_support_count: usize,
    pub low_probability_count: usize,
    pub low_probability_rate: f64,
    /// None when any observation had zero predictive probability.
    pub mean_log_loss: Option<f64>,
    pub mean_multiclass_brier_score: Option<f64>,
    pub minimum_observed_probability: Option<f64>,
    pub thresholds: ModelAdequacyThresholds,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ModelAdequacyError {
    #[error("model-adequacy thresholds are invalid")]
    InvalidThresholds,
    #[error("observation {observation_index} has no outcome categories")]
    EmptyOutcomeSet { observation_index: usize },
    #[error(
        "observation {observation_index} has an invalid probability at category {category_index}"
    )]
    InvalidProbability {
        observation_index: usize,
        category_index: usize,
    },
    #[error("observation {observation_index} probability mass sums to {sum}, expected 1")]
    ProbabilityMassNotNormalized { observation_index: usize, sum: f64 },
    #[error(
        "observation {observation_index} has outcome index {outcome_index} outside 0..{outcome_count}"
    )]
    OutcomeIndexOutOfRange {
        observation_index: usize,
        outcome_index: usize,
        outcome_count: usize,
    },
}

/// Evaluate frozen, pre-observation predictions against subsequent outcomes.
///
/// A zero-support observation is never repaired by adding epsilon or silently
/// renormalizing the registered family. Its log score is None, while the
/// support-violation count forces the Inadequate status.
pub fn evaluate_prequential_model_adequacy(
    observations: &[PrequentialCategoricalObservation],
    thresholds: ModelAdequacyThresholds,
) -> Result<ModelAdequacyReport, ModelAdequacyError> {
    validate_adequacy_thresholds(thresholds)?;

    let mut zero_support_count = 0;
    let mut low_probability_count = 0;
    let mut log_loss_sum = 0.0;
    let mut brier_sum = 0.0;
    let mut minimum_observed_probability: Option<f64> = None;

    for (observation_index, observation) in observations.iter().enumerate() {
        let probabilities = &observation.predicted_outcome_probabilities;
        if probabilities.is_empty() {
            return Err(ModelAdequacyError::EmptyOutcomeSet { observation_index });
        }
        let mut probability_sum = 0.0;
        for (category_index, probability) in probabilities.iter().copied().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(ModelAdequacyError::InvalidProbability {
                    observation_index,
                    category_index,
                });
            }
            probability_sum += probability;
        }
        if (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(ModelAdequacyError::ProbabilityMassNotNormalized {
                observation_index,
                sum: probability_sum,
            });
        }
        if observation.observed_outcome_index >= probabilities.len() {
            return Err(ModelAdequacyError::OutcomeIndexOutOfRange {
                observation_index,
                outcome_index: observation.observed_outcome_index,
                outcome_count: probabilities.len(),
            });
        }

        let observed_probability = probabilities[observation.observed_outcome_index];
        minimum_observed_probability = Some(
            minimum_observed_probability.map_or(observed_probability, |minimum| {
                minimum.min(observed_probability)
            }),
        );
        if observed_probability == 0.0 {
            zero_support_count += 1;
        } else {
            log_loss_sum -= observed_probability.ln();
        }
        if observed_probability < thresholds.minimum_predictive_probability {
            low_probability_count += 1;
        }

        let brier_score = probabilities
            .iter()
            .enumerate()
            .map(|(category_index, probability)| {
                let target = if category_index == observation.observed_outcome_index {
                    1.0
                } else {
                    0.0
                };
                (probability - target).powi(2)
            })
            .sum::<f64>();
        brier_sum += brier_score;
    }

    let observation_count = observations.len();
    let low_probability_rate = if observation_count == 0 {
        0.0
    } else {
        low_probability_count as f64 / observation_count as f64
    };
    let mean_log_loss = if observation_count == 0 || zero_support_count > 0 {
        None
    } else {
        Some(log_loss_sum / observation_count as f64)
    };
    let mean_multiclass_brier_score = if observation_count == 0 {
        None
    } else {
        Some(brier_sum / observation_count as f64)
    };

    let status = if zero_support_count > 0 {
        ModelAdequacyStatus::Inadequate
    } else if observation_count < thresholds.minimum_observations {
        ModelAdequacyStatus::InsufficientEvidence
    } else if mean_log_loss.is_some_and(|score| score > thresholds.maximum_mean_log_loss)
        || mean_multiclass_brier_score
            .is_some_and(|score| score > thresholds.maximum_mean_brier_score)
        || low_probability_rate > thresholds.maximum_low_probability_rate
    {
        ModelAdequacyStatus::Inadequate
    } else {
        ModelAdequacyStatus::NoDetectedInadequacy
    };

    Ok(ModelAdequacyReport {
        status,
        observation_count,
        zero_support_count,
        low_probability_count,
        low_probability_rate,
        mean_log_loss,
        mean_multiclass_brier_score,
        minimum_observed_probability,
        thresholds,
    })
}

/// What the open-world control plane permits after model-adequacy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenWorldGateDisposition {
    /// Collect more prequential evidence before generating parameter edits.
    EvidenceCollectionOnly,
    /// Stop automatic repair proposals and revise/expand the predictive family.
    ModelRevisionRequired,
    /// Detached candidate generation/evaluation is permitted, but not acceptance.
    DetachedCandidateEvaluationPermitted,
}

/// Explicit authority boundary derived from model adequacy and sequential evidence.
///
/// This gate never authorizes candidate acceptance or live-model mutation.
/// A status of NoDetectedInadequacy is not proof that the model family is correct;
/// it only permits detached candidate evaluation subject to the normal SWI and
/// independent-verifier contracts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenWorldDecisionGateReport {
    pub declared_adequacy_status: ModelAdequacyStatus,
    pub effective_adequacy_status: ModelAdequacyStatus,
    /// First one-based observation at which a valid prequential e-process crossed.
    pub prequential_evidence_crossed_at: Option<usize>,
    pub disposition: OpenWorldGateDisposition,
    /// Selection/evaluation of explicitly safe diagnostic probes remains possible.
    /// This is not permission to execute a probe; external safety checks still apply.
    pub diagnostic_probe_selection_permitted: bool,
    pub weight_program_generation_permitted: bool,
    pub detached_candidate_evaluation_permitted: bool,
    pub independent_verifier_required: bool,
    /// Always false: only the independent verifier can accept a candidate.
    pub candidate_acceptance_authorized: bool,
    /// Always false: this research gate cannot mutate a live model.
    pub live_model_mutation_authorized: bool,
}

/// Combine proper-score/support adequacy with optional anytime-valid evidence.
///
/// A prequential e-process crossing overrides an aggregate-score state of
/// NoDetectedInadequacy. Insufficient evidence and detected inadequacy both
/// prevent WeightProgram generation; diagnostics may still be considered by
/// a separate cost/safety-constrained probe policy.
pub fn evaluate_open_world_decision_gate(
    adequacy: &ModelAdequacyReport,
    prequential_evidence_crossed_at: Option<usize>,
) -> OpenWorldDecisionGateReport {
    let effective_adequacy_status = if adequacy.status == ModelAdequacyStatus::Inadequate
        || prequential_evidence_crossed_at.is_some()
    {
        ModelAdequacyStatus::Inadequate
    } else {
        adequacy.status
    };
    let disposition = match effective_adequacy_status {
        ModelAdequacyStatus::InsufficientEvidence => {
            OpenWorldGateDisposition::EvidenceCollectionOnly
        }
        ModelAdequacyStatus::NoDetectedInadequacy => {
            OpenWorldGateDisposition::DetachedCandidateEvaluationPermitted
        }
        ModelAdequacyStatus::Inadequate => OpenWorldGateDisposition::ModelRevisionRequired,
    };
    let candidate_work_permitted = disposition
        == OpenWorldGateDisposition::DetachedCandidateEvaluationPermitted;

    OpenWorldDecisionGateReport {
        declared_adequacy_status: adequacy.status,
        effective_adequacy_status,
        prequential_evidence_crossed_at,
        disposition,
        diagnostic_probe_selection_permitted: true,
        weight_program_generation_permitted: candidate_work_permitted,
        detached_candidate_evaluation_permitted: candidate_work_permitted,
        independent_verifier_required: true,
        candidate_acceptance_authorized: false,
        live_model_mutation_authorized: false,
    }
}

fn validate_adequacy_thresholds(
    thresholds: ModelAdequacyThresholds,
) -> Result<(), ModelAdequacyError> {
    if thresholds.minimum_observations == 0
        || !thresholds.minimum_predictive_probability.is_finite()
        || !(0.0..=1.0).contains(&thresholds.minimum_predictive_probability)
        || thresholds.minimum_predictive_probability == 0.0
        || !thresholds.maximum_mean_log_loss.is_finite()
        || thresholds.maximum_mean_log_loss < 0.0
        || !thresholds.maximum_mean_brier_score.is_finite()
        || !(0.0..=2.0).contains(&thresholds.maximum_mean_brier_score)
        || !thresholds.maximum_low_probability_rate.is_finite()
        || !(0.0..=1.0).contains(&thresholds.maximum_low_probability_rate)
    {
        return Err(ModelAdequacyError::InvalidThresholds);
    }
    Ok(())
}

/// Finite categorical total-variation robust diagnostic-selection result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TvRobustDecisionEvaluation {
    pub total_variation_radius: f64,
    pub no_probe_bayes_risk: f64,
    pub nominal_selected_action_index: Option<usize>,
    pub robust_selected_action_index: Option<usize>,
    pub actions: Vec<TvRobustActionEvaluation>,
}

/// Nominal and worst-case action values under a declared TV ambiguity set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TvRobustActionEvaluation {
    pub action_index: usize,
    pub action_id: String,
    pub eligible: bool,
    pub nominal_expected_posterior_bayes_risk: f64,
    pub worst_case_expected_posterior_bayes_risk: f64,
    pub nominal_total_score: Option<f64>,
    pub worst_case_total_score: Option<f64>,
    pub nominal_expected_value_of_sample_information: f64,
    pub worst_case_expected_value_of_sample_information: f64,
    pub nominal_model_information_gain_nats: f64,
    pub nominal_decision_information_gain_nats: f64,
    /// Adversarial outcome probabilities attaining the worst-case TV risk.
    pub worst_case_outcome_probabilities: Vec<f64>,
    /// Outcomes with zero nominal support use the maximum declared task loss.
    pub unsupported_outcome_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TvRobustDecisionError {
    #[error(transparent)]
    DecisionDiagnostic(#[from] DecisionDiagnosticError),
    #[error("total-variation ambiguity radius must be finite and in [0, 1]")]
    InvalidTotalVariationRadius,
    #[error("probability and loss vectors must have the same non-zero length")]
    VectorLengthMismatch,
    #[error("probability at index {index} is invalid")]
    InvalidProbability { index: usize },
    #[error("probability mass sums to {sum}, expected 1")]
    ProbabilityMassNotNormalized { sum: f64 },
    #[error("loss at index {index} is non-finite")]
    InvalidLoss { index: usize },
    #[error("robust expected risk is non-finite")]
    NonFiniteRisk,
}

/// Evaluate decision-focused diagnostics against a total-variation ball around
/// each nominal predictive outcome distribution.
///
/// For each action a, the ambiguity set is Q_a(rho) = {q : TV(q, p_a) <= rho}.
/// The exact worst-case expectation of outcome-specific posterior Bayes risk
/// is computed by transferring up to rho probability mass from the least
/// costly outcomes to the most costly outcomes. Zero-support outcomes receive
/// the maximum declared task loss as a conservative bound.
///
/// The no-probe option retains nominal prior Bayes risk. Thus the robust
/// selector protects against outcome-distribution perturbation conditional on
/// the supplied model prior and posterior update rule; it does not robustify
/// those model priors or prove coverage outside the declared TV radius.
pub fn evaluate_tv_robust_decision_diagnostics(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    total_variation_radius: f64,
) -> Result<TvRobustDecisionEvaluation, TvRobustDecisionError> {
    if !total_variation_radius.is_finite() || !(0.0..=1.0).contains(&total_variation_radius) {
        return Err(TvRobustDecisionError::InvalidTotalVariationRadius);
    }
    let nominal = evaluate_decision_focused_diagnostics(hypotheses, actions, decision_config)?;
    let maximum_declared_loss = hypotheses
        .iter()
        .flat_map(|hypothesis| hypothesis.loss_by_final_decision.iter().copied())
        .fold(0.0_f64, f64::max);
    let mut robust_selected_action_index = None;
    let mut best_robust_score = nominal.prior_bayes_risk;
    let mut robust_actions = Vec::with_capacity(nominal.actions.len());

    for action in &nominal.actions {
        let mut outcome_losses = Vec::with_capacity(action.posterior_bayes_risk_by_outcome.len());
        let mut unsupported_outcome_indices = Vec::new();
        for (outcome_index, posterior_risk) in
            action.posterior_bayes_risk_by_outcome.iter().enumerate()
        {
            match posterior_risk {
                Some(risk) => outcome_losses.push(*risk),
                None => {
                    outcome_losses.push(maximum_declared_loss);
                    unsupported_outcome_indices.push(outcome_index);
                }
            }
        }
        let worst_case_distribution = worst_case_distribution_within_tv(
            &action.outcome_probabilities,
            &outcome_losses,
            total_variation_radius,
        )?;
        let worst_case_expected_posterior_bayes_risk = worst_case_distribution
            .iter()
            .zip(&outcome_losses)
            .map(|(probability, risk)| probability * risk)
            .sum::<f64>();
        if !worst_case_expected_posterior_bayes_risk.is_finite() {
            return Err(TvRobustDecisionError::NonFiniteRisk);
        }
        let nominal_expected_value_of_sample_information =
            nominal.prior_bayes_risk - action.expected_posterior_bayes_risk;
        let worst_case_expected_value_of_sample_information =
            nominal.prior_bayes_risk - worst_case_expected_posterior_bayes_risk;

        let worst_case_total_score = if action.eligible {
            Some(
                worst_case_expected_posterior_bayes_risk
                    + decision_config.observation_cost_weight * action.observation_cost
                    + decision_config.safety_risk_weight * action.safety_risk,
            )
        } else {
            None
        };
        if let Some(score) = worst_case_total_score
            && score + RISK_TOLERANCE < best_robust_score
        {
            best_robust_score = score;
            robust_selected_action_index = Some(action.action_index);
        }

        robust_actions.push(TvRobustActionEvaluation {
            action_index: action.action_index,
            action_id: action.action_id.clone(),
            eligible: action.eligible,
            nominal_expected_posterior_bayes_risk: action.expected_posterior_bayes_risk,
            worst_case_expected_posterior_bayes_risk,
            nominal_total_score: action.total_score,
            worst_case_total_score,
            nominal_expected_value_of_sample_information,
            worst_case_expected_value_of_sample_information,
            nominal_model_information_gain_nats: action.model_information_gain_nats,
            nominal_decision_information_gain_nats: action.decision_information_gain_nats,
            worst_case_outcome_probabilities: worst_case_distribution,
            unsupported_outcome_indices,
        });
    }

    Ok(TvRobustDecisionEvaluation {
        total_variation_radius,
        no_probe_bayes_risk: nominal.prior_bayes_risk,
        nominal_selected_action_index: nominal.selected_action_index,
        robust_selected_action_index,
        actions: robust_actions,
    })
}

/// Finite-sample calibration of a categorical total-variation radius.
///
/// The bound uses Hoeffding's inequality for each of K categories and a
/// union bound across categories. With n independent calibration observations,
/// let epsilon = sqrt(ln(2K/alpha) / (2n)); then, with probability at least
/// 1-alpha, every coordinate differs from its true probability by at most
/// epsilon, implying TV(p_hat, p_true) <= min(1, K*epsilon/2).
///
/// This is deliberately a simple distribution-free reference, not a tight
/// multinomial confidence region. Calibration samples must be independent of
/// the episodes used to evaluate the policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TvAmbiguityCalibration {
    pub category_count: usize,
    pub sample_count: u64,
    pub alpha: f64,
    pub confidence_level: f64,
    pub empirical_probabilities: Vec<f64>,
    pub per_category_absolute_error_bound: f64,
    pub total_variation_radius: f64,
    pub method: String,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TvAmbiguityCalibrationError {
    #[error("at least one categorical outcome is required")]
    EmptyCategorySet,
    #[error("calibration sample count must be positive")]
    EmptyCalibrationSample,
    #[error("calibration count sum overflowed u64")]
    CountOverflow,
    #[error("alpha must be finite and in (0, 1)")]
    InvalidAlpha,
    #[error("calibration radius calculation became non-finite")]
    NonFiniteBound,
}

/// Calibrate a finite categorical TV ambiguity radius from independent counts.
///
/// The returned radius satisfies the stated Hoeffding/union-bound coverage
/// under i.i.d. categorical calibration draws. It should not be treated as a
/// valid coverage radius under temporal dependence, covariate shift between
/// calibration and deployment, adaptive reuse of the same observations, or
/// misspecified category definitions.
pub fn calibrate_categorical_tv_radius_from_counts(
    category_counts: &[u64],
    alpha: f64,
) -> Result<TvAmbiguityCalibration, TvAmbiguityCalibrationError> {
    if category_counts.is_empty() {
        return Err(TvAmbiguityCalibrationError::EmptyCategorySet);
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(TvAmbiguityCalibrationError::InvalidAlpha);
    }
    let sample_count = category_counts
        .iter()
        .try_fold(0_u64, |total, count| total.checked_add(*count))
        .ok_or(TvAmbiguityCalibrationError::CountOverflow)?;
    if sample_count == 0 {
        return Err(TvAmbiguityCalibrationError::EmptyCalibrationSample);
    }

    let empirical_probabilities = category_counts
        .iter()
        .map(|count| *count as f64 / sample_count as f64)
        .collect::<Vec<_>>();
    let category_count = category_counts.len();
    let per_category_absolute_error_bound = if category_count == 1 {
        0.0
    } else {
        (((2.0 * category_count as f64).ln() - alpha.ln()) / (2.0 * sample_count as f64)).sqrt()
    };
    let total_variation_radius =
        (0.5 * category_count as f64 * per_category_absolute_error_bound).min(1.0);
    if !per_category_absolute_error_bound.is_finite() || !total_variation_radius.is_finite() {
        return Err(TvAmbiguityCalibrationError::NonFiniteBound);
    }

    Ok(TvAmbiguityCalibration {
        category_count,
        sample_count,
        alpha,
        confidence_level: 1.0 - alpha,
        empirical_probabilities,
        per_category_absolute_error_bound,
        total_variation_radius,
        method: "categorical_hoeffding_union_bound_tv_radius_v1".to_owned(),
    })
}

/// Return an adversarial distribution q maximizing E_q[loss] subject to
/// TV(q, p) <= radius. This is exact for a finite categorical outcome space.
pub fn worst_case_distribution_within_tv(
    nominal_probabilities: &[f64],
    outcome_losses: &[f64],
    radius: f64,
) -> Result<Vec<f64>, TvRobustDecisionError> {
    if !radius.is_finite() || !(0.0..=1.0).contains(&radius) {
        return Err(TvRobustDecisionError::InvalidTotalVariationRadius);
    }
    if nominal_probabilities.is_empty() || nominal_probabilities.len() != outcome_losses.len() {
        return Err(TvRobustDecisionError::VectorLengthMismatch);
    }
    let mut probability_mass = 0.0;
    for (index, probability) in nominal_probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(TvRobustDecisionError::InvalidProbability { index });
        }
        probability_mass += probability;
    }
    if (probability_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(TvRobustDecisionError::ProbabilityMassNotNormalized {
            sum: probability_mass,
        });
    }
    for (index, loss) in outcome_losses.iter().copied().enumerate() {
        if !loss.is_finite() {
            return Err(TvRobustDecisionError::InvalidLoss { index });
        }
    }

    let mut worst_case = nominal_probabilities.to_vec();
    let mut donors: Vec<usize> = (0..outcome_losses.len()).collect();
    let mut recipients = donors.clone();
    donors.sort_by(|left, right| {
        outcome_losses[*left]
            .total_cmp(&outcome_losses[*right])
            .then_with(|| left.cmp(right))
    });
    recipients.sort_by(|left, right| {
        outcome_losses[*right]
            .total_cmp(&outcome_losses[*left])
            .then_with(|| left.cmp(right))
    });

    let mut donor_index = 0;
    let mut recipient_index = 0;
    let mut remaining_budget = radius;
    while remaining_budget > PROBABILITY_TOLERANCE {
        while donor_index < donors.len() && worst_case[donors[donor_index]] <= PROBABILITY_TOLERANCE
        {
            donor_index += 1;
        }
        while recipient_index < recipients.len()
            && worst_case[recipients[recipient_index]] >= 1.0 - PROBABILITY_TOLERANCE
        {
            recipient_index += 1;
        }
        if donor_index >= donors.len() || recipient_index >= recipients.len() {
            break;
        }
        let donor = donors[donor_index];
        let recipient = recipients[recipient_index];
        if donor == recipient || outcome_losses[recipient] <= outcome_losses[donor] {
            break;
        }
        let transferable = remaining_budget
            .min(worst_case[donor])
            .min(1.0 - worst_case[recipient]);
        if transferable <= PROBABILITY_TOLERANCE {
            break;
        }
        worst_case[donor] -= transferable;
        worst_case[recipient] += transferable;
        remaining_budget -= transferable;
    }

    let final_mass = worst_case.iter().sum::<f64>();
    if worst_case
        .iter()
        .any(|probability| !probability.is_finite())
        || (final_mass - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(TvRobustDecisionError::NonFiniteRisk);
    }
    Ok(worst_case)
}

/// A preregistered alternative predictive model for sequential adequacy testing.
///
/// Its forecast at time t must be measurable with respect to information
/// available before observing outcome t. Alternatives must not be fitted to
/// the evaluation outcomes they are used to test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialAlternativeModel {
    pub id: String,
    pub prior_weight: f64,
    pub predicted_outcome_probabilities: Vec<Vec<f64>>,
}

/// A likelihood-ratio component in the fixed alternative mixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlternativeEProcessComponent {
    pub id: String,
    pub prior_weight: f64,
    /// None means this alternative assigned zero probability to an observed outcome.
    pub log_likelihood_ratio: Option<f64>,
}

/// Anytime-valid evidence report against a declared prequential null model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialEProcessReport {
    pub alpha: f64,
    pub log_e_threshold: f64,
    pub log_e_value: Option<f64>,
    pub log_e_value_by_observation: Vec<Option<f64>>,
    /// First one-based observation index at which the e-value crossed 1 / alpha.
    pub crossed_at_observation: Option<usize>,
    pub alternatives: Vec<AlternativeEProcessComponent>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum PrequentialEProcessError {
    #[error("alpha must be finite and in (0, 1)")]
    InvalidAlpha,
    #[error("at least one alternative predictive model is required")]
    EmptyAlternativeSet,
    #[error("alternative {alternative_index} has an empty identifier")]
    EmptyAlternativeId { alternative_index: usize },
    #[error("alternative {alternative_index} has an invalid prior weight")]
    InvalidAlternativeWeight { alternative_index: usize },
    #[error("alternative prior weights sum to {sum}, expected 1")]
    AlternativeWeightsNotNormalized { sum: f64 },
    #[error("alternative {alternative_index} has {actual} forecasts; expected {expected}")]
    AlternativeObservationCountMismatch {
        alternative_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "alternative {alternative_index}, observation {observation_index} has {actual} categories; expected {expected}"
    )]
    AlternativeCategoryCountMismatch {
        alternative_index: usize,
        observation_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "alternative {alternative_index}, observation {observation_index}, category {category_index} has invalid probability"
    )]
    InvalidAlternativeProbability {
        alternative_index: usize,
        observation_index: usize,
        category_index: usize,
    },
    #[error(
        "alternative {alternative_index}, observation {observation_index} probability mass sums to {sum}, expected 1"
    )]
    AlternativeProbabilityMassNotNormalized {
        alternative_index: usize,
        observation_index: usize,
        sum: f64,
    },
    #[error("null observation {observation_index} has no outcome categories")]
    EmptyNullOutcomeSet { observation_index: usize },
    #[error(
        "null observation {observation_index} has an invalid probability at category {category_index}"
    )]
    InvalidNullProbability {
        observation_index: usize,
        category_index: usize,
    },
    #[error("null observation {observation_index} probability mass sums to {sum}, expected 1")]
    NullProbabilityMassNotNormalized { observation_index: usize, sum: f64 },
    #[error(
        "null observation {observation_index} has outcome index {outcome_index} outside 0..{outcome_count}"
    )]
    NullOutcomeIndexOutOfRange {
        observation_index: usize,
        outcome_index: usize,
        outcome_count: usize,
    },
    #[error("null observation {observation_index} has zero probability under the declared null")]
    NullSupportViolation { observation_index: usize },
    #[error("sequential log-likelihood ratio became non-finite")]
    NonFiniteLogLikelihoodRatio,
    #[error("Dirichlet alternative concentration is empty")]
    EmptyDirichletConcentration,
    #[error("Dirichlet alternative concentration is invalid at category {category_index}")]
    InvalidDirichletConcentration { category_index: usize },
    #[error("Dirichlet concentration sum is non-finite")]
    NonFiniteDirichletConcentrationMass,
    #[error("null observation {observation_index} has {actual} categories; expected {expected}")]
    DirichletCategoryCountMismatch {
        observation_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("prequential observation count overflowed the supported counter")]
    ObservationCountOverflow,
}

/// Compute a fixed mixture of likelihood-ratio e-processes.
///
/// Under the null, each null predictive distribution must be correctly
/// specified conditional on the past. Each alternative forecast must also be
/// predictable (fixed before the corresponding observation). For alternative
/// j, E_t^j = product_s q_s^j(y_s) / p_s(y_s) is a non-negative martingale
/// under the null; a fixed convex mixture remains an e-process. Therefore
/// P(sup_t E_t >= 1 / alpha) <= alpha by Ville's inequality.
///
/// This guarantee is conditional on the declared null predictive model. The
/// finite alternative set affects power, not validity under that null. A
/// zero-probability null observation is returned as a support violation and
/// must be handled by the model-adequacy gate rather than epsilon-smoothed.
pub fn evaluate_prequential_e_process(
    null_observations: &[PrequentialCategoricalObservation],
    alternatives: &[PrequentialAlternativeModel],
    alpha: f64,
) -> Result<PrequentialEProcessReport, PrequentialEProcessError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(PrequentialEProcessError::InvalidAlpha);
    }
    if alternatives.is_empty() {
        return Err(PrequentialEProcessError::EmptyAlternativeSet);
    }

    let mut ids = std::collections::BTreeSet::new();
    let mut weight_sum = 0.0;
    for (alternative_index, alternative) in alternatives.iter().enumerate() {
        if alternative.id.trim().is_empty() || !ids.insert(alternative.id.clone()) {
            return Err(PrequentialEProcessError::EmptyAlternativeId { alternative_index });
        }
        if !alternative.prior_weight.is_finite() || alternative.prior_weight <= 0.0 {
            return Err(PrequentialEProcessError::InvalidAlternativeWeight { alternative_index });
        }
        weight_sum += alternative.prior_weight;
        if alternative.predicted_outcome_probabilities.len() != null_observations.len() {
            return Err(
                PrequentialEProcessError::AlternativeObservationCountMismatch {
                    alternative_index,
                    expected: null_observations.len(),
                    actual: alternative.predicted_outcome_probabilities.len(),
                },
            );
        }
    }
    if (weight_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(PrequentialEProcessError::AlternativeWeightsNotNormalized { sum: weight_sum });
    }

    let mut log_likelihood_ratios = vec![Some(0.0_f64); alternatives.len()];
    let mut log_e_value_by_observation = Vec::with_capacity(null_observations.len());
    let mut crossed_at_observation = None;
    let log_e_threshold = -alpha.ln();

    for (observation_index, null_observation) in null_observations.iter().enumerate() {
        let null_probabilities = &null_observation.predicted_outcome_probabilities;
        if null_probabilities.is_empty() {
            return Err(PrequentialEProcessError::EmptyNullOutcomeSet { observation_index });
        }
        validate_categorical_distribution(null_probabilities).map_err(
            |(category_index, sum)| {
                if let Some(category_index) = category_index {
                    PrequentialEProcessError::InvalidNullProbability {
                        observation_index,
                        category_index,
                    }
                } else {
                    PrequentialEProcessError::NullProbabilityMassNotNormalized {
                        observation_index,
                        sum,
                    }
                }
            },
        )?;
        if null_observation.observed_outcome_index >= null_probabilities.len() {
            return Err(PrequentialEProcessError::NullOutcomeIndexOutOfRange {
                observation_index,
                outcome_index: null_observation.observed_outcome_index,
                outcome_count: null_probabilities.len(),
            });
        }
        let null_probability = null_probabilities[null_observation.observed_outcome_index];
        if null_probability == 0.0 {
            return Err(PrequentialEProcessError::NullSupportViolation { observation_index });
        }

        for (alternative_index, alternative) in alternatives.iter().enumerate() {
            let probabilities = &alternative.predicted_outcome_probabilities[observation_index];
            if probabilities.len() != null_probabilities.len() {
                return Err(PrequentialEProcessError::AlternativeCategoryCountMismatch {
                    alternative_index,
                    observation_index,
                    expected: null_probabilities.len(),
                    actual: probabilities.len(),
                });
            }
            let sum = probabilities.iter().sum::<f64>();
            for (category_index, probability) in probabilities.iter().copied().enumerate() {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(PrequentialEProcessError::InvalidAlternativeProbability {
                        alternative_index,
                        observation_index,
                        category_index,
                    });
                }
            }
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(
                    PrequentialEProcessError::AlternativeProbabilityMassNotNormalized {
                        alternative_index,
                        observation_index,
                        sum,
                    },
                );
            }

            if let Some(current_log_ratio) = log_likelihood_ratios[alternative_index] {
                let alternative_probability =
                    probabilities[null_observation.observed_outcome_index];
                if alternative_probability == 0.0 {
                    log_likelihood_ratios[alternative_index] = None;
                } else {
                    let updated =
                        current_log_ratio + alternative_probability.ln() - null_probability.ln();
                    if !updated.is_finite() {
                        return Err(PrequentialEProcessError::NonFiniteLogLikelihoodRatio);
                    }
                    log_likelihood_ratios[alternative_index] = Some(updated);
                }
            }
        }

        let log_e_value = mixture_log_e_value(alternatives, &log_likelihood_ratios);
        if crossed_at_observation.is_none()
            && log_e_value.is_some_and(|value| value >= log_e_threshold)
        {
            crossed_at_observation = Some(observation_index + 1);
        }
        log_e_value_by_observation.push(log_e_value);
    }

    let normalized_weight_sum = alternatives
        .iter()
        .map(|alternative| alternative.prior_weight)
        .sum::<f64>();
    let components = alternatives
        .iter()
        .zip(log_likelihood_ratios)
        .map(
            |(alternative, log_likelihood_ratio)| AlternativeEProcessComponent {
                id: alternative.id.clone(),
                prior_weight: alternative.prior_weight / normalized_weight_sum,
                log_likelihood_ratio,
            },
        )
        .collect();
    Ok(PrequentialEProcessReport {
        alpha,
        log_e_threshold,
        log_e_value: log_e_value_by_observation.last().copied().flatten(),
        log_e_value_by_observation,
        crossed_at_observation,
        alternatives: components,
    })
}

/// Anytime-valid e-process against a declared prequential null using a
/// Dirichlet-multinomial universal categorical alternative.
///
/// Before observing time t, the alternative predictive probability is
///
/// q_t(k) = (alpha_k + N_{t-1,k}) / (sum_j alpha_j + t - 1),
///
/// where N_{t-1,k} counts earlier outcomes in category k. Because q_t is
/// predictable, the likelihood ratio product_t q_t(y_t) / p_t(y_t) is a
/// non-negative martingale under any correctly specified prequential null
/// p_t = P(Y_t | history), even when p_t changes over time. Ville's inequality
/// therefore controls the probability of ever crossing 1/alpha.
///
/// This universal alternative does not require a finite hand-authored list of
/// misspecification forecasts. Its power is strongest against persistent
/// categorical frequency shifts; context-specific or rapidly changing
/// alternatives should use separately preregistered predictable specialists.
/// A zero-probability null observation remains a hard support violation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletPrequentialEProcessReport {
    pub alpha: f64,
    pub log_e_threshold: f64,
    pub log_e_value: f64,
    pub log_e_value_by_observation: Vec<f64>,
    pub log_likelihood_ratio_increment_by_observation: Vec<f64>,
    /// First one-based observation index at which the e-value crossed 1 / alpha.
    pub crossed_at_observation: Option<usize>,
    pub prior_concentration: Vec<f64>,
    pub final_counts: Vec<u64>,
    /// Posterior-predictive categorical distribution after all supplied data.
    pub final_alternative_predictive_probabilities: Vec<f64>,
}

/// Evaluate a Dirichlet-multinomial alternative against frozen pre-observation
/// null forecasts. The supplied predictions must be in a stable category order.
pub fn evaluate_dirichlet_prequential_e_process(
    null_observations: &[PrequentialCategoricalObservation],
    prior_concentration: &[f64],
    alpha: f64,
) -> Result<DirichletPrequentialEProcessReport, PrequentialEProcessError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(PrequentialEProcessError::InvalidAlpha);
    }
    if prior_concentration.is_empty() {
        return Err(PrequentialEProcessError::EmptyDirichletConcentration);
    }

    let mut concentration_mass = 0.0;
    for (category_index, concentration) in prior_concentration.iter().copied().enumerate() {
        if !concentration.is_finite() || concentration <= 0.0 {
            return Err(PrequentialEProcessError::InvalidDirichletConcentration {
                category_index,
            });
        }
        concentration_mass += concentration;
    }
    if !concentration_mass.is_finite() || concentration_mass <= 0.0 {
        return Err(PrequentialEProcessError::NonFiniteDirichletConcentrationMass);
    }

    let category_count = prior_concentration.len();
    let mut counts = vec![0_u64; category_count];
    let mut observation_count = 0_u64;
    let mut log_e_value = 0.0_f64;
    let log_e_threshold = -alpha.ln();
    let mut crossed_at_observation = None;
    let mut log_e_value_by_observation = Vec::with_capacity(null_observations.len());
    let mut log_likelihood_ratio_increment_by_observation =
        Vec::with_capacity(null_observations.len());

    for (observation_index, observation) in null_observations.iter().enumerate() {
        let null_probabilities = &observation.predicted_outcome_probabilities;
        if null_probabilities.is_empty() {
            return Err(PrequentialEProcessError::EmptyNullOutcomeSet { observation_index });
        }
        if null_probabilities.len() != category_count {
            return Err(
                PrequentialEProcessError::DirichletCategoryCountMismatch {
                    observation_index,
                    expected: category_count,
                    actual: null_probabilities.len(),
                },
            );
        }
        validate_categorical_distribution(null_probabilities).map_err(
            |(category_index, sum)| {
                if let Some(category_index) = category_index {
                    PrequentialEProcessError::InvalidNullProbability {
                        observation_index,
                        category_index,
                    }
                } else {
                    PrequentialEProcessError::NullProbabilityMassNotNormalized {
                        observation_index,
                        sum,
                    }
                }
            },
        )?;
        let outcome_index = observation.observed_outcome_index;
        if outcome_index >= category_count {
            return Err(PrequentialEProcessError::NullOutcomeIndexOutOfRange {
                observation_index,
                outcome_index,
                outcome_count: category_count,
            });
        }
        let null_probability = null_probabilities[outcome_index];
        if null_probability == 0.0 {
            return Err(PrequentialEProcessError::NullSupportViolation { observation_index });
        }

        let denominator = concentration_mass + observation_count as f64;
        let alternative_probability =
            (prior_concentration[outcome_index] + counts[outcome_index] as f64) / denominator;
        if !alternative_probability.is_finite() || alternative_probability <= 0.0 {
            return Err(PrequentialEProcessError::NonFiniteLogLikelihoodRatio);
        }
        let increment = alternative_probability.ln() - null_probability.ln();
        if !increment.is_finite() {
            return Err(PrequentialEProcessError::NonFiniteLogLikelihoodRatio);
        }
        log_e_value += increment;
        if !log_e_value.is_finite() {
            return Err(PrequentialEProcessError::NonFiniteLogLikelihoodRatio);
        }

        counts[outcome_index] = counts[outcome_index]
            .checked_add(1)
            .ok_or(PrequentialEProcessError::ObservationCountOverflow)?;
        observation_count = observation_count
            .checked_add(1)
            .ok_or(PrequentialEProcessError::ObservationCountOverflow)?;

        if crossed_at_observation.is_none() && log_e_value >= log_e_threshold {
            crossed_at_observation = Some(observation_index + 1);
        }
        log_likelihood_ratio_increment_by_observation.push(increment);
        log_e_value_by_observation.push(log_e_value);
    }

    let final_denominator = concentration_mass + observation_count as f64;
    let final_alternative_predictive_probabilities = prior_concentration
        .iter()
        .zip(&counts)
        .map(|(concentration, count)| (concentration + *count as f64) / final_denominator)
        .collect();

    Ok(DirichletPrequentialEProcessReport {
        alpha,
        log_e_threshold,
        log_e_value,
        log_e_value_by_observation,
        log_likelihood_ratio_increment_by_observation,
        crossed_at_observation,
        prior_concentration: prior_concentration.to_vec(),
        final_counts: counts,
        final_alternative_predictive_probabilities,
    })
}

fn mixture_log_e_value(
    alternatives: &[PrequentialAlternativeModel],
    log_likelihood_ratios: &[Option<f64>],
) -> Option<f64> {
    let weight_sum = alternatives
        .iter()
        .map(|alternative| alternative.prior_weight)
        .sum::<f64>();
    let terms = alternatives
        .iter()
        .zip(log_likelihood_ratios)
        .filter_map(|(alternative, log_ratio)| {
            log_ratio.map(|value| (alternative.prior_weight / weight_sum).ln() + value)
        })
        .collect::<Vec<_>>();
    let maximum = terms.iter().copied().reduce(f64::max)?;
    let log_sum = terms.iter().map(|term| (term - maximum).exp()).sum::<f64>();
    Some(maximum + log_sum.ln())
}

fn validate_categorical_distribution(probabilities: &[f64]) -> Result<(), (Option<usize>, f64)> {
    if probabilities.is_empty() {
        return Err((None, 0.0));
    }
    let mut sum = 0.0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err((Some(index), sum));
        }
        sum += probability;
    }
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err((None, sum));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> ModelAdequacyThresholds {
        ModelAdequacyThresholds {
            minimum_observations: 4,
            minimum_predictive_probability: 0.05,
            maximum_mean_log_loss: 1.0,
            maximum_mean_brier_score: 0.5,
            maximum_low_probability_rate: 0.25,
        }
    }

    fn binary_problem() -> (Vec<DecisionHypothesis>, Vec<DiagnosticAction>) {
        (
            vec![
                DecisionHypothesis {
                    id: "fault-a".into(),
                    prior_probability: 0.5,
                    outcome_probabilities_by_action: vec![vec![0.99, 0.01]],
                    loss_by_final_decision: vec![0.0, 1.0],
                },
                DecisionHypothesis {
                    id: "fault-b".into(),
                    prior_probability: 0.5,
                    outcome_probabilities_by_action: vec![vec![0.5, 0.5]],
                    loss_by_final_decision: vec![1.0, 0.0],
                },
            ],
            vec![DiagnosticAction {
                id: "inspect".into(),
                observation_cost: 0.23,
                safety_risk: 0.0,
            }],
        )
    }

    #[test]
    fn dirichlet_e_process_has_unit_expected_value_for_one_step_null() {
        let null = [0.7, 0.3];
        let mut expected_e_value = 0.0;
        for outcome in 0..2 {
            let report = evaluate_dirichlet_prequential_e_process(
                &[PrequentialCategoricalObservation {
                    predicted_outcome_probabilities: null.to_vec(),
                    observed_outcome_index: outcome,
                }],
                &[1.0, 1.0],
                0.05,
            )
            .expect("valid one-step e-process");
            expected_e_value += null[outcome] * report.log_e_value.exp();
        }
        assert!((expected_e_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dirichlet_e_process_is_valid_for_arbitrary_predictable_null_sequence() {
        let first_null = [0.8, 0.2];
        let second_null_after_zero = [0.3, 0.7];
        let second_null_after_one = [0.9, 0.1];
        let mut expected_e_value = 0.0;

        for first_outcome in 0..2 {
            let second_null = if first_outcome == 0 {
                second_null_after_zero
            } else {
                second_null_after_one
            };
            for second_outcome in 0..2 {
                let path_probability =
                    first_null[first_outcome] * second_null[second_outcome];
                let report = evaluate_dirichlet_prequential_e_process(
                    &[
                        PrequentialCategoricalObservation {
                            predicted_outcome_probabilities: first_null.to_vec(),
                            observed_outcome_index: first_outcome,
                        },
                        PrequentialCategoricalObservation {
                            predicted_outcome_probabilities: second_null.to_vec(),
                            observed_outcome_index: second_outcome,
                        },
                    ],
                    &[1.0, 1.0],
                    0.05,
                )
                .expect("valid predictable null path");
                expected_e_value += path_probability * report.log_e_value.exp();
            }
        }

        assert!((expected_e_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dirichlet_e_process_detects_persistent_frequency_misspecification() {
        let observations = (0..40)
            .map(|_| PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![0.5, 0.5],
                observed_outcome_index: 0,
            })
            .collect::<Vec<_>>();
        let report = evaluate_dirichlet_prequential_e_process(
            &observations,
            &[1.0, 1.0],
            0.05,
        )
        .expect("valid persistent-shift sequence");

        assert!(report.log_e_value >= report.log_e_threshold);
        assert!(report.crossed_at_observation.is_some());
        assert_eq!(report.final_counts, vec![40, 0]);
        assert!(report.final_alternative_predictive_probabilities[0] > 0.95);
    }

    #[test]
    fn dirichlet_e_process_rejects_null_support_and_category_drift() {
        let support_violation = evaluate_dirichlet_prequential_e_process(
            &[PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![1.0, 0.0],
                observed_outcome_index: 1,
            }],
            &[1.0, 1.0],
            0.05,
        );
        assert!(matches!(
            support_violation,
            Err(PrequentialEProcessError::NullSupportViolation {
                observation_index: 0
            })
        ));

        let category_drift = evaluate_dirichlet_prequential_e_process(
            &[PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![1.0, 0.0, 0.0],
                observed_outcome_index: 0,
            }],
            &[1.0, 1.0],
            0.05,
        );
        assert!(matches!(
            category_drift,
            Err(PrequentialEProcessError::DirichletCategoryCountMismatch {
                observation_index: 0,
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn dirichlet_e_process_rejects_invalid_concentration_and_alpha() {
        assert!(matches!(
            evaluate_dirichlet_prequential_e_process(&[], &[], 0.05),
            Err(PrequentialEProcessError::EmptyDirichletConcentration)
        ));
        assert!(matches!(
            evaluate_dirichlet_prequential_e_process(&[], &[1.0, 0.0], 0.05),
            Err(PrequentialEProcessError::InvalidDirichletConcentration {
                category_index: 1
            })
        ));
        assert!(matches!(
            evaluate_dirichlet_prequential_e_process(&[], &[1.0, 1.0], 1.0),
            Err(PrequentialEProcessError::InvalidAlpha)
        ));
    }

    fn adequacy_report_with_status(status: ModelAdequacyStatus) -> ModelAdequacyReport {
        ModelAdequacyReport {
            status,
            observation_count: if status == ModelAdequacyStatus::InsufficientEvidence {
                0
            } else {
                40
            },
            zero_support_count: if status == ModelAdequacyStatus::Inadequate {
                1
            } else {
                0
            },
            low_probability_count: 0,
            low_probability_rate: 0.0,
            mean_log_loss: Some(0.5),
            mean_multiclass_brier_score: Some(0.2),
            minimum_observed_probability: Some(0.2),
            thresholds: thresholds(),
        }
    }

    #[test]
    fn open_world_gate_requires_more_evidence_before_weight_programming() {
        let adequacy = adequacy_report_with_status(ModelAdequacyStatus::InsufficientEvidence);
        let gate = evaluate_open_world_decision_gate(&adequacy, None);

        assert_eq!(
            gate.disposition,
            OpenWorldGateDisposition::EvidenceCollectionOnly
        );
        assert!(gate.diagnostic_probe_selection_permitted);
        assert!(!gate.weight_program_generation_permitted);
        assert!(!gate.detached_candidate_evaluation_permitted);
        assert!(gate.independent_verifier_required);
        assert!(!gate.candidate_acceptance_authorized);
        assert!(!gate.live_model_mutation_authorized);
    }

    #[test]
    fn open_world_gate_requires_model_revision_after_detected_inadequacy() {
        let adequacy = adequacy_report_with_status(ModelAdequacyStatus::Inadequate);
        let gate = evaluate_open_world_decision_gate(&adequacy, None);

        assert_eq!(
            gate.disposition,
            OpenWorldGateDisposition::ModelRevisionRequired
        );
        assert!(gate.diagnostic_probe_selection_permitted);
        assert!(!gate.weight_program_generation_permitted);
        assert!(!gate.detached_candidate_evaluation_permitted);
        assert!(!gate.candidate_acceptance_authorized);
        assert!(!gate.live_model_mutation_authorized);
    }

    #[test]
    fn anytime_evidence_overrides_aggregate_no_detected_inadequacy() {
        let adequacy =
            adequacy_report_with_status(ModelAdequacyStatus::NoDetectedInadequacy);
        let gate = evaluate_open_world_decision_gate(&adequacy, Some(66));

        assert_eq!(
            gate.declared_adequacy_status,
            ModelAdequacyStatus::NoDetectedInadequacy
        );
        assert_eq!(
            gate.effective_adequacy_status,
            ModelAdequacyStatus::Inadequate
        );
        assert_eq!(gate.prequential_evidence_crossed_at, Some(66));
        assert_eq!(
            gate.disposition,
            OpenWorldGateDisposition::ModelRevisionRequired
        );
        assert!(!gate.weight_program_generation_permitted);
    }

    #[test]
    fn open_world_gate_only_permits_detached_candidates_and_never_live_mutation() {
        let adequacy =
            adequacy_report_with_status(ModelAdequacyStatus::NoDetectedInadequacy);
        let gate = evaluate_open_world_decision_gate(&adequacy, None);

        assert_eq!(
            gate.disposition,
            OpenWorldGateDisposition::DetachedCandidateEvaluationPermitted
        );
        assert!(gate.weight_program_generation_permitted);
        assert!(gate.detached_candidate_evaluation_permitted);
        assert!(gate.independent_verifier_required);
        assert!(!gate.candidate_acceptance_authorized);
        assert!(!gate.live_model_mutation_authorized);
    }

    #[test]
    fn tv_ball_moves_probability_from_low_loss_to_high_loss_exactly() {
        let nominal = vec![0.8, 0.2];
        let losses = vec![0.0, 1.0];
        let worst = worst_case_distribution_within_tv(&nominal, &losses, 0.1)
            .expect("valid finite TV ball");
        assert!((worst[0] - 0.7).abs() < 1e-12);
        assert!((worst[1] - 0.3).abs() < 1e-12);
        let tv_distance = 0.5
            * worst
                .iter()
                .zip(&nominal)
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>();
        assert!((tv_distance - 0.1).abs() < 1e-12);
    }

    #[test]
    fn zero_radius_matches_nominal_evsi_selection() {
        let (hypotheses, actions) = binary_problem();
        let nominal = evaluate_decision_focused_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
        )
        .expect("nominal diagnostics");
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.0,
        )
        .expect("zero-radius TV diagnostics");
        assert_eq!(
            robust.nominal_selected_action_index,
            nominal.selected_action_index
        );
        assert_eq!(
            robust.robust_selected_action_index,
            nominal.selected_action_index
        );
        assert!(
            (robust.actions[0].nominal_expected_posterior_bayes_risk
                - robust.actions[0].worst_case_expected_posterior_bayes_risk)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn nominal_probe_is_rejected_when_tv_shift_erases_its_decision_value() {
        let (hypotheses, actions) = binary_problem();
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.2,
        )
        .expect("robust diagnostics");
        assert_eq!(robust.nominal_selected_action_index, Some(0));
        assert_eq!(robust.robust_selected_action_index, None);
        assert!(robust.actions[0].worst_case_total_score.unwrap() > robust.no_probe_bayes_risk);
    }

    #[test]
    fn unsupported_outcomes_receive_the_declared_maximum_loss_bound() {
        let (mut hypotheses, actions) = binary_problem();
        hypotheses[0].outcome_probabilities_by_action[0] = vec![1.0, 0.0];
        hypotheses[1].outcome_probabilities_by_action[0] = vec![1.0, 0.0];
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.25,
        )
        .expect("robust diagnostics with an open-support outcome");
        assert_eq!(robust.actions[0].unsupported_outcome_indices, vec![1]);
        assert!(robust.actions[0].worst_case_outcome_probabilities[1] > 0.0);
        assert!(robust.actions[0].worst_case_expected_posterior_bayes_risk > 0.0);
    }

    #[test]
    fn adequacy_gate_detects_out_of_support_observations() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![1.0, 0.0],
            observed_outcome_index: 1,
        }];
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential record");
        assert_eq!(report.status, ModelAdequacyStatus::Inadequate);
        assert_eq!(report.zero_support_count, 1);
        assert_eq!(report.mean_log_loss, None);
    }

    #[test]
    fn adequacy_gate_requires_minimum_evidence_and_never_claims_proof() {
        let observations = (0..4)
            .map(|_| PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![0.8, 0.2],
                observed_outcome_index: 0,
            })
            .collect::<Vec<_>>();
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential records");
        assert_eq!(report.status, ModelAdequacyStatus::NoDetectedInadequacy);
        assert_eq!(report.observation_count, 4);
        assert!(report.mean_log_loss.unwrap() < 1.0);
        assert!(report.mean_multiclass_brier_score.unwrap() < 0.5);
    }

    #[test]
    fn adequacy_gate_abstains_when_evidence_is_insufficient() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.8, 0.2],
            observed_outcome_index: 0,
        }];
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential record");
        assert_eq!(report.status, ModelAdequacyStatus::InsufficientEvidence);
    }

    #[test]
    fn malformed_predictive_mass_fails_closed() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.8, 0.3],
            observed_outcome_index: 0,
        }];
        assert!(matches!(
            evaluate_prequential_model_adequacy(&observations, thresholds()),
            Err(ModelAdequacyError::ProbabilityMassNotNormalized { .. })
        ));
    }

    #[test]
    fn mixture_e_value_has_unit_expectation_under_the_declared_null() {
        let null = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.75, 0.25],
            observed_outcome_index: 0,
        }];
        let alternatives = vec![
            PrequentialAlternativeModel {
                id: "alternative-a".into(),
                prior_weight: 0.4,
                predicted_outcome_probabilities: vec![vec![0.2, 0.8]],
            },
            PrequentialAlternativeModel {
                id: "alternative-b".into(),
                prior_weight: 0.6,
                predicted_outcome_probabilities: vec![vec![0.9, 0.1]],
            },
        ];
        let e_for_zero = evaluate_prequential_e_process(&null, &alternatives, 0.05)
            .expect("first outcome e-process");
        let null_for_one = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.75, 0.25],
            observed_outcome_index: 1,
        }];
        let e_for_one = evaluate_prequential_e_process(&null_for_one, &alternatives, 0.05)
            .expect("second outcome e-process");
        let expectation = 0.75 * e_for_zero.log_e_value.unwrap().exp()
            + 0.25 * e_for_one.log_e_value.unwrap().exp();
        assert!((expectation - 1.0).abs() < 1e-12);
    }

    #[test]
    fn prequential_e_process_crosses_a_declared_sequential_threshold() {
        let observations = (0..8)
            .map(|_| PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![0.5, 0.5],
                observed_outcome_index: 0,
            })
            .collect::<Vec<_>>();
        let alternative = PrequentialAlternativeModel {
            id: "persistent-regime-shift".into(),
            prior_weight: 1.0,
            predicted_outcome_probabilities: (0..8).map(|_| vec![0.9, 0.1]).collect(),
        };
        let report = evaluate_prequential_e_process(&observations, &[alternative], 0.05)
            .expect("valid sequential e-process");
        assert_eq!(report.crossed_at_observation, Some(6));
        assert!(report.log_e_value.unwrap() >= report.log_e_threshold);
    }

    #[test]
    fn prequential_e_process_keeps_tiny_valid_alpha_threshold_finite() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.5, 0.5],
            observed_outcome_index: 0,
        }];
        let alternative = PrequentialAlternativeModel {
            id: "other".into(),
            prior_weight: 1.0,
            predicted_outcome_probabilities: vec![vec![0.5, 0.5]],
        };
        let report =
            evaluate_prequential_e_process(&observations, &[alternative], f64::from_bits(1))
                .expect("subnormal but valid alpha");
        assert!(report.log_e_threshold.is_finite());
        assert!((report.log_e_threshold + f64::from_bits(1).ln()).abs() < 1e-12);
    }

    #[test]
    fn prequential_e_process_requires_strictly_positive_alpha_below_one() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.5, 0.5],
            observed_outcome_index: 0,
        }];
        let alternative = PrequentialAlternativeModel {
            id: "other".into(),
            prior_weight: 1.0,
            predicted_outcome_probabilities: vec![vec![0.5, 0.5]],
        };
        for alpha in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
            assert!(matches!(
                evaluate_prequential_e_process(&observations, &[alternative.clone()], alpha),
                Err(PrequentialEProcessError::InvalidAlpha)
            ));
        }
    }

    #[test]
    fn prequential_e_process_rejects_zero_support_under_the_null() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![1.0, 0.0],
            observed_outcome_index: 1,
        }];
        let alternative = PrequentialAlternativeModel {
            id: "other".into(),
            prior_weight: 1.0,
            predicted_outcome_probabilities: vec![vec![0.5, 0.5]],
        };
        assert!(matches!(
            evaluate_prequential_e_process(&observations, &[alternative], 0.05),
            Err(PrequentialEProcessError::NullSupportViolation {
                observation_index: 0
            })
        ));
    }

    #[test]
    fn categorical_tv_calibration_matches_the_declared_hoeffding_bound() {
        let calibration = calibrate_categorical_tv_radius_from_counts(&[25, 25], 0.05)
            .expect("valid independent calibration counts");
        let expected_coordinate_bound = ((2.0 * 2.0_f64 / 0.05).ln() / (2.0 * 50.0)).sqrt();
        assert_eq!(calibration.sample_count, 50);
        assert_eq!(calibration.confidence_level, 0.95);
        assert_eq!(calibration.empirical_probabilities, vec![0.5, 0.5]);
        assert!(
            (calibration.per_category_absolute_error_bound - expected_coordinate_bound).abs()
                < 1e-12
        );
        assert!((calibration.total_variation_radius - expected_coordinate_bound).abs() < 1e-12);
    }

    #[test]
    fn tv_calibration_uses_log_domain_for_tiny_valid_alpha() {
        let calibration = calibrate_categorical_tv_radius_from_counts(&[25, 25], f64::from_bits(1))
            .expect("subnormal but valid calibration alpha");
        assert!(calibration.per_category_absolute_error_bound.is_finite());
        assert_eq!(calibration.total_variation_radius, 1.0);
    }

    #[test]
    fn calibrated_tv_radius_shrinks_with_more_independent_samples() {
        let smaller = calibrate_categorical_tv_radius_from_counts(&[25, 25], 0.05)
            .expect("small calibration");
        let larger = calibrate_categorical_tv_radius_from_counts(&[250, 250], 0.05)
            .expect("large calibration");
        assert!(larger.total_variation_radius < smaller.total_variation_radius);
    }

    #[test]
    fn tv_calibration_rejects_invalid_alpha_empty_samples_and_overflow() {
        assert!(matches!(
            calibrate_categorical_tv_radius_from_counts(&[], 0.05),
            Err(TvAmbiguityCalibrationError::EmptyCategorySet)
        ));
        assert!(matches!(
            calibrate_categorical_tv_radius_from_counts(&[0, 0], 0.05),
            Err(TvAmbiguityCalibrationError::EmptyCalibrationSample)
        ));
        for alpha in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
            assert!(matches!(
                calibrate_categorical_tv_radius_from_counts(&[1, 1], alpha),
                Err(TvAmbiguityCalibrationError::InvalidAlpha)
            ));
        }
        assert!(matches!(
            calibrate_categorical_tv_radius_from_counts(&[u64::MAX, 1], 0.05),
            Err(TvAmbiguityCalibrationError::CountOverflow)
        ));
    }

    #[test]
    fn invalid_tv_radius_fails_closed() {
        assert!(matches!(
            worst_case_distribution_within_tv(&[0.5, 0.5], &[0.0, 1.0], 1.1),
            Err(TvRobustDecisionError::InvalidTotalVariationRadius)
        ));
    }
}
