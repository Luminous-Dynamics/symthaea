// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source-bound known-truth stress harness for open-world FEP diagnostics.
//!
//! This is an exact finite synthetic fixture, not a live ActiveInferenceAgent
//! evaluation. It contrasts nominal EVSI and finite-TV robust selection under
//! a deliberately misspecified predictive model, then demonstrates prequential
//! score monitoring, an out-of-support observation, and sequential e-process
//! evidence under a predictable alternative family.

use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use serde_json::json;
use symthaea_fep::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionHypothesis, DiagnosticAction,
    evaluate_decision_focused_diagnostics,
};
use symthaea_fep::open_world_robust_decision::{
    ModelAdequacyStatus, ModelAdequacyThresholds, PrequentialAlternativeModel,
    PrequentialCategoricalObservation, evaluate_prequential_e_process,
    evaluate_prequential_model_adequacy, evaluate_tv_robust_decision_diagnostics,
};

const TV_RADIUS: f64 = 0.2;
const E_PROCESS_ALPHA: f64 = 0.05;

fn source_sha() -> Result<String, Box<dyn Error>> {
    let sha = env::var("SYMTHAEA_SOURCE_SHA")?;
    if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal Git SHA",
        )
        .into());
    }
    Ok(sha)
}

fn decision_problem() -> (
    Vec<DecisionHypothesis>,
    Vec<DiagnosticAction>,
    Vec<Vec<Vec<f64>>>,
) {
    let hypotheses = vec![
        DecisionHypothesis {
            id: "fault-a".into(),
            prior_probability: 0.5,
            outcome_probabilities_by_action: vec![
                vec![0.99, 0.01],
                vec![0.80, 0.20],
            ],
            loss_by_final_decision: vec![0.0, 1.0],
        },
        DecisionHypothesis {
            id: "fault-b".into(),
            prior_probability: 0.5,
            outcome_probabilities_by_action: vec![
                vec![0.50, 0.50],
                vec![0.10, 0.90],
            ],
            loss_by_final_decision: vec![1.0, 0.0],
        },
    ];
    let actions = vec![
        DiagnosticAction {
            id: "nominally-sharp-but-misspecified".into(),
            observation_cost: 0.10,
            safety_risk: 0.0,
        },
        DiagnosticAction {
            id: "moderate-but-stable-probe".into(),
            observation_cost: 0.25,
            safety_risk: 0.0,
        },
    ];

    // Deliberately misspecify action 0 under fault-b while retaining the
    // declared binary support. Action 1 remains correctly specified.
    let truth = vec![
        vec![vec![0.99, 0.01], vec![0.80, 0.20]],
        vec![vec![0.95, 0.05], vec![0.10, 0.90]],
    ];
    (hypotheses, actions, truth)
}

fn marginal_truth_by_action(
    hypotheses: &[DecisionHypothesis],
    truth: &[Vec<Vec<f64>>],
    action_index: usize,
) -> Vec<f64> {
    let outcome_count = truth[0][action_index].len();
    (0..outcome_count)
        .map(|outcome_index| {
            hypotheses
                .iter()
                .zip(truth)
                .map(|(hypothesis, truth_actions)| {
                    hypothesis.prior_probability * truth_actions[action_index][outcome_index]
                })
                .sum::<f64>()
        })
        .collect()
}

fn actual_expected_scores(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    truth: &[Vec<Vec<f64>>],
    nominal: &symthaea_fep::decision_relevant_information::DecisionDiagnosticEvaluation,
) -> Vec<f64> {
    let maximum_loss = hypotheses
        .iter()
        .flat_map(|hypothesis| hypothesis.loss_by_final_decision.iter().copied())
        .fold(0.0_f64, f64::max);
    nominal
        .actions
        .iter()
        .zip(actions)
        .map(|(evaluation, action)| {
            let truth_predictive =
                marginal_truth_by_action(hypotheses, truth, evaluation.action_index);
            let risk = truth_predictive
                .iter()
                .zip(&evaluation.posterior_bayes_risk_by_outcome)
                .map(|(probability, posterior_risk)| {
                    probability * posterior_risk.unwrap_or(maximum_loss)
                })
                .sum::<f64>();
            risk + action.observation_cost
        })
        .collect()
}

fn adequacy_thresholds() -> ModelAdequacyThresholds {
    ModelAdequacyThresholds {
        minimum_observations: 32,
        minimum_predictive_probability: 0.05,
        maximum_mean_log_loss: 0.80,
        maximum_mean_brier_score: 0.60,
        maximum_low_probability_rate: 0.25,
    }
}

fn alternating_observations(count: usize) -> Vec<PrequentialCategoricalObservation> {
    (0..count)
        .map(|index| PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.5, 0.5],
            observed_outcome_index: index % 2,
        })
        .collect()
}

fn regime_shift_observations() -> Vec<PrequentialCategoricalObservation> {
    let mut observations = alternating_observations(40);
    observations.extend((0..80).map(|_| PrequentialCategoricalObservation {
        predicted_outcome_probabilities: vec![0.5, 0.5],
        observed_outcome_index: 0,
    }));
    observations
}

fn alternative_models(observation_count: usize) -> Vec<PrequentialAlternativeModel> {
    vec![
        PrequentialAlternativeModel {
            id: "persistent-zero-regime".into(),
            prior_weight: 0.5,
            predicted_outcome_probabilities: (0..observation_count)
                .map(|_| vec![0.9, 0.1])
                .collect(),
        },
        PrequentialAlternativeModel {
            id: "persistent-one-regime".into(),
            prior_weight: 0.5,
            predicted_outcome_probabilities: (0..observation_count)
                .map(|_| vec![0.1, 0.9])
                .collect(),
        },
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = source_sha()?;
    let (hypotheses, actions, truth) = decision_problem();
    let decision_config = DecisionDiagnosticConfig {
        observation_cost_weight: 1.0,
        safety_risk_weight: 0.0,
        maximum_safety_risk: 1.0,
    };

    let nominal =
        evaluate_decision_focused_diagnostics(&hypotheses, &actions, decision_config)?;
    let robust = evaluate_tv_robust_decision_diagnostics(
        &hypotheses,
        &actions,
        decision_config,
        TV_RADIUS,
    )?;
    let actual_scores = actual_expected_scores(&hypotheses, &actions, &truth, &nominal);
    let no_probe_score = nominal.prior_bayes_risk;
    let oracle_score = actual_scores
        .iter()
        .copied()
        .fold(no_probe_score, f64::min);
    let nominal_index = nominal.selected_action_index;
    let robust_index = robust.robust_selected_action_index;
    let selected_score = |index: Option<usize>| match index {
        Some(index) => actual_scores[index],
        None => no_probe_score,
    };
    let nominal_actual_score = selected_score(nominal_index);
    let robust_actual_score = selected_score(robust_index);
    let maximum_regret = (nominal_actual_score - oracle_score)
        .max(robust_actual_score - oracle_score)
        .max(0.0);

    if nominal_index != Some(0) || robust_index != Some(1) {
        return Err(IoError::other(
            "known-truth fixture no longer produces the preregistered nominal-to-robust switch",
        )
        .into());
    }
    if robust_actual_score >= nominal_actual_score {
        return Err(IoError::other(
            "robust selector did not improve known-truth expected decision loss in this fixture",
        )
        .into());
    }

    let thresholds = adequacy_thresholds();
    let no_shift = alternating_observations(120);
    let shifted = regime_shift_observations();
    let no_shift_adequacy = evaluate_prequential_model_adequacy(&no_shift, thresholds)?;
    let shifted_adequacy = evaluate_prequential_model_adequacy(&shifted, thresholds)?;

    let no_shift_alternatives = alternative_models(no_shift.len());
    let shift_alternatives = alternative_models(shifted.len());
    let no_shift_e_process =
        evaluate_prequential_e_process(&no_shift, &no_shift_alternatives, E_PROCESS_ALPHA)?;
    let shift_e_process =
        evaluate_prequential_e_process(&shifted, &shift_alternatives, E_PROCESS_ALPHA)?;

    // A subsequent observation outside the entire registered support must
    // trigger an adequacy violation even when the aggregate score gate is not
    // sufficiently informative.
    let out_of_support = vec![PrequentialCategoricalObservation {
        predicted_outcome_probabilities: vec![1.0, 0.0],
        observed_outcome_index: 1,
    }];
    let out_of_support_report =
        evaluate_prequential_model_adequacy(&out_of_support, thresholds)?;

    if no_shift_adequacy.status != ModelAdequacyStatus::NoDetectedInadequacy {
        return Err(IoError::other("alternating null adequacy control failed").into());
    }
    if shifted_adequacy.status != ModelAdequacyStatus::NoDetectedInadequacy {
        return Err(IoError::other(
            "shift fixture must demonstrate that aggregate score thresholds alone can miss a symmetric-predictive regime change",
        )
        .into());
    }
    if no_shift_e_process.crossed_at_observation.is_some() {
        return Err(IoError::other("alternating null control crossed the e-process threshold").into());
    }
    if shift_e_process.crossed_at_observation.is_none() {
        return Err(IoError::other("persistent regime shift did not cross the e-process threshold").into());
    }
    if out_of_support_report.status != ModelAdequacyStatus::Inadequate {
        return Err(IoError::other("out-of-support observation did not fail the adequacy gate").into());
    }

    let report = json!({
        "schema_version": "hlb-open-world-robustness-result-v1",
        "status": "synthetic_known_truth_stress_only",
        "claim_ceiling": "Exact finite decision problem and deterministic prequential sequence controls; not a live ActiveInferenceAgent result, field study, or SOTA comparison.",
        "source_sha": source_sha,
        "decision_problem": {
            "id": "nominal-sharp-vs-misspecified-probe",
            "hypotheses": hypotheses,
            "actions": actions,
            "truth_outcome_probabilities_by_hypothesis_action": truth,
            "tv_radius": TV_RADIUS,
            "nominal_selected_action_index": nominal_index,
            "robust_selected_action_index": robust_index,
            "no_probe_bayes_risk": no_probe_score,
            "nominal_actual_expected_score": nominal_actual_score,
            "robust_actual_expected_score": robust_actual_score,
            "oracle_actual_expected_score": oracle_score,
            "nominal_actual_regret": (nominal_actual_score - oracle_score).max(0.0),
            "robust_actual_regret": (robust_actual_score - oracle_score).max(0.0),
            "maximum_reported_regret": maximum_regret,
            "nominal_policy_evaluation": nominal,
            "robust_policy_evaluation": robust,
            "actual_expected_score_by_action": actual_scores
        },
        "prequential_adequacy": {
            "thresholds": thresholds,
            "no_shift_observation_count": no_shift.len(),
            "no_shift_report": no_shift_adequacy,
            "shifted_observation_count": shifted.len(),
            "shifted_aggregate_score_report": shifted_adequacy,
            "out_of_support_report": out_of_support_report
        },
        "anytime_e_process": {
            "alpha": E_PROCESS_ALPHA,
            "null_predictive_model": [0.5, 0.5],
            "alternative_models": shift_alternatives,
            "no_shift_report": no_shift_e_process,
            "regime_shift_report": shift_e_process,
            "guarantee_scope": "Ville bound conditional on the declared null predictive model and alternatives frozen before each evaluated outcome; this single deterministic fixture is not a type-I error-rate study."
        }
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
