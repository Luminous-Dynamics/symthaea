// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_science_evidence::{
    CandidateExperiment, DiscriminationHypothesis, GaussianModelPrediction, MeasurementNoiseModel,
    ModelDiscriminationConfig, ModelPriorScenario, ModelUniverseCoverage, NoiseCalibrationStatus,
    PredictionDisposition, gaussian_predictive_kl_nats, rank_model_discriminating_experiments,
};

fn hypothesis(id: &str, source_byte: u8) -> DiscriminationHypothesis {
    DiscriminationHypothesis {
        model_id: id.to_string(),
        source_revision_sha: (0..40).map(|_| source_byte as char).collect(),
        specification_reference: format!("synthetic://model-spec/{id}/v1"),
    }
}

fn quantitative_prediction(
    model_id: &str,
    mean: f64,
    predictive_sigma: f64,
) -> GaussianModelPrediction {
    GaussianModelPrediction {
        model_id: model_id.to_string(),
        disposition: PredictionDisposition::Quantitative,
        predictive_mean: Some(mean),
        predictive_sigma: Some(predictive_sigma),
        derivation_reference: Some(format!("synthetic://derivation/{model_id}/v1")),
    }
}

fn not_derived_prediction(model_id: &str) -> GaussianModelPrediction {
    GaussianModelPrediction {
        model_id: model_id.to_string(),
        disposition: PredictionDisposition::NotDerived,
        predictive_mean: None,
        predictive_sigma: None,
        derivation_reference: None,
    }
}

fn experiment(
    id: &str,
    observable: &str,
    cost: f64,
    availability_probability: f64,
    measurement_sigma: f64,
    predictions: Vec<GaussianModelPrediction>,
) -> CandidateExperiment {
    CandidateExperiment {
        experiment_id: id.to_string(),
        observable_id: observable.to_string(),
        domain_id: "synthetic-known-answer-domain".to_string(),
        units: "dimensionless".to_string(),
        measurement_noise_model: MeasurementNoiseModel::Gaussian,
        measurement_sigma: Some(measurement_sigma),
        noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
        calibration_reference: Some("synthetic://noise-calibration/v1".to_string()),
        availability_probability,
        cost,
        exclusive_group: None,
        predictions,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // This is an analytic/synthetic decision-rule fixture, not a physical
    // prediction for Chain Theory, causal sets, electromagnetism or a UAP.
    let hypotheses = vec![
        hypothesis("synthetic-model-a", b'a'),
        hypothesis("synthetic-model-b", b'b'),
        hypothesis("synthetic-model-c", b'c'),
    ];
    let prior_scenarios = vec![
        ModelPriorScenario {
            scenario_id: "equal-prior".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![1.0, 1.0, 1.0],
        },
        ModelPriorScenario {
            scenario_id: "family-weighted-prior".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![2.0, 1.0, 1.0],
        },
    ];
    let experiments = vec![
        experiment(
            "low-cost-separation",
            "synthetic-observable-1",
            0.6,
            0.95,
            0.20,
            vec![
                quantitative_prediction("synthetic-model-a", 0.0, 0.04),
                quantitative_prediction("synthetic-model-b", 0.8, 0.06),
                quantitative_prediction("synthetic-model-c", 1.7, 0.08),
            ],
        ),
        experiment(
            "high-cost-high-precision",
            "synthetic-observable-2",
            3.0,
            0.55,
            0.07,
            vec![
                quantitative_prediction("synthetic-model-a", 0.0, 0.03),
                quantitative_prediction("synthetic-model-b", 0.45, 0.03),
                quantitative_prediction("synthetic-model-c", 0.9, 0.04),
            ],
        ),
        experiment(
            "partial-prediction-control",
            "synthetic-observable-3",
            0.2,
            0.80,
            0.15,
            vec![
                quantitative_prediction("synthetic-model-a", 0.0, 0.05),
                quantitative_prediction("synthetic-model-b", 0.1, 0.05),
                not_derived_prediction("synthetic-model-c"),
            ],
        ),
    ];
    let config = ModelDiscriminationConfig {
        model_universe: ModelUniverseCoverage::OpenWorld,
        prior_scenarios,
        quadrature_intervals: 256,
        quadrature_tolerance_nats: 1.0e-6,
    };

    let report = rank_model_discriminating_experiments(&hypotheses, &experiments, &config)?;
    println!("epistemic_status=SYNTHETIC_MODEL_IDENTITY_EIG_NOT_PHYSICAL_THEORY_VALIDATION");
    println!("objective=mutual_information_between_named_model_and_scalar_observation");
    println!("open_world_unresolved_excluded=true");
    println!("model_count={}", report.model_ids.len());
    println!("prior_scenarios={}", report.prior_scenario_ids.join(";"));
    println!(
        "experiment,availability,status,conditional_worst_eig,conditional_mean_eig,adjusted_worst_eig,adjusted_mean_eig,adjusted_worst_eig_per_cost,adjusted_mean_eig_per_cost"
    );

    for score in &report.experiments {
        let status = match &score.status {
            symthaea_science_evidence::ExperimentDiscriminationStatus::Scored => "SCORED",
            symthaea_science_evidence::ExperimentDiscriminationStatus::NotComparable { .. } => {
                "NOT_COMPARABLE"
            }
            symthaea_science_evidence::ExperimentDiscriminationStatus::QuadratureNotConverged {
                ..
            } => "QUADRATURE_NOT_CONVERGED",
        };
        println!(
            "{},{:.3},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            score.experiment_id,
            score.availability_probability,
            status,
            score.worst_case_model_identity_eig_nats.unwrap_or(f64::NAN),
            score
                .weighted_mean_model_identity_eig_nats
                .unwrap_or(f64::NAN),
            score
                .availability_adjusted_worst_case_model_identity_eig_nats
                .unwrap_or(f64::NAN),
            score
                .availability_adjusted_weighted_mean_model_identity_eig_nats
                .unwrap_or(f64::NAN),
            score.worst_case_eig_per_cost.unwrap_or(f64::NAN),
            score.weighted_mean_eig_per_cost.unwrap_or(f64::NAN),
        );
        for scenario in &score.prior_scenario_scores {
            println!(
                "prior_scenario={},experiment={},conditional_eig_nats={:.9},availability_adjusted_eig_nats={:.9},expected_posterior_entropy_nats={:.9},quadrature_error_nats={:.3e},prior_entropy_nats={:.9}",
                scenario.scenario_id,
                score.experiment_id,
                scenario.refined_eig_nats,
                scenario.availability_adjusted_eig_nats,
                scenario.expected_posterior_entropy_nats,
                scenario.quadrature_error_nats,
                scenario.prior_entropy_nats,
            );
        }
        if let symthaea_science_evidence::ExperimentDiscriminationStatus::NotComparable {
            blockers,
        } = &score.status
        {
            println!(
                "blocking_reasons={},{}",
                score.experiment_id,
                blockers.len()
            );
        }
    }

    println!("claim_boundary=ranking_declared_synthetic_predictive_distributions_only");
    println!(
        "not_claimed=physical_model_selection,new_law,UAP_origin,parameter_EIG,closed_world_truth"
    );

    // Compare a different design objective on exactly the same candidate
    // experiments. This is the minimum directed pairwise KL divergence across
    // all ordered model pairs, not model-identity EIG. A partial prediction
    // matrix remains NOT_COMPARABLE rather than receiving an artificial score.
    println!("objective=minimum_directed_pairwise_predictive_KL_not_model_identity_EIG");
    println!(
        "experiment,status,min_directed_pairwise_kl_nats,worst_pair,eig_per_cost_for_reference"
    );
    for candidate in &experiments {
        let Some(measurement_sigma) = candidate.measurement_sigma else {
            println!("{},NOT_COMPARABLE,,,", candidate.experiment_id);
            continue;
        };
        if candidate.measurement_noise_model != MeasurementNoiseModel::Gaussian
            || candidate.noise_calibration == NoiseCalibrationStatus::Uncalibrated
            || candidate.predictions.len() != hypotheses.len()
            || candidate.predictions.iter().any(|prediction| {
                prediction.disposition != PredictionDisposition::Quantitative
                    || prediction.predictive_mean.is_none()
                    || prediction.predictive_sigma.is_none()
                    || prediction.derivation_reference.is_none()
            })
        {
            println!("{},NOT_COMPARABLE,,,", candidate.experiment_id);
            continue;
        }

        let mut minimum: Option<(String, String, f64)> = None;
        for (left_index, left) in candidate.predictions.iter().enumerate() {
            for (right_index, right) in candidate.predictions.iter().enumerate() {
                if left_index == right_index {
                    continue;
                }
                let divergence = gaussian_predictive_kl_nats(
                    left.predictive_mean.unwrap(),
                    left.predictive_sigma.unwrap(),
                    right.predictive_mean.unwrap(),
                    right.predictive_sigma.unwrap(),
                    measurement_sigma,
                )?;
                let pair = (left.model_id.clone(), right.model_id.clone());
                let replace = minimum
                    .as_ref()
                    .is_none_or(|(best_left, best_right, best_value)| {
                        divergence
                            .total_cmp(best_value)
                            .then_with(|| pair.0.cmp(best_left))
                            .then_with(|| pair.1.cmp(best_right))
                            .is_lt()
                    });
                if replace {
                    minimum = Some((pair.0, pair.1, divergence));
                }
            }
        }

        if let Some((left, right, divergence)) = minimum {
            let eig_per_cost = report
                .experiments
                .iter()
                .find(|score| score.experiment_id == candidate.experiment_id)
                .and_then(|score| score.weighted_mean_eig_per_cost);
            println!(
                "{},SCORED,{:.9},{left}->{right},{:.9}",
                candidate.experiment_id,
                divergence,
                eig_per_cost.unwrap_or(f64::NAN),
            );
        } else {
            println!("{},NOT_COMPARABLE,,,", candidate.experiment_id);
        }
    }
    println!(
        "KL direction is explicit: D_KL(p_left || p_right); the minimum is over ordered pairs."
    );
    println!(
        "The minimum directed KL, model-identity EIG and model-recovery accuracy are distinct objectives."
    );
    Ok(())
}
