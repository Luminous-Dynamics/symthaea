// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_science_evidence::{
    CandidateExperiment, DetectionAwareCandidateExperiment, DiscriminationHypothesis,
    GaussianModelPrediction, MeasurementNoiseModel, ModelDetectionProbability,
    ModelDiscriminationConfig, ModelPriorScenario, ModelUniverseCoverage, NoiseCalibrationStatus,
    PredictionDisposition, rank_detection_aware_model_discriminating_experiments,
};

fn hypothesis(id: &str, source_byte: u8) -> DiscriminationHypothesis {
    DiscriminationHypothesis {
        model_id: id.to_string(),
        source_revision_sha: std::iter::repeat_n(source_byte as char, 40).collect(),
        specification_reference: format!("synthetic://model-spec/{id}/v1"),
    }
}

fn prediction(id: &str, mean: f64, predictive_sigma: f64) -> GaussianModelPrediction {
    GaussianModelPrediction {
        model_id: id.to_string(),
        disposition: PredictionDisposition::Quantitative,
        predictive_mean: Some(mean),
        predictive_sigma: Some(predictive_sigma),
        derivation_reference: Some(format!("synthetic://observable-derivation/{id}/v1")),
    }
}

fn detection(id: &str, probability_detected: f64) -> ModelDetectionProbability {
    ModelDetectionProbability {
        model_id: id.to_string(),
        probability_detected,
        derivation_reference: format!("synthetic://instrument-detection/{id}/v1"),
    }
}

fn candidate(
    id: &str,
    execution_probability: f64,
    cost: f64,
    means: [f64; 3],
    predictive_sigmas: [f64; 3],
    detection_probabilities: [f64; 3],
) -> DetectionAwareCandidateExperiment {
    let ids = [
        "synthetic-model-a",
        "synthetic-model-b",
        "synthetic-model-c",
    ];
    let predictions = ids
        .iter()
        .enumerate()
        .map(|(index, model_id)| prediction(model_id, means[index], predictive_sigmas[index]))
        .collect();
    let detection_probabilities = ids
        .iter()
        .enumerate()
        .map(|(index, model_id)| detection(model_id, detection_probabilities[index]))
        .collect();

    DetectionAwareCandidateExperiment {
        experiment: CandidateExperiment {
            experiment_id: id.to_string(),
            observable_id: "synthetic-scalar-response".to_string(),
            domain_id: "synthetic-detection-channel-domain".to_string(),
            units: "dimensionless".to_string(),
            measurement_noise_model: MeasurementNoiseModel::Gaussian,
            measurement_sigma: Some(0.10),
            noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
            calibration_reference: Some("synthetic://measurement-noise/v1".to_string()),
            // Operational execution success is separate from model-dependent
            // probability of detection after execution.
            availability_probability: execution_probability,
            cost,
            exclusive_group: None,
            predictions,
        },
        detection_probabilities,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Three distinct channels make the decomposition visible:
    // 1. model-dependent detection plus different measured values;
    // 2. identical values, so all model information comes from detection;
    // 3. model-independent detection, which reduces to the old q * EIG rule.
    let hypotheses = vec![
        hypothesis("synthetic-model-a", b'a'),
        hypothesis("synthetic-model-b", b'b'),
        hypothesis("synthetic-model-c", b'c'),
    ];
    let prior_scenarios = vec![
        ModelPriorScenario {
            scenario_id: "balanced-prior".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![1.0, 1.0, 1.0],
        },
        ModelPriorScenario {
            scenario_id: "family-weighted-prior".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![3.0, 1.0, 1.0],
        },
    ];

    let experiments = vec![
        candidate(
            "detection-and-value",
            0.90,
            1.0,
            [0.0, 0.5, 1.25],
            [0.08, 0.10, 0.12],
            [0.95, 0.60, 0.15],
        ),
        candidate(
            "detection-only",
            0.80,
            0.8,
            [0.4, 0.4, 0.4],
            [0.12, 0.12, 0.12],
            [0.95, 0.55, 0.10],
        ),
        candidate(
            "model-independent-detection",
            0.55,
            1.5,
            [-0.4, 0.35, 1.1],
            [0.08, 0.10, 0.12],
            [0.75, 0.75, 0.75],
        ),
    ];

    let config = ModelDiscriminationConfig {
        model_universe: ModelUniverseCoverage::OpenWorld,
        prior_scenarios,
        quadrature_intervals: 256,
        quadrature_tolerance_nats: 1.0e-6,
    };
    let report =
        rank_detection_aware_model_discriminating_experiments(&hypotheses, &experiments, &config)?;

    println!(
        "epistemic_status=SYNTHETIC_BERNOULLI_GAUSSIAN_DETECTION_CHANNEL_NOT_PHYSICAL_VALIDATION"
    );
    println!(
        "observation_model=execution_success_then_model_conditional_detection_then_gaussian_value"
    );
    println!("open_world_unresolved_excluded=true");
    println!("model_count={}", report.model_ids.len());
    println!("prior_scenarios={}", report.prior_scenario_ids.join(";"));
    println!(
        "experiment,execution_probability,status,worst_case_total_eig,weighted_mean_total_eig,worst_case_eig_per_cost"
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
            "{},{:.3},{},{:.9},{:.9},{:.9}",
            score.experiment_id,
            score.execution_probability,
            status,
            score.worst_case_total_eig_nats.unwrap_or(f64::NAN),
            score.weighted_mean_total_eig_nats.unwrap_or(f64::NAN),
            score.worst_case_total_eig_per_cost.unwrap_or(f64::NAN),
        );
        for scenario in &score.prior_scenario_scores {
            println!(
                "scenario={},experiment={},prior_entropy={:.9},p_detect={:.9},I_model_detection={:.9},I_value_given_detection={},I_total_given_execution={:.9},EIG_after_execution={:.9},expected_posterior_entropy={:.9},conditional_quadrature_error={:.3e},operational_error_bound={:.3e}",
                scenario.scenario_id,
                score.experiment_id,
                scenario.prior_entropy_nats,
                scenario.expected_detection_probability,
                scenario.detection_indicator_eig_nats,
                scenario
                    .conditional_detected_value_eig_nats
                    .map(|value| format!("{value:.9}"))
                    .unwrap_or_else(|| "not-defined".to_string()),
                scenario.total_eig_given_execution_nats,
                scenario.availability_adjusted_total_eig_nats,
                scenario.expected_posterior_entropy_nats,
                scenario.quadrature_error_nats,
                scenario.operational_quadrature_error_bound_nats,
            );
        }
    }

    println!("claim_boundary=declared_synthetic_Bernoulli_Gaussian_two_stage_observation_channel");
    println!(
        "not_claimed=real_detection_calibration,physical_model_truth,model_set_completeness,UAP_identity,new_physics_or_propulsion"
    );
    Ok(())
}
