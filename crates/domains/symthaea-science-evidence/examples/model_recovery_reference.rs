// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_science_evidence::{
    CandidateExperiment, DiscriminationHypothesis, GaussianModelPrediction, MeasurementNoiseModel,
    ModelPriorScenario, NoiseCalibrationStatus, PredictionDisposition, RecoveryGenerator,
    evaluate_model_recovery,
};

fn hypothesis(id: &str, byte: u8) -> DiscriminationHypothesis {
    DiscriminationHypothesis {
        model_id: id.to_string(),
        source_revision_sha: std::iter::repeat_n(byte as char, 40).collect(),
        specification_reference: format!("synthetic://recovery-model/{id}/v1"),
    }
}

fn prediction(id: &str, mean: f64, sigma: f64) -> GaussianModelPrediction {
    GaussianModelPrediction {
        model_id: id.to_string(),
        disposition: PredictionDisposition::Quantitative,
        predictive_mean: Some(mean),
        predictive_sigma: Some(sigma),
        derivation_reference: Some(format!("synthetic://recovery-prediction/{id}/v1")),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // This is a deterministic synthetic model-recovery fixture. It is not a
    // physical theory comparison, and the extra generator is not an inference
    // candidate: it tests whether an open-world observation triggers abstention.
    let hypotheses = vec![
        hypothesis("synthetic-model-a", b'a'),
        hypothesis("synthetic-model-b", b'b'),
        hypothesis("synthetic-model-c", b'c'),
    ];
    let experiment = CandidateExperiment {
        experiment_id: "synthetic-model-recovery".to_string(),
        observable_id: "synthetic-scalar-observable".to_string(),
        domain_id: "synthetic-known-answer-domain".to_string(),
        units: "dimensionless".to_string(),
        measurement_noise_model: MeasurementNoiseModel::Gaussian,
        measurement_sigma: Some(0.05),
        noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
        calibration_reference: Some("synthetic://calibration/v1".to_string()),
        availability_probability: 1.0,
        cost: 1.0,
        exclusive_group: None,
        predictions: vec![
            prediction("synthetic-model-a", -1.0, 0.1),
            prediction("synthetic-model-b", 0.0, 0.1),
            prediction("synthetic-model-c", 1.0, 0.1),
        ],
    };
    let prior = ModelPriorScenario {
        scenario_id: "equal-prior".to_string(),
        scenario_weight: 1.0,
        probabilities: vec![1.0, 1.0, 1.0],
    };
    let generators = vec![
        RecoveryGenerator {
            generator_id: "synthetic-model-a".to_string(),
            predictive_mean: -1.0,
            predictive_sigma: 0.1,
        },
        RecoveryGenerator {
            generator_id: "synthetic-model-b".to_string(),
            predictive_mean: 0.0,
            predictive_sigma: 0.1,
        },
        RecoveryGenerator {
            generator_id: "synthetic-model-c".to_string(),
            predictive_mean: 1.0,
            predictive_sigma: 0.1,
        },
        RecoveryGenerator {
            generator_id: "OTHER_UNRESOLVED".to_string(),
            predictive_mean: 0.45,
            predictive_sigma: 0.1,
        },
    ];

    let report = evaluate_model_recovery(
        &hypotheses,
        &experiment,
        &prior,
        &generators,
        2_000,
        0x5C1E_2026,
        0.90,
    )?;

    println!("epistemic_status=DETERMINISTIC_SYNTHETIC_MODEL_RECOVERY_NOT_PHYSICAL_VALIDATION");
    println!("experiment_id={}", report.experiment_id);
    println!("prior_scenario={}", report.prior_scenario_id);
    println!("samples_per_generator={}", report.samples_per_generator);
    println!("abstention_threshold={:.6}", report.abstention_threshold);
    println!("inference_models={}", report.model_ids.join(";"));
    println!("generators={}", report.generator_ids.join(";"));
    println!(
        "confusion_matrix_columns={},ABSTAIN",
        report.model_ids.join(",")
    );
    for (generator_id, row) in report
        .generator_ids
        .iter()
        .zip(report.confusion_counts.iter())
    {
        let row_text = row
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("confusion_row={generator_id},{row_text}");
    }
    println!("named_model_accuracy={:.9}", report.named_model_accuracy);
    println!(
        "prior_weighted_named_model_accuracy={:.9}",
        report.prior_weighted_named_model_accuracy
    );
    println!("named_model_coverage={:.9}", report.named_model_coverage);
    println!(
        "named_model_selective_accuracy={}",
        report
            .named_model_selective_accuracy
            .map(|value| format!("{value:.9}"))
            .unwrap_or_else(|| "undefined".to_string())
    );
    println!(
        "open_world_false_confident_assignment_rate={}",
        report
            .open_world_false_confident_assignment_rate
            .map(|value| format!("{value:.9}"))
            .unwrap_or_else(|| "not_tested".to_string())
    );
    println!(
        "open_world_abstention_rate={}",
        report
            .open_world_abstention_rate
            .map(|value| format!("{value:.9}"))
            .unwrap_or_else(|| "not_tested".to_string())
    );
    println!(
        "claim_boundary=known_synthetic_scalar_gaussian_generator_and_frozen_MAP_abstention_rule"
    );
    println!(
        "not_claimed=general_Bayesian_recovery,physical_model_truth,model_set_completeness,parameter_EIG"
    );
    Ok(())
}
