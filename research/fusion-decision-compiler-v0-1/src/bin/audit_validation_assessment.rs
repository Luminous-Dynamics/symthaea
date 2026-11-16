use fusion_decision_compiler_contract::validation_assessment::{
    BlindHoldoutAssessmentCase, MeasuredObservable, ModelRunIdentity,
    ObservableValidationSpec, PredictedObservable, ValidationAssessmentCampaign,
    ValidationAssessmentProfile, PredictiveAssessmentDisposition,
    assess_blind_holdout_campaign,
};
use fusion_decision_compiler_contract::validation_domain::{
    BlindHoldoutMeasurement, CalibrationExposure, HoldoutExposure, MeasuredObservation,
    ValidationDomain, ValidationPlan,
};
use fusion_decision_compiler_contract::{EvidenceClass, ModelUse};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn observation(
    record_id: &str,
    dataset_id: &str,
    artifact: char,
    facility_id: &str,
    case_id: &str,
    operating_point_id: &str,
    episode_id: &str,
) -> MeasuredObservation {
    MeasuredObservation {
        record_id: record_id.to_string(),
        evidence_class: EvidenceClass::MeasuredObservation,
        dataset_id: dataset_id.to_string(),
        source_artifact_sha256: sha(artifact),
        regime_id: "tokamak-hmode".to_string(),
        geometry_sha256: sha('b'),
        facility_id: facility_id.to_string(),
        case_id: case_id.to_string(),
        operating_point_id: operating_point_id.to_string(),
        episode_ids: vec![episode_id.to_string()],
        diagnostic_revision: "diagnostics-r3".to_string(),
        measurement_procedure_ref: "measurement-procedure-v1".to_string(),
        diagnostic_calibration_ref: "diagnostic-calibration-v1".to_string(),
        measurement_uncertainty_ref: "measurement-uncertainty-v1".to_string(),
    }
}

fn make_case(index: usize) -> BlindHoldoutAssessmentCase {
    let holdout_id = format!("holdout-{index}");
    let holdout_artifact = char::from(b'd' + index as u8);
    let calibration = CalibrationExposure {
        record_id: format!("calibration-{index}"),
        dataset_id: format!("calibration-dataset-{index}"),
        source_artifact_sha256: sha('c'),
        regime_id: "tokamak-hmode".to_string(),
        geometry_sha256: sha('b'),
        episode_ids: vec![format!("calibration-episode-{index}")],
        preprocessing_fit_id: "calibration-only-preprocessing-v1".to_string(),
    };
    let test_stand = observation(
        &format!("test-stand-{index}"),
        &format!("test-stand-dataset-{index}"),
        'z',
        "test-stand-facility",
        "test-stand-case",
        "test-stand-operating-point",
        &format!("test-stand-episode-{index}"),
    );
    let holdout = observation(
        &holdout_id,
        &format!("validation-dataset-{index}"),
        holdout_artifact,
        if index == 0 { "facility-a" } else { "facility-b" },
        &format!("validation-case-{index}"),
        &format!("operating-point-{index}"),
        &format!("validation-episode-{index}"),
    );

    let plan = ValidationPlan {
        model_revision: "transport-model-r17".to_string(),
        requested_use: ModelUse::SingleStepPrediction,
        domain: ValidationDomain {
            domain_id: "tokamak-hmode-domain-v1".to_string(),
            model_revision: "transport-model-r17".to_string(),
            regime_id: "tokamak-hmode".to_string(),
            supported_uses: vec![ModelUse::SingleStepPrediction],
            allowed_geometry_sha256: vec![sha('b')],
            allowed_facility_ids: vec![
                "test-stand-facility".to_string(),
                "facility-a".to_string(),
                "facility-b".to_string(),
            ],
            allowed_case_ids: vec![
                "test-stand-case".to_string(),
                format!("validation-case-{index}"),
            ],
            allowed_operating_point_ids: vec![
                "test-stand-operating-point".to_string(),
                format!("operating-point-{index}"),
            ],
        },
        calibration: vec![calibration],
        test_stand,
        blind_holdout: BlindHoldoutMeasurement {
            observation: holdout,
            exposure: HoldoutExposure {
                used_for_training: false,
                used_for_normalization_fit: false,
                used_for_feature_selection: false,
                used_for_hyperparameter_selection: false,
                used_for_model_or_policy_selection: false,
                used_for_calibration: false,
            },
            withheld_until_final_evaluation: true,
        },
    };

    let (measured, predicted, lower, upper) = match index {
        0 => (100.0, 105.0, 90.0, 110.0),
        1 => (120.0, 118.0, 110.0, 125.0),
        _ => (80.0, 100.0, 95.0, 105.0),
    };
    let model_run_id = format!("model-run-{index}");
    let model_output_sha256 = sha(char::from(b'j' + index as u8));

    BlindHoldoutAssessmentCase {
        plan,
        unseal_receipt_ref: format!("final-unseal-receipt-{index}"),
        unseal_receipt_sha256: sha('e'),
        model_run: ModelRunIdentity {
            run_id: model_run_id.clone(),
            model_revision: "transport-model-r17".to_string(),
            input_sha256: sha(char::from(b'g' + index as u8)),
            output_sha256: model_output_sha256.clone(),
            run_receipt_ref: format!("model-run-receipt-{index}"),
        },
        predictions: vec![PredictedObservable {
            observable_id: "net-heat-flux".to_string(),
            model_run_id,
            model_output_sha256,
            interval_method_ref: "synthetic-empirical-quantile-band-v1".to_string(),
            interval_assumption_ref: "synthetic-known-answer-fixture-v1".to_string(),
            unit_ref: "MW/m2".to_string(),
            service_boundary_ref: "divertor-target-surface-v1".to_string(),
            averaging_window_ref: "steady-state-1s-v1".to_string(),
            predicted_value: predicted,
            prediction_interval_lower: lower,
            prediction_interval_upper: upper,
            source_locator: format!("prediction[{index}].net-heat-flux"),
            evidence_ref: format!("prediction-evidence-{index}"),
        }],
        measurements: vec![MeasuredObservable {
            observable_id: "net-heat-flux".to_string(),
            unit_ref: "MW/m2".to_string(),
            service_boundary_ref: "divertor-target-surface-v1".to_string(),
            averaging_window_ref: "steady-state-1s-v1".to_string(),
            measured_value: measured,
            measurement_interval_lower: measured - 2.0,
            measurement_interval_upper: measured + 2.0,
            source_artifact_sha256: sha(holdout_artifact),
            source_locator: format!("measurement[{index}].net-heat-flux"),
            evidence_ref: format!("measurement-evidence-{index}"),
        }],
    }
}

fn main() {
    let campaign = ValidationAssessmentCampaign {
        profile: ValidationAssessmentProfile {
            profile_id: "synthetic-blind-holdout-profile-v1".to_string(),
            profile_sha256: sha('a'),
            domain_id: "tokamak-hmode-domain-v1".to_string(),
            model_revision: "transport-model-r17".to_string(),
            regime_id: "tokamak-hmode".to_string(),
            requested_use: ModelUse::SingleStepPrediction,
            minimum_holdout_count: 3,
            minimum_distinct_facilities: 2,
            minimum_distinct_operating_points: 3,
            observables: vec![ObservableValidationSpec {
                observable_id: "net-heat-flux".to_string(),
                unit_ref: "MW/m2".to_string(),
                service_boundary_ref: "divertor-target-surface-v1".to_string(),
                averaging_window_ref: "steady-state-1s-v1".to_string(),
                normalization_scale: 100.0,
                nominal_interval_coverage: 0.95,
                interval_method_ref: "synthetic-empirical-quantile-band-v1".to_string(),
                interval_assumption_ref: "synthetic-known-answer-fixture-v1".to_string(),
                minimum_empirical_point_coverage: 0.60,
                maximum_normalized_rmse: 0.20,
                maximum_absolute_normalized_bias: 0.10,
                maximum_mean_normalized_interval_width: 0.25,
            }],
        },
        cases: vec![make_case(0), make_case(1), make_case(2)],
    };

    let report = assess_blind_holdout_campaign(&campaign)
        .unwrap_or_else(|errors| {
            eprintln!("blind-holdout assessment refused: {errors:?}");
            std::process::exit(1);
        });
    println!("contract_version={}", report.contract_version);
    println!("profile={}", report.profile_id);
    println!("model_revision={}", report.model_revision);
    println!("holdout_count={}", report.holdout_count);
    println!("distinct_facility_count={}", report.distinct_facility_count);
    println!("distinct_operating_point_count={}", report.distinct_operating_point_count);
    println!("disposition={:?}", report.disposition);

    for metric in &report.per_observable_metrics {
        println!("observable={}", metric.observable_id);
        println!("sample_count={}", metric.sample_count);
        println!("mean_signed_error={:.6}", metric.mean_signed_error);
        println!("mae={:.6}", metric.mean_absolute_error);
        println!("rmse={:.6}", metric.root_mean_square_error);
        println!("normalized_bias={:.6}", metric.normalized_bias);
        println!("normalized_rmse={:.6}", metric.normalized_rmse);
        println!(
            "empirical_point_interval_coverage={:.6}",
            metric.empirical_point_interval_coverage
        );
        println!(
            "measurement_interval_overlap_coverage={:.6}",
            metric.measurement_interval_overlap_coverage
        );
        println!(
            "mean_normalized_interval_width={:.6}",
            metric.mean_normalized_prediction_interval_width
        );
    }

    println!(
        "SCOPE: synthetic split-integrity and predictive-metric fixture only; no physical model validation or statistical confidence claim"
    );

    if report.disposition != PredictiveAssessmentDisposition::WithinDeclaredAcceptance {
        eprintln!("synthetic known-answer validation profile did not meet declared thresholds");
        std::process::exit(1);
    }
}
