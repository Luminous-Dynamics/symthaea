use fusion_decision_compiler_contract::EvidenceClass;
use fusion_decision_compiler_contract::ModelUse;
use fusion_decision_compiler_contract::measured_holdout_metrics::{
    CommitmentEvidenceClass, HoldoutEvaluationCampaign, HoldoutEvaluationDisposition,
    HoldoutMetricProfile, MeasurementAttestationClass, MeasurementVerificationReceipt,
    PairedValidationSample, PredictionCommitment, PredictionInterval, evaluate_measured_holdout,
};
use fusion_decision_compiler_contract::validation_domain::{
    BlindHoldoutMeasurement, CalibrationExposure, HoldoutExposure, MeasuredObservation,
    ValidationDomain, ValidationPlan,
};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn observation(
    record_id: &str,
    dataset_id: &str,
    artifact: char,
    facility: &str,
    case_id: &str,
    operating_point_id: &str,
    episode_id: &str,
) -> MeasuredObservation {
    MeasuredObservation {
        record_id: record_id.into(),
        evidence_class: EvidenceClass::MeasuredObservation,
        dataset_id: dataset_id.into(),
        source_artifact_sha256: digest(artifact),
        regime_id: "synthetic-tokamak-regime".into(),
        geometry_sha256: digest('b'),
        facility_id: facility.into(),
        case_id: case_id.into(),
        operating_point_id: operating_point_id.into(),
        episode_ids: vec![episode_id.into()],
        diagnostic_revision: "synthetic-diagnostic-r1".into(),
        measurement_procedure_ref: "synthetic-procedure-r1".into(),
        diagnostic_calibration_ref: "synthetic-calibration-r1".into(),
        measurement_uncertainty_ref: "synthetic-measurement-uncertainty-r1".into(),
    }
}

fn sample(
    id: &str,
    group: &str,
    record: &str,
    artifact: char,
    output: char,
    measured: f64,
    predicted: f64,
) -> PairedValidationSample {
    PairedValidationSample {
        sample_id: id.into(),
        group_id: group.into(),
        observation_record_id: record.into(),
        observable_id: "plasma-current".into(),
        unit_ref: "MA".into(),
        coordinate_frame_ref: "synthetic-magnetic-axis-frame-v1".into(),
        sampling_window_ref: "synthetic-steady-window-1s-v1".into(),
        measured_value: measured,
        measurement_standard_uncertainty: Some(0.05),
        predicted_value: Some(predicted),
        prediction_interval: Some(PredictionInterval {
            lower: predicted - 0.15,
            upper: predicted + 0.15,
        }),
        measurement_artifact_sha256: digest(artifact),
        prediction_output_sha256: digest(output),
        measurement_evidence_ref: format!("synthetic-measurement-{id}"),
        prediction_evidence_ref: format!("synthetic-prediction-{id}"),
    }
}

fn main() {
    let validation_plan = ValidationPlan {
        model_revision: "synthetic-model-r1".into(),
        requested_use: ModelUse::SingleStepPrediction,
        domain: ValidationDomain {
            domain_id: "synthetic-tokamak-validation-domain-v1".into(),
            model_revision: "synthetic-model-r1".into(),
            regime_id: "synthetic-tokamak-regime".into(),
            supported_uses: vec![ModelUse::SingleStepPrediction],
            allowed_geometry_sha256: vec![digest('b')],
            allowed_facility_ids: vec!["facility-a".into(), "facility-b".into()],
            allowed_case_ids: vec!["case-a".into(), "case-b".into()],
            allowed_operating_point_ids: vec!["op-a".into(), "op-b".into()],
        },
        calibration: vec![CalibrationExposure {
            record_id: "synthetic-calibration-record".into(),
            dataset_id: "synthetic-calibration-dataset".into(),
            source_artifact_sha256: digest('c'),
            regime_id: "synthetic-tokamak-regime".into(),
            geometry_sha256: digest('b'),
            episode_ids: vec!["synthetic-calibration-shot".into()],
            preprocessing_fit_id: "synthetic-fit-calibration-only".into(),
        }],
        test_stand: observation(
            "synthetic-test-stand",
            "synthetic-test-dataset",
            'd',
            "facility-a",
            "case-a",
            "op-a",
            "synthetic-test-shot",
        ),
        blind_holdout: BlindHoldoutMeasurement {
            observation: observation(
                "synthetic-blind-holdout",
                "synthetic-holdout-dataset",
                'e',
                "facility-b",
                "case-b",
                "op-b",
                "synthetic-holdout-shot",
            ),
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
    let metric_profile = HoldoutMetricProfile {
        profile_id: "synthetic-plasma-current-validation-metrics-v1".into(),
        profile_sha256: digest('f'),
        observable_id: "plasma-current".into(),
        unit_ref: "MA".into(),
        coordinate_frame_ref: "synthetic-magnetic-axis-frame-v1".into(),
        sampling_window_ref: "synthetic-steady-window-1s-v1".into(),
        normalized_error_scale: 1.0,
        absolute_error_tolerance: 0.2,
        maximum_mean_absolute_error: 0.15,
        maximum_root_mean_square_error: 0.2,
        maximum_absolute_error: 0.3,
        maximum_failed_predictions_per_set: 0,
        minimum_holdout_group_count: 2,
        minimum_empirical_interval_coverage: Some(0.8),
        maximum_mean_interval_width: Some(0.5),
    };
    let prediction_commitment = PredictionCommitment {
        commitment_id: "synthetic-prediction-commitment-v1".into(),
        commitment_ref: "synthetic-commitment-receipt".into(),
        commitment_sha256: digest('1'),
        model_revision: "synthetic-model-r1".into(),
        model_config_sha256: digest('2'),
        input_manifest_sha256: digest('3'),
        test_stand_output_sha256: digest('4'),
        holdout_output_sha256: digest('5'),
        observable_profile_sha256: digest('f'),
        test_stand_record_id: "synthetic-test-stand".into(),
        holdout_record_id: "synthetic-blind-holdout".into(),
        test_stand_artifact_sha256: digest('d'),
        holdout_artifact_sha256: digest('e'),
        expected_test_stand_sample_count: 2,
        expected_holdout_sample_count: 2,
        committed_before_test_stand_reveal: true,
        committed_before_holdout_reveal: true,
        commitment_evidence_class: CommitmentEvidenceClass::SyntheticFixture,
        timestamp_authority_ref: "synthetic-timestamp-placeholder".into(),
    };
    let verification = |record: &str, artifact: char| MeasurementVerificationReceipt {
        observation_record_id: record.into(),
        source_artifact_sha256: digest(artifact),
        verifier_ref: "synthetic-verifier-placeholder".into(),
        attestation_ref: format!("synthetic-attestation-{record}"),
        attestation_sha256: digest('6'),
        evidence_class: MeasurementAttestationClass::SyntheticFixture,
    };
    let campaign = HoldoutEvaluationCampaign {
        validation_plan,
        metric_profile,
        prediction_commitment,
        test_stand_measurement_verification: verification("synthetic-test-stand", 'd'),
        holdout_measurement_verification: verification("synthetic-blind-holdout", 'e'),
        test_stand_samples: vec![
            sample(
                "test-a",
                "test-group-a",
                "synthetic-test-stand",
                'd',
                '4',
                10.0,
                10.05,
            ),
            sample(
                "test-b",
                "test-group-b",
                "synthetic-test-stand",
                'd',
                '4',
                11.0,
                11.08,
            ),
        ],
        holdout_samples: vec![
            sample(
                "holdout-a",
                "holdout-group-a",
                "synthetic-blind-holdout",
                'e',
                '5',
                12.0,
                12.04,
            ),
            sample(
                "holdout-b",
                "holdout-group-b",
                "synthetic-blind-holdout",
                'e',
                '5',
                13.0,
                13.07,
            ),
        ],
    };

    let report = evaluate_measured_holdout(&campaign);
    let test = report
        .test_stand_metrics
        .as_ref()
        .expect("synthetic test metrics");
    let holdout = report
        .holdout_metrics
        .as_ref()
        .expect("synthetic holdout metrics");
    println!("contract_version={}", report.contract_version);
    println!("disposition={:?}", report.disposition);
    println!("test_stand_samples={}", test.sample_count);
    println!(
        "test_stand_mae={:.6}",
        test.mean_absolute_error.unwrap_or(f64::NAN)
    );
    println!(
        "test_stand_rmse={:.6}",
        test.root_mean_square_error.unwrap_or(f64::NAN)
    );
    println!("holdout_samples={}", holdout.sample_count);
    println!("holdout_groups={}", holdout.unique_group_count);
    println!(
        "holdout_mae={:.6}",
        holdout.mean_absolute_error.unwrap_or(f64::NAN)
    );
    println!(
        "holdout_rmse={:.6}",
        holdout.root_mean_square_error.unwrap_or(f64::NAN)
    );
    println!(
        "holdout_interval_empirical_coverage={:.6}",
        holdout
            .empirical_prediction_interval_coverage
            .unwrap_or(f64::NAN)
    );
    println!("claim_ceiling={}", report.claim_ceiling);
    println!(
        "SCOPE: all values, dataset identities, timestamps and verification receipts in this runner are synthetic placeholders; the expected result is DescriptiveOnly, not physical validation"
    );

    if report.disposition != HoldoutEvaluationDisposition::DescriptiveOnly {
        eprintln!("synthetic validation fixture must never be promoted to a measured holdout pass");
        std::process::exit(1);
    }
}
