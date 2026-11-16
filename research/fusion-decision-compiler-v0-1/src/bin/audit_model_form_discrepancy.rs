use fusion_decision_compiler_contract::model_form_discrepancy::{
    ApplicabilityReceipt, ApplicabilityState, CalibrationResidual, ConformalRankError,
    ModelFormDiscrepancyPlan, ModelFormDiscrepancyProfile, ModelFormDisposition,
    ModelFormEvidenceClass, ModelUseIdentity, ParametricNetPowerLowerBound,
    conformal_absolute_residual_allowance, evaluate_model_form_discrepancy,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn digest(value: usize) -> String {
    format!("{value:064x}")
}

fn residual(
    profile: &ModelFormDiscrepancyProfile,
    index: usize,
    evidence_class: ModelFormEvidenceClass,
) -> CalibrationResidual {
    CalibrationResidual {
        record_id: format!("synthetic-residual-{index}"),
        source_case_id: format!("synthetic-case-{index}"),
        episode_id: format!("synthetic-episode-{index}"),
        facility_id: "synthetic-facility".to_string(),
        solver_id: profile.model_family.clone(),
        solver_revision: profile.solver_revision.clone(),
        model_run_id: format!("synthetic-model-run-{index}"),
        model_output_sha256: digest(index + 1),
        measured_observation_id: format!("synthetic-measurement-{index}"),
        measured_artifact_sha256: digest(index + 100),
        observable_id: profile.observable_id.clone(),
        unit_ref: profile.unit_ref.clone(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        regime_ref: profile.regime_ref.clone(),
        geometry_sha256: profile.geometry_sha256.clone(),
        absolute_residual_mwe: 1.0 + index as f64 * 0.01,
        evidence_class,
        metrology_reference: format!("synthetic-metrology-{index}"),
        evidence_ref: format!("synthetic-residual-evidence-{index}"),
        used_for_model_fit: false,
        used_for_normalization: false,
        used_for_feature_selection: false,
        used_for_hyperparameter_selection: false,
        used_for_model_selection: false,
    }
}

fn main() {
    let nineteen: Vec<f64> = (1..=19).map(|value| value as f64).collect();
    let rank_95 = conformal_absolute_residual_allowance(&nineteen, 0.95).unwrap();
    println!(
        "known_answer_n19_coverage=0.950,rank={},allowance_mwe={:.3}",
        rank_95.rank_one_based, rank_95.value_mwe
    );

    let profile = ModelFormDiscrepancyProfile {
        profile_id: "synthetic-D0FUS-net-power-discrepancy-v1".to_string(),
        profile_sha256: sha('a'),
        model_family: "D0FUS".to_string(),
        solver_revision: "d0fus-3.2.0-a682c351".to_string(),
        observable_id: "P_elec".to_string(),
        unit_ref: "MWe".to_string(),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        averaging_window_ref: "iter-run-steady-state-window-v1".to_string(),
        sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        regime_ref: "synthetic-ITER-like-regime-v1".to_string(),
        geometry_sha256: sha('b'),
        validity_domain_ref: "synthetic-validity-domain-v1".to_string(),
        validity_domain_sha256: sha('c'),
        requested_model_use_ref: "synthetic-reference-prediction-v1".to_string(),
        target_total_coverage: 0.95,
        parametric_coverage: 0.975,
        discrepancy_coverage: 0.975,
        material_discrepancy_tolerance: 25.0,
        tolerance_inclusive: true,
        minimum_calibration_samples: 19,
        calibration_dataset_manifest_sha256: sha('d'),
        exchangeability_scope_ref: "synthetic-independent-episode-split-v1".to_string(),
    };
    let target = ModelUseIdentity {
        run_id: "synthetic-target-run-900".to_string(),
        candidate_revision: "synthetic-candidate-r1".to_string(),
        solver_id: profile.model_family.clone(),
        solver_revision: profile.solver_revision.clone(),
        input_sha256: sha('e'),
        output_sha256: sha('f'),
        schema_ref: "synthetic-net-power-schema-v1".to_string(),
        schema_sha256: sha('a'),
        observable_id: profile.observable_id.clone(),
        point_estimate_mwe: 120.0,
        unit_ref: profile.unit_ref.clone(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        regime_ref: profile.regime_ref.clone(),
        geometry_sha256: profile.geometry_sha256.clone(),
        requested_model_use_ref: profile.requested_model_use_ref.clone(),
    };

    let plan = ModelFormDiscrepancyPlan {
        profile: profile.clone(),
        target: target.clone(),
        calibration_residuals: (0..39)
            .map(|index| residual(&profile, index, ModelFormEvidenceClass::SyntheticFixture))
            .collect(),
        applicability: ApplicabilityReceipt {
            receipt_ref: "synthetic-applicability-receipt".to_string(),
            receipt_sha256: sha('b'),
            run_id: target.run_id.clone(),
            solver_revision: target.solver_revision.clone(),
            input_sha256: target.input_sha256.clone(),
            regime_ref: target.regime_ref.clone(),
            geometry_sha256: target.geometry_sha256.clone(),
            requested_model_use_ref: target.requested_model_use_ref.clone(),
            validity_domain_ref: profile.validity_domain_ref.clone(),
            validity_domain_sha256: profile.validity_domain_sha256.clone(),
            method_profile_ref: profile.profile_id.clone(),
            evidence_ref: "synthetic-applicability-evidence".to_string(),
            evidence_sha256: sha('c'),
            state: ApplicabilityState::InDeclaredDomain,
        },
        parametric_bound: ParametricNetPowerLowerBound {
            receipt_ref: "synthetic-parametric-lower-bound".to_string(),
            receipt_sha256: sha('d'),
            run_id: target.run_id.clone(),
            output_sha256: target.output_sha256.clone(),
            schema_ref: target.schema_ref.clone(),
            schema_sha256: target.schema_sha256.clone(),
            observable_id: target.observable_id.clone(),
            lower_bound_mwe: 110.0,
            one_sided_coverage: 0.975,
            coverage_scope_ref: "synthetic-parametric-coverage-scope".to_string(),
            method_ref: "synthetic-joint-parametric-net-power-uq".to_string(),
            assumption_manifest_sha256: sha('e'),
            unit_ref: target.unit_ref.clone(),
            service_boundary_ref: target.service_boundary_ref.clone(),
            averaging_window_ref: target.averaging_window_ref.clone(),
            sign_convention_ref: target.sign_convention_ref.clone(),
            evidence_ref: "synthetic-parametric-evidence".to_string(),
            evidence_sha256: sha('f'),
        },
        decision_receipt_ref: "synthetic-model-form-decision-receipt".to_string(),
        decision_receipt_sha256: sha('a'),
    };

    let report = evaluate_model_form_discrepancy(&plan);
    println!("contract_version={}", report.contract_version);
    println!(
        "calibration_sample_count={}",
        report.calibration_sample_count
    );
    println!(
        "combined_coverage_floor={:.3}",
        report.combined_coverage_floor.unwrap_or(f64::NAN)
    );
    println!("model_form_disposition={:?}", report.disposition);
    println!("model_form_validity={:?}", report.model_form_validity);
    println!(
        "downstream_robust_receipt_emitted={}",
        report.to_net_export_receipt(&plan).is_some()
    );
    println!(
        "SCOPE: residuals and receipts are synthetic software fixtures; no measured fusion calibration or physical model-discrepancy claim"
    );

    if rank_95.rank_one_based != 19
        || report.disposition != ModelFormDisposition::SyntheticOnly
        || report.to_net_export_receipt(&plan).is_some()
    {
        eprintln!("model-form discrepancy synthetic contract fixture failed");
        std::process::exit(1);
    }

    let insufficient = conformal_absolute_residual_allowance(&nineteen[..18], 0.95);
    if insufficient != Err(ConformalRankError::RankUnavailable) {
        eprintln!("18-sample 95% conformal control did not refuse the unavailable rank");
        std::process::exit(1);
    }
}
