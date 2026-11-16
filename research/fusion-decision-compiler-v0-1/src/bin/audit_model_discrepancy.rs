use fusion_decision_compiler_contract::cross_solver_comparison::ApplicabilityState;
use fusion_decision_compiler_contract::model_discrepancy::{
    ApplicabilityReceipt, CalibrationExposure, CalibrationResidual,
    DiscrepancyCalibrationProfile, DiscrepancyTarget, ModelDiscrepancyDisposition,
    ParametricLowerBoundReceipt, evaluate_model_discrepancy,
};
use fusion_decision_compiler_contract::plant_service_gate::{
    PowerBalanceProfile, PowerBalanceRun, PowerObservable, PowerTermEvidence,
};
use fusion_decision_compiler_contract::robust_net_export::ModelFormValidity;
use fusion_decision_compiler_contract::{EvidenceClass, ModelUse};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn power_profile() -> PowerBalanceProfile {
    PowerBalanceProfile {
        schema_ref: "synthetic-fusion-power-schema-v1".to_string(),
        schema_sha256: sha('a'),
        service_boundary_ref: "synthetic-plant-net-export-v1".to_string(),
        unit_ref: "MWe".to_string(),
        averaging_window_ref: "synthetic-steady-state-60s-v1".to_string(),
        sign_convention_ref: "net-export-positive-v1".to_string(),
        absolute_balance_tolerance_mw: 0.001,
        required_net_export_reserve_mw: 50.0,
        reserve_threshold_inclusive: true,
    }
}

fn power_run(profile: &PowerBalanceProfile) -> PowerBalanceRun {
    PowerBalanceRun {
        run_id: "synthetic-fusion-net-run-1".to_string(),
        solver_id: "synthetic-systems-code".to_string(),
        solver_revision: "synthetic-solver-r1".to_string(),
        candidate_revision: "synthetic-candidate-r1".to_string(),
        output_sha256: sha('b'),
        schema_ref: profile.schema_ref.clone(),
        schema_sha256: profile.schema_sha256.clone(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        unit_ref: profile.unit_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        receipt_ref: "synthetic-solver-run-receipt-1".to_string(),
        terms: vec![PowerTermEvidence {
            observable: PowerObservable::NetElectric,
            value_mw: 150.0,
            source_locator: "synthetic.outputs.P_elec".to_string(),
            evidence_ref: "synthetic-net-electric-output".to_string(),
        }],
    }
}

fn discrepancy_profile(power: &PowerBalanceProfile) -> DiscrepancyCalibrationProfile {
    DiscrepancyCalibrationProfile {
        profile_id: "synthetic-net-power-discrepancy-v1".to_string(),
        profile_sha256: sha('c'),
        solver_id: "synthetic-systems-code".to_string(),
        solver_revision: "synthetic-solver-r1".to_string(),
        validation_domain_id: "synthetic-spherical-tokamak-domain-v1".to_string(),
        validation_domain_sha256: sha('d'),
        regime_id: "synthetic-steady-state-spherical-tokamak".to_string(),
        observable_id: "P_elec".to_string(),
        quantity_ref: "net-electric-output".to_string(),
        unit_ref: "MWe".to_string(),
        service_boundary_ref: power.service_boundary_ref.clone(),
        averaging_window_ref: power.averaging_window_ref.clone(),
        sign_convention_ref: power.sign_convention_ref.clone(),
        allowed_geometry_sha256: vec![sha('e')],
        calibration_dataset_id: "synthetic-calibration-dataset-v1".to_string(),
        calibration_manifest_sha256: sha('f'),
        exchangeability_scope_ref: "synthetic-one-residual-per-independent-shot-v1".to_string(),
        calibration_method_ref: "split-conformal-absolute-residual-v1".to_string(),
        discrepancy_coverage_basis_points: 9750,
        required_total_coverage_basis_points: 9500,
        minimum_calibration_samples: 19,
        maximum_discrepancy_mwe: 25.0,
        discrepancy_tolerance_inclusive: true,
        evaluation_envelope_ref: "synthetic-model-form-envelope-1".to_string(),
        evaluation_envelope_sha256: sha('g'),
    }
}

fn target() -> DiscrepancyTarget {
    DiscrepancyTarget {
        candidate_revision: "synthetic-candidate-r1".to_string(),
        input_sha256: sha('h'),
        geometry_sha256: sha('e'),
        regime_id: "synthetic-steady-state-spherical-tokamak".to_string(),
        observable_id: "P_elec".to_string(),
        quantity_ref: "net-electric-output".to_string(),
        requested_use: ModelUse::EngineeringScreening,
        point_estimate_mwe: 150.0,
    }
}

fn residual(index: usize) -> CalibrationResidual {
    CalibrationResidual {
        record_id: format!("synthetic-residual-{index}"),
        dataset_id: "synthetic-calibration-dataset-v1".to_string(),
        independence_cluster_id: format!("synthetic-cluster-{index}"),
        episode_id: format!("synthetic-shot-{index}"),
        facility_id: "synthetic-facility".to_string(),
        case_id: format!("synthetic-case-{index}"),
        operating_point_id: format!("synthetic-operating-point-{index}"),
        solver_id: "synthetic-systems-code".to_string(),
        model_revision: "synthetic-solver-r1".to_string(),
        model_run_id: format!("synthetic-calibration-run-{index}"),
        model_output_sha256: sha('i'),
        observation_id: format!("synthetic-observation-{index}"),
        observation_artifact_sha256: sha('j'),
        evidence_class: EvidenceClass::SyntheticFixture,
        measurement_uncertainty_ref: format!("synthetic-measurement-uncertainty-{index}"),
        validation_domain_id: "synthetic-spherical-tokamak-domain-v1".to_string(),
        validation_domain_sha256: sha('d'),
        regime_id: "synthetic-steady-state-spherical-tokamak".to_string(),
        geometry_sha256: sha('e'),
        observable_id: "P_elec".to_string(),
        quantity_ref: "net-electric-output".to_string(),
        unit_ref: "MWe".to_string(),
        service_boundary_ref: "synthetic-plant-net-export-v1".to_string(),
        averaging_window_ref: "synthetic-steady-state-60s-v1".to_string(),
        sign_convention_ref: "net-export-positive-v1".to_string(),
        absolute_residual_mwe: index as f64 / 2.0,
        exposure: CalibrationExposure {
            used_for_model_fit: false,
            used_for_normalization_fit: false,
            used_for_feature_selection: false,
            used_for_hyperparameter_selection: false,
            used_for_model_selection: false,
            used_for_policy_selection: false,
        },
    }
}

fn applicability(run: &PowerBalanceRun) -> ApplicabilityReceipt {
    ApplicabilityReceipt {
        receipt_ref: "synthetic-applicability-receipt".to_string(),
        receipt_sha256: sha('k'),
        evidence_ref: "synthetic-applicability-evidence".to_string(),
        evidence_sha256: sha('l'),
        solver_id: run.solver_id.clone(),
        solver_revision: run.solver_revision.clone(),
        run_id: run.run_id.clone(),
        input_sha256: sha('h'),
        output_sha256: run.output_sha256.clone(),
        candidate_revision: run.candidate_revision.clone(),
        geometry_sha256: sha('e'),
        regime_id: "synthetic-steady-state-spherical-tokamak".to_string(),
        validation_domain_id: "synthetic-spherical-tokamak-domain-v1".to_string(),
        validation_domain_sha256: sha('d'),
        requested_use: ModelUse::EngineeringScreening,
        state: ApplicabilityState::InDeclaredDomain,
        evidence_class: EvidenceClass::IndependentVerification,
        method_ref: "synthetic-domain-classifier-v1".to_string(),
    }
}

fn parametric(run: &PowerBalanceRun, power: &PowerBalanceProfile) -> ParametricLowerBoundReceipt {
    ParametricLowerBoundReceipt {
        receipt_ref: "synthetic-parametric-lower-bound".to_string(),
        receipt_sha256: sha('m'),
        evidence_ref: "synthetic-parametric-uq-evidence".to_string(),
        evidence_sha256: sha('n'),
        method_ref: "synthetic-joint-covariance-contrast-v1".to_string(),
        assumption_manifest_sha256: sha('o'),
        run_id: run.run_id.clone(),
        input_sha256: sha('h'),
        output_sha256: run.output_sha256.clone(),
        schema_ref: power.schema_ref.clone(),
        schema_sha256: power.schema_sha256.clone(),
        service_boundary_ref: power.service_boundary_ref.clone(),
        unit_ref: power.unit_ref.clone(),
        averaging_window_ref: power.averaging_window_ref.clone(),
        sign_convention_ref: power.sign_convention_ref.clone(),
        lower_bound_mwe: 120.0,
        one_sided_coverage_basis_points: 9750,
        coverage_scope_ref: "synthetic-single-run-coverage-scope-v1".to_string(),
    }
}

fn main() {
    let power = power_profile();
    let run = power_run(&power);
    let profile = discrepancy_profile(&power);
    let target = target();
    let residuals: Vec<_> = (1..=39).map(residual).collect();
    let report = evaluate_model_discrepancy(
        &profile,
        &target,
        &power,
        &run,
        &residuals,
        &applicability(&run),
        Some(&parametric(&run, &power)),
    );

    println!("contract_version={}", report.contract_version);
    println!("disposition={:?}", report.disposition);
    println!("model_form_validity={:?}", report.model_form_validity);
    println!("calibration_samples={}", report.calibration_sample_count);
    println!("conformal_rank={}", report.conformal_rank.unwrap_or_default());
    println!(
        "discrepancy_bound_mwe={:.6}",
        report.discrepancy_bound_mwe.unwrap_or(f64::NAN)
    );
    println!(
        "parametric_lower_bound_mwe={:.6}",
        report.parametric_lower_bound_mwe.unwrap_or(f64::NAN)
    );
    println!(
        "adjusted_lower_bound_mwe={:.6}",
        report.adjusted_lower_bound_mwe.unwrap_or(f64::NAN)
    );
    println!(
        "combined_coverage_floor_basis_points={}",
        report.combined_coverage_floor_basis_points.unwrap_or_default()
    );
    println!(
        "required_total_coverage_basis_points={}",
        report.required_total_coverage_basis_points
    );
    println!(
        "SCOPE: synthetic residuals, applicability and parametric receipt; illustrative arithmetic only, never a model-form qualification"
    );

    if report.disposition != ModelDiscrepancyDisposition::SyntheticOnly
        || report.model_form_validity != ModelFormValidity::Unknown
        || report.adjusted_lower_bound_mwe != Some(100.5)
    {
        eprintln!("synthetic model-discrepancy fixture did not remain synthetic-only");
        std::process::exit(1);
    }
}
