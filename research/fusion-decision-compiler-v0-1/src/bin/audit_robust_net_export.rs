use fusion_decision_compiler_contract::plant_service_gate::{
    PowerBalanceProfile, PowerBalanceRun, PowerObservable, PowerTermEvidence,
    evaluate_plant_power_balance,
};
use fusion_decision_compiler_contract::robust_net_export::{
    ModelFormValidity, NetExportBoundMethod, NetExportEvidenceClass,
    NetExportLowerBoundReceipt, RobustNetExportDisposition, RobustNetExportProfile,
    evaluate_robust_net_export,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn main() {
    let profile = PowerBalanceProfile {
        schema_ref: "synthetic-process-net-power-schema-v1".to_string(),
        schema_sha256: sha('a'),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        averaging_window_ref: "steady-state-60s-v1".to_string(),
        sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
        absolute_balance_tolerance_mw: 0.001,
        required_net_export_reserve_mw: 50.0,
        reserve_threshold_inclusive: true,
    };

    let run = PowerBalanceRun {
        run_id: "synthetic-run-robust-export-001".to_string(),
        solver_id: "synthetic-systems-code".to_string(),
        solver_revision: "synthetic-revision-r1".to_string(),
        candidate_revision: "synthetic-candidate-r1".to_string(),
        output_sha256: sha('b'),
        schema_ref: profile.schema_ref.clone(),
        schema_sha256: profile.schema_sha256.clone(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        unit_ref: profile.unit_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        receipt_ref: "synthetic-solver-run-receipt".to_string(),
        terms: vec![
            PowerTermEvidence {
                observable: PowerObservable::GrossElectric,
                value_mw: 300.0,
                source_locator: "synthetic.outputs.gross".to_string(),
                evidence_ref: "synthetic-gross-evidence".to_string(),
            },
            PowerTermEvidence {
                observable: PowerObservable::VariableRecirculated,
                value_mw: 100.0,
                source_locator: "synthetic.outputs.variable_recirc".to_string(),
                evidence_ref: "synthetic-variable-evidence".to_string(),
            },
            PowerTermEvidence {
                observable: PowerObservable::FixedRecirculated,
                value_mw: 50.0,
                source_locator: "synthetic.outputs.fixed_recirc".to_string(),
                evidence_ref: "synthetic-fixed-evidence".to_string(),
            },
            PowerTermEvidence {
                observable: PowerObservable::TotalRecirculated,
                value_mw: 150.0,
                source_locator: "synthetic.outputs.total_recirc".to_string(),
                evidence_ref: "synthetic-total-evidence".to_string(),
            },
            PowerTermEvidence {
                observable: PowerObservable::NetElectric,
                value_mw: 150.0,
                source_locator: "synthetic.outputs.net".to_string(),
                evidence_ref: "synthetic-net-evidence".to_string(),
            },
        ],
    };
    let point = evaluate_plant_power_balance(&profile, &run);
    let robust_profile = RobustNetExportProfile {
        required_one_sided_coverage_level: 0.95,
        numerical_tolerance_mw: 0.001,
    };

    for lower_bound in [20.0, 60.0] {
        let receipt = NetExportLowerBoundReceipt {
            receipt_id: format!("synthetic-net-uq-receipt-{lower_bound:.0}"),
            run_id: run.run_id.clone(),
            output_sha256: run.output_sha256.clone(),
            schema_ref: profile.schema_ref.clone(),
            schema_sha256: profile.schema_sha256.clone(),
            service_boundary_ref: profile.service_boundary_ref.clone(),
            unit_ref: profile.unit_ref.clone(),
            averaging_window_ref: profile.averaging_window_ref.clone(),
            sign_convention_ref: profile.sign_convention_ref.clone(),
            lower_bound_mwe: lower_bound,
            one_sided_coverage_level: 0.95,
            evidence_class: NetExportEvidenceClass::SyntheticFixture,
            coverage_scope_ref: "synthetic-single-reserve-test-v1".to_string(),
            method: NetExportBoundMethod::JointCovarianceContrast,
            method_ref: "synthetic-joint-net-power-contrast-v1".to_string(),
            method_evidence_sha256: sha('c'),
            method_run_id: run.run_id.clone(),
            method_output_sha256: run.output_sha256.clone(),
            method_variable_order: vec![
                "P_e_gross".to_string(),
                "P_recirc_var".to_string(),
                "P_recirc_fix".to_string(),
            ],
            distribution_assumption_ref: "synthetic-known-answer-normal-fixture-v1".to_string(),
            assumption_manifest_sha256: sha('d'),
            validity_domain_ref: "synthetic-declared-domain-v1".to_string(),
            model_form_validity: ModelFormValidity::WithinDeclaredDomain,
            model_form_evidence_ref: "synthetic-model-form-validity-evidence".to_string(),
            model_form_evidence_sha256: sha('f'),
            covariance_receipt_ref: Some("synthetic-covariance-receipt-v1".to_string()),
            covariance_receipt_sha256: Some(sha('e')),
            covariance_run_id: Some(run.run_id.clone()),
            covariance_output_sha256: Some(run.output_sha256.clone()),
            covariance_variable_order: vec![
                "P_e_gross".to_string(),
                "P_recirc_var".to_string(),
                "P_recirc_fix".to_string(),
            ],
            simultaneous_allocation_ref: None,
            direct_net_analysis_ref: None,
            evidence_ref: "synthetic-net-lower-bound-evidence".to_string(),
        };

        let report = evaluate_robust_net_export(
            &profile,
            &run,
            &point,
            &robust_profile,
            Some(&receipt),
        );
        println!(
            "lower_bound_mwe={lower_bound:.3},point_estimate_mwe={:.3},reserve_mwe={:.3},coverage={:.3},disposition={:?}",
            report.point_estimate_net_mwe.unwrap_or(f64::NAN),
            report.required_reserve_mwe,
            report.one_sided_coverage_level.unwrap_or(f64::NAN),
            report.disposition
        );

        // Synthetic receipts are deliberately prevented from making a
        // robust qualification claim, even when their numeric bound clears
        // the reserve threshold.
        let expected = RobustNetExportDisposition::SyntheticOnly;
        if report.disposition != expected {
            eprintln!("robust net export fixture did not match its expected reserve disposition");
            std::process::exit(1);
        }
    }

    println!(
        "SCOPE: synthetic uncertainty receipt and reserve-gate contract only; no physical uncertainty calibration or plant qualification"
    );
}
