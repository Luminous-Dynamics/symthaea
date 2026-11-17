use fusion_decision_compiler_contract::cross_solver_comparison::{
    ApplicabilityState, ConvergenceState,
};
use fusion_decision_compiler_contract::decision_envelope::{
    DecisionEnvelopePlan, DecisionEnvelopeProfile, ModelEnvelopeInput, ThresholdDirection,
};
use fusion_decision_compiler_contract::empirical_holdout_stress::{
    EmpiricalStressDecision, EmpiricalStressEvidenceDisposition, EmpiricalStressPlan,
    EmpiricalStressProfile, HoldoutMetricResult, ModelHoldoutStressReceipt, SliceStressEvidence,
    StressEvidenceClass, evaluate_empirical_holdout_stress,
};
use fusion_decision_compiler_contract::evidence_dependence::{
    DependencyIdentity, DependencyKind, ObservableDependency, RunDependencyManifest,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn model(run_id: &str, solver_id: &str, value: f64, output: char) -> ModelEnvelopeInput {
    ModelEnvelopeInput {
        run_id: run_id.to_string(),
        solver_id: solver_id.to_string(),
        solver_revision: format!("{solver_id}-r1"),
        case_id: "synthetic-st-case-v1".to_string(),
        candidate_id: "synthetic-candidate-42".to_string(),
        candidate_revision: "synthetic-candidate-r7".to_string(),
        geometry_sha256: sha('a'),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        observable_id: "net-electric-export".to_string(),
        canonical_quantity_ref: "quantity-net-electric-power".to_string(),
        unit_ref: "MWe".to_string(),
        frame_ref: "plant-terminal".to_string(),
        sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        time_basis_ref: "steady-state-60s-v1".to_string(),
        normalization_contract_ref: "net-electric-normalization-v1".to_string(),
        normalized_value: value,
        output_sha256: sha(output),
        run_receipt_ref: format!("synthetic-run-receipt-{run_id}"),
        output_evidence_ref: format!("synthetic-output-evidence-{run_id}"),
        applicability: ApplicabilityState::InDeclaredDomain,
        convergence: ConvergenceState::Converged,
        dependencies: RunDependencyManifest {
            run_id: run_id.to_string(),
            output_observables: vec!["net-electric-export".to_string()],
            complete_observables: vec!["net-electric-export".to_string()],
            dependencies: vec![ObservableDependency {
                observable_id: "net-electric-export".to_string(),
                dependency: DependencyIdentity {
                    node_id: "shared-scenario-closure".to_string(),
                    revision: "r1".to_string(),
                    kind: DependencyKind::SharedScenarioAssumption,
                    content_sha256: Some(sha('d')),
                    source_evidence_ref: Some("synthetic-shared-scenario-receipt".to_string()),
                },
            }],
        },
    }
}

fn slice(id: &str, residual: f64) -> SliceStressEvidence {
    SliceStressEvidence {
        slice_id: id.to_string(),
        facility_id: format!("synthetic-facility-{id}"),
        operating_point_id: format!("synthetic-operating-point-{id}"),
        case_family_id: format!("synthetic-case-family-{id}"),
        in_declared_domain: true,
        sample_count: 12,
        independent_group_count: 6,
        maximum_absolute_residual: residual,
        residual_unit_ref: "MWe".to_string(),
        residual_basis_ref: "absolute-measured-minus-predicted-v1".to_string(),
        residual_profile_sha256: sha('6'),
        evidence_ref: format!("synthetic-residual-evidence-{id}"),
    }
}

fn receipt(
    model: &ModelEnvelopeInput,
    evidence_class: StressEvidenceClass,
    first_slice_residual: f64,
    second_slice_residual: f64,
) -> ModelHoldoutStressReceipt {
    ModelHoldoutStressReceipt {
        receipt_id: format!("synthetic-holdout-receipt-{}", model.run_id),
        receipt_sha256: sha('1'),
        run_id: model.run_id.clone(),
        solver_id: model.solver_id.clone(),
        solver_revision: model.solver_revision.clone(),
        candidate_id: model.candidate_id.clone(),
        candidate_revision: model.candidate_revision.clone(),
        geometry_sha256: model.geometry_sha256.clone(),
        case_id: model.case_id.clone(),
        observable_id: model.observable_id.clone(),
        canonical_quantity_ref: model.canonical_quantity_ref.clone(),
        unit_ref: model.unit_ref.clone(),
        frame_ref: model.frame_ref.clone(),
        sign_convention_ref: model.sign_convention_ref.clone(),
        time_basis_ref: model.time_basis_ref.clone(),
        normalization_contract_ref: model.normalization_contract_ref.clone(),
        output_sha256: model.output_sha256.clone(),
        run_receipt_ref: model.run_receipt_ref.clone(),
        run_receipt_sha256: sha('2'),
        model_artifact_sha256: sha(if model.solver_id == "PROCESS" { '3' } else { '4' }),
        validation_domain_id: "synthetic-validation-domain-v1".to_string(),
        validation_domain_audit_sha256: sha('5'),
        validation_profile_sha256: sha('6'),
        decision_profile_ref: "synthetic-net-export-decision-profile-v1".to_string(),
        decision_profile_sha256: sha('7'),
        prediction_commitment_ref: "synthetic-pre-reveal-commitment".to_string(),
        prediction_commitment_sha256: sha('8'),
        committed_before_holdout_reveal: true,
        test_stand_artifact_sha256: sha('9'),
        holdout_artifact_sha256: sha('a'),
        measurement_attestation_ref: "synthetic-measurement-attestation".to_string(),
        measurement_attestation_sha256: sha('b'),
        measurement_payload_externally_verified: true,
        validation_domain_admissible: true,
        holdout_metric_result: HoldoutMetricResult::PassedDeclaredMetricPolicy,
        evidence_class,
        slice_manifest_sha256: sha('c'),
        required_slices: vec!["facility-a-op-a".to_string(), "facility-b-op-b".to_string()],
        slices: vec![
            slice("facility-a-op-a", first_slice_residual),
            slice("facility-b-op-b", second_slice_residual),
        ],
    }
}

fn main() {
    let process = model("process-run", "PROCESS", 92.0, 'e');
    let saras = model("saras-run", "SARAS", 108.0, 'f');
    let decision_plan = DecisionEnvelopePlan {
        profile: DecisionEnvelopeProfile {
            case_id: "synthetic-st-case-v1".to_string(),
            candidate_id: "synthetic-candidate-42".to_string(),
            candidate_revision: "synthetic-candidate-r7".to_string(),
            geometry_sha256: sha('a'),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            observable_id: "net-electric-export".to_string(),
            canonical_quantity_ref: "quantity-net-electric-power".to_string(),
            unit_ref: "MWe".to_string(),
            frame_ref: "plant-terminal".to_string(),
            sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
            time_basis_ref: "steady-state-60s-v1".to_string(),
            normalization_contract_ref: "net-electric-normalization-v1".to_string(),
            threshold_value: 100.0,
            threshold_direction: ThresholdDirection::AtLeast,
            threshold_inclusive: true,
            absolute_spread_tolerance: 10.0,
            relative_spread_reference_scale: Some(100.0),
        },
        models: vec![process.clone(), saras.clone()],
    };

    let plan = EmpiricalStressPlan {
        decision_plan,
        stress_profile: EmpiricalStressProfile {
            decision_profile_ref: "synthetic-net-export-decision-profile-v1".to_string(),
            decision_profile_sha256: sha('7'),
            validation_domain_id: "synthetic-validation-domain-v1".to_string(),
            validation_profile_sha256: sha('6'),
            residual_basis_ref: "absolute-measured-minus-predicted-v1".to_string(),
            minimum_sample_count_per_slice: 10,
            minimum_independent_groups_per_slice: 5,
            required_slice_ids: vec![
                "facility-a-op-a".to_string(),
                "facility-b-op-b".to_string(),
            ],
        },
        holdout_receipts: vec![
            receipt(&process, StressEvidenceClass::SyntheticFixture, 3.0, 7.0),
            receipt(&saras, StressEvidenceClass::SyntheticFixture, 4.0, 8.0),
        ],
    };

    let report = evaluate_empirical_holdout_stress(&plan);
    println!("contract_version={}", report.contract_version);
    println!("case_id={}", report.case_id);
    println!("candidate_id={}", report.candidate_id);
    println!("observable_id={}", report.observable_id);
    println!("base_decision={:?}", report.base_decision_disposition);
    println!("stress_decision={:?}", report.stress_decision);
    println!("evidence_disposition={:?}", report.evidence_disposition);
    for model in &report.models {
        println!(
            "solver={},point={:.3},radius={:.3},stress=[{:.3},{:.3}],side={:?},worst_slice={}",
            model.solver_id,
            model.point_estimate,
            model.empirical_stress_radius,
            model.stress_lower,
            model.stress_upper,
            model.stress_side,
            model.worst_slice_id
        );
    }
    println!(
        "SCOPE: synthetic finite-model and worst-slice residual fixture; empirical stress is not a confidence interval or future-error guarantee"
    );

    if report.stress_decision != Some(EmpiricalStressDecision::StressDecisionSplit)
        || report.evidence_disposition != EmpiricalStressEvidenceDisposition::SyntheticOnly
    {
        eprintln!("synthetic empirical-stress fixture did not preserve its expected numeric/evidence dispositions");
        std::process::exit(1);
    }
}
