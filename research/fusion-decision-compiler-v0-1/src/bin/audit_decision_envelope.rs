use fusion_decision_compiler_contract::cross_solver_comparison::{
    ApplicabilityState, ConvergenceState,
};
use fusion_decision_compiler_contract::decision_envelope::{
    DecisionEnvelopePlan, DecisionEnvelopeProfile, ModelEnvelopeInput, ThresholdDirection,
    evaluate_decision_envelope,
};
use fusion_decision_compiler_contract::evidence_dependence::{
    DependencyIdentity, DependencyKind, ObservableDependency, RunDependencyManifest,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn model(run_id: &str, solver_id: &str, value: f64, shared_assumption: &str) -> ModelEnvelopeInput {
    let observable = "net-electric-export".to_string();
    ModelEnvelopeInput {
        run_id: run_id.to_string(),
        solver_id: solver_id.to_string(),
        solver_revision: format!("{solver_id}-reference-r1"),
        case_id: "synthetic-st-tokamak-case-v1".to_string(),
        candidate_id: "synthetic-candidate-42".to_string(),
        candidate_revision: "synthetic-candidate-r7".to_string(),
        geometry_sha256: sha('a'),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        observable_id: observable.clone(),
        canonical_quantity_ref: "quantity-net-electric-power".to_string(),
        unit_ref: "MWe".to_string(),
        frame_ref: "plant-terminal".to_string(),
        sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        time_basis_ref: "steady-state-60s-v1".to_string(),
        normalization_contract_ref: "net-electric-normalization-v1".to_string(),
        normalized_value: value,
        output_sha256: sha(if solver_id == "PROCESS" { 'b' } else { 'c' }),
        run_receipt_ref: format!("synthetic-run-receipt-{run_id}"),
        output_evidence_ref: format!("synthetic-output-evidence-{run_id}"),
        applicability: ApplicabilityState::InDeclaredDomain,
        convergence: ConvergenceState::Converged,
        dependencies: RunDependencyManifest {
            run_id: run_id.to_string(),
            output_observables: vec![observable.clone()],
            complete_observables: vec![observable.clone()],
            dependencies: vec![ObservableDependency {
                observable_id: observable,
                dependency: DependencyIdentity {
                    node_id: shared_assumption.to_string(),
                    revision: "scenario-assumption-r1".to_string(),
                    kind: DependencyKind::SharedScenarioAssumption,
                    content_sha256: Some(sha('d')),
                    source_evidence_ref: Some("synthetic-shared-assumption-receipt".to_string()),
                },
            }],
        },
    }
}

fn main() {
    let plan = DecisionEnvelopePlan {
        profile: DecisionEnvelopeProfile {
            case_id: "synthetic-st-tokamak-case-v1".to_string(),
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
        models: vec![
            model("process-run-001", "PROCESS", 92.0, "shared-density-closure-v1"),
            model("saras-run-001", "SARAS", 108.0, "shared-density-closure-v1"),
        ],
    };

    let report = evaluate_decision_envelope(&plan);
    println!("contract_version={}", report.contract_version);
    println!("case_id={}", report.case_id);
    println!("candidate_id={}", report.candidate_id);
    println!("observable_id={}", report.observable_id);
    println!("threshold={:.3} MWe", report.threshold_value);
    println!("decision_envelope={:?}", report.disposition);
    println!("spread_disposition={:?}", report.spread_disposition);
    println!("minimum={:.3} MWe", report.minimum_value.unwrap_or(f64::NAN));
    println!("maximum={:.3} MWe", report.maximum_value.unwrap_or(f64::NAN));
    println!("absolute_range={:.3} MW", report.absolute_range.unwrap_or(f64::NAN));
    println!(
        "relative_range={:.3}",
        report.relative_range.unwrap_or(f64::NAN)
    );
    for model in &report.model_decisions {
        println!(
            "model={},run={},value_side={:?},output={}",
            model.solver_id,
            model.run_id,
            model.threshold_result,
            model.output_sha256
        );
    }
    for pair in &report.dependence_reports {
        println!(
            "dependence_pair={}/{} state={:?} shared_dependencies={}",
            pair.first_run_id,
            pair.second_run_id,
            pair.assessment.state,
            pair.assessment.shared_dependencies.len()
        );
    }
    println!(
        "SCOPE: synthetic finite model-set decision split and common-dependency fixture; not a probability interval, solver validation or physical truth"
    );

    if report.disposition
        != fusion_decision_compiler_contract::decision_envelope::DecisionEnvelopeDisposition::DecisionSplit
    {
        eprintln!("synthetic decision-envelope fixture did not preserve the threshold split");
        std::process::exit(1);
    }
}
