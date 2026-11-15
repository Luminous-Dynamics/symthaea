use fusion_decision_compiler_contract::cross_solver_comparison::{
    ApplicabilityState, ConvergenceState, CrossCovarianceEvidence, CrossCovarianceState,
    FrozenComparisonCase, SolverObservableOutput, compare_solver_outputs,
};
use fusion_decision_compiler_contract::evidence_dependence::{
    DependencyIdentity, DependencyKind, ObservableDependency, RunDependencyManifest,
};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn comparison_case() -> FrozenComparisonCase {
    FrozenComparisonCase {
        case_id: "synthetic-plant-reference-case-v1".to_string(),
        candidate_id: "candidate-a".to_string(),
        candidate_revision: "design-r7".to_string(),
        geometry_sha256: digest('a'),
        service_boundary_ref: "net-electric-export-boundary-v1".to_string(),
        observable_id: "net-electric-export-mw".to_string(),
        canonical_quantity_ref: "quantity-power-net-electric".to_string(),
        unit_ref: "unit-megawatt-electric".to_string(),
        frame_ref: "frame-grid-interface".to_string(),
        sign_convention_ref: "grid-export-positive-v1".to_string(),
        time_basis_ref: "steady-state-window-300s-v1".to_string(),
        normalization_contract_ref: "plant-output-normalization-v1".to_string(),
    }
}

fn output(run_id: &str, solver_id: &str, value: f64, digest_char: char) -> SolverObservableOutput {
    let case = comparison_case();
    SolverObservableOutput {
        run_id: run_id.to_string(),
        solver_id: solver_id.to_string(),
        solver_revision: "synthetic-pinned-revision".to_string(),
        case_id: case.case_id,
        candidate_id: case.candidate_id,
        candidate_revision: case.candidate_revision,
        geometry_sha256: case.geometry_sha256,
        service_boundary_ref: case.service_boundary_ref,
        observable_id: case.observable_id,
        canonical_quantity_ref: case.canonical_quantity_ref,
        unit_ref: case.unit_ref,
        frame_ref: case.frame_ref,
        sign_convention_ref: case.sign_convention_ref,
        time_basis_ref: case.time_basis_ref,
        normalization_contract_ref: case.normalization_contract_ref,
        normalized_value: value,
        output_sha256: digest(digest_char),
        run_receipt_ref: format!("synthetic-run-receipt-{run_id}"),
        output_evidence_ref: format!("synthetic-output-evidence-{run_id}"),
        applicability: ApplicabilityState::InDeclaredDomain,
        convergence: ConvergenceState::Converged,
    }
}

fn dependencies(run_id: &str, source: &str) -> RunDependencyManifest {
    RunDependencyManifest {
        run_id: run_id.to_string(),
        output_observables: vec!["net-electric-export-mw".to_string()],
        complete_observables: vec!["net-electric-export-mw".to_string()],
        dependencies: vec![ObservableDependency {
            observable_id: "net-electric-export-mw".to_string(),
            dependency: DependencyIdentity {
                node_id: source.to_string(),
                revision: "synthetic-source-r1".to_string(),
                kind: DependencyKind::InputDataset,
                content_sha256: None,
                source_evidence_ref: Some(format!("synthetic-source-receipt-{source}")),
            },
        }],
    }
}

fn main() {
    let case = comparison_case();
    let process = output("run-process", "PROCESS", 1_200.0, 'b');
    let d0fus = output("run-d0fus", "D0FUS", 1_185.0, 'c');

    // Missing cross-covariance is intentionally explicit. It is not replaced
    // with zero merely because the two declared dependency inventories differ.
    let covariance = CrossCovarianceEvidence {
        state: CrossCovarianceState::Unknown,
        receipt_ref: None,
        first_run_id: None,
        second_run_id: None,
        observable_id: None,
        first_output_sha256: None,
        second_output_sha256: None,
    };

    match compare_solver_outputs(
        &case,
        &process,
        &d0fus,
        &dependencies("run-process", "process-inputs"),
        &dependencies("run-d0fus", "d0fus-inputs"),
        &covariance,
    ) {
        Ok(result) => {
            println!("contract_version={}", result.contract_version);
            println!("case={}", result.case_id);
            println!("observable={}", result.observable_id);
            println!("first_solver={}", result.first_solver_id);
            println!("second_solver={}", result.second_solver_id);
            println!("disposition={:?}", result.disposition);
            println!("cross_covariance={:?}", result.cross_covariance);
            println!("dependency_state={:?}", result.dependence.state);
            match result.second_minus_first {
                Some(value) => println!("second_minus_first={value:.3} MW"),
                None => println!("second_minus_first=unavailable"),
            }
            match result.absolute_difference {
                Some(value) => println!("absolute_difference={value:.3} MW"),
                None => println!("absolute_difference=unavailable"),
            }
            println!(
                "SCOPE: synthetic matched-case comparison only; descriptive difference, no inferential significance or physical validation"
            );
        }
        Err(errors) => {
            eprintln!("cross-solver comparison refused: {errors:?}");
            std::process::exit(1);
        }
    }
}
