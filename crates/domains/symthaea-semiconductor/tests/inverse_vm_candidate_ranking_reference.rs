use std::collections::BTreeMap;

use symthaea_semiconductor::benchmark::{
    CandidateConstraintPredictionV1, CandidateDispositionV1, CandidateEvidenceStatusV1,
    CandidateTargetPredictionV1, INVERSE_VM_CANDIDATE_SCHEMA_V1,
    INVERSE_VM_RANKING_SCHEMA_V1, InverseVmCandidateV1,
    InverseVmRankingInputV1, InverseVmSideConstraintV1, InverseVmTargetSpecV1,
    RecipeControlBoundV1, rank_feasible_inverse_vm_candidates_v1,
    seal_inverse_vm_candidate_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn candidate(
    id: &str,
    pressure: f64,
    power: f64,
    target: f64,
    uncertainty: f64,
    constraint_value: f64,
    status: CandidateEvidenceStatusV1,
    support_distance: f64,
) -> InverseVmCandidateV1 {
    let mut item = InverseVmCandidateV1 {
        schema: INVERSE_VM_CANDIDATE_SCHEMA_V1.into(),
        candidate_id: id.into(),
        recipe_schema_sha256: digest('a'),
        control_values: BTreeMap::from([
            ("pressure".into(), pressure),
            ("power".into(), power),
        ]),
        model_artifact_sha256: digest('c'),
        forward_prediction_receipt_sha256: digest('d'),
        model_evidence_receipt_sha256: digest('e'),
        model_evidence_status: status,
        model_evidence_available_at_unix_seconds: 1_800_000_000,
        support_manifest_sha256: digest('b'),
        support_receipt_sha256: digest('f'),
        historical_support_distance: support_distance,
        historical_support_radius: 1.0,
        historical_support_count: 50,
        target_predictions: vec![CandidateTargetPredictionV1 {
            target_id: "thickness".into(),
            unit: "nm".into(),
            predicted_value: target,
            standard_uncertainty: uncertainty,
        }],
        constraint_predictions: vec![CandidateConstraintPredictionV1 {
            constraint_id: "uniformity".into(),
            unit: "%".into(),
            predicted_value: constraint_value,
            standard_uncertainty: 0.2,
        }],
        candidate_sha256: String::new(),
    };
    item.candidate_sha256 = seal_inverse_vm_candidate_v1(&item).unwrap();
    item
}

fn fixture() -> InverseVmRankingInputV1 {
    InverseVmRankingInputV1 {
        schema: INVERSE_VM_RANKING_SCHEMA_V1.into(),
        ranking_id: "fixed-independent-topk-oracle".into(),
        recipe_schema_sha256: digest('a'),
        historical_support_manifest_sha256: digest('b'),
        knowledge_cutoff_unix_seconds: 1_900_000_000,
        confidence_multiplier: 2.0,
        minimum_support_count: 20,
        requested_top_k: 2,
        support_score_weight: 0.0,
        constraint_margin_weight: 0.0,
        diversity_weight: 0.5,
        minimum_diversity_distance: 0.2,
        control_bounds: vec![
            RecipeControlBoundV1 {
                control_id: "pressure".into(),
                unit: "kPa".into(),
                minimum: 10.0,
                maximum: 20.0,
            },
            RecipeControlBoundV1 {
                control_id: "power".into(),
                unit: "W".into(),
                minimum: 100.0,
                maximum: 200.0,
            },
        ],
        target_specs: vec![InverseVmTargetSpecV1 {
            target_id: "thickness".into(),
            unit: "nm".into(),
            desired_value: 100.0,
            tolerance: 2.0,
        }],
        side_constraints: vec![InverseVmSideConstraintV1 {
            constraint_id: "uniformity".into(),
            unit: "%".into(),
            minimum: 0.0,
            maximum: 5.0,
        }],
        candidates: vec![
            candidate("a-near", 12.0, 120.0, 100.0, 0.1, 2.0, CandidateEvidenceStatusV1::CurrentEligible, 0.1),
            candidate("b-far", 19.0, 190.0, 100.0, 0.1, 2.0, CandidateEvidenceStatusV1::CurrentEligible, 0.1),
            candidate("c-near-duplicate", 12.1, 121.0, 100.0, 0.1, 2.0, CandidateEvidenceStatusV1::CurrentEligible, 0.1),
            candidate("d-side-constraint-fail", 14.0, 140.0, 100.0, 0.1, 4.9, CandidateEvidenceStatusV1::CurrentEligible, 0.1),
            candidate("e-unknown-model", 16.0, 160.0, 100.0, 0.1, 2.0, CandidateEvidenceStatusV1::Unknown, 0.1),
            candidate("f-outside-support", 17.0, 170.0, 100.0, 0.1, 2.0, CandidateEvidenceStatusV1::CurrentEligible, 1.2),
        ],
    }
}

/// Independent, frozen known answer: A and B are equally feasible on target
/// and support score, but B is sufficiently far from A in normalized control
/// space to beat the near-duplicate C. D violates the robust side constraint,
/// E has unknown model evidence and F lies outside historical support.
#[test]
fn fixed_topk_fixture_matches_independent_candidate_and_diversity_oracle() {
    let input = fixture();
    let report = rank_feasible_inverse_vm_candidates_v1(&input).unwrap();

    assert_eq!(report.eligible_count, 3);
    assert_eq!(report.infeasible_count, 1);
    assert_eq!(report.unknown_count, 1);
    assert_eq!(report.unsupported_count, 1);
    assert_eq!(
        report.selected_candidate_ids,
        vec!["a-near".to_string(), "b-far".to_string()]
    );
    assert!(report.top_k_filled);
    assert!(!report.source_authenticity_proven);
    assert!(!report.physical_success_proven);
    assert!(!report.recipe_write_executed);
    assert!(!report.apc_r2r_authority);
    assert!(!report.equipment_control_authority);

    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "a-near").unwrap().disposition,
        CandidateDispositionV1::Eligible
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "b-far").unwrap().disposition,
        CandidateDispositionV1::Eligible
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "c-near-duplicate").unwrap().disposition,
        CandidateDispositionV1::Eligible
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "d-side-constraint-fail").unwrap().disposition,
        CandidateDispositionV1::Infeasible
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "e-unknown-model").unwrap().disposition,
        CandidateDispositionV1::Unknown
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "f-outside-support").unwrap().disposition,
        CandidateDispositionV1::Unsupported
    );
    assert_eq!(
        report.evaluations.iter().find(|value| value.candidate_id == "c-near-duplicate").unwrap().selected_rank,
        None
    );
}

#[test]
fn candidate_permutation_preserves_the_sealed_shortlist_and_receipt() {
    let mut first = fixture();
    let mut second = fixture();
    second.candidates.reverse();

    let report_a = rank_feasible_inverse_vm_candidates_v1(&first).unwrap();
    let report_b = rank_feasible_inverse_vm_candidates_v1(&second).unwrap();
    assert_eq!(report_a.selected_candidate_ids, report_b.selected_candidate_ids);
    assert_eq!(report_a.report_sha256, report_b.report_sha256);

    first.candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    assert_eq!(
        report_a.input_sha256,
        rank_feasible_inverse_vm_candidates_v1(&first).unwrap().input_sha256
    );
}
