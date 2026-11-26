use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    FORWARD_EVALUATION_RECEIPT_SCHEMA_V1, HISTORICAL_SUPPORT_RECEIPT_SCHEMA_V1,
    CandidateScreenReasonV1, CandidateScreenStateV1, ForwardEvaluationReceiptV1,
    ForwardEvidenceStateV1, HistoricalSupportEvidenceV1, HistoricalSupportReceiptV1,
    INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1, INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1,
    InverseVmCandidateV1, InverseVmShortlistInputV1, InverseVmShortlistPolicyV1,
    MetrologyTargetV1, NumericRangeV1, PredictedMetricIntervalV1, RecipeKnobConstraintV1,
    RecipeSettingV1, SideConstraintV1, evaluate_inverse_vm_shortlist_v1,
    seal_forward_evaluation_receipt_v1, seal_historical_support_receipt_v1,
    seal_inverse_vm_candidate_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_digest<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn sorted_recipe_digest(settings: &[RecipeSettingV1]) -> String {
    let mut canonical = settings.to_vec();
    canonical.sort_by(|a, b| a.knob_id.cmp(&b.knob_id));
    json_digest(&canonical)
}

fn sorted_metric_digest(metrics: &[PredictedMetricIntervalV1]) -> String {
    let mut canonical = metrics.to_vec();
    canonical.sort_by(|a, b| a.measurand_id.cmp(&b.measurand_id));
    json_digest(&canonical)
}

fn policy() -> InverseVmShortlistPolicyV1 {
    InverseVmShortlistPolicyV1 {
        schema: INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1.into(),
        policy_id: "forward-receipt-reference-policy.v1".into(),
        revision: 1,
        recipe_constraint_source_sha256: digest('a'),
        recipe_constraints_complete: true,
        recipe_knob_constraints: vec![
            RecipeKnobConstraintV1 {
                knob_id: "rf_power".into(),
                unit: "W".into(),
                allowed: NumericRangeV1 { lower: 100.0, upper: 500.0 },
            },
            RecipeKnobConstraintV1 {
                knob_id: "pressure".into(),
                unit: "mTorr".into(),
                allowed: NumericRangeV1 { lower: 10.0, upper: 100.0 },
            },
        ],
        side_constraints_complete: true,
        side_constraints: vec![SideConstraintV1 {
            constraint_id: "uniformity".into(),
            measurand_id: "uniformity".into(),
            unit: "%".into(),
            allowed: NumericRangeV1 { lower: 0.0, upper: 3.0 },
        }],
        target: MetrologyTargetV1 {
            measurand_id: "epi_thickness".into(),
            unit: "nm".into(),
            target_value: 100.0,
            acceptable: NumericRangeV1 { lower: 98.0, upper: 102.0 },
        },
        required_forward_model_sha256: digest('b'),
        support_manifest_sha256: digest('c'),
        support_policy_sha256: digest('d'),
        minimum_support_neighbours: 2,
        maximum_normalized_support_distance: 0.75,
        maximum_interval_width_fraction: 0.75,
        target_error_weight: 1.0,
        interval_width_weight: 0.5,
        support_distance_weight: 0.25,
        diversity_weight: 0.05,
        minimum_recipe_diversity_distance: 0.02,
        top_k: 1,
    }
}

fn support_receipt(
    process_snapshot_sha256: &str,
    forward_settings: &[RecipeSettingV1],
    neighbour_ids: &[String],
    distance: f64,
) -> HistoricalSupportReceiptV1 {
    let mut sorted_neighbours = neighbour_ids.to_vec();
    sorted_neighbours.sort();
    let mut receipt = HistoricalSupportReceiptV1 {
        schema: HISTORICAL_SUPPORT_RECEIPT_SCHEMA_V1.into(),
        evaluator_id: "independent-support-evaluator.v1".into(),
        process_snapshot_sha256: process_snapshot_sha256.into(),
        support_manifest_sha256: digest('c'),
        support_policy_sha256: digest('d'),
        recipe_settings_sha256: sorted_recipe_digest(forward_settings),
        neighbour_ids_sha256: json_digest(&sorted_neighbours),
        normalized_distance: distance,
        clock_domain_id: "fab-clock".into(),
        clock_mapping_sha256: digest('9'),
        evaluated_at_unix_seconds: 80,
        available_at_unix_seconds: 90,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = seal_historical_support_receipt_v1(&receipt).unwrap();
    receipt
}

fn candidate(available_at: i64) -> InverseVmCandidateV1 {
    let settings = vec![
        RecipeSettingV1 { knob_id: "rf_power".into(), unit: "W".into(), value: 220.0 },
        RecipeSettingV1 { knob_id: "pressure".into(), unit: "mTorr".into(), value: 50.0 },
    ];
    let metrics = vec![
        PredictedMetricIntervalV1 {
            measurand_id: "epi_thickness".into(),
            unit: "nm".into(),
            lower: 99.0,
            upper: 101.0,
        },
        PredictedMetricIntervalV1 {
            measurand_id: "uniformity".into(),
            unit: "%",
            lower: 1.0,
            upper: 2.0,
        },
    ];
    let mut value = InverseVmCandidateV1 {
        candidate_id: "candidate-001".into(),
        process_snapshot_sha256: digest('e'),
        recipe_settings: settings,
        historical_support: Some(HistoricalSupportEvidenceV1 {
            support_manifest_sha256: digest('c'),
            neighbour_ids: vec!["neighbour-1".into(), "neighbour-2".into()],
            normalized_distance: 0.25,
            support_policy_sha256: digest('d'),
            receipt: None,
        }),
        forward_model_sha256: digest('b'),
        forward_evidence_state: ForwardEvidenceStateV1::Qualified,
        forward_evaluation_receipt: None,
        predicted_metrics: metrics,
        candidate_evidence_sha256: String::new(),
    };
    let support_receipt = support_receipt(
        &value.process_snapshot_sha256,
        &value.recipe_settings,
        &value.historical_support.as_ref().unwrap().neighbour_ids,
        0.25,
    );
    value.historical_support.as_mut().unwrap().receipt = Some(support_receipt);
    let mut receipt = ForwardEvaluationReceiptV1 {
        schema: FORWARD_EVALUATION_RECEIPT_SCHEMA_V1.into(),
        evaluator_id: "independent-forward-evaluator.v1".into(),
        clock_domain_id: "fab-clock".into(),
        clock_mapping_sha256: digest('9'),
        process_snapshot_sha256: value.process_snapshot_sha256.clone(),
        forward_model_sha256: value.forward_model_sha256.clone(),
        model_qualification_report_sha256: Some(digest('f')),
        recipe_settings_sha256: sorted_recipe_digest(&value.recipe_settings),
        predicted_metrics_sha256: sorted_metric_digest(&value.predicted_metrics),
        evidence_state: ForwardEvidenceStateV1::Qualified,
        evaluated_at_unix_seconds: 90,
        available_at_unix_seconds: available_at,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = seal_forward_evaluation_receipt_v1(&receipt).unwrap();
    value.forward_evaluation_receipt = Some(receipt);
    value.candidate_evidence_sha256 = seal_inverse_vm_candidate_v1(&value).unwrap();
    value
}

fn input(candidate: InverseVmCandidateV1, decision_at: i64) -> InverseVmShortlistInputV1 {
    InverseVmShortlistInputV1 {
        schema: INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1.into(),
        input_id: "forward-receipt-reference-input.v1".into(),
        process_snapshot_sha256: digest('e'),
        decision_at_unix_seconds: decision_at,
        clock_domain_id: "fab-clock".into(),
        clock_mapping_sha256: digest('9'),
        policy: policy(),
        candidates: vec![candidate],
    }
}

#[test]
fn independently_built_receipt_is_eligible_only_before_the_decision_frontier() {
    let timely = evaluate_inverse_vm_shortlist_v1(&input(candidate(99), 100)).unwrap();
    assert_eq!(timely.evaluations[0].state, CandidateScreenStateV1::Feasible);
    assert_eq!(timely.top_k_candidate_ids, vec!["candidate-001"]);
    assert!(!timely.forward_evaluation_receipt_authenticity_proven);

    let late = evaluate_inverse_vm_shortlist_v1(&input(candidate(101), 100)).unwrap();
    assert_eq!(late.evaluations[0].state, CandidateScreenStateV1::Unknown);
    assert!(
        late.evaluations[0]
            .reasons
            .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptNotYetAvailable)
    );
    assert!(late.top_k_candidate_ids.is_empty());
}

#[test]
fn independently_built_support_receipt_binds_candidate_recipe_and_neighbour_set() {
    let timely = evaluate_inverse_vm_shortlist_v1(&input(candidate(99), 100)).unwrap();
    assert_eq!(timely.evaluations[0].state, CandidateScreenStateV1::Feasible);
    assert!(!timely.historical_support_receipt_authenticity_proven);

    let mut altered = candidate(99);
    altered.historical_support.as_mut().unwrap().neighbour_ids[0] =
        "different-neighbour".into();
    altered.candidate_evidence_sha256 = seal_inverse_vm_candidate_v1(&altered).unwrap();
    let report = evaluate_inverse_vm_shortlist_v1(&input(altered, 100)).unwrap();
    assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Rejected);
    assert!(
        report.evaluations[0]
            .reasons
            .contains(&CandidateScreenReasonV1::HistoricalSupportReceiptNeighbourMismatch)
    );
}

#[test]
fn independently_built_receipt_detects_recipe_prediction_and_clock_changes() {
    let mut recipe_changed = candidate(99);
    recipe_changed.recipe_settings[0].value += 1.0;
    recipe_changed.candidate_evidence_sha256 =
        seal_inverse_vm_candidate_v1(&recipe_changed).unwrap();
    let recipe_report = evaluate_inverse_vm_shortlist_v1(&input(recipe_changed, 100)).unwrap();
    assert_eq!(recipe_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
    assert!(
        recipe_report.evaluations[0]
            .reasons
            .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptRecipeMismatch)
    );

    let mut metrics_changed = candidate(99);
    metrics_changed.predicted_metrics[0].upper = 100.5;
    metrics_changed.candidate_evidence_sha256 =
        seal_inverse_vm_candidate_v1(&metrics_changed).unwrap();
    let metric_report = evaluate_inverse_vm_shortlist_v1(&input(metrics_changed, 100)).unwrap();
    assert_eq!(metric_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
    assert!(
        metric_report.evaluations[0]
            .reasons
            .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptPredictionMismatch)
    );

    let mut clock_changed = candidate(99);
    let receipt = clock_changed.forward_evaluation_receipt.as_mut().unwrap();
    receipt.clock_mapping_sha256 = digest('8');
    receipt.receipt_sha256 = seal_forward_evaluation_receipt_v1(receipt).unwrap();
    clock_changed.candidate_evidence_sha256 =
        seal_inverse_vm_candidate_v1(&clock_changed).unwrap();
    let clock_report = evaluate_inverse_vm_shortlist_v1(&input(clock_changed, 100)).unwrap();
    assert_eq!(clock_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
    assert!(
        clock_report.evaluations[0]
            .reasons
            .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptClockDomainMismatch)
    );
}
