use symthaea_semiconductor::benchmark::{
    CAUSAL_DRIFT_EVIDENCE_SCHEMA_V1, CAUSAL_DRIFT_INPUT_SCHEMA_V1,
    CausalAssessmentStatusV1, CausalDriftDiagnosisInputV1, CausalDriftEvidenceKindV1,
    CausalDriftEvidenceV1, CausalDriftHypothesisInputV1, CausalDriftHypothesisKindV1,
    CausalDriftPolicyV1, ControlledInterventionEvidenceV1, ExpectedEffectDirectionV1,
    MatchedDifferenceInDifferencesV1, NegativeControlEvidenceV1, diagnose_causal_drift_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn evidence(
    hypothesis_id: &str,
    evidence_id: &str,
    kind: CausalDriftEvidenceKindV1,
) -> CausalDriftEvidenceV1 {
    CausalDriftEvidenceV1 {
        schema: CAUSAL_DRIFT_EVIDENCE_SCHEMA_V1.into(),
        evidence_id: evidence_id.into(),
        evidence_kind: kind,
        hypothesis_id: hypothesis_id.into(),
        event_time_unix_seconds: 100,
        available_at_unix_seconds: 110,
        clock_domain_id: "fab-clock".into(),
        source_artifact_sha256: digest('a'),
        authority_ref: format!("receipt://{evidence_id}"),
        attribution_score: Some(0.9),
    }
}

fn matched_contrast() -> MatchedDifferenceInDifferencesV1 {
    MatchedDifferenceInDifferencesV1 {
        treated_pre_mean: 10.0,
        treated_post_mean: 14.0,
        control_pre_mean: 10.0,
        control_post_mean: 11.0,
        treated_pre_variance: 1.0,
        treated_post_variance: 1.0,
        control_pre_variance: 1.0,
        control_post_variance: 1.0,
        treated_pre_count: 100,
        treated_post_count: 100,
        control_pre_count: 100,
        control_post_count: 100,
        maximum_absolute_covariate_smd: 0.05,
        treated_pretrend_slope: 0.10,
        control_pretrend_slope: 0.11,
        protocol_sha256: digest('b'),
        source_summary_sha256: digest('c'),
    }
}

fn hypothesis(
    id: &str,
    evidence: Vec<CausalDriftEvidenceV1>,
) -> CausalDriftHypothesisInputV1 {
    CausalDriftHypothesisInputV1 {
        hypothesis_id: id.into(),
        hypothesis: CausalDriftHypothesisKindV1::RecipeOrProcessChange,
        expected_effect_direction: ExpectedEffectDirectionV1::Positive,
        scope: "tool-1/chamber-2".into(),
        evidence,
        matched_contrast: None,
        controlled_intervention: None,
        negative_controls: Vec::new(),
        measured_confounders: vec!["recipe_id".into()],
        unmeasured_confounders: vec!["resist_lot".into()],
    }
}

fn input(hypotheses: Vec<CausalDriftHypothesisInputV1>) -> CausalDriftDiagnosisInputV1 {
    CausalDriftDiagnosisInputV1 {
        schema: CAUSAL_DRIFT_INPUT_SCHEMA_V1.into(),
        incident_id: "reference-incident".into(),
        incident_clock_domain_id: "fab-clock".into(),
        incident_time_unix_seconds: 100,
        analysis_cutoff_unix_seconds: 200,
        source_inventory_complete: true,
        source_inventory_sha256: digest('d'),
        clock_join_integrity: true,
        clock_join_audit_sha256: digest('e'),
        policy: CausalDriftPolicyV1 {
            minimum_group_count: 10,
            maximum_covariate_smd: 0.10,
            maximum_pretrend_slope_difference: 0.05,
        },
        hypotheses,
    }
}

#[test]
fn difference_in_differences_matches_independent_known_answer_but_stays_associational() {
    let mut candidate = hypothesis(
        "process-change",
        vec![evidence(
            "process-change",
            "matched-contrast",
            CausalDriftEvidenceKindV1::MatchedContrast,
        )],
    );
    candidate.matched_contrast = Some(matched_contrast());

    let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
    let assessment = &report.status_by_hypothesis[0];
    let estimate = assessment.matched_contrast.as_ref().unwrap();

    // (14 - 10) - (11 - 10) = 3; SE = sqrt(4 * 1/100) = 0.2.
    assert!((estimate.estimate - 3.0).abs() < 1.0e-12);
    assert!((estimate.standard_error - 0.2).abs() < 1.0e-12);
    assert!((estimate.lower_95 - (3.0 - 1.959_963_984_540_054 * 0.2)).abs() < 1.0e-12);
    assert_eq!(assessment.status, CausalAssessmentStatusV1::AssociationOnly);
    assert!(!assessment.causal_claim_authorized);
}

#[test]
fn independent_oracle_confirms_negative_control_falsification() {
    let mut candidate = hypothesis(
        "process-change",
        vec![evidence(
            "process-change",
            "matched-contrast",
            CausalDriftEvidenceKindV1::MatchedContrast,
        )],
    );
    candidate.matched_contrast = Some(matched_contrast());
    candidate.negative_controls.push(NegativeControlEvidenceV1 {
        control_id: "unaffected-target".into(),
        source_artifact_sha256: digest('f'),
        effect_estimate: 1.0,
        standard_error: 0.1,
        materiality_margin: 0.1,
    });

    let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
    assert_eq!(
        report.status_by_hypothesis[0].status,
        CausalAssessmentStatusV1::Unidentifiable
    );
    assert_eq!(
        report.status_by_hypothesis[0].negative_control_violation_ids,
        vec!["unaffected-target".to_string()]
    );
}

#[test]
fn intervention_consistency_requires_every_declared_design_gate() {
    let mut candidate = hypothesis(
        "controlled-repair",
        vec![evidence(
            "controlled-repair",
            "randomized-trial",
            CausalDriftEvidenceKindV1::ControlledIntervention,
        )],
    );
    candidate.controlled_intervention = Some(ControlledInterventionEvidenceV1 {
        randomized_assignment: true,
        preregistered_analysis: true,
        control_group_present: true,
        manipulation_check_passed: true,
        interference_assumption_declared: true,
        treated_count: 100,
        control_count: 100,
        effect_estimate: 2.0,
        standard_error: 0.25,
        assignment_protocol_sha256: digest('1'),
        preregistration_sha256: digest('2'),
        independent_verification_receipt_sha256: digest('3'),
    });
    let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
    assert_eq!(
        report.status_by_hypothesis[0].status,
        CausalAssessmentStatusV1::InterventionConsistent
    );
    assert!(!report.causal_claim_authorized);
    assert!(!report.physical_action_authorized);
}
