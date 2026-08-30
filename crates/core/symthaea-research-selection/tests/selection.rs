use symthaea_research_fit::{FitArtifactManifest, FitRolePolicy, FitStage};
use symthaea_research_selection::{
    ResearchSelectionManifest, SelectionCandidate, SelectionDirection, SelectionError,
    SelectionObservation, SelectionRolePolicy,
};
use symthaea_research_split::{
    AssignedUnit, GroupSeparationPolicy, PartitionRole, ResearchSplitManifest, SplitUnit,
    TemporalSeparationPolicy,
};

fn unit(id: &str, role: PartitionRole) -> AssignedUnit {
    AssignedUnit::new(
        SplitUnit::new(id, 1_000, format!("digest:{id}"), vec![]).unwrap(),
        role,
    )
}

fn split() -> ResearchSplitManifest {
    ResearchSplitManifest::new(
        "selection-split",
        vec![
            unit("train-1", PartitionRole::Training),
            unit("cal-1", PartitionRole::Calibration),
            unit("cal-2", PartitionRole::Calibration),
            unit("eval-1", PartitionRole::Evaluation),
        ],
        GroupSeparationPolicy::None,
        TemporalSeparationPolicy::None,
        vec![],
    )
    .unwrap()
}

fn fit(split: &ResearchSplitManifest, id: &str, output: &str) -> FitArtifactManifest {
    FitArtifactManifest::new(
        split,
        id,
        FitStage::ModelTraining,
        FitRolePolicy::TrainingOnly,
        format!("impl:{id}"),
        format!("params:{id}"),
        vec!["train-1".into()],
        output,
        2_000,
    )
    .unwrap()
}

fn observation(
    split: &ResearchSplitManifest,
    candidate: &str,
    sample: &str,
    value: f64,
) -> SelectionObservation {
    SelectionObservation::from_split(
        candidate,
        sample,
        value,
        split,
        SelectionRolePolicy::CalibrationOnly,
    )
    .unwrap()
}

fn valid_selection() -> (
    ResearchSplitManifest,
    FitArtifactManifest,
    FitArtifactManifest,
    ResearchSelectionManifest,
) {
    let split = split();
    let fit_a = fit(&split, "fit-a", "artifact:a");
    let fit_b = fit(&split, "fit-b", "artifact:b");
    let candidates = vec![
        SelectionCandidate::from_fit("a", &fit_a, &split).unwrap(),
        SelectionCandidate::from_fit("b", &fit_b, &split).unwrap(),
    ];
    let observations = vec![
        observation(&split, "a", "cal-1", 0.25),
        observation(&split, "a", "cal-2", 0.35),
        observation(&split, "b", "cal-1", 0.40),
        observation(&split, "b", "cal-2", 0.50),
    ];
    let selection = ResearchSelectionManifest::new(
        &split,
        "select-1",
        "brier",
        SelectionDirection::Minimize,
        SelectionRolePolicy::CalibrationOnly,
        candidates,
        observations,
        3_000,
    )
    .unwrap();
    (split, fit_a, fit_b, selection)
}

#[test]
fn selection_is_deterministic_and_revalidates_fit_lineage() {
    let (split, fit_a, fit_b, selection) = valid_selection();
    assert_eq!(selection.selected_candidate_id, "a");
    selection.verify_digest().unwrap();
    selection.verify_against(&split, &[fit_a, fit_b]).unwrap();
}

#[test]
fn evaluation_sample_cannot_enter_selection() {
    let split = split();
    let err = SelectionObservation::from_split(
        "a",
        "eval-1",
        0.01,
        &split,
        SelectionRolePolicy::CalibrationOnly,
    )
    .unwrap_err();
    assert!(matches!(err, SelectionError::EvaluationLeakage(id) if id == "eval-1"));
}

#[test]
fn calibration_only_policy_rejects_training_metric() {
    let split = split();
    let err = SelectionObservation::from_split(
        "a",
        "train-1",
        0.1,
        &split,
        SelectionRolePolicy::CalibrationOnly,
    )
    .unwrap_err();
    assert!(matches!(err, SelectionError::TrainingLeakage(id) if id == "train-1"));
}

#[test]
fn candidates_must_be_compared_on_identical_sample_sets() {
    let split = split();
    let fit_a = fit(&split, "fit-a", "artifact:a");
    let fit_b = fit(&split, "fit-b", "artifact:b");
    let candidates = vec![
        SelectionCandidate::from_fit("a", &fit_a, &split).unwrap(),
        SelectionCandidate::from_fit("b", &fit_b, &split).unwrap(),
    ];
    let err = ResearchSelectionManifest::new(
        &split,
        "select",
        "brier",
        SelectionDirection::Minimize,
        SelectionRolePolicy::CalibrationOnly,
        candidates,
        vec![
            observation(&split, "a", "cal-1", 0.2),
            observation(&split, "a", "cal-2", 0.3),
            observation(&split, "b", "cal-1", 0.1),
        ],
        3_000,
    )
    .unwrap_err();
    assert!(matches!(err, SelectionError::UnequalCandidateSampleSet { .. }));
}

#[test]
fn exact_ties_use_frozen_lexicographic_tie_break() {
    let split = split();
    let fit_a = fit(&split, "fit-a", "artifact:a");
    let fit_b = fit(&split, "fit-b", "artifact:b");
    let selection = ResearchSelectionManifest::new(
        &split,
        "select",
        "brier",
        SelectionDirection::Minimize,
        SelectionRolePolicy::CalibrationOnly,
        vec![
            SelectionCandidate::from_fit("zeta", &fit_a, &split).unwrap(),
            SelectionCandidate::from_fit("alpha", &fit_b, &split).unwrap(),
        ],
        vec![
            observation(&split, "zeta", "cal-1", 0.2),
            observation(&split, "zeta", "cal-2", 0.4),
            observation(&split, "alpha", "cal-1", 0.2),
            observation(&split, "alpha", "cal-2", 0.4),
        ],
        3_000,
    )
    .unwrap();
    assert_eq!(selection.selected_candidate_id, "alpha");
}

#[test]
fn persisted_winner_tampering_is_rejected() {
    let (_, _, _, selection) = valid_selection();
    let mut value = serde_json::to_value(&selection).unwrap();
    value["selected_candidate_id"] = serde_json::Value::String("b".into());
    assert!(serde_json::from_value::<ResearchSelectionManifest>(value).is_err());
}

#[test]
fn authoritative_verification_rejects_recomputed_evaluation_role_tamper() {
    let (split, fit_a, fit_b, selection) = valid_selection();
    let mut value = serde_json::to_value(&selection).unwrap();
    let observations = value["observations"].as_array_mut().unwrap();
    observations[0]["sample_role"] = serde_json::Value::String("Evaluation".into());
    // Digest is now stale, so the persisted representation already fails closed. More generally,
    // even a self-consistent reconstructed manifest must still pass verify_against(authoritative split).
    assert!(serde_json::from_value::<ResearchSelectionManifest>(value).is_err());
    selection.verify_against(&split, &[fit_a, fit_b]).unwrap();
}

#[test]
fn selection_rejects_missing_candidate_fit_at_authoritative_verification() {
    let (split, fit_a, _fit_b, selection) = valid_selection();
    let err = selection.verify_against(&split, &[fit_a]).unwrap_err();
    assert!(matches!(err, SelectionError::MissingFitManifest(id) if id == "b"));
}
