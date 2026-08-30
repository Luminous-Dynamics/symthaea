use symthaea_research_fit::{
    FitArtifactManifest, FitError, FitRolePolicy, FitStage, TransformReceipt,
};
use symthaea_research_split::{
    AssignedUnit, GroupRef, GroupSeparationPolicy, PartitionRole, ResearchSplitManifest, SplitUnit,
    TemporalSeparationPolicy,
};

fn assigned(id: &str, time: i64, group: &str, role: PartitionRole) -> AssignedUnit {
    AssignedUnit::new(
        SplitUnit::new(
            id,
            time,
            format!("digest:{id}"),
            vec![GroupRef::new("spatial-block", group).unwrap()],
        )
        .unwrap(),
        role,
    )
}

fn split() -> ResearchSplitManifest {
    ResearchSplitManifest::new(
        "fit-split",
        vec![
            assigned("train-a", 1_000, "a", PartitionRole::Training),
            assigned("train-b", 1_100, "b", PartitionRole::Training),
            assigned("cal-a", 1_200, "c", PartitionRole::Calibration),
            assigned("eval-a", 2_000, "d", PartitionRole::Evaluation),
        ],
        GroupSeparationPolicy::EvaluationDisjoint {
            dimensions: vec!["spatial-block".into()],
        },
        TemporalSeparationPolicy::ForwardEvaluation { embargo_ms: 500 },
        vec![],
    )
    .unwrap()
}

fn training_fit(split: &ResearchSplitManifest) -> FitArtifactManifest {
    FitArtifactManifest::new(
        split,
        "normalizer-v1",
        FitStage::Preprocessing,
        FitRolePolicy::TrainingOnly,
        "digest:normalizer-code",
        "digest:normalizer-config",
        vec!["train-a".into(), "train-b".into()],
        "digest:fitted-normalizer",
        1_300,
    )
    .unwrap()
}

#[test]
fn training_only_fit_is_content_addressed_and_revalidates_against_split() {
    let split = split();
    let fit = training_fit(&split);
    fit.verify_digest().unwrap();
    fit.verify_against_split(&split).unwrap();
    assert_eq!(fit.influences.len(), 2);
    assert!(fit
        .influences
        .iter()
        .all(|influence| influence.role == PartitionRole::Training));
}

#[test]
fn evaluation_data_cannot_influence_global_preprocessing() {
    let split = split();
    let result = FitArtifactManifest::new(
        &split,
        "global-standardizer",
        FitStage::Preprocessing,
        FitRolePolicy::TrainingAndCalibration,
        "digest:code",
        "digest:params",
        vec!["train-a".into(), "eval-a".into()],
        "digest:artifact",
        1_300,
    );
    assert!(matches!(
        result,
        Err(FitError::EvaluationLeakage(sample)) if sample == "eval-a"
    ));
}

#[test]
fn training_only_policy_rejects_calibration_influence() {
    let split = split();
    let result = FitArtifactManifest::new(
        &split,
        "pca-training-only",
        FitStage::RepresentationLearning,
        FitRolePolicy::TrainingOnly,
        "digest:pca-code",
        "digest:pca-config",
        vec!["train-a".into(), "cal-a".into()],
        "digest:pca-artifact",
        1_300,
    );
    assert!(matches!(
        result,
        Err(FitError::CalibrationLeakage(sample)) if sample == "cal-a"
    ));
}

#[test]
fn calibration_policy_allows_declared_calibration_without_allowing_evaluation() {
    let split = split();
    let fit = FitArtifactManifest::new(
        &split,
        "threshold-selection",
        FitStage::ThresholdSelection,
        FitRolePolicy::TrainingAndCalibration,
        "digest:threshold-code",
        "digest:threshold-search-space",
        vec!["train-a".into(), "cal-a".into()],
        "digest:selected-threshold",
        1_300,
    )
    .unwrap();
    fit.verify_against_split(&split).unwrap();
    assert!(fit
        .influences
        .iter()
        .any(|influence| influence.role == PartitionRole::Calibration));
}

#[test]
fn duplicate_influence_ids_fail_closed() {
    let split = split();
    let result = FitArtifactManifest::new(
        &split,
        "dup",
        FitStage::ModelTraining,
        FitRolePolicy::TrainingOnly,
        "digest:code",
        "digest:params",
        vec!["train-a".into(), "train-a".into()],
        "digest:artifact",
        1_300,
    );
    assert!(matches!(
        result,
        Err(FitError::DuplicateInfluence(sample)) if sample == "train-a"
    ));
}

#[test]
fn persisted_fit_detects_internal_digest_mutation() {
    let split = split();
    let mut fit = training_fit(&split);
    fit.output_artifact_digest.push_str("-mutated");
    assert_eq!(fit.verify_digest().unwrap_err(), FitError::FitDigestMismatch);
}

#[test]
fn persisted_fit_rejects_tampered_serialized_role_even_with_stale_digest() {
    let split = split();
    let fit = training_fit(&split);
    let mut json = serde_json::to_value(&fit).unwrap();
    json["influences"][0]["role"] = serde_json::json!("Evaluation");
    let decoded = serde_json::from_value::<FitArtifactManifest>(json);
    assert!(decoded.is_err());
}

#[test]
fn fit_must_be_revalidated_against_the_exact_split_lineage() {
    let split = split();
    let fit = training_fit(&split);

    let different_split = ResearchSplitManifest::new(
        "other-split",
        vec![
            assigned("train-a", 1_000, "a", PartitionRole::Training),
            assigned("eval-a", 2_000, "d", PartitionRole::Evaluation),
        ],
        GroupSeparationPolicy::EvaluationDisjoint {
            dimensions: vec!["spatial-block".into()],
        },
        TemporalSeparationPolicy::ForwardEvaluation { embargo_ms: 500 },
        vec![],
    )
    .unwrap();

    assert_eq!(
        fit.verify_against_split(&different_split).unwrap_err(),
        FitError::SplitManifestMismatch
    );
}

#[test]
fn frozen_fit_can_be_applied_to_evaluation_without_making_evaluation_a_fit_influence() {
    let split = split();
    let fit = training_fit(&split);
    let receipt = TransformReceipt::new(
        "eval-transform",
        &fit,
        &split,
        "eval-a",
        "digest:normalized-eval-a",
        2_100,
    )
    .unwrap();
    receipt.verify_digest().unwrap();
    assert_eq!(receipt.sample_role, PartitionRole::Evaluation);
    assert!(fit
        .influences
        .iter()
        .all(|influence| influence.sample_id != "eval-a"));
}

#[test]
fn transform_receipt_rejects_unknown_sample() {
    let split = split();
    let fit = training_fit(&split);
    let result = TransformReceipt::new(
        "unknown-transform",
        &fit,
        &split,
        "not-in-split",
        "digest:output",
        2_100,
    );
    assert!(matches!(
        result,
        Err(FitError::UnknownSample(sample)) if sample == "not-in-split"
    ));
}
