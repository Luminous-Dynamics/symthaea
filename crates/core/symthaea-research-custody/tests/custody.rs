use symthaea_research_custody::{
    AccessReceipt, CustodyAction, CustodyAsset, CustodyAssetKind, CustodyError, CustodyPhase,
    CustodyPrincipal, ResearchCustodyManifest,
};
use symthaea_research_split::{
    AssignedUnit, GroupSeparationPolicy, PartitionRole, ResearchSplitManifest, SplitUnit,
    TemporalSeparationPolicy,
};

fn unit(id: &str, role: PartitionRole) -> AssignedUnit {
    AssignedUnit::new(
        SplitUnit::new(id, 1_000, format!("sample:{id}"), vec![]).unwrap(),
        role,
    )
}

fn split() -> ResearchSplitManifest {
    ResearchSplitManifest::new(
        "custody-split",
        vec![
            unit("train", PartitionRole::Training),
            unit("cal", PartitionRole::Calibration),
            unit("eval", PartitionRole::Evaluation),
        ],
        GroupSeparationPolicy::None,
        TemporalSeparationPolicy::None,
        vec![],
    )
    .unwrap()
}

fn manifest() -> ResearchCustodyManifest {
    let split = split();
    ResearchCustodyManifest::new(
        &split,
        "custody-1",
        vec![
            CustodyAsset::evaluation_input(&split, "eval-input", "eval", "asset:input").unwrap(),
            CustodyAsset::evaluation_outcome(
                &split,
                "eval-outcome",
                "eval",
                "asset:outcome",
                CustodyAssetKind::VerificationOutcome,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn evaluation_input_opens_to_model_only_at_input_phase() {
    let manifest = manifest();
    assert!(!manifest
        .is_allowed(
            "eval-input",
            CustodyPrincipal::ModelProcess,
            CustodyAction::Read,
            CustodyPhase::SelectionFrozen,
        )
        .unwrap());
    assert!(manifest
        .is_allowed(
            "eval-input",
            CustodyPrincipal::ModelProcess,
            CustodyAction::Read,
            CustodyPhase::EvaluationInputsOpen,
        )
        .unwrap());
}

#[test]
fn model_cannot_read_held_out_outcome_before_reveal() {
    let manifest = manifest();
    let err = AccessReceipt::new(
        &manifest,
        "r1",
        "eval-outcome",
        CustodyPrincipal::ModelProcess,
        CustodyAction::Read,
        CustodyPhase::EvaluationInputsOpen,
        "evidence:input-open",
        2_000,
    )
    .unwrap_err();
    assert!(matches!(err, CustodyError::AccessDenied { asset_id, .. } if asset_id == "eval-outcome"));
}

#[test]
fn verifier_can_hold_outcome_before_model_and_score_after_reveal() {
    let manifest = manifest();
    AccessReceipt::new(
        &manifest,
        "hold",
        "eval-outcome",
        CustodyPrincipal::Verifier,
        CustodyAction::Read,
        CustodyPhase::SelectionFrozen,
        "evidence:selection-frozen",
        2_000,
    )
    .unwrap();
    AccessReceipt::new(
        &manifest,
        "score",
        "eval-outcome",
        CustodyPrincipal::Verifier,
        CustodyAction::Score,
        CustodyPhase::OutcomeRevealed,
        "evidence:forecast-commit",
        3_000,
    )
    .unwrap();
}

#[test]
fn outcome_asset_must_belong_to_evaluation_partition() {
    let split = split();
    let err = CustodyAsset::evaluation_outcome(
        &split,
        "bad-outcome",
        "cal",
        "asset:bad",
        CustodyAssetKind::GroundTruthLabel,
    )
    .unwrap_err();
    assert!(matches!(err, CustodyError::OutcomeNotEvaluation(id) if id == "cal"));
}

#[test]
fn access_receipt_binds_asset_phase_and_phase_evidence() {
    let manifest = manifest();
    let receipt = AccessReceipt::new(
        &manifest,
        "r",
        "eval-input",
        CustodyPrincipal::ModelProcess,
        CustodyAction::Read,
        CustodyPhase::EvaluationInputsOpen,
        "evidence:selection-receipt",
        2_500,
    )
    .unwrap();
    receipt.verify_against_manifest(&manifest).unwrap();
    assert_eq!(receipt.asset_content_digest, "asset:input");
    assert_eq!(receipt.phase_evidence_digest, "evidence:selection-receipt");
}

#[test]
fn receipt_tampering_is_detected_on_deserialization() {
    let manifest = manifest();
    let receipt = AccessReceipt::new(
        &manifest,
        "r",
        "eval-input",
        CustodyPrincipal::ModelProcess,
        CustodyAction::Read,
        CustodyPhase::EvaluationInputsOpen,
        "evidence:selection-receipt",
        2_500,
    )
    .unwrap();
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["phase"] = serde_json::Value::String("Development".into());
    assert!(serde_json::from_value::<AccessReceipt>(value).is_err());
}

#[test]
fn custody_manifest_revalidates_against_authoritative_split() {
    let split = split();
    let manifest = manifest();
    manifest.verify_against_split(&split).unwrap();

    let different = ResearchSplitManifest::new(
        "different",
        vec![
            unit("train", PartitionRole::Training),
            unit("cal", PartitionRole::Calibration),
            AssignedUnit::new(
                SplitUnit::new("eval", 1_000, "changed:eval", vec![]).unwrap(),
                PartitionRole::Evaluation,
            ),
        ],
        GroupSeparationPolicy::None,
        TemporalSeparationPolicy::None,
        vec![],
    )
    .unwrap();
    assert!(manifest.verify_against_split(&different).is_err());
}

#[test]
fn manifest_tampering_is_rejected_on_deserialization() {
    let manifest = manifest();
    let mut value = serde_json::to_value(&manifest).unwrap();
    value["assets"][0]["asset_content_digest"] = serde_json::Value::String("forged".into());
    assert!(serde_json::from_value::<ResearchCustodyManifest>(value).is_err());
}
