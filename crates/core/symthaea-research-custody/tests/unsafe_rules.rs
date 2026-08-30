use symthaea_research_custody::{
    AccessRule, CustodyAction, CustodyAsset, CustodyAssetKind, CustodyError, CustodyPhase,
    CustodyPrincipal,
};
use symthaea_research_split::{
    AssignedUnit, GroupSeparationPolicy, PartitionRole, ResearchSplitManifest, SplitUnit,
    TemporalSeparationPolicy,
};

fn split() -> ResearchSplitManifest {
    ResearchSplitManifest::new(
        "custody-rule-split",
        vec![
            AssignedUnit::new(
                SplitUnit::new("train", 1_000, "sample:train", vec![]).unwrap(),
                PartitionRole::Training,
            ),
            AssignedUnit::new(
                SplitUnit::new("eval", 2_000, "sample:eval", vec![]).unwrap(),
                PartitionRole::Evaluation,
            ),
        ],
        GroupSeparationPolicy::None,
        TemporalSeparationPolicy::None,
        vec![],
    )
    .unwrap()
}

#[test]
fn custom_rule_cannot_give_model_early_outcome_read() {
    let split = split();
    let err = CustodyAsset::from_split(
        &split,
        "outcome",
        "eval",
        "asset:outcome",
        CustodyAssetKind::VerificationOutcome,
        vec![AccessRule::new(
            CustodyPrincipal::ModelProcess,
            CustodyAction::Read,
            CustodyPhase::EvaluationInputsOpen,
        )],
    )
    .unwrap_err();
    assert!(matches!(err, CustodyError::UnsafeOutcomeRule { .. }));
}

#[test]
fn custom_rule_cannot_score_before_predictions_are_committed() {
    let split = split();
    let err = CustodyAsset::from_split(
        &split,
        "label",
        "eval",
        "asset:label",
        CustodyAssetKind::GroundTruthLabel,
        vec![AccessRule::new(
            CustodyPrincipal::Verifier,
            CustodyAction::Score,
            CustodyPhase::EvaluationInputsOpen,
        )],
    )
    .unwrap_err();
    assert!(matches!(err, CustodyError::UnsafeOutcomeRule { .. }));
}

#[test]
fn verifier_may_hold_hidden_outcome_before_reveal() {
    let split = split();
    CustodyAsset::from_split(
        &split,
        "outcome",
        "eval",
        "asset:outcome",
        CustodyAssetKind::VerificationOutcome,
        vec![
            AccessRule::new(
                CustodyPrincipal::Verifier,
                CustodyAction::Read,
                CustodyPhase::SelectionFrozen,
            ),
            AccessRule::new(
                CustodyPrincipal::Verifier,
                CustodyAction::Score,
                CustodyPhase::OutcomeRevealed,
            ),
        ],
    )
    .unwrap();
}

#[test]
fn public_outcome_reveal_is_not_allowed_before_publication() {
    let split = split();
    let err = CustodyAsset::from_split(
        &split,
        "outcome",
        "eval",
        "asset:outcome",
        CustodyAssetKind::VerificationOutcome,
        vec![AccessRule::new(
            CustodyPrincipal::Public,
            CustodyAction::Reveal,
            CustodyPhase::OutcomeRevealed,
        )],
    )
    .unwrap_err();
    assert!(matches!(err, CustodyError::UnsafeOutcomeRule { .. }));
}
