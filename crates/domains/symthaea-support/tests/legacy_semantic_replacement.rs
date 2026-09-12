use std::collections::BTreeMap;

use symthaea_support::{
    build_legacy_five_platform_portfolio_v1,
    build_legacy_qualification_successor_manifest_v1,
    initial_legacy_qualification_manifest_v1,
    legacy_procedure_replacement_review_binding_v1,
    legacy_qualification_manifest_commitment_v1,
    require_legacy_semantic_replacement_equivalence_v1,
    LegacyProcedureReplacementReviewReceiptV1,
    LegacySemanticReplacementErrorV1,
    LegacySemanticReplacementJudgmentV1,
    LegacySemanticReplacementReviewLedgerV1,
    LegacySemanticReviewMethodV1,
};

fn procedure_rebaseline(
    judgment: LegacySemanticReplacementJudgmentV1,
) -> (
    symthaea_support::LegacyComputingPackV1,
    symthaea_support::LegacyQualificationManifestV1,
    symthaea_support::LegacyQualificationManifestV1,
    LegacyProcedureReplacementReviewReceiptV1,
) {
    let (mut pack, _, _) =
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
    let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
    let old = pack.procedures.first().unwrap().clone();
    let mut new = old.clone();
    new.id = format!("{}:rebaseline", old.id);
    new.title = format!("{} — re-reviewed baseline", old.title);
    pack.procedures.push(new.clone());
    pack.validate().unwrap();

    let successor = build_legacy_qualification_successor_manifest_v1(
        &pack,
        &predecessor,
        BTreeMap::new(),
        BTreeMap::from([(old.id.clone(), new.id.clone())]),
    )
    .unwrap();

    let predecessor_digest = legacy_qualification_manifest_commitment_v1(&predecessor).unwrap();
    let successor_digest = legacy_qualification_manifest_commitment_v1(&successor).unwrap();
    let review_basis = "b".repeat(64);
    let reviewed_at = 1_800_000_000_200;
    let binding = legacy_procedure_replacement_review_binding_v1(
        &old,
        &new,
        &predecessor_digest,
        &successor_digest,
        judgment,
        LegacySemanticReviewMethodV1::HumanReview,
        "legacy-procedure-semantic-review-v1",
        &review_basis,
        reviewed_at,
    )
    .unwrap();

    let receipt = LegacyProcedureReplacementReviewReceiptV1 {
        predecessor_procedure_id: old.id,
        successor_procedure_id: new.id,
        predecessor_manifest_blake3: predecessor_digest,
        successor_manifest_blake3: successor_digest,
        judgment,
        review_method: LegacySemanticReviewMethodV1::HumanReview,
        reviewer_profile: "legacy-procedure-semantic-review-v1".into(),
        review_basis_blake3: review_basis,
        reviewed_at_unix_ms: reviewed_at,
        replacement_binding_blake3: binding,
    };
    (pack, predecessor, successor, receipt)
}

#[test]
fn equivalent_procedure_receipt_allows_role_carry_forward() {
    let (pack, predecessor, successor, receipt) =
        procedure_rebaseline(LegacySemanticReplacementJudgmentV1::Equivalent);
    let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
    assert!(reviews
        .register_procedure_receipt(&pack, &predecessor, &successor, receipt)
        .unwrap());
    let assessment = require_legacy_semantic_replacement_equivalence_v1(
        &pack,
        &predecessor,
        &successor,
        &reviews,
    )
    .unwrap();
    assert!(assessment.complete);
    assert!(assessment.semantically_equivalent);
    assert_eq!(assessment.required_procedure_receipts, 1);
}

#[test]
fn broader_procedure_cannot_inherit_existing_qualification_role() {
    let (pack, predecessor, successor, receipt) =
        procedure_rebaseline(LegacySemanticReplacementJudgmentV1::Broader);
    let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
    reviews
        .register_procedure_receipt(&pack, &predecessor, &successor, receipt)
        .unwrap();
    assert!(matches!(
        require_legacy_semantic_replacement_equivalence_v1(
            &pack,
            &predecessor,
            &successor,
            &reviews,
        ),
        Err(LegacySemanticReplacementErrorV1::NonEquivalentReplacement { .. })
    ));
}

#[test]
fn tampered_procedure_binding_is_rejected() {
    let (pack, predecessor, successor, mut receipt) =
        procedure_rebaseline(LegacySemanticReplacementJudgmentV1::Equivalent);
    receipt.replacement_binding_blake3 = "c".repeat(64);
    let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
    assert!(matches!(
        reviews.register_procedure_receipt(&pack, &predecessor, &successor, receipt),
        Err(LegacySemanticReplacementErrorV1::BindingMismatch(_))
    ));
}
