// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance, QualificationOutcome,
    QualificationPlan, QualificationResult, StableId, SubjectComponent, SubjectComponentKind,
    SubjectManifest, SupportTier,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

#[test]
fn artifact_content_or_provenance_change_changes_result_identity() {
    let subject = SubjectManifest::new(
        id("agent-v1"),
        vec![SubjectComponent {
            kind: SubjectComponentKind::Model,
            digest: digest('a'),
        }],
    )
    .unwrap();
    let claim = Claim::new(
        id("claim-v1"),
        subject.subject_id(),
        "declared behavior is observed",
        id("scope-v1"),
    )
    .unwrap();
    let plan = QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        SupportTier::Observed,
        Default::default(),
    );
    let make = |artifact_digest: DigestSha256, producer: &str| {
        EvidenceArtifact::new(
            id("observation-v1"),
            subject.subject_id(),
            claim.digest(),
            EvidenceKind::Observation,
            artifact_digest,
            EvidenceProvenance::new(id(producer), id("runner"), None, None),
        )
    };

    let first = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan,
        &[make(digest('b'), "producer-a")],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();
    let changed_bytes = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan,
        &[make(digest('c'), "producer-a")],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();
    let changed_provenance = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan,
        &[make(digest('b'), "producer-b")],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();

    assert_ne!(first.digest(), changed_bytes.digest());
    assert_ne!(first.digest(), changed_provenance.digest());
}
