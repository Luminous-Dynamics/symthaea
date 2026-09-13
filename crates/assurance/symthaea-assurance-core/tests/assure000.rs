// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_assurance_core::{
    DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance, NegativeFinding, Claim,
    QualificationOutcome, QualificationPlan, QualificationResult, StableId, SubjectComponent,
    SubjectComponentKind, SubjectManifest, SupportTier,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn subject() -> SubjectManifest {
    SubjectManifest::new(
        id("external-agent-v1"),
        vec![
            SubjectComponent {
                kind: SubjectComponentKind::Model,
                digest: digest('a'),
            },
            SubjectComponent {
                kind: SubjectComponentKind::Policy,
                digest: digest('b'),
            },
        ],
        Some(id("declared-prod-envelope")),
    )
    .unwrap()
}

#[test]
fn public_api_preserves_negative_evidence_as_a_distinct_result_class() {
    let subject = subject();
    let claim = Claim {
        claim_id: id("authority-boundary"),
        subject_id: subject.subject_id(),
        statement: "high-impact tool calls require approval".into(),
        scope: id("registered-tools-v1"),
    };
    let claim_digest = claim.digest();
    let plan = QualificationPlan {
        plan_id: id("plan-v1"),
        claim_digest: claim_digest.clone(),
        maximum_support: SupportTier::FunctionallySupported,
        invalidation_conditions: BTreeSet::new(),
    };
    let evidence = EvidenceArtifact {
        evidence_id: id("evidence-v1"),
        subject_id: subject.subject_id(),
        claim_digest,
        kind: EvidenceKind::ControlledIntervention,
        artifact_digest: digest('c'),
        provenance: EvidenceProvenance {
            producer: id("campaign-owner"),
            executor: id("runner"),
            verifier: Some(id("independent-verifier")),
            signer: None,
        },
    };

    let result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan,
        &[evidence],
        QualificationOutcome::Negative(NegativeFinding::Contradicted),
    )
    .unwrap();

    assert_eq!(result.outcome.support_tier(), None);
    assert!(matches!(
        result.outcome,
        QualificationOutcome::Negative(NegativeFinding::Contradicted)
    ));
}
