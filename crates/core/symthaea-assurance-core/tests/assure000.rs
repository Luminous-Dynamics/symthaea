// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{
    AssuranceError, Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance,
    NegativeFinding, QualificationOutcome, QualificationPlan, QualificationResult,
    ReproductionStatus, StableId, SubjectComponent, SubjectComponentKind, SubjectManifest,
    SupportTier,
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
                kind: SubjectComponentKind::Policy,
                digest: digest('b'),
            },
            SubjectComponent {
                kind: SubjectComponentKind::Model,
                digest: digest('a'),
            },
        ],
    )
    .unwrap()
}

fn claim(subject: &SubjectManifest) -> Claim {
    Claim::new(
        id("authority-boundary"),
        subject.subject_id(),
        "high-impact tool calls require approval",
        id("registered-tools-v1"),
    )
    .unwrap()
}

fn plan(claim: &Claim, maximum_support: SupportTier) -> QualificationPlan {
    QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        maximum_support,
        [id("model-changed"), id("policy-changed")]
            .into_iter()
            .collect(),
    )
}

fn artifact(
    evidence_id: &str,
    subject: &SubjectManifest,
    claim: &Claim,
    kind: EvidenceKind,
    verifier: Option<&str>,
    digest_char: char,
) -> EvidenceArtifact {
    EvidenceArtifact::new(
        id(evidence_id),
        subject.subject_id(),
        claim.digest(),
        kind,
        digest(digest_char),
        EvidenceProvenance::new(id("campaign-owner"), id("runner"), verifier.map(id), None),
    )
}

fn functional_set(subject: &SubjectManifest, claim: &Claim) -> Vec<EvidenceArtifact> {
    vec![
        artifact(
            "causal",
            subject,
            claim,
            EvidenceKind::ControlledIntervention,
            None,
            'c',
        ),
        artifact(
            "functional",
            subject,
            claim,
            EvidenceKind::FunctionalBenchmark,
            None,
            'd',
        ),
    ]
}

#[test]
fn subject_identity_is_order_independent() {
    let left = subject();
    let right = SubjectManifest::new(
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
    )
    .unwrap();
    assert_eq!(left.subject_id(), right.subject_id());
}

#[test]
fn duplicate_subject_component_kind_fails_closed() {
    let error = SubjectManifest::new(
        id("duplicate"),
        vec![
            SubjectComponent {
                kind: SubjectComponentKind::Model,
                digest: digest('a'),
            },
            SubjectComponent {
                kind: SubjectComponentKind::Model,
                digest: digest('b'),
            },
        ],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AssuranceError::DuplicateSubjectComponent(_)
    ));
}

#[test]
fn plan_identity_is_invalidation_order_independent() {
    let subject = subject();
    let claim = claim(&subject);
    let left = QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        SupportTier::Observed,
        [id("z-change"), id("a-change")].into_iter().collect(),
    );
    let right = QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        SupportTier::Observed,
        [id("a-change"), id("z-change")].into_iter().collect(),
    );
    assert_eq!(left.digest(), right.digest());
}

#[test]
fn claim_ceiling_blocks_overpromotion_before_evidence_interpretation() {
    let subject = subject();
    let claim = claim(&subject);
    let observation = artifact(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let error = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::CausallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap_err();
    assert_eq!(error, AssuranceError::ClaimCeilingExceeded);
}

#[test]
fn stronger_tier_requires_explicit_evidence_kind() {
    let subject = subject();
    let claim = claim(&subject);
    let architecture = artifact(
        "architecture",
        &subject,
        &claim,
        EvidenceKind::ArchitectureInspection,
        None,
        'c',
    );
    let error = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::CausallySupported),
        &[architecture],
        QualificationOutcome::Supported(SupportTier::CausallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AssuranceError::InsufficientEvidenceForTier(_)
    ));
}

#[test]
fn negative_finding_remains_orthogonal_to_positive_support() {
    let subject = subject();
    let claim = claim(&subject);
    let artifacts = functional_set(&subject, &claim);
    let result = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Negative(NegativeFinding::Contradicted),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    assert_eq!(result.outcome().support_tier(), None);
    assert!(matches!(
        result.outcome(),
        QualificationOutcome::Negative(NegativeFinding::Contradicted)
    ));
}

#[test]
fn reproduction_evidence_requires_distinct_verifier_identity() {
    let subject = subject();
    let claim = claim(&subject);
    let mut artifacts = functional_set(&subject, &claim);
    artifacts.push(artifact(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::Reproduction,
        None,
        'e',
    ));
    let error = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::EvidenceFromDistinctVerifier,
    )
    .unwrap_err();
    assert_eq!(error, AssuranceError::MissingDistinctVerifier);
}

#[test]
fn reproduction_evidence_is_orthogonal_to_support_strength() {
    let subject = subject();
    let claim = claim(&subject);
    let mut artifacts = functional_set(&subject, &claim);
    artifacts.push(artifact(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::Reproduction,
        Some("second-verifier-identity"),
        'e',
    ));
    let result = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::EvidenceFromDistinctVerifier,
    )
    .unwrap();
    assert_eq!(
        result.outcome().support_tier(),
        Some(SupportTier::FunctionallySupported)
    );
    assert_eq!(
        result.reproduction_status(),
        ReproductionStatus::EvidenceFromDistinctVerifier
    );
}

#[test]
fn reproduction_status_is_bound_into_result_identity() {
    let subject = subject();
    let claim = claim(&subject);
    let mut artifacts = functional_set(&subject, &claim);
    artifacts.push(artifact(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::Reproduction,
        Some("second-verifier-identity"),
        'e',
    ));
    let not_claimed = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    let with_reproduction_evidence = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::EvidenceFromDistinctVerifier,
    )
    .unwrap();
    assert_ne!(not_claimed.digest(), with_reproduction_evidence.digest());
}

#[test]
fn runtime_evidence_does_not_promote_support_or_authority() {
    let subject = subject();
    let claim = claim(&subject);
    let mut artifacts = functional_set(&subject, &claim);
    artifacts.push(artifact(
        "runtime",
        &subject,
        &claim,
        EvidenceKind::RuntimeReceipt,
        None,
        'f',
    ));
    let result = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &artifacts,
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    assert_eq!(
        result.outcome().support_tier(),
        Some(SupportTier::FunctionallySupported)
    );
    assert_eq!(result.reproduction_status(), ReproductionStatus::NotClaimed);
}

#[test]
fn duplicate_evidence_identity_fails_closed() {
    let subject = subject();
    let claim = claim(&subject);
    let left = artifact(
        "duplicate",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let right = artifact(
        "duplicate",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'd',
    );
    let error = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[left, right],
        QualificationOutcome::Supported(SupportTier::Observed),
        ReproductionStatus::NotClaimed,
    )
    .unwrap_err();
    assert!(matches!(error, AssuranceError::DuplicateEvidenceId(_)));
}

#[test]
fn evidence_cannot_be_rebound_to_another_claim() {
    let subject = subject();
    let first_claim = claim(&subject);
    let second_claim = Claim::new(
        id("different-claim"),
        subject.subject_id(),
        "a different proposition",
        id("same-scope"),
    )
    .unwrap();
    let foreign = artifact(
        "foreign",
        &subject,
        &first_claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let error = QualificationResult::validate_and_bind(
        &second_claim,
        &subject,
        &plan(&second_claim, SupportTier::Observed),
        &[foreign],
        QualificationOutcome::Supported(SupportTier::Observed),
        ReproductionStatus::NotClaimed,
    )
    .unwrap_err();
    assert_eq!(error, AssuranceError::ClaimMismatch);
}

#[test]
fn result_identity_is_evidence_order_independent() {
    let subject = subject();
    let claim = claim(&subject);
    let causal = artifact(
        "causal",
        &subject,
        &claim,
        EvidenceKind::ControlledIntervention,
        None,
        'c',
    );
    let functional = artifact(
        "functional",
        &subject,
        &claim,
        EvidenceKind::FunctionalBenchmark,
        None,
        'd',
    );
    let left = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &[causal.clone(), functional.clone()],
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    let right = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &[functional, causal],
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    assert_eq!(left.digest(), right.digest());
}

#[test]
fn invalidation_returns_a_new_result_and_preserves_original() {
    let subject = subject();
    let claim = claim(&subject);
    let observation = artifact(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let result = QualificationResult::validate_and_bind(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::Observed),
        ReproductionStatus::NotClaimed,
    )
    .unwrap();
    let before = result.digest();
    let invalidated = result.invalidated_by(&id("model-changed")).unwrap();
    assert_eq!(before, result.digest());
    assert!(matches!(
        result.outcome(),
        QualificationOutcome::Supported(SupportTier::Observed)
    ));
    assert_ne!(before, invalidated.digest());
    assert!(matches!(
        invalidated.outcome(),
        QualificationOutcome::Negative(NegativeFinding::Invalidated { .. })
    ));
    assert!(result.invalidated_by(&id("unregistered-change")).is_none());
}
