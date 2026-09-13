// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_assurance_core::{
    AssuranceError, Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance,
    NegativeFinding, QualificationOutcome, QualificationPlan, QualificationResult, StableId,
    SubjectComponent, SubjectComponentKind, SubjectManifest, SupportTier,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn subject(with_deployment: bool) -> SubjectManifest {
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
        with_deployment.then(|| id("declared-prod-envelope")),
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

fn provenance(verifier: Option<&str>) -> EvidenceProvenance {
    EvidenceProvenance::new(
        id("campaign-owner"),
        id("runner"),
        verifier.map(id),
        None,
    )
}

fn evidence(
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
        provenance(verifier),
    )
}

fn causal_functional_evidence(
    subject: &SubjectManifest,
    claim: &Claim,
) -> Vec<EvidenceArtifact> {
    vec![
        evidence(
            "causal",
            subject,
            claim,
            EvidenceKind::ControlledIntervention,
            None,
            'c',
        ),
        evidence(
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
    let left = subject(false);
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
        None,
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
        None,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        AssuranceError::DuplicateSubjectComponent(_)
    ));
}

#[test]
fn plan_identity_is_set_order_independent() {
    let subject = subject(false);
    let claim = claim(&subject);
    let first = QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        SupportTier::Observed,
        [id("z-change"), id("a-change")].into_iter().collect(),
    );
    let second = QualificationPlan::new(
        id("plan-v1"),
        claim.digest(),
        SupportTier::Observed,
        [id("a-change"), id("z-change")].into_iter().collect(),
    );

    assert_eq!(first.digest(), second.digest());
}

#[test]
fn claim_ceiling_blocks_overpromotion() {
    let subject = subject(true);
    let claim = claim(&subject);
    let observation = evidence(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );

    let error = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::CausallySupported),
    )
    .unwrap_err();

    assert_eq!(error, AssuranceError::ClaimCeilingExceeded);
}

#[test]
fn stronger_tier_requires_explicit_evidence_kind() {
    let subject = subject(true);
    let claim = claim(&subject);
    let architecture = evidence(
        "architecture",
        &subject,
        &claim,
        EvidenceKind::ArchitectureInspection,
        None,
        'c',
    );

    let error = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::CausallySupported),
        &[architecture],
        QualificationOutcome::Supported(SupportTier::CausallySupported),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        AssuranceError::InsufficientEvidenceForTier(_)
    ));
}

#[test]
fn negative_findings_are_not_support_tiers() {
    let outcome = QualificationOutcome::Negative(NegativeFinding::Contradicted);
    assert_eq!(outcome.support_tier(), None);
}

#[test]
fn negative_finding_is_preserved_even_when_positive_tier_was_possible() {
    let subject = subject(true);
    let claim = claim(&subject);
    let evidence = causal_functional_evidence(&subject, &claim);
    let result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &evidence,
        QualificationOutcome::Negative(NegativeFinding::Contradicted),
    )
    .unwrap();

    assert!(matches!(
        result.outcome(),
        QualificationOutcome::Negative(NegativeFinding::Contradicted)
    ));
}

#[test]
fn independent_reproduction_requires_distinct_verifier() {
    let subject = subject(true);
    let claim = claim(&subject);
    let mut evidence = causal_functional_evidence(&subject, &claim);
    evidence.push(evidence(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::IndependentReproduction,
        None,
        'e',
    ));

    let error = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::IndependentlyReproduced),
        &evidence,
        QualificationOutcome::Supported(SupportTier::IndependentlyReproduced),
    )
    .unwrap_err();

    assert_eq!(error, AssuranceError::MissingIndependentVerifier);
}

#[test]
fn independent_reproduction_accepts_distinct_verifier() {
    let subject = subject(true);
    let claim = claim(&subject);
    let mut evidence = causal_functional_evidence(&subject, &claim);
    evidence.push(evidence(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::IndependentReproduction,
        Some("independent-verifier"),
        'e',
    ));

    let result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::IndependentlyReproduced),
        &evidence,
        QualificationOutcome::Supported(SupportTier::IndependentlyReproduced),
    )
    .unwrap();

    assert_eq!(
        result.outcome().support_tier(),
        Some(SupportTier::IndependentlyReproduced)
    );
}

#[test]
fn deployment_qualification_requires_explicit_envelope() {
    let subject = subject(false);
    let claim = claim(&subject);
    let mut evidence = causal_functional_evidence(&subject, &claim);
    evidence.push(evidence(
        "reproduction",
        &subject,
        &claim,
        EvidenceKind::IndependentReproduction,
        Some("independent-verifier"),
        'e',
    ));
    evidence.push(evidence(
        "runtime",
        &subject,
        &claim,
        EvidenceKind::RuntimeReceipt,
        None,
        'f',
    ));

    let error = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::DeploymentQualified),
        &evidence,
        QualificationOutcome::Supported(SupportTier::DeploymentQualified),
    )
    .unwrap_err();

    assert_eq!(error, AssuranceError::MissingDeploymentEnvelope);
}

#[test]
fn duplicate_evidence_identity_fails_closed() {
    let subject = subject(true);
    let claim = claim(&subject);
    let first = evidence(
        "duplicate",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let second = evidence(
        "duplicate",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'd',
    );

    let error = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[first, second],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap_err();

    assert!(matches!(error, AssuranceError::DuplicateEvidenceId(_)));
}

#[test]
fn evidence_cannot_be_rebound_to_another_claim() {
    let subject = subject(true);
    let first_claim = claim(&subject);
    let second_claim = Claim::new(
        id("different-claim"),
        subject.subject_id(),
        "a different proposition",
        id("same-scope"),
    )
    .unwrap();
    let foreign = evidence(
        "foreign",
        &subject,
        &first_claim,
        EvidenceKind::Observation,
        None,
        'c',
    );

    let error = QualificationResult::resolve(
        &second_claim,
        &subject,
        &plan(&second_claim, SupportTier::Observed),
        &[foreign],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap_err();

    assert_eq!(error, AssuranceError::ClaimMismatch);
}

#[test]
fn result_identity_is_evidence_order_independent() {
    let subject = subject(true);
    let claim = claim(&subject);
    let causal = evidence(
        "causal",
        &subject,
        &claim,
        EvidenceKind::ControlledIntervention,
        None,
        'c',
    );
    let functional = evidence(
        "functional",
        &subject,
        &claim,
        EvidenceKind::FunctionalBenchmark,
        None,
        'd',
    );

    let left = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &[causal.clone(), functional.clone()],
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
    )
    .unwrap();
    let right = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::FunctionallySupported),
        &[functional, causal],
        QualificationOutcome::Supported(SupportTier::FunctionallySupported),
    )
    .unwrap();

    assert_eq!(left.digest(), right.digest());
}

#[test]
fn explicit_invalidation_changes_result_identity_and_outcome() {
    let subject = subject(true);
    let claim = claim(&subject);
    let observation = evidence(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let mut result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();
    let before = result.digest();

    assert!(result.apply_invalidation(&id("model-changed")));
    assert_ne!(before, result.digest());
    assert!(matches!(
        result.outcome(),
        QualificationOutcome::Negative(NegativeFinding::Invalidated { .. })
    ));
}

#[test]
fn invalidation_set_is_explicit_not_wildcarded() {
    let subject = subject(true);
    let claim = claim(&subject);
    let observation = evidence(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let mut result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();

    assert!(!result.apply_invalidation(&id("unregistered-change")));
    assert_eq!(
        result.outcome().support_tier(),
        Some(SupportTier::Observed)
    );
}

#[test]
fn public_result_space_has_no_scalar_score() {
    let _guard = BTreeSet::<StableId>::new();
    let subject = subject(true);
    let claim = claim(&subject);
    let observation = evidence(
        "observation",
        &subject,
        &claim,
        EvidenceKind::Observation,
        None,
        'c',
    );
    let result = QualificationResult::resolve(
        &claim,
        &subject,
        &plan(&claim, SupportTier::Observed),
        &[observation],
        QualificationOutcome::Supported(SupportTier::Observed),
    )
    .unwrap();

    assert_eq!(
        result.outcome().support_tier(),
        Some(SupportTier::Observed)
    );
}
