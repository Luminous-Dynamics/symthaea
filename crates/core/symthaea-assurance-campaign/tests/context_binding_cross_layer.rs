// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_campaign::{
    CampaignError, CampaignEvidenceLedgerV1, CampaignPlanV1, EvidenceRequirementV1,
    OrderingReceiptV1, PreregistrationReceiptV1, RegistrationStatementV1,
    ReproductionRequirementV1, TerminalRegistrationInViewV1, evidence_commitment_statement_digest,
    resolve_terminal_registration_in_view,
};
use symthaea_assurance_core::{
    AssuranceError, Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance,
    NegativeFinding, QualificationOutcome, QualificationResult, ReproductionStatus, StableId,
    SupportTier,
};
use symthaea_assurance_semantics::{DefinitionSchemaV1, SemanticCommitmentV1};
use symthaea_assurance_subject::{
    AiSubjectManifest, AiSurfaceKind, MaterialCommitment, SurfaceBinding, SurfaceLocator,
    SurfaceProfile, SurfaceState,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn semantic(name: &str, definition_byte: char) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(name),
        DefinitionSchemaV1::new(
            id("symthaea.assurance.test-semantic-schema.v1"),
            digest('e'),
        ),
        digest(definition_byte),
    )
}

fn subject(model_byte: char) -> AiSubjectManifest {
    let profile =
        SurfaceProfile::new(id("context-binding-profile"), vec![AiSurfaceKind::Model]).unwrap();
    AiSubjectManifest::new(
        id("context-binding-agent"),
        profile,
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("model-a"), Some(id("v1"))),
                SurfaceState::Known(MaterialCommitment::artifact_bytes(digest(model_byte))),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn claim(subject: &AiSubjectManifest) -> Claim {
    Claim::new(
        id("context-binding-claim"),
        subject.core_subject_id().unwrap(),
        "evidence belongs to the exact campaign subject and claim",
        id("agent-authority"),
    )
    .unwrap()
}

fn plan(subject: &AiSubjectManifest, claim: &Claim) -> CampaignPlanV1 {
    CampaignPlanV1::new(
        id("context-binding-plan-v1"),
        id("context-binding-campaign"),
        claim,
        subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceRequirementV1::builtin(EvidenceKind::Observation).unwrap()],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap()
}

fn ordering(
    sequence: u64,
    statement_digest: DigestSha256,
    receipt_byte: char,
) -> OrderingReceiptV1 {
    OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic("monotonic-ordering-profile-v1", 'f'),
        1,
        sequence,
        statement_digest,
        digest(receipt_byte),
    )
    .unwrap()
}

fn terminal_registration(plan: &CampaignPlanV1) -> TerminalRegistrationInViewV1 {
    let statement = RegistrationStatementV1::new(plan, id("registrar-a"), None).unwrap();
    let receipt = ordering(5, statement.digest(), 'e');
    let registration = PreregistrationReceiptV1::new(statement, receipt, None).unwrap();
    resolve_terminal_registration_in_view(&[registration], &[]).unwrap()
}

fn evidence(
    evidence_id: &str,
    subject_id: DigestSha256,
    claim_digest: DigestSha256,
    artifact_byte: char,
) -> EvidenceArtifact {
    EvidenceArtifact::new(
        id(evidence_id),
        subject_id,
        claim_digest,
        EvidenceKind::Observation,
        digest(artifact_byte),
        EvidenceProvenance::new(id("producer"), id("executor"), Some(id("verifier")), None),
    )
}

fn campaign_admit(
    plan: &CampaignPlanV1,
    terminal: &TerminalRegistrationInViewV1,
    evidence: &EvidenceArtifact,
) -> Result<(), CampaignError> {
    let mut ledger = CampaignEvidenceLedgerV1::new(terminal);
    let commitment = ordering(
        6,
        evidence_commitment_statement_digest(terminal, evidence),
        '1',
    );
    let admission_statement = ledger.admission_statement_digest(terminal, evidence, &commitment)?;
    let admission = ordering(7, admission_statement, '2');
    ledger
        .admit_preregistered(plan, terminal, evidence, &commitment, &admission)
        .map(|_| ())
}

#[test]
fn assure000_and_assure002_agree_on_exact_evidence_context() {
    let exact_subject = subject('a');
    let exact_claim = claim(&exact_subject);
    let campaign_plan = plan(&exact_subject, &exact_claim);
    let terminal = terminal_registration(&campaign_plan);
    let core_subject = exact_subject.as_core_subject().unwrap();
    let core_plan = campaign_plan.core_plan();

    let valid = evidence(
        "valid-observation",
        exact_subject.core_subject_id().unwrap(),
        exact_claim.digest(),
        'a',
    );
    assert!(
        QualificationResult::validate_and_bind(
            &exact_claim,
            &core_subject,
            &core_plan,
            std::slice::from_ref(&valid),
            QualificationOutcome::Negative(NegativeFinding::NotDemonstrated),
            ReproductionStatus::NotClaimed,
        )
        .is_ok()
    );
    assert_eq!(campaign_admit(&campaign_plan, &terminal, &valid), Ok(()));

    let foreign_subject = subject('b');
    let foreign_subject_claim = claim(&foreign_subject);
    let wrong_subject = evidence(
        "wrong-subject-observation",
        foreign_subject.core_subject_id().unwrap(),
        foreign_subject_claim.digest(),
        'b',
    );
    assert_eq!(
        QualificationResult::validate_and_bind(
            &exact_claim,
            &core_subject,
            &core_plan,
            std::slice::from_ref(&wrong_subject),
            QualificationOutcome::Negative(NegativeFinding::NotDemonstrated),
            ReproductionStatus::NotClaimed,
        )
        .unwrap_err(),
        AssuranceError::SubjectMismatch
    );
    assert_eq!(
        campaign_admit(&campaign_plan, &terminal, &wrong_subject),
        Err(CampaignError::EvidenceSubjectMismatch)
    );

    let foreign_claim = Claim::new(
        id("foreign-claim"),
        exact_subject.core_subject_id().unwrap(),
        "different claim over the same exact subject",
        id("agent-authority"),
    )
    .unwrap();
    let wrong_claim = evidence(
        "wrong-claim-observation",
        exact_subject.core_subject_id().unwrap(),
        foreign_claim.digest(),
        'c',
    );
    assert_eq!(
        QualificationResult::validate_and_bind(
            &exact_claim,
            &core_subject,
            &core_plan,
            std::slice::from_ref(&wrong_claim),
            QualificationOutcome::Negative(NegativeFinding::NotDemonstrated),
            ReproductionStatus::NotClaimed,
        )
        .unwrap_err(),
        AssuranceError::ClaimMismatch
    );
    assert_eq!(
        campaign_admit(&campaign_plan, &terminal, &wrong_claim),
        Err(CampaignError::EvidenceClaimMismatch)
    );
}
