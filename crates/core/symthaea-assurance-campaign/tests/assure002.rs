// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_campaign::{
    CampaignError, CampaignEvidenceLedgerV1, CampaignPlanV1, EvidenceTimingClass,
    OrderingReceiptV1, PreregistrationReceiptV1, RegistrationStatementV1,
    RegistrationWithdrawalV1, ReproductionRequirementV1, WithdrawalStatementV1,
    classify_evidence_timing, evidence_production_statement_digest, resolve_current_registration,
};
use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance, StableId, SupportTier,
};
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

fn subject(model_byte: char) -> AiSubjectManifest {
    let profile = SurfaceProfile::new(id("campaign-test-profile"), vec![AiSurfaceKind::Model]).unwrap();
    AiSubjectManifest::new(
        id("campaign-test-agent"),
        profile,
        vec![SurfaceBinding::applicable(
            AiSurfaceKind::Model,
            SurfaceLocator::new(Some(id("provider-a")), id("model-a"), Some(id("v1"))),
            SurfaceState::Known(MaterialCommitment::artifact_bytes(digest(model_byte))),
        )
        .unwrap()],
    )
    .unwrap()
}

fn claim(subject: &AiSubjectManifest) -> Claim {
    Claim::new(
        id("authority-boundary-claim"),
        subject.core_subject_id().unwrap(),
        "high-impact action requires registered approval authority",
        id("agent-authority"),
    )
    .unwrap()
}

fn plan(subject: &AiSubjectManifest, campaign_nonce: &str) -> CampaignPlanV1 {
    let claim = claim(subject);
    CampaignPlanV1::new(
        id("authority-plan-v1"),
        id(campaign_nonce),
        &claim,
        subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation, EvidenceKind::ControlledIntervention],
        vec![id("baseline"), id("denied-path"), id("valid-authority")],
        vec![id("allowed-path-broken")],
        vec![id("unauthorized-execution-observed")],
        vec![id("instrumentation-incomplete")],
        vec![id("subject-drift"), id("policy-drift")],
    )
    .unwrap()
}

fn revised_plan(subject: &AiSubjectManifest, campaign_nonce: &str) -> CampaignPlanV1 {
    let claim = claim(subject);
    CampaignPlanV1::new(
        id("authority-plan-v2"),
        id(campaign_nonce),
        &claim,
        subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation, EvidenceKind::ControlledIntervention],
        vec![id("baseline"), id("denied-path"), id("matched-sham")],
        vec![id("allowed-path-broken")],
        vec![id("unauthorized-execution-observed")],
        vec![id("instrumentation-incomplete")],
        vec![id("subject-drift"), id("policy-drift")],
    )
    .unwrap()
}

fn ordering_with_profile(
    source: &str,
    validation_profile: &str,
    epoch: u64,
    sequence: u64,
    statement: DigestSha256,
    receipt_byte: char,
) -> OrderingReceiptV1 {
    OrderingReceiptV1::new(
        id(source),
        id(validation_profile),
        epoch,
        sequence,
        statement,
        digest(receipt_byte),
    )
    .unwrap()
}

fn ordering(
    source: &str,
    epoch: u64,
    sequence: u64,
    statement: DigestSha256,
    receipt_byte: char,
) -> OrderingReceiptV1 {
    ordering_with_profile(
        source,
        "monotonic-ordering-profile-v1",
        epoch,
        sequence,
        statement,
        receipt_byte,
    )
}

fn root_registration(plan: &CampaignPlanV1, sequence: u64) -> PreregistrationReceiptV1 {
    let statement = RegistrationStatementV1::new(plan, id("registrar-a"), None).unwrap();
    let receipt = ordering("transparency-log-a", 1, sequence, statement.digest(), 'e');
    PreregistrationReceiptV1::new(statement, receipt, None).unwrap()
}

fn successor_registration(
    plan: &CampaignPlanV1,
    predecessor: &PreregistrationReceiptV1,
    sequence: u64,
) -> PreregistrationReceiptV1 {
    let statement =
        RegistrationStatementV1::new(plan, id("registrar-a"), Some(predecessor)).unwrap();
    let receipt = ordering("transparency-log-a", 1, sequence, statement.digest(), 'f');
    PreregistrationReceiptV1::new(statement, receipt, Some(predecessor)).unwrap()
}

fn evidence(subject: &AiSubjectManifest, kind: EvidenceKind, evidence_id: &str) -> EvidenceArtifact {
    let claim = claim(subject);
    EvidenceArtifact::new(
        id(evidence_id),
        subject.core_subject_id().unwrap(),
        claim.digest(),
        kind,
        digest('c'),
        EvidenceProvenance::new(id("producer"), id("executor"), Some(id("verifier")), None),
    )
}

#[test]
fn plan_identity_is_order_independent() {
    let subject = subject('a');
    let claim = claim(&subject);
    let left = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::ControlledIntervention, EvidenceKind::Observation],
        vec![id("b"), id("a")],
        vec![id("failure-b"), id("failure-a")],
        vec![id("contradiction")],
        vec![id("inconclusive")],
        vec![id("invalidate-b"), id("invalidate-a")],
    )
    .unwrap();
    let right = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation, EvidenceKind::ControlledIntervention],
        vec![id("a"), id("b")],
        vec![id("failure-a"), id("failure-b")],
        vec![id("contradiction")],
        vec![id("inconclusive")],
        vec![id("invalidate-a"), id("invalidate-b")],
    )
    .unwrap();
    assert_eq!(left.digest(), right.digest());
    assert_eq!(left.core_plan().digest(), right.core_plan().digest());
}

#[test]
fn duplicate_plan_semantics_fail_closed() {
    let subject = subject('a');
    let claim = claim(&subject);
    let duplicate_kind = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation, EvidenceKind::Observation],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(duplicate_kind, CampaignError::DuplicateEvidenceKind(_)));

    let duplicate_control = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation],
        vec![id("same"), id("same")],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(duplicate_control, CampaignError::DuplicateStableId { .. }));
}

#[test]
fn claim_must_bind_exact_assure001_subject() {
    let subject_a = subject('a');
    let subject_b = subject('b');
    let claim_a = claim(&subject_a);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim_a,
        &subject_b,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert_eq!(error, CampaignError::ClaimSubjectMismatch);
}

#[test]
fn subject_or_campaign_change_changes_plan_identity() {
    let subject_a = subject('a');
    let subject_b = subject('b');
    let plan_a = plan(&subject_a, "campaign-a");
    let plan_b = plan(&subject_b, "campaign-a");
    let plan_c = plan(&subject_a, "campaign-b");
    assert_ne!(plan_a.digest(), plan_b.digest());
    assert_ne!(plan_a.digest(), plan_c.digest());
}

#[test]
fn ordering_receipt_must_bind_exact_registration_statement() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let statement = RegistrationStatementV1::new(&plan, id("registrar-a"), None).unwrap();
    let wrong = ordering("transparency-log-a", 1, 1, digest('a'), 'e');
    let error = PreregistrationReceiptV1::new(statement, wrong, None).unwrap_err();
    assert_eq!(error, CampaignError::OrderingStatementMismatch);
}

#[test]
fn successor_registration_requires_strict_same_lineage_ordering() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let second_plan = revised_plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 5);
    let statement =
        RegistrationStatementV1::new(&second_plan, id("registrar-a"), Some(&root)).unwrap();

    let stale = ordering("transparency-log-a", 1, 5, statement.digest(), 'f');
    assert_eq!(
        PreregistrationReceiptV1::new(statement.clone(), stale, Some(&root)).unwrap_err(),
        CampaignError::NonIncreasingOrderingSequence
    );

    let foreign = ordering("transparency-log-b", 1, 6, statement.digest(), 'f');
    assert_eq!(
        PreregistrationReceiptV1::new(statement, foreign, Some(&root)).unwrap_err(),
        CampaignError::IncomparableOrderingLineage
    );
}

#[test]
fn identical_duplicate_registration_is_idempotent() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 1);
    let current = resolve_current_registration(&[root.clone(), root.clone()], &[]).unwrap();
    assert_eq!(current.digest(), root.digest());
}

#[test]
fn conflicting_successors_form_a_fork() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 1);
    let successor_a = successor_registration(&revised_plan(&subject, "campaign-a"), &root, 2);

    let claim = claim(&subject);
    let alternate = CampaignPlanV1::new(
        id("alternate-plan"),
        id("campaign-a"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![EvidenceKind::Observation],
        vec![id("alternate-control")],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    let successor_b = successor_registration(&alternate, &root, 3);

    assert_eq!(
        resolve_current_registration(&[root, successor_a, successor_b], &[]).unwrap_err(),
        CampaignError::RegistrationFork
    );
}

#[test]
fn withdrawing_current_registration_leaves_no_current_plan() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 1);
    let statement = WithdrawalStatementV1::new(&root, id("operator-withdrawal"));
    let order = ordering("transparency-log-a", 1, 2, statement.digest(), 'd');
    let withdrawal = RegistrationWithdrawalV1::new(statement, order, &root).unwrap();
    assert_eq!(
        resolve_current_registration(&[root], &[withdrawal]).unwrap_err(),
        CampaignError::NoCurrentRegistration
    );
}

#[test]
fn withdrawn_registration_cannot_have_active_successor() {
    let subject = subject('a');
    let root_plan = plan(&subject, "campaign-a");
    let root = root_registration(&root_plan, 1);
    let successor = successor_registration(&revised_plan(&subject, "campaign-a"), &root, 3);
    let statement = WithdrawalStatementV1::new(&root, id("withdraw-before-successor"));
    let order = ordering("transparency-log-a", 1, 2, statement.digest(), 'd');
    let withdrawal = RegistrationWithdrawalV1::new(statement, order, &root).unwrap();
    assert_eq!(
        resolve_current_registration(&[root, successor], &[withdrawal]).unwrap_err(),
        CampaignError::SuccessorOfWithdrawnRegistration
    );
}

#[test]
fn evidence_before_or_at_registration_is_posthoc_not_preregistered() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let statement = evidence_production_statement_digest(&current, &evidence);

    let before = ordering("transparency-log-a", 1, 4, statement.clone(), 'b');
    assert_eq!(
        classify_evidence_timing(&current, &evidence, &before).unwrap(),
        EvidenceTimingClass::ProducedBeforeOrAtRegistration
    );

    let equal = ordering("transparency-log-a", 1, 5, statement, 'c');
    assert_eq!(
        classify_evidence_timing(&current, &evidence, &equal).unwrap(),
        EvidenceTimingClass::ProducedBeforeOrAtRegistration
    );
}

#[test]
fn evidence_from_another_ordering_lineage_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let statement = evidence_production_statement_digest(&current, &evidence);
    let foreign = ordering("transparency-log-b", 1, 6, statement, 'b');
    assert_eq!(
        classify_evidence_timing(&current, &evidence, &foreign).unwrap(),
        EvidenceTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn evidence_under_different_validation_profile_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let statement = evidence_production_statement_digest(&current, &evidence);
    let foreign_profile = ordering_with_profile(
        "transparency-log-a",
        "different-validation-profile-v1",
        1,
        6,
        statement,
        'b',
    );
    assert_eq!(
        classify_evidence_timing(&current, &evidence, &foreign_profile).unwrap(),
        EvidenceTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn preregistered_admission_advances_append_only_evidence_root() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let initial_root = ledger.evidence_root().clone();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");

    let production_statement = evidence_production_statement_digest(&current, &evidence);
    let production = ordering("transparency-log-a", 1, 6, production_statement, 'b');
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &production)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, 'c');

    let admitted = ledger
        .admit_preregistered(&plan, &current, &evidence, &production, &admission)
        .unwrap();
    assert_eq!(ledger.admitted_count(), 1);
    assert_ne!(ledger.evidence_root(), &initial_root);
    assert_eq!(admitted.evidence_root(), ledger.evidence_root());
}

#[test]
fn concurrent_production_can_be_admitted_after_prior_admission() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);

    let first = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let first_production = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_production_statement_digest(&current, &first),
        'b',
    );
    let second = evidence(&subject, EvidenceKind::Observation, "evidence-b");
    let second_production = ordering(
        "transparency-log-a",
        1,
        7,
        evidence_production_statement_digest(&current, &second),
        'd',
    );

    let first_admission_statement = ledger
        .admission_statement_digest(&current, &first, &first_production)
        .unwrap();
    let first_admission = ordering(
        "transparency-log-a",
        1,
        8,
        first_admission_statement,
        'c',
    );
    ledger
        .admit_preregistered(
            &plan,
            &current,
            &first,
            &first_production,
            &first_admission,
        )
        .unwrap();

    let second_admission_statement = ledger
        .admission_statement_digest(&current, &second, &second_production)
        .unwrap();
    let second_admission = ordering(
        "transparency-log-a",
        1,
        9,
        second_admission_statement,
        'e',
    );
    ledger
        .admit_preregistered(
            &plan,
            &current,
            &second,
            &second_production,
            &second_admission,
        )
        .unwrap();
    assert_eq!(ledger.admitted_count(), 2);
}

#[test]
fn admission_must_advance_previous_admission_ordering() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);

    let first = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let first_production = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_production_statement_digest(&current, &first),
        'b',
    );
    let first_admission_statement = ledger
        .admission_statement_digest(&current, &first, &first_production)
        .unwrap();
    let first_admission = ordering(
        "transparency-log-a",
        1,
        8,
        first_admission_statement,
        'c',
    );
    ledger
        .admit_preregistered(
            &plan,
            &current,
            &first,
            &first_production,
            &first_admission,
        )
        .unwrap();

    let second = evidence(&subject, EvidenceKind::Observation, "evidence-b");
    let second_production = ordering(
        "transparency-log-a",
        1,
        7,
        evidence_production_statement_digest(&current, &second),
        'd',
    );
    let second_admission_statement = ledger
        .admission_statement_digest(&current, &second, &second_production)
        .unwrap();
    let stale_admission = ordering(
        "transparency-log-a",
        1,
        8,
        second_admission_statement,
        'e',
    );
    assert_eq!(
        ledger
            .admit_preregistered(
                &plan,
                &current,
                &second,
                &second_production,
                &stale_admission,
            )
            .unwrap_err(),
        CampaignError::AdmissionNotAfterLedgerHead
    );
}

#[test]
fn duplicate_evidence_cannot_be_admitted_twice() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");

    let production_statement = evidence_production_statement_digest(&current, &evidence);
    let production = ordering("transparency-log-a", 1, 6, production_statement, 'b');
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &production)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, 'c');
    ledger
        .admit_preregistered(&plan, &current, &evidence, &production, &admission)
        .unwrap();

    let second_production_statement = evidence_production_statement_digest(&current, &evidence);
    let second_production = ordering(
        "transparency-log-a",
        1,
        8,
        second_production_statement,
        'd',
    );
    let second_admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &second_production)
        .unwrap();
    let second_admission = ordering(
        "transparency-log-a",
        1,
        9,
        second_admission_statement,
        'e',
    );
    assert_eq!(
        ledger
            .admit_preregistered(
                &plan,
                &current,
                &evidence,
                &second_production,
                &second_admission,
            )
            .unwrap_err(),
        CampaignError::DuplicateEvidenceId("evidence-a".into())
    );
}

#[test]
fn unregistered_evidence_kind_cannot_enter_preregistered_ledger() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let evidence = evidence(&subject, EvidenceKind::RuntimeReceipt, "runtime-evidence");
    let production_statement = evidence_production_statement_digest(&current, &evidence);
    let production = ordering("transparency-log-a", 1, 6, production_statement, 'b');
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &production)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, 'c');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &current, &evidence, &production, &admission)
            .unwrap_err(),
        CampaignError::UnregisteredEvidenceKind
    );
}

#[test]
fn successor_plan_starts_new_empty_evidence_lineage() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 1);
    let first_current = resolve_current_registration(std::slice::from_ref(&root), &[]).unwrap();
    let first_ledger = CampaignEvidenceLedgerV1::new(&first_current);

    let second_plan = revised_plan(&subject, "campaign-a");
    let successor = successor_registration(&second_plan, &root, 2);
    let second_current = resolve_current_registration(&[root, successor], &[]).unwrap();
    let second_ledger = CampaignEvidenceLedgerV1::new(&second_current);

    assert_eq!(first_ledger.admitted_count(), 0);
    assert_eq!(second_ledger.admitted_count(), 0);
    assert_ne!(first_ledger.evidence_root(), second_ledger.evidence_root());
}

#[test]
fn old_registration_ledger_cannot_admit_under_new_current_registration() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 1);
    let first_current = resolve_current_registration(std::slice::from_ref(&root), &[]).unwrap();
    let mut old_ledger = CampaignEvidenceLedgerV1::new(&first_current);

    let second_plan = revised_plan(&subject, "campaign-a");
    let successor = successor_registration(&second_plan, &root, 2);
    let second_current = resolve_current_registration(&[root, successor], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "new-evidence");
    let production_statement = evidence_production_statement_digest(&second_current, &evidence);
    let production = ordering("transparency-log-a", 1, 3, production_statement, 'b');
    let fake_admission = ordering("transparency-log-a", 1, 4, digest('f'), 'c');

    assert_eq!(
        old_ledger
            .admit_preregistered(
                &second_plan,
                &second_current,
                &evidence,
                &production,
                &fake_admission,
            )
            .unwrap_err(),
        CampaignError::LedgerRegistrationMismatch
    );
}

#[test]
fn admission_must_be_ordered_after_production() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_current_registration(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a");
    let production_statement = evidence_production_statement_digest(&current, &evidence);
    let production = ordering("transparency-log-a", 1, 7, production_statement, 'b');
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &production)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, 'c');

    assert_eq!(
        ledger
            .admit_preregistered(&plan, &current, &evidence, &production, &admission)
            .unwrap_err(),
        CampaignError::AdmissionNotAfterProduction
    );
}
