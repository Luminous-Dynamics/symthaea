// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_campaign::{
    CampaignError, CampaignEvidenceLedgerV1, CampaignPlanV1, EvidenceCommitmentTimingClass,
    EvidenceRequirementV1, OrderingReceiptV1, PreregistrationReceiptV1, RegistrationStatementV1,
    RegistrationWithdrawalV1, ReproductionRequirementV1, SupportCriterionV1, WithdrawalStatementV1,
    classify_evidence_commitment_timing, evidence_commitment_statement_digest,
    resolve_terminal_registration_in_view,
};
use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance, StableId, SupportTier,
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

fn semantic_with_schema(
    name: &str,
    schema_id: &str,
    specification_byte: char,
    definition_byte: char,
) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(name),
        DefinitionSchemaV1::new(id(schema_id), digest(specification_byte)),
        digest(definition_byte),
    )
}

fn semantic(name: &str, byte: char) -> SemanticCommitmentV1 {
    semantic_with_schema(
        name,
        "symthaea.assurance.test-semantic-schema.v1",
        'e',
        byte,
    )
}

fn req(kind: EvidenceKind) -> EvidenceRequirementV1 {
    EvidenceRequirementV1::builtin(kind).unwrap()
}

fn subject(model_byte: char) -> AiSubjectManifest {
    let profile =
        SurfaceProfile::new(id("campaign-test-profile"), vec![AiSurfaceKind::Model]).unwrap();
    AiSubjectManifest::new(
        id("campaign-test-agent"),
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
        id("authority-boundary-claim"),
        subject.core_subject_id().unwrap(),
        "high-impact action requires registered approval authority",
        id("agent-authority"),
    )
    .unwrap()
}

fn plan_with_control_semantic(
    subject: &AiSubjectManifest,
    campaign_nonce: &str,
    control: SemanticCommitmentV1,
) -> CampaignPlanV1 {
    let claim = claim(subject);
    CampaignPlanV1::new(
        id("authority-plan-v1"),
        id(campaign_nonce),
        &claim,
        subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![
            req(EvidenceKind::Observation),
            req(EvidenceKind::ControlledIntervention),
        ],
        vec![
            SupportCriterionV1::new(
                SupportTier::Observed,
                semantic("observable-authority-decision", '1'),
            ),
            SupportCriterionV1::new(
                SupportTier::CausallySupported,
                semantic("authority-boundary-causal-effect", '2'),
            ),
        ],
        vec![
            semantic("baseline", '3'),
            control,
            semantic("valid-authority", '5'),
        ],
        vec![semantic("allowed-path-broken", '6')],
        vec![semantic("unauthorized-execution-observed", '7')],
        vec![semantic("instrumentation-incomplete", '8')],
        vec![
            semantic("subject-drift", '9'),
            semantic("policy-drift", 'a'),
        ],
    )
    .unwrap()
}

fn plan_with_control_definition(
    subject: &AiSubjectManifest,
    campaign_nonce: &str,
    control_definition: char,
) -> CampaignPlanV1 {
    plan_with_control_semantic(
        subject,
        campaign_nonce,
        semantic("denied-path", control_definition),
    )
}

fn plan(subject: &AiSubjectManifest, campaign_nonce: &str) -> CampaignPlanV1 {
    plan_with_control_definition(subject, campaign_nonce, '4')
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
        vec![
            req(EvidenceKind::ControlledIntervention),
            req(EvidenceKind::Observation),
        ],
        vec![
            SupportCriterionV1::new(
                SupportTier::CausallySupported,
                semantic("authority-boundary-causal-effect", '2'),
            ),
            SupportCriterionV1::new(
                SupportTier::Observed,
                semantic("observable-authority-decision", '1'),
            ),
        ],
        vec![
            semantic("matched-sham", 'b'),
            semantic("baseline", '3'),
            semantic("denied-path", '4'),
        ],
        vec![semantic("allowed-path-broken", '6')],
        vec![semantic("unauthorized-execution-observed", '7')],
        vec![semantic("instrumentation-incomplete", '8')],
        vec![
            semantic("policy-drift", 'a'),
            semantic("subject-drift", '9'),
        ],
    )
    .unwrap()
}

fn ordering_with_profile(
    source: &str,
    profile_name: &str,
    profile_definition: char,
    epoch: u64,
    sequence: u64,
    statement: DigestSha256,
    receipt_byte: char,
) -> OrderingReceiptV1 {
    OrderingReceiptV1::new(
        id(source),
        semantic(profile_name, profile_definition),
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
        'f',
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
    let receipt = ordering("transparency-log-a", 1, sequence, statement.digest(), 'd');
    PreregistrationReceiptV1::new(statement, receipt, Some(predecessor)).unwrap()
}

fn evidence(
    subject: &AiSubjectManifest,
    kind: EvidenceKind,
    evidence_id: &str,
    artifact_byte: char,
) -> EvidenceArtifact {
    let claim = claim(subject);
    EvidenceArtifact::new(
        id(evidence_id),
        subject.core_subject_id().unwrap(),
        claim.digest(),
        kind,
        digest(artifact_byte),
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
        vec![
            req(EvidenceKind::ControlledIntervention),
            req(EvidenceKind::Observation),
        ],
        vec![
            SupportCriterionV1::new(SupportTier::Observed, semantic("criterion-b", '2')),
            SupportCriterionV1::new(SupportTier::Structural, semantic("criterion-a", '1')),
        ],
        vec![semantic("control-b", '4'), semantic("control-a", '3')],
        vec![semantic("failure-b", '6'), semantic("failure-a", '5')],
        vec![semantic("contradiction", '7')],
        vec![semantic("inconclusive", '8')],
        vec![semantic("invalidate-b", 'a'), semantic("invalidate-a", '9')],
    )
    .unwrap();
    let right = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![
            req(EvidenceKind::Observation),
            req(EvidenceKind::ControlledIntervention),
        ],
        vec![
            SupportCriterionV1::new(SupportTier::Structural, semantic("criterion-a", '1')),
            SupportCriterionV1::new(SupportTier::Observed, semantic("criterion-b", '2')),
        ],
        vec![semantic("control-a", '3'), semantic("control-b", '4')],
        vec![semantic("failure-a", '5'), semantic("failure-b", '6')],
        vec![semantic("contradiction", '7')],
        vec![semantic("inconclusive", '8')],
        vec![semantic("invalidate-a", '9'), semantic("invalidate-b", 'a')],
    )
    .unwrap();
    assert_eq!(left.digest(), right.digest());
    assert_eq!(left.core_plan().digest(), right.core_plan().digest());
}

#[test]
fn semantic_definition_drift_changes_richer_plan_not_generic_core_plan() {
    let subject = subject('a');
    let original = plan_with_control_definition(&subject, "campaign-a", '4');
    let redefined = plan_with_control_definition(&subject, "campaign-a", 'b');
    assert_ne!(original.digest(), redefined.digest());
    assert_eq!(
        original.core_plan().digest(),
        redefined.core_plan().digest()
    );
}

#[test]
fn duplicate_semantic_ids_fail_closed_even_with_different_definitions() {
    let subject = subject('a');
    let claim = claim(&subject);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![req(EvidenceKind::Observation)],
        vec![],
        vec![semantic("same-control", '1'), semantic("same-control", '2')],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CampaignError::DuplicateSemanticId {
            set: "controls",
            ..
        }
    ));
}

#[test]
fn duplicate_builtin_evidence_kinds_fail_closed() {
    let subject = subject('a');
    let claim = claim(&subject);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![
            req(EvidenceKind::Observation),
            req(EvidenceKind::Observation),
        ],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(error, CampaignError::DuplicateEvidenceKind(_)));
}

#[test]
fn custom_evidence_requires_definition_commitment() {
    assert_eq!(
        EvidenceRequirementV1::builtin(EvidenceKind::Custom(id("custom-evidence"))).unwrap_err(),
        CampaignError::CustomEvidenceRequiresDefinition
    );
}

#[test]
fn custom_evidence_definition_changes_plan_identity() {
    let subject = subject('a');
    let claim = claim(&subject);
    let make = |definition| {
        CampaignPlanV1::new(
            id("plan"),
            id("campaign"),
            &claim,
            &subject,
            SupportTier::Observed,
            ReproductionRequirementV1::NotRequired,
            vec![EvidenceRequirementV1::custom(semantic(
                "custom-evidence",
                definition,
            ))],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap()
    };
    assert_ne!(make('1').digest(), make('2').digest());
}

#[test]
fn support_criterion_above_ceiling_fails_closed() {
    let subject = subject('a');
    let claim = claim(&subject);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![req(EvidenceKind::Observation)],
        vec![SupportCriterionV1::new(
            SupportTier::CausallySupported,
            semantic("too-strong", '1'),
        )],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert_eq!(error, CampaignError::SupportCriterionAboveCeiling);
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
        vec![req(EvidenceKind::Observation)],
        vec![],
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
    assert_ne!(
        plan(&subject_a, "campaign-a").digest(),
        plan(&subject_b, "campaign-a").digest()
    );
    assert_ne!(
        plan(&subject_a, "campaign-a").digest(),
        plan(&subject_a, "campaign-b").digest()
    );
}

#[test]
fn ordering_receipt_must_bind_exact_registration_statement() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let statement = RegistrationStatementV1::new(&plan, id("registrar-a"), None).unwrap();
    let wrong = ordering("transparency-log-a", 1, 1, digest('0'), 'e');
    assert_eq!(
        PreregistrationReceiptV1::new(statement, wrong, None).unwrap_err(),
        CampaignError::OrderingStatementMismatch
    );
}

#[test]
fn successor_registration_requires_strict_same_lineage_ordering() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let second_plan = revised_plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 5);
    let statement =
        RegistrationStatementV1::new(&second_plan, id("registrar-a"), Some(&root)).unwrap();
    let stale = ordering("transparency-log-a", 1, 5, statement.digest(), 'd');
    assert_eq!(
        PreregistrationReceiptV1::new(statement.clone(), stale, Some(&root)).unwrap_err(),
        CampaignError::NonIncreasingOrderingSequence
    );
    let foreign = ordering("transparency-log-b", 1, 6, statement.digest(), 'd');
    assert_eq!(
        PreregistrationReceiptV1::new(statement, foreign, Some(&root)).unwrap_err(),
        CampaignError::IncomparableOrderingLineage
    );
}

#[test]
fn validation_profile_definition_is_part_of_ordering_lineage() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let changed_definition = ordering_with_profile(
        "transparency-log-a",
        "monotonic-ordering-profile-v1",
        '0',
        1,
        6,
        evidence_commitment_statement_digest(&current, &evidence),
        'b',
    );
    assert_eq!(
        classify_evidence_commitment_timing(&current, &evidence, &changed_definition).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn identical_duplicate_registration_is_idempotent() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 1);
    let current =
        resolve_terminal_registration_in_view(&[root.clone(), root.clone()], &[]).unwrap();
    assert_eq!(current.digest(), root.digest());
}

#[test]
fn conflicting_successors_form_a_fork() {
    let subject = subject('a');
    let root_plan = plan(&subject, "campaign-a");
    let root = root_registration(&root_plan, 1);
    let successor_a = successor_registration(&revised_plan(&subject, "campaign-a"), &root, 2);
    let alternate = plan_with_control_definition(&subject, "campaign-a", 'c');
    let successor_b = successor_registration(&alternate, &root, 3);
    assert_eq!(
        resolve_terminal_registration_in_view(&[root, successor_a, successor_b], &[]).unwrap_err(),
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
        resolve_terminal_registration_in_view(&[root], &[withdrawal]).unwrap_err(),
        CampaignError::NoTerminalRegistrationInView
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
        resolve_terminal_registration_in_view(&[root, successor], &[withdrawal]).unwrap_err(),
        CampaignError::SuccessorOfWithdrawnRegistration
    );
}

#[test]
fn evidence_before_or_at_registration_is_posthoc_not_preregistered() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let statement = evidence_commitment_statement_digest(&current, &evidence);
    let before = ordering("transparency-log-a", 1, 4, statement.clone(), 'b');
    assert_eq!(
        classify_evidence_commitment_timing(&current, &evidence, &before).unwrap(),
        EvidenceCommitmentTimingClass::CommittedBeforeOrAtTerminalRegistrationInView
    );
    let equal = ordering("transparency-log-a", 1, 5, statement, 'c');
    assert_eq!(
        classify_evidence_commitment_timing(&current, &evidence, &equal).unwrap(),
        EvidenceCommitmentTimingClass::CommittedBeforeOrAtTerminalRegistrationInView
    );
}

#[test]
fn evidence_from_another_ordering_lineage_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let foreign = ordering(
        "transparency-log-b",
        1,
        6,
        evidence_commitment_statement_digest(&current, &evidence),
        'b',
    );
    assert_eq!(
        classify_evidence_commitment_timing(&current, &evidence, &foreign).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn preregistered_admission_advances_append_only_evidence_root() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let initial_root = ledger.evidence_root().clone();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&current, &evidence),
        'b',
    );
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, 'c');
    let admitted = ledger
        .admit_preregistered(&plan, &current, &evidence, &commitment, &admission)
        .unwrap();
    assert_eq!(ledger.admitted_count(), 1);
    assert_ne!(ledger.evidence_root(), &initial_root);
    assert_eq!(admitted.evidence_root(), ledger.evidence_root());
}

#[test]
fn concurrent_production_can_be_admitted_serially() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let first = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let second = evidence(&subject, EvidenceKind::Observation, "evidence-b", 'd');
    let first_production = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&current, &first),
        '1',
    );
    let second_production = ordering(
        "transparency-log-a",
        1,
        7,
        evidence_commitment_statement_digest(&current, &second),
        '2',
    );
    let first_admission_statement = ledger
        .admission_statement_digest(&current, &first, &first_production)
        .unwrap();
    let first_admission = ordering("transparency-log-a", 1, 8, first_admission_statement, '3');
    ledger
        .admit_preregistered(&plan, &current, &first, &first_production, &first_admission)
        .unwrap();
    let second_admission_statement = ledger
        .admission_statement_digest(&current, &second, &second_production)
        .unwrap();
    let second_admission = ordering("transparency-log-a", 1, 9, second_admission_statement, '4');
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
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let first = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let first_production = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&current, &first),
        '1',
    );
    let first_admission_statement = ledger
        .admission_statement_digest(&current, &first, &first_production)
        .unwrap();
    let first_admission = ordering("transparency-log-a", 1, 8, first_admission_statement, '2');
    ledger
        .admit_preregistered(&plan, &current, &first, &first_production, &first_admission)
        .unwrap();
    let second = evidence(&subject, EvidenceKind::Observation, "evidence-b", 'd');
    let second_production = ordering(
        "transparency-log-a",
        1,
        7,
        evidence_commitment_statement_digest(&current, &second),
        '3',
    );
    let second_admission_statement = ledger
        .admission_statement_digest(&current, &second, &second_production)
        .unwrap();
    let stale_admission = ordering("transparency-log-a", 1, 8, second_admission_statement, '4');
    assert_eq!(
        ledger
            .admit_preregistered(
                &plan,
                &current,
                &second,
                &second_production,
                &stale_admission
            )
            .unwrap_err(),
        CampaignError::AdmissionNotAfterLedgerHead
    );
}

#[test]
fn admission_must_be_ordered_after_its_production() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let commitment = ordering(
        "transparency-log-a",
        1,
        7,
        evidence_commitment_statement_digest(&current, &evidence),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &current, &evidence, &commitment, &admission)
            .unwrap_err(),
        CampaignError::AdmissionNotAfterCommitment
    );
}

#[test]
fn duplicate_evidence_id_cannot_be_admitted_twice() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let first = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&current, &first),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&current, &first, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    ledger
        .admit_preregistered(&plan, &current, &first, &commitment, &admission)
        .unwrap();
    let changed_content = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'd');
    let production2 = ordering(
        "transparency-log-a",
        1,
        8,
        evidence_commitment_statement_digest(&current, &changed_content),
        '3',
    );
    let admission_statement2 = ledger
        .admission_statement_digest(&current, &changed_content, &production2)
        .unwrap();
    let admission2 = ordering("transparency-log-a", 1, 9, admission_statement2, '4');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &current, &changed_content, &production2, &admission2)
            .unwrap_err(),
        CampaignError::DuplicateEvidenceId("evidence-a".into())
    );
}

#[test]
fn unregistered_evidence_kind_cannot_enter_preregistered_ledger() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let current = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    let evidence = evidence(
        &subject,
        EvidenceKind::RuntimeReceipt,
        "runtime-evidence",
        'c',
    );
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&current, &evidence),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&current, &evidence, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &current, &evidence, &commitment, &admission)
            .unwrap_err(),
        CampaignError::UnregisteredEvidenceKind
    );
}

#[test]
fn successor_plan_starts_new_empty_evidence_lineage() {
    let subject = subject('a');
    let first_plan = plan(&subject, "campaign-a");
    let root = root_registration(&first_plan, 1);
    let first_current =
        resolve_terminal_registration_in_view(std::slice::from_ref(&root), &[]).unwrap();
    let first_ledger = CampaignEvidenceLedgerV1::new(&first_current);
    let second_plan = revised_plan(&subject, "campaign-a");
    let successor = successor_registration(&second_plan, &root, 2);
    let second_current = resolve_terminal_registration_in_view(&[root, successor], &[]).unwrap();
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
    let first_current =
        resolve_terminal_registration_in_view(std::slice::from_ref(&root), &[]).unwrap();
    let mut old_ledger = CampaignEvidenceLedgerV1::new(&first_current);
    let second_plan = revised_plan(&subject, "campaign-a");
    let successor = successor_registration(&second_plan, &root, 2);
    let second_current = resolve_terminal_registration_in_view(&[root, successor], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "new-evidence", 'c');
    let commitment = ordering(
        "transparency-log-a",
        1,
        3,
        evidence_commitment_statement_digest(&second_current, &evidence),
        '1',
    );
    let fake_admission = ordering("transparency-log-a", 1, 4, digest('0'), '2');
    assert_eq!(
        old_ledger
            .admit_preregistered(
                &second_plan,
                &second_current,
                &evidence,
                &commitment,
                &fake_admission
            )
            .unwrap_err(),
        CampaignError::LedgerRegistrationMismatch
    );
}

#[test]
fn semantic_schema_id_drift_changes_richer_plan_identity() {
    let subject = subject('a');
    let original = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'e', '4'),
    );
    let changed = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-b", 'e', '4'),
    );
    assert_ne!(original.digest(), changed.digest());
    assert_eq!(original.core_plan().digest(), changed.core_plan().digest());
}

#[test]
fn semantic_schema_specification_drift_changes_richer_plan_identity() {
    let subject = subject('a');
    let original = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'e', '4'),
    );
    let changed = plan_with_control_semantic(
        &subject,
        "campaign-a",
        semantic_with_schema("denied-path", "schema-a", 'd', '4'),
    );
    assert_ne!(original.digest(), changed.digest());
    assert_eq!(original.core_plan().digest(), changed.core_plan().digest());
}

#[test]
fn duplicate_semantic_ids_fail_closed_across_schema_drift() {
    let subject = subject('a');
    let claim = claim(&subject);
    let error = CampaignPlanV1::new(
        id("plan"),
        id("campaign"),
        &claim,
        &subject,
        SupportTier::Observed,
        ReproductionRequirementV1::NotRequired,
        vec![req(EvidenceKind::Observation)],
        vec![],
        vec![
            semantic_with_schema("same-control", "schema-a", 'e', '1'),
            semantic_with_schema("same-control", "schema-b", 'e', '1'),
        ],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CampaignError::DuplicateSemanticId {
            set: "controls",
            ..
        }
    ));
}

#[test]
fn validation_profile_schema_id_drift_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let changed = OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic_with_schema(
            "monotonic-ordering-profile-v1",
            "different-schema",
            'e',
            'f',
        ),
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &evidence),
        digest('b'),
    )
    .unwrap();
    assert_eq!(
        classify_evidence_commitment_timing(&terminal, &evidence, &changed).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn validation_profile_schema_specification_drift_is_incomparable() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let evidence = evidence(&subject, EvidenceKind::Observation, "evidence-a", 'c');
    let changed = OrderingReceiptV1::new(
        id("transparency-log-a"),
        semantic_with_schema(
            "monotonic-ordering-profile-v1",
            "symthaea.assurance.test-semantic-schema.v1",
            '0',
            'f',
        ),
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &evidence),
        digest('b'),
    )
    .unwrap();
    assert_eq!(
        classify_evidence_commitment_timing(&terminal, &evidence, &changed).unwrap(),
        EvidenceCommitmentTimingClass::IncomparableOrderingLineage
    );
}

#[test]
fn foreign_subject_evidence_fails_before_ledger_mutation() {
    let subject = subject('a');
    let foreign_subject = subject('b');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&terminal);
    let initial_root = ledger.evidence_root().clone();
    let foreign = evidence(
        &foreign_subject,
        EvidenceKind::Observation,
        "context-id",
        'c',
    );
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &foreign),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&terminal, &foreign, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &terminal, &foreign, &commitment, &admission)
            .unwrap_err(),
        CampaignError::EvidenceSubjectMismatch
    );
    assert_eq!(ledger.admitted_count(), 0);
    assert_eq!(ledger.evidence_root(), &initial_root);

    let valid = evidence(&subject, EvidenceKind::Observation, "context-id", 'd');
    let valid_commitment = ordering(
        "transparency-log-a",
        1,
        8,
        evidence_commitment_statement_digest(&terminal, &valid),
        '3',
    );
    let valid_admission_statement = ledger
        .admission_statement_digest(&terminal, &valid, &valid_commitment)
        .unwrap();
    let valid_admission = ordering("transparency-log-a", 1, 9, valid_admission_statement, '4');
    ledger
        .admit_preregistered(
            &plan,
            &terminal,
            &valid,
            &valid_commitment,
            &valid_admission,
        )
        .unwrap();
    assert_eq!(ledger.admitted_count(), 1);
}

#[test]
fn foreign_claim_evidence_fails_before_ledger_mutation() {
    let subject = subject('a');
    let plan = plan(&subject, "campaign-a");
    let root = root_registration(&plan, 5);
    let terminal = resolve_terminal_registration_in_view(&[root], &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&terminal);
    let initial_root = ledger.evidence_root().clone();
    let foreign_claim = Claim::new(
        id("foreign-claim"),
        subject.core_subject_id().unwrap(),
        "foreign proposition",
        id("agent-authority"),
    )
    .unwrap();
    let foreign = EvidenceArtifact::new(
        id("foreign-claim-evidence"),
        subject.core_subject_id().unwrap(),
        foreign_claim.digest(),
        EvidenceKind::Observation,
        digest('c'),
        EvidenceProvenance::new(id("producer"), id("executor"), Some(id("verifier")), None),
    );
    let commitment = ordering(
        "transparency-log-a",
        1,
        6,
        evidence_commitment_statement_digest(&terminal, &foreign),
        '1',
    );
    let admission_statement = ledger
        .admission_statement_digest(&terminal, &foreign, &commitment)
        .unwrap();
    let admission = ordering("transparency-log-a", 1, 7, admission_statement, '2');
    assert_eq!(
        ledger
            .admit_preregistered(&plan, &terminal, &foreign, &commitment, &admission)
            .unwrap_err(),
        CampaignError::EvidenceClaimMismatch
    );
    assert_eq!(ledger.admitted_count(), 0);
    assert_eq!(ledger.evidence_root(), &initial_root);
}
