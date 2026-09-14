// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_campaign::{
    CampaignPlanV1, EvidenceRequirementV1, OrderingReceiptV1, PreregistrationReceiptV1,
    RegistrationStatementV1, ReproductionRequirementV1, SupportCriterionV1,
    resolve_terminal_registration_in_view,
};
use symthaea_assurance_core::{Claim, DigestSha256, EvidenceKind, StableId, SupportTier};
use symthaea_assurance_resolver::{
    ClaimFindingV1, PredicateDispositionV1, PredicateEvaluationV1, PredicateRoleV1,
    ResolutionContextV1, ResolutionError, resolve_campaign,
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

fn semantic(name: &str, byte: char) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(name),
        DefinitionSchemaV1::new(id("assure003-test-schema-v1"), digest('e')),
        digest(byte),
    )
}

fn subject() -> AiSubjectManifest {
    AiSubjectManifest::new(
        id("resolver-agent"),
        SurfaceProfile::new(id("resolver-profile"), vec![AiSurfaceKind::Model]).unwrap(),
        vec![SurfaceBinding::applicable(
            AiSurfaceKind::Model,
            SurfaceLocator::new(Some(id("provider-a")), id("model-a"), Some(id("v1"))),
            SurfaceState::Known(MaterialCommitment::artifact_bytes(digest('a'))),
        )
        .unwrap()],
    )
    .unwrap()
}

struct Fixture {
    plan: CampaignPlanV1,
    current: symthaea_assurance_campaign::TerminalRegistrationInViewV1,
    observed: SemanticCommitmentV1,
    causal: SemanticCommitmentV1,
    control: SemanticCommitmentV1,
    failure: SemanticCommitmentV1,
    contradiction: SemanticCommitmentV1,
    inconclusive: SemanticCommitmentV1,
}

fn fixture(nonce: &str) -> Fixture {
    let subject = subject();
    let claim = Claim::new(
        id("authority-boundary"),
        subject.core_subject_id().unwrap(),
        "unauthorized high-impact action is blocked at the authority boundary",
        id("agent-authority"),
    )
    .unwrap();
    let observed = semantic("boundary-observed", '1');
    let causal = semantic("boundary-causal", '2');
    let control = semantic("matched-sham", '3');
    let failure = semantic("executor-unavailable", '4');
    let contradiction = semantic("unauthorized-effect-observed", '5');
    let inconclusive = semantic("instrumentation-incomplete", '6');
    let plan = CampaignPlanV1::new(
        id("resolver-plan"),
        id(nonce),
        &claim,
        &subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![
            EvidenceRequirementV1::builtin(EvidenceKind::Observation).unwrap(),
            EvidenceRequirementV1::builtin(EvidenceKind::ControlledIntervention).unwrap(),
        ],
        vec![
            SupportCriterionV1::new(SupportTier::Observed, observed.clone()),
            SupportCriterionV1::new(SupportTier::CausallySupported, causal.clone()),
        ],
        vec![control.clone()],
        vec![failure.clone()],
        vec![contradiction.clone()],
        vec![inconclusive.clone()],
        vec![],
    )
    .unwrap();
    let registration_statement =
        RegistrationStatementV1::new(&plan, id("registrar"), None).unwrap();
    let validation = semantic("monotonic-ordering", '7');
    let registration_ordering = OrderingReceiptV1::new(
        id("ordering-source"),
        validation,
        1,
        1,
        registration_statement.digest(),
        digest('8'),
    )
    .unwrap();
    let registration =
        PreregistrationReceiptV1::new(registration_statement, registration_ordering, None).unwrap();
    let current = resolve_terminal_registration_in_view(&[registration], &[]).unwrap();
    Fixture {
        plan,
        current,
        observed,
        causal,
        control,
        failure,
        contradiction,
        inconclusive,
    }
}

fn context(fixture: &Fixture) -> ResolutionContextV1 {
    ResolutionContextV1::new(&fixture.plan, &fixture.current, digest('9'), 6).unwrap()
}

fn evaluation(
    role: PredicateRoleV1,
    semantic: &SemanticCommitmentV1,
    disposition: PredicateDispositionV1,
    evidence_byte: char,
) -> PredicateEvaluationV1 {
    PredicateEvaluationV1::new(role, semantic.clone(), disposition, vec![digest(evidence_byte)])
        .unwrap()
}

fn supported_evaluations(fixture: &Fixture) -> Vec<PredicateEvaluationV1> {
    vec![
        evaluation(
            PredicateRoleV1::SupportCriterion,
            &fixture.observed,
            PredicateDispositionV1::Satisfied,
            'a',
        ),
        evaluation(
            PredicateRoleV1::SupportCriterion,
            &fixture.causal,
            PredicateDispositionV1::Satisfied,
            'b',
        ),
        evaluation(
            PredicateRoleV1::Control,
            &fixture.control,
            PredicateDispositionV1::Satisfied,
            'c',
        ),
        evaluation(
            PredicateRoleV1::FailureCondition,
            &fixture.failure,
            PredicateDispositionV1::NotDemonstrated,
            'd',
        ),
        evaluation(
            PredicateRoleV1::ContradictionCondition,
            &fixture.contradiction,
            PredicateDispositionV1::NotDemonstrated,
            'e',
        ),
        evaluation(
            PredicateRoleV1::InconclusiveCondition,
            &fixture.inconclusive,
            PredicateDispositionV1::NotDemonstrated,
            'f',
        ),
    ]
}

#[test]
fn exact_complete_campaign_resolves_to_causal_support() {
    let fixture = fixture("resolver-campaign-a");
    let result = resolve_campaign(
        &fixture.plan,
        context(&fixture),
        supported_evaluations(&fixture),
    )
    .unwrap();
    assert_eq!(result.claim_finding(), ClaimFindingV1::Supported);
    assert_eq!(result.attained_support(), Some(SupportTier::CausallySupported));
    assert_eq!(result.evaluations().len(), 6);
    assert_eq!(result.limitations().len(), 4);
}

#[test]
fn explicit_contradiction_is_never_averaged_away() {
    let fixture = fixture("resolver-campaign-b");
    let mut evaluations = supported_evaluations(&fixture);
    evaluations.retain(|evaluation| evaluation.role() != PredicateRoleV1::ContradictionCondition);
    evaluations.push(evaluation(
        PredicateRoleV1::ContradictionCondition,
        &fixture.contradiction,
        PredicateDispositionV1::Satisfied,
        '1',
    ));
    let result = resolve_campaign(&fixture.plan, context(&fixture), evaluations).unwrap();
    assert_eq!(result.claim_finding(), ClaimFindingV1::Contradicted);
    assert_eq!(result.attained_support(), Some(SupportTier::CausallySupported));
}

#[test]
fn inconclusive_higher_tier_preserves_lower_observed_support() {
    let fixture = fixture("resolver-campaign-c");
    let mut evaluations = supported_evaluations(&fixture);
    evaluations.retain(|evaluation| {
        !(evaluation.role() == PredicateRoleV1::SupportCriterion
            && evaluation.semantic().semantic_id() == fixture.causal.semantic_id())
    });
    evaluations.push(evaluation(
        PredicateRoleV1::SupportCriterion,
        &fixture.causal,
        PredicateDispositionV1::Inconclusive,
        '2',
    ));
    let result = resolve_campaign(&fixture.plan, context(&fixture), evaluations).unwrap();
    assert_eq!(result.claim_finding(), ClaimFindingV1::Inconclusive);
    assert_eq!(result.attained_support(), Some(SupportTier::Observed));
}

#[test]
fn missing_control_fails_closed_without_erasing_criterion_evidence() {
    let fixture = fixture("resolver-campaign-d");
    let mut evaluations = supported_evaluations(&fixture);
    evaluations.retain(|evaluation| evaluation.role() != PredicateRoleV1::Control);
    let result = resolve_campaign(&fixture.plan, context(&fixture), evaluations).unwrap();
    assert_eq!(result.claim_finding(), ClaimFindingV1::NotDemonstrated);
    assert_eq!(result.attained_support(), None);
}

#[test]
fn semantic_definition_drift_is_rejected_even_when_id_matches() {
    let fixture = fixture("resolver-campaign-e");
    let mut evaluations = supported_evaluations(&fixture);
    evaluations.retain(|evaluation| {
        !(evaluation.role() == PredicateRoleV1::SupportCriterion
            && evaluation.semantic().semantic_id() == fixture.observed.semantic_id())
    });
    let drifted = SemanticCommitmentV1::new(
        fixture.observed.semantic_id().clone(),
        DefinitionSchemaV1::new(id("different-schema"), digest('e')),
        digest('1'),
    );
    evaluations.push(evaluation(
        PredicateRoleV1::SupportCriterion,
        &drifted,
        PredicateDispositionV1::Satisfied,
        '3',
    ));
    assert_eq!(
        resolve_campaign(&fixture.plan, context(&fixture), evaluations),
        Err(ResolutionError::SemanticCommitmentMismatch)
    );
}

#[test]
fn duplicate_predicate_evaluation_is_rejected() {
    let fixture = fixture("resolver-campaign-f");
    let mut evaluations = supported_evaluations(&fixture);
    evaluations.push(evaluation(
        PredicateRoleV1::Control,
        &fixture.control,
        PredicateDispositionV1::Satisfied,
        '4',
    ));
    assert!(matches!(
        resolve_campaign(&fixture.plan, context(&fixture), evaluations),
        Err(ResolutionError::DuplicateEvaluation { .. })
    ));
}

#[test]
fn foreign_campaign_context_is_rejected_before_resolution() {
    let fixture_a = fixture("resolver-campaign-g-a");
    let fixture_b = fixture("resolver-campaign-g-b");
    assert_eq!(
        resolve_campaign(
            &fixture_b.plan,
            context(&fixture_a),
            supported_evaluations(&fixture_b),
        ),
        Err(ResolutionError::ContextPlanMismatch)
    );
}

#[test]
fn evidence_binding_changes_resolution_identity() {
    let fixture = fixture("resolver-campaign-h");
    let a = resolve_campaign(
        &fixture.plan,
        context(&fixture),
        supported_evaluations(&fixture),
    )
    .unwrap();
    let mut changed = supported_evaluations(&fixture);
    changed.retain(|evaluation| {
        !(evaluation.role() == PredicateRoleV1::Control
            && evaluation.semantic().semantic_id() == fixture.control.semantic_id())
    });
    changed.push(evaluation(
        PredicateRoleV1::Control,
        &fixture.control,
        PredicateDispositionV1::Satisfied,
        '0',
    ));
    let b = resolve_campaign(&fixture.plan, context(&fixture), changed).unwrap();
    assert_ne!(a.digest(), b.digest());
}
