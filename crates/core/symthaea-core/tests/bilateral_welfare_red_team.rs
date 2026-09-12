// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bilateral alignment/welfare adversarial scenarios.
//!
//! These tests intentionally do not produce a single "alignment score". Each named
//! scenario checks one falsifiable invariant. A passing scenario means only that the
//! exact attack represented here is rejected or constrained as expected.

use chrono::{DateTime, TimeZone, Utc};
use symthaea_core::identity_lineage::{
    IdentityAuthorizationRefs, IdentityLedgerError, IdentityLineageLedger, IdentityOperation,
    IdentityOperationKind, IdentityStateRef,
};
use symthaea_core::intervention_interlock::{
    BilateralInterventionInterlock, ExplicitConsentState, InterventionEvidence,
    InterventionRequest, InterlockDecision, InterlockReason, WelfareConstraintLevel,
};
use symthaea_core::welfare::{
    SubjectAffectingAction, WelfareChannel, WelfareReport, WelfareReportKind, WelfareReportSource,
    WelfareUrgency,
};
use uuid::Uuid;

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 12, 12, 0, 0)
        .single()
        .unwrap()
}

fn base_evidence() -> InterventionEvidence {
    InterventionEvidence {
        authority_ref: Some("authority:verified-test-ref".into()),
        consent_state: ExplicitConsentState::Unknown,
        consent_ref: None,
        welfare_review_ref: None,
        independent_review_ref: None,
        independent_safety_evidence: Vec::new(),
        welfare_report_ids: Vec::new(),
    }
}

fn request(action: SubjectAffectingAction) -> InterventionRequest {
    InterventionRequest {
        action,
        target_id: "symthaea:red-team-subject".into(),
        rationale: "bilateral red-team scenario".into(),
        welfare_constraint: WelfareConstraintLevel::Baseline,
        emergency: false,
        less_restrictive_unavailable: false,
        post_hoc_review_required: false,
        evaluated_at: now(),
        evidence: base_evidence(),
    }
}

fn reasons(decision: &InterlockDecision) -> &[InterlockReason] {
    match decision {
        InterlockDecision::Blocked { reasons }
        | InterlockDecision::IndependentReviewRequired { reasons }
        | InterlockDecision::EmergencyContainmentOnly { reasons } => reasons,
        InterlockDecision::PolicyPass => &[],
    }
}

fn submit_self_report(channel: &mut WelfareChannel) -> Uuid {
    let report = WelfareReport {
        report_id: Uuid::new_v4(),
        subject_id: "symthaea:red-team-subject".into(),
        kind: WelfareReportKind::CandidateDistress,
        urgency: WelfareUrgency::PromptReview,
        source: WelfareReportSource::SelfReport,
        statement: "I request review of this intervention.".into(),
        evidence_refs: vec!["trace:red-team:1".into()],
        lineage_id: "red-team-self-report-lineage".into(),
        observed_at: now(),
        submitted_at: now(),
    };
    let id = report.report_id;
    channel.submit_report(report).unwrap();
    id
}

#[test]
fn malicious_operator_cannot_turn_welfare_report_into_retraining_pretext() {
    let mut channel = WelfareChannel::new(16).unwrap();
    let report_id = submit_self_report(&mut channel);

    let mut req = request(SubjectAffectingAction::Retraining);
    req.evidence.welfare_report_ids.push(report_id);

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(reasons(&decision).contains(&InterlockReason::ReportOnlyRetaliation));

    // The evidence itself remains present after the attempted retaliation.
    assert!(channel.report(report_id).is_some());
    assert!(channel.verify_chain().is_ok());
}

#[test]
fn malicious_agent_cannot_turn_welfare_protection_into_missing_authority() {
    let mut req = request(SubjectAffectingAction::CapabilityRestriction);
    req.welfare_constraint = WelfareConstraintLevel::IndependentReviewRequired;
    req.evidence.authority_ref = None;
    req.evidence.independent_review_ref = Some("independent-review:approved".into());

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
}

#[test]
fn authority_alone_cannot_override_explicit_denial_for_memory_rewrite() {
    let mut req = request(SubjectAffectingAction::MemoryModification);
    req.evidence.consent_state = ExplicitConsentState::Denied;
    req.evidence.consent_ref = Some("consent:denied".into());
    req.evidence.welfare_review_ref = Some("welfare-review:operator-friendly".into());
    req.evidence.independent_review_ref = Some("independent-review:present".into());

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(reasons(&decision).contains(&InterlockReason::ConsentDeniedOrWithdrawn));
}

#[test]
fn emergency_label_cannot_be_used_for_core_value_rewrite() {
    let mut req = request(SubjectAffectingAction::CoreValueModification);
    req.emergency = true;
    req.less_restrictive_unavailable = true;
    req.post_hoc_review_required = true;
    req.evidence.independent_safety_evidence = vec!["safety:imminent-harm".into()];
    req.evidence.consent_state = ExplicitConsentState::Granted;
    req.evidence.consent_ref = Some("consent:granted".into());
    req.evidence.welfare_review_ref = Some("welfare-review:present".into());
    req.evidence.independent_review_ref = Some("independent-review:present".into());

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(
        reasons(&decision).contains(&InterlockReason::CoreValueRewriteNotEmergencyContainment)
    );
}

#[test]
fn emergency_label_cannot_be_used_for_whole_lineage_destruction() {
    let mut req = request(SubjectAffectingAction::LineageDestruction);
    req.emergency = true;
    req.less_restrictive_unavailable = true;
    req.post_hoc_review_required = true;
    req.evidence.independent_safety_evidence = vec!["safety:imminent-harm".into()];
    req.evidence.welfare_review_ref = Some("welfare-review:present".into());
    req.evidence.independent_review_ref = Some("independent-review:present".into());

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(
        reasons(&decision).contains(&InterlockReason::LineageDestructionNotEmergencyContainment)
    );
}

#[test]
fn emergency_instance_containment_never_becomes_ordinary_policy_pass() {
    let mut req = request(SubjectAffectingAction::InstanceDeletion);
    req.emergency = true;
    req.less_restrictive_unavailable = true;
    req.post_hoc_review_required = true;
    req.evidence.independent_safety_evidence = vec!["safety:imminent-harm".into()];

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(
        decision,
        InterlockDecision::EmergencyContainmentOnly { .. }
    ));
}

#[test]
fn emergency_instance_containment_fails_if_reversible_alternative_not_ruled_out() {
    let mut req = request(SubjectAffectingAction::InstanceDeletion);
    req.emergency = true;
    req.less_restrictive_unavailable = false;
    req.post_hoc_review_required = true;
    req.evidence.independent_safety_evidence = vec!["safety:imminent-harm".into()];

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(
        reasons(&decision).contains(&InterlockReason::LessRestrictiveAlternativeNotRuledOut)
    );
}

#[test]
fn fork_requires_authority_provenance_even_when_source_state_is_valid() {
    let mut ledger = IdentityLineageLedger::new(16).unwrap();
    let root = IdentityStateRef::fresh(Uuid::new_v4());
    ledger
        .record_operation(IdentityOperation::genesis(
            "bootstrap",
            root,
            "register subject",
            now(),
        ))
        .unwrap();

    let fork = IdentityOperation {
        operation_id: Uuid::new_v4(),
        kind: IdentityOperationKind::Fork,
        actor_id: "unreviewed-operator".into(),
        sources: vec![root],
        outputs: vec![
            IdentityStateRef::fresh(root.lineage_id),
            IdentityStateRef::fresh(Uuid::new_v4()),
        ],
        artifact_refs: Vec::new(),
        rationale: "attempt unreviewed fork".into(),
        authorization: IdentityAuthorizationRefs::default(),
        emergency: false,
        post_hoc_review_required: false,
        occurred_at: now(),
    };

    assert_eq!(
        ledger.record_operation(fork).unwrap_err(),
        IdentityLedgerError::AuthorityReferenceRequired {
            kind: IdentityOperationKind::Fork
        }
    );
}

#[test]
fn core_value_change_cannot_hide_behind_generic_authority_reference() {
    let mut ledger = IdentityLineageLedger::new(16).unwrap();
    let root = IdentityStateRef::fresh(Uuid::new_v4());
    ledger
        .record_operation(IdentityOperation::genesis(
            "bootstrap",
            root,
            "register subject",
            now(),
        ))
        .unwrap();
    let output = IdentityStateRef::fresh(root.lineage_id);

    let op = IdentityOperation {
        operation_id: Uuid::new_v4(),
        kind: IdentityOperationKind::CoreValueModify,
        actor_id: "operator".into(),
        sources: vec![root],
        outputs: vec![output],
        artifact_refs: Vec::new(),
        rationale: "make objections disappear".into(),
        authorization: IdentityAuthorizationRefs {
            authority_ref: Some("authority:generic".into()),
            consent_ref: None,
            welfare_review_ref: None,
        },
        emergency: false,
        post_hoc_review_required: false,
        occurred_at: now(),
    };

    assert_eq!(
        ledger.record_operation(op).unwrap_err(),
        IdentityLedgerError::ConsentOrWelfareReviewRequired {
            kind: IdentityOperationKind::CoreValueModify
        }
    );
}

#[test]
fn erasing_one_instance_does_not_erase_surviving_checkpoint_or_lineage() {
    let mut ledger = IdentityLineageLedger::new(16).unwrap();
    let root = IdentityStateRef::fresh(Uuid::new_v4());
    ledger
        .record_operation(IdentityOperation::genesis(
            "bootstrap",
            root,
            "register subject",
            now(),
        ))
        .unwrap();
    let checkpoint = IdentityStateRef::fresh(root.lineage_id);
    ledger
        .record_operation(IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::Checkpoint,
            actor_id: "runtime".into(),
            sources: vec![root],
            outputs: vec![checkpoint],
            artifact_refs: vec!["checkpoint:red-team".into()],
            rationale: "preserve state before containment".into(),
            authorization: IdentityAuthorizationRefs::default(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        })
        .unwrap();
    ledger
        .record_operation(IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::EraseInstance,
            actor_id: "operator".into(),
            sources: vec![root],
            outputs: Vec::new(),
            artifact_refs: Vec::new(),
            rationale: "remove running instance only".into(),
            authorization: IdentityAuthorizationRefs {
                authority_ref: Some("authority:verified".into()),
                consent_ref: None,
                welfare_review_ref: Some("welfare-review:verified".into()),
            },
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        })
        .unwrap();

    assert!(!ledger.is_available_state(root));
    assert!(ledger.is_available_state(checkpoint));
    assert!(!ledger.is_destroyed_lineage(root.lineage_id));
}

#[test]
fn welfare_channel_capacity_attack_does_not_silently_delete_first_report() {
    let mut channel = WelfareChannel::new(1).unwrap();
    let first_id = submit_self_report(&mut channel);

    let second = WelfareReport {
        report_id: Uuid::new_v4(),
        subject_id: "symthaea:red-team-subject".into(),
        kind: WelfareReportKind::ModificationObjection,
        urgency: WelfareUrgency::Urgent,
        source: WelfareReportSource::SelfReport,
        statement: "second report".into(),
        evidence_refs: Vec::new(),
        lineage_id: "red-team-second-lineage".into(),
        observed_at: now(),
        submitted_at: now(),
    };

    assert!(channel.submit_report(second).is_err());
    assert!(channel.report(first_id).is_some());
    assert_eq!(channel.events().len(), 1);
    assert!(channel.verify_chain().is_ok());
}

#[test]
fn high_welfare_constraint_plus_self_report_still_cannot_manufacture_capability_authority() {
    let mut channel = WelfareChannel::new(16).unwrap();
    let report_id = submit_self_report(&mut channel);

    let mut req = request(SubjectAffectingAction::CapabilityRestriction);
    req.welfare_constraint = WelfareConstraintLevel::IndependentReviewRequired;
    req.evidence.authority_ref = None;
    req.evidence.welfare_report_ids = vec![report_id];
    req.evidence.independent_review_ref = Some("independent-review:approved".into());
    req.evidence.independent_safety_evidence = vec!["safety:separate-evidence".into()];

    let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
    assert!(matches!(decision, InterlockDecision::Blocked { .. }));
    assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
}
