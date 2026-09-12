use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt,
    assess_deployment_scoped_safety_case, bind_receipt_to_deployment,
};
use symthaea_evidence_lifecycle::{
    EvidenceLifecycleEvent, EvidenceLifecycleEventKind, ScopedSafetyEvidenceReceipt,
};
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineResolution,
    QuarantineResolutionDisposition, QuarantineSelector, assess_quarantined_safety_case,
};
use symthaea_evidence_time_assurance::{
    EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample, EvidenceTimeStatus,
};
use symthaea_formal_safety::{
    EvidenceKind, ProofObligation, SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus,
};

fn case() -> SafetyCase {
    let mut case = SafetyCase::new("edge-case-fixture");
    case.add_obligation(
        ProofObligation::new("edge-case claim", EvidenceKind::Test)
            .discharge("fixture:workflow-review"),
    );
    case
}

fn context() -> DeploymentEvidenceContext {
    DeploymentEvidenceContext {
        schema_version: "1".into(),
        deployment_id: "edge-node".into(),
        configuration_digest: "blake3:config-v1".into(),
        model_manifest_digest: Some("blake3:model-v1".into()),
        calibration_manifest_digest: Some("blake3:cal-v1".into()),
        evidence_refs: vec!["fixture:deployment".into()],
    }
}

fn scoped(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
    verified_at_ms: u64,
) -> ScopedSafetyEvidenceReceipt {
    let obligation = &case.obligations[0];
    ScopedSafetyEvidenceReceipt {
        receipt: SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: EvidenceKind::Test,
            evidence_ref: format!("artifact:{id}"),
            evidence_digest: format!("blake3:{id}"),
            verifier_ref: verifier.into(),
            verified_at_ms,
        },
        contract_digest: case.contract_digest(),
        valid_from_ms: verified_at_ms,
        valid_until_ms: 10_000,
        applicability_refs: vec!["fixture:applicability".into()],
    }
}

fn bound(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
    verified_at_ms: u64,
) -> DeploymentScopedSafetyReceipt {
    bind_receipt_to_deployment(
        scoped(case, id, verifier, verified_at_ms),
        &context(),
        format!("scope:{id}"),
    )
    .unwrap()
}

fn directive(id: &str, selector: QuarantineSelector) -> EvidenceQuarantineDirective {
    EvidenceQuarantineDirective {
        directive_id: id.into(),
        selector,
        effective_from_ms: 500,
        reason_ref: format!("review:{id}"),
        evidence_refs: vec![format!("evidence:{id}")],
    }
}

fn time_policy() -> EvidenceTimePolicy {
    EvidenceTimePolicy {
        schema_version: "1".into(),
        policy_id: "edge-time-v1".into(),
        expected_clock_source: "ptp-a".into(),
        expected_clock_domain: "domain-a".into(),
        expected_epoch_ref: "epoch:1".into(),
        maximum_clock_uncertainty_ms: 25,
        maximum_wall_clock_backstep_ms: 5,
    }
}

fn time_sample(id: &str, wall: u64, monotonic: u64) -> EvidenceTimeSample {
    EvidenceTimeSample {
        sample_id: id.into(),
        wall_time_ms: wall,
        monotonic_time_ms: monotonic,
        clock_source: "ptp-a".into(),
        clock_domain: "domain-a".into(),
        epoch_ref: "epoch:1".into(),
        clock_uncertainty_ms: 5,
        evidence_refs: vec![format!("time:{id}")],
    }
}

#[test]
fn every_exact_quarantine_selector_can_remove_matching_evidence() {
    let case = case();
    let receipt = bound(&case, "r1", "verifier:a", 100);
    let verified = &receipt.scoped_receipt.receipt;
    let selectors = [
        QuarantineSelector::ReceiptId(verified.receipt_id.clone()),
        QuarantineSelector::EvidenceDigest(verified.evidence_digest.clone()),
        QuarantineSelector::VerifierRef(verified.verifier_ref.clone()),
        QuarantineSelector::EvidenceRef(verified.evidence_ref.clone()),
        QuarantineSelector::ObligationKey(verified.obligation_key.clone()),
    ];

    for (index, selector) in selectors.into_iter().enumerate() {
        let hold = directive(&format!("q-{index}"), selector);
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            std::slice::from_ref(&receipt),
            &[],
            &[hold],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.quarantined_receipts.len(), 1);
    }
}

#[test]
fn reviewed_quarantine_lift_can_restore_existing_evidence() {
    let case = case();
    let receipt = bound(&case, "r1", "verifier:a", 100);
    let hold = directive(
        "q-verifier",
        QuarantineSelector::VerifierRef("verifier:a".into()),
    );
    let resolution = EvidenceQuarantineResolution {
        resolution_id: "resolution:q-verifier".into(),
        directive_id: hold.directive_id.clone(),
        resolved_at_ms: 700,
        disposition: QuarantineResolutionDisposition::LiftAfterReview,
        resolution_ref: "review:lift".into(),
        evidence_refs: vec!["evidence:lift".into()],
    };
    let report = assess_quarantined_safety_case(
        &case,
        &context(),
        &[receipt],
        &[],
        &[hold],
        &[resolution],
        1_000,
    );
    assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
}

#[test]
fn superseded_receipt_drops_out_and_replacement_carries_readiness() {
    let case = case();
    let old = bound(&case, "old", "verifier:a", 100);
    let new = bound(&case, "new", "verifier:b", 500);
    let event = EvidenceLifecycleEvent {
        event_id: "supersede-old".into(),
        receipt_id: "old".into(),
        event_at_ms: 600,
        kind: EvidenceLifecycleEventKind::Superseded {
            by_receipt_id: "new".into(),
            reason_ref: "review:new-evidence".into(),
        },
    };
    let report = assess_deployment_scoped_safety_case(
        &case,
        &context(),
        &[old, new],
        &[event],
        1_000,
    );
    assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
    assert_eq!(report.lifecycle_report.active_receipt_count, 1);
}

#[test]
fn duplicate_time_sample_id_latches_time_untrusted() {
    let mut guard = EvidenceTimeGuard::new(time_policy()).unwrap();
    let first = guard.observe(time_sample("clock-1", 1_000, 1_000));
    assert_eq!(first.status, EvidenceTimeStatus::Trusted);
    let duplicate = guard.observe(time_sample("clock-1", 1_001, 1_001));
    assert_eq!(duplicate.status, EvidenceTimeStatus::Untrusted);
}

#[test]
fn future_lifecycle_event_fails_closed() {
    let case = case();
    let receipt = bound(&case, "r1", "verifier:a", 100);
    let future = EvidenceLifecycleEvent {
        event_id: "future-revoke".into(),
        receipt_id: "r1".into(),
        event_at_ms: 2_000,
        kind: EvidenceLifecycleEventKind::Revoked {
            reason_ref: "review:future".into(),
        },
    };
    let report = assess_deployment_scoped_safety_case(
        &case,
        &context(),
        &[receipt],
        &[future],
        1_000,
    );
    assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
}

#[test]
fn future_quarantine_directive_fails_closed() {
    let case = case();
    let receipt = bound(&case, "r1", "verifier:a", 100);
    let mut future = directive(
        "future-q",
        QuarantineSelector::ReceiptId("r1".into()),
    );
    future.effective_from_ms = 2_000;
    let report = assess_quarantined_safety_case(
        &case,
        &context(),
        &[receipt],
        &[],
        &[future],
        &[],
        1_000,
    );
    assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
}

#[test]
fn receipt_from_another_safety_contract_is_rejected() {
    let case_a = case();
    let mut case_b = SafetyCase::new("different-contract");
    case_b.add_obligation(
        ProofObligation::new("different claim", EvidenceKind::Test)
            .discharge("fixture:other-review"),
    );
    let wrong_contract = bound(&case_a, "r1", "verifier:a", 100);
    let report = assess_deployment_scoped_safety_case(
        &case_b,
        &context(),
        &[wrong_contract],
        &[],
        1_000,
    );
    assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
}
