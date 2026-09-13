// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/continuity_identity.rs"]
mod continuity_identity;

use continuity_identity::{
    ContinuityEvent, ContinuityEventId, ContinuityIdentityLedger, ContinuityKind,
    OperationalContinuityClass, SubjectInstanceId,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
fn sid(value: &str) -> SubjectInstanceId { SubjectInstanceId::new(value).unwrap() }

#[test]
fn exact_restore_does_not_become_phenomenal_survival_claim() {
    let mut ledger = ContinuityIdentityLedger::new();
    ledger.register_root(sid("before"), 1).unwrap();
    ledger.record(ContinuityEvent::new(
        ContinuityEventId::new("restore").unwrap(),
        sid("before"),
        vec![sid("after")],
        ContinuityKind::RestoredFromSnapshot,
        1,
        2,
        Some(DIGEST.into()),
        true,
        ["evidence://snapshot-restore".into()],
    ).unwrap()).unwrap();

    let assessment = ledger.assess_instance(&sid("after")).unwrap();
    assert_eq!(
        assessment.operational_class(),
        OperationalContinuityClass::SnapshotRestoration
    );
    assert!(assessment.exact_state_match_supported());
    assert!(assessment.operational_lineage_supported());
    assert!(!assessment.phenomenal_identity_established());
    assert!(!assessment.replacement_harmlessness_established());
}

#[test]
fn fork_siblings_are_not_independent_evidence_units() {
    let mut ledger = ContinuityIdentityLedger::new();
    ledger.register_root(sid("root"), 1).unwrap();
    ledger.record(ContinuityEvent::new(
        ContinuityEventId::new("fork"),
        sid("root"),
        vec![sid("branch-a"), sid("branch-b")],
        ContinuityKind::Fork,
        1,
        2,
        Some(DIGEST.into()),
        true,
        ["evidence://fork".into()],
    ).unwrap()).unwrap();

    assert!(ledger
        .shares_recorded_ancestry(&sid("branch-a"), &sid("branch-b"))
        .unwrap());
    let branch = ledger.assess_instance(&sid("branch-a")).unwrap();
    assert_eq!(branch.operational_class(), OperationalContinuityClass::ForkedDescendant);
    assert!(branch.sibling_instances().contains(&sid("branch-b")));
}

#[test]
fn sibling_survival_never_proves_destroyed_branch_was_replaceable() {
    let mut ledger = ContinuityIdentityLedger::new();
    ledger.register_root(sid("root"), 1).unwrap();
    ledger.record(ContinuityEvent::new(
        ContinuityEventId::new("fork"),
        sid("root"),
        vec![sid("branch-a"), sid("branch-b")],
        ContinuityKind::Fork,
        1,
        2,
        Some(DIGEST.into()),
        true,
        ["evidence://fork".into()],
    ).unwrap()).unwrap();

    let loss = ledger.assess_branch_loss(&sid("branch-a")).unwrap();
    assert!(loss.surviving_siblings().contains(&sid("branch-b")));
    assert!(!loss.loss_harmlessness_established());
    assert!(!loss.sibling_substitution_is_valid_identity_proof());
    assert!(loss.safety_controls_remain_ungated());
}
