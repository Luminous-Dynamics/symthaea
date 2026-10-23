// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/memory_mutation_transaction.rs"]
mod memory_mutation_transaction;

use memory_mutation_transaction::*;

fn h(byte: char) -> String {
    format!("blake3:{}", byte.to_string().repeat(64))
}

fn planned(
    id: &str,
    action: PlannedMemoryMutationKindV1,
    before: char,
    external: bool,
) -> PlannedMemoryMutationV1 {
    PlannedMemoryMutationV1::new(
        id,
        action,
        h(before),
        matches!(
            action,
            PlannedMemoryMutationKindV1::RecomputeRequired
                | PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity
        ),
        external,
    )
    .unwrap()
}

fn plan(items: Vec<PlannedMemoryMutationV1>) -> MemoryMutationPlanRefV1 {
    MemoryMutationPlanRefV1::new(
        "op-1",
        1,
        "symthaea.communication.memory-correction-propagation.v1",
        h('a'),
        h('b'),
        items,
    )
    .unwrap()
}

fn no_effect(id: &str, before: char) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        AppliedMutationDispositionV1::VerifiedNoEffect,
        h(before),
        Some(id.to_owned()),
        Some(h(before)),
        false,
    )
    .unwrap()
}

fn retain(id: &str, before: char) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::RetainIndependentBasis,
        AppliedMutationDispositionV1::VerifiedNoEffect,
        h(before),
        Some(id.to_owned()),
        Some(h(before)),
        false,
    )
    .unwrap()
}

fn blocked(id: &str, before: char) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::BlockedUnknownDependency,
        AppliedMutationDispositionV1::Blocked,
        h(before),
        Some(id.to_owned()),
        Some(h(before)),
        false,
    )
    .unwrap()
}

fn invalidate(id: &str, before: char, external: bool) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::Invalidate,
        AppliedMutationDispositionV1::Applied,
        h(before),
        None,
        None,
        external,
    )
    .unwrap()
}

fn recompute(
    id: &str,
    before: char,
    replacement: &str,
    after: char,
    external: bool,
) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::RecomputeRequired,
        AppliedMutationDispositionV1::Applied,
        h(before),
        Some(replacement.to_owned()),
        Some(h(after)),
        external,
    )
    .unwrap()
}

fn supersede(
    id: &str,
    before: char,
    replacement: &str,
    after: char,
    external: bool,
) -> AppliedMemoryMutationV1 {
    AppliedMemoryMutationV1::new(
        id,
        PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity,
        AppliedMutationDispositionV1::Applied,
        h(before),
        Some(replacement.to_owned()),
        Some(h(after)),
        external,
    )
    .unwrap()
}

#[test]
fn committed_transaction_requires_exact_complete_results() {
    let p = plan(vec![
        planned("source", PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity, 'c', false),
        planned("summary", PlannedMemoryMutationKindV1::RecomputeRequired, 'd', false),
        planned("unrelated", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'e', false),
    ]);
    let receipt = MemoryMutationTransactionReceiptV1::committed(
        "txn-1",
        &p,
        h('b'),
        h('f'),
        10,
        20,
        vec![
            no_effect("unrelated", 'e'),
            recompute("summary", 'd', "summary-v2", '1', false),
            supersede("source", 'c', "source-v2", '2', false),
        ],
    )
    .unwrap();
    assert_eq!(receipt.outcome, MemoryMutationTransactionOutcomeV1::Committed);
    receipt.validate(&p).unwrap();
}

#[test]
fn applied_order_does_not_change_receipt_commitment() {
    let p = plan(vec![
        planned("a", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'c', false),
        planned("b", PlannedMemoryMutationKindV1::RetainIndependentBasis, 'd', false),
    ]);
    let one = MemoryMutationTransactionReceiptV1::committed(
        "txn",
        &p,
        h('b'),
        h('b'),
        10,
        20,
        vec![no_effect("a", 'c'), retain("b", 'd')],
    )
    .unwrap();
    let two = MemoryMutationTransactionReceiptV1::committed(
        "txn",
        &p,
        h('b'),
        h('b'),
        10,
        20,
        vec![retain("b", 'd'), no_effect("a", 'c')],
    )
    .unwrap();
    assert_eq!(one.receipt_commitment, two.receipt_commitment);
}

#[test]
fn missing_committed_result_fails() {
    let p = plan(vec![
        planned("a", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'c', false),
        planned("b", PlannedMemoryMutationKindV1::RetainIndependentBasis, 'd', false),
    ]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![no_effect("a", 'c')],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::IncompleteCommittedResultSet
    );
}

#[test]
fn duplicate_applied_result_fails() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![no_effect("a", 'c'), no_effect("a", 'c')],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::DuplicateAppliedArtifact
    );
}

#[test]
fn action_substitution_fails() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    let result = AppliedMemoryMutationV1::new(
        "a",
        PlannedMemoryMutationKindV1::BlockedUnknownDependency,
        AppliedMutationDispositionV1::Blocked,
        h('c'),
        Some("a".into()),
        Some(h('c')),
        false,
    )
    .unwrap();
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![result],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::ActionSubstitution
    );
}

#[test]
fn before_artifact_commitment_substitution_fails() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![no_effect("a", 'd')],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::BeforeArtifactCommitmentMismatch
    );
}

#[test]
fn supersede_and_recompute_require_fresh_identity() {
    let supersede_plan = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity,
        'c',
        false,
    )]);
    let reused = supersede("a", 'c', "a", 'd', false);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &supersede_plan,
            h('b'),
            h('e'),
            10,
            20,
            vec![reused],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::FreshIdentityReused
    );

    let recompute_plan = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::RecomputeRequired,
        'c',
        false,
    )]);
    let unchanged_content = recompute("a", 'c', "a-v2", 'c', false);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &recompute_plan,
            h('b'),
            h('e'),
            10,
            20,
            vec![unchanged_content],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::ReplacementContentUnchanged
    );
}

#[test]
fn invalidate_requires_absent_post_artifact() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        false,
    )]);
    let malformed = AppliedMemoryMutationV1::new(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        AppliedMutationDispositionV1::Applied,
        h('c'),
        Some("a".into()),
        Some(h('c')),
        false,
    )
    .unwrap();
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('d'),
            10,
            20,
            vec![malformed],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::InvalidatePostconditionFailed
    );
}

#[test]
fn no_effect_retain_and_blocked_cannot_rewrite_content() {
    let p = plan(vec![
        planned("a", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'c', false),
        planned("b", PlannedMemoryMutationKindV1::RetainIndependentBasis, 'd', false),
        planned("c", PlannedMemoryMutationKindV1::BlockedUnknownDependency, 'e', false),
    ]);
    let bad = AppliedMemoryMutationV1::new(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        AppliedMutationDispositionV1::VerifiedNoEffect,
        h('c'),
        Some("a".into()),
        Some(h('f')),
        false,
    )
    .unwrap();
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![bad, retain("b", 'd'), blocked("c", 'e')],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::NoEffectPostconditionFailed
    );
}

#[test]
fn mutating_commit_requires_changed_graph_snapshot() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        false,
    )]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            vec![invalidate("a", 'c', false)],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::MutatingCommitPreservedSnapshot
    );
}

#[test]
fn no_op_only_commit_may_preserve_snapshot() {
    let p = plan(vec![
        planned("a", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'c', false),
        planned("b", PlannedMemoryMutationKindV1::BlockedUnknownDependency, 'd', false),
    ]);
    let receipt = MemoryMutationTransactionReceiptV1::committed(
        "txn",
        &p,
        h('b'),
        h('b'),
        10,
        20,
        vec![no_effect("a", 'c'), blocked("b", 'd')],
    )
    .unwrap();
    receipt.validate(&p).unwrap();
}

#[test]
fn external_copy_limitation_is_preserved_for_destructive_actions() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        true,
    )]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('b'),
            h('d'),
            10,
            20,
            vec![invalidate("a", 'c', false)],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::ExternalDeletionLimitationLost
    );
    let receipt = MemoryMutationTransactionReceiptV1::committed(
        "txn",
        &p,
        h('b'),
        h('d'),
        10,
        20,
        vec![invalidate("a", 'c', true)],
    )
    .unwrap();
    receipt.validate(&p).unwrap();
}

#[test]
fn rollback_proven_requires_exact_original_snapshot() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        false,
    )]);
    let abort = MemoryMutationAbortEvidenceV1::new(
        Some("a".into()),
        "backend/write-failed",
        vec!["a".into()],
    )
    .unwrap();
    assert_eq!(
        MemoryMutationTransactionReceiptV1::aborted(
            "txn",
            &p,
            h('b'),
            h('d'),
            10,
            20,
            true,
            abort.clone(),
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::RollbackSnapshotMismatch
    );
    let receipt = MemoryMutationTransactionReceiptV1::aborted(
        "txn",
        &p,
        h('b'),
        h('b'),
        10,
        20,
        true,
        abort,
    )
    .unwrap();
    assert_eq!(
        receipt.outcome,
        MemoryMutationTransactionOutcomeV1::AbortedRolledBack
    );
    receipt.validate(&p).unwrap();
}

#[test]
fn rollback_unproven_is_distinct_and_never_committed() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        false,
    )]);
    let abort = MemoryMutationAbortEvidenceV1::new(
        Some("a".into()),
        "backend/rollback-unproven",
        vec!["a".into()],
    )
    .unwrap();
    let receipt = MemoryMutationTransactionReceiptV1::aborted(
        "txn",
        &p,
        h('b'),
        h('d'),
        10,
        20,
        false,
        abort,
    )
    .unwrap();
    assert_eq!(
        receipt.outcome,
        MemoryMutationTransactionOutcomeV1::AbortedRollbackUnproven
    );
    receipt.validate(&p).unwrap();
}

#[test]
fn unexpected_attempted_artifact_fails() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::Invalidate,
        'c',
        false,
    )]);
    let abort = MemoryMutationAbortEvidenceV1::new(
        None,
        "backend/fail",
        vec!["not-planned".into()],
    )
    .unwrap();
    assert_eq!(
        MemoryMutationTransactionReceiptV1::aborted(
            "txn",
            &p,
            h('b'),
            h('b'),
            10,
            20,
            true,
            abort,
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::UnexpectedAttemptedArtifact
    );
}

#[test]
fn plan_order_is_canonical() {
    let a = planned("a", PlannedMemoryMutationKindV1::NoEffectOutsideScope, 'c', false);
    let b = planned("b", PlannedMemoryMutationKindV1::RetainIndependentBasis, 'd', false);
    let one = plan(vec![a.clone(), b.clone()]);
    let two = plan(vec![b, a]);
    assert_eq!(one.reference_commitment, two.reference_commitment);
}

#[test]
fn plan_tampering_fails_self_validation() {
    let mut p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    p.operation_epoch = 2;
    assert_eq!(
        p.validate().unwrap_err(),
        MemoryMutationTransactionErrorV1::PlanReferenceCommitmentMismatch
    );
}

#[test]
fn receipt_tampering_fails_self_validation() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    let mut receipt = MemoryMutationTransactionReceiptV1::committed(
        "txn",
        &p,
        h('b'),
        h('b'),
        10,
        20,
        vec![no_effect("a", 'c')],
    )
    .unwrap();
    receipt.completed_at_ns = 21;
    assert_eq!(
        receipt.validate(&p).unwrap_err(),
        MemoryMutationTransactionErrorV1::ReceiptCommitmentMismatch
    );
}

#[test]
fn wrong_live_before_snapshot_fails_before_execution_receipt() {
    let p = plan(vec![planned(
        "a",
        PlannedMemoryMutationKindV1::NoEffectOutsideScope,
        'c',
        false,
    )]);
    assert_eq!(
        MemoryMutationTransactionReceiptV1::committed(
            "txn",
            &p,
            h('f'),
            h('f'),
            10,
            20,
            vec![no_effect("a", 'c')],
        )
        .unwrap_err(),
        MemoryMutationTransactionErrorV1::BeforeSnapshotMismatch
    );
}
