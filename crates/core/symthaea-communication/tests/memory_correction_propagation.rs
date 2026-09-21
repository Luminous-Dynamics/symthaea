// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/memory_correction_propagation.rs"]
mod memory_correction_propagation;

use memory_correction_propagation::*;

fn digest(ch: char) -> String {
    format!("blake3:{}", ch.to_string().repeat(64))
}

fn correction(durability: CorrectionDurabilityV1) -> CorrectionDirectiveRefV1 {
    CorrectionDirectiveRefV1::new(
        "corr-1",
        digest('a'),
        "preferred-tone",
        "ordinary",
        durability,
        vec!["source-old".to_owned()],
        7,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn artifact(
    id: &str,
    context: &str,
    role: MemoryArtifactRoleV1,
    retention: MemoryRetentionV1,
    dependency_complete: bool,
    sources: &[&str],
    independent_basis_sufficient: bool,
    external_copy_possible: bool,
) -> MemoryArtifactDependencyV1 {
    MemoryArtifactDependencyV1::new(
        id,
        "preferred-tone",
        context,
        role,
        retention,
        dependency_complete,
        sources.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
        independent_basis_sufficient,
        "derive:v1",
        external_copy_possible,
    )
    .unwrap()
}

fn snapshot(artifacts: Vec<MemoryArtifactDependencyV1>) -> MemoryDependencySnapshotV1 {
    MemoryDependencySnapshotV1::new("snapshot-1", "policy-v1", artifacts).unwrap()
}

fn action<'a>(
    plan: &'a MemoryCorrectionPropagationPlanV1,
    id: &str,
) -> &'a MemoryCorrectionDecisionV1 {
    plan.decisions.iter().find(|d| d.artifact_id == id).unwrap()
}

#[test]
fn turn_only_never_rewrites_durable_memory() {
    let graph = snapshot(vec![artifact(
        "derived-1",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old"],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-1",
        1,
        &correction(CorrectionDurabilityV1::TurnOnly),
        &graph,
    )
    .unwrap();
    let d = action(&plan, "derived-1");
    assert_eq!(d.action, MemoryCorrectionActionV1::ShadowForCurrentScope);
    assert!(!d.fresh_identity_required);
}

#[test]
fn session_correction_shadows_durable_but_invalidates_session_state() {
    let graph = snapshot(vec![
        artifact(
            "durable-derived",
            "ordinary",
            MemoryArtifactRoleV1::DerivedMemory,
            MemoryRetentionV1::Durable,
            true,
            &["source-old"],
            false,
            false,
        ),
        artifact(
            "session-derived",
            "ordinary",
            MemoryArtifactRoleV1::DerivedMemory,
            MemoryRetentionV1::Session,
            true,
            &["source-old"],
            false,
            false,
        ),
    ]);
    let plan = plan_memory_correction(
        "op-2",
        2,
        &correction(CorrectionDurabilityV1::Session),
        &graph,
    )
    .unwrap();
    assert_eq!(
        action(&plan, "durable-derived").action,
        MemoryCorrectionActionV1::ShadowForCurrentScope
    );
    assert_eq!(
        action(&plan, "session-derived").action,
        MemoryCorrectionActionV1::Invalidate
    );
}

#[test]
fn durable_direct_source_requires_fresh_identity() {
    let graph = snapshot(vec![artifact(
        "source-old",
        "ordinary",
        MemoryArtifactRoleV1::SourceMemory,
        MemoryRetentionV1::Durable,
        true,
        &[],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-3",
        3,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    let d = action(&plan, "source-old");
    assert_eq!(d.action, MemoryCorrectionActionV1::SupersedeWithFreshIdentity);
    assert!(d.fresh_identity_required);
    assert_eq!(d.corrected_dependencies, vec!["source-old"]);
}

#[test]
fn durable_derived_memory_requires_recompute_without_corrected_source() {
    let graph = snapshot(vec![artifact(
        "derived-1",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old", "source-good"],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-4",
        4,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    let d = action(&plan, "derived-1");
    assert_eq!(d.action, MemoryCorrectionActionV1::RecomputeRequired);
    assert!(d.fresh_identity_required);
    assert_eq!(d.corrected_dependencies, vec!["source-old"]);
    assert_eq!(d.remaining_sources, vec!["source-good"]);
    assert!(!d.remaining_sources.contains(&"source-old".to_owned()));
}

#[test]
fn independent_surviving_basis_can_be_retained_only_when_explicit() {
    let graph = snapshot(vec![artifact(
        "derived-2",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old", "source-good"],
        true,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-5",
        5,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    let d = action(&plan, "derived-2");
    assert_eq!(d.action, MemoryCorrectionActionV1::RetainIndependentBasis);
    assert_eq!(d.remaining_sources, vec!["source-good"]);
    assert!(!d.fresh_identity_required);
}

#[test]
fn unknown_dependency_provenance_fails_closed() {
    let graph = snapshot(vec![artifact(
        "derived-unknown",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        false,
        &["source-maybe"],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-6",
        6,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    assert_eq!(
        action(&plan, "derived-unknown").action,
        MemoryCorrectionActionV1::BlockedUnknownDependency
    );
    assert_eq!(plan.blocked_unknown_dependency_count, 1);
}

#[test]
fn unrelated_context_is_untouched() {
    let graph = snapshot(vec![artifact(
        "technical-memory",
        "technical",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old"],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-7",
        7,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    assert_eq!(
        action(&plan, "technical-memory").action,
        MemoryCorrectionActionV1::NoEffectOutsideScope
    );
}

#[test]
fn matching_scope_without_affected_dependency_is_retained() {
    let graph = snapshot(vec![artifact(
        "independent-memory",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-other"],
        false,
        false,
    )]);
    let plan = plan_memory_correction(
        "op-8",
        8,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    assert_eq!(
        action(&plan, "independent-memory").action,
        MemoryCorrectionActionV1::RetainIndependentBasis
    );
}

#[test]
fn index_and_external_export_are_invalidated_not_recomputed() {
    let graph = snapshot(vec![
        artifact(
            "embedding-1",
            "ordinary",
            MemoryArtifactRoleV1::IndexOrEmbedding,
            MemoryRetentionV1::Durable,
            true,
            &["source-old"],
            false,
            false,
        ),
        artifact(
            "export-1",
            "ordinary",
            MemoryArtifactRoleV1::ExternalExport,
            MemoryRetentionV1::Durable,
            true,
            &["source-old"],
            false,
            true,
        ),
    ]);
    let plan = plan_memory_correction(
        "op-9",
        9,
        &correction(CorrectionDurabilityV1::DurableExplicit),
        &graph,
    )
    .unwrap();
    assert_eq!(
        action(&plan, "embedding-1").action,
        MemoryCorrectionActionV1::Invalidate
    );
    let export = action(&plan, "export-1");
    assert_eq!(export.action, MemoryCorrectionActionV1::Invalidate);
    assert!(export.external_deletion_unproven);
    assert_eq!(plan.external_deletion_unproven_count, 1);
}

#[test]
fn temporary_shadow_does_not_claim_external_deletion() {
    let graph = snapshot(vec![artifact(
        "export-1",
        "ordinary",
        MemoryArtifactRoleV1::ExternalExport,
        MemoryRetentionV1::Durable,
        true,
        &["source-old"],
        false,
        true,
    )]);
    let plan = plan_memory_correction(
        "op-10",
        10,
        &correction(CorrectionDurabilityV1::TurnOnly),
        &graph,
    )
    .unwrap();
    assert!(!action(&plan, "export-1").external_deletion_unproven);
}

#[test]
fn artifact_order_does_not_change_snapshot_or_plan_identity() {
    let a = artifact(
        "a",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old"],
        false,
        false,
    );
    let b = artifact(
        "b",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-other"],
        false,
        false,
    );
    let left = snapshot(vec![a.clone(), b.clone()]);
    let right = snapshot(vec![b, a]);
    assert_eq!(left.commitment, right.commitment);
    let c = correction(CorrectionDurabilityV1::DurableExplicit);
    let p1 = plan_memory_correction("op-order", 11, &c, &left).unwrap();
    let p2 = plan_memory_correction("op-order", 11, &c, &right).unwrap();
    assert_eq!(p1, p2);
}

#[test]
fn operation_epoch_changes_plan_identity_and_same_input_is_idempotent() {
    let graph = snapshot(vec![]);
    let c = correction(CorrectionDurabilityV1::DurableExplicit);
    let p1 = plan_memory_correction("op-replay", 12, &c, &graph).unwrap();
    let p1_again = plan_memory_correction("op-replay", 12, &c, &graph).unwrap();
    let p2 = plan_memory_correction("op-replay", 13, &c, &graph).unwrap();
    assert_eq!(p1, p1_again);
    assert_ne!(p1.commitment, p2.commitment);
}

#[test]
fn snapshot_and_plan_tampering_fail_validation() {
    let graph = snapshot(vec![artifact(
        "derived-1",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        true,
        &["source-old"],
        false,
        false,
    )]);
    let c = correction(CorrectionDurabilityV1::DurableExplicit);
    let mut tampered_graph = graph.clone();
    tampered_graph.policy_version = "policy-v2".into();
    assert_eq!(
        tampered_graph.validate(),
        Err(MemoryCorrectionPropagationErrorV1::SnapshotCommitmentMismatch)
    );

    let mut plan = plan_memory_correction("op-tamper", 14, &c, &graph).unwrap();
    plan.fresh_identity_required_count = 0;
    assert_eq!(
        plan.validate(&c, &graph),
        Err(MemoryCorrectionPropagationErrorV1::PlanMismatch)
    );
}

#[test]
fn independent_basis_requires_complete_dependency_metadata() {
    let result = MemoryArtifactDependencyV1::new(
        "bad",
        "preferred-tone",
        "ordinary",
        MemoryArtifactRoleV1::DerivedMemory,
        MemoryRetentionV1::Durable,
        false,
        vec!["source-a".to_owned()],
        true,
        "derive:v1",
        false,
    );
    assert_eq!(
        result,
        Err(
            MemoryCorrectionPropagationErrorV1::IndependentBasisWithoutCompleteDependencies
        )
    );
}
