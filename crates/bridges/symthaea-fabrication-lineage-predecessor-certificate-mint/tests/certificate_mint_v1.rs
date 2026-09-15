// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn certificate_root_facts_come_only_from_opaque_global_root() {
    let source = include_str!("../src/lib.rs");
    for needle in [
        "root.id().to_hex()",
        "root.global_head_id().to_hex()",
        "root.current_head_id().to_hex()",
        "root.finalized_upgrade_id().to_hex()",
        "root.record_digest()",
        "root.prior_predecessor_root_digest()",
        "root.finalization_sequence()",
        "root.endpoint().clone()",
        "root.endpoint_digest()",
        "root.rollback_target_digest()",
        "root.evidence_checkpoint_digest()",
        "root.transparency_log_digest()",
        "root.clock_envelope_id().to_hex()",
        "root.operational_basis_id().to_hex()",
    ] {
        assert!(source.contains(needle), "missing root-derived fact: {needle}");
    }
}

#[test]
fn mint_requires_exact_current_head_governance_and_registry_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("root.current_head_id() != current_head.id()"));
    assert!(source.contains("current_head.governance_view_id() != governance_view.id()"));
    assert!(source.contains("governance_view.registry_head_id() != registry_head.id()"));
    assert!(source.contains("root.evidence_checkpoint_digest() != governance_view.checkpoint_digest()"));
    assert!(source.contains("root.transparency_log_digest() != governance_view.transparency_log_digest()"));
    assert!(source.contains("root.clock_envelope_id() != governance_view.observation_clock_envelope_id()"));
}

#[test]
fn governance_facts_are_copied_from_opaque_witnessed_capabilities() {
    let source = include_str!("../src/lib.rs");
    for needle in [
        "governance_view.id().to_hex()",
        "registry_head.id().to_hex()",
        "registry_head.registry_digest()",
        "registry_head.sequence()",
        "registry_head.trust_snapshot_digest()",
        "governance_view.containment_state_digest()",
        "governance_view.compromise_tracker_digest()",
        "governance_view.containment_generation()",
    ] {
        assert!(source.contains(needle), "missing witnessed governance fact: {needle}");
    }
}

#[test]
fn mint_accepts_no_caller_supplied_endpoint_sequence_or_checkpoint() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("root: &LineageBoundFinalizedPredecessorRootV1"));
    assert!(source.contains("current_head: &CurrentLineageBoundFinalizedUpgradeHeadV1"));
    assert!(source.contains("governance_view: &ContainmentCurrentWitnessRegistryHeadV1"));
    assert!(source.contains("registry_head: &QuorumObservedWitnessRegistryHeadV1"));
    assert!(!source.contains("endpoint: UpgradeEndpoint"));
    assert!(!source.contains("finalization_sequence: u64"));
    assert!(!source.contains("checkpoint_digest: Sha256Digest"));
}

#[test]
fn minted_record_is_revalidated_by_low_level_certificate_logic() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("digest_lineage_finalized_predecessor_certificate_v1(&certificate)"));
}
