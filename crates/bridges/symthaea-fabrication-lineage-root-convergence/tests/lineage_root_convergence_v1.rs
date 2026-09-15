// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn adapter_is_borrowed_from_opaque_lineage_native_authorities() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("root: &'a LineageBoundFinalizedPredecessorRootV1"));
    assert!(source.contains("head: &'a CurrentLineageBoundFinalizedUpgradeHeadV1"));
    assert!(source.contains("LineageNativePredecessorRootViewV1::new(root)"));
    assert!(source.contains("LineageNativeCurrentHeadViewV1::new(head)"));
}

#[test]
fn both_views_identify_as_lineage_native() {
    let source = include_str!("../src/lib.rs");
    let marker = "LineageBoundPredecessorAuthorityKindV1::LineageNativeV1";
    assert!(source.matches(marker).count() >= 2);
}

#[test]
fn predecessor_projection_contains_no_caller_supplied_authority_facts() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("self.root.id().as_digest()"));
    assert!(source.contains("self.root.current_head_id().as_digest()"));
    assert!(source.contains("self.root.endpoint()"));
    assert!(source.contains("self.root.rollback_target_digest()"));
    assert!(source.contains("self.root.evidence_checkpoint_digest()"));
    assert!(source.contains("self.root.transparency_log_digest()"));
    assert!(!source.contains("endpoint: UpgradeEndpoint"));
    assert!(!source.contains("finalization_sequence: u64"));
}

#[test]
fn current_head_projection_uses_exact_lineage_head_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("self.head.id().as_digest()"));
    assert!(source.contains("self.head.governance_view_id().as_digest()"));
    assert!(source.contains("self.head.current_checkpoint_digest()"));
    assert!(source.contains("self.head.current_transparency_log_digest()"));
    assert!(source.contains("self.head.current_clock_envelope_id()"));
    assert!(source.contains("self.head.current_operational_basis_id()"));
}

#[test]
fn adapter_does_not_mint_or_deserialize_live_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("#![deny(unsafe_code)]"));
    assert!(!source.contains("Serialize"));
    assert!(!source.contains("Deserialize"));
    assert!(!source.contains("Sha256Digest(["));
    assert!(!source.contains("from_hex"));
    assert!(!source.contains("unsafe {"));
}
