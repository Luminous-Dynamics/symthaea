// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn plan_derives_predecessor_rollback_and_checkpoint_from_root_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("R: LineageBoundPredecessorRootViewV1 + ?Sized"));
    assert!(source.contains("predecessor_root.endpoint().clone()"));
    assert!(source.contains("predecessor_root.rollback_target_digest()"));
    assert!(source.contains("predecessor_root.evidence_checkpoint_digest()"));
    assert!(!source.contains("predecessor: UpgradeEndpoint"));
    assert!(!source.contains("rollback_target_digest: Sha256Digest"));
}

#[test]
fn bootstrap_and_lineage_native_authority_have_distinct_domains() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => PREPARED_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => AUTHORIZED_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => PREPARED_LINEAGE_NATIVE_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => AUTHORIZED_LINEAGE_NATIVE_DOMAIN"));
    assert!(source.contains("symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1\\0"));
    assert!(source.contains("symthaea.fabrication.lineage-bound-upgrade-handoff.v1\\0"));
}

#[test]
fn root_and_current_head_must_share_one_authority_kind_and_exact_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_head.authority_kind() != kind"));
    assert!(source.contains("PredecessorAuthorityKindMismatch"));
    assert!(source.contains("predecessor_root.current_head_digest() != current_head.id_digest()"));
    assert!(source.contains("current_head.governance_view_digest() != governance_view.id().as_digest()"));
    assert!(source.contains("predecessor_root.transparency_log_digest() != current_head.transparency_log_digest()"));
    assert!(source.contains("predecessor_root.clock_envelope_id() != current_head.clock_envelope_id()"));
}

#[test]
fn bootstrap_types_implement_the_generic_read_only_contract() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("impl LineageBoundPredecessorRootViewV1 for FinalizedUpgradePredecessorRootV1"));
    assert!(source.contains("impl LineageBoundCurrentHeadViewV1 for CurrentFinalizedUpgradeHeadV1"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1"));
}

#[test]
fn live_handoff_preserves_digest_refs_without_exposing_inner_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root_id: LineageBoundPredecessorRootRefIdV1"));
    assert!(source.contains("current_head_id: LineageBoundCurrentHeadRefIdV1"));
    assert!(source.contains("pub fn predecessor_root_digest(&self) -> Sha256Digest"));
    assert!(source.contains("pub fn current_head_digest(&self) -> Sha256Digest"));
    assert!(source.contains("inner_handoff: ClockGovernedUpgradeHandoffV1"));
    assert!(!source.contains("pub fn inner_handoff(&self)"));
    assert!(source.contains("pub fn inner_handoff_id(&self)"));
}

#[test]
fn exact_governance_context_remains_required() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("governance_view.registry_head_id() != registry_head.id()"));
    assert!(source.contains("trust_snapshot_digest != registry_head.trust_snapshot_digest()"));
    assert!(source.contains("containment_state_digest != governance_view.containment_state_digest()"));
    assert!(source.contains("operational_basis.id() != predecessor_root.operational_basis_id()"));
}

#[test]
fn legacy_scalar_authority_stays_out_of_the_live_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("prepared_at_unix_ms"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".unwrap("));
    assert!(!source.contains(".expect("));
}
