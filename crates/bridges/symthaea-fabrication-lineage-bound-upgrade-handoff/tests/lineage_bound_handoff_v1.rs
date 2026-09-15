// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn plan_derives_predecessor_rollback_and_checkpoint_from_root() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root.endpoint().clone()"));
    assert!(source.contains("predecessor_root.rollback_target_digest()"));
    assert!(source.contains("predecessor_root.evidence_checkpoint_digest()"));
    assert!(!source.contains("predecessor: UpgradeEndpoint"));
    assert!(!source.contains("rollback_target_digest: Sha256Digest"));
}

#[test]
fn preparation_is_locked_to_same_finalized_governance_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root.current_head_id() != current_head.id()"));
    assert!(source.contains("current_head.governance_view_id() != governance_view.id()"));
    assert!(source.contains("governance_view.registry_head_id() != registry_head.id()"));
    assert!(source.contains("trust_snapshot_digest != registry_head.trust_snapshot_digest()"));
    assert!(source.contains("containment_state_digest != governance_view.containment_state_digest()"));
    assert!(source.contains("operational_basis.id() != predecessor_root.operational_basis_id()"));
}

#[test]
fn live_wrapper_does_not_expose_inner_authorized_handoff() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundClockGovernedUpgradeHandoffV1"));
    assert!(source.contains("inner_handoff: ClockGovernedUpgradeHandoffV1"));
    assert!(!source.contains("pub fn inner_handoff(&self)"));
    assert!(source.contains("pub fn inner_handoff_id(&self)"));
}

#[test]
fn legacy_scalar_authority_stays_out_of_the_live_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("prepared_at_unix_ms"));
    assert!(!source.contains("saturating_mul"));
}
