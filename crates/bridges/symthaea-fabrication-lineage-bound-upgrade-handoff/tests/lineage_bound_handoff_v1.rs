// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn bootstrap_plan_still_derives_predecessor_rollback_and_checkpoint_from_opaque_root() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root.endpoint().clone()"));
    assert!(source.contains("predecessor_root.rollback_target_digest()"));
    assert!(source.contains("predecessor_root.evidence_checkpoint_digest()"));
    assert!(!source.contains("predecessor: UpgradeEndpoint"));
}

#[test]
fn certified_lineage_plan_derives_all_predecessor_facts_from_certificate() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("build_certified_lineage_upgrade_handoff_plan_v1"));
    assert!(source.contains("certificate.endpoint.clone()"));
    assert!(source.contains("certificate.rollback_target_digest"));
    assert!(source.contains("certificate.evidence_checkpoint_digest"));
    assert!(source.contains("digest_lineage_finalized_predecessor_certificate_v1(certificate)"));
}

#[test]
fn bootstrap_and_certified_paths_have_distinct_domains() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1\\0"));
    assert!(source.contains("symthaea.fabrication.lineage-bound-upgrade-handoff.v1\\0"));
    assert!(source.contains("symthaea.fabrication.prepared-certified-lineage-upgrade-handoff.v1\\0"));
    assert!(source.contains("symthaea.fabrication.certified-lineage-upgrade-handoff.v1\\0"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::CertifiedLineageV1"));
}

#[test]
fn certified_preparation_reuses_exact_current_governance_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("governance_view.id().to_hex() != certificate.governance_view_id"));
    assert!(source.contains("registry_head.id().to_hex() != certificate.registry_head_id"));
    assert!(source.contains("registry_head.registry_digest() != certificate.registry_digest"));
    assert!(source.contains("registry_head.sequence() != certificate.registry_sequence"));
    assert!(source.contains("registry_head.trust_snapshot_digest() != certificate.trust_snapshot_digest"));
    assert!(source.contains("governance_view.checkpoint_digest() != certificate.evidence_checkpoint_digest"));
    assert!(source.contains("governance_view.transparency_log_digest() != certificate.transparency_log_digest"));
    assert!(source.contains("governance_view.containment_state_digest() != certificate.containment_state_digest"));
    assert!(source.contains("governance_view.compromise_tracker_digest() != certificate.compromise_tracker_digest"));
    assert!(source.contains("governance_view.containment_generation() != certificate.containment_generation"));
    assert!(source.contains("governance_view.observation_operational_basis_id().to_hex() != certificate.operational_basis_id"));
    assert!(source.contains("governance_view.observation_clock_envelope_id().to_hex() != certificate.clock_envelope_id"));
    assert!(source.contains("verify_lineage_finalized_predecessor_certificate_v1(certificate, certificate_ceremony, &inner_prepared)"));
}

#[test]
fn certificate_proof_is_committed_separately_from_actual_handoff_quorum() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_certificate_digest"));
    assert!(source.contains("verified_predecessor_certificate_id"));
    assert!(source.contains("predecessor_certificate_ceremony_id"));
    assert!(source.contains("predecessor_certificate_ceremony_digest"));
    assert!(source.contains("threshold_ceremony_id: ceremony.id().to_hex()"));
}

#[test]
fn downstream_waist_uses_non_constructible_digest_reference_ids() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundPredecessorRootRefIdV1(Sha256Digest)"));
    assert!(source.contains("pub struct LineageBoundCurrentHeadRefIdV1(Sha256Digest)"));
    assert!(source.contains("pub fn predecessor_root_id(&self) -> LineageBoundPredecessorRootRefIdV1"));
    assert!(source.contains("pub fn current_head_id(&self) -> LineageBoundCurrentHeadRefIdV1"));
    assert!(!source.contains("pub fn new_predecessor_root_ref"));
    assert!(!source.contains("pub fn new_current_head_ref"));
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
fn legacy_scalar_and_panic_authority_stay_out_of_live_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("prepared_at_unix_ms"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".unwrap("));
    assert!(!source.contains(".expect("));
}
