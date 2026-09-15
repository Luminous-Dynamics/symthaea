// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lockfile-neutral cross-crate contract ratchets for one certified lineage upgrade cycle.
//!
//! This test deliberately does not add a dependency from the handoff crate back to the upper
//! certificate-mint/finalization stack. Instead it binds the exact public source contracts across
//! that seam while preserving the acyclic production package graph and existing Cargo.lock.

#[test]
fn opaque_global_root_is_the_only_source_of_portable_certificate_facts() {
    let mint = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate-mint/src/lib.rs"
    );

    assert!(mint.contains("pub fn build_lineage_finalized_predecessor_certificate_v1"));
    assert!(mint.contains("root: &LineageBoundFinalizedPredecessorRootV1"));
    assert!(mint.contains("current_head: &CurrentLineageBoundFinalizedUpgradeHeadV1"));
    assert!(mint.contains("governance_view: &ContainmentCurrentWitnessRegistryHeadV1"));
    assert!(mint.contains("registry_head: &QuorumObservedWitnessRegistryHeadV1"));
    assert!(mint.contains(
        ") -> Result<LineageFinalizedPredecessorCertificateV1, LineagePredecessorCertificateMintError>"
    ));
    assert!(mint.contains("root.endpoint().clone()"));
    assert!(mint.contains("root.finalization_sequence()"));
    assert!(mint.contains("root.evidence_checkpoint_digest()"));
    assert!(mint.contains("root.transparency_log_digest()"));
    assert!(mint.contains("digest_lineage_finalized_predecessor_certificate_v1(&certificate)"));

    assert!(!mint.contains("endpoint: UpgradeEndpoint"));
    assert!(!mint.contains("finalization_sequence: u64"));
    assert!(!mint.contains("checkpoint_digest: Sha256Digest"));
}

#[test]
fn minted_certificate_type_is_the_exact_type_consumed_by_certified_handoff() {
    let mint = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate-mint/src/lib.rs"
    );
    let handoff = include_str!("../src/lib.rs");

    assert!(mint.contains("LineageFinalizedPredecessorCertificateV1"));
    assert!(handoff.contains("pub fn build_certified_lineage_upgrade_handoff_plan_v1"));
    assert!(handoff.contains("certificate: &LineageFinalizedPredecessorCertificateV1"));
    assert!(handoff.contains("pub fn prepare_certified_lineage_upgrade_handoff_v1"));
    assert!(handoff.contains(
        "verify_lineage_finalized_predecessor_certificate_v1(certificate, certificate_ceremony, &inner_prepared)"
    ));
}

#[test]
fn certificate_notarization_and_actual_handoff_authorization_are_distinct_quorums() {
    let certificate = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate/src/lib.rs"
    );
    let handoff = include_str!("../src/lib.rs");

    assert!(certificate.contains("lineage-finalized-predecessor-certificate-v1"));
    assert!(certificate.contains(
        "certificate_ceremony.purpose() != LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_PURPOSE"
    ));
    assert!(handoff.contains("certificate_ceremony: &ClockGovernedThresholdCeremonyV1"));
    assert!(handoff.contains(
        "prepared: PreparedLineageBoundUpgradeHandoffV1, ceremony: &ClockGovernedThresholdCeremonyV1"
    ));
    assert!(handoff.contains("predecessor_certificate_ceremony_id"));
    assert!(handoff.contains("threshold_ceremony_id: ceremony.id()"));
}

#[test]
fn certified_path_converges_on_the_existing_opaque_downstream_waist() {
    let handoff = include_str!("../src/lib.rs");

    assert!(handoff.contains("LineageBoundPredecessorAuthorityKindV1::CertifiedLineageV1"));
    assert!(handoff.contains("pub struct LineageBoundClockGovernedUpgradeHandoffV1"));
    assert!(handoff.contains(
        "Result<LineageBoundClockGovernedUpgradeHandoffV1, LineageBoundUpgradeHandoffError>"
    ));
    assert!(handoff.contains("predecessor_certificate_digest"));
    assert!(handoff.contains("verified_predecessor_certificate_id"));
    assert!(handoff.contains("predecessor_certificate_ceremony_id"));
    assert!(!handoff.contains("pub fn inner_handoff(&self)"));
}

#[test]
fn certified_predecessor_provenance_survives_runtime_activation() {
    let runtime = include_str!(
        "../../symthaea-fabrication-lineage-bound-upgrade-runtime/src/lib.rs"
    );

    for field in [
        "predecessor_root_digest: Sha256Digest",
        "current_head_digest: Sha256Digest",
        "governance_view_digest: Sha256Digest",
        "registry_head_digest: Sha256Digest",
        "predecessor_finalization_sequence: u64",
    ] {
        assert!(runtime.contains(field), "runtime lost provenance field: {field}");
    }
    assert!(runtime.contains(
        "predecessor_root_digest: handoff.predecessor_root_id().as_digest()"
    ));
    assert!(runtime.contains(
        "current_head_digest: handoff.current_head_id().as_digest()"
    ));
    assert!(runtime.contains(
        "predecessor_finalization_sequence: handoff.predecessor_finalization_sequence()"
    ));
}

#[test]
fn rollback_free_operational_binding_rechecks_the_same_predecessor_lineage() {
    let no_rollback = include_str!(
        "../../symthaea-fabrication-lineage-bound-current-no-rollback/src/lib.rs"
    );

    assert!(no_rollback.contains(
        "probation.predecessor_root_digest() != handoff.predecessor_root_id().as_digest()"
    ));
    assert!(no_rollback.contains(
        "activation.predecessor_root_digest() != probation.predecessor_root_digest()"
    ));
    assert!(no_rollback.contains(
        "telemetry.predecessor_root_digest() != probation.predecessor_root_digest()"
    ));
    assert!(no_rollback.contains(
        "authority.predecessor_root_digest() != probation.predecessor_root_digest()"
    ));
    assert!(no_rollback.contains(
        "authority.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()"
    ));
}

#[test]
fn finalization_waist_requires_all_live_authorities_to_match_the_same_predecessor() {
    let context = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalization-context/src/lib.rs"
    );

    assert!(context.contains(
        "let predecessor_root_digest = handoff.predecessor_root_id().as_digest();"
    ));
    assert!(context.contains(
        "activation.predecessor_root_digest() == predecessor_root_digest"
    ));
    assert!(context.contains(
        "probation.predecessor_root_digest() == predecessor_root_digest"
    ));
    assert!(context.contains(
        "telemetry.predecessor_root_digest() == predecessor_root_digest"
    ));
    assert!(context.contains(
        "evidence_binding.predecessor_root_digest() == predecessor_root_digest"
    ));
    assert!(context.contains(
        "no_rollback.predecessor_root_digest() == predecessor_root_digest"
    ));
    assert!(context.contains(
        "authority.predecessor_root_digest() != predecessor_root_digest"
    ));
}

#[test]
fn production_dependency_direction_preserves_the_crate_dag() {
    let handoff_manifest = include_str!("../Cargo.toml");
    let mint_manifest = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate-mint/Cargo.toml"
    );
    let current_no_rollback_manifest = include_str!(
        "../../symthaea-fabrication-lineage-bound-current-no-rollback/Cargo.toml"
    );

    // The upper finalized lineage already descends through current-no-rollback into this handoff.
    assert!(current_no_rollback_manifest.contains(
        "symthaea-fabrication-lineage-bound-upgrade-handoff"
    ));

    // Therefore the lower handoff must never depend upward on the certificate mint.
    assert!(!handoff_manifest.contains(
        "symthaea-fabrication-lineage-predecessor-certificate-mint"
    ));
    assert!(!mint_manifest.contains(
        "symthaea-fabrication-lineage-bound-upgrade-handoff"
    ));

    assert!(mint_manifest.contains("symthaea-fabrication-lineage-bound-finalized-head"));
    assert!(mint_manifest.contains("symthaea-fabrication-lineage-bound-global-upgrade-head"));
}

#[test]
fn cross_crate_cycle_contract_excludes_legacy_current_time_and_forgeable_adapters() {
    let mint = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate-mint/src/lib.rs"
    );
    let handoff = include_str!("../src/lib.rs");

    for source in [mint, handoff] {
        assert!(!source.contains("now_unix_s"));
        assert!(!source.contains("VerifiedClockWindow"));
        assert!(!source.contains("saturating_mul"));
        assert!(!source.contains("AuthorizedUpgradeHandoff"));
        assert!(!source.contains("pub trait Predecessor"));
        assert!(!source.contains("pub fn new_predecessor_root_ref"));
    }
}
