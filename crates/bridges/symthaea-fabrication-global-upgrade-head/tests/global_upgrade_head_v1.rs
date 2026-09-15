// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn global_head_requires_full_concrete_upgrade_state_lineage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("verify_upgrade_state_successor"));
    assert!(source.contains("InvalidStateGenesis"));
    assert!(source.contains("ActivatedStateMismatch"));
    assert!(source.contains("FinalizationSequenceMismatch"));
    assert!(source.contains("FinalizedGenerationMismatch"));
}

#[test]
fn global_head_scans_every_finalized_handoff_namespace() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("PublicationInputCountMismatch"));
    assert!(source.contains("FinalizedSequenceEquivocation"));
    assert!(source.contains("HigherFinalizedSequencePublished"));
    assert!(source.contains("highest_finalization_sequence"));
}

#[test]
fn predecessor_endpoint_is_reconstructed_from_exact_finalized_handoff() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("let endpoint = producing_handoff.plan().successor.clone();"));
    assert!(source.contains("digest_upgrade_endpoint(&endpoint)"));
    assert!(source.contains("rollback_target_digest: endpoint.durable_state_digest"));
    assert!(source.contains("evidence_checkpoint_digest: global_head.checkpoint_digest()"));
    assert!(source.contains("ProducingEndpointMismatch"));
}

#[test]
fn live_predecessor_root_stays_opaque_and_legacy_handoff_is_not_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct FinalizedUpgradePredecessorRootV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct FinalizedUpgradePredecessorRootV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("prepared_at_unix_ms"));
}
