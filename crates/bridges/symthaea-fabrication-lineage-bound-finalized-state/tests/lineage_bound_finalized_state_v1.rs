// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn terminal_sequence_and_generation_are_strictly_adjacent() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_finalization_sequence()\n        .checked_add(1)"));
    assert!(source.contains("state_binding\n        .upgrade_state_generation()\n        .checked_add(1)"));
    assert!(source.contains("predecessor_finalization_sequence.checked_add(1) != Some(record.finalization_sequence)"));
    assert!(source.contains("predecessor_upgrade_state_generation.checked_add(1)"));
}

#[test]
fn complete_successor_endpoint_is_canonical_terminal_evidence() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub successor_endpoint: UpgradeEndpoint"));
    assert!(source.contains("digest_upgrade_endpoint(&handoff.plan().successor)"));
    assert!(source.contains("successor_endpoint: handoff.plan().successor.clone()"));
    assert!(source.contains("digest_upgrade_endpoint(&record.successor_endpoint)"));
    assert!(source.contains("successor_digest != record.successor_endpoint_digest"));
}

#[test]
fn predecessor_endpoint_and_global_predecessor_provenance_are_exact() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("digest_upgrade_endpoint(&handoff.plan().predecessor)"));
    assert!(source.contains("predecessor_endpoint_digest != handoff.predecessor_endpoint_digest()"));
    assert!(source.contains("handoff.predecessor_root_id().as_digest() != context.predecessor_root_digest()"));
    assert!(source.contains("handoff.current_head_id().as_digest() != context.predecessor_current_head_digest()"));
    assert!(source.contains("handoff.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()"));
}

#[test]
fn terminal_record_is_deterministic_and_has_no_caller_timestamp_or_nonce() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("record_digest: record_digest.to_hex()"));
    assert!(source.contains("authorization_id: authorization.id().to_hex()"));
    assert!(source.contains("execution_permit_id: execution_permit.id().to_hex()"));
    assert!(source.contains("state_binding_id: state_binding.id().to_hex()"));
    assert!(source.contains("successor_endpoint_digest: successor_endpoint_digest.to_hex()"));
    assert!(!source.contains("finalized_at_unix_ms"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
    assert!(!source.contains("nonce"));
    assert!(!source.contains("rand::"));
}

#[test]
fn portable_record_is_canonical_but_live_terminal_authority_is_opaque() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundUpgradeFinalizationRecordV1"));
    assert!(source.contains("Serialize, Deserialize"));
    assert!(source.contains("pub struct LineageBoundFinalizedUpgradeV1"));
    assert!(!source.contains("pub struct LineageBoundFinalizedUpgradeV1 {\n    pub"));
    assert!(source.contains("fn canonical_hex_id(value: &str) -> bool"));
    assert!(source.contains("value.len() == 64"));
    assert!(source.contains("record.machine_ids.windows(2).any(|pair| pair[0] >= pair[1])"));
    assert!(source.contains("InvalidRecord(\"machine_order\")"));
}

#[test]
fn terminal_authority_requires_exact_lineage_authorization_execution_state_and_handoff() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("execution_permit.authorization_id() != authorization.id()"));
    assert!(source.contains("execution_permit.context_id() != context.id()"));
    assert!(source.contains("state_binding.authorization_id() != authorization.id()"));
    assert!(source.contains("state_binding.execution_permit_id() != execution_permit.id()"));
    assert!(source.contains("context.lineage_handoff_id() != handoff.id()"));
    assert!(source.contains("state_binding.handoff_plan_digest() != handoff.plan_digest()"));
}

#[test]
fn terminal_stage_is_fixed_and_legacy_authority_does_not_reenter() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("terminal_stage: UpgradeStage::Finalized"));
    assert!(!source.contains("AuthorizedClockGovernedUpgradeFinalizationV1"));
    assert!(!source.contains("ClockGovernedUpgradeFinalizationExecutionPermitV1"));
    assert!(!source.contains("ExecutionBoundUpgradeFinalizationStateV1"));
    assert!(!source.contains("ClockGovernedUpgradeHandoffV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("UpgradeHandoffTracker"));
}

#[test]
fn source_has_no_panic_or_saturating_arithmetic_path() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(".expect("));
    assert!(!source.contains(".unwrap("));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains("saturating_add"));
    assert!(!source.contains("saturating_sub"));
}
