// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::quarantine_intent_ledger::{
    EpisodicQuarantineIntentLedger, QuarantineIntentLedgerError,
};

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

#[test]
fn pending_quarantine_intent_survives_recovery_and_is_uuid_exact() {
    let duplicate = Episode::new(
        ContinuousHV::from_values(vec![1.0, 2.0]),
        ContinuousHV::from_values(vec![3.0, 4.0]),
        0.9,
        42,
    );
    let content_id = episode_content_id(&duplicate).unwrap();
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let first = memory.store_if_significant_with_id(duplicate.clone()).unwrap();
    let second = memory.store_if_significant_with_id(duplicate).unwrap();
    assert_ne!(first, second);

    let target = format!("symthaea:self:episodic-memory:instance:{first}");
    let mut ledger = EpisodicQuarantineIntentLedger::new();
    ledger
        .append_prepared(
            "exec:quarantine:pending",
            &target,
            first,
            content_id,
            100,
            digest(1),
            digest(2),
            "escrow:pending:1",
        )
        .unwrap();

    let trusted_head = ledger.head_hash();
    let recovered =
        EpisodicQuarantineIntentLedger::recover_anchored(ledger.events(), trusted_head).unwrap();
    assert!(recovered.pending_state(first).is_some());
    assert!(recovered.pending_state(second).is_none());
}

#[test]
fn committed_intent_cannot_be_rolled_back_to_prepared_under_later_anchor() {
    let episode = Episode::new(
        ContinuousHV::from_values(vec![5.0, 6.0]),
        ContinuousHV::from_values(vec![7.0, 8.0]),
        0.9,
        43,
    );
    let content_id = episode_content_id(&episode).unwrap();
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let id = memory.store_if_significant_with_id(episode).unwrap();
    let target = format!("symthaea:self:episodic-memory:instance:{id}");

    let mut ledger = EpisodicQuarantineIntentLedger::new();
    ledger
        .append_prepared(
            "exec:quarantine:commit",
            &target,
            id,
            content_id,
            100,
            digest(3),
            digest(4),
            "escrow:commit:1",
        )
        .unwrap();
    let prepared_only = vec![ledger.events()[0].clone()];
    ledger
        .append_committed(
            "exec:quarantine:commit",
            &target,
            id,
            content_id,
            101,
            digest(5),
        )
        .unwrap();
    let committed_head = ledger.head_hash();

    assert!(matches!(
        EpisodicQuarantineIntentLedger::recover_anchored(&prepared_only, committed_head),
        Err(QuarantineIntentLedgerError::HeadAnchorMismatch { .. })
    ));
}
