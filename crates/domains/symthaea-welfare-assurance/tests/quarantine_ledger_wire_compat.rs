// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::quarantine_state_ledger::QuarantineLedgerEventKind;

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

fn exact_identity() -> (
    symthaea_memory::episodic_replay::EpisodeInstanceId,
    symthaea_welfare_assurance::memory_identity::EpisodeContentId,
) {
    let source = Episode::new(
        ContinuousHV::from_values(vec![0.1, 0.2, 0.3]),
        ContinuousHV::from_values(vec![0.4, 0.5, 0.6]),
        0.82,
        42,
    );
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let instance_id = memory.store_if_significant_with_id(source).unwrap();
    let stored = memory
        .get_top_episode_instances(1)
        .into_iter()
        .next()
        .expect("stored occurrence must be visible")
        .1;
    let content_id = episode_content_id(&stored).unwrap();
    (instance_id, content_id)
}

fn v1_events() -> [QuarantineLedgerEventKind; 4] {
    let (instance_id, content_id) = exact_identity();
    let target_id = format!("symthaea:self:episodic-memory:instance:{instance_id}");
    let execution_id = "exec:restore:wire-compat:v1".to_string();

    [
        QuarantineLedgerEventKind::Quarantined {
            target_id: target_id.clone(),
            instance_id,
            content_id,
            quarantined_at_unix_s: 100,
            escrow_digest: digest(1),
            escrow_persistence_ref: "escrow:wire-compat:v1".into(),
        },
        QuarantineLedgerEventKind::RestorePrepared {
            target_id: target_id.clone(),
            instance_id,
            content_id,
            prepared_at_unix_s: 110,
            execution_id: execution_id.clone(),
        },
        QuarantineLedgerEventKind::Restored {
            target_id: target_id.clone(),
            instance_id,
            content_id,
            restored_at_unix_s: 120,
            execution_id: execution_id.clone(),
            restore_result_digest: digest(2),
        },
        QuarantineLedgerEventKind::RestoreAborted {
            target_id,
            instance_id,
            content_id,
            aborted_at_unix_s: 120,
            execution_id,
            reason_digest: digest(3),
        },
    ]
}

fn bincode_variant_index(event: &QuarantineLedgerEventKind) -> u32 {
    let bytes = bincode::serialize(event).expect("V1 event must serialize");
    let prefix: [u8; 4] = bytes
        .get(..4)
        .expect("bincode enum encoding must contain a u32 variant index")
        .try_into()
        .expect("prefix is exactly four bytes");
    u32::from_le_bytes(prefix)
}

#[test]
fn v1_bincode_variant_indices_are_frozen() {
    // This is a persistence compatibility boundary. New enum variants MUST be appended after the
    // existing four variants. Inserting a variant in the middle would renumber historical bincode
    // discriminants and could reinterpret already-persisted ledger bytes.
    let events = v1_events();
    assert_eq!(bincode_variant_index(&events[0]), 0, "Quarantined moved");
    assert_eq!(bincode_variant_index(&events[1]), 1, "RestorePrepared moved");
    assert_eq!(bincode_variant_index(&events[2]), 2, "Restored moved");
    assert_eq!(bincode_variant_index(&events[3]), 3, "RestoreAborted moved");
}

#[test]
fn v1_variants_roundtrip_without_reinterpretation() {
    for event in v1_events() {
        let bytes = bincode::serialize(&event).expect("V1 event must serialize");
        let recovered: QuarantineLedgerEventKind =
            bincode::deserialize(&bytes).expect("V1 event must deserialize");
        assert_eq!(recovered, event);
    }
}
