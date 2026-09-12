// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use rusqlite::{Connection, params};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_episodic_continuity::{ContinuityStoreError, SqliteEpisodicContinuityStore};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_intent_persistence::QuarantineIntentLedgerPersistence;
use tempfile::tempdir;

const STORE: &str = "symthaea:self:episodic-memory";

fn stored_episode() -> Episode {
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let source = Episode::new(
        ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
        ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
        0.84,
        42,
    );
    let id = memory
        .store_if_significant_with_id(source)
        .expect("episode should be stored");
    memory
        .get_top_episode_instances(1)
        .into_iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|(_, episode)| episode)
        .expect("stored occurrence should be visible")
}

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

#[test]
fn externally_tampered_occurrence_row_identity_is_rejected_after_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("continuity.sqlite");
    let episode = stored_episode();
    let instance_id = episode.instance_id.unwrap();

    {
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, episode, 100, 1).unwrap(),
            )
            .unwrap();
    }

    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE episodic_occurrences SET instance_id = ?1 WHERE instance_id = ?2",
            params!["00000000-0000-0000-0000-000000000000", instance_id.to_string()],
        )
        .unwrap();
    }

    let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
    assert!(matches!(
        store.load_validated_occurrences(),
        Err(ContinuityStoreError::OccurrenceRowIdentityMismatch { .. })
    ));
}

#[test]
fn externally_tampered_episode_bytes_are_rejected_after_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("continuity.sqlite");
    let episode = stored_episode();
    let record_id = format!("episodic:{}", episode.instance_id.unwrap());

    {
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, episode, 100, 1).unwrap(),
            )
            .unwrap();
    }

    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE episodic_occurrences SET envelope = ?1 WHERE record_id = ?2",
            params![vec![0x01u8, 0x02, 0x03, 0x04], record_id],
        )
        .unwrap();
    }

    let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
    assert!(matches!(
        store.load_validated_occurrences(),
        Err(ContinuityStoreError::PersistedEnvelope(_))
    ));
}

#[test]
fn sqlite_head_tamper_is_detected_even_when_event_chain_and_external_anchor_agree() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("continuity.sqlite");
    let episode = stored_episode();
    let id = episode.instance_id.unwrap();
    let content_id = episode_content_id(&episode).unwrap();
    let target = format!("{STORE}:instance:{id}");

    let trusted_head = {
        let mut ledger = EpisodicQuarantineIntentLedger::new();
        ledger
            .append_prepared(
                "exec:q:tamper-head",
                target,
                id,
                content_id,
                100,
                digest(3),
                digest(4),
                "escrow:tamper-head",
            )
            .unwrap();
        let head = ledger.head_hash();
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .persist_quarantine_intent_ledger(ledger.events(), head)
            .unwrap();
        head
    };

    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE quarantine_intent_ledger SET head_hash = ?1 WHERE singleton = 1",
            [vec![0xA5u8; 32]],
        )
        .unwrap();
    }

    let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
    assert!(matches!(
        store.recover_intent_ledger(trusted_head),
        Err(ContinuityStoreError::StoredHeadMismatch { .. })
    ));
}

#[test]
fn external_anchor_rejects_locally_self_consistent_older_ledger_after_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("continuity.sqlite");
    let episode = stored_episode();
    let id = episode.instance_id.unwrap();
    let content_id = episode_content_id(&episode).unwrap();
    let target = format!("{STORE}:instance:{id}");

    let mut old = EpisodicQuarantineIntentLedger::new();
    old.append_prepared(
        "exec:q:rollback",
        &target,
        id,
        content_id,
        100,
        digest(5),
        digest(6),
        "escrow:rollback",
    )
    .unwrap();
    let old_head = old.head_hash();

    let mut later = old.clone();
    later
        .append_aborted(
            "exec:q:rollback",
            &target,
            id,
            content_id,
            101,
            digest(7),
        )
        .unwrap();
    let trusted_later_head = later.head_hash();

    {
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        // Simulate rollback of the local database to an earlier but internally valid chain.
        store
            .persist_quarantine_intent_ledger(old.events(), old_head)
            .unwrap();
    }

    let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
    assert!(matches!(
        store.recover_intent_ledger(trusted_later_head),
        Err(ContinuityStoreError::IntentLedger(_))
    ));
}
