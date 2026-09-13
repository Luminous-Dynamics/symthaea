// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::HashSet;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_escrow_lookup::SqliteEpisodicEscrowLookup;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_welfare_assurance::canonical_memory_restart::reconstruct_canonical_episodic_memory;
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::memory_intervention::digest_episodic_memory;
use symthaea_welfare_assurance::memory_quarantine::{
    EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrow,
    EpisodicQuarantineEscrowPersistence, digest_episodic_quarantine_escrow,
    episodic_instance_target_id,
};
use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;
use symthaea_welfare_assurance::persisted_memory_restore::EpisodicQuarantineEscrowLookup;
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_intent_persistence::QuarantineIntentLedgerPersistence;
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;

const STORE: &str = "symthaea:self:episodic-memory";

fn config() -> EpisodicReplayConfig {
    EpisodicReplayConfig {
        capacity: 8,
        psi_threshold: 0.0,
        ..EpisodicReplayConfig::default()
    }
}

fn duplicate_pair() -> (EpisodicMemory, Episode, Episode) {
    let mut memory = EpisodicMemory::new(config());
    let source = Episode::new(
        ContinuousHV::from_vec(vec![0.25; 8]),
        ContinuousHV::from_vec(vec![0.75; 8]),
        0.85,
        10,
    );
    let first_id = memory.store_if_significant_with_id(source.clone()).unwrap();
    let second_id = memory.store_if_significant_with_id(source).unwrap();
    assert_ne!(first_id, second_id);

    let values = memory.get_top_episode_instances(8);
    let first = values
        .iter()
        .find(|(id, _)| *id == first_id)
        .unwrap()
        .1
        .clone();
    let second = values
        .iter()
        .find(|(id, _)| *id == second_id)
        .unwrap()
        .1
        .clone();
    assert_eq!(episode_content_id(&first).unwrap(), episode_content_id(&second).unwrap());
    (memory, first, second)
}

#[test]
fn kill_restart_exact_lookup_restore_and_second_restart_preserve_occurrence_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("continuity.sqlite");

    // Process 1: two byte-identical occurrences are distinct durable identities.
    let (mut original, a_episode, b_episode) = duplicate_pair();
    let a = a_episode.instance_id.unwrap();
    let b = b_episode.instance_id.unwrap();
    let content_id = episode_content_id(&a_episode).unwrap();
    let target = episodic_instance_target_id(STORE, a).unwrap();

    let (intent_head, quarantine_head) = {
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, a_episode.clone(), 100, 1).unwrap(),
            )
            .unwrap();
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, b_episode.clone(), 100, 1).unwrap(),
            )
            .unwrap();

        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: a,
            content_id,
            captured_at_unix_s: 110,
            pre_active_state_digest: digest_episodic_memory(&original).unwrap(),
            episode: a_episode.clone(),
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let escrow_ref = store
            .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
            .unwrap();

        let mut quarantine = EpisodicQuarantineStateLedger::new();
        quarantine
            .append_quarantined(
                &target,
                a,
                content_id,
                111,
                escrow_digest,
                &escrow_ref,
            )
            .unwrap();
        store
            .persist_quarantine_ledger(quarantine.events(), quarantine.head_hash())
            .unwrap();

        let mut intent = EpisodicQuarantineIntentLedger::new();
        intent
            .append_prepared(
                "exec:q:kill-restart",
                &target,
                a,
                content_id,
                110,
                Sha256Digest([3; 32]),
                escrow_digest,
                &escrow_ref,
            )
            .unwrap();
        intent
            .append_committed(
                "exec:q:kill-restart",
                &target,
                a,
                content_id,
                112,
                quarantine.head_hash(),
            )
            .unwrap();
        store
            .persist_quarantine_intent_ledger(intent.events(), intent.head_hash())
            .unwrap();

        // The live process also isolates A before it dies. Durable restart policy does not rely on
        // this in-memory map surviving.
        original.quarantine_instance(a).unwrap();
        assert_eq!(original.quarantined_len(), 1);
        assert_eq!(original.len(), 1);
        (intent.head_hash(), quarantine.head_hash())
    };

    drop(original); // process death: all live/quarantine heap state is gone.

    // Process 2: recovery materializes B only; A remains absent from all live memory.
    let mut recovered = {
        let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .recover_restart(STORE, intent_head, quarantine_head)
            .unwrap()
    };
    assert_eq!(recovered.activation_plan.active.len(), 1);
    assert_eq!(recovered.activation_plan.active[0].instance_id, b);
    assert_eq!(recovered.activation_plan.inactive.len(), 1);
    assert_eq!(recovered.activation_plan.inactive[0].instance_id, a);

    let mut restarted = reconstruct_canonical_episodic_memory(config(), 20, &recovered.import_batch)
        .unwrap();
    let ids: HashSet<_> = restarted
        .get_top_episode_instances(8)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, HashSet::from([b]));
    assert_eq!(restarted.quarantined_len(), 0);

    // A is recoverable only by its exact UUID through the purpose-separated read-only escrow view.
    let row = {
        let lookup = SqliteEpisodicEscrowLookup::open(&path).unwrap();
        lookup
            .load_episodic_quarantine_escrow(a)
            .unwrap()
            .expect("exact escrow row for A")
    };
    assert_eq!(row.escrow.instance_id, a);
    assert_eq!(row.escrow.content_id, content_id);
    let ledger_state = recovered.quarantine_ledger.unresolved_state(a).unwrap();
    assert_eq!(row.stored_digest, ledger_state.escrow_digest);
    assert_eq!(row.persistence_ref, ledger_state.escrow_persistence_ref);

    // This integration test exercises the mechanism after the same exact evidence bindings checked
    // by the governed adapter. Permit/consent execution is separately tested by welfare assurance.
    let restored = restarted
        .restore_validated_persisted_occurrence(row.escrow.episode.clone())
        .unwrap();
    assert_eq!(restored, a);
    let ids: HashSet<_> = restarted
        .get_top_episode_instances(8)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, HashSet::from([a, b]));

    // Commit the same two-phase durable domain state used by the governed persisted-restore
    // executor. Once Restored is durable, the next process must derive A active independently.
    let final_quarantine_head = {
        let quarantine = &mut recovered.quarantine_ledger;
        quarantine
            .append_restore_prepared(
                &target,
                a,
                content_id,
                130,
                "exec:r:kill-restart",
            )
            .unwrap();
        quarantine
            .append_restored(
                &target,
                a,
                content_id,
                131,
                "exec:r:kill-restart",
                Sha256Digest([9; 32]),
            )
            .unwrap();
        let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .persist_quarantine_ledger(quarantine.events(), quarantine.head_hash())
            .unwrap();
        quarantine.head_hash()
    };

    drop(restarted);
    drop(recovered); // second process death.

    // Process 3: durable evidence alone now reconstructs A+B active with the original two UUIDs.
    let recovered_again = {
        let store = SqliteEpisodicContinuityStore::open(&path).unwrap();
        store
            .recover_restart(STORE, intent_head, final_quarantine_head)
            .unwrap()
    };
    assert!(recovered_again.activation_plan.inactive.is_empty());
    let restarted_again = reconstruct_canonical_episodic_memory(
        config(),
        40,
        &recovered_again.import_batch,
    )
    .unwrap();
    let ids: HashSet<_> = restarted_again
        .get_top_episode_instances(8)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, HashSet::from([a, b]));
    assert_eq!(restarted_again.len(), 2);
    assert_eq!(restarted_again.quarantined_len(), 0);
}
