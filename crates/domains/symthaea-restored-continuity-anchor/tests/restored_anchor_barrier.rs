// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_continuity_anchor::{
    ContinuityAnchorSnapshot, ContinuityHeadAnchor, advance_anchor_after_durable_store,
    bootstrap_anchor_from_store, recover_with_anchor,
};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_restored_continuity_anchor::SqliteRestoredContinuityAnchorBarrier;
use symthaea_welfare_assurance::memory_identity::episode_content_id;
use symthaea_welfare_assurance::memory_quarantine::{
    EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrow,
    EpisodicQuarantineEscrowPersistence, digest_episodic_quarantine_escrow,
    episodic_instance_target_id,
};
use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;
use symthaea_welfare_assurance::restored_continuity_promotion::{
    RestoredContinuityPromotionBarrier, RestoredContinuityPromotionRequest,
};
use tempfile::tempdir;

const STORE: &str = "symthaea:self:episodic-memory";

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

#[derive(Default)]
struct MockAnchor {
    current: Option<ContinuityAnchorSnapshot>,
    fail_next_cas: bool,
    substitute_after_cas: bool,
}

impl ContinuityHeadAnchor for MockAnchor {
    type Error = io::Error;

    fn load(&self, _store_target_id: &str) -> Result<Option<ContinuityAnchorSnapshot>, Self::Error> {
        Ok(self.current.clone())
    }

    fn compare_and_swap(
        &mut self,
        _store_target_id: &str,
        expected_current: Option<Sha256Digest>,
        next: &ContinuityAnchorSnapshot,
    ) -> Result<String, Self::Error> {
        if self.fail_next_cas {
            self.fail_next_cas = false;
            return Err(io::Error::other("injected anchor CAS failure"));
        }
        let actual = self
            .current
            .as_ref()
            .map(|snapshot| snapshot.commitment().unwrap());
        if actual != expected_current {
            return Err(io::Error::other("stale anchor CAS expectation"));
        }
        let mut stored = next.clone();
        if self.substitute_after_cas {
            self.substitute_after_cas = false;
            stored.continuity_manifest_digest = digest(0xFA);
        }
        self.current = Some(stored);
        Ok(format!("test:continuity-anchor:revision:{}", next.revision))
    }
}

struct Fixture {
    path: std::path::PathBuf,
    anchor: MockAnchor,
    previous: ContinuityAnchorSnapshot,
    intent_head: Sha256Digest,
    restored_head: Sha256Digest,
    first_id: symthaea_memory::episodic_replay::EpisodeInstanceId,
    content_id: symthaea_welfare_assurance::memory_identity::EpisodeContentId,
    target_id: String,
}

fn fixture() -> Fixture {
    let dir = tempdir().unwrap();
    let path = dir.keep().join("continuity.sqlite");

    let source = Episode::new(
        ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
        ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
        0.84,
        42,
    );
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let first_id = memory.store_if_significant_with_id(source.clone()).unwrap();
    let second_id = memory.store_if_significant_with_id(source).unwrap();
    let values = memory.get_top_episode_instances(10);
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
    let content_id = episode_content_id(&first).unwrap();
    assert_eq!(content_id, episode_content_id(&second).unwrap());
    let target_id = episodic_instance_target_id(STORE, first_id).unwrap();

    let mut store = SqliteEpisodicContinuityStore::open(&path).unwrap();
    store
        .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, first.clone(), 100, 1).unwrap())
        .unwrap();
    store
        .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, second, 100, 1).unwrap())
        .unwrap();

    let mut anchor = MockAnchor::default();
    let (bootstrap, _) = bootstrap_anchor_from_store(&store, &mut anchor, STORE, 101).unwrap();
    assert_eq!(bootstrap.revision, 1);

    let escrow = EpisodicQuarantineEscrow {
        schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
        target_id: target_id.clone(),
        instance_id: first_id,
        content_id,
        captured_at_unix_s: 102,
        pre_active_state_digest: digest(8),
        episode: first,
    };
    let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
    let escrow_ref = store
        .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
        .unwrap();

    let mut quarantine = EpisodicQuarantineStateLedger::new();
    quarantine
        .append_quarantined(
            &target_id,
            first_id,
            content_id,
            103,
            escrow_digest,
            &escrow_ref,
        )
        .unwrap();
    store
        .persist_quarantine_ledger(quarantine.events(), quarantine.head_hash())
        .unwrap();

    let intent = EpisodicQuarantineIntentLedger::new();
    let intent_head = intent.head_hash();
    let (quarantined_anchor, _) = advance_anchor_after_durable_store(
        &store,
        &mut anchor,
        &bootstrap,
        intent_head,
        quarantine.head_hash(),
        104,
    )
    .unwrap();
    assert_eq!(quarantined_anchor.revision, 2);
    let anchored = recover_with_anchor(&store, &anchor, STORE).unwrap();
    assert!(anchored
        .recovered
        .activation_plan
        .inactive
        .iter()
        .any(|entry| entry.instance_id == first_id));

    quarantine
        .append_restore_prepared(&target_id, first_id, content_id, 105, "exec:restore:1")
        .unwrap();
    quarantine
        .append_restored(
            &target_id,
            first_id,
            content_id,
            106,
            "exec:restore:1",
            digest(12),
        )
        .unwrap();
    let restored_head = quarantine.head_hash();
    store
        .persist_quarantine_ledger(quarantine.events(), restored_head)
        .unwrap();
    drop(store);

    Fixture {
        path,
        anchor,
        previous: quarantined_anchor,
        intent_head,
        restored_head,
        first_id,
        content_id,
        target_id,
    }
}

fn request(fixture: &Fixture) -> RestoredContinuityPromotionRequest {
    RestoredContinuityPromotionRequest::try_new(
        STORE,
        &fixture.target_id,
        fixture.first_id,
        fixture.content_id,
        "exec:restore:1",
        fixture.restored_head,
    )
    .unwrap()
}

#[test]
fn successful_cas_proves_exact_restored_occurrence_active_under_new_anchor() {
    let mut fixture = fixture();
    let request = request(&fixture);
    let mut barrier = SqliteRestoredContinuityAnchorBarrier::new(
        &fixture.path,
        &mut fixture.anchor,
        fixture.previous.clone(),
        fixture.intent_head,
        107,
    )
    .unwrap();
    let evidence = barrier.commit_restored_continuity(&request).unwrap();
    assert_eq!(evidence.next_anchor_revision(), 3);
    assert_eq!(evidence.restored_quarantine_head(), fixture.restored_head);
    assert_ne!(
        evidence.previous_anchor_commitment(),
        evidence.next_anchor_commitment()
    );

    // Model a crash immediately after CAS and before the welfare executor swaps its candidate heap.
    // A fresh anchored restart must still reconstruct the exact restored occurrence active.
    drop(barrier);
    let store = SqliteEpisodicContinuityStore::open(&fixture.path).unwrap();
    let recovered = recover_with_anchor(&store, &fixture.anchor, STORE).unwrap();
    let active = recovered
        .recovered
        .activation_plan
        .active
        .iter()
        .find(|entry| entry.instance_id == fixture.first_id)
        .unwrap();
    assert_eq!(active.content_id, fixture.content_id);
}

#[test]
fn stale_previous_anchor_is_rejected() {
    let mut fixture = fixture();
    let request = request(&fixture);
    let stale = ContinuityAnchorSnapshot {
        revision: 1,
        previous_anchor_commitment: None,
        quarantine_generation: 0,
        quarantine_head: EpisodicQuarantineStateLedger::new().head_hash(),
        continuity_manifest_digest: digest(77),
        committed_at_unix_s: 100,
        ..fixture.previous.clone()
    };
    let mut barrier = SqliteRestoredContinuityAnchorBarrier::new(
        &fixture.path,
        &mut fixture.anchor,
        stale,
        fixture.intent_head,
        107,
    )
    .unwrap();
    assert!(barrier.commit_restored_continuity(&request).is_err());
}

#[test]
fn remote_cas_failure_does_not_advance_anchor() {
    let mut fixture = fixture();
    let request = request(&fixture);
    let before = fixture.anchor.current.clone().unwrap().commitment().unwrap();
    fixture.anchor.fail_next_cas = true;
    let mut barrier = SqliteRestoredContinuityAnchorBarrier::new(
        &fixture.path,
        &mut fixture.anchor,
        fixture.previous.clone(),
        fixture.intent_head,
        107,
    )
    .unwrap();
    assert!(barrier.commit_restored_continuity(&request).is_err());
    drop(barrier);
    assert_eq!(
        fixture.anchor.current.as_ref().unwrap().commitment().unwrap(),
        before
    );
}

#[test]
fn substituted_post_cas_snapshot_fails_revalidation() {
    let mut fixture = fixture();
    let request = request(&fixture);
    fixture.anchor.substitute_after_cas = true;
    let mut barrier = SqliteRestoredContinuityAnchorBarrier::new(
        &fixture.path,
        &mut fixture.anchor,
        fixture.previous.clone(),
        fixture.intent_head,
        107,
    )
    .unwrap();
    assert!(barrier.commit_restored_continuity(&request).is_err());
}
