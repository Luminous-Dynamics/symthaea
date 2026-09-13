// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::intervention_interlock::WelfareConstraintLevel;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_continuity_anchor::{
    ContinuityAnchorSnapshot, ContinuityHeadAnchor, advance_anchor_after_durable_store,
    bootstrap_anchor_from_store,
};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
use symthaea_psych_bench::moral_patient::ProtectionDisposition;
use symthaea_restore_execution_reconciliation::{
    RestoreExecutionReconciliationError, reconcile_anchored_persisted_restore,
};
use symthaea_welfare_assurance::execution_adapter::ExecutionJournalPersistence;
use symthaea_welfare_assurance::execution_recovery::{
    AutomaticRetryDecision, EXECUTION_JOURNAL_SCHEMA, ExecutionJournalEnvelope,
    InterventionExecutionJournal, PreparedInterventionExecution,
};
use symthaea_welfare_assurance::memory_identity::{EpisodeContentId, episode_content_id};
use symthaea_welfare_assurance::memory_quarantine::episodic_instance_target_id;
use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;

const STORE: &str = "symthaea:self:episodic-memory";
const DOMAIN_EXECUTION: &str = "exec:restore:forensic:domain";
const SUBSTITUTE_EXECUTION: &str = "exec:restore:forensic:substitute";

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

#[derive(Default)]
struct MemoryAnchor {
    snapshot: Option<ContinuityAnchorSnapshot>,
}

impl ContinuityHeadAnchor for MemoryAnchor {
    type Error = io::Error;

    fn load(&self, _store_target_id: &str) -> Result<Option<ContinuityAnchorSnapshot>, Self::Error> {
        Ok(self.snapshot.clone())
    }

    fn compare_and_swap(
        &mut self,
        _store_target_id: &str,
        expected_current: Option<Sha256Digest>,
        next: &ContinuityAnchorSnapshot,
    ) -> Result<String, Self::Error> {
        let actual = self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.commitment())
            .transpose()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if actual != expected_current {
            return Err(io::Error::other("stale anchor writer"));
        }
        self.snapshot = Some(next.clone());
        Ok(format!("memory-anchor:revision:{}", next.revision))
    }
}

#[derive(Default)]
struct JournalPersistence {
    calls: usize,
}

impl ExecutionJournalPersistence for JournalPersistence {
    type Error = io::Error;

    fn persist_execution_journal(
        &mut self,
        _events: &[ExecutionJournalEnvelope],
        _head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        self.calls += 1;
        Ok(format!("test:execution-journal:{}", self.calls))
    }
}

struct Fixture {
    store: SqliteEpisodicContinuityStore,
    anchor: MemoryAnchor,
    instance_id: symthaea_memory::episodic_replay::EpisodeInstanceId,
    content_id: EpisodeContentId,
    target_id: String,
}

fn episode() -> Episode {
    Episode::new(
        ContinuousHV::from_values(vec![0.11, 0.22, 0.33]),
        ContinuousHV::from_values(vec![0.44, 0.55, 0.66]),
        0.84,
        42,
    )
}

fn base_occurrence() -> (
    SqliteEpisodicContinuityStore,
    MemoryAnchor,
    symthaea_memory::episodic_replay::EpisodeInstanceId,
    EpisodeContentId,
    String,
    ContinuityAnchorSnapshot,
) {
    let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
    let instance_id = memory.store_if_significant_with_id(episode()).unwrap();
    let exact = memory
        .get_top_episode_instances(1)
        .into_iter()
        .next()
        .unwrap()
        .1;
    let content_id = episode_content_id(&exact).unwrap();
    let target_id = episodic_instance_target_id(STORE, instance_id).unwrap();

    let envelope = PersistedEpisodicEnvelope::new(STORE, exact, 80, 1).unwrap();
    let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
    store.upsert_occurrence(&envelope).unwrap();

    let mut anchor = MemoryAnchor::default();
    let (bootstrap, _) = bootstrap_anchor_from_store(&store, &mut anchor, STORE, 90).unwrap();
    (store, anchor, instance_id, content_id, target_id, bootstrap)
}

fn active_without_restore_history_fixture() -> Fixture {
    let (store, anchor, instance_id, content_id, target_id, _) = base_occurrence();
    Fixture {
        store,
        anchor,
        instance_id,
        content_id,
        target_id,
    }
}

fn restored_fixture() -> Fixture {
    let (mut store, mut anchor, instance_id, content_id, target_id, bootstrap) = base_occurrence();

    let mut quarantine = EpisodicQuarantineStateLedger::new();
    quarantine
        .append_quarantined(
            &target_id,
            instance_id,
            content_id,
            100,
            digest(70),
            format!("escrow:{instance_id}"),
        )
        .unwrap();
    quarantine
        .append_restore_prepared(
            &target_id,
            instance_id,
            content_id,
            110,
            DOMAIN_EXECUTION,
        )
        .unwrap();
    quarantine
        .append_restored(
            &target_id,
            instance_id,
            content_id,
            120,
            DOMAIN_EXECUTION,
            digest(71),
        )
        .unwrap();
    store
        .persist_quarantine_ledger(quarantine.events(), quarantine.head_hash())
        .unwrap();

    let intent = EpisodicQuarantineIntentLedger::new();
    advance_anchor_after_durable_store(
        &store,
        &mut anchor,
        &bootstrap,
        intent.head_hash(),
        quarantine.head_hash(),
        130,
    )
    .unwrap();

    Fixture {
        store,
        anchor,
        instance_id,
        content_id,
        target_id,
    }
}

fn prepared(
    execution_id: &str,
    target_id: String,
    prepared_at_unix_s: u64,
) -> PreparedInterventionExecution {
    PreparedInterventionExecution {
        schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
        execution_id: execution_id.into(),
        authority_id: "authority:restore:forensic".into(),
        target_id,
        action: SubjectAffectingAction::MemoryModification,
        rationale_digest: digest(1),
        welfare_profile_digest: digest(2),
        precaution_policy_digest: digest(3),
        protection_disposition: ProtectionDisposition::Baseline,
        welfare_constraint: WelfareConstraintLevel::Baseline,
        replay_generation: 1,
        replay_snapshot_digest: digest(4),
        replay_persistence_ref: "replay:forensic:1".into(),
        prepared_at_unix_s,
        permit_not_after_unix_s: 300,
    }
}

fn recovered_prepared_journal(
    execution_id: &str,
    target_id: String,
    prepared_at_unix_s: u64,
) -> InterventionExecutionJournal {
    let mut journal = InterventionExecutionJournal::new();
    journal
        .append_prepared(prepared(execution_id, target_id, prepared_at_unix_s))
        .unwrap();
    InterventionExecutionJournal::from_events(journal.events().to_vec()).unwrap()
}

#[test]
fn active_now_without_matching_restored_history_cannot_close_prepared_execution() {
    let fixture = active_without_restore_history_fixture();
    let mut journal = recovered_prepared_journal(DOMAIN_EXECUTION, fixture.target_id.clone(), 105);
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore(
        DOMAIN_EXECUTION,
        STORE,
        fixture.instance_id,
        fixture.content_id,
        140,
        &mut journal,
        &mut persistence,
        &fixture.store,
        &fixture.anchor,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        RestoreExecutionReconciliationError::MissingRestoredEvidence
    ));
    assert_eq!(persistence.calls, 0);
    assert_eq!(
        journal.automatic_retry_decision(DOMAIN_EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
}

#[test]
fn substituted_execution_id_cannot_reuse_some_other_restored_history() {
    let fixture = restored_fixture();
    let mut journal = recovered_prepared_journal(
        SUBSTITUTE_EXECUTION,
        fixture.target_id.clone(),
        105,
    );
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore(
        SUBSTITUTE_EXECUTION,
        STORE,
        fixture.instance_id,
        fixture.content_id,
        140,
        &mut journal,
        &mut persistence,
        &fixture.store,
        &fixture.anchor,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        RestoreExecutionReconciliationError::MissingRestoredEvidence
    ));
    assert_eq!(persistence.calls, 0);
    assert_eq!(
        journal.automatic_retry_decision(SUBSTITUTE_EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
}

#[test]
fn generic_prepared_after_domain_restored_transition_is_rejected() {
    let fixture = restored_fixture();
    let mut journal = recovered_prepared_journal(DOMAIN_EXECUTION, fixture.target_id.clone(), 125);
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore(
        DOMAIN_EXECUTION,
        STORE,
        fixture.instance_id,
        fixture.content_id,
        140,
        &mut journal,
        &mut persistence,
        &fixture.store,
        &fixture.anchor,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        RestoreExecutionReconciliationError::RestoredTimeOrderInvalid
    ));
    assert_eq!(persistence.calls, 0);
    assert_eq!(
        journal.automatic_retry_decision(DOMAIN_EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
}
