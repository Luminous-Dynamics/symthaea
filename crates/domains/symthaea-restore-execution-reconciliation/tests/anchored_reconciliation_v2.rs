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
use symthaea_restore_execution_reconciliation::v2::{
    PreparedDigestBoundRestoreReconciliationOutcome, RestoreExecutionReconciliationV2Error,
    reconcile_anchored_persisted_restore_v2,
};
use symthaea_welfare_assurance::execution_adapter::ExecutionJournalPersistence;
use symthaea_welfare_assurance::execution_recovery::{
    AutomaticRetryDecision, EXECUTION_JOURNAL_SCHEMA, ExecutionJournalEnvelope,
    InterventionExecutionJournal, PreparedInterventionExecution,
};
use symthaea_welfare_assurance::memory_identity::{EpisodeContentId, episode_content_id};
use symthaea_welfare_assurance::memory_quarantine::episodic_instance_target_id;
use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;
use symthaea_welfare_assurance::persisted_restore_correlation_v2::
    digest_persisted_restore_correlation_from_prepared_v2;
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;

const STORE: &str = "symthaea:self:episodic-memory";
const EXECUTION: &str = "exec:restore:reconcile:v2:e2e";

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
    snapshots: Vec<Vec<ExecutionJournalEnvelope>>,
}

impl ExecutionJournalPersistence for JournalPersistence {
    type Error = io::Error;

    fn persist_execution_journal(
        &mut self,
        events: &[ExecutionJournalEnvelope],
        _head_hash: Sha256Digest,
    ) -> Result<String, Self::Error> {
        self.calls += 1;
        self.snapshots.push(events.to_vec());
        Ok(format!("test:execution-journal:v2:{}", self.calls))
    }
}

struct Fixture {
    store: SqliteEpisodicContinuityStore,
    anchor: MemoryAnchor,
    instance_id: symthaea_memory::episodic_replay::EpisodeInstanceId,
    content_id: EpisodeContentId,
    target_id: String,
    actual_prepared: PreparedInterventionExecution,
    actual_prepared_event_hash: Sha256Digest,
    correlation_digest: Sha256Digest,
}

fn episode() -> Episode {
    Episode::new(
        ContinuousHV::from_values(vec![0.17, 0.29, 0.41]),
        ContinuousHV::from_values(vec![0.83, 0.71, 0.59]),
        0.86,
        42,
    )
}

fn prepared(authority_id: &str, target_id: String) -> PreparedInterventionExecution {
    PreparedInterventionExecution {
        schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
        execution_id: EXECUTION.into(),
        authority_id: authority_id.into(),
        target_id,
        action: SubjectAffectingAction::MemoryModification,
        rationale_digest: digest(1),
        welfare_profile_digest: digest(2),
        precaution_policy_digest: digest(3),
        protection_disposition: ProtectionDisposition::Baseline,
        welfare_constraint: WelfareConstraintLevel::Baseline,
        replay_generation: 1,
        replay_snapshot_digest: digest(4),
        replay_persistence_ref: "replay:restore-reconciliation:v2:e2e".into(),
        prepared_at_unix_s: 105,
        permit_not_after_unix_s: 300,
    }
}

fn prepared_event_hash(prepared: &PreparedInterventionExecution) -> Sha256Digest {
    let mut journal = InterventionExecutionJournal::new();
    journal.append_prepared(prepared.clone()).unwrap();
    journal.head_hash()
}

fn fixture() -> Fixture {
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
    let actual_prepared = prepared("authority:restore:v2:actual", target_id.clone());
    let actual_prepared_event_hash = prepared_event_hash(&actual_prepared);

    let envelope = PersistedEpisodicEnvelope::new(STORE, exact, 80, 1).unwrap();
    let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
    store.upsert_occurrence(&envelope).unwrap();

    let mut anchor = MemoryAnchor::default();
    let (bootstrap, _) = bootstrap_anchor_from_store(&store, &mut anchor, STORE, 90).unwrap();

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
    let restore_prepared_event_hash = quarantine
        .append_restore_prepared(&target_id, instance_id, content_id, 110, EXECUTION)
        .unwrap();
    let correlation_digest = digest_persisted_restore_correlation_from_prepared_v2(
        &actual_prepared,
        actual_prepared_event_hash,
        instance_id,
        content_id,
        120,
        restore_prepared_event_hash,
    )
    .unwrap();
    quarantine
        .append_restored(
            &target_id,
            instance_id,
            content_id,
            120,
            EXECUTION,
            correlation_digest,
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
        actual_prepared,
        actual_prepared_event_hash,
        correlation_digest,
    }
}

fn recovered_prepared_journal(prepared: PreparedInterventionExecution) -> InterventionExecutionJournal {
    let mut journal = InterventionExecutionJournal::new();
    journal.append_prepared(prepared).unwrap();
    InterventionExecutionJournal::from_events(journal.events().to_vec()).unwrap()
}

#[test]
fn anchored_v2_correlation_closes_exact_prepared_event_without_retry() {
    let fixture = fixture();
    let mut journal = recovered_prepared_journal(fixture.actual_prepared.clone());
    assert_eq!(journal.head_hash(), fixture.actual_prepared_event_hash);
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
    let mut persistence = JournalPersistence::default();

    let outcome = reconcile_anchored_persisted_restore_v2(
        EXECUTION,
        STORE,
        fixture.instance_id,
        fixture.content_id,
        140,
        &mut journal,
        &mut persistence,
        &fixture.store,
        &fixture.anchor,
    )
    .unwrap();

    let PreparedDigestBoundRestoreReconciliationOutcome::Completed { evidence, .. } = outcome else {
        panic!("expected strict V2 reconciliation completion");
    };
    assert_eq!(evidence.generic_prepared_sequence, 0);
    assert_eq!(
        evidence.generic_prepared_event_hash,
        fixture.actual_prepared_event_hash
    );
    assert_eq!(
        evidence.expected_restore_correlation_digest,
        fixture.correlation_digest
    );
    assert_eq!(
        evidence.base.restore_result_digest,
        fixture.correlation_digest
    );
    assert_eq!(persistence.calls, 1);
    assert!(journal.recovery_report().in_doubt.is_empty());
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseAlreadyCompleted
    );
}

#[test]
fn substituted_prepared_authority_cannot_close_same_visible_restore_identity() {
    let fixture = fixture();
    let substituted = prepared("authority:restore:v2:substituted", fixture.target_id.clone());
    assert_eq!(substituted.execution_id, fixture.actual_prepared.execution_id);
    assert_eq!(substituted.target_id, fixture.actual_prepared.target_id);
    assert_ne!(substituted.authority_id, fixture.actual_prepared.authority_id);

    let mut journal = recovered_prepared_journal(substituted);
    assert_ne!(journal.head_hash(), fixture.actual_prepared_event_hash);
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore_v2(
        EXECUTION,
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
        RestoreExecutionReconciliationV2Error::PreparedCorrelationMismatch { .. }
    ));
    assert_eq!(persistence.calls, 0);
    assert_eq!(journal.recovery_report().completed, 0);
    assert_eq!(journal.recovery_report().in_doubt.len(), 1);
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
}
