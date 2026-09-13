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
    AnchoredRestoreReconciliationOutcome, RestoreExecutionReconciliationError,
    reconcile_anchored_persisted_restore,
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
const EXECUTION: &str = "exec:restore:reconcile:1";

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
    fail: bool,
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
        if self.fail {
            return Err(io::Error::other("injected execution-journal persistence failure"));
        }
        Ok(format!("test:execution-journal:{}", self.calls))
    }
}

struct AnchoredFixture {
    store: SqliteEpisodicContinuityStore,
    anchor: MemoryAnchor,
    instance_id: symthaea_memory::episodic_replay::EpisodeInstanceId,
    content_id: EpisodeContentId,
    target_id: String,
}

fn episode() -> Episode {
    Episode::new(
        ContinuousHV::from_values(vec![0.2, 0.4, 0.6]),
        ContinuousHV::from_values(vec![0.8, 0.1, 0.3]),
        0.81,
        42,
    )
}

fn restored_fixture(advance_anchor: bool) -> AnchoredFixture {
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
        .append_restore_prepared(&target_id, instance_id, content_id, 110, EXECUTION)
        .unwrap();
    quarantine
        .append_restored(
            &target_id,
            instance_id,
            content_id,
            120,
            EXECUTION,
            digest(71),
        )
        .unwrap();
    store
        .persist_quarantine_ledger(quarantine.events(), quarantine.head_hash())
        .unwrap();

    if advance_anchor {
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
    }

    AnchoredFixture {
        store,
        anchor,
        instance_id,
        content_id,
        target_id,
    }
}

fn prepared(target_id: String) -> PreparedInterventionExecution {
    PreparedInterventionExecution {
        schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
        execution_id: EXECUTION.into(),
        authority_id: "authority:restore:1".into(),
        target_id,
        action: SubjectAffectingAction::MemoryModification,
        rationale_digest: digest(1),
        welfare_profile_digest: digest(2),
        precaution_policy_digest: digest(3),
        protection_disposition: ProtectionDisposition::Baseline,
        welfare_constraint: WelfareConstraintLevel::Baseline,
        replay_generation: 1,
        replay_snapshot_digest: digest(4),
        replay_persistence_ref: "replay:1".into(),
        prepared_at_unix_s: 105,
        permit_not_after_unix_s: 300,
    }
}

fn recovered_prepared_journal(target_id: String) -> InterventionExecutionJournal {
    let mut journal = InterventionExecutionJournal::new();
    journal.append_prepared(prepared(target_id)).unwrap();
    InterventionExecutionJournal::from_events(journal.events().to_vec()).unwrap()
}

#[test]
fn anchored_restored_state_closes_prepared_execution_without_retry() {
    let fixture = restored_fixture(true);
    let mut journal = recovered_prepared_journal(fixture.target_id.clone());
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
    let mut persistence = JournalPersistence::default();

    let outcome = reconcile_anchored_persisted_restore(
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

    let AnchoredRestoreReconciliationOutcome::Completed { evidence, .. } = outcome else {
        panic!("expected persisted reconciliation completion");
    };
    assert_eq!(evidence.instance_id, fixture.instance_id);
    assert_eq!(evidence.content_id, fixture.content_id);
    assert_eq!(evidence.anchor_revision, 2);
    assert_eq!(evidence.restored_generation, 3);
    assert_eq!(persistence.calls, 1);
    assert_eq!(journal.recovery_report().completed, 1);
    assert!(journal.recovery_report().in_doubt.is_empty());
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseAlreadyCompleted
    );
}

#[test]
fn stale_external_anchor_rejects_locally_restored_state() {
    let fixture = restored_fixture(false);
    let mut journal = recovered_prepared_journal(fixture.target_id.clone());
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore(
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

    assert!(matches!(error, RestoreExecutionReconciliationError::Anchor(_)));
    assert_eq!(persistence.calls, 0);
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseInDoubt
    );
}

#[test]
fn prepared_target_substitution_is_rejected_before_anchor_reconciliation() {
    let fixture = restored_fixture(true);
    let mut journal = recovered_prepared_journal(format!("{}:wrong", fixture.target_id));
    let mut persistence = JournalPersistence::default();

    let error = reconcile_anchored_persisted_restore(
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
        RestoreExecutionReconciliationError::PreparedTargetMismatch { .. }
    ));
    assert_eq!(persistence.calls, 0);
}

#[test]
fn completion_persistence_failure_never_reopens_retry_in_current_process() {
    let fixture = restored_fixture(true);
    let mut journal = recovered_prepared_journal(fixture.target_id.clone());
    let mut persistence = JournalPersistence {
        fail: true,
        ..Default::default()
    };

    let outcome = reconcile_anchored_persisted_restore(
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

    assert!(matches!(
        outcome,
        AnchoredRestoreReconciliationOutcome::CompletionPersistenceInDoubt { .. }
    ));
    assert_eq!(persistence.calls, 1);
    assert_eq!(
        journal.automatic_retry_decision(EXECUTION),
        AutomaticRetryDecision::RefuseAlreadyCompleted
    );
}

#[test]
fn reconciliation_source_has_no_memory_mutation_entrypoints() {
    let source = include_str!("../src/lib.rs");
    for forbidden in [
        "restore_validated_persisted_occurrence(",
        "store_if_significant",
        "execute_governed_persisted_episodic_restore(",
        "from_validated_persisted_active_state(",
    ] {
        assert!(
            !source.contains(forbidden),
            "reconciliation source must remain evidence-only; found forbidden entrypoint {forbidden:?}"
        );
    }
}
