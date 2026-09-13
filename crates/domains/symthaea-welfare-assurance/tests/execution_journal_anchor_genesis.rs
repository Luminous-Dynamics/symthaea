// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io;

use symthaea_core::intervention_interlock::WelfareConstraintLevel;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_psych_bench::moral_patient::ProtectionDisposition;
use symthaea_welfare_assurance::execution_journal_anchor::{
    ExecutionJournalAnchorCommitOutcome, ExecutionJournalAnchorProtocolError,
    ExecutionJournalAnchorSnapshot, ExecutionJournalHeadAnchor, advance_execution_journal_anchor,
    bootstrap_execution_journal_anchor, recover_execution_journal_with_anchor,
};
use symthaea_welfare_assurance::execution_recovery::{
    EXECUTION_JOURNAL_SCHEMA, InterventionExecutionJournal, PreparedInterventionExecution,
};

const NAMESPACE: &str = "symthaea:self:welfare-execution-journal:genesis-test";

#[derive(Default)]
struct MemoryAnchor {
    snapshot: Option<ExecutionJournalAnchorSnapshot>,
}

impl ExecutionJournalHeadAnchor for MemoryAnchor {
    type Error = io::Error;

    fn load(
        &self,
        journal_namespace: &str,
    ) -> Result<Option<ExecutionJournalAnchorSnapshot>, Self::Error> {
        Ok(self
            .snapshot
            .as_ref()
            .filter(|snapshot| snapshot.journal_namespace == journal_namespace)
            .cloned())
    }

    fn compare_and_swap(
        &mut self,
        journal_namespace: &str,
        expected_current: Option<Sha256Digest>,
        next: &ExecutionJournalAnchorSnapshot,
    ) -> Result<String, Self::Error> {
        if next.journal_namespace != journal_namespace {
            return Err(io::Error::other("namespace mismatch"));
        }
        let actual = self
            .snapshot
            .as_ref()
            .map(ExecutionJournalAnchorSnapshot::commitment)
            .transpose()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if actual != expected_current {
            return Err(io::Error::other("stale writer"));
        }
        self.snapshot = Some(next.clone());
        Ok(format!("memory-anchor:revision:{}", next.revision))
    }
}

fn committed(
    outcome: ExecutionJournalAnchorCommitOutcome<io::Error>,
) -> ExecutionJournalAnchorSnapshot {
    match outcome {
        ExecutionJournalAnchorCommitOutcome::Committed { snapshot, .. } => snapshot,
        ExecutionJournalAnchorCommitOutcome::InDoubt { .. } => {
            panic!("in-memory anchor should acknowledge successful CAS")
        }
    }
}

fn digest(seed: u8) -> Sha256Digest {
    Sha256Digest([seed; 32])
}

fn prepared() -> PreparedInterventionExecution {
    PreparedInterventionExecution {
        schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
        execution_id: "exec:genesis:first".into(),
        authority_id: "authority:genesis:first".into(),
        target_id: "symthaea:self:episodic-memory:instance:genesis".into(),
        action: SubjectAffectingAction::MemoryModification,
        rationale_digest: digest(1),
        welfare_profile_digest: digest(2),
        precaution_policy_digest: digest(3),
        protection_disposition: ProtectionDisposition::Baseline,
        welfare_constraint: WelfareConstraintLevel::Baseline,
        replay_generation: 1,
        replay_snapshot_digest: digest(4),
        replay_persistence_ref: "replay:genesis:first".into(),
        prepared_at_unix_s: 100,
        permit_not_after_unix_s: 200,
    }
}

#[test]
fn empty_journal_is_the_only_valid_zero_head_genesis_and_first_local_event_requires_anchor_advance() {
    let mut journal = InterventionExecutionJournal::new();
    assert!(journal.events().is_empty());
    assert_eq!(journal.head_hash(), Sha256Digest([0; 32]));

    let mut anchor = MemoryAnchor::default();
    let genesis = committed(
        bootstrap_execution_journal_anchor(&journal, &mut anchor, NAMESPACE, 90).unwrap(),
    );
    assert_eq!(genesis.revision, 1);
    assert_eq!(genesis.event_count, 0);
    assert_eq!(genesis.head_hash, Sha256Digest([0; 32]));

    let recovered =
        recover_execution_journal_with_anchor(Vec::new(), &anchor, NAMESPACE).unwrap();
    assert!(recovered.journal.events().is_empty());
    assert_eq!(recovered.anchor, genesis);

    journal.append_prepared(prepared()).unwrap();
    let unanchored =
        recover_execution_journal_with_anchor(journal.events().to_vec(), &anchor, NAMESPACE)
            .unwrap_err();
    assert!(matches!(
        unanchored,
        ExecutionJournalAnchorProtocolError::EventCountMismatch {
            anchored: 0,
            actual: 1
        }
    ));

    let successor = committed(
        advance_execution_journal_anchor(&journal, &mut anchor, &genesis, 110).unwrap(),
    );
    assert_eq!(successor.revision, 2);
    assert_eq!(successor.event_count, 1);
    assert_eq!(successor.head_hash, journal.head_hash());
    assert_eq!(successor.previous_commitment, genesis.commitment().unwrap());

    let recovered =
        recover_execution_journal_with_anchor(journal.events().to_vec(), &anchor, NAMESPACE)
            .unwrap();
    assert_eq!(recovered.anchor, successor);
    assert_eq!(recovered.journal.head_hash(), journal.head_hash());
}
