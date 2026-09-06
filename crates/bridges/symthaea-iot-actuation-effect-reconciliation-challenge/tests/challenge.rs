use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalHeadV1,
    IndependentEffectAttemptHeadAnchor, RollbackProtectedEffectAttemptJournal,
    RollbackProtectedEffectAttemptJournalError,
};
use symthaea_iot_actuation_effect_dispatch::{
    PhysicalEffectAttemptCorrelation, RollbackProtectedPhysicalEffectAttemptJournal,
};
use symthaea_iot_actuation_effect_reconciliation_challenge::{
    EffectReconciliationChallengeIssueError, ReconciliationSourceStateV1,
    issue_effect_reconciliation_challenge,
};

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-effect-reconciliation-challenge-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[derive(Debug)]
struct TestAnchorError;

impl fmt::Display for TestAnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("test anchor error")
    }
}

impl StdError for TestAnchorError {}

#[derive(Clone)]
struct TestAnchor {
    head: Arc<Mutex<DurableEffectAttemptJournalHeadV1>>,
}

impl TestAnchor {
    fn new(head: DurableEffectAttemptJournalHeadV1) -> Self {
        Self {
            head: Arc::new(Mutex::new(head)),
        }
    }

    fn set_head(&self, head: DurableEffectAttemptJournalHeadV1) {
        *self.head.lock().unwrap() = head;
    }
}

impl IndependentEffectAttemptHeadAnchor for TestAnchor {
    type Error = TestAnchorError;

    fn current_head(&mut self) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        Ok(*self.head.lock().unwrap())
    }

    fn compare_and_swap(
        &mut self,
        expected: DurableEffectAttemptJournalHeadV1,
        next: DurableEffectAttemptJournalHeadV1,
    ) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        let mut head = self.head.lock().unwrap();
        if *head != expected || next.generation() != expected.generation().saturating_add(1) {
            return Err(TestAnchorError);
        }
        *head = next;
        Ok(next)
    }
}

fn genesis(device: &ResourceRef) -> DurableEffectAttemptJournalHeadV1 {
    DurableEffectAttemptJournalCheckpointV1::genesis(device)
        .unwrap()
        .head()
        .unwrap()
}

#[test]
fn prepared_attempt_issues_fresh_nonce_bound_challenges_for_exact_protected_head() {
    let root = temp_root("prepared");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);
    journal.persist_prepared_anchored(&correlation).unwrap();

    let first = issue_effect_reconciliation_challenge(&mut journal).unwrap();
    let second = issue_effect_reconciliation_challenge(&mut journal).unwrap();
    assert_eq!(first.journal_generation(), 1);
    assert_eq!(first.journal_digest(), observer.current_head().unwrap().digest());
    assert_eq!(first.command_digest(), Digest32([0x11; 32]));
    assert_eq!(first.envelope_digest(), Digest32([0x22; 32]));
    assert_eq!(first.composition_digest(), Digest32([0x33; 32]));
    assert_eq!(first.device(), &device);
    assert_eq!(first.sequence(), 1);
    assert_eq!(first.source_state(), ReconciliationSourceStateV1::Prepared);
    assert_ne!(first.nonce(), [0; 32]);
    assert_ne!(first.nonce(), second.nonce());
    assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    assert!(first.is_fresh_at(first.issued_at_unix_ms()));
    assert!(!first.canonical_bytes().unwrap().is_empty());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn acknowledged_attempt_challenge_retains_exact_adapter_evidence_without_claiming_realization() {
    let root = temp_root("acknowledged");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);
    let prepared = journal.persist_prepared_anchored(&correlation).unwrap();
    journal
        .persist_adapter_acknowledged_anchored(&prepared, Digest32([0xE7; 32]))
        .unwrap();

    let challenge = issue_effect_reconciliation_challenge(&mut journal).unwrap();
    assert!(matches!(
        challenge.source_state(),
        ReconciliationSourceStateV1::AdapterAcknowledged {
            adapter_evidence_digest
        } if adapter_evidence_digest == Digest32([0xE7; 32])
    ));
    assert_eq!(challenge.journal_generation(), 2);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn abandoned_before_port_state_cannot_issue_reconciliation_challenge() {
    let root = temp_root("abandoned");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);
    let prepared = journal.persist_prepared_anchored(&correlation).unwrap();
    journal
        .persist_abandoned_before_port_anchored(&prepared)
        .unwrap();

    let result = issue_effect_reconciliation_challenge(&mut journal);
    assert!(matches!(
        result,
        Err(EffectReconciliationChallengeIssueError::NoUnresolvedAttempt)
    ));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn moved_independent_anchor_prevents_challenge_issuance() {
    let root = temp_root("anchor-moved");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);
    journal.persist_prepared_anchored(&correlation).unwrap();
    observer.set_head(
        DurableEffectAttemptJournalHeadV1::from_anchor_parts(2, Digest32([0x99; 32])).unwrap(),
    );

    let result = issue_effect_reconciliation_challenge(&mut journal);
    assert!(matches!(
        result,
        Err(EffectReconciliationChallengeIssueError::Journal(
            RollbackProtectedEffectAttemptJournalError::AnchorMoved { .. }
        ))
    ));

    fs::remove_dir_all(root).unwrap();
}
