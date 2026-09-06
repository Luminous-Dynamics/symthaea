use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalHeadV1,
    DurableEffectAttemptStateV1, EffectAttemptJournalError, IndependentEffectAttemptHeadAnchor,
    RollbackProtectedEffectAttemptJournal, RollbackProtectedEffectAttemptJournalError,
};
use symthaea_iot_actuation_effect_dispatch::{
    PhysicalEffectAttemptCorrelation, RollbackProtectedPhysicalEffectAttemptJournal,
};

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-protected-effect-attempt-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnchorMode {
    Normal,
    FailBeforeApply,
    ApplyThenError,
    ApplyThenReportStaleOnce,
    ApplyThenReadErrorOnce,
}

#[derive(Debug)]
struct TestAnchorError;

impl fmt::Display for TestAnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("test anchor failure")
    }
}

impl StdError for TestAnchorError {}

#[derive(Debug)]
struct AnchorState {
    head: DurableEffectAttemptJournalHeadV1,
    mode: AnchorMode,
    stale_read_once: Option<DurableEffectAttemptJournalHeadV1>,
    fail_read_once: bool,
}

#[derive(Clone)]
struct TestAnchor {
    state: Arc<Mutex<AnchorState>>,
}

impl TestAnchor {
    fn new(head: DurableEffectAttemptJournalHeadV1) -> Self {
        Self {
            state: Arc::new(Mutex::new(AnchorState {
                head,
                mode: AnchorMode::Normal,
                stale_read_once: None,
                fail_read_once: false,
            })),
        }
    }

    fn set_mode(&self, mode: AnchorMode) {
        self.state.lock().unwrap().mode = mode;
    }

    fn observed_head(&self) -> DurableEffectAttemptJournalHeadV1 {
        self.state.lock().unwrap().head
    }
}

impl IndependentEffectAttemptHeadAnchor for TestAnchor {
    type Error = TestAnchorError;

    fn current_head(&mut self) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        let mut state = self.state.lock().unwrap();
        if state.fail_read_once {
            state.fail_read_once = false;
            return Err(TestAnchorError);
        }
        if let Some(stale) = state.stale_read_once.take() {
            return Ok(stale);
        }
        Ok(state.head)
    }

    fn compare_and_swap(
        &mut self,
        expected: DurableEffectAttemptJournalHeadV1,
        next: DurableEffectAttemptJournalHeadV1,
    ) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error> {
        let mut state = self.state.lock().unwrap();
        if state.head != expected || next.generation() != expected.generation().saturating_add(1) {
            return Err(TestAnchorError);
        }
        match state.mode {
            AnchorMode::Normal => {
                state.head = next;
                Ok(next)
            }
            AnchorMode::FailBeforeApply => Err(TestAnchorError),
            AnchorMode::ApplyThenError => {
                state.head = next;
                Err(TestAnchorError)
            }
            AnchorMode::ApplyThenReportStaleOnce => {
                state.head = next;
                state.stale_read_once = Some(expected);
                Ok(next)
            }
            AnchorMode::ApplyThenReadErrorOnce => {
                state.head = next;
                state.fail_read_once = true;
                Ok(next)
            }
        }
    }
}

fn genesis(device: &ResourceRef) -> DurableEffectAttemptJournalHeadV1 {
    DurableEffectAttemptJournalCheckpointV1::genesis(device)
        .unwrap()
        .head()
        .unwrap()
}

#[test]
fn anchor_head_round_trips_through_external_parts() {
    let device = ResourceRef("iot:valve:72".into());
    let head = genesis(&device);
    let reconstructed =
        DurableEffectAttemptJournalHeadV1::from_anchor_parts(head.generation(), head.digest())
            .unwrap();
    assert_eq!(reconstructed, head);
}

#[test]
fn protected_prepare_and_acknowledgement_advance_local_and_anchor_together() {
    let root = temp_root("success");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);

    let prepared = journal.persist_prepared_anchored(&correlation).unwrap();
    assert_eq!(journal.anchored_head().generation(), 1);
    assert_eq!(observer.observed_head(), journal.anchored_head());
    let prepared_checkpoint = journal.current_checkpoint().unwrap();
    assert!(matches!(
        prepared_checkpoint.latest(),
        Some(DurableEffectAttemptStateV1::Prepared { .. })
    ));

    let acknowledged = journal
        .persist_adapter_acknowledged_anchored(&prepared, Digest32([0xE7; 32]))
        .unwrap();
    assert_eq!(acknowledged.generation(), 2);
    assert_eq!(journal.anchored_head().generation(), 2);
    assert_eq!(observer.observed_head(), journal.anchored_head());
    let checkpoint = journal.current_checkpoint().unwrap();
    assert!(matches!(
        checkpoint.latest(),
        Some(DurableEffectAttemptStateV1::AdapterAcknowledged {
            adapter_evidence_digest,
            ..
        }) if *adapter_evidence_digest == Digest32([0xE7; 32])
    ));
    assert!(checkpoint.latest().unwrap().requires_reconciliation());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn anchor_failure_before_apply_leaves_local_ahead_and_reopen_fails_closed() {
    let root = temp_root("anchor-fail");
    let device = ResourceRef("iot:valve:72".into());
    let genesis = genesis(&device);
    let anchor = TestAnchor::new(genesis);
    anchor.set_mode(AnchorMode::FailBeforeApply);
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);

    let error = journal.persist_prepared_anchored(&correlation).unwrap_err();
    assert!(matches!(
        error,
        RollbackProtectedEffectAttemptJournalError::AnchorAdvance { .. }
    ));
    assert!(journal.is_poisoned());
    assert_eq!(observer.observed_head(), genesis);
    drop(journal);

    let reopen = RollbackProtectedEffectAttemptJournal::open(&root, &device, observer.clone());
    assert!(matches!(
        reopen,
        Err(RollbackProtectedEffectAttemptJournalError::Local(
            EffectAttemptJournalError::TrustedJournalHeadMismatch
        ))
    ));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn apply_then_error_poisoning_recovers_as_anchored_unresolved_prepared() {
    let root = temp_root("apply-then-error");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    anchor.set_mode(AnchorMode::ApplyThenError);
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);

    let error = journal.persist_prepared_anchored(&correlation).unwrap_err();
    assert!(matches!(
        error,
        RollbackProtectedEffectAttemptJournalError::AnchorAdvance { .. }
    ));
    assert!(journal.is_poisoned());
    assert_eq!(observer.observed_head().generation(), 1);
    drop(journal);

    observer.set_mode(AnchorMode::Normal);
    let mut reopened =
        RollbackProtectedEffectAttemptJournal::open(&root, &device, observer.clone()).unwrap();
    let checkpoint = reopened.current_checkpoint().unwrap();
    assert_eq!(checkpoint.generation(), 1);
    assert!(matches!(
        checkpoint.latest(),
        Some(DurableEffectAttemptStateV1::Prepared { .. })
    ));
    assert!(checkpoint.latest().unwrap().requires_reconciliation());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn successful_cas_followed_by_stale_read_never_confirms_transition() {
    let root = temp_root("stale-post-read");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    anchor.set_mode(AnchorMode::ApplyThenReportStaleOnce);
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);

    let error = journal.persist_prepared_anchored(&correlation).unwrap_err();
    assert!(matches!(
        error,
        RollbackProtectedEffectAttemptJournalError::AnchorPostAdvanceMismatch { .. }
    ));
    assert!(journal.is_poisoned());
    assert_eq!(observer.observed_head().generation(), 1);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn successful_cas_followed_by_read_error_never_confirms_transition() {
    let root = temp_root("failed-post-read");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    anchor.set_mode(AnchorMode::ApplyThenReadErrorOnce);
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let correlation = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);

    let error = journal.persist_prepared_anchored(&correlation).unwrap_err();
    assert!(matches!(
        error,
        RollbackProtectedEffectAttemptJournalError::AnchorPostAdvanceRead { .. }
    ));
    assert!(journal.is_poisoned());
    assert_eq!(observer.observed_head().generation(), 1);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn anchored_unresolved_attempt_blocks_next_sequence() {
    let root = temp_root("unresolved-blocks");
    let device = ResourceRef("iot:valve:72".into());
    let anchor = TestAnchor::new(genesis(&device));
    let observer = anchor.clone();
    let mut journal = RollbackProtectedEffectAttemptJournal::open(&root, &device, anchor).unwrap();
    let first = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 1);
    journal.persist_prepared_anchored(&first).unwrap();
    drop(journal);

    let mut reopened =
        RollbackProtectedEffectAttemptJournal::open(&root, &device, observer.clone()).unwrap();
    let second = PhysicalEffectAttemptCorrelation::qualification_fixture(device.clone(), 2);
    let error = reopened.persist_prepared_anchored(&second).unwrap_err();
    assert!(matches!(
        error,
        RollbackProtectedEffectAttemptJournalError::Local(
            EffectAttemptJournalError::UnresolvedAttemptExists
        )
    ));
    assert!(reopened.is_poisoned());
    assert_eq!(observer.observed_head().generation(), 1);

    fs::remove_dir_all(root).unwrap();
}
