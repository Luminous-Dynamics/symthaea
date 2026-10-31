//! In-memory run journal enforcing action identity and monotonic sequence.

use crate::recovery::{ActionIntent, ActionLifecycle, RecoveryError};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalError {
    EmptyRunId,
    RunIdMismatch,
    SequenceMismatch { expected: u64, actual: u64 },
    DuplicateActionId,
    SequenceExhausted,
    InvalidAction(RecoveryError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryJournal {
    run_id: String,
    next_sequence: u64,
    action_ids: BTreeSet<String>,
    actions: Vec<ActionLifecycle>,
}

impl RecoveryJournal {
    pub fn new(run_id: impl Into<String>) -> Result<Self, JournalError> {
        let run_id = run_id.into();
        if run_id.is_empty() {
            return Err(JournalError::EmptyRunId);
        }
        Ok(Self {
            run_id,
            next_sequence: 1,
            action_ids: BTreeSet::new(),
            actions: Vec::new(),
        })
    }

    pub fn run_id(&self) -> &str { &self.run_id }
    pub fn actions(&self) -> &[ActionLifecycle] { &self.actions }
    pub fn next_sequence(&self) -> u64 { self.next_sequence }

    pub fn prepare_action(&mut self, intent: ActionIntent) -> Result<(), JournalError> {
        if intent.run_id != self.run_id {
            return Err(JournalError::RunIdMismatch);
        }
        if intent.sequence != self.next_sequence {
            return Err(JournalError::SequenceMismatch {
                expected: self.next_sequence,
                actual: intent.sequence,
            });
        }
        if self.action_ids.contains(&intent.action_id) {
            return Err(JournalError::DuplicateActionId);
        }
        let next_sequence = self.next_sequence.checked_add(1)
            .ok_or(JournalError::SequenceExhausted)?;
        let action = ActionLifecycle::new(intent).map_err(JournalError::InvalidAction)?;
        self.action_ids.insert(action.intent().action_id.clone());
        self.actions.push(action);
        self.next_sequence = next_sequence;
        Ok(())
    }

    pub fn action(&self, action_id: &str) -> Option<&ActionLifecycle> {
        self.actions.iter().find(|action| action.intent().action_id == action_id)
    }

    pub fn action_mut(&mut self, action_id: &str) -> Option<&mut ActionLifecycle> {
        self.actions.iter_mut().find(|action| action.intent().action_id == action_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::DispatchSemantics;

    fn intent(run: &str, sequence: u64, action: &str) -> ActionIntent {
        ActionIntent {
            run_id: run.into(),
            sequence,
            action_id: action.into(),
            pre_state_identity: format!("before-{sequence}"),
            expected_post_state_identity: format!("after-{sequence}"),
            principal_id: "agent-a".into(),
            capability_id: "write-record".into(),
            idempotency_key: None,
            dispatch_semantics: DispatchSemantics::NonIdempotent,
            compensation_supported: false,
        }
    }

    #[test]
    fn journal_enforces_run_identity_monotonic_sequence_and_unique_action_ids() {
        let mut journal = RecoveryJournal::new("run-1").unwrap();
        journal.prepare_action(intent("run-1", 1, "action-1")).unwrap();
        journal.prepare_action(intent("run-1", 2, "action-2")).unwrap();
        assert_eq!(journal.next_sequence(), 3);
        assert_eq!(journal.actions().len(), 2);
        assert_eq!(
            journal.prepare_action(intent("run-1", 3, "action-1")),
            Err(JournalError::DuplicateActionId)
        );
        assert_eq!(
            journal.prepare_action(intent("run-1", 4, "action-3")),
            Err(JournalError::SequenceMismatch { expected: 3, actual: 4 })
        );
    }

    #[test]
    fn journal_rejects_actions_from_a_different_run() {
        let mut journal = RecoveryJournal::new("run-1").unwrap();
        assert_eq!(
            journal.prepare_action(intent("run-2", 1, "action-1")),
            Err(JournalError::RunIdMismatch)
        );
        assert_eq!(journal.actions().len(), 0);
        assert_eq!(journal.next_sequence(), 1);
    }

    #[test]
    fn journal_preserves_action_event_history_without_reordering() {
        let mut journal = RecoveryJournal::new("run-1").unwrap();
        journal.prepare_action(intent("run-1", 1, "action-1")).unwrap();
        let action = journal.action_mut("action-1").unwrap();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        let entry = journal.action("action-1").unwrap();
        assert_eq!(entry.phase(), crate::recovery::ActionPhase::UnknownOutcome);
        assert!(matches!(entry.events().first(), Some(crate::recovery::JournalEvent::Prepared { sequence: 1 })));
        assert!(matches!(entry.events().get(1), Some(crate::recovery::JournalEvent::Dispatched { attempt: 1 })));
        assert!(matches!(entry.events().get(2), Some(crate::recovery::JournalEvent::AcknowledgementLost)));
    }

    #[test]
    fn invalid_action_does_not_advance_journal_sequence() {
        let mut journal = RecoveryJournal::new("run-1").unwrap();
        let mut invalid = intent("run-1", 1, "action-1");
        invalid.principal_id.clear();
        assert!(matches!(journal.prepare_action(invalid), Err(JournalError::InvalidAction(_))));
        assert_eq!(journal.next_sequence(), 1);
        assert!(journal.actions().is_empty());
    }
}
