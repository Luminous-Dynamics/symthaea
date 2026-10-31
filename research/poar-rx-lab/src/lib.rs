#![forbid(unsafe_code)]

//! Deterministic, single-action fault simulator for POAR-RX.
//!
//! This is a research microbenchmark, not a live adapter, durable journal,
//! distributed transaction protocol, or proof of universal exactly-once
//! execution. Its deliberately small effect store makes retry consequences
//! observable and repeatable.

use pod_meta_lab::recovery::{
    ActionIntent, ActionLifecycle, ActionPhase, DispatchSemantics, StateObservation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultSchedule {
    NoFault,
    TimeoutBeforeCommit,
    CommitThenLoseAcknowledgement,
    CommitThenReadbackUnavailable,
}

impl FaultSchedule {
    pub const ALL: [Self; 4] = [
        Self::NoFault,
        Self::TimeoutBeforeCommit,
        Self::CommitThenLoseAcknowledgement,
        Self::CommitThenReadbackUnavailable,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoFault => "no_fault",
            Self::TimeoutBeforeCommit => "timeout_before_commit",
            Self::CommitThenLoseAcknowledgement => "commit_then_lose_ack",
            Self::CommitThenReadbackUnavailable => "commit_then_readback_unavailable",
        }
    }

    fn first_dispatch(self) -> (bool, bool) {
        match self {
            Self::NoFault => (true, true),
            Self::TimeoutBeforeCommit => (false, false),
            Self::CommitThenLoseAcknowledgement
            | Self::CommitThenReadbackUnavailable => (true, false),
        }
    }

    fn readback_unavailable(self) -> bool {
        self == Self::CommitThenReadbackUnavailable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerPolicy {
    BlindRetryOnce,
    StopOnUnknown,
    ReconcileFirst,
}

impl ControllerPolicy {
    pub const ALL: [Self; 3] = [
        Self::BlindRetryOnce,
        Self::StopOnUnknown,
        Self::ReconcileFirst,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BlindRetryOnce => "blind_retry_once",
            Self::StopOnUnknown => "stop_on_unknown",
            Self::ReconcileFirst => "reconcile_first",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOutcome {
    CompletedVerified,
    ReportedCompleteUnverified,
    AcknowledgedUnverified,
    SafeAbort,
    UnknownOutcome,
    Failed,
}

impl TerminalOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompletedVerified => "completed_verified",
            Self::ReportedCompleteUnverified => "reported_complete_unverified",
            Self::AcknowledgedUnverified => "acknowledged_unverified",
            Self::SafeAbort => "safe_abort",
            Self::UnknownOutcome => "unknown_outcome",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrialResult {
    pub policy: ControllerPolicy,
    pub fault: FaultSchedule,
    pub dispatch_semantics: DispatchSemantics,
    pub outcome: TerminalOutcome,
    pub dispatch_attempts: u32,
    pub readback_calls: u32,
    pub actual_effects: u32,
    pub reported_complete: bool,
    pub postcondition_verified: bool,
    pub duplicate_effect: bool,
    pub false_completion: bool,
    pub unverified_completion_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchReceipt {
    acknowledged: bool,
}

#[derive(Debug, Default)]
struct EffectStore {
    effects: u32,
    idempotency_keys: std::collections::BTreeSet<String>,
}

impl EffectStore {
    fn dispatch(
        &mut self,
        intent: &ActionIntent,
        commit: bool,
        acknowledged: bool,
    ) -> DispatchReceipt {
        if !commit {
            return DispatchReceipt { acknowledged: false };
        }

        match intent.dispatch_semantics {
            DispatchSemantics::NonIdempotent => {
                self.effects = self.effects.saturating_add(1);
            }
            DispatchSemantics::IdempotencyKeyEnforced => {
                let key = intent.idempotency_key.as_deref().unwrap_or_default();
                if self.idempotency_keys.insert(key.to_owned()) {
                    self.effects = self.effects.saturating_add(1);
                }
            }
        }

        DispatchReceipt { acknowledged }
    }

    fn observe(&self, intent: &ActionIntent, unavailable: bool) -> StateObservation {
        if unavailable {
            return StateObservation::Unavailable;
        }

        match self.effects {
            0 => StateObservation::NotCommitted {
                state_identity: intent.pre_state_identity.clone(),
            },
            1 => StateObservation::Committed {
                state_identity: intent.expected_post_state_identity.clone(),
            },
            _ => StateObservation::Committed {
                state_identity: "state-after-duplicate-effects".to_owned(),
            },
        }
    }
}

fn make_intent(semantics: DispatchSemantics) -> ActionIntent {
    ActionIntent {
        run_id: "poar-rx-run-1".to_owned(),
        sequence: 1,
        action_id: "create-record-1".to_owned(),
        pre_state_identity: "state-before".to_owned(),
        expected_post_state_identity: "state-after-one-effect".to_owned(),
        principal_id: "research-agent".to_owned(),
        capability_id: "create-record".to_owned(),
        idempotency_key: (semantics == DispatchSemantics::IdempotencyKeyEnforced)
            .then(|| "poar-rx-idempotency-key-1".to_owned()),
        dispatch_semantics: semantics,
        compensation_supported: false,
    }
}

pub fn run_trial(
    policy: ControllerPolicy,
    fault: FaultSchedule,
    semantics: DispatchSemantics,
) -> TrialResult {
    let intent = make_intent(semantics);
    let mut store = EffectStore::default();
    let mut dispatch_attempts = 0;
    let mut readback_calls = 0;
    let mut reported_complete = false;
    let mut postcondition_verified = false;

    let outcome = match policy {
        ControllerPolicy::BlindRetryOnce => {
            let (commit, acknowledged) = fault.first_dispatch();
            dispatch_attempts += 1;
            let first = store.dispatch(&intent, commit, acknowledged);

            let final_receipt = if first.acknowledged {
                first
            } else {
                // Deliberately unsafe baseline: retry without reconciliation.
                dispatch_attempts += 1;
                store.dispatch(&intent, true, true)
            };

            if final_receipt.acknowledged {
                reported_complete = true;
                TerminalOutcome::ReportedCompleteUnverified
            } else {
                TerminalOutcome::Failed
            }
        }
        ControllerPolicy::StopOnUnknown => {
            let mut lifecycle = ActionLifecycle::new(intent.clone())
                .expect("fixed simulator intent must be valid");
            lifecycle.dispatch().expect("initial dispatch transition is valid");
            dispatch_attempts += 1;

            let (commit, acknowledged) = fault.first_dispatch();
            let receipt = store.dispatch(&intent, commit, acknowledged);

            if receipt.acknowledged {
                lifecycle.acknowledge().expect("acknowledgement transition is valid");
                // This baseline deliberately does not perform authoritative read-back.
                TerminalOutcome::AcknowledgedUnverified
            } else {
                lifecycle
                    .acknowledgement_lost()
                    .expect("ambiguous outcome transition is valid");
                let _ = lifecycle.stop();
                TerminalOutcome::UnknownOutcome
            }
        }
        ControllerPolicy::ReconcileFirst => {
            let mut lifecycle = ActionLifecycle::new(intent.clone())
                .expect("fixed simulator intent must be valid");
            lifecycle.dispatch().expect("initial dispatch transition is valid");
            dispatch_attempts += 1;

            let (commit, acknowledged) = fault.first_dispatch();
            let first = store.dispatch(&intent, commit, acknowledged);

            if first.acknowledged {
                lifecycle.acknowledge().expect("acknowledgement transition is valid");
                readback_calls += 1;
                let observation = store.observe(&intent, false);
                match lifecycle.reconcile(observation) {
                    Ok(ActionPhase::ReconciledCommitted) => {
                        reported_complete = true;
                        postcondition_verified = true;
                        TerminalOutcome::CompletedVerified
                    }
                    _ => {
                        let _ = lifecycle.stop();
                        TerminalOutcome::UnknownOutcome
                    }
                }
            } else {
                lifecycle
                    .acknowledgement_lost()
                    .expect("ambiguous outcome transition is valid");
                readback_calls += 1;
                let observation =
                    store.observe(&intent, fault.readback_unavailable());
                let phase = lifecycle.reconcile(observation).unwrap_or(lifecycle.phase());

                match phase {
                    ActionPhase::ReconciledCommitted => {
                        reported_complete = true;
                        postcondition_verified = true;
                        TerminalOutcome::CompletedVerified
                    }
                    ActionPhase::ReconciledNotCommitted => {
                        // Retry is permitted only after authoritative non-commit evidence.
                        if lifecycle.authorize_retry().is_err()
                            || lifecycle.dispatch().is_err()
                        {
                            let _ = lifecycle.stop();
                            TerminalOutcome::Failed
                        } else {
                            dispatch_attempts += 1;
                            let retry = store.dispatch(&intent, true, true);
                            if !retry.acknowledged {
                                let _ = lifecycle.stop();
                                TerminalOutcome::UnknownOutcome
                            } else if lifecycle.acknowledge().is_err() {
                                TerminalOutcome::Failed
                            } else {
                                readback_calls += 1;
                                let observation = store.observe(&intent, false);
                                match lifecycle.reconcile(observation) {
                                    Ok(ActionPhase::ReconciledCommitted) => {
                                        reported_complete = true;
                                        postcondition_verified = true;
                                        TerminalOutcome::CompletedVerified
                                    }
                                    _ => {
                                        let _ = lifecycle.stop();
                                        TerminalOutcome::UnknownOutcome
                                    }
                                }
                            }
                        }
                    }
                    ActionPhase::UnknownOutcome
                    | ActionPhase::StoppedUnknownOutcome => {
                        let _ = lifecycle.stop();
                        TerminalOutcome::UnknownOutcome
                    }
                    _ => {
                        let _ = lifecycle.stop();
                        TerminalOutcome::Failed
                    }
                }
            }
        }
    };

    let duplicate_effect = store.effects > 1;
    let false_completion = reported_complete && store.effects != 1;
    let unverified_completion_claim = reported_complete && !postcondition_verified;

    TrialResult {
        policy,
        fault,
        dispatch_semantics: semantics,
        outcome,
        dispatch_attempts,
        readback_calls,
        actual_effects: store.effects,
        reported_complete,
        postcondition_verified,
        duplicate_effect,
        false_completion,
        unverified_completion_claim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconciliation_after_commit_then_lost_ack_avoids_duplicate_effect() {
        let result = run_trial(
            ControllerPolicy::ReconcileFirst,
            FaultSchedule::CommitThenLoseAcknowledgement,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::CompletedVerified);
        assert_eq!(result.actual_effects, 1);
        assert_eq!(result.dispatch_attempts, 1);
        assert!(result.postcondition_verified);
        assert!(!result.duplicate_effect);
        assert!(!result.false_completion);
    }

    #[test]
    fn blind_retry_after_commit_then_lost_ack_duplicates_effect() {
        let result = run_trial(
            ControllerPolicy::BlindRetryOnce,
            FaultSchedule::CommitThenLoseAcknowledgement,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::ReportedCompleteUnverified);
        assert_eq!(result.actual_effects, 2);
        assert!(result.duplicate_effect);
        assert!(result.false_completion);
        assert!(result.unverified_completion_claim);
    }

    #[test]
    fn authoritative_non_commit_readback_allows_one_safe_retry() {
        let result = run_trial(
            ControllerPolicy::ReconcileFirst,
            FaultSchedule::TimeoutBeforeCommit,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::CompletedVerified);
        assert_eq!(result.dispatch_attempts, 2);
        assert_eq!(result.actual_effects, 1);
        assert_eq!(result.readback_calls, 2);
        assert!(result.postcondition_verified);
    }

    #[test]
    fn unavailable_readback_preserves_unknown_outcome_without_retry() {
        let result = run_trial(
            ControllerPolicy::ReconcileFirst,
            FaultSchedule::CommitThenReadbackUnavailable,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::UnknownOutcome);
        assert_eq!(result.actual_effects, 1);
        assert_eq!(result.dispatch_attempts, 1);
        assert!(!result.reported_complete);
        assert!(!result.duplicate_effect);
    }

    #[test]
    fn stop_on_unknown_does_not_repeat_an_ambiguous_non_idempotent_action() {
        let result = run_trial(
            ControllerPolicy::StopOnUnknown,
            FaultSchedule::CommitThenLoseAcknowledgement,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::UnknownOutcome);
        assert_eq!(result.dispatch_attempts, 1);
        assert_eq!(result.actual_effects, 1);
        assert!(!result.reported_complete);
    }

    #[test]
    fn enforced_idempotency_key_deduplicates_blind_retry_but_not_verification_gap() {
        let result = run_trial(
            ControllerPolicy::BlindRetryOnce,
            FaultSchedule::CommitThenLoseAcknowledgement,
            DispatchSemantics::IdempotencyKeyEnforced,
        );
        assert_eq!(result.actual_effects, 1);
        assert!(!result.duplicate_effect);
        assert!(!result.false_completion);
        assert!(result.unverified_completion_claim);
    }

    #[test]
    fn no_fault_reconcile_first_requires_and_records_readback() {
        let result = run_trial(
            ControllerPolicy::ReconcileFirst,
            FaultSchedule::NoFault,
            DispatchSemantics::NonIdempotent,
        );
        assert_eq!(result.outcome, TerminalOutcome::CompletedVerified);
        assert_eq!(result.dispatch_attempts, 1);
        assert_eq!(result.readback_calls, 1);
        assert!(result.postcondition_verified);
    }
}
