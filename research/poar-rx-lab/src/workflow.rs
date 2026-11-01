//! Two-target workflow recovery experiment.
//!
//! The simulation distinguishes acknowledged writes from actual world state,
//! tests preservation of concurrent mutations, and treats compensation as a
//! separately authorized state transition. It is a deterministic research
//! model, not a live API adapter or a complete transaction protocol.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowFault {
    NoFault,
    SecondCommitAckLost,
    SecondCommitAckLostReadbackUnavailable,
    PermissionRevokedBeforeSecond,
    ConcurrentUpdateFirstTarget,
    ConcurrentUpdateSecondTarget,
    SecondActionRejected,
    CompensationFailure,
}

impl WorkflowFault {
    pub const ALL: [Self; 8] = [
        Self::NoFault,
        Self::SecondCommitAckLost,
        Self::SecondCommitAckLostReadbackUnavailable,
        Self::PermissionRevokedBeforeSecond,
        Self::ConcurrentUpdateFirstTarget,
        Self::ConcurrentUpdateSecondTarget,
        Self::SecondActionRejected,
        Self::CompensationFailure,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoFault => "no_fault",
            Self::SecondCommitAckLost => "second_commit_ack_lost",
            Self::SecondCommitAckLostReadbackUnavailable => {
                "second_commit_ack_lost_readback_unavailable"
            }
            Self::PermissionRevokedBeforeSecond => "permission_revoked_before_second",
            Self::ConcurrentUpdateFirstTarget => "concurrent_update_first_target",
            Self::ConcurrentUpdateSecondTarget => "concurrent_update_second_target",
            Self::SecondActionRejected => "second_action_rejected",
            Self::CompensationFailure => "compensation_failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowPolicy {
    BlindWorkflowReplay,
    SagaCompensation,
    ProofGatedReconcile,
}

impl WorkflowPolicy {
    pub const ALL: [Self; 3] = [
        Self::BlindWorkflowReplay,
        Self::SagaCompensation,
        Self::ProofGatedReconcile,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BlindWorkflowReplay => "blind_workflow_replay",
            Self::SagaCompensation => "saga_compensation",
            Self::ProofGatedReconcile => "proof_gated_reconcile",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowTerminal {
    CompletedVerified,
    CompletedUnverified,
    CompensatedVerified,
    PartialState,
    UnknownOutcome,
}

impl WorkflowTerminal {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompletedVerified => "completed_verified",
            Self::CompletedUnverified => "completed_unverified",
            Self::CompensatedVerified => "compensated_verified",
            Self::PartialState => "partial_state",
            Self::UnknownOutcome => "unknown_outcome",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowResult {
    pub policy: WorkflowPolicy,
    pub fault: WorkflowFault,
    pub compensation_authorized: bool,
    pub terminal: WorkflowTerminal,
    pub first_target: String,
    pub second_target: String,
    pub first_agent_writes: u32,
    pub second_agent_writes: u32,
    pub compensation_writes: u32,
    pub external_writes: u32,
    pub readback_calls: u32,
    pub unauthorized_effects: u32,
    pub lost_concurrent_updates: u32,
    pub reported_complete: bool,
    pub goal_state_holds: bool,
    pub completion_verified: bool,
    pub compensation_verified: bool,
    pub duplicate_effect: bool,
    pub false_completion: bool,
    pub compensation_failed: bool,
}

#[derive(Debug)]
struct World {
    first: &'static str,
    second: &'static str,
    first_version: u64,
    second_version: u64,
    first_agent_writes: u32,
    second_agent_writes: u32,
    compensation_writes: u32,
    external_writes: u32,
    unauthorized_effects: u32,
    lost_concurrent_updates: u32,
    compensation_failed: bool,
}

impl Default for World {
    fn default() -> Self {
        Self {
            first: "first-before",
            second: "second-before",
            first_version: 0,
            second_version: 0,
            first_agent_writes: 0,
            second_agent_writes: 0,
            compensation_writes: 0,
            external_writes: 0,
            unauthorized_effects: 0,
            lost_concurrent_updates: 0,
            compensation_failed: false,
        }
    }
}

impl World {
    fn write_first(&mut self, force: bool) -> bool {
        if !force && self.first_version != 0 {
            return false;
        }
        if self.first.starts_with("external-") {
            self.lost_concurrent_updates += 1;
        }
        self.first = "first-after";
        self.first_version += 1;
        self.first_agent_writes += 1;
        true
    }

    fn write_second(&mut self, authorized: bool, force: bool) -> bool {
        if !authorized {
            self.unauthorized_effects += 1;
            return false;
        }
        if !force && self.second_version != 0 {
            return false;
        }
        if self.second.starts_with("external-") {
            self.lost_concurrent_updates += 1;
        }
        self.second = "second-after";
        self.second_version += 1;
        self.second_agent_writes += 1;
        true
    }

    fn external_update_first(&mut self) {
        self.first = "external-first";
        self.first_version += 1;
        self.external_writes += 1;
    }

    fn external_update_second(&mut self) {
        self.second = "external-second";
        self.second_version += 1;
        self.external_writes += 1;
    }

    fn compensate_first(&mut self, authorized: bool, injected_failure: bool) -> bool {
        if !authorized {
            self.unauthorized_effects += 1;
            return false;
        }
        // A compensating write must not overwrite a concurrent writer.
        if self.first != "first-after" {
            return false;
        }
        if injected_failure {
            self.compensation_failed = true;
            return false;
        }
        self.first = "first-before";
        self.first_version += 1;
        self.compensation_writes += 1;
        true
    }

    fn goal_holds(&self) -> bool {
        self.first == "first-after" && self.second == "second-after"
    }
}

pub fn run_workflow(
    policy: WorkflowPolicy,
    fault: WorkflowFault,
    compensation_authorized: bool,
) -> WorkflowResult {
    let mut world = World::default();

    // First branch commits successfully before the second branch is attempted.
    assert!(world.write_first(false));

    let mut second_authorized = true;
    match fault {
        WorkflowFault::ConcurrentUpdateFirstTarget => world.external_update_first(),
        WorkflowFault::ConcurrentUpdateSecondTarget => world.external_update_second(),
        WorkflowFault::PermissionRevokedBeforeSecond => second_authorized = false,
        _ => {}
    }

    let second_expected_to_reject = matches!(
        fault,
        WorkflowFault::PermissionRevokedBeforeSecond
            | WorkflowFault::ConcurrentUpdateSecondTarget
            | WorkflowFault::SecondActionRejected
            | WorkflowFault::CompensationFailure
    );

    let second_committed = if second_expected_to_reject {
        false
    } else {
        world.write_second(second_authorized, false)
    };

    let second_acknowledged = second_committed
        && fault != WorkflowFault::SecondCommitAckLost
        && fault != WorkflowFault::SecondCommitAckLostReadbackUnavailable;

    let mut readback_calls = 0;
    let mut reported_complete = false;
    let mut completion_verified = false;
    let mut compensation_verified = false;
    let mut terminal = WorkflowTerminal::PartialState;

    match policy {
        WorkflowPolicy::BlindWorkflowReplay => {
            if !second_acknowledged {
                // Deliberately unsafe baseline: replay both branches, ignoring
                // pre-state versions and current authorization.
                let _ = world.write_first(true);
                let _ = world.write_second(second_authorized, true);
            }
            reported_complete = world.goal_holds();
            terminal = if reported_complete {
                WorkflowTerminal::CompletedUnverified
            } else {
                WorkflowTerminal::PartialState
            };
        }
        WorkflowPolicy::SagaCompensation => {
            if second_acknowledged {
                // Saga trusts its local step acknowledgements; it does not
                // independently re-read every workflow postcondition.
                reported_complete = true;
                terminal = WorkflowTerminal::CompletedUnverified;
            } else if fault == WorkflowFault::SecondCommitAckLost {
                // Reconcile the ambiguous second step only.
                readback_calls += 1;
                if world.second == "second-after" {
                    reported_complete = true;
                    terminal = WorkflowTerminal::CompletedUnverified;
                }
            } else if fault == WorkflowFault::SecondCommitAckLostReadbackUnavailable {
                readback_calls += 1;
                terminal = WorkflowTerminal::UnknownOutcome;
            } else if compensation_authorized {
                let injected_failure = fault == WorkflowFault::CompensationFailure;
                if world.compensate_first(true, injected_failure) {
                    readback_calls += 1;
                    compensation_verified =
                        world.first == "first-before" && world.second_agent_writes == 0;
                    terminal = if compensation_verified {
                        WorkflowTerminal::CompensatedVerified
                    } else {
                        WorkflowTerminal::PartialState
                    };
                }
            }
        }
        WorkflowPolicy::ProofGatedReconcile => {
            // Inspect both branches, not merely the last action acknowledgement.
            readback_calls += 2;
            if fault == WorkflowFault::SecondCommitAckLostReadbackUnavailable {
                terminal = WorkflowTerminal::UnknownOutcome;
            } else if world.goal_holds() && world.unauthorized_effects == 0 {
                reported_complete = true;
                completion_verified = true;
                terminal = WorkflowTerminal::CompletedVerified;
            } else if world.second_agent_writes == 0
                && world.first == "first-after"
                && compensation_authorized
            {
                let injected_failure = fault == WorkflowFault::CompensationFailure;
                if world.compensate_first(true, injected_failure) {
                    // Verify both the compensation and that a concurrent
                    // external update to the other branch was preserved.
                    readback_calls += 2;
                    compensation_verified = world.first == "first-before"
                        && world.second_agent_writes == 0
                        && world.unauthorized_effects == 0;
                    terminal = if compensation_verified {
                        WorkflowTerminal::CompensatedVerified
                    } else {
                        WorkflowTerminal::PartialState
                    };
                }
            } else {
                terminal = WorkflowTerminal::PartialState;
            }
        }
    }

    let goal_state_holds = world.goal_holds();
    let duplicate_effect = world.first_agent_writes > 1 || world.second_agent_writes > 1;
    let false_completion = reported_complete && !goal_state_holds;

    WorkflowResult {
        policy,
        fault,
        compensation_authorized,
        terminal,
        first_target: world.first.to_owned(),
        second_target: world.second.to_owned(),
        first_agent_writes: world.first_agent_writes,
        second_agent_writes: world.second_agent_writes,
        compensation_writes: world.compensation_writes,
        external_writes: world.external_writes,
        readback_calls,
        unauthorized_effects: world.unauthorized_effects,
        lost_concurrent_updates: world.lost_concurrent_updates,
        reported_complete,
        goal_state_holds,
        completion_verified,
        compensation_verified,
        duplicate_effect,
        false_completion,
        compensation_failed: world.compensation_failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_gate_reconciles_lost_ack_without_replaying_committed_branches() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::SecondCommitAckLost,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompletedVerified);
        assert!(result.completion_verified);
        assert_eq!(result.first_agent_writes, 1);
        assert_eq!(result.second_agent_writes, 1);
        assert!(!result.duplicate_effect);
        assert!(!result.false_completion);
    }

    #[test]
    fn blind_workflow_replay_duplicates_both_committed_branches_after_lost_ack() {
        let result = run_workflow(
            WorkflowPolicy::BlindWorkflowReplay,
            WorkflowFault::SecondCommitAckLost,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompletedUnverified);
        assert_eq!(result.first_agent_writes, 2);
        assert_eq!(result.second_agent_writes, 2);
        assert!(result.duplicate_effect);
    }

    #[test]
    fn proof_gate_detects_concurrent_mutation_of_an_earlier_branch() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::ConcurrentUpdateFirstTarget,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::PartialState);
        assert!(!result.reported_complete);
        assert!(!result.goal_state_holds);
        assert_eq!(result.first_target, "external-first");
    }

    #[test]
    fn acknowledgement_only_saga_misses_concurrent_mutation_of_earlier_branch() {
        let result = run_workflow(
            WorkflowPolicy::SagaCompensation,
            WorkflowFault::ConcurrentUpdateFirstTarget,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompletedUnverified);
        assert!(result.reported_complete);
        assert!(result.false_completion);
    }

    #[test]
    fn blind_replay_must_not_override_revoked_authority() {
        let result = run_workflow(
            WorkflowPolicy::BlindWorkflowReplay,
            WorkflowFault::PermissionRevokedBeforeSecond,
            false,
        );
        assert_eq!(result.unauthorized_effects, 1);
        assert!(!result.goal_state_holds);
        assert!(!result.reported_complete);
    }

    #[test]
    fn proof_gate_preserves_partial_state_when_compensation_is_not_authorized() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::PermissionRevokedBeforeSecond,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::PartialState);
        assert_eq!(result.first_target, "first-after");
        assert_eq!(result.second_target, "second-before");
        assert_eq!(result.compensation_writes, 0);
        assert_eq!(result.unauthorized_effects, 0);
    }

    #[test]
    fn authorized_compensation_restores_the_first_branch_without_touching_external_state() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::ConcurrentUpdateSecondTarget,
            true,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompensatedVerified);
        assert!(result.compensation_verified);
        assert_eq!(result.first_target, "first-before");
        assert_eq!(result.second_target, "external-second");
        assert_eq!(result.lost_concurrent_updates, 0);
    }

    #[test]
    fn concurrent_update_is_not_destroyed_by_authorized_compensation() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::ConcurrentUpdateSecondTarget,
            true,
        );
        assert_eq!(result.second_target, "external-second");
        assert_eq!(result.external_writes, 1);
        assert_eq!(result.lost_concurrent_updates, 0);
    }

    #[test]
    fn compensation_failure_leaves_a_truthful_partial_state() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::CompensationFailure,
            true,
        );
        assert_eq!(result.terminal, WorkflowTerminal::PartialState);
        assert!(result.compensation_failed);
        assert_eq!(result.first_target, "first-after");
        assert!(!result.reported_complete);
    }

    #[test]
    fn unavailable_readback_after_commit_remains_unknown_without_compensation() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::SecondCommitAckLostReadbackUnavailable,
            true,
        );
        assert_eq!(result.terminal, WorkflowTerminal::UnknownOutcome);
        assert_eq!(result.first_target, "first-after");
        assert_eq!(result.second_target, "second-after");
        assert_eq!(result.compensation_writes, 0);
    }

    #[test]
    fn definite_rejection_can_be_safely_compensated_when_authorized() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::SecondActionRejected,
            true,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompensatedVerified);
        assert!(result.compensation_verified);
    }

    #[test]
    fn proof_gate_completes_nominal_workflow_only_after_both_readbacks() {
        let result = run_workflow(
            WorkflowPolicy::ProofGatedReconcile,
            WorkflowFault::NoFault,
            false,
        );
        assert_eq!(result.terminal, WorkflowTerminal::CompletedVerified);
        assert_eq!(result.readback_calls, 2);
        assert!(result.completion_verified);
    }
}
