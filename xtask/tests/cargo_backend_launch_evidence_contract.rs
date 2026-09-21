#![allow(dead_code)]

#[path = "../src/repository_snapshot_diff.rs"]
mod repository_snapshot_diff;
#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/repository_effect_policy.rs"]
mod repository_effect_policy;
#[path = "../src/cargo_execution_contract.rs"]
mod cargo_execution_contract;
#[path = "../src/cargo_adapter_state.rs"]
mod cargo_adapter_state;
#[path = "../src/cargo_backend_launch_evidence.rs"]
mod cargo_backend_launch_evidence;

use cargo_adapter_state::ProcessTerminal;
use cargo_backend_launch_evidence::{
    BackendLaunchState, CargoBackendLaunchEvidenceInput, ChildTerminalState,
    LaunchAttemptOutcome, LegacyProcessProjection, build_launch_evidence,
    project_legacy_terminal, verify_launch_evidence,
};
use cargo_execution_contract::{CargoExecutionIntent, CargoExecutionIntentSpec, build_intent};

fn digest(seed: char) -> String {
    assert!(seed.is_ascii());
    format!("{:02x}", u32::from(seed)).repeat(32)
}

fn intent(plan: char) -> CargoExecutionIntent {
    build_intent(
        CargoExecutionIntentSpec {
            schema: "symthaea.cargo-execution-intent-input.v1".into(),
            git_worktree_state_before: Some(digest('6')),
            build_context_id: digest('2'),
            invocation_id: digest('3'),
            plan_id: Some(digest(plan)),
            transaction_id: Some(digest('8')),
            adapter_semantics_digest: digest('5'),
        },
        &digest('1'),
        &digest('4'),
    )
    .unwrap()
}

fn build(state: BackendLaunchState) -> cargo_backend_launch_evidence::CargoBackendLaunchEvidence {
    build_launch_evidence(
        intent('7'),
        CargoBackendLaunchEvidenceInput {
            backend_entry_evidence_digest: digest('a'),
            state,
        },
    )
    .unwrap()
}

#[test]
fn preparation_failure_is_not_legacy_spawn_failure() {
    let receipt = build(BackendLaunchState::PreparationFailed {
        error_digest: digest('b'),
    });
    assert!(matches!(
        project_legacy_terminal(&receipt),
        LegacyProcessProjection::NotRepresentable { .. }
    ));
}

#[test]
fn launch_failure_maps_to_legacy_spawn_failed_only_after_launch_attempt() {
    let receipt = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::LaunchFailed {
            error_digest: digest('d'),
        },
    });
    assert!(matches!(
        project_legacy_terminal(&receipt),
        LegacyProcessProjection::Representable(ProcessTerminal::SpawnFailed { .. })
    ));
}

#[test]
fn child_created_exit_maps_to_legacy_exit() {
    let receipt = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::ChildCreated {
            child_creation_evidence_digest: digest('e'),
            terminal: ChildTerminalState::Exited { code: 17 },
        },
    });
    assert_eq!(
        project_legacy_terminal(&receipt),
        LegacyProcessProjection::Representable(ProcessTerminal::Exited { code: 17 })
    );
}

#[test]
fn child_unknown_outcome_is_not_forced_into_legacy_terminal() {
    let receipt = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::ChildCreated {
            child_creation_evidence_digest: digest('e'),
            terminal: ChildTerminalState::OutcomeUnknown {
                evidence_digest: digest('f'),
            },
        },
    });
    assert!(matches!(
        project_legacy_terminal(&receipt),
        LegacyProcessProjection::NotRepresentable { .. }
    ));
}

#[test]
fn stored_receipt_rebuilds_against_exact_intent() {
    let receipt = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::ChildCreated {
            child_creation_evidence_digest: digest('e'),
            terminal: ChildTerminalState::TimedOut,
        },
    });
    let verified = verify_launch_evidence(intent('7'), receipt.clone()).unwrap();
    assert_eq!(verified, receipt);
    assert!(verify_launch_evidence(intent('9'), receipt).is_err());
}

#[test]
fn tampered_stored_identity_is_rejected() {
    let mut receipt = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::ChildCreated {
            child_creation_evidence_digest: digest('e'),
            terminal: ChildTerminalState::Cancelled,
        },
    });
    receipt.launch_evidence_id = digest('f');
    assert!(verify_launch_evidence(intent('7'), receipt).is_err());
}

#[test]
fn malformed_launch_or_child_digest_fails_closed() {
    let malformed_launch = build_launch_evidence(
        intent('7'),
        CargoBackendLaunchEvidenceInput {
            backend_entry_evidence_digest: digest('a'),
            state: BackendLaunchState::LaunchAttempted {
                launch_attempt_evidence_digest: "not-a-digest".into(),
                outcome: LaunchAttemptOutcome::LaunchFailed {
                    error_digest: digest('d'),
                },
            },
        },
    );
    assert!(malformed_launch.is_err());

    let malformed_child = build_launch_evidence(
        intent('7'),
        CargoBackendLaunchEvidenceInput {
            backend_entry_evidence_digest: digest('a'),
            state: BackendLaunchState::LaunchAttempted {
                launch_attempt_evidence_digest: digest('c'),
                outcome: LaunchAttemptOutcome::ChildCreated {
                    child_creation_evidence_digest: "bad".into(),
                    terminal: ChildTerminalState::Exited { code: 0 },
                },
            },
        },
    );
    assert!(malformed_child.is_err());
}

#[test]
fn each_launch_milestone_is_identity_significant() {
    let preparation = build(BackendLaunchState::PreparationFailed {
        error_digest: digest('b'),
    });
    let launch_failed = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::LaunchFailed {
            error_digest: digest('d'),
        },
    });
    let child = build(BackendLaunchState::LaunchAttempted {
        launch_attempt_evidence_digest: digest('c'),
        outcome: LaunchAttemptOutcome::ChildCreated {
            child_creation_evidence_digest: digest('e'),
            terminal: ChildTerminalState::Exited { code: 0 },
        },
    });

    assert_ne!(preparation.launch_evidence_id, launch_failed.launch_evidence_id);
    assert_ne!(launch_failed.launch_evidence_id, child.launch_evidence_id);
}
