// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::convert::Infallible;

use symthaea_wisdom::{
    ActionEthicsContext, ActionExecutionCoordinator, ActionId, ActionRequest, ActorId,
    AuthorizationStatus, CapabilityId, CapabilityScope, ConsentStatus, DeceptionStatus,
    EvidenceLedger, EvidenceSigner, EvidenceVerifier, ExecutorResult, LedgerRevision, ProcessId,
    ProcessStatus, ResourceId, RuntimeEventEnvelope, RuntimeIntegration, RuntimeProcessEvent,
    RuntimeSourceIdentity, TrustRegistry, WisdomObservation, WisdomState,
    attest_authority_checkpoint, attest_retention_checkpoint, fingerprint_bytes,
    recover_authority_from_checkpoint,
};

struct ToySigner;
impl EvidenceSigner for ToySigner {
    type Error = Infallible;
    fn algorithm(&self) -> &str {
        "toy"
    }
    fn key_id(&self) -> &str {
        "checkpoint-key"
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Self::Error> {
        Ok(fingerprint_bytes(message).to_le_bytes().to_vec())
    }
}
impl EvidenceVerifier for ToySigner {
    type Error = Infallible;
    fn algorithm(&self) -> &str {
        "toy"
    }
    fn key_id(&self) -> &str {
        "checkpoint-key"
    }
    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, Self::Error> {
        Ok(signature == fingerprint_bytes(message).to_le_bytes())
    }
}

fn safe_request(action: u64) -> ActionRequest {
    ActionRequest::new(
        ActionId::new(action),
        ActorId::new(2),
        CapabilityScope::new(CapabilityId::new(3), ResourceId::new(4)),
        10_000,
        ActionEthicsContext::new(
            AuthorizationStatus::Authorized,
            ConsentStatus::Granted,
            DeceptionStatus::Absent,
        )
        .with_expected_harm(0.01)
        .with_worst_case_harm(0.02)
        .with_reversibility(0.95)
        .with_externality_coverage(0.95),
    )
}

#[test]
fn retained_suffix_restores_permits_executions_and_runtime_cursors() {
    let mut state = WisdomState::new();
    *state.evidence_ledger_mut() = EvidenceLedger::new(16);
    state.update_from_observation_with_evidence(
        WisdomObservation::new(0.2, 0.3, 0.8),
        1,
        "observation",
    );

    let mut runtime = RuntimeIntegration::new("scheduler");
    runtime
        .apply(
            &mut state,
            RuntimeEventEnvelope::new(
                0,
                2,
                RuntimeProcessEvent::Register {
                    process: ProcessId::new(1),
                    label: "planner".to_owned(),
                },
            ),
        )
        .unwrap();

    let mut execution = ActionExecutionCoordinator::new();
    let prepared = execution
        .prepare_execution(&mut state, safe_request(10), 3, "executor")
        .unwrap()
        .prepared
        .unwrap();

    let registry = TrustRegistry::new();
    let retention = attest_retention_checkpoint(&state, 4, 8, &ToySigner).unwrap();
    let checkpoint = attest_authority_checkpoint(
        &state,
        &execution,
        &[runtime.clone()],
        &registry,
        &retention,
        LedgerRevision::from_ledger(state.evidence_ledger()),
        4,
        &ToySigner,
    )
    .unwrap();

    execution
        .complete_execution(
            &mut state,
            prepared,
            ExecutorResult::succeeded(77),
            5,
            "executor",
        )
        .unwrap();
    state
        .issue_action_permit_with_evidence(safe_request(11), 6, "queued")
        .unwrap();
    runtime
        .apply(
            &mut state,
            RuntimeEventEnvelope::new(
                1,
                7,
                RuntimeProcessEvent::StatusChanged {
                    process: ProcessId::new(1),
                    status: ProcessStatus::Degraded,
                },
            ),
        )
        .unwrap();
    for sequence in 0..8_u64 {
        state.update_from_observation_with_evidence(
            WisdomObservation::new(0.1, 0.2, 0.9),
            8 + sequence,
            "retained-suffix",
        );
    }
    assert!(state.evidence_ledger().evicted_records() > 0);

    let mut restored = state.clone();
    let sources = vec![RuntimeSourceIdentity::new("scheduler", 1, "toy", "scheduler-key").unwrap()];
    let outcome = recover_authority_from_checkpoint(
        &mut restored,
        state.evidence_ledger(),
        &checkpoint,
        &retention,
        &ToySigner,
        &ToySigner,
        &registry,
        &sources,
        20,
    )
    .unwrap();

    assert_eq!(outcome.report.execution_suffix.in_doubt_actions, 0);
    assert_eq!(outcome.report.outstanding_permits, 1);
    assert_eq!(
        outcome
            .runtime_integrations
            .get("scheduler")
            .unwrap()
            .expected_sequence(),
        2
    );
    assert!(matches!(
        outcome.execution.status(ActionId::new(10)),
        Some(symthaea_wisdom::ActionDispatchStatus::Completed { .. })
    ));
}

#[test]
fn unrelated_retention_checkpoint_is_rejected() {
    let mut state = WisdomState::new();
    state.update_from_observation_with_evidence(
        WisdomObservation::new(0.2, 0.3, 0.8),
        1,
        "observation",
    );
    let registry = TrustRegistry::new();
    let execution = ActionExecutionCoordinator::new();
    let retention = attest_retention_checkpoint(&state, 2, 1, &ToySigner).unwrap();
    let checkpoint = attest_authority_checkpoint(
        &state,
        &execution,
        &[],
        &registry,
        &retention,
        LedgerRevision::from_ledger(state.evidence_ledger()),
        2,
        &ToySigner,
    )
    .unwrap();

    let mut unrelated_state = state.clone();
    unrelated_state.update_from_observation_with_evidence(
        WisdomObservation::new(0.4, 0.2, 0.7),
        3,
        "other",
    );
    let unrelated = attest_retention_checkpoint(&unrelated_state, 4, 1, &ToySigner).unwrap();
    let mut restored = unrelated_state.clone();
    assert!(
        recover_authority_from_checkpoint(
            &mut restored,
            unrelated_state.evidence_ledger(),
            &checkpoint,
            &unrelated,
            &ToySigner,
            &ToySigner,
            &registry,
            &[],
            5,
        )
        .is_err()
    );
}
