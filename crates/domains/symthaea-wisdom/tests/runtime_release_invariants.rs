// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::convert::Infallible;

use symthaea_wisdom::{
    ActionDispatchError, ActionEthicsContext, ActionExecution, ActionExecutionCoordinator,
    ActionExecutor, ActionId, ActionRequest, ActorId, AttestationStatus, AuthorizationStatus,
    CapabilityId, CapabilityScope, ConsentStatus, DeceptionStatus, DeclaredWisdomClaims,
    DeploymentAttestationStatus, DeploymentManifest, DeploymentReleaseError, EvidenceEvent,
    EvidenceSigner, EvidenceVerifier, ExecutionAttemptId, ExecutionRecoveryStatus, ExecutorResult,
    PermitUseError, ProcessId, ReleaseEvidenceRequirements, ReleaseGateFinding,
    ReleaseRuntimeEvidence, ResourceId, RuntimeEventEnvelope, RuntimeIntegration,
    RuntimeIntegrationError, RuntimeProcessEvent, StructuralEthicsPolicy, WisdomState,
    attest_deployment_release, evaluate_release_evidence_with_runtime, fingerprint_bytes,
    recover_execution_journal, replay_operational_evidence, verify_deployment_release,
};

#[derive(Default)]
struct CountingExecutor {
    calls: u64,
}

impl ActionExecutor for CountingExecutor {
    fn execute(&mut self, _execution: ActionExecution) -> ExecutorResult {
        self.calls = self.calls.saturating_add(1);
        ExecutorResult::succeeded(self.calls)
    }
}

struct TestDeploymentIdentity;

impl TestDeploymentIdentity {
    fn signature(message: &[u8]) -> Vec<u8> {
        fingerprint_bytes(message).to_le_bytes().to_vec()
    }
}

impl EvidenceSigner for TestDeploymentIdentity {
    type Error = Infallible;

    fn algorithm(&self) -> &str {
        "test-only-fingerprint"
    }

    fn key_id(&self) -> &str {
        "series-iv-test-key"
    }

    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Self::Error> {
        Ok(Self::signature(message))
    }
}

impl EvidenceVerifier for TestDeploymentIdentity {
    type Error = Infallible;

    fn algorithm(&self) -> &str {
        "test-only-fingerprint"
    }

    fn key_id(&self) -> &str {
        "series-iv-test-key"
    }

    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, Self::Error> {
        Ok(Self::signature(message) == signature)
    }
}

fn safe_request(action: u64) -> ActionRequest {
    ActionRequest::new(
        ActionId::new(action),
        ActorId::new(20),
        CapabilityScope::new(CapabilityId::new(30), ResourceId::new(40)),
        1_000,
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
fn blocked_request_never_crosses_executor_boundary() {
    let mut request = safe_request(1);
    request.ethics.authorization = AuthorizationStatus::Unauthorized;

    let mut wisdom = WisdomState::new();
    let mut coordinator = ActionExecutionCoordinator::new();
    let mut executor = CountingExecutor::default();
    let outcome = coordinator
        .authorize_and_execute(&mut wisdom, request, &mut executor, 10, "scheduler")
        .unwrap();

    assert!(!outcome.executed());
    assert_eq!(executor.calls, 0);
    assert!(
        wisdom.evidence_ledger().records().iter().all(|record| {
            !matches!(record.event, EvidenceEvent::ActionExecutionStarted { .. })
        })
    );
}

#[test]
fn recovered_in_doubt_action_is_never_automatically_retried() {
    let mut wisdom = WisdomState::new();
    let request = safe_request(7);
    let fingerprint = wisdom.config_fingerprint();
    wisdom.evidence_ledger_mut().append(
        10,
        "executor",
        fingerprint,
        EvidenceEvent::ActionExecutionStarted {
            attempt: ExecutionAttemptId::new(2),
            nonce: 3,
            action: request.action,
            actor: request.actor,
            capability: request.scope.capability,
            resource: request.scope.resource,
        },
    );

    let recovery = recover_execution_journal(wisdom.evidence_ledger()).unwrap();
    assert_eq!(recovery.in_doubt_count(), 1);
    let mut coordinator = ActionExecutionCoordinator::from_recovery(&recovery).unwrap();
    let mut executor = CountingExecutor::default();
    let error = coordinator
        .authorize_and_execute(&mut wisdom, request, &mut executor, 11, "scheduler")
        .unwrap_err();

    assert_eq!(
        error,
        ActionDispatchError::DuplicateAction {
            action: request.action
        }
    );
    assert_eq!(executor.calls, 0);
}

#[test]
fn runtime_cursor_survives_restart_and_rejects_replay() {
    let mut wisdom = WisdomState::new();
    let mut runtime = RuntimeIntegration::new("scheduler");
    runtime
        .apply(
            &mut wisdom,
            RuntimeEventEnvelope::new(
                0,
                10,
                RuntimeProcessEvent::Register {
                    process: ProcessId::new(1),
                    label: "planner".to_owned(),
                },
            ),
        )
        .unwrap();

    let mut recovered = RuntimeIntegration::recover("scheduler", wisdom.evidence_ledger()).unwrap();
    assert_eq!(recovered.expected_sequence(), 1);
    let replay_error = recovered
        .apply(
            &mut wisdom,
            RuntimeEventEnvelope::new(
                0,
                11,
                RuntimeProcessEvent::Register {
                    process: ProcessId::new(2),
                    label: "executor".to_owned(),
                },
            ),
        )
        .unwrap_err();
    assert_eq!(
        replay_error,
        RuntimeIntegrationError::UnexpectedSequence {
            expected: 1,
            actual: 0,
        }
    );

    recovered
        .apply(
            &mut wisdom,
            RuntimeEventEnvelope::new(
                1,
                11,
                RuntimeProcessEvent::Register {
                    process: ProcessId::new(2),
                    label: "executor".to_owned(),
                },
            ),
        )
        .unwrap();
    assert_eq!(recovered.expected_sequence(), 2);
}

#[test]
fn policy_change_revokes_preexisting_authority() {
    let mut wisdom = WisdomState::new();
    let request = safe_request(9);
    let issue = wisdom
        .issue_action_permit_with_evidence(request, 10, "scheduler")
        .unwrap();
    let permit = issue.issue.permit.unwrap();

    let mut changed_policy = StructuralEthicsPolicy::default();
    changed_policy.max_expected_harm = 0.005;
    let revocations =
        wisdom.set_ethics_policy_with_evidence(changed_policy, 11, "policy-controller");
    assert_eq!(revocations.len(), 1);

    let use_evidence = wisdom.consume_action_permit_with_evidence(
        permit,
        ActionExecution::new(request.action, request.actor, request.scope),
        12,
        "scheduler",
    );
    assert_eq!(use_evidence.result, Err(PermitUseError::PolicyChanged));
}

#[test]
fn deployment_identity_and_execution_recovery_jointly_gate_release() {
    let wisdom = WisdomState::new();
    let replay = replay_operational_evidence(
        wisdom.evidence_ledger(),
        wisdom.config(),
        Default::default(),
    )
    .unwrap();
    let declared = DeclaredWisdomClaims::none();
    let manifest = DeploymentManifest::new(
        "deployment-1",
        "build-1",
        "production",
        "sha256",
        vec![1; 32],
        wisdom.config_fingerprint(),
        wisdom.ethics_policy().fingerprint(),
        10,
    )
    .unwrap();
    let attestation = attest_deployment_release(
        &manifest,
        &wisdom,
        &replay.report,
        declared,
        &TestDeploymentIdentity,
    )
    .unwrap();
    verify_deployment_release(
        &manifest,
        &wisdom,
        &replay.report,
        declared,
        &attestation,
        &TestDeploymentIdentity,
    )
    .unwrap();
    let recovery = recover_execution_journal(wisdom.evidence_ledger()).unwrap();

    let mut requirements = ReleaseEvidenceRequirements::hardened_runtime();
    requirements.minimum_records = 0;
    requirements.require_attestation = false;
    let report = evaluate_release_evidence_with_runtime(
        &wisdom,
        declared,
        AttestationStatus::NotRequired,
        Some(&replay.report),
        requirements,
        ReleaseRuntimeEvidence {
            deployment_attestation: DeploymentAttestationStatus::Verified,
            execution_recovery: ExecutionRecoveryStatus::Verified(&recovery),
        },
    );
    assert!(report.is_ready(), "{:?}", report.findings);

    let mut altered_manifest = manifest;
    altered_manifest.artifact_digest[0] ^= 0xff;
    assert_eq!(
        verify_deployment_release(
            &altered_manifest,
            &wisdom,
            &replay.report,
            declared,
            &attestation,
            &TestDeploymentIdentity,
        ),
        Err(DeploymentReleaseError::ManifestFingerprintMismatch)
    );
}

#[test]
fn stale_execution_recovery_is_visible_at_release_boundary() {
    let mut wisdom = WisdomState::new();
    let recovery = recover_execution_journal(wisdom.evidence_ledger()).unwrap();
    let fingerprint = wisdom.config_fingerprint();
    wisdom.evidence_ledger_mut().append(
        1,
        "late-record",
        fingerprint,
        EvidenceEvent::PredictionObserved {
            ticket_id: 1,
            actual_error: 0.2,
        },
    );

    let mut requirements = ReleaseEvidenceRequirements::hardened_runtime();
    requirements.minimum_records = 0;
    requirements.require_attestation = false;
    requirements.require_operational_replay = false;
    requirements.require_deployment_attestation = false;
    let report = evaluate_release_evidence_with_runtime(
        &wisdom,
        DeclaredWisdomClaims::none(),
        AttestationStatus::NotRequired,
        None,
        requirements,
        ReleaseRuntimeEvidence {
            deployment_attestation: DeploymentAttestationStatus::NotRequired,
            execution_recovery: ExecutionRecoveryStatus::Verified(&recovery),
        },
    );
    assert!(report.findings.iter().any(|finding| matches!(
        finding,
        ReleaseGateFinding::ExecutionRecoveryRecordCountMismatch { .. }
            | ReleaseGateFinding::ExecutionRecoveryLedgerMismatch { .. }
    )));
}
