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
#[path = "../src/cargo_adapter_postflight.rs"]
mod cargo_adapter_postflight;
#[path = "../src/cargo_execution_attempt.rs"]
mod cargo_execution_attempt;
#[path = "../src/cargo_adapter_commit.rs"]
mod cargo_adapter_commit;
#[path = "../src/autonomous_cargo_adapter_state.rs"]
mod autonomous_cargo_adapter_state;
#[path = "../src/autonomous_cargo_effect_binding.rs"]
mod autonomous_cargo_effect_binding;
#[path = "../src/autonomous_cargo_freshness_binding.rs"]
mod autonomous_cargo_freshness_binding;
#[path = "../src/autonomous_cargo_persistence_binding.rs"]
mod autonomous_cargo_persistence_binding;
#[path = "../src/autonomous_cargo_attempt_commit.rs"]
mod autonomous_cargo_attempt_commit;

use autonomous_cargo_adapter_state::AutonomousAdapterRunReport;
use autonomous_cargo_attempt_commit::{
    AutonomousAttemptCommitOutcome, commit_persistence_bound_autonomous_attempt,
};
use autonomous_cargo_effect_binding::{
    CommitmentAwareEffectAdmission, CommitmentBoundAutonomousRunReport,
    ExpectedEffectAdmissionBinding,
};
use autonomous_cargo_freshness_binding::{
    CanonicalFreshnessV2Evidence, FreshnessBoundAutonomousRunReport,
};
use autonomous_cargo_persistence_binding::{
    CanonicalAdmissionPersistenceEvidence, ExpectedAdmissionPersistenceBinding,
    PersistenceBoundAutonomousRunReport,
};
use cargo_adapter_commit::{CapturedSourceEffectOutcome, CapturedSourcePostflightBoundary};
use cargo_adapter_postflight::{
    CargoObservationVerificationBoundary, ObservationVerificationOutcome,
    VerifiedCargoObservation,
};
use cargo_adapter_state::{AdapterRunReport, PostflightCapture, ProcessCapture, ProcessTerminal};
use cargo_execution_attempt::{CargoObservationState, CargoTerminalState};
use cargo_execution_contract::{CargoExecutionIntent, CargoExecutionIntentSpec, build_intent};

fn digest(seed: char) -> String {
    assert!(seed.is_ascii());
    format!("{:02x}", u32::from(seed)).repeat(32)
}

fn intent() -> CargoExecutionIntent {
    build_intent(
        CargoExecutionIntentSpec {
            schema: "symthaea.cargo-execution-intent-input.v1".into(),
            git_worktree_state_before: Some(digest('6')),
            build_context_id: digest('2'),
            invocation_id: digest('3'),
            plan_id: Some(digest('7')),
            transaction_id: Some(digest('8')),
            adapter_semantics_digest: digest('5'),
        },
        &digest('1'),
        &digest('4'),
    )
    .unwrap()
}

fn effect_expectation() -> ExpectedEffectAdmissionBinding {
    ExpectedEffectAdmissionBinding {
        execution_intent_id: intent().intent_id,
        eligibility_id: digest('a'),
        prepared_intent_id: digest('b'),
        action_binding: digest('c'),
        authority_snapshot_digest: digest('d'),
        adapter_semantics_digest: digest('5'),
        effect_admission_commitment_digest: digest('e'),
    }
}

fn persistence_expectation() -> ExpectedAdmissionPersistenceBinding {
    ExpectedAdmissionPersistenceBinding {
        authority_snapshot_digest: digest('d'),
        required_persistence_profile_id: digest('p'),
    }
}

fn admission() -> CommitmentAwareEffectAdmission {
    let expected = effect_expectation();
    CommitmentAwareEffectAdmission {
        receipt_digest: digest('f'),
        action_binding: expected.action_binding,
        authority_snapshot_digest: expected.authority_snapshot_digest,
        adapter_semantics_digest: expected.adapter_semantics_digest,
        effect_admission_commitment_digest: expected.effect_admission_commitment_digest,
    }
}

fn persistence() -> CanonicalAdmissionPersistenceEvidence {
    CanonicalAdmissionPersistenceEvidence {
        persistence_ack_digest: digest('9'),
        persistence_profile_id: digest('p'),
        persistence_record_id: "record-0001".into(),
        control_plane_id: "control-plane-alpha".into(),
        control_plane_generation: 42,
        evidence_root_digest: Some(digest('q')),
        anti_rollback_evidence_digest: Some(digest('r')),
    }
}

fn freshness() -> CanonicalFreshnessV2Evidence {
    CanonicalFreshnessV2Evidence {
        freshness_receipt_id: digest('0'),
        prepared_intent_id: digest('b'),
        runtime_binding_id: digest('g'),
        materialization_receipt_id: digest('h'),
        tool_attestation_id: digest('i'),
        git_worktree_state_id: digest('6'),
    }
}

fn backend_report() -> AdapterRunReport {
    AdapterRunReport::BackendEntered {
        intent_id: intent().intent_id,
        repository_source_before: digest('1'),
        effect_admission_digest: digest('f'),
        persistence_ack_digest: digest('9'),
        pre_spawn_freshness_digest: digest('0'),
        process: ProcessCapture {
            terminal: ProcessTerminal::Exited { code: 0 },
            stdout_sha256: digest('u'),
            stderr_sha256: digest('v'),
        },
        postflight: PostflightCapture::Captured {
            repository_source_after: digest('1'),
        },
    }
}

fn full_report() -> PersistenceBoundAutonomousRunReport {
    let frozen = intent();
    let expected = effect_expectation();
    PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
        intent_id: frozen.intent_id.clone(),
        accepted_persistence: Some(persistence()),
        inner: FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            intent_id: frozen.intent_id.clone(),
            accepted_freshness: Some(freshness()),
            inner: CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                intent_id: frozen.intent_id.clone(),
                expected: expected.clone(),
                accepted_admission: Some(admission()),
                inner: AutonomousAdapterRunReport::AfterEligibility {
                    intent_id: frozen.intent_id,
                    eligibility_id: expected.eligibility_id,
                    prepared_intent_id: expected.prepared_intent_id,
                    inner: backend_report(),
                },
            },
        },
    }
}

struct Verifier {
    calls: usize,
}

impl CargoObservationVerificationBoundary for Verifier {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
        _process: &ProcessCapture,
    ) -> ObservationVerificationOutcome {
        self.calls += 1;
        ObservationVerificationOutcome::Verified {
            observation: VerifiedCargoObservation {
                observation_id: digest('w'),
                context_id: digest('2'),
                invocation_id: digest('3'),
                transcript_sha256: digest('u'),
            },
        }
    }
}

struct Finalizer {
    calls: usize,
}

impl CapturedSourcePostflightBoundary for Finalizer {
    fn finalize(
        &mut self,
        _intent: &CargoExecutionIntent,
        _repository_source_after: &str,
    ) -> CapturedSourceEffectOutcome {
        self.calls += 1;
        CapturedSourceEffectOutcome::Evaluated {
            git_worktree_state_after: Some(digest('6')),
            observed_diff_id: digest('x'),
            effect_evaluation_sha256: digest('y'),
            effect_allowed: true,
        }
    }
}

fn no_calls() -> (Verifier, Finalizer) {
    (Verifier { calls: 0 }, Finalizer { calls: 0 })
}

#[test]
fn exact_full_lineage_commits_canonical_attempt() {
    let mut verifier = Verifier { calls: 0 };
    let mut finalizer = Finalizer { calls: 0 };
    let outcome = commit_persistence_bound_autonomous_attempt(
        intent(),
        persistence_expectation(),
        full_report(),
        &mut verifier,
        &mut finalizer,
    )
    .unwrap();

    assert_eq!(verifier.calls, 1);
    assert_eq!(finalizer.calls, 1);
    match outcome {
        AutonomousAttemptCommitOutcome::Committed { result, report, .. } => {
            assert!(matches!(result.terminal, CargoTerminalState::Exited { code: 0 }));
            assert!(matches!(result.observation, CargoObservationState::Verified { .. }));
            assert!(matches!(
                report,
                PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { .. }
            ));
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[test]
fn outer_pre_eligibility_rejection_never_reaches_observation_or_finalizer() {
    let frozen = intent();
    let report =
        PersistenceBoundAutonomousRunReport::PersistenceExpectationRejectedBeforeEligibility {
            intent_id: frozen.intent_id,
            effect_authority_snapshot_digest: digest('d'),
            persistence_authority_snapshot_digest: digest('z'),
            reason: "authority snapshot mismatch".into(),
        };
    let (mut verifier, mut finalizer) = no_calls();
    let outcome = commit_persistence_bound_autonomous_attempt(
        intent(),
        persistence_expectation(),
        report,
        &mut verifier,
        &mut finalizer,
    )
    .unwrap();
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
    assert!(matches!(
        outcome,
        AutonomousAttemptCommitOutcome::NotCommitted { .. }
    ));
}

#[test]
fn missing_accepted_freshness_on_backend_path_fails_before_observation() {
    let mut report = full_report();
    if let PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { inner, .. } = &mut report
    {
        if let FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness,
            ..
        } = inner
        {
            *accepted_freshness = None;
        }
    }
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            persistence_expectation(),
            report,
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}

#[test]
fn admission_receipt_substitution_fails_before_observation() {
    let mut report = full_report();
    if let PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { inner, .. } = &mut report
    {
        if let FreshnessBoundAutonomousRunReport::AfterFreshnessBinding { inner, .. } = inner {
            if let CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                accepted_admission,
                ..
            } = inner
            {
                accepted_admission.as_mut().unwrap().receipt_digest = digest('j');
            }
        }
    }
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            persistence_expectation(),
            report,
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}

#[test]
fn persistence_ack_substitution_fails_before_observation() {
    let mut report = full_report();
    if let PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
        accepted_persistence,
        ..
    } = &mut report
    {
        accepted_persistence.as_mut().unwrap().persistence_ack_digest = digest('k');
    }
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            persistence_expectation(),
            report,
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}

#[test]
fn freshness_receipt_substitution_fails_before_observation() {
    let mut report = full_report();
    if let PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { inner, .. } = &mut report
    {
        if let FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness,
            ..
        } = inner
        {
            accepted_freshness.as_mut().unwrap().freshness_receipt_id = digest('l');
        }
    }
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            persistence_expectation(),
            report,
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}

#[test]
fn required_persistence_profile_substitution_fails_before_observation() {
    let mut expected = persistence_expectation();
    expected.required_persistence_profile_id = digest('m');
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            expected,
            full_report(),
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}

#[test]
fn impossible_outer_rejection_wrapping_backend_is_rejected() {
    let full = full_report();
    let inner = match full {
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { inner, .. } => inner,
        _ => unreachable!(),
    };
    let report = PersistenceBoundAutonomousRunReport::PersistenceBindingRejected {
        intent_id: intent().intent_id,
        required_persistence_profile_id: digest('p'),
        actual: persistence(),
        reason: "synthetic impossible state".into(),
        inner,
    };
    let (mut verifier, mut finalizer) = no_calls();
    assert!(
        commit_persistence_bound_autonomous_attempt(
            intent(),
            persistence_expectation(),
            report,
            &mut verifier,
            &mut finalizer,
        )
        .is_err()
    );
    assert_eq!(verifier.calls, 0);
    assert_eq!(finalizer.calls, 0);
}
