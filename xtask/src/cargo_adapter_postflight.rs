use anyhow::bail;
use serde::Serialize;

use crate::cargo_adapter_state::{AdapterRunReport, ProcessCapture, ProcessTerminal};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// A transcript-backed Cargo observation that has already passed the external
/// verifier boundary (production mapping: #4554).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct VerifiedCargoObservation {
    pub observation_id: String,
    pub context_id: String,
    pub invocation_id: String,
    pub transcript_sha256: String,
}

/// Observation verification result returned by the injected boundary.
/// Missing transcript material and failed verification remain distinct evidence.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum ObservationVerificationOutcome {
    Verified { observation: VerifiedCargoObservation },
    Unavailable { reason: String },
    VerificationFailed { reason: String },
}

/// Production integration should be backed by #4554 and the exact transcript
/// captured by the trusted Cargo adapter. This layer never mints observations.
pub(crate) trait CargoObservationVerificationBoundary {
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
        process: &ProcessCapture,
    ) -> ObservationVerificationOutcome;
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum ObservationStage {
    NotReached { reason: String },
    NotApplicable { reason: String },
    Unavailable { reason: String },
    VerificationFailed { reason: String },
    Verified { observation: VerifiedCargoObservation },
}

/// Preserves all adapter evidence even when observation verification fails.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct PostflightPipelineReport {
    pub adapter: AdapterRunReport,
    pub observation: ObservationStage,
}

/// Extend an adapter report through conditional observation verification.
///
/// Canonical result commitment is deliberately deferred until #4633 makes
/// observation presence terminal-sensitive. `SpawnFailed` can never be forced
/// through a fabricated Cargo observation.
pub(crate) fn continue_with_observation<V>(
    mut intent: CargoExecutionIntent,
    adapter: AdapterRunReport,
    verifier: &mut V,
) -> anyhow::Result<PostflightPipelineReport>
where
    V: CargoObservationVerificationBoundary,
{
    validate_intent(&mut intent)?;

    let observation = match &adapter {
        AdapterRunReport::RejectedBeforeEffect { .. } => ObservationStage::NotReached {
            reason: "adapter rejected before effect entry".into(),
        },
        AdapterRunReport::EffectRejected { .. } => ObservationStage::NotReached {
            reason: "effect entry was denied".into(),
        },
        AdapterRunReport::BackendEntered { process, .. } => match &process.terminal {
            ProcessTerminal::SpawnFailed { .. } => ObservationStage::NotApplicable {
                reason: "Cargo process did not spawn; no Cargo transcript can truthfully exist".into(),
            },
            ProcessTerminal::Exited { .. }
            | ProcessTerminal::TimedOut
            | ProcessTerminal::Cancelled => match verifier.verify(&intent, process) {
                ObservationVerificationOutcome::Unavailable { reason } => {
                    ObservationStage::Unavailable { reason }
                }
                ObservationVerificationOutcome::VerificationFailed { reason } => {
                    ObservationStage::VerificationFailed { reason }
                }
                ObservationVerificationOutcome::Verified { mut observation } => {
                    normalize_verified_observation(&mut observation)?;
                    if observation.context_id != intent.build_context_id {
                        ObservationStage::VerificationFailed {
                            reason: format!(
                                "verified observation context {} does not match intent {}",
                                observation.context_id, intent.build_context_id
                            ),
                        }
                    } else if observation.invocation_id != intent.invocation_id {
                        ObservationStage::VerificationFailed {
                            reason: format!(
                                "verified observation invocation {} does not match intent {}",
                                observation.invocation_id, intent.invocation_id
                            ),
                        }
                    } else if observation.transcript_sha256 != process.stdout_sha256 {
                        ObservationStage::VerificationFailed {
                            reason: format!(
                                "verified observation transcript {} does not match captured stdout {}",
                                observation.transcript_sha256, process.stdout_sha256
                            ),
                        }
                    } else {
                        ObservationStage::Verified { observation }
                    }
                }
            },
        },
    };

    Ok(PostflightPipelineReport {
        adapter,
        observation,
    })
}

fn normalize_verified_observation(
    observation: &mut VerifiedCargoObservation,
) -> anyhow::Result<()> {
    normalize_digest("observation.observation_id", &mut observation.observation_id)?;
    normalize_digest("observation.context_id", &mut observation.context_id)?;
    normalize_digest("observation.invocation_id", &mut observation.invocation_id)?;
    normalize_digest(
        "observation.transcript_sha256",
        &mut observation.transcript_sha256,
    )?;
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_state::{
        CargoProcessBackend, EffectAdmission, EffectEntryGate, EffectRejection,
        PostflightCapture, RepositorySourceProbe, ValidatedPreflightBindings, run_with_backend,
    };
    use crate::cargo_execution_contract::{CargoExecutionIntentSpec, build_intent};

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn intent() -> CargoExecutionIntent {
        build_intent(
            CargoExecutionIntentSpec {
                schema: "symthaea.cargo-execution-intent-input.v1".into(),
                git_worktree_state_before: Some(digest('9')),
                build_context_id: digest('b'),
                invocation_id: digest('c'),
                plan_id: Some(digest('e')),
                transaction_id: Some(digest('f')),
                adapter_semantics_digest: digest('1'),
            },
            &digest('a'),
            &digest('d'),
        )
        .unwrap()
    }

    fn bindings() -> ValidatedPreflightBindings {
        ValidatedPreflightBindings {
            build_context_id: digest('b'),
            invocation_id: digest('c'),
            effect_policy_id: digest('d'),
            git_worktree_state_id: Some(digest('9')),
        }
    }

    struct FakeProbe {
        pre: Option<anyhow::Result<String>>,
        post: Option<anyhow::Result<String>>,
    }

    impl RepositorySourceProbe for FakeProbe {
        fn preflight_source_id(&mut self) -> anyhow::Result<String> {
            self.pre.take().expect("pre source configured")
        }

        fn postflight_source_id(&mut self) -> anyhow::Result<String> {
            self.post.take().expect("post source configured")
        }
    }

    struct FakeGate {
        decision: Option<Result<EffectAdmission, EffectRejection>>,
    }

    impl EffectEntryGate for FakeGate {
        fn admit(
            &mut self,
            _intent: &CargoExecutionIntent,
        ) -> Result<EffectAdmission, EffectRejection> {
            self.decision.take().expect("effect decision configured")
        }
    }

    struct FakeBackend {
        capture: Option<ProcessCapture>,
    }

    impl CargoProcessBackend for FakeBackend {
        fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
            self.capture.take().expect("process capture configured")
        }
    }

    struct FakeVerifier {
        outcome: Option<ObservationVerificationOutcome>,
        calls: usize,
    }

    impl CargoObservationVerificationBoundary for FakeVerifier {
        fn verify(
            &mut self,
            _intent: &CargoExecutionIntent,
            _process: &ProcessCapture,
        ) -> ObservationVerificationOutcome {
            self.calls += 1;
            self.outcome.take().expect("verification outcome configured")
        }
    }

    fn admitted_gate() -> FakeGate {
        FakeGate {
            decision: Some(Ok(EffectAdmission {
                receipt_digest: digest('2'),
            })),
        }
    }

    fn report_for(terminal: ProcessTerminal) -> AdapterRunReport {
        let mut probe = FakeProbe {
            pre: Some(Ok(digest('a'))),
            post: Some(Ok(digest('a'))),
        };
        let mut gate = admitted_gate();
        let mut backend = FakeBackend {
            capture: Some(ProcessCapture {
                terminal,
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
        };
        run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend).unwrap()
    }

    fn verified() -> ObservationVerificationOutcome {
        ObservationVerificationOutcome::Verified {
            observation: VerifiedCargoObservation {
                observation_id: digest('5'),
                context_id: digest('b'),
                invocation_id: digest('c'),
                transcript_sha256: digest('3'),
            },
        }
    }

    #[test]
    fn exited_process_can_produce_verified_observation() {
        let adapter = report_for(ProcessTerminal::Exited { code: 0 });
        let mut verifier = FakeVerifier {
            outcome: Some(verified()),
            calls: 0,
        };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert_eq!(verifier.calls, 1);
        assert!(matches!(report.observation, ObservationStage::Verified { .. }));
    }

    #[test]
    fn spawn_failure_never_calls_observation_verifier() {
        let adapter = report_for(ProcessTerminal::SpawnFailed {
            error_digest: digest('6'),
        });
        let mut verifier = FakeVerifier { outcome: None, calls: 0 };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert_eq!(verifier.calls, 0);
        assert!(matches!(
            report.observation,
            ObservationStage::NotApplicable { .. }
        ));
    }

    #[test]
    fn unavailable_timeout_transcript_is_explicit_not_fabricated() {
        let adapter = report_for(ProcessTerminal::TimedOut);
        let mut verifier = FakeVerifier {
            outcome: Some(ObservationVerificationOutcome::Unavailable {
                reason: "partial transcript unavailable".into(),
            }),
            calls: 0,
        };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert_eq!(verifier.calls, 1);
        assert!(matches!(report.observation, ObservationStage::Unavailable { .. }));
    }

    #[test]
    fn verifier_failure_retains_backend_and_postflight_evidence() {
        let adapter = report_for(ProcessTerminal::Exited { code: 1 });
        let mut verifier = FakeVerifier {
            outcome: Some(ObservationVerificationOutcome::VerificationFailed {
                reason: "transcript substitution".into(),
            }),
            calls: 0,
        };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert!(matches!(
            report.observation,
            ObservationStage::VerificationFailed { .. }
        ));
        match report.adapter {
            AdapterRunReport::BackendEntered { process, postflight, .. } => {
                assert_eq!(process.terminal, ProcessTerminal::Exited { code: 1 });
                assert!(matches!(postflight, PostflightCapture::Captured { .. }));
            }
            other => panic!("unexpected adapter report: {other:?}"),
        }
    }

    #[test]
    fn observation_context_substitution_becomes_typed_failure() {
        let adapter = report_for(ProcessTerminal::Exited { code: 0 });
        let mut bad = match verified() {
            ObservationVerificationOutcome::Verified { observation } => observation,
            _ => unreachable!(),
        };
        bad.context_id = digest('7');
        let mut verifier = FakeVerifier {
            outcome: Some(ObservationVerificationOutcome::Verified { observation: bad }),
            calls: 0,
        };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert!(matches!(
            report.observation,
            ObservationStage::VerificationFailed { .. }
        ));
    }

    #[test]
    fn transcript_substitution_becomes_typed_failure() {
        let adapter = report_for(ProcessTerminal::Exited { code: 0 });
        let mut bad = match verified() {
            ObservationVerificationOutcome::Verified { observation } => observation,
            _ => unreachable!(),
        };
        bad.transcript_sha256 = digest('8');
        let mut verifier = FakeVerifier {
            outcome: Some(ObservationVerificationOutcome::Verified { observation: bad }),
            calls: 0,
        };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert!(matches!(
            report.observation,
            ObservationStage::VerificationFailed { .. }
        ));
    }

    #[test]
    fn pre_effect_rejection_does_not_reach_observation_boundary() {
        let adapter = AdapterRunReport::RejectedBeforeEffect {
            intent_id: intent().intent_id,
            reason: "stale source".into(),
        };
        let mut verifier = FakeVerifier { outcome: None, calls: 0 };
        let report = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
        assert_eq!(verifier.calls, 0);
        assert!(matches!(report.observation, ObservationStage::NotReached { .. }));
    }
}
