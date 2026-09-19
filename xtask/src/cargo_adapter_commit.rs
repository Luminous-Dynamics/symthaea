use anyhow::bail;
use serde::Serialize;

use crate::cargo_adapter_postflight::{
    ObservationStage, PostflightPipelineReport, VerifiedCargoObservation,
};
use crate::cargo_adapter_state::{AdapterRunReport, PostflightCapture, ProcessTerminal};
use crate::cargo_execution_attempt::{
    CargoEffectPostflightState, CargoExecutionAttemptInput, CargoExecutionAttemptResult,
    CargoObservationState, CargoPostflightState, CargoTerminalState,
    ObservationNotApplicableReason, build_attempt_result,
};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Effect/Git evidence derived after an exact source-after subject is known.
/// Production integration must compute this from the admitted policy + exact
/// pre/post subjects rather than accept arbitrary caller assertions.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum CapturedSourceEffectOutcome {
    Evaluated {
        git_worktree_state_after: Option<String>,
        observed_diff_id: String,
        effect_evaluation_sha256: String,
        effect_allowed: bool,
    },
    Unavailable {
        git_worktree_state_after: Option<String>,
        reason: String,
    },
}

/// Boundary responsible for completing Git/diff/effect evidence once source
/// postflight has succeeded. The deterministic tests inject a fake; the real
/// adapter should map this to #4562 + #4520/#4524.
pub(crate) trait CapturedSourcePostflightBoundary {
    fn finalize(
        &mut self,
        intent: &CargoExecutionIntent,
        repository_source_after: &str,
    ) -> CapturedSourceEffectOutcome;
}

/// A pre-effect/effect-denied run intentionally has no backend-attempt result.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum AttemptCommitOutcome {
    NotCommitted {
        intent_id: String,
        reason: String,
    },
    Committed {
        result: CargoExecutionAttemptResult,
    },
}

/// Convert the complete deterministic adapter/postflight pipeline into the
/// canonical attempt-result contract.
///
/// Only `BackendEntered` can create an attempt result. Every committed result
/// must preserve exact intent lineage; no observation/source/effect values are
/// fabricated to fill a missing stage.
pub(crate) fn commit_pipeline_attempt<F>(
    mut intent: CargoExecutionIntent,
    pipeline: PostflightPipelineReport,
    finalizer: &mut F,
) -> anyhow::Result<AttemptCommitOutcome>
where
    F: CapturedSourcePostflightBoundary,
{
    validate_intent(&mut intent)?;
    let report_intent_id = adapter_intent_id(&pipeline.adapter);
    if report_intent_id != intent.intent_id {
        bail!(
            "adapter report intent {} does not match frozen intent {}",
            report_intent_id,
            intent.intent_id
        );
    }

    match pipeline.adapter {
        AdapterRunReport::RejectedBeforeEffect { reason, .. } => Ok(
            AttemptCommitOutcome::NotCommitted {
                intent_id: intent.intent_id,
                reason: format!("rejected before effect entry: {reason}"),
            },
        ),
        AdapterRunReport::EffectRejected {
            repository_source_before,
            reason,
            ..
        } => {
            require_source_before(&intent, &repository_source_before)?;
            Ok(AttemptCommitOutcome::NotCommitted {
                intent_id: intent.intent_id,
                reason: format!("effect entry denied: {reason}"),
            })
        }
        AdapterRunReport::BackendEntered {
            repository_source_before,
            effect_admission_digest,
            process,
            postflight,
            ..
        } => {
            require_source_before(&intent, &repository_source_before)?;
            let terminal = map_terminal(process.terminal.clone());
            let postflight = map_postflight(&intent, postflight, finalizer)?;
            let observation = map_observation(pipeline.observation, &process.terminal)?;
            let input = CargoExecutionAttemptInput {
                effect_admission_digest,
                terminal,
                stdout_sha256: process.stdout_sha256,
                stderr_sha256: process.stderr_sha256,
                postflight,
                observation,
            };
            let result = build_attempt_result(intent, input)?;
            Ok(AttemptCommitOutcome::Committed { result })
        }
    }
}

fn adapter_intent_id(report: &AdapterRunReport) -> &str {
    match report {
        AdapterRunReport::RejectedBeforeEffect { intent_id, .. }
        | AdapterRunReport::EffectRejected { intent_id, .. }
        | AdapterRunReport::BackendEntered { intent_id, .. } => intent_id,
    }
}

fn require_source_before(
    intent: &CargoExecutionIntent,
    repository_source_before: &str,
) -> anyhow::Result<()> {
    if repository_source_before != intent.repository_source_before {
        bail!(
            "adapter source-before {} does not match frozen intent {}",
            repository_source_before,
            intent.repository_source_before
        );
    }
    Ok(())
}

fn map_terminal(terminal: ProcessTerminal) -> CargoTerminalState {
    match terminal {
        ProcessTerminal::Exited { code } => CargoTerminalState::Exited { code },
        ProcessTerminal::SpawnFailed { error_digest } => {
            CargoTerminalState::SpawnFailed { error_digest }
        }
        ProcessTerminal::TimedOut => CargoTerminalState::TimedOut,
        ProcessTerminal::Cancelled => CargoTerminalState::Cancelled,
    }
}

fn map_postflight<F>(
    intent: &CargoExecutionIntent,
    postflight: PostflightCapture,
    finalizer: &mut F,
) -> anyhow::Result<CargoPostflightState>
where
    F: CapturedSourcePostflightBoundary,
{
    match postflight {
        PostflightCapture::CaptureFailed { error } => {
            Ok(CargoPostflightState::SourceCaptureFailed { reason: error })
        }
        PostflightCapture::Captured {
            repository_source_after,
        } => {
            let effect = match finalizer.finalize(intent, &repository_source_after) {
                CapturedSourceEffectOutcome::Evaluated {
                    git_worktree_state_after,
                    observed_diff_id,
                    effect_evaluation_sha256,
                    effect_allowed,
                } => CargoPostflightState::SourceCaptured {
                    repository_source_after,
                    git_worktree_state_after,
                    effect: CargoEffectPostflightState::Evaluated {
                        observed_diff_id,
                        effect_evaluation_sha256,
                        effect_allowed,
                    },
                },
                CapturedSourceEffectOutcome::Unavailable {
                    git_worktree_state_after,
                    reason,
                } => CargoPostflightState::SourceCaptured {
                    repository_source_after,
                    git_worktree_state_after,
                    effect: CargoEffectPostflightState::Unavailable { reason },
                },
            };
            Ok(effect)
        }
    }
}

fn map_observation(
    observation: ObservationStage,
    terminal: &ProcessTerminal,
) -> anyhow::Result<CargoObservationState> {
    match observation {
        ObservationStage::NotReached { reason } => {
            bail!("backend-entered report cannot commit NotReached observation: {reason}")
        }
        ObservationStage::NotApplicable { .. } => {
            if !matches!(terminal, ProcessTerminal::SpawnFailed { .. }) {
                bail!("NotApplicable observation requires SpawnFailed terminal state");
            }
            Ok(CargoObservationState::NotApplicable {
                reason: ObservationNotApplicableReason::CargoDidNotSpawn,
            })
        }
        ObservationStage::Unavailable { reason } => Ok(CargoObservationState::Unavailable {
            reason,
            transcript_sha256: None,
        }),
        ObservationStage::VerificationFailed { reason } => {
            Ok(CargoObservationState::VerificationFailed {
                reason,
                transcript_sha256: None,
            })
        }
        ObservationStage::Verified { observation } => Ok(map_verified(observation)),
    }
}

fn map_verified(observation: VerifiedCargoObservation) -> CargoObservationState {
    CargoObservationState::Verified {
        observation_id: observation.observation_id,
        context_id: observation.context_id,
        invocation_id: observation.invocation_id,
        transcript_sha256: observation.transcript_sha256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_postflight::{
        CargoObservationVerificationBoundary, ObservationVerificationOutcome,
        VerifiedCargoObservation, continue_with_observation,
    };
    use crate::cargo_adapter_state::{
        CargoProcessBackend, EffectAdmission, EffectEntryGate, EffectRejection,
        ProcessCapture, RepositorySourceProbe, ValidatedPreflightBindings, run_with_backend,
    };
    use crate::cargo_execution_attempt::{CargoEffectPostflightState, CargoPostflightState};
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

    struct Probe {
        pre: Option<anyhow::Result<String>>,
        post: Option<anyhow::Result<String>>,
    }
    impl RepositorySourceProbe for Probe {
        fn preflight_source_id(&mut self) -> anyhow::Result<String> {
            self.pre.take().unwrap()
        }
        fn postflight_source_id(&mut self) -> anyhow::Result<String> {
            self.post.take().unwrap()
        }
    }

    struct Gate {
        decision: Option<Result<EffectAdmission, EffectRejection>>,
    }
    impl EffectEntryGate for Gate {
        fn admit(&mut self, _intent: &CargoExecutionIntent) -> Result<EffectAdmission, EffectRejection> {
            self.decision.take().unwrap()
        }
    }

    struct Backend {
        capture: Option<ProcessCapture>,
    }
    impl CargoProcessBackend for Backend {
        fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
            self.capture.take().unwrap()
        }
    }

    struct Verifier {
        outcome: Option<ObservationVerificationOutcome>,
        calls: usize,
    }
    impl CargoObservationVerificationBoundary for Verifier {
        fn verify(
            &mut self,
            _intent: &CargoExecutionIntent,
            _process: &ProcessCapture,
        ) -> ObservationVerificationOutcome {
            self.calls += 1;
            self.outcome.take().unwrap()
        }
    }

    struct Finalizer {
        outcome: Option<CapturedSourceEffectOutcome>,
        calls: usize,
    }
    impl CapturedSourcePostflightBoundary for Finalizer {
        fn finalize(
            &mut self,
            _intent: &CargoExecutionIntent,
            _repository_source_after: &str,
        ) -> CapturedSourceEffectOutcome {
            self.calls += 1;
            self.outcome.take().unwrap()
        }
    }

    fn gate_allowed() -> Gate {
        Gate {
            decision: Some(Ok(EffectAdmission {
                receipt_digest: digest('2'),
            })),
        }
    }

    fn verified() -> ObservationVerificationOutcome {
        ObservationVerificationOutcome::Verified {
            observation: VerifiedCargoObservation {
                observation_id: digest('7'),
                context_id: digest('b'),
                invocation_id: digest('c'),
                transcript_sha256: digest('3'),
            },
        }
    }

    fn finalizer_evaluated() -> Finalizer {
        Finalizer {
            outcome: Some(CapturedSourceEffectOutcome::Evaluated {
                git_worktree_state_after: Some(digest('9')),
                observed_diff_id: digest('5'),
                effect_evaluation_sha256: digest('6'),
                effect_allowed: true,
            }),
            calls: 0,
        }
    }

    fn run_pipeline(
        terminal: ProcessTerminal,
        post: anyhow::Result<String>,
        observation: ObservationVerificationOutcome,
    ) -> PostflightPipelineReport {
        let frozen = intent();
        let mut probe = Probe {
            pre: Some(Ok(digest('a'))),
            post: Some(post),
        };
        let mut gate = gate_allowed();
        let mut backend = Backend {
            capture: Some(ProcessCapture {
                terminal,
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
        };
        let adapter = run_with_backend(
            frozen.clone(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut backend,
        )
        .unwrap();
        let mut verifier = Verifier {
            outcome: Some(observation),
            calls: 0,
        };
        continue_with_observation(frozen, adapter, &mut verifier).unwrap()
    }

    #[test]
    fn verified_exit_commits_one_canonical_attempt_result() {
        let pipeline = run_pipeline(
            ProcessTerminal::Exited { code: 0 },
            Ok(digest('a')),
            verified(),
        );
        let mut finalizer = finalizer_evaluated();
        let outcome = commit_pipeline_attempt(intent(), pipeline, &mut finalizer).unwrap();

        match outcome {
            AttemptCommitOutcome::Committed { result } => {
                assert!(matches!(result.terminal, CargoTerminalState::Exited { code: 0 }));
                assert!(matches!(result.observation, CargoObservationState::Verified { .. }));
                assert!(matches!(
                    result.postflight,
                    CargoPostflightState::SourceCaptured {
                        effect: CargoEffectPostflightState::Evaluated { .. },
                        ..
                    }
                ));
            }
            other => panic!("unexpected commit outcome: {other:?}"),
        }
        assert_eq!(finalizer.calls, 1);
    }

    #[test]
    fn spawn_failure_commits_without_fabricated_observation() {
        let frozen = intent();
        let mut probe = Probe {
            pre: Some(Ok(digest('a'))),
            post: Some(Ok(digest('a'))),
        };
        let mut gate = gate_allowed();
        let mut backend = Backend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::SpawnFailed {
                    error_digest: digest('8'),
                },
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
        };
        let adapter = run_with_backend(
            frozen.clone(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut backend,
        )
        .unwrap();
        let mut verifier = Verifier { outcome: None, calls: 0 };
        let pipeline = continue_with_observation(frozen, adapter, &mut verifier).unwrap();
        let mut finalizer = finalizer_evaluated();
        let outcome = commit_pipeline_attempt(intent(), pipeline, &mut finalizer).unwrap();

        assert_eq!(verifier.calls, 0);
        match outcome {
            AttemptCommitOutcome::Committed { result } => {
                assert!(matches!(result.terminal, CargoTerminalState::SpawnFailed { .. }));
                assert!(matches!(
                    result.observation,
                    CargoObservationState::NotApplicable { .. }
                ));
            }
            other => panic!("unexpected commit outcome: {other:?}"),
        }
    }

    #[test]
    fn source_postflight_failure_commits_without_fabricated_effects() {
        let pipeline = run_pipeline(
            ProcessTerminal::Exited { code: 1 },
            Err(anyhow::anyhow!("postflight probe failed")),
            ObservationVerificationOutcome::VerificationFailed {
                reason: "compiler transcript incomplete".into(),
            },
        );
        let mut finalizer = Finalizer { outcome: None, calls: 0 };
        let outcome = commit_pipeline_attempt(intent(), pipeline, &mut finalizer).unwrap();

        assert_eq!(finalizer.calls, 0);
        match outcome {
            AttemptCommitOutcome::Committed { result } => assert!(matches!(
                result.postflight,
                CargoPostflightState::SourceCaptureFailed { .. }
            )),
            other => panic!("unexpected commit outcome: {other:?}"),
        }
    }

    #[test]
    fn effect_evaluation_unavailable_keeps_known_source_subject() {
        let pipeline = run_pipeline(
            ProcessTerminal::Exited { code: 0 },
            Ok(digest('a')),
            verified(),
        );
        let mut finalizer = Finalizer {
            outcome: Some(CapturedSourceEffectOutcome::Unavailable {
                git_worktree_state_after: Some(digest('9')),
                reason: "effect evaluator unavailable".into(),
            }),
            calls: 0,
        };
        let outcome = commit_pipeline_attempt(intent(), pipeline, &mut finalizer).unwrap();

        match outcome {
            AttemptCommitOutcome::Committed { result } => assert!(matches!(
                result.postflight,
                CargoPostflightState::SourceCaptured {
                    effect: CargoEffectPostflightState::Unavailable { .. },
                    ..
                }
            )),
            other => panic!("unexpected commit outcome: {other:?}"),
        }
    }

    #[test]
    fn observation_verification_failure_still_commits_execution_evidence() {
        let pipeline = run_pipeline(
            ProcessTerminal::Exited { code: 1 },
            Ok(digest('a')),
            ObservationVerificationOutcome::VerificationFailed {
                reason: "transcript substitution".into(),
            },
        );
        let mut finalizer = finalizer_evaluated();
        let outcome = commit_pipeline_attempt(intent(), pipeline, &mut finalizer).unwrap();
        match outcome {
            AttemptCommitOutcome::Committed { result } => assert!(matches!(
                result.observation,
                CargoObservationState::VerificationFailed { .. }
            )),
            other => panic!("unexpected commit outcome: {other:?}"),
        }
    }

    #[test]
    fn pre_effect_rejection_never_mints_attempt_result() {
        let frozen = intent();
        let pipeline = PostflightPipelineReport {
            adapter: AdapterRunReport::RejectedBeforeEffect {
                intent_id: frozen.intent_id.clone(),
                reason: "stale source".into(),
            },
            observation: ObservationStage::NotReached {
                reason: "adapter rejected before effect entry".into(),
            },
        };
        let mut finalizer = Finalizer { outcome: None, calls: 0 };
        let outcome = commit_pipeline_attempt(frozen, pipeline, &mut finalizer).unwrap();
        assert!(matches!(outcome, AttemptCommitOutcome::NotCommitted { .. }));
        assert_eq!(finalizer.calls, 0);
    }

    #[test]
    fn intent_substitution_is_rejected_before_commit() {
        let mut frozen = intent();
        let pipeline = run_pipeline(
            ProcessTerminal::Exited { code: 0 },
            Ok(digest('a')),
            verified(),
        );
        frozen = build_intent(
            CargoExecutionIntentSpec {
                schema: "symthaea.cargo-execution-intent-input.v1".into(),
                git_worktree_state_before: Some(digest('9')),
                build_context_id: digest('b'),
                invocation_id: digest('c'),
                plan_id: Some(digest('0')),
                transaction_id: Some(digest('f')),
                adapter_semantics_digest: digest('1'),
            },
            &digest('a'),
            &digest('d'),
        )
        .unwrap();
        let mut finalizer = finalizer_evaluated();
        assert!(commit_pipeline_attempt(frozen, pipeline, &mut finalizer).is_err());
        assert_eq!(finalizer.calls, 0);
    }
}
