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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct PostflightPipelineReport {
    pub adapter: AdapterRunReport,
    pub observation: ObservationStage,
}

/// Extend adapter evidence through conditional Cargo-observation verification.
///
/// Every pre-backend rejection remains `NotReached`. `SpawnFailed` is
/// `NotApplicable`, because a Cargo transcript cannot truthfully exist.
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
        AdapterRunReport::AdmissionPersistenceRejected { .. } => ObservationStage::NotReached {
            reason: "admission persistence rejected before backend entry".into(),
        },
        AdapterRunReport::FreshnessRejected { .. } => ObservationStage::NotReached {
            reason: "pre-spawn freshness rejected before backend entry".into(),
        },
        AdapterRunReport::BackendEntered { process, .. } => match &process.terminal {
            ProcessTerminal::SpawnFailed { .. } => ObservationStage::NotApplicable {
                reason: "Cargo process did not spawn; no Cargo transcript can truthfully exist"
                    .into(),
            },
            ProcessTerminal::Exited { .. }
            | ProcessTerminal::TimedOut
            | ProcessTerminal::Cancelled => verify_observation(&intent, process, verifier)?,
        },
    };

    Ok(PostflightPipelineReport {
        adapter,
        observation,
    })
}

fn verify_observation<V>(
    intent: &CargoExecutionIntent,
    process: &ProcessCapture,
    verifier: &mut V,
) -> anyhow::Result<ObservationStage>
where
    V: CargoObservationVerificationBoundary,
{
    Ok(match verifier.verify(intent, process) {
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
        AdmissionPersistence, AdmissionPersistenceAck, AdmissionPersistenceRejection,
        CargoProcessBackend, EffectAdmission, EffectEntryGate, EffectRejection,
        PostflightCapture, PreSpawnFreshnessApproval, PreSpawnFreshnessGate,
        PreSpawnFreshnessRejection, RepositorySourceProbe, ValidatedPreflightBindings,
        run_with_backend,
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

    struct Probe(Option<anyhow::Result<String>>, Option<anyhow::Result<String>>);
    impl RepositorySourceProbe for Probe {
        fn preflight_source_id(&mut self) -> anyhow::Result<String> {
            self.0.take().unwrap()
        }
        fn postflight_source_id(&mut self) -> anyhow::Result<String> {
            self.1.take().unwrap()
        }
    }

    struct Gate(Option<Result<EffectAdmission, EffectRejection>>);
    impl EffectEntryGate for Gate {
        fn admit(
            &mut self,
            _intent: &CargoExecutionIntent,
        ) -> Result<EffectAdmission, EffectRejection> {
            self.0.take().unwrap()
        }
    }

    struct Persistence;
    impl AdmissionPersistence for Persistence {
        fn persist(
            &mut self,
            _intent: &CargoExecutionIntent,
            admission: &EffectAdmission,
        ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection> {
            Ok(AdmissionPersistenceAck {
                admission_receipt_digest: admission.receipt_digest.clone(),
                persistence_ack_digest: digest('0'),
            })
        }
    }

    struct Freshness;
    impl PreSpawnFreshnessGate for Freshness {
        fn verify(
            &mut self,
            _intent: &CargoExecutionIntent,
            admission: &EffectAdmission,
            persistence: &AdmissionPersistenceAck,
        ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection> {
            Ok(PreSpawnFreshnessApproval {
                admission_receipt_digest: admission.receipt_digest.clone(),
                persistence_ack_digest: persistence.persistence_ack_digest.clone(),
                freshness_evidence_digest: digest('5'),
            })
        }
    }

    struct Backend(Option<ProcessCapture>);
    impl CargoProcessBackend for Backend {
        fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
            self.0.take().unwrap()
        }
    }

    struct Verifier(Option<ObservationVerificationOutcome>, usize);
    impl CargoObservationVerificationBoundary for Verifier {
        fn verify(
            &mut self,
            _intent: &CargoExecutionIntent,
            _process: &ProcessCapture,
        ) -> ObservationVerificationOutcome {
            self.1 += 1;
            self.0.take().unwrap()
        }
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

    fn report(terminal: ProcessTerminal) -> AdapterRunReport {
        let mut probe = Probe(Some(Ok(digest('a'))), Some(Ok(digest('a'))));
        let mut gate = Gate(Some(Ok(EffectAdmission {
            receipt_digest: digest('2'),
        })));
        let mut persistence = Persistence;
        let mut freshness = Freshness;
        let mut backend = Backend(Some(ProcessCapture {
            terminal,
            stdout_sha256: digest('3'),
            stderr_sha256: digest('4'),
        }));
        run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap()
    }

    #[test]
    fn exited_process_can_produce_verified_observation() {
        let mut verifier = Verifier(Some(verified()), 0);
        let result = continue_with_observation(
            intent(),
            report(ProcessTerminal::Exited { code: 0 }),
            &mut verifier,
        )
        .unwrap();
        assert_eq!(verifier.1, 1);
        assert!(matches!(result.observation, ObservationStage::Verified { .. }));
    }

    #[test]
    fn spawn_failure_never_calls_observation_verifier() {
        let mut verifier = Verifier(None, 0);
        let result = continue_with_observation(
            intent(),
            report(ProcessTerminal::SpawnFailed {
                error_digest: digest('6'),
            }),
            &mut verifier,
        )
        .unwrap();
        assert_eq!(verifier.1, 0);
        assert!(matches!(
            result.observation,
            ObservationStage::NotApplicable { .. }
        ));
    }

    #[test]
    fn timeout_can_preserve_unavailable_observation() {
        let mut verifier = Verifier(
            Some(ObservationVerificationOutcome::Unavailable {
                reason: "partial transcript unavailable".into(),
            }),
            0,
        );
        let result = continue_with_observation(
            intent(),
            report(ProcessTerminal::TimedOut),
            &mut verifier,
        )
        .unwrap();
        assert_eq!(verifier.1, 1);
        assert!(matches!(
            result.observation,
            ObservationStage::Unavailable { .. }
        ));
    }

    #[test]
    fn verification_failure_retains_backend_and_postflight() {
        let mut verifier = Verifier(
            Some(ObservationVerificationOutcome::VerificationFailed {
                reason: "transcript substitution".into(),
            }),
            0,
        );
        let result = continue_with_observation(
            intent(),
            report(ProcessTerminal::Exited { code: 1 }),
            &mut verifier,
        )
        .unwrap();
        assert!(matches!(
            result.observation,
            ObservationStage::VerificationFailed { .. }
        ));
        assert!(matches!(
            result.adapter,
            AdapterRunReport::BackendEntered {
                postflight: PostflightCapture::Captured { .. },
                ..
            }
        ));
    }

    #[test]
    fn observation_subject_substitution_becomes_typed_failure() {
        for field in ["context", "transcript"] {
            let mut observation = match verified() {
                ObservationVerificationOutcome::Verified { observation } => observation,
                _ => unreachable!(),
            };
            match field {
                "context" => observation.context_id = digest('7'),
                "transcript" => observation.transcript_sha256 = digest('8'),
                _ => unreachable!(),
            }
            let mut verifier = Verifier(
                Some(ObservationVerificationOutcome::Verified { observation }),
                0,
            );
            let result = continue_with_observation(
                intent(),
                report(ProcessTerminal::Exited { code: 0 }),
                &mut verifier,
            )
            .unwrap();
            assert!(matches!(
                result.observation,
                ObservationStage::VerificationFailed { .. }
            ));
        }
    }

    #[test]
    fn every_pre_backend_rejection_skips_observation_verifier() {
        let frozen = intent();
        let cases = [
            AdapterRunReport::RejectedBeforeEffect {
                intent_id: frozen.intent_id.clone(),
                reason: "stale source".into(),
            },
            AdapterRunReport::EffectRejected {
                intent_id: frozen.intent_id.clone(),
                repository_source_before: digest('a'),
                reason: "effect denied".into(),
            },
            AdapterRunReport::AdmissionPersistenceRejected {
                intent_id: frozen.intent_id.clone(),
                repository_source_before: digest('a'),
                effect_admission_digest: digest('2'),
                reason: "durability unavailable".into(),
            },
            AdapterRunReport::FreshnessRejected {
                intent_id: frozen.intent_id,
                repository_source_before: digest('a'),
                effect_admission_digest: digest('2'),
                persistence_ack_digest: digest('0'),
                reason: "currentness drift".into(),
            },
        ];
        for adapter in cases {
            let mut verifier = Verifier(None, 0);
            let result = continue_with_observation(intent(), adapter, &mut verifier).unwrap();
            assert_eq!(verifier.1, 0);
            assert!(matches!(
                result.observation,
                ObservationStage::NotReached { .. }
            ));
        }
    }
}
