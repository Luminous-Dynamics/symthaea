use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

const ATTEMPT_SCHEMA: &str = "symthaea.cargo-execution-attempt-result.v1";
const ATTEMPT_HASH_DOMAIN: &[u8] = b"symthaea.cargo-execution-attempt-result.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CargoTerminalState {
    Exited { code: i32 },
    SpawnFailed { error_digest: String },
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservationNotApplicableReason {
    CargoDidNotSpawn,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum CargoObservationState {
    NotApplicable {
        reason: ObservationNotApplicableReason,
    },
    Unavailable {
        reason: String,
        transcript_sha256: Option<String>,
    },
    VerificationFailed {
        reason: String,
        transcript_sha256: Option<String>,
    },
    Verified {
        observation_id: String,
        context_id: String,
        invocation_id: String,
        transcript_sha256: String,
    },
}

/// Effect evaluation is a separate postflight fact from source capture. The
/// source subject may be known even when diff/effect evaluation cannot finish.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum CargoEffectPostflightState {
    Evaluated {
        observed_diff_id: String,
        effect_evaluation_sha256: String,
        effect_allowed: bool,
    },
    Unavailable {
        reason: String,
    },
}

/// Repository source sensing is itself terminal evidence. A source-capture
/// failure must not force fabrication of source-after, diff, or effect values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum CargoPostflightState {
    SourceCaptured {
        repository_source_after: String,
        git_worktree_state_after: Option<String>,
        effect: CargoEffectPostflightState,
    },
    SourceCaptureFailed {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoExecutionAttemptInput {
    pub effect_admission_digest: String,
    pub terminal: CargoTerminalState,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
    pub postflight: CargoPostflightState,
    pub observation: CargoObservationState,
}

/// Canonical execution-evidence commitment for one admitted Cargo backend
/// attempt. This object does not imply qualification or behavioral correctness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoExecutionAttemptResult {
    pub attempt_result_id: String,
    pub schema: String,
    pub intent_id: String,
    pub effect_admission_digest: String,
    pub repository_source_before: String,
    pub git_worktree_state_before: Option<String>,
    pub build_context_id: String,
    pub invocation_id: String,
    pub effect_policy_id: String,
    pub plan_id: Option<String>,
    pub transaction_id: Option<String>,
    pub adapter_semantics_digest: String,
    pub terminal: CargoTerminalState,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
    pub postflight: CargoPostflightState,
    pub observation: CargoObservationState,
}

#[derive(Serialize)]
struct AttemptIdentity<'a> {
    schema: &'static str,
    intent_id: &'a str,
    effect_admission_digest: &'a str,
    repository_source_before: &'a str,
    git_worktree_state_before: &'a Option<String>,
    build_context_id: &'a str,
    invocation_id: &'a str,
    effect_policy_id: &'a str,
    plan_id: &'a Option<String>,
    transaction_id: &'a Option<String>,
    adapter_semantics_digest: &'a str,
    terminal: &'a CargoTerminalState,
    stdout_sha256: &'a str,
    stderr_sha256: &'a str,
    postflight: &'a CargoPostflightState,
    observation: &'a CargoObservationState,
}

pub(crate) fn build_attempt_result(
    mut intent: CargoExecutionIntent,
    mut input: CargoExecutionAttemptInput,
) -> anyhow::Result<CargoExecutionAttemptResult> {
    validate_intent(&mut intent)?;
    normalize_input(&intent, &mut input)?;

    let identity = AttemptIdentity {
        schema: ATTEMPT_SCHEMA,
        intent_id: &intent.intent_id,
        effect_admission_digest: &input.effect_admission_digest,
        repository_source_before: &intent.repository_source_before,
        git_worktree_state_before: &intent.git_worktree_state_before,
        build_context_id: &intent.build_context_id,
        invocation_id: &intent.invocation_id,
        effect_policy_id: &intent.effect_policy_id,
        plan_id: &intent.plan_id,
        transaction_id: &intent.transaction_id,
        adapter_semantics_digest: &intent.adapter_semantics_digest,
        terminal: &input.terminal,
        stdout_sha256: &input.stdout_sha256,
        stderr_sha256: &input.stderr_sha256,
        postflight: &input.postflight,
        observation: &input.observation,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize Cargo execution attempt-result identity")?;
    let attempt_result_id = domain_sha256(ATTEMPT_HASH_DOMAIN, &bytes);

    Ok(CargoExecutionAttemptResult {
        attempt_result_id,
        schema: ATTEMPT_SCHEMA.into(),
        intent_id: intent.intent_id,
        effect_admission_digest: input.effect_admission_digest,
        repository_source_before: intent.repository_source_before,
        git_worktree_state_before: intent.git_worktree_state_before,
        build_context_id: intent.build_context_id,
        invocation_id: intent.invocation_id,
        effect_policy_id: intent.effect_policy_id,
        plan_id: intent.plan_id,
        transaction_id: intent.transaction_id,
        adapter_semantics_digest: intent.adapter_semantics_digest,
        terminal: input.terminal,
        stdout_sha256: input.stdout_sha256,
        stderr_sha256: input.stderr_sha256,
        postflight: input.postflight,
        observation: input.observation,
    })
}

fn normalize_input(
    intent: &CargoExecutionIntent,
    input: &mut CargoExecutionAttemptInput,
) -> anyhow::Result<()> {
    normalize_digest("effect_admission_digest", &mut input.effect_admission_digest)?;
    normalize_digest("stdout_sha256", &mut input.stdout_sha256)?;
    normalize_digest("stderr_sha256", &mut input.stderr_sha256)?;

    if let CargoTerminalState::SpawnFailed { error_digest } = &mut input.terminal {
        normalize_digest("terminal.error_digest", error_digest)?;
    }

    normalize_postflight(&mut input.postflight)?;
    normalize_observation(
        intent,
        &input.terminal,
        &input.stdout_sha256,
        &mut input.observation,
    )
}

fn normalize_postflight(postflight: &mut CargoPostflightState) -> anyhow::Result<()> {
    match postflight {
        CargoPostflightState::SourceCaptured {
            repository_source_after,
            git_worktree_state_after,
            effect,
        } => {
            normalize_digest("postflight.repository_source_after", repository_source_after)?;
            normalize_optional_digest(
                "postflight.git_worktree_state_after",
                git_worktree_state_after,
            )?;
            match effect {
                CargoEffectPostflightState::Evaluated {
                    observed_diff_id,
                    effect_evaluation_sha256,
                    ..
                } => {
                    normalize_digest("postflight.effect.observed_diff_id", observed_diff_id)?;
                    normalize_digest(
                        "postflight.effect.effect_evaluation_sha256",
                        effect_evaluation_sha256,
                    )?;
                }
                CargoEffectPostflightState::Unavailable { reason } => {
                    require_reason("effect evaluation unavailable", reason)?;
                }
            }
            Ok(())
        }
        CargoPostflightState::SourceCaptureFailed { reason } => {
            require_reason("postflight source-capture failure", reason)
        }
    }
}

fn normalize_observation(
    intent: &CargoExecutionIntent,
    terminal: &CargoTerminalState,
    stdout_sha256: &str,
    observation: &mut CargoObservationState,
) -> anyhow::Result<()> {
    match (terminal, observation) {
        (
            CargoTerminalState::SpawnFailed { .. },
            CargoObservationState::NotApplicable {
                reason: ObservationNotApplicableReason::CargoDidNotSpawn,
            },
        ) => Ok(()),
        (CargoTerminalState::SpawnFailed { .. }, _) => {
            bail!("SpawnFailed requires observation state not_applicable/cargo_did_not_spawn")
        }
        (_, CargoObservationState::NotApplicable { .. }) => {
            bail!("observation NotApplicable is only valid when Cargo failed to spawn")
        }
        (
            _,
            CargoObservationState::Unavailable {
                reason,
                transcript_sha256,
            }
            | CargoObservationState::VerificationFailed {
                reason,
                transcript_sha256,
            },
        ) => {
            require_reason("unavailable/verification-failed observation", reason)?;
            if let Some(transcript) = transcript_sha256 {
                normalize_digest("observation.transcript_sha256", transcript)?;
                if transcript.as_str() != stdout_sha256 {
                    bail!(
                        "observation transcript {} does not match captured stdout {}",
                        transcript,
                        stdout_sha256
                    );
                }
            }
            Ok(())
        }
        (
            _,
            CargoObservationState::Verified {
                observation_id,
                context_id,
                invocation_id,
                transcript_sha256,
            },
        ) => {
            normalize_digest("observation.observation_id", observation_id)?;
            normalize_digest("observation.context_id", context_id)?;
            normalize_digest("observation.invocation_id", invocation_id)?;
            normalize_digest("observation.transcript_sha256", transcript_sha256)?;
            if context_id.as_str() != intent.build_context_id.as_str() {
                bail!(
                    "verified observation context {} does not match intent {}",
                    context_id,
                    intent.build_context_id
                );
            }
            if invocation_id.as_str() != intent.invocation_id.as_str() {
                bail!(
                    "verified observation invocation {} does not match intent {}",
                    invocation_id,
                    intent.invocation_id
                );
            }
            if transcript_sha256.as_str() != stdout_sha256 {
                bail!(
                    "verified observation transcript {} does not match captured stdout {}",
                    transcript_sha256,
                    stdout_sha256
                );
            }
            Ok(())
        }
    }
}

fn require_reason(label: &str, reason: &str) -> anyhow::Result<()> {
    if reason.trim().is_empty() {
        bail!("{label} requires a non-empty reason");
    }
    Ok(())
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        normalize_digest(name, value)?;
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn evaluated_effect() -> CargoEffectPostflightState {
        CargoEffectPostflightState::Evaluated {
            observed_diff_id: digest('5'),
            effect_evaluation_sha256: digest('6'),
            effect_allowed: true,
        }
    }

    fn captured_postflight() -> CargoPostflightState {
        CargoPostflightState::SourceCaptured {
            repository_source_after: digest('a'),
            git_worktree_state_after: Some(digest('9')),
            effect: evaluated_effect(),
        }
    }

    fn base_input(
        terminal: CargoTerminalState,
        observation: CargoObservationState,
    ) -> CargoExecutionAttemptInput {
        CargoExecutionAttemptInput {
            effect_admission_digest: digest('2'),
            terminal,
            stdout_sha256: digest('3'),
            stderr_sha256: digest('4'),
            postflight: captured_postflight(),
            observation,
        }
    }

    fn verified() -> CargoObservationState {
        CargoObservationState::Verified {
            observation_id: digest('7'),
            context_id: digest('b'),
            invocation_id: digest('c'),
            transcript_sha256: digest('3'),
        }
    }

    #[test]
    fn spawn_failure_accepts_only_not_applicable_observation() {
        let result = build_attempt_result(
            intent(),
            base_input(
                CargoTerminalState::SpawnFailed {
                    error_digest: digest('8'),
                },
                CargoObservationState::NotApplicable {
                    reason: ObservationNotApplicableReason::CargoDidNotSpawn,
                },
            ),
        )
        .unwrap();
        assert!(matches!(result.terminal, CargoTerminalState::SpawnFailed { .. }));
        assert!(matches!(
            result.observation,
            CargoObservationState::NotApplicable { .. }
        ));
    }

    #[test]
    fn spawn_failure_rejects_fabricated_verified_observation() {
        assert!(
            build_attempt_result(
                intent(),
                base_input(
                    CargoTerminalState::SpawnFailed {
                        error_digest: digest('8'),
                    },
                    verified(),
                ),
            )
            .is_err()
        );
    }

    #[test]
    fn exited_process_rejects_not_applicable_observation() {
        assert!(
            build_attempt_result(
                intent(),
                base_input(
                    CargoTerminalState::Exited { code: 0 },
                    CargoObservationState::NotApplicable {
                        reason: ObservationNotApplicableReason::CargoDidNotSpawn,
                    },
                ),
            )
            .is_err()
        );
    }

    #[test]
    fn verified_observation_binds_context_invocation_and_transcript() {
        let result = build_attempt_result(
            intent(),
            base_input(CargoTerminalState::Exited { code: 0 }, verified()),
        )
        .unwrap();
        assert!(matches!(result.observation, CargoObservationState::Verified { .. }));

        let mut bad = verified();
        if let CargoObservationState::Verified { transcript_sha256, .. } = &mut bad {
            *transcript_sha256 = digest('0');
        }
        assert!(
            build_attempt_result(
                intent(),
                base_input(CargoTerminalState::Exited { code: 0 }, bad),
            )
            .is_err()
        );
    }

    #[test]
    fn timeout_can_truthfully_record_unavailable_observation() {
        let result = build_attempt_result(
            intent(),
            base_input(
                CargoTerminalState::TimedOut,
                CargoObservationState::Unavailable {
                    reason: "no complete transcript".into(),
                    transcript_sha256: None,
                },
            ),
        )
        .unwrap();
        assert!(matches!(result.observation, CargoObservationState::Unavailable { .. }));
    }

    #[test]
    fn source_capture_failure_still_produces_attempt_result() {
        let mut input = base_input(CargoTerminalState::Exited { code: 1 }, verified());
        input.postflight = CargoPostflightState::SourceCaptureFailed {
            reason: "source probe failed".into(),
        };
        let result = build_attempt_result(intent(), input).unwrap();
        assert!(matches!(
            result.postflight,
            CargoPostflightState::SourceCaptureFailed { .. }
        ));
    }

    #[test]
    fn source_can_be_known_while_effect_evaluation_is_unavailable() {
        let mut input = base_input(CargoTerminalState::Exited { code: 0 }, verified());
        input.postflight = CargoPostflightState::SourceCaptured {
            repository_source_after: digest('a'),
            git_worktree_state_after: Some(digest('9')),
            effect: CargoEffectPostflightState::Unavailable {
                reason: "diff engine unavailable".into(),
            },
        };
        let result = build_attempt_result(intent(), input).unwrap();
        assert!(matches!(
            result.postflight,
            CargoPostflightState::SourceCaptured {
                effect: CargoEffectPostflightState::Unavailable { .. },
                ..
            }
        ));
    }

    #[test]
    fn empty_failure_reasons_are_rejected() {
        let mut input = base_input(
            CargoTerminalState::Cancelled,
            CargoObservationState::Unavailable {
                reason: "   ".into(),
                transcript_sha256: None,
            },
        );
        assert!(build_attempt_result(intent(), input.clone()).is_err());

        input.observation = CargoObservationState::Unavailable {
            reason: "cancelled".into(),
            transcript_sha256: None,
        };
        input.postflight = CargoPostflightState::SourceCaptureFailed {
            reason: "".into(),
        };
        assert!(build_attempt_result(intent(), input).is_err());
    }

    #[test]
    fn effect_entry_and_effect_state_are_identity_significant() {
        let original = build_attempt_result(
            intent(),
            base_input(CargoTerminalState::Exited { code: 0 }, verified()),
        )
        .unwrap();

        let mut changed_admission =
            base_input(CargoTerminalState::Exited { code: 0 }, verified());
        changed_admission.effect_admission_digest = digest('0');
        let changed_admission = build_attempt_result(intent(), changed_admission).unwrap();
        assert_ne!(original.attempt_result_id, changed_admission.attempt_result_id);

        let mut unavailable_effect =
            base_input(CargoTerminalState::Exited { code: 0 }, verified());
        unavailable_effect.postflight = CargoPostflightState::SourceCaptured {
            repository_source_after: digest('a'),
            git_worktree_state_after: Some(digest('9')),
            effect: CargoEffectPostflightState::Unavailable {
                reason: "effect evaluator failed".into(),
            },
        };
        let unavailable_effect = build_attempt_result(intent(), unavailable_effect).unwrap();
        assert_ne!(original.attempt_result_id, unavailable_effect.attempt_result_id);
    }

    #[test]
    fn terminal_and_source_postflight_state_are_identity_significant() {
        let exited = build_attempt_result(
            intent(),
            base_input(CargoTerminalState::Exited { code: 1 }, verified()),
        )
        .unwrap();

        let mut failed_postflight =
            base_input(CargoTerminalState::Exited { code: 1 }, verified());
        failed_postflight.postflight = CargoPostflightState::SourceCaptureFailed {
            reason: "probe failed".into(),
        };
        let failed_postflight = build_attempt_result(intent(), failed_postflight).unwrap();
        assert_ne!(exited.attempt_result_id, failed_postflight.attempt_result_id);

        let timed_out = build_attempt_result(
            intent(),
            base_input(
                CargoTerminalState::TimedOut,
                CargoObservationState::Unavailable {
                    reason: "timeout".into(),
                    transcript_sha256: Some(digest('3')),
                },
            ),
        )
        .unwrap();
        assert_ne!(exited.attempt_result_id, timed_out.attempt_result_id);
    }
}
