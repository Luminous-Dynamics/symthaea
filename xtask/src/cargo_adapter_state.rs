use anyhow::{Context, bail};
use serde::Serialize;

use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Externally validated identities required before the executor may approach the
/// effect-entry boundary. This type carries no authority by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedPreflightBindings {
    pub build_context_id: String,
    pub invocation_id: String,
    pub effect_policy_id: String,
    pub git_worktree_state_id: Option<String>,
}

/// Opaque effect-entry evidence supplied by an external assurance/effect gate.
/// The adapter orchestrator does not mint or interpret authority semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectAdmission {
    pub receipt_digest: String,
}

/// Explicit denial from the external effect-entry gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectRejection {
    pub reason: String,
}

/// External effect-entry interface. Production integration is expected to be
/// backed by AI Assurance / the effect-entry domain rather than an xtask-local
/// grant system.
pub(crate) trait EffectEntryGate {
    fn admit(
        &mut self,
        intent: &CargoExecutionIntent,
    ) -> Result<EffectAdmission, EffectRejection>;
}

/// Terminal subprocess state. A non-zero Cargo exit remains ordinary execution
/// evidence rather than an orchestration error.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum ProcessTerminal {
    Exited { code: i32 },
    SpawnFailed { error_digest: String },
    TimedOut,
    Cancelled,
}

/// Capture returned by the process backend boundary.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ProcessCapture {
    pub terminal: ProcessTerminal,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
}

/// Process execution boundary. The first production implementation must be the
/// trusted, sandboxed Cargo adapter from #4558. Tests use a deterministic fake.
pub(crate) trait CargoProcessBackend {
    fn execute(&mut self, intent: &CargoExecutionIntent) -> ProcessCapture;
}

/// Source-subject sensor separated from process execution. The production probe
/// will use repository-source receipts/snapshots; tests inject deterministic
/// subjects to prove call ordering without touching the filesystem.
pub(crate) trait RepositorySourceProbe {
    fn preflight_source_id(&mut self) -> anyhow::Result<String>;
    fn postflight_source_id(&mut self) -> anyhow::Result<String>;
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum PostflightCapture {
    Captured { repository_source_after: String },
    CaptureFailed { error: String },
}

/// Observable orchestration result for the backend-independent executor model.
/// This is deliberately not an authority/evidence/qualification receipt.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum AdapterRunReport {
    RejectedBeforeEffect {
        intent_id: String,
        reason: String,
    },
    EffectRejected {
        intent_id: String,
        repository_source_before: String,
        reason: String,
    },
    /// Effect admission succeeded and the adapter backend boundary was invoked.
    /// `process.terminal` separately states whether a child process actually
    /// spawned/exited, failed to spawn, timed out, or was cancelled.
    BackendEntered {
        intent_id: String,
        repository_source_before: String,
        effect_admission_digest: String,
        process: ProcessCapture,
        postflight: PostflightCapture,
    },
}

impl AdapterRunReport {
    #[cfg(test)]
    fn backend_was_called(&self) -> bool {
        matches!(self, Self::BackendEntered { .. })
    }
}

/// Run the executor orchestration against injected boundaries.
///
/// Ordering theorem:
///
/// 1. validate the immutable execution intent;
/// 2. validate independently established Cargo/policy/Git bindings;
/// 3. probe the current source subject and require the exact admitted subject;
/// 4. acquire external effect-entry admission;
/// 5. only then invoke the process backend;
/// 6. after backend return, always attempt postflight source capture before
///    interpreting/validating backend output.
///
/// Invalid/malformed immutable inputs return `Err` before any external boundary
/// is called. Ordinary preflight mismatches and effect denial are represented as
/// reports so tests can prove the backend remained unreachable.
pub(crate) fn run_with_backend<P, G, B>(
    mut intent: CargoExecutionIntent,
    mut bindings: ValidatedPreflightBindings,
    source_probe: &mut P,
    effect_gate: &mut G,
    backend: &mut B,
) -> anyhow::Result<AdapterRunReport>
where
    P: RepositorySourceProbe,
    G: EffectEntryGate,
    B: CargoProcessBackend,
{
    validate_intent(&mut intent)?;
    normalize_bindings(&mut bindings)?;

    if bindings.build_context_id != intent.build_context_id {
        return Ok(rejected(
            &intent,
            format!(
                "validated build context {} does not match intent {}",
                bindings.build_context_id, intent.build_context_id
            ),
        ));
    }
    if bindings.invocation_id != intent.invocation_id {
        return Ok(rejected(
            &intent,
            format!(
                "validated invocation {} does not match intent {}",
                bindings.invocation_id, intent.invocation_id
            ),
        ));
    }
    if bindings.effect_policy_id != intent.effect_policy_id {
        return Ok(rejected(
            &intent,
            format!(
                "validated effect policy {} does not match intent {}",
                bindings.effect_policy_id, intent.effect_policy_id
            ),
        ));
    }
    if bindings.git_worktree_state_id != intent.git_worktree_state_before {
        return Ok(rejected(
            &intent,
            "validated Git worktree state does not match intent".to_string(),
        ));
    }

    let mut current_source = source_probe
        .preflight_source_id()
        .context("capture preflight repository source subject")?;
    normalize_digest("preflight_repository_source", &mut current_source)?;
    if current_source != intent.repository_source_before {
        return Ok(rejected(
            &intent,
            format!(
                "current repository source {} does not match intent subject {}",
                current_source, intent.repository_source_before
            ),
        ));
    }

    let admission = match effect_gate.admit(&intent) {
        Ok(mut admission) => {
            normalize_digest("effect_admission.receipt_digest", &mut admission.receipt_digest)?;
            admission
        }
        Err(rejection) => {
            return Ok(AdapterRunReport::EffectRejected {
                intent_id: intent.intent_id,
                repository_source_before: current_source,
                reason: rejection.reason,
            });
        }
    };

    let mut process = backend.execute(&intent);

    // Once backend entry has occurred, preserve the ordering invariant that
    // postflight sensing is attempted before interpreting backend output. This
    // matters even when the backend returns malformed/incomplete capture data.
    let postflight = match source_probe.postflight_source_id() {
        Ok(mut source_id) => {
            normalize_digest("postflight_repository_source", &mut source_id)?;
            PostflightCapture::Captured {
                repository_source_after: source_id,
            }
        }
        Err(error) => PostflightCapture::CaptureFailed {
            error: error.to_string(),
        },
    };

    normalize_process_capture(&mut process)?;

    Ok(AdapterRunReport::BackendEntered {
        intent_id: intent.intent_id,
        repository_source_before: current_source,
        effect_admission_digest: admission.receipt_digest,
        process,
        postflight,
    })
}

fn rejected(intent: &CargoExecutionIntent, reason: String) -> AdapterRunReport {
    AdapterRunReport::RejectedBeforeEffect {
        intent_id: intent.intent_id.clone(),
        reason,
    }
}

fn normalize_bindings(bindings: &mut ValidatedPreflightBindings) -> anyhow::Result<()> {
    normalize_digest("bindings.build_context_id", &mut bindings.build_context_id)?;
    normalize_digest("bindings.invocation_id", &mut bindings.invocation_id)?;
    normalize_digest("bindings.effect_policy_id", &mut bindings.effect_policy_id)?;
    if let Some(value) = &mut bindings.git_worktree_state_id {
        normalize_digest("bindings.git_worktree_state_id", value)?;
    }
    Ok(())
}

fn normalize_process_capture(capture: &mut ProcessCapture) -> anyhow::Result<()> {
    normalize_digest("process.stdout_sha256", &mut capture.stdout_sha256)?;
    normalize_digest("process.stderr_sha256", &mut capture.stderr_sha256)?;
    if let ProcessTerminal::SpawnFailed { error_digest } = &mut capture.terminal {
        normalize_digest("process.error_digest", error_digest)?;
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

    fn bindings() -> ValidatedPreflightBindings {
        ValidatedPreflightBindings {
            build_context_id: digest('b'),
            invocation_id: digest('c'),
            effect_policy_id: digest('d'),
            git_worktree_state_id: Some(digest('9')),
        }
    }

    #[derive(Default)]
    struct FakeProbe {
        pre: Option<anyhow::Result<String>>,
        post: Option<anyhow::Result<String>>,
        pre_calls: usize,
        post_calls: usize,
    }

    impl RepositorySourceProbe for FakeProbe {
        fn preflight_source_id(&mut self) -> anyhow::Result<String> {
            self.pre_calls += 1;
            self.pre
                .take()
                .expect("fake preflight source configured")
        }

        fn postflight_source_id(&mut self) -> anyhow::Result<String> {
            self.post_calls += 1;
            self.post
                .take()
                .expect("fake postflight source configured")
        }
    }

    struct FakeGate {
        decision: Option<Result<EffectAdmission, EffectRejection>>,
        calls: usize,
    }

    impl EffectEntryGate for FakeGate {
        fn admit(
            &mut self,
            _intent: &CargoExecutionIntent,
        ) -> Result<EffectAdmission, EffectRejection> {
            self.calls += 1;
            self.decision.take().expect("fake admission configured")
        }
    }

    struct FakeBackend {
        capture: Option<ProcessCapture>,
        calls: usize,
    }

    impl CargoProcessBackend for FakeBackend {
        fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
            self.calls += 1;
            self.capture.take().expect("fake process capture configured")
        }
    }

    fn probe(pre: anyhow::Result<String>, post: anyhow::Result<String>) -> FakeProbe {
        FakeProbe {
            pre: Some(pre),
            post: Some(post),
            ..Default::default()
        }
    }

    fn gate_allowed() -> FakeGate {
        FakeGate {
            decision: Some(Ok(EffectAdmission {
                receipt_digest: digest('2'),
            })),
            calls: 0,
        }
    }

    fn backend_exit(code: i32) -> FakeBackend {
        FakeBackend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::Exited { code },
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
            calls: 0,
        }
    }

    #[test]
    fn stale_source_rejects_before_effect_or_backend() {
        let mut probe = probe(Ok(digest('8')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend)
            .unwrap();

        assert!(!report.backend_was_called());
        assert_eq!(probe.pre_calls, 1);
        assert_eq!(probe.post_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(backend.calls, 0);
    }

    #[test]
    fn context_substitution_rejects_before_source_probe() {
        let mut bad = bindings();
        bad.build_context_id = digest('7');
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(intent(), bad, &mut probe, &mut gate, &mut backend).unwrap();

        assert!(!report.backend_was_called());
        assert_eq!(probe.pre_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(backend.calls, 0);
    }

    #[test]
    fn effect_denial_keeps_backend_unreachable() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = FakeGate {
            decision: Some(Err(EffectRejection {
                reason: "revoked authority epoch".into(),
            })),
            calls: 0,
        };
        let mut backend = backend_exit(0);
        let report = run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend)
            .unwrap();

        assert!(matches!(report, AdapterRunReport::EffectRejected { .. }));
        assert_eq!(gate.calls, 1);
        assert_eq!(backend.calls, 0);
        assert_eq!(probe.post_calls, 0);
    }

    #[test]
    fn nonzero_process_exit_is_retained_and_postflight_still_runs() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = backend_exit(17);
        let report = run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend)
            .unwrap();

        match report {
            AdapterRunReport::BackendEntered { process, postflight, .. } => {
                assert_eq!(process.terminal, ProcessTerminal::Exited { code: 17 });
                assert!(matches!(postflight, PostflightCapture::Captured { .. }));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(backend.calls, 1);
        assert_eq!(probe.post_calls, 1);
    }

    #[test]
    fn postflight_capture_failure_does_not_erase_process_evidence() {
        let mut probe = probe(Ok(digest('a')), Err(anyhow::anyhow!("probe failed")));
        let mut gate = gate_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend)
            .unwrap();

        match report {
            AdapterRunReport::BackendEntered { process, postflight, .. } => {
                assert_eq!(process.terminal, ProcessTerminal::Exited { code: 0 });
                assert!(matches!(postflight, PostflightCapture::CaptureFailed { .. }));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(backend.calls, 1);
        assert_eq!(probe.post_calls, 1);
    }

    #[test]
    fn spawn_failure_is_not_mislabeled_as_process_execution() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = FakeBackend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::SpawnFailed {
                    error_digest: digest('6'),
                },
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
            calls: 0,
        };
        let report = run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend)
            .unwrap();

        match report {
            AdapterRunReport::BackendEntered { process, .. } => {
                assert!(matches!(process.terminal, ProcessTerminal::SpawnFailed { .. }));
            }
            other => panic!("unexpected report: {other:?}"),
        }
    }

    #[test]
    fn malformed_backend_capture_still_attempts_postflight_before_error() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = FakeBackend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::Exited { code: 0 },
                stdout_sha256: "not-a-digest".into(),
                stderr_sha256: digest('4'),
            }),
            calls: 0,
        };

        assert!(run_with_backend(intent(), bindings(), &mut probe, &mut gate, &mut backend).is_err());
        assert_eq!(backend.calls, 1);
        assert_eq!(probe.post_calls, 1);
    }

    #[test]
    fn tampered_intent_rejects_before_any_external_boundary() {
        let mut tampered = intent();
        tampered.adapter_semantics_digest = digest('8');
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut backend = backend_exit(0);
        assert!(run_with_backend(tampered, bindings(), &mut probe, &mut gate, &mut backend).is_err());
        assert_eq!(probe.pre_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(backend.calls, 0);
    }
}
