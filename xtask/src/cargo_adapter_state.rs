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

/// Opaque effect-entry admission evidence supplied by an external assurance gate.
///
/// Production integration is expected to map this to the corrected AI Assurance
/// `EffectAdmissionReceipt` contract. The xtask model treats the digest as opaque
/// and never infers adapter entry or external-effect occurrence from it.
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

/// Opaque acknowledgement that the exact admission evidence reached the
/// persistence strength selected by the trusted host.
///
/// The deterministic model does not claim that this is fsync, a database commit,
/// replication, a Xenia signature, or any other concrete durability mechanism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdmissionPersistenceAck {
    pub admission_receipt_digest: String,
    pub persistence_ack_digest: String,
}

/// Persistence policy rejected or failed after effect admission had already won.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdmissionPersistenceRejection {
    pub reason: String,
}

/// Host-selected persistence boundary for already-won effect admission evidence.
pub(crate) trait AdmissionPersistence {
    fn persist(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
    ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection>;
}

/// Opaque approval that immediate pre-spawn freshness still matches the exact
/// admitted/persisted execution subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreSpawnFreshnessApproval {
    pub admission_receipt_digest: String,
    pub persistence_ack_digest: String,
    pub freshness_evidence_digest: String,
}

/// Freshness rejected after admission evidence was persisted/acknowledged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreSpawnFreshnessRejection {
    pub reason: String,
}

/// Immediate pre-backend freshness boundary.
///
/// Production integration is expected to be backed by the canonical pre-spawn
/// freshness receipts rather than this deterministic fake interface.
pub(crate) trait PreSpawnFreshnessGate {
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        persistence: &AdmissionPersistenceAck,
    ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection>;
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
    /// Effect admission won, but the selected persistence policy rejected or its
    /// acknowledgement was malformed. Admission evidence is retained explicitly.
    AdmissionPersistenceRejected {
        intent_id: String,
        repository_source_before: String,
        effect_admission_digest: String,
        reason: String,
    },
    /// Admission evidence was persisted/acknowledged, but immediate freshness
    /// rejected before the process backend became reachable.
    FreshnessRejected {
        intent_id: String,
        repository_source_before: String,
        effect_admission_digest: String,
        persistence_ack_digest: String,
        reason: String,
    },
    /// Admission, persistence, and freshness succeeded and the adapter backend
    /// boundary was invoked. `process.terminal` separately states whether a child
    /// process actually spawned/exited, failed to spawn, timed out, or cancelled.
    BackendEntered {
        intent_id: String,
        repository_source_before: String,
        effect_admission_digest: String,
        persistence_ack_digest: String,
        pre_spawn_freshness_digest: String,
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
/// 4. acquire external effect-entry admission evidence;
/// 5. persist/ack that exact admission evidence under host policy;
/// 6. require immediate pre-spawn freshness bound to the same admission + ack;
/// 7. only then invoke the process backend;
/// 8. after backend return, always attempt postflight source capture before
///    interpreting/validating backend output.
///
/// Invalid/malformed immutable inputs return `Err` before any external boundary
/// is called. Ordinary preflight/admission/persistence/freshness rejections are
/// represented as reports so tests can preserve already-established evidence and
/// prove that later boundaries remained unreachable.
pub(crate) fn run_with_backend<P, G, S, F, B>(
    mut intent: CargoExecutionIntent,
    mut bindings: ValidatedPreflightBindings,
    source_probe: &mut P,
    effect_gate: &mut G,
    persistence: &mut S,
    freshness: &mut F,
    backend: &mut B,
) -> anyhow::Result<AdapterRunReport>
where
    P: RepositorySourceProbe,
    G: EffectEntryGate,
    S: AdmissionPersistence,
    F: PreSpawnFreshnessGate,
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

    let persistence_ack = match persistence.persist(&intent, &admission) {
        Ok(mut ack) => {
            let validation = validate_persistence_ack(&mut ack, &admission);
            if let Err(error) = validation {
                return Ok(AdapterRunReport::AdmissionPersistenceRejected {
                    intent_id: intent.intent_id.clone(),
                    repository_source_before: current_source,
                    effect_admission_digest: admission.receipt_digest,
                    reason: error.to_string(),
                });
            }
            ack
        }
        Err(rejection) => {
            return Ok(AdapterRunReport::AdmissionPersistenceRejected {
                intent_id: intent.intent_id.clone(),
                repository_source_before: current_source,
                effect_admission_digest: admission.receipt_digest,
                reason: rejection.reason,
            });
        }
    };

    let freshness_approval = match freshness.verify(&intent, &admission, &persistence_ack) {
        Ok(mut approval) => {
            let validation = validate_freshness_approval(
                &mut approval,
                &admission,
                &persistence_ack,
            );
            if let Err(error) = validation {
                return Ok(AdapterRunReport::FreshnessRejected {
                    intent_id: intent.intent_id.clone(),
                    repository_source_before: current_source,
                    effect_admission_digest: admission.receipt_digest,
                    persistence_ack_digest: persistence_ack.persistence_ack_digest,
                    reason: error.to_string(),
                });
            }
            approval
        }
        Err(rejection) => {
            return Ok(AdapterRunReport::FreshnessRejected {
                intent_id: intent.intent_id.clone(),
                repository_source_before: current_source,
                effect_admission_digest: admission.receipt_digest,
                persistence_ack_digest: persistence_ack.persistence_ack_digest,
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
        persistence_ack_digest: persistence_ack.persistence_ack_digest,
        pre_spawn_freshness_digest: freshness_approval.freshness_evidence_digest,
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

fn validate_persistence_ack(
    ack: &mut AdmissionPersistenceAck,
    admission: &EffectAdmission,
) -> anyhow::Result<()> {
    normalize_digest(
        "persistence.admission_receipt_digest",
        &mut ack.admission_receipt_digest,
    )?;
    normalize_digest(
        "persistence.persistence_ack_digest",
        &mut ack.persistence_ack_digest,
    )?;
    if ack.admission_receipt_digest != admission.receipt_digest {
        bail!("persistence acknowledgement is bound to another admission receipt");
    }
    Ok(())
}

fn validate_freshness_approval(
    approval: &mut PreSpawnFreshnessApproval,
    admission: &EffectAdmission,
    persistence: &AdmissionPersistenceAck,
) -> anyhow::Result<()> {
    normalize_digest(
        "freshness.admission_receipt_digest",
        &mut approval.admission_receipt_digest,
    )?;
    normalize_digest(
        "freshness.persistence_ack_digest",
        &mut approval.persistence_ack_digest,
    )?;
    normalize_digest(
        "freshness.freshness_evidence_digest",
        &mut approval.freshness_evidence_digest,
    )?;
    if approval.admission_receipt_digest != admission.receipt_digest {
        bail!("freshness approval is bound to another admission receipt");
    }
    if approval.persistence_ack_digest != persistence.persistence_ack_digest {
        bail!("freshness approval is bound to another persistence acknowledgement");
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
    use std::cell::RefCell;
    use std::rc::Rc;

    type Trace = Rc<RefCell<Vec<&'static str>>>;

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
        trace: Option<Trace>,
    }

    impl RepositorySourceProbe for FakeProbe {
        fn preflight_source_id(&mut self) -> anyhow::Result<String> {
            self.pre_calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("preflight_source");
            }
            self.pre
                .take()
                .expect("fake preflight source configured")
        }

        fn postflight_source_id(&mut self) -> anyhow::Result<String> {
            self.post_calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("postflight_source");
            }
            self.post
                .take()
                .expect("fake postflight source configured")
        }
    }

    struct FakeGate {
        decision: Option<Result<EffectAdmission, EffectRejection>>,
        calls: usize,
        trace: Option<Trace>,
    }

    impl EffectEntryGate for FakeGate {
        fn admit(
            &mut self,
            _intent: &CargoExecutionIntent,
        ) -> Result<EffectAdmission, EffectRejection> {
            self.calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("admit");
            }
            self.decision.take().expect("fake admission configured")
        }
    }

    struct FakePersistence {
        decision: Option<Result<AdmissionPersistenceAck, AdmissionPersistenceRejection>>,
        calls: usize,
        trace: Option<Trace>,
    }

    impl AdmissionPersistence for FakePersistence {
        fn persist(
            &mut self,
            _intent: &CargoExecutionIntent,
            _admission: &EffectAdmission,
        ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection> {
            self.calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("persist");
            }
            self.decision.take().expect("fake persistence configured")
        }
    }

    struct FakeFreshness {
        decision: Option<Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection>>,
        calls: usize,
        trace: Option<Trace>,
    }

    impl PreSpawnFreshnessGate for FakeFreshness {
        fn verify(
            &mut self,
            _intent: &CargoExecutionIntent,
            _admission: &EffectAdmission,
            _persistence: &AdmissionPersistenceAck,
        ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection> {
            self.calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("freshness");
            }
            self.decision.take().expect("fake freshness configured")
        }
    }

    struct FakeBackend {
        capture: Option<ProcessCapture>,
        calls: usize,
        trace: Option<Trace>,
    }

    impl CargoProcessBackend for FakeBackend {
        fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
            self.calls += 1;
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("backend");
            }
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
            trace: None,
        }
    }

    fn persistence_allowed() -> FakePersistence {
        FakePersistence {
            decision: Some(Ok(AdmissionPersistenceAck {
                admission_receipt_digest: digest('2'),
                persistence_ack_digest: digest('5'),
            })),
            calls: 0,
            trace: None,
        }
    }

    fn freshness_allowed() -> FakeFreshness {
        FakeFreshness {
            decision: Some(Ok(PreSpawnFreshnessApproval {
                admission_receipt_digest: digest('2'),
                persistence_ack_digest: digest('5'),
                freshness_evidence_digest: digest('7'),
            })),
            calls: 0,
            trace: None,
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
            trace: None,
        }
    }

    #[test]
    fn stale_source_rejects_before_effect_or_later_boundaries() {
        let mut probe = probe(Ok(digest('8')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        assert!(!report.backend_was_called());
        assert_eq!(probe.pre_calls, 1);
        assert_eq!(probe.post_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(persistence.calls, 0);
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
    }

    #[test]
    fn context_substitution_rejects_before_source_probe() {
        let mut bad = bindings();
        bad.build_context_id = digest('7');
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bad,
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        assert!(!report.backend_was_called());
        assert_eq!(probe.pre_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(persistence.calls, 0);
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
    }

    #[test]
    fn effect_denial_keeps_persistence_freshness_and_backend_unreachable() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = FakeGate {
            decision: Some(Err(EffectRejection {
                reason: "revoked authority epoch".into(),
            })),
            calls: 0,
            trace: None,
        };
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        assert!(matches!(report, AdapterRunReport::EffectRejected { .. }));
        assert_eq!(gate.calls, 1);
        assert_eq!(persistence.calls, 0);
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
        assert_eq!(probe.post_calls, 0);
    }

    #[test]
    fn persistence_failure_retains_admission_and_blocks_later_boundaries() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = FakePersistence {
            decision: Some(Err(AdmissionPersistenceRejection {
                reason: "durability acknowledgement unavailable".into(),
            })),
            calls: 0,
            trace: None,
        };
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::AdmissionPersistenceRejected {
                effect_admission_digest,
                reason,
                ..
            } => {
                assert_eq!(effect_admission_digest, digest('2'));
                assert!(reason.contains("durability"));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(gate.calls, 1);
        assert_eq!(persistence.calls, 1);
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
        assert_eq!(probe.post_calls, 0);
    }

    #[test]
    fn malformed_persistence_binding_retains_admission_and_fails_closed() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = FakePersistence {
            decision: Some(Ok(AdmissionPersistenceAck {
                admission_receipt_digest: digest('8'),
                persistence_ack_digest: digest('5'),
            })),
            calls: 0,
            trace: None,
        };
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::AdmissionPersistenceRejected {
                effect_admission_digest,
                reason,
                ..
            } => {
                assert_eq!(effect_admission_digest, digest('2'));
                assert!(reason.contains("another admission"));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
    }

    #[test]
    fn freshness_failure_retains_admission_and_persistence_ack() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = FakeFreshness {
            decision: Some(Err(PreSpawnFreshnessRejection {
                reason: "source changed after admission persistence".into(),
            })),
            calls: 0,
            trace: None,
        };
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::FreshnessRejected {
                effect_admission_digest,
                persistence_ack_digest,
                reason,
                ..
            } => {
                assert_eq!(effect_admission_digest, digest('2'));
                assert_eq!(persistence_ack_digest, digest('5'));
                assert!(reason.contains("source changed"));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(gate.calls, 1);
        assert_eq!(persistence.calls, 1);
        assert_eq!(freshness.calls, 1);
        assert_eq!(backend.calls, 0);
        assert_eq!(probe.post_calls, 0);
    }

    #[test]
    fn freshness_binding_substitution_fails_before_backend() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = FakeFreshness {
            decision: Some(Ok(PreSpawnFreshnessApproval {
                admission_receipt_digest: digest('2'),
                persistence_ack_digest: digest('6'),
                freshness_evidence_digest: digest('7'),
            })),
            calls: 0,
            trace: None,
        };
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::FreshnessRejected { reason, .. } => {
                assert!(reason.contains("another persistence acknowledgement"));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(backend.calls, 0);
        assert_eq!(probe.post_calls, 0);
    }

    #[test]
    fn nonzero_process_exit_retains_all_pre_spawn_evidence_and_runs_postflight() {
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(17);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::BackendEntered {
                effect_admission_digest,
                persistence_ack_digest,
                pre_spawn_freshness_digest,
                process,
                postflight,
                ..
            } => {
                assert_eq!(effect_admission_digest, digest('2'));
                assert_eq!(persistence_ack_digest, digest('5'));
                assert_eq!(pre_spawn_freshness_digest, digest('7'));
                assert_eq!(process.terminal, ProcessTerminal::Exited { code: 17 });
                assert!(matches!(postflight, PostflightCapture::Captured { .. }));
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(backend.calls, 1);
        assert_eq!(probe.post_calls, 1);
    }

    #[test]
    fn exact_order_is_admit_then_persist_then_freshness_then_backend_then_postflight() {
        let trace: Trace = Rc::new(RefCell::new(Vec::new()));
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        probe.trace = Some(Rc::clone(&trace));
        let mut gate = gate_allowed();
        gate.trace = Some(Rc::clone(&trace));
        let mut persistence = persistence_allowed();
        persistence.trace = Some(Rc::clone(&trace));
        let mut freshness = freshness_allowed();
        freshness.trace = Some(Rc::clone(&trace));
        let mut backend = backend_exit(0);
        backend.trace = Some(Rc::clone(&trace));

        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();
        assert!(report.backend_was_called());
        assert_eq!(
            *trace.borrow(),
            vec![
                "preflight_source",
                "admit",
                "persist",
                "freshness",
                "backend",
                "postflight_source",
            ]
        );
    }

    #[test]
    fn postflight_capture_failure_does_not_erase_process_evidence() {
        let mut probe = probe(Ok(digest('a')), Err(anyhow::anyhow!("probe failed")));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AdapterRunReport::BackendEntered {
                process, postflight, ..
            } => {
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
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = FakeBackend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::SpawnFailed {
                    error_digest: digest('6'),
                },
                stdout_sha256: digest('3'),
                stderr_sha256: digest('4'),
            }),
            calls: 0,
            trace: None,
        };
        let report = run_with_backend(
            intent(),
            bindings(),
            &mut probe,
            &mut gate,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
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
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = FakeBackend {
            capture: Some(ProcessCapture {
                terminal: ProcessTerminal::Exited { code: 0 },
                stdout_sha256: "not-a-digest".into(),
                stderr_sha256: digest('4'),
            }),
            calls: 0,
            trace: None,
        };

        assert!(
            run_with_backend(
                intent(),
                bindings(),
                &mut probe,
                &mut gate,
                &mut persistence,
                &mut freshness,
                &mut backend,
            )
            .is_err()
        );
        assert_eq!(backend.calls, 1);
        assert_eq!(probe.post_calls, 1);
    }

    #[test]
    fn tampered_intent_rejects_before_any_external_boundary() {
        let mut tampered = intent();
        tampered.adapter_semantics_digest = digest('8');
        let mut probe = probe(Ok(digest('a')), Ok(digest('a')));
        let mut gate = gate_allowed();
        let mut persistence = persistence_allowed();
        let mut freshness = freshness_allowed();
        let mut backend = backend_exit(0);
        assert!(
            run_with_backend(
                tampered,
                bindings(),
                &mut probe,
                &mut gate,
                &mut persistence,
                &mut freshness,
                &mut backend,
            )
            .is_err()
        );
        assert_eq!(probe.pre_calls, 0);
        assert_eq!(gate.calls, 0);
        assert_eq!(persistence.calls, 0);
        assert_eq!(freshness.calls, 0);
        assert_eq!(backend.calls, 0);
    }
}
