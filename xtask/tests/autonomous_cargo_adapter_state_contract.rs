#![allow(dead_code)]

#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/repository_snapshot_diff.rs"]
mod repository_snapshot_diff;
#[path = "../src/repository_effect_policy.rs"]
mod repository_effect_policy;
#[path = "../src/cargo_execution_contract.rs"]
mod cargo_execution_contract;
#[path = "../src/cargo_adapter_state.rs"]
mod cargo_adapter_state;
#[path = "../src/autonomous_cargo_adapter_state.rs"]
mod autonomous_cargo_adapter_state;

use std::cell::RefCell;
use std::rc::Rc;

use autonomous_cargo_adapter_state::{
    AutonomousAdapterRunReport, AutonomousEligibilityEvidence, AutonomousEligibilityGate,
    AutonomousEligibilityRejection, run_autonomous_with_backend,
};
use cargo_adapter_state::{
    AdapterRunReport, AdmissionPersistence, AdmissionPersistenceAck,
    AdmissionPersistenceRejection, CargoProcessBackend, EffectAdmission, EffectEntryGate,
    EffectRejection, PreSpawnFreshnessApproval, PreSpawnFreshnessGate,
    PreSpawnFreshnessRejection, ProcessCapture, ProcessTerminal, RepositorySourceProbe,
    ValidatedPreflightBindings,
};
use cargo_execution_contract::{CargoExecutionIntent, CargoExecutionIntentSpec, build_intent};

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
            plan_id: Some(digest('7')),
            transaction_id: Some(digest('8')),
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

struct FakeEligibility {
    decision: Option<Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection>>,
    calls: usize,
    trace: Trace,
}

impl AutonomousEligibilityGate for FakeEligibility {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
    ) -> Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection> {
        self.calls += 1;
        self.trace.borrow_mut().push("eligibility");
        self.decision.take().expect("eligibility decision configured")
    }
}

struct FakeProbe {
    pre: Option<anyhow::Result<String>>,
    post: Option<anyhow::Result<String>>,
    pre_calls: usize,
    post_calls: usize,
    trace: Trace,
}

impl RepositorySourceProbe for FakeProbe {
    fn preflight_source_id(&mut self) -> anyhow::Result<String> {
        self.pre_calls += 1;
        self.trace.borrow_mut().push("preflight_source");
        self.pre.take().expect("preflight source configured")
    }

    fn postflight_source_id(&mut self) -> anyhow::Result<String> {
        self.post_calls += 1;
        self.trace.borrow_mut().push("postflight_source");
        self.post.take().expect("postflight source configured")
    }
}

struct FakeEffect {
    decision: Option<Result<EffectAdmission, EffectRejection>>,
    calls: usize,
    trace: Trace,
}

impl EffectEntryGate for FakeEffect {
    fn admit(
        &mut self,
        _intent: &CargoExecutionIntent,
    ) -> Result<EffectAdmission, EffectRejection> {
        self.calls += 1;
        self.trace.borrow_mut().push("effect");
        self.decision.take().expect("effect decision configured")
    }
}

struct FakePersistence {
    decision: Option<Result<AdmissionPersistenceAck, AdmissionPersistenceRejection>>,
    calls: usize,
    trace: Trace,
}

impl AdmissionPersistence for FakePersistence {
    fn persist(
        &mut self,
        _intent: &CargoExecutionIntent,
        _admission: &EffectAdmission,
    ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection> {
        self.calls += 1;
        self.trace.borrow_mut().push("persistence");
        self.decision.take().expect("persistence decision configured")
    }
}

struct FakeFreshness {
    decision: Option<Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection>>,
    calls: usize,
    trace: Trace,
}

impl PreSpawnFreshnessGate for FakeFreshness {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
        _admission: &EffectAdmission,
        _persistence: &AdmissionPersistenceAck,
    ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection> {
        self.calls += 1;
        self.trace.borrow_mut().push("freshness");
        self.decision.take().expect("freshness decision configured")
    }
}

struct FakeBackend {
    capture: Option<ProcessCapture>,
    calls: usize,
    trace: Trace,
}

impl CargoProcessBackend for FakeBackend {
    fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
        self.calls += 1;
        self.trace.borrow_mut().push("backend");
        self.capture.take().expect("backend capture configured")
    }
}

fn positive_eligibility(intent: &CargoExecutionIntent, trace: Trace) -> FakeEligibility {
    FakeEligibility {
        decision: Some(Ok(AutonomousEligibilityEvidence {
            eligibility_id: digest('e'),
            prepared_intent_id: digest('f'),
            execution_intent_id: intent.intent_id.clone(),
        })),
        calls: 0,
        trace,
    }
}

fn probe(pre: String, trace: Trace) -> FakeProbe {
    FakeProbe {
        pre: Some(Ok(pre)),
        post: Some(Ok(digest('a'))),
        pre_calls: 0,
        post_calls: 0,
        trace,
    }
}

fn positive_effect(trace: Trace) -> FakeEffect {
    FakeEffect {
        decision: Some(Ok(EffectAdmission {
            receipt_digest: digest('2'),
        })),
        calls: 0,
        trace,
    }
}

fn positive_persistence(trace: Trace) -> FakePersistence {
    FakePersistence {
        decision: Some(Ok(AdmissionPersistenceAck {
            admission_receipt_digest: digest('2'),
            persistence_ack_digest: digest('3'),
        })),
        calls: 0,
        trace,
    }
}

fn positive_freshness(trace: Trace) -> FakeFreshness {
    FakeFreshness {
        decision: Some(Ok(PreSpawnFreshnessApproval {
            admission_receipt_digest: digest('2'),
            persistence_ack_digest: digest('3'),
            freshness_evidence_digest: digest('4'),
        })),
        calls: 0,
        trace,
    }
}

fn backend(trace: Trace) -> FakeBackend {
    FakeBackend {
        capture: Some(ProcessCapture {
            terminal: ProcessTerminal::Exited { code: 0 },
            stdout_sha256: digest('5'),
            stderr_sha256: digest('6'),
        }),
        calls: 0,
        trace,
    }
}

#[test]
fn eligibility_denial_keeps_all_inner_boundaries_unreachable() {
    let trace = Rc::new(RefCell::new(vec![]));
    let mut eligibility = FakeEligibility {
        decision: Some(Err(AutonomousEligibilityRejection {
            reason: "not autonomous eligible".into(),
        })),
        calls: 0,
        trace: Rc::clone(&trace),
    };
    let mut probe = probe(digest('a'), Rc::clone(&trace));
    let mut effect = positive_effect(Rc::clone(&trace));
    let mut persistence = positive_persistence(Rc::clone(&trace));
    let mut freshness = positive_freshness(Rc::clone(&trace));
    let mut backend = backend(Rc::clone(&trace));

    let report = run_autonomous_with_backend(
        intent(),
        bindings(),
        &mut eligibility,
        &mut probe,
        &mut effect,
        &mut persistence,
        &mut freshness,
        &mut backend,
    )
    .unwrap();

    assert!(matches!(report, AutonomousAdapterRunReport::EligibilityRejected { .. }));
    assert_eq!(*trace.borrow(), vec!["eligibility"]);
    assert_eq!(probe.pre_calls, 0);
    assert_eq!(effect.calls, 0);
    assert_eq!(persistence.calls, 0);
    assert_eq!(freshness.calls, 0);
    assert_eq!(backend.calls, 0);
}

#[test]
fn malformed_eligibility_evidence_fails_before_inner_boundaries() {
    let trace = Rc::new(RefCell::new(vec![]));
    let current = intent();
    let mut eligibility = FakeEligibility {
        decision: Some(Ok(AutonomousEligibilityEvidence {
            eligibility_id: "not-a-digest".into(),
            prepared_intent_id: digest('f'),
            execution_intent_id: current.intent_id.clone(),
        })),
        calls: 0,
        trace: Rc::clone(&trace),
    };
    let mut probe = probe(digest('a'), Rc::clone(&trace));
    let mut effect = positive_effect(Rc::clone(&trace));
    let mut persistence = positive_persistence(Rc::clone(&trace));
    let mut freshness = positive_freshness(Rc::clone(&trace));
    let mut backend = backend(Rc::clone(&trace));

    assert!(
        run_autonomous_with_backend(
            current,
            bindings(),
            &mut eligibility,
            &mut probe,
            &mut effect,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .is_err()
    );
    assert_eq!(*trace.borrow(), vec!["eligibility"]);
    assert_eq!(probe.pre_calls, 0);
    assert_eq!(effect.calls, 0);
    assert_eq!(backend.calls, 0);
}

#[test]
fn cross_intent_eligibility_rejects_with_positive_lineage_preserved() {
    let trace = Rc::new(RefCell::new(vec![]));
    let current = intent();
    let mut eligibility = FakeEligibility {
        decision: Some(Ok(AutonomousEligibilityEvidence {
            eligibility_id: digest('e'),
            prepared_intent_id: digest('f'),
            execution_intent_id: digest('0'),
        })),
        calls: 0,
        trace: Rc::clone(&trace),
    };
    let mut probe = probe(digest('a'), Rc::clone(&trace));
    let mut effect = positive_effect(Rc::clone(&trace));
    let mut persistence = positive_persistence(Rc::clone(&trace));
    let mut freshness = positive_freshness(Rc::clone(&trace));
    let mut backend = backend(Rc::clone(&trace));

    let report = run_autonomous_with_backend(
        current,
        bindings(),
        &mut eligibility,
        &mut probe,
        &mut effect,
        &mut persistence,
        &mut freshness,
        &mut backend,
    )
    .unwrap();

    match report {
        AutonomousAdapterRunReport::EligibilityEvidenceRejected {
            eligibility_id,
            prepared_intent_id,
            ..
        } => {
            assert_eq!(eligibility_id, digest('e'));
            assert_eq!(prepared_intent_id, digest('f'));
        }
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(*trace.borrow(), vec!["eligibility"]);
    assert_eq!(probe.pre_calls, 0);
    assert_eq!(effect.calls, 0);
    assert_eq!(backend.calls, 0);
}

#[test]
fn stale_source_after_eligibility_retains_eligibility_lineage() {
    let trace = Rc::new(RefCell::new(vec![]));
    let current = intent();
    let mut eligibility = positive_eligibility(&current, Rc::clone(&trace));
    let mut probe = probe(digest('0'), Rc::clone(&trace));
    let mut effect = positive_effect(Rc::clone(&trace));
    let mut persistence = positive_persistence(Rc::clone(&trace));
    let mut freshness = positive_freshness(Rc::clone(&trace));
    let mut backend = backend(Rc::clone(&trace));

    let report = run_autonomous_with_backend(
        current,
        bindings(),
        &mut eligibility,
        &mut probe,
        &mut effect,
        &mut persistence,
        &mut freshness,
        &mut backend,
    )
    .unwrap();

    match report {
        AutonomousAdapterRunReport::AfterEligibility {
            eligibility_id,
            prepared_intent_id,
            inner: AdapterRunReport::RejectedBeforeEffect { .. },
            ..
        } => {
            assert_eq!(eligibility_id, digest('e'));
            assert_eq!(prepared_intent_id, digest('f'));
        }
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(*trace.borrow(), vec!["eligibility", "preflight_source"]);
    assert_eq!(effect.calls, 0);
    assert_eq!(backend.calls, 0);
}

#[test]
fn effect_and_post_admission_rejections_retain_eligibility_lineage() {
    for stage in ["effect", "persistence", "freshness"] {
        let trace = Rc::new(RefCell::new(vec![]));
        let current = intent();
        let mut eligibility = positive_eligibility(&current, Rc::clone(&trace));
        let mut probe = probe(digest('a'), Rc::clone(&trace));
        let mut effect = if stage == "effect" {
            FakeEffect {
                decision: Some(Err(EffectRejection {
                    reason: "effect denied".into(),
                })),
                calls: 0,
                trace: Rc::clone(&trace),
            }
        } else {
            positive_effect(Rc::clone(&trace))
        };
        let mut persistence = if stage == "persistence" {
            FakePersistence {
                decision: Some(Err(AdmissionPersistenceRejection {
                    reason: "persistence denied".into(),
                })),
                calls: 0,
                trace: Rc::clone(&trace),
            }
        } else {
            positive_persistence(Rc::clone(&trace))
        };
        let mut freshness = if stage == "freshness" {
            FakeFreshness {
                decision: Some(Err(PreSpawnFreshnessRejection {
                    reason: "freshness denied".into(),
                })),
                calls: 0,
                trace: Rc::clone(&trace),
            }
        } else {
            positive_freshness(Rc::clone(&trace))
        };
        let mut backend = backend(Rc::clone(&trace));

        let report = run_autonomous_with_backend(
            current,
            bindings(),
            &mut eligibility,
            &mut probe,
            &mut effect,
            &mut persistence,
            &mut freshness,
            &mut backend,
        )
        .unwrap();

        match report {
            AutonomousAdapterRunReport::AfterEligibility {
                eligibility_id,
                prepared_intent_id,
                inner,
                ..
            } => {
                assert_eq!(eligibility_id, digest('e'));
                assert_eq!(prepared_intent_id, digest('f'));
                match (stage, inner) {
                    ("effect", AdapterRunReport::EffectRejected { .. })
                    | ("persistence", AdapterRunReport::AdmissionPersistenceRejected { .. })
                    | ("freshness", AdapterRunReport::FreshnessRejected { .. }) => {}
                    (_, other) => panic!("unexpected inner report at {stage}: {other:?}"),
                }
            }
            other => panic!("unexpected report: {other:?}"),
        }
        assert_eq!(backend.calls, 0);
    }
}

#[test]
fn successful_path_preserves_all_lineage_and_exact_call_order() {
    let trace = Rc::new(RefCell::new(vec![]));
    let current = intent();
    let expected_intent_id = current.intent_id.clone();
    let mut eligibility = positive_eligibility(&current, Rc::clone(&trace));
    let mut probe = probe(digest('a'), Rc::clone(&trace));
    let mut effect = positive_effect(Rc::clone(&trace));
    let mut persistence = positive_persistence(Rc::clone(&trace));
    let mut freshness = positive_freshness(Rc::clone(&trace));
    let mut backend = backend(Rc::clone(&trace));

    let report = run_autonomous_with_backend(
        current,
        bindings(),
        &mut eligibility,
        &mut probe,
        &mut effect,
        &mut persistence,
        &mut freshness,
        &mut backend,
    )
    .unwrap();

    match report {
        AutonomousAdapterRunReport::AfterEligibility {
            intent_id,
            eligibility_id,
            prepared_intent_id,
            inner:
                AdapterRunReport::BackendEntered {
                    effect_admission_digest,
                    persistence_ack_digest,
                    pre_spawn_freshness_digest,
                    process,
                    ..
                },
        } => {
            assert_eq!(intent_id, expected_intent_id);
            assert_eq!(eligibility_id, digest('e'));
            assert_eq!(prepared_intent_id, digest('f'));
            assert_eq!(effect_admission_digest, digest('2'));
            assert_eq!(persistence_ack_digest, digest('3'));
            assert_eq!(pre_spawn_freshness_digest, digest('4'));
            assert_eq!(process.terminal, ProcessTerminal::Exited { code: 0 });
        }
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(
        *trace.borrow(),
        vec![
            "eligibility",
            "preflight_source",
            "effect",
            "persistence",
            "freshness",
            "backend",
            "postflight_source",
        ]
    );
}
