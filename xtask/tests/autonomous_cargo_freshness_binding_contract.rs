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
#[path = "../src/autonomous_cargo_adapter_state.rs"]
mod autonomous_cargo_adapter_state;
#[path = "../src/autonomous_cargo_effect_binding.rs"]
mod autonomous_cargo_effect_binding;
#[path = "../src/autonomous_cargo_freshness_binding.rs"]
mod autonomous_cargo_freshness_binding;

use std::cell::RefCell;
use std::rc::Rc;

use autonomous_cargo_adapter_state::{
    AutonomousAdapterRunReport, AutonomousEligibilityEvidence, AutonomousEligibilityGate,
    AutonomousEligibilityRejection,
};
use autonomous_cargo_effect_binding::{
    CommitmentAwareEffectAdmission, CommitmentAwareEffectEntryGate,
    CommitmentBoundAutonomousRunReport, ExpectedEffectAdmissionBinding,
};
use autonomous_cargo_freshness_binding::{
    CanonicalFreshnessV2Evidence, CanonicalFreshnessV2Gate, FreshnessBoundAutonomousRunReport,
    run_freshness_bound_autonomous,
};
use cargo_adapter_state::{
    AdapterRunReport, AdmissionPersistence, AdmissionPersistenceAck,
    AdmissionPersistenceRejection, CargoProcessBackend, EffectAdmission, EffectRejection,
    PostflightCapture, PreSpawnFreshnessRejection, ProcessCapture, ProcessTerminal,
    RepositorySourceProbe, ValidatedPreflightBindings,
};
use cargo_execution_contract::{CargoExecutionIntent, CargoExecutionIntentSpec, build_intent};

type Trace = Rc<RefCell<Vec<&'static str>>>;

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

fn expectation() -> ExpectedEffectAdmissionBinding {
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

fn admission() -> CommitmentAwareEffectAdmission {
    let expected = expectation();
    CommitmentAwareEffectAdmission {
        receipt_digest: digest('f'),
        action_binding: expected.action_binding,
        authority_snapshot_digest: expected.authority_snapshot_digest,
        adapter_semantics_digest: expected.adapter_semantics_digest,
        effect_admission_commitment_digest: expected.effect_admission_commitment_digest,
    }
}

fn freshness() -> CanonicalFreshnessV2Evidence {
    CanonicalFreshnessV2Evidence {
        freshness_receipt_id: digest('0'),
        prepared_intent_id: digest('b'),
        runtime_binding_id: digest('1'),
        materialization_receipt_id: digest('r'),
        tool_attestation_id: digest('t'),
        git_worktree_state_id: digest('6'),
    }
}

fn bindings() -> ValidatedPreflightBindings {
    ValidatedPreflightBindings {
        build_context_id: digest('2'),
        invocation_id: digest('3'),
        effect_policy_id: digest('4'),
        git_worktree_state_id: Some(digest('6')),
    }
}

struct FakeEligibility {
    decision: Option<Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection>>,
    trace: Trace,
}

impl AutonomousEligibilityGate for FakeEligibility {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
    ) -> Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection> {
        self.trace.borrow_mut().push("eligibility");
        self.decision.take().expect("eligibility decision configured")
    }
}

struct FakeSource {
    pre: Option<anyhow::Result<String>>,
    post: Option<anyhow::Result<String>>,
    trace: Trace,
}

impl RepositorySourceProbe for FakeSource {
    fn preflight_source_id(&mut self) -> anyhow::Result<String> {
        self.trace.borrow_mut().push("preflight_source");
        self.pre.take().expect("preflight source configured")
    }

    fn postflight_source_id(&mut self) -> anyhow::Result<String> {
        self.trace.borrow_mut().push("postflight_source");
        self.post.take().expect("postflight source configured")
    }
}

struct FakeEffect {
    decision: Option<Result<CommitmentAwareEffectAdmission, EffectRejection>>,
    trace: Trace,
}

impl CommitmentAwareEffectEntryGate for FakeEffect {
    fn admit(
        &mut self,
        _intent: &CargoExecutionIntent,
    ) -> Result<CommitmentAwareEffectAdmission, EffectRejection> {
        self.trace.borrow_mut().push("effect");
        self.decision.take().expect("effect decision configured")
    }
}

struct FakePersistence {
    decision: Option<Result<AdmissionPersistenceAck, AdmissionPersistenceRejection>>,
    trace: Trace,
}

impl AdmissionPersistence for FakePersistence {
    fn persist(
        &mut self,
        _intent: &CargoExecutionIntent,
        _admission: &EffectAdmission,
    ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection> {
        self.trace.borrow_mut().push("persistence");
        self.decision.take().expect("persistence decision configured")
    }
}

struct FakeFreshnessV2 {
    decision: Option<Result<CanonicalFreshnessV2Evidence, PreSpawnFreshnessRejection>>,
    trace: Trace,
    seen_bindings: Rc<RefCell<Vec<(String, String)>>>,
}

impl CanonicalFreshnessV2Gate for FakeFreshnessV2 {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        persistence: &AdmissionPersistenceAck,
    ) -> Result<CanonicalFreshnessV2Evidence, PreSpawnFreshnessRejection> {
        self.trace.borrow_mut().push("freshness_v2");
        self.seen_bindings.borrow_mut().push((
            admission.receipt_digest.clone(),
            persistence.persistence_ack_digest.clone(),
        ));
        self.decision.take().expect("freshness decision configured")
    }
}

struct FakeBackend {
    capture: Option<ProcessCapture>,
    trace: Trace,
}

impl CargoProcessBackend for FakeBackend {
    fn execute(&mut self, _intent: &CargoExecutionIntent) -> ProcessCapture {
        self.trace.borrow_mut().push("backend");
        self.capture.take().expect("backend capture configured")
    }
}

struct Rig {
    trace: Trace,
    seen_freshness_bindings: Rc<RefCell<Vec<(String, String)>>>,
    eligibility: FakeEligibility,
    source: FakeSource,
    effect: FakeEffect,
    persistence: FakePersistence,
    freshness: FakeFreshnessV2,
    backend: FakeBackend,
}

impl Rig {
    fn new() -> Self {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let seen_freshness_bindings = Rc::new(RefCell::new(Vec::new()));
        let expected = expectation();
        Self {
            trace: Rc::clone(&trace),
            seen_freshness_bindings: Rc::clone(&seen_freshness_bindings),
            eligibility: FakeEligibility {
                decision: Some(Ok(AutonomousEligibilityEvidence {
                    eligibility_id: expected.eligibility_id,
                    prepared_intent_id: expected.prepared_intent_id,
                    execution_intent_id: expected.execution_intent_id,
                })),
                trace: Rc::clone(&trace),
            },
            source: FakeSource {
                pre: Some(Ok(digest('1'))),
                post: Some(Ok(digest('1'))),
                trace: Rc::clone(&trace),
            },
            effect: FakeEffect {
                decision: Some(Ok(admission())),
                trace: Rc::clone(&trace),
            },
            persistence: FakePersistence {
                decision: Some(Ok(AdmissionPersistenceAck {
                    admission_receipt_digest: digest('f'),
                    persistence_ack_digest: digest('9'),
                })),
                trace: Rc::clone(&trace),
            },
            freshness: FakeFreshnessV2 {
                decision: Some(Ok(freshness())),
                trace: Rc::clone(&trace),
                seen_bindings: seen_freshness_bindings,
            },
            backend: FakeBackend {
                capture: Some(ProcessCapture {
                    terminal: ProcessTerminal::Exited { code: 0 },
                    stdout_sha256: digest('a'),
                    stderr_sha256: digest('b'),
                }),
                trace,
            },
        }
    }

    fn run(&mut self) -> anyhow::Result<FreshnessBoundAutonomousRunReport> {
        run_freshness_bound_autonomous(
            intent(),
            bindings(),
            expectation(),
            &mut self.eligibility,
            &mut self.effect,
            &mut self.source,
            &mut self.persistence,
            &mut self.freshness,
            &mut self.backend,
        )
    }
}

#[test]
fn exact_freshness_subject_preserves_full_order_and_receipt_identity() {
    let mut rig = Rig::new();
    let report = rig.run().unwrap();

    match report {
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness: Some(accepted),
            inner:
                CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                    inner:
                        AutonomousAdapterRunReport::AfterEligibility {
                            inner:
                                AdapterRunReport::BackendEntered {
                                    effect_admission_digest,
                                    persistence_ack_digest,
                                    pre_spawn_freshness_digest,
                                    postflight: PostflightCapture::Captured { .. },
                                    ..
                                },
                            ..
                        },
                    ..
                },
            ..
        } => {
            assert_eq!(accepted, freshness());
            assert_eq!(effect_admission_digest, digest('f'));
            assert_eq!(persistence_ack_digest, digest('9'));
            assert_eq!(pre_spawn_freshness_digest, digest('0'));
        }
        other => panic!("unexpected report: {other:?}"),
    }

    assert_eq!(
        rig.trace.borrow().as_slice(),
        [
            "eligibility",
            "preflight_source",
            "effect",
            "persistence",
            "freshness_v2",
            "backend",
            "postflight_source",
        ]
    );
    assert_eq!(
        rig.seen_freshness_bindings.borrow().as_slice(),
        [(digest('f'), digest('9'))]
    );
}

#[test]
fn freshness_for_another_prepared_subject_rejects_before_backend_and_preserves_evidence() {
    let mut rig = Rig::new();
    rig.freshness
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .prepared_intent_id = digest('0');

    let report = rig.run().unwrap();
    match report {
        FreshnessBoundAutonomousRunReport::FreshnessBindingRejected {
            expected_prepared_intent_id,
            actual,
            inner:
                CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                    inner:
                        AutonomousAdapterRunReport::AfterEligibility {
                            inner: AdapterRunReport::FreshnessRejected { .. },
                            ..
                        },
                    ..
                },
            ..
        } => {
            assert_eq!(expected_prepared_intent_id, digest('b'));
            assert_eq!(actual.prepared_intent_id, digest('0'));
            assert_eq!(actual.freshness_receipt_id, digest('0'));
        }
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence", "freshness_v2"]
    );
}

#[test]
fn malformed_canonical_freshness_evidence_fails_closed_before_backend() {
    let mut rig = Rig::new();
    rig.freshness
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .freshness_receipt_id = "not-a-digest".into();

    assert!(rig.run().is_err());
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence", "freshness_v2"]
    );
}

#[test]
fn persistence_failure_keeps_freshness_provider_unreachable() {
    let mut rig = Rig::new();
    rig.persistence.decision = Some(Err(AdmissionPersistenceRejection {
        reason: "durability unavailable".into(),
    }));

    let report = rig.run().unwrap();
    assert!(matches!(
        report,
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness: None,
            inner:
                CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                    inner:
                        AutonomousAdapterRunReport::AfterEligibility {
                            inner: AdapterRunReport::AdmissionPersistenceRejected { .. },
                            ..
                        },
                    ..
                },
            ..
        }
    ));
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence"]
    );
}

#[test]
fn admission_binding_failure_keeps_freshness_provider_unreachable() {
    let mut rig = Rig::new();
    rig.effect
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .action_binding = digest('0');

    let report = rig.run().unwrap();
    assert!(matches!(
        report,
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness: None,
            inner: CommitmentBoundAutonomousRunReport::AdmissionBindingRejected { .. },
            ..
        }
    ));
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect"]
    );
}

#[test]
fn ordinary_freshness_rejection_preserves_prior_evidence_without_backend_entry() {
    let mut rig = Rig::new();
    rig.freshness.decision = Some(Err(PreSpawnFreshnessRejection {
        reason: "freshness probe unavailable".into(),
    }));

    let report = rig.run().unwrap();
    assert!(matches!(
        report,
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness: None,
            inner:
                CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
                    accepted_admission: Some(_),
                    inner:
                        AutonomousAdapterRunReport::AfterEligibility {
                            inner: AdapterRunReport::FreshnessRejected { .. },
                            ..
                        },
                    ..
                },
            ..
        }
    ));
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence", "freshness_v2"]
    );
}

#[test]
fn accepted_freshness_retains_all_constituent_audit_ids() {
    let mut rig = Rig::new();
    let report = rig.run().unwrap();
    match report {
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            accepted_freshness: Some(value),
            ..
        } => {
            assert_eq!(value.runtime_binding_id, digest('1'));
            assert_eq!(value.materialization_receipt_id, digest('r'));
            assert_eq!(value.tool_attestation_id, digest('t'));
            assert_eq!(value.git_worktree_state_id, digest('6'));
        }
        other => panic!("unexpected report: {other:?}"),
    }
}
