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
#[path = "../src/autonomous_cargo_persistence_binding.rs"]
mod autonomous_cargo_persistence_binding;

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
};
use autonomous_cargo_persistence_binding::{
    CanonicalAdmissionPersistenceEvidence, CanonicalAdmissionPersistenceGate,
    ExpectedAdmissionPersistenceBinding, PersistenceBoundAutonomousRunReport,
    run_persistence_bound_autonomous,
};
use cargo_adapter_state::{
    AdapterRunReport, AdmissionPersistenceAck, AdmissionPersistenceRejection,
    CargoProcessBackend, EffectAdmission, EffectRejection, PostflightCapture,
    PreSpawnFreshnessRejection, ProcessCapture, ProcessTerminal, RepositorySourceProbe,
    ValidatedPreflightBindings,
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

fn persistence_evidence() -> CanonicalAdmissionPersistenceEvidence {
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
        runtime_binding_id: digest('1'),
        materialization_receipt_id: digest('s'),
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
    decision: Option<Result<CanonicalAdmissionPersistenceEvidence, AdmissionPersistenceRejection>>,
    trace: Trace,
    seen: Rc<RefCell<Vec<(String, String)>>>,
}

impl CanonicalAdmissionPersistenceGate for FakePersistence {
    fn persist(
        &mut self,
        _intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        required_persistence_profile_id: &str,
    ) -> Result<CanonicalAdmissionPersistenceEvidence, AdmissionPersistenceRejection> {
        self.trace.borrow_mut().push("persistence_profile");
        self.seen.borrow_mut().push((
            admission.receipt_digest.clone(),
            required_persistence_profile_id.to_string(),
        ));
        self.decision.take().expect("persistence decision configured")
    }
}

struct FakeFreshness {
    decision: Option<Result<CanonicalFreshnessV2Evidence, PreSpawnFreshnessRejection>>,
    trace: Trace,
    seen: Rc<RefCell<Vec<(String, String)>>>,
}

impl CanonicalFreshnessV2Gate for FakeFreshness {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        persistence: &AdmissionPersistenceAck,
    ) -> Result<CanonicalFreshnessV2Evidence, PreSpawnFreshnessRejection> {
        self.trace.borrow_mut().push("freshness_v2");
        self.seen.borrow_mut().push((
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
    seen_persistence: Rc<RefCell<Vec<(String, String)>>>,
    seen_freshness: Rc<RefCell<Vec<(String, String)>>>,
    eligibility: FakeEligibility,
    source: FakeSource,
    effect: FakeEffect,
    persistence: FakePersistence,
    freshness: FakeFreshness,
    backend: FakeBackend,
}

impl Rig {
    fn new() -> Self {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let seen_persistence = Rc::new(RefCell::new(Vec::new()));
        let seen_freshness = Rc::new(RefCell::new(Vec::new()));
        let expected = effect_expectation();
        Self {
            trace: Rc::clone(&trace),
            seen_persistence: Rc::clone(&seen_persistence),
            seen_freshness: Rc::clone(&seen_freshness),
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
                decision: Some(Ok(persistence_evidence())),
                trace: Rc::clone(&trace),
                seen: seen_persistence,
            },
            freshness: FakeFreshness {
                decision: Some(Ok(freshness())),
                trace: Rc::clone(&trace),
                seen: seen_freshness,
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

    fn run(
        &mut self,
        persistence_expectation: ExpectedAdmissionPersistenceBinding,
    ) -> anyhow::Result<PersistenceBoundAutonomousRunReport> {
        run_persistence_bound_autonomous(
            intent(),
            bindings(),
            effect_expectation(),
            persistence_expectation,
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
fn exact_profile_preserves_full_order_and_binds_actual_admission() {
    let mut rig = Rig::new();
    let report = rig.run(persistence_expectation()).unwrap();

    match report {
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
            accepted_persistence: Some(accepted),
            inner:
                FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
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
                },
            ..
        } => {
            assert_eq!(accepted, persistence_evidence());
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
            "persistence_profile",
            "freshness_v2",
            "backend",
            "postflight_source",
        ]
    );
    assert_eq!(
        rig.seen_persistence.borrow().as_slice(),
        [(digest('f'), digest('p'))]
    );
    assert_eq!(
        rig.seen_freshness.borrow().as_slice(),
        [(digest('f'), digest('9'))]
    );
}

#[test]
fn persistence_expectation_for_another_authority_snapshot_rejects_before_eligibility() {
    let mut rig = Rig::new();
    let mut expected = persistence_expectation();
    expected.authority_snapshot_digest = digest('0');

    assert!(matches!(
        rig.run(expected).unwrap(),
        PersistenceBoundAutonomousRunReport::PersistenceExpectationRejectedBeforeEligibility { .. }
    ));
    assert!(rig.trace.borrow().is_empty());
}

#[test]
fn different_persistence_profile_rejects_before_freshness_and_preserves_evidence() {
    let mut rig = Rig::new();
    rig.persistence
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .persistence_profile_id = digest('0');

    let report = rig.run(persistence_expectation()).unwrap();
    match report {
        PersistenceBoundAutonomousRunReport::PersistenceBindingRejected {
            required_persistence_profile_id,
            actual,
            inner:
                FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
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
                },
            ..
        } => {
            assert_eq!(required_persistence_profile_id, digest('p'));
            assert_eq!(actual.persistence_profile_id, digest('0'));
            assert_eq!(actual.persistence_ack_digest, digest('9'));
        }
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence_profile"]
    );
}

#[test]
fn malformed_persistence_evidence_fails_closed_before_freshness() {
    let mut rig = Rig::new();
    rig.persistence
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .persistence_ack_digest = "not-a-digest".into();

    assert!(rig.run(persistence_expectation()).is_err());
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence_profile"]
    );
}

#[test]
fn ordinary_persistence_rejection_keeps_freshness_unreachable() {
    let mut rig = Rig::new();
    rig.persistence.decision = Some(Err(AdmissionPersistenceRejection {
        reason: "configured sink unavailable".into(),
    }));

    let report = rig.run(persistence_expectation()).unwrap();
    assert!(matches!(
        report,
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
            accepted_persistence: None,
            inner:
                FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
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
                },
            ..
        }
    ));
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence_profile"]
    );
}

#[test]
fn admission_binding_failure_keeps_persistence_provider_unreachable() {
    let mut rig = Rig::new();
    rig.effect
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .action_binding = digest('0');

    let report = rig.run(persistence_expectation()).unwrap();
    assert!(matches!(
        report,
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
            accepted_persistence: None,
            inner:
                FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
                    inner: CommitmentBoundAutonomousRunReport::AdmissionBindingRejected { .. },
                    ..
                },
            ..
        }
    ));
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect"]
    );
}

#[test]
fn accepted_persistence_retains_profile_record_generation_and_optional_evidence() {
    let mut rig = Rig::new();
    let report = rig.run(persistence_expectation()).unwrap();
    match report {
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
            accepted_persistence: Some(value),
            ..
        } => {
            assert_eq!(value.persistence_profile_id, digest('p'));
            assert_eq!(value.persistence_record_id, "record-0001");
            assert_eq!(value.control_plane_id, "control-plane-alpha");
            assert_eq!(value.control_plane_generation, 42);
            assert_eq!(value.evidence_root_digest.as_deref(), Some(digest('q').as_str()));
            assert_eq!(
                value.anti_rollback_evidence_digest.as_deref(),
                Some(digest('r').as_str())
            );
        }
        other => panic!("unexpected report: {other:?}"),
    }
}

#[test]
fn noncanonical_record_or_control_plane_identifiers_fail_closed() {
    for kind in ["record", "control"] {
        let mut rig = Rig::new();
        let actual = rig.persistence.decision.as_mut().unwrap().as_mut().unwrap();
        match kind {
            "record" => actual.persistence_record_id = " record-0001".into(),
            "control" => actual.control_plane_id = "control\nplane".into(),
            _ => unreachable!(),
        }
        assert!(rig.run(persistence_expectation()).is_err());
        assert_eq!(
            rig.trace.borrow().as_slice(),
            ["eligibility", "preflight_source", "effect", "persistence_profile"]
        );
    }
}
