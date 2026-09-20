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

use std::cell::RefCell;
use std::rc::Rc;

use autonomous_cargo_adapter_state::{
    AutonomousAdapterRunReport, AutonomousEligibilityEvidence, AutonomousEligibilityGate,
    AutonomousEligibilityRejection,
};
use autonomous_cargo_effect_binding::{
    CommitmentAwareEffectAdmission, CommitmentAwareEffectEntryGate,
    CommitmentBoundAutonomousRunReport, ExpectedEffectAdmissionBinding,
    run_commitment_bound_autonomous,
};
use cargo_adapter_state::{
    AdapterRunReport, AdmissionPersistence, AdmissionPersistenceAck,
    AdmissionPersistenceRejection, CargoProcessBackend, EffectAdmission, EffectRejection,
    PostflightCapture, PreSpawnFreshnessApproval, PreSpawnFreshnessGate,
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

struct FakeFreshness {
    decision: Option<Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection>>,
    trace: Trace,
}

impl PreSpawnFreshnessGate for FakeFreshness {
    fn verify(
        &mut self,
        _intent: &CargoExecutionIntent,
        _admission: &EffectAdmission,
        _persistence: &AdmissionPersistenceAck,
    ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection> {
        self.trace.borrow_mut().push("freshness");
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
        let expected = expectation();
        Self {
            trace: Rc::clone(&trace),
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
            freshness: FakeFreshness {
                decision: Some(Ok(PreSpawnFreshnessApproval {
                    admission_receipt_digest: digest('f'),
                    persistence_ack_digest: digest('9'),
                    freshness_evidence_digest: digest('0'),
                })),
                trace: Rc::clone(&trace),
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
        expected: ExpectedEffectAdmissionBinding,
    ) -> anyhow::Result<CommitmentBoundAutonomousRunReport> {
        run_commitment_bound_autonomous(
            intent(),
            bindings(),
            expected,
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
fn exact_commitment_preserves_existing_full_ordering() {
    let mut rig = Rig::new();
    let report = rig.run(expectation()).unwrap();

    match report {
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
            accepted_admission: Some(accepted),
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
        } => {
            assert_eq!(accepted, admission());
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
            "freshness",
            "backend",
            "postflight_source",
        ]
    );
}

#[test]
fn expected_intent_or_adapter_subject_mismatch_rejects_before_eligibility() {
    for kind in ["intent", "adapter_semantics"] {
        let mut rig = Rig::new();
        let mut expected = expectation();
        match kind {
            "intent" => expected.execution_intent_id = digest('0'),
            "adapter_semantics" => expected.adapter_semantics_digest = digest('0'),
            _ => unreachable!(),
        }

        assert!(matches!(
            rig.run(expected).unwrap(),
            CommitmentBoundAutonomousRunReport::ExpectationRejectedBeforeEligibility { .. }
        ));
        assert!(rig.trace.borrow().is_empty());
    }
}

#[test]
fn eligibility_or_prepared_lineage_substitution_rejects_before_source_preflight() {
    for kind in ["eligibility", "prepared"] {
        let mut rig = Rig::new();
        let evidence = rig
            .eligibility
            .decision
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap();
        match kind {
            "eligibility" => evidence.eligibility_id = digest('0'),
            "prepared" => evidence.prepared_intent_id = digest('0'),
            _ => unreachable!(),
        }

        let report = rig.run(expectation()).unwrap();
        assert!(matches!(
            report,
            CommitmentBoundAutonomousRunReport::EligibilityBindingRejected { .. }
        ));
        assert_eq!(rig.trace.borrow().as_slice(), ["eligibility"]);
    }
}

#[test]
fn each_admission_commitment_dimension_rejects_before_persistence() {
    for kind in ["action", "authority", "semantics", "commitment"] {
        let mut rig = Rig::new();
        let actual = rig.effect.decision.as_mut().unwrap().as_mut().unwrap();
        match kind {
            "action" => actual.action_binding = digest('0'),
            "authority" => actual.authority_snapshot_digest = digest('0'),
            "semantics" => actual.adapter_semantics_digest = digest('0'),
            "commitment" => actual.effect_admission_commitment_digest = digest('0'),
            _ => unreachable!(),
        }

        let report = rig.run(expectation()).unwrap();
        match report {
            CommitmentBoundAutonomousRunReport::AdmissionBindingRejected { actual, .. } => {
                assert_eq!(actual.receipt_digest, digest('f'));
            }
            other => panic!("unexpected report for {kind}: {other:?}"),
        }
        assert_eq!(
            rig.trace.borrow().as_slice(),
            ["eligibility", "preflight_source", "effect"]
        );
    }
}

#[test]
fn malformed_positive_admission_evidence_fails_closed_before_persistence() {
    let mut rig = Rig::new();
    rig.effect
        .decision
        .as_mut()
        .unwrap()
        .as_mut()
        .unwrap()
        .effect_admission_commitment_digest = "not-a-digest".into();

    assert!(rig.run(expectation()).is_err());
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect"]
    );
}

#[test]
fn ordinary_eligibility_denial_never_reaches_source_or_effect() {
    let mut rig = Rig::new();
    rig.eligibility.decision = Some(Err(AutonomousEligibilityRejection {
        reason: "profile denied".into(),
    }));

    let report = rig.run(expectation()).unwrap();
    assert!(matches!(
        report,
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
            accepted_admission: None,
            inner: AutonomousAdapterRunReport::EligibilityRejected { .. },
            ..
        }
    ));
    assert_eq!(rig.trace.borrow().as_slice(), ["eligibility"]);
}

#[test]
fn downstream_persistence_rejection_retains_accepted_admission() {
    let mut rig = Rig::new();
    rig.persistence.decision = Some(Err(AdmissionPersistenceRejection {
        reason: "durability unavailable".into(),
    }));

    let report = rig.run(expectation()).unwrap();
    match report {
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
            accepted_admission: Some(accepted),
            inner:
                AutonomousAdapterRunReport::AfterEligibility {
                    inner: AdapterRunReport::AdmissionPersistenceRejected { .. },
                    ..
                },
            ..
        } => assert_eq!(accepted, admission()),
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(
        rig.trace.borrow().as_slice(),
        ["eligibility", "preflight_source", "effect", "persistence"]
    );
}

#[test]
fn downstream_freshness_rejection_retains_accepted_admission() {
    let mut rig = Rig::new();
    rig.freshness.decision = Some(Err(PreSpawnFreshnessRejection {
        reason: "freshness drift".into(),
    }));

    let report = rig.run(expectation()).unwrap();
    match report {
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
            accepted_admission: Some(accepted),
            inner:
                AutonomousAdapterRunReport::AfterEligibility {
                    inner: AdapterRunReport::FreshnessRejected { .. },
                    ..
                },
            ..
        } => assert_eq!(accepted, admission()),
        other => panic!("unexpected report: {other:?}"),
    }
    assert_eq!(
        rig.trace.borrow().as_slice(),
        [
            "eligibility",
            "preflight_source",
            "effect",
            "persistence",
            "freshness",
        ]
    );
}
