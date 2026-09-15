use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use symthaea_scientific_view_control_plane::{
    CandidateScientificViewControlPlaneTransitionV1, CandidateSourceMigrationV1,
    DeploymentBootstrapEvidenceRefV1, PredecessorAuthorizationEvidenceRefV1,
};
use symthaea_scientific_view_control_plane_occurrence::{
    attempt_observed_control_plane_commit, reconcile_observed_control_plane_commit,
    ControlPlaneCommitPlanV1, ControlPlaneOccurrenceHeadV1, ControlPlaneOccurrenceStoreBindingV1,
    ControlPlaneOccurrenceStoreV1, ControlPlaneStoreCasResultV1,
    ControlPlaneStoreOperationResolutionV1, ObservedControlPlaneCommitOutcomeV1,
    ObservedNonCommitReasonV1, ProposedControlPlaneOccurrenceV1,
    RawControlPlaneOccurrenceRecordV1,
};
use symthaea_scientific_view_profile::{
    AuthoritySourceBindingV1, AuthoritySourceOccurrenceV1, Commitment32,
    RoleSemanticRevisionV1, ScientificAuthorityRoleV1 as Role,
    ScientificViewDeploymentBindingV1, ScientificViewSemanticProfileV1,
};

fn c(byte: u8) -> Commitment32 {
    Commitment32::from_bytes([byte; 32])
}

fn expected(hex: &str) -> Commitment32 {
    let mut bytes = [0u8; 32];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    Commitment32::from_bytes(bytes)
}

fn profile(policy: u8) -> ScientificViewSemanticProfileV1 {
    ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, c(11)).unwrap(),
            RoleSemanticRevisionV1::new(Role::VerifierPolicyHead, c(12)).unwrap(),
        ],
        c(201), c(202), c(203), c(policy),
    ).unwrap()
}

fn binding(p: &ScientificViewSemanticProfileV1, research_id: &str, epoch: u64) -> ScientificViewDeploymentBindingV1 {
    ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        p,
        vec![
            AuthoritySourceBindingV1::new(Role::ResearchSemanticHead, research_id, epoch, c(21)).unwrap(),
            AuthoritySourceBindingV1::new(Role::VerifierPolicyHead, "verifier-policy/store-a", 3, c(22)).unwrap(),
        ],
    ).unwrap()
}

fn genesis() -> (ScientificViewSemanticProfileV1, ScientificViewDeploymentBindingV1, CandidateScientificViewControlPlaneTransitionV1) {
    let p = profile(204);
    let b = binding(&p, "research/store-a", 7);
    let bootstrap = DeploymentBootstrapEvidenceRefV1::new(&p, &b, c(41), c(42)).unwrap();
    let candidate = CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(&p, &b, &bootstrap).unwrap();
    (p, b, candidate)
}

fn migration(
    p: &ScientificViewSemanticProfileV1,
    old_binding: &ScientificViewDeploymentBindingV1,
    predecessor: &CandidateScientificViewControlPlaneTransitionV1,
    research_id: &str,
    epoch: u64,
    auth: u8,
) -> CandidateScientificViewControlPlaneTransitionV1 {
    let new_binding = binding(p, research_id, epoch);
    let old_occ = AuthoritySourceOccurrenceV1::genesis(old_binding, Role::ResearchSemanticHead, c(31)).unwrap();
    let new_occ = AuthoritySourceOccurrenceV1::genesis(&new_binding, Role::ResearchSemanticHead, c(31)).unwrap();
    let mig = CandidateSourceMigrationV1::new(
        p, old_binding, &new_binding, Role::ResearchSemanticHead, &old_occ, &new_occ, c(31)
    ).unwrap();
    let authorization = PredecessorAuthorizationEvidenceRefV1::for_source_migration(
        predecessor, p, &new_binding, &mig, c(auth), c(auth + 1)
    ).unwrap();
    CandidateScientificViewControlPlaneTransitionV1::source_migration_candidate(
        predecessor, p, old_binding, &new_binding, &mig, &authorization
    ).unwrap()
}

fn store_binding(epoch: u64) -> ControlPlaneOccurrenceStoreBindingV1 {
    ControlPlaneOccurrenceStoreBindingV1::new(
        "deployment/site01-a", "lunar/site01", "control-plane/store-a", epoch, c(71)
    ).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelError { Ambiguous, Load, Resolve }
impl fmt::Display for ModelError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") } }
impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Normal, CommitThenError, ErrorWithoutCommit, MalformedAck, MissingReadback, MismatchedReadback, ReadbackError, Conflict, ResolveError }

#[derive(Default)]
struct State {
    frontier: Option<RawControlPlaneOccurrenceRecordV1>,
    operations: BTreeMap<Commitment32, RawControlPlaneOccurrenceRecordV1>,
    cas_calls: usize,
    load_calls: usize,
}

#[derive(Clone)]
struct ModelStore { state: Arc<Mutex<State>>, mode: Mode }
impl ModelStore {
    fn new(mode: Mode) -> Self { Self { state: Arc::new(Mutex::new(State::default())), mode } }
    fn shared(state: Arc<Mutex<State>>, mode: Mode) -> Self { Self { state, mode } }
    fn set_mode(&mut self, mode: Mode) { self.mode = mode; self.state.lock().unwrap().load_calls = 0; }
    fn cas_calls(&self) -> usize { self.state.lock().unwrap().cas_calls }
    fn install(&mut self, record: RawControlPlaneOccurrenceRecordV1) {
        let mut s = self.state.lock().unwrap();
        s.operations.insert(record.occurrence().operation_id().commitment(), record.clone());
        s.frontier = Some(record);
    }
}

impl ControlPlaneOccurrenceStoreV1 for ModelStore {
    type Error = ModelError;

    fn load_frontier(&self, _: &ControlPlaneOccurrenceStoreBindingV1) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> {
        let mut s = self.state.lock().unwrap();
        s.load_calls += 1;
        if s.load_calls >= 2 {
            match self.mode {
                Mode::MissingReadback => return Ok(None),
                Mode::ReadbackError => return Err(ModelError::Load),
                Mode::MismatchedReadback => {
                    if let Some(record) = s.frontier.clone() {
                        return Ok(Some(RawControlPlaneOccurrenceRecordV1::new_unqualified(record.occurrence().clone(), "different-ref").unwrap()));
                    }
                }
                _ => {}
            }
        }
        Ok(s.frontier.clone())
    }

    fn compare_and_swap(
        &mut self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        expected: Option<ControlPlaneOccurrenceHeadV1>,
        proposed: &ProposedControlPlaneOccurrenceV1,
    ) -> Result<ControlPlaneStoreCasResultV1, Self::Error> {
        let mut s = self.state.lock().unwrap();
        s.cas_calls += 1;
        let current = s.frontier.as_ref().map(|r| r.head());
        if self.mode == Mode::Conflict || current != expected {
            return Ok(ControlPlaneStoreCasResultV1::Conflict { actual_frontier: current });
        }
        if self.mode == Mode::ErrorWithoutCommit { return Err(ModelError::Ambiguous); }
        let reference = format!("store-ref-{}", proposed.sequence());
        let record = RawControlPlaneOccurrenceRecordV1::new_unqualified(proposed.clone(), &reference).unwrap();
        s.operations.insert(proposed.operation_id().commitment(), record.clone());
        s.frontier = Some(record);
        if self.mode == Mode::CommitThenError { return Err(ModelError::Ambiguous); }
        if self.mode == Mode::MalformedAck {
            return Ok(ControlPlaneStoreCasResultV1::Applied { store_reference: "bad\nref".into() });
        }
        Ok(ControlPlaneStoreCasResultV1::Applied { store_reference: reference })
    }

    fn resolve_operation(
        &self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        operation_id: symthaea_scientific_view_control_plane_occurrence::ControlPlaneCommitOperationIdV1,
    ) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> {
        if self.mode == Mode::ResolveError { return Err(ModelError::Resolve); }
        let s = self.state.lock().unwrap();
        if let Some(record) = s.operations.get(&operation_id.commitment()) {
            return Ok(ControlPlaneStoreOperationResolutionV1::Found(record.clone()));
        }
        Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent {
            current_frontier: s.frontier.as_ref().map(|r| r.head()),
        })
    }
}

#[test]
fn golden_commit_and_occurrence_vectors_match_oracle() {
    let (p, b, c1) = genesis();
    let store = store_binding(7);
    let p1 = ControlPlaneCommitPlanV1::prepare(store.clone(), &c1, None).unwrap();
    assert_eq!(store.commitment(), expected("cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05"));
    assert_eq!(p1.operation_id().commitment(), expected("fc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0"));
    assert_eq!(p1.occurrence().commitment(), expected("f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0"));
    let c2 = migration(&p, &b, &c1, "research/store-a", 8, 51);
    let p2 = ControlPlaneCommitPlanV1::prepare(store, &c2, Some(p1.occurrence())).unwrap();
    assert_eq!(p2.operation_id().commitment(), expected("31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428"));
    assert_eq!(p2.occurrence().commitment(), expected("629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0"));
}

#[test]
fn store_reprovisioning_changes_operation_and_occurrence_identity() {
    let (_, _, c1) = genesis();
    let a = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let b = ControlPlaneCommitPlanV1::prepare(store_binding(8), &c1, None).unwrap();
    assert_ne!(a.operation_id(), b.operation_id());
    assert_ne!(a.occurrence().commitment(), b.occurrence().commitment());
}

#[test]
fn exact_post_cas_readback_is_required_for_committed_observation() {
    let (_, _, c1) = genesis();
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let expected = plan.occurrence().commitment();
    let mut store = ModelStore::new(Mode::Normal);
    match attempt_observed_control_plane_commit(&mut store, plan).unwrap() {
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(o) => assert_eq!(o.occurrence().commitment(), expected),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn stale_preflight_is_noncommit_without_cas() {
    let (_, _, c1) = genesis();
    let existing = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let mut store = ModelStore::new(Mode::Normal);
    store.install(RawControlPlaneOccurrenceRecordV1::new_unqualified(existing.occurrence().clone(), "existing-ref").unwrap());
    let p2 = profile(205);
    let b2 = binding(&p2, "research/store-a", 7);
    let bootstrap2 = DeploymentBootstrapEvidenceRefV1::new(&p2, &b2, c(43), c(44)).unwrap();
    let different = CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(&p2, &b2, &bootstrap2).unwrap();
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &different, None).unwrap();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(ref n)
            if n.reason() == ObservedNonCommitReasonV1::PreflightFrontierMismatch
    ));
    assert_eq!(store.cas_calls(), 0);
}

#[test]
fn explicit_cas_conflict_is_noncommit() {
    let (_, _, c1) = genesis();
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let mut store = ModelStore::new(Mode::Conflict);
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(ref n)
            if n.reason() == ObservedNonCommitReasonV1::CasConflict
    ));
}

#[test]
fn commit_then_ack_loss_requires_same_operation_reconciliation() {
    let (_, _, c1) = genesis();
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let expected = plan.occurrence().commitment();
    let mut store = ModelStore::new(Mode::CommitThenError);
    let ambiguity = match attempt_observed_control_plane_commit(&mut store, plan).unwrap() {
        ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(a) => a,
        other => panic!("unexpected {other:?}"),
    };
    store.set_mode(Mode::Normal);
    match reconcile_observed_control_plane_commit(&store, ambiguity) {
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(o) => assert_eq!(o.occurrence().commitment(), expected),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(store.cas_calls(), 1);
}

#[test]
fn ambiguous_no_write_reconciles_to_proven_absent() {
    let (_, _, c1) = genesis();
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let mut store = ModelStore::new(Mode::ErrorWithoutCommit);
    let ambiguity = match attempt_observed_control_plane_commit(&mut store, plan).unwrap() {
        ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(a) => a,
        other => panic!("unexpected {other:?}"),
    };
    store.set_mode(Mode::Normal);
    assert!(matches!(
        reconcile_observed_control_plane_commit(&store, ambiguity),
        ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(ref n)
            if n.reason() == ObservedNonCommitReasonV1::ReconciledAbsent
    ));
}

#[test]
fn malformed_ack_and_post_cas_readback_uncertainty_never_become_plain_errors() {
    for mode in [Mode::MalformedAck, Mode::MissingReadback, Mode::MismatchedReadback, Mode::ReadbackError] {
        let (_, _, c1) = genesis();
        let plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
        let mut store = ModelStore::new(mode);
        assert!(attempt_observed_control_plane_commit(&mut store, plan).unwrap().requires_reconciliation());
    }
}

#[test]
fn two_writers_from_same_frontier_cannot_both_observe_successor_commit() {
    let shared = Arc::new(Mutex::new(State::default()));
    let mut a = ModelStore::shared(shared.clone(), Mode::Normal);
    let mut b = ModelStore::shared(shared, Mode::Normal);
    let (p, binding1, c1) = genesis();
    let genesis_plan = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c1, None).unwrap();
    let genesis_occ = genesis_plan.occurrence().clone();
    assert!(matches!(attempt_observed_control_plane_commit(&mut a, genesis_plan).unwrap(), ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)));
    let c2a = migration(&p, &binding1, &c1, "research/store-a", 8, 51);
    let c2b = migration(&p, &binding1, &c1, "research/store-b", 1, 53);
    let pa = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c2a, Some(&genesis_occ)).unwrap();
    let pb = ControlPlaneCommitPlanV1::prepare(store_binding(7), &c2b, Some(&genesis_occ)).unwrap();
    assert!(matches!(attempt_observed_control_plane_commit(&mut a, pa).unwrap(), ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)));
    assert!(matches!(attempt_observed_control_plane_commit(&mut b, pb).unwrap(), ObservedControlPlaneCommitOutcomeV1::ProvenNotCommitted(_)));
}

#[test]
fn public_surface_stops_before_historical_or_live_authority() {
    let source = include_str!("../src/lib.rs");
    let manifest = include_str!("../Cargo.toml");
    assert!(source.contains("ObservedExactControlPlaneCommitV1"));
    assert!(!source.contains("pub struct HistoricalCommittedControlPlaneTransitionV1"));
    assert!(!source.contains("pub struct QualifiedScientificViewControlPlaneHeadV1"));
    assert!(!source.contains("pub fn is_current"));
    assert!(!manifest.contains("serde"));
}
