use std::collections::BTreeMap;
use std::fmt;

use symthaea_scientific_view_control_plane::{
    CandidateScientificViewControlPlaneTransitionV1, CandidateSourceMigrationV1,
    DeploymentBootstrapEvidenceRefV1, PredecessorAuthorizationEvidenceRefV1,
};
use symthaea_scientific_view_control_plane_occurrence::{
    attempt_observed_control_plane_commit, ControlPlaneCommitOperationIdV1,
    ControlPlaneCommitPlanV1, ControlPlaneCommitProtocolError, ControlPlaneOccurrenceHeadV1,
    ControlPlaneOccurrenceStoreBindingV1, ControlPlaneOccurrenceStoreV1,
    ControlPlaneStoreCasResultV1, ControlPlaneStoreOperationResolutionV1,
    ObservedControlPlaneCommitOutcomeV1, ProposedControlPlaneOccurrenceV1,
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

fn profile() -> ScientificViewSemanticProfileV1 {
    ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, c(11)).unwrap(),
            RoleSemanticRevisionV1::new(Role::VerifierPolicyHead, c(12)).unwrap(),
        ],
        c(201), c(202), c(203), c(204),
    ).unwrap()
}

fn binding(p: &ScientificViewSemanticProfileV1, id: &str, epoch: u64) -> ScientificViewDeploymentBindingV1 {
    ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        p,
        vec![
            AuthoritySourceBindingV1::new(Role::ResearchSemanticHead, id, epoch, c(21)).unwrap(),
            AuthoritySourceBindingV1::new(Role::VerifierPolicyHead, "verifier-policy/store-a", 3, c(22)).unwrap(),
        ],
    ).unwrap()
}

fn genesis(p: &ScientificViewSemanticProfileV1, b: &ScientificViewDeploymentBindingV1) -> CandidateScientificViewControlPlaneTransitionV1 {
    let bootstrap = DeploymentBootstrapEvidenceRefV1::new(p, b, c(41), c(42)).unwrap();
    CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(p, b, &bootstrap).unwrap()
}

fn migrate(
    p: &ScientificViewSemanticProfileV1,
    old_binding: &ScientificViewDeploymentBindingV1,
    predecessor: &CandidateScientificViewControlPlaneTransitionV1,
    new_id: &str,
    new_epoch: u64,
    auth_byte: u8,
) -> (ScientificViewDeploymentBindingV1, CandidateScientificViewControlPlaneTransitionV1) {
    let new_binding = binding(p, new_id, new_epoch);
    let old_occ = AuthoritySourceOccurrenceV1::genesis(old_binding, Role::ResearchSemanticHead, c(31)).unwrap();
    let new_occ = AuthoritySourceOccurrenceV1::genesis(&new_binding, Role::ResearchSemanticHead, c(31)).unwrap();
    let migration = CandidateSourceMigrationV1::new(
        p, old_binding, &new_binding, Role::ResearchSemanticHead, &old_occ, &new_occ, c(31),
    ).unwrap();
    let authorization = PredecessorAuthorizationEvidenceRefV1::for_source_migration(
        predecessor, p, &new_binding, &migration, c(auth_byte), c(auth_byte + 1),
    ).unwrap();
    let candidate = CandidateScientificViewControlPlaneTransitionV1::source_migration_candidate(
        predecessor, p, old_binding, &new_binding, &migration, &authorization,
    ).unwrap();
    (new_binding, candidate)
}

fn store_binding() -> ControlPlaneOccurrenceStoreBindingV1 {
    ControlPlaneOccurrenceStoreBindingV1::new(
        "deployment/site01-a", "lunar/site01", "control-plane/store-a", 7, c(71),
    ).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelError { ResolveUnavailable }
impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    ConcurrentSameOperationOnCas,
}

struct Store {
    frontier: Option<RawControlPlaneOccurrenceRecordV1>,
    operations: BTreeMap<Commitment32, RawControlPlaneOccurrenceRecordV1>,
    mode: Mode,
    cas_calls: usize,
}

impl Store {
    fn new(mode: Mode) -> Self {
        Self { frontier: None, operations: BTreeMap::new(), mode, cas_calls: 0 }
    }
}

impl ControlPlaneOccurrenceStoreV1 for Store {
    type Error = ModelError;

    fn load_frontier(
        &self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
    ) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> {
        Ok(self.frontier.clone())
    }

    fn compare_and_swap(
        &mut self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        expected: Option<ControlPlaneOccurrenceHeadV1>,
        proposed: &ProposedControlPlaneOccurrenceV1,
    ) -> Result<ControlPlaneStoreCasResultV1, Self::Error> {
        self.cas_calls += 1;
        let current = self.frontier.as_ref().map(RawControlPlaneOccurrenceRecordV1::head);
        if current != expected {
            return Ok(ControlPlaneStoreCasResultV1::Conflict { actual_frontier: current });
        }

        if self.mode == Mode::ConcurrentSameOperationOnCas {
            let record = RawControlPlaneOccurrenceRecordV1::new_unqualified(
                proposed.clone(),
                format!("concurrent-store-ref-{}", proposed.sequence()),
            ).unwrap();
            self.operations.insert(proposed.operation_id().commitment(), record.clone());
            self.frontier = Some(record);
            return Ok(ControlPlaneStoreCasResultV1::Conflict {
                actual_frontier: self.frontier.as_ref().map(RawControlPlaneOccurrenceRecordV1::head),
            });
        }

        let reference = format!("store-ref-{}", proposed.sequence());
        let record = RawControlPlaneOccurrenceRecordV1::new_unqualified(
            proposed.clone(),
            &reference,
        ).unwrap();
        self.operations.insert(proposed.operation_id().commitment(), record.clone());
        self.frontier = Some(record);
        Ok(ControlPlaneStoreCasResultV1::Applied { store_reference: reference })
    }

    fn resolve_operation(
        &self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        operation_id: ControlPlaneCommitOperationIdV1,
    ) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> {
        if let Some(record) = self.operations.get(&operation_id.commitment()) {
            return Ok(ControlPlaneStoreOperationResolutionV1::Found(record.clone()));
        }
        Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent {
            current_frontier: self.frontier.as_ref().map(RawControlPlaneOccurrenceRecordV1::head),
        })
    }
}

#[test]
fn earlier_committed_operation_is_found_after_frontier_advances() {
    let p = profile();
    let b1 = binding(&p, "research/store-a", 7);
    let c1 = genesis(&p, &b1);
    let store_id = store_binding();
    let mut store = Store::new(Mode::Normal);

    let plan1 = ControlPlaneCommitPlanV1::prepare(store_id.clone(), &c1, None).unwrap();
    let o1 = plan1.occurrence().clone();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan1).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)
    ));

    let (b2, c2) = migrate(&p, &b1, &c1, "research/store-a", 8, 51);
    let plan2 = ControlPlaneCommitPlanV1::prepare(store_id.clone(), &c2, Some(&o1)).unwrap();
    let o2 = plan2.occurrence().clone();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan2).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)
    ));

    let (_b3, c3) = migrate(&p, &b2, &c2, "research/store-a", 9, 53);
    let plan3 = ControlPlaneCommitPlanV1::prepare(store_id.clone(), &c3, Some(&o2)).unwrap();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan3).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)
    ));

    let cas_calls_before_retry = store.cas_calls;
    let retry_plan2 = ControlPlaneCommitPlanV1::prepare(store_id, &c2, Some(&o1)).unwrap();
    match attempt_observed_control_plane_commit(&mut store, retry_plan2).unwrap() {
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(observed) => {
            assert_eq!(observed.occurrence(), &o2);
        }
        other => panic!("expected earlier exact commit, got {other:?}"),
    }
    assert_eq!(store.cas_calls, cas_calls_before_retry, "retry must resolve history without a second CAS");
}

#[test]
fn concurrent_same_operation_before_conflict_resolves_as_committed() {
    let p = profile();
    let b1 = binding(&p, "research/store-a", 7);
    let c1 = genesis(&p, &b1);
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(), &c1, None).unwrap();
    let expected = plan.occurrence().clone();
    let mut store = Store::new(Mode::ConcurrentSameOperationOnCas);

    match attempt_observed_control_plane_commit(&mut store, plan).unwrap() {
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(observed) => {
            assert_eq!(observed.occurrence(), &expected);
        }
        other => panic!("expected same operation to resolve committed, got {other:?}"),
    }
    assert_eq!(store.cas_calls, 1);
}

struct ConflictResolveErrorStore {
    resolve_count: std::cell::Cell<usize>,
}

impl ControlPlaneOccurrenceStoreV1 for ConflictResolveErrorStore {
    type Error = ModelError;

    fn load_frontier(&self, _: &ControlPlaneOccurrenceStoreBindingV1) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> {
        Ok(None)
    }

    fn compare_and_swap(
        &mut self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        _: Option<ControlPlaneOccurrenceHeadV1>,
        _: &ProposedControlPlaneOccurrenceV1,
    ) -> Result<ControlPlaneStoreCasResultV1, Self::Error> {
        Ok(ControlPlaneStoreCasResultV1::Conflict { actual_frontier: None })
    }

    fn resolve_operation(
        &self,
        _: &ControlPlaneOccurrenceStoreBindingV1,
        _: ControlPlaneCommitOperationIdV1,
    ) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> {
        let next = self.resolve_count.get() + 1;
        self.resolve_count.set(next);
        if next == 1 {
            Ok(ControlPlaneStoreOperationResolutionV1::ProvenAbsent { current_frontier: None })
        } else {
            Err(ModelError::ResolveUnavailable)
        }
    }
}

#[test]
fn conflict_without_operation_resolution_stays_outcome_unknown() {
    let p = profile();
    let b1 = binding(&p, "research/store-a", 7);
    let c1 = genesis(&p, &b1);
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(), &c1, None).unwrap();
    let mut store = ConflictResolveErrorStore { resolve_count: std::cell::Cell::new(0) };

    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::OutcomeUnknown(_)
    ));
}

#[test]
fn preflight_operation_resolution_failure_is_before_cas() {
    struct FailingStore;
    impl ControlPlaneOccurrenceStoreV1 for FailingStore {
        type Error = ModelError;
        fn load_frontier(&self, _: &ControlPlaneOccurrenceStoreBindingV1) -> Result<Option<RawControlPlaneOccurrenceRecordV1>, Self::Error> { Ok(None) }
        fn compare_and_swap(&mut self, _: &ControlPlaneOccurrenceStoreBindingV1, _: Option<ControlPlaneOccurrenceHeadV1>, _: &ProposedControlPlaneOccurrenceV1) -> Result<ControlPlaneStoreCasResultV1, Self::Error> { panic!("CAS must not run") }
        fn resolve_operation(&self, _: &ControlPlaneOccurrenceStoreBindingV1, _: ControlPlaneCommitOperationIdV1) -> Result<ControlPlaneStoreOperationResolutionV1, Self::Error> { Err(ModelError::ResolveUnavailable) }
    }

    let p = profile();
    let b1 = binding(&p, "research/store-a", 7);
    let c1 = genesis(&p, &b1);
    let plan = ControlPlaneCommitPlanV1::prepare(store_binding(), &c1, None).unwrap();
    let mut store = FailingStore;
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan),
        Err(ControlPlaneCommitProtocolError::PreflightResolve(ModelError::ResolveUnavailable))
    ));
}
