use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::Connection;
use symthaea_scientific_view_control_plane::{
    CandidateScientificViewControlPlaneTransitionV1, CandidateSourceMigrationV1,
    DeploymentBootstrapEvidenceRefV1, PredecessorAuthorizationEvidenceRefV1,
};
use symthaea_scientific_view_control_plane_occurrence::{
    ControlPlaneCommitPlanV1, ControlPlaneOccurrenceStoreBindingV1, ControlPlaneOccurrenceStoreV1,
    ControlPlaneStoreCasResultV1, ControlPlaneStoreOperationResolutionV1,
    ObservedControlPlaneCommitOutcomeV1, attempt_observed_control_plane_commit,
};
use symthaea_scientific_view_control_plane_occurrence_sqlite::{
    SqliteControlPlaneOccurrenceStoreError, SqliteControlPlaneOccurrenceStoreV1,
};
use symthaea_scientific_view_profile::{
    AuthoritySourceBindingV1, AuthoritySourceOccurrenceV1, Commitment32, RoleSemanticRevisionV1,
    ScientificAuthorityRoleV1 as Role, ScientificViewDeploymentBindingV1,
    ScientificViewSemanticProfileV1,
};

static NEXT_DB: AtomicU64 = AtomicU64::new(0);

fn c(byte: u8) -> Commitment32 {
    Commitment32::from_bytes([byte; 32])
}

fn db_path(label: &str) -> std::path::PathBuf {
    let id = NEXT_DB.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "symthaea-sci014-{label}-{}-{id}.sqlite",
        std::process::id()
    ))
}

fn cleanup(path: &std::path::Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn profile(policy: u8) -> ScientificViewSemanticProfileV1 {
    ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, c(11)).unwrap(),
            RoleSemanticRevisionV1::new(Role::VerifierPolicyHead, c(12)).unwrap(),
        ],
        c(201),
        c(202),
        c(203),
        c(policy),
    )
    .unwrap()
}

fn deployment_binding(
    profile: &ScientificViewSemanticProfileV1,
    research_id: &str,
    epoch: u64,
) -> ScientificViewDeploymentBindingV1 {
    ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        profile,
        vec![
            AuthoritySourceBindingV1::new(Role::ResearchSemanticHead, research_id, epoch, c(21))
                .unwrap(),
            AuthoritySourceBindingV1::new(
                Role::VerifierPolicyHead,
                "verifier-policy/store-a",
                3,
                c(22),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn genesis() -> (
    ScientificViewSemanticProfileV1,
    ScientificViewDeploymentBindingV1,
    CandidateScientificViewControlPlaneTransitionV1,
) {
    let profile = profile(204);
    let binding = deployment_binding(&profile, "research/store-a", 7);
    let bootstrap =
        DeploymentBootstrapEvidenceRefV1::new(&profile, &binding, c(41), c(42)).unwrap();
    let candidate = CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(
        &profile, &binding, &bootstrap,
    )
    .unwrap();
    (profile, binding, candidate)
}

fn migration(
    profile: &ScientificViewSemanticProfileV1,
    old_binding: &ScientificViewDeploymentBindingV1,
    predecessor: &CandidateScientificViewControlPlaneTransitionV1,
    research_id: &str,
    epoch: u64,
    auth: u8,
) -> CandidateScientificViewControlPlaneTransitionV1 {
    let new_binding = deployment_binding(profile, research_id, epoch);
    let old_occurrence =
        AuthoritySourceOccurrenceV1::genesis(old_binding, Role::ResearchSemanticHead, c(31))
            .unwrap();
    let new_occurrence =
        AuthoritySourceOccurrenceV1::genesis(&new_binding, Role::ResearchSemanticHead, c(31))
            .unwrap();
    let migration = CandidateSourceMigrationV1::new(
        profile,
        old_binding,
        &new_binding,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_occurrence,
        c(31),
    )
    .unwrap();
    let authorization = PredecessorAuthorizationEvidenceRefV1::for_source_migration(
        predecessor,
        profile,
        &new_binding,
        &migration,
        c(auth),
        c(auth + 1),
    )
    .unwrap();
    CandidateScientificViewControlPlaneTransitionV1::source_migration_candidate(
        predecessor,
        profile,
        old_binding,
        &new_binding,
        &migration,
        &authorization,
    )
    .unwrap()
}

fn occurrence_binding(epoch: u64) -> ControlPlaneOccurrenceStoreBindingV1 {
    ControlPlaneOccurrenceStoreBindingV1::new(
        "deployment/site01-a",
        "lunar/site01",
        "control-plane/store-a",
        epoch,
        c(71),
    )
    .unwrap()
}

#[test]
fn genesis_successor_and_old_operation_survive_reopen() {
    let path = db_path("reopen");
    cleanup(&path);

    let (profile, deployment, c1) = genesis();
    let binding = occurrence_binding(7);
    let mut store =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(81)).unwrap();

    let plan1 = ControlPlaneCommitPlanV1::prepare(binding.clone(), &c1, None).unwrap();
    let op1 = plan1.operation_id();
    let occurrence1 = plan1.occurrence().clone();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan1).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)
    ));

    let c2 = migration(&profile, &deployment, &c1, "research/store-a", 8, 51);
    let plan2 =
        ControlPlaneCommitPlanV1::prepare(binding.clone(), &c2, Some(&occurrence1)).unwrap();
    let occurrence2 = plan2.occurrence().clone();
    assert!(matches!(
        attempt_observed_control_plane_commit(&mut store, plan2).unwrap(),
        ObservedControlPlaneCommitOutcomeV1::CommittedObserved(_)
    ));
    drop(store);

    let reopened =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(81)).unwrap();
    assert_eq!(
        reopened
            .load_frontier(&binding)
            .unwrap()
            .unwrap()
            .occurrence(),
        &occurrence2
    );
    assert!(matches!(
        reopened.resolve_operation(&binding, op1).unwrap(),
        ControlPlaneStoreOperationResolutionV1::Found(ref record)
            if record.occurrence() == &occurrence1
    ));

    cleanup(&path);
}

#[test]
fn two_connections_cannot_both_advance_one_frontier() {
    let path = db_path("concurrency");
    cleanup(&path);

    let (profile, deployment, c1) = genesis();
    let binding = occurrence_binding(7);
    let mut first =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(82)).unwrap();
    let plan1 = ControlPlaneCommitPlanV1::prepare(binding.clone(), &c1, None).unwrap();
    let occurrence1 = plan1.occurrence().clone();
    assert!(matches!(
        first
            .compare_and_swap(&binding, None, plan1.occurrence())
            .unwrap(),
        ControlPlaneStoreCasResultV1::Applied { .. }
    ));

    let mut second =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(82)).unwrap();
    let c2a = migration(&profile, &deployment, &c1, "research/store-a", 8, 61);
    let c2b = migration(&profile, &deployment, &c1, "research/store-b", 1, 71);
    let plan2a =
        ControlPlaneCommitPlanV1::prepare(binding.clone(), &c2a, Some(&occurrence1)).unwrap();
    let plan2b =
        ControlPlaneCommitPlanV1::prepare(binding.clone(), &c2b, Some(&occurrence1)).unwrap();

    assert!(matches!(
        first
            .compare_and_swap(&binding, Some(occurrence1.head()), plan2a.occurrence())
            .unwrap(),
        ControlPlaneStoreCasResultV1::Applied { .. }
    ));
    assert!(matches!(
        second
            .compare_and_swap(&binding, Some(occurrence1.head()), plan2b.occurrence())
            .unwrap(),
        ControlPlaneStoreCasResultV1::Conflict { .. }
    ));

    cleanup(&path);
}

#[test]
fn operation_absence_and_frontier_come_from_one_validated_snapshot() {
    let path = db_path("absence");
    cleanup(&path);

    let (profile, deployment, c1) = genesis();
    let binding = occurrence_binding(7);
    let mut store =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(83)).unwrap();
    let plan1 = ControlPlaneCommitPlanV1::prepare(binding.clone(), &c1, None).unwrap();
    let occurrence1 = plan1.occurrence().clone();
    store
        .compare_and_swap(&binding, None, plan1.occurrence())
        .unwrap();

    let uncommitted = migration(&profile, &deployment, &c1, "research/store-a", 8, 81);
    let plan2 =
        ControlPlaneCommitPlanV1::prepare(binding.clone(), &uncommitted, Some(&occurrence1))
            .unwrap();

    assert!(matches!(
        store.resolve_operation(&binding, plan2.operation_id()).unwrap(),
        ControlPlaneStoreOperationResolutionV1::ProvenAbsent {
            current_frontier: Some(frontier)
        } if frontier == occurrence1.head()
    ));

    cleanup(&path);
}

#[test]
fn copied_database_cannot_be_reopened_under_another_store_epoch() {
    let path = db_path("epoch");
    cleanup(&path);

    let store =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, occurrence_binding(7), c(84)).unwrap();
    drop(store);

    assert!(matches!(
        SqliteControlPlaneOccurrenceStoreV1::open(&path, occurrence_binding(8), c(84)),
        Err(SqliteControlPlaneOccurrenceStoreError::StoreMetadataMismatch)
    ));

    cleanup(&path);
}

#[test]
fn canonical_occurrence_corruption_fails_closed() {
    let path = db_path("corrupt");
    cleanup(&path);

    let (_, _, c1) = genesis();
    let binding = occurrence_binding(7);
    let mut store =
        SqliteControlPlaneOccurrenceStoreV1::open(&path, binding.clone(), c(85)).unwrap();
    let plan = ControlPlaneCommitPlanV1::prepare(binding.clone(), &c1, None).unwrap();
    store
        .compare_and_swap(&binding, None, plan.occurrence())
        .unwrap();

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE control_plane_occurrences
             SET occurrence_bytes = zeroblob(length(occurrence_bytes))
             WHERE sequence = 1",
            [],
        )
        .unwrap();
    drop(connection);

    assert!(matches!(
        store.load_frontier(&binding),
        Err(SqliteControlPlaneOccurrenceStoreError::Wire(_))
    ));

    cleanup(&path);
}
