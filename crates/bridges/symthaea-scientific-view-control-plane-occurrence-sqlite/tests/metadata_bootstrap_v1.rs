use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::Connection;
use symthaea_scientific_view_control_plane::{
    CandidateScientificViewControlPlaneTransitionV1, DeploymentBootstrapEvidenceRefV1,
};
use symthaea_scientific_view_control_plane_occurrence::{
    ControlPlaneCommitPlanV1, ControlPlaneOccurrenceStoreBindingV1,
    ControlPlaneOccurrenceStoreV1,
};
use symthaea_scientific_view_control_plane_occurrence_sqlite::{
    SqliteControlPlaneOccurrenceStoreError, SqliteControlPlaneOccurrenceStoreV1,
};
use symthaea_scientific_view_profile::{
    AuthoritySourceBindingV1, Commitment32, RoleSemanticRevisionV1,
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
        "symthaea-sci014-metadata-{label}-{}-{id}.sqlite",
        std::process::id()
    ))
}

fn cleanup(path: &std::path::Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn occurrence_binding() -> ControlPlaneOccurrenceStoreBindingV1 {
    ControlPlaneOccurrenceStoreBindingV1::new(
        "deployment/site01-a",
        "lunar/site01",
        "control-plane/store-a",
        7,
        c(71),
    )
    .unwrap()
}

fn genesis_candidate() -> CandidateScientificViewControlPlaneTransitionV1 {
    let profile = ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, c(11)).unwrap(),
            RoleSemanticRevisionV1::new(Role::VerifierPolicyHead, c(12)).unwrap(),
        ],
        c(201),
        c(202),
        c(203),
        c(204),
    )
    .unwrap();
    let binding = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![
            AuthoritySourceBindingV1::new(
                Role::ResearchSemanticHead,
                "research/store-a",
                7,
                c(21),
            )
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
    .unwrap();
    let bootstrap = DeploymentBootstrapEvidenceRefV1::new(&profile, &binding, c(41), c(42))
        .unwrap();
    CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(
        &profile,
        &binding,
        &bootstrap,
    )
    .unwrap()
}

fn seeded_store(path: &std::path::Path) {
    let binding = occurrence_binding();
    let mut store = SqliteControlPlaneOccurrenceStoreV1::open(path, binding.clone(), c(81))
        .unwrap();
    let candidate = genesis_candidate();
    let plan = ControlPlaneCommitPlanV1::prepare(binding.clone(), &candidate, None).unwrap();
    store
        .compare_and_swap(&binding, None, plan.occurrence())
        .unwrap();
}

fn reopen_rejects_unprovisioned_state(
    path: &std::path::Path,
    expected_occurrences: i64,
    expected_frontier: i64,
) {
    assert!(matches!(
        SqliteControlPlaneOccurrenceStoreV1::open(path, occurrence_binding(), c(81)),
        Err(SqliteControlPlaneOccurrenceStoreError::UnprovisionedStoreContainsState {
            occurrence_count,
            frontier_count,
        }) if occurrence_count == expected_occurrences && frontier_count == expected_frontier
    ));

    let connection = Connection::open(path).unwrap();
    let metadata_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM control_plane_store_metadata",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(metadata_count, 0, "failed reopen must not adopt the store");
}

#[test]
fn missing_metadata_over_full_existing_state_is_rejected_without_adoption() {
    let path = db_path("full-state");
    cleanup(&path);
    seeded_store(&path);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute("DELETE FROM control_plane_store_metadata", [])
        .unwrap();
    drop(connection);

    reopen_rejects_unprovisioned_state(&path, 1, 1);
    cleanup(&path);
}

#[test]
fn missing_metadata_over_occurrence_only_state_is_rejected() {
    let path = db_path("occurrence-only");
    cleanup(&path);
    seeded_store(&path);

    let connection = Connection::open(&path).unwrap();
    connection.execute("PRAGMA foreign_keys=OFF", []).unwrap();
    connection
        .execute("DELETE FROM control_plane_store_metadata", [])
        .unwrap();
    connection
        .execute("DELETE FROM control_plane_frontier", [])
        .unwrap();
    drop(connection);

    reopen_rejects_unprovisioned_state(&path, 1, 0);
    cleanup(&path);
}

#[test]
fn missing_metadata_over_frontier_only_state_is_rejected() {
    let path = db_path("frontier-only");
    cleanup(&path);
    seeded_store(&path);

    let connection = Connection::open(&path).unwrap();
    connection.execute("PRAGMA foreign_keys=OFF", []).unwrap();
    connection
        .execute("DELETE FROM control_plane_store_metadata", [])
        .unwrap();
    connection
        .execute("DELETE FROM control_plane_occurrences", [])
        .unwrap();
    drop(connection);

    reopen_rejects_unprovisioned_state(&path, 0, 1);
    cleanup(&path);
}

#[test]
fn atomically_new_path_is_the_only_first_bootstrap_case() {
    let path = db_path("new-path");
    cleanup(&path);

    let store = SqliteControlPlaneOccurrenceStoreV1::open(&path, occurrence_binding(), c(81))
        .unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    let journal_mode: String = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    let metadata_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM control_plane_store_metadata",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let occurrence_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM control_plane_occurrences", [], |row| row.get(0))
        .unwrap();
    let frontier_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM control_plane_frontier", [], |row| row.get(0))
        .unwrap();
    assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
    assert_eq!((metadata_count, occurrence_count, frontier_count), (1, 0, 0));

    cleanup(&path);
}

#[test]
fn preexisting_empty_file_is_not_adopted_or_converted_to_wal() {
    let path = db_path("preexisting-empty");
    cleanup(&path);
    fs::File::create(&path).unwrap();

    assert!(matches!(
        SqliteControlPlaneOccurrenceStoreV1::open(&path, occurrence_binding(), c(81)),
        Err(SqliteControlPlaneOccurrenceStoreError::ExistingStoreUnprovisioned)
    ));

    let connection = Connection::open(&path).unwrap();
    let journal_mode: String = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    let application_object_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(journal_mode.to_ascii_lowercase(), "delete");
    assert_eq!(application_object_count, 0);

    cleanup(&path);
}
