use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::Connection;
use symthaea_scientific_view_control_plane_occurrence::{
    ControlPlaneOccurrenceStoreBindingV1, ControlPlaneOccurrenceStoreV1,
};
use symthaea_scientific_view_control_plane_occurrence_sqlite::{
    SqliteControlPlaneOccurrenceStoreError, SqliteControlPlaneOccurrenceStoreV1,
};
use symthaea_scientific_view_profile::Commitment32;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn c(byte: u8) -> Commitment32 {
    Commitment32::from_bytes([byte; 32])
}

fn binding() -> ControlPlaneOccurrenceStoreBindingV1 {
    ControlPlaneOccurrenceStoreBindingV1::new(
        "deployment/site01-a",
        "lunar/site01",
        "control-plane/store-a",
        7,
        c(71),
    )
    .unwrap()
}

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "symthaea-sci014-r2-schema-{label}-{}-{}.sqlite",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

struct Cleanup(PathBuf);

impl Cleanup {
    fn new(path: PathBuf) -> Self {
        Self(path)
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = fs::remove_file(format!("{}{}", self.0.display(), suffix));
        }
    }
}

fn open_store(
    path: &Path,
) -> Result<SqliteControlPlaneOccurrenceStoreV1, SqliteControlPlaneOccurrenceStoreError> {
    SqliteControlPlaneOccurrenceStoreV1::open(path, binding(), c(72))
}

fn journal_mode(path: &Path) -> String {
    let connection = Connection::open(path).unwrap();
    connection
        .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
        .unwrap()
        .to_ascii_lowercase()
}

fn require_schema_mismatch<T>(result: Result<T, SqliteControlPlaneOccurrenceStoreError>) {
    match result {
        Err(SqliteControlPlaneOccurrenceStoreError::SchemaContractMismatch) => {}
        Err(other) => panic!("expected SchemaContractMismatch, got {other}"),
        Ok(_) => panic!("schema drift was accepted"),
    }
}

#[test]
fn clean_exact_v1_schema_reopens_and_operates() {
    let path = temp_path("clean");
    let _cleanup = Cleanup::new(path.clone());

    drop(open_store(&path).unwrap());
    assert_eq!(journal_mode(&path), "wal");
    let store = open_store(&path).unwrap();
    assert!(store.load_frontier(&binding()).unwrap().is_none());
}

#[test]
fn extra_trigger_is_rejected_on_reopen() {
    let path = temp_path("trigger");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER rogue_occurrence_trigger
             AFTER INSERT ON control_plane_occurrences
             BEGIN
                 SELECT 1;
             END;",
        )
        .unwrap();
    drop(connection);

    require_schema_mismatch(open_store(&path));
}

#[test]
fn extra_view_is_rejected_on_reopen() {
    let path = temp_path("view");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE VIEW rogue_occurrence_view AS
             SELECT sequence, occurrence_commitment FROM control_plane_occurrences;",
        )
        .unwrap();
    drop(connection);

    require_schema_mismatch(open_store(&path));
}

#[test]
fn explicit_index_is_rejected_on_reopen() {
    let path = temp_path("index");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE INDEX rogue_candidate_index
             ON control_plane_occurrences(candidate_transition_commitment);",
        )
        .unwrap();
    drop(connection);

    require_schema_mismatch(open_store(&path));
}

#[test]
fn sqlite_lookalike_name_is_not_mistaken_for_internal_object() {
    let path = temp_path("sqlite-lookalike");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE VIEW sqliteX_rogue_view AS
             SELECT singleton FROM control_plane_store_metadata;",
        )
        .unwrap();
    drop(connection);

    require_schema_mismatch(open_store(&path));
}

#[test]
fn missing_required_table_is_rejected_without_auto_repair() {
    let path = temp_path("missing-table");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch("DROP TABLE control_plane_frontier;")
        .unwrap();
    drop(connection);

    require_schema_mismatch(open_store(&path));

    let connection = Connection::open(&path).unwrap();
    let frontier_table_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema
             WHERE type = 'table' AND name = 'control_plane_frontier'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(frontier_table_count, 0, "failed open must not repair schema");
}

#[test]
fn foreign_partial_schema_is_rejected_without_wal_conversion() {
    let path = temp_path("foreign-partial");
    let _cleanup = Cleanup::new(path.clone());

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch("CREATE TABLE foreign_state(id INTEGER PRIMARY KEY, payload TEXT);")
        .unwrap();
    drop(connection);
    assert_eq!(journal_mode(&path), "delete");

    require_schema_mismatch(open_store(&path));
    assert_eq!(journal_mode(&path), "delete");

    let connection = Connection::open(&path).unwrap();
    let foreign_table_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name='foreign_state'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(foreign_table_count, 1);
}

#[test]
fn exact_store_downgraded_from_wal_is_rejected_without_repair() {
    let path = temp_path("downgraded-journal");
    let _cleanup = Cleanup::new(path.clone());
    drop(open_store(&path).unwrap());
    assert_eq!(journal_mode(&path), "wal");

    let connection = Connection::open(&path).unwrap();
    let mode: String = connection
        .query_row("PRAGMA journal_mode=DELETE", [], |row| row.get(0))
        .unwrap();
    assert_eq!(mode.to_ascii_lowercase(), "delete");
    drop(connection);

    assert!(matches!(
        open_store(&path),
        Err(SqliteControlPlaneOccurrenceStoreError::DurabilityProfileMismatch)
    ));
    assert_eq!(journal_mode(&path), "delete");
}

#[test]
fn post_open_ddl_drift_is_rejected_at_point_of_use() {
    let path = temp_path("live-drift");
    let _cleanup = Cleanup::new(path.clone());
    let store = open_store(&path).unwrap();

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE VIEW post_open_rogue_view AS
             SELECT singleton FROM control_plane_store_metadata;",
        )
        .unwrap();
    drop(connection);

    require_schema_mismatch(store.load_frontier(&binding()));
}
