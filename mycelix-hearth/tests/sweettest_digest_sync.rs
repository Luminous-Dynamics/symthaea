//! # Mycelix Hearth — Digest Sync Sweettest Integration Tests
//!
//! Tests the hearth_sync (weekly digest assembly) flow: the bridge zome
//! aggregates data from kinship (bond snapshots), gratitude, care, and
//! rhythm zomes into a WeeklyDigest entry.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_digest_sync -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (hearth creation)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HearthType {
    Nuclear,
    Extended,
    Chosen,
    Blended,
    Multigenerational,
    Intentional,
    CoPod,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateHearthInput {
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub max_members: Option<u32>,
}

// ============================================================================
// Mirror types — bridge sync
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HearthSyncInput {
    pub hearth_hash: ActionHash,
    pub epoch_start: Timestamp,
    pub epoch_end: Timestamp,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn hearth_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-hearth/
    path.push("dna");
    path.push("mycelix_hearth.dna");
    path
}

// ============================================================================
// Digest Sync Tests
// ============================================================================

/// Alice creates a hearth, triggers hearth_sync on the bridge with a 7-day
/// epoch window. Then get_weekly_digests returns at least 1 record.
///
/// The hearth_sync function dispatches cross-zome calls to gratitude, care,
/// and rhythm zomes for epoch summaries, queries kinship for bond snapshots,
/// and assembles a WeeklyDigest entry stored via kinship.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_hearth_sync_creates_digest() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Digest Sync Hearth".to_string(),
                description: "Testing weekly digest assembly".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Trigger hearth_sync with a 7-day epoch window
    let now = Timestamp::now();
    let seven_days_micros: i64 = 7 * 24 * 60 * 60 * 1_000_000;
    let epoch_start = Timestamp::from_micros(now.as_micros() - seven_days_micros);

    let sync_input = HearthSyncInput {
        hearth_hash: hearth_hash.clone(),
        epoch_start,
        epoch_end: now,
    };

    let digest_record: Record = conductor
        .call(&alice.zome("hearth_bridge"), "hearth_sync", sync_input)
        .await;

    assert!(
        digest_record.action().author() == alice.agent_pubkey(),
        "Digest should be authored by Alice"
    );

    // get_weekly_digests should return at least 1 record
    let digests: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_bridge"),
            "get_weekly_digests",
            hearth_hash,
        )
        .await;

    assert!(
        !digests.is_empty(),
        "get_weekly_digests should return at least 1 digest after hearth_sync"
    );
}
