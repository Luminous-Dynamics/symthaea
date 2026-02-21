//! # Mycelix Hearth — Bridge Sweettest Integration Tests
//!
//! Tests the hearth bridge zome's dispatch and health_check functionality.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_bridge -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (needed for dispatch payload)
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
// Mirror types — bridge dispatch
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchInput {
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
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
// Bridge Tests
// ============================================================================

/// Call health_check on hearth_bridge, returns BridgeHealth with healthy=true
/// and 10 domains (kinship, gratitude, stories, care, autonomy, emergency,
/// decisions, resources, milestones, rhythms).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_health_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("hearth_bridge"), "health_check", ())
        .await;

    assert!(health.healthy, "Bridge should be healthy");
    assert_eq!(
        health.domains.len(),
        10,
        "Hearth bridge should have exactly 10 domains"
    );
    assert!(health.domains.contains(&"kinship".to_string()));
    assert!(health.domains.contains(&"gratitude".to_string()));
    assert!(health.domains.contains(&"stories".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"autonomy".to_string()));
    assert!(health.domains.contains(&"emergency".to_string()));
    assert!(health.domains.contains(&"decisions".to_string()));
    assert!(health.domains.contains(&"resources".to_string()));
    assert!(health.domains.contains(&"milestones".to_string()));
    assert!(health.domains.contains(&"rhythms".to_string()));
}

/// Serialize a CreateHearthInput using rmp_serde::to_vec, dispatch to
/// hearth_kinship/create_hearth via the bridge, verify success=true.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_dispatch_to_kinship() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Serialize a CreateHearthInput as msgpack bytes
    let hearth_input = CreateHearthInput {
        name: "Dispatched Hearth".to_string(),
        description: "Created via bridge dispatch".to_string(),
        hearth_type: HearthType::Intentional,
        max_members: Some(5),
    };

    let payload = rmp_serde::to_vec(&hearth_input)
        .expect("Failed to msgpack-serialize CreateHearthInput");

    let dispatch = DispatchInput {
        zome: "hearth_kinship".to_string(),
        fn_name: "create_hearth".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("hearth_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(
        result.success,
        "Dispatch to hearth_kinship/create_hearth should succeed. Error: {:?}",
        result.error
    );
    assert!(
        result.response.is_some(),
        "Successful dispatch should include a response payload"
    );
}
