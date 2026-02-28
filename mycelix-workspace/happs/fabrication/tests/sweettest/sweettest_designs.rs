//! # Fabrication — Designs Sweettest
//!
//! Integration tests for design CRUD, file management, forking, and
//! authorization enforcement.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_designs -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — designs coordinator
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateDesignInput {
    pub title: String,
    pub description: String,
    pub category: String,
    pub license: String,
    pub safety_class: String,
    pub files: Vec<String>,
    pub tags: Vec<String>,
    pub material_compatibility: Vec<String>,
    pub parametric_schema: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateDesignInput {
    pub original_action_hash: ActionHash,
    pub title: Option<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddFileInput {
    pub design_hash: ActionHash,
    pub file: DesignFile,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DesignFile {
    pub filename: String,
    pub file_type: String,
    pub ipfs_cid: String,
    pub size_bytes: u64,
    pub checksum_sha256: String,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn fabrication_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // sweettest/ -> tests/
    path.pop(); // tests/ -> fabrication/
    path.push("workdir");
    path.push("fabrication.dna");
    path
}

// ============================================================================
// Design CRUD Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_design_create_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = CreateDesignInput {
        title: "Test Bracket".to_string(),
        description: "A structural bracket for testing".to_string(),
        category: "Parts".to_string(),
        license: "PublicDomain".to_string(),
        safety_class: "Class1Functional".to_string(),
        files: vec![],
        tags: vec!["test".to_string()],
        material_compatibility: vec![],
        parametric_schema: None,
    };

    let record: Record = conductor
        .call(&alice.zome("designs_coordinator"), "create_design", input)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Retrieve it
    let design_hash = record.action_address().clone();
    let retrieved: Option<Record> = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "get_design",
            design_hash,
        )
        .await;

    assert!(retrieved.is_some());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_design_update_requires_author() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let ((alice,), (bob,)) = conductor
        .setup_app_for_zipped_agents(
            "test-app",
            &[dna_file.clone()],
            &[2],
        )
        .await
        .unwrap()
        .into_tuples();

    // Alice creates a design
    let input = CreateDesignInput {
        title: "Alice's Design".to_string(),
        description: "Only Alice should update this".to_string(),
        category: "Parts".to_string(),
        license: "PublicDomain".to_string(),
        safety_class: "Class0Decorative".to_string(),
        files: vec![],
        tags: vec![],
        material_compatibility: vec![],
        parametric_schema: None,
    };

    let record: Record = conductor
        .call(&alice.zome("designs_coordinator"), "create_design", input)
        .await;

    let design_hash = record.action_address().clone();

    // Bob tries to update — should fail
    let update_input = UpdateDesignInput {
        original_action_hash: design_hash.clone(),
        title: Some("Bob's Hostile Update".to_string()),
        description: None,
    };

    let bob_result: Result<Record, _> = conductor
        .call_fallible(
            &bob.zome("designs_coordinator"),
            "update_design",
            update_input,
        )
        .await;

    assert!(
        bob_result.is_err(),
        "Bob should not be able to update Alice's design"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_design_delete_requires_author() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let ((alice,), (bob,)) = conductor
        .setup_app_for_zipped_agents(
            "test-app",
            &[dna_file.clone()],
            &[2],
        )
        .await
        .unwrap()
        .into_tuples();

    // Alice creates a design
    let input = CreateDesignInput {
        title: "Delete Auth Test".to_string(),
        description: "Only author can delete".to_string(),
        category: "Parts".to_string(),
        license: "PublicDomain".to_string(),
        safety_class: "Class0Decorative".to_string(),
        files: vec![],
        tags: vec![],
        material_compatibility: vec![],
        parametric_schema: None,
    };

    let record: Record = conductor
        .call(&alice.zome("designs_coordinator"), "create_design", input)
        .await;

    let design_hash = record.action_address().clone();

    // Bob tries to delete — should fail
    let bob_result: Result<ActionHash, _> = conductor
        .call_fallible(
            &bob.zome("designs_coordinator"),
            "delete_design",
            design_hash,
        )
        .await;

    assert!(
        bob_result.is_err(),
        "Bob should not be able to delete Alice's design"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_add_file_requires_design_author() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let ((alice,), (bob,)) = conductor
        .setup_app_for_zipped_agents(
            "test-app",
            &[dna_file.clone()],
            &[2],
        )
        .await
        .unwrap()
        .into_tuples();

    // Alice creates a design
    let input = CreateDesignInput {
        title: "File Auth Test".to_string(),
        description: "Only author can add files".to_string(),
        category: "Parts".to_string(),
        license: "PublicDomain".to_string(),
        safety_class: "Class0Decorative".to_string(),
        files: vec![],
        tags: vec![],
        material_compatibility: vec![],
        parametric_schema: None,
    };

    let record: Record = conductor
        .call(&alice.zome("designs_coordinator"), "create_design", input)
        .await;

    let design_hash = record.action_address().clone();

    // Bob tries to add a file — should fail
    let file_input = AddFileInput {
        design_hash,
        file: DesignFile {
            filename: "malicious.stl".to_string(),
            file_type: "STL".to_string(),
            ipfs_cid: "QmFakeHash".to_string(),
            size_bytes: 1024,
            checksum_sha256: "abc123".to_string(),
        },
    };

    let bob_result: Result<Record, _> = conductor
        .call_fallible(
            &bob.zome("designs_coordinator"),
            "add_design_file",
            file_input,
        )
        .await;

    assert!(
        bob_result.is_err(),
        "Bob should not be able to add files to Alice's design"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
