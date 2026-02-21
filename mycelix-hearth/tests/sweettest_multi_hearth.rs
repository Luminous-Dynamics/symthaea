//! # Mycelix Hearth — Multi-Hearth Isolation Sweettest Integration Tests
//!
//! Tests that an agent in multiple hearths has proper data isolation:
//! decisions created in one hearth do not appear in another.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_multi_hearth -- --ignored --test-threads=2
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
pub enum MemberRole {
    Founder,
    Elder,
    Adult,
    Youth,
    Child,
    Guest,
    Ancestor,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateHearthInput {
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub max_members: Option<u32>,
}

// ============================================================================
// Mirror types — decisions
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DecisionType {
    Consensus,
    MajorityVote,
    ElderDecision,
    GuardianDecision,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateDecisionInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub eligible_roles: Vec<MemberRole>,
    pub options: Vec<String>,
    pub deadline: Timestamp,
    pub quorum_bp: Option<u32>,
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
// Multi-Hearth Isolation Tests
// ============================================================================

/// Alice creates hearth1 and hearth2. get_my_hearths returns 2. Alice creates
/// a decision in hearth1. get_hearth_decisions for hearth2 returns 0 (proving
/// cross-hearth isolation).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_multi_hearth_isolation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates hearth1
    let hearth1_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Hearth One".to_string(),
                description: "First hearth for isolation test".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(8),
            },
        )
        .await;

    let hearth1_hash = hearth1_record.action_address().clone();

    // 2. Alice creates hearth2
    let hearth2_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Hearth Two".to_string(),
                description: "Second hearth for isolation test".to_string(),
                hearth_type: HearthType::Intentional,
                max_members: Some(5),
            },
        )
        .await;

    let hearth2_hash = hearth2_record.action_address().clone();

    // 3. get_my_hearths should return 2
    let hearths: Vec<Record> = conductor
        .call(&alice.zome("hearth_kinship"), "get_my_hearths", ())
        .await;

    assert_eq!(
        hearths.len(),
        2,
        "get_my_hearths should return 2 hearths"
    );

    // 4. Alice creates a decision in hearth1
    let deadline_micros = Timestamp::now().as_micros() + 3_600_000_000; // 1 hour
    let deadline = Timestamp::from_micros(deadline_micros);

    let decision_input = CreateDecisionInput {
        hearth_hash: hearth1_hash.clone(),
        title: "Hearth One Decision".to_string(),
        description: "This decision belongs to hearth1 only".to_string(),
        decision_type: DecisionType::MajorityVote,
        eligible_roles: vec![MemberRole::Founder, MemberRole::Adult],
        options: vec!["Option A".to_string(), "Option B".to_string()],
        deadline,
        quorum_bp: None,
    };

    let _decision: Record = conductor
        .call(
            &alice.zome("hearth_decisions"),
            "create_decision",
            decision_input,
        )
        .await;

    // 5. get_hearth_decisions for hearth1 should return 1
    let hearth1_decisions: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_hearth_decisions",
            hearth1_hash,
        )
        .await;

    assert_eq!(
        hearth1_decisions.len(),
        1,
        "hearth1 should have exactly 1 decision"
    );

    // 6. get_hearth_decisions for hearth2 should return 0 (isolation)
    let hearth2_decisions: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_hearth_decisions",
            hearth2_hash,
        )
        .await;

    assert_eq!(
        hearth2_decisions.len(),
        0,
        "hearth2 should have 0 decisions (cross-hearth isolation)"
    );
}
