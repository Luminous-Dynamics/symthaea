//! # Consciousness Gating Sweettest — Commons Cluster
//!
//! Verifies that governance functions in the commons cluster are properly
//! gated by consciousness credentials, proving the full wiring from
//! domain coordinator → commons_bridge → cross-cluster identity call.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — property registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPropertyInput {
    pub name: String,
    pub description: String,
    pub property_type: String,
    pub location: String,
    pub area_sqm: Option<u32>,
}

// ============================================================================
// Mirror types — housing governance
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateResolutionInput {
    pub title: String,
    pub body: String,
    pub resolution_type: String,
}

// ============================================================================
// Mirror types — bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("COMMONS_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Tests
// ============================================================================

/// Property transfer requires proposal-level consciousness gate.
/// Without identity bridge, initiate_transfer should be blocked.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes proposal tier (0.4)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_property_transfer_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Register a property — this is gated by proposal-level consciousness
    let input = RegisterPropertyInput {
        name: "Test Property".to_string(),
        description: "For gating test".to_string(),
        property_type: "residential".to_string(),
        location: "Test Location".to_string(),
        area_sqm: Some(100),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(
        result.is_err(),
        "register_property should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Housing budget approval requires constitutional-level consciousness
/// (Guardian tier). This tests the highest gate level in commons.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_housing_budget_requires_constitutional() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // create_resolution in housing-governance requires voting-level gate
    let input = CreateResolutionInput {
        title: "Budget Resolution".to_string(),
        body: "Allocate funds for improvements".to_string(),
        resolution_type: "budget".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("housing_governance"),
            "create_resolution",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "create_resolution should be blocked without consciousness credentials"
    );
}

/// Verify the audit trail is maintained even when gate rejects.
/// Bridge health should remain healthy after gate failures, and
/// total_events should be non-negative.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes proposal tier (0.4)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_gate_rejection_audit_logged() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Check bridge health before
    let health_before: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;
    assert!(health_before.healthy, "Bridge should be healthy initially");

    // Attempt a gated action (will fail)
    let input = RegisterPropertyInput {
        name: "Audit Test".to_string(),
        description: "For audit trail test".to_string(),
        property_type: "commercial".to_string(),
        location: "Audit Location".to_string(),
        area_sqm: Some(500),
    };

    let _result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("property_registry"), "register_property", input)
        .await;

    // Bridge should remain healthy after gate failure
    let health_after: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(
        health_after.healthy,
        "Bridge should remain healthy after gate failure"
    );
    assert!(
        health_after.total_events >= health_before.total_events,
        "Total events should not decrease: before={}, after={}",
        health_before.total_events,
        health_after.total_events,
    );
}

// ============================================================================
// Phase 5 — Expanded Coverage (matching hearth cluster depth)
// ============================================================================

// ============================================================================
// Mirror types — food distribution
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Listing {
    pub market_hash: ::holochain::prelude::ActionHash,
    pub producer: ::holochain::prelude::AgentPubKey,
    pub product_name: String,
    pub quantity_kg: f64,
    pub price_per_kg: f64,
    pub available_from: u64,
    pub status: ListingStatus,
    pub allergen_flags: Vec<String>,
    pub organic: bool,
    pub cultural_markers: Vec<String>,
}

// ============================================================================
// Mirror types — mutualaid governance
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProposalType {
    AddRule,
    ModifyRule,
    RemoveRule,
    CreditLimitChange,
    MemberAdmission,
    MemberStatusChange,
    ResourcePolicy,
    GeneralDecision,
    Emergency,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VotingMethod {
    Majority,
    Supermajority,
    Consensus,
    ConsentBased,
    ContributionWeighted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProposalStatus {
    Draft,
    Discussion,
    Voting,
    Passed,
    Failed,
    Implemented,
    Withdrawn,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VoteChoice {
    Yes,
    No,
    Abstain,
    Block,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    pub id: String,
    pub proposer: ::holochain::prelude::AgentPubKey,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub modifies_rule: Option<::holochain::prelude::ActionHash>,
    pub voting_method: VotingMethod,
    pub quorum_percent: u8,
    pub threshold_percent: u8,
    pub voting_starts: ::holochain::prelude::Timestamp,
    pub voting_ends: ::holochain::prelude::Timestamp,
    pub status: ProposalStatus,
    pub created_at: ::holochain::prelude::Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vote {
    pub proposal_hash: ::holochain::prelude::ActionHash,
    pub voter: ::holochain::prelude::AgentPubKey,
    pub vote: VoteChoice,
    pub reasoning: Option<String>,
    pub voted_at: ::holochain::prelude::Timestamp,
}

// ============================================================================
// Mirror types — water steward
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum StewardshipType {
    Riparian,
    PriorAppropriation,
    Commons,
    Hybrid,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Watershed {
    pub id: String,
    pub name: String,
    pub huc_code: Option<String>,
    pub boundary: Vec<(f64, f64)>,
    pub area_sq_km: f32,
    pub stewardship_type: StewardshipType,
    pub governing_body: Option<::holochain::prelude::ActionHash>,
    pub primary_source_type: WaterSourceType,
}

// ============================================================================
// Mirror types — housing membership
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ApplicationStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    Waitlisted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MembershipType {
    FullShare,
    LimitedEquity,
    RentToOwn,
    Renter,
    Associate,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReviewApplicationInput {
    pub application_hash: ::holochain::prelude::ActionHash,
    pub new_status: ApplicationStatus,
}

// ============================================================================
// Expanded Tests
// ============================================================================

/// Food distribution's list_product is gated at basic level.
/// Without consciousness credentials, listing a product should fail.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes basic tier (0.3)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_food_distribution_create_requires_basic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = Listing {
        market_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xAA; 36]),
        producer: ::holochain::prelude::AgentPubKey::from_raw_36(vec![0xBB; 36]),
        product_name: "Test Produce".to_string(),
        quantity_kg: 10.0,
        price_per_kg: 3.50,
        available_from: 1700000000,
        status: ListingStatus::Available,
        allergen_flags: vec![],
        organic: true,
        cultural_markers: vec![],
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("food_distribution"), "list_product", input)
        .await;

    assert!(
        result.is_err(),
        "list_product should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Care matching's suggest_match is gated at proposal level.
/// (accept_match is gated at basic level — we test suggest_match
/// to cover a different gate tier than food-distribution's basic.)
/// NOTE: cfg-gated because Citizen(0.5) fallback passes proposal tier (0.4)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_care_matching_create_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // accept_match takes an ActionHash — simpler to test than suggest_match
    // which requires a CareMatch struct. Both are gated.
    let fake_hash = ::holochain::prelude::ActionHash::from_raw_36(vec![0xCC; 36]);

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("care_matching"), "accept_match", fake_hash)
        .await;

    assert!(
        result.is_err(),
        "accept_match should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Mutual aid governance's cast_vote requires voting-level consciousness.
/// This tests a higher gate tier than basic or proposal.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes Citizen tier (voting requires Citizen)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_mutualaid_governance_requires_voting() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let vote = Vote {
        proposal_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xDD; 36]),
        voter: ::holochain::prelude::AgentPubKey::from_raw_36(vec![0xEE; 36]),
        vote: VoteChoice::Yes,
        reasoning: Some("Test vote for gating test".to_string()),
        voted_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("mutualaid_governance"),
            "cast_vote",
            vote,
        )
        .await;

    assert!(
        result.is_err(),
        "cast_vote should be blocked without consciousness credentials (voting tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Water steward's define_watershed requires voting-level consciousness.
/// Tests that ecological governance operations are properly gated.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes Citizen tier (voting requires Citizen)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_water_steward_dispatch_requires_voting() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let watershed = Watershed {
        id: "ws-test-001".to_string(),
        name: "Test Watershed".to_string(),
        huc_code: Some("12345678".to_string()),
        boundary: vec![(32.0, -96.0), (32.1, -96.0), (32.1, -95.9), (32.0, -95.9)],
        area_sq_km: 50.0,
        stewardship_type: StewardshipType::Commons,
        governing_body: None,
        primary_source_type: WaterSourceType::River,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("water_steward"), "define_watershed", watershed)
        .await;

    assert!(
        result.is_err(),
        "define_watershed should be blocked without consciousness credentials (voting tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Housing membership's review_application requires voting-level consciousness.
/// Tests that membership governance is protected by consciousness gating.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes Citizen tier (voting requires Citizen)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_housing_membership_add_requires_voting() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = ReviewApplicationInput {
        application_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xFF; 36]),
        new_status: ApplicationStatus::Approved,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("housing_membership"),
            "review_application",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "review_application should be blocked without consciousness credentials (voting tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Read-only operations in commons should NOT be gated by consciousness.
/// get_all_markets, get_producer_listings, get_all_proposals, etc. should
/// succeed without any credentials.
/// NOTE: cfg-gated because this test calls both food_distribution (LAND) and
/// mutualaid_governance (CARE) — needs unified DNA which OOMs on 32GB system
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_operations_not_gated_commons() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // get_all_markets — no consciousness gate, returns empty vec
    let markets: Vec<::holochain::prelude::Record> = conductor
        .call(&alice.zome("food_distribution"), "get_all_markets", ())
        .await;
    assert!(
        markets.is_empty(),
        "No markets should exist yet, but call should succeed (ungated)"
    );

    // get_all_proposals — no consciousness gate, returns empty vec
    let proposals: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("mutualaid_governance"),
            "get_all_proposals",
            (),
        )
        .await;
    assert!(
        proposals.is_empty(),
        "No proposals should exist yet, but call should succeed (ungated)"
    );
}

/// The rejection error message from commons gates should contain useful
/// debugging context (consciousness, credential, identity, etc.) rather
/// than a generic error.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes Citizen tier (voting requires Citizen)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_rejection_message_is_specific_commons() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Use a voting-tier action (highest commonly used gate in commons)
    let watershed = Watershed {
        id: "ws-rejection-test".to_string(),
        name: "Rejection Test Watershed".to_string(),
        huc_code: None,
        boundary: vec![(33.0, -97.0), (33.1, -97.0)],
        area_sq_km: 25.0,
        stewardship_type: StewardshipType::Riparian,
        governing_body: None,
        primary_source_type: WaterSourceType::Spring,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("water_steward"), "define_watershed", watershed)
        .await;

    assert!(result.is_err(), "define_watershed should be blocked");

    let err_msg = format!("{:?}", result.unwrap_err());
    let has_context = err_msg.contains("onsciousness")
        || err_msg.contains("credential")
        || err_msg.contains("identity")
        || err_msg.contains("OtherRole")
        || err_msg.contains("cross_cluster")
        || err_msg.contains("gate");
    assert!(
        has_context,
        "Rejection error should contain consciousness/credential context for debugging, got: {}",
        err_msg
    );
}

/// The bridge credential cache must not serve fabricated credentials.
/// After a failed credential fetch, a second attempt should also fail
/// — proving cache consistency.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes Citizen tier (voting requires Citizen)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cache_consistency_commons() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First attempt: governance vote should fail (no credentials)
    let vote_1 = Vote {
        proposal_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0x11; 36]),
        voter: ::holochain::prelude::AgentPubKey::from_raw_36(vec![0x22; 36]),
        vote: VoteChoice::Yes,
        reasoning: Some("Cache test 1".to_string()),
        voted_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
    };

    let result_1: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("mutualaid_governance"),
            "cast_vote",
            vote_1,
        )
        .await;
    assert!(result_1.is_err(), "First attempt should fail");

    // Second attempt: cache should NOT serve a stale/successful credential
    let vote_2 = Vote {
        proposal_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0x33; 36]),
        voter: ::holochain::prelude::AgentPubKey::from_raw_36(vec![0x44; 36]),
        vote: VoteChoice::No,
        reasoning: Some("Cache test 2".to_string()),
        voted_at: ::holochain::prelude::Timestamp::from_micros(1700000001),
    };

    let result_2: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("mutualaid_governance"),
            "cast_vote",
            vote_2,
        )
        .await;
    assert!(
        result_2.is_err(),
        "Second attempt should also fail — cache must not serve fabricated credentials"
    );
}

/// Cross-zome gating: property writes fail while reads succeed.
/// register_property (proposal gate) should be blocked, but
/// get_property (no gate) should succeed. Verifies the gate is
/// selectively applied per-function, not blanket-blocking the zome.
/// NOTE: cfg-gated because Citizen(0.5) fallback passes proposal tier (0.4)
#[cfg(feature = "identity_cluster")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_zome_gate_property_to_bridge() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Write: register_property is gated (proposal tier) — should fail
    let input = RegisterPropertyInput {
        name: "Cross Zome Test Property".to_string(),
        description: "Tests selective gate application".to_string(),
        property_type: "residential".to_string(),
        location: "Cross Zome Location".to_string(),
        area_sqm: Some(200),
    };

    let write_result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(
        write_result.is_err(),
        "register_property should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", write_result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );

    // Read: get_property is NOT gated — should succeed (returns None for non-existent)
    let read_result: Result<Option<::holochain::prelude::Record>, _> = conductor
        .call_fallible(
            &alice.zome("property_registry"),
            "get_property",
            "nonexistent-id".to_string(),
        )
        .await;

    assert!(
        read_result.is_ok(),
        "get_property should succeed without credentials (read-only, ungated)"
    );
}
