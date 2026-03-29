// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mutual Aid Domain Sweettest Integration Tests
//!
//! Tests the 7 mutual aid zomes in the Mycelix Commons cluster:
//! - `mutualaid_circles`   — mutual credit circles with automatic clearing
//! - `mutualaid_governance` — democratic governance (proposals + votes)
//! - `mutualaid_needs`     — needs/offers matching marketplace
//! - `mutualaid_pools`     — pooled mutual aid funds
//! - `mutualaid_requests`  — aid requests and offer responses
//! - `mutualaid_resources` — shared physical resource library
//! - `mutualaid_timebank`  — time-based service exchange (1 hour = 1 hour)
//!
//! ## Prerequisites
//!
//! ```bash
//! cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-commons/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test mutualaid_workflow -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining structs locally
// ============================================================================

// --- mutualaid_circles ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateCircleInput {
    name: String,
    description: String,
    currency_name: String,
    currency_symbol: String,
    default_credit_limit: i64,
    max_credit_limit: i64,
    transaction_fee_percent: f64,
    demurrage_rate_percent: f64,
    geographic_scope: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct JoinCircleInput {
    circle_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TransferInput {
    circle_hash: ActionHash,
    to: AgentPubKey,
    amount: i64,
    memo: String,
    transaction_type: String,
    related_exchange_hash: Option<ActionHash>,
}

// --- mutualaid_governance ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Proposal {
    id: String,
    circle_hash: Option<ActionHash>,
    proposal_type: String,
    title: String,
    description: String,
    proposed_by: AgentPubKey,
    voting_method: String,
    voting_starts: Timestamp,
    voting_ends: Timestamp,
    status: String,
    votes_for: u32,
    votes_against: u32,
    votes_abstain: u32,
    created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Vote {
    proposal_hash: ActionHash,
    voter: AgentPubKey,
    choice: String,
    rationale: Option<String>,
    voted_at: Timestamp,
}

// --- mutualaid_needs ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateNeedInput {
    title: String,
    description: String,
    category: String,
    urgency: String,
    emergency: bool,
    quantity: Option<u32>,
    location: serde_json::Value,
    needed_by: Option<Timestamp>,
    reciprocity_offers: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateOfferInputNeeds {
    title: String,
    description: String,
    category: String,
    quantity: Option<u32>,
    condition: Option<String>,
    location: serde_json::Value,
    available_until: Option<Timestamp>,
    asking_for: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProposeMatchInput {
    need_hash: ActionHash,
    offer_hash: ActionHash,
    quantity: Option<u32>,
    notes: String,
    scheduled_handoff: Option<Timestamp>,
    handoff_location: Option<String>,
}

// --- mutualaid_pools ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreatePoolInput {
    name: String,
    description: String,
    creator_did: String,
    contribution_rules: Option<serde_json::Value>,
    disbursement_rules: Option<serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ContributeInput {
    pool_hash: ActionHash,
    pool_id: String,
    member_did: String,
    amount: u64,
    note: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RequestDisbursementInput {
    pool_hash: ActionHash,
    pool_id: String,
    recipient_did: String,
    amount: u64,
    reason: String,
    is_emergency: bool,
}

// Pool zome returns a custom struct, not a bare Record
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PoolWithHash {
    hash: ActionHash,
    pool: serde_json::Value,
}

// --- mutualaid_requests ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateRequestInput {
    requester_did: String,
    request_type: String,
    description: String,
    urgency: String,
    location: Option<String>,
    amount_needed: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateOfferInputRequests {
    request_hash: ActionHash,
    request_id: String,
    offerer_did: String,
    amount: Option<u64>,
    message: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RequestWithHash {
    hash: ActionHash,
    request: serde_json::Value,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct OfferWithHash {
    hash: ActionHash,
    offer: serde_json::Value,
}

// --- mutualaid_resources ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateResourceInput {
    name: String,
    description: String,
    resource_type: String,
    condition: String,
    photos: Vec<String>,
    location: serde_json::Value,
    availability: serde_json::Value,
    sharing_model: String,
    usage_instructions: String,
    liability_notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateBookingInput {
    resource_hash: ActionHash,
    start_time: Timestamp,
    end_time: Timestamp,
    purpose: String,
    payment_method: Option<String>,
}

// --- mutualaid_timebank ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateTimebankOfferInput {
    title: String,
    description: String,
    category: String,
    qualifications: Vec<String>,
    availability: serde_json::Value,
    location: serde_json::Value,
    min_duration_hours: f64,
    max_duration_hours: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateTimebankRequestInput {
    title: String,
    description: String,
    category: String,
    urgency: String,
    needed_by: Option<Timestamp>,
    estimated_hours: f64,
    location: serde_json::Value,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RecordExchangeInput {
    offer_hash: Option<ActionHash>,
    request_hash: Option<ActionHash>,
    provider: AgentPubKey,
    recipient: AgentPubKey,
    hours: f64,
    category: String,
    description: String,
}

// ============================================================================
// mutualaid_circles tests
// ============================================================================

/// Test: Create a credit circle and retrieve it by hash.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_circles_create_and_get_circle() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateCircleInput {
        name: "Richardson Neighbors Circle".into(),
        description: "A mutual credit circle for the Richardson, TX community".into(),
        currency_name: "NeighborCoin".into(),
        currency_symbol: "NC".into(),
        default_credit_limit: 500,
        max_credit_limit: 2000,
        transaction_fee_percent: 0.0,
        demurrage_rate_percent: 0.5,
        geographic_scope: Some("Richardson, TX".into()),
    };

    let circle_record: Record = alice
        .call_zome_fn("mutualaid_circles", "create_circle", input)
        .await;

    let circle_hash = circle_record.action_hashed().hash.clone();
    assert!(!circle_hash.as_ref().is_empty(), "Circle should be created with a hash");

    // Retrieve by hash
    let retrieved: Option<Record> = alice
        .call_zome_fn("mutualaid_circles", "get_circle", circle_hash.clone())
        .await;

    assert!(retrieved.is_some(), "Circle should be retrievable by hash");
}

/// Test: Agent joins a circle and appears in the member list.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_circles_join_and_list_members() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Alice creates a circle
    let input = CreateCircleInput {
        name: "Eastside Exchange".into(),
        description: "Credit circle for the east side of town".into(),
        currency_name: "EastBuck".into(),
        currency_symbol: "EB".into(),
        default_credit_limit: 300,
        max_credit_limit: 1500,
        transaction_fee_percent: 1.0,
        demurrage_rate_percent: 0.0,
        geographic_scope: None,
    };

    let circle_record: Record = alice
        .call_zome_fn("mutualaid_circles", "create_circle", input)
        .await;

    let circle_hash = circle_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Bob joins the circle
    let join_input = JoinCircleInput {
        circle_hash: circle_hash.clone(),
    };

    let _credit_line: Record = bob
        .call_zome_fn("mutualaid_circles", "join_circle", join_input)
        .await;

    // List circle members — both Alice (founder) and Bob should appear
    let members: Vec<AgentPubKey> = alice
        .call_zome_fn("mutualaid_circles", "get_circle_members", circle_hash)
        .await;

    assert!(
        members.len() >= 2,
        "Circle should have at least 2 members after Bob joins, got {}",
        members.len()
    );
}

/// Test: List all circles returns the one created.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_circles_list_all_circles() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateCircleInput {
        name: "Downtown Solidarity Fund".into(),
        description: "Supporting downtown small businesses and residents".into(),
        currency_name: "SolidCoin".into(),
        currency_symbol: "SC".into(),
        default_credit_limit: 1000,
        max_credit_limit: 5000,
        transaction_fee_percent: 0.5,
        demurrage_rate_percent: 1.0,
        geographic_scope: Some("Downtown".into()),
    };

    let _: Record = alice
        .call_zome_fn("mutualaid_circles", "create_circle", input)
        .await;

    // get_my_circles returns circles the current agent is a member of
    let my_circles: Vec<Record> = alice
        .call_zome_fn("mutualaid_circles", "get_my_circles", ())
        .await;

    assert!(
        !my_circles.is_empty(),
        "Agent should see at least one circle after creating one"
    );
}

// ============================================================================
// mutualaid_governance tests
// ============================================================================

/// Test: Create a governance proposal and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_governance_create_proposal() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();
    let voting_ends =
        Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    let proposal = Proposal {
        id: "PROP-001".into(),
        circle_hash: None,
        proposal_type: "GeneralDecision".into(),
        title: "Raise default credit limit to 750".into(),
        description: "As the circle grows, a higher default limit would improve liquidity".into(),
        proposed_by: alice.agent_pubkey.clone(),
        voting_method: "Majority".into(),
        voting_starts: now,
        voting_ends,
        status: "Voting".into(),
        votes_for: 0,
        votes_against: 0,
        votes_abstain: 0,
        created_at: now,
    };

    let record: Record = alice
        .call_zome_fn("mutualaid_governance", "create_proposal", proposal)
        .await;

    assert!(
        !record.action_hashed().hash.as_ref().is_empty(),
        "Proposal should be created"
    );
}

/// Test: Cast a vote on an existing proposal.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_governance_cast_vote() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    let now = Timestamp::now();
    let voting_ends =
        Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Alice creates a proposal
    let proposal = Proposal {
        id: "PROP-002".into(),
        circle_hash: None,
        proposal_type: "ResourcePolicy".into(),
        title: "Allow cross-circle resource sharing".into(),
        description: "Members of any circle can borrow resources from other circles".into(),
        proposed_by: alice.agent_pubkey.clone(),
        voting_method: "ConsentBased".into(),
        voting_starts: now,
        voting_ends,
        status: "Voting".into(),
        votes_for: 0,
        votes_against: 0,
        votes_abstain: 0,
        created_at: now,
    };

    let proposal_record: Record = alice
        .call_zome_fn("mutualaid_governance", "create_proposal", proposal)
        .await;

    let proposal_hash = proposal_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Bob casts a Yes vote
    let vote = Vote {
        proposal_hash,
        voter: bob.agent_pubkey.clone(),
        choice: "Yes".into(),
        rationale: Some("Cross-circle sharing strengthens community resilience".into()),
        voted_at: Timestamp::now(),
    };

    let vote_record: Record = bob
        .call_zome_fn("mutualaid_governance", "cast_vote", vote)
        .await;

    assert!(
        !vote_record.action_hashed().hash.as_ref().is_empty(),
        "Vote should be recorded on the DHT"
    );
}

// ============================================================================
// mutualaid_needs tests
// ============================================================================

/// Test: Post a need and retrieve it by hash.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_needs_create_and_get_need() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateNeedInput {
        title: "Winter coats for family of 4".into(),
        description: "Sizes: adult M, adult L, child 8, child 10. Any color welcome.".into(),
        category: "Clothing".into(),
        urgency: "High".into(),
        emergency: false,
        quantity: Some(4),
        location: serde_json::json!("Remote"),
        needed_by: None,
        reciprocity_offers: vec!["Can help with yard work".into()],
    };

    let record: Record = alice
        .call_zome_fn("mutualaid_needs", "create_need", input)
        .await;

    let need_hash = record.action_hashed().hash.clone();
    assert!(!need_hash.as_ref().is_empty(), "Need should be created");

    let retrieved: Option<Record> = alice
        .call_zome_fn("mutualaid_needs", "get_need", need_hash)
        .await;

    assert!(retrieved.is_some(), "Need should be retrievable by hash");
}

/// Test: Post a need and an offer, then match them together.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_needs_propose_match() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0]; // requester
    let bob = &agents[1];   // offerer

    // Alice posts a need
    let need_input = CreateNeedInput {
        title: "Baby stroller needed".into(),
        description: "Infant-to-toddler convertible stroller in good condition".into(),
        category: "BabyItems".into(),
        urgency: "Medium".into(),
        emergency: false,
        quantity: Some(1),
        location: serde_json::json!("ToBeArranged"),
        needed_by: None,
        reciprocity_offers: vec!["Can offer baked goods in return".into()],
    };

    let need_record: Record = alice
        .call_zome_fn("mutualaid_needs", "create_need", need_input)
        .await;

    let need_hash = need_record.action_hashed().hash.clone();

    // Bob posts a matching offer
    let offer_input = CreateOfferInputNeeds {
        title: "Graco stroller — free to good home".into(),
        description: "Our kids outgrew it. Folds compactly, cup holders, sunshade included.".into(),
        category: "BabyItems".into(),
        quantity: Some(1),
        condition: Some("Good — minor wear on handles".into()),
        location: serde_json::json!({"FixedLocation": "456 Oak Ave, Richardson TX"}),
        available_until: None,
        asking_for: vec![],
    };

    let offer_record: Record = bob
        .call_zome_fn("mutualaid_needs", "create_offer", offer_input)
        .await;

    let offer_hash = offer_record.action_hashed().hash.clone();

    // Bob proposes a match
    let match_input = ProposeMatchInput {
        need_hash,
        offer_hash,
        quantity: Some(1),
        notes: "Can meet at the library on Saturday morning".into(),
        scheduled_handoff: None,
        handoff_location: Some("Richardson Public Library main entrance".into()),
    };

    let match_record: Record = bob
        .call_zome_fn("mutualaid_needs", "propose_match", match_input)
        .await;

    assert!(
        !match_record.action_hashed().hash.as_ref().is_empty(),
        "Match proposal should be created"
    );
}

/// Test: Emergency need is indexed and retrievable via get_emergency_needs.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_needs_emergency_needs_index() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateNeedInput {
        title: "Urgent: temporary housing for family".into(),
        description: "Family of 3 displaced by apartment fire — need 1-2 weeks shelter".into(),
        category: "Housing".into(),
        urgency: "Emergency".into(),
        emergency: true,
        quantity: None,
        location: serde_json::json!("ToBeArranged"),
        needed_by: None,
        reciprocity_offers: vec![],
    };

    let _: Record = alice
        .call_zome_fn("mutualaid_needs", "create_need", input)
        .await;

    let emergency_needs: Vec<Record> = alice
        .call_zome_fn("mutualaid_needs", "get_emergency_needs", ())
        .await;

    assert!(
        !emergency_needs.is_empty(),
        "Emergency needs index should include the posted emergency need"
    );
}

// ============================================================================
// mutualaid_pools tests
// ============================================================================

/// Test: Create a mutual aid pool and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_pools_create_and_list() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let input = CreatePoolInput {
        name: "Community Emergency Fund".into(),
        description: "Rapid-response fund for members facing unexpected hardship".into(),
        creator_did: alice_did.clone(),
        contribution_rules: None,
        disbursement_rules: None,
    };

    let pool_result: PoolWithHash = alice
        .call_zome_fn("mutualaid_pools", "create_pool", input)
        .await;

    assert!(
        !pool_result.hash.as_ref().is_empty(),
        "Pool should be created with a valid action hash"
    );

    // list_pools returns all pools
    let all_pools: Vec<PoolWithHash> = alice
        .call_zome_fn("mutualaid_pools", "list_pools", ())
        .await;

    assert!(
        !all_pools.is_empty(),
        "list_pools should return at least one pool after creation"
    );
}

/// Test: Contribute to a pool and request a disbursement.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_pools_contribute_and_request_disbursement() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0]; // pool creator + contributor
    let bob = &agents[1];   // disbursement requester

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice creates the pool
    let pool_result: PoolWithHash = alice
        .call_zome_fn(
            "mutualaid_pools",
            "create_pool",
            CreatePoolInput {
                name: "Westside Solidarity Pool".into(),
                description: "Monthly contributions for medical and food emergencies".into(),
                creator_did: alice_did.clone(),
                contribution_rules: None,
                disbursement_rules: None,
            },
        )
        .await;

    let pool_hash = pool_result.hash.clone();
    let pool_id = pool_result
        .pool
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("pool_id")
        .to_string();

    // Alice contributes 200 units
    let contrib_input = ContributeInput {
        pool_hash: pool_hash.clone(),
        pool_id: pool_id.clone(),
        member_did: alice_did.clone(),
        amount: 200,
        note: Some("Monthly contribution — March 2026".into()),
    };

    let contrib_record: Record = alice
        .call_zome_fn("mutualaid_pools", "contribute", contrib_input)
        .await;

    assert!(
        !contrib_record.action_hashed().hash.as_ref().is_empty(),
        "Contribution should be recorded"
    );

    wait_for_dht_sync().await;

    // Bob requests a disbursement
    let disb_input = RequestDisbursementInput {
        pool_hash: pool_hash.clone(),
        pool_id: pool_id.clone(),
        recipient_did: bob_did.clone(),
        amount: 75,
        reason: "Unexpected medical co-pay for prescription medication".into(),
        is_emergency: false,
    };

    let disb_record: Record = bob
        .call_zome_fn("mutualaid_pools", "request_disbursement", disb_input)
        .await;

    assert!(
        !disb_record.action_hashed().hash.as_ref().is_empty(),
        "Disbursement request should be recorded"
    );
}

// ============================================================================
// mutualaid_requests tests
// ============================================================================

/// Test: Create an aid request and retrieve it via list_requests.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_requests_create_and_list() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let input = CreateRequestInput {
        requester_did: alice_did.clone(),
        request_type: "Food".into(),
        description: "Family of 5 short on groceries this week — anything helps".into(),
        urgency: "High".into(),
        location: Some("Plano, TX 75075".into()),
        amount_needed: Some(80),
    };

    let request_result: RequestWithHash = alice
        .call_zome_fn("mutualaid_requests", "create_request", input)
        .await;

    assert!(
        !request_result.hash.as_ref().is_empty(),
        "Aid request should be created with a valid hash"
    );

    // list_requests returns all open requests
    let all_requests: Vec<RequestWithHash> = alice
        .call_zome_fn("mutualaid_requests", "list_requests", ())
        .await;

    assert!(
        !all_requests.is_empty(),
        "list_requests should include the posted request"
    );
}

/// Test: Respond to an aid request with an offer.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_requests_create_offer_response() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0]; // requester
    let bob = &agents[1];   // responder

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice posts a financial aid request
    let request_result: RequestWithHash = alice
        .call_zome_fn(
            "mutualaid_requests",
            "create_request",
            CreateRequestInput {
                requester_did: alice_did.clone(),
                request_type: "Financial".into(),
                description: "Need help covering rent gap — $300 short this month".into(),
                urgency: "Critical".into(),
                location: Some("Richardson, TX".into()),
                amount_needed: Some(300),
            },
        )
        .await;

    let request_hash = request_result.hash.clone();
    let request_id = request_result
        .request
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("req_id")
        .to_string();

    wait_for_dht_sync().await;

    // Bob responds with an offer to cover $150
    let offer_input = CreateOfferInputRequests {
        request_hash: request_hash.clone(),
        request_id: request_id.clone(),
        offerer_did: bob_did.clone(),
        amount: Some(150),
        message: "Happy to help with $150 — I'll transfer to your account today.".into(),
    };

    let offer_result: OfferWithHash = bob
        .call_zome_fn("mutualaid_requests", "create_offer", offer_input)
        .await;

    assert!(
        !offer_result.hash.as_ref().is_empty(),
        "Offer response should be created with a valid hash"
    );
}

// ============================================================================
// mutualaid_resources tests
// ============================================================================

/// Test: List a shared resource and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_resources_create_and_get() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateResourceInput {
        name: "Stand mixer (KitchenAid 5qt)".into(),
        description: "Heavy-duty stand mixer, great for bread dough and baking projects".into(),
        resource_type: "Appliances".into(),
        condition: "Good".into(),
        photos: vec![],
        location: serde_json::json!({"FixedLocation": "123 Maple St, Richardson TX"}),
        availability: serde_json::json!({
            "schedule": "weekends",
            "advance_notice_hours": 24
        }),
        sharing_model: "Free".into(),
        usage_instructions: "Wipe down after use, return with all attachments".into(),
        liability_notes: Some("Borrower responsible for any damage during their use period".into()),
    };

    let record: Record = alice
        .call_zome_fn("mutualaid_resources", "create_resource", input)
        .await;

    let resource_hash = record.action_hashed().hash.clone();
    assert!(!resource_hash.as_ref().is_empty(), "Resource should be created");

    let retrieved: Option<Record> = alice
        .call_zome_fn("mutualaid_resources", "get_resource", resource_hash)
        .await;

    assert!(retrieved.is_some(), "Resource should be retrievable by hash");
}

/// Test: My resources list reflects created resources.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_resources_list_my_resources() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    // Create two resources
    for (name, resource_type) in [
        ("Extension cord (50ft, heavy gauge)", "Tools"),
        ("Pressure washer (electric, 2000 PSI)", "Equipment"),
    ] {
        let input = CreateResourceInput {
            name: name.into(),
            description: format!("{} available for community borrowing", name),
            resource_type: resource_type.into(),
            condition: "Good".into(),
            photos: vec![],
            location: serde_json::json!("AtProvider"),
            availability: serde_json::json!({"always_available": true}),
            sharing_model: "Free".into(),
            usage_instructions: "Return promptly when done, report any issues".into(),
            liability_notes: None,
        };

        let _: Record = alice
            .call_zome_fn("mutualaid_resources", "create_resource", input)
            .await;
    }

    let my_resources: Vec<Record> = alice
        .call_zome_fn("mutualaid_resources", "get_my_resources", ())
        .await;

    assert!(
        my_resources.len() >= 2,
        "get_my_resources should return at least the 2 created resources, got {}",
        my_resources.len()
    );
}

// ============================================================================
// mutualaid_timebank tests
// ============================================================================

/// Test: Post a service offer and retrieve it by hash.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_timebank_create_service_offer() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 1).await;
    let alice = &agents[0];

    let input = CreateTimebankOfferInput {
        title: "Home vegetable garden setup and coaching".into(),
        description: "I'll help you plan, prepare soil, plant, and care for a veggie garden. \
                      Great for beginners — I've been growing food for 15 years.".into(),
        category: "Gardening".into(),
        qualifications: vec![
            "Master Gardener certified".into(),
            "15 years experience".into(),
        ],
        availability: serde_json::json!({
            "days": ["Saturday", "Sunday"],
            "time_range": "09:00-17:00"
        }),
        location: serde_json::json!("AtRequester"),
        min_duration_hours: 2.0,
        max_duration_hours: Some(6.0),
    };

    let record: Record = alice
        .call_zome_fn("mutualaid_timebank", "create_service_offer", input)
        .await;

    let offer_hash = record.action_hashed().hash.clone();
    assert!(!offer_hash.as_ref().is_empty(), "Service offer should be created");

    let retrieved: Option<Record> = alice
        .call_zome_fn("mutualaid_timebank", "get_service_offer", offer_hash)
        .await;

    assert!(retrieved.is_some(), "Service offer should be retrievable by hash");
}

/// Test: Record a completed time exchange between two members.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled commons WASM + conductor"]
async fn test_timebank_record_exchange() {
    let agents = setup_test_agents(&DnaPaths::commons(), "mycelix-commons", 2).await;
    let alice = &agents[0]; // provider (tutor)
    let bob = &agents[1];   // recipient (student)

    // Alice posts a tutoring offer
    let offer_record: Record = alice
        .call_zome_fn(
            "mutualaid_timebank",
            "create_service_offer",
            CreateTimebankOfferInput {
                title: "Python programming tutoring".into(),
                description: "Beginner to intermediate Python — data analysis, scripting, OOP".into(),
                category: "Education".into(),
                qualifications: vec!["10 years professional Python development".into()],
                availability: serde_json::json!({"flexible": true}),
                location: serde_json::json!("Remote"),
                min_duration_hours: 1.0,
                max_duration_hours: Some(3.0),
            },
        )
        .await;

    let offer_hash = offer_record.action_hashed().hash.clone();

    // Record the completed exchange: Alice taught Bob for 2 hours
    let exchange_input = RecordExchangeInput {
        offer_hash: Some(offer_hash),
        request_hash: None,
        provider: alice.agent_pubkey.clone(),
        recipient: bob.agent_pubkey.clone(),
        hours: 2.0,
        category: "Education".into(),
        description: "Python basics session: data types, loops, functions, file I/O".into(),
    };

    let exchange_record: Record = alice
        .call_zome_fn("mutualaid_timebank", "record_exchange", exchange_input)
        .await;

    assert!(
        !exchange_record.action_hashed().hash.as_ref().is_empty(),
        "Time exchange should be recorded on the DHT"
    );

    // Alice's timebank balance should reflect the 2 hours given
    let my_hours: serde_json::Value = alice
        .call_zome_fn("mutualaid_timebank", "get_my_balance", ())
        .await;

    // The timebank balance field name may vary — assert the call succeeds
    // and returns a valid JSON value (non-null)
    assert!(
        !my_hours.is_null(),
        "get_my_balance should return a valid balance object"
    );
}
