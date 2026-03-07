//! # Mycelix Hearth — Resources Sweettest Integration Tests
//!
//! Tests the resources zome: register resources, lend/return lifecycle,
//! budget tracking, and membership authorization.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_resources -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — kinship (needed for hearth creation + invite flow)
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InviteMemberInput {
    pub hearth_hash: ActionHash,
    pub invitee_agent: AgentPubKey,
    pub proposed_role: MemberRole,
    pub message: String,
    pub expires_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcceptInvitationInput {
    pub invitation_hash: ActionHash,
    pub display_name: String,
}

// ============================================================================
// Mirror types — resources
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ResourceType {
    Tool,
    Vehicle,
    Book,
    Kitchen,
    Electronics,
    Clothing,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum LoanStatus {
    Active,
    Returned,
    Overdue,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterResourceInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub condition: String,
    pub location: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LendResourceInput {
    pub resource_hash: ActionHash,
    pub borrower: AgentPubKey,
    pub due_date: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateBudgetInput {
    pub hearth_hash: ActionHash,
    pub category: String,
    pub monthly_target_cents: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogExpenseInput {
    pub budget_hash: ActionHash,
    pub amount_cents: u64,
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
// Resources Tests
// ============================================================================

/// Alice creates a hearth and registers a shared resource. Query should
/// return it via get_hearth_inventory.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_register_and_query_resource() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Resource Test Hearth".to_string(),
                description: "Testing resource registration".to_string(),
                hearth_type: HearthType::Chosen,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice registers a shared resource
    let resource_record: Record = conductor
        .call(
            &alice.zome("hearth_resources"),
            "register_resource",
            RegisterResourceInput {
                hearth_hash: hearth_hash.clone(),
                name: "Power Drill".to_string(),
                description: "18V cordless drill".to_string(),
                resource_type: ResourceType::Tool,
                condition: "Good".to_string(),
                location: "Garage shelf A".to_string(),
            },
        )
        .await;

    assert!(resource_record.action().author() == alice.agent_pubkey());

    // 3. Query inventory — should return 1 resource
    let inventory: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_resources"),
            "get_hearth_inventory",
            hearth_hash,
        )
        .await;

    assert_eq!(
        inventory.len(),
        1,
        "Hearth inventory should have exactly 1 resource"
    );
}

/// Full lending lifecycle: register resource -> lend to Bob -> return.
/// get_resource_loans should show the loan, and return should succeed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_lend_and_return_resource() {
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let bob_agent = bob.agent_pubkey().clone();

    // 1. Alice creates a hearth
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Lending Test Hearth".to_string(),
                description: "Testing resource lending".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(10),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice invites Bob
    let invitation_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "invite_member",
            InviteMemberInput {
                hearth_hash: hearth_hash.clone(),
                invitee_agent: bob_agent.clone(),
                proposed_role: MemberRole::Adult,
                message: "Join for lending test".to_string(),
                expires_at: Timestamp::from_micros(
                    Timestamp::now().as_micros() + 86_400_000_000,
                ),
            },
        )
        .await;

    let invitation_hash = invitation_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 3. Bob accepts
    let _: Record = bob_conductor
        .call(
            &bob.zome("hearth_kinship"),
            "accept_invitation",
            AcceptInvitationInput {
                invitation_hash,
                display_name: "Bob".to_string(),
            },
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 4. Alice registers a resource
    let resource_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_resources"),
            "register_resource",
            RegisterResourceInput {
                hearth_hash: hearth_hash.clone(),
                name: "Tent".to_string(),
                description: "4-person camping tent".to_string(),
                resource_type: ResourceType::Custom("Camping".to_string()),
                condition: "Like new".to_string(),
                location: "Attic".to_string(),
            },
        )
        .await;

    let resource_hash = resource_record.action_address().clone();

    // 5. Alice lends the resource to Bob (Alice is Founder = guardian)
    let due_date = Timestamp::from_micros(
        Timestamp::now().as_micros() + 604_800_000_000, // 7 days
    );

    let loan_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_resources"),
            "lend_resource",
            LendResourceInput {
                resource_hash: resource_hash.clone(),
                borrower: bob_agent.clone(),
                due_date,
            },
        )
        .await;

    let loan_hash = loan_record.action_address().clone();

    // 6. Check loans exist for this resource
    let loans: Vec<Record> = alice_conductor
        .call(
            &alice.zome("hearth_resources"),
            "get_resource_loans",
            resource_hash,
        )
        .await;

    assert_eq!(loans.len(), 1, "Resource should have exactly 1 loan");

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 7. Bob returns the resource (borrower can return)
    let returned_record: Record = bob_conductor
        .call(
            &bob.zome("hearth_resources"),
            "return_resource",
            loan_hash,
        )
        .await;

    assert!(
        returned_record.action().author() == bob.agent_pubkey(),
        "Return should be authored by Bob"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

/// Alice creates a hearth, creates a budget category, logs expenses,
/// and verifies the budget summary returns the category.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_budget_and_log_expense() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 1. Alice creates a hearth
    let hearth_record: Record = conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Budget Test Hearth".to_string(),
                description: "Testing budget tracking".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(5),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // 2. Alice creates a budget category (she is Founder = guardian)
    let budget_record: Record = conductor
        .call(
            &alice.zome("hearth_resources"),
            "create_budget_category",
            CreateBudgetInput {
                hearth_hash: hearth_hash.clone(),
                category: "Groceries".to_string(),
                monthly_target_cents: 60000, // $600
            },
        )
        .await;

    let budget_hash = budget_record.action_address().clone();

    // 3. Alice logs an expense
    let _expense_record: Record = conductor
        .call(
            &alice.zome("hearth_resources"),
            "log_expense",
            LogExpenseInput {
                budget_hash: budget_hash.clone(),
                amount_cents: 4299, // $42.99
            },
        )
        .await;

    // 4. Alice logs a second expense
    let _expense_record_2: Record = conductor
        .call(
            &alice.zome("hearth_resources"),
            "log_expense",
            LogExpenseInput {
                budget_hash,
                amount_cents: 8750, // $87.50
            },
        )
        .await;

    // 5. Get budget summary — should return 1 category
    let budgets: Vec<Record> = conductor
        .call(
            &alice.zome("hearth_resources"),
            "get_budget_summary",
            hearth_hash,
        )
        .await;

    assert_eq!(
        budgets.len(),
        1,
        "Budget summary should have exactly 1 category"
    );
}

/// A non-member agent cannot register resources for a hearth they don't
/// belong to. This tests the membership authorization hardening.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_membership_required_for_register() {
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    // 1. Alice creates a hearth (Bob is NOT invited)
    let hearth_record: Record = alice_conductor
        .call(
            &alice.zome("hearth_kinship"),
            "create_hearth",
            CreateHearthInput {
                name: "Exclusive Hearth".to_string(),
                description: "Testing non-member rejection".to_string(),
                hearth_type: HearthType::Nuclear,
                max_members: Some(5),
            },
        )
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    // 2. Bob tries to register a resource -- should fail (not a member)
    let result: Result<Record, _> = bob_conductor
        .call_fallible(
            &bob.zome("hearth_resources"),
            "register_resource",
            RegisterResourceInput {
                hearth_hash,
                name: "Unauthorized Item".to_string(),
                description: "This should be rejected".to_string(),
                resource_type: ResourceType::Tool,
                condition: "N/A".to_string(),
                location: "N/A".to_string(),
            },
        )
        .await;

    assert!(
        result.is_err(),
        "Non-member should not be able to register a resource"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
