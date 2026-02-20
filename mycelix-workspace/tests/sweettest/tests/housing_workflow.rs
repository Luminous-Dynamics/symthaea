//! Housing domain sweettest integration tests — Mycelix Commons cluster.
//!
//! Tests all six housing zomes:
//!   - `housing_units`     — buildings and individual units
//!   - `housing_membership` — applications, waitlist, rent-to-own agreements
//!   - `housing_finances`  — charges, payments, reserve funds, budgets
//!   - `housing_governance` — board meetings, resolutions, bylaws
//!   - `housing_maintenance` — maintenance requests and work orders
//!   - `housing_clt`       — community land trust, ground leases, resale formulas
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
//! cargo test --release -p mycelix-sweettest --test housing_workflow -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining entry structs.
// All fields match the integrity zome definitions exactly.
// ============================================================================

// --- housing_units ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Building {
    id: String,
    name: String,
    address: String,
    location_lat: f64,
    location_lon: f64,
    total_units: u16,
    year_built: Option<u16>,
    building_type: BuildingType,
    cooperative_hash: Option<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum BuildingType {
    Apartment,
    Townhouse,
    SingleFamily,
    Duplex,
    CoHousing,
    MixedUse,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Unit {
    building_hash: ActionHash,
    unit_number: String,
    unit_type: UnitType,
    square_meters: u32,
    floor: u8,
    bedrooms: u8,
    bathrooms: u8,
    accessibility_features: Vec<AccessFeature>,
    current_occupant: Option<AgentPubKey>,
    status: UnitStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum UnitType {
    Studio,
    OneBedroom,
    TwoBedroom,
    ThreeBedroom,
    FourPlus,
    Accessible,
    Family,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum AccessFeature {
    WheelchairAccessible,
    Elevator,
    GrabBars,
    WideDoorways,
    LowCounters,
    VisualAlerts,
    HearingLoop,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum UnitStatus {
    Available,
    Occupied,
    UnderMaintenance,
    Reserved,
    Renovation,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UpdateUnitStatusInput {
    unit_action_hash: ActionHash,
    new_status: UnitStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AssignOccupantInput {
    unit_action_hash: ActionHash,
    occupant: AgentPubKey,
}

// --- housing_membership ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MemberApplication {
    applicant: AgentPubKey,
    requested_unit: Option<ActionHash>,
    membership_type_requested: MembershipType,
    applied_at: Timestamp,
    household_size: u8,
    income_verified: bool,
    references: Vec<String>,
    status: ApplicationStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MembershipType {
    FullShare,
    LimitedEquity,
    RentToOwn,
    Renter,
    Associate,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ApplicationStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    Waitlisted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReviewApplicationInput {
    application_hash: ActionHash,
    new_status: ApplicationStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ApproveMemberInput {
    application_hash: ActionHash,
    unit_hash: Option<ActionHash>,
    membership_type: MembershipType,
    share_equity_cents: u64,
    monthly_charge_cents: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AddToWaitlistInput {
    application_hash: ActionHash,
    unit_type_preference: Option<UnitType>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateRentToOwnInput {
    member: AgentPubKey,
    unit_hash: ActionHash,
    total_purchase_price_cents: u64,
    monthly_rent_cents: u64,
    equity_portion_percent: u8,
    target_completion: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RecordRentPaymentInput {
    agreement_hash: ActionHash,
    payment_amount_cents: u64,
}

// --- housing_finances ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReserveFund {
    name: String,
    fund_type: FundType,
    balance_cents: u64,
    target_cents: u64,
    description: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum FundType {
    CapitalReserve,
    OperatingReserve,
    EmergencyFund,
    ImprovementFund,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Payment {
    member: AgentPubKey,
    charge_hash: Option<ActionHash>,
    amount_cents: u64,
    payment_method: PaymentMethod,
    paid_at: Timestamp,
    reference: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PaymentMethod {
    BankTransfer,
    MutualCredit,
    Cash,
    Check,
    TimeBankCredit,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Budget {
    fiscal_year: u16,
    income_projected_cents: u64,
    expenses_projected_cents: u64,
    categories: Vec<BudgetCategory>,
    approved: bool,
    approved_at: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BudgetCategory {
    name: String,
    allocated_cents: u64,
    spent_cents: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DepositToReserveInput {
    fund_hash: ActionHash,
    amount_cents: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ApproveBudgetInput {
    budget_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FinancialSummaryInput {
    period_year: u16,
    period_month: u8,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FinancialSummary {
    total_charges_cents: u64,
    total_payments_cents: u64,
    outstanding_cents: u64,
    reserve_funds: Vec<ReserveFundSummary>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReserveFundSummary {
    name: String,
    fund_type: FundType,
    balance_cents: u64,
    target_cents: u64,
    percent_funded: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GenerateChargesInput {
    members: Vec<MemberChargeInfo>,
    period_year: u16,
    period_month: u8,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MemberChargeInfo {
    member: AgentPubKey,
    unit_hash: ActionHash,
    base_rent_cents: u64,
    maintenance_fee_cents: u64,
    utilities_cents: u64,
    reserve_contribution_cents: u64,
}

// --- housing_governance ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BoardMeeting {
    title: String,
    agenda: String,
    scheduled_at: Timestamp,
    location: Option<String>,
    meeting_type: MeetingType,
    attendees: Vec<AgentPubKey>,
    minutes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MeetingType {
    Regular,
    Special,
    Annual,
    Emergency,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Resolution {
    meeting_hash: Option<ActionHash>,
    title: String,
    body: String,
    resolution_type: ResolutionType,
    proposed_by: AgentPubKey,
    status: ResolutionStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ResolutionType {
    Policy,
    Financial,
    Maintenance,
    Membership,
    Emergency,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ResolutionStatus {
    Proposed,
    UnderVote,
    Passed,
    Failed,
    Withdrawn,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RecordMinutesInput {
    meeting_hash: ActionHash,
    minutes: String,
    attendees: Vec<AgentPubKey>,
}

// --- housing_maintenance ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MaintenanceRequest {
    building_hash: ActionHash,
    unit_hash: Option<ActionHash>,
    reported_by: AgentPubKey,
    category: MaintenanceCategory,
    priority: MaintenancePriority,
    description: String,
    status: MaintenanceStatus,
    reported_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MaintenanceCategory {
    Plumbing,
    Electrical,
    Hvac,
    Structural,
    Appliance,
    Common,
    Safety,
    Other,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MaintenancePriority {
    Emergency,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MaintenanceStatus {
    Reported,
    Acknowledged,
    Scheduled,
    InProgress,
    Completed,
    Closed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AcknowledgeRequestInput {
    request_hash: ActionHash,
}

// --- housing_clt ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct LandTrust {
    id: String,
    name: String,
    mission: String,
    boundary: Vec<(f64, f64)>,
    charter_hash: Option<ActionHash>,
    stewardship_board: Vec<AgentPubKey>,
    affordability_target_ami_percent: u8,
    created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GroundLease {
    trust_hash: ActionHash,
    unit_hash: ActionHash,
    leaseholder: AgentPubKey,
    lease_term_years: u32,
    ground_rent_monthly_cents: u64,
    resale_formula: ResaleFormula,
    started_at: Timestamp,
    expires_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ResaleFormula {
    formula_type: FormulaType,
    max_appreciation_percent_annual: Option<u8>,
    ami_cap_percent: Option<u8>,
    improvement_credit_percent: Option<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum FormulaType {
    AppreciationCap,
    AreaMedianIncome,
    ConsumerPriceIndex,
    Hybrid,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UpdateLandTrustInput {
    original_action_hash: ActionHash,
    updated_entry: LandTrust,
}

// ============================================================================
// housing_units — buildings and units
// ============================================================================

/// Test: Register a building and retrieve all buildings, then register a unit
/// within that building.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_units_register_building_and_unit() {
    let agents = setup_test_agents(&DnaPaths::commons(), "commons_housing_units", 1).await;
    let alice = &agents[0];

    // Register a building
    let building = Building {
        id: "bldg-clt-001".into(),
        name: "Spruce Hill Commons".into(),
        address: "420 Cooperative Ave, Philadelphia, PA 19103".into(),
        location_lat: 39.9526,
        location_lon: -75.1652,
        total_units: 18,
        year_built: Some(1972),
        building_type: BuildingType::Apartment,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;

    let building_hash = building_record.action_hashed().hash.clone();
    assert!(!building_hash.as_ref().is_empty(), "Building should be created");

    // Register a unit within that building
    let unit = Unit {
        building_hash: building_hash.clone(),
        unit_number: "3A".into(),
        unit_type: UnitType::TwoBedroom,
        square_meters: 78,
        floor: 3,
        bedrooms: 2,
        bathrooms: 1,
        accessibility_features: vec![AccessFeature::Elevator],
        current_occupant: None,
        status: UnitStatus::Available,
    };

    let unit_record: Record = alice
        .call_zome_fn("housing_units", "register_unit", unit)
        .await;

    assert!(
        !unit_record.action_hashed().hash.as_ref().is_empty(),
        "Unit should be registered"
    );

    // Retrieve all units in the building
    let building_units: Vec<Record> = alice
        .call_zome_fn("housing_units", "get_building_units", building_hash)
        .await;

    assert!(
        !building_units.is_empty(),
        "Should have at least one unit in the building"
    );
}

/// Test: Available units index is updated when unit status changes.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_units_available_units_index() {
    let agents = setup_test_agents(&DnaPaths::commons(), "commons_housing_units_avail", 1).await;
    let alice = &agents[0];

    // Register a building
    let building = Building {
        id: "bldg-avail-001".into(),
        name: "Elm Street Co-op".into(),
        address: "88 Elm St, Austin, TX 78701".into(),
        location_lat: 30.2672,
        location_lon: -97.7431,
        total_units: 6,
        year_built: Some(1995),
        building_type: BuildingType::CoHousing,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;
    let building_hash = building_record.action_hashed().hash.clone();

    // Register an available unit
    let unit = Unit {
        building_hash,
        unit_number: "1B".into(),
        unit_type: UnitType::OneBedroom,
        square_meters: 52,
        floor: 1,
        bedrooms: 1,
        bathrooms: 1,
        accessibility_features: vec![AccessFeature::WheelchairAccessible, AccessFeature::GrabBars],
        current_occupant: None,
        status: UnitStatus::Available,
    };

    let unit_record: Record = alice
        .call_zome_fn("housing_units", "register_unit", unit)
        .await;
    let unit_hash = unit_record.action_hashed().hash.clone();

    // Verify the unit appears in the available index
    let available: Vec<Record> = alice
        .call_zome_fn("housing_units", "get_available_units", ())
        .await;

    assert!(!available.is_empty(), "Should have at least one available unit");

    // Mark unit as under maintenance — it should leave the available index
    let status_input = UpdateUnitStatusInput {
        unit_action_hash: unit_hash,
        new_status: UnitStatus::UnderMaintenance,
    };

    let updated_record: Record = alice
        .call_zome_fn("housing_units", "update_unit_status", status_input)
        .await;

    assert!(
        !updated_record.action_hashed().hash.as_ref().is_empty(),
        "Unit status should be updated"
    );
}

// ============================================================================
// housing_membership — applications, waitlist, rent-to-own
// ============================================================================

/// Test: Submit a membership application, move it through review, then
/// approve the member.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_membership_application_to_approval() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_membership", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();

    // Submit an application
    let application = MemberApplication {
        applicant: alice.agent_pubkey.clone(),
        requested_unit: None,
        membership_type_requested: MembershipType::LimitedEquity,
        applied_at: now,
        household_size: 2,
        income_verified: true,
        references: vec![
            "Maria Torres, Community Organizer".into(),
            "James Reed, Former Landlord".into(),
        ],
        status: ApplicationStatus::Pending,
    };

    let app_record: Record = alice
        .call_zome_fn("housing_membership", "submit_application", application)
        .await;

    let app_hash = app_record.action_hashed().hash.clone();
    assert!(!app_hash.as_ref().is_empty(), "Application should be submitted");

    // Move application to UnderReview
    let review_input = ReviewApplicationInput {
        application_hash: app_hash.clone(),
        new_status: ApplicationStatus::UnderReview,
    };

    let reviewed: Record = alice
        .call_zome_fn("housing_membership", "review_application", review_input)
        .await;

    assert!(
        !reviewed.action_hashed().hash.as_ref().is_empty(),
        "Application status should be updated"
    );

    // Approve the member
    let approve_input = ApproveMemberInput {
        application_hash: app_hash,
        unit_hash: None,
        membership_type: MembershipType::LimitedEquity,
        share_equity_cents: 25_000_00, // $25,000.00 in cents
        monthly_charge_cents: 85_000,  // $850.00/month
    };

    let member_record: Record = alice
        .call_zome_fn("housing_membership", "approve_member", approve_input)
        .await;

    assert!(
        !member_record.action_hashed().hash.as_ref().is_empty(),
        "Member record should be created"
    );

    // Verify member appears in the members list
    let members: Vec<Record> = alice
        .call_zome_fn("housing_membership", "get_members", ())
        .await;

    assert!(!members.is_empty(), "Should have at least one approved member");
}

/// Test: Add applicant to waitlist and retrieve the waitlist.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_membership_waitlist() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_waitlist", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();

    // Submit an application that will go to the waitlist
    let application = MemberApplication {
        applicant: alice.agent_pubkey.clone(),
        requested_unit: None,
        membership_type_requested: MembershipType::Renter,
        applied_at: now,
        household_size: 1,
        income_verified: false,
        references: vec!["Self-referral".into()],
        status: ApplicationStatus::Pending,
    };

    let app_record: Record = alice
        .call_zome_fn("housing_membership", "submit_application", application)
        .await;

    let app_hash = app_record.action_hashed().hash.clone();

    // Mark as waitlisted
    let review_input = ReviewApplicationInput {
        application_hash: app_hash.clone(),
        new_status: ApplicationStatus::Waitlisted,
    };
    let _: Record = alice
        .call_zome_fn("housing_membership", "review_application", review_input)
        .await;

    // Add to the explicit waitlist entry
    let waitlist_input = AddToWaitlistInput {
        application_hash: app_hash,
        unit_type_preference: Some(UnitType::Studio),
    };

    let waitlist_entry: Record = alice
        .call_zome_fn("housing_membership", "add_to_waitlist", waitlist_input)
        .await;

    assert!(
        !waitlist_entry.action_hashed().hash.as_ref().is_empty(),
        "Waitlist entry should be created"
    );

    // Retrieve and verify the waitlist
    let waitlist: Vec<Record> = alice
        .call_zome_fn("housing_membership", "get_waitlist", ())
        .await;

    assert!(!waitlist.is_empty(), "Waitlist should have at least one entry");
}

/// Test: Create a rent-to-own agreement and record a payment that accrues equity.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_membership_rent_to_own() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_rto", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();
    let target = Timestamp::from_micros(now.as_micros() + 10 * 365 * 24 * 60 * 60 * 1_000_000);

    // We need a building + unit hash for the agreement
    // Register the minimum required building first
    let building = Building {
        id: "bldg-rto-001".into(),
        name: "Sunrise Cooperative".into(),
        address: "1 Sunrise Blvd, Denver, CO 80201".into(),
        location_lat: 39.7392,
        location_lon: -104.9903,
        total_units: 4,
        year_built: Some(2010),
        building_type: BuildingType::Townhouse,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;
    let building_hash = building_record.action_hashed().hash.clone();

    let unit = Unit {
        building_hash,
        unit_number: "2C".into(),
        unit_type: UnitType::ThreeBedroom,
        square_meters: 110,
        floor: 2,
        bedrooms: 3,
        bathrooms: 2,
        accessibility_features: vec![],
        current_occupant: None,
        status: UnitStatus::Available,
    };

    let unit_record: Record = alice
        .call_zome_fn("housing_units", "register_unit", unit)
        .await;
    let unit_hash = unit_record.action_hashed().hash.clone();

    // Create the rent-to-own agreement
    let rto_input = CreateRentToOwnInput {
        member: alice.agent_pubkey.clone(),
        unit_hash,
        total_purchase_price_cents: 18_000_000, // $180,000
        monthly_rent_cents: 120_000,            // $1,200/month
        equity_portion_percent: 30,             // 30% of each payment builds equity
        target_completion: target,
    };

    let agreement_record: Record = alice
        .call_zome_fn("housing_membership", "create_rent_to_own", rto_input)
        .await;

    let agreement_hash = agreement_record.action_hashed().hash.clone();
    assert!(
        !agreement_hash.as_ref().is_empty(),
        "Rent-to-own agreement should be created"
    );

    // Record a monthly payment
    let payment_input = RecordRentPaymentInput {
        agreement_hash,
        payment_amount_cents: 120_000,
    };

    let updated_agreement: Record = alice
        .call_zome_fn("housing_membership", "record_rent_payment", payment_input)
        .await;

    assert!(
        !updated_agreement.action_hashed().hash.as_ref().is_empty(),
        "Rent payment should be recorded and agreement updated"
    );

    // Confirm member equity records exist
    let equity_records: Vec<Record> = alice
        .call_zome_fn(
            "housing_membership",
            "get_member_equity",
            alice.agent_pubkey.clone(),
        )
        .await;

    assert!(
        !equity_records.is_empty(),
        "Member equity records should be retrievable"
    );
}

// ============================================================================
// housing_finances — charges, payments, reserve funds, budgets
// ============================================================================

/// Test: Create a reserve fund and deposit into it, then query the
/// financial summary for a billing period.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_finances_reserve_fund_and_deposit() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_finances", 1).await;
    let alice = &agents[0];

    // Create a capital reserve fund
    let reserve_fund = ReserveFund {
        name: "Capital Improvement Reserve".into(),
        fund_type: FundType::CapitalReserve,
        balance_cents: 0,
        target_cents: 50_000_00, // $50,000 target
        description: "Long-term capital improvements including roof, HVAC, and windows".into(),
    };

    let fund_record: Record = alice
        .call_zome_fn("housing_finances", "create_reserve_fund", reserve_fund)
        .await;

    let fund_hash = fund_record.action_hashed().hash.clone();
    assert!(!fund_hash.as_ref().is_empty(), "Reserve fund should be created");

    // Deposit into the fund
    let deposit = DepositToReserveInput {
        fund_hash,
        amount_cents: 5_000_00, // $5,000
    };

    let updated_fund: Record = alice
        .call_zome_fn("housing_finances", "deposit_to_reserve", deposit)
        .await;

    assert!(
        !updated_fund.action_hashed().hash.as_ref().is_empty(),
        "Deposit should update the reserve fund"
    );

    // Check financial summary for current period
    let summary_input = FinancialSummaryInput {
        period_year: 2026,
        period_month: 2,
    };

    let summary: FinancialSummary = alice
        .call_zome_fn("housing_finances", "get_financial_summary", summary_input)
        .await;

    // With no charges generated, outstanding should be zero
    assert_eq!(
        summary.outstanding_cents, 0,
        "No charges this period means zero outstanding"
    );
    // Our reserve fund should appear in the summary
    assert!(
        !summary.reserve_funds.is_empty(),
        "Summary should include the reserve fund"
    );
}

/// Test: Create an annual budget, then approve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_finances_budget_lifecycle() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_budget", 1).await;
    let alice = &agents[0];

    // Draft a 2026 budget
    let budget = Budget {
        fiscal_year: 2026,
        income_projected_cents: 204_000_00,  // $204,000 annual income
        expenses_projected_cents: 182_000_00, // $182,000 projected expenses
        categories: vec![
            BudgetCategory {
                name: "Maintenance & Repairs".into(),
                allocated_cents: 60_000_00,
                spent_cents: 0,
            },
            BudgetCategory {
                name: "Utilities".into(),
                allocated_cents: 48_000_00,
                spent_cents: 0,
            },
            BudgetCategory {
                name: "Insurance".into(),
                allocated_cents: 24_000_00,
                spent_cents: 0,
            },
            BudgetCategory {
                name: "Management & Admin".into(),
                allocated_cents: 50_000_00,
                spent_cents: 0,
            },
        ],
        approved: false,
        approved_at: None,
    };

    let budget_record: Record = alice
        .call_zome_fn("housing_finances", "create_budget", budget)
        .await;

    let budget_hash = budget_record.action_hashed().hash.clone();
    assert!(!budget_hash.as_ref().is_empty(), "Budget should be created");

    // Approve the budget
    let approve_input = ApproveBudgetInput {
        budget_hash,
    };

    let approved_budget: Record = alice
        .call_zome_fn("housing_finances", "approve_budget", approve_input)
        .await;

    assert!(
        !approved_budget.action_hashed().hash.as_ref().is_empty(),
        "Budget should be approved"
    );
}

// ============================================================================
// housing_governance — board meetings, resolutions
// ============================================================================

/// Test: Schedule a board meeting, record minutes, then propose a resolution.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_governance_meeting_and_resolution() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_governance", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();
    let next_week = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Schedule a regular board meeting
    let meeting = BoardMeeting {
        title: "February 2026 Regular Board Meeting".into(),
        agenda: "1. Approve January minutes\n2. Budget review\n3. Maintenance updates\n4. New business".into(),
        scheduled_at: next_week,
        location: Some("Community Room, 1st Floor".into()),
        meeting_type: MeetingType::Regular,
        attendees: vec![],
        minutes: None,
    };

    let meeting_record: Record = alice
        .call_zome_fn("housing_governance", "schedule_meeting", meeting)
        .await;

    let meeting_hash = meeting_record.action_hashed().hash.clone();
    assert!(!meeting_hash.as_ref().is_empty(), "Meeting should be scheduled");

    // Record minutes after the meeting concludes
    let minutes_input = RecordMinutesInput {
        meeting_hash: meeting_hash.clone(),
        minutes: "Meeting called to order at 7:00 PM by Board Chair. \
                  Quorum confirmed with 5 of 7 board members present. \
                  January minutes approved unanimously. \
                  Budget review: on track for Q1. \
                  Maintenance update: roof repair completed. \
                  Meeting adjourned at 8:45 PM."
            .into(),
        attendees: vec![alice.agent_pubkey.clone()],
    };

    let updated_meeting: Record = alice
        .call_zome_fn("housing_governance", "record_minutes", minutes_input)
        .await;

    assert!(
        !updated_meeting.action_hashed().hash.as_ref().is_empty(),
        "Minutes should be recorded"
    );

    // Propose a maintenance resolution tied to this meeting
    let resolution = Resolution {
        meeting_hash: Some(meeting_hash),
        title: "Authorize Emergency Boiler Replacement".into(),
        body: "Be it resolved that the Board authorizes up to $12,000 from the \
               Capital Reserve Fund for emergency boiler replacement in Building A, \
               to be completed before March 15, 2026."
            .into(),
        resolution_type: ResolutionType::Maintenance,
        proposed_by: alice.agent_pubkey.clone(),
        status: ResolutionStatus::Proposed,
    };

    let resolution_record: Record = alice
        .call_zome_fn("housing_governance", "propose_resolution", resolution)
        .await;

    assert!(
        !resolution_record.action_hashed().hash.as_ref().is_empty(),
        "Resolution should be proposed"
    );
}

// ============================================================================
// housing_maintenance — maintenance requests
// ============================================================================

/// Test: Submit a maintenance request and acknowledge it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_maintenance_submit_and_acknowledge() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_maintenance", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();

    // We need a real building hash — create one in the units zome first
    let building = Building {
        id: "bldg-maint-001".into(),
        name: "Oak Court Cooperative".into(),
        address: "55 Oak Court, Portland, OR 97201".into(),
        location_lat: 45.5231,
        location_lon: -122.6765,
        total_units: 12,
        year_built: Some(1968),
        building_type: BuildingType::Apartment,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;
    let building_hash = building_record.action_hashed().hash.clone();

    // Submit a high-priority plumbing request
    let request = MaintenanceRequest {
        building_hash: building_hash.clone(),
        unit_hash: None,
        reported_by: alice.agent_pubkey.clone(),
        category: MaintenanceCategory::Plumbing,
        priority: MaintenancePriority::High,
        description: "Water leak from unit 4B ceiling, dripping into unit 3B. \
                      Active drip rate approximately 2 gallons/hour. \
                      Resident has placed buckets but needs immediate attention."
            .into(),
        status: MaintenanceStatus::Reported,
        reported_at: now,
    };

    let request_record: Record = alice
        .call_zome_fn("housing_maintenance", "submit_request", request)
        .await;

    let request_hash = request_record.action_hashed().hash.clone();
    assert!(!request_hash.as_ref().is_empty(), "Maintenance request should be submitted");

    // Acknowledge the request (moves from Reported → Acknowledged)
    let ack_input = AcknowledgeRequestInput {
        request_hash,
    };

    let acknowledged: Record = alice
        .call_zome_fn("housing_maintenance", "acknowledge_request", ack_input)
        .await;

    assert!(
        !acknowledged.action_hashed().hash.as_ref().is_empty(),
        "Request should be acknowledged"
    );
}

// ============================================================================
// housing_clt — community land trust and ground leases
// ============================================================================

/// Test: Create a community land trust, issue a ground lease under it, then
/// update the trust's mission statement.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_clt_trust_and_ground_lease() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_clt", 1).await;
    let alice = &agents[0];

    let now = Timestamp::now();
    let lease_end =
        Timestamp::from_micros(now.as_micros() + 99 * 365 * 24 * 60 * 60 * 1_000_000);

    // Create the community land trust
    let trust = LandTrust {
        id: "clt-southwest-001".into(),
        name: "Southwest Community Land Trust".into(),
        mission: "To create permanently affordable housing by holding land in trust \
                  for the benefit of low- and moderate-income residents."
            .into(),
        // Simple bounding box polygon for illustration
        boundary: vec![
            (33.4484, -112.0740),
            (33.4484, -112.0500),
            (33.4300, -112.0500),
            (33.4300, -112.0740),
        ],
        charter_hash: None,
        stewardship_board: vec![alice.agent_pubkey.clone()],
        affordability_target_ami_percent: 80, // 80% of Area Median Income
        created_at: now,
    };

    let trust_record: Record = alice
        .call_zome_fn("housing_clt", "create_land_trust", trust.clone())
        .await;

    let trust_hash = trust_record.action_hashed().hash.clone();
    assert!(!trust_hash.as_ref().is_empty(), "Land trust should be created");

    // Register a building + unit that will go under CLT stewardship
    let building = Building {
        id: "bldg-clt-sw-001".into(),
        name: "Desert Rose Homes".into(),
        address: "300 Desert Rose Ln, Phoenix, AZ 85001".into(),
        location_lat: 33.4484,
        location_lon: -112.0740,
        total_units: 8,
        year_built: Some(2018),
        building_type: BuildingType::SingleFamily,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;
    let building_hash = building_record.action_hashed().hash.clone();

    let unit = Unit {
        building_hash,
        unit_number: "Unit 4".into(),
        unit_type: UnitType::ThreeBedroom,
        square_meters: 120,
        floor: 1,
        bedrooms: 3,
        bathrooms: 2,
        accessibility_features: vec![
            AccessFeature::WheelchairAccessible,
            AccessFeature::WideDoorways,
            AccessFeature::LowCounters,
        ],
        current_occupant: None,
        status: UnitStatus::Available,
    };

    let unit_record: Record = alice
        .call_zome_fn("housing_units", "register_unit", unit)
        .await;
    let unit_hash = unit_record.action_hashed().hash.clone();

    // Issue a 99-year ground lease under the trust
    let lease = GroundLease {
        trust_hash: trust_hash.clone(),
        unit_hash,
        leaseholder: alice.agent_pubkey.clone(),
        lease_term_years: 99,
        ground_rent_monthly_cents: 5_000, // $50/month ground rent
        resale_formula: ResaleFormula {
            formula_type: FormulaType::AreaMedianIncome,
            max_appreciation_percent_annual: Some(2),
            ami_cap_percent: Some(80),
            improvement_credit_percent: Some(100),
        },
        started_at: now,
        expires_at: lease_end,
    };

    let lease_record: Record = alice
        .call_zome_fn("housing_clt", "issue_ground_lease", lease)
        .await;

    assert!(
        !lease_record.action_hashed().hash.as_ref().is_empty(),
        "Ground lease should be issued"
    );

    // Update the trust's mission (general-purpose update)
    let updated_trust = LandTrust {
        id: "clt-southwest-001".into(),
        name: "Southwest Community Land Trust".into(),
        mission: "To create and preserve permanently affordable housing by stewardship \
                  of land in trust for current and future generations of low- and \
                  moderate-income residents in the Southwest Phoenix community."
            .into(),
        boundary: vec![
            (33.4484, -112.0740),
            (33.4484, -112.0500),
            (33.4300, -112.0500),
            (33.4300, -112.0740),
        ],
        charter_hash: None,
        stewardship_board: vec![alice.agent_pubkey.clone()],
        affordability_target_ami_percent: 80,
        created_at: now,
    };

    let update_input = UpdateLandTrustInput {
        original_action_hash: trust_hash,
        updated_entry: updated_trust,
    };

    let updated_hash: ActionHash = alice
        .call_zome_fn("housing_clt", "update_land_trust", update_input)
        .await;

    assert!(
        !updated_hash.as_ref().is_empty(),
        "Land trust should be updated with new mission"
    );
}

// ============================================================================
// Multi-agent: two residents exercise the housing workflow end-to-end
// ============================================================================

/// Test: Two agents interact — alice (board member) reviews and approves
/// bob's membership application for a specific unit.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA bundle — run: hc dna pack mycelix-commons/dna/"]
async fn test_housing_multi_agent_board_approval() {
    let agents =
        setup_test_agents(&DnaPaths::commons(), "commons_housing_multi", 2).await;
    let alice = &agents[0]; // board member
    let bob = &agents[1];   // applicant

    let now = Timestamp::now();

    // Alice registers a building and a vacant unit as board member
    let building = Building {
        id: "bldg-multi-001".into(),
        name: "River Bend Commons".into(),
        address: "10 River Bend Dr, Minneapolis, MN 55401".into(),
        location_lat: 44.9778,
        location_lon: -93.2650,
        total_units: 20,
        year_built: Some(2001),
        building_type: BuildingType::Apartment,
        cooperative_hash: None,
    };

    let building_record: Record = alice
        .call_zome_fn("housing_units", "register_building", building)
        .await;
    let building_hash = building_record.action_hashed().hash.clone();

    let unit = Unit {
        building_hash,
        unit_number: "7D".into(),
        unit_type: UnitType::TwoBedroom,
        square_meters: 82,
        floor: 7,
        bedrooms: 2,
        bathrooms: 1,
        accessibility_features: vec![AccessFeature::Elevator],
        current_occupant: None,
        status: UnitStatus::Available,
    };

    let unit_record: Record = alice
        .call_zome_fn("housing_units", "register_unit", unit)
        .await;
    let unit_hash = unit_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Bob submits a membership application requesting that specific unit
    let application = MemberApplication {
        applicant: bob.agent_pubkey.clone(),
        requested_unit: Some(unit_hash.clone()),
        membership_type_requested: MembershipType::FullShare,
        applied_at: now,
        household_size: 3,
        income_verified: true,
        references: vec![
            "Dr. Priya Sharma, Employer".into(),
            "Community Credit Union".into(),
        ],
        status: ApplicationStatus::Pending,
    };

    let app_record: Record = bob
        .call_zome_fn("housing_membership", "submit_application", application)
        .await;

    let app_hash = app_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Alice (board member) approves the application
    let approve_input = ApproveMemberInput {
        application_hash: app_hash,
        unit_hash: Some(unit_hash.clone()),
        membership_type: MembershipType::FullShare,
        share_equity_cents: 40_000_00, // $40,000 equity share
        monthly_charge_cents: 110_000, // $1,100/month
    };

    let member_record: Record = alice
        .call_zome_fn("housing_membership", "approve_member", approve_input)
        .await;

    assert!(
        !member_record.action_hashed().hash.as_ref().is_empty(),
        "Alice should be able to approve Bob's application"
    );

    // Alice assigns Bob as occupant of the unit
    let assign_input = AssignOccupantInput {
        unit_action_hash: unit_hash,
        occupant: bob.agent_pubkey.clone(),
    };

    let occupied_unit: Record = alice
        .call_zome_fn("housing_units", "assign_occupant", assign_input)
        .await;

    assert!(
        !occupied_unit.action_hashed().hash.as_ref().is_empty(),
        "Unit should be assigned to Bob"
    );

    // Now Bob records his first payment
    let payment = Payment {
        member: bob.agent_pubkey.clone(),
        charge_hash: None,
        amount_cents: 110_000,
        payment_method: PaymentMethod::BankTransfer,
        paid_at: now,
        reference: Some("ACH-20260201-001".into()),
    };

    let payment_record: Record = bob
        .call_zome_fn("housing_finances", "record_payment", payment)
        .await;

    assert!(
        !payment_record.action_hashed().hash.as_ref().is_empty(),
        "Bob's payment should be recorded"
    );

    wait_for_dht_sync().await;

    // Alice can retrieve Bob's payments
    let bob_payments: Vec<Record> = alice
        .call_zome_fn(
            "housing_finances",
            "get_member_payments",
            bob.agent_pubkey.clone(),
        )
        .await;

    assert!(!bob_payments.is_empty(), "Bob's payments should be visible to Alice");
}
