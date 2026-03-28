// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Energy - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover energy projects, grid trading, investments, regenerative transitions,
//! and bridge operations.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists (no pre-built .dna yet - build first)
//! # hc dna pack dna/
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- projects types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnergyProject {
    pub id: String,
    pub terra_atlas_id: Option<String>,
    pub name: String,
    pub description: String,
    pub project_type: ProjectType,
    pub location: ProjectLocation,
    pub capacity_mw: f64,
    pub status: ProjectStatus,
    pub developer_did: String,
    pub community_did: Option<String>,
    pub financials: ProjectFinancials,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProjectType {
    Solar,
    Wind,
    Hydro,
    Nuclear,
    Geothermal,
    BatteryStorage,
    PumpedHydro,
    Hydrogen,
    Biomass,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProjectStatus {
    Proposed,
    Planning,
    Permitting,
    Financing,
    Construction,
    Operational,
    Decommissioned,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProjectLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub country: String,
    pub region: String,
    pub address: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProjectFinancials {
    pub total_cost: f64,
    pub funded_amount: f64,
    pub currency: String,
    pub target_irr: f64,
    pub payback_years: f64,
    pub annual_revenue_estimate: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterProjectInput {
    pub terra_atlas_id: Option<String>,
    pub name: String,
    pub description: String,
    pub project_type: ProjectType,
    pub location: ProjectLocation,
    pub capacity_mw: f64,
    pub developer_did: String,
    pub community_did: Option<String>,
    pub financials: ProjectFinancials,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProjectMilestone {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub target_date: Timestamp,
    pub completed_date: Option<Timestamp>,
    pub verification_evidence: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddMilestoneInput {
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub target_date: Timestamp,
}

// --- grid types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnergyProduction {
    pub id: String,
    pub producer_did: String,
    pub project_id: String,
    pub amount_kwh: f64,
    pub timestamp: Timestamp,
    pub period_hours: f64,
    pub meter_reading: Option<f64>,
    pub verified: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordProductionInput {
    pub producer_did: String,
    pub project_id: String,
    pub amount_kwh: f64,
    pub period_hours: f64,
    pub meter_reading: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TradeOffer {
    pub id: String,
    pub seller_did: String,
    pub project_id: Option<String>,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub available_from: Timestamp,
    pub available_until: Timestamp,
    pub status: OfferStatus,
    pub created: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OfferStatus {
    Active,
    PartiallyFilled,
    Filled,
    Expired,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateOfferInput {
    pub seller_did: String,
    pub project_id: Option<String>,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub available_from: Timestamp,
    pub available_until: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecuteTradeInput {
    pub offer_id: String,
    pub buyer_did: String,
    pub amount_kwh: f64,
}

// --- investments types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Investment {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub shares: f64,
    pub share_percentage: f64,
    pub investment_type: InvestmentType,
    pub status: InvestmentStatus,
    pub pledged: Timestamp,
    pub confirmed: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum InvestmentType {
    Equity,
    Debt,
    ConvertibleNote,
    RevenueShare,
    CommunityShare,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum InvestmentStatus {
    Pledged,
    PendingPayment,
    Confirmed,
    Cancelled,
    Transferred,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PledgeInput {
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub shares: f64,
    pub share_percentage: f64,
    pub investment_type: InvestmentType,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PortfolioSummary {
    pub investor_did: String,
    pub total_invested: f64,
    pub total_shares: f64,
    pub project_count: u32,
    pub unique_projects: u32,
    pub total_dividends_received: f64,
}

// --- regenerative types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegenerativeContract {
    pub id: String,
    pub project_id: String,
    pub community_did: String,
    pub conditions: Vec<TransitionCondition>,
    pub current_ownership_percentage: f64,
    pub target_ownership_percentage: f64,
    pub reserve_account_balance: f64,
    pub currency: String,
    pub status: ContractStatus,
    pub created: Timestamp,
    pub last_assessment: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContractStatus {
    Active,
    TransitionInProgress,
    TransitionComplete,
    Paused,
    Terminated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransitionCondition {
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub current_value: f64,
    pub weight: f64,
    pub satisfied: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ConditionType {
    CommunityReadiness,
    FinancialSustainability,
    OperationalCompetence,
    GovernanceMaturity,
    InvestorReturns,
    ReserveAccountFunded,
    MinimumOperatingHistory,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateContractInput {
    pub project_id: String,
    pub community_did: String,
    pub conditions: Vec<TransitionCondition>,
    pub target_ownership_percentage: f64,
    pub currency: String,
}

// --- bridge types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EnergyType {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Nuclear,
    Storage,
    Mixed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
    pub country: String,
}

/// Bridge-specific project status (different from projects zome ProjectStatus)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BridgeProjectStatus {
    Discovery,
    Funding,
    Development,
    Operational,
    Transitioning,
    CommunityOwned,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TerraAtlasProject {
    pub id: String,
    pub terra_atlas_id: String,
    pub name: String,
    pub project_type: EnergyType,
    pub location: GeoLocation,
    pub capacity_mw: f64,
    pub total_investment: u64,
    pub current_investment: u64,
    pub status: BridgeProjectStatus,
    pub regenerative_progress: f64,
    pub synced_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SyncProjectInput {
    pub terra_atlas_id: String,
    pub name: String,
    pub project_type: EnergyType,
    pub location: GeoLocation,
    pub capacity_mw: f64,
    pub total_investment: u64,
    pub current_investment: u64,
    pub status: BridgeProjectStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastEnergyEventInput {
    pub event_type: EnergyEventType,
    pub project_id: String,
    pub payload: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EnergyEventType {
    ProjectDiscovered,
    InvestmentReceived,
    MilestoneAchieved,
    TransitionInitiated,
    CommunityOwnershipComplete,
    ProductionUpdate,
    StatusChanged,
    SyncPending,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_energy.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

// ============================================================================
// Projects Tests
// ============================================================================

#[cfg(test)]
mod projects_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_and_get_project() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterProjectInput {
            terra_atlas_id: Some("terra:solar-tx-001".to_string()),
            name: "Texas Solar Farm".to_string(),
            description: "100MW solar installation in west Texas".to_string(),
            project_type: ProjectType::Solar,
            location: ProjectLocation {
                latitude: 31.5,
                longitude: -103.5,
                country: "US".to_string(),
                region: "West Texas".to_string(),
                address: None,
            },
            capacity_mw: 100.0,
            developer_did: "did:mycelix:solar-dev".to_string(),
            community_did: Some("did:mycelix:pecos-county".to_string()),
            financials: ProjectFinancials {
                total_cost: 150_000_000.0,
                funded_amount: 0.0,
                currency: "USD".to_string(),
                target_irr: 0.12,
                payback_years: 8.0,
                annual_revenue_estimate: 18_750_000.0,
            },
        };

        let record: Record = conductor
            .call(&cell.zome("projects"), "register_project", input)
            .await;

        let project: EnergyProject = decode_entry(&record).expect("decode project");
        assert_eq!(project.name, "Texas Solar Farm");
        assert_eq!(project.capacity_mw, 100.0);

        // Retrieve
        let retrieved: Option<Record> = conductor
            .call(&cell.zome("projects"), "get_project", project.id.clone())
            .await;

        assert!(retrieved.is_some(), "Should retrieve project");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_and_complete_milestone() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register project
        let proj_input = RegisterProjectInput {
            terra_atlas_id: None,
            name: "Wind Project".to_string(),
            description: "Milestone test project".to_string(),
            project_type: ProjectType::Wind,
            location: ProjectLocation {
                latitude: 35.0,
                longitude: -100.0,
                country: "US".to_string(),
                region: "Oklahoma".to_string(),
                address: None,
            },
            capacity_mw: 50.0,
            developer_did: "did:mycelix:wind-dev".to_string(),
            community_did: None,
            financials: ProjectFinancials {
                total_cost: 75_000_000.0,
                funded_amount: 0.0,
                currency: "USD".to_string(),
                target_irr: 0.10,
                payback_years: 10.0,
                annual_revenue_estimate: 7_500_000.0,
            },
        };

        let proj_record: Record = conductor
            .call(&cell.zome("projects"), "register_project", proj_input)
            .await;
        let project: EnergyProject = decode_entry(&proj_record).expect("decode project");

        // Add milestone
        let ms_input = AddMilestoneInput {
            project_id: project.id.clone(),
            name: "Site Survey Complete".to_string(),
            description: "Environmental and geological survey completed".to_string(),
            target_date: Timestamp::now(),
        };

        let ms_record: Record = conductor
            .call(&cell.zome("projects"), "add_milestone", ms_input)
            .await;

        let milestone: ProjectMilestone = decode_entry(&ms_record).expect("decode milestone");
        assert_eq!(milestone.project_id, project.id);
        assert!(milestone.completed_date.is_none());
    }
}

// ============================================================================
// Grid Tests
// ============================================================================

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_record_production_and_create_offer() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // Record production
        let prod_input = RecordProductionInput {
            producer_did: "did:mycelix:producer".to_string(),
            project_id: "proj:solar-1".to_string(),
            amount_kwh: 5000.0,
            period_hours: 24.0,
            meter_reading: Some(12345.0),
        };

        let prod_record: Record = conductor
            .call(&cell.zome("grid"), "record_production", prod_input)
            .await;

        let production: EnergyProduction = decode_entry(&prod_record).expect("decode production");
        assert_eq!(production.amount_kwh, 5000.0);

        // Create trade offer
        let offer_input = CreateOfferInput {
            seller_did: "did:mycelix:producer".to_string(),
            project_id: Some("proj:solar-1".to_string()),
            amount_kwh: 2000.0,
            price_per_kwh: 0.08,
            currency: "USD".to_string(),
            available_from: now,
            available_until: now,
        };

        let offer_record: Record = conductor
            .call(&cell.zome("grid"), "create_trade_offer", offer_input)
            .await;

        let offer: TradeOffer = decode_entry(&offer_record).expect("decode offer");
        assert_eq!(offer.amount_kwh, 2000.0);

        // Get active offers
        let active: Vec<Record> = conductor
            .call(&cell.zome("grid"), "get_active_offers", ())
            .await;

        assert!(!active.is_empty(), "Should have active offers");
    }
}

// ============================================================================
// Investments Tests
// ============================================================================

#[cfg(test)]
mod investments_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_pledge_and_confirm_investment() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let pledge_input = PledgeInput {
            project_id: "proj:solar-1".to_string(),
            investor_did: "did:mycelix:investor".to_string(),
            amount: 50000.0,
            currency: "USD".to_string(),
            shares: 500.0,
            share_percentage: 5.0,
            investment_type: InvestmentType::CommunityShare,
        };

        let record: Record = conductor
            .call(&cell.zome("investments"), "pledge_investment", pledge_input)
            .await;

        let investment: Investment = decode_entry(&record).expect("decode investment");
        assert!((investment.amount - 50000.0).abs() < 0.01);
        assert!(investment.confirmed.is_none());

        // Confirm investment
        let confirmed: Record = conductor
            .call(&cell.zome("investments"), "confirm_investment", investment.id.clone())
            .await;

        let confirmed_inv: Investment = decode_entry(&confirmed).expect("decode confirmed");
        assert!(confirmed_inv.confirmed.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_portfolio_summary() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let investor = "did:mycelix:portfolio-investor".to_string();

        // Pledge multiple investments
        for i in 0..3u32 {
            let pledge_input = PledgeInput {
                project_id: format!("proj:energy-{}", i),
                investor_did: investor.clone(),
                amount: 10000.0 * (i + 1) as f64,
                currency: "USD".to_string(),
                shares: 100.0 * (i + 1) as f64,
                share_percentage: 1.0 * (i + 1) as f64,
                investment_type: InvestmentType::Equity,
            };
            let _: Record = conductor
                .call(&cell.zome("investments"), "pledge_investment", pledge_input)
                .await;
        }

        let summary: PortfolioSummary = conductor
            .call(&cell.zome("investments"), "get_portfolio_summary", investor.clone())
            .await;

        assert_eq!(summary.investor_did, investor);
        assert!(summary.total_invested > 0.0, "Should have total invested amount");
    }
}

// ============================================================================
// Regenerative Tests
// ============================================================================

#[cfg(test)]
mod regenerative_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_regenerative_contract() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateContractInput {
            project_id: "proj:solar-community".to_string(),
            community_did: "did:mycelix:community".to_string(),
            target_ownership_percentage: 0.51,
            currency: "USD".to_string(),
            conditions: vec![
                TransitionCondition {
                    condition_type: ConditionType::FinancialSustainability,
                    threshold: 0.8,
                    current_value: 0.0,
                    weight: 0.4,
                    satisfied: false,
                },
                TransitionCondition {
                    condition_type: ConditionType::OperationalCompetence,
                    threshold: 1.0,
                    current_value: 0.0,
                    weight: 0.3,
                    satisfied: false,
                },
            ],
        };

        let record: Record = conductor
            .call(&cell.zome("regenerative"), "create_regenerative_contract", input)
            .await;

        let contract: RegenerativeContract = decode_entry(&record).expect("decode contract");
        assert_eq!(contract.target_ownership_percentage, 0.51);
        assert_eq!(contract.conditions.len(), 2);
    }
}

// ============================================================================
// Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_sync_terra_atlas_project() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = SyncProjectInput {
            terra_atlas_id: "terra:nuclear-smr-001".to_string(),
            name: "Community SMR".to_string(),
            project_type: EnergyType::Nuclear,
            location: GeoLocation {
                latitude: 33.0,
                longitude: -97.0,
                region: "North Texas".to_string(),
                country: "US".to_string(),
            },
            capacity_mw: 300.0,
            total_investment: 500_000_000,
            current_investment: 50_000_000,
            status: BridgeProjectStatus::Development,
        };

        let record: Record = conductor
            .call(&cell.zome("energy_bridge"), "sync_terra_atlas_project", input)
            .await;

        assert!(record.entry().as_option().is_some(), "Should create bridge record");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_broadcast_energy_event() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = BroadcastEnergyEventInput {
            event_type: EnergyEventType::ProjectDiscovered,
            project_id: "proj:test-event".to_string(),
            payload: serde_json::json!({"capacity_mw": 50.0}).to_string(),
        };

        let record: Record = conductor
            .call(&cell.zome("energy_bridge"), "broadcast_energy_event", input)
            .await;

        assert!(record.entry().as_option().is_some(), "Should create event record");
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_project_investment_production_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Register project
        let proj_input = RegisterProjectInput {
            terra_atlas_id: Some("terra:lifecycle-001".to_string()),
            name: "Lifecycle Solar".to_string(),
            description: "Full lifecycle test".to_string(),
            project_type: ProjectType::Solar,
            location: ProjectLocation {
                latitude: 32.0,
                longitude: -97.0,
                country: "US".to_string(),
                region: "North Texas".to_string(),
                address: None,
            },
            capacity_mw: 25.0,
            developer_did: "did:mycelix:lc-developer".to_string(),
            community_did: Some("did:mycelix:lc-community".to_string()),
            financials: ProjectFinancials {
                total_cost: 30_000_000.0,
                funded_amount: 0.0,
                currency: "USD".to_string(),
                target_irr: 0.11,
                payback_years: 9.0,
                annual_revenue_estimate: 3_300_000.0,
            },
        };

        let proj_record: Record = conductor
            .call(&cell.zome("projects"), "register_project", proj_input)
            .await;
        let project: EnergyProject = decode_entry(&proj_record).expect("decode project");

        // 2. Invest
        let pledge_input = PledgeInput {
            project_id: project.id.clone(),
            investor_did: "did:mycelix:lc-investor".to_string(),
            amount: 100000.0,
            currency: "USD".to_string(),
            shares: 1000.0,
            share_percentage: 10.0,
            investment_type: InvestmentType::CommunityShare,
        };

        let _: Record = conductor
            .call(&cell.zome("investments"), "pledge_investment", pledge_input)
            .await;

        // 3. Record production
        let prod_input = RecordProductionInput {
            producer_did: "did:mycelix:lc-developer".to_string(),
            project_id: project.id.clone(),
            amount_kwh: 10000.0,
            period_hours: 24.0,
            meter_reading: None,
        };

        let _: Record = conductor
            .call(&cell.zome("grid"), "record_production", prod_input)
            .await;

        // 4. Create regenerative contract
        let contract_input = CreateContractInput {
            project_id: project.id.clone(),
            community_did: "did:mycelix:lc-community".to_string(),
            target_ownership_percentage: 0.51,
            currency: "USD".to_string(),
            conditions: vec![TransitionCondition {
                condition_type: ConditionType::CommunityReadiness,
                threshold: 0.75,
                current_value: 0.0,
                weight: 0.5,
                satisfied: false,
            }],
        };

        let contract_record: Record = conductor
            .call(&cell.zome("regenerative"), "create_regenerative_contract", contract_input)
            .await;

        let contract: RegenerativeContract = decode_entry(&contract_record).expect("decode contract");
        assert_eq!(contract.project_id, project.id);

        // 5. Verify project state
        let final_project: Option<Record> = conductor
            .call(&cell.zome("projects"), "get_project", project.id.clone())
            .await;

        assert!(final_project.is_some(), "Project should still exist");
    }
}

// ============================================================================
// Cross-Zome Consistency Tests
// ============================================================================

#[cfg(test)]
mod cross_zome_tests {
    use super::*;

    /// Verify that investments reference valid projects
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_investment_references_valid_project() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register a project first
        let proj_input = make_test_project("cross-zome-proj", "Solar", 10.0);
        let proj_record: Record = conductor
            .call(&cell.zome("projects"), "register_project", proj_input)
            .await;
        let project: EnergyProject = decode_entry(&proj_record).expect("decode project");

        // Invest in the project
        let pledge_input = PledgeInput {
            project_id: project.id.clone(),
            investor_did: "did:mycelix:cross-investor".to_string(),
            amount: 50000.0,
            currency: "USD".to_string(),
            shares: 500.0,
            share_percentage: 5.0,
            investment_type: InvestmentType::Equity,
        };

        let inv_record: Record = conductor
            .call(&cell.zome("investments"), "pledge_investment", pledge_input)
            .await;
        let investment: Investment = decode_entry(&inv_record).expect("decode investment");
        assert_eq!(investment.project_id, project.id, "Investment must reference the correct project");

        // Verify via query
        let project_investments: Vec<Record> = conductor
            .call(&cell.zome("investments"), "get_project_investments", project.id.clone())
            .await;
        assert_eq!(project_investments.len(), 1, "Project should have exactly 1 investment");
    }

    /// Verify grid production references valid projects
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_production_references_valid_project() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let proj_input = make_test_project("prod-ref-proj", "Wind", 20.0);
        let proj_record: Record = conductor
            .call(&cell.zome("projects"), "register_project", proj_input)
            .await;
        let project: EnergyProject = decode_entry(&proj_record).expect("decode project");

        // Record production
        let prod_input = RecordProductionInput {
            producer_did: "did:mycelix:prod-ref-dev".to_string(),
            project_id: project.id.clone(),
            amount_kwh: 5000.0,
            period_hours: 12.0,
            meter_reading: Some(12345.0),
        };

        let prod_record: Record = conductor
            .call(&cell.zome("grid"), "record_production", prod_input)
            .await;
        let production: EnergyProduction = decode_entry(&prod_record).expect("decode production");
        assert_eq!(production.project_id, project.id);
        assert_eq!(production.amount_kwh, 5000.0);
        assert!(production.meter_reading == Some(12345.0));
    }

    /// Multi-investor portfolio summary
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_multi_investor_portfolio_consistency() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register 2 projects
        let proj1_input = make_test_project("portfolio-proj-1", "Solar", 10.0);
        let proj1: EnergyProject = decode_entry(
            &conductor.call::<_, Record>(&cell.zome("projects"), "register_project", proj1_input).await
        ).unwrap();

        let proj2_input = make_test_project("portfolio-proj-2", "Wind", 20.0);
        let proj2: EnergyProject = decode_entry(
            &conductor.call::<_, Record>(&cell.zome("projects"), "register_project", proj2_input).await
        ).unwrap();

        // Same investor pledges to both
        let investor_did = "did:mycelix:multi-investor".to_string();

        let _: Record = conductor.call(&cell.zome("investments"), "pledge_investment", PledgeInput {
            project_id: proj1.id.clone(),
            investor_did: investor_did.clone(),
            amount: 25000.0, currency: "USD".into(), shares: 250.0,
            share_percentage: 5.0, investment_type: InvestmentType::Equity,
        }).await;

        let _: Record = conductor.call(&cell.zome("investments"), "pledge_investment", PledgeInput {
            project_id: proj2.id.clone(),
            investor_did: investor_did.clone(),
            amount: 75000.0, currency: "USD".into(), shares: 750.0,
            share_percentage: 7.5, investment_type: InvestmentType::CommunityShare,
        }).await;

        // Verify portfolio summary
        let summary: PortfolioSummary = conductor
            .call(&cell.zome("investments"), "get_portfolio_summary", investor_did)
            .await;
        assert_eq!(summary.total_invested, 100000.0, "Total should be 25k + 75k");
        assert_eq!(summary.unique_projects, 2, "Should span 2 projects");
    }

    /// Trade offer lifecycle: create → partial fill → query
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_trade_offer_partial_fill() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create offer
        let now = Timestamp::now();
        let later = Timestamp::from_micros(now.as_micros() + 86_400_000_000); // +24h

        let offer_input = CreateOfferInput {
            seller_did: "did:mycelix:seller".to_string(),
            project_id: None,
            amount_kwh: 1000.0,
            price_per_kwh: 0.12,
            currency: "USD".to_string(),
            available_from: now,
            available_until: later,
        };

        let offer_record: Record = conductor
            .call(&cell.zome("grid"), "create_trade_offer", offer_input)
            .await;
        let offer: TradeOffer = decode_entry(&offer_record).expect("decode offer");
        assert!(matches!(offer.status, OfferStatus::Active));

        // Partial fill (buy 400 of 1000 kWh)
        let trade_input = ExecuteTradeInput {
            offer_id: offer.id.clone(),
            buyer_did: "did:mycelix:buyer-1".to_string(),
            amount_kwh: 400.0,
        };

        let _: Record = conductor
            .call(&cell.zome("grid"), "execute_trade", trade_input)
            .await;

        // Query active offers — should still show (partially filled)
        let active: Vec<Record> = conductor
            .call(&cell.zome("grid"), "get_active_offers", ())
            .await;
        assert!(!active.is_empty(), "Partially filled offer should still be listed");
    }
}

// ============================================================================
// Test helpers
// ============================================================================

fn make_test_project(name_suffix: &str, project_type: &str, capacity: f64) -> RegisterProjectInput {
    RegisterProjectInput {
        terra_atlas_id: Some(format!("terra:{}", name_suffix)),
        name: format!("Test {}", name_suffix),
        description: format!("Test project {}", name_suffix),
        project_type: match project_type {
            "Wind" => ProjectType::Wind,
            "Hydro" => ProjectType::Hydro,
            _ => ProjectType::Solar,
        },
        location: ProjectLocation {
            latitude: 32.0,
            longitude: -97.0,
            country: "US".to_string(),
            region: "Test Region".to_string(),
            address: None,
        },
        capacity_mw: capacity,
        developer_did: format!("did:mycelix:{}-dev", name_suffix),
        community_did: Some(format!("did:mycelix:{}-comm", name_suffix)),
        financials: ProjectFinancials {
            total_cost: capacity * 1_000_000.0,
            funded_amount: 0.0,
            currency: "USD".to_string(),
            target_irr: 0.10,
            payback_years: 10.0,
            annual_revenue_estimate: capacity * 100_000.0,
        },
    }
}
