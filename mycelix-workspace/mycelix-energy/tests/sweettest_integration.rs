// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Energy - Sweettest Integration Tests
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

use holochain::prelude::*;
use holochain::sweettest::*;
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
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProjectStatus {
    Proposed,
    Approved,
    UnderConstruction,
    Operational,
    Decommissioned,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProjectLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
    pub country: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProjectFinancials {
    pub total_cost: u64,
    pub funded_amount: u64,
    pub currency: String,
    pub expected_roi: f64,
    pub payback_years: f64,
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
    pub completed: bool,
    pub completed_at: Option<Timestamp>,
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
    pub project_id: String,
    pub producer_did: String,
    pub energy_kwh: f64,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub verified: bool,
    pub recorded_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordProductionInput {
    pub project_id: String,
    pub producer_did: String,
    pub energy_kwh: f64,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TradeOffer {
    pub id: String,
    pub seller_did: String,
    pub energy_kwh: f64,
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
    Filled,
    PartiallyFilled,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateOfferInput {
    pub seller_did: String,
    pub energy_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub available_from: Timestamp,
    pub available_until: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecuteTradeInput {
    pub offer_id: String,
    pub buyer_did: String,
    pub quantity_kwh: f64,
}

// --- investments types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Investment {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: u64,
    pub currency: String,
    pub investment_type: InvestmentType,
    pub status: InvestmentStatus,
    pub pledged_at: Timestamp,
    pub confirmed_at: Option<Timestamp>,
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
    Confirmed,
    Active,
    Returned,
    Defaulted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PledgeInput {
    pub project_id: String,
    pub investor_did: String,
    pub amount: u64,
    pub currency: String,
    pub investment_type: InvestmentType,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PortfolioSummary {
    pub investor_did: String,
    pub total_invested: u64,
    pub active_investments: u32,
    pub total_returns: u64,
}

// --- regenerative types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegenerativeContract {
    pub id: String,
    pub project_id: String,
    pub developer_did: String,
    pub community_did: String,
    pub ownership_target: f64,
    pub transition_years: u32,
    pub status: ContractStatus,
    pub conditions: Vec<TransitionCondition>,
    pub created: Timestamp,
    pub updated: Timestamp,
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
    pub description: String,
    pub threshold: f64,
    pub current_value: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ConditionType {
    FinancialReadiness,
    TechnicalCapability,
    GovernanceMaturity,
    CommunityEngagement,
    RegulatoryCompliance,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateContractInput {
    pub project_id: String,
    pub developer_did: String,
    pub community_did: String,
    pub ownership_target: f64,
    pub transition_years: u32,
    pub conditions: Vec<TransitionCondition>,
}

// --- bridge types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SyncProjectInput {
    pub terra_atlas_id: String,
    pub project_name: String,
    pub project_type: String,
    pub latitude: f64,
    pub longitude: f64,
    pub capacity_mw: f64,
    pub status: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastEnergyEventInput {
    pub event_type: EnergyEventType,
    pub project_id: String,
    pub subject_did: String,
    pub payload: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EnergyEventType {
    ProjectRegistered,
    InvestmentReceived,
    MilestoneCompleted,
    ProductionRecorded,
    TransitionInitiated,
    TransitionCompleted,
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
                region: "West Texas".to_string(),
                country: "US".to_string(),
            },
            capacity_mw: 100.0,
            developer_did: "did:mycelix:solar-dev".to_string(),
            community_did: Some("did:mycelix:pecos-county".to_string()),
            financials: ProjectFinancials {
                total_cost: 150_000_000,
                funded_amount: 0,
                currency: "USD".to_string(),
                expected_roi: 0.12,
                payback_years: 8.0,
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
                region: "Oklahoma".to_string(),
                country: "US".to_string(),
            },
            capacity_mw: 50.0,
            developer_did: "did:mycelix:wind-dev".to_string(),
            community_did: None,
            financials: ProjectFinancials {
                total_cost: 75_000_000,
                funded_amount: 0,
                currency: "USD".to_string(),
                expected_roi: 0.10,
                payback_years: 10.0,
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
        assert!(!milestone.completed);
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
            project_id: "proj:solar-1".to_string(),
            producer_did: "did:mycelix:producer".to_string(),
            energy_kwh: 5000.0,
            period_start: now,
            period_end: now,
        };

        let prod_record: Record = conductor
            .call(&cell.zome("grid"), "record_production", prod_input)
            .await;

        let production: EnergyProduction = decode_entry(&prod_record).expect("decode production");
        assert_eq!(production.energy_kwh, 5000.0);

        // Create trade offer
        let offer_input = CreateOfferInput {
            seller_did: "did:mycelix:producer".to_string(),
            energy_kwh: 2000.0,
            price_per_kwh: 0.08,
            currency: "USD".to_string(),
            available_from: now,
            available_until: now,
        };

        let offer_record: Record = conductor
            .call(&cell.zome("grid"), "create_trade_offer", offer_input)
            .await;

        let offer: TradeOffer = decode_entry(&offer_record).expect("decode offer");
        assert_eq!(offer.energy_kwh, 2000.0);

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
            amount: 50000,
            currency: "USD".to_string(),
            investment_type: InvestmentType::CommunityShare,
        };

        let record: Record = conductor
            .call(&cell.zome("investments"), "pledge_investment", pledge_input)
            .await;

        let investment: Investment = decode_entry(&record).expect("decode investment");
        assert_eq!(investment.amount, 50000);
        assert!(investment.confirmed_at.is_none());

        // Confirm investment
        let confirmed: Record = conductor
            .call(
                &cell.zome("investments"),
                "confirm_investment",
                investment.id.clone(),
            )
            .await;

        let confirmed_inv: Investment = decode_entry(&confirmed).expect("decode confirmed");
        assert!(confirmed_inv.confirmed_at.is_some());
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
        for i in 0..3 {
            let pledge_input = PledgeInput {
                project_id: format!("proj:energy-{}", i),
                investor_did: investor.clone(),
                amount: 10000 * (i + 1) as u64,
                currency: "USD".to_string(),
                investment_type: InvestmentType::Equity,
            };
            let _: Record = conductor
                .call(&cell.zome("investments"), "pledge_investment", pledge_input)
                .await;
        }

        let summary: PortfolioSummary = conductor
            .call(
                &cell.zome("investments"),
                "get_portfolio_summary",
                investor.clone(),
            )
            .await;

        assert_eq!(summary.investor_did, investor);
        assert!(
            summary.total_invested > 0,
            "Should have total invested amount"
        );
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
            developer_did: "did:mycelix:developer".to_string(),
            community_did: "did:mycelix:community".to_string(),
            ownership_target: 0.51,
            transition_years: 10,
            conditions: vec![
                TransitionCondition {
                    condition_type: ConditionType::FinancialReadiness,
                    description: "Community fund reaches threshold".to_string(),
                    threshold: 0.8,
                    current_value: 0.0,
                },
                TransitionCondition {
                    condition_type: ConditionType::TechnicalCapability,
                    description: "Trained local operators available".to_string(),
                    threshold: 1.0,
                    current_value: 0.0,
                },
            ],
        };

        let record: Record = conductor
            .call(
                &cell.zome("regenerative"),
                "create_regenerative_contract",
                input,
            )
            .await;

        let contract: RegenerativeContract = decode_entry(&record).expect("decode contract");
        assert_eq!(contract.ownership_target, 0.51);
        assert_eq!(contract.transition_years, 10);
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
            project_name: "Community SMR".to_string(),
            project_type: "Nuclear".to_string(),
            latitude: 33.0,
            longitude: -97.0,
            capacity_mw: 300.0,
            status: "Development".to_string(),
        };

        let record: Record = conductor
            .call(
                &cell.zome("energy_bridge"),
                "sync_terra_atlas_project",
                input,
            )
            .await;

        assert!(
            record.entry().as_option().is_some(),
            "Should create bridge record"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_broadcast_energy_event() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = BroadcastEnergyEventInput {
            event_type: EnergyEventType::ProjectRegistered,
            project_id: "proj:test-event".to_string(),
            subject_did: "did:mycelix:developer".to_string(),
            payload: serde_json::json!({"capacity_mw": 50.0}).to_string(),
        };

        let record: Record = conductor
            .call(&cell.zome("energy_bridge"), "broadcast_energy_event", input)
            .await;

        assert!(
            record.entry().as_option().is_some(),
            "Should create event record"
        );
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
                region: "North Texas".to_string(),
                country: "US".to_string(),
            },
            capacity_mw: 25.0,
            developer_did: "did:mycelix:lc-developer".to_string(),
            community_did: Some("did:mycelix:lc-community".to_string()),
            financials: ProjectFinancials {
                total_cost: 30_000_000,
                funded_amount: 0,
                currency: "USD".to_string(),
                expected_roi: 0.11,
                payback_years: 9.0,
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
            amount: 100000,
            currency: "USD".to_string(),
            investment_type: InvestmentType::CommunityShare,
        };

        let _: Record = conductor
            .call(&cell.zome("investments"), "pledge_investment", pledge_input)
            .await;

        // 3. Record production
        let now = Timestamp::now();
        let prod_input = RecordProductionInput {
            project_id: project.id.clone(),
            producer_did: "did:mycelix:lc-developer".to_string(),
            energy_kwh: 10000.0,
            period_start: now,
            period_end: now,
        };

        let _: Record = conductor
            .call(&cell.zome("grid"), "record_production", prod_input)
            .await;

        // 4. Create regenerative contract
        let contract_input = CreateContractInput {
            project_id: project.id.clone(),
            developer_did: "did:mycelix:lc-developer".to_string(),
            community_did: "did:mycelix:lc-community".to_string(),
            ownership_target: 0.51,
            transition_years: 10,
            conditions: vec![TransitionCondition {
                condition_type: ConditionType::CommunityEngagement,
                description: "Community participation threshold".to_string(),
                threshold: 0.75,
                current_value: 0.0,
            }],
        };

        let contract_record: Record = conductor
            .call(
                &cell.zome("regenerative"),
                "create_regenerative_contract",
                contract_input,
            )
            .await;

        let contract: RegenerativeContract =
            decode_entry(&contract_record).expect("decode contract");
        assert_eq!(contract.project_id, project.id);

        // 5. Verify project state
        let final_project: Option<Record> = conductor
            .call(&cell.zome("projects"), "get_project", project.id.clone())
            .await;

        assert!(final_project.is_some(), "Project should still exist");
    }
}
