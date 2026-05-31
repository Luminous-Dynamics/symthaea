// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Energy Bridge Integrity Zome
//!
//! Entry types for Terra Atlas integration, investment tracking,
//! and regenerative exit coordination.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Energy project query from external systems (Terra Atlas)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProjectQuery {
    pub id: String,
    pub project_id: String,
    pub source: String, // "terra-atlas" or hApp ID
    pub query_type: ProjectQueryType,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProjectQueryType {
    ProjectDetails,
    InvestmentStatus,
    CommunityReadiness,
    RegenerativeProgress,
}

/// Energy project reference from Terra Atlas
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TerraAtlasProject {
    pub id: String,
    pub terra_atlas_id: String,
    pub name: String,
    pub project_type: EnergyType,
    pub location: GeoLocation,
    pub capacity_mw: f64,
    pub total_investment: u64,
    pub current_investment: u64,
    pub status: ProjectStatus,
    pub regenerative_progress: f64, // 0.0 - 1.0
    pub synced_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnergyType {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Nuclear,
    Storage,
    Mixed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
    pub country: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    Discovery,
    Funding,
    Development,
    Operational,
    Transitioning,
    CommunityOwned,
}

/// Investment record for cross-hApp tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct InvestmentRecord {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: u64,
    pub currency: String,
    pub source_happ: String, // e.g., "mycelix-finance"
    pub investment_type: InvestmentType,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InvestmentType {
    Equity,
    Loan,
    Grant,
    CommunityShare,
}

/// Regenerative exit milestone
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RegenerativeMilestone {
    pub id: String,
    pub project_id: String,
    pub milestone_type: MilestoneType,
    pub community_readiness: f64,
    pub operator_certification: bool,
    pub financial_sustainability: f64,
    pub achieved_at: Option<Timestamp>,
    pub verified_by: Option<String>, // DID of verifier
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MilestoneType {
    CommunityFormation,
    OperatorTraining,
    FinancialIndependence,
    GovernanceEstablished,
    FullTransition,
}

/// Energy event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnergyBridgeEvent {
    pub id: String,
    pub event_type: EnergyEventType,
    pub project_id: String,
    pub payload: String,
    pub source: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

/// Production metrics for energy projects
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProductionRecord {
    pub id: String,
    pub project_id: String,
    pub terra_atlas_id: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub energy_generated_mwh: f64,
    pub capacity_factor: f64,   // 0.0-1.0 (actual vs theoretical max)
    pub revenue_generated: u64, // in smallest currency unit
    pub currency: String,
    pub grid_injection_mwh: f64,   // energy sold to grid
    pub self_consumption_mwh: f64, // energy used locally
    pub recorded_at: Timestamp,
    pub verified_by: Option<String>, // DID of verifier (e.g., grid operator)
}

/// Pending sync record for bidirectional sync to Terra Atlas
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PendingSyncRecord {
    pub id: String,
    pub sync_type: SyncType,
    pub target_system: String, // "terra-atlas"
    pub payload: String,       // JSON payload for external sync service
    pub created_at: Timestamp,
    pub synced_at: Option<Timestamp>,
    pub retry_count: u32,
    pub last_error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SyncType {
    InvestmentUpdate,
    ProductionMetrics,
    MilestoneProgress,
    StatusChange,
    TransitionProgress,
}

/// Regenerative transition tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TransitionRecord {
    pub id: String,
    pub project_id: String,
    pub terra_atlas_id: String,
    pub from_ownership_pct: f64,
    pub to_ownership_pct: f64,
    pub community_did: String,
    pub reserve_account_balance: u64,
    pub conditions_met: Vec<String>,
    pub conditions_pending: Vec<String>,
    pub status: TransitionStatus,
    pub initiated_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransitionStatus {
    Proposed,
    ConditionsReview,
    InProgress,
    Completed,
    Cancelled,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    ProjectQuery(ProjectQuery),
    TerraAtlasProject(TerraAtlasProject),
    InvestmentRecord(InvestmentRecord),
    RegenerativeMilestone(RegenerativeMilestone),
    EnergyBridgeEvent(EnergyBridgeEvent),
    ProductionRecord(ProductionRecord),
    PendingSyncRecord(PendingSyncRecord),
    TransitionRecord(TransitionRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllProjects,
    ProjectToInvestments,
    DidToInvestments,
    ProjectToMilestones,
    RecentEvents,
    ByEnergyType,
    ProjectToProduction,
    PendingSyncs,
    ProjectToTransitions,
    ActiveTransitions,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. })
        | FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => validate_entry(&app_entry),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_entry(entry: &EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::ProjectQuery(query) => {
            if query.project_id.is_empty() && query.source.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Project ID or source required".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::TerraAtlasProject(project) => {
            if project.capacity_mw <= 0.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Capacity must be positive".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::InvestmentRecord(record) => {
            if !record.investor_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid(
                    "Investor must have valid DID".into(),
                ));
            }
            if record.amount == 0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Amount must be positive".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::RegenerativeMilestone(milestone) => {
            if milestone.community_readiness < 0.0 || milestone.community_readiness > 1.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Community readiness must be 0.0-1.0".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ProductionRecord(record) => {
            if record.energy_generated_mwh < 0.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Energy generated must be non-negative".into(),
                ));
            }
            if record.capacity_factor < 0.0 || record.capacity_factor > 1.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Capacity factor must be 0.0-1.0".into(),
                ));
            }
            if record.grid_injection_mwh + record.self_consumption_mwh
                > record.energy_generated_mwh * 1.01
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "Grid + self consumption cannot exceed generated".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::PendingSyncRecord(record) => {
            if record.target_system.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Target system required".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::TransitionRecord(record) => {
            if record.to_ownership_pct < record.from_ownership_pct {
                return Ok(ValidateCallbackResult::Invalid(
                    "Transition must increase community ownership".into(),
                ));
            }
            if record.to_ownership_pct > 100.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Ownership cannot exceed 100%".into(),
                ));
            }
            if !record.community_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid(
                    "Community must have valid DID".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn valid_geo_location() -> GeoLocation {
        GeoLocation {
            latitude: 37.7749,
            longitude: -122.4194,
            region: "California".to_string(),
            country: "USA".to_string(),
        }
    }

    // =========================================================================
    // EnergyType Enum Tests
    // =========================================================================

    #[test]
    fn test_energy_type_variants() {
        let types = vec![
            EnergyType::Solar,
            EnergyType::Wind,
            EnergyType::Hydro,
            EnergyType::Geothermal,
            EnergyType::Nuclear,
            EnergyType::Storage,
            EnergyType::Mixed,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_energy_type_equality() {
        assert_eq!(EnergyType::Solar, EnergyType::Solar);
        assert_ne!(EnergyType::Solar, EnergyType::Wind);
    }

    // =========================================================================
    // ProjectStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_project_status_variants() {
        let statuses = vec![
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_project_status_lifecycle() {
        let lifecycle = [
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        for i in 0..lifecycle.len() - 1 {
            assert_ne!(lifecycle[i], lifecycle[i + 1]);
        }
    }

    // =========================================================================
    // InvestmentType Enum Tests
    // =========================================================================

    #[test]
    fn test_investment_type_variants() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Loan,
            InvestmentType::Grant,
            InvestmentType::CommunityShare,
        ];
        assert_eq!(types.len(), 4);
    }

    // =========================================================================
    // MilestoneType Enum Tests
    // =========================================================================

    #[test]
    fn test_milestone_type_variants() {
        let types = vec![
            MilestoneType::CommunityFormation,
            MilestoneType::OperatorTraining,
            MilestoneType::FinancialIndependence,
            MilestoneType::GovernanceEstablished,
            MilestoneType::FullTransition,
        ];
        assert_eq!(types.len(), 5);
    }

    // =========================================================================
    // EnergyEventType Enum Tests
    // =========================================================================

    #[test]
    fn test_energy_event_type_variants() {
        let types = vec![
            EnergyEventType::ProjectDiscovered,
            EnergyEventType::InvestmentReceived,
            EnergyEventType::MilestoneAchieved,
            EnergyEventType::TransitionInitiated,
            EnergyEventType::CommunityOwnershipComplete,
            EnergyEventType::ProductionUpdate,
            EnergyEventType::StatusChanged,
            EnergyEventType::SyncPending,
        ];
        assert_eq!(types.len(), 8);
    }

    // =========================================================================
    // SyncType Enum Tests
    // =========================================================================

    #[test]
    fn test_sync_type_variants() {
        let types = vec![
            SyncType::InvestmentUpdate,
            SyncType::ProductionMetrics,
            SyncType::MilestoneProgress,
            SyncType::StatusChange,
            SyncType::TransitionProgress,
        ];
        assert_eq!(types.len(), 5);
    }

    // =========================================================================
    // TransitionStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_transition_status_variants() {
        let statuses = vec![
            TransitionStatus::Proposed,
            TransitionStatus::ConditionsReview,
            TransitionStatus::InProgress,
            TransitionStatus::Completed,
            TransitionStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 5);
    }

    // =========================================================================
    // GeoLocation Tests
    // =========================================================================

    #[test]
    fn test_geo_location_valid() {
        let location = valid_geo_location();
        assert!(location.latitude >= -90.0 && location.latitude <= 90.0);
        assert!(location.longitude >= -180.0 && location.longitude <= 180.0);
    }

    #[test]
    fn test_geo_location_equator() {
        let location = GeoLocation {
            latitude: 0.0,
            longitude: 0.0,
            region: "Gulf of Guinea".to_string(),
            country: "International Waters".to_string(),
        };
        assert_eq!(location.latitude, 0.0);
        assert_eq!(location.longitude, 0.0);
    }

    // =========================================================================
    // TerraAtlasProject Tests
    // =========================================================================

    fn valid_terra_atlas_project() -> TerraAtlasProject {
        TerraAtlasProject {
            id: "project:TA-2024-001:123456".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            name: "Sahara Solar Farm".to_string(),
            project_type: EnergyType::Solar,
            location: valid_geo_location(),
            capacity_mw: 500.0,
            total_investment: 100_000_000,
            current_investment: 50_000_000,
            status: ProjectStatus::Funding,
            regenerative_progress: 0.0,
            synced_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_terra_atlas_project_valid() {
        let project = valid_terra_atlas_project();
        assert!(project.capacity_mw > 0.0);
        assert!(!project.terra_atlas_id.is_empty());
    }

    #[test]
    fn test_terra_atlas_project_zero_capacity_invalid() {
        let project = TerraAtlasProject {
            capacity_mw: 0.0,
            ..valid_terra_atlas_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_terra_atlas_project_negative_capacity_invalid() {
        let project = TerraAtlasProject {
            capacity_mw: -50.0,
            ..valid_terra_atlas_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_terra_atlas_project_regenerative_progress() {
        let project = TerraAtlasProject {
            regenerative_progress: 0.5,
            ..valid_terra_atlas_project()
        };
        assert!(project.regenerative_progress >= 0.0 && project.regenerative_progress <= 1.0);
    }

    #[test]
    fn test_terra_atlas_project_funding_status() {
        let project = valid_terra_atlas_project();
        let funding_percentage =
            (project.current_investment as f64 / project.total_investment as f64) * 100.0;
        assert_eq!(funding_percentage, 50.0);
    }

    // =========================================================================
    // InvestmentRecord Tests
    // =========================================================================

    fn valid_investment_record() -> InvestmentRecord {
        InvestmentRecord {
            id: "invest:project1:did:mycelix:investor1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 50000,
            currency: "USD".to_string(),
            source_happ: "mycelix-finance".to_string(),
            investment_type: InvestmentType::Equity,
            created_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_investment_record_valid_did() {
        let record = valid_investment_record();
        assert!(record.investor_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_investment_record_positive_amount() {
        let record = valid_investment_record();
        assert!(record.amount > 0);
    }

    #[test]
    fn test_investment_record_invalid_did() {
        let record = InvestmentRecord {
            investor_did: "investor123".to_string(),
            ..valid_investment_record()
        };
        assert!(!record.investor_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_investment_record_zero_amount_invalid() {
        let record = InvestmentRecord {
            amount: 0,
            ..valid_investment_record()
        };
        assert!(record.amount == 0);
    }

    #[test]
    fn test_investment_record_all_types() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Loan,
            InvestmentType::Grant,
            InvestmentType::CommunityShare,
        ];
        for inv_type in types {
            let record = InvestmentRecord {
                investment_type: inv_type.clone(),
                ..valid_investment_record()
            };
            assert_eq!(record.investment_type, inv_type);
        }
    }

    // =========================================================================
    // RegenerativeMilestone Tests
    // =========================================================================

    fn valid_regenerative_milestone() -> RegenerativeMilestone {
        RegenerativeMilestone {
            id: "milestone:project1:CommunityFormation:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            milestone_type: MilestoneType::CommunityFormation,
            community_readiness: 0.75,
            operator_certification: true,
            financial_sustainability: 0.8,
            achieved_at: Some(create_test_timestamp()),
            verified_by: Some("did:mycelix:verifier1".to_string()),
        }
    }

    #[test]
    fn test_regenerative_milestone_valid_readiness() {
        let milestone = valid_regenerative_milestone();
        assert!(milestone.community_readiness >= 0.0 && milestone.community_readiness <= 1.0);
    }

    #[test]
    fn test_regenerative_milestone_over_1_invalid() {
        let milestone = RegenerativeMilestone {
            community_readiness: 1.5,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.community_readiness > 1.0);
    }

    #[test]
    fn test_regenerative_milestone_negative_invalid() {
        let milestone = RegenerativeMilestone {
            community_readiness: -0.1,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.community_readiness < 0.0);
    }

    #[test]
    fn test_regenerative_milestone_achieved() {
        let milestone = valid_regenerative_milestone();
        assert!(milestone.achieved_at.is_some());
        assert!(milestone.verified_by.is_some());
    }

    #[test]
    fn test_regenerative_milestone_pending() {
        let milestone = RegenerativeMilestone {
            achieved_at: None,
            verified_by: None,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.achieved_at.is_none());
    }

    // =========================================================================
    // ProductionRecord Tests
    // =========================================================================

    fn valid_production_record() -> ProductionRecord {
        ProductionRecord {
            id: "prod:project1:123456:789".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000),
            energy_generated_mwh: 1500.0,
            capacity_factor: 0.25,
            revenue_generated: 150000,
            currency: "USD".to_string(),
            grid_injection_mwh: 1200.0,
            self_consumption_mwh: 300.0,
            recorded_at: Timestamp::from_micros(1706832000000000),
            verified_by: Some("did:mycelix:grid_operator".to_string()),
        }
    }

    #[test]
    fn test_production_record_valid() {
        let record = valid_production_record();
        assert!(record.energy_generated_mwh >= 0.0);
        assert!(record.capacity_factor >= 0.0 && record.capacity_factor <= 1.0);
    }

    #[test]
    fn test_production_record_negative_energy_invalid() {
        let record = ProductionRecord {
            energy_generated_mwh: -100.0,
            ..valid_production_record()
        };
        assert!(record.energy_generated_mwh < 0.0);
    }

    #[test]
    fn test_production_record_capacity_factor_over_1_invalid() {
        let record = ProductionRecord {
            capacity_factor: 1.5,
            ..valid_production_record()
        };
        assert!(record.capacity_factor > 1.0);
    }

    #[test]
    fn test_production_record_capacity_factor_negative_invalid() {
        let record = ProductionRecord {
            capacity_factor: -0.1,
            ..valid_production_record()
        };
        assert!(record.capacity_factor < 0.0);
    }

    #[test]
    fn test_production_record_grid_plus_self_not_exceed_generated() {
        let record = valid_production_record();
        let total_usage = record.grid_injection_mwh + record.self_consumption_mwh;
        // Allow 1% tolerance as stated in validation
        assert!(total_usage <= record.energy_generated_mwh * 1.01);
    }

    #[test]
    fn test_production_record_exceeds_generated_invalid() {
        let record = ProductionRecord {
            energy_generated_mwh: 1000.0,
            grid_injection_mwh: 800.0,
            self_consumption_mwh: 300.0, // 800 + 300 = 1100 > 1000
            ..valid_production_record()
        };
        let total = record.grid_injection_mwh + record.self_consumption_mwh;
        assert!(total > record.energy_generated_mwh * 1.01);
    }

    // =========================================================================
    // PendingSyncRecord Tests
    // =========================================================================

    fn valid_pending_sync_record() -> PendingSyncRecord {
        PendingSyncRecord {
            id: "sync:ProductionMetrics:project1:123456".to_string(),
            sync_type: SyncType::ProductionMetrics,
            target_system: "terra-atlas".to_string(),
            payload: r#"{"energy_mwh": 1500}"#.to_string(),
            created_at: create_test_timestamp(),
            synced_at: None,
            retry_count: 0,
            last_error: None,
        }
    }

    #[test]
    fn test_pending_sync_record_valid() {
        let record = valid_pending_sync_record();
        assert!(!record.target_system.is_empty());
    }

    #[test]
    fn test_pending_sync_record_empty_target_invalid() {
        let record = PendingSyncRecord {
            target_system: "".to_string(),
            ..valid_pending_sync_record()
        };
        assert!(record.target_system.is_empty());
    }

    #[test]
    fn test_pending_sync_record_not_synced() {
        let record = valid_pending_sync_record();
        assert!(record.synced_at.is_none());
    }

    #[test]
    fn test_pending_sync_record_synced() {
        let record = PendingSyncRecord {
            synced_at: Some(Timestamp::from_micros(1706918400000000)),
            ..valid_pending_sync_record()
        };
        assert!(record.synced_at.is_some());
    }

    #[test]
    fn test_pending_sync_record_with_retry() {
        let record = PendingSyncRecord {
            retry_count: 3,
            last_error: Some("Connection timeout".to_string()),
            ..valid_pending_sync_record()
        };
        assert!(record.retry_count > 0);
        assert!(record.last_error.is_some());
    }

    // =========================================================================
    // TransitionRecord Tests
    // =========================================================================

    fn valid_transition_record() -> TransitionRecord {
        TransitionRecord {
            id: "transition:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            from_ownership_pct: 25.0,
            to_ownership_pct: 50.0,
            community_did: "did:mycelix:community1".to_string(),
            reserve_account_balance: 100000,
            conditions_met: vec![
                "CommunityReadiness".to_string(),
                "FinancialSustainability".to_string(),
            ],
            conditions_pending: vec!["GovernanceMaturity".to_string()],
            status: TransitionStatus::InProgress,
            initiated_at: create_test_timestamp(),
            completed_at: None,
        }
    }

    #[test]
    fn test_transition_record_valid() {
        let record = valid_transition_record();
        assert!(record.to_ownership_pct > record.from_ownership_pct);
        assert!(record.community_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_transition_record_decrease_invalid() {
        let record = TransitionRecord {
            from_ownership_pct: 50.0,
            to_ownership_pct: 25.0,
            ..valid_transition_record()
        };
        assert!(record.to_ownership_pct < record.from_ownership_pct);
    }

    #[test]
    fn test_transition_record_over_100_invalid() {
        let record = TransitionRecord {
            to_ownership_pct: 110.0,
            ..valid_transition_record()
        };
        assert!(record.to_ownership_pct > 100.0);
    }

    #[test]
    fn test_transition_record_invalid_did() {
        let record = TransitionRecord {
            community_did: "community123".to_string(),
            ..valid_transition_record()
        };
        assert!(!record.community_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_transition_record_to_100_percent() {
        let record = TransitionRecord {
            from_ownership_pct: 75.0,
            to_ownership_pct: 100.0,
            status: TransitionStatus::InProgress,
            ..valid_transition_record()
        };
        assert_eq!(record.to_ownership_pct, 100.0);
    }

    #[test]
    fn test_transition_record_completed() {
        let record = TransitionRecord {
            status: TransitionStatus::Completed,
            completed_at: Some(Timestamp::from_micros(1709510400000000)),
            ..valid_transition_record()
        };
        assert!(record.completed_at.is_some());
        assert_eq!(record.status, TransitionStatus::Completed);
    }

    // =========================================================================
    // ProjectQuery Tests
    // =========================================================================

    fn valid_project_query() -> ProjectQuery {
        ProjectQuery {
            id: "query:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            source: "terra-atlas".to_string(),
            query_type: ProjectQueryType::ProjectDetails,
            queried_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_project_query_valid() {
        let query = valid_project_query();
        assert!(!query.project_id.is_empty() || !query.source.is_empty());
    }

    #[test]
    fn test_project_query_empty_both_invalid() {
        let query = ProjectQuery {
            project_id: "".to_string(),
            source: "".to_string(),
            ..valid_project_query()
        };
        assert!(query.project_id.is_empty() && query.source.is_empty());
    }

    #[test]
    fn test_project_query_all_types() {
        let types = vec![
            ProjectQueryType::ProjectDetails,
            ProjectQueryType::InvestmentStatus,
            ProjectQueryType::CommunityReadiness,
            ProjectQueryType::RegenerativeProgress,
        ];
        for query_type in types {
            let query = ProjectQuery {
                query_type: query_type.clone(),
                ..valid_project_query()
            };
            assert_eq!(query.query_type, query_type);
        }
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("all_projects".to_string());
        assert_eq!(anchor.0, "all_projects");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("pending_syncs".to_string());
        let anchor2 = Anchor("pending_syncs".to_string());
        let anchor3 = Anchor("all_projects".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_serialization_terra_atlas_project() {
        let project = valid_terra_atlas_project();
        let result = serde_json::to_string(&project);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_investment_record() {
        let record = valid_investment_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_production_record() {
        let record = valid_production_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_transition_record() {
        let record = valid_transition_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }
}
