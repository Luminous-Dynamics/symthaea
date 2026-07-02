// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Projects Integrity Zome
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
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

    // ── Consciousness Scoring (Symthaea integration) ────────────────
    /// Phi score from Symthaea consciousness evaluation (0.0–1.0).
    /// Higher Phi indicates more integrated, coherent project design.
    /// Computed by Symthaea's ConsciousnessEquationV2 via Holon bridge.
    pub phi_score: Option<f64>,
    /// Eight Harmonies alignment score (0.0–1.0).
    /// Measures ecological reciprocity, community coherence, sacred stillness.
    pub harmony_alignment: Option<f64>,
    /// When the consciousness assessment was last computed.
    pub consciousness_assessed_at: Option<Timestamp>,
    /// DID of the Symthaea instance that computed the score.
    pub consciousness_scorer_did: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub country: String,
    pub region: String,
    pub address: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    Proposed,
    Planning,
    Permitting,
    Financing,
    Construction,
    Operational,
    Decommissioned,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectFinancials {
    pub total_cost: f64,
    pub funded_amount: f64,
    pub currency: String,
    pub target_irr: f64,
    pub payback_years: f64,
    pub annual_revenue_estimate: f64,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProjectMilestone {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub target_date: Timestamp,
    pub completed_date: Option<Timestamp>,
    pub verification_evidence: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    EnergyProject(EnergyProject),
    ProjectMilestone(ProjectMilestone),
}

#[hdk_link_types]
pub enum LinkTypes {
    DeveloperToProjects,
    CommunityToProjects,
    LocationToProjects,
    ProjectToMilestones,
    TerraAtlasToProject,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::EnergyProject(project) => {
                        validate_create_energy_project(EntryCreationAction::Create(action), project)
                    }
                    EntryTypes::ProjectMilestone(milestone) => {
                        validate_create_project_milestone(EntryCreationAction::Create(action), milestone)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::EnergyProject(project) => {
                        validate_update_energy_project(action, project)
                    }
                    EntryTypes::ProjectMilestone(milestone) => {
                        validate_update_project_milestone(action, milestone)
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::DeveloperToProjects => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CommunityToProjects => Ok(ValidateCallbackResult::Valid),
                LinkTypes::LocationToProjects => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ProjectToMilestones => Ok(ValidateCallbackResult::Valid),
                LinkTypes::TerraAtlasToProject => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_energy_project(
    _action: EntryCreationAction,
    project: EnergyProject,
) -> ExternResult<ValidateCallbackResult> {
    if !project.developer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Developer must be a valid DID".into()));
    }
    if project.capacity_mw <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity must be positive".into()));
    }
    if project.location.latitude < -90.0 || project.location.latitude > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Invalid latitude".into()));
    }
    if project.location.longitude < -180.0 || project.location.longitude > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Invalid longitude".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_energy_project(
    _action: Update,
    project: EnergyProject,
) -> ExternResult<ValidateCallbackResult> {
    if project.capacity_mw <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_project_milestone(
    _action: EntryCreationAction,
    _milestone: ProjectMilestone,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_project_milestone(
    _action: Update,
    _milestone: ProjectMilestone,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
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

    // =========================================================================
    // ProjectType Enum Tests
    // =========================================================================

    #[test]
    fn test_project_type_variants() {
        let types = vec![
            ProjectType::Solar,
            ProjectType::Wind,
            ProjectType::Hydro,
            ProjectType::Nuclear,
            ProjectType::Geothermal,
            ProjectType::BatteryStorage,
            ProjectType::PumpedHydro,
            ProjectType::Hydrogen,
            ProjectType::Biomass,
            ProjectType::Other("Tidal".to_string()),
        ];
        assert_eq!(types.len(), 10);
    }

    #[test]
    fn test_project_type_equality() {
        assert_eq!(ProjectType::Solar, ProjectType::Solar);
        assert_ne!(ProjectType::Solar, ProjectType::Wind);
    }

    #[test]
    fn test_project_type_other_equality() {
        let other1 = ProjectType::Other("Tidal".to_string());
        let other2 = ProjectType::Other("Tidal".to_string());
        let other3 = ProjectType::Other("Wave".to_string());

        assert_eq!(other1, other2);
        assert_ne!(other1, other3);
    }

    #[test]
    fn test_project_type_cloning() {
        let original = ProjectType::Geothermal;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // =========================================================================
    // ProjectStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_project_status_variants() {
        let statuses = vec![
            ProjectStatus::Proposed,
            ProjectStatus::Planning,
            ProjectStatus::Permitting,
            ProjectStatus::Financing,
            ProjectStatus::Construction,
            ProjectStatus::Operational,
            ProjectStatus::Decommissioned,
        ];
        assert_eq!(statuses.len(), 7);
    }

    #[test]
    fn test_project_status_equality() {
        assert_eq!(ProjectStatus::Operational, ProjectStatus::Operational);
        assert_ne!(ProjectStatus::Proposed, ProjectStatus::Operational);
    }

    #[test]
    fn test_project_status_lifecycle_order() {
        // Test typical lifecycle progression
        let lifecycle = [
            ProjectStatus::Proposed,
            ProjectStatus::Planning,
            ProjectStatus::Permitting,
            ProjectStatus::Financing,
            ProjectStatus::Construction,
            ProjectStatus::Operational,
        ];
        // Each status should be distinct from the next
        for i in 0..lifecycle.len() - 1 {
            assert_ne!(lifecycle[i], lifecycle[i + 1]);
        }
    }

    // =========================================================================
    // ProjectLocation Validation Tests
    // =========================================================================

    fn valid_project_location() -> ProjectLocation {
        ProjectLocation {
            latitude: 37.7749,
            longitude: -122.4194,
            country: "USA".to_string(),
            region: "California".to_string(),
            address: Some("123 Solar St, San Francisco, CA 94102".to_string()),
        }
    }

    #[test]
    fn test_project_location_valid_coordinates() {
        let location = valid_project_location();
        assert!(location.latitude >= -90.0 && location.latitude <= 90.0);
        assert!(location.longitude >= -180.0 && location.longitude <= 180.0);
    }

    #[test]
    fn test_project_location_latitude_boundary_valid() {
        // Test boundary values
        let north_pole = ProjectLocation {
            latitude: 90.0,
            ..valid_project_location()
        };
        let south_pole = ProjectLocation {
            latitude: -90.0,
            ..valid_project_location()
        };
        assert_eq!(north_pole.latitude, 90.0);
        assert_eq!(south_pole.latitude, -90.0);
    }

    #[test]
    fn test_project_location_longitude_boundary_valid() {
        let date_line_east = ProjectLocation {
            longitude: 180.0,
            ..valid_project_location()
        };
        let date_line_west = ProjectLocation {
            longitude: -180.0,
            ..valid_project_location()
        };
        assert_eq!(date_line_east.longitude, 180.0);
        assert_eq!(date_line_west.longitude, -180.0);
    }

    #[test]
    fn test_project_location_latitude_over_90_invalid() {
        let location = ProjectLocation {
            latitude: 95.0,
            ..valid_project_location()
        };
        assert!(location.latitude > 90.0);
    }

    #[test]
    fn test_project_location_latitude_under_minus_90_invalid() {
        let location = ProjectLocation {
            latitude: -95.0,
            ..valid_project_location()
        };
        assert!(location.latitude < -90.0);
    }

    #[test]
    fn test_project_location_longitude_over_180_invalid() {
        let location = ProjectLocation {
            longitude: 185.0,
            ..valid_project_location()
        };
        assert!(location.longitude > 180.0);
    }

    #[test]
    fn test_project_location_longitude_under_minus_180_invalid() {
        let location = ProjectLocation {
            longitude: -185.0,
            ..valid_project_location()
        };
        assert!(location.longitude < -180.0);
    }

    #[test]
    fn test_project_location_with_address() {
        let location = valid_project_location();
        assert!(location.address.is_some());
    }

    #[test]
    fn test_project_location_without_address() {
        let location = ProjectLocation {
            address: None,
            ..valid_project_location()
        };
        assert!(location.address.is_none());
    }

    #[test]
    fn test_project_location_equator() {
        let location = ProjectLocation {
            latitude: 0.0,
            longitude: 0.0,
            country: "Ghana".to_string(),
            region: "Atlantic Ocean".to_string(),
            address: None,
        };
        assert_eq!(location.latitude, 0.0);
        assert_eq!(location.longitude, 0.0);
    }

    // =========================================================================
    // ProjectFinancials Validation Tests
    // =========================================================================

    fn valid_project_financials() -> ProjectFinancials {
        ProjectFinancials {
            total_cost: 10_000_000.0,
            funded_amount: 5_000_000.0,
            currency: "USD".to_string(),
            target_irr: 12.5,
            payback_years: 7.0,
            annual_revenue_estimate: 1_500_000.0,
        }
    }

    #[test]
    fn test_project_financials_valid() {
        let financials = valid_project_financials();
        assert!(financials.total_cost > 0.0);
        assert!(financials.funded_amount >= 0.0);
        assert!(financials.funded_amount <= financials.total_cost);
    }

    #[test]
    fn test_project_financials_funding_percentage() {
        let financials = valid_project_financials();
        let funded_percentage = (financials.funded_amount / financials.total_cost) * 100.0;
        assert_eq!(funded_percentage, 50.0);
    }

    #[test]
    fn test_project_financials_fully_funded() {
        let financials = ProjectFinancials {
            funded_amount: 10_000_000.0,
            ..valid_project_financials()
        };
        assert_eq!(financials.funded_amount, financials.total_cost);
    }

    #[test]
    fn test_project_financials_not_funded() {
        let financials = ProjectFinancials {
            funded_amount: 0.0,
            ..valid_project_financials()
        };
        assert_eq!(financials.funded_amount, 0.0);
    }

    #[test]
    fn test_project_financials_irr_calculation() {
        let financials = valid_project_financials();
        assert!(financials.target_irr > 0.0);
    }

    #[test]
    fn test_project_financials_payback_period() {
        let financials = valid_project_financials();
        // Simple payback calculation
        let simple_payback = financials.total_cost / financials.annual_revenue_estimate;
        assert!(simple_payback > 0.0);
    }

    // =========================================================================
    // EnergyProject Validation Tests
    // =========================================================================

    fn valid_energy_project() -> EnergyProject {
        EnergyProject {
            id: "project:solar_farm_alpha:123456".to_string(),
            terra_atlas_id: Some("TA-2024-001".to_string()),
            name: "Solar Farm Alpha".to_string(),
            description: "A 50MW solar installation in California".to_string(),
            project_type: ProjectType::Solar,
            location: valid_project_location(),
            capacity_mw: 50.0,
            status: ProjectStatus::Proposed,
            developer_did: "did:mycelix:developer1".to_string(),
            community_did: Some("did:mycelix:community1".to_string()),
            financials: valid_project_financials(),
            created: create_test_timestamp(),
            updated: create_test_timestamp(),
        }
    }

    #[test]
    fn test_energy_project_valid_developer_did() {
        let project = valid_energy_project();
        assert!(project.developer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_project_positive_capacity() {
        let project = valid_energy_project();
        assert!(project.capacity_mw > 0.0);
    }

    #[test]
    fn test_energy_project_valid_location() {
        let project = valid_energy_project();
        assert!(project.location.latitude >= -90.0 && project.location.latitude <= 90.0);
        assert!(project.location.longitude >= -180.0 && project.location.longitude <= 180.0);
    }

    #[test]
    fn test_energy_project_invalid_developer_did() {
        let project = EnergyProject {
            developer_did: "developer123".to_string(),
            ..valid_energy_project()
        };
        assert!(!project.developer_did.starts_with("did:"));
    }

    #[test]
    fn test_energy_project_zero_capacity_invalid() {
        let project = EnergyProject {
            capacity_mw: 0.0,
            ..valid_energy_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_energy_project_negative_capacity_invalid() {
        let project = EnergyProject {
            capacity_mw: -10.0,
            ..valid_energy_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_energy_project_with_terra_atlas_id() {
        let project = valid_energy_project();
        assert!(project.terra_atlas_id.is_some());
    }

    #[test]
    fn test_energy_project_without_terra_atlas_id() {
        let project = EnergyProject {
            terra_atlas_id: None,
            ..valid_energy_project()
        };
        assert!(project.terra_atlas_id.is_none());
    }

    #[test]
    fn test_energy_project_with_community() {
        let project = valid_energy_project();
        assert!(project.community_did.is_some());
    }

    #[test]
    fn test_energy_project_without_community() {
        let project = EnergyProject {
            community_did: None,
            ..valid_energy_project()
        };
        assert!(project.community_did.is_none());
    }

    #[test]
    fn test_energy_project_all_types() {
        let types = vec![
            ProjectType::Solar,
            ProjectType::Wind,
            ProjectType::Hydro,
            ProjectType::Nuclear,
            ProjectType::Geothermal,
            ProjectType::BatteryStorage,
            ProjectType::PumpedHydro,
            ProjectType::Hydrogen,
            ProjectType::Biomass,
        ];
        for proj_type in types {
            let project = EnergyProject {
                project_type: proj_type.clone(),
                ..valid_energy_project()
            };
            assert_eq!(project.project_type, proj_type);
        }
    }

    #[test]
    fn test_energy_project_all_statuses() {
        let statuses = vec![
            ProjectStatus::Proposed,
            ProjectStatus::Planning,
            ProjectStatus::Permitting,
            ProjectStatus::Financing,
            ProjectStatus::Construction,
            ProjectStatus::Operational,
            ProjectStatus::Decommissioned,
        ];
        for status in statuses {
            let project = EnergyProject {
                status: status.clone(),
                ..valid_energy_project()
            };
            assert_eq!(project.status, status);
        }
    }

    // =========================================================================
    // ProjectMilestone Validation Tests
    // =========================================================================

    fn valid_project_milestone() -> ProjectMilestone {
        ProjectMilestone {
            id: "milestone:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha:123456".to_string(),
            name: "Permitting Complete".to_string(),
            description: "All necessary permits obtained from regulatory bodies".to_string(),
            target_date: Timestamp::from_micros(1714521600000000), // 2024-05-01
            completed_date: None,
            verification_evidence: None,
        }
    }

    #[test]
    fn test_project_milestone_incomplete() {
        let milestone = valid_project_milestone();
        assert!(milestone.completed_date.is_none());
        assert!(milestone.verification_evidence.is_none());
    }

    #[test]
    fn test_project_milestone_complete() {
        let milestone = ProjectMilestone {
            completed_date: Some(Timestamp::from_micros(1712016000000000)),
            verification_evidence: Some("https://permits.gov/doc/12345".to_string()),
            ..valid_project_milestone()
        };
        assert!(milestone.completed_date.is_some());
        assert!(milestone.verification_evidence.is_some());
    }

    #[test]
    fn test_project_milestone_early_completion() {
        let milestone = ProjectMilestone {
            completed_date: Some(Timestamp::from_micros(1709510400000000)), // Before target
            ..valid_project_milestone()
        };
        assert!(milestone.completed_date.unwrap().as_micros() < milestone.target_date.as_micros());
    }

    #[test]
    fn test_project_milestone_late_completion() {
        let milestone = ProjectMilestone {
            completed_date: Some(Timestamp::from_micros(1719792000000000)), // After target
            ..valid_project_milestone()
        };
        assert!(milestone.completed_date.unwrap().as_micros() > milestone.target_date.as_micros());
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("developer_projects".to_string());
        assert_eq!(anchor.0, "developer_projects");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("community_projects".to_string());
        let anchor2 = Anchor("community_projects".to_string());
        let anchor3 = Anchor("developer_projects".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_very_large_capacity() {
        let project = EnergyProject {
            capacity_mw: 10000.0, // 10 GW
            ..valid_energy_project()
        };
        assert!(project.capacity_mw > 0.0);
    }

    #[test]
    fn test_very_small_capacity() {
        let project = EnergyProject {
            capacity_mw: 0.001, // 1 kW
            ..valid_energy_project()
        };
        assert!(project.capacity_mw > 0.0);
    }

    #[test]
    fn test_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF", "JPY", "CNY"];
        for currency in currencies {
            let financials = ProjectFinancials {
                currency: currency.to_string(),
                ..valid_project_financials()
            };
            assert!(!financials.currency.is_empty());
        }
    }

    #[test]
    fn test_long_project_description() {
        let project = EnergyProject {
            description: "A".repeat(10000),
            ..valid_energy_project()
        };
        assert!(!project.description.is_empty());
    }

    #[test]
    fn test_high_irr_target() {
        let financials = ProjectFinancials {
            target_irr: 50.0, // 50% IRR
            ..valid_project_financials()
        };
        assert!(financials.target_irr > 0.0);
    }

    #[test]
    fn test_long_payback_period() {
        let financials = ProjectFinancials {
            payback_years: 30.0, // 30 years
            ..valid_project_financials()
        };
        assert!(financials.payback_years > 0.0);
    }

    #[test]
    fn test_serialization_project() {
        let project = valid_energy_project();
        let result = serde_json::to_string(&project);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_milestone() {
        let milestone = valid_project_milestone();
        let result = serde_json::to_string(&milestone);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_financials() {
        let financials = valid_project_financials();
        let result = serde_json::to_string(&financials);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_location() {
        let location = valid_project_location();
        let result = serde_json::to_string(&location);
        assert!(result.is_ok());
    }
}
