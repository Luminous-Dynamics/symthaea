// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Projects Coordinator Zome
use hdk::prelude::*;
use projects_integrity::*;

/// Create or retrieve an anchor entry hash for deterministic link bases
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn register_project(input: RegisterProjectInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let project = EnergyProject {
        id: format!("project:{}:{}", input.name.replace(' ', "_"), now.as_micros()),
        terra_atlas_id: input.terra_atlas_id,
        name: input.name,
        description: input.description,
        project_type: input.project_type,
        location: input.location.clone(),
        capacity_mw: input.capacity_mw,
        status: ProjectStatus::Proposed,
        developer_did: input.developer_did.clone(),
        community_did: input.community_did.clone(),
        financials: input.financials,
        created: now,
        updated: now,
        phi_score: None,
        harmony_alignment: None,
        consciousness_assessed_at: None,
        consciousness_scorer_did: None,
    };

    let action_hash = create_entry(&EntryTypes::EnergyProject(project.clone()))?;
    create_link(anchor_hash(&input.developer_did)?, action_hash.clone(), LinkTypes::DeveloperToProjects, ())?;

    if let Some(ref community) = input.community_did {
        create_link(anchor_hash(community)?, action_hash.clone(), LinkTypes::CommunityToProjects, ())?;
    }

    // Index by location
    let geo_key = format!("energy:{}:{}", (input.location.latitude * 100.0) as i64, (input.location.longitude * 100.0) as i64);
    create_link(anchor_hash(&geo_key)?, action_hash.clone(), LinkTypes::LocationToProjects, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
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

#[hdk_extern]
pub fn update_project_status(input: UpdateStatusInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == input.project_id {
                let now = sys_time()?;
                let updated = EnergyProject { status: input.new_status, updated: now, ..project };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::EnergyProject(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateStatusInput {
    pub project_id: String,
    pub new_status: ProjectStatus,
}

#[hdk_extern]
pub fn add_milestone(input: AddMilestoneInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let milestone = ProjectMilestone {
        id: format!("milestone:{}:{}", input.project_id, now.as_micros()),
        project_id: input.project_id.clone(),
        name: input.name,
        description: input.description,
        target_date: input.target_date,
        completed_date: None,
        verification_evidence: None,
    };

    let action_hash = create_entry(&EntryTypes::ProjectMilestone(milestone))?;
    create_link(anchor_hash(&input.project_id)?, action_hash.clone(), LinkTypes::ProjectToMilestones, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddMilestoneInput {
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub target_date: Timestamp,
}

#[hdk_extern]
pub fn search_projects_by_location(input: LocationSearchInput) -> ExternResult<Vec<Record>> {
    let geo_key = format!("energy:{}:{}", (input.latitude * 100.0) as i64, (input.longitude * 100.0) as i64);
    let mut projects = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&geo_key)?, LinkTypes::LocationToProjects)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            projects.push(record);
        }
    }
    Ok(projects)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LocationSearchInput {
    pub latitude: f64,
    pub longitude: f64,
    pub radius_km: f64,
}

#[hdk_extern]
pub fn get_project(project_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == project_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all projects by developer
#[hdk_extern]
pub fn get_developer_projects(developer_did: String) -> ExternResult<Vec<Record>> {
    let mut projects = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&developer_did)?, LinkTypes::DeveloperToProjects)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            projects.push(record);
        }
    }
    Ok(projects)
}

/// Get all projects by community
#[hdk_extern]
pub fn get_community_projects(community_did: String) -> ExternResult<Vec<Record>> {
    let mut projects = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&community_did)?, LinkTypes::CommunityToProjects)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            projects.push(record);
        }
    }
    Ok(projects)
}

/// Get project milestones
#[hdk_extern]
pub fn get_project_milestones(project_id: String) -> ExternResult<Vec<Record>> {
    let mut milestones = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToMilestones)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            milestones.push(record);
        }
    }
    Ok(milestones)
}

/// Complete a milestone with verification evidence
#[hdk_extern]
pub fn complete_milestone(input: CompleteMilestoneInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::ProjectMilestone)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(milestone) = record.entry().to_app_option::<ProjectMilestone>().ok().flatten() {
            if milestone.id == input.milestone_id {
                let now = sys_time()?;
                let updated = ProjectMilestone {
                    completed_date: Some(now),
                    verification_evidence: Some(input.evidence),
                    ..milestone
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::ProjectMilestone(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Milestone not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteMilestoneInput {
    pub milestone_id: String,
    pub evidence: String,
}

/// Update project financials
#[hdk_extern]
pub fn update_project_financials(input: UpdateFinancialsInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == input.project_id {
                // Only developer can update financials
                if project.developer_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only developer can update financials".into())));
                }

                let now = sys_time()?;
                let updated = EnergyProject {
                    financials: input.financials,
                    updated: now,
                    ..project
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::EnergyProject(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateFinancialsInput {
    pub project_id: String,
    pub requester_did: String,
    pub financials: ProjectFinancials,
}

/// Get projects by status
#[hdk_extern]
pub fn get_projects_by_status(status: ProjectStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Get projects by type
#[hdk_extern]
pub fn get_projects_by_type(project_type: ProjectType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.project_type == project_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Assign community to project
#[hdk_extern]
pub fn assign_community(input: AssignCommunityInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == input.project_id {
                // Only developer can assign community
                if project.developer_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only developer can assign community".into())));
                }

                let now = sys_time()?;
                let action_hash = record.action_address().clone();

                // Create link to new community
                create_link(anchor_hash(&input.community_did)?, action_hash.clone(), LinkTypes::CommunityToProjects, ())?;

                let updated = EnergyProject {
                    community_did: Some(input.community_did),
                    updated: now,
                    ..project
                };
                let new_hash = update_entry(action_hash, &EntryTypes::EnergyProject(updated))?;
                return get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssignCommunityInput {
    pub project_id: String,
    pub requester_did: String,
    pub community_did: String,
}

/// Link project to Terra Atlas
#[hdk_extern]
pub fn link_to_terra_atlas(input: LinkTerraAtlasInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == input.project_id {
                let now = sys_time()?;
                let updated = EnergyProject {
                    terra_atlas_id: Some(input.terra_atlas_id),
                    updated: now,
                    ..project
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::EnergyProject(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LinkTerraAtlasInput {
    pub project_id: String,
    pub terra_atlas_id: String,
}

/// Update project capacity
#[hdk_extern]
pub fn update_capacity(input: UpdateCapacityInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::EnergyProject)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(project) = record.entry().to_app_option::<EnergyProject>().ok().flatten() {
            if project.id == input.project_id {
                if project.developer_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only developer can update capacity".into())));
                }

                let now = sys_time()?;
                let updated = EnergyProject {
                    capacity_mw: input.capacity_mw,
                    updated: now,
                    ..project
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::EnergyProject(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCapacityInput {
    pub project_id: String,
    pub requester_did: String,
    pub capacity_mw: f64,
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

    fn valid_project_location() -> ProjectLocation {
        ProjectLocation {
            latitude: 37.7749,
            longitude: -122.4194,
            country: "USA".to_string(),
            region: "California".to_string(),
            address: Some("123 Solar St, San Francisco, CA 94102".to_string()),
        }
    }

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

    // =========================================================================
    // RegisterProjectInput Tests
    // =========================================================================

    fn valid_register_project_input() -> RegisterProjectInput {
        RegisterProjectInput {
            terra_atlas_id: Some("TA-2024-001".to_string()),
            name: "Solar Farm Alpha".to_string(),
            description: "A 50MW solar installation in California".to_string(),
            project_type: ProjectType::Solar,
            location: valid_project_location(),
            capacity_mw: 50.0,
            developer_did: "did:mycelix:developer1".to_string(),
            community_did: Some("did:mycelix:community1".to_string()),
            financials: valid_project_financials(),
        }
    }

    #[test]
    fn test_register_project_input_valid() {
        let input = valid_register_project_input();
        assert!(input.developer_did.starts_with("did:"));
        assert!(input.capacity_mw > 0.0);
        assert!(!input.name.is_empty());
    }

    #[test]
    fn test_register_project_input_all_project_types() {
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
            let input = RegisterProjectInput {
                project_type: proj_type.clone(),
                ..valid_register_project_input()
            };
            assert_eq!(input.project_type, proj_type);
        }
    }

    #[test]
    fn test_register_project_input_with_terra_atlas() {
        let input = valid_register_project_input();
        assert!(input.terra_atlas_id.is_some());
    }

    #[test]
    fn test_register_project_input_without_terra_atlas() {
        let input = RegisterProjectInput {
            terra_atlas_id: None,
            ..valid_register_project_input()
        };
        assert!(input.terra_atlas_id.is_none());
    }

    #[test]
    fn test_register_project_input_with_community() {
        let input = valid_register_project_input();
        assert!(input.community_did.is_some());
    }

    #[test]
    fn test_register_project_input_without_community() {
        let input = RegisterProjectInput {
            community_did: None,
            ..valid_register_project_input()
        };
        assert!(input.community_did.is_none());
    }

    #[test]
    fn test_register_project_input_serialization() {
        let input = valid_register_project_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    #[test]
    fn test_register_project_input_location_validation() {
        let input = valid_register_project_input();
        assert!(input.location.latitude >= -90.0 && input.location.latitude <= 90.0);
        assert!(input.location.longitude >= -180.0 && input.location.longitude <= 180.0);
    }

    // =========================================================================
    // UpdateStatusInput Tests
    // =========================================================================

    fn valid_update_status_input() -> UpdateStatusInput {
        UpdateStatusInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            new_status: ProjectStatus::Planning,
        }
    }

    #[test]
    fn test_update_status_input_valid() {
        let input = valid_update_status_input();
        assert!(!input.project_id.is_empty());
    }

    #[test]
    fn test_update_status_input_all_statuses() {
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
            let input = UpdateStatusInput {
                new_status: status.clone(),
                ..valid_update_status_input()
            };
            assert_eq!(input.new_status, status);
        }
    }

    #[test]
    fn test_update_status_input_serialization() {
        let input = valid_update_status_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // AddMilestoneInput Tests
    // =========================================================================

    fn valid_add_milestone_input() -> AddMilestoneInput {
        AddMilestoneInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            name: "Site Preparation Complete".to_string(),
            description: "All site preparation work including grading and access roads completed".to_string(),
            target_date: Timestamp::from_micros(1714521600000000),
        }
    }

    #[test]
    fn test_add_milestone_input_valid() {
        let input = valid_add_milestone_input();
        assert!(!input.project_id.is_empty());
        assert!(!input.name.is_empty());
    }

    #[test]
    fn test_add_milestone_input_future_date() {
        let input = valid_add_milestone_input();
        // Target date should typically be in the future
        assert!(input.target_date.as_micros() > 0);
    }

    #[test]
    fn test_add_milestone_input_serialization() {
        let input = valid_add_milestone_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // LocationSearchInput Tests
    // =========================================================================

    fn valid_location_search_input() -> LocationSearchInput {
        LocationSearchInput {
            latitude: 37.7749,
            longitude: -122.4194,
            radius_km: 50.0,
        }
    }

    #[test]
    fn test_location_search_input_valid() {
        let input = valid_location_search_input();
        assert!(input.latitude >= -90.0 && input.latitude <= 90.0);
        assert!(input.longitude >= -180.0 && input.longitude <= 180.0);
        assert!(input.radius_km > 0.0);
    }

    #[test]
    fn test_location_search_input_small_radius() {
        let input = LocationSearchInput {
            radius_km: 1.0,
            ..valid_location_search_input()
        };
        assert!(input.radius_km > 0.0);
    }

    #[test]
    fn test_location_search_input_large_radius() {
        let input = LocationSearchInput {
            radius_km: 1000.0,
            ..valid_location_search_input()
        };
        assert!(input.radius_km > 0.0);
    }

    #[test]
    fn test_location_search_input_equator() {
        let input = LocationSearchInput {
            latitude: 0.0,
            longitude: 0.0,
            radius_km: 100.0,
        };
        assert_eq!(input.latitude, 0.0);
        assert_eq!(input.longitude, 0.0);
    }

    // =========================================================================
    // CompleteMilestoneInput Tests
    // =========================================================================

    fn valid_complete_milestone_input() -> CompleteMilestoneInput {
        CompleteMilestoneInput {
            milestone_id: "milestone:project1:123456".to_string(),
            evidence: "https://verification.example.com/doc/12345".to_string(),
        }
    }

    #[test]
    fn test_complete_milestone_input_valid() {
        let input = valid_complete_milestone_input();
        assert!(!input.milestone_id.is_empty());
        assert!(!input.evidence.is_empty());
    }

    #[test]
    fn test_complete_milestone_input_url_evidence() {
        let input = valid_complete_milestone_input();
        assert!(input.evidence.starts_with("https://"));
    }

    #[test]
    fn test_complete_milestone_input_document_evidence() {
        let input = CompleteMilestoneInput {
            evidence: "Document reference: PERMIT-2024-001-APPROVED".to_string(),
            ..valid_complete_milestone_input()
        };
        assert!(!input.evidence.is_empty());
    }

    // =========================================================================
    // UpdateFinancialsInput Tests
    // =========================================================================

    fn valid_update_financials_input() -> UpdateFinancialsInput {
        UpdateFinancialsInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            requester_did: "did:mycelix:developer1".to_string(),
            financials: valid_project_financials(),
        }
    }

    #[test]
    fn test_update_financials_input_valid() {
        let input = valid_update_financials_input();
        assert!(!input.project_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
    }

    #[test]
    fn test_update_financials_input_increased_funding() {
        let input = UpdateFinancialsInput {
            financials: ProjectFinancials {
                funded_amount: 8_000_000.0,
                ..valid_project_financials()
            },
            ..valid_update_financials_input()
        };
        assert!(input.financials.funded_amount > 5_000_000.0);
    }

    #[test]
    fn test_update_financials_input_fully_funded() {
        let input = UpdateFinancialsInput {
            financials: ProjectFinancials {
                funded_amount: 10_000_000.0,
                ..valid_project_financials()
            },
            ..valid_update_financials_input()
        };
        assert_eq!(input.financials.funded_amount, input.financials.total_cost);
    }

    // =========================================================================
    // AssignCommunityInput Tests
    // =========================================================================

    fn valid_assign_community_input() -> AssignCommunityInput {
        AssignCommunityInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            requester_did: "did:mycelix:developer1".to_string(),
            community_did: "did:mycelix:community1".to_string(),
        }
    }

    #[test]
    fn test_assign_community_input_valid() {
        let input = valid_assign_community_input();
        assert!(!input.project_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
        assert!(input.community_did.starts_with("did:"));
    }

    #[test]
    fn test_assign_community_input_different_dids() {
        let input = valid_assign_community_input();
        assert_ne!(input.requester_did, input.community_did);
    }

    // =========================================================================
    // LinkTerraAtlasInput Tests
    // =========================================================================

    fn valid_link_terra_atlas_input() -> LinkTerraAtlasInput {
        LinkTerraAtlasInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
        }
    }

    #[test]
    fn test_link_terra_atlas_input_valid() {
        let input = valid_link_terra_atlas_input();
        assert!(!input.project_id.is_empty());
        assert!(!input.terra_atlas_id.is_empty());
    }

    #[test]
    fn test_link_terra_atlas_input_various_formats() {
        let ids = vec![
            "TA-2024-001",
            "TERRA-ATLAS-PROJECT-12345",
            "ta_project_001",
        ];
        for id in ids {
            let input = LinkTerraAtlasInput {
                terra_atlas_id: id.to_string(),
                ..valid_link_terra_atlas_input()
            };
            assert!(!input.terra_atlas_id.is_empty());
        }
    }

    // =========================================================================
    // UpdateCapacityInput Tests
    // =========================================================================

    fn valid_update_capacity_input() -> UpdateCapacityInput {
        UpdateCapacityInput {
            project_id: "project:solar_farm_alpha:123456".to_string(),
            requester_did: "did:mycelix:developer1".to_string(),
            capacity_mw: 75.0,
        }
    }

    #[test]
    fn test_update_capacity_input_valid() {
        let input = valid_update_capacity_input();
        assert!(!input.project_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
        assert!(input.capacity_mw > 0.0);
    }

    #[test]
    fn test_update_capacity_input_increase() {
        let input = UpdateCapacityInput {
            capacity_mw: 100.0,
            ..valid_update_capacity_input()
        };
        assert!(input.capacity_mw > 75.0);
    }

    #[test]
    fn test_update_capacity_input_decrease() {
        let input = UpdateCapacityInput {
            capacity_mw: 25.0,
            ..valid_update_capacity_input()
        };
        assert!(input.capacity_mw < 75.0);
    }

    #[test]
    fn test_update_capacity_input_large_capacity() {
        let input = UpdateCapacityInput {
            capacity_mw: 5000.0, // 5 GW
            ..valid_update_capacity_input()
        };
        assert!(input.capacity_mw > 0.0);
    }

    #[test]
    fn test_update_capacity_input_small_capacity() {
        let input = UpdateCapacityInput {
            capacity_mw: 0.1, // 100 kW
            ..valid_update_capacity_input()
        };
        assert!(input.capacity_mw > 0.0);
    }

    // =========================================================================
    // Business Logic Edge Cases
    // =========================================================================

    #[test]
    fn test_geo_key_generation() {
        // Test the geo key format used for location indexing
        let lat = 37.7749;
        let lon = -122.4194;
        let geo_key = format!("energy:{}:{}", (lat * 100.0) as i64, (lon * 100.0) as i64);
        assert!(!geo_key.is_empty());
        assert!(geo_key.starts_with("energy:"));
    }

    #[test]
    fn test_project_id_format() {
        // Test ID generation format
        let name = "Solar Farm Alpha";
        let timestamp = 1704067200000000_u64;
        let id = format!("project:{}:{}", name.replace(' ', "_"), timestamp);
        assert!(id.starts_with("project:"));
        assert!(id.contains("Solar_Farm_Alpha"));
    }

    #[test]
    fn test_milestone_id_format() {
        // Test milestone ID format
        let project_id = "project:solar_farm:123";
        let timestamp = 1704067200000000_u64;
        let id = format!("milestone:{}:{}", project_id, timestamp);
        assert!(id.starts_with("milestone:"));
    }

    #[test]
    fn test_location_precision() {
        // Test high precision coordinates
        let input = RegisterProjectInput {
            location: ProjectLocation {
                latitude: 37.77490123456789,
                longitude: -122.41941234567890,
                ..valid_project_location()
            },
            ..valid_register_project_input()
        };
        assert!(input.location.latitude.is_finite());
        assert!(input.location.longitude.is_finite());
    }

    #[test]
    fn test_project_name_special_characters() {
        let input = RegisterProjectInput {
            name: "Solar Farm #1 - Phase A (North)".to_string(),
            ..valid_register_project_input()
        };
        assert!(!input.name.is_empty());
    }

    #[test]
    fn test_unicode_project_name() {
        let input = RegisterProjectInput {
            name: "Sonnenkraftwerk München".to_string(),
            ..valid_register_project_input()
        };
        assert!(!input.name.is_empty());
    }

    #[test]
    fn test_long_description() {
        let input = RegisterProjectInput {
            description: "A".repeat(5000),
            ..valid_register_project_input()
        };
        assert_eq!(input.description.len(), 5000);
    }

    #[test]
    fn test_deserialization_roundtrip() {
        let input = valid_register_project_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: RegisterProjectInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, input.name);
        assert_eq!(deserialized.capacity_mw, input.capacity_mw);
    }

    #[test]
    fn test_multiple_milestones_same_project() {
        let milestones: Vec<AddMilestoneInput> = (0..5)
            .map(|i| AddMilestoneInput {
                name: format!("Milestone {}", i + 1),
                target_date: Timestamp::from_micros(1704067200000000 + (i as i64 * 86400000000)),
                ..valid_add_milestone_input()
            })
            .collect();

        assert_eq!(milestones.len(), 5);
        for (i, m) in milestones.iter().enumerate() {
            assert!(m.name.contains(&format!("{}", i + 1)));
        }
    }
}
