//! Maintenance Coordinator Zome
//! Business logic for maintenance requests, work orders, and inspections.

use hdk::prelude::*;
use housing_maintenance_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Submit a new maintenance request

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn submit_request(req: MaintenanceRequest) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "submit_request")?;
    let action_hash = create_entry(&EntryTypes::MaintenanceRequest(req.clone()))?;

    // Link to open requests
    create_entry(&EntryTypes::Anchor(Anchor("open_requests".to_string())))?;
    create_link(
        anchor_hash("open_requests")?,
        action_hash.clone(),
        LinkTypes::OpenRequests,
        (),
    )?;

    // Link building to request
    create_link(
        req.building_hash,
        action_hash.clone(),
        LinkTypes::BuildingToRequest,
        (),
    )?;

    // Link reporter to request
    create_link(
        req.reported_by,
        action_hash.clone(),
        LinkTypes::ReporterToRequest,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcknowledgeRequestInput {
    pub request_hash: ActionHash,
}

/// Acknowledge a maintenance request
#[hdk_extern]
pub fn acknowledge_request(input: AcknowledgeRequestInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "acknowledge_request")?;
    let record = get(input.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Request not found".into())
    ))?;

    let mut req: MaintenanceRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    if req.status != MaintenanceStatus::Reported {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request must be in Reported status to acknowledge".into()
        )));
    }

    req.status = MaintenanceStatus::Acknowledged;

    let new_hash = update_entry(input.request_hash, &EntryTypes::MaintenanceRequest(req))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated request".into()
    )))
}

/// Create a work order for a maintenance request
#[hdk_extern]
pub fn create_work_order(order: WorkOrder) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "create_work_order")?;
    if order.assigned_to.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Assigned-to must be at most 256 characters".into()
        )));
    }

    // Update the request status to Scheduled
    let req_record = get(order.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Request not found".into())
    ))?;

    let mut req: MaintenanceRequest = req_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    req.status = MaintenanceStatus::Scheduled;
    update_entry(
        order.request_hash.clone(),
        &EntryTypes::MaintenanceRequest(req),
    )?;

    let action_hash = create_entry(&EntryTypes::WorkOrder(order.clone()))?;

    // Link request to work order
    create_link(
        order.request_hash,
        action_hash.clone(),
        LinkTypes::RequestToWorkOrder,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created work order".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteWorkOrderInput {
    pub work_order_hash: ActionHash,
    pub actual_cost_cents: Option<u64>,
    pub notes: String,
}

/// Complete a work order and mark the request as completed
#[hdk_extern]
pub fn complete_work_order(input: CompleteWorkOrderInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "complete_work_order")?;
    let record = get(input.work_order_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Work order not found".into())
    ))?;

    let mut order: WorkOrder = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid work order entry".into()
        )))?;

    let now = sys_time()?;
    order.completed_date = Some(now);
    order.actual_cost_cents = input.actual_cost_cents;
    if !input.notes.is_empty() {
        order.notes = input.notes;
    }

    let new_hash = update_entry(input.work_order_hash, &EntryTypes::WorkOrder(order.clone()))?;

    // Update the maintenance request status to Completed
    let req_record = get(order.request_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Original request not found".into())
    ))?;

    let mut req: MaintenanceRequest = req_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid request entry".into()
        )))?;

    req.status = MaintenanceStatus::Completed;
    update_entry(
        order.request_hash.clone(),
        &EntryTypes::MaintenanceRequest(req),
    )?;

    // Move from open to completed
    let open_links = get_links(
        LinkQuery::try_new(anchor_hash("open_requests")?, LinkTypes::OpenRequests)?,
        GetStrategy::default(),
    )?;
    for link in open_links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == order.request_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    create_entry(&EntryTypes::Anchor(Anchor(
        "completed_requests".to_string(),
    )))?;
    create_link(
        anchor_hash("completed_requests")?,
        order.request_hash,
        LinkTypes::CompletedRequests,
        (),
    )?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated work order".into()
    )))
}

/// Schedule a building inspection
#[hdk_extern]
pub fn schedule_inspection(inspection: Inspection) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "schedule_inspection")?;
    let action_hash = create_entry(&EntryTypes::Inspection(inspection.clone()))?;

    create_link(
        inspection.building_hash,
        action_hash.clone(),
        LinkTypes::BuildingToInspection,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created inspection".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordInspectionInput {
    pub inspection_hash: ActionHash,
    pub findings: Vec<String>,
    pub passed: bool,
    pub next_due: Option<Timestamp>,
}

/// Record the results of an inspection
#[hdk_extern]
pub fn record_inspection(input: RecordInspectionInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "record_inspection")?;
    let record = get(input.inspection_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Inspection not found".into())
    ))?;

    let mut inspection: Inspection = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid inspection entry".into()
        )))?;

    inspection.findings = input.findings;
    inspection.passed = input.passed;
    inspection.next_due = input.next_due;

    let new_hash = update_entry(input.inspection_hash, &EntryTypes::Inspection(inspection))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated inspection".into()
    )))
}

/// Get all open maintenance requests
#[hdk_extern]
pub fn get_open_requests(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("open_requests")?, LinkTypes::OpenRequests)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            requests.push(record);
        }
    }

    // Sort by priority (Emergency first)
    requests.sort_by(|a, b| {
        let priority_ord = |r: &Record| -> u8 {
            r.entry()
                .to_app_option::<MaintenanceRequest>()
                .ok()
                .flatten()
                .map(|req| match req.priority {
                    MaintenancePriority::Emergency => 0,
                    MaintenancePriority::Urgent => 1,
                    MaintenancePriority::Normal => 2,
                    MaintenancePriority::Low => 3,
                    MaintenancePriority::Scheduled => 4,
                })
                .unwrap_or(255)
        };
        priority_ord(a).cmp(&priority_ord(b))
    });

    Ok(requests)
}

/// Get maintenance history for a building
#[hdk_extern]
pub fn get_building_maintenance_history(building_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(building_hash, LinkTypes::BuildingToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            requests.push(record);
        }
    }

    // Sort by reported time
    requests.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(requests)
}

// ============================================================================
// Cross-domain: Search community tool library for maintenance resources
// ============================================================================

/// Wire-compatible copy of mutualaid ResourceType for deserialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MutualAidResourceType {
    PowerTool,
    HandTool,
    GardenTool,
    CookingEquipment,
    CraftingSupplies,
    Car,
    Truck,
    Bicycle,
    Trailer,
    Boat,
    MeetingRoom,
    Workshop,
    Kitchen,
    GardenPlot,
    StorageSpace,
    ParkingSpot,
    CampingGear,
    SportsEquipment,
    MusicInstrument,
    Photography,
    Projector,
    Custom(String),
}

/// Wire-compatible copy of mutualaid SearchResourcesInput.
#[derive(Serialize, Deserialize, Debug)]
struct LocalSearchResourcesInput {
    pub resource_type: Option<MutualAidResourceType>,
    pub available_only: bool,
    pub query: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindCommunityResourcesInput {
    /// The maintenance category to search for (maps to resource types).
    pub maintenance_category: String,
    /// Optional text query for resource names/descriptions.
    pub query: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CommunityResourcesResult {
    pub resources_found: u32,
    pub has_resources: bool,
    pub resource_type_searched: Option<String>,
    pub error: Option<String>,
}

/// Search the community mutual-aid tool library for resources matching
/// a maintenance request.
///
/// Cross-domain call: housing-maintenance queries mutualaid_resources
/// via `call(CallTargetCell::Local, ...)` to find community-owned tools
/// and equipment that could be used for housing repairs.
#[hdk_extern]
pub fn find_community_resources_for_repair(
    input: FindCommunityResourcesInput,
) -> ExternResult<CommunityResourcesResult> {
    let resource_type = map_maintenance_to_resource_type(&input.maintenance_category);
    let type_label = resource_type.as_ref().map(|t| format!("{:?}", t));

    let search = LocalSearchResourcesInput {
        resource_type,
        available_only: true,
        query: input.query,
        limit: Some(10),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("mutualaid_resources"),
        FunctionName::from("search_resources"),
        None,
        search,
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;
            let count = records.len() as u32;
            Ok(CommunityResourcesResult {
                resources_found: count,
                has_resources: count > 0,
                resource_type_searched: type_label,
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(CommunityResourcesResult {
            resources_found: 0,
            has_resources: false,
            resource_type_searched: type_label,
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(CommunityResourcesResult {
            resources_found: 0,
            has_resources: false,
            resource_type_searched: type_label,
            error: Some("Failed to query mutualaid resources".into()),
        }),
    }
}

/// Map maintenance category strings to mutualaid resource types.
fn map_maintenance_to_resource_type(category: &str) -> Option<MutualAidResourceType> {
    match category.to_lowercase().as_str() {
        "plumbing" | "electrical" | "structural" => Some(MutualAidResourceType::PowerTool),
        "painting" | "carpentry" | "general" => Some(MutualAidResourceType::HandTool),
        "landscaping" | "grounds" => Some(MutualAidResourceType::GardenTool),
        "workshop" | "fabrication" => Some(MutualAidResourceType::CraftingSupplies),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    // ── Pure function tests: map_maintenance_to_resource_type ──────────

    #[test]
    fn map_plumbing_to_power_tool() {
        let result = map_maintenance_to_resource_type("plumbing");
        assert!(matches!(result, Some(MutualAidResourceType::PowerTool)));
    }

    #[test]
    fn map_electrical_to_power_tool() {
        let result = map_maintenance_to_resource_type("electrical");
        assert!(matches!(result, Some(MutualAidResourceType::PowerTool)));
    }

    #[test]
    fn map_structural_to_power_tool() {
        let result = map_maintenance_to_resource_type("structural");
        assert!(matches!(result, Some(MutualAidResourceType::PowerTool)));
    }

    #[test]
    fn map_painting_to_hand_tool() {
        let result = map_maintenance_to_resource_type("painting");
        assert!(matches!(result, Some(MutualAidResourceType::HandTool)));
    }

    #[test]
    fn map_carpentry_to_hand_tool() {
        let result = map_maintenance_to_resource_type("carpentry");
        assert!(matches!(result, Some(MutualAidResourceType::HandTool)));
    }

    #[test]
    fn map_general_to_hand_tool() {
        let result = map_maintenance_to_resource_type("general");
        assert!(matches!(result, Some(MutualAidResourceType::HandTool)));
    }

    #[test]
    fn map_landscaping_to_garden_tool() {
        let result = map_maintenance_to_resource_type("landscaping");
        assert!(matches!(result, Some(MutualAidResourceType::GardenTool)));
    }

    #[test]
    fn map_grounds_to_garden_tool() {
        let result = map_maintenance_to_resource_type("grounds");
        assert!(matches!(result, Some(MutualAidResourceType::GardenTool)));
    }

    #[test]
    fn map_workshop_to_crafting_supplies() {
        let result = map_maintenance_to_resource_type("workshop");
        assert!(matches!(
            result,
            Some(MutualAidResourceType::CraftingSupplies)
        ));
    }

    #[test]
    fn map_fabrication_to_crafting_supplies() {
        let result = map_maintenance_to_resource_type("fabrication");
        assert!(matches!(
            result,
            Some(MutualAidResourceType::CraftingSupplies)
        ));
    }

    #[test]
    fn map_unknown_category_returns_none() {
        assert!(map_maintenance_to_resource_type("hvac").is_none());
        assert!(map_maintenance_to_resource_type("roofing").is_none());
        assert!(map_maintenance_to_resource_type("pest control").is_none());
        assert!(map_maintenance_to_resource_type("").is_none());
    }

    #[test]
    fn map_category_case_insensitive() {
        assert!(matches!(
            map_maintenance_to_resource_type("PLUMBING"),
            Some(MutualAidResourceType::PowerTool)
        ));
        assert!(matches!(
            map_maintenance_to_resource_type("Painting"),
            Some(MutualAidResourceType::HandTool)
        ));
        assert!(matches!(
            map_maintenance_to_resource_type("LANDSCAPING"),
            Some(MutualAidResourceType::GardenTool)
        ));
        assert!(matches!(
            map_maintenance_to_resource_type("Workshop"),
            Some(MutualAidResourceType::CraftingSupplies)
        ));
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn acknowledge_request_input_serde_roundtrip() {
        let input = AcknowledgeRequestInput {
            request_hash: fake_action_hash(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AcknowledgeRequestInput = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn complete_work_order_input_serde_roundtrip() {
        let input = CompleteWorkOrderInput {
            work_order_hash: fake_action_hash(),
            actual_cost_cents: Some(25000),
            notes: "Fixed the leak under the sink".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteWorkOrderInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.actual_cost_cents, Some(25000));
        assert_eq!(decoded.notes, "Fixed the leak under the sink");
    }

    #[test]
    fn complete_work_order_input_no_cost_serde_roundtrip() {
        let input = CompleteWorkOrderInput {
            work_order_hash: fake_action_hash(),
            actual_cost_cents: None,
            notes: String::new(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteWorkOrderInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.actual_cost_cents.is_none());
        assert!(decoded.notes.is_empty());
    }

    #[test]
    fn record_inspection_input_serde_roundtrip() {
        let input = RecordInspectionInput {
            inspection_hash: fake_action_hash(),
            findings: vec![
                "Minor crack in wall".to_string(),
                "Water stain on ceiling".to_string(),
            ],
            passed: false,
            next_due: Some(Timestamp::from_micros(1_000_000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordInspectionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.findings.len(), 2);
        assert!(!decoded.passed);
        assert!(decoded.next_due.is_some());
    }

    #[test]
    fn record_inspection_input_passed_no_next_due_serde() {
        let input = RecordInspectionInput {
            inspection_hash: fake_action_hash(),
            findings: vec![],
            passed: true,
            next_due: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordInspectionInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.findings.is_empty());
        assert!(decoded.passed);
        assert!(decoded.next_due.is_none());
    }

    #[test]
    fn find_community_resources_input_serde_roundtrip() {
        let input = FindCommunityResourcesInput {
            maintenance_category: "plumbing".to_string(),
            query: Some("wrench".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindCommunityResourcesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.maintenance_category, "plumbing");
        assert_eq!(decoded.query, Some("wrench".to_string()));
    }

    #[test]
    fn find_community_resources_input_no_query_serde() {
        let input = FindCommunityResourcesInput {
            maintenance_category: "electrical".to_string(),
            query: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindCommunityResourcesInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.query.is_none());
    }

    #[test]
    fn community_resources_result_found_serde_roundtrip() {
        let result = CommunityResourcesResult {
            resources_found: 3,
            has_resources: true,
            resource_type_searched: Some("PowerTool".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: CommunityResourcesResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.resources_found, 3);
        assert!(decoded.has_resources);
        assert_eq!(
            decoded.resource_type_searched,
            Some("PowerTool".to_string())
        );
        assert!(decoded.error.is_none());
    }

    #[test]
    fn community_resources_result_error_serde_roundtrip() {
        let result = CommunityResourcesResult {
            resources_found: 0,
            has_resources: false,
            resource_type_searched: None,
            error: Some("Network error: connection refused".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: CommunityResourcesResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.resources_found, 0);
        assert!(!decoded.has_resources);
        assert!(decoded
            .error
            .as_ref()
            .unwrap()
            .contains("connection refused"));
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn maintenance_category_all_variants_serde() {
        let variants = vec![
            MaintenanceCategory::Plumbing,
            MaintenanceCategory::Electrical,
            MaintenanceCategory::HVAC,
            MaintenanceCategory::Structural,
            MaintenanceCategory::Appliance,
            MaintenanceCategory::Exterior,
            MaintenanceCategory::CommonArea,
            MaintenanceCategory::Safety,
            MaintenanceCategory::Pest,
            MaintenanceCategory::Other("Custom".to_string()),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MaintenanceCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn maintenance_priority_all_variants_serde() {
        let variants = vec![
            MaintenancePriority::Emergency,
            MaintenancePriority::Urgent,
            MaintenancePriority::Normal,
            MaintenancePriority::Low,
            MaintenancePriority::Scheduled,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MaintenancePriority = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn maintenance_status_all_variants_serde() {
        let variants = vec![
            MaintenanceStatus::Reported,
            MaintenanceStatus::Acknowledged,
            MaintenanceStatus::Scheduled,
            MaintenanceStatus::InProgress,
            MaintenanceStatus::Completed,
            MaintenanceStatus::Deferred,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MaintenanceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn inspection_type_all_variants_serde() {
        let variants = vec![
            InspectionType::Annual,
            InspectionType::Safety,
            InspectionType::Code,
            InspectionType::PreMove,
            InspectionType::PostMove,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: InspectionType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn maintenance_request_serde_roundtrip() {
        let req = MaintenanceRequest {
            unit_hash: Some(fake_action_hash()),
            building_hash: fake_action_hash(),
            reported_by: fake_agent(),
            title: "Leaking faucet".to_string(),
            description: "Kitchen faucet drips constantly".to_string(),
            category: MaintenanceCategory::Plumbing,
            priority: MaintenancePriority::Normal,
            status: MaintenanceStatus::Reported,
            reported_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: MaintenanceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn work_order_serde_roundtrip() {
        let order = WorkOrder {
            request_hash: fake_action_hash(),
            assigned_to: "Bob the Plumber".to_string(),
            description: "Replace faucet cartridge".to_string(),
            estimated_cost_cents: Some(15000),
            actual_cost_cents: None,
            scheduled_date: Some(Timestamp::from_micros(2000)),
            completed_date: None,
            notes: "Need to order parts first".to_string(),
        };
        let json = serde_json::to_string(&order).unwrap();
        let decoded: WorkOrder = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, order);
    }

    #[test]
    fn inspection_serde_roundtrip() {
        let inspection = Inspection {
            building_hash: fake_action_hash(),
            inspector: fake_agent(),
            inspection_type: InspectionType::Annual,
            date: Timestamp::from_micros(3000),
            findings: vec!["All clear".to_string()],
            passed: true,
            next_due: Some(Timestamp::from_micros(4000)),
        };
        let json = serde_json::to_string(&inspection).unwrap();
        let decoded: Inspection = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, inspection);
    }

    // ── MutualAidResourceType serde roundtrip ─────────────────────────

    #[test]
    fn mutual_aid_resource_type_serde_roundtrip() {
        let variants: Vec<MutualAidResourceType> = vec![
            MutualAidResourceType::PowerTool,
            MutualAidResourceType::HandTool,
            MutualAidResourceType::GardenTool,
            MutualAidResourceType::CookingEquipment,
            MutualAidResourceType::CraftingSupplies,
            MutualAidResourceType::Car,
            MutualAidResourceType::Truck,
            MutualAidResourceType::Bicycle,
            MutualAidResourceType::Trailer,
            MutualAidResourceType::Boat,
            MutualAidResourceType::MeetingRoom,
            MutualAidResourceType::Workshop,
            MutualAidResourceType::Kitchen,
            MutualAidResourceType::GardenPlot,
            MutualAidResourceType::StorageSpace,
            MutualAidResourceType::ParkingSpot,
            MutualAidResourceType::CampingGear,
            MutualAidResourceType::SportsEquipment,
            MutualAidResourceType::MusicInstrument,
            MutualAidResourceType::Photography,
            MutualAidResourceType::Projector,
            MutualAidResourceType::Custom("Custom tool".to_string()),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MutualAidResourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", decoded), format!("{:?}", v));
        }
    }
}
