//! Transport Routes Coordinator Zome
//! Business logic for vehicle registration, route creation, and stop management.

use transport_routes_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// VEHICLE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn register_vehicle(vehicle: Vehicle) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Vehicle(vehicle.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_vehicles".to_string())))?;
    create_link(anchor_hash("all_vehicles")?, action_hash.clone(), LinkTypes::AllVehicles, ())?;
    create_link(vehicle.owner, action_hash.clone(), LinkTypes::OwnerToVehicle, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created vehicle".into())))
}

#[hdk_extern]
pub fn get_vehicle(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_my_vehicles(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::OwnerToVehicle)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateVehicleStatusInput {
    pub vehicle_hash: ActionHash,
    pub new_status: VehicleStatus,
}

#[hdk_extern]
pub fn update_vehicle_status(input: UpdateVehicleStatusInput) -> ExternResult<Record> {
    let record = get(input.vehicle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Vehicle not found".into())))?;
    let mut vehicle: Vehicle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid vehicle entry".into())))?;

    let agent = agent_info()?.agent_initial_pubkey;
    if vehicle.owner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest("Only the owner can update vehicle status".into())));
    }

    vehicle.status = input.new_status;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::Vehicle(vehicle))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated vehicle".into())))
}

// ============================================================================
// ROUTE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn create_route(route: Route) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Route(route.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_routes".to_string())))?;
    create_link(anchor_hash("all_routes")?, action_hash.clone(), LinkTypes::AllRoutes, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created route".into())))
}

#[hdk_extern]
pub fn get_all_routes(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_routes")?, LinkTypes::AllRoutes)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// STOP MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn add_stop(stop: Stop) -> ExternResult<Record> {
    let _route = get(stop.route_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Route not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Stop(stop.clone()))?;
    create_link(stop.route_hash, action_hash.clone(), LinkTypes::RouteToStop, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created stop".into())))
}

#[hdk_extern]
pub fn get_route_stops(route_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(route_hash, LinkTypes::RouteToStop)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
