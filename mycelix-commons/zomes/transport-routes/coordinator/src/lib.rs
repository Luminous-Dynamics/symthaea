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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_vehicle_status_input_serde_available() {
        let input = UpdateVehicleStatusInput {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: VehicleStatus::Available,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateVehicleStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, VehicleStatus::Available);
    }

    #[test]
    fn update_vehicle_status_input_serde_maintenance() {
        let input = UpdateVehicleStatusInput {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            new_status: VehicleStatus::Maintenance,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateVehicleStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, VehicleStatus::Maintenance);
    }

    #[test]
    fn vehicle_status_all_variants_serialize() {
        let statuses = vec![
            VehicleStatus::Available,
            VehicleStatus::InUse,
            VehicleStatus::Maintenance,
            VehicleStatus::Retired,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: VehicleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn vehicle_type_all_variants_serialize() {
        let types = vec![
            VehicleType::Car,
            VehicleType::Van,
            VehicleType::Bike,
            VehicleType::Bus,
            VehicleType::Cargo,
            VehicleType::ElectricScooter,
        ];
        for vt in types {
            let json = serde_json::to_string(&vt).unwrap();
            let decoded: VehicleType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, vt);
        }
    }

    // ========================================================================
    // TransportMode enum serde roundtrip
    // ========================================================================

    #[test]
    fn transport_mode_all_variants_serde_roundtrip() {
        let variants = vec![
            TransportMode::Driving,
            TransportMode::Cycling,
            TransportMode::Walking,
            TransportMode::Transit,
            TransportMode::Mixed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: TransportMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // StopType enum serde roundtrip
    // ========================================================================

    #[test]
    fn stop_type_all_variants_serde_roundtrip() {
        let variants = vec![
            StopType::Pickup,
            StopType::Dropoff,
            StopType::Transfer,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: StopType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Waypoint struct serde roundtrip
    // ========================================================================

    #[test]
    fn waypoint_serde_roundtrip_with_label() {
        let wp = Waypoint {
            lat: 32.95,
            lon: -96.73,
            label: Some("Downtown Station".to_string()),
        };
        let json = serde_json::to_string(&wp).unwrap();
        let decoded: Waypoint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lat, 32.95);
        assert_eq!(decoded.lon, -96.73);
        assert_eq!(decoded.label, Some("Downtown Station".to_string()));
    }

    #[test]
    fn waypoint_serde_roundtrip_without_label() {
        let wp = Waypoint {
            lat: -33.87,
            lon: 151.21,
            label: None,
        };
        let json = serde_json::to_string(&wp).unwrap();
        let decoded: Waypoint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lat, -33.87);
        assert_eq!(decoded.lon, 151.21);
        assert_eq!(decoded.label, None);
    }

    // ========================================================================
    // Route struct serde roundtrip
    // ========================================================================

    #[test]
    fn route_serde_roundtrip() {
        let route = Route {
            id: "rt-42".to_string(),
            name: "Downtown Loop".to_string(),
            waypoints: vec![
                Waypoint { lat: 32.95, lon: -96.73, label: Some("Start".to_string()) },
                Waypoint { lat: 32.96, lon: -96.74, label: None },
                Waypoint { lat: 32.97, lon: -96.75, label: Some("End".to_string()) },
            ],
            distance_km: 5.2,
            estimated_minutes: 20,
            mode: TransportMode::Driving,
        };
        let json = serde_json::to_string(&route).unwrap();
        let decoded: Route = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "rt-42");
        assert_eq!(decoded.name, "Downtown Loop");
        assert_eq!(decoded.waypoints.len(), 3);
        assert_eq!(decoded.waypoints[0].label, Some("Start".to_string()));
        assert_eq!(decoded.waypoints[1].label, None);
        assert_eq!(decoded.distance_km, 5.2);
        assert_eq!(decoded.estimated_minutes, 20);
        assert_eq!(decoded.mode, TransportMode::Driving);
    }

    #[test]
    fn route_serde_all_modes() {
        for mode in [
            TransportMode::Driving,
            TransportMode::Cycling,
            TransportMode::Walking,
            TransportMode::Transit,
            TransportMode::Mixed,
        ] {
            let route = Route {
                id: "rt-mode".to_string(),
                name: "Mode Test".to_string(),
                waypoints: vec![
                    Waypoint { lat: 0.0, lon: 0.0, label: None },
                    Waypoint { lat: 1.0, lon: 1.0, label: None },
                ],
                distance_km: 1.0,
                estimated_minutes: 10,
                mode: mode.clone(),
            };
            let json = serde_json::to_string(&route).unwrap();
            let decoded: Route = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.mode, mode);
        }
    }

    #[test]
    fn route_serde_minimal_waypoints() {
        let route = Route {
            id: "rt-min".to_string(),
            name: "Short Route".to_string(),
            waypoints: vec![
                Waypoint { lat: 32.95, lon: -96.73, label: None },
                Waypoint { lat: 32.96, lon: -96.74, label: None },
            ],
            distance_km: 0.5,
            estimated_minutes: 3,
            mode: TransportMode::Walking,
        };
        let json = serde_json::to_string(&route).unwrap();
        let decoded: Route = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.waypoints.len(), 2);
        assert_eq!(decoded.distance_km, 0.5);
    }

    // ========================================================================
    // Stop struct serde roundtrip
    // ========================================================================

    #[test]
    fn stop_serde_roundtrip_with_scheduled_time() {
        let stop = Stop {
            route_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            name: "Main St Station".to_string(),
            location_lat: 32.95,
            location_lon: -96.73,
            scheduled_time: Some(1700000000),
            stop_type: StopType::Pickup,
        };
        let json = serde_json::to_string(&stop).unwrap();
        let decoded: Stop = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Main St Station");
        assert_eq!(decoded.location_lat, 32.95);
        assert_eq!(decoded.location_lon, -96.73);
        assert_eq!(decoded.scheduled_time, Some(1700000000));
        assert_eq!(decoded.stop_type, StopType::Pickup);
    }

    #[test]
    fn stop_serde_roundtrip_without_scheduled_time() {
        let stop = Stop {
            route_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            name: "Flex Stop".to_string(),
            location_lat: 33.45,
            location_lon: -96.50,
            scheduled_time: None,
            stop_type: StopType::Dropoff,
        };
        let json = serde_json::to_string(&stop).unwrap();
        let decoded: Stop = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Flex Stop");
        assert_eq!(decoded.scheduled_time, None);
        assert_eq!(decoded.stop_type, StopType::Dropoff);
    }

    #[test]
    fn stop_serde_all_stop_types() {
        for st in [StopType::Pickup, StopType::Dropoff, StopType::Transfer] {
            let stop = Stop {
                route_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                name: "Type Test".to_string(),
                location_lat: 0.0,
                location_lon: 0.0,
                scheduled_time: None,
                stop_type: st.clone(),
            };
            let json = serde_json::to_string(&stop).unwrap();
            let decoded: Stop = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.stop_type, st);
        }
    }

    // ========================================================================
    // Vehicle struct serde roundtrip
    // ========================================================================

    #[test]
    fn vehicle_serde_roundtrip() {
        let vehicle = Vehicle {
            id: "v-99".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0xab; 36]),
            vehicle_type: VehicleType::Van,
            capacity_kg: 1200.0,
            capacity_passengers: 8,
            status: VehicleStatus::Available,
        };
        let json = serde_json::to_string(&vehicle).unwrap();
        let decoded: Vehicle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "v-99");
        assert_eq!(decoded.vehicle_type, VehicleType::Van);
        assert_eq!(decoded.capacity_kg, 1200.0);
        assert_eq!(decoded.capacity_passengers, 8);
        assert_eq!(decoded.status, VehicleStatus::Available);
    }

    #[test]
    fn vehicle_serde_all_types_and_statuses() {
        let types = [VehicleType::Bike, VehicleType::Bus, VehicleType::ElectricScooter];
        let statuses = [VehicleStatus::InUse, VehicleStatus::Maintenance, VehicleStatus::Retired];
        for (vt, vs) in types.iter().zip(statuses.iter()) {
            let vehicle = Vehicle {
                id: "v-combo".to_string(),
                owner: AgentPubKey::from_raw_36(vec![0xab; 36]),
                vehicle_type: vt.clone(),
                capacity_kg: 0.0,
                capacity_passengers: 1,
                status: vs.clone(),
            };
            let json = serde_json::to_string(&vehicle).unwrap();
            let decoded: Vehicle = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.vehicle_type, *vt);
            assert_eq!(decoded.status, *vs);
        }
    }
}
