// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport Routes Coordinator Zome
//! Business logic for vehicle registration, route creation, and stop management.

use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;
use transport_routes_integrity::*;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// ============================================================================
// VEHICLE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn register_vehicle(vehicle: Vehicle) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "register_vehicle")?;
    let action_hash = create_entry(&EntryTypes::Vehicle(vehicle.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_vehicles".to_string())))?;
    create_link(
        anchor_hash("all_vehicles")?,
        action_hash.clone(),
        LinkTypes::AllVehicles,
        (),
    )?;
    create_link(
        vehicle.owner,
        action_hash.clone(),
        LinkTypes::OwnerToVehicle,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created vehicle".into()
    )))
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
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_vehicle_status")?;
    let record = get(input.vehicle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Vehicle not found".into())
    ))?;
    let mut vehicle: Vehicle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid vehicle entry".into()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    if vehicle.owner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the owner can update vehicle status".into()
        )));
    }

    vehicle.status = input.new_status;
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Vehicle(vehicle),
    )?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated vehicle".into()
    )))
}

// ============================================================================
// ROUTE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn create_route(route: Route) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_route")?;
    let action_hash = create_entry(&EntryTypes::Route(route.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_routes".to_string())))?;
    create_link(
        anchor_hash("all_routes")?,
        action_hash.clone(),
        LinkTypes::AllRoutes,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created route".into()
    )))
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
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "add_stop")?;
    let _route = get(stop.route_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Route not found".into())))?;

    let action_hash = create_entry(&EntryTypes::Stop(stop.clone()))?;
    create_link(
        stop.route_hash,
        action_hash.clone(),
        LinkTypes::RouteToStop,
        (),
    )?;

    // Geo-spatial index for stop location
    let geo_hash = commons_types::geo::geohash_encode(stop.location_lat, stop.location_lon, 6);
    let geo_anchor_str = format!("geo:{}", geo_hash);
    create_entry(&EntryTypes::Anchor(Anchor(geo_anchor_str.clone())))?;
    create_link(
        anchor_hash(&geo_anchor_str)?,
        action_hash.clone(),
        LinkTypes::GeoIndex,
        geo_hash.as_bytes().to_vec(),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created stop".into()
    )))
}

#[hdk_extern]
pub fn get_route_stops(route_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(route_hash, LinkTypes::RouteToStop)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// MAINTENANCE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn log_maintenance(record_entry: MaintenanceRecord) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "log_maintenance")?;
    let action_hash = create_entry(&EntryTypes::MaintenanceRecord(record_entry.clone()))?;

    create_link(
        record_entry.vehicle_hash,
        action_hash.clone(),
        LinkTypes::VehicleToMaintenance,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created maintenance record".into()
    )))
}

#[hdk_extern]
pub fn get_vehicle_maintenance(vehicle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(vehicle_hash, LinkTypes::VehicleToMaintenance)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn set_vehicle_features(features: VehicleFeatures) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "set_vehicle_features")?;
    let action_hash = create_entry(&EntryTypes::VehicleFeatures(features.clone()))?;

    create_link(
        features.vehicle_hash,
        action_hash.clone(),
        LinkTypes::VehicleToFeatures,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created vehicle features".into()
    )))
}

#[hdk_extern]
pub fn get_accessible_vehicles(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_vehicles")?, LinkTypes::AllVehicles)?,
        GetStrategy::default(),
    )?;

    let mut accessible = Vec::new();
    for link in links {
        let vehicle_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        // Check if this vehicle has features with wheelchair_accessible = true
        let feature_links = get_links(
            LinkQuery::try_new(vehicle_hash.clone(), LinkTypes::VehicleToFeatures)?,
            GetStrategy::default(),
        )?;

        for flink in feature_links {
            let feat_hash = ActionHash::try_from(flink.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            if let Some(feat_record) = get(feat_hash, GetOptions::default())? {
                if let Ok(Some(features)) = feat_record.entry().to_app_option::<VehicleFeatures>() {
                    if features.wheelchair_accessible {
                        if let Some(record) = get(vehicle_hash.clone(), GetOptions::default())? {
                            accessible.push(record);
                        }
                        break;
                    }
                }
            }
        }
    }
    Ok(accessible)
}

#[hdk_extern]
pub fn get_vehicles_needing_maintenance(current_time: u64) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_vehicles")?, LinkTypes::AllVehicles)?,
        GetStrategy::default(),
    )?;

    let mut needing_maintenance = Vec::new();
    for link in links {
        let vehicle_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        let maint_links = get_links(
            LinkQuery::try_new(vehicle_hash.clone(), LinkTypes::VehicleToMaintenance)?,
            GetStrategy::default(),
        )?;

        for mlink in maint_links {
            let maint_hash = ActionHash::try_from(mlink.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            if let Some(maint_record) = get(maint_hash, GetOptions::default())? {
                if let Ok(Some(maintenance)) =
                    maint_record.entry().to_app_option::<MaintenanceRecord>()
                {
                    if let Some(next_due) = maintenance.next_due {
                        if next_due <= current_time {
                            if let Some(record) = get(vehicle_hash.clone(), GetOptions::default())?
                            {
                                needing_maintenance.push(record);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(needing_maintenance)
}

// ============================================================================
// GEO QUERIES
// ============================================================================

/// Get transport routes near a geographic location using geohash-based indexing.
#[hdk_extern]
pub fn get_nearby_routes(input: commons_types::geo::NearbyQuery) -> ExternResult<Vec<Record>> {
    let center_hash = commons_types::geo::geohash_encode(input.latitude, input.longitude, 6);
    let mut all_cells = vec![center_hash.clone()];
    all_cells.extend(commons_types::geo::geohash_neighbors(&center_hash));

    let mut records = Vec::new();
    for cell in &all_cells {
        let anchor_str = format!("geo:{}", cell);
        let anchor_entry = Anchor(anchor_str);
        let anchor_hash = hash_entry(&anchor_entry)?;
        if let Ok(links) = get_links(
            LinkQuery::try_new(anchor_hash, LinkTypes::GeoIndex)?,
            GetStrategy::Local,
        ) {
            for link in links {
                if let Ok(action_hash) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(action_hash, GetOptions::default())? {
                        records.push(record);
                    }
                }
            }
        }
    }

    let _ = input.radius_km;
    Ok(records)
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
            VehicleType::Helicopter,
            VehicleType::EVTOL,
            VehicleType::AirTaxi,
            VehicleType::Ferry,
            VehicleType::Boat,
            VehicleType::Train,
            VehicleType::Tram,
            VehicleType::Skateboard,
            VehicleType::Wheelchair,
            VehicleType::Segway,
            VehicleType::AutonomousVehicle,
            VehicleType::Drone,
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
            TransportMode::Flying,
            TransportMode::Water,
            TransportMode::Rail,
            TransportMode::Micromobility,
            TransportMode::Autonomous,
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
        let variants = vec![StopType::Pickup, StopType::Dropoff, StopType::Transfer];
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
                Waypoint {
                    lat: 32.95,
                    lon: -96.73,
                    label: Some("Start".to_string()),
                },
                Waypoint {
                    lat: 32.96,
                    lon: -96.74,
                    label: None,
                },
                Waypoint {
                    lat: 32.97,
                    lon: -96.75,
                    label: Some("End".to_string()),
                },
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
            TransportMode::Flying,
            TransportMode::Water,
            TransportMode::Rail,
            TransportMode::Micromobility,
            TransportMode::Autonomous,
        ] {
            let route = Route {
                id: "rt-mode".to_string(),
                name: "Mode Test".to_string(),
                waypoints: vec![
                    Waypoint {
                        lat: 0.0,
                        lon: 0.0,
                        label: None,
                    },
                    Waypoint {
                        lat: 1.0,
                        lon: 1.0,
                        label: None,
                    },
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
                Waypoint {
                    lat: 32.95,
                    lon: -96.73,
                    label: None,
                },
                Waypoint {
                    lat: 32.96,
                    lon: -96.74,
                    label: None,
                },
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
        let types = [
            VehicleType::Bike,
            VehicleType::Bus,
            VehicleType::ElectricScooter,
        ];
        let statuses = [
            VehicleStatus::InUse,
            VehicleStatus::Maintenance,
            VehicleStatus::Retired,
        ];
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

    // ── Additional edge-case and boundary tests ─────────────────────────

    #[test]
    fn update_vehicle_status_input_serde_all_statuses() {
        for status in [
            VehicleStatus::Available,
            VehicleStatus::InUse,
            VehicleStatus::Maintenance,
            VehicleStatus::Retired,
        ] {
            let input = UpdateVehicleStatusInput {
                vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateVehicleStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn vehicle_zero_capacity_and_passengers_roundtrip() {
        let vehicle = Vehicle {
            id: "v-empty".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0xab; 36]),
            vehicle_type: VehicleType::ElectricScooter,
            capacity_kg: 0.0,
            capacity_passengers: 0,
            status: VehicleStatus::Available,
        };
        let json = serde_json::to_string(&vehicle).unwrap();
        let decoded: Vehicle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.capacity_kg, 0.0);
        assert_eq!(decoded.capacity_passengers, 0);
    }

    #[test]
    fn vehicle_large_capacity_roundtrip() {
        let vehicle = Vehicle {
            id: "v-cargo-ship".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0xab; 36]),
            vehicle_type: VehicleType::Cargo,
            capacity_kg: 100_000.0,
            capacity_passengers: u32::MAX,
            status: VehicleStatus::InUse,
        };
        let json = serde_json::to_string(&vehicle).unwrap();
        let decoded: Vehicle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.capacity_kg, 100_000.0);
        assert_eq!(decoded.capacity_passengers, u32::MAX);
    }

    #[test]
    fn route_empty_waypoints_serde_roundtrip() {
        // Empty waypoints is invalid for validation but serde must still roundtrip
        let route = Route {
            id: "rt-empty".to_string(),
            name: "Empty Route".to_string(),
            waypoints: vec![],
            distance_km: 0.0,
            estimated_minutes: 0,
            mode: TransportMode::Walking,
        };
        let json = serde_json::to_string(&route).unwrap();
        let decoded: Route = serde_json::from_str(&json).unwrap();
        assert!(decoded.waypoints.is_empty());
        assert_eq!(decoded.distance_km, 0.0);
        assert_eq!(decoded.estimated_minutes, 0);
    }

    #[test]
    fn route_many_waypoints_serde_roundtrip() {
        let waypoints: Vec<Waypoint> = (0..200)
            .map(|i| Waypoint {
                lat: -90.0 + (i as f64) * 0.9,
                lon: -180.0 + (i as f64) * 1.8,
                label: if i % 2 == 0 {
                    Some(format!("wp-{}", i))
                } else {
                    None
                },
            })
            .collect();
        let route = Route {
            id: "rt-long".to_string(),
            name: "Cross-Country".to_string(),
            waypoints,
            distance_km: 5000.0,
            estimated_minutes: u32::MAX,
            mode: TransportMode::Mixed,
        };
        let json = serde_json::to_string(&route).unwrap();
        let decoded: Route = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.waypoints.len(), 200);
        assert_eq!(decoded.estimated_minutes, u32::MAX);
    }

    #[test]
    fn stop_extreme_coordinates_serde_roundtrip() {
        let stop = Stop {
            route_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            name: "South Pole Stop".to_string(),
            location_lat: -90.0,
            location_lon: 180.0,
            scheduled_time: Some(u64::MAX),
            stop_type: StopType::Transfer,
        };
        let json = serde_json::to_string(&stop).unwrap();
        let decoded: Stop = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, -90.0);
        assert_eq!(decoded.location_lon, 180.0);
        assert_eq!(decoded.scheduled_time, Some(u64::MAX));
    }

    #[test]
    fn waypoint_extreme_coordinates_serde_roundtrip() {
        let wp = Waypoint {
            lat: -90.0,
            lon: -180.0,
            label: Some("Edge of the world".to_string()),
        };
        let json = serde_json::to_string(&wp).unwrap();
        let decoded: Waypoint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.lat, -90.0);
        assert_eq!(decoded.lon, -180.0);
        assert_eq!(decoded.label, Some("Edge of the world".to_string()));
    }

    // ========================================================================
    // MaintenanceRecord serde roundtrip
    // ========================================================================

    #[test]
    fn maintenance_record_serde_roundtrip() {
        let m = MaintenanceRecord {
            id: "m-42".to_string(),
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Scheduled,
            description: "Oil change and filter replacement".to_string(),
            cost: 89.99,
            completed_at: 1700000000,
            next_due: Some(1703000000),
            mechanic_notes: "Everything looks good".to_string(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let decoded: MaintenanceRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "m-42");
        assert_eq!(decoded.maintenance_type, MaintenanceType::Scheduled);
        assert_eq!(decoded.cost, 89.99);
        assert_eq!(decoded.next_due, Some(1703000000));
    }

    #[test]
    fn maintenance_record_serde_no_next_due() {
        let m = MaintenanceRecord {
            id: "m-43".to_string(),
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Repair,
            description: "Flat tire repair".to_string(),
            cost: 25.0,
            completed_at: 1700000000,
            next_due: None,
            mechanic_notes: String::new(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let decoded: MaintenanceRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.next_due, None);
        assert_eq!(decoded.mechanic_notes, "");
    }

    #[test]
    fn maintenance_type_all_variants_serde() {
        for mt in [
            MaintenanceType::Scheduled,
            MaintenanceType::Repair,
            MaintenanceType::Inspection,
        ] {
            let json = serde_json::to_string(&mt).unwrap();
            let decoded: MaintenanceType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mt);
        }
    }

    // ========================================================================
    // VehicleFeatures serde roundtrip
    // ========================================================================

    #[test]
    fn vehicle_features_serde_roundtrip() {
        let f = VehicleFeatures {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            wheelchair_accessible: true,
            child_seat: true,
            pet_friendly: false,
            air_conditioning: true,
            bike_rack: false,
            luggage_capacity_liters: 350,
        };
        let json = serde_json::to_string(&f).unwrap();
        let decoded: VehicleFeatures = serde_json::from_str(&json).unwrap();
        assert!(decoded.wheelchair_accessible);
        assert!(decoded.child_seat);
        assert!(!decoded.pet_friendly);
        assert!(decoded.air_conditioning);
        assert!(!decoded.bike_rack);
        assert_eq!(decoded.luggage_capacity_liters, 350);
    }

    #[test]
    fn vehicle_features_serde_all_false() {
        let f = VehicleFeatures {
            vehicle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            wheelchair_accessible: false,
            child_seat: false,
            pet_friendly: false,
            air_conditioning: false,
            bike_rack: false,
            luggage_capacity_liters: 0,
        };
        let json = serde_json::to_string(&f).unwrap();
        let decoded: VehicleFeatures = serde_json::from_str(&json).unwrap();
        assert!(!decoded.wheelchair_accessible);
        assert_eq!(decoded.luggage_capacity_liters, 0);
    }
}
