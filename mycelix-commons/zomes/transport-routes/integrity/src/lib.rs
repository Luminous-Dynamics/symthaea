//! Transport Routes Integrity Zome
//! Entry types and validation for vehicles, routes, and stops.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// VEHICLE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VehicleType {
    Car,
    Van,
    Bike,
    Bus,
    Cargo,
    ElectricScooter,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VehicleStatus {
    Available,
    InUse,
    Maintenance,
    Retired,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vehicle {
    pub id: String,
    pub owner: AgentPubKey,
    pub vehicle_type: VehicleType,
    pub capacity_kg: f64,
    pub capacity_passengers: u32,
    pub status: VehicleStatus,
}

// ============================================================================
// ROUTE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransportMode {
    Driving,
    Cycling,
    Walking,
    Transit,
    Mixed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Waypoint {
    pub lat: f64,
    pub lon: f64,
    pub label: Option<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Route {
    pub id: String,
    pub name: String,
    pub waypoints: Vec<Waypoint>,
    pub distance_km: f64,
    pub estimated_minutes: u32,
    pub mode: TransportMode,
}

// ============================================================================
// STOP
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StopType {
    Pickup,
    Dropoff,
    Transfer,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Stop {
    pub route_hash: ActionHash,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub scheduled_time: Option<u64>,
    pub stop_type: StopType,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Vehicle(Vehicle),
    Route(Route),
    Stop(Stop),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllVehicles,
    AllRoutes,
    OwnerToVehicle,
    RouteToStop,
    VehicleToRoute,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Vehicle(v) => validate_vehicle(v),
                EntryTypes::Route(r) => validate_route(r),
                EntryTypes::Stop(s) => validate_stop(s),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Vehicle(v) => validate_vehicle(v),
                EntryTypes::Route(r) => validate_route(r),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            match link_type {
                LinkTypes::AllVehicles => {
                    if tag.0.len() > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "AllVehicles link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AllRoutes => {
                    if tag.0.len() > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "AllRoutes link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::OwnerToVehicle => {
                    if tag.0.len() > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "OwnerToVehicle link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::RouteToStop => {
                    if tag.0.len() > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "RouteToStop link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::VehicleToRoute => {
                    if tag.0.len() > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "VehicleToRoute link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_vehicle(v: Vehicle) -> ExternResult<ValidateCallbackResult> {
    if v.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Vehicle ID cannot be empty".into()));
    }
    if v.capacity_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_route(r: Route) -> ExternResult<ValidateCallbackResult> {
    if r.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Route ID cannot be empty".into()));
    }
    if r.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Route name cannot be empty".into()));
    }
    if r.waypoints.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid("Route must have at least 2 waypoints".into()));
    }
    if r.distance_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Distance must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_stop(s: Stop) -> ExternResult<ValidateCallbackResult> {
    if s.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Stop name cannot be empty".into()));
    }
    if s.location_lat < -90.0 || s.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
    }
    if s.location_lon < -180.0 || s.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }
    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_vehicle() -> Vehicle {
        Vehicle {
            id: "v-1".into(),
            owner: fake_agent(),
            vehicle_type: VehicleType::Car,
            capacity_kg: 500.0,
            capacity_passengers: 4,
            status: VehicleStatus::Available,
        }
    }

    fn valid_route() -> Route {
        Route {
            id: "r-1".into(),
            name: "Downtown Loop".into(),
            waypoints: vec![
                Waypoint { lat: 32.95, lon: -96.73, label: Some("Start".into()) },
                Waypoint { lat: 32.96, lon: -96.74, label: Some("End".into()) },
            ],
            distance_km: 3.5,
            estimated_minutes: 15,
            mode: TransportMode::Driving,
        }
    }

    fn valid_stop() -> Stop {
        Stop {
            route_hash: fake_action_hash(),
            name: "Main St Station".into(),
            location_lat: 32.95,
            location_lon: -96.73,
            scheduled_time: Some(1700000000),
            stop_type: StopType::Pickup,
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_vehicle_type() {
        let types = vec![
            VehicleType::Car, VehicleType::Van, VehicleType::Bike,
            VehicleType::Bus, VehicleType::Cargo, VehicleType::ElectricScooter,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: VehicleType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, t);
        }
    }

    #[test]
    fn serde_roundtrip_vehicle_status() {
        let statuses = vec![
            VehicleStatus::Available, VehicleStatus::InUse,
            VehicleStatus::Maintenance, VehicleStatus::Retired,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: VehicleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_transport_mode() {
        let modes = vec![
            TransportMode::Driving, TransportMode::Cycling,
            TransportMode::Walking, TransportMode::Transit, TransportMode::Mixed,
        ];
        for m in &modes {
            let json = serde_json::to_string(m).unwrap();
            let back: TransportMode = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, m);
        }
    }

    #[test]
    fn serde_roundtrip_stop_type() {
        let types = vec![StopType::Pickup, StopType::Dropoff, StopType::Transfer];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: StopType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, t);
        }
    }

    #[test]
    fn serde_roundtrip_waypoint() {
        let wp = Waypoint { lat: 32.95, lon: -96.73, label: Some("Home".into()) };
        let json = serde_json::to_string(&wp).unwrap();
        let back: Waypoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, wp);
    }

    #[test]
    fn serde_roundtrip_waypoint_no_label() {
        let wp = Waypoint { lat: 0.0, lon: 0.0, label: None };
        let json = serde_json::to_string(&wp).unwrap();
        let back: Waypoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, wp);
    }

    #[test]
    fn serde_roundtrip_vehicle() {
        let v = valid_vehicle();
        let json = serde_json::to_string(&v).unwrap();
        let back: Vehicle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn serde_roundtrip_route() {
        let r = valid_route();
        let json = serde_json::to_string(&r).unwrap();
        let back: Route = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_stop() {
        let s = valid_stop();
        let json = serde_json::to_string(&s).unwrap();
        let back: Stop = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    // ── validate_vehicle: id ────────────────────────────────────────────

    #[test]
    fn valid_vehicle_passes() {
        assert_valid(validate_vehicle(valid_vehicle()));
    }

    #[test]
    fn vehicle_empty_id_rejected() {
        let mut v = valid_vehicle();
        v.id = String::new();
        assert_invalid(validate_vehicle(v), "Vehicle ID cannot be empty");
    }

    #[test]
    fn vehicle_whitespace_id_rejected() {
        let mut v = valid_vehicle();
        v.id = " ".into();
        assert_invalid(validate_vehicle(v), "Vehicle ID cannot be empty");
    }

    // ── validate_vehicle: capacity_kg ───────────────────────────────────

    #[test]
    fn vehicle_negative_capacity_rejected() {
        let mut v = valid_vehicle();
        v.capacity_kg = -1.0;
        assert_invalid(validate_vehicle(v), "Capacity cannot be negative");
    }

    #[test]
    fn vehicle_barely_negative_capacity_rejected() {
        let mut v = valid_vehicle();
        v.capacity_kg = -0.001;
        assert_invalid(validate_vehicle(v), "Capacity cannot be negative");
    }

    #[test]
    fn vehicle_zero_capacity_valid() {
        let mut v = valid_vehicle();
        v.capacity_kg = 0.0;
        assert_valid(validate_vehicle(v));
    }

    #[test]
    fn vehicle_large_capacity_valid() {
        let mut v = valid_vehicle();
        v.capacity_kg = 50000.0;
        assert_valid(validate_vehicle(v));
    }

    // ── validate_vehicle: type variants ─────────────────────────────────

    #[test]
    fn all_vehicle_types_valid() {
        for vt in [VehicleType::Car, VehicleType::Van, VehicleType::Bike,
                    VehicleType::Bus, VehicleType::Cargo, VehicleType::ElectricScooter] {
            let mut v = valid_vehicle();
            v.vehicle_type = vt;
            assert_valid(validate_vehicle(v));
        }
    }

    // ── validate_vehicle: status variants ───────────────────────────────

    #[test]
    fn all_vehicle_statuses_valid() {
        for status in [VehicleStatus::Available, VehicleStatus::InUse,
                       VehicleStatus::Maintenance, VehicleStatus::Retired] {
            let mut v = valid_vehicle();
            v.status = status;
            assert_valid(validate_vehicle(v));
        }
    }

    // ── validate_vehicle: combined invalid ──────────────────────────────

    #[test]
    fn vehicle_empty_id_negative_capacity_rejects_id_first() {
        let mut v = valid_vehicle();
        v.id = String::new();
        v.capacity_kg = -10.0;
        assert_invalid(validate_vehicle(v), "Vehicle ID cannot be empty");
    }

    // ── validate_vehicle: passengers not validated ──────────────────────

    #[test]
    fn vehicle_zero_passengers_valid() {
        let mut v = valid_vehicle();
        v.capacity_passengers = 0;
        assert_valid(validate_vehicle(v));
    }

    // ── validate_route: id ──────────────────────────────────────────────

    #[test]
    fn valid_route_passes() {
        assert_valid(validate_route(valid_route()));
    }

    #[test]
    fn route_empty_id_rejected() {
        let mut r = valid_route();
        r.id = String::new();
        assert_invalid(validate_route(r), "Route ID cannot be empty");
    }

    #[test]
    fn route_whitespace_id_rejected() {
        let mut r = valid_route();
        r.id = " ".into();
        assert_invalid(validate_route(r), "Route ID cannot be empty");
    }

    // ── validate_route: name ────────────────────────────────────────────

    #[test]
    fn route_empty_name_rejected() {
        let mut r = valid_route();
        r.name = String::new();
        assert_invalid(validate_route(r), "Route name cannot be empty");
    }

    #[test]
    fn route_whitespace_name_rejected() {
        let mut r = valid_route();
        r.name = "  ".into();
        assert_invalid(validate_route(r), "Route name cannot be empty");
    }

    // ── validate_route: waypoints ───────────────────────────────────────

    #[test]
    fn route_zero_waypoints_rejected() {
        let mut r = valid_route();
        r.waypoints = vec![];
        assert_invalid(validate_route(r), "Route must have at least 2 waypoints");
    }

    #[test]
    fn route_single_waypoint_rejected() {
        let mut r = valid_route();
        r.waypoints = vec![Waypoint { lat: 32.95, lon: -96.73, label: None }];
        assert_invalid(validate_route(r), "Route must have at least 2 waypoints");
    }

    #[test]
    fn route_two_waypoints_valid() {
        let mut r = valid_route();
        r.waypoints = vec![
            Waypoint { lat: 32.95, lon: -96.73, label: None },
            Waypoint { lat: 32.96, lon: -96.74, label: None },
        ];
        assert_valid(validate_route(r));
    }

    #[test]
    fn route_many_waypoints_valid() {
        let mut r = valid_route();
        r.waypoints = (0..100).map(|i| Waypoint {
            lat: 32.0 + (i as f64) * 0.01,
            lon: -96.0 + (i as f64) * 0.01,
            label: Some(format!("wp_{i}")),
        }).collect();
        assert_valid(validate_route(r));
    }

    // ── validate_route: distance_km ─────────────────────────────────────

    #[test]
    fn route_zero_distance_rejected() {
        let mut r = valid_route();
        r.distance_km = 0.0;
        assert_invalid(validate_route(r), "Distance must be positive");
    }

    #[test]
    fn route_negative_distance_rejected() {
        let mut r = valid_route();
        r.distance_km = -5.0;
        assert_invalid(validate_route(r), "Distance must be positive");
    }

    #[test]
    fn route_barely_positive_distance_valid() {
        let mut r = valid_route();
        r.distance_km = 0.001;
        assert_valid(validate_route(r));
    }

    #[test]
    fn route_large_distance_valid() {
        let mut r = valid_route();
        r.distance_km = 50000.0;
        assert_valid(validate_route(r));
    }

    // ── validate_route: mode variants ───────────────────────────────────

    #[test]
    fn route_all_transport_modes_valid() {
        for mode in [TransportMode::Driving, TransportMode::Cycling,
                     TransportMode::Walking, TransportMode::Transit, TransportMode::Mixed] {
            let mut r = valid_route();
            r.mode = mode;
            assert_valid(validate_route(r));
        }
    }

    // ── validate_route: estimated_minutes not validated ──────────────────

    #[test]
    fn route_zero_minutes_valid() {
        let mut r = valid_route();
        r.estimated_minutes = 0;
        assert_valid(validate_route(r));
    }

    // ── validate_route: combined invalid ────────────────────────────────

    #[test]
    fn route_empty_id_rejects_before_empty_name() {
        let mut r = valid_route();
        r.id = String::new();
        r.name = String::new();
        assert_invalid(validate_route(r), "Route ID cannot be empty");
    }

    #[test]
    fn route_empty_name_rejects_before_single_waypoint() {
        let mut r = valid_route();
        r.name = String::new();
        r.waypoints = vec![Waypoint { lat: 0.0, lon: 0.0, label: None }];
        assert_invalid(validate_route(r), "Route name cannot be empty");
    }

    // ── validate_stop: name ─────────────────────────────────────────────

    #[test]
    fn valid_stop_passes() {
        assert_valid(validate_stop(valid_stop()));
    }

    #[test]
    fn stop_empty_name_rejected() {
        let mut s = valid_stop();
        s.name = String::new();
        assert_invalid(validate_stop(s), "Stop name cannot be empty");
    }

    #[test]
    fn stop_whitespace_name_rejected() {
        let mut s = valid_stop();
        s.name = " ".into();
        assert_invalid(validate_stop(s), "Stop name cannot be empty");
    }

    // ── validate_stop: latitude ─────────────────────────────────────────

    #[test]
    fn stop_lat_too_low_rejected() {
        let mut s = valid_stop();
        s.location_lat = -91.0;
        assert_invalid(validate_stop(s), "Latitude must be between -90 and 90");
    }

    #[test]
    fn stop_lat_too_high_rejected() {
        let mut s = valid_stop();
        s.location_lat = 91.0;
        assert_invalid(validate_stop(s), "Latitude must be between -90 and 90");
    }

    #[test]
    fn stop_lat_at_boundary_neg90_valid() {
        let mut s = valid_stop();
        s.location_lat = -90.0;
        assert_valid(validate_stop(s));
    }

    #[test]
    fn stop_lat_at_boundary_pos90_valid() {
        let mut s = valid_stop();
        s.location_lat = 90.0;
        assert_valid(validate_stop(s));
    }

    #[test]
    fn stop_lat_zero_valid() {
        let mut s = valid_stop();
        s.location_lat = 0.0;
        assert_valid(validate_stop(s));
    }

    // ── validate_stop: longitude ────────────────────────────────────────

    #[test]
    fn stop_lon_too_low_rejected() {
        let mut s = valid_stop();
        s.location_lon = -181.0;
        assert_invalid(validate_stop(s), "Longitude must be between -180 and 180");
    }

    #[test]
    fn stop_lon_too_high_rejected() {
        let mut s = valid_stop();
        s.location_lon = 181.0;
        assert_invalid(validate_stop(s), "Longitude must be between -180 and 180");
    }

    #[test]
    fn stop_lon_at_boundary_neg180_valid() {
        let mut s = valid_stop();
        s.location_lon = -180.0;
        assert_valid(validate_stop(s));
    }

    #[test]
    fn stop_lon_at_boundary_pos180_valid() {
        let mut s = valid_stop();
        s.location_lon = 180.0;
        assert_valid(validate_stop(s));
    }

    // ── validate_stop: stop_type variants ───────────────────────────────

    #[test]
    fn stop_all_types_valid() {
        for st in [StopType::Pickup, StopType::Dropoff, StopType::Transfer] {
            let mut s = valid_stop();
            s.stop_type = st;
            assert_valid(validate_stop(s));
        }
    }

    // ── validate_stop: optional scheduled_time ──────────────────────────

    #[test]
    fn stop_no_scheduled_time_valid() {
        let mut s = valid_stop();
        s.scheduled_time = None;
        assert_valid(validate_stop(s));
    }

    #[test]
    fn stop_with_scheduled_time_valid() {
        let mut s = valid_stop();
        s.scheduled_time = Some(1700000000);
        assert_valid(validate_stop(s));
    }

    // ── validate_stop: combined invalid ─────────────────────────────────

    #[test]
    fn stop_empty_name_with_invalid_lat_rejects_name_first() {
        let mut s = valid_stop();
        s.name = String::new();
        s.location_lat = 91.0;
        assert_invalid(validate_stop(s), "Stop name cannot be empty");
    }

    #[test]
    fn stop_invalid_lat_with_invalid_lon_rejects_lat_first() {
        let mut s = valid_stop();
        s.location_lat = 91.0;
        s.location_lon = 181.0;
        assert_invalid(validate_stop(s), "Latitude must be between -90 and 90");
    }

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_vehicles".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::AllVehicles
            | LinkTypes::AllRoutes
            | LinkTypes::OwnerToVehicle
            | LinkTypes::RouteToStop
            | LinkTypes::VehicleToRoute => 256,
        };
        let name = match link_type {
            LinkTypes::AllVehicles => "AllVehicles",
            LinkTypes::AllRoutes => "AllRoutes",
            LinkTypes::OwnerToVehicle => "OwnerToVehicle",
            LinkTypes::RouteToStop => "RouteToStop",
            LinkTypes::VehicleToRoute => "VehicleToRoute",
        };
        if tag.0.len() > max {
            ValidateCallbackResult::Invalid(
                format!("{} link tag too long (max {} bytes)", name, max),
            )
        } else {
            ValidateCallbackResult::Valid
        }
    }

    #[test]
    fn test_link_all_vehicles_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllVehicles, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_vehicles_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllVehicles, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_all_routes_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllRoutes, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_routes_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllRoutes, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_owner_to_vehicle_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::OwnerToVehicle, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_owner_to_vehicle_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::OwnerToVehicle, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_route_to_stop_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::RouteToStop, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_route_to_stop_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::RouteToStop, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_vehicle_to_route_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::VehicleToRoute, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_vehicle_to_route_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::VehicleToRoute, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
