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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_vehicle(v: Vehicle) -> ExternResult<ValidateCallbackResult> {
    if v.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Vehicle ID cannot be empty".into()));
    }
    if v.capacity_kg < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_route(r: Route) -> ExternResult<ValidateCallbackResult> {
    if r.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Route ID cannot be empty".into()));
    }
    if r.name.is_empty() {
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
    if s.name.is_empty() {
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

    #[test]
    fn valid_vehicle_passes() {
        assert_eq!(validate_vehicle(valid_vehicle()).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn vehicle_empty_id_rejected() {
        let mut v = valid_vehicle();
        v.id = String::new();
        assert!(matches!(validate_vehicle(v).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn vehicle_negative_capacity_rejected() {
        let mut v = valid_vehicle();
        v.capacity_kg = -1.0;
        assert!(matches!(validate_vehicle(v).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn vehicle_zero_capacity_valid() {
        let mut v = valid_vehicle();
        v.capacity_kg = 0.0;
        assert_eq!(validate_vehicle(v).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn all_vehicle_types_valid() {
        for vt in [VehicleType::Car, VehicleType::Van, VehicleType::Bike,
                    VehicleType::Bus, VehicleType::Cargo, VehicleType::ElectricScooter] {
            let mut v = valid_vehicle();
            v.vehicle_type = vt;
            assert_eq!(validate_vehicle(v).unwrap(), ValidateCallbackResult::Valid);
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

    #[test]
    fn valid_route_passes() {
        assert_eq!(validate_route(valid_route()).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn route_empty_id_rejected() {
        let mut r = valid_route();
        r.id = String::new();
        assert!(matches!(validate_route(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn route_empty_name_rejected() {
        let mut r = valid_route();
        r.name = String::new();
        assert!(matches!(validate_route(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn route_single_waypoint_rejected() {
        let mut r = valid_route();
        r.waypoints = vec![Waypoint { lat: 32.95, lon: -96.73, label: None }];
        assert!(matches!(validate_route(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn route_zero_distance_rejected() {
        let mut r = valid_route();
        r.distance_km = 0.0;
        assert!(matches!(validate_route(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_stop_passes() {
        let s = Stop {
            route_hash: fake_action_hash(),
            name: "Main St Station".into(),
            location_lat: 32.95,
            location_lon: -96.73,
            scheduled_time: Some(1700000000),
            stop_type: StopType::Pickup,
        };
        assert_eq!(validate_stop(s).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn stop_empty_name_rejected() {
        let s = Stop {
            route_hash: fake_action_hash(),
            name: String::new(),
            location_lat: 32.95,
            location_lon: -96.73,
            scheduled_time: None,
            stop_type: StopType::Dropoff,
        };
        assert!(matches!(validate_stop(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn stop_invalid_lat_rejected() {
        let s = Stop {
            route_hash: fake_action_hash(),
            name: "Bad Stop".into(),
            location_lat: 91.0,
            location_lon: -96.73,
            scheduled_time: None,
            stop_type: StopType::Transfer,
        };
        assert!(matches!(validate_stop(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn stop_invalid_lon_rejected() {
        let s = Stop {
            route_hash: fake_action_hash(),
            name: "Bad Stop".into(),
            location_lat: 32.95,
            location_lon: 181.0,
            scheduled_time: None,
            stop_type: StopType::Transfer,
        };
        assert!(matches!(validate_stop(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
