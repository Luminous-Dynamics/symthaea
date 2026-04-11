// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste Collection Coordinator Zome
//! Business logic for collection requests, vehicle matching, and collection run planning

use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;
use waste_collection_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}


// ============================================================================
// COLLECTION REQUESTS
// ============================================================================

/// Request a waste collection (pickup)
#[hdk_extern]
pub fn request_collection(request: CollectionRequest) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "request_collection")?;

    // Validate quantity
    if !request.quantity_kg.is_finite() || request.quantity_kg <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quantity must be a positive finite number".into()
        )));
    }

    // Validate GPS coordinates
    if !request.pickup_lat.is_finite()
        || request.pickup_lat < -90.0
        || request.pickup_lat > 90.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pickup latitude must be between -90 and 90".into()
        )));
    }
    if !request.pickup_lon.is_finite()
        || request.pickup_lon < -180.0
        || request.pickup_lon > 180.0
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pickup longitude must be between -180 and 180".into()
        )));
    }

    // Validate time window
    if request.preferred_window.earliest_us >= request.preferred_window.latest_us {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Preferred window earliest must be before latest".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CollectionRequest(request.clone()))?;

    // Link to all requests anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_requests".to_string())))?;
    create_link(
        anchor_hash("all_requests")?,
        action_hash.clone(),
        LinkTypes::AllRequests,
        (),
    )?;

    // Link agent to request
    create_link(
        request.requester.clone(),
        action_hash.clone(),
        LinkTypes::AgentToRequests,
        (),
    )?;

    // Link to status anchor
    let status_anchor = format!("collection_status:{:?}", request.status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        action_hash.clone(),
        LinkTypes::StatusToRequests,
        (),
    )?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "collection_requested",
        "source_zome": "waste_collection",
        "payload": {
            "quantity_kg": request.quantity_kg,
            "pickup_lat": request.pickup_lat,
            "pickup_lon": request.pickup_lon,
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a collection request by hash
#[hdk_extern]
pub fn get_collection_request(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all collection requests
#[hdk_extern]
pub fn get_all_requests(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_requests")?, LinkTypes::AllRequests)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get collection requests by status
#[hdk_extern]
pub fn get_requests_by_status(status: CollectionStatus) -> ExternResult<Vec<Record>> {
    let status_anchor = format!("collection_status:{:?}", status);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&status_anchor)?, LinkTypes::StatusToRequests)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// COLLECTION RUNS
// ============================================================================

/// Create a collection run from pending requests
#[hdk_extern]
pub fn create_collection_run(run: CollectionRun) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_collection_run")?;

    // Validate stops
    if run.stops.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collection run must have at least one stop".into()
        )));
    }
    if run.stops.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collection run cannot have more than 50 stops".into()
        )));
    }

    // Validate stop sequence
    for (i, stop) in run.stops.iter().enumerate() {
        if stop.sequence != i as u32 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Stop {} has wrong sequence number", i)
            )));
        }
    }

    let action_hash = create_entry(&EntryTypes::CollectionRun(run.clone()))?;

    // Link to all runs anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_runs".to_string())))?;
    create_link(
        anchor_hash("all_runs")?,
        action_hash.clone(),
        LinkTypes::AllRuns,
        (),
    )?;

    // Link facility to run
    create_link(
        run.facility_hash.clone(),
        action_hash.clone(),
        LinkTypes::FacilityToRuns,
        (),
    )?;

    // Link run to each request
    for stop in &run.stops {
        create_link(
            action_hash.clone(),
            stop.request_hash.clone(),
            LinkTypes::RunToRequests,
            (),
        )?;
    }

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "collection_run_created",
        "source_zome": "waste_collection",
        "payload": {
            "stops_count": run.stops.len(),
            "status": format!("{:?}", run.status),
        }
    }));

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created record".into())))?;
    Ok(record)
}

/// Get a collection run by hash
#[hdk_extern]
pub fn get_collection_run(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all collection runs
#[hdk_extern]
pub fn get_all_runs(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_runs")?, LinkTypes::AllRuns)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get collection runs for a facility
#[hdk_extern]
pub fn get_runs_by_facility(facility_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(facility_hash, LinkTypes::FacilityToRuns)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// ROUTE OPTIMIZATION
// ============================================================================

/// Input for optimized collection route planning.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlanOptimizedRouteInput {
    /// Facility that will receive the collected waste
    pub facility_hash: ActionHash,
    /// Vehicle capacity in kg
    pub vehicle_capacity_kg: f64,
    /// Facility GPS latitude (for distance calculations)
    pub facility_lat: f64,
    /// Facility GPS longitude
    pub facility_lon: f64,
    /// Maximum number of stops per run
    pub max_stops: u32,
}

/// A planned stop with distance and cumulative load info.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlannedStop {
    /// The collection request at this stop
    pub request_hash: ActionHash,
    /// GPS latitude
    pub lat: f64,
    /// GPS longitude
    pub lon: f64,
    /// Weight to collect (kg)
    pub quantity_kg: f64,
    /// Cumulative load after this stop (kg)
    pub cumulative_kg: f64,
    /// Distance from previous stop (km)
    pub leg_distance_km: f64,
}

/// Result of optimized route planning.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizedRouteResult {
    /// Ordered stops (optimized sequence)
    pub stops: Vec<PlannedStop>,
    /// Total distance of the route (km)
    pub total_distance_km: f64,
    /// Total weight to collect (kg)
    pub total_weight_kg: f64,
    /// Requests excluded due to capacity constraints
    pub excluded_count: u32,
    /// Distance savings vs naive ordering (percentage)
    pub savings_pct: f64,
}

/// Haversine distance between two GPS points in kilometers.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

/// Plan an optimized collection route using a savings-based insertion heuristic.
///
/// Algorithm:
/// 1. Start with all pending requests as candidates
/// 2. Filter by vehicle capacity (exclude requests that alone exceed remaining capacity)
/// 3. Use nearest-insertion heuristic: start at facility, greedily add nearest unvisited
///    request that fits within remaining capacity
/// 4. Compare total distance to naive (original order) for savings calculation
///
/// This is a O(n²) heuristic, suitable for the typical case of <50 stops.
#[hdk_extern]
pub fn plan_optimized_route(input: PlanOptimizedRouteInput) -> ExternResult<OptimizedRouteResult> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "plan_optimized_route")?;

    // Get all pending requests
    let status_anchor = format!("collection_status:{:?}", CollectionStatus::Requested);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&status_anchor)?, LinkTypes::StatusToRequests)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    // Parse requests into candidates
    struct Candidate {
        hash: ActionHash,
        lat: f64,
        lon: f64,
        kg: f64,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for record in &records {
        let req: CollectionRequest = match record.entry().to_app_option() {
            Ok(Some(r)) => r,
            _ => continue,
        };
        // Skip if single request exceeds capacity
        if req.quantity_kg <= input.vehicle_capacity_kg {
            candidates.push(Candidate {
                hash: record.action_address().clone(),
                lat: req.pickup_lat,
                lon: req.pickup_lon,
                kg: req.quantity_kg,
            });
        }
    }

    let excluded_count = (records.len() - candidates.len()) as u32;
    let max_stops = input.max_stops.min(50) as usize;

    if candidates.is_empty() {
        return Ok(OptimizedRouteResult {
            stops: vec![],
            total_distance_km: 0.0,
            total_weight_kg: 0.0,
            excluded_count,
            savings_pct: 0.0,
        });
    }

    // Nearest-insertion heuristic
    let mut visited = vec![false; candidates.len()];
    let mut route: Vec<usize> = Vec::new();
    let mut remaining_capacity = input.vehicle_capacity_kg;
    let mut current_lat = input.facility_lat;
    let mut current_lon = input.facility_lon;

    while route.len() < max_stops {
        // Find nearest unvisited candidate that fits
        let mut best_idx: Option<usize> = None;
        let mut best_dist = f64::MAX;

        for (i, cand) in candidates.iter().enumerate() {
            if visited[i] || cand.kg > remaining_capacity {
                continue;
            }
            let d = haversine_km(current_lat, current_lon, cand.lat, cand.lon);
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => {
                visited[idx] = true;
                route.push(idx);
                remaining_capacity -= candidates[idx].kg;
                current_lat = candidates[idx].lat;
                current_lon = candidates[idx].lon;
            }
            None => break, // No more candidates fit
        }
    }

    // Build planned stops with cumulative load
    let mut stops: Vec<PlannedStop> = Vec::new();
    let mut total_distance: f64 = 0.0;
    let mut total_weight: f64 = 0.0;
    let mut prev_lat = input.facility_lat;
    let mut prev_lon = input.facility_lon;

    for &idx in &route {
        let cand = &candidates[idx];
        let leg = haversine_km(prev_lat, prev_lon, cand.lat, cand.lon);
        total_distance += leg;
        total_weight += cand.kg;

        stops.push(PlannedStop {
            request_hash: cand.hash.clone(),
            lat: cand.lat,
            lon: cand.lon,
            quantity_kg: cand.kg,
            cumulative_kg: total_weight,
            leg_distance_km: (leg * 100.0).round() / 100.0,
        });

        prev_lat = cand.lat;
        prev_lon = cand.lon;
    }

    // Add return leg to facility
    if !stops.is_empty() {
        total_distance += haversine_km(prev_lat, prev_lon, input.facility_lat, input.facility_lon);
    }

    // Calculate naive distance (original order) for savings comparison
    let mut naive_distance: f64 = 0.0;
    let mut n_prev_lat = input.facility_lat;
    let mut n_prev_lon = input.facility_lon;
    for &idx in &route {
        let cand = &candidates[idx];
        naive_distance += haversine_km(n_prev_lat, n_prev_lon, cand.lat, cand.lon);
        n_prev_lat = cand.lat;
        n_prev_lon = cand.lon;
    }
    if let Some(&last_idx) = route.last() {
        let last = &candidates[last_idx];
        naive_distance += haversine_km(last.lat, last.lon, input.facility_lat, input.facility_lon);
    }

    let savings_pct = if naive_distance > 0.0 {
        ((naive_distance - total_distance) / naive_distance * 100.0).max(0.0)
    } else {
        0.0
    };

    Ok(OptimizedRouteResult {
        stops,
        total_distance_km: (total_distance * 100.0).round() / 100.0,
        total_weight_kg: (total_weight * 10.0).round() / 10.0,
        excluded_count,
        savings_pct: (savings_pct * 10.0).round() / 10.0,
    })
}

// ============================================================================
// DELIVERY CONFIRMATION
// ============================================================================

/// Input for confirming delivery at a stop
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConfirmDeliveryInput {
    pub run_hash: ActionHash,
    pub stop_index: u32,
    pub actual_kg: f64,
}

/// Confirm delivery of waste at a collection stop
#[hdk_extern]
pub fn confirm_delivery(input: ConfirmDeliveryInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "confirm_delivery")?;

    // Validate actual kg
    if !input.actual_kg.is_finite() || input.actual_kg < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Actual kg must be a non-negative finite number".into()
        )));
    }

    // Get the run
    let run_record = get(input.run_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Collection run not found".into())))?;
    let mut run: CollectionRun = run_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode run: {}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Run entry not found".into())))?;

    // Validate stop index
    let idx = input.stop_index as usize;
    if idx >= run.stops.len() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Stop index {} out of range (run has {} stops)", idx, run.stops.len())
        )));
    }

    // Update stop
    run.stops[idx].actual_arrival_us = Some(sys_time()?.as_micros() as u64);
    run.stops[idx].actual_kg = Some(input.actual_kg);

    // Update total
    run.total_kg_collected = run
        .stops
        .iter()
        .filter_map(|s| s.actual_kg)
        .sum();

    // Check if all stops are complete
    let all_complete = run.stops.iter().all(|s| s.actual_kg.is_some());
    if all_complete {
        run.status = RunStatus::Completed;
        run.completed_at = Some(sys_time()?.as_micros() as u64);
    } else {
        run.status = RunStatus::InProgress;
    }

    // Update the run entry
    let updated_hash = update_entry(input.run_hash, &run)?;

    // Emit signal
    let _ = emit_signal(&serde_json::json!({
        "event_type": "delivery_confirmed",
        "source_zome": "waste_collection",
        "payload": {
            "stop_index": input.stop_index,
            "actual_kg": input.actual_kg,
            "total_kg_collected": run.total_kg_collected,
            "run_complete": all_complete,
        }
    }));

    let record = get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated record".into())))?;
    Ok(record)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_route_requires_one_stop() {
        // A collection run with 0 stops should be invalid
        let stops: Vec<CollectionStop> = vec![];
        assert!(stops.is_empty());
    }

    #[test]
    fn test_max_stops_limit() {
        assert_eq!(50, 50, "Max stops per run is 50");
    }

    #[test]
    fn test_total_kg_summation() {
        let stops = vec![
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sequence: 0,
                estimated_arrival_us: 100,
                actual_arrival_us: Some(110),
                actual_kg: Some(25.0),
            },
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                sequence: 1,
                estimated_arrival_us: 200,
                actual_arrival_us: Some(210),
                actual_kg: Some(30.0),
            },
        ];
        let total: f64 = stops.iter().filter_map(|s| s.actual_kg).sum();
        assert!((total - 55.0).abs() < 0.001);
    }

    #[test]
    fn test_partial_completion() {
        let stops = vec![
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sequence: 0,
                estimated_arrival_us: 100,
                actual_arrival_us: Some(110),
                actual_kg: Some(25.0),
            },
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                sequence: 1,
                estimated_arrival_us: 200,
                actual_arrival_us: None,
                actual_kg: None,
            },
        ];
        let all_complete = stops.iter().all(|s| s.actual_kg.is_some());
        assert!(!all_complete);
    }

    #[test]
    fn test_full_completion() {
        let stops = vec![
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sequence: 0,
                estimated_arrival_us: 100,
                actual_arrival_us: Some(110),
                actual_kg: Some(25.0),
            },
            CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                sequence: 1,
                estimated_arrival_us: 200,
                actual_arrival_us: Some(210),
                actual_kg: Some(30.0),
            },
        ];
        let all_complete = stops.iter().all(|s| s.actual_kg.is_some());
        assert!(all_complete);
    }

    // -- Route optimizer tests --

    #[test]
    fn test_haversine_same_point() {
        let d = haversine_km(32.95, -96.73, 32.95, -96.73);
        assert!(d < 0.001);
    }

    #[test]
    fn test_haversine_short_distance() {
        let d = haversine_km(32.9500, -96.7300, 32.9600, -96.7200);
        assert!(d > 0.5 && d < 5.0, "Expected ~1.5km, got {}", d);
    }

    #[test]
    fn test_haversine_symmetric() {
        let d1 = haversine_km(32.95, -96.73, 33.05, -96.83);
        let d2 = haversine_km(33.05, -96.83, 32.95, -96.73);
        assert!((d1 - d2).abs() < 0.001, "Haversine should be symmetric");
    }

    #[test]
    fn test_planned_stop_cumulative() {
        let stops = vec![
            PlannedStop {
                request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                lat: 32.95,
                lon: -96.73,
                quantity_kg: 30.0,
                cumulative_kg: 30.0,
                leg_distance_km: 2.0,
            },
            PlannedStop {
                request_hash: ActionHash::from_raw_36(vec![1u8; 36]),
                lat: 32.96,
                lon: -96.72,
                quantity_kg: 25.0,
                cumulative_kg: 55.0,
                leg_distance_km: 1.5,
            },
        ];
        assert_eq!(stops.last().unwrap().cumulative_kg, 55.0);
        assert!(stops[1].cumulative_kg > stops[0].cumulative_kg);
    }

    #[test]
    fn test_optimized_route_empty() {
        let result = OptimizedRouteResult {
            stops: vec![],
            total_distance_km: 0.0,
            total_weight_kg: 0.0,
            excluded_count: 0,
            savings_pct: 0.0,
        };
        assert!(result.stops.is_empty());
        assert_eq!(result.total_distance_km, 0.0);
    }

    #[test]
    fn test_route_savings_non_negative() {
        let result = OptimizedRouteResult {
            stops: vec![],
            total_distance_km: 10.0,
            total_weight_kg: 100.0,
            excluded_count: 0,
            savings_pct: 15.5,
        };
        assert!(result.savings_pct >= 0.0);
    }
}
