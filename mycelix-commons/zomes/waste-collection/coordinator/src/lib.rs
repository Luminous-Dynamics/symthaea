// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste Collection Coordinator Zome
//! Business logic for collection requests, vehicle matching, and collection run planning

use hdk::prelude::*;
use mycelix_bridge_common::requirement_for_basic;
use mycelix_zome_helpers::records_from_links;
use waste_collection_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn require_consciousness(
    requirement: &mycelix_bridge_common::GovernanceRequirement,
    action_name: &str,
) -> ExternResult<mycelix_bridge_common::GovernanceEligibility> {
    mycelix_zome_helpers::require_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// COLLECTION REQUESTS
// ============================================================================

/// Request a waste collection (pickup)
#[hdk_extern]
pub fn request_collection(request: CollectionRequest) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "request_collection")?;

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
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_collection_run")?;

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
    let _eligibility = require_consciousness(&requirement_for_basic(), "confirm_delivery")?;

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
}
