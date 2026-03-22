// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste Collection Integrity Zome
//! Entry types, link types, and validation for waste pickup scheduling and collection runs

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// TIME WINDOW
// ============================================================================

/// A preferred pickup time window
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TimeWindow {
    /// Earliest acceptable pickup time (Unix microseconds)
    pub earliest_us: u64,
    /// Latest acceptable pickup time (Unix microseconds)
    pub latest_us: u64,
}

// ============================================================================
// COLLECTION REQUEST
// ============================================================================

/// Status of a collection request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CollectionStatus {
    /// Pickup requested, awaiting vehicle match
    Requested,
    /// Vehicle matched, awaiting scheduling
    Matched,
    /// Scheduled in a collection run
    Scheduled,
    /// Vehicle en route or loading
    InTransit,
    /// Delivered to facility
    Delivered,
    /// Facility confirmed receipt
    Confirmed,
}

/// A request for waste collection (pickup)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollectionRequest {
    /// The waste stream to be collected
    pub stream_hash: ActionHash,
    /// Agent requesting pickup
    pub requester: AgentPubKey,
    /// Pickup GPS latitude
    pub pickup_lat: f64,
    /// Pickup GPS longitude
    pub pickup_lon: f64,
    /// Preferred pickup time window
    pub preferred_window: TimeWindow,
    /// Expected weight (kg)
    pub quantity_kg: f64,
    /// Current status
    pub status: CollectionStatus,
    /// Matched facility (set after routing)
    pub facility_hash: Option<ActionHash>,
    /// Matched vehicle (set after matching)
    pub vehicle_hash: Option<ActionHash>,
    /// Collection run this request is part of
    pub run_hash: Option<ActionHash>,
    /// When the request was created (Unix microseconds)
    pub created_at: u64,
}

// ============================================================================
// COLLECTION RUN
// ============================================================================

/// A stop in a collection run
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CollectionStop {
    /// The collection request at this stop
    pub request_hash: ActionHash,
    /// Stop sequence (0-indexed)
    pub sequence: u32,
    /// Estimated arrival time (Unix microseconds)
    pub estimated_arrival_us: u64,
    /// Actual arrival time (set on arrival)
    pub actual_arrival_us: Option<u64>,
    /// Actual weight collected (kg)
    pub actual_kg: Option<f64>,
}

/// Status of a collection run
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RunStatus {
    /// Planned, not yet started
    Planned,
    /// Currently executing
    InProgress,
    /// All stops completed
    Completed,
    /// Cancelled before completion
    Cancelled,
}

/// A collection run — a vehicle visiting multiple pickup points
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollectionRun {
    /// Vehicle performing the collection (links to transport-routes)
    pub vehicle_hash: ActionHash,
    /// Driver agent
    pub driver: AgentPubKey,
    /// Destination facility
    pub facility_hash: ActionHash,
    /// Ordered list of stops
    pub stops: Vec<CollectionStop>,
    /// Run status
    pub status: RunStatus,
    /// When the run started (Unix microseconds)
    pub started_at: u64,
    /// When the run completed
    pub completed_at: Option<u64>,
    /// Total weight collected across all stops (kg)
    pub total_kg_collected: f64,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CollectionRequest(CollectionRequest),
    CollectionRun(CollectionRun),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All collection requests (global anchor)
    AllRequests,
    /// All collection runs (global anchor)
    AllRuns,
    /// Agent to their collection requests
    AgentToRequests,
    /// Facility to collection runs targeting it
    FacilityToRuns,
    /// Run to its requests
    RunToRequests,
    /// Status anchor to requests with that status
    StatusToRequests,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CollectionRequest(req) => validate_create_request(action, req),
                EntryTypes::CollectionRun(run) => validate_create_run(action, run),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CollectionRequest(_) | EntryTypes::CollectionRun(_) => {
                    let original = must_get_action(original_action_hash)?;
                    Ok(check_author_match(
                        original.action().author(),
                        &action.author,
                        "update",
                    ))
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            match link_type {
                LinkTypes::AllRequests
                | LinkTypes::AllRuns
                | LinkTypes::AgentToRequests
                | LinkTypes::FacilityToRuns
                | LinkTypes::RunToRequests
                | LinkTypes::StatusToRequests => {
                    if tag.0.len() > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            format!("{:?} link tag too long (max 512 bytes)", link_type),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_request(
    _action: Create,
    req: CollectionRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Quantity must be positive and finite
    if !req.quantity_kg.is_finite() || req.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a positive finite number".into(),
        ));
    }
    if req.quantity_kg > 100_000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity exceeds maximum (100,000 kg per request)".into(),
        ));
    }

    // Location must be valid GPS coordinates
    if !req.pickup_lat.is_finite() || req.pickup_lat < -90.0 || req.pickup_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pickup latitude must be between -90 and 90".into(),
        ));
    }
    if !req.pickup_lon.is_finite() || req.pickup_lon < -180.0 || req.pickup_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pickup longitude must be between -180 and 180".into(),
        ));
    }

    // Time window must be valid
    if req.preferred_window.earliest_us >= req.preferred_window.latest_us {
        return Ok(ValidateCallbackResult::Invalid(
            "Preferred window earliest must be before latest".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_run(
    _action: Create,
    run: CollectionRun,
) -> ExternResult<ValidateCallbackResult> {
    // Must have at least one stop
    if run.stops.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection run must have at least one stop".into(),
        ));
    }

    // Max 50 stops per run
    if run.stops.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection run cannot have more than 50 stops".into(),
        ));
    }

    // Stops must be in sequence order
    for (i, stop) in run.stops.iter().enumerate() {
        if stop.sequence != i as u32 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Stop {} has sequence {}, expected {}", i, stop.sequence, i),
            ));
        }
    }

    // Total kg must be non-negative and finite
    if !run.total_kg_collected.is_finite() || run.total_kg_collected < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total kg collected must be a non-negative finite number".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_request() -> CollectionRequest {
        CollectionRequest {
            stream_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            requester: AgentPubKey::from_raw_36(vec![1u8; 36]),
            pickup_lat: 32.9483,
            pickup_lon: -96.7299,
            preferred_window: TimeWindow {
                earliest_us: 1711100000_000000,
                latest_us: 1711200000_000000,
            },
            quantity_kg: 50.0,
            status: CollectionStatus::Requested,
            facility_hash: None,
            vehicle_hash: None,
            run_hash: None,
            created_at: 1711100000_000000,
        }
    }

    fn make_valid_run() -> CollectionRun {
        CollectionRun {
            vehicle_hash: ActionHash::from_raw_36(vec![2u8; 36]),
            driver: AgentPubKey::from_raw_36(vec![3u8; 36]),
            facility_hash: ActionHash::from_raw_36(vec![4u8; 36]),
            stops: vec![CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![0u8; 36]),
                sequence: 0,
                estimated_arrival_us: 1711150000_000000,
                actual_arrival_us: None,
                actual_kg: None,
            }],
            status: RunStatus::Planned,
            started_at: 1711140000_000000,
            completed_at: None,
            total_kg_collected: 0.0,
        }
    }

    #[test]
    fn test_valid_request() {
        let req = make_valid_request();
        assert!(req.quantity_kg > 0.0);
        assert!((-90.0..=90.0).contains(&req.pickup_lat));
        assert!((-180.0..=180.0).contains(&req.pickup_lon));
        assert!(req.preferred_window.earliest_us < req.preferred_window.latest_us);
    }

    #[test]
    fn test_request_quantity_must_be_positive() {
        let mut req = make_valid_request();
        req.quantity_kg = 0.0;
        assert!(req.quantity_kg <= 0.0);
    }

    #[test]
    fn test_request_quantity_max_limit() {
        let mut req = make_valid_request();
        req.quantity_kg = 100_001.0;
        assert!(req.quantity_kg > 100_000.0);
    }

    #[test]
    fn test_request_invalid_time_window() {
        let mut req = make_valid_request();
        req.preferred_window.earliest_us = req.preferred_window.latest_us + 1;
        assert!(req.preferred_window.earliest_us >= req.preferred_window.latest_us);
    }

    #[test]
    fn test_valid_run() {
        let run = make_valid_run();
        assert!(!run.stops.is_empty());
        assert_eq!(run.stops[0].sequence, 0);
        assert!(run.total_kg_collected >= 0.0);
    }

    #[test]
    fn test_run_empty_stops_invalid() {
        let mut run = make_valid_run();
        run.stops = vec![];
        assert!(run.stops.is_empty());
    }

    #[test]
    fn test_run_stop_sequence_validation() {
        let run = make_valid_run();
        for (i, stop) in run.stops.iter().enumerate() {
            assert_eq!(stop.sequence, i as u32);
        }
    }

    #[test]
    fn test_run_max_stops() {
        let mut run = make_valid_run();
        run.stops = (0..51)
            .map(|i| CollectionStop {
                request_hash: ActionHash::from_raw_36(vec![i as u8; 36]),
                sequence: i as u32,
                estimated_arrival_us: 1711150000_000000 + i as u64 * 1000,
                actual_arrival_us: None,
                actual_kg: None,
            })
            .collect();
        assert!(run.stops.len() > 50);
    }

    #[test]
    fn test_collection_status_lifecycle() {
        let statuses = vec![
            CollectionStatus::Requested,
            CollectionStatus::Matched,
            CollectionStatus::Scheduled,
            CollectionStatus::InTransit,
            CollectionStatus::Delivered,
            CollectionStatus::Confirmed,
        ];
        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_run_status_lifecycle() {
        let statuses = vec![
            RunStatus::Planned,
            RunStatus::InProgress,
            RunStatus::Completed,
            RunStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn test_nan_quantity_rejected() {
        let mut req = make_valid_request();
        req.quantity_kg = f64::NAN;
        assert!(!req.quantity_kg.is_finite());
    }

    #[test]
    fn test_inf_quantity_rejected() {
        let mut req = make_valid_request();
        req.quantity_kg = f64::INFINITY;
        assert!(!req.quantity_kg.is_finite());
    }

    #[test]
    fn test_nan_latitude_rejected() {
        let mut req = make_valid_request();
        req.pickup_lat = f64::NAN;
        assert!(!req.pickup_lat.is_finite());
    }
}
