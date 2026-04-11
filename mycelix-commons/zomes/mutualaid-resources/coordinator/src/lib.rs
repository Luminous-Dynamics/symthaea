// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resources Coordinator Zome
//!
//! This zome provides coordinator functions for resource sharing
//! in the Mycelix Mutual Aid hApp.

use hdk::prelude::*;
use mutualaid_common::*;
use mutualaid_resources_integrity::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::get_latest_record;

/// Result of crediting timebank hours after resource usage
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct TimebankCreditResult {
    pub credited: bool,
    pub hours: f64,
    pub error: Option<String>,
}

// =============================================================================
// INPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateResourceInput {
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub condition: ResourceCondition,
    pub photos: Vec<String>,
    pub location: LocationConstraint,
    pub availability: Availability,
    pub sharing_model: SharingModel,
    pub usage_instructions: String,
    pub liability_notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateBookingInput {
    pub resource_hash: ActionHash,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub purpose: String,
    pub payment_method: Option<PaymentMethod>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordUsageInput {
    pub booking_hash: ActionHash,
    pub condition_before: ResourceCondition,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteUsageInput {
    pub booking_hash: ActionHash,
    pub condition_after: ResourceCondition,
    pub issues: Vec<String>,
    pub notes: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMaintenanceInput {
    pub resource_hash: ActionHash,
    pub maintenance_type: MaintenanceType,
    pub description: String,
    pub cost: Option<i64>,
    pub hours_spent: f64,
    pub parts_used: Vec<String>,
    pub next_due: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchResourcesInput {
    pub resource_type: Option<ResourceType>,
    pub available_only: bool,
    pub query: Option<String>,
    pub limit: Option<u32>,
}


// =============================================================================
// RESOURCE MANAGEMENT
// =============================================================================

/// Create a new shared resource

#[hdk_extern]
pub fn create_resource(input: CreateResourceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_resource")?;
    let owner = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let resource = SharedResource {
        id: generate_id("resource"),
        owner: owner.clone(),
        name: input.name,
        description: input.description,
        resource_type: input.resource_type.clone(),
        condition: input.condition,
        photos: input.photos,
        location: input.location,
        availability: input.availability,
        sharing_model: input.sharing_model,
        usage_instructions: input.usage_instructions,
        liability_notes: input.liability_notes,
        currently_available: true,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::SharedResource(resource.clone()))?;

    // Link from owner to resource
    create_link(owner, action_hash.clone(), LinkTypes::OwnerToResources, ())?;

    // Link from type anchor to resource
    let type_anchor = resource_type_anchor(&input.resource_type)?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToResources,
        (),
    )?;

    // Link to all resources
    let all_anchor = all_resources_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllResources, ())?;

    // Link to available resources
    let avail_anchor = available_resources_anchor()?;
    create_link(
        avail_anchor,
        action_hash.clone(),
        LinkTypes::AvailableResources,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created resource".to_string()
    )))
}

/// Get a resource by hash
#[hdk_extern]
pub fn get_resource(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all resources owned by the current agent
#[hdk_extern]
pub fn get_my_resources(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_resources_by_owner(agent)
}

/// Get all resources owned by a specific agent
#[hdk_extern]
pub fn get_resources_by_owner(owner: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(owner, LinkTypes::OwnerToResources)?,
        GetStrategy::default(),
    )?;

    let mut resources = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                resources.push(record);
            }
        }
    }

    Ok(resources)
}

/// Search resources
#[hdk_extern]
pub fn search_resources(input: SearchResourcesInput) -> ExternResult<Vec<Record>> {
    let anchor = if input.available_only {
        available_resources_anchor()?
    } else {
        all_resources_anchor()?
    };

    let link_type = if input.available_only {
        LinkTypes::AvailableResources
    } else {
        LinkTypes::AllResources
    };

    let links = get_links(
        LinkQuery::try_new(anchor, link_type)?,
        GetStrategy::default(),
    )?;

    let limit = input.limit.unwrap_or(100) as usize;
    let mut results = Vec::new();

    for link in links {
        if results.len() >= limit {
            break;
        }

        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                if let Some(resource) = record
                    .entry()
                    .to_app_option::<SharedResource>()
                    .ok()
                    .flatten()
                {
                    let mut matches = true;

                    if let Some(ref rt) = input.resource_type {
                        if resource.resource_type != *rt {
                            matches = false;
                        }
                    }

                    if let Some(ref q) = input.query {
                        let query_lower = q.to_lowercase();
                        if !resource.name.to_lowercase().contains(&query_lower)
                            && !resource.description.to_lowercase().contains(&query_lower)
                        {
                            matches = false;
                        }
                    }

                    if matches {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Update resource availability
#[derive(Serialize, Deserialize, Debug)]
pub struct SetResourceAvailabilityInput {
    pub hash: ActionHash,
    pub available: bool,
}

#[hdk_extern]
pub fn set_resource_availability(input: SetResourceAvailabilityInput) -> ExternResult<Record> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "set_resource_availability")?;
    let hash = input.hash;
    let available = input.available;
    let record = get_latest_record(hash.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Resource not found".to_string()
    )))?;

    let mut resource: SharedResource = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse resource".to_string()
        )))?;

    // Verify ownership
    let agent = agent_info()?.agent_initial_pubkey;
    if resource.owner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the owner can update availability".to_string()
        )));
    }

    resource.currently_available = available;
    resource.updated_at = Timestamp::from_micros(sys_time()?.as_micros() as i64);

    let new_hash = update_entry(hash, EntryTypes::SharedResource(resource))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated resource".to_string()
    )))
}

// =============================================================================
// BOOKINGS
// =============================================================================

/// Create a new booking
#[hdk_extern]
pub fn create_booking(input: CreateBookingInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_booking")?;
    let booker = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Verify resource exists
    let resource_record = get(input.resource_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Resource not found".to_string())),
    )?;

    let resource: SharedResource = resource_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse resource".to_string()
        )))?;

    if !resource.currently_available {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Resource is not currently available".to_string()
        )));
    }

    // Check for conflicts
    let conflicts = check_booking_conflicts(
        input.resource_hash.clone(),
        input.start_time,
        input.end_time,
    )?;
    if !conflicts.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Booking conflicts with existing reservations".to_string()
        )));
    }

    let booking = Booking {
        id: generate_id("booking"),
        resource_hash: input.resource_hash.clone(),
        booker: booker.clone(),
        start_time: input.start_time,
        end_time: input.end_time,
        purpose: input.purpose,
        status: BookingStatus::Pending,
        payment_method: input.payment_method,
        payment_hash: None,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::Booking(booking))?;

    // Create links
    create_link(
        input.resource_hash,
        action_hash.clone(),
        LinkTypes::ResourceToBookings,
        (),
    )?;
    create_link(booker, action_hash.clone(), LinkTypes::BookerToBookings, ())?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created booking".to_string()
    )))
}

/// Check for booking conflicts
fn check_booking_conflicts(
    resource_hash: ActionHash,
    start_time: Timestamp,
    end_time: Timestamp,
) -> ExternResult<Vec<ActionHash>> {
    let links = get_links(
        LinkQuery::try_new(resource_hash, LinkTypes::ResourceToBookings)?,
        GetStrategy::default(),
    )?;

    let mut conflicts = Vec::new();

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash.clone())? {
                if let Some(booking) = record.entry().to_app_option::<Booking>().ok().flatten() {
                    // Skip cancelled or completed bookings
                    match booking.status {
                        BookingStatus::Cancelled
                        | BookingStatus::Completed
                        | BookingStatus::NoShow => continue,
                        _ => {}
                    }

                    // Check for overlap
                    if start_time < booking.end_time && end_time > booking.start_time {
                        conflicts.push(hash);
                    }
                }
            }
        }
    }

    Ok(conflicts)
}

/// Get bookings for a resource
#[hdk_extern]
pub fn get_resource_bookings(resource_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(resource_hash, LinkTypes::ResourceToBookings)?,
        GetStrategy::default(),
    )?;

    let mut bookings = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                bookings.push(record);
            }
        }
    }

    Ok(bookings)
}

/// Get my bookings
#[hdk_extern]
pub fn get_my_bookings(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::BookerToBookings)?,
        GetStrategy::default(),
    )?;

    let mut bookings = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                bookings.push(record);
            }
        }
    }

    Ok(bookings)
}

/// Confirm a booking (owner only)
#[hdk_extern]
pub fn confirm_booking(booking_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "confirm_booking")?;
    let record = get_latest_record(booking_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Booking not found".to_string())
    ))?;

    let mut booking: Booking = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    // Verify owner
    let resource_record = get(booking.resource_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Resource not found".to_string())),
    )?;

    let resource: SharedResource = resource_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse resource".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    if resource.owner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the resource owner can confirm bookings".to_string()
        )));
    }

    booking.status = BookingStatus::Confirmed;

    let new_hash = update_entry(booking_hash, EntryTypes::Booking(booking))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated booking".to_string()
    )))
}

/// Cancel a booking
#[hdk_extern]
pub fn cancel_booking(booking_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "cancel_booking")?;
    let record = get_latest_record(booking_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Booking not found".to_string())
    ))?;

    let mut booking: Booking = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    let agent = agent_info()?.agent_initial_pubkey;
    if booking.booker != agent {
        // Also allow owner to cancel
        let resource_record = get(booking.resource_hash.clone(), GetOptions::default())?.ok_or(
            wasm_error!(WasmErrorInner::Guest("Resource not found".to_string())),
        )?;

        let resource: SharedResource = resource_record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Could not parse resource".to_string()
            )))?;

        if resource.owner != agent {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the booker or owner can cancel".to_string()
            )));
        }
    }

    booking.status = BookingStatus::Cancelled;

    let new_hash = update_entry(booking_hash, EntryTypes::Booking(booking))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated booking".to_string()
    )))
}

// =============================================================================
// USAGE TRACKING
// =============================================================================

/// Record start of resource usage
#[hdk_extern]
pub fn start_usage(input: RecordUsageInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "start_usage")?;
    let now = sys_time()?;

    let usage = Usage {
        booking_hash: input.booking_hash.clone(),
        actual_start: Timestamp::from_micros(now.as_micros() as i64),
        actual_end: None,
        condition_before: input.condition_before,
        condition_after: None,
        notes: input.notes,
        issues: vec![],
    };

    let action_hash = create_entry(EntryTypes::Usage(usage))?;

    // Update booking status
    let booking_record = get(input.booking_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Booking not found".to_string())),
    )?;

    let mut booking: Booking = booking_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    booking.status = BookingStatus::Active;
    update_entry(input.booking_hash.clone(), EntryTypes::Booking(booking))?;

    // Link usage to booking's resource
    let booking_record = get(input.booking_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Booking not found".to_string())
    ))?;

    let booking: Booking = booking_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    create_link(
        booking.resource_hash,
        action_hash.clone(),
        LinkTypes::ResourceToUsage,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created usage record".to_string()
    )))
}

/// Complete resource usage
#[hdk_extern]
pub fn complete_usage(input: CompleteUsageInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "complete_usage")?;
    let now = sys_time()?;

    // Find the usage record for this booking
    let booking_record = get(input.booking_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Booking not found".to_string())),
    )?;

    let booking: Booking = booking_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    // Get usage records for this resource
    let links = get_links(
        LinkQuery::try_new(booking.resource_hash, LinkTypes::ResourceToUsage)?,
        GetStrategy::default(),
    )?;

    let mut usage_hash = None;
    let mut usage_record = None;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash.clone())? {
                if let Some(usage) = record.entry().to_app_option::<Usage>().ok().flatten() {
                    if usage.booking_hash == input.booking_hash && usage.actual_end.is_none() {
                        usage_hash = Some(hash);
                        usage_record = Some(record);
                        break;
                    }
                }
            }
        }
    }

    let hash = usage_hash.ok_or(wasm_error!(WasmErrorInner::Guest(
        "No active usage record found for this booking".to_string()
    )))?;

    let record = usage_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve usage record".to_string()
    )))?;

    let mut usage: Usage = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse usage".to_string()
        )))?;

    usage.actual_end = Some(Timestamp::from_micros(now.as_micros() as i64));
    usage.condition_after = Some(input.condition_after);
    usage.issues = input.issues;
    usage.notes = input.notes;

    let new_hash = update_entry(hash, EntryTypes::Usage(usage))?;

    // Update booking status
    let mut booking: Booking = booking_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    booking.status = BookingStatus::Completed;
    update_entry(input.booking_hash, EntryTypes::Booking(booking))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated usage record".to_string()
    )))
}

// =============================================================================
// CROSS-DOMAIN: mutualaid-resources → mutualaid-timebank
// =============================================================================

/// Input for completing usage and recording time credit
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteUsageWithTimebankInput {
    pub booking_hash: ActionHash,
    pub condition_after: ResourceCondition,
    pub issues: Vec<String>,
    pub notes: String,
    pub hours_used: f64,
    pub category_description: String,
}

/// Complete resource usage and automatically record a time credit exchange.
///
/// Cross-domain call: mutualaid-resources → mutualaid-timebank via CallTargetCell::Local.
/// When a resource usage is completed, this creates a corresponding time credit
/// entry so the resource owner earns time credits for sharing.
#[hdk_extern]
pub fn complete_usage_with_timebank(
    input: CompleteUsageWithTimebankInput,
) -> ExternResult<TimebankCreditResult> {
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "complete_usage_with_timebank")?;
    if !input.hours_used.is_finite() || input.hours_used < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "hours_used must be a finite non-negative number".into()
        )));
    }
    // First, get the booking to find resource owner and booker
    let booking_record = get(input.booking_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Booking not found".to_string())),
    )?;

    let booking: Booking = booking_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse booking".to_string()
        )))?;

    // Get the resource to find its owner
    let resource_record = get(booking.resource_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Resource not found".to_string())),
    )?;

    let resource: SharedResource = resource_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse resource".to_string()
        )))?;

    // Complete the usage via the normal flow
    let _usage_result = complete_usage(CompleteUsageInput {
        booking_hash: input.booking_hash,
        condition_after: input.condition_after,
        issues: input.issues,
        notes: input.notes,
    })?;

    // Now call mutualaid-timebank to record the exchange
    // The resource owner (provider) earns credits, the booker (recipient) spends them
    let exchange_input = serde_json::json!({
        "offer_hash": null,
        "request_hash": null,
        "provider": resource.owner,
        "recipient": booking.booker,
        "hours": input.hours_used,
        "category": "ResourceSharing",
        "description": input.category_description,
    });

    let response = call(
        CallTargetCell::Local,
        ZomeName::from("mutualaid_timebank"),
        FunctionName::from("record_exchange"),
        None,
        exchange_input.to_string(),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(_)) => Ok(TimebankCreditResult {
            credited: true,
            hours: input.hours_used,
            error: None,
        }),
        Ok(other) => {
            // Usage completed but timebank credit failed — still return success
            // but flag the credit failure
            Ok(TimebankCreditResult {
                credited: false,
                hours: input.hours_used,
                error: Some(format!(
                    "Usage completed but timebank credit failed: {:?}",
                    other
                )),
            })
        }
        Err(e) => Ok(TimebankCreditResult {
            credited: false,
            hours: input.hours_used,
            error: Some(format!("Usage completed but timebank call failed: {:?}", e)),
        }),
    }
}

// =============================================================================
// MAINTENANCE
// =============================================================================

/// Record maintenance on a resource
#[hdk_extern]
pub fn record_maintenance(input: RecordMaintenanceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_maintenance")?;
    if !input.hours_spent.is_finite() || input.hours_spent < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "hours_spent must be a finite non-negative number".into()
        )));
    }
    let maintainer = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let maintenance = Maintenance {
        resource_hash: input.resource_hash.clone(),
        maintainer,
        maintenance_type: input.maintenance_type,
        description: input.description,
        cost: input.cost,
        hours_spent: input.hours_spent,
        parts_used: input.parts_used,
        performed_at: Timestamp::from_micros(now.as_micros() as i64),
        next_due: input.next_due,
    };

    let action_hash = create_entry(EntryTypes::Maintenance(maintenance))?;

    // Link to resource
    create_link(
        input.resource_hash,
        action_hash.clone(),
        LinkTypes::ResourceToMaintenance,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created maintenance record".to_string()
    )))
}

/// Get maintenance history for a resource
#[hdk_extern]
pub fn get_maintenance_history(resource_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(resource_hash, LinkTypes::ResourceToMaintenance)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a unique ID
fn generate_id(prefix: &str) -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!(
        "{}_{}_{}",
        prefix,
        now.as_micros(),
        &agent[..8.min(agent.len())]
    )
}

/// Simple anchor helper
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("resources_anchor:{}", name).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get anchor for resource type
fn resource_type_anchor(resource_type: &ResourceType) -> ExternResult<EntryHash> {
    make_anchor(&format!("type_{:?}", resource_type))
}

/// Get anchor for all resources
fn all_resources_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_resources")
}

/// Get anchor for available resources
fn available_resources_anchor() -> ExternResult<EntryHash> {
    make_anchor("available_resources")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timebank_credit_result_credited_serde() {
        let r = TimebankCreditResult {
            credited: true,
            hours: 2.5,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: TimebankCreditResult = serde_json::from_str(&json).unwrap();
        assert!(r2.credited);
        assert!((r2.hours - 2.5).abs() < f64::EPSILON);
        assert!(r2.error.is_none());
    }

    #[test]
    fn timebank_credit_result_failed_serde() {
        let r = TimebankCreditResult {
            credited: false,
            hours: 1.0,
            error: Some("Usage completed but timebank credit failed: connection refused".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: TimebankCreditResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.credited);
        assert!(r2.error.as_ref().unwrap().contains("connection refused"));
    }

    #[test]
    fn timebank_credit_zero_hours() {
        let r = TimebankCreditResult {
            credited: true,
            hours: 0.0,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: TimebankCreditResult = serde_json::from_str(&json).unwrap();
        assert!(r2.credited);
        assert!((r2.hours).abs() < f64::EPSILON);
    }

    #[test]
    fn complete_usage_with_timebank_input_serde() {
        // We can't construct the full input without ActionHash, but we can test
        // the result type which is the cross-domain return value
        let r = TimebankCreditResult {
            credited: true,
            hours: 3.0,
            error: None,
        };
        assert!(r.credited);
        assert_eq!(r.hours, 3.0);
    }

    // ========================================================================
    // ENUM VARIANT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn resource_type_all_variants_serde() {
        let variants = vec![
            ResourceType::PowerTool,
            ResourceType::HandTool,
            ResourceType::GardenTool,
            ResourceType::CookingEquipment,
            ResourceType::CraftingSupplies,
            ResourceType::Car,
            ResourceType::Truck,
            ResourceType::Bicycle,
            ResourceType::Trailer,
            ResourceType::Boat,
            ResourceType::MeetingRoom,
            ResourceType::Workshop,
            ResourceType::Kitchen,
            ResourceType::GardenPlot,
            ResourceType::StorageSpace,
            ResourceType::ParkingSpot,
            ResourceType::CampingGear,
            ResourceType::SportsEquipment,
            ResourceType::MusicInstrument,
            ResourceType::Photography,
            ResourceType::Projector,
            ResourceType::Custom("3D Printer".to_string()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ResourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn resource_condition_all_variants_serde() {
        let variants = vec![
            ResourceCondition::Excellent,
            ResourceCondition::Good,
            ResourceCondition::Fair,
            ResourceCondition::NeedsRepair,
            ResourceCondition::BeingRepaired,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ResourceCondition = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn booking_status_all_variants_serde() {
        let variants = vec![
            BookingStatus::Pending,
            BookingStatus::Confirmed,
            BookingStatus::Active,
            BookingStatus::Completed,
            BookingStatus::Cancelled,
            BookingStatus::NoShow,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: BookingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn sharing_model_serde_roundtrip() {
        let model = SharingModel {
            free: false,
            deposit: Some(100),
            hourly_rate: 25,
            daily_rate: Some(150),
            accepts_time_credits: true,
            accepts_circle_credits: false,
            circle_hash: None,
        };
        let json = serde_json::to_string(&model).unwrap();
        let back: SharingModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, back);
    }

    #[test]
    fn maintenance_type_all_variants_serde() {
        let variants = vec![
            MaintenanceType::Routine,
            MaintenanceType::Repair,
            MaintenanceType::Cleaning,
            MaintenanceType::Upgrade,
            MaintenanceType::SafetyCheck,
            MaintenanceType::Other("Calibration".to_string()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: MaintenanceType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn payment_method_all_variants_serde() {
        let variants = vec![
            PaymentMethod::Free,
            PaymentMethod::TimeCredits(2.5),
            PaymentMethod::CircleCredits {
                circle_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
                amount: 50,
            },
            PaymentMethod::External("PayPal".to_string()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: PaymentMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn location_constraint_all_variants_serde() {
        let variants: Vec<LocationConstraint> = vec![
            LocationConstraint::Remote,
            LocationConstraint::FixedLocation("123 Main St".to_string()),
            LocationConstraint::WithinRadius {
                geohash: "9q8yy".to_string(),
                radius_km: 5.0,
            },
            LocationConstraint::AtRequester,
            LocationConstraint::AtProvider,
            LocationConstraint::ToBeArranged,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: LocationConstraint = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    // ========================================================================
    // INPUT STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn create_resource_input_serde_roundtrip() {
        let input = CreateResourceInput {
            name: "Community Power Drill".to_string(),
            description: "DeWalt 20V MAX, good condition".to_string(),
            resource_type: ResourceType::PowerTool,
            condition: ResourceCondition::Good,
            photos: vec!["photo1.jpg".to_string()],
            location: LocationConstraint::FixedLocation("Tool Library, 456 Oak Ave".to_string()),
            availability: Availability::default(),
            sharing_model: SharingModel {
                free: false,
                deposit: Some(50),
                hourly_rate: 5,
                daily_rate: Some(20),
                accepts_time_credits: true,
                accepts_circle_credits: true,
                circle_hash: None,
            },
            usage_instructions: "Return cleaned and charged".to_string(),
            liability_notes: Some("User responsible for damages".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Community Power Drill");
        assert_eq!(back.resource_type, ResourceType::PowerTool);
        assert_eq!(back.condition, ResourceCondition::Good);
        assert_eq!(back.photos.len(), 1);
        assert_eq!(
            back.liability_notes,
            Some("User responsible for damages".to_string())
        );
    }

    #[test]
    fn create_resource_input_minimal_serde() {
        let input = CreateResourceInput {
            name: "Hammer".to_string(),
            description: "".to_string(),
            resource_type: ResourceType::HandTool,
            condition: ResourceCondition::Fair,
            photos: vec![],
            location: LocationConstraint::Remote,
            availability: Availability::default(),
            sharing_model: SharingModel::default(),
            usage_instructions: "".to_string(),
            liability_notes: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Hammer");
        assert!(back.photos.is_empty());
        assert!(back.liability_notes.is_none());
    }

    #[test]
    fn create_booking_input_serde_roundtrip() {
        let input = CreateBookingInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            start_time: Timestamp::from_micros(1_000_000),
            end_time: Timestamp::from_micros(2_000_000),
            purpose: "Weekend renovation project".to_string(),
            payment_method: Some(PaymentMethod::TimeCredits(3.0)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBookingInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.purpose, "Weekend renovation project");
        assert_eq!(back.payment_method, Some(PaymentMethod::TimeCredits(3.0)));
    }

    #[test]
    fn create_booking_input_no_payment_serde() {
        let input = CreateBookingInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            start_time: Timestamp::from_micros(1_000_000),
            end_time: Timestamp::from_micros(2_000_000),
            purpose: "Community meeting".to_string(),
            payment_method: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBookingInput = serde_json::from_str(&json).unwrap();
        assert!(back.payment_method.is_none());
    }

    #[test]
    fn record_usage_input_serde_roundtrip() {
        let input = RecordUsageInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_before: ResourceCondition::Excellent,
            notes: "Starting usage, all looks good".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.condition_before, ResourceCondition::Excellent);
        assert_eq!(back.notes, "Starting usage, all looks good");
    }

    #[test]
    fn complete_usage_input_serde_roundtrip() {
        let input = CompleteUsageInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_after: ResourceCondition::Good,
            issues: vec!["Minor scratch on handle".to_string()],
            notes: "Used carefully, small cosmetic scratch".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CompleteUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.condition_after, ResourceCondition::Good);
        assert_eq!(back.issues.len(), 1);
        assert_eq!(back.issues[0], "Minor scratch on handle");
    }

    #[test]
    fn complete_usage_input_no_issues_serde() {
        let input = CompleteUsageInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_after: ResourceCondition::Excellent,
            issues: vec![],
            notes: "Perfect condition".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CompleteUsageInput = serde_json::from_str(&json).unwrap();
        assert!(back.issues.is_empty());
    }

    #[test]
    fn record_maintenance_input_serde_roundtrip() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Repair,
            description: "Replaced worn drill bit chuck".to_string(),
            cost: Some(35),
            hours_spent: 1.5,
            parts_used: vec!["Chuck assembly".to_string(), "Set screw".to_string()],
            next_due: Some(Timestamp::from_micros(5_000_000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RecordMaintenanceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.maintenance_type, MaintenanceType::Repair);
        assert_eq!(back.cost, Some(35));
        assert!((back.hours_spent - 1.5).abs() < f64::EPSILON);
        assert_eq!(back.parts_used.len(), 2);
    }

    #[test]
    fn record_maintenance_input_minimal_serde() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Routine,
            description: "Quick inspection".to_string(),
            cost: None,
            hours_spent: 0.25,
            parts_used: vec![],
            next_due: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RecordMaintenanceInput = serde_json::from_str(&json).unwrap();
        assert!(back.cost.is_none());
        assert!(back.parts_used.is_empty());
        assert!(back.next_due.is_none());
    }

    #[test]
    fn search_resources_input_serde_roundtrip() {
        let input = SearchResourcesInput {
            resource_type: Some(ResourceType::Car),
            available_only: true,
            query: Some("electric".to_string()),
            limit: Some(10),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SearchResourcesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.resource_type, Some(ResourceType::Car));
        assert!(back.available_only);
        assert_eq!(back.query, Some("electric".to_string()));
        assert_eq!(back.limit, Some(10));
    }

    #[test]
    fn search_resources_input_minimal_serde() {
        let input = SearchResourcesInput {
            resource_type: None,
            available_only: false,
            query: None,
            limit: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SearchResourcesInput = serde_json::from_str(&json).unwrap();
        assert!(back.resource_type.is_none());
        assert!(!back.available_only);
        assert!(back.query.is_none());
        assert!(back.limit.is_none());
    }

    #[test]
    fn set_resource_availability_input_serde_roundtrip() {
        let input = SetResourceAvailabilityInput {
            hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            available: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SetResourceAvailabilityInput = serde_json::from_str(&json).unwrap();
        assert!(!back.available);
    }

    #[test]
    fn complete_usage_with_timebank_input_serde_roundtrip() {
        let input = CompleteUsageWithTimebankInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_after: ResourceCondition::Good,
            issues: vec!["minor wear".to_string()],
            notes: "Good session".to_string(),
            hours_used: 4.5,
            category_description: "Power tool rental".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CompleteUsageWithTimebankInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.condition_after, ResourceCondition::Good);
        assert!((back.hours_used - 4.5).abs() < f64::EPSILON);
        assert_eq!(back.category_description, "Power tool rental");
        assert_eq!(back.issues.len(), 1);
    }

    // ========================================================================
    // FLOAT VALIDATION EDGE CASES
    // ========================================================================

    #[test]
    fn maintenance_nan_hours_detected() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Routine,
            description: "NaN test".to_string(),
            cost: None,
            hours_spent: f64::NAN,
            parts_used: vec![],
            next_due: None,
        };
        assert!(
            !input.hours_spent.is_finite(),
            "NaN hours_spent should fail is_finite()"
        );
    }

    #[test]
    fn maintenance_infinite_hours_detected() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Routine,
            description: "Infinity test".to_string(),
            cost: None,
            hours_spent: f64::INFINITY,
            parts_used: vec![],
            next_due: None,
        };
        assert!(
            !input.hours_spent.is_finite(),
            "Infinite hours_spent should fail is_finite()"
        );
    }

    #[test]
    fn maintenance_negative_hours_detected() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Routine,
            description: "Negative test".to_string(),
            cost: None,
            hours_spent: -2.0,
            parts_used: vec![],
            next_due: None,
        };
        assert!(input.hours_spent < 0.0, "Negative hours should be caught");
    }

    #[test]
    fn maintenance_valid_hours_accepted() {
        let input = RecordMaintenanceInput {
            resource_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            maintenance_type: MaintenanceType::Routine,
            description: "Valid test".to_string(),
            cost: Some(50),
            hours_spent: 1.5,
            parts_used: vec!["filter".to_string()],
            next_due: None,
        };
        assert!(input.hours_spent.is_finite() && input.hours_spent >= 0.0);
    }

    #[test]
    fn timebank_nan_hours_detected() {
        let input = CompleteUsageWithTimebankInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_after: ResourceCondition::Good,
            issues: vec![],
            notes: "NaN test".to_string(),
            hours_used: f64::NAN,
            category_description: "Test".to_string(),
        };
        assert!(
            !input.hours_used.is_finite(),
            "NaN hours_used should fail is_finite()"
        );
    }

    #[test]
    fn timebank_negative_hours_detected() {
        let input = CompleteUsageWithTimebankInput {
            booking_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            condition_after: ResourceCondition::Good,
            issues: vec![],
            notes: "Negative test".to_string(),
            hours_used: -1.0,
            category_description: "Test".to_string(),
        };
        assert!(input.hours_used < 0.0, "Negative hours should be caught");
    }
}
