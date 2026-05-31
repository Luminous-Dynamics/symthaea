// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Resources Coordinator Zome
//!
//! This zome provides coordinator functions for resource sharing
//! in the Mycelix Mutual Aid hApp.

use hdk::prelude::*;
use mutualaid_common::*;
use resources_integrity::*;

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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get(hash, GetOptions::default())? {
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
            if let Some(record) = get(hash, GetOptions::default())? {
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
#[hdk_extern]
pub fn set_resource_availability(hash: ActionHash, available: bool) -> ExternResult<Record> {
    let record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Resource not found".to_string())
    ))?;

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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated resource".to_string()
    )))
}

// =============================================================================
// BOOKINGS
// =============================================================================

/// Create a new booking
#[hdk_extern]
pub fn create_booking(input: CreateBookingInput) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
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
            if let Some(record) = get(hash, GetOptions::default())? {
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
            if let Some(record) = get(hash, GetOptions::default())? {
                bookings.push(record);
            }
        }
    }

    Ok(bookings)
}

/// Confirm a booking (owner only)
#[hdk_extern]
pub fn confirm_booking(booking_hash: ActionHash) -> ExternResult<Record> {
    let record = get(booking_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated booking".to_string()
    )))
}

/// Cancel a booking
#[hdk_extern]
pub fn cancel_booking(booking_hash: ActionHash) -> ExternResult<Record> {
    let record = get(booking_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated booking".to_string()
    )))
}

// =============================================================================
// USAGE TRACKING
// =============================================================================

/// Record start of resource usage
#[hdk_extern]
pub fn start_usage(input: RecordUsageInput) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created usage record".to_string()
    )))
}

/// Complete resource usage
#[hdk_extern]
pub fn complete_usage(input: CompleteUsageInput) -> ExternResult<Record> {
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
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated usage record".to_string()
    )))
}

// =============================================================================
// MAINTENANCE
// =============================================================================

/// Record maintenance on a resource
#[hdk_extern]
pub fn record_maintenance(input: RecordMaintenanceInput) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get(hash, GetOptions::default())? {
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
