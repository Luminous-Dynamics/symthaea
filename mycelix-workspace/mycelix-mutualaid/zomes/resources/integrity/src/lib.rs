// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Resources Integrity Zome
//!
//! This zome defines entry types and validation rules for resource sharing
//! in the Mycelix Mutual Aid hApp. Supports tools, vehicles, spaces, and equipment.

use hdi::prelude::*;
use mutualaid_common::*;

/// Entry types for the resources zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A shared resource
    #[entry_type(visibility = "public")]
    SharedResource(SharedResource),
    /// A booking for a resource
    #[entry_type(visibility = "public")]
    Booking(Booking),
    /// Usage record
    #[entry_type(visibility = "public")]
    Usage(Usage),
    /// Maintenance record
    #[entry_type(visibility = "public")]
    Maintenance(Maintenance),
}

/// Link types for the resources zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from owner to their resources
    OwnerToResources,
    /// Link from resource type anchor to resources
    TypeToResources,
    /// Link from resource to its bookings
    ResourceToBookings,
    /// Link from booker to their bookings
    BookerToBookings,
    /// Link from resource to usage records
    ResourceToUsage,
    /// Link from resource to maintenance records
    ResourceToMaintenance,
    /// Link for all resources discovery
    AllResources,
    /// Link for available resources
    AvailableResources,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::SharedResource(resource) => validate_shared_resource(resource),
        EntryTypes::Booking(booking) => validate_booking(booking),
        EntryTypes::Usage(usage) => validate_usage(usage),
        EntryTypes::Maintenance(maintenance) => validate_maintenance(maintenance),
    }
}

/// Validate a shared resource
fn validate_shared_resource(resource: SharedResource) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if resource.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".to_string(),
        ));
    }

    // Name must not be empty
    if resource.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if resource.name.len() > 150 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot exceed 150 characters".to_string(),
        ));
    }

    // Description length limit
    if resource.description.len() > 3000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource description cannot exceed 3000 characters".to_string(),
        ));
    }

    // Photos limit
    if resource.photos.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 10 photos".to_string(),
        ));
    }

    // Sharing model validation
    if resource.sharing_model.hourly_rate < 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hourly rate cannot be negative".to_string(),
        ));
    }

    if let Some(daily) = resource.sharing_model.daily_rate {
        if daily < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Daily rate cannot be negative".to_string(),
            ));
        }
    }

    if let Some(deposit) = resource.sharing_model.deposit {
        if deposit < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Deposit cannot be negative".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a booking
fn validate_booking(booking: Booking) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if booking.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Booking ID cannot be empty".to_string(),
        ));
    }

    // End time must be after start time
    if booking.end_time <= booking.start_time {
        return Ok(ValidateCallbackResult::Invalid(
            "End time must be after start time".to_string(),
        ));
    }

    // Purpose length limit
    if booking.purpose.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Booking purpose cannot exceed 500 characters".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a usage record
fn validate_usage(usage: Usage) -> ExternResult<ValidateCallbackResult> {
    // If actual_end is set, it must be after actual_start
    if let Some(end) = usage.actual_end {
        if end <= usage.actual_start {
            return Ok(ValidateCallbackResult::Invalid(
                "End time must be after start time".to_string(),
            ));
        }
    }

    // Notes length limit
    if usage.notes.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage notes cannot exceed 1000 characters".to_string(),
        ));
    }

    // Issues limit
    if usage.issues.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot report more than 10 issues".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a maintenance record
fn validate_maintenance(maintenance: Maintenance) -> ExternResult<ValidateCallbackResult> {
    // Description must not be empty
    if maintenance.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Maintenance description cannot be empty".to_string(),
        ));
    }

    // Description length limit
    if maintenance.description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maintenance description cannot exceed 2000 characters".to_string(),
        ));
    }

    // Hours spent must be non-negative
    if maintenance.hours_spent < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours spent cannot be negative".to_string(),
        ));
    }

    // Hours spent should be reasonable
    if maintenance.hours_spent > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours spent cannot exceed 100".to_string(),
        ));
    }

    // Cost must be non-negative if present
    if let Some(cost) = maintenance.cost {
        if cost < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Maintenance cost cannot be negative".to_string(),
            ));
        }
    }

    // Parts limit
    if maintenance.parts_used.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot list more than 20 parts".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::OwnerToResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::TypeToResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToBookings => Ok(ValidateCallbackResult::Valid),
        LinkTypes::BookerToBookings => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToUsage => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToMaintenance => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AvailableResources => Ok(ValidateCallbackResult::Valid),
    }
}
