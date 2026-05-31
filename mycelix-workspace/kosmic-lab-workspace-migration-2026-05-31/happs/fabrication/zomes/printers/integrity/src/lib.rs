// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Printers Integrity Zome
//!
//! This zome defines the entry types and validation rules for 3D printers
//! in the Mycelix Fabrication hApp. It manages a distributed registry of
//! printers with their capabilities and availability.

use fabrication_common::*;
use hdi::prelude::*;

/// Entry types for the printers zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A registered 3D printer
    #[entry_type(visibility = "public")]
    Printer(Printer),
    /// Printer status update
    #[entry_type(visibility = "public")]
    PrinterStatus(PrinterStatus),
}

/// Link types for the printers zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from owner agent to their printers
    OwnerToPrinters,
    /// Link from geohash anchor to printers in that area
    GeohashToPrinters,
    /// Link from printer type anchor to printers
    PrinterTypeToPrinters,
    /// Link for all printers discovery
    AllPrinters,
    /// Link for available printers
    AvailablePrinters,
    /// Link from printer to status updates
    PrinterToStatus,
}

/// A registered 3D printer
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Printer {
    /// Unique identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Owner's public key
    pub owner: AgentPubKey,
    /// Geographic location for matching
    pub location: Option<GeoLocation>,
    /// Type of printer (FDM, SLA, etc.)
    pub printer_type: PrinterType,
    /// Capabilities and specifications
    pub capabilities: PrinterCapabilities,
    /// Materials currently loaded/available
    pub materials_available: Vec<MaterialType>,
    /// Current availability status
    pub availability: AvailabilityStatus,
    /// Commercial printing rates (if applicable)
    pub rates: Option<PrinterRates>,
    /// When the printer was registered
    pub created_at: Timestamp,
    /// When the printer info was last updated
    pub updated_at: Timestamp,
}

/// Printer status update
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrinterStatus {
    /// Printer this status belongs to
    pub printer_hash: ActionHash,
    /// Current status
    pub status: AvailabilityStatus,
    /// Optional status message
    pub message: Option<String>,
    /// Estimated time until available (in minutes)
    pub eta_available: Option<u32>,
    /// Current job (if printing)
    pub current_job: Option<ActionHash>,
    /// Queue length
    pub queue_length: u32,
    /// When status was updated
    pub updated_at: Timestamp,
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
            // Links can be deleted by their creators
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::Printer(printer) => validate_printer(printer),
        EntryTypes::PrinterStatus(status) => validate_status(status),
    }
}

/// Validate a printer entry
fn validate_printer(printer: Printer) -> ExternResult<ValidateCallbackResult> {
    // Name must not be empty
    if printer.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Printer name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if printer.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Printer name cannot exceed 100 characters".to_string(),
        ));
    }

    // Validate build volume
    if printer.capabilities.build_volume.x <= 0.0
        || printer.capabilities.build_volume.y <= 0.0
        || printer.capabilities.build_volume.z <= 0.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Build volume dimensions must be positive".to_string(),
        ));
    }

    // Validate temperature limits
    if printer.capabilities.max_temp_hotend == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum hotend temperature must be specified".to_string(),
        ));
    }

    // Validate layer heights
    if printer.capabilities.layer_heights.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one layer height must be specified".to_string(),
        ));
    }

    for height in &printer.capabilities.layer_heights {
        if *height <= 0.0 || *height > 2.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Layer heights must be between 0 and 2mm".to_string(),
            ));
        }
    }

    // Validate nozzle diameters
    if printer.capabilities.nozzle_diameters.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one nozzle diameter must be specified".to_string(),
        ));
    }

    // Validate location if provided
    if let Some(ref loc) = printer.location {
        if loc.country.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Country must be specified in location".to_string(),
            ));
        }
    }

    // Validate rates if provided
    if let Some(ref rates) = printer.rates {
        if rates.hourly_rate < 0.0 || rates.material_rate < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rates cannot be negative".to_string(),
            ));
        }
        if rates.currency.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Currency must be specified for rates".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a status entry
fn validate_status(status: PrinterStatus) -> ExternResult<ValidateCallbackResult> {
    // Queue length should be reasonable
    if status.queue_length > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Queue length seems unreasonably high".to_string(),
        ));
    }

    // ETA should be reasonable (max 30 days)
    if let Some(eta) = status.eta_available {
        if eta > 43200 {
            // 30 days in minutes
            return Ok(ValidateCallbackResult::Invalid(
                "ETA cannot exceed 30 days".to_string(),
            ));
        }
    }

    // Message length limit
    if let Some(ref msg) = status.message {
        if msg.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid(
                "Status message cannot exceed 500 characters".to_string(),
            ));
        }
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
        LinkTypes::OwnerToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::GeohashToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterTypeToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AvailablePrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterToStatus => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate link deletion (reserved for future use)
#[allow(dead_code)]
fn validate_delete_link(
    _link_type: LinkTypes,
    _original_link: CreateLink,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
