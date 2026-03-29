// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Printers Integrity Zome
//!
//! This zome defines the entry types and validation rules for 3D printers
//! in the Mycelix Fabrication hApp. It manages a distributed registry of
//! printers with their capabilities and availability.

use hdi::prelude::*;
use fabrication_common::*;
use fabrication_common::validation;

/// Entry types for the printers zome
#[allow(clippy::large_enum_variant)] // Holochain entry serialization requires unboxed variants
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
    /// Per-agent rate limiting bucket
    RateLimitBucket,
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
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            if action.author != *original_action.action().author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(op_update) => {
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
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
    // --- String field guards ---

    // id: non-empty, max 64 chars
    check!(validation::require_non_empty(&printer.id, "printer id"));
    check!(validation::require_max_len(&printer.id, 64, "printer id"));

    // name: non-empty (trim), max 256 chars
    check!(validation::require_non_empty(&printer.name, "printer name"));
    check!(validation::require_max_len(&printer.name, 256, "printer name"));

    // --- Build volume: finite, positive, max 10000mm ---
    check!(validation::require_finite(printer.capabilities.build_volume.x, "build_volume.x"));
    if printer.capabilities.build_volume.x <= 0.0 || printer.capabilities.build_volume.x > 10000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "build_volume.x must be > 0 and <= 10000mm".to_string(),
        ));
    }
    check!(validation::require_finite(printer.capabilities.build_volume.y, "build_volume.y"));
    if printer.capabilities.build_volume.y <= 0.0 || printer.capabilities.build_volume.y > 10000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "build_volume.y must be > 0 and <= 10000mm".to_string(),
        ));
    }
    check!(validation::require_finite(printer.capabilities.build_volume.z, "build_volume.z"));
    if printer.capabilities.build_volume.z <= 0.0 || printer.capabilities.build_volume.z > 10000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "build_volume.z must be > 0 and <= 10000mm".to_string(),
        ));
    }

    // --- Layer heights: non-empty, max 32, each in range 0.001..2.0 ---
    if printer.capabilities.layer_heights.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one layer height must be specified".to_string(),
        ));
    }
    check!(validation::require_max_vec_len(&printer.capabilities.layer_heights, 32, "layer_heights"));
    for (i, &height) in printer.capabilities.layer_heights.iter().enumerate() {
        check!(validation::require_in_range(height, 0.001, 2.0, &format!("layer_heights[{}]", i)));
    }

    // --- Nozzle diameters: non-empty, max 32, each finite and > 0 ---
    if printer.capabilities.nozzle_diameters.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one nozzle diameter must be specified".to_string(),
        ));
    }
    check!(validation::require_max_vec_len(&printer.capabilities.nozzle_diameters, 32, "nozzle_diameters"));
    for (i, &diam) in printer.capabilities.nozzle_diameters.iter().enumerate() {
        check!(validation::require_finite(diam, &format!("nozzle_diameters[{}]", i)));
        if diam <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("nozzle_diameters[{}] must be > 0", i),
            ));
        }
    }

    // --- Materials available: max 64 ---
    check!(validation::require_max_vec_len(&printer.materials_available, 64, "materials_available"));

    // --- Features: max 64 ---
    check!(validation::require_max_vec_len(&printer.capabilities.features, 64, "features"));

    // --- Validate temperature limits ---
    if printer.capabilities.max_temp_hotend == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum hotend temperature must be specified".to_string(),
        ));
    }

    // --- Location string guards ---
    if let Some(ref loc) = printer.location {
        check!(validation::require_non_empty(&loc.country, "location country"));
        check!(validation::require_max_len(&loc.country, 10, "location country"));
        if let Some(ref city) = loc.city {
            check!(validation::require_max_len(city, 256, "location city"));
        }
        if let Some(ref region) = loc.region {
            check!(validation::require_max_len(region, 256, "location region"));
        }
        check!(validation::require_max_len(&loc.geohash, 64, "location geohash"));
    }

    // --- Rates guards ---
    if let Some(ref rates) = printer.rates {
        check!(validation::require_finite_f64(rates.hourly_rate, "hourly_rate"));
        if rates.hourly_rate < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "hourly_rate must be >= 0".to_string(),
            ));
        }
        check!(validation::require_finite_f64(rates.material_rate, "material_rate"));
        if rates.material_rate < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "material_rate must be >= 0".to_string(),
            ));
        }
        check!(validation::require_non_empty(&rates.currency, "rates currency"));
        check!(validation::require_max_len(&rates.currency, 10, "rates currency"));
        if let Some(min_order) = rates.minimum_order {
            check!(validation::require_finite_f64(min_order, "minimum_order"));
            if min_order < 0.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "minimum_order must be >= 0".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a status entry
fn validate_status(status: PrinterStatus) -> ExternResult<ValidateCallbackResult> {
    // Queue length should be reasonable
    if status.queue_length > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Queue length cannot exceed 1000".to_string(),
        ));
    }

    // ETA should be reasonable (max 30 days in minutes)
    if let Some(eta) = status.eta_available {
        if eta > 43200 {
            return Ok(ValidateCallbackResult::Invalid(
                "ETA cannot exceed 30 days".to_string(),
            ));
        }
    }

    // Message length limit
    if let Some(ref msg) = status.message {
        check!(validation::require_max_len(msg, 500, "status message"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    // Validate tag length per link type:
    // - GeohashToPrinters: 512 bytes (encodes geohash prefix data)
    // - All others: 256 bytes
    let max_len = match link_type {
        LinkTypes::GeohashToPrinters => 512,
        _ => 256,
    };
    check!(validation::require_max_tag_len(&tag, max_len, &format!("{:?}", link_type)));

    match link_type {
        LinkTypes::OwnerToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::GeohashToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterTypeToPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllPrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AvailablePrinters => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterToStatus => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RateLimitBucket => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_capabilities() -> PrinterCapabilities {
        PrinterCapabilities {
            build_volume: BuildVolume {
                x: 220.0,
                y: 220.0,
                z: 250.0,
            },
            layer_heights: vec![0.1, 0.2, 0.3],
            nozzle_diameters: vec![0.4],
            heated_bed: true,
            enclosure: false,
            multi_material: None,
            max_temp_hotend: 260,
            max_temp_bed: 100,
            features: vec![PrinterFeature::AutoLeveling],
        }
    }

    fn valid_printer() -> Printer {
        Printer {
            id: "printer-001".to_string(),
            name: "Prusa MK4".to_string(),
            owner: test_agent(),
            location: Some(GeoLocation {
                geohash: "9q5c".to_string(),
                lat: Some(32.9483),
                lon: Some(-96.7299),
                city: Some("Richardson".to_string()),
                region: Some("TX".to_string()),
                country: "US".to_string(),
            }),
            printer_type: PrinterType::FDM,
            capabilities: valid_capabilities(),
            materials_available: vec![MaterialType::PLA, MaterialType::PETG],
            availability: AvailabilityStatus::Available,
            rates: Some(PrinterRates {
                hourly_rate: 5.0,
                material_rate: 0.05,
                currency: "USD".to_string(),
                minimum_order: Some(10.0),
            }),
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        }
    }

    fn valid_status() -> PrinterStatus {
        PrinterStatus {
            printer_hash: test_action_hash(),
            status: AvailabilityStatus::Available,
            message: Some("Ready to print".to_string()),
            eta_available: None,
            current_job: None,
            queue_length: 0,
            updated_at: Timestamp::from_micros(0),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    // --- Printer tests ---

    #[test]
    fn test_valid_printer_passes() {
        let result = validate_printer(valid_printer());
        assert!(is_valid(&result), "Valid printer should pass: {:?}", result);
    }

    #[test]
    fn test_empty_name_rejected() {
        let mut p = valid_printer();
        p.name = "".to_string();
        let result = validate_printer(p);
        assert!(is_invalid(&result), "Empty name should be rejected");

        // Whitespace-only name should also be rejected
        let mut p2 = valid_printer();
        p2.name = "   ".to_string();
        let result2 = validate_printer(p2);
        assert!(is_invalid(&result2), "Whitespace-only name should be rejected");
    }

    #[test]
    fn test_nan_build_volume_rejected() {
        let mut p = valid_printer();
        p.capabilities.build_volume.x = f32::NAN;
        let result = validate_printer(p);
        assert!(is_invalid(&result), "NaN build_volume.x should be rejected");

        let mut p2 = valid_printer();
        p2.capabilities.build_volume.y = f32::INFINITY;
        let result2 = validate_printer(p2);
        assert!(is_invalid(&result2), "Infinite build_volume.y should be rejected");
    }

    #[test]
    fn test_zero_build_volume_rejected() {
        let mut p = valid_printer();
        p.capabilities.build_volume.z = 0.0;
        let result = validate_printer(p);
        assert!(is_invalid(&result), "Zero build_volume.z should be rejected");

        let mut p2 = valid_printer();
        p2.capabilities.build_volume.x = -1.0;
        let result2 = validate_printer(p2);
        assert!(is_invalid(&result2), "Negative build_volume.x should be rejected");
    }

    #[test]
    fn test_layer_height_nan_rejected() {
        let mut p = valid_printer();
        p.capabilities.layer_heights = vec![0.2, f32::NAN, 0.1];
        let result = validate_printer(p);
        assert!(is_invalid(&result), "NaN layer height should be rejected");
    }

    #[test]
    fn test_too_many_layer_heights() {
        let mut p = valid_printer();
        p.capabilities.layer_heights = (0..33).map(|i| 0.05 + (i as f32) * 0.05).collect();
        // Clamp values to valid range
        for h in p.capabilities.layer_heights.iter_mut() {
            if *h > 2.0 {
                *h = 1.99;
            }
        }
        let result = validate_printer(p);
        assert!(is_invalid(&result), "More than 32 layer heights should be rejected");
    }

    #[test]
    fn test_too_many_materials() {
        let mut p = valid_printer();
        p.materials_available = (0..65).map(|_| MaterialType::PLA).collect();
        let result = validate_printer(p);
        assert!(is_invalid(&result), "More than 64 materials should be rejected");
    }

    #[test]
    fn test_negative_hourly_rate_rejected() {
        let mut p = valid_printer();
        p.rates = Some(PrinterRates {
            hourly_rate: -1.0,
            material_rate: 0.05,
            currency: "USD".to_string(),
            minimum_order: None,
        });
        let result = validate_printer(p);
        assert!(is_invalid(&result), "Negative hourly rate should be rejected");
    }

    #[test]
    fn test_nan_hourly_rate_rejected() {
        let mut p = valid_printer();
        p.rates = Some(PrinterRates {
            hourly_rate: f64::NAN,
            material_rate: 0.05,
            currency: "USD".to_string(),
            minimum_order: None,
        });
        let result = validate_printer(p);
        assert!(is_invalid(&result), "NaN hourly rate should be rejected");
    }

    #[test]
    fn test_empty_country_in_location_rejected() {
        let mut p = valid_printer();
        p.location = Some(GeoLocation {
            geohash: "9q5c".to_string(),
            lat: None,
            lon: None,
            city: None,
            region: None,
            country: "".to_string(),
        });
        let result = validate_printer(p);
        assert!(is_invalid(&result), "Empty country should be rejected");
    }

    #[test]
    fn test_nozzle_diameter_zero_rejected() {
        let mut p = valid_printer();
        p.capabilities.nozzle_diameters = vec![0.0];
        let result = validate_printer(p);
        assert!(is_invalid(&result), "Zero nozzle diameter should be rejected");
    }

    // --- Status tests ---

    #[test]
    fn test_queue_length_too_high() {
        let mut s = valid_status();
        s.queue_length = 1001;
        let result = validate_status(s);
        assert!(is_invalid(&result), "Queue length > 1000 should be rejected");
    }

    #[test]
    fn test_valid_status_passes() {
        let result = validate_status(valid_status());
        assert!(is_valid(&result), "Valid status should pass: {:?}", result);
    }

    // --- Link tag validation tests ---

    #[test]
    fn test_link_tag_at_max_length_passes() {
        let max_tag = LinkTag(vec![0u8; 256]);
        let result = validate_create_link(
            LinkTypes::OwnerToPrinters,
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            max_tag,
        );
        assert!(is_valid(&result), "Tag at max length (256) should pass: {:?}", result);
    }

    #[test]
    fn test_link_tag_over_max_length_rejected() {
        let oversized_tag = LinkTag(vec![0u8; 257]);
        let result = validate_create_link(
            LinkTypes::OwnerToPrinters,
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            oversized_tag,
        );
        assert!(is_invalid(&result), "Tag over max length (257) should be rejected");
    }

    #[test]
    fn test_geohash_link_tag_at_512_bytes_passes() {
        let max_tag = LinkTag(vec![0u8; 512]);
        let result = validate_create_link(
            LinkTypes::GeohashToPrinters,
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            max_tag,
        );
        assert!(is_valid(&result), "Geohash tag at max length (512) should pass: {:?}", result);
    }

    #[test]
    fn test_geohash_link_tag_over_512_bytes_rejected() {
        let oversized_tag = LinkTag(vec![0u8; 513]);
        let result = validate_create_link(
            LinkTypes::GeohashToPrinters,
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            AnyLinkableHash::from(AgentPubKey::from_raw_36(vec![0u8; 36])),
            oversized_tag,
        );
        assert!(is_invalid(&result), "Geohash tag over max length (513) should be rejected");
    }
}
