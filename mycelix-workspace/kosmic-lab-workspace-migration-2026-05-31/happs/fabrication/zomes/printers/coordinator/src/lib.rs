// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Printers Coordinator Zome
//!
//! This zome provides the coordinator functions for managing 3D printers
//! in the Mycelix Fabrication hApp. It includes registration, discovery,
//! matching, and status management.

use fabrication_common::*;
use hdk::prelude::*;
use printers_integrity::*;

/// Input for registering a printer
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterPrinterInput {
    pub name: String,
    pub location: Option<GeoLocation>,
    pub printer_type: PrinterType,
    pub capabilities: PrinterCapabilities,
    pub materials_available: Vec<MaterialType>,
    pub rates: Option<PrinterRates>,
}

/// Input for updating a printer
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePrinterInput {
    pub original_action_hash: ActionHash,
    pub name: Option<String>,
    pub location: Option<GeoLocation>,
    pub capabilities: Option<PrinterCapabilities>,
    pub materials_available: Option<Vec<MaterialType>>,
    pub rates: Option<PrinterRates>,
}

/// Requirements for printer matching
#[derive(Serialize, Deserialize, Debug)]
pub struct PrinterRequirements {
    pub min_build_volume: Option<BuildVolume>,
    pub material: Option<MaterialType>,
    pub printer_type: Option<PrinterType>,
    pub min_layer_height: Option<f32>,
    pub max_layer_height: Option<f32>,
    pub heated_bed_required: bool,
    pub enclosure_required: bool,
    pub min_hotend_temp: Option<u16>,
}

/// Result of printer matching
#[derive(Serialize, Deserialize, Debug)]
pub struct PrinterMatch {
    pub printer_hash: ActionHash,
    pub printer: Printer,
    pub compatibility_score: f32,
    pub distance_km: Option<f32>,
}

/// Result of compatibility check
#[derive(Serialize, Deserialize, Debug)]
pub struct CompatibilityResult {
    pub compatible: bool,
    pub score: f32,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

// =============================================================================
// CRUD OPERATIONS
// =============================================================================

/// Register a new printer
#[hdk_extern]
pub fn register_printer(input: RegisterPrinterInput) -> ExternResult<Record> {
    let owner = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let printer = Printer {
        id: generate_id(),
        name: input.name,
        owner: owner.clone(),
        location: input.location.clone(),
        printer_type: input.printer_type.clone(),
        capabilities: input.capabilities,
        materials_available: input.materials_available,
        availability: AvailabilityStatus::Available,
        rates: input.rates,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::Printer(printer.clone()))?;

    // Link from owner
    create_link(owner, action_hash.clone(), LinkTypes::OwnerToPrinters, ())?;

    // Link from printer type
    let type_anchor = printer_type_anchor(&input.printer_type)?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::PrinterTypeToPrinters,
        (),
    )?;

    // Link from geohash if location provided
    if let Some(ref loc) = input.location {
        let geo_anchor = geohash_anchor(&loc.geohash)?;
        create_link(
            geo_anchor,
            action_hash.clone(),
            LinkTypes::GeohashToPrinters,
            (),
        )?;
    }

    // Link to all printers
    let all_anchor = all_printers_anchor()?;
    create_link(
        all_anchor.clone(),
        action_hash.clone(),
        LinkTypes::AllPrinters,
        (),
    )?;

    // Link to available printers
    let available_anchor = available_printers_anchor()?;
    create_link(
        available_anchor,
        action_hash.clone(),
        LinkTypes::AvailablePrinters,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve registered printer".to_string()
    )))
}

/// Get a printer by hash
#[hdk_extern]
pub fn get_printer(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Update a printer
#[hdk_extern]
pub fn update_printer(input: UpdatePrinterInput) -> ExternResult<Record> {
    let original = get(input.original_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Printer not found".to_string())),
    )?;

    let original_printer: Printer = original
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    // Verify owner
    let owner = agent_info()?.agent_initial_pubkey;
    if original_printer.owner != owner {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the owner can update a printer".to_string()
        )));
    }

    let now = sys_time()?;

    let updated_printer = Printer {
        id: original_printer.id,
        name: input.name.unwrap_or(original_printer.name),
        owner: original_printer.owner,
        location: input.location.or(original_printer.location),
        printer_type: original_printer.printer_type,
        capabilities: input.capabilities.unwrap_or(original_printer.capabilities),
        materials_available: input
            .materials_available
            .unwrap_or(original_printer.materials_available),
        availability: original_printer.availability,
        rates: input.rates.or(original_printer.rates),
        created_at: original_printer.created_at,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let new_hash = update_entry(
        input.original_action_hash,
        EntryTypes::Printer(updated_printer),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated printer".to_string()
    )))
}

/// Deactivate a printer (marks as offline)
#[hdk_extern]
pub fn deactivate_printer(hash: ActionHash) -> ExternResult<ActionHash> {
    let printer_record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Printer not found".to_string())
    ))?;

    let printer: Printer = printer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    // Verify owner
    let owner = agent_info()?.agent_initial_pubkey;
    if printer.owner != owner {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the owner can deactivate a printer".to_string()
        )));
    }

    // Update availability to offline
    let now = sys_time()?;
    let deactivated = Printer {
        availability: AvailabilityStatus::Offline,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
        ..printer
    };

    update_entry(hash, EntryTypes::Printer(deactivated))
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Get all printers owned by the current agent
#[hdk_extern]
pub fn get_my_printers(_: ()) -> ExternResult<Vec<Record>> {
    let owner = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(owner, LinkTypes::OwnerToPrinters)?,
        GetStrategy::default(),
    )?;

    let mut printers = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                printers.push(record);
            }
        }
    }

    Ok(printers)
}

/// Find printers nearby a location
#[derive(Serialize, Deserialize, Debug)]
pub struct FindNearbyInput {
    pub location: GeoLocation,
    pub radius_km: u32,
}

#[hdk_extern]
pub fn find_printers_nearby(input: FindNearbyInput) -> ExternResult<Vec<PrinterMatch>> {
    // Use geohash prefixes for proximity search
    // Shorter geohash = larger area
    let precision = match input.radius_km {
        0..=1 => 7,
        2..=5 => 6,
        6..=20 => 5,
        21..=100 => 4,
        _ => 3,
    };

    let geohash_prefix = if input.location.geohash.len() >= precision {
        &input.location.geohash[..precision]
    } else {
        &input.location.geohash
    };

    let geo_anchor = geohash_anchor(geohash_prefix)?;
    let links = get_links(
        LinkQuery::try_new(geo_anchor, LinkTypes::GeohashToPrinters)?,
        GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(printer) = record.entry().to_app_option::<Printer>().ok().flatten() {
                    matches.push(PrinterMatch {
                        printer_hash: hash,
                        printer,
                        compatibility_score: 1.0, // Would calculate based on distance
                        distance_km: None,        // Would calculate actual distance
                    });
                }
            }
        }
    }

    Ok(matches)
}

/// Find printers by capability requirements
#[hdk_extern]
pub fn find_printers_by_capability(
    requirements: PrinterRequirements,
) -> ExternResult<Vec<PrinterMatch>> {
    // Get all printers then filter
    let all_anchor = all_printers_anchor()?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllPrinters)?,
        GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(printer) = record.entry().to_app_option::<Printer>().ok().flatten() {
                    let result = check_printer_meets_requirements(&printer, &requirements);
                    if result.compatible {
                        matches.push(PrinterMatch {
                            printer_hash: hash,
                            printer,
                            compatibility_score: result.score,
                            distance_km: None,
                        });
                    }
                }
            }
        }
    }

    // Sort by compatibility score
    matches.sort_by(|a, b| {
        b.compatibility_score
            .partial_cmp(&a.compatibility_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(matches)
}

/// Get all available printers
#[hdk_extern]
pub fn get_available_printers(_: ()) -> ExternResult<Vec<Record>> {
    let available_anchor = available_printers_anchor()?;
    let links = get_links(
        LinkQuery::try_new(available_anchor, LinkTypes::AvailablePrinters)?,
        GetStrategy::default(),
    )?;

    let mut printers = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(printer) = record.entry().to_app_option::<Printer>().ok().flatten() {
                    // Only include actually available printers
                    if matches!(printer.availability, AvailabilityStatus::Available) {
                        printers.push(record);
                    }
                }
            }
        }
    }

    Ok(printers)
}

// =============================================================================
// MATCHING
// =============================================================================

/// Match a design to compatible printers
#[derive(Serialize, Deserialize, Debug)]
pub struct MatchDesignInput {
    pub design_hash: ActionHash,
    pub location: Option<GeoLocation>,
    pub limit: Option<u32>,
}

#[hdk_extern]
pub fn match_design_to_printers(input: MatchDesignInput) -> ExternResult<Vec<PrinterMatch>> {
    // This would normally fetch the design and determine requirements
    // For now, return available printers
    let available = get_available_printers(())?;
    let limit = input.limit.unwrap_or(10) as usize;

    let mut matches = Vec::new();
    for record in available.into_iter().take(limit) {
        if let Some(printer) = record.entry().to_app_option::<Printer>().ok().flatten() {
            matches.push(PrinterMatch {
                printer_hash: record.action_address().clone(),
                printer,
                compatibility_score: 0.8, // Placeholder
                distance_km: None,
            });
        }
    }

    Ok(matches)
}

/// Check compatibility between a printer and a design
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckCompatibilityInput {
    pub printer_hash: ActionHash,
    pub design_hash: ActionHash,
}

#[hdk_extern]
pub fn check_printer_compatibility(
    input: CheckCompatibilityInput,
) -> ExternResult<CompatibilityResult> {
    let printer_record = get(input.printer_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Printer not found".to_string())
    ))?;

    let _printer: Printer = printer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    // Would fetch design and compare with printer capabilities
    // For now, return a placeholder result
    Ok(CompatibilityResult {
        compatible: true,
        score: 0.85,
        issues: vec![],
        recommendations: vec!["Consider using supports for overhangs".to_string()],
    })
}

// =============================================================================
// STATUS MANAGEMENT
// =============================================================================

/// Update printer availability
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateAvailabilityInput {
    pub printer_hash: ActionHash,
    pub status: AvailabilityStatus,
    pub message: Option<String>,
    pub eta_available: Option<u32>,
    pub current_job: Option<ActionHash>,
}

#[hdk_extern]
pub fn update_availability(input: UpdateAvailabilityInput) -> ExternResult<Record> {
    let printer_record = get(input.printer_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Printer not found".to_string())),
    )?;

    let printer: Printer = printer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    // Verify owner
    let owner = agent_info()?.agent_initial_pubkey;
    if printer.owner != owner {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the owner can update availability".to_string()
        )));
    }

    let now = sys_time()?;

    // Create status entry
    let status = PrinterStatus {
        printer_hash: input.printer_hash.clone(),
        status: input.status.clone(),
        message: input.message,
        eta_available: input.eta_available,
        current_job: input.current_job,
        queue_length: 0,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let status_hash = create_entry(EntryTypes::PrinterStatus(status))?;

    // Link status to printer
    create_link(
        input.printer_hash.clone(),
        status_hash.clone(),
        LinkTypes::PrinterToStatus,
        (),
    )?;

    // Update printer availability
    let updated_printer = Printer {
        availability: input.status,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
        ..printer
    };

    update_entry(input.printer_hash, EntryTypes::Printer(updated_printer))?;

    get(status_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve status".to_string()
    )))
}

/// Get printer queue (jobs waiting)
#[hdk_extern]
pub fn get_printer_queue(printer_hash: ActionHash) -> ExternResult<Vec<Record>> {
    // Would fetch from prints zome via bridge
    // For now, return empty
    let _ = printer_hash;
    Ok(vec![])
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn check_printer_meets_requirements(
    printer: &Printer,
    requirements: &PrinterRequirements,
) -> CompatibilityResult {
    let mut issues = Vec::new();
    let mut score: f32 = 1.0;

    // Check printer type
    if let Some(ref required_type) = requirements.printer_type {
        if printer.printer_type != *required_type {
            issues.push(format!(
                "Printer is {:?}, but {:?} required",
                printer.printer_type, required_type
            ));
            return CompatibilityResult {
                compatible: false,
                score: 0.0,
                issues,
                recommendations: vec![],
            };
        }
    }

    // Check build volume
    if let Some(ref min_vol) = requirements.min_build_volume {
        if printer.capabilities.build_volume.x < min_vol.x
            || printer.capabilities.build_volume.y < min_vol.y
            || printer.capabilities.build_volume.z < min_vol.z
        {
            issues.push("Build volume too small".to_string());
            score -= 0.3;
        }
    }

    // Check material
    if let Some(ref material) = requirements.material {
        if !printer.materials_available.contains(material) {
            issues.push(format!("Material {:?} not available", material));
            score -= 0.2;
        }
    }

    // Check heated bed
    if requirements.heated_bed_required && !printer.capabilities.heated_bed {
        issues.push("Heated bed required but not available".to_string());
        score -= 0.3;
    }

    // Check enclosure
    if requirements.enclosure_required && !printer.capabilities.enclosure {
        issues.push("Enclosure required but not available".to_string());
        score -= 0.2;
    }

    // Check temperature
    if let Some(min_temp) = requirements.min_hotend_temp {
        if printer.capabilities.max_temp_hotend < min_temp {
            issues.push(format!(
                "Hotend max temp {} is below required {}",
                printer.capabilities.max_temp_hotend, min_temp
            ));
            score -= 0.4;
        }
    }

    CompatibilityResult {
        compatible: score > 0.5,
        score: score.max(0.0),
        issues,
        recommendations: vec![],
    }
}

fn generate_id() -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!(
        "printer_{}_{}",
        now.as_micros(),
        &agent[..8.min(agent.len())]
    )
}

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes =
        SerializedBytes::from(UnsafeBytes::from(format!("anchor:{}", name).into_bytes()));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn printer_type_anchor(printer_type: &PrinterType) -> ExternResult<EntryHash> {
    make_anchor(&format!("printer_type_{:?}", printer_type))
}

fn geohash_anchor(geohash: &str) -> ExternResult<EntryHash> {
    make_anchor(&format!("geohash_{}", geohash))
}

fn all_printers_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_printers")
}

fn available_printers_anchor() -> ExternResult<EntryHash> {
    make_anchor("available_printers")
}
