//! Printers Coordinator Zome
//!
//! This zome provides the coordinator functions for managing 3D printers
//! in the Mycelix Fabrication hApp. It includes registration, discovery,
//! matching, and status management.

use hdk::prelude::*;
use printers_integrity::*;
use fabrication_common::*;
use std::cell::RefCell;

thread_local! {
    static CONFIG: RefCell<Option<FabricationConfig>> = const { RefCell::new(None) };
}

fn get_config() -> FabricationConfig {
    CONFIG.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                dna_info()
                    .map(|info| FabricationConfig::from_properties_or_default(info.modifiers.properties.bytes()))
                    .unwrap_or_default()
            })
            .clone()
    })
}

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

#[derive(Serialize, Deserialize, Debug)]
pub struct GetAvailablePrintersInput {
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindByCapabilityInput {
    pub requirements: PrinterRequirements,
    pub pagination: Option<PaginationInput>,
}

// =============================================================================
// CRUD OPERATIONS
// =============================================================================

// =============================================================================
// RATE LIMITING
// =============================================================================

fn rate_limit_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("rate_limit:{}", agent).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn enforce_rate_limit(caller: &AgentPubKey) -> ExternResult<()> {
    let cfg = get_config();
    let max_ops = cfg.rate_limit_max_ops as usize;
    let window_micros = cfg.rate_limit_window_secs as i64 * 1_000_000;

    let anchor = rate_limit_anchor(caller)?;
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::RateLimitBucket)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let window_start = now.as_micros() - window_micros;

    let recent_count = links
        .iter()
        .filter(|l| l.timestamp.as_micros() >= window_start)
        .count();

    if recent_count >= max_ops {
        return Err(FabricationError::RateLimited {
            max_ops: cfg.rate_limit_max_ops,
            window_secs: cfg.rate_limit_window_secs,
        }.to_wasm_error());
    }

    create_link(anchor.clone(), anchor, LinkTypes::RateLimitBucket, ())?;
    Ok(())
}

fn rate_limit_caller() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    enforce_rate_limit(&agent)
}

// =============================================================================
// REGISTRATION
// =============================================================================

/// Register a new printer
#[hdk_extern]
pub fn register_printer(input: RegisterPrinterInput) -> ExternResult<Record> {
    rate_limit_caller()?;
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Printer,
        event_type: FabricationEventType::PrinterRegistered,
        payload: format!(r#"{{"hash":"{}"}}"#, action_hash),
    });

    // Link from owner
    create_link(
        owner,
        action_hash.clone(),
        LinkTypes::OwnerToPrinters,
        (),
    )?;

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
    let original = get(input.original_action_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Printer", &input.original_action_hash))?;

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
        return Err(FabricationError::unauthorized("update_printer", "Only the owner can update a printer"));
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Printer,
        event_type: FabricationEventType::PrinterUpdated,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated printer".to_string()
    )))
}

/// Deactivate a printer (marks as offline)
#[hdk_extern]
pub fn deactivate_printer(hash: ActionHash) -> ExternResult<ActionHash> {
    let printer_record = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Printer", &hash))?;

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
        return Err(FabricationError::unauthorized("deactivate_printer", "Only the owner can deactivate a printer"));
    }

    // Update availability to offline
    let now = sys_time()?;
    let deactivated = Printer {
        availability: AvailabilityStatus::Offline,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
        ..printer
    };

    let deactivated_hash = update_entry(hash.clone(), EntryTypes::Printer(deactivated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Printer,
        event_type: FabricationEventType::PrinterDeactivated,
        payload: format!(r#"{{"hash":"{}"}}"#, deactivated_hash),
    });

    Ok(deactivated_hash)
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Input for paginated agent-scoped queries
#[derive(Serialize, Deserialize, Debug)]
pub struct MyPrintersInput {
    pub pagination: Option<PaginationInput>,
}

/// Get all printers owned by the current agent
#[hdk_extern]
pub fn get_my_printers(input: MyPrintersInput) -> ExternResult<PaginatedResponse<Record>> {
    let owner = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(owner, LinkTypes::OwnerToPrinters)?, GetStrategy::default(),
    )?;

    let mut printers = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                printers.push(record);
            }
        }
    }

    Ok(paginate(printers, input.pagination.as_ref()))
}

/// Find printers nearby a location
#[derive(Serialize, Deserialize, Debug)]
pub struct FindNearbyInput {
    pub location: GeoLocation,
    pub radius_km: u32,
    pub pagination: Option<PaginationInput>,
}

#[hdk_extern]
pub fn find_printers_nearby(input: FindNearbyInput) -> ExternResult<PaginatedResponse<PrinterMatch>> {
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
        LinkQuery::try_new(geo_anchor, LinkTypes::GeohashToPrinters)?, GetStrategy::default(),
    )?;

    // Stage 1: collect all printers found via geohash prefix.
    // Stage 2: refine with Haversine when both the query and the printer
    //          carry lat/lon coordinates.
    let query_lat = input.location.lat;
    let query_lon = input.location.lon;
    let radius_km = input.radius_km as f64;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(printer) = record
                    .entry()
                    .to_app_option::<Printer>()
                    .ok()
                    .flatten()
                {
                    // --- Stage 2: Haversine refinement ---
                    let distance_km: Option<f32> =
                        match (query_lat, query_lon, printer.location.as_ref()) {
                            (Some(qlat), Some(qlon), Some(ploc))
                                if ploc.lat.is_some() && ploc.lon.is_some() =>
                            {
                                let plat = ploc.lat.unwrap();
                                let plon = ploc.lon.unwrap();
                                let d = haversine_distance_km(qlat, qlon, plat, plon);
                                Some(d as f32)
                            }
                            _ => None,
                        };

                    // If we could compute a precise distance, enforce the radius filter.
                    // Printers without lat/lon are included as geohash-only matches.
                    if let Some(d) = distance_km {
                        if d as f64 > radius_km {
                            continue;
                        }
                    }

                    // Compatibility score: 1.0 at the query point, decays linearly to 0.5
                    // at the edge of the radius.  Geohash-only matches receive 1.0.
                    let compatibility_score = match distance_km {
                        Some(d) if radius_km > 0.0 => {
                            (1.0_f32 - (d / radius_km as f32) * 0.5).max(0.5)
                        }
                        _ => 1.0,
                    };

                    matches.push(PrinterMatch {
                        printer_hash: hash,
                        printer,
                        compatibility_score,
                        distance_km,
                    });
                }
            }
        }
    }

    // Sort by distance ascending (printers without distance go last).
    matches.sort_by(|a, b| match (a.distance_km, b.distance_km) {
        (Some(da), Some(db)) => da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    Ok(paginate(matches, input.pagination.as_ref()))
}

/// Find printers by capability requirements
#[hdk_extern]
pub fn find_printers_by_capability(input: FindByCapabilityInput) -> ExternResult<PaginatedResponse<PrinterMatch>> {
    // Get all printers then filter
    let all_anchor = all_printers_anchor()?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllPrinters)?, GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(printer) = record
                    .entry()
                    .to_app_option::<Printer>()
                    .ok()
                    .flatten()
                {
                    let result = check_printer_meets_requirements(&printer, &input.requirements);
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

    Ok(paginate(matches, input.pagination.as_ref()))
}

/// Get all available printers
#[hdk_extern]
pub fn get_available_printers(input: GetAvailablePrintersInput) -> ExternResult<PaginatedResponse<Record>> {
    let available_anchor = available_printers_anchor()?;
    let links = get_links(
        LinkQuery::try_new(available_anchor, LinkTypes::AvailablePrinters)?, GetStrategy::default(),
    )?;

    let mut printers = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(printer) = record
                    .entry()
                    .to_app_option::<Printer>()
                    .ok()
                    .flatten()
                {
                    // Only include actually available printers
                    if matches!(printer.availability, AvailabilityStatus::Available) {
                        printers.push(record);
                    }
                }
            }
        }
    }

    Ok(paginate(printers, input.pagination.as_ref()))
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

/// Minimal view of a Design entry used for printer matching.
///
/// The full `Design` type is defined in `designs_integrity` which is not a
/// dependency of this crate.  We implement `TryFrom<SerializedBytes>` via the
/// `holochain_serialized_bytes` derive so that `to_app_option::<DesignSummary>()`
/// works without coupling the two zomes at the Rust type level.  Extra fields
/// present in the real Design entry are silently ignored; missing fields use
/// their serde defaults.
#[derive(Serialize, Deserialize, Debug, Default, SerializedBytes)]
struct DesignSummary {
    /// Materials that are compatible with this design (best-fit first).
    #[serde(default)]
    material_compatibility: Vec<MaterialCompatibilityEntry>,
}

/// Minimal copy of `MaterialBinding` fields needed for printer matching.
#[derive(Serialize, Deserialize, Debug)]
struct MaterialCompatibilityEntry {
    material: MaterialType,
    #[serde(default)]
    compatibility: f32,
}

#[hdk_extern]
pub fn match_design_to_printers(input: MatchDesignInput) -> ExternResult<Vec<PrinterMatch>> {
    // --- Step 1: Fetch and validate the design record. ---
    let design_record = get(input.design_hash.clone(), GetOptions::default())?.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Design not found — invalid design_hash".to_string()
        ))
    })?;

    // --- Step 2: Extract a minimal design summary to build requirements. ---
    // `DesignSummary` derives `SerializedBytes` so `to_app_option` can
    // decode the MessagePack entry bytes.  Extra fields from the real Design
    // are silently ignored; missing ones use serde defaults.
    let design_summary: DesignSummary = design_record
        .entry()
        .to_app_option::<DesignSummary>()
        .ok()
        .flatten()
        .unwrap_or_default();

    // Pick the highest-compatibility material binding (if any) as the primary
    // material requirement.
    let preferred_material: Option<MaterialType> = design_summary
        .material_compatibility
        .iter()
        .max_by(|a, b| a.compatibility.partial_cmp(&b.compatibility).unwrap_or(std::cmp::Ordering::Equal))
        .map(|mb| mb.material.clone());

    let requirements = PrinterRequirements {
        min_build_volume: None,
        material: preferred_material,
        printer_type: None,
        min_layer_height: None,
        max_layer_height: None,
        heated_bed_required: false,
        enclosure_required: false,
        min_hotend_temp: None,
    };

    // --- Step 3: Score every available printer against the requirements. ---
    let available = get_available_printers(GetAvailablePrintersInput { pagination: None })?;
    let limit = input.limit.unwrap_or(10) as usize;

    let mut matches: Vec<PrinterMatch> = Vec::new();
    for record in available.items {
        if let Some(printer) = record
            .entry()
            .to_app_option::<Printer>()
            .ok()
            .flatten()
        {
            let result = check_printer_meets_requirements(&printer, &requirements);
            if result.compatible {
                // Optionally compute distance if location is provided.
                let distance_km: Option<f32> = match (
                    input.location.as_ref(),
                    printer.location.as_ref(),
                ) {
                    (Some(qloc), Some(ploc))
                        if qloc.lat.is_some()
                            && qloc.lon.is_some()
                            && ploc.lat.is_some()
                            && ploc.lon.is_some() =>
                    {
                        let d = haversine_distance_km(
                            qloc.lat.unwrap(),
                            qloc.lon.unwrap(),
                            ploc.lat.unwrap(),
                            ploc.lon.unwrap(),
                        );
                        Some(d as f32)
                    }
                    _ => None,
                };

                matches.push(PrinterMatch {
                    printer_hash: record.action_address().clone(),
                    printer,
                    compatibility_score: result.score,
                    distance_km,
                });
            }
        }
    }

    // --- Step 4: Sort by compatibility score descending, apply limit. ---
    matches.sort_by(|a, b| {
        b.compatibility_score
            .partial_cmp(&a.compatibility_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(limit);

    Ok(matches)
}

/// Check compatibility between a printer and a design
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckCompatibilityInput {
    pub printer_hash: ActionHash,
    pub design_hash: ActionHash,
}

#[hdk_extern]
pub fn check_printer_compatibility(input: CheckCompatibilityInput) -> ExternResult<CompatibilityResult> {
    let printer_record = get(input.printer_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Printer", &input.printer_hash))?;

    let printer: Printer = printer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    // Fetch design and extract material requirements via DesignSummary
    let design_record = get(input.design_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &input.design_hash))?;
    let design_summary: DesignSummary = design_record
        .entry()
        .to_app_option::<DesignSummary>()
        .ok()
        .flatten()
        .unwrap_or_default();

    // Build requirements from design's material compatibility
    let preferred_material = design_summary
        .material_compatibility
        .iter()
        .max_by(|a, b| a.compatibility.partial_cmp(&b.compatibility).unwrap_or(std::cmp::Ordering::Equal))
        .map(|mb| mb.material.clone());

    let requirements = PrinterRequirements {
        min_build_volume: None,
        material: preferred_material,
        printer_type: None,
        min_layer_height: None,
        max_layer_height: None,
        heated_bed_required: false,
        enclosure_required: false,
        min_hotend_temp: None,
    };

    let result = check_printer_meets_requirements(&printer, &requirements);
    Ok(CompatibilityResult {
        compatible: result.compatible,
        score: result.score,
        issues: result.issues,
        recommendations: if result.score < 1.0 && result.compatible {
            vec!["Some capabilities are partially matched; verify settings before printing".to_string()]
        } else {
            vec![]
        },
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
    let printer_record = get(input.printer_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Printer", &input.printer_hash))?;

    let printer: Printer = printer_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(FabricationError::not_found("Printer", &input.printer_hash))?;

    // Verify owner
    let owner = agent_info()?.agent_initial_pubkey;
    if printer.owner != owner {
        return Err(FabricationError::unauthorized("update_availability", "Only the owner can update availability"));
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Printer,
        event_type: FabricationEventType::AvailabilityChanged,
        payload: format!(r#"{{"hash":"{}"}}"#, status_hash),
    });

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
    format!("printer_{}_{}", now.as_micros(), &agent[..8.min(agent.len())])
}

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", name).into_bytes()
    ));
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

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn test_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    /// A full-featured FDM printer with 220×220×250 mm build volume,
    /// heated bed, no enclosure, 260 °C hotend, PLA + PETG available.
    fn base_printer() -> Printer {
        Printer {
            id: "printer-test-001".to_string(),
            name: "Test Printer".to_string(),
            owner: test_agent(),
            location: None,
            printer_type: PrinterType::FDM,
            capabilities: PrinterCapabilities {
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
                features: vec![],
            },
            materials_available: vec![MaterialType::PLA, MaterialType::PETG],
            availability: AvailabilityStatus::Available,
            rates: None,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        }
    }

    /// Minimal requirements — no constraints at all.
    fn empty_requirements() -> PrinterRequirements {
        PrinterRequirements {
            min_build_volume: None,
            material: None,
            printer_type: None,
            min_layer_height: None,
            max_layer_height: None,
            heated_bed_required: false,
            enclosure_required: false,
            min_hotend_temp: None,
        }
    }

    // -------------------------------------------------------------------------
    // 1. All requirements met → score = 1.0, compatible = true
    // -------------------------------------------------------------------------
    #[test]
    fn test_all_requirements_met() {
        let printer = base_printer();
        let requirements = PrinterRequirements {
            min_build_volume: Some(BuildVolume { x: 200.0, y: 200.0, z: 200.0 }),
            material: Some(MaterialType::PLA),
            printer_type: Some(PrinterType::FDM),
            min_layer_height: None,
            max_layer_height: None,
            heated_bed_required: true,
            enclosure_required: false,
            min_hotend_temp: Some(240),
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.compatible, "Should be compatible when all requirements are met");
        assert!((result.score - 1.0).abs() < f32::EPSILON, "Score should be 1.0, got {}", result.score);
        assert!(result.issues.is_empty(), "Should have no issues, got: {:?}", result.issues);
    }

    // -------------------------------------------------------------------------
    // 2. Build volume too small → score deducted by 0.3
    // -------------------------------------------------------------------------
    #[test]
    fn test_volume_too_small() {
        let printer = base_printer(); // 220×220×250
        let requirements = PrinterRequirements {
            // Requires 300×300×300 — exceeds printer on all axes
            min_build_volume: Some(BuildVolume { x: 300.0, y: 300.0, z: 300.0 }),
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.issues.iter().any(|i| i.contains("Build volume")),
            "Should report build volume issue, issues: {:?}", result.issues);
        // Score 1.0 - 0.3 = 0.7; compatible threshold is > 0.5 so still compatible
        let expected_score = 0.7_f32;
        assert!((result.score - expected_score).abs() < 1e-5, "Score should be ~0.7, got {}", result.score);
        assert!(result.compatible, "Score 0.7 is above 0.5 threshold so should be compatible");
    }

    // -------------------------------------------------------------------------
    // 3. Required material not in printer's list → score deducted by 0.2
    // -------------------------------------------------------------------------
    #[test]
    fn test_material_missing() {
        let printer = base_printer(); // has PLA + PETG
        let requirements = PrinterRequirements {
            material: Some(MaterialType::ABS), // not available
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.issues.iter().any(|i| i.contains("ABS") || i.contains("Material")),
            "Should report material issue, issues: {:?}", result.issues);
        let expected_score = 0.8_f32;
        assert!((result.score - expected_score).abs() < 1e-5, "Score should be ~0.8, got {}", result.score);
        assert!(result.compatible, "Score 0.8 is above 0.5 threshold so should be compatible");
    }

    // -------------------------------------------------------------------------
    // 4. Heated bed required but printer lacks one → score deducted by 0.3
    // -------------------------------------------------------------------------
    #[test]
    fn test_heated_bed_missing() {
        let mut printer = base_printer();
        printer.capabilities.heated_bed = false; // no heated bed
        let requirements = PrinterRequirements {
            heated_bed_required: true,
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.issues.iter().any(|i| i.contains("Heated bed") || i.contains("heated bed")),
            "Should report heated bed issue, issues: {:?}", result.issues);
        let expected_score = 0.7_f32;
        assert!((result.score - expected_score).abs() < 1e-5, "Score should be ~0.7, got {}", result.score);
        assert!(result.compatible, "Score 0.7 is above 0.5 threshold so should be compatible");
    }

    // -------------------------------------------------------------------------
    // 5. Enclosure required but printer has none → score deducted by 0.2
    // -------------------------------------------------------------------------
    #[test]
    fn test_enclosure_missing() {
        let printer = base_printer(); // enclosure = false
        let requirements = PrinterRequirements {
            enclosure_required: true, // requires enclosure
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.issues.iter().any(|i| i.contains("nclosure")),
            "Should report enclosure issue, issues: {:?}", result.issues);
        let expected_score = 0.8_f32;
        assert!((result.score - expected_score).abs() < 1e-5, "Score should be ~0.8, got {}", result.score);
        assert!(result.compatible, "Score 0.8 is above 0.5 threshold so should be compatible");
    }

    // -------------------------------------------------------------------------
    // 6. Hotend max_temp lower than required → score deducted by 0.4
    // -------------------------------------------------------------------------
    #[test]
    fn test_temp_too_low() {
        let printer = base_printer(); // max_temp_hotend = 260
        let requirements = PrinterRequirements {
            min_hotend_temp: Some(300), // requires 300 °C
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.issues.iter().any(|i| i.contains("temp") || i.contains("Hotend")),
            "Should report temperature issue, issues: {:?}", result.issues);
        let expected_score = 0.6_f32;
        assert!((result.score - expected_score).abs() < 1e-5, "Score should be ~0.6, got {}", result.score);
        assert!(result.compatible, "Score 0.6 is above 0.5 threshold so should be compatible");
    }

    // -------------------------------------------------------------------------
    // 7. Wrong printer type → score = 0.0, compatible = false (hard fail)
    // -------------------------------------------------------------------------
    #[test]
    fn test_wrong_printer_type() {
        let printer = base_printer(); // FDM
        let requirements = PrinterRequirements {
            printer_type: Some(PrinterType::SLA), // requires resin printer
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(!result.compatible, "Different printer type should be incompatible");
        assert!((result.score).abs() < f32::EPSILON, "Score should be 0.0 for wrong printer type, got {}", result.score);
        assert!(!result.issues.is_empty(), "Should have at least one issue reported");
    }

    // -------------------------------------------------------------------------
    // 8. Volume too small AND material missing → combined deduction (0.3 + 0.2 = 0.5)
    // -------------------------------------------------------------------------
    #[test]
    fn test_volume_and_material_combined() {
        let printer = base_printer(); // 220×220×250, has PLA + PETG
        let requirements = PrinterRequirements {
            min_build_volume: Some(BuildVolume { x: 300.0, y: 300.0, z: 300.0 }), // too big
            material: Some(MaterialType::PEEK),  // not available
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        // volume issue (-0.3) + material issue (-0.2) = score 0.5
        // compatible threshold is > 0.5, so 0.5 is NOT compatible
        let expected_score = 0.5_f32;
        assert!((result.score - expected_score).abs() < 1e-5,
            "Score should be ~0.5, got {}", result.score);
        assert!(!result.compatible,
            "Score of 0.5 is not > 0.5, so should be incompatible");
        assert_eq!(result.issues.len(), 2, "Should have exactly 2 issues, got: {:?}", result.issues);
    }

    // -------------------------------------------------------------------------
    // 9. Volume small + no enclosure + temp too low → multiple failures, incompatible
    // -------------------------------------------------------------------------
    #[test]
    fn test_volume_enclosure_temp_combined() {
        let printer = base_printer(); // 220×220×250, no enclosure, 260 °C
        let requirements = PrinterRequirements {
            min_build_volume: Some(BuildVolume { x: 400.0, y: 400.0, z: 400.0 }), // -0.3
            enclosure_required: true,  // -0.2
            min_hotend_temp: Some(400), // -0.4
            ..empty_requirements()
        };

        let result = check_printer_meets_requirements(&printer, &requirements);

        // 1.0 - 0.3 - 0.2 - 0.4 = 0.1, clamped at 0.1 (max(0.0, 0.1))
        // compatible = 0.1 > 0.5 → false
        assert!(!result.compatible, "Should be incompatible with score well below 0.5");
        assert!(result.score < 0.5, "Score should be below 0.5, got {}", result.score);
        assert_eq!(result.issues.len(), 3, "Should have exactly 3 issues, got: {:?}", result.issues);
    }

    // -------------------------------------------------------------------------
    // 10. No requirements specified → perfect compatibility, score = 1.0
    // -------------------------------------------------------------------------
    #[test]
    fn test_no_requirements_specified() {
        let printer = base_printer();
        let requirements = empty_requirements();

        let result = check_printer_meets_requirements(&printer, &requirements);

        assert!(result.compatible, "No requirements should always be compatible");
        assert!((result.score - 1.0).abs() < f32::EPSILON,
            "No requirements should yield score 1.0, got {}", result.score);
        assert!(result.issues.is_empty(), "No requirements should yield no issues");
    }

    // =========================================================================
    // Proximity scoring formula tests
    // =========================================================================

    #[test]
    fn test_proximity_score_at_origin() {
        // At distance 0, score should be 1.0
        let d: f32 = 0.0;
        let radius_km: f32 = 10.0;
        let score = (1.0_f32 - (d / radius_km) * 0.5).max(0.5);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proximity_score_at_edge() {
        // At distance = radius, score should be 0.5
        let d: f32 = 10.0;
        let radius_km: f32 = 10.0;
        let score = (1.0_f32 - (d / radius_km) * 0.5).max(0.5);
        assert!((score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proximity_score_midpoint() {
        // At distance = radius/2, score should be 0.75
        let d: f32 = 5.0;
        let radius_km: f32 = 10.0;
        let score = (1.0_f32 - (d / radius_km) * 0.5).max(0.5);
        assert!((score - 0.75).abs() < f32::EPSILON);
    }
}
