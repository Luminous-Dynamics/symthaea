//! Prints Coordinator Zome
//!
//! This zome provides the coordinator functions for managing print jobs
//! in the Mycelix Fabrication hApp. It implements:
//!
//! - Print job lifecycle management
//! - Proof of Grounded Fabrication (PoGF) scoring
//! - Cincinnati Algorithm for quality monitoring
//! - MYCELIUM (CIV) reputation integration

use hdk::prelude::*;
use prints_integrity::*;
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

// Use PrinterInfo from fabrication_common for cross-zome deserialization
// (importing printers_integrity causes duplicate symbol errors)
use fabrication_common::PrinterInfo as Printer;

// =============================================================================
// PROOF OF GROUNDED FABRICATION (PoGF)
// =============================================================================

/// Calculate Proof of Grounded Fabrication score
/// PoG = (E_renewable × w_e) + (M_circular × w_m) + (Q_verified × w_q) + (L_local × w_l)
/// Weights loaded from DNA properties via FabricationConfig.
fn calculate_pog_score(
    energy_renewable_fraction: f32,
    material_circularity: f32,
    quality_verified: f32,
    local_participation: f32,
) -> f32 {
    let cfg = get_config();
    let e = energy_renewable_fraction.clamp(0.0, 1.0);
    let m = material_circularity.clamp(0.0, 1.0);
    let q = quality_verified.clamp(0.0, 1.0);
    let l = local_participation.clamp(0.0, 1.0);

    (e * cfg.pog_energy_weight) + (m * cfg.pog_material_weight) + (q * cfg.pog_quality_weight) + (l * cfg.pog_local_weight)
}

/// Calculate MYCELIUM (CIV) reputation earned from a print
fn calculate_mycelium_reward(pog_score: f32, quality_score: f32) -> u64 {
    let cfg = get_config();
    if pog_score < cfg.min_pog_for_mycelium {
        return 0;
    }

    // Base reward multiplied by PoGF and quality scores
    let multiplier = (pog_score + quality_score) / 2.0;
    let bonus = (pog_score - cfg.min_pog_for_mycelium) * 50.0; // Bonus for exceeding minimum

    (cfg.base_mycelium_reward as f32 * multiplier + bonus) as u64
}

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

/// Input for creating a print job
#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePrintJobInput {
    pub design_hash: ActionHash,
    pub printer_hash: ActionHash,
    pub settings: PrintSettings,
    pub energy_source: Option<EnergyType>,
    pub material_passport: Option<MaterialPassport>,
}

/// Input for recording a print result
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordPrintInput {
    pub job_hash: ActionHash,
    pub result: PrintResult,
    pub energy_used_kwh: f32,
    pub photos: Vec<String>,
    pub notes: String,
    pub issues: Vec<PrintIssue>,
    pub dimensional_measurements: Option<Vec<DimensionalMeasurement>>,
    /// Optional PoGF attestation bundle for verifiable sustainability claims
    pub attestations: Option<PogfAttestationBundle>,
}

/// Input for starting Cincinnati monitoring
#[derive(Serialize, Deserialize, Debug)]
pub struct StartCincinnatiInput {
    pub job_hash: ActionHash,
    pub total_layers: u32,
    pub sampling_rate_hz: u32,
}

/// Input for reporting a Cincinnati anomaly
#[derive(Serialize, Deserialize, Debug)]
pub struct ReportAnomalyInput {
    pub session_id: String,
    pub anomaly: AnomalyEvent,
}

/// Statistics for a design
#[derive(Serialize, Deserialize, Debug)]
pub struct DesignPrintStats {
    pub design_hash: ActionHash,
    pub total_prints: u32,
    pub successful_prints: u32,
    pub failed_prints: u32,
    pub average_quality: f32,
    pub average_pog_score: f32,
    pub total_mycelium_earned: u64,
    pub common_issues: Vec<(PrintIssue, u32)>,
}

// =============================================================================
// JOB MANAGEMENT
// =============================================================================

/// Ensure the caller is the owner of the printer associated with this job.
fn ensure_caller_is_printer_owner(printer_hash: &ActionHash) -> ExternResult<()> {
    let caller = agent_info()?.agent_initial_pubkey;
    let record = get(printer_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Printer", printer_hash))?;

    let printer: Printer = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(FabricationError::not_found("Printer", printer_hash))?;

    if printer.owner != caller {
        return Err(FabricationError::unauthorized(
            "printer_operation", "Only the printer owner can perform this action",
        ));
    }

    Ok(())
}

/// Ensure the caller is either the requester or the printer owner for this job.
fn ensure_caller_is_requester_or_owner(job: &PrintJob) -> ExternResult<()> {
    let caller = agent_info()?.agent_initial_pubkey;
    if caller == job.requester {
        return Ok(());
    }

    ensure_caller_is_printer_owner(&job.printer_hash)
}

// Local Design struct for cross-zome deserialization
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
struct DesignForSafetyCheck {
    pub safety_class: SafetyClass,
}

/// Check if a design with Class3+ safety requires passing verification
fn enforce_safety_class(design_hash: &ActionHash) -> ExternResult<()> {
    // Fetch the design to read its safety class
    let design_record = get(design_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Design not found".to_string()
        )))?;

    let design: DesignForSafetyCheck = design_record
        .entry()
        .to_app_option()
        .map_err(|_| {
            // If we can't parse safety class, allow the job (backward compatibility)
            wasm_error!(WasmErrorInner::Guest(
                "Could not parse design safety class".to_string()
            ))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Design entry is empty".to_string()
        )))?;

    if !design.safety_class.requires_verification() {
        return Ok(());
    }

    // For Class3+, check verification records via cross-zome call
    let response = call(
        CallTargetCell::Local,
        "verification",
        "get_design_verifications".into(),
        None,
        design_hash.clone(),
    )?;

    let verifications: Vec<Record> = match response {
        ZomeCallResponse::Ok(bytes) => bytes
            .decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Failed to decode verification response: {}", e)
            )))?,
        ZomeCallResponse::NetworkError(_) => {
            // Network error → allow job (best-effort)
            return Ok(());
        }
        _ => {
            // Other responses → allow job
            return Ok(());
        }
    };

    // Check if any verification passed
    let has_passing = verifications.iter().any(|record: &Record| {
        record
            .entry()
            .to_app_option::<DesignVerificationCheck>()
            .ok()
            .flatten()
            .map(|v| matches!(v.result, VerificationResult::Passed { .. }))
            .unwrap_or(false)
    });

    if !has_passing {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Design has safety class {:?} which requires at least one passing verification before printing",
            design.safety_class
        ))));
    }

    Ok(())
}

// Local struct for verification deserialization
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
struct DesignVerificationCheck {
    pub result: VerificationResult,
}

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
// PRINT JOBS
// =============================================================================

/// Create a new print job
#[hdk_extern]
pub fn create_print_job(input: CreatePrintJobInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    // Enforce safety class verification for Class3+ designs
    // Best-effort: if verification zome is unavailable, allow the job
    match enforce_safety_class(&input.design_hash) {
        Ok(()) => {},
        Err(e) => {
            let msg = format!("{}", e);
            if msg.contains("requires at least one passing verification") {
                return Err(e);
            }
            // Other errors (zome unavailable, parse error) → allow job with warning signal
            let _ = emit_signal(&TypedFabricationSignal {
                domain: FabricationDomain::Print,
                event_type: FabricationEventType::SafetyCheckSkipped,
                payload: format!(r#"{{"reason":"{}"}}"#, msg.replace('"', "'")),
            });
        }
    }

    let requester = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let job = PrintJob {
        id: generate_job_id(),
        design_hash: input.design_hash.clone(),
        printer_hash: input.printer_hash.clone(),
        requester: requester.clone(),
        settings: input.settings,
        status: PrintJobStatus::Pending,
        grounding_certificate: None,
        energy_source: input.energy_source,
        material_passport: input.material_passport,
        cincinnati_session: None,
        quality_predictions: vec![],
        estimated_time_minutes: None,
        actual_time_minutes: None,
        material_used_grams: None,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        started_at: None,
        completed_at: None,
    };

    let action_hash = create_entry(EntryTypes::PrintJob(job))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::JobCreated,
        payload: format!(r#"{{"hash":"{}"}}"#, action_hash),
    });

    // Create links
    create_link(
        input.design_hash,
        action_hash.clone(),
        LinkTypes::DesignToPrints,
        (),
    )?;

    create_link(
        input.printer_hash,
        action_hash.clone(),
        LinkTypes::PrinterToPrints,
        (),
    )?;

    create_link(
        requester,
        action_hash.clone(),
        LinkTypes::RequesterToJobs,
        (),
    )?;

    // Link to pending jobs
    let pending_anchor = pending_jobs_anchor()?;
    create_link(
        pending_anchor,
        action_hash.clone(),
        LinkTypes::PendingJobs,
        (),
    )?;

    // Link to all jobs
    let all_anchor = all_jobs_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllPrintJobs, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created job".to_string()
    )))
}

/// Accept a print job (by printer owner)
#[hdk_extern]
pub fn accept_print_job(hash: ActionHash) -> ExternResult<Record> {
    let job = get_print_job_mut(hash.clone())?;

    // Only the printer owner may accept a job
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    if !matches!(job.status, PrintJobStatus::Pending) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Job is not pending".to_string()
        )));
    }

    let updated = PrintJob {
        status: PrintJobStatus::Accepted,
        ..job
    };

    let new_hash = update_entry(hash, EntryTypes::PrintJob(updated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::JobAccepted,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated job".to_string()
    )))
}

/// Start printing a job
#[hdk_extern]
pub fn start_print(hash: ActionHash) -> ExternResult<Record> {
    let job = get_print_job_mut(hash.clone())?;

    // Only the printer owner may start a job
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    if !matches!(job.status, PrintJobStatus::Accepted | PrintJobStatus::Queued) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Job must be accepted or queued to start".to_string()
        )));
    }

    let now = sys_time()?;
    let updated = PrintJob {
        status: PrintJobStatus::Printing,
        started_at: Some(Timestamp::from_micros(now.as_micros() as i64)),
        ..job
    };

    let new_hash = update_entry(hash, EntryTypes::PrintJob(updated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::PrintStarted,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated job".to_string()
    )))
}

/// Update print progress
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateProgressInput {
    pub job_hash: ActionHash,
    pub progress_percent: u8,
    pub current_layer: Option<u32>,
    pub material_used_grams: Option<f32>,
}

#[hdk_extern]
pub fn update_print_progress(input: UpdateProgressInput) -> ExternResult<Record> {
    let job = get_print_job_mut(input.job_hash.clone())?;

    // Only the printer owner may update progress
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    // Basic bounds on progress percentage
    if input.progress_percent > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Progress percent must be between 0 and 100".to_string()
        )));
    }

    if !matches!(job.status, PrintJobStatus::Printing) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Job is not currently printing".to_string()
        )));
    }

    let updated = PrintJob {
        material_used_grams: input.material_used_grams.or(job.material_used_grams),
        ..job
    };

    let new_hash = update_entry(input.job_hash, EntryTypes::PrintJob(updated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::ProgressUpdated,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated job".to_string()
    )))
}

/// Complete a print job
#[derive(Serialize, Deserialize, Debug)]
pub struct CompletePrintInput {
    pub job_hash: ActionHash,
    pub result: PrintResult,
}

#[hdk_extern]
pub fn complete_print(input: CompletePrintInput) -> ExternResult<Record> {
    let job = get_print_job_mut(input.job_hash.clone())?;

    // Only the printer owner may mark a job as complete/failed
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    let now = sys_time()?;
    let actual_time = if let Some(started) = job.started_at {
        let duration = now.as_micros() as i64 - started.as_micros();
        Some((duration / 60_000_000) as u32) // Convert to minutes
    } else {
        None
    };

    let new_status = match input.result {
        PrintResult::Success | PrintResult::PartialSuccess => PrintJobStatus::Completed,
        PrintResult::Failed(_) => PrintJobStatus::Failed,
    };

    let updated = PrintJob {
        status: new_status,
        actual_time_minutes: actual_time,
        completed_at: Some(Timestamp::from_micros(now.as_micros() as i64)),
        ..job
    };

    let new_hash = update_entry(input.job_hash, EntryTypes::PrintJob(updated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::PrintCompleted,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated job".to_string()
    )))
}

/// Cancel a print job
#[derive(Serialize, Deserialize, Debug)]
pub struct CancelPrintInput {
    pub job_hash: ActionHash,
    pub reason: String,
}

#[hdk_extern]
pub fn cancel_print(input: CancelPrintInput) -> ExternResult<Record> {
    let job = get_print_job_mut(input.job_hash.clone())?;

    // Only the requester or printer owner may cancel
    ensure_caller_is_requester_or_owner(&job)?;

    if input.reason.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cancellation reason is required".to_string()
        )));
    }

    if matches!(job.status, PrintJobStatus::Completed | PrintJobStatus::Cancelled) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Job is already completed or cancelled".to_string()
        )));
    }

    let now = sys_time()?;
    let updated = PrintJob {
        status: PrintJobStatus::Cancelled,
        completed_at: Some(Timestamp::from_micros(now.as_micros() as i64)),
        ..job
    };

    let new_hash = update_entry(input.job_hash, EntryTypes::PrintJob(updated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::PrintCancelled,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated job".to_string()
    )))
}

// =============================================================================
// PRINT RECORDS (with PoGF)
// =============================================================================

/// Record a print result with full PoGF metrics
#[hdk_extern]
pub fn record_print_result(input: RecordPrintInput) -> ExternResult<Record> {
    let job_record = get(input.job_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Job not found".to_string()
        )))?;

    let job: PrintJob = job_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse job".to_string()
        )))?;

    // Only the printer owner may submit print results
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    // Validate float inputs
    if !input.energy_used_kwh.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "energy_used_kwh must be a finite number".to_string()
        )));
    }
    if input.energy_used_kwh < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Energy used cannot be negative".to_string()
        )));
    }
    if input.photos.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Too many photos attached to a single print record".to_string()
        )));
    }
    if input.notes.len() > 10_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Notes cannot exceed 10000 characters".to_string()
        )));
    }

    // Calculate PoGF components
    let energy_renewable = match job.energy_source {
        Some(EnergyType::Solar) | Some(EnergyType::Wind) |
        Some(EnergyType::Hydro) | Some(EnergyType::Geothermal) => 1.0,
        Some(EnergyType::Nuclear) => 0.8,
        Some(EnergyType::GridMix) => 0.3,
        _ => 0.0,
    };

    let material_circularity = job
        .material_passport
        .as_ref()
        .map(|p| p.recycled_content_percent / 100.0)
        .unwrap_or(0.0);

    // Quality from Cincinnati or issues
    let quality_score = if input.issues.is_empty() {
        0.9
    } else {
        1.0 - (input.issues.len() as f32 * 0.1).min(0.5)
    };

    // Local participation (HEARTH funding)
    let local_participation = if job.grounding_certificate.as_ref()
        .map(|c| c.hearth_funding_hash.is_some())
        .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };

    // Calculate raw PoGF score
    let raw_pog_score = calculate_pog_score(
        energy_renewable,
        material_circularity,
        quality_score,
        local_participation,
    );

    // Apply attestation multiplier: unattested claims are penalized
    let attestation_multiplier = input.attestations
        .as_ref()
        .map(|a| a.attestation_multiplier())
        .unwrap_or(0.25); // No attestations = 75% penalty
    let pog_score = raw_pog_score * attestation_multiplier;

    // Calculate MYCELIUM reward
    let mycelium_earned = if matches!(input.result, PrintResult::Success) {
        calculate_mycelium_reward(pog_score, quality_score)
    } else {
        0
    };

    // Carbon offset calculation (simplified)
    let carbon_offset = if energy_renewable >= 0.8 {
        Some(input.energy_used_kwh * 0.4) // ~400g CO2 per kWh avoided
    } else {
        None
    };

    // Dimensional accuracy
    let dimensional_accuracy = input.dimensional_measurements.map(|measurements| {
        let deviations: Vec<f32> = measurements
            .iter()
            .map(|m| (m.actual - m.expected).abs())
            .collect();
        let avg = deviations.iter().sum::<f32>() / deviations.len().max(1) as f32;
        let max = deviations.iter().cloned().fold(0.0_f32, f32::max);
        DimensionalAccuracy {
            average_deviation: avg,
            max_deviation: max,
            measurements,
        }
    });

    let now = sys_time()?;
    let record = PrintRecord {
        job_hash: input.job_hash.clone(),
        result: input.result,
        quality_score: Some(quality_score),
        pog_score,
        energy_used_kwh: input.energy_used_kwh,
        carbon_offset_kg: carbon_offset,
        material_circularity,
        mycelium_earned,
        cincinnati_report: None, // Would be fetched from Cincinnati session
        dimensional_accuracy,
        photos: input.photos,
        notes: input.notes,
        issues: input.issues,
        verification: None,
        recorded_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let record_hash = create_entry(EntryTypes::PrintRecord(record))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::ResultRecorded,
        payload: format!(r#"{{"hash":"{}"}}"#, record_hash),
    });

    // Link job to record
    create_link(
        input.job_hash,
        record_hash.clone(),
        LinkTypes::JobToRecord,
        (),
    )?;

    get(record_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve record".to_string()
    )))
}

/// Get print record for a job
#[hdk_extern]
pub fn get_print_record(job_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(job_hash, LinkTypes::JobToRecord)?, GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            return get(hash, GetOptions::default());
        }
    }

    Ok(None)
}

/// Add photos to a print record
#[derive(Serialize, Deserialize, Debug)]
pub struct AddPhotosInput {
    pub record_hash: ActionHash,
    pub photos: Vec<String>,
}

#[hdk_extern]
pub fn add_print_photos(input: AddPhotosInput) -> ExternResult<Record> {
    let record = get(input.record_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Record not found".to_string()
        )))?;

    let mut print_record: PrintRecord = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse record".to_string()
        )))?;

    // Only the printer owner can add photos
    let job = get_print_job_mut(print_record.job_hash.clone())?;
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    print_record.photos.extend(input.photos);

    let new_hash = update_entry(input.record_hash, EntryTypes::PrintRecord(print_record))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated record".to_string()
    )))
}

// =============================================================================
// CINCINNATI ALGORITHM MONITORING
// =============================================================================

/// Start Cincinnati monitoring for a print job
#[hdk_extern]
pub fn start_cincinnati_monitoring(input: StartCincinnatiInput) -> ExternResult<Record> {
    // Only the printer owner may start monitoring
    let job = get_print_job_mut(input.job_hash.clone())?;
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    let now = sys_time()?;
    let session_id = format!("cin_{}_{}", now.as_micros(), &input.job_hash.to_string()[..8]);

    let session = CincinnatiSession {
        session_id: session_id.clone(),
        estimator_version: "1.0.0".to_string(),
        sampling_rate_hz: input.sampling_rate_hz,
        baseline_signature: vec![],
        active: true,
        started_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let session_entry = CincinnatiSessionEntry {
        session,
        print_job_hash: input.job_hash.clone(),
        current_layer: 0,
        total_layers: input.total_layers,
        running_health_score: 1.0,
        anomaly_count: 0,
    };

    let session_hash = create_entry(EntryTypes::CincinnatiSessionEntry(session_entry))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::CincinnatiStarted,
        payload: format!(r#"{{"hash":"{}"}}"#, session_hash),
    });

    // Link job to session
    create_link(
        input.job_hash,
        session_hash.clone(),
        LinkTypes::JobToCincinnati,
        (),
    )?;

    // Link session anchor to session entry (for auth lookups in anomaly reporting)
    let session_anchor = cincinnati_session_anchor(&session_id)?;
    create_link(
        session_anchor,
        session_hash.clone(),
        LinkTypes::SessionAnchorToSession,
        (),
    )?;

    get(session_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve session".to_string()
    )))
}

/// Report an anomaly detected by Cincinnati monitoring
#[hdk_extern]
pub fn report_cincinnati_anomaly(input: ReportAnomalyInput) -> ExternResult<Record> {
    // Require anomalies to be reported by the printer owner.
    // Look up the session via SessionAnchorToSession, then trace to the job for auth.
    let session_anchor = cincinnati_session_anchor(&input.session_id)?;
    let session_links = get_links(
        LinkQuery::try_new(session_anchor.clone(), LinkTypes::SessionAnchorToSession)?,
        GetStrategy::default(),
    )?;
    if let Some(session_link) = session_links.first() {
        if let Some(session_hash) = session_link.target.clone().into_action_hash() {
            if let Some(session_record) = get(session_hash, GetOptions::default())? {
                if let Some(session_entry) = session_record
                    .entry()
                    .to_app_option::<CincinnatiSessionEntry>()
                    .ok()
                    .flatten()
                {
                    let job_record = get(session_entry.print_job_hash, GetOptions::default())?
                        .ok_or(wasm_error!(WasmErrorInner::Guest(
                            "Job not found for session".to_string()
                        )))?;
                    let job: PrintJob = job_record
                        .entry()
                        .to_app_option()
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                        .ok_or(wasm_error!(WasmErrorInner::Guest(
                            "Could not parse job".to_string()
                        )))?;
                    ensure_caller_is_printer_owner(&job.printer_hash)?;
                }
            }
        }
    }

    let anomaly_entry = CincinnatiAnomalyEntry {
        session_id: input.session_id.clone(),
        event: input.anomaly,
    };

    let anomaly_hash = create_entry(EntryTypes::CincinnatiAnomalyEntry(anomaly_entry))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Print,
        event_type: FabricationEventType::AnomalyDetected,
        payload: format!(r#"{{"hash":"{}"}}"#, anomaly_hash),
    });

    // Link to session via anchor
    let session_anchor = cincinnati_session_anchor(&input.session_id)?;
    create_link(
        session_anchor,
        anomaly_hash.clone(),
        LinkTypes::CincinnatiToAnomalies,
        (),
    )?;

    get(anomaly_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve anomaly".to_string()
    )))
}

/// Update Cincinnati session with layer progress
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCincinnatiInput {
    pub session_hash: ActionHash,
    pub current_layer: u32,
    pub health_score: f32,
    pub anomaly_count: u32,
}

#[hdk_extern]
pub fn update_cincinnati_session(input: UpdateCincinnatiInput) -> ExternResult<Record> {
    let record = get(input.session_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Session not found".to_string()
        )))?;

    let mut session: CincinnatiSessionEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse session".to_string()
        )))?;

    // Only the printer owner may update monitoring session state
    let job_record = get(session.print_job_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Job not found for session".to_string()
        )))?;
    let job: PrintJob = job_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse job".to_string()
        )))?;
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    // Validate float inputs
    if !input.health_score.is_finite() || input.health_score < 0.0 || input.health_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "health_score must be a finite number between 0.0 and 1.0".to_string()
        )));
    }

    session.current_layer = input.current_layer;
    session.running_health_score = input.health_score;
    session.anomaly_count = input.anomaly_count;

    let new_hash = update_entry(input.session_hash, EntryTypes::CincinnatiSessionEntry(session))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated session".to_string()
    )))
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Get all print jobs for the current agent
#[hdk_extern]
pub fn get_my_print_jobs(input: AgentPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let requester = agent_info()?.agent_initial_pubkey;
    let items = get_jobs_by_link(requester.into(), LinkTypes::RequesterToJobs)?;
    Ok(paginate(items, input.pagination.as_ref()))
}

/// Get print jobs for a printer
#[hdk_extern]
pub fn get_printer_jobs(input: HashPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let items = get_jobs_by_link(input.hash.into(), LinkTypes::PrinterToPrints)?;
    Ok(paginate(items, input.pagination.as_ref()))
}

/// Get prints for a design (internal: returns all records for statistics)
fn get_design_prints_all(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    get_jobs_by_link(design_hash.into(), LinkTypes::DesignToPrints)
}

/// Get prints for a design
#[hdk_extern]
pub fn get_design_prints(input: HashPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let items = get_design_prints_all(input.hash)?;
    Ok(paginate(items, input.pagination.as_ref()))
}

/// Get print statistics for a design
#[hdk_extern]
pub fn get_print_statistics(design_hash: ActionHash) -> ExternResult<DesignPrintStats> {
    let jobs = get_design_prints_all(design_hash.clone())?;

    let mut total = 0u32;
    let mut success = 0u32;
    let mut failed = 0u32;
    let mut quality_sum = 0.0f32;
    let mut pog_sum = 0.0f32;
    let mut mycelium_total = 0u64;
    let mut issue_counts: std::collections::HashMap<PrintIssue, u32> = std::collections::HashMap::new();

    for job_record in jobs {
        if let Some(job) = job_record.entry().to_app_option::<PrintJob>().ok().flatten() {
            total += 1;
            match job.status {
                PrintJobStatus::Completed => success += 1,
                PrintJobStatus::Failed => failed += 1,
                _ => {}
            }

            // Get the print record for more metrics
            if let Ok(Some(record)) = get_print_record(job_record.action_address().clone()) {
                if let Some(print_record) = record.entry().to_app_option::<PrintRecord>().ok().flatten() {
                    if let Some(q) = print_record.quality_score {
                        quality_sum += q;
                    }
                    pog_sum += print_record.pog_score;
                    mycelium_total += print_record.mycelium_earned;

                    for issue in &print_record.issues {
                        *issue_counts.entry(issue.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let avg_quality = if success > 0 { quality_sum / success as f32 } else { 0.0 };
    let avg_pog = if total > 0 { pog_sum / total as f32 } else { 0.0 };

    // Convert issues to sorted vec
    let mut common_issues: Vec<(PrintIssue, u32)> = issue_counts.into_iter().collect();
    common_issues.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(DesignPrintStats {
        design_hash,
        total_prints: total,
        successful_prints: success,
        failed_prints: failed,
        average_quality: avg_quality,
        average_pog_score: avg_pog,
        total_mycelium_earned: mycelium_total,
        common_issues,
    })
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn get_print_job_mut(hash: ActionHash) -> ExternResult<PrintJob> {
    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Job not found".to_string()
        )))?;

    record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse job".to_string()
        )))
}

fn get_jobs_by_link(base: AnyLinkableHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(base, link_type)?, GetStrategy::default(),
    )?;

    let mut jobs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                jobs.push(record);
            }
        }
    }

    Ok(jobs)
}

fn generate_job_id() -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!("job_{}_{}", now.as_micros(), &agent[..8.min(agent.len())])
}

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", name).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn pending_jobs_anchor() -> ExternResult<EntryHash> {
    make_anchor("pending_jobs")
}

fn all_jobs_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_jobs")
}

fn cincinnati_session_anchor(session_id: &str) -> ExternResult<EntryHash> {
    make_anchor(&format!("cincinnati_{}", session_id))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── calculate_pog_score ─────────────────────────────────────────────

    #[test]
    fn test_calculate_pog_score_all_zeros() {
        let score = calculate_pog_score(0.0, 0.0, 0.0, 0.0);
        assert!((score - 0.0).abs() < f32::EPSILON, "All-zero inputs should yield 0.0, got {}", score);
    }

    #[test]
    fn test_calculate_pog_score_all_ones() {
        let score = calculate_pog_score(1.0, 1.0, 1.0, 1.0);
        // 1.0*0.3 + 1.0*0.3 + 1.0*0.2 + 1.0*0.2 = 1.0
        assert!((score - 1.0).abs() < f32::EPSILON, "All-one inputs should yield 1.0, got {}", score);
    }

    #[test]
    fn test_calculate_pog_score_typical_values() {
        // Renewable energy=0.8, circularity=0.5, quality=0.9, local=0.0
        let score = calculate_pog_score(0.8, 0.5, 0.9, 0.0);
        // 0.8*0.3 + 0.5*0.3 + 0.9*0.2 + 0.0*0.2 = 0.24 + 0.15 + 0.18 + 0.0 = 0.57
        let expected = 0.24 + 0.15 + 0.18;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected ~{}, got {}",
            expected, score
        );
    }

    #[test]
    fn test_calculate_pog_score_weights_sum_to_one() {
        // With all inputs at the same value x, the result should be x
        let x = 0.42;
        let score = calculate_pog_score(x, x, x, x);
        // x*(0.3+0.3+0.2+0.2) = x*1.0 = x
        assert!(
            (score - x).abs() < 1e-6,
            "Uniform inputs of {} should yield {}, got {}",
            x, x, score
        );
    }

    #[test]
    fn test_calculate_pog_score_clamps_above_one() {
        // Values above 1.0 should be clamped to 1.0
        let score = calculate_pog_score(2.0, 1.5, 3.0, 10.0);
        // After clamping: all 1.0 → score = 1.0
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "Over-range inputs should clamp to 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_calculate_pog_score_clamps_below_zero() {
        // Negative values should be clamped to 0.0
        let score = calculate_pog_score(-1.0, -0.5, -3.0, -10.0);
        assert!(
            (score - 0.0).abs() < f32::EPSILON,
            "Negative inputs should clamp to 0.0, got {}",
            score
        );
    }

    #[test]
    fn test_calculate_pog_score_mixed_clamp() {
        // Mix of in-range, above, and below values
        let score = calculate_pog_score(-0.5, 0.5, 1.5, 0.8);
        // After clamping: 0.0, 0.5, 1.0, 0.8
        // 0.0*0.3 + 0.5*0.3 + 1.0*0.2 + 0.8*0.2 = 0.0 + 0.15 + 0.20 + 0.16 = 0.51
        let expected = 0.0 + 0.15 + 0.20 + 0.16;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected ~{}, got {}",
            expected, score
        );
    }

    #[test]
    fn test_calculate_pog_score_energy_only() {
        // Only energy component active
        let score = calculate_pog_score(1.0, 0.0, 0.0, 0.0);
        assert!(
            (score - 0.3).abs() < f32::EPSILON,
            "Energy-only should yield 0.3, got {}",
            score
        );
    }

    #[test]
    fn test_calculate_pog_score_local_only() {
        // Only local participation active
        let score = calculate_pog_score(0.0, 0.0, 0.0, 1.0);
        assert!(
            (score - 0.2).abs() < f32::EPSILON,
            "Local-only should yield 0.2, got {}",
            score
        );
    }

    // ── calculate_mycelium_reward ───────────────────────────────────────

    #[test]
    fn test_mycelium_reward_below_threshold_returns_zero() {
        // pog_score < 0.6 → no reward
        assert_eq!(calculate_mycelium_reward(0.59, 1.0), 0);
        assert_eq!(calculate_mycelium_reward(0.0, 1.0), 0);
        assert_eq!(calculate_mycelium_reward(0.3, 0.9), 0);
    }

    #[test]
    fn test_mycelium_reward_at_threshold_boundary() {
        // pog_score exactly at 0.6 should earn a reward (not below threshold)
        // multiplier = (0.6 + 0.9) / 2.0 = 0.75
        // bonus = (0.6 - 0.6) * 50.0 = 0.0
        // total = 10.0 * 0.75 + 0.0 = 7.5 → 7
        let reward = calculate_mycelium_reward(0.6, 0.9);
        assert_eq!(reward, 7, "At threshold with quality=0.9, expected 7, got {}", reward);
    }

    #[test]
    fn test_mycelium_reward_above_threshold() {
        // pog=0.8, quality=0.9
        // multiplier = (0.8 + 0.9) / 2.0 = 0.85
        // bonus = (0.8 - 0.6) * 50.0 = 10.0
        // total = 10.0 * 0.85 + 10.0 = 18.5 → 18
        let reward = calculate_mycelium_reward(0.8, 0.9);
        assert_eq!(reward, 18, "pog=0.8, quality=0.9 should yield 18, got {}", reward);
    }

    #[test]
    fn test_mycelium_reward_perfect_scores() {
        // pog=1.0, quality=1.0
        // multiplier = (1.0 + 1.0) / 2.0 = 1.0
        // bonus = (1.0 - 0.6f32) * 50.0
        // 0.6f32 is ~0.6000000238, so bonus = 0.3999999762 * 50.0 = ~19.999998
        // total = 10.0 * 1.0 + ~19.999998 = ~29.999998 → truncated to 29
        let reward = calculate_mycelium_reward(1.0, 1.0);
        assert_eq!(reward, 29, "Perfect scores should yield 29 (f32 truncation of ~29.999998), got {}", reward);
    }

    #[test]
    fn test_mycelium_reward_high_pog_low_quality() {
        // pog=1.0, quality=0.0
        // multiplier = (1.0 + 0.0) / 2.0 = 0.5
        // bonus = (1.0 - 0.6f32) * 50.0 = ~19.999998 (f32 precision)
        // total = 10.0 * 0.5 + ~19.999998 = ~24.999998 → truncated to 24
        let reward = calculate_mycelium_reward(1.0, 0.0);
        assert_eq!(reward, 24, "High pog, zero quality should yield 24 (f32 truncation), got {}", reward);
    }

    #[test]
    fn test_mycelium_reward_just_above_threshold() {
        // pog=0.61, quality=0.5
        // multiplier = (0.61 + 0.5) / 2.0 = 0.555
        // bonus = (0.61 - 0.6) * 50.0 = 0.5
        // total = 10.0 * 0.555 + 0.5 = 6.05 → 6
        let reward = calculate_mycelium_reward(0.61, 0.5);
        assert_eq!(reward, 6, "Just above threshold should yield 6, got {}", reward);
    }

    #[test]
    fn test_mycelium_reward_zero_pog_zero_quality() {
        // Below threshold → 0
        assert_eq!(calculate_mycelium_reward(0.0, 0.0), 0);
    }

    #[test]
    fn test_mycelium_reward_negative_pog() {
        // Negative pog is below threshold → 0
        assert_eq!(calculate_mycelium_reward(-1.0, 1.0), 0);
    }

    #[test]
    fn test_mycelium_reward_threshold_with_zero_quality() {
        // pog=0.6, quality=0.0
        // multiplier = (0.6 + 0.0) / 2.0 = 0.3
        // bonus = (0.6 - 0.6) * 50.0 = 0.0
        // total = 10.0 * 0.3 + 0.0 = 3.0 → 3
        let reward = calculate_mycelium_reward(0.6, 0.0);
        assert_eq!(reward, 3, "Threshold pog with zero quality should yield 3, got {}", reward);
    }

    // ── pog_score with attestation multiplier ──────────────────────────

    #[test]
    fn test_pog_score_with_full_attestation() {
        // Full attestation → multiplier = 1.0 → no penalty
        let raw = calculate_pog_score(1.0, 1.0, 1.0, 1.0);
        let bundle = PogfAttestationBundle {
            energy: Some(EnergyAttestation {
                energy_type: EnergyType::Solar,
                kwh_consumed: 1.0,
                grid_carbon_intensity: 25.0,
                terra_atlas_hash: None,
                attester: "a".to_string(),
                attested_at_micros: 0,
            }),
            material: Some(MaterialAttestation {
                batch_id: "b".to_string(),
                origin: MaterialOrigin::PostConsumer,
                recycled_fraction: 0.9,
                supply_chain_hash: None,
                certifications: vec![],
                attester: "b".to_string(),
                attested_at_micros: 0,
            }),
            local: Some(LocalAttestation {
                hearth_funding_hash: Some("h".to_string()),
                local_printer: true,
                distance_km: None,
                attester: "c".to_string(),
                attested_at_micros: 0,
            }),
        };
        let adjusted = raw * bundle.attestation_multiplier();
        assert!((adjusted - raw).abs() < 0.001, "Full attestation should not penalize");
    }

    #[test]
    fn test_pog_score_with_no_attestation() {
        let raw = calculate_pog_score(1.0, 1.0, 1.0, 1.0);
        // No attestations → multiplier = 0.25
        let adjusted = raw * 0.25;
        assert!((adjusted - 0.25).abs() < 0.001, "No attestation should yield 25% of raw");
    }

    #[test]
    fn test_pog_score_partial_attestation_scales() {
        let raw = calculate_pog_score(1.0, 1.0, 1.0, 1.0);
        let bundle = PogfAttestationBundle {
            energy: Some(EnergyAttestation {
                energy_type: EnergyType::Wind,
                kwh_consumed: 0.5,
                grid_carbon_intensity: 0.0,
                terra_atlas_hash: None,
                attester: "a".to_string(),
                attested_at_micros: 0,
            }),
            material: Some(MaterialAttestation {
                batch_id: "b".to_string(),
                origin: MaterialOrigin::Virgin,
                recycled_fraction: 0.0,
                supply_chain_hash: None,
                certifications: vec![],
                attester: "b".to_string(),
                attested_at_micros: 0,
            }),
            local: None,
        };
        let m = bundle.attestation_multiplier();
        let adjusted = raw * m;
        // 2/3 attested → 0.25 + 0.75*(2/3) = 0.75
        assert!((adjusted - 0.75).abs() < 0.001, "2/3 attested should yield 0.75, got {}", adjusted);
    }
}
