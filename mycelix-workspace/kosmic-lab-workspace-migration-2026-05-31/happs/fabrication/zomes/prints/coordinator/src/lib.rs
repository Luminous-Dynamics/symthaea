// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Prints Coordinator Zome
//!
//! This zome provides the coordinator functions for managing print jobs
//! in the Mycelix Fabrication hApp. It implements:
//!
//! - Print job lifecycle management
//! - Proof of Grounded Fabrication (PoGF) scoring
//! - Cincinnati Algorithm for quality monitoring
//! - MYCELIUM (CIV) reputation integration

use fabrication_common::*;
use hdk::prelude::*;
use prints_integrity::*;

// Local Printer type for cross-zome data deserialization
// (importing printers_integrity causes duplicate symbol errors)
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes, PartialEq)]
pub struct Printer {
    pub id: String,
    pub name: String,
    pub owner: AgentPubKey,
    pub location: Option<GeoLocation>,
    pub printer_type: PrinterType,
    pub capabilities: PrinterCapabilities,
    pub materials_available: Vec<MaterialType>,
    pub availability: AvailabilityStatus,
    pub rates: Option<PrinterRates>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

// =============================================================================
// PROOF OF GROUNDED FABRICATION (PoGF)
// =============================================================================

/// PoGF scoring weights
const POG_ENERGY_WEIGHT: f32 = 0.3;
const POG_MATERIAL_WEIGHT: f32 = 0.3;
const POG_QUALITY_WEIGHT: f32 = 0.2;
const POG_LOCAL_WEIGHT: f32 = 0.2;

/// Minimum PoGF score to earn MYCELIUM reputation
const MIN_POG_FOR_MYCELIUM: f32 = 0.6;

/// Base MYCELIUM reward per successful print
const BASE_MYCELIUM_REWARD: u64 = 10;

/// Calculate Proof of Grounded Fabrication score
/// PoG = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)
fn calculate_pog_score(
    energy_renewable_fraction: f32,
    material_circularity: f32,
    quality_verified: f32,
    local_participation: f32,
) -> f32 {
    let e = energy_renewable_fraction.clamp(0.0, 1.0);
    let m = material_circularity.clamp(0.0, 1.0);
    let q = quality_verified.clamp(0.0, 1.0);
    let l = local_participation.clamp(0.0, 1.0);

    (e * POG_ENERGY_WEIGHT)
        + (m * POG_MATERIAL_WEIGHT)
        + (q * POG_QUALITY_WEIGHT)
        + (l * POG_LOCAL_WEIGHT)
}

/// Calculate MYCELIUM (CIV) reputation earned from a print
fn calculate_mycelium_reward(pog_score: f32, quality_score: f32) -> u64 {
    if pog_score < MIN_POG_FOR_MYCELIUM {
        return 0;
    }

    // Base reward multiplied by PoGF and quality scores
    let multiplier = (pog_score + quality_score) / 2.0;
    let bonus = (pog_score - MIN_POG_FOR_MYCELIUM) * 50.0; // Bonus for exceeding minimum

    (BASE_MYCELIUM_REWARD as f32 * multiplier + bonus) as u64
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
    let record = get(printer_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Printer not found".to_string())
    ))?;

    let printer: Printer = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse printer".to_string()
        )))?;

    if printer.owner != caller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the printer owner can perform this action".to_string()
        )));
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

/// Create a new print job
#[hdk_extern]
pub fn create_print_job(input: CreatePrintJobInput) -> ExternResult<Record> {
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

    let _now = sys_time()?; // Reserved for future timestamp tracking
    let updated = PrintJob {
        status: PrintJobStatus::Accepted,
        ..job
    };

    let new_hash = update_entry(hash, EntryTypes::PrintJob(updated))?;

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

    if !matches!(
        job.status,
        PrintJobStatus::Accepted | PrintJobStatus::Queued
    ) {
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

    if matches!(
        job.status,
        PrintJobStatus::Completed | PrintJobStatus::Cancelled
    ) {
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
    let job_record = get(input.job_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Job not found".to_string())
    ))?;

    let job: PrintJob = job_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse job".to_string()
        )))?;

    // Only the printer owner may submit print results
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    // Basic input bounds to avoid abuse
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
        Some(EnergyType::Solar)
        | Some(EnergyType::Wind)
        | Some(EnergyType::Hydro)
        | Some(EnergyType::Geothermal) => 1.0,
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
    let local_participation = if job
        .grounding_certificate
        .as_ref()
        .map(|c| c.hearth_funding_hash.is_some())
        .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };

    // Calculate PoGF score
    let pog_score = calculate_pog_score(
        energy_renewable,
        material_circularity,
        quality_score,
        local_participation,
    );

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
        LinkQuery::try_new(job_hash, LinkTypes::JobToRecord)?,
        GetStrategy::default(),
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
    let record = get(input.record_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Record not found".to_string())
    ))?;

    let mut print_record: PrintRecord = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse record".to_string()
        )))?;

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
    let session_id = format!(
        "cin_{}_{}",
        now.as_micros(),
        input.job_hash.to_string()[..8].to_string()
    );

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

    // Link job to session
    create_link(
        input.job_hash,
        session_hash.clone(),
        LinkTypes::JobToCincinnati,
        (),
    )?;

    get(session_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve session".to_string()
    )))
}

/// Report an anomaly detected by Cincinnati monitoring
#[hdk_extern]
pub fn report_cincinnati_anomaly(input: ReportAnomalyInput) -> ExternResult<Record> {
    // For now, require anomalies to be reported by the printer owner
    // by looking up the session and associated job.
    let session_anchor = cincinnati_session_anchor(&input.session_id)?;
    let links = get_links(
        LinkQuery::try_new(session_anchor.clone(), LinkTypes::CincinnatiToAnomalies)?,
        GetStrategy::default(),
    )?;
    // If there are existing anomalies, use one to trace back to the session/job; otherwise skip auth
    if let Some(existing) = links.first() {
        if let Some(anomaly_hash) = existing.target.clone().into_action_hash() {
            if let Some(record) = get(anomaly_hash, GetOptions::default())? {
                if let Some(existing_anomaly) = record
                    .entry()
                    .to_app_option::<CincinnatiAnomalyEntry>()
                    .ok()
                    .flatten()
                {
                    // Use the session id to find the session entry and job
                    let session_links = get_links(
                        LinkQuery::try_new(
                            cincinnati_session_anchor(&existing_anomaly.session_id)?,
                            LinkTypes::CincinnatiToAnomalies,
                        )?,
                        GetStrategy::default(),
                    )?;
                    let _ = session_links; // Best-effort; full auth can be added when session index is richer
                }
            }
        }
    }

    let anomaly_entry = CincinnatiAnomalyEntry {
        session_id: input.session_id.clone(),
        event: input.anomaly,
    };

    let anomaly_hash = create_entry(EntryTypes::CincinnatiAnomalyEntry(anomaly_entry))?;

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
    let record = get(input.session_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Session not found".to_string())
    ))?;

    let mut session: CincinnatiSessionEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse session".to_string()
        )))?;

    // Only the printer owner may update monitoring session state
    let job_record =
        get(session.print_job_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Job not found for session".to_string())
        ))?;
    let job: PrintJob = job_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse job".to_string()
        )))?;
    ensure_caller_is_printer_owner(&job.printer_hash)?;

    session.current_layer = input.current_layer;
    session.running_health_score = input.health_score;
    session.anomaly_count = input.anomaly_count;

    let new_hash = update_entry(
        input.session_hash,
        EntryTypes::CincinnatiSessionEntry(session),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated session".to_string()
    )))
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Get all print jobs for the current agent
#[hdk_extern]
pub fn get_my_print_jobs(_: ()) -> ExternResult<Vec<Record>> {
    let requester = agent_info()?.agent_initial_pubkey;
    get_jobs_by_link(requester.into(), LinkTypes::RequesterToJobs)
}

/// Get print jobs for a printer
#[hdk_extern]
pub fn get_printer_jobs(printer_hash: ActionHash) -> ExternResult<Vec<Record>> {
    get_jobs_by_link(printer_hash.into(), LinkTypes::PrinterToPrints)
}

/// Get prints for a design
#[hdk_extern]
pub fn get_design_prints(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    get_jobs_by_link(design_hash.into(), LinkTypes::DesignToPrints)
}

/// Get print statistics for a design
#[hdk_extern]
pub fn get_print_statistics(design_hash: ActionHash) -> ExternResult<DesignPrintStats> {
    let jobs = get_design_prints(design_hash.clone())?;

    let mut total = 0u32;
    let mut success = 0u32;
    let mut failed = 0u32;
    let mut quality_sum = 0.0f32;
    let mut pog_sum = 0.0f32;
    let mut mycelium_total = 0u64;
    let mut issue_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

    for job_record in jobs {
        if let Some(job) = job_record
            .entry()
            .to_app_option::<PrintJob>()
            .ok()
            .flatten()
        {
            total += 1;
            match job.status {
                PrintJobStatus::Completed => success += 1,
                PrintJobStatus::Failed => failed += 1,
                _ => {}
            }

            // Get the print record for more metrics
            if let Ok(Some(record)) = get_print_record(job_record.action_address().clone()) {
                if let Some(print_record) =
                    record.entry().to_app_option::<PrintRecord>().ok().flatten()
                {
                    if let Some(q) = print_record.quality_score {
                        quality_sum += q;
                    }
                    pog_sum += print_record.pog_score;
                    mycelium_total += print_record.mycelium_earned;

                    for issue in &print_record.issues {
                        let key = format!("{:?}", issue);
                        *issue_counts.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let avg_quality = if success > 0 {
        quality_sum / success as f32
    } else {
        0.0
    };
    let avg_pog = if total > 0 {
        pog_sum / total as f32
    } else {
        0.0
    };

    // Convert issues to sorted vec
    let mut common_issues: Vec<(PrintIssue, u32)> = issue_counts
        .into_iter()
        .filter_map(|(key, count)| {
            // Parse back to PrintIssue (simplified)
            if key.contains("Warping") {
                Some((PrintIssue::Warping, count))
            } else if key.contains("LayerShift") {
                Some((PrintIssue::LayerShift, count))
            } else if key.contains("Stringing") {
                Some((PrintIssue::Stringing, count))
            } else {
                Some((PrintIssue::Other(key), count))
            }
        })
        .collect();
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
    let record = get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    let links = get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::default())?;

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
    let anchor_bytes =
        SerializedBytes::from(UnsafeBytes::from(format!("anchor:{}", name).into_bytes()));
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
