// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Prints Integrity Zome
//!
//! This zome defines the entry types and validation rules for print jobs
//! in the Mycelix Fabrication hApp. It implements:
//!
//! - Proof of Grounded Fabrication (PoGF) for metabolic accountability
//! - Cincinnati Algorithm for teleomorphic quality monitoring
//! - MYCELIUM (CIV) integration for reputation earning

use fabrication_common::*;
use hdi::prelude::*;

/// Entry types for the prints zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A print job request
    #[entry_type(visibility = "public")]
    PrintJob(PrintJob),
    /// Completed print record with quality data
    #[entry_type(visibility = "public")]
    PrintRecord(PrintRecord),
    /// Cincinnati monitoring session
    #[entry_type(visibility = "public")]
    CincinnatiSessionEntry(CincinnatiSessionEntry),
    /// Individual anomaly event from Cincinnati
    #[entry_type(visibility = "public")]
    CincinnatiAnomalyEntry(CincinnatiAnomalyEntry),
}

/// Link types for the prints zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from design to print jobs
    DesignToPrints,
    /// Link from printer to print jobs
    PrinterToPrints,
    /// Link from requester to their jobs
    RequesterToJobs,
    /// Link from job to print record
    JobToRecord,
    /// Link from job to Cincinnati session
    JobToCincinnati,
    /// Link from Cincinnati session to anomaly events
    CincinnatiToAnomalies,
    /// Link for all print jobs
    AllPrintJobs,
    /// Link for pending jobs
    PendingJobs,
}

/// A print job entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrintJob {
    /// Unique identifier
    pub id: String,
    /// Design being printed
    pub design_hash: ActionHash,
    /// Printer executing the job
    pub printer_hash: ActionHash,
    /// Who requested the print
    pub requester: AgentPubKey,
    /// Print settings
    pub settings: PrintSettings,
    /// Current status
    pub status: PrintJobStatus,

    // === PROOF OF GROUNDED FABRICATION ===
    /// Grounding certificate for metabolic accountability
    pub grounding_certificate: Option<GroundingCertificate>,
    /// Energy source used
    pub energy_source: Option<EnergyType>,
    /// Material passport for traceability
    pub material_passport: Option<MaterialPassport>,

    // === CINCINNATI MONITORING ===
    /// Active Cincinnati monitoring session
    pub cincinnati_session: Option<String>, // Session ID
    /// Quality predictions from Cincinnati
    pub quality_predictions: Vec<QualityPrediction>,

    /// Estimated time in minutes
    pub estimated_time_minutes: Option<u32>,
    /// Actual time taken
    pub actual_time_minutes: Option<u32>,
    /// Material used in grams
    pub material_used_grams: Option<f32>,
    /// When job was created
    pub created_at: Timestamp,
    /// When printing started
    pub started_at: Option<Timestamp>,
    /// When printing completed
    pub completed_at: Option<Timestamp>,
}

/// Print record with quality data and metabolic metrics
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrintRecord {
    /// Link to the print job
    pub job_hash: ActionHash,
    /// Result of the print
    pub result: PrintResult,
    /// Quality score from Cincinnati
    pub quality_score: Option<f32>,

    // === METABOLIC METRICS (PoGF) ===
    /// Proof of Grounding composite score
    /// PoG = (E×0.3) + (M×0.3) + (Q×0.2) + (L×0.2)
    pub pog_score: f32,
    /// Actual energy consumed
    pub energy_used_kwh: f32,
    /// Carbon offset if renewable powered
    pub carbon_offset_kg: Option<f32>,
    /// Material circularity (recycled content %)
    pub material_circularity: f32,
    /// CIV reputation earned from this print
    pub mycelium_earned: u64,

    // === QUALITY ASSURANCE ===
    /// Cincinnati monitoring report
    pub cincinnati_report: Option<CincinnatiReport>,
    /// Dimensional accuracy measurements
    pub dimensional_accuracy: Option<DimensionalAccuracy>,

    /// Photos of the completed print
    pub photos: Vec<String>,
    /// Notes from the operator
    pub notes: String,
    /// Issues encountered during print
    pub issues: Vec<PrintIssue>,
    /// Link to verification entry
    pub verification: Option<ActionHash>,
    /// When record was created
    pub recorded_at: Timestamp,
}

/// Cincinnati monitoring session entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CincinnatiSessionEntry {
    /// Session configuration
    pub session: CincinnatiSession,
    /// Print job being monitored
    pub print_job_hash: ActionHash,
    /// Current layer being printed
    pub current_layer: u32,
    /// Total layers expected
    pub total_layers: u32,
    /// Running health score
    pub running_health_score: f32,
    /// Number of anomalies so far
    pub anomaly_count: u32,
}

/// Individual anomaly event from Cincinnati monitoring
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CincinnatiAnomalyEntry {
    /// Session this anomaly belongs to
    pub session_id: String,
    /// The anomaly event details
    pub event: AnomalyEvent,
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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::PrintJob(job) => validate_print_job(job),
        EntryTypes::PrintRecord(record) => validate_print_record(record),
        EntryTypes::CincinnatiSessionEntry(session) => validate_cincinnati_session(session),
        EntryTypes::CincinnatiAnomalyEntry(anomaly) => validate_cincinnati_anomaly(anomaly),
    }
}

/// Validate a print job
fn validate_print_job(job: PrintJob) -> ExternResult<ValidateCallbackResult> {
    // Layer height must be positive
    if job.settings.layer_height <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Layer height must be positive".to_string(),
        ));
    }

    // Infill must be 0-100
    if job.settings.infill_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Infill must be between 0 and 100".to_string(),
        ));
    }

    // Estimated time should be reasonable
    if let Some(time) = job.estimated_time_minutes {
        if time > 43200 {
            // 30 days
            return Ok(ValidateCallbackResult::Invalid(
                "Estimated time exceeds 30 days".to_string(),
            ));
        }
    }

    // Validate PoGF certificate if present
    if let Some(ref cert) = job.grounding_certificate {
        if cert.grid_carbon_intensity < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Grid carbon intensity cannot be negative".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a print record
fn validate_print_record(record: PrintRecord) -> ExternResult<ValidateCallbackResult> {
    // PoGF score must be 0-1
    if record.pog_score < 0.0 || record.pog_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "PoGF score must be between 0 and 1".to_string(),
        ));
    }

    // Energy used must be non-negative
    if record.energy_used_kwh < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Energy used cannot be negative".to_string(),
        ));
    }

    // Material circularity must be 0-1
    if record.material_circularity < 0.0 || record.material_circularity > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Material circularity must be between 0 and 1".to_string(),
        ));
    }

    // Quality score must be 0-1 if present
    if let Some(score) = record.quality_score {
        if score < 0.0 || score > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Quality score must be between 0 and 1".to_string(),
            ));
        }
    }

    // Notes length limit
    if record.notes.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Notes cannot exceed 10000 characters".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Cincinnati session
fn validate_cincinnati_session(
    session: CincinnatiSessionEntry,
) -> ExternResult<ValidateCallbackResult> {
    // Sampling rate must be reasonable
    if session.session.sampling_rate_hz == 0 || session.session.sampling_rate_hz > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Sampling rate must be between 1 and 10000 Hz".to_string(),
        ));
    }

    // Running health score must be 0-1
    if session.running_health_score < 0.0 || session.running_health_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Health score must be between 0 and 1".to_string(),
        ));
    }

    // Current layer cannot exceed total
    if session.current_layer > session.total_layers {
        return Ok(ValidateCallbackResult::Invalid(
            "Current layer cannot exceed total layers".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Cincinnati anomaly
fn validate_cincinnati_anomaly(
    anomaly: CincinnatiAnomalyEntry,
) -> ExternResult<ValidateCallbackResult> {
    // Severity must be 0-1
    if anomaly.event.severity < 0.0 || anomaly.event.severity > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Anomaly severity must be between 0 and 1".to_string(),
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
        LinkTypes::DesignToPrints => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterToPrints => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RequesterToJobs => Ok(ValidateCallbackResult::Valid),
        LinkTypes::JobToRecord => Ok(ValidateCallbackResult::Valid),
        LinkTypes::JobToCincinnati => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CincinnatiToAnomalies => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllPrintJobs => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PendingJobs => Ok(ValidateCallbackResult::Valid),
    }
}
