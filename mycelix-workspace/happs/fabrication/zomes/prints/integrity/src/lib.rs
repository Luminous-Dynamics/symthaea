// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prints Integrity Zome
//!
//! This zome defines the entry types and validation rules for print jobs
//! in the Mycelix Fabrication hApp. It implements:
//!
//! - Proof of Grounded Fabrication (PoGF) for metabolic accountability
//! - Cincinnati Algorithm for teleomorphic quality monitoring
//! - MYCELIUM (CIV) integration for reputation earning

use hdi::prelude::*;
use fabrication_common::*;
use fabrication_common::validation;

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
    /// Link from Cincinnati session anchor to session entry (for auth lookups)
    SessionAnchorToSession,
    /// Link for all print jobs
    AllPrintJobs,
    /// Link for pending jobs
    PendingJobs,
    /// Per-agent rate limiting bucket
    RateLimitBucket,
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
            match op_update {
                // PrintJob supports multi-party updates (requester + printer operator)
                OpUpdate::Entry { app_entry: EntryTypes::PrintJob(_), .. } => {
                    Ok(ValidateCallbackResult::Valid)
                }
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => {
                    let original = must_get_action(action.original_action_address.clone())?;
                    if action.author != *original.hashed.author() {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Only the original author can update this entry".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
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
        EntryTypes::PrintJob(job) => validate_print_job(job),
        EntryTypes::PrintRecord(record) => validate_print_record(record),
        EntryTypes::CincinnatiSessionEntry(session) => validate_cincinnati_session(session),
        EntryTypes::CincinnatiAnomalyEntry(anomaly) => validate_cincinnati_anomaly(anomaly),
    }
}

/// Validate a print job
fn validate_print_job(job: PrintJob) -> ExternResult<ValidateCallbackResult> {
    // Job ID: non-empty, max 64 chars
    check!(validation::require_non_empty(&job.id, "job.id"));
    check!(validation::require_max_len(&job.id, 64, "job.id"));

    // Layer height: finite, in range 0.01..2.0
    check!(validation::require_in_range(
        job.settings.layer_height, 0.01, 2.0, "settings.layer_height"
    ));

    // Infill must be 0-100 (u8, so >= 0 is guaranteed)
    if job.settings.infill_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Infill must be between 0 and 100".to_string(),
        ));
    }

    // Estimated time: max 43200 minutes (30 days)
    if let Some(time) = job.estimated_time_minutes {
        if time > 43200 {
            return Ok(ValidateCallbackResult::Invalid(
                "Estimated time exceeds 30 days".to_string(),
            ));
        }
    }

    // Material used: finite + non-negative
    if let Some(grams) = job.material_used_grams {
        check!(validation::require_in_range(
            grams, 0.0, f32::MAX, "material_used_grams"
        ));
    }

    // Grounding certificate: grid_carbon_intensity finite + non-negative
    if let Some(ref cert) = job.grounding_certificate {
        check!(validation::require_in_range(
            cert.grid_carbon_intensity, 0.0, f32::MAX, "grid_carbon_intensity"
        ));
    }

    // Quality predictions: max 100 items
    check!(validation::require_max_vec_len(
        &job.quality_predictions, 100, "quality_predictions"
    ));

    // Cincinnati session ID: max 256 chars
    if let Some(ref session_id) = job.cincinnati_session {
        check!(validation::require_max_len(session_id, 256, "cincinnati_session"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a print record
fn validate_print_record(record: PrintRecord) -> ExternResult<ValidateCallbackResult> {
    // PoGF score: finite, in range 0.0..1.0
    check!(validation::require_in_range(
        record.pog_score, 0.0, 1.0, "pog_score"
    ));

    // Energy used: finite + non-negative
    check!(validation::require_in_range(
        record.energy_used_kwh, 0.0, f32::MAX, "energy_used_kwh"
    ));

    // Material circularity: finite, in range 0.0..1.0
    check!(validation::require_in_range(
        record.material_circularity, 0.0, 1.0, "material_circularity"
    ));

    // Quality score: if present, finite, in range 0.0..1.0
    if let Some(score) = record.quality_score {
        check!(validation::require_in_range(
            score, 0.0, 1.0, "quality_score"
        ));
    }

    // Carbon offset: if present, finite + non-negative
    if let Some(offset) = record.carbon_offset_kg {
        check!(validation::require_in_range(
            offset, 0.0, f32::MAX, "carbon_offset_kg"
        ));
    }

    // Photos: max 50 items
    check!(validation::require_max_vec_len(&record.photos, 50, "photos"));

    // Issues: max 100 items
    check!(validation::require_max_vec_len(&record.issues, 100, "issues"));

    // Notes: max 10000 chars
    check!(validation::require_max_len(&record.notes, 10000, "notes"));

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Cincinnati session
fn validate_cincinnati_session(session: CincinnatiSessionEntry) -> ExternResult<ValidateCallbackResult> {
    // Sampling rate must be 1-10000 Hz
    if session.session.sampling_rate_hz == 0 || session.session.sampling_rate_hz > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Sampling rate must be between 1 and 10000 Hz".to_string(),
        ));
    }

    // Running health score: finite, in range 0.0..1.0
    check!(validation::require_in_range(
        session.running_health_score, 0.0, 1.0, "running_health_score"
    ));

    // Current layer cannot exceed total layers
    if session.current_layer > session.total_layers {
        return Ok(ValidateCallbackResult::Invalid(
            "Current layer cannot exceed total layers".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Cincinnati anomaly
fn validate_cincinnati_anomaly(anomaly: CincinnatiAnomalyEntry) -> ExternResult<ValidateCallbackResult> {
    // session_id: max 256 chars
    check!(validation::require_max_len(&anomaly.session_id, 256, "session_id"));

    // Severity: finite, in range 0.0..1.0
    check!(validation::require_in_range(
        anomaly.event.severity, 0.0, 1.0, "severity"
    ));

    // Sensor data: all required fields must be finite
    let sd = &anomaly.event.sensor_data;
    check!(validation::require_finite(sd.hotend_temp, "sensor_data.hotend_temp"));
    check!(validation::require_finite(sd.bed_temp, "sensor_data.bed_temp"));
    for (i, &current) in sd.stepper_currents.iter().enumerate() {
        check!(validation::require_finite(
            current,
            &format!("sensor_data.stepper_currents[{}]", i)
        ));
    }
    check!(validation::require_finite(sd.vibration_rms, "sensor_data.vibration_rms"));

    // Optional sensor fields: if present, must be finite
    if let Some(tension) = sd.filament_tension {
        check!(validation::require_finite(tension, "sensor_data.filament_tension"));
    }
    if let Some(temp) = sd.ambient_temp {
        check!(validation::require_finite(temp, "sensor_data.ambient_temp"));
    }
    if let Some(hum) = sd.humidity {
        check!(validation::require_finite(hum, "sensor_data.humidity"));
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
    let max_len: usize = 256;
    check!(validation::require_max_tag_len(&tag, max_len, &format!("{:?}", link_type)));
    match link_type {
        LinkTypes::DesignToPrints => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PrinterToPrints => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RequesterToJobs => Ok(ValidateCallbackResult::Valid),
        LinkTypes::JobToRecord => Ok(ValidateCallbackResult::Valid),
        LinkTypes::JobToCincinnati => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CincinnatiToAnomalies => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllPrintJobs => Ok(ValidateCallbackResult::Valid),
        LinkTypes::PendingJobs => Ok(ValidateCallbackResult::Valid),
        LinkTypes::SessionAnchorToSession => Ok(ValidateCallbackResult::Valid),
        LinkTypes::RateLimitBucket => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test helpers
    // =========================================================================

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn test_agent_pub_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn test_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn valid_print_settings() -> PrintSettings {
        PrintSettings {
            layer_height: 0.2,
            infill_percent: 20,
            material: MaterialType::PLA,
            supports: false,
            raft: false,
            print_speed: Some(60),
            temperatures: TemperatureSettings {
                hotend: 210,
                bed: Some(60),
                chamber: None,
            },
            custom_gcode: None,
        }
    }

    fn valid_grounding_certificate() -> GroundingCertificate {
        GroundingCertificate {
            certificate_id: "cert-001".to_string(),
            terra_atlas_energy_hash: None,
            energy_type: EnergyType::Solar,
            grid_carbon_intensity: 50.0,
            material_passports: vec![],
            hearth_funding_hash: None,
            issued_at: test_timestamp(),
            issuer_signature: vec![0u8; 32],
        }
    }

    fn valid_print_job() -> PrintJob {
        PrintJob {
            id: "job-001".to_string(),
            design_hash: test_action_hash(),
            printer_hash: test_action_hash(),
            requester: test_agent_pub_key(),
            settings: valid_print_settings(),
            status: PrintJobStatus::Pending,
            grounding_certificate: Some(valid_grounding_certificate()),
            energy_source: Some(EnergyType::Solar),
            material_passport: None,
            cincinnati_session: None,
            quality_predictions: vec![],
            estimated_time_minutes: Some(120),
            actual_time_minutes: None,
            material_used_grams: Some(50.0),
            created_at: test_timestamp(),
            started_at: None,
            completed_at: None,
        }
    }

    fn valid_print_record() -> PrintRecord {
        PrintRecord {
            job_hash: test_action_hash(),
            result: PrintResult::Success,
            quality_score: Some(0.95),
            pog_score: 0.85,
            energy_used_kwh: 0.5,
            carbon_offset_kg: Some(0.1),
            material_circularity: 0.6,
            mycelium_earned: 100,
            cincinnati_report: None,
            dimensional_accuracy: None,
            photos: vec!["photo1.jpg".to_string()],
            notes: "Good print".to_string(),
            issues: vec![],
            verification: None,
            recorded_at: test_timestamp(),
        }
    }

    fn valid_sensor_snapshot() -> SensorSnapshot {
        SensorSnapshot {
            hotend_temp: 210.0,
            bed_temp: 60.0,
            stepper_currents: [0.8, 0.8, 1.0, 1.2],
            vibration_rms: 0.05,
            filament_tension: Some(1.5),
            ambient_temp: Some(22.0),
            humidity: Some(45.0),
        }
    }

    fn valid_cincinnati_session_entry() -> CincinnatiSessionEntry {
        CincinnatiSessionEntry {
            session: CincinnatiSession {
                session_id: "session-001".to_string(),
                estimator_version: "1.0.0".to_string(),
                sampling_rate_hz: 100,
                baseline_signature: vec![0.0; 10],
                active: true,
                started_at: test_timestamp(),
            },
            print_job_hash: test_action_hash(),
            current_layer: 10,
            total_layers: 100,
            running_health_score: 0.95,
            anomaly_count: 0,
        }
    }

    fn valid_cincinnati_anomaly_entry() -> CincinnatiAnomalyEntry {
        CincinnatiAnomalyEntry {
            session_id: "session-001".to_string(),
            event: AnomalyEvent {
                timestamp_ms: 1000,
                layer_number: 5,
                anomaly_type: AnomalyType::TemperatureDeviation,
                severity: 0.3,
                sensor_data: valid_sensor_snapshot(),
                action_taken: Some(CincinnatiAction::Continue),
            },
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    // =========================================================================
    // PrintJob tests
    // =========================================================================

    #[test]
    fn test_valid_job_passes() {
        let result = validate_print_job(valid_print_job());
        assert!(is_valid(&result), "Valid print job should pass validation");
    }

    #[test]
    fn test_nan_layer_height_rejected() {
        let mut job = valid_print_job();
        job.settings.layer_height = f32::NAN;
        let result = validate_print_job(job);
        assert!(is_invalid(&result), "NaN layer height should be rejected");
    }

    #[test]
    fn test_negative_material_used_rejected() {
        let mut job = valid_print_job();
        job.material_used_grams = Some(-10.0);
        let result = validate_print_job(job);
        assert!(is_invalid(&result), "Negative material_used_grams should be rejected");
    }

    #[test]
    fn test_empty_job_id_rejected() {
        let mut job = valid_print_job();
        job.id = "".to_string();
        let result = validate_print_job(job);
        assert!(is_invalid(&result), "Empty job ID should be rejected");
    }

    #[test]
    fn test_too_many_quality_predictions_rejected() {
        let mut job = valid_print_job();
        job.quality_predictions = (0..101)
            .map(|_| QualityPrediction {
                predicted_quality: 0.9,
                confidence: 0.8,
                potential_issues: vec![],
                recommendations: vec![],
            })
            .collect();
        let result = validate_print_job(job);
        assert!(is_invalid(&result), "More than 100 quality_predictions should be rejected");
    }

    // =========================================================================
    // PrintRecord tests
    // =========================================================================

    #[test]
    fn test_valid_record_passes() {
        let result = validate_print_record(valid_print_record());
        assert!(is_valid(&result), "Valid print record should pass validation");
    }

    #[test]
    fn test_nan_pog_score_rejected() {
        let mut record = valid_print_record();
        record.pog_score = f32::NAN;
        let result = validate_print_record(record);
        assert!(is_invalid(&result), "NaN pog_score should be rejected");
    }

    #[test]
    fn test_quality_score_out_of_range_rejected() {
        let mut record = valid_print_record();
        record.quality_score = Some(1.5);
        let result = validate_print_record(record);
        assert!(is_invalid(&result), "quality_score > 1.0 should be rejected");
    }

    #[test]
    fn test_too_many_photos_rejected() {
        let mut record = valid_print_record();
        record.photos = (0..51).map(|i| format!("photo{}.jpg", i)).collect();
        let result = validate_print_record(record);
        assert!(is_invalid(&result), "More than 50 photos should be rejected");
    }

    #[test]
    fn test_too_many_issues_rejected() {
        let mut record = valid_print_record();
        record.issues = (0..101)
            .map(|i| PrintIssue::Other(format!("issue-{}", i)))
            .collect();
        let result = validate_print_record(record);
        assert!(is_invalid(&result), "More than 100 issues should be rejected");
    }

    // =========================================================================
    // CincinnatiSessionEntry tests
    // =========================================================================

    #[test]
    fn test_valid_session_passes() {
        let result = validate_cincinnati_session(valid_cincinnati_session_entry());
        assert!(is_valid(&result), "Valid Cincinnati session should pass validation");
    }

    #[test]
    fn test_nan_health_score_rejected() {
        let mut session = valid_cincinnati_session_entry();
        session.running_health_score = f32::NAN;
        let result = validate_cincinnati_session(session);
        assert!(is_invalid(&result), "NaN running_health_score should be rejected");
    }

    #[test]
    fn test_current_exceeds_total_layers_rejected() {
        let mut session = valid_cincinnati_session_entry();
        session.current_layer = 200;
        session.total_layers = 100;
        let result = validate_cincinnati_session(session);
        assert!(is_invalid(&result), "current_layer > total_layers should be rejected");
    }

    // =========================================================================
    // CincinnatiAnomalyEntry tests
    // =========================================================================

    #[test]
    fn test_valid_anomaly_passes() {
        let result = validate_cincinnati_anomaly(valid_cincinnati_anomaly_entry());
        assert!(is_valid(&result), "Valid Cincinnati anomaly should pass validation");
    }

    #[test]
    fn test_nan_severity_rejected() {
        let mut anomaly = valid_cincinnati_anomaly_entry();
        anomaly.event.severity = f32::NAN;
        let result = validate_cincinnati_anomaly(anomaly);
        assert!(is_invalid(&result), "NaN severity should be rejected");
    }

    #[test]
    fn test_nan_sensor_hotend_temp_rejected() {
        let mut anomaly = valid_cincinnati_anomaly_entry();
        anomaly.event.sensor_data.hotend_temp = f32::NAN;
        let result = validate_cincinnati_anomaly(anomaly);
        assert!(is_invalid(&result), "NaN hotend_temp should be rejected");
    }

    #[test]
    fn test_nan_stepper_current_rejected() {
        let mut anomaly = valid_cincinnati_anomaly_entry();
        anomaly.event.sensor_data.stepper_currents[2] = f32::INFINITY;
        let result = validate_cincinnati_anomaly(anomaly);
        assert!(is_invalid(&result), "Infinite stepper_current should be rejected");
    }

    // =========================================================================
    // Edge cases: infinity, boundary values, combined failures
    // =========================================================================

    #[test]
    fn test_infinity_pog_score_rejected() {
        let mut r = valid_print_record();
        r.pog_score = f32::INFINITY;
        let result = validate_print_record(r);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_neg_infinity_energy_rejected() {
        let mut r = valid_print_record();
        r.energy_used_kwh = f32::NEG_INFINITY;
        let result = validate_print_record(r);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_pog_score_at_zero_passes() {
        let mut r = valid_print_record();
        r.pog_score = 0.0;
        let result = validate_print_record(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn test_pog_score_at_one_passes() {
        let mut r = valid_print_record();
        r.pog_score = 1.0;
        let result = validate_print_record(r);
        assert!(is_valid(&result));
    }

    #[test]
    fn test_layer_height_at_min_boundary_passes() {
        let mut j = valid_print_job();
        j.settings.layer_height = 0.01;
        let result = validate_print_job(j);
        assert!(is_valid(&result));
    }

    #[test]
    fn test_layer_height_below_min_rejected() {
        let mut j = valid_print_job();
        j.settings.layer_height = 0.009;
        let result = validate_print_job(j);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_infill_at_zero_passes() {
        let mut j = valid_print_job();
        j.settings.infill_percent = 0;
        let result = validate_print_job(j);
        assert!(is_valid(&result));
    }

    #[test]
    fn test_infill_at_hundred_passes() {
        let mut j = valid_print_job();
        j.settings.infill_percent = 100;
        let result = validate_print_job(j);
        assert!(is_valid(&result));
    }

    #[test]
    fn test_health_score_at_boundaries_passes() {
        let mut s = valid_cincinnati_session_entry();
        s.running_health_score = 0.0;
        assert!(is_valid(&validate_cincinnati_session(s)));

        let mut s2 = valid_cincinnati_session_entry();
        s2.running_health_score = 1.0;
        assert!(is_valid(&validate_cincinnati_session(s2)));
    }

    #[test]
    fn test_severity_at_boundaries_passes() {
        let mut a = valid_cincinnati_anomaly_entry();
        a.event.severity = 0.0;
        assert!(is_valid(&validate_cincinnati_anomaly(a)));

        let mut a2 = valid_cincinnati_anomaly_entry();
        a2.event.severity = 1.0;
        assert!(is_valid(&validate_cincinnati_anomaly(a2)));
    }

    // =========================================================================
    // Session ID validation tests
    // =========================================================================

    #[test]
    fn test_anomaly_session_id_over_max_rejected() {
        let mut a = valid_cincinnati_anomaly_entry();
        a.session_id = "x".repeat(257);
        let result = validate_cincinnati_anomaly(a);
        assert!(matches!(result.unwrap(), ValidateCallbackResult::Invalid(msg) if msg.contains("session_id")));
    }

    #[test]
    fn test_anomaly_session_id_at_max_passes() {
        let mut a = valid_cincinnati_anomaly_entry();
        a.session_id = "x".repeat(256);
        assert!(is_valid(&validate_cincinnati_anomaly(a)));
    }

    // =========================================================================
    // Link tag validation tests
    // =========================================================================

    #[test]
    fn test_link_tag_at_max_passes() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(result.is_err()); // Err(()) means "no validation issue found"
    }

    #[test]
    fn test_link_tag_over_max_rejected() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(msg)) if msg.contains("link tag")));
    }
}
