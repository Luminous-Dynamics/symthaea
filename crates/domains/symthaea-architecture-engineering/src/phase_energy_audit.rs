//! Marker-synchronized, phase-resolved and whole-application-event energy audit.
//!
//! This module joins a raw PPK2 sample CSV to a separate trigger-marker CSV using a
//! content-addressed clock map. It integrates each declared phase and the complete event
//! envelope from the same timestamped current samples. It never infers phase boundaries
//! from current-amplitude changes.
//!
//! The result is measurement evidence, not a claim that a marker corresponds to correct
//! application semantics. Firmware/protocol provenance and task-outcome evidence are separate.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

use csv::{ReaderBuilder, StringRecord, Trim};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ppk2_trace_audit::{
    audit_ppk2_trace_csv, Ppk2TraceAuditConfig, Ppk2TraceAuditError,
};

const CLOCK_SCHEMA: &str = "symthaea.nas.phase-clock-mapping";
const PROTOCOL_SCHEMA: &str = "symthaea.nas.application-phase-protocol";
const RECEIPT_SCHEMA: &str = "symthaea.nas.application-phase-energy-receipt";
const VERSION: &str = "0.1.0";
const CLOCK_DOMAIN: &[u8] = b"symthaea.nas.phase-clock-mapping.v0.1";
const PROTOCOL_DOMAIN: &[u8] = b"symthaea.nas.application-phase-protocol.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.application-phase-energy-receipt.v0.1";
const NOMINAL_MS_PER_NS: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseEnergyAuditError {
    pub field: String,
    pub message: String,
}

impl PhaseEnergyAuditError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl fmt::Display for PhaseEnergyAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for PhaseEnergyAuditError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseClockMappingManifest {
    pub schema: String,
    pub version: String,
    pub mapping_id: String,
    pub marker_clock_domain_id: String,
    pub marker_source_id: String,
    pub ppk2_clock_domain_id: String,
    pub marker_origin_ns: u64,
    pub ppk2_origin_ms: f64,
    /// Calibrated milliseconds-per-marker-nanosecond scale.
    pub ppk2_ms_per_marker_ns: f64,
    pub scale_tolerance_fraction: f64,
    pub maximum_alignment_error_ms: f64,
    pub clock_calibration_evidence_sha256: String,
    pub mapping_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationPhaseDefinition {
    pub phase_id: String,
    pub order: u32,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationPhaseProtocolManifest {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub protocol_id: String,
    /// Strictly ordered phase graph. At least one phase must be required.
    pub phases: Vec<ApplicationPhaseDefinition>,
    /// A complete event is only labelled complete if unassigned gaps are within this limit.
    pub maximum_unattributed_gap_ms: f64,
    pub protocol_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseEnergyEventStatus {
    CompletePhaseCoverage,
    UnattributedIntervalsPresent,
    MissingRequiredPhase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseEnergyWindowMetrics {
    pub phase_id: String,
    pub order: u32,
    pub start_ppk2_time_ms: f64,
    pub end_ppk2_time_ms: f64,
    pub duration_ms: f64,
    pub energy_uj: f64,
    pub time_weighted_mean_current_ua: f64,
    pub mean_power_uw: f64,
    pub maximum_observed_current_ua: f64,
    pub boundary_alignment_error_bound_ms: f64,
    /// Worst-case energy impact of both uncertain phase boundaries under the maximum observed current.
    pub boundary_alignment_energy_uncertainty_uj: f64,
    pub boundary_alignment_energy_uncertainty_uj: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseEnergyEventMetrics {
    pub event_id: String,
    pub segment_id: String,
    pub status: PhaseEnergyEventStatus,
    pub missing_phase_ids: Vec<String>,
    pub missing_required_phase_ids: Vec<String>,
    pub phase_windows: Vec<PhaseEnergyWindowMetrics>,
    pub event_start_ppk2_time_ms: Option<f64>,
    pub event_end_ppk2_time_ms: Option<f64>,
    pub total_event_duration_ms: Option<f64>,
    /// Full envelope integral, including any gaps between declared phases.
    pub total_event_energy_uj: Option<f64>,
    pub maximum_event_current_ua: Option<f64>,
    pub total_event_boundary_alignment_uncertainty_uj: Option<f64>,
    pub declared_phase_energy_sum_uj: f64,
    pub unattributed_duration_ms: Option<f64>,
    pub unattributed_energy_uj: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseEnergyAuditReceipt {
    pub schema: String,
    pub version: String,
    pub sample_csv_sha256: String,
    pub marker_csv_sha256: String,
    pub sample_csv_bytes: usize,
    pub marker_csv_bytes: usize,
    pub trace_audit_receipt_sha256: String,
    pub clock_mapping_sha256: String,
    pub phase_protocol_sha256: String,
    pub voltage_v: f64,
    pub expected_sample_rate_hz: f64,
    pub sample_rate_tolerance_fraction: f64,
    pub maximum_sample_gap_ratio: f64,
    pub maximum_unattributed_gap_ms: f64,
    pub event_count: usize,
    pub complete_phase_coverage_events: usize,
    pub unattributed_interval_events: usize,
    pub missing_required_phase_events: usize,
    pub events: Vec<PhaseEnergyEventMetrics>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseEnergyAuditResult {
    pub trace_receipt_sha256: String,
    pub receipt: PhaseEnergyAuditReceipt,
}

#[derive(Debug, Clone)]
struct Sample {
    time_ms: f64,
    current_ua: f64,
}

#[derive(Debug, Clone, Default)]
struct PhaseEdges {
    start_ms: Option<f64>,
    end_ms: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct ParsedEventMarkers {
    segment_id: Option<String>,
    sequence_times_ms: BTreeMap<u64, f64>,
    phases: BTreeMap<String, PhaseEdges>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn compute_phase_clock_mapping_sha256(
    mapping: &PhaseClockMappingManifest,
) -> Result<String, PhaseEnergyAuditError> {
    let mut canonical = mapping.clone();
    canonical.mapping_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|e| {
        PhaseEnergyAuditError::new("clock_mapping", format!("cannot serialize: {e}"))
    })?;
    Ok(digest(CLOCK_DOMAIN, &bytes))
}

pub fn validate_phase_clock_mapping(
    mapping: &PhaseClockMappingManifest,
) -> Result<(), PhaseEnergyAuditError> {
    if mapping.schema != CLOCK_SCHEMA || mapping.version != VERSION {
        return Err(PhaseEnergyAuditError::new("clock_mapping.schema_version", "unsupported schema/version"));
    }
    for (field, value) in [
        ("mapping_id", mapping.mapping_id.as_str()),
        ("marker_clock_domain_id", mapping.marker_clock_domain_id.as_str()),
        ("marker_source_id", mapping.marker_source_id.as_str()),
        ("ppk2_clock_domain_id", mapping.ppk2_clock_domain_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(PhaseEnergyAuditError::new(format!("clock_mapping.{field}"), "must be non-empty"));
        }
    }
    if !is_sha256(&mapping.clock_calibration_evidence_sha256)
        || !is_sha256(&mapping.mapping_sha256)
    {
        return Err(PhaseEnergyAuditError::new("clock_mapping.digest", "calibration and mapping digests must be lowercase SHA-256 hex"));
    }
    if !mapping.ppk2_origin_ms.is_finite()
        || !mapping.ppk2_ms_per_marker_ns.is_finite()
        || mapping.ppk2_ms_per_marker_ns <= 0.0
        || !mapping.scale_tolerance_fraction.is_finite()
        || !(0.0..=0.1).contains(&mapping.scale_tolerance_fraction)
        || !mapping.maximum_alignment_error_ms.is_finite()
        || mapping.maximum_alignment_error_ms < 0.0
    {
        return Err(PhaseEnergyAuditError::new("clock_mapping.parameters", "clock mapping parameters are outside valid bounds"));
    }
    let scale_error = (mapping.ppk2_ms_per_marker_ns - NOMINAL_MS_PER_NS).abs() / NOMINAL_MS_PER_NS;
    if scale_error > mapping.scale_tolerance_fraction {
        return Err(PhaseEnergyAuditError::new(
            "clock_mapping.ppk2_ms_per_marker_ns",
            format!("clock scale differs from ns→ms by {:.3}% beyond configured {:.3}% tolerance",
                scale_error * 100.0, mapping.scale_tolerance_fraction * 100.0),
        ));
    }
    if compute_phase_clock_mapping_sha256(mapping)? != mapping.mapping_sha256 {
        return Err(PhaseEnergyAuditError::new("clock_mapping.mapping_sha256", "does not match canonical mapping"));
    }
    Ok(())
}

pub fn compute_application_phase_protocol_sha256(
    protocol: &ApplicationPhaseProtocolManifest,
) -> Result<String, PhaseEnergyAuditError> {
    let mut canonical = protocol.clone();
    canonical.protocol_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|e| {
        PhaseEnergyAuditError::new("phase_protocol", format!("cannot serialize: {e}"))
    })?;
    Ok(digest(PROTOCOL_DOMAIN, &bytes))
}

pub fn validate_application_phase_protocol(
    protocol: &ApplicationPhaseProtocolManifest,
) -> Result<(), PhaseEnergyAuditError> {
    if protocol.schema != PROTOCOL_SCHEMA || protocol.version != VERSION {
        return Err(PhaseEnergyAuditError::new("phase_protocol.schema_version", "unsupported schema/version"));
    }
    if protocol.benchmark_id.trim().is_empty() || protocol.protocol_id.trim().is_empty() {
        return Err(PhaseEnergyAuditError::new("phase_protocol.identity", "benchmark and protocol IDs must be non-empty"));
    }
    if protocol.phases.is_empty()
        || !protocol.phases.iter().any(|phase| phase.required)
        || !protocol.phases.first().is_some_and(|phase| phase.required)
        || !protocol.phases.last().is_some_and(|phase| phase.required)
        || !protocol.maximum_unattributed_gap_ms.is_finite()
        || protocol.maximum_unattributed_gap_ms < 0.0
    {
        return Err(PhaseEnergyAuditError::new("phase_protocol.phases", "at least one required phase, required first/last boundary phases, and a finite non-negative gap tolerance are required"));
    }
    let mut ids = HashSet::with_capacity(protocol.phases.len());
    for (index, phase) in protocol.phases.iter().enumerate() {
        if phase.phase_id.trim().is_empty() || !ids.insert(phase.phase_id.as_str()) {
            return Err(PhaseEnergyAuditError::new(format!("phase_protocol.phases[{index}].phase_id"), "phase IDs must be non-empty and unique"));
        }
        if phase.order as usize != index {
            return Err(PhaseEnergyAuditError::new(format!("phase_protocol.phases[{index}].order"), "phase order must be dense and sorted from zero"));
        }
    }
    if !is_sha256(&protocol.protocol_sha256)
        || compute_application_phase_protocol_sha256(protocol)? != protocol.protocol_sha256
    {
        return Err(PhaseEnergyAuditError::new("phase_protocol.protocol_sha256", "does not match canonical protocol"));
    }
    Ok(())
}

fn parse_header(
    headers: &StringRecord,
    required: &[&str],
    table: &str,
) -> Result<BTreeMap<String, usize>, PhaseEnergyAuditError> {
    let mut seen = HashSet::with_capacity(headers.len());
    for header in headers.iter() {
        if header.trim().is_empty() || !seen.insert(header.to_owned()) {
            return Err(PhaseEnergyAuditError::new(format!("{table}.header"), "empty or duplicate header"));
        }
    }
    let mut columns = BTreeMap::new();
    for name in required {
        let index = headers.iter().position(|value| value == *name).ok_or_else(|| {
            PhaseEnergyAuditError::new(format!("{table}.header"), format!("required column {name:?} is missing"))
        })?;
        columns.insert((*name).to_owned(), index);
    }
    Ok(columns)
}

fn field<'a>(
    row: &'a StringRecord,
    columns: &BTreeMap<String, usize>,
    name: &str,
    row_number: usize,
    table: &str,
) -> Result<&'a str, PhaseEnergyAuditError> {
    let column = columns.get(name).ok_or_else(|| {
        PhaseEnergyAuditError::new(format!("{table}.header"), format!("internal missing column {name}"))
    })?;
    row.get(*column).ok_or_else(|| {
        PhaseEnergyAuditError::new(format!("{table}.row[{row_number}].{name}"), "value is missing")
    })
}

fn parse_finite(raw: &str, field_name: &str, row_number: usize, table: &str) -> Result<f64, PhaseEnergyAuditError> {
    let value = raw.parse::<f64>().map_err(|_| {
        PhaseEnergyAuditError::new(format!("{table}.row[{row_number}].{field_name}"), format!("expected finite number, got {raw:?}"))
    })?;
    if !value.is_finite() {
        return Err(PhaseEnergyAuditError::new(format!("{table}.row[{row_number}].{field_name}"), "NaN and infinity are forbidden"));
    }
    Ok(value)
}

fn parse_samples(
    samples_csv: &str,
    trace_config: Ppk2TraceAuditConfig,
) -> Result<(BTreeMap<String, Vec<Sample>>, String), PhaseEnergyAuditError> {
    let trace_receipt = audit_ppk2_trace_csv(samples_csv, trace_config)
        .map_err(|error: Ppk2TraceAuditError| PhaseEnergyAuditError::new("sample_trace", error.to_string()))?;

    let mut reader = ReaderBuilder::new().has_headers(true).flexible(false).trim(Trim::All)
        .from_reader(samples_csv.as_bytes());
    let headers = reader.headers().map_err(|e| PhaseEnergyAuditError::new("samples.header", e.to_string()))?.clone();
    let columns = parse_header(&headers, &["segment_id", "time_ms", "current_uA"], "samples")?;
    let mut segments = BTreeMap::<String, Vec<Sample>>::new();

    for (index, record) in reader.records().enumerate() {
        let row_number = index + 2;
        let row = record.map_err(|e| PhaseEnergyAuditError::new(format!("samples.row[{row_number}]"), e.to_string()))?;
        let segment_id = field(&row, &columns, "segment_id", row_number, "samples")?.to_owned();
        if segment_id.trim().is_empty() {
            return Err(PhaseEnergyAuditError::new(format!("samples.row[{row_number}].segment_id"), "must be non-empty"));
        }
        let time_ms = parse_finite(field(&row, &columns, "time_ms", row_number, "samples")?, "time_ms", row_number, "samples")?;
        let current_ua = parse_finite(field(&row, &columns, "current_uA", row_number, "samples")?, "current_uA", row_number, "samples")?;
        if current_ua < 0.0 {
            return Err(PhaseEnergyAuditError::new(format!("samples.row[{row_number}].current_uA"), "must be non-negative"));
        }
        segments.entry(segment_id).or_default().push(Sample { time_ms, current_ua });
    }
    Ok((segments, trace_receipt.receipt.receipt_sha256))
}

fn marker_time_to_ppk2_ms(
    marker_timestamp_ns: u64,
    mapping: &PhaseClockMappingManifest,
) -> f64 {
    let delta_ns = marker_timestamp_ns as i128 - mapping.marker_origin_ns as i128;
    mapping.ppk2_origin_ms + delta_ns as f64 * mapping.ppk2_ms_per_marker_ns
}

fn parse_markers(
    markers_csv: &str,
    mapping: &PhaseClockMappingManifest,
    protocol: &ApplicationPhaseProtocolManifest,
) -> Result<BTreeMap<String, ParsedEventMarkers>, PhaseEnergyAuditError> {
    let mut reader = ReaderBuilder::new().has_headers(true).flexible(false).trim(Trim::All)
        .from_reader(markers_csv.as_bytes());
    let headers = reader.headers().map_err(|e| PhaseEnergyAuditError::new("markers.header", e.to_string()))?.clone();
    let columns = parse_header(
        &headers,
        &["event_id", "segment_id", "phase_id", "edge", "marker_timestamp_ns", "sequence", "clock_domain_id", "marker_source_id"],
        "markers",
    )?;

    let phases: HashSet<&str> = protocol.phases.iter().map(|phase| phase.phase_id.as_str()).collect();
    let mut events = BTreeMap::<String, ParsedEventMarkers>::new();
    for (index, record) in reader.records().enumerate() {
        let row_number = index + 2;
        let row = record.map_err(|e| PhaseEnergyAuditError::new(format!("markers.row[{row_number}]"), e.to_string()))?;
        let event_id = field(&row, &columns, "event_id", row_number, "markers")?.to_owned();
        let segment_id = field(&row, &columns, "segment_id", row_number, "markers")?.to_owned();
        let phase_id = field(&row, &columns, "phase_id", row_number, "markers")?.to_owned();
        let edge = field(&row, &columns, "edge", row_number, "markers")?;
        let clock_domain = field(&row, &columns, "clock_domain_id", row_number, "markers")?;
        let marker_source = field(&row, &columns, "marker_source_id", row_number, "markers")?;

        if event_id.trim().is_empty() || segment_id.trim().is_empty() || phase_id.trim().is_empty() {
            return Err(PhaseEnergyAuditError::new(format!("markers.row[{row_number}].identity"), "event, segment and phase IDs must be non-empty"));
        }
        if !phases.contains(phase_id.as_str()) {
            return Err(PhaseEnergyAuditError::new(format!("markers.row[{row_number}].phase_id"), format!("phase {phase_id:?} is not declared in the protocol")));
        }
        if clock_domain != mapping.marker_clock_domain_id || marker_source != mapping.marker_source_id {
            return Err(PhaseEnergyAuditError::new(format!("markers.row[{row_number}].clock_identity"), "marker clock domain or source does not match the pinned mapping"));
        }

        let marker_timestamp_ns = field(&row, &columns, "marker_timestamp_ns", row_number, "markers")?
            .parse::<u64>()
            .map_err(|_| PhaseEnergyAuditError::new(format!("markers.row[{row_number}].marker_timestamp_ns"), "must be an unsigned integer"))?;
        let sequence = field(&row, &columns, "sequence", row_number, "markers")?
            .parse::<u64>()
            .map_err(|_| PhaseEnergyAuditError::new(format!("markers.row[{row_number}].sequence"), "must be an unsigned integer"))?;
        let ppk2_time_ms = marker_time_to_ppk2_ms(marker_timestamp_ns, mapping);
        if !ppk2_time_ms.is_finite() {
            return Err(PhaseEnergyAuditError::new(format!("markers.row[{row_number}].timestamp"), "mapped timestamp is non-finite"));
        }

        let event = events.entry(event_id.clone()).or_default();
        if let Some(previous_segment) = &event.segment_id {
            if previous_segment != &segment_id {
                return Err(PhaseEnergyAuditError::new(format!("markers.{event_id}.segment_id"), "one event ID cannot span multiple PPK2 trace segments"));
            }
        } else {
            event.segment_id = Some(segment_id);
        }
        if event.sequence_times_ms.insert(sequence, ppk2_time_ms).is_some() {
            return Err(PhaseEnergyAuditError::new(format!("markers.{event_id}.sequence"), "sequence numbers must be unique within each event"));
        }

        let phase_edges = event.phases.entry(phase_id.clone()).or_default();
        match edge {
            "START" => {
                if phase_edges.start_ms.replace(ppk2_time_ms).is_some() {
                    return Err(PhaseEnergyAuditError::new(format!("markers.{event_id}.{phase_id}.START"), "duplicate phase start marker"));
                }
            }
            "END" => {
                if phase_edges.end_ms.replace(ppk2_time_ms).is_some() {
                    return Err(PhaseEnergyAuditError::new(format!("markers.{event_id}.{phase_id}.END"), "duplicate phase end marker"));
                }
            }
            other => return Err(PhaseEnergyAuditError::new(format!("markers.row[{row_number}].edge"), format!("expected START or END, got {other:?}"))),
        }
    }

    if events.is_empty() {
        return Err(PhaseEnergyAuditError::new("markers.rows", "no marker events were supplied"));
    }

    for (event_id, event) in &events {
        let sequence_times: Vec<f64> = event.sequence_times_ms.values().copied().collect();
        if sequence_times.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(PhaseEnergyAuditError::new(format!("markers.{event_id}.sequence"), "mapped marker timestamps must be non-decreasing in sequence order"));
        }
    }
    Ok(events)
}

fn interpolate_current(
    samples: &[Sample],
    target_time_ms: f64,
    segment_id: &str,
) -> Result<f64, PhaseEnergyAuditError> {
    if samples.len() < 2 {
        return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.samples"), "at least two samples are required"));
    }
    if target_time_ms < samples[0].time_ms || target_time_ms > samples[samples.len() - 1].time_ms {
        return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.boundary"), "phase boundary falls outside the raw sample trace"));
    }
    for pair in samples.windows(2) {
        if target_time_ms >= pair[0].time_ms && target_time_ms <= pair[1].time_ms {
            let dt = pair[1].time_ms - pair[0].time_ms;
            if dt <= 0.0 {
                return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.timestamps"), "sample timestamps must strictly increase"));
            }
            let ratio = (target_time_ms - pair[0].time_ms) / dt;
            return Ok(pair[0].current_ua + ratio * (pair[1].current_ua - pair[0].current_ua));
        }
    }
    Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.boundary"), "could not bracket a phase boundary"))
}

fn integrate_window(
    samples: &[Sample],
    start_ms: f64,
    end_ms: f64,
    voltage_v: f64,
    segment_id: &str,
) -> Result<(f64, f64, f64, f64), PhaseEnergyAuditError> {
    if !start_ms.is_finite() || !end_ms.is_finite() || end_ms <= start_ms {
        return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.window"), "window must have finite positive duration"));
    }
    let start_current = interpolate_current(samples, start_ms, segment_id)?;
    let end_current = interpolate_current(samples, end_ms, segment_id)?;
    let mut points = Vec::<(f64, f64)>::with_capacity(samples.len() + 2);
    points.push((start_ms, start_current));
    points.extend(samples.iter()
        .filter(|sample| sample.time_ms > start_ms && sample.time_ms < end_ms)
        .map(|sample| (sample.time_ms, sample.current_ua)));
    points.push((end_ms, end_current));

    let maximum_current_ua = points.iter()
        .map(|point| point.1)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let mut integral_ua_ms = 0.0;
    for pair in points.windows(2) {
        let dt = pair[1].0 - pair[0].0;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.window"), "integrated points must strictly increase"));
        }
        integral_ua_ms += (pair[0].1 + pair[1].1) * 0.5 * dt;
    }
    let duration_ms = end_ms - start_ms;
    let mean_current_ua = integral_ua_ms / duration_ms;
    let energy_uj = integral_ua_ms * voltage_v / 1_000.0;
    if !mean_current_ua.is_finite() || !energy_uj.is_finite() || energy_uj < 0.0 {
        return Err(PhaseEnergyAuditError::new(format!("segments.{segment_id}.window"), "integrated current or energy is invalid"));
    }
    Ok((energy_uj, mean_current_ua, mean_current_ua * voltage_v, maximum_current_ua))
}

pub fn compute_phase_energy_audit_receipt_sha256(
    receipt: &PhaseEnergyAuditReceipt,
) -> Result<String, PhaseEnergyAuditError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|e| {
        PhaseEnergyAuditError::new("receipt", format!("cannot serialize: {e}"))
    })?;
    Ok(digest(RECEIPT_DOMAIN, &bytes))
}

/// Audit phase markers against a raw PPK2 sample CSV. Marker timestamps are mapped into
/// the PPK2 time base via the pinned affine clock mapping. All event/phase intervals are
/// integrated from the same raw current samples.
pub fn audit_application_phase_energy(
    sample_csv: &str,
    marker_csv: &str,
    trace_config: Ppk2TraceAuditConfig,
    mapping: &PhaseClockMappingManifest,
    protocol: &ApplicationPhaseProtocolManifest,
) -> Result<PhaseEnergyAuditResult, PhaseEnergyAuditError> {
    validate_phase_clock_mapping(mapping)?;
    validate_application_phase_protocol(protocol)?;

    let (segments, trace_receipt_sha256) = parse_samples(sample_csv, trace_config)?;
    let events = parse_markers(marker_csv, mapping, protocol)?;
    let mut event_segments = HashSet::with_capacity(events.len());
    for (event_id, event) in &events {
        let segment_id = event.segment_id.as_deref().ok_or_else(|| {
            PhaseEnergyAuditError::new(format!("markers.{event_id}.segment_id"), "event has no segment identity")
        })?;
        if !segments.contains_key(segment_id) {
            return Err(PhaseEnergyAuditError::new(
                format!("markers.{event_id}.segment_id"),
                "referenced PPK2 sample segment is absent",
            ));
        }
        if !event_segments.insert(segment_id) {
            return Err(PhaseEnergyAuditError::new(
                format!("markers.{event_id}.segment_id"),
                "a PPK2 segment may be assigned to only one application event",
            ));
        }
    }
    let mut event_metrics = Vec::with_capacity(events.len());

    for (event_id, parsed) in &events {
        let segment_id = parsed.segment_id.as_deref().ok_or_else(|| {
            PhaseEnergyAuditError::new(format!("markers.{event_id}.segment_id"), "event has no segment identity")
        })?;
        let samples = segments.get(segment_id).ok_or_else(|| {
            PhaseEnergyAuditError::new(format!("markers.{event_id}.segment_id"), "referenced PPK2 sample segment is absent")
        })?;

        let mut missing_phase_ids = Vec::new();
        let mut missing_required_phase_ids = Vec::new();
        let mut windows = Vec::<(&ApplicationPhaseDefinition, f64, f64)>::new();
        for phase in &protocol.phases {
            match parsed.phases.get(&phase.phase_id) {
                None => {
                    missing_phase_ids.push(phase.phase_id.clone());
                    if phase.required {
                        missing_required_phase_ids.push(phase.phase_id.clone());
                    }
                }
                Some(edges) => match (edges.start_ms, edges.end_ms) {
                    (Some(start), Some(end)) if end > start => windows.push((phase, start, end)),
                    _ => {
                        missing_phase_ids.push(phase.phase_id.clone());
                        if phase.required {
                            missing_required_phase_ids.push(phase.phase_id.clone());
                        }
                    }
                },
            }
        }

        let mut previous_end: Option<f64> = None;
        for (phase, start, end) in &windows {
            if let Some(prior_end) = previous_end {
                if *start < prior_end {
                    return Err(PhaseEnergyAuditError::new(
                        format!("events.{event_id}.phase_order"),
                        format!("phase {:?} overlaps the previous phase", phase.phase_id),
                    ));
                }
            }
            previous_end = Some(*end);
        }

        let mut phase_windows = Vec::with_capacity(windows.len());
        let mut declared_phase_energy_sum_uj = 0.0;
        let mut declared_duration_sum_ms = 0.0;
        for (phase, start, end) in &windows {
            let (energy, mean_current, mean_power, maximum_current) = integrate_window(
                samples,
                *start,
                *end,
                trace_config.voltage_v,
                segment_id,
            )?;
            let duration = *end - *start;
            declared_phase_energy_sum_uj += energy;
            declared_duration_sum_ms += duration;
            phase_windows.push(PhaseEnergyWindowMetrics {
                phase_id: phase.phase_id.clone(),
                order: phase.order,
                start_ppk2_time_ms: *start,
                end_ppk2_time_ms: *end,
                duration_ms: duration,
                energy_uj: energy,
                time_weighted_mean_current_ua: mean_current,
                mean_power_uw: mean_power,
                maximum_observed_current_ua: maximum_current,
                boundary_alignment_error_bound_ms: mapping.maximum_alignment_error_ms,
                boundary_alignment_energy_uncertainty_uj:
                    2.0 * maximum_current * trace_config.voltage_v
                        * mapping.maximum_alignment_error_ms / 1_000.0,
            });
        }

        if !missing_required_phase_ids.is_empty() || windows.is_empty() {
            event_metrics.push(PhaseEnergyEventMetrics {
                event_id: event_id.clone(),
                segment_id: segment_id.to_owned(),
                status: PhaseEnergyEventStatus::MissingRequiredPhase,
                missing_phase_ids,
                missing_required_phase_ids,
                phase_windows,
                event_start_ppk2_time_ms: None,
                event_end_ppk2_time_ms: None,
                total_event_duration_ms: None,
                total_event_energy_uj: None,
                maximum_event_current_ua: None,
                total_event_boundary_alignment_uncertainty_uj: None,
                declared_phase_energy_sum_uj,
                unattributed_duration_ms: None,
                unattributed_energy_uj: None,
            });
            continue;
        }

        let event_start = windows.first().expect("non-empty phase windows").1;
        let event_end = windows.last().expect("non-empty phase windows").2;
        let (total_energy, _, _, maximum_event_current) = integrate_window(
            samples,
            event_start,
            event_end,
            trace_config.voltage_v,
            segment_id,
        )?;
        let total_duration = event_end - event_start;
        let unattributed_duration = total_duration - declared_duration_sum_ms;
        let unattributed_energy = total_energy - declared_phase_energy_sum_uj;
        let tolerance = 1e-9 * total_energy.abs().max(1.0);
        if unattributed_duration < -1e-9 || unattributed_energy < -tolerance {
            return Err(PhaseEnergyAuditError::new(
                format!("events.{event_id}.phase_partition"),
                "phase sums exceed total event envelope; phase intervals may overlap or integration is inconsistent",
            ));
        }
        let unattributed_duration = unattributed_duration.max(0.0);
        let unattributed_energy = unattributed_energy.max(0.0);
        let status = if !missing_phase_ids.is_empty()
            || unattributed_duration > protocol.maximum_unattributed_gap_ms
        {
            PhaseEnergyEventStatus::UnattributedIntervalsPresent
        } else {
            PhaseEnergyEventStatus::CompletePhaseCoverage
        };

        event_metrics.push(PhaseEnergyEventMetrics {
            event_id: event_id.clone(),
            segment_id: segment_id.to_owned(),
            status,
            missing_phase_ids,
            missing_required_phase_ids,
            phase_windows,
            event_start_ppk2_time_ms: Some(event_start),
            event_end_ppk2_time_ms: Some(event_end),
            total_event_duration_ms: Some(total_duration),
            total_event_energy_uj: Some(total_energy),
            maximum_event_current_ua: Some(maximum_event_current),
            total_event_boundary_alignment_uncertainty_uj: Some(
                2.0 * maximum_event_current * trace_config.voltage_v
                    * mapping.maximum_alignment_error_ms / 1_000.0,
            ),
            declared_phase_energy_sum_uj,
            unattributed_duration_ms: Some(unattributed_duration),
            unattributed_energy_uj: Some(unattributed_energy),
        });
    }

    let complete_count = event_metrics.iter().filter(|event| event.status == PhaseEnergyEventStatus::CompletePhaseCoverage).count();
    let unattributed_count = event_metrics.iter().filter(|event| event.status == PhaseEnergyEventStatus::UnattributedIntervalsPresent).count();
    let missing_count = event_metrics.iter().filter(|event| event.status == PhaseEnergyEventStatus::MissingRequiredPhase).count();

    let mut receipt = PhaseEnergyAuditReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        sample_csv_sha256: format!("{:x}", Sha256::digest(sample_csv.as_bytes())),
        marker_csv_sha256: format!("{:x}", Sha256::digest(marker_csv.as_bytes())),
        sample_csv_bytes: sample_csv.len(),
        marker_csv_bytes: marker_csv.len(),
        trace_audit_receipt_sha256,
        clock_mapping_sha256: mapping.mapping_sha256.clone(),
        phase_protocol_sha256: protocol.protocol_sha256.clone(),
        voltage_v: trace_config.voltage_v,
        expected_sample_rate_hz: trace_config.expected_sample_rate_hz,
        sample_rate_tolerance_fraction: trace_config.sample_rate_tolerance_fraction,
        maximum_sample_gap_ratio: trace_config.maximum_gap_ratio,
        maximum_unattributed_gap_ms: protocol.maximum_unattributed_gap_ms,
        event_count: event_metrics.len(),
        complete_phase_coverage_events: complete_count,
        unattributed_interval_events: unattributed_count,
        missing_required_phase_events: missing_count,
        events: event_metrics,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_phase_energy_audit_receipt_sha256(&receipt)?;
    Ok(PhaseEnergyAuditResult {
        trace_receipt_sha256,
        receipt,
    })
}

/// Revalidate the portable phase receipt before downstream scoring.
/// This checks canonical digest, event counters, phase ordering, phase/event energy sums,
/// and recorded boundary-alignment uncertainty. Source replay is stronger and is provided
/// by verify_phase_energy_receipt_against_sources.
pub fn validate_phase_energy_audit_receipt(
    receipt: &PhaseEnergyAuditReceipt,
) -> Result<(), PhaseEnergyAuditError> {
    if receipt.schema != RECEIPT_SCHEMA || receipt.version != VERSION {
        return Err(PhaseEnergyAuditError::new("receipt.schema_version", "unsupported schema/version"));
    }
    for (field, value) in [
        ("sample_csv_sha256", receipt.sample_csv_sha256.as_str()),
        ("marker_csv_sha256", receipt.marker_csv_sha256.as_str()),
        ("trace_audit_receipt_sha256", receipt.trace_audit_receipt_sha256.as_str()),
        ("clock_mapping_sha256", receipt.clock_mapping_sha256.as_str()),
        ("phase_protocol_sha256", receipt.phase_protocol_sha256.as_str()),
        ("receipt_sha256", receipt.receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(PhaseEnergyAuditError::new(
                format!("receipt.{field}"),
                "must be lowercase SHA-256 hex",
            ));
        }
    }
    if receipt.sample_csv_bytes == 0
        || receipt.marker_csv_bytes == 0
        || receipt.event_count == 0
        || receipt.event_count != receipt.events.len()
        || !receipt.voltage_v.is_finite()
        || receipt.voltage_v <= 0.0
        || !receipt.expected_sample_rate_hz.is_finite()
        || receipt.expected_sample_rate_hz <= 0.0
        || !receipt.sample_rate_tolerance_fraction.is_finite()
        || !(0.0..=1.0).contains(&receipt.sample_rate_tolerance_fraction)
        || !receipt.maximum_sample_gap_ratio.is_finite()
        || receipt.maximum_sample_gap_ratio < 1.0
        || !receipt.maximum_unattributed_gap_ms.is_finite()
        || receipt.maximum_unattributed_gap_ms < 0.0
    {
        return Err(PhaseEnergyAuditError::new("receipt.inventory_or_config", "invalid byte/event inventory or trace configuration"));
    }

    let mut event_ids = HashSet::with_capacity(receipt.events.len());
    let mut complete_count = 0usize;
    let mut unattributed_count = 0usize;
    let mut missing_count = 0usize;
    for event in &receipt.events {
        if event.event_id.trim().is_empty()
            || event.segment_id.trim().is_empty()
            || !event_ids.insert(event.event_id.as_str())
        {
            return Err(PhaseEnergyAuditError::new("receipt.events.identity", "event and segment IDs must be non-empty; event IDs must be unique"));
        }
        let mut phase_ids = HashSet::with_capacity(event.phase_windows.len());
        let mut declared_energy = 0.0;
        let mut declared_duration = 0.0;
        let mut previous_order = None;
        let mut previous_end = None;
        for window in &event.phase_windows {
            if window.phase_id.trim().is_empty() || !phase_ids.insert(window.phase_id.as_str()) {
                return Err(PhaseEnergyAuditError::new(
                    format!("receipt.events.{}.phase_windows.phase_id", event.event_id),
                    "phase IDs must be non-empty and unique",
                ));
            }
            for (field, value) in [
                ("start_ppk2_time_ms", window.start_ppk2_time_ms),
                ("end_ppk2_time_ms", window.end_ppk2_time_ms),
                ("duration_ms", window.duration_ms),
                ("energy_uj", window.energy_uj),
                ("time_weighted_mean_current_ua", window.time_weighted_mean_current_ua),
                ("mean_power_uw", window.mean_power_uw),
                ("maximum_observed_current_ua", window.maximum_observed_current_ua),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(PhaseEnergyAuditError::new(
                        format!("receipt.events.{}.phase_windows.{}.{}", event.event_id, window.phase_id, field),
                        "must be finite and non-negative",
                    ));
                }
            }
            for (field, value) in [
                ("duration_ms", window.duration_ms),
                ("time_weighted_mean_current_ua", window.time_weighted_mean_current_ua),
                ("mean_power_uw", window.mean_power_uw),
                ("boundary_alignment_energy_uncertainty_uj", window.boundary_alignment_energy_uncertainty_uj),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(PhaseEnergyAuditError::new(
                        format!("receipt.events.{}.phase_windows.{}.{}", event.event_id, window.phase_id, field),
                        "must be finite and positive",
                    ));
                }
            }
            if !window.boundary_alignment_error_bound_ms.is_finite()
                || window.boundary_alignment_error_bound_ms < 0.0
                || window.end_ppk2_time_ms <= window.start_ppk2_time_ms
                || (window.end_ppk2_time_ms - window.start_ppk2_time_ms - window.duration_ms).abs() > 1e-9
                || previous_order.is_some_and(|order| window.order <= order)
                || previous_end.is_some_and(|end| window.start_ppk2_time_ms < end)
            {
                return Err(PhaseEnergyAuditError::new(
                    format!("receipt.events.{}.phase_windows.{}.consistency", event.event_id, window.phase_id),
                    "phase timing, order or non-overlap consistency failed",
                ));
            }
            let expected_power = window.time_weighted_mean_current_ua * receipt.voltage_v;
            let expected_energy = window.mean_power_uw * window.duration_ms / 1_000.0;
            let expected_alignment_energy =
                2.0 * window.maximum_observed_current_ua * receipt.voltage_v
                    * window.boundary_alignment_error_bound_ms / 1_000.0;
            let close = |left: f64, right: f64| {
                (left - right).abs() <= 1e-8 * left.abs().max(right.abs()).max(1.0)
            };
            if !close(window.mean_power_uw, expected_power)
                || !close(window.energy_uj, expected_energy)
                || !close(window.boundary_alignment_energy_uncertainty_uj, expected_alignment_energy)
            {
                return Err(PhaseEnergyAuditError::new(
                    format!("receipt.events.{}.phase_windows.{}.derived_metrics", event.event_id, window.phase_id),
                    "phase energy, power or alignment uncertainty is inconsistent",
                ));
            }
            declared_energy += window.energy_uj;
            declared_duration += window.duration_ms;
            previous_order = Some(window.order);
            previous_end = Some(window.end_ppk2_time_ms);
        }
        if !event.declared_phase_energy_sum_uj.is_finite()
            || event.declared_phase_energy_sum_uj < 0.0
            || (event.declared_phase_energy_sum_uj - declared_energy).abs()
                > 1e-8 * declared_energy.abs().max(1.0)
        {
            return Err(PhaseEnergyAuditError::new(
                format!("receipt.events.{}.declared_phase_energy_sum_uj", event.event_id),
                "does not equal the phase-window energy sum",
            ));
        }

        match event.status {
            PhaseEnergyEventStatus::MissingRequiredPhase => {
                missing_count += 1;
                if event.missing_required_phase_ids.is_empty()
                    || event.total_event_energy_uj.is_some()
                    || event.total_event_duration_ms.is_some()
                    || event.total_event_boundary_alignment_uncertainty_uj.is_some()
                    || event.unattributed_energy_uj.is_some()
                    || event.unattributed_duration_ms.is_some()
                {
                    return Err(PhaseEnergyAuditError::new(
                        format!("receipt.events.{}.status", event.event_id),
                        "missing required phase must not claim a complete event envelope",
                    ));
                }
            }
            PhaseEnergyEventStatus::UnattributedIntervalsPresent
            | PhaseEnergyEventStatus::CompletePhaseCoverage => {
                if !event.missing_required_phase_ids.is_empty()
                    || event.event_start_ppk2_time_ms.is_none()
                    || event.event_end_ppk2_time_ms.is_none()
                    || event.total_event_duration_ms.is_none()
                    || event.total_event_energy_uj.is_none()
                    || event.maximum_event_current_ua.is_none()
                    || event.total_event_boundary_alignment_uncertainty_uj.is_none()
                    || event.unattributed_duration_ms.is_none()
                    || event.unattributed_energy_uj.is_none()
                {
                    return Err(PhaseEnergyAuditError::new(
                        format!("receipt.events.{}.status", event.event_id),
                        "measured event envelope is missing required metrics",
                    ));
                }
                let start = event.event_start_ppk2_time_ms.unwrap();
                let end = event.event_end_ppk2_time_ms.unwrap();
                let duration = event.total_event_duration_ms.unwrap();
                let energy = event.total_event_energy_uj.unwrap();
                let max_current = event.maximum_event_current_ua.unwrap();
                let boundary_energy = event.total_event_boundary_alignment_uncertainty_uj.unwrap();
                let gap_duration = event.unattributed_duration_ms.unwrap();
                let gap_energy = event.unattributed_energy_uj.unwrap();
                for (field, value) in [
                    ("event_start_ppk2_time_ms", start),
                    ("event_end_ppk2_time_ms", end),
                    ("total_event_duration_ms", duration),
                    ("total_event_energy_uj", energy),
                    ("maximum_event_current_ua", max_current),
                    ("total_event_boundary_alignment_uncertainty_uj", boundary_energy),
                    ("unattributed_duration_ms", gap_duration),
                    ("unattributed_energy_uj", gap_energy),
                ] {
                    if !value.is_finite() || value < 0.0 {
                        return Err(PhaseEnergyAuditError::new(
                            format!("receipt.events.{}.{}", event.event_id, field),
                            "must be finite and non-negative",
                        ));
                    }
                }
                let expected_boundary_energy =
                    2.0 * max_current * receipt.voltage_v
                        * event.phase_windows.first().map(|w| w.boundary_alignment_error_bound_ms).unwrap_or(0.0)
                        / 1_000.0;
                let close = |left: f64, right: f64| {
                    (left - right).abs() <= 1e-8 * left.abs().max(right.abs()).max(1.0)
                };
                if end <= start
                    || !close(end - start, duration)
                    || !close(duration - declared_duration, gap_duration)
                    || !close(energy - declared_energy, gap_energy)
                    || gap_duration > receipt.maximum_unattributed_gap_ms
                        && event.status == PhaseEnergyEventStatus::CompletePhaseCoverage
                    || !event.missing_phase_ids.is_empty()
                        && event.status == PhaseEnergyEventStatus::CompletePhaseCoverage
                    || !close(boundary_energy, expected_boundary_energy)
                {
                    return Err(PhaseEnergyAuditError::new(
                        format!("receipt.events.{}.envelope_consistency", event.event_id),
                        "event duration, phase sum, unattributed residual, status or alignment bound is inconsistent",
                    ));
                }
                if event.status == PhaseEnergyEventStatus::CompletePhaseCoverage {
                    complete_count += 1;
                } else {
                    unattributed_count += 1;
                }
            }
        }
    }
    if complete_count != receipt.complete_phase_coverage_events
        || unattributed_count != receipt.unattributed_interval_events
        || missing_count != receipt.missing_required_phase_events
    {
        return Err(PhaseEnergyAuditError::new("receipt.event_status_counts", "status counts do not match event inventory"));
    }
    if compute_phase_energy_audit_receipt_sha256(receipt)? != receipt.receipt_sha256 {
        return Err(PhaseEnergyAuditError::new("receipt.receipt_sha256", "does not match canonical receipt"));
    }
    Ok(())
}

/// Rebuild the complete phase audit from source sample/marker bytes and compare the receipt.
pub fn verify_phase_energy_receipt_against_sources(
    sample_csv: &str,
    marker_csv: &str,
    trace_config: Ppk2TraceAuditConfig,
    mapping: &PhaseClockMappingManifest,
    protocol: &ApplicationPhaseProtocolManifest,
    expected: &PhaseEnergyAuditReceipt,
) -> Result<(), PhaseEnergyAuditError> {
    validate_phase_energy_audit_receipt(expected)?
    let rebuilt = audit_application_phase_energy(sample_csv, marker_csv, trace_config, mapping, protocol)?;
    if rebuilt.receipt != *expected {
        return Err(PhaseEnergyAuditError::new("receipt.source_reproduction", "recomputed phase receipt does not exactly match expected receipt"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn trace_config() -> Ppk2TraceAuditConfig {
        Ppk2TraceAuditConfig {
            voltage_v: 3.3,
            expected_sample_rate_hz: 100_000.0,
            sample_rate_tolerance_fraction: 0.01,
            maximum_gap_ratio: 1.1,
        }
    }

    fn mapping() -> PhaseClockMappingManifest {
        let mut value = PhaseClockMappingManifest {
            schema: CLOCK_SCHEMA.into(),
            version: VERSION.into(),
            mapping_id: "fixture-clock-map".into(),
            marker_clock_domain_id: "logic-analyzer-clock".into(),
            marker_source_id: "gpio-sync-1".into(),
            ppk2_clock_domain_id: "ppk2-time-ms".into(),
            marker_origin_ns: 0,
            ppk2_origin_ms: 0.0,
            ppk2_ms_per_marker_ns: 1e-6,
            scale_tolerance_fraction: 0.001,
            maximum_alignment_error_ms: 0.0001,
            clock_calibration_evidence_sha256: sha('a'),
            mapping_sha256: String::new(),
        };
        value.mapping_sha256 = compute_phase_clock_mapping_sha256(&value).unwrap();
        value
    }

    fn protocol(max_gap_ms: f64) -> ApplicationPhaseProtocolManifest {
        let mut value = ApplicationPhaseProtocolManifest {
            schema: PROTOCOL_SCHEMA.into(),
            version: VERSION.into(),
            benchmark_id: "tiny-sensor-event-v1".into(),
            protocol_id: "wake-inference-post-v1".into(),
            phases: vec![
                ApplicationPhaseDefinition { phase_id: "wake".into(), order: 0, required: true },
                ApplicationPhaseDefinition { phase_id: "inference".into(), order: 1, required: true },
                ApplicationPhaseDefinition { phase_id: "postprocess".into(), order: 2, required: true },
            ],
            maximum_unattributed_gap_ms: max_gap_ms,
            protocol_sha256: String::new(),
        };
        value.protocol_sha256 = compute_application_phase_protocol_sha256(&value).unwrap();
        value
    }

    fn markers(wake_end_ns: u64, inference_start_ns: u64) -> String {
        format!(
            "event_id,segment_id,phase_id,edge,marker_timestamp_ns,sequence,clock_domain_id,marker_source_id\n\
             e1,e1,wake,START,0,0,logic-analyzer-clock,gpio-sync-1\n\
             e1,e1,wake,END,{wake_end_ns},1,logic-analyzer-clock,gpio-sync-1\n\
             e1,e1,inference,START,{inference_start_ns},2,logic-analyzer-clock,gpio-sync-1\n\
             e1,e1,inference,END,30000,3,logic-analyzer-clock,gpio-sync-1\n\
             e1,e1,postprocess,START,30000,4,logic-analyzer-clock,gpio-sync-1\n\
             e1,e1,postprocess,END,40000,5,logic-analyzer-clock,gpio-sync-1\n"
        )
    }

    fn samples() -> &'static str {
        concat!(
            "segment_id,time_ms,current_uA\n",
            "e1,0.00,1000\n",
            "e1,0.01,1000\n",
            "e1,0.02,1000\n",
            "e1,0.03,1000\n",
            "e1,0.04,1000\n",
        )
    }

    #[test]
    fn integrates_complete_phase_windows_and_whole_event() {
        let result = audit_application_phase_energy(
            samples(),
            &markers(10_000, 10_000),
            trace_config(),
            &mapping(),
            &protocol(0.0),
        ).unwrap();
        let event = &result.receipt.events[0];
        assert_eq!(event.status, PhaseEnergyEventStatus::CompletePhaseCoverage);
        assert_eq!(event.phase_windows.len(), 3);
        assert!((event.total_event_energy_uj.unwrap() - 0.132).abs() < 1e-12);
        assert!((event.phase_windows.iter().map(|window| window.energy_uj).sum::<f64>() - 0.000132).abs() < 1e-12);
        verify_phase_energy_receipt_against_sources(
            samples(),
            &markers(10_000, 10_000),
            trace_config(),
            &mapping(),
            &protocol(0.0),
            &result.receipt,
        ).unwrap();
    }

    #[test]
    fn integrates_transition_gaps_but_marks_them_unattributed() {
        let result = audit_application_phase_energy(
            samples(),
            &markers(10_000, 20_000),
            trace_config(),
            &mapping(),
            &protocol(0.001),
        ).unwrap();
        let event = &result.receipt.events[0];
        assert_eq!(event.status, PhaseEnergyEventStatus::UnattributedIntervalsPresent);
        assert!((event.unattributed_duration_ms.unwrap() - 0.01).abs() < 1e-12);
        assert!((event.unattributed_energy_uj.unwrap() - 0.033).abs() < 1e-12);
        assert!((event.total_event_energy_uj.unwrap() - 0.000132).abs() < 1e-12);
    }

    #[test]
    fn missing_required_phase_never_receives_complete_event_energy() {
        let csv = concat!(
            "event_id,segment_id,phase_id,edge,marker_timestamp_ns,sequence,clock_domain_id,marker_source_id\n",
            "e1,e1,wake,START,0,0,logic-analyzer-clock,gpio-sync-1\n",
            "e1,e1,wake,END,10000,1,logic-analyzer-clock,gpio-sync-1\n",
            "e1,e1,postprocess,START,30000,2,logic-analyzer-clock,gpio-sync-1\n",
            "e1,e1,postprocess,END,40000,3,logic-analyzer-clock,gpio-sync-1\n",
        );
        let result = audit_application_phase_energy(
            samples(), csv, trace_config(), &mapping(), &protocol(1.0),
        ).unwrap();
        let event = &result.receipt.events[0];
        assert_eq!(event.status, PhaseEnergyEventStatus::MissingRequiredPhase);
        assert!(event.total_event_energy_uj.is_none());
        assert_eq!(event.missing_required_phase_ids, vec!["inference".to_owned()]);
    }

    #[test]
    fn rejects_overlapping_phase_windows() {
        let result = audit_application_phase_energy(
            samples(),
            &markers(20_000, 10_000),
            trace_config(),
            &mapping(),
            &protocol(1.0),
        );
        assert!(result.unwrap_err().to_string().contains("overlaps"));
    }

    #[test]
    fn rejects_clock_domain_substitution() {
        let marker = markers(10_000, 10_000).replace("logic-analyzer-clock", "untrusted-clock");
        assert!(audit_application_phase_energy(
            samples(), &marker, trace_config(), &mapping(), &protocol(0.0),
        ).unwrap_err().to_string().contains("clock domain"));
    }
}
