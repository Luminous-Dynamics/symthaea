//! File-backed EDLC trace evidence receipt audit.
//!
//! Verifies relative paths, artifact byte lengths, SHA-256 digests, CSV structure,
//! event/profile linkage, and declared device/instrument/calibration consistency.
//! A hash match establishes byte integrity only; it does not authenticate origin.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

const FIXTURE_PATH: &str =
    "fixtures/energy/electrochemical-capacitor/trace_evidence_receipt_known_answer_v0_1.json";
const SCHEMA_PATH: &str =
    "schemas/energy/edlc_trace_evidence_receipt_v0_1.schema.json";
const EXPECTED_RECEIPT_ID: &str = "EDLC-TRACE-RECEIPT-KNOWN-ANSWER-001";
const EXPECTED_EVIDENCE_CLASS: &str = "synthetic_artifact_integrity";
const EXPECTED_CLAIM_CEILING: &str =
    "ARTIFACT_HASH_PATH_AND_STRUCTURAL_METADATA_INTEGRITY_ONLY_NOT_PHYSICAL_ORIGIN_AUTHENTICITY_OR_RELIABILITY";
const EXPECTED_INTEGRITY_SCOPE: &str =
    "sha256_byte_integrity_and_structural_metadata_only";
const EXPECTED_PROVENANCE_LIMIT: &str =
    "SHA256_MATCH_DOES_NOT_AUTHENTICATE_CREATOR_INSTRUMENT_CALIBRATION_OR_PHYSICAL_DEVICE";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: String,
    receipt_id: String,
    evidence_class: String,
    claim_ceiling: String,
    integrity_scope: String,
    provenance_limit: String,
    acquisition: Acquisition,
    device: Device,
    instrument: Instrument,
    protocol: Protocol,
    environment: Environment,
    trace: Trace,
    events: Vec<Event>,
    artifacts: Vec<Artifact>,
    expected: Expected,
    negative_controls: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Acquisition {
    start_unix_ms: u64,
    end_unix_ms: u64,
    sample_clock_reference: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Device {
    manufacturer: String,
    model: String,
    physical_device_serial: String,
    lot_id: String,
    device_boundary: String,
    identity_artifact_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Instrument {
    instrument_id: String,
    model: String,
    serial: String,
    sample_rate_hz: f64,
    timebase_reference: String,
    configuration_artifact_id: String,
    calibration_artifact_id: String,
    calibration_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    protocol_id: String,
    sequence_profile_id: String,
    standby_mode: String,
    device_boundary: String,
    artifact_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    ambient_temperature_c: f64,
    relative_humidity_percent: f64,
    artifact_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Trace {
    artifact_id: String,
    expected_row_count: usize,
    required_columns: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    event_index: usize,
    event_id: String,
    load_profile_id: String,
    load_profile_artifact_id: String,
    readiness_probe_profile_id: Option<String>,
    readiness_probe_artifact_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    artifact_id: String,
    role: String,
    relative_path: String,
    sha256: String,
    byte_length: u64,
    media_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    artifact_count: usize,
    raw_trace_rows: usize,
}

#[derive(Debug, PartialEq)]
struct AuditReport {
    artifact_count: usize,
    raw_trace_rows: usize,
    event_count: usize,
    device_serial: String,
    sequence_profile_id: String,
}

fn nonempty(value: &str, context: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{context} must not be empty");
    }
    Ok(())
}

fn close_enough(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}

fn artifact_by_id<'a>(
    artifacts: &'a BTreeMap<String, &'a Artifact>,
    artifact_id: &str,
    context: &str,
) -> Result<&'a Artifact> {
    artifacts
        .get(artifact_id)
        .copied()
        .with_context(|| format!("{context} references unknown artifact {artifact_id}"))
}

fn safe_artifact_path(root: &Path, relative_path: &str) -> Result<PathBuf> {
    nonempty(relative_path, "artifact.relative_path")?;
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            !matches!(component, Component::Normal(_))
        })
    {
        bail!("artifact path must be relative and contain no parent/root/prefix components");
    }

    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("could not canonicalize workspace root {}", root.display()))?;
    let candidate = canonical_root.join(relative);
    let canonical_file = candidate
        .canonicalize()
        .with_context(|| format!("artifact file does not exist: {}", candidate.display()))?;
    if !canonical_file.starts_with(&canonical_root) {
        bail!("artifact path resolves outside the workspace root");
    }
    let metadata = std::fs::metadata(&canonical_file)
        .with_context(|| format!("could not stat artifact {}", canonical_file.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        bail!("artifact must be a non-empty regular file: {}", canonical_file.display());
    }
    Ok(canonical_file)
}

fn validate_digest(value: &str, context: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{context} must be a lowercase 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn validate_artifacts(root: &Path, receipt: &Receipt) -> Result<BTreeMap<String, &Artifact>> {
    if receipt.artifacts.is_empty() {
        bail!("receipt must contain artifacts");
    }
    if receipt.expected.artifact_count != receipt.artifacts.len() {
        bail!(
            "artifact count mismatch: manifest expects {}, contains {}",
            receipt.expected.artifact_count,
            receipt.artifacts.len()
        );
    }

    let mut artifacts = BTreeMap::new();
    let mut paths = BTreeSet::new();
    let mut singleton_roles = BTreeSet::new();

    for artifact in &receipt.artifacts {
        nonempty(&artifact.artifact_id, "artifact.artifact_id")?;
        nonempty(&artifact.role, "artifact.role")?;
        nonempty(&artifact.media_type, "artifact.media_type")?;
        validate_digest(&artifact.sha256, "artifact.sha256")?;
        if !artifacts
            .insert(artifact.artifact_id.clone(), artifact)
            .is_none()
        {
            bail!("duplicate artifact_id {}", artifact.artifact_id);
        }
        if !paths.insert(artifact.relative_path.clone()) {
            bail!("duplicate artifact path {}", artifact.relative_path);
        }

        let path = safe_artifact_path(root, &artifact.relative_path)?;
        let bytes = std::fs::read(&path)
            .with_context(|| format!("could not read artifact {}", path.display()))?;
        let observed_length = u64::try_from(bytes.len())
            .context("artifact byte length does not fit u64")?;
        if observed_length != artifact.byte_length {
            bail!(
                "{} byte length mismatch: manifest {}, actual {}",
                artifact.artifact_id,
                artifact.byte_length,
                observed_length
            );
        }
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != artifact.sha256 {
            bail!(
                "{} SHA-256 mismatch: manifest {}, actual {}",
                artifact.artifact_id,
                artifact.sha256,
                digest
            );
        }

        let expected_media_type = match artifact.role.as_str() {
            "raw_terminal_trace" | "load_profile" | "readiness_probe_profile" => "text/csv",
            "device_identity_record"
            | "instrument_configuration"
            | "calibration_certificate"
            | "test_protocol"
            | "environment_log" => "application/json",
            other => bail!("unsupported artifact role {other}"),
        };
        if artifact.media_type != expected_media_type {
            bail!(
                "{} media type {} conflicts with role {}",
                artifact.artifact_id,
                artifact.media_type,
                artifact.role
            );
        }

        if matches!(
            artifact.role.as_str(),
            "raw_terminal_trace"
                | "device_identity_record"
                | "instrument_configuration"
                | "calibration_certificate"
                | "test_protocol"
                | "environment_log"
        ) && !singleton_roles.insert(artifact.role.clone())
        {
            bail!("duplicate singleton artifact role {}", artifact.role);
        }
    }

    for role in [
        "raw_terminal_trace",
        "device_identity_record",
        "instrument_configuration",
        "calibration_certificate",
        "test_protocol",
        "environment_log",
    ] {
        if !singleton_roles.contains(role) {
            bail!("required artifact role {role} is missing");
        }
    }

    Ok(artifacts)
}

fn artifact_path(root: &Path, artifact: &Artifact) -> Result<PathBuf> {
    safe_artifact_path(root, &artifact.relative_path)
}

fn read_json_artifact(root: &Path, artifact: &Artifact) -> Result<Value> {
    let path = artifact_path(root, artifact)?;
    let bytes = std::fs::read(&path)?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not valid JSON", artifact.artifact_id))
}

fn json_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))
}

fn json_number(value: &Value, key: &str, context: &str) -> Result<f64> {
    value[key]
        .as_f64()
        .with_context(|| format!("{context}.{key} must be a number"))
}

fn column_index(headers: &csv::StringRecord, name: &str, context: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header == name)
        .with_context(|| format!("{context} is missing required CSV column {name}"))
}

fn parse_csv_number(record: &csv::StringRecord, index: usize, context: &str) -> Result<f64> {
    let text = record
        .get(index)
        .with_context(|| format!("{context} is missing a CSV value"))?;
    let value = text
        .parse::<f64>()
        .with_context(|| format!("{context} is not a valid number: {text}"))?;
    if !value.is_finite() {
        bail!("{context} must be finite");
    }
    Ok(value)
}

fn validate_profile_csv(
    root: &Path,
    artifact: &Artifact,
    value_column: &str,
) -> Result<usize> {
    if artifact.role != "load_profile" && artifact.role != "readiness_probe_profile" {
        bail!("{} is not a profile artifact", artifact.artifact_id);
    }
    let path = artifact_path(root, artifact)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .with_context(|| format!("could not open profile CSV {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let time_index = column_index(&headers, "elapsed_s", &artifact.artifact_id)?;
    let value_index = column_index(&headers, value_column, &artifact.artifact_id)?;
    let mut rows = 0_usize;
    let mut previous_time = None;

    for record in reader.records() {
        let record = record?;
        let elapsed = parse_csv_number(&record, time_index, "profile.elapsed_s")?;
        let value = parse_csv_number(&record, value_index, value_column)?;
        if elapsed < 0.0 || value < 0.0 {
            bail!("profile time and load values must be non-negative");
        }
        if previous_time.is_some_and(|previous| elapsed <= previous) {
            bail!("profile elapsed_s must increase strictly");
        }
        previous_time = Some(elapsed);
        rows += 1;
    }
    if rows < 2 {
        bail!("profile {} must contain at least two samples", artifact.artifact_id);
    }
    Ok(rows)
}

fn validate_raw_trace(
    root: &Path,
    receipt: &Receipt,
    artifact: &Artifact,
) -> Result<usize> {
    if artifact.role != "raw_terminal_trace" {
        bail!("trace.artifact_id must reference the raw_terminal_trace role");
    }
    let path = artifact_path(root, artifact)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .with_context(|| format!("could not open raw trace {}", path.display()))?;
    let headers = reader.headers()?.clone();
    for column in &receipt.trace.required_columns {
        column_index(&headers, column, "trace.required_columns")?;
    }

    let event_index = column_index(&headers, "event_id", "raw_trace")?;
    let elapsed_index = column_index(&headers, "elapsed_s", "raw_trace")?;
    let terminal_voltage_index = column_index(&headers, "terminal_voltage_v", "raw_trace")?;
    let terminal_current_index = column_index(&headers, "terminal_current_a", "raw_trace")?;
    let source_voltage_index = column_index(&headers, "source_voltage_v", "raw_trace")?;
    let source_current_index = column_index(&headers, "source_current_a", "raw_trace")?;
    let temperature_index = column_index(&headers, "temperature_c", "raw_trace")?;

    let expected_event_ids: BTreeSet<&str> =
        receipt.events.iter().map(|event| event.event_id.as_str()).collect();
    let mut event_rows = BTreeMap::<String, usize>::new();
    let mut last_elapsed = BTreeMap::<String, f64>::new();
    let mut rows = 0_usize;

    for record in reader.records() {
        let record = record?;
        let event_id = record
            .get(event_index)
            .context("raw trace row is missing event_id")?;
        if !expected_event_ids.contains(event_id) {
            bail!("raw trace contains undeclared event_id {event_id}");
        }
        let elapsed = parse_csv_number(&record, elapsed_index, "raw_trace.elapsed_s")?;
        let terminal_voltage =
            parse_csv_number(&record, terminal_voltage_index, "raw_trace.terminal_voltage_v")?;
        let _terminal_current =
            parse_csv_number(&record, terminal_current_index, "raw_trace.terminal_current_a")?;
        let _source_voltage =
            parse_csv_number(&record, source_voltage_index, "raw_trace.source_voltage_v")?;
        let source_current =
            parse_csv_number(&record, source_current_index, "raw_trace.source_current_a")?;
        let _temperature =
            parse_csv_number(&record, temperature_index, "raw_trace.temperature_c")?;

        if elapsed < 0.0 || terminal_voltage < 0.0 {
            bail!("raw trace elapsed time and terminal voltage must be non-negative");
        }
        if receipt.protocol.standby_mode == "open_circuit" && source_current.abs() > 1.0e-12 {
            bail!("open-circuit trace contains non-zero external source current");
        }
        if let Some(previous) = last_elapsed.get(event_id) {
            if elapsed <= *previous {
                bail!("raw trace elapsed_s must increase strictly within each event");
            }
        }
        last_elapsed.insert(event_id.to_string(), elapsed);
        *event_rows.entry(event_id.to_string()).or_insert(0) += 1;
        rows += 1;
    }

    if rows != receipt.trace.expected_row_count || rows != receipt.expected.raw_trace_rows {
        bail!(
            "raw trace row count mismatch: observed {rows}, trace.expected_row_count={}, expected.raw_trace_rows={}",
            receipt.trace.expected_row_count,
            receipt.expected.raw_trace_rows
        );
    }
    for event_id in &expected_event_ids {
        let count = event_rows.get(*event_id).copied().unwrap_or(0);
        if count < 2 {
            bail!("raw trace event {event_id} must contain at least two rows");
        }
    }
    if event_rows.len() != expected_event_ids.len() {
        bail!("raw trace event coverage does not match declared sequence");
    }
    Ok(rows)
}

fn validate_json_cross_links(
    root: &Path,
    receipt: &Receipt,
    artifacts: &BTreeMap<String, &Artifact>,
) -> Result<()> {
    let device_artifact = artifact_by_id(
        artifacts,
        &receipt.device.identity_artifact_id,
        "device.identity_artifact_id",
    )?;
    if device_artifact.role != "device_identity_record" {
        bail!("device identity link references the wrong artifact role");
    }
    let device_value = read_json_artifact(root, device_artifact)?;
    for (key, expected) in [
        ("manufacturer", receipt.device.manufacturer.as_str()),
        ("model", receipt.device.model.as_str()),
        (
            "physical_device_serial",
            receipt.device.physical_device_serial.as_str(),
        ),
        ("lot_id", receipt.device.lot_id.as_str()),
        ("device_boundary", receipt.device.device_boundary.as_str()),
    ] {
        if json_string(&device_value, key, "device_identity")? != expected {
            bail!("device identity artifact does not match manifest field {key}");
        }
    }

    let instrument_artifact = artifact_by_id(
        artifacts,
        &receipt.instrument.configuration_artifact_id,
        "instrument.configuration_artifact_id",
    )?;
    if instrument_artifact.role != "instrument_configuration" {
        bail!("instrument configuration link references the wrong artifact role");
    }
    let instrument_value = read_json_artifact(root, instrument_artifact)?;
    for (key, expected) in [
        ("instrument_id", receipt.instrument.instrument_id.as_str()),
        ("model", receipt.instrument.model.as_str()),
        ("serial", receipt.instrument.serial.as_str()),
        (
            "timebase_reference",
            receipt.instrument.timebase_reference.as_str(),
        ),
    ] {
        if json_string(&instrument_value, key, "instrument_configuration")? != expected {
            bail!("instrument configuration artifact does not match manifest field {key}");
        }
    }
    if !close_enough(
        json_number(&instrument_value, "sample_rate_hz", "instrument_configuration")?,
        receipt.instrument.sample_rate_hz,
    ) {
        bail!("instrument sample rate does not match instrument configuration artifact");
    }

    let calibration_artifact = artifact_by_id(
        artifacts,
        &receipt.instrument.calibration_artifact_id,
        "instrument.calibration_artifact_id",
    )?;
    if calibration_artifact.role != "calibration_certificate" {
        bail!("calibration link references the wrong artifact role");
    }
    let calibration_value = read_json_artifact(root, calibration_artifact)?;
    if json_string(&calibration_value, "calibration_id", "calibration_certificate")?
        != receipt.instrument.calibration_id
    {
        bail!("calibration ID does not match calibration certificate");
    }
    if json_string(&calibration_value, "instrument_serial", "calibration_certificate")?
        != receipt.instrument.serial
    {
        bail!("calibration certificate instrument serial mismatch");
    }
    let valid_from = calibration_value["valid_from_unix_ms"]
        .as_u64()
        .context("calibration.valid_from_unix_ms must be an unsigned integer")?;
    let valid_until = calibration_value["valid_until_unix_ms"]
        .as_u64()
        .context("calibration.valid_until_unix_ms must be an unsigned integer")?;
    if valid_from > receipt.acquisition.start_unix_ms
        || valid_until < receipt.acquisition.end_unix_ms
        || valid_from >= valid_until
    {
        bail!("calibration validity interval does not cover the full acquisition");
    }

    let protocol_artifact = artifact_by_id(
        artifacts,
        &receipt.protocol.artifact_id,
        "protocol.artifact_id",
    )?;
    if protocol_artifact.role != "test_protocol" {
        bail!("protocol link references the wrong artifact role");
    }
    let protocol_value = read_json_artifact(root, protocol_artifact)?;
    for (key, expected) in [
        ("protocol_id", receipt.protocol.protocol_id.as_str()),
        (
            "sequence_profile_id",
            receipt.protocol.sequence_profile_id.as_str(),
        ),
        ("standby_mode", receipt.protocol.standby_mode.as_str()),
    ] {
        if json_string(&protocol_value, key, "protocol")? != expected {
            bail!("protocol artifact does not match manifest field {key}");
        }
    }
    let order = protocol_value["event_order"]
        .as_array()
        .context("protocol.event_order must be an array")?;
    if order.len() != receipt.events.len()
        || order.iter().zip(&receipt.events).any(|(left, right)| {
            left.as_str() != Some(right.event_id.as_str())
        })
    {
        bail!("protocol event order does not match receipt event order");
    }
    if json_string(&protocol_value, "device_boundary", "protocol")
        .unwrap_or(receipt.protocol.device_boundary)
        != receipt.protocol.device_boundary
    {
        bail!("protocol device boundary mismatch");
    }

    let environment_artifact = artifact_by_id(
        artifacts,
        &receipt.environment.artifact_id,
        "environment.artifact_id",
    )?;
    if environment_artifact.role != "environment_log" {
        bail!("environment link references the wrong artifact role");
    }
    let environment_value = read_json_artifact(root, environment_artifact)?;
    if !close_enough(
        json_number(&environment_value, "ambient_temperature_c", "environment_log")?,
        receipt.environment.ambient_temperature_c,
    ) || !close_enough(
        json_number(&environment_value, "relative_humidity_percent", "environment_log")?,
        receipt.environment.relative_humidity_percent,
    ) {
        bail!("environment record does not match manifest values");
    }

    Ok(())
}

fn audit_receipt(root: &Path, receipt: &Receipt) -> Result<AuditReport> {
    if receipt.schema_version != "0.1.0" {
        bail!("unsupported EDLC trace receipt schema_version");
    }
    if receipt.receipt_id != EXPECTED_RECEIPT_ID {
        bail!("unexpected EDLC trace receipt ID");
    }
    if receipt.evidence_class != EXPECTED_EVIDENCE_CLASS
        || receipt.claim_ceiling != EXPECTED_CLAIM_CEILING
        || receipt.integrity_scope != EXPECTED_INTEGRITY_SCOPE
        || receipt.provenance_limit != EXPECTED_PROVENANCE_LIMIT
    {
        bail!("trace receipt evidence ceiling or integrity scope was changed");
    }
    if receipt.negative_controls.is_empty() {
        bail!("trace receipt must enumerate negative controls");
    }
    for (index, control) in receipt.negative_controls.iter().enumerate() {
        nonempty(control, &format!("negative_controls[{index}]"))?;
    }

    if receipt.acquisition.start_unix_ms == 0
        || receipt.acquisition.end_unix_ms <= receipt.acquisition.start_unix_ms
    {
        bail!("acquisition interval must be positive and ordered");
    }
    nonempty(
        &receipt.acquisition.sample_clock_reference,
        "acquisition.sample_clock_reference",
    )?;
    nonempty(&receipt.device.manufacturer, "device.manufacturer")?;
    nonempty(&receipt.device.model, "device.model")?;
    nonempty(
        &receipt.device.physical_device_serial,
        "device.physical_device_serial",
    )?;
    nonempty(&receipt.device.lot_id, "device.lot_id")?;
    nonempty(&receipt.instrument.instrument_id, "instrument.instrument_id")?;
    nonempty(&receipt.instrument.model, "instrument.model")?;
    nonempty(&receipt.instrument.serial, "instrument.serial")?;
    nonempty(&receipt.instrument.timebase_reference, "instrument.timebase_reference")?;
    if !receipt.instrument.sample_rate_hz.is_finite() || receipt.instrument.sample_rate_hz <= 0.0 {
        bail!("instrument.sample_rate_hz must be finite and positive");
    }
    if receipt.instrument.timebase_reference != receipt.acquisition.sample_clock_reference {
        bail!("instrument and acquisition sample-clock references disagree");
    }
    if !matches!(
        receipt.protocol.standby_mode.as_str(),
        "open_circuit" | "regulated_voltage_hold"
    ) {
        bail!("unsupported standby mode");
    }
    if !matches!(
        receipt.device.device_boundary.as_str(),
        "cell" | "complete_device" | "installed_system"
    ) || receipt.protocol.device_boundary != receipt.device.device_boundary
    {
        bail!("device and protocol boundaries must match");
    }
    if !receipt.environment.ambient_temperature_c.is_finite()
        || !(-100.0..=200.0).contains(&receipt.environment.ambient_temperature_c)
        || !receipt.environment.relative_humidity_percent.is_finite()
        || !(0.0..=100.0).contains(&receipt.environment.relative_humidity_percent)
    {
        bail!("environment metadata is outside admissible numeric bounds");
    }
    if receipt.events.len() < 2 {
        bail!("a sequential reserve receipt must contain at least two events");
    }
    if receipt.expected.raw_trace_rows == 0 || receipt.trace.expected_row_count == 0 {
        bail!("raw trace must declare at least one row");
    }

    let mut event_ids = BTreeSet::new();
    for (position, event) in receipt.events.iter().enumerate() {
        if event.event_index != position + 1 {
            bail!("event indexes must be contiguous and start at one");
        }
        nonempty(&event.event_id, "event.event_id")?;
        nonempty(&event.load_profile_id, "event.load_profile_id")?;
        if !event_ids.insert(event.event_id.clone()) {
            bail!("duplicate event ID {}", event.event_id);
        }
        let first = position == 0;
        if first {
            if event.readiness_probe_profile_id.is_some()
                || event.readiness_probe_artifact_id.is_some()
            {
                bail!("first event must not reference an inter-call readiness probe");
            }
        } else {
            let profile_id = event
                .readiness_probe_profile_id
                .as_deref()
                .context("later event requires readiness_probe_profile_id")?;
            nonempty(profile_id, "event.readiness_probe_profile_id")?;
            if event.readiness_probe_artifact_id.is_none() {
                bail!("later event requires readiness_probe_artifact_id");
            }
        }
    }

    let artifacts = validate_artifacts(root, receipt)?;
    validate_json_cross_links(root, receipt, &artifacts)?;

    let raw_trace = artifact_by_id(&artifacts, &receipt.trace.artifact_id, "trace.artifact_id")?;
    let row_count = validate_raw_trace(root, receipt, raw_trace)?;
    let mut profile_artifact_ids = BTreeSet::new();
    for event in &receipt.events {
        let load = artifact_by_id(
            &artifacts,
            &event.load_profile_artifact_id,
            "event.load_profile_artifact_id",
        )?;
        if load.role != "load_profile" {
            bail!("event {} load profile link references the wrong role", event.event_id);
        }
        if !profile_artifact_ids.insert(event.load_profile_artifact_id.clone()) {
            bail!("a load-profile artifact cannot be silently reused across distinct events");
        }
        validate_profile_csv(root, load, "load_current_a")?;

        if let Some(probe_id) = &event.readiness_probe_artifact_id {
            let probe = artifact_by_id(&artifacts, probe_id, "event.readiness_probe_artifact_id")?;
            if probe.role != "readiness_probe_profile" {
                bail!("event {} readiness probe link references the wrong role", event.event_id);
            }
            validate_profile_csv(root, probe, "probe_current_a")?;
        }
    }

    if receipt.expected.artifact_count != receipt.artifacts.len() {
        bail!("expected artifact count does not match manifest");
    }

    Ok(AuditReport {
        artifact_count: receipt.artifacts.len(),
        raw_trace_rows: row_count,
        event_count: receipt.events.len(),
        device_serial: receipt.device.physical_device_serial.clone(),
        sequence_profile_id: receipt.protocol.sequence_profile_id.clone(),
    })
}

fn audit_schema(root: &Path) -> Result<()> {
    let path = root.join(SCHEMA_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read trace receipt schema {}", path.display()))?;
    let schema: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("trace receipt schema {} is not valid JSON", path.display()))?;
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        bail!("trace receipt schema must declare JSON Schema draft 2020-12");
    }
    if schema["additionalProperties"] != Value::Bool(false) {
        bail!("trace receipt schema must reject unknown top-level fields");
    }
    let required = schema["required"]
        .as_array()
        .context("trace receipt schema must declare required top-level fields")?;
    for field in [
        "schema_version",
        "receipt_id",
        "evidence_class",
        "claim_ceiling",
        "integrity_scope",
        "provenance_limit",
        "acquisition",
        "device",
        "instrument",
        "protocol",
        "environment",
        "trace",
        "events",
        "artifacts",
        "expected",
        "negative_controls",
    ] {
        if !required.iter().any(|entry| entry.as_str() == Some(field)) {
            bail!("trace receipt schema must require {field}");
        }
    }
    for definition in [
        "acquisition",
        "device",
        "instrument",
        "protocol",
        "environment",
        "trace",
        "event",
        "artifact",
        "expected",
    ] {
        if schema["$defs"][definition]["additionalProperties"] != Value::Bool(false) {
            bail!("trace receipt schema definition {definition} must reject unknown fields");
        }
    }
    if schema["$defs"]["event"]["allOf"]
        .as_array()
        .is_none_or(|rules| rules.len() < 2)
    {
        bail!("event schema must encode first and subsequent readiness-probe rules");
    }
    if schema["$defs"]["artifact"]["properties"]["sha256"]["pattern"].as_str()
        != Some("^[0-9a-f]{64}$")
    {
        bail!("artifact schema must require lowercase SHA-256 digests");
    }
    Ok(())
}

pub fn audit(root: &Path) -> Result<()> {
    audit_schema(root)?;
    let path = root.join(FIXTURE_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read trace receipt fixture {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("trace receipt fixture {} is not valid JSON", path.display()))?;
    let receipt: Receipt = serde_json::from_value(value)
        .context("trace receipt fixture does not satisfy its strict data contract")?;
    let report = audit_receipt(root, &receipt)?;

    println!("edlc-trace-evidence-audit: PASS (synthetic artifact integrity only)");
    println!(
        "artifacts={}, raw trace rows={}, events={}, device_serial={}, sequence_profile={}",
        report.artifact_count,
        report.raw_trace_rows,
        report.event_count,
        report.device_serial,
        report.sequence_profile_id
    );
    println!(
        "claim ceiling: SHA-256 and structural metadata integrity only; no origin authentication or physical device reliability"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::json;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask lives one level below the workspace root")
            .to_path_buf()
    }

    fn fixture_value() -> Value {
        let path = root().join(FIXTURE_PATH);
        let bytes = std::fs::read(path).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn audit_value(value: Value) -> Result<AuditReport> {
        let receipt: Receipt = serde_json::from_value(value)
            .context("trace receipt fixture does not satisfy its strict data contract")?;
        audit_receipt(&root(), &receipt)
    }

    #[test]
    fn known_answer_receipt_verifies_all_artifacts_and_cross_links() {
        let receipt: Receipt = serde_json::from_value(fixture_value()).unwrap();
        let report = audit_receipt(&root(), &receipt).unwrap();
        assert_eq!(report.artifact_count, 9);
        assert_eq!(report.raw_trace_rows, 4);
        assert_eq!(report.event_count, 2);
        assert_eq!(report.device_serial, "SYNTH-TRACE-001");
    }

    #[test]
    fn claim_ceiling_cannot_be_weakened() {
        let mut value = fixture_value();
        value["claim_ceiling"] = json!("PHYSICAL_DEVICE_VERIFIED");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("strict data contract") || error.contains("evidence ceiling"));
    }

    #[test]
    fn absolute_and_parent_traversal_paths_are_rejected() {
        let mut value = fixture_value();
        value["artifacts"][0]["relative_path"] = json!("../Cargo.toml");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("relative") || error.contains("outside") || error.contains("SHA-256"));
    }

    #[test]
    fn sha256_mismatch_is_rejected() {
        let mut value = fixture_value();
        value["artifacts"][0]["sha256"] = json!("0".repeat(64));
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("SHA-256 mismatch"));
    }

    #[test]
    fn byte_length_mismatch_is_rejected() {
        let mut value = fixture_value();
        value["artifacts"][0]["byte_length"] = json!(1);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("byte length mismatch"));
    }

    #[test]
    fn duplicate_artifact_ids_are_rejected() {
        let mut value = fixture_value();
        value["artifacts"][1]["artifact_id"] = json!("raw-terminal-trace");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("duplicate artifact_id"));
    }

    #[test]
    fn duplicate_artifact_paths_are_rejected() {
        let mut value = fixture_value();
        value["artifacts"][1]["relative_path"] = value["artifacts"][0]["relative_path"].clone();
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("duplicate artifact path"));
    }

    #[test]
    fn required_artifact_roles_cannot_be_omitted() {
        let mut value = fixture_value();
        value["artifacts"].as_array_mut().unwrap().retain(|artifact| {
            artifact["role"] != json!("calibration_certificate")
        });
        value["expected"]["artifact_count"] = json!(8);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("required artifact role") || error.contains("artifact count mismatch"));
    }

    #[test]
    fn instrument_and_acquisition_timebases_must_match() {
        let mut value = fixture_value();
        value["instrument"]["timebase_reference"] = json!("OTHER-CLOCK");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("sample-clock references disagree"));
    }

    #[test]
    fn calibration_must_cover_full_acquisition() {
        let mut value = fixture_value();
        value["acquisition"]["end_unix_ms"] = json!(1796000000000_u64);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("calibration validity interval"));
    }

    #[test]
    fn device_identity_artifact_must_match_receipt() {
        let mut value = fixture_value();
        value["device"]["physical_device_serial"] = json!("OTHER-SERIAL");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("device identity artifact"));
    }

    #[test]
    fn protocol_event_order_must_match_receipt() {
        let mut value = fixture_value();
        value["events"][0]["event_id"] = json!("WRONG-EVENT");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("protocol event order"));
    }

    #[test]
    fn first_event_must_not_claim_inter_call_readiness() {
        let mut value = fixture_value();
        value["events"][0]["readiness_probe_profile_id"] = json!("PROBE-1");
        value["events"][0]["readiness_probe_artifact_id"] = json!("probe-call-02");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("first event"));
    }

    #[test]
    fn later_events_require_readiness_probe_artifacts() {
        let mut value = fixture_value();
        value["events"][1]["readiness_probe_profile_id"] = Value::Null;
        value["events"][1]["readiness_probe_artifact_id"] = Value::Null;
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("readiness_probe_profile_id") || error.contains("strict data contract"));
    }

    #[test]
    fn event_indexes_must_be_contiguous_and_serialized_once() {
        let mut value = fixture_value();
        value["events"][1]["event_index"] = json!(3);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("contiguous"));
    }

    #[test]
    fn duplicate_event_ids_are_rejected() {
        let mut value = fixture_value();
        value["events"][1]["event_id"] = json!("CALL-01");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("duplicate event ID"));
    }

    #[test]
    fn load_profile_links_must_reference_profile_artifacts() {
        let mut value = fixture_value();
        value["events"][0]["load_profile_artifact_id"] = json!("device-identity");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("load profile link references the wrong role"));
    }

    #[test]
    fn raw_trace_row_count_is_reproducible() {
        let mut value = fixture_value();
        value["trace"]["expected_row_count"] = json!(5);
        value["expected"]["raw_trace_rows"] = json!(5);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("raw trace row count mismatch"));
    }

    #[test]
    fn open_circuit_source_current_cannot_be_waived_by_manifest_edit() {
        let mut value = fixture_value();
        value["protocol"]["standby_mode"] = json!("regulated_voltage_hold");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("protocol artifact does not match manifest field standby_mode"));
    }
}
