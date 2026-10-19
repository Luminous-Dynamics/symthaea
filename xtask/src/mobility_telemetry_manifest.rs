//! Cross-vehicle measurement manifest contract and fail-closed claim eligibility.
//!
//! A valid manifest describes how observations are collected. It does not
//! establish sensor accuracy or physical qualification. Recorded/closed
//! sessions additionally verify every artifact's relative path, byte length
//! and SHA-256 before emitting a report.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

const SCHEMA_VERSION: &str = "0.1.0";
const EVIDENCE_CLASSES: &[&str] = &[
    "DesignOnly",
    "Synthetic",
    "Simulated",
    "InstrumentedPhysical",
    "FieldObserved",
];
const PHYSICAL_EVIDENCE: &[&str] = &["InstrumentedPhysical", "FieldObserved"];
// Do not infer synchronization verification from a free-form label containing "verified".
const VERIFIED_SYNC_METHODS: &[&str] = &[
    "gnss_pps_verified",
    "hardware_timestamp_verified",
    "ptp_verified",
    "shared_clock_verified",
];
const MAX_PHYSICAL_ALIGNMENT_UNCERTAINTY_NS: u64 = 2_000_000;
const SESSION_STATES: &[&str] = &["planned", "recorded", "closed"];
const CALIBRATION_STATES: &[&str] = &[
    "planned",
    "uncalibrated",
    "calibrated",
    "expired",
    "not_applicable",
    "unknown",
];
const RIGHTS_STATES: &[&str] = &[
    "not_applicable_design_only",
    "pending",
    "project_owned",
    "explicit_permission",
    "cc_by_4_0",
    "other_documented_license",
];
const KNOWN_UNITS: &[&str] = &[
    "m/s", "m", "deg", "rad", "rad/s", "m/s^2", "N", "N*m", "V", "A",
    "W", "J", "Wh", "kg", "kg/s", "s", "ns", "degC", "kPa", "Pa",
];
const ARTIFACT_KINDS: &[&str] = &[
    "raw_sensor_log",
    "raw_bus_capture",
    "measurement_protocol",
    "calibration_record",
    "rights_receipt",
    "collection_authorization",
    "event_log",
    "metadata",
];

const CLAIM_REQUIREMENTS: &[(&str, &[&str])] = &[
    (
        "ride_energy",
        &[
            "dc_bus.voltage_v",
            "dc_bus.current_a",
            "gnss.speed_mps",
            "gnss.latitude_deg",
            "gnss.longitude_deg",
        ],
    ),
    (
        "longitudinal_dynamics",
        &[
            "gnss.speed_mps",
            "imu.accel_x_mps2",
            "wheel.front_speed_rads",
            "wheel.rear_speed_rads",
        ],
    ),
    (
        "tire_health",
        &[
            "tire.front_pressure_kpa",
            "tire.rear_pressure_kpa",
            "tire.front_temperature_c",
            "tire.rear_temperature_c",
        ],
    ),
    (
        "motor_map",
        &[
            "dyno.motor_torque_nm",
            "dyno.motor_speed_rads",
            "dc_bus.voltage_v",
            "dc_bus.current_a",
            "motor.temperature_c",
            "inverter.temperature_c",
        ],
    ),
    (
        "tire_force_characterization",
        &[
            "rig.normal_load_n",
            "rig.longitudinal_force_n",
            "rig.lateral_force_n",
            "rig.slip_ratio",
            "rig.slip_angle_rad",
            "rig.camber_angle_rad",
        ],
    ),
    (
        "ice_fuel_energy",
        &[
            "ice.fuel_mass_flow_kg_s",
            "ice.engine_speed_rads",
            "dyno.wheel_torque_nm",
            "dyno.wheel_speed_rads",
        ],
    ),
];

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Vehicle {
    vehicle_id: String,
    vehicle_class: String,
    propulsion: String,
    configuration_digest: String,
    mass_kg: Option<f64>,
    payload_kg: Option<f64>,
    tire_front: Option<TireContext>,
    tire_rear: Option<TireContext>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TireContext {
    make_model: String,
    size: String,
    pressure_kpa: Option<f64>,
    tread_state: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Timebase {
    master_clock: String,
    sync_method: String,
    max_alignment_uncertainty_ns: u64,
    gnss_utc_anchor_required: bool,
    gap_policy: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DataRights {
    rights_state: String,
    owner_or_rightsholder: String,
    permission_receipt_id: Option<String>,
    source_reference: Option<String>,
    redistribution_status: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Sensor {
    sensor_id: String,
    sensor_kind: String,
    model: String,
    calibration_state: String,
    calibration_receipt_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Channel {
    channel_id: String,
    sensor_id: String,
    quantity: String,
    unit: String,
    sample_rate_hz: f64,
    clock_domain: String,
    sign_convention: String,
    axis_or_frame: String,
    source_mode: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Artifact {
    relative_path: String,
    kind: String,
    byte_len: u64,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Manifest {
    manifest_id: String,
    schema_version: String,
    session_state: String,
    evidence_class: String,
    claim_ceiling: String,
    vehicle: Vehicle,
    timebase: Timebase,
    data_rights: DataRights,
    sensors: Vec<Sensor>,
    channels: Vec<Channel>,
    claim_targets: Vec<String>,
    artifacts: Vec<Artifact>,
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ClaimEligibility {
    claim_target: String,
    physical_eligible: bool,
    reasons: Vec<String>,
    missing_channels: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_version: String,
    manifest_id: String,
    manifest_sha256: String,
    session_state: String,
    evidence_class: String,
    structural_validation_pass: bool,
    artifact_verification: String,
    requested_claim_count: usize,
    physically_eligible_claim_count: usize,
    claims: Vec<ClaimEligibility>,
    claim_ceiling: String,
}

fn valid_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn known_claim(claim: &str) -> bool {
    CLAIM_REQUIREMENTS.iter().any(|(name, _)| *name == claim)
}

fn required_channels(claim: &str) -> Option<&'static [&'static str]> {
    CLAIM_REQUIREMENTS
        .iter()
        .find_map(|(name, channels)| (*name == claim).then_some(*channels))
}

fn valid_relative_path(path: &str) -> bool {
    // Reject both separators when interpreting the manifest on any host OS.
    // Otherwise a Windows-style path could look like a single filename on Unix.
    if path.trim().is_empty() || path.contains('\\') {
        return false;
    }
    let parsed = Path::new(path);
    !parsed.is_absolute()
        && parsed.components().all(|component| matches!(component, Component::Normal(_)))
}

fn validate_structure(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != SCHEMA_VERSION {
        bail!("unsupported mobility telemetry schema version: {}", manifest.schema_version);
    }
    if !SESSION_STATES.contains(&manifest.session_state.as_str()) {
        bail!("unsupported session_state: {}", manifest.session_state);
    }
    if !EVIDENCE_CLASSES.contains(&manifest.evidence_class.as_str()) {
        bail!("unsupported evidence_class: {}", manifest.evidence_class);
    }
    if manifest.manifest_id.trim().is_empty() || manifest.claim_ceiling.trim().is_empty() {
        bail!("manifest ID and claim ceiling are required");
    }
    if manifest.vehicle.vehicle_id.trim().is_empty()
        || manifest.vehicle.vehicle_class.trim().is_empty()
        || manifest.vehicle.propulsion.trim().is_empty()
    {
        bail!("vehicle ID, class and propulsion are required");
    }
    if !valid_sha256(&manifest.vehicle.configuration_digest) {
        bail!("vehicle configuration_digest must be a sha256-prefixed 64-hex digest");
    }
    for (name, value) in [("vehicle mass", manifest.vehicle.mass_kg), ("payload", manifest.vehicle.payload_kg)] {
        if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
            bail!("{name} must be finite and non-negative when present");
        }
    }
    for tire in [manifest.vehicle.tire_front.as_ref(), manifest.vehicle.tire_rear.as_ref()]
        .into_iter()
        .flatten()
    {
        if tire.make_model.trim().is_empty() || tire.size.trim().is_empty() || tire.tread_state.trim().is_empty() {
            bail!("tire identity and state fields may not be empty");
        }
        if tire.pressure_kpa.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            bail!("tire pressure must be finite and positive when present");
        }
    }
    if manifest.timebase.master_clock != "logger_monotonic_ns" {
        bail!("v0.1 requires logger_monotonic_ns as the master timebase");
    }
    if manifest.timebase.sync_method.trim().is_empty()
        || manifest.timebase.gap_policy.trim().is_empty()
        || manifest.timebase.max_alignment_uncertainty_ns == 0
    {
        bail!("clock synchronization method, nonzero uncertainty and explicit gap policy are required");
    }
    if !RIGHTS_STATES.contains(&manifest.data_rights.rights_state.as_str()) {
        bail!("unsupported rights_state: {}", manifest.data_rights.rights_state);
    }
    if manifest.data_rights.owner_or_rightsholder.trim().is_empty()
        || manifest.data_rights.redistribution_status.trim().is_empty()
    {
        bail!("rights holder and redistribution status are required");
    }
    if manifest.sensors.is_empty() || manifest.channels.is_empty() {
        bail!("manifest requires at least one sensor and one channel");
    }

    let mut sensor_ids = BTreeSet::new();
    for sensor in &manifest.sensors {
        if sensor.sensor_id.trim().is_empty() || !sensor_ids.insert(sensor.sensor_id.as_str()) {
            bail!("empty or duplicate sensor ID: {}", sensor.sensor_id);
        }
        if sensor.sensor_kind.trim().is_empty() || sensor.model.trim().is_empty() {
            bail!("sensor {} needs kind and model", sensor.sensor_id);
        }
        if !CALIBRATION_STATES.contains(&sensor.calibration_state.as_str()) {
            bail!("sensor {} has unsupported calibration state", sensor.sensor_id);
        }
        if sensor.calibration_state == "calibrated"
            && sensor.calibration_receipt_id.as_deref().is_none_or(str::is_empty)
        {
            bail!("sensor {} claims calibrated without a calibration receipt", sensor.sensor_id);
        }
    }

    let mut channel_ids = BTreeSet::new();
    for channel in &manifest.channels {
        if channel.channel_id.trim().is_empty() || !channel_ids.insert(channel.channel_id.as_str()) {
            bail!("empty or duplicate channel ID: {}", channel.channel_id);
        }
        if !sensor_ids.contains(channel.sensor_id.as_str()) {
            bail!("channel {} references unknown sensor {}", channel.channel_id, channel.sensor_id);
        }
        if channel.quantity.trim().is_empty()
            || channel.unit.trim().is_empty()
            || !KNOWN_UNITS.contains(&channel.unit.as_str())
            || channel.sign_convention.trim().is_empty()
            || channel.axis_or_frame.trim().is_empty()
            || channel.source_mode.trim().is_empty()
        {
            bail!("channel {} needs quantity, unit, sign, frame and source mode", channel.channel_id);
        }
        if !channel.sample_rate_hz.is_finite() || channel.sample_rate_hz <= 0.0 {
            bail!("channel {} sample_rate_hz must be finite and positive", channel.channel_id);
        }
        if !["logger_monotonic_ns", "sensor_ticks", "gnss_utc"].contains(&channel.clock_domain.as_str()) {
            bail!("channel {} has unsupported clock domain", channel.channel_id);
        }
    }

    if manifest.claim_targets.is_empty() {
        bail!("at least one claim target must be declared");
    }
    let mut claims = BTreeSet::new();
    for claim in &manifest.claim_targets {
        if !known_claim(claim) {
            bail!("unknown claim target: {claim}");
        }
        if !claims.insert(claim.as_str()) {
            bail!("duplicate claim target: {claim}");
        }
    }

    let mut artifact_paths = BTreeSet::new();
    for artifact in &manifest.artifacts {
        if !valid_relative_path(&artifact.relative_path) {
            bail!("artifact path must be a safe relative path: {}", artifact.relative_path);
        }
        if !artifact_paths.insert(artifact.relative_path.as_str()) {
            bail!("duplicate artifact path: {}", artifact.relative_path);
        }
        if !ARTIFACT_KINDS.contains(&artifact.kind.as_str()) {
            bail!("unsupported artifact kind {} for {}", artifact.kind, artifact.relative_path);
        }
        if artifact.byte_len == 0 || !valid_sha256(&artifact.sha256) {
            bail!("artifact {} needs nonzero byte_len and sha256-prefixed digest", artifact.relative_path);
        }
    }

    if manifest.session_state != "planned" && manifest.artifacts.is_empty() {
        bail!("recorded or closed sessions require at least one content-addressed artifact");
    }
    if manifest.session_state != "planned" && manifest.evidence_class == "DesignOnly" {
        bail!("recorded sessions cannot use DesignOnly evidence class");
    }
    Ok(())
}

fn verify_artifacts(manifest_path: &Path, artifacts: &[Artifact]) -> Result<()> {
    let manifest_dir = manifest_path
        .parent()
        .context("manifest path has no parent directory")?
        .canonicalize()
        .with_context(|| format!("canonicalizing manifest directory for {}", manifest_path.display()))?;

    if artifacts.is_empty() {
        bail!("no artifacts to verify");
    }
    for artifact in artifacts {
        if !valid_relative_path(&artifact.relative_path) {
            bail!("unsafe artifact path: {}", artifact.relative_path);
        }
        let candidate = manifest_dir.join(&artifact.relative_path);
        let canonical = candidate
            .canonicalize()
            .with_context(|| format!("resolving artifact {}", artifact.relative_path))?;
        if !canonical.starts_with(&manifest_dir) {
            bail!("artifact escapes manifest directory: {}", artifact.relative_path);
        }
        let raw = fs::read(&canonical)
            .with_context(|| format!("reading artifact {}", artifact.relative_path))?;
        if raw.len() as u64 != artifact.byte_len {
            bail!("artifact byte length mismatch for {}", artifact.relative_path);
        }
        let digest = format!("sha256:{:x}", Sha256::digest(&raw));
        if digest != artifact.sha256 {
            bail!("artifact SHA-256 mismatch for {}", artifact.relative_path);
        }
    }
    Ok(())
}

fn claim_eligibility(manifest: &Manifest, claim: &str) -> ClaimEligibility {
    let mut reasons = Vec::new();
    let required = required_channels(claim).unwrap_or(&[]);
    let known_channels: BTreeSet<&str> = manifest.channels.iter().map(|channel| channel.channel_id.as_str()).collect();
    let missing_channels: Vec<String> = required
        .iter()
        .filter(|channel_id| !known_channels.contains(**channel_id))
        .map(|channel_id| (*channel_id).to_owned())
        .collect();

    if !PHYSICAL_EVIDENCE.contains(&manifest.evidence_class.as_str()) {
        reasons.push("evidence_not_physical_measurement".into());
    }
    if manifest.session_state != "closed" {
        reasons.push("session_not_closed".into());
    }
    if !["project_owned", "explicit_permission", "cc_by_4_0", "other_documented_license"]
        .contains(&manifest.data_rights.rights_state.as_str())
    {
        reasons.push("data_rights_not_cleared_for_physical_claim".into());
    }
    if manifest.data_rights.permission_receipt_id.as_deref().is_none_or(str::is_empty) {
        reasons.push("missing_rights_or_collection_receipt".into());
    }
    if !VERIFIED_SYNC_METHODS.contains(&manifest.timebase.sync_method.as_str()) {
        reasons.push("clock_sync_method_not_verified".into());
    }
    if !manifest.artifacts.iter().any(|artifact| artifact.kind == "clock_sync_record") {
        reasons.push("missing_clock_sync_record_artifact".into());
    }
    if manifest.timebase.max_alignment_uncertainty_ns > MAX_PHYSICAL_ALIGNMENT_UNCERTAINTY_NS {
        reasons.push("clock_alignment_uncertainty_exceeds_v0_1_limit".into());
    }
    if !manifest.artifacts.iter().any(|artifact| artifact.kind == "raw_sensor_log") {
        reasons.push("missing_raw_sensor_log_artifact".into());
    }
    if manifest.vehicle.mass_kg.is_none_or(|mass| !mass.is_finite() || mass <= 0.0) {
        reasons.push("missing_measured_vehicle_mass".into());
    }
    if manifest.vehicle.payload_kg.is_none_or(|payload| !payload.is_finite() || payload < 0.0) {
        reasons.push("missing_declared_payload".into());
    }
    if manifest.vehicle.configuration_digest == format!("sha256:{}", "0".repeat(64)) {
        reasons.push("placeholder_vehicle_configuration_digest".into());
    }
    for (position, tire) in [
        ("front", manifest.vehicle.tire_front.as_ref()),
        ("rear", manifest.vehicle.tire_rear.as_ref()),
    ] {
        match tire {
            Some(tire)
                if !["", "TBD", "not_recorded"].contains(&tire.make_model.as_str())
                    && !["", "TBD", "not_recorded"].contains(&tire.size.as_str())
                    && tire.pressure_kpa.is_some_and(|pressure| pressure.is_finite() && pressure > 0.0)
                    && !["", "not_recorded"].contains(&tire.tread_state.as_str()) => {}
            _ => reasons.push(format!("incomplete_{position}_tire_setup")),
        }
    }
    if !manifest.artifacts.iter().any(|artifact| artifact.kind == "measurement_protocol") {
        reasons.push("missing_measurement_protocol_artifact".into());
    }
    if !manifest.artifacts.iter().any(|artifact| artifact.kind == "calibration_record") {
        reasons.push("missing_calibration_record_artifact".into());
    }
    if ["explicit_permission", "cc_by_4_0", "other_documented_license"]
        .contains(&manifest.data_rights.rights_state.as_str())
        && !manifest.artifacts.iter().any(|artifact| artifact.kind == "rights_receipt")
    {
        reasons.push("missing_rights_receipt_artifact".into());
    }
    if manifest.data_rights.rights_state == "project_owned"
        && !manifest.artifacts.iter().any(|artifact| artifact.kind == "collection_authorization")
    {
        reasons.push("missing_collection_authorization_artifact".into());
    }
    if !missing_channels.is_empty() {
        reasons.push("claim_required_channels_missing".into());
    }

    let sensors: std::collections::BTreeMap<&str, &Sensor> = manifest
        .sensors
        .iter()
        .map(|sensor| (sensor.sensor_id.as_str(), sensor))
        .collect();
    for channel_id in required.iter().filter(|channel_id| known_channels.contains(**channel_id)) {
        if let Some(channel) = manifest.channels.iter().find(|channel| channel.channel_id == **channel_id) {
            match sensors.get(channel.sensor_id.as_str()) {
                Some(sensor)
                    if sensor.calibration_state == "calibrated"
                        && sensor.calibration_receipt_id.as_deref().is_some_and(|id| !id.is_empty()) => {}
                Some(sensor) => reasons.push(format!("sensor_not_calibrated:{}", sensor.sensor_id)),
                None => reasons.push(format!("channel_sensor_missing:{}", channel.channel_id)),
            }
        }
    }

    reasons.sort();
    reasons.dedup();
    ClaimEligibility {
        claim_target: claim.to_owned(),
        physical_eligible: reasons.is_empty(),
        reasons,
        missing_channels,
    }
}

pub fn run(input_path: &Path, output_path: Option<&Path>) -> Result<()> {
    let raw = fs::read(input_path)
        .with_context(|| format!("reading mobility telemetry manifest {}", input_path.display()))?;
    let manifest_sha256 = format!("sha256:{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing mobility telemetry manifest {}", input_path.display()))?;
    validate_structure(&manifest)?;

    let artifact_verification = if manifest.session_state == "planned" {
        "not_applicable_planned_session".to_owned()
    } else {
        verify_artifacts(input_path, &manifest.artifacts)?;
        "all_declared_artifacts_verified".to_owned()
    };

    let claims: Vec<ClaimEligibility> = manifest
        .claim_targets
        .iter()
        .map(|claim| claim_eligibility(&manifest, claim))
        .collect();
    let physically_eligible_claim_count = claims.iter().filter(|claim| claim.physical_eligible).count();
    let report = Report {
        report_version: SCHEMA_VERSION.to_owned(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        session_state: manifest.session_state,
        evidence_class: manifest.evidence_class,
        structural_validation_pass: true,
        artifact_verification,
        requested_claim_count: claims.len(),
        physically_eligible_claim_count,
        claims,
        claim_ceiling: "Manifest structure and optional file integrity only. Physical eligibility is a gate, not proof of accuracy or a safety/roadworthiness/approval claim. Planned/synthetic manifests cannot establish vehicle or tire performance.".to_owned(),
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing mobility telemetry report {}", path.display()))?;
    }
    println!("{encoded}");
    Ok(())
}


#[derive(Debug, Deserialize)]
struct RawTraceRow {
    timestamp_ns: u64,
    channel_id: String,
    raw_value: String,
    sample_status: String,
    sequence: u64,
}

#[derive(Debug, Serialize)]
struct TraceChannelSummary {
    channel_id: String,
    sample_count: usize,
    valid_count: usize,
    invalid_or_saturated_count: usize,
    missing_value_count: usize,
    suspected_gap_count: usize,
    maximum_inter_sample_gap_ns: Option<u64>,
}

#[derive(Debug, Serialize)]
struct TraceReport {
    report_version: String,
    trace_sha256: String,
    trace_byte_len: usize,
    row_count: usize,
    valid_sample_count: usize,
    invalid_or_saturated_sample_count: usize,
    missing_value_count: usize,
    suspected_gap_count: usize,
    channels: Vec<TraceChannelSummary>,
    claim_ceiling: String,
}

#[derive(Debug, Default)]
struct MutableChannelSummary {
    sample_count: usize,
    valid_count: usize,
    invalid_or_saturated_count: usize,
    missing_value_count: usize,
    suspected_gap_count: usize,
    maximum_inter_sample_gap_ns: Option<u64>,
    last_timestamp_ns: Option<u64>,
}

fn validate_trace_bytes(raw: &[u8], manifest: &Manifest) -> Result<TraceReport> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(std::io::Cursor::new(raw));
    const HEADER: &[&str] = &[
        "timestamp_ns",
        "channel_id",
        "raw_value",
        "sample_status",
        "sequence",
    ];
    let headers = reader.headers().context("reading raw telemetry trace header")?;
    if headers.len() != HEADER.len()
        || !headers.iter().zip(HEADER.iter()).all(|(actual, expected)| actual == *expected)
    {
        bail!("raw telemetry trace header does not match v0.1 contract");
    }

    let channels_by_id: std::collections::BTreeMap<&str, &Channel> = manifest
        .channels
        .iter()
        .map(|channel| (channel.channel_id.as_str(), channel))
        .collect();
    let mut channel_stats = std::collections::BTreeMap::<String, MutableChannelSummary>::new();
    let mut previous_global_timestamp: Option<u64> = None;
    let mut row_count = 0usize;
    let mut valid_sample_count = 0usize;
    let mut invalid_or_saturated_sample_count = 0usize;
    let mut missing_value_count = 0usize;
    let mut suspected_gap_count = 0usize;

    for (index, result) in reader.deserialize::<RawTraceRow>().enumerate() {
        let row_number = index + 2;
        let row = result.with_context(|| format!("parsing raw telemetry row {row_number}"))?;
        if row.sequence != row_count as u64 {
            bail!(
                "row {row_number} sequence {} is not the expected contiguous sequence {}",
                row.sequence,
                row_count
            );
        }
        if previous_global_timestamp.is_some_and(|last| row.timestamp_ns < last) {
            bail!("row {row_number} timestamp regressed in logger-monotonic order");
        }
        previous_global_timestamp = Some(row.timestamp_ns);

        let channel = channels_by_id
            .get(row.channel_id.as_str())
            .with_context(|| format!("row {row_number} references undeclared channel {}", row.channel_id))?;
        if !["VALID", "INVALID", "SATURATED", "SENSOR_FAULT"].contains(&row.sample_status.as_str()) {
            bail!("row {row_number} has unsupported sample_status {}", row.sample_status);
        }
        let value = if row.raw_value.trim().is_empty() {
            None
        } else {
            let parsed = row.raw_value.trim().parse::<f64>()
                .with_context(|| format!("row {row_number} has nonnumeric raw_value"))?;
            if !parsed.is_finite() {
                bail!("row {row_number} has non-finite raw_value");
            }
            Some(parsed)
        };
        if row.sample_status == "VALID" && value.is_none() {
            bail!("row {row_number} marked VALID with an empty raw_value");
        }

        let summary = channel_stats.entry(row.channel_id.clone()).or_default();
        if summary.last_timestamp_ns.is_some_and(|last| row.timestamp_ns <= last) {
            bail!("row {row_number} repeats or regresses the timestamp for channel {}", row.channel_id);
        }
        if let Some(last) = summary.last_timestamp_ns {
            let gap_ns = row.timestamp_ns - last;
            summary.maximum_inter_sample_gap_ns = Some(
                summary.maximum_inter_sample_gap_ns.map_or(gap_ns, |prior| prior.max(gap_ns))
            );
            let expected_period_ns = (1_000_000_000.0 / channel.sample_rate_hz).round() as u64;
            // This is a diagnostic only, not a data deletion or acceptance rule.
            // Gaps remain visible via timestamps and explicit logger event records.
            if expected_period_ns > 0 && gap_ns as f64 > expected_period_ns as f64 * 2.5 {
                summary.suspected_gap_count += 1;
                suspected_gap_count += 1;
            }
        }
        summary.last_timestamp_ns = Some(row.timestamp_ns);
        summary.sample_count += 1;
        if row.sample_status == "VALID" {
            summary.valid_count += 1;
            valid_sample_count += 1;
        } else {
            summary.invalid_or_saturated_count += 1;
            invalid_or_saturated_sample_count += 1;
        }
        if value.is_none() {
            summary.missing_value_count += 1;
            missing_value_count += 1;
        }
        row_count += 1;
    }
    if row_count == 0 {
        bail!("raw telemetry trace contains no rows");
    }

    let channels = channel_stats.into_iter().map(|(channel_id, summary)| TraceChannelSummary {
        channel_id,
        sample_count: summary.sample_count,
        valid_count: summary.valid_count,
        invalid_or_saturated_count: summary.invalid_or_saturated_count,
        missing_value_count: summary.missing_value_count,
        suspected_gap_count: summary.suspected_gap_count,
        maximum_inter_sample_gap_ns: summary.maximum_inter_sample_gap_ns,
    }).collect();

    Ok(TraceReport {
        report_version: SCHEMA_VERSION.to_owned(),
        trace_sha256: format!("sha256:{:x}", Sha256::digest(raw)),
        trace_byte_len: raw.len(),
        row_count,
        valid_sample_count,
        invalid_or_saturated_sample_count,
        missing_value_count,
        suspected_gap_count,
        channels,
        claim_ceiling: "CSV structure, declared channel membership, monotonic timestamps, raw-value validity and likely sample-gap diagnostics only. No physical accuracy, energy-efficiency, tire-grip, safety, roadworthiness or approval claim.".into(),
    })
}


#[derive(Debug, Serialize)]
struct FaultMatrixCaseReport {
    case_id: String,
    mutation: String,
    expected: String,
    observed: String,
    passed: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct FaultMatrixReport {
    report_version: String,
    suite_id: String,
    status: String,
    source_commit: String,
    build_identity: String,
    matrix_sha256: String,
    manifest_sha256: String,
    case_count: usize,
    passed_count: usize,
    failed_count: usize,
    cases: Vec<FaultMatrixCaseReport>,
    claim_ceiling: String,
}

fn trace_case_outcome(raw: &[u8], manifest: &Manifest) -> (String, String) {
    match validate_trace_bytes(raw, manifest) {
        Ok(report) => (
            "accept".to_owned(),
            format!(
                "rows={}, valid={}, invalid_or_saturated={}, missing={}, suspected_gaps={}",
                report.row_count,
                report.valid_sample_count,
                report.invalid_or_saturated_sample_count,
                report.missing_value_count,
                report.suspected_gap_count,
            ),
        ),
        Err(error) => ("reject".to_owned(), error.to_string()),
    }
}

fn structure_case_outcome(manifest: &Manifest) -> (String, String) {
    match validate_structure(manifest) {
        Ok(()) => ("accept".to_owned(), "manifest structure accepted".to_owned()),
        Err(error) => ("reject".to_owned(), error.to_string()),
    }
}

fn evaluate_fault_case(mutation: &str, manifest: &Manifest) -> (String, String) {
    match mutation {
        "known_valid_multi_rate_fixture" => {
            let raw = include_bytes!("../../docs/research/data/mobility_telemetry_trace_v0_1.csv");
            match validate_trace_bytes(raw, manifest) {
                Ok(report)
                    if report.row_count == 5
                        && report.valid_sample_count == 4
                        && report.invalid_or_saturated_sample_count == 1
                        && report.missing_value_count == 1 =>
                {
                    ("accept".to_owned(), "fixture accepted and expected counters match".to_owned())
                }
                Ok(report) => (
                    "accept_with_wrong_summary".to_owned(),
                    format!("unexpected baseline counters: rows={}, valid={}, invalid={}, missing={}",
                        report.row_count, report.valid_sample_count,
                        report.invalid_or_saturated_sample_count, report.missing_value_count),
                ),
                Err(error) => ("reject".to_owned(), error.to_string()),
            }
        }
        "header_only_empty_trace" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n";
            trace_case_outcome(raw, manifest)
        }
        "reordered_header" => {
            let raw = b"timestamp_ns,raw_value,channel_id,sample_status,sequence\n1,1,gnss.speed_mps,VALID,0\n";
            trace_case_outcome(raw, manifest)
        }
        "undeclared_channel" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,not.declared,1,VALID,0\n";
            trace_case_outcome(raw, manifest)
        }
        "valid_status_with_empty_raw_value" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,,VALID,0\n";
            trace_case_outcome(raw, manifest)
        }
        "NaN_or_infinity_raw_value" => {
            let nan = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,NaN,VALID,0\n";
            let infinity = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,inf,VALID,0\n";
            let a = validate_trace_bytes(nan, manifest);
            let b = validate_trace_bytes(infinity, manifest);
            match (a, b) {
                (Err(a), Err(b)) => ("reject".to_owned(), format!("NaN rejected ({a}); infinity rejected ({b})")),
                (a, b) => ("accept".to_owned(), format!("non-finite input incorrectly accepted: NaN={}, infinity={}",
                    a.is_ok(), b.is_ok())),
            }
        }
        "unknown_sample_status" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,1,UNKNOWN,0\n";
            trace_case_outcome(raw, manifest)
        }
        "global_sequence_gap" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,1,VALID,0\n2,gnss.speed_mps,2,VALID,2\n";
            trace_case_outcome(raw, manifest)
        }
        "logger_timestamp_regression" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n3,gnss.speed_mps,1,VALID,0\n2,dc_bus.voltage_v,48,VALID,1\n";
            trace_case_outcome(raw, manifest)
        }
        "repeated_timestamp_same_channel" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n3,gnss.speed_mps,1,VALID,0\n3,gnss.speed_mps,2,VALID,1\n";
            trace_case_outcome(raw, manifest)
        }
        "same_timestamp_on_distinct_channels" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n100,gnss.speed_mps,1,VALID,0\n100,dc_bus.voltage_v,48,VALID,1\n";
            trace_case_outcome(raw, manifest)
        }
        "sensor_fault_with_missing_value" => {
            let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,dc_bus.current_a,,SENSOR_FAULT,0\n";
            match validate_trace_bytes(raw, manifest) {
                Ok(report)
                    if report.row_count == 1
                        && report.valid_sample_count == 0
                        && report.invalid_or_saturated_sample_count == 1
                        && report.missing_value_count == 1 =>
                {
                    ("accept_with_diagnostic".to_owned(), "fault row and missing-value counters preserved".to_owned())
                }
                Ok(report) => (
                    "accept_without_required_diagnostic".to_owned(),
                    format!("fault-row counters mismatch: rows={}, invalid={}, missing={}",
                        report.row_count, report.invalid_or_saturated_sample_count, report.missing_value_count),
                ),
                Err(error) => ("reject".to_owned(), error.to_string()),
            }
        }
        "inter_sample_gap_above_diagnostic_threshold" => {
            let channel = match manifest.channels.iter().find(|channel| channel.channel_id == "gnss.speed_mps") {
                Some(channel) => channel,
                None => return ("reject".to_owned(), "fixture lacks gnss.speed_mps".to_owned()),
            };
            let period_ns = (1_000_000_000.0 / channel.sample_rate_hz).round() as u64;
            if period_ns == 0 {
                return ("reject".to_owned(), "GNSS sample period rounded to zero".to_owned());
            }
            let raw = format!(
                "timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,0,VALID,0\n{},gnss.speed_mps,1,VALID,1\n{},gnss.speed_mps,2,VALID,2\n",
                1 + period_ns,
                1 + period_ns * 6
            );
            match validate_trace_bytes(raw.as_bytes(), manifest) {
                Ok(report) if report.row_count == 3 && report.suspected_gap_count == 1 =>
                    ("accept_with_diagnostic".to_owned(), "all three rows retained; one suspected gap reported".to_owned()),
                Ok(report) => ("accept_without_required_diagnostic".to_owned(),
                    format!("rows={}, suspected_gaps={}", report.row_count, report.suspected_gap_count)),
                Err(error) => ("reject".to_owned(), error.to_string()),
            }
        }
        "duplicate_sensor_id" => {
            let mut candidate = manifest.clone();
            candidate.sensors.push(candidate.sensors[0].clone());
            structure_case_outcome(&candidate)
        }
        "duplicate_channel_id" => {
            let mut candidate = manifest.clone();
            candidate.channels.push(candidate.channels[0].clone());
            structure_case_outcome(&candidate)
        }
        "unresolved_sensor_reference" => {
            let mut candidate = manifest.clone();
            candidate.channels[0].sensor_id = "sensor-does-not-exist".to_owned();
            structure_case_outcome(&candidate)
        }
        "zero_negative_or_nonfinite_sample_rate" => {
            let mut outcomes = Vec::new();
            for rate in [0.0, -1.0, f64::NAN] {
                let mut candidate = manifest.clone();
                candidate.channels[0].sample_rate_hz = rate;
                outcomes.push(validate_structure(&candidate).is_err());
            }
            if outcomes.iter().all(|rejected| *rejected) {
                ("reject".to_owned(), "zero, negative and NaN sample rates all rejected".to_owned())
            } else {
                ("accept".to_owned(), format!("sample-rate rejection outcomes: {outcomes:?}"))
            }
        }
        "calibrated_sensor_without_receipt" => {
            let mut candidate = manifest.clone();
            candidate.sensors[0].calibration_state = "calibrated".to_owned();
            candidate.sensors[0].calibration_receipt_id = None;
            structure_case_outcome(&candidate)
        }
        "absolute_or_traversal_artifact_path" => {
            let paths = ["../secrets.txt", "/tmp/raw.csv", "data/../../secret", r"raw\..\secret.csv"];
            if paths.iter().all(|path| !valid_relative_path(path)) {
                ("reject".to_owned(), "absolute and traversal paths rejected".to_owned())
            } else {
                ("accept".to_owned(), "at least one unsafe path was accepted".to_owned())
            }
        }
        "malformed_or_unprefixed_sha256" => {
            let unprefixed = "a".repeat(64);
            let invalid_hex = format!("sha256:{}", "g".repeat(64));
            if !valid_sha256(&unprefixed) && !valid_sha256(&invalid_hex) {
                ("reject".to_owned(), "unprefixed and invalid-hex digests rejected".to_owned())
            } else {
                ("accept".to_owned(), "at least one malformed digest was accepted".to_owned())
            }
        }
        "claim_required_channel_removed" => {
            let mut candidate = manifest.clone();
            candidate.channels.retain(|channel| channel.channel_id != "dc_bus.current_a");
            match validate_structure(&candidate) {
                Err(error) => ("reject".to_owned(), error.to_string()),
                Ok(()) => {
                    let claim = claim_eligibility(&candidate, "ride_energy");
                    if !claim.physical_eligible
                        && claim.missing_channels.iter().any(|channel| channel == "dc_bus.current_a")
                        && claim.reasons.iter().any(|reason| reason == "claim_required_channels_missing") {
                        ("structurally_valid_but_ineligible".to_owned(),
                         "only ride_energy is blocked by its missing required channel".to_owned())
                    } else {
                        ("eligible_or_unexplained".to_owned(), format!("eligibility report: {claim:?}"))
                    }
                }
            }
        }
        "physical_claim_with_uncalibrated_sensor" => {
            let mut candidate = manifest.clone();
            candidate.session_state = "closed".to_owned();
            candidate.evidence_class = "InstrumentedPhysical".to_owned();
            candidate.data_rights.rights_state = "project_owned".to_owned();
            candidate.data_rights.permission_receipt_id = Some("collection-approval-001".to_owned());
            candidate.timebase.sync_method = "gnss_pps_verified".to_owned();
            candidate.artifacts = vec![
                Artifact { relative_path:"raw/session.csv".into(),kind:"raw_sensor_log".into(),byte_len:1,sha256:format!("sha256:{}", "a".repeat(64)) },
                Artifact { relative_path:"protocol/test.json".into(),kind:"measurement_protocol".into(),byte_len:1,sha256:format!("sha256:{}", "b".repeat(64)) },
                Artifact { relative_path:"calibration/cal.json".into(),kind:"calibration_record".into(),byte_len:1,sha256:format!("sha256:{}", "c".repeat(64)) },
                Artifact { relative_path:"authorization/collection.json".into(),kind:"collection_authorization".into(),byte_len:1,sha256:format!("sha256:{}", "d".repeat(64)) },
            ];
            let claim = claim_eligibility(&candidate, "ride_energy");
            if !claim.physical_eligible
                && claim.reasons.iter().any(|reason| reason == "sensor_not_calibrated:dc-voltage-01")
                && claim.reasons.iter().any(|reason| reason == "sensor_not_calibrated:dc-current-01") {
                ("ineligible".to_owned(), "both uncalibrated electrical sensors block physical energy claim".to_owned())
            } else {
                ("eligible_or_unexplained".to_owned(), format!("eligibility report: {claim:?}"))
            }
        }
        "planned_or_synthetic_fixture_claimed_as_physical" => {
            if manifest.claim_targets.iter().all(|claim| !claim_eligibility(manifest, claim).physical_eligible) {
                ("ineligible".to_owned(), "planned DesignOnly fixture cannot support physical claims".to_owned())
            } else {
                ("eligible".to_owned(), "a planned fixture unexpectedly supports a physical claim".to_owned())
            }
        }
        _ => ("unimplemented".to_owned(), format!("unknown fault-matrix mutation: {mutation}")),
    }
}

pub fn run_benchmark(
    manifest_path: &Path,
    matrix_path: &Path,
    source_commit: &str,
    output_path: Option<&Path>,
) -> Result<()> {
    let valid_commit_length = source_commit.len() == 40 || source_commit.len() == 64;
    if !valid_commit_length || !source_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("--source-commit must be a full 40- or 64-character hexadecimal commit ID");
    }
    let manifest_raw = fs::read(manifest_path)
        .with_context(|| format!("reading telemetry manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_raw)
        .with_context(|| format!("parsing telemetry manifest {}", manifest_path.display()))?;
    validate_structure(&manifest)?;

    let matrix_raw = fs::read(matrix_path)
        .with_context(|| format!("reading telemetry fault matrix {}", matrix_path.display()))?;
    let matrix: serde_json::Value = serde_json::from_slice(&matrix_raw)
        .with_context(|| format!("parsing telemetry fault matrix {}", matrix_path.display()))?;
    let suite_id = matrix.get("suite_id").and_then(serde_json::Value::as_str)
        .context("fault matrix has no string suite_id")?.to_owned();
    let cases = matrix.get("cases").and_then(serde_json::Value::as_array)
        .context("fault matrix has no cases array")?;
    if cases.is_empty() {
        bail!("fault matrix contains no cases");
    }

    let mut seen_ids = BTreeSet::<String>::new();
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let case_id = case.get("id").and_then(serde_json::Value::as_str)
            .context("fault-matrix case has no string id")?.to_owned();
        let mutation = case.get("mutation").and_then(serde_json::Value::as_str)
            .context("fault-matrix case has no string mutation")?.to_owned();
        let expected = case.get("expected").and_then(serde_json::Value::as_str)
            .context("fault-matrix case has no string expected outcome")?.to_owned();
        if !seen_ids.insert(case_id.clone()) {
            bail!("duplicate fault-matrix case ID {case_id}");
        }

        let (observed, detail) = evaluate_fault_case(&mutation, &manifest);
        let passed = observed == expected;
        results.push(FaultMatrixCaseReport {
            case_id,
            mutation,
            expected,
            observed,
            passed,
            detail,
        });
    }

    let case_count = results.len();
    let passed_count = results.iter().filter(|case| case.passed).count();
    let failed_count = case_count - passed_count;
    let report = FaultMatrixReport {
        report_version: SCHEMA_VERSION.to_owned(),
        suite_id,
        status: if failed_count == 0 { "pass" } else { "fail" }.to_owned(),
        source_commit: source_commit.to_owned(),
        build_identity: format!(
            "xtask_package_version={};target={}-{};rustc_version={}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::ARCH,
            std::env::consts::OS,
            option_env!("RUSTC_VERSION").unwrap_or("not_embedded")
        ),
        matrix_sha256: format!("sha256:{:x}", Sha256::digest(&matrix_raw)),
        manifest_sha256: format!("sha256:{:x}", Sha256::digest(&manifest_raw)),
        case_count,
        passed_count,
        failed_count,
        cases: results,
        claim_ceiling: "The suite checks declared software-contract behavior against synthetic fixtures only. A passing report does not establish hardware reliability, metrological accuracy, physical vehicle/tire performance, roadworthiness or approval.".to_owned(),
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating benchmark report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing telemetry benchmark report {}", path.display()))?;
    }
    println!("{encoded}");
    if failed_count != 0 {
        bail!("mobility telemetry fault matrix failed {failed_count} of {case_count} cases");
    }
    Ok(())
}

pub fn run_trace(
    manifest_path: &Path,
    trace_path: &Path,
    output_path: Option<&Path>,
) -> Result<()> {
    let manifest_raw = fs::read(manifest_path)
        .with_context(|| format!("reading telemetry manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_raw)
        .with_context(|| format!("parsing telemetry manifest {}", manifest_path.display()))?;
    validate_structure(&manifest)?;

    let trace_raw = fs::read(trace_path)
        .with_context(|| format!("reading telemetry trace {}", trace_path.display()))?;
    if manifest.session_state != "planned" {
        verify_artifacts(manifest_path, &manifest.artifacts)?;
        let root = manifest_path.parent().context("manifest has no parent")?.canonicalize()?;
        let canonical_trace = trace_path.canonicalize()
            .with_context(|| format!("resolving telemetry trace {}", trace_path.display()))?;
        if !canonical_trace.starts_with(&root) {
            bail!("recorded telemetry trace must remain inside the manifest directory");
        }
        let relative = canonical_trace.strip_prefix(&root)
            .context("computing trace path relative to manifest directory")?
            .to_string_lossy()
            .replace('\\', "/");
        let declared = manifest.artifacts.iter().any(|artifact| {
            artifact.relative_path == relative && artifact.kind == "raw_sensor_log"
        });
        if !declared {
            bail!("trace path must match a declared raw_sensor_log artifact for recorded/closed sessions");
        }
    }

    let report = validate_trace_bytes(&trace_raw, &manifest)?;
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing telemetry trace report {}", path.display()))?;
    }
    println!("{encoded}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/mobility_telemetry_manifest_v0_1.json");

    fn manifest() -> Manifest {
        serde_json::from_str(FIXTURE).expect("planned manifest fixture parses")
    }

    #[test]
    fn planned_fixture_is_structurally_valid_but_not_physical_evidence() {
        let fixture = manifest();
        validate_structure(&fixture).expect("planned fixture must be structurally valid");
        assert_eq!(fixture.sensors.len(), 12);
        assert_eq!(fixture.channels.len(), 20);
        assert!(fixture.claim_targets.iter().all(|claim| !claim_eligibility(&fixture, claim).physical_eligible));
    }

    #[test]
    fn duplicate_sensor_ids_are_rejected() {
        let mut fixture = manifest();
        fixture.sensors.push(fixture.sensors[0].clone());
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn duplicate_channel_ids_are_rejected() {
        let mut fixture = manifest();
        fixture.channels.push(fixture.channels[0].clone());
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn unknown_channel_sensor_reference_is_rejected() {
        let mut fixture = manifest();
        fixture.channels[0].sensor_id = "sensor-does-not-exist".into();
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn non_finite_or_nonpositive_sample_rate_is_rejected() {
        let mut fixture = manifest();
        fixture.channels[0].sample_rate_hz = 0.0;
        assert!(validate_structure(&fixture).is_err());
        fixture.channels[0].sample_rate_hz = f64::NAN;
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn missing_claim_channels_make_only_that_claim_ineligible() {
        let mut fixture = manifest();
        fixture.channels.retain(|channel| channel.channel_id != "dc_bus.current_a");
        validate_structure(&fixture).expect("missing data coverage is not a malformed schema");
        let claim = claim_eligibility(&fixture, "ride_energy");
        assert!(!claim.physical_eligible);
        assert!(claim.missing_channels.contains(&"dc_bus.current_a".to_owned()));
        assert!(claim.reasons.contains(&"claim_required_channels_missing".to_owned()));
    }

    #[test]
    fn calibrated_sensor_requires_a_receipt() {
        let mut fixture = manifest();
        fixture.sensors[0].calibration_state = "calibrated".into();
        fixture.sensors[0].calibration_receipt_id = None;
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn artifact_paths_must_not_escape_the_manifest_directory() {
        assert!(!valid_relative_path("../secrets.txt"));
        assert!(!valid_relative_path("/tmp/raw.csv"));
        assert!(!valid_relative_path("data/../../secret"));
        assert!(!valid_relative_path(r"raw\..\secret.csv"));
        assert!(valid_relative_path("raw/session.csv"));
    }

    #[test]
    fn artifact_digest_must_be_sha256_prefixed_and_well_formed() {
        assert!(valid_sha256(&format!("sha256:{}", "a".repeat(64))));
        assert!(!valid_sha256("a".repeat(64).as_str()));
        assert!(!valid_sha256(&format!("sha256:{}", "g".repeat(64))));
    }

    #[test]
    fn uncalibrated_sensor_blocks_physical_claim_eligibility() {
        let mut fixture = manifest();
        fixture.session_state = "closed".into();
        fixture.evidence_class = "InstrumentedPhysical".into();
        fixture.data_rights.rights_state = "project_owned".into();
        fixture.data_rights.permission_receipt_id = Some("collection-approval-001".into());
        fixture.timebase.sync_method = "gnss_pps_verified".into();
        fixture.artifacts = vec![
            Artifact { relative_path:"raw/session.csv".into(),kind:"raw_sensor_log".into(),byte_len:1,sha256:format!("sha256:{}", "a".repeat(64)) },
            Artifact { relative_path:"protocol/test.json".into(),kind:"measurement_protocol".into(),byte_len:1,sha256:format!("sha256:{}", "b".repeat(64)) },
            Artifact { relative_path:"calibration/cal.json".into(),kind:"calibration_record".into(),byte_len:1,sha256:format!("sha256:{}", "c".repeat(64)) },
            Artifact { relative_path:"authorization/collection.json".into(),kind:"collection_authorization".into(),byte_len:1,sha256:format!("sha256:{}", "d".repeat(64)) },
        ];
        let claim = claim_eligibility(&fixture, "ride_energy");
        assert!(!claim.physical_eligible);
        assert!(claim.reasons.iter().any(|reason| reason == "sensor_not_calibrated:dc-voltage-01"));
        assert!(claim.reasons.iter().any(|reason| reason == "sensor_not_calibrated:dc-current-01"));
    }

    #[test]
    fn synthetic_trace_parses_without_claiming_physical_validity() {
        let fixture = manifest();
        let raw = include_bytes!("../../docs/research/data/mobility_telemetry_trace_v0_1.csv");
        let report = validate_trace_bytes(raw, &fixture).expect("synthetic trace parses");
        assert_eq!(report.row_count, 5);
        assert_eq!(report.valid_sample_count, 4);
        assert_eq!(report.invalid_or_saturated_sample_count, 1);
        assert_eq!(report.missing_value_count, 1);
        assert_eq!(report.trace_sha256.len(), 71);
    }

    #[test]
    fn valid_sample_may_not_have_an_empty_value() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,,VALID,0\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn unknown_channel_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,unknown.speed,1,VALID,0\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn sequence_gap_is_rejected_instead_of_hidden() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,1,VALID,0\n2,gnss.speed_mps,2,VALID,2\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn timestamp_regression_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n3,gnss.speed_mps,1,VALID,0\n2,dc_bus.voltage_v,2,VALID,1\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn repeated_timestamp_for_same_channel_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n3,gnss.speed_mps,1,VALID,0\n3,gnss.speed_mps,2,VALID,1\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn wrong_master_clock_is_rejected() {
        let mut fixture = manifest();
        fixture.timebase.master_clock = "wall_clock_guess".into();
        assert!(validate_structure(&fixture).is_err());
    }

    #[test]
    fn empty_trace_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn wrong_header_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,raw_value,channel_id,sample_status,sequence\n1,1,gnss.speed_mps,VALID,0\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn non_finite_raw_value_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,NaN,VALID,0\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
        let raw_inf = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,inf,VALID,0\n";
        assert!(validate_trace_bytes(raw_inf, &fixture).is_err());
    }

    #[test]
    fn unsupported_sample_status_is_rejected() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,1,UNKNOWN,0\n";
        assert!(validate_trace_bytes(raw, &fixture).is_err());
    }

    #[test]
    fn equal_timestamps_across_distinct_channels_are_accepted() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n100,gnss.speed_mps,1,VALID,0\n100,dc_bus.voltage_v,48,VALID,1\n";
        let report = validate_trace_bytes(raw, &fixture)
            .expect("same-clock simultaneous observations from different channels are valid");
        assert_eq!(report.row_count, 2);
        assert_eq!(report.valid_sample_count, 2);
        assert_eq!(report.suspected_gap_count, 0);
    }

    #[test]
    fn fault_rows_remain_explicit_and_missing() {
        let fixture = manifest();
        let raw = b"timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,dc_bus.current_a,,SENSOR_FAULT,0\n";
        let report = validate_trace_bytes(raw, &fixture)
            .expect("an explicit sensor fault is a valid trace row, not a valid measurement");
        assert_eq!(report.row_count, 1);
        assert_eq!(report.valid_sample_count, 0);
        assert_eq!(report.invalid_or_saturated_sample_count, 1);
        assert_eq!(report.missing_value_count, 1);
        assert_eq!(report.channels[0].sample_count, 1);
        assert_eq!(report.channels[0].missing_value_count, 1);
    }

    #[test]
    fn large_gap_is_reported_without_filling() {
        let fixture = manifest();
        let channel = fixture.channels.iter()
            .find(|channel| channel.channel_id == "gnss.speed_mps")
            .expect("fixture declares GNSS speed");
        let period_ns = (1_000_000_000.0 / channel.sample_rate_hz).round() as u64;
        assert!(period_ns > 0);
        let raw = format!(
            "timestamp_ns,channel_id,raw_value,sample_status,sequence\n1,gnss.speed_mps,0,VALID,0\n{},gnss.speed_mps,1,VALID,1\n{},gnss.speed_mps,2,VALID,2\n",
            1 + period_ns,
            1 + period_ns * 6
        );
        let report = validate_trace_bytes(raw.as_bytes(), &fixture)
            .expect("gaps are diagnosed rather than causing row deletion");
        assert_eq!(report.row_count, 3);
        assert_eq!(report.valid_sample_count, 3);
        assert_eq!(report.suspected_gap_count, 1);
        assert_eq!(report.channels[0].sample_count, 3);
        assert_eq!(report.channels[0].suspected_gap_count, 1);
    }

    #[test]
    fn frozen_fault_matrix_cases_match_expected_outcomes() {
        let fixture = manifest();
        let matrix: serde_json::Value = serde_json::from_str(include_str!(
            "../../docs/research/data/mobility_telemetry_fault_matrix_v0_1.json"
        )).expect("frozen telemetry fault matrix parses");
        let cases = matrix.get("cases").and_then(serde_json::Value::as_array)
            .expect("fault matrix has a case array");
        assert_eq!(cases.len(), 23, "matrix size changes require deliberate review");
        let mut ids = BTreeSet::new();
        for case in cases {
            let id = case.get("id").and_then(serde_json::Value::as_str)
                .expect("case has ID");
            let mutation = case.get("mutation").and_then(serde_json::Value::as_str)
                .expect("case has mutation");
            let expected = case.get("expected").and_then(serde_json::Value::as_str)
                .expect("case has expected outcome");
            assert!(ids.insert(id), "duplicate case ID {id}");
            let (observed, detail) = evaluate_fault_case(mutation, &fixture);
            assert_eq!(observed, expected, "fault case {id} ({mutation}): {detail}");
        }
    }
}
