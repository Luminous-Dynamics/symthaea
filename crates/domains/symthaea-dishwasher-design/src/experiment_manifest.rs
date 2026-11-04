//! Canonical physical-run evidence bundle validation for InstantClean.
//!
//! This validates identity, time, calibration, assay, trace and resource linkage
//! before data enters grouped validation or observer evaluation. It does not
//! authenticate archive bytes, verify signatures, prove instrument independence,
//! or establish assay fitness. SHA-256 strings are format-checked only.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::quality::{Endpoint, OutcomeObservation};
use crate::validation::{RunPartitionRecord, ValidationPartition};

pub const EVIDENCE_BUNDLE_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    ApplianceWallElectricity,
    UpstreamHotWaterEnergy,
    FreshWater,
    DetergentMass,
    RinseAidMass,
    ActiveOperatorTime,
}

impl ResourceKind {
    pub const fn required_unit(self) -> &'static str {
        match self {
            Self::ApplianceWallElectricity | Self::UpstreamHotWaterEnergy => "kWh",
            Self::FreshWater => "L",
            Self::DetergentMass | Self::RinseAidMass => "g",
            Self::ActiveOperatorTime => "s",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstrumentCalibration {
    pub calibration_id: String,
    pub instrument_id: String,
    pub valid_from_utc_s: u64,
    pub valid_until_utc_s: u64,
    pub certificate_uri: String,
    pub certificate_sha256: String,
    pub calibration_bundle_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorTraceArtifact {
    pub artifact_id: String,
    pub source_run_id: String,
    pub instrument_id: String,
    pub channel_id: String,
    pub metric_id: String,
    pub unit: String,
    pub start_utc_s: u64,
    pub end_utc_s: u64,
    pub sample_count: u64,
    pub sample_rate_hz: f64,
    pub calibration_id: String,
    pub archive_uri: String,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssayArtifact {
    pub record_id: String,
    pub observation: OutcomeObservation,
    pub instrument_id: String,
    pub calibration_id: String,
    pub assayed_at_utc_s: u64,
    pub archive_uri: String,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceMeasurement {
    /// Physical wash run whose resource increment this row measures.
    pub source_run_id: String,
    pub measurement_id: String,
    /// Declared physical scope of this resource increment (not proof of disjointness).
    pub measurement_boundary_id: String,
    pub kind: ResourceKind,
    pub value: f64,
    pub unit: String,
    pub start_utc_s: u64,
    pub end_utc_s: u64,
    pub instrument_id: Option<String>,
    pub calibration_id: Option<String>,
    pub method_id: String,
    pub evidence_uri: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalItemRecord {
    pub item_id: String,
    pub material: String,
    pub surface_finish: String,
    pub geometry: String,
    pub location_stratum: String,
    pub item_batch_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalRunEvidenceBundle {
    pub schema_version: u16,
    pub run: RunPartitionRecord,
    pub machine_model_id: String,
    pub programme_id: String,
    pub firmware_sha256: String,
    pub protocol_id: String,
    pub randomization_block_id: String,
    pub treatment_assignment_id: String,
    pub soil_age_at_start_s: u64,
    pub water_hardness_ppm_as_caco3: u16,
    pub inlet_temperature_millic: u32,
    pub ambient_temperature_millic: u32,
    pub ambient_relative_humidity_permille: u16,
    pub start_utc_s: u64,
    pub end_utc_s: u64,
    pub items: Vec<PhysicalItemRecord>,
    pub required_endpoints: Vec<Endpoint>,
    pub calibrations: Vec<InstrumentCalibration>,
    pub sensor_traces: Vec<SensorTraceArtifact>,
    pub assays: Vec<AssayArtifact>,
    pub resources: Vec<ResourceMeasurement>,
    pub archive_uri: String,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceBundleError {
    UnsupportedSchemaVersion(u16),
    InvalidIdentity(&'static str),
    InvalidDigest(&'static str),
    InvalidRunInterval,
    EmptyItemSet,
    DuplicateItem(String),
    EmptyEndpointSet,
    DuplicateEndpoint(Endpoint),
    EmptyRequiredResourceSet,
    DuplicateRequiredResource(ResourceKind),
    MissingRequiredResource(ResourceKind),
    MissingSensorTraces,
    DuplicateCalibrationId(String),
    DuplicateArtifactId(String),
    UnknownCalibration(String),
    CalibrationInstrumentMismatch { calibration_id: String, expected: String, observed: String },
    CalibrationNotValidAt { calibration_id: String, timestamp_utc_s: u64 },
    InvalidCalibrationInterval(String),
    InvalidTraceInterval(String),
    TraceOutsideRun(String),
    InvalidTraceSampling(String),
    TraceSamplingWindowMismatch(String),
    TraceRunMismatch { artifact_id: String, observed_run_id: String },
    InvalidAssayRun { record_id: String, observed_run_id: String },
    InvalidResourceRun { measurement_id: String, observed_run_id: String },
    UnknownAssayItem { record_id: String, item_id: String },
    UnexpectedAssayEndpoint { record_id: String, endpoint: Endpoint },
    DuplicateAssay { item_id: String, endpoint: Endpoint },
    MissingAssay { item_id: String, endpoint: Endpoint },
    AssayProtocolMismatch(String),
    AssayArchiveReferenceMismatch(String),
    AssayCalibrationDigestMismatch(String),
    InvalidAssayMeasurement(String),
    InvalidResourceInterval(String),
    ResourceOutsideRun(String),
    InvalidResourceValue(String),
    ResourceUnitMismatch { measurement_id: String, expected: String, observed: String },
    ResourceInstrumentCalibrationMismatch(String),
    ManualResourceNotAllowed(ResourceKind),
    ResourceEvidenceMissing(String),
    ResourceTotalOverflow(ResourceKind),
    OverlappingResourceBoundaryIntervals {
        kind: ResourceKind,
        measurement_boundary_id: String,
        first_measurement_id: String,
        second_measurement_id: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceBundleValidationReport {
    pub schema_version: u16,
    pub run_id: String,
    pub physical_item_count: usize,
    pub required_endpoint_count: usize,
    pub sensor_trace_count: usize,
    pub assay_record_count: usize,
    pub resource_measurement_count: usize,
    pub calibration_count: usize,
    pub resource_totals: Vec<(ResourceKind, f64, &'static str)>,
    pub partition: ValidationPartition,
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn validate_archive(id: &str, uri: &str, digest: &str, id_field: &'static str, digest_field: &'static str) -> Result<(), EvidenceBundleError> {
    if !non_empty(id) || !non_empty(uri) {
        return Err(EvidenceBundleError::InvalidIdentity(id_field));
    }
    if !valid_digest(digest) {
        return Err(EvidenceBundleError::InvalidDigest(digest_field));
    }
    Ok(())
}

fn calibration_for<'a>(
    calibrations: &'a HashMap<&str, &'a InstrumentCalibration>,
    calibration_id: &str,
) -> Result<&'a InstrumentCalibration, EvidenceBundleError> {
    calibrations.get(calibration_id).copied()
        .ok_or_else(|| EvidenceBundleError::UnknownCalibration(calibration_id.to_owned()))
}

fn validate_calibration<'a>(
    calibrations: &'a HashMap<&str, &'a InstrumentCalibration>,
    calibration_id: &str,
    instrument_id: &str,
    timestamp_utc_s: u64,
) -> Result<&'a InstrumentCalibration, EvidenceBundleError> {
    let calibration = calibration_for(calibrations, calibration_id)?;
    if calibration.instrument_id != instrument_id {
        return Err(EvidenceBundleError::CalibrationInstrumentMismatch {
            calibration_id: calibration_id.to_owned(),
            expected: calibration.instrument_id.clone(),
            observed: instrument_id.to_owned(),
        });
    }
    if timestamp_utc_s < calibration.valid_from_utc_s || timestamp_utc_s > calibration.valid_until_utc_s {
        return Err(EvidenceBundleError::CalibrationNotValidAt {
            calibration_id: calibration_id.to_owned(),
            timestamp_utc_s,
        });
    }
    Ok(calibration)
}

/// Validate one physical run. Every resource kind declared as required must
/// have an explicit row, including a measured zero. Resource totals are summed
/// by kind and retain their canonical units.
///
/// Manual records are allowed only for active operator time. A manual fresh
/// water volume must be represented by a calibrated measuring instrument.
pub fn validate_physical_run_bundle(
    bundle: &PhysicalRunEvidenceBundle,
    required_resources: &[ResourceKind],
) -> Result<EvidenceBundleValidationReport, EvidenceBundleError> {
    if bundle.schema_version != EVIDENCE_BUNDLE_SCHEMA_VERSION {
        return Err(EvidenceBundleError::UnsupportedSchemaVersion(bundle.schema_version));
    }
    for (field, value) in [
        ("run_id", bundle.run.run_id.as_str()),
        ("machine_unit_id", bundle.run.machine_unit_id.as_str()),
        ("test_session_id", bundle.run.test_session_id.as_str()),
        ("soil_batch_id", bundle.run.soil_batch_id.as_str()),
        ("detergent_batch_id", bundle.run.detergent_batch_id.as_str()),
        ("water_batch_id", bundle.run.water_batch_id.as_str()),
        ("operator_id", bundle.run.operator_id.as_str()),
        ("assay_lab_id", bundle.run.assay_lab_id.as_str()),
        ("test_site_id", bundle.run.test_site_id.as_str()),
        ("service_request_id", bundle.run.service_request_id.as_str()),
        ("compartment_id", bundle.run.compartment_id.as_str()),
        ("utility_topology_id", bundle.run.utility_topology_id.as_str()),
        ("machine_model_id", bundle.machine_model_id.as_str()),
        ("programme_id", bundle.programme_id.as_str()),
        ("protocol_id", bundle.protocol_id.as_str()),
        ("randomization_block_id", bundle.randomization_block_id.as_str()),
        ("treatment_assignment_id", bundle.treatment_assignment_id.as_str()),
        ("archive_uri", bundle.archive_uri.as_str()),
    ] {
        if !non_empty(value) {
            return Err(EvidenceBundleError::InvalidIdentity(field));
        }
    }
    if !valid_digest(&bundle.firmware_sha256) {
        return Err(EvidenceBundleError::InvalidDigest("firmware_sha256"));
    }
    if !valid_digest(&bundle.run.compartment_configuration_sha256) {
        return Err(EvidenceBundleError::InvalidDigest("compartment_configuration_sha256"));
    }
    if !valid_digest(&bundle.archive_sha256) {
        return Err(EvidenceBundleError::InvalidDigest("bundle_archive_sha256"));
    }
    if bundle.end_utc_s <= bundle.start_utc_s {
        return Err(EvidenceBundleError::InvalidRunInterval);
    }

    if bundle.items.is_empty() {
        return Err(EvidenceBundleError::EmptyItemSet);
    }
    if bundle.ambient_relative_humidity_permille > 1_000 {
        return Err(EvidenceBundleError::InvalidIdentity("ambient_relative_humidity_permille"));
    }
    let mut items = BTreeSet::new();
    for item in &bundle.items {
        for (field, value) in [
            ("item_id", item.item_id.as_str()),
            ("item.material", item.material.as_str()),
            ("item.surface_finish", item.surface_finish.as_str()),
            ("item.geometry", item.geometry.as_str()),
            ("item.location_stratum", item.location_stratum.as_str()),
            ("item.item_batch_id", item.item_batch_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(EvidenceBundleError::InvalidIdentity(field));
            }
        }
        if !items.insert(item.item_id.as_str()) {
            return Err(EvidenceBundleError::DuplicateItem(item.item_id.clone()));
        }
    }

    if bundle.required_endpoints.is_empty() {
        return Err(EvidenceBundleError::EmptyEndpointSet);
    }
    let mut endpoints = BTreeSet::new();
    for endpoint in &bundle.required_endpoints {
        if !endpoints.insert(*endpoint) {
            return Err(EvidenceBundleError::DuplicateEndpoint(*endpoint));
        }
    }

    if required_resources.is_empty() {
        return Err(EvidenceBundleError::EmptyRequiredResourceSet);
    }
    let mut required = BTreeSet::new();
    for kind in required_resources {
        if !required.insert(*kind) {
            return Err(EvidenceBundleError::DuplicateRequiredResource(*kind));
        }
    }

    let mut calibrations = HashMap::with_capacity(bundle.calibrations.len());
    for calibration in &bundle.calibrations {
        if !non_empty(&calibration.calibration_id)
            || !non_empty(&calibration.instrument_id)
            || !non_empty(&calibration.certificate_uri)
        {
            return Err(EvidenceBundleError::InvalidIdentity("calibration"));
        }
        if calibration.valid_until_utc_s < calibration.valid_from_utc_s {
            return Err(EvidenceBundleError::InvalidCalibrationInterval(calibration.calibration_id.clone()));
        }
        if !valid_digest(&calibration.certificate_sha256) {
            return Err(EvidenceBundleError::InvalidDigest("calibration_certificate"));
        }
        if !valid_digest(&calibration.calibration_bundle_sha256) {
            return Err(EvidenceBundleError::InvalidDigest("calibration_bundle"));
        }
        if calibrations.insert(calibration.calibration_id.as_str(), calibration).is_some() {
            return Err(EvidenceBundleError::DuplicateCalibrationId(calibration.calibration_id.clone()));
        }
    }

    if bundle.sensor_traces.is_empty() {
        return Err(EvidenceBundleError::MissingSensorTraces);
    }
    let mut artifact_ids = HashSet::new();
    for trace in &bundle.sensor_traces {
        validate_archive(&trace.artifact_id, &trace.archive_uri, &trace.archive_sha256, "sensor_trace", "sensor_trace_archive")?;
        if !artifact_ids.insert(trace.artifact_id.as_str()) {
            return Err(EvidenceBundleError::DuplicateArtifactId(trace.artifact_id.clone()));
        }
        if trace.source_run_id != bundle.run.run_id {
            return Err(EvidenceBundleError::TraceRunMismatch {
                artifact_id: trace.artifact_id.clone(),
                observed_run_id: trace.source_run_id.clone(),
            });
        }
        for (field, value) in [
            ("trace.instrument_id", trace.instrument_id.as_str()),
            ("trace.channel_id", trace.channel_id.as_str()),
            ("trace.metric_id", trace.metric_id.as_str()),
            ("trace.unit", trace.unit.as_str()),
        ] {
            if !non_empty(value) {
                return Err(EvidenceBundleError::InvalidIdentity(field));
            }
        }
        if trace.end_utc_s < trace.start_utc_s {
            return Err(EvidenceBundleError::InvalidTraceInterval(trace.artifact_id.clone()));
        }
        if trace.start_utc_s < bundle.start_utc_s || trace.end_utc_s > bundle.end_utc_s {
            return Err(EvidenceBundleError::TraceOutsideRun(trace.artifact_id.clone()));
        }
        // A regular time series needs at least two samples to establish a temporal span.
        // Single observations belong in a point-observation representation, not this trace type.
        if trace.sample_count < 2 || !trace.sample_rate_hz.is_finite() || trace.sample_rate_hz <= 0.0 {
            return Err(EvidenceBundleError::InvalidTraceSampling(trace.artifact_id.clone()));
        }
        // The schema stores interval endpoints at one-second resolution.
        // For a regularly sampled trace with inclusive first/last samples,
        // N samples span (N - 1) / f seconds; allow one second of timestamp
        // quantization but reject internally contradictory metadata.
        let expected_span_s = (trace.sample_count - 1) as f64 / trace.sample_rate_hz;
        let observed_span_s = (trace.end_utc_s - trace.start_utc_s) as f64;
        if !expected_span_s.is_finite() || (expected_span_s - observed_span_s).abs() > 1.0 {
            return Err(EvidenceBundleError::TraceSamplingWindowMismatch(trace.artifact_id.clone()));
        }
        validate_calibration(&calibrations, &trace.calibration_id, &trace.instrument_id, trace.start_utc_s)?;
        validate_calibration(&calibrations, &trace.calibration_id, &trace.instrument_id, trace.end_utc_s)?;
    }

    let mut assay_pairs = HashSet::new();
    for assay in &bundle.assays {
        validate_archive(&assay.record_id, &assay.archive_uri, &assay.archive_sha256, "assay_record", "assay_archive")?;
        if !artifact_ids.insert(assay.record_id.as_str()) {
            return Err(EvidenceBundleError::DuplicateArtifactId(assay.record_id.clone()));
        }
        if assay.observation.run_id != bundle.run.run_id {
            return Err(EvidenceBundleError::InvalidAssayRun {
                record_id: assay.record_id.clone(),
                observed_run_id: assay.observation.run_id.clone(),
            });
        }
        if !items.contains(assay.observation.item_id.as_str()) {
            return Err(EvidenceBundleError::UnknownAssayItem {
                record_id: assay.record_id.clone(),
                item_id: assay.observation.item_id.clone(),
            });
        }
        if !endpoints.contains(&assay.observation.endpoint) {
            return Err(EvidenceBundleError::UnexpectedAssayEndpoint {
                record_id: assay.record_id.clone(),
                endpoint: assay.observation.endpoint,
            });
        }
        if assay.observation.protocol_id != bundle.protocol_id {
            return Err(EvidenceBundleError::AssayProtocolMismatch(assay.record_id.clone()));
        }
        for (field, value) in [
            ("assay.criterion_id", assay.observation.criterion_id.as_str()),
            ("assay.metric_id", assay.observation.metric_id.as_str()),
            ("assay.method_id", assay.observation.method_id.as_str()),
            ("assay.unit", assay.observation.unit.as_str()),
        ] {
            if !non_empty(value) {
                return Err(EvidenceBundleError::InvalidIdentity(field));
            }
        }
        let measurement = assay.observation.measured_value;
        let uncertainty = assay.observation.expanded_uncertainty;
        if !measurement.is_finite()
            || measurement < 0.0
            || !uncertainty.is_finite()
            || uncertainty < 0.0
            || !(measurement - uncertainty).is_finite()
            || !(measurement + uncertainty).is_finite()
        {
            return Err(EvidenceBundleError::InvalidAssayMeasurement(assay.record_id.clone()));
        }
        if assay.observation.evidence_ref != assay.archive_uri {
            return Err(EvidenceBundleError::AssayArchiveReferenceMismatch(assay.record_id.clone()));
        }
        if !non_empty(&assay.instrument_id) || !non_empty(&assay.calibration_id) {
            return Err(EvidenceBundleError::InvalidIdentity("assay_instrument_calibration"));
        }
        let calibration = validate_calibration(
            &calibrations,
            &assay.calibration_id,
            &assay.instrument_id,
            assay.assayed_at_utc_s,
        )?;
        if calibration.calibration_bundle_sha256 != assay.observation.calibration_bundle_sha256 {
            return Err(EvidenceBundleError::AssayCalibrationDigestMismatch(assay.record_id.clone()));
        }
        if !matches!(assay.observation.evidence_kind, crate::quality::EvidenceKind::IndependentOutcomeAssay) {
            return Err(EvidenceBundleError::InvalidIdentity("assay_must_be_independent_outcome"));
        }
        let pair = (assay.observation.item_id.as_str(), assay.observation.endpoint);
        if !assay_pairs.insert(pair) {
            return Err(EvidenceBundleError::DuplicateAssay {
                item_id: assay.observation.item_id.clone(),
                endpoint: assay.observation.endpoint,
            });
        }
    }
    for item in &bundle.items {
        for endpoint in &bundle.required_endpoints {
            if !assay_pairs.contains(&(item.item_id.as_str(), *endpoint)) {
                return Err(EvidenceBundleError::MissingAssay {
                    item_id: item.item_id.clone(),
                    endpoint: *endpoint,
                });
            }
        }
    }

    let mut resource_ids = HashSet::new();
    let mut present_resources = BTreeSet::new();
    let mut resource_totals: BTreeMap<ResourceKind, f64> = BTreeMap::new();
    let mut boundary_intervals: BTreeMap<(ResourceKind, String), Vec<(u64, u64, String)>> =
        BTreeMap::new();
    for resource in &bundle.resources {
        if resource.source_run_id != bundle.run.run_id {
            return Err(EvidenceBundleError::InvalidResourceRun {
                measurement_id: resource.measurement_id.clone(),
                observed_run_id: resource.source_run_id.clone(),
            });
        }
        if !non_empty(&resource.measurement_id) {
            return Err(EvidenceBundleError::InvalidIdentity("resource_measurement_id"));
        }
        if !non_empty(&resource.measurement_boundary_id) {
            return Err(EvidenceBundleError::InvalidIdentity("resource_measurement_boundary_id"));
        }
        if !resource_ids.insert(resource.measurement_id.as_str())
            || !artifact_ids.insert(resource.measurement_id.as_str())
        {
            return Err(EvidenceBundleError::DuplicateArtifactId(resource.measurement_id.clone()));
        }
        if resource.end_utc_s <= resource.start_utc_s {
            return Err(EvidenceBundleError::InvalidResourceInterval(resource.measurement_id.clone()));
        }
        if resource.start_utc_s < bundle.start_utc_s || resource.end_utc_s > bundle.end_utc_s {
            return Err(EvidenceBundleError::ResourceOutsideRun(resource.measurement_id.clone()));
        }
        if !resource.value.is_finite() || resource.value < 0.0 {
            return Err(EvidenceBundleError::InvalidResourceValue(resource.measurement_id.clone()));
        }
        let expected_unit = resource.kind.required_unit();
        if resource.unit != expected_unit {
            return Err(EvidenceBundleError::ResourceUnitMismatch {
                measurement_id: resource.measurement_id.clone(),
                expected: expected_unit.to_owned(),
                observed: resource.unit.clone(),
            });
        }
        if !non_empty(&resource.method_id) || !non_empty(&resource.evidence_uri) || !valid_digest(&resource.evidence_sha256) {
            return Err(EvidenceBundleError::ResourceEvidenceMissing(resource.measurement_id.clone()));
        }
        // Overlap is checked by declared physical boundary rather than
        // instrument identity, so meter replacement or redundant instruments
        // cannot bypass the duplicate-increment guard.
        let boundary_key = (resource.kind, resource.measurement_boundary_id.clone());
        let intervals = boundary_intervals.entry(boundary_key).or_default();
        if let Some((_, _, first_id)) = intervals.iter().find(|(start, end, _)| {
            resource.start_utc_s < *end && *start < resource.end_utc_s
        }) {
            return Err(EvidenceBundleError::OverlappingResourceBoundaryIntervals {
                kind: resource.kind,
                measurement_boundary_id: resource.measurement_boundary_id.clone(),
                first_measurement_id: first_id.clone(),
                second_measurement_id: resource.measurement_id.clone(),
            });
        }
        intervals.push((
            resource.start_utc_s,
            resource.end_utc_s,
            resource.measurement_id.clone(),
        ));

        match (&resource.instrument_id, &resource.calibration_id) {
            (Some(instrument), Some(calibration_id)) if non_empty(instrument) && non_empty(calibration_id) => {
                validate_calibration(&calibrations, calibration_id, instrument, resource.start_utc_s)
                    .map_err(|_| EvidenceBundleError::ResourceInstrumentCalibrationMismatch(resource.measurement_id.clone()))?;
                validate_calibration(&calibrations, calibration_id, instrument, resource.end_utc_s)
                    .map_err(|_| EvidenceBundleError::ResourceInstrumentCalibrationMismatch(resource.measurement_id.clone()))?;

            }
            (None, None) if resource.kind == ResourceKind::ActiveOperatorTime => {}
            (None, None) => return Err(EvidenceBundleError::ManualResourceNotAllowed(resource.kind)),
            _ => return Err(EvidenceBundleError::ResourceInstrumentCalibrationMismatch(resource.measurement_id.clone())),
        }
        present_resources.insert(resource.kind);
        let total = resource_totals.entry(resource.kind).or_insert(0.0);
        *total += resource.value;
        if !total.is_finite() {
            return Err(EvidenceBundleError::ResourceTotalOverflow(resource.kind));
        }
    }

    for kind in required {
        if !present_resources.contains(&kind) {
            return Err(EvidenceBundleError::MissingRequiredResource(kind));
        }
    }
    let resource_totals = resource_totals.into_iter()
        .map(|(kind, value)| (kind, value, kind.required_unit()))
        .collect();

    Ok(EvidenceBundleValidationReport {
        schema_version: bundle.schema_version,
        run_id: bundle.run.run_id.clone(),
        physical_item_count: bundle.items.len(),
        required_endpoint_count: bundle.required_endpoints.len(),
        sensor_trace_count: bundle.sensor_traces.len(),
        assay_record_count: bundle.assays.len(),
        resource_measurement_count: bundle.resources.len(),
        calibration_count: bundle.calibrations.len(),
        resource_totals,
        partition: bundle.run.partition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{EvidenceKind, OutcomeObservation};

    fn digest(ch: char) -> String { std::iter::repeat(ch).take(64).collect() }

    fn calibration(id: &str, instrument: &str, bundle_digest: &str) -> InstrumentCalibration {
        InstrumentCalibration {
            calibration_id: id.to_owned(),
            instrument_id: instrument.to_owned(),
            valid_from_utc_s: 900,
            valid_until_utc_s: 2_000,
            certificate_uri: format!("evidence://calibration/{id}"),
            certificate_sha256: digest('c'),
            calibration_bundle_sha256: bundle_digest.to_owned(),
        }
    }

    fn run_record() -> RunPartitionRecord {
        RunPartitionRecord {
            run_id: "run-001".to_owned(),
            partition: ValidationPartition::LockedConfirmation,
            machine_unit_id: "machine-01".to_owned(),
            test_session_id: "session-01".to_owned(),
            soil_batch_id: "soil-01".to_owned(),
            detergent_batch_id: "detergent-01".to_owned(),
            water_batch_id: "water-01".to_owned(),
            operator_id: "operator-anon-01".to_owned(),
            assay_lab_id: "lab-01".to_owned(),
            test_site_id: "site-01".to_owned(),
            service_request_id: "service-request-001".to_owned(),
            compartment_id: "cell-01".to_owned(),
            compartment_configuration_sha256: digest('7'),
            utility_topology_id: "isolated-wet-loop-shared-thermal-bus-v1".to_owned(),
        }
    }

    fn assay(item: &str, endpoint: Endpoint, record_id: &str) -> AssayArtifact {
        AssayArtifact {
            record_id: record_id.to_owned(),
            observation: OutcomeObservation {
                run_id: "run-001".to_owned(),
                item_id: item.to_owned(),
                endpoint,
                criterion_id: format!("criterion-{endpoint:?}"),
                metric_id: format!("metric-{endpoint:?}"),
                evidence_kind: EvidenceKind::IndependentOutcomeAssay,
                method_id: format!("method-{endpoint:?}"),
                protocol_id: "protocol-v1".to_owned(),
                measured_value: 0.1,
                expanded_uncertainty: 0.01,
                unit: "mg".to_owned(),
                evidence_ref: format!("evidence://assay/{record_id}"),
                calibration_bundle_sha256: digest('a'),
            },
            instrument_id: "assay-meter-01".to_owned(),
            calibration_id: "assay-cal-01".to_owned(),
            assayed_at_utc_s: 1_500,
            archive_uri: format!("evidence://assay/{record_id}"),
            archive_sha256: digest('d'),
        }
    }

    fn valid_bundle() -> PhysicalRunEvidenceBundle {
        PhysicalRunEvidenceBundle {
            schema_version: EVIDENCE_BUNDLE_SCHEMA_VERSION,
            run: run_record(),
            machine_model_id: "prototype-v1".to_owned(),
            programme_id: "quickcell-v1".to_owned(),
            firmware_sha256: digest('f'),
            protocol_id: "protocol-v1".to_owned(),
            randomization_block_id: "block-01".to_owned(),
            treatment_assignment_id: "assignment-01".to_owned(),
            soil_age_at_start_s: 3_600,
            water_hardness_ppm_as_caco3: 75,
            inlet_temperature_millic: 18_000,
            ambient_temperature_millic: 22_000,
            ambient_relative_humidity_permille: 500,
            start_utc_s: 1_000,
            end_utc_s: 1_800,
            items: vec![PhysicalItemRecord {
                item_id: "cup-01".to_owned(),
                material: "glass".to_owned(),
                surface_finish: "smooth".to_owned(),
                geometry: "cup".to_owned(),
                location_stratum: "shadow-rear".to_owned(),
                item_batch_id: "glassware-lot-01".to_owned(),
            }],
            required_endpoints: vec![
                Endpoint::SoilRemovalResidual,
                Endpoint::FinalRinseResidue,
                Endpoint::CrossContamination,
            ],
            calibrations: vec![
                calibration("pressure-cal-01", "pressure-01", &digest('b')),
                calibration("assay-cal-01", "assay-meter-01", &digest('a')),
                calibration("power-cal-01", "power-01", &digest('e')),
                calibration("power-cal-02", "power-02", &digest('e')),
                calibration("water-cal-01", "water-meter-01", &digest('6')),
            ],
            sensor_traces: vec![SensorTraceArtifact {
                artifact_id: "trace-pressure-01".to_owned(),
                source_run_id: "run-001".to_owned(),
                instrument_id: "pressure-01".to_owned(),
                channel_id: "pump-discharge-pressure".to_owned(),
                metric_id: "pressure".to_owned(),
                unit: "kPa".to_owned(),
                start_utc_s: 1_000,
                end_utc_s: 1_800,
                sample_count: 80_001,
                sample_rate_hz: 100.0,
                calibration_id: "pressure-cal-01".to_owned(),
                archive_uri: "evidence://trace/pressure-01".to_owned(),
                archive_sha256: digest('1'),
            }],
            assays: vec![
                assay("cup-01", Endpoint::SoilRemovalResidual, "assay-soil-01"),
                assay("cup-01", Endpoint::FinalRinseResidue, "assay-rinse-01"),
                assay("cup-01", Endpoint::CrossContamination, "assay-cross-01"),
            ],
            resources: vec![
                ResourceMeasurement {
                    source_run_id: "run-001".to_owned(),
                    measurement_id: "wall-kwh-01".to_owned(),
                    measurement_boundary_id: "appliance-main-feed".to_owned(),
                    kind: ResourceKind::ApplianceWallElectricity,
                    value: 0.05,
                    unit: "kWh".to_owned(),
                    start_utc_s: 1_000,
                    end_utc_s: 1_800,
                    instrument_id: Some("power-01".to_owned()),
                    calibration_id: Some("power-cal-01".to_owned()),
                    method_id: "power-integral-v1".to_owned(),
                    evidence_uri: "evidence://resource/wall-kwh-01".to_owned(),
                    evidence_sha256: digest('2'),
                },
                ResourceMeasurement {
                    source_run_id: "run-001".to_owned(),
                    measurement_id: "water-l-01".to_owned(),
                    measurement_boundary_id: "fresh-water-inlet".to_owned(),
                    kind: ResourceKind::FreshWater,
                    value: 0.6,
                    unit: "L".to_owned(),
                    start_utc_s: 1_000,
                    end_utc_s: 1_800,
                    instrument_id: Some("water-meter-01".to_owned()),
                    calibration_id: Some("water-cal-01".to_owned()),
                    method_id: "water-total-v1".to_owned(),
                    evidence_uri: "evidence://resource/water-l-01".to_owned(),
                    evidence_sha256: digest('3'),
                },
                ResourceMeasurement {
                    source_run_id: "run-001".to_owned(),
                    measurement_id: "operator-s-01".to_owned(),
                    measurement_boundary_id: "operator-activity-log".to_owned(),
                    kind: ResourceKind::ActiveOperatorTime,
                    value: 30.0,
                    unit: "s".to_owned(),
                    start_utc_s: 1_000,
                    end_utc_s: 1_800,
                    instrument_id: None,
                    calibration_id: None,
                    method_id: "operator-time-log-v1".to_owned(),
                    evidence_uri: "evidence://resource/operator-s-01".to_owned(),
                    evidence_sha256: digest('4'),
                },
            ],
            archive_uri: "evidence://bundle/run-001".to_owned(),
            archive_sha256: digest('5'),
        }
    }

    #[test]
    fn validates_complete_physical_run_bundle_and_sums_resources() {
        let report = validate_physical_run_bundle(
            &valid_bundle(),
            &[
                ResourceKind::ApplianceWallElectricity,
                ResourceKind::FreshWater,
                ResourceKind::ActiveOperatorTime,
            ],
        ).unwrap();
        assert_eq!(report.run_id, "run-001");
        assert_eq!(report.physical_item_count, 1);
        assert_eq!(report.required_endpoint_count, 3);
        assert_eq!(report.sensor_trace_count, 1);
        assert_eq!(report.assay_record_count, 3);
        assert_eq!(report.resource_measurement_count, 3);
        assert_eq!(report.resource_totals.len(), 3);
    }

    #[test]
    fn rejects_item_without_material_or_geometry_metadata() {
        let mut bundle = valid_bundle();
        bundle.items[0].material.clear();
        assert_eq!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidIdentity("item.material"))
        );
    }

    #[test]
    fn rejects_relative_humidity_outside_permille_range() {
        let mut bundle = valid_bundle();
        bundle.ambient_relative_humidity_permille = 1_001;
        assert_eq!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidIdentity("ambient_relative_humidity_permille"))
        );
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let mut bundle = valid_bundle();
        bundle.schema_version = 2;
        assert_eq!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::UnsupportedSchemaVersion(2))
        );
    }

    #[test]
    fn rejects_assay_linked_to_another_run() {
        let mut bundle = valid_bundle();
        bundle.assays[0].observation.run_id = "other-run".to_owned();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidAssayRun { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_items_and_duplicate_assay_pairs() {
        let mut bundle = valid_bundle();
        bundle.items.push(bundle.items[0].clone());
        assert_eq!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::DuplicateItem("cup-01".to_owned()))
        );
        let mut bundle = valid_bundle();
        bundle.assays.push(bundle.assays[0].clone());
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::DuplicateAssay { .. })
        ));
    }

    #[test]
    fn rejects_missing_item_endpoint_assay() {
        let mut bundle = valid_bundle();
        bundle.assays.pop();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::MissingAssay { endpoint: Endpoint::CrossContamination, .. })
        ));
    }

    #[test]
    fn rejects_sensor_trace_outside_run_and_wrong_run_id() {
        let mut bundle = valid_bundle();
        bundle.sensor_traces[0].end_utc_s += 1;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::TraceOutsideRun(_))
        ));
        let mut bundle = valid_bundle();
        bundle.sensor_traces[0].source_run_id = "other-run".to_owned();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::TraceRunMismatch { .. })
        ));
    }

    #[test]
    fn rejects_trace_with_expired_calibration() {
        let mut bundle = valid_bundle();
        bundle.calibrations[0].valid_until_utc_s = 1_200;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::CalibrationNotValidAt { .. })
        ));
    }

    #[test]
    fn rejects_process_proxy_as_an_independent_endpoint_assay() {
        let mut bundle = valid_bundle();
        bundle.assays[0].observation.evidence_kind = crate::quality::EvidenceKind::CalibratedProcessProxy;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidIdentity("assay_must_be_independent_outcome"))
        ));
    }

    #[test]
    fn rejects_assay_with_mismatched_calibration_digest() {
        let mut bundle = valid_bundle();
        bundle.assays[0].observation.calibration_bundle_sha256 = digest('9');
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::AssayCalibrationDigestMismatch(_))
        ));
    }

    #[test]
    fn rejects_resource_unit_or_value_mismatch() {
        let mut bundle = valid_bundle();
        bundle.resources[0].unit = "Wh".to_owned();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::ResourceUnitMismatch { .. })
        ));
        let mut bundle = valid_bundle();
        bundle.resources[0].value = f64::NAN;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::InvalidResourceValue(_))
        ));
    }

    #[test]
    fn requires_explicit_resource_rows_and_rejects_manual_water() {
        let mut bundle = valid_bundle();
        bundle.resources.retain(|row| row.kind != ResourceKind::FreshWater);
        assert_eq!(
            validate_physical_run_bundle(
                &bundle,
                &[ResourceKind::FreshWater, ResourceKind::ActiveOperatorTime],
            ),
            Err(EvidenceBundleError::MissingRequiredResource(ResourceKind::FreshWater))
        );
        let mut bundle = valid_bundle();
        bundle.resources[1].instrument_id = None;
        bundle.resources[1].calibration_id = None;
        assert_eq!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::ManualResourceNotAllowed(ResourceKind::FreshWater))
        );
    }

    #[test]
    fn rejects_overlapping_intervals_from_the_same_resource_meter() {
        let mut bundle = valid_bundle();
        let mut duplicate_interval = bundle.resources[0].clone();
        duplicate_interval.measurement_id = "wall-kwh-overlap".to_owned();
        duplicate_interval.start_utc_s = 1_500;
        duplicate_interval.end_utc_s = 1_700;
        duplicate_interval.value = 0.01;
        bundle.resources.push(duplicate_interval);
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::OverlappingResourceBoundaryIntervals {
                kind: ResourceKind::ApplianceWallElectricity,
                ..
            })
        ));
    }

    #[test]
    fn allows_disjoint_intervals_from_the_same_resource_meter() {
        let mut bundle = valid_bundle();
        let mut second_interval = bundle.resources[0].clone();
        second_interval.measurement_id = "wall-kwh-second".to_owned();
        second_interval.start_utc_s = 1_400;
        second_interval.end_utc_s = 1_800;
        bundle.resources[0].end_utc_s = 1_400;
        bundle.resources.push(second_interval);
        assert!(validate_physical_run_bundle(
            &bundle,
            &[ResourceKind::ApplianceWallElectricity],
        ).is_ok());
    }

    #[test]
    fn rejects_trace_sampling_window_that_disagrees_with_sample_count_and_rate() {
        let mut bundle = valid_bundle();
        bundle.sensor_traces[0].sample_count = 40_000;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::TraceSamplingWindowMismatch(_))
        ));
    }

    #[test]
    fn rejects_single_sample_trace_without_a_temporal_span() {
        let mut bundle = valid_bundle();
        bundle.sensor_traces[0].sample_count = 1;
        bundle.sensor_traces[0].sample_rate_hz = 100.0;
        bundle.sensor_traces[0].end_utc_s = bundle.sensor_traces[0].start_utc_s;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidTraceSampling(_))
        ));
    }

    #[test]
    fn rejects_overlapping_resource_boundary_intervals_from_different_meters() {
        let mut bundle = valid_bundle();
        let mut second_meter = bundle.resources[0].clone();
        second_meter.measurement_id = "wall-kwh-overlap-secondary".to_owned();
        second_meter.start_utc_s = 1_500;
        second_meter.end_utc_s = 1_700;
        second_meter.value = 0.01;
        second_meter.instrument_id = Some("power-02".to_owned());
        second_meter.calibration_id = Some("power-cal-02".to_owned());
        second_meter.evidence_uri = "evidence://resource/wall-kwh-overlap-secondary".to_owned();
        second_meter.evidence_sha256 = digest('5');
        bundle.resources.push(second_meter);
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::OverlappingResourceBoundaryIntervals {
                kind: ResourceKind::ApplianceWallElectricity,
                measurement_boundary_id,
                ..
            }) if measurement_boundary_id == "appliance-main-feed"
        ));
    }

    #[test]
    fn allows_overlapping_intervals_for_distinct_declared_resource_boundaries() {
        let mut bundle = valid_bundle();
        let mut branch_meter = bundle.resources[0].clone();
        branch_meter.measurement_id = "wall-kwh-subcircuit".to_owned();
        branch_meter.measurement_boundary_id = "preheat-subcircuit".to_owned();
        branch_meter.instrument_id = Some("power-02".to_owned());
        branch_meter.calibration_id = Some("power-cal-02".to_owned());
        branch_meter.evidence_uri = "evidence://resource/wall-kwh-subcircuit".to_owned();
        branch_meter.evidence_sha256 = digest('5');
        bundle.resources.push(branch_meter);
        assert!(validate_physical_run_bundle(
            &bundle,
            &[ResourceKind::ApplianceWallElectricity],
        ).is_ok());
    }

    #[test]
    fn rejects_resource_measurement_with_empty_boundary_identity() {
        let mut bundle = valid_bundle();
        bundle.resources[0].measurement_boundary_id.clear();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::InvalidIdentity("resource_measurement_boundary_id"))
        ));
    }

    #[test]
    fn rejects_resource_measurement_linked_to_another_physical_run() {
        let mut bundle = valid_bundle();
        bundle.resources[0].source_run_id = "other-run".to_owned();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ApplianceWallElectricity]),
            Err(EvidenceBundleError::InvalidResourceRun {
                measurement_id,
                observed_run_id,
            }) if measurement_id == "wall-kwh-01" && observed_run_id == "other-run"
        ));
    }

    #[test]
    fn rejects_invalid_assay_measurement_uncertainty() {
        let mut bundle = valid_bundle();
        bundle.assays[0].observation.expanded_uncertainty = f64::NAN;
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::InvalidAssayMeasurement(_))
        ));
    }

    #[test]
    fn rejects_missing_resource_provenance_and_duplicate_artifact_ids() {
        let mut bundle = valid_bundle();
        bundle.resources[2].evidence_uri.clear();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::ActiveOperatorTime]),
            Err(EvidenceBundleError::ResourceEvidenceMissing(_))
        ));
        let mut bundle = valid_bundle();
        bundle.assays[0].record_id = bundle.sensor_traces[0].artifact_id.clone();
        assert!(matches!(
            validate_physical_run_bundle(&bundle, &[ResourceKind::FreshWater]),
            Err(EvidenceBundleError::DuplicateArtifactId(_))
        ));
    }
}
