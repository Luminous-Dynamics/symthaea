// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound photobiological exposure and response observations for Mycelial.
//!
//! Illumination is not assumed to be a passive observation. Fungal light sensing
//! can alter developmental and metabolic state independently of phototoxicity.
//! This contract records the optical treatment and separately typed outcomes;
//! it does not infer a species-general light-response curve or safe-dose limit.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const MYCELIAL_PHOTOBIOLOGICAL_EXPOSURE_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_PHOTOBIOLOGICAL_RESPONSE_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_PHOTOBIOLOGICAL_CHANNELS: usize = 64;
pub const MAX_MYCELIAL_PHOTOBIOLOGICAL_SCHEDULE_INTERVALS: usize = 65_536;
pub const MAX_MYCELIAL_PHOTOBIOLOGICAL_FOOTPRINT_CELLS: usize = 16_384;
pub const MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_PHOTOBIOLOGICAL_REASON_BYTES: usize = 512;
const DIGEST_HEX_BYTES: usize = 64;
const MAX_UNIT_BYTES: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialIlluminationPurpose {
    ImagingIllumination,
    DeliberatePhotoperiod,
    Photostimulation,
    Photoinactivation,
    DarkAmbientControl,
    UnknownPurpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialPhotobiologicalStage {
    VegetativeColonization,
    CasingColonization,
    PrimordiumInitiation,
    Fruiting,
    Sporulation,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialIlluminationScheduleKind {
    Continuous,
    Pulsed,
    Fractionated,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialPhotoperiodPhase {
    Light,
    Dark,
    Dawn,
    Dusk,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialPhotobiologicalChannelExposure {
    pub channel_id: String,
    pub minimum_wavelength_nm: u16,
    pub maximum_wavelength_nm: u16,
    /// Digest of the measured spectral distribution, not merely a channel label.
    pub spectral_distribution_digest: String,
    /// Specimen-plane radiometric calibration identity.
    pub calibration_digest: String,
    /// Conservative upper bound at the specimen plane, in W/cm².
    pub irradiance_watts_per_cm2_upper: f64,
    /// Conservative upper bound on integrated radiant exposure, in J/cm².
    pub fluence_joules_per_cm2_upper: f64,
    /// Conservative upper bound, in photons/µm².
    pub photon_dose_per_um2_upper: f64,
    /// Longest uninterrupted illumination interval for this channel.
    pub continuous_illumination_duration_ns: u64,
    /// Fraction of the scheduled interval illuminated, in parts per million.
    pub duty_cycle_parts_per_million: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialIlluminationInterval {
    pub channel_id: String,
    /// Start offset from the exposure's monotonic start, inclusive.
    pub start_offset_ns: u64,
    /// End offset from the exposure's monotonic start, exclusive.
    pub end_offset_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialPhotobiologicalExposureRecord {
    pub schema_version: u16,
    pub exposure_id: String,
    pub experiment_id: String,
    pub treatment_arm_id: String,
    pub sample_id: String,
    pub species_id: String,
    pub strain_or_isolate_id: String,
    pub biological_replicate_id: String,
    pub cultivation_unit_id: String,
    pub developmental_stage: MycelialPhotobiologicalStage,
    pub substrate_or_medium_id: String,
    pub substrate_condition_digest: String,
    /// Water activity in parts per million of the unit interval, when measured.
    pub water_activity_parts_per_million: Option<u32>,
    pub hydration_evidence_digest: Option<String>,
    /// Specimen temperature in millikelvin, when measured.
    pub specimen_temperature_millikelvin: Option<u32>,
    pub temperature_evidence_digest: Option<String>,
    pub illumination_purpose: MycelialIlluminationPurpose,
    pub schedule_kind: MycelialIlluminationScheduleKind,
    /// Digest of the exact sequence of channel on/off intervals and pulse timing.
    pub illumination_schedule_digest: String,
    /// Exact schedule intervals, canonicalized by channel then start/end offset.
    pub illumination_intervals: Vec<MycelialIlluminationInterval>,
    /// Ambient biological photoperiod, distinct from microscope channel scheduling.
    pub photoperiod_phase: MycelialPhotoperiodPhase,
    pub photoperiod_cycle_id: String,
    pub elapsed_since_photoperiod_transition_ns: Option<u64>,
    pub photoperiod_evidence_digest: String,
    pub acquisition_session_id: String,
    pub timebase_id: String,
    pub timebase_evidence_digest: String,
    pub exposure_start_ns: u64,
    pub exposure_end_ns: u64,
    pub coordinate_frame_id: String,
    pub acquisition_plan_digest: String,
    pub device_capability_manifest_digest: String,
    /// Digest commits coordinate frame, plan, device capability and source cells.
    pub spatial_footprint_digest: String,
    /// Canonical sorted source-grid cells illuminated by this exposure.
    pub spatial_footprint_cell_ids: Vec<String>,
    /// External instrument/session receipt; this module does not issue commands.
    pub execution_receipt_digest: String,
    /// Exact source-artifact inventory when image output exists; None for a recorded
    /// treatment-only exposure without an image artifact.
    pub output_artifact_inventory_digest: Option<String>,
    /// Channels are strictly sorted by channel_id.
    pub channels: Vec<MycelialPhotobiologicalChannelExposure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialPhotobiologicalObservationWindow {
    BaselineBeforeExposure,
    DuringExposure,
    ImmediatePostExposure,
    RecoveryFollowUp,
    LongitudinalFollowUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialPhotobiologicalMissingStatus {
    NotAcquired,
    UnavailableOrCorrupted,
    Unmeasurable,
    BelowDetection,
    Censored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MycelialPhotobiologicalMeasurement {
    /// Explicit missingness; never substitute numeric zero.
    MeasurementUnavailable {
        status: MycelialPhotobiologicalMissingStatus,
        reason: String,
    },
    /// Measured displacement and elapsed interval; downstream code may derive a rate.
    HyphalTipExtension {
        distance_micrometers: f64,
        elapsed_ns: u64,
        measurement_method_id: String,
    },
    /// Raw count and sampled centerline length; not a viability measure.
    BranchingDensity {
        branch_count: u64,
        sampled_centerline_length_micrometers: f64,
        measurement_method_id: String,
    },
    /// Structural cord coverage; not hydraulic conductance or biomass.
    CordAreaFraction {
        fraction: f64,
        sampled_roi_area_square_micrometers: f64,
        measurement_method_id: String,
    },
    /// Count and sampled area; not a hyphal-growth or viability proxy.
    ConidiationCount {
        conidia_count: u64,
        sampled_area_square_micrometers: f64,
        measurement_method_id: String,
    },
    /// Assay-defined viable fraction; it is not interchangeable with growth rate.
    ViabilityFraction { fraction: f64, assay_id: String },
    /// Assay-defined ROS signal; not a direct cell-death measurement.
    ReactiveOxygenSpeciesProxy {
        signal_value: f64,
        signal_unit: String,
        assay_id: String,
    },
    /// Analyte-specific result; no generic metabolite scalar is assumed.
    SecondaryMetaboliteConcentration {
        analyte_id: String,
        concentration: f64,
        concentration_unit: String,
        assay_id: String,
    },
    /// Time from a declared developmental origin to a fruiting/primordium event.
    FruitingInitiationLatency {
        developmental_origin_id: String,
        elapsed_ns: u64,
        measurement_method_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialPhotobiologicalResponseObservation {
    pub schema_version: u16,
    pub observation_id: String,
    pub exposure_id: String,
    pub exposure_digest: String,
    pub experiment_id: String,
    pub treatment_arm_id: String,
    pub sample_id: String,
    pub biological_replicate_id: String,
    pub cultivation_unit_id: String,
    pub timebase_id: String,
    pub observation_timestamp_ns: u64,
    pub observation_window: MycelialPhotobiologicalObservationWindow,
    pub source_artifact_digest: String,
    pub measurement_method_id: String,
    pub measurement_config_digest: String,
    pub measurement_calibration_digest: String,
    pub measurement: MycelialPhotobiologicalMeasurement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialPhotobiologicalExposureError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidExposure(&'static str),
    InvalidChannel(&'static str),
    InvalidResponse(&'static str),
    InvalidMeasurement(&'static str),
    ExposureBindingMismatch(&'static str),
    ObservationWindowMismatch,
    ResourceLimitExceeded(&'static str),
}

impl fmt::Display for MycelialPhotobiologicalExposureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported Mycelial photobiological schema version {version}"
                )
            }
            Self::InvalidIdentifier(field) => {
                write!(
                    f,
                    "invalid or unbounded photobiological identifier: {field}"
                )
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid photobiological evidence digest: {field}")
            }
            Self::InvalidExposure(field) => {
                write!(f, "invalid photobiological exposure record: {field}")
            }
            Self::InvalidChannel(field) => {
                write!(f, "invalid photobiological channel exposure: {field}")
            }
            Self::InvalidResponse(field) => {
                write!(f, "invalid photobiological response observation: {field}")
            }
            Self::InvalidMeasurement(field) => {
                write!(f, "invalid typed photobiological measurement: {field}")
            }
            Self::ExposureBindingMismatch(field) => {
                write!(
                    f,
                    "photobiological response disagrees with its exposure: {field}"
                )
            }
            Self::ObservationWindowMismatch => {
                write!(
                    f,
                    "photobiological observation timestamp disagrees with its declared window"
                )
            }
            Self::ResourceLimitExceeded(field) => {
                write!(f, "photobiological resource limit exceeded: {field}")
            }
        }
    }
}

impl Error for MycelialPhotobiologicalExposureError {}

fn valid_id(value: &str, maximum_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= maximum_bytes && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_identifier(
    value: &str,
    field: &'static str,
    maximum_bytes: usize,
) -> Result<(), MycelialPhotobiologicalExposureError> {
    if !valid_id(value, maximum_bytes) {
        return Err(MycelialPhotobiologicalExposureError::InvalidIdentifier(
            field,
        ));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialPhotobiologicalExposureError> {
    if !valid_digest(value) {
        return Err(MycelialPhotobiologicalExposureError::InvalidDigest(field));
    }
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn hash_option_string(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

/// Canonical digest for the complete per-channel illumination interval schedule.
pub fn mycelial_illumination_schedule_digest(
    exposure_duration_ns: u64,
    intervals: &[MycelialIlluminationInterval],
) -> Result<String, MycelialPhotobiologicalExposureError> {
    if exposure_duration_ns == 0
        || intervals.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_SCHEDULE_INTERVALS
    {
        return Err(MycelialPhotobiologicalExposureError::ResourceLimitExceeded(
            "illumination schedule intervals",
        ));
    }
    let mut previous: Option<(&str, u64, u64)> = None;
    let mut previous_end_by_channel = std::collections::BTreeMap::<String, u64>::new();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-illumination-schedule-v1");
    hasher.update(&exposure_duration_ns.to_le_bytes());
    hasher.update(&(intervals.len() as u64).to_le_bytes());
    for interval in intervals {
        validate_identifier(
            &interval.channel_id,
            "illumination_interval.channel_id",
            MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
        )?;
        if interval.start_offset_ns >= interval.end_offset_ns
            || interval.end_offset_ns > exposure_duration_ns
        {
            return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                "illumination interval must be positive and inside the exposure window",
            ));
        }
        let current = (
            interval.channel_id.as_str(),
            interval.start_offset_ns,
            interval.end_offset_ns,
        );
        if previous.is_some_and(|prior| prior >= current) {
            return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                "illumination intervals must be strictly canonical by channel/start/end",
            ));
        }
        if let Some(previous_end) = previous_end_by_channel.get(&interval.channel_id) {
            if interval.start_offset_ns < *previous_end {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "illumination intervals overlap within a channel",
                ));
            }
        }
        previous_end_by_channel.insert(interval.channel_id.clone(), interval.end_offset_ns);
        hash_string(&mut hasher, &interval.channel_id);
        hasher.update(&interval.start_offset_ns.to_le_bytes());
        hasher.update(&interval.end_offset_ns.to_le_bytes());
        previous = Some(current);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Canonical digest for an exposure footprint bound to its coordinate frame,
/// compiled acquisition plan, device capability manifest and sorted source cells.
pub fn mycelial_photobiological_footprint_digest(
    coordinate_frame_id: &str,
    acquisition_plan_digest: &str,
    device_capability_manifest_digest: &str,
    cells: &[String],
) -> Result<String, MycelialPhotobiologicalExposureError> {
    validate_identifier(
        coordinate_frame_id,
        "coordinate_frame_id",
        MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
    )?;
    validate_digest(acquisition_plan_digest, "acquisition_plan_digest")?;
    validate_digest(
        device_capability_manifest_digest,
        "device_capability_manifest_digest",
    )?;
    if cells.is_empty() || cells.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_FOOTPRINT_CELLS {
        return Err(MycelialPhotobiologicalExposureError::ResourceLimitExceeded(
            "spatial footprint cells",
        ));
    }
    let mut previous: Option<&str> = None;
    for cell in cells {
        validate_identifier(
            cell,
            "spatial_footprint_cell_id",
            MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
        )?;
        if previous.is_some_and(|prior| prior >= cell.as_str()) {
            return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                "spatial footprint cells must be strictly sorted and unique",
            ));
        }
        previous = Some(cell);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-photobiological-footprint-v1");
    hash_string(&mut hasher, coordinate_frame_id);
    hash_string(&mut hasher, acquisition_plan_digest);
    hash_string(&mut hasher, device_capability_manifest_digest);
    hasher.update(&(cells.len() as u64).to_le_bytes());
    for cell in cells {
        hash_string(&mut hasher, cell);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

impl MycelialPhotobiologicalExposureRecord {
    pub fn validate(&self) -> Result<(), MycelialPhotobiologicalExposureError> {
        if self.schema_version != MYCELIAL_PHOTOBIOLOGICAL_EXPOSURE_SCHEMA_VERSION {
            return Err(MycelialPhotobiologicalExposureError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }

        for (value, field) in [
            (&self.exposure_id, "exposure_id"),
            (&self.experiment_id, "experiment_id"),
            (&self.treatment_arm_id, "treatment_arm_id"),
            (&self.sample_id, "sample_id"),
            (&self.species_id, "species_id"),
            (&self.strain_or_isolate_id, "strain_or_isolate_id"),
            (&self.biological_replicate_id, "biological_replicate_id"),
            (&self.cultivation_unit_id, "cultivation_unit_id"),
            (&self.substrate_or_medium_id, "substrate_or_medium_id"),
            (&self.photoperiod_cycle_id, "photoperiod_cycle_id"),
            (&self.acquisition_session_id, "acquisition_session_id"),
            (&self.timebase_id, "timebase_id"),
            (&self.coordinate_frame_id, "coordinate_frame_id"),
        ] {
            validate_identifier(value, field, MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES)?;
        }

        for (value, field) in [
            (
                &self.substrate_condition_digest,
                "substrate_condition_digest",
            ),
            (&self.timebase_evidence_digest, "timebase_evidence_digest"),
            (
                &self.photoperiod_evidence_digest,
                "photoperiod_evidence_digest",
            ),
            (
                &self.illumination_schedule_digest,
                "illumination_schedule_digest",
            ),
            (&self.acquisition_plan_digest, "acquisition_plan_digest"),
            (
                &self.device_capability_manifest_digest,
                "device_capability_manifest_digest",
            ),
            (&self.spatial_footprint_digest, "spatial_footprint_digest"),
            (&self.execution_receipt_digest, "execution_receipt_digest"),
        ] {
            validate_digest(value, field)?;
        }

        match (
            self.water_activity_parts_per_million,
            self.hydration_evidence_digest.as_deref(),
        ) {
            (Some(value), Some(digest)) if value <= 1_000_000 => {
                validate_digest(digest, "hydration_evidence_digest")?;
            }
            (None, None) => {}
            _ => {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "water activity and hydration evidence must be jointly present, with water activity in [0, 1]",
                ));
            }
        }

        match (
            self.specimen_temperature_millikelvin,
            self.temperature_evidence_digest.as_deref(),
        ) {
            (Some(value), Some(digest)) if value > 0 => {
                validate_digest(digest, "temperature_evidence_digest")?;
            }
            (None, None) => {}
            _ => {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "specimen temperature and evidence must be jointly present",
                ));
            }
        }

        if let Some(digest) = self.output_artifact_inventory_digest.as_deref() {
            validate_digest(digest, "output_artifact_inventory_digest")?;
        }

        if self.exposure_start_ns >= self.exposure_end_ns {
            return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                "exposure start must precede exposure end",
            ));
        }
        match (
            self.photoperiod_phase,
            self.elapsed_since_photoperiod_transition_ns,
        ) {
            (MycelialPhotoperiodPhase::Unknown, None) => {}
            (MycelialPhotoperiodPhase::Unknown, Some(_)) | (_, None) => {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "unknown photoperiod phase cannot claim a transition age; known phases require one",
                ));
            }
            (_, Some(_)) => {}
        }

        if self.spatial_footprint_cell_ids.is_empty()
            || self.spatial_footprint_cell_ids.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_FOOTPRINT_CELLS
        {
            return Err(MycelialPhotobiologicalExposureError::ResourceLimitExceeded(
                "spatial footprint cells",
            ));
        }

        let mut previous_cell: Option<&str> = None;
        for cell_id in &self.spatial_footprint_cell_ids {
            validate_identifier(
                cell_id,
                "spatial_footprint_cell_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if previous_cell.is_some_and(|prior| prior >= cell_id.as_str()) {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "spatial footprint cell IDs must be strictly sorted and unique",
                ));
            }
            previous_cell = Some(cell_id);
        }

        if mycelial_photobiological_footprint_digest(
            &self.coordinate_frame_id,
            &self.acquisition_plan_digest,
            &self.device_capability_manifest_digest,
            &self.spatial_footprint_cell_ids,
        )? != self.spatial_footprint_digest
        {
            return Err(MycelialPhotobiologicalExposureError::InvalidDigest(
                "spatial_footprint_digest",
            ));
        }

        if self.illumination_intervals.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_SCHEDULE_INTERVALS {
            return Err(MycelialPhotobiologicalExposureError::ResourceLimitExceeded(
                "illumination schedule intervals",
            ));
        }
        if self.channels.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_CHANNELS {
            return Err(MycelialPhotobiologicalExposureError::ResourceLimitExceeded(
                "illumination channels",
            ));
        }

        match self.illumination_purpose {
            MycelialIlluminationPurpose::DarkAmbientControl
                if !self.channels.is_empty() || !self.illumination_intervals.is_empty() =>
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "dark/ambient control must not declare illumination channels or intervals",
                ));
            }
            MycelialIlluminationPurpose::DarkAmbientControl => {}
            _ if self.channels.is_empty() || self.illumination_intervals.is_empty() => {
                return Err(MycelialPhotobiologicalExposureError::InvalidExposure(
                    "illuminated exposure requires at least one channel",
                ));
            }
            _ => {}
        }

        let exposure_duration_ns = self.exposure_end_ns - self.exposure_start_ns;
        if mycelial_illumination_schedule_digest(
            exposure_duration_ns,
            &self.illumination_intervals,
        )? != self.illumination_schedule_digest
        {
            return Err(MycelialPhotobiologicalExposureError::InvalidDigest(
                "illumination_schedule_digest",
            ));
        }
        let mut previous_channel: Option<&str> = None;
        let mut channel_totals = std::collections::BTreeMap::<String, (u64, u64)>::new();
        let mut channel_interval_counts = std::collections::BTreeMap::<String, usize>::new();
        for interval in &self.illumination_intervals {
            if !self
                .channels
                .iter()
                .any(|channel| channel.channel_id == interval.channel_id)
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidChannel(
                    "illumination interval references a channel absent from the channel inventory",
                ));
            }
            let total = channel_totals
                .entry(interval.channel_id.clone())
                .or_insert((0, 0));
            let pulse_duration = interval.end_offset_ns - interval.start_offset_ns;
            total.0 = total.0.checked_add(pulse_duration).ok_or(
                MycelialPhotobiologicalExposureError::InvalidExposure(
                    "illumination duration accumulation overflow",
                ),
            )?;
            total.1 = total.1.max(pulse_duration);
            *channel_interval_counts
                .entry(interval.channel_id.clone())
                .or_insert(0) += 1;
        }
        for channel in &self.channels {
            validate_identifier(
                &channel.channel_id,
                "channel_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            for (digest, field) in [
                (
                    channel.spectral_distribution_digest.as_str(),
                    "spectral_distribution_digest",
                ),
                (
                    channel.calibration_digest.as_str(),
                    "channel.calibration_digest",
                ),
            ] {
                validate_digest(digest, field)?;
            }
            if previous_channel.is_some_and(|prior| prior >= channel.channel_id.as_str()) {
                return Err(MycelialPhotobiologicalExposureError::InvalidChannel(
                    "channels must be strictly sorted and unique by channel_id",
                ));
            }
            previous_channel = Some(&channel.channel_id);

            let Some((total_on_ns, maximum_contiguous_ns)) =
                channel_totals.get(&channel.channel_id).copied()
            else {
                return Err(MycelialPhotobiologicalExposureError::InvalidChannel(
                    "every illuminated channel must have at least one scheduled interval",
                ));
            };
            let expected_duty_cycle =
                ((total_on_ns as f64 / exposure_duration_ns as f64) * 1_000_000.0).round() as u32;
            let minimum_consistent_fluence =
                channel.irradiance_watts_per_cm2_upper * (total_on_ns as f64 / 1_000_000_000.0);
            let interval_count = channel_interval_counts
                .get(&channel.channel_id)
                .copied()
                .unwrap_or(0);
            let schedule_kind_matches = match self.schedule_kind {
                MycelialIlluminationScheduleKind::Continuous => interval_count == 1,
                MycelialIlluminationScheduleKind::Pulsed
                | MycelialIlluminationScheduleKind::Fractionated => interval_count >= 2,
                MycelialIlluminationScheduleKind::Mixed
                | MycelialIlluminationScheduleKind::Unknown => true,
            };
            if !schedule_kind_matches
                || maximum_contiguous_ns != channel.continuous_illumination_duration_ns
                || channel
                    .duty_cycle_parts_per_million
                    .abs_diff(expected_duty_cycle)
                    > 1
                || channel.fluence_joules_per_cm2_upper + 1e-12 < minimum_consistent_fluence
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidChannel(
                    "channel dwell, duty cycle or fluence disagrees with the exact interval schedule",
                ));
            }
            if channel.minimum_wavelength_nm == 0
                || channel.minimum_wavelength_nm > channel.maximum_wavelength_nm
                || !channel.irradiance_watts_per_cm2_upper.is_finite()
                || channel.irradiance_watts_per_cm2_upper <= 0.0
                || !channel.fluence_joules_per_cm2_upper.is_finite()
                || channel.fluence_joules_per_cm2_upper <= 0.0
                || !channel.photon_dose_per_um2_upper.is_finite()
                || channel.photon_dose_per_um2_upper <= 0.0
                || channel.continuous_illumination_duration_ns == 0
                || channel.continuous_illumination_duration_ns
                    > self.exposure_end_ns.saturating_sub(self.exposure_start_ns)
                || channel.duty_cycle_parts_per_million == 0
                || channel.duty_cycle_parts_per_million > 1_000_000
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidChannel(
                    "invalid wavelength, positive upper-bound exposure, dwell or duty cycle",
                ));
            }
        }

        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialPhotobiologicalExposureError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-photobiological-exposure-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for value in [
            self.exposure_id.as_str(),
            self.experiment_id.as_str(),
            self.treatment_arm_id.as_str(),
            self.sample_id.as_str(),
            self.species_id.as_str(),
            self.strain_or_isolate_id.as_str(),
            self.biological_replicate_id.as_str(),
            self.cultivation_unit_id.as_str(),
            self.substrate_or_medium_id.as_str(),
            self.substrate_condition_digest.as_str(),
            self.acquisition_session_id.as_str(),
            self.timebase_id.as_str(),
            self.timebase_evidence_digest.as_str(),
            self.coordinate_frame_id.as_str(),
            self.acquisition_plan_digest.as_str(),
            self.device_capability_manifest_digest.as_str(),
            self.spatial_footprint_digest.as_str(),
            self.execution_receipt_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&(self.illumination_intervals.len() as u64).to_le_bytes());
        for interval in &self.illumination_intervals {
            hash_string(&mut hasher, &interval.channel_id);
            hasher.update(&interval.start_offset_ns.to_le_bytes());
            hasher.update(&interval.end_offset_ns.to_le_bytes());
        }
        hasher.update(&[
            stage_tag(self.developmental_stage),
            purpose_tag(self.illumination_purpose),
            schedule_tag(self.schedule_kind),
            photoperiod_phase_tag(self.photoperiod_phase),
        ]);
        hash_string(&mut hasher, &self.illumination_schedule_digest);
        hash_string(&mut hasher, &self.photoperiod_cycle_id);
        match self.elapsed_since_photoperiod_transition_ns {
            Some(elapsed) => {
                hasher.update(&[1]);
                hasher.update(&elapsed.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_string(&mut hasher, &self.photoperiod_evidence_digest);
        hasher.update(&self.exposure_start_ns.to_le_bytes());
        hasher.update(&self.exposure_end_ns.to_le_bytes());
        match self.water_activity_parts_per_million {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_option_string(&mut hasher, self.hydration_evidence_digest.as_deref());
        match self.specimen_temperature_millikelvin {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_option_string(&mut hasher, self.temperature_evidence_digest.as_deref());
        hash_option_string(
            &mut hasher,
            self.output_artifact_inventory_digest.as_deref(),
        );
        hasher.update(&(self.spatial_footprint_cell_ids.len() as u64).to_le_bytes());
        for cell_id in &self.spatial_footprint_cell_ids {
            hash_string(&mut hasher, cell_id);
        }
        hasher.update(&(self.channels.len() as u64).to_le_bytes());
        for channel in &self.channels {
            hash_string(&mut hasher, &channel.channel_id);
            hasher.update(&channel.minimum_wavelength_nm.to_le_bytes());
            hasher.update(&channel.maximum_wavelength_nm.to_le_bytes());
            hash_string(&mut hasher, &channel.spectral_distribution_digest);
            hash_string(&mut hasher, &channel.calibration_digest);
            hash_f64(&mut hasher, channel.irradiance_watts_per_cm2_upper);
            hash_f64(&mut hasher, channel.fluence_joules_per_cm2_upper);
            hash_f64(&mut hasher, channel.photon_dose_per_um2_upper);
            hasher.update(&channel.continuous_illumination_duration_ns.to_le_bytes());
            hasher.update(&channel.duty_cycle_parts_per_million.to_le_bytes());
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

impl MycelialPhotobiologicalResponseObservation {
    pub fn validate(&self) -> Result<(), MycelialPhotobiologicalExposureError> {
        if self.schema_version != MYCELIAL_PHOTOBIOLOGICAL_RESPONSE_SCHEMA_VERSION {
            return Err(MycelialPhotobiologicalExposureError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        for (value, field) in [
            (&self.observation_id, "observation_id"),
            (&self.exposure_id, "exposure_id"),
            (&self.experiment_id, "experiment_id"),
            (&self.treatment_arm_id, "treatment_arm_id"),
            (&self.sample_id, "sample_id"),
            (&self.biological_replicate_id, "biological_replicate_id"),
            (&self.cultivation_unit_id, "cultivation_unit_id"),
            (&self.timebase_id, "timebase_id"),
            (&self.measurement_method_id, "measurement_method_id"),
        ] {
            validate_identifier(value, field, MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES)?;
        }
        for (digest, field) in [
            (self.exposure_digest.as_str(), "exposure_digest"),
            (
                self.source_artifact_digest.as_str(),
                "source_artifact_digest",
            ),
            (
                self.measurement_config_digest.as_str(),
                "measurement_config_digest",
            ),
            (
                self.measurement_calibration_digest.as_str(),
                "measurement_calibration_digest",
            ),
        ] {
            validate_digest(digest, field)?;
        }
        validate_measurement(&self.measurement)?;
        Ok(())
    }

    pub fn validate_against_exposure(
        &self,
        exposure: &MycelialPhotobiologicalExposureRecord,
    ) -> Result<(), MycelialPhotobiologicalExposureError> {
        self.validate()?;
        let exposure_digest = exposure.canonical_digest_hex()?;
        if self.exposure_digest != exposure_digest {
            return Err(
                MycelialPhotobiologicalExposureError::ExposureBindingMismatch("exposure_digest"),
            );
        }
        for (left, right, field) in [
            (
                self.exposure_id.as_str(),
                exposure.exposure_id.as_str(),
                "exposure_id",
            ),
            (
                self.experiment_id.as_str(),
                exposure.experiment_id.as_str(),
                "experiment_id",
            ),
            (
                self.treatment_arm_id.as_str(),
                exposure.treatment_arm_id.as_str(),
                "treatment_arm_id",
            ),
            (
                self.sample_id.as_str(),
                exposure.sample_id.as_str(),
                "sample_id",
            ),
            (
                self.biological_replicate_id.as_str(),
                exposure.biological_replicate_id.as_str(),
                "biological_replicate_id",
            ),
            (
                self.cultivation_unit_id.as_str(),
                exposure.cultivation_unit_id.as_str(),
                "cultivation_unit_id",
            ),
            (
                self.timebase_id.as_str(),
                exposure.timebase_id.as_str(),
                "timebase_id",
            ),
        ] {
            if left != right {
                return Err(MycelialPhotobiologicalExposureError::ExposureBindingMismatch(field));
            }
        }

        let window_is_valid = match self.observation_window {
            MycelialPhotobiologicalObservationWindow::BaselineBeforeExposure => {
                self.observation_timestamp_ns < exposure.exposure_start_ns
            }
            MycelialPhotobiologicalObservationWindow::DuringExposure => {
                self.observation_timestamp_ns >= exposure.exposure_start_ns
                    && self.observation_timestamp_ns <= exposure.exposure_end_ns
            }
            MycelialPhotobiologicalObservationWindow::ImmediatePostExposure
            | MycelialPhotobiologicalObservationWindow::RecoveryFollowUp
            | MycelialPhotobiologicalObservationWindow::LongitudinalFollowUp => {
                self.observation_timestamp_ns > exposure.exposure_end_ns
            }
        };
        if !window_is_valid {
            return Err(MycelialPhotobiologicalExposureError::ObservationWindowMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialPhotobiologicalExposureError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-photobiological-response-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for value in [
            self.observation_id.as_str(),
            self.exposure_id.as_str(),
            self.exposure_digest.as_str(),
            self.experiment_id.as_str(),
            self.treatment_arm_id.as_str(),
            self.sample_id.as_str(),
            self.biological_replicate_id.as_str(),
            self.cultivation_unit_id.as_str(),
            self.timebase_id.as_str(),
            self.source_artifact_digest.as_str(),
            self.measurement_method_id.as_str(),
            self.measurement_config_digest.as_str(),
            self.measurement_calibration_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&self.observation_timestamp_ns.to_le_bytes());
        hasher.update(&[observation_window_tag(self.observation_window)]);
        hash_measurement(&mut hasher, &self.measurement);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_measurement(
    measurement: &MycelialPhotobiologicalMeasurement,
) -> Result<(), MycelialPhotobiologicalExposureError> {
    match measurement {
        MycelialPhotobiologicalMeasurement::MeasurementUnavailable { reason, .. } => {
            if reason.trim().is_empty()
                || reason.len() > MAX_MYCELIAL_PHOTOBIOLOGICAL_REASON_BYTES
                || reason.chars().any(char::is_control)
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "missing/censored observation requires a bounded reason",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::HyphalTipExtension {
            distance_micrometers,
            elapsed_ns,
            measurement_method_id,
        } => {
            validate_identifier(
                measurement_method_id,
                "measurement_method_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if !distance_micrometers.is_finite() || *distance_micrometers < 0.0 || *elapsed_ns == 0
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "hyphal extension requires finite non-negative distance and positive interval",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::BranchingDensity {
            sampled_centerline_length_micrometers,
            measurement_method_id,
            ..
        } => {
            validate_identifier(
                measurement_method_id,
                "measurement_method_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if !sampled_centerline_length_micrometers.is_finite()
                || *sampled_centerline_length_micrometers <= 0.0
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "branching density requires positive measured centerline length",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::CordAreaFraction {
            fraction,
            sampled_roi_area_square_micrometers,
            measurement_method_id,
        } => {
            validate_identifier(
                measurement_method_id,
                "measurement_method_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if !fraction.is_finite()
                || !(0.0..=1.0).contains(fraction)
                || !sampled_roi_area_square_micrometers.is_finite()
                || *sampled_roi_area_square_micrometers <= 0.0
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "cord area fraction requires [0,1] fraction and positive ROI area",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::ConidiationCount {
            sampled_area_square_micrometers,
            measurement_method_id,
            ..
        } => {
            validate_identifier(
                measurement_method_id,
                "measurement_method_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if !sampled_area_square_micrometers.is_finite()
                || *sampled_area_square_micrometers <= 0.0
            {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "conidiation count requires positive sampled area",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::ViabilityFraction { fraction, assay_id } => {
            validate_identifier(assay_id, "assay_id", MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES)?;
            if !fraction.is_finite() || !(0.0..=1.0).contains(fraction) {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "viability fraction must be in [0,1]",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::ReactiveOxygenSpeciesProxy {
            signal_value,
            signal_unit,
            assay_id,
        } => {
            validate_identifier(signal_unit, "signal_unit", MAX_UNIT_BYTES)?;
            validate_identifier(assay_id, "assay_id", MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES)?;
            if !signal_value.is_finite() || *signal_value < 0.0 {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "ROS signal must be finite and non-negative",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::SecondaryMetaboliteConcentration {
            analyte_id,
            concentration,
            concentration_unit,
            assay_id,
        } => {
            validate_identifier(
                analyte_id,
                "analyte_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            validate_identifier(concentration_unit, "concentration_unit", MAX_UNIT_BYTES)?;
            validate_identifier(assay_id, "assay_id", MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES)?;
            if !concentration.is_finite() || *concentration < 0.0 {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "metabolite concentration must be finite and non-negative",
                ));
            }
        }
        MycelialPhotobiologicalMeasurement::FruitingInitiationLatency {
            developmental_origin_id,
            elapsed_ns,
            measurement_method_id,
        } => {
            validate_identifier(
                developmental_origin_id,
                "developmental_origin_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            validate_identifier(
                measurement_method_id,
                "measurement_method_id",
                MAX_MYCELIAL_PHOTOBIOLOGICAL_ID_BYTES,
            )?;
            if *elapsed_ns == 0 {
                return Err(MycelialPhotobiologicalExposureError::InvalidMeasurement(
                    "fruiting initiation latency must be positive",
                ));
            }
        }
    }
    Ok(())
}

fn hash_measurement(hasher: &mut blake3::Hasher, measurement: &MycelialPhotobiologicalMeasurement) {
    match measurement {
        MycelialPhotobiologicalMeasurement::MeasurementUnavailable { status, reason } => {
            hasher.update(&[0, missing_status_tag(*status)]);
            hash_string(hasher, reason);
        }
        MycelialPhotobiologicalMeasurement::HyphalTipExtension {
            distance_micrometers,
            elapsed_ns,
            measurement_method_id,
        } => {
            hasher.update(&[1]);
            hash_f64(hasher, *distance_micrometers);
            hasher.update(&elapsed_ns.to_le_bytes());
            hash_string(hasher, measurement_method_id);
        }
        MycelialPhotobiologicalMeasurement::BranchingDensity {
            branch_count,
            sampled_centerline_length_micrometers,
            measurement_method_id,
        } => {
            hasher.update(&[2]);
            hasher.update(&branch_count.to_le_bytes());
            hash_f64(hasher, *sampled_centerline_length_micrometers);
            hash_string(hasher, measurement_method_id);
        }
        MycelialPhotobiologicalMeasurement::CordAreaFraction {
            fraction,
            sampled_roi_area_square_micrometers,
            measurement_method_id,
        } => {
            hasher.update(&[3]);
            hash_f64(hasher, *fraction);
            hash_f64(hasher, *sampled_roi_area_square_micrometers);
            hash_string(hasher, measurement_method_id);
        }
        MycelialPhotobiologicalMeasurement::ConidiationCount {
            conidia_count,
            sampled_area_square_micrometers,
            measurement_method_id,
        } => {
            hasher.update(&[4]);
            hasher.update(&conidia_count.to_le_bytes());
            hash_f64(hasher, *sampled_area_square_micrometers);
            hash_string(hasher, measurement_method_id);
        }
        MycelialPhotobiologicalMeasurement::ViabilityFraction { fraction, assay_id } => {
            hasher.update(&[5]);
            hash_f64(hasher, *fraction);
            hash_string(hasher, assay_id);
        }
        MycelialPhotobiologicalMeasurement::ReactiveOxygenSpeciesProxy {
            signal_value,
            signal_unit,
            assay_id,
        } => {
            hasher.update(&[6]);
            hash_f64(hasher, *signal_value);
            hash_string(hasher, signal_unit);
            hash_string(hasher, assay_id);
        }
        MycelialPhotobiologicalMeasurement::SecondaryMetaboliteConcentration {
            analyte_id,
            concentration,
            concentration_unit,
            assay_id,
        } => {
            hasher.update(&[7]);
            hash_string(hasher, analyte_id);
            hash_f64(hasher, *concentration);
            hash_string(hasher, concentration_unit);
            hash_string(hasher, assay_id);
        }
        MycelialPhotobiologicalMeasurement::FruitingInitiationLatency {
            developmental_origin_id,
            elapsed_ns,
            measurement_method_id,
        } => {
            hasher.update(&[8]);
            hash_string(hasher, developmental_origin_id);
            hasher.update(&elapsed_ns.to_le_bytes());
            hash_string(hasher, measurement_method_id);
        }
    }
}

fn stage_tag(stage: MycelialPhotobiologicalStage) -> u8 {
    match stage {
        MycelialPhotobiologicalStage::VegetativeColonization => 0,
        MycelialPhotobiologicalStage::CasingColonization => 1,
        MycelialPhotobiologicalStage::PrimordiumInitiation => 2,
        MycelialPhotobiologicalStage::Fruiting => 3,
        MycelialPhotobiologicalStage::Sporulation => 4,
        MycelialPhotobiologicalStage::Unknown => 5,
    }
}

fn purpose_tag(purpose: MycelialIlluminationPurpose) -> u8 {
    match purpose {
        MycelialIlluminationPurpose::ImagingIllumination => 0,
        MycelialIlluminationPurpose::DeliberatePhotoperiod => 1,
        MycelialIlluminationPurpose::Photostimulation => 2,
        MycelialIlluminationPurpose::Photoinactivation => 3,
        MycelialIlluminationPurpose::DarkAmbientControl => 4,
        MycelialIlluminationPurpose::UnknownPurpose => 5,
    }
}

fn schedule_tag(schedule: MycelialIlluminationScheduleKind) -> u8 {
    match schedule {
        MycelialIlluminationScheduleKind::Continuous => 0,
        MycelialIlluminationScheduleKind::Pulsed => 1,
        MycelialIlluminationScheduleKind::Fractionated => 2,
        MycelialIlluminationScheduleKind::Mixed => 3,
        MycelialIlluminationScheduleKind::Unknown => 4,
    }
}

fn photoperiod_phase_tag(phase: MycelialPhotoperiodPhase) -> u8 {
    match phase {
        MycelialPhotoperiodPhase::Light => 0,
        MycelialPhotoperiodPhase::Dark => 1,
        MycelialPhotoperiodPhase::Dawn => 2,
        MycelialPhotoperiodPhase::Dusk => 3,
        MycelialPhotoperiodPhase::Unknown => 4,
    }
}

fn observation_window_tag(window: MycelialPhotobiologicalObservationWindow) -> u8 {
    match window {
        MycelialPhotobiologicalObservationWindow::BaselineBeforeExposure => 0,
        MycelialPhotobiologicalObservationWindow::DuringExposure => 1,
        MycelialPhotobiologicalObservationWindow::ImmediatePostExposure => 2,
        MycelialPhotobiologicalObservationWindow::RecoveryFollowUp => 3,
        MycelialPhotobiologicalObservationWindow::LongitudinalFollowUp => 4,
    }
}

fn missing_status_tag(status: MycelialPhotobiologicalMissingStatus) -> u8 {
    match status {
        MycelialPhotobiologicalMissingStatus::NotAcquired => 0,
        MycelialPhotobiologicalMissingStatus::UnavailableOrCorrupted => 1,
        MycelialPhotobiologicalMissingStatus::Unmeasurable => 2,
        MycelialPhotobiologicalMissingStatus::BelowDetection => 3,
        MycelialPhotobiologicalMissingStatus::Censored => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("{:064x}", character as u32)
    }

    fn exposure() -> MycelialPhotobiologicalExposureRecord {
        let mut record = MycelialPhotobiologicalExposureRecord {
            schema_version: MYCELIAL_PHOTOBIOLOGICAL_EXPOSURE_SCHEMA_VERSION,
            exposure_id: "exposure-001".to_string(),
            experiment_id: "photobiology-study-001".to_string(),
            treatment_arm_id: "blue-pulsed".to_string(),
            sample_id: "sample-001".to_string(),
            species_id: "agaricus-bisporus".to_string(),
            strain_or_isolate_id: "isolate-001".to_string(),
            biological_replicate_id: "replicate-001".to_string(),
            cultivation_unit_id: "box-001".to_string(),
            developmental_stage: MycelialPhotobiologicalStage::VegetativeColonization,
            substrate_or_medium_id: "compost-001".to_string(),
            substrate_condition_digest: digest('1'),
            water_activity_parts_per_million: Some(950_000),
            hydration_evidence_digest: Some(digest('2')),
            specimen_temperature_millikelvin: Some(298_150),
            temperature_evidence_digest: Some(digest('3')),
            illumination_purpose: MycelialIlluminationPurpose::ImagingIllumination,
            schedule_kind: MycelialIlluminationScheduleKind::Fractionated,
            illumination_schedule_digest: digest('e'),
            photoperiod_phase: MycelialPhotoperiodPhase::Light,
            photoperiod_cycle_id: "cycle-001".to_string(),
            elapsed_since_photoperiod_transition_ns: Some(3_600_000_000_000),
            photoperiod_evidence_digest: digest('d'),
            acquisition_session_id: "session-001".to_string(),
            timebase_id: "monotonic-ns-v1".to_string(),
            timebase_evidence_digest: digest('4'),
            exposure_start_ns: 1_000,
            exposure_end_ns: 2_000,
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            acquisition_plan_digest: digest('5'),
            device_capability_manifest_digest: digest('6'),
            spatial_footprint_digest: String::new(),
            spatial_footprint_cell_ids: vec!["cell-a".to_string(), "cell-b".to_string()],
            execution_receipt_digest: digest('7'),
            output_artifact_inventory_digest: Some(digest('8')),
            illumination_intervals: vec![
                MycelialIlluminationInterval {
                    channel_id: "blue".to_string(),
                    start_offset_ns: 0,
                    end_offset_ns: 250,
                },
                MycelialIlluminationInterval {
                    channel_id: "blue".to_string(),
                    start_offset_ns: 375,
                    end_offset_ns: 625,
                },
            ],
            channels: vec![MycelialPhotobiologicalChannelExposure {
                channel_id: "blue".to_string(),
                minimum_wavelength_nm: 440,
                maximum_wavelength_nm: 480,
                spectral_distribution_digest: digest('9'),
                calibration_digest: digest('a'),
                irradiance_watts_per_cm2_upper: 0.000_5,
                fluence_joules_per_cm2_upper: 0.01,
                photon_dose_per_um2_upper: 100.0,
                continuous_illumination_duration_ns: 250,
                duty_cycle_parts_per_million: 500_000,
            }],
        };
        record.illumination_schedule_digest = mycelial_illumination_schedule_digest(
            record.exposure_end_ns - record.exposure_start_ns,
            &record.illumination_intervals,
        )
        .unwrap();
        record.spatial_footprint_digest = mycelial_photobiological_footprint_digest(
            &record.coordinate_frame_id,
            &record.acquisition_plan_digest,
            &record.device_capability_manifest_digest,
            &record.spatial_footprint_cell_ids,
        )
        .unwrap();
        record
    }

    fn response(
        exposure: &MycelialPhotobiologicalExposureRecord,
    ) -> MycelialPhotobiologicalResponseObservation {
        MycelialPhotobiologicalResponseObservation {
            schema_version: MYCELIAL_PHOTOBIOLOGICAL_RESPONSE_SCHEMA_VERSION,
            observation_id: "response-001".to_string(),
            exposure_id: exposure.exposure_id.clone(),
            exposure_digest: exposure.canonical_digest_hex().unwrap(),
            experiment_id: exposure.experiment_id.clone(),
            treatment_arm_id: exposure.treatment_arm_id.clone(),
            sample_id: exposure.sample_id.clone(),
            biological_replicate_id: exposure.biological_replicate_id.clone(),
            cultivation_unit_id: exposure.cultivation_unit_id.clone(),
            timebase_id: exposure.timebase_id.clone(),
            observation_timestamp_ns: 2_500,
            observation_window: MycelialPhotobiologicalObservationWindow::RecoveryFollowUp,
            source_artifact_digest: digest('b'),
            measurement_method_id: "manual-segmented-hyphal-front-v1".to_string(),
            measurement_config_digest: digest('c'),
            measurement_calibration_digest: digest('d'),
            measurement: MycelialPhotobiologicalMeasurement::HyphalTipExtension {
                distance_micrometers: 42.0,
                elapsed_ns: 10_000,
                measurement_method_id: "hyphal-front-tracking-v1".to_string(),
            },
        }
    }

    #[test]
    fn exposure_record_is_source_bound_and_has_stable_digest() {
        let record = exposure();
        record.validate().unwrap();
        let first = record.canonical_digest_hex().unwrap();
        let second = record.canonical_digest_hex().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), DIGEST_HEX_BYTES);
    }

    #[test]
    fn response_must_bind_to_exact_exposure_and_experimental_unit() {
        let record = exposure();
        let response = response(&record);
        response.validate_against_exposure(&record).unwrap();

        let mut wrong_unit = response;
        wrong_unit.cultivation_unit_id = "box-002".to_string();
        assert!(matches!(
            wrong_unit.validate_against_exposure(&record),
            Err(
                MycelialPhotobiologicalExposureError::ExposureBindingMismatch(
                    "cultivation_unit_id"
                )
            )
        ));
    }

    #[test]
    fn observation_window_must_match_exposure_timestamps() {
        let record = exposure();
        let mut response = response(&record);
        response.observation_window =
            MycelialPhotobiologicalObservationWindow::BaselineBeforeExposure;
        response.observation_timestamp_ns = record.exposure_start_ns;
        assert_eq!(
            response.validate_against_exposure(&record),
            Err(MycelialPhotobiologicalExposureError::ObservationWindowMismatch)
        );
    }

    #[test]
    fn same_fluence_does_not_collapse_irradiance_dwell_or_schedule() {
        let mut continuous = exposure();
        continuous.schedule_kind = MycelialIlluminationScheduleKind::Continuous;
        continuous.illumination_intervals = vec![MycelialIlluminationInterval {
            channel_id: "blue".to_string(),
            start_offset_ns: 0,
            end_offset_ns: 1_000,
        }];
        continuous.illumination_schedule_digest = mycelial_illumination_schedule_digest(
            continuous.exposure_end_ns - continuous.exposure_start_ns,
            &continuous.illumination_intervals,
        )
        .unwrap();
        continuous.channels[0].continuous_illumination_duration_ns = 1_000;
        continuous.channels[0].duty_cycle_parts_per_million = 1_000_000;
        continuous.channels[0].fluence_joules_per_cm2_upper = 0.01;
        continuous.validate().unwrap();

        let mut fractionated = exposure();
        fractionated.schedule_kind = MycelialIlluminationScheduleKind::Fractionated;
        fractionated.channels[0].continuous_illumination_duration_ns = 250;
        fractionated.channels[0].duty_cycle_parts_per_million = 500_000;
        fractionated.validate().unwrap();

        assert_eq!(
            continuous.channels[0].fluence_joules_per_cm2_upper,
            fractionated.channels[0].fluence_joules_per_cm2_upper
        );
        assert_ne!(
            continuous.canonical_digest_hex().unwrap(),
            fractionated.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn illumination_intervals_recompute_schedule_digest_and_reject_overlap() {
        let record = exposure();
        assert_eq!(
            mycelial_illumination_schedule_digest(
                record.exposure_end_ns - record.exposure_start_ns,
                &record.illumination_intervals,
            )
            .unwrap(),
            record.illumination_schedule_digest
        );
        let overlapping = vec![
            MycelialIlluminationInterval {
                channel_id: "blue".to_string(),
                start_offset_ns: 0,
                end_offset_ns: 600,
            },
            MycelialIlluminationInterval {
                channel_id: "blue".to_string(),
                start_offset_ns: 500,
                end_offset_ns: 750,
            },
        ];
        assert!(matches!(
            mycelial_illumination_schedule_digest(1_000, &overlapping),
            Err(MycelialPhotobiologicalExposureError::InvalidExposure(_))
        ));
    }

    #[test]
    fn schedule_interval_cannot_reference_an_unknown_channel() {
        let mut record = exposure();
        record.illumination_intervals[0].channel_id = "near_ir".to_string();
        record.illumination_intervals.sort_by(|left, right| {
            (&left.channel_id, left.start_offset_ns, left.end_offset_ns).cmp(&(
                &right.channel_id,
                right.start_offset_ns,
                right.end_offset_ns,
            ))
        });
        record.illumination_schedule_digest = mycelial_illumination_schedule_digest(
            record.exposure_end_ns - record.exposure_start_ns,
            &record.illumination_intervals,
        )
        .unwrap();
        assert!(matches!(
            record.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidChannel(_))
        ));
    }

    #[test]
    fn channel_summary_must_match_exact_interval_inventory() {
        let mut record = exposure();
        record.channels[0].continuous_illumination_duration_ns = 500;
        assert!(matches!(
            record.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidChannel(_))
        ));
    }

    #[test]
    fn endpoint_types_remain_distinct() {
        let cord = MycelialPhotobiologicalMeasurement::CordAreaFraction {
            fraction: 0.4,
            sampled_roi_area_square_micrometers: 1_000.0,
            measurement_method_id: "cord-mask-v1".to_string(),
        };
        let viability = MycelialPhotobiologicalMeasurement::ViabilityFraction {
            fraction: 0.4,
            assay_id: "viability-assay-v1".to_string(),
        };
        validate_measurement(&cord).unwrap();
        validate_measurement(&viability).unwrap();

        let mut first = blake3::Hasher::new();
        let mut second = blake3::Hasher::new();
        hash_measurement(&mut first, &cord);
        hash_measurement(&mut second, &viability);
        assert_ne!(
            first.finalize().to_hex().to_string(),
            second.finalize().to_hex().to_string()
        );
    }

    #[test]
    fn dark_control_is_explicit_and_does_not_fake_an_illumination_channel() {
        let mut record = exposure();
        record.illumination_purpose = MycelialIlluminationPurpose::DarkAmbientControl;
        assert!(matches!(
            record.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidExposure(_))
        ));
        record.channels.clear();
        record.illumination_intervals.clear();
        record.illumination_schedule_digest = mycelial_illumination_schedule_digest(
            record.exposure_end_ns - record.exposure_start_ns,
            &record.illumination_intervals,
        )
        .unwrap();
        record.validate().unwrap();
    }

    #[test]
    fn photoperiod_phase_requires_a_consistent_transition_clock() {
        let mut record = exposure();
        record.photoperiod_phase = MycelialPhotoperiodPhase::Unknown;
        record.elapsed_since_photoperiod_transition_ns = None;
        record.validate().unwrap();

        let mut inconsistent = exposure();
        inconsistent.elapsed_since_photoperiod_transition_ns = None;
        assert!(matches!(
            inconsistent.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidExposure(_))
        ));
    }

    #[test]
    fn schedule_digest_changes_the_canonical_exposure_identity() {
        let first = exposure();
        let mut second = exposure();
        second.illumination_intervals[0].start_offset_ns = 1;
        second.illumination_intervals[0].end_offset_ns = 251;
        second.illumination_schedule_digest = mycelial_illumination_schedule_digest(
            second.exposure_end_ns - second.exposure_start_ns,
            &second.illumination_intervals,
        )
        .unwrap();
        assert_ne!(
            first.canonical_digest_hex().unwrap(),
            second.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn water_activity_and_temperature_evidence_must_be_jointly_present() {
        let mut record = exposure();
        record.hydration_evidence_digest = None;
        assert!(matches!(
            record.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidExposure(_))
        ));
        let mut record = exposure();
        record.specimen_temperature_millikelvin = None;
        assert!(matches!(
            record.validate(),
            Err(MycelialPhotobiologicalExposureError::InvalidExposure(_))
        ));
    }

    #[test]
    fn unknown_purpose_is_preserved_not_relabelled_as_imaging() {
        let mut record = exposure();
        record.illumination_purpose = MycelialIlluminationPurpose::UnknownPurpose;
        assert_eq!(
            record.canonical_digest_hex().unwrap().len(),
            DIGEST_HEX_BYTES
        );
    }
}
