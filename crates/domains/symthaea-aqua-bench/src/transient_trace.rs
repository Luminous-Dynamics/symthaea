//! Provenance-bound admission for Luminous Aqua pressure-transient event traces.
//!
//! This module strictly decodes a JSON event manifest and then admits decoded channel samples.
//! It does not authenticate external issuers, calibrate sensors, authorize a pressure rig,
//! calculate hydraulic states, or qualify bathing-task performance. Those remain with the
//! schema/decoder, Mycelix metrology, competent apparatus authority, Symtropy and independent
//! task/laboratory assessors respectively.
//!
//! The module is deliberately fail-closed: malformed, incomplete, unsynchronized or
//! non-comparable traces cannot be promoted to a valid event replay.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use chrono::DateTime;
use serde::Deserialize;

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_text(field: &'static str, value: &str) -> Result<(), TransientTraceError> {
    if value.trim().is_empty() {
        return Err(TransientTraceError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), TransientTraceError> {
    if !is_sha256(value) {
        return Err(TransientTraceError::InvalidDigest { field });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileKind {
    SiteObserved,
    Synthetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventFamily {
    SupplySag,
    SupplyRamp,
    ConcurrentDemandStep,
    IntermittentSupply,
    RepeatedDemandCycling,
    RapidValveEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampBasis {
    Monotonic,
    UtcDisciplined,
    Ptp,
    OtherDeclared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum CommandKind {
    #[serde(rename = "upstream_pressure_command")]
    UpstreamPressure,
    #[serde(rename = "branch_demand_command")]
    BranchDemand,
    #[serde(rename = "valve_state_command")]
    ValveState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum CommandUnit {
    #[serde(rename = "kPa")]
    Kpa,
    #[serde(rename = "L/min")]
    LitresPerMinute,
    #[serde(rename = "boolean")]
    Boolean,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum CommandValue {
    Numeric(f64),
    Boolean(bool),
}

impl CommandValue {
    fn numeric(self) -> Option<f64> {
        match self {
            Self::Numeric(value) => Some(value),
            Self::Boolean(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandSample {
    /// Microseconds from the beginning of the event capture window.
    pub offset_us: u64,
    pub value: CommandValue,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventCommand {
    #[serde(rename = "input_kind")]
    pub kind: CommandKind,
    pub channel_id: String,
    pub unit: CommandUnit,
    /// Digest of the canonical command waveform and its exact units.
    pub waveform_sha256: String,
    pub samples: Vec<CommandSample>,
}

impl EventCommand {
    fn validate(&self, family: EventFamily, duration_us: u64) -> Result<(), TransientTraceError> {
        require_text("command.channel_id", &self.channel_id)?;
        require_digest("command.waveform_sha256", &self.waveform_sha256)?;

        let compatible = match family {
            EventFamily::SupplySag | EventFamily::SupplyRamp | EventFamily::IntermittentSupply => {
                self.kind == CommandKind::UpstreamPressure
            }
            EventFamily::ConcurrentDemandStep => self.kind == CommandKind::BranchDemand,
            EventFamily::RepeatedDemandCycling => {
                matches!(self.kind, CommandKind::BranchDemand | CommandKind::ValveState)
            }
            EventFamily::RapidValveEvent => self.kind == CommandKind::ValveState,
        };
        if !compatible {
            return Err(TransientTraceError::CommandFamilyMismatch);
        }
        if self.samples.len() < 2 {
            return Err(TransientTraceError::InsufficientCommandSamples);
        }

        let mut previous_offset = None;
        for sample in &self.samples {
            if sample.offset_us > duration_us {
                return Err(TransientTraceError::CommandOutsideEvent);
            }
            if previous_offset.is_some_and(|previous| sample.offset_us <= previous) {
                return Err(TransientTraceError::NonMonotonicCommandTime);
            }
            previous_offset = Some(sample.offset_us);

            match (self.kind, self.unit, sample.value) {
                (CommandKind::UpstreamPressure, CommandUnit::Kpa, CommandValue::Numeric(value))
                | (CommandKind::BranchDemand, CommandUnit::LitresPerMinute, CommandValue::Numeric(value)) => {
                    if !value.is_finite() || value < 0.0 {
                        return Err(TransientTraceError::InvalidCommandValue);
                    }
                }
                (CommandKind::ValveState, CommandUnit::Boolean, CommandValue::Boolean(_)) => {}
                _ => return Err(TransientTraceError::CommandUnitMismatch),
            }
        }

        let transition_count = self
            .samples
            .windows(2)
            .filter(|pair| pair[0].value != pair[1].value)
            .count();
        let shape_valid = match (family, self.kind) {
            (EventFamily::SupplySag, CommandKind::UpstreamPressure) => self
                .samples
                .windows(2)
                .any(|pair| matches!((&pair[0].value, &pair[1].value),
                    (CommandValue::Numeric(before), CommandValue::Numeric(after)) if after < before)),
            (EventFamily::SupplyRamp, CommandKind::UpstreamPressure) => {
                let values: Vec<f64> = self.samples.iter().filter_map(|sample| {
                    sample.value.numeric()
                }).collect();
                let nondecreasing = values.windows(2).all(|pair| pair[1] >= pair[0]);
                let nonincreasing = values.windows(2).all(|pair| pair[1] <= pair[0]);
                transition_count > 0 && (nondecreasing || nonincreasing)
            }
            (EventFamily::IntermittentSupply, CommandKind::UpstreamPressure) => self
                .samples
                .iter()
                .any(|sample| matches!(sample.value, CommandValue::Numeric(value) if value == 0.0)),
            (EventFamily::ConcurrentDemandStep, CommandKind::BranchDemand) => self
                .samples
                .windows(2)
                .any(|pair| matches!((&pair[0].value, &pair[1].value),
                    (CommandValue::Numeric(before), CommandValue::Numeric(after)) if after > before)),
            (EventFamily::RepeatedDemandCycling, _) => transition_count >= 2,
            (EventFamily::RapidValveEvent, CommandKind::ValveState) => transition_count >= 1,
            _ => false,
        };
        if !shape_valid {
            return Err(TransientTraceError::CommandShapeMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationWindow {
    pub start: String,
    pub end: String,
    pub coverage_statement: String,
    #[serde(default)]
    pub known_gaps: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteObservationEvidence {
    pub site_profile_id: String,
    pub site_profile_revision: u32,
    pub site_conditions_sha256: String,
    pub survey_receipt_sha256: String,
    pub observation_window: ObservationWindow,
}

impl SiteObservationEvidence {
    fn validate(&self) -> Result<(), TransientTraceError> {
        require_text("site_profile_id", &self.site_profile_id)?;
        require_text("observation_window.start", &self.observation_window.start)?;
        require_text("observation_window.end", &self.observation_window.end)?;
        require_text("observation_window.coverage_statement", &self.observation_window.coverage_statement)?;
        let start = DateTime::parse_from_rfc3339(&self.observation_window.start)
            .map_err(|_| TransientTraceError::InvalidTimestamp { field: "observation_window.start" })?;
        let end = DateTime::parse_from_rfc3339(&self.observation_window.end)
            .map_err(|_| TransientTraceError::InvalidTimestamp { field: "observation_window.end" })?;
        if end <= start {
            return Err(TransientTraceError::InvalidTimestamp { field: "observation_window.end" });
        }
        if self.site_profile_revision == 0 {
            return Err(TransientTraceError::InvalidRevision { field: "site_profile_revision" });
        }
        require_digest("site_conditions_sha256", &self.site_conditions_sha256)?;
        require_digest("survey_receipt_sha256", &self.survey_receipt_sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticProfileEvidence {
    pub generator_id: String,
    pub generator_sha256: String,
    pub generator_parameters_sha256: String,
}

impl SyntheticProfileEvidence {
    fn validate(&self) -> Result<(), TransientTraceError> {
        require_text("generator_id", &self.generator_id)?;
        require_digest("generator_sha256", &self.generator_sha256)?;
        require_digest("generator_parameters_sha256", &self.generator_parameters_sha256)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelRole {
    UpstreamStaticPressure,
    UpstreamResidualPressure,
    TerminalDynamicPressure,
    PointOfDischargePressure,
    HotInletPressure,
    ColdInletPressure,
    FaceplateCenterPressure,
    BranchPressure,
    PrimaryFlow,
    SecondaryBranchFlow,
    OutletTemperature,
}

impl ChannelRole {
    fn is_pressure(self) -> bool {
        matches!(
            self,
            Self::UpstreamStaticPressure
                | Self::UpstreamResidualPressure
                | Self::TerminalDynamicPressure
                | Self::PointOfDischargePressure
                | Self::HotInletPressure
                | Self::ColdInletPressure
                | Self::FaceplateCenterPressure
                | Self::BranchPressure
        )
    }

    fn is_flow(self) -> bool {
        matches!(self, Self::PrimaryFlow | Self::SecondaryBranchFlow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelDatum {
    StaticSupply,
    UpstreamServicePoint,
    TerminalFittingInlet,
    PointOfDischarge,
    HotMixerInlet,
    ColdMixerInlet,
    FaceplateCenter,
    BranchPoint,
    FixtureOutlet,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ChannelUnit {
    #[serde(rename = "kPa")]
    Kpa,
    #[serde(rename = "L/min")]
    LitresPerMinute,
    #[serde(rename = "degC")]
    DegreesCelsius,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelQuality {
    Valid,
    Clipped,
    Saturated,
    OutOfRange,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceSample {
    /// Microseconds from the beginning of the event capture window.
    pub offset_us: u64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceChannel {
    pub channel_id: String,
    pub role: ChannelRole,
    pub location: String,
    pub datum: ChannelDatum,
    pub unit: ChannelUnit,
    pub instrument_id: String,
    pub calibration_receipt_sha256: String,
    pub raw_trace_sha256: String,
    pub measurement_uncertainty: f64,
    #[serde(rename = "quality_state")]
    pub quality: ChannelQuality,
    pub sample_rate_hz: f64,
    pub effective_bandwidth_hz: f64,
    pub anti_alias_filter_id: String,
    pub sample_count: usize,
    pub dropped_sample_count: usize,
    pub sample_gap_report_sha256: Option<String>,
    /// Populated by the raw-trace decoder after the manifest is read.
    #[serde(skip)]
    pub samples: Vec<TraceSample>,
}

impl TraceChannel {
    fn validate(&self, duration_us: u64) -> Result<(), TransientTraceError> {
        for (field, value) in [
            ("channel_id", self.channel_id.as_str()),
            ("location", self.location.as_str()),
            ("instrument_id", self.instrument_id.as_str()),
            ("anti_alias_filter_id", self.anti_alias_filter_id.as_str()),
        ] {
            require_text(field, value)?;
        }
        require_digest("calibration_receipt_sha256", &self.calibration_receipt_sha256)?;
        require_digest("raw_trace_sha256", &self.raw_trace_sha256)?;

        if self.quality != ChannelQuality::Valid {
            return Err(TransientTraceError::ChannelNotUsable {
                channel_id: self.channel_id.clone(),
                quality: self.quality,
            });
        }
        if !self.measurement_uncertainty.is_finite() || self.measurement_uncertainty < 0.0 {
            return Err(TransientTraceError::InvalidChannelMeasurement {
                channel_id: self.channel_id.clone(),
                field: "measurement_uncertainty",
            });
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(TransientTraceError::InvalidChannelSampling {
                channel_id: self.channel_id.clone(),
                field: "sample_rate_hz",
            });
        }
        if !self.effective_bandwidth_hz.is_finite()
            || self.effective_bandwidth_hz <= 0.0
            || self.effective_bandwidth_hz * 2.0 >= self.sample_rate_hz
        {
            return Err(TransientTraceError::InvalidChannelSampling {
                channel_id: self.channel_id.clone(),
                field: "effective_bandwidth_hz",
            });
        }
        if self.sample_count < 2 || self.sample_count != self.samples.len() {
            return Err(TransientTraceError::SampleCountMismatch {
                channel_id: self.channel_id.clone(),
            });
        }
        if self.dropped_sample_count > 0 {
            let digest = self.sample_gap_report_sha256.as_deref().ok_or_else(|| {
                TransientTraceError::MissingSampleGapReceipt {
                    channel_id: self.channel_id.clone(),
                }
            })?;
            require_digest("sample_gap_report_sha256", digest)?;
        } else if let Some(digest) = self.sample_gap_report_sha256.as_deref() {
            require_digest("sample_gap_report_sha256", digest)?;
        }

        self.validate_role_datum_unit()?;

        let mut previous_offset = None;
        for sample in &self.samples {
            if sample.offset_us > duration_us {
                return Err(TransientTraceError::SampleOutsideEvent {
                    channel_id: self.channel_id.clone(),
                });
            }
            if previous_offset.is_some_and(|previous| sample.offset_us <= previous) {
                return Err(TransientTraceError::NonMonotonicSampleTime {
                    channel_id: self.channel_id.clone(),
                });
            }
            if !sample.value.is_finite() {
                return Err(TransientTraceError::NonFiniteSample {
                    channel_id: self.channel_id.clone(),
                });
            }
            if self.role.is_pressure() || self.role.is_flow() {
                if sample.value < 0.0 {
                    return Err(TransientTraceError::NegativePhysicalSample {
                        channel_id: self.channel_id.clone(),
                    });
                }
            }
            previous_offset = Some(sample.offset_us);
        }
        Ok(())
    }

    fn validate_role_datum_unit(&self) -> Result<(), TransientTraceError> {
        let (expected_datum, expected_unit) = match self.role {
            ChannelRole::UpstreamStaticPressure => (ChannelDatum::StaticSupply, ChannelUnit::Kpa),
            ChannelRole::UpstreamResidualPressure => {
                (ChannelDatum::UpstreamServicePoint, ChannelUnit::Kpa)
            }
            ChannelRole::TerminalDynamicPressure => {
                (ChannelDatum::TerminalFittingInlet, ChannelUnit::Kpa)
            }
            ChannelRole::PointOfDischargePressure => {
                (ChannelDatum::PointOfDischarge, ChannelUnit::Kpa)
            }
            ChannelRole::HotInletPressure => (ChannelDatum::HotMixerInlet, ChannelUnit::Kpa),
            ChannelRole::ColdInletPressure => (ChannelDatum::ColdMixerInlet, ChannelUnit::Kpa),
            ChannelRole::FaceplateCenterPressure => {
                (ChannelDatum::FaceplateCenter, ChannelUnit::Kpa)
            }
            ChannelRole::BranchPressure => (ChannelDatum::BranchPoint, ChannelUnit::Kpa),
            ChannelRole::PrimaryFlow | ChannelRole::SecondaryBranchFlow => {
                (ChannelDatum::NotApplicable, ChannelUnit::LitresPerMinute)
            }
            ChannelRole::OutletTemperature => {
                (ChannelDatum::FixtureOutlet, ChannelUnit::DegreesCelsius)
            }
        };
        if self.datum != expected_datum || self.unit != expected_unit {
            return Err(TransientTraceError::RoleDatumUnitMismatch {
                channel_id: self.channel_id.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionMetadata {
    pub clock_id: String,
    pub timestamp_basis: TimestampBasis,
    pub clock_sync_uncertainty_us: f64,
    /// Maximum synchronization uncertainty fixed by the test protocol before the run.
    pub maximum_clock_sync_uncertainty_us: f64,
    /// Maximum allowed nearest-sample skew between any channel and the reference channel.
    pub maximum_channel_alignment_skew_us: u64,
    pub clock_sync_receipt_sha256: String,
    pub channels: Vec<TraceChannel>,
}

impl AcquisitionMetadata {
    fn validate(&self, duration_us: u64) -> Result<(), TransientTraceError> {
        require_text("acquisition.clock_id", &self.clock_id)?;
        require_digest("clock_sync_receipt_sha256", &self.clock_sync_receipt_sha256)?;
        if !self.clock_sync_uncertainty_us.is_finite()
            || self.clock_sync_uncertainty_us < 0.0
            || !self.maximum_clock_sync_uncertainty_us.is_finite()
            || self.maximum_clock_sync_uncertainty_us < 0.0
            || self.clock_sync_uncertainty_us > self.maximum_clock_sync_uncertainty_us
        {
            return Err(TransientTraceError::ClockSynchronizationUnqualified);
        }
        if self.channels.len() < 2 {
            return Err(TransientTraceError::InsufficientChannels);
        }

        let mut channel_ids = HashSet::with_capacity(self.channels.len());
        for channel in &self.channels {
            if !channel_ids.insert(channel.channel_id.as_str()) {
                return Err(TransientTraceError::DuplicateChannelId {
                    channel_id: channel.channel_id.clone(),
                });
            }
            channel.validate(duration_us)?;
        }
        self.validate_channel_alignment()?;
        Ok(())
    }

    fn validate_channel_alignment(&self) -> Result<(), TransientTraceError> {
        let Some(reference) = self.channels.first() else {
            return Err(TransientTraceError::InsufficientChannels);
        };
        for channel in self.channels.iter().skip(1) {
            for sample in &reference.samples {
                let nearest = nearest_sample_distance(&channel.samples, sample.offset_us);
                if nearest > self.maximum_channel_alignment_skew_us {
                    return Err(TransientTraceError::ChannelAlignmentExceeded {
                        reference_channel_id: reference.channel_id.clone(),
                        channel_id: channel.channel_id.clone(),
                    });
                }
            }
            for sample in &channel.samples {
                let nearest = nearest_sample_distance(&reference.samples, sample.offset_us);
                if nearest > self.maximum_channel_alignment_skew_us {
                    return Err(TransientTraceError::ChannelAlignmentExceeded {
                        reference_channel_id: reference.channel_id.clone(),
                        channel_id: channel.channel_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

fn nearest_sample_distance(samples: &[TraceSample], target_us: u64) -> u64 {
    match samples.binary_search_by_key(&target_us, |sample| sample.offset_us) {
        Ok(_) => 0,
        Err(index) => {
            let after = samples.get(index).map(|sample| sample.offset_us);
            let before = index
                .checked_sub(1)
                .and_then(|previous| samples.get(previous).map(|sample| sample.offset_us));
            match (before, after) {
                (Some(left), Some(right)) => {
                    target_us.saturating_sub(left).min(right.saturating_sub(target_us))
                }
                (Some(left), None) => target_us.saturating_sub(left),
                (None, Some(right)) => right.saturating_sub(target_us),
                (None, None) => u64::MAX,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawArtifactRole {
    UnfilteredTrace,
    EventLog,
    CalibrationBundle,
    ClockSyncReceipt,
    SiteObservationRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawArtifact {
    pub role: RawArtifactRole,
    pub sha256: String,
    pub byte_length: u64,
}

impl RawArtifact {
    fn validate(&self) -> Result<(), TransientTraceError> {
        require_digest("raw_artifact.sha256", &self.sha256)?;
        if self.byte_length == 0 {
            return Err(TransientTraceError::EmptyRawArtifact);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMetadata {
    pub analysis_code_sha256: Option<String>,
    pub filter_definition_sha256: Option<String>,
    pub exclusion_manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransientEventProfile {
    pub schema_version: String,
    pub profile_id: String,
    pub profile_revision: u32,
    pub profile_kind: ProfileKind,
    pub event_family: EventFamily,
    pub task_profile_id: String,
    pub conditions_sha256: String,
    pub protocol_sha256: String,
    /// RFC3339 event-start timestamp, checked against the manifest schema date-time format.
    pub event_start: String,
    pub event_duration_ms: u64,
    pub site_observation: Option<SiteObservationEvidence>,
    pub synthetic_profile: Option<SyntheticProfileEvidence>,
    pub command: EventCommand,
    pub acquisition: AcquisitionMetadata,
    pub raw_artifacts: Vec<RawArtifact>,
    pub analysis: Option<AnalysisMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransientTraceError {
    MissingField { field: &'static str },
    InvalidProfileId,
    ChannelSamplesAlreadyAttached { channel_id: String },
    InvalidDigest { field: &'static str },
    InvalidRevision { field: &'static str },
    InvalidEventDuration,
    ContradictoryProfileProvenance,
    MissingProfileProvenance,
    MissingUnfilteredTrace,
    MissingEventLog,
    MissingSiteObservationArtifact,
    MissingTraceArtifactReference { channel_id: String },
    TraceArtifactNotFound { channel_id: String },
    DuplicateArtifactRole { role: RawArtifactRole },
    EmptyRawArtifact,
    CommandFamilyMismatch,
    CommandUnitMismatch,
    CommandShapeMismatch,
    InvalidCommandValue,
    InsufficientCommandSamples,
    NonMonotonicCommandTime,
    CommandOutsideEvent,
    ClockSynchronizationUnqualified,
    InsufficientChannels,
    DuplicateChannelId { channel_id: String },
    ChannelNotUsable { channel_id: String, quality: ChannelQuality },
    InvalidChannelMeasurement { channel_id: String, field: &'static str },
    InvalidChannelSampling { channel_id: String, field: &'static str },
    SampleCountMismatch { channel_id: String },
    MissingSampleGapReceipt { channel_id: String },
    RoleDatumUnitMismatch { channel_id: String },
    NonMonotonicSampleTime { channel_id: String },
    SampleOutsideEvent { channel_id: String },
    NonFiniteSample { channel_id: String },
    NegativePhysicalSample { channel_id: String },
    ChannelAlignmentExceeded { reference_channel_id: String, channel_id: String },
    MissingPressureChannel,
    MissingPrimaryFlowChannel,
    MissingSecondaryFlowChannel,
    InsufficientSpatialPressureChannels,
    EventReplayMismatch { field: &'static str },
    ManifestDecode { message: String },
    InvalidTimestamp { field: &'static str },
    UnknownChannel { channel_id: String },
}

impl Display for TransientTraceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing transient trace field: {field}"),
            Self::InvalidProfileId => write!(f, "profile ID must be a 2-128 character lowercase slug"),
            Self::ChannelSamplesAlreadyAttached { channel_id } => write!(f, "raw samples are already attached for channel: {channel_id}"),
            Self::InvalidDigest { field } => write!(f, "invalid SHA-256 digest for: {field}"),
            Self::InvalidRevision { field } => write!(f, "invalid profile revision: {field}"),
            Self::InvalidEventDuration => write!(f, "event duration must be finite and greater than zero"),
            Self::ContradictoryProfileProvenance => {
                write!(f, "site-observed and synthetic profile provenance cannot be combined")
            }
            Self::MissingProfileProvenance => write!(f, "profile provenance is missing or incomplete"),
            Self::MissingUnfilteredTrace => write!(f, "raw unfiltered trace artifact is missing"),
            Self::MissingEventLog => write!(f, "event log artifact is missing"),
            Self::MissingSiteObservationArtifact => {
                write!(f, "site-observed profile lacks its source observation artifact")
            }
            Self::MissingTraceArtifactReference { channel_id } => {
                write!(f, "channel {channel_id} does not identify its unfiltered raw-trace artifact")
            }
            Self::TraceArtifactNotFound { channel_id } => {
                write!(f, "channel {channel_id} references a raw-trace digest absent from the manifest")
            }
            Self::DuplicateArtifactRole { role } => write!(f, "duplicate raw artifact role: {role:?}"),
            Self::EmptyRawArtifact => write!(f, "raw artifact byte length must be greater than zero"),
            Self::CommandFamilyMismatch => write!(f, "command type is incompatible with event family"),
            Self::CommandUnitMismatch => write!(f, "command sample value does not match its declared unit"),
            Self::CommandShapeMismatch => write!(f, "command waveform does not exhibit the declared event-family behavior"),
            Self::InvalidCommandValue => write!(f, "command value must be finite and non-negative"),
            Self::InsufficientCommandSamples => write!(f, "event command needs at least two samples"),
            Self::NonMonotonicCommandTime => write!(f, "command timestamps must be strictly increasing"),
            Self::CommandOutsideEvent => write!(f, "command sample lies outside the event capture window"),
            Self::ClockSynchronizationUnqualified => {
                write!(f, "clock synchronization uncertainty exceeds the preregistered limit")
            }
            Self::InsufficientChannels => write!(f, "event acquisition needs at least two channels"),
            Self::DuplicateChannelId { channel_id } => write!(f, "duplicate channel identifier: {channel_id}"),
            Self::ChannelNotUsable { channel_id, quality } => {
                write!(f, "channel {channel_id} is not valid: {quality:?}")
            }
            Self::InvalidChannelMeasurement { channel_id, field } => {
                write!(f, "invalid measurement metadata for {channel_id}: {field}")
            }
            Self::InvalidChannelSampling { channel_id, field } => {
                write!(f, "invalid sampling metadata for {channel_id}: {field}")
            }
            Self::SampleCountMismatch { channel_id } => {
                write!(f, "sample count does not match decoded samples for {channel_id}")
            }
            Self::MissingSampleGapReceipt { channel_id } => {
                write!(f, "dropped samples lack a gap report for {channel_id}")
            }
            Self::RoleDatumUnitMismatch { channel_id } => {
                write!(f, "channel role, pressure datum and unit disagree for {channel_id}")
            }
            Self::NonMonotonicSampleTime { channel_id } => {
                write!(f, "trace sample timestamps are not strictly increasing for {channel_id}")
            }
            Self::SampleOutsideEvent { channel_id } => {
                write!(f, "trace sample lies outside the event capture window for {channel_id}")
            }
            Self::NonFiniteSample { channel_id } => {
                write!(f, "trace sample is not finite for {channel_id}")
            }
            Self::NegativePhysicalSample { channel_id } => {
                write!(f, "pressure or flow sample is negative for {channel_id}")
            }
            Self::ChannelAlignmentExceeded { reference_channel_id, channel_id } => {
                write!(f, "channel alignment exceeds tolerance: {reference_channel_id} vs {channel_id}")
            }
            Self::MissingPressureChannel => write!(f, "event trace has no pressure channel"),
            Self::MissingPrimaryFlowChannel => write!(f, "event trace has no primary flow channel"),
            Self::MissingSecondaryFlowChannel => write!(f, "concurrent-demand event lacks secondary branch flow"),
            Self::InsufficientSpatialPressureChannels => {
                write!(f, "rapid-valve event requires pressure at two distinct plumbing locations")
            }
            Self::EventReplayMismatch { field } => {
                write!(f, "matched event replays differ at: {field}")
            }
            Self::ManifestDecode { message } => write!(f, "transient event manifest decode failed: {message}"),
            Self::InvalidTimestamp { field } => write!(f, "invalid or reversed RFC3339 timestamp: {field}"),
            Self::UnknownChannel { channel_id } => write!(f, "raw trace references an unknown channel: {channel_id}"),
        }
    }
}

impl Error for TransientTraceError {}

impl TransientEventProfile {
    /// Strictly decode a versioned JSON manifest. Raw channel samples are attached separately
    /// from the referenced unfiltered trace artifact before full admission.
    pub fn from_json_manifest(json: &str) -> Result<Self, TransientTraceError> {
        serde_json::from_str(json)
            .map_err(|error| TransientTraceError::ManifestDecode {
                message: error.to_string(),
            })
    }

    /// Attach decoded samples to the exact manifest channel named by the raw-trace decoder.
    pub fn attach_channel_samples(
        &mut self,
        channel_id: &str,
        samples: Vec<TraceSample>,
    ) -> Result<(), TransientTraceError> {
        let channel = self
            .acquisition
            .channels
            .iter_mut()
            .find(|channel| channel.channel_id == channel_id)
            .ok_or_else(|| TransientTraceError::UnknownChannel {
                channel_id: channel_id.to_owned(),
            })?;
        if !channel.samples.is_empty() {
            return Err(TransientTraceError::ChannelSamplesAlreadyAttached {
                channel_id: channel.channel_id.clone(),
            });
        }
        if channel.sample_count != samples.len() {
            return Err(TransientTraceError::SampleCountMismatch {
                channel_id: channel.channel_id.clone(),
            });
        }
        channel.samples = samples;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), TransientTraceError> {
        require_text("schema_version", &self.schema_version)?;
        if self.schema_version != "0.1.0" {
            return Err(TransientTraceError::InvalidRevision { field: "schema_version" });
        }
        require_text("profile_id", &self.profile_id)?;
        if self.profile_id.len() < 2
            || self.profile_id.len() > 128
            || !self.profile_id.as_bytes()[0].is_ascii_lowercase()
                && !self.profile_id.as_bytes()[0].is_ascii_digit()
            || !self.profile_id.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.' || byte == b'_' || byte == b'-'
            })
        {
            return Err(TransientTraceError::InvalidProfileId);
        }
        require_text("task_profile_id", &self.task_profile_id)?;
        require_text("event_start", &self.event_start)?;
        DateTime::parse_from_rfc3339(&self.event_start)
            .map_err(|_| TransientTraceError::InvalidTimestamp { field: "event_start" })?;
        require_digest("conditions_sha256", &self.conditions_sha256)?;
        require_digest("protocol_sha256", &self.protocol_sha256)?;
        if self.profile_revision == 0 {
            return Err(TransientTraceError::InvalidRevision { field: "profile_revision" });
        }
        if self.event_duration_ms == 0 {
            return Err(TransientTraceError::InvalidEventDuration);
        }
        let duration_us = self
            .event_duration_ms
            .checked_mul(1_000)
            .ok_or(TransientTraceError::InvalidEventDuration)?;

        match self.profile_kind {
            ProfileKind::SiteObserved => {
                if self.synthetic_profile.is_some() {
                    return Err(TransientTraceError::ContradictoryProfileProvenance);
                }
                self.site_observation
                    .as_ref()
                    .ok_or(TransientTraceError::MissingProfileProvenance)?
                    .validate()?;
            }
            ProfileKind::Synthetic => {
                if self.site_observation.is_some() {
                    return Err(TransientTraceError::ContradictoryProfileProvenance);
                }
                self.synthetic_profile
                    .as_ref()
                    .ok_or(TransientTraceError::MissingProfileProvenance)?
                    .validate()?;
            }
        }

        self.command.validate(self.event_family, duration_us)?;
        self.acquisition.validate(duration_us)?;
        self.validate_required_channels()?;
        self.validate_artifacts()?;
        Ok(())
    }

    fn validate_required_channels(&self) -> Result<(), TransientTraceError> {
        let channels = &self.acquisition.channels;
        if !channels.iter().any(|channel| channel.role.is_pressure()) {
            return Err(TransientTraceError::MissingPressureChannel);
        }
        if !channels.iter().any(|channel| channel.role == ChannelRole::PrimaryFlow) {
            return Err(TransientTraceError::MissingPrimaryFlowChannel);
        }
        match self.event_family {
            EventFamily::ConcurrentDemandStep => {
                if !channels.iter().any(|channel| channel.role == ChannelRole::SecondaryBranchFlow) {
                    return Err(TransientTraceError::MissingSecondaryFlowChannel);
                }
            }
            EventFamily::RapidValveEvent => {
                let mut locations = HashSet::new();
                for channel in channels.iter().filter(|channel| channel.role.is_pressure()) {
                    locations.insert(channel.location.as_str());
                }
                if locations.len() < 2 {
                    return Err(TransientTraceError::InsufficientSpatialPressureChannels);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_artifacts(&self) -> Result<(), TransientTraceError> {
        // A raw trace may be sharded by channel or time window. Other artifact roles are
        // singleton references so their identity cannot be silently ambiguous.
        let mut singleton_roles = HashSet::with_capacity(self.raw_artifacts.len());
        let mut roles = HashSet::with_capacity(self.raw_artifacts.len());
        let mut unfiltered_trace_digests = HashSet::new();
        for artifact in &self.raw_artifacts {
            artifact.validate()?;
            roles.insert(artifact.role);
            if artifact.role == RawArtifactRole::UnfilteredTrace {
                unfiltered_trace_digests.insert(artifact.sha256.as_str());
            }
            if artifact.role != RawArtifactRole::UnfilteredTrace
                && !singleton_roles.insert(artifact.role)
            {
                return Err(TransientTraceError::DuplicateArtifactRole { role: artifact.role });
            }
        }
        if !roles.contains(&RawArtifactRole::UnfilteredTrace) {
            return Err(TransientTraceError::MissingUnfilteredTrace);
        }
        if !roles.contains(&RawArtifactRole::EventLog) {
            return Err(TransientTraceError::MissingEventLog);
        }
        if self.profile_kind == ProfileKind::SiteObserved
            && !roles.contains(&RawArtifactRole::SiteObservationRecord)
        {
            return Err(TransientTraceError::MissingSiteObservationArtifact);
        }
        for channel in &self.acquisition.channels {
            if !is_sha256(&channel.raw_trace_sha256) {
                return Err(TransientTraceError::MissingTraceArtifactReference {
                    channel_id: channel.channel_id.clone(),
                });
            }
            if !unfiltered_trace_digests.contains(channel.raw_trace_sha256.as_str()) {
                return Err(TransientTraceError::TraceArtifactNotFound {
                    channel_id: channel.channel_id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Admit two runs as the same event replay before comparing their measured response.
///
/// Device-specific specimen/configuration and instrument identities may differ, but the
/// external event waveform, task, shared conditions, protocol, pressure datum set and
/// acquisition characteristics must remain matched. This function does not claim that
/// both physical rigs actually reproduced the same hydraulic boundary conditions; the
/// independent apparatus / metrology evidence must establish that.
pub fn validate_matched_event_replay(
    control: &TransientEventProfile,
    candidate: &TransientEventProfile,
) -> Result<(), TransientTraceError> {
    control.validate()?;
    candidate.validate()?;

    for (field, left, right) in [
        ("event_family", format!("{:?}", control.event_family), format!("{:?}", candidate.event_family)),
        ("profile_kind", format!("{:?}", control.profile_kind), format!("{:?}", candidate.profile_kind)),
        ("task_profile_id", control.task_profile_id.clone(), candidate.task_profile_id.clone()),
        ("conditions_sha256", control.conditions_sha256.clone(), candidate.conditions_sha256.clone()),
        ("protocol_sha256", control.protocol_sha256.clone(), candidate.protocol_sha256.clone()),
        ("command.waveform_sha256", control.command.waveform_sha256.clone(), candidate.command.waveform_sha256.clone()),
        ("command.channel_id", control.command.channel_id.clone(), candidate.command.channel_id.clone()),
    ] {
        if left != right {
            return Err(TransientTraceError::EventReplayMismatch { field });
        }
    }
    if control.event_duration_ms != candidate.event_duration_ms {
        return Err(TransientTraceError::EventReplayMismatch { field: "event_duration_ms" });
    }
    if control.command.kind != candidate.command.kind {
        return Err(TransientTraceError::EventReplayMismatch { field: "command.kind" });
    }
    if control.command.unit != candidate.command.unit {
        return Err(TransientTraceError::EventReplayMismatch { field: "command.unit" });
    }
    // Event-log byte digests are run-specific: actual trigger times and capture metadata
    // may differ. The matched command waveform and command samples above define the replay;
    // each run separately retains its own event-log artifact for audit.
    if control.command.samples != candidate.command.samples {
        return Err(TransientTraceError::EventReplayMismatch { field: "command.samples" });
    }
    if control.acquisition.timestamp_basis != candidate.acquisition.timestamp_basis {
        return Err(TransientTraceError::EventReplayMismatch { field: "timestamp_basis" });
    }
    if control.acquisition.maximum_channel_alignment_skew_us
        != candidate.acquisition.maximum_channel_alignment_skew_us
    {
        return Err(TransientTraceError::EventReplayMismatch {
            field: "maximum_channel_alignment_skew_us",
        });
    }
    if control.acquisition.maximum_clock_sync_uncertainty_us
        != candidate.acquisition.maximum_clock_sync_uncertainty_us
    {
        return Err(TransientTraceError::EventReplayMismatch {
            field: "maximum_clock_sync_uncertainty_us",
        });
    }

    let mut control_channels: Vec<_> = control
        .acquisition
        .channels
        .iter()
        .map(channel_signature)
        .collect();
    let mut candidate_channels: Vec<_> = candidate
        .acquisition
        .channels
        .iter()
        .map(channel_signature)
        .collect();
    control_channels.sort();
    candidate_channels.sort();
    if control_channels != candidate_channels {
        return Err(TransientTraceError::EventReplayMismatch { field: "channel_signatures" });
    }

    match (&control.site_observation, &candidate.site_observation) {
        (Some(left), Some(right)) => {
            if left.site_profile_id != right.site_profile_id
                || left.site_profile_revision != right.site_profile_revision
                || left.site_conditions_sha256 != right.site_conditions_sha256
                || left.survey_receipt_sha256 != right.survey_receipt_sha256
                || left.observation_window != right.observation_window
                || artifact_digest(control, RawArtifactRole::SiteObservationRecord)
                    != artifact_digest(candidate, RawArtifactRole::SiteObservationRecord)
            {
                return Err(TransientTraceError::EventReplayMismatch { field: "site_observation" });
            }
        }
        (None, None) => {
            let left = control.synthetic_profile.as_ref().expect("validated synthetic profile");
            let right = candidate.synthetic_profile.as_ref().expect("validated synthetic profile");
            if left.generator_id != right.generator_id
                || left.generator_sha256 != right.generator_sha256
                || left.generator_parameters_sha256 != right.generator_parameters_sha256
            {
                return Err(TransientTraceError::EventReplayMismatch { field: "synthetic_generator" });
            }
        }
        _ => return Err(TransientTraceError::EventReplayMismatch { field: "profile_provenance" }),
    }
    Ok(())
}

fn artifact_digest(profile: &TransientEventProfile, role: RawArtifactRole) -> Option<&str> {
    profile
        .raw_artifacts
        .iter()
        .find(|artifact| artifact.role == role)
        .map(|artifact| artifact.sha256.as_str())
}

fn channel_signature(
    channel: &TraceChannel,
) -> (String, String, String, String, String, String, u64, u64, u64, u64, u64) {
    (
        channel.channel_id.clone(),
        format!("{:?}", channel.role),
        channel.location.clone(),
        format!("{:?}", channel.datum),
        format!("{:?}", channel.unit),
        channel.anti_alias_filter_id.clone(),
        channel.sample_rate_hz.to_bits(),
        channel.effective_bandwidth_hz.to_bits(),
        channel.measurement_uncertainty.to_bits(),
        channel.sample_count as u64,
        channel.dropped_sample_count as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn samples(value: f64) -> Vec<TraceSample> {
        vec![
            TraceSample { offset_us: 0, value },
            TraceSample { offset_us: 1_000, value },
            TraceSample { offset_us: 2_000, value },
        ]
    }

    fn pressure_channel(id: &str, location: &str) -> TraceChannel {
        TraceChannel {
            channel_id: id.to_owned(),
            role: ChannelRole::TerminalDynamicPressure,
            location: location.to_owned(),
            datum: ChannelDatum::TerminalFittingInlet,
            unit: ChannelUnit::Kpa,
            instrument_id: format!("sensor-{id}"),
            calibration_receipt_sha256: digest('a'),
            raw_trace_sha256: digest('3'),
            measurement_uncertainty: 1.0,
            quality: ChannelQuality::Valid,
            sample_rate_hz: 1_000.0,
            effective_bandwidth_hz: 400.0,
            anti_alias_filter_id: "aa-filter-v1".to_owned(),
            sample_count: 3,
            dropped_sample_count: 0,
            sample_gap_report_sha256: None,
            samples: samples(250.0),
        }
    }

    fn flow_channel() -> TraceChannel {
        TraceChannel {
            channel_id: "primary-flow".to_owned(),
            role: ChannelRole::PrimaryFlow,
            location: "shower-outlet".to_owned(),
            datum: ChannelDatum::NotApplicable,
            unit: ChannelUnit::LitresPerMinute,
            instrument_id: "flow-meter-1".to_owned(),
            calibration_receipt_sha256: digest('b'),
            raw_trace_sha256: digest('3'),
            measurement_uncertainty: 0.05,
            quality: ChannelQuality::Valid,
            sample_rate_hz: 1_000.0,
            effective_bandwidth_hz: 400.0,
            anti_alias_filter_id: "aa-filter-v1".to_owned(),
            sample_count: 3,
            dropped_sample_count: 0,
            sample_gap_report_sha256: None,
            samples: samples(5.0),
        }
    }

    fn profile() -> TransientEventProfile {
        TransientEventProfile {
            schema_version: "0.1.0".to_owned(),
            profile_id: "synthetic-sag-001".to_owned(),
            profile_revision: 1,
            profile_kind: ProfileKind::Synthetic,
            event_family: EventFamily::SupplySag,
            task_profile_id: "full-shower-v1".to_owned(),
            conditions_sha256: digest('c'),
            protocol_sha256: digest('d'),
            event_start: "2026-11-19T00:00:00Z".to_owned(),
            event_duration_ms: 2,
            site_observation: None,
            synthetic_profile: Some(SyntheticProfileEvidence {
                generator_id: "event-generator-v1".to_owned(),
                generator_sha256: digest('e'),
                generator_parameters_sha256: digest('f'),
            }),
            command: EventCommand {
                kind: CommandKind::UpstreamPressure,
                channel_id: "upstream-pressure-command".to_owned(),
                unit: CommandUnit::Kpa,
                waveform_sha256: digest('1'),
                samples: vec![
                    CommandSample { offset_us: 0, value: CommandValue::Numeric(300.0) },
                    CommandSample { offset_us: 1_000, value: CommandValue::Numeric(200.0) },
                    CommandSample { offset_us: 2_000, value: CommandValue::Numeric(300.0) },
                ],
            },
            acquisition: AcquisitionMetadata {
                clock_id: "ptp-clock-1".to_owned(),
                timestamp_basis: TimestampBasis::Ptp,
                clock_sync_uncertainty_us: 2.0,
                maximum_clock_sync_uncertainty_us: 5.0,
                maximum_channel_alignment_skew_us: 0,
                clock_sync_receipt_sha256: digest('2'),
                channels: vec![pressure_channel("terminal-pressure", "terminal-outlet"), flow_channel()],
            },
            raw_artifacts: vec![
                RawArtifact { role: RawArtifactRole::UnfilteredTrace, sha256: digest('3'), byte_length: 10_000 },
                RawArtifact { role: RawArtifactRole::EventLog, sha256: digest('4'), byte_length: 1_024 },
            ],
            analysis: None,
        }
    }

    #[test]
    fn decodes_json_manifest_and_attaches_trace_samples() {
        let digest = |character: char| character.to_string().repeat(64);
        let channel = |channel_id: &str,
                       role: &str,
                       location: &str,
                       datum: &str,
                       unit: &str,
                       instrument_id: &str,
                       calibration: char,
                       uncertainty: f64| {
            serde_json::json!({
                "channel_id": channel_id,
                "quality_state": "valid",
                "role": role,
                "location": location,
                "datum": datum,
                "unit": unit,
                "instrument_id": instrument_id,
                "calibration_receipt_sha256": digest(calibration),
                "raw_trace_sha256": digest('3'),
                "measurement_uncertainty": uncertainty,
                "sample_rate_hz": 1000.0,
                "effective_bandwidth_hz": 400.0,
                "anti_alias_filter_id": "aa-filter-v1",
                "sample_count": 3,
                "dropped_sample_count": 0
            })
        };
        let manifest = serde_json::json!({
            "schema_version": "0.1.0",
            "profile_id": "synthetic-sag-json-001",
            "profile_revision": 1,
            "profile_kind": "synthetic",
            "event_family": "supply_sag",
            "task_profile_id": "full-shower-v1",
            "conditions_sha256": digest('c'),
            "protocol_sha256": digest('d'),
            "event_start": "2026-11-19T00:00:00Z",
            "event_duration_ms": 2,
            "synthetic_profile": {
                "generator_id": "event-generator-v1",
                "generator_sha256": digest('e'),
                "generator_parameters_sha256": digest('f')
            },
            "command": {
                "input_kind": "upstream_pressure_command",
                "channel_id": "upstream-pressure-command",
                "unit": "kPa",
                "waveform_sha256": digest('1'),
                "samples": [
                    {"offset_us": 0, "value": 300.0},
                    {"offset_us": 1000, "value": 200.0},
                    {"offset_us": 2000, "value": 300.0}
                ]
            },
            "acquisition": {
                "clock_id": "ptp-clock-1",
                "timestamp_basis": "ptp",
                "clock_sync_uncertainty_us": 2.0,
                "maximum_clock_sync_uncertainty_us": 5.0,
                "maximum_channel_alignment_skew_us": 0,
                "clock_sync_receipt_sha256": digest('2'),
                "channels": [
                    channel("terminal-pressure", "terminal_dynamic_pressure", "terminal-outlet",
                        "terminal_fitting_inlet", "kPa", "pressure-1", 'a', 1.0),
                    channel("primary-flow", "primary_flow", "shower-outlet",
                        "not_applicable", "L/min", "flow-1", 'b', 0.05)
                ]
            },
            "raw_artifacts": [
                {"role": "unfiltered_trace", "sha256": digest('3'), "byte_length": 10000},
                {"role": "event_log", "sha256": digest('4'), "byte_length": 1024}
            ]
        });

        let mut decoded = TransientEventProfile::from_json_manifest(&manifest.to_string())
            .expect("valid manifest decodes");
        decoded
            .attach_channel_samples("terminal-pressure", samples(250.0))
            .expect("pressure samples attach to declared channel");
        decoded
            .attach_channel_samples("primary-flow", samples(5.0))
            .expect("flow samples attach to declared channel");

        assert_eq!(decoded.validate(), Ok(()));
    }

    #[test]
    fn strict_json_decoder_rejects_unknown_manifest_fields() {
        let mut manifest = serde_json::json!({
            "schema_version": "0.1.0",
            "profile_id": "strict-test",
            "profile_revision": 1
        });
        manifest["unexpected_authority_override"] = serde_json::json!(true);
        assert!(matches!(
            TransientEventProfile::from_json_manifest(&manifest.to_string()),
            Err(TransientTraceError::ManifestDecode { .. })
        ));
    }

    #[test]
    fn failed_sample_attachment_does_not_mutate_channel_state() {
        let mut value = profile();
        value.acquisition.channels[0].samples.clear();

        assert_eq!(
            value.attach_channel_samples(
                "terminal-pressure",
                vec![
                    TraceSample { offset_us: 0, value: 1.0 },
                    TraceSample { offset_us: 1_000, value: 2.0 },
                ],
            ),
            Err(TransientTraceError::SampleCountMismatch {
                channel_id: "terminal-pressure".to_owned()
            })
        );
        assert!(value.acquisition.channels[0].samples.is_empty());
    }

    #[test]
    fn rejects_channel_trace_digest_absent_from_raw_artifact_manifest() {
        let mut value = profile();
        value.acquisition.channels[0].raw_trace_sha256 = digest('9');
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::TraceArtifactNotFound {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn raw_trace_decoder_cannot_attach_samples_to_unknown_channel() {
        let mut value = profile();
        assert_eq!(
            value.attach_channel_samples("not-a-channel", samples(1.0)),
            Err(TransientTraceError::UnknownChannel {
                channel_id: "not-a-channel".to_owned()
            })
        );
    }

    #[test]
    fn accepts_complete_synthetic_supply_sag_profile() {
        assert_eq!(profile().validate(), Ok(()));
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let mut value = profile();
        value.schema_version = "0.2.0".to_owned();
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::InvalidRevision { field: "schema_version" })
        );
    }

    #[test]
    fn rejects_missing_synthetic_generator_provenance() {
        let mut value = profile();
        value.synthetic_profile = None;
        assert_eq!(value.validate(), Err(TransientTraceError::MissingProfileProvenance));
    }

    #[test]
    fn rejects_combined_site_and_synthetic_provenance() {
        let mut value = profile();
        value.site_observation = Some(SiteObservationEvidence {
            site_profile_id: "site-01".to_owned(),
            site_profile_revision: 1,
            site_conditions_sha256: digest('5'),
            survey_receipt_sha256: digest('6'),
            observation_window: ObservationWindow {
                start: "2026-01-01T00:00:00Z".to_owned(),
                end: "2026-01-01T00:10:00Z".to_owned(),
                coverage_statement: "complete".to_owned(),
                known_gaps: Vec::new(),
            },
        });
        assert_eq!(value.validate(), Err(TransientTraceError::ContradictoryProfileProvenance));
    }

    #[test]
    fn rejects_bad_clock_sync_uncertainty() {
        let mut value = profile();
        value.acquisition.clock_sync_uncertainty_us = 10.0;
        assert_eq!(value.validate(), Err(TransientTraceError::ClockSynchronizationUnqualified));
    }

    #[test]
    fn rejects_wrong_role_datum_and_unit_combination() {
        let mut value = profile();
        value.acquisition.channels[0].datum = ChannelDatum::PointOfDischarge;
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::RoleDatumUnitMismatch {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_clipped_or_unknown_sensor_channels() {
        for quality in [ChannelQuality::Clipped, ChannelQuality::Saturated, ChannelQuality::OutOfRange, ChannelQuality::Unknown] {
            let mut value = profile();
            value.acquisition.channels[0].quality = quality;
            assert!(matches!(value.validate(), Err(TransientTraceError::ChannelNotUsable { .. })));
        }
    }

    #[test]
    fn rejects_bandwidth_at_or_above_nyquist_limit() {
        let mut value = profile();
        value.acquisition.channels[0].effective_bandwidth_hz = 500.0;
        assert!(matches!(value.validate(), Err(TransientTraceError::InvalidChannelSampling { .. })));
    }

    #[test]
    fn rejects_sample_count_that_does_not_match_decoded_trace() {
        let mut value = profile();
        value.acquisition.channels[0].sample_count = 4;
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::SampleCountMismatch {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_dropped_samples_without_gap_report() {
        let mut value = profile();
        value.acquisition.channels[0].dropped_sample_count = 1;
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::MissingSampleGapReceipt {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_non_monotonic_trace_timestamps() {
        let mut value = profile();
        value.acquisition.channels[0].samples[2].offset_us = 1_000;
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::NonMonotonicSampleTime {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_samples_outside_capture_window() {
        let mut value = profile();
        value.acquisition.channels[0].samples[2].offset_us = 2_001;
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::SampleOutsideEvent {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_negative_pressure_or_flow_samples() {
        let mut value = profile();
        value.acquisition.channels[0].samples[1].value = -0.1;
        assert!(matches!(value.validate(), Err(TransientTraceError::NegativePhysicalSample { .. })));
    }

    #[test]
    fn rejects_channel_alignment_outside_preregistered_skew() {
        let mut value = profile();
        value.acquisition.maximum_channel_alignment_skew_us = 100;
        value.acquisition.channels[1].samples[1].offset_us = 1_500;
        assert!(matches!(value.validate(), Err(TransientTraceError::ChannelAlignmentExceeded { .. })));
    }

    #[test]
    fn requires_secondary_flow_for_concurrent_demand_step() {
        let mut value = profile();
        value.event_family = EventFamily::ConcurrentDemandStep;
        value.command.kind = CommandKind::BranchDemand;
        value.command.unit = CommandUnit::LitresPerMinute;
        value.command.samples.iter_mut().enumerate().for_each(|(index, sample)| {
            sample.value = CommandValue::Numeric(if index == 0 { 0.0 } else { 3.0 });
        });
        assert_eq!(value.validate(), Err(TransientTraceError::MissingSecondaryFlowChannel));
    }

    #[test]
    fn requires_spatial_pressure_channels_for_rapid_valve_event() {
        let mut value = profile();
        value.event_family = EventFamily::RapidValveEvent;
        value.command.kind = CommandKind::ValveState;
        value.command.unit = CommandUnit::Boolean;
        value.command.samples[0].value = CommandValue::Boolean(false);
        value.command.samples[1].value = CommandValue::Boolean(true);
        value.command.samples[2].value = CommandValue::Boolean(false);
        assert_eq!(value.validate(), Err(TransientTraceError::InsufficientSpatialPressureChannels));
    }

    #[test]
    fn accepts_rapid_valve_event_with_multiple_pressure_locations() {
        let mut value = profile();
        value.event_family = EventFamily::RapidValveEvent;
        value.command.kind = CommandKind::ValveState;
        value.command.unit = CommandUnit::Boolean;
        value.command.samples[0].value = CommandValue::Boolean(false);
        value.command.samples[1].value = CommandValue::Boolean(true);
        value.command.samples[2].value = CommandValue::Boolean(false);
        value.acquisition.channels.insert(1, pressure_channel("adjacent-branch-pressure", "adjacent-terminal-branch"));
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn rejects_supply_sag_label_without_a_pressure_decrease() {
        let mut value = profile();
        value.command.samples.iter_mut().for_each(|sample| {
            sample.value = CommandValue::Numeric(300.0);
        });
        assert_eq!(value.validate(), Err(TransientTraceError::CommandShapeMismatch));
    }

    #[test]
    fn rejects_intermittent_supply_without_a_zero_supply_interval() {
        let mut value = profile();
        value.event_family = EventFamily::IntermittentSupply;
        value.command.samples.iter_mut().for_each(|sample| {
            sample.value = CommandValue::Numeric(150.0);
        });
        assert_eq!(value.validate(), Err(TransientTraceError::CommandShapeMismatch));
    }

    #[test]
    fn rejects_valve_event_without_a_state_transition() {
        let mut value = profile();
        value.event_family = EventFamily::RapidValveEvent;
        value.command.kind = CommandKind::ValveState;
        value.command.unit = CommandUnit::Boolean;
        value.command.samples.iter_mut().for_each(|sample| {
            sample.value = CommandValue::Boolean(true);
        });
        assert_eq!(value.validate(), Err(TransientTraceError::CommandShapeMismatch));
    }

    #[test]
    fn rejects_command_type_that_does_not_match_event_family() {
        let mut value = profile();
        value.command.kind = CommandKind::ValveState;
        assert_eq!(value.validate(), Err(TransientTraceError::CommandFamilyMismatch));
    }

    #[test]
    fn rejects_duplicate_channel_ids() {
        let mut value = profile();
        let mut duplicate = pressure_channel("terminal-pressure", "another-location");
        duplicate.role = ChannelRole::BranchPressure;
        duplicate.datum = ChannelDatum::BranchPoint;
        value.acquisition.channels.push(duplicate);
        assert_eq!(
            value.validate(),
            Err(TransientTraceError::DuplicateChannelId {
                channel_id: "terminal-pressure".to_owned()
            })
        );
    }

    #[test]
    fn rejects_missing_unfiltered_trace_or_event_log() {
        let mut value = profile();
        value.raw_artifacts.retain(|artifact| artifact.role != RawArtifactRole::UnfilteredTrace);
        assert_eq!(value.validate(), Err(TransientTraceError::MissingUnfilteredTrace));

        let mut value = profile();
        value.raw_artifacts.retain(|artifact| artifact.role != RawArtifactRole::EventLog);
        assert_eq!(value.validate(), Err(TransientTraceError::MissingEventLog));
    }

    #[test]
    fn rejects_site_observed_profile_without_source_observation_record() {
        let mut value = profile();
        value.profile_kind = ProfileKind::SiteObserved;
        value.synthetic_profile = None;
        value.site_observation = Some(SiteObservationEvidence {
            site_profile_id: "site-01".to_owned(),
            site_profile_revision: 1,
            site_conditions_sha256: digest('5'),
            survey_receipt_sha256: digest('6'),
            observation_window: ObservationWindow {
                start: "2026-01-01T00:00:00Z".to_owned(),
                end: "2026-01-01T00:10:00Z".to_owned(),
                coverage_statement: "complete".to_owned(),
                known_gaps: Vec::new(),
            },
        });
        assert_eq!(value.validate(), Err(TransientTraceError::MissingSiteObservationArtifact));
    }

    #[test]
    fn rejects_matched_replay_with_different_waveform_digest() {
        let control = profile();
        let mut candidate = profile();
        candidate.command.waveform_sha256 = digest('7');
        assert_eq!(
            validate_matched_event_replay(&control, &candidate),
            Err(TransientTraceError::EventReplayMismatch { field: "command.waveform_sha256" })
        );
    }

    #[test]
    fn accepts_matched_replay_when_event_and_acquisition_contracts_match() {
        let control = profile();
        let mut candidate = profile();
        candidate.profile_id = "candidate-sag-001".to_owned();
        assert_eq!(validate_matched_event_replay(&control, &candidate), Ok(()));
    }

    #[test]
    fn matched_replay_allows_distinct_sensor_ids_when_method_contract_matches() {
        let control = profile();
        let mut candidate = profile();
        candidate.profile_id = "candidate-sag-001".to_owned();
        candidate.acquisition.channels[0].instrument_id = "independent-pressure-meter".to_owned();
        candidate.acquisition.channels[0].calibration_receipt_sha256 = digest('8');
        candidate.acquisition.channels[1].instrument_id = "independent-flow-meter".to_owned();
        candidate.acquisition.channels[1].calibration_receipt_sha256 = digest('9');

        assert_eq!(validate_matched_event_replay(&control, &candidate), Ok(()));
    }

    #[test]
    fn rejects_matched_replay_with_different_channel_signatures() {
        let control = profile();
        let mut candidate = profile();
        candidate.acquisition.channels[0].location = "different-terminal".to_owned();
        assert_eq!(
            validate_matched_event_replay(&control, &candidate),
            Err(TransientTraceError::EventReplayMismatch { field: "channel_signatures" })
        );
    }
}
