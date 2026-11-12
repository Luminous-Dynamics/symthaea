// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent evidence-contract validator for rotor aeroacoustic runs.
//!
//! This crate validates identities, units, calibration references, microphone
//! geometry, rotor phase/speed references, time-window coverage and declared
//! environmental conditions. It does not process waveform samples, calculate
//! SPL, perform beamforming, predict noise or establish certification evidence.
//!
//! All fixtures are synthetic. A valid bundle means only that the declared
//! metadata is internally consistent under this contract.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

pub const MODEL_VERSION: &str = "airroad-acoustic-evidence-oracle-v0.1.0";
const UNIT_VECTOR_TOLERANCE: f64 = 1.0e-3;

/// Aircraft/rotor operating regime for one synchronized measurement run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    IsolatedRotor,
    InstalledHover,
    Climb,
    Descent,
    Transition,
    EdgewiseFlow,
    ForwardFlight,
}

/// Physical direction of rotor rotation, viewed in the declared rotor frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationDirection {
    Clockwise,
    CounterClockwise,
}

/// Raw channel role. Rotor-associated roles name the exact rotor identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelKind {
    Microphone {
        microphone_id: String,
        array_id: String,
    },
    RotorPhase {
        rotor_id: String,
    },
    RotorSpeed {
        rotor_id: String,
    },
    RotorThrust {
        rotor_id: String,
    },
    RotorTorque {
        rotor_id: String,
    },
    DcBusVoltage,
    DcBusCurrent,
    Accelerometer {
        sensor_id: String,
    },
}

/// Explicit engineering unit for one raw acquisition channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelUnit {
    Pascal,
    Radian,
    RevolutionsPerMinute,
    Newton,
    NewtonMetre,
    Volt,
    Ampere,
    MetresPerSecondSquared,
}

/// Immutable identities and the exact analysis time window for one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticRunManifest {
    pub run_id: String,
    pub asset_id: String,
    pub configuration_digest: String,
    pub geometry_digest: String,
    pub context_digest: String,
    pub procedure_revision: String,
    pub evidence_manifest_digest: String,
    pub coordinate_frame_id: String,
    pub reference_clock_id: String,
    pub analysis_start_time_ns: i64,
    pub analysis_end_time_ns: i64,
    pub maximum_clock_uncertainty_ns: u64,
    pub maximum_channel_skew_ns: u64,
}

/// One acquired channel. Raw samples remain in external immutable artifacts;
/// this record binds the artifact and its acquisition metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticChannel {
    pub channel_id: String,
    pub kind: ChannelKind,
    pub unit: ChannelUnit,
    pub calibration_id: String,
    pub calibration_certificate_digest: String,
    pub raw_data_digest: String,
    pub clock_source_id: String,
    pub clock_offset_to_reference_ns: i64,
    pub clock_uncertainty_ns: u64,
    pub start_time_ns: i64,
    pub end_time_ns: i64,
    pub sample_rate_hz: f64,
    pub sample_count: u64,
}

/// Microphone position and outward-facing unit boresight in one exact frame.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrophonePosition {
    pub microphone_id: String,
    pub channel_id: String,
    pub array_id: String,
    pub coordinate_frame_id: String,
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
    pub boresight_x: f64,
    pub boresight_y: f64,
    pub boresight_z: f64,
    pub position_uncertainty_m: f64,
}

/// Rotor hardware/configuration identity needed to interpret phase and spectra.
#[derive(Debug, Clone, PartialEq)]
pub struct RotorDeclaration {
    pub rotor_id: String,
    pub geometry_digest: String,
    pub blade_count: u32,
    pub diameter_m: f64,
    pub rotation_direction: RotationDirection,
    pub phase_channel_id: String,
    pub speed_channel_id: String,
}

/// Synchronized aerodynamic operating point for one declared rotor.
#[derive(Debug, Clone, PartialEq)]
pub struct RotorOperatingPoint {
    pub rotor_id: String,
    pub rpm: f64,
    pub thrust_n: f64,
    /// Signed according to the declared shaft convention.
    pub shaft_torque_nm: f64,
    pub blade_pitch_deg: Option<f64>,
    pub tilt_deg: Option<f64>,
}

/// Measured environmental conditions associated with this run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticEnvironment {
    pub air_density_kg_m3: f64,
    pub static_pressure_pa: f64,
    pub temperature_k: f64,
    pub relative_humidity_fraction: f64,
    pub free_stream_speed_m_s: f64,
    pub turbulence_intensity_fraction: f64,
}

/// Complete metadata bundle for one acoustic measurement run.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticRunBundle {
    pub manifest: AcousticRunManifest,
    pub channels: Vec<AcousticChannel>,
    pub microphone_positions: Vec<MicrophonePosition>,
    pub rotors: Vec<RotorDeclaration>,
    pub rotor_operating_points: Vec<RotorOperatingPoint>,
    pub environment: AcousticEnvironment,
}

/// Typed non-PASS outcomes. Validation may return more than one finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcousticValidationError {
    EmptyIdentity(&'static str),
    InvalidSha256Digest(&'static str),
    InvalidAnalysisWindow,
    InvalidClockLimit,
    DuplicateChannelId,
    DuplicateMicrophoneId,
    DuplicateMicrophoneChannel,
    DuplicateRotorId,
    DuplicateRotorOperatingPoint,
    MissingChannels,
    MissingMicrophoneArray,
    MicrophoneArrayTooSmall(String),
    MissingMicrophonePosition(String),
    UnexpectedMicrophonePosition(String),
    MissingRotorOperatingPoint(String),
    UnexpectedRotorOperatingPoint(String),
    MissingRotorPhaseChannel(String),
    MissingRotorSpeedChannel(String),
    ChannelKindMismatch(String),
    ChannelUnitMismatch(String),
    InvalidChannelTimeWindow(String),
    ChannelDoesNotCoverAnalysisWindow(String),
    ClockUncertaintyExceeded(String),
    ChannelSkewExceeded(String),
    InvalidSampleRate(String),
    InvalidSampleCount(String),
    InvalidMicrophoneGeometry(String),
    CoordinateFrameMismatch(String),
    InvalidRotorGeometry(String),
    InvalidRotorOperatingPoint(String),
    TransitionMissingTilt(String),
    InvalidEnvironment(&'static str),
}

/// Summary returned only when the complete metadata bundle validates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticValidationSummary {
    pub model_version: &'static str,
    pub channel_count: usize,
    pub microphone_count: usize,
    pub microphone_array_count: usize,
    pub rotor_count: usize,
    pub analysis_duration_ns: u64,
}

/// Validate the complete bundle. Missing, incompatible or unsupported evidence
/// is a typed error; it is never replaced with a nominal or zero value.
pub fn validate_bundle(
    bundle: &AcousticRunBundle,
) -> Result<AcousticValidationSummary, Vec<AcousticValidationError>> {
    let mut errors = Vec::new();
    validate_manifest(&bundle.manifest, &mut errors);
    validate_environment(bundle.environment, &mut errors);

    if bundle.channels.is_empty() {
        errors.push(AcousticValidationError::MissingChannels);
    }

    let mut channels_by_id: HashMap<&str, &AcousticChannel> = HashMap::new();
    let mut microphone_channel_ids = HashSet::new();

    for channel in &bundle.channels {
        if channel.channel_id.trim().is_empty() {
            errors.push(AcousticValidationError::EmptyIdentity("channel_id"));
        } else if channels_by_id
            .insert(channel.channel_id.as_str(), channel)
            .is_some()
        {
            errors.push(AcousticValidationError::DuplicateChannelId);
        }

        validate_channel(channel, &bundle.manifest, &mut errors);

        if matches!(channel.kind, ChannelKind::Microphone { .. }) {
            microphone_channel_ids.insert(channel.channel_id.as_str());
        }
    }

    let mut microphone_ids = HashSet::new();
    let mut positioned_microphone_channels = HashSet::new();
    let mut array_sizes: HashMap<&str, usize> = HashMap::new();

    for position in &bundle.microphone_positions {
        if position.microphone_id.trim().is_empty()
            || position.channel_id.trim().is_empty()
            || position.array_id.trim().is_empty()
        {
            errors.push(AcousticValidationError::EmptyIdentity(
                "microphone geometry identity",
            ));
        }

        if !microphone_ids.insert(position.microphone_id.as_str()) {
            errors.push(AcousticValidationError::DuplicateMicrophoneId);
        }
        if !positioned_microphone_channels.insert(position.channel_id.as_str()) {
            errors.push(AcousticValidationError::DuplicateMicrophoneChannel);
        }

        if position.coordinate_frame_id != bundle.manifest.coordinate_frame_id {
            errors.push(AcousticValidationError::CoordinateFrameMismatch(
                position.microphone_id.clone(),
            ));
        }

        if !valid_microphone_geometry(position) {
            errors.push(AcousticValidationError::InvalidMicrophoneGeometry(
                position.microphone_id.clone(),
            ));
        }

        match channels_by_id.get(position.channel_id.as_str()) {
            Some(channel) => match &channel.kind {
                ChannelKind::Microphone {
                    microphone_id,
                    array_id,
                } if microphone_id == &position.microphone_id
                    && array_id == &position.array_id => {}
                _ => errors.push(AcousticValidationError::ChannelKindMismatch(
                    position.channel_id.clone(),
                )),
            },
            None => errors.push(AcousticValidationError::MissingMicrophonePosition(
                position.microphone_id.clone(),
            )),
        }

        *array_sizes.entry(position.array_id.as_str()).or_insert(0) += 1;
    }

    if array_sizes.is_empty() {
        errors.push(AcousticValidationError::MissingMicrophoneArray);
    }
    for (array_id, count) in &array_sizes {
        if *count < 2 {
            errors.push(AcousticValidationError::MicrophoneArrayTooSmall(
                (*array_id).to_owned(),
            ));
        }
    }

    for channel_id in &microphone_channel_ids {
        if !positioned_microphone_channels.contains(channel_id) {
            errors.push(AcousticValidationError::UnexpectedMicrophonePosition(
                (*channel_id).to_owned(),
            ));
        }
    }

    validate_rotors(
        bundle,
        &channels_by_id,
        &mut errors,
    );

    if errors.is_empty() {
        let duration = bundle
            .manifest
            .analysis_end_time_ns
            .checked_sub(bundle.manifest.analysis_start_time_ns)
            .and_then(|value| u64::try_from(value).ok())
            .unwrap_or(0);

        Ok(AcousticValidationSummary {
            model_version: MODEL_VERSION,
            channel_count: bundle.channels.len(),
            microphone_count: bundle.microphone_positions.len(),
            microphone_array_count: array_sizes.len(),
            rotor_count: bundle.rotors.len(),
            analysis_duration_ns: duration,
        })
    } else {
        Err(errors)
    }
}

fn validate_manifest(
    manifest: &AcousticRunManifest,
    errors: &mut Vec<AcousticValidationError>,
) {
    for (name, value) in [
        ("run_id", manifest.run_id.as_str()),
        ("asset_id", manifest.asset_id.as_str()),
        ("procedure_revision", manifest.procedure_revision.as_str()),
        ("coordinate_frame_id", manifest.coordinate_frame_id.as_str()),
        ("reference_clock_id", manifest.reference_clock_id.as_str()),
    ] {
        if value.trim().is_empty() {
            errors.push(AcousticValidationError::EmptyIdentity(name));
        }
    }

    for (name, digest) in [
        ("configuration_digest", manifest.configuration_digest.as_str()),
        ("geometry_digest", manifest.geometry_digest.as_str()),
        ("context_digest", manifest.context_digest.as_str()),
        (
            "evidence_manifest_digest",
            manifest.evidence_manifest_digest.as_str(),
        ),
    ] {
        if !is_sha256_digest(digest) {
            errors.push(AcousticValidationError::InvalidSha256Digest(name));
        }
    }

    if manifest.analysis_start_time_ns >= manifest.analysis_end_time_ns {
        errors.push(AcousticValidationError::InvalidAnalysisWindow);
    }
    if manifest.maximum_clock_uncertainty_ns == 0
        || manifest.maximum_channel_skew_ns == 0
    {
        errors.push(AcousticValidationError::InvalidClockLimit);
    }
}

fn validate_channel(
    channel: &AcousticChannel,
    manifest: &AcousticRunManifest,
    errors: &mut Vec<AcousticValidationError>,
) {
    if channel.calibration_id.trim().is_empty() {
        errors.push(AcousticValidationError::EmptyIdentity("calibration_id"));
    }
    if !is_sha256_digest(&channel.calibration_certificate_digest) {
        errors.push(AcousticValidationError::InvalidSha256Digest(
            "calibration_certificate_digest",
        ));
    }
    if !is_sha256_digest(&channel.raw_data_digest) {
        errors.push(AcousticValidationError::InvalidSha256Digest(
            "raw_data_digest",
        ));
    }
    if channel.clock_source_id.trim().is_empty() {
        errors.push(AcousticValidationError::EmptyIdentity(
            "channel clock_source_id",
        ));
    }
    if channel.start_time_ns >= channel.end_time_ns {
        errors.push(AcousticValidationError::InvalidChannelTimeWindow(
            channel.channel_id.clone(),
        ));
        return;
    }

    if channel.clock_uncertainty_ns > manifest.maximum_clock_uncertainty_ns {
        errors.push(AcousticValidationError::ClockUncertaintyExceeded(
            channel.channel_id.clone(),
        ));
    }

    let clock_budget = channel
        .clock_offset_to_reference_ns
        .unsigned_abs()
        .saturating_add(channel.clock_uncertainty_ns);
    if clock_budget > manifest.maximum_channel_skew_ns {
        errors.push(AcousticValidationError::ChannelSkewExceeded(
            channel.channel_id.clone(),
        ));
    }

    let adjusted_start = channel
        .start_time_ns
        .checked_add(channel.clock_offset_to_reference_ns);
    let adjusted_end = channel
        .end_time_ns
        .checked_add(channel.clock_offset_to_reference_ns);

    match (adjusted_start, adjusted_end) {
        (Some(start), Some(end))
            if start <= manifest.analysis_start_time_ns
                && end >= manifest.analysis_end_time_ns => {}
        _ => errors.push(AcousticValidationError::ChannelDoesNotCoverAnalysisWindow(
            channel.channel_id.clone(),
        )),
    }

    if !channel.sample_rate_hz.is_finite() || channel.sample_rate_hz <= 0.0 {
        errors.push(AcousticValidationError::InvalidSampleRate(
            channel.channel_id.clone(),
        ));
    }
    if channel.sample_count < 2 {
        errors.push(AcousticValidationError::InvalidSampleCount(
            channel.channel_id.clone(),
        ));
    }

    if !unit_matches_kind(channel.unit, &channel.kind) {
        errors.push(AcousticValidationError::ChannelUnitMismatch(
            channel.channel_id.clone(),
        ));
    }
}

fn unit_matches_kind(unit: ChannelUnit, kind: &ChannelKind) -> bool {
    matches!(
        (kind, unit),
        (ChannelKind::Microphone { .. }, ChannelUnit::Pascal)
            | (ChannelKind::RotorPhase { .. }, ChannelUnit::Radian)
            | (
                ChannelKind::RotorSpeed { .. },
                ChannelUnit::RevolutionsPerMinute
            )
            | (ChannelKind::RotorThrust { .. }, ChannelUnit::Newton)
            | (ChannelKind::RotorTorque { .. }, ChannelUnit::NewtonMetre)
            | (ChannelKind::DcBusVoltage, ChannelUnit::Volt)
            | (ChannelKind::DcBusCurrent, ChannelUnit::Ampere)
            | (
                ChannelKind::Accelerometer { .. },
                ChannelUnit::MetresPerSecondSquared
            )
    )
}

fn valid_microphone_geometry(position: &MicrophonePosition) -> bool {
    let values = [
        position.x_m,
        position.y_m,
        position.z_m,
        position.boresight_x,
        position.boresight_y,
        position.boresight_z,
        position.position_uncertainty_m,
    ];
    if values.iter().any(|value| !value.is_finite())
        || position.position_uncertainty_m < 0.0
    {
        return false;
    }

    let norm = position.boresight_x * position.boresight_x
        + position.boresight_y * position.boresight_y
        + position.boresight_z * position.boresight_z;
    norm.is_finite() && (norm.sqrt() - 1.0).abs() <= UNIT_VECTOR_TOLERANCE
}

fn validate_rotors(
    bundle: &AcousticRunBundle,
    channels: &HashMap<&str, &AcousticChannel>,
    errors: &mut Vec<AcousticValidationError>,
) {
    let mut rotor_ids = HashSet::new();
    let mut operating_points: HashMap<&str, &RotorOperatingPoint> = HashMap::new();

    for point in &bundle.rotor_operating_points {
        if operating_points
            .insert(point.rotor_id.as_str(), point)
            .is_some()
        {
            errors.push(AcousticValidationError::DuplicateRotorOperatingPoint);
        }
    }

    for rotor in &bundle.rotors {
        if rotor.rotor_id.trim().is_empty() {
            errors.push(AcousticValidationError::EmptyIdentity("rotor_id"));
        }
        if !rotor_ids.insert(rotor.rotor_id.as_str()) {
            errors.push(AcousticValidationError::DuplicateRotorId);
        }
        if !is_sha256_digest(&rotor.geometry_digest) {
            errors.push(AcousticValidationError::InvalidSha256Digest(
                "rotor geometry_digest",
            ));
        }
        if rotor.blade_count == 0
            || !rotor.diameter_m.is_finite()
            || rotor.diameter_m <= 0.0
        {
            errors.push(AcousticValidationError::InvalidRotorGeometry(
                rotor.rotor_id.clone(),
            ));
        }

        match channels.get(rotor.phase_channel_id.as_str()) {
            Some(channel)
                if matches!(
                    &channel.kind,
                    ChannelKind::RotorPhase { rotor_id } if rotor_id == &rotor.rotor_id
                ) => {}
            Some(_) => errors.push(AcousticValidationError::ChannelKindMismatch(
                rotor.phase_channel_id.clone(),
            )),
            None => errors.push(AcousticValidationError::MissingRotorPhaseChannel(
                rotor.rotor_id.clone(),
            )),
        }

        match channels.get(rotor.speed_channel_id.as_str()) {
            Some(channel)
                if matches!(
                    &channel.kind,
                    ChannelKind::RotorSpeed { rotor_id } if rotor_id == &rotor.rotor_id
                ) => {}
            Some(_) => errors.push(AcousticValidationError::ChannelKindMismatch(
                rotor.speed_channel_id.clone(),
            )),
            None => errors.push(AcousticValidationError::MissingRotorSpeedChannel(
                rotor.rotor_id.clone(),
            )),
        }

        match operating_points.get(rotor.rotor_id.as_str()) {
            Some(point) => {
                if !valid_rotor_operating_point(point) {
                    errors.push(AcousticValidationError::InvalidRotorOperatingPoint(
                        rotor.rotor_id.clone(),
                    ));
                }
                if bundle.manifest.test_mode == TestMode::Transition
                    && point.tilt_deg.is_none()
                {
                    errors.push(AcousticValidationError::TransitionMissingTilt(
                        rotor.rotor_id.clone(),
                    ));
                }
            }
            None => errors.push(AcousticValidationError::MissingRotorOperatingPoint(
                rotor.rotor_id.clone(),
            )),
        }
    }

    for point in &bundle.rotor_operating_points {
        if !rotor_ids.contains(point.rotor_id.as_str()) {
            errors.push(AcousticValidationError::UnexpectedRotorOperatingPoint(
                point.rotor_id.clone(),
            ));
        }
    }
}

fn valid_rotor_operating_point(point: &RotorOperatingPoint) -> bool {
    let values = [
        point.rpm,
        point.thrust_n,
        point.shaft_torque_nm,
    ];
    if values.iter().any(|value| !value.is_finite())
        || point.rpm <= 0.0
        || point.thrust_n < 0.0
    {
        return false;
    }

    point
        .blade_pitch_deg
        .is_none_or(f64::is_finite)
        && point.tilt_deg.is_none_or(f64::is_finite)
}

fn validate_environment(
    environment: AcousticEnvironment,
    errors: &mut Vec<AcousticValidationError>,
) {
    if !environment.air_density_kg_m3.is_finite()
        || environment.air_density_kg_m3 <= 0.0
    {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "air density must be finite and positive",
        ));
    }
    if !environment.static_pressure_pa.is_finite()
        || environment.static_pressure_pa <= 0.0
    {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "static pressure must be finite and positive",
        ));
    }
    if !environment.temperature_k.is_finite() || environment.temperature_k <= 0.0 {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "temperature must be finite and positive in kelvin",
        ));
    }
    if !environment.relative_humidity_fraction.is_finite()
        || !(0.0..=1.0).contains(&environment.relative_humidity_fraction)
    {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "relative humidity must be a fraction in [0, 1]",
        ));
    }
    if !environment.free_stream_speed_m_s.is_finite()
        || environment.free_stream_speed_m_s < 0.0
    {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "free-stream speed must be finite and non-negative",
        ));
    }
    if !environment.turbulence_intensity_fraction.is_finite()
        || !(0.0..=1.0).contains(&environment.turbulence_intensity_fraction)
    {
        errors.push(AcousticValidationError::InvalidEnvironment(
            "turbulence intensity must be a fraction in [0, 1]",
        ));
    }
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn microphone_channel(channel_id: &str, microphone_id: &str) -> AcousticChannel {
        AcousticChannel {
            channel_id: channel_id.to_owned(),
            kind: ChannelKind::Microphone {
                microphone_id: microphone_id.to_owned(),
                array_id: "array-01".to_owned(),
            },
            unit: ChannelUnit::Pascal,
            calibration_id: format!("cal-{microphone_id}"),
            calibration_certificate_digest: digest('a'),
            raw_data_digest: digest('b'),
            clock_source_id: "clock-tai-01".to_owned(),
            clock_offset_to_reference_ns: 0,
            clock_uncertainty_ns: 50,
            start_time_ns: 0,
            end_time_ns: 1_000_000_000,
            sample_rate_hz: 48_000.0,
            sample_count: 48_001,
        }
    }

    fn bundle() -> AcousticRunBundle {
        AcousticRunBundle {
            manifest: AcousticRunManifest {
                run_id: "run-001".to_owned(),
                asset_id: "asset-rotorcraft-01".to_owned(),
                configuration_digest: digest('1'),
                geometry_digest: digest('2'),
                context_digest: digest('3'),
                procedure_revision: "acoustic-procedure-v1".to_owned(),
                evidence_manifest_digest: digest('4'),
                coordinate_frame_id: "aircraft-body-frame".to_owned(),
                reference_clock_id: "clock-tai-01".to_owned(),
                analysis_start_time_ns: 100,
                analysis_end_time_ns: 999_999_900,
                maximum_clock_uncertainty_ns: 100,
                maximum_channel_skew_ns: 200,
            },
            channels: vec![
                microphone_channel("mic-ch-01", "mic-01"),
                microphone_channel("mic-ch-02", "mic-02"),
                AcousticChannel {
                    channel_id: "rotor-phase-01".to_owned(),
                    kind: ChannelKind::RotorPhase {
                        rotor_id: "rotor-01".to_owned(),
                    },
                    unit: ChannelUnit::Radian,
                    calibration_id: "phase-cal-01".to_owned(),
                    calibration_certificate_digest: digest('5'),
                    raw_data_digest: digest('6'),
                    clock_source_id: "clock-tai-01".to_owned(),
                    clock_offset_to_reference_ns: 0,
                    clock_uncertainty_ns: 50,
                    start_time_ns: 0,
                    end_time_ns: 1_000_000_000,
                    sample_rate_hz: 10_000.0,
                    sample_count: 10_001,
                },
                AcousticChannel {
                    channel_id: "rotor-speed-01".to_owned(),
                    kind: ChannelKind::RotorSpeed {
                        rotor_id: "rotor-01".to_owned(),
                    },
                    unit: ChannelUnit::RevolutionsPerMinute,
                    calibration_id: "rpm-cal-01".to_owned(),
                    calibration_certificate_digest: digest('7'),
                    raw_data_digest: digest('8'),
                    clock_source_id: "clock-tai-01".to_owned(),
                    clock_offset_to_reference_ns: 0,
                    clock_uncertainty_ns: 50,
                    start_time_ns: 0,
                    end_time_ns: 1_000_000_000,
                    sample_rate_hz: 1_000.0,
                    sample_count: 1_001,
                },
            ],
            microphone_positions: vec![
                MicrophonePosition {
                    microphone_id: "mic-01".to_owned(),
                    channel_id: "mic-ch-01".to_owned(),
                    array_id: "array-01".to_owned(),
                    coordinate_frame_id: "aircraft-body-frame".to_owned(),
                    x_m: 2.0,
                    y_m: -1.0,
                    z_m: 1.5,
                    boresight_x: -1.0,
                    boresight_y: 0.0,
                    boresight_z: 0.0,
                    position_uncertainty_m: 0.001,
                },
                MicrophonePosition {
                    microphone_id: "mic-02".to_owned(),
                    channel_id: "mic-ch-02".to_owned(),
                    array_id: "array-01".to_owned(),
                    coordinate_frame_id: "aircraft-body-frame".to_owned(),
                    x_m: 2.0,
                    y_m: 1.0,
                    z_m: 1.5,
                    boresight_x: -1.0,
                    boresight_y: 0.0,
                    boresight_z: 0.0,
                    position_uncertainty_m: 0.001,
                },
            ],
            rotors: vec![RotorDeclaration {
                rotor_id: "rotor-01".to_owned(),
                geometry_digest: digest('9'),
                blade_count: 3,
                diameter_m: 1.2,
                rotation_direction: RotationDirection::Clockwise,
                phase_channel_id: "rotor-phase-01".to_owned(),
                speed_channel_id: "rotor-speed-01".to_owned(),
            }],
            rotor_operating_points: vec![RotorOperatingPoint {
                rotor_id: "rotor-01".to_owned(),
                rpm: 3_200.0,
                thrust_n: 400.0,
                shaft_torque_nm: 12.0,
                blade_pitch_deg: Some(12.0),
                tilt_deg: Some(0.0),
            }],
            environment: AcousticEnvironment {
                air_density_kg_m3: 1.2,
                static_pressure_pa: 101_325.0,
                temperature_k: 288.15,
                relative_humidity_fraction: 0.45,
                free_stream_speed_m_s: 0.0,
                turbulence_intensity_fraction: 0.01,
            },
        }
    }

    #[test]
    fn complete_synthetic_bundle_validates_and_reports_counts() {
        let summary = validate_bundle(&bundle()).unwrap();
        assert_eq!(summary.channel_count, 4);
        assert_eq!(summary.microphone_count, 2);
        assert_eq!(summary.microphone_array_count, 1);
        assert_eq!(summary.rotor_count, 1);
        assert_eq!(summary.analysis_duration_ns, 999_999_800);
        assert_eq!(summary.model_version, MODEL_VERSION);
    }

    #[test]
    fn invalid_digest_and_missing_procedure_identity_fail_closed() {
        let mut input = bundle();
        input.manifest.geometry_digest = "not-a-sha256".to_owned();
        input.manifest.procedure_revision.clear();

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(&AcousticValidationError::InvalidSha256Digest(
            "geometry_digest"
        )));
        assert!(errors.contains(&AcousticValidationError::EmptyIdentity(
            "procedure_revision"
        )));
    }

    #[test]
    fn microphone_geometry_must_match_frame_and_have_unit_boresight() {
        let mut input = bundle();
        input.microphone_positions[0].coordinate_frame_id = "wrong-frame".to_owned();
        input.microphone_positions[1].boresight_x = 0.0;
        input.microphone_positions[1].boresight_y = 0.0;
        input.microphone_positions[1].boresight_z = 0.0;

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(
            &AcousticValidationError::CoordinateFrameMismatch("mic-01".to_owned())
        ));
        assert!(errors.contains(
            &AcousticValidationError::InvalidMicrophoneGeometry("mic-02".to_owned())
        ));
    }

    #[test]
    fn acoustic_channels_must_cover_analysis_window_and_clock_budget() {
        let mut input = bundle();
        input.channels[0].clock_offset_to_reference_ns = 500;
        input.channels[1].end_time_ns = 900_000_000;

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(
            &AcousticValidationError::ChannelSkewExceeded("mic-ch-01".to_owned())
        ));
        assert!(errors.contains(
            &AcousticValidationError::ChannelDoesNotCoverAnalysisWindow(
                "mic-ch-02".to_owned()
            )
        ));
    }

    #[test]
    fn missing_microphone_position_and_too_small_array_are_not_silently_accepted() {
        let mut input = bundle();
        input.microphone_positions.pop();

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(
            &AcousticValidationError::UnexpectedMicrophonePosition(
                "mic-ch-02".to_owned()
            )
        ));
        assert!(errors.contains(
            &AcousticValidationError::MicrophoneArrayTooSmall("array-01".to_owned())
        ));
    }

    #[test]
    fn rotor_phase_and_speed_channels_must_match_the_declared_rotor() {
        let mut input = bundle();
        input.channels[2].kind = ChannelKind::RotorPhase {
            rotor_id: "other-rotor".to_owned(),
        };

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(
            &AcousticValidationError::ChannelKindMismatch(
                "rotor-phase-01".to_owned()
            )
        ));
    }

    #[test]
    fn transition_requires_a_declared_rotor_tilt_angle() {
        let mut input = bundle();
        input.manifest.test_mode = TestMode::Transition;
        input.rotor_operating_points[0].tilt_deg = None;

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(
            &AcousticValidationError::TransitionMissingTilt("rotor-01".to_owned())
        ));
    }

    #[test]
    fn duplicate_channels_rotors_and_operating_points_fail_closed() {
        let mut input = bundle();
        input.channels.push(input.channels[0].clone());
        input.rotors.push(input.rotors[0].clone());
        input.rotor_operating_points.push(input.rotor_operating_points[0].clone());

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(&AcousticValidationError::DuplicateChannelId));
        assert!(errors.contains(&AcousticValidationError::DuplicateRotorId));
        assert!(errors.contains(
            &AcousticValidationError::DuplicateRotorOperatingPoint
        ));
    }

    #[test]
    fn invalid_environment_and_operating_point_are_rejected() {
        let mut input = bundle();
        input.environment.air_density_kg_m3 = f64::NAN;
        input.environment.relative_humidity_fraction = 1.2;
        input.rotor_operating_points[0].rpm = -1.0;

        let errors = validate_bundle(&input).unwrap_err();
        assert!(errors.contains(&AcousticValidationError::InvalidEnvironment(
            "air density must be finite and positive"
        )));
        assert!(errors.contains(&AcousticValidationError::InvalidEnvironment(
            "relative humidity must be a fraction in [0, 1]"
        )));
        assert!(errors.contains(
            &AcousticValidationError::InvalidRotorOperatingPoint("rotor-01".to_owned())
        ));
    }
}
