// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent reference for calibrated propulsion-test channel provenance
//! and bounded timestamp synchronization.
//!
//! This crate selects actual, valid samples near an anchor timestamp. It does
//! not interpolate, extrapolate, impute, calibrate sensors or convert clocks.

use std::collections::BTreeSet;
use std::f64::consts::TAU;
use std::fmt;

pub const MODEL_VERSION: &str = "electric-propulsion-observation-oracle-v0.2.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeDomain {
    Monotonic,
    Tai,
    Utc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementQuantity {
    Torque,
    RotationalSpeed,
    Thrust,
    Voltage,
    Current,
    ElectricalPower,
    Temperature,
    Pressure,
    CoolantMassFlow,
    SoundPressure,
    VibrationAcceleration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementUnit {
    NewtonMetre,
    RevolutionsPerMinute,
    Newton,
    Volt,
    Ampere,
    Watt,
    Kelvin,
    Pascal,
    KilogramPerSecond,
    PascalRms,
    MetrePerSecondSquared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignConvention {
    PositiveMotoring,
    PositiveCharge,
    PositiveDischarge,
    PositiveThrust,
    SignedEngineering,
    NonNegative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStatus {
    Current,
    Expired,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleQuality {
    Valid,
    Suspect,
    Invalid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunManifest {
    pub run_id: String,
    pub asset_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub procedure_revision: String,
    pub evidence_manifest_digest: String,
    pub time_domain: TimeDomain,
    pub maximum_clock_uncertainty_ns: u64,
    pub maximum_sample_time_error_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementSample {
    pub source_sequence: u64,
    pub timestamp_ns: i64,
    pub value: f64,
    pub absolute_uncertainty: f64,
    pub quality: SampleQuality,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementChannel {
    pub channel_id: String,
    pub physical_boundary_id: String,
    pub run_id: String,
    pub asset_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub quantity: MeasurementQuantity,
    pub unit: MeasurementUnit,
    pub sign_convention: SignConvention,
    pub calibration_id: String,
    pub calibration_certificate_digest: String,
    pub calibration_status: CalibrationStatus,
    pub raw_data_digest: String,
    pub clock_source_id: String,
    pub clock_uncertainty_ns: u64,
    pub samples: Vec<MeasurementSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SynchronizedObservation {
    pub channel_id: String,
    pub physical_boundary_id: String,
    pub quantity: MeasurementQuantity,
    pub unit: MeasurementUnit,
    pub sign_convention: SignConvention,
    pub calibration_id: String,
    pub calibration_certificate_digest: String,
    pub raw_data_digest: String,
    pub clock_source_id: String,
    pub clock_uncertainty_ns: u64,
    pub source_sequence: u64,
    pub source_timestamp_ns: i64,
    pub timestamp_offset_ns: i128,
    pub sample_skew_ns: u64,
    pub combined_time_error_ns: u64,
    pub value: f64,
    pub absolute_uncertainty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SynchronizedOperatingPoint {
    pub model_version: &'static str,
    pub run_id: String,
    pub asset_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub evidence_manifest_digest: String,
    pub time_domain: TimeDomain,
    pub anchor_timestamp_ns: i64,
    pub maximum_observed_sample_skew_ns: u64,
    pub maximum_clock_uncertainty_ns: u64,
    pub maximum_combined_time_error_ns: u64,
    pub observations: Vec<SynchronizedObservation>,
}

/// A motor-shaft operating point projected from two synchronized, calibrated
/// raw observations. The power interval is a rectangular absolute-bound
/// propagation; it is not a probabilistic confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct MotorMapOperatingPoint {
    pub observation_model_version: &'static str,
    pub run_id: String,
    pub asset_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub evidence_manifest_digest: String,
    pub time_domain: TimeDomain,
    pub anchor_timestamp_ns: i64,
    pub motor_shaft_boundary_id: String,
    pub torque_channel_id: String,
    pub torque_calibration_id: String,
    pub torque_calibration_certificate_digest: String,
    pub torque_raw_data_digest: String,
    pub torque_clock_source_id: String,
    pub torque_clock_uncertainty_ns: u64,
    pub torque_timestamp_offset_ns: i128,
    pub torque_sample_skew_ns: u64,
    pub torque_combined_time_error_ns: u64,
    pub torque_source_sequence: u64,
    pub torque_timestamp_ns: i64,
    pub torque_value_nm: f64,
    pub torque_absolute_uncertainty_nm: f64,
    pub speed_channel_id: String,
    pub speed_calibration_id: String,
    pub speed_calibration_certificate_digest: String,
    pub speed_raw_data_digest: String,
    pub speed_clock_source_id: String,
    pub speed_clock_uncertainty_ns: u64,
    pub speed_timestamp_offset_ns: i128,
    pub speed_sample_skew_ns: u64,
    pub speed_combined_time_error_ns: u64,
    pub speed_source_sequence: u64,
    pub speed_timestamp_ns: i64,
    pub speed_value_rpm: f64,
    pub speed_absolute_uncertainty_rpm: f64,
    pub motor_angular_velocity_rad_s: f64,
    pub shaft_power_w: f64,
    pub shaft_power_lower_bound_w: f64,
    pub shaft_power_upper_bound_w: f64,
    pub shaft_power_absolute_uncertainty_bound_w: f64,
    pub maximum_combined_time_error_ns: u64,
    pub torque_speed_sample_time_delta_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationError {
    InvalidManifest(&'static str),
    InvalidChannel(&'static str),
    InvalidUnit {
        quantity: MeasurementQuantity,
        unit: MeasurementUnit,
    },
    CalibrationNotCurrent {
        channel_id: String,
        status: CalibrationStatus,
    },
    DuplicateIdentity(&'static str),
    MissingRequiredChannel(String),
    OutOfOrderSamples {
        channel_id: String,
    },
    DuplicateSampleTimestamp {
        channel_id: String,
        timestamp_ns: i64,
    },
    InvalidSample {
        channel_id: String,
        sequence: u64,
    },
    NoValidSample {
        channel_id: String,
    },
    Unsynchronized {
        channel_id: String,
        nearest_sample_skew_ns: u64,
        allowed_time_error_ns: u64,
    },
    ClockUncertaintyExceeded {
        channel_id: String,
        clock_uncertainty_ns: u64,
        allowed_clock_uncertainty_ns: u64,
    },
    InvalidMotorMapOperatingPoint(&'static str),
    NumericalOverflow,
}

impl fmt::Display for ObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ObservationError {}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn unit_matches(quantity: MeasurementQuantity, unit: MeasurementUnit) -> bool {
    matches!(
        (quantity, unit),
        (MeasurementQuantity::Torque, MeasurementUnit::NewtonMetre)
            | (
                MeasurementQuantity::RotationalSpeed,
                MeasurementUnit::RevolutionsPerMinute
            )
            | (MeasurementQuantity::Thrust, MeasurementUnit::Newton)
            | (MeasurementQuantity::Voltage, MeasurementUnit::Volt)
            | (MeasurementQuantity::Current, MeasurementUnit::Ampere)
            | (MeasurementQuantity::ElectricalPower, MeasurementUnit::Watt)
            | (MeasurementQuantity::Temperature, MeasurementUnit::Kelvin)
            | (MeasurementQuantity::Pressure, MeasurementUnit::Pascal)
            | (
                MeasurementQuantity::CoolantMassFlow,
                MeasurementUnit::KilogramPerSecond
            )
            | (
                MeasurementQuantity::SoundPressure,
                MeasurementUnit::PascalRms
            )
            | (
                MeasurementQuantity::VibrationAcceleration,
                MeasurementUnit::MetrePerSecondSquared
            )
    )
}

fn validate_manifest(manifest: &RunManifest) -> Result<(), ObservationError> {
    if !non_empty(&manifest.run_id)
        || !non_empty(&manifest.asset_id)
        || !non_empty(&manifest.configuration_digest)
        || !non_empty(&manifest.context_digest)
        || !non_empty(&manifest.procedure_revision)
        || !non_empty(&manifest.evidence_manifest_digest)
    {
        return Err(ObservationError::InvalidManifest(
            "run, asset, configuration, context, procedure and evidence identities must be present",
        ));
    }
    Ok(())
}

fn validate_channel_identity(
    manifest: &RunManifest,
    channel: &MeasurementChannel,
) -> Result<(), ObservationError> {
    if !non_empty(&channel.channel_id)
        || !non_empty(&channel.physical_boundary_id)
        || !non_empty(&channel.calibration_id)
        || !non_empty(&channel.calibration_certificate_digest)
        || !non_empty(&channel.raw_data_digest)
        || !non_empty(&channel.clock_source_id)
    {
        return Err(ObservationError::InvalidChannel(
            "channel, boundary, calibration, raw-data and clock identities must be present",
        ));
    }
    if channel.run_id != manifest.run_id
        || channel.asset_id != manifest.asset_id
        || channel.configuration_digest != manifest.configuration_digest
        || channel.context_digest != manifest.context_digest
    {
        return Err(ObservationError::InvalidChannel(
            "channel run, asset, configuration and context must match the run manifest",
        ));
    }
    if !unit_matches(channel.quantity, channel.unit) {
        return Err(ObservationError::InvalidUnit {
            quantity: channel.quantity,
            unit: channel.unit,
        });
    }
    if channel.clock_uncertainty_ns > manifest.maximum_clock_uncertainty_ns {
        return Err(ObservationError::ClockUncertaintyExceeded {
            channel_id: channel.channel_id.clone(),
            clock_uncertainty_ns: channel.clock_uncertainty_ns,
            allowed_clock_uncertainty_ns: manifest.maximum_clock_uncertainty_ns,
        });
    }
    if channel.calibration_status != CalibrationStatus::Current {
        return Err(ObservationError::CalibrationNotCurrent {
            channel_id: channel.channel_id.clone(),
            status: channel.calibration_status,
        });
    }
    if channel.samples.is_empty() {
        return Err(ObservationError::InvalidChannel(
            "measurement channel must contain source samples",
        ));
    }

    let mut previous_sequence = None;
    let mut previous_timestamp = None;
    for sample in &channel.samples {
        if previous_sequence.is_some_and(|previous| sample.source_sequence <= previous)
            || previous_timestamp.is_some_and(|previous| sample.timestamp_ns < previous)
        {
            return Err(ObservationError::OutOfOrderSamples {
                channel_id: channel.channel_id.clone(),
            });
        }
        if previous_timestamp == Some(sample.timestamp_ns) {
            return Err(ObservationError::DuplicateSampleTimestamp {
                channel_id: channel.channel_id.clone(),
                timestamp_ns: sample.timestamp_ns,
            });
        }
        if !sample.value.is_finite()
            || !sample.absolute_uncertainty.is_finite()
            || sample.absolute_uncertainty < 0.0
        {
            return Err(ObservationError::InvalidSample {
                channel_id: channel.channel_id.clone(),
                sequence: sample.source_sequence,
            });
        }
        previous_sequence = Some(sample.source_sequence);
        previous_timestamp = Some(sample.timestamp_ns);
    }
    Ok(())
}

/// Select nearest actual valid samples for the required channels at one anchor.
///
/// The channel list may contain several channels on the same physical boundary
/// (for example voltage/current at motor terminals). Channel IDs must be unique.
/// Equal-distance sample ties select the earlier timestamp then lower sequence.
/// No interpolation, extrapolation, sample hold or missing-value repair occurs.
pub fn synchronize_at(
    manifest: &RunManifest,
    channels: &[MeasurementChannel],
    required_channel_ids: &[&str],
    anchor_timestamp_ns: i64,
) -> Result<SynchronizedOperatingPoint, ObservationError> {
    validate_manifest(manifest)?;
    if channels.is_empty() || required_channel_ids.is_empty() {
        return Err(ObservationError::InvalidManifest(
            "channels and required channel IDs must not be empty",
        ));
    }

    let mut seen_channel_ids = BTreeSet::new();
    for channel in channels {
        if !seen_channel_ids.insert(channel.channel_id.as_str()) {
            return Err(ObservationError::DuplicateIdentity("channel ID"));
        }
        if channel.run_id != manifest.run_id
            || channel.asset_id != manifest.asset_id
            || channel.configuration_digest != manifest.configuration_digest
            || channel.context_digest != manifest.context_digest
        {
            return Err(ObservationError::InvalidChannel(
                "channel ownership does not match run manifest",
            ));
        }
    }

    let mut seen_required = BTreeSet::new();
    for channel_id in required_channel_ids {
        if !seen_required.insert(*channel_id) {
            return Err(ObservationError::DuplicateIdentity(
                "required channel ID",
            ));
        }
    }

    let mut ordered_channels = channels.to_vec();
    ordered_channels.sort_by(|a, b| a.channel_id.cmp(&b.channel_id));

    let mut observations = Vec::with_capacity(required_channel_ids.len());
    let mut maximum_sample_skew_ns = 0_u64;
    let mut maximum_clock_uncertainty_ns = 0_u64;
    let mut maximum_combined_time_error_ns = 0_u64;

    for required_id in seen_required {
        let channel = ordered_channels
            .iter()
            .find(|channel| channel.channel_id == required_id)
            .ok_or_else(|| ObservationError::MissingRequiredChannel(required_id.to_owned()))?;

        validate_channel_identity(manifest, channel)?;

        let sample = channel
            .samples
            .iter()
            .filter(|sample| sample.quality == SampleQuality::Valid)
            .min_by_key(|sample| {
                (
                    sample.timestamp_ns.abs_diff(anchor_timestamp_ns),
                    sample.timestamp_ns,
                    sample.source_sequence,
                )
            })
            .ok_or_else(|| ObservationError::NoValidSample {
                channel_id: channel.channel_id.clone(),
            })?;

        let sample_skew_ns = sample.timestamp_ns.abs_diff(anchor_timestamp_ns);
        let combined_time_error_ns = sample_skew_ns
            .checked_add(channel.clock_uncertainty_ns)
            .ok_or(ObservationError::NumericalOverflow)?;
        if combined_time_error_ns > manifest.maximum_sample_time_error_ns {
            return Err(ObservationError::Unsynchronized {
                channel_id: channel.channel_id.clone(),
                nearest_sample_skew_ns: sample_skew_ns,
                allowed_time_error_ns: manifest.maximum_sample_time_error_ns,
            });
        }
        let timestamp_offset_ns =
            i128::from(sample.timestamp_ns) - i128::from(anchor_timestamp_ns);

        maximum_sample_skew_ns = maximum_sample_skew_ns.max(sample_skew_ns);
        maximum_clock_uncertainty_ns =
            maximum_clock_uncertainty_ns.max(channel.clock_uncertainty_ns);
        maximum_combined_time_error_ns =
            maximum_combined_time_error_ns.max(combined_time_error_ns);

        observations.push(SynchronizedObservation {
            channel_id: channel.channel_id.clone(),
            physical_boundary_id: channel.physical_boundary_id.clone(),
            quantity: channel.quantity,
            unit: channel.unit,
            sign_convention: channel.sign_convention,
            calibration_id: channel.calibration_id.clone(),
            calibration_certificate_digest: channel.calibration_certificate_digest.clone(),
            raw_data_digest: channel.raw_data_digest.clone(),
            clock_source_id: channel.clock_source_id.clone(),
            clock_uncertainty_ns: channel.clock_uncertainty_ns,
            source_sequence: sample.source_sequence,
            source_timestamp_ns: sample.timestamp_ns,
            timestamp_offset_ns,
            sample_skew_ns,
            combined_time_error_ns,
            value: sample.value,
            absolute_uncertainty: sample.absolute_uncertainty,
        });
    }

    observations.sort_by(|a, b| a.channel_id.cmp(&b.channel_id));

    Ok(SynchronizedOperatingPoint {
        model_version: MODEL_VERSION,
        run_id: manifest.run_id.clone(),
        asset_id: manifest.asset_id.clone(),
        configuration_digest: manifest.configuration_digest.clone(),
        context_digest: manifest.context_digest.clone(),
        evidence_manifest_digest: manifest.evidence_manifest_digest.clone(),
        time_domain: manifest.time_domain,
        anchor_timestamp_ns,
        maximum_observed_sample_skew_ns: maximum_sample_skew_ns,
        maximum_clock_uncertainty_ns,
        maximum_combined_time_error_ns,
        observations,
    })
}


/// Derive a motor-shaft torque/RPM/power operating point from a synchronized
/// observation bundle.
///
/// Both channels must use the requested same motor-shaft physical boundary,
/// carry the explicit positive-motoring sign convention and contain positive
/// values. The power interval is computed from the rectangular absolute
/// torque and RPM uncertainty bounds. No interpolation or unit conversion is
/// performed except RPM to rad/s.
pub fn derive_motor_map_operating_point(
    point: &SynchronizedOperatingPoint,
    torque_channel_id: &str,
    speed_channel_id: &str,
    expected_motor_shaft_boundary_id: &str,
) -> Result<MotorMapOperatingPoint, ObservationError> {
    if !non_empty(torque_channel_id)
        || !non_empty(speed_channel_id)
        || !non_empty(expected_motor_shaft_boundary_id)
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "torque channel, speed channel and motor-shaft boundary IDs must be present",
        ));
    }
    if torque_channel_id == speed_channel_id {
        return Err(ObservationError::DuplicateIdentity(
            "motor-map torque and speed channel IDs",
        ));
    }
    if !non_empty(&point.run_id)
        || !non_empty(&point.asset_id)
        || !non_empty(&point.configuration_digest)
        || !non_empty(&point.context_digest)
        || !non_empty(&point.evidence_manifest_digest)
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "synchronized provenance identities are required",
        ));
    }

    let torque_matches = point
        .observations
        .iter()
        .filter(|observation| observation.channel_id == torque_channel_id)
        .count();
    let speed_matches = point
        .observations
        .iter()
        .filter(|observation| observation.channel_id == speed_channel_id)
        .count();
    if torque_matches > 1 || speed_matches > 1 {
        return Err(ObservationError::DuplicateIdentity(
            "motor-map torque or speed observation",
        ));
    }
    let torque = point
        .observations
        .iter()
        .find(|observation| observation.channel_id == torque_channel_id)
        .ok_or_else(|| ObservationError::MissingRequiredChannel(torque_channel_id.to_owned()))?;
    let speed = point
        .observations
        .iter()
        .find(|observation| observation.channel_id == speed_channel_id)
        .ok_or_else(|| ObservationError::MissingRequiredChannel(speed_channel_id.to_owned()))?;

    for observation in [torque, speed] {
        if !non_empty(&observation.channel_id)
            || !non_empty(&observation.physical_boundary_id)
            || !non_empty(&observation.calibration_id)
            || !non_empty(&observation.calibration_certificate_digest)
            || !non_empty(&observation.raw_data_digest)
            || !non_empty(&observation.clock_source_id)
        {
            return Err(ObservationError::InvalidMotorMapOperatingPoint(
                "selected channel provenance is incomplete",
            ));
        }
        let expected_offset =
            i128::from(observation.source_timestamp_ns) - i128::from(point.anchor_timestamp_ns);
        let expected_skew = observation
            .source_timestamp_ns
            .abs_diff(point.anchor_timestamp_ns);
        let expected_combined = expected_skew
            .checked_add(observation.clock_uncertainty_ns)
            .ok_or(ObservationError::NumericalOverflow)?;
        if observation.timestamp_offset_ns != expected_offset
            || observation.sample_skew_ns != expected_skew
            || observation.combined_time_error_ns != expected_combined
            || observation.clock_uncertainty_ns > point.maximum_clock_uncertainty_ns
            || observation.sample_skew_ns > point.maximum_observed_sample_skew_ns
            || observation.combined_time_error_ns > point.maximum_combined_time_error_ns
        {
            return Err(ObservationError::InvalidMotorMapOperatingPoint(
                "selected channel timing provenance is inconsistent",
            ));
        }
    }

    if torque.quantity != MeasurementQuantity::Torque
        || torque.unit != MeasurementUnit::NewtonMetre
        || speed.quantity != MeasurementQuantity::RotationalSpeed
        || speed.unit != MeasurementUnit::RevolutionsPerMinute
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "motor map requires torque in N m and rotational speed in RPM",
        ));
    }
    if torque.physical_boundary_id != expected_motor_shaft_boundary_id
        || speed.physical_boundary_id != expected_motor_shaft_boundary_id
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "torque and speed must refer to the declared same motor-shaft boundary",
        ));
    }
    if torque.sign_convention != SignConvention::PositiveMotoring
        || speed.sign_convention != SignConvention::PositiveMotoring
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "torque and speed must use the explicit positive-motoring sign convention",
        ));
    }
    if !torque.value.is_finite()
        || !speed.value.is_finite()
        || !torque.absolute_uncertainty.is_finite()
        || !speed.absolute_uncertainty.is_finite()
        || torque.absolute_uncertainty < 0.0
        || speed.absolute_uncertainty < 0.0
        || torque.value <= 0.0
        || speed.value <= 0.0
    {
        return Err(ObservationError::InvalidMotorMapOperatingPoint(
            "motor torque and RPM must be finite and positive with finite non-negative uncertainty",
        ));
    }

    let motor_angular_velocity_rad_s = speed.value * TAU / 60.0;
    let speed_absolute_uncertainty_rad_s = speed.absolute_uncertainty * TAU / 60.0;
    let shaft_power_w = torque.value * motor_angular_velocity_rad_s;

    let torque_lower_nm = (torque.value - torque.absolute_uncertainty).max(0.0);
    let torque_upper_nm = torque.value + torque.absolute_uncertainty;
    let angular_velocity_lower_rad_s =
        (motor_angular_velocity_rad_s - speed_absolute_uncertainty_rad_s).max(0.0);
    let angular_velocity_upper_rad_s =
        motor_angular_velocity_rad_s + speed_absolute_uncertainty_rad_s;
    let shaft_power_lower_bound_w =
        torque_lower_nm * angular_velocity_lower_rad_s;
    let shaft_power_upper_bound_w =
        torque_upper_nm * angular_velocity_upper_rad_s;
    let shaft_power_absolute_uncertainty_bound_w = (shaft_power_w
        - shaft_power_lower_bound_w)
        .max(shaft_power_upper_bound_w - shaft_power_w);

    let derived = [
        motor_angular_velocity_rad_s,
        speed_absolute_uncertainty_rad_s,
        shaft_power_w,
        torque_upper_nm,
        angular_velocity_upper_rad_s,
        shaft_power_lower_bound_w,
        shaft_power_upper_bound_w,
        shaft_power_absolute_uncertainty_bound_w,
    ];
    if derived.iter().any(|value| !value.is_finite())
        || shaft_power_lower_bound_w > shaft_power_w
        || shaft_power_w > shaft_power_upper_bound_w
    {
        return Err(ObservationError::NumericalOverflow);
    }

    Ok(MotorMapOperatingPoint {
        observation_model_version: point.model_version,
        run_id: point.run_id.clone(),
        asset_id: point.asset_id.clone(),
        configuration_digest: point.configuration_digest.clone(),
        context_digest: point.context_digest.clone(),
        evidence_manifest_digest: point.evidence_manifest_digest.clone(),
        time_domain: point.time_domain,
        anchor_timestamp_ns: point.anchor_timestamp_ns,
        motor_shaft_boundary_id: expected_motor_shaft_boundary_id.to_owned(),
        torque_channel_id: torque.channel_id.clone(),
        torque_calibration_id: torque.calibration_id.clone(),
        torque_calibration_certificate_digest: torque.calibration_certificate_digest.clone(),
        torque_raw_data_digest: torque.raw_data_digest.clone(),
        torque_clock_source_id: torque.clock_source_id.clone(),
        torque_clock_uncertainty_ns: torque.clock_uncertainty_ns,
        torque_timestamp_offset_ns: torque.timestamp_offset_ns,
        torque_sample_skew_ns: torque.sample_skew_ns,
        torque_combined_time_error_ns: torque.combined_time_error_ns,
        torque_source_sequence: torque.source_sequence,
        torque_timestamp_ns: torque.source_timestamp_ns,
        torque_value_nm: torque.value,
        torque_absolute_uncertainty_nm: torque.absolute_uncertainty,
        speed_channel_id: speed.channel_id.clone(),
        speed_calibration_id: speed.calibration_id.clone(),
        speed_calibration_certificate_digest: speed.calibration_certificate_digest.clone(),
        speed_raw_data_digest: speed.raw_data_digest.clone(),
        speed_clock_source_id: speed.clock_source_id.clone(),
        speed_clock_uncertainty_ns: speed.clock_uncertainty_ns,
        speed_timestamp_offset_ns: speed.timestamp_offset_ns,
        speed_sample_skew_ns: speed.sample_skew_ns,
        speed_combined_time_error_ns: speed.combined_time_error_ns,
        speed_source_sequence: speed.source_sequence,
        speed_timestamp_ns: speed.source_timestamp_ns,
        speed_value_rpm: speed.value,
        speed_absolute_uncertainty_rpm: speed.absolute_uncertainty,
        motor_angular_velocity_rad_s,
        shaft_power_w,
        shaft_power_lower_bound_w,
        shaft_power_upper_bound_w,
        shaft_power_absolute_uncertainty_bound_w,
        maximum_combined_time_error_ns: torque
            .combined_time_error_ns
            .max(speed.combined_time_error_ns),
        torque_speed_sample_time_delta_ns: torque
            .source_timestamp_ns
            .abs_diff(speed.source_timestamp_ns),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> RunManifest {
        RunManifest {
            run_id: "run-001".to_owned(),
            asset_id: "motor-serial-001".to_owned(),
            configuration_digest: "config-sha256".to_owned(),
            context_digest: "context-sha256".to_owned(),
            procedure_revision: "procedure-v1".to_owned(),
            evidence_manifest_digest: "manifest-sha256".to_owned(),
            time_domain: TimeDomain::Tai,
            maximum_clock_uncertainty_ns: 100,
            maximum_sample_time_error_ns: 500,
        }
    }

    fn sample(sequence: u64, timestamp_ns: i64, value: f64) -> MeasurementSample {
        MeasurementSample {
            source_sequence: sequence,
            timestamp_ns,
            value,
            absolute_uncertainty: 0.01,
            quality: SampleQuality::Valid,
        }
    }

    fn channel(
        channel_id: &str,
        quantity: MeasurementQuantity,
        unit: MeasurementUnit,
        timestamp_ns: i64,
        value: f64,
    ) -> MeasurementChannel {
        MeasurementChannel {
            channel_id: channel_id.to_owned(),
            physical_boundary_id: "motor-dc-terminals".to_owned(),
            run_id: "run-001".to_owned(),
            asset_id: "motor-serial-001".to_owned(),
            configuration_digest: "config-sha256".to_owned(),
            context_digest: "context-sha256".to_owned(),
            quantity,
            unit,
            sign_convention: SignConvention::SignedEngineering,
            calibration_id: format!("cal-{channel_id}"),
            calibration_certificate_digest: format!("cal-sha-{channel_id}"),
            calibration_status: CalibrationStatus::Current,
            raw_data_digest: format!("raw-sha-{channel_id}"),
            clock_source_id: "ptp-grandmaster-01".to_owned(),
            clock_uncertainty_ns: 50,
            samples: vec![
                sample(1, timestamp_ns - 100, value - 0.1),
                sample(2, timestamp_ns, value),
                sample(3, timestamp_ns + 100, value + 0.1),
            ],
        }
    }

    fn required() -> [&'static str; 5] {
        ["torque", "rpm", "voltage", "current", "temperature"]
    }

    fn channels() -> Vec<MeasurementChannel> {
        let mut values = vec![
            channel(
                "torque",
                MeasurementQuantity::Torque,
                MeasurementUnit::NewtonMetre,
                1_000,
                7.5,
            ),
            channel(
                "rpm",
                MeasurementQuantity::RotationalSpeed,
                MeasurementUnit::RevolutionsPerMinute,
                1_000,
                15_000.0,
            ),
            channel(
                "voltage",
                MeasurementQuantity::Voltage,
                MeasurementUnit::Volt,
                1_000,
                400.0,
            ),
            channel(
                "current",
                MeasurementQuantity::Current,
                MeasurementUnit::Ampere,
                1_000,
                30.0,
            ),
            channel(
                "temperature",
                MeasurementQuantity::Temperature,
                MeasurementUnit::Kelvin,
                1_000,
                323.15,
            ),
        ];
        values[0].physical_boundary_id = "motor-shaft".to_owned();
        values[1].physical_boundary_id = "motor-shaft".to_owned();
        values[0].sign_convention = SignConvention::PositiveMotoring;
        values[1].sign_convention = SignConvention::PositiveMotoring;
        values
    }

    #[test]
    fn motor_map_projection_derives_shaft_power_and_preserves_provenance() {
        let run = manifest();
        let synced = synchronize_at(&run, &channels(), &required(), 1_000).unwrap();
        let result = derive_motor_map_operating_point(
            &synced,
            "torque",
            "rpm",
            "motor-shaft",
        )
        .unwrap();

        let expected_omega = 15_000.0 * TAU / 60.0;
        let expected_power = 7.5 * expected_omega;
        assert!((result.motor_angular_velocity_rad_s - expected_omega).abs() < 1.0e-12);
        assert!((result.shaft_power_w - expected_power).abs() < 1.0e-9);
        assert!(result.shaft_power_lower_bound_w < result.shaft_power_w);
        assert!(result.shaft_power_upper_bound_w > result.shaft_power_w);
        assert!(
            (result.shaft_power_absolute_uncertainty_bound_w
                - (result.shaft_power_w - result.shaft_power_lower_bound_w)
                    .max(result.shaft_power_upper_bound_w - result.shaft_power_w))
                .abs()
                < 1.0e-12
        );
        assert_eq!(result.run_id, "run-001");
        assert_eq!(result.configuration_digest, "config-sha256");
        assert_eq!(result.evidence_manifest_digest, "manifest-sha256");
        assert_eq!(result.motor_shaft_boundary_id, "motor-shaft");
        assert_eq!(result.torque_calibration_id, "cal-torque");
        assert_eq!(result.speed_calibration_id, "cal-rpm");
        assert_eq!(result.torque_calibration_certificate_digest, "cal-sha-torque");
        assert_eq!(result.torque_clock_source_id, "ptp-grandmaster-01");
        assert_eq!(result.speed_clock_source_id, "ptp-grandmaster-01");
        assert_eq!(result.torque_clock_uncertainty_ns, 50);
        assert_eq!(result.speed_clock_uncertainty_ns, 50);
        assert_eq!(result.torque_timestamp_offset_ns, 0);
        assert_eq!(result.speed_timestamp_offset_ns, 0);
        assert_eq!(result.torque_sample_skew_ns, 0);
        assert_eq!(result.speed_sample_skew_ns, 0);
        assert_eq!(result.torque_combined_time_error_ns, 50);
        assert_eq!(result.speed_combined_time_error_ns, 50);
        assert_eq!(result.torque_speed_sample_time_delta_ns, 0);
        assert_eq!(result.speed_calibration_certificate_digest, "cal-sha-rpm");
        assert_eq!(result.torque_raw_data_digest, "raw-sha-torque");
        assert_eq!(result.speed_raw_data_digest, "raw-sha-rpm");
        assert_eq!(result.torque_timestamp_ns, 1_000);
        assert_eq!(result.speed_timestamp_ns, 1_000);
    }

    #[test]
    fn motor_map_projection_rejects_tampered_timing_provenance() {
        let run = manifest();
        let mut synced = synchronize_at(&run, &channels(), &required(), 1_000).unwrap();
        let torque = synced
            .observations
            .iter_mut()
            .find(|observation| observation.channel_id == "torque")
            .unwrap();
        torque.timestamp_offset_ns = 1;
        assert!(matches!(
            derive_motor_map_operating_point(
                &synced,
                "torque",
                "rpm",
                "motor-shaft",
            ),
            Err(ObservationError::InvalidMotorMapOperatingPoint(_))
        ));
    }

    #[test]
    fn motor_map_projection_rejects_different_shaft_boundaries() {
        let run = manifest();
        let mut values = channels();
        values[1].physical_boundary_id = "rotor-shaft".to_owned();
        let synced = synchronize_at(&run, &values, &required(), 1_000).unwrap();
        assert!(matches!(
            derive_motor_map_operating_point(
                &synced,
                "torque",
                "rpm",
                "motor-shaft"
            ),
            Err(ObservationError::InvalidMotorMapOperatingPoint(_))
        ));
    }

    #[test]
    fn motor_map_projection_requires_positive_motoring_convention() {
        let run = manifest();
        let mut values = channels();
        values[0].sign_convention = SignConvention::SignedEngineering;
        let synced = synchronize_at(&run, &values, &required(), 1_000).unwrap();
        assert!(matches!(
            derive_motor_map_operating_point(
                &synced,
                "torque",
                "rpm",
                "motor-shaft"
            ),
            Err(ObservationError::InvalidMotorMapOperatingPoint(_))
        ));
    }

    #[test]
    fn motor_map_projection_rejects_zero_or_negative_operating_points() {
        let run = manifest();
        let mut values = channels();
        values[0].samples[1].value = 0.0;
        let synced = synchronize_at(&run, &values, &required(), 1_000).unwrap();
        assert!(matches!(
            derive_motor_map_operating_point(
                &synced,
                "torque",
                "rpm",
                "motor-shaft"
            ),
            Err(ObservationError::InvalidMotorMapOperatingPoint(_))
        ));

        let mut values = channels();
        values[1].samples[1].value = -1.0;
        let synced = synchronize_at(&run, &values, &required(), 1_000).unwrap();
        assert!(matches!(
            derive_motor_map_operating_point(
                &synced,
                "torque",
                "rpm",
                "motor-shaft"
            ),
            Err(ObservationError::InvalidMotorMapOperatingPoint(_))
        ));
    }

    #[test]
    fn valid_five_channel_operating_point_is_synchronized_and_traceable() {
        let run = manifest();
        let result = synchronize_at(&run, &channels(), &required(), 1_000).unwrap();
        assert_eq!(result.observations.len(), 5);
        assert_eq!(result.maximum_observed_sample_skew_ns, 0);
        assert_eq!(result.maximum_clock_uncertainty_ns, 50);
        assert_eq!(result.maximum_combined_time_error_ns, 50);
        assert_eq!(result.configuration_digest, "config-sha256");
        assert_eq!(result.evidence_manifest_digest, "manifest-sha256");
        assert_eq!(result.observations[0].channel_id, "current");
        assert_eq!(result.observations[0].calibration_certificate_digest, "cal-sha-current");
        assert_eq!(result.observations[0].raw_data_digest, "raw-sha-current");
    }

    #[test]
    fn nearest_actual_sample_is_selected_within_declared_skew() {
        let run = manifest();
        let mut values = channels();
        values[0].samples = vec![
            sample(1, 900, 7.0),
            sample(2, 1_080, 7.8),
            sample(3, 1_300, 8.0),
        ];
        let result = synchronize_at(&run, &values, &required(), 1_000).unwrap();
        let torque = result
            .observations
            .iter()
            .find(|observation| observation.channel_id == "torque")
            .unwrap();
        assert_eq!(torque.source_sequence, 2);
        assert_eq!(torque.source_timestamp_ns, 1_080);
        assert_eq!(torque.timestamp_offset_ns, 80);
        assert_eq!(torque.value, 7.8);
        assert_eq!(torque.sample_skew_ns, 80);
        assert_eq!(torque.combined_time_error_ns, 130);
    }

    #[test]
    fn equal_distance_tie_selects_earlier_timestamp() {
        let run = manifest();
        let mut values = channels();
        values[0].samples = vec![sample(1, 900, 7.0), sample(2, 1_100, 8.0)];
        let result = synchronize_at(&run, &values, &["torque"], 1_000).unwrap();
        assert_eq!(result.observations[0].source_timestamp_ns, 900);
        assert_eq!(result.observations[0].source_sequence, 1);
    }

    #[test]
    fn sample_skew_beyond_budget_is_rejected() {
        let run = manifest();
        let mut values = channels();
        values[0].samples = vec![sample(1, 1_800, 7.5)];
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::Unsynchronized { .. })
        ));
    }

    #[test]
    fn clock_uncertainty_alone_cannot_exceed_manifest_budget() {
        let run = manifest();
        let mut values = channels();
        values[0].clock_uncertainty_ns = 101;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::ClockUncertaintyExceeded { .. })
        ));
    }

    #[test]
    fn sample_offset_plus_clock_uncertainty_must_fit_combined_budget() {
        let mut run = manifest();
        run.maximum_sample_time_error_ns = 98;
        let mut values = channels();
        values[0].clock_uncertainty_ns = 50;
        values[0].samples = vec![sample(1, 951, 7.5)];
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::Unsynchronized { .. })
        ));
    }

    #[test]
    fn no_valid_sample_is_not_imputed_or_carried_forward() {
        let run = manifest();
        let mut values = channels();
        values[0].samples[1].quality = SampleQuality::Suspect;
        values[0].samples[0].quality = SampleQuality::Invalid;
        values[0].samples[2].quality = SampleQuality::Invalid;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::NoValidSample { .. })
        ));
    }

    #[test]
    fn suspect_samples_are_skipped_and_never_promoted() {
        let run = manifest();
        let mut values = channels();
        values[0].samples[1].quality = SampleQuality::Suspect;
        let result = synchronize_at(&run, &values, &["torque"], 1_000).unwrap();
        assert_eq!(result.observations[0].source_timestamp_ns, 900);
        assert_eq!(result.observations[0].value, 7.4);
    }

    #[test]
    fn mismatched_run_asset_configuration_or_context_fails() {
        let run = manifest();
        for (field, value) in [
            ("run", "other-run"),
            ("asset", "other-motor"),
            ("configuration", "other-config"),
            ("context", "other-context"),
        ] {
            let mut values = channels();
            match field {
                "run" => values[0].run_id = value.to_owned(),
                "asset" => values[0].asset_id = value.to_owned(),
                "configuration" => values[0].configuration_digest = value.to_owned(),
                _ => values[0].context_digest = value.to_owned(),
            }
            assert!(matches!(
                synchronize_at(&run, &values, &["torque"], 1_000),
                Err(ObservationError::InvalidChannel(_))
            ));
        }
    }

    #[test]
    fn quantity_unit_mismatch_is_rejected() {
        let run = manifest();
        let mut values = channels();
        values[0].unit = MeasurementUnit::Newton;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::InvalidUnit { .. })
        ));
    }

    #[test]
    fn expired_or_unknown_calibration_is_rejected() {
        let run = manifest();
        for status in [CalibrationStatus::Expired, CalibrationStatus::Unknown] {
            let mut values = channels();
            values[0].calibration_status = status;
            assert!(matches!(
                synchronize_at(&run, &values, &["torque"], 1_000),
                Err(ObservationError::CalibrationNotCurrent { .. })
            ));
        }
    }

    #[test]
    fn missing_calibration_raw_or_clock_identity_is_rejected() {
        let run = manifest();
        for identity in ["calibration", "raw", "clock"] {
            let mut values = channels();
            match identity {
                "calibration" => values[0].calibration_certificate_digest.clear(),
                "raw" => values[0].raw_data_digest.clear(),
                _ => values[0].clock_source_id.clear(),
            }
            assert!(matches!(
                synchronize_at(&run, &values, &["torque"], 1_000),
                Err(ObservationError::InvalidChannel(_))
            ));
        }
    }

    #[test]
    fn out_of_order_sequence_or_timestamp_is_rejected() {
        let run = manifest();
        let mut values = channels();
        values[0].samples.swap(0, 1);
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::OutOfOrderSamples { .. })
        ));

        let mut values = channels();
        values[0].samples[1].timestamp_ns = 900;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::OutOfOrderSamples { .. })
                | Err(ObservationError::DuplicateSampleTimestamp { .. })
        ));
    }

    #[test]
    fn duplicate_channel_and_duplicate_timestamp_are_rejected() {
        let run = manifest();
        let mut values = channels();
        values.push(values[0].clone());
        assert_eq!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::DuplicateIdentity("channel ID"))
        );

        let mut values = channels();
        values[0].samples[1].timestamp_ns = values[0].samples[0].timestamp_ns;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::DuplicateSampleTimestamp { .. })
        ));
    }

    #[test]
    fn non_finite_values_and_negative_uncertainty_fail_closed() {
        let run = manifest();
        let mut values = channels();
        values[0].samples[1].value = f64::NAN;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::InvalidSample { .. })
        ));

        let mut values = channels();
        values[0].samples[1].absolute_uncertainty = -0.1;
        assert!(matches!(
            synchronize_at(&run, &values, &["torque"], 1_000),
            Err(ObservationError::InvalidSample { .. })
        ));
    }

    #[test]
    fn zero_is_a_valid_measured_value_and_replay_is_deterministic() {
        let run = manifest();
        let mut values = channels();
        values[0].samples[1].value = 0.0;
        let first = synchronize_at(&run, &values, &["torque"], 1_000).unwrap();
        let second = synchronize_at(&run, &values, &["torque"], 1_000).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.observations[0].value, 0.0);
        assert_eq!(first.observations[0].absolute_uncertainty, 0.01);
    }

    #[test]
    fn missing_required_channel_and_duplicate_requirement_fail() {
        let run = manifest();
        let values = channels();
        assert!(matches!(
            synchronize_at(&run, &values, &["missing"], 1_000),
            Err(ObservationError::MissingRequiredChannel(_))
        ));
        assert_eq!(
            synchronize_at(&run, &values, &["torque", "torque"], 1_000),
            Err(ObservationError::DuplicateIdentity("required channel ID"))
        );
    }
}
