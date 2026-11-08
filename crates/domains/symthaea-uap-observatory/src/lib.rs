// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibration-aware, provenance-preserving observation contracts.
//!
//! This crate begins at the scientific data-quality boundary:
//! - preserve sensor modality, coordinate frame, TAI timestamp and raw digest;
//! - preserve explicit measurement and calibration uncertainties;
//! - validate calibration validity windows and timestamp integrity;
//! - report evidence completeness without forcing a physical classification.
//!
//! It does not classify an observation as extraterrestrial, non-human or a
//! propulsion event. "Unresolved" is an admissible downstream outcome.
//! Missing calibration is never silently replaced with a guessed value.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Instrument family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SensorModality {
    ElectroOptical,
    Infrared,
    Radar,
    RadioFrequency,
    Magnetometer,
    ImuGnss,
    Other,
}

/// Coordinate system in which a measurement is expressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateFrame {
    EarthCenteredEarthFixed,
    LocalEastNorthUp,
    SensorBody,
    FrequencyDomain,
    Unknown,
}

/// A measurement with its reported one-sigma uncertainties.
#[derive(Clone, Debug, PartialEq)]
pub enum Measurement {
    Bearing {
        azimuth_rad: f64,
        elevation_rad: f64,
        sigma_azimuth_rad: f64,
        sigma_elevation_rad: f64,
    },
    RangeBearing {
        range_m: f64,
        azimuth_rad: f64,
        elevation_rad: f64,
        sigma_range_m: f64,
        sigma_azimuth_rad: f64,
        sigma_elevation_rad: f64,
    },
    PositionEcef {
        x_m: f64,
        y_m: f64,
        z_m: f64,
        sigma_x_m: f64,
        sigma_y_m: f64,
        sigma_z_m: f64,
    },
    RfSpectrum {
        center_frequency_hz: f64,
        bandwidth_hz: f64,
        sigma_frequency_hz: f64,
    },
    MagneticVector {
        x_tesla: f64,
        y_tesla: f64,
        z_tesla: f64,
        sigma_tesla: f64,
    },
}

impl Measurement {
    pub fn validate(&self) -> Result<(), EvidenceError> {
        match self {
            Self::Bearing {
                azimuth_rad,
                elevation_rad,
                sigma_azimuth_rad,
                sigma_elevation_rad,
            } => {
                finite("azimuth_rad", *azimuth_rad)?;
                bounded(
                    "elevation_rad",
                    *elevation_rad,
                    -std::f64::consts::FRAC_PI_2,
                    std::f64::consts::FRAC_PI_2,
                )?;
                positive("sigma_azimuth_rad", *sigma_azimuth_rad)?;
                positive("sigma_elevation_rad", *sigma_elevation_rad)?;
            }
            Self::RangeBearing {
                range_m,
                azimuth_rad,
                elevation_rad,
                sigma_range_m,
                sigma_azimuth_rad,
                sigma_elevation_rad,
            } => {
                nonnegative("range_m", *range_m)?;
                finite("azimuth_rad", *azimuth_rad)?;
                bounded(
                    "elevation_rad",
                    *elevation_rad,
                    -std::f64::consts::FRAC_PI_2,
                    std::f64::consts::FRAC_PI_2,
                )?;
                positive("sigma_range_m", *sigma_range_m)?;
                positive("sigma_azimuth_rad", *sigma_azimuth_rad)?;
                positive("sigma_elevation_rad", *sigma_elevation_rad)?;
            }
            Self::PositionEcef {
                x_m,
                y_m,
                z_m,
                sigma_x_m,
                sigma_y_m,
                sigma_z_m,
            } => {
                finite("x_m", *x_m)?;
                finite("y_m", *y_m)?;
                finite("z_m", *z_m)?;
                positive("sigma_x_m", *sigma_x_m)?;
                positive("sigma_y_m", *sigma_y_m)?;
                positive("sigma_z_m", *sigma_z_m)?;
            }
            Self::RfSpectrum {
                center_frequency_hz,
                bandwidth_hz,
                sigma_frequency_hz,
            } => {
                positive("center_frequency_hz", *center_frequency_hz)?;
                nonnegative("bandwidth_hz", *bandwidth_hz)?;
                positive("sigma_frequency_hz", *sigma_frequency_hz)?;
            }
            Self::MagneticVector {
                x_tesla,
                y_tesla,
                z_tesla,
                sigma_tesla,
            } => {
                finite("x_tesla", *x_tesla)?;
                finite("y_tesla", *y_tesla)?;
                finite("z_tesla", *z_tesla)?;
                positive("sigma_tesla", *sigma_tesla)?;
            }
        }
        Ok(())
    }

    pub fn required_frame(&self) -> CoordinateFrame {
        match self {
            Self::Bearing { .. } | Self::RangeBearing { .. } => CoordinateFrame::SensorBody,
            Self::PositionEcef { .. } => CoordinateFrame::EarthCenteredEarthFixed,
            Self::RfSpectrum { .. } => CoordinateFrame::FrequencyDomain,
            Self::MagneticVector { .. } => CoordinateFrame::LocalEastNorthUp,
        }
    }

    fn supported_by(&self, modality: SensorModality) -> bool {
        match modality {
            SensorModality::ElectroOptical | SensorModality::Infrared => matches!(
                self,
                Self::Bearing { .. } | Self::RangeBearing { .. } | Self::PositionEcef { .. }
            ),
            SensorModality::Radar => {
                matches!(self, Self::RangeBearing { .. } | Self::PositionEcef { .. })
            }
            SensorModality::RadioFrequency => matches!(self, Self::RfSpectrum { .. }),
            SensorModality::Magnetometer => matches!(self, Self::MagneticVector { .. }),
            SensorModality::ImuGnss => matches!(self, Self::PositionEcef { .. }),
            SensorModality::Other => true,
        }
    }
}

/// Non-identifying device context useful for interpreting a sensor record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SensorMetadata {
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub firmware_version: Option<String>,
    pub sample_rate_hz: Option<f64>,
    pub bit_depth: Option<u16>,
}

impl SensorMetadata {
    fn validate(&self) -> Result<(), EvidenceError> {
        if let Some(rate) = self.sample_rate_hz {
            positive("sample_rate_hz", rate)?;
        }
        if self.bit_depth == Some(0) {
            return Err(EvidenceError::InvalidMetadata("bit_depth must be nonzero"));
        }
        Ok(())
    }

    pub fn completeness_score(&self) -> u8 {
        [
            self.manufacturer.is_some(),
            self.model.is_some(),
            self.firmware_version.is_some(),
            self.sample_rate_hz.is_some(),
            self.bit_depth.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count() as u8
    }
}

/// Calibration summary for one sensor and modality.
///
/// clock_offset_ns is defined as: recorded timestamp = true TAI timestamp
/// + clock_offset_ns. Corrected time subtracts this offset. metadata_digest
/// is the SHA-256 digest of an immutable calibration artefact, not a digest
/// of this Rust struct.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorCalibration {
    pub sensor_id: String,
    pub modality: SensorModality,
    pub coordinate_frame: CoordinateFrame,
    pub valid_from_tai_ns: i64,
    pub valid_until_tai_ns: i64,
    pub clock_offset_ns: i64,
    pub clock_uncertainty_ns: f64,
    pub boresight_uncertainty_rad: Option<f64>,
    pub range_uncertainty_m: Option<f64>,
    pub position_uncertainty_m: Option<f64>,
    pub metadata: SensorMetadata,
    pub metadata_digest: [u8; 32],
}

impl SensorCalibration {
    pub fn validate(&self) -> Result<(), EvidenceError> {
        nonempty("sensor_id", &self.sensor_id)?;
        if self.valid_from_tai_ns >= self.valid_until_tai_ns {
            return Err(EvidenceError::InvalidCalibrationWindow {
                sensor_id: self.sensor_id.clone(),
            });
        }
        positive("clock_uncertainty_ns", self.clock_uncertainty_ns)?;
        if let Some(value) = self.boresight_uncertainty_rad {
            positive("boresight_uncertainty_rad", value)?;
        }
        if let Some(value) = self.range_uncertainty_m {
            positive("range_uncertainty_m", value)?;
        }
        if let Some(value) = self.position_uncertainty_m {
            positive("position_uncertainty_m", value)?;
        }
        self.metadata.validate()?;
        if self.metadata_digest.iter().all(|byte| *byte == 0) {
            return Err(EvidenceError::MissingCalibrationDigest {
                sensor_id: self.sensor_id.clone(),
            });
        }
        Ok(())
    }

    pub fn valid_at(&self, timestamp_tai_ns: i64) -> bool {
        timestamp_tai_ns >= self.valid_from_tai_ns && timestamp_tai_ns < self.valid_until_tai_ns
    }

    pub fn corrected_timestamp_tai_ns(
        &self,
        recorded_timestamp_tai_ns: i64,
    ) -> Result<i64, EvidenceError> {
        recorded_timestamp_tai_ns
            .checked_sub(self.clock_offset_ns)
            .ok_or_else(|| EvidenceError::TimestampOverflow {
                sensor_id: self.sensor_id.clone(),
            })
    }
}

/// Immutable observation record. Raw evidence remains external and is
/// addressed by its SHA-256 digest; this record never overwrites source data.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationRecord {
    pub sensor_id: String,
    /// Recorded TAI nanoseconds before applying the calibrated clock offset.
    pub recorded_timestamp_tai_ns: i64,
    pub sequence_number: u64,
    pub modality: SensorModality,
    pub coordinate_frame: CoordinateFrame,
    pub raw_sha256: [u8; 32],
    pub measurement: Measurement,
}

impl ObservationRecord {
    pub fn validate(&self) -> Result<(), EvidenceError> {
        nonempty("sensor_id", &self.sensor_id)?;
        if self.raw_sha256.iter().all(|byte| *byte == 0) {
            return Err(EvidenceError::MissingRawDigest {
                sensor_id: self.sensor_id.clone(),
                sequence_number: self.sequence_number,
            });
        }
        self.measurement.validate()?;
        if !self.measurement.supported_by(self.modality) {
            return Err(EvidenceError::MeasurementModalityMismatch {
                sensor_id: self.sensor_id.clone(),
            });
        }
        let expected_frame = self.measurement.required_frame();
        if self.coordinate_frame != expected_frame {
            return Err(EvidenceError::FrameMismatch {
                sensor_id: self.sensor_id.clone(),
                expected: expected_frame,
                actual: self.coordinate_frame,
            });
        }
        Ok(())
    }
}

/// Structural validation errors. Missing calibration is reported in the
/// quality summary rather than making the raw observation disappear.
#[derive(Clone, Debug, PartialEq)]
pub enum EvidenceError {
    EmptyIdentifier(&'static str),
    NonFiniteValue(&'static str),
    NonPositiveValue(&'static str),
    NegativeValue(&'static str),
    OutOfBounds(&'static str),
    InvalidMetadata(&'static str),
    InvalidCalibrationWindow {
        sensor_id: String,
    },
    MissingCalibrationDigest {
        sensor_id: String,
    },
    MissingRawDigest {
        sensor_id: String,
        sequence_number: u64,
    },
    DuplicateCalibration {
        sensor_id: String,
    },
    DuplicateObservation {
        sensor_id: String,
        sequence_number: u64,
    },
    CalibrationModalityMismatch {
        sensor_id: String,
    },
    CalibrationFrameMismatch {
        sensor_id: String,
    },
    CalibrationNotValidAtObservation {
        sensor_id: String,
        corrected_timestamp_tai_ns: i64,
    },
    MeasurementModalityMismatch {
        sensor_id: String,
    },
    FrameMismatch {
        sensor_id: String,
        expected: CoordinateFrame,
        actual: CoordinateFrame,
    },
    TimestampOverflow {
        sensor_id: String,
    },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "{field} must not be empty"),
            Self::NonFiniteValue(field) => write!(f, "{field} must be finite"),
            Self::NonPositiveValue(field) => write!(f, "{field} must be greater than zero"),
            Self::NegativeValue(field) => write!(f, "{field} must be nonnegative"),
            Self::OutOfBounds(field) => write!(f, "{field} is outside its physical bounds"),
            Self::InvalidMetadata(message) => write!(f, "{message}"),
            Self::InvalidCalibrationWindow { sensor_id } => {
                write!(f, "invalid calibration window for sensor {sensor_id}")
            }
            Self::MissingCalibrationDigest { sensor_id } => {
                write!(f, "sensor {sensor_id} calibration digest is missing")
            }
            Self::MissingRawDigest {
                sensor_id,
                sequence_number,
            } => write!(
                f,
                "sensor {sensor_id} observation {sequence_number} has no raw digest"
            ),
            Self::DuplicateCalibration { sensor_id } => {
                write!(f, "duplicate calibration for sensor {sensor_id}")
            }
            Self::DuplicateObservation {
                sensor_id,
                sequence_number,
            } => write!(f, "duplicate observation {sensor_id}:{sequence_number}"),
            Self::CalibrationModalityMismatch { sensor_id } => {
                write!(f, "calibration modality mismatch for sensor {sensor_id}")
            }
            Self::CalibrationFrameMismatch { sensor_id } => {
                write!(f, "calibration frame mismatch for sensor {sensor_id}")
            }
            Self::CalibrationNotValidAtObservation {
                sensor_id,
                corrected_timestamp_tai_ns,
            } => write!(
                f,
                "calibration for sensor {sensor_id} is not valid at corrected TAI timestamp {corrected_timestamp_tai_ns}"
            ),
            Self::MeasurementModalityMismatch { sensor_id } => write!(
                f,
                "measurement is incompatible with sensor modality for {sensor_id}"
            ),
            Self::FrameMismatch {
                sensor_id,
                expected,
                actual,
            } => write!(
                f,
                "sensor {sensor_id} frame mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::TimestampOverflow { sensor_id } => write!(
                f,
                "clock correction overflows TAI timestamp for {sensor_id}"
            ),
        }
    }
}

impl Error for EvidenceError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceTier {
    NoObservations,
    Uncalibrated,
    MixedCalibration,
    SingleSensorCalibrated,
    MultiSensorCalibrated,
}

/// Evidence completeness, not a conclusion about the identity or nature of
/// any observed object.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceQuality {
    pub observations: usize,
    pub unique_sensors: usize,
    pub calibrated_observations: usize,
    pub uncalibrated_observations: usize,
    pub metadata_fields_present: usize,
    pub metadata_fields_total: usize,
    pub tier: EvidenceTier,
}

impl EvidenceQuality {
    pub fn metadata_completeness(&self) -> Option<f64> {
        (self.metadata_fields_total > 0)
            .then(|| self.metadata_fields_present as f64 / self.metadata_fields_total as f64)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvidenceBundle {
    pub calibrations: Vec<SensorCalibration>,
    pub observations: Vec<ObservationRecord>,
}

impl EvidenceBundle {
    /// Validate raw record structure and any supplied calibration link.
    ///
    /// Uncalibrated observations remain in the bundle and lower its quality
    /// tier; calibration absence is never silently repaired or imputed.
    pub fn validate(&self) -> Result<(), EvidenceError> {
        let mut calibrations = BTreeMap::new();
        for calibration in &self.calibrations {
            calibration.validate()?;
            if calibrations
                .insert(calibration.sensor_id.as_str(), calibration)
                .is_some()
            {
                return Err(EvidenceError::DuplicateCalibration {
                    sensor_id: calibration.sensor_id.clone(),
                });
            }
        }

        let mut observation_keys = BTreeSet::new();
        for observation in &self.observations {
            observation.validate()?;
            if !observation_keys
                .insert((observation.sensor_id.as_str(), observation.sequence_number))
            {
                return Err(EvidenceError::DuplicateObservation {
                    sensor_id: observation.sensor_id.clone(),
                    sequence_number: observation.sequence_number,
                });
            }
            if let Some(calibration) = calibrations.get(observation.sensor_id.as_str()) {
                if calibration.modality != observation.modality {
                    return Err(EvidenceError::CalibrationModalityMismatch {
                        sensor_id: observation.sensor_id.clone(),
                    });
                }
                if calibration.coordinate_frame != observation.coordinate_frame {
                    return Err(EvidenceError::CalibrationFrameMismatch {
                        sensor_id: observation.sensor_id.clone(),
                    });
                }
                let corrected_timestamp = calibration
                    .corrected_timestamp_tai_ns(observation.recorded_timestamp_tai_ns)?;
                if !calibration.valid_at(corrected_timestamp) {
                    return Err(EvidenceError::CalibrationNotValidAtObservation {
                        sensor_id: observation.sensor_id.clone(),
                        corrected_timestamp_tai_ns: corrected_timestamp,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn quality(&self) -> Result<EvidenceQuality, EvidenceError> {
        self.validate()?;
        let calibration_by_sensor: BTreeMap<&str, &SensorCalibration> = self
            .calibrations
            .iter()
            .map(|calibration| (calibration.sensor_id.as_str(), calibration))
            .collect();
        let unique_sensors: BTreeSet<&str> = self
            .observations
            .iter()
            .map(|observation| observation.sensor_id.as_str())
            .collect();
        let calibrated_observations = self
            .observations
            .iter()
            .filter(|observation| {
                calibration_by_sensor.contains_key(observation.sensor_id.as_str())
            })
            .count();
        let uncalibrated_observations = self.observations.len() - calibrated_observations;
        let metadata_fields_present: usize = self
            .calibrations
            .iter()
            .map(|calibration| calibration.metadata.completeness_score() as usize)
            .sum();
        let metadata_fields_total = self.calibrations.len() * 5;
        let tier = if self.observations.is_empty() {
            EvidenceTier::NoObservations
        } else if calibrated_observations == 0 {
            EvidenceTier::Uncalibrated
        } else if uncalibrated_observations > 0 {
            EvidenceTier::MixedCalibration
        } else if unique_sensors.len() == 1 {
            EvidenceTier::SingleSensorCalibrated
        } else {
            EvidenceTier::MultiSensorCalibrated
        };

        Ok(EvidenceQuality {
            observations: self.observations.len(),
            unique_sensors: unique_sensors.len(),
            calibrated_observations,
            uncalibrated_observations,
            metadata_fields_present,
            metadata_fields_total,
            tier,
        })
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), EvidenceError> {
    if value.trim().is_empty() {
        Err(EvidenceError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), EvidenceError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvidenceError::NonFiniteValue(field))
    }
}

fn positive(field: &'static str, value: f64) -> Result<(), EvidenceError> {
    finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(EvidenceError::NonPositiveValue(field))
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), EvidenceError> {
    finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(EvidenceError::NegativeValue(field))
    }
}

fn bounded(
    field: &'static str,
    value: f64,
    minimum: f64,
    maximum: f64,
) -> Result<(), EvidenceError> {
    finite(field, value)?;
    if (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(EvidenceError::OutOfBounds(field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration(sensor_id: &str, modality: SensorModality) -> SensorCalibration {
        SensorCalibration {
            sensor_id: sensor_id.to_string(),
            modality,
            coordinate_frame: CoordinateFrame::SensorBody,
            valid_from_tai_ns: 1_000,
            valid_until_tai_ns: 10_000,
            clock_offset_ns: -12,
            clock_uncertainty_ns: 50.0,
            boresight_uncertainty_rad: Some(0.0001),
            range_uncertainty_m: Some(2.0),
            position_uncertainty_m: None,
            metadata: SensorMetadata {
                manufacturer: Some("Test Instruments".to_string()),
                model: Some("Calibrated-1".to_string()),
                firmware_version: Some("1.0".to_string()),
                sample_rate_hz: Some(60.0),
                bit_depth: Some(12),
            },
            metadata_digest: [0xA5; 32],
        }
    }

    fn bearing(
        sensor_id: &str,
        modality: SensorModality,
        sequence_number: u64,
        timestamp_tai_ns: i64,
    ) -> ObservationRecord {
        ObservationRecord {
            sensor_id: sensor_id.to_string(),
            recorded_timestamp_tai_ns: timestamp_tai_ns,
            sequence_number,
            modality,
            coordinate_frame: CoordinateFrame::SensorBody,
            raw_sha256: [0x5A; 32],
            measurement: Measurement::Bearing {
                azimuth_rad: 0.3,
                elevation_rad: 0.1,
                sigma_azimuth_rad: 0.001,
                sigma_elevation_rad: 0.0015,
            },
        }
    }

    #[test]
    fn two_calibrated_sensors_are_multisensor_evidence_not_object_identity() {
        let bundle = EvidenceBundle {
            calibrations: vec![
                calibration("eo-01", SensorModality::ElectroOptical),
                calibration("ir-02", SensorModality::Infrared),
            ],
            observations: vec![
                bearing("eo-01", SensorModality::ElectroOptical, 1, 2_000),
                bearing("ir-02", SensorModality::Infrared, 1, 2_000),
            ],
        };
        let quality = bundle.quality().unwrap();
        assert_eq!(quality.tier, EvidenceTier::MultiSensorCalibrated);
        assert_eq!(quality.unique_sensors, 2);
        assert_eq!(quality.calibrated_observations, 2);
        assert_eq!(quality.metadata_completeness(), Some(1.0));
    }

    #[test]
    fn missing_calibration_is_preserved_and_reported_not_imputed() {
        let bundle = EvidenceBundle {
            observations: vec![bearing(
                "unknown-sensor",
                SensorModality::ElectroOptical,
                7,
                2_000,
            )],
            calibrations: vec![],
        };
        let quality = bundle.quality().unwrap();
        assert_eq!(quality.tier, EvidenceTier::Uncalibrated);
        assert_eq!(quality.uncalibrated_observations, 1);
    }

    #[test]
    fn expired_calibration_is_rejected_for_a_linked_observation() {
        let observation = bearing("eo-01", SensorModality::ElectroOptical, 1, 10_000);
        let bundle = EvidenceBundle {
            calibrations: vec![calibration("eo-01", SensorModality::ElectroOptical)],
            observations: vec![observation],
        };
        assert!(matches!(
            bundle.validate(),
            Err(EvidenceError::CalibrationNotValidAtObservation { .. })
        ));
    }

    #[test]
    fn corrected_clock_time_controls_calibration_validity() {
        let mut calibration = calibration("eo-01", SensorModality::ElectroOptical);
        calibration.clock_offset_ns = 12;
        let observation = bearing("eo-01", SensorModality::ElectroOptical, 1, 10_005);
        // The recorded timestamp is just outside the window, but the
        // calibrated true TAI timestamp is 9,993 ns and remains valid.
        let bundle = EvidenceBundle {
            calibrations: vec![calibration],
            observations: vec![observation],
        };
        assert!(bundle.validate().is_ok());
    }

    #[test]
    fn clock_correction_overflow_is_rejected() {
        let mut calibration = calibration("eo-01", SensorModality::ElectroOptical);
        calibration.clock_offset_ns = i64::MIN;
        assert!(matches!(
            calibration.corrected_timestamp_tai_ns(i64::MAX),
            Err(EvidenceError::TimestampOverflow { .. })
        ));
    }

    #[test]
    fn malformed_uncertainties_and_frames_fail_closed() {
        let mut observation = bearing("eo-01", SensorModality::ElectroOptical, 1, 2_000);
        observation.measurement = Measurement::Bearing {
            azimuth_rad: 0.0,
            elevation_rad: 0.0,
            sigma_azimuth_rad: 0.0,
            sigma_elevation_rad: f64::NAN,
        };
        assert!(observation.validate().is_err());

        let mut observation = bearing("eo-01", SensorModality::ElectroOptical, 2, 2_000);
        observation.coordinate_frame = CoordinateFrame::EarthCenteredEarthFixed;
        assert!(matches!(
            observation.validate(),
            Err(EvidenceError::FrameMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_raw_records_are_rejected() {
        let record = bearing("eo-01", SensorModality::ElectroOptical, 1, 2_000);
        let bundle = EvidenceBundle {
            observations: vec![record.clone(), record],
            calibrations: vec![],
        };
        assert!(matches!(
            bundle.validate(),
            Err(EvidenceError::DuplicateObservation { .. })
        ));
    }

    #[test]
    fn clock_offset_is_applied_without_losing_tai_scale() {
        let record = calibration("radar-01", SensorModality::Radar);
        assert_eq!(record.corrected_timestamp_tai_ns(5_000).unwrap(), 5_012);
    }
}

/// Calibrated bearing-only multi-sensor triangulation and local uncertainty.
pub mod triangulation;

pub use triangulation::{
    BearingPositionClockJacobian, Quaternion as SensorToEcefQuaternion, RayObservation,
    RobustSensorPlacementScore, SensorPlacementCandidate, SensorPlacementScore, TargetHypothesis,
    TriangulatedPosition, TriangulationConfig, TriangulationError, Vector3 as EcefVector3,
    bearing_fisher_information, bearing_position_clock_jacobian, rank_sensor_placements,
    rank_sensor_placements_over_target_region, triangulate_bearings,
    triangulate_bearings_with_config,
};

/// Frame-level multi-target GOSPA metrics.
pub mod tracking_metrics;

pub use tracking_metrics::{
    GospaBreakdown, GospaConfig, GospaError, MAX_GOSPA_ASSIGNMENT_SIDE, gospa_frame,
};

/// Fixed-association OSPA(2) track-set evaluation over a discrete time window.
pub mod trajectory_metrics;

pub use trajectory_metrics::{
    MAX_OSPA2_ASSIGNMENT_SIDE, MAX_OSPA2_WINDOW_LENGTH, Ospa2Breakdown, Ospa2Config, Ospa2Error,
    TrackSeries, ospa2_track_distance, ospa2_track_sets,
};

/// Budgeted multi-scenario experimental design with an exhaustive small-set oracle.
pub mod budgeted_design;

pub use budgeted_design::{
    BudgetedDesignConfig, BudgetedDesignError, BudgetedDesignReport, DesignScenario,
    DesignSelectionStep, ExperimentOption, ScenarioDesignScore, exhaustive_budgeted_design,
    greedy_budgeted_design,
};

/// Data-only Jacobian rank and singular-spectrum diagnostics.
pub mod observability;

pub use observability::{
    MAX_OBSERVABILITY_PARAMETERS, ObservabilityError, ObservabilityReport, analyze_jacobian,
};

/// Residual-qualified screening for discrete global solution ambiguities.
pub mod global_identifiability;

pub use global_identifiability::{
    AmbiguousSolutionPair, CandidateFit, CandidateSolution, GlobalIdentifiabilityConfig,
    GlobalIdentifiabilityError, GlobalIdentifiabilityReport, MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES,
    screen_global_ambiguities,
};
