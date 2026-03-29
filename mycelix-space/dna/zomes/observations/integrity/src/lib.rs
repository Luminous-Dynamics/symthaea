// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observations Integrity Zome
//!
//! Defines entry types for sensor observations of orbital objects.
//! Observations are the raw data that feeds into the catalog and
//! conjunction analysis systems.

use hdi::prelude::*;
use mycelix_space_shared::{GroundLocation, QualityScore, SpaceTimestamp};

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// Raw observation from a sensor
    Observation(Observation),

    /// Sensor registration
    Sensor(Sensor),

    /// Batch of observations
    ObservationBatch(ObservationBatch),

    /// Multi-sensor fused state estimate
    FusedEstimate(FusedEstimate),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Observations for an object
    ObjectObservations,
    /// Observations from a sensor
    SensorObservations,
    /// All sensors index
    AllSensors,
    /// Fused estimates for an object
    ObjectFusedEstimates,
}

/// A single observation of a space object
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Observation {
    /// Object observed (may be unknown/uncorrelated)
    pub norad_id: Option<u32>,

    /// Observation time
    pub observation_time: SpaceTimestamp,

    /// Observer location (for ground sensors)
    pub observer_location: Option<GroundLocation>,

    /// Observation type
    pub observation_type: ObservationType,

    /// Measurement data
    pub measurement: Measurement,

    /// Quality score
    pub quality: QualityScore,

    /// Sensor that made this observation
    pub sensor_id: String,

    /// Submitting agent
    pub submitted_by: AgentPubKey,

    /// Submission time
    pub submitted_at: SpaceTimestamp,
}

/// Type of observation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObservationType {
    /// Optical (telescope)
    Optical,
    /// Radar
    Radar,
    /// Radio frequency
    Rf,
    /// Laser ranging
    Laser,
    /// Passive RF (signal detection)
    PassiveRf,
}

/// Measurement data from observation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Measurement {
    /// Right ascension / declination angles
    AnglesOnly {
        ra_deg: f64,
        dec_deg: f64,
        ra_sigma_deg: Option<f64>,
        dec_sigma_deg: Option<f64>,
    },
    /// Range measurement
    Range {
        range_km: f64,
        range_rate_kms: Option<f64>,
        range_sigma_km: Option<f64>,
    },
    /// Full state vector observation
    StateVector {
        position_km: [f64; 3],
        velocity_kms: Option<[f64; 3]>,
        covariance: Option<[f64; 21]>,
    },
    /// Visual magnitude
    Photometric {
        magnitude: f64,
        magnitude_sigma: Option<f64>,
        filter: Option<String>,
    },
}

/// Sensor registration
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Sensor {
    /// Unique sensor ID
    pub sensor_id: String,

    /// Sensor name
    pub name: String,

    /// Sensor type
    pub sensor_type: ObservationType,

    /// Location (for fixed ground sensors)
    pub location: Option<GroundLocation>,

    /// Operator/owner
    pub operator: AgentPubKey,

    /// Capabilities description
    pub capabilities: SensorCapabilities,

    /// Registration time
    pub registered_at: SpaceTimestamp,
}

/// Sensor capabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorCapabilities {
    /// Minimum detectable size (m)
    pub min_size_m: Option<f64>,
    /// Maximum range (km)
    pub max_range_km: Option<f64>,
    /// Field of view (degrees)
    pub fov_deg: Option<f64>,
    /// Tracking accuracy (arcsec)
    pub accuracy_arcsec: Option<f64>,
}

/// Batch of observations
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ObservationBatch {
    /// Batch ID
    pub batch_id: String,

    /// Sensor that made these observations
    pub sensor_id: String,

    /// Start time of batch
    pub start_time: SpaceTimestamp,

    /// End time of batch
    pub end_time: SpaceTimestamp,

    /// Number of observations in batch
    pub observation_count: u32,

    /// Hash of the observation data
    pub data_hash: [u8; 32],
}

/// Multi-sensor fused state estimate for an orbital object.
///
/// Created by `fuse_observations_for_object()` in the coordinator — combines
/// observations from multiple sensors using inverse-variance weighted fusion.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct FusedEstimate {
    /// NORAD catalog ID
    pub norad_id: u32,

    /// Epoch of the fused estimate
    pub epoch: SpaceTimestamp,

    /// State vector [x, y, z, vx, vy, vz] in km and km/s (TEME frame)
    pub state_vector: Vec<f64>,

    /// Diagonal of the 6×6 covariance matrix [σ_x², σ_y², σ_z², σ_vx², σ_vy², σ_vz²]
    pub covariance_diagonal: Vec<f64>,

    /// Sensor IDs that contributed to this estimate
    pub contributing_sensors: Vec<String>,

    /// Fused quality score (0.0 = poor, 1.0 = excellent)
    pub fused_quality: f64,

    /// Chi-square consistency statistic (lower = more consistent inputs)
    pub chi_square_consistency: f64,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Observation(obs) => validate_observation(&obs),
            EntryTypes::Sensor(sensor) => validate_sensor(&sensor),
            EntryTypes::ObservationBatch(batch) => validate_batch(&batch),
            EntryTypes::FusedEstimate(est) => validate_fused_estimate(&est),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_observation(obs: &Observation) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID if present must be valid
    if let Some(norad_id) = obs.norad_id {
        if norad_id == 0 || norad_id > 999999 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid NORAD ID: {}",
                norad_id
            )));
        }
    }

    // Sensor ID must not be empty
    if obs.sensor_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Sensor ID cannot be empty".to_string(),
        ));
    }

    // Validate observer location if present
    if let Some(ref loc) = obs.observer_location {
        if let Err(msg) = validate_ground_location(loc) {
            return Ok(ValidateCallbackResult::Invalid(msg));
        }
    }

    // Validate measurement data
    if let Err(msg) = validate_measurement(&obs.measurement) {
        return Ok(ValidateCallbackResult::Invalid(msg));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_ground_location(loc: &GroundLocation) -> Result<(), String> {
    if loc.latitude_deg < -90.0 || loc.latitude_deg > 90.0 {
        return Err(format!(
            "Latitude must be between -90 and 90 degrees, got {}",
            loc.latitude_deg
        ));
    }
    if loc.longitude_deg < -180.0 || loc.longitude_deg > 180.0 {
        return Err(format!(
            "Longitude must be between -180 and 180 degrees, got {}",
            loc.longitude_deg
        ));
    }
    if loc.altitude_m < -500.0 || loc.altitude_m > 1_000_000.0 {
        return Err(format!(
            "Altitude must be between -500m and 1,000,000m, got {}",
            loc.altitude_m
        ));
    }
    Ok(())
}

fn validate_measurement(m: &Measurement) -> Result<(), String> {
    match m {
        Measurement::AnglesOnly {
            ra_deg,
            dec_deg,
            ra_sigma_deg,
            dec_sigma_deg,
        } => {
            if *ra_deg < 0.0 || *ra_deg >= 360.0 {
                return Err(format!(
                    "Right ascension must be in [0, 360), got {}",
                    ra_deg
                ));
            }
            if *dec_deg < -90.0 || *dec_deg > 90.0 {
                return Err(format!("Declination must be in [-90, 90], got {}", dec_deg));
            }
            if let Some(sigma) = ra_sigma_deg {
                if *sigma < 0.0 {
                    return Err("RA uncertainty (sigma) cannot be negative".to_string());
                }
            }
            if let Some(sigma) = dec_sigma_deg {
                if *sigma < 0.0 {
                    return Err("Dec uncertainty (sigma) cannot be negative".to_string());
                }
            }
        }
        Measurement::Range {
            range_km,
            range_rate_kms: _,
            range_sigma_km,
        } => {
            if *range_km <= 0.0 {
                return Err(format!("Range must be positive, got {} km", range_km));
            }
            if let Some(sigma) = range_sigma_km {
                if *sigma < 0.0 {
                    return Err("Range uncertainty (sigma) cannot be negative".to_string());
                }
            }
        }
        Measurement::StateVector {
            position_km,
            velocity_kms: _,
            covariance,
        } => {
            // Position must not be all zeros (degenerate)
            let pos_mag = position_km.iter().map(|x| x * x).sum::<f64>().sqrt();
            if pos_mag < 100.0 {
                return Err(format!(
                    "Position vector magnitude {} km is below minimum orbital altitude",
                    pos_mag
                ));
            }
            // Covariance diagonal elements must be non-negative
            if let Some(cov) = covariance {
                // Lower triangular: indices 0, 2, 5, 9, 14, 20 are diagonal
                let diag_indices = [0, 2, 5, 9, 14, 20];
                for &i in &diag_indices {
                    if cov[i] < 0.0 {
                        return Err(format!(
                            "Covariance diagonal element [{}] must be non-negative, got {}",
                            i, cov[i]
                        ));
                    }
                }
            }
        }
        Measurement::Photometric {
            magnitude,
            magnitude_sigma,
            filter: _,
        } => {
            // Visual magnitude range: roughly -27 (Sun) to +30 (faintest detectable)
            if *magnitude < -30.0 || *magnitude > 35.0 {
                return Err(format!(
                    "Magnitude {} is outside plausible range [-30, 35]",
                    magnitude
                ));
            }
            if let Some(sigma) = magnitude_sigma {
                if *sigma < 0.0 {
                    return Err("Magnitude uncertainty (sigma) cannot be negative".to_string());
                }
            }
        }
    }
    Ok(())
}

fn validate_sensor(sensor: &Sensor) -> ExternResult<ValidateCallbackResult> {
    if sensor.sensor_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Sensor ID cannot be empty".to_string(),
        ));
    }

    if sensor.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Sensor name cannot be empty".to_string(),
        ));
    }

    // Validate sensor location if present
    if let Some(ref loc) = sensor.location {
        if let Err(msg) = validate_ground_location(loc) {
            return Ok(ValidateCallbackResult::Invalid(msg));
        }
    }

    // Validate sensor capabilities (positive values when present)
    let cap = &sensor.capabilities;
    if let Some(min_size) = cap.min_size_m {
        if min_size <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Sensor min_size_m must be positive, got {}",
                min_size
            )));
        }
    }
    if let Some(max_range) = cap.max_range_km {
        if max_range <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Sensor max_range_km must be positive, got {}",
                max_range
            )));
        }
    }
    if let Some(fov) = cap.fov_deg {
        if fov <= 0.0 || fov > 360.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Sensor fov_deg must be in (0, 360], got {}",
                fov
            )));
        }
    }
    if let Some(accuracy) = cap.accuracy_arcsec {
        if accuracy <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Sensor accuracy_arcsec must be positive, got {}",
                accuracy
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_batch(batch: &ObservationBatch) -> ExternResult<ValidateCallbackResult> {
    if batch.batch_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Batch ID cannot be empty".to_string(),
        ));
    }

    if batch.end_time.micros < batch.start_time.micros {
        return Ok(ValidateCallbackResult::Invalid(
            "Batch end time cannot be before start time".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_fused_estimate(est: &FusedEstimate) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if est.norad_id == 0 || est.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}",
            est.norad_id
        )));
    }

    // State vector must have exactly 6 elements
    if est.state_vector.len() != 6 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "State vector must have 6 elements, got {}",
            est.state_vector.len()
        )));
    }

    // All state vector elements must be finite
    for (i, &v) in est.state_vector.iter().enumerate() {
        if !v.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "State vector element [{}] is not finite: {}",
                i, v
            )));
        }
    }

    // Covariance diagonal must have exactly 6 elements
    if est.covariance_diagonal.len() != 6 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Covariance diagonal must have 6 elements, got {}",
            est.covariance_diagonal.len()
        )));
    }

    // Covariance diagonal elements must be non-negative and finite
    for (i, &v) in est.covariance_diagonal.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Covariance diagonal [{}] must be non-negative and finite, got {}",
                i, v
            )));
        }
    }

    // Quality must be in [0, 1]
    if !est.fused_quality.is_finite() || est.fused_quality < 0.0 || est.fused_quality > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Fused quality must be between 0 and 1, got {}",
            est.fused_quality
        )));
    }

    // Chi-square must be non-negative and finite
    if !est.chi_square_consistency.is_finite() || est.chi_square_consistency < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Chi-square consistency must be non-negative and finite, got {}",
            est.chi_square_consistency
        )));
    }

    // Must have at least one contributing sensor
    if est.contributing_sensors.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have at least one contributing sensor".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
