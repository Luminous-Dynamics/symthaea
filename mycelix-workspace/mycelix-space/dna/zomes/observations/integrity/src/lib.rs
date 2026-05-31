// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Observations Integrity Zome
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
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Observations for an object
    ObjectObservations,
    /// Observations from a sensor
    SensorObservations,
    /// All sensors index
    AllSensors,
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

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Observation(obs) => validate_observation(&obs),
                EntryTypes::Sensor(sensor) => validate_sensor(&sensor),
                EntryTypes::ObservationBatch(batch) => validate_batch(&batch),
            },
            _ => Ok(ValidateCallbackResult::Valid),
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

    Ok(ValidateCallbackResult::Valid)
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
