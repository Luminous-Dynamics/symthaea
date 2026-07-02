// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observations Coordinator Zome
//!
//! Functions for submitting and querying sensor observations.

use hdk::prelude::*;
use mycelix_space_shared::{
    GroundLocation, PaginatedResponse, PaginationParams, QualityScore, SpaceError, SpaceErrorCode,
    SpaceTimestamp, TrustLevel, gate_space_operation, requirement_for_fusion,
    requirement_for_observation, validate_latitude, validate_longitude, validate_string_field,
};
use observations_integrity::*;
use orbital_mechanics::covariance::CovarianceMatrix;
use orbital_mechanics::fusion::{FusionPipeline, SensorMeasurement, TrustWeighting};
use orbital_mechanics::state::{DataSource, StateVector as OmStateVector};

// =============================================================================
// Signal types
// =============================================================================

/// Signal types emitted by the observations zome
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub enum ObservationSignal {
    /// A new observation was submitted
    NewObservation {
        norad_id: Option<u32>,
        sensor_id: String,
        observation_hash: ActionHash,
    },
    /// A new sensor was registered
    SensorRegistered {
        sensor_id: String,
        sensor_hash: ActionHash,
    },
}

// =============================================================================
// Anchor helpers (deterministic DHT-discoverable paths)
// =============================================================================

/// Anchor for all observations of a given NORAD ID.
fn anchor_for_object_observations(norad_id: u32) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("obs_by_object.{}", norad_id));
    let typed = path.typed(LinkTypes::ObjectObservations)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for all observations from a given sensor.
fn anchor_for_sensor_observations(sensor_id: &str) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("obs_by_sensor.{}", sensor_id));
    let typed = path.typed(LinkTypes::SensorObservations)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Anchor for the global sensor index.
fn anchor_for_all_sensors() -> ExternResult<AnyLinkableHash> {
    let path = Path::from("all_sensors");
    let typed = path.typed(LinkTypes::AllSensors)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

// =============================================================================
// Write operations
// =============================================================================

/// Submit a sensor observation to the DHT.
///
/// Links the observation to both its target NORAD ID (if known) and its
/// sensor ID for query discoverability.
///
/// Returns the `ActionHash` of the created observation entry.
#[hdk_extern]
pub fn submit_observation(input: SubmitObservationInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_observation(), "submit_observation")?;
    // --- Input validation ---
    validate_string_field(&input.sensor_id, "sensor_id", 256).map_err(|e| e.into_wasm_error())?;

    if let Some(norad_id) = input.norad_id {
        if norad_id == 0 || norad_id > 999999 {
            return Err(
                SpaceError::new(SpaceErrorCode::InvalidInput, "NORAD ID must be 1-999999")
                    .with_context(format!("got: {}", norad_id))
                    .into_wasm_error(),
            );
        }
    }

    if let Some(ref loc) = input.observer_location {
        validate_latitude(loc.latitude_deg).map_err(|e| e.into_wasm_error())?;
        validate_longitude(loc.longitude_deg).map_err(|e| e.into_wasm_error())?;
    }

    // Validate measurement-specific fields
    match &input.measurement {
        Measurement::AnglesOnly {
            ra_deg, dec_deg, ..
        } => {
            if !(0.0..=360.0).contains(ra_deg) || ra_deg.is_nan() {
                return Err(SpaceError::new(
                    SpaceErrorCode::InvalidMeasurement,
                    "RA must be 0-360 degrees",
                )
                .into_wasm_error());
            }
            if !(-90.0..=90.0).contains(dec_deg) || dec_deg.is_nan() {
                return Err(SpaceError::new(
                    SpaceErrorCode::InvalidMeasurement,
                    "Dec must be -90 to 90 degrees",
                )
                .into_wasm_error());
            }
        }
        Measurement::Range { range_km, .. } => {
            if *range_km < 0.0 || range_km.is_nan() {
                return Err(SpaceError::new(
                    SpaceErrorCode::InvalidMeasurement,
                    "Range must be non-negative",
                )
                .into_wasm_error());
            }
        }
        Measurement::Photometric { magnitude, .. } => {
            if magnitude.is_nan() {
                return Err(SpaceError::new(
                    SpaceErrorCode::InvalidMeasurement,
                    "Magnitude must not be NaN",
                )
                .into_wasm_error());
            }
        }
        Measurement::StateVector { .. } => {} // position/velocity validated structurally
    }

    let agent = agent_info()?.agent_initial_pubkey;

    let observation = Observation {
        norad_id: input.norad_id,
        observation_time: input.observation_time,
        observer_location: input.observer_location,
        observation_type: input.observation_type,
        measurement: input.measurement,
        quality: input.quality.unwrap_or_default(),
        sensor_id: input.sensor_id.clone(),
        submitted_by: agent,
        submitted_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::Observation(observation))?;

    // Link observation to its NORAD ID anchor (if present)
    if let Some(norad_id) = input.norad_id {
        let obj_anchor = anchor_for_object_observations(norad_id)?;
        create_link(
            obj_anchor,
            action_hash.clone(),
            LinkTypes::ObjectObservations,
            LinkTag::new(format!("obs:{}", norad_id)),
        )?;
    }

    // Link observation to its sensor anchor
    let sensor_anchor = anchor_for_sensor_observations(&input.sensor_id)?;
    create_link(
        sensor_anchor,
        action_hash.clone(),
        LinkTypes::SensorObservations,
        LinkTag::new(format!("sensor_obs:{}", input.sensor_id)),
    )?;

    // Emit signal
    emit_signal(ObservationSignal::NewObservation {
        norad_id: input.norad_id,
        sensor_id: input.sensor_id,
        observation_hash: action_hash.clone(),
    })?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitObservationInput {
    pub norad_id: Option<u32>,
    pub observation_time: SpaceTimestamp,
    pub observer_location: Option<GroundLocation>,
    pub observation_type: ObservationType,
    pub measurement: Measurement,
    pub quality: Option<QualityScore>,
    pub sensor_id: String,
}

/// Register a sensor in the global sensor index.
///
/// The sensor is linked to the `all_sensors` anchor for discoverability via
/// `list_sensors()`. Returns the `ActionHash` of the created sensor entry.
#[hdk_extern]
pub fn register_sensor(input: RegisterSensorInput) -> ExternResult<ActionHash> {
    gate_space_operation(&requirement_for_observation(), "register_sensor")?;
    // --- Input validation ---
    validate_string_field(&input.sensor_id, "sensor_id", 256).map_err(|e| e.into_wasm_error())?;
    validate_string_field(&input.name, "name", 256).map_err(|e| e.into_wasm_error())?;

    if let Some(ref loc) = input.location {
        validate_latitude(loc.latitude_deg).map_err(|e| e.into_wasm_error())?;
        validate_longitude(loc.longitude_deg).map_err(|e| e.into_wasm_error())?;
    }

    if let Some(max_range) = input.capabilities.max_range_km {
        if max_range <= 0.0 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidSensorConfig,
                "max_range_km must be positive",
            )
            .into_wasm_error());
        }
    }
    if let Some(accuracy) = input.capabilities.accuracy_arcsec {
        if accuracy <= 0.0 {
            return Err(SpaceError::new(
                SpaceErrorCode::InvalidSensorConfig,
                "accuracy_arcsec must be positive",
            )
            .into_wasm_error());
        }
    }

    let agent = agent_info()?.agent_initial_pubkey;

    let sensor = Sensor {
        sensor_id: input.sensor_id.clone(),
        name: input.name,
        sensor_type: input.sensor_type,
        location: input.location,
        operator: agent,
        capabilities: input.capabilities,
        registered_at: SpaceTimestamp::now(),
    };

    let action_hash = create_entry(&EntryTypes::Sensor(sensor))?;

    // Link sensor to the global sensor index
    let sensors_anchor = anchor_for_all_sensors()?;
    create_link(
        sensors_anchor,
        action_hash.clone(),
        LinkTypes::AllSensors,
        LinkTag::new(format!("sensor:{}", input.sensor_id)),
    )?;

    // Emit signal
    emit_signal(ObservationSignal::SensorRegistered {
        sensor_id: input.sensor_id,
        sensor_hash: action_hash.clone(),
    })?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterSensorInput {
    pub sensor_id: String,
    pub name: String,
    pub sensor_type: ObservationType,
    pub location: Option<GroundLocation>,
    pub capabilities: SensorCapabilities,
}

// =============================================================================
// Query operations
// =============================================================================

/// Get all observations for a specific NORAD ID.
///
/// Queries the `obs_by_object.{norad_id}` anchor on the DHT.
/// Returns observations from all agents, unsorted.
#[hdk_extern]
pub fn get_observations_for_object(norad_id: u32) -> ExternResult<Vec<Observation>> {
    let anchor = anchor_for_object_observations(norad_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ObjectObservations)?,
        GetStrategy::Network,
    )?;

    let mut observations = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(obs) = record.entry().to_app_option::<Observation>().ok().flatten() {
            observations.push(obs);
        }
    }

    Ok(observations)
}

/// Get all observations from a specific sensor.
///
/// Queries the `obs_by_sensor.{sensor_id}` anchor on the DHT.
#[hdk_extern]
pub fn get_sensor_observations(sensor_id: String) -> ExternResult<Vec<Observation>> {
    let anchor = anchor_for_sensor_observations(&sensor_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SensorObservations)?,
        GetStrategy::Network,
    )?;

    let mut observations = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(obs) = record.entry().to_app_option::<Observation>().ok().flatten() {
            observations.push(obs);
        }
    }

    Ok(observations)
}

/// List all registered sensors on the network.
///
/// Returns every sensor linked to the global `all_sensors` anchor.
#[hdk_extern]
pub fn list_sensors(_: ()) -> ExternResult<Vec<Sensor>> {
    let anchor = anchor_for_all_sensors()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllSensors)?,
        GetStrategy::Network,
    )?;

    let mut sensors = Vec::new();
    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(sensor) = record.entry().to_app_option::<Sensor>().ok().flatten() {
            sensors.push(sensor);
        }
    }

    Ok(sensors)
}

// =============================================================================
// Fusion operations
// =============================================================================

/// Anchor for fused estimates of a given NORAD ID.
fn anchor_for_object_fused_estimates(norad_id: u32) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("fused_by_object.{}", norad_id));
    let typed = path.typed(LinkTypes::ObjectFusedEstimates)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}

/// Fuse all observations for an object into a single best-estimate state.
///
/// Fetches observations via `get_observations_for_object`, converts StateVector
/// and Range measurements to `SensorMeasurement`s, runs the fusion pipeline, and
/// stores the result as a `FusedEstimate` entry linked to the object's anchor.
#[hdk_extern]
pub fn fuse_observations_for_object(norad_id: u32) -> ExternResult<FusedEstimate> {
    gate_space_operation(&requirement_for_fusion(), "fuse_observations_for_object")?;
    if norad_id == 0 || norad_id > 999999 {
        return Err(
            SpaceError::new(SpaceErrorCode::InvalidInput, "NORAD ID must be 1-999999")
                .with_context(format!("got: {}", norad_id))
                .into_wasm_error(),
        );
    }

    let observations = get_observations_for_object(norad_id)?;
    if observations.is_empty() {
        return Err(
            SpaceError::new(SpaceErrorCode::NotFound, "No observations found for object")
                .with_context(format!("norad_id: {}", norad_id))
                .into_wasm_error(),
        );
    }

    // Convert observations to SensorMeasurement (only StateVector and Range types)
    let now = chrono::Utc::now();
    let mut measurements = Vec::new();

    for obs in &observations {
        match &obs.measurement {
            Measurement::StateVector {
                position_km,
                velocity_kms,
                covariance: _,
            } => {
                let vel = velocity_kms.unwrap_or([0.0, 0.0, 0.0]);
                // Default covariance: 1 km position, 0.01 km/s velocity
                let sigma_pos = 1.0;
                let sigma_vel = 0.01;

                measurements.push(SensorMeasurement {
                    time: now, // Use current time as common reference
                    state: OmStateVector::new(
                        position_km[0],
                        position_km[1],
                        position_km[2],
                        vel[0],
                        vel[1],
                        vel[2],
                    ),
                    covariance: CovarianceMatrix::diagonal([
                        sigma_pos, sigma_pos, sigma_pos, sigma_vel, sigma_vel, sigma_vel,
                    ]),
                    sensor_id: obs.sensor_id.clone(),
                    data_source: DataSource::GroundObservation {
                        sensor_id: obs.sensor_id.clone(),
                        sensor_type: orbital_mechanics::state::SensorType::Radar,
                    },
                    quality: obs.quality.value() as f64 / 100.0,
                });
            }
            Measurement::Range {
                range_km,
                range_rate_kms: _,
                range_sigma_km,
            } => {
                // Convert range-only to a position estimate along the observer-to-zenith line
                // This is approximate — real systems would use the observer's location
                let sigma = range_sigma_km.unwrap_or(10.0);
                let r = *range_km;

                // Place object at +X axis at the given range as rough position
                measurements.push(SensorMeasurement {
                    time: now,
                    state: OmStateVector::new(r, 0.0, 0.0, 0.0, 0.0, 0.0),
                    covariance: CovarianceMatrix::diagonal([sigma, sigma, sigma, 0.1, 0.1, 0.1]),
                    sensor_id: obs.sensor_id.clone(),
                    data_source: DataSource::GroundObservation {
                        sensor_id: obs.sensor_id.clone(),
                        sensor_type: orbital_mechanics::state::SensorType::Radar,
                    },
                    quality: obs.quality.value() as f64 / 100.0,
                });
            }
            // Skip AnglesOnly and Photometric — insufficient for state vector fusion
            _ => continue,
        }
    }

    if measurements.is_empty() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            "No fusible measurements (need StateVector or Range)",
        )
        .with_context(format!(
            "norad_id: {} had {} observations but none with state/range data",
            norad_id,
            observations.len()
        ))
        .into_wasm_error());
    }

    // Build trust weighting from submitter trust levels.
    // Each observation's sensor_id is mapped to its submitter's trust weight.
    let mut trust_weighting = TrustWeighting::default();
    for obs in &observations {
        let trust_level = lookup_agent_trust_level(&obs.submitted_by);
        trust_weighting
            .sensor_trust
            .entry(obs.sensor_id.clone())
            .or_insert(trust_level.weight());
    }

    // Run fusion pipeline with trust-weighted quality
    let pipeline = FusionPipeline::default().with_trust_weighting(trust_weighting);
    let fused = pipeline.fuse(&measurements).map_err(|e| {
        SpaceError::new(SpaceErrorCode::InvalidInput, "Fusion pipeline failed")
            .with_context(e)
            .into_wasm_error()
    })?;

    // Convert to integrity FusedEstimate
    let sv = &fused.state.state;
    let cov_diag: Vec<f64> = fused
        .state
        .covariance
        .as_ref()
        .map(|c| {
            let m = c.matrix();
            (0..6).map(|i| m[(i, i)]).collect()
        })
        .unwrap_or_else(|| vec![0.0; 6]);

    let estimate = observations_integrity::FusedEstimate {
        norad_id,
        epoch: SpaceTimestamp::now(),
        state_vector: vec![sv.x, sv.y, sv.z, sv.vx, sv.vy, sv.vz],
        covariance_diagonal: cov_diag,
        contributing_sensors: fused.contributing_sensors,
        fused_quality: fused.fused_quality,
        chi_square_consistency: fused.chi_square_consistency,
    };

    // Store on DHT
    let action_hash = create_entry(&EntryTypes::FusedEstimate(estimate.clone()))?;

    // Link to object's fused estimates anchor
    let anchor = anchor_for_object_fused_estimates(norad_id)?;
    create_link(
        anchor,
        action_hash,
        LinkTypes::ObjectFusedEstimates,
        LinkTag::new(format!("fused:{}", norad_id)),
    )?;

    Ok(estimate)
}

/// Get the most recent fused state estimate for an object.
///
/// Returns `None` if no fused estimate exists for this NORAD ID.
#[hdk_extern]
pub fn get_fused_state(norad_id: u32) -> ExternResult<Option<FusedEstimate>> {
    let anchor = anchor_for_object_fused_estimates(norad_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ObjectFusedEstimates)?,
        GetStrategy::Network,
    )?;

    // Find the most recent fused estimate by iterating links
    let mut best: Option<FusedEstimate> = None;
    let mut best_epoch = 0_i64;

    for link in links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(est) = record
            .entry()
            .to_app_option::<FusedEstimate>()
            .ok()
            .flatten()
        {
            if est.epoch.micros > best_epoch {
                best_epoch = est.epoch.micros;
                best = Some(est);
            }
        }
    }

    Ok(best)
}

// =============================================================================
// Trust level lookup
// =============================================================================

/// Look up the trust level for an agent via cross-role call to the identity cluster.
///
/// Queries the identity cluster's `trust_credentials` zome for the agent's trust
/// level. Falls back to `TrustLevel::Unverified` if:
/// - The identity cluster is unreachable (standalone deployment)
/// - The agent has no trust credential on file
/// - Any deserialization or call error occurs
///
/// This is intentionally fail-open: observations are sensor data weighted by trust,
/// not gated by it. Chi-square consistency checks in the fusion pipeline protect
/// against bad data regardless of trust level.
fn lookup_agent_trust_level(agent: &AgentPubKey) -> TrustLevel {
    match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from("trust_credentials"),
        FunctionName::from("get_agent_trust_level"),
        None,
        agent.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result
            .decode::<TrustLevel>()
            .unwrap_or(TrustLevel::Unverified),
        _ => {
            // Identity cluster unreachable or agent has no trust credential
            // Default to Unverified — chi-square gating still protects against bad data
            TrustLevel::Unverified
        }
    }
}

// =============================================================================
// Paginated query operations
// =============================================================================

/// Paginated input for observation queries by NORAD ID
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedObjectObsQuery {
    pub norad_id: u32,
    #[serde(default)]
    pub pagination: PaginationParams,
}

/// Get observations for a NORAD ID with pagination
#[hdk_extern]
pub fn get_observations_paginated(
    input: PaginatedObjectObsQuery,
) -> ExternResult<PaginatedResponse<Observation>> {
    let anchor = anchor_for_object_observations(input.norad_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ObjectObservations)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<Observation>(links, &input.pagination)
}

/// Paginated input for sensor observation queries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedSensorObsQuery {
    pub sensor_id: String,
    #[serde(default)]
    pub pagination: PaginationParams,
}

/// Get observations from a sensor with pagination
#[hdk_extern]
pub fn get_sensor_observations_paginated(
    input: PaginatedSensorObsQuery,
) -> ExternResult<PaginatedResponse<Observation>> {
    let anchor = anchor_for_sensor_observations(&input.sensor_id)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SensorObservations)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<Observation>(links, &input.pagination)
}

/// List sensors with pagination
#[hdk_extern]
pub fn list_sensors_paginated(
    pagination: PaginationParams,
) -> ExternResult<PaginatedResponse<Sensor>> {
    let anchor = anchor_for_all_sensors()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllSensors)?,
        GetStrategy::Network,
    )?;
    resolve_links_paginated::<Sensor>(links, &pagination)
}

/// Resolve links into a paginated response, only fetching entries in the requested page.
fn resolve_links_paginated<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    links: Vec<Link>,
    params: &PaginationParams,
) -> ExternResult<PaginatedResponse<T>> {
    let total = links.len() as u32;
    let offset = params.effective_offset();
    let limit = params.effective_limit();

    let page_links = links
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();

    let mut items = Vec::with_capacity(page_links.len());
    for link in page_links {
        let Some(target) = link.target.into_action_hash() else {
            continue;
        };
        let Some(record) = get(target, GetOptions::default())? else {
            continue;
        };
        if let Some(item) = record.entry().to_app_option::<T>().ok().flatten() {
            items.push(item);
        }
    }

    let effective_offset = offset as u32;
    let effective_limit = limit as u32;
    Ok(PaginatedResponse {
        has_more: effective_offset + effective_limit < total,
        items,
        total,
        offset: effective_offset,
        limit: effective_limit,
    })
}
