//! Observations Coordinator Zome
//!
//! Functions for submitting and querying sensor observations.

use hdk::prelude::*;
use observations_integrity::*;
use mycelix_space_shared::{SpaceTimestamp, QualityScore, GroundLocation};

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

/// Submit an observation
#[hdk_extern]
pub fn submit_observation(input: SubmitObservationInput) -> ExternResult<ActionHash> {
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

/// Register a sensor
#[hdk_extern]
pub fn register_sensor(input: RegisterSensorInput) -> ExternResult<ActionHash> {
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

/// Get all observations for a specific NORAD ID
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
        if let Some(obs) = record
            .entry()
            .to_app_option::<Observation>()
            .ok()
            .flatten()
        {
            observations.push(obs);
        }
    }

    Ok(observations)
}

/// Get all observations from a specific sensor
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
        if let Some(obs) = record
            .entry()
            .to_app_option::<Observation>()
            .ok()
            .flatten()
        {
            observations.push(obs);
        }
    }

    Ok(observations)
}

/// List all registered sensors
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
        if let Some(sensor) = record
            .entry()
            .to_app_option::<Sensor>()
            .ok()
            .flatten()
        {
            sensors.push(sensor);
        }
    }

    Ok(sensors)
}
