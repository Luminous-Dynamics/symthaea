//! Observations Coordinator Zome
//!
//! Functions for submitting and querying sensor observations.

use hdk::prelude::*;
use observations_integrity::*;
use mycelix_space_shared::{SpaceTimestamp, QualityScore, GroundLocation};

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
        sensor_id: input.sensor_id,
        submitted_by: agent,
        submitted_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::Observation(observation))
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
        sensor_id: input.sensor_id,
        name: input.name,
        sensor_type: input.sensor_type,
        location: input.location,
        operator: agent,
        capabilities: input.capabilities,
        registered_at: SpaceTimestamp::now(),
    };

    create_entry(&EntryTypes::Sensor(sensor))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterSensorInput {
    pub sensor_id: String,
    pub name: String,
    pub sensor_type: ObservationType,
    pub location: Option<GroundLocation>,
    pub capabilities: SensorCapabilities,
}
