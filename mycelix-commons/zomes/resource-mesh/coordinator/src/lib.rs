use hdk::prelude::*;
use resource_mesh_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_voting,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Record a sensor reading (Observer+, low bar for IoT).
#[hdk_extern]
pub fn record_sensor_reading(reading: SensorReading) -> ExternResult<Record> {
    // Low consciousness bar for IoT sensors
    let action_hash = create_entry(&EntryTypes::SensorReading(reading.clone()))?;

    // Link from all readings
    let all_anchor = ensure_anchor("all_readings")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllReadings, ())?;

    // Link from resource type
    let type_anchor = ensure_anchor(&format!("resource/{}", reading.resource_type))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::ResourceTypeReadings, ())?;

    // Link from location
    if !reading.location.is_empty() {
        let loc_anchor = ensure_anchor(&format!("location/{}", reading.location))?;
        create_link(loc_anchor, action_hash.clone(), LinkTypes::LocationReadings, ())?;
    }

    // Link from sensor
    let sensor_anchor = ensure_anchor(&format!("sensor/{}", reading.sensor_id))?;
    create_link(sensor_anchor, action_hash.clone(), LinkTypes::SensorToReadings, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Create a resource alert (Participant+).
#[hdk_extern]
pub fn create_resource_alert(alert: ResourceAlert) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_resource_alert")?;

    let action_hash = create_entry(&EntryTypes::ResourceAlert(alert))?;

    let alerts_anchor = ensure_anchor("all_alerts")?;
    create_link(alerts_anchor, action_hash.clone(), LinkTypes::AllAlerts, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResourceStatusInput {
    pub resource_type: String,
    pub location: Option<String>,
    pub limit: usize,
}

/// Get resource status (recent readings).
#[hdk_extern]
pub fn get_resource_status(input: ResourceStatusInput) -> ExternResult<Vec<SensorReading>> {
    let base = if let Some(ref loc) = input.location {
        anchor_hash(&format!("location/{}", loc))?
    } else {
        anchor_hash(&format!("resource/{}", input.resource_type))?
    };

    let link_type = if input.location.is_some() {
        LinkTypes::LocationReadings
    } else {
        LinkTypes::ResourceTypeReadings
    };

    let links = get_links(
        LinkQuery::try_new(base, link_type)?,
        GetStrategy::default(),
    )?;

    let mut readings = Vec::new();
    for link in links.into_iter().rev().take(input.limit.min(100)) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(reading) = record.entry().to_app_option::<SensorReading>().ok().flatten() {
                    readings.push(reading);
                }
            }
        }
    }
    Ok(readings)
}

/// Match emergency to available resources (cross-domain).
#[hdk_extern]
pub fn match_emergency_to_resources(emergency_description: String) -> ExternResult<Vec<SensorReading>> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "match_emergency")?;

    // Aggregate recent readings across all resource types
    let mut all_resources = Vec::new();
    for resource_type in &["water", "power", "temperature"] {
        let readings = get_resource_status(ResourceStatusInput {
            resource_type: resource_type.to_string(),
            location: None,
            limit: 10,
        })?;
        all_resources.extend(readings);
    }
    Ok(all_resources)
}

/// Publish a demand forecast (Citizen+).
#[hdk_extern]
pub fn publish_forecast(forecast: DemandForecast) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "publish_forecast")?;

    let action_hash = create_entry(&EntryTypes::DemandForecast(forecast))?;

    let forecasts_anchor = ensure_anchor("all_forecasts")?;
    create_link(forecasts_anchor, action_hash.clone(), LinkTypes::AllForecasts, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}
