// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// A sensor reading from IoT infrastructure.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct SensorReading {
    /// Sensor identifier.
    pub sensor_id: String,
    /// Resource type: "water", "power", "temperature", "humidity", "air_quality".
    pub resource_type: String,
    /// Numeric value.
    pub value: f64,
    /// Unit of measurement.
    pub unit: String,
    /// Location descriptor.
    pub location: String,
    /// Reading timestamp (µs since epoch).
    pub timestamp_us: u64,
}

/// An alert generated from abnormal sensor readings.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ResourceAlert {
    /// Sensor that triggered the alert.
    pub sensor_id: String,
    /// Resource type.
    pub resource_type: String,
    /// Severity: "info", "warning", "critical", "emergency".
    pub severity: String,
    /// Description of the alert.
    pub description: String,
    /// Value that triggered the alert.
    pub trigger_value: f64,
    /// Threshold exceeded.
    pub threshold: f64,
    /// Alert timestamp (µs since epoch).
    pub timestamp_us: u64,
}

/// A demand forecast for resource planning.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct DemandForecast {
    /// Resource type.
    pub resource_type: String,
    /// Forecast horizon in hours.
    pub horizon_hours: u32,
    /// Predicted consumption rate (units/hour).
    pub predicted_rate: f64,
    /// Predicted total consumption.
    pub predicted_total: f64,
    /// Confidence (0.0-1.0).
    pub confidence: f64,
    /// Generation timestamp (µs since epoch).
    pub generated_at: u64,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "SensorReading", visibility = "public")]
    SensorReading(SensorReading),
    #[entry_type(name = "ResourceAlert", visibility = "public")]
    ResourceAlert(ResourceAlert),
    #[entry_type(name = "DemandForecast", visibility = "public")]
    DemandForecast(DemandForecast),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllReadings,
    ResourceTypeReadings,
    LocationReadings,
    AllAlerts,
    AllForecasts,
    SensorToReadings,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::SensorReading(reading) => {
                        if reading.sensor_id.is_empty() || reading.sensor_id.len() > 256 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Sensor ID must be 1-256 chars".to_string(),
                            ));
                        }
                        if !reading.value.is_finite() {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Value must be finite".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::ResourceAlert(alert) => {
                        if !["info", "warning", "critical", "emergency"].contains(&alert.severity.as_str()) {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Invalid severity level".to_string(),
                            ));
                        }
                        if alert.description.len() > 1024 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Description too long (max 1024)".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::DemandForecast(forecast) => {
                        if forecast.confidence < 0.0 || forecast.confidence > 1.0 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Confidence must be in [0.0, 1.0]".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}
