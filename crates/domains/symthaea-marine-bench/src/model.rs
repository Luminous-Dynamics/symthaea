// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    SyntheticIllustration,
    EngineeringAssumption,
    Measured,
    ValidatedSolverOutput,
    PublishedReference,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionUse {
    NotForDecision,
    ResearchScreeningOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceValue {
    Known {
        value: f64,
        unit: String,
        provenance_id: String,
    },
    Unknown {
        reason: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceRecord {
    pub id: String,
    pub class: EvidenceClass,
    pub description: String,
    pub source_uri: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SystemBoundary {
    pub includes: Vec<String>,
    pub excludes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RouteEnvelope {
    pub maximum_vessel_length_m: EvidenceValue,
    pub maximum_vessel_beam_m: EvidenceValue,
    pub maximum_vessel_draft_m: EvidenceValue,
    pub available_water_depth_m: EvidenceValue,
    pub minimum_under_keel_clearance_m: EvidenceValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VesselGeometry {
    pub length_overall_m: EvidenceValue,
    pub beam_m: EvidenceValue,
    pub design_draft_m: EvidenceValue,
    pub cargo_capacity_tonnes: EvidenceValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DutyCycle {
    pub loaded_leg_propulsion_kwh: EvidenceValue,
    pub loaded_leg_auxiliary_kwh: EvidenceValue,
    pub return_leg_propulsion_kwh: EvidenceValue,
    pub return_leg_auxiliary_kwh: EvidenceValue,
    pub required_payload_tonnes: EvidenceValue,
    pub round_trip_duration_h: EvidenceValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BatterySystem {
    pub architecture_id: String,
    pub nominal_battery_kwh: EvidenceValue,
    pub temperature_capacity_factor: EvidenceValue,
    pub end_of_life_capacity_factor: EvidenceValue,
    pub reserve_fraction: EvidenceValue,
    pub charger_power_kw: EvidenceValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TerminalInterface {
    pub berth_length_m: EvidenceValue,
    pub berth_beam_limit_m: EvidenceValue,
    pub shore_connection_power_kw: EvidenceValue,
    pub charging_efficiency: EvidenceValue,
    pub available_charge_window_h: EvidenceValue,
    pub available_service_window_h: EvidenceValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MarineFreightCase {
    pub schema_version: String,
    pub case_id: String,
    pub evidence_class: EvidenceClass,
    pub decision_use: DecisionUse,
    pub system_boundary: SystemBoundary,
    pub provenance: Vec<ProvenanceRecord>,
    pub route: RouteEnvelope,
    pub vessel: VesselGeometry,
    pub duty_cycle: DutyCycle,
    pub energy_system: BatterySystem,
    pub terminal: TerminalInterface,
    pub claim_ceiling: String,
}
