// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-bounded architecture candidates for the cross-repository Energy System Design Studio.
//!
//! This module deliberately produces a small deterministic *candidate set*, not a
//! "best design". It establishes a typed and independently checkable hand-off to
//! Symtropy's candidate contract. Ranking, detailed physics, dispatch optimization,
//! evidence authentication and field qualification remain separate responsibilities.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub const CANDIDATE_SCHEMA_VERSION: &str = "energy-system-candidate-v0.1";
pub const GENERATOR_VERSION: &str = "0.1.0";

/// Confidence/provenance class. This is metadata, never an implicit performance multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    AnalystAssumption,
    PublishedModel,
    VendorTarget,
    Measured,
    FieldPilot,
    OperatingPlant,
    ReplicatedFleet,
    IndependentlyAudited,
}

impl EvidenceClass {
    fn requires_source(self) -> bool {
        self != Self::AnalystAssumption
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyCarrier {
    Electricity,
    ThermalHeat,
    ChemicalFuel,
    MechanicalWork,
    RadiantEnergy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDirection {
    Input,
    Output,
    Bidirectional,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Source,
    Converter,
    Storage,
    Distribution,
    Controller,
    ServiceInterface,
    Sink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceKind {
    Electricity,
    ThermalHeat,
    Water,
    Communications,
    Mobility,
    IndustrialProcess,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyKind {
    NoneDeclared,
    Interval,
    StandardDeviation,
    Distribution,
    Qualitative,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintSeverity {
    Hard,
    Soft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveId {
    EssentialUnservedEnergy,
    FlexibleUnservedEnergy,
    ServiceReliability,
    LifecycleCost,
    OperatingCost,
    Emissions,
    WaterUse,
    LandUse,
    MaterialUse,
    Curtailment,
    MaintenanceBurden,
    RecoveryTime,
    CandidateComputeCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestedEvaluation {
    SchemaValidation,
    ConservationAudit,
    ExactDispatchReference,
    ThermalExergy,
    ChronologicalReliability,
    SymtropyTrajectory,
    FaultReplay,
    Sensitivity,
    ExternalReferenceComparison,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorIdentity {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemBoundary {
    pub site_id: String,
    pub description: String,
    pub included_boundaries: Vec<String>,
    pub excluded_boundaries: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignHorizon {
    pub start_unix_seconds: i64,
    pub end_unix_seconds: i64,
    pub timezone: String,
    pub interval_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterUncertainty {
    pub kind: UncertaintyKind,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterClaim {
    pub parameter_id: String,
    pub value: Value,
    pub unit: String,
    pub evidence_class: EvidenceClass,
    pub source_ref: Option<String>,
    pub as_of: Option<String>,
    pub applicability: String,
    pub uncertainty: ParameterUncertainty,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyPort {
    pub port_id: String,
    pub direction: PortDirection,
    pub carrier: EnergyCarrier,
    pub unit: String,
    pub temperature_grade_c: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyComponent {
    pub component_id: String,
    pub kind: ComponentKind,
    pub name: String,
    pub ports: Vec<EnergyPort>,
    pub parameter_claims: Vec<ParameterClaim>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyConnection {
    pub connection_id: String,
    pub from_component: String,
    pub from_port: String,
    pub to_component: String,
    pub to_port: String,
    pub carrier: EnergyCarrier,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceContract {
    pub service_id: String,
    pub service_kind: ServiceKind,
    pub component_id: String,
    pub essential: bool,
    pub unit: String,
    pub minimum_delivery_temperature_c: Option<f64>,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAvailability {
    pub component_id: String,
    pub available_kwh: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceRequest {
    pub service_id: String,
    pub requested_quantity: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioInterval {
    pub interval_id: String,
    pub start_unix_seconds: i64,
    pub duration_seconds: u64,
    pub grid_outage: bool,
    pub resource_available_kwh: Vec<ResourceAvailability>,
    pub service_requested: Vec<ServiceRequest>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyScenario {
    pub scenario_id: String,
    pub evidence_class: EvidenceClass,
    pub source_ref: Option<String>,
    pub description: String,
    pub intervals: Vec<ScenarioInterval>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignConstraint {
    pub constraint_id: String,
    pub description: String,
    pub severity: ConstraintSeverity,
    pub verifier_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignObjective {
    pub objective_id: ObjectiveId,
    pub direction: ObjectiveDirection,
    pub unit: String,
    pub description: String,
}

/// Candidate document matching Symtropy's energy-system-candidate-v0.1 contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergySystemCandidate {
    pub schema_version: String,
    pub candidate_id: String,
    pub generator: GeneratorIdentity,
    pub synthetic: bool,
    pub evidence_class: EvidenceClass,
    pub evidence_ref: Option<String>,
    pub system_boundary: SystemBoundary,
    pub horizon: DesignHorizon,
    pub components: Vec<EnergyComponent>,
    pub connections: Vec<EnergyConnection>,
    pub services: Vec<ServiceContract>,
    pub scenarios: Vec<EnergyScenario>,
    pub hard_constraints: Vec<DesignConstraint>,
    pub objectives: Vec<DesignObjective>,
    pub requested_evaluations: Vec<RequestedEvaluation>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateValidationError {
    pub violations: Vec<String>,
}

impl std::fmt::Display for CandidateValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "candidate validation failed: {}", self.violations.join("; "))
    }
}

impl std::error::Error for CandidateValidationError {}

fn valid_nonempty(text: &str) -> bool {
    !text.trim().is_empty()
}

fn supported_unit(unit: &str) -> bool {
    matches!(
        unit,
        "kW"
            | "kWh"
            | "MW"
            | "MWh"
            | "kW_th"
            | "kWh_th"
            | "kW_e"
            | "kWh_e"
            | "degC"
            | "K"
            | "kg"
            | "kg_CO2e"
            | "currency"
            | "L"
            | "m3"
            | "vehicle_km"
            | "unitless"
    )
}

/// Independently validate IDs, evidence envelopes, typed connections and scenario chronology.
///
/// This is a structural contract validator, not the physical energy-balance verifier.
/// The latter must run on evaluated schedules/trajectories and remains a separate component.
pub fn validate_candidate(
    candidate: &EnergySystemCandidate,
) -> Result<(), CandidateValidationError> {
    let mut violations = Vec::new();

    if candidate.schema_version != CANDIDATE_SCHEMA_VERSION {
        violations.push(format!(
            "unsupported schema version {}",
            candidate.schema_version
        ));
    }
    if !valid_nonempty(&candidate.candidate_id) {
        violations.push("candidate ID is empty".to_owned());
    }
    if !valid_nonempty(&candidate.generator.name) || !valid_nonempty(&candidate.generator.version) {
        violations.push("generator identity is incomplete".to_owned());
    }
    if candidate.evidence_class.requires_source()
        && candidate
            .evidence_ref
            .as_deref()
            .is_none_or(|source| !valid_nonempty(source))
    {
        violations.push("non-assumption candidate evidence requires a source reference".to_owned());
    }
    if !valid_nonempty(&candidate.system_boundary.site_id)
        || !valid_nonempty(&candidate.system_boundary.description)
        || candidate.system_boundary.included_boundaries.is_empty()
        || candidate
            .system_boundary
            .included_boundaries
            .iter()
            .any(|boundary| !valid_nonempty(boundary))
        || candidate
            .system_boundary
            .excluded_boundaries
            .iter()
            .any(|boundary| !valid_nonempty(boundary))
    {
        violations.push("system boundary is incomplete".to_owned());
    }
    if candidate.components.is_empty() {
        violations.push("candidate has no components".to_owned());
    }
    if candidate.services.is_empty() {
        violations.push("candidate has no declared services".to_owned());
    }
    if candidate.horizon.end_unix_seconds <= candidate.horizon.start_unix_seconds
        || candidate.horizon.interval_seconds == 0
        || !valid_nonempty(&candidate.horizon.timezone)
    {
        violations.push("horizon bounds, timezone or interval are invalid".to_owned());
    }

    let mut components = BTreeMap::<&str, &EnergyComponent>::new();
    for component in &candidate.components {
        if !valid_nonempty(&component.component_id) || !valid_nonempty(&component.name) {
            violations.push("component ID/name is empty".to_owned());
        }
        if components.insert(&component.component_id, component).is_some() {
            violations.push(format!("duplicate component ID {}", component.component_id));
        }
        if component.ports.is_empty() {
            violations.push(format!("component {} has no ports", component.component_id));
        }
        let mut port_ids = BTreeSet::new();
        for port in &component.ports {
            if !valid_nonempty(&port.port_id) || !port_ids.insert(port.port_id.as_str()) {
                violations.push(format!(
                    "component {} has empty or duplicate port ID {}",
                    component.component_id, port.port_id
                ));
            }
            if !supported_unit(&port.unit) {
                violations.push(format!(
                    "port {}.{} has unsupported unit {}",
                    component.component_id, port.port_id, port.unit
                ));
            }
            match (port.carrier, port.temperature_grade_c) {
                (EnergyCarrier::ThermalHeat, Some(temp))
                    if temp.is_finite() && temp > -273.15 => {}
                (EnergyCarrier::ThermalHeat, _) => violations.push(format!(
                    "thermal port {}.{} needs a finite temperature grade above absolute zero",
                    component.component_id, port.port_id
                )),
                (_, None) => {}
                (_, Some(_)) => violations.push(format!(
                    "non-thermal port {}.{} must not declare heat temperature",
                    component.component_id, port.port_id
                )),
            }
        }
        let mut parameter_ids = BTreeSet::new();
        for claim in &component.parameter_claims {
            if !valid_nonempty(&claim.parameter_id)
                || !parameter_ids.insert(claim.parameter_id.as_str())
            {
                violations.push(format!(
                    "component {} has empty or duplicate parameter ID {}",
                    component.component_id, claim.parameter_id
                ));
            }
            if !supported_unit(&claim.unit) {
                violations.push(format!(
                    "parameter {} has unsupported unit {}",
                    claim.parameter_id, claim.unit
                ));
            }
            if !valid_nonempty(&claim.applicability) {
                violations.push(format!("parameter {} lacks applicability", claim.parameter_id));
            }
            if claim.evidence_class.requires_source()
                && claim
                    .source_ref
                    .as_deref()
                    .is_none_or(|source| !valid_nonempty(source))
            {
                violations.push(format!(
                    "parameter {} is not assumption-classed but has no source",
                    claim.parameter_id
                ));
            }
            if let (Some(lower), Some(upper)) =
                (claim.uncertainty.lower, claim.uncertainty.upper)
            {
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    violations.push(format!(
                        "parameter {} has invalid uncertainty bounds",
                        claim.parameter_id
                    ));
                }
            }
        }
    }

    let mut connection_ids = BTreeSet::new();
    for connection in &candidate.connections {
        if !valid_nonempty(&connection.connection_id)
            || !connection_ids.insert(connection.connection_id.as_str())
        {
            violations.push(format!(
                "empty or duplicate connection ID {}",
                connection.connection_id
            ));
        }
        let source = components
            .get(connection.from_component.as_str())
            .and_then(|component| {
                component
                    .ports
                    .iter()
                    .find(|port| port.port_id == connection.from_port)
                    .map(|port| (*component, port))
            });
        let target = components
            .get(connection.to_component.as_str())
            .and_then(|component| {
                component
                    .ports
                    .iter()
                    .find(|port| port.port_id == connection.to_port)
                    .map(|port| (*component, port))
            });

        match (source, target) {
            (Some((_, source_port)), Some((_, target_port))) => {
                if !matches!(
                    source_port.direction,
                    PortDirection::Output | PortDirection::Bidirectional
                ) {
                    violations.push(format!(
                        "connection {} starts at a non-output port",
                        connection.connection_id
                    ));
                }
                if !matches!(
                    target_port.direction,
                    PortDirection::Input | PortDirection::Bidirectional
                ) {
                    violations.push(format!(
                        "connection {} ends at a non-input port",
                        connection.connection_id
                    ));
                }
                if source_port.carrier != target_port.carrier
                    || source_port.carrier != connection.carrier
                {
                    violations.push(format!(
                        "connection {} has incompatible energy carriers",
                        connection.connection_id
                    ));
                }
                if source_port.unit != target_port.unit {
                    violations.push(format!(
                        "connection {} joins unlike units without an explicit conversion stage",
                        connection.connection_id
                    ));
                }
                if source_port.temperature_grade_c != target_port.temperature_grade_c {
                    violations.push(format!(
                        "connection {} crosses incompatible heat temperature grades",
                        connection.connection_id
                    ));
                }
            }
            _ => violations.push(format!(
                "connection {} references an unknown component or port",
                connection.connection_id
            )),
        }
    }

    let mut services = BTreeMap::<&str, &ServiceContract>::new();
    for service in &candidate.services {
        if !valid_nonempty(&service.service_id)
            || services.insert(&service.service_id, service).is_some()
        {
            violations.push(format!("empty or duplicate service ID {}", service.service_id));
        }
        if !valid_nonempty(&service.description) {
            violations.push(format!("service {} has an empty description", service.service_id));
        }
        if !components.contains_key(service.component_id.as_str()) {
            violations.push(format!(
                "service {} references unknown component {}",
                service.service_id, service.component_id
            ));
        }
        if !supported_unit(&service.unit) {
            violations.push(format!(
                "service {} has unsupported unit {}",
                service.service_id, service.unit
            ));
        }
        match (service.service_kind, service.minimum_delivery_temperature_c) {
            (ServiceKind::ThermalHeat, Some(temp)) if temp.is_finite() && temp > -273.15 => {}
            (ServiceKind::ThermalHeat, _) => violations.push(format!(
                "thermal service {} needs a minimum delivery temperature",
                service.service_id
            )),
            (_, None) => {}
            (_, Some(_)) => violations.push(format!(
                "non-thermal service {} must not declare a heat temperature",
                service.service_id
            )),
        }
    }

    if candidate.scenarios.is_empty() {
        violations.push("candidate has no scenarios".to_owned());
    }
    let mut scenario_ids = BTreeSet::new();
    for scenario in &candidate.scenarios {
        if !valid_nonempty(&scenario.scenario_id)
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            violations.push(format!("empty or duplicate scenario ID {}", scenario.scenario_id));
        }
        if scenario.evidence_class.requires_source()
            && scenario
                .source_ref
                .as_deref()
                .is_none_or(|source| !valid_nonempty(source))
        {
            violations.push(format!(
                "scenario {} is not assumption-classed but has no source",
                scenario.scenario_id
            ));
        }
        if !valid_nonempty(&scenario.description) {
            violations.push(format!("scenario {} has an empty description", scenario.scenario_id));
        }
        if scenario.intervals.is_empty() {
            violations.push(format!("scenario {} has no intervals", scenario.scenario_id));
            continue;
        }
        let mut expected_start = candidate.horizon.start_unix_seconds;
        let mut interval_ids = BTreeSet::new();
        for interval in &scenario.intervals {
            if !valid_nonempty(&interval.interval_id)
                || !interval_ids.insert(interval.interval_id.as_str())
            {
                violations.push(format!(
                    "scenario {} has empty or duplicate interval ID {}",
                    scenario.scenario_id, interval.interval_id
                ));
            }
            if interval.start_unix_seconds != expected_start {
                violations.push(format!(
                    "scenario {} has chronology gap/overlap at interval {}",
                    scenario.scenario_id, interval.interval_id
                ));
            }
            if interval.duration_seconds == 0 {
                violations.push(format!(
                    "scenario {} interval {} has zero duration",
                    scenario.scenario_id, interval.interval_id
                ));
            }
            expected_start = match i64::try_from(interval.duration_seconds)
                .ok()
                .and_then(|duration| interval.start_unix_seconds.checked_add(duration))
            {
                Some(next) => next,
                None => {
                    violations.push(format!(
                        "scenario {} interval {} overflows chronology",
                        scenario.scenario_id, interval.interval_id
                    ));
                    interval.start_unix_seconds
                }
            };

            let mut resources = BTreeSet::new();
            for availability in &interval.resource_available_kwh {
                if !resources.insert(availability.component_id.as_str()) {
                    violations.push(format!(
                        "scenario {} interval {} duplicates resource {}",
                        scenario.scenario_id, interval.interval_id, availability.component_id
                    ));
                }
                if !components
                    .get(availability.component_id.as_str())
                    .is_some_and(|component| component.kind == ComponentKind::Source)
                {
                    violations.push(format!(
                        "scenario {} interval {} resource {} is not a declared source",
                        scenario.scenario_id, interval.interval_id, availability.component_id
                    ));
                }
                if !availability.available_kwh.is_finite() || availability.available_kwh < 0.0 {
                    violations.push(format!(
                        "scenario {} interval {} has invalid resource availability",
                        scenario.scenario_id, interval.interval_id
                    ));
                }
            }

            let mut requested = BTreeSet::new();
            for demand in &interval.service_requested {
                if !requested.insert(demand.service_id.as_str()) {
                    violations.push(format!(
                        "scenario {} interval {} duplicates demand {}",
                        scenario.scenario_id, interval.interval_id, demand.service_id
                    ));
                }
                if !services.contains_key(demand.service_id.as_str()) {
                    violations.push(format!(
                        "scenario {} interval {} references unknown service {}",
                        scenario.scenario_id, interval.interval_id, demand.service_id
                    ));
                }
                if !demand.requested_quantity.is_finite() || demand.requested_quantity < 0.0 {
                    violations.push(format!(
                        "scenario {} interval {} has invalid requested service",
                        scenario.scenario_id, interval.interval_id
                    ));
                }
            }
            let required: BTreeSet<_> = services.keys().copied().collect();
            if requested != required {
                violations.push(format!(
                    "scenario {} interval {} must include exactly one request for every service",
                    scenario.scenario_id, interval.interval_id
                ));
            }
        }
        if expected_start != candidate.horizon.end_unix_seconds {
            violations.push(format!(
                "scenario {} does not cover the declared horizon",
                scenario.scenario_id
            ));
        }
    }

    let mut constraint_ids = BTreeSet::new();
    for constraint in &candidate.hard_constraints {
        if !valid_nonempty(&constraint.constraint_id)
            || !constraint_ids.insert(constraint.constraint_id.as_str())
        {
            violations.push(format!(
                "empty or duplicate constraint ID {}",
                constraint.constraint_id
            ));
        }
        if !valid_nonempty(&constraint.description) || !valid_nonempty(&constraint.verifier_id) {
            violations.push(format!(
                "constraint {} lacks description or verifier",
                constraint.constraint_id
            ));
        }
    }
    let mut objective_ids = BTreeSet::new();
    for objective in &candidate.objectives {
        if !objective_ids.insert(objective.objective_id) {
            violations.push(format!("duplicate objective {:?}", objective.objective_id));
        }
        if !supported_unit(&objective.unit) || !valid_nonempty(&objective.description) {
            violations.push(format!("objective {:?} has invalid unit/description", objective.objective_id));
        }
    }
    if candidate.objectives.is_empty() || candidate.requested_evaluations.is_empty() {
        violations.push("candidate lacks objectives or requested evaluations".to_owned());
    }
    if !valid_nonempty(&candidate.claim_ceiling) {
        violations.push("candidate claim ceiling is empty".to_owned());
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(CandidateValidationError { violations })
    }
}

fn assumption_claim(
    parameter_id: &str,
    value: Value,
    unit: &str,
    applicability: &str,
) -> ParameterClaim {
    ParameterClaim {
        parameter_id: parameter_id.to_owned(),
        value,
        unit: unit.to_owned(),
        evidence_class: EvidenceClass::AnalystAssumption,
        source_ref: None,
        as_of: None,
        applicability: applicability.to_owned(),
        uncertainty: ParameterUncertainty {
            kind: UncertaintyKind::NoneDeclared,
            lower: None,
            upper: None,
            notes: "Synthetic reference-template value; not a real component specification.".to_owned(),
        },
    }
}

fn electric_port(port_id: &str, direction: PortDirection) -> EnergyPort {
    EnergyPort {
        port_id: port_id.to_owned(),
        direction,
        carrier: EnergyCarrier::Electricity,
        unit: "kWh".to_owned(),
        temperature_grade_c: None,
    }
}

fn source_component(
    id: &str,
    name: &str,
    output_port: &str,
    claims: Vec<ParameterClaim>,
) -> EnergyComponent {
    EnergyComponent {
        component_id: id.to_owned(),
        kind: ComponentKind::Source,
        name: name.to_owned(),
        ports: vec![electric_port(output_port, PortDirection::Output)],
        parameter_claims: claims,
    }
}

fn load_component() -> EnergyComponent {
    EnergyComponent {
        component_id: "essential-load".to_owned(),
        kind: ComponentKind::Sink,
        name: "Essential electrical service".to_owned(),
        ports: vec![electric_port("supply", PortDirection::Input)],
        parameter_claims: Vec::new(),
    }
}

fn battery_component() -> EnergyComponent {
    let applicability = "Synthetic reference template only; requires calibration for a real asset.";
    EnergyComponent {
        component_id: "battery".to_owned(),
        kind: ComponentKind::Storage,
        name: "Idealized battery storage".to_owned(),
        ports: vec![
            electric_port("charge", PortDirection::Input),
            electric_port("discharge", PortDirection::Output),
        ],
        parameter_claims: vec![
            assumption_claim("capacity_kwh", json!(2.0), "kWh", applicability),
            assumption_claim("initial_soc_kwh", json!(0.0), "kWh", applicability),
            assumption_claim("terminal_reserve_kwh", json!(0.0), "kWh", applicability),
            assumption_claim("max_charge_power_kw", json!(2.0), "kW", applicability),
            assumption_claim("max_discharge_power_kw", json!(2.0), "kW", applicability),
            assumption_claim("charge_efficiency", json!(1.0), "unitless", applicability),
            assumption_claim("discharge_efficiency", json!(1.0), "unitless", applicability),
        ],
    }
}

fn grid_component() -> EnergyComponent {
    source_component(
        "grid",
        "Synthetic grid import",
        "import",
        vec![
            assumption_claim(
                "interval_import_energy_cap_kwh",
                json!(2.0),
                "kWh",
                "Synthetic interval energy cap, not an interconnection rating.",
            ),
            assumption_claim(
                "grid_import_power_cap_kw",
                json!(2.0),
                "kW",
                "Synthetic power cap; network and protection constraints are excluded.",
            ),
        ],
    )
}

fn pv_component() -> EnergyComponent {
    source_component(
        "pv",
        "Synthetic photovoltaic source",
        "electricity",
        vec![
            assumption_claim(
                "interval_available_energy_cap_kwh",
                json!(2.0),
                "kWh",
                "Synthetic first-interval resource value; not a weather or PV forecast.",
            ),
            assumption_claim(
                "rated_power_kw",
                json!(2.0),
                "kW",
                "Synthetic nameplate value, not validated PV performance.",
            ),
        ],
    )
}

fn service_contract() -> ServiceContract {
    ServiceContract {
        service_id: "essential-electricity".to_owned(),
        service_kind: ServiceKind::Electricity,
        component_id: "essential-load".to_owned(),
        essential: true,
        unit: "kWh".to_owned(),
        minimum_delivery_temperature_c: None,
        description: "Essential electrical service; demand is specified per scenario interval."
            .to_owned(),
    }
}

fn scenario(components: &[EnergyComponent], start: i64) -> EnergyScenario {
    let intervals = [
        ("pv-surplus", start, 3600_u64, false, 0.0_f64),
        ("grid-outage", start + 3600, 3600_u64, true, 2.0_f64),
    ]
    .into_iter()
    .map(
        |(interval_id, start_unix_seconds, duration_seconds, grid_outage, requested)| {
            let resource_available_kwh = components
                .iter()
                .filter(|component| component.kind == ComponentKind::Source)
                .map(|component| ResourceAvailability {
                    component_id: component.component_id.clone(),
                    available_kwh: match component.component_id.as_str() {
                        "pv" if interval_id == "pv-surplus" => 2.0,
                        "grid" if !grid_outage => 2.0,
                        _ => 0.0,
                    },
                })
                .collect();

            ScenarioInterval {
                interval_id: interval_id.to_owned(),
                start_unix_seconds,
                duration_seconds,
                grid_outage,
                resource_available_kwh,
                service_requested: vec![ServiceRequest {
                    service_id: "essential-electricity".to_owned(),
                    requested_quantity: requested,
                }],
            }
        },
    )
    .collect();

    EnergyScenario {
        scenario_id: "synthetic-pv-surplus-then-grid-outage".to_owned(),
        evidence_class: EvidenceClass::AnalystAssumption,
        source_ref: None,
        description: "Synthetic two-hour case: PV resource precedes a grid outage; the declared service remains essential.".to_owned(),
        intervals,
    }
}

fn base_candidate(candidate_id: &str, components: Vec<EnergyComponent>, start: i64) -> EnergySystemCandidate {
    let candidate_component_ids: BTreeSet<_> = components.iter().map(|c| c.component_id.as_str()).collect();
    let mut constraints = vec![
        DesignConstraint {
            constraint_id: "energy-balance".to_owned(),
            description: "Each evaluated interval must close carrier-specific energy balance.".to_owned(),
            severity: ConstraintSeverity::Hard,
            verifier_id: "independent-energy-balance".to_owned(),
        },
        DesignConstraint {
            constraint_id: "grid-outage-import".to_owned(),
            description: "Grid import is prohibited during declared grid outages.".to_owned(),
            severity: ConstraintSeverity::Hard,
            verifier_id: "grid-outage-import-guard".to_owned(),
        },
        DesignConstraint {
            constraint_id: "service-reconciliation".to_owned(),
            description: "Delivered plus unserved service must equal requested service per interval.".to_owned(),
            severity: ConstraintSeverity::Hard,
            verifier_id: "service-demand-reconciliation".to_owned(),
        },
    ];
    if candidate_component_ids.contains("battery") {
        constraints.push(DesignConstraint {
            constraint_id: "battery-state-bounds".to_owned(),
            description: "Battery state of charge remains between zero and the declared capacity.".to_owned(),
            severity: ConstraintSeverity::Hard,
            verifier_id: "storage-state-transition".to_owned(),
        });
    }

    EnergySystemCandidate {
        schema_version: CANDIDATE_SCHEMA_VERSION.to_owned(),
        candidate_id: candidate_id.to_owned(),
        generator: GeneratorIdentity {
            name: "symthaea-energy-architecture-reference".to_owned(),
            version: GENERATOR_VERSION.to_owned(),
        },
        synthetic: true,
        evidence_class: EvidenceClass::AnalystAssumption,
        evidence_ref: None,
        system_boundary: SystemBoundary {
            site_id: "synthetic-site-001".to_owned(),
            description: "Synthetic single-site essential electric service hub.".to_owned(),
            included_boundaries: vec![
                "PV electrical source where present".to_owned(),
                "grid import connection where present".to_owned(),
                "battery storage where present".to_owned(),
                "essential electrical service".to_owned(),
            ],
            excluded_boundaries: vec![
                "AC power flow and protection".to_owned(),
                "inverter switching physics".to_owned(),
                "forecast uncertainty".to_owned(),
                "field metering and hardware behavior".to_owned(),
                "capital costs and lifecycle replacement".to_owned(),
            ],
        },
        horizon: DesignHorizon {
            start_unix_seconds: start,
            end_unix_seconds: start + 7200,
            timezone: "UTC".to_owned(),
            interval_seconds: 3600,
        },
        scenarios: vec![scenario(&components, start)],
        components,
        connections: Vec::new(),
        services: vec![service_contract()],
        hard_constraints: constraints,
        objectives: vec![
            DesignObjective {
                objective_id: ObjectiveId::EssentialUnservedEnergy,
                direction: ObjectiveDirection::Minimize,
                unit: "kWh".to_owned(),
                description: "Minimize unmet essential electrical service.".to_owned(),
            },
            DesignObjective {
                objective_id: ObjectiveId::OperatingCost,
                direction: ObjectiveDirection::Minimize,
                unit: "currency".to_owned(),
                description: "Not scoreable until sourced tariff and operating-cost data are bound; missing cost is unknown, not zero.".to_owned(),
            },
            DesignObjective {
                objective_id: ObjectiveId::Emissions,
                direction: ObjectiveDirection::Minimize,
                unit: "kg_CO2e".to_owned(),
                description: "Not scoreable until source-specific emissions factors are bound; missing factors are unknown, not zero.".to_owned(),
            },
            DesignObjective {
                objective_id: ObjectiveId::Curtailment,
                direction: ObjectiveDirection::Minimize,
                unit: "kWh".to_owned(),
                description: "Minimize unused renewable resource after satisfying harder service constraints.".to_owned(),
            },
        ],
        requested_evaluations: vec![
            RequestedEvaluation::SchemaValidation,
            RequestedEvaluation::ConservationAudit,
            RequestedEvaluation::ExactDispatchReference,
            RequestedEvaluation::ChronologicalReliability,
            RequestedEvaluation::SymtropyTrajectory,
        ],
        claim_ceiling: "Synthetic template candidate only; not an optimized, physically validated, field-measured or safety-qualified design.".to_owned(),
    }
}

/// Produce three deterministic, structurally different reference architectures.
///
/// The grid-only and PV+grid options are deliberately retained even though the
/// frozen outage scenario exposes their lack of storage. Candidate generation
/// must preserve weak/infeasible alternatives so comparison is not selection-biased.
pub fn reference_service_hub_candidates() -> Vec<EnergySystemCandidate> {
    let start = 1_800_000_000_i64;

    let mut grid_only = base_candidate(
        "synthetic-grid-only",
        vec![grid_component(), load_component()],
        start,
    );
    grid_only.connections = vec![EnergyConnection {
        connection_id: "grid-to-essential-load".to_owned(),
        from_component: "grid".to_owned(),
        from_port: "import".to_owned(),
        to_component: "essential-load".to_owned(),
        to_port: "supply".to_owned(),
        carrier: EnergyCarrier::Electricity,
    }];

    let mut pv_grid = base_candidate(
        "synthetic-pv-grid",
        vec![pv_component(), grid_component(), load_component()],
        start,
    );
    pv_grid.connections = vec![
        EnergyConnection {
            connection_id: "pv-to-essential-load".to_owned(),
            from_component: "pv".to_owned(),
            from_port: "electricity".to_owned(),
            to_component: "essential-load".to_owned(),
            to_port: "supply".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
        EnergyConnection {
            connection_id: "grid-to-essential-load".to_owned(),
            from_component: "grid".to_owned(),
            from_port: "import".to_owned(),
            to_component: "essential-load".to_owned(),
            to_port: "supply".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
    ];

    let mut pv_battery_grid = base_candidate(
        "synthetic-pv-battery-grid",
        vec![pv_component(), grid_component(), battery_component(), load_component()],
        start,
    );
    pv_battery_grid.connections = vec![
        EnergyConnection {
            connection_id: "pv-to-essential-load".to_owned(),
            from_component: "pv".to_owned(),
            from_port: "electricity".to_owned(),
            to_component: "essential-load".to_owned(),
            to_port: "supply".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
        EnergyConnection {
            connection_id: "pv-to-battery-charge".to_owned(),
            from_component: "pv".to_owned(),
            from_port: "electricity".to_owned(),
            to_component: "battery".to_owned(),
            to_port: "charge".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
        EnergyConnection {
            connection_id: "battery-to-essential-load".to_owned(),
            from_component: "battery".to_owned(),
            from_port: "discharge".to_owned(),
            to_component: "essential-load".to_owned(),
            to_port: "supply".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
        EnergyConnection {
            connection_id: "grid-to-battery-charge".to_owned(),
            from_component: "grid".to_owned(),
            from_port: "import".to_owned(),
            to_component: "battery".to_owned(),
            to_port: "charge".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
        EnergyConnection {
            connection_id: "grid-to-essential-load".to_owned(),
            from_component: "grid".to_owned(),
            from_port: "import".to_owned(),
            to_component: "essential-load".to_owned(),
            to_port: "supply".to_owned(),
            carrier: EnergyCarrier::Electricity,
        },
    ];

    vec![grid_only, pv_grid, pv_battery_grid]
}


/// A deterministic rejection record. Invalid candidates are retained as data,
/// rather than silently removed from the search statistics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRejection {
    pub candidate_id: String,
    pub violations: Vec<String>,
}

/// Output of the first explicit bounded design-space sweep. Each candidate is
/// still only a synthetic hypothesis; this manifest is not an optimizer result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignSpaceSearchResult {
    pub search_schema_version: String,
    pub generator_version: String,
    pub requested_candidate_count: usize,
    pub axis_values: BTreeMap<String, Vec<f64>>,
    pub candidates: Vec<EnergySystemCandidate>,
    pub rejected_candidates: Vec<CandidateRejection>,
    pub claim_ceiling: String,
}

const PARAMETRIC_PV_POWER_KW: [f64; 3] = [1.0, 2.0, 3.0];
const PARAMETRIC_BATTERY_CAPACITY_KWH: [f64; 3] = [1.0, 2.0, 3.0];
const PARAMETRIC_BATTERY_POWER_KW: [f64; 2] = [1.0, 2.0];
const PARAMETRIC_GRID_POWER_KW: [f64; 2] = [1.0, 2.0];

fn update_numeric_claim(
    candidate: &mut EnergySystemCandidate,
    component_id: &str,
    parameter_id: &str,
    value: f64,
) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "parameter {component_id}.{parameter_id} must be finite and non-negative"
        ));
    }
    let component = candidate
        .components
        .iter_mut()
        .find(|component| component.component_id == component_id)
        .ok_or_else(|| format!("missing component {component_id}"))?;
    let claim = component
        .parameter_claims
        .iter_mut()
        .find(|claim| claim.parameter_id == parameter_id)
        .ok_or_else(|| format!("missing parameter claim {component_id}.{parameter_id}"))?;

    claim.value = json!(value);
    claim.applicability =
        "Bounded synthetic design-space value; not a vendor specification or field measurement."
            .to_owned();
    claim.uncertainty.notes = format!(
        "Synthetic sweep input: {} {}. No real-site uncertainty is implied.",
        value, claim.unit
    );
    Ok(())
}

/// Enumerate a small, declared Cartesian design space over PV power, battery
/// energy capacity, battery charge/discharge power, and grid import power.
///
/// This is deliberately a transparent baseline search, not a learned or
/// open-ended topology invention algorithm. The same source-to-load topology
/// is retained so improvements can first be attributed to sizing variables.
pub fn bounded_parametric_design_space() -> DesignSpaceSearchResult {
    let base = reference_service_hub_candidates()
        .into_iter()
        .find(|candidate| candidate.candidate_id == "synthetic-pv-battery-grid")
        .expect("the frozen reference set includes the PV+battery+grid topology");

    let requested_candidate_count = PARAMETRIC_PV_POWER_KW.len()
        * PARAMETRIC_BATTERY_CAPACITY_KWH.len()
        * PARAMETRIC_BATTERY_POWER_KW.len()
        * PARAMETRIC_GRID_POWER_KW.len();
    let mut axis_values = BTreeMap::new();
    axis_values.insert(
        "pv_rated_power_kw".to_owned(),
        PARAMETRIC_PV_POWER_KW.to_vec(),
    );
    axis_values.insert(
        "battery_capacity_kwh".to_owned(),
        PARAMETRIC_BATTERY_CAPACITY_KWH.to_vec(),
    );
    axis_values.insert(
        "battery_charge_discharge_power_kw".to_owned(),
        PARAMETRIC_BATTERY_POWER_KW.to_vec(),
    );
    axis_values.insert(
        "grid_import_power_kw".to_owned(),
        PARAMETRIC_GRID_POWER_KW.to_vec(),
    );

    let mut candidates = Vec::with_capacity(requested_candidate_count);
    let mut rejected_candidates = Vec::new();

    for pv_power_kw in PARAMETRIC_PV_POWER_KW {
        for battery_capacity_kwh in PARAMETRIC_BATTERY_CAPACITY_KWH {
            for battery_power_kw in PARAMETRIC_BATTERY_POWER_KW {
                for grid_power_kw in PARAMETRIC_GRID_POWER_KW {
                    let candidate_id = format!(
                        concat!(
                            "synthetic-parametric-pv-{:.0}kw-",
                            "battery-{:.0}kwh-battery-power-{:.0}kw-",
                            "grid-{:.0}kw"
                        ),
                        pv_power_kw,
                        battery_capacity_kwh,
                        battery_power_kw,
                        grid_power_kw
                    );
                    let mut candidate = base.clone();
                    candidate.candidate_id = candidate_id.clone();
                    candidate.generator = GeneratorIdentity {
                        name: "symthaea-energy-architecture-bounded-grid-search".to_owned(),
                        version: GENERATOR_VERSION.to_owned(),
                    };

                    let parameter_updates = [
                        ("pv", "rated_power_kw", pv_power_kw),
                        ("pv", "interval_available_energy_cap_kwh", pv_power_kw),
                        ("grid", "grid_import_power_cap_kw", grid_power_kw),
                        ("grid", "interval_import_energy_cap_kwh", grid_power_kw),
                        ("battery", "capacity_kwh", battery_capacity_kwh),
                        ("battery", "max_charge_power_kw", battery_power_kw),
                        ("battery", "max_discharge_power_kw", battery_power_kw),
                    ];
                    let mut mutation_error = None;
                    for (component_id, parameter_id, value) in parameter_updates {
                        if let Err(error) = update_numeric_claim(
                            &mut candidate,
                            component_id,
                            parameter_id,
                            value,
                        ) {
                            mutation_error = Some(error);
                            break;
                        }
                    }

                    if mutation_error.is_none() {
                        for scenario in &mut candidate.scenarios {
                            for interval in &mut scenario.intervals {
                                let interval_hours =
                                    interval.duration_seconds as f64 / 3600.0;
                                for resource in &mut interval.resource_available_kwh {
                                    resource.available_kwh = match resource.component_id.as_str() {
                                        "pv" if interval.interval_id == "pv-surplus" => {
                                            pv_power_kw * interval_hours
                                        }
                                        "grid" if !interval.grid_outage => {
                                            grid_power_kw * interval_hours
                                        }
                                        _ => 0.0,
                                    };
                                }
                            }
                        }
                    }

                    if let Some(error) = mutation_error {
                        rejected_candidates.push(CandidateRejection {
                            candidate_id,
                            violations: vec![error],
                        });
                        continue;
                    }

                    match validate_candidate(&candidate) {
                        Ok(()) => candidates.push(candidate),
                        Err(error) => rejected_candidates.push(CandidateRejection {
                            candidate_id,
                            violations: error.violations,
                        }),
                    }
                }
            }
        }
    }

    DesignSpaceSearchResult {
        search_schema_version: "energy-architecture-search-v0.1".to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        requested_candidate_count,
        axis_values,
        candidates,
        rejected_candidates,
        claim_ceiling: concat!(
            "Deterministic synthetic sizing sweep only; no real-site optimization, ",
            "dynamic-simulation, lifecycle-cost, physical-safety or field-performance claim."
        )
        .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_candidates_are_valid_deterministic_and_structurally_distinct() {
        let first = reference_service_hub_candidates();
        let second = reference_service_hub_candidates();
        assert_eq!(first.len(), 3);
        assert_eq!(first, second, "template enumeration must be deterministic");

        let ids: Vec<_> = first.iter().map(|candidate| candidate.candidate_id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "synthetic-grid-only",
                "synthetic-pv-grid",
                "synthetic-pv-battery-grid"
            ]
        );
        let component_counts: Vec<_> = first
            .iter()
            .map(|candidate| candidate.components.len())
            .collect();
        assert_eq!(component_counts, vec![2, 3, 4]);

        for candidate in &first {
            validate_candidate(candidate).expect("all template structures satisfy the contract");
            let bytes = serde_json::to_vec(candidate).expect("candidate serializes");
            let round_trip: EnergySystemCandidate =
                serde_json::from_slice(&bytes).expect("candidate deserializes");
            assert_eq!(*candidate, round_trip);
        }
    }

    #[test]
    fn serialized_candidates_use_the_symtropy_contract_field_names() {
        let candidates = reference_service_hub_candidates();
        let candidate = &candidates[2];
        let value = serde_json::to_value(candidate).expect("candidate serializes");
        assert_eq!(value["schema_version"], CANDIDATE_SCHEMA_VERSION);
        assert_eq!(value["synthetic"], true);
        assert_eq!(value["evidence_class"], "analyst_assumption");
        assert!(value["system_boundary"].is_object());
        assert!(value["components"].is_array());
        assert!(value["connections"].is_array());
        assert!(value["scenarios"].is_array());
        assert_eq!(value["services"][0]["service_kind"], "electricity");
        assert_eq!(value["components"][0]["ports"][0]["carrier"], "electricity");
    }

    #[test]
    fn candidates_without_components_or_services_are_rejected() {
        let mut candidates = reference_service_hub_candidates();
        candidates[0].components.clear();
        candidates[0].services.clear();
        let error = validate_candidate(&candidates[0]).unwrap_err();
        assert!(error.violations.iter().any(|violation| violation.contains("no components")));
        assert!(
            error
                .violations
                .iter()
                .any(|violation| violation.contains("no declared services"))
        );
    }

    #[test]
    fn carrier_mismatch_is_rejected_before_any_ranking() {
        let mut candidates = reference_service_hub_candidates();
        candidates[1].connections[0].carrier = EnergyCarrier::MechanicalWork;
        let error = validate_candidate(&candidates[1]).unwrap_err();
        assert!(
            error
                .violations
                .iter()
                .any(|violation| violation.contains("incompatible energy carriers"))
        );
    }

    #[test]
    fn chronology_gap_is_rejected() {
        let mut candidates = reference_service_hub_candidates();
        candidates[0].scenarios[0].intervals[1].start_unix_seconds += 1;
        let error = validate_candidate(&candidates[0]).unwrap_err();
        assert!(
            error
                .violations
                .iter()
                .any(|violation| violation.contains("chronology gap/overlap"))
        );
    }

    #[test]
    fn evidence_class_cannot_be_promoted_without_a_source() {
        let mut candidates = reference_service_hub_candidates();
        candidates[0].components[0].parameter_claims[0].evidence_class =
            EvidenceClass::PublishedModel;
        candidates[0].components[0].parameter_claims[0].source_ref = None;
        let error = validate_candidate(&candidates[0]).unwrap_err();
        assert!(
            error
                .violations
                .iter()
                .any(|violation| violation.contains("has no source"))
        );
    }

    #[test]
    fn weak_architectures_are_retained_for_unbiased_comparison() {
        let candidates = reference_service_hub_candidates();
        assert!(candidates.iter().any(|candidate| candidate.candidate_id == "synthetic-grid-only"));
        assert!(candidates.iter().any(|candidate| candidate.candidate_id == "synthetic-pv-grid"));
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.candidate_id == "synthetic-pv-battery-grid")
        );
    }
 
    #[test]
    fn bounded_parametric_design_space_is_deterministic_and_fully_validated() {
        let first = bounded_parametric_design_space();
        let second = bounded_parametric_design_space();
        assert_eq!(
            first, second,
            "sweep ordering and candidate serialization inputs must be stable"
        );
        assert_eq!(first.search_schema_version, "energy-architecture-search-v0.1");
        assert_eq!(first.requested_candidate_count, 36);
        assert_eq!(first.candidates.len() + first.rejected_candidates.len(), 36);
        assert_eq!(first.candidates.len(), 36);
        assert!(first.rejected_candidates.is_empty());

        let ids: BTreeSet<_> = first
            .candidates
            .iter()
            .map(|candidate| candidate.candidate_id.as_str())
            .collect();
        assert_eq!(ids.len(), 36, "candidate IDs must be unique");
        for candidate in &first.candidates {
            validate_candidate(candidate)
                .expect("every emitted search candidate must pass validation");
        }
    }

    #[test]
    fn parametric_sweep_covers_each_declared_axis_endpoint() {
        let result = bounded_parametric_design_space();
        assert_eq!(result.axis_values["pv_rated_power_kw"], vec![1.0, 2.0, 3.0]);
        assert_eq!(
            result.axis_values["battery_capacity_kwh"],
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            result.axis_values["battery_charge_discharge_power_kw"],
            vec![1.0, 2.0]
        );
        assert_eq!(result.axis_values["grid_import_power_kw"], vec![1.0, 2.0]);

        let candidate = result
            .candidates
            .iter()
            .find(|candidate| {
                candidate.candidate_id
                    == "synthetic-parametric-pv-1kw-battery-1kwh-battery-power-1kw-grid-1kw"
            })
            .expect("the minimum-axis design is emitted");
        assert_eq!(candidate.components.len(), 4);
        assert_eq!(candidate.horizon.interval_seconds, 3600);
        assert!(candidate.synthetic);
        assert!(candidate.claim_ceiling.contains("synthetic sizing sweep"));
        assert_eq!(
            candidate.scenarios[0].intervals[0]
                .resource_available_kwh
                .iter()
                .find(|resource| resource.component_id == "pv")
                .unwrap()
                .available_kwh,
            1.0
        );
        assert_eq!(
            candidate.scenarios[0].intervals[0]
                .resource_available_kwh
                .iter()
                .find(|resource| resource.component_id == "grid")
                .unwrap()
                .available_kwh,
            1.0
        );
        assert_eq!(
            candidate.scenarios[0].intervals[1]
                .resource_available_kwh
                .iter()
                .find(|resource| resource.component_id == "grid")
                .unwrap()
                .available_kwh,
            0.0,
            "outage resources remain zero after sizing mutations"
        );
    }

    #[test]
    fn parametric_variants_keep_missing_economic_evidence_unknown() {
        let result = bounded_parametric_design_space();
        let candidate = &result.candidates[0];
        let cost = candidate
            .objectives
            .iter()
            .find(|objective| objective.objective_id == ObjectiveId::OperatingCost)
            .expect("operating-cost objective is declared");
        assert!(cost.description.contains("unknown, not zero"));
        let emissions = candidate
            .objectives
            .iter()
            .find(|objective| objective.objective_id == ObjectiveId::Emissions)
            .expect("emissions objective is declared");
        assert!(emissions.description.contains("unknown, not zero"));
        assert!(result.claim_ceiling.contains("no real-site optimization"));
    }

}
