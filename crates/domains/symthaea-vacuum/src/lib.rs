// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bounded vacuum and gas-network topology semantics.
//!
//! ENG-VAC-001A models topology, identity, state/evidence planes, gas/process identity,
//! and flow-regime/model profiles without inventing a local physical-quantity system.
//! Numeric pressure/flow/temperature/etc. compose with the canonical SE quantity layer later.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_engineering_catalog::ComponentSubjectId;
use thiserror::Error;

pub const VACUUM_NETWORK_SCHEMA_ID: &str = "symthaea-vacuum-gas-network-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubjectKind {
    VacuumVolume,
    Pump,
    Valve,
    GasSource,
    Regulator,
    FlowController,
    PressureSensor,
    GasCompositionSensor,
    TrapFilterScrubber,
    Feedthrough,
    Seal,
    VentPurge,
    LeakIngress,
    OutgassingSource,
    PermeationSource,
    OtherComponent,
}

impl SubjectKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::VacuumVolume => "vacuum_volume",
            Self::Pump => "pump",
            Self::Valve => "valve",
            Self::GasSource => "gas_source",
            Self::Regulator => "regulator",
            Self::FlowController => "flow_controller",
            Self::PressureSensor => "pressure_sensor",
            Self::GasCompositionSensor => "gas_composition_sensor",
            Self::TrapFilterScrubber => "trap_filter_scrubber",
            Self::Feedthrough => "feedthrough",
            Self::Seal => "seal",
            Self::VentPurge => "vent_purge",
            Self::LeakIngress => "leak_ingress",
            Self::OutgassingSource => "outgassing_source",
            Self::PermeationSource => "permeation_source",
            Self::OtherComponent => "other_component",
        }
    }

    fn component_required(self) -> bool {
        matches!(
            self,
            Self::Pump
                | Self::Valve
                | Self::GasSource
                | Self::Regulator
                | Self::FlowController
                | Self::PressureSensor
                | Self::GasCompositionSensor
                | Self::TrapFilterScrubber
                | Self::Feedthrough
                | Self::Seal
                | Self::OtherComponent
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkSubject {
    pub id: String,
    pub kind: SubjectKind,
    pub component_subject_id: Option<ComponentSubjectId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortKind {
    VacuumGasPath,
    ExhaustBacking,
    ProcessInjection,
    VentPurge,
    MeasurementTap,
    FeedthroughBoundary,
    ContainmentBoundary,
}

impl PortKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::VacuumGasPath => "vacuum_gas_path",
            Self::ExhaustBacking => "exhaust_backing",
            Self::ProcessInjection => "process_injection",
            Self::VentPurge => "vent_purge",
            Self::MeasurementTap => "measurement_tap",
            Self::FeedthroughBoundary => "feedthrough_boundary",
            Self::ContainmentBoundary => "containment_boundary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDirection {
    Inlet,
    Outlet,
    Bidirectional,
    ObservationOnly,
}

impl PortDirection {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Inlet => "inlet",
            Self::Outlet => "outlet",
            Self::Bidirectional => "bidirectional",
            Self::ObservationOnly => "observation_only",
        }
    }

    fn may_source_flow(self) -> bool {
        matches!(self, Self::Outlet | Self::Bidirectional)
    }

    fn may_sink_flow(self) -> bool {
        matches!(self, Self::Inlet | Self::Bidirectional)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPort {
    pub id: String,
    pub subject_id: String,
    pub kind: PortKind,
    pub direction: PortDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionKind {
    FlowPath,
    MeasurementTap,
    ContainmentInterface,
}

impl ConnectionKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::FlowPath => "flow_path",
            Self::MeasurementTap => "measurement_tap",
            Self::ContainmentInterface => "containment_interface",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkConnection {
    pub id: String,
    pub kind: ConnectionKind,
    pub from_port: String,
    pub to_port: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidencePlane {
    DeclaredRated,
    ConfiguredCommanded,
    ObservedMeasured,
    InferredDerived,
    Predicted,
}

impl EvidencePlane {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::DeclaredRated => "declared_rated",
            Self::ConfiguredCommanded => "configured_commanded",
            Self::ObservedMeasured => "observed_measured",
            Self::InferredDerived => "inferred_derived",
            Self::Predicted => "predicted",
        }
    }

    pub fn can_substitute_for(self, required: Self) -> bool {
        self == required
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatePropertyKind {
    Pressure,
    DifferentialPressure,
    Flow,
    GasComposition,
    Temperature,
    ValvePosition,
    PumpOperatingState,
    LeakOutgassingState,
    ProcessState,
}

impl StatePropertyKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Pressure => "pressure",
            Self::DifferentialPressure => "differential_pressure",
            Self::Flow => "flow",
            Self::GasComposition => "gas_composition",
            Self::Temperature => "temperature",
            Self::ValvePosition => "valve_position",
            Self::PumpOperatingState => "pump_operating_state",
            Self::LeakOutgassingState => "leak_outgassing_state",
            Self::ProcessState => "process_state",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateEvidenceBinding {
    pub subject_id: String,
    pub property: StatePropertyKind,
    pub plane: EvidencePlane,
    pub evidence_ref: String,
    pub model_profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GasProcessIdentity {
    pub id: String,
    pub species_ids: Vec<String>,
    pub composition_evidence_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlowRegime {
    ContinuumViscous,
    Transitional,
    Molecular,
    ChokedCompressible,
    RarefiedKineticRequired,
    UnknownNotAdmitted,
}

impl FlowRegime {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::ContinuumViscous => "continuum_viscous",
            Self::Transitional => "transitional",
            Self::Molecular => "molecular",
            Self::ChokedCompressible => "choked_compressible",
            Self::RarefiedKineticRequired => "rarefied_kinetic_required",
            Self::UnknownNotAdmitted => "unknown_not_admitted",
        }
    }

    pub fn admitted_for_prediction(self) -> bool {
        !matches!(self, Self::RarefiedKineticRequired | Self::UnknownNotAdmitted)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowModelProfile {
    pub id: String,
    pub regime: FlowRegime,
    pub applicability_ref: String,
    pub source_evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VacuumGasNetworkV1 {
    pub schema_id: String,
    pub id: String,
    pub subjects: Vec<NetworkSubject>,
    pub ports: Vec<NetworkPort>,
    pub connections: Vec<NetworkConnection>,
    #[serde(default)]
    pub gas_processes: Vec<GasProcessIdentity>,
    #[serde(default)]
    pub model_profiles: Vec<FlowModelProfile>,
    #[serde(default)]
    pub evidence_bindings: Vec<StateEvidenceBinding>,
}

impl VacuumGasNetworkV1 {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            schema_id: VACUUM_NETWORK_SCHEMA_ID.into(),
            id: id.into(),
            subjects: Vec::new(),
            ports: Vec::new(),
            connections: Vec::new(),
            gas_processes: Vec::new(),
            model_profiles: Vec::new(),
            evidence_bindings: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), VacuumError> {
        if self.schema_id != VACUUM_NETWORK_SCHEMA_ID {
            return Err(VacuumError::UnsupportedSchema(self.schema_id.clone()));
        }
        require_nonempty("network id", &self.id)?;

        let mut subjects = BTreeMap::new();
        for subject in &self.subjects {
            require_nonempty("subject id", &subject.id)?;
            if subject.label.as_deref().is_some_and(|label| label.trim().is_empty()) {
                return Err(VacuumError::EmptyLabel(subject.id.clone()));
            }
            if subject.kind.component_required() && subject.component_subject_id.is_none() {
                return Err(VacuumError::ComponentRequired(subject.id.clone()));
            }
            if let Some(component) = &subject.component_subject_id {
                validate_component_subject_id(component)?;
            }
            if subjects.insert(subject.id.as_str(), subject).is_some() {
                return Err(VacuumError::DuplicateSubject(subject.id.clone()));
            }
        }

        let mut ports = BTreeMap::new();
        for port in &self.ports {
            require_nonempty("port id", &port.id)?;
            require_nonempty("port subject id", &port.subject_id)?;
            if !subjects.contains_key(port.subject_id.as_str()) {
                return Err(VacuumError::UnknownPortSubject {
                    port: port.id.clone(),
                    subject: port.subject_id.clone(),
                });
            }
            if ports.insert(port.id.as_str(), port).is_some() {
                return Err(VacuumError::DuplicatePort(port.id.clone()));
            }
        }

        let mut connection_ids = BTreeSet::new();
        for connection in &self.connections {
            require_nonempty("connection id", &connection.id)?;
            if !connection_ids.insert(connection.id.as_str()) {
                return Err(VacuumError::DuplicateConnection(connection.id.clone()));
            }
            if connection.from_port == connection.to_port {
                return Err(VacuumError::SelfConnection(connection.id.clone()));
            }
            let from = ports.get(connection.from_port.as_str()).ok_or_else(|| {
                VacuumError::UnknownConnectionPort {
                    connection: connection.id.clone(),
                    port: connection.from_port.clone(),
                }
            })?;
            let to = ports.get(connection.to_port.as_str()).ok_or_else(|| {
                VacuumError::UnknownConnectionPort {
                    connection: connection.id.clone(),
                    port: connection.to_port.clone(),
                }
            })?;

            match connection.kind {
                ConnectionKind::FlowPath => {
                    if !from.direction.may_source_flow() || !to.direction.may_sink_flow() {
                        return Err(VacuumError::InvalidFlowDirection(connection.id.clone()));
                    }
                    if from.kind == PortKind::MeasurementTap || to.kind == PortKind::MeasurementTap {
                        return Err(VacuumError::MeasurementTapUsedAsFlowPath(
                            connection.id.clone(),
                        ));
                    }
                }
                ConnectionKind::MeasurementTap => {
                    let is_measurement = from.kind == PortKind::MeasurementTap
                        || to.kind == PortKind::MeasurementTap
                        || from.direction == PortDirection::ObservationOnly
                        || to.direction == PortDirection::ObservationOnly;
                    if !is_measurement {
                        return Err(VacuumError::MeasurementConnectionHasNoTap(
                            connection.id.clone(),
                        ));
                    }
                }
                ConnectionKind::ContainmentInterface => {}
            }
        }

        let mut gas_ids = BTreeSet::new();
        for gas in &self.gas_processes {
            require_nonempty("gas process id", &gas.id)?;
            require_nonempty("gas composition evidence ref", &gas.composition_evidence_ref)?;
            if !gas_ids.insert(gas.id.as_str()) {
                return Err(VacuumError::DuplicateGasProcess(gas.id.clone()));
            }
            if gas.species_ids.is_empty() {
                return Err(VacuumError::EmptySpeciesSet(gas.id.clone()));
            }
            let mut species = BTreeSet::new();
            for species_id in &gas.species_ids {
                require_nonempty("species id", species_id)?;
                if !species.insert(species_id.as_str()) {
                    return Err(VacuumError::DuplicateSpecies {
                        gas_process: gas.id.clone(),
                        species: species_id.clone(),
                    });
                }
            }
        }

        let mut models = BTreeMap::new();
        for profile in &self.model_profiles {
            require_nonempty("model profile id", &profile.id)?;
            require_nonempty("model applicability ref", &profile.applicability_ref)?;
            if profile.source_evidence_refs.is_empty()
                || profile
                    .source_evidence_refs
                    .iter()
                    .any(|reference| reference.trim().is_empty())
            {
                return Err(VacuumError::MissingModelSource(profile.id.clone()));
            }
            let mut refs = BTreeSet::new();
            for reference in &profile.source_evidence_refs {
                if !refs.insert(reference.as_str()) {
                    return Err(VacuumError::DuplicateModelSource {
                        profile: profile.id.clone(),
                        reference: reference.clone(),
                    });
                }
            }
            if models.insert(profile.id.as_str(), profile).is_some() {
                return Err(VacuumError::DuplicateModelProfile(profile.id.clone()));
            }
        }

        let mut bindings = BTreeSet::new();
        for binding in &self.evidence_bindings {
            require_nonempty("evidence binding subject id", &binding.subject_id)?;
            require_nonempty("evidence ref", &binding.evidence_ref)?;
            if !subjects.contains_key(binding.subject_id.as_str()) {
                return Err(VacuumError::UnknownEvidenceSubject(binding.subject_id.clone()));
            }

            let model = match binding.model_profile_id.as_deref() {
                Some(id) => Some(
                    *models
                        .get(id)
                        .ok_or_else(|| VacuumError::UnknownModelProfile(id.into()))?,
                ),
                None => None,
            };

            match binding.plane {
                EvidencePlane::Predicted => {
                    let model = model.ok_or(VacuumError::PredictedEvidenceRequiresModelProfile)?;
                    if !model.regime.admitted_for_prediction() {
                        return Err(VacuumError::ModelRegimeNotPredictive(model.id.clone()));
                    }
                }
                EvidencePlane::ObservedMeasured if model.is_some() => {
                    return Err(VacuumError::MeasuredEvidenceCannotUseModelProfile);
                }
                _ => {}
            }

            let key = (
                binding.subject_id.as_str(),
                binding.property,
                binding.plane,
                binding.evidence_ref.as_str(),
                binding.model_profile_id.as_deref(),
            );
            if !bindings.insert(key) {
                return Err(VacuumError::DuplicateEvidenceBinding {
                    subject: binding.subject_id.clone(),
                    reference: binding.evidence_ref.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn topology_digest(&self) -> Result<String, VacuumError> {
        self.validate()?;
        let mut bytes = Vec::new();
        push_field(&mut bytes, "schema", &self.schema_id);
        push_field(&mut bytes, "network_id", &self.id);

        let mut subjects: Vec<_> = self.subjects.iter().collect();
        subjects.sort_by(|a, b| a.id.cmp(&b.id));
        for subject in subjects {
            push_field(&mut bytes, "subject", &subject.id);
            push_field(&mut bytes, "subject_kind", subject.kind.canonical_name());
            if let Some(component) = &subject.component_subject_id {
                push_field(&mut bytes, "component_subject", &component.0);
            }
        }

        let mut ports: Vec<_> = self.ports.iter().collect();
        ports.sort_by(|a, b| a.id.cmp(&b.id));
        for port in ports {
            push_field(&mut bytes, "port", &port.id);
            push_field(&mut bytes, "port_subject", &port.subject_id);
            push_field(&mut bytes, "port_kind", port.kind.canonical_name());
            push_field(&mut bytes, "port_direction", port.direction.canonical_name());
        }

        let mut connections: Vec<_> = self.connections.iter().collect();
        connections.sort_by(|a, b| a.id.cmp(&b.id));
        for connection in connections {
            push_field(&mut bytes, "connection", &connection.id);
            push_field(&mut bytes, "connection_kind", connection.kind.canonical_name());
            push_field(&mut bytes, "from", &connection.from_port);
            push_field(&mut bytes, "to", &connection.to_port);
        }

        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }

    pub fn semantic_snapshot_digest(&self) -> Result<String, VacuumError> {
        let topology = self.topology_digest()?;
        let mut bytes = Vec::new();
        push_field(&mut bytes, "topology", &topology);

        let mut gases: Vec<_> = self.gas_processes.iter().collect();
        gases.sort_by(|a, b| a.id.cmp(&b.id));
        for gas in gases {
            push_field(&mut bytes, "gas", &gas.id);
            let mut species = gas.species_ids.clone();
            species.sort();
            for species_id in species {
                push_field(&mut bytes, "species", &species_id);
            }
            push_field(&mut bytes, "composition_evidence", &gas.composition_evidence_ref);
        }

        let mut profiles: Vec<_> = self.model_profiles.iter().collect();
        profiles.sort_by(|a, b| a.id.cmp(&b.id));
        for profile in profiles {
            push_field(&mut bytes, "model", &profile.id);
            push_field(&mut bytes, "regime", profile.regime.canonical_name());
            push_field(&mut bytes, "applicability", &profile.applicability_ref);
            let mut refs = profile.source_evidence_refs.clone();
            refs.sort();
            for reference in refs {
                push_field(&mut bytes, "model_source", &reference);
            }
        }

        let mut bindings: Vec<_> = self.evidence_bindings.iter().collect();
        bindings.sort_by(|a, b| {
            (
                &a.subject_id,
                a.property,
                a.plane,
                &a.evidence_ref,
                &a.model_profile_id,
            )
                .cmp(&(
                    &b.subject_id,
                    b.property,
                    b.plane,
                    &b.evidence_ref,
                    &b.model_profile_id,
                ))
        });
        for binding in bindings {
            push_field(&mut bytes, "binding_subject", &binding.subject_id);
            push_field(&mut bytes, "property", binding.property.canonical_name());
            push_field(&mut bytes, "plane", binding.plane.canonical_name());
            push_field(&mut bytes, "evidence", &binding.evidence_ref);
            if let Some(model) = &binding.model_profile_id {
                push_field(&mut bytes, "model_profile", model);
            }
        }

        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }
}

fn validate_component_subject_id(id: &ComponentSubjectId) -> Result<(), VacuumError> {
    if id.0.len() != 64 || !id.0.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(VacuumError::InvalidComponentSubjectId(id.0.clone()));
    }
    Ok(())
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), VacuumError> {
    if value.trim().is_empty() {
        Err(VacuumError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn push_field(bytes: &mut Vec<u8>, name: &str, value: &str) {
    bytes.extend_from_slice(&(name.len() as u64).to_be_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VacuumError {
    #[error("unsupported vacuum network schema {0:?}")]
    UnsupportedSchema(String),
    #[error("{0} cannot be empty")]
    EmptyField(&'static str),
    #[error("subject {0:?} has an empty navigation label")]
    EmptyLabel(String),
    #[error("physical subject {0:?} requires canonical ENG-CATALOG component identity")]
    ComponentRequired(String),
    #[error("invalid catalog component-subject id {0:?}")]
    InvalidComponentSubjectId(String),
    #[error("duplicate subject id {0:?}")]
    DuplicateSubject(String),
    #[error("duplicate port id {0:?}")]
    DuplicatePort(String),
    #[error("port {port:?} references unknown subject {subject:?}")]
    UnknownPortSubject { port: String, subject: String },
    #[error("duplicate connection id {0:?}")]
    DuplicateConnection(String),
    #[error("connection {0:?} connects a port to itself")]
    SelfConnection(String),
    #[error("connection {connection:?} references unknown port {port:?}")]
    UnknownConnectionPort { connection: String, port: String },
    #[error("flow connection {0:?} violates declared port direction")]
    InvalidFlowDirection(String),
    #[error("flow connection {0:?} attempts to use a measurement tap as a flow path")]
    MeasurementTapUsedAsFlowPath(String),
    #[error("measurement connection {0:?} has no measurement/observation port")]
    MeasurementConnectionHasNoTap(String),
    #[error("duplicate gas-process id {0:?}")]
    DuplicateGasProcess(String),
    #[error("gas process {0:?} declares no species")]
    EmptySpeciesSet(String),
    #[error("gas process {gas_process:?} repeats species {species:?}")]
    DuplicateSpecies { gas_process: String, species: String },
    #[error("duplicate model-profile id {0:?}")]
    DuplicateModelProfile(String),
    #[error("model profile {0:?} has no valid source evidence")]
    MissingModelSource(String),
    #[error("model profile {profile:?} repeats source evidence {reference:?}")]
    DuplicateModelSource { profile: String, reference: String },
    #[error("state/evidence binding references unknown subject {0:?}")]
    UnknownEvidenceSubject(String),
    #[error("state/evidence binding references unknown model profile {0:?}")]
    UnknownModelProfile(String),
    #[error("predicted vacuum evidence requires an exact local model profile")]
    PredictedEvidenceRequiresModelProfile,
    #[error("flow/model regime is a higher-fidelity requirement and cannot itself back prediction: {0:?}")]
    ModelRegimeNotPredictive(String),
    #[error("observed/measured vacuum evidence must not use a prediction model as measurement authority")]
    MeasuredEvidenceCannotUseModelProfile,
    #[error("duplicate evidence binding for subject {subject:?} and ref {reference:?}")]
    DuplicateEvidenceBinding { subject: String, reference: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(c: char) -> ComponentSubjectId {
        ComponentSubjectId(std::iter::repeat_n(c, 64).collect())
    }

    fn base_network() -> VacuumGasNetworkV1 {
        VacuumGasNetworkV1 {
            schema_id: VACUUM_NETWORK_SCHEMA_ID.into(),
            id: "vac-fixture-001".into(),
            subjects: vec![
                NetworkSubject {
                    id: "chamber".into(),
                    kind: SubjectKind::VacuumVolume,
                    component_subject_id: None,
                    label: Some("Main Chamber".into()),
                },
                NetworkSubject {
                    id: "pump".into(),
                    kind: SubjectKind::Pump,
                    component_subject_id: Some(cid('a')),
                    label: Some("Pump A".into()),
                },
                NetworkSubject {
                    id: "gauge".into(),
                    kind: SubjectKind::PressureSensor,
                    component_subject_id: Some(cid('b')),
                    label: Some("Gauge".into()),
                },
            ],
            ports: vec![
                NetworkPort {
                    id: "chamber-pump-out".into(),
                    subject_id: "chamber".into(),
                    kind: PortKind::VacuumGasPath,
                    direction: PortDirection::Outlet,
                },
                NetworkPort {
                    id: "pump-inlet".into(),
                    subject_id: "pump".into(),
                    kind: PortKind::VacuumGasPath,
                    direction: PortDirection::Inlet,
                },
                NetworkPort {
                    id: "chamber-gauge".into(),
                    subject_id: "chamber".into(),
                    kind: PortKind::MeasurementTap,
                    direction: PortDirection::Bidirectional,
                },
                NetworkPort {
                    id: "gauge-tap".into(),
                    subject_id: "gauge".into(),
                    kind: PortKind::MeasurementTap,
                    direction: PortDirection::ObservationOnly,
                },
            ],
            connections: vec![
                NetworkConnection {
                    id: "pump-line".into(),
                    kind: ConnectionKind::FlowPath,
                    from_port: "chamber-pump-out".into(),
                    to_port: "pump-inlet".into(),
                },
                NetworkConnection {
                    id: "gauge-tap-link".into(),
                    kind: ConnectionKind::MeasurementTap,
                    from_port: "chamber-gauge".into(),
                    to_port: "gauge-tap".into(),
                },
            ],
            gas_processes: vec![GasProcessIdentity {
                id: "argon-process".into(),
                species_ids: vec!["Ar".into()],
                composition_evidence_ref: "evidence:gas:argon-lot-1".into(),
            }],
            model_profiles: vec![FlowModelProfile {
                id: "molecular-v1".into(),
                regime: FlowRegime::Molecular,
                applicability_ref: "profile:vac:molecular-v1".into(),
                source_evidence_refs: vec!["evidence:model:molecular-conductance".into()],
            }],
            evidence_bindings: vec![StateEvidenceBinding {
                subject_id: "gauge".into(),
                property: StatePropertyKind::Pressure,
                plane: EvidencePlane::ObservedMeasured,
                evidence_ref: "field:pressure:gauge:run-1".into(),
                model_profile_id: None,
            }],
        }
    }

    #[test]
    fn valid_network_round_trips_with_stable_identity() {
        let network = base_network();
        network.validate().unwrap();
        let topology = network.topology_digest().unwrap();
        let semantic = network.semantic_snapshot_digest().unwrap();
        let json = serde_json::to_string(&network).unwrap();
        let decoded: VacuumGasNetworkV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(topology, decoded.topology_digest().unwrap());
        assert_eq!(semantic, decoded.semantic_snapshot_digest().unwrap());
    }

    #[test]
    fn pump_path_is_chamber_to_pump_inlet_not_pump_outlet_to_chamber() {
        let network = base_network();
        assert!(network.validate().is_ok());
        let mut wrong = network;
        let pump_inlet = wrong
            .ports
            .iter_mut()
            .find(|port| port.id == "pump-inlet")
            .unwrap();
        pump_inlet.direction = PortDirection::Outlet;
        assert_eq!(
            wrong.validate(),
            Err(VacuumError::InvalidFlowDirection("pump-line".into()))
        );
    }

    #[test]
    fn physical_pump_and_gauge_require_catalog_identity() {
        let mut network = base_network();
        network
            .subjects
            .iter_mut()
            .find(|subject| subject.id == "pump")
            .unwrap()
            .component_subject_id = None;
        assert_eq!(network.validate(), Err(VacuumError::ComponentRequired("pump".into())));
    }

    #[test]
    fn component_substitution_changes_topology_identity() {
        let a = base_network();
        let mut b = a.clone();
        b.subjects
            .iter_mut()
            .find(|subject| subject.id == "pump")
            .unwrap()
            .component_subject_id = Some(cid('c'));
        assert_ne!(a.topology_digest().unwrap(), b.topology_digest().unwrap());
    }

    #[test]
    fn rarefied_kinetic_required_is_not_prediction_authority() {
        let mut network = base_network();
        network.model_profiles.push(FlowModelProfile {
            id: "kinetic-required".into(),
            regime: FlowRegime::RarefiedKineticRequired,
            applicability_ref: "profile:kinetic-required".into(),
            source_evidence_refs: vec!["source:regime-selection".into()],
        });
        network.evidence_bindings.push(StateEvidenceBinding {
            subject_id: "chamber".into(),
            property: StatePropertyKind::Pressure,
            plane: EvidencePlane::Predicted,
            evidence_ref: "prediction:false".into(),
            model_profile_id: Some("kinetic-required".into()),
        });
        assert_eq!(
            network.validate(),
            Err(VacuumError::ModelRegimeNotPredictive("kinetic-required".into()))
        );
    }

    #[test]
    fn predicted_evidence_requires_resolved_predictive_model() {
        let mut network = base_network();
        network.evidence_bindings.push(StateEvidenceBinding {
            subject_id: "chamber".into(),
            property: StatePropertyKind::Pressure,
            plane: EvidencePlane::Predicted,
            evidence_ref: "prediction:molecular".into(),
            model_profile_id: Some("molecular-v1".into()),
        });
        assert!(network.validate().is_ok());

        network.evidence_bindings.last_mut().unwrap().model_profile_id = Some("missing".into());
        assert_eq!(
            network.validate(),
            Err(VacuumError::UnknownModelProfile("missing".into()))
        );
    }

    #[test]
    fn measured_pressure_cannot_use_prediction_model_as_authority() {
        let mut network = base_network();
        network.evidence_bindings[0].model_profile_id = Some("molecular-v1".into());
        assert_eq!(
            network.validate(),
            Err(VacuumError::MeasuredEvidenceCannotUseModelProfile)
        );
    }

    #[test]
    fn configured_commanded_never_substitutes_for_observed_measured() {
        assert!(!EvidencePlane::ConfiguredCommanded
            .can_substitute_for(EvidencePlane::ObservedMeasured));
        assert!(EvidencePlane::ObservedMeasured
            .can_substitute_for(EvidencePlane::ObservedMeasured));
    }

    #[test]
    fn measurement_tap_cannot_masquerade_as_flow_path() {
        let mut network = base_network();
        network.connections[1].kind = ConnectionKind::FlowPath;
        assert_eq!(
            network.validate(),
            Err(VacuumError::MeasurementTapUsedAsFlowPath(
                "gauge-tap-link".into()
            ))
        );
    }

    #[test]
    fn observed_evidence_changes_snapshot_not_topology() {
        let a = base_network();
        let mut b = a.clone();
        b.evidence_bindings[0].evidence_ref = "field:pressure:gauge:run-2".into();
        assert_eq!(a.topology_digest().unwrap(), b.topology_digest().unwrap());
        assert_ne!(
            a.semantic_snapshot_digest().unwrap(),
            b.semantic_snapshot_digest().unwrap()
        );
    }
}
