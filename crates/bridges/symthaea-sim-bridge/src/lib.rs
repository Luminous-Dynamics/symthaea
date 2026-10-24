// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simulator-agnostic engineering bridge traits.
//!
//! Symthaea should reason over topology, uncertainty, safety cases, and
//! cross-domain analogies. External solvers remain the source of numerical
//! truth for FEA, CFD, multibody dynamics, circuits, process simulation, and
//! other high-fidelity physics.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::seed_from_name;
use thiserror::Error;
use symthaea_types::{ModelMaturity, PhysicalType, TypeJudgement};

/// Simple deterministic text embedding for HDC space.
pub fn embed_text(text: &str, dimension: usize) -> ContinuousHV {
    ContinuousHV::random(dimension, seed_from_name(text))
}

/// Broad solver families Symthaea can request without binding to one vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverKind {
    /// Finite-element structural or thermal analysis.
    FiniteElement,
    /// Computational fluid dynamics.
    ComputationalFluidDynamics,
    /// Rigid/flexible-body dynamics and contacts.
    MultibodyDynamics,
    /// Circuit and signal-integrity simulation.
    Circuit,
    /// Chemical/process simulation.
    Process,
    /// CAD geometry, meshing, and parametric model interrogation.
    CadGeometry,
    /// Coupled run that coordinates more than one solver family.
    MultiPhysics,
    /// Custom domain solver registered by an adapter crate.
    Custom,
}

/// Engineering discipline attached to a simulation request or result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineeringDomain {
    /// Structures, geotechnical systems, transport, and civic infrastructure.
    Civil,
    /// Mechanisms, thermal systems, robotics, and machines.
    Mechanical,
    /// Circuits, power systems, RF, and signal integrity.
    Electrical,
    /// Flight, orbital, and tightly coupled aero-thermo-structural systems.
    Aerospace,
    /// Chemical, process, and reaction systems.
    ChemicalProcess,
    /// Embodied agents, robot dynamics, manipulation, locomotion, and sim-to-real.
    Robotics,
    /// Nuclear structure, reactor-adjacent analysis, radiation, safeguards, and nuclear safety.
    Nuclear,
    /// Materials, degradation, manufacturability, metamaterials, and critical minerals.
    Materials,
    /// Environmental systems, sustainability, carbon, water, ecology, and climate resilience.
    Environmental,
    /// Cross-domain safety-critical systems engineering.
    Systems,
}

/// How multiple solver stages exchange state in a multi-physics workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouplingMode {
    /// One solver feeds another once, with no iteration.
    OneWay,
    /// Solvers iterate until exchanged quantities stabilize.
    Iterative,
    /// Strongly coupled co-simulation with synchronized time stepping.
    CoSimulation,
}

/// One stage in a coupled multi-physics workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPhysicalPort {
    /// Stable signal/field name shared across coupled stages.
    pub name: String,
    /// Canonical physical semantics for the signal.
    pub physical_type: PhysicalType,
}

impl TypedPhysicalPort {
    pub fn new(name: impl Into<String>, physical_type: PhysicalType) -> Self {
        Self { name: name.into(), physical_type }
    }
}

/// One stage in a coupled multi-physics workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoupledSimulationStage {
    /// Stage identifier unique within the request.
    pub id: String,
    /// Domain covered by the stage.
    pub domain: EngineeringDomain,
    /// Solver family used by the stage.
    pub solver: SolverKind,
    /// Legacy name-only inputs retained for compatibility.
    pub consumes: Vec<String>,
    /// Legacy name-only outputs retained for compatibility.
    pub produces: Vec<String>,
    /// Typed inputs used for semantic compatibility checking.
    #[serde(default)]
    pub typed_consumes: Vec<TypedPhysicalPort>,
    /// Typed outputs used for semantic compatibility checking.
    #[serde(default)]
    pub typed_produces: Vec<TypedPhysicalPort>,
}

impl CoupledSimulationStage {
    /// Construct a stage with no declared dependencies.
    pub fn new(id: impl Into<String>, domain: EngineeringDomain, solver: SolverKind) -> Self {
        Self {
            id: id.into(),
            domain,
            solver,
            consumes: Vec::new(),
            produces: Vec::new(),
            typed_consumes: Vec::new(),
            typed_produces: Vec::new(),
        }
    }

    /// Declare inputs consumed from previous stages.
    pub fn consumes(mut self, inputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.consumes = inputs.into_iter().map(Into::into).collect();
        self
    }

    /// Declare outputs produced by this stage.
    pub fn produces(mut self, outputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.produces = outputs.into_iter().map(Into::into).collect();
        self
    }

    /// Declare typed physical inputs consumed from previous stages.
    pub fn typed_consumes(mut self, inputs: impl IntoIterator<Item = TypedPhysicalPort>) -> Self {
        self.typed_consumes = inputs.into_iter().collect();
        self
    }

    /// Declare typed physical outputs emitted by this stage.
    pub fn typed_produces(mut self, outputs: impl IntoIterator<Item = TypedPhysicalPort>) -> Self {
        self.typed_produces = outputs.into_iter().collect();
        self
    }
}

/// First-class multi-physics request for orchestrating solver stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiPhysicsRequest {
    /// Stable request identifier.
    pub id: String,
    /// Human-readable purpose.
    pub objective: String,
    /// Coupling strategy.
    pub coupling: CouplingMode,
    /// Ordered solver stages.
    pub stages: Vec<CoupledSimulationStage>,
    /// Explicit typed topology. This is the preferred coupling contract.
    #[serde(default)]
    pub typed_connections: Vec<TypedPhysicalConnection>,
    /// Coupling convergence tolerance for iterative/co-simulation workflows.
    pub coupling_tolerance: f64,
}

/// Explicit typed connection between a producer stage/port and consumer stage/port.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPhysicalConnection {
    pub producer_stage: String,
    pub producer_port: String,
    pub consumer_stage: String,
    pub consumer_port: String,
}

impl TypedPhysicalConnection {
    pub fn new(
        producer_stage: impl Into<String>,
        producer_port: impl Into<String>,
        consumer_stage: impl Into<String>,
        consumer_port: impl Into<String>,
    ) -> Self {
        Self {
            producer_stage: producer_stage.into(),
            producer_port: producer_port.into(),
            consumer_stage: consumer_stage.into(),
            consumer_port: consumer_port.into(),
        }
    }
}

impl MultiPhysicsRequest {
    /// Construct an empty coupled request.
    pub fn new(
        id: impl Into<String>,
        objective: impl Into<String>,
        coupling: CouplingMode,
    ) -> Self {
        Self {
            id: id.into(),
            objective: objective.into(),
            coupling,
            stages: Vec::new(),
            typed_connections: Vec::new(),
            coupling_tolerance: 1e-3,
        }
    }

    /// Append a stage.
    pub fn with_stage(mut self, stage: CoupledSimulationStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Append an explicit typed producer-to-consumer connection.
    pub fn with_typed_connection(mut self, connection: TypedPhysicalConnection) -> Self {
        self.typed_connections.push(connection);
        self
    }


    /// Validate explicit stage/port topology. Unknown semantics never pass
    /// as compatible because an executable coupling requires a known contract.
    pub fn validate_typed_topology(&self) -> Result<(), SimulationError> {
        use std::collections::{HashMap, HashSet};
        let mut stages = HashMap::new();
        let mut destinations = HashSet::new();
        let mut edges = HashSet::new();
        for stage in &self.stages {
            if stage.id.trim().is_empty() {
                return Err(SimulationError::InvalidRequest(
                    "typed topology stage id cannot be empty".into(),
                ));
            }
            if stages.insert(stage.id.as_str(), stage).is_some() {
                return Err(SimulationError::InvalidRequest(format!(
                    "typed topology contains duplicate stage id {:?}",
                    stage.id
                )));
            }

            let mut inputs = HashMap::new();
            for port in &stage.typed_consumes {
                if port.name.trim().is_empty() {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed input name cannot be empty in stage {:?}",
                        stage.id
                    )));
                }
                if inputs.insert(port.name.as_str(), port).is_some() {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed stage {:?} contains duplicate input port {:?}",
                        stage.id, port.name
                    )));
                }
            }

            let mut outputs = HashMap::new();
            for port in &stage.typed_produces {
                if port.name.trim().is_empty() {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed output name cannot be empty in stage {:?}",
                        stage.id
                    )));
                }
                if outputs.insert(port.name.as_str(), port).is_some() {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed stage {:?} contains duplicate output port {:?}",
                        stage.id, port.name
                    )));
                }
            }
        }

        for connection in &self.typed_connections {
            let edge = (
                connection.producer_stage.as_str(),
                connection.producer_port.as_str(),
                connection.consumer_stage.as_str(),
                connection.consumer_port.as_str(),
            );
            if !edges.insert(edge) {
                return Err(SimulationError::InvalidRequest(format!(
                    "typed topology contains duplicate connection {}:{} -> {}:{}",
                    connection.producer_stage,
                    connection.producer_port,
                    connection.consumer_stage,
                    connection.consumer_port
                )));
            }
            let destination = (
                connection.consumer_stage.as_str(),
                connection.consumer_port.as_str(),
            );
            if !destinations.insert(destination) {
                return Err(SimulationError::InvalidRequest(format!(
                    "typed topology has multiple producers for consumer endpoint {}:{}",
                    connection.consumer_stage, connection.consumer_port
                )));
            }

            let producer = stages.get(connection.producer_stage.as_str()).ok_or_else(|| {
                SimulationError::InvalidRequest(format!(
                    "typed connection references missing producer stage {:?}",
                    connection.producer_stage
                ))
            })?;
            let consumer = stages.get(connection.consumer_stage.as_str()).ok_or_else(|| {
                SimulationError::InvalidRequest(format!(
                    "typed connection references missing consumer stage {:?}",
                    connection.consumer_stage
                ))
            })?;

            let output = producer
                .typed_produces
                .iter()
                .find(|port| port.name == connection.producer_port)
                .ok_or_else(|| SimulationError::InvalidRequest(format!(
                    "typed connection references missing producer port {:?}",
                    connection.producer_port
                )))?;
            let input = consumer
                .typed_consumes
                .iter()
                .find(|port| port.name == connection.consumer_port)
                .ok_or_else(|| SimulationError::InvalidRequest(format!(
                    "typed connection references missing consumer port {:?}",
                    connection.consumer_port
                )))?;

            match output
                .physical_type
                .judge_numeric_compatibility(&input.physical_type)
            {
                TypeJudgement::Valid(()) => {}
                TypeJudgement::Invalid(error) => {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed connection {}:{} -> {}:{} is incompatible: {}",
                        connection.producer_stage,
                        connection.producer_port,
                        connection.consumer_stage,
                        connection.consumer_port,
                        error.reason
                    )));
                }
                TypeJudgement::Unknown(reason) => {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed connection {}:{} -> {}:{} is unknown: {}",
                        connection.producer_stage,
                        connection.producer_port,
                        connection.consumer_stage,
                        connection.consumer_port,
                        reason
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate typed producer/consumer edges. Unknown semantics never pass
    /// as compatible because an executable coupling requires a known contract.
    pub fn validate_typed_connections(&self) -> Result<(), SimulationError> {
        use std::collections::HashMap;
        let mut producers = HashMap::<&str, &PhysicalType>::new();
        for stage in &self.stages {
            for port in &stage.typed_produces {
                if port.name.trim().is_empty() {
                    return Err(SimulationError::InvalidRequest("typed output name cannot be empty".into()));
                }
                if producers.contains_key(port.name.as_str()) {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed output {:?} has multiple producers",
                        port.name
                    )));
                }
                producers.insert(port.name.as_str(), &port.physical_type);
            }
        }
        for stage in &self.stages {
            for port in &stage.typed_consumes {
                if port.name.trim().is_empty() {
                    return Err(SimulationError::InvalidRequest("typed input name cannot be empty".into()));
                }
                let Some(source) = producers.get(port.name.as_str()) else {
                    return Err(SimulationError::InvalidRequest(format!(
                        "typed input {:?} has no declared producer",
                        port.name
                    )));
                };
                match source.judge_numeric_compatibility(&port.physical_type) {
                    TypeJudgement::Valid(()) => {}
                    TypeJudgement::Invalid(error) => {
                        return Err(SimulationError::InvalidRequest(format!(
                            "typed coupling {:?} is incompatible: {}",
                            port.name, error.reason
                        )));
                    }
                    TypeJudgement::Unknown(reason) => {
                        return Err(SimulationError::InvalidRequest(format!(
                            "typed coupling {:?} is unknown: {}",
                            port.name, reason
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A scalar or categorical model parameter with unit/provenance metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelParameter {
    /// Stable parameter name, such as `yield_strength` or `beam_length`.
    pub name: String,
    /// Numeric value when the parameter is scalar.
    pub value: f64,
    /// Unit string in the source model's convention.
    pub unit: String,
    /// Canonical physical semantics when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_type: Option<PhysicalType>,
    /// Source or assumption label for auditability.
    pub provenance: String,
    /// Optional quantified uncertainty for the parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintyEstimate>,
}

/// Closed interval for bounded engineering quantities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

impl Interval {
    /// Construct an interval, ordering bounds if needed.
    pub fn new(a: f64, b: f64) -> Self {
        if !a.is_finite() || !b.is_finite() {
            return Self {
                lower: f64::NAN,
                upper: f64::NAN,
            };
        }
        Self {
            lower: a.min(b),
            upper: a.max(b),
        }
    }

    /// Interval width.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Midpoint.
    pub fn midpoint(&self) -> f64 {
        0.5 * (self.lower + self.upper)
    }

    /// Whether a scalar falls inside the interval.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Minimal uncertainty decomposition for engineering evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Model/knowledge uncertainty, 0.0 certain to 1.0 unknown.
    pub epistemic: f64,
    /// Noise/irreducible variability, 0.0 deterministic to 1.0 high noise.
    pub aleatoric: f64,
    /// Optional confidence interval in the metric's unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<Interval>,
}

impl UncertaintyEstimate {
    /// Construct a bounded uncertainty estimate.
    pub fn new(epistemic: f64, aleatoric: f64) -> Self {
        Self {
            epistemic: if epistemic.is_finite() {
                epistemic.clamp(0.0, 1.0)
            } else {
                1.0
            },
            aleatoric: if aleatoric.is_finite() {
                aleatoric.clamp(0.0, 1.0)
            } else {
                1.0
            },
            interval: None,
        }
    }

    /// Attach an interval.
    pub fn with_interval(mut self, interval: Interval) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Conservative scalar summary.
    pub fn total(&self) -> f64 {
        if self.epistemic.is_finite() && self.aleatoric.is_finite() {
            (self.epistemic + self.aleatoric).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

impl Default for UncertaintyEstimate {
    fn default() -> Self {
        Self::new(0.5, 0.5)
    }
}

/// A solver request produced by a reasoning layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationRequest {
    /// Stable request identifier.
    pub id: String,
    /// Domain being analyzed.
    pub domain: EngineeringDomain,
    /// Solver family to dispatch to.
    pub solver: SolverKind,
    /// Human-readable purpose for the run.
    pub objective: String,
    /// Model parameters known to Symthaea before dispatch.
    pub parameters: Vec<ModelParameter>,
    /// Requested output metrics, such as `max_stress_mpa` or `drag_coefficient`.
    pub requested_metrics: Vec<String>,
}

impl SimulationRequest {
    /// Construct a minimal simulation request.
    pub fn new(
        id: impl Into<String>,
        domain: EngineeringDomain,
        solver: SolverKind,
        objective: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            domain,
            solver,
            objective: objective.into(),
            parameters: Vec::new(),
            requested_metrics: Vec::new(),
        }
    }

    /// Add a scalar parameter and preserve chainability for adapters/tests.
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        provenance: impl Into<String>,
    ) -> Self {
        self.parameters.push(ModelParameter {
            name: name.into(),
            value,
            unit: unit.into(),
            physical_type: None,
            provenance: provenance.into(),
            uncertainty: None,
        });
        self
    }

    /// Add a parameter with canonical physical semantics.
    pub fn with_typed_parameter(
        mut self,
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        physical_type: PhysicalType,
        provenance: impl Into<String>,
    ) -> Self {
        self.parameters.push(ModelParameter {
            name: name.into(),
            value,
            unit: unit.into(),
            physical_type: Some(physical_type),
            provenance: provenance.into(),
            uncertainty: None,
        });
        self
    }

    /// Attach uncertainty to the most recently added parameter.
    pub fn with_last_parameter_uncertainty(mut self, uncertainty: UncertaintyEstimate) -> Self {
        if let Some(parameter) = self.parameters.last_mut() {
            parameter.uncertainty = Some(uncertainty);
        }
        self
    }

    /// Validate values and provenance before an adapter sees the request.
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.id.trim().is_empty() {
            return Err(SimulationError::InvalidRequest(
                "request id cannot be empty".into(),
            ));
        }
        if self.objective.trim().is_empty() {
            return Err(SimulationError::InvalidRequest(
                "request objective cannot be empty".into(),
            ));
        }
        for parameter in &self.parameters {
            if parameter.name.trim().is_empty()
                || parameter.unit.trim().is_empty()
                || parameter.provenance.trim().is_empty()
            {
                return Err(SimulationError::InvalidRequest(
                    "parameters require a name, unit, and provenance".into(),
                ));
            }
            if !parameter.value.is_finite() {
                return Err(SimulationError::InvalidRequest(format!(
                    "parameter {:?} is not finite",
                    parameter.name
                )));
            }
            if let Some(uncertainty) = parameter.uncertainty {
                validate_uncertainty(uncertainty).map_err(SimulationError::InvalidRequest)?;
            }
        }
        if self
            .requested_metrics
            .iter()
            .any(|metric| metric.trim().is_empty())
        {
            return Err(SimulationError::InvalidRequest(
                "requested metric names cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

fn validate_uncertainty(uncertainty: UncertaintyEstimate) -> Result<(), String> {
    if !uncertainty.epistemic.is_finite()
        || !(0.0..=1.0).contains(&uncertainty.epistemic)
        || !uncertainty.aleatoric.is_finite()
        || !(0.0..=1.0).contains(&uncertainty.aleatoric)
    {
        return Err("uncertainty components must be finite values in [0, 1]".into());
    }
    if let Some(interval) = uncertainty.interval {
        if !interval.lower.is_finite()
            || !interval.upper.is_finite()
            || interval.lower > interval.upper
        {
            return Err("uncertainty interval must have finite ordered bounds".into());
        }
    }
    Ok(())
}

/// Metric returned by a solver adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationMetric {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: f64,
    /// Unit string.
    pub unit: String,
    /// Canonical physical semantics when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_type: Option<PhysicalType>,
    /// Optional uncertainty estimate for this metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintyEstimate>,
}

/// How a normalized result was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Legacy/deserialized result with no trustworthy execution provenance.
    #[default]
    Unknown,
    /// Deterministic orchestration fixture; never engineering evidence.
    DryRun,
    /// Parsed output from an external solver invocation.
    ExternalSolver,
}

/// Provenance needed to distinguish fixtures from solver-backed evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SimulationEvidence {
    /// Execution path that produced the result.
    #[serde(default)]
    pub mode: ExecutionMode,
    /// Stable adapter/backend name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Solver version reported by the external executable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_version: Option<String>,
    /// Digest of the fully rendered solver input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_digest: Option<String>,
    /// Digest of the raw solver output parsed by the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
    /// Version of the adapter/parser that normalized the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser_version: Option<String>,
    /// Maturity of the underlying model, independent of execution provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_maturity: Option<ModelMaturity>,
}

impl SimulationEvidence {
    fn has_complete_provenance(&self) -> bool {
        self.backend
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            && self
                .solver_version
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            && self
                .input_digest
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            && self
                .output_digest
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            && self
                .parser_version
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    }
}

/// Solver result normalized for downstream reasoning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Request identifier this result satisfies.
    pub request_id: String,
    /// Whether the solver completed without domain or numerical failure.
    pub converged: bool,
    /// Normalized confidence in the result, from 0.0 to 1.0.
    pub confidence: f64,
    /// Overall uncertainty for this run.
    #[serde(default)]
    pub uncertainty: UncertaintyEstimate,
    /// Metrics extracted from the solver output.
    pub metrics: Vec<SimulationMetric>,
    /// Adapter warnings, simplifications, or unit conversions.
    pub warnings: Vec<String>,
    /// Execution and parser provenance. Defaults to `Unknown` for legacy data.
    #[serde(default)]
    pub evidence: SimulationEvidence,
}

impl SimulationResult {
    /// Create a converged result with bounded confidence.
    pub fn converged(request_id: impl Into<String>, confidence: f64) -> Self {
        let confidence = if confidence.is_finite() {
            confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            request_id: request_id.into(),
            converged: true,
            confidence,
            uncertainty: UncertaintyEstimate::new(1.0 - confidence, 0.0),
            metrics: Vec::new(),
            warnings: Vec::new(),
            evidence: SimulationEvidence::default(),
        }
    }

    /// Create a deterministic fixture result that cannot be mistaken for
    /// solver-backed engineering evidence.
    pub fn dry_run(
        request_id: impl Into<String>,
        backend: impl Into<String>,
        confidence: f64,
    ) -> Self {
        let mut result = Self::converged(request_id, confidence);
        result.evidence = SimulationEvidence {
            mode: ExecutionMode::DryRun,
            backend: Some(backend.into()),
            ..SimulationEvidence::default()
        };
        result
            .warnings
            .push("dry-run fixture: metrics were not produced by an external solver".into());
        result
    }

    /// Attach provenance for a genuinely parsed external-solver result.
    pub fn with_external_evidence(mut self, evidence: SimulationEvidence) -> Self {
        self.evidence = evidence;
        self
    }

    /// Record model maturity separately from solver execution provenance.
    pub fn with_model_maturity(mut self, maturity: ModelMaturity) -> Self {
        self.evidence.model_maturity = Some(maturity);
        self
    }

    /// True only for a parsed external-solver result with complete provenance.
    pub fn is_engineering_evidence(&self) -> bool {
        self.validate().is_ok()
            && self.converged
            && !self.metrics.is_empty()
            && self.evidence.mode == ExecutionMode::ExternalSolver
            && self.evidence.has_complete_provenance()
    }

    /// Add a metric with no explicit uncertainty.
    pub fn with_metric(
        mut self,
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
    ) -> Self {
        self.metrics.push(SimulationMetric {
            name: name.into(),
            value,
            unit: unit.into(),
            uncertainty: None,
        });
        self
    }

    /// Attach run-level uncertainty.
    pub fn with_uncertainty(mut self, uncertainty: UncertaintyEstimate) -> Self {
        self.uncertainty = uncertainty;
        self.confidence = (1.0 - uncertainty.total()).clamp(0.0, 1.0);
        self
    }

    /// Validate normalized values before downstream reasoning or encoding.
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.request_id.trim().is_empty() {
            return Err(SimulationError::Adapter(
                "simulation result has an empty request id".into(),
            ));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(SimulationError::Adapter(
                "simulation result confidence must be finite and in [0, 1]".into(),
            ));
        }
        validate_uncertainty(self.uncertainty).map_err(|error| {
            SimulationError::Adapter(format!("invalid result uncertainty: {error}"))
        })?;
        for metric in &self.metrics {
            if metric.name.trim().is_empty() || metric.unit.trim().is_empty() {
                return Err(SimulationError::Adapter(
                    "simulation metrics require a name and unit".into(),
                ));
            }
            if !metric.value.is_finite() {
                return Err(SimulationError::Adapter(format!(
                    "simulation metric {:?} is not finite",
                    metric.name
                )));
            }
            if let Some(uncertainty) = metric.uncertainty {
                validate_uncertainty(uncertainty).map_err(|error| {
                    SimulationError::Adapter(format!(
                        "invalid uncertainty for metric {:?}: {error}",
                        metric.name
                    ))
                })?;
            }
        }
        Ok(())
    }
}

/// Errors reported by simulator adapters.
#[derive(Debug, Error)]
pub enum SimulationError {
    /// The requested solver family is unavailable in the current installation.
    #[error("solver unavailable: {0:?}")]
    SolverUnavailable(SolverKind),
    /// The request was malformed or under-specified.
    #[error("invalid simulation request: {0}")]
    InvalidRequest(String),
    /// Adapter failed while translating to or from the external solver.
    #[error("adapter error: {0}")]
    Adapter(String),
}

/// Trait implemented by solver-specific adapter crates.
pub trait SimulationBackend: std::fmt::Debug + Send + Sync {
    /// Stable backend name, such as `opensees`, `elmer`, `mujoco`, or `ngspice`.
    fn name(&self) -> &'static str;

    /// Solver families supported by this backend.
    fn supported_solvers(&self) -> &[SolverKind];

    /// Run a normalized simulation request.
    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError>;

    /// Spawns the simulator as a persistent background daemon/process.
    /// Returns the child process handle if supported, or None if not supported.
    fn spawn_daemon(
        &self,
        _request: &SimulationRequest,
    ) -> Result<Option<std::process::Child>, SimulationError> {
        Ok(None)
    }
}

/// Orchestrator for multiple simulation backends.
#[derive(Default, Debug)]
pub struct SimulationRegistry {
    backends: Vec<Box<dyn SimulationBackend>>,
}

impl SimulationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new backend.
    pub fn register(&mut self, backend: impl SimulationBackend + 'static) {
        self.backends.push(Box::new(backend));
    }

    /// Find a backend that supports the given solver family.
    pub fn find_backend(&self, solver: SolverKind) -> Option<&dyn SimulationBackend> {
        self.backends
            .iter()
            .find(|b| b.supported_solvers().contains(&solver))
            .map(|b| b.as_ref())
    }

    /// Run a simulation request using the first matching backend.
    pub fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        request.validate()?;
        let backend = self
            .find_backend(request.solver)
            .ok_or(SimulationError::SolverUnavailable(request.solver))?;
        let result = backend.run(request)?;
        result.validate()?;
        if result.request_id != request.id {
            return Err(SimulationError::Adapter(format!(
                "backend {:?} returned result for request {:?}, expected {:?}",
                backend.name(),
                result.request_id,
                request.id
            )));
        }
        if result.evidence.mode == ExecutionMode::ExternalSolver {
            if result.evidence.backend.as_deref() != Some(backend.name()) {
                return Err(SimulationError::Adapter(format!(
                    "external evidence backend does not match dispatched backend {:?}",
                    backend.name()
                )));
            }
            if !result.evidence.has_complete_provenance() {
                return Err(SimulationError::Adapter(
                    "external solver result has incomplete provenance".into(),
                ));
            }
        }
        Ok(result)
    }

    /// Spawn a simulation daemon using the first matching backend.
    pub fn spawn_daemon(
        &self,
        request: &SimulationRequest,
    ) -> Result<Option<std::process::Child>, SimulationError> {
        request.validate()?;
        let backend = self
            .find_backend(request.solver)
            .ok_or(SimulationError::SolverUnavailable(request.solver))?;
        backend.spawn_daemon(request)
    }
}

/// Translates raw simulation metrics into Holographic Sensation Vectors.
#[derive(Debug, Clone, Default)]
pub struct MetricEncoder {
    /// Dimension of the target HDC space (default 16,384).
    pub dimension: usize,
}

impl MetricEncoder {
    /// Create a new encoder.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Project a validated simulation result into an HDC vector.
    pub fn encode_result(
        &self,
        result: &SimulationResult,
    ) -> Result<ContinuousHV, SimulationError> {
        result.validate()?;
        if self.dimension == 0 {
            return Err(SimulationError::InvalidRequest(
                "metric encoder dimension must be greater than zero".into(),
            ));
        }
        if result.metrics.is_empty() {
            return Err(SimulationError::Adapter(
                "cannot encode a simulation result with no metrics".into(),
            ));
        }
        let mut hv = ContinuousHV::zero(self.dimension);

        for (i, metric) in result.metrics.iter().enumerate() {
            // Bind the metric name and value into the vector
            let name_hv = ContinuousHV::random(self.dimension, seed_from_name(&metric.name));
            // Simple scalar projection: scale the random vector by the metric value
            let sensation = name_hv.scale(metric.value as f32);

            if i == 0 {
                hv = sensation;
            } else {
                hv = ContinuousHV::bundle(&[&hv, &sensation]);
            }
        }

        Ok(hv.normalize())
    }
}

/// Helper for backends that execute external command-line solvers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSolver {
    /// Name of the executable.
    pub cmd: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables.
    pub env: std::collections::HashMap<String, String>,
    /// Wall-clock execution limit in milliseconds.
    #[serde(default = "default_solver_timeout_ms")]
    pub timeout_ms: u64,
    /// Maximum bytes retained from each output stream.
    #[serde(default = "default_solver_output_limit")]
    pub max_output_bytes: usize,
}

const MAX_SOLVER_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_SOLVER_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
/// How long to keep draining output after the direct child process exits.
/// A solver can spawn descendants that inherit its pipes, so EOF may never
/// arrive on its own -- this bounds that wait instead of blocking forever.
const POST_EXIT_DRAIN: Duration = Duration::from_secs(2);

const fn default_solver_timeout_ms() -> u64 {
    5 * 60 * 1000
}

const fn default_solver_output_limit() -> usize {
    1024 * 1024
}

/// Put a pipe file descriptor in non-blocking mode.
///
/// # Safety-relevant context (not `unsafe fn` -- the unsafety is contained)
/// Calls `libc::fcntl` on `fd`, which must be a valid, open file
/// descriptor for the duration of this call. Every caller in this module
/// passes an `AsRawFd` borrow of a live `ChildStdout`/`ChildStderr` it
/// owns, so this precondition always holds.
///
/// This crate is `#![deny(unsafe_code)]`; this is the one, tightly-scoped
/// exception, needed because std has no safe API for putting an
/// already-open pipe fd into non-blocking mode. Kept local to this single
/// function (rather than reached via an external crate that wraps the
/// same libc call) so the unsafety stays visible and auditable here
/// instead of hidden behind a dependency boundary.
#[allow(unsafe_code)]
fn set_nonblocking(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    // SAFETY: `fd` is valid and open (see doc comment); `fcntl` with
    // F_GETFL/F_SETFL cannot invalidate it or invoke UB on a valid fd.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Drain everything currently available from a non-blocking reader,
/// retaining up to `limit` bytes total and discarding (but still
/// consuming, so the pipe doesn't back up and block the writer) anything
/// beyond that. Returns `Ok(true)` on EOF (the write end closed), `Ok(false)`
/// if there was nothing more to read right now (`WouldBlock`), or `Err` on a
/// real I/O error.
///
/// Deliberately synchronous/single-threaded: an earlier version of this
/// function ran on a dedicated OS thread per stream so it could block on
/// `read()`, but a solver that spawns a descendant inheriting its pipes
/// could keep that thread blocked forever after the direct child exited --
/// a slow, unbounded thread leak under repeated adversarial subprocess
/// behavior. Polling a non-blocking fd from the same loop that polls
/// `try_wait()` makes that leak structurally impossible: there is no
/// blocking read left to leak a thread on.
fn drain_nonblocking<R: Read>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    limit: usize,
    exceeded: &mut bool,
) -> std::io::Result<bool> {
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => return Ok(true),
            Ok(n) => {
                let keep = n.min(limit.saturating_sub(buf.len()));
                buf.extend_from_slice(&chunk[..keep]);
                if keep < n {
                    *exceeded = true;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(e) => return Err(e),
        }
    }
}

impl CommandSolver {
    /// Construct a new command solver.
    pub fn new(cmd: impl Into<String>) -> Self {
        Self {
            cmd: cmd.into(),
            args: Vec::new(),
            env: std::collections::HashMap::new(),
            timeout_ms: default_solver_timeout_ms(),
            max_output_bytes: default_solver_output_limit(),
        }
    }

    /// Add an argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set the wall-clock execution limit. Values are checked by [`execute`](Self::execute).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = timeout.as_millis().min(u64::MAX as u128) as u64;
        self
    }

    /// Set the maximum bytes retained from stdout and stderr independently.
    pub fn max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Execute the solver, spawning `cmd` with `args`/`env`, and return its
    /// captured stdout. Execution is killed on timeout or as soon as either
    /// output stream exceeds the configured retention limit.
    ///
    /// Returns `SimulationError::Adapter` if the process cannot be spawned
    /// (e.g. the solver binary is not installed) or exits with a non-zero
    /// status. Callers must not assume a successful exit means the solver's
    /// results are convergent -- that determination requires parsing the
    /// returned stdout, which is solver-specific and is the caller's
    /// responsibility.
    pub fn execute(&self) -> Result<String, SimulationError> {
        if self.cmd.trim().is_empty() {
            return Err(SimulationError::Adapter(
                "solver command cannot be empty".into(),
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_SOLVER_TIMEOUT_MS {
            return Err(SimulationError::Adapter(format!(
                "solver timeout must be between 1 ms and {MAX_SOLVER_TIMEOUT_MS} ms"
            )));
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > MAX_SOLVER_OUTPUT_BYTES {
            return Err(SimulationError::Adapter(format!(
                "solver output limit must be between 1 and {MAX_SOLVER_OUTPUT_BYTES} bytes"
            )));
        }

        let mut child = std::process::Command::new(&self.cmd)
            .args(&self.args)
            .envs(&self.env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                SimulationError::Adapter(format!("failed to spawn '{}': {e}", self.cmd))
            })?;

        let mut stdout = child.stdout.take().ok_or_else(|| {
            SimulationError::Adapter(format!("failed to capture '{}' stdout", self.cmd))
        })?;
        let mut stderr = child.stderr.take().ok_or_else(|| {
            SimulationError::Adapter(format!("failed to capture '{}' stderr", self.cmd))
        })?;
        use std::os::fd::AsRawFd;
        set_nonblocking(stdout.as_raw_fd()).map_err(|e| {
            SimulationError::Adapter(format!(
                "failed to set '{}' stdout non-blocking: {e}",
                self.cmd
            ))
        })?;
        set_nonblocking(stderr.as_raw_fd()).map_err(|e| {
            SimulationError::Adapter(format!(
                "failed to set '{}' stderr non-blocking: {e}",
                self.cmd
            ))
        })?;

        let output_limit = self.max_output_bytes;
        let mut stdout_buf = Vec::with_capacity(output_limit.min(8192));
        let mut stderr_buf = Vec::with_capacity(output_limit.min(8192));
        let mut stdout_exceeded = false;
        let mut stderr_exceeded = false;
        let mut stdout_eof = false;
        let mut stderr_eof = false;

        let started = Instant::now();
        let timeout = Duration::from_millis(self.timeout_ms);
        let mut timed_out = false;
        let mut killed_for_output = false;

        // Draining both streams alongside polling try_wait() (rather than on
        // dedicated blocking-read threads) keeps the child's pipes from
        // filling up and blocking its writes without ever risking a thread
        // stuck on a read that a lingering grandchild process could hold
        // open indefinitely -- see drain_nonblocking's doc comment.
        macro_rules! drain_both {
            () => {{
                if !stdout_eof {
                    match drain_nonblocking(
                        &mut stdout,
                        &mut stdout_buf,
                        output_limit,
                        &mut stdout_exceeded,
                    ) {
                        Ok(eof) => stdout_eof = eof,
                        Err(e) => {
                            let _ = child.kill();
                            let _ = child.wait();
                            return Err(SimulationError::Adapter(format!(
                                "failed reading '{}' stdout: {e}",
                                self.cmd
                            )));
                        }
                    }
                }
                if !stderr_eof {
                    match drain_nonblocking(
                        &mut stderr,
                        &mut stderr_buf,
                        output_limit,
                        &mut stderr_exceeded,
                    ) {
                        Ok(eof) => stderr_eof = eof,
                        Err(e) => {
                            let _ = child.kill();
                            let _ = child.wait();
                            return Err(SimulationError::Adapter(format!(
                                "failed reading '{}' stderr: {e}",
                                self.cmd
                            )));
                        }
                    }
                }
            }};
        }

        let status = loop {
            drain_both!();
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if stdout_exceeded || stderr_exceeded => {
                    killed_for_output = true;
                    let _ = child.kill();
                    break child.wait().map_err(|e| {
                        SimulationError::Adapter(format!(
                            "failed waiting for '{}' after output limit: {e}",
                            self.cmd
                        ))
                    })?;
                }
                Ok(None) if started.elapsed() >= timeout => {
                    timed_out = true;
                    let _ = child.kill();
                    break child.wait().map_err(|e| {
                        SimulationError::Adapter(format!(
                            "failed waiting for '{}' after timeout: {e}",
                            self.cmd
                        ))
                    })?;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(SimulationError::Adapter(format!(
                        "failed polling '{}': {error}",
                        self.cmd
                    )));
                }
            }
        };

        // A solver can spawn descendants that inherit its pipes. Do not let
        // those descendants make this API wait forever after the child
        // exits -- drain for a bounded window and then stop, regardless of
        // whether EOF ever arrives.
        let capture_deadline = Instant::now() + POST_EXIT_DRAIN;
        while Instant::now() < capture_deadline && (!stdout_eof || !stderr_eof) {
            drain_both!();
            if !stdout_eof || !stderr_eof {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        if timed_out {
            return Err(SimulationError::Adapter(format!(
                "'{}' exceeded its {} ms timeout",
                self.cmd, self.timeout_ms
            )));
        }
        if killed_for_output || stdout_exceeded || stderr_exceeded {
            return Err(SimulationError::Adapter(format!(
                "'{}' exceeded the {} byte per-stream output limit",
                self.cmd, self.max_output_bytes
            )));
        }

        if !status.success() {
            return Err(SimulationError::Adapter(format!(
                "'{}' exited with {}: {}",
                self.cmd,
                status,
                String::from_utf8_lossy(&stderr_buf)
            )));
        }

        Ok(String::from_utf8_lossy(&stdout_buf).into_owned())
    }
}

/// Monitor that detects when simulation surprise exceeds thresholds.
///
/// Implements the Active Inference loop for engineering:
/// High Surprise (FEP) → Trigger Simulation → Update Model → Minimize Free Energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseMonitor {
    /// Threshold for "epistemic surprise" above which a simulation is triggered.
    pub surprise_threshold: f64,
    /// Latest measured surprise value.
    pub current_surprise: f64,
}

impl Default for SurpriseMonitor {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.7,
            current_surprise: 0.0,
        }
    }
}

impl SurpriseMonitor {
    /// Update the monitor with a new surprise value (e.g. from FEP agent).
    pub fn update(&mut self, surprise: f64) {
        self.current_surprise = surprise;
    }

    /// Returns true if a simulation should be triggered to resolve uncertainty.
    pub fn should_trigger_sim(&self) -> bool {
        self.current_surprise > self.surprise_threshold
    }
}

/// A high-speed safety interlock inspired by the biological amygdala.
///
/// Overrides motor commands or terminates simulations if "physical surprise"
/// or unsafe thresholds are breached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmygdalaInterlock {
    /// Maximum allowable force/torque before triggering an emergency stop.
    pub torque_limit: f32,
    /// Maximum allowable vibration/oscillation.
    pub vibration_threshold: f32,
    /// Current safety status.
    status: SafetyStatus,
    /// Emergency stops remain active until an explicit reset.
    #[serde(default)]
    latched_stop: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyStatus {
    /// Normal operation.
    Green,
    /// Approaching limits, apply damping.
    Yellow,
    /// E-Stop triggered, clamp all outputs.
    Red,
}

impl Default for AmygdalaInterlock {
    fn default() -> Self {
        Self {
            torque_limit: 100.0,
            vibration_threshold: 10.0,
            status: SafetyStatus::Green,
            latched_stop: false,
        }
    }
}

impl AmygdalaInterlock {
    /// Current safety status.
    pub fn status(&self) -> SafetyStatus {
        self.status
    }

    /// Process sensory data and update the safety status.
    pub fn monitor(&mut self, peak_torque: f32, vibration: f32) -> SafetyStatus {
        if self.latched_stop {
            self.status = SafetyStatus::Red;
            return self.status;
        }

        let invalid_input = !peak_torque.is_finite()
            || !vibration.is_finite()
            || !self.torque_limit.is_finite()
            || !self.vibration_threshold.is_finite()
            || self.torque_limit <= 0.0
            || self.vibration_threshold <= 0.0;
        if invalid_input
            || peak_torque > self.torque_limit * 1.5
            || vibration > self.vibration_threshold * 2.0
        {
            self.status = SafetyStatus::Red;
            self.latched_stop = true;
        } else if peak_torque > self.torque_limit || vibration > self.vibration_threshold {
            self.status = SafetyStatus::Yellow;
        } else {
            self.status = SafetyStatus::Green;
        }
        self.status
    }

    /// Apply the interlock to an output value.
    pub fn apply_override(&self, raw_value: f32) -> f32 {
        if !raw_value.is_finite() {
            return 0.0;
        }
        match self.status {
            SafetyStatus::Green => raw_value,
            SafetyStatus::Yellow => raw_value * 0.5, // Dampen output
            SafetyStatus::Red => 0.0,                // Clamp output
        }
    }

    /// Manually trigger a high-speed emergency stop.
    pub fn trigger_emergency_stop(&mut self) {
        self.status = SafetyStatus::Red;
        self.latched_stop = true;
    }

    /// Whether an emergency stop is currently latched.
    pub fn is_latched(&self) -> bool {
        self.latched_stop
    }

    /// Explicitly re-arm the interlock after the external cause has been cleared.
    pub fn reset_emergency_stop(&mut self) {
        self.latched_stop = false;
        self.status = SafetyStatus::Green;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct MockBackend;
    impl SimulationBackend for MockBackend {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::FiniteElement]
        }
        fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
            Ok(SimulationResult::converged(&request.id, 1.0))
        }
    }

    #[test]
    fn explicit_typed_topology_rejects_unit_presence_mismatch() {
        let metres = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        )
        .with_unit(symthaea_types::UnitRef {
            symbol: "m".into(),
            transform_to_si: symthaea_types::UnitTransform::IDENTITY,
            semantic_id: None,
        });
        let unlabelled = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );

        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", metres)]);

        let consumer = CoupledSimulationStage::new(
            "consumer",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", unlabelled)]);

        let request = MultiPhysicsRequest::new(
            "unit-mismatch",
            "numeric transport must be explicit",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer)
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer",
            "length",
            "consumer",
            "length",
        ));

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_allows_fan_out_to_distinct_destinations() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer_a = CoupledSimulationStage::new(
            "consumer-a",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length.clone())]);
        let consumer_b = CoupledSimulationStage::new(
            "consumer-b",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let request = MultiPhysicsRequest::new(
            "fan-out",
            "preserve legitimate one-to-many topology",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer_a)
        .with_stage(consumer_b)
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer", "length", "consumer-a", "length",
        ))
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer", "length", "consumer-b", "length",
        ));

        assert!(request.validate_typed_topology().is_ok());
    }

    #[test]
    fn explicit_typed_topology_rejects_duplicate_connections() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer = CoupledSimulationStage::new(
            "consumer",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let edge = TypedPhysicalConnection::new(
            "producer", "length", "consumer", "length",
        );
        let request = MultiPhysicsRequest::new(
            "duplicate-edge",
            "reject duplicate typed edge identity",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer)
        .with_typed_connection(edge.clone())
        .with_typed_connection(edge);

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_rejects_multiple_producers_for_one_input() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer_a = CoupledSimulationStage::new(
            "producer-a",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let producer_b = CoupledSimulationStage::new(
            "producer-b",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer = CoupledSimulationStage::new(
            "consumer",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let request = MultiPhysicsRequest::new(
            "ambiguous-destination",
            "reject many-to-one typed endpoint",
            CouplingMode::OneWay,
        )
        .with_stage(producer_a)
        .with_stage(producer_b)
        .with_stage(consumer)
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer-a", "length", "consumer", "length",
        ))
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer-b", "length", "consumer", "length",
        ));

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_rejects_duplicate_stage_ids() {
        let producer = CoupledSimulationStage::new(
            "duplicate",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        );
        let consumer = CoupledSimulationStage::new(
            "duplicate",
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
        );

        let request = MultiPhysicsRequest::new(
            "duplicate-stage",
            "reject ambiguous stage identity",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer);

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_rejects_duplicate_port_names() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([
            TypedPhysicalPort::new("length", length.clone()),
            TypedPhysicalPort::new("length", length),
        ]);

        let request = MultiPhysicsRequest::new(
            "duplicate-port",
            "reject ambiguous port identity",
            CouplingMode::OneWay,
        )
        .with_stage(producer);

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn registry_dispatches_to_matching_backend() {
        let mut registry = SimulationRegistry::new();
        registry.register(MockBackend);

        let request = SimulationRequest::new(
            "test-1",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "test run",
        );

        let result = registry.run(&request).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn registry_fails_if_no_backend_supports_solver() {
        let mut registry = SimulationRegistry::new();
        registry.register(MockBackend);

        let request = SimulationRequest::new(
            "test-1",
            EngineeringDomain::Aerospace,
            SolverKind::ComputationalFluidDynamics,
            "test run",
        );

        let result = registry.run(&request);
        assert!(matches!(
            result,
            Err(SimulationError::SolverUnavailable(
                SolverKind::ComputationalFluidDynamics
            ))
        ));
    }

    #[test]
    fn registry_rejects_non_finite_requests_before_dispatch() {
        let mut registry = SimulationRegistry::new();
        registry.register(MockBackend);
        let request = SimulationRequest::new(
            "invalid-1",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "invalid load case",
        )
        .with_parameter("load", f64::NAN, "N", "sensor");

        assert!(matches!(
            registry.run(&request),
            Err(SimulationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn registry_rejects_result_for_a_different_request() {
        #[derive(Debug)]
        struct WrongIdBackend;
        impl SimulationBackend for WrongIdBackend {
            fn name(&self) -> &'static str {
                "wrong-id"
            }
            fn supported_solvers(&self) -> &[SolverKind] {
                &[SolverKind::FiniteElement]
            }
            fn run(
                &self,
                _request: &SimulationRequest,
            ) -> Result<SimulationResult, SimulationError> {
                Ok(SimulationResult::converged("another-request", 0.9))
            }
        }

        let mut registry = SimulationRegistry::new();
        registry.register(WrongIdBackend);
        let request = SimulationRequest::new(
            "expected-request",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "load case",
        );
        assert!(matches!(
            registry.run(&request),
            Err(SimulationError::Adapter(_))
        ));
    }

    #[test]
    fn request_builder_preserves_parameters() {
        let request = SimulationRequest::new(
            "bridge-load-001",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "screen first-span live-load response",
        )
        .with_parameter("span", 42.0, "m", "concept sketch");

        assert_eq!(request.parameters.len(), 1);
        assert_eq!(request.parameters[0].name, "span");
    }

    #[test]
    fn result_confidence_is_bounded() {
        let result = SimulationResult::converged("run-1", 1.7);
        assert_eq!(result.confidence, 1.0);

        let non_finite = SimulationResult::converged("run-2", f64::NAN);
        assert_eq!(non_finite.confidence, 0.0);
        assert_eq!(non_finite.uncertainty.epistemic, 1.0);
    }

    #[test]
    fn dry_run_is_explicitly_not_engineering_evidence() {
        let result = SimulationResult::dry_run("fixture-1", "mock", 0.8);
        assert_eq!(result.evidence.mode, ExecutionMode::DryRun);
        assert!(!result.is_engineering_evidence());
        assert!(result.warnings.iter().any(|w| w.contains("dry-run")));
    }

    #[test]
    fn engineering_evidence_requires_valid_metrics() {
        let evidence = SimulationEvidence {
            mode: ExecutionMode::ExternalSolver,
            backend: Some("mock".into()),
            solver_version: Some("1.0".into()),
            input_digest: Some("input-digest".into()),
            output_digest: Some("output-digest".into()),
            parser_version: Some("parser-1".into()),
            model_maturity: Some(ModelMaturity::ValidatedNumerical),
        };
        let valid = SimulationResult::converged("run-1", 0.9)
            .with_metric("stress", 12.0, "MPa")
            .with_external_evidence(evidence);
        assert!(valid.is_engineering_evidence());

        let mut invalid = valid;
        invalid.metrics[0].value = f64::NAN;
        assert!(!invalid.is_engineering_evidence());
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn metric_encoder_rejects_invalid_or_empty_results() {
        let encoder = MetricEncoder::new(128);
        let empty = SimulationResult::converged("empty", 0.8);
        assert!(encoder.encode_result(&empty).is_err());

        let invalid =
            SimulationResult::converged("invalid", 0.8).with_metric("stress", f64::INFINITY, "MPa");
        assert!(encoder.encode_result(&invalid).is_err());
    }

    #[test]
    fn emergency_stop_is_latched_and_nan_fails_closed() {
        let mut interlock = AmygdalaInterlock::default();
        assert_eq!(interlock.monitor(f32::NAN, 0.0), SafetyStatus::Red);
        assert!(interlock.is_latched());
        assert_eq!(interlock.monitor(0.0, 0.0), SafetyStatus::Red);
        assert_eq!(interlock.apply_override(f32::NAN), 0.0);

        interlock.reset_emergency_stop();
        assert!(!interlock.is_latched());
        assert_eq!(interlock.monitor(0.0, 0.0), SafetyStatus::Green);
    }

    #[test]
    fn uncertainty_and_interval_are_bounded() {
        let interval = Interval::new(10.0, 2.0);
        assert_eq!(interval.lower, 2.0);
        assert!(interval.contains(6.0));

        let uncertainty = UncertaintyEstimate::new(1.7, -0.2).with_interval(interval);
        assert_eq!(uncertainty.epistemic, 1.0);
        assert_eq!(uncertainty.aleatoric, 0.0);
        assert_eq!(uncertainty.interval.unwrap().midpoint(), 6.0);

        let non_finite = UncertaintyEstimate::new(f64::NAN, f64::INFINITY);
        assert_eq!(non_finite.epistemic, 1.0);
        assert_eq!(non_finite.aleatoric, 1.0);
        assert_eq!(non_finite.total(), 1.0);
    }

    #[test]
    fn multiphysics_request_tracks_coupling_stages() {
        let request = MultiPhysicsRequest::new(
            "aero-thermal-structural-001",
            "screen coupled wing thermal stress",
            CouplingMode::Iterative,
        )
        .with_stage(
            CoupledSimulationStage::new(
                "cfd",
                EngineeringDomain::Aerospace,
                SolverKind::ComputationalFluidDynamics,
            )
            .produces(["pressure_field", "heat_flux"]),
        )
        .with_stage(
            CoupledSimulationStage::new(
                "fea",
                EngineeringDomain::Materials,
                SolverKind::FiniteElement,
            )
            .consumes(["pressure_field", "heat_flux"])
            .produces(["stress_field"]),
        );

        assert_eq!(request.stages.len(), 2);
        assert_eq!(request.coupling, CouplingMode::Iterative);
    }

    #[test]
    fn typed_multiphysics_connections_accept_matching_types() {
        let pressure = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Pressure,
            symthaea_types::PhysicalDimension::PRESSURE,
        );
        let request = MultiPhysicsRequest::new(
            "typed-1",
            "pressure transfer",
            CouplingMode::OneWay,
        )
        .with_stage(
            CoupledSimulationStage::new(
                "cfd",
                EngineeringDomain::Aerospace,
                SolverKind::ComputationalFluidDynamics,
            )
            .typed_produces([TypedPhysicalPort::new("pressure", pressure.clone())]),
        )
        .with_stage(
            CoupledSimulationStage::new(
                "fea",
                EngineeringDomain::Materials,
                SolverKind::FiniteElement,
            )
            .typed_consumes([TypedPhysicalPort::new("pressure", pressure)]),
        );
        assert!(request.validate_typed_connections().is_ok());
    }

    #[test]
    fn typed_multiphysics_connections_reject_dimension_mismatch() {
        let pressure = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Pressure,
            symthaea_types::PhysicalDimension::PRESSURE,
        );
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let request = MultiPhysicsRequest::new("typed-2", "bad transfer", CouplingMode::OneWay)
            .with_stage(
                CoupledSimulationStage::new(
                    "source",
                    EngineeringDomain::Aerospace,
                    SolverKind::ComputationalFluidDynamics,
                )
                .typed_produces([TypedPhysicalPort::new("signal", pressure)]),
            )
            .with_stage(
                CoupledSimulationStage::new(
                    "sink",
                    EngineeringDomain::Materials,
                    SolverKind::FiniteElement,
                )
                .typed_consumes([TypedPhysicalPort::new("signal", length)]),
            );
        assert!(request.validate_typed_connections().is_err());
    }

    #[test]
    fn typed_multiphysics_connections_reject_unknown_semantics() {
        let request = MultiPhysicsRequest::new("typed-3", "unknown transfer", CouplingMode::OneWay)
            .with_stage(
                CoupledSimulationStage::new(
                    "source",
                    EngineeringDomain::Aerospace,
                    SolverKind::ComputationalFluidDynamics,
                )
                .typed_produces([TypedPhysicalPort::new("signal", PhysicalType::unknown())]),
            )
            .with_stage(
                CoupledSimulationStage::new(
                    "sink",
                    EngineeringDomain::Materials,
                    SolverKind::FiniteElement,
                )
                .typed_consumes([TypedPhysicalPort::new("signal", PhysicalType::unknown())]),
            );
        assert!(request.validate_typed_connections().is_err());
    }

    #[test]
    fn typed_parameter_and_metric_preserve_semantics() {
        let energy = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Energy,
            symthaea_types::PhysicalDimension::ENERGY,
        );
        let request = SimulationRequest::new(
            "typed-4",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
            "energy check",
        )
        .with_typed_parameter("energy", 10.0, "J", energy.clone(), "fixture");
        assert_eq!(request.parameters[0].physical_type, Some(energy.clone()));

        let result = SimulationResult::converged("typed-4", 0.9)
            .with_typed_metric("energy", 10.0, "J", energy.clone());
        assert_eq!(result.metrics[0].physical_type, Some(energy));
    }

    #[test]
    fn explicit_typed_topology_targets_exact_stage_endpoints() {
        let pressure = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Pressure,
            symthaea_types::PhysicalDimension::PRESSURE,
        );
        let load = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Force,
            symthaea_types::PhysicalDimension::FORCE,
        );
        let request = MultiPhysicsRequest::new(
            "typed-topology",
            "explicit coupling",
            CouplingMode::OneWay,
        )
        .with_stage(
            CoupledSimulationStage::new(
                "cfd",
                EngineeringDomain::Aerospace,
                SolverKind::ComputationalFluidDynamics,
            )
            .typed_produces([TypedPhysicalPort::new("pressure", pressure)]),
        )
        .with_stage(
            CoupledSimulationStage::new(
                "fea",
                EngineeringDomain::Materials,
                SolverKind::FiniteElement,
            )
            .typed_consumes([TypedPhysicalPort::new("load", load)]),
        )
        .with_typed_connection(TypedPhysicalConnection::new(
            "cfd", "pressure", "fea", "load",
        ));
        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn command_solver_actually_spawns_and_captures_stdout() {
        let solver = CommandSolver::new("echo").arg("hello-from-solver");
        let output = solver.execute().expect("echo should always succeed");
        assert!(
            output.contains("hello-from-solver"),
            "expected real echo output, got: {output:?}"
        );
    }

    #[test]
    fn command_solver_deserialization_applies_resource_defaults() {
        let solver: CommandSolver =
            serde_json::from_str(r#"{"cmd":"echo","args":[],"env":{}}"#).unwrap();
        assert_eq!(solver.timeout_ms, default_solver_timeout_ms());
        assert_eq!(solver.max_output_bytes, default_solver_output_limit());
    }

    #[test]
    fn command_solver_errors_on_missing_binary() {
        let solver = CommandSolver::new("symthaea-nonexistent-solver-binary-xyz");
        let err = solver.execute().unwrap_err();
        assert!(matches!(err, SimulationError::Adapter(_)));
    }

    #[test]
    fn command_solver_errors_on_nonzero_exit() {
        // `false` is a standard POSIX utility that always exits non-zero.
        let solver = CommandSolver::new("false");
        let err = solver.execute().unwrap_err();
        assert!(matches!(err, SimulationError::Adapter(_)));
    }

    #[test]
    fn command_solver_kills_timed_out_process() {
        let solver = CommandSolver::new("sleep")
            .arg("1")
            .timeout(Duration::from_millis(20));
        let started = Instant::now();
        let err = solver.execute().unwrap_err();
        assert!(matches!(err, SimulationError::Adapter(_)));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn command_solver_rejects_excess_output() {
        let solver = CommandSolver::new("printf")
            .arg("0123456789abcdef")
            .max_output_bytes(8);
        let err = solver.execute().unwrap_err();
        assert!(matches!(err, SimulationError::Adapter(_)));
    }
}
