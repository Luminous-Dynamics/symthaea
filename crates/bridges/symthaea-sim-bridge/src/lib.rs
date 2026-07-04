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
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::seed_from_name;
use thiserror::Error;

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
pub struct CoupledSimulationStage {
    /// Stage identifier unique within the request.
    pub id: String,
    /// Domain covered by the stage.
    pub domain: EngineeringDomain,
    /// Solver family used by the stage.
    pub solver: SolverKind,
    /// Outputs from previous stages consumed by this stage.
    pub consumes: Vec<String>,
    /// Metrics or fields emitted for later stages.
    pub produces: Vec<String>,
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
    /// Coupling convergence tolerance for iterative/co-simulation workflows.
    pub coupling_tolerance: f64,
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
            coupling_tolerance: 1e-3,
        }
    }

    /// Append a stage.
    pub fn with_stage(mut self, stage: CoupledSimulationStage) -> Self {
        self.stages.push(stage);
        self
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
            epistemic: epistemic.clamp(0.0, 1.0),
            aleatoric: aleatoric.clamp(0.0, 1.0),
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
        (self.epistemic + self.aleatoric).clamp(0.0, 1.0)
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
    /// Optional uncertainty estimate for this metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintyEstimate>,
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
}

impl SimulationResult {
    /// Create a converged result with bounded confidence.
    pub fn converged(request_id: impl Into<String>, confidence: f64) -> Self {
        Self {
            request_id: request_id.into(),
            converged: true,
            confidence: confidence.clamp(0.0, 1.0),
            uncertainty: UncertaintyEstimate::new(1.0 - confidence.clamp(0.0, 1.0), 0.0),
            metrics: Vec::new(),
            warnings: Vec::new(),
        }
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
        let backend = self
            .find_backend(request.solver)
            .ok_or(SimulationError::SolverUnavailable(request.solver))?;
        backend.run(request)
    }

    /// Spawn a simulation daemon using the first matching backend.
    pub fn spawn_daemon(
        &self,
        request: &SimulationRequest,
    ) -> Result<Option<std::process::Child>, SimulationError> {
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

    /// Project a simulation result into an HDC vector.
    pub fn encode_result(&self, result: &SimulationResult) -> ContinuousHV {
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

        hv.normalize()
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
}

impl CommandSolver {
    /// Construct a new command solver.
    pub fn new(cmd: impl Into<String>) -> Self {
        Self {
            cmd: cmd.into(),
            args: Vec::new(),
            env: std::collections::HashMap::new(),
        }
    }

    /// Add an argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Execute the solver, spawning `cmd` with `args`/`env`, and return its
    /// captured stdout.
    ///
    /// Returns `SimulationError::Adapter` if the process cannot be spawned
    /// (e.g. the solver binary is not installed) or exits with a non-zero
    /// status. Callers must not assume a successful exit means the solver's
    /// results are convergent -- that determination requires parsing the
    /// returned stdout, which is solver-specific and is the caller's
    /// responsibility.
    pub fn execute(&self) -> Result<String, SimulationError> {
        let output = std::process::Command::new(&self.cmd)
            .args(&self.args)
            .envs(&self.env)
            .output()
            .map_err(|e| {
                SimulationError::Adapter(format!("failed to spawn '{}': {e}", self.cmd))
            })?;

        if !output.status.success() {
            return Err(SimulationError::Adapter(format!(
                "'{}' exited with {}: {}",
                self.cmd,
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
    pub status: SafetyStatus,
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
        }
    }
}

impl AmygdalaInterlock {
    /// Process sensory data and update the safety status.
    pub fn monitor(&mut self, peak_torque: f32, vibration: f32) -> SafetyStatus {
        if peak_torque > self.torque_limit * 1.5 || vibration > self.vibration_threshold * 2.0 {
            self.status = SafetyStatus::Red;
        } else if peak_torque > self.torque_limit || vibration > self.vibration_threshold {
            self.status = SafetyStatus::Yellow;
        } else {
            self.status = SafetyStatus::Green;
        }
        self.status
    }

    /// Apply the interlock to an output value.
    pub fn apply_override(&self, raw_value: f32) -> f32 {
        match self.status {
            SafetyStatus::Green => raw_value,
            SafetyStatus::Yellow => raw_value * 0.5, // Dampen output
            SafetyStatus::Red => 0.0,                // Clamp output
        }
    }

    /// Manually trigger a high-speed emergency stop.
    pub fn trigger_emergency_stop(&mut self) {
        self.status = SafetyStatus::Red;
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
    fn command_solver_actually_spawns_and_captures_stdout() {
        let solver = CommandSolver::new("echo").arg("hello-from-solver");
        let output = solver.execute().expect("echo should always succeed");
        assert!(
            output.contains("hello-from-solver"),
            "expected real echo output, got: {output:?}"
        );
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
}
