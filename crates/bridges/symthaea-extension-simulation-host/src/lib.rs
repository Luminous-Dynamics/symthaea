// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed zero-import simulation data-plane host for Symthaea Components.
//!
//! This crate promotes the `simulation-provider-v1` conformance mechanics into
//! a production-shaped runtime boundary while deliberately leaving signer trust,
//! admission, provider selection, currentness, and engineering-evidence policy
//! outside Wasmtime.
//!
//! Invocation is correctness-first: the exact manifest/component bytes are first
//! re-inspected through [`symthaea_extension_host::ControlPlaneHost`], then the
//! Component is instantiated through an independently empty typed linker for the
//! simulation call. This currently compiles the Component twice. A future cache
//! may remove that cost only if it preserves exact-byte/profile binding.
//!
//! As with the control host, Wasmtime `Store` limits, fuel, and epoch deadlines
//! bound guest execution after compilation. They do not by themselves contain
//! hostile-input JIT compilation resource use.

#![deny(unsafe_code)]

use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use symthaea_extension_core::ExtensionKind;
use symthaea_extension_host::{
    ControlHostError, ControlHostPolicy, ControlPlaneHost, GuestHealthState,
};
use symthaea_sim_bridge::{
    EngineeringDomain as NativeDomain, Interval as NativeInterval,
    ModelParameter as NativeParameter, SimulationEvidence, SimulationMetric as NativeMetric,
    SimulationRequest as NativeRequest, SimulationResult, SolverKind as NativeSolver,
    UncertaintyEstimate as NativeUncertainty,
};
use thiserror::Error;
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder};

wasmtime::component::bindgen!({
    world: "simulation-provider-v1",
    path: "../../core/symthaea-extension-core/wit",
});

use exports::luminous::symthaea_extension::control::HealthState;
use exports::luminous::symthaea_extension::simulation_types::{
    Interval as WitInterval, ModelParameter as WitParameter,
    SimulationDomain as WitDomain, SimulationMetric as WitMetric,
    SimulationOutput as WitOutput, SimulationProviderError as WitProviderError,
    SimulationRequest as WitRequest, SolverKind as WitSolver,
    Uncertainty as WitUncertainty,
};

/// Stable runtime/codegen profile for typed simulation execution.
pub const SIMULATION_WASM_PROFILE_V1: &str =
    "wasmtime-44.0.1/component-model/empty-linker/simulation-v1";
/// Exact public WIT world consumed by this host.
pub const SIMULATION_WIT_V1: &str = "luminous:symthaea-extension/simulation-provider-v1@1.0.0";
/// Host adapter identity for native <-> WIT conversion and output validation.
pub const SIMULATION_ADAPTER_V1: &str = "symthaea-extension-simulation-host-v1";

/// Guest-declared computational failure. These values carry no host authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuestSimulationFailure {
    InvalidRequest(String),
    UnsupportedSolver,
    DomainFailure(String),
    ResourceExhausted,
    Internal(String),
}

/// Failures owned by the technical simulation host.
#[derive(Debug, Error)]
pub enum SimulationHostError {
    #[error(transparent)]
    Control(#[from] ControlHostError),
    #[error("native simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error("extension manifest is not a simulation extension")]
    NotSimulationExtension,
    #[error("control inspection did not report ready health")]
    InspectionNotReady,
    #[error("same-instance control health is not ready")]
    InstanceNotReady,
    #[error("same-instance Component control surface drifted from control inspection")]
    ControlSurfaceDrift,
    #[error("host memory limit does not fit this platform")]
    MemoryLimitOverflow,
    #[error("simulation component compilation failed: {0}")]
    ComponentCompile(String),
    #[error("simulation component has unexpected imports or type mismatch: {0}")]
    UnexpectedImportsOrTypeMismatch(String),
    #[error("simulation component instantiation/execution failed: {0}")]
    Execution(String),
    #[error("guest simulation failed: {0:?}")]
    Guest(GuestSimulationFailure),
    #[error("guest returned an invalid normalized simulation output: {0}")]
    InvalidOutput(String),
    #[error("guest returned result for request {guest:?}, expected {expected:?}")]
    RequestIdMismatch { guest: String, expected: String },
    #[error("simulation output exceeds the declared/host output budget")]
    OutputTooLarge,
}

/// Host-minted technical execution receipt.
///
/// Fields are private and this type is not serializable. It is not admission or
/// engineering evidence; it only proves that this in-process host path produced
/// a validated normalized result from the exact inspected bytes.
#[derive(Debug, Clone)]
pub struct SimulationInvocation {
    result: SimulationResult,
    extension_id: String,
    extension_version: String,
    manifest_sha256: [u8; 32],
    component_sha256: [u8; 32],
    control_wasm_profile: &'static str,
    simulation_wasm_profile: &'static str,
    wit_version: &'static str,
    adapter_version: &'static str,
}

impl SimulationInvocation {
    pub fn result(&self) -> &SimulationResult {
        &self.result
    }

    pub fn into_result(self) -> SimulationResult {
        self.result
    }

    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    pub const fn component_sha256(&self) -> [u8; 32] {
        self.component_sha256
    }

    pub const fn control_wasm_profile(&self) -> &'static str {
        self.control_wasm_profile
    }

    pub const fn simulation_wasm_profile(&self) -> &'static str {
        self.simulation_wasm_profile
    }

    pub const fn wit_version(&self) -> &'static str {
        self.wit_version
    }

    pub const fn adapter_version(&self) -> &'static str {
        self.adapter_version
    }
}

/// Production-shaped typed simulation host with zero ambient imports.
#[derive(Debug, Clone, Copy)]
pub struct SimulationComponentHost {
    control: ControlPlaneHost,
}

impl Default for SimulationComponentHost {
    fn default() -> Self {
        Self::new(ControlHostPolicy::default())
    }
}

impl SimulationComponentHost {
    pub fn new(policy: ControlHostPolicy) -> Self {
        Self {
            control: ControlPlaneHost::new(policy),
        }
    }

    pub fn policy(&self) -> ControlHostPolicy {
        self.control.policy()
    }

    /// Inspect exact package bytes, execute the typed simulation world with an
    /// empty linker, validate the output, and return a non-authoritative receipt.
    pub fn invoke(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
        request: &NativeRequest,
    ) -> Result<SimulationInvocation, SimulationHostError> {
        request
            .validate()
            .map_err(|error| SimulationHostError::InvalidRequest(error.to_string()))?;

        let inspection = self.control.inspect(manifest_bytes, component_bytes)?;
        if inspection.manifest.kind != ExtensionKind::Simulation {
            return Err(SimulationHostError::NotSimulationExtension);
        }
        if inspection.health.state != GuestHealthState::Ready {
            return Err(SimulationHostError::InspectionNotReady);
        }

        let engine = simulation_engine()?;
        let component = Component::new(&engine, component_bytes)
            .map_err(|error| SimulationHostError::ComponentCompile(error.to_string()))?;
        let linker = Linker::<HostState>::new(&engine);
        let instance_pre = linker
            .instantiate_pre(&component)
            .map_err(|error| {
                SimulationHostError::UnexpectedImportsOrTypeMismatch(error.to_string())
            })?;
        let bindings_pre = SimulationProviderV1Pre::new(instance_pre)
            .map_err(|error| SimulationHostError::UnexpectedImportsOrTypeMismatch(error.to_string()))?;

        let memory_size = usize::try_from(inspection.manifest.resources.memory_bytes)
            .map_err(|_| SimulationHostError::MemoryLimitOverflow)?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_size)
            .instances(8)
            .tables(4)
            .memories(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&engine, HostState { limits });
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(inspection.manifest.resources.fuel)
            .map_err(|error| SimulationHostError::Execution(error.to_string()))?;
        store.set_epoch_deadline(1);
        store.epoch_deadline_trap();

        let deadline = Duration::from_millis(inspection.manifest.resources.max_wall_time_ms);
        let (cancel_tx, cancel_rx) = mpsc::channel::<()>();
        let deadline_engine = engine.clone();
        let timer = thread::spawn(move || {
            if cancel_rx.recv_timeout(deadline).is_err() {
                deadline_engine.increment_epoch();
            }
        });

        let execution = (|| {
            let bindings = bindings_pre
                .instantiate(&mut store)
                .map_err(|error| SimulationHostError::Execution(error.to_string()))?;

            let identity = bindings
                .luminous_symthaea_extension_control()
                .call_identity(&mut store)
                .map_err(|error| SimulationHostError::Execution(error.to_string()))?;
            if identity.id != inspection.identity.id
                || identity.version != inspection.identity.version
                || identity.abi_major != inspection.identity.abi_major
                || identity.abi_minor != inspection.identity.abi_minor
                || identity.manifest_digest != inspection.identity.manifest_digest
            {
                return Err(SimulationHostError::ControlSurfaceDrift);
            }

            let health = bindings
                .luminous_symthaea_extension_control()
                .call_health(&mut store)
                .map_err(|error| SimulationHostError::Execution(error.to_string()))?;
            if !matches!(health.state, HealthState::Ready) {
                return Err(SimulationHostError::InstanceNotReady);
            }
            if health.message != inspection.health.message {
                return Err(SimulationHostError::ControlSurfaceDrift);
            }

            let wit_request = to_wit_request(request);
            let output = bindings
                .luminous_symthaea_extension_simulation_provider()
                .call_simulate(&mut store, &wit_request)
                .map_err(|error| SimulationHostError::Execution(error.to_string()))?
                .map_err(|error| SimulationHostError::Guest(map_guest_failure(error)))?;

            normalize_output(request, output)
        })();

        let _ = cancel_tx.send(());
        let _ = timer.join();
        let result = execution?;
        validate_output_budget(
            &result,
            inspection.manifest.resources.max_output_bytes,
            self.control.policy().max_output_bytes,
        )?;

        Ok(SimulationInvocation {
            result,
            extension_id: inspection.identity.id,
            extension_version: inspection.identity.version,
            manifest_sha256: inspection.manifest_sha256,
            component_sha256: inspection.component_sha256,
            control_wasm_profile: inspection.wasm_profile,
            simulation_wasm_profile: SIMULATION_WASM_PROFILE_V1,
            wit_version: SIMULATION_WIT_V1,
            adapter_version: SIMULATION_ADAPTER_V1,
        })
    }
}

#[derive(Debug)]
struct HostState {
    limits: StoreLimits,
}

fn simulation_engine() -> Result<Engine, SimulationHostError> {
    let mut config = Config::new();
    config
        .wasm_component_model(true)
        .wasm_relaxed_simd(false)
        .relaxed_simd_deterministic(true)
        .wasm_memory64(false)
        .wasm_multi_memory(false)
        .wasm_tail_call(false)
        .wasm_stack_switching(false)
        .cranelift_nan_canonicalization(true)
        .consume_fuel(true)
        .epoch_interruption(true);
    Engine::new(&config).map_err(|error| SimulationHostError::Execution(error.to_string()))
}

fn to_wit_request(request: &NativeRequest) -> WitRequest {
    WitRequest {
        id: request.id.clone(),
        domain: to_wit_domain(request.domain),
        solver: to_wit_solver(request.solver),
        objective: request.objective.clone(),
        parameters: request.parameters.iter().map(to_wit_parameter).collect(),
        requested_metrics: request.requested_metrics.clone(),
    }
}

fn to_wit_parameter(parameter: &NativeParameter) -> WitParameter {
    WitParameter {
        name: parameter.name.clone(),
        value: parameter.value,
        unit: parameter.unit.clone(),
        provenance: parameter.provenance.clone(),
        uncertainty: parameter.uncertainty.map(to_wit_uncertainty),
    }
}

fn to_wit_uncertainty(value: NativeUncertainty) -> WitUncertainty {
    WitUncertainty {
        epistemic: value.epistemic,
        aleatoric: value.aleatoric,
        interval: value.interval.map(|interval| WitInterval {
            lower: interval.lower,
            upper: interval.upper,
        }),
    }
}

fn to_wit_domain(domain: NativeDomain) -> WitDomain {
    match domain {
        NativeDomain::Civil => WitDomain::Civil,
        NativeDomain::Mechanical => WitDomain::Mechanical,
        NativeDomain::Electrical => WitDomain::Electrical,
        NativeDomain::Aerospace => WitDomain::Aerospace,
        NativeDomain::ChemicalProcess => WitDomain::ChemicalProcess,
        NativeDomain::Robotics => WitDomain::Robotics,
        NativeDomain::Nuclear => WitDomain::Nuclear,
        NativeDomain::Materials => WitDomain::Materials,
        NativeDomain::Environmental => WitDomain::Environmental,
        NativeDomain::Systems => WitDomain::Systems,
    }
}

fn to_wit_solver(solver: NativeSolver) -> WitSolver {
    match solver {
        NativeSolver::FiniteElement => WitSolver::FiniteElement,
        NativeSolver::ComputationalFluidDynamics => WitSolver::ComputationalFluidDynamics,
        NativeSolver::MultibodyDynamics => WitSolver::MultibodyDynamics,
        NativeSolver::Circuit => WitSolver::Circuit,
        NativeSolver::Process => WitSolver::Process,
        NativeSolver::CadGeometry => WitSolver::CadGeometry,
        NativeSolver::MultiPhysics => WitSolver::MultiPhysics,
        NativeSolver::Custom => WitSolver::Custom,
    }
}

fn normalize_output(
    request: &NativeRequest,
    output: WitOutput,
) -> Result<SimulationResult, SimulationHostError> {
    if output.request_id != request.id {
        return Err(SimulationHostError::RequestIdMismatch {
            guest: output.request_id,
            expected: request.id.clone(),
        });
    }

    let result = SimulationResult {
        request_id: output.request_id,
        converged: output.converged,
        confidence: output.confidence,
        uncertainty: from_wit_uncertainty(output.uncertainty),
        metrics: output.metrics.into_iter().map(from_wit_metric).collect(),
        warnings: output.warnings,
        evidence: SimulationEvidence::default(),
    };
    result
        .validate()
        .map_err(|error| SimulationHostError::InvalidOutput(error.to_string()))?;
    Ok(result)
}

fn from_wit_metric(metric: WitMetric) -> NativeMetric {
    NativeMetric {
        name: metric.name,
        value: metric.value,
        unit: metric.unit,
        uncertainty: metric.uncertainty.map(from_wit_uncertainty),
    }
}

fn from_wit_uncertainty(value: WitUncertainty) -> NativeUncertainty {
    NativeUncertainty {
        epistemic: value.epistemic,
        aleatoric: value.aleatoric,
        interval: value.interval.map(|interval| NativeInterval {
            lower: interval.lower,
            upper: interval.upper,
        }),
    }
}

fn map_guest_failure(error: WitProviderError) -> GuestSimulationFailure {
    match error {
        WitProviderError::InvalidRequest(message) => GuestSimulationFailure::InvalidRequest(message),
        WitProviderError::UnsupportedSolver => GuestSimulationFailure::UnsupportedSolver,
        WitProviderError::DomainFailure(message) => GuestSimulationFailure::DomainFailure(message),
        WitProviderError::ResourceExhausted => GuestSimulationFailure::ResourceExhausted,
        WitProviderError::Internal(message) => GuestSimulationFailure::Internal(message),
    }
}

fn validate_output_budget(
    result: &SimulationResult,
    manifest_limit: u64,
    host_limit: u64,
) -> Result<(), SimulationHostError> {
    // Conservative logical v1 size: charge every scalar/tag and every string/list
    // length prefix. This is not a claim about Wasmtime's internal canonical-ABI
    // allocation; guest memory is independently bounded by StoreLimits.
    let mut bytes = 8usize
        .saturating_add(result.request_id.len())
        .saturating_add(1)
        .saturating_add(8)
        .saturating_add(uncertainty_size(result.uncertainty))
        .saturating_add(8)
        .saturating_add(8);
    for metric in &result.metrics {
        bytes = bytes
            .saturating_add(8)
            .saturating_add(metric.name.len())
            .saturating_add(8)
            .saturating_add(8)
            .saturating_add(metric.unit.len())
            .saturating_add(1)
            .saturating_add(metric.uncertainty.map_or(0, uncertainty_size));
    }
    for warning in &result.warnings {
        bytes = bytes.saturating_add(8).saturating_add(warning.len());
    }
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    if bytes > manifest_limit || bytes > host_limit {
        return Err(SimulationHostError::OutputTooLarge);
    }
    Ok(())
}

fn uncertainty_size(value: NativeUncertainty) -> usize {
    16usize
        .saturating_add(1)
        .saturating_add(value.interval.map_or(0, |_| 16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_request_maps_every_public_enum_without_loss() {
        let domains = [
            NativeDomain::Civil,
            NativeDomain::Mechanical,
            NativeDomain::Electrical,
            NativeDomain::Aerospace,
            NativeDomain::ChemicalProcess,
            NativeDomain::Robotics,
            NativeDomain::Nuclear,
            NativeDomain::Materials,
            NativeDomain::Environmental,
            NativeDomain::Systems,
        ];
        for domain in domains {
            let request = NativeRequest::new("enum", domain, NativeSolver::Custom, "enum map");
            let mapped = to_wit_request(&request);
            assert_eq!(mapped.id, "enum");
        }

        let solvers = [
            NativeSolver::FiniteElement,
            NativeSolver::ComputationalFluidDynamics,
            NativeSolver::MultibodyDynamics,
            NativeSolver::Circuit,
            NativeSolver::Process,
            NativeSolver::CadGeometry,
            NativeSolver::MultiPhysics,
            NativeSolver::Custom,
        ];
        for solver in solvers {
            let request = NativeRequest::new("solver", NativeDomain::Systems, solver, "solver map");
            assert_eq!(to_wit_request(&request).id, "solver");
        }
    }

    #[test]
    fn typed_guest_failures_remain_distinct_from_host_failures() {
        assert_eq!(
            map_guest_failure(WitProviderError::UnsupportedSolver),
            GuestSimulationFailure::UnsupportedSolver
        );
        assert_eq!(
            map_guest_failure(WitProviderError::ResourceExhausted),
            GuestSimulationFailure::ResourceExhausted
        );
        assert_eq!(
            map_guest_failure(WitProviderError::DomainFailure("domain".into())),
            GuestSimulationFailure::DomainFailure("domain".into())
        );
    }

    #[test]
    fn output_budget_counts_guest_owned_strings_and_metrics() {
        let result = SimulationResult::converged("request", 0.5)
            .with_metric("metric", 1.0, "unit");
        assert!(validate_output_budget(&result, 1024, 1024).is_ok());
        assert!(matches!(
            validate_output_budget(&result, 1, 1024),
            Err(SimulationHostError::OutputTooLarge)
        ));
    }

    #[test]
    fn invocation_receipt_is_not_simulation_authority() {
        let result = SimulationResult::converged("fixture", 0.0);
        let receipt = SimulationInvocation {
            result,
            extension_id: "org.example.fixture".into(),
            extension_version: "0.1.0".into(),
            manifest_sha256: [1; 32],
            component_sha256: [2; 32],
            control_wasm_profile: "control",
            simulation_wasm_profile: SIMULATION_WASM_PROFILE_V1,
            wit_version: SIMULATION_WIT_V1,
            adapter_version: SIMULATION_ADAPTER_V1,
        };
        assert_eq!(receipt.result().evidence, SimulationEvidence::default());
        assert_eq!(receipt.component_sha256(), [2; 32]);
    }
}
