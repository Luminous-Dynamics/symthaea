// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Admission-bound Wasm Component simulation backend.
//!
//! This crate is the narrow join between extension routing/authority and the
//! zero-import typed simulation runtime. It deliberately does not cache or mint
//! admission authority. The lazy router proves exact manifest/payload bindings
//! and live currentness around execution; this backend executes the immutable
//! bytes and constructs host-owned execution provenance from the resulting
//! technical receipt plus canonical request/output digests.

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::sync::Arc;
use symthaea_extension_admission::Sha256Digest;
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_extension_host::ControlHostPolicy;
use symthaea_extension_simulation_host::{
    GuestSimulationFailure, SimulationComponentHost, SimulationHostError,
};
use symthaea_sim_bridge::{
    ExecutionMode, ExtensionComponentEvidence, SimulationBackend, SimulationError,
    SimulationEvidence, SimulationRequest, SimulationResult, SolverKind,
};
use symthaea_sim_digest::{
    SIMULATION_DIGEST_PROFILE_V1, canonical_output_sha256_v1, canonical_request_sha256_v1,
};
use symthaea_sim_extension_routing::{
    DescriptorProblem, SelectedExecutionPermit, SimulationBackendFactory,
    SimulationProviderDescriptor,
};
use thiserror::Error;

/// Stable backend name used by every public zero-import Wasm simulation Component.
pub const WASM_SIMULATION_BACKEND_V1: &str = "extension-component-v1";
/// Outer evidence-construction profile layered over the typed runtime adapter.
pub const WASM_SIMULATION_EVIDENCE_ADAPTER_V1: &str =
    "symthaea-extension-simulation-backend-evidence-v1";

/// Construction failure before the factory is registered for routing.
#[derive(Debug, Error)]
pub enum WasmSimulationFactoryError {
    #[error("manifest JSON is invalid: {0}")]
    ManifestJson(#[from] serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("manifest is not a simulation extension")]
    NotSimulationExtension,
    #[error("simulation Component factory requires runtime=wasm")]
    RuntimeNotWasm,
    #[error("component byte buffer cannot be empty")]
    EmptyComponent,
    #[error("simulation provider descriptor is invalid: {0:?}")]
    Descriptor(DescriptorProblem),
}

/// Cold, immutable factory for one exact Wasm simulation package.
///
/// Construction parses the descriptor and computes byte commitments but does
/// not compile or instantiate the Component. Wasmtime work therefore remains
/// behind deterministic provider selection.
#[derive(Debug, Clone)]
pub struct WasmSimulationComponentFactory {
    descriptor: SimulationProviderDescriptor,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
    manifest_sha256: Sha256Digest,
    component_sha256: Sha256Digest,
    policy: ControlHostPolicy,
}

impl WasmSimulationComponentFactory {
    pub fn new(
        manifest_bytes: Vec<u8>,
        component_bytes: Vec<u8>,
        supported_solvers: Vec<SolverKind>,
        policy: ControlHostPolicy,
    ) -> Result<Self, WasmSimulationFactoryError> {
        if component_bytes.is_empty() {
            return Err(WasmSimulationFactoryError::EmptyComponent);
        }

        let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes)?;
        manifest.validate().map_err(|problems| {
            WasmSimulationFactoryError::ManifestInvalid(format!("{problems:?}"))
        })?;
        if manifest.kind != ExtensionKind::Simulation {
            return Err(WasmSimulationFactoryError::NotSimulationExtension);
        }
        if manifest.runtime != RuntimeKind::Wasm {
            return Err(WasmSimulationFactoryError::RuntimeNotWasm);
        }

        let descriptor = SimulationProviderDescriptor {
            manifest,
            backend_name: WASM_SIMULATION_BACKEND_V1.into(),
            supported_solvers,
        };
        descriptor
            .validate()
            .map_err(WasmSimulationFactoryError::Descriptor)?;

        Ok(Self {
            descriptor,
            manifest_sha256: digest(&manifest_bytes),
            component_sha256: digest(&component_bytes),
            manifest_bytes: Arc::from(manifest_bytes),
            component_bytes: Arc::from(component_bytes),
            policy,
        })
    }

    pub fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    pub const fn manifest_digest(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    pub const fn component_digest(&self) -> Sha256Digest {
        self.component_sha256
    }
}

impl SimulationBackendFactory for WasmSimulationComponentFactory {
    fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    fn manifest_sha256(&self) -> Option<Sha256Digest> {
        Some(self.manifest_sha256)
    }

    fn executable_payload_sha256(&self) -> Option<Sha256Digest> {
        Some(self.component_sha256)
    }

    fn create(
        &self,
        _permit: &SelectedExecutionPermit,
    ) -> Result<Box<dyn SimulationBackend>, SimulationError> {
        Ok(Box::new(WasmSimulationComponentBackend {
            manifest_bytes: Arc::clone(&self.manifest_bytes),
            component_bytes: Arc::clone(&self.component_bytes),
            manifest_sha256: self.manifest_sha256,
            component_sha256: self.component_sha256,
            expected_extension_id: self.descriptor.manifest.id.as_str().to_owned(),
            expected_extension_version: self.descriptor.manifest.version.clone(),
            supported_solvers: self.descriptor.supported_solvers.clone(),
            policy: self.policy,
        }))
    }
}

#[derive(Debug)]
struct WasmSimulationComponentBackend {
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
    manifest_sha256: Sha256Digest,
    component_sha256: Sha256Digest,
    expected_extension_id: String,
    expected_extension_version: String,
    supported_solvers: Vec<SolverKind>,
    policy: ControlHostPolicy,
}

impl SimulationBackend for WasmSimulationComponentBackend {
    fn name(&self) -> &'static str {
        WASM_SIMULATION_BACKEND_V1
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &self.supported_solvers
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        let request_sha256 = canonical_request_sha256_v1(request).map_err(|error| {
            SimulationError::Adapter(format!("failed to canonicalize simulation request: {error}"))
        })?;

        let host = SimulationComponentHost::new(self.policy);
        let invocation = host
            .invoke(&self.manifest_bytes, &self.component_bytes, request)
            .map_err(|error| map_host_error(error, request.solver))?;

        if invocation.manifest_sha256() != self.manifest_sha256.0 {
            return Err(SimulationError::Adapter(
                "runtime manifest digest drifted from the factory-bound bytes".into(),
            ));
        }
        if invocation.component_sha256() != self.component_sha256.0 {
            return Err(SimulationError::Adapter(
                "runtime Component digest drifted from the factory-bound bytes".into(),
            ));
        }
        if invocation.extension_id() != self.expected_extension_id
            || invocation.extension_version() != self.expected_extension_version
        {
            return Err(SimulationError::Adapter(
                "runtime extension identity drifted from the routed descriptor".into(),
            ));
        }
        if invocation.result().evidence != SimulationEvidence::default() {
            return Err(SimulationError::Adapter(
                "technical simulation host unexpectedly minted result evidence".into(),
            ));
        }

        // Hash the provider-owned result before host evidence is attached. In
        // particular, the WIT warnings list is provider-owned and must remain
        // byte-for-byte semantically unchanged by evidence construction.
        let output_sha256 = canonical_output_sha256_v1(invocation.result()).map_err(|error| {
            SimulationError::Adapter(format!("failed to canonicalize simulation output: {error}"))
        })?;

        let evidence = ExtensionComponentEvidence {
            extension_id: invocation.extension_id().to_owned(),
            extension_version: invocation.extension_version().to_owned(),
            manifest_sha256: hex_digest(invocation.manifest_sha256()),
            component_sha256: hex_digest(invocation.component_sha256()),
            runtime_profile: format!(
                "control={};simulation={}",
                invocation.control_wasm_profile(),
                invocation.simulation_wasm_profile()
            ),
            digest_profile: SIMULATION_DIGEST_PROFILE_V1.into(),
            request_sha256: hex_digest(request_sha256),
            output_sha256: hex_digest(output_sha256),
            wit_version: invocation.wit_version().into(),
            adapter_version: format!(
                "runtime={};evidence={}",
                invocation.adapter_version(),
                WASM_SIMULATION_EVIDENCE_ADAPTER_V1
            ),
        };

        let result = attach_extension_evidence(invocation.into_result(), evidence)?;
        if result.is_engineering_evidence() {
            return Err(SimulationError::Adapter(
                "extension Component execution must never self-promote to engineering evidence"
                    .into(),
            ));
        }
        Ok(result)
    }
}

fn attach_extension_evidence(
    mut result: SimulationResult,
    evidence: ExtensionComponentEvidence,
) -> Result<SimulationResult, SimulationError> {
    if result.evidence != SimulationEvidence::default() {
        return Err(SimulationError::Adapter(
            "extension evidence can only be attached to an evidence-empty technical result".into(),
        ));
    }
    result.evidence = SimulationEvidence {
        mode: ExecutionMode::ExtensionComponent,
        backend: Some(WASM_SIMULATION_BACKEND_V1.into()),
        extension: Some(evidence),
        ..SimulationEvidence::default()
    };
    Ok(result)
}

fn map_host_error(error: SimulationHostError, solver: SolverKind) -> SimulationError {
    match error {
        SimulationHostError::InvalidRequest(message)
        | SimulationHostError::Guest(GuestSimulationFailure::InvalidRequest(message)) => {
            SimulationError::InvalidRequest(message)
        }
        SimulationHostError::Guest(GuestSimulationFailure::UnsupportedSolver) => {
            SimulationError::SolverUnavailable(solver)
        }
        other => SimulationError::Adapter(format!("extension Component host failed: {other}")),
    }
}

fn digest(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::new(Sha256::digest(bytes).into())
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{SimulationMetric, UncertaintyEstimate};

    const HELLO_MANIFEST: &[u8] = include_bytes!(
        "../../../../examples/extensions/hello-simulation/manifest.json"
    );

    #[test]
    fn factory_derives_descriptor_and_commitments_from_exact_bytes() {
        let component = b"synthetic-component-bytes".to_vec();
        let factory = WasmSimulationComponentFactory::new(
            HELLO_MANIFEST.to_vec(),
            component.clone(),
            vec![SolverKind::Custom],
            ControlHostPolicy::default(),
        )
        .unwrap();

        assert_eq!(
            factory.descriptor().manifest.id.as_str(),
            "org.example.hello-simulation"
        );
        assert_eq!(factory.descriptor().manifest.runtime, RuntimeKind::Wasm);
        assert_eq!(factory.manifest_digest(), digest(HELLO_MANIFEST));
        assert_eq!(factory.component_digest(), digest(&component));
        assert_eq!(factory.descriptor().backend_name, WASM_SIMULATION_BACKEND_V1);
    }

    #[test]
    fn factory_rejects_solver_not_declared_by_exact_manifest() {
        let error = WasmSimulationComponentFactory::new(
            HELLO_MANIFEST.to_vec(),
            b"component".to_vec(),
            vec![SolverKind::Circuit],
            ControlHostPolicy::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            WasmSimulationFactoryError::Descriptor(DescriptorProblem::MissingSolverCapability(
                SolverKind::Circuit
            ))
        ));
    }

    #[test]
    fn evidence_attachment_does_not_mutate_provider_warnings() {
        let result = SimulationResult {
            request_id: "fixture".into(),
            converged: true,
            confidence: 0.0,
            uncertainty: UncertaintyEstimate::new(1.0, 0.0),
            metrics: vec![SimulationMetric {
                name: "fixture.metric".into(),
                value: 1.0,
                unit: "1".into(),
                uncertainty: None,
            }],
            warnings: vec!["provider-owned warning".into()],
            evidence: SimulationEvidence::default(),
        };
        let original_warnings = result.warnings.clone();
        let evidence = ExtensionComponentEvidence {
            extension_id: "org.example.fixture".into(),
            extension_version: "1.0.0".into(),
            manifest_sha256: "a".repeat(64),
            component_sha256: "b".repeat(64),
            runtime_profile: "runtime-v1".into(),
            digest_profile: SIMULATION_DIGEST_PROFILE_V1.into(),
            request_sha256: "c".repeat(64),
            output_sha256: "d".repeat(64),
            wit_version: "simulation-provider-v1".into(),
            adapter_version: WASM_SIMULATION_EVIDENCE_ADAPTER_V1.into(),
        };

        let result = attach_extension_evidence(result, evidence).unwrap();
        assert_eq!(result.warnings, original_warnings);
        assert_eq!(result.evidence.mode, ExecutionMode::ExtensionComponent);
        assert!(!result.is_engineering_evidence());
    }
}
