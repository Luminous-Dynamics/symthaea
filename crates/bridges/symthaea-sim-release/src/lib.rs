// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Host-minted release receipts for simulation results.
//!
//! A routed simulation result and a routing decision are useful facts, but a
//! loose pair can be accidentally or maliciously re-paired after execution.
//! This crate adds a narrow release boundary that binds:
//!
//! - the selected admission identity and immutable package commitments,
//! - the exact selected extension version,
//! - the canonical request and provider-owned output,
//! - the complete normalized `SimulationEvidence`, and
//! - the core selected routing decision.
//!
//! Release finalization is deliberately independent of where the technical
//! execution occurred. [`run_released`] keeps the original in-process routing
//! path. [`release_routed_deployment`] accepts an unforgeable safe-Rust
//! `RoutedDeploymentInvocation`, attaches host-owned execution lineage from its
//! independently verified worker/deployment facts, and then applies the same
//! private receipt theorem.
//!
//! A receipt is **audit evidence, not durable authority**. Policy/trust state may
//! change immediately after issuance and must be checked again before any later
//! privileged use.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem,
};
use symthaea_extension_core::{ExtensionId, ExtensionManifest, RuntimeKind};
use symthaea_extension_router::{RoutingConstraints, RoutingDecision};
use symthaea_sim_bridge::{
    ExecutionMode, ExtensionComponentEvidence, SimulationEvidence, SimulationRequest,
    SimulationResult,
};
use symthaea_sim_deployment::{BoundSimulationDeployment, SimulationDeploymentError};
use symthaea_sim_deployment_routing::RoutedDeploymentInvocation;
use symthaea_sim_digest::{
    CanonicalDigestError, SIMULATION_DIGEST_PROFILE_V1, canonical_output_sha256_v1,
    canonical_request_sha256_v1,
};
use symthaea_sim_extension_routing::{
    LazySimulationError, LazySimulationRegistry, solver_capability,
};
use symthaea_sim_worker::{SUPERVISOR_PROFILE_V1, WORKER_PROTOCOL_V1};
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::SEALED_WORKER_IMAGE_PROFILE_V1;
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
    WORKER_QUALIFICATION_PROFILE_V1,
};
use thiserror::Error;

/// Versioned profile for the release receipt encoding and issuance semantics.
pub const SIMULATION_RELEASE_PROFILE_V1: &str = "symthaea.simulation.release.v1";
/// Stable backend identity for host-promoted contained Component execution.
pub const CONTAINED_DEPLOYMENT_BACKEND_V1: &str =
    "extension-component-contained-deployment-v1";
/// Stable adapter identity for contained-deployment execution lineage promotion.
pub const CONTAINED_DEPLOYMENT_RELEASE_ADAPTER_V1: &str =
    "symthaea-sim-deployment-release-v1";

const RELEASE_DOMAIN_V1: &[u8] = b"symthaea.simulation.release.v1\0";
const EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.simulation.evidence.v1\0";

/// Persistable audit representation of a release receipt.
///
/// Deserializing this type never recreates [`ReleasedSimulation`] and never
/// recreates authority. It is evidence about a completed host decision only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationReleaseEvidence {
    pub profile: String,
    pub selected_extension: String,
    pub selected_version: String,
    pub capability: String,
    pub runtime: RuntimeKind,
    pub admission_generation: u64,
    pub trust_generation: u64,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub policy_sha256: String,
    pub request_sha256: String,
    pub output_sha256: String,
    pub evidence_sha256: String,
    pub release_sha256: String,
}

/// In-process receipt minted at the final release boundary.
///
/// Fields are private and this type is not deserializable. Possessing persisted
/// receipt evidence cannot manufacture a fresh release receipt.
#[derive(Debug)]
pub struct SimulationReleaseReceipt {
    selected_extension: String,
    selected_version: String,
    capability: String,
    runtime: RuntimeKind,
    admission_generation: u64,
    trust_generation: u64,
    manifest_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    request_sha256: [u8; 32],
    output_sha256: [u8; 32],
    evidence_sha256: [u8; 32],
    release_sha256: [u8; 32],
}

impl SimulationReleaseReceipt {
    pub fn selected_extension(&self) -> &str {
        &self.selected_extension
    }

    pub fn selected_version(&self) -> &str {
        &self.selected_version
    }

    pub fn capability(&self) -> &str {
        &self.capability
    }

    pub const fn runtime(&self) -> RuntimeKind {
        self.runtime
    }

    pub const fn admission_generation(&self) -> u64 {
        self.admission_generation
    }

    pub const fn trust_generation(&self) -> u64 {
        self.trust_generation
    }

    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    pub const fn payload_sha256(&self) -> [u8; 32] {
        self.payload_sha256
    }

    pub const fn policy_sha256(&self) -> [u8; 32] {
        self.policy_sha256
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub const fn output_sha256(&self) -> [u8; 32] {
        self.output_sha256
    }

    pub const fn evidence_sha256(&self) -> [u8; 32] {
        self.evidence_sha256
    }

    pub const fn release_sha256(&self) -> [u8; 32] {
        self.release_sha256
    }

    pub fn evidence(&self) -> SimulationReleaseEvidence {
        SimulationReleaseEvidence {
            profile: SIMULATION_RELEASE_PROFILE_V1.into(),
            selected_extension: self.selected_extension.clone(),
            selected_version: self.selected_version.clone(),
            capability: self.capability.clone(),
            runtime: self.runtime,
            admission_generation: self.admission_generation,
            trust_generation: self.trust_generation,
            manifest_sha256: hex_digest(self.manifest_sha256),
            payload_sha256: hex_digest(self.payload_sha256),
            policy_sha256: hex_digest(self.policy_sha256),
            request_sha256: hex_digest(self.request_sha256),
            output_sha256: hex_digest(self.output_sha256),
            evidence_sha256: hex_digest(self.evidence_sha256),
            release_sha256: hex_digest(self.release_sha256),
        }
    }
}

/// A simulation result sealed together with the exact decision and receipt that
/// authorized its release.
///
/// No mutable result/decision accessors are exposed. Callers can inspect the
/// completed release or persist [`SimulationReleaseEvidence`], but cannot mutate
/// the sealed pair in place.
#[derive(Debug)]
pub struct ReleasedSimulation {
    result: SimulationResult,
    decision: RoutingDecision,
    receipt: SimulationReleaseReceipt,
}

impl ReleasedSimulation {
    pub fn result(&self) -> &SimulationResult {
        &self.result
    }

    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub fn receipt(&self) -> &SimulationReleaseReceipt {
        &self.receipt
    }

    pub fn evidence(&self) -> SimulationReleaseEvidence {
        self.receipt.evidence()
    }

    /// Recompute every commitment from the sealed result, caller-supplied
    /// request, and embedded decision. This proves internal consistency of the
    /// receipt; it does not query current authority again.
    pub fn verify(&self, request: &SimulationRequest) -> Result<(), SimulationReleaseError> {
        verify_release(request, &self.result, &self.decision, &self.receipt)
    }
}

#[derive(Debug, Error)]
pub enum SimulationReleaseError {
    #[error("simulation routing/execution failed: {0:?}")]
    Execution(LazySimulationError),
    #[error(transparent)]
    Canonical(#[from] CanonicalDigestError),
    #[error(transparent)]
    Deployment(#[from] SimulationDeploymentError),
    #[error("selected provider disappeared from the registry: {0:?}")]
    SelectedProviderMissing(ExtensionId),
    #[error("selected manifest identity does not match the routing decision")]
    SelectedManifestMismatch,
    #[error("selected release admission is missing or duplicated")]
    SelectedAdmissionMissingOrDuplicate,
    #[error("selected release admission does not match the routed manifest/decision")]
    SelectedAdmissionMismatch,
    #[error("release-finalization admission currentness failed: {0:?}")]
    ReleaseCurrentness(AdmissionProblem),
    #[error("release-finalization worker qualification currentness failed: {0}")]
    ReleaseWorkerCurrentness(WorkerQualificationError),
    #[error("routed invocation does not match the exact release deployment authority")]
    ReleaseDeploymentMismatch,
    #[error("routing decision capability does not match the request solver")]
    CapabilityMismatch,
    #[error("released result request id does not match the request")]
    RequestIdMismatch,
    #[error("routed invocation request digest does not match canonical request")]
    RoutedRequestDigestMismatch,
    #[error("contained worker attempted to mint simulation evidence")]
    WorkerMintedEvidence,
    #[error("contained worker technical lineage does not match routed authority: {0}")]
    WorkerLineageMismatch(&'static str),
    #[error("wasm execution did not return ExtensionComponent evidence")]
    MissingWasmExecutionEvidence,
    #[error("non-wasm provider returned ExtensionComponent evidence")]
    UnexpectedComponentEvidence,
    #[error("data-only extension cannot produce a simulation release")]
    DataOnlyRuntime,
    #[error("released extension Component attempted to self-promote to engineering evidence")]
    ComponentPromotedToEngineeringEvidence,
    #[error("extension execution lineage does not match selected admission/routing: {0}")]
    ExtensionLineageMismatch(&'static str),
    #[error("release request commitment does not match")]
    RequestDigestMismatch,
    #[error("release output commitment does not match")]
    OutputDigestMismatch,
    #[error("release evidence commitment does not match")]
    EvidenceDigestMismatch,
    #[error("release routing/admission commitment does not match: {0}")]
    DecisionReceiptMismatch(&'static str),
    #[error("release receipt digest does not match")]
    ReleaseDigestMismatch,
    #[error("string length cannot be represented by release profile v1")]
    LengthOverflow,
}

impl From<LazySimulationError> for SimulationReleaseError {
    fn from(value: LazySimulationError) -> Self {
        Self::Execution(value)
    }
}

/// Route, execute, verify lineage, recheck the exact selected admission once
/// more, and mint a sealed release receipt.
///
/// `LazySimulationRegistry::run` already performs currentness checks before
/// construction, immediately before execution, and after execution. This
/// function preserves that path and delegates the common final release theorem
/// to the private finalizer.
pub fn run_released(
    registry: &LazySimulationRegistry,
    request: &SimulationRequest,
    constraints: RoutingConstraints,
    admissions: &[ActiveAdmission],
    currentness: &dyn AdmissionCurrentnessSource,
) -> Result<ReleasedSimulation, SimulationReleaseError> {
    let (result, decision) = registry.run(request, constraints, admissions, currentness)?;
    let manifest = registry
        .catalog()
        .get(&decision.selected)
        .ok_or_else(|| SimulationReleaseError::SelectedProviderMissing(decision.selected.clone()))?;
    finalize_preexecuted_release(
        request,
        result,
        decision,
        manifest,
        admissions,
        currentness,
        || Ok(()),
    )
}

/// Attach host-owned execution lineage to a completed, deterministically routed
/// contained deployment and mint the same release receipt as [`run_released`].
///
/// `RoutedDeploymentInvocation` has private fields and no public constructor or
/// deserializer. Safe Rust can obtain one only through the route -> exact active
/// admission -> exact deployment -> exact active worker qualification execution
/// path. Release additionally requires the exact deployment and live worker
/// qualification so both authorization domains can be rechecked immediately
/// before receipt minting.
#[allow(clippy::too_many_arguments)]
pub fn release_routed_deployment(
    request: &SimulationRequest,
    routed: &RoutedDeploymentInvocation,
    deployment: &BoundSimulationDeployment,
    worker_qualification: &ActiveWorkerQualification,
    worker_currentness: &dyn WorkerQualificationCurrentnessSource,
    manifest: &ExtensionManifest,
    admissions: &[ActiveAdmission],
    currentness: &dyn AdmissionCurrentnessSource,
) -> Result<ReleasedSimulation, SimulationReleaseError> {
    let request_sha256 = canonical_request_sha256_v1(request)?;
    if routed.request_sha256() != request_sha256 {
        return Err(SimulationReleaseError::RoutedRequestDigestMismatch);
    }

    let decision = routed.decision();
    let bound = routed.invocation();
    if bound.deployment_sha256() != deployment.deployment_sha256() {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    let selected_admission = exact_selected_admission(decision, manifest, admissions)?;
    deployment.verify(selected_admission, worker_qualification)?;

    let sealed = bound.invocation();
    let supervised = sealed.invocation();
    let worker = supervised.worker();
    let technical = supervised.result();
    if worker_qualification.worker_sha256() != supervised.worker_sha256()
        || worker_qualification.qualification_evidence_sha256()
            != bound.worker_qualification_evidence_sha256()
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    if technical.evidence != SimulationEvidence::default() {
        return Err(SimulationReleaseError::WorkerMintedEvidence);
    }

    let output_sha256 = canonical_output_sha256_v1(technical)?;
    let expected_manifest = hex_digest(decision.selected_manifest_sha256.0);
    let expected_payload = hex_digest(decision.selected_payload_sha256.0);
    let expected_request = hex_digest(request_sha256);
    let expected_output = hex_digest(output_sha256);

    require_worker_equal("extension id", &worker.extension_id, decision.selected.as_str())?;
    require_worker_equal("extension version", &worker.extension_version, &manifest.version)?;
    require_worker_equal("manifest digest", &worker.manifest_sha256, &expected_manifest)?;
    require_worker_equal("component digest", &worker.component_sha256, &expected_payload)?;
    require_worker_equal("request digest", &worker.request_sha256, &expected_request)?;
    require_worker_equal("output digest", &worker.output_sha256, &expected_output)?;

    let deployment_sha256 = hex_digest(bound.deployment_sha256());
    let qualification_evidence_sha256 =
        hex_digest(bound.worker_qualification_evidence_sha256());
    let warnings_before = technical.warnings.clone();
    let mut result = technical.clone();

    let runtime_profile = format!(
        "control={};simulation={};supervisor={};worker={};image={};process={};filesystem={};deployment={}",
        worker.control_wasm_profile,
        worker.simulation_wasm_profile,
        SUPERVISOR_PROFILE_V1,
        WORKER_PROTOCOL_V1,
        SEALED_WORKER_IMAGE_PROFILE_V1,
        WORKER_CONTAINMENT_PROFILE_V1,
        WORKER_FILESYSTEM_PROFILE_V1,
        deployment_sha256,
    );
    let adapter_version = format!(
        "{};{};worker-qualification={};qualification-evidence={}",
        worker.adapter_version,
        CONTAINED_DEPLOYMENT_RELEASE_ADAPTER_V1,
        WORKER_QUALIFICATION_PROFILE_V1,
        qualification_evidence_sha256,
    );

    result.evidence = SimulationEvidence {
        mode: ExecutionMode::ExtensionComponent,
        backend: Some(CONTAINED_DEPLOYMENT_BACKEND_V1.into()),
        extension: Some(ExtensionComponentEvidence {
            extension_id: worker.extension_id.clone(),
            extension_version: worker.extension_version.clone(),
            manifest_sha256: worker.manifest_sha256.clone(),
            component_sha256: worker.component_sha256.clone(),
            runtime_profile,
            digest_profile: SIMULATION_DIGEST_PROFILE_V1.into(),
            request_sha256: worker.request_sha256.clone(),
            output_sha256: worker.output_sha256.clone(),
            wit_version: worker.wit_version.clone(),
            adapter_version,
        }),
        ..SimulationEvidence::default()
    };

    // Warnings are provider-owned and included in the canonical output digest.
    // This host boundary may attach execution evidence only.
    if result.warnings != warnings_before {
        return Err(SimulationReleaseError::WorkerLineageMismatch(
            "provider warnings mutated",
        ));
    }
    if result.is_engineering_evidence() {
        return Err(SimulationReleaseError::ComponentPromotedToEngineeringEvidence);
    }
    if canonical_output_sha256_v1(&result)? != output_sha256 {
        return Err(SimulationReleaseError::WorkerLineageMismatch(
            "output changed while attaching evidence",
        ));
    }

    finalize_preexecuted_release(
        request,
        result,
        decision.clone(),
        manifest,
        admissions,
        currentness,
        || {
            deployment.verify(selected_admission, worker_qualification)?;
            worker_qualification
                .recheck_currentness(worker_currentness)
                .map_err(SimulationReleaseError::ReleaseWorkerCurrentness)?;
            Ok(())
        },
    )
}

/// Private common final release theorem. Keeping this private is load-bearing:
/// public callers must arrive through either `LazySimulationRegistry::run` or an
/// unforgeable `RoutedDeploymentInvocation`, not arbitrary public structs.
fn finalize_preexecuted_release(
    request: &SimulationRequest,
    result: SimulationResult,
    decision: RoutingDecision,
    manifest: &ExtensionManifest,
    admissions: &[ActiveAdmission],
    currentness: &dyn AdmissionCurrentnessSource,
    before_mint: impl FnOnce() -> Result<(), SimulationReleaseError>,
) -> Result<ReleasedSimulation, SimulationReleaseError> {
    let request_sha256 = canonical_request_sha256_v1(request)?;
    let output_sha256 = canonical_output_sha256_v1(&result)?;

    if manifest.id != decision.selected {
        return Err(SimulationReleaseError::SelectedManifestMismatch);
    }
    if decision.capability != solver_capability(request.solver) {
        return Err(SimulationReleaseError::CapabilityMismatch);
    }
    if result.request_id != request.id {
        return Err(SimulationReleaseError::RequestIdMismatch);
    }

    validate_execution_lineage(
        manifest.runtime,
        &manifest.version,
        &result,
        &decision,
        request_sha256,
        output_sha256,
    )?;

    let evidence_sha256 = evidence_sha256_v1(&result.evidence)?;
    let selected_admission = exact_selected_admission(&decision, manifest, admissions)?;
    selected_admission
        .recheck_currentness(currentness)
        .map_err(SimulationReleaseError::ReleaseCurrentness)?;
    before_mint()?;

    let mut receipt = SimulationReleaseReceipt {
        selected_extension: decision.selected.as_str().to_owned(),
        selected_version: manifest.version.clone(),
        capability: decision.capability.as_str().to_owned(),
        runtime: manifest.runtime,
        admission_generation: decision.selected_admission_generation,
        trust_generation: decision.selected_trust_generation,
        manifest_sha256: decision.selected_manifest_sha256.0,
        payload_sha256: decision.selected_payload_sha256.0,
        policy_sha256: decision.selected_policy_sha256.0,
        request_sha256,
        output_sha256,
        evidence_sha256,
        release_sha256: [0; 32],
    };
    receipt.release_sha256 = release_sha256_v1(&receipt)?;

    let release = ReleasedSimulation {
        result,
        decision,
        receipt,
    };
    release.verify(request)?;
    Ok(release)
}

fn require_worker_equal(
    field: &'static str,
    actual: &str,
    expected: &str,
) -> Result<(), SimulationReleaseError> {
    if actual != expected {
        return Err(SimulationReleaseError::WorkerLineageMismatch(field));
    }
    Ok(())
}

fn exact_selected_admission<'a>(
    decision: &RoutingDecision,
    manifest: &ExtensionManifest,
    admissions: &'a [ActiveAdmission],
) -> Result<&'a ActiveAdmission, SimulationReleaseError> {
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or(SimulationReleaseError::SelectedAdmissionMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(SimulationReleaseError::SelectedAdmissionMissingOrDuplicate);
    }
    if !admission.matches_manifest(manifest)
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
    {
        return Err(SimulationReleaseError::SelectedAdmissionMismatch);
    }
    Ok(admission)
}

fn validate_execution_lineage(
    runtime: RuntimeKind,
    expected_version: &str,
    result: &SimulationResult,
    decision: &RoutingDecision,
    request_sha256: [u8; 32],
    output_sha256: [u8; 32],
) -> Result<(), SimulationReleaseError> {
    match runtime {
        RuntimeKind::Wasm => {
            if result.evidence.mode != ExecutionMode::ExtensionComponent {
                return Err(SimulationReleaseError::MissingWasmExecutionEvidence);
            }
            if result.is_engineering_evidence() {
                return Err(SimulationReleaseError::ComponentPromotedToEngineeringEvidence);
            }
            let extension = result
                .evidence
                .extension
                .as_ref()
                .ok_or(SimulationReleaseError::MissingWasmExecutionEvidence)?;
            let expected_manifest = hex_digest(decision.selected_manifest_sha256.0);
            let expected_payload = hex_digest(decision.selected_payload_sha256.0);
            let expected_request = hex_digest(request_sha256);
            let expected_output = hex_digest(output_sha256);

            if extension.extension_id != decision.selected.as_str() {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "extension id",
                ));
            }
            if extension.extension_version != expected_version {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "extension version",
                ));
            }
            if extension.manifest_sha256 != expected_manifest {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "manifest digest",
                ));
            }
            if extension.component_sha256 != expected_payload {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "component digest",
                ));
            }
            if extension.request_sha256 != expected_request {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "request digest",
                ));
            }
            if extension.output_sha256 != expected_output {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "output digest",
                ));
            }
            if extension.digest_profile != SIMULATION_DIGEST_PROFILE_V1 {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "digest profile",
                ));
            }
            if extension.runtime_profile.trim().is_empty()
                || extension.wit_version.trim().is_empty()
                || extension.adapter_version.trim().is_empty()
            {
                return Err(SimulationReleaseError::ExtensionLineageMismatch(
                    "incomplete execution provenance",
                ));
            }
        }
        RuntimeKind::Native | RuntimeKind::Remote => {
            if result.evidence.mode == ExecutionMode::ExtensionComponent
                || result.evidence.extension.is_some()
            {
                return Err(SimulationReleaseError::UnexpectedComponentEvidence);
            }
        }
        RuntimeKind::DataOnly => return Err(SimulationReleaseError::DataOnlyRuntime),
    }
    Ok(())
}

fn verify_release(
    request: &SimulationRequest,
    result: &SimulationResult,
    decision: &RoutingDecision,
    receipt: &SimulationReleaseReceipt,
) -> Result<(), SimulationReleaseError> {
    if decision.selected.as_str() != receipt.selected_extension {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "selected extension",
        ));
    }
    if receipt.selected_version.trim().is_empty() {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "selected version",
        ));
    }
    if decision.capability.as_str() != receipt.capability {
        return Err(SimulationReleaseError::DecisionReceiptMismatch("capability"));
    }
    if decision.selected_admission_generation != receipt.admission_generation {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "admission generation",
        ));
    }
    if decision.selected_trust_generation != receipt.trust_generation {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "trust generation",
        ));
    }
    if decision.selected_manifest_sha256.0 != receipt.manifest_sha256 {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "manifest digest",
        ));
    }
    if decision.selected_payload_sha256.0 != receipt.payload_sha256 {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "payload digest",
        ));
    }
    if decision.selected_policy_sha256.0 != receipt.policy_sha256 {
        return Err(SimulationReleaseError::DecisionReceiptMismatch(
            "policy digest",
        ));
    }
    if decision.capability != solver_capability(request.solver) {
        return Err(SimulationReleaseError::CapabilityMismatch);
    }
    if result.request_id != request.id {
        return Err(SimulationReleaseError::RequestIdMismatch);
    }

    let request_sha256 = canonical_request_sha256_v1(request)?;
    if request_sha256 != receipt.request_sha256 {
        return Err(SimulationReleaseError::RequestDigestMismatch);
    }
    let output_sha256 = canonical_output_sha256_v1(result)?;
    if output_sha256 != receipt.output_sha256 {
        return Err(SimulationReleaseError::OutputDigestMismatch);
    }
    validate_execution_lineage(
        receipt.runtime,
        &receipt.selected_version,
        result,
        decision,
        request_sha256,
        output_sha256,
    )?;

    let evidence_sha256 = evidence_sha256_v1(&result.evidence)?;
    if evidence_sha256 != receipt.evidence_sha256 {
        return Err(SimulationReleaseError::EvidenceDigestMismatch);
    }
    if release_sha256_v1(receipt)? != receipt.release_sha256 {
        return Err(SimulationReleaseError::ReleaseDigestMismatch);
    }
    Ok(())
}

fn evidence_sha256_v1(evidence: &SimulationEvidence) -> Result<[u8; 32], SimulationReleaseError> {
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    hasher.update([execution_mode_tag(evidence.mode)]);
    put_option_string(&mut hasher, evidence.backend.as_deref())?;
    put_option_string(&mut hasher, evidence.solver_version.as_deref())?;
    put_option_string(&mut hasher, evidence.input_digest.as_deref())?;
    put_option_string(&mut hasher, evidence.output_digest.as_deref())?;
    put_option_string(&mut hasher, evidence.parser_version.as_deref())?;
    match &evidence.extension {
        None => hasher.update([0]),
        Some(extension) => {
            hasher.update([1]);
            put_string(&mut hasher, &extension.extension_id)?;
            put_string(&mut hasher, &extension.extension_version)?;
            put_string(&mut hasher, &extension.manifest_sha256)?;
            put_string(&mut hasher, &extension.component_sha256)?;
            put_string(&mut hasher, &extension.runtime_profile)?;
            put_string(&mut hasher, &extension.digest_profile)?;
            put_string(&mut hasher, &extension.request_sha256)?;
            put_string(&mut hasher, &extension.output_sha256)?;
            put_string(&mut hasher, &extension.wit_version)?;
            put_string(&mut hasher, &extension.adapter_version)?;
        }
    }
    Ok(hasher.finalize().into())
}

fn release_sha256_v1(
    receipt: &SimulationReleaseReceipt,
) -> Result<[u8; 32], SimulationReleaseError> {
    let mut hasher = Sha256::new();
    hasher.update(RELEASE_DOMAIN_V1);
    put_string(&mut hasher, &receipt.selected_extension)?;
    put_string(&mut hasher, &receipt.selected_version)?;
    put_string(&mut hasher, &receipt.capability)?;
    hasher.update([runtime_tag(receipt.runtime)]);
    hasher.update(receipt.admission_generation.to_le_bytes());
    hasher.update(receipt.trust_generation.to_le_bytes());
    hasher.update(receipt.manifest_sha256);
    hasher.update(receipt.payload_sha256);
    hasher.update(receipt.policy_sha256);
    hasher.update(receipt.request_sha256);
    hasher.update(receipt.output_sha256);
    hasher.update(receipt.evidence_sha256);
    Ok(hasher.finalize().into())
}

fn put_option_string(
    hasher: &mut Sha256,
    value: Option<&str>,
) -> Result<(), SimulationReleaseError> {
    match value {
        None => hasher.update([0]),
        Some(value) => {
            hasher.update([1]);
            put_string(hasher, value)?;
        }
    }
    Ok(())
}

fn put_string(hasher: &mut Sha256, value: &str) -> Result<(), SimulationReleaseError> {
    let len = u64::try_from(value.len()).map_err(|_| SimulationReleaseError::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

const fn execution_mode_tag(mode: ExecutionMode) -> u8 {
    match mode {
        ExecutionMode::Unknown => 0,
        ExecutionMode::DryRun => 1,
        ExecutionMode::ExternalSolver => 2,
        ExecutionMode::ExtensionComponent => 3,
    }
}

const fn runtime_tag(runtime: RuntimeKind) -> u8 {
    match runtime {
        RuntimeKind::Native => 0,
        RuntimeKind::Wasm => 1,
        RuntimeKind::Remote => 2,
        RuntimeKind::DataOnly => 3,
    }
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
    use symthaea_extension_admission::Sha256Digest;
    use symthaea_sim_bridge::{ExtensionComponentEvidence, SimulationEvidence};

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::new([byte; 32])
    }

    #[test]
    fn evidence_digest_binds_every_extension_lineage_field() {
        let base = SimulationEvidence {
            mode: ExecutionMode::ExtensionComponent,
            backend: Some("extension-component-v1".into()),
            extension: Some(ExtensionComponentEvidence {
                extension_id: "org.example.fixture".into(),
                extension_version: "1.0.0".into(),
                manifest_sha256: "a".repeat(64),
                component_sha256: "b".repeat(64),
                runtime_profile: "runtime".into(),
                digest_profile: SIMULATION_DIGEST_PROFILE_V1.into(),
                request_sha256: "c".repeat(64),
                output_sha256: "d".repeat(64),
                wit_version: "wit".into(),
                adapter_version: "adapter".into(),
            }),
            ..SimulationEvidence::default()
        };
        let first = evidence_sha256_v1(&base).unwrap();
        let mut changed = base;
        changed
            .extension
            .as_mut()
            .unwrap()
            .adapter_version
            .push_str("-changed");
        let second = evidence_sha256_v1(&changed).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn release_digest_binds_version_policy_and_authority_generations() {
        let mut receipt = SimulationReleaseReceipt {
            selected_extension: "org.example.fixture".into(),
            selected_version: "1.0.0".into(),
            capability: "engineering.simulation.custom".into(),
            runtime: RuntimeKind::Wasm,
            admission_generation: 7,
            trust_generation: 11,
            manifest_sha256: digest(1).0,
            payload_sha256: digest(2).0,
            policy_sha256: digest(3).0,
            request_sha256: [4; 32],
            output_sha256: [5; 32],
            evidence_sha256: [6; 32],
            release_sha256: [0; 32],
        };
        let first = release_sha256_v1(&receipt).unwrap();
        receipt.selected_version.push_str("-changed");
        let version_changed = release_sha256_v1(&receipt).unwrap();
        assert_ne!(first, version_changed);
        receipt.selected_version = "1.0.0".into();
        receipt.trust_generation += 1;
        let generation_changed = release_sha256_v1(&receipt).unwrap();
        assert_ne!(first, generation_changed);
        receipt.trust_generation -= 1;
        receipt.policy_sha256[0] ^= 1;
        let policy_changed = release_sha256_v1(&receipt).unwrap();
        assert_ne!(first, policy_changed);
    }
}
