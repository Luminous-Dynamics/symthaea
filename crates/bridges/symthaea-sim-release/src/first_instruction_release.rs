// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical release adapter for routed first-instruction cgroup executions.
//!
//! This module lives inside `symthaea-sim-release` so it can reuse the private
//! final release theorem without exposing a generic preexecuted-result minting
//! API. The public entry point accepts only the private-field routed invocation
//! capability produced by the exact route -> first-instruction deployment path.

use super::*;
use symthaea_sim_first_instruction_deployment::BoundFirstInstructionDeployment;
use symthaea_sim_first_instruction_routing::RoutedFirstInstructionInvocation;

pub const FIRST_INSTRUCTION_DEPLOYMENT_BACKEND_V1: &str =
    "extension-component-first-instruction-cgroup-deployment-v1";
pub const FIRST_INSTRUCTION_DEPLOYMENT_RELEASE_ADAPTER_V1: &str =
    "symthaea-sim-first-instruction-release-v1";

#[allow(clippy::too_many_arguments)]
pub fn release_routed_first_instruction(
    request: &SimulationRequest,
    routed: &RoutedFirstInstructionInvocation,
    base_deployment: &BoundSimulationDeployment,
    deployment: &BoundFirstInstructionDeployment,
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
    let invocation = routed.invocation();
    if invocation.base_deployment_sha256() != base_deployment.deployment_sha256()
        || invocation.binding_sha256() != deployment.binding_sha256()
        || invocation.worker_qualification_evidence_sha256()
            != worker_qualification.qualification_evidence_sha256()
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    let selected_admission = exact_selected_admission(decision, manifest, admissions)?;
    base_deployment.verify(selected_admission, worker_qualification)?;
    deployment
        .verify(base_deployment, selected_admission, worker_qualification)
        .map_err(|_| SimulationReleaseError::ReleaseDeploymentMismatch)?;

    let qualification_scope = worker_qualification.evidence();
    let live_worker_sha = hex_digest(worker_qualification.worker_sha256());
    let qualification_evidence_sha256 =
        hex_digest(worker_qualification.qualification_evidence_sha256());
    if qualification_scope.profile != WORKER_QUALIFICATION_PROFILE_V1
        || qualification_scope.worker_sha256 != live_worker_sha
        || qualification_scope.qualification_evidence_sha256 != qualification_evidence_sha256
        || qualification_scope.generation != worker_qualification.generation()
        || qualification_scope.worker_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
        || qualification_scope.supervisor_profile != SUPERVISOR_PROFILE_V1
        || qualification_scope.worker_protocol != WORKER_PROTOCOL_V1
        || qualification_scope.process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
        || qualification_scope.filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    let execution = invocation.evidence();
    execution
        .verify()
        .map_err(|_| SimulationReleaseError::ReleaseDeploymentMismatch)?;
    if execution.deployment.binding_sha256 != hex_digest(deployment.binding_sha256())
        || execution.deployment.base_deployment.deployment_sha256
            != hex_digest(base_deployment.deployment_sha256())
        || execution.deployment.worker_qualification_evidence_sha256
            != qualification_evidence_sha256
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    let worker_invocation = invocation.worker_invocation();
    let worker_evidence = worker_invocation.evidence();
    let worker = worker_invocation.worker();
    let technical = invocation.result();

    if technical.evidence != SimulationEvidence::default() {
        return Err(SimulationReleaseError::WorkerMintedEvidence);
    }
    if worker_evidence.containment_profiles_established_by_this_layer
        || worker_evidence.expected_process_containment_profile
            != qualification_scope.process_containment_profile
        || worker_evidence.expected_filesystem_containment_profile
            != qualification_scope.filesystem_containment_profile
        || worker_evidence.worker_protocol != qualification_scope.worker_protocol
        || worker_evidence.legacy_preexec_rlimits_applied
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
    }

    if worker_evidence.launch.image_sha256 != live_worker_sha
        || worker_evidence.launch.proc_exe_sha256 != live_worker_sha
    {
        return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
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
    require_worker_equal(
        "first-instruction evidence manifest digest",
        &worker_evidence.manifest_sha256,
        &expected_manifest,
    )?;
    require_worker_equal(
        "first-instruction evidence component digest",
        &worker_evidence.component_sha256,
        &expected_payload,
    )?;
    require_worker_equal(
        "first-instruction evidence request digest",
        &worker_evidence.request_sha256,
        &expected_request,
    )?;
    require_worker_equal(
        "first-instruction evidence output digest",
        &worker_evidence.output_sha256,
        &expected_output,
    )?;

    let warnings_before = technical.warnings.clone();
    let mut result = technical.clone();
    let base_deployment_sha256 = hex_digest(base_deployment.deployment_sha256());
    let first_deployment_sha256 = hex_digest(deployment.binding_sha256());

    let runtime_profile = format!(
        "control={};simulation={};worker-protocol={};image={};supervisor={};process={};filesystem={};base-deployment={};first-deployment={};first-execution={};cgroup={};sealed-launch={}",
        worker.control_wasm_profile,
        worker.simulation_wasm_profile,
        qualification_scope.worker_protocol,
        qualification_scope.worker_image_profile,
        qualification_scope.supervisor_profile,
        qualification_scope.process_containment_profile,
        qualification_scope.filesystem_containment_profile,
        base_deployment_sha256,
        first_deployment_sha256,
        execution.execution_sha256,
        execution.cgroup.evidence_sha256,
        worker_evidence.launch.evidence_sha256,
    );
    let adapter_version = format!(
        "{};{};worker-qualification={};qualification-evidence={};qualification-generation={};first-worker-evidence={};legacy-preexec-rlimits=false",
        worker.adapter_version,
        FIRST_INSTRUCTION_DEPLOYMENT_RELEASE_ADAPTER_V1,
        qualification_scope.profile,
        qualification_evidence_sha256,
        qualification_scope.generation,
        worker_evidence.evidence_sha256,
    );

    result.evidence = SimulationEvidence {
        mode: ExecutionMode::ExtensionComponent,
        backend: Some(FIRST_INSTRUCTION_DEPLOYMENT_BACKEND_V1.into()),
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
            "output changed while attaching first-instruction evidence",
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
            base_deployment.verify(selected_admission, worker_qualification)?;
            deployment
                .verify(base_deployment, selected_admission, worker_qualification)
                .map_err(|_| SimulationReleaseError::ReleaseDeploymentMismatch)?;
            let final_scope = worker_qualification.evidence();
            if final_scope != qualification_scope
                || invocation.binding_sha256() != deployment.binding_sha256()
                || invocation.base_deployment_sha256() != base_deployment.deployment_sha256()
                || invocation.worker_qualification_evidence_sha256()
                    != worker_qualification.qualification_evidence_sha256()
            {
                return Err(SimulationReleaseError::ReleaseDeploymentMismatch);
            }
            worker_qualification
                .recheck_currentness(worker_currentness)
                .map_err(SimulationReleaseError::ReleaseWorkerCurrentness)?;
            Ok(())
        },
    )
}
