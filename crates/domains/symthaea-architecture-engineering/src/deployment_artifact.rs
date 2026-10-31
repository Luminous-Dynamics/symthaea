//! Exact deployment-pipeline, build-output and measurement-binding receipts for hardware-aware NAS.
//!
//! The artifact boundary is the compiled model + firmware pair, not an architecture label.
//! This module validates content commitments and cross-layer identity. It does not authenticate
//! a builder, prove that a board was flashed, or independently attest physical measurements.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hardware_objectives::{
    compute_hardware_profile_sha256, compute_objective_registry_sha256,
    score_objective_observation, validate_hardware_profile, validate_objective_registry,
    HardwareProfileManifest, ObjectiveObservation, ObjectiveRegistry, ScoredMeasurement,
};

const PIPELINE_SCHEMA: &str = "symthaea.nas.deployment-pipeline";
const BUILD_SCHEMA: &str = "symthaea.nas.deployment-build-receipt";
const MEASUREMENT_SCHEMA: &str = "symthaea.nas.deployment-measurement-receipt";
const VERSION: &str = "0.1.0";
const PIPELINE_DOMAIN: &[u8] = b"symthaea.nas.deployment-pipeline.v0.1";
const BUILD_DOMAIN: &[u8] = b"symthaea.nas.deployment-build-receipt.v0.1";
const ARTIFACT_DOMAIN: &[u8] = b"symthaea.nas.deployable-artifact-identity.v0.1";
const MEASUREMENT_DOMAIN: &[u8] = b"symthaea.nas.deployment-measurement-binding.v0.1";
const SHA256_HEX_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineComponentKind {
    ArchitectureGraph,
    QuantizationSpec,
    CalibrationDataset,
    CompilerToolchain,
    CompilerFlags,
    OperatorLibrary,
    MemoryPlanner,
    Firmware,
    Runtime,
    BoardSupportPackage,
    LinkerScript,
    BuildEnvironment,
}

impl PipelineComponentKind {
    const REQUIRED: [Self; 12] = [
        Self::ArchitectureGraph,
        Self::QuantizationSpec,
        Self::CalibrationDataset,
        Self::CompilerToolchain,
        Self::CompilerFlags,
        Self::OperatorLibrary,
        Self::MemoryPlanner,
        Self::Firmware,
        Self::Runtime,
        Self::BoardSupportPackage,
        Self::LinkerScript,
        Self::BuildEnvironment,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineComponent {
    pub kind: PipelineComponentKind,
    pub version: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentPipelineManifest {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub objective_registry_sha256: String,
    /// Exactly one of each required component, sorted by PipelineComponentKind.
    pub components: Vec<PipelineComponent>,
    pub pipeline_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildDisposition {
    Built,
    Failed,
    NotAttempted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentBuildReceipt {
    pub schema: String,
    pub version: String,
    pub pipeline_sha256: String,
    pub disposition: BuildDisposition,
    pub build_log_sha256: Option<String>,
    pub model_binary_sha256: Option<String>,
    pub firmware_binary_sha256: Option<String>,
    pub linker_map_sha256: Option<String>,
    pub flash_report_sha256: Option<String>,
    pub ram_report_sha256: Option<String>,
    pub reason_code: Option<String>,
    pub build_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentArtifactError {
    pub field: String,
    pub message: String,
}

impl DeploymentArtifactError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for DeploymentArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for DeploymentArtifactError {}

pub fn compute_deployment_pipeline_sha256(
    manifest: &DeploymentPipelineManifest,
) -> Result<String, DeploymentArtifactError> {
    validate_pipeline_fields(manifest)?;
    let mut canonical = manifest.clone();
    canonical.pipeline_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentArtifactError::new("manifest", error.to_string()))?;
    Ok(digest_bytes(PIPELINE_DOMAIN, &bytes))
}

pub fn validate_deployment_pipeline_manifest(
    manifest: &DeploymentPipelineManifest,
) -> Result<(), DeploymentArtifactError> {
    validate_pipeline_fields(manifest)?;
    if !is_sha256(&manifest.pipeline_sha256)
        || compute_deployment_pipeline_sha256(manifest)? != manifest.pipeline_sha256
    {
        return Err(DeploymentArtifactError::new(
            "pipeline_sha256",
            "does not match the canonical deployment pipeline",
        ));
    }
    Ok(())
}

/// Cross-checks the pipeline against the exact hardware profile and objective registry.
/// The four profile-owned components must exactly match the profile's pinned identities.
pub fn validate_deployment_pipeline_against(
    manifest: &DeploymentPipelineManifest,
    profile: &HardwareProfileManifest,
    registry: &ObjectiveRegistry,
) -> Result<(), DeploymentArtifactError> {
    validate_deployment_pipeline_manifest(manifest)?;
    validate_hardware_profile(profile)
        .map_err(|error| DeploymentArtifactError::new(
            format!("hardware_profile.{}", error.field), error.message,
        ))?;
    validate_objective_registry(registry)
        .map_err(|error| DeploymentArtifactError::new(
            format!("objective_registry.{}", error.field), error.message,
        ))?;

    if manifest.target_id != profile.profile_id
        || manifest.hardware_profile_sha256 != profile.profile_sha256
        || compute_hardware_profile_sha256(profile)
            .map_err(|error| DeploymentArtifactError::new("hardware_profile", error.to_string()))?
            != manifest.hardware_profile_sha256
    {
        return Err(DeploymentArtifactError::new(
            "target_identity",
            "pipeline target/profile does not match the supplied hardware profile",
        ));
    }
    if manifest.objective_registry_sha256 != registry.registry_sha256
        || compute_objective_registry_sha256(registry)
            .map_err(|error| DeploymentArtifactError::new("objective_registry", error.to_string()))?
            != manifest.objective_registry_sha256
    {
        return Err(DeploymentArtifactError::new(
            "objective_registry_sha256",
            "pipeline is bound to a different or invalid objective registry",
        ));
    }
    if !registry.hardware_profiles.iter().any(|candidate| {
        candidate.profile_id == profile.profile_id
            && candidate.profile_sha256 == profile.profile_sha256
    }) {
        return Err(DeploymentArtifactError::new(
            "objective_registry.hardware_profiles",
            "the exact target profile is absent from the objective registry",
        ));
    }

    for (kind, expected) in [
        (PipelineComponentKind::CompilerToolchain, profile.compiler_toolchain_sha256.as_str()),
        (PipelineComponentKind::QuantizationSpec, profile.quantization_spec_sha256.as_str()),
        (PipelineComponentKind::Firmware, profile.firmware_sha256.as_str()),
        (PipelineComponentKind::Runtime, profile.runtime_sha256.as_str()),
    ] {
        let actual = component_digest(manifest, kind)?;
        if actual != expected {
            return Err(DeploymentArtifactError::new(
                format!("components.{kind:?}"),
                "component digest does not match the exact hardware profile",
            ));
        }
    }
    Ok(())
}

fn validate_pipeline_fields(
    manifest: &DeploymentPipelineManifest,
) -> Result<(), DeploymentArtifactError> {
    if manifest.schema != PIPELINE_SCHEMA || manifest.version != VERSION {
        return Err(DeploymentArtifactError::new(
            "schema_version",
            "unsupported deployment pipeline schema/version",
        ));
    }
    for (field, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("candidate_id", manifest.candidate_id.as_str()),
        ("target_id", manifest.target_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentArtifactError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("candidate_config_sha256", manifest.candidate_config_sha256.as_str()),
        ("hardware_profile_sha256", manifest.hardware_profile_sha256.as_str()),
        ("objective_registry_sha256", manifest.objective_registry_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    if manifest.components.len() != PipelineComponentKind::REQUIRED.len() {
        return Err(DeploymentArtifactError::new(
            "components",
            format!(
                "expected exactly {} required components, found {}",
                PipelineComponentKind::REQUIRED.len(),
                manifest.components.len()
            ),
        ));
    }

    let mut seen = HashSet::with_capacity(manifest.components.len());
    let mut previous = None;
    for (index, component) in manifest.components.iter().enumerate() {
        if !seen.insert(component.kind) {
            return Err(DeploymentArtifactError::new(
                format!("components[{index}].kind"),
                "duplicate pipeline component",
            ));
        }
        if previous.is_some_and(|prior| prior >= component.kind) {
            return Err(DeploymentArtifactError::new(
                format!("components[{index}].kind"),
                "components must be strictly sorted by kind",
            ));
        }
        previous = Some(component.kind);
        if component.version.trim().is_empty() {
            return Err(DeploymentArtifactError::new(
                format!("components[{index}].version"),
                "must be non-empty",
            ));
        }
        require_sha256(&format!("components[{index}].sha256"), &component.sha256)?;
    }
    for required in PipelineComponentKind::REQUIRED {
        if !seen.contains(&required) {
            return Err(DeploymentArtifactError::new(
                "components",
                format!("missing required component {required:?}"),
            ));
        }
    }
    Ok(())
}

fn component_digest(
    manifest: &DeploymentPipelineManifest,
    kind: PipelineComponentKind,
) -> Result<&str, DeploymentArtifactError> {
    manifest.components.iter()
        .find(|component| component.kind == kind)
        .map(|component| component.sha256.as_str())
        .ok_or_else(|| DeploymentArtifactError::new(
            "components", format!("missing component {kind:?}"),
        ))
}

pub fn compute_deployment_build_receipt_sha256(
    receipt: &DeploymentBuildReceipt,
) -> Result<String, DeploymentArtifactError> {
    validate_build_receipt_fields(receipt)?;
    let mut canonical = receipt.clone();
    canonical.build_receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentArtifactError::new("build_receipt", error.to_string()))?;
    Ok(digest_bytes(BUILD_DOMAIN, &bytes))
}

pub fn validate_deployment_build_receipt(
    receipt: &DeploymentBuildReceipt,
    expected_pipeline_sha256: &str,
) -> Result<(), DeploymentArtifactError> {
    validate_build_receipt_fields(receipt)?;
    require_sha256("expected_pipeline_sha256", expected_pipeline_sha256)?;
    if receipt.pipeline_sha256 != expected_pipeline_sha256 {
        return Err(DeploymentArtifactError::new(
            "pipeline_sha256",
            "build receipt belongs to a different deployment pipeline",
        ));
    }
    if !is_sha256(&receipt.build_receipt_sha256)
        || compute_deployment_build_receipt_sha256(receipt)? != receipt.build_receipt_sha256
    {
        return Err(DeploymentArtifactError::new(
            "build_receipt_sha256",
            "does not match the canonical build receipt",
        ));
    }
    Ok(())
}

fn validate_build_receipt_fields(
    receipt: &DeploymentBuildReceipt,
) -> Result<(), DeploymentArtifactError> {
    if receipt.schema != BUILD_SCHEMA || receipt.version != VERSION {
        return Err(DeploymentArtifactError::new(
            "schema_version",
            "unsupported deployment build receipt schema/version",
        ));
    }
    require_sha256("pipeline_sha256", &receipt.pipeline_sha256)?;

    let digests = [
        ("build_log_sha256", receipt.build_log_sha256.as_deref()),
        ("model_binary_sha256", receipt.model_binary_sha256.as_deref()),
        ("firmware_binary_sha256", receipt.firmware_binary_sha256.as_deref()),
        ("linker_map_sha256", receipt.linker_map_sha256.as_deref()),
        ("flash_report_sha256", receipt.flash_report_sha256.as_deref()),
        ("ram_report_sha256", receipt.ram_report_sha256.as_deref()),
    ];
    match receipt.disposition {
        BuildDisposition::Built => {
            for (field, digest) in digests {
                require_sha256(field, digest.ok_or_else(|| {
                    DeploymentArtifactError::new(field, "successful builds require this digest")
                })?)?;
            }
            if receipt.reason_code.is_some() {
                return Err(DeploymentArtifactError::new(
                    "reason_code",
                    "successful builds cannot carry a failure reason",
                ));
            }
        }
        BuildDisposition::Failed => {
            require_sha256(
                "build_log_sha256",
                receipt.build_log_sha256.as_deref().ok_or_else(|| {
                    DeploymentArtifactError::new("build_log_sha256", "failed builds require a log")
                })?,
            )?;
            if receipt.reason_code.as_deref().is_none_or(|value| value.trim().is_empty()) {
                return Err(DeploymentArtifactError::new(
                    "reason_code",
                    "failed builds require a non-empty reason code",
                ));
            }
            if receipt.model_binary_sha256.is_some()
                || receipt.firmware_binary_sha256.is_some()
                || receipt.linker_map_sha256.is_some()
                || receipt.flash_report_sha256.is_some()
                || receipt.ram_report_sha256.is_some()
            {
                return Err(DeploymentArtifactError::new(
                    "build_outputs",
                    "failed builds cannot claim successful output artifacts",
                ));
            }
        }
        BuildDisposition::NotAttempted => {
            if receipt.reason_code.as_deref().is_none_or(|value| value.trim().is_empty()) {
                return Err(DeploymentArtifactError::new(
                    "reason_code",
                    "not-attempted builds require a non-empty reason code",
                ));
            }
            if digests.iter().any(|(_, digest)| digest.is_some()) {
                return Err(DeploymentArtifactError::new(
                    "build_outputs",
                    "not-attempted builds cannot carry build or output digests",
                ));
            }
        }
    }
    Ok(())
}

/// An in-process capability returned only after all pipeline/profile/registry/build checks pass.
/// It is intentionally not serializable or constructible outside this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedDeployableArtifact {
    candidate_id: String,
    candidate_config_sha256: String,
    target_id: String,
    hardware_profile_sha256: String,
    objective_registry_sha256: String,
    pipeline_sha256: String,
    build_receipt_sha256: String,
    model_binary_sha256: String,
    firmware_binary_sha256: String,
    artifact_sha256: String,
}

impl VerifiedDeployableArtifact {
    pub fn candidate_id(&self) -> &str { &self.candidate_id }
    pub fn candidate_config_sha256(&self) -> &str { &self.candidate_config_sha256 }
    pub fn target_id(&self) -> &str { &self.target_id }
    pub fn hardware_profile_sha256(&self) -> &str { &self.hardware_profile_sha256 }
    pub fn objective_registry_sha256(&self) -> &str { &self.objective_registry_sha256 }
    pub fn pipeline_sha256(&self) -> &str { &self.pipeline_sha256 }
    pub fn build_receipt_sha256(&self) -> &str { &self.build_receipt_sha256 }
    pub fn model_binary_sha256(&self) -> &str { &self.model_binary_sha256 }
    pub fn firmware_binary_sha256(&self) -> &str { &self.firmware_binary_sha256 }
    pub fn artifact_sha256(&self) -> &str { &self.artifact_sha256 }
}

#[derive(Serialize)]
struct ArtifactDigestMaterial<'a> {
    schema: &'static str,
    version: &'static str,
    candidate_id: &'a str,
    candidate_config_sha256: &'a str,
    target_id: &'a str,
    hardware_profile_sha256: &'a str,
    objective_registry_sha256: &'a str,
    pipeline_sha256: &'a str,
    build_receipt_sha256: &'a str,
    model_binary_sha256: &'a str,
    firmware_binary_sha256: &'a str,
}

/// Derives the deployable identity only from a successful, fully evidenced build.
pub fn verify_deployable_artifact(
    manifest: &DeploymentPipelineManifest,
    profile: &HardwareProfileManifest,
    registry: &ObjectiveRegistry,
    build: &DeploymentBuildReceipt,
) -> Result<VerifiedDeployableArtifact, DeploymentArtifactError> {
    validate_deployment_pipeline_against(manifest, profile, registry)?;
    validate_deployment_build_receipt(build, &manifest.pipeline_sha256)?;
    if build.disposition != BuildDisposition::Built {
        return Err(DeploymentArtifactError::new(
            "build.disposition",
            "only a successful build can produce a deployable artifact identity",
        ));
    }

    let model_binary_sha256 = build.model_binary_sha256.as_deref().expect("validated built receipt");
    let firmware_binary_sha256 = build.firmware_binary_sha256.as_deref().expect("validated built receipt");
    let material = ArtifactDigestMaterial {
        schema: "symthaea.nas.deployable-artifact",
        version: VERSION,
        candidate_id: &manifest.candidate_id,
        candidate_config_sha256: &manifest.candidate_config_sha256,
        target_id: &manifest.target_id,
        hardware_profile_sha256: &manifest.hardware_profile_sha256,
        objective_registry_sha256: &manifest.objective_registry_sha256,
        pipeline_sha256: &manifest.pipeline_sha256,
        build_receipt_sha256: &build.build_receipt_sha256,
        model_binary_sha256,
        firmware_binary_sha256,
    };
    let bytes = serde_json::to_vec(&material)
        .map_err(|error| DeploymentArtifactError::new("artifact_identity", error.to_string()))?;
    let artifact_sha256 = digest_bytes(ARTIFACT_DOMAIN, &bytes);

    Ok(VerifiedDeployableArtifact {
        candidate_id: manifest.candidate_id.clone(),
        candidate_config_sha256: manifest.candidate_config_sha256.clone(),
        target_id: manifest.target_id.clone(),
        hardware_profile_sha256: manifest.hardware_profile_sha256.clone(),
        objective_registry_sha256: manifest.objective_registry_sha256.clone(),
        pipeline_sha256: manifest.pipeline_sha256.clone(),
        build_receipt_sha256: build.build_receipt_sha256.clone(),
        model_binary_sha256: model_binary_sha256.to_owned(),
        firmware_binary_sha256: firmware_binary_sha256.to_owned(),
        artifact_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentMeasurementReceipt {
    pub schema: String,
    pub version: String,
    pub artifact_sha256: String,
    pub pipeline_sha256: String,
    pub build_receipt_sha256: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub objective_registry_sha256: String,
    pub objective_id: String,
    pub observation: ObjectiveObservation,
    pub binding_sha256: String,
}

pub fn compute_deployment_measurement_receipt_sha256(
    receipt: &DeploymentMeasurementReceipt,
) -> Result<String, DeploymentArtifactError> {
    validate_measurement_receipt_fields(receipt)?;
    let mut canonical = receipt.clone();
    canonical.binding_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentArtifactError::new("measurement_receipt", error.to_string()))?;
    Ok(digest_bytes(MEASUREMENT_DOMAIN, &bytes))
}

/// Creates a digest-bound link between the exact built artifact and a typed objective observation.
/// The observation's source artifact remains separately pinned.
pub fn create_deployment_measurement_receipt(
    artifact: &VerifiedDeployableArtifact,
    registry: &ObjectiveRegistry,
    observation: ObjectiveObservation,
) -> Result<DeploymentMeasurementReceipt, DeploymentArtifactError> {
    score_objective_observation(registry, &observation)
        .map_err(|error| DeploymentArtifactError::new("observation", error.to_string()))?;
    validate_observation_artifact_scope(artifact, registry, &observation)?;

    let mut receipt = DeploymentMeasurementReceipt {
        schema: MEASUREMENT_SCHEMA.into(),
        version: VERSION.into(),
        artifact_sha256: artifact.artifact_sha256.clone(),
        pipeline_sha256: artifact.pipeline_sha256.clone(),
        build_receipt_sha256: artifact.build_receipt_sha256.clone(),
        candidate_id: artifact.candidate_id.clone(),
        candidate_config_sha256: artifact.candidate_config_sha256.clone(),
        target_id: artifact.target_id.clone(),
        hardware_profile_sha256: artifact.hardware_profile_sha256.clone(),
        objective_registry_sha256: artifact.objective_registry_sha256.clone(),
        objective_id: observation.objective_id.clone(),
        observation,
        binding_sha256: String::new(),
    };
    receipt.binding_sha256 = compute_deployment_measurement_receipt_sha256(&receipt)?;
    Ok(receipt)
}

/// Recomputes both the objective score and binding digest; rehashing a forged receipt cannot
/// hide a candidate, configuration, profile, registry or artifact substitution.
pub fn verify_deployment_measurement_receipt(
    artifact: &VerifiedDeployableArtifact,
    registry: &ObjectiveRegistry,
    receipt: &DeploymentMeasurementReceipt,
) -> Result<ScoredMeasurement, DeploymentArtifactError> {
    validate_measurement_receipt_fields(receipt)?;
    if receipt.artifact_sha256 != artifact.artifact_sha256
        || receipt.pipeline_sha256 != artifact.pipeline_sha256
        || receipt.build_receipt_sha256 != artifact.build_receipt_sha256
        || receipt.candidate_id != artifact.candidate_id
        || receipt.candidate_config_sha256 != artifact.candidate_config_sha256
        || receipt.target_id != artifact.target_id
        || receipt.hardware_profile_sha256 != artifact.hardware_profile_sha256
        || receipt.objective_registry_sha256 != artifact.objective_registry_sha256
        || receipt.objective_id != receipt.observation.objective_id
    {
        return Err(DeploymentArtifactError::new(
            "measurement_scope",
            "receipt scope does not match the verified deployable artifact",
        ));
    }
    if registry.registry_sha256 != artifact.objective_registry_sha256 {
        return Err(DeploymentArtifactError::new(
            "objective_registry_sha256",
            "verifier supplied a different objective registry",
        ));
    }
    if !is_sha256(&receipt.binding_sha256)
        || compute_deployment_measurement_receipt_sha256(receipt)? != receipt.binding_sha256
    {
        return Err(DeploymentArtifactError::new(
            "binding_sha256",
            "does not match the canonical deployment measurement receipt",
        ));
    }
    validate_observation_artifact_scope(artifact, registry, &receipt.observation)?;
    score_objective_observation(registry, &receipt.observation)
        .map_err(|error| DeploymentArtifactError::new("observation", error.to_string()))
}

fn validate_observation_artifact_scope(
    artifact: &VerifiedDeployableArtifact,
    registry: &ObjectiveRegistry,
    observation: &ObjectiveObservation,
) -> Result<(), DeploymentArtifactError> {
    if observation.candidate_id != artifact.candidate_id
        || observation.candidate_config_sha256 != artifact.candidate_config_sha256
        || observation.registry_sha256 != artifact.objective_registry_sha256
        || observation.hardware_profile_sha256.as_deref()
            != Some(artifact.hardware_profile_sha256.as_str())
        || registry.registry_sha256 != artifact.objective_registry_sha256
    {
        return Err(DeploymentArtifactError::new(
            "observation_scope",
            "objective observation is not bound to this candidate, configuration, target profile and registry",
        ));
    }
    Ok(())
}

fn validate_measurement_receipt_fields(
    receipt: &DeploymentMeasurementReceipt,
) -> Result<(), DeploymentArtifactError> {
    if receipt.schema != MEASUREMENT_SCHEMA || receipt.version != VERSION {
        return Err(DeploymentArtifactError::new(
            "schema_version",
            "unsupported deployment measurement receipt schema/version",
        ));
    }
    for (field, value) in [
        ("artifact_sha256", receipt.artifact_sha256.as_str()),
        ("pipeline_sha256", receipt.pipeline_sha256.as_str()),
        ("build_receipt_sha256", receipt.build_receipt_sha256.as_str()),
        ("candidate_config_sha256", receipt.candidate_config_sha256.as_str()),
        ("hardware_profile_sha256", receipt.hardware_profile_sha256.as_str()),
        ("objective_registry_sha256", receipt.objective_registry_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    for (field, value) in [
        ("candidate_id", receipt.candidate_id.as_str()),
        ("target_id", receipt.target_id.as_str()),
        ("objective_id", receipt.objective_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentArtifactError::new(field, "must be non-empty"));
        }
    }
    Ok(())
}

fn require_sha256(field: &str, value: &str) -> Result<(), DeploymentArtifactError> {
    if !is_sha256(value) {
        return Err(DeploymentArtifactError::new(
            field,
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_objectives::{
        compute_hardware_profile_sha256, compute_objective_registry_sha256,
        HardwareProfileManifest, MeasurementCost, MeasurementOutcome, ObjectiveDefinition,
        ObjectiveDirection, ObjectiveNormalization, ObjectiveSource,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture() -> (
        DeploymentPipelineManifest,
        HardwareProfileManifest,
        ObjectiveRegistry,
        DeploymentBuildReceipt,
    ) {
        let mut profile = HardwareProfileManifest {
            profile_id: "nrf5340-rev-a".into(),
            device_model: "nRF5340".into(),
            board_id: "board-001".into(),
            board_revision: "rev-a".into(),
            firmware_sha256: sha('b'),
            runtime_sha256: sha('d'),
            compiler_toolchain_sha256: sha('c'),
            quantization_spec_sha256: sha('e'),
            measurement_protocol_sha256: sha('a'),
            instrument_model: "PPK2".into(),
            instrument_id: "meter-001".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = compute_hardware_profile_sha256(&profile).unwrap();

        let mut registry = ObjectiveRegistry {
            schema: "symthaea.nas.objective-registry".into(),
            version: "0.1.0".into(),
            benchmark_id: "tinyml-fixture-v1".into(),
            objectives: vec![ObjectiveDefinition {
                objective_id: "latency_ms".into(),
                value_unit: "ms".into(),
                direction: ObjectiveDirection::Minimize,
                source: ObjectiveSource::PhysicalDeviceMeasurement,
                normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 100.0 },
                measurement_protocol_sha256: profile.measurement_protocol_sha256.clone(),
                hardware_profile_sha256: Some(profile.profile_sha256.clone()),
                measurement_cost_unit: "measurement-queries".into(),
                measurement_cost_model_sha256: sha('f'),
            }],
            hardware_profiles: vec![profile.clone()],
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = compute_objective_registry_sha256(&registry).unwrap();

        let components = vec![
            (PipelineComponentKind::ArchitectureGraph, sha('1')),
            (PipelineComponentKind::QuantizationSpec, profile.quantization_spec_sha256.clone()),
            (PipelineComponentKind::CalibrationDataset, sha('2')),
            (PipelineComponentKind::CompilerToolchain, profile.compiler_toolchain_sha256.clone()),
            (PipelineComponentKind::CompilerFlags, sha('3')),
            (PipelineComponentKind::OperatorLibrary, sha('4')),
            (PipelineComponentKind::MemoryPlanner, sha('5')),
            (PipelineComponentKind::Firmware, profile.firmware_sha256.clone()),
            (PipelineComponentKind::Runtime, profile.runtime_sha256.clone()),
            (PipelineComponentKind::BoardSupportPackage, sha('6')),
            (PipelineComponentKind::LinkerScript, sha('7')),
            (PipelineComponentKind::BuildEnvironment, sha('8')),
        ];
        let mut manifest = DeploymentPipelineManifest {
            schema: PIPELINE_SCHEMA.into(),
            version: VERSION.into(),
            benchmark_id: "tinyml-fixture-v1".into(),
            candidate_id: "candidate-001".into(),
            candidate_config_sha256: sha('9'),
            target_id: profile.profile_id.clone(),
            hardware_profile_sha256: profile.profile_sha256.clone(),
            objective_registry_sha256: registry.registry_sha256.clone(),
            components: components.into_iter().map(|(kind, sha256)| PipelineComponent {
                kind, version: "fixture-v1".into(), sha256,
            }).collect(),
            pipeline_sha256: String::new(),
        };
        manifest.pipeline_sha256 = compute_deployment_pipeline_sha256(&manifest).unwrap();

        let mut build = DeploymentBuildReceipt {
            schema: BUILD_SCHEMA.into(),
            version: VERSION.into(),
            pipeline_sha256: manifest.pipeline_sha256.clone(),
            disposition: BuildDisposition::Built,
            build_log_sha256: Some(sha('a')),
            model_binary_sha256: Some(sha('b')),
            firmware_binary_sha256: Some(sha('c')),
            linker_map_sha256: Some(sha('d')),
            flash_report_sha256: Some(sha('e')),
            ram_report_sha256: Some(sha('f')),
            reason_code: None,
            build_receipt_sha256: String::new(),
        };
        build.build_receipt_sha256 = compute_deployment_build_receipt_sha256(&build).unwrap();
        (manifest, profile, registry, build)
    }

    fn measured_observation(
        manifest: &DeploymentPipelineManifest,
        registry: &ObjectiveRegistry,
        profile: &HardwareProfileManifest,
    ) -> ObjectiveObservation {
        ObjectiveObservation {
            candidate_id: manifest.candidate_id.clone(),
            candidate_config_sha256: manifest.candidate_config_sha256.clone(),
            objective_id: "latency_ms".into(),
            registry_sha256: registry.registry_sha256.clone(),
            measurement_protocol_sha256: profile.measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(profile.profile_sha256.clone()),
            outcome: MeasurementOutcome::Measured {
                raw_value_bits: 42.0_f64.to_bits(),
                replicate_id: "replicate-01".into(),
                sample_count: 10,
                source_artifact_sha256: sha('7'),
                cost: MeasurementCost {
                    resource_unit: "measurement-queries".into(),
                    units: 10,
                },
            },
        }
    }

    #[test]
    fn exact_pipeline_and_successful_build_produce_stable_artifact_identity() {
        let (manifest, profile, registry, build) = fixture();
        let first = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        let second = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.model_binary_sha256(), build.model_binary_sha256.as_deref().unwrap());
        assert_eq!(first.firmware_binary_sha256(), build.firmware_binary_sha256.as_deref().unwrap());
    }

    #[test]
    fn missing_required_pipeline_component_is_rejected() {
        let (mut manifest, _, _, _) = fixture();
        manifest.components.retain(|component| component.kind != PipelineComponentKind::BoardSupportPackage);
        assert!(compute_deployment_pipeline_sha256(&manifest).unwrap_err().to_string().contains("expected exactly"));
    }

    #[test]
    fn duplicate_or_unsorted_components_are_rejected() {
        let (mut manifest, _, _, _) = fixture();
        manifest.components[1].kind = PipelineComponentKind::ArchitectureGraph;
        assert!(compute_deployment_pipeline_sha256(&manifest).is_err());
    }

    #[test]
    fn component_digest_must_match_profile_owned_toolchain() {
        let (mut manifest, profile, registry, _) = fixture();
        let compiler = manifest.components.iter_mut()
            .find(|component| component.kind == PipelineComponentKind::CompilerToolchain).unwrap();
        compiler.sha256 = sha('0');
        manifest.pipeline_sha256 = compute_deployment_pipeline_sha256(&manifest).unwrap();
        let error = validate_deployment_pipeline_against(&manifest, &profile, &registry).unwrap_err();
        assert!(error.to_string().contains("does not match the exact hardware profile"));
    }

    #[test]
    fn target_and_registry_substitution_are_rejected() {
        let (mut manifest, profile, registry, _) = fixture();
        manifest.target_id = "different-board".into();
        manifest.pipeline_sha256 = compute_deployment_pipeline_sha256(&manifest).unwrap();
        assert!(validate_deployment_pipeline_against(&manifest, &profile, &registry).is_err());

        let (manifest, profile, registry, _) = fixture();
        let mut wrong_registry = registry.clone();
        wrong_registry.benchmark_id = "other-benchmark".into();
        wrong_registry.registry_sha256 = compute_objective_registry_sha256(&wrong_registry).unwrap();
        assert!(validate_deployment_pipeline_against(&manifest, &profile, &wrong_registry).is_err());
    }

    #[test]
    fn failed_and_unattempted_builds_never_produce_artifact_capabilities() {
        let (manifest, profile, registry, mut build) = fixture();
        build.disposition = BuildDisposition::Failed;
        build.model_binary_sha256 = None;
        build.firmware_binary_sha256 = None;
        build.linker_map_sha256 = None;
        build.flash_report_sha256 = None;
        build.ram_report_sha256 = None;
        build.reason_code = Some("compiler_error".into());
        build.build_receipt_sha256 = compute_deployment_build_receipt_sha256(&build).unwrap();
        assert!(verify_deployable_artifact(&manifest, &profile, &registry, &build).is_err());

        build.disposition = BuildDisposition::NotAttempted;
        build.build_log_sha256 = None;
        build.reason_code = Some("not_scheduled".into());
        build.build_receipt_sha256 = compute_deployment_build_receipt_sha256(&build).unwrap();
        assert!(verify_deployable_artifact(&manifest, &profile, &registry, &build).is_err());
    }

    #[test]
    fn altered_binary_digest_invalidates_build_receipt() {
        let (manifest, profile, registry, mut build) = fixture();
        build.model_binary_sha256 = Some(sha('0'));
        assert!(verify_deployable_artifact(&manifest, &profile, &registry, &build).is_err());
    }

    #[test]
    fn successful_build_requires_every_output_and_resource_report_digest() {
        let (_, _, _, mut build) = fixture();
        build.ram_report_sha256 = None;
        assert!(compute_deployment_build_receipt_sha256(&build).is_err());
    }

    #[test]
    fn measurement_receipt_binds_objective_to_exact_deployable_binary_pair() {
        let (manifest, profile, registry, build) = fixture();
        let artifact = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        let observation = measured_observation(&manifest, &registry, &profile);
        let receipt = create_deployment_measurement_receipt(&artifact, &registry, observation).unwrap();
        let scored = verify_deployment_measurement_receipt(&artifact, &registry, &receipt).unwrap();
        assert!(matches!(scored, ScoredMeasurement::Measured { raw_value: 42.0, .. }));
    }

    #[test]
    fn candidate_configuration_substitution_is_rejected_even_with_rehashed_receipt() {
        let (manifest, profile, registry, build) = fixture();
        let artifact = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        let mut receipt = create_deployment_measurement_receipt(
            &artifact, &registry, measured_observation(&manifest, &registry, &profile),
        ).unwrap();
        receipt.candidate_config_sha256 = sha('0');
        receipt.binding_sha256 = compute_deployment_measurement_receipt_sha256(&receipt).unwrap();
        assert!(verify_deployment_measurement_receipt(&artifact, &registry, &receipt).is_err());
    }

    #[test]
    fn observation_from_another_target_is_rejected() {
        let (manifest, profile, registry, build) = fixture();
        let artifact = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        let mut observation = measured_observation(&manifest, &registry, &profile);
        observation.hardware_profile_sha256 = Some(sha('0'));
        assert!(create_deployment_measurement_receipt(&artifact, &registry, observation).is_err());
    }

    #[test]
    fn measurement_receipt_rejects_unknown_fields_on_json_round_trip() {
        let (manifest, profile, registry, build) = fixture();
        let artifact = verify_deployable_artifact(&manifest, &profile, &registry, &build).unwrap();
        let receipt = create_deployment_measurement_receipt(
            &artifact, &registry, measured_observation(&manifest, &registry, &profile),
        ).unwrap();
        let encoded = serde_json::to_string(&receipt).unwrap();
        let decoded: DeploymentMeasurementReceipt = serde_json::from_str(&encoded).unwrap();
        assert_eq!(receipt, decoded);
        let injected = encoded.replacen(
            "\"binding_sha256\"",
            "\"unrecognized_field\":true,\"binding_sha256\"",
            1,
        );
        assert!(serde_json::from_str::<DeploymentMeasurementReceipt>(&injected).is_err());
    }

    #[test]
    fn missing_or_wrong_build_pipeline_binding_is_rejected() {
        let (manifest, profile, registry, mut build) = fixture();
        build.pipeline_sha256 = sha('0');
        build.build_receipt_sha256 = compute_deployment_build_receipt_sha256(&build).unwrap();
        assert!(verify_deployable_artifact(&manifest, &profile, &registry, &build).is_err());
    }
}
