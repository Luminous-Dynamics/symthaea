// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deployment configuration binding for simulator, SIL, HIL, and hardware runs.
//!
//! A successful test is not meaningful if the runtime silently uses a different
//! calibration, scenario, hardware contract, or software revision. This module
//! binds those identities into one canonical manifest and reports every mismatch.
//! Its built-in FNV digest is only a deterministic identifier; authenticity must
//! be provided by an external cryptographic signature over the canonical bytes.

use serde::{Deserialize, Serialize};

use crate::claim_ledger::AssuranceLevel;
use crate::hardware_interface::HardwareBackendKind;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentArtifactDigests {
    pub calibration: String,
    pub scenario: String,
    pub qualification_campaign: String,
    pub hardware_contract: String,
    pub claim_ledger: String,
}

impl DeploymentArtifactDigests {
    fn validate(&self) -> bool {
        [
            self.calibration.as_str(),
            self.scenario.as_str(),
            self.qualification_campaign.as_str(),
            self.hardware_contract.as_str(),
            self.claim_ledger.as_str(),
        ]
        .iter()
        .all(|digest| valid_digest_identifier(digest))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleVersionBinding {
    pub module: String,
    pub version: String,
}

impl ModuleVersionBinding {
    fn validate(&self) -> bool {
        !self.module.trim().is_empty() && !self.version.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentAuthenticityReference {
    pub digest_algorithm: String,
    pub signature_scheme: String,
    pub key_id: String,
    pub signature_artifact_id: String,
}

impl DeploymentAuthenticityReference {
    fn validate(&self) -> bool {
        [
            self.digest_algorithm.as_str(),
            self.signature_scheme.as_str(),
            self.key_id.as_str(),
            self.signature_artifact_id.as_str(),
        ]
        .iter()
        .all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentManifest {
    pub schema_version: String,
    pub deployment_id: String,
    pub airframe_id: String,
    pub software_revision: String,
    pub backend_kind: HardwareBackendKind,
    pub maximum_claim_level: AssuranceLevel,
    pub artifact_digests: DeploymentArtifactDigests,
    /// Sorted feature names that must be enabled in the runtime image.
    pub required_features: Vec<String>,
    /// Sorted module/version bindings used by the deployment.
    pub module_versions: Vec<ModuleVersionBinding>,
    /// Required for a physical backend; verification remains external.
    pub authenticity: Option<DeploymentAuthenticityReference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentRuntimeIdentity {
    pub airframe_id: String,
    pub software_revision: String,
    pub backend_kind: HardwareBackendKind,
    pub artifact_digests: DeploymentArtifactDigests,
    pub enabled_features: Vec<String>,
    pub module_versions: Vec<ModuleVersionBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentBindingStatus {
    Bound,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentMismatch {
    AirframeId,
    SoftwareRevision,
    BackendKind,
    CalibrationDigest,
    ScenarioDigest,
    QualificationCampaignDigest,
    HardwareContractDigest,
    ClaimLedgerDigest,
    MissingFeature(String),
    ModuleVersion {
        module: String,
        expected: String,
        observed: Option<String>,
    },
    PhysicalBackendMissingAuthenticityReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentBindingReport {
    pub deployment_id: String,
    pub manifest_digest_fnv1a64: String,
    pub status: DeploymentBindingStatus,
    pub mismatches: Vec<DeploymentMismatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentManifestError {
    InvalidManifest,
    DuplicateFeature,
    DuplicateModule,
    UnsortedBindings,
    SerializationFailed,
}

impl DeploymentManifest {
    pub fn validate(&self) -> Result<(), DeploymentManifestError> {
        if self.schema_version.trim().is_empty()
            || self.deployment_id.trim().is_empty()
            || self.airframe_id.trim().is_empty()
            || self.software_revision.trim().is_empty()
            || !self.artifact_digests.validate()
            || self
                .required_features
                .iter()
                .any(|feature| feature.trim().is_empty())
            || self
                .module_versions
                .iter()
                .any(|binding| !binding.validate())
            || self
                .authenticity
                .as_ref()
                .is_some_and(|value| !value.validate())
        {
            return Err(DeploymentManifestError::InvalidManifest);
        }
        if !strictly_sorted_unique(&self.required_features) {
            return Err(if has_duplicate(&self.required_features) {
                DeploymentManifestError::DuplicateFeature
            } else {
                DeploymentManifestError::UnsortedBindings
            });
        }
        let modules: Vec<_> = self
            .module_versions
            .iter()
            .map(|binding| binding.module.clone())
            .collect();
        if !strictly_sorted_unique(&modules) {
            return Err(if has_duplicate(&modules) {
                DeploymentManifestError::DuplicateModule
            } else {
                DeploymentManifestError::UnsortedBindings
            });
        }
        if self.backend_kind == HardwareBackendKind::PhysicalHardware && self.authenticity.is_none()
        {
            return Err(DeploymentManifestError::InvalidManifest);
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, DeploymentManifestError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| DeploymentManifestError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, DeploymentManifestError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }

    pub fn verify_runtime(
        &self,
        runtime: &DeploymentRuntimeIdentity,
    ) -> Result<DeploymentBindingReport, DeploymentManifestError> {
        self.validate()?;
        let mut mismatches = Vec::new();
        if runtime.airframe_id != self.airframe_id {
            mismatches.push(DeploymentMismatch::AirframeId);
        }
        if runtime.software_revision != self.software_revision {
            mismatches.push(DeploymentMismatch::SoftwareRevision);
        }
        if runtime.backend_kind != self.backend_kind {
            mismatches.push(DeploymentMismatch::BackendKind);
        }
        compare_artifact_digests(
            &self.artifact_digests,
            &runtime.artifact_digests,
            &mut mismatches,
        );

        for feature in &self.required_features {
            if !runtime.enabled_features.contains(feature) {
                mismatches.push(DeploymentMismatch::MissingFeature(feature.clone()));
            }
        }
        for expected in &self.module_versions {
            let observed = runtime
                .module_versions
                .iter()
                .find(|binding| binding.module == expected.module)
                .map(|binding| binding.version.clone());
            if observed.as_deref() != Some(expected.version.as_str()) {
                mismatches.push(DeploymentMismatch::ModuleVersion {
                    module: expected.module.clone(),
                    expected: expected.version.clone(),
                    observed,
                });
            }
        }
        if self.backend_kind == HardwareBackendKind::PhysicalHardware && self.authenticity.is_none()
        {
            mismatches.push(DeploymentMismatch::PhysicalBackendMissingAuthenticityReference);
        }

        Ok(DeploymentBindingReport {
            deployment_id: self.deployment_id.clone(),
            manifest_digest_fnv1a64: self.digest_fnv1a64()?,
            status: if mismatches.is_empty() {
                DeploymentBindingStatus::Bound
            } else {
                DeploymentBindingStatus::Rejected
            },
            mismatches,
        })
    }
}

fn compare_artifact_digests(
    expected: &DeploymentArtifactDigests,
    observed: &DeploymentArtifactDigests,
    mismatches: &mut Vec<DeploymentMismatch>,
) {
    if expected.calibration != observed.calibration {
        mismatches.push(DeploymentMismatch::CalibrationDigest);
    }
    if expected.scenario != observed.scenario {
        mismatches.push(DeploymentMismatch::ScenarioDigest);
    }
    if expected.qualification_campaign != observed.qualification_campaign {
        mismatches.push(DeploymentMismatch::QualificationCampaignDigest);
    }
    if expected.hardware_contract != observed.hardware_contract {
        mismatches.push(DeploymentMismatch::HardwareContractDigest);
    }
    if expected.claim_ledger != observed.claim_ledger {
        mismatches.push(DeploymentMismatch::ClaimLedgerDigest);
    }
}

fn valid_digest_identifier(value: &str) -> bool {
    let Some((algorithm, digest)) = value.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && digest.len() >= 8
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

fn strictly_sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|window| window[0] < window[1])
}

fn has_duplicate(values: &[String]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[index + 1..].contains(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifacts() -> DeploymentArtifactDigests {
        DeploymentArtifactDigests {
            calibration: "sha256:11111111".into(),
            scenario: "sha256:22222222".into(),
            qualification_campaign: "sha256:33333333".into(),
            hardware_contract: "sha256:44444444".into(),
            claim_ledger: "sha256:55555555".into(),
        }
    }

    fn manifest() -> DeploymentManifest {
        DeploymentManifest {
            schema_version: "symthaea-helicopter-deployment-v1".into(),
            deployment_id: "deployment-001".into(),
            airframe_id: "research-airframe-01".into(),
            software_revision: "abc123".into(),
            backend_kind: HardwareBackendKind::SimulationOnly,
            maximum_claim_level: AssuranceLevel::DeterministicSimulation,
            artifact_digests: artifacts(),
            required_features: vec!["evidence".into(), "navigation".into()],
            module_versions: vec![
                ModuleVersionBinding {
                    module: "controller".into(),
                    version: "1".into(),
                },
                ModuleVersionBinding {
                    module: "rotor".into(),
                    version: "2".into(),
                },
            ],
            authenticity: None,
        }
    }

    fn runtime() -> DeploymentRuntimeIdentity {
        let expected = manifest();
        DeploymentRuntimeIdentity {
            airframe_id: expected.airframe_id,
            software_revision: expected.software_revision,
            backend_kind: expected.backend_kind,
            artifact_digests: expected.artifact_digests,
            enabled_features: expected.required_features,
            module_versions: expected.module_versions,
        }
    }

    #[test]
    fn exact_runtime_binding_passes() {
        let report = manifest().verify_runtime(&runtime()).unwrap();
        assert_eq!(report.status, DeploymentBindingStatus::Bound);
        assert!(report.mismatches.is_empty());
        assert!(report.manifest_digest_fnv1a64.starts_with("fnv1a64:"));
    }

    #[test]
    fn changed_calibration_and_revision_are_rejected() {
        let mut runtime = runtime();
        runtime.software_revision = "different".into();
        runtime.artifact_digests.calibration = "sha256:aaaaaaaa".into();
        let report = manifest().verify_runtime(&runtime).unwrap();
        assert_eq!(report.status, DeploymentBindingStatus::Rejected);
        assert!(
            report
                .mismatches
                .contains(&DeploymentMismatch::SoftwareRevision)
        );
        assert!(
            report
                .mismatches
                .contains(&DeploymentMismatch::CalibrationDigest)
        );
    }

    #[test]
    fn missing_feature_and_module_version_are_reported() {
        let mut runtime = runtime();
        runtime.enabled_features.clear();
        runtime.module_versions[0].version = "wrong".into();
        let report = manifest().verify_runtime(&runtime).unwrap();
        assert!(
            report
                .mismatches
                .contains(&DeploymentMismatch::MissingFeature("evidence".into()))
        );
        assert!(report.mismatches.iter().any(|mismatch| matches!(
            mismatch,
            DeploymentMismatch::ModuleVersion { module, .. } if module == "controller"
        )));
    }

    #[test]
    fn physical_manifest_requires_external_authenticity_reference() {
        let mut physical = manifest();
        physical.backend_kind = HardwareBackendKind::PhysicalHardware;
        assert_eq!(
            physical.validate(),
            Err(DeploymentManifestError::InvalidManifest)
        );
    }

    #[test]
    fn unsorted_features_are_refused_for_canonical_identity() {
        let mut invalid = manifest();
        invalid.required_features = vec!["navigation".into(), "evidence".into()];
        assert_eq!(
            invalid.validate(),
            Err(DeploymentManifestError::UnsortedBindings)
        );
    }
}
