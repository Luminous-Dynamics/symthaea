//! Provider declarations and support claims generated from benchmark evidence.

use crate::benchmark::{BenchmarkReport, ReleaseGate};
use crate::{CapabilityLevel, Modality, content_hash};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const PROVIDER_MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProviderTask {
    LanguageIdentification,
    Asr,
    TextUnderstanding,
    SpeechTranslation,
    TextTranslation,
    TextGeneration,
    SpeechGeneration,
    Embedding,
    UnitDiscovery,
    StructureDiscovery,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseUse {
    ResearchOnly,
    NonCommercial,
    CommercialAllowed,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderManifest {
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub artifact_uri: String,
    pub artifact_hash: String,
    pub license_id: String,
    pub license_use: LicenseUse,
    pub local: bool,
    pub sovereign: bool,
    pub quantization: Option<String>,
    pub tasks: BTreeSet<ProviderTask>,
    pub modalities: Vec<Modality>,
    /// Keys are BCP-47 language tags; values are supported tasks.
    pub languages: BTreeMap<String, BTreeSet<ProviderTask>>,
    #[serde(default)]
    pub components: Vec<ArtifactComponent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactComponent {
    pub id: String,
    pub uri: String,
    pub revision: String,
    pub relative_path: String,
    pub artifact_hash: String,
    pub license_id: String,
}

impl ProviderManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PROVIDER_MANIFEST_VERSION {
            return Err("unsupported provider manifest version".into());
        }
        if self.id.trim().is_empty() || self.version.trim().is_empty() {
            return Err("provider id and version are required".into());
        }
        if self.artifact_hash.len() < 16 {
            return Err("a pinned artifact hash is required".into());
        }
        if self.sovereign && (!self.local || self.license_use == LicenseUse::Unknown) {
            return Err("sovereign providers must be local with a known license".into());
        }
        for (language, tasks) in &self.languages {
            if language.trim().is_empty() || tasks.is_empty() || !tasks.is_subset(&self.tasks) {
                return Err(format!("invalid language declaration: {language}"));
            }
        }
        if self.components.iter().any(|component| {
            component.id.is_empty()
                || component.uri.is_empty()
                || component.revision.is_empty()
                || component.relative_path.is_empty()
                || component.artifact_hash.len() < 16
                || component.license_id.is_empty()
        }) {
            return Err("every provider component must be revisioned, hashed, and licensed".into());
        }
        Ok(())
    }

    pub fn manifest_hash(&self) -> Result<String, serde_json::Error> {
        serde_json::to_vec(self).map(|bytes| content_hash(&bytes))
    }

    pub fn verify_artifact(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        crate::artifact::verify_path(path, &self.artifact_hash)
    }

    pub fn verify_components(&self, root: &Path) -> Result<(), String> {
        self.verify_artifact(root)?;
        for component in &self.components {
            crate::artifact::verify_path(
                &root.join(&component.relative_path),
                &component.artifact_hash,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SupportClaim {
    pub provider: String,
    pub scope: String,
    pub capability: CapabilityLevel,
    pub report_hash: String,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SupportRegistry {
    pub claims: Vec<SupportClaim>,
}

impl SupportRegistry {
    pub fn from_passing_report(
        report: &BenchmarkReport,
        gate: &ReleaseGate,
    ) -> Result<Self, Vec<crate::benchmark::GateFailure>> {
        gate.evaluate(report)?;
        let bytes = serde_json::to_vec(report).expect("serializing a report cannot fail");
        let report_hash = content_hash(&bytes);
        Ok(Self {
            claims: report
                .scopes
                .iter()
                .map(|scope| SupportClaim {
                    provider: report.provider.clone(),
                    scope: scope.scope.clone(),
                    capability: report.claimed_capability,
                    report_hash: report_hash.clone(),
                    metrics: scope
                        .metrics
                        .iter()
                        .map(|metric| (metric.name.clone(), metric.value))
                        .collect(),
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sovereign_requires_local_known_license() {
        let manifest = ProviderManifest {
            schema_version: PROVIDER_MANIFEST_VERSION,
            id: "p".into(),
            version: "1".into(),
            artifact_uri: "local:model".into(),
            artifact_hash: "0123456789abcdef".into(),
            license_id: "unknown".into(),
            license_use: LicenseUse::Unknown,
            local: true,
            sovereign: true,
            quantization: None,
            tasks: BTreeSet::new(),
            modalities: vec![],
            languages: BTreeMap::new(),
            components: vec![],
        };
        assert!(manifest.validate().is_err());
    }
}
