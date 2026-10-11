// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-describing integration manifests and capability declarations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const INTEGRATION_MANIFEST_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IntegrationId(pub String);

impl IntegrationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IntegrationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Evidence-based maturity ladder. A manifest may only claim a level backed
/// by the corresponding qualification artifacts; v0.1 does not enforce the
/// external artifact store, but preserves the contract in the type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MaturityLevel {
    E0Schema,
    E1FixtureParsing,
    E2LiveObservation,
    E3TopologyValidated,
    E4DiagnosisValidated,
    E5SimulationValidated,
    E6ControlledActionValidated,
    E7RollbackValidated,
    E8BoundedAutonomyQualified,
    E9AuditedProduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityClass {
    Observe,
    Discover,
    Simulate,
    Actuate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessMode {
    ReadOnly,
    Simulated,
    Mutating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskClass {
    ReadOnly,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDeclaration {
    /// Stable operation name, e.g. `observe.interface.counters`.
    pub name: String,
    pub class: CapabilityClass,
    pub access: AccessMode,
    pub risk: RiskClass,
    /// Whether the integration itself knows a mechanical rollback operation.
    pub reversible: bool,
    /// Mutation-capable operations MUST remain disabled by default.
    pub default_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialKind {
    ApiKey,
    OAuth2,
    BearerToken,
    Mtls,
    UsernamePassword,
    SshKey,
    CloudRole,
    WorkloadIdentity,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialRequirement {
    /// Logical credential name; never contains the secret itself.
    pub name: String,
    pub kind: CredentialKind,
    /// Human/machine-readable minimum scope description.
    pub scope: String,
    /// True when this credential can authorize state mutation upstream.
    pub mutation_capable: bool,
}

/// Static self-description of an integration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationManifest {
    pub schema_version: u16,
    pub id: IntegrationId,
    pub display_name: String,
    pub version: String,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protocols: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_kinds: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<CapabilityDeclaration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<CredentialRequirement>,
    pub maturity: MaturityLevel,
    /// Deployment policy hint: the adapter should come up unable to mutate.
    pub default_read_only: bool,
}

impl IntegrationManifest {
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.schema_version != INTEGRATION_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_non_empty("id", self.id.as_str())?;
        require_non_empty("display_name", &self.display_name)?;
        require_non_empty("version", &self.version)?;
        require_non_empty("provider", &self.provider)?;

        let mut names = BTreeSet::new();
        for capability in &self.capabilities {
            require_non_empty("capability.name", &capability.name)?;
            if !names.insert(capability.name.clone()) {
                return Err(ManifestValidationError::DuplicateCapability(
                    capability.name.clone(),
                ));
            }

            match (capability.class, capability.access) {
                (CapabilityClass::Observe | CapabilityClass::Discover, AccessMode::ReadOnly) => {}
                (CapabilityClass::Observe | CapabilityClass::Discover, _) => {
                    return Err(ManifestValidationError::InvalidAccessMode {
                        capability: capability.name.clone(),
                        class: capability.class,
                        access: capability.access,
                    });
                }
                (CapabilityClass::Simulate, AccessMode::ReadOnly | AccessMode::Simulated) => {}
                (CapabilityClass::Simulate, AccessMode::Mutating) => {
                    return Err(ManifestValidationError::InvalidAccessMode {
                        capability: capability.name.clone(),
                        class: capability.class,
                        access: capability.access,
                    });
                }
                (CapabilityClass::Actuate, AccessMode::Mutating) => {}
                (CapabilityClass::Actuate, _) => {
                    return Err(ManifestValidationError::InvalidAccessMode {
                        capability: capability.name.clone(),
                        class: capability.class,
                        access: capability.access,
                    });
                }
            }

            if capability.access == AccessMode::Mutating && capability.default_enabled {
                return Err(ManifestValidationError::MutationEnabledByDefault(
                    capability.name.clone(),
                ));
            }
        }

        let mut credential_names = BTreeSet::new();
        for credential in &self.credentials {
            require_non_empty("credential.name", &credential.name)?;
            require_non_empty("credential.scope", &credential.scope)?;
            if !credential_names.insert(credential.name.clone()) {
                return Err(ManifestValidationError::DuplicateCredential(
                    credential.name.clone(),
                ));
            }
        }

        Ok(())
    }

    /// Stricter v0.1 gate used by the current read-only integration fabric.
    pub fn validate_read_only_profile(&self) -> Result<(), ManifestValidationError> {
        self.validate()?;
        if !self.default_read_only {
            return Err(ManifestValidationError::NotReadOnlyByDefault);
        }
        if let Some(capability) = self
            .capabilities
            .iter()
            .find(|capability| capability.class == CapabilityClass::Actuate)
        {
            return Err(ManifestValidationError::ActuationDeclaredInReadOnlyProfile(
                capability.name.clone(),
            ));
        }
        if let Some(credential) = self
            .credentials
            .iter()
            .find(|credential| credential.mutation_capable)
        {
            return Err(ManifestValidationError::MutationCredentialInReadOnlyProfile(
                credential.name.clone(),
            ));
        }
        Ok(())
    }

    pub fn declares(&self, capability_name: &str) -> bool {
        self.capabilities
            .iter()
            .any(|capability| capability.name == capability_name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ManifestValidationError {
    #[error("unsupported integration manifest schema version {0}")]
    UnsupportedSchemaVersion(u16),
    #[error("required manifest field `{0}` is empty")]
    EmptyField(&'static str),
    #[error("duplicate capability `{0}`")]
    DuplicateCapability(String),
    #[error("duplicate credential declaration `{0}`")]
    DuplicateCredential(String),
    #[error("capability `{capability}` has invalid access {access:?} for {class:?}")]
    InvalidAccessMode {
        capability: String,
        class: CapabilityClass,
        access: AccessMode,
    },
    #[error("mutating capability `{0}` may not be enabled by default")]
    MutationEnabledByDefault(String),
    #[error("v0.1 read-only profile requires default_read_only=true")]
    NotReadOnlyByDefault,
    #[error("v0.1 read-only profile may not declare actuation capability `{0}`")]
    ActuationDeclaredInReadOnlyProfile(String),
    #[error("v0.1 read-only profile may not request mutation-capable credential `{0}`")]
    MutationCredentialInReadOnlyProfile(String),
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ManifestValidationError> {
    if value.trim().is_empty() {
        Err(ManifestValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readonly_manifest() -> IntegrationManifest {
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new("fixture"),
            display_name: "Fixture integration".into(),
            version: "0.1.0".into(),
            provider: "symthaea-test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities: vec![CapabilityDeclaration {
                name: "observe.host.metrics".into(),
                class: CapabilityClass::Observe,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            }],
            credentials: vec![],
            maturity: MaturityLevel::E1FixtureParsing,
            default_read_only: true,
        }
    }

    #[test]
    fn readonly_manifest_passes_strict_profile() {
        assert!(readonly_manifest().validate_read_only_profile().is_ok());
    }

    #[test]
    fn mutating_capability_cannot_be_default_enabled() {
        let mut manifest = readonly_manifest();
        manifest.capabilities.push(CapabilityDeclaration {
            name: "execute.host.reboot".into(),
            class: CapabilityClass::Actuate,
            access: AccessMode::Mutating,
            risk: RiskClass::High,
            reversible: false,
            default_enabled: true,
        });
        assert!(matches!(
            manifest.validate(),
            Err(ManifestValidationError::MutationEnabledByDefault(_))
        ));
    }

    #[test]
    fn read_only_profile_rejects_mutation_credentials() {
        let mut manifest = readonly_manifest();
        manifest.credentials.push(CredentialRequirement {
            name: "admin-token".into(),
            kind: CredentialKind::BearerToken,
            scope: "admin".into(),
            mutation_capable: true,
        });
        assert!(matches!(
            manifest.validate_read_only_profile(),
            Err(ManifestValidationError::MutationCredentialInReadOnlyProfile(_))
        ));
    }

    #[test]
    fn serde_roundtrip_is_stable() {
        let manifest = readonly_manifest();
        let json = serde_json::to_string(&manifest).unwrap();
        let restored: IntegrationManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, manifest);
    }
}
