// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stable, dependency-light contracts for Symthaea extensions.
//!
//! This crate deliberately contains no cognitive implementation, runtime loader,
//! networking stack, or solver integration. It defines the shared vocabulary used
//! to describe extensions before a host decides whether and how to load them.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Stable identifier for an extension.
///
/// Reverse-domain notation is recommended, for example
/// `org.example.astronomy` or `io.luminous.mujoco`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExtensionId(pub String);

impl ExtensionId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable semantic capability identifier.
///
/// Capabilities describe what an extension can do rather than which crate or
/// implementation provides it, e.g. `science.astronomy.orbit_propagation`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityId(pub String);

impl CapabilityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Version of the host/guest extension ABI.
///
/// Hosts may accept a newer minor version when the major version matches, but
/// compatibility policy belongs to the host rather than this data crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiVersion {
    pub major: u16,
    pub minor: u16,
}

impl AbiVersion {
    pub const V1: Self = Self { major: 1, minor: 0 };
}

impl Default for AbiVersion {
    fn default() -> Self {
        Self::V1
    }
}

/// Broad extension role. An extension may expose several concrete capabilities
/// while retaining one primary role for discovery and UX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionKind {
    Domain,
    CognitiveSubsystem,
    Tool,
    Perception,
    Action,
    Simulation,
    Embodiment,
    Bridge,
    Model,
    KnowledgePack,
    SkillPack,
    Visualization,
    Other,
}

/// Execution substrate requested by an extension package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    /// Linked into a trusted Symthaea distribution.
    Native,
    /// Executed in the hardened WebAssembly extension host.
    Wasm,
    /// Capability is provided by a separately supervised process or service.
    Remote,
    /// Declarative extension containing no executable code.
    DataOnly,
}

/// Expected side-effect class for a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectClass {
    Pure,
    ReadOnly,
    SideEffecting,
    SafetyCritical,
}

/// One capability exported by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDescriptor {
    pub id: CapabilityId,
    pub description: String,
    #[serde(default)]
    pub effect: EffectClass,
}

impl Default for EffectClass {
    fn default() -> Self {
        Self::Pure
    }
}

/// Network access requested by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "mode", content = "hosts")]
pub enum NetworkPermission {
    #[default]
    None,
    /// Outbound connections only to explicit host/domain names.
    Allowlist(Vec<String>),
    /// Reserved for trusted native extensions; hosts should reject this for
    /// untrusted community code by policy.
    Unrestricted,
}

/// Filesystem access requested by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "mode", content = "paths")]
pub enum FilesystemPermission {
    #[default]
    None,
    ReadOnly(Vec<String>),
    ReadWrite(Vec<String>),
}

/// Capability-based permissions. Defaults intentionally deny ambient authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PermissionSet {
    #[serde(default)]
    pub network: NetworkPermission,
    #[serde(default)]
    pub filesystem: FilesystemPermission,
    #[serde(default)]
    pub gpu: bool,
    #[serde(default)]
    pub wall_clock: bool,
    #[serde(default)]
    pub randomness: bool,
    #[serde(default)]
    pub sensors: Vec<String>,
    #[serde(default)]
    pub actuators: Vec<String>,
}

/// Host-enforced resource envelope for an extension invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum guest memory in bytes.
    pub memory_bytes: u64,
    /// Instruction-like execution budget. Primarily maps to Wasmtime fuel.
    pub fuel: u64,
    /// Maximum wall-clock time for one invocation.
    pub max_wall_time_ms: u64,
    /// Maximum bytes returned by one invocation.
    pub max_output_bytes: u64,
    /// Maximum concurrent invocations for this extension.
    pub max_concurrency: u16,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            memory_bytes: 64 * 1024 * 1024,
            fuel: 50_000_000,
            max_wall_time_ms: 1_000,
            max_output_bytes: 4 * 1024 * 1024,
            max_concurrency: 1,
        }
    }
}

/// Portable manifest for native, WASM, remote, and data-only extensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub id: ExtensionId,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub abi: AbiVersion,
    pub kind: ExtensionKind,
    pub runtime: RuntimeKind,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub provides: Vec<CapabilityDescriptor>,
    #[serde(default)]
    pub requires: Vec<CapabilityId>,
    #[serde(default)]
    pub permissions: PermissionSet,
    #[serde(default)]
    pub resources: ResourceBudget,
}

/// Structural validation failure. Security policy remains a host concern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestProblem {
    pub field: &'static str,
    pub message: String,
}

impl ExtensionManifest {
    /// Validate structural invariants without imposing trust policy.
    pub fn validate(&self) -> Result<(), Vec<ManifestProblem>> {
        let mut problems = Vec::new();

        if !is_namespaced_identifier(self.id.as_str()) {
            problems.push(ManifestProblem {
                field: "id",
                message: "extension id must be a namespaced dot-separated identifier".into(),
            });
        }
        if self.name.trim().is_empty() {
            problems.push(ManifestProblem {
                field: "name",
                message: "name cannot be empty".into(),
            });
        }
        if self.version.trim().is_empty() {
            problems.push(ManifestProblem {
                field: "version",
                message: "version cannot be empty".into(),
            });
        }
        if self.abi.major == 0 {
            problems.push(ManifestProblem {
                field: "abi",
                message: "ABI major version must be non-zero".into(),
            });
        }
        if self.resources.memory_bytes == 0
            || self.resources.fuel == 0
            || self.resources.max_wall_time_ms == 0
            || self.resources.max_output_bytes == 0
            || self.resources.max_concurrency == 0
        {
            problems.push(ManifestProblem {
                field: "resources",
                message: "resource limits must all be non-zero".into(),
            });
        }

        let mut provided = HashSet::new();
        for capability in &self.provides {
            if !is_namespaced_identifier(capability.id.as_str()) {
                problems.push(ManifestProblem {
                    field: "provides",
                    message: format!(
                        "capability {:?} must be a namespaced dot-separated identifier",
                        capability.id.as_str()
                    ),
                });
            }
            if capability.description.trim().is_empty() {
                problems.push(ManifestProblem {
                    field: "provides",
                    message: format!(
                        "capability {:?} requires a non-empty description",
                        capability.id.as_str()
                    ),
                });
            }
            if !provided.insert(capability.id.as_str()) {
                problems.push(ManifestProblem {
                    field: "provides",
                    message: format!("duplicate capability {:?}", capability.id.as_str()),
                });
            }
        }

        let mut required = HashSet::new();
        for capability in &self.requires {
            if !is_namespaced_identifier(capability.as_str()) {
                problems.push(ManifestProblem {
                    field: "requires",
                    message: format!(
                        "capability {:?} must be a namespaced dot-separated identifier",
                        capability.as_str()
                    ),
                });
            }
            if !required.insert(capability.as_str()) {
                problems.push(ManifestProblem {
                    field: "requires",
                    message: format!("duplicate capability requirement {:?}", capability.as_str()),
                });
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems)
        }
    }
}

fn is_namespaced_identifier(value: &str) -> bool {
    let mut segments = value.split('.');
    let first = segments.next();
    let second = segments.next();
    if first.is_none() || second.is_none() {
        return false;
    }

    value.split('.').all(|segment| {
        !segment.is_empty()
            && segment
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new("org.example.astronomy"),
            name: "Example Astronomy".into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Domain,
            runtime: RuntimeKind::Wasm,
            description: "Orbit and photometry helpers".into(),
            provides: vec![CapabilityDescriptor {
                id: CapabilityId::new("science.astronomy.orbit_propagation"),
                description: "Propagate an orbit from an initial state".into(),
                effect: EffectClass::Pure,
            }],
            requires: vec![],
            permissions: PermissionSet::default(),
            resources: ResourceBudget::default(),
        }
    }

    #[test]
    fn default_permissions_deny_ambient_authority() {
        let permissions = PermissionSet::default();
        assert_eq!(permissions.network, NetworkPermission::None);
        assert_eq!(permissions.filesystem, FilesystemPermission::None);
        assert!(!permissions.gpu);
        assert!(permissions.sensors.is_empty());
        assert!(permissions.actuators.is_empty());
    }

    #[test]
    fn valid_manifest_passes_structural_validation() {
        fixture().validate().unwrap();
    }

    #[test]
    fn duplicate_capability_is_rejected() {
        let mut manifest = fixture();
        manifest.provides.push(manifest.provides[0].clone());
        let problems = manifest.validate().unwrap_err();
        assert!(problems.iter().any(|problem| problem.message.contains("duplicate capability")));
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = fixture();
        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: ExtensionManifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);
    }
}
