// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transitive input-closure evidence for external engineering solvers.
//!
//! A top-level model/netlist/case digest does not prove that every byte and
//! ambient configuration source capable of influencing a solver result was
//! bound. This crate provides a deterministic closure contract that adapters can
//! populate from already-admitted artifacts.
//!
//! It does **not** discover files, execute solvers, download model libraries,
//! or grant shell/process authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const CLOSURE_SCHEMA_ID: &str = "symthaea-solver-input-closure-v1";

/// Role played by one artifact in a complete external-solver input closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverInputRole {
    /// Canonical top-level request/model/case manifest.
    Primary,
    /// Transitively referenced immutable input.
    Referenced,
    /// Derived mesh/case/netlist expansion produced from already-bound inputs.
    GeneratedIntermediate,
    /// Solver executable bytes and reported identity.
    SolverExecutable,
    /// Runtime-loaded plugin, code model, or shared library.
    RuntimePlugin,
    /// Solver/application configuration capable of affecting the result.
    Configuration,
    /// Explicit environment/search-path binding, represented by a digest rather
    /// than storing a potentially sensitive raw value.
    EnvironmentBinding,
}

impl SolverInputRole {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Referenced => "referenced",
            Self::GeneratedIntermediate => "generated_intermediate",
            Self::SolverExecutable => "solver_executable",
            Self::RuntimePlugin => "runtime_plugin",
            Self::Configuration => "configuration",
            Self::EnvironmentBinding => "environment_binding",
        }
    }

    fn is_root(self) -> bool {
        matches!(self, Self::Primary | Self::SolverExecutable)
    }
}

/// Digest algorithm used by a bound artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestAlgorithm {
    Blake3,
    Sha256,
    Sha512,
}

impl DigestAlgorithm {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Blake3 => "blake3",
            Self::Sha256 => "sha256",
            Self::Sha512 => "sha512",
        }
    }

    fn expected_hex_len(self) -> usize {
        match self {
            Self::Blake3 | Self::Sha256 => 64,
            Self::Sha512 => 128,
        }
    }
}

/// Content digest of an immutable input artifact.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentDigest {
    pub algorithm: DigestAlgorithm,
    pub hex: String,
}

impl ContentDigest {
    pub fn new(algorithm: DigestAlgorithm, hex: impl Into<String>) -> Result<Self, ClosureError> {
        let digest = Self {
            algorithm,
            hex: hex.into().to_ascii_lowercase(),
        };
        digest.validate()?;
        Ok(digest)
    }

    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            algorithm: DigestAlgorithm::Blake3,
            hex: blake3::hash(bytes).to_hex().to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), ClosureError> {
        let expected = self.algorithm.expected_hex_len();
        if self.hex.len() != expected || !self.hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ClosureError::InvalidDigest {
                algorithm: self.algorithm,
                value: self.hex.clone(),
            });
        }
        Ok(())
    }

    /// Canonical textual digest identity. Case differences introduced through
    /// deserialization do not create a second identity for the same digest.
    pub fn canonical_string(&self) -> String {
        format!(
            "{}:{}",
            self.algorithm.canonical_name(),
            self.hex.to_ascii_lowercase()
        )
    }
}

/// One explicitly bound member of the solver input closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverInputArtifact {
    /// Stable identifier unique inside one closure.
    pub id: String,
    pub role: SolverInputRole,
    pub digest: ContentDigest,
    /// Parent artifact IDs explain how this influence enters the closure.
    /// Parent order is non-semantic and canonicalized before hashing.
    #[serde(default)]
    pub parents: Vec<String>,
    /// Audit/navigation locator only. It is intentionally excluded from closure
    /// identity; if path/search-location semantics affect solver behavior, that
    /// context must be modeled explicitly as configuration/environment input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Reported semantic identity such as `ngspice-45.2` or plugin ABI/version.
    /// This complements rather than replaces the content digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_identity: Option<String>,
}

impl SolverInputArtifact {
    pub fn new(id: impl Into<String>, role: SolverInputRole, digest: ContentDigest) -> Self {
        Self {
            id: id.into(),
            role,
            digest,
            parents: Vec::new(),
            locator: None,
            reported_identity: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parents.push(parent.into());
        self
    }

    pub fn with_locator(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }

    pub fn with_reported_identity(mut self, identity: impl Into<String>) -> Self {
        self.reported_identity = Some(identity.into());
        self
    }

    fn validate_local(&self) -> Result<(), ClosureError> {
        validate_canonical_id("artifact", &self.id)?;
        self.digest.validate()?;

        if self
            .locator
            .as_deref()
            .is_some_and(|locator| locator.trim().is_empty())
        {
            return Err(ClosureError::EmptyLocator(self.id.clone()));
        }

        if let Some(identity) = self.reported_identity.as_deref() {
            if identity.trim().is_empty() {
                return Err(ClosureError::EmptyReportedIdentity(self.id.clone()));
            }
            if identity.trim() != identity {
                return Err(ClosureError::NonCanonicalReportedIdentity(self.id.clone()));
            }
        }

        let mut seen = BTreeSet::new();
        for parent in &self.parents {
            validate_canonical_id("parent", parent)?;
            if parent == &self.id {
                return Err(ClosureError::SelfParent(self.id.clone()));
            }
            if !seen.insert(parent.as_str()) {
                return Err(ClosureError::DuplicateParent {
                    artifact: self.id.clone(),
                    parent: parent.clone(),
                });
            }
        }

        if self.role.is_root() && !self.parents.is_empty() {
            return Err(ClosureError::RootArtifactHasParent {
                artifact: self.id.clone(),
                role: self.role,
            });
        }
        if !self.role.is_root() && self.parents.is_empty() {
            return Err(ClosureError::MissingParent(self.id.clone()));
        }

        if self.role == SolverInputRole::SolverExecutable
            && self
                .reported_identity
                .as_deref()
                .is_none_or(|identity| identity.trim().is_empty())
        {
            return Err(ClosureError::MissingSolverIdentity(self.id.clone()));
        }

        Ok(())
    }
}

/// Whether ambient discovery is prohibited or admitted only through explicit
/// closure members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmbientDiscoveryPolicy {
    /// Adapter invokes the solver so ambient discovery mechanisms are disabled.
    Prohibited,
    /// Ambient/configuration influences are represented as explicit closure
    /// artifacts; no undeclared discovery is admitted.
    ExplicitBindingsOnly,
}

impl AmbientDiscoveryPolicy {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Prohibited => "prohibited",
            Self::ExplicitBindingsOnly => "explicit_bindings_only",
        }
    }
}

/// Complete, canonicalizable solver input closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverInputClosure {
    pub schema_id: String,
    /// Adapter-specific closure policy/version. Changing this changes identity.
    pub profile_id: String,
    pub ambient_policy: AmbientDiscoveryPolicy,
    /// Audit-readable commitments such as `user .spiceinit disabled via -n`.
    pub ambient_constraints: Vec<String>,
    pub artifacts: Vec<SolverInputArtifact>,
}

impl SolverInputClosure {
    pub fn new(
        profile_id: impl Into<String>,
        ambient_policy: AmbientDiscoveryPolicy,
        artifacts: Vec<SolverInputArtifact>,
        ambient_constraints: Vec<String>,
    ) -> Result<Self, ClosureError> {
        let closure = Self {
            schema_id: CLOSURE_SCHEMA_ID.into(),
            profile_id: profile_id.into(),
            ambient_policy,
            ambient_constraints,
            artifacts,
        };
        closure.validate()?;
        Ok(closure)
    }

    pub fn validate(&self) -> Result<(), ClosureError> {
        if self.schema_id != CLOSURE_SCHEMA_ID {
            return Err(ClosureError::UnsupportedSchema(self.schema_id.clone()));
        }
        validate_canonical_id("profile", &self.profile_id)?;
        if self.artifacts.is_empty() {
            return Err(ClosureError::EmptyClosure);
        }
        if self.ambient_constraints.is_empty() {
            return Err(ClosureError::MissingAmbientConstraint);
        }

        let mut constraints = BTreeSet::new();
        for constraint in &self.ambient_constraints {
            if constraint.trim().is_empty() {
                return Err(ClosureError::MissingAmbientConstraint);
            }
            if constraint.trim() != constraint {
                return Err(ClosureError::NonCanonicalAmbientConstraint(
                    constraint.clone(),
                ));
            }
            if !constraints.insert(constraint.as_str()) {
                return Err(ClosureError::DuplicateAmbientConstraint(
                    constraint.clone(),
                ));
            }
        }

        let mut by_id = BTreeMap::new();
        for artifact in &self.artifacts {
            artifact.validate_local()?;
            if by_id.insert(artifact.id.as_str(), artifact).is_some() {
                return Err(ClosureError::DuplicateArtifactId(artifact.id.clone()));
            }
        }

        let primary_count = self
            .artifacts
            .iter()
            .filter(|artifact| artifact.role == SolverInputRole::Primary)
            .count();
        if primary_count != 1 {
            return Err(ClosureError::PrimaryCount(primary_count));
        }

        let solver_count = self
            .artifacts
            .iter()
            .filter(|artifact| artifact.role == SolverInputRole::SolverExecutable)
            .count();
        if solver_count != 1 {
            return Err(ClosureError::SolverExecutableCount(solver_count));
        }

        if self.ambient_policy == AmbientDiscoveryPolicy::ExplicitBindingsOnly
            && !self.artifacts.iter().any(|artifact| {
                matches!(
                    artifact.role,
                    SolverInputRole::EnvironmentBinding | SolverInputRole::Configuration
                )
            })
        {
            return Err(ClosureError::ExplicitAmbientBindingsMissing);
        }

        for artifact in &self.artifacts {
            for parent in &artifact.parents {
                if !by_id.contains_key(parent.as_str()) {
                    return Err(ClosureError::UnknownParent {
                        artifact: artifact.id.clone(),
                        parent: parent.clone(),
                    });
                }
            }
        }

        assert_acyclic(&by_id)?;
        Ok(())
    }

    /// Stable closure identity. Artifact/parent/ambient-constraint ordering is
    /// canonicalized. Audit-only locators are intentionally excluded.
    pub fn closure_id(&self) -> Result<String, ClosureError> {
        self.validate()?;
        let mut bytes = Vec::new();
        push_field(&mut bytes, "schema", &self.schema_id);
        push_field(&mut bytes, "profile", &self.profile_id);
        push_field(
            &mut bytes,
            "ambient_policy",
            self.ambient_policy.canonical_name(),
        );

        let mut constraints: Vec<_> = self.ambient_constraints.iter().map(String::as_str).collect();
        constraints.sort_unstable();
        for constraint in constraints {
            push_field(&mut bytes, "ambient_constraint", constraint);
        }

        let mut artifacts: Vec<_> = self.artifacts.iter().collect();
        artifacts.sort_by(|a, b| a.id.cmp(&b.id));
        for artifact in artifacts {
            push_field(&mut bytes, "artifact_id", &artifact.id);
            push_field(&mut bytes, "role", artifact.role.canonical_name());
            push_field(&mut bytes, "digest", &artifact.digest.canonical_string());
            push_field(
                &mut bytes,
                "reported_identity",
                artifact.reported_identity.as_deref().unwrap_or(""),
            );
            let mut parents: Vec<_> = artifact.parents.iter().map(String::as_str).collect();
            parents.sort_unstable();
            for parent in parents {
                push_field(&mut bytes, "parent", parent);
            }
            push_field(&mut bytes, "artifact_end", "");
        }

        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }
}

/// Builder that preserves unresolved dependencies until finalization. Any
/// unresolved input makes closure finalization fail closed.
#[derive(Debug, Clone, Default)]
pub struct SolverInputClosureBuilder {
    profile_id: String,
    ambient_policy: Option<AmbientDiscoveryPolicy>,
    ambient_constraints: Vec<String>,
    artifacts: Vec<SolverInputArtifact>,
    unresolved: Vec<String>,
}

impl SolverInputClosureBuilder {
    pub fn new(profile_id: impl Into<String>) -> Self {
        Self {
            profile_id: profile_id.into(),
            ..Self::default()
        }
    }

    pub fn ambient_policy(mut self, policy: AmbientDiscoveryPolicy) -> Self {
        self.ambient_policy = Some(policy);
        self
    }

    pub fn ambient_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.ambient_constraints.push(constraint.into());
        self
    }

    pub fn artifact(mut self, artifact: SolverInputArtifact) -> Self {
        self.artifacts.push(artifact);
        self
    }

    pub fn unresolved(mut self, dependency: impl Into<String>) -> Self {
        self.unresolved.push(dependency.into());
        self
    }

    pub fn finalize(self) -> Result<SolverInputClosure, ClosureError> {
        if self.unresolved.iter().any(|item| item.trim().is_empty()) {
            return Err(ClosureError::EmptyUnresolvedDependency);
        }
        if !self.unresolved.is_empty() {
            return Err(ClosureError::UnresolvedDependencies(self.unresolved));
        }
        let policy = self
            .ambient_policy
            .ok_or(ClosureError::MissingAmbientPolicy)?;
        SolverInputClosure::new(
            self.profile_id,
            policy,
            self.artifacts,
            self.ambient_constraints,
        )
    }
}

fn validate_canonical_id(kind: &'static str, value: &str) -> Result<(), ClosureError> {
    if value.trim().is_empty() {
        return Err(ClosureError::EmptyIdentifier(kind));
    }
    if value.trim() != value {
        return Err(ClosureError::NonCanonicalIdentifier {
            kind,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn assert_acyclic(by_id: &BTreeMap<&str, &SolverInputArtifact>) -> Result<(), ClosureError> {
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in by_id.keys().copied() {
        visit(id, by_id, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn visit<'a>(
    id: &'a str,
    by_id: &BTreeMap<&'a str, &'a SolverInputArtifact>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
) -> Result<(), ClosureError> {
    if visited.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(ClosureError::DependencyCycle(id.to_string()));
    }
    let artifact = by_id
        .get(id)
        .copied()
        .ok_or_else(|| ClosureError::UnknownArtifact(id.to_string()))?;
    for parent in &artifact.parents {
        visit(parent, by_id, visiting, visited)?;
    }
    visiting.remove(id);
    visited.insert(id);
    Ok(())
}

fn push_field(bytes: &mut Vec<u8>, name: &str, value: &str) {
    let name_bytes = name.as_bytes();
    let value_bytes = value.as_bytes();
    bytes.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name_bytes);
    bytes.extend_from_slice(&(value_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value_bytes);
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ClosureError {
    #[error("unsupported solver-input closure schema {0:?}")]
    UnsupportedSchema(String),
    #[error("{0} identifier cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("{kind} identifier is not canonical: {value:?}")]
    NonCanonicalIdentifier { kind: &'static str, value: String },
    #[error("solver input closure cannot be empty")]
    EmptyClosure,
    #[error("artifact {0:?} has an empty locator")]
    EmptyLocator(String),
    #[error("artifact {0:?} has an empty reported identity")]
    EmptyReportedIdentity(String),
    #[error("artifact {0:?} has a reported identity with surrounding whitespace")]
    NonCanonicalReportedIdentity(String),
    #[error("artifact {0:?} cannot be its own parent")]
    SelfParent(String),
    #[error("artifact {artifact:?} repeats parent {parent:?}")]
    DuplicateParent { artifact: String, parent: String },
    #[error("non-root artifact {0:?} requires at least one parent")]
    MissingParent(String),
    #[error("root artifact {artifact:?} with role {role:?} cannot declare a parent")]
    RootArtifactHasParent {
        artifact: String,
        role: SolverInputRole,
    },
    #[error("solver executable artifact {0:?} requires reported identity/version")]
    MissingSolverIdentity(String),
    #[error("invalid {algorithm:?} digest {value:?}")]
    InvalidDigest {
        algorithm: DigestAlgorithm,
        value: String,
    },
    #[error("duplicate artifact id {0:?}")]
    DuplicateArtifactId(String),
    #[error("closure requires exactly one primary artifact, found {0}")]
    PrimaryCount(usize),
    #[error("closure requires exactly one solver executable, found {0}")]
    SolverExecutableCount(usize),
    #[error("ambient constraints must contain at least one non-empty commitment")]
    MissingAmbientConstraint,
    #[error("ambient constraint is not canonical: {0:?}")]
    NonCanonicalAmbientConstraint(String),
    #[error("duplicate ambient constraint {0:?}")]
    DuplicateAmbientConstraint(String),
    #[error("explicit-bindings ambient policy requires configuration/environment artifact")]
    ExplicitAmbientBindingsMissing,
    #[error("artifact {artifact:?} references unknown parent {parent:?}")]
    UnknownParent { artifact: String, parent: String },
    #[error("unknown artifact {0:?}")]
    UnknownArtifact(String),
    #[error("solver-input dependency graph contains a cycle at {0:?}")]
    DependencyCycle(String),
    #[error("ambient discovery policy is required")]
    MissingAmbientPolicy,
    #[error("unresolved dependency marker cannot be empty")]
    EmptyUnresolvedDependency,
    #[error("solver input closure contains unresolved dependencies: {0:?}")]
    UnresolvedDependencies(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> ContentDigest {
        ContentDigest::blake3(label.as_bytes())
    }

    fn primary() -> SolverInputArtifact {
        SolverInputArtifact::new("primary", SolverInputRole::Primary, d("primary-bytes"))
            .with_locator("case/main.in")
    }

    fn solver() -> SolverInputArtifact {
        SolverInputArtifact::new(
            "solver",
            SolverInputRole::SolverExecutable,
            d("solver-binary"),
        )
        .with_reported_identity("solver-1.0")
        .with_locator("/nix/store/example/bin/solver")
    }

    fn referenced() -> SolverInputArtifact {
        SolverInputArtifact::new("material", SolverInputRole::Referenced, d("material-data"))
            .with_parent("primary")
            .with_locator("materials/copper.toml")
    }

    fn config() -> SolverInputArtifact {
        SolverInputArtifact::new("config", SolverInputRole::Configuration, d("config-data"))
            .with_parent("solver")
    }

    fn fixture() -> SolverInputClosure {
        SolverInputClosure::new(
            "fixture-profile-v1",
            AmbientDiscoveryPolicy::ExplicitBindingsOnly,
            vec![primary(), solver(), referenced(), config()],
            vec![
                "ambient model discovery disabled".into(),
                "configuration is closure-bound".into(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn closure_identity_is_order_independent() {
        let a = fixture();
        let mut b = fixture();
        b.artifacts.reverse();
        b.ambient_constraints.reverse();
        assert_eq!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn transitive_artifact_bytes_change_closure_identity() {
        let a = fixture();
        let mut b = fixture();
        let material = b.artifacts.iter_mut().find(|a| a.id == "material").unwrap();
        material.digest = d("different-material-data");
        assert_ne!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn locator_is_navigation_not_content_identity() {
        let a = fixture();
        let mut b = fixture();
        b.artifacts
            .iter_mut()
            .find(|artifact| artifact.id == "material")
            .unwrap()
            .locator = Some("some/other/audit/path.toml".into());
        assert_eq!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn digest_case_is_canonical_after_deserialization() {
        let a = fixture();
        let mut b = fixture();
        b.artifacts[0].digest.hex = b.artifacts[0].digest.hex.to_ascii_uppercase();
        assert!(b.validate().is_ok());
        assert_eq!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn unresolved_dependency_blocks_finalization() {
        let result = SolverInputClosureBuilder::new("profile")
            .ambient_policy(AmbientDiscoveryPolicy::Prohibited)
            .ambient_constraint("ambient discovery disabled")
            .artifact(primary())
            .artifact(solver())
            .unresolved("vendor-model.lib")
            .finalize();
        assert!(matches!(result, Err(ClosureError::UnresolvedDependencies(_))));
    }

    #[test]
    fn empty_unresolved_marker_fails_closed() {
        let result = SolverInputClosureBuilder::new("profile")
            .ambient_policy(AmbientDiscoveryPolicy::Prohibited)
            .ambient_constraint("ambient discovery disabled")
            .artifact(primary())
            .artifact(solver())
            .unresolved("  ")
            .finalize();
        assert_eq!(result, Err(ClosureError::EmptyUnresolvedDependency));
    }

    #[test]
    fn unknown_parent_is_rejected() {
        let mut closure = fixture();
        closure
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.id == "material")
            .unwrap()
            .parents = vec!["missing".into()];
        assert!(matches!(closure.validate(), Err(ClosureError::UnknownParent { .. })));
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let a = SolverInputArtifact::new("a", SolverInputRole::Referenced, d("a"))
            .with_parent("b");
        let b = SolverInputArtifact::new("b", SolverInputRole::Referenced, d("b"))
            .with_parent("a");
        let closure = SolverInputClosure {
            schema_id: CLOSURE_SCHEMA_ID.into(),
            profile_id: "profile".into(),
            ambient_policy: AmbientDiscoveryPolicy::Prohibited,
            ambient_constraints: vec!["ambient discovery disabled".into()],
            artifacts: vec![primary(), solver(), a, b],
        };
        assert!(matches!(closure.validate(), Err(ClosureError::DependencyCycle(_))));
    }

    #[test]
    fn primary_and_solver_cardinality_are_enforced() {
        let mut no_primary = fixture();
        no_primary.artifacts.retain(|artifact| artifact.role != SolverInputRole::Primary);
        assert_eq!(no_primary.validate(), Err(ClosureError::PrimaryCount(0)));

        let mut two_solvers = fixture();
        two_solvers.artifacts.push(
            SolverInputArtifact::new("solver-2", SolverInputRole::SolverExecutable, d("solver-2"))
                .with_reported_identity("solver-2.0"),
        );
        assert_eq!(
            two_solvers.validate(),
            Err(ClosureError::SolverExecutableCount(2))
        );
    }

    #[test]
    fn explicit_ambient_policy_requires_bound_config_or_environment() {
        let closure = SolverInputClosure::new(
            "profile",
            AmbientDiscoveryPolicy::ExplicitBindingsOnly,
            vec![primary(), solver(), referenced()],
            vec!["ambient discovery restricted".into()],
        );
        assert_eq!(closure, Err(ClosureError::ExplicitAmbientBindingsMissing));
    }

    #[test]
    fn every_non_root_influence_requires_parent_lineage() {
        for role in [
            SolverInputRole::Referenced,
            SolverInputRole::GeneratedIntermediate,
            SolverInputRole::RuntimePlugin,
            SolverInputRole::Configuration,
            SolverInputRole::EnvironmentBinding,
        ] {
            let artifact = SolverInputArtifact::new("child", role, d("child"));
            assert_eq!(
                artifact.validate_local(),
                Err(ClosureError::MissingParent("child".into()))
            );
        }
    }

    #[test]
    fn root_artifacts_cannot_hide_dependency_edges() {
        let bad_primary = primary().with_parent("solver");
        assert!(matches!(
            bad_primary.validate_local(),
            Err(ClosureError::RootArtifactHasParent { .. })
        ));
    }

    #[test]
    fn malformed_digest_is_rejected() {
        assert!(ContentDigest::new(DigestAlgorithm::Sha256, "not-a-digest").is_err());
    }

    #[test]
    fn duplicate_ambient_commitment_is_rejected() {
        let mut closure = fixture();
        closure
            .ambient_constraints
            .push("ambient model discovery disabled".into());
        assert_eq!(
            closure.validate(),
            Err(ClosureError::DuplicateAmbientConstraint(
                "ambient model discovery disabled".into()
            ))
        );
    }

    #[test]
    fn json_round_trip_preserves_closure_identity() {
        let closure = fixture();
        let before = closure.closure_id().unwrap();
        let bytes = serde_json::to_vec(&closure).unwrap();
        let decoded: SolverInputClosure = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(before, decoded.closure_id().unwrap());
    }

    #[test]
    fn deserialization_does_not_bypass_validation() {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["artifacts"][0]["digest"]["hex"] = serde_json::Value::String("00".into());
        let decoded: SolverInputClosure = serde_json::from_value(value).unwrap();
        assert!(matches!(decoded.validate(), Err(ClosureError::InvalidDigest { .. })));
    }
}
