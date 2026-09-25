// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transitive input-closure evidence for external engineering solvers.
//!
//! A top-level model/netlist/case digest does not prove that every byte and
//! ambient configuration source capable of influencing a solver result was
//! bound. This crate provides a small, deterministic closure contract that
//! adapters can populate from already-admitted artifacts.
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
    /// than storing the potentially sensitive raw value.
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

    pub fn canonical_string(&self) -> String {
        format!("{}:{}", self.algorithm.canonical_name(), self.hex)
    }
}

/// One explicitly bound member of the solver input closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverInputArtifact {
    /// Stable identifier unique inside one closure.
    pub id: String,
    pub role: SolverInputRole,
    pub digest: ContentDigest,
    /// Parent artifact IDs explaining how a referenced/generated artifact enters
    /// the closure. Parent order is non-semantic and canonicalized before hash.
    #[serde(default)]
    pub parents: Vec<String>,
    /// Optional non-authoritative locator for audit/navigation. The digest binds
    /// bytes; a path/URL/name by itself never does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Optional reported semantic identity, such as `ngspice-45.2` or a plugin
    /// ABI/version. This complements rather than replaces the content digest.
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
        if self.id.trim().is_empty() {
            return Err(ClosureError::EmptyArtifactId);
        }
        self.digest.validate()?;
        if self
            .locator
            .as_deref()
            .is_some_and(|locator| locator.trim().is_empty())
        {
            return Err(ClosureError::EmptyLocator(self.id.clone()));
        }
        if self
            .reported_identity
            .as_deref()
            .is_some_and(|identity| identity.trim().is_empty())
        {
            return Err(ClosureError::EmptyReportedIdentity(self.id.clone()));
        }

        let mut seen = BTreeSet::new();
        for parent in &self.parents {
            if parent.trim().is_empty() {
                return Err(ClosureError::EmptyParentId(self.id.clone()));
            }
            if parent == &self.id {
                return Err(ClosureError::SelfParent(self.id.clone()));
            }
            if !seen.insert(parent) {
                return Err(ClosureError::DuplicateParent {
                    artifact: self.id.clone(),
                    parent: parent.clone(),
                });
            }
        }

        if matches!(self.role, SolverInputRole::Referenced | SolverInputRole::GeneratedIntermediate)
            && self.parents.is_empty()
        {
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
    /// Human/audit-readable constraints such as `user .spiceinit disabled via -n`.
    /// These are commitments and belong to closure identity.
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
        if self.profile_id.trim().is_empty() {
            return Err(ClosureError::EmptyProfileId);
        }
        if self.artifacts.is_empty() {
            return Err(ClosureError::EmptyClosure);
        }
        if self.ambient_constraints.is_empty()
            || self
                .ambient_constraints
                .iter()
                .any(|constraint| constraint.trim().is_empty())
        {
            return Err(ClosureError::MissingAmbientConstraint);
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
    /// canonicalized, so semantically identical closures hash identically.
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

        let mut constraints: Vec<_> = self
            .ambient_constraints
            .iter()
            .map(|constraint| constraint.trim())
            .collect();
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
                "locator",
                artifact.locator.as_deref().unwrap_or(""),
            );
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
        let unresolved: Vec<_> = self
            .unresolved
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .collect();
        if !unresolved.is_empty() {
            return Err(ClosureError::UnresolvedDependencies(unresolved));
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

fn assert_acyclic(
    by_id: &BTreeMap<&str, &SolverInputArtifact>,
) -> Result<(), ClosureError> {
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
        visit(parent.as_str(), by_id, visiting, visited)?;
    }
    visiting.remove(id);
    visited.insert(id);
    Ok(())
}

fn push_field(bytes: &mut Vec<u8>, label: &str, value: &str) {
    bytes.extend_from_slice(&(label.len() as u64).to_le_bytes());
    bytes.extend_from_slice(label.as_bytes());
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ClosureError {
    #[error("unsupported solver input closure schema {0:?}")]
    UnsupportedSchema(String),
    #[error("closure profile id cannot be empty")]
    EmptyProfileId,
    #[error("solver input closure cannot be empty")]
    EmptyClosure,
    #[error("solver input closure requires an explicit ambient discovery policy")]
    MissingAmbientPolicy,
    #[error("solver input closure requires at least one non-empty ambient constraint")]
    MissingAmbientConstraint,
    #[error("artifact id cannot be empty")]
    EmptyArtifactId,
    #[error("artifact {0:?} has an empty locator")]
    EmptyLocator(String),
    #[error("artifact {0:?} has an empty reported identity")]
    EmptyReportedIdentity(String),
    #[error("artifact {0:?} has an empty parent id")]
    EmptyParentId(String),
    #[error("artifact {0:?} cannot be its own parent")]
    SelfParent(String),
    #[error("artifact {artifact:?} repeats parent {parent:?}")]
    DuplicateParent { artifact: String, parent: String },
    #[error("artifact {0:?} requires at least one parent")]
    MissingParent(String),
    #[error("solver executable artifact {0:?} requires a reported identity/version")]
    MissingSolverIdentity(String),
    #[error("duplicate artifact id {0:?}")]
    DuplicateArtifactId(String),
    #[error("closure requires exactly one primary artifact, found {0}")]
    PrimaryCount(usize),
    #[error("closure requires exactly one solver executable artifact, found {0}")]
    SolverExecutableCount(usize),
    #[error("explicit ambient binding policy requires configuration or environment-binding artifacts")]
    ExplicitAmbientBindingsMissing,
    #[error("artifact {artifact:?} references unknown parent {parent:?}")]
    UnknownParent { artifact: String, parent: String },
    #[error("dependency graph contains a cycle involving {0:?}")]
    DependencyCycle(String),
    #[error("unknown artifact {0:?}")]
    UnknownArtifact(String),
    #[error("unresolved solver inputs prevent closure finalization: {0:?}")]
    UnresolvedDependencies(Vec<String>),
    #[error("invalid {algorithm:?} digest {value:?}")]
    InvalidDigest {
        algorithm: DigestAlgorithm,
        value: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> ContentDigest {
        ContentDigest::blake3(label.as_bytes())
    }

    fn primary() -> SolverInputArtifact {
        SolverInputArtifact::new("case", SolverInputRole::Primary, digest("case"))
            .with_locator("case/input.dat")
    }

    fn solver() -> SolverInputArtifact {
        SolverInputArtifact::new(
            "solver",
            SolverInputRole::SolverExecutable,
            digest("solver-bytes"),
        )
        .with_locator("/nix/store/example/bin/solver")
        .with_reported_identity("solver 1.2.3")
    }

    fn closure(artifacts: Vec<SolverInputArtifact>) -> SolverInputClosure {
        SolverInputClosure::new(
            "test-profile-v1",
            AmbientDiscoveryPolicy::Prohibited,
            artifacts,
            vec!["user configuration disabled".into()],
        )
        .unwrap()
    }

    #[test]
    fn semantically_identical_orderings_have_same_identity() {
        let referenced = SolverInputArtifact::new(
            "material",
            SolverInputRole::Referenced,
            digest("material"),
        )
        .with_parent("case");
        let a = closure(vec![primary(), solver(), referenced.clone()]);
        let b = closure(vec![solver(), referenced, primary()]);
        assert_eq!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn changing_transitive_bytes_changes_closure_identity() {
        let a_ref = SolverInputArtifact::new(
            "material",
            SolverInputRole::Referenced,
            digest("material-a"),
        )
        .with_parent("case");
        let b_ref = SolverInputArtifact::new(
            "material",
            SolverInputRole::Referenced,
            digest("material-b"),
        )
        .with_parent("case");
        let a = closure(vec![primary(), solver(), a_ref]);
        let b = closure(vec![primary(), solver(), b_ref]);
        assert_ne!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }

    #[test]
    fn path_is_not_content_identity() {
        let a = SolverInputArtifact::new("case", SolverInputRole::Primary, digest("a"))
            .with_locator("same/path");
        let b = SolverInputArtifact::new("case", SolverInputRole::Primary, digest("b"))
            .with_locator("same/path");
        assert_ne!(
            closure(vec![a, solver()]).closure_id().unwrap(),
            closure(vec![b, solver()]).closure_id().unwrap()
        );
    }

    #[test]
    fn unresolved_dependency_fails_closed() {
        let result = SolverInputClosureBuilder::new("v1")
            .ambient_policy(AmbientDiscoveryPolicy::Prohibited)
            .ambient_constraint("ambient disabled")
            .artifact(primary())
            .artifact(solver())
            .unresolved("vendor-model.lib")
            .finalize();
        assert!(matches!(result, Err(ClosureError::UnresolvedDependencies(_))));
    }

    #[test]
    fn unknown_parent_fails_closed() {
        let referenced = SolverInputArtifact::new(
            "material",
            SolverInputRole::Referenced,
            digest("material"),
        )
        .with_parent("missing");
        let result = SolverInputClosure::new(
            "v1",
            AmbientDiscoveryPolicy::Prohibited,
            vec![primary(), solver(), referenced],
            vec!["ambient disabled".into()],
        );
        assert!(matches!(result, Err(ClosureError::UnknownParent { .. })));
    }

    #[test]
    fn dependency_cycle_fails_closed() {
        let a = SolverInputArtifact::new("a", SolverInputRole::Referenced, digest("a"))
            .with_parent("b");
        let b = SolverInputArtifact::new("b", SolverInputRole::Referenced, digest("b"))
            .with_parent("a");
        let result = SolverInputClosure::new(
            "v1",
            AmbientDiscoveryPolicy::Prohibited,
            vec![primary(), solver(), a, b],
            vec!["ambient disabled".into()],
        );
        assert!(matches!(result, Err(ClosureError::DependencyCycle(_))));
    }

    #[test]
    fn exactly_one_primary_and_solver_are_required() {
        assert!(matches!(
            SolverInputClosure::new(
                "v1",
                AmbientDiscoveryPolicy::Prohibited,
                vec![solver()],
                vec!["ambient disabled".into()],
            ),
            Err(ClosureError::PrimaryCount(0))
        ));
        assert!(matches!(
            SolverInputClosure::new(
                "v1",
                AmbientDiscoveryPolicy::Prohibited,
                vec![primary()],
                vec!["ambient disabled".into()],
            ),
            Err(ClosureError::SolverExecutableCount(0))
        ));
    }

    #[test]
    fn explicit_ambient_policy_requires_bound_configuration_or_environment() {
        let result = SolverInputClosure::new(
            "v1",
            AmbientDiscoveryPolicy::ExplicitBindingsOnly,
            vec![primary(), solver()],
            vec!["HOME search path captured".into()],
        );
        assert_eq!(result, Err(ClosureError::ExplicitAmbientBindingsMissing));

        let environment = SolverInputArtifact::new(
            "env-home",
            SolverInputRole::EnvironmentBinding,
            digest("HOME=/qualified/root"),
        )
        .with_locator("env:HOME");
        assert!(SolverInputClosure::new(
            "v1",
            AmbientDiscoveryPolicy::ExplicitBindingsOnly,
            vec![primary(), solver(), environment],
            vec!["HOME search path captured".into()],
        )
        .is_ok());
    }

    #[test]
    fn generated_intermediate_requires_parent() {
        let generated = SolverInputArtifact::new(
            "mesh",
            SolverInputRole::GeneratedIntermediate,
            digest("mesh"),
        );
        assert!(matches!(
            generated.validate_local(),
            Err(ClosureError::MissingParent(id)) if id == "mesh"
        ));
    }

    #[test]
    fn malformed_digest_is_rejected() {
        assert!(matches!(
            ContentDigest::new(DigestAlgorithm::Sha256, "abc"),
            Err(ClosureError::InvalidDigest { .. })
        ));
    }

    #[test]
    fn ambient_constraint_order_is_non_semantic() {
        let mut a = closure(vec![primary(), solver()]);
        a.ambient_constraints = vec!["b".into(), "a".into()];
        let mut b = closure(vec![primary(), solver()]);
        b.ambient_constraints = vec!["a".into(), "b".into()];
        assert_eq!(a.closure_id().unwrap(), b.closure_id().unwrap());
    }
}
