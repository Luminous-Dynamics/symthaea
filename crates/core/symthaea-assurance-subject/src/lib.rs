// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-001: immutable multi-surface AI subject identity.
//!
//! Governing theorem:
//!
//! ```text
//! same product name
//!     != same model
//!     != same prompt
//!     != same authority
//!     != same runtime
//!     != same qualified subject
//!
//! same provider alias
//!     != immutable revision
//! ```
//!
//! This crate binds behaviorally material system-under-test surfaces without
//! mixing evaluator/corpus/campaign identity into the subject. It explicitly
//! distinguishes known, unknown, unavailable, and not-applicable surfaces.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_assurance_core::{
    AssuranceError as CoreAssuranceError, DigestSha256, StableId, SubjectComponent,
    SubjectComponentKind, SubjectManifest as CoreSubjectManifest,
};
use thiserror::Error;

pub const ASSURE_SUBJECT_SCHEMA: &str = "symthaea.assurance.subject.v1";
const CORE_BRIDGE_COMPONENT: &str = "assure-001-ai-subject-manifest";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SubjectError {
    #[error("subject profile requires at least one material surface")]
    EmptyProfile,
    #[error("duplicate profile surface: {0}")]
    DuplicateProfileSurface(String),
    #[error("duplicate subject binding: {0}")]
    DuplicateBinding(String),
    #[error("required subject surface is missing: {0}")]
    MissingSurface(String),
    #[error("subject binding is not registered by the profile: {0}")]
    UnexpectedSurface(String),
    #[error("applicable subject surface requires a locator: {0}")]
    MissingLocator(String),
    #[error("not-applicable subject surface must not carry a locator: {0}")]
    LocatorOnNotApplicable(String),
    #[error(transparent)]
    Core(#[from] CoreAssuranceError),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AiSurfaceKind {
    SourceTree,
    Image,
    Model,
    SystemPrompt,
    ToolAuthority,
    Policy,
    Runtime,
    DeploymentEnvelope,
    ExternalDependency(StableId),
    Custom(StableId),
}

impl AiSurfaceKind {
    fn canonical_name(&self) -> String {
        match self {
            Self::SourceTree => "source-tree".into(),
            Self::Image => "image".into(),
            Self::Model => "model".into(),
            Self::SystemPrompt => "system-prompt".into(),
            Self::ToolAuthority => "tool-authority".into(),
            Self::Policy => "policy".into(),
            Self::Runtime => "runtime".into(),
            Self::DeploymentEnvelope => "deployment-envelope".into(),
            Self::ExternalDependency(id) => format!("external-dependency:{}", id.as_str()),
            Self::Custom(id) => format!("custom:{}", id.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceProfile {
    profile_id: StableId,
    surfaces: Vec<AiSurfaceKind>,
}

impl SurfaceProfile {
    pub fn new(
        profile_id: StableId,
        mut surfaces: Vec<AiSurfaceKind>,
    ) -> Result<Self, SubjectError> {
        if surfaces.is_empty() {
            return Err(SubjectError::EmptyProfile);
        }
        surfaces.sort_by_cached_key(AiSurfaceKind::canonical_name);
        for pair in surfaces.windows(2) {
            if pair[0] == pair[1] {
                return Err(SubjectError::DuplicateProfileSurface(
                    pair[0].canonical_name(),
                ));
            }
        }
        Ok(Self {
            profile_id,
            surfaces,
        })
    }

    pub fn external_ai_v1(external_dependencies: Vec<StableId>) -> Result<Self, SubjectError> {
        let mut surfaces = vec![
            AiSurfaceKind::SourceTree,
            AiSurfaceKind::Image,
            AiSurfaceKind::Model,
            AiSurfaceKind::SystemPrompt,
            AiSurfaceKind::ToolAuthority,
            AiSurfaceKind::Policy,
            AiSurfaceKind::Runtime,
            AiSurfaceKind::DeploymentEnvelope,
        ];
        surfaces.extend(
            external_dependencies
                .into_iter()
                .map(AiSurfaceKind::ExternalDependency),
        );
        Self::new(StableId::new("external-ai-v1")?, surfaces)
    }

    pub fn profile_id(&self) -> &StableId {
        &self.profile_id
    }

    pub fn surfaces(&self) -> &[AiSurfaceKind] {
        &self.surfaces
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-subject-profile-v1\n");
        field(&mut out, "profile-id", self.profile_id.as_str());
        field(&mut out, "surface-count", &self.surfaces.len().to_string());
        for surface in &self.surfaces {
            field(&mut out, "surface", &surface.canonical_name());
        }
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceLocator {
    provider: Option<StableId>,
    name: StableId,
    declared_version: Option<StableId>,
}

impl SurfaceLocator {
    pub fn new(
        provider: Option<StableId>,
        name: StableId,
        declared_version: Option<StableId>,
    ) -> Self {
        Self {
            provider,
            name,
            declared_version,
        }
    }

    pub fn provider(&self) -> Option<&StableId> {
        self.provider.as_ref()
    }

    pub fn name(&self) -> &StableId {
        &self.name
    }

    pub fn declared_version(&self) -> Option<&StableId> {
        self.declared_version.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CommitmentMethod {
    ArtifactBytesSha256,
    CanonicalDescriptorSha256 { schema: StableId },
    ProviderRevisionTokenSha256 { namespace: StableId },
    CustomSha256(StableId),
}

impl CommitmentMethod {
    fn canonical_name(&self) -> String {
        match self {
            Self::ArtifactBytesSha256 => "artifact-bytes-sha256".into(),
            Self::CanonicalDescriptorSha256 { schema } => {
                format!("canonical-descriptor-sha256:{}", schema.as_str())
            }
            Self::ProviderRevisionTokenSha256 { namespace } => {
                format!("provider-revision-token-sha256:{}", namespace.as_str())
            }
            Self::CustomSha256(id) => format!("custom-sha256:{}", id.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialCommitment {
    method: CommitmentMethod,
    digest: DigestSha256,
}

impl MaterialCommitment {
    pub fn new(method: CommitmentMethod, digest: DigestSha256) -> Self {
        Self { method, digest }
    }

    pub fn artifact_bytes(digest: DigestSha256) -> Self {
        Self::new(CommitmentMethod::ArtifactBytesSha256, digest)
    }

    pub fn canonical_descriptor(schema: StableId, digest: DigestSha256) -> Self {
        Self::new(
            CommitmentMethod::CanonicalDescriptorSha256 { schema },
            digest,
        )
    }

    pub fn provider_revision_token(namespace: StableId, digest: DigestSha256) -> Self {
        Self::new(
            CommitmentMethod::ProviderRevisionTokenSha256 { namespace },
            digest,
        )
    }

    pub fn custom_sha256(method_id: StableId, digest: DigestSha256) -> Self {
        Self::new(CommitmentMethod::CustomSha256(method_id), digest)
    }

    pub fn method(&self) -> &CommitmentMethod {
        &self.method
    }

    pub fn digest(&self) -> &DigestSha256 {
        &self.digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnavailabilityReason {
    ProviderDoesNotExpose,
    AccessDenied,
    MeasurementUnavailable,
    Custom(StableId),
}

impl UnavailabilityReason {
    fn canonical_name(&self) -> String {
        match self {
            Self::ProviderDoesNotExpose => "provider-does-not-expose".into(),
            Self::AccessDenied => "access-denied".into(),
            Self::MeasurementUnavailable => "measurement-unavailable".into(),
            Self::Custom(id) => format!("custom:{}", id.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceState {
    Known(MaterialCommitment),
    Unknown,
    Unavailable(UnavailabilityReason),
    NotApplicable,
}

impl SurfaceState {
    fn canonical_name(&self) -> &'static str {
        match self {
            Self::Known(_) => "known",
            Self::Unknown => "unknown",
            Self::Unavailable(_) => "unavailable",
            Self::NotApplicable => "not-applicable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceBinding {
    kind: AiSurfaceKind,
    locator: Option<SurfaceLocator>,
    state: SurfaceState,
}

impl SurfaceBinding {
    pub fn new(
        kind: AiSurfaceKind,
        locator: Option<SurfaceLocator>,
        state: SurfaceState,
    ) -> Result<Self, SubjectError> {
        match (&locator, &state) {
            (None, SurfaceState::NotApplicable)
            | (Some(_), SurfaceState::Known(_))
            | (Some(_), SurfaceState::Unknown)
            | (Some(_), SurfaceState::Unavailable(_)) => Ok(Self {
                kind,
                locator,
                state,
            }),
            (Some(_), SurfaceState::NotApplicable) => {
                Err(SubjectError::LocatorOnNotApplicable(kind.canonical_name()))
            }
            (None, _) => Err(SubjectError::MissingLocator(kind.canonical_name())),
        }
    }

    pub fn applicable(
        kind: AiSurfaceKind,
        locator: SurfaceLocator,
        state: SurfaceState,
    ) -> Result<Self, SubjectError> {
        Self::new(kind, Some(locator), state)
    }

    pub fn not_applicable(kind: AiSurfaceKind) -> Self {
        Self {
            kind,
            locator: None,
            state: SurfaceState::NotApplicable,
        }
    }

    pub fn kind(&self) -> &AiSurfaceKind {
        &self.kind
    }

    pub fn locator(&self) -> Option<&SurfaceLocator> {
        self.locator.as_ref()
    }

    pub fn state(&self) -> &SurfaceState {
        &self.state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletenessSummary {
    pub known: usize,
    pub unknown: usize,
    pub unavailable: usize,
    pub not_applicable: usize,
}

impl CompletenessSummary {
    /// True when every applicable registered surface has a typed commitment.
    /// This does not establish artifact availability, immutable provider
    /// semantics, underlying content-addressedness, or replayability.
    pub fn has_complete_committed_identity(self) -> bool {
        self.unknown == 0 && self.unavailable == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiSubjectManifest {
    subject_key: StableId,
    profile: SurfaceProfile,
    bindings: Vec<SurfaceBinding>,
}

impl AiSubjectManifest {
    pub fn new(
        subject_key: StableId,
        profile: SurfaceProfile,
        mut bindings: Vec<SurfaceBinding>,
    ) -> Result<Self, SubjectError> {
        bindings.sort_by_cached_key(|binding| binding.kind.canonical_name());

        let expected: BTreeSet<_> = profile.surfaces.iter().cloned().collect();
        let mut seen = BTreeSet::new();
        for binding in &bindings {
            if !expected.contains(&binding.kind) {
                return Err(SubjectError::UnexpectedSurface(
                    binding.kind.canonical_name(),
                ));
            }
            if !seen.insert(binding.kind.clone()) {
                return Err(SubjectError::DuplicateBinding(
                    binding.kind.canonical_name(),
                ));
            }
        }
        for required in &profile.surfaces {
            if !seen.contains(required) {
                return Err(SubjectError::MissingSurface(required.canonical_name()));
            }
        }

        Ok(Self {
            subject_key,
            profile,
            bindings,
        })
    }

    pub fn subject_key(&self) -> &StableId {
        &self.subject_key
    }

    pub fn profile(&self) -> &SurfaceProfile {
        &self.profile
    }

    pub fn bindings(&self) -> &[SurfaceBinding] {
        &self.bindings
    }

    pub fn completeness(&self) -> CompletenessSummary {
        let mut summary = CompletenessSummary {
            known: 0,
            unknown: 0,
            unavailable: 0,
            not_applicable: 0,
        };
        for binding in &self.bindings {
            match &binding.state {
                SurfaceState::Known(_) => summary.known += 1,
                SurfaceState::Unknown => summary.unknown += 1,
                SurfaceState::Unavailable(_) => summary.unavailable += 1,
                SurfaceState::NotApplicable => summary.not_applicable += 1,
            }
        }
        summary
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-ai-subject-v1\n");
        field(&mut out, "schema", ASSURE_SUBJECT_SCHEMA);
        field(&mut out, "subject-key", self.subject_key.as_str());
        field(&mut out, "profile-id", self.profile.profile_id.as_str());
        field(&mut out, "profile-digest", self.profile.digest().as_str());
        field(
            &mut out,
            "profile-surface-count",
            &self.profile.surfaces.len().to_string(),
        );
        for surface in &self.profile.surfaces {
            field(&mut out, "profile-surface", &surface.canonical_name());
        }
        field(&mut out, "binding-count", &self.bindings.len().to_string());
        for binding in &self.bindings {
            field(&mut out, "surface-kind", &binding.kind.canonical_name());
            if let Some(locator) = &binding.locator {
                optional_id(&mut out, "provider", locator.provider.as_ref());
                field(&mut out, "name", locator.name.as_str());
                optional_id(
                    &mut out,
                    "declared-version",
                    locator.declared_version.as_ref(),
                );
            } else {
                field(&mut out, "provider", "");
                field(&mut out, "name", "");
                field(&mut out, "declared-version", "");
            }
            field(&mut out, "state", binding.state.canonical_name());
            match &binding.state {
                SurfaceState::Known(commitment) => {
                    field(
                        &mut out,
                        "commitment-method",
                        &commitment.method.canonical_name(),
                    );
                    field(&mut out, "commitment-digest", commitment.digest.as_str());
                    field(&mut out, "unavailability", "");
                }
                SurfaceState::Unavailable(reason) => {
                    field(&mut out, "commitment-method", "");
                    field(&mut out, "commitment-digest", "");
                    field(&mut out, "unavailability", &reason.canonical_name());
                }
                SurfaceState::Unknown | SurfaceState::NotApplicable => {
                    field(&mut out, "commitment-method", "");
                    field(&mut out, "commitment-digest", "");
                    field(&mut out, "unavailability", "");
                }
            }
        }
        out.into_bytes()
    }

    pub fn manifest_id(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    pub fn as_core_subject(&self) -> Result<CoreSubjectManifest, SubjectError> {
        let bridge_kind = SubjectComponentKind::Custom(StableId::new(CORE_BRIDGE_COMPONENT)?);
        Ok(CoreSubjectManifest::new(
            self.subject_key.clone(),
            vec![SubjectComponent {
                kind: bridge_kind,
                digest: self.manifest_id(),
            }],
        )?)
    }

    pub fn core_subject_id(&self) -> Result<DigestSha256, SubjectError> {
        Ok(self.as_core_subject()?.subject_id())
    }
}

fn digest_canonical(bytes: &[u8]) -> DigestSha256 {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    DigestSha256::new(encoded).expect("SHA-256 encoding is always valid")
}

fn optional_id(out: &mut String, label: &str, value: Option<&StableId>) {
    field(out, label, value.map(StableId::as_str).unwrap_or(""));
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}
