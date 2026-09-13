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
        surfaces.sort();
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

    /// Standard ASSURE-001 external-AI profile.
    ///
    /// Evaluator, corpus, intervention schedule, and verifier identity are
    /// deliberately absent: they are campaign/evidence identity, not subject
    /// identity. Material remote services are registered explicitly as
    /// external dependencies.
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
    /// Exact material commitment is available.
    Known(DigestSha256),
    /// The surface is material, but its exact identity is not known.
    Unknown,
    /// The surface is material, but the exact identity cannot currently be
    /// obtained for a declared reason class.
    Unavailable(UnavailabilityReason),
    /// The profile registers the surface, but it does not apply to this
    /// subject. This differs from omission and from unavailability.
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
    locator: SurfaceLocator,
    state: SurfaceState,
}

impl SurfaceBinding {
    pub fn new(kind: AiSurfaceKind, locator: SurfaceLocator, state: SurfaceState) -> Self {
        Self {
            kind,
            locator,
            state,
        }
    }

    pub fn kind(&self) -> &AiSurfaceKind {
        &self.kind
    }

    pub fn locator(&self) -> &SurfaceLocator {
        &self.locator
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
    pub fn is_exactly_replayable(self) -> bool {
        self.unknown == 0 && self.unavailable == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiSubjectManifest {
    subject_name: StableId,
    profile: SurfaceProfile,
    bindings: Vec<SurfaceBinding>,
}

impl AiSubjectManifest {
    pub fn new(
        subject_name: StableId,
        profile: SurfaceProfile,
        mut bindings: Vec<SurfaceBinding>,
    ) -> Result<Self, SubjectError> {
        bindings.sort_by(|left, right| left.kind.cmp(&right.kind));

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
            subject_name,
            profile,
            bindings,
        })
    }

    pub fn subject_name(&self) -> &StableId {
        &self.subject_name
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
        field(&mut out, "subject-name", self.subject_name.as_str());
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
            optional_id(&mut out, "provider", binding.locator.provider.as_ref());
            field(&mut out, "name", binding.locator.name.as_str());
            optional_id(
                &mut out,
                "declared-version",
                binding.locator.declared_version.as_ref(),
            );
            field(&mut out, "state", binding.state.canonical_name());
            match &binding.state {
                SurfaceState::Known(commitment) => {
                    field(&mut out, "commitment", commitment.as_str());
                    field(&mut out, "unavailability", "");
                }
                SurfaceState::Unavailable(reason) => {
                    field(&mut out, "commitment", "");
                    field(&mut out, "unavailability", &reason.canonical_name());
                }
                SurfaceState::Unknown | SurfaceState::NotApplicable => {
                    field(&mut out, "commitment", "");
                    field(&mut out, "unavailability", "");
                }
            }
        }
        out.into_bytes()
    }

    pub fn manifest_id(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    /// Bridges the richer ASSURE-001 identity into the stable ASSURE-000
    /// subject model without modifying the qualified core kernel.
    pub fn as_core_subject(&self) -> Result<CoreSubjectManifest, SubjectError> {
        let bridge_kind = SubjectComponentKind::Custom(StableId::new(CORE_BRIDGE_COMPONENT)?);
        Ok(CoreSubjectManifest::new(
            self.subject_name.clone(),
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
