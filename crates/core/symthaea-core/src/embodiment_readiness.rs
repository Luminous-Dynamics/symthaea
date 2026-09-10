// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-derived readiness assessments for bound Embodiment v2 providers.
//!
//! Provider lifecycle and provider readiness are related but distinct propositions.
//! A lifecycle observation can say that a bound instance is `Ready`; this module
//! makes the *reason* for that proposition explicit as a provider-scoped,
//! canonical prerequisite profile plus one evidence state for every requirement.
//!
//! Readiness is deliberately not authority, capability, plant safety, or execution
//! permission. A fully satisfied readiness assessment may support a `Ready`
//! lifecycle proposition, but it does not authorize arming, actuation, mode changes,
//! or goal-directed behavior.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_provider::{
    BackendProviderId, BoundEmbodimentIdentityV1, BoundEmbodimentLifecycleStateV1,
    BoundEmbodimentLifecycleV1, ProviderBindingValidationError,
};

/// Schema version for provider-readiness records in this module.
pub const PROVIDER_READINESS_SCHEMA_V1: u16 = 1;
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.embodiment.readiness-profile.v1\0";
const REQUIREMENT_DOMAIN_V1: &[u8] = b"symthaea.embodiment.readiness-requirement.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] = b"symthaea.embodiment.readiness-assessment.v1\0";

/// Stable provider/profile-specific identity for one readiness prerequisite.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReadinessRequirementId(String);

impl ReadinessRequirementId {
    /// Construct a validated requirement identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ReadinessValidationError> {
        let value = value.into();
        validate_identifier(&value, "readiness_requirement_id")?;
        Ok(Self(value))
    }

    /// Borrow the canonical identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Content-addressed declaration of the complete prerequisite set for one
/// backend-provider readiness policy.
///
/// The profile prevents an assessment from reaching `Ready` by omitting a check
/// and prevents product-specific readiness rules from being applied to a different
/// backend provider merely because their string identifiers look similar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderReadinessProfileV1 {
    /// Schema version. Must equal [`PROVIDER_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact backend provider to which this readiness policy applies.
    pub backend_provider_id: BackendProviderId,
    /// Stable versioned profile identifier chosen by the provider implementation.
    pub profile_id: String,
    /// Canonically sorted, unique set of required prerequisite identities.
    pub required_requirements: Vec<ReadinessRequirementId>,
    /// Domain-separated content commitment over this exact profile declaration.
    pub profile_digest_hex: String,
}

impl ProviderReadinessProfileV1 {
    /// Construct, canonicalize, content-bind, and validate a readiness profile.
    pub fn new(
        backend_provider_id: BackendProviderId,
        profile_id: impl Into<String>,
        mut required_requirements: Vec<ReadinessRequirementId>,
    ) -> Result<Self, ReadinessValidationError> {
        let profile_id = profile_id.into();
        validate_identifier(backend_provider_id.as_str(), "backend_provider_id")?;
        validate_identifier(&profile_id, "readiness_profile_id")?;
        if required_requirements.is_empty() {
            return Err(ReadinessValidationError::EmptyRequirementSet);
        }
        for requirement_id in &required_requirements {
            validate_identifier(requirement_id.as_str(), "readiness_requirement_id")?;
        }
        required_requirements.sort();
        reject_duplicate_requirement_ids(&required_requirements)?;

        let mut value = Self {
            schema_version: PROVIDER_READINESS_SCHEMA_V1,
            backend_provider_id,
            profile_id,
            required_requirements,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, provider scope, canonical requirement set, and commitment.
    pub fn validate(&self) -> Result<(), ReadinessValidationError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(ReadinessValidationError::DigestMismatch("readiness_profile"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact provider-scoped prerequisite profile.
    pub fn profile_content_id(&self) -> Result<String, ReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.readiness-profile.v1:{}",
            self.profile_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), ReadinessValidationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(self.backend_provider_id.as_str(), "backend_provider_id")?;
        validate_identifier(&self.profile_id, "readiness_profile_id")?;
        if self.required_requirements.is_empty() {
            return Err(ReadinessValidationError::EmptyRequirementSet);
        }
        for requirement_id in &self.required_requirements {
            validate_identifier(requirement_id.as_str(), "readiness_requirement_id")?;
        }
        ensure_canonical_requirement_id_order(&self.required_requirements)?;
        reject_duplicate_requirement_ids(&self.required_requirements)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, self.backend_provider_id.as_str());
        feed_str(&mut hasher, &self.profile_id);
        hasher.update(&(self.required_requirements.len() as u64).to_le_bytes());
        for requirement_id in &self.required_requirements {
            feed_str(&mut hasher, requirement_id.as_str());
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Evidence state of one readiness prerequisite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessRequirementStateV1 {
    /// Current evidence establishes that the prerequisite is satisfied.
    Satisfied,
    /// Current evidence establishes that the prerequisite is not satisfied.
    Unsatisfied,
    /// The provider cannot currently establish either satisfaction or failure.
    Unknown,
    /// Supporting evidence exists but is no longer current under its own profile.
    Stale,
}

impl ReadinessRequirementStateV1 {
    /// Stable wire token used by content commitments and edge adapters.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::Satisfied => "satisfied",
            Self::Unsatisfied => "unsatisfied",
            Self::Unknown => "unknown",
            Self::Stale => "stale",
        }
    }

    /// Whether this state requires a concrete evidence reference.
    pub const fn requires_evidence(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Content-addressed state of one readiness prerequisite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderReadinessRequirementV1 {
    /// Schema version. Must equal [`PROVIDER_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Stable provider/profile-specific prerequisite identity.
    pub requirement_id: ReadinessRequirementId,
    /// Current state of this prerequisite.
    pub state: ReadinessRequirementStateV1,
    /// Evidence reference establishing this state when one exists.
    pub evidence_id: Option<String>,
    /// Domain-separated content commitment over this exact requirement state.
    pub requirement_digest_hex: String,
}

impl ProviderReadinessRequirementV1 {
    /// Construct, content-bind, and validate one requirement state.
    pub fn new(
        requirement_id: ReadinessRequirementId,
        state: ReadinessRequirementStateV1,
        evidence_id: Option<String>,
    ) -> Result<Self, ReadinessValidationError> {
        let mut value = Self {
            schema_version: PROVIDER_READINESS_SCHEMA_V1,
            requirement_id,
            state,
            evidence_id,
            requirement_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.requirement_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, identifier, evidence requirements, and content commitment.
    pub fn validate(&self) -> Result<(), ReadinessValidationError> {
        self.validate_without_digest()?;
        if self.requirement_digest_hex != self.compute_digest_hex() {
            return Err(ReadinessValidationError::DigestMismatch(
                "readiness_requirement",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact prerequisite state.
    pub fn requirement_state_id(&self) -> Result<String, ReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.readiness-requirement.v1:{}",
            self.requirement_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), ReadinessValidationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(self.requirement_id.as_str(), "readiness_requirement_id")?;
        match self.evidence_id.as_deref() {
            Some(evidence_id) => validate_identifier(evidence_id, "readiness_evidence_id")?,
            None if self.state.requires_evidence() => {
                return Err(ReadinessValidationError::MissingEvidence(
                    self.state.wire_token(),
                ));
            }
            None => {}
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REQUIREMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, self.requirement_id.as_str());
        feed_str(&mut hasher, self.state.wire_token());
        match &self.evidence_id {
            Some(evidence_id) => {
                hasher.update(&[1]);
                feed_str(&mut hasher, evidence_id);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Aggregate readiness result derived from a complete prerequisite set.
///
/// This enum deliberately does not derive `Ord`; readiness is not an authority,
/// safety, capability, or trust ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderReadinessStatusV1 {
    /// Every required prerequisite is currently satisfied.
    Ready,
    /// At least one required prerequisite is explicitly unsatisfied.
    NotReady,
    /// No prerequisite is explicitly unsatisfied, but at least one is unknown or stale.
    Incomplete,
}

impl ProviderReadinessStatusV1 {
    /// Stable wire token used by content commitments and edge adapters.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NotReady => "not_ready",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Content-addressed readiness assessment for one bound provider instance.
///
/// Requirements are canonicalized by identity and must exactly match the bound
/// profile's declared prerequisite set. The profile's backend provider must also
/// equal the bound instance provider before aggregate readiness is derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderReadinessAssessmentV1 {
    /// Schema version. Must equal [`PROVIDER_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact bound runtime instance being assessed.
    pub identity: BoundEmbodimentIdentityV1,
    /// Exact provider-scoped readiness profile used for this assessment.
    pub readiness_profile: ProviderReadinessProfileV1,
    /// Canonically sorted, unique state for every profile requirement.
    pub requirements: Vec<ProviderReadinessRequirementV1>,
    /// Aggregate status deterministically derived from `requirements`.
    pub status: ProviderReadinessStatusV1,
    /// Domain-separated content commitment over identity, profile, requirements, and status.
    pub assessment_digest_hex: String,
}

impl ProviderReadinessAssessmentV1 {
    /// Construct, canonicalize, verify provider/profile completeness, derive status, and commit.
    pub fn new(
        identity: BoundEmbodimentIdentityV1,
        readiness_profile: ProviderReadinessProfileV1,
        mut requirements: Vec<ProviderReadinessRequirementV1>,
    ) -> Result<Self, ReadinessValidationError> {
        identity
            .validate()
            .map_err(ReadinessValidationError::Provider)?;
        readiness_profile.validate()?;
        require_matching_provider(&identity, &readiness_profile)?;
        for requirement in &requirements {
            requirement.validate()?;
        }
        requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
        reject_duplicate_requirements(&requirements)?;
        require_exact_profile_set(&readiness_profile, &requirements)?;
        let status = derive_status(&requirements);
        let mut value = Self {
            schema_version: PROVIDER_READINESS_SCHEMA_V1,
            identity,
            readiness_profile,
            requirements,
            status,
            assessment_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.assessment_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate nested evidence, provider/profile scope, completeness, status, and commitment.
    pub fn validate(&self) -> Result<(), ReadinessValidationError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(ReadinessValidationError::DigestMismatch(
                "readiness_assessment",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact readiness assessment.
    pub fn assessment_id(&self) -> Result<String, ReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.readiness-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Build a `Ready` lifecycle proposition supported by this exact assessment.
    ///
    /// This succeeds only when every declared prerequisite is present and satisfied.
    /// The resulting lifecycle proposition is descriptive evidence, not motor authority.
    pub fn ready_lifecycle(&self) -> Result<BoundEmbodimentLifecycleV1, ReadinessValidationError> {
        self.validate()?;
        if self.status != ProviderReadinessStatusV1::Ready {
            return Err(ReadinessValidationError::AssessmentNotReady(self.status));
        }
        BoundEmbodimentLifecycleV1::new(
            self.identity.clone(),
            BoundEmbodimentLifecycleStateV1::Ready,
            self.assessment_id()?,
        )
        .map_err(ReadinessValidationError::Provider)
    }

    fn validate_without_digest(&self) -> Result<(), ReadinessValidationError> {
        validate_schema(self.schema_version)?;
        self.identity
            .validate()
            .map_err(ReadinessValidationError::Provider)?;
        self.readiness_profile.validate()?;
        require_matching_provider(&self.identity, &self.readiness_profile)?;
        for requirement in &self.requirements {
            requirement.validate()?;
        }
        ensure_canonical_requirement_order(&self.requirements)?;
        reject_duplicate_requirements(&self.requirements)?;
        require_exact_profile_set(&self.readiness_profile, &self.requirements)?;
        let derived = derive_status(&self.requirements);
        if self.status != derived {
            return Err(ReadinessValidationError::StatusMismatch {
                stored: self.status,
                derived,
            });
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.identity.identity_digest_hex);
        feed_str(&mut hasher, &self.readiness_profile.profile_digest_hex);
        hasher.update(&(self.requirements.len() as u64).to_le_bytes());
        for requirement in &self.requirements {
            feed_str(&mut hasher, &requirement.requirement_digest_hex);
        }
        feed_str(&mut hasher, self.status.wire_token());
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn derive_status(requirements: &[ProviderReadinessRequirementV1]) -> ProviderReadinessStatusV1 {
    if requirements
        .iter()
        .any(|requirement| requirement.state == ReadinessRequirementStateV1::Unsatisfied)
    {
        ProviderReadinessStatusV1::NotReady
    } else if requirements.iter().any(|requirement| {
        matches!(
            requirement.state,
            ReadinessRequirementStateV1::Unknown | ReadinessRequirementStateV1::Stale
        )
    }) {
        ProviderReadinessStatusV1::Incomplete
    } else {
        ProviderReadinessStatusV1::Ready
    }
}

fn require_matching_provider(
    identity: &BoundEmbodimentIdentityV1,
    profile: &ProviderReadinessProfileV1,
) -> Result<(), ReadinessValidationError> {
    if identity.target.backend_provider_id != profile.backend_provider_id {
        return Err(ReadinessValidationError::ProviderProfileMismatch);
    }
    Ok(())
}

fn require_exact_profile_set(
    profile: &ProviderReadinessProfileV1,
    requirements: &[ProviderReadinessRequirementV1],
) -> Result<(), ReadinessValidationError> {
    if profile.required_requirements.len() != requirements.len()
        || profile
            .required_requirements
            .iter()
            .zip(requirements)
            .any(|(expected, actual)| expected != &actual.requirement_id)
    {
        return Err(ReadinessValidationError::RequirementSetMismatch);
    }
    Ok(())
}

fn ensure_canonical_requirement_id_order(
    requirements: &[ReadinessRequirementId],
) -> Result<(), ReadinessValidationError> {
    if requirements.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(ReadinessValidationError::NonCanonicalRequirementOrder);
    }
    Ok(())
}

fn ensure_canonical_requirement_order(
    requirements: &[ProviderReadinessRequirementV1],
) -> Result<(), ReadinessValidationError> {
    if requirements
        .windows(2)
        .any(|pair| pair[0].requirement_id > pair[1].requirement_id)
    {
        return Err(ReadinessValidationError::NonCanonicalRequirementOrder);
    }
    Ok(())
}

fn reject_duplicate_requirement_ids(
    requirements: &[ReadinessRequirementId],
) -> Result<(), ReadinessValidationError> {
    if let Some(pair) = requirements.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(ReadinessValidationError::DuplicateRequirement(
            pair[0].as_str().to_string(),
        ));
    }
    Ok(())
}

fn reject_duplicate_requirements(
    requirements: &[ProviderReadinessRequirementV1],
) -> Result<(), ReadinessValidationError> {
    if let Some(pair) = requirements
        .windows(2)
        .find(|pair| pair[0].requirement_id == pair[1].requirement_id)
    {
        return Err(ReadinessValidationError::DuplicateRequirement(
            pair[0].requirement_id.as_str().to_string(),
        ));
    }
    Ok(())
}

fn validate_schema(schema_version: u16) -> Result<(), ReadinessValidationError> {
    if schema_version != PROVIDER_READINESS_SCHEMA_V1 {
        return Err(ReadinessValidationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), ReadinessValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(ReadinessValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for provider-readiness evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessValidationError {
    /// Record uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// A required identifier was empty, padded, too long, or contained controls.
    InvalidIdentifier(&'static str),
    /// Requirement state needs evidence but no evidence reference was supplied.
    MissingEvidence(&'static str),
    /// Readiness profiles must contain at least one declared prerequisite.
    EmptyRequirementSet,
    /// The same requirement identity appeared more than once.
    DuplicateRequirement(String),
    /// Persisted requirement order is not the canonical identity ordering.
    NonCanonicalRequirementOrder,
    /// Assessment requirements do not exactly match the selected profile declaration.
    RequirementSetMismatch,
    /// Readiness profile backend differs from the bound instance backend provider.
    ProviderProfileMismatch,
    /// Persisted aggregate status differs from the status derived from requirements.
    StatusMismatch {
        /// Status stored in the record.
        stored: ProviderReadinessStatusV1,
        /// Status recomputed from the prerequisite set.
        derived: ProviderReadinessStatusV1,
    },
    /// A non-ready/incomplete assessment was asked to support a Ready lifecycle proposition.
    AssessmentNotReady(ProviderReadinessStatusV1),
    /// Nested provider/binding/lifecycle evidence is invalid.
    Provider(ProviderBindingValidationError),
    /// Stored content commitment no longer matches its record.
    DigestMismatch(&'static str),
}

impl std::fmt::Display for ReadinessValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported provider readiness schema version {found}")
            }
            Self::InvalidIdentifier(field) => write!(f, "invalid {field}"),
            Self::MissingEvidence(state) => write!(f, "readiness state {state} requires evidence"),
            Self::EmptyRequirementSet => write!(f, "readiness profile has no requirements"),
            Self::DuplicateRequirement(id) => write!(f, "duplicate readiness requirement {id}"),
            Self::NonCanonicalRequirementOrder => {
                write!(f, "readiness requirements are not canonically ordered")
            }
            Self::RequirementSetMismatch => {
                write!(f, "assessment requirements do not match readiness profile")
            }
            Self::ProviderProfileMismatch => {
                write!(f, "readiness profile does not apply to bound backend provider")
            }
            Self::StatusMismatch { stored, derived } => write!(
                f,
                "stored readiness status {stored:?} does not match derived status {derived:?}"
            ),
            Self::AssessmentNotReady(status) => {
                write!(f, "readiness assessment is not Ready: {status:?}")
            }
            Self::Provider(error) => write!(f, "invalid provider evidence: {error}"),
            Self::DigestMismatch(record) => write!(f, "{record} content commitment mismatch"),
        }
    }
}

impl std::error::Error for ReadinessValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_provider::{
        EmbodimentBindingResultV1, EmbodimentBindingTargetV1, EmbodimentInstanceId,
        EmbodimentProfileId, PlatformFamilyId,
    };

    fn identity_for(provider: &str) -> BoundEmbodimentIdentityV1 {
        let target = EmbodimentBindingTargetV1::new(
            PlatformFamilyId::new("org.luminous.multirotor").unwrap(),
            EmbodimentProfileId::new("org.luminous.multirotor.quad-x.v1").unwrap(),
            BackendProviderId::new(provider).unwrap(),
        )
        .unwrap();
        EmbodimentBindingResultV1::bound(
            target,
            EmbodimentInstanceId::new("fixture:instance-1").unwrap(),
            "binding:fixture",
        )
        .unwrap()
        .bound_identity()
        .unwrap()
        .unwrap()
    }

    fn identity() -> BoundEmbodimentIdentityV1 {
        identity_for("org.luminous.px4.mavlink.v1")
    }

    fn id(value: &str) -> ReadinessRequirementId {
        ReadinessRequirementId::new(value).unwrap()
    }

    fn profile_for(provider: &str) -> ProviderReadinessProfileV1 {
        ProviderReadinessProfileV1::new(
            BackendProviderId::new(provider).unwrap(),
            "org.luminous.px4.readiness.sitl.v1",
            vec![id("heartbeat_fresh"), id("timesync_acceptable")],
        )
        .unwrap()
    }

    fn profile() -> ProviderReadinessProfileV1 {
        profile_for("org.luminous.px4.mavlink.v1")
    }

    fn requirement(
        value: &str,
        state: ReadinessRequirementStateV1,
    ) -> ProviderReadinessRequirementV1 {
        let evidence = if state == ReadinessRequirementStateV1::Unknown {
            None
        } else {
            Some(format!("evidence:{value}"))
        };
        ProviderReadinessRequirementV1::new(id(value), state, evidence).unwrap()
    }

    #[test]
    fn profile_order_is_canonical_provider_scoped_and_content_bound() {
        let first = ProviderReadinessProfileV1::new(
            BackendProviderId::new("org.luminous.px4.mavlink.v1").unwrap(),
            "org.luminous.px4.readiness.sitl.v1",
            vec![id("timesync_acceptable"), id("heartbeat_fresh")],
        )
        .unwrap();
        let second = profile();
        assert_eq!(first, second);
        assert_eq!(first.required_requirements[0].as_str(), "heartbeat_fresh");

        let other_provider = profile_for("org.luminous.mujoco.v1");
        assert_ne!(first.profile_digest_hex, other_provider.profile_digest_hex);
    }

    #[test]
    fn duplicate_profile_requirement_fails_closed() {
        assert!(matches!(
            ProviderReadinessProfileV1::new(
                BackendProviderId::new("org.luminous.px4.mavlink.v1").unwrap(),
                "org.luminous.px4.readiness.sitl.v1",
                vec![id("heartbeat_fresh"), id("heartbeat_fresh")]
            ),
            Err(ReadinessValidationError::DuplicateRequirement(_))
        ));
    }

    #[test]
    fn readiness_profile_cannot_assess_a_different_backend() {
        assert_eq!(
            ProviderReadinessAssessmentV1::new(
                identity_for("org.luminous.mujoco.v1"),
                profile(),
                vec![
                    requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                    requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
                ]
            ),
            Err(ReadinessValidationError::ProviderProfileMismatch)
        );
    }

    #[test]
    fn omitted_or_unexpected_requirement_cannot_derive_ready() {
        let omitted = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![requirement(
                "heartbeat_fresh",
                ReadinessRequirementStateV1::Satisfied,
            )],
        );
        assert_eq!(omitted, Err(ReadinessValidationError::RequirementSetMismatch));

        let unexpected = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
                requirement("extra_check", ReadinessRequirementStateV1::Satisfied),
            ],
        );
        assert_eq!(unexpected, Err(ReadinessValidationError::RequirementSetMismatch));
    }

    #[test]
    fn all_required_satisfied_derives_ready_and_can_support_lifecycle() {
        let assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
            ],
        )
        .unwrap();
        assert_eq!(assessment.status, ProviderReadinessStatusV1::Ready);
        let lifecycle = assessment.ready_lifecycle().unwrap();
        assert_eq!(lifecycle.state, BoundEmbodimentLifecycleStateV1::Ready);
        assert_eq!(lifecycle.evidence_id, assessment.assessment_id().unwrap());
    }

    #[test]
    fn explicit_negative_evidence_derives_not_ready() {
        let assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Unsatisfied),
            ],
        )
        .unwrap();
        assert_eq!(assessment.status, ProviderReadinessStatusV1::NotReady);
        assert_eq!(
            assessment.ready_lifecycle(),
            Err(ReadinessValidationError::AssessmentNotReady(
                ProviderReadinessStatusV1::NotReady
            ))
        );
    }

    #[test]
    fn unknown_or_stale_without_explicit_failure_derives_incomplete() {
        for state in [
            ReadinessRequirementStateV1::Unknown,
            ReadinessRequirementStateV1::Stale,
        ] {
            let assessment = ProviderReadinessAssessmentV1::new(
                identity(),
                profile(),
                vec![
                    requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                    requirement("timesync_acceptable", state),
                ],
            )
            .unwrap();
            assert_eq!(assessment.status, ProviderReadinessStatusV1::Incomplete);
        }
    }

    #[test]
    fn stronger_requirement_states_need_evidence_but_unknown_may_be_bare() {
        let requirement_id = id("heartbeat_fresh");
        for state in [
            ReadinessRequirementStateV1::Satisfied,
            ReadinessRequirementStateV1::Unsatisfied,
            ReadinessRequirementStateV1::Stale,
        ] {
            assert!(matches!(
                ProviderReadinessRequirementV1::new(requirement_id.clone(), state, None),
                Err(ReadinessValidationError::MissingEvidence(_))
            ));
        }
        ProviderReadinessRequirementV1::new(
            requirement_id,
            ReadinessRequirementStateV1::Unknown,
            None,
        )
        .unwrap();
    }

    #[test]
    fn assessment_input_order_does_not_change_identity() {
        let a = requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied);
        let b = requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied);
        let first = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![a.clone(), b.clone()],
        )
        .unwrap();
        let second = ProviderReadinessAssessmentV1::new(identity(), profile(), vec![b, a]).unwrap();
        assert_eq!(first.assessment_digest_hex, second.assessment_digest_hex);
        assert_eq!(first.requirements, second.requirements);
    }

    #[test]
    fn profile_status_or_requirement_mutation_breaks_validation() {
        let mut assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
            ],
        )
        .unwrap();
        assessment.status = ProviderReadinessStatusV1::Incomplete;
        assert!(matches!(
            assessment.validate(),
            Err(ReadinessValidationError::StatusMismatch { .. })
        ));

        let mut assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
            ],
        )
        .unwrap();
        assessment.requirements[0].evidence_id = Some("evidence:other".to_string());
        assert!(matches!(
            assessment.validate(),
            Err(ReadinessValidationError::DigestMismatch(
                "readiness_requirement"
            ))
        ));

        let mut assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
            ],
        )
        .unwrap();
        assessment.readiness_profile.profile_id = "org.luminous.px4.readiness.other.v1".into();
        assert!(matches!(
            assessment.validate(),
            Err(ReadinessValidationError::DigestMismatch("readiness_profile"))
        ));
    }

    #[test]
    fn serde_round_trip_preserves_assessment_identity() {
        let assessment = ProviderReadinessAssessmentV1::new(
            identity(),
            profile(),
            vec![
                requirement("heartbeat_fresh", ReadinessRequirementStateV1::Satisfied),
                requirement("timesync_acceptable", ReadinessRequirementStateV1::Satisfied),
            ],
        )
        .unwrap();
        let bytes = serde_json::to_vec(&assessment).unwrap();
        let restored: ProviderReadinessAssessmentV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, assessment);
        restored.validate().unwrap();
    }
}
