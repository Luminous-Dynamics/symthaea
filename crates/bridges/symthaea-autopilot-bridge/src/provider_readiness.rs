// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lossless projection of PX4 readiness into generic Embodiment-v2 provider readiness.
//!
//! PX4 readiness and generic provider readiness are intentionally separate
//! schemas. This module bridges them without reducing the product assessment to
//! a boolean: every required PX4 proposition maps to a stable generic requirement
//! identity, observed evidence references are preserved, aggregate outcome
//! correspondence is checked, and the original PX4 assessment timestamp is
//! retained in the generic timed-readiness receipt.
//!
//! The projection is descriptive evidence only. `Ready` is not operator
//! authority, arm permission, Offboard permission, plant safety, transport
//! delivery, backend application, or measured physical execution.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::TimestampV1;
use symthaea_core::embodiment_provider::{
    BackendProviderId, BoundEmbodimentIdentityV1, ProviderBindingValidationError,
};
use symthaea_core::embodiment_provider_timing::TimedBoundEmbodimentLifecycleV1;
use symthaea_core::embodiment_readiness::{
    ProviderReadinessAssessmentV1, ProviderReadinessProfileV1,
    ProviderReadinessRequirementV1, ProviderReadinessStatusV1, ReadinessRequirementId,
    ReadinessRequirementStateV1, ReadinessValidationError,
};
use symthaea_core::embodiment_readiness_timing::{
    TimedProviderReadinessAssessmentV1, TimedReadinessValidationError,
};
use thiserror::Error;

use crate::readiness::{
    Px4ReadinessAssessmentV1, Px4ReadinessFactStateV1, Px4ReadinessOutcomeV1,
    Px4ReadinessRequirementV1, Px4ReadinessValidationError,
};

/// Schema version for PX4 -> generic-provider readiness projection evidence.
pub const PX4_PROVIDER_READINESS_PROJECTION_SCHEMA_V1: u16 = 1;
const MAPPING_PROFILE_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.provider-readiness-mapping-profile.v1\0";
const PROJECTION_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.provider-readiness-projection.v1\0";
const GENERIC_REQUIREMENT_NAMESPACE_V1: &str = "symthaea.px4.readiness";

/// Explicit product-producer -> generic-backend mapping used for one projection.
///
/// This profile makes the integration assertion visible and content-addressed.
/// It does not authenticate either identifier merely by storing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ProviderReadinessMappingProfileV1 {
    /// Projection schema version.
    pub schema_version: u16,
    /// Stable versioned mapping-policy identity.
    pub profile_id: String,
    /// Generic backend provider this mapping is allowed to describe.
    pub backend_provider_id: BackendProviderId,
    /// Exact PX4 readiness producer profile accepted by this mapping.
    pub px4_producer_profile_id: String,
    /// Domain-separated content commitment.
    pub profile_digest_hex: String,
}

impl Px4ProviderReadinessMappingProfileV1 {
    /// Construct, content-bind, and validate a mapping profile.
    pub fn new(
        profile_id: impl Into<String>,
        backend_provider_id: BackendProviderId,
        px4_producer_profile_id: impl Into<String>,
    ) -> Result<Self, Px4ProviderReadinessProjectionError> {
        let mut value = Self {
            schema_version: PX4_PROVIDER_READINESS_PROJECTION_SCHEMA_V1,
            profile_id: profile_id.into(),
            backend_provider_id,
            px4_producer_profile_id: px4_producer_profile_id.into(),
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate identifiers, schema, and content commitment.
    pub fn validate(&self) -> Result<(), Px4ProviderReadinessProjectionError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4ProviderReadinessProjectionError::DigestMismatch(
                "mapping_profile",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact integration policy.
    pub fn profile_content_id(&self) -> Result<String, Px4ProviderReadinessProjectionError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.provider-readiness-mapping-profile.v1:{}",
            self.profile_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ProviderReadinessProjectionError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "mapping_profile_id")?;
        validate_identifier(
            self.backend_provider_id.as_str(),
            "mapping_backend_provider_id",
        )?;
        validate_identifier(
            &self.px4_producer_profile_id,
            "px4_producer_profile_id",
        )
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(MAPPING_PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        feed_str(&mut hasher, self.backend_provider_id.as_str());
        feed_str(&mut hasher, &self.px4_producer_profile_id);
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Content-addressed receipt proving one deterministic PX4 -> generic readiness projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ProviderReadinessProjectionV1 {
    /// Projection schema version.
    pub schema_version: u16,
    /// Exact producer/backend integration policy used by this projection.
    pub mapping_profile: Px4ProviderReadinessMappingProfileV1,
    /// Exact product-specific PX4 readiness assessment being lowered.
    pub px4_assessment: Px4ReadinessAssessmentV1,
    /// Generic readiness assessment with the original PX4 assessment time preserved.
    pub provider_readiness: TimedProviderReadinessAssessmentV1,
    /// Domain-separated content commitment over both evidence models and mapping policy.
    pub projection_digest_hex: String,
}

impl Px4ProviderReadinessProjectionV1 {
    /// Deterministically project one validated PX4 assessment for one exact bound instance.
    pub fn new(
        mapping_profile: Px4ProviderReadinessMappingProfileV1,
        px4_assessment: Px4ReadinessAssessmentV1,
        identity: BoundEmbodimentIdentityV1,
    ) -> Result<Self, Px4ProviderReadinessProjectionError> {
        mapping_profile.validate()?;
        px4_assessment.validate()?;
        identity
            .validate()
            .map_err(Px4ProviderReadinessProjectionError::Provider)?;
        require_mapping_scope(&mapping_profile, &px4_assessment, &identity)?;

        let provider_readiness = project_provider_readiness(
            &mapping_profile,
            &px4_assessment,
            identity,
        )?;
        require_outcome_correspondence(&px4_assessment, &provider_readiness.assessment)?;

        let mut value = Self {
            schema_version: PX4_PROVIDER_READINESS_PROJECTION_SCHEMA_V1,
            mapping_profile,
            px4_assessment,
            provider_readiness,
            projection_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.projection_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate both nested evidence models, reconstruct the deterministic lowering,
    /// prove aggregate outcome correspondence, and verify the projection commitment.
    pub fn validate(&self) -> Result<(), Px4ProviderReadinessProjectionError> {
        self.validate_without_digest()?;
        if self.projection_digest_hex != self.compute_digest_hex() {
            return Err(Px4ProviderReadinessProjectionError::DigestMismatch(
                "projection",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact product -> generic projection.
    pub fn projection_id(&self) -> Result<String, Px4ProviderReadinessProjectionError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.provider-readiness-projection.v1:{}",
            self.projection_digest_hex
        ))
    }

    /// Attempt freshness-gated generic `Ready` lifecycle graduation.
    ///
    /// This delegates to the generic timed-readiness contract; it cannot re-stamp
    /// the original PX4 assessment time and does not grant any motor authority.
    pub fn ready_lifecycle_receipt(
        &self,
        now: &TimestampV1,
        max_age_ns: u64,
    ) -> Result<TimedBoundEmbodimentLifecycleV1, Px4ProviderReadinessProjectionError> {
        self.validate()?;
        self.provider_readiness
            .ready_lifecycle_receipt(now, max_age_ns)
            .map_err(Px4ProviderReadinessProjectionError::TimedReadiness)
    }

    fn validate_without_digest(&self) -> Result<(), Px4ProviderReadinessProjectionError> {
        validate_schema(self.schema_version)?;
        self.mapping_profile.validate()?;
        self.px4_assessment.validate()?;
        self.provider_readiness.validate()?;

        let identity = self.provider_readiness.assessment.identity.clone();
        require_mapping_scope(&self.mapping_profile, &self.px4_assessment, &identity)?;
        if self.provider_readiness.assessed_at != self.px4_assessment.assessed_at {
            return Err(Px4ProviderReadinessProjectionError::AssessmentTimestampMismatch);
        }

        let expected = project_provider_readiness(
            &self.mapping_profile,
            &self.px4_assessment,
            identity,
        )?;
        if expected != self.provider_readiness {
            return Err(Px4ProviderReadinessProjectionError::ProjectionMismatch);
        }
        require_outcome_correspondence(&self.px4_assessment, &self.provider_readiness.assessment)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROJECTION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.mapping_profile.profile_digest_hex);
        feed_str(&mut hasher, &self.px4_assessment.assessment_digest_hex);
        feed_str(&mut hasher, &self.provider_readiness.receipt_digest_hex);
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Stable generic requirement identity corresponding to one PX4 readiness proposition.
pub fn generic_requirement_id(
    requirement: Px4ReadinessRequirementV1,
) -> Result<ReadinessRequirementId, Px4ProviderReadinessProjectionError> {
    ReadinessRequirementId::new(format!(
        "{GENERIC_REQUIREMENT_NAMESPACE_V1}.{}",
        requirement.wire_token()
    ))
    .map_err(Px4ProviderReadinessProjectionError::GenericReadiness)
}

fn project_provider_readiness(
    mapping_profile: &Px4ProviderReadinessMappingProfileV1,
    px4_assessment: &Px4ReadinessAssessmentV1,
    identity: BoundEmbodimentIdentityV1,
) -> Result<TimedProviderReadinessAssessmentV1, Px4ProviderReadinessProjectionError> {
    let mut requirement_ids = Vec::with_capacity(px4_assessment.profile.required_facts.len());
    let mut requirements = Vec::with_capacity(px4_assessment.facts.len());

    for fact in &px4_assessment.facts {
        let requirement_id = generic_requirement_id(fact.requirement)?;
        requirement_ids.push(requirement_id.clone());
        let (state, evidence_id) = match fact.state {
            Px4ReadinessFactStateV1::Satisfied => (
                ReadinessRequirementStateV1::Satisfied,
                fact.evidence_id.clone(),
            ),
            Px4ReadinessFactStateV1::Unsatisfied => (
                ReadinessRequirementStateV1::Unsatisfied,
                fact.evidence_id.clone(),
            ),
            Px4ReadinessFactStateV1::Unobserved => {
                (ReadinessRequirementStateV1::Unknown, None)
            }
        };
        requirements.push(ProviderReadinessRequirementV1::new(
            requirement_id,
            state,
            evidence_id,
        )?);
    }

    // Mapping-profile + product-profile identity together define the exact
    // generic readiness profile. Different product prerequisite sets cannot
    // silently collapse into one generic profile identity.
    let generic_profile_id = format!(
        "symthaea.px4.provider-readiness.v1:{}:{}",
        mapping_profile.profile_digest_hex, px4_assessment.profile.profile_digest_hex
    );
    let generic_profile = ProviderReadinessProfileV1::new(
        mapping_profile.backend_provider_id.clone(),
        generic_profile_id,
        requirement_ids,
    )?;
    let generic_assessment = ProviderReadinessAssessmentV1::new(
        identity,
        generic_profile,
        requirements,
    )?;
    TimedProviderReadinessAssessmentV1::new(
        generic_assessment,
        px4_assessment.assessed_at.clone(),
    )
    .map_err(Px4ProviderReadinessProjectionError::TimedReadiness)
}

fn require_mapping_scope(
    mapping_profile: &Px4ProviderReadinessMappingProfileV1,
    px4_assessment: &Px4ReadinessAssessmentV1,
    identity: &BoundEmbodimentIdentityV1,
) -> Result<(), Px4ProviderReadinessProjectionError> {
    if identity.target.backend_provider_id != mapping_profile.backend_provider_id {
        return Err(Px4ProviderReadinessProjectionError::BackendProviderMismatch);
    }
    if px4_assessment.producer_profile_id != mapping_profile.px4_producer_profile_id {
        return Err(Px4ProviderReadinessProjectionError::ProducerProfileMismatch);
    }
    Ok(())
}

fn require_outcome_correspondence(
    px4_assessment: &Px4ReadinessAssessmentV1,
    provider_assessment: &ProviderReadinessAssessmentV1,
) -> Result<(), Px4ProviderReadinessProjectionError> {
    let px4_outcome = px4_assessment.outcome()?;
    let expected = match px4_outcome {
        Px4ReadinessOutcomeV1::Qualified => ProviderReadinessStatusV1::Ready,
        Px4ReadinessOutcomeV1::Blocked { .. } => ProviderReadinessStatusV1::NotReady,
        Px4ReadinessOutcomeV1::Incomplete { .. } => ProviderReadinessStatusV1::Incomplete,
    };
    if provider_assessment.status != expected {
        return Err(Px4ProviderReadinessProjectionError::OutcomeMismatch);
    }
    Ok(())
}

fn validate_schema(schema_version: u16) -> Result<(), Px4ProviderReadinessProjectionError> {
    if schema_version != PX4_PROVIDER_READINESS_PROJECTION_SCHEMA_V1 {
        return Err(Px4ProviderReadinessProjectionError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4ProviderReadinessProjectionError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4ProviderReadinessProjectionError::InvalidIdentifier(field));
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

/// Validation failure for PX4 -> generic provider readiness projection.
#[derive(Debug, Error)]
pub enum Px4ProviderReadinessProjectionError {
    /// Projection uses an unsupported schema version.
    #[error("unsupported PX4 provider-readiness projection schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// A required integration identifier is malformed.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Product assessment failed PX4-specific validation.
    #[error("invalid PX4 readiness assessment: {0}")]
    Px4Readiness(#[from] Px4ReadinessValidationError),
    /// Generic provider readiness construction/validation failed.
    #[error("invalid generic provider readiness: {0}")]
    GenericReadiness(#[from] ReadinessValidationError),
    /// Generic timed-readiness construction/validation failed.
    #[error("invalid timed provider readiness: {0}")]
    TimedReadiness(#[from] TimedReadinessValidationError),
    /// Generic bound-provider identity failed validation.
    #[error("invalid bound provider identity: {0}")]
    Provider(ProviderBindingValidationError),
    /// Bound instance provider does not match the explicit mapping profile.
    #[error("bound backend provider does not match PX4 readiness mapping profile")]
    BackendProviderMismatch,
    /// PX4 assessment producer does not match the explicit mapping profile.
    #[error("PX4 readiness producer profile does not match mapping profile")]
    ProducerProfileMismatch,
    /// Product assessment time and generic timed-assessment time differ.
    #[error("PX4 and generic provider readiness assessment timestamps differ")]
    AssessmentTimestampMismatch,
    /// Reconstructed deterministic generic projection differs from the stored projection.
    #[error("stored generic readiness does not match deterministic PX4 projection")]
    ProjectionMismatch,
    /// Product and generic aggregate readiness outcomes do not correspond.
    #[error("PX4 and generic provider readiness outcomes do not correspond")]
    OutcomeMismatch,
    /// Content commitment no longer matches the projection record.
    #[error("{0} content commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};
    use symthaea_core::embodiment_provider::{
        EmbodimentBindingResultV1, EmbodimentBindingTargetV1, EmbodimentInstanceId,
        EmbodimentProfileId, PlatformFamilyId,
    };

    const PRODUCER: &str = "symthaea.autopilot.px4.readiness.fixture.v1";
    const BACKEND: &str = "org.luminous.px4.mavlink.v1";

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new("px4.hrt").unwrap(), ns)
    }

    fn mapping() -> Px4ProviderReadinessMappingProfileV1 {
        Px4ProviderReadinessMappingProfileV1::new(
            "symthaea.px4.mavlink.readiness-mapping.v1",
            BackendProviderId::new(BACKEND).unwrap(),
            PRODUCER,
        )
        .unwrap()
    }

    fn bound_identity(provider: &str) -> BoundEmbodimentIdentityV1 {
        EmbodimentBindingResultV1::bound(
            EmbodimentBindingTargetV1::new(
                PlatformFamilyId::new("org.luminous.multirotor").unwrap(),
                EmbodimentProfileId::new("org.luminous.multirotor.quad-x.v1").unwrap(),
                BackendProviderId::new(provider).unwrap(),
            )
            .unwrap(),
            EmbodimentInstanceId::new("px4:sysid-1:compid-1").unwrap(),
            "binding:px4:fixture:1",
        )
        .unwrap()
        .bound_identity()
        .unwrap()
        .unwrap()
    }

    fn assessment(state: Px4ReadinessFactStateV1) -> Px4ReadinessAssessmentV1 {
        let requirement = Px4ReadinessRequirementV1::TimesyncQualified;
        let profile = crate::readiness::Px4ReadinessProfileV1::new(
            "px4.readiness.fixture.v1",
            vec![requirement],
        )
        .unwrap();
        let fact = match state {
            Px4ReadinessFactStateV1::Satisfied => {
                crate::readiness::Px4ReadinessFactV1::satisfied(
                    requirement,
                    "evidence:timesync:qualified:1",
                )
                .unwrap()
            }
            Px4ReadinessFactStateV1::Unsatisfied => {
                crate::readiness::Px4ReadinessFactV1::unsatisfied(
                    requirement,
                    "evidence:timesync:not-qualified:1",
                )
                .unwrap()
            }
            Px4ReadinessFactStateV1::Unobserved => {
                crate::readiness::Px4ReadinessFactV1::unobserved(requirement)
            }
        };
        Px4ReadinessAssessmentV1::new(PRODUCER, profile, ts(1_000), vec![fact]).unwrap()
    }

    #[test]
    fn every_px4_requirement_maps_to_unique_namespaced_generic_identity() {
        let requirements = [
            Px4ReadinessRequirementV1::HeartbeatFresh,
            Px4ReadinessRequirementV1::TimesyncQualified,
            Px4ReadinessRequirementV1::AngularVelocityValid,
            Px4ReadinessRequirementV1::AttitudeValid,
            Px4ReadinessRequirementV1::LocalAltitudeValid,
            Px4ReadinessRequirementV1::LocalPositionValid,
            Px4ReadinessRequirementV1::LocalVelocityValid,
            Px4ReadinessRequirementV1::GlobalPositionValid,
            Px4ReadinessRequirementV1::OffboardSignalPresent,
            Px4ReadinessRequirementV1::AcceptsOffboardSetpoints,
            Px4ReadinessRequirementV1::FailsafeInactive,
            Px4ReadinessRequirementV1::ReadyToArm,
            Px4ReadinessRequirementV1::LockdownInactive,
            Px4ReadinessRequirementV1::KillInactive,
            Px4ReadinessRequirementV1::TerminationInactive,
            Px4ReadinessRequirementV1::TransportAuthenticationSatisfied,
            Px4ReadinessRequirementV1::ControlProfileSupported,
        ];
        let mut seen = HashSet::new();
        for requirement in requirements {
            let id = generic_requirement_id(requirement).unwrap();
            assert!(id.as_str().starts_with("symthaea.px4.readiness."));
            assert!(seen.insert(id.as_str().to_string()));
        }
        assert_eq!(seen.len(), requirements.len());
    }

    #[test]
    fn product_outcomes_map_losslessly_to_generic_readiness_status() {
        let cases = [
            (
                Px4ReadinessFactStateV1::Satisfied,
                ProviderReadinessStatusV1::Ready,
            ),
            (
                Px4ReadinessFactStateV1::Unsatisfied,
                ProviderReadinessStatusV1::NotReady,
            ),
            (
                Px4ReadinessFactStateV1::Unobserved,
                ProviderReadinessStatusV1::Incomplete,
            ),
        ];

        for (product_state, expected_generic) in cases {
            let projection = Px4ProviderReadinessProjectionV1::new(
                mapping(),
                assessment(product_state),
                bound_identity(BACKEND),
            )
            .unwrap();
            assert_eq!(projection.provider_readiness.assessment.status, expected_generic);
            assert_eq!(projection.provider_readiness.assessed_at, projection.px4_assessment.assessed_at);
            projection.validate().unwrap();
        }
    }

    #[test]
    fn observed_evidence_ids_are_preserved_and_unobserved_stays_bare_unknown() {
        let satisfied = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity(BACKEND),
        )
        .unwrap();
        let requirement = &satisfied.provider_readiness.assessment.requirements[0];
        assert_eq!(requirement.state, ReadinessRequirementStateV1::Satisfied);
        assert_eq!(
            requirement.evidence_id.as_deref(),
            Some("evidence:timesync:qualified:1")
        );

        let unknown = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Unobserved),
            bound_identity(BACKEND),
        )
        .unwrap();
        let requirement = &unknown.provider_readiness.assessment.requirements[0];
        assert_eq!(requirement.state, ReadinessRequirementStateV1::Unknown);
        assert!(requirement.evidence_id.is_none());
    }

    #[test]
    fn mismatched_backend_or_producer_fails_closed() {
        let error = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity("org.luminous.mujoco.v1"),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Px4ProviderReadinessProjectionError::BackendProviderMismatch
        ));

        let mut wrong_producer = assessment(Px4ReadinessFactStateV1::Satisfied);
        wrong_producer.producer_profile_id = "another.px4.producer.v1".to_string();
        // Updating a committed product field without rebuilding is already invalid.
        assert!(matches!(
            Px4ProviderReadinessProjectionV1::new(
                mapping(),
                wrong_producer,
                bound_identity(BACKEND),
            ),
            Err(Px4ProviderReadinessProjectionError::Px4Readiness(_))
        ));
    }

    #[test]
    fn ready_lifecycle_is_freshness_gated_and_keeps_px4_assessment_time() {
        let projection = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity(BACKEND),
        )
        .unwrap();

        let ready = projection.ready_lifecycle_receipt(&ts(1_100), 100).unwrap();
        assert_eq!(ready.observed_at, projection.px4_assessment.assessed_at);
        assert_eq!(
            ready.lifecycle.evidence_id,
            projection.provider_readiness.receipt_id().unwrap()
        );

        assert!(matches!(
            projection.ready_lifecycle_receipt(&ts(2_000), 999),
            Err(Px4ProviderReadinessProjectionError::TimedReadiness(
                TimedReadinessValidationError::AssessmentTooOld { .. }
            ))
        ));
    }

    #[test]
    fn nested_mutation_invalidates_projection() {
        let mut projection = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity(BACKEND),
        )
        .unwrap();
        projection.px4_assessment.assessed_at.nanoseconds += 1;
        assert!(projection.validate().is_err());

        let mut projection = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity(BACKEND),
        )
        .unwrap();
        projection.provider_readiness.assessed_at.nanoseconds += 1;
        assert!(projection.validate().is_err());
    }

    #[test]
    fn mapping_profile_identity_binds_backend_and_product_producer() {
        let base = mapping();
        let other_backend = Px4ProviderReadinessMappingProfileV1::new(
            "symthaea.px4.mavlink.readiness-mapping.v1",
            BackendProviderId::new("org.luminous.px4.dds.v1").unwrap(),
            PRODUCER,
        )
        .unwrap();
        let other_producer = Px4ProviderReadinessMappingProfileV1::new(
            "symthaea.px4.mavlink.readiness-mapping.v1",
            BackendProviderId::new(BACKEND).unwrap(),
            "symthaea.autopilot.px4.readiness.other.v1",
        )
        .unwrap();
        assert_ne!(base.profile_digest_hex, other_backend.profile_digest_hex);
        assert_ne!(base.profile_digest_hex, other_producer.profile_digest_hex);
    }

    #[test]
    fn serde_round_trip_preserves_projection_identity() {
        let projection = Px4ProviderReadinessProjectionV1::new(
            mapping(),
            assessment(Px4ReadinessFactStateV1::Satisfied),
            bound_identity(BACKEND),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&projection).unwrap();
        let restored: Px4ProviderReadinessProjectionV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, projection);
        restored.validate().unwrap();
    }
}
