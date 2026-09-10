// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider, binding, and bound-instance identity primitives for Embodiment v2.
//!
//! These types separate platform family, embodiment profile, backend provider,
//! and one bound runtime instance. They are deliberately behavior-neutral: a
//! provider identity is not a live robot, a successful binding is not a safety
//! approval, and a lifecycle state is not motor authority.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// Schema version for provider/binding records in this module.
pub const EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1: u16 = 1;
const TARGET_DOMAIN_V1: &[u8] = b"symthaea.embodiment.binding-target.v1\0";
const RESULT_DOMAIN_V1: &[u8] = b"symthaea.embodiment.binding-result.v1\0";
const IDENTITY_DOMAIN_V1: &[u8] = b"symthaea.embodiment.bound-identity.v1\0";
const LIFECYCLE_DOMAIN_V1: &[u8] = b"symthaea.embodiment.bound-lifecycle.v1\0";

/// Stable namespaced identity of a physical platform family, independent of backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlatformFamilyId(String);

impl PlatformFamilyId {
    /// Construct a validated platform-family identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ProviderBindingValidationError> {
        let value = value.into();
        validate_identifier(&value, "platform_family_id")?;
        Ok(Self(value))
    }

    /// Borrow the canonical identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identity of one exact morphology/airframe/sensor/actuator profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbodimentProfileId(String);

impl EmbodimentProfileId {
    /// Construct a validated embodiment-profile identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ProviderBindingValidationError> {
        let value = value.into();
        validate_identifier(&value, "embodiment_profile_id")?;
        Ok(Self(value))
    }

    /// Borrow the canonical identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identity of a backend provider implementation/profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendProviderId(String);

impl BackendProviderId {
    /// Construct a validated backend-provider identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ProviderBindingValidationError> {
        let value = value.into();
        validate_identifier(&value, "backend_provider_id")?;
        Ok(Self(value))
    }

    /// Borrow the canonical identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Runtime identity of one concrete simulator/device/robot instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbodimentInstanceId(String);

impl EmbodimentInstanceId {
    /// Construct a validated bound-instance identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ProviderBindingValidationError> {
        let value = value.into();
        validate_identifier(&value, "embodiment_instance_id")?;
        Ok(Self(value))
    }

    /// Borrow the canonical identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Exact platform/profile/provider target selected for a binding attempt.
///
/// `backend_provider_id` is mandatory: this type has no implicit simulator or
/// hardware fallback. A developer convenience layer may choose a default before
/// constructing this target, but the persisted binding evidence records the
/// provider that was actually selected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodimentBindingTargetV1 {
    /// Schema version. Must equal [`EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Platform family requested by the caller.
    pub platform_family_id: PlatformFamilyId,
    /// Exact embodiment/morphology profile requested by the caller.
    pub embodiment_profile_id: EmbodimentProfileId,
    /// Explicit backend provider selected for the attempt.
    pub backend_provider_id: BackendProviderId,
    /// Domain-separated content commitment over the three identities above.
    pub target_digest_hex: String,
}

impl EmbodimentBindingTargetV1 {
    /// Construct, commit, and validate an explicit binding target.
    pub fn new(
        platform_family_id: PlatformFamilyId,
        embodiment_profile_id: EmbodimentProfileId,
        backend_provider_id: BackendProviderId,
    ) -> Result<Self, ProviderBindingValidationError> {
        let mut value = Self {
            schema_version: EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1,
            platform_family_id,
            embodiment_profile_id,
            backend_provider_id,
            target_digest_hex: String::new(),
        };
        value.target_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, identifiers, and content commitment.
    pub fn validate(&self) -> Result<(), ProviderBindingValidationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(self.platform_family_id.as_str(), "platform_family_id")?;
        validate_identifier(self.embodiment_profile_id.as_str(), "embodiment_profile_id")?;
        validate_identifier(self.backend_provider_id.as_str(), "backend_provider_id")?;
        if self.target_digest_hex != self.compute_digest_hex() {
            return Err(ProviderBindingValidationError::DigestMismatch("binding_target"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact requested target.
    pub fn target_id(&self) -> Result<String, ProviderBindingValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.binding-target.v1:{}",
            self.target_digest_hex
        ))
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(TARGET_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, self.platform_family_id.as_str());
        feed_str(&mut hasher, self.embodiment_profile_id.as_str());
        feed_str(&mut hasher, self.backend_provider_id.as_str());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Outcome of one binding attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodimentBindingOutcomeV1 {
    /// Provider established one concrete instance identity.
    Bound {
        /// Runtime identity established by the binding evidence.
        instance_id: EmbodimentInstanceId,
    },
    /// Provider failed to establish a live instance.
    Failed {
        /// Stable provider/application failure code; not free-form secret material.
        failure_code: String,
    },
}

/// Content-addressed result of one explicit provider binding attempt.
///
/// A failed result contains no instance identity, making it impossible for a
/// binding failure to masquerade as a live body through this record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodimentBindingResultV1 {
    /// Schema version. Must equal [`EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact binding target that was attempted.
    pub target: EmbodimentBindingTargetV1,
    /// Bound or failed outcome.
    pub outcome: EmbodimentBindingOutcomeV1,
    /// Evidence/provenance reference establishing the result.
    pub evidence_id: String,
    /// Domain-separated content commitment over target, outcome, and evidence reference.
    pub result_digest_hex: String,
}

impl EmbodimentBindingResultV1 {
    /// Construct and commit a successful binding result.
    pub fn bound(
        target: EmbodimentBindingTargetV1,
        instance_id: EmbodimentInstanceId,
        evidence_id: impl Into<String>,
    ) -> Result<Self, ProviderBindingValidationError> {
        Self::new(
            target,
            EmbodimentBindingOutcomeV1::Bound { instance_id },
            evidence_id.into(),
        )
    }

    /// Construct and commit a failed binding result with no fake instance identity.
    pub fn failed(
        target: EmbodimentBindingTargetV1,
        failure_code: impl Into<String>,
        evidence_id: impl Into<String>,
    ) -> Result<Self, ProviderBindingValidationError> {
        let failure_code = failure_code.into();
        validate_identifier(&failure_code, "failure_code")?;
        Self::new(
            target,
            EmbodimentBindingOutcomeV1::Failed { failure_code },
            evidence_id.into(),
        )
    }

    /// Validate nested target, outcome identities, evidence reference, and digest.
    pub fn validate(&self) -> Result<(), ProviderBindingValidationError> {
        validate_schema(self.schema_version)?;
        self.target.validate()?;
        validate_identifier(&self.evidence_id, "binding_evidence_id")?;
        match &self.outcome {
            EmbodimentBindingOutcomeV1::Bound { instance_id } => {
                validate_identifier(instance_id.as_str(), "embodiment_instance_id")?;
            }
            EmbodimentBindingOutcomeV1::Failed { failure_code } => {
                validate_identifier(failure_code, "failure_code")?;
            }
        }
        if self.result_digest_hex != self.compute_digest_hex() {
            return Err(ProviderBindingValidationError::DigestMismatch("binding_result"));
        }
        Ok(())
    }

    /// Derive the bound-instance identity only for a successful result.
    pub fn bound_identity(
        &self,
    ) -> Result<Option<BoundEmbodimentIdentityV1>, ProviderBindingValidationError> {
        self.validate()?;
        match &self.outcome {
            EmbodimentBindingOutcomeV1::Bound { instance_id } => Ok(Some(
                BoundEmbodimentIdentityV1::new(self.target.clone(), instance_id.clone())?,
            )),
            EmbodimentBindingOutcomeV1::Failed { .. } => Ok(None),
        }
    }

    /// Content-addressed identity of this exact binding outcome.
    pub fn result_id(&self) -> Result<String, ProviderBindingValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.binding-result.v1:{}",
            self.result_digest_hex
        ))
    }

    fn new(
        target: EmbodimentBindingTargetV1,
        outcome: EmbodimentBindingOutcomeV1,
        evidence_id: String,
    ) -> Result<Self, ProviderBindingValidationError> {
        let mut value = Self {
            schema_version: EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1,
            target,
            outcome,
            evidence_id,
            result_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.result_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    fn validate_without_digest(&self) -> Result<(), ProviderBindingValidationError> {
        validate_schema(self.schema_version)?;
        self.target.validate()?;
        validate_identifier(&self.evidence_id, "binding_evidence_id")?;
        match &self.outcome {
            EmbodimentBindingOutcomeV1::Bound { instance_id } => {
                validate_identifier(instance_id.as_str(), "embodiment_instance_id")
            }
            EmbodimentBindingOutcomeV1::Failed { failure_code } => {
                validate_identifier(failure_code, "failure_code")
            }
        }
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RESULT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.target.target_digest_hex);
        match &self.outcome {
            EmbodimentBindingOutcomeV1::Bound { instance_id } => {
                hasher.update(&[1]);
                feed_str(&mut hasher, instance_id.as_str());
            }
            EmbodimentBindingOutcomeV1::Failed { failure_code } => {
                hasher.update(&[2]);
                feed_str(&mut hasher, failure_code);
            }
        }
        feed_str(&mut hasher, &self.evidence_id);
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Stable content-bound identity of one successfully bound runtime instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundEmbodimentIdentityV1 {
    /// Exact family/profile/provider target this instance realizes.
    pub target: EmbodimentBindingTargetV1,
    /// Runtime instance identity established during binding.
    pub instance_id: EmbodimentInstanceId,
    /// Domain-separated identity commitment.
    pub identity_digest_hex: String,
}

impl BoundEmbodimentIdentityV1 {
    /// Construct and validate a bound-instance identity.
    pub fn new(
        target: EmbodimentBindingTargetV1,
        instance_id: EmbodimentInstanceId,
    ) -> Result<Self, ProviderBindingValidationError> {
        let mut value = Self {
            target,
            instance_id,
            identity_digest_hex: String::new(),
        };
        value.identity_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate nested target, instance identity, and content commitment.
    pub fn validate(&self) -> Result<(), ProviderBindingValidationError> {
        self.target.validate()?;
        validate_identifier(self.instance_id.as_str(), "embodiment_instance_id")?;
        if self.identity_digest_hex != self.compute_digest_hex() {
            return Err(ProviderBindingValidationError::DigestMismatch("bound_identity"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact bound instance.
    pub fn identity_id(&self) -> Result<String, ProviderBindingValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.bound-identity.v1:{}",
            self.identity_digest_hex
        ))
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(IDENTITY_DOMAIN_V1);
        feed_str(&mut hasher, &self.target.target_digest_hex);
        feed_str(&mut hasher, self.instance_id.as_str());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Lifecycle state of an already-bound embodiment instance.
///
/// This enum deliberately has no `Ord`: lifecycle position is not a safety,
/// trust, or authority ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundEmbodimentLifecycleStateV1 {
    /// Binding completed but readiness prerequisites are not yet established.
    Bound,
    /// Provider-specific readiness prerequisites are currently satisfied.
    Ready,
    /// Instance is participating in its selected operational mode.
    Active,
    /// Instance remains bound but one or more qualified capabilities are degraded.
    Degraded,
    /// A controlled disconnect/handoff has begun.
    Disconnecting,
    /// The previously bound instance is no longer connected/active.
    Disconnected,
    /// Provider reports a fault affecting the bound instance lifecycle.
    Faulted,
}

impl BoundEmbodimentLifecycleStateV1 {
    /// Stable wire token for persisted evidence and edge adapters.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Ready => "ready",
            Self::Active => "active",
            Self::Degraded => "degraded",
            Self::Disconnecting => "disconnecting",
            Self::Disconnected => "disconnected",
            Self::Faulted => "faulted",
        }
    }
}

/// Evidence-bound lifecycle observation for one already-bound instance.
///
/// Every lifecycle state requires an evidence reference. This record does not
/// define allowed state transitions; protocol/device-specific lifecycle machines
/// may skip or revisit states while preserving each observed proposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundEmbodimentLifecycleV1 {
    /// Schema version. Must equal [`EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Concrete instance whose lifecycle state is being observed.
    pub identity: BoundEmbodimentIdentityV1,
    /// Observed provider lifecycle state.
    pub state: BoundEmbodimentLifecycleStateV1,
    /// Evidence/provenance reference establishing this lifecycle observation.
    pub evidence_id: String,
    /// Domain-separated content commitment.
    pub lifecycle_digest_hex: String,
}

impl BoundEmbodimentLifecycleV1 {
    /// Construct, content-bind, and validate a lifecycle observation.
    pub fn new(
        identity: BoundEmbodimentIdentityV1,
        state: BoundEmbodimentLifecycleStateV1,
        evidence_id: impl Into<String>,
    ) -> Result<Self, ProviderBindingValidationError> {
        let mut value = Self {
            schema_version: EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1,
            identity,
            state,
            evidence_id: evidence_id.into(),
            lifecycle_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.lifecycle_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate identity, evidence reference, schema, and content commitment.
    pub fn validate(&self) -> Result<(), ProviderBindingValidationError> {
        self.validate_without_digest()?;
        if self.lifecycle_digest_hex != self.compute_digest_hex() {
            return Err(ProviderBindingValidationError::DigestMismatch("bound_lifecycle"));
        }
        Ok(())
    }

    /// Content-addressed identity of this lifecycle observation.
    pub fn lifecycle_id(&self) -> Result<String, ProviderBindingValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.bound-lifecycle.v1:{}",
            self.lifecycle_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), ProviderBindingValidationError> {
        validate_schema(self.schema_version)?;
        self.identity.validate()?;
        validate_identifier(&self.evidence_id, "lifecycle_evidence_id")
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(LIFECYCLE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.identity.identity_digest_hex);
        feed_str(&mut hasher, self.state.wire_token());
        feed_str(&mut hasher, &self.evidence_id);
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), ProviderBindingValidationError> {
    if schema_version != EMBODIMENT_PROVIDER_BINDING_SCHEMA_V1 {
        return Err(ProviderBindingValidationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), ProviderBindingValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(ProviderBindingValidationError::InvalidIdentifier(field));
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

/// Validation failure for provider/binding evidence primitives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderBindingValidationError {
    /// Record uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// A required identifier was empty, padded, too long, or contained controls.
    InvalidIdentifier(&'static str),
    /// A stored content commitment no longer matches its record.
    DigestMismatch(&'static str),
}

impl std::fmt::Display for ProviderBindingValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported embodiment provider/binding schema version {found}")
            }
            Self::InvalidIdentifier(field) => write!(f, "invalid {field}"),
            Self::DigestMismatch(record) => write!(f, "{record} content commitment mismatch"),
        }
    }
}

impl std::error::Error for ProviderBindingValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(provider: &str) -> EmbodimentBindingTargetV1 {
        EmbodimentBindingTargetV1::new(
            PlatformFamilyId::new("org.luminous.multirotor").unwrap(),
            EmbodimentProfileId::new("org.luminous.multirotor.quad-x.v1").unwrap(),
            BackendProviderId::new(provider).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn backend_selection_is_explicit_and_part_of_target_identity() {
        let simple = target("org.luminous.simple-sim.v1");
        let mujoco = target("org.luminous.mujoco.v1");
        assert_ne!(simple.target_id().unwrap(), mujoco.target_id().unwrap());
        assert_eq!(simple.platform_family_id, mujoco.platform_family_id);
        assert_eq!(simple.embodiment_profile_id, mujoco.embodiment_profile_id);
    }

    #[test]
    fn failed_binding_cannot_fabricate_a_bound_identity() {
        let result = EmbodimentBindingResultV1::failed(
            target("org.luminous.px4.mavlink.v1"),
            "transport_unavailable",
            "binding-attempt:42",
        )
        .unwrap();
        assert!(matches!(result.outcome, EmbodimentBindingOutcomeV1::Failed { .. }));
        assert_eq!(result.bound_identity().unwrap(), None);
    }

    #[test]
    fn successful_binding_establishes_distinct_runtime_instance_identity() {
        let target = target("org.luminous.px4.mavlink.v1");
        let first = EmbodimentBindingResultV1::bound(
            target.clone(),
            EmbodimentInstanceId::new("px4:sysid-1:compid-1").unwrap(),
            "binding:1",
        )
        .unwrap();
        let second = EmbodimentBindingResultV1::bound(
            target,
            EmbodimentInstanceId::new("px4:sysid-2:compid-1").unwrap(),
            "binding:2",
        )
        .unwrap();

        let first_identity = first.bound_identity().unwrap().unwrap();
        let second_identity = second.bound_identity().unwrap().unwrap();
        assert_ne!(first_identity.identity_id().unwrap(), second_identity.identity_id().unwrap());
    }

    #[test]
    fn lifecycle_state_requires_bound_identity_and_evidence() {
        let result = EmbodimentBindingResultV1::bound(
            target("org.luminous.simple-sim.v1"),
            EmbodimentInstanceId::new("fixture:1").unwrap(),
            "binding:fixture:1",
        )
        .unwrap();
        let identity = result.bound_identity().unwrap().unwrap();
        let ready = BoundEmbodimentLifecycleV1::new(
            identity,
            BoundEmbodimentLifecycleStateV1::Ready,
            "readiness:fixture:1",
        )
        .unwrap();
        assert_eq!(ready.state, BoundEmbodimentLifecycleStateV1::Ready);
        assert!(ready.lifecycle_id().unwrap().contains("bound-lifecycle.v1:"));
    }

    #[test]
    fn lifecycle_tokens_are_stable_but_not_a_trust_order() {
        let cases = [
            (BoundEmbodimentLifecycleStateV1::Bound, "bound"),
            (BoundEmbodimentLifecycleStateV1::Ready, "ready"),
            (BoundEmbodimentLifecycleStateV1::Active, "active"),
            (BoundEmbodimentLifecycleStateV1::Degraded, "degraded"),
            (BoundEmbodimentLifecycleStateV1::Disconnecting, "disconnecting"),
            (BoundEmbodimentLifecycleStateV1::Disconnected, "disconnected"),
            (BoundEmbodimentLifecycleStateV1::Faulted, "faulted"),
        ];
        for (state, token) in cases {
            assert_eq!(state.wire_token(), token);
            assert_eq!(serde_json::to_string(&state).unwrap(), format!("\"{token}\""));
        }
    }

    #[test]
    fn mutation_breaks_target_result_identity_and_lifecycle_commitments() {
        let mut target_value = target("org.luminous.simple-sim.v1");
        target_value.backend_provider_id = BackendProviderId::new("org.luminous.mujoco.v1").unwrap();
        assert_eq!(
            target_value.validate(),
            Err(ProviderBindingValidationError::DigestMismatch("binding_target"))
        );

        let result = EmbodimentBindingResultV1::bound(
            target("org.luminous.simple-sim.v1"),
            EmbodimentInstanceId::new("fixture:1").unwrap(),
            "binding:fixture:1",
        )
        .unwrap();
        let mut identity = result.bound_identity().unwrap().unwrap();
        identity.instance_id = EmbodimentInstanceId::new("fixture:2").unwrap();
        assert_eq!(
            identity.validate(),
            Err(ProviderBindingValidationError::DigestMismatch("bound_identity"))
        );
    }

    #[test]
    fn malformed_identifiers_fail_closed() {
        for bad in ["", " padded", "padded ", "bad\nvalue"] {
            assert!(PlatformFamilyId::new(bad).is_err());
            assert!(BackendProviderId::new(bad).is_err());
        }
    }
}
