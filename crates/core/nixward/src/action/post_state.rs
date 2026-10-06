// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Effect-bound post-state evidence for governed Nixward service operations.
//!
//! This module is deliberately evidence/proof-oriented:
//!
//! authority -> exact typed effect -> observed outcome -> postcondition claim
//!
//! It never mints execution authority and never treats Phi, confidence, or an
//! observation as permission to execute.
//!
//! The current implementation is transport-neutral. A later systemd adapter
//! can populate these observations from a read-only D-Bus observer without
//! changing the receipt or proof semantics.

use super::authorization::{
    NixActionDescriptorV1, NixActionIntentV1, NixAuthorizationDecisionV1,
    NixExecutionAuthorizationRecordV1,
};
use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use super::service_state::{
    ServiceActiveStateV1, ServiceUnitFileStateV1,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const EFFECT_DIGEST_DOMAIN_V1: &[u8] = b"nixward-service-effect-v1";
const UNIT_DEFINITION_DOMAIN_V1: &[u8] = b"nixward-systemd-unit-definition-v1";
const POST_STATE_RECEIPT_DOMAIN_V1: &[u8] = b"nixward-post-state-receipt-v1";
const INVOCATION_ID_HEX_LEN: usize = 32;
const MAX_PATH_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixSystemdJobTypeV1 {
    Start,
    Stop,
    Restart,
    Reload,
}

/// Unforgeable-at-API-boundary observation token.
///
/// The underlying observation remains data, but receipt construction accepts only
/// this observer-produced wrapper. There is intentionally no public constructor
/// or serde implementation; the checked-in transport layer is responsible for
/// producing it.
pub struct NixVerifiedPostStateObservationV1 {
    observation: NixServicePostStateObservationV1,
}

impl NixVerifiedPostStateObservationV1 {
    pub(crate) fn from_observer(
        observation: NixServicePostStateObservationV1,
    ) -> Result<Self, NixPostStateErrorV1> {
        observation.as_ref().validate_shape()?;
        Ok(Self { observation })
    }

    fn as_ref(&self) -> &NixServicePostStateObservationV1 {
        &self.observation
    }
}

    pub fn for_operation(operation: NixServiceOperationKindV1) -> Option<Self> {
        match operation {
            NixServiceOperationKindV1::Start => Some(Self::Start),
            NixServiceOperationKindV1::Stop => Some(Self::Stop),
            NixServiceOperationKindV1::Restart => Some(Self::Restart),
            NixServiceOperationKindV1::Reload => Some(Self::Reload),
            NixServiceOperationKindV1::Enable | NixServiceOperationKindV1::Disable => None,
        }
    }

    fn discriminant(self) -> u8 {
        match self {
            Self::Start => 0,
            Self::Stop => 1,
            Self::Restart => 2,
            Self::Reload => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdJobEvidenceV1 {
    pub id: u32,
    pub job_type: NixSystemdJobTypeV1,
    /// Canonical unit name carried by systemd's JobRemoved signal.
    pub unit: String,
    /// Job object path returned by systemd.
    pub object_path: String,
    /// systemd JobRemoved result. Only the exact `done` value is accepted as
    /// successful evidence; unknown future vocabulary remains recordable but
    /// cannot satisfy the proof predicate.
    pub result: String,
}

impl NixSystemdJobEvidenceV1 {
    /// Re-bind a serialized receipt to the trusted typed intent and authorization record.
    ///
    /// This is the verifier-side complement to `build`: self-consistency of a
    /// receipt digest is not enough because an attacker could otherwise replace
    /// the referenced intent/auth digests and present the mutated receipt to a
    /// consumer that does not dereference those records.
    pub fn verify_against(
        &self,
        intent: &NixActionIntentV1,
        authorization: &NixExecutionAuthorizationRecordV1,
    ) -> Result<(), NixPostStateErrorV1> {
        self.validate_shape()?;

        let intent_digest = intent
            .digest()
            .map_err(|_| NixPostStateErrorV1::InvalidBoundIntent)?;
        let authorization_digest = authorization
            .digest()
            .map_err(|_| NixPostStateErrorV1::InvalidBoundAuthorization)?;

        if authorization.decision != NixAuthorizationDecisionV1::Approved {
            return Err(NixPostStateErrorV1::AuthorizationNotApproved);
        }
        if self.action_intent_digest != intent_digest {
            return Err(NixPostStateErrorV1::AuthorizationIntentMismatch);
        }
        if self.authorization_record_digest != authorization_digest {
            return Err(NixPostStateErrorV1::AuthorizationRecordMismatch);
        }
        if authorization.action_intent_digest != intent_digest {
            return Err(NixPostStateErrorV1::AuthorizationIntentMismatch);
        }

        match &intent.action {
            NixActionDescriptorV1::Service { operation, unit }
                if *operation == self.operation && unit == &self.target_unit => {}
            _ => return Err(NixPostStateErrorV1::IntentEffectMismatch),
        }
        Ok(())
    }

    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        if self.id == 0 {
            return Err(NixPostStateErrorV1::InvalidJobId);
        }
        NixServiceOperationV1::new(self.unit.clone(), NixServiceOperationKindV1::Start)
            .map_err(|_| NixPostStateErrorV1::InvalidJobUnit)?;
        if self.unit != self.unit.trim() {
            return Err(NixPostStateErrorV1::InvalidJobUnit);
        }
        require_nonempty(&self.object_path, "systemd job object path")?;
        if !self.object_path.starts_with("/org/freedesktop/systemd1/job/") {
            return Err(NixPostStateErrorV1::InvalidJobObjectPath);
        }
        if !self.object_path.ends_with(&format!("/{}", self.id)) {
            return Err(NixPostStateErrorV1::InvalidJobObjectPath);
        }
        require_nonempty(&self.result, "systemd job result")?;
        Ok(())
    }

    pub fn succeeded(&self) -> bool {
        self.result == "done"
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdUnitDefinitionIdentityV1 {
    /// The exact FragmentPath reported for the observed unit.
    pub fragment_path: String,
    /// The exact DropInPaths set reported for the observed unit, normalized
    /// deterministically. This is source identity, not a content hash.
    pub drop_in_paths: Vec<String>,
}

impl NixSystemdUnitDefinitionIdentityV1 {
    pub fn new(
        fragment_path: impl Into<String>,
        drop_in_paths: Vec<String>,
    ) -> Result<Self, NixPostStateErrorV1> {
        let value = Self {
            fragment_path: fragment_path.into(),
            drop_in_paths,
        };
        value.validate_shape()?;
        Ok(value)
    }

    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        validate_absolute_path(&self.fragment_path, "fragment path")?;

        let mut normalized = self.drop_in_paths.clone();
        for path in &normalized {
            validate_absolute_path(path, "drop-in path")?;
        }
        normalized.sort();
        normalized.dedup();
        if normalized.len() != self.drop_in_paths.len() {
            return Err(NixPostStateErrorV1::DuplicateDropInPath);
        }
        if normalized != self.drop_in_paths {
            return Err(NixPostStateErrorV1::UnsortedDropInPaths);
        }
        Ok(())
    }

    pub fn digest(&self, unit: &str) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        require_nonempty(unit, "service unit")?;

        let mut h = Hasher::new();
        h.update(UNIT_DEFINITION_DOMAIN_V1);
        put_str(&mut h, unit);
        put_str(&mut h, &self.fragment_path);
        put_str_vec(&mut h, &self.drop_in_paths);
        Ok(h.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServicePostStateExpectationV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub authorized_generation: u64,
    /// Exact systemd unit-definition source identity captured before execution.
    pub authorized_definition_digest: String,
    /// Restart proof requires both pre- and post-invocation identities.
    pub pre_invocation_id: Option<String>,
    /// Zero disables stability as a claim requirement. Non-zero requires a
    /// stability evidence window of at least this duration.
    pub required_stability_us: u64,
}

impl NixServicePostStateExpectationV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        NixServiceOperationV1::new(self.unit.clone(), self.operation)
            .map_err(|_| NixPostStateErrorV1::InvalidServiceUnit)?;
        if self.authorized_generation == 0 {
            return Err(NixPostStateErrorV1::InvalidGeneration);
        }
        validate_digest(&self.authorized_definition_digest, "authorized definition digest")?;
        validate_optional_invocation_id(self.pre_invocation_id.as_deref(), "pre-invocation id")?;
        Ok(())
    }

    pub fn effect_digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        Ok(service_effect_digest(
            self.operation,
            &self.unit,
            self.authorized_generation,
            &self.authorized_definition_digest,
            self.pre_invocation_id.as_deref(),
            self.required_stability_us,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServicePostStateObservationV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub observed_generation: u64,
    pub definition_identity: NixSystemdUnitDefinitionIdentityV1,
    pub active_state: ServiceActiveStateV1,
    pub sub_state: String,
    pub unit_file_state: ServiceUnitFileStateV1,
    pub systemd_job: Option<NixSystemdJobEvidenceV1>,
    pub invocation_id: Option<String>,
    /// systemd StateChangeTimestampMonotonic represented as monotonic microseconds.
    pub state_change_at_monotonic_us: u64,
    pub observed_at_monotonic_us: u64,
}

impl NixServicePostStateObservationV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        NixServiceOperationV1::new(self.unit.clone(), self.operation)
            .map_err(|_| NixPostStateErrorV1::InvalidServiceUnit)?;
        if self.observed_generation == 0 {
            return Err(NixPostStateErrorV1::InvalidGeneration);
        }
        require_nonempty(&self.sub_state, "observed service sub-state")?;
        self.definition_identity.validate_shape()?;
        if let Some(job) = &self.systemd_job {
            job.validate_shape()?;
        }
        validate_optional_invocation_id(self.invocation_id.as_deref(), "post-invocation id")?;
        if self.observed_at_monotonic_us < self.state_change_at_monotonic_us {
            return Err(NixPostStateErrorV1::ObservationBeforeStateChange);
        }
        Ok(())
    }

    pub fn definition_digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.definition_identity.digest(&self.unit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixPostStateStabilityEvidenceV1 {
    /// Measured on the same monotonic clock as the systemd state-change timestamp.
    pub required_window_us: u64,
    pub window_start_monotonic_us: u64,
    pub window_end_monotonic_us: u64,
    pub last_state_change_at_monotonic_us: u64,
    pub sample_count: u32,
}

impl NixPostStateStabilityEvidenceV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        if self.required_window_us == 0 {
            return Err(NixPostStateErrorV1::InvalidStabilityWindow);
        }
        if self.window_end_monotonic_us < self.window_start_monotonic_us {
            return Err(NixPostStateErrorV1::InvalidStabilityWindow);
        }
        if self
            .window_end_monotonic_us
            .saturating_sub(self.window_start_monotonic_us)
            < self.required_window_us
        {
            return Err(NixPostStateErrorV1::StabilityWindowTooShort);
        }
        if self.sample_count < 2 {
            return Err(NixPostStateErrorV1::InsufficientStabilitySamples);
        }
        // If systemd's StateChangeTimestamp is at or before the beginning of
        // the stability window, the observed state has not changed since that
        // timestamp according to the unit's own state-change clock.
        if self.last_state_change_at_monotonic_us > self.window_start_monotonic_us {
            return Err(NixPostStateErrorV1::StateChangedDuringStabilityWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixPostconditionAssessmentV1 {
    Satisfied,
    Violated,
    Unproven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixPostStateClaimV1 {
    Observed,
    Proven,
    Violated,
    Unproven,
}

/// A cryptographically bound post-state receipt for one exact service effect.
///
/// The receipt is not authorization. It is a claim about an already-authorized
/// effect, and it is only marked `Proven` when the effect-specific predicate
/// and its required stability contract both pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixPostStateReceiptV1 {
    pub action_intent_digest: String,
    pub authorization_record_digest: String,
    pub effect_digest: String,
    pub target_unit: String,
    pub authorized_generation: u64,
    pub observed_generation: u64,
    pub authorized_definition_digest: String,
    pub observed_definition_digest: String,
    pub operation: NixServiceOperationKindV1,
    pub systemd_job_id: Option<u32>,
    pub systemd_job_type: Option<NixSystemdJobTypeV1>,
    pub systemd_job_unit: Option<String>,
    pub systemd_job_object_path: Option<String>,
    pub systemd_job_result: Option<String>,
    pub pre_invocation_id: Option<String>,
    pub post_invocation_id: Option<String>,
    pub postcondition: NixPostconditionAssessmentV1,
    pub claim: NixPostStateClaimV1,
    /// The stability contract that was part of the exact effect identity.
    /// Retaining it in the receipt makes a serialized receipt self-checking:
    /// a forged `Proven` claim cannot silently downgrade the required window.
    pub required_stability_us: u64,
    pub observed_at_monotonic_us: u64,
    pub stability: Option<NixPostStateStabilityEvidenceV1>,
    pub observer_identity: String,
    pub observer_version: String,
}

impl NixPostStateReceiptV1 {
    pub fn build(
        intent: &NixActionIntentV1,
        authorization: &NixExecutionAuthorizationRecordV1,
        expectation: &NixServicePostStateExpectationV1,
        observation: &NixVerifiedPostStateObservationV1,
        stability: Option<NixPostStateStabilityEvidenceV1>,
        observer_identity: impl Into<String>,
        observer_version: impl Into<String>,
    ) -> Result<Self, NixPostStateErrorV1> {
        let action_intent_digest = intent
            .digest()
            .map_err(|_| NixPostStateErrorV1::InvalidBoundIntent)?;
        let authorization_record_digest = authorization
            .digest()
            .map_err(|_| NixPostStateErrorV1::InvalidBoundAuthorization)?;

        if authorization.decision != NixAuthorizationDecisionV1::Approved {
            return Err(NixPostStateErrorV1::AuthorizationNotApproved);
        }
        if authorization.action_intent_digest != action_intent_digest {
            return Err(NixPostStateErrorV1::AuthorizationIntentMismatch);
        }
        match &intent.action {
            NixActionDescriptorV1::Service { operation, unit }
                if *operation == expectation.operation && unit == &expectation.unit => {}
            _ => return Err(NixPostStateErrorV1::IntentEffectMismatch),
        }
        expectation.validate_shape()?;
        observation.validate_shape()?;
        require_nonempty(&observer_identity, "observer identity")?;
        require_nonempty(&observer_version, "observer version")?;

        let observation = observation.as_ref();
        if expectation.operation != observation.operation {
            return Err(NixPostStateErrorV1::OperationMismatch);
        }
        if expectation.unit != observation.unit {
            return Err(NixPostStateErrorV1::UnitMismatch);
        }
        if expectation.authorized_generation != observation.observed_generation {
            return Err(NixPostStateErrorV1::GenerationMismatch);
        }

        let observed_definition_digest = observation.definition_digest()?;
        if expectation.authorized_definition_digest != observed_definition_digest {
            return Err(NixPostStateErrorV1::DefinitionMismatch);
        }

        if let Some(stability) = &stability {
            stability.validate_shape()?;
            if stability.window_end_monotonic_us > observation.observed_at_monotonic_us {
                return Err(NixPostStateErrorV1::StabilityAfterObservation);
            }
        }

        let assessment = evaluate_postcondition(expectation, observation)?;
        let claim = match assessment {
            NixPostconditionAssessmentV1::Satisfied => {
                if expectation.required_stability_us == 0 {
                    NixPostStateClaimV1::Observed
                } else {
                    match &stability {
                        Some(stability)
                            if stability.required_window_us >= expectation.required_stability_us =>
                        {
                            NixPostStateClaimV1::Proven
                        }
                        _ => NixPostStateClaimV1::Unproven,
                    }
                }
            }
            NixPostconditionAssessmentV1::Violated => NixPostStateClaimV1::Violated,
            NixPostconditionAssessmentV1::Unproven => NixPostStateClaimV1::Unproven,
        };

        let (systemd_job_id, systemd_job_type, systemd_job_unit, systemd_job_object_path, systemd_job_result) =
            match &observation.systemd_job {
                Some(job) => (
                    Some(job.id),
                    Some(job.job_type),
                    Some(job.unit.clone()),
                    Some(job.object_path.clone()),
                    Some(job.result.clone()),
                ),
                None => (None, None, None, None, None),
            };

        let receipt = Self {
            action_intent_digest,
            authorization_record_digest,
            effect_digest: expectation.effect_digest()?,
            target_unit: expectation.unit.clone(),
            authorized_generation: expectation.authorized_generation,
            observed_generation: observation.observed_generation,
            authorized_definition_digest: expectation.authorized_definition_digest.clone(),
            observed_definition_digest,
            operation: expectation.operation,
            systemd_job_id,
            systemd_job_type,
            systemd_job_unit,
            systemd_job_object_path,
            systemd_job_result,
            pre_invocation_id: expectation.pre_invocation_id.clone(),
            post_invocation_id: observation.invocation_id.clone(),
            postcondition: assessment,
            claim,
            required_stability_us: expectation.required_stability_us,
            observed_at_monotonic_us: observation.observed_at_monotonic_us,
            stability,
            observer_identity,
            observer_version,
        };
        receipt.validate_shape()?;
        Ok(receipt)
    }

    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        validate_digest(&self.action_intent_digest, "action intent digest")?;
        validate_digest(
            &self.authorization_record_digest,
            "authorization record digest",
        )?;
        validate_digest(&self.effect_digest, "effect digest")?;
        NixServiceOperationV1::new(self.target_unit.clone(), self.operation)
            .map_err(|_| NixPostStateErrorV1::InvalidServiceUnit)?;
        let expected_effect_digest = service_effect_digest(
            self.operation,
            &self.target_unit,
            self.authorized_generation,
            &self.authorized_definition_digest,
            self.pre_invocation_id.as_deref(),
            self.required_stability_us,
        );
        if self.effect_digest != expected_effect_digest {
            return Err(NixPostStateErrorV1::EffectDigestMismatch);
        }
        if self.authorized_generation == 0 || self.observed_generation == 0 {
            return Err(NixPostStateErrorV1::InvalidGeneration);
        }
        validate_digest(
            &self.authorized_definition_digest,
            "authorized definition digest",
        )?;
        validate_digest(&self.observed_definition_digest, "observed definition digest")?;
        if self.authorized_generation != self.observed_generation {
            return Err(NixPostStateErrorV1::GenerationMismatch);
        }
        if let Some(id) = self.systemd_job_id {
            if id == 0 {
                return Err(NixPostStateErrorV1::InvalidJobId);
            }
            if self.systemd_job_type.is_none()
                || self.systemd_job_unit.is_none()
                || self.systemd_job_object_path.is_none()
                || self.systemd_job_result.is_none()
            {
                return Err(NixPostStateErrorV1::IncompleteJobEvidence);
            }
        } else if self.systemd_job_type.is_some()
            || self.systemd_job_unit.is_some()
            || self.systemd_job_object_path.is_some()
            || self.systemd_job_result.is_some()
        {
            return Err(NixPostStateErrorV1::IncompleteJobEvidence);
        }
        if let Some(unit) = &self.systemd_job_unit {
            NixServiceOperationV1::new(unit.clone(), NixServiceOperationKindV1::Start)
                .map_err(|_| NixPostStateErrorV1::InvalidJobUnit)?;
        }
        if let Some(path) = &self.systemd_job_object_path {
            require_nonempty(path, "systemd job object path")?;
        }
        if let Some(result) = &self.systemd_job_result {
            require_nonempty(result, "systemd job result")?;
        }
        validate_optional_invocation_id(self.pre_invocation_id.as_deref(), "pre-invocation id")?;
        validate_optional_invocation_id(self.post_invocation_id.as_deref(), "post-invocation id")?;
        require_nonempty(&self.observer_identity, "observer identity")?;
        require_nonempty(&self.observer_version, "observer version")?;

        if let Some(stability) = &self.stability {
            stability.validate_shape()?;
            if stability.window_end_monotonic_us > self.observed_at_monotonic_us {
                return Err(NixPostStateErrorV1::StabilityAfterObservation);
            }
        }

        match self.claim {
            NixPostStateClaimV1::Proven => {
                if self.postcondition != NixPostconditionAssessmentV1::Satisfied {
                    return Err(NixPostStateErrorV1::InvalidClaim);
                }
                let Some(stability) = &self.stability else {
                    return Err(NixPostStateErrorV1::InvalidClaim);
                };
                if self.required_stability_us == 0
                    || stability.required_window_us < self.required_stability_us
                {
                    return Err(NixPostStateErrorV1::InvalidClaim);
                }
                if let Some(job_type) = self.systemd_job_type {
                    if Some(job_type)
                        != NixSystemdJobTypeV1::for_operation(self.operation)
                    {
                        return Err(NixPostStateErrorV1::InvalidClaim);
                    }
                }
                if let (Some(job_unit), Some(job_object_path), Some(job_id)) = (
                    self.systemd_job_unit.as_deref(),
                    self.systemd_job_object_path.as_deref(),
                    self.systemd_job_id,
                ) {
                    if job_unit != self.target_unit
                        || !job_object_path.ends_with(&format!("/{job_id}"))
                    {
                        return Err(NixPostStateErrorV1::InvalidClaim);
                    }
                } else {
                    return Err(NixPostStateErrorV1::InvalidClaim);
                }
                if matches!(
                    self.operation,
                    NixServiceOperationKindV1::Start
                        | NixServiceOperationKindV1::Stop
                        | NixServiceOperationKindV1::Restart
                        | NixServiceOperationKindV1::Reload
                ) && (self.systemd_job_id.is_none()
                    || self.systemd_job_type.is_none()
                    || self.systemd_job_result.as_deref() != Some("done"))
                {
                    return Err(NixPostStateErrorV1::InvalidClaim);
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(POST_STATE_RECEIPT_DOMAIN_V1);
        put_str(&mut h, &self.action_intent_digest);
        put_str(&mut h, &self.authorization_record_digest);
        put_str(&mut h, &self.effect_digest);
        put_str(&mut h, &self.target_unit);
        put_u64(&mut h, self.authorized_generation);
        put_u64(&mut h, self.observed_generation);
        put_str(&mut h, &self.authorized_definition_digest);
        put_str(&mut h, &self.observed_definition_digest);
        put_u8(&mut h, operation_tag(self.operation));
        put_opt_u32(&mut h, self.systemd_job_id);
        match self.systemd_job_type {
            Some(job_type) => {
                put_u8(&mut h, 1);
                put_u8(&mut h, job_type.discriminant());
            }
            None => put_u8(&mut h, 0),
        }
        put_opt_str(&mut h, self.systemd_job_unit.as_deref());
        put_opt_str(&mut h, self.systemd_job_object_path.as_deref());
        put_opt_str(&mut h, self.systemd_job_result.as_deref());
        put_opt_str(&mut h, self.pre_invocation_id.as_deref());
        put_opt_str(&mut h, self.post_invocation_id.as_deref());
        put_u8(&mut h, assessment_tag(self.postcondition));
        put_u8(&mut h, claim_tag(self.claim));
        put_u64(&mut h, self.required_stability_us);
        put_u64(&mut h, self.observed_at_monotonic_us);

        match &self.stability {
            Some(stability) => {
                put_u8(&mut h, 1);
                put_u64(&mut h, stability.required_window_us);
                put_u64(&mut h, stability.window_start_monotonic_us);
                put_u64(&mut h, stability.window_end_monotonic_us);
                put_u64(&mut h, stability.last_state_change_at_monotonic_us);
                put_u32(&mut h, stability.sample_count);
            }
            None => put_u8(&mut h, 0),
        }

        put_str(&mut h, &self.observer_identity);
        put_str(&mut h, &self.observer_version);
        Ok(h.finalize().to_hex().to_string())
    }
}

fn evaluate_postcondition(
    expectation: &NixServicePostStateExpectationV1,
    observation: &NixServicePostStateObservationV1,
) -> Result<NixPostconditionAssessmentV1, NixPostStateErrorV1> {
    let expected_job_type = NixSystemdJobTypeV1::for_operation(expectation.operation);
    match expectation.operation {
        NixServiceOperationKindV1::Start
        | NixServiceOperationKindV1::Stop
        | NixServiceOperationKindV1::Restart
        | NixServiceOperationKindV1::Reload => {
            let Some(job) = &observation.systemd_job else {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            };
            job.validate_shape()?;
            if job.unit != expectation.unit {
                return Ok(NixPostconditionAssessmentV1::Violated);
            }
            if Some(job.job_type) != expected_job_type {
                return Ok(NixPostconditionAssessmentV1::Violated);
            }
            if !job.succeeded() {
                return Ok(NixPostconditionAssessmentV1::Violated);
            }
        }
        NixServiceOperationKindV1::Enable | NixServiceOperationKindV1::Disable => {}
    }

    match expectation.operation {
        NixServiceOperationKindV1::Start => {
            if observation.active_state != ServiceActiveStateV1::Active {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
        NixServiceOperationKindV1::Stop => {
            if observation.active_state != ServiceActiveStateV1::Inactive {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
        NixServiceOperationKindV1::Restart => {
            if observation.active_state != ServiceActiveStateV1::Active {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
            let (Some(pre), Some(post)) = (
                expectation.pre_invocation_id.as_deref(),
                observation.invocation_id.as_deref(),
            ) else {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            };
            if pre == post {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
        NixServiceOperationKindV1::Reload => {
            if observation.active_state != ServiceActiveStateV1::Active {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
        NixServiceOperationKindV1::Enable => {
            if observation.unit_file_state != ServiceUnitFileStateV1::Enabled {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
        NixServiceOperationKindV1::Disable => {
            if observation.unit_file_state != ServiceUnitFileStateV1::Disabled {
                return Ok(NixPostconditionAssessmentV1::Unproven);
            }
        }
    }

    Ok(NixPostconditionAssessmentV1::Satisfied)
}

fn service_effect_digest(
    operation: NixServiceOperationKindV1,
    unit: &str,
    authorized_generation: u64,
    authorized_definition_digest: &str,
    pre_invocation_id: Option<&str>,
    required_stability_us: u64,
) -> String {
    let mut h = Hasher::new();
    h.update(EFFECT_DIGEST_DOMAIN_V1);
    h.update(&[operation_tag(operation)]);
    put_str(&mut h, unit);
    put_u64(&mut h, authorized_generation);
    put_str(&mut h, authorized_definition_digest);
    put_opt_str(&mut h, pre_invocation_id);
    put_u64(&mut h, required_stability_us);
    h.finalize().to_hex().to_string()
}

fn operation_tag(operation: NixServiceOperationKindV1) -> u8 {
    match operation {
        NixServiceOperationKindV1::Start => 0,
        NixServiceOperationKindV1::Stop => 1,
        NixServiceOperationKindV1::Restart => 2,
        NixServiceOperationKindV1::Reload => 3,
        NixServiceOperationKindV1::Enable => 4,
        NixServiceOperationKindV1::Disable => 5,
    }
}

fn assessment_tag(assessment: NixPostconditionAssessmentV1) -> u8 {
    match assessment {
        NixPostconditionAssessmentV1::Satisfied => 0,
        NixPostconditionAssessmentV1::Violated => 1,
        NixPostconditionAssessmentV1::Unproven => 2,
    }
}

fn claim_tag(claim: NixPostStateClaimV1) -> u8 {
    match claim {
        NixPostStateClaimV1::Observed => 0,
        NixPostStateClaimV1::Proven => 1,
        NixPostStateClaimV1::Violated => 2,
        NixPostStateClaimV1::Unproven => 3,
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), NixPostStateErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NixPostStateErrorV1::InvalidDigest(field));
    }
    Ok(())
}

fn validate_optional_invocation_id(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), NixPostStateErrorV1> {
    if let Some(value) = value {
        if value.len() != INVOCATION_ID_HEX_LEN
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(NixPostStateErrorV1::InvalidInvocationId(field));
        }
    }
    Ok(())
}

fn validate_absolute_path(value: &str, field: &'static str) -> Result<(), NixPostStateErrorV1> {
    require_nonempty(value, field)?;
    if value.len() > MAX_PATH_BYTES {
        return Err(NixPostStateErrorV1::PathTooLong);
    }
    if !value.starts_with('/') {
        return Err(NixPostStateErrorV1::NonAbsolutePath(field));
    }
    if value.contains('\0') {
        return Err(NixPostStateErrorV1::InvalidPath(field));
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), NixPostStateErrorV1> {
    if value.trim().is_empty() {
        Err(NixPostStateErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u32(h: &mut Hasher, value: u32) {
    h.update(&value.to_be_bytes());
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_str_vec(h: &mut Hasher, values: &[String]) {
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value);
    }
}

fn put_opt_str(h: &mut Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_str(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_opt_u32(h: &mut Hasher, value: Option<u32>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_u32(h, value);
        }
        None => put_u8(h, 0),
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixPostStateErrorV1 {
    #[error("empty required field: {0}")]
    EmptyField(&'static str),
    #[error("invalid digest: {0}")]
    InvalidDigest(&'static str),
    #[error("invalid invocation id: {0}")]
    InvalidInvocationId(&'static str),
    #[error("invalid generation")]
    InvalidGeneration,
    #[error("invalid job id")]
    InvalidJobId,
    #[error("invalid systemd job unit")]
    InvalidJobUnit,
    #[error("invalid systemd job object path")]
    InvalidJobObjectPath,
    #[error("invalid stability window")]
    InvalidStabilityWindow,
    #[error("stability window is too short")]
    StabilityWindowTooShort,
    #[error("insufficient stability samples")]
    InsufficientStabilitySamples,
    #[error("state changed during stability window")]
    StateChangedDuringStabilityWindow,
    #[error("observation predates the recorded state-change timestamp")]
    ObservationBeforeStateChange,
    #[error("stability evidence ends after the observation")]
    StabilityAfterObservation,
    #[error("service operation mismatch")]
    OperationMismatch,
    #[error("service unit mismatch")]
    UnitMismatch,
    #[error("observed generation does not match authorized generation")]
    GenerationMismatch,
    #[error("observed systemd unit-definition identity does not match authorization")]
    DefinitionMismatch,
    #[error("incomplete systemd job evidence")]
    IncompleteJobEvidence,
    #[error("duplicate drop-in path")]
    DuplicateDropInPath,
    #[error("drop-in paths are not deterministically sorted")]
    UnsortedDropInPaths,
    #[error("path is not absolute: {0}")]
    NonAbsolutePath(&'static str),
    #[error("invalid path: {0}")]
    InvalidPath(&'static str),
    #[error("path exceeds maximum length")]
    PathTooLong,
    #[error("invalid post-state claim")]
    InvalidClaim,
    #[error("invalid service unit")]
    InvalidServiceUnit,
    #[error("effect digest does not match its bound effect fields")]
    EffectDigestMismatch,
    #[error("bound action intent is invalid")]
    InvalidBoundIntent,
    #[error("bound authorization record is invalid")]
    InvalidBoundAuthorization,
    #[error("authorization record is not approved")]
    AuthorizationNotApproved,
    #[error("authorization record is bound to a different action intent")]
    AuthorizationIntentMismatch,
    #[error("receipt is bound to a different authorization record")]
    AuthorizationRecordMismatch,
    #[error("action intent does not describe the expected service effect")]
    IntentEffectMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> NixSystemdUnitDefinitionIdentityV1 {
        NixSystemdUnitDefinitionIdentityV1::new(
            "/nix/store/authorized-unit.service",
            vec![
                "/etc/systemd/system/nginx.service.d/override.conf".to_string(),
            ],
        )
        .unwrap()
    }

    fn expectation(operation: NixServiceOperationKindV1) -> NixServicePostStateExpectationV1 {
        NixServicePostStateExpectationV1 {
            operation,
            unit: "nginx.service".to_string(),
            authorized_generation: 42,
            authorized_definition_digest: definition().digest("nginx.service").unwrap(),
            pre_invocation_id: Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
            required_stability_us: 0,
        }
    }


    fn build_receipt(
        exp: &NixServicePostStateExpectationV1,
        obs: &NixServicePostStateObservationV1,
        stability: Option<NixPostStateStabilityEvidenceV1>,
    ) -> Result<NixPostStateReceiptV1, NixPostStateErrorV1> {
        let verified = NixVerifiedPostStateObservationV1::from_observer(obs.clone())?;

        use super::super::authorization::{
            NixActionIntentV1, NixAuthorizationProfileV1, NixActionScopeV1,
        };
        let intent = NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: Some(format!("generation:{}", exp.authorized_generation)),
            action: NixActionDescriptorV1::Service {
                operation: exp.operation,
                unit: exp.unit.clone(),
            },
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        let authorization = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest().unwrap(),
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".to_string(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        };
        NixPostStateReceiptV1::build(
            &intent,
            &authorization,
            exp,
            &verified,
            stability,
            "systemd-observer-v1",
            "1",
        )
    }

    fn observation(
        operation: NixServiceOperationKindV1,
        active_state: ServiceActiveStateV1,
        unit_file_state: ServiceUnitFileStateV1,
    ) -> NixServicePostStateObservationV1 {
        NixServicePostStateObservationV1 {
            operation,
            unit: "nginx.service".to_string(),
            observed_generation: 42,
            definition_identity: definition(),
            active_state,
            sub_state: "running".to_string(),
            unit_file_state,
            systemd_job: NixSystemdJobTypeV1::for_operation(operation).map(|job_type| {
                NixSystemdJobEvidenceV1 {
                    id: 7,
                    job_type,
                    unit: "nginx.service".to_string(),
                    object_path: "/org/freedesktop/systemd1/job/7".to_string(),
                    result: "done".to_string(),
                }
            }),
            invocation_id: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()),
            state_change_at_monotonic_us: 900,
            observed_at_monotonic_us: 2_000,
        }
    }

    #[test]
    fn start_success_is_observed_without_stability_claim() {
        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &observation(
                NixServiceOperationKindV1::Start,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();

        assert_eq!(receipt.postcondition, NixPostconditionAssessmentV1::Satisfied);
        assert_eq!(receipt.claim, NixPostStateClaimV1::Observed);
        assert!(receipt.digest().is_ok());
    }

    #[test]
    fn restart_requires_a_new_invocation_identity() {
        let exp = expectation(NixServiceOperationKindV1::Restart);
        let mut obs = observation(
            NixServiceOperationKindV1::Restart,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );

        obs.invocation_id = exp.pre_invocation_id.clone();
        let receipt = build_receipt(
            &exp,
            &obs,
            None,
        )
        .unwrap();

        assert_eq!(receipt.claim, NixPostStateClaimV1::Unproven);
    }

    #[test]
    fn restart_with_new_invocation_is_satisfied() {
        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Restart),
            &observation(
                NixServiceOperationKindV1::Restart,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();

        assert_eq!(receipt.postcondition, NixPostconditionAssessmentV1::Satisfied);
    }

    #[test]
    fn job_success_does_not_prove_wrong_post_state() {
        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &observation(
                NixServiceOperationKindV1::Start,
                ServiceActiveStateV1::Failed,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();

        assert_eq!(receipt.claim, NixPostStateClaimV1::Unproven);
    }

    #[test]
    fn wrong_job_type_is_not_proven() {
        let mut obs = observation(
            NixServiceOperationKindV1::Restart,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_job.as_mut().unwrap().job_type = NixSystemdJobTypeV1::Start;

        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Restart),
            &obs,
            None,
        )
        .unwrap();

        assert_eq!(receipt.postcondition, NixPostconditionAssessmentV1::Violated);
        assert_eq!(receipt.claim, NixPostStateClaimV1::Violated);
    }

    #[test]
    fn stable_claim_requires_window_and_unchanged_state_change_timestamp() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.required_stability_us = 1_000;

        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );

        let receipt = build_receipt(
            &exp,
            &obs,
            Some(NixPostStateStabilityEvidenceV1 {
                required_window_us: 1_000,
                window_start_monotonic_us: 1_000,
                window_end_monotonic_us: 2_000,
                last_state_change_at_monotonic_us: 900,
                sample_count: 2,
            }),
        )
        .unwrap();

        assert_eq!(receipt.claim, NixPostStateClaimV1::Proven);
    }

    #[test]
    fn short_stability_window_is_fail_closed() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.required_stability_us = 1_000;

        let result = build_receipt(
            &exp,
            &observation(
                NixServiceOperationKindV1::Start,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            Some(NixPostStateStabilityEvidenceV1 {
                required_window_us: 1_000,
                window_start_monotonic_us: 1_500,
                window_end_monotonic_us: 2_000,
                last_state_change_at_monotonic_us: 900,
                sample_count: 2,
            }),
        );

        assert_eq!(
            result.unwrap_err(),
            NixPostStateErrorV1::StabilityWindowTooShort
        );
    }

    #[test]
    fn generation_and_definition_mismatch_are_not_hidden_by_matching_service_state() {
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.observed_generation = 43;

        let result = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &obs,
            None,
        );

        assert_eq!(result.unwrap_err(), NixPostStateErrorV1::GenerationMismatch);

        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.definition_identity =
            NixSystemdUnitDefinitionIdentityV1::new("/nix/store/different.service", vec![])
                .unwrap();

        let result = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &obs,
            None,
        );

        assert_eq!(result.unwrap_err(), NixPostStateErrorV1::DefinitionMismatch);
    }

    #[test]
    fn receipt_digest_changes_on_intent_authority_target_generation_job_invocation_and_observer_mutation() {
        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Restart),
            &observation(
                NixServiceOperationKindV1::Restart,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();

        let baseline = receipt.digest().unwrap();

        let mut variants = Vec::new();

        let mut changed = receipt.clone();
        changed.action_intent_digest = "1111111111111111111111111111111111111111111111111111111111111111".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.authorization_record_digest = "2222222222222222222222222222222222222222222222222222222222222222".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.target_unit = "sshd.service".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.authorized_generation = 43;
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.systemd_job_id = Some(8);
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.post_invocation_id = Some("cccccccccccccccccccccccccccccccc".into());
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observer_version = "2".into();
        variants.push(changed);

        for variant in variants {
            assert_ne!(baseline, variant.digest().unwrap());
        }
    }

    #[test]
    fn zero_generation_and_malformed_invocation_ids_fail_closed() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.authorized_generation = 0;
        assert_eq!(
            exp.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidGeneration
        );

        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.invocation_id = Some("not-an-invocation".into());
        assert_eq!(
            obs.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidInvocationId("post-invocation id")
        );
    }

    #[test]
    fn serialized_proven_claim_cannot_reduce_or_remove_required_stability() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.required_stability_us = 1_000;
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(
            &exp,
            &obs,
            Some(NixPostStateStabilityEvidenceV1 {
                required_window_us: 1_000,
                window_start_monotonic_us: 1_000,
                window_end_monotonic_us: 2_000,
                last_state_change_at_monotonic_us: 900,
                sample_count: 2,
            }),
        )
        .unwrap();
        assert_eq!(receipt.claim, NixPostStateClaimV1::Proven);
        receipt.required_stability_us = 2_000;
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidClaim
        );
        receipt.required_stability_us = 1_000;
        receipt.stability = None;
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidClaim
        );
    }

    #[test]
    fn proven_lifecycle_claim_requires_done_job_evidence() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.required_stability_us = 1;
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_job.as_mut().unwrap().result = "running".to_string();
        let result = build_receipt(
            &exp,
            &obs,
            Some(NixPostStateStabilityEvidenceV1 {
                required_window_us: 1,
                window_start_monotonic_us: 1_000,
                window_end_monotonic_us: 2_000,
                last_state_change_at_monotonic_us: 900,
                sample_count: 2,
            }),
        );
        assert_eq!(result.unwrap_err(), NixPostStateErrorV1::Violated);
    }

    #[test]
    fn receipt_rejects_tampered_effect_digest_even_when_shape_remains_valid() {
        let mut receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &observation(
                NixServiceOperationKindV1::Start,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();
        receipt.effect_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::EffectDigestMismatch
        );
    }

    #[test]
    fn rejected_job_result_cannot_become_a_satisfied_postcondition() {
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_job.as_mut().unwrap().result = "failed".to_string();
        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &obs,
            None,
        )
        .unwrap();

        assert_eq!(receipt.postcondition, NixPostconditionAssessmentV1::Violated);
        assert_eq!(receipt.claim, NixPostStateClaimV1::Violated);
    }

    #[test]
    fn mismatched_typed_intent_is_rejected_even_when_observation_is_valid() {
        use super::super::authorization::{NixActionIntentV1, NixActionScopeV1, NixAuthorizationProfileV1};

        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let wrong_intent = NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: Some("generation:42".to_string()),
            action: NixActionDescriptorV1::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "nginx.service".to_string(),
            },
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        let authorization = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: wrong_intent.digest().unwrap(),
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".to_string(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        };

        assert_eq!(
            NixPostStateReceiptV1::build(
                &wrong_intent,
                &authorization,
                &exp,
                &obs,
                None,
                "systemd-observer-v1",
                "1",
            )
            .unwrap_err(),
            NixPostStateErrorV1::IntentEffectMismatch
        );
    }

    #[test]
    fn serialized_receipt_requires_trusted_intent_and_authorization_rebinding() {
        use super::super::authorization::{NixActionIntentV1, NixActionScopeV1, NixAuthorizationProfileV1};

        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &observation(
                NixServiceOperationKindV1::Start,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();

        let intent = NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: Some("generation:42".to_string()),
            action: NixActionDescriptorV1::Service {
                operation: NixServiceOperationKindV1::Start,
                unit: "nginx.service".to_string(),
            },
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        let authorization = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest().unwrap(),
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".to_string(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        };

        receipt.verify_against(&intent, &authorization).unwrap();

        let mut tampered = receipt.clone();
        tampered.action_intent_digest =
            "1111111111111111111111111111111111111111111111111111111111111111".into();
        assert_eq!(
            tampered.verify_against(&intent, &authorization).unwrap_err(),
            NixPostStateErrorV1::AuthorizationIntentMismatch
        );

        let mut tampered = receipt;
        tampered.authorization_record_digest =
            "2222222222222222222222222222222222222222222222222222222222222222".into();
        assert_eq!(
            tampered.verify_against(&intent, &authorization).unwrap_err(),
            NixPostStateErrorV1::AuthorizationRecordMismatch
        );
    }

    #[test]
    fn enable_and_disable_use_unit_file_state_not_lifecycle_state() {
        let enable = build_receipt(
            &expectation(NixServiceOperationKindV1::Enable),
            &observation(
                NixServiceOperationKindV1::Enable,
                ServiceActiveStateV1::Inactive,
                ServiceUnitFileStateV1::Enabled,
            ),
            None,
        )
        .unwrap();
        assert_eq!(enable.postcondition, NixPostconditionAssessmentV1::Satisfied);

        let disable = build_receipt(
            &expectation(NixServiceOperationKindV1::Disable),
            &observation(
                NixServiceOperationKindV1::Disable,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Disabled,
            ),
            None,
        )
        .unwrap();
        assert_eq!(disable.postcondition, NixPostconditionAssessmentV1::Satisfied);
    }
}
