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
    ServiceActiveStateV1, ServiceLoadStateV1, ServiceUnitFileStateV1,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const EFFECT_DIGEST_DOMAIN_V1: &[u8] = b"nixward-service-effect-v1";
const UNIT_DEFINITION_DOMAIN_V1: &[u8] = b"nixward-systemd-unit-definition-v1";
const POST_STATE_RECEIPT_DOMAIN_V1: &[u8] = b"nixward-post-state-receipt-v1";
const STABILITY_SAMPLE_DOMAIN_V1: &[u8] = b"nixward-post-state-stability-sample-v1";
const STABILITY_SEQUENCE_DOMAIN_V1: &[u8] = b"nixward-post-state-stability-sequence-v1";
const INVOCATION_ID_HEX_LEN: usize = 32;
const MAX_PATH_BYTES: usize = 4096;
const SYSTEMD_UNIT_PATH_PREFIX: &str = "/org/freedesktop/systemd1/unit/";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixSystemdJobTypeV1 {
    Start,
    Stop,
    Restart,
    Reload,
}

impl NixSystemdJobTypeV1 {
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

/// Observer-sealed post-state observation token.
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
        observation.validate_shape()?;
        Ok(Self { observation })
    }

    pub(crate) fn as_ref(&self) -> &NixServicePostStateObservationV1 {
        &self.observation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdJobEvidenceV1 {
    pub id: u32,
    pub job_type: NixSystemdJobTypeV1,
    /// Canonical unit name carried by systemd's JobRemoved signal.
    pub unit: String,
    /// Unique D-Bus owner of org.freedesktop.systemd1 for this job epoch.
    ///
    /// Unique names are connection-scoped and never change owner, so retaining
    /// this value prevents a durable receipt from collapsing two systemd
    /// manager incarnations that happen to reuse other job identifiers.
    pub manager_owner: String,
    /// Job object path returned by systemd.
    pub object_path: String,
    /// systemd JobRemoved result. Only the exact `done` value is accepted as
    /// successful evidence; unknown future vocabulary remains recordable but
    /// cannot satisfy the proof predicate.
    pub result: String,
}

impl NixSystemdJobEvidenceV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        if self.id == 0 {
            return Err(NixPostStateErrorV1::InvalidJobId);
        }
        NixServiceOperationV1::new(self.unit.clone(), NixServiceOperationKindV1::Start)
            .map_err(|_| NixPostStateErrorV1::InvalidJobUnit)?;
        if self.unit != self.unit.trim() {
            return Err(NixPostStateErrorV1::InvalidJobUnit);
        }
        validate_unique_manager_owner(&self.manager_owner)?;
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
    /// Observer-sealed BLAKE3 commitment over the referenced definition bytes.
    pub authorized_definition_content_digest: String,
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
        validate_digest(
            &self.authorized_definition_content_digest,
            "authorized definition content digest",
        )?;
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
            &self.authorized_definition_content_digest,
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
    /// Exact systemd Unit object identity used for the observation.
    pub unit_object_path: String,
    pub definition_identity: NixSystemdUnitDefinitionIdentityV1,
    /// Observer-sealed BLAKE3 commitment over the definition bytes captured for this observation.
    pub definition_content_digest: String,
    pub load_state: ServiceLoadStateV1,
    pub active_state: ServiceActiveStateV1,
    pub sub_state: String,
    pub unit_file_state: ServiceUnitFileStateV1,
    /// systemd Service.Result from the same observation snapshot.
    pub service_result: String,
    pub systemd_job: Option<NixSystemdJobEvidenceV1>,
    /// Unique D-Bus owner of org.freedesktop.systemd1 for this observation.
    pub systemd_manager_owner: Option<String>,
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
        validate_systemd_unit_object_path(&self.unit_object_path)?;
        require_nonempty(&self.sub_state, "observed service sub-state")?;
        require_nonempty(&self.service_result, "observed service result")?;
        self.definition_identity.validate_shape()?;
        validate_digest(
            &self.definition_content_digest,
            "observed definition content digest",
        )?;
        if let Some(owner) = self.systemd_manager_owner.as_deref() {
            validate_unique_manager_owner(owner)?;
        }
        if let Some(job) = &self.systemd_job {
            job.validate_shape()?;
            if self.systemd_manager_owner.as_deref() != Some(job.manager_owner.as_str()) {
                return Err(NixPostStateErrorV1::ManagerOwnerMismatch);
            }
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

    pub fn state_digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        Ok(semantic_state_digest(
            self.operation,
            &self.unit,
            self.load_state,
            self.active_state,
            &self.sub_state,
            self.unit_file_state,
            &self.service_result,
            self.invocation_id.as_deref(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixPostStateStabilitySampleV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub unit_object_path: String,
    pub observed_generation: u64,
    pub definition_digest: String,
    /// Observer-sealed byte-content commitment for the same sample.
    pub definition_content_digest: String,
    pub state_digest: String,
    pub manager_owner: String,
    pub invocation_id: Option<String>,
    pub state_change_at_monotonic_us: u64,
    pub captured_at_monotonic_us: u64,
}

impl NixPostStateStabilitySampleV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        NixServiceOperationV1::new(self.unit.clone(), self.operation)
            .map_err(|_| NixPostStateErrorV1::InvalidServiceUnit)?;
        if self.observed_generation == 0 {
            return Err(NixPostStateErrorV1::InvalidGeneration);
        }
        validate_systemd_unit_object_path(&self.unit_object_path)?;
        validate_digest(&self.definition_digest, "stability definition digest")?;
        validate_digest(
            &self.definition_content_digest,
            "stability definition content digest",
        )?;
        validate_digest(&self.state_digest, "stability state digest")?;
        validate_unique_manager_owner(&self.manager_owner)?;
        validate_optional_invocation_id(self.invocation_id.as_deref(), "stability invocation id")?;
        if self.captured_at_monotonic_us < self.state_change_at_monotonic_us {
            return Err(NixPostStateErrorV1::ObservationBeforeStateChange);
        }
        Ok(())
    }

    pub(crate) fn digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(STABILITY_SAMPLE_DOMAIN_V1);
        put_u8(&mut h, operation_tag(self.operation));
        put_str(&mut h, &self.unit);
        put_str(&mut h, &self.unit_object_path);
        put_u64(&mut h, self.observed_generation);
        put_str(&mut h, &self.definition_digest);
        put_str(&mut h, &self.definition_content_digest);
        put_str(&mut h, &self.state_digest);
        put_str(&mut h, &self.manager_owner);
        put_opt_str(&mut h, self.invocation_id.as_deref());
        put_u64(&mut h, self.state_change_at_monotonic_us);
        put_u64(&mut h, self.captured_at_monotonic_us);
        Ok(h.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixPostStateStabilityEvidenceV1 {
    pub required_window_us: u64,
    pub window_start_monotonic_us: u64,
    pub window_end_monotonic_us: u64,
    pub samples: Vec<NixPostStateStabilitySampleV1>,
    pub sequence_digest: String,
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
        if self.samples.len() < 2 {
            return Err(NixPostStateErrorV1::InsufficientStabilitySamples);
        }
        if self.samples.len() > 64 {
            return Err(NixPostStateErrorV1::TooManyStabilitySamples);
        }
        validate_digest(&self.sequence_digest, "stability sequence digest")?;

        for sample in &self.samples {
            sample.validate_shape()?;
            if sample.captured_at_monotonic_us < self.window_start_monotonic_us
                || sample.captured_at_monotonic_us > self.window_end_monotonic_us
            {
                return Err(NixPostStateErrorV1::StabilitySampleOutsideWindow);
            }
            if sample.state_change_at_monotonic_us > self.window_start_monotonic_us {
                return Err(NixPostStateErrorV1::StateChangedDuringStabilityWindow);
            }
        }

        for pair in self.samples.windows(2) {
            if pair[1].captured_at_monotonic_us <= pair[0].captured_at_monotonic_us {
                return Err(NixPostStateErrorV1::StabilitySamplesNotIncreasing);
            }
        }

        let first = &self.samples[0];
        for sample in &self.samples[1..] {
            if sample.operation != first.operation
                || sample.unit != first.unit
                || sample.unit_object_path != first.unit_object_path
                || sample.observed_generation != first.observed_generation
                || sample.definition_digest != first.definition_digest
                || sample.state_digest != first.state_digest
                || sample.manager_owner != first.manager_owner
                || sample.invocation_id != first.invocation_id
                || sample.state_change_at_monotonic_us != first.state_change_at_monotonic_us
            {
                return Err(NixPostStateErrorV1::StabilityIdentityOrStateChanged);
            }
        }

        let expected = stability_sequence_digest(&self.samples)?;
        if self.sequence_digest != expected {
            return Err(NixPostStateErrorV1::StabilitySequenceDigestMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixPostStateErrorV1> {
        self.validate_shape()?;
        Ok(self.sequence_digest.clone())
    }
}

/// Observer-sealed stability evidence. Serialized stability data must cross the
/// observer boundary before it may be used to promote a receipt to Proven.
pub struct NixVerifiedPostStateStabilityEvidenceV1 {
    evidence: NixPostStateStabilityEvidenceV1,
}

impl NixVerifiedPostStateStabilityEvidenceV1 {
    pub(crate) fn from_observer(
        evidence: NixPostStateStabilityEvidenceV1,
    ) -> Result<Self, NixPostStateErrorV1> {
        evidence.validate_shape()?;
        Ok(Self { evidence })
    }

    pub(crate) fn as_ref(&self) -> &NixPostStateStabilityEvidenceV1 {
        &self.evidence
    }
}

pub(crate) fn stability_sequence_digest(
    samples: &[NixPostStateStabilitySampleV1],
) -> Result<String, NixPostStateErrorV1> {
    let mut h = Hasher::new();
    h.update(STABILITY_SEQUENCE_DOMAIN_V1);
    put_u32(&mut h, u32::try_from(samples.len()).map_err(|_| {
        NixPostStateErrorV1::TooManyStabilitySamples
    })?);
    for sample in samples {
        put_str(&mut h, &sample.digest()?);
    }
    Ok(h.finalize().to_hex().to_string())
}

fn validate_stability_against_observation(
    stability: &NixPostStateStabilityEvidenceV1,
    observation: &NixServicePostStateObservationV1,
) -> Result<(), NixPostStateErrorV1> {
    let last = stability
        .samples
        .last()
        .ok_or(NixPostStateErrorV1::InsufficientStabilitySamples)?;

    if last.operation != observation.operation
        || last.unit != observation.unit
        || last.unit_object_path != observation.unit_object_path
        || last.observed_generation != observation.observed_generation
        || last.definition_digest != observation.definition_digest()?
        || last.definition_content_digest != observation.definition_content_digest
        || last.state_digest != observation.state_digest()?
        || last.manager_owner != observation
            .systemd_manager_owner
            .as_deref()
            .ok_or(NixPostStateErrorV1::MissingManagerOwner)?
        || last.invocation_id != observation.invocation_id
        || last.state_change_at_monotonic_us != observation.state_change_at_monotonic_us
        || last.captured_at_monotonic_us > observation.observed_at_monotonic_us
    {
        return Err(NixPostStateErrorV1::StabilityIdentityOrStateChanged);
    }
    Ok(())
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
    /// Observer-sealed byte-content commitment bound to the authorized Service intent.
    pub authorized_definition_content_digest: String,
    /// Exact systemd FragmentPath + DropInPaths identity from the observed unit.
    pub observed_definition_identity: NixSystemdUnitDefinitionIdentityV1,
    pub observed_definition_digest: String,
    /// Observer-sealed byte-content commitment from the post-state observation.
    pub observed_definition_content_digest: String,
    pub operation: NixServiceOperationKindV1,
    pub systemd_job_id: Option<u32>,
    pub systemd_job_type: Option<NixSystemdJobTypeV1>,
    pub systemd_job_unit: Option<String>,
    pub systemd_job_object_path: Option<String>,
    pub systemd_job_result: Option<String>,
    /// Exact systemd Unit object path corresponding to the persisted semantic state.
    pub observed_unit_object_path: String,
    pub observed_load_state: ServiceLoadStateV1,
    pub observed_active_state: ServiceActiveStateV1,
    pub observed_sub_state: String,
    pub observed_unit_file_state: ServiceUnitFileStateV1,
    pub observed_service_result: String,
    /// Unique D-Bus owner of systemd1 for the observed service-manager epoch.
    pub systemd_manager_owner: String,
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
        stability: Option<&NixVerifiedPostStateStabilityEvidenceV1>,
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
        authorization
            .validate_against_intent(intent)
            .map_err(|error| match error {
                super::authorization::NixAuthorizationErrorV1::MissingServiceEffectContext => {
                    NixPostStateErrorV1::MissingServiceEffectContext
                }
                super::authorization::NixAuthorizationErrorV1::ServiceEffectContextMismatch => {
                    NixPostStateErrorV1::ServiceEffectContextMismatch
                }
                _ => NixPostStateErrorV1::AuthorizationIntentMismatch,
            })?;
        match &intent.action {
            NixActionDescriptorV1::Service { operation, unit }
                if *operation == expectation.operation && unit == &expectation.unit => {}
            _ => return Err(NixPostStateErrorV1::IntentEffectMismatch),
        }
        expectation.validate_shape()?;
        validate_expectation_against_intent(intent, expectation)?;
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
        let observed_definition_content_digest = observation.definition_content_digest.clone();
        if expectation.authorized_definition_content_digest != observed_definition_content_digest {
            return Err(NixPostStateErrorV1::DefinitionContentMismatch);
        }
        let manager_owner = observation
            .systemd_manager_owner
            .clone()
            .ok_or(NixPostStateErrorV1::MissingManagerOwner)?;

        if let Some(stability) = stability {
            let stability = stability.as_ref();
            stability.validate_shape()?;
            if stability.window_end_monotonic_us > observation.observed_at_monotonic_us {
                return Err(NixPostStateErrorV1::StabilityAfterObservation);
            }
            validate_stability_against_observation(stability, observation)?;
        }

        let assessment = evaluate_postcondition(expectation, observation)?;
        let claim = match assessment {
            NixPostconditionAssessmentV1::Satisfied => {
                if expectation.required_stability_us == 0 {
                    NixPostStateClaimV1::Observed
                } else {
                    match stability.map(NixVerifiedPostStateStabilityEvidenceV1::as_ref) {
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

        let (
            systemd_job_id,
            systemd_job_type,
            systemd_job_unit,
            systemd_job_object_path,
            systemd_job_result,
        ) = match &observation.systemd_job {
            Some(job) => (
                Some(job.id),
                Some(job.job_type),
                Some(job.unit.clone()),
                Some(job.object_path.clone()),
                Some(job.result.clone()),
            ),
            None => (None, None, None, None, None),
        };

        let stability = stability.map(|value| value.as_ref().clone());

        let receipt = Self {
            action_intent_digest,
            authorization_record_digest,
            effect_digest: expectation.effect_digest()?,
            target_unit: expectation.unit.clone(),
            authorized_generation: expectation.authorized_generation,
            observed_generation: observation.observed_generation,
            authorized_definition_digest: expectation.authorized_definition_digest.clone(),
            authorized_definition_content_digest: expectation.authorized_definition_content_digest.clone(),
            observed_definition_identity: observation.definition_identity.clone(),
            observed_definition_digest,
            observed_definition_content_digest,
            operation: expectation.operation,
            systemd_job_id,
            systemd_job_type,
            systemd_job_unit,
            systemd_job_object_path,
            systemd_job_result,
            observed_unit_object_path: observation.unit_object_path.clone(),
            observed_load_state: observation.load_state,
            observed_active_state: observation.active_state,
            observed_sub_state: observation.sub_state.clone(),
            observed_unit_file_state: observation.unit_file_state,
            observed_service_result: observation.service_result.clone(),
            systemd_manager_owner: manager_owner,
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

    /// Re-bind a serialized receipt to the trusted typed intent and authorization record.
    ///
    /// Self-consistency of a receipt is insufficient: the referenced records
    /// are the trust anchors and must be checked independently.
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
        authorization
            .validate_against_intent(intent)
            .map_err(|error| match error {
                super::authorization::NixAuthorizationErrorV1::MissingServiceEffectContext => {
                    NixPostStateErrorV1::MissingServiceEffectContext
                }
                super::authorization::NixAuthorizationErrorV1::ServiceEffectContextMismatch => {
                    NixPostStateErrorV1::ServiceEffectContextMismatch
                }
                _ => NixPostStateErrorV1::InvalidBoundAuthorization,
            })?;

        let rebound_expectation = NixServicePostStateExpectationV1 {
            operation: self.operation,
            unit: self.target_unit.clone(),
            authorized_generation: self.authorized_generation,
            authorized_definition_digest: self.authorized_definition_digest.clone(),
            authorized_definition_content_digest: self.authorized_definition_content_digest.clone(),
            pre_invocation_id: self.pre_invocation_id.clone(),
            required_stability_us: self.required_stability_us,
        };
        validate_expectation_against_intent(intent, &rebound_expectation)?;
        Ok(())
    }

    fn evaluate_postcondition_from_receipt(
        &self,
    ) -> Result<NixPostconditionAssessmentV1, NixPostStateErrorV1> {
        let expected_job_type = NixSystemdJobTypeV1::for_operation(self.operation);
        match self.operation {
            NixServiceOperationKindV1::Start
            | NixServiceOperationKindV1::Stop
            | NixServiceOperationKindV1::Restart
            | NixServiceOperationKindV1::Reload => {
                let (
                    Some(job_id),
                    Some(job_type),
                    Some(job_unit),
                    Some(job_object_path),
                    Some(job_result),
                ) = (
                    self.systemd_job_id,
                    self.systemd_job_type,
                    self.systemd_job_unit.as_deref(),
                    self.systemd_job_object_path.as_deref(),
                    self.systemd_job_result.as_deref(),
                )
                else {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                };

                if job_id == 0
                    || job_unit != self.target_unit
                    || job_object_path != format!(
                        "/org/freedesktop/systemd1/job/{job_id}"
                    )
                    || Some(job_type) != expected_job_type
                {
                    return Ok(NixPostconditionAssessmentV1::Violated);
                }
                if job_result != "done" {
                    return Ok(NixPostconditionAssessmentV1::Violated);
                }
            }
            NixServiceOperationKindV1::Enable | NixServiceOperationKindV1::Disable => {}
        }

        match self.operation {
            NixServiceOperationKindV1::Start => {
                if self.observed_active_state != ServiceActiveStateV1::Active {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
            NixServiceOperationKindV1::Stop => {
                if self.observed_active_state != ServiceActiveStateV1::Inactive {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
            NixServiceOperationKindV1::Restart => {
                if self.observed_active_state != ServiceActiveStateV1::Active {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
                let (Some(pre), Some(post)) = (
                    self.pre_invocation_id.as_deref(),
                    self.post_invocation_id.as_deref(),
                ) else {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                };
                if pre == post {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
            NixServiceOperationKindV1::Reload => {
                if self.observed_active_state != ServiceActiveStateV1::Active {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
            NixServiceOperationKindV1::Enable => {
                if self.observed_unit_file_state != ServiceUnitFileStateV1::Enabled {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
            NixServiceOperationKindV1::Disable => {
                if self.observed_unit_file_state != ServiceUnitFileStateV1::Disabled {
                    return Ok(NixPostconditionAssessmentV1::Unproven);
                }
            }
        }

        Ok(NixPostconditionAssessmentV1::Satisfied)
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
            &self.authorized_definition_content_digest,
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
        self.observed_definition_identity.validate_shape()?;
        let recomputed_definition_digest = self
            .observed_definition_identity
            .digest(&self.target_unit)?;
        if self.observed_definition_digest != recomputed_definition_digest
            || self.authorized_definition_digest != self.observed_definition_digest
        {
            return Err(NixPostStateErrorV1::DefinitionMismatch);
        }
        validate_digest(
            &self.authorized_definition_content_digest,
            "authorized definition content digest",
        )?;
        validate_digest(
            &self.observed_definition_content_digest,
            "observed definition content digest",
        )?;
        if self.authorized_definition_content_digest != self.observed_definition_content_digest {
            return Err(NixPostStateErrorV1::DefinitionContentMismatch);
        }
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
        validate_systemd_unit_object_path(&self.observed_unit_object_path)?;
        require_nonempty(&self.observed_sub_state, "observed service sub-state")?;
        require_nonempty(&self.observed_service_result, "observed service result")?;
        validate_unique_manager_owner(&self.systemd_manager_owner)?;
        let recomputed_state_digest = semantic_state_digest(
            self.operation,
            &self.target_unit,
            self.observed_load_state,
            self.observed_active_state,
            &self.observed_sub_state,
            self.observed_unit_file_state,
            &self.observed_service_result,
            self.post_invocation_id.as_deref(),
        );
        let recomputed_postcondition = self.evaluate_postcondition_from_receipt()?;
        if self.postcondition != recomputed_postcondition {
            return Err(NixPostStateErrorV1::PostconditionMismatch);
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
            let last = stability
                .samples
                .last()
                .ok_or(NixPostStateErrorV1::InsufficientStabilitySamples)?;
            if last.operation != self.operation
                || last.unit != self.target_unit
                || last.unit_object_path != self.observed_unit_object_path
                || last.observed_generation != self.observed_generation
                || last.definition_digest != self.observed_definition_digest
                || last.definition_content_digest != self.observed_definition_content_digest
                || last.state_digest != recomputed_state_digest
                || last.manager_owner != self.systemd_manager_owner
                || last.invocation_id != self.post_invocation_id
            {
                return Err(NixPostStateErrorV1::StabilityIdentityOrStateChanged);
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
        put_str(&mut h, &self.authorized_definition_content_digest);
        put_str(&mut h, &self.observed_definition_digest);
        put_str(&mut h, &self.observed_definition_content_digest);
        put_str(&mut h, &self.observed_definition_identity.fragment_path);
        put_str_vec(&mut h, &self.observed_definition_identity.drop_in_paths);
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
        put_str(&mut h, &self.observed_unit_object_path);
        put_u8(&mut h, load_state_tag(self.observed_load_state));
        put_u8(&mut h, active_state_tag(self.observed_active_state));
        put_str(&mut h, &self.observed_sub_state);
        put_u8(&mut h, unit_file_state_tag(self.observed_unit_file_state));
        put_str(&mut h, &self.observed_service_result);
        put_str(&mut h, &self.systemd_manager_owner);
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
                put_str(&mut h, &stability.sequence_digest);
            }
            None => put_u8(&mut h, 0),
        }

        put_str(&mut h, &self.observer_identity);
        put_str(&mut h, &self.observer_version);
        Ok(h.finalize().to_hex().to_string())
    }
}

fn validate_expectation_against_intent(
    intent: &NixActionIntentV1,
    expectation: &NixServicePostStateExpectationV1,
) -> Result<(), NixPostStateErrorV1> {
    match &intent.action {
        NixActionDescriptorV1::Service { operation, unit }
            if *operation == expectation.operation && unit == &expectation.unit => {}
        _ => return Err(NixPostStateErrorV1::IntentEffectMismatch),
    }

    let Some(pre_state_identity) = intent.pre_state_identity.as_deref() else {
        return Err(NixPostStateErrorV1::MissingBoundPreState);
    };

    let Some(context) = intent.service_effect_context.as_ref() else {
        return Err(NixPostStateErrorV1::MissingServiceEffectContext);
    };

    if context.operation != expectation.operation || context.unit != expectation.unit {
        return Err(NixPostStateErrorV1::ServiceEffectContextMismatch);
    }
    if context.authorized_generation != expectation.authorized_generation {
        return Err(NixPostStateErrorV1::GenerationMismatch);
    }
    if context.authorized_definition_digest != expectation.authorized_definition_digest {
        return Err(NixPostStateErrorV1::DefinitionMismatch);
    }
    if context.authorized_definition_content_digest
        != expectation.authorized_definition_content_digest
    {
        return Err(NixPostStateErrorV1::DefinitionContentMismatch);
    }
    if context.pre_invocation_id != expectation.pre_invocation_id {
        return Err(NixPostStateErrorV1::InvocationMismatch);
    }
    if context.required_stability_us != expectation.required_stability_us {
        return Err(NixPostStateErrorV1::StabilityContractMismatch);
    }

    let prefix = "nixward-service-pre-state-v1|generation=";
    let Some(rest) = pre_state_identity.strip_prefix(prefix) else {
        return Err(NixPostStateErrorV1::MissingServiceEffectContext);
    };
    let (generation, rest) = rest
        .split_once("|unit=")
        .ok_or(NixPostStateErrorV1::InvalidBoundPreState)?;
    let generation = generation
        .parse::<u64>()
        .map_err(|_| NixPostStateErrorV1::InvalidBoundPreState)?;
    let (unit, state) = rest
        .split_once("|state=")
        .ok_or(NixPostStateErrorV1::InvalidBoundPreState)?;

    if generation != expectation.authorized_generation {
        return Err(NixPostStateErrorV1::GenerationMismatch);
    }
    if unit != expectation.unit {
        return Err(NixPostStateErrorV1::UnitMismatch);
    }
    if state != context.pre_state_digest {
        return Err(NixPostStateErrorV1::PreStateDigestMismatch);
    }
    context
        .validate_shape()
        .map_err(|error| NixPostStateErrorV1::InvalidServiceEffectContext(error.to_string()))?;
    let context_digest = context
        .digest()
        .map_err(|error| NixPostStateErrorV1::InvalidServiceEffectContext(error.to_string()))?;
    if context_digest.is_empty() {
        return Err(NixPostStateErrorV1::InvalidServiceEffectContext(
            "empty service effect context digest".to_string(),
        ));
    }
    Ok(())
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
    authorized_definition_content_digest: &str,
    pre_invocation_id: Option<&str>,
    required_stability_us: u64,
) -> String {
    let mut h = Hasher::new();
    h.update(EFFECT_DIGEST_DOMAIN_V1);
    h.update(&[operation_tag(operation)]);
    put_str(&mut h, unit);
    put_u64(&mut h, authorized_generation);
    put_str(&mut h, authorized_definition_digest);
    put_str(&mut h, authorized_definition_content_digest);
    put_opt_str(&mut h, pre_invocation_id);
    put_u64(&mut h, required_stability_us);
    h.finalize().to_hex().to_string()
}

fn semantic_state_digest(
    operation: NixServiceOperationKindV1,
    unit: &str,
    load_state: ServiceLoadStateV1,
    active_state: ServiceActiveStateV1,
    sub_state: &str,
    unit_file_state: ServiceUnitFileStateV1,
    service_result: &str,
    invocation_id: Option<&str>,
) -> String {
    let mut h = Hasher::new();
    h.update(STABILITY_SAMPLE_DOMAIN_V1);
    put_u8(&mut h, operation_tag(operation));
    put_str(&mut h, unit);
    put_u8(&mut h, load_state_tag(load_state));
    put_u8(&mut h, active_state_tag(active_state));
    put_str(&mut h, sub_state);
    put_u8(&mut h, unit_file_state_tag(unit_file_state));
    put_str(&mut h, service_result);
    put_opt_str(&mut h, invocation_id);
    h.finalize().to_hex().to_string()
}

fn load_state_tag(state: ServiceLoadStateV1) -> u8 {
    match state {
        ServiceLoadStateV1::Stub => 0,
        ServiceLoadStateV1::Loaded => 1,
        ServiceLoadStateV1::NotFound => 2,
        ServiceLoadStateV1::BadSetting => 3,
        ServiceLoadStateV1::Error => 4,
        ServiceLoadStateV1::Merged => 5,
        ServiceLoadStateV1::Masked => 6,
    }
}

fn active_state_tag(state: ServiceActiveStateV1) -> u8 {
    match state {
        ServiceActiveStateV1::Active => 0,
        ServiceActiveStateV1::Reloading => 1,
        ServiceActiveStateV1::Inactive => 2,
        ServiceActiveStateV1::Failed => 3,
        ServiceActiveStateV1::Activating => 4,
        ServiceActiveStateV1::Deactivating => 5,
        ServiceActiveStateV1::Maintenance => 6,
        ServiceActiveStateV1::Refreshing => 7,
    }
}

fn unit_file_state_tag(state: ServiceUnitFileStateV1) -> u8 {
    match state {
        ServiceUnitFileStateV1::Enabled => 0,
        ServiceUnitFileStateV1::EnabledRuntime => 1,
        ServiceUnitFileStateV1::Linked => 2,
        ServiceUnitFileStateV1::LinkedRuntime => 3,
        ServiceUnitFileStateV1::Alias => 4,
        ServiceUnitFileStateV1::Masked => 5,
        ServiceUnitFileStateV1::MaskedRuntime => 6,
        ServiceUnitFileStateV1::Static => 7,
        ServiceUnitFileStateV1::Disabled => 8,
        ServiceUnitFileStateV1::Indirect => 9,
        ServiceUnitFileStateV1::Generated => 10,
        ServiceUnitFileStateV1::Transient => 11,
        ServiceUnitFileStateV1::Bad => 12,
    }
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

fn validate_unique_manager_owner(value: &str) -> Result<(), NixPostStateErrorV1> {
    // D-Bus unique connection names begin with ':' and contain at least two
    // non-empty dot-separated elements. Their maximum name length is 255.
    if value.is_empty() || value.len() > 255 || !value.starts_with(':') {
        return Err(NixPostStateErrorV1::InvalidManagerOwner);
    }
    let mut elements = value[1..].split('.');
    let first = elements.next().unwrap_or_default();
    if first.is_empty() || elements.next().is_none() {
        return Err(NixPostStateErrorV1::InvalidManagerOwner);
    }
    for element in std::iter::once(first).chain(elements) {
        if element.is_empty()
            || !element
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err(NixPostStateErrorV1::InvalidManagerOwner);
        }
    }
    Ok(())
}

fn validate_systemd_unit_object_path(value: &str) -> Result<(), NixPostStateErrorV1> {
    if value.is_empty()
        || value.len() > MAX_PATH_BYTES
        || !value.starts_with(SYSTEMD_UNIT_PATH_PREFIX)
        || value.ends_with('/')
    {
        return Err(NixPostStateErrorV1::InvalidPath("systemd unit object path"));
    }
    let suffix = &value[SYSTEMD_UNIT_PATH_PREFIX.len()..];
    if suffix.is_empty()
        || suffix.split('/').any(|element| {
            element.is_empty()
                || !element
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
    {
        return Err(NixPostStateErrorV1::InvalidPath("systemd unit object path"));
    }
    Ok(())
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
    #[error("invalid systemd manager unique owner")]
    InvalidManagerOwner,
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
    #[error("stability sample falls outside the declared window")]
    StabilitySampleOutsideWindow,
    #[error("stability samples are not strictly increasing")]
    StabilitySamplesNotIncreasing,
    #[error("stability sample identity or semantic state changed")]
    StabilityIdentityOrStateChanged,
    #[error("stability sequence digest does not match the samples")]
    StabilitySequenceDigestMismatch,
    #[error("too many stability samples")]
    TooManyStabilitySamples,
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
    #[error("service invocation identity does not match the authorized effect")]
    InvocationMismatch,
    #[error("stability contract does not match the authorized effect")]
    StabilityContractMismatch,
    #[error("pre-state digest does not match the authorized effect context")]
    PreStateDigestMismatch,
    #[error("missing authority-bound service effect context")]
    MissingServiceEffectContext,
    #[error("invalid service effect context: {0}")]
    InvalidServiceEffectContext(String),
    #[error("observed generation does not match authorized generation")]
    GenerationMismatch,
    #[error("observed systemd unit-definition identity does not match authorization")]
    DefinitionMismatch,
    #[error("observed definition content does not match the authorization-bound commitment")]
    DefinitionContentMismatch,
    #[error("systemd manager incarnation is missing")]
    MissingManagerOwner,
    #[error("systemd manager incarnation does not match the observation/job binding")]
    ManagerOwnerMismatch,
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
    #[error("serialized postcondition does not match its persisted semantic state")]
    PostconditionMismatch,
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
    #[error("service effect has no bound pre-state identity")]
    MissingBoundPreState,
    #[error("bound pre-state identity is malformed")]
    InvalidBoundPreState,
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
            authorized_definition_content_digest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            pre_invocation_id: Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
            required_stability_us: 0,
        }
    }


    fn contextual_intent(
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
        definition_digest: String,
        pre_invocation_id: Option<String>,
        required_stability_us: u64,
    ) -> NixActionIntentV1 {
        NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: Some(format!(
                "nixward-service-pre-state-v1|generation={}|unit={}|state={}",
                generation,
                unit,
                "1111111111111111111111111111111111111111111111111111111111111111"
            )),
            action: NixActionDescriptorV1::Service {
                operation,
                unit: unit.to_string(),
            },
            service_effect_context: Some(
                super::authorization::NixServiceEffectContextV1::new(
                    operation,
                    unit.to_string(),
                    generation,
                    "1111111111111111111111111111111111111111111111111111111111111111",
                    definition_digest,
                    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    pre_invocation_id,
                    required_stability_us,
                )
                .unwrap(),
            ),
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        }
    }

    fn contextual_authorization(
        intent: &NixActionIntentV1,
    ) -> NixExecutionAuthorizationRecordV1 {
        NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest().unwrap(),
            service_effect_context_digest: intent
                .service_effect_context
                .as_ref()
                .map(|context| context.digest().unwrap()),
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".to_string(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        }
    }

    fn build_receipt(
        exp: &NixServicePostStateExpectationV1,
        obs: &NixServicePostStateObservationV1,
        stability: Option<NixPostStateStabilityEvidenceV1>,
    ) -> Result<NixPostStateReceiptV1, NixPostStateErrorV1> {
        let verified = NixVerifiedPostStateObservationV1::from_observer(obs.clone())?;
        let verified_stability = stability.map(|evidence| {
            NixVerifiedPostStateStabilityEvidenceV1 { evidence }
        });

        use super::super::authorization::{
            NixActionIntentV1, NixAuthorizationProfileV1, NixActionScopeV1,
        };
        let intent = NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: Some(format!(
                "nixward-service-pre-state-v1|generation={}|unit={}|state={}",
                exp.authorized_generation,
                exp.unit,
                "1111111111111111111111111111111111111111111111111111111111111111"
            )),
            action: NixActionDescriptorV1::Service {
                operation: exp.operation,
                unit: exp.unit.clone(),
            },
            service_effect_context: Some(
                super::authorization::NixServiceEffectContextV1::new(
                    exp.operation,
                    exp.unit.clone(),
                    exp.authorized_generation,
                    "1111111111111111111111111111111111111111111111111111111111111111",
                    exp.authorized_definition_digest.clone(),
                    exp.authorized_definition_content_digest.clone(),
                    exp.pre_invocation_id.clone(),
                    exp.required_stability_us,
                )
                .unwrap(),
            ),
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        let authorization = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest().unwrap(),
            service_effect_context_digest: Some(
                intent
                    .service_effect_context
                    .as_ref()
                    .unwrap()
                    .digest()
                    .unwrap(),
            ),
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
            verified_stability.as_ref(),
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
            unit_object_path: "/org/freedesktop/systemd1/unit/nginx_2eservice".to_string(),
            definition_identity: definition(),
            definition_content_digest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            load_state: ServiceLoadStateV1::Loaded,
            active_state,
            sub_state: "running".to_string(),
            unit_file_state,
            service_result: "success".to_string(),
            systemd_job: NixSystemdJobTypeV1::for_operation(operation).map(|job_type| {
                NixSystemdJobEvidenceV1 {
                    id: 7,
                    job_type,
                    unit: "nginx.service".to_string(),
                    object_path: "/org/freedesktop/systemd1/job/7".to_string(),
                    result: "done".to_string(),
                    manager_owner: ":1.123".to_string(),
                }
            }),
            systemd_manager_owner: Some(":1.123".to_string()),
            invocation_id: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()),
            state_change_at_monotonic_us: 900,
            observed_at_monotonic_us: 2_000,
        }
    }

    fn stability(
        obs: &NixServicePostStateObservationV1,
        required_window_us: u64,
        window_start_monotonic_us: u64,
        window_end_monotonic_us: u64,
        captured_at: &[u64],
    ) -> NixPostStateStabilityEvidenceV1 {
        let samples = captured_at
            .iter()
            .map(|captured_at_monotonic_us| NixPostStateStabilitySampleV1 {
                operation: obs.operation,
                unit: obs.unit.clone(),
                unit_object_path: obs.unit_object_path.clone(),
                observed_generation: obs.observed_generation,
                definition_digest: obs.definition_digest().unwrap(),
                definition_content_digest: obs.definition_content_digest.clone(),
                state_digest: obs.state_digest().unwrap(),
                manager_owner: obs.systemd_manager_owner.clone().unwrap(),
                invocation_id: obs.invocation_id.clone(),
                state_change_at_monotonic_us: obs.state_change_at_monotonic_us,
                captured_at_monotonic_us: *captured_at_monotonic_us,
            })
            .collect::<Vec<_>>();
        let sequence_digest = stability_sequence_digest(&samples).unwrap();
        NixPostStateStabilityEvidenceV1 {
            required_window_us,
            window_start_monotonic_us: window_start_monotonic_us,
            window_end_monotonic_us: window_end_monotonic_us,
            samples,
            sequence_digest,
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
            Some(stability(
                &obs,
                1_000,
                1_000,
                2_000,
                &[1_000, 2_000],
            )),
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
            Some(stability(
                &observation(
                    NixServiceOperationKindV1::Start,
                    ServiceActiveStateV1::Active,
                    ServiceUnitFileStateV1::Enabled,
                ),
                1_000,
                1_500,
                2_000,
                &[1_500, 2_000],
            )),
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
        changed.systemd_manager_owner = ":1.124".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observed_active_state = ServiceActiveStateV1::Failed;
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observed_load_state = ServiceLoadStateV1::Masked;
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observed_service_result = "exit-code".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observed_unit_object_path =
            "/org/freedesktop/systemd1/unit/sshd_2eservice".into();
        variants.push(changed);

        let mut changed = receipt.clone();
        changed.observer_version = "2".into();
        variants.push(changed);

        for variant in variants {
            assert_ne!(baseline, variant.digest().unwrap());
        }
    }

    #[test]
    fn malformed_systemd_manager_owner_fails_closed() {
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_job.as_mut().unwrap().manager_owner = ":not-valid".to_string();
        assert_eq!(
            obs.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidManagerOwner
        );

        obs.systemd_job.as_mut().unwrap().manager_owner = "org.example".to_string();
        assert_eq!(
            obs.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidManagerOwner
        );

        obs.systemd_job.as_mut().unwrap().manager_owner = ":1".to_string();
        assert_eq!(
            obs.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidManagerOwner
        );

        obs.systemd_job.as_mut().unwrap().manager_owner = ":1.2.3".to_string();
        obs.validate_shape().unwrap();
    }

    #[test]
    fn receipt_requires_observation_manager_incarnation() {
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_manager_owner = None;
        assert_eq!(
            build_receipt(
                &expectation(NixServiceOperationKindV1::Start),
                &obs,
                None,
            ).unwrap_err(),
            NixPostStateErrorV1::MissingManagerOwner
        );
    }

    #[test]
    fn receipt_manager_incarnation_is_not_optional_for_enablement() {
        let exp = expectation(NixServiceOperationKindV1::Enable);
        let obs = observation(
            NixServiceOperationKindV1::Enable,
            ServiceActiveStateV1::Inactive,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(&exp, &obs, None).unwrap();
        receipt.systemd_manager_owner = "org.freedesktop.systemd1".into();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::InvalidManagerOwner
        );
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
    fn stability_sequence_rejects_non_increasing_capture_times() {
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        evidence.samples[1].captured_at_monotonic_us = 1_000;
        evidence.sequence_digest = stability_sequence_digest(&evidence.samples).unwrap();
        assert_eq!(
            evidence.validate_shape().unwrap_err(),
            NixPostStateErrorV1::StabilitySamplesNotIncreasing
        );
    }

    #[test]
    fn stability_sequence_detects_state_identity_change_even_when_digest_is_recomputed() {
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        evidence.samples[1].state_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        evidence.sequence_digest = stability_sequence_digest(&evidence.samples).unwrap();
        assert_eq!(
            evidence.validate_shape().unwrap_err(),
            NixPostStateErrorV1::StabilityIdentityOrStateChanged
        );
    }

    #[test]
    fn stability_sequence_rejects_sequence_digest_tampering() {
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        evidence.sequence_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert_eq!(
            evidence.validate_shape().unwrap_err(),
            NixPostStateErrorV1::StabilitySequenceDigestMismatch
        );
    }

    #[test]
    fn stability_sequence_cannot_be_rebound_to_another_manager_epoch() {
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        evidence.samples[1].manager_owner = ":1.124".into();
        evidence.sequence_digest = stability_sequence_digest(&evidence.samples).unwrap();
        assert_eq!(
            evidence.validate_shape().unwrap_err(),
            NixPostStateErrorV1::StabilityIdentityOrStateChanged
        );
    }

    #[test]
    fn receipt_rejects_stability_from_a_different_unit_object() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        evidence.samples[1].unit_object_path =
            "/org/freedesktop/systemd1/unit/sshd_2eservice".into();
        evidence.sequence_digest = stability_sequence_digest(&evidence.samples).unwrap();
        let result = build_receipt(&exp, &obs, Some(evidence));
        assert_eq!(
            result.unwrap_err(),
            NixPostStateErrorV1::StabilityIdentityOrStateChanged
        );
    }

    #[test]
    fn stable_sequence_is_committed_into_receipt_digest() {
        let exp = {
            let mut value = expectation(NixServiceOperationKindV1::Start);
            value.required_stability_us = 1_000;
            value
        };
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        let receipt_a = build_receipt(&exp, &obs, Some(evidence.clone())).unwrap();
        let mut evidence_changed = evidence;
        evidence_changed.samples[1].captured_at_monotonic_us = 2_100;
        evidence_changed.window_end_monotonic_us = 2_100;
        evidence_changed.sequence_digest =
            stability_sequence_digest(&evidence_changed.samples).unwrap();
        let receipt_b = build_receipt(&exp, &obs, Some(evidence_changed)).unwrap();
        assert_ne!(receipt_a.digest().unwrap(), receipt_b.digest().unwrap());
    }

    #[test]
    fn serialized_receipt_rejects_definition_identity_mutation() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(&exp, &obs, None).unwrap();
        receipt.observed_definition_identity.fragment_path =
            "/nix/store/changed.service".into();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::DefinitionMismatch
        );
    }

    #[test]
    fn serialized_receipt_rejects_recommitted_definition_against_authorized_digest() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(&exp, &obs, None).unwrap();
        receipt.observed_definition_identity =
            NixSystemdUnitDefinitionIdentityV1::new("/nix/store/changed.service", vec![])
                .unwrap();
        receipt.observed_definition_digest = receipt
            .observed_definition_identity
            .digest(&receipt.target_unit)
            .unwrap();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::DefinitionMismatch
        );
    }

    #[test]
    fn serialized_receipt_definition_identity_is_part_of_receipt_digest() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let receipt = build_receipt(&exp, &obs, None).unwrap();
        let baseline = receipt.digest().unwrap();
        let mut changed = receipt.clone();
        changed.observed_definition_identity.drop_in_paths =
            vec!["/nix/store/changed.conf".into()];
        changed.observed_definition_digest = changed
            .observed_definition_identity
            .digest(&changed.target_unit)
            .unwrap();
        assert_ne!(baseline, changed.digest().unwrap());
    }

    #[test]
    fn serialized_receipt_rejects_definition_digest_mismatch() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(&exp, &obs, None).unwrap();
        receipt.observed_definition_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::DefinitionMismatch
        );
    }

    #[test]
    fn serialized_claim_recomputes_postcondition_from_persisted_state() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let receipt = build_receipt(&exp, &obs, None).unwrap();
        assert_eq!(
            receipt.postcondition,
            NixPostconditionAssessmentV1::Satisfied
        );

        let mut forged = receipt.clone();
        forged.observed_active_state = ServiceActiveStateV1::Inactive;
        assert_eq!(
            forged.validate_shape().unwrap_err(),
            NixPostStateErrorV1::PostconditionMismatch
        );
    }

    #[test]
    fn serialized_receipt_rejects_inconsistent_postcondition_label() {
        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let mut receipt = build_receipt(&exp, &obs, None).unwrap();
        receipt.postcondition = NixPostconditionAssessmentV1::Violated;
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::PostconditionMismatch
        );
    }

    #[test]
    fn serialized_proven_claim_rejects_forged_state_digest() {
        let exp = {
            let mut value = expectation(NixServiceOperationKindV1::Start);
            value.required_stability_us = 1_000;
            value
        };
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let evidence = stability(&obs, 1_000, 1_000, 2_000, &[1_000, 2_000]);
        let mut receipt = build_receipt(&exp, &obs, Some(evidence)).unwrap();
        receipt.stability.as_mut().unwrap().samples[0].state_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        receipt.stability.as_mut().unwrap().samples[1].state_digest =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        receipt.stability.as_mut().unwrap().sequence_digest =
            stability_sequence_digest(&receipt.stability.as_ref().unwrap().samples).unwrap();
        assert_eq!(
            receipt.validate_shape().unwrap_err(),
            NixPostStateErrorV1::StabilityIdentityOrStateChanged
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
            Some(stability(
                &obs,
                1_000,
                1_000,
                2_000,
                &[1_000, 2_000],
            )),
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
            Some(stability(
                &obs,
                1,
                1_000,
                2_000,
                &[1_000, 2_000],
            )),
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
        let wrong_intent = contextual_intent(
            NixServiceOperationKindV1::Restart,
            "nginx.service",
            42,
            definition().digest("nginx.service").unwrap(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            0,
        );
        let authorization = contextual_authorization(&wrong_intent);

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

        let intent = contextual_intent(
            NixServiceOperationKindV1::Start,
            "nginx.service",
            42,
            definition().digest("nginx.service").unwrap(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            0,
        );
        let authorization = contextual_authorization(&intent);

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
    fn expectation_generation_must_match_authorized_intent_pre_state() {
        let mut exp = expectation(NixServiceOperationKindV1::Start);
        exp.authorized_generation = 43;
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        assert_eq!(
            build_receipt(&exp, &obs, None).unwrap_err(),
            NixPostStateErrorV1::GenerationMismatch
        );
    }

    #[test]
    fn missing_pre_state_binding_is_not_acceptable_for_service_receipts() {
        use super::super::authorization::{
            NixActionIntentV1, NixActionScopeV1, NixAuthorizationProfileV1,
        };

        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let verified = NixVerifiedPostStateObservationV1::from_observer(obs).unwrap();
        let intent = NixActionIntentV1 {
            subject_identity: "host:test".to_string(),
            pre_state_identity: None,
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
            service_effect_context_digest: None,
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".to_string(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        };
        assert_eq!(
            NixPostStateReceiptV1::build(
                &intent,
                &authorization,
                &exp,
                &verified,
                None,
                "systemd-observer-v1",
                "1",
            )
            .unwrap_err(),
            NixPostStateErrorV1::MissingServiceEffectContext
        );
    }

    #[test]
    fn trusted_verifier_rejects_recomputed_generation_tampering() {
        use super::super::authorization::{
            NixActionIntentV1, NixActionScopeV1, NixAuthorizationProfileV1,
        };

        let exp = expectation(NixServiceOperationKindV1::Start);
        let obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        let receipt = build_receipt(&exp, &obs, None).unwrap();

        let intent = contextual_intent(
            NixServiceOperationKindV1::Start,
            "nginx.service",
            42,
            definition().digest("nginx.service").unwrap(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            0,
        );
        let authorization = contextual_authorization(&intent);

        let mut tampered = receipt;
        tampered.authorized_generation = 43;
        tampered.observed_generation = 43;
        tampered.effect_digest = service_effect_digest(
            tampered.operation,
            &tampered.target_unit,
            tampered.authorized_generation,
            &tampered.authorized_definition_digest,
            tampered.pre_invocation_id.as_deref(),
            tampered.required_stability_us,
        );

        assert_eq!(
            tampered.verify_against(&intent, &authorization).unwrap_err(),
            NixPostStateErrorV1::GenerationMismatch
        );
    }

    #[test]
    fn wrong_job_unit_is_not_proven_even_with_matching_job_type() {
        let mut obs = observation(
            NixServiceOperationKindV1::Start,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
        );
        obs.systemd_job.as_mut().unwrap().unit = "sshd.service".to_string();

        let receipt = build_receipt(
            &expectation(NixServiceOperationKindV1::Start),
            &obs,
            None,
        )
        .unwrap();

        assert_eq!(
            receipt.postcondition,
            NixPostconditionAssessmentV1::Violated
        );
        assert_eq!(receipt.claim, NixPostStateClaimV1::Violated);
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
