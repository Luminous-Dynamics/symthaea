// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy composition for commissioning bound to an exact frozen observation.
//!
//! Ordinary commissioning admission proves one exact commissioning transition is
//! locally admissible. Frozen-observation admission independently proves one exact
//! observer transaction is locally admissible against a live configuration freeze.
//! The outer [`ObservedCommissioningAuthorization`] is still only raw evidence.
//!
//! This module closes the composition seam: success requires the outer artifact to
//! contain the exact commissioning transition and exact frozen observation already
//! represented by those two opaque policy witnesses. The returned value retains the
//! *outer* canonical bytes as the commissioning-authority authentication surface.
//!
//! This remains non-cryptographic and non-persistent. A production adapter must
//! additionally require independent Xenia verifier-owned proof for:
//! 1. the exact frozen-observation bytes under the configuration-observer root, and
//! 2. the exact outer observed-commissioning bytes under the commissioning root.

use crate::ConfigurationDigest;
use crate::admission::PolicyCheckedCommissioningAuthorization;
use crate::observed_authorization::{
    ObservedCommissioningAuthorization, ObservedCommissioningAuthorizationDigest,
    ObservedCommissioningAuthorizationError,
};
use symthaea_safety_qualification::frozen_observation_admission::
    PolicyCheckedFrozenSafetyConfigurationObservation;
use thiserror::Error;

/// Compose one outer observed-commissioning artifact with the exact two policy
/// witnesses that justify its nested commissioning and observation evidence.
pub fn admit_observed_commissioning_authorization(
    observed: &ObservedCommissioningAuthorization,
    checked_commissioning: &PolicyCheckedCommissioningAuthorization,
    checked_frozen_observation: &PolicyCheckedFrozenSafetyConfigurationObservation,
) -> Result<
    PolicyCheckedObservedCommissioningAuthorization,
    ObservedCommissioningAuthorizationAdmissionError,
> {
    observed.validate()?;

    // Structural equality is intentionally checked first: both nested types keep
    // private schema/provenance fields, so this rejects pairing a policy witness for
    // one object with a different-but-similar outer object before byte comparison.
    if observed.commissioning_transition() != checked_commissioning.transition() {
        return Err(
            ObservedCommissioningAuthorizationAdmissionError::CommissioningTransitionPolicyMismatch,
        );
    }
    if observed.frozen_observation() != checked_frozen_observation.frozen_observation() {
        return Err(
            ObservedCommissioningAuthorizationAdmissionError::FrozenObservationPolicyMismatch,
        );
    }

    // Retained canonical bytes are the actual cross-tool contracts. Compare them as
    // well so a future structural-equality refactor cannot weaken this boundary.
    let observed_transition_bytes = observed
        .commissioning_transition()
        .canonical_signing_bytes()
        .map_err(ObservedCommissioningAuthorizationError::from)?;
    if observed_transition_bytes.as_slice() != checked_commissioning.canonical_transition_bytes() {
        return Err(
            ObservedCommissioningAuthorizationAdmissionError::CommissioningTransitionBytesMismatch,
        );
    }

    let observed_frozen_bytes = observed
        .frozen_observation()
        .canonical_signing_bytes()
        .map_err(ObservedCommissioningAuthorizationError::from)?;
    if observed_frozen_bytes.as_slice()
        != checked_frozen_observation.canonical_frozen_observation_bytes()
    {
        return Err(
            ObservedCommissioningAuthorizationAdmissionError::FrozenObservationBytesMismatch,
        );
    }

    let commissioning_configuration = checked_commissioning.configuration_digest();
    let observed_configuration = checked_frozen_observation
        .checked_observation()
        .configuration_digest();
    if commissioning_configuration != observed_configuration {
        return Err(
            ObservedCommissioningAuthorizationAdmissionError::ConfigurationDigestMismatch {
                commissioning: commissioning_configuration,
                observation: observed_configuration,
            },
        );
    }

    Ok(PolicyCheckedObservedCommissioningAuthorization {
        observed: observed.clone(),
        canonical_observed_authorization_bytes: observed.canonical_signing_bytes()?,
        observed_authorization_digest: observed.observed_authorization_digest()?,
        checked_commissioning: checked_commissioning.clone(),
        checked_frozen_observation: checked_frozen_observation.clone(),
        configuration_digest: commissioning_configuration,
    })
}

/// Opaque non-serializable proof that one raw outer commissioning artifact contains
/// the exact commissioning and frozen-observation artifacts already accepted by the
/// two independent local policy layers.
///
/// This type proves no signature. The outer bytes are the commissioning-root
/// authentication surface; the nested frozen-observation witness separately retains
/// the observer-root authentication surface.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyCheckedObservedCommissioningAuthorization {
    observed: ObservedCommissioningAuthorization,
    canonical_observed_authorization_bytes: Vec<u8>,
    observed_authorization_digest: ObservedCommissioningAuthorizationDigest,
    checked_commissioning: PolicyCheckedCommissioningAuthorization,
    checked_frozen_observation: PolicyCheckedFrozenSafetyConfigurationObservation,
    configuration_digest: ConfigurationDigest,
}

impl PolicyCheckedObservedCommissioningAuthorization {
    pub fn observed(&self) -> &ObservedCommissioningAuthorization {
        &self.observed
    }

    pub fn canonical_observed_authorization_bytes(&self) -> &[u8] {
        &self.canonical_observed_authorization_bytes
    }

    pub fn observed_authorization_digest(&self) -> ObservedCommissioningAuthorizationDigest {
        self.observed_authorization_digest
    }

    pub fn checked_commissioning(&self) -> &PolicyCheckedCommissioningAuthorization {
        &self.checked_commissioning
    }

    pub fn checked_frozen_observation(
        &self,
    ) -> &PolicyCheckedFrozenSafetyConfigurationObservation {
        &self.checked_frozen_observation
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObservedCommissioningAuthorizationAdmissionError {
    #[error(transparent)]
    ObservedAuthorization(#[from] ObservedCommissioningAuthorizationError),
    #[error("outer observed commissioning contains a different commissioning transition than the policy witness")]
    CommissioningTransitionPolicyMismatch,
    #[error("outer observed commissioning contains a different frozen observation than the policy witness")]
    FrozenObservationPolicyMismatch,
    #[error("outer commissioning-transition bytes differ from the exact policy-retained bytes")]
    CommissioningTransitionBytesMismatch,
    #[error("outer frozen-observation bytes differ from the exact policy-retained bytes")]
    FrozenObservationBytesMismatch,
    #[error("commissioning and frozen-observation policy witnesses bind different configurations")]
    ConfigurationDigestMismatch {
        commissioning: ConfigurationDigest,
        observation: ConfigurationDigest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{
        CommissioningAuthorityRootSnapshot, CommissioningAuthorizationAdmissionPolicy,
        CommissioningAuthorizationHead,
    };
    use crate::authorization::transition::CommissioningAuthorizationTransition;
    use crate::authorization::{
        CommissioningAuthorizationSubject, CommissioningAuthorityRootDigest,
    };
    use crate::observed_authorization::ObservedCommissioningAuthorization;
    use crate::{CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord};
    use std::collections::BTreeSet;
    use symthaea_operating_autonomy::LocalSafetyEnvelope;
    use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
    use symthaea_resource_hierarchy::{NodeScale, ResourceHierarchy, ResourceNode};
    use symthaea_resource_model::{
        ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
    };
    use symthaea_safety_configuration::frozen_observation::
        SafetyConfigurationFrozenObservation;
    use symthaea_safety_configuration::observation::{
        ConfigurationObservationChallenge, ConfigurationObserverRootDigest,
        SafetyConfigurationObservation,
    };
    use symthaea_safety_configuration::state::{
        SafetyConfigurationFreezeToken, SafetyConfigurationState,
    };
    use symthaea_safety_configuration::{
        ConfigurationComponent, SafetyConfigurationManifest,
        SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::admission::SafetyProfileAuthorizationHead;
    use symthaea_safety_profile::authorization::{
        ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject,
    };
    use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
    use symthaea_safety_profile::transition::SafetyProfileAuthorizationTransition;
    use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };
    use symthaea_safety_qualification::frozen_observation_admission::{
        PolicyCheckedFrozenSafetyConfigurationObservation,
        admit_frozen_safety_configuration_observation,
    };
    use symthaea_safety_qualification::observation_admission::{
        ConfigurationObserverRootSnapshot, SafetyConfigurationObservationAdmissionPolicy,
        SafetyConfigurationObservationHead,
    };
    use symthaea_safety_qualification::{
        QualifiedSafetyConfiguration, qualify_safety_configuration,
    };

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn commissioning_root(byte: u8) -> CommissioningAuthorityRootDigest {
        CommissioningAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile_root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn observer_root(byte: u8) -> ConfigurationObserverRootDigest {
        ConfigurationObserverRootDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
    }

    fn clock() -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, 1_550, 1_600).unwrap()
    }

    fn power_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn local() -> LocalSafetyEnvelope {
        let mut modes = BTreeSet::new();
        modes.insert(OperatingMode::Normal);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: "rack".into(),
            allowed_modes: modes,
            resource_constraints: vec![ResourceConstraint {
                key: power_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(100.0),
            }],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn qualified() -> QualifiedSafetyConfiguration {
        let profile = profile();
        qualify_safety_configuration(
            &profile,
            SafetyConfigurationManifest {
                schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
                node_id: "rack".to_owned(),
                profile_id: profile.profile_id.clone(),
                profile_digest: profile.digest().unwrap(),
                hardware_inventory: ConfigurationComponent::Digest(digest(0x01)),
                firmware: ConfigurationComponent::Digest(digest(0x02)),
                software_closure: ConfigurationComponent::Digest(digest(0x03)),
                electrical_topology: ConfigurationComponent::Digest(digest(0x04)),
                thermal_topology: ConfigurationComponent::Digest(digest(0x05)),
                protection_settings: ConfigurationComponent::Digest(digest(0x06)),
                sensor_map: ConfigurationComponent::Digest(digest(0x07)),
                actuator_map: ConfigurationComponent::Digest(digest(0x08)),
                calibration: ConfigurationComponent::Digest(digest(0x09)),
                network_topology: ConfigurationComponent::Digest(digest(0x0a)),
            },
        )
        .unwrap()
    }

    fn profile_transition() -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        SafetyProfileAuthorizationTransition::bootstrap(
            SafetyProfileAuthorizationSubject::new(
                "profile-auth-1",
                "profile-root-v1",
                profile_root(0x44),
                "rack",
                1,
                1_000,
                3_000,
                &profile,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn profile_lifecycle(
        profile_transition: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationLifecycleState {
        let head = SafetyProfileAuthorizationHead::from_transition(profile_transition).unwrap();
        SafetyProfileAuthorizationLifecycleState::active(head).unwrap()
    }

    fn record(config: ConfigurationDigest) -> CommissioningRecord {
        CommissioningRecord::new(
            1,
            CommissionedLocalSafetyEnvelope::new(
                local(),
                CommissioningBinding::new(config, "commissioning-evidence-1").unwrap(),
            ),
        )
        .unwrap()
    }

    fn commissioning_transition(
        qualified: &QualifiedSafetyConfiguration,
        profile_transition: &SafetyProfileAuthorizationTransition,
    ) -> CommissioningAuthorizationTransition {
        let record = record(qualified.configuration_digest());
        CommissioningAuthorizationTransition::bootstrap(
            CommissioningAuthorizationSubject::new(
                "commission-auth-1",
                "commission-root-v1",
                commissioning_root(0x55),
                1_200,
                2_500,
                &record,
                &hierarchy(),
                qualified,
                profile_transition,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn checked_commissioning(
        qualified: &QualifiedSafetyConfiguration,
        profile_transition: &SafetyProfileAuthorizationTransition,
        transition: &CommissioningAuthorizationTransition,
    ) -> PolicyCheckedCommissioningAuthorization {
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            CommissioningAuthorityRootSnapshot::new(
                "commission-root-v1",
                commissioning_root(0x55),
                3,
            )
            .unwrap(),
            CommissioningAuthorizationHead::Uninitialized,
            profile_lifecycle(profile_transition),
        )
        .unwrap();
        policy
            .check(
                &clock(),
                transition,
                &record(qualified.configuration_digest()),
                &hierarchy(),
                qualified,
                profile_transition,
            )
            .unwrap()
    }

    fn frozen_setup(
        qualified: &QualifiedSafetyConfiguration,
        challenge_value: u8,
    ) -> (
        SafetyConfigurationState,
        SafetyConfigurationFreezeToken,
        SafetyConfigurationFrozenObservation,
        PolicyCheckedFrozenSafetyConfigurationObservation,
    ) {
        let mut state = SafetyConfigurationState::initialize(
            "rack",
            qualified.configuration_digest(),
        )
        .unwrap();
        let freeze = state.begin_freeze(challenge(challenge_value)).unwrap();
        let observation = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            1,
            1_500,
            observer_root(0x66),
            challenge(challenge_value),
            qualified.configuration_digest(),
        )
        .unwrap();
        let frozen = SafetyConfigurationFrozenObservation::new(observation, &state, &freeze).unwrap();
        let observation_policy = SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            ConfigurationObserverRootSnapshot::new("observer-1", observer_root(0x66), 4).unwrap(),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge(challenge_value),
            500,
        )
        .unwrap();
        let checked = admit_frozen_safety_configuration_observation(
            &observation_policy,
            &clock(),
            &frozen,
            &state,
            &freeze,
            qualified,
        )
        .unwrap();
        (state, freeze, frozen, checked)
    }

    #[test]
    fn exact_policy_witnesses_compose_into_outer_policy_witness() {
        let qualified = qualified();
        let profile_transition = profile_transition();
        let transition = commissioning_transition(&qualified, &profile_transition);
        let checked_commissioning =
            checked_commissioning(&qualified, &profile_transition, &transition);
        let (_state, _freeze, frozen, checked_frozen) = frozen_setup(&qualified, 0x77);
        let observed = ObservedCommissioningAuthorization::new(transition, frozen).unwrap();

        let checked = admit_observed_commissioning_authorization(
            &observed,
            &checked_commissioning,
            &checked_frozen,
        )
        .unwrap();

        assert_eq!(checked.observed(), &observed);
        assert_eq!(checked.configuration_digest(), qualified.configuration_digest());
        assert_eq!(
            checked.canonical_observed_authorization_bytes(),
            observed.canonical_signing_bytes().unwrap().as_slice()
        );
    }

    #[test]
    fn policy_witness_for_different_frozen_observation_is_rejected() {
        let qualified = qualified();
        let profile_transition = profile_transition();
        let transition = commissioning_transition(&qualified, &profile_transition);
        let checked_commissioning =
            checked_commissioning(&qualified, &profile_transition, &transition);
        let (_state_a, _freeze_a, frozen_a, _checked_a) = frozen_setup(&qualified, 0x77);
        let (_state_b, _freeze_b, _frozen_b, checked_b) = frozen_setup(&qualified, 0x78);
        let observed = ObservedCommissioningAuthorization::new(transition, frozen_a).unwrap();

        assert_eq!(
            admit_observed_commissioning_authorization(
                &observed,
                &checked_commissioning,
                &checked_b,
            ),
            Err(
                ObservedCommissioningAuthorizationAdmissionError::FrozenObservationPolicyMismatch
            )
        );
    }

    #[test]
    fn policy_witness_for_different_commissioning_transition_is_rejected() {
        let qualified = qualified();
        let profile_transition = profile_transition();
        let transition_a = commissioning_transition(&qualified, &profile_transition);

        let record = record(qualified.configuration_digest());
        let transition_b = CommissioningAuthorizationTransition::bootstrap(
            CommissioningAuthorizationSubject::new(
                "commission-auth-2",
                "commission-root-v1",
                commissioning_root(0x55),
                1_200,
                2_500,
                &record,
                &hierarchy(),
                &qualified,
                &profile_transition,
            )
            .unwrap(),
        )
        .unwrap();
        let checked_b = checked_commissioning(&qualified, &profile_transition, &transition_b);
        let (_state, _freeze, frozen, _checked_frozen) = frozen_setup(&qualified, 0x77);
        let observed = ObservedCommissioningAuthorization::new(transition_a, frozen).unwrap();
        let (_state2, _freeze2, _frozen2, checked_frozen2) = frozen_setup(&qualified, 0x77);

        assert_eq!(
            admit_observed_commissioning_authorization(&observed, &checked_b, &checked_frozen2),
            Err(
                ObservedCommissioningAuthorizationAdmissionError::CommissioningTransitionPolicyMismatch
            )
        );
    }
}
