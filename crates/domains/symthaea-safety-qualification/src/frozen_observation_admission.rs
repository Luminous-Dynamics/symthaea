// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy admission for configuration observations bound to an exact live freeze.
//!
//! The ordinary observation admission layer proves node/root/challenge/lineage/time
//! policy for the nested observation. This module additionally requires the exact
//! freeze lineage authenticated by [`SafetyConfigurationFrozenObservation`] to still
//! be live in the local [`SafetyConfigurationState`].
//!
//! Success remains non-cryptographic. The returned opaque witness retains the
//! canonical *frozen-observation* bytes that a Xenia verifier must authenticate;
//! callers cannot accidentally substitute the weaker inner-observation bytes.

use crate::QualifiedSafetyConfiguration;
use crate::observation_admission::{
    PolicyCheckedCurrentSafetyConfigurationObservation,
    SafetyConfigurationObservationAdmissionError, SafetyConfigurationObservationAdmissionPolicy,
};
use symthaea_safety_configuration::frozen_observation::{
    SafetyConfigurationFrozenObservation, SafetyConfigurationFrozenObservationDigest,
    SafetyConfigurationFrozenObservationError,
};
use symthaea_safety_configuration::state::{
    SafetyConfigurationFreezeToken, SafetyConfigurationState,
};
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Admit one frozen configuration observation against both ordinary observation
/// policy and the exact current local freeze state.
pub fn admit_frozen_safety_configuration_observation(
    policy: &SafetyConfigurationObservationAdmissionPolicy,
    current_clock: &TrustedAuthorizationClockObservation,
    frozen_observation: &SafetyConfigurationFrozenObservation,
    state: &SafetyConfigurationState,
    freeze: &SafetyConfigurationFreezeToken,
    qualified: &QualifiedSafetyConfiguration,
) -> Result<
    PolicyCheckedFrozenSafetyConfigurationObservation,
    FrozenSafetyConfigurationObservationAdmissionError,
> {
    frozen_observation.validate()?;
    frozen_observation.require_live_freeze(state, freeze)?;

    let checked_observation = policy.check(
        current_clock,
        frozen_observation.observation(),
        qualified,
    )?;

    // The live-freeze check already proves these relationships. Keep explicit
    // assertions so future refactors cannot silently weaken the composition.
    debug_assert_eq!(
        frozen_observation.configuration_epoch(),
        freeze.configuration_epoch()
    );
    debug_assert_eq!(
        frozen_observation.freeze_generation(),
        freeze.freeze_generation()
    );
    debug_assert_eq!(
        checked_observation.configuration_digest(),
        freeze.configuration_digest()
    );
    debug_assert_eq!(checked_observation.expected_challenge(), freeze.challenge());

    Ok(PolicyCheckedFrozenSafetyConfigurationObservation {
        frozen_observation: frozen_observation.clone(),
        canonical_frozen_observation_bytes: frozen_observation.canonical_signing_bytes()?,
        frozen_observation_digest: frozen_observation.frozen_observation_digest()?,
        checked_observation,
        expected_freeze: freeze.clone(),
    })
}

/// Opaque non-serializable result of exact frozen-observation policy checks.
///
/// This proves no signature. A cryptographic adapter must authenticate
/// [`Self::canonical_frozen_observation_bytes`] under the observer root retained by
/// [`Self::checked_observation`] before this evidence may participate in
/// commissioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedFrozenSafetyConfigurationObservation {
    frozen_observation: SafetyConfigurationFrozenObservation,
    canonical_frozen_observation_bytes: Vec<u8>,
    frozen_observation_digest: SafetyConfigurationFrozenObservationDigest,
    checked_observation: PolicyCheckedCurrentSafetyConfigurationObservation,
    expected_freeze: SafetyConfigurationFreezeToken,
}

impl PolicyCheckedFrozenSafetyConfigurationObservation {
    pub fn frozen_observation(&self) -> &SafetyConfigurationFrozenObservation {
        &self.frozen_observation
    }

    pub fn canonical_frozen_observation_bytes(&self) -> &[u8] {
        &self.canonical_frozen_observation_bytes
    }

    pub fn frozen_observation_digest(&self) -> SafetyConfigurationFrozenObservationDigest {
        self.frozen_observation_digest
    }

    pub fn checked_observation(&self) -> &PolicyCheckedCurrentSafetyConfigurationObservation {
        &self.checked_observation
    }

    pub fn expected_freeze(&self) -> &SafetyConfigurationFreezeToken {
        &self.expected_freeze
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrozenSafetyConfigurationObservationAdmissionError {
    #[error(transparent)]
    FrozenObservation(#[from] SafetyConfigurationFrozenObservationError),
    #[error(transparent)]
    ObservationAdmission(#[from] SafetyConfigurationObservationAdmissionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation_admission::{
        ConfigurationObserverRootSnapshot, SafetyConfigurationObservationHead,
    };
    use crate::qualify_safety_configuration;
    use symthaea_safety_configuration::observation::{
        ConfigurationObservationChallenge, ConfigurationObserverRootDigest,
        SafetyConfigurationObservation,
    };
    use symthaea_safety_configuration::state::SafetyConfigurationStateError;
    use symthaea_safety_configuration::{
        ConfigurationComponent, ConfigurationDigest, SafetyConfigurationManifest,
        SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn root(byte: u8) -> ConfigurationObserverRootDigest {
        ConfigurationObserverRootDigest::Blake3_256([byte; 32])
    }

    fn challenge(byte: u8) -> ConfigurationObservationChallenge {
        ConfigurationObservationChallenge::new([byte; 32]).unwrap()
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
        let manifest = SafetyConfigurationManifest {
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
        };
        qualify_safety_configuration(&profile, manifest).unwrap()
    }

    fn clock() -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, 1_550, 1_600).unwrap()
    }

    fn policy(challenge: ConfigurationObservationChallenge) -> SafetyConfigurationObservationAdmissionPolicy {
        SafetyConfigurationObservationAdmissionPolicy::new(
            "rack",
            ConfigurationObserverRootSnapshot::new("observer-1", root(0x44), 3).unwrap(),
            SafetyConfigurationObservationHead::Uninitialized,
            challenge,
            500,
        )
        .unwrap()
    }

    fn setup() -> (
        QualifiedSafetyConfiguration,
        SafetyConfigurationState,
        SafetyConfigurationFreezeToken,
        SafetyConfigurationFrozenObservation,
    ) {
        let qualified = qualified();
        let mut state = SafetyConfigurationState::initialize(
            "rack",
            qualified.configuration_digest(),
        )
        .unwrap();
        let freeze = state.begin_freeze(challenge(0x55)).unwrap();
        let observation = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            "rack",
            1,
            1_500,
            root(0x44),
            challenge(0x55),
            qualified.configuration_digest(),
        )
        .unwrap();
        let frozen = SafetyConfigurationFrozenObservation::new(observation, &state, &freeze).unwrap();
        (qualified, state, freeze, frozen)
    }

    #[test]
    fn exact_active_freeze_produces_opaque_policy_witness() {
        let (qualified, state, freeze, frozen) = setup();
        let checked = admit_frozen_safety_configuration_observation(
            &policy(challenge(0x55)),
            &clock(),
            &frozen,
            &state,
            &freeze,
            &qualified,
        )
        .unwrap();
        assert_eq!(checked.expected_freeze(), &freeze);
        assert_eq!(
            checked.checked_observation().configuration_digest(),
            qualified.configuration_digest()
        );
        assert_eq!(
            checked.canonical_frozen_observation_bytes(),
            frozen.canonical_signing_bytes().unwrap().as_slice()
        );
    }

    #[test]
    fn external_mutation_before_admission_invalidates_freeze() {
        let (qualified, mut state, freeze, frozen) = setup();
        state
            .record_external_mutation(1, digest(0xee))
            .unwrap();
        assert_eq!(
            admit_frozen_safety_configuration_observation(
                &policy(challenge(0x55)),
                &clock(),
                &frozen,
                &state,
                &freeze,
                &qualified,
            ),
            Err(FrozenSafetyConfigurationObservationAdmissionError::FrozenObservation(
                SafetyConfigurationFrozenObservationError::State(
                    SafetyConfigurationStateError::NotFrozen
                )
            ))
        );
    }

    #[test]
    fn release_reacquire_aba_invalidates_old_frozen_observation() {
        let (qualified, mut state, freeze, frozen) = setup();
        state.end_freeze(&freeze).unwrap();
        let new_freeze = state.begin_freeze(challenge(0x55)).unwrap();
        assert_ne!(freeze.freeze_generation(), new_freeze.freeze_generation());
        assert!(admit_frozen_safety_configuration_observation(
            &policy(challenge(0x55)),
            &clock(),
            &frozen,
            &state,
            &freeze,
            &qualified,
        )
        .is_err());
    }

    #[test]
    fn policy_challenge_must_still_match_nested_observation() {
        let (qualified, state, freeze, frozen) = setup();
        assert!(matches!(
            admit_frozen_safety_configuration_observation(
                &policy(challenge(0x56)),
                &clock(),
                &frozen,
                &state,
                &freeze,
                &qualified,
            ),
            Err(FrozenSafetyConfigurationObservationAdmissionError::ObservationAdmission(
                SafetyConfigurationObservationAdmissionError::ChallengeMismatch
            ))
        ));
    }
}