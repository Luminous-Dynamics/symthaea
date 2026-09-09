// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical commissioning evidence bound to one exact frozen configuration observation.
//!
//! The existing commissioning-authorization transition binds commissioning lineage,
//! configuration, local safety, and profile provenance. The existing frozen
//! configuration observation binds observer identity, challenge, observation time,
//! configuration epoch, and freeze generation. Neither v1 format should be rewritten
//! merely to compose them.
//!
//! This module therefore defines one outer canonical authentication surface containing
//! the complete canonical bytes of both artifacts. A commissioning authority that
//! authenticates these outer bytes explicitly authorizes this commissioning action
//! against this exact observer transaction and freeze lineage.
//!
//! This is still raw evidence. It does not prove that either nested artifact has been
//! admitted, that either required signature is valid, or that the freeze is still live.
//! Those remain separate policy, cryptographic, and commit-time gates.

use crate::ConfigurationDigest;
use crate::authorization::transition::{
    CommissioningAuthorizationTransition, CommissioningAuthorizationTransitionError,
};
use symthaea_safety_configuration::frozen_observation::{
    SafetyConfigurationFrozenObservation, SafetyConfigurationFrozenObservationError,
};
use thiserror::Error;

pub const OBSERVED_COMMISSIONING_AUTHORIZATION_SCHEMA_V1: &str =
    "symthaea-observed-commissioning-authorization-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:observed-commissioning-authorization:v1\0";

/// BLAKE3-256 identity of one canonical observed-commissioning authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservedCommissioningAuthorizationDigest {
    Blake3_256([u8; 32]),
}

impl ObservedCommissioningAuthorizationDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Raw outer evidence binding one commissioning transition to one exact frozen
/// configuration observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedCommissioningAuthorization {
    schema_version: String,
    commissioning_transition: CommissioningAuthorizationTransition,
    frozen_observation: SafetyConfigurationFrozenObservation,
}

impl ObservedCommissioningAuthorization {
    pub fn new(
        commissioning_transition: CommissioningAuthorizationTransition,
        frozen_observation: SafetyConfigurationFrozenObservation,
    ) -> Result<Self, ObservedCommissioningAuthorizationError> {
        commissioning_transition.validate()?;
        frozen_observation.validate()?;

        let commissioning_subject = commissioning_transition.subject();
        let observation = frozen_observation.observation();

        if commissioning_subject.subject_node_id() != observation.subject_node_id() {
            return Err(ObservedCommissioningAuthorizationError::SubjectNodeMismatch {
                commissioning: commissioning_subject.subject_node_id().to_owned(),
                observation: observation.subject_node_id().to_owned(),
            });
        }
        if commissioning_subject.configuration_digest() != observation.configuration_digest() {
            return Err(
                ObservedCommissioningAuthorizationError::ConfigurationDigestMismatch {
                    commissioning: commissioning_subject.configuration_digest(),
                    observation: observation.configuration_digest(),
                },
            );
        }

        let observed = Self {
            schema_version: OBSERVED_COMMISSIONING_AUTHORIZATION_SCHEMA_V1.to_owned(),
            commissioning_transition,
            frozen_observation,
        };
        observed.validate()?;
        Ok(observed)
    }

    pub fn validate(&self) -> Result<(), ObservedCommissioningAuthorizationError> {
        if self.schema_version != OBSERVED_COMMISSIONING_AUTHORIZATION_SCHEMA_V1 {
            return Err(
                ObservedCommissioningAuthorizationError::UnsupportedSchemaVersion(
                    self.schema_version.clone(),
                ),
            );
        }
        self.commissioning_transition.validate()?;
        self.frozen_observation.validate()?;

        let commissioning_subject = self.commissioning_transition.subject();
        let observation = self.frozen_observation.observation();
        if commissioning_subject.subject_node_id() != observation.subject_node_id() {
            return Err(ObservedCommissioningAuthorizationError::SubjectNodeMismatch {
                commissioning: commissioning_subject.subject_node_id().to_owned(),
                observation: observation.subject_node_id().to_owned(),
            });
        }
        if commissioning_subject.configuration_digest() != observation.configuration_digest() {
            return Err(
                ObservedCommissioningAuthorizationError::ConfigurationDigestMismatch {
                    commissioning: commissioning_subject.configuration_digest(),
                    observation: observation.configuration_digest(),
                },
            );
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes intended for commissioning-authority
    /// authentication.
    ///
    /// Encoding v1:
    /// - fixed ASCII domain separator
    /// - schema as big-endian u32 length-prefixed UTF-8
    /// - complete canonical commissioning-transition bytes, u32 length-prefixed
    /// - complete canonical frozen-observation bytes, u32 length-prefixed
    pub fn canonical_signing_bytes(
        &self,
    ) -> Result<Vec<u8>, ObservedCommissioningAuthorizationError> {
        self.validate()?;
        let transition_bytes = self.commissioning_transition.canonical_signing_bytes()?;
        let observation_bytes = self.frozen_observation.canonical_signing_bytes()?;

        let mut out = Vec::with_capacity(
            DOMAIN_SEPARATOR.len()
                + self.schema_version.len()
                + transition_bytes.len()
                + observation_bytes.len()
                + 12,
        );
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_bytes(&mut out, "schema_version", self.schema_version.as_bytes())?;
        push_bytes(
            &mut out,
            "commissioning_transition",
            &transition_bytes,
        )?;
        push_bytes(&mut out, "frozen_observation", &observation_bytes)?;
        Ok(out)
    }

    pub fn observed_authorization_digest(
        &self,
    ) -> Result<ObservedCommissioningAuthorizationDigest, ObservedCommissioningAuthorizationError>
    {
        Ok(ObservedCommissioningAuthorizationDigest::blake3_256(
            &self.canonical_signing_bytes()?,
        ))
    }

    pub fn commissioning_transition(&self) -> &CommissioningAuthorizationTransition {
        &self.commissioning_transition
    }

    pub fn frozen_observation(&self) -> &SafetyConfigurationFrozenObservation {
        &self.frozen_observation
    }
}

fn push_bytes(
    out: &mut Vec<u8>,
    field: &'static str,
    bytes: &[u8],
) -> Result<(), ObservedCommissioningAuthorizationError> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| ObservedCommissioningAuthorizationError::FieldTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObservedCommissioningAuthorizationError {
    #[error(transparent)]
    CommissioningTransition(#[from] CommissioningAuthorizationTransitionError),
    #[error(transparent)]
    FrozenObservation(#[from] SafetyConfigurationFrozenObservationError),
    #[error("unsupported observed-commissioning authorization schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("observed commissioning subject node mismatch: commissioning {commissioning}, observation {observation}")]
    SubjectNodeMismatch {
        commissioning: String,
        observation: String,
    },
    #[error("observed commissioning configuration digest mismatch")]
    ConfigurationDigestMismatch {
        commissioning: ConfigurationDigest,
        observation: ConfigurationDigest,
    },
    #[error("observed commissioning field {0} exceeds canonical u32 length")]
    FieldTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::{
        CommissioningAuthorizationSubject, CommissioningAuthorityRootDigest,
    };
    use crate::{
        CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord,
    };
    use std::collections::BTreeSet;
    use symthaea_operating_autonomy::LocalSafetyEnvelope;
    use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
    use symthaea_resource_hierarchy::{NodeScale, ResourceHierarchy, ResourceNode};
    use symthaea_resource_model::{
        ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
    };
    use symthaea_safety_configuration::frozen_observation::SafetyConfigurationFrozenObservation;
    use symthaea_safety_configuration::observation::{
        ConfigurationObservationChallenge, ConfigurationObserverRootDigest,
        SafetyConfigurationObservation,
    };
    use symthaea_safety_configuration::state::SafetyConfigurationState;
    use symthaea_safety_configuration::{
        ConfigurationComponent, SafetyConfigurationManifest,
        SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::authorization::{
        ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject,
    };
    use symthaea_safety_profile::transition::SafetyProfileAuthorizationTransition;
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
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

    fn local(node: &str) -> LocalSafetyEnvelope {
        let mut modes = BTreeSet::new();
        modes.insert(OperatingMode::Normal);
        modes.insert(OperatingMode::Islanded);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: node.into(),
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

    fn qualified(node: &str) -> QualifiedSafetyConfiguration {
        let profile = profile();
        let manifest = SafetyConfigurationManifest {
            schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
            node_id: node.to_owned(),
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

    fn profile_transition(node: &str) -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        SafetyProfileAuthorizationTransition::bootstrap(
            SafetyProfileAuthorizationSubject::new(
                "profile-auth-1",
                "profile-root-v1",
                profile_root(0x44),
                node,
                1,
                1_000,
                3_000,
                &profile,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn record(
        node: &str,
        generation: u64,
        config: ConfigurationDigest,
    ) -> CommissioningRecord {
        CommissioningRecord::new(
            generation,
            CommissionedLocalSafetyEnvelope::new(
                local(node),
                CommissioningBinding::new(config, "commissioning-evidence-1").unwrap(),
            ),
        )
        .unwrap()
    }

    fn commissioning_transition(
        node: &str,
        authorization_id: &str,
        qualified: &QualifiedSafetyConfiguration,
    ) -> CommissioningAuthorizationTransition {
        let record = record(node, 1, qualified.configuration_digest());
        CommissioningAuthorizationTransition::bootstrap(
            CommissioningAuthorizationSubject::new(
                authorization_id,
                "commission-root-v1",
                commissioning_root(0x55),
                1_200,
                2_500,
                &record,
                &hierarchy(),
                qualified,
                &profile_transition(node),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn frozen_observation(
        node: &str,
        config: ConfigurationDigest,
        challenge_value: u8,
        freeze_cycles: usize,
    ) -> SafetyConfigurationFrozenObservation {
        let mut state = SafetyConfigurationState::initialize(node, config).unwrap();
        let mut freeze = state.begin_freeze(challenge(challenge_value)).unwrap();
        for _ in 1..freeze_cycles {
            state.end_freeze(&freeze).unwrap();
            freeze = state.begin_freeze(challenge(challenge_value)).unwrap();
        }
        let observation = SafetyConfigurationObservation::new(
            "obs-1",
            "observer-1",
            node,
            1,
            1_500,
            observer_root(0x66),
            challenge(challenge_value),
            config,
        )
        .unwrap();
        SafetyConfigurationFrozenObservation::new(observation, &state, &freeze).unwrap()
    }

    fn observed() -> ObservedCommissioningAuthorization {
        let qualified = qualified("rack");
        ObservedCommissioningAuthorization::new(
            commissioning_transition("rack", "commission-auth-1", &qualified),
            frozen_observation("rack", qualified.configuration_digest(), 0x77, 1),
        )
        .unwrap()
    }

    #[test]
    fn exact_matching_transition_and_frozen_observation_are_bound() {
        let observed = observed();
        assert_eq!(
            observed.commissioning_transition().subject().subject_node_id(),
            observed.frozen_observation().observation().subject_node_id()
        );
        assert_eq!(
            observed.commissioning_transition().subject().configuration_digest(),
            observed.frozen_observation().observation().configuration_digest()
        );
        assert!(!observed.canonical_signing_bytes().unwrap().is_empty());
    }

    #[test]
    fn identical_outer_evidence_has_identical_identity() {
        let first = observed();
        let second = first.clone();
        assert_eq!(
            first.canonical_signing_bytes().unwrap(),
            second.canonical_signing_bytes().unwrap()
        );
        assert_eq!(
            first.observed_authorization_digest().unwrap(),
            second.observed_authorization_digest().unwrap()
        );
    }

    #[test]
    fn wrong_configuration_is_rejected() {
        let qualified = qualified("rack");
        let transition = commissioning_transition("rack", "commission-auth-1", &qualified);
        let wrong = frozen_observation("rack", digest(0xee), 0x77, 1);
        assert!(matches!(
            ObservedCommissioningAuthorization::new(transition, wrong),
            Err(ObservedCommissioningAuthorizationError::ConfigurationDigestMismatch { .. })
        ));
    }

    #[test]
    fn wrong_node_is_rejected() {
        let qualified = qualified("rack");
        let transition = commissioning_transition("rack", "commission-auth-1", &qualified);
        let wrong = frozen_observation("other-rack", qualified.configuration_digest(), 0x77, 1);
        assert!(matches!(
            ObservedCommissioningAuthorization::new(transition, wrong),
            Err(ObservedCommissioningAuthorizationError::SubjectNodeMismatch { .. })
        ));
    }

    #[test]
    fn changing_observation_challenge_changes_outer_identity() {
        let qualified = qualified("rack");
        let transition = commissioning_transition("rack", "commission-auth-1", &qualified);
        let first = ObservedCommissioningAuthorization::new(
            transition.clone(),
            frozen_observation("rack", qualified.configuration_digest(), 0x77, 1),
        )
        .unwrap();
        let second = ObservedCommissioningAuthorization::new(
            transition,
            frozen_observation("rack", qualified.configuration_digest(), 0x78, 1),
        )
        .unwrap();
        assert_ne!(
            first.observed_authorization_digest().unwrap(),
            second.observed_authorization_digest().unwrap()
        );
    }

    #[test]
    fn release_reacquire_freeze_changes_outer_identity_even_with_same_config_and_challenge() {
        let qualified = qualified("rack");
        let transition = commissioning_transition("rack", "commission-auth-1", &qualified);
        let first = ObservedCommissioningAuthorization::new(
            transition.clone(),
            frozen_observation("rack", qualified.configuration_digest(), 0x77, 1),
        )
        .unwrap();
        let second = ObservedCommissioningAuthorization::new(
            transition,
            frozen_observation("rack", qualified.configuration_digest(), 0x77, 2),
        )
        .unwrap();
        assert_ne!(
            first.observed_authorization_digest().unwrap(),
            second.observed_authorization_digest().unwrap()
        );
    }

    #[test]
    fn changing_commissioning_transition_changes_outer_identity() {
        let qualified = qualified("rack");
        let frozen = frozen_observation("rack", qualified.configuration_digest(), 0x77, 1);
        let first = ObservedCommissioningAuthorization::new(
            commissioning_transition("rack", "commission-auth-1", &qualified),
            frozen.clone(),
        )
        .unwrap();
        let second = ObservedCommissioningAuthorization::new(
            commissioning_transition("rack", "commission-auth-2", &qualified),
            frozen,
        )
        .unwrap();
        assert_ne!(
            first.observed_authorization_digest().unwrap(),
            second.observed_authorization_digest().unwrap()
        );
    }
}