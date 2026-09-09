// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Byte-exact cross-tool vector for the complete observed-commissioning v1 message.
//!
//! This integration test intentionally reconstructs the public fixture through the
//! public APIs and pins the resulting outer authentication bytes. The expected bytes
//! were independently reconstructed from the documented lower-layer canonical
//! encodings and validated against the already-pinned frozen-observation vector.

use std::collections::BTreeSet;

use symthaea_operating_autonomy::LocalSafetyEnvelope;
use symthaea_operating_commissioning::authorization::{
    transition::CommissioningAuthorizationTransition, CommissioningAuthorizationSubject,
    CommissioningAuthorityRootDigest,
};
use symthaea_operating_commissioning::observed_authorization::ObservedCommissioningAuthorization;
use symthaea_operating_commissioning::{
    CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord,
    ConfigurationDigest,
};
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
    ConfigurationComponent, SafetyConfigurationManifest, SAFETY_CONFIGURATION_SCHEMA_V1,
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

const PINNED_OBSERVED_COMMISSIONING_V1_HEX: &str =
    "73796d74686165613a6f627365727665642d636f6d6d697373696f6e696e672d617574686f72697a6174696f6e3a7631000000003073796d74686165612d6f627365727665642d636f6d6d697373696f6e696e672d617574686f72697a6174696f6e2d7631000001df73796d74686165613a636f6d6d697373696f6e696e672d617574686f72697a6174696f6e2d7472616e736974696f6e3a7631000000003373796d74686165612d636f6d6d697373696f6e696e672d617574686f72697a6174696f6e2d7472616e736974696f6e2d7631000000000000017173796d74686165613a636f6d6d697373696f6e696e672d617574686f72697a6174696f6e3a7631000000002973796d74686165612d636f6d6d697373696f6e696e672d617574686f72697a6174696f6e2d763100000011636f6d6d697373696f6e2d617574682d3100000012636f6d6d697373696f6e2d726f6f742d7631000000047261636b000000000000000100000000000004b000000000000009c401555555555555555555555555555555555555555555555555555555555555555501137b801f68dfce0863e6f5959b5ec1dc599f8931fce91b4c332adbbbb93a780601457c1c3b055335df8335a717f08ee09d3f0b38533240a2479ea4302785d0365e00000022636f6d707574652d636f6d6d6f6e732d6175746f6e6f6d6f75732d6e6f64652d76310164ec08e2079b55cfcb509d0b57e4f6be310c900a0099ffd919fefa1f087af01b0000000000000001010e8f59cc49a8fb1ea81c803220a5036bcced15e4304edcc266667be83d0ba6ee0000016d73796d74686165613a7361666574792d636f6e66696775726174696f6e2d66726f7a656e2d6f62736572766174696f6e3a7631000000003373796d74686165612d7361666574792d636f6e66696775726174696f6e2d66726f7a656e2d6f62736572766174696f6e2d763100000000000000010000000000000001000000f473796d74686165613a7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e3a7631000000002c73796d74686165612d7361666574792d636f6e66696775726174696f6e2d6f62736572766174696f6e2d7631000000056f62732d310000000a6f627365727665722d31000000047261636b000000000000000100000000000005dc016666666666666666666666666666666666666666666666666666666666666666777777777777777777777777777777777777777777777777777777777777777701457c1c3b055335df8335a717f08ee09d3f0b38533240a2479ea4302785d0365e";

const PINNED_OBSERVED_COMMISSIONING_V1_DIGEST_HEX: &str =
    "3cd5ef1b3bd0358e8bdbcfd6a1745cb7ab41683aa3b7877c929f91556dee0295";

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
) -> SafetyConfigurationFrozenObservation {
    let mut state = SafetyConfigurationState::initialize(node, config).unwrap();
    let freeze = state.begin_freeze(challenge(challenge_value)).unwrap();
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
        frozen_observation("rack", qualified.configuration_digest(), 0x77),
    )
    .unwrap()
}

fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[test]
fn observed_commissioning_v1_vector_is_byte_exact() {
    let observed = observed();
    let bytes = observed.canonical_signing_bytes().unwrap();

    assert_eq!(bytes.len(), 953);
    assert_eq!(hex(&bytes), PINNED_OBSERVED_COMMISSIONING_V1_HEX);
    assert_eq!(
        hex(
            &observed
                .observed_authorization_digest()
                .unwrap()
                .into_blake3_256()
        ),
        PINNED_OBSERVED_COMMISSIONING_V1_DIGEST_HEX
    );
}
