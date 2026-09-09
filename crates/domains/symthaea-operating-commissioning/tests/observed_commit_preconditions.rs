// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_operating_autonomy::LocalSafetyEnvelope;
use symthaea_operating_commissioning::admission::{
    CommissioningAuthorityRootSnapshot, CommissioningAuthorizationAdmissionPolicy,
    CommissioningAuthorizationHead,
};
use symthaea_operating_commissioning::authorization::transition::
    CommissioningAuthorizationTransition;
use symthaea_operating_commissioning::authorization::{
    CommissioningAuthorizationSubject, CommissioningAuthorityRootDigest,
};
use symthaea_operating_commissioning::observed_admission::
    admit_observed_commissioning_authorization;
use symthaea_operating_commissioning::observed_authorization::
    ObservedCommissioningAuthorization;
use symthaea_operating_commissioning::observed_commit_preconditions::{
    ObservedCommissioningAuthorizationCommitError,
    ObservedCommissioningAuthorizationCommitPreconditions,
    ObservedCommissioningAuthorizationCommitState,
};
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
use symthaea_safety_profile::admission::SafetyProfileAuthorizationHead;
use symthaea_safety_profile::authorization::{
    ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject,
};
use symthaea_safety_profile::commit_preconditions::ProfileAuthorityRootSnapshot;
use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
use symthaea_safety_profile::transition::SafetyProfileAuthorizationTransition;
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use symthaea_safety_profile::{
    ComponentRequirement, SafetyConfigurationProfile,
    SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
};
use symthaea_safety_qualification::frozen_observation_admission::
    admit_frozen_safety_configuration_observation;
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

fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
    TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
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

struct Fixture {
    preconditions: ObservedCommissioningAuthorizationCommitPreconditions,
    commissioning_root: CommissioningAuthorityRootSnapshot,
    profile_root: ProfileAuthorityRootSnapshot,
    profile_lifecycle: SafetyProfileAuthorizationLifecycleState,
    observer_root: ConfigurationObserverRootSnapshot,
    state: SafetyConfigurationState,
}

fn fixture() -> Fixture {
    let qualified = qualified();
    let profile_transition = profile_transition();
    let profile_head = SafetyProfileAuthorizationHead::from_transition(&profile_transition).unwrap();
    let profile_lifecycle = SafetyProfileAuthorizationLifecycleState::active(profile_head).unwrap();

    let commissioning_root = CommissioningAuthorityRootSnapshot::new(
        "commission-root-v1",
        commissioning_root(0x55),
        3,
    )
    .unwrap();
    let profile_root_snapshot =
        ProfileAuthorityRootSnapshot::new("profile-root-v1", profile_root(0x44), 5).unwrap();

    let commissioning_record = record(qualified.configuration_digest());
    let commissioning_transition = CommissioningAuthorizationTransition::bootstrap(
        CommissioningAuthorizationSubject::new(
            "commission-auth-1",
            "commission-root-v1",
            commissioning_root(0x55),
            1_200,
            2_500,
            &commissioning_record,
            &hierarchy(),
            &qualified,
            &profile_transition,
        )
        .unwrap(),
    )
    .unwrap();
    let commissioning_policy = CommissioningAuthorizationAdmissionPolicy::new(
        "rack",
        commissioning_root.clone(),
        CommissioningAuthorizationHead::Uninitialized,
        profile_lifecycle.clone(),
    )
    .unwrap();
    let checked_commissioning = commissioning_policy
        .check(
            &clock(1_550, 1_600),
            &commissioning_transition,
            &commissioning_record,
            &hierarchy(),
            &qualified,
            &profile_transition,
        )
        .unwrap();

    let observer_root_snapshot =
        ConfigurationObserverRootSnapshot::new("observer-1", observer_root(0x66), 4).unwrap();
    let mut state =
        SafetyConfigurationState::initialize("rack", qualified.configuration_digest()).unwrap();
    let freeze = state.begin_freeze(challenge(0x77)).unwrap();
    let observation = SafetyConfigurationObservation::new(
        "obs-1",
        "observer-1",
        "rack",
        1,
        1_500,
        observer_root(0x66),
        challenge(0x77),
        qualified.configuration_digest(),
    )
    .unwrap();
    let frozen = SafetyConfigurationFrozenObservation::new(observation, &state, &freeze).unwrap();
    let observation_policy = SafetyConfigurationObservationAdmissionPolicy::new(
        "rack",
        observer_root_snapshot.clone(),
        SafetyConfigurationObservationHead::Uninitialized,
        challenge(0x77),
        500,
    )
    .unwrap();
    let checked_frozen = admit_frozen_safety_configuration_observation(
        &observation_policy,
        &clock(1_550, 1_600),
        &frozen,
        &state,
        &freeze,
        &qualified,
    )
    .unwrap();

    let outer = ObservedCommissioningAuthorization::new(commissioning_transition, frozen).unwrap();
    let checked_outer = admit_observed_commissioning_authorization(
        &outer,
        &checked_commissioning,
        &checked_frozen,
    )
    .unwrap();
    let preconditions = ObservedCommissioningAuthorizationCommitPreconditions::from_policy_checked(
        &checked_outer,
        profile_root_snapshot.clone(),
    )
    .unwrap();

    Fixture {
        preconditions,
        commissioning_root,
        profile_root: profile_root_snapshot,
        profile_lifecycle,
        observer_root: observer_root_snapshot,
        state,
    }
}

#[test]
fn all_current_components_are_ready_to_commit() {
    let fixture = fixture();
    let p = &fixture.preconditions;
    assert_eq!(
        p.recheck_commit_observation(
            &clock(1_600, 1_650),
            &fixture.commissioning_root,
            &fixture.profile_root,
            p.commissioning().expected_predecessor_head(),
            &fixture.profile_lifecycle,
            None,
            &fixture.observer_root,
            p.frozen_observation().inner().expected_predecessor_head(),
            p.frozen_observation().inner().expected_challenge(),
            None,
            None,
            &fixture.state,
            None,
        )
        .unwrap(),
        ObservedCommissioningAuthorizationCommitState::ReadyToCommit
    );
}

#[test]
fn exact_complete_history_is_idempotent_acknowledgement() {
    let mut fixture = fixture();
    let freeze = fixture.preconditions.frozen_observation().expected_freeze().clone();
    fixture.state.end_freeze(&freeze).unwrap();
    let p = &fixture.preconditions;
    assert_eq!(
        p.recheck_commit_observation(
            &clock(9_000, 9_100),
            &fixture.commissioning_root,
            &fixture.profile_root,
            p.commissioning().candidate_head(),
            &fixture.profile_lifecycle,
            Some(p.commissioning().commissioning_record_digest()),
            &fixture.observer_root,
            p.frozen_observation().inner().candidate_head(),
            challenge(0x99),
            Some(p.frozen_observation().inner().observation_digest()),
            Some(p.frozen_observation().frozen_observation_digest()),
            &fixture.state,
            Some(p.observed_authorization_digest()),
        )
        .unwrap(),
        ObservedCommissioningAuthorizationCommitState::AlreadyCommitted
    );
}

#[test]
fn partially_committed_components_fail_closed() {
    let fixture = fixture();
    let p = &fixture.preconditions;
    assert!(matches!(
        p.recheck_commit_observation(
            &clock(1_600, 1_650),
            &fixture.commissioning_root,
            &fixture.profile_root,
            p.commissioning().candidate_head(),
            &fixture.profile_lifecycle,
            Some(p.commissioning().commissioning_record_digest()),
            &fixture.observer_root,
            p.frozen_observation().inner().expected_predecessor_head(),
            p.frozen_observation().inner().expected_challenge(),
            None,
            None,
            &fixture.state,
            None,
        ),
        Err(ObservedCommissioningAuthorizationCommitError::PartialCompositeCommit { .. })
    ));
}

#[test]
fn same_digest_external_mutation_is_detected_by_freeze_lineage() {
    let mut fixture = fixture();
    let same_digest = fixture.state.configuration_digest();
    let epoch = fixture.state.configuration_epoch();
    fixture
        .state
        .record_external_mutation(epoch, same_digest)
        .unwrap();
    let p = &fixture.preconditions;

    assert!(matches!(
        p.recheck_commit_observation(
            &clock(1_600, 1_650),
            &fixture.commissioning_root,
            &fixture.profile_root,
            p.commissioning().expected_predecessor_head(),
            &fixture.profile_lifecycle,
            None,
            &fixture.observer_root,
            p.frozen_observation().inner().expected_predecessor_head(),
            p.frozen_observation().inner().expected_challenge(),
            None,
            None,
            &fixture.state,
            None,
        ),
        Err(ObservedCommissioningAuthorizationCommitError::FrozenObservation(_))
    ));
}
