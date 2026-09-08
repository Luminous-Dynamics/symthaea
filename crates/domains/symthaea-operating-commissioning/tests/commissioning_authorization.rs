use std::collections::BTreeSet;

use symthaea_operating_autonomy::LocalSafetyEnvelope;
use symthaea_operating_commissioning::authorization::{
    CommissioningAuthorityRootDigest, CommissioningAuthorizationError,
    CommissioningAuthorizationSubject,
};
use symthaea_operating_commissioning::{
    CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord, ConfigurationDigest,
};
use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
use symthaea_resource_hierarchy::{NodeScale, ResourceHierarchy, ResourceNode};
use symthaea_resource_model::{
    ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
};
use symthaea_safety_configuration::{
    ConfigurationComponent, SafetyConfigurationManifest, SAFETY_CONFIGURATION_SCHEMA_V1,
};
use symthaea_safety_profile::authorization::{
    ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject,
};
use symthaea_safety_profile::transition::SafetyProfileAuthorizationTransition;
use symthaea_safety_profile::{
    ComponentRequirement, SafetyConfigurationProfile, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
};
use symthaea_safety_qualification::{
    QualifiedSafetyConfiguration, qualify_safety_configuration,
};

fn digest(byte: u8) -> ConfigurationDigest {
    ConfigurationDigest::Blake3_256([byte; 32])
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

fn power_key() -> symthaea_resource_model::ResourceKey {
    ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
        .unwrap()
        .key
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

fn component(byte: u8) -> ConfigurationComponent {
    ConfigurationComponent::Digest(digest(byte))
}

fn manifest(profile: &SafetyConfigurationProfile) -> SafetyConfigurationManifest {
    SafetyConfigurationManifest {
        schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
        node_id: "rack".to_owned(),
        profile_id: profile.profile_id.clone(),
        profile_digest: profile.digest().unwrap(),
        hardware_inventory: component(0x01),
        firmware: component(0x02),
        software_closure: component(0x03),
        electrical_topology: component(0x04),
        thermal_topology: component(0x05),
        protection_settings: component(0x06),
        sensor_map: component(0x07),
        actuator_map: component(0x08),
        calibration: component(0x09),
        network_topology: component(0x0a),
    }
}

fn local() -> LocalSafetyEnvelope {
    let mut modes = BTreeSet::new();
    modes.insert(OperatingMode::Normal);
    modes.insert(OperatingMode::Islanded);
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

fn qualified(profile: &SafetyConfigurationProfile) -> QualifiedSafetyConfiguration {
    qualify_safety_configuration(profile, manifest(profile)).unwrap()
}

fn profile_transition(
    profile: &SafetyConfigurationProfile,
    node: &str,
    authorization_id: &str,
) -> SafetyProfileAuthorizationTransition {
    let subject = SafetyProfileAuthorizationSubject::new(
        authorization_id,
        "profile-root-v1",
        ProfileAuthorityRootDigest::Blake3_256([0x33; 32]),
        node,
        1,
        1_000,
        2_000,
        profile,
    )
    .unwrap();
    SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap()
}

fn record(configuration: ConfigurationDigest) -> CommissioningRecord {
    CommissioningRecord::new(
        1,
        CommissionedLocalSafetyEnvelope::new(
            local(),
            CommissioningBinding::new(configuration, "commissioning/rack/g1").unwrap(),
        ),
    )
    .unwrap()
}

fn commissioning_root() -> CommissioningAuthorityRootDigest {
    CommissioningAuthorityRootDigest::Blake3_256([0x55; 32])
}

#[test]
fn exact_qualified_profile_lineage_and_record_form_one_subject() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let transition = profile_transition(&profile, "rack", "profile-auth-1");
    let record = record(qualified.configuration_digest());

    let subject = CommissioningAuthorizationSubject::new(
        "commission-auth-1",
        "commission-root-v1",
        commissioning_root(),
        1_200,
        1_800,
        &record,
        &hierarchy,
        &qualified,
        &transition,
    )
    .unwrap();

    assert_eq!(subject.subject_node_id(), "rack");
    assert_eq!(subject.commissioning_generation(), 1);
    assert_eq!(subject.configuration_digest(), qualified.configuration_digest());
    assert_eq!(subject.profile_id(), qualified.profile_id());
    assert_eq!(subject.profile_digest(), qualified.profile_digest());
    assert_eq!(subject.profile_authorization_generation(), 1);
    assert_eq!(
        subject.profile_authorization_transition_digest(),
        transition.transition_digest().unwrap()
    );
    assert!(!subject.canonical_signing_bytes().unwrap().is_empty());
}

#[test]
fn wrong_configuration_cannot_be_commissioned_as_qualified() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let transition = profile_transition(&profile, "rack", "profile-auth-1");
    let record = record(digest(0xee));

    assert!(matches!(
        CommissioningAuthorizationSubject::new(
            "commission-auth-1",
            "commission-root-v1",
            commissioning_root(),
            1_200,
            1_800,
            &record,
            &hierarchy,
            &qualified,
            &transition,
        ),
        Err(CommissioningAuthorizationError::RecordQualificationConfigurationMismatch { .. })
    ));
}

#[test]
fn profile_authorization_from_wrong_node_cannot_supply_provenance() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let transition = profile_transition(&profile, "site", "profile-auth-wrong-node");
    let record = record(qualified.configuration_digest());

    assert!(matches!(
        CommissioningAuthorizationSubject::new(
            "commission-auth-1",
            "commission-root-v1",
            commissioning_root(),
            1_200,
            1_800,
            &record,
            &hierarchy,
            &qualified,
            &transition,
        ),
        Err(CommissioningAuthorizationError::ProfileAuthorizationNodeMismatch { .. })
    ));
}

#[test]
fn exact_profile_authorization_lineage_is_authenticated() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let record = record(qualified.configuration_digest());
    let first_transition = profile_transition(&profile, "rack", "profile-auth-a");
    let second_transition = profile_transition(&profile, "rack", "profile-auth-b");

    let first = CommissioningAuthorizationSubject::new(
        "commission-auth-1",
        "commission-root-v1",
        commissioning_root(),
        1_200,
        1_800,
        &record,
        &hierarchy,
        &qualified,
        &first_transition,
    )
    .unwrap();
    let second = CommissioningAuthorizationSubject::new(
        "commission-auth-1",
        "commission-root-v1",
        commissioning_root(),
        1_200,
        1_800,
        &record,
        &hierarchy,
        &qualified,
        &second_transition,
    )
    .unwrap();

    assert_ne!(
        first.profile_authorization_transition_digest(),
        second.profile_authorization_transition_digest()
    );
    assert_ne!(
        first.canonical_signing_bytes().unwrap(),
        second.canonical_signing_bytes().unwrap()
    );
}

#[test]
fn commissioning_root_is_a_separate_raw_key_fingerprint_type() {
    let raw_key = [0x42; 1952];
    let root = CommissioningAuthorityRootDigest::from_raw_verifier_key(&raw_key);
    assert_eq!(
        root.into_blake3_256(),
        ConfigurationDigest::blake3_256(&raw_key).into_blake3_256()
    );
}

#[test]
fn invalid_commissioning_action_window_fails_closed() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let transition = profile_transition(&profile, "rack", "profile-auth-1");
    let record = record(qualified.configuration_digest());

    assert!(matches!(
        CommissioningAuthorizationSubject::new(
            "commission-auth-1",
            "commission-root-v1",
            commissioning_root(),
            1_800,
            1_800,
            &record,
            &hierarchy,
            &qualified,
            &transition,
        ),
        Err(CommissioningAuthorizationError::InvalidValidityWindow { .. })
    ));
}