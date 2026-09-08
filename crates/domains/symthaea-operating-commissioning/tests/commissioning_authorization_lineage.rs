use std::collections::BTreeSet;

use symthaea_operating_autonomy::LocalSafetyEnvelope;
use symthaea_operating_commissioning::authorization::transition::{
    CommissioningAuthorizationPredecessor, CommissioningAuthorizationTransition,
    CommissioningAuthorizationTransitionError,
};
use symthaea_operating_commissioning::authorization::{
    CommissioningAuthorityRootDigest, CommissioningAuthorizationSubject,
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

fn profile_transition(profile: &SafetyConfigurationProfile) -> SafetyProfileAuthorizationTransition {
    let subject = SafetyProfileAuthorizationSubject::new(
        "profile-auth-1",
        "profile-root-v1",
        ProfileAuthorityRootDigest::Blake3_256([0x33; 32]),
        "rack",
        1,
        1_000,
        2_000,
        profile,
    )
    .unwrap();
    SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap()
}

fn commissioning_root(byte: u8) -> CommissioningAuthorityRootDigest {
    CommissioningAuthorityRootDigest::Blake3_256([byte; 32])
}

fn record(generation: u64, configuration: ConfigurationDigest, evidence: &str) -> CommissioningRecord {
    CommissioningRecord::new(
        generation,
        CommissionedLocalSafetyEnvelope::new(
            local(),
            CommissioningBinding::new(configuration, evidence).unwrap(),
        ),
    )
    .unwrap()
}

fn subject(
    generation: u64,
    authorization_id: &str,
    root: CommissioningAuthorityRootDigest,
    hierarchy: &ResourceHierarchy,
    qualified: &QualifiedSafetyConfiguration,
    profile_transition: &SafetyProfileAuthorizationTransition,
) -> CommissioningAuthorizationSubject {
    let record = record(
        generation,
        qualified.configuration_digest(),
        &format!("commissioning/rack/g{generation}"),
    );
    CommissioningAuthorizationSubject::new(
        authorization_id,
        "commission-root-v1",
        root,
        1_200,
        1_800,
        &record,
        hierarchy,
        qualified,
        profile_transition,
    )
    .unwrap()
}

#[test]
fn bootstrap_and_exact_successor_form_digest_bound_lineage() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let profile_transition = profile_transition(&profile);

    let first = CommissioningAuthorizationTransition::bootstrap(subject(
        1,
        "commission-auth-1",
        commissioning_root(0x55),
        &hierarchy,
        &qualified,
        &profile_transition,
    ))
    .unwrap();
    let second = CommissioningAuthorizationTransition::successor(
        &first,
        subject(
            2,
            "commission-auth-2",
            commissioning_root(0x55),
            &hierarchy,
            &qualified,
            &profile_transition,
        ),
    )
    .unwrap();

    assert_eq!(first.generation(), 1);
    assert_eq!(second.generation(), 2);
    assert_eq!(
        second.predecessor(),
        CommissioningAuthorizationPredecessor::Previous(first.transition_digest().unwrap())
    );
    assert_ne!(first.transition_digest().unwrap(), second.transition_digest().unwrap());
}

#[test]
fn generation_skip_fails_closed() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let profile_transition = profile_transition(&profile);
    let first = CommissioningAuthorizationTransition::bootstrap(subject(
        1,
        "commission-auth-1",
        commissioning_root(0x55),
        &hierarchy,
        &qualified,
        &profile_transition,
    ))
    .unwrap();

    assert!(matches!(
        CommissioningAuthorizationTransition::successor(
            &first,
            subject(
                3,
                "commission-auth-3",
                commissioning_root(0x55),
                &hierarchy,
                &qualified,
                &profile_transition,
            ),
        ),
        Err(CommissioningAuthorizationTransitionError::GenerationNotSuccessor {
            expected: 2,
            observed: 3,
            ..
        })
    ));
}

#[test]
fn ordinary_successor_cannot_rotate_commissioning_root() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let profile_transition = profile_transition(&profile);
    let first = CommissioningAuthorizationTransition::bootstrap(subject(
        1,
        "commission-auth-1",
        commissioning_root(0x55),
        &hierarchy,
        &qualified,
        &profile_transition,
    ))
    .unwrap();

    assert_eq!(
        CommissioningAuthorizationTransition::successor(
            &first,
            subject(
                2,
                "commission-auth-2",
                commissioning_root(0x66),
                &hierarchy,
                &qualified,
                &profile_transition,
            ),
        ),
        Err(CommissioningAuthorizationTransitionError::AuthorityRootChanged)
    );
}

#[test]
fn non_initial_generation_cannot_bootstrap() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let profile_transition = profile_transition(&profile);

    assert_eq!(
        CommissioningAuthorizationTransition::bootstrap(subject(
            2,
            "commission-auth-2",
            commissioning_root(0x55),
            &hierarchy,
            &qualified,
            &profile_transition,
        )),
        Err(CommissioningAuthorizationTransitionError::BootstrapGenerationMustBeOne {
            observed: 2,
        })
    );
}

#[test]
fn same_generation_forks_have_distinct_transition_identity() {
    let hierarchy = hierarchy();
    let profile = profile();
    let qualified = qualified(&profile);
    let profile_transition = profile_transition(&profile);

    let first = CommissioningAuthorizationTransition::bootstrap(subject(
        1,
        "commission-auth-a",
        commissioning_root(0x55),
        &hierarchy,
        &qualified,
        &profile_transition,
    ))
    .unwrap();
    let second = CommissioningAuthorizationTransition::bootstrap(subject(
        1,
        "commission-auth-b",
        commissioning_root(0x55),
        &hierarchy,
        &qualified,
        &profile_transition,
    ))
    .unwrap();

    assert_ne!(first.transition_digest().unwrap(), second.transition_digest().unwrap());
    assert_ne!(
        first.canonical_signing_bytes().unwrap(),
        second.canonical_signing_bytes().unwrap()
    );
}
