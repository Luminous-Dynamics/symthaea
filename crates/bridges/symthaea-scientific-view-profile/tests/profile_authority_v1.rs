use symthaea_scientific_view_profile::{
    AuthoritySourceBindingV1, AuthoritySourceOccurrenceV1, Commitment32,
    RoleSemanticRevisionV1, ScientificAuthorityRoleV1 as Role,
    ScientificViewDeploymentBindingV1, ScientificViewProfileError,
    ScientificViewSemanticProfileV1,
};

fn c(byte: u8) -> Commitment32 {
    assert_ne!(byte, 0);
    Commitment32::from_bytes([byte; 32])
}

fn hex(commitment: Commitment32) -> String {
    commitment
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn semantics(role: Role, byte: u8) -> RoleSemanticRevisionV1 {
    RoleSemanticRevisionV1::new(role, c(byte)).unwrap()
}

fn profile_with_roles(roles: &[(Role, u8)]) -> ScientificViewSemanticProfileV1 {
    ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        roles
            .iter()
            .map(|(role, byte)| semantics(*role, *byte))
            .collect(),
        c(201),
        c(202),
        c(203),
        c(204),
    )
    .unwrap()
}

fn source(role: Role, id: &str, epoch: u64, byte: u8) -> AuthoritySourceBindingV1 {
    AuthoritySourceBindingV1::new(role, id, epoch, c(byte)).unwrap()
}

#[test]
fn semantic_profile_is_input_order_invariant_but_role_semantics_are_material() {
    let a = profile_with_roles(&[
        (Role::VerifierPolicyHead, 12),
        (Role::ResearchSemanticHead, 11),
    ]);
    let b = profile_with_roles(&[
        (Role::ResearchSemanticHead, 11),
        (Role::VerifierPolicyHead, 12),
    ]);
    let changed = profile_with_roles(&[
        (Role::ResearchSemanticHead, 11),
        (Role::VerifierPolicyHead, 13),
    ]);

    assert_eq!(a.commitment(), b.commitment());
    assert_eq!(
        a.required_roles().collect::<Vec<_>>(),
        vec![Role::ResearchSemanticHead, Role::VerifierPolicyHead]
    );
    assert_ne!(a.commitment(), changed.commitment());
}

#[test]
fn duplicate_profile_role_is_rejected() {
    let err = ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            semantics(Role::ResearchSemanticHead, 11),
            semantics(Role::ResearchSemanticHead, 12),
        ],
        c(201),
        c(202),
        c(203),
        c(204),
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewProfileError::DuplicateRole(Role::ResearchSemanticHead)
    );
}

#[test]
fn deployment_binding_requires_exact_role_census_and_is_input_order_invariant() {
    let profile = profile_with_roles(&[
        (Role::ResearchSemanticHead, 11),
        (Role::VerifierPolicyHead, 12),
    ]);

    let a = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![
            source(Role::VerifierPolicyHead, "verifier-policy/store-a", 1, 22),
            source(Role::ResearchSemanticHead, "research/store-a", 1, 21),
        ],
    )
    .unwrap();

    let b = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![
            source(Role::ResearchSemanticHead, "research/store-a", 1, 21),
            source(Role::VerifierPolicyHead, "verifier-policy/store-a", 1, 22),
        ],
    )
    .unwrap();

    assert_eq!(a.commitment(), b.commitment());
    assert_eq!(a.source_roster_commitment(), b.source_roster_commitment());

    let missing = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![source(
            Role::ResearchSemanticHead,
            "research/store-a",
            1,
            21,
        )],
    )
    .unwrap_err();
    assert_eq!(
        missing,
        ScientificViewProfileError::MissingRequiredRole(Role::VerifierPolicyHead)
    );

    let extra = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![
            source(Role::ResearchSemanticHead, "research/store-a", 1, 21),
            source(Role::VerifierPolicyHead, "verifier-policy/store-a", 1, 22),
            source(
                Role::DiscoveryCompletenessHead,
                "discovery/store-a",
                1,
                23,
            ),
        ],
    )
    .unwrap_err();
    assert_eq!(
        extra,
        ScientificViewProfileError::ExtraRole(Role::DiscoveryCompletenessHead)
    );
}

#[test]
fn same_sources_under_different_deployment_are_not_same_authority_binding() {
    let profile = profile_with_roles(&[(Role::ResearchSemanticHead, 11)]);
    let sources = vec![source(
        Role::ResearchSemanticHead,
        "research/store-a",
        1,
        21,
    )];

    let a =
        ScientificViewDeploymentBindingV1::new("deployment/site01-a", &profile, sources.clone())
            .unwrap();
    let b = ScientificViewDeploymentBindingV1::new("deployment/site01-b", &profile, sources)
        .unwrap();

    assert_ne!(a.commitment(), b.commitment());
    assert_ne!(a.source_roster_commitment(), b.source_roster_commitment());
}

#[test]
fn source_occurrence_successor_is_exact_adjacent_and_noop_is_rejected() {
    let profile = profile_with_roles(&[(Role::ResearchSemanticHead, 11)]);
    let binding = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![source(
            Role::ResearchSemanticHead,
            "research/store-a",
            7,
            21,
        )],
    )
    .unwrap();

    let first =
        AuthoritySourceOccurrenceV1::genesis(&binding, Role::ResearchSemanticHead, c(31)).unwrap();
    let second = first.successor(&binding, c(32)).unwrap();

    assert_eq!(first.lineage_position(), 1);
    assert_eq!(first.predecessor_occurrence(), None);
    assert_eq!(second.lineage_position(), 2);
    assert_eq!(second.predecessor_occurrence(), Some(first.commitment()));
    assert_eq!(second.provisioning_epoch(), 7);
    assert_ne!(first.commitment(), second.commitment());

    let noop = second.successor(&binding, c(32)).unwrap_err();
    assert_eq!(noop, ScientificViewProfileError::NoOpOccurrence);
}

#[test]
fn reprovisioning_same_logical_state_is_distinct_occurrence_identity() {
    let profile = profile_with_roles(&[(Role::ResearchSemanticHead, 11)]);

    let epoch_1 = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![source(
            Role::ResearchSemanticHead,
            "research/store-a",
            1,
            21,
        )],
    )
    .unwrap();
    let epoch_2 = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![source(
            Role::ResearchSemanticHead,
            "research/store-a",
            2,
            21,
        )],
    )
    .unwrap();

    let state = c(31);
    let old =
        AuthoritySourceOccurrenceV1::genesis(&epoch_1, Role::ResearchSemanticHead, state).unwrap();
    let reprovisioned =
        AuthoritySourceOccurrenceV1::genesis(&epoch_2, Role::ResearchSemanticHead, state).unwrap();

    assert_ne!(epoch_1.commitment(), epoch_2.commitment());
    assert_ne!(old.commitment(), reprovisioned.commitment());
    assert_eq!(old.state_commitment(), reprovisioned.state_commitment());

    let crossing = old.successor(&epoch_2, c(32)).unwrap_err();
    assert_eq!(
        crossing,
        ScientificViewProfileError::DeploymentBindingMismatch
    );
}

#[test]
fn stable_ids_are_canonical_and_placeholder_commitments_fail_closed() {
    let uppercase = AuthoritySourceBindingV1::new(
        Role::ResearchSemanticHead,
        "Research/Store-A",
        1,
        c(21),
    )
    .unwrap_err();
    assert_eq!(
        uppercase,
        ScientificViewProfileError::InvalidStableId {
            field: "source_identity"
        }
    );

    let zero = RoleSemanticRevisionV1::new(Role::ResearchSemanticHead, Commitment32::ZERO)
        .unwrap_err();
    assert_eq!(
        zero,
        ScientificViewProfileError::ZeroCommitment {
            field: "role_semantic_commitment"
        }
    );
}

#[test]
fn canonical_sha256_vectors_are_frozen() {
    let research_semantics = semantics(Role::ResearchSemanticHead, 11);
    let verifier_semantics = semantics(Role::VerifierPolicyHead, 12);

    assert_eq!(
        hex(research_semantics.commitment()),
        "ce44d2ad7faf5fd658124d2aac4591fef7fdba736428309a2b97d78527de6d01"
    );
    assert_eq!(
        hex(verifier_semantics.commitment()),
        "924ad1bd149bf177477cf97d1681d17efdf359daf109da05095afca64076a4e2"
    );

    let profile = ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![verifier_semantics, research_semantics],
        c(201),
        c(202),
        c(203),
        c(204),
    )
    .unwrap();

    assert_eq!(
        hex(profile.commitment()),
        "2b4329ca9f8b66c9420322ed38231e9482c6e49886a6a25de6d707d627d1bd07"
    );

    let research = source(Role::ResearchSemanticHead, "research/store-a", 7, 21);
    let verifier = source(
        Role::VerifierPolicyHead,
        "verifier-policy/store-a",
        3,
        22,
    );

    assert_eq!(
        hex(research.commitment()),
        "174a5b0bcb86b25874d8dde5d3d761be5cd02b4d7665378bce6fab144eb2d4bd"
    );
    assert_eq!(
        hex(verifier.commitment()),
        "f6eb21328c0a5f8e1472e8cc685ebe9bda4a0e7a78083fdd91c2e217b2e6b812"
    );

    let binding = ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        &profile,
        vec![verifier, research],
    )
    .unwrap();

    assert_eq!(
        hex(binding.source_roster_commitment()),
        "c1142e997ffb0c44c49ad3b78c059c0d3f56b9a360a7c39a8c590b44bbc5b068"
    );
    assert_eq!(
        hex(binding.commitment()),
        "c6fe69a183709332abfc377222fe37d7eefce67bda7b9a462d7efd7b4dcb15e6"
    );

    let first =
        AuthoritySourceOccurrenceV1::genesis(&binding, Role::ResearchSemanticHead, c(31)).unwrap();
    assert_eq!(
        hex(first.commitment()),
        "22626fe00e3134e56149d0aabe1709fff8f662d401fd44261d64839b0f396b97"
    );

    let second = first.successor(&binding, c(32)).unwrap();
    assert_eq!(
        hex(second.commitment()),
        "5a1682d7a98d9361c93a7ecec3122d764332f3308bbef85deaaea0474b660aab"
    );
}

#[test]
fn production_surface_contains_no_live_currentness_or_action_authority_dependency() {
    let manifest = include_str!("../Cargo.toml");
    let source = include_str!("../src/lib.rs");

    assert!(manifest.contains("sha2"));
    assert!(!manifest.contains("blake3"));
    assert!(!manifest.contains("serde"));
    assert!(!source.contains("pub fn is_current"));
    assert!(!source.contains("CurrentScientificAuthorityEnvelope"));
    assert!(!source.contains("ActionAuthority"));
}
