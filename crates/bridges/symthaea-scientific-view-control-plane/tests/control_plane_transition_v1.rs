use symthaea_scientific_view_control_plane::{
    classify_supplied_candidate_successors, CandidateScientificViewControlPlaneTransitionV1,
    CandidateSourceMigrationV1, ControlPlaneTransitionKindV1, DeploymentBootstrapEvidenceRefV1,
    PredecessorAuthorizationEvidenceRefV1, ScientificViewControlPlaneError,
    SuppliedCandidateSuccessorDispositionV1,
};
use symthaea_scientific_view_profile::{
    AuthoritySourceBindingV1, AuthoritySourceOccurrenceV1, Commitment32,
    RoleSemanticRevisionV1, ScientificAuthorityRoleV1 as Role,
    ScientificViewDeploymentBindingV1, ScientificViewSemanticProfileV1,
};

fn c(byte: u8) -> Commitment32 {
    assert_ne!(byte, 0);
    Commitment32::from_bytes([byte; 32])
}

fn expected(hex: &str) -> Commitment32 {
    assert_eq!(hex.len(), 64);
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap();
    }
    Commitment32::from_bytes(bytes)
}

fn semantics(role: Role, byte: u8) -> RoleSemanticRevisionV1 {
    RoleSemanticRevisionV1::new(role, c(byte)).unwrap()
}

fn profile(use_policy_byte: u8) -> ScientificViewSemanticProfileV1 {
    ScientificViewSemanticProfileV1::new(
        "lunar/site01",
        "site01-confirmatory",
        vec![
            semantics(Role::ResearchSemanticHead, 11),
            semantics(Role::VerifierPolicyHead, 12),
        ],
        c(201),
        c(202),
        c(203),
        c(use_policy_byte),
    )
    .unwrap()
}

fn source(role: Role, id: &str, epoch: u64, byte: u8) -> AuthoritySourceBindingV1 {
    AuthoritySourceBindingV1::new(role, id, epoch, c(byte)).unwrap()
}

fn binding(
    profile: &ScientificViewSemanticProfileV1,
    research_id: &str,
    research_epoch: u64,
    research_qualification: u8,
    verifier_epoch: u64,
) -> ScientificViewDeploymentBindingV1 {
    ScientificViewDeploymentBindingV1::new(
        "deployment/site01-a",
        profile,
        vec![
            source(
                Role::ResearchSemanticHead,
                research_id,
                research_epoch,
                research_qualification,
            ),
            source(
                Role::VerifierPolicyHead,
                "verifier-policy/store-a",
                verifier_epoch,
                22,
            ),
        ],
    )
    .unwrap()
}

fn genesis() -> (
    ScientificViewSemanticProfileV1,
    ScientificViewDeploymentBindingV1,
    DeploymentBootstrapEvidenceRefV1,
    CandidateScientificViewControlPlaneTransitionV1,
) {
    let profile = profile(204);
    let binding = binding(&profile, "research/store-a", 7, 21, 3);
    let bootstrap = DeploymentBootstrapEvidenceRefV1::new(&profile, &binding, c(41), c(42)).unwrap();
    let candidate = CandidateScientificViewControlPlaneTransitionV1::bootstrap_candidate(
        &profile,
        &binding,
        &bootstrap,
    )
    .unwrap();
    (profile, binding, bootstrap, candidate)
}

fn migration_candidate(
    profile: &ScientificViewSemanticProfileV1,
    old_binding: &ScientificViewDeploymentBindingV1,
    predecessor: &CandidateScientificViewControlPlaneTransitionV1,
    new_binding: &ScientificViewDeploymentBindingV1,
    policy_byte: u8,
    evidence_byte: u8,
) -> (
    CandidateSourceMigrationV1,
    PredecessorAuthorizationEvidenceRefV1,
    CandidateScientificViewControlPlaneTransitionV1,
) {
    let old = AuthoritySourceOccurrenceV1::genesis(
        old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new = AuthoritySourceOccurrenceV1::genesis(
        new_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let migration = CandidateSourceMigrationV1::new(
        profile,
        old_binding,
        new_binding,
        Role::ResearchSemanticHead,
        &old,
        &new,
        c(31),
    )
    .unwrap();
    let authorization = PredecessorAuthorizationEvidenceRefV1::for_source_migration(
        predecessor,
        profile,
        new_binding,
        &migration,
        c(policy_byte),
        c(evidence_byte),
    )
    .unwrap();
    let candidate = CandidateScientificViewControlPlaneTransitionV1::source_migration_candidate(
        predecessor,
        profile,
        old_binding,
        new_binding,
        &migration,
        &authorization,
    )
    .unwrap();
    (migration, authorization, candidate)
}

#[test]
fn bootstrap_candidate_is_exactly_bound_and_matches_golden_vector() {
    let (profile, binding, bootstrap, candidate) = genesis();

    assert_eq!(
        bootstrap.commitment(),
        expected("f1285b91dfb497cb0335cb3a5c3626e4b347e9a5e30039fda793f9268e5ed346")
    );
    assert_eq!(
        candidate.commitment(),
        expected("dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8d")
    );
    assert_eq!(candidate.sequence(), 1);
    assert_eq!(candidate.predecessor_transition(), None);
    assert_eq!(candidate.source_profile_commitment(), None);
    assert_eq!(candidate.source_binding_commitment(), None);
    assert_eq!(candidate.destination_profile_commitment(), profile.commitment());
    assert_eq!(candidate.destination_binding_commitment(), binding.commitment());
    assert_eq!(
        candidate.transition_kind(),
        ControlPlaneTransitionKindV1::BootstrapActivation
    );
}

#[test]
fn migration_candidate_preserves_state_but_changes_authority_identity() {
    let (profile, old_binding, _, predecessor) = genesis();
    let new_binding = binding(&profile, "research/store-a", 8, 21, 3);
    let old_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &new_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();

    let (migration, authorization, candidate) = migration_candidate(
        &profile,
        &old_binding,
        &predecessor,
        &new_binding,
        51,
        52,
    );

    assert_eq!(
        new_binding
            .source_for_role(Role::ResearchSemanticHead)
            .unwrap()
            .commitment(),
        expected("39c89e9aa78acd2becc734bfe39517d3a1a7c9f64fb382cece865523e067d25f")
    );
    assert_eq!(
        new_binding.source_roster_commitment(),
        expected("c8ff011c09da4b0cea2ab7cd14a71dcd06e874feb0ae4d241924d93e2bbb2955")
    );
    assert_eq!(
        new_binding.commitment(),
        expected("669bf3a6a47d6c7d685d0e4365bc685987efd0adc0c1e40132343e661e8e08b8")
    );
    assert_eq!(
        new_occurrence.commitment(),
        expected("db5139e72f017fee7413638be1f78331feb5ebd0116bd9afad00d6da1fb0e18c")
    );
    assert_eq!(
        migration.commitment(),
        expected("7e49f3deff9a4862da5084559084f3c6cdea9499f2c19fdb530cf05f00de4283")
    );
    assert_eq!(
        authorization.commitment(),
        expected("b0e959c9907bc59d5976c1796b141acd3d52d6e3a99121b9404fffe917d3dae3")
    );
    assert_eq!(
        candidate.commitment(),
        expected("a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf")
    );

    assert_eq!(old_occurrence.state_commitment(), new_occurrence.state_commitment());
    assert_ne!(old_occurrence.commitment(), new_occurrence.commitment());
    assert_ne!(old_binding.commitment(), new_binding.commitment());
    assert_eq!(candidate.predecessor_transition(), Some(predecessor.commitment()));
}

#[test]
fn same_source_reprovisioning_requires_epoch_advance() {
    let (profile, old_binding, _, _) = genesis();
    let same_epoch_changed_qualification = binding(&profile, "research/store-a", 7, 23, 3);
    let old_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &same_epoch_changed_qualification,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();

    let err = CandidateSourceMigrationV1::new(
        &profile,
        &old_binding,
        &same_epoch_changed_qualification,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_occurrence,
        c(31),
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewControlPlaneError::SourceMigrationEpochDidNotAdvance
    );
}

#[test]
fn one_source_migration_cannot_hide_other_source_changes() {
    let (profile, old_binding, _, _) = genesis();
    let two_changes = binding(&profile, "research/store-a", 8, 21, 4);
    let old_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &two_changes,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();

    let err = CandidateSourceMigrationV1::new(
        &profile,
        &old_binding,
        &two_changes,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_occurrence,
        c(31),
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewControlPlaneError::SourceMigrationChangedUnexpectedRoles
    );
}

#[test]
fn migration_destination_occurrence_must_be_new_genesis() {
    let (profile, old_binding, _, _) = genesis();
    let new_binding = binding(&profile, "research/store-a", 8, 21, 3);
    let old_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_genesis = AuthoritySourceOccurrenceV1::genesis(
        &new_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_successor = new_genesis.successor(&new_binding, c(32)).unwrap();

    let err = CandidateSourceMigrationV1::new(
        &profile,
        &old_binding,
        &new_binding,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_successor,
        c(31),
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewControlPlaneError::NewSourceOccurrenceMustBeGenesis
    );
}

#[test]
fn profile_activation_cannot_smuggle_existing_source_replacement() {
    let (source_profile, source_binding, _, predecessor) = genesis();
    let destination_profile = profile(205);
    let replaced_binding = binding(&destination_profile, "research/store-a", 8, 21, 3);
    let authorization = PredecessorAuthorizationEvidenceRefV1::for_profile_activation(
        &predecessor,
        &destination_profile,
        &replaced_binding,
        c(61),
        c(62),
    )
    .unwrap();

    let err = CandidateScientificViewControlPlaneTransitionV1::profile_activation_candidate(
        &predecessor,
        &source_profile,
        &source_binding,
        &destination_profile,
        &replaced_binding,
        &authorization,
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewControlPlaneError::ProfileActivationChangedExistingSource
    );
}

#[test]
fn authorization_reference_is_bound_to_exact_successor_intent() {
    let (profile, old_binding, _, predecessor) = genesis();
    let destination_a = binding(&profile, "research/store-a", 8, 21, 3);
    let destination_b = binding(&profile, "research/store-b", 1, 21, 3);
    let old_occurrence = AuthoritySourceOccurrenceV1::genesis(
        &old_binding,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_a = AuthoritySourceOccurrenceV1::genesis(
        &destination_a,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let new_b = AuthoritySourceOccurrenceV1::genesis(
        &destination_b,
        Role::ResearchSemanticHead,
        c(31),
    )
    .unwrap();
    let migration_a = CandidateSourceMigrationV1::new(
        &profile,
        &old_binding,
        &destination_a,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_a,
        c(31),
    )
    .unwrap();
    let migration_b = CandidateSourceMigrationV1::new(
        &profile,
        &old_binding,
        &destination_b,
        Role::ResearchSemanticHead,
        &old_occurrence,
        &new_b,
        c(31),
    )
    .unwrap();
    let authorization_a = PredecessorAuthorizationEvidenceRefV1::for_source_migration(
        &predecessor,
        &profile,
        &destination_a,
        &migration_a,
        c(51),
        c(52),
    )
    .unwrap();

    let err = CandidateScientificViewControlPlaneTransitionV1::source_migration_candidate(
        &predecessor,
        &profile,
        &old_binding,
        &destination_b,
        &migration_b,
        &authorization_a,
    )
    .unwrap_err();

    assert_eq!(
        err,
        ScientificViewControlPlaneError::AuthorizationIntentMismatch
    );
}

#[test]
fn same_predecessor_multiple_candidates_do_not_imply_committed_fork() {
    let (profile, old_binding, _, predecessor) = genesis();
    let binding_a = binding(&profile, "research/store-a", 8, 21, 3);
    let binding_b = binding(&profile, "research/store-b", 1, 21, 3);
    let (_, _, candidate_a) = migration_candidate(
        &profile,
        &old_binding,
        &predecessor,
        &binding_a,
        51,
        52,
    );
    let (_, _, candidate_b) = migration_candidate(
        &profile,
        &old_binding,
        &predecessor,
        &binding_b,
        53,
        54,
    );

    let disposition = classify_supplied_candidate_successors(
        &predecessor,
        &[candidate_b.clone(), candidate_a.clone(), candidate_a.clone()],
    )
    .unwrap();

    match disposition {
        SuppliedCandidateSuccessorDispositionV1::MultipleCandidatesInSuppliedSet {
            candidates,
        } => {
            assert_eq!(candidates.len(), 2);
            assert!(candidates.contains(&candidate_a.commitment()));
            assert!(candidates.contains(&candidate_b.commitment()));
        }
        other => panic!("expected multiple supplied candidates, got {other:?}"),
    }
}

#[test]
fn unique_supplied_candidate_is_not_occurrence_or_currentness() {
    let (profile, old_binding, _, predecessor) = genesis();
    let new_binding = binding(&profile, "research/store-a", 8, 21, 3);
    let (_, _, candidate) = migration_candidate(
        &profile,
        &old_binding,
        &predecessor,
        &new_binding,
        51,
        52,
    );

    assert_eq!(
        classify_supplied_candidate_successors(&predecessor, &[candidate.clone()]).unwrap(),
        SuppliedCandidateSuccessorDispositionV1::UniqueInSuppliedSet {
            transition: candidate.commitment()
        }
    );

    let source = include_str!("../src/lib.rs");
    let manifest = include_str!("../Cargo.toml");

    assert!(source.contains("mod mechanics;"));
    assert!(!source.contains("pub mod mechanics;"));
    assert!(!source.contains("pub struct HistoricalScientificViewControlPlaneTransitionV1"));
    assert!(!source.contains("pub struct QualifiedScientificViewControlPlaneHeadV1"));
    assert!(!source.contains("pub fn is_current"));
    assert!(!source.contains("CurrentScientificAuthorityEnvelope"));
    assert!(!source.contains("ActionAuthority"));
    assert!(!manifest.contains("serde"));
}
