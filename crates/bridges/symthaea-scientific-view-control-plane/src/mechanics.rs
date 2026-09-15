//! Historical SCI-014 scientific-view control-plane transition mechanics.
//!
//! This crate deliberately stops before live/current authority.
//!
//! It provides:
//! - provisioning evidence bound to one exact candidate profile/binding;
//! - predecessor authorization evidence bound to one exact successor intent;
//! - exact source-migration evidence;
//! - append-only, exact-predecessor historical control-plane transitions;
//! - supplied-set fork/contention classification.
//!
//! It does **not** authenticate an authority source, decide which historical
//! transition is current, prove candidate-set completeness, mint a live control-
//! plane head, capture a coherent scientific view, establish scientific truth,
//! or grant external action authority.
//!
//! Architecture: SCI-014R / #3413.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_scientific_view_profile::{
    AuthoritySourceOccurrenceV1, Commitment32, ScientificAuthorityRoleV1,
    ScientificViewDeploymentBindingV1, ScientificViewSemanticProfileV1,
};

pub const COMMITMENT_ALGORITHM_V1: &str = "sha256";

const BOOTSTRAP_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.bootstrap-evidence.v1";
const PREDECESSOR_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.predecessor-evidence.v1";
const SOURCE_MIGRATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.science.view.control-plane.source-migration-evidence.v1";
const TRANSITION_DOMAIN: &[u8] = b"symthaea.science.view.control-plane.transition.v1";

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControlPlaneTransitionKindV1 {
    BootstrapActivation = 1,
    ProfileActivation = 2,
    SourceMigration = 3,
}

impl ControlPlaneTransitionKindV1 {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificViewControlPlaneError {
    ZeroCommitment { field: &'static str },
    ProfileBindingMismatch,
    ScopeMismatch,
    BootstrapCandidateMismatch,
    PredecessorStateMismatch,
    AuthorizationPredecessorMismatch,
    AuthorizationIntentMismatch,
    LineageOverflow,
    NoOpTransition,
    ProfileActivationRequiresProfileChange,
    ProfileActivationChangedExistingSource,
    SourceMigrationRequiresStableProfile,
    SourceMigrationRoleUnchanged,
    SourceMigrationChangedUnexpectedRoles,
    SourceMigrationEpochDidNotAdvance,
    OldSourceOccurrenceMismatch,
    NewSourceOccurrenceMismatch,
    NewSourceOccurrenceMustBeGenesis,
    TransferredStateMismatch,
    SuppliedCandidateNotImmediateSuccessor,
}

/// Historical provisioning evidence scoped to one exact candidate profile and
/// deployment binding.
///
/// This value is a content-addressed evidence reference. Constructing it does
/// not authenticate the bootstrap trust root and does not activate the
/// candidate profile/binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentBootstrapEvidenceV1 {
    deployment_id: String,
    view_namespace: String,
    candidate_profile_commitment: Commitment32,
    candidate_binding_commitment: Commitment32,
    bootstrap_trust_root_commitment: Commitment32,
    provisioning_evidence_commitment: Commitment32,
    commitment: Commitment32,
}

impl DeploymentBootstrapEvidenceV1 {
    pub fn new(
        profile: &ScientificViewSemanticProfileV1,
        binding: &ScientificViewDeploymentBindingV1,
        bootstrap_trust_root_commitment: Commitment32,
        provisioning_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_profile_binding(profile, binding)?;
        require_nonzero(
            "bootstrap_trust_root_commitment",
            bootstrap_trust_root_commitment,
        )?;
        require_nonzero(
            "provisioning_evidence_commitment",
            provisioning_evidence_commitment,
        )?;

        let commitment = hash_with(BOOTSTRAP_EVIDENCE_DOMAIN, |hasher| {
            put_text(hasher, binding.deployment_id());
            put_text(hasher, binding.view_namespace());
            put_commitment(hasher, profile.commitment());
            put_commitment(hasher, binding.commitment());
            put_commitment(hasher, bootstrap_trust_root_commitment);
            put_commitment(hasher, provisioning_evidence_commitment);
        });

        Ok(Self {
            deployment_id: binding.deployment_id().to_owned(),
            view_namespace: binding.view_namespace().to_owned(),
            candidate_profile_commitment: profile.commitment(),
            candidate_binding_commitment: binding.commitment(),
            bootstrap_trust_root_commitment,
            provisioning_evidence_commitment,
            commitment,
        })
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn view_namespace(&self) -> &str {
        &self.view_namespace
    }

    pub const fn candidate_profile_commitment(&self) -> Commitment32 {
        self.candidate_profile_commitment
    }

    pub const fn candidate_binding_commitment(&self) -> Commitment32 {
        self.candidate_binding_commitment
    }

    pub const fn bootstrap_trust_root_commitment(&self) -> Commitment32 {
        self.bootstrap_trust_root_commitment
    }

    pub const fn provisioning_evidence_commitment(&self) -> Commitment32 {
        self.provisioning_evidence_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Exact evidence for replacing/reprovisioning one mutable authority source.
///
/// The destination occurrence must be genesis under the new deployment binding.
/// Equal logical state is permitted, but equal state does not preserve source
/// occurrence identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMigrationEvidenceV1 {
    role: ScientificAuthorityRoleV1,
    old_source_binding_commitment: Commitment32,
    new_source_binding_commitment: Commitment32,
    transferred_state_commitment: Commitment32,
    old_source_occurrence_commitment: Commitment32,
    new_source_occurrence_commitment: Commitment32,
    commitment: Commitment32,
}

impl SourceMigrationEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: &ScientificViewSemanticProfileV1,
        old_binding: &ScientificViewDeploymentBindingV1,
        new_binding: &ScientificViewDeploymentBindingV1,
        role: ScientificAuthorityRoleV1,
        old_occurrence: &AuthoritySourceOccurrenceV1,
        new_occurrence: &AuthoritySourceOccurrenceV1,
        transferred_state_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_profile_binding(profile, old_binding)?;
        validate_profile_binding(profile, new_binding)?;
        require_same_scope(old_binding, new_binding)?;
        require_nonzero(
            "transferred_state_commitment",
            transferred_state_commitment,
        )?;

        let old_source = old_binding
            .source_for_role(role)
            .ok_or(ScientificViewControlPlaneError::SourceMigrationRoleUnchanged)?;
        let new_source = new_binding
            .source_for_role(role)
            .ok_or(ScientificViewControlPlaneError::SourceMigrationRoleUnchanged)?;

        if old_source == new_source {
            return Err(ScientificViewControlPlaneError::SourceMigrationRoleUnchanged);
        }

        let mut changed_roles = 0usize;
        for required_role in profile.required_roles() {
            let old = old_binding
                .source_for_role(required_role)
                .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;
            let new = new_binding
                .source_for_role(required_role)
                .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;
            if old != new {
                changed_roles += 1;
                if required_role != role {
                    return Err(
                        ScientificViewControlPlaneError::SourceMigrationChangedUnexpectedRoles,
                    );
                }
            }
        }

        if changed_roles != 1 {
            return Err(
                ScientificViewControlPlaneError::SourceMigrationChangedUnexpectedRoles,
            );
        }

        if old_source.source_identity() == new_source.source_identity()
            && new_source.provisioning_epoch() <= old_source.provisioning_epoch()
        {
            return Err(
                ScientificViewControlPlaneError::SourceMigrationEpochDidNotAdvance,
            );
        }

        validate_occurrence_against_binding(old_occurrence, old_binding, role)
            .map_err(|_| ScientificViewControlPlaneError::OldSourceOccurrenceMismatch)?;
        validate_occurrence_against_binding(new_occurrence, new_binding, role)
            .map_err(|_| ScientificViewControlPlaneError::NewSourceOccurrenceMismatch)?;

        if new_occurrence.lineage_position() != 1
            || new_occurrence.predecessor_occurrence().is_some()
        {
            return Err(
                ScientificViewControlPlaneError::NewSourceOccurrenceMustBeGenesis,
            );
        }

        if old_occurrence.state_commitment() != transferred_state_commitment
            || new_occurrence.state_commitment() != transferred_state_commitment
        {
            return Err(ScientificViewControlPlaneError::TransferredStateMismatch);
        }

        let commitment = hash_with(SOURCE_MIGRATION_EVIDENCE_DOMAIN, |hasher| {
            put_u8(hasher, role.code());
            put_commitment(hasher, old_source.commitment());
            put_commitment(hasher, new_source.commitment());
            put_commitment(hasher, transferred_state_commitment);
            put_commitment(hasher, old_occurrence.commitment());
            put_commitment(hasher, new_occurrence.commitment());
        });

        Ok(Self {
            role,
            old_source_binding_commitment: old_source.commitment(),
            new_source_binding_commitment: new_source.commitment(),
            transferred_state_commitment,
            old_source_occurrence_commitment: old_occurrence.commitment(),
            new_source_occurrence_commitment: new_occurrence.commitment(),
            commitment,
        })
    }

    pub const fn role(&self) -> ScientificAuthorityRoleV1 {
        self.role
    }

    pub const fn old_source_binding_commitment(&self) -> Commitment32 {
        self.old_source_binding_commitment
    }

    pub const fn new_source_binding_commitment(&self) -> Commitment32 {
        self.new_source_binding_commitment
    }

    pub const fn transferred_state_commitment(&self) -> Commitment32 {
        self.transferred_state_commitment
    }

    pub const fn old_source_occurrence_commitment(&self) -> Commitment32 {
        self.old_source_occurrence_commitment
    }

    pub const fn new_source_occurrence_commitment(&self) -> Commitment32 {
        self.new_source_occurrence_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Historical predecessor-authorization evidence bound to one exact successor
/// intent.
///
/// This object binds intent; it does not prove that the policy/evidence
/// commitments name authentic or current authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredecessorAuthorizationEvidenceV1 {
    authorizing_transition: Commitment32,
    transition_kind: ControlPlaneTransitionKindV1,
    destination_profile_commitment: Commitment32,
    destination_binding_commitment: Commitment32,
    migration_evidence_commitment: Option<Commitment32>,
    authorization_policy_commitment: Commitment32,
    authorization_evidence_commitment: Commitment32,
    commitment: Commitment32,
}

impl PredecessorAuthorizationEvidenceV1 {
    pub fn for_profile_activation(
        predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
        destination_profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        authorization_policy_commitment: Commitment32,
        authorization_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_profile_binding(destination_profile, destination_binding)?;
        require_destination_scope(predecessor, destination_binding)?;

        Self::build(
            predecessor,
            ControlPlaneTransitionKindV1::ProfileActivation,
            destination_profile.commitment(),
            destination_binding.commitment(),
            None,
            authorization_policy_commitment,
            authorization_evidence_commitment,
        )
    }

    pub fn for_source_migration(
        predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
        profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        migration: &SourceMigrationEvidenceV1,
        authorization_policy_commitment: Commitment32,
        authorization_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_profile_binding(profile, destination_binding)?;
        require_destination_scope(predecessor, destination_binding)?;

        let destination_source = destination_binding
            .source_for_role(migration.role())
            .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;
        if destination_source.commitment() != migration.new_source_binding_commitment() {
            return Err(ScientificViewControlPlaneError::AuthorizationIntentMismatch);
        }

        Self::build(
            predecessor,
            ControlPlaneTransitionKindV1::SourceMigration,
            profile.commitment(),
            destination_binding.commitment(),
            Some(migration.commitment()),
            authorization_policy_commitment,
            authorization_evidence_commitment,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
        transition_kind: ControlPlaneTransitionKindV1,
        destination_profile_commitment: Commitment32,
        destination_binding_commitment: Commitment32,
        migration_evidence_commitment: Option<Commitment32>,
        authorization_policy_commitment: Commitment32,
        authorization_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        require_nonzero(
            "authorization_policy_commitment",
            authorization_policy_commitment,
        )?;
        require_nonzero(
            "authorization_evidence_commitment",
            authorization_evidence_commitment,
        )?;

        let commitment = hash_with(PREDECESSOR_EVIDENCE_DOMAIN, |hasher| {
            put_commitment(hasher, predecessor.commitment());
            put_u8(hasher, transition_kind.code());
            put_commitment(hasher, destination_profile_commitment);
            put_commitment(hasher, destination_binding_commitment);
            put_optional_commitment(hasher, migration_evidence_commitment);
            put_commitment(hasher, authorization_policy_commitment);
            put_commitment(hasher, authorization_evidence_commitment);
        });

        Ok(Self {
            authorizing_transition: predecessor.commitment(),
            transition_kind,
            destination_profile_commitment,
            destination_binding_commitment,
            migration_evidence_commitment,
            authorization_policy_commitment,
            authorization_evidence_commitment,
            commitment,
        })
    }

    pub const fn authorizing_transition(&self) -> Commitment32 {
        self.authorizing_transition
    }

    pub const fn transition_kind(&self) -> ControlPlaneTransitionKindV1 {
        self.transition_kind
    }

    pub const fn destination_profile_commitment(&self) -> Commitment32 {
        self.destination_profile_commitment
    }

    pub const fn destination_binding_commitment(&self) -> Commitment32 {
        self.destination_binding_commitment
    }

    pub const fn migration_evidence_commitment(&self) -> Option<Commitment32> {
        self.migration_evidence_commitment
    }

    pub const fn authorization_policy_commitment(&self) -> Commitment32 {
        self.authorization_policy_commitment
    }

    pub const fn authorization_evidence_commitment(&self) -> Commitment32 {
        self.authorization_evidence_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Structurally checked historical control-plane transition.
///
/// Positive current authority is intentionally absent. A transition may be
/// historically valid while no present-current control-plane head is available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalScientificViewControlPlaneTransitionV1 {
    deployment_id: String,
    view_namespace: String,
    sequence: u64,
    predecessor_transition: Option<Commitment32>,
    source_profile_commitment: Option<Commitment32>,
    source_binding_commitment: Option<Commitment32>,
    destination_profile_commitment: Commitment32,
    destination_binding_commitment: Commitment32,
    transition_kind: ControlPlaneTransitionKindV1,
    authorization_evidence_commitment: Commitment32,
    migration_evidence_commitment: Option<Commitment32>,
    commitment: Commitment32,
}

impl HistoricalScientificViewControlPlaneTransitionV1 {
    pub fn bootstrap_activate(
        profile: &ScientificViewSemanticProfileV1,
        binding: &ScientificViewDeploymentBindingV1,
        bootstrap: &DeploymentBootstrapEvidenceV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_profile_binding(profile, binding)?;

        if bootstrap.deployment_id() != binding.deployment_id()
            || bootstrap.view_namespace() != binding.view_namespace()
            || bootstrap.candidate_profile_commitment() != profile.commitment()
            || bootstrap.candidate_binding_commitment() != binding.commitment()
        {
            return Err(ScientificViewControlPlaneError::BootstrapCandidateMismatch);
        }

        Ok(Self::build(
            binding.deployment_id(),
            binding.view_namespace(),
            1,
            None,
            None,
            None,
            profile.commitment(),
            binding.commitment(),
            ControlPlaneTransitionKindV1::BootstrapActivation,
            bootstrap.commitment(),
            None,
        ))
    }

    pub fn activate_profile(
        predecessor: &Self,
        source_profile: &ScientificViewSemanticProfileV1,
        source_binding: &ScientificViewDeploymentBindingV1,
        destination_profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        authorization: &PredecessorAuthorizationEvidenceV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_predecessor_source(predecessor, source_profile, source_binding)?;
        validate_profile_binding(destination_profile, destination_binding)?;
        require_destination_scope(predecessor, destination_binding)?;

        if destination_profile.commitment() == source_profile.commitment() {
            return Err(
                ScientificViewControlPlaneError::ProfileActivationRequiresProfileChange,
            );
        }

        validate_profile_activation_source_continuity(
            source_profile,
            source_binding,
            destination_profile,
            destination_binding,
        )?;

        validate_authorization_intent(
            predecessor,
            destination_profile.commitment(),
            destination_binding.commitment(),
            ControlPlaneTransitionKindV1::ProfileActivation,
            None,
            authorization,
        )?;

        let sequence = predecessor
            .sequence
            .checked_add(1)
            .ok_or(ScientificViewControlPlaneError::LineageOverflow)?;

        Ok(Self::build(
            predecessor.deployment_id(),
            predecessor.view_namespace(),
            sequence,
            Some(predecessor.commitment()),
            Some(source_profile.commitment()),
            Some(source_binding.commitment()),
            destination_profile.commitment(),
            destination_binding.commitment(),
            ControlPlaneTransitionKindV1::ProfileActivation,
            authorization.commitment(),
            None,
        ))
    }

    pub fn migrate_source(
        predecessor: &Self,
        profile: &ScientificViewSemanticProfileV1,
        old_binding: &ScientificViewDeploymentBindingV1,
        new_binding: &ScientificViewDeploymentBindingV1,
        migration: &SourceMigrationEvidenceV1,
        authorization: &PredecessorAuthorizationEvidenceV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        validate_predecessor_source(predecessor, profile, old_binding)?;
        validate_profile_binding(profile, new_binding)?;
        require_destination_scope(predecessor, new_binding)?;

        if old_binding.semantic_profile_commitment() != new_binding.semantic_profile_commitment()
            || profile.commitment() != predecessor.destination_profile_commitment
        {
            return Err(
                ScientificViewControlPlaneError::SourceMigrationRequiresStableProfile,
            );
        }

        if old_binding.commitment() == new_binding.commitment() {
            return Err(ScientificViewControlPlaneError::NoOpTransition);
        }

        let old_source = old_binding
            .source_for_role(migration.role())
            .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;
        let new_source = new_binding
            .source_for_role(migration.role())
            .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;

        if old_source.commitment() != migration.old_source_binding_commitment()
            || new_source.commitment() != migration.new_source_binding_commitment()
        {
            return Err(ScientificViewControlPlaneError::AuthorizationIntentMismatch);
        }

        validate_authorization_intent(
            predecessor,
            profile.commitment(),
            new_binding.commitment(),
            ControlPlaneTransitionKindV1::SourceMigration,
            Some(migration.commitment()),
            authorization,
        )?;

        let sequence = predecessor
            .sequence
            .checked_add(1)
            .ok_or(ScientificViewControlPlaneError::LineageOverflow)?;

        Ok(Self::build(
            predecessor.deployment_id(),
            predecessor.view_namespace(),
            sequence,
            Some(predecessor.commitment()),
            Some(profile.commitment()),
            Some(old_binding.commitment()),
            profile.commitment(),
            new_binding.commitment(),
            ControlPlaneTransitionKindV1::SourceMigration,
            authorization.commitment(),
            Some(migration.commitment()),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        deployment_id: &str,
        view_namespace: &str,
        sequence: u64,
        predecessor_transition: Option<Commitment32>,
        source_profile_commitment: Option<Commitment32>,
        source_binding_commitment: Option<Commitment32>,
        destination_profile_commitment: Commitment32,
        destination_binding_commitment: Commitment32,
        transition_kind: ControlPlaneTransitionKindV1,
        authorization_evidence_commitment: Commitment32,
        migration_evidence_commitment: Option<Commitment32>,
    ) -> Self {
        let commitment = hash_with(TRANSITION_DOMAIN, |hasher| {
            put_text(hasher, deployment_id);
            put_text(hasher, view_namespace);
            put_u64(hasher, sequence);
            put_optional_commitment(hasher, predecessor_transition);
            put_optional_commitment(hasher, source_profile_commitment);
            put_optional_commitment(hasher, source_binding_commitment);
            put_commitment(hasher, destination_profile_commitment);
            put_commitment(hasher, destination_binding_commitment);
            put_u8(hasher, transition_kind.code());
            put_commitment(hasher, authorization_evidence_commitment);
            put_optional_commitment(hasher, migration_evidence_commitment);
        });

        Self {
            deployment_id: deployment_id.to_owned(),
            view_namespace: view_namespace.to_owned(),
            sequence,
            predecessor_transition,
            source_profile_commitment,
            source_binding_commitment,
            destination_profile_commitment,
            destination_binding_commitment,
            transition_kind,
            authorization_evidence_commitment,
            migration_evidence_commitment,
            commitment,
        }
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn view_namespace(&self) -> &str {
        &self.view_namespace
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn predecessor_transition(&self) -> Option<Commitment32> {
        self.predecessor_transition
    }

    pub const fn source_profile_commitment(&self) -> Option<Commitment32> {
        self.source_profile_commitment
    }

    pub const fn source_binding_commitment(&self) -> Option<Commitment32> {
        self.source_binding_commitment
    }

    pub const fn destination_profile_commitment(&self) -> Commitment32 {
        self.destination_profile_commitment
    }

    pub const fn destination_binding_commitment(&self) -> Commitment32 {
        self.destination_binding_commitment
    }

    pub const fn transition_kind(&self) -> ControlPlaneTransitionKindV1 {
        self.transition_kind
    }

    pub const fn authorization_evidence_commitment(&self) -> Commitment32 {
        self.authorization_evidence_commitment
    }

    pub const fn migration_evidence_commitment(&self) -> Option<Commitment32> {
        self.migration_evidence_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// What can be concluded only from a caller-supplied successor set.
///
/// `UniqueInSuppliedSet` does not prove that no omitted successor exists and is
/// therefore not a current-head capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuppliedControlPlaneSuccessorDispositionV1 {
    NoSuppliedSuccessor,
    UniqueInSuppliedSet {
        transition: Commitment32,
    },
    ContestedInSuppliedSet {
        candidates: Vec<Commitment32>,
    },
}

pub fn classify_supplied_successors(
    predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
    candidates: &[HistoricalScientificViewControlPlaneTransitionV1],
) -> Result<SuppliedControlPlaneSuccessorDispositionV1, ScientificViewControlPlaneError> {
    let expected_sequence = predecessor
        .sequence()
        .checked_add(1)
        .ok_or(ScientificViewControlPlaneError::LineageOverflow)?;

    let mut unique = BTreeSet::new();
    for candidate in candidates {
        if candidate.deployment_id() != predecessor.deployment_id()
            || candidate.view_namespace() != predecessor.view_namespace()
            || candidate.sequence() != expected_sequence
            || candidate.predecessor_transition() != Some(predecessor.commitment())
            || candidate.source_profile_commitment()
                != Some(predecessor.destination_profile_commitment())
            || candidate.source_binding_commitment()
                != Some(predecessor.destination_binding_commitment())
        {
            return Err(
                ScientificViewControlPlaneError::SuppliedCandidateNotImmediateSuccessor,
            );
        }
        unique.insert(candidate.commitment());
    }

    Ok(match unique.len() {
        0 => SuppliedControlPlaneSuccessorDispositionV1::NoSuppliedSuccessor,
        1 => SuppliedControlPlaneSuccessorDispositionV1::UniqueInSuppliedSet {
            transition: *unique.iter().next().expect("one element by length"),
        },
        _ => SuppliedControlPlaneSuccessorDispositionV1::ContestedInSuppliedSet {
            candidates: unique.into_iter().collect(),
        },
    })
}

fn validate_profile_binding(
    profile: &ScientificViewSemanticProfileV1,
    binding: &ScientificViewDeploymentBindingV1,
) -> Result<(), ScientificViewControlPlaneError> {
    if binding.view_namespace() != profile.view_namespace()
        || binding.semantic_profile_commitment() != profile.commitment()
    {
        return Err(ScientificViewControlPlaneError::ProfileBindingMismatch);
    }
    Ok(())
}

fn require_same_scope(
    left: &ScientificViewDeploymentBindingV1,
    right: &ScientificViewDeploymentBindingV1,
) -> Result<(), ScientificViewControlPlaneError> {
    if left.deployment_id() != right.deployment_id()
        || left.view_namespace() != right.view_namespace()
    {
        return Err(ScientificViewControlPlaneError::ScopeMismatch);
    }
    Ok(())
}

fn require_destination_scope(
    predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
    destination_binding: &ScientificViewDeploymentBindingV1,
) -> Result<(), ScientificViewControlPlaneError> {
    if predecessor.deployment_id() != destination_binding.deployment_id()
        || predecessor.view_namespace() != destination_binding.view_namespace()
    {
        return Err(ScientificViewControlPlaneError::ScopeMismatch);
    }
    Ok(())
}

fn validate_predecessor_source(
    predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
    source_profile: &ScientificViewSemanticProfileV1,
    source_binding: &ScientificViewDeploymentBindingV1,
) -> Result<(), ScientificViewControlPlaneError> {
    validate_profile_binding(source_profile, source_binding)?;
    require_destination_scope(predecessor, source_binding)?;

    if predecessor.destination_profile_commitment() != source_profile.commitment()
        || predecessor.destination_binding_commitment() != source_binding.commitment()
    {
        return Err(ScientificViewControlPlaneError::PredecessorStateMismatch);
    }
    Ok(())
}

fn validate_profile_activation_source_continuity(
    source_profile: &ScientificViewSemanticProfileV1,
    source_binding: &ScientificViewDeploymentBindingV1,
    destination_profile: &ScientificViewSemanticProfileV1,
    destination_binding: &ScientificViewDeploymentBindingV1,
) -> Result<(), ScientificViewControlPlaneError> {
    for role in source_profile.required_roles() {
        if !destination_profile.requires_role(role) {
            continue;
        }

        let source = source_binding
            .source_for_role(role)
            .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;
        let destination = destination_binding
            .source_for_role(role)
            .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;

        if source != destination {
            return Err(
                ScientificViewControlPlaneError::ProfileActivationChangedExistingSource,
            );
        }
    }

    Ok(())
}

fn validate_occurrence_against_binding(
    occurrence: &AuthoritySourceOccurrenceV1,
    binding: &ScientificViewDeploymentBindingV1,
    role: ScientificAuthorityRoleV1,
) -> Result<(), ScientificViewControlPlaneError> {
    let source = binding
        .source_for_role(role)
        .ok_or(ScientificViewControlPlaneError::ProfileBindingMismatch)?;

    if occurrence.deployment_binding_commitment() != binding.commitment()
        || occurrence.role() != role
        || occurrence.source_identity() != source.source_identity()
        || occurrence.provisioning_epoch() != source.provisioning_epoch()
    {
        return Err(ScientificViewControlPlaneError::ProfileBindingMismatch);
    }
    Ok(())
}

fn validate_authorization_intent(
    predecessor: &HistoricalScientificViewControlPlaneTransitionV1,
    destination_profile_commitment: Commitment32,
    destination_binding_commitment: Commitment32,
    transition_kind: ControlPlaneTransitionKindV1,
    migration_evidence_commitment: Option<Commitment32>,
    authorization: &PredecessorAuthorizationEvidenceV1,
) -> Result<(), ScientificViewControlPlaneError> {
    if authorization.authorizing_transition() != predecessor.commitment() {
        return Err(
            ScientificViewControlPlaneError::AuthorizationPredecessorMismatch,
        );
    }

    if authorization.transition_kind() != transition_kind
        || authorization.destination_profile_commitment() != destination_profile_commitment
        || authorization.destination_binding_commitment() != destination_binding_commitment
        || authorization.migration_evidence_commitment() != migration_evidence_commitment
    {
        return Err(ScientificViewControlPlaneError::AuthorizationIntentMismatch);
    }

    Ok(())
}

fn require_nonzero(
    field: &'static str,
    commitment: Commitment32,
) -> Result<(), ScientificViewControlPlaneError> {
    if commitment.is_zero() {
        return Err(ScientificViewControlPlaneError::ZeroCommitment { field });
    }
    Ok(())
}

fn hash_with(domain: &[u8], write_fields: impl FnOnce(&mut Sha256)) -> Commitment32 {
    let mut hasher = Sha256::new();
    put_bytes(&mut hasher, domain);
    write_fields(&mut hasher);
    Commitment32::from_bytes(hasher.finalize().into())
}

fn put_u8(hasher: &mut Sha256, value: u8) {
    hasher.update([value]);
}

fn put_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn put_bytes(hasher: &mut Sha256, value: &[u8]) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn put_text(hasher: &mut Sha256, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

fn put_commitment(hasher: &mut Sha256, value: Commitment32) {
    hasher.update(value.as_bytes());
}

fn put_optional_commitment(hasher: &mut Sha256, value: Option<Commitment32>) {
    match value {
        None => put_u8(hasher, 0),
        Some(commitment) => {
            put_u8(hasher, 1);
            put_commitment(hasher, commitment);
        }
    }
}
