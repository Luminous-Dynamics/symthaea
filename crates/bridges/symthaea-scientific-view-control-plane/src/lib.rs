//! Candidate SCI-014 scientific-view control-plane transition identities.
//!
//! This crate deliberately separates three theorem levels:
//!
//! ```text
//! candidate transition identity
//!     != committed historical occurrence
//!     != current live control-plane head
//! ```
//!
//! The private `mechanics` module performs deterministic structural checks and
//! commitment construction. This public surface exposes only candidate/ref
//! semantics so callers cannot mistake offline-constructible bytes for proof
//! that a transition occurred or is current.
//!
//! Architecture: SCI-014R / #3413.

#![forbid(unsafe_code)]

mod mechanics;

use symthaea_scientific_view_profile::{
    AuthoritySourceOccurrenceV1, Commitment32, ScientificAuthorityRoleV1,
    ScientificViewDeploymentBindingV1, ScientificViewSemanticProfileV1,
};

pub use mechanics::{
    ControlPlaneTransitionKindV1, ScientificViewControlPlaneError, COMMITMENT_ALGORITHM_V1,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentBootstrapEvidenceRefV1(mechanics::DeploymentBootstrapEvidenceV1);

impl DeploymentBootstrapEvidenceRefV1 {
    pub fn new(
        profile: &ScientificViewSemanticProfileV1,
        binding: &ScientificViewDeploymentBindingV1,
        bootstrap_trust_root_commitment: Commitment32,
        provisioning_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::DeploymentBootstrapEvidenceV1::new(
            profile,
            binding,
            bootstrap_trust_root_commitment,
            provisioning_evidence_commitment,
        )
        .map(Self)
    }

    pub fn deployment_id(&self) -> &str {
        self.0.deployment_id()
    }

    pub fn view_namespace(&self) -> &str {
        self.0.view_namespace()
    }

    pub const fn candidate_profile_commitment(&self) -> Commitment32 {
        self.0.candidate_profile_commitment()
    }

    pub const fn candidate_binding_commitment(&self) -> Commitment32 {
        self.0.candidate_binding_commitment()
    }

    pub const fn bootstrap_trust_root_commitment(&self) -> Commitment32 {
        self.0.bootstrap_trust_root_commitment()
    }

    pub const fn provisioning_evidence_commitment(&self) -> Commitment32 {
        self.0.provisioning_evidence_commitment()
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.0.commitment()
    }
}

/// Structurally checked source-migration candidate.
///
/// This binds exact old/new source identities and equal transferred-state
/// commitments. It does not prove that a transfer occurred or that the old
/// occurrence was current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSourceMigrationV1(mechanics::SourceMigrationEvidenceV1);

impl CandidateSourceMigrationV1 {
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
        mechanics::SourceMigrationEvidenceV1::new(
            profile,
            old_binding,
            new_binding,
            role,
            old_occurrence,
            new_occurrence,
            transferred_state_commitment,
        )
        .map(Self)
    }

    pub const fn role(&self) -> ScientificAuthorityRoleV1 {
        self.0.role()
    }

    pub const fn old_source_binding_commitment(&self) -> Commitment32 {
        self.0.old_source_binding_commitment()
    }

    pub const fn new_source_binding_commitment(&self) -> Commitment32 {
        self.0.new_source_binding_commitment()
    }

    pub const fn transferred_state_commitment(&self) -> Commitment32 {
        self.0.transferred_state_commitment()
    }

    pub const fn old_source_occurrence_commitment(&self) -> Commitment32 {
        self.0.old_source_occurrence_commitment()
    }

    pub const fn new_source_occurrence_commitment(&self) -> Commitment32 {
        self.0.new_source_occurrence_commitment()
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.0.commitment()
    }
}

/// Reference to predecessor-authorization material scoped to one exact
/// successor intent.
///
/// The referenced policy/evidence is not authenticated merely by constructing
/// this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredecessorAuthorizationEvidenceRefV1(
    mechanics::PredecessorAuthorizationEvidenceV1,
);

impl PredecessorAuthorizationEvidenceRefV1 {
    pub fn for_profile_activation(
        predecessor: &CandidateScientificViewControlPlaneTransitionV1,
        destination_profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        authorization_policy_commitment: Commitment32,
        authorization_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::PredecessorAuthorizationEvidenceV1::for_profile_activation(
            &predecessor.0,
            destination_profile,
            destination_binding,
            authorization_policy_commitment,
            authorization_evidence_commitment,
        )
        .map(Self)
    }

    pub fn for_source_migration(
        predecessor: &CandidateScientificViewControlPlaneTransitionV1,
        profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        migration: &CandidateSourceMigrationV1,
        authorization_policy_commitment: Commitment32,
        authorization_evidence_commitment: Commitment32,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::PredecessorAuthorizationEvidenceV1::for_source_migration(
            &predecessor.0,
            profile,
            destination_binding,
            &migration.0,
            authorization_policy_commitment,
            authorization_evidence_commitment,
        )
        .map(Self)
    }

    pub const fn authorizing_transition(&self) -> Commitment32 {
        self.0.authorizing_transition()
    }

    pub const fn transition_kind(&self) -> ControlPlaneTransitionKindV1 {
        self.0.transition_kind()
    }

    pub const fn destination_profile_commitment(&self) -> Commitment32 {
        self.0.destination_profile_commitment()
    }

    pub const fn destination_binding_commitment(&self) -> Commitment32 {
        self.0.destination_binding_commitment()
    }

    pub const fn migration_evidence_commitment(&self) -> Option<Commitment32> {
        self.0.migration_evidence_commitment()
    }

    pub const fn authorization_policy_commitment(&self) -> Commitment32 {
        self.0.authorization_policy_commitment()
    }

    pub const fn authorization_evidence_commitment(&self) -> Commitment32 {
        self.0.authorization_evidence_commitment()
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.0.commitment()
    }
}

/// Offline-constructible candidate control-plane transition.
///
/// A value of this type is not evidence that the transition was persisted,
/// committed, observed, selected, or current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateScientificViewControlPlaneTransitionV1(
    mechanics::HistoricalScientificViewControlPlaneTransitionV1,
);

impl CandidateScientificViewControlPlaneTransitionV1 {
    pub fn bootstrap_candidate(
        profile: &ScientificViewSemanticProfileV1,
        binding: &ScientificViewDeploymentBindingV1,
        bootstrap: &DeploymentBootstrapEvidenceRefV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::HistoricalScientificViewControlPlaneTransitionV1::bootstrap_activate(
            profile,
            binding,
            &bootstrap.0,
        )
        .map(Self)
    }

    pub fn profile_activation_candidate(
        predecessor: &Self,
        source_profile: &ScientificViewSemanticProfileV1,
        source_binding: &ScientificViewDeploymentBindingV1,
        destination_profile: &ScientificViewSemanticProfileV1,
        destination_binding: &ScientificViewDeploymentBindingV1,
        authorization: &PredecessorAuthorizationEvidenceRefV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::HistoricalScientificViewControlPlaneTransitionV1::activate_profile(
            &predecessor.0,
            source_profile,
            source_binding,
            destination_profile,
            destination_binding,
            &authorization.0,
        )
        .map(Self)
    }

    pub fn source_migration_candidate(
        predecessor: &Self,
        profile: &ScientificViewSemanticProfileV1,
        old_binding: &ScientificViewDeploymentBindingV1,
        new_binding: &ScientificViewDeploymentBindingV1,
        migration: &CandidateSourceMigrationV1,
        authorization: &PredecessorAuthorizationEvidenceRefV1,
    ) -> Result<Self, ScientificViewControlPlaneError> {
        mechanics::HistoricalScientificViewControlPlaneTransitionV1::migrate_source(
            &predecessor.0,
            profile,
            old_binding,
            new_binding,
            &migration.0,
            &authorization.0,
        )
        .map(Self)
    }

    pub fn deployment_id(&self) -> &str {
        self.0.deployment_id()
    }

    pub fn view_namespace(&self) -> &str {
        self.0.view_namespace()
    }

    pub const fn sequence(&self) -> u64 {
        self.0.sequence()
    }

    pub const fn predecessor_transition(&self) -> Option<Commitment32> {
        self.0.predecessor_transition()
    }

    pub const fn source_profile_commitment(&self) -> Option<Commitment32> {
        self.0.source_profile_commitment()
    }

    pub const fn source_binding_commitment(&self) -> Option<Commitment32> {
        self.0.source_binding_commitment()
    }

    pub const fn destination_profile_commitment(&self) -> Commitment32 {
        self.0.destination_profile_commitment()
    }

    pub const fn destination_binding_commitment(&self) -> Commitment32 {
        self.0.destination_binding_commitment()
    }

    pub const fn transition_kind(&self) -> ControlPlaneTransitionKindV1 {
        self.0.transition_kind()
    }

    pub const fn authorization_evidence_commitment(&self) -> Commitment32 {
        self.0.authorization_evidence_commitment()
    }

    pub const fn migration_evidence_commitment(&self) -> Option<Commitment32> {
        self.0.migration_evidence_commitment()
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.0.commitment()
    }
}

/// Conclusion available from only the caller-supplied candidate set.
///
/// `UniqueInSuppliedSet` does not prove occurrence, completeness, or currentness.
/// `MultipleCandidatesInSuppliedSet` is candidate multiplicity, not a committed fork.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuppliedCandidateSuccessorDispositionV1 {
    NoSuppliedSuccessor,
    UniqueInSuppliedSet {
        transition: Commitment32,
    },
    MultipleCandidatesInSuppliedSet {
        candidates: Vec<Commitment32>,
    },
}

pub fn classify_supplied_candidate_successors(
    predecessor: &CandidateScientificViewControlPlaneTransitionV1,
    candidates: &[CandidateScientificViewControlPlaneTransitionV1],
) -> Result<SuppliedCandidateSuccessorDispositionV1, ScientificViewControlPlaneError> {
    let inner_candidates = candidates
        .iter()
        .map(|candidate| candidate.0.clone())
        .collect::<Vec<_>>();

    mechanics::classify_supplied_successors(&predecessor.0, &inner_candidates).map(
        |disposition| match disposition {
            mechanics::SuppliedControlPlaneSuccessorDispositionV1::NoSuppliedSuccessor => {
                SuppliedCandidateSuccessorDispositionV1::NoSuppliedSuccessor
            }
            mechanics::SuppliedControlPlaneSuccessorDispositionV1::UniqueInSuppliedSet {
                transition,
            } => SuppliedCandidateSuccessorDispositionV1::UniqueInSuppliedSet { transition },
            mechanics::SuppliedControlPlaneSuccessorDispositionV1::ContestedInSuppliedSet {
                candidates,
            } => SuppliedCandidateSuccessorDispositionV1::MultipleCandidatesInSuppliedSet {
                candidates,
            },
        },
    )
}
