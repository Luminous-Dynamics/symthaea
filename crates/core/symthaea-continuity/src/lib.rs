// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computing-continuity kernel.
//!
//! This crate deliberately separates observed facts from dependency claims,
//! continuity requirements, verification evidence, and execution authority.
//! None of these values grants migration authority by itself.

#![deny(unsafe_code)]

pub mod active_lkg;
pub mod auth_wire;
mod compose;
pub mod commit_currentness;
pub mod commit_eligibility;
pub mod contract;
pub mod crash_reconciliation;
pub mod distributed;
pub mod distributed_currentness;
pub mod distributed_evidence;
mod distributed_evidence_error_bridge;
mod distributed_health_common;
pub mod distributed_qualification;
pub mod distributed_state;
pub mod exact_distributed_health;
pub mod exact_local_health;
pub mod exact_policy;
pub mod execution_capability;
pub mod execution_coordinator;
pub mod execution_journal;
pub mod execution_journal_anchor;
pub mod execution_result;
pub mod failure_domain;
pub mod known_good;
pub mod observation;
pub mod post_execution_health;
pub mod post_execution_observation;
pub mod post_transition_distributed_health;
pub mod profile_adoption;
pub mod promotion_eligibility;
pub mod recovery_qualification;
pub mod scope;
pub mod subject_contract;
pub mod subject_witness;
pub mod transition_authority;
pub mod transition_lineage;
pub mod trusted_commit_epoch;
pub mod verifier;
mod witness;

pub use active_lkg::{
    ACTIVE_KNOWN_GOOD_SELECTION_RECORD_SCHEMA_V1, ActiveKnownGoodSelectionError,
    ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionRecordV1,
    ActiveKnownGoodSelectionV1,
};
pub use auth_wire::{
    CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA, CONTINUITY_VERIFICATION_CLAIM_HASH_ALGORITHM,
    CONTINUITY_VERIFICATION_XENIA_PURPOSE, canonical_verification_claim_bytes,
    canonical_verification_claim_digest,
};
pub use commit_currentness::{
    LOCAL_COMMIT_CURRENTNESS_POLICY_SCHEMA_V1, LOCAL_COMMIT_STATE_CLAIM_SCHEMA_V1,
    AuthenticatedLocalCommitStateEvidenceId, LocalCommitCurrentnessError,
    LocalCommitCurrentnessPolicyId, LocalCommitCurrentnessPolicyV1,
    LocalCommitObservationOutcomeV1, LocalCommitStateClaimId, LocalCommitStateClaimV1,
    QualifiedLocalCommitCurrentnessId, QualifiedLocalCommitCurrentnessV1,
    ValidatedLocalCommitCurrentnessPolicyV1,
};
pub use commit_eligibility::{
    CommitEligibilityError, CommitEligibleTransitionId, CommitEligibleTransitionV1,
};
pub use contract::{
    ApprovalBasis, ContinuityContractId, ContinuityContractV1, ContinuityRequirementId,
    ContinuityRequirementV1, ContractError, EquivalencePredicate, RequirementCriticality,
    ValidatedContinuityContractV1,
};
pub use crash_reconciliation::{
    CRASH_RECONCILIATION_RECORD_SCHEMA_V1, CrashReconciliationClassificationV1,
    CrashReconciliationError, CrashReconciliationId, CrashReconciliationNextProofV1,
    CrashReconciliationRecordV1, QualifiedCrashReconciliationV1,
};
pub use distributed::{
    DISTRIBUTED_CHANGE_BUDGET_SCHEMA_V1, DistributedChangeBudgetError, DistributedChangeBudgetId,
    DistributedChangeBudgetV1, MutualExclusionSetV1, RecoveryPathClassV1,
    ValidatedDistributedChangeBudgetV1,
};
pub use distributed_currentness::{
    DISTRIBUTED_CURRENTNESS_POLICY_SCHEMA_V1, DistributedCurrentnessPolicyError,
    DistributedCurrentnessPolicyId, DistributedCurrentnessPolicyV1,
    ValidatedDistributedCurrentnessPolicyV1,
};
pub use distributed_evidence::{
    FAILURE_DOMAIN_STATE_CLAIM_SCHEMA_V1, RECOVERY_PATH_STATE_CLAIM_SCHEMA_V1,
    AuthenticatedFailureDomainStateEvidenceId, AuthenticatedRecoveryPathStateEvidenceId,
    DistributedEvidenceError, FailureDomainObservationOutcomeV1, FailureDomainStateClaimId,
    FailureDomainStateClaimV1, RecoveryPathObservationOutcomeV1, RecoveryPathStateClaimId,
    RecoveryPathStateClaimV1,
};
pub use distributed_qualification::{
    DistributedCurrentStateDigest, DistributedQualificationError, DistributedVerifierSnapshotV1,
    QualifiedDistributedTransitionWitnessId, QualifiedDistributedTransitionWitnessV1,
};
pub use distributed_state::{
    DISTRIBUTED_STATE_CONTEXT_SCHEMA_V1, PARTICIPANT_STATE_CLAIM_SCHEMA_V1,
    AuthenticatedParticipantStateEvidenceId, DistributedStateContextId,
    DistributedStateContextV1, DistributedStateError, ParticipantOperationalStateV1,
    ParticipantSetDigest, ParticipantStateClaimId, ParticipantStateClaimV1,
    ValidatedDistributedStateContextV1,
};
pub use exact_distributed_health::{
    ExactDistributedHealthError, ExactDistributedRecoveryPathV2, ExactDistributedStateDigestV2,
    ExactDistributedVerifierSnapshotV2, QualifiedExactDistributedHealthIdV2,
    QualifiedExactDistributedHealthV2,
};
pub use exact_local_health::{
    CRASH_SOURCE_HEALTH_CLAIM_SCHEMA_V1, AuthenticatedCrashSourceHealthId,
    CrashSourceHealthClaimId, CrashSourceHealthClaimV1, CrashSourceHealthPolicyId,
    CrashSourceHealthPolicyV1, ExactLocalHealthError, HealthyLocalSnapshotBasisV1,
    QualifiedCrashSourceHealthId, QualifiedCrashSourceHealthV1,
    QualifiedHealthyLocalSnapshotId, QualifiedHealthyLocalSnapshotV1,
};
pub use exact_policy::{
    ExactVerificationPolicyError, ExactVerificationPolicyId, ExactVerificationPolicyV1,
};
pub use execution_capability::{
    EXECUTION_ATTEMPT_INTENT_SCHEMA_V1, EXECUTION_ATTEMPT_RECEIPT_SCHEMA_V1,
    EXECUTION_BACKEND_PROFILE_SCHEMA_V1, ExecutionAttemptId, ExecutionAttemptIntentV1,
    ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptId, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError,
    ExecutionEpochAnchorModeV1, ExecutionSessionId, ExecutionSessionV1,
    OneUseExecutionCapabilityId, OneUseExecutionCapabilityV1, PreparedExecutionAttemptV1,
};
pub use execution_coordinator::{
    KnownGoodExecutionCoordinatorError, PendingAnchoredKnownGoodExecutionV1,
    ReadyKnownGoodExecutionAttemptV1, prepare_known_good_execution,
};
pub use execution_journal::{
    ExecutionJournalDigest, ExecutionJournalError, JournalAttemptDispositionV1,
    JournalAttemptEntryV1, ReconstructedExecutionJournalV1,
};
pub use execution_journal_anchor::{
    EXECUTION_JOURNAL_ANCHOR_AUTH_PURPOSE, EXECUTION_JOURNAL_ANCHOR_CLAIM_SCHEMA_V1,
    EXECUTION_JOURNAL_ANCHOR_PROFILE_SCHEMA_V1, AuthenticatedExecutionJournalAnchorId,
    ExecutionJournalAnchorClaimId, ExecutionJournalAnchorClaimV1, ExecutionJournalAnchorError,
    ExecutionJournalAnchorProfileId, ExecutionJournalAnchorProfileV1,
    QualifiedExecutionJournalAnchorId, QualifiedExecutionJournalAnchorV1,
    canonical_execution_journal_anchor_claim_bytes,
    canonical_execution_journal_anchor_claim_digest,
};
pub use execution_result::{
    CanonicalExecutionAttemptResultId, CanonicalExecutionAttemptResultV1,
    ExecutionResultBindingError,
};
pub use failure_domain::{
    FAILURE_DOMAIN_POLICY_SCHEMA_V1, FailureDomainGroupV1, FailureDomainKindV1,
    FailureDomainPolicyError, FailureDomainPolicyId, FailureDomainPolicyV1,
    ValidatedFailureDomainPolicyV1,
};
pub use known_good::{
    KNOWN_GOOD_CHECKPOINT_RECORD_SCHEMA_V1, KnownGoodCheckpointError,
    KnownGoodCheckpointId, KnownGoodCheckpointRecordV1, KnownGoodRecoveryPathV1,
    QualifiedKnownGoodCheckpointV1,
};
pub use observation::{
    DependencyBasis, DependencyClaimId, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
    ObservationEnvelopeV1, ObservationError, ObservationId,
};
pub use post_execution_health::{
    POST_EXECUTION_HEALTH_CLAIM_SCHEMA_V1, AuthenticatedPostExecutionHealthId,
    PostExecutionHealthClaimId, PostExecutionHealthClaimV1, PostExecutionHealthError,
    PostExecutionHealthOutcomeV1, PostExecutionHealthPolicyId, PostExecutionHealthPolicyV1,
    QualifiedPostExecutionHealthId, QualifiedPostExecutionHealthV1,
};
pub use post_execution_observation::{
    POST_EXECUTION_OBSERVATION_CLAIM_SCHEMA_V1, AuthenticatedPostExecutionObservationId,
    PostExecutionObservationClaimId, PostExecutionObservationClaimV1,
    PostExecutionObservationError, PostExecutionObservationPolicyId,
    PostExecutionObservationPolicyV1, PostExecutionObservedStateV1,
    QualifiedPostExecutionObservationId, QualifiedPostExecutionObservationV1,
};
pub use post_transition_distributed_health::{
    PostTransitionDistributedHealthError, PostTransitionDistributedStateDigest,
    PostTransitionVerifierSnapshotV1, QualifiedPostTransitionDistributedHealthId,
    QualifiedPostTransitionDistributedHealthV1, QualifiedRecoveryPathSnapshotV1,
};
pub use profile_adoption::{
    VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1, VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1,
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionSubjectId, VerifierProfileAdoptionSubjectV1,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
pub use promotion_eligibility::{
    LkgPromotionEligibilityId, LkgPromotionEligibilityV1, LkgPromotionError,
};
pub use recovery_qualification::{
    QualifiedRecoveryToActiveKnownGoodId, QualifiedRecoveryToActiveKnownGoodV1,
    RecoveryQualificationError,
};
pub use scope::{
    CONTINUITY_SUBJECT_SCHEMA_V1, ContinuityScopeV1, ContinuitySubjectError,
    ContinuitySubjectId, ContinuitySubjectV1,
};
pub use subject_contract::{
    SubjectBoundContinuityContractId, SubjectBoundContinuityContractV1,
    SubjectContractBindingError,
};
pub use subject_witness::{
    SubjectBoundQualifiedContinuityWitnessId, SubjectBoundQualifiedContinuityWitnessV1,
    SubjectWitnessBindingError,
};
pub use transition_authority::{
    TRANSITION_AUTHORITY_CLAIM_SCHEMA_V1, TRANSITION_AUTHORITY_POLICY_SCHEMA_V1,
    TRANSITION_AUTHORITY_PROFILE_SCHEMA_V1, TRANSITION_AUTHORITY_XENIA_PURPOSE,
    AuthenticatedTransitionAuthorityId, TransitionAuthorityClaimId, TransitionAuthorityClaimV1,
    TransitionAuthorityError, TransitionAuthorityPolicyId, TransitionAuthorityPolicyV1,
    TransitionAuthorityProfileId, TransitionAuthorityProfileV1,
    ValidatedTransitionAuthorityPolicyV1, canonical_transition_authority_claim_bytes,
    canonical_transition_authority_claim_digest,
};
pub use transition_lineage::{
    KNOWN_GOOD_EXECUTION_INTENT_SCHEMA_V1, KNOWN_GOOD_TRANSITION_LINEAGE_SCHEMA_V1,
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodBoundTrustedCommitEligibilityV1,
    KnownGoodExecutionIntentId, KnownGoodTransitionLineageError, KnownGoodTransitionLineageId,
    KnownGoodTransitionLineageV1,
};
pub use trusted_commit_epoch::{
    TRUSTED_COMMIT_CLOCK_PROFILE_SCHEMA_V1, TRUSTED_COMMIT_EPOCH_CLAIM_SCHEMA_V1,
    TRUSTED_COMMIT_EPOCH_POLICY_SCHEMA_V1, TRUSTED_COMMIT_EPOCH_XENIA_PURPOSE,
    AuthenticatedTrustedCommitEpochId, QualifiedTrustedCommitEpochId,
    QualifiedTrustedCommitEpochV1, TrustedCommitClockProfileId, TrustedCommitClockProfileV1,
    TrustedCommitEligibilityId, TrustedCommitEligibilityV1, TrustedCommitEpochClaimId,
    TrustedCommitEpochClaimV1, TrustedCommitEpochError, TrustedCommitEpochPolicyId,
    TrustedCommitEpochPolicyV1, ValidatedTrustedCommitEpochPolicyV1,
    canonical_trusted_commit_epoch_claim_bytes, canonical_trusted_commit_epoch_claim_digest,
    validate_trusted_commit_epoch_progression,
};
pub use verifier::{
    AuthenticatedVerificationEvidenceId, VerificationAdmissionError, VerificationEvidenceClaimId,
    VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileId, VerifierProfileV1,
};
pub use witness::{
    EvidenceClass, ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId,
    VerificationObligationId, VerificationPolicyEntryV1, WitnessError, WitnessId,
    WitnessManifestId,
};