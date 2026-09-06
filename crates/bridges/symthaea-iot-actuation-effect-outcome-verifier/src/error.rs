// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EffectOutcomeError {
    #[error("unsupported physical-effect outcome evidence schema")]
    UnsupportedEvidenceSchema,
    #[error("physical-effect outcome evidence contains an invalid identity")]
    InvalidEvidenceIdentity,
    #[error("physical-effect outcome evidence contains a zero security commitment")]
    ZeroEvidenceCommitment,
    #[error("physical-effect outcome evidence uses an unsupported signature algorithm")]
    UnsupportedEvidenceAlgorithm,
    #[error("physical-effect outcome evidence time window is malformed")]
    InvalidEvidenceWindow,
    #[error("physical-effect outcome claim structure is malformed")]
    InvalidClaimStructure,
    #[error("physical-effect outcome canonical encoding length overflow")]
    EncodingLengthOverflow,

    #[error("unsupported physical-effect outcome policy schema")]
    UnsupportedPolicySchema,
    #[error("physical-effect outcome policy has an invalid device or operation")]
    InvalidPolicyTarget,
    #[error("physical-effect outcome policy verifier surface is invalid")]
    InvalidPolicyVerifierSurface,
    #[error("physical-effect outcome policy claim surface is invalid")]
    InvalidPolicyClaimSurface,
    #[error("physical-effect outcome policy has a zero outcome-profile digest")]
    ZeroOutcomeProfileDigest,
    #[error("physical-effect outcome policy has invalid accepted reference values")]
    InvalidPolicyReferenceValues,
    #[error("physical-effect outcome policy has a zero appraisal-policy digest")]
    ZeroAppraisalPolicyDigest,
    #[error("physical-effect outcome policy evidence lifetime is invalid")]
    InvalidPolicyEvidenceLifetime,

    #[error("unsupported physical-effect outcome trust schema")]
    UnsupportedTrustSchema,
    #[error("physical-effect outcome trust sequence is zero")]
    TrustSequenceZero,
    #[error("physical-effect outcome trust window is invalid")]
    InvalidTrustWindow,
    #[error("physical-effect outcome trust key count is invalid")]
    InvalidTrustKeyCount,
    #[error("physical-effect outcome trust genesis has a predecessor")]
    GenesisHasPredecessor,
    #[error("physical-effect outcome trust successor is missing its predecessor")]
    SuccessorMissingPredecessor,
    #[error("physical-effect outcome verifier key identity is invalid")]
    InvalidVerifierKeyIdentity,
    #[error("physical-effect outcome verifier algorithm is unsupported")]
    UnsupportedVerifierAlgorithm,
    #[error("physical-effect outcome verifier public key is invalid")]
    InvalidVerifierPublicKey,
    #[error("physical-effect outcome verifier key window is invalid")]
    InvalidVerifierKeyWindow,
    #[error("physical-effect outcome verifier key evidence lifetime is invalid")]
    InvalidVerifierKeyEvidenceLifetime,
    #[error("duplicate physical-effect outcome verifier key identity")]
    DuplicateVerifierKeyIdentity,
    #[error("physical-effect outcome trust snapshot is not genesis")]
    NotGenesis,
    #[error("physical-effect outcome trust sequence overflow")]
    TrustSequenceOverflow,
    #[error("physical-effect outcome trust sequence is not the exact successor: expected {expected}, proposed {proposed}")]
    TrustSequenceNotNext { expected: u64, proposed: u64 },
    #[error("physical-effect outcome trust predecessor does not match")]
    TrustPredecessorMismatch,
    #[error("physical-effect outcome trust issued-at time regressed")]
    TrustIssuedAtRegressed,
    #[error("trusted physical-effect outcome verifier key was deleted")]
    TrustedKeyDeleted,
    #[error("trusted physical-effect outcome verifier key immutable identity was mutated")]
    TrustedKeyMutated,
    #[error("trusted physical-effect outcome verifier key was reactivated")]
    TrustedKeyReactivated,
    #[error("trusted physical-effect outcome verifier key expiry was extended")]
    TrustedKeyExpiryExtended,
    #[error("trusted physical-effect outcome verifier key evidence lifetime was extended")]
    TrustedKeyEvidenceLifetimeExtended,
    #[error("physical-effect outcome trust snapshot does not match retained head")]
    TrustedHeadMismatch,
    #[error("physical-effect outcome trust snapshot is not fresh")]
    TrustSnapshotNotFresh,
    #[error("no exact active physical-effect outcome verifier key exists")]
    NoActiveVerifierKey,
    #[error("physical-effect outcome verifier key is not active for the required times")]
    VerifierKeyNotActive,

    #[error("physical-effect outcome policy digest does not match its independent anchor")]
    AnchoredPolicyDigestMismatch,
    #[error("physical-effect outcome trust head does not match its independent anchor")]
    AnchoredTrustHeadMismatch,
    #[error("reconciliation challenge is invalid")]
    InvalidReconciliationChallenge,
    #[error("reconciliation challenge is not fresh")]
    ReconciliationChallengeNotFresh,
    #[error("outcome evidence targets another device")]
    EvidenceDeviceMismatch,
    #[error("outcome evidence targets another operation")]
    EvidenceOperationMismatch,
    #[error("outcome evidence targets another executor")]
    EvidenceExecutorMismatch,
    #[error("outcome evidence is bound to another reconciliation challenge")]
    EvidenceChallengeMismatch,
    #[error("outcome evidence is bound to another command")]
    EvidenceCommandMismatch,
    #[error("outcome evidence command sequence differs from the challenge")]
    EvidenceSequenceMismatch,
    #[error("outcome evidence profile is not the exact guard-owned device-class profile")]
    EvidenceOutcomeProfileMismatch,
    #[error("outcome evidence reference values are not accepted")]
    EvidenceReferenceValuesDenied,
    #[error("outcome evidence appraisal policy differs from the exact guard-owned policy")]
    EvidenceAppraisalPolicyMismatch,
    #[error("outcome evidence verifier is denied by policy")]
    EvidenceVerifierDenied,
    #[error("outcome evidence claim kind is denied by policy")]
    EvidenceClaimKindDenied,
    #[error("outcome evidence predates the fresh reconciliation challenge")]
    EvidencePredatesChallenge,
    #[error("outcome evidence predates the current outcome-verifier trust generation")]
    EvidencePredatesCurrentTrustGeneration,
    #[error("outcome evidence outlives the fresh reconciliation challenge")]
    EvidenceOutlivesChallenge,
    #[error("outcome evidence is not currently fresh")]
    EvidenceNotFresh,
    #[error("outcome evidence lifetime exceeds guard or key policy")]
    EvidenceLifetimeExceedsPolicy,
    #[error("recorded effect execution is outside the original actuation window")]
    ExecutionRecordOutsideActuationWindow,
    #[error("postcondition observation is not fresh for the reconciliation challenge")]
    PostconditionObservationNotFresh,
    #[error("postcondition observation time is causally inconsistent")]
    PostconditionObservationTimeInvalid,
    #[error("non-execution proof does not cover the complete original actuation window")]
    NonExecutionCoverageIncomplete,
    #[error("non-execution proof claims coverage beyond evidence issuance")]
    NonExecutionCoverageAfterEvidence,
    #[error("physical-effect outcome signature is invalid")]
    InvalidEvidenceSignature,

    #[error("verified outcome proof policy no longer matches current guard policy")]
    CurrentProofPolicyMismatch,
    #[error("verified outcome proof trust head no longer matches current guard trust")]
    CurrentProofTrustHeadMismatch,
    #[error("verified outcome proof commitment changed")]
    CurrentProofCommitmentMismatch,
    #[error("verified outcome proof verifier/key identity changed")]
    CurrentProofVerifierKeyMismatch,
    #[error("verified outcome proof clock regressed")]
    CurrentProofClockRegressed,
    #[error("verified outcome proof natural validity window elapsed")]
    CurrentProofWindowElapsed,

    #[error("system wall clock is before the Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system wall clock does not fit in u64 milliseconds")]
    SystemClockOverflow,
}
