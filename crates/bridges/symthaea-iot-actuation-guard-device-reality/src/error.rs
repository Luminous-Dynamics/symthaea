// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

/// Fail-closed errors for privileged post-reservation device-reality verification.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeviceRealityError {
    #[error("unsupported device-reality policy schema")]
    UnsupportedPolicySchema,
    #[error("device-reality policy targets an invalid device")]
    InvalidPolicyDevice,
    #[error("device-reality policy verifier surface is invalid")]
    InvalidPolicyVerifierSurface,
    #[error("device-reality policy reference-value surface is invalid")]
    InvalidPolicyReferenceValues,
    #[error("device-reality policy contains a zero appraisal-policy commitment")]
    ZeroAppraisalPolicyDigest,
    #[error("device-reality result lifetime policy is invalid")]
    InvalidPolicyResultLifetime,
    #[error("guard device-reality policy does not match independently anchored digest")]
    AnchoredPolicyDigestMismatch,

    #[error("unsupported device-reality trust schema")]
    UnsupportedTrustSchema,
    #[error("device-reality verifier key identity is invalid")]
    InvalidVerifierKeyIdentity,
    #[error("device-reality verifier algorithm is not the fixed Ed25519 profile")]
    UnsupportedVerifierAlgorithm,
    #[error("device-reality verifier public key is invalid")]
    InvalidVerifierPublicKey,
    #[error("device-reality verifier key validity window is invalid")]
    InvalidVerifierKeyWindow,
    #[error("device-reality verifier maximum result lifetime is invalid")]
    InvalidVerifierKeyResultLifetime,
    #[error("device-reality trust sequence is zero")]
    TrustSequenceZero,
    #[error("device-reality trust snapshot validity window is invalid")]
    InvalidTrustWindow,
    #[error("device-reality trust snapshot key count is invalid")]
    InvalidTrustKeyCount,
    #[error("device-reality trust genesis unexpectedly has a predecessor")]
    GenesisHasPredecessor,
    #[error("device-reality trust successor is missing a predecessor")]
    SuccessorMissingPredecessor,
    #[error("duplicate device-reality verifier key identity")]
    DuplicateVerifierKeyIdentity,
    #[error("snapshot is not device-reality trust genesis")]
    NotGenesis,
    #[error("device-reality trust sequence overflow")]
    TrustSequenceOverflow,
    #[error("device-reality trust sequence is not next: expected {expected}, proposed {proposed}")]
    TrustSequenceNotNext { expected: u64, proposed: u64 },
    #[error("device-reality trust predecessor mismatch")]
    TrustPredecessorMismatch,
    #[error("device-reality trust issue time regressed")]
    TrustIssuedAtRegressed,
    #[error("persisted device-reality trust snapshot does not match trusted head")]
    TrustedHeadMismatch,
    #[error("guard device-reality trust registry does not match independently anchored head")]
    AnchoredTrustHeadMismatch,
    #[error("existing device-reality verifier key identity was deleted")]
    TrustedKeyDeleted,
    #[error("device-reality verifier key immutable identity/material changed")]
    TrustedKeyMutated,
    #[error("device-reality verifier key lifecycle attempted reactivation")]
    TrustedKeyReactivated,
    #[error("device-reality verifier key expiry was extended")]
    TrustedKeyExpiryExtended,
    #[error("device-reality verifier result-lifetime ceiling was extended")]
    TrustedKeyResultLifetimeExtended,
    #[error("device-reality trust snapshot is not fresh")]
    TrustSnapshotNotFresh,
    #[error("no exact active trusted device-reality verifier key exists")]
    NoActiveVerifierKey,
    #[error("device-reality verifier key is not active at required times")]
    VerifierKeyNotActive,

    #[error("semantic-reservation challenge is invalid")]
    InvalidReservationChallenge,
    #[error("semantic-reservation challenge is not fresh")]
    ReservationChallengeNotFresh,
    #[error("admission-reservation challenge is invalid")]
    InvalidAdmissionChallenge,
    #[error("admission-reservation challenge is not fresh")]
    AdmissionChallengeNotFresh,
    #[error("device attestation result structure is invalid")]
    InvalidAttestationResult,
    #[error("device attestation signature length is not the fixed Ed25519 length")]
    InvalidAttestationSignatureLength,
    #[error("device attestation algorithm is not the fixed Ed25519 profile")]
    AttestationAlgorithmMismatch,
    #[error("device attestation result targets another device")]
    AttestationDeviceMismatch,
    #[error("device attestation result does not bind the exact semantic-reservation challenge")]
    AttestationChallengeMismatch,
    #[error("device attestation result does not bind the exact admission-reservation challenge")]
    AttestationAdmissionChallengeMismatch,
    #[error("device attestation verifier is not allowed by guard-owned policy")]
    AttestationVerifierDenied,
    #[error("device attestation reference-values lineage is not allowed by guard-owned policy")]
    AttestationReferenceValuesDenied,
    #[error("device attestation appraisal policy differs from guard-owned policy")]
    AttestationAppraisalPolicyMismatch,
    #[error("device attestation appraisal predates durable semantic persistence")]
    AttestationPredatesSemanticPersistence,
    #[error("device attestation appraisal predates durable admission reservation/challenge issuance")]
    AttestationPredatesAdmissionReservation,
    #[error("device attestation result is stale, future-dated, or outside the challenge window")]
    AttestationNotFreshForReservation,
    #[error("device attestation result lifetime exceeds guard-owned policy")]
    AttestationLifetimeExceedsPolicy,
    #[error("device attestation predates the current verifier-trust generation")]
    AttestationPredatesCurrentTrustGeneration,
    #[error("device attestation signature is invalid under the exact current trusted key")]
    InvalidAttestationSignature,
    #[error("device attestation result commitment could not be constructed")]
    AttestationObjectCommitmentFailed,
    #[error("system wall clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system wall-clock milliseconds do not fit the protocol time domain")]
    SystemClockOverflow,
    #[error("seconds-to-milliseconds conversion overflowed")]
    TimeConversionOverflow,
}
