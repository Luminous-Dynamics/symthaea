// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdmissionChallengeError {
    #[error("unsupported admission-reality challenge schema")]
    UnsupportedChallengeSchema,
    #[error("admission-reality challenge nonce is zero")]
    ZeroChallengeNonce,
    #[error("operating-system entropy is unavailable for admission challenge")]
    EntropyUnavailable,
    #[error("system clock is before the Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("admission-reality challenge contains a zero security commitment")]
    ZeroSecurityCommitment,
    #[error("admission-reality challenge device identity is invalid")]
    InvalidDeviceIdentity,
    #[error("admission-reality challenge violates durable causal ordering/deadline")]
    InvalidChallengeOrdering,
    #[error("admission-reality challenge validity window is invalid")]
    InvalidChallengeWindow,
    #[error("admission-reality challenge time conversion overflow")]
    TimeOverflow,
    #[error("unsupported admission device-reality response schema")]
    UnsupportedResponseSchema,
    #[error("admission device attestation size is outside accepted bounds")]
    AttestationSizeOutOfBounds,
    #[error("admission device-reality response size is outside accepted bounds")]
    ResponseSizeOutOfBounds,
    #[error("admission device-reality response is not canonically encoded")]
    NonCanonicalResponseEncoding,
    #[error("device-attestation result encoding is invalid")]
    InvalidAttestationEncoding,
    #[error("device-attestation result is not canonically encoded")]
    NonCanonicalAttestationEncoding,
    #[error("device-attestation result structure is invalid")]
    InvalidAttestationStructure,
    #[error("device-attestation signature size is invalid")]
    InvalidAttestationSignatureSize,
    #[error("device-attestation result targets another device")]
    AttestationDeviceMismatch,
    #[error("device-attestation result does not bind the exact admission reservation challenge")]
    AttestationChallengeMismatch,
    #[error("admission challenge/response encoding failed: {0}")]
    Encoding(#[source] bincode::Error),
    #[error("admission challenge/response decoding failed: {0}")]
    Decoding(#[source] bincode::Error),
}
