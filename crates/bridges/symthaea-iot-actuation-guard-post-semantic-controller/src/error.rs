// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PostSemanticControllerError {
    #[error("unsupported post-semantic controller challenge schema")]
    UnsupportedChallengeSchema,
    #[error("post-semantic controller challenge nonce is zero")]
    ZeroChallengeNonce,
    #[error("post-semantic controller challenge contains a zero security commitment")]
    ZeroSecurityCommitment,
    #[error("post-semantic controller challenge targets an invalid device")]
    InvalidDeviceIdentity,
    #[error("post-semantic controller challenge violates causal ordering")]
    InvalidChallengeOrdering,
    #[error("post-semantic controller challenge validity window is invalid or already closed")]
    InvalidChallengeWindow,
    #[error("system wall clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("time conversion overflow")]
    TimeOverflow,
    #[error("OS entropy is unavailable")]
    EntropyUnavailable,
    #[error("unsupported post-semantic controller response schema")]
    UnsupportedResponseSchema,
    #[error("post-semantic controller response size is outside accepted bounds")]
    ResponseSizeOutOfBounds,
    #[error("post-semantic controller report size is outside accepted bounds")]
    ReportSizeOutOfBounds,
    #[error("post-semantic controller evidence size is outside accepted bounds")]
    EvidenceSizeOutOfBounds,
    #[error("post-semantic controller response is not canonically encoded")]
    NonCanonicalResponseEncoding,
    #[error("post-semantic controller report encoding is invalid")]
    InvalidReportEncoding,
    #[error("post-semantic controller report is not canonically encoded")]
    NonCanonicalReportEncoding,
    #[error("controller evidence does not match its report commitment")]
    EvidenceDigestMismatch,
    #[error("controller report does not bind the exact post-semantic challenge")]
    ChallengeBindingMismatch,
    #[error("controller report does not bind the exact authenticated device appraisal")]
    DeviceRealityBindingMismatch,
    #[error("controller report targets another device")]
    DeviceMismatch,
    #[error("controller report binds another physical envelope")]
    EnvelopeMismatch,
    #[error("controller report binds another durable semantic head")]
    SemanticHeadMismatch,
    #[error("controller report binds another Xenia transport-trust generation")]
    TransportTrustMismatch,
    #[error("controller observation predates durable semantic persistence or challenge issuance")]
    ControllerObservationPredatesChallenge,
    #[error("controller report outlives the post-semantic challenge")]
    ControllerReportOutlivesChallenge,
    #[error("controller challenge/response encoding failed: {0}")]
    Encoding(#[source] bincode::Error),
    #[error("controller challenge/response decoding failed: {0}")]
    Decoding(#[source] bincode::Error),
}
