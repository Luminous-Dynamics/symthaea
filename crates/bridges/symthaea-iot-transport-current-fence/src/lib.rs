// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrowed last-moment currentness fence for an already fixed-cryptography Xenia proof.
//!
//! `RevalidatedXeniaTransport` already proves that the exact retained receipt/payload passed the
//! fixed Ed25519 + ML-DSA-65 verifier under one exact transport-trust generation and records the
//! exact selected key-record digest plus the earliest natural receipt/key/snapshot expiry.
//!
//! This crate deliberately does not parse receipts or reimplement transport-key lifecycle rules.
//! At the later physical-attempt boundary it only requires that the independently anchored current
//! registry is still the exact same committed generation, the same exact key record is present,
//! time has not regressed, and the previously proven earliest natural deadline has not elapsed.
//! Any lifecycle mutation changes the trust head; any natural expiry is already represented by
//! `valid_until_unix_ms`.
//!
//! Success remains non-authorizing and borrows both the exact current registry holder and the
//! historical revalidated transport proof for the lifetime of one later attempt.

#![deny(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_iot_transport_current_revalidation::RevalidatedXeniaTransport;
use symthaea_iot_transport_receipt::{
    TransportReceiptError, TransportTrustHead, TransportTrustRegistry,
};
use thiserror::Error;

/// Guard-owned independently anchored current Xenia transport trust for one final fence.
#[derive(Debug)]
pub struct CurrentXeniaTransportFenceGuard {
    registry: TransportTrustRegistry,
    anchored_current_head: TransportTrustHead,
}

impl CurrentXeniaTransportFenceGuard {
    pub fn new(
        registry: TransportTrustRegistry,
        anchored_current_head: TransportTrustHead,
    ) -> Result<Self, CurrentXeniaTransportFenceError> {
        if registry.head() != anchored_current_head {
            return Err(CurrentXeniaTransportFenceError::RegistryHeadNotAnchored);
        }
        Ok(Self {
            registry,
            anchored_current_head,
        })
    }

    pub const fn anchored_current_head(&self) -> TransportTrustHead {
        self.anchored_current_head
    }

    /// Fence one exact revalidated transport proof against guard-owned current trust.
    ///
    /// The caller cannot supply current time, a key, a registry, or a trust head.
    pub fn fence_current<'a>(
        &'a self,
        proof: &'a RevalidatedXeniaTransport,
    ) -> Result<CurrentXeniaTransportFence<'a>, CurrentXeniaTransportFenceError> {
        self.fence_current_at(proof, system_unix_ms()?)
    }

    fn fence_current_at<'a>(
        &'a self,
        proof: &'a RevalidatedXeniaTransport,
        now_unix_ms: u64,
    ) -> Result<CurrentXeniaTransportFence<'a>, CurrentXeniaTransportFenceError> {
        if self.registry.head() != self.anchored_current_head {
            return Err(CurrentXeniaTransportFenceError::RegistryHeadNotAnchored);
        }
        if proof.transport_trust_head() != self.anchored_current_head {
            return Err(CurrentXeniaTransportFenceError::ProofTrustGenerationNotCurrent);
        }
        if now_unix_ms < proof.revalidated_at_unix_ms() {
            return Err(CurrentXeniaTransportFenceError::ClockRegressedSinceRevalidation);
        }

        // Head equality commits the complete snapshot. We exact-lookup only the key identity that
        // #476's fixed verifier already selected and require its complete record digest to match.
        // This is not a second lifecycle-selection algorithm.
        let snapshot = self.registry.snapshot();
        let current_key = snapshot
            .keys
            .iter()
            .find(|key| key.attestor_id == proof.attestor_id() && key.key_id == proof.key_id())
            .ok_or(CurrentXeniaTransportFenceError::ExactKeyRecordMissing)?;
        if current_key.digest()? != proof.transport_key_digest() {
            return Err(CurrentXeniaTransportFenceError::ExactKeyRecordMismatch);
        }
        if current_key.not_after_unix_ms != proof.transport_key_not_after_unix_ms()
            || snapshot.expires_at_unix_ms != proof.trust_snapshot_expires_at_unix_ms()
        {
            return Err(CurrentXeniaTransportFenceError::CapturedDeadlineMismatch);
        }

        let expected_valid_until = proof
            .receipt_expires_at_unix_ms()
            .min(current_key.not_after_unix_ms)
            .min(snapshot.expires_at_unix_ms);
        if expected_valid_until != proof.valid_until_unix_ms() {
            return Err(CurrentXeniaTransportFenceError::CapturedDeadlineMismatch);
        }
        if now_unix_ms >= expected_valid_until {
            return Err(CurrentXeniaTransportFenceError::CurrentTransportWindowElapsed);
        }

        Ok(CurrentXeniaTransportFence {
            _guard: self,
            proof,
            fenced_at_unix_ms: now_unix_ms,
            valid_until_unix_ms: expected_valid_until,
        })
    }
}

/// Borrowed proof that one fixed-cryptography Xenia transport proof is still current now.
#[derive(Debug)]
pub struct CurrentXeniaTransportFence<'a> {
    _guard: &'a CurrentXeniaTransportFenceGuard,
    proof: &'a RevalidatedXeniaTransport,
    fenced_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl<'a> CurrentXeniaTransportFence<'a> {
    pub const fn proof(&self) -> &'a RevalidatedXeniaTransport {
        self.proof
    }

    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.fenced_at_unix_ms
    }

    /// Earliest exclusive receipt/key/snapshot deadline proven by #476 and re-bound here.
    pub const fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
}

fn system_unix_ms() -> Result<u64, CurrentXeniaTransportFenceError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| CurrentXeniaTransportFenceError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| CurrentXeniaTransportFenceError::ClockOverflow)
}

#[derive(Debug, Error)]
pub enum CurrentXeniaTransportFenceError {
    #[error("current transport registry does not match independently retained head")]
    RegistryHeadNotAnchored,
    #[error("revalidated transport proof belongs to another trust generation")]
    ProofTrustGenerationNotCurrent,
    #[error("system wall clock regressed behind fixed current transport revalidation")]
    ClockRegressedSinceRevalidation,
    #[error("exact transport key record captured by current revalidation is missing")]
    ExactKeyRecordMissing,
    #[error("exact transport key record differs from the fixed-current proof")]
    ExactKeyRecordMismatch,
    #[error("captured transport natural-expiry metadata disagrees with the exact current snapshot")]
    CapturedDeadlineMismatch,
    #[error("the fixed receipt/key/trust currentness window has elapsed")]
    CurrentTransportWindowElapsed,
    #[error("system clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system wall-clock milliseconds overflow")]
    ClockOverflow,
    #[error("current transport key record is invalid: {0}")]
    Transport(#[from] TransportReceiptError),
}
