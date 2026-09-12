// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistence boundary for the write-ahead episodic-quarantine intent ledger.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;

use crate::quarantine_intent_ledger::QuarantineIntentEnvelope;

/// Durable persistence boundary for the complete quarantine-intent chain.
///
/// Implementations should atomically replace/append their durable representation and retain the
/// returned reference as audit evidence. Rollback resistance additionally requires the caller to
/// retain the supplied head hash in an external monotonic/trusted anchor.
pub trait QuarantineIntentLedgerPersistence {
    type Error: StdError + Send + Sync + 'static;

    fn persist_quarantine_intent_ledger(
        &mut self,
        events: &[QuarantineIntentEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error>;
}
