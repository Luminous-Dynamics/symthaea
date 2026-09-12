// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable persistence boundary for the episodic quarantine state ledger.
//!
//! Implementations must persist the complete event stream and exact head hash before returning.
//! The returned reference identifies that durable commit. Rollback resistance still requires an
//! external trusted copy of the committed head hash; persistence and anchoring are distinct roles.

use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;

use crate::quarantine_state_ledger::QuarantineLedgerEnvelope;

pub trait QuarantineLedgerPersistence {
    type Error: StdError + Send + Sync + 'static;

    fn persist_quarantine_ledger(
        &mut self,
        events: &[QuarantineLedgerEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error>;
}
