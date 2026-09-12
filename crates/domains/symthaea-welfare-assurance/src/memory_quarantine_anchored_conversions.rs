// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error-conversion glue for the anchored episodic-quarantine executor.

use std::error::Error as StdError;

use crate::memory_identity::EpisodeContentIdError;
use crate::memory_quarantine::EpisodicQuarantineInterventionError;
use crate::memory_quarantine_anchored::{
    AnchoredQuarantineExecutionError, AnchoredQuarantineInterventionError,
};
use crate::quarantine_intent_ledger::QuarantineIntentLedgerError;
use crate::quarantine_state_ledger::QuarantineLedgerError;

impl<E, I, S> From<EpisodeContentIdError> for AnchoredQuarantineExecutionError<E, I, S>
where
    E: StdError + Send + Sync + 'static,
    I: StdError + Send + Sync + 'static,
    S: StdError + Send + Sync + 'static,
{
    fn from(value: EpisodeContentIdError) -> Self {
        Self::Intervention(AnchoredQuarantineInterventionError::ContentIdentity(value))
    }
}

impl<E, I, S> From<QuarantineIntentLedgerError> for AnchoredQuarantineExecutionError<E, I, S>
where
    E: StdError + Send + Sync + 'static,
    I: StdError + Send + Sync + 'static,
    S: StdError + Send + Sync + 'static,
{
    fn from(value: QuarantineIntentLedgerError) -> Self {
        Self::Intervention(AnchoredQuarantineInterventionError::IntentLedger(value))
    }
}

impl<E, I, S> From<QuarantineLedgerError> for AnchoredQuarantineExecutionError<E, I, S>
where
    E: StdError + Send + Sync + 'static,
    I: StdError + Send + Sync + 'static,
    S: StdError + Send + Sync + 'static,
{
    fn from(value: QuarantineLedgerError) -> Self {
        Self::Intervention(AnchoredQuarantineInterventionError::QuarantineLedger(value))
    }
}

impl<E, I, S> From<EpisodicQuarantineInterventionError>
    for AnchoredQuarantineExecutionError<E, I, S>
where
    E: StdError + Send + Sync + 'static,
    I: StdError + Send + Sync + 'static,
    S: StdError + Send + Sync + 'static,
{
    fn from(value: EpisodicQuarantineInterventionError) -> Self {
        Self::Intervention(AnchoredQuarantineInterventionError::Escrow(value.to_string()))
    }
}
