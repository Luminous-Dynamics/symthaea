// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrowed currentness fence for the crash-durable admission reservation head.
//!
//! A plain `current_head()` read proves only a momentary observation because its local and
//! cross-process locks are released before the caller can continue. This module keeps those same
//! owner-local mutation barriers held for the entire borrowed fence lifetime. The caller supplies
//! only the exact head already committed by the composed evidence; it cannot choose filesystem
//! state, lock paths, or any authority input.

use std::fs::File;
use std::sync::MutexGuard;

use thiserror::Error;

use crate::{
    AdmissionReservationCheckpointV1, AdmissionReservationError, AdmissionReservationHead,
    DurableAdmissionReservationStore,
};

impl DurableAdmissionReservationStore {
    /// Re-read the exact durable reservation under both owner-local mutation barriers and hold
    /// those barriers until the returned fence is dropped.
    pub fn fence_current<'a>(
        &'a self,
        expected_head: AdmissionReservationHead,
    ) -> Result<CurrentAdmissionReservationFence<'a>, CurrentAdmissionReservationFenceError> {
        let local = self
            .local_lock
            .lock()
            .map_err(|_| AdmissionReservationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(AdmissionReservationError::Io)?;

        let checkpoint = self.read_state_locked()?;
        self.validate_checkpoint_for_store(&checkpoint)?;
        let current_head = checkpoint.head()?;
        if current_head != expected_head {
            return Err(CurrentAdmissionReservationFenceError::HeadMismatch {
                expected: expected_head,
                current: current_head,
            });
        }

        Ok(CurrentAdmissionReservationFence {
            _local: local,
            _kernel: kernel,
            checkpoint,
            head: current_head,
        })
    }
}

/// Borrowed proof that the exact expected admission-reservation head is still the durable head
/// while the store's local mutex and cross-process kernel lock remain held.
#[derive(Debug)]
pub struct CurrentAdmissionReservationFence<'a> {
    _local: MutexGuard<'a, ()>,
    _kernel: File,
    checkpoint: AdmissionReservationCheckpointV1,
    head: AdmissionReservationHead,
}

impl CurrentAdmissionReservationFence<'_> {
    pub const fn head(&self) -> AdmissionReservationHead {
        self.head
    }

    pub fn checkpoint(&self) -> &AdmissionReservationCheckpointV1 {
        &self.checkpoint
    }
}

#[derive(Debug, Error)]
pub enum CurrentAdmissionReservationFenceError {
    #[error("admission-reservation store could not establish currentness: {0}")]
    Store(#[from] AdmissionReservationError),
    #[error("durable admission-reservation head differs from composed evidence")]
    HeadMismatch {
        expected: AdmissionReservationHead,
        current: AdmissionReservationHead,
    },
}
