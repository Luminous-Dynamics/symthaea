// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrowed currentness fence for the crash-durable device-semantic head.
//!
//! Semantic persistence already serializes mutation under one local mutex and one kernel file
//! lock. The missing property was retaining those same barriers through the later actuation
//! handoff. This module re-reads and validates the checkpoint while both locks are held and keeps
//! them alive in a non-clone borrowed fence.

use std::fs::File;
use std::sync::MutexGuard;

use symthaea_iot_device_protocol::{DeviceSemanticCheckpointV1, DeviceSemanticHead};
use thiserror::Error;

use crate::{DurableSemanticAcceptanceStore, SemanticPersistenceError};

impl DurableSemanticAcceptanceStore {
    /// Re-read the exact durable semantic checkpoint under both owner-local mutation barriers and
    /// retain those barriers until the returned fence is dropped.
    pub fn fence_current<'a>(
        &'a self,
        expected_head: DeviceSemanticHead,
    ) -> Result<CurrentSemanticHeadFence<'a>, CurrentSemanticHeadFenceError> {
        let local = self
            .local_lock
            .lock()
            .map_err(|_| SemanticPersistenceError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(SemanticPersistenceError::Io)?;

        let checkpoint = self.read_state_locked()?;
        self.verify_loaded_checkpoint(&checkpoint)?;
        let current_head = checkpoint.head()?;
        if current_head != expected_head {
            return Err(CurrentSemanticHeadFenceError::HeadMismatch {
                expected: expected_head,
                current: current_head,
            });
        }

        Ok(CurrentSemanticHeadFence {
            _local: local,
            _kernel: kernel,
            checkpoint,
            head: current_head,
        })
    }
}

/// Borrowed proof that the exact composed device-semantic head is still crash-durable while
/// semantic mutation remains excluded locally and across processes.
#[derive(Debug)]
pub struct CurrentSemanticHeadFence<'a> {
    _local: MutexGuard<'a, ()>,
    _kernel: File,
    checkpoint: DeviceSemanticCheckpointV1,
    head: DeviceSemanticHead,
}

impl CurrentSemanticHeadFence<'_> {
    pub const fn head(&self) -> DeviceSemanticHead {
        self.head
    }

    pub fn checkpoint(&self) -> &DeviceSemanticCheckpointV1 {
        &self.checkpoint
    }
}

#[derive(Debug, Error)]
pub enum CurrentSemanticHeadFenceError {
    #[error("semantic-persistence store could not establish currentness: {0}")]
    Store(#[from] SemanticPersistenceError),
    #[error("durable device-semantic head differs from composed evidence")]
    HeadMismatch {
        expected: DeviceSemanticHead,
        current: DeviceSemanticHead,
    },
}
