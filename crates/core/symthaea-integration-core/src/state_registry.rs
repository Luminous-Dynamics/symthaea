// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Registry admission for already-produced desired/observed state evidence.
//!
//! v0.1 does not yet add another invocation/provider trait. State assertions are
//! derived from existing registered read-only sources, then rebound to that
//! registry slot and subjected to centrally chosen limits before they can enter
//! the world model or temporal reasoning path.
//!
//! Structural and chronology validation live on [`StateSnapshot`] itself. The
//! registry intentionally reuses that canonical validator rather than maintaining
//! a second timestamp policy that could drift from direct state consumers.

use crate::{
    IntegrationError, IntegrationId, IntegrationRegistry, StateHistory, StateHistoryLimits,
    StateLimits, StateSnapshot, validate_state_snapshot_origins,
};

impl IntegrationRegistry {
    pub fn admit_state_snapshot(
        &self,
        id: &IntegrationId,
        snapshot: &StateSnapshot,
    ) -> Result<(), IntegrationError> {
        self.admit_state_snapshot_with_limits(id, snapshot, &StateLimits::default())
    }

    pub fn admit_state_snapshot_with_limits(
        &self,
        id: &IntegrationId,
        snapshot: &StateSnapshot,
        limits: &StateLimits,
    ) -> Result<(), IntegrationError> {
        self.require_registered_state_source(id)?;
        if snapshot.integration_id != id.as_str() {
            return Err(IntegrationError::InvalidOutput(format!(
                "state source `{id}` returned snapshot attributed to `{}`",
                snapshot.integration_id
            )));
        }
        snapshot.validate_with_limits(limits).map_err(|error| {
            IntegrationError::InvalidOutput(format!(
                "integration `{id}` state evidence rejected by admission budget: {error}"
            ))
        })?;
        validate_state_snapshot_origins(snapshot).map_err(|error| {
            IntegrationError::InvalidOutput(format!(
                "integration `{id}` state evidence rejected by origin contract: {error}"
            ))
        })
    }

    pub fn admit_state_history(
        &self,
        id: &IntegrationId,
        history: &StateHistory,
    ) -> Result<(), IntegrationError> {
        self.admit_state_history_with_limits(id, history, &StateHistoryLimits::default())
    }

    pub fn admit_state_history_with_limits(
        &self,
        id: &IntegrationId,
        history: &StateHistory,
        limits: &StateHistoryLimits,
    ) -> Result<(), IntegrationError> {
        self.require_registered_state_source(id)?;
        if history.integration_id != id.as_str() {
            return Err(IntegrationError::InvalidOutput(format!(
                "state source `{id}` returned history attributed to `{}`",
                history.integration_id
            )));
        }
        history.validate_with_limits(limits).map_err(|error| {
            IntegrationError::InvalidOutput(format!(
                "integration `{id}` state history rejected by admission budget: {error}"
            ))
        })?;
        for snapshot in &history.snapshots {
            validate_state_snapshot_origins(snapshot).map_err(|error| {
                IntegrationError::InvalidOutput(format!(
                    "integration `{id}` state history rejected by origin contract: {error}"
                ))
            })?;
        }
        Ok(())
    }

    fn require_registered_state_source(&self, id: &IntegrationId) -> Result<(), IntegrationError> {
        if self.manifest(id).is_none() {
            return Err(IntegrationError::Unsupported(format!(
                "no registered integration manifest for state source `{id}`"
            )));
        }
        Ok(())
    }
}
