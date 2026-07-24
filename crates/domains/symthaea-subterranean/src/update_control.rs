// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safe activation policy for externally verified software/configuration updates.
//!
//! Artifact bytes and signatures are verified outside this crate. This module
//! validates externally supplied digests, operational preconditions, schema
//! compatibility, health deadlines and rollback state transitions.

use crate::operator_authority::OperatorConstraint;
use serde::{Deserialize, Serialize};

pub const UPDATE_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactDigest(pub [u8; 32]);

impl ArtifactDigest {
    pub const fn is_valid(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateManifest {
    pub schema_version: u32,
    pub release_id: u64,
    pub artifact_digest: ArtifactDigest,
    pub configuration_digest: ArtifactDigest,
    pub rollback_digest: ArtifactDigest,
    pub minimum_checkpoint_schema: u32,
    pub issued_epoch: u64,
    pub expires_step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UpdatePreconditions {
    pub at_surface_or_service_bay: bool,
    pub active_work: bool,
    pub physical_hazard_clear: bool,
    pub battery_ratio: f64,
    pub operator_constraint: OperatorConstraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateState {
    Idle,
    Staged,
    PendingHealth,
    Active,
    RollbackRequired,
    RolledBack,
}

impl UpdateState {
    pub const fn code(self) -> u16 {
        match self {
            Self::Idle => 0,
            Self::Staged => 1,
            Self::PendingHealth => 2,
            Self::Active => 3,
            Self::RollbackRequired => 4,
            Self::RolledBack => 5,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Staged => "staged",
            Self::PendingHealth => "pending_health",
            Self::Active => "active",
            Self::RollbackRequired => "rollback_required",
            Self::RolledBack => "rolled_back",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateRejection {
    InvalidManifestSchema,
    InvalidDigest,
    Expired,
    StaleEpoch,
    SameArtifact,
    RollbackMismatch,
    CheckpointIncompatible,
    NotAtSafeLocation,
    ActiveWork,
    PhysicalHazard,
    InsufficientBattery,
    MaintenanceLockRequired,
    InvalidTransition,
    HealthDeadlineExpired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateManager {
    current_digest: ArtifactDigest,
    current_epoch: u64,
    staged: Option<UpdateManifest>,
    previous_digest: Option<ArtifactDigest>,
    state: UpdateState,
    activation_step: Option<u64>,
    health_deadline_step: Option<u64>,
    successful_activations: u64,
    rollbacks: u64,
}

impl UpdateManager {
    pub fn new(
        current_digest: ArtifactDigest,
        current_epoch: u64,
    ) -> Result<Self, UpdateRejection> {
        if !current_digest.is_valid() {
            return Err(UpdateRejection::InvalidDigest);
        }
        Ok(Self {
            current_digest,
            current_epoch,
            staged: None,
            previous_digest: None,
            state: UpdateState::Idle,
            activation_step: None,
            health_deadline_step: None,
            successful_activations: 0,
            rollbacks: 0,
        })
    }

    pub fn current_digest(&self) -> ArtifactDigest {
        self.current_digest
    }

    pub fn validate(&self) -> bool {
        if !self.current_digest.is_valid() {
            return false;
        }
        match self.state {
            UpdateState::Idle | UpdateState::Active | UpdateState::RolledBack => {
                self.staged.is_none()
                    && self.previous_digest.is_none()
                    && self.activation_step.is_none()
                    && self.health_deadline_step.is_none()
            }
            UpdateState::Staged => {
                self.staged.is_some()
                    && self.previous_digest.is_none()
                    && self.activation_step.is_none()
                    && self.health_deadline_step.is_none()
            }
            UpdateState::PendingHealth | UpdateState::RollbackRequired => {
                self.staged.is_some()
                    && self.previous_digest.is_some()
                    && self.activation_step.is_some()
                    && self.health_deadline_step.is_some()
            }
        }
    }

    pub fn state(&self) -> UpdateState {
        self.state
    }

    pub fn staged_manifest(&self) -> Option<UpdateManifest> {
        self.staged
    }

    pub fn successful_activations(&self) -> u64 {
        self.successful_activations
    }

    pub fn rollbacks(&self) -> u64 {
        self.rollbacks
    }

    pub fn stage(
        &mut self,
        manifest: UpdateManifest,
        now_step: u64,
        checkpoint_schema: u32,
        preconditions: UpdatePreconditions,
    ) -> Result<(), UpdateRejection> {
        if manifest.schema_version != UPDATE_MANIFEST_SCHEMA_VERSION {
            return Err(UpdateRejection::InvalidManifestSchema);
        }
        if !manifest.artifact_digest.is_valid()
            || !manifest.configuration_digest.is_valid()
            || !manifest.rollback_digest.is_valid()
        {
            return Err(UpdateRejection::InvalidDigest);
        }
        if now_step > manifest.expires_step {
            return Err(UpdateRejection::Expired);
        }
        if manifest.issued_epoch <= self.current_epoch {
            return Err(UpdateRejection::StaleEpoch);
        }
        if manifest.artifact_digest == self.current_digest {
            return Err(UpdateRejection::SameArtifact);
        }
        if manifest.rollback_digest != self.current_digest {
            return Err(UpdateRejection::RollbackMismatch);
        }
        if checkpoint_schema < manifest.minimum_checkpoint_schema {
            return Err(UpdateRejection::CheckpointIncompatible);
        }
        Self::validate_preconditions(preconditions)?;
        if !matches!(
            self.state,
            UpdateState::Idle | UpdateState::RolledBack | UpdateState::Active
        ) {
            return Err(UpdateRejection::InvalidTransition);
        }
        self.staged = Some(manifest);
        self.state = UpdateState::Staged;
        Ok(())
    }

    fn validate_preconditions(preconditions: UpdatePreconditions) -> Result<(), UpdateRejection> {
        if !preconditions.at_surface_or_service_bay {
            return Err(UpdateRejection::NotAtSafeLocation);
        }
        if preconditions.active_work {
            return Err(UpdateRejection::ActiveWork);
        }
        if !preconditions.physical_hazard_clear {
            return Err(UpdateRejection::PhysicalHazard);
        }
        if !preconditions.battery_ratio.is_finite() || preconditions.battery_ratio < 0.4 {
            return Err(UpdateRejection::InsufficientBattery);
        }
        if preconditions.operator_constraint != OperatorConstraint::MaintenanceLock {
            return Err(UpdateRejection::MaintenanceLockRequired);
        }
        Ok(())
    }

    pub fn activate(
        &mut self,
        now_step: u64,
        health_window_steps: u64,
        preconditions: UpdatePreconditions,
    ) -> Result<ArtifactDigest, UpdateRejection> {
        if self.state != UpdateState::Staged {
            return Err(UpdateRejection::InvalidTransition);
        }
        Self::validate_preconditions(preconditions)?;
        let manifest = self.staged.ok_or(UpdateRejection::InvalidTransition)?;
        if now_step > manifest.expires_step {
            return Err(UpdateRejection::Expired);
        }
        self.previous_digest = Some(self.current_digest);
        self.current_digest = manifest.artifact_digest;
        self.current_epoch = manifest.issued_epoch;
        self.activation_step = Some(now_step);
        self.health_deadline_step = Some(now_step.saturating_add(health_window_steps.max(1)));
        self.state = UpdateState::PendingHealth;
        Ok(self.current_digest)
    }

    pub fn observe_health(
        &mut self,
        healthy: bool,
        now_step: u64,
    ) -> Result<UpdateState, UpdateRejection> {
        if self.state != UpdateState::PendingHealth {
            return Err(UpdateRejection::InvalidTransition);
        }
        let deadline = self
            .health_deadline_step
            .ok_or(UpdateRejection::InvalidTransition)?;
        if !healthy {
            self.state = UpdateState::RollbackRequired;
            return Ok(self.state);
        }
        if now_step > deadline {
            self.state = UpdateState::RollbackRequired;
            return Err(UpdateRejection::HealthDeadlineExpired);
        }
        self.state = UpdateState::Active;
        self.successful_activations = self.successful_activations.saturating_add(1);
        self.staged = None;
        self.previous_digest = None;
        self.activation_step = None;
        self.health_deadline_step = None;
        Ok(self.state)
    }

    pub fn require_rollback(&mut self) -> Result<(), UpdateRejection> {
        if !matches!(self.state, UpdateState::PendingHealth | UpdateState::Active) {
            return Err(UpdateRejection::InvalidTransition);
        }
        self.state = UpdateState::RollbackRequired;
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<ArtifactDigest, UpdateRejection> {
        if self.state != UpdateState::RollbackRequired {
            return Err(UpdateRejection::InvalidTransition);
        }
        let previous = self
            .previous_digest
            .ok_or(UpdateRejection::InvalidTransition)?;
        self.current_digest = previous;
        self.staged = None;
        self.previous_digest = None;
        self.activation_step = None;
        self.health_deadline_step = None;
        self.state = UpdateState::RolledBack;
        self.rollbacks = self.rollbacks.saturating_add(1);
        Ok(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> ArtifactDigest {
        ArtifactDigest([byte; 32])
    }

    fn manifest() -> UpdateManifest {
        UpdateManifest {
            schema_version: UPDATE_MANIFEST_SCHEMA_VERSION,
            release_id: 2,
            artifact_digest: digest(2),
            configuration_digest: digest(3),
            rollback_digest: digest(1),
            minimum_checkpoint_schema: 1,
            issued_epoch: 2,
            expires_step: 1_000,
        }
    }

    fn safe_preconditions() -> UpdatePreconditions {
        UpdatePreconditions {
            at_surface_or_service_bay: true,
            active_work: false,
            physical_hazard_clear: true,
            battery_ratio: 0.9,
            operator_constraint: OperatorConstraint::MaintenanceLock,
        }
    }

    #[test]
    fn update_cannot_stage_during_active_work() {
        let mut manager = UpdateManager::new(digest(1), 1).expect("valid current digest");
        let mut preconditions = safe_preconditions();
        preconditions.active_work = true;
        assert_eq!(
            manager.stage(manifest(), 10, 1, preconditions),
            Err(UpdateRejection::ActiveWork)
        );
    }

    #[test]
    fn failed_health_check_requires_and_completes_rollback() {
        let mut manager = UpdateManager::new(digest(1), 1).expect("valid current digest");
        manager
            .stage(manifest(), 10, 1, safe_preconditions())
            .expect("stage should succeed");
        assert_eq!(
            manager
                .activate(11, 50, safe_preconditions())
                .expect("activation should succeed"),
            digest(2)
        );
        assert_eq!(
            manager.observe_health(false, 12),
            Ok(UpdateState::RollbackRequired)
        );
        assert_eq!(manager.rollback(), Ok(digest(1)));
        assert_eq!(manager.state(), UpdateState::RolledBack);
    }

    #[test]
    fn successful_health_check_commits_activation() {
        let mut manager = UpdateManager::new(digest(1), 1).expect("valid current digest");
        manager
            .stage(manifest(), 10, 1, safe_preconditions())
            .expect("stage should succeed");
        manager
            .activate(11, 50, safe_preconditions())
            .expect("activation should succeed");
        assert_eq!(manager.observe_health(true, 20), Ok(UpdateState::Active));
        assert_eq!(manager.current_digest(), digest(2));
        assert_eq!(manager.successful_activations(), 1);
    }
}
