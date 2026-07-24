// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic runtime failure containment for an authorized print job.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::telemetry::VerifiedMachineTelemetry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGuardPolicy {
    pub heartbeat_timeout_s: f64,
    pub progress_stall_timeout_s: f64,
    pub minimum_progress_delta: f32,
    pub thermal_settle_time_s: f64,
    pub max_nozzle_deviation_c: f32,
    pub max_bed_deviation_c: f32,
    pub absolute_max_nozzle_c: f32,
    pub absolute_max_bed_c: f32,
}

impl Default for ExecutionGuardPolicy {
    fn default() -> Self {
        Self {
            heartbeat_timeout_s: 10.0,
            progress_stall_timeout_s: 120.0,
            minimum_progress_delta: 0.0001,
            thermal_settle_time_s: 90.0,
            max_nozzle_deviation_c: 15.0,
            max_bed_deviation_c: 10.0,
            absolute_max_nozzle_c: 320.0,
            absolute_max_bed_c: 150.0,
        }
    }
}

impl ExecutionGuardPolicy {
    pub fn validate(&self) -> Result<(), &'static str> {
        for (name, value) in [
            ("heartbeat_timeout_s", self.heartbeat_timeout_s),
            ("progress_stall_timeout_s", self.progress_stall_timeout_s),
            ("thermal_settle_time_s", self.thermal_settle_time_s),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(name);
            }
        }
        for (name, value) in [
            ("minimum_progress_delta", self.minimum_progress_delta),
            ("max_nozzle_deviation_c", self.max_nozzle_deviation_c),
            ("max_bed_deviation_c", self.max_bed_deviation_c),
            ("absolute_max_nozzle_c", self.absolute_max_nozzle_c),
            ("absolute_max_bed_c", self.absolute_max_bed_c),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(name);
            }
        }
        if self.minimum_progress_delta > 1.0 {
            return Err("minimum_progress_delta");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    pub elapsed_s: f64,
    pub heartbeat_sequence: u64,
    pub progress: f32,
    pub nozzle_actual_c: f32,
    pub nozzle_target_c: f32,
    pub bed_actual_c: f32,
    pub bed_target_c: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContainmentAction {
    Continue,
    Pause,
    Cancel,
    EmergencyStop,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GuardViolation {
    InvalidPolicy(&'static str),
    NonFiniteTelemetry(&'static str),
    TimeRegressed,
    ProgressOutOfRange(f32),
    ProgressRegressed {
        previous: f32,
        current: f32,
    },
    HeartbeatStale {
        age_s: f64,
        maximum_s: f64,
    },
    ProgressStalled {
        age_s: f64,
        maximum_s: f64,
    },
    NozzleDeviation {
        actual_c: f32,
        target_c: f32,
        maximum_c: f32,
    },
    BedDeviation {
        actual_c: f32,
        target_c: f32,
        maximum_c: f32,
    },
    NozzleOverTemperature {
        actual_c: f32,
        maximum_c: f32,
    },
    BedOverTemperature {
        actual_c: f32,
        maximum_c: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GuardDecision {
    pub action: ContainmentAction,
    pub new_violations: Vec<GuardViolation>,
    pub latched_action: ContainmentAction,
}

/// Runtime decision bound to the exact verified telemetry evidence that
/// produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedGuardDecision {
    pub telemetry_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub machine_id: String,
    pub printer_job_id: String,
    pub frame_sequence: u64,
    pub decision: GuardDecision,
}

pub const EXECUTION_GUARD_CHECKPOINT_SCHEMA: &str =
    "symthaea.fabrication.execution-guard-checkpoint.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGuardCheckpoint {
    pub schema_version: String,
    pub policy: ExecutionGuardPolicy,
    pub start_time_s: Option<f64>,
    pub last_elapsed_s: Option<f64>,
    pub last_heartbeat_sequence: Option<u64>,
    pub last_heartbeat_time_s: Option<f64>,
    pub last_progress: Option<f32>,
    pub last_progress_time_s: Option<f64>,
    pub latched_action: ContainmentAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionCheckpointError {
    UnsupportedSchema,
    InvalidPolicy(&'static str),
    NonFiniteField(&'static str),
    NegativeField(&'static str),
    ProgressOutOfRange,
    TimeInconsistent(&'static str),
    HeartbeatStateInconsistent,
    ProgressStateInconsistent,
    Encoding(String),
}

impl ExecutionGuardCheckpoint {
    pub fn validate(&self) -> Result<(), ExecutionCheckpointError> {
        if self.schema_version != EXECUTION_GUARD_CHECKPOINT_SCHEMA {
            return Err(ExecutionCheckpointError::UnsupportedSchema);
        }
        self.policy
            .validate()
            .map_err(ExecutionCheckpointError::InvalidPolicy)?;
        for (name, value) in [
            ("start_time_s", self.start_time_s),
            ("last_elapsed_s", self.last_elapsed_s),
            ("last_heartbeat_time_s", self.last_heartbeat_time_s),
            ("last_progress_time_s", self.last_progress_time_s),
        ] {
            if let Some(value) = value {
                if !value.is_finite() {
                    return Err(ExecutionCheckpointError::NonFiniteField(name));
                }
                if value < 0.0 {
                    return Err(ExecutionCheckpointError::NegativeField(name));
                }
            }
        }
        if self
            .last_progress
            .is_some_and(|progress| !progress.is_finite() || !(0.0..=1.0).contains(&progress))
        {
            return Err(ExecutionCheckpointError::ProgressOutOfRange);
        }
        if self
            .start_time_s
            .is_some_and(|start| self.last_elapsed_s.is_some_and(|last| start > last))
        {
            return Err(ExecutionCheckpointError::TimeInconsistent(
                "start_time_s exceeds last_elapsed_s",
            ));
        }
        for (name, value) in [
            ("last_heartbeat_time_s", self.last_heartbeat_time_s),
            ("last_progress_time_s", self.last_progress_time_s),
        ] {
            if value.is_some_and(|time| self.last_elapsed_s.is_none_or(|last| time > last)) {
                return Err(ExecutionCheckpointError::TimeInconsistent(name));
            }
        }
        if self.last_heartbeat_sequence.is_some() != self.last_heartbeat_time_s.is_some() {
            return Err(ExecutionCheckpointError::HeartbeatStateInconsistent);
        }
        if self.last_progress.is_some() != self.last_progress_time_s.is_some() {
            return Err(ExecutionCheckpointError::ProgressStateInconsistent);
        }
        Ok(())
    }

    pub fn progress(&self) -> Option<f32> {
        self.last_progress
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionGuard {
    policy: ExecutionGuardPolicy,
    start_time_s: Option<f64>,
    last_elapsed_s: Option<f64>,
    last_heartbeat_sequence: Option<u64>,
    last_heartbeat_time_s: Option<f64>,
    last_progress: Option<f32>,
    last_progress_time_s: Option<f64>,
    latched_action: ContainmentAction,
}

impl ExecutionGuard {
    pub fn new(policy: ExecutionGuardPolicy) -> Result<Self, GuardViolation> {
        policy.validate().map_err(GuardViolation::InvalidPolicy)?;
        Ok(Self {
            policy,
            start_time_s: None,
            last_elapsed_s: None,
            last_heartbeat_sequence: None,
            last_heartbeat_time_s: None,
            last_progress: None,
            last_progress_time_s: None,
            latched_action: ContainmentAction::Continue,
        })
    }

    pub fn checkpoint(&self) -> ExecutionGuardCheckpoint {
        ExecutionGuardCheckpoint {
            schema_version: EXECUTION_GUARD_CHECKPOINT_SCHEMA.into(),
            policy: self.policy,
            start_time_s: self.start_time_s,
            last_elapsed_s: self.last_elapsed_s,
            last_heartbeat_sequence: self.last_heartbeat_sequence,
            last_heartbeat_time_s: self.last_heartbeat_time_s,
            last_progress: self.last_progress,
            last_progress_time_s: self.last_progress_time_s,
            latched_action: self.latched_action,
        }
    }

    pub fn restore(checkpoint: ExecutionGuardCheckpoint) -> Result<Self, ExecutionCheckpointError> {
        checkpoint.validate()?;
        Ok(Self {
            policy: checkpoint.policy,
            start_time_s: checkpoint.start_time_s,
            last_elapsed_s: checkpoint.last_elapsed_s,
            last_heartbeat_sequence: checkpoint.last_heartbeat_sequence,
            last_heartbeat_time_s: checkpoint.last_heartbeat_time_s,
            last_progress: checkpoint.last_progress,
            last_progress_time_s: checkpoint.last_progress_time_s,
            latched_action: checkpoint.latched_action,
        })
    }

    pub fn policy(&self) -> ExecutionGuardPolicy {
        self.policy
    }

    /// Observe only lifecycle-governed telemetry and retain its identity in
    /// the returned decision evidence.
    pub fn observe_verified(
        &mut self,
        telemetry: &VerifiedMachineTelemetry,
    ) -> VerifiedGuardDecision {
        let payload = telemetry.payload();
        VerifiedGuardDecision {
            telemetry_digest: telemetry.telemetry_digest(),
            trust_snapshot_digest: telemetry.trust_snapshot_digest(),
            machine_id: payload.machine_id.clone(),
            printer_job_id: payload.printer_job_id.clone(),
            frame_sequence: payload.frame_sequence,
            decision: self.observe(telemetry.execution_telemetry()),
        }
    }

    pub fn observe(&mut self, telemetry: ExecutionTelemetry) -> GuardDecision {
        let mut violations = Vec::new();
        validate_finite(telemetry, &mut violations);
        if let Some(previous) = self.last_elapsed_s {
            if telemetry.elapsed_s < previous {
                violations.push(GuardViolation::TimeRegressed);
            }
        }
        if !(0.0..=1.0).contains(&telemetry.progress) {
            violations.push(GuardViolation::ProgressOutOfRange(telemetry.progress));
        }
        let start = *self.start_time_s.get_or_insert(telemetry.elapsed_s);

        if self.last_heartbeat_sequence != Some(telemetry.heartbeat_sequence) {
            self.last_heartbeat_sequence = Some(telemetry.heartbeat_sequence);
            self.last_heartbeat_time_s = Some(telemetry.elapsed_s);
        } else if let Some(last_heartbeat) = self.last_heartbeat_time_s {
            let age = telemetry.elapsed_s - last_heartbeat;
            if age > self.policy.heartbeat_timeout_s {
                violations.push(GuardViolation::HeartbeatStale {
                    age_s: age,
                    maximum_s: self.policy.heartbeat_timeout_s,
                });
            }
        }

        match self.last_progress {
            None => {
                self.last_progress = Some(telemetry.progress);
                self.last_progress_time_s = Some(telemetry.elapsed_s);
            }
            Some(previous) if telemetry.progress + f32::EPSILON < previous => {
                violations.push(GuardViolation::ProgressRegressed {
                    previous,
                    current: telemetry.progress,
                });
            }
            Some(previous)
                if telemetry.progress - previous >= self.policy.minimum_progress_delta =>
            {
                self.last_progress = Some(telemetry.progress);
                self.last_progress_time_s = Some(telemetry.elapsed_s);
            }
            Some(_) if telemetry.progress < 1.0 => {
                if let Some(last_progress_time) = self.last_progress_time_s {
                    let age = telemetry.elapsed_s - last_progress_time;
                    if age > self.policy.progress_stall_timeout_s {
                        violations.push(GuardViolation::ProgressStalled {
                            age_s: age,
                            maximum_s: self.policy.progress_stall_timeout_s,
                        });
                    }
                }
            }
            Some(_) => {}
        }

        if telemetry.nozzle_actual_c > self.policy.absolute_max_nozzle_c {
            violations.push(GuardViolation::NozzleOverTemperature {
                actual_c: telemetry.nozzle_actual_c,
                maximum_c: self.policy.absolute_max_nozzle_c,
            });
        }
        if telemetry.bed_actual_c > self.policy.absolute_max_bed_c {
            violations.push(GuardViolation::BedOverTemperature {
                actual_c: telemetry.bed_actual_c,
                maximum_c: self.policy.absolute_max_bed_c,
            });
        }
        if telemetry.elapsed_s - start >= self.policy.thermal_settle_time_s {
            if (telemetry.nozzle_actual_c - telemetry.nozzle_target_c).abs()
                > self.policy.max_nozzle_deviation_c
            {
                violations.push(GuardViolation::NozzleDeviation {
                    actual_c: telemetry.nozzle_actual_c,
                    target_c: telemetry.nozzle_target_c,
                    maximum_c: self.policy.max_nozzle_deviation_c,
                });
            }
            if (telemetry.bed_actual_c - telemetry.bed_target_c).abs()
                > self.policy.max_bed_deviation_c
            {
                violations.push(GuardViolation::BedDeviation {
                    actual_c: telemetry.bed_actual_c,
                    target_c: telemetry.bed_target_c,
                    maximum_c: self.policy.max_bed_deviation_c,
                });
            }
        }

        let action = violations
            .iter()
            .map(action_for_violation)
            .max()
            .unwrap_or(ContainmentAction::Continue);
        self.latched_action = self.latched_action.max(action);
        self.last_elapsed_s = Some(telemetry.elapsed_s);
        GuardDecision {
            action,
            new_violations: violations,
            latched_action: self.latched_action,
        }
    }

    pub fn latched_action(&self) -> ContainmentAction {
        self.latched_action
    }
}

pub fn digest_execution_checkpoint(
    checkpoint: &ExecutionGuardCheckpoint,
) -> Result<Sha256Digest, ExecutionCheckpointError> {
    checkpoint.validate()?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| ExecutionCheckpointError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.execution-guard-checkpoint-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_finite(telemetry: ExecutionTelemetry, violations: &mut Vec<GuardViolation>) {
    for (name, value) in [
        ("elapsed_s", telemetry.elapsed_s),
        ("progress", telemetry.progress as f64),
        ("nozzle_actual_c", telemetry.nozzle_actual_c as f64),
        ("nozzle_target_c", telemetry.nozzle_target_c as f64),
        ("bed_actual_c", telemetry.bed_actual_c as f64),
        ("bed_target_c", telemetry.bed_target_c as f64),
    ] {
        if !value.is_finite() {
            violations.push(GuardViolation::NonFiniteTelemetry(name));
        }
    }
}

fn action_for_violation(violation: &GuardViolation) -> ContainmentAction {
    match violation {
        GuardViolation::NonFiniteTelemetry(_)
        | GuardViolation::TimeRegressed
        | GuardViolation::NozzleOverTemperature { .. }
        | GuardViolation::BedOverTemperature { .. } => ContainmentAction::EmergencyStop,
        GuardViolation::HeartbeatStale { .. } | GuardViolation::ProgressRegressed { .. } => {
            ContainmentAction::Cancel
        }
        GuardViolation::InvalidPolicy(_)
        | GuardViolation::ProgressOutOfRange(_)
        | GuardViolation::ProgressStalled { .. }
        | GuardViolation::NozzleDeviation { .. }
        | GuardViolation::BedDeviation { .. } => ContainmentAction::Pause,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn telemetry(elapsed_s: f64, heartbeat_sequence: u64, progress: f32) -> ExecutionTelemetry {
        ExecutionTelemetry {
            elapsed_s,
            heartbeat_sequence,
            progress,
            nozzle_actual_c: 200.0,
            nozzle_target_c: 200.0,
            bed_actual_c: 60.0,
            bed_target_c: 60.0,
        }
    }

    #[test]
    fn healthy_progress_remains_clear() {
        let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
        assert_eq!(
            guard.observe(telemetry(0.0, 1, 0.0)).action,
            ContainmentAction::Continue
        );
        assert_eq!(
            guard.observe(telemetry(30.0, 2, 0.2)).action,
            ContainmentAction::Continue
        );
    }

    #[test]
    fn stale_heartbeat_cancels_and_latches() {
        let mut policy = ExecutionGuardPolicy::default();
        policy.heartbeat_timeout_s = 5.0;
        let mut guard = ExecutionGuard::new(policy).unwrap();
        guard.observe(telemetry(0.0, 1, 0.0));
        let decision = guard.observe(telemetry(6.0, 1, 0.1));
        assert_eq!(decision.action, ContainmentAction::Cancel);
        assert_eq!(guard.latched_action(), ContainmentAction::Cancel);
    }

    #[test]
    fn absolute_overtemperature_emergency_stops() {
        let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
        let mut sample = telemetry(0.0, 1, 0.0);
        sample.nozzle_actual_c = 321.0;
        assert_eq!(
            guard.observe(sample).latched_action,
            ContainmentAction::EmergencyStop
        );
    }

    #[test]
    fn progress_stall_pauses_after_budget() {
        let mut policy = ExecutionGuardPolicy::default();
        policy.progress_stall_timeout_s = 5.0;
        let mut guard = ExecutionGuard::new(policy).unwrap();
        guard.observe(telemetry(0.0, 1, 0.2));
        let decision = guard.observe(telemetry(6.0, 2, 0.2));
        assert_eq!(decision.action, ContainmentAction::Pause);
    }

    #[test]
    fn checkpoint_round_trip_preserves_latched_state() {
        let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
        guard.observe(telemetry(0.0, 1, 0.1));
        let mut hot = telemetry(1.0, 2, 0.2);
        hot.nozzle_actual_c = 321.0;
        guard.observe(hot);
        let checkpoint = guard.checkpoint();
        let digest = digest_execution_checkpoint(&checkpoint).unwrap();
        let restored = ExecutionGuard::restore(checkpoint.clone()).unwrap();
        assert_eq!(restored.latched_action(), ContainmentAction::EmergencyStop);
        assert_eq!(digest, digest_execution_checkpoint(&checkpoint).unwrap());
    }

    #[test]
    fn checkpoint_time_tampering_is_rejected() {
        let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
        guard.observe(telemetry(10.0, 1, 0.1));
        let mut checkpoint = guard.checkpoint();
        checkpoint.last_progress_time_s = Some(11.0);
        assert!(matches!(
            ExecutionGuard::restore(checkpoint),
            Err(ExecutionCheckpointError::TimeInconsistent(
                "last_progress_time_s"
            ))
        ));
    }
}
