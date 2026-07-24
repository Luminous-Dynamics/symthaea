// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simplex-style runtime assurance for adaptive flight control.
//!
//! An advanced controller may command the aircraft only while independent
//! estimator, timing, envelope, controllability, and command-validity evidence
//! remains healthy. Any violation transfers immediately to a deterministic
//! baseline. Return to advanced control requires a stable dwell and bumpless
//! transition.

use serde::{Deserialize, Serialize};

use crate::types::HelicopterCommand;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeAssuranceConfig {
    pub schema_version: String,
    pub monitor_id: String,
    pub minimum_envelope_margin: f64,
    pub minimum_controllability_margin: f64,
    pub maximum_command_disagreement: [f32; 6],
    pub recovery_dwell_s: f64,
    pub transition_duration_s: f64,
}

impl Default for RuntimeAssuranceConfig {
    fn default() -> Self {
        Self {
            schema_version: "symthaea.helicopter.runtime-assurance.v1".into(),
            monitor_id: "primary-simplex-monitor".into(),
            minimum_envelope_margin: 0.1,
            minimum_controllability_margin: 0.1,
            maximum_command_disagreement: [0.5, 0.5, 0.5, 0.5, 0.35, 0.35],
            recovery_dwell_s: 2.0,
            transition_duration_s: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeAssuranceMode {
    Advanced,
    Baseline,
    RecoveryDwell,
    TransitionToAdvanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeAssuranceReason {
    AdvancedCommandInvalid,
    BaselineCommandInvalid,
    EstimatorUnhealthy,
    RealtimeUnhealthy,
    EnvelopeMarginInsufficient,
    ControllabilityMarginInsufficient,
    CommandDisagreement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAssuranceObservation {
    pub timestamp_s: f64,
    pub advanced_command: HelicopterCommand,
    pub baseline_command: HelicopterCommand,
    pub advanced_command_valid: bool,
    pub estimator_healthy: bool,
    pub realtime_healthy: bool,
    pub envelope_margin: f64,
    pub controllability_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAssuranceDecision {
    pub schema_version: String,
    pub monitor_id: String,
    pub timestamp_s: f64,
    pub mode: RuntimeAssuranceMode,
    pub selected_command: HelicopterCommand,
    pub advanced_weight: f32,
    pub reasons: Vec<RuntimeAssuranceReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeAssuranceError {
    InvalidConfiguration,
    InvalidTimestamp,
    NonMonotonicTimestamp,
    InvalidObservation,
    InvalidBaselineCommand,
}

#[derive(Debug, Clone)]
pub struct RuntimeAssuranceMonitor {
    config: RuntimeAssuranceConfig,
    mode: RuntimeAssuranceMode,
    last_timestamp_s: Option<f64>,
    safe_since_s: Option<f64>,
    transition_started_s: Option<f64>,
}

impl RuntimeAssuranceMonitor {
    pub fn new(config: RuntimeAssuranceConfig) -> Result<Self, RuntimeAssuranceError> {
        if config.schema_version.trim().is_empty()
            || config.monitor_id.trim().is_empty()
            || !config.minimum_envelope_margin.is_finite()
            || !config.minimum_controllability_margin.is_finite()
            || config
                .maximum_command_disagreement
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !config.recovery_dwell_s.is_finite()
            || config.recovery_dwell_s < 0.0
            || !config.transition_duration_s.is_finite()
            || config.transition_duration_s < 0.0
        {
            return Err(RuntimeAssuranceError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            mode: RuntimeAssuranceMode::Advanced,
            last_timestamp_s: None,
            safe_since_s: None,
            transition_started_s: None,
        })
    }

    pub fn mode(&self) -> RuntimeAssuranceMode {
        self.mode
    }

    pub fn reset(&mut self) {
        self.mode = RuntimeAssuranceMode::Advanced;
        self.last_timestamp_s = None;
        self.safe_since_s = None;
        self.transition_started_s = None;
    }

    pub fn evaluate(
        &mut self,
        observation: &RuntimeAssuranceObservation,
    ) -> Result<RuntimeAssuranceDecision, RuntimeAssuranceError> {
        if !observation.timestamp_s.is_finite() || observation.timestamp_s < 0.0 {
            return Err(RuntimeAssuranceError::InvalidTimestamp);
        }
        if self
            .last_timestamp_s
            .is_some_and(|previous| observation.timestamp_s < previous)
        {
            return Err(RuntimeAssuranceError::NonMonotonicTimestamp);
        }
        if !command_is_finite(&observation.baseline_command) {
            return Err(RuntimeAssuranceError::InvalidBaselineCommand);
        }
        if !observation.envelope_margin.is_finite()
            || !observation.controllability_margin.is_finite()
        {
            return Err(RuntimeAssuranceError::InvalidObservation);
        }
        self.last_timestamp_s = Some(observation.timestamp_s);

        let reasons = self.reasons(observation);
        let unsafe_now = !reasons.is_empty();
        if unsafe_now {
            self.mode = RuntimeAssuranceMode::Baseline;
            self.safe_since_s = None;
            self.transition_started_s = None;
            return Ok(self.decision(
                observation,
                RuntimeAssuranceMode::Baseline,
                observation.baseline_command.clamped(),
                0.0,
                reasons,
            ));
        }

        match self.mode {
            RuntimeAssuranceMode::Advanced => Ok(self.decision(
                observation,
                RuntimeAssuranceMode::Advanced,
                observation.advanced_command.clamped(),
                1.0,
                reasons,
            )),
            RuntimeAssuranceMode::Baseline | RuntimeAssuranceMode::RecoveryDwell => {
                let safe_since = *self.safe_since_s.get_or_insert(observation.timestamp_s);
                let safe_elapsed = observation.timestamp_s - safe_since;
                if safe_elapsed < self.config.recovery_dwell_s {
                    self.mode = RuntimeAssuranceMode::RecoveryDwell;
                    Ok(self.decision(
                        observation,
                        self.mode,
                        observation.baseline_command.clamped(),
                        0.0,
                        reasons,
                    ))
                } else {
                    self.mode = RuntimeAssuranceMode::TransitionToAdvanced;
                    let started = *self
                        .transition_started_s
                        .get_or_insert(observation.timestamp_s);
                    self.transition_decision(observation, started, reasons)
                }
            }
            RuntimeAssuranceMode::TransitionToAdvanced => {
                let started = self.transition_started_s.unwrap_or(observation.timestamp_s);
                self.transition_decision(observation, started, reasons)
            }
        }
    }

    fn transition_decision(
        &mut self,
        observation: &RuntimeAssuranceObservation,
        started_s: f64,
        reasons: Vec<RuntimeAssuranceReason>,
    ) -> Result<RuntimeAssuranceDecision, RuntimeAssuranceError> {
        let weight = if self.config.transition_duration_s == 0.0 {
            1.0
        } else {
            ((observation.timestamp_s - started_s) / self.config.transition_duration_s)
                .clamp(0.0, 1.0) as f32
        };
        let selected = observation
            .baseline_command
            .blend(observation.advanced_command, weight);
        if weight >= 1.0 {
            self.mode = RuntimeAssuranceMode::Advanced;
            self.safe_since_s = None;
            self.transition_started_s = None;
        }
        Ok(self.decision(observation, self.mode, selected, weight, reasons))
    }

    fn reasons(&self, observation: &RuntimeAssuranceObservation) -> Vec<RuntimeAssuranceReason> {
        let mut reasons = Vec::new();
        if !observation.advanced_command_valid || !command_is_finite(&observation.advanced_command)
        {
            reasons.push(RuntimeAssuranceReason::AdvancedCommandInvalid);
        }
        if !command_is_finite(&observation.baseline_command) {
            reasons.push(RuntimeAssuranceReason::BaselineCommandInvalid);
        }
        if !observation.estimator_healthy {
            reasons.push(RuntimeAssuranceReason::EstimatorUnhealthy);
        }
        if !observation.realtime_healthy {
            reasons.push(RuntimeAssuranceReason::RealtimeUnhealthy);
        }
        if observation.envelope_margin < self.config.minimum_envelope_margin {
            reasons.push(RuntimeAssuranceReason::EnvelopeMarginInsufficient);
        }
        if observation.controllability_margin < self.config.minimum_controllability_margin {
            reasons.push(RuntimeAssuranceReason::ControllabilityMarginInsufficient);
        }
        if command_disagreement(
            &observation.advanced_command,
            &observation.baseline_command,
            &self.config.maximum_command_disagreement,
        ) {
            reasons.push(RuntimeAssuranceReason::CommandDisagreement);
        }
        reasons
    }

    fn decision(
        &self,
        observation: &RuntimeAssuranceObservation,
        mode: RuntimeAssuranceMode,
        selected_command: HelicopterCommand,
        advanced_weight: f32,
        reasons: Vec<RuntimeAssuranceReason>,
    ) -> RuntimeAssuranceDecision {
        RuntimeAssuranceDecision {
            schema_version: self.config.schema_version.clone(),
            monitor_id: self.config.monitor_id.clone(),
            timestamp_s: observation.timestamp_s,
            mode,
            selected_command,
            advanced_weight,
            reasons,
        }
    }
}

fn command_is_finite(command: &HelicopterCommand) -> bool {
    command.to_ctrl().iter().all(|value| value.is_finite())
}

fn command_disagreement(
    advanced: &HelicopterCommand,
    baseline: &HelicopterCommand,
    maximum: &[f32; 6],
) -> bool {
    let advanced = advanced.to_ctrl();
    let baseline = baseline.to_ctrl();
    advanced
        .iter()
        .zip(baseline.iter())
        .zip(maximum.iter())
        .any(|((advanced, baseline), maximum)| (advanced - baseline).abs() > f64::from(*maximum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(timestamp_s: f64) -> RuntimeAssuranceObservation {
        RuntimeAssuranceObservation {
            timestamp_s,
            advanced_command: HelicopterCommand {
                collective: 0.32,
                cyclic_lon: 0.1,
                cyclic_lat: 0.0,
                pedal: 0.0,
                thrust: 0.61,
                tail_rotor: 0.5,
            },
            baseline_command: HelicopterCommand::hover(),
            advanced_command_valid: true,
            estimator_healthy: true,
            realtime_healthy: true,
            envelope_margin: 0.5,
            controllability_margin: 0.5,
        }
    }

    #[test]
    fn healthy_advanced_controller_retains_authority() {
        let mut monitor = RuntimeAssuranceMonitor::new(RuntimeAssuranceConfig::default()).unwrap();
        let decision = monitor.evaluate(&observation(0.0)).unwrap();
        assert_eq!(decision.mode, RuntimeAssuranceMode::Advanced);
        assert_eq!(decision.advanced_weight, 1.0);
    }

    #[test]
    fn estimator_failure_transfers_immediately() {
        let mut monitor = RuntimeAssuranceMonitor::new(RuntimeAssuranceConfig::default()).unwrap();
        let mut observed = observation(0.0);
        observed.estimator_healthy = false;
        let decision = monitor.evaluate(&observed).unwrap();
        assert_eq!(decision.mode, RuntimeAssuranceMode::Baseline);
        assert_eq!(decision.advanced_weight, 0.0);
        assert!(
            decision
                .reasons
                .contains(&RuntimeAssuranceReason::EstimatorUnhealthy)
        );
    }

    #[test]
    fn recovery_requires_dwell_and_transition() {
        let mut monitor = RuntimeAssuranceMonitor::new(RuntimeAssuranceConfig {
            recovery_dwell_s: 1.0,
            transition_duration_s: 1.0,
            ..Default::default()
        })
        .unwrap();
        let mut failed = observation(0.0);
        failed.realtime_healthy = false;
        monitor.evaluate(&failed).unwrap();

        assert_eq!(
            monitor.evaluate(&observation(0.5)).unwrap().mode,
            RuntimeAssuranceMode::RecoveryDwell
        );
        let transition = monitor.evaluate(&observation(1.5)).unwrap();
        assert_eq!(transition.mode, RuntimeAssuranceMode::TransitionToAdvanced);
        assert_eq!(transition.advanced_weight, 0.0);
        let halfway = monitor.evaluate(&observation(2.0)).unwrap();
        assert!((halfway.advanced_weight - 0.5).abs() < 1e-6);
        let complete = monitor.evaluate(&observation(2.5)).unwrap();
        assert_eq!(complete.mode, RuntimeAssuranceMode::Advanced);
        assert_eq!(complete.advanced_weight, 1.0);
    }

    #[test]
    fn unsafe_transition_returns_to_baseline() {
        let mut monitor = RuntimeAssuranceMonitor::new(RuntimeAssuranceConfig {
            recovery_dwell_s: 0.0,
            transition_duration_s: 1.0,
            ..Default::default()
        })
        .unwrap();
        let mut failed = observation(0.0);
        failed.estimator_healthy = false;
        monitor.evaluate(&failed).unwrap();
        monitor.evaluate(&observation(0.1)).unwrap();
        let mut failed_again = observation(0.2);
        failed_again.envelope_margin = 0.0;
        let decision = monitor.evaluate(&failed_again).unwrap();
        assert_eq!(decision.mode, RuntimeAssuranceMode::Baseline);
    }

    #[test]
    fn non_monotonic_time_is_rejected() {
        let mut monitor = RuntimeAssuranceMonitor::new(RuntimeAssuranceConfig::default()).unwrap();
        monitor.evaluate(&observation(1.0)).unwrap();
        assert!(matches!(
            monitor.evaluate(&observation(0.5)),
            Err(RuntimeAssuranceError::NonMonotonicTimestamp)
        ));
    }
}
