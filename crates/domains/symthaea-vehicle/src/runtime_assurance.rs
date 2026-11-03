// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simulation-only, road-vehicle runtime-assurance contract.
//!
//! The learned controller proposes a command. This monitor independently
//! checks the supplied observation and either permits the proposal, selects an
//! explicitly admissible reversionary command, or emits no command. It does not
//! implement a vehicle actuator interface, prove the baseline safe, or qualify
//! a physical vehicle.
//!
//! Default thresholds are provisional research values, not automotive limits.
//! A real profile must bind calibrated limits, validated models, configuration
//! identity, and independent evidence before any operational use.

use serde::{Deserialize, Serialize};

use crate::types::{VehicleCommand, VehicleState};

/// Versioned profile identity and provisional monitor thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaConfig {
    pub schema_version: String,
    pub profile_id: String,
    pub maximum_observation_gap_s: f64,
    pub maximum_state_age_s: f64,
    pub maximum_speed_mps: f64,
    pub maximum_abs_yaw_rate_rps: f64,
    pub maximum_abs_lateral_velocity_mps: f64,
    pub maximum_abs_tire_slip_rad: f64,
    pub minimum_envelope_margin: f64,
    pub minimum_controllability_margin: f64,
    /// Allowed absolute difference in [steering, throttle, brake].
    pub maximum_command_disagreement: [f32; 3],
    pub recovery_dwell_s: f64,
    pub transition_duration_s: f64,
}

impl Default for VehicleRtaConfig {
    fn default() -> Self {
        Self {
            schema_version: "symthaea.vehicle.rta.v0.1".into(),
            profile_id: "simulation-provisional-only".into(),
            maximum_observation_gap_s: 0.1,
            maximum_state_age_s: 0.05,
            maximum_speed_mps: 35.0,
            maximum_abs_yaw_rate_rps: 1.2,
            maximum_abs_lateral_velocity_mps: 5.0,
            maximum_abs_tire_slip_rad: 0.2,
            minimum_envelope_margin: 0.1,
            minimum_controllability_margin: 0.1,
            maximum_command_disagreement: [0.25, 0.3, 0.3],
            recovery_dwell_s: 2.0,
            transition_duration_s: 1.0,
        }
    }
}

/// Runtime-assurance operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VehicleRtaMode {
    Advanced,
    Baseline,
    RecoveryDwell,
    TransitionToAdvanced,
    /// No command is selected because no admissible fallback is available.
    NoAdmissibleFallback,
}

/// Explicit findings. Findings are not root-cause diagnoses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VehicleRtaReason {
    ObservationTimeInvalid,
    ObservationTimeNonMonotonic,
    ObservationGapExceeded,
    StateInvalid,
    StateStale,
    StateOutsideEnvelope,
    MarginInvalid,
    EstimatorUnhealthy,
    RealtimeUnhealthy,
    ActuatorUnhealthy,
    ModelNotApplicable,
    AdvancedCommandInvalid,
    BaselineCommandInvalid,
    BaselineNotAdmissible,
    BaselineInputsUnhealthy,
    BaselineControllerUnhealthy,
    BaselineControlPathUnhealthy,
    AdvancedControllerIdentityInvalid,
    BaselineControllerIdentityInvalid,
    ControllerIdentityAliased,
    BaselineIndependenceEvidenceMissing,
    EnvelopeMarginInsufficient,
    ControllabilityMarginInsufficient,
    CommandDisagreement,
}

/// Inputs required for one monitor evaluation.
///
/// baseline_profile_admissible is an explicit caller assertion. This crate
/// does not authenticate that assertion or independently qualify the baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaObservation {
    pub timestamp_s: f64,
    pub state: VehicleState,
    pub advanced_command: VehicleCommand,
    pub baseline_command: VehicleCommand,
    /// 64-hex identifier for the frozen advanced controller and configuration.
    pub advanced_controller_digest: String,
    /// Identifier for the separately implemented baseline controller and configuration.
    pub baseline_controller_digest: String,
    /// External record ID for the baseline-independence argument; not authenticated here.
    pub baseline_independence_evidence_id: Option<String>,
    pub advanced_command_valid: bool,
    pub baseline_profile_admissible: bool,
    /// Independently supplied health of the reversionary controller's inputs.
    pub baseline_inputs_healthy: bool,
    /// Health of the separately implemented reversionary controller.
    pub baseline_controller_healthy: bool,
    /// Health of the reversionary command execution path and scheduler.
    pub baseline_control_path_healthy: bool,
    pub estimator_healthy: bool,
    pub realtime_healthy: bool,
    pub actuator_healthy: bool,
    pub model_applicable: bool,
    pub envelope_margin: f64,
    pub controllability_margin: f64,
}

/// Compact witness retained with each decision for deterministic replay/audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaWitness {
    pub state_timestamp_s: Option<f64>,
    pub speed_mps: Option<f64>,
    pub lateral_velocity_mps: Option<f64>,
    pub yaw_rate_rps: Option<f64>,
    pub maximum_tire_slip_rad: Option<f64>,
    pub advanced_command: [f32; 3],
    pub baseline_command: [f32; 3],
    pub envelope_margin: Option<f64>,
    pub controllability_margin: Option<f64>,
    /// Caller-supplied controller/config identities; not authenticated by this monitor.
    pub advanced_controller_digest: String,
    pub baseline_controller_digest: String,
    pub baseline_independence_evidence_id: Option<String>,
    /// BLAKE3 over serde_json serialization of the complete input observation, when serializable.
    /// This is a replay fingerprint, not a signature or evidence-authentication mechanism.
    pub observation_digest: Option<String>,
}

/// None selected_command means this contract did not select an action. It is
/// not a validated emergency-stop action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaDecision {
    pub schema_version: String,
    pub profile_id: String,
    /// BLAKE3 over serde_json serialization of VehicleRtaConfig; not a cross-language canonical encoding.
    pub profile_digest: String,
    pub observation_timestamp_s: Option<f64>,
    pub mode: VehicleRtaMode,
    pub selected_command: Option<VehicleCommand>,
    pub advanced_weight: f32,
    pub reasons: Vec<VehicleRtaReason>,
    pub witness: VehicleRtaWitness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VehicleRtaError {
    InvalidConfiguration,
}

/// Stateful Simplex-style monitor for the simulated road-vehicle domain.
#[derive(Debug, Clone)]
pub struct VehicleRuntimeAssurance {
    config: VehicleRtaConfig,
    mode: VehicleRtaMode,
    last_timestamp_s: Option<f64>,
    safe_since_s: Option<f64>,
    transition_started_s: Option<f64>,
    profile_digest: String,
}

impl VehicleRuntimeAssurance {
    pub fn new(config: VehicleRtaConfig) -> Result<Self, VehicleRtaError> {
        if !config_is_valid(&config) {
            return Err(VehicleRtaError::InvalidConfiguration);
        }
        let profile_bytes =
            serde_json::to_vec(&config).map_err(|_| VehicleRtaError::InvalidConfiguration)?;
        let profile_digest = blake3::hash(&profile_bytes).to_hex().to_string();

        Ok(Self {
            config,
            mode: VehicleRtaMode::Baseline,
            last_timestamp_s: None,
            safe_since_s: None,
            transition_started_s: None,
            profile_digest,
        })
    }

    pub fn mode(&self) -> VehicleRtaMode {
        self.mode
    }

    pub fn profile_id(&self) -> &str {
        &self.config.profile_id
    }

    /// Integrity fingerprint for the frozen monitor configuration.
    /// This is not a canonical cross-language encoding or signed identity.
    pub fn profile_digest(&self) -> &str {
        &self.profile_digest
    }

    /// Reset temporal history into the conservative baseline state.
    /// Reset does not prove that the baseline is admissible.
    pub fn reset(&mut self) {
        self.mode = VehicleRtaMode::Baseline;
        self.last_timestamp_s = None;
        self.safe_since_s = None;
        self.transition_started_s = None;
    }

    /// Evaluate one observation and select an admissible command, if any.
    ///
    /// Invalid timestamps become findings rather than being silently repaired.
    /// Any finding selects the baseline immediately, but only when the caller
    /// has declared that baseline admissible and its command is well-formed.
    pub fn evaluate(&mut self, observation: &VehicleRtaObservation) -> VehicleRtaDecision {
        let mut reasons = Vec::with_capacity(16);
        let valid_time = observation.timestamp_s.is_finite() && observation.timestamp_s >= 0.0;

        if !valid_time {
            reasons.push(VehicleRtaReason::ObservationTimeInvalid);
        } else if self
            .last_timestamp_s
            .is_some_and(|previous| observation.timestamp_s <= previous)
        {
            reasons.push(VehicleRtaReason::ObservationTimeNonMonotonic);
        } else {
            if let Some(previous) = self.last_timestamp_s {
                if observation.timestamp_s - previous > self.config.maximum_observation_gap_s {
                    reasons.push(VehicleRtaReason::ObservationGapExceeded);
                }
            }
            self.last_timestamp_s = Some(observation.timestamp_s);
        }

        if !state_is_finite(&observation.state) {
            reasons.push(VehicleRtaReason::StateInvalid);
        } else {
            if valid_time {
                let age = observation.timestamp_s - observation.state.timestamp;
                if age < 0.0 || age > self.config.maximum_state_age_s {
                    reasons.push(VehicleRtaReason::StateStale);
                }
            }
            if !state_is_within_envelope(&self.config, &observation.state) {
                reasons.push(VehicleRtaReason::StateOutsideEnvelope);
            }
        }

        if !observation.estimator_healthy {
            reasons.push(VehicleRtaReason::EstimatorUnhealthy);
        }
        if !observation.realtime_healthy {
            reasons.push(VehicleRtaReason::RealtimeUnhealthy);
        }
        if !observation.actuator_healthy {
            reasons.push(VehicleRtaReason::ActuatorUnhealthy);
        }
        if !observation.model_applicable {
            reasons.push(VehicleRtaReason::ModelNotApplicable);
        }
        if !command_is_valid(&observation.advanced_command) || !observation.advanced_command_valid {
            reasons.push(VehicleRtaReason::AdvancedCommandInvalid);
        }
        if !command_is_valid(&observation.baseline_command) {
            reasons.push(VehicleRtaReason::BaselineCommandInvalid);
        }
        if !observation.baseline_profile_admissible {
            reasons.push(VehicleRtaReason::BaselineNotAdmissible);
        }
        if !observation.baseline_inputs_healthy {
            reasons.push(VehicleRtaReason::BaselineInputsUnhealthy);
        }
        if !observation.baseline_controller_healthy {
            reasons.push(VehicleRtaReason::BaselineControllerUnhealthy);
        }
        if !observation.baseline_control_path_healthy {
            reasons.push(VehicleRtaReason::BaselineControlPathUnhealthy);
        }
        let advanced_identity_valid = controller_digest_is_valid(&observation.advanced_controller_digest);
        let baseline_identity_valid = controller_digest_is_valid(&observation.baseline_controller_digest);
        if !advanced_identity_valid {
            reasons.push(VehicleRtaReason::AdvancedControllerIdentityInvalid);
        }
        if !baseline_identity_valid {
            reasons.push(VehicleRtaReason::BaselineControllerIdentityInvalid);
        }
        if advanced_identity_valid
            && baseline_identity_valid
            && observation.advanced_controller_digest.eq_ignore_ascii_case(&observation.baseline_controller_digest)
        {
            reasons.push(VehicleRtaReason::ControllerIdentityAliased);
        }
        let baseline_independence_recorded = observation
            .baseline_independence_evidence_id
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty());
        if !baseline_independence_recorded {
            reasons.push(VehicleRtaReason::BaselineIndependenceEvidenceMissing);
        }

        if !observation.envelope_margin.is_finite()
            || !observation.controllability_margin.is_finite()
        {
            reasons.push(VehicleRtaReason::MarginInvalid);
        } else {
            if observation.envelope_margin < self.config.minimum_envelope_margin {
                reasons.push(VehicleRtaReason::EnvelopeMarginInsufficient);
            }
            if observation.controllability_margin < self.config.minimum_controllability_margin {
                reasons.push(VehicleRtaReason::ControllabilityMarginInsufficient);
            }
        }

        if command_disagreement(
            &observation.advanced_command,
            &observation.baseline_command,
            &self.config.maximum_command_disagreement,
        ) {
            reasons.push(VehicleRtaReason::CommandDisagreement);
        }

        let witness = witness_for(observation);
        let timestamp = valid_time.then_some(observation.timestamp_s);
        let fallback_admissible = observation.baseline_profile_admissible
            && observation.baseline_inputs_healthy
            && observation.baseline_controller_healthy
            && observation.baseline_control_path_healthy
            && baseline_identity_valid
            && !(advanced_identity_valid
                && observation.advanced_controller_digest.eq_ignore_ascii_case(&observation.baseline_controller_digest))
            && baseline_independence_recorded
            && observation.realtime_healthy
            && observation.actuator_healthy
            && command_is_valid(&observation.baseline_command);

        if !fallback_admissible {
            self.mode = VehicleRtaMode::NoAdmissibleFallback;
            self.safe_since_s = None;
            self.transition_started_s = None;
            return self.decision(timestamp, None, 0.0, reasons, witness);
        }

        if !reasons.is_empty() {
            self.mode = VehicleRtaMode::Baseline;
            self.safe_since_s = None;
            self.transition_started_s = None;
            return self.decision(
                timestamp,
                Some(observation.baseline_command),
                0.0,
                reasons,
                witness,
            );
        }

        match self.mode {
            VehicleRtaMode::Advanced => self.decision(
                timestamp,
                Some(observation.advanced_command),
                1.0,
                reasons,
                witness,
            ),
            VehicleRtaMode::Baseline
            | VehicleRtaMode::RecoveryDwell
            | VehicleRtaMode::NoAdmissibleFallback => {
                let now = observation.timestamp_s;
                let safe_since = *self.safe_since_s.get_or_insert(now);
                if now - safe_since < self.config.recovery_dwell_s {
                    self.mode = VehicleRtaMode::RecoveryDwell;
                    self.decision(
                        timestamp,
                        Some(observation.baseline_command),
                        0.0,
                        reasons,
                        witness,
                    )
                } else {
                    self.mode = VehicleRtaMode::TransitionToAdvanced;
                    let started = *self.transition_started_s.get_or_insert(now);
                    self.transition_decision(observation, started, reasons, witness)
                }
            }
            VehicleRtaMode::TransitionToAdvanced => {
                let started = self.transition_started_s.unwrap_or(observation.timestamp_s);
                self.transition_decision(observation, started, reasons, witness)
            }
        }
    }

    fn transition_decision(
        &mut self,
        observation: &VehicleRtaObservation,
        started_s: f64,
        reasons: Vec<VehicleRtaReason>,
        witness: VehicleRtaWitness,
    ) -> VehicleRtaDecision {
        let weight = if self.config.transition_duration_s == 0.0 {
            1.0
        } else {
            ((observation.timestamp_s - started_s) / self.config.transition_duration_s)
                .clamp(0.0, 1.0) as f32
        };
        let selected = blend_commands(
            observation.baseline_command,
            observation.advanced_command,
            weight,
        );
        if weight >= 1.0 {
            self.mode = VehicleRtaMode::Advanced;
            self.safe_since_s = None;
            self.transition_started_s = None;
        }
        self.decision(
            Some(observation.timestamp_s),
            Some(selected),
            weight,
            reasons,
            witness,
        )
    }

    fn decision(
        &self,
        observation_timestamp_s: Option<f64>,
        selected_command: Option<VehicleCommand>,
        advanced_weight: f32,
        reasons: Vec<VehicleRtaReason>,
        witness: VehicleRtaWitness,
    ) -> VehicleRtaDecision {
        VehicleRtaDecision {
            schema_version: self.config.schema_version.clone(),
            profile_id: self.config.profile_id.clone(),
            profile_digest: self.profile_digest.clone(),
            observation_timestamp_s,
            mode: self.mode,
            selected_command,
            advanced_weight,
            reasons,
            witness,
        }
    }
}

fn config_is_valid(config: &VehicleRtaConfig) -> bool {
    !config.schema_version.trim().is_empty()
        && !config.profile_id.trim().is_empty()
        && config.maximum_observation_gap_s.is_finite()
        && config.maximum_observation_gap_s > 0.0
        && config.maximum_state_age_s.is_finite()
        && config.maximum_state_age_s >= 0.0
        && config.maximum_speed_mps.is_finite()
        && config.maximum_speed_mps > 0.0
        && config.maximum_abs_yaw_rate_rps.is_finite()
        && config.maximum_abs_yaw_rate_rps > 0.0
        && config.maximum_abs_lateral_velocity_mps.is_finite()
        && config.maximum_abs_lateral_velocity_mps > 0.0
        && config.maximum_abs_tire_slip_rad.is_finite()
        && config.maximum_abs_tire_slip_rad > 0.0
        && config.minimum_envelope_margin.is_finite()
        && config.minimum_envelope_margin >= 0.0
        && config.minimum_controllability_margin.is_finite()
        && config.minimum_controllability_margin >= 0.0
        && config
            .maximum_command_disagreement
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && config.recovery_dwell_s.is_finite()
        && config.recovery_dwell_s >= 0.0
        && config.transition_duration_s.is_finite()
        && config.transition_duration_s >= 0.0
}

fn command_is_valid(command: &VehicleCommand) -> bool {
    command.steering.is_finite()
        && (-1.0..=1.0).contains(&command.steering)
        && command.throttle.is_finite()
        && (0.0..=1.0).contains(&command.throttle)
        && command.brake.is_finite()
        && (0.0..=1.0).contains(&command.brake)
}

fn state_is_finite(state: &VehicleState) -> bool {
    let values = [
        state.speed,
        state.lateral_velocity,
        state.yaw_rate,
        state.heading,
        state.position_x,
        state.position_y,
        state.steering_angle,
        state.throttle_position,
        state.brake_pressure,
        state.longitudinal_accel,
        state.lateral_accel,
        state.tire_slip_front,
        state.tire_slip_rear,
        state.pitch,
        state.lateral_offset,
        state.curvature_ahead,
        state.nearest_peer_distance,
        state.peer_relative_speed,
        state.mesh_brake_density,
        state.mesh_confidence,
        state.timestamp,
    ];
    values.iter().all(|value| value.is_finite())
}

fn state_is_within_envelope(config: &VehicleRtaConfig, state: &VehicleState) -> bool {
    state.speed >= 0.0
        && state.speed <= config.maximum_speed_mps
        && state.yaw_rate.abs() <= config.maximum_abs_yaw_rate_rps
        && state.lateral_velocity.abs() <= config.maximum_abs_lateral_velocity_mps
        && state.tire_slip_front.abs() <= config.maximum_abs_tire_slip_rad
        && state.tire_slip_rear.abs() <= config.maximum_abs_tire_slip_rad
        && (0.0..=1.0).contains(&state.throttle_position)
        && (0.0..=1.0).contains(&state.brake_pressure)
        && state.nearest_peer_distance >= 0.0
        && (0.0..=1.0).contains(&state.mesh_brake_density)
        && (0.0..=1.0).contains(&state.mesh_confidence)
}

fn command_disagreement(
    advanced: &VehicleCommand,
    baseline: &VehicleCommand,
    limits: &[f32; 3],
) -> bool {
    (advanced.steering - baseline.steering).abs() > limits[0]
        || (advanced.throttle - baseline.throttle).abs() > limits[1]
        || (advanced.brake - baseline.brake).abs() > limits[2]
}

fn blend_commands(
    baseline: VehicleCommand,
    advanced: VehicleCommand,
    advanced_weight: f32,
) -> VehicleCommand {
    let weight = advanced_weight.clamp(0.0, 1.0);
    let inverse = 1.0 - weight;
    VehicleCommand {
        steering: baseline.steering * inverse + advanced.steering * weight,
        throttle: baseline.throttle * inverse + advanced.throttle * weight,
        brake: baseline.brake * inverse + advanced.brake * weight,
    }
}

fn witness_for(observation: &VehicleRtaObservation) -> VehicleRtaWitness {
    let state = &observation.state;
    let finite = state_is_finite(state);
    VehicleRtaWitness {
        state_timestamp_s: finite.then_some(state.timestamp),
        speed_mps: finite.then_some(state.speed),
        lateral_velocity_mps: finite.then_some(state.lateral_velocity),
        yaw_rate_rps: finite.then_some(state.yaw_rate),
        maximum_tire_slip_rad: finite.then_some(state.tire_slip_front.abs().max(state.tire_slip_rear.abs())),
        advanced_command: [
            observation.advanced_command.steering,
            observation.advanced_command.throttle,
            observation.advanced_command.brake,
        ],
        baseline_command: [
            observation.baseline_command.steering,
            observation.baseline_command.throttle,
            observation.baseline_command.brake,
        ],
        envelope_margin: observation.envelope_margin.is_finite().then_some(observation.envelope_margin),
        controllability_margin: observation.controllability_margin.is_finite().then_some(observation.controllability_margin),
        advanced_controller_digest: observation.advanced_controller_digest.clone(),
        baseline_controller_digest: observation.baseline_controller_digest.clone(),
        baseline_independence_evidence_id: observation.baseline_independence_evidence_id.clone(),
        observation_digest: observation_is_fingerprintable(observation)
            .then(|| serde_json::to_vec(observation))
            .and_then(|serialized| serialized.ok())
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string()),
    }
}

fn observation_is_fingerprintable(observation: &VehicleRtaObservation) -> bool {
    let advanced = &observation.advanced_command;
    let baseline = &observation.baseline_command;
    observation.timestamp_s.is_finite()
        && state_is_finite(&observation.state)
        && advanced.steering.is_finite()
        && advanced.throttle.is_finite()
        && advanced.brake.is_finite()
        && baseline.steering.is_finite()
        && baseline.throttle.is_finite()
        && baseline.brake.is_finite()
        && observation.envelope_margin.is_finite()
        && observation.controllability_margin.is_finite()
}

fn controller_digest_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(timestamp_s: f64) -> VehicleRtaObservation {
        let mut state = VehicleState::cruising(10.0);
        state.timestamp = timestamp_s;
        VehicleRtaObservation {
            timestamp_s,
            state,
            advanced_command: VehicleCommand { steering: 0.05, throttle: 0.2, brake: 0.0 },
            baseline_command: VehicleCommand { steering: 0.05, throttle: 0.2, brake: 0.0 },
            advanced_controller_digest: "a".repeat(64),
            baseline_controller_digest: "b".repeat(64),
            baseline_independence_evidence_id: Some("test-fixture-independent-baseline-record".into()),
            advanced_command_valid: true,
            baseline_profile_admissible: true,
            baseline_inputs_healthy: true,
            baseline_controller_healthy: true,
            baseline_control_path_healthy: true,
            estimator_healthy: true,
            realtime_healthy: true,
            actuator_healthy: true,
            model_applicable: true,
            envelope_margin: 0.5,
            controllability_margin: 0.5,
        }
    }

    fn monitor() -> VehicleRuntimeAssurance {
        let mut config = VehicleRtaConfig::default();
        config.recovery_dwell_s = 1.0;
        config.transition_duration_s = 1.0;
        config.maximum_observation_gap_s = 2.0;
        VehicleRuntimeAssurance::new(config).expect("default profile should validate")
    }

    #[test]
    fn requires_distinct_controller_identities_and_an_independence_record() {
        let mut monitor = monitor();
        let mut missing = observation(0.0);
        missing.baseline_independence_evidence_id = None;
        let decision = monitor.evaluate(&missing);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineIndependenceEvidenceMissing));

        let mut monitor = monitor();
        let mut aliased = observation(0.0);
        aliased.baseline_controller_digest = aliased.advanced_controller_digest.clone();
        let decision = monitor.evaluate(&aliased);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.reasons.contains(&VehicleRtaReason::ControllerIdentityAliased));

        let mut monitor = monitor();
        let mut malformed = observation(0.0);
        malformed.baseline_controller_digest = "not-a-digest".into();
        let decision = monitor.evaluate(&malformed);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineControllerIdentityInvalid));
    }

    #[test]
    fn invalid_advanced_identity_falls_back_to_valid_baseline() {
        let mut monitor = monitor();
        let mut invalid_advanced = observation(0.0);
        invalid_advanced.advanced_controller_digest = "missing-or-malformed".into();
        let decision = monitor.evaluate(&invalid_advanced);
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
        assert!(decision.selected_command.is_some());
        assert!(decision.reasons.contains(&VehicleRtaReason::AdvancedControllerIdentityInvalid));
    }

    #[test]
    fn decision_witness_binds_controller_ids_and_serializable_observation() {
        let mut monitor = monitor();
        let decision = monitor.evaluate(&observation(0.0));
        assert_eq!(decision.witness.advanced_controller_digest, "a".repeat(64));
        assert_eq!(decision.witness.baseline_controller_digest, "b".repeat(64));
        assert!(decision.witness.observation_digest.is_some());

        let mut invalid = observation(0.1);
        invalid.advanced_command.throttle = f32::NAN;
        let decision = monitor.evaluate(&invalid);
        assert!(decision.witness.observation_digest.is_none());
    }

    #[test]
    fn startup_requires_dwell_before_advanced_command() {
        let mut monitor = monitor();
        let first = monitor.evaluate(&observation(0.0));
        assert_eq!(first.mode, VehicleRtaMode::RecoveryDwell);
        assert_eq!(first.advanced_weight, 0.0);
        assert!(first.selected_command.is_some());

        let dwell_complete = monitor.evaluate(&observation(1.1));
        assert_eq!(dwell_complete.mode, VehicleRtaMode::TransitionToAdvanced);
        assert_eq!(dwell_complete.advanced_weight, 0.0);

        let transition_complete = monitor.evaluate(&observation(2.2));
        assert_eq!(transition_complete.mode, VehicleRtaMode::Advanced);
        assert_eq!(transition_complete.advanced_weight, 1.0);
        assert!(transition_complete.reasons.is_empty());
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_advanced_commands() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.advanced_command.steering = f32::NAN;
        let decision = monitor.evaluate(&bad);
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
        assert!(decision.reasons.contains(&VehicleRtaReason::AdvancedCommandInvalid));
        assert_eq!(decision.selected_command.expect("baseline").throttle, 0.2);

        let mut bad = observation(0.1);
        bad.advanced_command.throttle = 1.1;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::AdvancedCommandInvalid));
    }

    #[test]
    fn advanced_estimator_fault_uses_independently_healthy_baseline() {
        let mut monitor = monitor();
        let mut observation = observation(0.0);
        observation.estimator_healthy = false;
        observation.model_applicable = false;
        let decision = monitor.evaluate(&observation);
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
        assert!(decision.selected_command.is_some());
        assert!(decision.reasons.contains(&VehicleRtaReason::EstimatorUnhealthy));
        assert!(decision.reasons.contains(&VehicleRtaReason::ModelNotApplicable));
    }

    #[test]
    fn refuses_to_select_an_inadmissible_or_invalid_baseline() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.baseline_profile_admissible = false;
        let decision = monitor.evaluate(&bad);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineNotAdmissible));

        let mut bad = observation(0.1);
        bad.baseline_command.brake = f32::INFINITY;
        let decision = monitor.evaluate(&bad);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineCommandInvalid));
    }

    #[test]
    fn refuses_fallback_when_its_independent_path_is_unhealthy() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.baseline_inputs_healthy = false;
        let decision = monitor.evaluate(&bad);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineInputsUnhealthy));

        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.baseline_controller_healthy = false;
        let decision = monitor.evaluate(&bad);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineControllerUnhealthy));

        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.baseline_control_path_healthy = false;
        let decision = monitor.evaluate(&bad);
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::BaselineControlPathUnhealthy));
    }

    #[test]
    fn detects_invalid_and_non_monotonic_observation_time() {
        let mut monitor = monitor();
        let _ = monitor.evaluate(&observation(1.0));
        let decision = monitor.evaluate(&observation(0.5));
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
        assert!(decision.reasons.contains(&VehicleRtaReason::ObservationTimeNonMonotonic));

        let mut bad = observation(1.11);
        bad.timestamp_s = f64::NAN;
        let decision = monitor.evaluate(&bad);
        assert!(decision.observation_timestamp_s.is_none());
        assert!(decision.reasons.contains(&VehicleRtaReason::ObservationTimeInvalid));
    }

    #[test]
    fn detects_observation_gap_and_stale_state() {
        let config = VehicleRtaConfig::default();
        let mut monitor = VehicleRuntimeAssurance::new(config).expect("default profile");
        let _ = monitor.evaluate(&observation(0.0));
        let decision = monitor.evaluate(&observation(0.2));
        assert!(decision.reasons.contains(&VehicleRtaReason::ObservationGapExceeded));

        let mut stale = observation(0.3);
        stale.state.timestamp = 0.0;
        let decision = monitor.evaluate(&stale);
        assert!(decision.reasons.contains(&VehicleRtaReason::StateStale));
    }

    #[test]
    fn checks_all_state_fields_and_configured_envelope() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.state.brake_pressure = f64::NAN;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::StateInvalid));

        let mut bad = observation(0.1);
        bad.state.speed = 50.0;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::StateOutsideEnvelope));
    }

    #[test]
    fn gates_estimator_realtime_actuator_and_model_applicability() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.estimator_healthy = false;
        bad.realtime_healthy = false;
        bad.actuator_healthy = false;
        bad.model_applicable = false;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::EstimatorUnhealthy));
        assert!(decision.reasons.contains(&VehicleRtaReason::RealtimeUnhealthy));
        assert!(decision.reasons.contains(&VehicleRtaReason::ActuatorUnhealthy));
        assert!(decision.reasons.contains(&VehicleRtaReason::ModelNotApplicable));
        // A late monitor / unavailable actuator path cannot deliver a fallback.
        assert_eq!(decision.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(decision.selected_command.is_none());
    }

    #[test]
    fn gates_margins_and_command_disagreement() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.envelope_margin = 0.0;
        bad.controllability_margin = 0.0;
        bad.advanced_command.steering = 0.8;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::EnvelopeMarginInsufficient));
        assert!(decision.reasons.contains(&VehicleRtaReason::ControllabilityMarginInsufficient));
        assert!(decision.reasons.contains(&VehicleRtaReason::CommandDisagreement));
    }

    #[test]
    fn requires_clean_dwell_and_bounded_transition_before_reentry() {
        let mut monitor = monitor();
        let mut fault = observation(0.0);
        fault.estimator_healthy = false;
        let _ = monitor.evaluate(&fault);

        let first_clean = monitor.evaluate(&observation(0.1));
        assert_eq!(first_clean.mode, VehicleRtaMode::RecoveryDwell);
        assert_eq!(first_clean.advanced_weight, 0.0);

        let dwell_complete = monitor.evaluate(&observation(1.1));
        assert_eq!(dwell_complete.mode, VehicleRtaMode::TransitionToAdvanced);
        assert_eq!(dwell_complete.advanced_weight, 0.0);

        let halfway = monitor.evaluate(&observation(1.6));
        assert_eq!(halfway.mode, VehicleRtaMode::TransitionToAdvanced);
        assert!((halfway.advanced_weight - 0.5).abs() < 1e-6);

        let complete = monitor.evaluate(&observation(2.1));
        assert_eq!(complete.mode, VehicleRtaMode::Advanced);
        assert_eq!(complete.advanced_weight, 1.0);
    }

    #[test]
    fn fault_during_transition_returns_to_baseline() {
        let mut monitor = monitor();
        let mut fault = observation(0.0);
        fault.actuator_healthy = false;
        let _ = monitor.evaluate(&fault);
        let _ = monitor.evaluate(&observation(0.1));
        let _ = monitor.evaluate(&observation(1.1));
        let mut interrupted = observation(1.2);
        interrupted.model_applicable = false;
        let decision = monitor.evaluate(&interrupted);
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
        assert_eq!(decision.advanced_weight, 0.0);
    }

    #[test]
    fn monitor_decisions_bind_deterministic_configuration_digest() {
        let config = VehicleRtaConfig::default();
        let mut first = VehicleRuntimeAssurance::new(config.clone()).expect("valid profile");
        let mut second = VehicleRuntimeAssurance::new(config.clone()).expect("same profile");
        let first_decision = first.evaluate(&observation(0.0));
        let second_decision = second.evaluate(&observation(0.0));
        assert_eq!(first_decision.profile_digest, second_decision.profile_digest);
        assert_eq!(first_decision.profile_id, second_decision.profile_id);

        let mut changed_config = config;
        changed_config.maximum_speed_mps += 1.0;
        let mut changed = VehicleRuntimeAssurance::new(changed_config).expect("changed profile");
        let changed_decision = changed.evaluate(&observation(0.0));
        assert_ne!(first_decision.profile_digest, changed_decision.profile_digest);
    }

    #[test]
    fn rejects_non_finite_safety_margins_as_unknown() {
        let mut monitor = monitor();
        let mut bad = observation(0.0);
        bad.envelope_margin = f64::NAN;
        let decision = monitor.evaluate(&bad);
        assert!(decision.reasons.contains(&VehicleRtaReason::MarginInvalid));
        assert_eq!(decision.mode, VehicleRtaMode::Baseline);
    }

    #[test]
    fn rejects_invalid_configuration() {
        let mut config = VehicleRtaConfig::default();
        config.maximum_observation_gap_s = f64::NAN;
        assert_eq!(
            VehicleRuntimeAssurance::new(config).err(),
            Some(VehicleRtaError::InvalidConfiguration)
        );
    }

    #[test]
    fn reset_returns_monitor_to_baseline_not_advanced() {
        let mut monitor = monitor();
        monitor.reset();
        assert_eq!(monitor.mode(), VehicleRtaMode::Baseline);
    }
}
