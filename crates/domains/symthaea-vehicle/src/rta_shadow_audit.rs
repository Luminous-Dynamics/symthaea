// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shadow-only audit adapter for the vehicle runtime-assurance monitor.
//!
//! The adapter records decisions and discards every proposed selected command.
//! It has no simulator, actuator or embodiment reference. Caller-supplied
//! admissibility/health/margin fields remain assertions and must be backed by
//! independently reviewed evidence in any substantive evaluation.

use serde::{Deserialize, Serialize};

use crate::runtime_assurance::{
    VehicleRtaConfig, VehicleRtaDecision, VehicleRtaError, VehicleRtaMode, VehicleRtaObservation,
    VehicleRtaReason, VehicleRuntimeAssurance,
};

pub const VEHICLE_RTA_SHADOW_AUDIT_SCHEMA: &str =
    "symthaea.vehicle.rta-shadow-audit.v0.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaShadowFrame {
    pub schema_version: String,
    pub frame_index: u64,
    pub observation_timestamp_s: Option<f64>,
    pub mode: VehicleRtaMode,
    pub reasons: Vec<VehicleRtaReason>,
    pub advanced_weight: f32,
    /// Indicates only that the monitor returned a command proposal.
    pub command_proposal_present: bool,
    /// Always true: this adapter never forwards a command to simulation or actuation.
    pub command_proposal_discarded: bool,
    pub monitor_profile_digest: String,
    pub observation_digest: Option<String>,
    pub decision_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleRtaShadowReport {
    pub schema_version: String,
    pub monitor_profile_id: String,
    pub monitor_profile_digest: String,
    pub total_frames: usize,
    pub advanced_frames: usize,
    pub baseline_frames: usize,
    pub recovery_dwell_frames: usize,
    pub transition_frames: usize,
    pub no_admissible_fallback_frames: usize,
    pub command_proposal_frames: usize,
    pub command_proposal_discarded_frames: usize,
    pub reason_counts: Vec<(VehicleRtaReason, usize)>,
    pub frames: Vec<VehicleRtaShadowFrame>,
    /// BLAKE3 over the serialized report with this field empty; not a signature.
    pub report_digest: String,
}

pub struct VehicleRtaShadowAudit {
    monitor: VehicleRuntimeAssurance,
    profile_id: String,
    profile_digest: String,
    frames: Vec<VehicleRtaShadowFrame>,
    advanced_frames: usize,
    baseline_frames: usize,
    recovery_dwell_frames: usize,
    transition_frames: usize,
    no_admissible_fallback_frames: usize,
    command_proposal_frames: usize,
    reason_counts: Vec<(VehicleRtaReason, usize)>,
}

impl VehicleRtaShadowAudit {
    pub fn new(config: VehicleRtaConfig) -> Result<Self, VehicleRtaError> {
        let monitor = VehicleRuntimeAssurance::new(config)?;
        let profile_id = monitor.profile_id().to_owned();
        let profile_digest = monitor.profile_digest().to_owned();
        Ok(Self {
            monitor,
            profile_id,
            profile_digest,
            frames: Vec::new(),
            advanced_frames: 0,
            baseline_frames: 0,
            recovery_dwell_frames: 0,
            transition_frames: 0,
            no_admissible_fallback_frames: 0,
            command_proposal_frames: 0,
            reason_counts: all_reason_variants()
                .into_iter()
                .map(|reason| (reason, 0))
                .collect(),
        })
    }

    /// Evaluate and record a monitor observation, then discard its
    /// selected-command proposal at this adapter boundary.
    ///
    /// The caller retains the observation, but no command is returned.
    pub fn observe(
        &mut self,
        observation: &VehicleRtaObservation,
    ) -> &VehicleRtaShadowFrame {
        let decision = self.monitor.evaluate(observation);
        self.record_decision(decision);
        self.frames.last().expect("shadow frame appended")
    }

    /// Explicit reset begins a new monitor epoch; existing evidence remains
    /// in the trace and frame indices continue monotonically.
    pub fn reset_monitor_epoch(&mut self) {
        self.monitor.reset();
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn profile_digest(&self) -> &str {
        &self.profile_digest
    }

    pub fn report(&self) -> VehicleRtaShadowReport {
        let mut report = VehicleRtaShadowReport {
            schema_version: VEHICLE_RTA_SHADOW_AUDIT_SCHEMA.into(),
            monitor_profile_id: self.profile_id.clone(),
            monitor_profile_digest: self.profile_digest.clone(),
            total_frames: self.frames.len(),
            advanced_frames: self.advanced_frames,
            baseline_frames: self.baseline_frames,
            recovery_dwell_frames: self.recovery_dwell_frames,
            transition_frames: self.transition_frames,
            no_admissible_fallback_frames: self.no_admissible_fallback_frames,
            command_proposal_frames: self.command_proposal_frames,
            command_proposal_discarded_frames: self.command_proposal_frames,
            reason_counts: self.reason_counts.clone(),
            frames: self.frames.clone(),
            report_digest: String::new(),
        };
        if let Ok(bytes) = serde_json::to_vec(&report) {
            report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        }
        report
    }

    fn record_decision(&mut self, decision: VehicleRtaDecision) {
        match decision.mode {
            VehicleRtaMode::Advanced => self.advanced_frames += 1,
            VehicleRtaMode::Baseline => self.baseline_frames += 1,
            VehicleRtaMode::RecoveryDwell => self.recovery_dwell_frames += 1,
            VehicleRtaMode::TransitionToAdvanced => self.transition_frames += 1,
            VehicleRtaMode::NoAdmissibleFallback => self.no_admissible_fallback_frames += 1,
        }

        for reason in &decision.reasons {
            if let Some((_, count)) = self
                .reason_counts
                .iter_mut()
                .find(|(known, _)| known == reason)
            {
                *count += 1;
            }
        }

        let command_proposal_present = decision.selected_command.is_some();
        if command_proposal_present {
            self.command_proposal_frames += 1;
        }
        let decision_digest = serde_json::to_vec(&decision)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string());

        self.frames.push(VehicleRtaShadowFrame {
            schema_version: VEHICLE_RTA_SHADOW_AUDIT_SCHEMA.into(),
            frame_index: self.frames.len() as u64,
            observation_timestamp_s: decision.observation_timestamp_s,
            mode: decision.mode,
            reasons: decision.reasons,
            advanced_weight: decision.advanced_weight,
            command_proposal_present,
            command_proposal_discarded: true,
            monitor_profile_digest: decision.profile_digest,
            observation_digest: decision.witness.observation_digest,
            decision_digest,
        });
        // The selected command is intentionally not copied or returned.
    }
}

fn all_reason_variants() -> Vec<VehicleRtaReason> {
    use VehicleRtaReason::*;
    vec![
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
    ]
}

impl VehicleRtaShadowReport {
    pub fn identity_is_valid(&self) -> bool {
        if self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.frames.len() != self.total_frames
            || self.command_proposal_discarded_frames != self.command_proposal_frames
            || self.frames.iter().enumerate().any(|(index, frame)| {
                frame.frame_index != index as u64 || !frame.command_proposal_discarded
            })
        {
            return false;
        }

        if self.reason_counts.iter().map(|(reason, _)| *reason).collect::<Vec<_>>()
            != all_reason_variants()
            || self.frames.iter().any(|frame| {
                frame.schema_version != VEHICLE_RTA_SHADOW_AUDIT_SCHEMA
                    || frame.monitor_profile_digest != self.monitor_profile_digest
            })
        {
            return false;
        }

        let mode_count = |mode| self.frames.iter().filter(|frame| frame.mode == mode).count();
        if self.advanced_frames != mode_count(VehicleRtaMode::Advanced)
            || self.baseline_frames != mode_count(VehicleRtaMode::Baseline)
            || self.recovery_dwell_frames != mode_count(VehicleRtaMode::RecoveryDwell)
            || self.transition_frames != mode_count(VehicleRtaMode::TransitionToAdvanced)
            || self.no_admissible_fallback_frames != mode_count(VehicleRtaMode::NoAdmissibleFallback)
            || self.command_proposal_frames
                != self.frames.iter().filter(|frame| frame.command_proposal_present).count()
        {
            return false;
        }

        for (reason, declared_count) in &self.reason_counts {
            let observed_count = self
                .frames
                .iter()
                .flat_map(|frame| frame.reasons.iter())
                .filter(|observed| *observed == reason)
                .count();
            if observed_count != *declared_count {
                return false;
            }
        }
        let mut report = self.clone();
        let expected = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_assurance::VehicleRtaConfig;
    use crate::types::{VehicleCommand, VehicleState};

    fn observation(timestamp_s: f64) -> VehicleRtaObservation {
        let mut state = VehicleState::cruising(10.0);
        state.timestamp = timestamp_s;
        VehicleRtaObservation {
            timestamp_s,
            state,
            advanced_command: VehicleCommand {
                steering: 0.0,
                throttle: 0.5,
                brake: 0.0,
            },
            baseline_command: VehicleCommand {
                steering: 0.0,
                throttle: 0.5,
                brake: 0.0,
            },
            advanced_controller_digest: "a".repeat(64),
            baseline_controller_digest: "b".repeat(64),
            baseline_independence_evidence_id: Some("test-fixture-only".into()),
            advanced_command_valid: true,
            baseline_profile_admissible: true,
            baseline_inputs_healthy: true,
            baseline_controller_healthy: true,
            baseline_control_path_healthy: true,
            estimator_healthy: true,
            realtime_healthy: true,
            actuator_healthy: true,
            model_applicable: true,
            envelope_margin: 1.0,
            controllability_margin: 1.0,
        }
    }

    #[test]
    fn shadow_audit_discards_monitor_command_proposal() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        let frame = audit.observe(&observation(0.0));
        assert!(frame.command_proposal_present);
        assert!(frame.command_proposal_discarded);
        assert_eq!(frame.mode, VehicleRtaMode::RecoveryDwell);
        let report = audit.report();
        assert_eq!(report.total_frames, 1);
        assert_eq!(report.command_proposal_discarded_frames, 1);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn seeded_monitor_faults_are_reported_without_forwarding_commands() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        let mut sample = observation(0.0);
        sample.state.speed = 40.0;
        let frame = audit.observe(&sample);
        assert_eq!(frame.mode, VehicleRtaMode::Baseline);
        assert!(frame.reasons.contains(&VehicleRtaReason::StateOutsideEnvelope));
        assert!(frame.command_proposal_discarded);
        let report = audit.report();
        assert_eq!(report.baseline_frames, 1);
        assert_eq!(
            report.reason_counts
                .iter()
                .find(|(reason, _)| *reason == VehicleRtaReason::StateOutsideEnvelope)
                .unwrap()
                .1,
            1
        );
    }

    #[test]
    fn unavailable_fallback_is_recorded_as_no_admissible_fallback() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        let mut sample = observation(0.0);
        sample.baseline_control_path_healthy = false;
        let frame = audit.observe(&sample);
        assert_eq!(frame.mode, VehicleRtaMode::NoAdmissibleFallback);
        assert!(!frame.command_proposal_present);
        assert!(frame.command_proposal_discarded);
        assert_eq!(audit.report().no_admissible_fallback_frames, 1);
    }

    #[test]
    fn aggregate_counts_must_match_frames_even_when_digest_is_recomputed() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        audit.observe(&observation(0.0));
        let mut report = audit.report();
        report.baseline_frames += 1;
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }

    #[test]
    fn complete_reason_vocabulary_is_required_even_when_digest_is_recomputed() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        audit.observe(&observation(0.0));
        let mut report = audit.report();
        report.reason_counts.pop();
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }

    #[test]
    fn monitor_epoch_reset_does_not_erase_existing_audit_frames() {
        let mut audit = VehicleRtaShadowAudit::new(VehicleRtaConfig::default()).unwrap();
        audit.observe(&observation(0.0));
        audit.reset_monitor_epoch();
        audit.observe(&observation(0.005));
        let report = audit.report();
        assert_eq!(report.total_frames, 2);
        assert_eq!(report.frames[0].frame_index, 0);
        assert_eq!(report.frames[1].frame_index, 1);
        assert!(report.identity_is_valid());
    }
}
