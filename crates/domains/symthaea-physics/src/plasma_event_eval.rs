//! Leakage-resistant causal plasma disruption alarm policy and event metrics.
//!
//! The online policy accepts one timestamped observation at a time. It has no
//! event-label, disruption-time, future-window, or full-shot-mean input.
//! Event references are used only by the separate post-hoc scorer.
//!
//! This module does not calibrate physical probabilities, select thresholds
//! from a test set, or authorize machine actions.

use crate::plasma_observability::{
    BlockedChannel, DecisionChannelRequirement, DecisionInputGate, PlasmaObservationFrame,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlasmaEventEvalError {
    pub message: String,
}

impl PlasmaEventEvalError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for PlasmaEventEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PlasmaEventEvalError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlarmEvidenceQuality {
    DirectMeasured,
    Synthetic,
    Unavailable,
    Invalid,
    Delayed,
    Degraded,
}

#[derive(Debug, Clone, PartialEq)]
struct AlarmObservation {
    pub shot_id: String,
    pub window_end_ns: u64,
    /// A predeclared risk score, not assumed to be a calibrated probability.
    pub risk_score: Option<f64>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub risk_evidence_sha256: Option<String>,
    pub evidence_quality: AlarmEvidenceQuality,
}

/// Exact model/split/calibration/uncertainty tuple authorized by one frozen
/// alarm policy. The policy digest commits to every field; score evidence
/// carrying a different but syntactically valid identity is rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskModelBinding {
    pub model_id: String,
    pub model_revision: String,
    pub model_artifact_sha256: String,
    pub training_split_sha256: String,
    pub calibration_id: String,
    pub calibration_revision: String,
    pub calibration_receipt_sha256: String,
    pub uncertainty_method_id: String,
    pub nominal_coverage: f64,
}

impl RiskModelBinding {
    fn validate(&self) -> Result<(), PlasmaEventEvalError> {
        for (value, field) in [
            (self.model_id.as_str(), "risk model_id"),
            (self.model_revision.as_str(), "risk model_revision"),
            (self.calibration_id.as_str(), "risk calibration_id"),
            (self.calibration_revision.as_str(), "risk calibration_revision"),
            (self.uncertainty_method_id.as_str(), "risk uncertainty_method_id"),
        ] {
            if value.trim().is_empty() {
                return Err(PlasmaEventEvalError::new(format!("{field} must be non-empty")));
            }
        }
        for (digest, field) in [
            (self.model_artifact_sha256.as_str(), "risk model_artifact_sha256"),
            (self.training_split_sha256.as_str(), "risk training_split_sha256"),
            (self.calibration_receipt_sha256.as_str(), "risk calibration_receipt_sha256"),
        ] {
            if !valid_sha256(digest) {
                return Err(PlasmaEventEvalError::new(format!(
                    "{field} must be 64 lowercase hexadecimal characters"
                )));
            }
        }
        if !self.nominal_coverage.is_finite()
            || self.nominal_coverage <= 0.0
            || self.nominal_coverage >= 1.0
        {
            return Err(PlasmaEventEvalError::new(
                "risk model nominal_coverage must be finite and strictly between zero and one",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlarmPolicyConfig {
    pub policy_id: String,
    /// Identity of the numeric scale shared by thresholds and model score evidence.
    pub score_scale_id: String,
    /// SHA-256 content identity for the frozen validation/calibration receipt.
    pub validation_receipt_sha256: String,
    /// Exact model artifact, training split, calibration and uncertainty method
    /// that the validation receipt authorized for this policy.
    pub risk_model: RiskModelBinding,
    pub trigger_threshold: f64,
    pub clear_threshold: f64,
    pub trigger_persistence_windows: usize,
    pub clear_persistence_windows: usize,
    /// Minimum lead time for an alarm to count as a valid disruption warning.
    pub required_warning_ns: u64,
    /// Maximum lead time; alarms earlier than this belong to an excessive-early-warning region.
    /// Freeze this value on validation, never tune it on the test cohort.
    pub max_warning_ns: u64,
}

impl AlarmPolicyConfig {
    pub fn validate(&self) -> Result<(), PlasmaEventEvalError> {
        if self.policy_id.trim().is_empty() {
            return Err(PlasmaEventEvalError::new("policy_id must be non-empty"));
        }
        if self.score_scale_id.trim().is_empty() {
            return Err(PlasmaEventEvalError::new("score_scale_id must be non-empty"));
        }
        if self.validation_receipt_sha256.len() != 64
            || !self.validation_receipt_sha256.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
        {
            return Err(PlasmaEventEvalError::new(
                "validation receipt must be 64 lowercase hexadecimal characters",
            ));
        }
        self.risk_model.validate()?;
        if self.required_warning_ns == 0 {
            return Err(PlasmaEventEvalError::new("required_warning_ns must be positive"));
        }
        if self.max_warning_ns < self.required_warning_ns {
            return Err(PlasmaEventEvalError::new(
                "max_warning_ns must be greater than or equal to required_warning_ns",
            ));
        }
        if !self.trigger_threshold.is_finite() || !self.clear_threshold.is_finite() {
            return Err(PlasmaEventEvalError::new(
                "alarm thresholds must be finite",
            ));
        }
        if self.clear_threshold > self.trigger_threshold {
            return Err(PlasmaEventEvalError::new(
                "clear_threshold must not exceed trigger_threshold",
            ));
        }
        if self.trigger_persistence_windows == 0 || self.clear_persistence_windows == 0 {
            return Err(PlasmaEventEvalError::new(
                "alarm persistence window counts must be positive",
            ));
        }
        Ok(())
    }

    /// SHA-256 over length-prefixed fields. Length-prefixing prevents
    /// delimiter characters inside caller-provided identifiers from creating
    /// ambiguous policy encodings. Floating-point thresholds use their exact
    /// IEEE-754 bit patterns.
    pub fn policy_config_sha256(&self) -> String {
        let mut canonical = Vec::new();
        let mut append = |field: &[u8]| {
            canonical.extend_from_slice(&(field.len() as u64).to_be_bytes());
            canonical.extend_from_slice(field);
        };
        append(b"luminous.plasma-alarm-policy.v0.5");
        append(self.policy_id.as_bytes());
        append(self.score_scale_id.as_bytes());
        append(self.validation_receipt_sha256.as_bytes());
        append(self.risk_model.model_id.as_bytes());
        append(self.risk_model.model_revision.as_bytes());
        append(self.risk_model.model_artifact_sha256.as_bytes());
        append(self.risk_model.training_split_sha256.as_bytes());
        append(self.risk_model.calibration_id.as_bytes());
        append(self.risk_model.calibration_revision.as_bytes());
        append(self.risk_model.calibration_receipt_sha256.as_bytes());
        append(self.risk_model.uncertainty_method_id.as_bytes());
        append(format!("{:016x}", self.risk_model.nominal_coverage.to_bits()).as_bytes());
        append(format!("{:016x}", self.trigger_threshold.to_bits()).as_bytes());
        append(format!("{:016x}", self.clear_threshold.to_bits()).as_bytes());
        append(self.trigger_persistence_windows.to_string().as_bytes());
        append(self.clear_persistence_windows.to_string().as_bytes());
        append(self.required_warning_ns.to_string().as_bytes());
        append(self.max_warning_ns.to_string().as_bytes());
        sha256_hex(&canonical)
    }
}


fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

/// Compute the deterministic content identity used to bind risk evidence
/// to one exact observation frame.
pub fn observation_frame_sha256(
    frame: &PlasmaObservationFrame,
) -> Result<String, PlasmaEventEvalError> {
    let canonical = serde_json::to_vec(frame).map_err(|error| {
        PlasmaEventEvalError::new(format!("observation-frame serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&canonical))
}

/// Model-bound score evidence for one exact observation frame and forecast horizon.
///
/// Digests are content identities, not signatures or proof that the model,
/// training split, or calibration process is trustworthy. Those claims require
/// independently verifiable registries and signed receipts outside this module.
pub const PLASMA_RISK_SCORE_EVIDENCE_SCHEMA: &str =
    "luminous.plasma-risk-score-evidence.v0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskScoreEvidence {
    pub schema_version: String,
    pub shot_id: String,
    pub frame_id: String,
    pub frame_time_ns: u64,
    /// SHA-256 of the exact serialized PlasmaObservationFrame.
    pub input_frame_sha256: String,
    pub horizon_ns: u64,
    pub target_time_ns: u64,
    pub model_id: String,
    pub model_revision: String,
    /// Numeric scale identifier; must equal the frozen policy's score_scale_id.
    pub score_scale_id: String,
    pub model_artifact_sha256: String,
    pub training_split_sha256: String,
    pub calibration_id: String,
    pub calibration_revision: String,
    pub calibration_receipt_sha256: String,
    pub uncertainty_method_id: String,
    /// Nominal coverage of the reported interval, not a guarantee of empirical coverage.
    pub nominal_coverage: f64,
    /// Point score and uncertainty bounds share the frozen policy threshold scale.
    pub estimate: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

impl RiskScoreEvidence {
    pub fn validate_for_frame(
        &self,
        frame: &PlasmaObservationFrame,
        policy: &AlarmPolicyConfig,
    ) -> Result<(), PlasmaEventEvalError> {
        if self.schema_version != PLASMA_RISK_SCORE_EVIDENCE_SCHEMA {
            return Err(PlasmaEventEvalError::new(
                "unsupported plasma risk score evidence schema",
            ));
        }
        for (value, field) in [
            (self.shot_id.as_str(), "risk evidence shot_id"),
            (self.frame_id.as_str(), "risk evidence frame_id"),
            (self.model_id.as_str(), "model_id"),
            (self.model_revision.as_str(), "model_revision"),
            (self.score_scale_id.as_str(), "score_scale_id"),
            (self.calibration_id.as_str(), "calibration_id"),
            (self.calibration_revision.as_str(), "calibration_revision"),
            (self.uncertainty_method_id.as_str(), "uncertainty_method_id"),
        ] {
            if value.trim().is_empty() {
                return Err(PlasmaEventEvalError::new(format!(
                    "{field} must be non-empty"
                )));
            }
        }

        for (digest, field) in [
            (self.input_frame_sha256.as_str(), "input_frame_sha256"),
            (self.model_artifact_sha256.as_str(), "model_artifact_sha256"),
            (self.training_split_sha256.as_str(), "training_split_sha256"),
            (
                self.calibration_receipt_sha256.as_str(),
                "calibration_receipt_sha256",
            ),
        ] {
            if !valid_sha256(digest) {
                return Err(PlasmaEventEvalError::new(format!(
                    "{field} must be 64 lowercase hexadecimal characters"
                )));
            }
        }

        if !self.nominal_coverage.is_finite()
            || self.nominal_coverage <= 0.0
            || self.nominal_coverage >= 1.0
        {
            return Err(PlasmaEventEvalError::new(
                "nominal_coverage must be finite and strictly between zero and one",
            ));
        }
        if !self.estimate.is_finite()
            || !self.lower_bound.is_finite()
            || !self.upper_bound.is_finite()
            || self.lower_bound > self.estimate
            || self.estimate > self.upper_bound
        {
            return Err(PlasmaEventEvalError::new(
                "risk estimate and interval bounds must be finite and ordered",
            ));
        }
        if self.score_scale_id != policy.score_scale_id {
            return Err(PlasmaEventEvalError::new(
                "risk evidence score_scale_id does not match the frozen alarm policy",
            ));
        }
        let binding = &policy.risk_model;
        if self.model_id != binding.model_id
            || self.model_revision != binding.model_revision
            || self.model_artifact_sha256 != binding.model_artifact_sha256
            || self.training_split_sha256 != binding.training_split_sha256
            || self.calibration_id != binding.calibration_id
            || self.calibration_revision != binding.calibration_revision
            || self.calibration_receipt_sha256 != binding.calibration_receipt_sha256
            || self.uncertainty_method_id != binding.uncertainty_method_id
            || self.nominal_coverage.to_bits() != binding.nominal_coverage.to_bits()
        {
            return Err(PlasmaEventEvalError::new(
                "risk evidence model, training split, calibration or uncertainty binding does not match the frozen alarm policy",
            ));
        }
        if self.horizon_ns == 0 {
            return Err(PlasmaEventEvalError::new(
                "risk evidence horizon_ns must be positive",
            ));
        }
        let expected_target = frame
            .frame_time_ns
            .checked_add(self.horizon_ns)
            .ok_or_else(|| PlasmaEventEvalError::new("risk target timestamp overflow"))?;
        if self.shot_id != frame.shot_id
            || self.frame_id != frame.frame_id
            || self.frame_time_ns != frame.frame_time_ns
            || self.target_time_ns != expected_target
        {
            return Err(PlasmaEventEvalError::new(
                "risk evidence shot/frame/time/horizon does not match the input frame",
            ));
        }
        let actual_frame_digest = observation_frame_sha256(frame)?;
        if self.input_frame_sha256 != actual_frame_digest {
            return Err(PlasmaEventEvalError::new(
                "risk evidence input_frame_sha256 does not match the exact observation frame",
            ));
        }
        Ok(())
    }

    /// Identity of this exact model/split/calibration/frame/horizon score record.
    pub fn evidence_sha256(&self) -> Result<String, PlasmaEventEvalError> {
        let canonical = serde_json::to_vec(self).map_err(|error| {
            PlasmaEventEvalError::new(format!("risk evidence serialization failed: {error}"))
        })?;
        Ok(sha256_hex(&canonical))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmAbstentionReason {
    EvidenceGateBlocked(Vec<BlockedChannel>),
}



#[derive(Debug, Clone, PartialEq)]
pub enum AlarmPolicyUpdate {
    BelowThreshold {
        at_ns: u64,
        risk_evidence_sha256: String,
    },
    PendingTrigger {
        at_ns: u64,
        consecutive_windows: usize,
        risk_evidence_sha256: String,
    },
    Raised {
        at_ns: u64,
        risk_evidence_sha256: String,
    },
    Active {
        at_ns: u64,
        risk_evidence_sha256: String,
    },
    PendingClear {
        at_ns: u64,
        consecutive_windows: usize,
        risk_evidence_sha256: String,
    },
    Cleared {
        at_ns: u64,
        risk_evidence_sha256: String,
    },
    Uncertain {
        at_ns: u64,
        estimate: f64,
        lower_bound: f64,
        upper_bound: f64,
        risk_evidence_sha256: String,
        alert_remains_active: bool,
    },
    Abstained {
        at_ns: u64,
        reason: AlarmAbstentionReason,
        alert_remains_active: bool,
    },
}

/// Online hysteresis/persistence state machine.
///
/// Thresholds and persistence counts must be selected before test evaluation.
/// This type has no event-time, label, full-shot, or future-window input.
#[derive(Debug, Clone)]
pub struct OnlinePlasmaAlarmPolicy {
    config: AlarmPolicyConfig,
    shot_id: Option<String>,
    last_window_end_ns: Option<u64>,
    high_run: usize,
    low_run: usize,
    active: bool,
}

impl OnlinePlasmaAlarmPolicy {
    pub fn new(config: AlarmPolicyConfig) -> Result<Self, PlasmaEventEvalError> {
        config.validate()?;
        Ok(Self {
            config,
            shot_id: None,
            last_window_end_ns: None,
            high_run: 0,
            low_run: 0,
            active: false,
        })
    }

    /// Explicitly starts a new discharge and clears prior alarm state.
    pub fn reset_for_shot(
        &mut self,
        shot_id: impl Into<String>,
    ) -> Result<(), PlasmaEventEvalError> {
        let shot_id = shot_id.into();
        if shot_id.trim().is_empty() {
            return Err(PlasmaEventEvalError::new("shot_id must be non-empty"));
        }
        self.shot_id = Some(shot_id);
        self.last_window_end_ns = None;
        self.high_run = 0;
        self.low_run = 0;
        self.active = false;
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn policy_config(&self) -> &AlarmPolicyConfig {
        &self.config
    }

    /// Public online entry point: only the measurement-evidence gate can
    /// qualify a frame for a risk score. A blocked frame cannot carry a score.
    ///
    /// Event labels and disruption times are absent from this interface. They
    /// enter only through the separate post-hoc scorer.
    pub fn update_from_frame(
        &mut self,
        frame: &PlasmaObservationFrame,
        requirements: &[DecisionChannelRequirement],
        risk_evidence: Option<RiskScoreEvidence>,
    ) -> Result<AlarmPolicyUpdate, PlasmaEventEvalError> {
        let gate = frame
            .evaluate_decision_requirements(requirements)
            .map_err(|error| PlasmaEventEvalError::new(error.message))?;
        match gate {
            DecisionInputGate::MeasuredInputsAvailable(inputs) => {
                let evidence = risk_evidence.ok_or_else(|| {
                    PlasmaEventEvalError::new(
                        "a qualified measured frame requires model-bound risk evidence",
                    )
                })?;
                evidence.validate_for_frame(frame, &self.config)?;
                let risk_evidence_sha256 = evidence.evidence_sha256()?;
                let observation = AlarmObservation {
                    shot_id: inputs.shot_id().to_owned(),
                    window_end_ns: inputs.frame_time_ns(),
                    risk_score: Some(evidence.estimate),
                    lower_bound: Some(evidence.lower_bound),
                    upper_bound: Some(evidence.upper_bound),
                    risk_evidence_sha256: Some(risk_evidence_sha256),
                    evidence_quality: AlarmEvidenceQuality::DirectMeasured,
                };
                self.update(&observation)
            }
            DecisionInputGate::Abstain {
                frame_id: _,
                blocked_channels,
            } => {
                if risk_evidence.is_some() {
                    return Err(PlasmaEventEvalError::new(
                        "a blocked evidence frame must not carry model-bound risk evidence",
                    ));
                }
                self.validate_timeline(&frame.shot_id, frame.frame_time_ns)?;
                self.last_window_end_ns = Some(frame.frame_time_ns);
                self.high_run = 0;
                self.low_run = 0;
                Ok(AlarmPolicyUpdate::Abstained {
                    at_ns: frame.frame_time_ns,
                    reason: AlarmAbstentionReason::EvidenceGateBlocked(blocked_channels),
                    alert_remains_active: self.active,
                })
            }
        }
    }

    fn validate_timeline(
        &self,
        shot_id: &str,
        window_end_ns: u64,
    ) -> Result<(), PlasmaEventEvalError> {
        let current_shot = self
            .shot_id
            .as_deref()
            .ok_or_else(|| PlasmaEventEvalError::new("reset_for_shot must be called first"))?;
        if shot_id != current_shot {
            return Err(PlasmaEventEvalError::new(
                "observation shot_id does not match the active discharge",
            ));
        }
        if let Some(previous) = self.last_window_end_ns
            && window_end_ns <= previous
        {
            return Err(PlasmaEventEvalError::new(
                "window_end_ns must be strictly increasing",
            ));
        }
        Ok(())
    }

    /// Internal state-machine step. It is deliberately not public: all public
    /// inputs must first pass update_from_frame and its measurement gate.
    fn update(
        &mut self,
        observation: &AlarmObservation,
    ) -> Result<AlarmPolicyUpdate, PlasmaEventEvalError> {
        self.validate_timeline(&observation.shot_id, observation.window_end_ns)?;

        let (score, lower_bound, upper_bound, risk_evidence_sha256) =
            match observation.evidence_quality {
                AlarmEvidenceQuality::DirectMeasured => {
                    let score = observation.risk_score.ok_or_else(|| {
                        PlasmaEventEvalError::new("measured evidence requires a risk score")
                    })?;
                    let lower = observation.lower_bound.ok_or_else(|| {
                        PlasmaEventEvalError::new("measured evidence requires a lower bound")
                    })?;
                    let upper = observation.upper_bound.ok_or_else(|| {
                        PlasmaEventEvalError::new("measured evidence requires an upper bound")
                    })?;
                    let digest = observation.risk_evidence_sha256.as_deref().ok_or_else(|| {
                        PlasmaEventEvalError::new("measured evidence requires an evidence digest")
                    })?;
                    if !score.is_finite()
                        || !lower.is_finite()
                        || !upper.is_finite()
                        || lower > score
                        || score > upper
                        || !valid_sha256(digest)
                    {
                        return Err(PlasmaEventEvalError::new(
                            "risk score, uncertainty interval or evidence digest is invalid",
                        ));
                    }
                    (score, lower, upper, digest.to_owned())
                }
                _ => {
                    if observation.risk_score.is_some()
                        || observation.lower_bound.is_some()
                        || observation.upper_bound.is_some()
                        || observation.risk_evidence_sha256.is_some()
                    {
                        return Err(PlasmaEventEvalError::new(
                            "non-measured evidence must not carry a risk score or evidence digest",
                        ));
                    }
                    return Err(PlasmaEventEvalError::new(
                        "internal alarm state machine accepts only gate-qualified measured input",
                    ));
                }
            };

        self.last_window_end_ns = Some(observation.window_end_ns);

        if !self.active {
            self.low_run = 0;
            if lower_bound >= self.config.trigger_threshold {
                self.high_run = self.high_run.saturating_add(1);
                if self.high_run >= self.config.trigger_persistence_windows {
                    self.active = true;
                    self.high_run = 0;
                    return Ok(AlarmPolicyUpdate::Raised {
                        at_ns: observation.window_end_ns,
                        risk_evidence_sha256,
                    });
                }
                return Ok(AlarmPolicyUpdate::PendingTrigger {
                    at_ns: observation.window_end_ns,
                    consecutive_windows: self.high_run,
                    risk_evidence_sha256,
                });
            }
            self.high_run = 0;
            if upper_bound < self.config.trigger_threshold {
                return Ok(AlarmPolicyUpdate::BelowThreshold {
                    at_ns: observation.window_end_ns,
                    risk_evidence_sha256,
                });
            }
            return Ok(AlarmPolicyUpdate::Uncertain {
                at_ns: observation.window_end_ns,
                estimate: score,
                lower_bound,
                upper_bound,
                risk_evidence_sha256,
                alert_remains_active: false,
            });
        }

        self.high_run = 0;
        if upper_bound <= self.config.clear_threshold {
            self.low_run = self.low_run.saturating_add(1);
            if self.low_run >= self.config.clear_persistence_windows {
                self.active = false;
                self.low_run = 0;
                return Ok(AlarmPolicyUpdate::Cleared {
                    at_ns: observation.window_end_ns,
                    risk_evidence_sha256,
                });
            }
            return Ok(AlarmPolicyUpdate::PendingClear {
                at_ns: observation.window_end_ns,
                consecutive_windows: self.low_run,
                risk_evidence_sha256,
            });
        }
        self.low_run = 0;
        if lower_bound > self.config.clear_threshold {
            return Ok(AlarmPolicyUpdate::Active {
                at_ns: observation.window_end_ns,
                risk_evidence_sha256,
            });
        }
        Ok(AlarmPolicyUpdate::Uncertain {
            at_ns: observation.window_end_ns,
            estimate: score,
            lower_bound,
            upper_bound,
            risk_evidence_sha256,
            alert_remains_active: true,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ShotReferenceOutcome {
    Disruptive { t_cut_ns: u64 },
    NonDisruptive { observation_end_ns: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShotAlarmEvaluation {
    pub shot_id: String,
    pub policy_id: String,
    pub validation_receipt_sha256: String,
    pub policy_config_sha256: String,
    pub required_warning_ns: u64,
    pub max_warning_ns: u64,
    pub disruptive: bool,
    pub event_detected: bool,
    pub false_alarm_events: usize,
    pub lead_time_ns: Option<u64>,
    /// Shot exposure from time zero to t_cut or observation end.
    pub exposure_duration_ns: u64,
    /// Post-hoc event reference; never supplied to the online policy.
    pub outcome: ShotReferenceOutcome,
    /// One timestamp per transition into Raised, not each Active window.
    pub alarm_times_ns: Vec<u64>,
}

/// Scores timestamps already emitted by the online policy. Event references
/// are used only here, after the online alarm trace has been recorded.
///
/// `alarm_times_ns` must contain one timestamp per transition into `Raised`,
/// not every subsequent `Active` window. Each timestamp therefore represents
/// the start of a distinct alarm episode; non-disruptive episodes count as
/// false-alarm events and are normalized by observed negative-shot exposure.
pub fn score_shot_alarms(
    shot_id: &str,
    outcome: ShotReferenceOutcome,
    alarm_times_ns: &[u64],
    policy: &AlarmPolicyConfig,
) -> Result<ShotAlarmEvaluation, PlasmaEventEvalError> {
    policy.validate()?;
    if shot_id.trim().is_empty() {
        return Err(PlasmaEventEvalError::new("shot_id must be non-empty"));
    }
    for pair in alarm_times_ns.windows(2) {
        if pair[1] <= pair[0] {
            return Err(PlasmaEventEvalError::new(
                "alarm timestamps must be strictly increasing",
            ));
        }
    }

    match outcome {
        ShotReferenceOutcome::Disruptive { t_cut_ns } => {
            if t_cut_ns == 0 {
                return Err(PlasmaEventEvalError::new(
                    "disruptive-shot exposure duration must be positive",
                ));
            }
            if alarm_times_ns.iter().any(|time| *time > t_cut_ns) {
                return Err(PlasmaEventEvalError::new(
                    "disruptive-shot alarm trace extends beyond t_cut_ns",
                ));
            }
            let eligible_alarm = alarm_times_ns.iter().find_map(|time| {
                let lead = t_cut_ns.saturating_sub(*time);
                (lead >= policy.required_warning_ns && lead <= policy.max_warning_ns)
                    .then_some(lead)
            });
            let false_alarm_events = alarm_times_ns.len().saturating_sub(usize::from(
                eligible_alarm.is_some(),
            ));
            Ok(ShotAlarmEvaluation {
                shot_id: shot_id.to_owned(),
                policy_id: policy.policy_id.clone(),
                validation_receipt_sha256: policy.validation_receipt_sha256.clone(),
                policy_config_sha256: policy.policy_config_sha256(),
                required_warning_ns: policy.required_warning_ns,
                max_warning_ns: policy.max_warning_ns,
                disruptive: true,
                event_detected: eligible_alarm.is_some(),
                false_alarm_events,
                lead_time_ns: eligible_alarm,
                exposure_duration_ns: t_cut_ns,
                outcome,
                alarm_times_ns: alarm_times_ns.to_vec(),
            })
        }
        ShotReferenceOutcome::NonDisruptive {
            observation_end_ns,
        } => {
            if observation_end_ns == 0 {
                return Err(PlasmaEventEvalError::new(
                    "non-disruptive observation exposure duration must be positive",
                ));
            }
            if alarm_times_ns
                .iter()
                .any(|time| *time > observation_end_ns)
            {
                return Err(PlasmaEventEvalError::new(
                    "non-disruptive alarm trace extends beyond observation_end_ns",
                ));
            }
            Ok(ShotAlarmEvaluation {
                shot_id: shot_id.to_owned(),
                policy_id: policy.policy_id.clone(),
                validation_receipt_sha256: policy.validation_receipt_sha256.clone(),
                policy_config_sha256: policy.policy_config_sha256(),
                required_warning_ns: policy.required_warning_ns,
                max_warning_ns: policy.max_warning_ns,
                disruptive: false,
                event_detected: false,
                false_alarm_events: alarm_times_ns.len(),
                lead_time_ns: None,
                exposure_duration_ns: observation_end_ns,
                outcome,
                alarm_times_ns: alarm_times_ns.to_vec(),
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlarmEvaluationSummary {
    pub policy_id: String,
    pub validation_receipt_sha256: String,
    pub policy_config_sha256: String,
    pub required_warning_ns: u64,
    pub max_warning_ns: u64,
    pub shot_count: usize,
    pub disruptive_shots: usize,
    pub non_disruptive_shots: usize,
    pub detected_disruptions: usize,
    pub missed_disruptions: usize,
    pub false_alarm_shots: usize,
    /// False alarm episodes on disruptive shots outside the accepted warning window.
    pub false_alarm_events_on_disruptive_shots: usize,
    /// False alarm episodes on non-disruptive shots; this is the numerator for FAR metrics.
    pub false_alarm_events_on_non_disruptive_shots: usize,
    /// Total false alarm episodes across both positive and negative shots.
    pub false_alarm_events: usize,
    pub total_non_disruptive_exposure_ns: u128,
    /// None when there are no disruptive shots.
    pub event_tpr: Option<f64>,
    /// None when there are no non-disruptive shots.
    pub false_alarm_shot_rate: Option<f64>,
    /// None when there are no non-disruptive shots.
    pub false_alarms_per_non_disruptive_shot: Option<f64>,
    /// False-alarm episodes divided by accumulated non-disruptive exposure in hours.
    /// None when the negative cohort has no exposure.
    pub false_alarms_per_non_disruptive_hour: Option<f64>,
    pub mean_warning_time_ns: Option<f64>,
    pub event_tpr_wilson_95: Option<RateConfidenceInterval>,
    pub false_alarm_shot_rate_wilson_95: Option<RateConfidenceInterval>,
    pub warning_time_p05_ns: Option<u64>,
    pub warning_time_p10_ns: Option<u64>,
    pub warning_time_p50_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
}

/// Wilson score interval for a binomial per-shot rate. Returns None for an
/// empty denominator cohort rather than inventing a zero rate or interval.
fn wilson_interval(successes: usize, trials: usize) -> Option<RateConfidenceInterval> {
    if trials == 0 || successes > trials {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width =
        z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt() / denominator;
    Some(RateConfidenceInterval {
        lower: (center - half_width).max(0.0),
        upper: (center + half_width).min(1.0),
    })
}

fn nearest_rank_quantile(sorted_values: &[u64], quantile: f64) -> Option<u64> {
    if sorted_values.is_empty() {
        return None;
    }
    let rank = (quantile * sorted_values.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_values.len() - 1);
    sorted_values.get(index).copied()
}

/// Aggregates fixed-policy shot outcomes; it does not select or tune thresholds.
pub fn summarize_alarm_evaluations(
    evaluations: &[ShotAlarmEvaluation],
) -> Result<AlarmEvaluationSummary, PlasmaEventEvalError> {
    if evaluations.is_empty() {
        return Err(PlasmaEventEvalError::new(
            "at least one shot evaluation is required",
        ));
    }

    let mut shot_ids = BTreeSet::new();
    let mut policy_id: Option<String> = None;
    let mut validation_receipt_sha256: Option<String> = None;
    let mut policy_config_sha256: Option<String> = None;
    let mut required_warning_ns: Option<u64> = None;
    let mut max_warning_ns: Option<u64> = None;
    let mut disruptive_shots = 0usize;
    let mut non_disruptive_shots = 0usize;
    let mut detected_disruptions = 0usize;
    let mut false_alarm_shots = 0usize;
    let mut false_alarm_events_on_disruptive_shots = 0usize;
    let mut false_alarm_events_on_non_disruptive_shots = 0usize;
    let mut false_alarm_events = 0usize;
    let mut total_non_disruptive_exposure_ns = 0u128;
    let mut warning_times_ns = Vec::new();

    for evaluation in evaluations {
        if evaluation.shot_id.trim().is_empty() || !shot_ids.insert(&evaluation.shot_id) {
            return Err(PlasmaEventEvalError::new(
                "shot IDs must be non-empty and unique in one evaluation cohort",
            ));
        }
        if evaluation.policy_id.trim().is_empty()
            || !valid_sha256(&evaluation.validation_receipt_sha256)
            || !valid_sha256(&evaluation.policy_config_sha256)
            || evaluation.required_warning_ns == 0
            || evaluation.max_warning_ns < evaluation.required_warning_ns
            || evaluation.exposure_duration_ns == 0
        {
            return Err(PlasmaEventEvalError::new(
                "shot evaluation has incomplete policy identity or exposure metadata",
            ));
        }
        match (
            &policy_id,
            &validation_receipt_sha256,
            &policy_config_sha256,
            required_warning_ns,
            max_warning_ns,
        ) {
            (None, None, None, None, None) => {
                policy_id = Some(evaluation.policy_id.clone());
                validation_receipt_sha256 = Some(evaluation.validation_receipt_sha256.clone());
                policy_config_sha256 = Some(evaluation.policy_config_sha256.clone());
                required_warning_ns = Some(evaluation.required_warning_ns);
                max_warning_ns = Some(evaluation.max_warning_ns);
            }
            (Some(id), Some(receipt), Some(config), Some(warning), Some(max_warning))
                if id == &evaluation.policy_id
                    && receipt == &evaluation.validation_receipt_sha256
                    && config == &evaluation.policy_config_sha256
                    && warning == evaluation.required_warning_ns
                    && max_warning == evaluation.max_warning_ns => {}
            _ => {
                return Err(PlasmaEventEvalError::new(
                    "one summary cannot mix alarm policy identities, config digests or warning requirements",
                ));
            }
        }

        if evaluation.disruptive {
            disruptive_shots += 1;
            if evaluation.event_detected != evaluation.lead_time_ns.is_some() {
                return Err(PlasmaEventEvalError::new(
                    "disruptive shot evaluation has inconsistent event metrics",
                ));
            }
            if let Some(lead_time_ns) = evaluation.lead_time_ns {
                if lead_time_ns < evaluation.required_warning_ns
                    || lead_time_ns > evaluation.max_warning_ns
                {
                    return Err(PlasmaEventEvalError::new(
                        "detected event lead time lies outside the frozen warning interval",
                    ));
                }
                detected_disruptions += 1;
                warning_times_ns.push(lead_time_ns);
            }
            false_alarm_events_on_disruptive_shots = false_alarm_events_on_disruptive_shots
                .checked_add(evaluation.false_alarm_events)
                .ok_or_else(|| PlasmaEventEvalError::new("false alarm event count overflow"))?;
            false_alarm_events = false_alarm_events
                .checked_add(evaluation.false_alarm_events)
                .ok_or_else(|| PlasmaEventEvalError::new("false alarm event count overflow"))?;
            if evaluation.event_detected {
                let expected_false_alarms = evaluation.alarm_times_ns.len().saturating_sub(1);
                if evaluation.false_alarm_events != expected_false_alarms {
                    return Err(PlasmaEventEvalError::new(
                        "disruptive shot false-alarm count does not match alarm episodes outside the single matched warning",
                    ));
                }
            } else if evaluation.false_alarm_events != evaluation.alarm_times_ns.len() {
                return Err(PlasmaEventEvalError::new(
                    "undetected disruptive shot must count every alarm episode as a false alarm",
                ));
            }
        } else {
            non_disruptive_shots += 1;
            total_non_disruptive_exposure_ns = total_non_disruptive_exposure_ns
                .checked_add(u128::from(evaluation.exposure_duration_ns))
                .ok_or_else(|| {
                    PlasmaEventEvalError::new("non-disruptive exposure duration overflow")
                })?;
            if evaluation.event_detected || evaluation.lead_time_ns.is_some() {
                return Err(PlasmaEventEvalError::new(
                    "non-disruptive shot cannot be marked as an event detection",
                ));
            }
            if evaluation.false_alarm_events > 0 {
                false_alarm_shots += 1;
            }
            false_alarm_events_on_non_disruptive_shots = false_alarm_events_on_non_disruptive_shots
                .checked_add(evaluation.false_alarm_events)
                .ok_or_else(|| PlasmaEventEvalError::new("false-alarm event count overflow"))?;
            false_alarm_events = false_alarm_events
                .checked_add(evaluation.false_alarm_events)
                .ok_or_else(|| PlasmaEventEvalError::new("false-alarm event count overflow"))?;
        }
    }

    warning_times_ns.sort_unstable();
    let missed_disruptions = disruptive_shots - detected_disruptions;
    let mean_warning_time_ns = if warning_times_ns.is_empty() {
        None
    } else {
        let total: u128 = warning_times_ns.iter().map(|value| u128::from(*value)).sum();
        Some(total as f64 / warning_times_ns.len() as f64)
    };

    Ok(AlarmEvaluationSummary {
        policy_id: policy_id.unwrap_or_default(),
        validation_receipt_sha256: validation_receipt_sha256.unwrap_or_default(),
        policy_config_sha256: policy_config_sha256.unwrap_or_default(),
        required_warning_ns: required_warning_ns.unwrap_or_default(),
        max_warning_ns: max_warning_ns.unwrap_or_default(),
        shot_count: evaluations.len(),
        disruptive_shots,
        non_disruptive_shots,
        detected_disruptions,
        missed_disruptions,
        false_alarm_shots,
        false_alarm_events_on_disruptive_shots,
        false_alarm_events_on_non_disruptive_shots,
        false_alarm_events,
        total_non_disruptive_exposure_ns,
        event_tpr: (disruptive_shots > 0)
            .then_some(detected_disruptions as f64 / disruptive_shots as f64),
        event_tpr_wilson_95: wilson_interval(detected_disruptions, disruptive_shots),
        false_alarm_shot_rate: (non_disruptive_shots > 0)
            .then_some(false_alarm_shots as f64 / non_disruptive_shots as f64),
        false_alarm_shot_rate_wilson_95: wilson_interval(
            false_alarm_shots,
            non_disruptive_shots,
        ),
        false_alarms_per_non_disruptive_shot: (non_disruptive_shots > 0)
            .then_some(false_alarm_events_on_non_disruptive_shots as f64 / non_disruptive_shots as f64),
        false_alarms_per_non_disruptive_hour: (total_non_disruptive_exposure_ns > 0).then_some(
            false_alarm_events_on_non_disruptive_shots as f64
                / (total_non_disruptive_exposure_ns as f64 / 3_600_000_000_000.0),
        ),
        mean_warning_time_ns,
        warning_time_p05_ns: nearest_rank_quantile(&warning_times_ns, 0.05),
        warning_time_p10_ns: nearest_rank_quantile(&warning_times_ns, 0.10),
        warning_time_p50_ns: nearest_rank_quantile(&warning_times_ns, 0.50),
    })
}

/// Versioned machine-readable result receipt for one fixed-policy TokaMark
/// event-evaluation cohort. Content digests are integrity identifiers, not
/// signatures or proof that the referenced dataset/model/calibration is authentic.
pub const PLASMA_EVENT_EVALUATION_RECEIPT_SCHEMA: &str =
    "luminous.plasma-event-evaluation-receipt.v0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaEventEvaluationReceiptBody {
    pub schema_version: String,
    pub benchmark_id: String,
    pub repository_commit: String,
    pub dataset_revision: String,
    pub shot_split_sha256: String,
    pub preprocessing_sha256: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_artifact_sha256: String,
    pub seed: u64,
    pub hardware: String,
    pub policy: AlarmPolicyConfig,
    pub shot_evaluations: Vec<ShotAlarmEvaluation>,
    pub summary: AlarmEvaluationSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaEventEvaluationReceipt {
    pub body: PlasmaEventEvaluationReceiptBody,
    pub receipt_sha256: String,
}

fn valid_git_object_id(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl PlasmaEventEvaluationReceiptBody {
    fn validate_metadata(&self) -> Result<(), PlasmaEventEvalError> {
        if self.schema_version != PLASMA_EVENT_EVALUATION_RECEIPT_SCHEMA
            || self.benchmark_id != "PLASMA-EVENT-EVAL-001"
        {
            return Err(PlasmaEventEvalError::new(
                "unsupported plasma event evaluation receipt identity",
            ));
        }
        if !valid_git_object_id(&self.repository_commit) {
            return Err(PlasmaEventEvalError::new(
                "repository_commit must be a lowercase 40- or 64-character Git object ID",
            ));
        }
        if self.dataset_revision.trim().is_empty()
            || self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.hardware.trim().is_empty()
        {
            return Err(PlasmaEventEvalError::new(
                "dataset revision, model identity and hardware metadata must be non-empty",
            ));
        }
        for (digest, field) in [
            (self.shot_split_sha256.as_str(), "shot_split_sha256"),
            (self.preprocessing_sha256.as_str(), "preprocessing_sha256"),
            (self.model_artifact_sha256.as_str(), "model_artifact_sha256"),
        ] {
            if !valid_sha256(digest) {
                return Err(PlasmaEventEvalError::new(format!(
                    "{field} must be 64 lowercase hexadecimal characters",
                )));
            }
        }
        self.policy.validate()?;
        Ok(())
    }

    fn verify_evaluations(&self) -> Result<(), PlasmaEventEvalError> {
        self.validate_metadata()?;
        for evaluation in &self.shot_evaluations {
            let recomputed = score_shot_alarms(
                &evaluation.shot_id,
                evaluation.outcome,
                &evaluation.alarm_times_ns,
                &self.policy,
            )?;
            if recomputed != *evaluation {
                return Err(PlasmaEventEvalError::new(format!(
                    "shot evaluation {} does not match its stored outcome and alarm timestamps",
                    evaluation.shot_id
                )));
            }
        }
        let recomputed_summary = summarize_alarm_evaluations(&self.shot_evaluations)?;
        if recomputed_summary != self.summary {
            return Err(PlasmaEventEvalError::new(
                "receipt cohort summary does not match its per-shot evaluations",
            ));
        }
        Ok(())
    }

    fn content_sha256(&self) -> Result<String, PlasmaEventEvalError> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            PlasmaEventEvalError::new(format!(
                "evaluation receipt serialization failed: {error}"
            ))
        })?;
        Ok(sha256_hex(&bytes))
    }
}

impl PlasmaEventEvaluationReceipt {
    /// Builds a receipt from fixed-policy post-hoc shot evaluations. The
    /// receipt recomputes all per-shot metrics and the cohort summary before
    /// it is emitted.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        repository_commit: String,
        dataset_revision: String,
        shot_split_sha256: String,
        preprocessing_sha256: String,
        model_id: String,
        model_revision: String,
        model_artifact_sha256: String,
        seed: u64,
        hardware: String,
        policy: AlarmPolicyConfig,
        shot_evaluations: Vec<ShotAlarmEvaluation>,
    ) -> Result<Self, PlasmaEventEvalError> {
        let summary = summarize_alarm_evaluations(&shot_evaluations)?;
        let body = PlasmaEventEvaluationReceiptBody {
            schema_version: PLASMA_EVENT_EVALUATION_RECEIPT_SCHEMA.to_owned(),
            benchmark_id: "PLASMA-EVENT-EVAL-001".to_owned(),
            repository_commit,
            dataset_revision,
            shot_split_sha256,
            preprocessing_sha256,
            model_id,
            model_revision,
            model_artifact_sha256,
            seed,
            hardware,
            policy,
            shot_evaluations,
            summary,
        };
        body.verify_evaluations()?;
        let receipt_sha256 = body.content_sha256()?;
        Ok(Self {
            body,
            receipt_sha256,
        })
    }

    /// Recomputes all per-shot outcomes, the cohort summary and the outer
    /// content digest. A successful check does not authenticate external
    /// dataset/model/validation artifacts.
    pub fn verify(&self) -> Result<(), PlasmaEventEvalError> {
        self.body.verify_evaluations()?;
        if self.receipt_sha256 != self.body.content_sha256()? {
            return Err(PlasmaEventEvalError::new(
                "plasma event evaluation receipt digest mismatch",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AlarmPolicyConfig {
        AlarmPolicyConfig {
            policy_id: "fixed-test-policy-v1".to_owned(),
            score_scale_id: "normalized-risk-0-1-v1".to_owned(),
            validation_receipt_sha256: "a".repeat(64),
            risk_model: RiskModelBinding {
                model_id: "symthaea-plasma-risk".to_owned(),
                model_revision: "test-model-v1".to_owned(),
                model_artifact_sha256: "1".repeat(64),
                training_split_sha256: "2".repeat(64),
                calibration_id: "plasma-risk-calibration".to_owned(),
                calibration_revision: "cal-v1".to_owned(),
                calibration_receipt_sha256: "3".repeat(64),
                uncertainty_method_id: "heldout-conformal-v1".to_owned(),
                nominal_coverage: 0.9,
            },
            trigger_threshold: 0.8,
            clear_threshold: 0.3,
            trigger_persistence_windows: 2,
            clear_persistence_windows: 2,
            required_warning_ns: 50,
            max_warning_ns: 90,
        }
    }

    use crate::plasma_observability::{
        CalibrationIdentity, DecisionChannelRequirement, DiagnosticChannelObservation,
        ObservationDisposition, PlasmaObservationFrame, PLASMA_OBSERVATION_SCHEMA,
    };

    fn observation_frame(
        disposition: ObservationDisposition,
        frame_time_ns: u64,
    ) -> (PlasmaObservationFrame, Vec<DecisionChannelRequirement>) {
        let calibration = CalibrationIdentity {
            calibration_id: "density-calibration".to_owned(),
            calibration_revision: "rev-1".to_owned(),
        };
        let frame = PlasmaObservationFrame {
            schema_version: PLASMA_OBSERVATION_SCHEMA.to_owned(),
            shot_id: "shot-1".to_owned(),
            frame_id: format!("frame-{frame_time_ns}"),
            clock_domain: "shot-relative-ns".to_owned(),
            frame_time_ns,
            expected_channel_ids: vec!["density".to_owned()],
            channels: vec![DiagnosticChannelObservation {
                channel_id: "density".to_owned(),
                diagnostic_system_id: "interferometer".to_owned(),
                disposition,
            }],
        };
        let requirements = vec![DecisionChannelRequirement {
            channel_id: "density".to_owned(),
            expected_units: "10^20 m^-3".to_owned(),
            max_age_ns: 1_000_000,
            allowed_source_ids: vec!["source-density".to_owned()],
            calibration,
        }];
        (frame, requirements)
    }

    fn measured_disposition(frame_time_ns: u64) -> ObservationDisposition {
        ObservationDisposition::Measured {
            value: 0.8,
            units: "10^20 m^-3".to_owned(),
            source_id: "source-density".to_owned(),
            source_digest: "a".repeat(64),
            calibration: CalibrationIdentity {
                calibration_id: "density-calibration".to_owned(),
                calibration_revision: "rev-1".to_owned(),
            },
            observed_at_ns: frame_time_ns,
        }
    }

    fn risk_evidence(
        frame: &PlasmaObservationFrame,
        estimate: f64,
        lower_bound: f64,
        upper_bound: f64,
    ) -> RiskScoreEvidence {
        RiskScoreEvidence {
            schema_version: PLASMA_RISK_SCORE_EVIDENCE_SCHEMA.to_owned(),
            shot_id: frame.shot_id.clone(),
            frame_id: frame.frame_id.clone(),
            frame_time_ns: frame.frame_time_ns,
            input_frame_sha256: observation_frame_sha256(frame).expect("frame digest"),
            horizon_ns: 10,
            target_time_ns: frame.frame_time_ns.checked_add(10).expect("target time"),
            model_id: "symthaea-plasma-risk".to_owned(),
            model_revision: "test-model-v1".to_owned(),
            score_scale_id: "normalized-risk-0-1-v1".to_owned(),
            model_artifact_sha256: "1".repeat(64),
            training_split_sha256: "2".repeat(64),
            calibration_id: "plasma-risk-calibration".to_owned(),
            calibration_revision: "cal-v1".to_owned(),
            calibration_receipt_sha256: "3".repeat(64),
            uncertainty_method_id: "heldout-conformal-v1".to_owned(),
            nominal_coverage: 0.9,
            estimate,
            lower_bound,
            upper_bound,
        }
    }

    fn synthetic_disposition() -> ObservationDisposition {
        ObservationDisposition::Synthetic {
            value: 0.8,
            units: "10^20 m^-3".to_owned(),
            model_id: "reconstructor".to_owned(),
            model_revision: "v1".to_owned(),
            input_digest: "b".repeat(64),
        }
    }

    fn unavailable_disposition() -> ObservationDisposition {
        ObservationDisposition::Unavailable {
            reason: "diagnostic acquisition missing".to_owned(),
        }
    }

    fn measured(shot_id: &str, time_ns: u64, score: f64) -> AlarmObservation {
        AlarmObservation {
            shot_id: shot_id.to_owned(),
            window_end_ns: time_ns,
            risk_score: Some(score),
            lower_bound: Some(score),
            upper_bound: Some(score),
            risk_evidence_sha256: Some("d".repeat(64)),
            evidence_quality: AlarmEvidenceQuality::DirectMeasured,
        }
    }

    #[test]
    fn frozen_protocol_manifest_forbids_test_threshold_optimization() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/plasma/event-eval-v0.json"
        ))
        .expect("event evaluation manifest should parse");
        assert_eq!(
            manifest["schema_version"],
            "luminous.plasma-event-evaluation.v0.1"
        );
        assert_eq!(manifest["benchmark_id"], "PLASMA-EVENT-EVAL-001");
        assert_eq!(manifest["population_and_split_contract"]["test_is_read_only"], true);
        assert_eq!(
            manifest["population_and_split_contract"]["split_tracks"]["tokamark_temporal_transfer"]["use_for_primary_temporal_generalization"],
            true
        );
        assert_eq!(
            manifest["population_and_split_contract"]["split_tracks"]["tokamark_random_split"]["use_as_secondary_only"],
            true
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["test_shots"],
            933
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["test_windows"],
            39967
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["common_f1_threshold"],
            0.4
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["macro_average"]["gentoka_f1"],
            0.5726
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["macro_average"]["gentoka_auroc"],
            0.9049
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["macro_average"]["tokamind_auroc"],
            0.9069
        );
        assert_eq!(
            manifest["reference_benchmarks"]["published_toka_mark_robustness"]["published_alarm_evaluation"]["false_alarm_rate"],
            "undefined_not_computable"
        );
        assert_eq!(
            manifest["online_policy_contract"]["test_threshold_optimization"],
            false
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["uncertainty_straddles_threshold"],
            "return_uncertain_and_preserve_current_alarm_state"
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["model_artifact_sha256_required"],
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["exact_input_frame_sha256_required"],
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["uncertainty_method_and_nominal_coverage_required"],
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["risk_evidence_schema_version"],
            PLASMA_RISK_SCORE_EVIDENCE_SCHEMA
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["threshold_scale_identity"],
            "risk evidence score_scale_id must exactly match frozen policy score_scale_id"
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["policy_config_digest_schema"],
            "luminous.plasma-alarm-policy.v0.5"
        );
        assert_eq!(
            manifest["online_policy_contract"]["policy_config_digest_schema"],
            "luminous.plasma-alarm-policy.v0.5"
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["exact_model_and_calibration_binding_required"],
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_score_contract"]["model_training_split_calibration_uncertainty_must_match_frozen_policy"],
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["risk_model_binding_fields"].as_array().map(Vec::len),
            Some(9)
        );
        assert_eq!(
            manifest["online_policy_contract"]["event_labels_available_online"],
            false
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["warning_requirement_ns"],
            50_000_000
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["maximum_warning_horizon_source"],
            "validation-selected and frozen in AlarmPolicyConfig.max_warning_ns"
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["alarms_outside_warning_interval"],
            "count as false alarm episodes; they do not satisfy event detection"
        );
        assert_eq!(
            manifest["online_policy_contract"]["policy_config_fields"]
                .as_array()
                .expect("policy field list")
                .iter()
                .any(|field| field == "max_warning_ns"),
            true
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["warning_interval"]["event_detected_when"],
            "at least one alarm episode has t_cut - alarm_time in [minimum_ns, maximum_ns]"
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["metrics"]
                .as_array()
                .expect("metrics array")
                .iter()
                .any(|metric| metric == "false_alarm_events_on_disruptive_shots"),
            true
        );
        assert_eq!(
            manifest["online_policy_contract"]["rolling_inference_contract"]["cut_interval_ns"],
            10_000_000
        );
        assert_eq!(
            manifest["reference_benchmarks"]["gen_toka_group6"]["architecture_limitations"][0],
            "GenToka generates a complete requested output window in one conditional pass, rather than advancing autoregressively with updated context"
        );
        assert!(
            manifest["offline_scoring_contract"]["metrics"]
                .as_array()
                .expect("metrics array")
                .iter()
                .any(|metric| metric == "false_alarms_per_non_disruptive_hour")
        );
        assert!(
            manifest["offline_scoring_contract"]["metrics"]
                .as_array()
                .expect("metrics array")
                .iter()
                .any(|metric| metric == "false_alarm_events_on_disruptive_shots")
        );
        assert_eq!(
            manifest["offline_scoring_contract"]["false_alarm_rate_denominators"]["per_hour"],
            "sum of non-disruptive shot exposure durations in hours"
        );
        assert_eq!(
            manifest["evaluation_receipt_contract"]["schema_version"],
            PLASMA_EVENT_EVALUATION_RECEIPT_SCHEMA
        );
        assert_eq!(
            manifest["evaluation_receipt_contract"]["verify_recomputes_per_shot_scores"],
            true
        );
        assert_eq!(
            manifest["evaluation_receipt_contract"]["outer_sha256_is_content_identity_not_authentication"],
            true
        );
        assert_eq!(
            manifest["target_gates"]["all_values_are_research_targets"],
            true
        );
    }

    #[test]
    fn evaluation_receipt_roundtrips_and_recomputes_its_cohort() {
        let positive = score_shot_alarms(
            "positive",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("positive");
        let negative = score_shot_alarms(
            "negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 3_600_000_000_000,
            },
            &[10, 20],
            &config(),
        )
        .expect("negative");
        let receipt = PlasmaEventEvaluationReceipt::build(
            "f".repeat(40),
            "UKAEA-IBM-STFC/tokamark-v1@frozen-revision".to_owned(),
            "a".repeat(64),
            "b".repeat(64),
            "symthaea-plasma-risk".to_owned(),
            "model-rev-1".to_owned(),
            "c".repeat(64),
            42,
            "cpu-x86_64-rust-1.96".to_owned(),
            config(),
            vec![positive, negative],
        )
        .expect("receipt should build");
        receipt.verify().expect("receipt should verify");

        let encoded = serde_json::to_string(&receipt).expect("receipt serialization");
        let decoded: PlasmaEventEvaluationReceipt =
            serde_json::from_str(&encoded).expect("receipt deserialization");
        decoded.verify().expect("round-tripped receipt should verify");
        assert_eq!(
            decoded.body.summary.false_alarms_per_non_disruptive_hour,
            Some(2.0)
        );
    }

    #[test]
    fn evaluation_receipt_detects_tampered_alarm_trace_even_after_digest_reseal() {
        let negative = score_shot_alarms(
            "negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[10],
            &config(),
        )
        .expect("negative");
        let mut receipt = PlasmaEventEvaluationReceipt::build(
            "f".repeat(40),
            "tokamark-v1@frozen-revision".to_owned(),
            "a".repeat(64),
            "b".repeat(64),
            "symthaea-plasma-risk".to_owned(),
            "model-rev-1".to_owned(),
            "c".repeat(64),
            42,
            "cpu-x86_64-rust-1.96".to_owned(),
            config(),
            vec![negative],
        )
        .expect("receipt should build");
        receipt.body.shot_evaluations[0].alarm_times_ns.push(20);
        receipt.receipt_sha256 = receipt
            .body
            .content_sha256()
            .expect("re-seal tampered content");
        assert!(receipt.verify().is_err());
    }

    #[test]
    fn evaluation_receipt_rejects_invalid_reproducibility_metadata() {
        let negative = score_shot_alarms(
            "negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[],
            &config(),
        )
        .expect("negative");
        assert!(PlasmaEventEvaluationReceipt::build(
            "not-a-git-commit".to_owned(),
            "tokamark-v1@frozen-revision".to_owned(),
            "a".repeat(64),
            "b".repeat(64),
            "symthaea-plasma-risk".to_owned(),
            "model-rev-1".to_owned(),
            "c".repeat(64),
            42,
            "cpu-x86_64-rust-1.96".to_owned(),
            config(),
            vec![negative],
        ).is_err());
    }

    #[test]
    fn policy_digest_covers_thresholds_persistence_and_warning_interval() {
        let original = config();
        let digest = original.policy_config_sha256();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));

        let mut changed_scale = original.clone();
        changed_scale.score_scale_id = "alternate-risk-score-v2".to_owned();
        assert_ne!(digest, changed_scale.policy_config_sha256());

        let mut changed_threshold = original.clone();
        changed_threshold.trigger_threshold += 0.01;
        assert_ne!(digest, changed_threshold.policy_config_sha256());

        let mut changed_warning = original.clone();
        changed_warning.required_warning_ns += 1;
        assert_ne!(digest, changed_warning.policy_config_sha256());

        let mut changed_max_warning = original.clone();
        changed_max_warning.max_warning_ns -= 1;
        assert_ne!(digest, changed_max_warning.policy_config_sha256());

        let mut changed_validation = original.clone();
        changed_validation.validation_receipt_sha256 = "b".repeat(64);
        assert_ne!(digest, changed_validation.policy_config_sha256());

        let mut changed_model_artifact = original.clone();
        changed_model_artifact.risk_model.model_artifact_sha256 = "f".repeat(64);
        assert_ne!(digest, changed_model_artifact.policy_config_sha256());

        let mut changed_training_split = original.clone();
        changed_training_split.risk_model.training_split_sha256 = "e".repeat(64);
        assert_ne!(digest, changed_training_split.policy_config_sha256());

        let mut changed_calibration = original.clone();
        changed_calibration.risk_model.calibration_revision = "cal-v2".to_owned();
        assert_ne!(digest, changed_calibration.policy_config_sha256());

        let mut changed_uncertainty = original.clone();
        changed_uncertainty.risk_model.uncertainty_method_id = "different-uncertainty-method".to_owned();
        assert_ne!(digest, changed_uncertainty.policy_config_sha256());

        let mut changed_coverage = original;
        changed_coverage.risk_model.nominal_coverage = 0.95;
        assert_ne!(digest, changed_coverage.policy_config_sha256());
    }

    #[test]
    fn policy_digest_is_unambiguous_when_identifiers_contain_delimiters() {
        let mut first = config();
        first.policy_id = "alpha\\nbeta".to_owned();
        first.score_scale_id = "gamma".to_owned();
        let mut second = config();
        second.policy_id = "alpha".to_owned();
        second.score_scale_id = "beta\\ngamma".to_owned();
        assert_ne!(first.policy_config_sha256(), second.policy_config_sha256());
    }

    #[test]
    fn online_policy_raises_only_after_consecutive_measured_windows() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        assert!(matches!(
            policy.update(&measured("shot-1", 10, 0.9)).expect("update"),
            AlarmPolicyUpdate::PendingTrigger {
                consecutive_windows: 1,
                ..
            }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 20, 0.2)).expect("update"),
            AlarmPolicyUpdate::BelowThreshold { .. }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 30, 0.9)).expect("update"),
            AlarmPolicyUpdate::PendingTrigger {
                consecutive_windows: 1,
                ..
            }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 40, 0.95)).expect("update"),
            AlarmPolicyUpdate::Raised { at_ns: 40 }
        ));
        assert!(policy.is_active());
    }

    #[test]
    fn active_alarm_uses_hysteresis_and_persistent_clear() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        policy.update(&measured("shot-1", 10, 0.9)).expect("update");
        policy.update(&measured("shot-1", 20, 0.9)).expect("update");
        assert!(matches!(
            policy.update(&measured("shot-1", 30, 0.2)).expect("update"),
            AlarmPolicyUpdate::PendingClear {
                consecutive_windows: 1,
                ..
            }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 40, 0.5)).expect("update"),
            AlarmPolicyUpdate::Active { .. }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 50, 0.2)).expect("update"),
            AlarmPolicyUpdate::PendingClear {
                consecutive_windows: 1,
                ..
            }
        ));
        assert!(matches!(
            policy.update(&measured("shot-1", 60, 0.1)).expect("update"),
            AlarmPolicyUpdate::Cleared { at_ns: 60, .. }
        ));
        assert!(!policy.is_active());
    }

    #[test]
    fn unqualified_evidence_abstains_and_breaks_persistence() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        let (measured_frame, requirements) =
            observation_frame(measured_disposition(10), 10);
        policy
            .update_from_frame(
                &measured_frame,
                &requirements,
                Some(risk_evidence(&measured_frame, 0.9, 0.9, 0.9)),
            )
            .expect("direct measurement should be admitted");

        let (synthetic_frame, requirements) = observation_frame(synthetic_disposition(), 20);
        let outcome = policy
            .update_from_frame(&synthetic_frame, &requirements, None)
            .expect("synthetic evidence should abstain");
        assert!(matches!(
            outcome,
            AlarmPolicyUpdate::Abstained {
                at_ns: 20,
                reason: AlarmAbstentionReason::EvidenceGateBlocked(ref blocked),
                alert_remains_active: false,
            } if blocked.len() == 1 && blocked[0].channel_id == "density"
        ));

        let (measured_frame, requirements) =
            observation_frame(measured_disposition(30), 30);
        assert!(matches!(
            policy
                .update_from_frame(
                    &measured_frame,
                    &requirements,
                    Some(risk_evidence(&measured_frame, 0.9, 0.9, 0.9)),
                )
                .expect("direct measurement should be admitted"),
            AlarmPolicyUpdate::PendingTrigger {
                consecutive_windows: 1,
                ..
            }
        ));
    }

    #[test]
    fn abstention_does_not_clear_an_existing_alarm() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        let (frame_10, requirements) = observation_frame(measured_disposition(10), 10);
        policy.update_from_frame(
            &frame_10,
            &requirements,
            Some(risk_evidence(&frame_10, 0.9, 0.9, 0.9)),
        ).expect("update");
        let (frame_20, requirements) = observation_frame(measured_disposition(20), 20);
        policy.update_from_frame(
            &frame_20,
            &requirements,
            Some(risk_evidence(&frame_20, 0.9, 0.9, 0.9)),
        ).expect("update");
        assert!(policy.is_active());

        let (frame_30, requirements) = observation_frame(unavailable_disposition(), 30);
        let outcome = policy
            .update_from_frame(&frame_30, &requirements, None)
            .expect("unavailable measurement should abstain");
        assert!(matches!(
            outcome,
            AlarmPolicyUpdate::Abstained {
                at_ns: 30,
                reason: AlarmAbstentionReason::EvidenceGateBlocked(ref blocked),
                alert_remains_active: true,
            } if blocked.len() == 1 && blocked[0].channel_id == "density"
        ));
        assert!(policy.is_active());
    }

    #[test]
    fn blocked_evidence_cannot_smuggle_a_numeric_score() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        let (frame, requirements) = observation_frame(synthetic_disposition(), 10);
        assert!(policy
            .update_from_frame(
                &frame,
                &requirements,
                Some(risk_evidence(&frame, 0.99, 0.99, 0.99)),
            )
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);
    }

    #[test]
    fn online_policy_rejects_non_monotonic_time_and_wrong_shot() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        policy.update(&measured("shot-1", 20, 0.2)).expect("update");
        assert!(policy.update(&measured("shot-1", 20, 0.9)).is_err());
        assert!(policy.update(&measured("shot-2", 30, 0.9)).is_err());
    }

    #[test]
    fn shot_scorer_uses_a_predeclared_warning_requirement() {
        let detected = score_shot_alarms(
            "shot-disruptive",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("valid shot evaluation");
        assert!(detected.event_detected);
        assert_eq!(detected.lead_time_ns, Some(80));

        let too_late = score_shot_alarms(
            "shot-late",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[80],
            &config(),
        )
        .expect("valid shot evaluation");
        assert!(!too_late.event_detected);
        assert_eq!(too_late.lead_time_ns, None);
    }

    #[test]
    fn disruption_warning_must_fall_inside_frozen_minimum_and_maximum_lead_window() {
        let valid = score_shot_alarms(
            "valid-window",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("valid warning window");
        assert!(valid.event_detected);
        assert_eq!(valid.lead_time_ns, Some(80));
        assert_eq!(valid.false_alarm_events, 0);

        let too_early = score_shot_alarms(
            "too-early",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[0],
            &config(),
        )
        .expect("too-early warning is still a valid evaluation");
        assert!(!too_early.event_detected);
        assert_eq!(too_early.lead_time_ns, None);
        assert_eq!(too_early.false_alarm_events, 1);

        let one_valid_one_outside = score_shot_alarms(
            "multiple-episodes",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[0, 20, 70],
            &config(),
        )
        .expect("mixed early, valid and late episodes");
        assert!(one_valid_one_outside.event_detected);
        assert_eq!(one_valid_one_outside.lead_time_ns, Some(80));
        assert_eq!(one_valid_one_outside.false_alarm_events, 2);

        let summary = summarize_alarm_evaluations(&[
            too_early,
            one_valid_one_outside,
        ]).expect("cohort should aggregate");
        assert_eq!(summary.detected_disruptions, 1);
        assert_eq!(summary.missed_disruptions, 1);
        assert_eq!(summary.false_alarm_events_on_disruptive_shots, 3);
        assert_eq!(summary.false_alarm_events_on_non_disruptive_shots, 0);
        assert_eq!(summary.false_alarm_events, 3);
        assert_eq!(summary.warning_time_p50_ns, Some(80));
        assert_eq!(summary.false_alarm_shot_rate, None);
    }

    #[test]
    fn negative_shots_enable_false_alarm_metrics() {
        let positive = score_shot_alarms(
            "positive",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("positive shot");
        let negative = score_shot_alarms(
            "negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[20, 70],
            &config(),
        )
        .expect("negative shot");
        let summary = summarize_alarm_evaluations(&[positive, negative]).expect("summary");
        assert_eq!(summary.policy_id, "fixed-test-policy-v1");
        assert_eq!(summary.validation_receipt_sha256, "a".repeat(64));
        assert_eq!(summary.policy_config_sha256, config().policy_config_sha256());
        assert_eq!(summary.required_warning_ns, 50);
        let tpr_ci = summary.event_tpr_wilson_95.expect("positive cohort interval");
        assert!(tpr_ci.lower < 1.0 && (tpr_ci.upper - 1.0).abs() < 1e-12);
        let far_ci = summary
            .false_alarm_shot_rate_wilson_95
            .expect("negative cohort interval");
        assert!(far_ci.lower < 1.0 && (far_ci.upper - 1.0).abs() < 1e-12);
        assert_eq!(summary.disruptive_shots, 1);
        assert_eq!(summary.detected_disruptions, 1);
        assert_eq!(summary.non_disruptive_shots, 1);
        assert_eq!(summary.false_alarm_shots, 1);
        assert_eq!(summary.false_alarm_events, 2);
        assert_eq!(summary.event_tpr, Some(1.0));
        assert_eq!(summary.false_alarm_shot_rate, Some(1.0));
        assert_eq!(summary.warning_time_p50_ns, Some(80));
    }

    #[test]
    fn cohort_rejects_forged_detection_below_warning_or_beyond_exposure() {
        let mut too_late = score_shot_alarms(
            "too-late",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("shot score");
        too_late.lead_time_ns = Some(10);
        assert!(summarize_alarm_evaluations(&[too_late]).is_err());

        let mut impossible_exposure = score_shot_alarms(
            "impossible-exposure",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("shot score");
        impossible_exposure.exposure_duration_ns = 50;
        assert!(summarize_alarm_evaluations(&[impossible_exposure]).is_err());
    }

    #[test]
    fn false_alarm_rate_is_normalized_by_negative_shot_exposure() {
        let negative = score_shot_alarms(
            "one-hour-negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 3_600_000_000_000,
            },
            &[10, 20],
            &config(),
        )
        .expect("negative shot");
        let summary = summarize_alarm_evaluations(&[negative]).expect("summary");
        assert_eq!(summary.total_non_disruptive_exposure_ns, 3_600_000_000_000);
        assert_eq!(summary.false_alarms_per_non_disruptive_hour, Some(2.0));
    }

    #[test]
    fn cohort_refuses_to_mix_frozen_policy_identities() {
        let mut other = config();
        other.policy_id = "different-frozen-policy-v2".to_owned();
        other.validation_receipt_sha256 = "b".repeat(64);
        let first = score_shot_alarms(
            "shot-a",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("first policy");
        let second = score_shot_alarms(
            "shot-b",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &other,
        )
        .expect("second policy");
        assert!(summarize_alarm_evaluations(&[first, second]).is_err());
    }

    #[test]
    fn no_negative_shots_means_false_alarm_rate_is_not_reported() {
        let positive = score_shot_alarms(
            "positive",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("positive shot");
        let summary = summarize_alarm_evaluations(&[positive]).expect("summary");
        assert_eq!(summary.false_alarm_shot_rate, None);
        assert_eq!(summary.false_alarm_shot_rate_wilson_95, None);
        assert_eq!(summary.false_alarms_per_non_disruptive_shot, None);
        assert_eq!(summary.false_alarms_per_non_disruptive_hour, None);
        assert_eq!(summary.total_non_disruptive_exposure_ns, 0);
    }

    #[test]
    fn risk_evidence_is_bound_to_frame_model_split_calibration_and_horizon() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        let (frame, requirements) = observation_frame(measured_disposition(10), 10);

        let mut wrong_frame_digest = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_frame_digest.input_frame_sha256 = "f".repeat(64);
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_frame_digest))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_artifact = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_artifact.model_artifact_sha256 = "f".repeat(64);
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_artifact))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_model = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_model.model_id = "different-model".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_model))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_split = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_split.training_split_sha256 = "e".repeat(64);
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_split))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_calibration = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_calibration.calibration_id = "different-calibration".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_calibration))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_calibration_revision = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_calibration_revision.calibration_revision = "rev-2".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_calibration_revision))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_uncertainty_method = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_uncertainty_method.uncertainty_method_id = "unvalidated-method".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_uncertainty_method))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_nominal_coverage = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_nominal_coverage.nominal_coverage = 0.95;
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_nominal_coverage))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_scale = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_scale.score_scale_id = "another-scale".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_scale))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_target = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_target.target_time_ns += 1;
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_target))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut wrong_schema = risk_evidence(&frame, 0.9, 0.9, 0.9);
        wrong_schema.schema_version = "unversioned-risk-score".to_owned();
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(wrong_schema))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut invalid_coverage = risk_evidence(&frame, 0.9, 0.9, 0.9);
        invalid_coverage.nominal_coverage = 1.0;
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(invalid_coverage))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);

        let mut invalid_interval = risk_evidence(&frame, 0.9, 0.95, 0.8);
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(invalid_interval.clone()))
            .is_err());
        invalid_interval.lower_bound = f64::NAN;
        assert!(policy
            .update_from_frame(&frame, &requirements, Some(invalid_interval))
            .is_err());
        assert_eq!(policy.last_window_end_ns, None);
    }

    #[test]
    fn uncertain_interval_breaks_trigger_persistence_without_raising() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");

        let (frame_10, requirements) = observation_frame(measured_disposition(10), 10);
        policy
            .update_from_frame(
                &frame_10,
                &requirements,
                Some(risk_evidence(&frame_10, 0.9, 0.9, 0.95)),
            )
            .expect("first high-confidence window");
        let (frame_20, requirements) = observation_frame(measured_disposition(20), 20);
        let evidence = risk_evidence(&frame_20, 0.85, 0.7, 0.9);
        let expected_evidence_digest = evidence.evidence_sha256().expect("evidence digest");
        let uncertain = policy
            .update_from_frame(&frame_20, &requirements, Some(evidence))
            .expect("uncertain interval is a valid evidence record");
        assert!(matches!(
            uncertain,
            AlarmPolicyUpdate::Uncertain {
                at_ns: 20,
                alert_remains_active: false,
                risk_evidence_sha256,
                ..
            } if risk_evidence_sha256 == expected_evidence_digest
        ));
        assert!(!policy.is_active());

        let (frame_30, requirements) = observation_frame(measured_disposition(30), 30);
        assert!(matches!(
            policy
                .update_from_frame(
                    &frame_30,
                    &requirements,
                    Some(risk_evidence(&frame_30, 0.92, 0.85, 0.97)),
                )
                .expect("high-confidence forecast"),
            AlarmPolicyUpdate::PendingTrigger {
                consecutive_windows: 1,
                ..
            }
        ));
    }

    #[test]
    fn uncertain_interval_does_not_clear_an_active_alarm() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        for time_ns in [10, 20] {
            let (frame, requirements) =
                observation_frame(measured_disposition(time_ns), time_ns);
            policy
                .update_from_frame(
                    &frame,
                    &requirements,
                    Some(risk_evidence(&frame, 0.95, 0.9, 1.0)),
                )
                .expect("high-confidence forecast");
        }
        assert!(policy.is_active());

        let (frame_30, requirements) = observation_frame(measured_disposition(30), 30);
        let uncertain = policy
            .update_from_frame(
                &frame_30,
                &requirements,
                Some(risk_evidence(&frame_30, 0.25, 0.2, 0.4)),
            )
            .expect("uncertain interval should preserve active alarm");
        assert!(matches!(
            uncertain,
            AlarmPolicyUpdate::Uncertain {
                at_ns: 30,
                alert_remains_active: true,
                ..
            }
        ));
        assert!(policy.is_active());

        let (frame_40, requirements) = observation_frame(measured_disposition(40), 40);
        assert!(matches!(
            policy
                .update_from_frame(
                    &frame_40,
                    &requirements,
                    Some(risk_evidence(&frame_40, 0.2, 0.1, 0.25)),
                )
                .expect("high-confidence clear evidence"),
            AlarmPolicyUpdate::PendingClear {
                consecutive_windows: 1,
                ..
            }
        ));
    }

    #[test]
    fn risk_evidence_digest_changes_with_model_split_calibration_and_interval() {
        let (frame, _) = observation_frame(measured_disposition(10), 10);
        let original = risk_evidence(&frame, 0.9, 0.8, 0.95);
        let original_digest = original.evidence_sha256().expect("risk evidence digest");

        let mut changed_scale = original.clone();
        changed_scale.score_scale_id = "alternate-risk-score-v2".to_owned();
        assert_ne!(
            original_digest,
            changed_scale.evidence_sha256().expect("changed digest")
        );

        let mut changed_model = original.clone();
        changed_model.model_revision = "test-model-v2".to_owned();
        assert_ne!(
            original_digest,
            changed_model.evidence_sha256().expect("changed digest")
        );

        let mut changed_split = original.clone();
        changed_split.training_split_sha256 = "4".repeat(64);
        assert_ne!(
            original_digest,
            changed_split.evidence_sha256().expect("changed digest")
        );

        let mut changed_calibration = original.clone();
        changed_calibration.calibration_revision = "cal-v2".to_owned();
        assert_ne!(
            original_digest,
            changed_calibration.evidence_sha256().expect("changed digest")
        );

        let mut changed_interval = original.clone();
        changed_interval.upper_bound = 0.99;
        assert_ne!(
            original_digest,
            changed_interval.evidence_sha256().expect("changed digest")
        );

        let mut changed_uncertainty_method = original;
        changed_uncertainty_method.uncertainty_method_id = "ensemble-quantile-v2".to_owned();
        assert_ne!(
            original_digest,
            changed_uncertainty_method
                .evidence_sha256()
                .expect("changed digest")
        );
    }

    #[test]
    fn duplicate_shots_and_out_of_horizon_alarms_are_rejected() {
        let positive = score_shot_alarms(
            "same",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            &config(),
        )
        .expect("positive shot");
        assert!(summarize_alarm_evaluations(&[positive.clone(), positive]).is_err());
        assert!(score_shot_alarms(
            "late",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[101],
            &config(),
        )
        .is_err());
    }
}
