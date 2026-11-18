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
    pub evidence_quality: AlarmEvidenceQuality,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlarmPolicyConfig {
    pub policy_id: String,
    /// SHA-256 content identity for the frozen validation/calibration receipt.
    pub validation_receipt_sha256: String,
    pub trigger_threshold: f64,
    pub clear_threshold: f64,
    pub trigger_persistence_windows: usize,
    pub clear_persistence_windows: usize,
    pub required_warning_ns: u64,
}

impl AlarmPolicyConfig {
    pub fn validate(&self) -> Result<(), PlasmaEventEvalError> {
        if self.policy_id.trim().is_empty() {
            return Err(PlasmaEventEvalError::new("policy_id must be non-empty"));
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
        if self.required_warning_ns == 0 {
            return Err(PlasmaEventEvalError::new("required_warning_ns must be positive"));
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

    /// SHA-256 over the exact policy ID, validation receipt, thresholds,
    /// persistence settings and warning interval. Threshold floats are
    /// represented by their IEEE-754 bit patterns for canonical identity.
    pub fn policy_config_sha256(&self) -> String {
        let canonical = format!(
            "luminous.plasma-alarm-policy.v0.1\\n{}\\n{}\\n{:016x}\\n{:016x}\\n{}\\n{}\\n{}",
            self.policy_id,
            self.validation_receipt_sha256,
            self.trigger_threshold.to_bits(),
            self.clear_threshold.to_bits(),
            self.trigger_persistence_windows,
            self.clear_persistence_windows,
            self.required_warning_ns,
        );
        let digest = Sha256::digest(canonical.as_bytes());
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut encoded, "{byte:02x}");
        }
        encoded
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmAbstentionReason {
    EvidenceGateBlocked(Vec<BlockedChannel>),
}



#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmPolicyUpdate {
    BelowThreshold {
        at_ns: u64,
    },
    PendingTrigger {
        at_ns: u64,
        consecutive_windows: usize,
    },
    Raised {
        at_ns: u64,
    },
    Active {
        at_ns: u64,
    },
    PendingClear {
        at_ns: u64,
        consecutive_windows: usize,
    },
    Cleared {
        at_ns: u64,
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
        risk_score: Option<f64>,
    ) -> Result<AlarmPolicyUpdate, PlasmaEventEvalError> {
        let gate = frame
            .evaluate_decision_requirements(requirements)
            .map_err(|error| PlasmaEventEvalError::new(error.message))?;
        match gate {
            DecisionInputGate::MeasuredInputsAvailable(inputs) => {
                let score = risk_score.ok_or_else(|| {
                    PlasmaEventEvalError::new(
                        "a qualified measured frame requires a finite risk score",
                    )
                })?;
                let observation = AlarmObservation {
                    shot_id: inputs.shot_id().to_owned(),
                    window_end_ns: inputs.frame_time_ns(),
                    risk_score: Some(score),
                    evidence_quality: AlarmEvidenceQuality::DirectMeasured,
                };
                self.update(&observation)
            }
            DecisionInputGate::Abstain {
                frame_id: _,
                blocked_channels,
            } => {
                if risk_score.is_some() {
                    return Err(PlasmaEventEvalError::new(
                        "a blocked evidence frame must not carry an alarm risk score",
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

        match (observation.evidence_quality, observation.risk_score) {
            (AlarmEvidenceQuality::DirectMeasured, Some(score)) if score.is_finite() => {}
            (AlarmEvidenceQuality::DirectMeasured, _) => {
                return Err(PlasmaEventEvalError::new(
                    "direct-measured observations require a finite risk_score",
                ));
            }
            (_, None) => {}
            (_, Some(_)) => {
                return Err(PlasmaEventEvalError::new(
                    "non-measured evidence must not carry a policy risk_score",
                ));
            }
        }

        self.last_window_end_ns = Some(observation.window_end_ns);

        if observation.evidence_quality != AlarmEvidenceQuality::DirectMeasured {
            return Err(PlasmaEventEvalError::new(
                "internal alarm state machine accepts only gate-qualified measured input",
            ));
        }

        let score = observation
            .risk_score
            .ok_or_else(|| PlasmaEventEvalError::new("validated risk_score disappeared"))?;

        if !self.active {
            self.low_run = 0;
            if score >= self.config.trigger_threshold {
                self.high_run = self.high_run.saturating_add(1);
                if self.high_run >= self.config.trigger_persistence_windows {
                    self.active = true;
                    self.high_run = 0;
                    return Ok(AlarmPolicyUpdate::Raised {
                        at_ns: observation.window_end_ns,
                    });
                }
                return Ok(AlarmPolicyUpdate::PendingTrigger {
                    at_ns: observation.window_end_ns,
                    consecutive_windows: self.high_run,
                });
            }
            self.high_run = 0;
            return Ok(AlarmPolicyUpdate::BelowThreshold {
                at_ns: observation.window_end_ns,
            });
        }

        self.high_run = 0;
        if score <= self.config.clear_threshold {
            self.low_run = self.low_run.saturating_add(1);
            if self.low_run >= self.config.clear_persistence_windows {
                self.active = false;
                self.low_run = 0;
                return Ok(AlarmPolicyUpdate::Cleared {
                    at_ns: observation.window_end_ns,
                });
            }
            return Ok(AlarmPolicyUpdate::PendingClear {
                at_ns: observation.window_end_ns,
                consecutive_windows: self.low_run,
            });
        }

        self.low_run = 0;
        Ok(AlarmPolicyUpdate::Active {
            at_ns: observation.window_end_ns,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotReferenceOutcome {
    Disruptive { t_cut_ns: u64 },
    NonDisruptive { observation_end_ns: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShotAlarmEvaluation {
    pub shot_id: String,
    pub policy_id: String,
    pub validation_receipt_sha256: String,
    pub policy_config_sha256: String,
    pub required_warning_ns: u64,
    pub disruptive: bool,
    pub event_detected: bool,
    pub false_alarm_events: usize,
    pub lead_time_ns: Option<u64>,
}

/// Scores timestamps already emitted by the online policy. Event references
/// are used only here, after the online alarm trace has been recorded.
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
            if alarm_times_ns.iter().any(|time| *time > t_cut_ns) {
                return Err(PlasmaEventEvalError::new(
                    "disruptive-shot alarm trace extends beyond t_cut_ns",
                ));
            }
            let lead_time_ns = alarm_times_ns
                .first()
                .copied()
                .filter(|time| *time < t_cut_ns)
                .map(|time| t_cut_ns - time)
                .filter(|lead| *lead >= policy.required_warning_ns);
            Ok(ShotAlarmEvaluation {
                shot_id: shot_id.to_owned(),
                policy_id: policy.policy_id.clone(),
                validation_receipt_sha256: policy.validation_receipt_sha256.clone(),
                policy_config_sha256: policy.policy_config_sha256(),
                required_warning_ns: policy.required_warning_ns,
                disruptive: true,
                event_detected: lead_time_ns.is_some(),
                false_alarm_events: 0,
                lead_time_ns,
            })
        }
        ShotReferenceOutcome::NonDisruptive {
            observation_end_ns,
        } => {
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
                disruptive: false,
                event_detected: false,
                false_alarm_events: alarm_times_ns.len(),
                lead_time_ns: None,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlarmEvaluationSummary {
    pub policy_id: String,
    pub validation_receipt_sha256: String,
    pub policy_config_sha256: String,
    pub required_warning_ns: u64,
    pub shot_count: usize,
    pub disruptive_shots: usize,
    pub non_disruptive_shots: usize,
    pub detected_disruptions: usize,
    pub missed_disruptions: usize,
    pub false_alarm_shots: usize,
    pub false_alarm_events: usize,
    /// None when there are no disruptive shots.
    pub event_tpr: Option<f64>,
    /// None when there are no non-disruptive shots.
    pub false_alarm_shot_rate: Option<f64>,
    /// None when there are no non-disruptive shots.
    pub false_alarms_per_non_disruptive_shot: Option<f64>,
    pub mean_warning_time_ns: Option<f64>,
    pub event_tpr_wilson_95: Option<RateConfidenceInterval>,
    pub false_alarm_shot_rate_wilson_95: Option<RateConfidenceInterval>,
    pub warning_time_p05_ns: Option<u64>,
    pub warning_time_p10_ns: Option<u64>,
    pub warning_time_p50_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    let mut disruptive_shots = 0usize;
    let mut non_disruptive_shots = 0usize;
    let mut detected_disruptions = 0usize;
    let mut false_alarm_shots = 0usize;
    let mut false_alarm_events = 0usize;
    let mut warning_times_ns = Vec::new();

    for evaluation in evaluations {
        if evaluation.shot_id.trim().is_empty() || !shot_ids.insert(&evaluation.shot_id) {
            return Err(PlasmaEventEvalError::new(
                "shot IDs must be non-empty and unique in one evaluation cohort",
            ));
        }
        match (
            &policy_id,
            &validation_receipt_sha256,
            &policy_config_sha256,
            required_warning_ns,
        ) {
            (None, None, None, None) => {
                policy_id = Some(evaluation.policy_id.clone());
                validation_receipt_sha256 = Some(evaluation.validation_receipt_sha256.clone());
                policy_config_sha256 = Some(evaluation.policy_config_sha256.clone());
                required_warning_ns = Some(evaluation.required_warning_ns);
            }
            (Some(id), Some(receipt), Some(config), Some(warning))
                if id == &evaluation.policy_id
                    && receipt == &evaluation.validation_receipt_sha256
                    && config == &evaluation.policy_config_sha256
                    && warning == evaluation.required_warning_ns => {}
            _ => {
                return Err(PlasmaEventEvalError::new(
                    "one summary cannot mix alarm policy identities, config digests or warning requirements",
                ));
            }
        }

        if evaluation.disruptive {
            disruptive_shots += 1;
            if evaluation.event_detected != evaluation.lead_time_ns.is_some()
                || evaluation.false_alarm_events != 0
            {
                return Err(PlasmaEventEvalError::new(
                    "disruptive shot evaluation has inconsistent event metrics",
                ));
            }
            if evaluation.event_detected {
                detected_disruptions += 1;
                warning_times_ns.push(evaluation.lead_time_ns.ok_or_else(|| {
                    PlasmaEventEvalError::new("detected event is missing lead time")
                })?);
            }
        } else {
            non_disruptive_shots += 1;
            if evaluation.event_detected || evaluation.lead_time_ns.is_some() {
                return Err(PlasmaEventEvalError::new(
                    "non-disruptive shot cannot be marked as an event detection",
                ));
            }
            if evaluation.false_alarm_events > 0 {
                false_alarm_shots += 1;
            }
            false_alarm_events =
                false_alarm_events.saturating_add(evaluation.false_alarm_events);
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
        shot_count: evaluations.len(),
        disruptive_shots,
        non_disruptive_shots,
        detected_disruptions,
        missed_disruptions,
        false_alarm_shots,
        false_alarm_events,
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
            .then_some(false_alarm_events as f64 / non_disruptive_shots as f64),
        mean_warning_time_ns,
        warning_time_p05_ns: nearest_rank_quantile(&warning_times_ns, 0.05),
        warning_time_p10_ns: nearest_rank_quantile(&warning_times_ns, 0.10),
        warning_time_p50_ns: nearest_rank_quantile(&warning_times_ns, 0.50),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AlarmPolicyConfig {
        AlarmPolicyConfig {
            policy_id: "fixed-test-policy-v1".to_owned(),
            validation_receipt_sha256: "a".repeat(64),
            trigger_threshold: 0.8,
            clear_threshold: 0.3,
            trigger_persistence_windows: 2,
            clear_persistence_windows: 2,
            required_warning_ns: 50,
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
            manifest["online_policy_contract"]["test_threshold_optimization"],
            false
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
            manifest["target_gates"]["all_values_are_research_targets"],
            true
        );
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
            AlarmPolicyUpdate::Cleared { at_ns: 60 }
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
            .update_from_frame(&measured_frame, &requirements, Some(0.9))
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
                .update_from_frame(&measured_frame, &requirements, Some(0.9))
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
        policy.update_from_frame(&frame_10, &requirements, Some(0.9)).expect("update");
        let (frame_20, requirements) = observation_frame(measured_disposition(20), 20);
        policy.update_from_frame(&frame_20, &requirements, Some(0.9)).expect("update");
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
            .update_from_frame(&frame, &requirements, Some(0.99))
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
        assert_eq!(summary.required_warning_ns, 50);
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
        assert_eq!(summary.false_alarms_per_non_disruptive_shot, None);
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
