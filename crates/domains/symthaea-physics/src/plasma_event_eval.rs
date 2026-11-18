//! Leakage-resistant causal plasma disruption alarm policy and event metrics.
//!
//! The online policy accepts one timestamped observation at a time. It has no
//! event-label, disruption-time, future-window, or full-shot-mean input.
//! Event references are used only by the separate post-hoc scorer.
//!
//! This module does not calibrate physical probabilities, select thresholds
//! from a test set, or authorize machine actions.

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
pub enum AlarmEvidenceQuality {
    DirectMeasured,
    Synthetic,
    Unavailable,
    Invalid,
    Delayed,
    Degraded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlarmObservation {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlarmAbstentionReason {
    Synthetic,
    Unavailable,
    Invalid,
    Delayed,
    Degraded,
}

fn abstention_reason(quality: AlarmEvidenceQuality) -> Option<AlarmAbstentionReason> {
    match quality {
        AlarmEvidenceQuality::DirectMeasured => None,
        AlarmEvidenceQuality::Synthetic => Some(AlarmAbstentionReason::Synthetic),
        AlarmEvidenceQuality::Unavailable => Some(AlarmAbstentionReason::Unavailable),
        AlarmEvidenceQuality::Invalid => Some(AlarmAbstentionReason::Invalid),
        AlarmEvidenceQuality::Delayed => Some(AlarmAbstentionReason::Delayed),
        AlarmEvidenceQuality::Degraded => Some(AlarmAbstentionReason::Degraded),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Processes exactly one observation in time order.
    ///
    /// Non-direct evidence causes abstention and breaks persistence continuity.
    /// An active alarm remains active across abstention: missing evidence cannot
    /// silently clear an existing alert.
    pub fn update(
        &mut self,
        observation: &AlarmObservation,
    ) -> Result<AlarmPolicyUpdate, PlasmaEventEvalError> {
        let current_shot = self
            .shot_id
            .as_deref()
            .ok_or_else(|| PlasmaEventEvalError::new("reset_for_shot must be called first"))?;
        if observation.shot_id != current_shot {
            return Err(PlasmaEventEvalError::new(
                "observation shot_id does not match the active discharge",
            ));
        }
        if let Some(previous) = self.last_window_end_ns
            && observation.window_end_ns <= previous
        {
            return Err(PlasmaEventEvalError::new(
                "window_end_ns must be strictly increasing",
            ));
        }

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

        if let Some(reason) = abstention_reason(observation.evidence_quality) {
            self.high_run = 0;
            self.low_run = 0;
            return Ok(AlarmPolicyUpdate::Abstained {
                at_ns: observation.window_end_ns,
                reason,
                alert_remains_active: self.active,
            });
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
    pub warning_time_p05_ns: Option<u64>,
    pub warning_time_p10_ns: Option<u64>,
    pub warning_time_p50_ns: Option<u64>,
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
        match (&policy_id, &validation_receipt_sha256, required_warning_ns) {
            (None, None, None) => {
                policy_id = Some(evaluation.policy_id.clone());
                validation_receipt_sha256 = Some(evaluation.validation_receipt_sha256.clone());
                required_warning_ns = Some(evaluation.required_warning_ns);
            }
            (Some(id), Some(digest), Some(warning))
                if id == &evaluation.policy_id
                    && digest == &evaluation.validation_receipt_sha256
                    && warning == evaluation.required_warning_ns => {}
            _ => {
                return Err(PlasmaEventEvalError::new(
                    "one summary cannot mix alarm policy identities or warning requirements",
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
        false_alarm_shot_rate: (non_disruptive_shots > 0)
            .then_some(false_alarm_shots as f64 / non_disruptive_shots as f64),
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
            trigger_threshold: 0.8,
            clear_threshold: 0.3,
            trigger_persistence_windows: 2,
            clear_persistence_windows: 2,
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
        policy.update(&measured("shot-1", 10, 0.9)).expect("update");
        let outcome = policy
            .update(&AlarmObservation {
                shot_id: "shot-1".to_owned(),
                window_end_ns: 20,
                risk_score: None,
                evidence_quality: AlarmEvidenceQuality::Synthetic,
            })
            .expect("synthetic evidence should abstain");
        assert_eq!(
            outcome,
            AlarmPolicyUpdate::Abstained {
                at_ns: 20,
                reason: AlarmAbstentionReason::Synthetic,
                alert_remains_active: false,
            }
        );
        assert!(matches!(
            policy.update(&measured("shot-1", 30, 0.9)).expect("update"),
            AlarmPolicyUpdate::PendingTrigger {
                consecutive_windows: 1,
                ..
            }
        ));
    }

    #[test]
    fn non_measured_evidence_cannot_smuggle_a_numeric_score() {
        let mut policy = OnlinePlasmaAlarmPolicy::new(config()).expect("valid policy");
        policy.reset_for_shot("shot-1").expect("valid shot");
        let observation = AlarmObservation {
            shot_id: "shot-1".to_owned(),
            window_end_ns: 10,
            risk_score: Some(0.99),
            evidence_quality: AlarmEvidenceQuality::Synthetic,
        };
        assert!(policy.update(&observation).is_err());
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
            50,
        )
        .expect("valid shot evaluation");
        assert!(detected.event_detected);
        assert_eq!(detected.lead_time_ns, Some(80));

        let too_late = score_shot_alarms(
            "shot-late",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[80],
            50,
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
            50,
        )
        .expect("positive shot");
        let negative = score_shot_alarms(
            "negative",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[20, 70],
            50,
        )
        .expect("negative shot");
        let summary = summarize_alarm_evaluations(&[positive, negative]).expect("summary");
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
    fn no_negative_shots_means_false_alarm_rate_is_not_reported() {
        let positive = score_shot_alarms(
            "positive",
            ShotReferenceOutcome::Disruptive { t_cut_ns: 100 },
            &[20],
            50,
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
            50,
        )
        .expect("positive shot");
        assert!(summarize_alarm_evaluations(&[positive.clone(), positive]).is_err());
        assert!(score_shot_alarms(
            "late",
            ShotReferenceOutcome::NonDisruptive {
                observation_end_ns: 100,
            },
            &[101],
            50,
        )
        .is_err());
    }
}
