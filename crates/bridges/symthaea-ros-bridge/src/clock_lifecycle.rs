//! Fail-closed lifecycle tracking for a ROS or externally supplied clock.
//!
//! This is deliberately a transport-independent state machine. It does not
//! subscribe to ROS \`/clock\` or claim a clock source is authentic. A qualified
//! adapter must provide the source identity, timestamp, source epoch, and local
//! monotonic receipt metadata. The tracker detects discontinuities and requires
//! an explicit rebind before a new timeline can be used for temporal admission.

use serde::{Deserialize, Serialize};

/// A sample of a source clock plus its local, monotonic reception evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosClockObservationV1 {
    /// Deployment-configured identity, e.g. \`ros:/clock\` or a device clock ID.
    pub clock_id: String,
    /// Semantic domain, e.g. \`ros-sim-time\`, \`system-time\`, or \`device-ptp\`.
    pub clock_domain: String,
    /// Adapter-observed generation/session ID for the source timeline.
    pub source_epoch_id: String,
    /// Source-clock time, in nanoseconds in that domain.
    pub timestamp_ns: u64,
    /// Receipt time in the bridge's local monotonic time base.
    pub received_monotonic_ns: u64,
    /// Local monotonic epoch to which \`received_monotonic_ns\` belongs.
    pub receive_epoch_id: String,
}

/// Thresholds are part of the frozen deployment profile, not auto-tuned from data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosClockLifecyclePolicyV1 {
    /// Maximum residence of a clock observation when evaluated at capture time.
    pub maximum_observation_age_ns: u64,
    /// Forward source-time movement larger than this requires explicit rebind.
    pub maximum_forward_jump_ns: u64,
    /// Repeated source time for at least this much local monotonic time is a pause.
    pub pause_after_ns: u64,
    /// ROS time zero is uninitialized; deployments with other semantics must opt out.
    pub reject_zero_timestamp: bool,
}

impl RosClockLifecyclePolicyV1 {
    fn validate(&self) -> Result<(), String> {
        if self.maximum_observation_age_ns == 0 {
            return Err("maximum observation age must be greater than zero".into());
        }
        if self.maximum_forward_jump_ns == 0 {
            return Err("maximum forward jump must be greater than zero".into());
        }
        if self.pause_after_ns == 0 {
            return Err("pause detection threshold must be greater than zero".into());
        }
        Ok(())
    }
}

/// Why a previously usable timeline needs an explicit lifecycle decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosClockTransitionKindV1 {
    BackwardJump,
    ForwardJump,
    UninitializedAfterInitialization,
    ClockIdentityChanged,
    ClockDomainChanged,
    SourceEpochChanged,
    ReceiveEpochChanged,
}

/// Auditable transition evidence. The tracker will not self-clear this condition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosClockTransitionV1 {
    pub kind: RosClockTransitionKindV1,
    pub previous_generation: u64,
    pub previous_timestamp_ns: u64,
    pub observed_timestamp_ns: u64,
    pub observed_clock_id: String,
    pub observed_clock_domain: String,
    pub observed_source_epoch_id: String,
    pub observed_receive_epoch_id: String,
}

/// Result of observing the source clock. Only \`Advancing\` is temporally admissible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosClockLifecycleDispositionV1 {
    Uninitialized,
    BaselineEstablished {
        generation: u64,
        timestamp_ns: u64,
    },
    Advancing {
        generation: u64,
        delta_ns: u64,
    },
    NotAdvancing {
        generation: u64,
        elapsed_since_progress_ns: u64,
        paused: bool,
    },
    ClockObservationStale {
        age_ns: u64,
        maximum_age_ns: u64,
    },
    TransitionRequired {
        transition: RosClockTransitionV1,
    },
    AwaitingExplicitRebind {
        kind: RosClockTransitionKindV1,
    },
    TransitionRebound {
        kind: RosClockTransitionKindV1,
        generation: u64,
        timestamp_ns: u64,
    },
    InvalidPolicy {
        reason: String,
    },
    InvalidObservation {
        reason: String,
    },
}

impl RosClockLifecycleDispositionV1 {
    /// A conservative gate: a baseline, pause, jump, or stale clock is not
    /// enough to authorize source-age/skew checks on downstream sensor data.
    pub fn permits_temporal_admission(&self) -> bool {
        matches!(self, Self::Advancing { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ClockBaselineV1 {
    clock_id: String,
    clock_domain: String,
    source_epoch_id: String,
    receive_epoch_id: String,
    timestamp_ns: u64,
    last_observation_received_monotonic_ns: u64,
    last_progress_received_monotonic_ns: u64,
    generation: u64,
}

/// Tracks clock progression separately from each sensor stream's temporal history.
///
/// A detected time jump is a hard gate. Callers must explicitly acknowledge the
/// event, reset/rebind downstream histories, and use the returned generation as
/// part of the source-clock epoch identity. Sensor samples never trigger resets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosClockLifecycleTrackerV1 {
    policy: RosClockLifecyclePolicyV1,
    baseline: Option<ClockBaselineV1>,
    pending_transition: Option<RosClockTransitionV1>,
}

impl RosClockLifecycleTrackerV1 {
    pub fn new(
        policy: RosClockLifecyclePolicyV1,
    ) -> Result<Self, RosClockLifecycleDispositionV1> {
        policy
            .validate()
            .map_err(|reason| RosClockLifecycleDispositionV1::InvalidPolicy { reason })?;
        Ok(Self {
            policy,
            baseline: None,
            pending_transition: None,
        })
    }

    pub fn generation(&self) -> Option<u64> {
        self.baseline.as_ref().map(|baseline| baseline.generation)
    }

    pub fn pending_transition(&self) -> Option<&RosClockTransitionV1> {
        self.pending_transition.as_ref()
    }

    /// Observe a clock sample. \`capture_receive_epoch_id\` and
    /// \`captured_monotonic_ns\` must come from the same local monotonic clock
    /// used to stamp callback receipts.
    pub fn observe(
        &mut self,
        observation: &RosClockObservationV1,
        capture_receive_epoch_id: &str,
        captured_monotonic_ns: u64,
    ) -> RosClockLifecycleDispositionV1 {
        if let Err(reason) = self.policy.validate() {
            return RosClockLifecycleDispositionV1::InvalidPolicy { reason };
        }
        if let Some(transition) = self.pending_transition.as_ref() {
            return RosClockLifecycleDispositionV1::AwaitingExplicitRebind {
                kind: transition.kind.clone(),
            };
        }
        if let Err(reason) = validate_identity(observation) {
            return RosClockLifecycleDispositionV1::InvalidObservation { reason };
        }
        if observation.received_monotonic_ns > captured_monotonic_ns {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "clock receipt timestamp postdates capture".into(),
            };
        }
        if observation.receive_epoch_id != capture_receive_epoch_id {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::ReceiveEpochChanged,
                "clock receipt epoch differs from capture epoch",
            );
        }
        let age_ns = captured_monotonic_ns - observation.received_monotonic_ns;
        if age_ns > self.policy.maximum_observation_age_ns {
            return RosClockLifecycleDispositionV1::ClockObservationStale {
                age_ns,
                maximum_age_ns: self.policy.maximum_observation_age_ns,
            };
        }
        if self.policy.reject_zero_timestamp && observation.timestamp_ns == 0 {
            if self.baseline.is_some() {
                return self.require_transition_or_reject(
                    observation,
                    RosClockTransitionKindV1::UninitializedAfterInitialization,
                    "clock returned to uninitialized zero after initialization",
                );
            }
            return RosClockLifecycleDispositionV1::Uninitialized;
        }

        let Some(baseline) = self.baseline.as_ref() else {
            self.baseline = Some(baseline_from_observation(observation, 1));
            return RosClockLifecycleDispositionV1::BaselineEstablished {
                generation: 1,
                timestamp_ns: observation.timestamp_ns,
            };
        };

        if observation.clock_id != baseline.clock_id {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::ClockIdentityChanged,
                "clock identity changed",
            );
        }
        if observation.clock_domain != baseline.clock_domain {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::ClockDomainChanged,
                "clock domain changed",
            );
        }
        if observation.source_epoch_id != baseline.source_epoch_id {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::SourceEpochChanged,
                "source clock epoch changed",
            );
        }
        if observation.receive_epoch_id != baseline.receive_epoch_id {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::ReceiveEpochChanged,
                "local monotonic receive epoch changed",
            );
        }
        if observation.received_monotonic_ns
            <= baseline.last_observation_received_monotonic_ns
        {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "clock receipt time is replayed or not strictly increasing".into(),
            };
        }

        let previous_timestamp_ns = baseline.timestamp_ns;
        if observation.timestamp_ns < previous_timestamp_ns {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::BackwardJump,
                "source clock moved backwards",
            );
        }

        let delta_ns = observation.timestamp_ns - previous_timestamp_ns;
        if delta_ns > self.policy.maximum_forward_jump_ns {
            return self.require_transition_or_reject(
                observation,
                RosClockTransitionKindV1::ForwardJump,
                "source clock advanced beyond the frozen jump threshold",
            );
        }

        if delta_ns == 0 {
            let baseline = self.baseline.as_mut().expect("baseline was checked above");
            baseline.last_observation_received_monotonic_ns =
                observation.received_monotonic_ns;
            let elapsed_since_progress_ns = captured_monotonic_ns
                .checked_sub(baseline.last_progress_received_monotonic_ns)
                .unwrap_or(0);
            return RosClockLifecycleDispositionV1::NotAdvancing {
                generation: baseline.generation,
                elapsed_since_progress_ns,
                paused: elapsed_since_progress_ns >= self.policy.pause_after_ns,
            };
        }

        let baseline = self.baseline.as_mut().expect("baseline was checked above");
        baseline.timestamp_ns = observation.timestamp_ns;
        baseline.last_observation_received_monotonic_ns = observation.received_monotonic_ns;
        baseline.last_progress_received_monotonic_ns = observation.received_monotonic_ns;
        RosClockLifecycleDispositionV1::Advancing {
            generation: baseline.generation,
            delta_ns,
        }
    }

    /// Acknowledge a previously reported transition only after the caller has
    /// decided that downstream state may be reset/rebound. The new observation
    /// must still be fresh and belong to the transition's observed source identity.
    pub fn acknowledge_transition(
        &mut self,
        observation: &RosClockObservationV1,
        capture_receive_epoch_id: &str,
        captured_monotonic_ns: u64,
    ) -> RosClockLifecycleDispositionV1 {
        let Some(transition) = self.pending_transition.as_ref().cloned() else {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "no pending clock transition to acknowledge".into(),
            };
        };
        if let Err(reason) = self.policy.validate() {
            return RosClockLifecycleDispositionV1::InvalidPolicy { reason };
        }
        if let Err(reason) = validate_identity(observation) {
            return RosClockLifecycleDispositionV1::InvalidObservation { reason };
        }
        if self.policy.reject_zero_timestamp && observation.timestamp_ns == 0 {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "cannot rebind to an uninitialized zero timestamp".into(),
            };
        }
        if observation.clock_id != transition.observed_clock_id
            || observation.clock_domain != transition.observed_clock_domain
            || observation.source_epoch_id != transition.observed_source_epoch_id
        {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "rebind observation does not match the pending transition's source identity".into(),
            };
        }
        if observation.receive_epoch_id != capture_receive_epoch_id {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "rebind receipt epoch differs from capture epoch".into(),
            };
        }
        if observation.receive_epoch_id != transition.observed_receive_epoch_id {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "rebind receipt epoch differs from the pending transition".into(),
            };
        }
        if observation.received_monotonic_ns > captured_monotonic_ns {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: "rebind receipt timestamp postdates capture".into(),
            };
        }
        let age_ns = captured_monotonic_ns - observation.received_monotonic_ns;
        if age_ns > self.policy.maximum_observation_age_ns {
            return RosClockLifecycleDispositionV1::ClockObservationStale {
                age_ns,
                maximum_age_ns: self.policy.maximum_observation_age_ns,
            };
        }
        if let Some(previous) = self.baseline.as_ref() {
            if observation.received_monotonic_ns
                <= previous.last_observation_received_monotonic_ns
                && observation.receive_epoch_id == previous.receive_epoch_id
            {
                return RosClockLifecycleDispositionV1::InvalidObservation {
                    reason: "rebind receipt time is not newer than the prior observation".into(),
                };
            }
        }
        let generation = match self
            .baseline
            .as_ref()
            .map(|baseline| baseline.generation)
            .unwrap_or(0)
            .checked_add(1)
        {
            Some(generation) => generation,
            None => {
                return RosClockLifecycleDispositionV1::InvalidObservation {
                    reason: "clock generation counter overflow".into(),
                };
            }
        };
        self.baseline = Some(baseline_from_observation(observation, generation));
        self.pending_transition = None;
        RosClockLifecycleDispositionV1::TransitionRebound {
            kind: transition.kind,
            generation,
            timestamp_ns: observation.timestamp_ns,
        }
    }

    fn require_transition_or_reject(
        &mut self,
        observation: &RosClockObservationV1,
        kind: RosClockTransitionKindV1,
        reason: &str,
    ) -> RosClockLifecycleDispositionV1 {
        let Some(baseline) = self.baseline.as_ref() else {
            return RosClockLifecycleDispositionV1::InvalidObservation {
                reason: format!("{reason}; no initialized baseline exists"),
            };
        };
        let transition = RosClockTransitionV1 {
            kind,
            previous_generation: baseline.generation,
            previous_timestamp_ns: baseline.timestamp_ns,
            observed_timestamp_ns: observation.timestamp_ns,
            observed_clock_id: observation.clock_id.clone(),
            observed_clock_domain: observation.clock_domain.clone(),
            observed_source_epoch_id: observation.source_epoch_id.clone(),
            observed_receive_epoch_id: observation.receive_epoch_id.clone(),
        };
        self.pending_transition = Some(transition.clone());
        RosClockLifecycleDispositionV1::TransitionRequired { transition }
    }
}

fn validate_identity(observation: &RosClockObservationV1) -> Result<(), String> {
    for (field, value) in [
        ("clock_id", observation.clock_id.as_str()),
        ("clock_domain", observation.clock_domain.as_str()),
        ("source_epoch_id", observation.source_epoch_id.as_str()),
        ("receive_epoch_id", observation.receive_epoch_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{field} must not be empty"));
        }
    }
    Ok(())
}

fn baseline_from_observation(
    observation: &RosClockObservationV1,
    generation: u64,
) -> ClockBaselineV1 {
    ClockBaselineV1 {
        clock_id: observation.clock_id.clone(),
        clock_domain: observation.clock_domain.clone(),
        source_epoch_id: observation.source_epoch_id.clone(),
        receive_epoch_id: observation.receive_epoch_id.clone(),
        timestamp_ns: observation.timestamp_ns,
        last_observation_received_monotonic_ns: observation.received_monotonic_ns,
        last_progress_received_monotonic_ns: observation.received_monotonic_ns,
        generation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RosClockLifecyclePolicyV1 {
        RosClockLifecyclePolicyV1 {
            maximum_observation_age_ns: 100,
            maximum_forward_jump_ns: 500,
            pause_after_ns: 20,
            reject_zero_timestamp: true,
        }
    }

    fn observation(timestamp_ns: u64, received_monotonic_ns: u64) -> RosClockObservationV1 {
        RosClockObservationV1 {
            clock_id: "ros:/clock".into(),
            clock_domain: "ros-sim-time".into(),
            source_epoch_id: "bag-run-1".into(),
            timestamp_ns,
            received_monotonic_ns,
            receive_epoch_id: "bridge-run-1".into(),
        }
    }

    fn initialized_tracker() -> RosClockLifecycleTrackerV1 {
        let mut tracker = RosClockLifecycleTrackerV1::new(policy()).unwrap();
        assert_eq!(
            tracker.observe(&observation(1_000, 1), "bridge-run-1", 1),
            RosClockLifecycleDispositionV1::BaselineEstablished {
                generation: 1,
                timestamp_ns: 1_000,
            }
        );
        tracker
    }

    #[test]
    fn zero_is_uninitialized_and_does_not_bind_a_baseline() {
        let mut tracker = RosClockLifecycleTrackerV1::new(policy()).unwrap();
        assert_eq!(
            tracker.observe(&observation(0, 1), "bridge-run-1", 1),
            RosClockLifecycleDispositionV1::Uninitialized
        );
        assert_eq!(tracker.generation(), None);
        assert!(matches!(
            tracker.observe(&observation(10, 2), "bridge-run-1", 2),
            RosClockLifecycleDispositionV1::BaselineEstablished { generation: 1, .. }
        ));
    }

    #[test]
    fn advancing_clock_is_the_only_admissible_state() {
        let mut tracker = initialized_tracker();
        let outcome = tracker.observe(&observation(1_010, 2), "bridge-run-1", 2);
        assert!(matches!(
            outcome,
            RosClockLifecycleDispositionV1::Advancing {
                generation: 1,
                delta_ns: 10
            }
        ));
        assert!(outcome.permits_temporal_admission());
        let repeated = tracker.observe(&observation(1_010, 3), "bridge-run-1", 23);
        assert!(matches!(
            repeated,
            RosClockLifecycleDispositionV1::NotAdvancing { paused: true, .. }
        ));
        assert!(!repeated.permits_temporal_admission());
    }

    #[test]
    fn backward_jump_locks_until_explicit_rebind() {
        let mut tracker = initialized_tracker();
        assert!(matches!(
            tracker.observe(&observation(900, 2), "bridge-run-1", 2),
            RosClockLifecycleDispositionV1::TransitionRequired {
                transition: RosClockTransitionV1 {
                    kind: RosClockTransitionKindV1::BackwardJump,
                    ..
                }
            }
        ));
        assert!(matches!(
            tracker.observe(&observation(910, 3), "bridge-run-1", 3),
            RosClockLifecycleDispositionV1::AwaitingExplicitRebind {
                kind: RosClockTransitionKindV1::BackwardJump
            }
        ));
        assert!(matches!(
            tracker.acknowledge_transition(&observation(910, 3), "bridge-run-1", 3),
            RosClockLifecycleDispositionV1::TransitionRebound {
                kind: RosClockTransitionKindV1::BackwardJump,
                generation: 2,
                ..
            }
        ));
        assert!(matches!(
            tracker.observe(&observation(920, 4), "bridge-run-1", 4),
            RosClockLifecycleDispositionV1::Advancing { generation: 2, .. }
        ));
    }

    #[test]
    fn forward_discontinuity_requires_rebind() {
        let mut tracker = initialized_tracker();
        assert!(matches!(
            tracker.observe(&observation(1_501, 2), "bridge-run-1", 2),
            RosClockLifecycleDispositionV1::TransitionRequired {
                transition: RosClockTransitionV1 {
                    kind: RosClockTransitionKindV1::ForwardJump,
                    ..
                }
            }
        ));
        assert!(tracker.pending_transition().is_some());
    }

    #[test]
    fn identity_and_receipt_epoch_changes_fail_closed() {
        let mut tracker = initialized_tracker();
        let mut changed_source = observation(1_010, 2);
        changed_source.source_epoch_id = "bag-run-2".into();
        assert!(matches!(
            tracker.observe(&changed_source, "bridge-run-1", 2),
            RosClockLifecycleDispositionV1::TransitionRequired {
                transition: RosClockTransitionV1 {
                    kind: RosClockTransitionKindV1::SourceEpochChanged,
                    ..
                }
            }
        ));

        let mut tracker = initialized_tracker();
        let mut changed_receipt = observation(1_010, 2);
        changed_receipt.receive_epoch_id = "bridge-run-2".into();
        assert!(matches!(
            tracker.observe(&changed_receipt, "bridge-run-2", 2),
            RosClockLifecycleDispositionV1::TransitionRequired {
                transition: RosClockTransitionV1 {
                    kind: RosClockTransitionKindV1::ReceiveEpochChanged,
                    ..
                }
            }
        ));
    }

    #[test]
    fn stale_future_and_replayed_receipts_are_not_accepted() {
        let mut tracker = initialized_tracker();
        assert!(matches!(
            tracker.observe(&observation(1_010, 2), "bridge-run-1", 1_000),
            RosClockLifecycleDispositionV1::ClockObservationStale { .. }
        ));
        assert!(matches!(
            tracker.observe(&observation(1_010, 10), "bridge-run-1", 9),
            RosClockLifecycleDispositionV1::InvalidObservation { .. }
        ));
        assert!(matches!(
            tracker.observe(&observation(1_010, 1), "bridge-run-1", 1),
            RosClockLifecycleDispositionV1::InvalidObservation { .. }
        ));
        assert_eq!(tracker.generation(), Some(1));
    }

    #[test]
    fn zero_after_initialization_requires_explicit_rebind() {
        let mut tracker = initialized_tracker();
        assert!(matches!(
            tracker.observe(&observation(0, 2), "bridge-run-1", 2),
            RosClockLifecycleDispositionV1::TransitionRequired {
                transition: RosClockTransitionV1 {
                    kind: RosClockTransitionKindV1::UninitializedAfterInitialization,
                    ..
                }
            }
        ));
    }

    #[test]
    fn invalid_policy_is_rejected() {
        let mut invalid = policy();
        invalid.pause_after_ns = 0;
        assert!(matches!(
            RosClockLifecycleTrackerV1::new(invalid),
            Err(RosClockLifecycleDispositionV1::InvalidPolicy { .. })
        ));
    }
}
