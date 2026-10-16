// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-ros-bridge
//!
//! Bridges Symthaea's holographic liquid brain (HLB) to the ROS2 ecosystem.
//!
//! ## Capabilities
//!
//! - **Proprioception**: Subscribes to `/joint_states` and transforms them into HDC vectors.
//! - **Consciousness Telemetry**: Publishes Φ (Phi), neuromodulator levels, and Eight Harmonies activations.
//! - **Motor Output**: Translates `RobotActuatorCommand` into `trajectory_msgs`.
//!
//! ## ROS2 Setup
//!
//! Enable the `ros2` Cargo feature to compile the ROS transport and qualify it
//! against an exact ROS distro/OS/RMW/architecture profile. Without that feature,
//! telemetry may still run, but trajectory publication returns
//! `RosTransportUnavailable`; it never simulates a successful robot command.
//! The ROS 2 support matrix and migration gates are tracked in issue #1287.

#![deny(unsafe_code)]

/// Pure, fail-closed source-clock lifecycle tracking.
pub mod clock_lifecycle;

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use thiserror::Error;

/// Errors reported by the ROS2 bridge.
#[derive(Debug, Error)]
pub enum RosBridgeError {
    #[error("ROS2 context not initialized")]
    NotInitialized,
    #[error("ROS2 trajectory publisher unavailable; command was not published")]
    RosTransportUnavailable,
    #[error("Communication error: {0}")]
    Communication(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid consciousness status: {0}")]
    InvalidStatus(String),
    #[error("Invalid trajectory command: {0}")]
    InvalidTrajectory(String),
    #[error("Invalid ROS JointState: {0}")]
    InvalidJointState(String),
    #[error("Sensor data unavailable: {0}")]
    SensorDataUnavailable(String),
    #[error("Invalid sensor data: {0}")]
    InvalidSensorData(String),
    #[error("Safety interlock blocked trajectory publication: {0}")]
    SafetyInterlock(String),
}

/// Normalized consciousness metrics for ROS2 broadcasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStatusMsg {
    /// Integrated Information Theory (IIT) Φ value [0, 1].
    pub phi: f64,
    /// Eight Harmonies activations [0, 1].
    pub harmonies: [f32; 8],
    /// Neuromodulator levels: [dopamine, serotonin, noradrenaline, acetylcholine].
    pub neuromodulators: [f32; 4],
    /// Global arousal level [0, 1].
    pub arousal: f32,
    /// Epistemic uncertainty [0, 1].
    pub uncertainty: f32,
    /// Current cycle timestamp (ISO 8601).
    pub timestamp: String,
}

/// Explicit ROS clock binding for a sensor timestamp.
///
/// ROS header stamps do not, by themselves, prove that two publishers use the
/// same clock domain. A freshly captured observation therefore defaults to
/// `Unbound` and is not eligible for cross-sensor age/skew claims until a
/// deployment profile binds the source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosClockDomainV1 {
    Unbound,
    Named(String),
}

/// Monotonic reception receipt captured inside one ROS subscription callback.
/// Sequence and time are scoped to one bridge-instance receive-clock epoch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosSampleReceiptV1 {
    pub sequence: u64,
    pub received_monotonic_ns: u64,
    pub receive_epoch_id: String,
}

/// Compute residence age only when receipt and capture belong to the same
/// monotonic epoch and the receipt does not postdate the capture.
fn receive_age_at_capture_ns(
    receipt: &RosSampleReceiptV1,
    capture_epoch_id: &str,
    captured_monotonic_ns: u64,
) -> Option<u64> {
    if receipt.receive_epoch_id != capture_epoch_id {
        return None;
    }
    captured_monotonic_ns.checked_sub(receipt.received_monotonic_ns)
}

/// Provenance metadata preserved from one ROS Header and its local receipt.
///
/// The publisher's source clock remains unbound until a deployment profile
/// names it. Receipt age is comparable only for samples with the same
/// receive-epoch identity; a monotonic receipt does not synchronize publisher
/// clocks or prove sensor calibration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosSampleMetadataV1 {
    pub stamp_sec: i32,
    pub stamp_nanosec: u32,
    pub frame_id: String,
    pub clock_domain: RosClockDomainV1,
    /// Time-source generation that changes on qualified clock discontinuity or reset.
    pub source_clock_epoch_id: Option<String>,
    pub received_monotonic_ns: Option<u64>,
    pub receive_sequence: Option<u64>,
    pub receive_epoch_id: Option<String>,
    /// Time this sample had resided in this bridge when the snapshot was assembled.
    /// This is not publisher-side sample age; source clock binding remains separate.
    pub receive_age_at_capture_ns: Option<u64>,
    pub calibration_id: Option<String>,
}

/// An individual physical field without conflating missing data with zero.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RosEvidenceValueV1<T> {
    Available {
        value: T,
        source: RosSampleMetadataV1,
    },
    Unavailable {
        reason: String,
        source: Option<RosSampleMetadataV1>,
    },
}

/// Interpreted ROS covariance semantics. All-zero covariance means unknown,
/// not zero uncertainty. A -1 first element denotes an unavailable estimate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RosCovarianceV1 {
    Unknown,
    Reported([f64; 9]),
    Unavailable,
}

/// Canonical DMC21 joint observations with independent availability per field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RosJointStateObservationV1 {
    pub joint_names: Vec<String>,
    pub positions: RosEvidenceValueV1<Vec<f64>>,
    pub velocities: RosEvidenceValueV1<Vec<f64>>,
    pub efforts: RosEvidenceValueV1<Vec<f64>>,
    pub source: RosSampleMetadataV1,
}

/// IMU values remain in their source frame. No implicit transform or sensor
/// fusion is performed at this boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RosImuObservationV1 {
    pub orientation_wxyz: RosEvidenceValueV1<[f64; 4]>,
    pub orientation_covariance: RosCovarianceV1,
    pub angular_velocity_body_rad_s: RosEvidenceValueV1<[f64; 3]>,
    pub angular_velocity_covariance: RosCovarianceV1,
    pub linear_acceleration_sensor_frame_mps2: RosEvidenceValueV1<[f64; 3]>,
    pub linear_acceleration_covariance: RosCovarianceV1,
    pub source: RosSampleMetadataV1,
}

/// A deliberately partial humanoid observation, not a HumanoidState.
///
/// This type can represent the data the current bridge actually subscribes to
/// without inventing root pose, root/COM velocity, extremity positions, sensor
/// time-domain bindings, or calibration evidence. Downstream fusion must check
/// each field's availability and bind clocks/frames before treating sources as
/// one synchronized physical state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RosPartialHumanoidObservationV1 {
    pub schema_version: u16,
    pub morphology_profile_id: String,
    /// Snapshot assembly time in the bridge-instance monotonic receive epoch.
    pub captured_monotonic_ns: u64,
    pub receive_epoch_id: String,
    pub joint_state: RosJointStateObservationV1,
    pub imu: RosEvidenceValueV1<RosImuObservationV1>,
    pub root_position_world_m: RosEvidenceValueV1<[f64; 3]>,
    pub root_linear_velocity_world_mps: RosEvidenceValueV1<[f64; 3]>,
    pub center_of_mass_velocity_world_mps: RosEvidenceValueV1<[f64; 3]>,
    pub head_height_world_m: RosEvidenceValueV1<f64>,
    pub extremities_world_m: RosEvidenceValueV1<Vec<f64>>,
}

/// Source-clock time supplied by a deployment-qualified clock adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosClockNowV1 {
    pub domain: RosClockDomainV1,
    pub source_clock_epoch_id: String,
    pub timestamp_ns: u64,
}

/// Profile-frozen temporal limits. The source clock must be explicitly bound;
/// these limits are not portable defaults for every robot/control loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosTemporalPolicyV1 {
    pub required_source_clock_domain: String,
    pub maximum_source_age_ns: u64,
    pub maximum_source_skew_ns: u64,
    pub maximum_bridge_residence_age_ns: u64,
    pub reject_zero_source_stamp: bool,
    /// Whether two successive samples from one stream may share the same source stamp.
    /// This is profile-specific because some sources use coarse timestamp resolution.
    pub reject_duplicate_source_stamp: bool,
}

/// Identifies which input failed the temporal gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosTemporalStreamV1 {
    JointState,
    Imu,
}

/// Temporal admission result only. `Fresh` does not establish calibration,
/// covariance quality, frame-transform validity, complete physical state, or
/// action authority.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosTemporalDispositionV1 {
    Fresh {
        joint_source_age_ns: u64,
        imu_source_age_ns: u64,
        source_skew_ns: u64,
        joint_bridge_residence_age_ns: u64,
        imu_bridge_residence_age_ns: u64,
    },
    Unsynchronized {
        reason: String,
    },
    InvalidPolicy {
        reason: String,
    },
    InvalidClock {
        stream: RosTemporalStreamV1,
        reason: String,
    },
    StaleSource {
        stream: RosTemporalStreamV1,
        age_ns: u64,
        maximum_age_ns: u64,
    },
    ExcessiveSourceSkew {
        skew_ns: u64,
        maximum_skew_ns: u64,
    },
    SourceUnavailable {
        stream: RosTemporalStreamV1,
        reason: String,
    },
    ReceiptUnavailable {
        stream: RosTemporalStreamV1,
        reason: String,
    },
    BridgeResidenceExpired {
        stream: RosTemporalStreamV1,
        age_ns: u64,
        maximum_age_ns: u64,
    },
}

fn ros_source_stamp_ns(
    sample: &RosSampleMetadataV1,
    stream: RosTemporalStreamV1,
    reject_zero: bool,
) -> Result<u64, RosTemporalDispositionV1> {
    if sample.stamp_sec < 0 {
        return Err(RosTemporalDispositionV1::InvalidClock {
            stream,
            reason: "negative source timestamp seconds are not admissible for this profile".into(),
        });
    }
    if sample.stamp_nanosec >= 1_000_000_000 {
        return Err(RosTemporalDispositionV1::InvalidClock {
            stream,
            reason: "source timestamp nanoseconds must be less than 1,000,000,000".into(),
        });
    }
    let seconds = u64::try_from(sample.stamp_sec).map_err(|_| {
        RosTemporalDispositionV1::InvalidClock {
            stream,
            reason: "source timestamp seconds cannot be represented as unsigned nanoseconds".into(),
        }
    })?;
    let timestamp_ns = seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(sample.stamp_nanosec as u64))
        .ok_or_else(|| RosTemporalDispositionV1::InvalidClock {
            stream,
            reason: "source timestamp overflows nanosecond representation".into(),
        })?;
    if reject_zero && timestamp_ns == 0 {
        return Err(RosTemporalDispositionV1::InvalidClock {
            stream,
            reason: "zero source timestamp is prohibited by the selected profile".into(),
        });
    }
    Ok(timestamp_ns)
}

fn ros_bridge_residence_age_ns(
    sample: &RosSampleMetadataV1,
    stream: RosTemporalStreamV1,
    capture_epoch_id: &str,
    captured_monotonic_ns: u64,
) -> Result<u64, RosTemporalDispositionV1> {
    let epoch_id = sample.receive_epoch_id.as_deref().ok_or_else(|| {
        RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "receive-clock epoch identity is absent".into(),
        }
    })?;
    if epoch_id != capture_epoch_id {
        return Err(RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "sample receipt and capture belong to different monotonic epochs".into(),
        });
    }
    let sequence = sample.receive_sequence.ok_or_else(|| {
        RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "callback receive sequence is absent".into(),
        }
    })?;
    if sequence == 0 {
        return Err(RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "callback receive sequence zero is invalid".into(),
        });
    }
    let received_ns = sample.received_monotonic_ns.ok_or_else(|| {
        RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "callback monotonic receive time is absent".into(),
        }
    })?;
    let declared_age = sample.receive_age_at_capture_ns.ok_or_else(|| {
        RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "bridge residence age is absent".into(),
        }
    })?;
    let calculated_age = captured_monotonic_ns.checked_sub(received_ns).ok_or_else(|| {
        RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "callback receive time postdates snapshot capture".into(),
        }
    })?;
    if declared_age != calculated_age {
        return Err(RosTemporalDispositionV1::ReceiptUnavailable {
            stream,
            reason: "declared bridge residence age disagrees with receive/capture timestamps".into(),
        });
    }
    Ok(calculated_age)
}

/// Deterministic admission check for a JointState + IMU temporal pair.
///
/// The caller must source `now` from a qualified clock adapter; passing a
/// timestamp does not itself establish its truth. This routine never compares
/// different or unbound publisher clocks, and its `Fresh` result is temporal
/// evidence only—not permission to construct a full state or actuate hardware.
pub fn evaluate_ros_temporal_pair_v1(
    joint_state: &RosSampleMetadataV1,
    imu: Option<&RosSampleMetadataV1>,
    now: &RosClockNowV1,
    policy: &RosTemporalPolicyV1,
    capture_epoch_id: &str,
    captured_monotonic_ns: u64,
) -> RosTemporalDispositionV1 {
    if policy.required_source_clock_domain.trim().is_empty() {
        return RosTemporalDispositionV1::InvalidPolicy {
            reason: "required source clock domain must not be empty".into(),
        };
    }

    let imu = match imu {
        Some(imu) => imu,
        None => {
            return RosTemporalDispositionV1::SourceUnavailable {
                stream: RosTemporalStreamV1::Imu,
                reason: "IMU source observation is unavailable".into(),
            };
        }
    };

    let expected_domain = &policy.required_source_clock_domain;
    for (stream, sample) in [
        (RosTemporalStreamV1::JointState, joint_state),
        (RosTemporalStreamV1::Imu, imu),
    ] {
        let sample_epoch = match sample.source_clock_epoch_id.as_deref() {
            Some(epoch) if !epoch.trim().is_empty() => epoch,
            _ => {
                return RosTemporalDispositionV1::Unsynchronized {
                    reason: format!("{stream:?} source-clock epoch is unbound"),
                };
            }
        };
        if sample_epoch != now.source_clock_epoch_id {
            return RosTemporalDispositionV1::Unsynchronized {
                reason: format!(
                    "{stream:?} source-clock epoch {sample_epoch:?} does not match current-time epoch {:?}",
                    now.source_clock_epoch_id
                ),
            };
        }
        match &sample.clock_domain {
            RosClockDomainV1::Unbound => {
                return RosTemporalDispositionV1::Unsynchronized {
                    reason: format!("{stream:?} publisher clock domain is unbound"),
                };
            }
            RosClockDomainV1::Named(domain) if domain.trim().is_empty() => {
                return RosTemporalDispositionV1::InvalidClock {
                    stream,
                    reason: "publisher clock binding has an empty name".into(),
                };
            }
            RosClockDomainV1::Named(domain) if domain != expected_domain => {
                return RosTemporalDispositionV1::Unsynchronized {
                    reason: format!(
                        "{stream:?} clock domain {domain:?} does not match required domain {expected_domain:?}"
                    ),
                };
            }
            RosClockDomainV1::Named(_) => {}
        }
    }
    if now.source_clock_epoch_id.trim().is_empty() {
        return RosTemporalDispositionV1::Unsynchronized {
            reason: "qualified current-time source-clock epoch ID is unbound".into(),
        };
    }
    match &now.domain {
        RosClockDomainV1::Named(domain) if domain == expected_domain => {}
        RosClockDomainV1::Named(domain) => {
            return RosTemporalDispositionV1::Unsynchronized {
                reason: format!(
                    "current-time clock domain {domain:?} does not match required domain {expected_domain:?}"
                ),
            };
        }
        RosClockDomainV1::Unbound => {
            return RosTemporalDispositionV1::Unsynchronized {
                reason: "current-time clock domain is unbound".into(),
            };
        }
    }

    let joint_stamp = match ros_source_stamp_ns(
        joint_state,
        RosTemporalStreamV1::JointState,
        policy.reject_zero_source_stamp,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };
    let imu_stamp = match ros_source_stamp_ns(
        imu,
        RosTemporalStreamV1::Imu,
        policy.reject_zero_source_stamp,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };

    let joint_residence_age = match ros_bridge_residence_age_ns(
        joint_state,
        RosTemporalStreamV1::JointState,
        capture_epoch_id,
        captured_monotonic_ns,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };
    let imu_residence_age = match ros_bridge_residence_age_ns(
        imu,
        RosTemporalStreamV1::Imu,
        capture_epoch_id,
        captured_monotonic_ns,
    ) {
        Ok(value) => value,
        Err(disposition) => return disposition,
    };

    if joint_residence_age > policy.maximum_bridge_residence_age_ns {
        return RosTemporalDispositionV1::BridgeResidenceExpired {
            stream: RosTemporalStreamV1::JointState,
            age_ns: joint_residence_age,
            maximum_age_ns: policy.maximum_bridge_residence_age_ns,
        };
    }
    if imu_residence_age > policy.maximum_bridge_residence_age_ns {
        return RosTemporalDispositionV1::BridgeResidenceExpired {
            stream: RosTemporalStreamV1::Imu,
            age_ns: imu_residence_age,
            maximum_age_ns: policy.maximum_bridge_residence_age_ns,
        };
    }

    if joint_stamp > now.timestamp_ns {
        return RosTemporalDispositionV1::InvalidClock {
            stream: RosTemporalStreamV1::JointState,
            reason: "JointState source stamp is later than qualified current time".into(),
        };
    }
    if imu_stamp > now.timestamp_ns {
        return RosTemporalDispositionV1::InvalidClock {
            stream: RosTemporalStreamV1::Imu,
            reason: "IMU source stamp is later than qualified current time".into(),
        };
    }
    let joint_age = now.timestamp_ns - joint_stamp;
    let imu_age = now.timestamp_ns - imu_stamp;
    if joint_age > policy.maximum_source_age_ns {
        return RosTemporalDispositionV1::StaleSource {
            stream: RosTemporalStreamV1::JointState,
            age_ns: joint_age,
            maximum_age_ns: policy.maximum_source_age_ns,
        };
    }
    if imu_age > policy.maximum_source_age_ns {
        return RosTemporalDispositionV1::StaleSource {
            stream: RosTemporalStreamV1::Imu,
            age_ns: imu_age,
            maximum_age_ns: policy.maximum_source_age_ns,
        };
    }

    let skew = joint_stamp.abs_diff(imu_stamp);
    if skew > policy.maximum_source_skew_ns {
        return RosTemporalDispositionV1::ExcessiveSourceSkew {
            skew_ns: skew,
            maximum_skew_ns: policy.maximum_source_skew_ns,
        };
    }

    RosTemporalDispositionV1::Fresh {
        joint_source_age_ns: joint_age,
        imu_source_age_ns: imu_age,
        source_skew_ns: skew,
        joint_bridge_residence_age_ns: joint_residence_age,
        imu_bridge_residence_age_ns: imu_residence_age,
    }
}

/// Per-stream history retained only within one explicit bridge receive epoch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosTemporalHistoryEntryV1 {
    pub receive_sequence: u64,
    pub source_stamp_ns: u64,
    pub source_clock_domain: String,
    pub source_clock_epoch_id: String,
}

/// Stateful replay/regression guard layered after pairwise temporal admission.
///
/// Rebinding to a new receipt epoch clears history explicitly. The gate never
/// silently resets when it sees a different epoch, because that could allow an
/// old sample to be replayed into a new run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RosTemporalHistoryV1 {
    pub receive_epoch_id: String,
    pub source_clock_epoch_id: String,
    pub last_joint_state: Option<RosTemporalHistoryEntryV1>,
    pub last_imu: Option<RosTemporalHistoryEntryV1>,
}

impl RosTemporalHistoryV1 {
    pub fn new(
        receive_epoch_id: impl Into<String>,
        source_clock_epoch_id: impl Into<String>,
    ) -> Result<Self, RosTemporalHistoryDispositionV1> {
        let receive_epoch_id = receive_epoch_id.into();
        let source_clock_epoch_id = source_clock_epoch_id.into();
        if receive_epoch_id.trim().is_empty() || source_clock_epoch_id.trim().is_empty() {
            return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
                reason: "receive epoch ID and source-clock epoch ID must not be empty".into(),
            });
        }
        Ok(Self {
            receive_epoch_id,
            source_clock_epoch_id,
            last_joint_state: None,
            last_imu: None,
        })
    }

    /// Explicitly rebind one or both clock epochs and clear retained stream history.
    /// This must only be called in response to an observed/qualified clock lifecycle
    /// event, never merely because a stale sample carries a different timestamp.
    pub fn reset_for_epochs(
        &mut self,
        receive_epoch_id: impl Into<String>,
        source_clock_epoch_id: impl Into<String>,
    ) -> Result<(), RosTemporalHistoryDispositionV1> {
        let receive_epoch_id = receive_epoch_id.into();
        let source_clock_epoch_id = source_clock_epoch_id.into();
        if receive_epoch_id.trim().is_empty() || source_clock_epoch_id.trim().is_empty() {
            return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
                reason: "receive epoch ID and source-clock epoch ID must not be empty".into(),
            });
        }
        self.receive_epoch_id = receive_epoch_id;
        self.source_clock_epoch_id = source_clock_epoch_id;
        self.last_joint_state = None;
        self.last_imu = None;
        Ok(())
    }
}

/// Stateful history-gate rejection. It is deliberately separate from freshness
/// so callers can tell a stale input from a replay or clock discontinuity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RosTemporalHistoryDispositionV1 {
    Accepted {
        joint_sequence_gap: u64,
        imu_sequence_gap: u64,
    },
    TemporalRejected {
        disposition: RosTemporalDispositionV1,
    },
    HistoryUninitialized {
        reason: String,
    },
    ReceiveEpochChanged {
        expected_epoch_id: String,
        observed_epoch_id: String,
    },
    InvalidHistory {
        reason: String,
    },
    SequenceNotIncreasing {
        stream: RosTemporalStreamV1,
        previous: u64,
        current: u64,
    },
    SourceTimestampRegressed {
        stream: RosTemporalStreamV1,
        previous_ns: u64,
        current_ns: u64,
    },
    DuplicateSourceTimestamp {
        stream: RosTemporalStreamV1,
        timestamp_ns: u64,
    },
    SourceClockChanged {
        stream: RosTemporalStreamV1,
        previous_domain: String,
        current_domain: String,
    },
    SourceClockEpochChanged {
        expected_epoch_id: String,
        observed_epoch_id: String,
    },
}

/// Construct a candidate history entry without mutating the retained history.
fn prepare_temporal_history_entry(
    sample: &RosSampleMetadataV1,
    stream: RosTemporalStreamV1,
    policy: &RosTemporalPolicyV1,
    receive_epoch_id: &str,
) -> Result<RosTemporalHistoryEntryV1, RosTemporalHistoryDispositionV1> {
    let observed_epoch = sample.receive_epoch_id.as_deref().ok_or_else(|| {
        RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!("{stream:?} receipt epoch is absent"),
        }
    })?;
    if observed_epoch != receive_epoch_id {
        return Err(RosTemporalHistoryDispositionV1::ReceiveEpochChanged {
            expected_epoch_id: receive_epoch_id.to_string(),
            observed_epoch_id: observed_epoch.to_string(),
        });
    }
    let sequence = sample.receive_sequence.ok_or_else(|| {
        RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!("{stream:?} receive sequence is absent"),
        }
    })?;
    if sequence == 0 {
        return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!("{stream:?} receive sequence must be greater than zero"),
        });
    }
    let source_clock_epoch_id = sample.source_clock_epoch_id.as_deref().ok_or_else(|| {
        RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!("{stream:?} source-clock epoch is unbound"),
        }
    })?;
    if source_clock_epoch_id.trim().is_empty() {
        return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!("{stream:?} source-clock epoch ID is empty"),
        });
    }
    let domain = match &sample.clock_domain {
        RosClockDomainV1::Unbound => {
            return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
                reason: format!("{stream:?} source clock domain is unbound"),
            });
        }
        RosClockDomainV1::Named(domain) if domain.trim().is_empty() => {
            return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
                reason: format!("{stream:?} source clock domain name is empty"),
            });
        }
        RosClockDomainV1::Named(domain) => domain.clone(),
    };
    if domain != policy.required_source_clock_domain {
        return Err(RosTemporalHistoryDispositionV1::InvalidHistory {
            reason: format!(
                "{stream:?} source clock domain {domain:?} does not match the frozen history profile {:?}",
                policy.required_source_clock_domain
            ),
        });
    }
    let source_stamp_ns = ros_source_stamp_ns(
        sample,
        stream,
        policy.reject_zero_source_stamp,
    )
    .map_err(|disposition| RosTemporalHistoryDispositionV1::TemporalRejected {
        disposition,
    })?;

    Ok(RosTemporalHistoryEntryV1 {
        receive_sequence: sequence,
        source_stamp_ns,
        source_clock_domain: domain,
        source_clock_epoch_id: source_clock_epoch_id.to_string(),
    })
}

/// Check a candidate against the previous accepted sample without committing it.
fn check_temporal_history_entry(
    stream: RosTemporalStreamV1,
    previous: Option<&RosTemporalHistoryEntryV1>,
    current: &RosTemporalHistoryEntryV1,
    policy: &RosTemporalPolicyV1,
) -> Result<u64, RosTemporalHistoryDispositionV1> {
    let Some(previous) = previous else {
        return Ok(0);
    };

    if current.source_clock_epoch_id != previous.source_clock_epoch_id {
        return Err(RosTemporalHistoryDispositionV1::SourceClockEpochChanged {
            expected_epoch_id: previous.source_clock_epoch_id.clone(),
            observed_epoch_id: current.source_clock_epoch_id.clone(),
        });
    }
    if current.source_clock_domain != previous.source_clock_domain {
        return Err(RosTemporalHistoryDispositionV1::SourceClockChanged {
            stream,
            previous_domain: previous.source_clock_domain.clone(),
            current_domain: current.source_clock_domain.clone(),
        });
    }
    if current.receive_sequence <= previous.receive_sequence {
        return Err(RosTemporalHistoryDispositionV1::SequenceNotIncreasing {
            stream,
            previous: previous.receive_sequence,
            current: current.receive_sequence,
        });
    }
    if current.source_stamp_ns < previous.source_stamp_ns {
        return Err(RosTemporalHistoryDispositionV1::SourceTimestampRegressed {
            stream,
            previous_ns: previous.source_stamp_ns,
            current_ns: current.source_stamp_ns,
        });
    }
    if policy.reject_duplicate_source_stamp
        && current.source_stamp_ns == previous.source_stamp_ns
    {
        return Err(RosTemporalHistoryDispositionV1::DuplicateSourceTimestamp {
            stream,
            timestamp_ns: current.source_stamp_ns,
        });
    }

    Ok(current
        .receive_sequence
        .saturating_sub(previous.receive_sequence)
        .saturating_sub(1))
}

/// Evaluate temporal validity and per-stream history as one atomic pair admission.
///
/// History advances only if both streams pass the temporal policy and both
/// candidates pass history validation. This prevents half-advancing JointState
/// history when the paired IMU sample is rejected.
pub fn evaluate_ros_temporal_pair_with_history_v1(
    history: &mut RosTemporalHistoryV1,
    joint_state: &RosSampleMetadataV1,
    imu: Option<&RosSampleMetadataV1>,
    now: &RosClockNowV1,
    policy: &RosTemporalPolicyV1,
    capture_epoch_id: &str,
    captured_monotonic_ns: u64,
) -> RosTemporalHistoryDispositionV1 {
    if history.receive_epoch_id.trim().is_empty() {
        return RosTemporalHistoryDispositionV1::HistoryUninitialized {
            reason: "history is not bound to a receive epoch".into(),
        };
    }
    if history.receive_epoch_id != capture_epoch_id {
        return RosTemporalHistoryDispositionV1::ReceiveEpochChanged {
            expected_epoch_id: history.receive_epoch_id.clone(),
            observed_epoch_id: capture_epoch_id.to_string(),
        };
    }
    if history.source_clock_epoch_id != now.source_clock_epoch_id {
        return RosTemporalHistoryDispositionV1::SourceClockEpochChanged {
            expected_epoch_id: history.source_clock_epoch_id.clone(),
            observed_epoch_id: now.source_clock_epoch_id.clone(),
        };
    }

    let temporal = evaluate_ros_temporal_pair_v1(
        joint_state,
        imu,
        now,
        policy,
        capture_epoch_id,
        captured_monotonic_ns,
    );
    if !matches!(&temporal, RosTemporalDispositionV1::Fresh { .. }) {
        return RosTemporalHistoryDispositionV1::TemporalRejected {
            disposition: temporal,
        };
    }
    let Some(imu) = imu else {
        return RosTemporalHistoryDispositionV1::TemporalRejected {
            disposition: RosTemporalDispositionV1::SourceUnavailable {
                stream: RosTemporalStreamV1::Imu,
                reason: "IMU source observation is unavailable".into(),
            },
        };
    };

    let joint_candidate = match prepare_temporal_history_entry(
        joint_state,
        RosTemporalStreamV1::JointState,
        policy,
        capture_epoch_id,
    ) {
        Ok(entry) => entry,
        Err(disposition) => return disposition,
    };
    let imu_candidate = match prepare_temporal_history_entry(
        imu,
        RosTemporalStreamV1::Imu,
        policy,
        capture_epoch_id,
    ) {
        Ok(entry) => entry,
        Err(disposition) => return disposition,
    };

    let joint_gap = match check_temporal_history_entry(
        RosTemporalStreamV1::JointState,
        history.last_joint_state.as_ref(),
        &joint_candidate,
        policy,
    ) {
        Ok(gap) => gap,
        Err(disposition) => return disposition,
    };
    let imu_gap = match check_temporal_history_entry(
        RosTemporalStreamV1::Imu,
        history.last_imu.as_ref(),
        &imu_candidate,
        policy,
    ) {
        Ok(gap) => gap,
        Err(disposition) => return disposition,
    };

    // Commit both histories only after validation for both streams succeeds.
    history.last_joint_state = Some(joint_candidate);
    history.last_imu = Some(imu_candidate);
    RosTemporalHistoryDispositionV1::Accepted {
        joint_sequence_gap: joint_gap,
        imu_sequence_gap: imu_gap,
    }
}

/// Reorder one ROS JointState value array into the bridge's explicit model order.
///
/// ROS message array position is not joint identity. This helper requires an exact
/// set of unique names and finite values; missing or unknown joints remain errors
/// rather than being silently filled, dropped, or shifted into another joint slot.
fn canonicalize_joint_values(
    observed_names: &[String],
    values: &[f64],
    expected_names: &[&str],
    field: &str,
) -> Result<Vec<f64>, RosBridgeError> {
    use std::collections::{HashMap, HashSet};

    if expected_names.is_empty() {
        return Err(RosBridgeError::InvalidJointState(
            "expected joint profile must not be empty".into(),
        ));
    }
    let expected_set: HashSet<&str> = expected_names.iter().copied().collect();
    if expected_set.len() != expected_names.len()
        || expected_names.iter().any(|name| name.trim().is_empty())
    {
        return Err(RosBridgeError::InvalidJointState(
            "expected joint profile contains empty or duplicate names".into(),
        ));
    }
    if observed_names.len() != values.len() {
        return Err(RosBridgeError::InvalidJointState(format!(
            "name/value cardinality mismatch: {} names but {} {field} values",
            observed_names.len(),
            values.len()
        )));
    }
    if observed_names.len() != expected_names.len() {
        return Err(RosBridgeError::InvalidJointState(format!(
            "expected exactly {} joints, received {}",
            expected_names.len(),
            observed_names.len()
        )));
    }

    let mut indices = HashMap::with_capacity(observed_names.len());
    for (index, name) in observed_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(RosBridgeError::InvalidJointState(
                "joint names must not be empty".into(),
            ));
        }
        if !expected_set.contains(name.as_str()) {
            return Err(RosBridgeError::InvalidJointState(format!(
                "unrecognized joint name: {name}"
            )));
        }
        if indices.insert(name.as_str(), index).is_some() {
            return Err(RosBridgeError::InvalidJointState(format!(
                "duplicate joint name: {name}"
            )));
        }
        if !values[index].is_finite() {
            return Err(RosBridgeError::InvalidJointState(format!(
                "{field} for joint {name} is not finite"
            )));
        }
    }

    expected_names
        .iter()
        .map(|name| {
            indices
                .get(name)
                .map(|&index| values[index])
                .ok_or_else(|| {
                    RosBridgeError::InvalidJointState(format!(
                        "required joint is missing: {name}"
                    ))
                })
        })
        .collect()
}

/// Validate and normalize an IMU quaternion in ROS [w, x, y, z] order.
fn normalize_imu_quaternion(quaternion: [f64; 4]) -> Result<[f64; 4], RosBridgeError> {
    if quaternion.iter().any(|component| !component.is_finite()) {
        return Err(RosBridgeError::InvalidSensorData(
            "IMU orientation quaternion contains a non-finite component".into(),
        ));
    }

    let norm_squared: f64 = quaternion.iter().map(|component| component * component).sum();
    if !norm_squared.is_finite() || norm_squared <= 1.0e-24 {
        return Err(RosBridgeError::InvalidSensorData(
            "IMU orientation quaternion is degenerate or cannot be normalized".into(),
        ));
    }

    let norm = norm_squared.sqrt();
    Ok(quaternion.map(|component| component / norm))
}

/// Reject non-finite 3D sensor vectors before they enter physical state.
fn validate_sensor_vector(field: &str, values: [f64; 3]) -> Result<[f64; 3], RosBridgeError> {
    if values.iter().any(|component| !component.is_finite()) {
        return Err(RosBridgeError::InvalidSensorData(format!(
            "{field} contains a non-finite component"
        )));
    }
    Ok(values)
}

#[cfg(feature = "ros2")]
fn ros_sample_metadata(
    header: &std_msgs::msg::Header,
    receipt: Option<RosSampleReceiptV1>,
    captured_monotonic_ns: u64,
    capture_epoch_id: &str,
) -> Result<RosSampleMetadataV1, RosBridgeError> {
    if header.stamp.nanosec >= 1_000_000_000 {
        return Err(RosBridgeError::InvalidSensorData(
            "ROS header nanosec must be less than 1,000,000,000".into(),
        ));
    }

    let (
        received_monotonic_ns,
        receive_sequence,
        receive_epoch_id,
        receive_age_at_capture_ns,
    ) = match receipt {
        Some(receipt) => (
            Some(receipt.received_monotonic_ns),
            Some(receipt.sequence),
            Some(receipt.receive_epoch_id.clone()),
            receive_age_at_capture_ns(
                &receipt,
                capture_epoch_id,
                captured_monotonic_ns,
            ),
        ),
        None => (None, None, None, None),
    };

    Ok(RosSampleMetadataV1 {
        stamp_sec: header.stamp.sec,
        stamp_nanosec: header.stamp.nanosec,
        frame_id: header.frame_id.clone(),
        clock_domain: RosClockDomainV1::Unbound,
        source_clock_epoch_id: None,
        received_monotonic_ns,
        receive_sequence,
        receive_epoch_id,
        receive_age_at_capture_ns,
        calibration_id: None,
    })
}

#[cfg(feature = "ros2")]
fn ros_covariance_v1(
    covariance: [f64; 9],
    field: &str,
) -> Result<RosCovarianceV1, RosBridgeError> {
    // ROS sensor_msgs/Imu specifies -1 in element 0 when this estimate is
    // unavailable. The other covariance cells are not interpreted in that case.
    classify_ros_covariance(covariance).map_err(|_| {
        RosBridgeError::InvalidSensorData(format!(
            "{field} covariance contains a non-finite value"
        ))
    })
}

#[cfg(feature = "ros2")]
fn evidence_from_imu_vector(
    value: [f64; 3],
    covariance: &RosCovarianceV1,
    source: &RosSampleMetadataV1,
    field: &str,
) -> Result<RosEvidenceValueV1<[f64; 3]>, RosBridgeError> {
    if matches!(covariance, RosCovarianceV1::Unavailable) {
        return Ok(RosEvidenceValueV1::Unavailable {
            reason: format!("ROS IMU marks {field} estimate unavailable (covariance[0] = -1)"),
            source: Some(source.clone()),
        });
    }

    Ok(RosEvidenceValueV1::Available {
        value: validate_sensor_vector(field, value)?,
        source: source.clone(),
    })
}

fn classify_ros_covariance(
    covariance: [f64; 9],
) -> Result<RosCovarianceV1, RosBridgeError> {
    // ROS sensor_msgs/Imu reserves covariance[0] == -1 for an unavailable
    // estimate. In that case the remaining covariance cells are not consulted.
    if covariance[0] == -1.0 {
        return Ok(RosCovarianceV1::Unavailable);
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(RosBridgeError::InvalidSensorData(
            "ROS covariance contains a non-finite value".into(),
        ));
    }
    if covariance.iter().all(|value| *value == 0.0) {
        Ok(RosCovarianceV1::Unknown)
    } else {
        Ok(RosCovarianceV1::Reported(covariance))
    }
}

/// Configuration for the ROS2 bridge node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosBridgeConfig {
    /// Node name in the ROS graph.
    pub node_name: String,
    /// Namespace for all topics.
    pub namespace: String,
    /// Update rate for telemetry publishing (Hz).
    pub telemetry_rate_hz: f32,
}

impl Default for RosBridgeConfig {
    fn default() -> Self {
        Self {
            node_name: "symthaea_brain_bridge".to_string(),
            namespace: "symthaea".to_string(),
            telemetry_rate_hz: 10.0,
        }
    }
}

/// Placeholder for the ROS2 bridge node.
///
/// In a real implementation with `rclrs`, this would own the `rclrs::Node`
/// and various publishers/subscribers.
pub struct RosBridgeNode {
    config: RosBridgeConfig,
    is_running: bool,
    pub phi: std::sync::Mutex<f64>,
    pub interlock: std::sync::Mutex<symthaea_sim_bridge::AmygdalaInterlock>,
    pub surprise_monitor: std::sync::Mutex<symthaea_sim_bridge::SurpriseMonitor>,
    pub predicted_state_hv: std::sync::Mutex<Option<ContinuousHV>>,
    pub telemetry_broadcaster: std::sync::Arc<symthaea_telemetry_grpc::TelemetryBroadcaster>,
    #[cfg(feature = "ros2")]
    pub latest_joint_state: std::sync::Arc<std::sync::Mutex<Option<sensor_msgs::msg::JointState>>>,
    #[cfg(feature = "ros2")]
    pub latest_imu: std::sync::Arc<std::sync::Mutex<Option<sensor_msgs::msg::Imu>>>,
    #[cfg(feature = "ros2")]
    latest_joint_state_receipt: std::sync::Arc<std::sync::Mutex<Option<RosSampleReceiptV1>>>,
    #[cfg(feature = "ros2")]
    latest_imu_receipt: std::sync::Arc<std::sync::Mutex<Option<RosSampleReceiptV1>>>,
    #[cfg(feature = "ros2")]
    receive_epoch: std::sync::Arc<std::time::Instant>,
    #[cfg(feature = "ros2")]
    receive_epoch_id: String,
    #[cfg(feature = "ros2")]
    context: Option<std::sync::Arc<rclrs::Context>>,
    #[cfg(feature = "ros2")]
    node: Option<std::sync::Arc<rclrs::Node>>,
    #[cfg(feature = "ros2")]
    trajectory_pub: Option<std::sync::Arc<rclrs::Publisher<trajectory_msgs::msg::JointTrajectory>>>,
    #[cfg(feature = "ros2")]
    status_pub: Option<std::sync::Arc<rclrs::Publisher<std_msgs::msg::String>>>,
    #[cfg(feature = "ros2")]
    joint_state_sub: Option<std::sync::Arc<rclrs::Subscription<sensor_msgs::msg::JointState>>>,
    #[cfg(feature = "ros2")]
    imu_sub: Option<std::sync::Arc<rclrs::Subscription<sensor_msgs::msg::Imu>>>,
    #[cfg(feature = "ros2")]
    scheduler: symthaea_control::ConsciousnessGainScheduler,
    #[cfg(feature = "ros2")]
    spin_thread: Option<std::thread::JoinHandle<()>>,
}

/// Command to control robot joint trajectories via ROS2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosTrajectoryCommand {
    pub joint_names: Vec<String>,
    pub points: Vec<RosTrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosTrajectoryPoint {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub efforts: Vec<f64>,
    pub time_from_start_ns: u64,
}

impl ConsciousnessStatusMsg {
    fn validate(&self) -> Result<(), RosBridgeError> {
        fn unit(value: f64, name: &str) -> Result<(), RosBridgeError> {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(RosBridgeError::InvalidStatus(format!(
                    "{name} must be finite and in [0, 1]"
                )))
            }
        }

        unit(self.phi, "phi")?;
        unit(self.arousal as f64, "arousal")?;
        unit(self.uncertainty as f64, "uncertainty")?;
        for (index, value) in self.harmonies.iter().enumerate() {
            unit(*value as f64, &format!("harmonies[{index}]"))?;
        }
        for (index, value) in self.neuromodulators.iter().enumerate() {
            unit(*value as f64, &format!("neuromodulators[{index}]"))?;
        }
        if self.timestamp.trim().is_empty() {
            return Err(RosBridgeError::InvalidStatus(
                "timestamp must not be empty".into(),
            ));
        }
        Ok(())
    }
}

impl RosTrajectoryCommand {
    fn validate(&self) -> Result<(), RosBridgeError> {
        use std::collections::HashSet;

        if self.joint_names.is_empty() {
            return Err(RosBridgeError::InvalidTrajectory(
                "joint_names must not be empty".into(),
            ));
        }
        if self.points.is_empty() {
            return Err(RosBridgeError::InvalidTrajectory(
                "trajectory must contain at least one point".into(),
            ));
        }
        let mut unique = HashSet::with_capacity(self.joint_names.len());
        for name in &self.joint_names {
            if name.trim().is_empty() || !unique.insert(name) {
                return Err(RosBridgeError::InvalidTrajectory(
                    "joint names must be non-empty and unique".into(),
                ));
            }
        }

        let joint_count = self.joint_names.len();
        let mut previous_time = None;
        for (point_index, point) in self.points.iter().enumerate() {
            if previous_time.is_some_and(|time| point.time_from_start_ns <= time) {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} time_from_start must be strictly increasing"
                )));
            }
            if point.time_from_start_ns / 1_000_000_000 > i32::MAX as u64 {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} time_from_start exceeds ROS Duration range"
                )));
            }
            previous_time = Some(point.time_from_start_ns);

            if point.positions.is_empty() && point.velocities.is_empty() && point.efforts.is_empty()
            {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} has no command values"
                )));
            }
            for (field, values) in [
                ("positions", &point.positions),
                ("velocities", &point.velocities),
                ("efforts", &point.efforts),
            ] {
                if !values.is_empty() && values.len() != joint_count {
                    return Err(RosBridgeError::InvalidTrajectory(format!(
                        "point {point_index} {field} length {} does not match {joint_count} joints",
                        values.len()
                    )));
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(RosBridgeError::InvalidTrajectory(format!(
                        "point {point_index} {field} contains a non-finite value"
                    )));
                }
            }
            if point
                .efforts
                .iter()
                .any(|value| value.abs() > f32::MAX as f64)
            {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} effort exceeds interlock numeric range"
                )));
            }
        }
        Ok(())
    }
}

impl RosBridgeNode {
    /// Create a new bridge node with the given configuration.
    pub fn new(config: RosBridgeConfig) -> Self {
        #[cfg(feature = "ros2")]
        let receive_epoch = std::sync::Arc::new(std::time::Instant::now());
        #[cfg(feature = "ros2")]
        let receive_epoch_id = format!(
            "ros-bridge:{}:{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        );

        Self {
            config,
            is_running: false,
            phi: std::sync::Mutex::new(0.0),
            interlock: std::sync::Mutex::new(symthaea_sim_bridge::AmygdalaInterlock::default()),
            surprise_monitor: std::sync::Mutex::new(symthaea_sim_bridge::SurpriseMonitor::default()),
            predicted_state_hv: std::sync::Mutex::new(None),
            telemetry_broadcaster: std::sync::Arc::new(
                symthaea_telemetry_grpc::TelemetryBroadcaster::new(),
            ),
            #[cfg(feature = "ros2")]
            latest_joint_state: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            latest_imu: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            latest_joint_state_receipt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            latest_imu_receipt: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            receive_epoch,
            #[cfg(feature = "ros2")]
            receive_epoch_id,
            #[cfg(feature = "ros2")]
            context: None,
            #[cfg(feature = "ros2")]
            node: None,
            #[cfg(feature = "ros2")]
            trajectory_pub: None,
            #[cfg(feature = "ros2")]
            status_pub: None,
            #[cfg(feature = "ros2")]
            joint_state_sub: None,
            #[cfg(feature = "ros2")]
            imu_sub: None,
            #[cfg(feature = "ros2")]
            scheduler: symthaea_control::ConsciousnessGainScheduler::default(),
            #[cfg(feature = "ros2")]
            spin_thread: None,
        }
    }

    /// Initialize ROS endpoints when the `ros2` feature is enabled and start telemetry.
    ///
    /// Without `ros2`, initialization is telemetry-only; robot trajectory
    /// publication remains unavailable and fails explicitly.
    pub fn init(&mut self) -> Result<(), RosBridgeError> {
        #[cfg(feature = "ros2")]
        {
            let context = std::sync::Arc::new(
                rclrs::Context::new(std::env::args())
                    .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?,
            );
            let node = rclrs::create_node(&context, &self.config.node_name)
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let trajectory_topic = format!("/{}/joint_trajectory", self.config.namespace);
            let trajectory_pub = node
                .create_publisher::<trajectory_msgs::msg::JointTrajectory>(
                    &trajectory_topic,
                    rclrs::QOS_PROFILE_DEFAULT,
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let status_topic = format!("/{}/status", self.config.namespace);
            let status_pub = node
                .create_publisher::<std_msgs::msg::String>(
                    &status_topic,
                    rclrs::QOS_PROFILE_DEFAULT,
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            // 1. Subscribe to joint states and IMU feedback
            let latest_joint_state_clone = self.latest_joint_state.clone();
            let latest_joint_state_receipt = self.latest_joint_state_receipt.clone();
            let joint_receive_epoch = self.receive_epoch.clone();
            let joint_receive_epoch_id = self.receive_epoch_id.clone();
            let joint_state_sub = node
                .create_subscription::<sensor_msgs::msg::JointState, _>(
                    "/joint_states",
                    rclrs::QOS_PROFILE_DEFAULT,
                    move |msg: sensor_msgs::msg::JointState| {
                        // Lock receipt first, then message: readers use the same
                        // order so message and receipt remain one atomic snapshot.
                        let mut receipt_lock = latest_joint_state_receipt.lock().unwrap();
                        let sequence = receipt_lock
                            .as_ref()
                            .map_or(1, |receipt| receipt.sequence.saturating_add(1));
                        let received_monotonic_ns = joint_receive_epoch
                            .elapsed()
                            .as_nanos()
                            .min(u64::MAX as u128) as u64;
                        let mut message_lock = latest_joint_state_clone.lock().unwrap();
                        *message_lock = Some(msg);
                        *receipt_lock = Some(RosSampleReceiptV1 {
                            sequence,
                            received_monotonic_ns,
                            receive_epoch_id: joint_receive_epoch_id.clone(),
                        });
                    },
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let latest_imu_clone = self.latest_imu.clone();
            let latest_imu_receipt = self.latest_imu_receipt.clone();
            let imu_receive_epoch = self.receive_epoch.clone();
            let imu_receive_epoch_id = self.receive_epoch_id.clone();
            let imu_sub = node
                .create_subscription::<sensor_msgs::msg::Imu, _>(
                    "/imu",
                    rclrs::QOS_PROFILE_DEFAULT,
                    move |msg: sensor_msgs::msg::Imu| {
                        let mut receipt_lock = latest_imu_receipt.lock().unwrap();
                        let sequence = receipt_lock
                            .as_ref()
                            .map_or(1, |receipt| receipt.sequence.saturating_add(1));
                        let received_monotonic_ns = imu_receive_epoch
                            .elapsed()
                            .as_nanos()
                            .min(u64::MAX as u128) as u64;
                        let mut message_lock = latest_imu_clone.lock().unwrap();
                        *message_lock = Some(msg);
                        *receipt_lock = Some(RosSampleReceiptV1 {
                            sequence,
                            received_monotonic_ns,
                            receive_epoch_id: imu_receive_epoch_id.clone(),
                        });
                    },
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            // 2. Spawn the background spinning executor thread
            let node_clone = node.clone();
            let spin_thread = std::thread::spawn(move || {
                if let Err(e) = rclrs::spin(node_clone) {
                    tracing::error!("ROS2 spin thread error: {:?}", e);
                }
            });

            self.context = Some(context);
            self.node = Some(node);
            self.trajectory_pub = Some(trajectory_pub);
            self.status_pub = Some(status_pub);
            self.joint_state_sub = Some(joint_state_sub);
            self.imu_sub = Some(imu_sub);
            self.spin_thread = Some(spin_thread);
        }
        let broadcaster_clone = self.telemetry_broadcaster.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let addr = "127.0.0.1:50051".parse().unwrap();
                if let Err(e) = broadcaster_clone.run(addr).await {
                    tracing::error!("gRPC Telemetry server error: {:?}", e);
                }
            });
        });
        self.is_running = true;
        Ok(())
    }

    /// Update the surprise monitor based on similarity between actual and expected proprioception vectors.
    #[cfg(feature = "ros2")]
    pub fn update_surprise_monitor(&self) -> Result<(), RosBridgeError> {
        let latest_joint_state_opt = self.latest_joint_state.lock().unwrap().clone();
        if let Some(joint_state) = latest_joint_state_opt {
            let canonical_positions = canonicalize_joint_values(
                &joint_state.name,
                &joint_state.position,
                &symthaea_humanoid::types::JOINT_NAMES,
                "position",
            )?;
            let joint_angles_f32: Vec<f32> =
                canonical_positions.iter().map(|&p| p as f32).collect();
            let current_hv = self.process_joint_states(&joint_angles_f32);
            let predicted_opt = self.predicted_state_hv.lock().unwrap().clone();
            if let Some(predicted_hv) = predicted_opt {
                let similarity = current_hv.similarity(&predicted_hv);
                let surprise = (1.0 - similarity as f64).clamp(0.0, 1.0);
                let mut surprise_monitor = self.surprise_monitor.lock().unwrap();
                surprise_monitor.update(surprise);
            }
        }
        Ok(())
    }

    /// Return a complete humanoid state only when the subscribed evidence supports it.
    ///
    /// This API intentionally remains unavailable until root pose and root/COM
    /// linear velocity are supplied by qualified, frame- and clock-bound sources.
    /// Use `get_partial_sensor_observation()` to inspect available fields without
    /// promoting partial measurements to a complete `HumanoidState`.
    #[cfg(feature = "ros2")]
    pub fn get_current_humanoid_state(
        &self,
    ) -> Result<symthaea_humanoid::types::HumanoidState, RosBridgeError> {
        let observation = self.get_partial_sensor_observation()?;
        let imu = match observation.imu {
            RosEvidenceValueV1::Available { value, .. } => value,
            RosEvidenceValueV1::Unavailable { reason, .. } => {
                return Err(RosBridgeError::SensorDataUnavailable(reason));
            }
        };

        if let RosEvidenceValueV1::Unavailable { reason, .. } = &imu.orientation_wxyz {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } =
            &observation.joint_state.velocities
        {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } = &imu.angular_velocity_body_rad_s {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } =
            &observation.root_linear_velocity_world_mps
        {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } =
            &observation.center_of_mass_velocity_world_mps
        {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } = &observation.root_position_world_m {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } = &observation.head_height_world_m {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }
        if let RosEvidenceValueV1::Unavailable { reason, .. } = &observation.extremities_world_m {
            return Err(RosBridgeError::SensorDataUnavailable(reason.clone()));
        }

        Err(RosBridgeError::SensorDataUnavailable(
            "complete humanoid state construction is blocked: every required field must be explicitly available and qualified".into(),
        ))
    }

    /// Capture currently available ROS sensor fields without fabricating a complete state.
    ///
    /// This method preserves each sensor's own header stamp/frame. Clock domains,
    /// callback receive times, and calibration identities are intentionally left
    /// unbound because the current subscriptions do not establish those facts.
    /// It does not claim cross-sensor synchronization or world-frame transforms.
    #[cfg(feature = "ros2")]
    pub fn get_partial_sensor_observation(
        &self,
    ) -> Result<RosPartialHumanoidObservationV1, RosBridgeError> {
        // All receipt/message readers use receipt-lock then message-lock,
        // matching each callback's write order for an atomic pair.
        let joint_receipt_guard = self
            .latest_joint_state_receipt
            .lock()
            .map_err(|_| RosBridgeError::Communication("joint-state receipt lock poisoned".into()))?;
        let joint_state_guard = self
            .latest_joint_state
            .lock()
            .map_err(|_| RosBridgeError::Communication("joint-state lock poisoned".into()))?;
        let joint_state = joint_state_guard
            .clone()
            .ok_or(RosBridgeError::NotInitialized)?;
        let joint_receipt = joint_receipt_guard.clone();
        drop(joint_state_guard);
        drop(joint_receipt_guard);

        let imu_receipt_guard = self
            .latest_imu_receipt
            .lock()
            .map_err(|_| RosBridgeError::Communication("IMU receipt lock poisoned".into()))?;
        let imu_guard = self
            .latest_imu
            .lock()
            .map_err(|_| RosBridgeError::Communication("IMU lock poisoned".into()))?;
        let imu = imu_guard.clone();
        let imu_receipt = imu_receipt_guard.clone();
        drop(imu_guard);
        drop(imu_receipt_guard);

        // Capture after taking both snapshots so the receipt-age estimate cannot
        // predate a message that was selected while waiting for the locks.
        let captured_monotonic_ns = self
            .receive_epoch
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        let receive_epoch_id = self.receive_epoch_id.clone();
        let joint_source = ros_sample_metadata(
            &joint_state.header,
            joint_receipt,
            captured_monotonic_ns,
            &receive_epoch_id,
        )?;
        let expected = &symthaea_humanoid::types::JOINT_NAMES;
        let positions = canonicalize_joint_values(
            &joint_state.name,
            &joint_state.position,
            expected,
            "position",
        )?;
        let velocities = if joint_state.velocity.is_empty() {
            RosEvidenceValueV1::Unavailable {
                reason: "JointState velocity array is empty; velocities were not reported".into(),
                source: Some(joint_source.clone()),
            }
        } else {
            RosEvidenceValueV1::Available {
                value: canonicalize_joint_values(
                    &joint_state.name,
                    &joint_state.velocity,
                    expected,
                    "velocity",
                )?,
                source: joint_source.clone(),
            }
        };
        let efforts = if joint_state.effort.is_empty() {
            RosEvidenceValueV1::Unavailable {
                reason: "JointState effort array is empty; efforts were not reported".into(),
                source: Some(joint_source.clone()),
            }
        } else {
            RosEvidenceValueV1::Available {
                value: canonicalize_joint_values(
                    &joint_state.name,
                    &joint_state.effort,
                    expected,
                    "effort",
                )?,
                source: joint_source.clone(),
            }
        };

        let imu_evidence = match imu {
            None => RosEvidenceValueV1::Unavailable {
                reason: "No IMU message has been received".into(),
                source: None,
            },
            Some(imu) => {
                let source = ros_sample_metadata(
                    &imu.header,
                    imu_receipt.clone(),
                    captured_monotonic_ns,
                    &receive_epoch_id,
                )?;
                let orientation_covariance =
                    ros_covariance_v1(imu.orientation_covariance, "orientation")?;
                let angular_velocity_covariance =
                    ros_covariance_v1(imu.angular_velocity_covariance, "angular velocity")?;
                let linear_acceleration_covariance =
                    ros_covariance_v1(imu.linear_acceleration_covariance, "linear acceleration")?;

                let orientation_wxyz = if matches!(
                    orientation_covariance,
                    RosCovarianceV1::Unavailable
                ) {
                    RosEvidenceValueV1::Unavailable {
                        reason: "ROS IMU marks orientation estimate unavailable (covariance[0] = -1)".into(),
                        source: Some(source.clone()),
                    }
                } else {
                    RosEvidenceValueV1::Available {
                        value: normalize_imu_quaternion([
                            imu.orientation.w,
                            imu.orientation.x,
                            imu.orientation.y,
                            imu.orientation.z,
                        ])?,
                        source: source.clone(),
                    }
                };
                let angular_velocity_body_rad_s = evidence_from_imu_vector(
                    [
                        imu.angular_velocity.x,
                        imu.angular_velocity.y,
                        imu.angular_velocity.z,
                    ],
                    &angular_velocity_covariance,
                    &source,
                    "IMU angular velocity",
                )?;
                let linear_acceleration_sensor_frame_mps2 = evidence_from_imu_vector(
                    [
                        imu.linear_acceleration.x,
                        imu.linear_acceleration.y,
                        imu.linear_acceleration.z,
                    ],
                    &linear_acceleration_covariance,
                    &source,
                    "IMU linear acceleration",
                )?;

                RosEvidenceValueV1::Available {
                    value: RosImuObservationV1 {
                        orientation_wxyz,
                        orientation_covariance,
                        angular_velocity_body_rad_s,
                        angular_velocity_covariance,
                        linear_acceleration_sensor_frame_mps2,
                        linear_acceleration_covariance,
                        source,
                    },
                    source: source.clone(),
                }
            }
        };

        Ok(RosPartialHumanoidObservationV1 {
            schema_version: 1,
            morphology_profile_id: "symthaea.humanoid.dmc21".into(),
            captured_monotonic_ns,
            receive_epoch_id,
            joint_state: RosJointStateObservationV1 {
                joint_names: expected.iter().map(|name| (*name).to_string()).collect(),
                positions: RosEvidenceValueV1::Available {
                    value: positions,
                    source: joint_source.clone(),
                },
                velocities,
                efforts,
                source: joint_source,
            },
            imu: imu_evidence,
            root_position_world_m: RosEvidenceValueV1::Unavailable {
                reason: "No qualified world-position source or frame transform is subscribed".into(),
                source: None,
            },
            root_linear_velocity_world_mps: RosEvidenceValueV1::Unavailable {
                reason: "No qualified odometry/state-estimator velocity source is subscribed".into(),
                source: None,
            },
            center_of_mass_velocity_world_mps: RosEvidenceValueV1::Unavailable {
                reason: "Center-of-mass velocity is not measured or estimated by this bridge".into(),
                source: None,
            },
            head_height_world_m: RosEvidenceValueV1::Unavailable {
                reason: "Head height requires qualified kinematics/model geometry or a tracked frame".into(),
                source: None,
            },
            extremities_world_m: RosEvidenceValueV1::Unavailable {
                reason: "Extremity positions require qualified kinematics and frame transforms".into(),
                source: None,
            },
        })
    }

    /// Log status and trajectory states to a local CSV file for offline replay and IIT diagnostics.
    pub fn log_telemetry(
        &self,
        status: &ConsciousnessStatusMsg,
        joint_names: &[String],
        positions: &[f64],
    ) -> Result<(), RosBridgeError> {
        let log_dir = "logs/telemetry";
        std::fs::create_dir_all(log_dir)
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        let path = format!("{}/episode_log.csv", log_dir);
        let file_exists = std::path::Path::new(&path).exists();

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        use std::io::Write;
        if !file_exists {
            writeln!(
                file,
                "timestamp,phi,arousal,uncertainty,joint_count,first_joint_pos"
            )
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;
        }

        let first_pos = positions.first().copied().unwrap_or(0.0);
        writeln!(
            file,
            "{},{},{},{},{},{}",
            status.timestamp,
            status.phi,
            status.arousal,
            status.uncertainty,
            joint_names.len(),
            first_pos
        )
        .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        Ok(())
    }

    /// Publish consciousness status to the `/symthaea/status` topic.
    pub fn publish_status(&self, status: ConsciousnessStatusMsg) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }
        status.validate()?;

        // Store the latest Φ value for gain scheduling
        {
            let mut stored_phi = self.phi.lock().unwrap();
            *stored_phi = status.phi;
        }

        #[cfg(feature = "ros2")]
        {
            if let Some(pub_status) = &self.status_pub {
                let serialized = serde_json::to_string(&status)
                    .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;
                let mut msg = std_msgs::msg::String::default();
                msg.data = serialized;
                pub_status
                    .publish(&msg)
                    .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;
            }
        }

        // Broadcast live telemetry via gRPC stream
        let surprise = self.surprise_monitor.lock().unwrap().current_surprise;
        let frame = symthaea_telemetry_grpc::TelemetryFrame {
            phi: status.phi,
            harmonies: status.harmonies.to_vec(),
            neuromodulators: status.neuromodulators.to_vec(),
            arousal: status.arousal,
            uncertainty: status.uncertainty,
            surprise,
            timestamp: status.timestamp.clone(),
            mental_movie: None,
            gwt_broadcast: 0.0,
            hot_metacognitive: 0.0,
            ast_temporal: 0.0,
            knowledge_coherence: 0.0,
            embodiment_level: 0.0,
            self_awareness: 0.0,
            topological_unity: 1.0,
        };
        self.telemetry_broadcaster.broadcast(frame);
        tracing::debug!(
            "ROS2 [{}] Publish Status: Φ={:.4}, Arousal={:.2}",
            self.config.node_name,
            status.phi,
            status.arousal
        );
        Ok(())
    }

    /// Submit a joint trajectory through the live ROS publisher, applying gain scheduling and safety interlocks.
    ///
    /// A successful return means only that the local ROS publisher accepted the
    /// message. It does not prove subscriber delivery, controller acceptance,
    /// actuator latch, or physical movement.
    pub fn publish_trajectory(
        &self,
        mut command: RosTrajectoryCommand,
    ) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }
        command.validate()?;

        // 1. Amygdala Interlock safety checks
        let mut peak_torque = 0.0f32;
        for pt in &command.points {
            for &eff in &pt.efforts {
                if eff.abs() as f32 > peak_torque {
                    peak_torque = eff.abs() as f32;
                }
            }
        }

        let mut interlock = self.interlock.lock().unwrap();
        let safety_status = interlock.monitor(peak_torque, 0.0);

        if safety_status == symthaea_sim_bridge::SafetyStatus::Red {
            tracing::warn!(
                "ROS2 Amygdala Interlock E-STOP triggered; trajectory was not published"
            );
            return Err(RosBridgeError::SafetyInterlock(
                "emergency stop is latched; explicitly reset it before sending commands".into(),
            ));
        }
        if safety_status == symthaea_sim_bridge::SafetyStatus::Yellow
            && command
                .points
                .iter()
                .any(|point| !point.positions.is_empty() || !point.velocities.is_empty())
        {
            return Err(RosBridgeError::SafetyInterlock(
                "yellow state cannot safely damp position/velocity trajectories".into(),
            ));
        }

        // Apply any dampening or clamping overrides
        for pt in &mut command.points {
            for eff in &mut pt.efforts {
                *eff = interlock.apply_override(*eff as f32) as f64;
            }
        }

        // Publish only through a live ROS publisher. Telemetry-only builds and
        // partially initialized ROS nodes must never report command success.
        #[cfg(feature = "ros2")]
        {
            let phi = *self.phi.lock().unwrap();
            let gain_scale = self.scheduler.alpha(phi);
            for pt in &mut command.points {
                for eff in &mut pt.efforts {
                    *eff *= gain_scale;
                }
            }

            let pub_traj = self
                .trajectory_pub
                .as_ref()
                .ok_or(RosBridgeError::RosTransportUnavailable)?;

            let joint_count = command.joint_names.len();
            let point_count = command.points.len();
            let mut msg = trajectory_msgs::msg::JointTrajectory::default();
            msg.joint_names = command.joint_names;

            for pt in command.points {
                let mut point = trajectory_msgs::msg::JointTrajectoryPoint::default();
                point.positions = pt.positions;
                point.velocities = pt.velocities;
                point.effort = pt.efforts;

                let sec = (pt.time_from_start_ns / 1_000_000_000) as i32;
                let nanosec = (pt.time_from_start_ns % 1_000_000_000) as u32;
                point.time_from_start = builtin_interfaces::msg::Duration { sec, nanosec };

                msg.points.push(point);
            }

            pub_traj
                .publish(&msg)
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            tracing::debug!(
                "ROS2 [{}] Publish Trajectory: {} joints, {} points, Φ scale={:.2}",
                self.config.node_name,
                joint_count,
                point_count,
                gain_scale
            );
            Ok(())
        }

        #[cfg(not(feature = "ros2"))]
        {
            // All local validation/interlock checks above still run, but no
            // ROS command transport exists in this build.
            Err(RosBridgeError::RosTransportUnavailable)
        }
    }

    /// Process a proprioception update and return an HDC-encoded sensor vector.
    pub fn process_joint_states(&self, joint_angles: &[f32]) -> ContinuousHV {
        let dimension = 16384;
        if joint_angles.is_empty() {
            return ContinuousHV::zero(dimension);
        }

        let mut hv = ContinuousHV::zero(dimension);
        let mut first = true;

        for (i, &angle) in joint_angles.iter().enumerate() {
            let joint_base = ContinuousHV::random(dimension, (i as u64) ^ 0x5E5C0DE);
            let sensation = joint_base.scale(angle);

            if first {
                hv = sensation;
                first = false;
            } else {
                hv = ContinuousHV::bundle(&[&hv, &sensation]);
            }
        }

        hv.normalize()
    }

    /// Converts a Symthaea HumanoidCommand into a ROS2 JointTrajectory format, using torque/effort control.
    pub fn command_to_ros_trajectory(
        &self,
        command: &symthaea_humanoid::types::HumanoidCommand,
        joint_names: Vec<String>,
        time_from_start_ns: u64,
    ) -> RosTrajectoryCommand {
        let efforts = command.torques.iter().map(|&t| t as f64).collect();
        let point = RosTrajectoryPoint {
            positions: vec![],
            velocities: vec![],
            efforts,
            time_from_start_ns,
        };
        RosTrajectoryCommand {
            joint_names,
            points: vec![point],
        }
    }

    // No JointState-only -> HumanoidState converter is exposed here: the
    // available ROS topics do not provide a complete, qualified root pose and
    // linear-velocity observation. See get_current_humanoid_state(), which fails
    // closed until those evidence sources exist.

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = RosBridgeConfig::default();
        assert_eq!(config.node_name, "symthaea_brain_bridge");
    }

    #[test]
    fn test_node_lifecycle() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        assert!(
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.8,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .is_err()
        );

        node.init().unwrap();
        assert!(
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.8,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .is_ok()
        );
    }

    #[test]
    fn test_amygdala_safety_and_gain_scaling() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();

        let normal_cmd = RosTrajectoryCommand {
            joint_names: vec!["joint_1".to_string()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: vec![0.0],
                efforts: vec![10.0],
                time_from_start_ns: 1000000,
            }],
        };

        node.publish_status(ConsciousnessStatusMsg {
            phi: 1.0,
            harmonies: [0.5; 8],
            neuromodulators: [0.5; 4],
            arousal: 0.5,
            uncertainty: 0.1,
            timestamp: "now".to_string(),
        })
        .unwrap();

        #[cfg(feature = "ros2")]
        node.publish_trajectory(normal_cmd.clone()).unwrap();
        #[cfg(not(feature = "ros2"))]
        assert!(matches!(
            node.publish_trajectory(normal_cmd.clone()),
            Err(RosBridgeError::RosTransportUnavailable)
        ));

        {
            let mut interlock = node.interlock.lock().unwrap();
            assert_eq!(interlock.status(), symthaea_sim_bridge::SafetyStatus::Green);

            // Peak torque exceeding limit * 1.5 triggers RED
            let status = interlock.monitor(200.0, 0.0);
            assert_eq!(status, symthaea_sim_bridge::SafetyStatus::Red);
            assert_eq!(interlock.apply_override(10.0), 0.0);
        }

        #[cfg(feature = "ros2")]
        {
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.0,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .unwrap();
            // A lower gain must not clear an already latched emergency stop.
            let cmd = normal_cmd.clone();
            assert!(matches!(
                node.publish_trajectory(cmd),
                Err(RosBridgeError::SafetyInterlock(_))
            ));
        }
    }

    #[cfg(feature = "ros2")]
    #[test]
    fn ros2_build_without_live_publisher_rejects_trajectory() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        // Simulate inconsistent lifecycle state without constructing ROS context.
        node.is_running = true;
        node.trajectory_pub = None;

        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };

        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::RosTransportUnavailable)
        ));
    }

    #[cfg(not(feature = "ros2"))]
    #[test]
    fn telemetry_only_build_rejects_robot_trajectory_publication() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        // Isolate command-path behavior without starting a telemetry server.
        node.is_running = true;

        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };

        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::RosTransportUnavailable)
        ));
    }

    #[test]
    fn invalid_and_estopped_trajectories_fail_closed() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();

        let invalid = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into(), "joint_2".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![f64::NAN, 0.0],
                velocities: Vec::new(),
                efforts: Vec::new(),
                time_from_start_ns: 1_000_000,
            }],
        };
        assert!(matches!(
            node.publish_trajectory(invalid),
            Err(RosBridgeError::InvalidTrajectory(_))
        ));

        node.interlock.lock().unwrap().trigger_emergency_stop();
        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: Vec::new(),
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };
        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::SafetyInterlock(_))
        ));
    }

    #[test]
    fn non_finite_status_is_rejected_before_gain_scheduling() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();
        let result = node.publish_status(ConsciousnessStatusMsg {
            phi: f64::NAN,
            harmonies: [0.5; 8],
            neuromodulators: [0.5; 4],
            arousal: 0.5,
            uncertainty: 0.1,
            timestamp: "now".into(),
        });
        assert!(matches!(result, Err(RosBridgeError::InvalidStatus(_))));
    }

    #[test]
    fn joint_state_values_are_canonicalized_by_name_not_array_order() {
        let expected = ["hip", "knee", "ankle"];
        let first_names = vec!["hip".into(), "knee".into(), "ankle".into()];
        let first_values = vec![0.1, -0.2, 0.3];
        let shuffled_names = vec!["ankle".into(), "hip".into(), "knee".into()];
        let shuffled_values = vec![0.3, 0.1, -0.2];

        let first =
            canonicalize_joint_values(&first_names, &first_values, &expected, "position")
                .unwrap();
        let shuffled =
            canonicalize_joint_values(&shuffled_names, &shuffled_values, &expected, "position")
                .unwrap();

        assert_eq!(first, vec![0.1, -0.2, 0.3]);
        assert_eq!(shuffled, first);
    }

    #[test]
    fn joint_state_canonicalization_rejects_duplicate_missing_unknown_and_nonfinite_data() {
        let expected = ["hip", "knee", "ankle"];
        let values = vec![0.1, -0.2, 0.3];

        let duplicate = vec!["hip".into(), "hip".into(), "ankle".into()];
        assert!(matches!(
            canonicalize_joint_values(&duplicate, &values, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let missing = vec!["hip".into(), "knee".into()];
        assert!(matches!(
            canonicalize_joint_values(&missing, &values[..2], &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let unknown = vec!["hip".into(), "knee".into(), "finger".into()];
        assert!(matches!(
            canonicalize_joint_values(&unknown, &values, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let nonfinite = vec![0.1, f64::NAN, 0.3];
        let names = vec!["hip".into(), "knee".into(), "ankle".into()];
        assert!(matches!(
            canonicalize_joint_values(&names, &nonfinite, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        assert!(matches!(
            canonicalize_joint_values(&names, &values[..2], &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));
    }

    #[test]
    fn receive_age_requires_same_epoch_and_nonfuture_receipt() {
        let receipt = RosSampleReceiptV1 {
            sequence: 7,
            received_monotonic_ns: 100,
            receive_epoch_id: "epoch-A".into(),
        };
        assert_eq!(receive_age_at_capture_ns(&receipt, "epoch-A", 135), Some(35));
        assert_eq!(receive_age_at_capture_ns(&receipt, "epoch-B", 135), None);
        assert_eq!(receive_age_at_capture_ns(&receipt, "epoch-A", 99), None);
    }

    fn temporal_test_sample(
        stamp_sec: i32,
        stamp_nanosec: u32,
        clock_domain: RosClockDomainV1,
    ) -> RosSampleMetadataV1 {
        RosSampleMetadataV1 {
            stamp_sec,
            stamp_nanosec,
            frame_id: "base_link".into(),
            clock_domain,
            source_clock_epoch_id: Some("source-epoch-1".into()),
            received_monotonic_ns: Some(90),
            receive_sequence: Some(1),
            receive_epoch_id: Some("receipt-epoch".into()),
            receive_age_at_capture_ns: Some(10),
            calibration_id: None,
        }
    }

    fn temporal_test_policy() -> RosTemporalPolicyV1 {
        RosTemporalPolicyV1 {
            required_source_clock_domain: "robot-sim-clock".into(),
            maximum_source_age_ns: 1_000_000_000,
            maximum_source_skew_ns: 20_000_000,
            maximum_bridge_residence_age_ns: 1_000,
            reject_zero_source_stamp: true,
            reject_duplicate_source_stamp: true,
        }
    }

    fn temporal_test_now() -> RosClockNowV1 {
        RosClockNowV1 {
            domain: RosClockDomainV1::Named("robot-sim-clock".into()),
            source_clock_epoch_id: "source-epoch-1".into(),
            timestamp_ns: 100_000_000_000,
        }
    }

    #[test]
    fn temporal_admission_accepts_only_bound_fresh_same_epoch_pair() {
        let joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        let imu = temporal_test_sample(
            100,
            5_000_000,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );

        let now = RosClockNowV1 {
            domain: RosClockDomainV1::Named("robot-sim-clock".into()),
            source_clock_epoch_id: "source-epoch-1".into(),
            timestamp_ns: 100_010_000_000,
        };
        assert_eq!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &now,
                &temporal_test_policy(),
                "receipt-epoch",
                100,
            ),
            RosTemporalDispositionV1::Fresh {
                joint_source_age_ns: 10_000_000,
                imu_source_age_ns: 5_000_000,
                source_skew_ns: 5_000_000,
                joint_bridge_residence_age_ns: 10,
                imu_bridge_residence_age_ns: 10,
            }
        );
    }

    #[test]
    fn temporal_admission_rejects_unbound_or_mismatched_source_clocks() {
        let joint = temporal_test_sample(100, 0, RosClockDomainV1::Unbound);
        let imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::Unsynchronized { .. }
        ));

        let joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        let imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("other-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::Unsynchronized { .. }
        ));
    }

    #[test]
    fn temporal_admission_rejects_stale_future_zero_and_skewed_sources() {
        let imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );

        let stale = temporal_test_sample(
            98,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &stale,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::StaleSource {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        let mut future = temporal_test_sample(
            101,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &future,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::InvalidClock {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        future = temporal_test_sample(
            0,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &future,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::InvalidClock {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        let skewed = temporal_test_sample(
            100,
            100_000_000,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        let now = RosClockNowV1 {
            domain: RosClockDomainV1::Named("robot-sim-clock".into()),
            source_clock_epoch_id: "source-epoch-1".into(),
            timestamp_ns: 100_200_000_000,
        };
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &imu,
                Some(&skewed),
                &now,
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::ExcessiveSourceSkew { .. }
        ));
    }

    #[test]
    fn temporal_admission_requires_receipts_that_match_capture_epoch_and_age() {
        let mut joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        let imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );

        joint.receive_epoch_id = Some("old-epoch".into());
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::ReceiptUnavailable {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        joint.receive_age_at_capture_ns = Some(99);
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::ReceiptUnavailable {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        joint.receive_age_at_capture_ns = Some(10_000);
        joint.received_monotonic_ns = Some(90_000);
        let mut late_imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        late_imu.receive_age_at_capture_ns = Some(10_000);
        late_imu.received_monotonic_ns = Some(90_000);
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&late_imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100_000
            ),
            RosTemporalDispositionV1::BridgeResidenceExpired {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));
    }

    #[test]
    fn temporal_admission_rejects_malformed_stamps_and_unverifiable_receipts() {
        let valid_domain = RosClockDomainV1::Named("robot-sim-clock".into());
        let imu = temporal_test_sample(100, 0, valid_domain.clone());

        let invalid_nanosec = temporal_test_sample(100, 1_000_000_000, valid_domain.clone());
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &invalid_nanosec,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::InvalidClock {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        let negative_stamp = temporal_test_sample(-1, 0, valid_domain.clone());
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &negative_stamp,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::InvalidClock {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        let mut no_sequence = temporal_test_sample(100, 0, valid_domain.clone());
        no_sequence.receive_sequence = None;
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &no_sequence,
                Some(&imu),
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::ReceiptUnavailable {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));

        let wrong_now = RosClockNowV1 {
            domain: RosClockDomainV1::Named("wall-clock".into()),
            source_clock_epoch_id: "source-epoch-1".into(),
            timestamp_ns: 100_000_000_000,
        };
        let joint = temporal_test_sample(100, 0, valid_domain);
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &wrong_now,
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::Unsynchronized { .. }
        ));
    }

    #[test]
    fn temporal_admission_rejects_missing_imu_and_invalid_policy() {
        let joint = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                None,
                &temporal_test_now(),
                &temporal_test_policy(),
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::SourceUnavailable {
                stream: RosTemporalStreamV1::Imu,
                ..
            }
        ));

        let mut invalid_policy = temporal_test_policy();
        invalid_policy.required_source_clock_domain.clear();
        let imu = temporal_test_sample(
            100,
            0,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        assert!(matches!(
            evaluate_ros_temporal_pair_v1(
                &joint,
                Some(&imu),
                &temporal_test_now(),
                &invalid_policy,
                "receipt-epoch",
                100
            ),
            RosTemporalDispositionV1::InvalidPolicy { .. }
        ));
    }

    fn temporal_history_sample(
        sequence: u64,
        stamp_sec: i32,
        stamp_nanosec: u32,
    ) -> RosSampleMetadataV1 {
        let mut sample = temporal_test_sample(
            stamp_sec,
            stamp_nanosec,
            RosClockDomainV1::Named("robot-sim-clock".into()),
        );
        sample.receive_sequence = Some(sequence);
        sample.received_monotonic_ns = Some(95);
        sample.receive_age_at_capture_ns = Some(5);
        sample
    }

    fn temporal_history_now() -> RosClockNowV1 {
        RosClockNowV1 {
            domain: RosClockDomainV1::Named("robot-sim-clock".into()),
            source_clock_epoch_id: "source-epoch-1".into(),
            timestamp_ns: 100_010_000_000,
        }
    }

    #[test]
    fn temporal_history_rejects_replay_and_keeps_pair_update_atomic() {
        let policy = temporal_test_policy();
        let mut history = RosTemporalHistoryV1::new("receipt-epoch", "source-epoch-1").unwrap();
        let joint_one = temporal_history_sample(1, 100, 0);
        let imu_one = temporal_history_sample(1, 100, 0);

        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &joint_one,
                Some(&imu_one),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted { .. }
        ));

        let joint_two = temporal_history_sample(2, 100, 2_000_000);
        // A fresh joint sample paired with replayed IMU sequence 1 must reject
        // atomically; JointState history must not advance on half a pair.
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &joint_two,
                Some(&imu_one),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::SequenceNotIncreasing {
                stream: RosTemporalStreamV1::Imu,
                previous: 1,
                current: 1
            }
        ));
        assert_eq!(
            history.last_joint_state.as_ref().unwrap().receive_sequence,
            1
        );

        let imu_two = temporal_history_sample(2, 100, 3_000_000);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &joint_two,
                Some(&imu_two),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted {
                joint_sequence_gap: 0,
                imu_sequence_gap: 0
            }
        ));
        assert_eq!(
            history.last_joint_state.as_ref().unwrap().receive_sequence,
            2
        );
    }

    #[test]
    fn temporal_history_rejects_source_time_regression_and_duplicates() {
        let policy = temporal_test_policy();
        let mut history = RosTemporalHistoryV1::new("receipt-epoch", "source-epoch-1").unwrap();
        let first_joint = temporal_history_sample(1, 100, 0);
        let first_imu = temporal_history_sample(1, 100, 0);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &first_joint,
                Some(&first_imu),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted { .. }
        ));

        let regressed_joint = temporal_history_sample(2, 99, 999_000_000);
        let next_imu = temporal_history_sample(2, 100, 1_000_000);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &regressed_joint,
                Some(&next_imu),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::SourceTimestampRegressed {
                stream: RosTemporalStreamV1::JointState,
                ..
            }
        ));
        assert_eq!(
            history.last_joint_state.as_ref().unwrap().receive_sequence,
            1
        );

        let duplicate_joint = temporal_history_sample(2, 100, 0);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &duplicate_joint,
                Some(&next_imu),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::DuplicateSourceTimestamp {
                stream: RosTemporalStreamV1::JointState,
                timestamp_ns: 100_000_000_000
            }
        ));

        // A profile with coarse source-stamp resolution may explicitly permit
        // identical source timestamps while still rejecting sequence replay.
        let mut coarse_clock_policy = policy.clone();
        coarse_clock_policy.reject_duplicate_source_stamp = false;
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &duplicate_joint,
                Some(&next_imu),
                &temporal_history_now(),
                &coarse_clock_policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted { .. }
        ));
    }

    #[test]
    fn temporal_history_requires_explicit_source_clock_jump_reset() {
        let policy = temporal_test_policy();
        let mut history =
            RosTemporalHistoryV1::new("receipt-epoch", "source-epoch-1").unwrap();
        let first_joint = temporal_history_sample(1, 100, 0);
        let first_imu = temporal_history_sample(1, 100, 0);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &first_joint,
                Some(&first_imu),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted { .. }
        ));

        // Simulated source time has rewound while the bridge's local receive
        // epoch has not. The time adapter must expose a new source-clock epoch.
        let mut rewound_joint = temporal_history_sample(2, 5, 0);
        rewound_joint.source_clock_epoch_id = Some("source-epoch-2".into());
        let mut rewound_imu = temporal_history_sample(2, 5, 1_000_000);
        rewound_imu.source_clock_epoch_id = Some("source-epoch-2".into());
        let rewound_now = RosClockNowV1 {
            domain: RosClockDomainV1::Named("robot-sim-clock".into()),
            source_clock_epoch_id: "source-epoch-2".into(),
            timestamp_ns: 5_010_000_000,
        };
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &rewound_joint,
                Some(&rewound_imu),
                &rewound_now,
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::SourceClockEpochChanged {
                expected_epoch_id,
                observed_epoch_id
            } if expected_epoch_id == "source-epoch-1" && observed_epoch_id == "source-epoch-2"
        ));
        assert_eq!(
            history.last_joint_state.as_ref().unwrap().source_stamp_ns,
            100_000_000_000
        );

        // Reset is an explicit lifecycle action; only after it does the new
        // source-clock timeline become eligible for fresh history.
        history
            .reset_for_epochs("receipt-epoch", "source-epoch-2")
            .unwrap();
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &rewound_joint,
                Some(&rewound_imu),
                &rewound_now,
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted {
                joint_sequence_gap: 0,
                imu_sequence_gap: 0
            }
        ));
    }

    #[test]
    fn temporal_history_requires_explicit_epoch_reset_and_reports_sequence_gaps() {
        let policy = temporal_test_policy();
        let mut history = RosTemporalHistoryV1::new("receipt-epoch", "source-epoch-1").unwrap();
        let joint_one = temporal_history_sample(1, 100, 0);
        let imu_one = temporal_history_sample(1, 100, 0);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &joint_one,
                Some(&imu_one),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted { .. }
        ));

        let joint_three = temporal_history_sample(3, 100, 3_000_000);
        let imu_three = temporal_history_sample(3, 100, 4_000_000);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &joint_three,
                Some(&imu_three),
                &temporal_history_now(),
                &policy,
                "receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted {
                joint_sequence_gap: 1,
                imu_sequence_gap: 1
            }
        ));

        let new_epoch_joint = temporal_history_sample(1, 100, 0);
        let new_epoch_imu = temporal_history_sample(1, 100, 0);
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &new_epoch_joint,
                Some(&new_epoch_imu),
                &temporal_history_now(),
                &policy,
                "new-receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::ReceiveEpochChanged { .. }
        ));

        history
            .reset_for_epochs("new-receipt-epoch", "source-epoch-1")
            .unwrap();
        let mut new_joint = new_epoch_joint;
        new_joint.receive_epoch_id = Some("new-receipt-epoch".into());
        new_joint.source_clock_epoch_id = Some("source-epoch-1".into());
        let mut new_imu = new_epoch_imu;
        new_imu.receive_epoch_id = Some("new-receipt-epoch".into());
        new_imu.source_clock_epoch_id = Some("source-epoch-1".into());
        assert!(matches!(
            evaluate_ros_temporal_pair_with_history_v1(
                &mut history,
                &new_joint,
                Some(&new_imu),
                &temporal_history_now(),
                &policy,
                "new-receipt-epoch",
                100
            ),
            RosTemporalHistoryDispositionV1::Accepted {
                joint_sequence_gap: 0,
                imu_sequence_gap: 0
            }
        ));
    }

    #[test]
    fn ros_covariance_semantics_keep_unknown_separate_from_unavailable() {
        // A zero matrix means the uncertainty is unknown; it is not zero uncertainty.
        assert!(matches!(
            classify_ros_covariance([0.0; 9]).unwrap(),
            RosCovarianceV1::Unknown
        ));

        // ROS marks a missing estimate with covariance[0] = -1.
        let mut unavailable = [0.0; 9];
        unavailable[0] = -1.0;
        unavailable[4] = f64::NAN; // remaining entries must not revive this unavailable estimate
        assert!(matches!(
            classify_ros_covariance(unavailable).unwrap(),
            RosCovarianceV1::Unavailable
        ));

        let mut invalid = [0.0; 9];
        invalid[4] = f64::NAN;
        assert!(classify_ros_covariance(invalid).is_err());
    }

    #[test]
    fn imu_quaternion_is_normalized_and_rejects_degenerate_or_nonfinite_values() {
        let normalized = normalize_imu_quaternion([2.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(normalized, [1.0, 0.0, 0.0, 0.0]);

        assert!(matches!(
            normalize_imu_quaternion([0.0, 0.0, 0.0, 0.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
        assert!(matches!(
            normalize_imu_quaternion([f64::NAN, 0.0, 0.0, 1.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
    }

    #[test]
    fn sensor_vectors_reject_nonfinite_components() {
        assert_eq!(
            validate_sensor_vector("angular velocity", [1.0, -2.0, 3.0]).unwrap(),
            [1.0, -2.0, 3.0]
        );
        assert!(matches!(
            validate_sensor_vector("angular velocity", [0.0, f64::INFINITY, 0.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
    }

    #[test]
    fn test_hdc_encoding_and_surprise() {
        let node = RosBridgeNode::new(RosBridgeConfig::default());
        let mut joints = vec![0.0f32; symthaea_humanoid::types::JOINT_NAMES.len()];
        joints[0] = 0.1;
        joints[1] = -0.2;
        joints[2] = 0.3;
        let hv = node.process_joint_states(&joints);
        assert_eq!(hv.dim(), 16384);

        {
            let mut predicted = node.predicted_state_hv.lock().unwrap();
            *predicted = Some(hv.clone());
        }

        #[cfg(feature = "ros2")]
        {
            assert!(node.get_current_humanoid_state().is_err());

            let mut js = sensor_msgs::msg::JointState::default();
            js.name = symthaea_humanoid::types::JOINT_NAMES
                .iter()
                .map(|name| (*name).to_string())
                .collect();
            js.position = joints.iter().map(|&value| value as f64).collect();
            js.velocity = vec![0.0; js.name.len()];
            *node.latest_joint_state.lock().unwrap() = Some(js);

            assert!(matches!(
                node.get_current_humanoid_state(),
                Err(RosBridgeError::SensorDataUnavailable(reason)) if reason.contains("IMU")
            ));

            let partial = node.get_partial_sensor_observation().unwrap();
            assert_eq!(partial.schema_version, 1);
            assert!(matches!(
                partial.imu,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                partial.root_linear_velocity_world_mps,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                partial.joint_state.efforts,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                partial.head_height_world_m,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                partial.extremities_world_m,
                RosEvidenceValueV1::Unavailable { .. }
            ));

            let mut imu = sensor_msgs::msg::Imu::default();
            // The covariance sentinel is authoritative: do not use even plausible
            // numeric fields when ROS explicitly says the estimate is unavailable.
            imu.orientation_covariance[0] = -1.0;
            imu.angular_velocity_covariance[0] = -1.0;
            imu.linear_acceleration_covariance[0] = -1.0;
            *node.latest_imu.lock().unwrap() = Some(imu);

            let partial = node.get_partial_sensor_observation().unwrap();
            let imu = match partial.imu {
                RosEvidenceValueV1::Available { value, .. } => value,
                RosEvidenceValueV1::Unavailable { .. } => panic!("IMU message existed"),
            };
            assert!(matches!(
                imu.orientation_wxyz,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                imu.angular_velocity_body_rad_s,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                imu.linear_acceleration_sensor_frame_mps2,
                RosEvidenceValueV1::Unavailable { .. }
            ));
            assert!(matches!(
                node.get_current_humanoid_state(),
                Err(RosBridgeError::SensorDataUnavailable(reason)) if reason.contains("orientation")
            ));

            let mut valid_imu = sensor_msgs::msg::Imu::default();
            valid_imu.orientation.w = 1.0;
            *node.latest_imu.lock().unwrap() = Some(valid_imu);
            assert!(matches!(
                node.get_current_humanoid_state(),
                Err(RosBridgeError::SensorDataUnavailable(reason)) if reason.contains("linear velocity")
            ));

            node.update_surprise_monitor().unwrap();
            let surprise = node.surprise_monitor.lock().unwrap().current_surprise;
            assert!(
                surprise < 0.1,
                "Surprise should be low when predicted matches current"
            );
        }
    }

    #[test]
    fn test_telemetry_logging() {
        let node = RosBridgeNode::new(RosBridgeConfig::default());
        let status = ConsciousnessStatusMsg {
            phi: 0.95,
            harmonies: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            neuromodulators: [0.1, 0.2, 0.3, 0.4],
            arousal: 0.85,
            uncertainty: 0.05,
            timestamp: "2026-06-25T09:44:00Z".to_string(),
        };
        let joint_names = vec!["joint_1".to_string()];
        let positions = vec![0.123];

        node.log_telemetry(&status, &joint_names, &positions)
            .unwrap();

        let path = "logs/telemetry/episode_log.csv";
        assert!(std::path::Path::new(path).exists());
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("2026-06-25T09:44:00Z"));
        assert!(content.contains("0.95"));

        let _ = std::fs::remove_file(path);
    }
}
