//! Pure, non-actuating audit primitives for the pinned HumanoidToolBench G1 profile.
//!
//! This crate only projects named joint positions into the published policy state
//! and decodes/encodes the published 36-value policy goal. It does not implement
//! a robot driver, controller, safety projector, hardware adapter, transport,
//! calibration path, or actuator authority.
//!
//! The raw 43-joint profile order is pinned from G1Sonic's declared joint list,
//! but the upstream producer of `observation["joint_qpos"]` has not yet been
//! traced end-to-end. Therefore callers must supply the names that they have
//! independently associated with each qpos value. This audit catches ordering,
//! missing-joint, duplicate-joint, and profile mismatches instead of assuming
//! that vector position is joint identity.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

pub const POLICY_STATE_DIMENSION: usize = 32;
pub const RAW_JOINT_DIMENSION: usize = 43;
pub const ACTION_DIMENSION: usize = 36;
pub const ACTION_SCHEMA_ID: &str = "decoupled_v1";
pub const CONTROL_HZ: u32 = 50;

/// Expected G1Sonic whole-body joint identity set from the pinned upstream profile.
/// The input arrays to `project_policy_state` may be in any order as long as each
/// name remains paired with the value it labels.
pub const RAW_JOINT_NAMES: [&str; RAW_JOINT_DIMENSION] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
];

/// Joint-position fields that form policy state indices 0..30.
/// State index 31 is previous base-height command history, not a measurement.
pub const POLICY_STATE_JOINT_NAMES: [&str; 31] = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingError {
    RawJointCount { names: usize, positions: usize },
    DuplicateJointName(String),
    UnknownJointName(String),
    MissingJointName(String),
    NonFiniteRawPosition(String),
    NonFiniteBaseHeight,
    ActionDimension(usize),
    NonFiniteAction(usize),
}

impl fmt::Display for MappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RawJointCount { names, positions } => {
                write!(f, "expected exactly 43 named positions; received {names} names and {positions} values")
            }
            Self::DuplicateJointName(name) => write!(f, "duplicate raw joint name: {name}"),
            Self::UnknownJointName(name) => write!(f, "unknown raw joint name: {name}"),
            Self::MissingJointName(name) => write!(f, "missing required raw joint name: {name}"),
            Self::NonFiniteRawPosition(name) => write!(f, "non-finite raw position for joint: {name}"),
            Self::NonFiniteBaseHeight => write!(f, "previous base-height command must be finite"),
            Self::ActionDimension(dimension) => {
                write!(f, "expected exactly 36 action values, received {dimension}")
            }
            Self::NonFiniteAction(index) => {
                write!(f, "action value at index {index} is not finite")
            }
        }
    }
}

impl Error for MappingError {}

/// Project named raw joint positions into the 32-value benchmark policy state.
///
/// - Every one of the 43 expected G1 joint names must occur exactly once.
/// - Input order is irrelevant because values are resolved by joint name.
/// - Joint positions are interpreted as radians under the pinned G1 profile.
/// - The last state element is supplied command history in metres, never sensor truth.
pub fn project_policy_state(
    raw_joint_names: &[&str],
    raw_joint_positions_rad: &[f32],
    previous_base_height_command_m: f32,
) -> Result<[f32; POLICY_STATE_DIMENSION], MappingError> {
    if raw_joint_names.len() != RAW_JOINT_DIMENSION
        || raw_joint_positions_rad.len() != RAW_JOINT_DIMENSION
    {
        return Err(MappingError::RawJointCount {
            names: raw_joint_names.len(),
            positions: raw_joint_positions_rad.len(),
        });
    }
    if !previous_base_height_command_m.is_finite() {
        return Err(MappingError::NonFiniteBaseHeight);
    }

    let expected: HashSet<&str> = RAW_JOINT_NAMES.into_iter().collect();
    let mut positions_by_name = HashMap::with_capacity(RAW_JOINT_DIMENSION);

    for (name, value) in raw_joint_names.iter().zip(raw_joint_positions_rad.iter()) {
        if !expected.contains(name) {
            return Err(MappingError::UnknownJointName((*name).to_string()));
        }
        if !value.is_finite() {
            return Err(MappingError::NonFiniteRawPosition((*name).to_string()));
        }
        if positions_by_name.insert(*name, *value).is_some() {
            return Err(MappingError::DuplicateJointName((*name).to_string()));
        }
    }

    for expected_name in RAW_JOINT_NAMES {
        if !positions_by_name.contains_key(expected_name) {
            return Err(MappingError::MissingJointName(expected_name.to_string()));
        }
    }

    let mut state = [0.0_f32; POLICY_STATE_DIMENSION];
    for (index, name) in POLICY_STATE_JOINT_NAMES.iter().enumerate() {
        state[index] = *positions_by_name
            .get(name)
            .ok_or_else(|| MappingError::MissingJointName((*name).to_string()))?;
    }
    state[31] = previous_base_height_command_m;
    Ok(state)
}

/// Navigation command interpretation in the public `decoupled_v1` schema.
///
/// The historical upstream field called `vyaw` is a turn flag, not yaw rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationGoal {
    pub vx_m_s: f32,
    pub vy_m_s: f32,
    pub turn_flag: f32,
    pub target_yaw_rad: f32,
}

impl NavigationGoal {
    /// Matches the upstream controller's threshold: |turn_flag| >= 0.1.
    pub fn requests_turn(self) -> bool {
        self.turn_flag.abs() >= 0.1
    }
}

/// Decoded whole-body position/navigation goal.
///
/// Joint positions are radians; base height is metres; navigation velocities
/// are metres/second; target yaw is an absolute heading in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoupledGoal {
    /// Natural named order: thumb(3), index(2), middle(2).
    pub left_hand_natural_rad: [f32; 7],
    /// Natural named order: thumb(3), index(2), middle(2).
    pub right_hand_natural_rad: [f32; 7],
    pub left_arm_position_rad: [f32; 7],
    pub right_arm_position_rad: [f32; 7],
    /// Roll, pitch, yaw.
    pub torso_rpy_rad: [f32; 3],
    pub base_height_command_m: f32,
    pub navigation: NavigationGoal,
}

fn copy_seven(values: &[f32]) -> [f32; 7] {
    let mut output = [0.0; 7];
    output.copy_from_slice(values);
    output
}

fn copy_three(values: &[f32]) -> [f32; 3] {
    let mut output = [0.0; 3];
    output.copy_from_slice(values);
    output
}

/// Decode one `decoupled_v1` action row into named goal fields.
///
/// This function performs semantic decoding only. It does not validate physical
/// joint limits, plan a trajectory, apply safety projection, publish transport,
/// or claim that a physical robot applied/measured the goal.
pub fn decode_action(values: &[f32]) -> Result<DecoupledGoal, MappingError> {
    if values.len() != ACTION_DIMENSION {
        return Err(MappingError::ActionDimension(values.len()));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(MappingError::NonFiniteAction(index));
        }
    }

    // The public flat left hand is thumb/middle/index; named G1 order is
    // thumb/index/middle. Apply this involutive permutation exactly once.
    let left_hand_natural_rad = [
        values[0], values[1], values[2], values[5], values[6], values[3], values[4],
    ];

    Ok(DecoupledGoal {
        left_hand_natural_rad,
        right_hand_natural_rad: copy_seven(&values[7..14]),
        left_arm_position_rad: copy_seven(&values[14..21]),
        right_arm_position_rad: copy_seven(&values[21..28]),
        torso_rpy_rad: copy_three(&values[28..31]),
        base_height_command_m: values[31],
        navigation: NavigationGoal {
            vx_m_s: values[32],
            vy_m_s: values[33],
            turn_flag: values[34],
            target_yaw_rad: values[35],
        },
    })
}

/// Re-encode a decoded goal into the canonical 36-value flat schema.
///
/// This exists to prove round-trip representation only; it is not an actuator API.
pub fn encode_action(goal: &DecoupledGoal) -> [f32; ACTION_DIMENSION] {
    let mut values = [0.0_f32; ACTION_DIMENSION];

    values[0..3].copy_from_slice(&goal.left_hand_natural_rad[0..3]);
    values[3..5].copy_from_slice(&goal.left_hand_natural_rad[5..7]);
    values[5..7].copy_from_slice(&goal.left_hand_natural_rad[3..5]);
    values[7..14].copy_from_slice(&goal.right_hand_natural_rad);
    values[14..21].copy_from_slice(&goal.left_arm_position_rad);
    values[21..28].copy_from_slice(&goal.right_arm_position_rad);
    values[28..31].copy_from_slice(&goal.torso_rpy_rad);
    values[31] = goal.base_height_command_m;
    values[32] = goal.navigation.vx_m_s;
    values[33] = goal.navigation.vy_m_s;
    values[34] = goal.navigation.turn_flag;
    values[35] = goal.navigation.target_yaw_rad;

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentinel_joint_positions() -> Vec<f32> {
        (0..RAW_JOINT_DIMENSION).map(|index| index as f32).collect()
    }

    fn project_in_pinned_order() -> [f32; POLICY_STATE_DIMENSION] {
        project_policy_state(&RAW_JOINT_NAMES, &sentinel_joint_positions(), 0.74).unwrap()
    }

    #[test]
    fn named_raw_positions_project_to_exact_32_value_golden_state() {
        let expected = [
            29.0, 30.0, 31.0, 34.0, 35.0, 32.0, 33.0,
            36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
            13.0, 14.0, 12.0, 0.74,
        ];
        assert_eq!(project_in_pinned_order(), expected);
    }

    #[test]
    fn raw_joint_array_order_does_not_change_name_bound_projection() {
        let mut names = RAW_JOINT_NAMES.to_vec();
        let mut positions = sentinel_joint_positions();
        names.reverse();
        positions.reverse();
        let projected = project_policy_state(&names, &positions, 0.74).unwrap();
        assert_eq!(projected, project_in_pinned_order());
    }

    #[test]
    fn incomplete_duplicate_and_unknown_joint_profiles_fail_closed() {
        let mut names = RAW_JOINT_NAMES.to_vec();
        let positions = sentinel_joint_positions();

        assert!(matches!(
            project_policy_state(&names[..42], &positions[..42], 0.74),
            Err(MappingError::RawJointCount { .. })
        ));

        names[42] = names[41];
        assert!(matches!(
            project_policy_state(&names, &positions, 0.74),
            Err(MappingError::DuplicateJointName(_))
        ));

        names = RAW_JOINT_NAMES.to_vec();
        names[42] = "unrecognized_joint";
        assert!(matches!(
            project_policy_state(&names, &positions, 0.74),
            Err(MappingError::UnknownJointName(_))
        ));
    }

    #[test]
    fn non_finite_raw_observation_or_command_history_is_rejected() {
        let names = RAW_JOINT_NAMES;
        let mut positions = sentinel_joint_positions();
        positions[0] = f32::NAN;
        assert!(matches!(
            project_policy_state(&names, &positions, 0.74),
            Err(MappingError::NonFiniteRawPosition(_))
        ));

        positions = sentinel_joint_positions();
        assert!(matches!(
            project_policy_state(&names, &positions, f32::INFINITY),
            Err(MappingError::NonFiniteBaseHeight)
        ));
    }

    #[test]
    fn decoupled_action_decodes_left_hand_permutation_and_named_fields() {
        let flat: Vec<f32> = (100..136).map(|value| value as f32).collect();
        let goal = decode_action(&flat).unwrap();

        assert_eq!(goal.left_hand_natural_rad, [100.0, 101.0, 102.0, 105.0, 106.0, 103.0, 104.0]);
        assert_eq!(goal.right_hand_natural_rad, [107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0]);
        assert_eq!(goal.left_arm_position_rad, [114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]);
        assert_eq!(goal.right_arm_position_rad, [121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0]);
        assert_eq!(goal.torso_rpy_rad, [128.0, 129.0, 130.0]);
        assert_eq!(goal.base_height_command_m, 131.0);
        assert_eq!(goal.navigation.vx_m_s, 132.0);
        assert_eq!(goal.navigation.vy_m_s, 133.0);
        assert_eq!(goal.navigation.turn_flag, 134.0);
        assert_eq!(goal.navigation.target_yaw_rad, 135.0);
    }

    #[test]
    fn action_decode_encode_round_trip_preserves_every_flat_value() {
        let flat: Vec<f32> = (0..ACTION_DIMENSION)
            .map(|value| value as f32 * 0.125 - 1.0)
            .collect();
        let goal = decode_action(&flat).unwrap();
        assert_eq!(encode_action(&goal), flat.as_slice());
    }

    #[test]
    fn invalid_action_lengths_and_non_finite_values_are_rejected() {
        assert!(matches!(decode_action(&[0.0; 35]), Err(MappingError::ActionDimension(35))));
        assert!(matches!(decode_action(&[0.0; 37]), Err(MappingError::ActionDimension(37))));

        let mut values = [0.0; ACTION_DIMENSION];
        values[34] = f32::NAN;
        assert!(matches!(decode_action(&values), Err(MappingError::NonFiniteAction(34))));
    }

    #[test]
    fn navigation_field_34_is_a_turn_flag_not_a_yaw_rate() {
        let flat: Vec<f32> = (0..ACTION_DIMENSION).map(|value| value as f32).collect();
        let goal = decode_action(&flat).unwrap();
        assert!(goal.navigation.requests_turn());
        assert_eq!(goal.navigation.target_yaw_rad, 35.0);

        let mut no_turn = [0.0; ACTION_DIMENSION];
        no_turn[34] = 0.099;
        assert!(!decode_action(&no_turn).unwrap().navigation.requests_turn());

        no_turn[34] = -0.1;
        assert!(decode_action(&no_turn).unwrap().navigation.requests_turn());
    }
}
