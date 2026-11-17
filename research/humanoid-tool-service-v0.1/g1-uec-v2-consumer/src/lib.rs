//! Read-only HumanoidToolBench G1 consumer for the UEC v2 contract.
//!
//! This crate projects pinned G1 joint observations into typed UEC v2 frames
//! and records decoupled_v1 policy outputs as Requested action frames.
//! It has no transport, controller, safety admission, HAL, PWM, torque,
//! simulator stepping, or actuator authority.

use symthaea_core::embodiment_v2::{
    ActionFrameV2, ActionStageV2, ActionValueV2, ClockStampV2, EmbodimentDescriptorV2,
    EmbodimentV2Adapter, EMBODIMENT_CONTRACT_VERSION_V2, ObservationFrameV2,
    ObservationValueV2,
};

pub const HUMANOID_TOOL_BENCH_COMMIT: &str =
    "3fec9bd36933a1648ff7f95468c983f3452243a0";
pub const PROFILE_ID: &str =
    "symthaea.external.humanoidtoolbench.g1_sonic.decoupled_v1.v1";
pub const MORPHOLOGY_ID: &str = "unitree.g1_sonic.dex3";
pub const OBSERVATION_SCHEMA_ID: &str = "humanoidtoolbench.g1_sonic.policy-input.v1";
pub const ACTION_SCHEMA_ID: &str = "humanoidtoolbench.g1_sonic.decoupled_v1.action.v1";
pub const OBSERVATION_SOURCE_ID: &str = "humanoidtoolbench.sonic_loco_manip._get_obs";
pub const POLICY_ACTION_SOURCE_ID: &str = "symthaea.g1.decoupled_v1.policy-output";
pub const RAW_JOINT_DIMENSION: usize = 43;
pub const ACTION_DIMENSION: usize = 36;

/// Exact upstream G1Sonic raw joint_qpos order at HUMANOID_TOOL_BENCH_COMMIT.
///
/// The producer chain was source-traced:
/// SonicLocoManip._get_obs -> MujocoEngine.get_robot_qpos ->
/// G1Sonic.get_robot_qpos, whose ordered joints dictionary is populated from
/// G1Sonic.WHOLE_BODY_JOINTS. Do not use this order for another upstream SHA.
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

#[derive(Debug, Clone, Copy)]
struct ActionBinding {
    address: &'static str,
    quantity: &'static str,
    unit: &'static str,
}

/// decoupled_v1 flat action semantics. The left hand intentionally follows
/// benchmark flat thumb/middle/index order, not anatomical index order.
const ACTION_BINDINGS: [ActionBinding; ACTION_DIMENSION] = [
    ActionBinding { address: "g1/left_hand_thumb_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_thumb_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_thumb_2_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_middle_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_middle_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_index_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_hand_index_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_thumb_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_thumb_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_thumb_2_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_index_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_index_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_middle_0_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_hand_middle_1_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_shoulder_pitch_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_shoulder_roll_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_shoulder_yaw_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_elbow_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_wrist_roll_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_wrist_pitch_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/left_wrist_yaw_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_shoulder_pitch_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_shoulder_roll_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_shoulder_yaw_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_elbow_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_wrist_roll_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_wrist_pitch_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/right_wrist_yaw_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/waist_roll_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/waist_pitch_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/waist_yaw_joint/position_target", quantity: "joint_position_target", unit: "rad" },
    ActionBinding { address: "g1/base/height_command", quantity: "base_height_command", unit: "m" },
    ActionBinding { address: "g1/base/navigation/vx", quantity: "linear_velocity_command", unit: "m/s" },
    ActionBinding { address: "g1/base/navigation/vy", quantity: "linear_velocity_command", unit: "m/s" },
    ActionBinding { address: "g1/base/navigation/turn_flag", quantity: "turn_flag", unit: "1" },
    ActionBinding { address: "g1/base/navigation/target_yaw", quantity: "absolute_heading_target", unit: "rad" },
];

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_joint_positions(values: &[f32]) -> Result<(), String> {
    if values.len() != RAW_JOINT_DIMENSION {
        return Err(format!(
            "expected exactly {RAW_JOINT_DIMENSION} source-ordered G1 qpos values, received {}",
            values.len()
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("joint qpos at source index {index} is not finite"));
        }
    }
    Ok(())
}

fn observation_frame(
    profile_id: &str,
    instance_id: &str,
    clock: ClockStampV2,
    source_id: &str,
    raw_joint_positions_rad: &[f32],
) -> Result<ObservationFrameV2, String> {
    non_empty(profile_id, "profile_id")?;
    non_empty(instance_id, "instance_id")?;
    non_empty(source_id, "source_id")?;
    clock.validate().map_err(|error| format!("clock: {error}"))?;
    validate_joint_positions(raw_joint_positions_rad)?;

    let observations = RAW_JOINT_NAMES
        .iter()
        .zip(raw_joint_positions_rad.iter())
        .map(|(joint_name, value)| ObservationValueV2 {
            address: format!("g1/{joint_name}/position"),
            quantity: "joint_position".into(),
            unit: "rad".into(),
            value: Some(*value as f64),
            missing_reason: None,
            uncertainty: None,
        })
        .collect();

    let frame = ObservationFrameV2 {
        contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
        profile_id: profile_id.into(),
        instance_id: instance_id.into(),
        clock,
        frame_graph_id: None,
        source_id: source_id.into(),
        observations,
    };
    frame.validate().map_err(|error| format!("observation frame: {error}"))?;
    Ok(frame)
}

fn descriptor(instance_id: &str, clock_domain_id: &str) -> Result<EmbodimentDescriptorV2, String> {
    non_empty(instance_id, "instance_id")?;
    non_empty(clock_domain_id, "clock_domain_id")?;
    let result = EmbodimentDescriptorV2 {
        contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
        profile_id: PROFILE_ID.into(),
        platform_family_id: "unitree.g1".into(),
        instance_id: instance_id.into(),
        morphology_id: MORPHOLOGY_ID.into(),
        observation_schema_id: OBSERVATION_SCHEMA_ID.into(),
        action_schema_id: ACTION_SCHEMA_ID.into(),
        frame_graph_id: None,
        clock_domain_id: clock_domain_id.into(),
        actuator_capability_ids: Vec::new(),
        sensor_capability_ids: vec!["g1/whole_body_joint_position_43".into()],
        safe_fallback_profile_id: "research.read_only.no_actuation.v1".into(),
        calibration_profile_id: None,
        backend_provenance_id: format!("SNU-PI/HumanoidToolBench@{HUMANOID_TOOL_BENCH_COMMIT}"),
        descriptor_digest: None,
    };
    result.validate().map_err(|error| format!("descriptor: {error}"))?;
    Ok(result)
}

/// Read-only source-pinned G1 observation/action adapter for UEC v2.
///
/// This represents a benchmark observation source and a policy's Requested
/// output. It is not a robot driver and has no transition to later action stages.
#[derive(Debug, Clone)]
pub struct G1ToolBenchUecV2Adapter {
    descriptor: EmbodimentDescriptorV2,
    observation: ObservationFrameV2,
    last_requested_action: Option<ActionFrameV2>,
}

impl G1ToolBenchUecV2Adapter {
    /// Construct from the raw 43-value qpos array and source-provided clock.
    ///
    /// ClockStampV2 is passed through unchanged. Missing time coordinates
    /// remain missing; this adapter does not create synthetic time coordinates.
    pub fn new(
        instance_id: &str,
        clock: ClockStampV2,
        source_id: &str,
        raw_joint_positions_rad: &[f32],
    ) -> Result<Self, String> {
        let descriptor = descriptor(instance_id, &clock.clock_domain_id)?;
        let observation = observation_frame(
            &descriptor.profile_id,
            instance_id,
            clock,
            source_id,
            raw_joint_positions_rad,
        )?;
        Ok(Self {
            descriptor,
            observation,
            last_requested_action: None,
        })
    }

    /// Replace the current observation with a newer source frame.
    ///
    /// When adjacent source sequence/timestamp values are both present they
    /// must be strictly increasing. Missing values are not synthesized.
    pub fn update_joint_qpos(
        &mut self,
        clock: ClockStampV2,
        source_id: &str,
        raw_joint_positions_rad: &[f32],
    ) -> Result<(), String> {
        if clock.clock_domain_id != self.descriptor.clock_domain_id {
            return Err("observation clock domain differs from the bound descriptor".into());
        }
        clock.validate().map_err(|error| format!("clock: {error}"))?;
        let previous = &self.observation.clock;
        if let (Some(old), Some(new)) = (previous.sequence, clock.sequence) {
            if new <= old {
                return Err("source sequence must increase within this adapter instance".into());
            }
        }
        if let (Some(old), Some(new)) = (previous.timestamp_ns, clock.timestamp_ns) {
            if new <= old {
                return Err("source timestamp must increase within this adapter instance".into());
            }
        }
        self.observation = observation_frame(
            &self.descriptor.profile_id,
            &self.descriptor.instance_id,
            clock,
            source_id,
            raw_joint_positions_rad,
        )?;
        Ok(())
    }

    /// Record a 36-value policy output as a UEC Requested frame.
    ///
    /// This only records what the policy proposed. It does not project limits,
    /// publish transport, step a simulator, or claim Applied/Measured evidence.
    /// The historical index-34 vyaw field is a dimensionless turn flag, not a
    /// yaw rate.
    pub fn record_requested_action(
        &mut self,
        clock: ClockStampV2,
        source_id: &str,
        flat_action: &[f32],
    ) -> Result<ActionFrameV2, String> {
        non_empty(source_id, "action source_id")?;
        clock.validate().map_err(|error| format!("clock: {error}"))?;
        if clock.clock_domain_id != self.descriptor.clock_domain_id {
            return Err("action clock domain differs from the bound descriptor".into());
        }
        if let (Some(observed), Some(requested)) =
            (self.observation.clock.timestamp_ns, clock.timestamp_ns)
        {
            if requested < observed {
                return Err("action request timestamp predates its source observation".into());
            }
        }
        if flat_action.len() != ACTION_DIMENSION {
            return Err(format!(
                "expected exactly {ACTION_DIMENSION} decoupled_v1 action values, received {}",
                flat_action.len()
            ));
        }
        for (index, value) in flat_action.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("action value at index {index} is not finite"));
            }
        }

        let actions = ACTION_BINDINGS
            .iter()
            .zip(flat_action.iter())
            .map(|(binding, value)| ActionValueV2 {
                address: binding.address.into(),
                quantity: binding.quantity.into(),
                unit: binding.unit.into(),
                value: *value as f64,
            })
            .collect();
        let frame = ActionFrameV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: self.descriptor.profile_id.clone(),
            instance_id: self.descriptor.instance_id.clone(),
            clock,
            frame_graph_id: None,
            source_id: source_id.into(),
            stage: ActionStageV2::Requested,
            actions,
        };
        frame.validate().map_err(|error| format!("requested action frame: {error}"))?;
        self.last_requested_action = Some(frame.clone());
        Ok(frame)
    }

    pub fn observation_snapshot(&self) -> &ObservationFrameV2 {
        &self.observation
    }

    pub fn descriptor_snapshot(&self) -> &EmbodimentDescriptorV2 {
        &self.descriptor
    }
}

impl EmbodimentV2Adapter for G1ToolBenchUecV2Adapter {
    fn descriptor_v2(&self, _instance_id: &str) -> EmbodimentDescriptorV2 {
        self.descriptor.clone()
    }

    fn observation_frame_v2(&self, instance_id: &str) -> Result<ObservationFrameV2, String> {
        if instance_id != self.descriptor.instance_id {
            return Err("requested instance does not match the bound G1 profile instance".into());
        }
        Ok(self.observation.clone())
    }

    fn last_action_frame_v2(&self, instance_id: &str) -> Result<Option<ActionFrameV2>, String> {
        if instance_id != self.descriptor.instance_id {
            return Err("requested instance does not match the bound G1 profile instance".into());
        }
        Ok(self.last_requested_action.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment_v2::validate_adapter_v2_conformance;

    fn clock(sequence: Option<u64>, timestamp_ns: Option<u64>) -> ClockStampV2 {
        ClockStampV2 {
            clock_domain_id: "humanoidtoolbench.g1_sonic.sim-clock.epoch-0".into(),
            sequence,
            timestamp_ns,
        }
    }

    fn qpos() -> Vec<f32> {
        (0..RAW_JOINT_DIMENSION).map(|index| index as f32).collect()
    }

    fn adapter() -> G1ToolBenchUecV2Adapter {
        G1ToolBenchUecV2Adapter::new(
            "g1-sim-instance-01",
            clock(Some(1), Some(1_000_000)),
            OBSERVATION_SOURCE_ID,
            &qpos(),
        )
        .unwrap()
    }

    #[test]
    fn observation_frame_preserves_all_43_named_source_positions() {
        let adapter = adapter();
        let frame = adapter.observation_snapshot();
        assert_eq!(frame.observations.len(), 43);
        assert_eq!(frame.observations[0].address, "g1/left_hip_pitch_joint/position");
        assert_eq!(frame.observations[12].address, "g1/waist_yaw_joint/position");
        assert_eq!(frame.observations[13].address, "g1/waist_roll_joint/position");
        assert_eq!(frame.observations[29].address, "g1/left_hand_thumb_0_joint/position");
        assert_eq!(frame.observations[32].address, "g1/left_hand_index_0_joint/position");
        assert_eq!(frame.observations[34].address, "g1/left_hand_middle_0_joint/position");
        assert_eq!(frame.observations[42].value, Some(42.0));
        assert!(frame.validate().is_ok());
    }

    #[test]
    fn malformed_or_non_finite_joint_observations_fail_closed() {
        let short = vec![0.0; 42];
        let long = vec![0.0; 44];
        assert!(G1ToolBenchUecV2Adapter::new(
            "g1-1", clock(Some(1), Some(1)), OBSERVATION_SOURCE_ID, &short
        ).is_err());
        assert!(G1ToolBenchUecV2Adapter::new(
            "g1-1", clock(Some(1), Some(1)), OBSERVATION_SOURCE_ID, &long
        ).is_err());
        let mut invalid = qpos();
        invalid[21] = f32::NAN;
        assert!(G1ToolBenchUecV2Adapter::new(
            "g1-1", clock(Some(1), Some(1)), OBSERVATION_SOURCE_ID, &invalid
        ).is_err());
    }

    #[test]
    fn observation_updates_require_bound_clock_and_increasing_coordinates() {
        let mut adapter = adapter();
        let mut wrong_domain = clock(Some(2), Some(2_000_000));
        wrong_domain.clock_domain_id = "different-clock".into();
        assert!(adapter.update_joint_qpos(wrong_domain, OBSERVATION_SOURCE_ID, &qpos()).is_err());
        assert!(adapter.update_joint_qpos(
            clock(Some(1), Some(2_000_000)), OBSERVATION_SOURCE_ID, &qpos()
        ).is_err());
        assert!(adapter.update_joint_qpos(
            clock(Some(2), Some(1_000_000)), OBSERVATION_SOURCE_ID, &qpos()
        ).is_err());
        assert!(adapter.update_joint_qpos(
            clock(Some(2), Some(2_000_000)), OBSERVATION_SOURCE_ID, &qpos()
        ).is_ok());
    }

    #[test]
    fn action_frame_preserves_36_named_requested_policy_values() {
        let mut adapter = adapter();
        let flat: Vec<f32> = (0..ACTION_DIMENSION).map(|index| index as f32).collect();
        let frame = adapter.record_requested_action(
            clock(Some(2), Some(1_500_000)),
            POLICY_ACTION_SOURCE_ID,
            &flat,
        ).unwrap();
        assert_eq!(frame.stage, ActionStageV2::Requested);
        assert_eq!(frame.actions.len(), 36);
        assert_eq!(frame.actions[3].address, "g1/left_hand_middle_0_joint/position_target");
        assert_eq!(frame.actions[5].address, "g1/left_hand_index_0_joint/position_target");
        assert_eq!(frame.actions[28].address, "g1/waist_roll_joint/position_target");
        assert_eq!(frame.actions[31].quantity, "base_height_command");
        assert_eq!(frame.actions[31].unit, "m");
        assert_eq!(frame.actions[32].unit, "m/s");
        assert_eq!(frame.actions[34].quantity, "turn_flag");
        assert_eq!(frame.actions[34].unit, "1");
        assert_eq!(frame.actions[35].quantity, "absolute_heading_target");
        assert_eq!(frame.actions[35].unit, "rad");
        assert!(frame.validate().is_ok());
    }

    #[test]
    fn invalid_policy_actions_never_create_a_request_frame() {
        let mut adapter = adapter();
        assert!(adapter.record_requested_action(
            clock(Some(2), Some(1_500_000)),
            POLICY_ACTION_SOURCE_ID,
            &[0.0; 35],
        ).is_err());
        let mut invalid = [0.0; ACTION_DIMENSION];
        invalid[34] = f32::INFINITY;
        assert!(adapter.record_requested_action(
            clock(Some(2), Some(1_500_000)),
            POLICY_ACTION_SOURCE_ID,
            &invalid,
        ).is_err());
        assert!(adapter.last_action_frame_v2("g1-sim-instance-01").unwrap().is_none());
    }

    #[test]
    fn adapter_conformance_is_read_only_and_instance_bound() {
        let adapter = adapter();
        assert!(validate_adapter_v2_conformance(
            &adapter,
            "g1-sim-instance-01",
        ).is_ok());
        assert!(adapter.observation_frame_v2("other-instance").is_err());
    }

    #[test]
    fn action_stage_cannot_be_upgraded_by_this_adapter() {
        let mut adapter = adapter();
        let frame = adapter.record_requested_action(
            clock(Some(2), Some(1_500_000)),
            POLICY_ACTION_SOURCE_ID,
            &[0.25; ACTION_DIMENSION],
        ).unwrap();
        assert_eq!(frame.stage, ActionStageV2::Requested);
        assert_eq!(adapter.last_action_frame_v2("g1-sim-instance-01").unwrap().unwrap().stage, ActionStageV2::Requested);
    }
}
