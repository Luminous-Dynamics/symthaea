// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal semantic view of `HumanoidState`.
//!
//! This module is deliberately parallel to the optimized `HumanoidHdcEncoder`.
//! It gives physical state a portable role-bound vocabulary without changing the
//! policy encoder, controller, safety envelope, or actuation path.
//!
//! The frame preserves the existing humanoid observation boundary:
//! `root_position` remains privileged world truth and is **not** mixed into the
//! policy-facing semantic HDC bundle. Timestamp and morphology identity also stay
//! explicit metadata rather than being hidden inside a hypervector.

use serde::{Deserialize, Serialize};
use std::fmt;

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::sensorimotor_contingencies::{
    MissingObservationReasonV1, SensorimotorAddressV1, SensorimotorComponentV1,
    SensorimotorFrameV1, SensorimotorHdcEncoderV1, SensorimotorMeasurementV1,
    SensorimotorObservationV1, SensorimotorQuantityV1, SensorimotorSubjectV1,
    SensorimotorUnitV1, SensorimotorValueContractV1,
};

use crate::morphology::HumanoidMorphology;
use crate::types::HumanoidState;

const ROOT_HEIGHT_MIN_M: f64 = 0.0;
const ROOT_HEIGHT_MAX_M: f64 = 2.0;
const ROOT_HEIGHT_BINS: u16 = 201;

const ORIENTATION_MIN: f64 = -1.0;
const ORIENTATION_MAX: f64 = 1.0;
const ORIENTATION_BINS: u16 = 129;

const ROOT_LINEAR_VELOCITY_MIN_MPS: f64 = -10.0;
const ROOT_LINEAR_VELOCITY_MAX_MPS: f64 = 10.0;
const ROOT_LINEAR_VELOCITY_BINS: u16 = 201;

// Deliberately identical to R3.2 SemanticImuFusionV1 so the exact semantic
// digest for body-root angular velocity aligns across the two adapters.
const ROOT_ANGULAR_VELOCITY_MIN_RADPS: f64 = -20.0;
const ROOT_ANGULAR_VELOCITY_MAX_RADPS: f64 = 20.0;
const ROOT_ANGULAR_VELOCITY_BINS: u16 = 401;

const JOINT_POSITION_BINS: u16 = 129;
const JOINT_VELOCITY_MIN_RADPS: f64 = -20.0;
const JOINT_VELOCITY_MAX_RADPS: f64 = 20.0;
const JOINT_VELOCITY_BINS: u16 = 201;

const HEAD_HEIGHT_MIN_M: f64 = 0.0;
const HEAD_HEIGHT_MAX_M: f64 = 2.0;
const HEAD_HEIGHT_BINS: u16 = 201;

const DIRECTION_MIN: f64 = -1.0;
const DIRECTION_MAX: f64 = 1.0;
const DIRECTION_BINS: u16 = 129;

const EXTREMITY_POSITION_BINS: u16 = 129;
const COM_VELOCITY_MIN_MPS: f64 = -10.0;
const COM_VELOCITY_MAX_MPS: f64 = 10.0;
const COM_VELOCITY_BINS: u16 = 201;

/// Structural/schema failure while constructing or validating a semantic frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticHumanoidErrorV1 {
    JointAngleCount { expected: usize, actual: usize },
    JointVelocityCount { expected: usize, actual: usize },
    ExtremityChannelCount { expected: usize, actual: usize },
    InvalidTimestamp,
    FrameSchemaMismatch,
    SourceSchemaMismatch,
    SensorimotorSchema(&'static str),
}

impl fmt::Display for SemanticHumanoidErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JointAngleCount { expected, actual } => {
                write!(f, "joint angle count {actual}, expected {expected}")
            }
            Self::JointVelocityCount { expected, actual } => {
                write!(f, "joint velocity count {actual}, expected {expected}")
            }
            Self::ExtremityChannelCount { expected, actual } => {
                write!(f, "extremity channel count {actual}, expected {expected}")
            }
            Self::InvalidTimestamp => write!(f, "timestamp must be finite and non-negative"),
            Self::FrameSchemaMismatch => write!(f, "unsupported semantic humanoid frame schema"),
            Self::SourceSchemaMismatch => write!(f, "humanoid source observation schema mismatch"),
            Self::SensorimotorSchema(message) => write!(f, "sensorimotor schema: {message}"),
        }
    }
}

impl std::error::Error for SemanticHumanoidErrorV1 {}

/// Self-describing semantic view of one humanoid observation frame.
///
/// `policy_observations` contains exactly the fields exposed by
/// `HumanoidState::to_channels()`, but addressed by physical meaning rather than
/// channel number. `privileged_root_position_world_m` stays outside that set to
/// preserve the existing privileged-truth boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticHumanoidFrameV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    pub source_observation_schema_id: String,
    pub timestamp_seconds: f64,
    pub policy_observations: Vec<SensorimotorObservationV1>,
    /// Privileged root position in world coordinates. `None` means that scalar
    /// was non-finite/invalid. These values are not included in policy HDC.
    pub privileged_root_position_world_m: [Option<f64>; 3],
}

impl SemanticHumanoidFrameV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.semantic-frame.v1";

    pub fn validate(&self) -> Result<(), SemanticHumanoidErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(SemanticHumanoidErrorV1::FrameSchemaMismatch);
        }
        if self.source_observation_schema_id != self.morphology.schema_id() {
            return Err(SemanticHumanoidErrorV1::SourceSchemaMismatch);
        }
        if !self.timestamp_seconds.is_finite() || self.timestamp_seconds < 0.0 {
            return Err(SemanticHumanoidErrorV1::InvalidTimestamp);
        }
        let expected = self.morphology.num_observation_channels();
        if self.policy_observations.len() != expected {
            return Err(SemanticHumanoidErrorV1::ExtremityChannelCount {
                expected,
                actual: self.policy_observations.len(),
            });
        }
        for observation in &self.policy_observations {
            observation
                .validate()
                .map_err(SemanticHumanoidErrorV1::SensorimotorSchema)?;
        }
        Ok(())
    }

    pub fn measured_count(&self) -> usize {
        self.policy_observations
            .iter()
            .filter(|observation| matches!(observation, SensorimotorObservationV1::Measured(_)))
            .count()
    }

    pub fn missing_count(&self) -> usize {
        self.policy_observations.len() - self.measured_count()
    }

    /// Shared body-motion subset useful for cross-embodiment alignment studies.
    /// This is six facts: root linear velocity XYZ and root angular velocity XYZ.
    pub fn shared_body_motion_observations(&self) -> Vec<SensorimotorObservationV1> {
        self.policy_observations
            .iter()
            .filter(|observation| {
                let address = observation.address();
                matches!(&address.subject, SensorimotorSubjectV1::BodyRoot)
                    && matches!(
                        &address.quantity,
                        SensorimotorQuantityV1::LinearVelocity
                            | SensorimotorQuantityV1::AngularVelocity
                    )
            })
            .cloned()
            .collect()
    }
}

/// Non-breaking adapter from structured humanoid state into universal semantics.
///
/// This adapter is measurement/research infrastructure. Nothing in this module
/// grants motor authority or changes the live humanoid controller.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemanticHumanoidEncoderV1 {
    encoder: SensorimotorHdcEncoderV1,
}

impl SemanticHumanoidEncoderV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frame(
        &self,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
    ) -> Result<SemanticHumanoidFrameV1, SemanticHumanoidErrorV1> {
        validate_structure(state, morphology)?;

        let mut observations = Vec::with_capacity(morphology.num_observation_channels());

        observations.push(observation(
            address(
                SensorimotorSubjectV1::BodyRoot,
                SensorimotorQuantityV1::Height,
                SensorimotorFrameV1::World,
                SensorimotorComponentV1::Scalar,
                SensorimotorUnitV1::Meter,
                ROOT_HEIGHT_MIN_M,
                ROOT_HEIGHT_MAX_M,
                ROOT_HEIGHT_BINS,
            ),
            state.root_height,
        ));

        for (component, value) in [
            (SensorimotorComponentV1::W, state.root_quaternion[0]),
            (SensorimotorComponentV1::X, state.root_quaternion[1]),
            (SensorimotorComponentV1::Y, state.root_quaternion[2]),
            (SensorimotorComponentV1::Z, state.root_quaternion[3]),
        ] {
            observations.push(observation(
                address(
                    SensorimotorSubjectV1::BodyRoot,
                    SensorimotorQuantityV1::Orientation,
                    SensorimotorFrameV1::World,
                    component,
                    SensorimotorUnitV1::Unitless,
                    ORIENTATION_MIN,
                    ORIENTATION_MAX,
                    ORIENTATION_BINS,
                ),
                value,
            ));
        }

        let joint_names = morphology.joint_names();
        let joint_limits = morphology.joint_limits();
        for ((name, limits), value) in joint_names
            .iter()
            .zip(joint_limits.iter())
            .zip(state.joint_angles.iter())
        {
            observations.push(observation(
                address(
                    SensorimotorSubjectV1::Joint(name.clone()),
                    SensorimotorQuantityV1::JointPosition,
                    SensorimotorFrameV1::ParentJoint,
                    SensorimotorComponentV1::Scalar,
                    SensorimotorUnitV1::Radian,
                    limits[0],
                    limits[1],
                    JOINT_POSITION_BINS,
                ),
                *value,
            ));
        }

        push_vector3(
            &mut observations,
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::LinearVelocity,
            SensorimotorFrameV1::World,
            SensorimotorUnitV1::MeterPerSecond,
            ROOT_LINEAR_VELOCITY_MIN_MPS,
            ROOT_LINEAR_VELOCITY_MAX_MPS,
            ROOT_LINEAR_VELOCITY_BINS,
            state.root_linear_velocity,
        );

        push_vector3(
            &mut observations,
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::AngularVelocity,
            SensorimotorFrameV1::Body,
            SensorimotorUnitV1::RadianPerSecond,
            ROOT_ANGULAR_VELOCITY_MIN_RADPS,
            ROOT_ANGULAR_VELOCITY_MAX_RADPS,
            ROOT_ANGULAR_VELOCITY_BINS,
            state.root_angular_velocity,
        );

        for (name, value) in joint_names.iter().zip(state.joint_velocities.iter()) {
            observations.push(observation(
                address(
                    SensorimotorSubjectV1::Joint(name.clone()),
                    SensorimotorQuantityV1::JointVelocity,
                    SensorimotorFrameV1::ParentJoint,
                    SensorimotorComponentV1::Scalar,
                    SensorimotorUnitV1::RadianPerSecond,
                    JOINT_VELOCITY_MIN_RADPS,
                    JOINT_VELOCITY_MAX_RADPS,
                    JOINT_VELOCITY_BINS,
                ),
                *value,
            ));
        }

        observations.push(observation(
            address(
                SensorimotorSubjectV1::Custom("head".into()),
                SensorimotorQuantityV1::Height,
                SensorimotorFrameV1::World,
                SensorimotorComponentV1::Scalar,
                SensorimotorUnitV1::Meter,
                HEAD_HEIGHT_MIN_M,
                HEAD_HEIGHT_MAX_M,
                HEAD_HEIGHT_BINS,
            ),
            state.head_height,
        ));

        push_vector3(
            &mut observations,
            SensorimotorSubjectV1::Custom("torso".into()),
            SensorimotorQuantityV1::Custom("vertical_direction".into()),
            SensorimotorFrameV1::World,
            SensorimotorUnitV1::Unitless,
            DIRECTION_MIN,
            DIRECTION_MAX,
            DIRECTION_BINS,
            state.torso_vertical,
        );

        for (name, values) in extremity_triplets(&state.extremities) {
            for (index, value) in values.into_iter().enumerate() {
                let component = xyz_component(index);
                let (min, max) = extremity_range(name, component);
                observations.push(observation(
                    address(
                        SensorimotorSubjectV1::EndEffector(name.to_string()),
                        SensorimotorQuantityV1::Position,
                        SensorimotorFrameV1::World,
                        component,
                        SensorimotorUnitV1::Meter,
                        min,
                        max,
                        EXTREMITY_POSITION_BINS,
                    ),
                    value,
                ));
            }
        }

        push_vector3(
            &mut observations,
            SensorimotorSubjectV1::Custom("center_of_mass".into()),
            SensorimotorQuantityV1::LinearVelocity,
            SensorimotorFrameV1::World,
            SensorimotorUnitV1::MeterPerSecond,
            COM_VELOCITY_MIN_MPS,
            COM_VELOCITY_MAX_MPS,
            COM_VELOCITY_BINS,
            state.com_velocity,
        );

        let frame = SemanticHumanoidFrameV1 {
            schema_id: SemanticHumanoidFrameV1::SCHEMA_ID.to_string(),
            morphology,
            source_observation_schema_id: morphology.schema_id().to_string(),
            timestamp_seconds: state.timestamp,
            policy_observations: observations,
            privileged_root_position_world_m: state.root_position.map(finite_option),
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Encode all policy-facing physical observations with the R3.1 encoder.
    ///
    /// This is not wired into the live controller. R3.4 should benchmark this
    /// path and a prepared/cached codebook before any runtime migration decision.
    pub fn encode_policy(
        &self,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
    ) -> Result<Option<ContinuousHV>, SemanticHumanoidErrorV1> {
        let frame = self.frame(state, morphology)?;
        self.encoder
            .encode_observations(&frame.policy_observations)
            .map_err(SemanticHumanoidErrorV1::SensorimotorSchema)
    }

    /// Encode only the six root body-motion facts. This bounded subset is useful
    /// for R3.4 cross-embodiment alignment experiments against IMU/vehicle/flight.
    pub fn encode_shared_body_motion(
        &self,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
    ) -> Result<Option<ContinuousHV>, SemanticHumanoidErrorV1> {
        let frame = self.frame(state, morphology)?;
        let shared = frame.shared_body_motion_observations();
        self.encoder
            .encode_observations(&shared)
            .map_err(SemanticHumanoidErrorV1::SensorimotorSchema)
    }
}

fn validate_structure(
    state: &HumanoidState,
    morphology: HumanoidMorphology,
) -> Result<(), SemanticHumanoidErrorV1> {
    let expected_joints = morphology.num_actuators();
    if state.joint_angles.len() != expected_joints {
        return Err(SemanticHumanoidErrorV1::JointAngleCount {
            expected: expected_joints,
            actual: state.joint_angles.len(),
        });
    }
    if state.joint_velocities.len() != expected_joints {
        return Err(SemanticHumanoidErrorV1::JointVelocityCount {
            expected: expected_joints,
            actual: state.joint_velocities.len(),
        });
    }
    let expected_extremities = morphology.num_extremity_channels();
    if state.extremities.len() != expected_extremities {
        return Err(SemanticHumanoidErrorV1::ExtremityChannelCount {
            expected: expected_extremities,
            actual: state.extremities.len(),
        });
    }
    if !state.timestamp.is_finite() || state.timestamp < 0.0 {
        return Err(SemanticHumanoidErrorV1::InvalidTimestamp);
    }
    Ok(())
}

fn address(
    subject: SensorimotorSubjectV1,
    quantity: SensorimotorQuantityV1,
    frame: SensorimotorFrameV1,
    component: SensorimotorComponentV1,
    unit: SensorimotorUnitV1,
    min: f64,
    max: f64,
    bins: u16,
) -> SensorimotorAddressV1 {
    SensorimotorAddressV1::new(
        subject,
        quantity,
        frame,
        component,
        SensorimotorValueContractV1 {
            unit,
            min,
            max,
            bins,
        },
    )
}

fn observation(address: SensorimotorAddressV1, value: f64) -> SensorimotorObservationV1 {
    if value.is_finite() {
        SensorimotorObservationV1::Measured(SensorimotorMeasurementV1 { address, value })
    } else {
        SensorimotorObservationV1::Missing {
            address,
            reason: MissingObservationReasonV1::Invalid,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_vector3(
    observations: &mut Vec<SensorimotorObservationV1>,
    subject: SensorimotorSubjectV1,
    quantity: SensorimotorQuantityV1,
    frame: SensorimotorFrameV1,
    unit: SensorimotorUnitV1,
    min: f64,
    max: f64,
    bins: u16,
    values: [f64; 3],
) {
    for (index, value) in values.into_iter().enumerate() {
        observations.push(observation(
            address(
                subject.clone(),
                quantity.clone(),
                frame.clone(),
                xyz_component(index),
                unit.clone(),
                min,
                max,
                bins,
            ),
            value,
        ));
    }
}

fn xyz_component(index: usize) -> SensorimotorComponentV1 {
    match index {
        0 => SensorimotorComponentV1::X,
        1 => SensorimotorComponentV1::Y,
        2 => SensorimotorComponentV1::Z,
        _ => unreachable!("3-vector component index"),
    }
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn extremity_triplets(extremities: &[f64]) -> Vec<(&'static str, [f64; 3])> {
    let names: &[&str] = if extremities.len() == 12 {
        &["right_hand", "left_hand", "right_foot", "left_foot"]
    } else {
        &[
            "right_hand",
            "left_hand",
            "right_foot",
            "left_foot",
            "right_hand_centroid",
            "left_hand_centroid",
        ]
    };

    names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let start = index * 3;
            (
                *name,
                [
                    extremities[start],
                    extremities[start + 1],
                    extremities[start + 2],
                ],
            )
        })
        .collect()
}

fn extremity_range(name: &str, component: SensorimotorComponentV1) -> (f64, f64) {
    match component {
        SensorimotorComponentV1::Z if name.contains("foot") => (0.0, 0.5),
        SensorimotorComponentV1::Z => (0.0, 2.0),
        SensorimotorComponentV1::X | SensorimotorComponentV1::Y => (-2.0, 2.0),
        SensorimotorComponentV1::Scalar | SensorimotorComponentV1::W => {
            unreachable!("extremity positions are XYZ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_observation<'a>(
        frame: &'a SemanticHumanoidFrameV1,
        subject: &SensorimotorSubjectV1,
        quantity: &SensorimotorQuantityV1,
        component: SensorimotorComponentV1,
    ) -> &'a SensorimotorObservationV1 {
        frame
            .policy_observations
            .iter()
            .find(|observation| {
                let address = observation.address();
                &address.subject == subject
                    && &address.quantity == quantity
                    && address.component == component
            })
            .expect("semantic observation")
    }

    #[test]
    fn every_morphology_preserves_policy_observation_count() {
        let encoder = SemanticHumanoidEncoderV1::new();
        for morphology in [
            HumanoidMorphology::Dmc21,
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::FullSpine,
        ] {
            let state = HumanoidState::default_for(morphology);
            let frame = encoder.frame(&state, morphology).unwrap();
            assert_eq!(
                frame.policy_observations.len(),
                morphology.num_observation_channels()
            );
            assert_eq!(frame.missing_count(), 0);
            frame.validate().unwrap();
        }
    }

    #[test]
    fn root_angular_velocity_identity_matches_universal_imu_contract() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let frame = encoder.frame(&state, HumanoidMorphology::Dmc21).unwrap();
        let humanoid = find_observation(
            &frame,
            &SensorimotorSubjectV1::BodyRoot,
            &SensorimotorQuantityV1::AngularVelocity,
            SensorimotorComponentV1::X,
        )
        .address();

        let expected = SensorimotorAddressV1::new(
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::AngularVelocity,
            SensorimotorFrameV1::Body,
            SensorimotorComponentV1::X,
            SensorimotorValueContractV1 {
                unit: SensorimotorUnitV1::RadianPerSecond,
                min: -20.0,
                max: 20.0,
                bins: 401,
            },
        );

        assert_eq!(
            humanoid.semantic_digest().unwrap(),
            expected.semantic_digest().unwrap()
        );
    }

    #[test]
    fn shared_joint_identity_survives_morphology_extension() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let dmc = encoder
            .frame(
                &HumanoidState::default_for(HumanoidMorphology::Dmc21),
                HumanoidMorphology::Dmc21,
            )
            .unwrap();
        let extended = encoder
            .frame(
                &HumanoidState::default_for(HumanoidMorphology::WithNeckWrist),
                HumanoidMorphology::WithNeckWrist,
            )
            .unwrap();

        let subject = SensorimotorSubjectV1::Joint("right_knee".into());
        let quantity = SensorimotorQuantityV1::JointPosition;
        let a = find_observation(&dmc, &subject, &quantity, SensorimotorComponentV1::Scalar)
            .address()
            .semantic_digest()
            .unwrap();
        let b = find_observation(
            &extended,
            &subject,
            &quantity,
            SensorimotorComponentV1::Scalar,
        )
        .address()
        .semantic_digest()
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_scalar_becomes_missing_not_zero() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.root_angular_velocity[0] = f64::NAN;
        let frame = encoder.frame(&state, HumanoidMorphology::Dmc21).unwrap();
        let observation = find_observation(
            &frame,
            &SensorimotorSubjectV1::BodyRoot,
            &SensorimotorQuantityV1::AngularVelocity,
            SensorimotorComponentV1::X,
        );
        assert!(matches!(
            observation,
            SensorimotorObservationV1::Missing {
                reason: MissingObservationReasonV1::Invalid,
                ..
            }
        ));
    }

    #[test]
    fn privileged_root_position_is_not_policy_hdc() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.root_position = [12.0, -4.0, 1.1];
        let frame = encoder.frame(&state, HumanoidMorphology::Dmc21).unwrap();

        assert_eq!(
            frame.privileged_root_position_world_m,
            [Some(12.0), Some(-4.0), Some(1.1)]
        );
        assert!(!frame.policy_observations.iter().any(|observation| {
            let address = observation.address();
            address.subject == SensorimotorSubjectV1::BodyRoot
                && address.quantity == SensorimotorQuantityV1::Position
        }));
    }

    #[test]
    fn structural_mismatch_fails_closed() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.joint_angles.pop();
        assert!(matches!(
            encoder.frame(&state, HumanoidMorphology::Dmc21),
            Err(SemanticHumanoidErrorV1::JointAngleCount { .. })
        ));
    }

    #[test]
    fn shared_body_motion_subset_is_six_and_deterministic() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.root_linear_velocity = [0.2, -0.1, 0.0];
        state.root_angular_velocity = [0.01, -0.02, 0.03];
        let frame = encoder.frame(&state, HumanoidMorphology::Dmc21).unwrap();
        assert_eq!(frame.shared_body_motion_observations().len(), 6);

        let a = encoder
            .encode_shared_body_motion(&state, HumanoidMorphology::Dmc21)
            .unwrap()
            .unwrap();
        let b = encoder
            .encode_shared_body_motion(&state, HumanoidMorphology::Dmc21)
            .unwrap()
            .unwrap();
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn serde_round_trip_preserves_exact_role_identity() {
        let encoder = SemanticHumanoidEncoderV1::new();
        let frame = encoder
            .frame(
                &HumanoidState::default_for(HumanoidMorphology::Dmc21),
                HumanoidMorphology::Dmc21,
            )
            .unwrap();
        let before = frame.policy_observations[0]
            .address()
            .semantic_digest()
            .unwrap();
        let wire = serde_json::to_string(&frame).unwrap();
        let restored: SemanticHumanoidFrameV1 = serde_json::from_str(&wire).unwrap();
        restored.validate().unwrap();
        let after = restored.policy_observations[0]
            .address()
            .semantic_digest()
            .unwrap();
        assert_eq!(before, after);
    }
}
