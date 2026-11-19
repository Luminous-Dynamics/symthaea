// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded handoff from relative-motion rendezvous to servicing-arm
//! capture. This is a gate contract, not a contact solver or flight authority.

pub const CAPTURE_ARM_DOF: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproachOutcome {
    /// The existing rendezvous scenario's relative position/speed window.
    /// This is not physical capture.
    DockedRelativeMotionOnly,
    CorridorAborted,
    DeltaVExhausted,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessStatus {
    ReadyForContactEvaluation,
    Rejected,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    Captured,
    Rejected,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatchState {
    Engaged,
    NotEngaged,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionTest {
    Passed,
    Failed,
    NotPerformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureReasonCode {
    UnsupportedApproachOutcome,
    InterfaceIdentityMismatch,
    InterfaceRevisionMismatch,
    ReferenceFrameMismatch,
    MissingFrameTransformIdentity,
    MissingNavigationEvidence,
    MissingKinematicsEvidence,
    MissingCalibrationEvidence,
    InvalidInterfaceSpecification,
    NonFiniteMeasurement,
    InvalidMeasurementRange,
    EpochMismatch,
    RelativePositionOutOfTolerance,
    RelativeSpeedOutOfTolerance,
    RelativeAngularRateOutOfTolerance,
    InterfaceOrientationOutOfTolerance,
    EndEffectorPositionOutOfTolerance,
    ArmJointPositionOutOfLimits,
    ArmJointVelocityOutOfLimits,
    InsufficientClearance,
    ReadinessNotAccepted,
    AttemptIdentityMismatch,
    MissingContactEvidence,
    MissingLoadEvidence,
    MissingLatchEvidence,
    MissingRetentionEvidence,
    ContactNotDetected,
    ContactImpulseExceeded,
    ContactForceExceeded,
    LatchNotEngaged,
    LatchStateUnknown,
    RetentionTestFailed,
    RetentionTestNotPerformed,
    PostCaptureRelativeSpeedExceeded,
}

/// Mission/interface limits supplied by the caller. They are never inferred
/// from the simulator's nominal arm parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureInterfaceSpec {
    pub interface_id: String,
    pub interface_revision: String,
    pub reference_frame_id: String,
    pub reference_epoch_s: f64,
    pub maximum_epoch_delta_s: f64,
    pub position_tolerance_m: f64,
    pub relative_speed_limit_mps: f64,
    pub relative_angular_rate_limit_rad_s: f64,
    pub orientation_error_limit_rad: f64,
    pub end_effector_position_tolerance_m: f64,
    pub minimum_clearance_m: f64,
    pub contact_impulse_limit_ns: f64,
    pub peak_contact_force_limit_n: f64,
    pub post_capture_relative_speed_limit_mps: f64,
    pub arm_joint_position_limits_rad: [[f64; 2]; CAPTURE_ARM_DOF],
    pub arm_joint_velocity_limits_rad_s: [f64; CAPTURE_ARM_DOF],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureReadinessObservation {
    pub attempt_id: String,
    pub approach_outcome: ApproachOutcome,
    pub interface_id: String,
    pub interface_revision: String,
    pub reference_frame_id: String,
    /// Transform identity used to express the following positions in the
    /// declared reference frame.
    pub frame_transform_id: String,
    pub measurement_epoch_s: f64,
    /// Chaser reference point relative to the target port origin, metres.
    pub relative_position_m: [f64; 3],
    /// End-effector location relative to the target port origin, metres.
    pub end_effector_position_m: [f64; 3],
    pub relative_velocity_mps: [f64; 3],
    pub relative_angular_rate_rad_s: [f64; 3],
    pub interface_orientation_error_rad: f64,
    pub clearance_m: f64,
    pub arm_joint_angles_rad: [f64; CAPTURE_ARM_DOF],
    pub arm_joint_velocities_rad_s: [f64; CAPTURE_ARM_DOF],
    pub navigation_evidence_ref: String,
    pub kinematics_evidence_ref: String,
    pub calibration_evidence_ref: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureReadinessReceipt {
    pub attempt_id: String,
    pub interface_id: String,
    pub interface_revision: String,
    pub reference_frame_id: String,
    pub frame_transform_id: String,
    pub measurement_epoch_s: f64,
    pub relative_position_error_m: Option<f64>,
    pub end_effector_error_m: Option<f64>,
    pub relative_speed_mps: Option<f64>,
    pub relative_angular_rate_rad_s: Option<f64>,
    pub status: ReadinessStatus,
    pub reason_codes: Vec<CaptureReasonCode>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureConfirmationObservation {
    pub attempt_id: String,
    pub interface_id: String,
    pub interface_revision: String,
    pub contact_detected: bool,
    pub contact_impulse_ns: f64,
    pub peak_contact_force_n: f64,
    pub latch_state: LatchState,
    pub retention_test: RetentionTest,
    pub post_capture_relative_speed_mps: f64,
    pub contact_evidence_ref: String,
    pub load_evidence_ref: String,
    pub latch_evidence_ref: String,
    pub retention_evidence_ref: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureReceipt {
    pub attempt_id: String,
    pub interface_id: String,
    pub interface_revision: String,
    pub status: CaptureStatus,
    pub reason_codes: Vec<CaptureReasonCode>,
    pub contact_impulse_ns: Option<f64>,
    pub peak_contact_force_n: Option<f64>,
    pub post_capture_relative_speed_mps: Option<f64>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

fn norm3(value: [f64; 3]) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn difference3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn finite3(value: [f64; 3]) -> bool {
    value.iter().all(|component| component.is_finite())
}

fn valid_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn present(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_spec(spec: &CaptureInterfaceSpec) -> bool {
    present(&spec.interface_id)
        && present(&spec.interface_revision)
        && present(&spec.reference_frame_id)
        && spec.reference_epoch_s.is_finite()
        && valid_nonnegative(spec.maximum_epoch_delta_s)
        && valid_positive(spec.position_tolerance_m)
        && valid_positive(spec.relative_speed_limit_mps)
        && valid_positive(spec.relative_angular_rate_limit_rad_s)
        && valid_positive(spec.orientation_error_limit_rad)
        && valid_positive(spec.end_effector_position_tolerance_m)
        && valid_nonnegative(spec.minimum_clearance_m)
        && valid_positive(spec.contact_impulse_limit_ns)
        && valid_positive(spec.peak_contact_force_limit_n)
        && valid_positive(spec.post_capture_relative_speed_limit_mps)
        && spec
            .arm_joint_position_limits_rad
            .iter()
            .all(|limits| limits[0].is_finite() && limits[1].is_finite() && limits[0] < limits[1])
        && spec
            .arm_joint_velocity_limits_rad_s
            .iter()
            .all(|limit| valid_positive(*limit))
}

fn readiness_receipt(
    observation: &CaptureReadinessObservation,
    status: ReadinessStatus,
    reason_codes: Vec<CaptureReasonCode>,
    relative_position_error_m: Option<f64>,
    end_effector_error_m: Option<f64>,
    relative_speed_mps: Option<f64>,
    relative_angular_rate_rad_s: Option<f64>,
) -> CaptureReadinessReceipt {
    CaptureReadinessReceipt {
        attempt_id: observation.attempt_id.clone(),
        interface_id: observation.interface_id.clone(),
        interface_revision: observation.interface_revision.clone(),
        reference_frame_id: observation.reference_frame_id.clone(),
        frame_transform_id: observation.frame_transform_id.clone(),
        measurement_epoch_s: observation.measurement_epoch_s,
        relative_position_error_m,
        end_effector_error_m,
        relative_speed_mps,
        relative_angular_rate_rad_s,
        status,
        reason_codes,
        evidence_refs: [
            observation.navigation_evidence_ref.clone(),
            observation.kinematics_evidence_ref.clone(),
            observation.calibration_evidence_ref.clone(),
        ]
        .into_iter()
        .filter(|reference| present(reference))
        .collect(),
        claim_ceiling:
            "Readiness to evaluate contact only; no contact, capture, servicing, command authority, or flight qualification is established.",
    }
}

/// Evaluate the handoff from relative-motion rendezvous to contact.
///
/// The existing approach outcome is only one prerequisite. This function
/// independently checks interface identity, frame/epoch, evidence, relative
/// state, arm joint state, end-effector pose, orientation and clearance.
/// It does not command or drive actuators.
pub fn evaluate_readiness(
    spec: &CaptureInterfaceSpec,
    observation: &CaptureReadinessObservation,
) -> CaptureReadinessReceipt {
    if !valid_spec(spec) {
        return readiness_receipt(
            observation,
            ReadinessStatus::Indeterminate,
            vec![CaptureReasonCode::InvalidInterfaceSpecification],
            None,
            None,
            None,
            None,
        );
    }

    let mut reasons = Vec::new();
    if !present(&observation.attempt_id) {
        reasons.push(CaptureReasonCode::MissingNavigationEvidence);
    }
    if !present(&observation.navigation_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingNavigationEvidence);
    }
    if !present(&observation.kinematics_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingKinematicsEvidence);
    }
    if !present(&observation.calibration_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingCalibrationEvidence);
    }
    if !present(&observation.frame_transform_id) {
        reasons.push(CaptureReasonCode::MissingFrameTransformIdentity);
    }
    if observation.approach_outcome != ApproachOutcome::DockedRelativeMotionOnly {
        reasons.push(CaptureReasonCode::UnsupportedApproachOutcome);
    }
    if observation.interface_id != spec.interface_id {
        reasons.push(CaptureReasonCode::InterfaceIdentityMismatch);
    }
    if observation.interface_revision != spec.interface_revision {
        reasons.push(CaptureReasonCode::InterfaceRevisionMismatch);
    }
    if observation.reference_frame_id != spec.reference_frame_id {
        reasons.push(CaptureReasonCode::ReferenceFrameMismatch);
    }

    let finite = observation.measurement_epoch_s.is_finite()
        && finite3(observation.relative_position_m)
        && finite3(observation.end_effector_position_m)
        && finite3(observation.relative_velocity_mps)
        && finite3(observation.relative_angular_rate_rad_s)
        && observation.interface_orientation_error_rad.is_finite()
        && observation.clearance_m.is_finite()
        && observation.arm_joint_angles_rad.iter().all(|v| v.is_finite())
        && observation
            .arm_joint_velocities_rad_s
            .iter()
            .all(|v| v.is_finite());
    if !finite {
        reasons.push(CaptureReasonCode::NonFiniteMeasurement);
    }
    if observation.interface_orientation_error_rad < 0.0 || observation.clearance_m < 0.0 {
        reasons.push(CaptureReasonCode::InvalidMeasurementRange);
    }

    if finite
        && (observation.measurement_epoch_s - spec.reference_epoch_s).abs()
            > spec.maximum_epoch_delta_s
    {
        reasons.push(CaptureReasonCode::EpochMismatch);
    }

    let indeterminate_reasons = [
        CaptureReasonCode::MissingNavigationEvidence,
        CaptureReasonCode::MissingKinematicsEvidence,
        CaptureReasonCode::MissingCalibrationEvidence,
        CaptureReasonCode::MissingFrameTransformIdentity,
        CaptureReasonCode::NonFiniteMeasurement,
    ];
    if reasons
        .iter()
        .any(|reason| indeterminate_reasons.contains(reason))
    {
        return readiness_receipt(
            observation,
            ReadinessStatus::Indeterminate,
            reasons,
            None,
            None,
            None,
            None,
        );
    }
    if !finite {
        return readiness_receipt(
            observation,
            ReadinessStatus::Indeterminate,
            reasons,
            None,
            None,
            None,
            None,
        );
    }

    let relative_position_error_m = norm3(observation.relative_position_m);
    let end_effector_error_m = norm3(observation.end_effector_position_m);
    let relative_speed_mps = norm3(observation.relative_velocity_mps);
    let relative_angular_rate_rad_s = norm3(observation.relative_angular_rate_rad_s);

    if relative_position_error_m > spec.position_tolerance_m {
        reasons.push(CaptureReasonCode::RelativePositionOutOfTolerance);
    }
    if relative_speed_mps > spec.relative_speed_limit_mps {
        reasons.push(CaptureReasonCode::RelativeSpeedOutOfTolerance);
    }
    if relative_angular_rate_rad_s > spec.relative_angular_rate_limit_rad_s {
        reasons.push(CaptureReasonCode::RelativeAngularRateOutOfTolerance);
    }
    if observation.interface_orientation_error_rad > spec.orientation_error_limit_rad {
        reasons.push(CaptureReasonCode::InterfaceOrientationOutOfTolerance);
    }
    if end_effector_error_m > spec.end_effector_position_tolerance_m {
        reasons.push(CaptureReasonCode::EndEffectorPositionOutOfTolerance);
    }
    if observation.clearance_m < spec.minimum_clearance_m {
        reasons.push(CaptureReasonCode::InsufficientClearance);
    }

    for index in 0..CAPTURE_ARM_DOF {
        let angle = observation.arm_joint_angles_rad[index];
        let [minimum, maximum] = spec.arm_joint_position_limits_rad[index];
        if angle < minimum || angle > maximum {
            reasons.push(CaptureReasonCode::ArmJointPositionOutOfLimits);
            break;
        }
    }
    for index in 0..CAPTURE_ARM_DOF {
        if observation.arm_joint_velocities_rad_s[index].abs()
            > spec.arm_joint_velocity_limits_rad_s[index]
        {
            reasons.push(CaptureReasonCode::ArmJointVelocityOutOfLimits);
            break;
        }
    }

    let status = if reasons.is_empty() {
        ReadinessStatus::ReadyForContactEvaluation
    } else {
        ReadinessStatus::Rejected
    };
    readiness_receipt(
        observation,
        status,
        reasons,
        Some(relative_position_error_m),
        Some(end_effector_error_m),
        Some(relative_speed_mps),
        Some(relative_angular_rate_rad_s),
    )
}

/// Confirm capture only after contact and independent retention evidence.
///
/// A ready receipt is necessary but never sufficient. The caller must
/// supply matching attempt/interface identity and contact, load, latch and
/// retention evidence. This function does not issue a servicing command.
pub fn evaluate_capture(
    spec: &CaptureInterfaceSpec,
    readiness: &CaptureReadinessReceipt,
    observation: &CaptureConfirmationObservation,
) -> CaptureReceipt {
    let mut reasons = Vec::new();
    if readiness.status != ReadinessStatus::ReadyForContactEvaluation {
        reasons.push(CaptureReasonCode::ReadinessNotAccepted);
    }
    if readiness.attempt_id != observation.attempt_id
        || readiness.interface_id != observation.interface_id
    {
        reasons.push(CaptureReasonCode::AttemptIdentityMismatch);
    }
    if readiness.interface_id != spec.interface_id
        || observation.interface_id != spec.interface_id
    {
        reasons.push(CaptureReasonCode::InterfaceIdentityMismatch);
    }
    if readiness.interface_revision != spec.interface_revision
        || observation.interface_revision != spec.interface_revision
    {
        reasons.push(CaptureReasonCode::InterfaceRevisionMismatch);
    }

    if !present(&observation.contact_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingContactEvidence);
    }
    if !present(&observation.load_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingLoadEvidence);
    }
    if !present(&observation.latch_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingLatchEvidence);
    }
    if !present(&observation.retention_evidence_ref) {
        reasons.push(CaptureReasonCode::MissingRetentionEvidence);
    }

    let measurements_finite = observation.contact_impulse_ns.is_finite()
        && observation.peak_contact_force_n.is_finite()
        && observation.post_capture_relative_speed_mps.is_finite();
    if !measurements_finite {
        reasons.push(CaptureReasonCode::NonFiniteMeasurement);
    }
    let evidence_present = present(&observation.contact_evidence_ref)
        && present(&observation.load_evidence_ref)
        && present(&observation.latch_evidence_ref)
        && present(&observation.retention_evidence_ref);
    if !measurements_finite || !evidence_present {
        return CaptureReceipt {
            attempt_id: observation.attempt_id.clone(),
            interface_id: observation.interface_id.clone(),
            interface_revision: observation.interface_revision.clone(),
            status: CaptureStatus::Indeterminate,
            reason_codes: reasons,
            contact_impulse_ns: None,
            peak_contact_force_n: None,
            post_capture_relative_speed_mps: None,
            evidence_refs: vec![
                observation.contact_evidence_ref.clone(),
                observation.load_evidence_ref.clone(),
                observation.latch_evidence_ref.clone(),
                observation.retention_evidence_ref.clone(),
            ]
            .into_iter()
            .filter(|reference| present(reference))
            .collect(),
            claim_ceiling:
                "No capture claim; missing or non-finite evidence leaves the outcome indeterminate.",
        };
    }

    if !observation.contact_detected {
        reasons.push(CaptureReasonCode::ContactNotDetected);
    }
    if observation.contact_impulse_ns < 0.0
        || observation.contact_impulse_ns > spec.contact_impulse_limit_ns
    {
        reasons.push(CaptureReasonCode::ContactImpulseExceeded);
    }
    if observation.peak_contact_force_n < 0.0
        || observation.peak_contact_force_n > spec.peak_contact_force_limit_n
    {
        reasons.push(CaptureReasonCode::ContactForceExceeded);
    }
    if observation.post_capture_relative_speed_mps < 0.0
        || observation.post_capture_relative_speed_mps
            > spec.post_capture_relative_speed_limit_mps
    {
        reasons.push(CaptureReasonCode::PostCaptureRelativeSpeedExceeded);
    }
    match observation.latch_state {
        LatchState::Engaged => {}
        LatchState::NotEngaged => reasons.push(CaptureReasonCode::LatchNotEngaged),
        LatchState::Unknown => reasons.push(CaptureReasonCode::LatchStateUnknown),
    }
    match observation.retention_test {
        RetentionTest::Passed => {}
        RetentionTest::Failed => reasons.push(CaptureReasonCode::RetentionTestFailed),
        RetentionTest::NotPerformed => {
            reasons.push(CaptureReasonCode::RetentionTestNotPerformed)
        }
    }

    let hard_rejection = reasons.iter().any(|reason| {
        matches!(
            reason,
            CaptureReasonCode::ReadinessNotAccepted
                | CaptureReasonCode::AttemptIdentityMismatch
                | CaptureReasonCode::InterfaceIdentityMismatch
                | CaptureReasonCode::InterfaceRevisionMismatch
                | CaptureReasonCode::ContactNotDetected
                | CaptureReasonCode::ContactImpulseExceeded
                | CaptureReasonCode::ContactForceExceeded
                | CaptureReasonCode::LatchNotEngaged
                | CaptureReasonCode::RetentionTestFailed
                | CaptureReasonCode::PostCaptureRelativeSpeedExceeded
        )
    });
    let status = if reasons.is_empty() {
        CaptureStatus::Captured
    } else if hard_rejection {
        CaptureStatus::Rejected
    } else {
        CaptureStatus::Indeterminate
    };

    CaptureReceipt {
        attempt_id: observation.attempt_id.clone(),
        interface_id: observation.interface_id.clone(),
        interface_revision: observation.interface_revision.clone(),
        status,
        reason_codes: reasons,
        contact_impulse_ns: Some(observation.contact_impulse_ns),
        peak_contact_force_n: Some(observation.peak_contact_force_n),
        post_capture_relative_speed_mps: Some(observation.post_capture_relative_speed_mps),
        evidence_refs: vec![
            observation.contact_evidence_ref.clone(),
            observation.load_evidence_ref.clone(),
            observation.latch_evidence_ref.clone(),
            observation.retention_evidence_ref.clone(),
        ],
        claim_ceiling:
            "Named interface capture evidence passed this declared contract only; no servicing, fluid/power/data mate, flight qualification, or mission authority is implied.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CaptureInterfaceSpec {
        CaptureInterfaceSpec {
            interface_id: "cooperative-grapple-v1".to_owned(),
            interface_revision: "rev-1".to_owned(),
            reference_frame_id: "target-port-frame".to_owned(),
            reference_epoch_s: 1_700_000_000.0,
            maximum_epoch_delta_s: 0.01,
            position_tolerance_m: 0.05,
            relative_speed_limit_mps: 0.02,
            relative_angular_rate_limit_rad_s: 0.005,
            orientation_error_limit_rad: 0.01,
            end_effector_position_tolerance_m: 0.01,
            minimum_clearance_m: 0.02,
            contact_impulse_limit_ns: 0.5,
            peak_contact_force_limit_n: 10.0,
            post_capture_relative_speed_limit_mps: 0.005,
            arm_joint_position_limits_rad: [[-2.9, 2.9]; CAPTURE_ARM_DOF],
            arm_joint_velocity_limits_rad_s: [0.5; CAPTURE_ARM_DOF],
        }
    }

    fn readiness_observation() -> CaptureReadinessObservation {
        CaptureReadinessObservation {
            attempt_id: "attempt-001".to_owned(),
            approach_outcome: ApproachOutcome::DockedRelativeMotionOnly,
            interface_id: "cooperative-grapple-v1".to_owned(),
            interface_revision: "rev-1".to_owned(),
            reference_frame_id: "target-port-frame".to_owned(),
            frame_transform_id: "transform-001".to_owned(),
            measurement_epoch_s: 1_700_000_000.005,
            relative_position_m: [0.01, 0.0, 0.0],
            end_effector_position_m: [0.005, 0.0, 0.0],
            relative_velocity_mps: [0.01, 0.0, 0.0],
            relative_angular_rate_rad_s: [0.001, 0.0, 0.0],
            interface_orientation_error_rad: 0.005,
            clearance_m: 0.03,
            arm_joint_angles_rad: [0.0; CAPTURE_ARM_DOF],
            arm_joint_velocities_rad_s: [0.0; CAPTURE_ARM_DOF],
            navigation_evidence_ref: "nav-observation-001".to_owned(),
            kinematics_evidence_ref: "arm-kinematics-001".to_owned(),
            calibration_evidence_ref: "calibration-001".to_owned(),
        }
    }

    fn confirmation_observation() -> CaptureConfirmationObservation {
        CaptureConfirmationObservation {
            attempt_id: "attempt-001".to_owned(),
            interface_id: "cooperative-grapple-v1".to_owned(),
            interface_revision: "rev-1".to_owned(),
            contact_detected: true,
            contact_impulse_ns: 0.2,
            peak_contact_force_n: 4.0,
            latch_state: LatchState::Engaged,
            retention_test: RetentionTest::Passed,
            post_capture_relative_speed_mps: 0.001,
            contact_evidence_ref: "contact-001".to_owned(),
            load_evidence_ref: "load-cell-001".to_owned(),
            latch_evidence_ref: "latch-sensor-001".to_owned(),
            retention_evidence_ref: "retention-test-001".to_owned(),
        }
    }

    #[test]
    fn approach_only_docked_enters_readiness_gate_but_is_not_capture() {
        let result = evaluate_readiness(&spec(), &readiness_observation());
        assert_eq!(result.status, ReadinessStatus::ReadyForContactEvaluation);
        assert!(result.reason_codes.is_empty());
        assert!(result.claim_ceiling.contains("no contact, capture"));
    }

    #[test]
    fn wrong_interface_revision_rejects_readiness() {
        let mut observation = readiness_observation();
        observation.interface_revision = "rev-0".to_owned();
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::InterfaceRevisionMismatch));
    }

    #[test]
    fn wrong_frame_rejects_even_when_relative_state_is_in_tolerance() {
        let mut observation = readiness_observation();
        observation.reference_frame_id = "inertial-frame".to_owned();
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::ReferenceFrameMismatch));
    }

    #[test]
    fn stale_epoch_rejects_readiness() {
        let mut observation = readiness_observation();
        observation.measurement_epoch_s += 1.0;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::EpochMismatch));
    }

    #[test]
    fn relative_position_and_speed_do_not_override_excess_angular_rate() {
        let mut observation = readiness_observation();
        observation.relative_angular_rate_rad_s = [0.006, 0.0, 0.0];
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::RelativeAngularRateOutOfTolerance));
    }

    #[test]
    fn good_relative_motion_does_not_override_end_effector_misalignment() {
        let mut observation = readiness_observation();
        observation.end_effector_position_m = [0.03, 0.0, 0.0];
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::EndEffectorPositionOutOfTolerance));
    }

    #[test]
    fn good_relative_motion_does_not_override_orientation_error() {
        let mut observation = readiness_observation();
        observation.interface_orientation_error_rad = 0.02;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::InterfaceOrientationOutOfTolerance));
    }

    #[test]
    fn arm_joint_limit_violation_rejects_readiness() {
        let mut observation = readiness_observation();
        observation.arm_joint_angles_rad[3] = 3.0;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::ArmJointPositionOutOfLimits));
    }

    #[test]
    fn arm_joint_velocity_violation_rejects_readiness() {
        let mut observation = readiness_observation();
        observation.arm_joint_velocities_rad_s[2] = 0.6;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::ArmJointVelocityOutOfLimits));
    }

    #[test]
    fn insufficient_clearance_rejects_readiness() {
        let mut observation = readiness_observation();
        observation.clearance_m = 0.0;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::InsufficientClearance));
    }

    #[test]
    fn negative_orientation_or_clearance_is_rejected_as_invalid_range() {
        let mut observation = readiness_observation();
        observation.interface_orientation_error_rad = -0.001;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::InvalidMeasurementRange));
    }

    #[test]
    fn missing_transform_or_nonfinite_measurement_is_indeterminate() {
        let mut observation = readiness_observation();
        observation.frame_transform_id.clear();
        observation.measurement_epoch_s = f64::NAN;
        let result = evaluate_readiness(&spec(), &observation);
        assert_eq!(result.status, ReadinessStatus::Indeterminate);
        assert!(result.reason_codes.contains(&CaptureReasonCode::MissingFrameTransformIdentity));
        assert!(result.reason_codes.contains(&CaptureReasonCode::NonFiniteMeasurement));
    }

    #[test]
    fn capture_confirmation_requires_matching_attempt_identity() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let mut confirmation = confirmation_observation();
        confirmation.attempt_id = "attempt-002".to_owned();
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::AttemptIdentityMismatch));
    }

    #[test]
    fn capture_requires_contact_latch_and_retention_evidence() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let result = evaluate_capture(&spec, &readiness, &confirmation_observation());
        assert_eq!(result.status, CaptureStatus::Captured);
        assert!(result.reason_codes.is_empty());
        assert!(result.claim_ceiling.contains("no servicing"));
    }

    #[test]
    fn readiness_alone_never_confirms_capture() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let mut confirmation = confirmation_observation();
        confirmation.retention_test = RetentionTest::NotPerformed;
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Indeterminate);
        assert!(result.reason_codes.contains(&CaptureReasonCode::RetentionTestNotPerformed));
    }

    #[test]
    fn rejected_readiness_cannot_be_overridden_by_unknown_latch() {
        let spec = spec();
        let mut approach = readiness_observation();
        approach.approach_outcome = ApproachOutcome::TimedOut;
        let readiness = evaluate_readiness(&spec, &approach);
        let mut confirmation = confirmation_observation();
        confirmation.latch_state = LatchState::Unknown;
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::ReadinessNotAccepted));
    }

    #[test]
    fn over_limit_contact_load_rejects_capture() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let mut confirmation = confirmation_observation();
        confirmation.peak_contact_force_n = 11.0;
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::ContactForceExceeded));
    }

    #[test]
    fn no_latch_engagement_rejects_capture() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let mut confirmation = confirmation_observation();
        confirmation.latch_state = LatchState::NotEngaged;
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Rejected);
        assert!(result.reason_codes.contains(&CaptureReasonCode::LatchNotEngaged));
    }

    #[test]
    fn missing_retention_reference_is_indeterminate() {
        let spec = spec();
        let readiness = evaluate_readiness(&spec, &readiness_observation());
        let mut confirmation = confirmation_observation();
        confirmation.retention_evidence_ref.clear();
        let result = evaluate_capture(&spec, &readiness, &confirmation);
        assert_eq!(result.status, CaptureStatus::Indeterminate);
        assert!(result.reason_codes.contains(&CaptureReasonCode::MissingRetentionEvidence));
    }
}
