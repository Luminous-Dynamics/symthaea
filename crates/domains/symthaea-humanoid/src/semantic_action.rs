// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement/research semantics for humanoid actuation.
//!
//! This module keeps causal action stages explicit without putting authority,
//! safety approval, or execution claims inside HDC physical-role identity.
//! It reuses the R3 sensorimotor address vocabulary for *what physical actuator
//! quantity is being commanded*, but deliberately does not wrap commands in
//! `SensorimotorObservationV1`: requested/applied actions are not observations.
//!
//! Stronger causal claims are evidence-bound. A deterministic requested physical
//! projection can exist without an external evidence identifier; safety-projected,
//! backend-applied, and independently measured execution stages require one.
//! The identifier binds the claim to an external record but does not itself prove
//! that record trustworthy or safety-qualified.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::sensorimotor_contingencies::{
    SensorimotorAddressV1, SensorimotorComponentV1, SensorimotorFrameV1,
    SensorimotorQuantityV1, SensorimotorSubjectV1, SensorimotorUnitV1,
    SensorimotorValueContractV1,
};

use crate::actuation::{ActuationAdaptation, ActuationAdaptationError, ActuationAdapter};
use crate::morphology::HumanoidMorphology;
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

const TORQUE_BINS: u16 = 401;
const POSITION_TARGET_BINS: u16 = 129;

/// Causal status of one physical actuation frame.
///
/// Stage is intentionally *not* part of the physical-role address: requested
/// and applied torque on the same actuator remain the same physical role while
/// being different causal events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidActuationStageV1 {
    /// SI-unit projection of policy/controller intent before downstream gates.
    RequestedPhysicalProjection,
    /// Physical command after a safety/projection layer has changed or accepted it.
    SafetyProjectedPhysical,
    /// Physical command a backend reports as applied to its plant/physics input.
    BackendAppliedPhysical,
    /// Independently measured actuator execution/feedback.
    MeasuredExecution,
}

impl HumanoidActuationStageV1 {
    /// Requested projection is a deterministic derivation. Every stronger stage
    /// makes an external boundary claim and therefore requires evidence binding.
    pub const fn requires_evidence_id(self) -> bool {
        !matches!(self, Self::RequestedPhysicalProjection)
    }
}

/// One physical action value at a stable R3 sensorimotor address.
///
/// This type is separate from `SensorimotorMeasurementV1` because a command is
/// not automatically an observation or sensor measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticHumanoidActuationValueV1 {
    pub address: SensorimotorAddressV1,
    pub value: f64,
}

impl SemanticHumanoidActuationValueV1 {
    pub fn validate(&self) -> Result<(), SemanticHumanoidActuationErrorV1> {
        self.address
            .validate()
            .map_err(SemanticHumanoidActuationErrorV1::Sensorimotor)?;
        if !self.value.is_finite() {
            return Err(SemanticHumanoidActuationErrorV1::NonFiniteValue);
        }
        Ok(())
    }
}

/// Versioned physical actuation frame for one humanoid command stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticHumanoidActuationFrameV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    pub stage: HumanoidActuationStageV1,
    /// External record identifying the stronger causal boundary claim, when
    /// required by `stage`. This is evidence provenance, not motor authority.
    pub evidence_id: Option<String>,
    /// Mode of the command before the physical projection represented here.
    pub source_mode: ActuationMode,
    /// Physical unit/mode represented by `values`.
    pub physical_mode: ActuationMode,
    /// Number of input joints clipped while producing this physical projection.
    pub clipped_joints: usize,
    pub values: Vec<SemanticHumanoidActuationValueV1>,
}

impl SemanticHumanoidActuationFrameV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.semantic-actuation-frame.v1";

    /// Convenience constructor for the ordinary controller-intent case. It
    /// converts normalized torque intent to SI torque through the repository's
    /// existing `ActuationAdapter` and makes only a requested-projection claim.
    pub fn project_requested_normalized_torque_command(
        command: &HumanoidCommand,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
    ) -> Result<Self, SemanticHumanoidActuationErrorV1> {
        Self::project_normalized_torque_command(
            command,
            state,
            morphology,
            HumanoidActuationStageV1::RequestedPhysicalProjection,
            None,
        )
    }

    /// Convert a normalized torque command into an SI torque frame through the
    /// existing `ActuationAdapter` rather than duplicating scale/clipping logic.
    /// Strong stages require a non-empty evidence ID.
    pub fn project_normalized_torque_command(
        command: &HumanoidCommand,
        state: &HumanoidState,
        morphology: HumanoidMorphology,
        stage: HumanoidActuationStageV1,
        evidence_id: Option<String>,
    ) -> Result<Self, SemanticHumanoidActuationErrorV1> {
        let adaptation = ActuationAdapter::default().adapt_normalized_torque_intent(
            command,
            state,
            morphology,
            ActuationMode::TorqueNewtonMetres,
        )?;
        Self::from_adaptation(&adaptation, morphology, stage, evidence_id)
    }

    /// Build a physical frame from an existing actuation adaptation.
    ///
    /// Normalized target modes are rejected: a unitless command must not be
    /// relabeled as torque or position merely to fit the portable schema.
    pub fn from_adaptation(
        adaptation: &ActuationAdaptation,
        morphology: HumanoidMorphology,
        stage: HumanoidActuationStageV1,
        evidence_id: Option<String>,
    ) -> Result<Self, SemanticHumanoidActuationErrorV1> {
        Self::from_physical_command_with_metadata(
            &adaptation.command,
            morphology,
            stage,
            evidence_id,
            adaptation.source_mode,
            adaptation.target_mode,
            adaptation.clipped_joints,
        )
    }

    /// Build a frame from a command that is already expressed in physical units.
    pub fn from_physical_command(
        command: &HumanoidCommand,
        morphology: HumanoidMorphology,
        stage: HumanoidActuationStageV1,
        physical_mode: ActuationMode,
        evidence_id: Option<String>,
    ) -> Result<Self, SemanticHumanoidActuationErrorV1> {
        Self::from_physical_command_with_metadata(
            command,
            morphology,
            stage,
            evidence_id,
            physical_mode,
            physical_mode,
            0,
        )
    }

    fn from_physical_command_with_metadata(
        command: &HumanoidCommand,
        morphology: HumanoidMorphology,
        stage: HumanoidActuationStageV1,
        evidence_id: Option<String>,
        source_mode: ActuationMode,
        physical_mode: ActuationMode,
        clipped_joints: usize,
    ) -> Result<Self, SemanticHumanoidActuationErrorV1> {
        validate_stage_evidence(stage, evidence_id.as_deref())?;
        if !matches!(
            physical_mode,
            ActuationMode::TorqueNewtonMetres | ActuationMode::PositionTargetRadians
        ) {
            return Err(SemanticHumanoidActuationErrorV1::NonPhysicalMode(
                physical_mode,
            ));
        }

        let expected = morphology.num_actuators();
        if command.num_actuators() != expected {
            return Err(SemanticHumanoidActuationErrorV1::ActuatorCount {
                expected,
                actual: command.num_actuators(),
            });
        }

        let names = morphology.joint_names();
        let torque_scales = morphology.joint_torque_scales();
        let joint_limits = morphology.joint_limits();
        let mut values = Vec::with_capacity(expected);

        for index in 0..expected {
            let raw = f64::from(command.torques[index]);
            if !raw.is_finite() {
                return Err(SemanticHumanoidActuationErrorV1::NonFiniteAt { index });
            }

            let address = physical_actuation_address(
                &names[index],
                physical_mode,
                torque_scales[index],
                joint_limits[index],
            )?;
            values.push(SemanticHumanoidActuationValueV1 {
                address,
                value: raw,
            });
        }

        let frame = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            morphology,
            stage,
            evidence_id,
            source_mode,
            physical_mode,
            clipped_joints,
            values,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<(), SemanticHumanoidActuationErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(SemanticHumanoidActuationErrorV1::SchemaMismatch);
        }
        validate_stage_evidence(self.stage, self.evidence_id.as_deref())?;
        if !matches!(
            self.physical_mode,
            ActuationMode::TorqueNewtonMetres | ActuationMode::PositionTargetRadians
        ) {
            return Err(SemanticHumanoidActuationErrorV1::NonPhysicalMode(
                self.physical_mode,
            ));
        }

        let expected = self.morphology.num_actuators();
        if self.values.len() != expected {
            return Err(SemanticHumanoidActuationErrorV1::ActuatorCount {
                expected,
                actual: self.values.len(),
            });
        }
        if self.clipped_joints > expected {
            return Err(SemanticHumanoidActuationErrorV1::InvalidClippedCount {
                clipped: self.clipped_joints,
                actuators: expected,
            });
        }

        let names = self.morphology.joint_names();
        let torque_scales = self.morphology.joint_torque_scales();
        let joint_limits = self.morphology.joint_limits();
        for (index, value) in self.values.iter().enumerate() {
            value.validate()?;
            let expected_address = physical_actuation_address(
                &names[index],
                self.physical_mode,
                torque_scales[index],
                joint_limits[index],
            )?;
            if value.address != expected_address {
                return Err(SemanticHumanoidActuationErrorV1::AddressMismatch { index });
            }
        }
        Ok(())
    }

    /// Number of physical command values outside their declared actuator range.
    /// The raw value is retained; this method makes envelope violations explicit
    /// rather than silently replacing them with a saturated value.
    pub fn out_of_contract_count(&self) -> usize {
        self.values
            .iter()
            .filter(|value| {
                value.value < value.address.value_contract.min
                    || value.value > value.address.value_contract.max
            })
            .count()
    }
}

fn validate_stage_evidence(
    stage: HumanoidActuationStageV1,
    evidence_id: Option<&str>,
) -> Result<(), SemanticHumanoidActuationErrorV1> {
    if let Some(value) = evidence_id {
        if value.trim().is_empty() {
            return Err(SemanticHumanoidActuationErrorV1::EmptyEvidenceId);
        }
    }
    if stage.requires_evidence_id() && evidence_id.is_none() {
        return Err(SemanticHumanoidActuationErrorV1::MissingStageEvidence(stage));
    }
    Ok(())
}

/// Stage-to-stage action evidence for a common exact physical command contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidActuationStageComparisonV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    pub source_stage: HumanoidActuationStageV1,
    pub target_stage: HumanoidActuationStageV1,
    pub source_evidence_id: Option<String>,
    pub target_evidence_id: Option<String>,
    pub physical_mode: ActuationMode,
    pub compared_actuators: usize,
    pub physical_role_matches: usize,
    pub exact_contract_matches: usize,
    pub exact_value_matches: usize,
    pub changed_actuators: usize,
    pub mean_abs_delta: f64,
    pub rms_delta: f64,
    pub max_abs_delta: f64,
    pub source_clipped_joints: usize,
    pub target_clipped_joints: usize,
}

impl HumanoidActuationStageComparisonV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.actuation-stage-comparison.v1";

    pub fn any_action_changed(&self) -> bool {
        self.changed_actuators > 0
    }
}

/// Compare two causal action stages only when they use the same physical command
/// mode and morphology. Physical-role/contract counts are reported explicitly;
/// a mismatch fails closed rather than pairing by vector position alone.
pub fn compare_humanoid_actuation_stages_v1(
    source: &SemanticHumanoidActuationFrameV1,
    target: &SemanticHumanoidActuationFrameV1,
) -> Result<HumanoidActuationStageComparisonV1, SemanticHumanoidActuationErrorV1> {
    source.validate()?;
    target.validate()?;
    if source.morphology != target.morphology {
        return Err(SemanticHumanoidActuationErrorV1::MorphologyMismatch);
    }
    if source.physical_mode != target.physical_mode {
        return Err(SemanticHumanoidActuationErrorV1::PhysicalModeMismatch);
    }

    let mut physical_role_matches = 0usize;
    let mut exact_contract_matches = 0usize;
    let mut exact_value_matches = 0usize;
    let mut changed_actuators = 0usize;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;

    for (index, (left, right)) in source.values.iter().zip(target.values.iter()).enumerate() {
        if left
            .address
            .same_physical_role(&right.address)
            .map_err(SemanticHumanoidActuationErrorV1::Sensorimotor)?
        {
            physical_role_matches += 1;
        } else {
            return Err(SemanticHumanoidActuationErrorV1::PhysicalRoleMismatch { index });
        }

        if left
            .address
            .contract_digest()
            .map_err(SemanticHumanoidActuationErrorV1::Sensorimotor)?
            == right
                .address
                .contract_digest()
                .map_err(SemanticHumanoidActuationErrorV1::Sensorimotor)?
        {
            exact_contract_matches += 1;
        } else {
            return Err(SemanticHumanoidActuationErrorV1::ContractMismatch { index });
        }

        if left.value == right.value {
            exact_value_matches += 1;
        } else {
            changed_actuators += 1;
        }
        let delta = right.value - left.value;
        let abs = delta.abs();
        sum_abs += abs;
        sum_sq += delta * delta;
        max_abs = max_abs.max(abs);
    }

    let n = source.values.len() as f64;
    Ok(HumanoidActuationStageComparisonV1 {
        schema_id: HumanoidActuationStageComparisonV1::SCHEMA_ID.to_string(),
        morphology: source.morphology,
        source_stage: source.stage,
        target_stage: target.stage,
        source_evidence_id: source.evidence_id.clone(),
        target_evidence_id: target.evidence_id.clone(),
        physical_mode: source.physical_mode,
        compared_actuators: source.values.len(),
        physical_role_matches,
        exact_contract_matches,
        exact_value_matches,
        changed_actuators,
        mean_abs_delta: if n > 0.0 { sum_abs / n } else { 0.0 },
        rms_delta: if n > 0.0 { (sum_sq / n).sqrt() } else { 0.0 },
        max_abs_delta: max_abs,
        source_clipped_joints: source.clipped_joints,
        target_clipped_joints: target.clipped_joints,
    })
}

fn physical_actuation_address(
    actuator_name: &str,
    mode: ActuationMode,
    torque_scale_nm: f64,
    joint_limits_rad: [f64; 2],
) -> Result<SensorimotorAddressV1, SemanticHumanoidActuationErrorV1> {
    let (quantity, unit, min, max, bins) = match mode {
        ActuationMode::TorqueNewtonMetres => (
            SensorimotorQuantityV1::Torque,
            SensorimotorUnitV1::NewtonMeter,
            -torque_scale_nm,
            torque_scale_nm,
            TORQUE_BINS,
        ),
        ActuationMode::PositionTargetRadians => (
            SensorimotorQuantityV1::JointPosition,
            SensorimotorUnitV1::Radian,
            joint_limits_rad[0],
            joint_limits_rad[1],
            POSITION_TARGET_BINS,
        ),
        other => return Err(SemanticHumanoidActuationErrorV1::NonPhysicalMode(other)),
    };

    let address = SensorimotorAddressV1::new(
        SensorimotorSubjectV1::Actuator(actuator_name.to_string()),
        quantity,
        SensorimotorFrameV1::ParentJoint,
        SensorimotorComponentV1::Scalar,
        SensorimotorValueContractV1 {
            unit,
            min,
            max,
            bins,
        },
    );
    address
        .validate()
        .map_err(SemanticHumanoidActuationErrorV1::Sensorimotor)?;
    Ok(address)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticHumanoidActuationErrorV1 {
    SchemaMismatch,
    NonPhysicalMode(ActuationMode),
    ActuatorCount { expected: usize, actual: usize },
    InvalidClippedCount { clipped: usize, actuators: usize },
    NonFiniteValue,
    NonFiniteAt { index: usize },
    AddressMismatch { index: usize },
    MissingStageEvidence(HumanoidActuationStageV1),
    EmptyEvidenceId,
    MorphologyMismatch,
    PhysicalModeMismatch,
    PhysicalRoleMismatch { index: usize },
    ContractMismatch { index: usize },
    Sensorimotor(&'static str),
    Adaptation(ActuationAdaptationError),
}

impl From<ActuationAdaptationError> for SemanticHumanoidActuationErrorV1 {
    fn from(value: ActuationAdaptationError) -> Self {
        Self::Adaptation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_with_first(value: f32) -> HumanoidCommand {
        let mut torques = vec![0.0; HumanoidMorphology::Dmc21.num_actuators()];
        torques[0] = value;
        HumanoidCommand { torques }
    }

    fn evidence(id: &str) -> Option<String> {
        Some(id.to_string())
    }

    #[test]
    fn normalized_intent_projects_to_newton_metres_without_unit_leakage() {
        let frame = SemanticHumanoidActuationFrameV1::project_requested_normalized_torque_command(
            &command_with_first(0.5),
            &HumanoidState::standing(),
            HumanoidMorphology::Dmc21,
        )
        .unwrap();

        assert_eq!(frame.stage, HumanoidActuationStageV1::RequestedPhysicalProjection);
        assert_eq!(frame.evidence_id, None);
        assert_eq!(frame.physical_mode, ActuationMode::TorqueNewtonMetres);
        assert_eq!(frame.source_mode, ActuationMode::NormalizedTorque);
        assert_eq!(frame.values[0].value, 50.0);
        assert!(matches!(
            frame.values[0].address.quantity,
            SensorimotorQuantityV1::Torque
        ));
        assert!(matches!(
            frame.values[0].address.value_contract.unit,
            SensorimotorUnitV1::NewtonMeter
        ));
        assert_eq!(frame.out_of_contract_count(), 0);
    }

    #[test]
    fn stage_changes_causal_status_not_physical_role_identity() {
        let requested = SemanticHumanoidActuationFrameV1::project_requested_normalized_torque_command(
            &command_with_first(0.5),
            &HumanoidState::standing(),
            HumanoidMorphology::Dmc21,
        )
        .unwrap();
        let applied = SemanticHumanoidActuationFrameV1::from_physical_command(
            &command_with_first(40.0),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::TorqueNewtonMetres,
            evidence("sim-step:42"),
        )
        .unwrap();

        assert_ne!(requested.stage, applied.stage);
        assert_eq!(
            requested.values[0].address.physical_role_digest().unwrap(),
            applied.values[0].address.physical_role_digest().unwrap()
        );
        assert_eq!(
            requested.values[0].address.contract_digest().unwrap(),
            applied.values[0].address.contract_digest().unwrap()
        );
    }

    #[test]
    fn stage_comparison_detects_action_changed_after_request() {
        let requested = SemanticHumanoidActuationFrameV1::from_physical_command(
            &command_with_first(50.0),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::RequestedPhysicalProjection,
            ActuationMode::TorqueNewtonMetres,
            None,
        )
        .unwrap();
        let applied = SemanticHumanoidActuationFrameV1::from_physical_command(
            &command_with_first(40.0),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::TorqueNewtonMetres,
            evidence("sim-step:43"),
        )
        .unwrap();

        let result = compare_humanoid_actuation_stages_v1(&requested, &applied).unwrap();
        assert_eq!(result.compared_actuators, 21);
        assert_eq!(result.physical_role_matches, 21);
        assert_eq!(result.exact_contract_matches, 21);
        assert_eq!(result.exact_value_matches, 20);
        assert_eq!(result.changed_actuators, 1);
        assert!(result.any_action_changed());
        assert_eq!(result.target_evidence_id.as_deref(), Some("sim-step:43"));
        assert!((result.max_abs_delta - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn clipping_evidence_survives_physical_projection() {
        let command = HumanoidCommand {
            torques: vec![1.25; HumanoidMorphology::Dmc21.num_actuators()],
        };
        let frame = SemanticHumanoidActuationFrameV1::project_normalized_torque_command(
            &command,
            &HumanoidState::standing(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::SafetyProjectedPhysical,
            evidence("safety-projection:test"),
        )
        .unwrap();

        assert_eq!(frame.clipped_joints, 21);
        assert_eq!(frame.out_of_contract_count(), 0);
        assert_eq!(frame.values[0].value, 100.0);
    }

    #[test]
    fn normalized_mode_cannot_masquerade_as_physical_action() {
        let error = SemanticHumanoidActuationFrameV1::from_physical_command(
            &HumanoidCommand::zero(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::NormalizedTorque,
            evidence("backend:test"),
        )
        .unwrap_err();
        assert_eq!(
            error,
            SemanticHumanoidActuationErrorV1::NonPhysicalMode(ActuationMode::NormalizedTorque)
        );
    }

    #[test]
    fn strong_stage_without_evidence_id_fails_closed() {
        let error = SemanticHumanoidActuationFrameV1::from_physical_command(
            &HumanoidCommand::zero(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::TorqueNewtonMetres,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error,
            SemanticHumanoidActuationErrorV1::MissingStageEvidence(
                HumanoidActuationStageV1::BackendAppliedPhysical
            )
        );

        let error = SemanticHumanoidActuationFrameV1::from_physical_command(
            &HumanoidCommand::zero(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::MeasuredExecution,
            ActuationMode::TorqueNewtonMetres,
            Some("   ".into()),
        )
        .unwrap_err();
        assert_eq!(error, SemanticHumanoidActuationErrorV1::EmptyEvidenceId);
    }

    #[test]
    fn position_targets_use_actuator_joint_position_contracts() {
        let frame = SemanticHumanoidActuationFrameV1::from_physical_command(
            &HumanoidCommand::zero(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::PositionTargetRadians,
            evidence("position-backend:test"),
        )
        .unwrap();
        assert!(matches!(
            frame.values[0].address.subject,
            SensorimotorSubjectV1::Actuator(_)
        ));
        assert!(matches!(
            frame.values[0].address.quantity,
            SensorimotorQuantityV1::JointPosition
        ));
        assert!(matches!(
            frame.values[0].address.value_contract.unit,
            SensorimotorUnitV1::Radian
        ));
    }

    #[test]
    fn malformed_action_frames_fail_closed() {
        let mut frame = SemanticHumanoidActuationFrameV1::from_physical_command(
            &HumanoidCommand::zero(),
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::RequestedPhysicalProjection,
            ActuationMode::TorqueNewtonMetres,
            None,
        )
        .unwrap();
        frame.values[0].value = f64::NAN;
        assert!(frame.validate().is_err());

        let short = HumanoidCommand {
            torques: vec![0.0; 20],
        };
        assert!(SemanticHumanoidActuationFrameV1::from_physical_command(
            &short,
            HumanoidMorphology::Dmc21,
            HumanoidActuationStageV1::BackendAppliedPhysical,
            ActuationMode::TorqueNewtonMetres,
            evidence("backend:test"),
        )
        .is_err());
    }
}
