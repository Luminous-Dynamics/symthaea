// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal physical-action stage primitives for Embodiment Contract v2.
//!
//! This module extracts only the embodiment-neutral causal stage semantics already
//! established by humanoid HDC-R3.10. It deliberately does not define actuator
//! addresses, morphology, physical units, transport acknowledgement, authority, or
//! safety policy. Those remain separate propositions.

use serde::{Deserialize, Serialize};

/// Schema version for [`ActionStageBindingV1`].
pub const ACTION_STAGE_BINDING_SCHEMA_V1: u16 = 1;

/// Causal status of one physical action representation.
///
/// Stage is metadata about *where in the actuation path a physical command value
/// came from*. It is not part of physical-role identity and is not a trust/safety
/// ranking. Variant names and snake-case serialization intentionally match the
/// existing humanoid HDC-R3.10 `HumanoidActuationStageV1` contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalActionStageV1 {
    /// SI/physical-unit projection of controller/policy intent before downstream gates.
    RequestedPhysicalProjection,
    /// Physical command after a safety/projection layer has accepted or changed it.
    SafetyProjectedPhysical,
    /// Physical command a backend reports as consumed by its plant/physics boundary.
    BackendAppliedPhysical,
    /// Independently measured actuator execution/feedback.
    MeasuredExecution,
}

impl PhysicalActionStageV1 {
    /// Stable wire token matching the R3.10 snake-case representation.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::RequestedPhysicalProjection => "requested_physical_projection",
            Self::SafetyProjectedPhysical => "safety_projected_physical",
            Self::BackendAppliedPhysical => "backend_applied_physical",
            Self::MeasuredExecution => "measured_execution",
        }
    }

    /// Whether this stage makes an external boundary claim that requires evidence.
    ///
    /// A requested physical projection may be a deterministic local derivation.
    /// Every stronger causal boundary must name the external evidence supporting it.
    pub const fn requires_evidence_id(self) -> bool {
        !matches!(self, Self::RequestedPhysicalProjection)
    }
}

/// Evidence binding for one physical causal-action stage.
///
/// This type says only *which stage is being claimed* and *which evidence record
/// supports that claim*. It intentionally contains no actuator values. R3.1
/// `SensorimotorAddressV1` remains the universal physical-role vocabulary for the
/// HDC-R3 lineage; platform-specific action frames should combine that vocabulary
/// with this stage type after the lineages converge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionStageBindingV1 {
    /// Schema version. Must equal [`ACTION_STAGE_BINDING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Causal stage represented by the associated physical action.
    pub stage: PhysicalActionStageV1,
    /// External evidence reference required by every stage after requested projection.
    pub evidence_id: Option<String>,
}

impl ActionStageBindingV1 {
    /// Construct and validate one causal-stage binding.
    pub fn new(
        stage: PhysicalActionStageV1,
        evidence_id: Option<String>,
    ) -> Result<Self, ActionStageValidationError> {
        let value = Self {
            schema_version: ACTION_STAGE_BINDING_SCHEMA_V1,
            stage,
            evidence_id,
        };
        value.validate()?;
        Ok(value)
    }

    /// Convenience constructor for a deterministic requested physical projection.
    pub fn requested_projection() -> Self {
        Self {
            schema_version: ACTION_STAGE_BINDING_SCHEMA_V1,
            stage: PhysicalActionStageV1::RequestedPhysicalProjection,
            evidence_id: None,
        }
    }

    /// Validate schema and evidence-strength invariants.
    pub fn validate(&self) -> Result<(), ActionStageValidationError> {
        if self.schema_version != ACTION_STAGE_BINDING_SCHEMA_V1 {
            return Err(ActionStageValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        if let Some(evidence_id) = &self.evidence_id {
            validate_evidence_id(evidence_id)?;
        }

        if self.stage.requires_evidence_id() && self.evidence_id.is_none() {
            return Err(ActionStageValidationError::MissingEvidenceForStage(
                self.stage,
            ));
        }

        Ok(())
    }
}

fn validate_evidence_id(value: &str) -> Result<(), ActionStageValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(ActionStageValidationError::InvalidEvidenceId);
    }
    Ok(())
}

/// Validation failure for [`ActionStageBindingV1`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionStageValidationError {
    /// Binding uses a schema version not understood by this v1 validator.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// A stronger causal stage omitted its required evidence reference.
    MissingEvidenceForStage(PhysicalActionStageV1),
    /// Evidence reference was empty, padded, or contained control characters.
    InvalidEvidenceId,
}

impl std::fmt::Display for ActionStageValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported physical-action stage schema version {found}")
            }
            Self::MissingEvidenceForStage(stage) => {
                write!(f, "physical-action stage {stage:?} requires evidence")
            }
            Self::InvalidEvidenceId => write!(f, "invalid physical-action evidence identifier"),
        }
    }
}

impl std::error::Error for ActionStageValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_tokens_match_r3_10_contract() {
        let cases = [
            (
                PhysicalActionStageV1::RequestedPhysicalProjection,
                "requested_physical_projection",
            ),
            (
                PhysicalActionStageV1::SafetyProjectedPhysical,
                "safety_projected_physical",
            ),
            (
                PhysicalActionStageV1::BackendAppliedPhysical,
                "backend_applied_physical",
            ),
            (
                PhysicalActionStageV1::MeasuredExecution,
                "measured_execution",
            ),
        ];

        for (stage, expected) in cases {
            assert_eq!(stage.wire_token(), expected);
            assert_eq!(serde_json::to_string(&stage).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn requested_projection_does_not_require_external_evidence() {
        let binding = ActionStageBindingV1::requested_projection();
        binding.validate().unwrap();
        assert_eq!(
            binding.stage,
            PhysicalActionStageV1::RequestedPhysicalProjection
        );
        assert_eq!(binding.evidence_id, None);
    }

    #[test]
    fn stronger_stages_require_evidence() {
        for stage in [
            PhysicalActionStageV1::SafetyProjectedPhysical,
            PhysicalActionStageV1::BackendAppliedPhysical,
            PhysicalActionStageV1::MeasuredExecution,
        ] {
            assert_eq!(
                ActionStageBindingV1::new(stage, None),
                Err(ActionStageValidationError::MissingEvidenceForStage(stage))
            );
            ActionStageBindingV1::new(stage, Some("evidence:sha256:01".to_string())).unwrap();
        }
    }

    #[test]
    fn malformed_evidence_ids_are_rejected() {
        for invalid in ["", "   ", " padded", "padded ", "bad\nvalue"] {
            assert_eq!(
                ActionStageBindingV1::new(
                    PhysicalActionStageV1::BackendAppliedPhysical,
                    Some(invalid.to_string()),
                ),
                Err(ActionStageValidationError::InvalidEvidenceId)
            );
        }
    }

    #[test]
    fn requested_projection_may_be_evidence_bound_when_desired() {
        let binding = ActionStageBindingV1::new(
            PhysicalActionStageV1::RequestedPhysicalProjection,
            Some("derivation:requested-command:01".to_string()),
        )
        .unwrap();
        assert_eq!(
            binding.evidence_id.as_deref(),
            Some("derivation:requested-command:01")
        );
    }

    #[test]
    fn serialization_round_trip_preserves_binding() {
        let original = ActionStageBindingV1::new(
            PhysicalActionStageV1::MeasuredExecution,
            Some("feedback:encoder:step-42".to_string()),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&original).unwrap();
        let restored: ActionStageBindingV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, original);
        restored.validate().unwrap();
    }

    #[test]
    fn unsupported_schema_version_fails_closed() {
        let mut binding = ActionStageBindingV1::requested_projection();
        binding.schema_version = 2;
        assert_eq!(
            binding.validate(),
            Err(ActionStageValidationError::UnsupportedSchemaVersion { found: 2 })
        );
    }
}
