// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact C0 JointLinkCoupon design-subject compiler.
//!
//! C0A maps exact nominal design intent into the already-qualified
//! `RobotDesignSubjectV1` identity model. It creates no fabrication, physical,
//! safety, or experiment authority.

use crate::design_parameters::{
    DesignLengthUm, DesignParameterError, DesignParameterSetId, DesignParameterSetV1,
};
use crate::{
    ContentDigest, DesignComponentId, DesignComponentV1, GeometryDesignId, GeometryDesignRefV1,
    GeometryRepresentationV1, MaterialAssignmentV1, RobotDesignError, RobotDesignId,
    RobotDesignSubjectV1, RobotMorphologyDesignV1,
};
use sha2::{Digest, Sha256};
use std::fmt;

pub const C0_COUPON_PROFILE: &str = "c0-joint-link-coupon-v1";
pub const C0_COMPONENT_ID: &str = "c0-link-coupon";
pub const C0_GEOMETRY_ID: &str = "c0-link-coupon-geometry";
pub const C0_WIDTH_PARAMETER_ID: &str = "section-width";
pub const C0_HEIGHT_PARAMETER_ID: &str = "section-height";
pub const C0_GEOMETRY_SCHEMA_ID: &str = "symthaea.robot-design.c0-geometry-intent.v1";
pub const C0_GEOMETRY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub enum C0CouponError {
    Parameters(DesignParameterError),
    RobotDesign(RobotDesignError),
    MissingParameter(&'static str),
    UnexpectedParameter(String),
    NonPositiveDimension(&'static str),
    ProfileCardinality(&'static str),
}

impl fmt::Display for C0CouponError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameters(error) => write!(formatter, "C0 parameter error: {error}"),
            Self::RobotDesign(error) => write!(formatter, "C0 robot-design error: {error}"),
            Self::MissingParameter(parameter) => {
                write!(formatter, "missing C0 parameter: {parameter}")
            }
            Self::UnexpectedParameter(parameter) => {
                write!(formatter, "unexpected C0 parameter: {parameter}")
            }
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 dimension must be positive: {field}")
            }
            Self::ProfileCardinality(reason) => {
                write!(formatter, "invalid C0 robot-design profile cardinality: {reason}")
            }
        }
    }
}

impl std::error::Error for C0CouponError {}

impl From<DesignParameterError> for C0CouponError {
    fn from(value: DesignParameterError) -> Self {
        Self::Parameters(value)
    }
}

impl From<RobotDesignError> for C0CouponError {
    fn from(value: RobotDesignError) -> Self {
        Self::RobotDesign(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C0JointLinkCouponTemplateV1 {
    pub fixed_length: DesignLengthUm,
    pub requirement_snapshot: ContentDigest,
    pub material_digest: ContentDigest,
    pub fabrication_constraints: ContentDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C0RectangularGeometryIntentV1 {
    pub length: DesignLengthUm,
    pub width: DesignLengthUm,
    pub height: DesignLengthUm,
}

impl C0RectangularGeometryIntentV1 {
    pub fn digest(self) -> Result<ContentDigest, C0CouponError> {
        validate_positive(self.length, "length")?;
        validate_positive(self.width, "width")?;
        validate_positive(self.height, "height")?;

        let mut out = Vec::new();
        put_str(&mut out, C0_GEOMETRY_SCHEMA_ID);
        put_u32(&mut out, C0_GEOMETRY_SCHEMA_VERSION);
        put_str(&mut out, "solid-rectangular");
        put_str(&mut out, "x=length;y=width;z=height;origin=center;rotation=identity");
        put_u64(&mut out, self.length.micrometres());
        put_u64(&mut out, self.width.micrometres());
        put_u64(&mut out, self.height.micrometres());
        Ok(ContentDigest::from_bytes(Sha256::digest(out).into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C0CompiledCouponV1 {
    pub subject: RobotDesignSubjectV1,
    pub design_id: RobotDesignId,
    pub parameter_set_id: DesignParameterSetId,
    pub geometry_intent: C0RectangularGeometryIntentV1,
    pub geometry_intent_digest: ContentDigest,
}

pub fn compile_c0_coupon(
    template: C0JointLinkCouponTemplateV1,
    parameter_set: &DesignParameterSetV1,
) -> Result<C0CompiledCouponV1, C0CouponError> {
    parameter_set.validate()?;
    validate_positive(template.fixed_length, "fixed length")?;

    for parameter in parameter_set.values() {
        match parameter.id.as_str() {
            C0_WIDTH_PARAMETER_ID | C0_HEIGHT_PARAMETER_ID => {}
            other => return Err(C0CouponError::UnexpectedParameter(other.to_string())),
        }
    }
    if parameter_set.values().len() != 2 {
        return Err(C0CouponError::ProfileCardinality(
            "C0 V1 requires exactly width and height",
        ));
    }

    let width = parameter_set
        .value(C0_WIDTH_PARAMETER_ID)
        .ok_or(C0CouponError::MissingParameter(C0_WIDTH_PARAMETER_ID))?;
    let height = parameter_set
        .value(C0_HEIGHT_PARAMETER_ID)
        .ok_or(C0CouponError::MissingParameter(C0_HEIGHT_PARAMETER_ID))?;
    validate_positive(width, "section width")?;
    validate_positive(height, "section height")?;

    let parameter_set_id = parameter_set.id()?;
    let geometry_intent = C0RectangularGeometryIntentV1 {
        length: template.fixed_length,
        width,
        height,
    };
    let geometry_intent_digest = geometry_intent.digest()?;

    let component_id = DesignComponentId::new(C0_COMPONENT_ID)?;
    let morphology = RobotMorphologyDesignV1 {
        profile: C0_COUPON_PROFILE.to_string(),
        components: vec![DesignComponentV1 {
            id: component_id.clone(),
            semantic_role: "structural-link-coupon".to_string(),
            display_name: None,
        }],
        joints: Vec::new(),
    };

    let mut subject = RobotDesignSubjectV1::new(
        template.requirement_snapshot,
        morphology,
        template.fabrication_constraints,
        ContentDigest::from_bytes(parameter_set_id.into_bytes()),
    );
    subject.geometry = vec![GeometryDesignRefV1 {
        id: GeometryDesignId::new(C0_GEOMETRY_ID)?,
        component: component_id.clone(),
        representation: GeometryRepresentationV1::Parametric,
        artifact_digest: geometry_intent_digest,
    }];
    subject.material_assignments = vec![MaterialAssignmentV1 {
        component: component_id,
        material_digest: template.material_digest,
    }];
    subject.validate()?;
    validate_c0_cardinality(&subject)?;
    let design_id = subject.design_id()?;

    Ok(C0CompiledCouponV1 {
        subject,
        design_id,
        parameter_set_id,
        geometry_intent,
        geometry_intent_digest,
    })
}

pub fn validate_c0_cardinality(subject: &RobotDesignSubjectV1) -> Result<(), C0CouponError> {
    if subject.morphology.components.len() != 1 {
        return Err(C0CouponError::ProfileCardinality(
            "exactly one component required",
        ));
    }
    if !subject.morphology.joints.is_empty() {
        return Err(C0CouponError::ProfileCardinality("joints must be empty"));
    }
    if subject.geometry.len() != 1 {
        return Err(C0CouponError::ProfileCardinality(
            "exactly one geometry reference required",
        ));
    }
    if subject.material_assignments.len() != 1 {
        return Err(C0CouponError::ProfileCardinality(
            "exactly one material assignment required",
        ));
    }
    if !subject.actuator_slots.is_empty() {
        return Err(C0CouponError::ProfileCardinality(
            "actuator slots must be empty",
        ));
    }
    if !subject.sensor_slots.is_empty() {
        return Err(C0CouponError::ProfileCardinality(
            "sensor slots must be empty",
        ));
    }
    if !subject.control_interfaces.is_empty() {
        return Err(C0CouponError::ProfileCardinality(
            "control interfaces must be empty",
        ));
    }
    Ok(())
}

fn validate_positive(value: DesignLengthUm, field: &'static str) -> Result<(), C0CouponError> {
    if value.micrometres() == 0 {
        Err(C0CouponError::NonPositiveDimension(field))
    } else {
        Ok(())
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_be_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design_parameters::{DesignLengthParameterV1, DesignParameterId};

    fn digest(byte: u8) -> ContentDigest {
        ContentDigest::from_bytes([byte; 32])
    }

    fn id(value: &str) -> DesignParameterId {
        DesignParameterId::new(value).unwrap()
    }

    fn parameters(width: u64, height: u64, reversed: bool) -> DesignParameterSetV1 {
        let width = DesignLengthParameterV1 {
            id: id(C0_WIDTH_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(width),
        };
        let height = DesignLengthParameterV1 {
            id: id(C0_HEIGHT_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(height),
        };
        DesignParameterSetV1::new(if reversed {
            vec![height, width]
        } else {
            vec![width, height]
        })
        .unwrap()
    }

    fn template() -> C0JointLinkCouponTemplateV1 {
        C0JointLinkCouponTemplateV1 {
            fixed_length: DesignLengthUm::from_micrometres(300_000),
            requirement_snapshot: digest(1),
            material_digest: digest(2),
            fabrication_constraints: digest(3),
        }
    }

    #[test]
    fn exact_same_intent_is_order_invariant() {
        let first = compile_c0_coupon(template(), &parameters(20_000, 6_000, false)).unwrap();
        let second = compile_c0_coupon(template(), &parameters(20_000, 6_000, true)).unwrap();
        assert_eq!(first.parameter_set_id, second.parameter_set_id);
        assert_eq!(first.geometry_intent_digest, second.geometry_intent_digest);
        assert_eq!(first.design_id, second.design_id);
    }

    #[test]
    fn width_and_height_are_semantic_axes_not_sorted_numbers() {
        let first = compile_c0_coupon(template(), &parameters(20_000, 6_000, false)).unwrap();
        let swapped = compile_c0_coupon(template(), &parameters(6_000, 20_000, false)).unwrap();
        assert_ne!(first.geometry_intent_digest, swapped.geometry_intent_digest);
        assert_ne!(first.design_id, swapped.design_id);
    }

    #[test]
    fn generated_subject_has_strict_one_component_zero_joint_profile() {
        let compiled = compile_c0_coupon(template(), &parameters(20_000, 6_000, false)).unwrap();
        validate_c0_cardinality(&compiled.subject).unwrap();
        assert_eq!(compiled.subject.morphology.components.len(), 1);
        assert!(compiled.subject.morphology.joints.is_empty());
        assert!(compiled.subject.actuator_slots.is_empty());
        assert!(compiled.subject.sensor_slots.is_empty());
        assert!(compiled.subject.control_interfaces.is_empty());
    }

    #[test]
    fn changed_requirement_material_or_fabrication_profile_changes_design_identity() {
        let params = parameters(20_000, 6_000, false);
        let original = compile_c0_coupon(template(), &params).unwrap();

        let mut changed = template();
        changed.requirement_snapshot = digest(9);
        assert_ne!(compile_c0_coupon(changed, &params).unwrap().design_id, original.design_id);

        let mut changed = template();
        changed.material_digest = digest(9);
        assert_ne!(compile_c0_coupon(changed, &params).unwrap().design_id, original.design_id);

        let mut changed = template();
        changed.fabrication_constraints = digest(9);
        assert_ne!(compile_c0_coupon(changed, &params).unwrap().design_id, original.design_id);
    }

    #[test]
    fn human_display_label_does_not_change_generic_design_identity() {
        let compiled = compile_c0_coupon(template(), &parameters(20_000, 6_000, false)).unwrap();
        let mut renamed = compiled.subject.clone();
        renamed.display_name = Some("friendly coupon label".to_string());
        assert_eq!(renamed.design_id().unwrap(), compiled.design_id);
    }

    #[test]
    fn zero_dimension_and_extra_parameter_reject() {
        assert!(compile_c0_coupon(template(), &parameters(0, 6_000, false)).is_err());

        let params = DesignParameterSetV1::new(vec![
            DesignLengthParameterV1 {
                id: id(C0_WIDTH_PARAMETER_ID),
                value: DesignLengthUm::from_micrometres(20_000),
            },
            DesignLengthParameterV1 {
                id: id(C0_HEIGHT_PARAMETER_ID),
                value: DesignLengthUm::from_micrometres(6_000),
            },
            DesignLengthParameterV1 {
                id: id("third-axis"),
                value: DesignLengthUm::from_micrometres(1),
            },
        ])
        .unwrap();
        assert!(matches!(
            compile_c0_coupon(template(), &params),
            Err(C0CouponError::UnexpectedParameter(_))
        ));
    }
}
