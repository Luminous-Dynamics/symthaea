// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded C0 JointLinkCoupon design semantics.
//!
//! This module compiles one exact rectangular structural coupon design into the
//! canonical robot-design subject. Search-domain provenance is validated and
//! retained in the compile receipt, but deliberately excluded from design
//! identity: the same exact design discovered under a different search domain
//! remains the same design.

use crate::exact_parameters::{
    DesignParameterId, ExactDesignLengthUmV1, ExactDesignParameterSetId,
    ExactDesignParameterSetV1, ExactDesignValueV1, ExactLengthDomainId,
    ExactLengthDomainV1, ExactParameterError,
};
use crate::{
    ContentDigest, DesignComponentId, GeometryDesignId, GeometryDesignRefV1,
    GeometryRepresentationV1, MaterialAssignmentV1, RobotDesignError, RobotDesignId,
    RobotDesignSubjectV1, RobotMorphologyDesignV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const C0_COUPON_SCHEMA_ID: &str = "symthaea.robot-design.c0-joint-link-coupon.v1";
pub const C0_COUPON_SCHEMA_VERSION: u32 = 1;
pub const C0_GEOMETRY_INTENT_SCHEMA_ID: &str =
    "symthaea.robot-design.c0-rectangular-geometry-intent.v1";
pub const C0_GEOMETRY_INTENT_SCHEMA_VERSION: u32 = 1;
pub const C0_SEARCH_DOMAIN_SCHEMA_ID: &str =
    "symthaea.robot-design.c0-coupon-search-domain.v1";
pub const C0_SEARCH_DOMAIN_SCHEMA_VERSION: u32 = 1;

pub const C0_MORPHOLOGY_PROFILE: &str = "c0-joint-link-coupon-v1";
pub const C0_COMPONENT_ID: &str = "joint-link-coupon";
pub const C0_COMPONENT_ROLE: &str = "structural-link-coupon";
pub const C0_GEOMETRY_ID: &str = "coupon-body";

pub const C0_PARAMETER_LINK_LENGTH: &str = "link_length";
pub const C0_PARAMETER_SECTION_WIDTH: &str = "section_width";
pub const C0_PARAMETER_SECTION_HEIGHT: &str = "section_height";

const C0_SECTION_FAMILY_SOLID_RECTANGULAR: u8 = 0;

/// Frozen section-axis semantics for the C0 rectangular coupon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0SectionOrientationV1 {
    /// X = longitudinal axis, Y = section width, Z = section height/bending depth.
    LongitudinalXWidthYHeightZ,
    /// X = longitudinal axis, Z = section width, Y = section height/bending depth.
    ///
    /// This is supported as an exact semantic distinction for identity tests and
    /// future campaigns; a concrete C0 campaign should freeze exactly one
    /// orientation before search.
    LongitudinalXWidthZHeightY,
}

impl C0SectionOrientationV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::LongitudinalXWidthYHeightZ => 0,
            Self::LongitudinalXWidthZHeightY => 1,
        }
    }
}

/// Frozen design-level template values that are not search variables in C0 V1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0JointLinkCouponTemplateV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub link_length: ExactDesignLengthUmV1,
    pub orientation: C0SectionOrientationV1,
    pub requirement_snapshot: ContentDigest,
    pub material_subject: ContentDigest,
    pub fabrication_constraints: ContentDigest,
}

impl C0JointLinkCouponTemplateV1 {
    pub fn new(
        link_length: ExactDesignLengthUmV1,
        orientation: C0SectionOrientationV1,
        requirement_snapshot: ContentDigest,
        material_subject: ContentDigest,
        fabrication_constraints: ContentDigest,
    ) -> Self {
        Self {
            schema_id: C0_COUPON_SCHEMA_ID.to_string(),
            schema_version: C0_COUPON_SCHEMA_VERSION,
            link_length,
            orientation,
            requirement_snapshot,
            material_subject,
            fabrication_constraints,
        }
    }

    pub fn validate(&self) -> Result<(), C0CouponError> {
        if self.schema_id != C0_COUPON_SCHEMA_ID || self.schema_version != C0_COUPON_SCHEMA_VERSION {
            return Err(C0CouponError::UnsupportedTemplateSchema {
                schema_id: self.schema_id.clone(),
                schema_version: self.schema_version,
            });
        }
        if self.link_length.as_um() == 0 {
            return Err(C0CouponError::NonPositiveParameter(C0_PARAMETER_LINK_LENGTH));
        }
        Ok(())
    }
}

/// Exact as-designed rectangular geometry intent. This is not CSG, a mesh, or
/// a fabricated article.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0RectangularGeometryIntentV1 {
    pub link_length: ExactDesignLengthUmV1,
    pub section_width: ExactDesignLengthUmV1,
    pub section_height: ExactDesignLengthUmV1,
    pub orientation: C0SectionOrientationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0GeometryIntentId(ContentDigest);

impl C0GeometryIntentId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for C0GeometryIntentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl C0RectangularGeometryIntentV1 {
    pub fn validate(&self) -> Result<(), C0CouponError> {
        if self.link_length.as_um() == 0 {
            return Err(C0CouponError::NonPositiveParameter(C0_PARAMETER_LINK_LENGTH));
        }
        if self.section_width.as_um() == 0 {
            return Err(C0CouponError::NonPositiveParameter(C0_PARAMETER_SECTION_WIDTH));
        }
        if self.section_height.as_um() == 0 {
            return Err(C0CouponError::NonPositiveParameter(C0_PARAMETER_SECTION_HEIGHT));
        }
        Ok(())
    }

    pub fn canonical_transcript(&self) -> Result<Vec<u8>, C0CouponError> {
        self.validate()?;
        let mut out = Vec::new();
        put_str(&mut out, C0_GEOMETRY_INTENT_SCHEMA_ID);
        put_u32(&mut out, C0_GEOMETRY_INTENT_SCHEMA_VERSION);
        put_u8(&mut out, C0_SECTION_FAMILY_SOLID_RECTANGULAR);
        put_u8(&mut out, self.orientation.tag());
        put_u64(&mut out, self.link_length.as_um());
        put_u64(&mut out, self.section_width.as_um());
        put_u64(&mut out, self.section_height.as_um());
        Ok(out)
    }

    pub fn geometry_intent_id(&self) -> Result<C0GeometryIntentId, C0CouponError> {
        let digest: [u8; 32] = Sha256::digest(self.canonical_transcript()?).into();
        Ok(C0GeometryIntentId(ContentDigest::from_bytes(digest)))
    }
}

/// The allowed width/height search domain. This identity describes how a design
/// was discovered; it is intentionally excluded from the design subject itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0CouponSearchDomainV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub width_domain: ExactLengthDomainV1,
    pub height_domain: ExactLengthDomainV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0CouponSearchDomainId(ContentDigest);

impl C0CouponSearchDomainId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for C0CouponSearchDomainId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl C0CouponSearchDomainV1 {
    pub fn new(width_domain: ExactLengthDomainV1, height_domain: ExactLengthDomainV1) -> Self {
        Self {
            schema_id: C0_SEARCH_DOMAIN_SCHEMA_ID.to_string(),
            schema_version: C0_SEARCH_DOMAIN_SCHEMA_VERSION,
            width_domain,
            height_domain,
        }
    }

    pub fn validate(&self) -> Result<(), C0CouponError> {
        if self.schema_id != C0_SEARCH_DOMAIN_SCHEMA_ID
            || self.schema_version != C0_SEARCH_DOMAIN_SCHEMA_VERSION
        {
            return Err(C0CouponError::UnsupportedSearchDomainSchema {
                schema_id: self.schema_id.clone(),
                schema_version: self.schema_version,
            });
        }
        self.width_domain.validate()?;
        self.height_domain.validate()?;
        validate_domain_semantics(
            &self.width_domain,
            C0_PARAMETER_SECTION_WIDTH,
            "width domain",
        )?;
        validate_domain_semantics(
            &self.height_domain,
            C0_PARAMETER_SECTION_HEIGHT,
            "height domain",
        )?;
        Ok(())
    }

    pub fn contains(
        &self,
        width: ExactDesignLengthUmV1,
        height: ExactDesignLengthUmV1,
    ) -> Result<bool, C0CouponError> {
        self.validate()?;
        Ok(self.width_domain.contains(width)? && self.height_domain.contains(height)?)
    }

    pub fn search_domain_id(&self) -> Result<C0CouponSearchDomainId, C0CouponError> {
        self.validate()?;
        let width_id = self.width_domain.domain_id()?;
        let height_id = self.height_domain.domain_id()?;
        let mut out = Vec::new();
        put_str(&mut out, C0_SEARCH_DOMAIN_SCHEMA_ID);
        put_u32(&mut out, C0_SEARCH_DOMAIN_SCHEMA_VERSION);
        put_digest(&mut out, width_id.digest());
        put_digest(&mut out, height_id.digest());
        let digest: [u8; 32] = Sha256::digest(out).into();
        Ok(C0CouponSearchDomainId(ContentDigest::from_bytes(digest)))
    }
}

/// Immutable provenance returned by the strict C0 compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0CouponCompileReceiptV1 {
    pub robot_design_id: RobotDesignId,
    pub geometry_intent_id: C0GeometryIntentId,
    pub parameter_set_id: ExactDesignParameterSetId,
    pub search_domain_id: C0CouponSearchDomainId,
    pub width_domain_id: ExactLengthDomainId,
    pub height_domain_id: ExactLengthDomainId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0CompiledCouponV1 {
    pub subject: RobotDesignSubjectV1,
    pub geometry_intent: C0RectangularGeometryIntentV1,
    pub receipt: C0CouponCompileReceiptV1,
}

/// Compile one exact C0 coupon design.
///
/// Search-domain membership is a compile-time admission rule and receipt input,
/// not a design-identity input.
pub fn compile_c0_joint_link_coupon(
    template: &C0JointLinkCouponTemplateV1,
    selected_parameters: &ExactDesignParameterSetV1,
    search_domain: &C0CouponSearchDomainV1,
) -> Result<C0CompiledCouponV1, C0CouponError> {
    template.validate()?;
    selected_parameters.validate()?;
    search_domain.validate()?;

    let (link_length, section_width, section_height) = extract_c0_lengths(selected_parameters)?;

    if link_length.as_um() != template.link_length.as_um() {
        return Err(C0CouponError::LinkLengthMismatch {
            template_um: template.link_length.as_um(),
            parameter_um: link_length.as_um(),
        });
    }

    if !search_domain.contains(section_width, section_height)? {
        return Err(C0CouponError::SelectedValueOutsideSearchDomain);
    }

    let parameter_set_id = selected_parameters.parameter_set_id()?;
    let geometry_intent = C0RectangularGeometryIntentV1 {
        link_length,
        section_width,
        section_height,
        orientation: template.orientation,
    };
    let geometry_intent_id = geometry_intent.geometry_intent_id()?;

    let component_id = DesignComponentId::new(C0_COMPONENT_ID)?;
    let geometry_id = GeometryDesignId::new(C0_GEOMETRY_ID)?;
    let morphology = RobotMorphologyDesignV1 {
        profile: C0_MORPHOLOGY_PROFILE.to_string(),
        components: vec![crate::DesignComponentV1 {
            id: component_id.clone(),
            semantic_role: C0_COMPONENT_ROLE.to_string(),
            display_name: None,
        }],
        joints: Vec::new(),
    };

    let mut subject = RobotDesignSubjectV1::new(
        template.requirement_snapshot,
        morphology,
        template.fabrication_constraints,
        parameter_set_id.digest(),
    );
    subject.geometry.push(GeometryDesignRefV1 {
        id: geometry_id,
        component: component_id.clone(),
        representation: GeometryRepresentationV1::Parametric,
        artifact_digest: geometry_intent_id.digest(),
    });
    subject.material_assignments.push(MaterialAssignmentV1 {
        component: component_id,
        material_digest: template.material_subject,
    });

    validate_strict_c0_subject(&subject)?;
    let robot_design_id = subject.design_id()?;
    let search_domain_id = search_domain.search_domain_id()?;
    let width_domain_id = search_domain.width_domain.domain_id()?;
    let height_domain_id = search_domain.height_domain.domain_id()?;

    Ok(C0CompiledCouponV1 {
        subject,
        geometry_intent,
        receipt: C0CouponCompileReceiptV1 {
            robot_design_id,
            geometry_intent_id,
            parameter_set_id,
            search_domain_id,
            width_domain_id,
            height_domain_id,
        },
    })
}

pub fn validate_strict_c0_subject(subject: &RobotDesignSubjectV1) -> Result<(), C0CouponError> {
    subject.validate()?;
    if subject.morphology.profile != C0_MORPHOLOGY_PROFILE
        || subject.morphology.components.len() != 1
        || !subject.morphology.joints.is_empty()
        || subject.geometry.len() != 1
        || subject.material_assignments.len() != 1
        || !subject.actuator_slots.is_empty()
        || !subject.sensor_slots.is_empty()
        || !subject.control_interfaces.is_empty()
    {
        return Err(C0CouponError::NotStrictC0Subject);
    }
    if subject.morphology.components[0].id.as_str() != C0_COMPONENT_ID
        || subject.morphology.components[0].semantic_role != C0_COMPONENT_ROLE
        || subject.geometry[0].id.as_str() != C0_GEOMETRY_ID
        || subject.geometry[0].component.as_str() != C0_COMPONENT_ID
        || subject.material_assignments[0].component.as_str() != C0_COMPONENT_ID
        || subject.geometry[0].representation != GeometryRepresentationV1::Parametric
    {
        return Err(C0CouponError::NotStrictC0Subject);
    }
    Ok(())
}

fn validate_domain_semantics(
    domain: &ExactLengthDomainV1,
    expected_parameter: &'static str,
    context: &'static str,
) -> Result<(), C0CouponError> {
    if domain.parameter_id.as_str() != expected_parameter {
        return Err(C0CouponError::DomainParameterMismatch {
            context,
            expected: expected_parameter,
            actual: domain.parameter_id.to_string(),
        });
    }
    if !domain.require_positive {
        return Err(C0CouponError::DomainMustRequirePositive(context));
    }
    Ok(())
}

fn extract_c0_lengths(
    selected_parameters: &ExactDesignParameterSetV1,
) -> Result<
    (
        ExactDesignLengthUmV1,
        ExactDesignLengthUmV1,
        ExactDesignLengthUmV1,
    ),
    C0CouponError,
> {
    let mut link_length = None;
    let mut section_width = None;
    let mut section_height = None;

    for parameter in &selected_parameters.parameters {
        let value = match &parameter.value {
            ExactDesignValueV1::LengthUm(value) => *value,
        };
        match parameter.id.as_str() {
            C0_PARAMETER_LINK_LENGTH => link_length = Some(value),
            C0_PARAMETER_SECTION_WIDTH => section_width = Some(value),
            C0_PARAMETER_SECTION_HEIGHT => section_height = Some(value),
            other => return Err(C0CouponError::UnexpectedParameterId(other.to_string())),
        }
    }

    if selected_parameters.parameters.len() != 3 {
        return Err(C0CouponError::UnexpectedParameterCount {
            actual: selected_parameters.parameters.len(),
        });
    }

    let link_length = link_length.ok_or(C0CouponError::MissingParameter(C0_PARAMETER_LINK_LENGTH))?;
    let section_width =
        section_width.ok_or(C0CouponError::MissingParameter(C0_PARAMETER_SECTION_WIDTH))?;
    let section_height =
        section_height.ok_or(C0CouponError::MissingParameter(C0_PARAMETER_SECTION_HEIGHT))?;

    for (name, value) in [
        (C0_PARAMETER_LINK_LENGTH, link_length),
        (C0_PARAMETER_SECTION_WIDTH, section_width),
        (C0_PARAMETER_SECTION_HEIGHT, section_height),
    ] {
        if value.as_um() == 0 {
            return Err(C0CouponError::NonPositiveParameter(name));
        }
    }

    Ok((link_length, section_width, section_height))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C0CouponError {
    UnsupportedTemplateSchema {
        schema_id: String,
        schema_version: u32,
    },
    UnsupportedSearchDomainSchema {
        schema_id: String,
        schema_version: u32,
    },
    ExactParameter(ExactParameterError),
    RobotDesign(RobotDesignError),
    UnexpectedParameterId(String),
    UnexpectedParameterCount {
        actual: usize,
    },
    MissingParameter(&'static str),
    NonPositiveParameter(&'static str),
    LinkLengthMismatch {
        template_um: u64,
        parameter_um: u64,
    },
    DomainParameterMismatch {
        context: &'static str,
        expected: &'static str,
        actual: String,
    },
    DomainMustRequirePositive(&'static str),
    SelectedValueOutsideSearchDomain,
    NotStrictC0Subject,
}

impl fmt::Display for C0CouponError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTemplateSchema {
                schema_id,
                schema_version,
            } => write!(
                formatter,
                "unsupported C0 coupon template schema {schema_id}@{schema_version}"
            ),
            Self::UnsupportedSearchDomainSchema {
                schema_id,
                schema_version,
            } => write!(
                formatter,
                "unsupported C0 coupon search-domain schema {schema_id}@{schema_version}"
            ),
            Self::ExactParameter(error) => write!(formatter, "exact-parameter error: {error}"),
            Self::RobotDesign(error) => write!(formatter, "robot-design error: {error}"),
            Self::UnexpectedParameterId(id) => write!(formatter, "unexpected C0 parameter id: {id}"),
            Self::UnexpectedParameterCount { actual } => {
                write!(formatter, "C0 parameter set must contain exactly 3 parameters, got {actual}")
            }
            Self::MissingParameter(id) => write!(formatter, "missing required C0 parameter: {id}"),
            Self::NonPositiveParameter(id) => {
                write!(formatter, "C0 parameter must be strictly positive: {id}")
            }
            Self::LinkLengthMismatch {
                template_um,
                parameter_um,
            } => write!(
                formatter,
                "C0 link length mismatch: template={template_um} um parameter={parameter_um} um"
            ),
            Self::DomainParameterMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} binds parameter {actual}, expected {expected}"
            ),
            Self::DomainMustRequirePositive(context) => {
                write!(formatter, "{context} must require strictly positive values")
            }
            Self::SelectedValueOutsideSearchDomain => {
                formatter.write_str("selected C0 width/height is outside the frozen search domain")
            }
            Self::NotStrictC0Subject => {
                formatter.write_str("robot-design subject does not satisfy strict C0 cardinality/profile")
            }
        }
    }
}

impl std::error::Error for C0CouponError {}

impl From<ExactParameterError> for C0CouponError {
    fn from(value: ExactParameterError) -> Self {
        Self::ExactParameter(value)
    }
}

impl From<RobotDesignError> for C0CouponError {
    fn from(value: RobotDesignError) -> Self {
        Self::RobotDesign(value)
    }
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
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

fn put_digest(out: &mut Vec<u8>, digest: ContentDigest) {
    out.extend_from_slice(&digest.into_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact_parameters::{
        ExactDesignParameterV1, ExactLengthDomainKindV1,
    };

    fn digest(seed: u8) -> ContentDigest {
        ContentDigest::from_bytes([seed; 32])
    }

    fn parameter(id: &str, value_um: u64) -> ExactDesignParameterV1 {
        ExactDesignParameterV1::length(
            DesignParameterId::new(id).unwrap(),
            ExactDesignLengthUmV1::from_um(value_um),
        )
    }

    fn reference_parameters() -> ExactDesignParameterSetV1 {
        ExactDesignParameterSetV1::new(vec![
            parameter(C0_PARAMETER_SECTION_WIDTH, 20_000),
            parameter(C0_PARAMETER_LINK_LENGTH, 300_000),
            parameter(C0_PARAMETER_SECTION_HEIGHT, 6_000),
        ])
    }

    fn domain_for(id: &str, values: &[u64]) -> ExactLengthDomainV1 {
        ExactLengthDomainV1::new(
            DesignParameterId::new(id).unwrap(),
            true,
            values.len() as u32,
            ExactLengthDomainKindV1::Explicit {
                values: values
                    .iter()
                    .copied()
                    .map(ExactDesignLengthUmV1::from_um)
                    .collect(),
            },
        )
    }

    fn reference_search_domain() -> C0CouponSearchDomainV1 {
        C0CouponSearchDomainV1::new(
            domain_for(C0_PARAMETER_SECTION_WIDTH, &[16_000, 18_000, 20_000, 22_000, 24_000]),
            domain_for(C0_PARAMETER_SECTION_HEIGHT, &[4_800, 5_400, 6_000, 6_600, 7_200]),
        )
    }

    fn reference_template() -> C0JointLinkCouponTemplateV1 {
        C0JointLinkCouponTemplateV1::new(
            ExactDesignLengthUmV1::from_um(300_000),
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
            digest(1),
            digest(2),
            digest(3),
        )
    }

    #[test]
    fn geometry_intent_golden_vector_is_stable() {
        let geometry = C0RectangularGeometryIntentV1 {
            link_length: ExactDesignLengthUmV1::from_um(300_000),
            section_width: ExactDesignLengthUmV1::from_um(20_000),
            section_height: ExactDesignLengthUmV1::from_um(6_000),
            orientation: C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        };
        assert_eq!(
            geometry.geometry_intent_id().unwrap().to_hex(),
            "e11301b930ac9acf28ac340aaa23eec3c112d59bac5c97ba1ce2ad1ed876fdf6"
        );
    }

    #[test]
    fn reference_design_id_is_stable() {
        let compiled = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        assert_eq!(
            compiled.receipt.robot_design_id.to_hex(),
            "47d25b5aa1851deecbc5c0f7b637d2a80ef39f603a6c898f5ec50a8c5c2a5f58"
        );
    }

    #[test]
    fn strict_c0_cardinality_has_no_robotic_actuation_surface() {
        let compiled = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        assert_eq!(compiled.subject.morphology.components.len(), 1);
        assert!(compiled.subject.morphology.joints.is_empty());
        assert_eq!(compiled.subject.geometry.len(), 1);
        assert_eq!(compiled.subject.material_assignments.len(), 1);
        assert!(compiled.subject.actuator_slots.is_empty());
        assert!(compiled.subject.sensor_slots.is_empty());
        assert!(compiled.subject.control_interfaces.is_empty());
    }

    #[test]
    fn parameter_insertion_order_does_not_change_design_identity() {
        let expected = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap()
        .receipt
        .robot_design_id;
        let reordered = ExactDesignParameterSetV1::new(vec![
            parameter(C0_PARAMETER_SECTION_HEIGHT, 6_000),
            parameter(C0_PARAMETER_SECTION_WIDTH, 20_000),
            parameter(C0_PARAMETER_LINK_LENGTH, 300_000),
        ]);
        let actual = compile_c0_joint_link_coupon(
            &reference_template(),
            &reordered,
            &reference_search_domain(),
        )
        .unwrap()
        .receipt
        .robot_design_id;
        assert_eq!(actual, expected);
    }

    #[test]
    fn search_domain_changes_provenance_not_design_identity() {
        let first = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        let wider = C0CouponSearchDomainV1::new(
            domain_for(
                C0_PARAMETER_SECTION_WIDTH,
                &[14_000, 16_000, 18_000, 20_000, 22_000, 24_000, 26_000],
            ),
            domain_for(
                C0_PARAMETER_SECTION_HEIGHT,
                &[4_200, 4_800, 5_400, 6_000, 6_600, 7_200, 7_800],
            ),
        );
        let second = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &wider,
        )
        .unwrap();
        assert_eq!(first.receipt.robot_design_id, second.receipt.robot_design_id);
        assert_eq!(first.receipt.parameter_set_id, second.receipt.parameter_set_id);
        assert_ne!(first.receipt.search_domain_id, second.receipt.search_domain_id);
    }

    #[test]
    fn selected_value_outside_domain_rejects() {
        let narrow = C0CouponSearchDomainV1::new(
            domain_for(C0_PARAMETER_SECTION_WIDTH, &[16_000, 18_000]),
            domain_for(C0_PARAMETER_SECTION_HEIGHT, &[5_400, 6_000]),
        );
        assert_eq!(
            compile_c0_joint_link_coupon(&reference_template(), &reference_parameters(), &narrow)
                .unwrap_err(),
            C0CouponError::SelectedValueOutsideSearchDomain
        );
    }

    #[test]
    fn extra_parameter_rejects_strict_c0_profile() {
        let with_extra = ExactDesignParameterSetV1::new(vec![
            parameter(C0_PARAMETER_LINK_LENGTH, 300_000),
            parameter(C0_PARAMETER_SECTION_WIDTH, 20_000),
            parameter(C0_PARAMETER_SECTION_HEIGHT, 6_000),
            parameter("bonus_dimension", 1_000),
        ]);
        assert!(matches!(
            compile_c0_joint_link_coupon(
                &reference_template(),
                &with_extra,
                &reference_search_domain(),
            ),
            Err(C0CouponError::UnexpectedParameterId(_))
        ));
    }

    #[test]
    fn zero_dimension_rejects_even_if_exact_primitive_can_represent_zero() {
        let zero_height = ExactDesignParameterSetV1::new(vec![
            parameter(C0_PARAMETER_LINK_LENGTH, 300_000),
            parameter(C0_PARAMETER_SECTION_WIDTH, 20_000),
            parameter(C0_PARAMETER_SECTION_HEIGHT, 0),
        ]);
        assert_eq!(
            compile_c0_joint_link_coupon(
                &reference_template(),
                &zero_height,
                &reference_search_domain(),
            )
            .unwrap_err(),
            C0CouponError::NonPositiveParameter(C0_PARAMETER_SECTION_HEIGHT)
        );
    }

    #[test]
    fn wrong_domain_parameter_binding_rejects() {
        let wrong = C0CouponSearchDomainV1::new(
            domain_for(C0_PARAMETER_SECTION_HEIGHT, &[20_000]),
            domain_for(C0_PARAMETER_SECTION_HEIGHT, &[6_000]),
        );
        assert!(matches!(
            wrong.validate(),
            Err(C0CouponError::DomainParameterMismatch { .. })
        ));
    }

    #[test]
    fn changed_orientation_changes_geometry_and_robot_design_identity() {
        let first = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        let mut rotated = reference_template();
        rotated.orientation = C0SectionOrientationV1::LongitudinalXWidthZHeightY;
        let second = compile_c0_joint_link_coupon(
            &rotated,
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        assert_ne!(first.receipt.geometry_intent_id, second.receipt.geometry_intent_id);
        assert_ne!(first.receipt.robot_design_id, second.receipt.robot_design_id);
    }

    #[test]
    fn display_only_labels_do_not_change_design_identity() {
        let compiled = compile_c0_joint_link_coupon(
            &reference_template(),
            &reference_parameters(),
            &reference_search_domain(),
        )
        .unwrap();
        let expected = compiled.receipt.robot_design_id;
        let mut relabeled = compiled.subject;
        relabeled.display_name = Some("C0 reference coupon".to_string());
        relabeled.morphology.components[0].display_name = Some("Coupon".to_string());
        assert_eq!(relabeled.design_id().unwrap(), expected);
    }

    #[test]
    fn link_length_is_template_frozen() {
        let changed = ExactDesignParameterSetV1::new(vec![
            parameter(C0_PARAMETER_LINK_LENGTH, 299_999),
            parameter(C0_PARAMETER_SECTION_WIDTH, 20_000),
            parameter(C0_PARAMETER_SECTION_HEIGHT, 6_000),
        ]);
        assert!(matches!(
            compile_c0_joint_link_coupon(
                &reference_template(),
                &changed,
                &reference_search_domain(),
            ),
            Err(C0CouponError::LinkLengthMismatch { .. })
        ));
    }
}
