// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Geometry-only analytical applicability screening for C0.
//!
//! Passing this screen does not establish physical beam validity, lateral
//! stability, elastic response, fixture ideality, safety, or prediction accuracy.

use crate::c0_normalized::{
    C0NormalizedError, C0NormalizedSectionV1, C0PositiveRationalV1,
};
use crate::exact_parameters::ExactDesignLengthUmV1;
use crate::{ContentDigest, RobotDesignId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_APPLICABILITY_SCHEMA_ID: &str = "symthaea.robot-design.c0-applicability.v1";
pub const C0_APPLICABILITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0ApplicabilityProfileId(ContentDigest);

impl C0ApplicabilityProfileId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0ApplicabilityAssessmentId(ContentDigest);

impl C0ApplicabilityAssessmentId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Explicit geometry-screen policy.
///
/// `max_height_to_width_ratio` is a bounded campaign admission rule only. It is
/// not a lateral-torsional buckling or general stability theorem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0ApplicabilityProfileV1 {
    pub max_conservative_shear_to_bending_ratio: C0PositiveRationalV1,
    pub max_height_to_width_ratio: C0PositiveRationalV1,
}

impl C0ApplicabilityProfileV1 {
    pub fn profile_id(self) -> C0ApplicabilityProfileId {
        let mut out = Vec::new();
        put_str(&mut out, C0_APPLICABILITY_SCHEMA_ID);
        put_u32(&mut out, C0_APPLICABILITY_SCHEMA_VERSION);
        put_rational(
            &mut out,
            self.max_conservative_shear_to_bending_ratio,
        );
        put_rational(&mut out, self.max_height_to_width_ratio);
        let digest: [u8; 32] = Sha256::digest(out).into();
        C0ApplicabilityProfileId(ContentDigest::from_bytes(digest))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0ApplicabilityDispositionV1 {
    AdmittedGeometryScreen,
    RejectedConservativeShearBound,
    RejectedSectionAspectProfile,
    RejectedMultipleGeometryScreens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0ApplicabilityAssessmentV1 {
    pub robot_design_id: RobotDesignId,
    pub support_span: ExactDesignLengthUmV1,
    pub profile_id: C0ApplicabilityProfileId,
    /// Conservative isotropic upper bound `3.6 * (h/L)^2`, represented exactly
    /// as `18 h^2 / (5 L^2)`.
    pub conservative_shear_to_bending_upper_bound: C0PositiveRationalV1,
    pub height_to_width_ratio: C0PositiveRationalV1,
    pub disposition: C0ApplicabilityDispositionV1,
}

impl C0ApplicabilityAssessmentV1 {
    pub fn assessment_id(self) -> C0ApplicabilityAssessmentId {
        let mut out = Vec::new();
        put_str(&mut out, C0_APPLICABILITY_SCHEMA_ID);
        put_u32(&mut out, C0_APPLICABILITY_SCHEMA_VERSION);
        put_digest(&mut out, self.robot_design_id.digest());
        put_u64(&mut out, self.support_span.as_um());
        put_digest(&mut out, self.profile_id.digest());
        put_rational(
            &mut out,
            self.conservative_shear_to_bending_upper_bound,
        );
        put_rational(&mut out, self.height_to_width_ratio);
        put_u8(&mut out, disposition_tag(self.disposition));
        let digest: [u8; 32] = Sha256::digest(out).into();
        C0ApplicabilityAssessmentId(ContentDigest::from_bytes(digest))
    }
}

pub fn assess_c0_geometry_applicability(
    section: C0NormalizedSectionV1,
    support_span: ExactDesignLengthUmV1,
    profile: C0ApplicabilityProfileV1,
) -> Result<C0ApplicabilityAssessmentV1, C0ApplicabilityError> {
    let span = positive_u128(support_span, "support_span")?;
    let width = positive_u128(section.width, "width")?;
    let height = positive_u128(section.height, "height")?;
    if support_span > section.link_length {
        return Err(C0ApplicabilityError::SupportSpanExceedsArticleLength {
            support_span_um: support_span.as_um(),
            article_length_um: section.link_length.as_um(),
        });
    }

    let h_squared = height
        .checked_mul(height)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow("height squared"))?;
    let l_squared = span
        .checked_mul(span)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow("span squared"))?;
    let shear_numerator = 18_u128
        .checked_mul(h_squared)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow(
            "conservative shear numerator",
        ))?;
    let shear_denominator = 5_u128
        .checked_mul(l_squared)
        .ok_or(C0ApplicabilityError::ArithmeticOverflow(
            "conservative shear denominator",
        ))?;

    let shear_upper = C0PositiveRationalV1::new(shear_numerator, shear_denominator)?;
    let aspect = C0PositiveRationalV1::new(height, width)?;

    let shear_ok = shear_upper
        .checked_cmp(profile.max_conservative_shear_to_bending_ratio)?
        != Ordering::Greater;
    let aspect_ok = aspect.checked_cmp(profile.max_height_to_width_ratio)? != Ordering::Greater;

    let disposition = match (shear_ok, aspect_ok) {
        (true, true) => C0ApplicabilityDispositionV1::AdmittedGeometryScreen,
        (false, true) => C0ApplicabilityDispositionV1::RejectedConservativeShearBound,
        (true, false) => C0ApplicabilityDispositionV1::RejectedSectionAspectProfile,
        (false, false) => C0ApplicabilityDispositionV1::RejectedMultipleGeometryScreens,
    };

    Ok(C0ApplicabilityAssessmentV1 {
        robot_design_id: section.robot_design_id,
        support_span,
        profile_id: profile.profile_id(),
        conservative_shear_to_bending_upper_bound: shear_upper,
        height_to_width_ratio: aspect,
        disposition,
    })
}

fn positive_u128(
    value: ExactDesignLengthUmV1,
    field: &'static str,
) -> Result<u128, C0ApplicabilityError> {
    if value.as_um() == 0 {
        Err(C0ApplicabilityError::NonPositiveDimension(field))
    } else {
        Ok(u128::from(value.as_um()))
    }
}

fn disposition_tag(disposition: C0ApplicabilityDispositionV1) -> u8 {
    match disposition {
        C0ApplicabilityDispositionV1::AdmittedGeometryScreen => 0,
        C0ApplicabilityDispositionV1::RejectedConservativeShearBound => 1,
        C0ApplicabilityDispositionV1::RejectedSectionAspectProfile => 2,
        C0ApplicabilityDispositionV1::RejectedMultipleGeometryScreens => 3,
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

fn put_u128(out: &mut Vec<u8>, value: u128) {
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

fn put_rational(out: &mut Vec<u8>, value: C0PositiveRationalV1) {
    put_u128(out, value.numerator());
    put_u128(out, value.denominator());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C0ApplicabilityError {
    Normalized(C0NormalizedError),
    NonPositiveDimension(&'static str),
    ArithmeticOverflow(&'static str),
    SupportSpanExceedsArticleLength {
        support_span_um: u64,
        article_length_um: u64,
    },
}

impl fmt::Display for C0ApplicabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normalized(error) => write!(formatter, "C0 applicability rational error: {error}"),
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 applicability dimension must be positive: {field}")
            }
            Self::ArithmeticOverflow(context) => {
                write!(formatter, "C0 applicability arithmetic overflow: {context}")
            }
            Self::SupportSpanExceedsArticleLength {
                support_span_um,
                article_length_um,
            } => write!(
                formatter,
                "C0 support span {support_span_um} um exceeds article length {article_length_um} um"
            ),
        }
    }
}

impl std::error::Error for C0ApplicabilityError {}

impl From<C0NormalizedError> for C0ApplicabilityError {
    fn from(value: C0NormalizedError) -> Self {
        Self::Normalized(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c0_joint_link_coupon::{
        C0CouponSearchDomainV1, C0JointLinkCouponTemplateV1, C0SectionOrientationV1,
        compile_c0_joint_link_coupon,
    };
    use crate::c0_normalized::C0NormalizedSectionV1;
    use crate::exact_parameters::{
        DesignParameterId, ExactDesignParameterV1, ExactDesignParameterSetV1,
        ExactLengthDomainKindV1, ExactLengthDomainV1,
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

    fn domain(id: &str, values: &[u64]) -> ExactLengthDomainV1 {
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

    fn section(width_um: u64, height_um: u64) -> C0NormalizedSectionV1 {
        let template = C0JointLinkCouponTemplateV1::new(
            ExactDesignLengthUmV1::from_um(300_000),
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
            digest(1),
            digest(2),
            digest(3),
        );
        let parameters = ExactDesignParameterSetV1::new(vec![
            parameter("link_length", 300_000),
            parameter("section_width", width_um),
            parameter("section_height", height_um),
        ]);
        let search = C0CouponSearchDomainV1::new(
            domain("section_width", &[10_000, 16_000, 20_000, 40_000]),
            domain("section_height", &[6_000, 7_200, 25_000, 30_000]),
        );
        let compiled = compile_c0_joint_link_coupon(&template, &parameters, &search).unwrap();
        C0NormalizedSectionV1::from_compiled(&compiled).unwrap()
    }

    fn profile() -> C0ApplicabilityProfileV1 {
        C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: C0PositiveRationalV1::new(1, 100).unwrap(),
            max_height_to_width_ratio: C0PositiveRationalV1::new(2, 1).unwrap(),
        }
    }

    #[test]
    fn baseline_has_exact_conservative_shear_upper_bound() {
        let assessment = assess_c0_geometry_applicability(
            section(20_000, 6_000),
            ExactDesignLengthUmV1::from_um(300_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.conservative_shear_to_bending_upper_bound,
            C0PositiveRationalV1::new(18, 12_500).unwrap()
        );
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
    }

    #[test]
    fn selected_reference_candidate_remains_admitted() {
        let assessment = assess_c0_geometry_applicability(
            section(16_000, 7_200),
            ExactDesignLengthUmV1::from_um(300_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
    }

    #[test]
    fn thick_section_can_fail_shear_budget() {
        let assessment = assess_c0_geometry_applicability(
            section(40_000, 30_000),
            ExactDesignLengthUmV1::from_um(300_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::RejectedConservativeShearBound
        );
    }

    #[test]
    fn tall_narrow_section_can_fail_aspect_profile() {
        let assessment = assess_c0_geometry_applicability(
            section(10_000, 25_000),
            ExactDesignLengthUmV1::from_um(300_000),
            profile(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            C0ApplicabilityDispositionV1::RejectedSectionAspectProfile
        );
    }

    #[test]
    fn support_span_longer_than_article_rejects() {
        assert!(matches!(
            assess_c0_geometry_applicability(
                section(20_000, 6_000),
                ExactDesignLengthUmV1::from_um(300_001),
                profile(),
            ),
            Err(C0ApplicabilityError::SupportSpanExceedsArticleLength { .. })
        ));
    }

    #[test]
    fn changing_policy_changes_profile_and_assessment_identity() {
        let section = section(20_000, 6_000);
        let first_profile = profile();
        let second_profile = C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: C0PositiveRationalV1::new(1, 200).unwrap(),
            ..first_profile
        };
        let first = assess_c0_geometry_applicability(
            section,
            ExactDesignLengthUmV1::from_um(300_000),
            first_profile,
        )
        .unwrap();
        let second = assess_c0_geometry_applicability(
            section,
            ExactDesignLengthUmV1::from_um(300_000),
            second_profile,
        )
        .unwrap();
        assert_ne!(first.profile_id, second.profile_id);
        assert_ne!(first.assessment_id(), second.assessment_id());
    }
}
