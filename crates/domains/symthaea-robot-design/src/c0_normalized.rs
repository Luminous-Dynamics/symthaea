// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact normalized mechanics for the bounded C0 JointLinkCoupon profile.
//!
//! This module evaluates dimensionless mass/deflection/stress ratios from exact
//! C0A design subjects. It does not establish absolute material properties,
//! physical fixture ideality, strength, safety, or physical improvement.

use crate::c0_joint_link_coupon::{
    C0CompiledCouponV1, C0CouponError, C0SectionOrientationV1, validate_strict_c0_subject,
};
use crate::exact_parameters::{ExactDesignLengthUmV1, ExactDesignParameterSetId};
use crate::{ContentDigest, RobotDesignError, RobotDesignId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_NORMALIZED_SCHEMA_ID: &str = "symthaea.robot-design.c0-normalized.v1";
pub const C0_NORMALIZED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0PositiveRationalV1 {
    numerator: u128,
    denominator: u128,
}

impl C0PositiveRationalV1 {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, C0NormalizedError> {
        if numerator == 0 {
            return Err(C0NormalizedError::ZeroRationalPart("numerator"));
        }
        if denominator == 0 {
            return Err(C0NormalizedError::ZeroRationalPart("denominator"));
        }
        let divisor = gcd_u128(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, C0NormalizedError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(C0NormalizedError::ArithmeticOverflow(
                "rational comparison left",
            ))?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or(C0NormalizedError::ArithmeticOverflow(
                "rational comparison right",
            ))?;
        Ok(left.cmp(&right))
    }

    /// Presentation only. Floating-point conversion never participates in
    /// canonical identity or exact feasibility decisions.
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

impl fmt::Display for C0PositiveRationalV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0MatchedInvariantProfileId(ContentDigest);

impl C0MatchedInvariantProfileId {
    pub const fn from_digest(digest: ContentDigest) -> Self {
        Self(digest)
    }

    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0NormalizedEvaluationId(ContentDigest);

impl C0NormalizedEvaluationId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Exact design fields needed by the normalized rectangular-section theorem.
/// Construct this through `from_compiled` so dimensions remain bound to an
/// exact C0 RobotDesignId rather than becoming anonymous numeric inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0NormalizedSectionV1 {
    pub robot_design_id: RobotDesignId,
    pub parameter_set_id: ExactDesignParameterSetId,
    pub requirement_snapshot: ContentDigest,
    pub material_subject: ContentDigest,
    pub link_length: ExactDesignLengthUmV1,
    pub width: ExactDesignLengthUmV1,
    pub height: ExactDesignLengthUmV1,
    pub orientation: C0SectionOrientationV1,
}

impl C0NormalizedSectionV1 {
    pub fn from_compiled(compiled: &C0CompiledCouponV1) -> Result<Self, C0NormalizedError> {
        validate_strict_c0_subject(&compiled.subject)?;
        let actual_design_id = compiled.subject.design_id()?;
        if actual_design_id != compiled.receipt.robot_design_id {
            return Err(C0NormalizedError::RobotDesignIdentityMismatch);
        }
        let actual_geometry_id = compiled.geometry_intent.geometry_intent_id()?;
        if actual_geometry_id != compiled.receipt.geometry_intent_id
            || compiled.subject.geometry[0].artifact_digest != actual_geometry_id.digest()
        {
            return Err(C0NormalizedError::GeometryIntentIdentityMismatch);
        }
        if compiled.subject.parameter_set != compiled.receipt.parameter_set_id.digest() {
            return Err(C0NormalizedError::ParameterSetIdentityMismatch);
        }
        let material_subject = compiled.subject.material_assignments[0].material_digest;
        Ok(Self {
            robot_design_id: actual_design_id,
            parameter_set_id: compiled.receipt.parameter_set_id,
            requirement_snapshot: compiled.subject.requirement_snapshot,
            material_subject,
            link_length: compiled.geometry_intent.link_length,
            width: compiled.geometry_intent.section_width,
            height: compiled.geometry_intent.section_height,
            orientation: compiled.geometry_intent.orientation,
        })
    }

    fn validate_positive(&self) -> Result<(), C0NormalizedError> {
        if self.link_length.as_um() == 0 {
            return Err(C0NormalizedError::NonPositiveDimension("link_length"));
        }
        if self.width.as_um() == 0 {
            return Err(C0NormalizedError::NonPositiveDimension("width"));
        }
        if self.height.as_um() == 0 {
            return Err(C0NormalizedError::NonPositiveDimension("height"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0NormalizedEvaluationV1 {
    pub baseline_design_id: RobotDesignId,
    pub candidate_design_id: RobotDesignId,
    pub baseline_parameter_set_id: ExactDesignParameterSetId,
    pub candidate_parameter_set_id: ExactDesignParameterSetId,
    pub matched_invariant_profile_id: C0MatchedInvariantProfileId,
    pub mass_ratio: C0PositiveRationalV1,
    pub deflection_ratio: C0PositiveRationalV1,
    pub stress_ratio: C0PositiveRationalV1,
}

impl C0NormalizedEvaluationV1 {
    pub fn evaluation_id(self) -> C0NormalizedEvaluationId {
        let mut out = Vec::new();
        put_str(&mut out, C0_NORMALIZED_SCHEMA_ID);
        put_u32(&mut out, C0_NORMALIZED_SCHEMA_VERSION);
        put_digest(&mut out, self.baseline_design_id.digest());
        put_digest(&mut out, self.candidate_design_id.digest());
        put_digest(&mut out, self.baseline_parameter_set_id.digest());
        put_digest(&mut out, self.candidate_parameter_set_id.digest());
        put_digest(&mut out, self.matched_invariant_profile_id.digest());
        put_rational(&mut out, self.mass_ratio);
        put_rational(&mut out, self.deflection_ratio);
        put_rational(&mut out, self.stress_ratio);
        let digest: [u8; 32] = Sha256::digest(out).into();
        C0NormalizedEvaluationId(ContentDigest::from_bytes(digest))
    }
}

pub fn evaluate_c0_normalized(
    baseline: C0NormalizedSectionV1,
    candidate: C0NormalizedSectionV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
) -> Result<C0NormalizedEvaluationV1, C0NormalizedError> {
    baseline.validate_positive()?;
    candidate.validate_positive()?;
    validate_matched_design_invariants(baseline, candidate)?;

    let b0 = u128::from(baseline.width.as_um());
    let h0 = u128::from(baseline.height.as_um());
    let b = u128::from(candidate.width.as_um());
    let h = u128::from(candidate.height.as_um());

    let mass_ratio = C0PositiveRationalV1::new(
        checked_mul(b, h, "candidate area")?,
        checked_mul(b0, h0, "baseline area")?,
    )?;

    let deflection_ratio = C0PositiveRationalV1::new(
        checked_mul(
            b0,
            checked_pow(h0, 3, "baseline h^3")?,
            "baseline b*h^3",
        )?,
        checked_mul(
            b,
            checked_pow(h, 3, "candidate h^3")?,
            "candidate b*h^3",
        )?,
    )?;

    let stress_ratio = C0PositiveRationalV1::new(
        checked_mul(
            b0,
            checked_pow(h0, 2, "baseline h^2")?,
            "baseline b*h^2",
        )?,
        checked_mul(
            b,
            checked_pow(h, 2, "candidate h^2")?,
            "candidate b*h^2",
        )?,
    )?;

    Ok(C0NormalizedEvaluationV1 {
        baseline_design_id: baseline.robot_design_id,
        candidate_design_id: candidate.robot_design_id,
        baseline_parameter_set_id: baseline.parameter_set_id,
        candidate_parameter_set_id: candidate.parameter_set_id,
        matched_invariant_profile_id,
        mass_ratio,
        deflection_ratio,
        stress_ratio,
    })
}

fn validate_matched_design_invariants(
    baseline: C0NormalizedSectionV1,
    candidate: C0NormalizedSectionV1,
) -> Result<(), C0NormalizedError> {
    if baseline.link_length != candidate.link_length {
        return Err(C0NormalizedError::MatchedInvariantMismatch("link_length"));
    }
    if baseline.orientation != candidate.orientation {
        return Err(C0NormalizedError::MatchedInvariantMismatch("orientation"));
    }
    if baseline.material_subject != candidate.material_subject {
        return Err(C0NormalizedError::MatchedInvariantMismatch("material_subject"));
    }
    if baseline.requirement_snapshot != candidate.requirement_snapshot {
        return Err(C0NormalizedError::MatchedInvariantMismatch(
            "requirement_snapshot",
        ));
    }
    Ok(())
}

fn checked_mul(
    left: u128,
    right: u128,
    context: &'static str,
) -> Result<u128, C0NormalizedError> {
    left.checked_mul(right)
        .ok_or(C0NormalizedError::ArithmeticOverflow(context))
}

fn checked_pow(
    mut base: u128,
    mut exponent: u32,
    context: &'static str,
) -> Result<u128, C0NormalizedError> {
    let mut result = 1_u128;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result
                .checked_mul(base)
                .ok_or(C0NormalizedError::ArithmeticOverflow(context))?;
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base
                .checked_mul(base)
                .ok_or(C0NormalizedError::ArithmeticOverflow(context))?;
        }
    }
    Ok(result)
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
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
pub enum C0NormalizedError {
    C0Design(C0CouponError),
    RobotDesign(RobotDesignError),
    RobotDesignIdentityMismatch,
    GeometryIntentIdentityMismatch,
    ParameterSetIdentityMismatch,
    NonPositiveDimension(&'static str),
    MatchedInvariantMismatch(&'static str),
    ZeroRationalPart(&'static str),
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for C0NormalizedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::C0Design(error) => write!(formatter, "C0 design error: {error}"),
            Self::RobotDesign(error) => write!(formatter, "robot-design error: {error}"),
            Self::RobotDesignIdentityMismatch => {
                formatter.write_str("C0 normalized input RobotDesignId does not match subject")
            }
            Self::GeometryIntentIdentityMismatch => formatter
                .write_str("C0 normalized input geometry-intent identity does not match subject"),
            Self::ParameterSetIdentityMismatch => formatter
                .write_str("C0 normalized input parameter-set identity does not match subject"),
            Self::NonPositiveDimension(field) => {
                write!(formatter, "C0 dimension must be positive: {field}")
            }
            Self::MatchedInvariantMismatch(field) => {
                write!(formatter, "C0 matched design invariant differs: {field}")
            }
            Self::ZeroRationalPart(field) => {
                write!(formatter, "positive rational {field} must be non-zero")
            }
            Self::ArithmeticOverflow(context) => {
                write!(formatter, "C0 exact arithmetic overflow: {context}")
            }
        }
    }
}

impl std::error::Error for C0NormalizedError {}

impl From<C0CouponError> for C0NormalizedError {
    fn from(value: C0CouponError) -> Self {
        Self::C0Design(value)
    }
}

impl From<RobotDesignError> for C0NormalizedError {
    fn from(value: RobotDesignError) -> Self {
        Self::RobotDesign(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c0_joint_link_coupon::{
        C0CouponSearchDomainV1, C0JointLinkCouponTemplateV1,
        compile_c0_joint_link_coupon,
    };
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

    fn compiled(
        width_um: u64,
        height_um: u64,
        material_seed: u8,
        requirement_seed: u8,
        orientation: C0SectionOrientationV1,
    ) -> C0CompiledCouponV1 {
        let template = C0JointLinkCouponTemplateV1::new(
            ExactDesignLengthUmV1::from_um(300_000),
            orientation,
            digest(requirement_seed),
            digest(material_seed),
            digest(3),
        );
        let parameters = ExactDesignParameterSetV1::new(vec![
            parameter("link_length", 300_000),
            parameter("section_width", width_um),
            parameter("section_height", height_um),
        ]);
        let search = C0CouponSearchDomainV1::new(
            domain("section_width", &[16_000, 18_000, 20_000, 22_000, 24_000]),
            domain("section_height", &[4_800, 5_400, 6_000, 6_600, 7_200]),
        );
        compile_c0_joint_link_coupon(&template, &parameters, &search).unwrap()
    }

    fn section(width_um: u64, height_um: u64) -> C0NormalizedSectionV1 {
        C0NormalizedSectionV1::from_compiled(&compiled(
            width_um,
            height_um,
            2,
            1,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap()
    }

    fn profile() -> C0MatchedInvariantProfileId {
        C0MatchedInvariantProfileId::from_digest(digest(9))
    }

    #[test]
    fn baseline_against_itself_is_exactly_one() {
        let baseline = section(20_000, 6_000);
        let result = evaluate_c0_normalized(baseline, baseline, profile()).unwrap();
        let one = C0PositiveRationalV1::new(1, 1).unwrap();
        assert_eq!(result.mass_ratio, one);
        assert_eq!(result.deflection_ratio, one);
        assert_eq!(result.stress_ratio, one);
    }

    #[test]
    fn known_08_width_12_height_candidate_matches_oracle() {
        let result = evaluate_c0_normalized(section(20_000, 6_000), section(16_000, 7_200), profile())
            .unwrap();
        assert_eq!(
            result.mass_ratio,
            C0PositiveRationalV1::new(24, 25).unwrap()
        );
        assert_eq!(
            result.deflection_ratio,
            C0PositiveRationalV1::new(625, 864).unwrap()
        );
        assert_eq!(
            result.stress_ratio,
            C0PositiveRationalV1::new(125, 144).unwrap()
        );
    }

    #[test]
    fn rational_inputs_reduce_canonically() {
        assert_eq!(
            C0PositiveRationalV1::new(48, 50).unwrap(),
            C0PositiveRationalV1::new(24, 25).unwrap()
        );
    }

    #[test]
    fn exact_comparison_does_not_use_float_rounding() {
        let less = C0PositiveRationalV1::new(24, 25).unwrap();
        let one = C0PositiveRationalV1::new(1, 1).unwrap();
        assert_eq!(less.checked_cmp(one).unwrap(), Ordering::Less);
    }

    #[test]
    fn material_mismatch_rejects_normalized_comparison() {
        let baseline = C0NormalizedSectionV1::from_compiled(&compiled(
            20_000,
            6_000,
            2,
            1,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap();
        let candidate = C0NormalizedSectionV1::from_compiled(&compiled(
            16_000,
            7_200,
            7,
            1,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap();
        assert_eq!(
            evaluate_c0_normalized(baseline, candidate, profile()).unwrap_err(),
            C0NormalizedError::MatchedInvariantMismatch("material_subject")
        );
    }

    #[test]
    fn requirement_mismatch_rejects_normalized_comparison() {
        let baseline = C0NormalizedSectionV1::from_compiled(&compiled(
            20_000,
            6_000,
            2,
            1,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap();
        let candidate = C0NormalizedSectionV1::from_compiled(&compiled(
            16_000,
            7_200,
            2,
            8,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap();
        assert_eq!(
            evaluate_c0_normalized(baseline, candidate, profile()).unwrap_err(),
            C0NormalizedError::MatchedInvariantMismatch("requirement_snapshot")
        );
    }

    #[test]
    fn orientation_mismatch_rejects_normalized_comparison() {
        let baseline = section(20_000, 6_000);
        let candidate = C0NormalizedSectionV1::from_compiled(&compiled(
            16_000,
            7_200,
            2,
            1,
            C0SectionOrientationV1::LongitudinalXWidthZHeightY,
        ))
        .unwrap();
        assert_eq!(
            evaluate_c0_normalized(baseline, candidate, profile()).unwrap_err(),
            C0NormalizedError::MatchedInvariantMismatch("orientation")
        );
    }

    #[test]
    fn evaluation_identity_binds_exact_design_subjects_and_profile() {
        let first = evaluate_c0_normalized(section(20_000, 6_000), section(16_000, 7_200), profile())
            .unwrap();
        let changed_candidate = evaluate_c0_normalized(
            section(20_000, 6_000),
            section(18_000, 7_200),
            profile(),
        )
        .unwrap();
        let changed_profile = evaluate_c0_normalized(
            section(20_000, 6_000),
            section(16_000, 7_200),
            C0MatchedInvariantProfileId::from_digest(digest(10)),
        )
        .unwrap();
        assert_ne!(first.evaluation_id(), changed_candidate.evaluation_id());
        assert_ne!(first.evaluation_id(), changed_profile.evaluation_id());
    }

    #[test]
    fn tampered_compiled_subject_rejects_before_ratio_evaluation() {
        let mut source = compiled(
            20_000,
            6_000,
            2,
            1,
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        );
        source.subject.requirement_snapshot = digest(55);
        assert_eq!(
            C0NormalizedSectionV1::from_compiled(&source).unwrap_err(),
            C0NormalizedError::RobotDesignIdentityMismatch
        );
    }
}
