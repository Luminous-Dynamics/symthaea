// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic exhaustive C0 candidate enumeration and selection.
//!
//! C0D intentionally uses complete bounded enumeration rather than a stochastic
//! optimizer. It establishes reproducible candidate evaluation/selection only;
//! it does not establish optimizer superiority or physical improvement.

use crate::c0_applicability::{
    C0ApplicabilityAssessmentId, C0ApplicabilityDispositionV1, C0ApplicabilityError,
    C0ApplicabilityProfileId, C0ApplicabilityProfileV1, assess_c0_geometry_applicability,
};
use crate::c0_joint_link_coupon::{
    C0CompiledCouponV1, C0CouponError, C0CouponSearchDomainId, C0CouponSearchDomainV1,
    C0JointLinkCouponTemplateV1, C0_PARAMETER_LINK_LENGTH, C0_PARAMETER_SECTION_HEIGHT,
    C0_PARAMETER_SECTION_WIDTH, compile_c0_joint_link_coupon,
};
use crate::c0_normalized::{
    C0MatchedInvariantProfileId, C0NormalizedError, C0NormalizedEvaluationId,
    C0NormalizedEvaluationV1, C0NormalizedSectionV1, C0PositiveRationalV1,
    evaluate_c0_normalized,
};
use crate::exact_parameters::{
    DesignParameterId, ExactDesignLengthUmV1, ExactDesignParameterV1,
    ExactDesignParameterSetV1, ExactParameterError,
};
use crate::{ContentDigest, RobotDesignId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;

pub const C0_ENUMERATION_SCHEMA_ID: &str = "symthaea.robot-design.c0-exhaustive.v1";
pub const C0_ENUMERATION_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_C0_CANDIDATE_BUDGET: u64 = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0CandidateEvaluationId(ContentDigest);

impl C0CandidateEvaluationId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0SelectionPolicyId(ContentDigest);

impl C0SelectionPolicyId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0ExhaustiveReceiptId(ContentDigest);

impl C0ExhaustiveReceiptId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0SelectionPolicyV1 {
    pub max_deflection_ratio: C0PositiveRationalV1,
    pub max_stress_ratio: C0PositiveRationalV1,
    pub require_applicability_admitted: bool,
}

impl C0SelectionPolicyV1 {
    pub fn policy_id(self) -> C0SelectionPolicyId {
        let mut out = Vec::new();
        put_str(&mut out, C0_ENUMERATION_SCHEMA_ID);
        put_u32(&mut out, C0_ENUMERATION_SCHEMA_VERSION);
        put_u8(&mut out, 1); // selection-policy subdomain
        put_rational(&mut out, self.max_deflection_ratio);
        put_rational(&mut out, self.max_stress_ratio);
        put_u8(&mut out, u8::from(self.require_applicability_admitted));
        C0SelectionPolicyId(hash_bytes(out))
    }

    fn admits(self, candidate: &C0CandidateEvaluationV1) -> Result<bool, C0EnumerationError> {
        let deflection_ok = candidate
            .normalized
            .deflection_ratio
            .checked_cmp(self.max_deflection_ratio)?
            != Ordering::Greater;
        let stress_ok = candidate
            .normalized
            .stress_ratio
            .checked_cmp(self.max_stress_ratio)?
            != Ordering::Greater;
        let applicability_ok = !self.require_applicability_admitted
            || candidate.applicability_disposition
                == C0ApplicabilityDispositionV1::AdmittedGeometryScreen;
        Ok(deflection_ok && stress_ok && applicability_ok)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0CandidateEvaluationV1 {
    pub robot_design_id: RobotDesignId,
    pub width: ExactDesignLengthUmV1,
    pub height: ExactDesignLengthUmV1,
    pub normalized: C0NormalizedEvaluationV1,
    pub normalized_evaluation_id: C0NormalizedEvaluationId,
    pub applicability_assessment_id: C0ApplicabilityAssessmentId,
    pub applicability_disposition: C0ApplicabilityDispositionV1,
}

impl C0CandidateEvaluationV1 {
    pub fn candidate_evaluation_id(self) -> C0CandidateEvaluationId {
        let mut out = Vec::new();
        put_str(&mut out, C0_ENUMERATION_SCHEMA_ID);
        put_u32(&mut out, C0_ENUMERATION_SCHEMA_VERSION);
        put_u8(&mut out, 0); // candidate-evaluation subdomain
        put_digest(&mut out, self.robot_design_id.digest());
        put_u64(&mut out, self.width.as_um());
        put_u64(&mut out, self.height.as_um());
        put_digest(&mut out, self.normalized_evaluation_id.digest());
        put_digest(&mut out, self.applicability_assessment_id.digest());
        put_u8(&mut out, applicability_tag(self.applicability_disposition));
        C0CandidateEvaluationId(hash_bytes(out))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0SelectionRationaleV1 {
    MassThenDeflectionThenRobotDesignId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0ExhaustiveEvaluationReceiptV1 {
    pub baseline_design_id: RobotDesignId,
    pub search_domain_id: C0CouponSearchDomainId,
    pub matched_invariant_profile_id: C0MatchedInvariantProfileId,
    pub applicability_profile_id: C0ApplicabilityProfileId,
    pub selection_policy_id: C0SelectionPolicyId,
    pub candidate_budget: u64,
    pub candidate_count: u64,
    pub candidate_set_root: ContentDigest,
    pub pareto_count: u64,
    pub pareto_front_root: ContentDigest,
    pub feasible_count: u64,
    pub feasible_set_root: ContentDigest,
    pub selected_design_id: RobotDesignId,
    pub selection_rationale: C0SelectionRationaleV1,
}

impl C0ExhaustiveEvaluationReceiptV1 {
    pub fn receipt_id(&self) -> C0ExhaustiveReceiptId {
        let mut out = Vec::new();
        put_str(&mut out, C0_ENUMERATION_SCHEMA_ID);
        put_u32(&mut out, C0_ENUMERATION_SCHEMA_VERSION);
        put_u8(&mut out, 2); // campaign receipt subdomain
        put_digest(&mut out, self.baseline_design_id.digest());
        put_digest(&mut out, self.search_domain_id.digest());
        put_digest(&mut out, self.matched_invariant_profile_id.digest());
        put_digest(&mut out, self.applicability_profile_id.digest());
        put_digest(&mut out, self.selection_policy_id.digest());
        put_u64(&mut out, self.candidate_budget);
        put_u64(&mut out, self.candidate_count);
        put_digest(&mut out, self.candidate_set_root);
        put_u64(&mut out, self.pareto_count);
        put_digest(&mut out, self.pareto_front_root);
        put_u64(&mut out, self.feasible_count);
        put_digest(&mut out, self.feasible_set_root);
        put_digest(&mut out, self.selected_design_id.digest());
        put_u8(&mut out, 0);
        C0ExhaustiveReceiptId(hash_bytes(out))
    }
}

#[derive(Debug, Clone)]
pub struct C0ExhaustiveEvaluationV1 {
    pub candidates: Vec<C0CandidateEvaluationV1>,
    pub pareto_frontier: Vec<C0CandidateEvaluationV1>,
    pub feasible_candidates: Vec<C0CandidateEvaluationV1>,
    pub selected: C0CandidateEvaluationV1,
    pub receipt: C0ExhaustiveEvaluationReceiptV1,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_c0_domain_exhaustively(
    template: &C0JointLinkCouponTemplateV1,
    search_domain: &C0CouponSearchDomainV1,
    baseline: &C0CompiledCouponV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
    support_span: ExactDesignLengthUmV1,
    applicability_profile: C0ApplicabilityProfileV1,
    selection_policy: C0SelectionPolicyV1,
    candidate_budget: u64,
) -> Result<C0ExhaustiveEvaluationV1, C0EnumerationError> {
    if candidate_budget == 0 {
        return Err(C0EnumerationError::ZeroCandidateBudget);
    }
    search_domain.validate()?;
    let width_values = search_domain.width_domain.enumerate()?;
    let height_values = search_domain.height_domain.enumerate()?;
    let candidate_count_usize = width_values
        .len()
        .checked_mul(height_values.len())
        .ok_or(C0EnumerationError::CandidateCountOverflow)?;
    let candidate_count = u64::try_from(candidate_count_usize)
        .map_err(|_| C0EnumerationError::CandidateCountOverflow)?;
    if candidate_count > candidate_budget {
        return Err(C0EnumerationError::CandidateBudgetExceeded {
            candidate_count,
            candidate_budget,
        });
    }

    validate_baseline_under_campaign(template, search_domain, baseline)?;
    let baseline_section = C0NormalizedSectionV1::from_compiled(baseline)?;

    let mut candidates = Vec::with_capacity(candidate_count_usize);
    for width in width_values {
        for height in &height_values {
            let parameter_set = make_parameter_set(template.link_length, width, *height)?;
            let compiled = compile_c0_joint_link_coupon(template, &parameter_set, search_domain)?;
            let candidate_section = C0NormalizedSectionV1::from_compiled(&compiled)?;
            let normalized = evaluate_c0_normalized(
                baseline_section,
                candidate_section,
                matched_invariant_profile_id,
            )?;
            let applicability = assess_c0_geometry_applicability(
                candidate_section,
                support_span,
                applicability_profile,
            )?;
            candidates.push(C0CandidateEvaluationV1 {
                robot_design_id: compiled.receipt.robot_design_id,
                width,
                height: *height,
                normalized_evaluation_id: normalized.evaluation_id(),
                applicability_assessment_id: applicability.assessment_id(),
                applicability_disposition: applicability.disposition,
                normalized,
            });
        }
    }
    candidates.sort_by(candidate_identity_order);
    assert_unique_complete_set(&candidates, candidate_count)?;

    let mut pareto_frontier = c0_pareto_frontier(&candidates)?;
    pareto_frontier.sort_by(candidate_identity_order);

    let mut feasible_candidates = Vec::new();
    for candidate in &candidates {
        if selection_policy.admits(candidate)? {
            feasible_candidates.push(*candidate);
        }
    }
    if feasible_candidates.is_empty() {
        return Err(C0EnumerationError::NoFeasibleCandidate);
    }
    preflight_selection_sort(&feasible_candidates)?;
    feasible_candidates.sort_by(selection_order);
    let selected = feasible_candidates[0];

    let search_domain_id = search_domain.search_domain_id()?;
    let candidate_set_root = set_root(&candidates);
    let pareto_front_root = set_root(&pareto_frontier);
    let feasible_set_root = set_root(&feasible_candidates);

    Ok(C0ExhaustiveEvaluationV1 {
        receipt: C0ExhaustiveEvaluationReceiptV1 {
            baseline_design_id: baseline.receipt.robot_design_id,
            search_domain_id,
            matched_invariant_profile_id,
            applicability_profile_id: applicability_profile.profile_id(),
            selection_policy_id: selection_policy.policy_id(),
            candidate_budget,
            candidate_count,
            candidate_set_root,
            pareto_count: pareto_frontier.len() as u64,
            pareto_front_root,
            feasible_count: feasible_candidates.len() as u64,
            feasible_set_root,
            selected_design_id: selected.robot_design_id,
            selection_rationale: C0SelectionRationaleV1::MassThenDeflectionThenRobotDesignId,
        },
        candidates,
        pareto_frontier,
        feasible_candidates,
        selected,
    })
}

pub fn c0_pareto_frontier(
    candidates: &[C0CandidateEvaluationV1],
) -> Result<Vec<C0CandidateEvaluationV1>, C0EnumerationError> {
    let mut frontier = Vec::new();
    'candidate: for candidate in candidates {
        for other in candidates {
            if candidate.robot_design_id == other.robot_design_id {
                continue;
            }
            if dominates_mass_deflection(other, candidate)? {
                continue 'candidate;
            }
        }
        frontier.push(*candidate);
    }
    Ok(frontier)
}

fn validate_baseline_under_campaign(
    template: &C0JointLinkCouponTemplateV1,
    search_domain: &C0CouponSearchDomainV1,
    baseline: &C0CompiledCouponV1,
) -> Result<(), C0EnumerationError> {
    let reconstructed = make_parameter_set(
        baseline.geometry_intent.link_length,
        baseline.geometry_intent.section_width,
        baseline.geometry_intent.section_height,
    )?;
    let recompiled = compile_c0_joint_link_coupon(template, &reconstructed, search_domain)
        .map_err(|error| match error {
            C0CouponError::SelectedValueOutsideSearchDomain => C0EnumerationError::BaselineNotAdmitted,
            other => C0EnumerationError::C0Design(other),
        })?;
    if recompiled.receipt.robot_design_id != baseline.receipt.robot_design_id {
        return Err(C0EnumerationError::BaselineTemplateMismatch);
    }
    Ok(())
}

fn make_parameter_set(
    link_length: ExactDesignLengthUmV1,
    width: ExactDesignLengthUmV1,
    height: ExactDesignLengthUmV1,
) -> Result<ExactDesignParameterSetV1, C0EnumerationError> {
    Ok(ExactDesignParameterSetV1::new(vec![
        ExactDesignParameterV1::length(
            DesignParameterId::new(C0_PARAMETER_LINK_LENGTH)?,
            link_length,
        ),
        ExactDesignParameterV1::length(
            DesignParameterId::new(C0_PARAMETER_SECTION_WIDTH)?,
            width,
        ),
        ExactDesignParameterV1::length(
            DesignParameterId::new(C0_PARAMETER_SECTION_HEIGHT)?,
            height,
        ),
    ]))
}

fn assert_unique_complete_set(
    candidates: &[C0CandidateEvaluationV1],
    expected_count: u64,
) -> Result<(), C0EnumerationError> {
    if candidates.len() as u64 != expected_count {
        return Err(C0EnumerationError::IncompleteCandidateSet {
            expected: expected_count,
            actual: candidates.len() as u64,
        });
    }
    for pair in candidates.windows(2) {
        if pair[0].robot_design_id == pair[1].robot_design_id {
            return Err(C0EnumerationError::DuplicateCandidateIdentity(
                pair[0].robot_design_id,
            ));
        }
    }
    Ok(())
}

fn dominates_mass_deflection(
    left: &C0CandidateEvaluationV1,
    right: &C0CandidateEvaluationV1,
) -> Result<bool, C0EnumerationError> {
    let mass = left
        .normalized
        .mass_ratio
        .checked_cmp(right.normalized.mass_ratio)?;
    let deflection = left
        .normalized
        .deflection_ratio
        .checked_cmp(right.normalized.deflection_ratio)?;
    Ok(mass != Ordering::Greater
        && deflection != Ordering::Greater
        && (mass == Ordering::Less || deflection == Ordering::Less))
}

fn candidate_identity_order(
    left: &C0CandidateEvaluationV1,
    right: &C0CandidateEvaluationV1,
) -> Ordering {
    left.width
        .cmp(&right.width)
        .then_with(|| left.height.cmp(&right.height))
        .then_with(|| left.robot_design_id.cmp(&right.robot_design_id))
}

fn preflight_selection_sort(candidates: &[C0CandidateEvaluationV1]) -> Result<(), C0EnumerationError> {
    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            candidates[left]
                .normalized
                .mass_ratio
                .checked_cmp(candidates[right].normalized.mass_ratio)?;
            candidates[left]
                .normalized
                .deflection_ratio
                .checked_cmp(candidates[right].normalized.deflection_ratio)?;
        }
    }
    Ok(())
}

fn selection_order(
    left: &C0CandidateEvaluationV1,
    right: &C0CandidateEvaluationV1,
) -> Ordering {
    rational_cmp_preflighted(left.normalized.mass_ratio, right.normalized.mass_ratio)
        .then_with(|| {
            rational_cmp_preflighted(
                left.normalized.deflection_ratio,
                right.normalized.deflection_ratio,
            )
        })
        .then_with(|| left.robot_design_id.cmp(&right.robot_design_id))
}

fn rational_cmp_preflighted(
    left: C0PositiveRationalV1,
    right: C0PositiveRationalV1,
) -> Ordering {
    left.numerator()
        .checked_mul(right.denominator())
        .expect("C0 selection comparison was preflighted")
        .cmp(
            &right
                .numerator()
                .checked_mul(left.denominator())
                .expect("C0 selection comparison was preflighted"),
        )
}

fn set_root(candidates: &[C0CandidateEvaluationV1]) -> ContentDigest {
    let mut ids = candidates
        .iter()
        .copied()
        .map(C0CandidateEvaluationV1::candidate_evaluation_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    let mut out = Vec::new();
    put_str(&mut out, C0_ENUMERATION_SCHEMA_ID);
    put_u32(&mut out, C0_ENUMERATION_SCHEMA_VERSION);
    put_len(&mut out, ids.len());
    for id in ids {
        put_digest(&mut out, id.digest());
    }
    hash_bytes(out)
}

fn applicability_tag(disposition: C0ApplicabilityDispositionV1) -> u8 {
    match disposition {
        C0ApplicabilityDispositionV1::AdmittedGeometryScreen => 0,
        C0ApplicabilityDispositionV1::RejectedConservativeShearBound => 1,
        C0ApplicabilityDispositionV1::RejectedSectionAspectProfile => 2,
        C0ApplicabilityDispositionV1::RejectedMultipleGeometryScreens => 3,
    }
}

fn hash_bytes(bytes: Vec<u8>) -> ContentDigest {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    ContentDigest::from_bytes(digest)
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
pub enum C0EnumerationError {
    ExactParameters(ExactParameterError),
    C0Design(C0CouponError),
    Normalized(C0NormalizedError),
    Applicability(C0ApplicabilityError),
    ZeroCandidateBudget,
    CandidateCountOverflow,
    CandidateBudgetExceeded {
        candidate_count: u64,
        candidate_budget: u64,
    },
    BaselineNotAdmitted,
    BaselineTemplateMismatch,
    DuplicateCandidateIdentity(RobotDesignId),
    IncompleteCandidateSet {
        expected: u64,
        actual: u64,
    },
    NoFeasibleCandidate,
}

impl fmt::Display for C0EnumerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactParameters(error) => write!(formatter, "C0 exact-parameter error: {error}"),
            Self::C0Design(error) => write!(formatter, "C0 design error: {error}"),
            Self::Normalized(error) => write!(formatter, "C0 normalized-mechanics error: {error}"),
            Self::Applicability(error) => write!(formatter, "C0 applicability error: {error}"),
            Self::ZeroCandidateBudget => formatter.write_str("C0 candidate budget must be positive"),
            Self::CandidateCountOverflow => formatter.write_str("C0 Cartesian candidate count overflow"),
            Self::CandidateBudgetExceeded { candidate_count, candidate_budget } => write!(formatter, "C0 candidate count {candidate_count} exceeds budget {candidate_budget}"),
            Self::BaselineNotAdmitted => formatter.write_str("C0 baseline is not admitted by the frozen search domain"),
            Self::BaselineTemplateMismatch => formatter.write_str("C0 baseline does not match the frozen template/campaign semantics"),
            Self::DuplicateCandidateIdentity(id) => write!(formatter, "duplicate C0 RobotDesignId during exhaustive enumeration: {id}"),
            Self::IncompleteCandidateSet { expected, actual } => write!(formatter, "incomplete C0 candidate set: expected {expected}, got {actual}"),
            Self::NoFeasibleCandidate => formatter.write_str("no C0 candidate satisfies the frozen selection policy"),
        }
    }
}

impl std::error::Error for C0EnumerationError {}

impl From<ExactParameterError> for C0EnumerationError {
    fn from(value: ExactParameterError) -> Self {
        Self::ExactParameters(value)
    }
}

impl From<C0CouponError> for C0EnumerationError {
    fn from(value: C0CouponError) -> Self {
        Self::C0Design(value)
    }
}

impl From<C0NormalizedError> for C0EnumerationError {
    fn from(value: C0NormalizedError) -> Self {
        Self::Normalized(value)
    }
}

impl From<C0ApplicabilityError> for C0EnumerationError {
    fn from(value: C0ApplicabilityError) -> Self {
        Self::Applicability(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c0_joint_link_coupon::C0SectionOrientationV1;
    use crate::exact_parameters::{ExactLengthDomainKindV1, ExactLengthDomainV1};
    use std::collections::BTreeSet;

    fn digest(seed: u8) -> ContentDigest {
        ContentDigest::from_bytes([seed; 32])
    }

    fn explicit_domain(id: &str, values: &[u64]) -> ExactLengthDomainV1 {
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

    fn reference_domain() -> C0CouponSearchDomainV1 {
        C0CouponSearchDomainV1::new(
            explicit_domain(C0_PARAMETER_SECTION_WIDTH, &[16_000, 18_000, 20_000, 22_000, 24_000]),
            explicit_domain(C0_PARAMETER_SECTION_HEIGHT, &[4_800, 5_400, 6_000, 6_600, 7_200]),
        )
    }

    fn template() -> C0JointLinkCouponTemplateV1 {
        C0JointLinkCouponTemplateV1::new(
            ExactDesignLengthUmV1::from_um(300_000),
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
            digest(1),
            digest(2),
            digest(3),
        )
    }

    fn baseline() -> C0CompiledCouponV1 {
        let parameters = make_parameter_set(
            ExactDesignLengthUmV1::from_um(300_000),
            ExactDesignLengthUmV1::from_um(20_000),
            ExactDesignLengthUmV1::from_um(6_000),
        )
        .unwrap();
        compile_c0_joint_link_coupon(&template(), &parameters, &reference_domain()).unwrap()
    }

    fn applicability_profile() -> C0ApplicabilityProfileV1 {
        C0ApplicabilityProfileV1 {
            max_conservative_shear_to_bending_ratio: C0PositiveRationalV1::new(1, 100).unwrap(),
            max_height_to_width_ratio: C0PositiveRationalV1::new(2, 1).unwrap(),
        }
    }

    fn selection_policy() -> C0SelectionPolicyV1 {
        C0SelectionPolicyV1 {
            max_deflection_ratio: C0PositiveRationalV1::new(19, 20).unwrap(),
            max_stress_ratio: C0PositiveRationalV1::new(1, 1).unwrap(),
            require_applicability_admitted: true,
        }
    }

    fn evaluate_reference() -> C0ExhaustiveEvaluationV1 {
        evaluate_c0_domain_exhaustively(
            &template(),
            &reference_domain(),
            &baseline(),
            C0MatchedInvariantProfileId::from_digest(digest(9)),
            ExactDesignLengthUmV1::from_um(300_000),
            applicability_profile(),
            selection_policy(),
            DEFAULT_C0_CANDIDATE_BUDGET,
        )
        .unwrap()
    }

    #[test]
    fn reference_domain_enumerates_25_unique_designs() {
        let result = evaluate_reference();
        assert_eq!(result.candidates.len(), 25);
        let ids = result
            .candidates
            .iter()
            .map(|candidate| candidate.robot_design_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 25);
        assert_eq!(result.receipt.candidate_count, 25);
    }

    #[test]
    fn reference_pareto_frontier_has_exact_nine_coordinates() {
        let result = evaluate_reference();
        let coordinates = result
            .pareto_frontier
            .iter()
            .map(|candidate| (candidate.width.as_um(), candidate.height.as_um()))
            .collect::<BTreeSet<_>>();
        let expected = [
            (16_000, 4_800),
            (16_000, 5_400),
            (16_000, 6_000),
            (16_000, 6_600),
            (16_000, 7_200),
            (18_000, 7_200),
            (20_000, 7_200),
            (22_000, 7_200),
            (24_000, 7_200),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        assert_eq!(coordinates, expected);
        assert_eq!(result.receipt.pareto_count, 9);
    }

    #[test]
    fn protected_policy_selects_reference_candidate() {
        let result = evaluate_reference();
        assert_eq!(result.selected.width.as_um(), 16_000);
        assert_eq!(result.selected.height.as_um(), 7_200);
        assert_eq!(
            result.selected.normalized.mass_ratio,
            C0PositiveRationalV1::new(24, 25).unwrap()
        );
        assert_eq!(
            result.selected.normalized.deflection_ratio,
            C0PositiveRationalV1::new(625, 864).unwrap()
        );
        assert_eq!(
            result.selected.normalized.stress_ratio,
            C0PositiveRationalV1::new(125, 144).unwrap()
        );
        assert_eq!(
            result.selected.applicability_disposition,
            C0ApplicabilityDispositionV1::AdmittedGeometryScreen
        );
        assert_eq!(result.receipt.selected_design_id, result.selected.robot_design_id);
    }

    #[test]
    fn zero_budget_rejects_before_enumeration() {
        assert_eq!(
            evaluate_c0_domain_exhaustively(
                &template(),
                &reference_domain(),
                &baseline(),
                C0MatchedInvariantProfileId::from_digest(digest(9)),
                ExactDesignLengthUmV1::from_um(300_000),
                applicability_profile(),
                selection_policy(),
                0,
            )
            .unwrap_err(),
            C0EnumerationError::ZeroCandidateBudget
        );
    }

    #[test]
    fn hostile_large_cartesian_domain_rejects_before_candidate_allocation() {
        let width = ExactLengthDomainV1::new(
            DesignParameterId::new(C0_PARAMETER_SECTION_WIDTH).unwrap(),
            true,
            1_000,
            ExactLengthDomainKindV1::Stepped {
                lower: ExactDesignLengthUmV1::from_um(1),
                upper: ExactDesignLengthUmV1::from_um(1_000),
                step: ExactDesignLengthUmV1::from_um(1),
            },
        );
        let height = ExactLengthDomainV1::new(
            DesignParameterId::new(C0_PARAMETER_SECTION_HEIGHT).unwrap(),
            true,
            1_000,
            ExactLengthDomainKindV1::Stepped {
                lower: ExactDesignLengthUmV1::from_um(1),
                upper: ExactDesignLengthUmV1::from_um(1_000),
                step: ExactDesignLengthUmV1::from_um(1),
            },
        );
        let huge = C0CouponSearchDomainV1::new(width, height);
        assert_eq!(
            evaluate_c0_domain_exhaustively(
                &template(),
                &huge,
                &baseline(),
                C0MatchedInvariantProfileId::from_digest(digest(9)),
                ExactDesignLengthUmV1::from_um(300_000),
                applicability_profile(),
                selection_policy(),
                DEFAULT_C0_CANDIDATE_BUDGET,
            )
            .unwrap_err(),
            C0EnumerationError::CandidateBudgetExceeded {
                candidate_count: 1_000_000,
                candidate_budget: DEFAULT_C0_CANDIDATE_BUDGET,
            }
        );
    }

    #[test]
    fn receipt_is_stable_under_repeat_evaluation() {
        let first = evaluate_reference();
        let second = evaluate_reference();
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.receipt_id(), second.receipt.receipt_id());
    }
}
