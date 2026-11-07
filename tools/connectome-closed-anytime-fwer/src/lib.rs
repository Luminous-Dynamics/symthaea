//! Exact closed-testing procedure for finite phase-response e-process families.
//!
//! For each nonempty intersection I of elementary null hypotheses, V1 forms
//! a fixed convex mixture of the elementary e-processes in I. Under H_I all
//! constituent nulls are true, so the mixture is a test supermartingale in
//! the common global filtration. Closed testing rejects H_i only after every
//! intersection containing i has crossed its Ville threshold.
//!
//! The crate assumes upstream elementary e-process validity. It checks shared
//! filtration/split commitments and exact family semantics; it does not
//! authenticate source receipts or prove those upstream processes valid.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Exact non-negative rational number with checked u128 arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, ClosedError> {
        if denominator == 0 {
            return Err(ClosedError("rational denominator is zero".into()));
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    pub const fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    pub fn numerator(self) -> u128 {
        self.numerator
    }

    pub fn denominator(self) -> u128 {
        self.denominator
    }

    pub fn is_zero(self) -> bool {
        self.numerator == 0
    }

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, ClosedError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or_else(|| ClosedError("rational comparison overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or_else(|| ClosedError("rational comparison overflow".into()))?;
        Ok(left.cmp(&right))
    }

    pub fn checked_add(self, other: Self) -> Result<Self, ClosedError> {
        let common = gcd(self.denominator, other.denominator);
        let left_factor = other.denominator / common;
        let right_factor = self.denominator / common;
        let left = self
            .numerator
            .checked_mul(left_factor)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let right = other
            .numerator
            .checked_mul(right_factor)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let numerator = left
            .checked_add(right)
            .ok_or_else(|| ClosedError("rational addition overflow".into()))?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or_else(|| ClosedError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, ClosedError> {
        let cross_left = gcd(self.numerator, other.denominator);
        let cross_right = gcd(other.numerator, self.denominator);
        let numerator = (self.numerator / cross_left)
            .checked_mul(other.numerator / cross_right)
            .ok_or_else(|| ClosedError("rational multiplication overflow".into()))?;
        let denominator = (self.denominator / cross_right)
            .checked_mul(other.denominator / cross_left)
            .ok_or_else(|| ClosedError("rational denominator overflow".into()))?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, ClosedError> {
        if other.numerator == 0 {
            return Err(ClosedError("division by zero rational".into()));
        }
        self.checked_mul(Self::new(other.denominator, other.numerator)?)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(formatter, "{}", self.numerator)
        } else {
            write!(formatter, "{}/{}", self.numerator, self.denominator)
        }
    }
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedError(pub String);

impl fmt::Display for ClosedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ClosedError {}

#[derive(Clone, Debug)]
pub struct ElementaryClaim {
    pub claim_id: String,
    pub null_spec_sha256: String,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    pub component_validity_receipt_sha256: String,
    pub component_status: String,
    /// E_0,...,E_T. Every process must start at one.
    pub e_process_values: Vec<Rational>,
}

#[derive(Clone, Debug)]
pub struct ClosedFamily {
    pub family_id: String,
    pub requested_guarantee: String,
    pub registered_before_confirmatory_outcomes: bool,
    pub confirmation_lockbox_opened_before_freeze: bool,
    pub family_commitment_sha256: String,
    pub global_filtration_sha256: String,
    pub confirmation_split_sha256: String,
    pub alpha: Rational,
    pub claims: Vec<ElementaryClaim>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntersectionAudit {
    /// Canonical sorted elementary claim IDs in this intersection.
    pub claim_ids: Vec<String>,
    /// E_I,t, the fixed equal-weight mixture of elementary e-processes.
    pub mixture_e_process_values: Vec<Rational>,
    /// Running supremum of E_I,t; this is not itself described as an e-value.
    pub running_suprema: Vec<Rational>,
    pub rejection_path: Vec<bool>,
    pub first_crossing_stage: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElementaryAudit {
    pub claim_id: String,
    /// Number of complete intersection hypotheses that contain this claim.
    pub required_intersection_count: usize,
    /// First stage at which every required intersection has crossed.
    pub first_rejection_stage: Option<usize>,
    pub rejection_path: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedAudit {
    pub family_id: String,
    pub alpha: Rational,
    pub threshold: Rational,
    pub horizon: usize,
    pub elementary_claim_count: usize,
    pub intersection_count: usize,
    pub intersections: Vec<IntersectionAudit>,
    pub elementary: Vec<ElementaryAudit>,
    pub family_fwer_bound: Rational,
    pub guarantee: &'static str,
    pub upstream_validity: &'static str,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Audit complete closed testing over all nonempty subsets (at most 8 claims).
pub fn audit_closed_family(family: &ClosedFamily) -> Result<ClosedAudit, ClosedError> {
    if family.family_id.is_empty() {
        return Err(ClosedError("family_id must be non-empty".into()));
    }
    if family.requested_guarantee != "FWER" {
        return Err(ClosedError(
            "V1 supports closed FWER only; FDR/FDR-sup is a different guarantee".into(),
        ));
    }
    if !family.registered_before_confirmatory_outcomes
        || family.confirmation_lockbox_opened_before_freeze
    {
        return Err(ClosedError(
            "confirmatory family was not frozen before lockbox outcomes".into(),
        ));
    }
    for (label, digest) in [
        ("family commitment", family.family_commitment_sha256.as_str()),
        ("global filtration", family.global_filtration_sha256.as_str()),
        ("confirmation split", family.confirmation_split_sha256.as_str()),
    ] {
        if !valid_digest(digest) {
            return Err(ClosedError(format!("malformed {label} digest")));
        }
    }
    if family.alpha.is_zero() || family.alpha.checked_cmp(Rational::one())? == Ordering::Greater {
        return Err(ClosedError("alpha must lie in (0,1]".into()));
    }
    let claim_count = family.claims.len();
    if claim_count == 0 {
        return Err(ClosedError("empty elementary claim family".into()));
    }
    if claim_count > 8 {
        return Err(ClosedError(
            "NOT_SUPPORTED: V1 complete closure is capped at 8 elementary claims".into(),
        ));
    }

    let horizon = family.claims[0]
        .e_process_values
        .len()
        .checked_sub(1)
        .ok_or_else(|| ClosedError("empty e-process path".into()))?;
    if horizon == 0 {
        return Err(ClosedError("V1 requires at least one evidence update".into()));
    }

    let mut ids = BTreeSet::new();
    for claim in &family.claims {
        if claim.claim_id.is_empty() || !ids.insert(claim.claim_id.as_str()) {
            return Err(ClosedError("empty or duplicate elementary claim ID".into()));
        }
        for (label, digest) in [
            ("null specification", claim.null_spec_sha256.as_str()),
            ("claim global filtration", claim.global_filtration_sha256.as_str()),
            ("claim confirmation split", claim.confirmation_split_sha256.as_str()),
            ("component validity receipt", claim.component_validity_receipt_sha256.as_str()),
        ] {
            if !valid_digest(digest) {
                return Err(ClosedError(format!(
                    "malformed {label} digest for {}",
                    claim.claim_id
                )));
            }
        }
        if claim.global_filtration_sha256 != family.global_filtration_sha256 {
            return Err(ClosedError(format!(
                "{} is not bound to the common global filtration",
                claim.claim_id
            )));
        }
        if claim.confirmation_split_sha256 != family.confirmation_split_sha256 {
            return Err(ClosedError(format!(
                "{} is not bound to the common confirmation split",
                claim.claim_id
            )));
        }
        if claim.component_status != "VALIDATED_GLOBAL_FILTRATION_EPROCESS" {
            return Err(ClosedError(format!(
                "{} lacks a validated global-filtration e-process receipt",
                claim.claim_id
            )));
        }
        if claim.e_process_values.len() != horizon + 1 {
            return Err(ClosedError(
                "elementary e-processes do not share one horizon".into(),
            ));
        }
        if claim.e_process_values[0] != Rational::one() {
            return Err(ClosedError(format!(
                "{}: elementary e-process must start at one",
                claim.claim_id
            )));
        }
    }

    let threshold = Rational::one().checked_div(family.alpha)?;
    let intersection_count = (1usize << claim_count) - 1;
    let mut intersections = Vec::with_capacity(intersection_count);
    let mut rejection_by_mask = vec![Vec::new(); intersection_count + 1];

    // Enumerate all nonempty subsets. For each I, E_I is the fixed equal
    // mixture of {E_i : i in I}; the weights do not depend on observed data.
    for mask in 1..=intersection_count {
        let member_indices: Vec<usize> = (0..claim_count)
            .filter(|index| (mask & (1usize << index)) != 0)
            .collect();
        let mixture_weight = Rational::one()
            .checked_div(Rational::new(member_indices.len() as u128, 1)?)?;
        let mut mixture_path = Vec::with_capacity(horizon + 1);
        let mut running_suprema = Vec::with_capacity(horizon + 1);
        let mut rejection_path = Vec::with_capacity(horizon + 1);
        let mut running_supremum = Rational::zero();
        let mut first_crossing_stage = None;

        for stage in 0..=horizon {
            let mut mixture_value = Rational::zero();
            for index in &member_indices {
                mixture_value = mixture_value.checked_add(
                    family.claims[*index].e_process_values[stage].checked_mul(mixture_weight)?,
                )?;
            }
            mixture_path.push(mixture_value);
            if mixture_value.checked_cmp(running_supremum)? == Ordering::Greater {
                running_supremum = mixture_value;
            }
            running_suprema.push(running_supremum);
            let crossed = running_supremum.checked_cmp(threshold)? != Ordering::Less;
            if crossed && first_crossing_stage.is_none() {
                first_crossing_stage = Some(stage);
            }
            rejection_path.push(crossed);
        }

        rejection_by_mask[mask] = rejection_path.clone();
        let claim_ids = member_indices
            .iter()
            .map(|index| family.claims[*index].claim_id.clone())
            .collect();
        intersections.push(IntersectionAudit {
            claim_ids,
            mixture_e_process_values: mixture_path,
            running_suprema,
            rejection_path,
            first_crossing_stage,
        });
    }

    // Closed testing: H_i is rejected only if every nonempty intersection
    // containing i has crossed. This yields a persistent elementary decision.
    let mut elementary = Vec::with_capacity(claim_count);
    for index in 0..claim_count {
        let required_masks: Vec<usize> = (1..=intersection_count)
            .filter(|mask| (mask & (1usize << index)) != 0)
            .collect();
        let mut rejection_path = vec![false; horizon + 1];
        let mut first_rejection_stage = None;
        for stage in 0..=horizon {
            let all_crossed = required_masks
                .iter()
                .all(|mask| rejection_by_mask[*mask][stage]);
            rejection_path[stage] = all_crossed;
            if all_crossed && first_rejection_stage.is_none() {
                first_rejection_stage = Some(stage);
            }
        }
        elementary.push(ElementaryAudit {
            claim_id: family.claims[index].claim_id.clone(),
            required_intersection_count: required_masks.len(),
            first_rejection_stage,
            rejection_path,
        });
    }

    Ok(ClosedAudit {
        family_id: family.family_id.clone(),
        alpha: family.alpha,
        threshold,
        horizon,
        elementary_claim_count: claim_count,
        intersection_count,
        intersections,
        elementary,
        family_fwer_bound: family.alpha,
        guarantee: "CLOSED_ANYTIME_FWER_INTERSECTION_EPROCESS",
        upstream_validity: "ELEMENTARY_GLOBAL_FILTRATION_EPROCESS_VALIDITY_ASSUMED",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: u128, d: u128) -> Rational {
        Rational::new(n, d).unwrap()
    }

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn claim(id: &str, null_byte: char, path: &[Rational]) -> ElementaryClaim {
        ElementaryClaim {
            claim_id: id.into(),
            null_spec_sha256: digest(null_byte),
            global_filtration_sha256: digest('2'),
            confirmation_split_sha256: digest('3'),
            component_validity_receipt_sha256: digest('d'),
            component_status: "VALIDATED_GLOBAL_FILTRATION_EPROCESS".into(),
            e_process_values: path.to_vec(),
        }
    }

    fn fixture() -> ClosedFamily {
        ClosedFamily {
            family_id: "phase-response-closed-family-v1".into(),
            requested_guarantee: "FWER".into(),
            registered_before_confirmatory_outcomes: true,
            confirmation_lockbox_opened_before_freeze: false,
            family_commitment_sha256: digest('1'),
            global_filtration_sha256: digest('2'),
            confirmation_split_sha256: digest('3'),
            alpha: r(1, 5),
            claims: vec![
                claim("H1-local-reset", 'a', &[r(1, 1), r(5, 1), r(20, 1)]),
                claim("H2-off-diagonal-transfer", 'b', &[r(1, 1), r(4, 1), r(15, 1)]),
                claim("H3-contact-modulation", 'c', &[r(1, 1), r(3, 2), r(2, 1)]),
            ],
        }
    }

    #[test]
    fn exact_closed_testing_known_answer() {
        let report = audit_closed_family(&fixture()).unwrap();
        assert_eq!(report.threshold, r(5, 1));
        assert_eq!(report.intersection_count, 7);
        let values: Vec<Rational> = report
            .intersections
            .iter()
            .map(|row| *row.mixture_e_process_values.last().unwrap())
            .collect();
        assert_eq!(
            values,
            vec![
                r(20, 1),
                r(15, 1),
                r(35, 2),
                r(2, 1),
                r(11, 1),
                r(17, 2),
                r(37, 3)
            ]
        );
        assert_eq!(report.elementary[0].first_rejection_stage, Some(2));
        assert_eq!(report.elementary[1].first_rejection_stage, Some(2));
        assert_eq!(report.elementary[2].first_rejection_stage, None);
        assert_eq!(report.elementary[0].rejection_path, vec![false, false, true]);
        assert_eq!(report.elementary[1].rejection_path, vec![false, false, true]);
        assert_eq!(report.elementary[2].rejection_path, vec![false, false, false]);
        assert_eq!(report.family_fwer_bound, r(1, 5));
    }

    #[test]
    fn every_intersection_is_enumerated_once() {
        let report = audit_closed_family(&fixture()).unwrap();
        let ids: BTreeSet<Vec<String>> = report
            .intersections
            .iter()
            .map(|row| row.claim_ids.clone())
            .collect();
        assert_eq!(ids.len(), 7);
        assert!(ids.contains(&vec!["H1-local-reset".into()]));
        assert!(ids.contains(&vec![
            "H1-local-reset".into(),
            "H2-off-diagonal-transfer".into()
        ]));
    }

    #[test]
    fn decisions_are_persistent_after_raw_processes_fall() {
        let mut family = fixture();
        family.claims[0].e_process_values = vec![r(1, 1), r(20, 1), r(2, 1)];
        family.claims[1].e_process_values = vec![r(1, 1), r(15, 1), r(1, 1)];
        let report = audit_closed_family(&family).unwrap();
        assert_eq!(report.elementary[0].rejection_path, vec![false, false, true]);
        assert_eq!(report.elementary[1].rejection_path, vec![false, false, true]);
    }

    #[test]
    fn rejects_more_than_eight_elementary_claims() {
        let mut family = fixture();
        for index in 3..9 {
            family.claims.push(claim(
                &format!("H{index}"),
                'd',
                &[r(1, 1), r(1, 1), r(1, 1)],
            ));
        }
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_duplicate_elementary_claim_ids() {
        let mut family = fixture();
        family.claims[1].claim_id = family.claims[0].claim_id.clone();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_mixed_global_filtrations() {
        let mut family = fixture();
        family.global_filtration_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_with_mixed_global_filtration_receipt() {
        let mut family = fixture();
        family.claims[1].global_filtration_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_with_mixed_confirmation_split() {
        let mut family = fixture();
        family.claims[1].confirmation_split_sha256 = digest('9');
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_claim_without_component_validity_receipt() {
        let mut family = fixture();
        family.claims[1].component_status = "LOCAL_FILTRATION_ONLY".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_lockbox_open_before_family_freeze() {
        let mut family = fixture();
        family.confirmation_lockbox_opened_before_freeze = true;
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_fdr_sup_request() {
        let mut family = fixture();
        family.requested_guarantee = "FDR_SUP".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_incomplete_or_inconsistent_horizon() {
        let mut family = fixture();
        family.claims[2].e_process_values.pop();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_e_process_not_started_at_one() {
        let mut family = fixture();
        family.claims[1].e_process_values[0] = r(2, 1);
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn rejects_malformed_null_digest() {
        let mut family = fixture();
        family.claims[0].null_spec_sha256 = "bad".into();
        assert!(audit_closed_family(&family).is_err());
    }

    #[test]
    fn exact_rational_mixture_is_normalized() {
        let report = audit_closed_family(&fixture()).unwrap();
        let intersection = report
            .intersections
            .iter()
            .find(|row| row.claim_ids.len() == 3)
            .unwrap();
        assert_eq!(intersection.mixture_e_process_values[2], r(37, 3));
    }
}
