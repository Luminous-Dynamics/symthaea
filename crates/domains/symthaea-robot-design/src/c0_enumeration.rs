// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic exhaustive C0 candidate enumeration and selection.
//!
//! C0D deliberately uses complete enumeration over a bounded exact domain.
//! It does not establish optimizer superiority, physical applicability, material
//! truth, manufacturing capability, or physical improvement.

use crate::c0_normalized::{
    C0MatchedInvariantProfileId, C0NormalizedError, C0NormalizedEvaluationV1,
    C0RectangularSectionV1, PositiveRationalV1, evaluate_c0_normalized,
};
use crate::design_parameters::{
    DesignLengthParameterV1, DesignLengthUm, DesignParameterError, DesignParameterId,
    DesignParameterSetV1, DesignSearchDomainV1,
};
use std::cmp::Ordering;
use std::fmt;

pub const C0_WIDTH_PARAMETER_ID: &str = "section-width";
pub const C0_HEIGHT_PARAMETER_ID: &str = "section-height";

#[derive(Debug)]
pub enum C0EnumerationError {
    Parameters(DesignParameterError),
    Normalized(C0NormalizedError),
    MissingDomainParameter(&'static str),
    MissingBaselineParameter(&'static str),
    UnsupportedDomainParameter(String),
    BaselineNotAdmitted,
    NoFeasibleCandidate,
}

impl fmt::Display for C0EnumerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameters(error) => write!(formatter, "C0 parameter error: {error}"),
            Self::Normalized(error) => write!(formatter, "C0 normalized mechanics error: {error}"),
            Self::MissingDomainParameter(parameter) => {
                write!(formatter, "missing C0 search-domain parameter: {parameter}")
            }
            Self::MissingBaselineParameter(parameter) => {
                write!(formatter, "missing C0 baseline parameter: {parameter}")
            }
            Self::UnsupportedDomainParameter(parameter) => {
                write!(formatter, "unsupported C0 search-domain parameter: {parameter}")
            }
            Self::BaselineNotAdmitted => {
                formatter.write_str("C0 baseline is not admitted by the frozen search domain")
            }
            Self::NoFeasibleCandidate => {
                formatter.write_str("no C0 candidate satisfies the frozen selection policy")
            }
        }
    }
}

impl std::error::Error for C0EnumerationError {}

impl From<DesignParameterError> for C0EnumerationError {
    fn from(value: DesignParameterError) -> Self {
        Self::Parameters(value)
    }
}

impl From<C0NormalizedError> for C0EnumerationError {
    fn from(value: C0NormalizedError) -> Self {
        Self::Normalized(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C0CandidateEvaluationV1 {
    pub parameter_set: DesignParameterSetV1,
    pub width: DesignLengthUm,
    pub height: DesignLengthUm,
    pub normalized: C0NormalizedEvaluationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C0SelectionPolicyV1 {
    pub max_deflection_ratio: PositiveRationalV1,
    pub max_stress_ratio: PositiveRationalV1,
}

impl C0SelectionPolicyV1 {
    pub fn admits(
        self,
        candidate: &C0CandidateEvaluationV1,
    ) -> Result<bool, C0EnumerationError> {
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
        Ok(deflection_ok && stress_ok)
    }
}

pub fn enumerate_c0_domain(
    domain: &DesignSearchDomainV1,
    baseline: &DesignParameterSetV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
) -> Result<Vec<C0CandidateEvaluationV1>, C0EnumerationError> {
    domain.validate()?;
    baseline.validate()?;
    if !domain.admits(baseline)? {
        return Err(C0EnumerationError::BaselineNotAdmitted);
    }

    let baseline_width = baseline
        .value(C0_WIDTH_PARAMETER_ID)
        .ok_or(C0EnumerationError::MissingBaselineParameter(
            C0_WIDTH_PARAMETER_ID,
        ))?;
    let baseline_height = baseline
        .value(C0_HEIGHT_PARAMETER_ID)
        .ok_or(C0EnumerationError::MissingBaselineParameter(
            C0_HEIGHT_PARAMETER_ID,
        ))?;
    let baseline_section = C0RectangularSectionV1::new(
        baseline.id()?,
        baseline_width,
        baseline_height,
    )?;

    let mut width_values = None;
    let mut height_values = None;
    for parameter in domain.parameters() {
        match parameter.parameter_id().as_str() {
            C0_WIDTH_PARAMETER_ID => width_values = Some(parameter.canonical_values()?),
            C0_HEIGHT_PARAMETER_ID => height_values = Some(parameter.canonical_values()?),
            other => {
                return Err(C0EnumerationError::UnsupportedDomainParameter(
                    other.to_string(),
                ));
            }
        }
    }
    let width_values = width_values.ok_or(C0EnumerationError::MissingDomainParameter(
        C0_WIDTH_PARAMETER_ID,
    ))?;
    let height_values = height_values.ok_or(C0EnumerationError::MissingDomainParameter(
        C0_HEIGHT_PARAMETER_ID,
    ))?;

    let count = width_values
        .len()
        .checked_mul(height_values.len())
        .ok_or(DesignParameterError::DomainTooLarge {
            parameter: "C0 Cartesian product".to_string(),
            count: usize::MAX,
        })?;
    let mut candidates = Vec::with_capacity(count);

    for width in width_values {
        for &height in &height_values {
            let parameter_set = DesignParameterSetV1::new(vec![
                DesignLengthParameterV1 {
                    id: DesignParameterId::new(C0_WIDTH_PARAMETER_ID)?,
                    value: width,
                },
                DesignLengthParameterV1 {
                    id: DesignParameterId::new(C0_HEIGHT_PARAMETER_ID)?,
                    value: height,
                },
            ])?;
            let candidate_section = C0RectangularSectionV1::new(
                parameter_set.id()?,
                width,
                height,
            )?;
            let normalized = evaluate_c0_normalized(
                baseline_section,
                candidate_section,
                matched_invariant_profile_id,
            )?;
            candidates.push(C0CandidateEvaluationV1 {
                parameter_set,
                width,
                height,
                normalized,
            });
        }
    }

    candidates.sort_by(|left, right| {
        left.width
            .cmp(&right.width)
            .then_with(|| left.height.cmp(&right.height))
            .then_with(|| {
                left.parameter_set
                    .id()
                    .expect("validated candidate parameter set")
                    .cmp(&right.parameter_set.id().expect("validated candidate parameter set"))
            })
    });
    Ok(candidates)
}

pub fn c0_pareto_frontier(
    candidates: &[C0CandidateEvaluationV1],
) -> Result<Vec<C0CandidateEvaluationV1>, C0EnumerationError> {
    let mut frontier = Vec::new();
    'candidate: for candidate in candidates {
        for other in candidates {
            if candidate.parameter_set.id()? == other.parameter_set.id()? {
                continue;
            }
            if dominates_mass_deflection(other, candidate)? {
                continue 'candidate;
            }
        }
        frontier.push(candidate.clone());
    }
    sort_by_selection_order(&mut frontier)?;
    Ok(frontier)
}

pub fn select_c0_candidate(
    candidates: &[C0CandidateEvaluationV1],
    policy: C0SelectionPolicyV1,
) -> Result<C0CandidateEvaluationV1, C0EnumerationError> {
    let mut feasible = Vec::new();
    for candidate in candidates {
        if policy.admits(candidate)? {
            feasible.push(candidate.clone());
        }
    }
    if feasible.is_empty() {
        return Err(C0EnumerationError::NoFeasibleCandidate);
    }
    sort_by_selection_order(&mut feasible)?;
    Ok(feasible.remove(0))
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

fn sort_by_selection_order(
    candidates: &mut [C0CandidateEvaluationV1],
) -> Result<(), C0EnumerationError> {
    // Precompute an exact, fallible ordering key so sorting itself stays infallible.
    let mut indexed = Vec::with_capacity(candidates.len());
    for candidate in candidates.iter() {
        indexed.push((
            candidate.normalized.mass_ratio,
            candidate.normalized.deflection_ratio,
            candidate.parameter_set.id()?,
        ));
    }
    // The normalized C0 domains are intentionally bounded. If exact rational
    // cross-products overflow, fail before sorting rather than fall back to floats.
    for left in 0..indexed.len() {
        for right in (left + 1)..indexed.len() {
            indexed[left].0.checked_cmp(indexed[right].0)?;
            indexed[left].1.checked_cmp(indexed[right].1)?;
        }
    }
    candidates.sort_by(|left, right| {
        rational_cmp_infallible(left.normalized.mass_ratio, right.normalized.mass_ratio)
            .then_with(|| {
                rational_cmp_infallible(
                    left.normalized.deflection_ratio,
                    right.normalized.deflection_ratio,
                )
            })
            .then_with(|| {
                left.parameter_set
                    .id()
                    .expect("validated candidate parameter set")
                    .cmp(&right.parameter_set.id().expect("validated candidate parameter set"))
            })
    });
    Ok(())
}

fn rational_cmp_infallible(left: PositiveRationalV1, right: PositiveRationalV1) -> Ordering {
    left.numerator()
        .checked_mul(right.denominator())
        .expect("C0 rational comparison preflighted")
        .cmp(
            &right
                .numerator()
                .checked_mul(left.denominator())
                .expect("C0 rational comparison preflighted"),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design_parameters::DesignLengthDomainV1;

    fn id(value: &str) -> DesignParameterId {
        DesignParameterId::new(value).unwrap()
    }

    fn reference_domain() -> DesignSearchDomainV1 {
        DesignSearchDomainV1::new(vec![
            DesignLengthDomainV1::Explicit {
                id: id(C0_WIDTH_PARAMETER_ID),
                values: [16_000, 18_000, 20_000, 22_000, 24_000]
                    .into_iter()
                    .map(DesignLengthUm::from_micrometres)
                    .collect(),
            },
            DesignLengthDomainV1::Explicit {
                id: id(C0_HEIGHT_PARAMETER_ID),
                values: [4_800, 5_400, 6_000, 6_600, 7_200]
                    .into_iter()
                    .map(DesignLengthUm::from_micrometres)
                    .collect(),
            },
        ])
        .unwrap()
    }

    fn baseline() -> DesignParameterSetV1 {
        DesignParameterSetV1::new(vec![
            DesignLengthParameterV1 {
                id: id(C0_WIDTH_PARAMETER_ID),
                value: DesignLengthUm::from_micrometres(20_000),
            },
            DesignLengthParameterV1 {
                id: id(C0_HEIGHT_PARAMETER_ID),
                value: DesignLengthUm::from_micrometres(6_000),
            },
        ])
        .unwrap()
    }

    #[test]
    fn reference_domain_enumerates_exactly_25_unique_candidates() {
        let candidates = enumerate_c0_domain(
            &reference_domain(),
            &baseline(),
            C0MatchedInvariantProfileId::from_bytes([9; 32]),
        )
        .unwrap();
        assert_eq!(candidates.len(), 25);
        let ids = candidates
            .iter()
            .map(|candidate| candidate.parameter_set.id().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 25);
    }

    #[test]
    fn reference_domain_has_nine_mass_deflection_pareto_points() {
        let candidates = enumerate_c0_domain(
            &reference_domain(),
            &baseline(),
            C0MatchedInvariantProfileId::from_bytes([9; 32]),
        )
        .unwrap();
        let frontier = c0_pareto_frontier(&candidates).unwrap();
        assert_eq!(frontier.len(), 9);
    }

    #[test]
    fn protected_reference_policy_selects_08_width_12_height() {
        let candidates = enumerate_c0_domain(
            &reference_domain(),
            &baseline(),
            C0MatchedInvariantProfileId::from_bytes([9; 32]),
        )
        .unwrap();
        let selected = select_c0_candidate(
            &candidates,
            C0SelectionPolicyV1 {
                max_deflection_ratio: PositiveRationalV1::new(19, 20).unwrap(),
                max_stress_ratio: PositiveRationalV1::new(1, 1).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(selected.width.micrometres(), 16_000);
        assert_eq!(selected.height.micrometres(), 7_200);
        assert_eq!(
            selected.normalized.mass_ratio,
            PositiveRationalV1::new(24, 25).unwrap()
        );
    }

    #[test]
    fn extra_search_parameter_rejects_instead_of_becoming_anonymous_dimension() {
        let domain = DesignSearchDomainV1::new(vec![
            DesignLengthDomainV1::Explicit {
                id: id(C0_WIDTH_PARAMETER_ID),
                values: vec![DesignLengthUm::from_micrometres(20_000)],
            },
            DesignLengthDomainV1::Explicit {
                id: id(C0_HEIGHT_PARAMETER_ID),
                values: vec![DesignLengthUm::from_micrometres(6_000)],
            },
            DesignLengthDomainV1::Explicit {
                id: id("anonymous-third-axis"),
                values: vec![DesignLengthUm::from_micrometres(1)],
            },
        ])
        .unwrap();
        assert!(matches!(
            enumerate_c0_domain(
                &domain,
                &baseline(),
                C0MatchedInvariantProfileId::from_bytes([9; 32])
            ),
            Err(C0EnumerationError::UnsupportedDomainParameter(_))
        ));
    }
}
