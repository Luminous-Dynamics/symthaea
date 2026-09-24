// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resource-bounded wrapper for exhaustive C0 enumeration.
//!
//! This module keeps the simple C0D enumerator honest by proving the complete
//! Cartesian domain fits a frozen candidate budget before candidate allocation.

use crate::c0_enumeration::{
    C0CandidateEvaluationV1, C0EnumerationError, enumerate_c0_domain,
};
use crate::c0_normalized::C0MatchedInvariantProfileId;
use crate::design_parameters::{
    DesignParameterError, DesignParameterSetV1, DesignSearchDomainV1,
};
use std::fmt;

pub const DEFAULT_MAX_C0_ENUMERATED_CANDIDATES: usize = 100_000;

#[derive(Debug)]
pub enum C0EnumerationGuardError {
    Parameters(DesignParameterError),
    Enumeration(C0EnumerationError),
    ZeroCandidateBudget,
    CandidateCountOverflow,
    CandidateBudgetExceeded { count: usize, maximum: usize },
}

impl fmt::Display for C0EnumerationGuardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameters(error) => write!(formatter, "C0 domain error: {error}"),
            Self::Enumeration(error) => write!(formatter, "C0 enumeration error: {error}"),
            Self::ZeroCandidateBudget => formatter.write_str("C0 candidate budget must be non-zero"),
            Self::CandidateCountOverflow => {
                formatter.write_str("C0 Cartesian candidate count overflowed usize")
            }
            Self::CandidateBudgetExceeded { count, maximum } => write!(
                formatter,
                "C0 Cartesian domain has {count} candidates, above frozen maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for C0EnumerationGuardError {}

impl From<DesignParameterError> for C0EnumerationGuardError {
    fn from(value: DesignParameterError) -> Self {
        Self::Parameters(value)
    }
}

impl From<C0EnumerationError> for C0EnumerationGuardError {
    fn from(value: C0EnumerationError) -> Self {
        Self::Enumeration(value)
    }
}

pub fn c0_cartesian_candidate_count(
    domain: &DesignSearchDomainV1,
) -> Result<usize, C0EnumerationGuardError> {
    domain.validate()?;
    let mut count = 1_usize;
    for parameter in domain.parameters() {
        let cardinality = parameter.canonical_values()?.len();
        count = count
            .checked_mul(cardinality)
            .ok_or(C0EnumerationGuardError::CandidateCountOverflow)?;
    }
    Ok(count)
}

pub fn enumerate_c0_domain_bounded(
    domain: &DesignSearchDomainV1,
    baseline: &DesignParameterSetV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
    maximum_candidates: usize,
) -> Result<Vec<C0CandidateEvaluationV1>, C0EnumerationGuardError> {
    if maximum_candidates == 0 {
        return Err(C0EnumerationGuardError::ZeroCandidateBudget);
    }
    let count = c0_cartesian_candidate_count(domain)?;
    if count > maximum_candidates {
        return Err(C0EnumerationGuardError::CandidateBudgetExceeded {
            count,
            maximum: maximum_candidates,
        });
    }
    Ok(enumerate_c0_domain(
        domain,
        baseline,
        matched_invariant_profile_id,
    )?)
}

pub fn enumerate_c0_domain_default_bounded(
    domain: &DesignSearchDomainV1,
    baseline: &DesignParameterSetV1,
    matched_invariant_profile_id: C0MatchedInvariantProfileId,
) -> Result<Vec<C0CandidateEvaluationV1>, C0EnumerationGuardError> {
    enumerate_c0_domain_bounded(
        domain,
        baseline,
        matched_invariant_profile_id,
        DEFAULT_MAX_C0_ENUMERATED_CANDIDATES,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c0_enumeration::{C0_HEIGHT_PARAMETER_ID, C0_WIDTH_PARAMETER_ID};
    use crate::design_parameters::{
        DesignLengthDomainV1, DesignLengthParameterV1, DesignLengthUm, DesignParameterId,
    };

    fn id(value: &str) -> DesignParameterId {
        DesignParameterId::new(value).unwrap()
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

    #[test]
    fn reference_domain_counts_25_before_enumeration() {
        assert_eq!(c0_cartesian_candidate_count(&reference_domain()).unwrap(), 25);
    }

    #[test]
    fn reference_domain_executes_under_small_explicit_budget() {
        let candidates = enumerate_c0_domain_bounded(
            &reference_domain(),
            &baseline(),
            C0MatchedInvariantProfileId::from_bytes([7; 32]),
            25,
        )
        .unwrap();
        assert_eq!(candidates.len(), 25);
    }

    #[test]
    fn over_budget_domain_rejects_before_candidate_allocation() {
        let domain = DesignSearchDomainV1::new(vec![
            DesignLengthDomainV1::SteppedInclusive {
                id: id(C0_WIDTH_PARAMETER_ID),
                lower: DesignLengthUm::from_micrometres(1),
                upper: DesignLengthUm::from_micrometres(1_000),
                step: DesignLengthUm::from_micrometres(1),
            },
            DesignLengthDomainV1::SteppedInclusive {
                id: id(C0_HEIGHT_PARAMETER_ID),
                lower: DesignLengthUm::from_micrometres(1),
                upper: DesignLengthUm::from_micrometres(1_000),
                step: DesignLengthUm::from_micrometres(1),
            },
        ])
        .unwrap();
        assert!(matches!(
            enumerate_c0_domain_bounded(
                &domain,
                &baseline(),
                C0MatchedInvariantProfileId::from_bytes([7; 32]),
                DEFAULT_MAX_C0_ENUMERATED_CANDIDATES
            ),
            Err(C0EnumerationGuardError::CandidateBudgetExceeded {
                count: 1_000_000,
                maximum: DEFAULT_MAX_C0_ENUMERATED_CANDIDATES
            })
        ));
    }

    #[test]
    fn zero_budget_rejects_even_a_tiny_domain() {
        assert!(matches!(
            enumerate_c0_domain_bounded(
                &reference_domain(),
                &baseline(),
                C0MatchedInvariantProfileId::from_bytes([7; 32]),
                0
            ),
            Err(C0EnumerationGuardError::ZeroCandidateBudget)
        ));
    }
}
