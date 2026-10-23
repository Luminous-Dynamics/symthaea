// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Time-neutral forecast probability payloads.
//!
//! [`ForecastDistribution`](crate::ForecastDistribution) remains the canonical
//! tick-indexed forecast used by existing simulation-backed Futures Laboratory
//! scenarios. [`ForecastPayload`] deliberately removes issuance/horizon semantics
//! while preserving the exact probability/outcome validation contract, so future
//! scenario families can bind temporal/provenance semantics outside the scoring
//! payload rather than manufacturing fake ticks.

use serde::{Deserialize, Serialize};

use crate::{ForecastBranch, ForecastError, OutcomeSpaceId, Probability, MASS_TOLERANCE};

#[derive(Deserialize)]
struct ForecastPayloadRepr {
    outcome_space: OutcomeSpaceId,
    branches: Vec<ForecastBranch>,
    unsupported_mass: Probability,
}

/// Validated probability mass over one outcome space, with no clock or horizon
/// assumption. Temporal semantics belong to the commitment/provenance layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ForecastPayloadRepr")]
pub struct ForecastPayload {
    outcome_space: OutcomeSpaceId,
    branches: Vec<ForecastBranch>,
    unsupported_mass: Probability,
}

impl ForecastPayload {
    /// Construct a time-neutral payload while enforcing the same partition and
    /// probability-mass invariants as [`crate::ForecastDistribution`].
    pub fn try_new(
        outcome_space: OutcomeSpaceId,
        branches: Vec<ForecastBranch>,
        unsupported_mass: Probability,
    ) -> Result<Self, ForecastError> {
        if branches.is_empty() {
            return Err(ForecastError::EmptyDistribution);
        }

        for (index, first) in branches.iter().enumerate() {
            for second in &branches[index + 1..] {
                match (&first.outcome, &second.outcome) {
                    (crate::OutcomeRegion::Interval(a), crate::OutcomeRegion::Interval(b)) => {
                        if a.overlaps(b) {
                            return Err(ForecastError::OverlappingIntervals {
                                first: (a.low(), a.high()),
                                second: (b.low(), b.high()),
                            });
                        }
                    }
                    (a, b) if a == b => return Err(ForecastError::DuplicateOutcomeRegion),
                    _ => {}
                }
            }
        }

        let total = branches
            .iter()
            .map(|branch| branch.probability.get())
            .sum::<f64>()
            + unsupported_mass.get();
        if (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(ForecastError::MassNotNormalized {
                total,
                tolerance: MASS_TOLERANCE,
            });
        }

        Ok(Self {
            outcome_space,
            branches,
            unsupported_mass,
        })
    }

    /// Convenience constructor from raw branch probabilities.
    pub fn try_from_raw(
        outcome_space: OutcomeSpaceId,
        branches: Vec<(f64, crate::OutcomeRegion, Vec<crate::AssumptionId>)>,
        unsupported_mass: f64,
    ) -> Result<Self, ForecastError> {
        let branches = branches
            .into_iter()
            .map(|(probability, outcome, assumptions)| {
                ForecastBranch::new(probability, outcome, assumptions)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(
            outcome_space,
            branches,
            Probability::new(unsupported_mass)?,
        )
    }

    pub fn outcome_space(&self) -> &OutcomeSpaceId {
        &self.outcome_space
    }

    pub fn branches(&self) -> &[ForecastBranch] {
        &self.branches
    }

    pub fn unsupported_mass(&self) -> Probability {
        self.unsupported_mass
    }
}

impl TryFrom<ForecastPayloadRepr> for ForecastPayload {
    type Error = ForecastError;

    fn try_from(repr: ForecastPayloadRepr) -> Result<Self, Self::Error> {
        Self::try_new(repr.outcome_space, repr.branches, repr.unsupported_mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ForecastDistribution, Horizon, OutcomeRegion};

    fn outcome_space() -> OutcomeSpaceId {
        OutcomeSpaceId("binary_event".into())
    }

    fn branches() -> Vec<(f64, OutcomeRegion, Vec<crate::AssumptionId>)> {
        vec![
            (0.7, OutcomeRegion::Boolean(true), vec![]),
            (0.3, OutcomeRegion::Boolean(false), vec![]),
        ]
    }

    #[test]
    fn neutral_payload_matches_legacy_probability_surface() {
        let payload = ForecastPayload::try_from_raw(outcome_space(), branches(), 0.0).unwrap();
        let legacy = ForecastDistribution::try_from_raw(
            41,
            Horizon(12),
            outcome_space(),
            branches(),
            0.0,
        )
        .unwrap();

        assert_eq!(payload.outcome_space(), legacy.outcome_space());
        assert_eq!(payload.branches(), legacy.branches());
        assert_eq!(payload.unsupported_mass(), legacy.unsupported_mass());
    }

    #[test]
    fn neutral_payload_rejects_empty_distribution() {
        let payload = ForecastPayload::try_from_raw(outcome_space(), vec![], 0.0).unwrap_err();
        let legacy = ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            outcome_space(),
            vec![],
            0.0,
        )
        .unwrap_err();
        assert_eq!(payload, legacy);
        assert_eq!(payload, ForecastError::EmptyDistribution);
    }

    #[test]
    fn neutral_payload_rejects_duplicate_regions() {
        let candidate = vec![
            (0.5, OutcomeRegion::Boolean(true), vec![]),
            (0.5, OutcomeRegion::Boolean(true), vec![]),
        ];
        let payload =
            ForecastPayload::try_from_raw(outcome_space(), candidate.clone(), 0.0).unwrap_err();
        let legacy = ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            outcome_space(),
            candidate,
            0.0,
        )
        .unwrap_err();
        assert_eq!(payload, legacy);
        assert_eq!(payload, ForecastError::DuplicateOutcomeRegion);
    }

    #[test]
    fn neutral_payload_rejects_overlapping_intervals() {
        let candidate = vec![
            (0.5, OutcomeRegion::interval(0.0, 2.0).unwrap(), vec![]),
            (0.5, OutcomeRegion::interval(1.0, 3.0).unwrap(), vec![]),
        ];
        let payload =
            ForecastPayload::try_from_raw(outcome_space(), candidate.clone(), 0.0).unwrap_err();
        let legacy = ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            outcome_space(),
            candidate,
            0.0,
        )
        .unwrap_err();
        assert_eq!(payload, legacy);
        assert!(matches!(payload, ForecastError::OverlappingIntervals { .. }));
    }

    #[test]
    fn neutral_payload_rejects_unnormalized_mass() {
        let candidate = vec![(0.4, OutcomeRegion::Boolean(true), vec![])];
        let payload =
            ForecastPayload::try_from_raw(outcome_space(), candidate.clone(), 0.4).unwrap_err();
        let legacy = ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            outcome_space(),
            candidate,
            0.4,
        )
        .unwrap_err();
        assert_eq!(payload, legacy);
        assert!(matches!(payload, ForecastError::MassNotNormalized { .. }));
    }
}
