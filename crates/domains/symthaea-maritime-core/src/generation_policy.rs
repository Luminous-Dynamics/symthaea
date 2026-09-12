// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-specific projection from conservative temporal lineage runway into
//! descendant-generation semantics.
//!
//! A successor can be founded, complete its maturity policy, and reproduce again
//! on different boundaries. This module keeps those counts distinct so a terminal
//! descendant that exists but never matures is not confused with a completed
//! reproductive cycle. The projection is purely diagnostic and carries no
//! manufacturing, qualification, operating, or physical-control authority.

use crate::RegenerativeHorizon;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegenerativeGenerationCountV1 {
    Finite(u64),
    IndefiniteUnderStaticModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenerationPolicyProjectionV1 {
    pub temporal_successor_horizon: RegenerativeHorizon,
    /// Minimum complete periods a descendant must survive before another
    /// reproduction handoff may occur.
    pub maturity_periods: u64,
    /// Descendants successfully founded, including the already-founded first
    /// successor represented by `temporal_successor_horizon` when its horizon is >0.
    pub founded_descendant_generations: RegenerativeGenerationCountV1,
    /// Descendants that complete the full maturity interval. The terminal founded
    /// descendant may be absent from this count if its residual runway is shorter.
    pub maturity_completed_descendant_generations: RegenerativeGenerationCountV1,
    /// Reproduction transitions after the already-founded first successor.
    pub descendant_reproduction_transitions: RegenerativeGenerationCountV1,
    /// Remaining complete periods in the terminal founded descendant for a finite
    /// projection. Zero when no viable descendant can complete a period; `None`
    /// for an indefinite-under-static-model result.
    pub terminal_generation_residual_periods: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeGenerationPolicyError {
    ZeroMaturityPeriods,
}

/// Project one conservative successor runway through a fixed descendant maturity policy.
///
/// For finite positive runway `H` and maturity `M`:
///
/// - founded descendants = `ceil(H / M)`;
/// - maturity-completed descendants = `floor(H / M)`;
/// - post-successor reproduction transitions = `floor((H - 1) / M)`.
///
/// The distinction matters at exact boundaries. A generation with `H == M` can
/// complete its maturity interval but consumes the final finite support while doing
/// so, leaving no positive reserve for another descendant handoff.
pub fn project_regenerative_generation_policy(
    temporal_successor_horizon: RegenerativeHorizon,
    maturity_periods: u64,
) -> Result<RegenerativeGenerationPolicyProjectionV1, RegenerativeGenerationPolicyError> {
    if maturity_periods == 0 {
        return Err(RegenerativeGenerationPolicyError::ZeroMaturityPeriods);
    }

    let (
        founded_descendant_generations,
        maturity_completed_descendant_generations,
        descendant_reproduction_transitions,
        terminal_generation_residual_periods,
    ) = match temporal_successor_horizon {
        RegenerativeHorizon::FinitePeriods(0) => (
            RegenerativeGenerationCountV1::Finite(0),
            RegenerativeGenerationCountV1::Finite(0),
            RegenerativeGenerationCountV1::Finite(0),
            Some(0),
        ),
        RegenerativeHorizon::FinitePeriods(periods) => {
            let transitions = (periods - 1) / maturity_periods;
            let founded = transitions + 1;
            let matured = periods / maturity_periods;
            let residual = periods - transitions * maturity_periods;
            (
                RegenerativeGenerationCountV1::Finite(founded),
                RegenerativeGenerationCountV1::Finite(matured),
                RegenerativeGenerationCountV1::Finite(transitions),
                Some(residual),
            )
        }
        RegenerativeHorizon::IndefiniteUnderStaticModel => (
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel,
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel,
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel,
            None,
        ),
    };

    Ok(RegenerativeGenerationPolicyProjectionV1 {
        temporal_successor_horizon,
        maturity_periods,
        founded_descendant_generations,
        maturity_completed_descendant_generations,
        descendant_reproduction_transitions,
        terminal_generation_residual_periods,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite(value: RegenerativeGenerationCountV1) -> u64 {
        match value {
            RegenerativeGenerationCountV1::Finite(value) => value,
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
                panic!("expected finite generation count")
            }
        }
    }

    #[test]
    fn terminal_founded_generation_can_fail_before_maturity() {
        let result = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(3),
            2,
        )
        .unwrap();
        assert_eq!(finite(result.founded_descendant_generations), 2);
        assert_eq!(finite(result.maturity_completed_descendant_generations), 1);
        assert_eq!(finite(result.descendant_reproduction_transitions), 1);
        assert_eq!(result.terminal_generation_residual_periods, Some(1));
    }

    #[test]
    fn exact_maturity_boundary_does_not_create_another_descendant() {
        let result = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(4),
            4,
        )
        .unwrap();
        assert_eq!(finite(result.founded_descendant_generations), 1);
        assert_eq!(finite(result.maturity_completed_descendant_generations), 1);
        assert_eq!(finite(result.descendant_reproduction_transitions), 0);
        assert_eq!(result.terminal_generation_residual_periods, Some(4));
    }

    #[test]
    fn two_period_policy_separates_founded_and_matured_counts() {
        let result = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(4),
            2,
        )
        .unwrap();
        assert_eq!(finite(result.founded_descendant_generations), 2);
        assert_eq!(finite(result.maturity_completed_descendant_generations), 2);
        assert_eq!(finite(result.descendant_reproduction_transitions), 1);
        assert_eq!(result.terminal_generation_residual_periods, Some(2));
    }

    #[test]
    fn zero_runway_founds_no_viable_descendant_period() {
        let result = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(0),
            3,
        )
        .unwrap();
        assert_eq!(finite(result.founded_descendant_generations), 0);
        assert_eq!(finite(result.maturity_completed_descendant_generations), 0);
        assert_eq!(finite(result.descendant_reproduction_transitions), 0);
        assert_eq!(result.terminal_generation_residual_periods, Some(0));
    }

    #[test]
    fn indefinite_static_runway_stays_explicitly_conditional() {
        let result = project_regenerative_generation_policy(
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            2,
        )
        .unwrap();
        assert_eq!(
            result.founded_descendant_generations,
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel
        );
        assert_eq!(result.terminal_generation_residual_periods, None);
    }

    #[test]
    fn maturity_policy_must_be_nonzero() {
        assert_eq!(
            project_regenerative_generation_policy(RegenerativeHorizon::FinitePeriods(4), 0),
            Err(RegenerativeGenerationPolicyError::ZeroMaturityPeriods)
        );
    }
}
