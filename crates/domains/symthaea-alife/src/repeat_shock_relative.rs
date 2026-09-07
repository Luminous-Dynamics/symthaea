// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic relative comparison between two repeated-shock transfer reports.
//!
//! This is the condition-agnostic form of the paired difference-in-differences question. It asks
//! whether a candidate condition changed more favorably from shock 1 to shock 2 than a reference
//! condition, across the same continuous recovery dimensions already oriented so positive means
//! "better on shock 2".

use crate::{RepeatShockLatencyV1, RepeatShockTransferV1};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedShockRelativeVerdictV1 {
    /// At least one continuous dimension favors the candidate and none favor the reference.
    ParetoCandidate,
    /// Some continuous dimensions favor each side.
    Mixed,
    /// No directional difference in the selected continuous dimensions.
    NoDirectionalDifference,
    /// At least one continuous dimension favors the reference and none favor the candidate.
    ParetoReference,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeatedShockRelativeV1 {
    pub minimum_during_candidate_minus_reference: f64,
    pub minimum_after_candidate_minus_reference: f64,
    pub final_fraction_candidate_minus_reference: f64,
    pub deficit_area_advantage_candidate_minus_reference: f64,
    /// Latency remains descriptive because recovery/non-recovery categories are only partially
    /// ordered and should not be forced onto an arbitrary numeric scale.
    pub reference_latency_transfer: RepeatShockLatencyV1,
    pub candidate_latency_transfer: RepeatShockLatencyV1,
    pub verdict: RepeatedShockRelativeVerdictV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedShockRelativeErrorV1 {
    NonFiniteMetric,
}

pub fn compare_repeated_shock_relative(
    reference: &RepeatShockTransferV1,
    candidate: &RepeatShockTransferV1,
) -> Result<RepeatedShockRelativeV1, RepeatedShockRelativeErrorV1> {
    let minimum_during_candidate_minus_reference =
        candidate.minimum_during_fraction_delta - reference.minimum_during_fraction_delta;
    let minimum_after_candidate_minus_reference =
        candidate.minimum_after_fraction_delta - reference.minimum_after_fraction_delta;
    let final_fraction_candidate_minus_reference =
        candidate.final_fraction_delta - reference.final_fraction_delta;
    let deficit_area_advantage_candidate_minus_reference =
        candidate.deficit_area_advantage - reference.deficit_area_advantage;

    let metrics = [
        minimum_during_candidate_minus_reference,
        minimum_after_candidate_minus_reference,
        final_fraction_candidate_minus_reference,
        deficit_area_advantage_candidate_minus_reference,
    ];
    if metrics.iter().any(|value| !value.is_finite()) {
        return Err(RepeatedShockRelativeErrorV1::NonFiniteMetric);
    }

    let mut any_candidate = false;
    let mut any_reference = false;
    for value in metrics {
        if value > 0.0 {
            any_candidate = true;
        } else if value < 0.0 {
            any_reference = true;
        }
    }

    let verdict = match (any_candidate, any_reference) {
        (true, false) => RepeatedShockRelativeVerdictV1::ParetoCandidate,
        (true, true) => RepeatedShockRelativeVerdictV1::Mixed,
        (false, false) => RepeatedShockRelativeVerdictV1::NoDirectionalDifference,
        (false, true) => RepeatedShockRelativeVerdictV1::ParetoReference,
    };

    Ok(RepeatedShockRelativeV1 {
        minimum_during_candidate_minus_reference,
        minimum_after_candidate_minus_reference,
        final_fraction_candidate_minus_reference,
        deficit_area_advantage_candidate_minus_reference,
        reference_latency_transfer: reference.latency,
        candidate_latency_transfer: candidate.latency,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RepeatShockTransferVerdictV1, RepeatShockTransferV1};

    fn transfer(during: f64, after: f64, final_fraction: f64, deficit: f64) -> RepeatShockTransferV1 {
        RepeatShockTransferV1 {
            first_baseline_mean_observed_population: 20.0,
            second_baseline_mean_observed_population: 20.0,
            minimum_during_fraction_delta: during,
            minimum_after_fraction_delta: after,
            final_fraction_delta: final_fraction,
            deficit_area_advantage: deficit,
            latency: RepeatShockLatencyV1::NeitherRecovered,
            verdict: RepeatShockTransferVerdictV1::Mixed,
        }
    }

    #[test]
    fn candidate_pareto_advantage_requires_no_reference_favoring_dimension() {
        let reference = transfer(0.01, 0.02, 0.00, 0.2);
        let candidate = transfer(0.03, 0.02, 0.04, 0.8);
        let result = compare_repeated_shock_relative(&reference, &candidate).expect("finite");
        assert_eq!(result.verdict, RepeatedShockRelativeVerdictV1::ParetoCandidate);
    }

    #[test]
    fn mixed_relative_evidence_remains_mixed() {
        let reference = transfer(0.01, 0.03, 0.00, 0.2);
        let candidate = transfer(0.03, 0.01, 0.04, 0.8);
        let result = compare_repeated_shock_relative(&reference, &candidate).expect("finite");
        assert_eq!(result.verdict, RepeatedShockRelativeVerdictV1::Mixed);
    }

    #[test]
    fn non_finite_metric_fails_closed() {
        let reference = transfer(0.0, 0.0, 0.0, 0.0);
        let candidate = transfer(f64::NAN, 0.0, 0.0, 0.0);
        assert_eq!(
            compare_repeated_shock_relative(&reference, &candidate),
            Err(RepeatedShockRelativeErrorV1::NonFiniteMetric)
        );
    }
}
