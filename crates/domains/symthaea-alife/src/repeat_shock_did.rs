// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired difference-in-differences analysis for repeated-shock ALife experiments.
//!
//! [`crate::compare_repeated_shocks`] asks whether one condition responded differently to its
//! second matched shock than to its first. This module asks the next, stricter question:
//!
//! > Did the evolving condition's shock-2-vs-shock-1 change exceed the frozen control's change?
//!
//! For every continuous recovery dimension already oriented so positive means "better on shock
//! 2", the paired difference-in-differences quantity is
//!
//! ```text
//! (evolving shock2 - shock1) - (frozen shock2 - shock1)
//! ```
//!
//! Positive therefore means the evolving condition improved more (or worsened less) across the
//! repeated exposure than the frozen control. The result is descriptive paired evidence only; it
//! is not a p-value, population-level causal estimate, or proof that mutation caused the change.

use crate::{RepeatShockLatencyV1, RepeatShockTransferV1};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedShockDidVerdictV1 {
    /// At least one continuous dimension favors evolving and none favor frozen.
    ParetoEvolving,
    /// Some continuous dimensions favor evolving while others favor frozen.
    Mixed,
    /// No directional difference in the selected continuous dimensions.
    NoDirectionalDifference,
    /// At least one continuous dimension favors frozen and none favor evolving.
    ParetoFrozen,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeatedShockDidV1 {
    /// Positive means evolving retained more of baseline during shock 2 relative to shock 1 than
    /// frozen did over the same repeated-shock contrast.
    pub minimum_during_fraction_did: f64,
    /// Positive means evolving retained more of baseline after shock 2 relative to shock 1 than
    /// frozen did.
    pub minimum_after_fraction_did: f64,
    /// Positive means evolving improved its matched-horizon final fraction more than frozen did.
    pub final_fraction_did: f64,
    /// Positive means evolving reduced normalized population-deficit area more across shocks than
    /// frozen did.
    pub deficit_area_advantage_did: f64,
    /// Kept descriptively. Latency is intentionally excluded from the Pareto verdict because
    /// `SecondOnlyRecovered` / `FirstOnlyRecovered` form a partially ordered categorical space;
    /// assigning them arbitrary numeric infinities would manufacture a scale that the evidence
    /// does not provide.
    pub frozen_latency_transfer: RepeatShockLatencyV1,
    pub evolving_latency_transfer: RepeatShockLatencyV1,
    pub verdict: RepeatedShockDidVerdictV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedShockDidErrorV1 {
    NonFiniteMetric,
}

/// Compare the already-matched repeated-shock transfer of an evolving condition with its frozen
/// control using a conservative continuous-metric difference-in-differences rule.
///
/// Callers must obtain both inputs from the same shock protocol. The canonical sweep does so by
/// applying the same perturbation schedule, baseline lookback, recovery fraction and bounded
/// horizon to both conditions before calling this function.
pub fn compare_repeated_shock_did(
    frozen: &RepeatShockTransferV1,
    evolving: &RepeatShockTransferV1,
) -> Result<RepeatedShockDidV1, RepeatedShockDidErrorV1> {
    let minimum_during_fraction_did =
        evolving.minimum_during_fraction_delta - frozen.minimum_during_fraction_delta;
    let minimum_after_fraction_did =
        evolving.minimum_after_fraction_delta - frozen.minimum_after_fraction_delta;
    let final_fraction_did = evolving.final_fraction_delta - frozen.final_fraction_delta;
    let deficit_area_advantage_did =
        evolving.deficit_area_advantage - frozen.deficit_area_advantage;

    let metrics = [
        minimum_during_fraction_did,
        minimum_after_fraction_did,
        final_fraction_did,
        deficit_area_advantage_did,
    ];
    if metrics.iter().any(|value| !value.is_finite()) {
        return Err(RepeatedShockDidErrorV1::NonFiniteMetric);
    }

    let mut any_evolving = false;
    let mut any_frozen = false;
    for value in metrics {
        if value > 0.0 {
            any_evolving = true;
        } else if value < 0.0 {
            any_frozen = true;
        }
    }

    let verdict = match (any_evolving, any_frozen) {
        (true, false) => RepeatedShockDidVerdictV1::ParetoEvolving,
        (true, true) => RepeatedShockDidVerdictV1::Mixed,
        (false, false) => RepeatedShockDidVerdictV1::NoDirectionalDifference,
        (false, true) => RepeatedShockDidVerdictV1::ParetoFrozen,
    };

    Ok(RepeatedShockDidV1 {
        minimum_during_fraction_did,
        minimum_after_fraction_did,
        final_fraction_did,
        deficit_area_advantage_did,
        frozen_latency_transfer: frozen.latency,
        evolving_latency_transfer: evolving.latency,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RepeatShockTransferVerdictV1, RepeatShockTransferV1};

    fn transfer(
        during: f64,
        after: f64,
        final_fraction: f64,
        deficit_advantage: f64,
    ) -> RepeatShockTransferV1 {
        RepeatShockTransferV1 {
            first_baseline_mean_observed_population: 20.0,
            second_baseline_mean_observed_population: 20.0,
            minimum_during_fraction_delta: during,
            minimum_after_fraction_delta: after,
            final_fraction_delta: final_fraction,
            deficit_area_advantage: deficit_advantage,
            latency: RepeatShockLatencyV1::NeitherRecovered,
            verdict: RepeatShockTransferVerdictV1::Mixed,
        }
    }

    #[test]
    fn pareto_evolving_requires_no_continuous_dimension_to_favor_frozen() {
        let frozen = transfer(0.01, 0.00, -0.02, 0.5);
        let evolving = transfer(0.05, 0.02, 0.01, 1.0);
        let did = compare_repeated_shock_did(&frozen, &evolving).expect("finite DID");
        assert_eq!(did.verdict, RepeatedShockDidVerdictV1::ParetoEvolving);
        assert!(did.minimum_during_fraction_did > 0.0);
        assert!(did.minimum_after_fraction_did > 0.0);
        assert!(did.final_fraction_did > 0.0);
        assert!(did.deficit_area_advantage_did > 0.0);
    }

    #[test]
    fn mixed_relative_transfer_is_not_collapsed_into_evolving_advantage() {
        let frozen = transfer(0.01, 0.02, 0.01, 0.5);
        let evolving = transfer(0.05, 0.01, 0.03, 1.0);
        let did = compare_repeated_shock_did(&frozen, &evolving).expect("finite DID");
        assert_eq!(did.verdict, RepeatedShockDidVerdictV1::Mixed);
    }

    #[test]
    fn exact_equality_is_directionally_neutral() {
        let frozen = transfer(0.01, 0.02, 0.03, 0.5);
        let evolving = frozen;
        let did = compare_repeated_shock_did(&frozen, &evolving).expect("finite DID");
        assert_eq!(
            did.verdict,
            RepeatedShockDidVerdictV1::NoDirectionalDifference
        );
    }

    #[test]
    fn non_finite_relative_metric_fails_closed() {
        let frozen = transfer(0.01, 0.02, 0.03, 0.5);
        let evolving = transfer(f64::NAN, 0.02, 0.03, 0.5);
        assert_eq!(
            compare_repeated_shock_did(&frozen, &evolving),
            Err(RepeatedShockDidErrorV1::NonFiniteMetric)
        );
    }
}
