//! Observable responses to challenging task episodes.
//!
//! These measures describe task processes, not stable traits such as grit,
//! motivation, intelligence, resilience, or mental health. A voluntary stop
//! is recorded as an exercise of agency, not a failure.

use anyhow::{Result, bail};
use serde::Serialize;

/// One independently coded, preregistered challenge opportunity.
///
/// None means unavailable or not applicable; it is never converted to false.
/// A reattempt outcome is applicable only when a setback was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChallengeEpisodeObservation {
    pub attempted_before_support: Option<bool>,
    pub setback_observed: Option<bool>,
    pub reattempted_after_setback: Option<bool>,
    pub requested_hint: Option<bool>,
    pub requested_full_solution: Option<bool>,
    pub voluntarily_stopped: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ObservedRate {
    pub observed_count: usize,
    pub positive_count: usize,
    pub rate: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ChallengeResponseSummary {
    pub opportunity_count: usize,
    pub attempted_before_support: ObservedRate,
    pub setback_observed: ObservedRate,
    /// Reattempt rate is conditional on an observed setback.
    pub reattempted_after_setback: ObservedRate,
    pub requested_hint: ObservedRate,
    pub requested_full_solution: ObservedRate,
    /// A voluntary stop is not scored as a negative outcome.
    pub voluntarily_stopped: ObservedRate,
}

impl ObservedRate {
    fn from_values(values: impl Iterator<Item = Option<bool>>) -> Self {
        let (observed_count, positive_count) = values.fold(
            (0usize, 0usize),
            |(observed, positive), value| match value {
                Some(value) => (observed + 1, positive + usize::from(value)),
                None => (observed, positive),
            },
        );
        Self {
            observed_count,
            positive_count,
            rate: (observed_count > 0).then_some(positive_count as f64 / observed_count as f64),
        }
    }
}

/// Summarize process observations without rewarding effort, withholding,
/// persistence, or low AI use in isolation. Interpret them beside independent
/// immediate outcomes, delayed retention, transfer, support dose and burden.
pub fn summarize_challenge_response(
    observations: &[ChallengeEpisodeObservation],
) -> Result<ChallengeResponseSummary> {
    for (index, observation) in observations.iter().enumerate() {
        if observation.reattempted_after_setback.is_some()
            && observation.setback_observed != Some(true)
        {
            bail!("challenge episode {index}: reattempt outcome requires an observed setback");
        }
    }

    let flags =
        |select: fn(&ChallengeEpisodeObservation) -> Option<bool>| observations.iter().map(select);

    Ok(ChallengeResponseSummary {
        opportunity_count: observations.len(),
        attempted_before_support: ObservedRate::from_values(flags(|o| o.attempted_before_support)),
        setback_observed: ObservedRate::from_values(flags(|o| o.setback_observed)),
        reattempted_after_setback: ObservedRate::from_values(
            observations
                .iter()
                .filter(|o| o.setback_observed == Some(true))
                .map(|o| o.reattempted_after_setback),
        ),
        requested_hint: ObservedRate::from_values(flags(|o| o.requested_hint)),
        requested_full_solution: ObservedRate::from_values(flags(|o| o.requested_full_solution)),
        voluntarily_stopped: ObservedRate::from_values(flags(|o| o.voluntarily_stopped)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episode(
        attempted: Option<bool>,
        setback: Option<bool>,
        reattempted: Option<bool>,
        hint: Option<bool>,
        solution: Option<bool>,
        stopped: Option<bool>,
    ) -> ChallengeEpisodeObservation {
        ChallengeEpisodeObservation {
            attempted_before_support: attempted,
            setback_observed: setback,
            reattempted_after_setback: reattempted,
            requested_hint: hint,
            requested_full_solution: solution,
            voluntarily_stopped: stopped,
        }
    }

    #[test]
    fn known_answer_fixture_uses_event_specific_denominators() {
        let observations = [
            episode(
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            ),
            episode(
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            ),
            episode(
                Some(true),
                Some(false),
                None,
                Some(false),
                Some(true),
                Some(false),
            ),
            episode(Some(true), None, None, None, None, Some(true)),
        ];
        let summary = summarize_challenge_response(&observations).unwrap();

        assert_eq!(summary.opportunity_count, 4);
        assert_eq!(summary.attempted_before_support.observed_count, 4);
        assert_eq!(summary.attempted_before_support.positive_count, 3);
        assert_eq!(summary.attempted_before_support.rate, Some(0.75));
        assert_eq!(summary.setback_observed.observed_count, 3);
        assert_eq!(summary.setback_observed.positive_count, 2);
        assert_eq!(summary.reattempted_after_setback.observed_count, 2);
        assert_eq!(summary.reattempted_after_setback.positive_count, 1);
        assert_eq!(summary.reattempted_after_setback.rate, Some(0.5));
        assert_eq!(summary.requested_hint.rate, Some(0.25));
        assert_eq!(summary.requested_full_solution.rate, Some(0.25));
        assert_eq!(summary.voluntarily_stopped.rate, Some(0.25));
    }

    #[test]
    fn unavailable_observations_are_not_imputed_as_negative() {
        let observations = [
            episode(None, None, None, None, None, None),
            episode(Some(true), Some(false), None, Some(false), None, None),
        ];
        let summary = summarize_challenge_response(&observations).unwrap();

        assert_eq!(summary.attempted_before_support.observed_count, 1);
        assert_eq!(summary.attempted_before_support.rate, Some(1.0));
        assert_eq!(summary.setback_observed.observed_count, 2);
        assert_eq!(summary.reattempted_after_setback.observed_count, 0);
        assert_eq!(summary.reattempted_after_setback.rate, None);
        assert_eq!(summary.requested_full_solution.observed_count, 0);
        assert_eq!(summary.requested_full_solution.rate, None);
    }

    #[test]
    fn reattempt_without_observed_setback_is_rejected() {
        let observations = [episode(
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
        )];
        let error = summarize_challenge_response(&observations).unwrap_err();
        assert!(error.to_string().contains("requires an observed setback"));
    }

    #[test]
    fn voluntary_stop_is_preserved_as_its_own_outcome() {
        let observations = [
            episode(
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(true),
            ),
            episode(
                Some(true),
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
            ),
        ];
        let summary = summarize_challenge_response(&observations).unwrap();

        assert_eq!(summary.voluntarily_stopped.positive_count, 1);
        assert_eq!(summary.voluntarily_stopped.rate, Some(0.5));
        assert_eq!(summary.reattempted_after_setback.rate, Some(0.0));
    }

    #[test]
    fn empty_input_returns_unavailable_rates() {
        let summary = summarize_challenge_response(&[]).unwrap();
        assert_eq!(summary.opportunity_count, 0);
        assert_eq!(summary.attempted_before_support.rate, None);
        assert_eq!(summary.reattempted_after_setback.rate, None);
        assert_eq!(summary.voluntarily_stopped.rate, None);
    }
}
