#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal analytical reference arithmetic for yield-aware display design.
//!
//! This crate deliberately does not model a fabrication process, defect
//! correlations, semiconductor physics, optical behavior, or a real panel.
//! It makes the assumptions behind a few calculations explicit so synthetic
//! benchmark fixtures can be independently checked.
//!
//! Expected counts from probabilities are not observations. Multiplying stage
//! yields is accepted only when the caller explicitly declares independence.
//! When dependencies are correlated or unknown, callers must provide joint
//! evidence or a bounded scenario rather than silently multiplying stages.

use core::fmt;

/// Fail-closed input or model-assumption errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YieldError {
    /// A population/trial denominator must be positive.
    ZeroDenominator,
    /// Probability or z-score is NaN or infinite.
    NonFiniteInput,
    /// A probability is outside the closed interval [0, 1].
    ProbabilityOutOfRange,
    /// Observed successes exceed the number of trials.
    SuccessesExceedTrials,
    /// A Wilson interval requires a finite, strictly positive z-score.
    InvalidZScore,
    /// Multiplying stage probabilities would assume independence.
    IndependenceNotEstablished,
}

impl fmt::Display for YieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => {
                formatter.write_str("population/trial denominator must be positive")
            }
            Self::NonFiniteInput => formatter.write_str("numeric input must be finite"),
            Self::ProbabilityOutOfRange => formatter.write_str("probability must be in [0, 1]"),
            Self::SuccessesExceedTrials => {
                formatter.write_str("observed successes cannot exceed trials")
            }
            Self::InvalidZScore => {
                formatter.write_str("z-score must be finite and strictly positive")
            }
            Self::IndependenceNotEstablished => formatter
                .write_str("stage-product yield requires an explicit independent-stage assumption"),
        }
    }
}

impl std::error::Error for YieldError {}

/// Output from a simple independent-emitter expected-count calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpectedCounts {
    /// Declared population denominator.
    pub population: u64,
    /// Per-emitter success probability supplied by the caller.
    pub success_probability: f64,
    /// Expected count, which need not be an integer or realized count.
    pub expected_good: f64,
    /// Expected count, which need not be an integer or realized count.
    pub expected_defects: f64,
}

/// Counts for a binomial confidence interval from an observed sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedCounts {
    pub successes: u64,
    pub trials: u64,
}

/// Closed interval for an observed probability estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityInterval {
    pub lower: f64,
    pub upper: f64,
}

/// An explicit assumption label for composing stage-level probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependenceAssumption {
    Independent,
    CorrelatedOrUnknown,
}

/// Computes expected good and defective counts under an independent,
/// equal-probability Bernoulli model.
///
/// This is an expected-value model. It does not predict defect clustering,
/// process-stage dependence, assembled-panel yield, inspection coverage,
/// repair outcomes, or a guarantee about realized counts.
///
/// # Errors
///
/// Rejects zero population, non-finite probabilities and probabilities outside [0, 1].
pub fn expected_counts(
    population: u64,
    success_probability: f64,
) -> Result<ExpectedCounts, YieldError> {
    if population == 0 {
        return Err(YieldError::ZeroDenominator);
    }
    validate_probability(success_probability)?;

    let population_f64 = population as f64;
    Ok(ExpectedCounts {
        population,
        success_probability,
        expected_good: population_f64 * success_probability,
        expected_defects: population_f64 * (1.0 - success_probability),
    })
}

/// Returns the observed success fraction after validating its denominator.
pub fn observed_yield(counts: ObservedCounts) -> Result<f64, YieldError> {
    if counts.trials == 0 {
        return Err(YieldError::ZeroDenominator);
    }
    if counts.successes > counts.trials {
        return Err(YieldError::SuccessesExceedTrials);
    }
    Ok(counts.successes as f64 / counts.trials as f64)
}

/// Computes the Wilson score interval for an observed Bernoulli proportion.
///
/// The confidence level is determined by z_score; use 1.959963984540054
/// for the conventional two-sided 95% interval. The caller must ensure the
/// sampled units represent the target population and that the Bernoulli
/// sampling assumptions are appropriate. This function cannot correct biased
/// sampling, clustered defects, incomplete inspection coverage or non-random
/// test selection.
///
/// # Errors
///
/// Rejects zero trials, successes greater than trials, and invalid z-score.
pub fn wilson_interval(
    counts: ObservedCounts,
    z_score: f64,
) -> Result<ProbabilityInterval, YieldError> {
    if counts.trials == 0 {
        return Err(YieldError::ZeroDenominator);
    }
    if counts.successes > counts.trials {
        return Err(YieldError::SuccessesExceedTrials);
    }
    if !z_score.is_finite() {
        return Err(YieldError::NonFiniteInput);
    }
    if z_score <= 0.0 {
        return Err(YieldError::InvalidZScore);
    }

    let n = counts.trials as f64;
    let p = counts.successes as f64 / n;
    let z2 = z_score * z_score;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z_score * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;

    Ok(ProbabilityInterval {
        lower: (center - half_width).clamp(0.0, 1.0),
        upper: (center + half_width).clamp(0.0, 1.0),
    })
}

/// Composes stage yields only when the independent-stage assumption is explicit.
///
/// For correlated or unknown stage failures, use measured joint outcomes or a
/// declared bounded scenario. No automatic fallback multiplies probabilities.
///
/// # Errors
///
/// Returns IndependenceNotEstablished unless the caller explicitly supplies
/// DependenceAssumption::Independent.
pub fn compose_stage_yields(
    stage_yields: &[f64],
    assumption: DependenceAssumption,
) -> Result<f64, YieldError> {
    if assumption != DependenceAssumption::Independent {
        return Err(YieldError::IndependenceNotEstablished);
    }
    if stage_yields.is_empty() {
        return Err(YieldError::ZeroDenominator);
    }

    let mut product = 1.0;
    for &yield_probability in stage_yields {
        validate_probability(yield_probability)?;
        product *= yield_probability;
    }
    Ok(product)
}

/// Computes a simple repair-adjusted final-good fraction.
///
/// Model: first-pass-good units remain good; failed units are recoverable with
/// probability repair_success_probability, and repaired units pass retest with
/// probability retest_success_probability. This reduced model omits repair-
/// induced damage, shared-cause failures, capacity, downtime and calibration
/// burden. Benchmark reports must track those quantities separately.
///
/// # Errors
///
/// Rejects non-finite or out-of-range probabilities.
pub fn repair_adjusted_yield(
    first_pass_yield: f64,
    repair_success_probability: f64,
    retest_success_probability: f64,
) -> Result<f64, YieldError> {
    validate_probability(first_pass_yield)?;
    validate_probability(repair_success_probability)?;
    validate_probability(retest_success_probability)?;

    Ok(first_pass_yield
        + (1.0 - first_pass_yield) * repair_success_probability * retest_success_probability)
}

/// Computes the fraction of values that meet an explicit minimum quality floor.
///
/// The caller supplies measured/predicted values and floors; this helper does
/// not infer or qualify the metrics themselves.
pub fn fraction_meeting_floor(values: &[f64], minimum: f64) -> Result<f64, YieldError> {
    if values.is_empty() {
        return Err(YieldError::ZeroDenominator);
    }
    if !minimum.is_finite() || values.iter().any(|value| !value.is_finite()) {
        return Err(YieldError::NonFiniteInput);
    }
    Ok(values.iter().filter(|value| **value >= minimum).count() as f64 / values.len() as f64)
}

fn validate_probability(probability: f64) -> Result<(), YieldError> {
    if !probability.is_finite() {
        return Err(YieldError::NonFiniteInput);
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(YieldError::ProbabilityOutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const Z_95: f64 = 1.959_963_984_540_054;
    const CORPUS: &str =
        include_str!("../../../docs/engineering/microdisplay-yield-corpus-v0.1.json");
    const SCHEMA: &str =
        include_str!("../../../docs/engineering/microdisplay-yield-corpus.schema.json");

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn checked_in_corpus_remains_unqualified_and_arithmetic_cases_drive_reference() {
        let corpus: Value =
            serde_json::from_str(CORPUS).expect("checked-in yield corpus JSON must parse");
        let schema: Value =
            serde_json::from_str(SCHEMA).expect("checked-in yield schema JSON must parse");
        let validator = jsonschema::draft202012::new(&schema)
            .expect("checked-in yield schema must compile as Draft 2020-12");
        let schema_errors: Vec<String> = validator
            .iter_errors(&corpus)
            .map(|error| error.to_string())
            .collect();
        assert!(
            schema_errors.is_empty(),
            "microdisplay corpus does not satisfy its schema: {schema_errors:#?}"
        );
        assert_eq!(
            corpus["corpus_id"].as_str(),
            Some("MICRODISP-YIELD-CORPUS-001")
        );
        assert_eq!(corpus["status"].as_str(), Some("UNQUALIFIED"));
        assert_eq!(
            schema["$schema"].as_str(),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
        assert_eq!(
            schema["properties"]["corpus_id"]["const"].as_str(),
            Some("MICRODISP-YIELD-CORPUS-001")
        );
        assert_eq!(
            schema["properties"]["case_count"]["const"].as_u64(),
            Some(20)
        );
        assert_eq!(schema["properties"]["cases"]["minItems"].as_u64(), Some(20));
        assert_eq!(schema["properties"]["cases"]["maxItems"].as_u64(), Some(20));
        assert_eq!(
            schema["properties"]["cases"]["items"]["properties"]["id"]["pattern"].as_str(),
            Some("^MICLED-[A-Z0-9-]+$")
        );

        let cases = corpus["cases"]
            .as_array()
            .expect("corpus cases must be an array");
        assert_eq!(corpus["case_count"].as_u64(), Some(cases.len() as u64));
        assert_eq!(cases.len(), 20);

        let known_sources: std::collections::BTreeSet<String> = corpus["references"]
            .as_array()
            .expect("reference sources must be an array")
            .iter()
            .map(|source| {
                source["id"]
                    .as_str()
                    .expect("source ID must be a string")
                    .to_owned()
            })
            .collect();
        let schema_sources: std::collections::BTreeSet<String> =
            schema["properties"]["cases"]["items"]["properties"]["reference_ids"]["items"]["enum"]
                .as_array()
                .expect("schema must enumerate reference IDs")
                .iter()
                .map(|id| {
                    id.as_str()
                        .expect("schema reference ID must be a string")
                        .to_owned()
                })
                .collect();
        assert_eq!(known_sources, schema_sources);

        let mut ids = std::collections::BTreeSet::new();
        for case in cases {
            assert_eq!(case["status"].as_str(), Some("UNQUALIFIED"));
            assert_eq!(case["evidence_class"].as_str(), Some("SYNTHETIC_MODEL"));
            let id = case["id"].as_str().expect("each case must have an ID");
            assert!(id.starts_with("MICLED-"), "case ID prefix mismatch: {id}");
            assert!(ids.insert(id.to_owned()), "duplicate case ID: {id}");
            let references = case["reference_ids"]
                .as_array()
                .expect("each case must have source references");
            assert!(!references.is_empty());
            for reference in references {
                assert!(
                    known_sources
                        .contains(reference.as_str().expect("reference ID must be a string")),
                    "unresolved reference in case {id}"
                );
            }
        }

        for id in ["MICLED-YIELD-ARITH-001", "MICLED-YIELD-ARITH-002"] {
            let case = cases
                .iter()
                .find(|case| case["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("required arithmetic case is missing: {id}"));
            let population = case["inputs"]["population_count"]
                .as_u64()
                .expect("population_count must be an integer");
            let probability = case["inputs"]["per_emitter_success_probability"]
                .as_f64()
                .expect("success probability must be numeric");
            let output = expected_counts(population, probability)
                .expect("corpus arithmetic inputs must be valid");
            assert!(output.expected_defects.is_finite());
            assert!(output.expected_good.is_finite());
            assert!(
                (output.expected_good + output.expected_defects - population as f64).abs() < 1e-6
            );
        }
    }

    #[test]
    fn literature_scale_arithmetic_is_reproduced_without_promoting_it_to_a_result() {
        let conventional = expected_counts(100_000_000, 0.9999).expect("valid inputs");
        close(conventional.expected_defects, 10_000.0, 1e-6);
        close(conventional.expected_good, 99_990_000.0, 1e-7);

        let frontier = expected_counts(100_000_000, 0.999_999_9).expect("valid inputs");
        close(frontier.expected_defects, 10.0, 1e-6);
        close(frontier.expected_good, 99_999_990.0, 1e-7);
        assert_eq!(frontier.population, 100_000_000);
        // These are expectations under a declared model, not measured counts.
    }

    #[test]
    fn exact_zero_and_perfect_probabilities_have_canonical_expected_counts() {
        assert_eq!(
            expected_counts(100, 1.0)
                .expect("valid probability")
                .expected_defects,
            0.0
        );
        assert_eq!(
            expected_counts(100, 0.0)
                .expect("valid probability")
                .expected_good,
            0.0
        );
    }

    #[test]
    fn rejects_zero_population_and_invalid_probabilities() {
        assert_eq!(expected_counts(0, 0.9), Err(YieldError::ZeroDenominator));
        for value in [-0.01, 1.01] {
            assert_eq!(
                expected_counts(10, value),
                Err(YieldError::ProbabilityOutOfRange)
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(expected_counts(10, value), Err(YieldError::NonFiniteInput));
        }
    }

    #[test]
    fn observed_yield_requires_valid_counts() {
        assert_eq!(
            observed_yield(ObservedCounts {
                successes: 0,
                trials: 0
            }),
            Err(YieldError::ZeroDenominator)
        );
        assert_eq!(
            observed_yield(ObservedCounts {
                successes: 11,
                trials: 10
            }),
            Err(YieldError::SuccessesExceedTrials)
        );
        close(
            observed_yield(ObservedCounts {
                successes: 997,
                trials: 1000,
            })
            .expect("valid counts"),
            0.997,
            1e-15,
        );
    }

    #[test]
    fn wilson_interval_stays_bounded_and_contains_observed_proportion() {
        let counts = ObservedCounts {
            successes: 997,
            trials: 1000,
        };
        let interval = wilson_interval(counts, Z_95).expect("valid counts and z-score");
        assert!((0.0..=1.0).contains(&interval.lower));
        assert!((0.0..=1.0).contains(&interval.upper));
        assert!(interval.lower <= 0.997 && 0.997 <= interval.upper);
        assert!(interval.lower < interval.upper);
        close(
            wilson_interval(
                ObservedCounts {
                    successes: 10,
                    trials: 10,
                },
                Z_95,
            )
            .expect("valid")
            .upper,
            1.0,
            1e-15,
        );
    }

    #[test]
    fn wilson_interval_rejects_invalid_inputs() {
        assert_eq!(
            wilson_interval(
                ObservedCounts {
                    successes: 0,
                    trials: 0
                },
                Z_95
            ),
            Err(YieldError::ZeroDenominator)
        );
        assert_eq!(
            wilson_interval(
                ObservedCounts {
                    successes: 2,
                    trials: 1
                },
                Z_95
            ),
            Err(YieldError::SuccessesExceedTrials)
        );
        assert_eq!(
            wilson_interval(
                ObservedCounts {
                    successes: 1,
                    trials: 1
                },
                0.0
            ),
            Err(YieldError::InvalidZScore)
        );
        assert_eq!(
            wilson_interval(
                ObservedCounts {
                    successes: 1,
                    trials: 1
                },
                f64::NAN
            ),
            Err(YieldError::NonFiniteInput)
        );
    }

    #[test]
    fn stage_composition_requires_independence_to_be_explicit() {
        assert_eq!(
            compose_stage_yields(&[0.99, 0.98], DependenceAssumption::CorrelatedOrUnknown),
            Err(YieldError::IndependenceNotEstablished)
        );
        close(
            compose_stage_yields(&[0.99, 0.98], DependenceAssumption::Independent)
                .expect("explicit assumption"),
            0.9702,
            1e-15,
        );
        assert_eq!(
            compose_stage_yields(&[], DependenceAssumption::Independent),
            Err(YieldError::ZeroDenominator)
        );
        assert_eq!(
            compose_stage_yields(&[0.9, 1.1], DependenceAssumption::Independent),
            Err(YieldError::ProbabilityOutOfRange)
        );
    }

    #[test]
    fn repair_adjusted_yield_is_separate_from_first_pass_yield() {
        close(
            repair_adjusted_yield(0.9, 0.8, 0.95).expect("valid probabilities"),
            0.9 + 0.1 * 0.8 * 0.95,
            1e-15,
        );
        assert_eq!(
            repair_adjusted_yield(0.9, f64::NAN, 0.95),
            Err(YieldError::NonFiniteInput)
        );
    }

    #[test]
    fn optical_quality_floors_can_make_a_higher_yield_candidate_infeasible() {
        let candidate_quality = [0.99, 0.91, 0.98, 0.92];
        close(
            fraction_meeting_floor(&candidate_quality, 0.95).expect("valid metric values"),
            0.5,
            1e-15,
        );
        assert_eq!(
            fraction_meeting_floor(&[], 0.9),
            Err(YieldError::ZeroDenominator)
        );
        assert_eq!(
            fraction_meeting_floor(&[0.9, f64::NAN], 0.8),
            Err(YieldError::NonFiniteInput)
        );
    }
}
