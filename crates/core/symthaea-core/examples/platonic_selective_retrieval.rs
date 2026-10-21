// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibrate an HDC retriever to abstain on low-confidence and ambiguous queries.
//!
//! This is a selective-classification benchmark, not a learning or SOTA claim.
//! Thresholds are selected on a calibration split, then evaluated on independently
//! generated queries. Unknown random patterns and mixtures of stored patterns
//! have no single correct label and count as errors whenever accepted.

use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 4_096;
const PROTOTYPE_COUNT: usize = 64;
const SELECTION_QUERY_COUNT: usize = 2_048;
const CERTIFICATION_QUERY_COUNT: usize = 2_048;
const TEST_QUERY_COUNT: usize = 4_096;
const KNOWN_QUERY_WEIGHT: usize = 7;
const OOD_QUERY_WEIGHT: usize = 2;
const MIXTURE_QUERY_WEIGHT: usize = 2;
const MAX_VALIDATION_RISK: f64 = 0.05;
const MIN_ACCEPTED_FOR_CALIBRATION: usize = 30;
// Family-wise error control across 3 policy candidates × 5 reported seeds.
const MIN_ACCEPTED_FOR_CERTIFICATION: usize = 112;
const CERTIFICATION_ALPHA_PER_RULE: f64 = 0.05 / 15.0;
const NOISE_LEVELS: [f64; 7] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50];
// Novel unregistered templates deliberately correlated with known prototypes.
// This is a separate identifiability challenge, not part of risk certification.
const NEAR_UNKNOWN_DISTANCES: [f64; 3] = [0.05, 0.10, 0.20];
const NEAR_UNKNOWN_PER_DISTANCE: usize = 128;
const SEEDS: [u64; 5] = [17, 29, 41, 53, 67];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryKind {
    Known { noise_index: usize },
    RandomOod,
    AmbiguousMixture,
    NearPrototypeUnknown { distance_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuerySample {
    vector: PatternVector,
    target: Option<usize>,
    kind: QueryKind,
}

#[derive(Debug, Clone)]
struct ScoredQuery {
    best_similarity: f64,
    top_two_margin: f64,
    correct: bool,
    kind: QueryKind,
}

#[derive(Debug, Clone, Copy)]
struct DecisionRule {
    min_similarity: f64,
    min_margin: f64,
}

#[derive(Debug, Clone, Copy)]
struct CertificationResult {
    accepted: usize,
    errors: usize,
    upper_risk: f64,
    passed: bool,
}

#[derive(Debug, Clone, Copy)]
struct Evaluation {
    accepted: usize,
    errors: usize,
    raw_correct: usize,
    total: usize,
}

impl Evaluation {
    fn coverage(self) -> f64 {
        if self.total == 0 { 0.0 } else { self.accepted as f64 / self.total as f64 }
    }

    fn selective_risk(self) -> Option<f64> {
        (self.accepted > 0).then_some(self.errors as f64 / self.accepted as f64)
    }

    fn raw_accuracy(self) -> f64 {
        if self.total == 0 { 0.0 } else { self.raw_correct as f64 / self.total as f64 }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    println!("# protocol=selective-retrieval-v1");
    println!("# calibration_rule=selection_split_max_coverage_then_independent_certification");
    println!("# certification=one_sided_clopper_pearson_upper_bound_bonferroni_adjusted_for_3_rules_x_5_seeds");
    println!("# query_mix_weights=known:random_ood:ambiguous_mixture=7:2:2");
    println!("# known_query_noise_level=uniform_over_0,0.10,0.20,0.30,0.40,0.45,0.50");
    println!("# separate_near_unknown_challenge_distances=0.05,0.10,0.20");
    println!("# near_unknown_challenge_is_not_in_certification_distribution=true");
    println!(
        "record_type,seed,strategy,query_kind,noise_fraction,dimension,prototypes,selection_n,certification_n,certified,cert_accept,cert_errors,cert_risk_upper95,test_n,min_similarity,min_margin,n,accepted,coverage,errors,selective_risk,risk95_low,risk95_high,top1_accuracy"
    );

    for seed in SEEDS {
        let prototypes = (0..PROTOTYPE_COUNT)
            .map(|index| {
                PatternVector::seeded_random(
                    DIMENSION,
                    derive_seed(seed, 0x5052_4f54, index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Separate threshold selection, independent risk certification, and final evaluation.
        // The certification split is never used to choose the thresholds.
        let selection_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x5345_4c45, 0),
            SELECTION_QUERY_COUNT,
        )?;
        let certification_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x4345_5254, 0),
            CERTIFICATION_QUERY_COUNT,
        )?;
        let test_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x5445_5354, 0),
            TEST_QUERY_COUNT,
        )?;
        let selection = score_queries(&selection_queries, &prototypes)?;
        let certification = score_queries(&certification_queries, &prototypes)?;
        let test = score_queries(&test_queries, &prototypes)?;
        let near_unknown_queries = generate_near_unknown_challenges(
            &prototypes,
            derive_seed(seed, 0x4e45_4152, 0),
            NEAR_UNKNOWN_PER_DISTANCE,
        )?;
        let near_unknown = score_queries(&near_unknown_queries, &prototypes)?;

        let always_accept = DecisionRule {
            min_similarity: -1.0,
            min_margin: -1.0,
        };
        report_row(
            "evaluation",
            seed,
            "always_top1",
            always_accept,
            0,
            0,
            None,
            &test,
        );
        report_row(
            "near_unknown_challenge",
            seed,
            "always_top1",
            always_accept,
            0,
            0,
            None,
            &near_unknown,
        );

        for (name, rule) in [
            (
                "calibrated_score_only",
                calibrate_rule(&selection, CalibrationMode::ScoreOnly),
            ),
            (
                "calibrated_margin_only",
                calibrate_rule(&selection, CalibrationMode::MarginOnly),
            ),
            (
                "calibrated_score_and_margin",
                calibrate_rule(&selection, CalibrationMode::ScoreAndMargin),
            ),
        ] {
            let certificate = certify_rule(&certification, rule);
            report_row(
                "evaluation",
                seed,
                name,
                rule,
                selection.len(),
                certification.len(),
                Some(certificate),
                &test,
            );
            report_row(
                "near_unknown_challenge",
                seed,
                name,
                rule,
                0,
                0,
                None,
                &near_unknown,
            );
        }
    }

    println!("# wall_ms={}", start.elapsed().as_millis());
    Ok(())
}

fn generate_near_unknown_challenges(
    prototypes: &[PatternVector],
    seed: u64,
    samples_per_distance: usize,
) -> Result<Vec<QuerySample>, Box<dyn Error>> {
    if prototypes.len() < 2 {
        return Err("near-unknown challenge requires at least two prototypes".into());
    }
    let mut rng_state = seed;
    let mut queries = Vec::with_capacity(samples_per_distance * NEAR_UNKNOWN_DISTANCES.len());
    for (distance_index, distance) in NEAR_UNKNOWN_DISTANCES.iter().copied().enumerate() {
        for _ in 0..samples_per_distance {
            let source = sample_bounded(&mut rng_state, prototypes.len());
            let query = prototypes[source].with_noise(distance, next_u64(&mut rng_state))?;
            // This template is declared as a novel, unregistered item even though
            // it lies close to a stored prototype. Its vector can overlap the
            // valid noisy-known distribution; rejecting it reliably from vector
            // features alone may be impossible. This red-team split is excluded
            // from calibration and certification by design.
            queries.push(QuerySample {
                vector: query,
                target: None,
                kind: QueryKind::NearPrototypeUnknown { distance_index },
            });
        }
    }
    Ok(queries)
}

fn generate_queries(
    prototypes: &[PatternVector],
    seed: u64,
    query_count: usize,
) -> Result<Vec<QuerySample>, Box<dyn Error>> {
    if prototypes.len() < 2 {
        return Err("selective retrieval requires at least two prototypes".into());
    }

    // Draw each query independently from a fixed synthetic mixture: 7/11
    // known-pattern queries, 2/11 random OOD, and 2/11 ambiguous mixtures.
    // Known queries sample both the target prototype and corruption level
    // uniformly; this makes the pooled risk gate meaningful for this declared
    // mixture, instead of treating a hand-balanced set of strata as iid.
    let query_type_count = KNOWN_QUERY_WEIGHT + OOD_QUERY_WEIGHT + MIXTURE_QUERY_WEIGHT;
    let mut rng_state = seed;
    let mut queries = Vec::with_capacity(query_count);

    for _ in 0..query_count {
        let query_type = sample_bounded(&mut rng_state, query_type_count);
        if query_type < KNOWN_QUERY_WEIGHT {
            let target = sample_bounded(&mut rng_state, prototypes.len());
            let noise_index = sample_bounded(&mut rng_state, NOISE_LEVELS.len());
            let query = prototypes[target].with_noise(
                NOISE_LEVELS[noise_index],
                next_u64(&mut rng_state),
            )?;
            queries.push(QuerySample {
                vector: query,
                target: Some(target),
                kind: QueryKind::Known { noise_index },
            });
        } else if query_type < KNOWN_QUERY_WEIGHT + OOD_QUERY_WEIGHT {
            let query = PatternVector::seeded_random(
                prototypes[0].dimension(),
                next_u64(&mut rng_state),
            )?;
            queries.push(QuerySample {
                vector: query,
                target: None,
                kind: QueryKind::RandomOod,
            });
        } else {
            let left = sample_bounded(&mut rng_state, prototypes.len());
            let offset = 1 + sample_bounded(&mut rng_state, prototypes.len() - 1);
            let right = (left + offset) % prototypes.len();
            let mixture = PatternVector::bundle(
                &[prototypes[left].clone(), prototypes[right].clone()],
                next_u64(&mut rng_state),
            )?;
            // A mixture is intentionally ambiguous: accepting either component
            // as a unique answer is counted as a selective-classification error.
            queries.push(QuerySample {
                vector: mixture,
                target: None,
                kind: QueryKind::AmbiguousMixture,
            });
        }
    }

    Ok(queries)
}

fn sample_bounded(state: &mut u64, upper_exclusive: usize) -> usize {
    let bound = upper_exclusive as u64;
    assert!(bound > 0, "sampling bound must be non-zero");
    // Rejection sampling avoids the modulo bias of simply reducing every u64.
    let rejection_floor = bound.wrapping_neg() % bound;
    loop {
        let value = next_u64(state);
        if value >= rejection_floor {
            return (value % bound) as usize;
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mix64(*state)
}

fn score_queries(
    queries: &[QuerySample],
    prototypes: &[PatternVector],
) -> Result<Vec<ScoredQuery>, Box<dyn Error>> {
    let mut scored = Vec::with_capacity(queries.len());

    for sample in queries {
        let query = &sample.vector;
        let mut best_index = 0;
        let mut best = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;

        for (index, prototype) in prototypes.iter().enumerate() {
            let similarity = query.similarity(prototype)?;
            if similarity > best {
                second = best;
                best = similarity;
                best_index = index;
            } else if similarity > second {
                second = similarity;
            }
        }

        scored.push(ScoredQuery {
            best_similarity: best,
            top_two_margin: best - second,
            correct: sample.target.is_some_and(|target_index| target_index == best_index),
            kind: sample.kind,
        });
    }

    Ok(scored)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalibrationMode {
    ScoreOnly,
    MarginOnly,
    ScoreAndMargin,
}

fn calibrate_rule(samples: &[ScoredQuery], mode: CalibrationMode) -> DecisionRule {
    let score_thresholds = thresholds(samples, |sample| sample.best_similarity);
    let margin_thresholds = thresholds(samples, |sample| sample.top_two_margin);
    let score_candidates: &[f64] = match mode {
        CalibrationMode::MarginOnly => &[-1.0],
        CalibrationMode::ScoreOnly | CalibrationMode::ScoreAndMargin => &score_thresholds,
    };
    let margin_candidates: &[f64] = match mode {
        CalibrationMode::ScoreOnly => &[-1.0],
        CalibrationMode::MarginOnly | CalibrationMode::ScoreAndMargin => &margin_thresholds,
    };

    let mut best_rule = DecisionRule {
        min_similarity: f64::INFINITY,
        min_margin: f64::INFINITY,
    };
    let mut best_accepted = 0;
    let mut best_risk_upper = f64::INFINITY;

    for &min_similarity in score_candidates {
        for &min_margin in margin_candidates {
            let rule = DecisionRule {
                min_similarity,
                min_margin,
            };
            let evaluation = evaluate(samples, rule);
            if evaluation.accepted < MIN_ACCEPTED_FOR_CALIBRATION {
                continue;
            }
            let risk_upper = wilson_upper_95(evaluation.errors, evaluation.accepted);
            if risk_upper > MAX_VALIDATION_RISK {
                continue;
            }
            if evaluation.accepted > best_accepted
                || (evaluation.accepted == best_accepted && risk_upper < best_risk_upper)
            {
                best_rule = rule;
                best_accepted = evaluation.accepted;
                best_risk_upper = risk_upper;
            }
        }
    }

    best_rule
}

fn thresholds(samples: &[ScoredQuery], metric: fn(&ScoredQuery) -> f64) -> Vec<f64> {
    let mut values: Vec<f64> = samples.iter().map(metric).collect();
    values.push(-1.0);
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| left.total_cmp(right).is_eq());

    if values.len() <= 21 {
        return values;
    }

    let last = values.len() - 1;
    let mut candidates: Vec<f64> = (0..=20)
        .map(|quantile| values[quantile * last / 20])
        .collect();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| left.total_cmp(right).is_eq());
    candidates
}

fn evaluate(samples: &[ScoredQuery], rule: DecisionRule) -> Evaluation {
    let mut accepted = 0;
    let mut errors = 0;
    let mut raw_correct = 0;

    for sample in samples {
        if sample.correct {
            raw_correct += 1;
        }
        let is_accepted = sample.best_similarity >= rule.min_similarity
            && sample.top_two_margin >= rule.min_margin;
        if is_accepted {
            accepted += 1;
            if !sample.correct {
                errors += 1;
            }
        }
    }

    Evaluation {
        accepted,
        errors,
        raw_correct,
        total: samples.len(),
    }
}

fn certify_rule(samples: &[ScoredQuery], rule: DecisionRule) -> CertificationResult {
    let result = evaluate(samples, rule);
    let upper_risk = clopper_pearson_upper_bound(
        result.errors,
        result.accepted,
        CERTIFICATION_ALPHA_PER_RULE,
    );
    CertificationResult {
        accepted: result.accepted,
        errors: result.errors,
        upper_risk,
        passed: result.accepted >= MIN_ACCEPTED_FOR_CERTIFICATION
            && upper_risk <= MAX_VALIDATION_RISK,
    }
}

fn report_row(
    report_family: &str,
    seed: u64,
    name: &str,
    rule: DecisionRule,
    selection_n: usize,
    certification_n: usize,
    certificate: Option<CertificationResult>,
    samples: &[ScoredQuery],
) {
    let (certified, cert_accepted, cert_errors, cert_upper_risk) =
        certificate.map_or(
            ("not_applicable", "NA".to_owned(), "NA".to_owned(), "NA".to_owned()),
            |certificate| (
                if certificate.passed { "pass" } else { "fail" },
                certificate.accepted.to_string(),
                certificate.errors.to_string(),
                format!("{:.6}", certificate.upper_risk),
            ),
        );

    write_evaluation_row(
        &format!("{report_family}_overall"),
        seed,
        name,
        "all",
        None,
        rule,
        selection_n,
        certification_n,
        certified,
        &cert_accepted,
        &cert_errors,
        &cert_upper_risk,
        samples.len(),
        evaluate(samples, rule),
    );

    for noise_index in 0..NOISE_LEVELS.len() {
        let kind = QueryKind::Known { noise_index };
        let subset: Vec<ScoredQuery> = samples
            .iter()
            .filter(|sample| sample.kind == kind)
            .cloned()
            .collect();
        if subset.is_empty() {
            continue;
        }
        write_evaluation_row(
            &format!("{report_family}_stratum"),
            seed,
            name,
            &format!("known_noise_{noise_index}"),
            Some(NOISE_LEVELS[noise_index]),
            rule,
            selection_n,
            certification_n,
            certified,
            &cert_accepted,
            &cert_errors,
            &cert_upper_risk,
            samples.len(),
            evaluate(&subset, rule),
        );
    }

    for (kind, label) in [
        (QueryKind::RandomOod, "random_ood"),
        (QueryKind::AmbiguousMixture, "ambiguous_mixture"),
    ] {
        let subset: Vec<ScoredQuery> = samples
            .iter()
            .filter(|sample| sample.kind == kind)
            .cloned()
            .collect();
        if subset.is_empty() {
            continue;
        }
        write_evaluation_row(
            &format!("{report_family}_stratum"),
            seed,
            name,
            label,
            None,
            rule,
            selection_n,
            certification_n,
            certified,
            &cert_accepted,
            &cert_errors,
            &cert_upper_risk,
            samples.len(),
            evaluate(&subset, rule),
        );
    }

    for (distance_index, distance) in NEAR_UNKNOWN_DISTANCES.iter().copied().enumerate() {
        let kind = QueryKind::NearPrototypeUnknown { distance_index };
        let subset: Vec<ScoredQuery> = samples
            .iter()
            .filter(|sample| sample.kind == kind)
            .cloned()
            .collect();
        if subset.is_empty() {
            continue;
        }
        write_evaluation_row(
            &format!("{report_family}_stratum"),
            seed,
            name,
            &format!("near_unknown_{distance_index}"),
            Some(distance),
            rule,
            selection_n,
            certification_n,
            certified,
            &cert_accepted,
            &cert_errors,
            &cert_upper_risk,
            samples.len(),
            evaluate(&subset, rule),
        );
    }
}

fn write_evaluation_row(
    record_type: &str,
    seed: u64,
    strategy: &str,
    query_kind: &str,
    noise_fraction: Option<f64>,
    rule: DecisionRule,
    selection_n: usize,
    certification_n: usize,
    certified: &str,
    cert_accepted: &str,
    cert_errors: &str,
    cert_upper_risk: &str,
    test_n: usize,
    result: Evaluation,
) {
    let (risk_low, risk_high) = wilson_interval_95(result.errors, result.accepted);
    let risk_text = result
        .selective_risk()
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".to_owned());
    let noise = noise_fraction
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "NA".to_owned());

    println!(
        "{record_type},{seed},{strategy},{query_kind},{noise},{DIMENSION},{PROTOTYPE_COUNT},{selection_n},{certification_n},{certified},{cert_accepted},{cert_errors},{cert_upper_risk},{test_n},{:.6},{:.6},{},{},{:.6},{},{},{:.6},{:.6},{:.6}",
        rule.min_similarity,
        rule.min_margin,
        result.total,
        result.accepted,
        result.coverage(),
        result.errors,
        risk_text,
        risk_low,
        risk_high,
        result.raw_accuracy()
    );
}

fn wilson_interval_95(errors: usize, accepted: usize) -> (f64, f64) {
    if accepted == 0 {
        return (0.0, 1.0);
    }
    let n = accepted as f64;
    let p = errors as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (p + z_squared / (2.0 * n)) / denominator;
    let half_width = z
        * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt()
        / denominator;
    (
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    )
}

fn wilson_upper_95(errors: usize, accepted: usize) -> f64 {
    wilson_upper_bound(errors, accepted, 1.644_853_626_951_472_2_f64)
}

/// Exact one-sided Clopper-Pearson upper bound for a binomial error rate.
/// The caller supplies per-policy alpha; certification uses alpha = 0.05 / 3
/// for the three candidate threshold policies compared per seed.
fn clopper_pearson_upper_bound(errors: usize, accepted: usize, alpha: f64) -> f64 {
    if accepted == 0 || errors >= accepted || !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return 1.0;
    }
    if errors == 0 {
        return 1.0 - alpha.powf(1.0 / accepted as f64);
    }

    // For k observed errors in n accepted queries, the one-sided upper bound
    // solves P_p[X <= k] = alpha. The binomial CDF is monotone in p, so bisection
    // gives a deterministic numerical solution without a statistics dependency.
    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..64 {
        let midpoint = (lower + upper) / 2.0;
        if binomial_cdf_at_most(errors, accepted, midpoint) > alpha {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    upper
}

fn binomial_cdf_at_most(errors: usize, trials: usize, probability: f64) -> f64 {
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return if errors >= trials { 1.0 } else { 0.0 };
    }

    let log_p = probability.ln();
    let log_q = (-probability).ln_1p();
    let mut log_choose = 0.0;
    let mut terms = Vec::with_capacity(errors + 1);
    for successes in 0..=errors {
        if successes > 0 {
            log_choose += ((trials - successes + 1) as f64 / successes as f64).ln();
        }
        terms.push(
            log_choose
                + successes as f64 * log_p
                + (trials - successes) as f64 * log_q,
        );
    }
    let max_log = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_log.is_finite() {
        return 0.0;
    }
    (max_log + terms.iter().map(|value| (value - max_log).exp()).sum::<f64>().ln())
        .exp()
        .clamp(0.0, 1.0)
}

fn wilson_upper_bound(errors: usize, accepted: usize, z: f64) -> f64 {
    if accepted == 0 {
        return 1.0;
    }
    let n = accepted as f64;
    let p = errors as f64 / n;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (p + z_squared / (2.0 * n)) / denominator;
    let half_width = z
        * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt()
        / denominator;
    (center + half_width).min(1.0)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_seed(seed: u64, first: u64, second: u64) -> u64 {
    mix64(seed ^ mix64(first) ^ mix64(second.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonferroni_exact_binomial_certificate_respects_sample_floor() {
        assert!(
            clopper_pearson_upper_bound(0, 111, CERTIFICATION_ALPHA_PER_RULE)
                > MAX_VALIDATION_RISK
        );
        assert!(
            clopper_pearson_upper_bound(0, 112, CERTIFICATION_ALPHA_PER_RULE)
                <= MAX_VALIDATION_RISK
        );
        assert!(
            clopper_pearson_upper_bound(1, 112, CERTIFICATION_ALPHA_PER_RULE)
                > MAX_VALIDATION_RISK
        );
    }

    #[test]
    fn certification_requires_minimum_accepted_examples_and_bounded_risk() {
        let rule = DecisionRule {
            min_similarity: 0.0,
            min_margin: 0.0,
        };
        let too_small = vec![
            ScoredQuery {
                best_similarity: 0.5,
                top_two_margin: 0.2,
                correct: true,
                kind: QueryKind::Known { noise_index: 0 },
            };
            MIN_ACCEPTED_FOR_CERTIFICATION - 1
        ];
        assert!(!certify_rule(&too_small, rule).passed);

        let enough_clean = vec![
            ScoredQuery {
                best_similarity: 0.5,
                top_two_margin: 0.2,
                correct: true,
                kind: QueryKind::Known { noise_index: 0 },
            };
            MIN_ACCEPTED_FOR_CERTIFICATION
        ];
        assert!(certify_rule(&enough_clean, rule).passed);
    }

    #[test]
    fn near_unknown_challenge_is_deterministic_and_unlabelled() {
        let prototypes = (0..8)
            .map(|index| PatternVector::seeded_random(256, 100 + index))
            .collect::<Result<Vec<_>, _>>()
            .expect("prototype bank");
        let first = generate_near_unknown_challenges(&prototypes, 0x4e45_4152, 8)
            .expect("challenge queries");
        let second = generate_near_unknown_challenges(&prototypes, 0x4e45_4152, 8)
            .expect("challenge repeat");

        assert_eq!(first, second);
        assert_eq!(first.len(), NEAR_UNKNOWN_DISTANCES.len() * 8);
        assert!(first.iter().all(|query| query.target.is_none()));
        for distance_index in 0..NEAR_UNKNOWN_DISTANCES.len() {
            assert_eq!(
                first
                    .iter()
                    .filter(|query| query.kind == QueryKind::NearPrototypeUnknown { distance_index })
                    .count(),
                8
            );
        }
    }

    #[test]
    fn accepted_unknown_queries_count_as_selective_errors() {
        let samples = vec![
            ScoredQuery {
                best_similarity: 0.9,
                top_two_margin: 0.2,
                correct: true,
                kind: QueryKind::Known { noise_index: 0 },
            },
            ScoredQuery {
                best_similarity: 0.8,
                top_two_margin: 0.1,
                correct: false,
                kind: QueryKind::RandomOod,
            },
        ];
        let result = evaluate(
            &samples,
            DecisionRule {
                min_similarity: -1.0,
                min_margin: -1.0,
            },
        );
        assert_eq!(result.accepted, 2);
        assert_eq!(result.errors, 1);
        assert_eq!(result.selective_risk(), Some(0.5));
    }
}

