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
const CALIBRATION_PER_NOISE: usize = 32;
const TEST_PER_NOISE: usize = 64;
const CALIBRATION_UNKNOWN: usize = 64;
const TEST_UNKNOWN: usize = 128;
const MAX_VALIDATION_RISK: f64 = 0.05;
const MIN_ACCEPTED_FOR_CALIBRATION: usize = 30;
const MIN_ACCEPTED_FOR_CERTIFICATION: usize = 52;
const BONFERRONI_Z_ONE_SIDED_95: f64 = 2.128_045_234_184_984;
const NOISE_LEVELS: [f64; 7] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50];
const SEEDS: [u64; 3] = [17, 29, 41];

#[derive(Debug, Clone)]
struct ScoredQuery {
    best_similarity: f64,
    top_two_margin: f64,
    correct: bool,
}

#[derive(Debug, Clone, Copy)]
struct DecisionRule {
    min_similarity: f64,
    min_margin: f64,
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
        self.accepted as f64 / self.total as f64
    }

    fn selective_risk(self) -> Option<f64> {
        (self.accepted > 0).then_some(self.errors as f64 / self.accepted as f64)
    }

    fn raw_accuracy(self) -> f64 {
        self.raw_correct as f64 / self.total as f64
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    println!("# protocol=selective-retrieval-v1");
    println!("# calibration_rule=selection_split_max_coverage_then_independent_certification");
    println!("# certification=one_sided_wilson_upper_bound_bonferroni_adjusted_for_3_rules");
    println!("# query_mix=known-noise+random-ood+two-prototype-mixture");
    println!(
        "seed,strategy,dimension,prototypes,selection_n,certification_n,certified,test_n,min_similarity,min_margin,accepted,coverage,errors,selective_risk,risk95_low,risk95_high,top1_accuracy"
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
            CALIBRATION_PER_NOISE,
            CALIBRATION_UNKNOWN,
            CALIBRATION_UNKNOWN,
        )?;
        let certification_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x4345_5254, 0),
            CALIBRATION_PER_NOISE,
            CALIBRATION_UNKNOWN,
            CALIBRATION_UNKNOWN,
        )?;
        let test_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x5445_5354, 0),
            TEST_PER_NOISE,
            TEST_UNKNOWN,
            TEST_UNKNOWN,
        )?;
        let selection = score_queries(&selection_queries, &prototypes)?;
        let certification = score_queries(&certification_queries, &prototypes)?;
        let test = score_queries(&test_queries, &prototypes)?;

        let always_accept = DecisionRule {
            min_similarity: -1.0,
            min_margin: -1.0,
        };
        report_row(
            seed,
            "always_top1",
            always_accept,
            0,
            0,
            "not_applicable",
            &test,
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
            let certified = certify_rule(&certification, rule);
            report_row(
                seed,
                name,
                rule,
                selection.len(),
                certification.len(),
                if certified { "pass" } else { "fail" },
                &test,
            );
        }
    }

    println!("# wall_ms={}", start.elapsed().as_millis());
    Ok(())
}

fn generate_queries(
    prototypes: &[PatternVector],
    seed: u64,
    positives_per_noise: usize,
    unknown_count: usize,
    mixture_count: usize,
) -> Result<Vec<(PatternVector, Option<usize>)>, Box<dyn Error>> {
    let mut queries = Vec::new();

    for (noise_index, noise) in NOISE_LEVELS.iter().copied().enumerate() {
        for sample_index in 0..positives_per_noise {
            let target = (sample_index * 13 + noise_index * 17) % prototypes.len();
            let query = prototypes[target].with_noise(
                noise,
                derive_seed(seed, noise_index as u64, sample_index as u64),
            )?;
            queries.push((query, Some(target)));
        }
    }

    for index in 0..unknown_count {
        queries.push((
            PatternVector::seeded_random(
                prototypes[0].dimension(),
                derive_seed(seed, 0x4f4f_4421, index as u64),
            )?,
            None,
        ));
    }

    for index in 0..mixture_count {
        let left = index % prototypes.len();
        let offset = 1 + (index * 17) % (prototypes.len() - 1);
        let right = (left + offset) % prototypes.len();
        let mixture = PatternVector::bundle(
            &[prototypes[left].clone(), prototypes[right].clone()],
            derive_seed(seed, 0x4d49_5854, index as u64),
        )?;
        // A mixture is intentionally ambiguous: accepting either component as
        // a unique answer is counted as a selective-classification error.
        queries.push((mixture, None));
    }

    Ok(queries)
}

fn score_queries(
    queries: &[(PatternVector, Option<usize>)],
    prototypes: &[PatternVector],
) -> Result<Vec<ScoredQuery>, Box<dyn Error>> {
    let mut scored = Vec::with_capacity(queries.len());

    for (query, target) in queries {
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
            correct: target.is_some_and(|target_index| target_index == best_index),
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

fn certify_rule(samples: &[ScoredQuery], rule: DecisionRule) -> bool {
    let result = evaluate(samples, rule);
    result.accepted >= MIN_ACCEPTED_FOR_CERTIFICATION
        && wilson_upper_bonferroni_95(result.errors, result.accepted)
            <= MAX_VALIDATION_RISK
}

fn report_row(
    seed: u64,
    name: &str,
    rule: DecisionRule,
    selection_n: usize,
    certification_n: usize,
    certified: &str,
    samples: &[ScoredQuery],
) {
    let result = evaluate(samples, rule);
    let (risk_low, risk_high) = wilson_interval_95(result.errors, result.accepted);
    let risk_text = result
        .selective_risk()
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".to_owned());

    println!(
        "{seed},{name},{DIMENSION},{PROTOTYPE_COUNT},{selection_n},{certification_n},{certified},{},{:.6},{:.6},{},{:.6},{},{},{:.6},{:.6},{:.6}",
        result.total,
        rule.min_similarity,
        rule.min_margin,
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

/// Per-rule certification uses a Bonferroni-adjusted one-sided bound for the
/// three candidate threshold policies compared per seed.
fn wilson_upper_bonferroni_95(errors: usize, accepted: usize) -> f64 {
    wilson_upper_bound(errors, accepted, BONFERRONI_Z_ONE_SIDED_95)
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
