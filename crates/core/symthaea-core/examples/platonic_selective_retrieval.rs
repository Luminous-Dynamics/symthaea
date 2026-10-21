// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibrate an HDC retriever to abstain on low-confidence and ambiguous queries.
//!
//! Compares selective classification with split-conformal, set-valued retrieval.
//! This is a synthetic benchmark, not a learning or SOTA claim.
//! Thresholds are selected on a calibration split, then evaluated on independently
//! generated queries. Unknown random patterns and mixtures of stored patterns
//! have no single correct label and count as errors whenever accepted.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 4_096;
const PROTOTYPE_COUNT: usize = 64;
const SELECTION_QUERY_COUNT: usize = 2_048;
const CERTIFICATION_QUERY_COUNT: usize = 2_048;
const TEST_QUERY_COUNT: usize = 4_096;
const CONFORMAL_CALIBRATION_QUERY_COUNT: usize = 2_048;
const KNOWN_QUERY_WEIGHT: usize = 7;
const OOD_QUERY_WEIGHT: usize = 2;
const MIXTURE_QUERY_WEIGHT: usize = 2;
const MAX_VALIDATION_RISK: f64 = 0.05;
const CONFORMAL_ALPHA: f64 = 0.05;
const DCO_GAP_PENALTY_CANDIDATES: [f64; 5] = [0.0, 0.25, 0.5, 1.0, 2.0];
const MIN_ACCEPTED_FOR_CALIBRATION: usize = 30;
// Family-wise error control across 3 policy candidates × 5 reported seeds.
const MIN_ACCEPTED_FOR_CERTIFICATION: usize = 125;
// Six threshold strategies × five seeds, controlling family-wise certification error.
const CERTIFICATION_ALPHA_PER_RULE: f64 = 0.05 / 30.0;
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
    /// Cosine similarity of the ground-truth label, when the query is labeled.
    target_similarity: Option<f64>,
    /// Cached similarity to each prototype, also reused for set-valued retrieval.
    prototype_similarities: Arc<[f64]>,
    kind: QueryKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConformalScoreMode {
    RawSimilarity,
    GapAdjusted { penalty: f64 },
}

#[derive(Debug, Clone, Copy)]
struct DcoScoreSelection {
    mode: ConformalScoreMode,
    provisional_quantile: f64,
    tuning_coverage: f64,
    mean_set_size: f64,
    tuning_n: usize,
}

#[derive(Debug, Clone, Copy)]
struct ConformalQueryResult {
    kind: QueryKind,
    labeled: bool,
    covered: bool,
    set_size: usize,
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
struct FixedSequenceCalibration {
    rule: DecisionRule,
    certificate: CertificationResult,
    certified_prefix_len: usize,
    candidate_count: usize,
    selected_candidate_index: Option<usize>,
    stopped_at_index: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct Evaluation {
    accepted: usize,
    /// Rejected because the best match failed its similarity threshold only.
    rejected_score_only: usize,
    /// Rejected because top-two separation failed its margin threshold only.
    rejected_margin_only: usize,
    /// Rejected because both score and separation thresholds failed.
    rejected_both: usize,
    errors: usize,
    raw_correct: usize,
    total: usize,
}

impl Evaluation {
    fn coverage(self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.accepted as f64 / self.total as f64
        }
    }

    fn selective_risk(self) -> Option<f64> {
        (self.accepted > 0).then_some(self.errors as f64 / self.accepted as f64)
    }

    fn raw_accuracy(self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.raw_correct as f64 / self.total as f64
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    println!("# protocol=selective-retrieval-v1");
    println!(
        "# calibration_rule=empirical_threshold_search_plus_independent_certification_and_fixed_sequence_CIC"
    );
    println!(
        "# certification=one_sided_clopper_pearson_upper_bound_bonferroni_adjusted_for_6_rules_x_5_seeds"
    );
    println!(
        "# fixed_sequence_rule=predeclared_strict_to_permissive_candidates_stop_at_first_failed_certificate"
    );
    println!("# fixed_sequence_uses_one_calibration_split=true");
    println!("# set_valued_baseline=split_conformal_and_dco_tuned_prediction_sets");
    println!("# conformal_alpha=0.05; marginal_known_query_coverage_target=0.95");
    println!("# dco_tuning_split=selection; fresh_conformal_calibration_split=true");
    println!(
        "# dco_tuning_objective=minimize_tuning_mean_set_size_subject_to_tuning_coverage_target"
    );
    println!("# dco_gap_penalty_candidates=0,0.25,0.5,1,2");
    println!(
        "# conformal_scope=exchangeable_known_queries_from_same_synthetic_generator; strata_are_descriptive"
    );
    println!("# query_mix_weights=known:random_ood:ambiguous_mixture=7:2:2");
    println!("# known_query_noise_level=uniform_over_0,0.10,0.20,0.30,0.40,0.45,0.50");
    println!("# separate_near_unknown_challenge_distances=0.05,0.10,0.20");
    println!("# near_unknown_challenge_is_not_in_certification_distribution=true");
    println!(
        "# identifiability_control=paired known/unknown labels share identical PatternVector; vector-only prediction must agree"
    );
    println!(
        "record_type,seed,strategy,query_kind,noise_fraction,dimension,prototypes,selection_n,certification_n,certified,cert_accept,cert_errors,cert_risk_upper95,test_n,min_similarity,min_margin,n,accepted,rejected_score_only,rejected_margin_only,rejected_both,coverage,errors,selective_risk,risk95_low,risk95_high,top1_accuracy"
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
        let conformal_calibration_queries = generate_queries(
            &prototypes,
            derive_seed(seed, 0x434f_4e46, 0),
            CONFORMAL_CALIBRATION_QUERY_COUNT,
        )?;
        let conformal_calibration = score_queries(&conformal_calibration_queries, &prototypes)?;
        let near_unknown_queries = generate_near_unknown_challenges(
            &prototypes,
            derive_seed(seed, 0x4e45_4152, 0),
            NEAR_UNKNOWN_PER_DISTANCE,
        )?;
        let near_unknown = score_queries(&near_unknown_queries, &prototypes)?;

        // Split conformal uses only the labeled known-query part of the selection
        // split. Its finite-sample guarantee is marginal over this generator's
        // known-query distribution; per-noise rows are descriptive, not conditional
        // coverage guarantees.
        // Baseline split conformal uses the raw similarity score, calibrated on
        // a fresh split that is disjoint from threshold/score tuning.
        let (conformal_q, conformal_calibration_n) = split_conformal_quantile(
            &conformal_calibration,
            CONFORMAL_ALPHA,
            ConformalScoreMode::RawSimilarity,
        )?;
        let conformal_cutoff_similarity = 1.0 - conformal_q;
        let conformal_test =
            score_conformal_sets(&test, conformal_q, ConformalScoreMode::RawSimilarity);
        let conformal_near_unknown = score_conformal_sets(
            &near_unknown,
            conformal_q,
            ConformalScoreMode::RawSimilarity,
        );
        println!(
            "# conformal_calibration=seed:{seed},alpha:{CONFORMAL_ALPHA:.4},q:{conformal_q:.6},cutoff_similarity:{conformal_cutoff_similarity:.6},known_calibration_n:{conformal_calibration_n},split:fresh"
        );
        report_conformal_set_summaries(
            seed,
            "evaluation_raw_conformal",
            CONFORMAL_ALPHA,
            conformal_q,
            conformal_calibration_n,
            &conformal_test,
        );
        report_conformal_set_summaries(
            seed,
            "near_unknown_raw_conformal",
            CONFORMAL_ALPHA,
            conformal_q,
            conformal_calibration_n,
            &conformal_near_unknown,
        );

        // DCO-style experiment: tune only the score structure on the selection
        // split, then compute the final quantile on fresh, independent examples.
        let dco = tune_dco_score(&selection, CONFORMAL_ALPHA)?;
        let (dco_q, dco_calibration_n) =
            split_conformal_quantile(&conformal_calibration, CONFORMAL_ALPHA, dco.mode)?;
        let dco_test = score_conformal_sets(&test, dco_q, dco.mode);
        let dco_near_unknown = score_conformal_sets(&near_unknown, dco_q, dco.mode);
        let dco_mode_label = match dco.mode {
            ConformalScoreMode::RawSimilarity => "raw_similarity".to_owned(),
            ConformalScoreMode::GapAdjusted { penalty } => {
                format!("gap_adjusted_{penalty:.2}")
            }
        };
        println!(
            "# dco_selection=seed:{seed},score:{dco_mode_label},provisional_q:{:.6},tuning_n:{},tuning_known_coverage:{:.6},tuning_mean_set_size:{:.6},fresh_calibration_n:{dco_calibration_n},fresh_q:{dco_q:.6}",
            dco.provisional_quantile, dco.tuning_n, dco.tuning_coverage, dco.mean_set_size
        );
        report_conformal_set_summaries(
            seed,
            "evaluation_dco_tuned",
            CONFORMAL_ALPHA,
            dco_q,
            dco_calibration_n,
            &dco_test,
        );
        report_conformal_set_summaries(
            seed,
            "near_unknown_dco_tuned",
            CONFORMAL_ALPHA,
            dco_q,
            dco_calibration_n,
            &dco_near_unknown,
        );

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
        let mut identity_policies = vec![("always_top1".to_owned(), always_accept)];

        for (mode_name, mode) in [
            ("score_only", CalibrationMode::ScoreOnly),
            ("margin_only", CalibrationMode::MarginOnly),
            ("score_and_margin", CalibrationMode::ScoreAndMargin),
        ] {
            // Baseline A: use the selection split to optimize thresholds and the
            // independent certification split to verify the chosen point.
            let baseline_name = format!("calibrated_{mode_name}");
            let rule = calibrate_rule(&selection, mode);
            let certificate = certify_rule(&certification, rule);
            report_row(
                "evaluation",
                seed,
                &baseline_name,
                rule,
                selection.len(),
                certification.len(),
                Some(certificate),
                &test,
            );
            report_row(
                "near_unknown_challenge",
                seed,
                &baseline_name,
                rule,
                0,
                0,
                None,
                &near_unknown,
            );
            identity_policies.push((baseline_name.clone(), rule));

            // Baseline B: fixed-sequence confidence-interval calibration (CIC).
            // The score/margin threshold schedule is fixed before observing the
            // calibration sample. One sample both certifies each candidate and
            // selects the last point in the consecutively certified prefix.
            let cic = calibrate_fixed_sequence_rule(&selection, mode);
            let stop_at = cic
                .stopped_at_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "none".to_owned());
            println!(
                "# fixed_sequence_seed={seed},policy={mode_name},certified_prefix={},candidate_count={},stopped_at_index={stop_at},selected_index={}",
                cic.certified_prefix_len,
                cic.candidate_count,
                cic.selected_candidate_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "none".to_owned())
            );

            let cic_name = format!("fixed_sequence_{mode_name}");
            report_row(
                "evaluation_cic",
                seed,
                &cic_name,
                cic.rule,
                0,
                selection.len(),
                Some(cic.certificate),
                &test,
            );
            report_row(
                "near_unknown_challenge_cic",
                seed,
                &cic_name,
                cic.rule,
                0,
                0,
                None,
                &near_unknown,
            );
            identity_policies.push((cic_name.clone(), cic.rule));
            report_risk_coverage_curve(seed, mode_name, mode, &test);
        }
        report_identity_collision_controls(seed, &prototypes, &identity_policies)?;
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
            let query = prototypes[target]
                .with_noise(NOISE_LEVELS[noise_index], next_u64(&mut rng_state))?;
            queries.push(QuerySample {
                vector: query,
                target: Some(target),
                kind: QueryKind::Known { noise_index },
            });
        } else if query_type < KNOWN_QUERY_WEIGHT + OOD_QUERY_WEIGHT {
            let query =
                PatternVector::seeded_random(prototypes[0].dimension(), next_u64(&mut rng_state))?;
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
        let mut prototype_similarities = Vec::with_capacity(prototypes.len());

        for (index, prototype) in prototypes.iter().enumerate() {
            let similarity = query.similarity(prototype)?;
            prototype_similarities.push(similarity);
            if similarity > best {
                second = best;
                best = similarity;
                best_index = index;
            } else if similarity > second {
                second = similarity;
            }
        }

        let target_similarity = match sample.target {
            Some(target_index) => Some(
                query.similarity(
                    prototypes
                        .get(target_index)
                        .ok_or("query target index is outside the prototype bank")?,
                )?,
            ),
            None => None,
        };
        scored.push(ScoredQuery {
            best_similarity: best,
            top_two_margin: best - second,
            correct: sample
                .target
                .is_some_and(|target_index| target_index == best_index),
            target_similarity,
            prototype_similarities: Arc::from(prototype_similarities),
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

/// A predeclared threshold grid in strict-to-permissive order. Each next
/// candidate relaxes (never tightens) its acceptance rule, so the accepted sets
/// are nested. Threshold values do not depend on the observed calibration data.
fn fixed_sequence_candidates(mode: CalibrationMode) -> Vec<DecisionRule> {
    match mode {
        CalibrationMode::ScoreOnly => (0..=100)
            .map(|step| DecisionRule {
                min_similarity: 1.0 - 2.0 * step as f64 / 100.0,
                min_margin: -1.0,
            })
            .collect(),
        CalibrationMode::MarginOnly => (0..=100)
            .map(|step| DecisionRule {
                min_similarity: -1.0,
                min_margin: 0.80 * (1.0 - step as f64 / 100.0),
            })
            .collect(),
        CalibrationMode::ScoreAndMargin => (0..=100)
            .map(|step| DecisionRule {
                min_similarity: 1.0 - 2.0 * step as f64 / 100.0,
                min_margin: 0.80 * (1.0 - step as f64 / 100.0),
            })
            .collect(),
    }
}

/// Fixed-sequence confidence-interval calibration. Start from a strict point,
/// certify using an exact one-sided Clopper-Pearson bound, then relax the
/// threshold. Stop at the first failed candidate; later candidates are not
/// examined, even if their empirical risk might happen to fall again. Family-wise
/// alpha is corrected over strategy × seed, but not over candidates: the
/// predeclared fixed sequence controls that selection effect.
fn calibrate_fixed_sequence_rule(
    samples: &[ScoredQuery],
    mode: CalibrationMode,
) -> FixedSequenceCalibration {
    let candidates = fixed_sequence_candidates(mode);
    let abstain_rule = DecisionRule {
        min_similarity: 2.0,
        min_margin: 3.0,
    };
    let empty_certificate = CertificationResult {
        accepted: 0,
        errors: 0,
        upper_risk: 1.0,
        passed: false,
    };

    let mut selected_rule = None;
    let mut selected_certificate = empty_certificate;
    let mut selected_candidate_index = None;
    let mut certified_prefix_len = 0;
    let mut stopped_at_index = None;

    for (index, candidate) in candidates.iter().copied().enumerate() {
        let certificate = certify_rule(samples, candidate);
        if !certificate.passed {
            stopped_at_index = Some(index);
            break;
        }

        selected_rule = Some(candidate);
        selected_certificate = certificate;
        selected_candidate_index = Some(index);
        certified_prefix_len += 1;
    }

    FixedSequenceCalibration {
        rule: selected_rule.unwrap_or(abstain_rule),
        certificate: selected_certificate,
        certified_prefix_len,
        candidate_count: candidates.len(),
        selected_candidate_index,
        stopped_at_index,
    }
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
    let mut rejected_score_only = 0;
    let mut rejected_margin_only = 0;
    let mut rejected_both = 0;
    let mut errors = 0;
    let mut raw_correct = 0;

    for sample in samples {
        if sample.correct {
            raw_correct += 1;
        }
        let score_pass = sample.best_similarity >= rule.min_similarity;
        let margin_pass = sample.top_two_margin >= rule.min_margin;
        match (score_pass, margin_pass) {
            (true, true) => {
                accepted += 1;
                if !sample.correct {
                    errors += 1;
                }
            }
            (false, true) => rejected_score_only += 1,
            (true, false) => rejected_margin_only += 1,
            (false, false) => rejected_both += 1,
        }
    }

    Evaluation {
        accepted,
        rejected_score_only,
        rejected_margin_only,
        rejected_both,
        errors,
        raw_correct,
        total: samples.len(),
    }
}

fn certify_rule(samples: &[ScoredQuery], rule: DecisionRule) -> CertificationResult {
    let result = evaluate(samples, rule);
    let upper_risk =
        clopper_pearson_upper_bound(result.errors, result.accepted, CERTIFICATION_ALPHA_PER_RULE);
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
    let (certified, cert_accepted, cert_errors, cert_upper_risk) = certificate.map_or(
        (
            "not_applicable",
            "NA".to_owned(),
            "NA".to_owned(),
            "NA".to_owned(),
        ),
        |certificate| {
            (
                if certificate.passed { "pass" } else { "fail" },
                certificate.accepted.to_string(),
                certificate.errors.to_string(),
                format!("{:.6}", certificate.upper_risk),
            )
        },
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

/// Emit held-out risk/coverage points for every predeclared threshold and
/// report trapezoidal area under the risk–coverage curve (lower is better).
/// These are descriptive test-set metrics only; test rows never certify or tune
/// a threshold.
fn report_risk_coverage_curve(
    seed: u64,
    mode_name: &str,
    mode: CalibrationMode,
    samples: &[ScoredQuery],
) {
    let strategy = format!("fixed_grid_{mode_name}");
    let candidates = fixed_sequence_candidates(mode);
    let not_applicable = ("not_applicable", "NA", "NA", "NA");
    let mut points = Vec::with_capacity(candidates.len() + 1);
    points.push((0.0, 0.0));

    for (index, rule) in candidates.iter().copied().enumerate() {
        let result = evaluate(samples, rule);
        let risk = result.selective_risk().unwrap_or(0.0);
        points.push((result.coverage(), risk));
        write_evaluation_row(
            "risk_coverage_curve",
            seed,
            &strategy,
            &format!("candidate_{index}"),
            None,
            rule,
            0,
            0,
            not_applicable.0,
            not_applicable.1,
            not_applicable.2,
            not_applicable.3,
            samples.len(),
            result,
        );
    }

    // Coverage is non-decreasing by construction, but sort defensively so the
    // area calculation remains correct if the candidate generator changes.
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    let trapezoidal_aurc = points
        .windows(2)
        .map(|pair| {
            let width = pair[1].0 - pair[0].0;
            width * (pair[0].1 + pair[1].1) / 2.0
        })
        .sum::<f64>();
    // Right-continuous step integral: each newly reached coverage interval is
    // assigned the risk observed at its permissive (right-hand) endpoint. This
    // avoids relying on a visually smooth linear interpolation between thresholds.
    let step_aurc = points
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0) * pair[1].1)
        .sum::<f64>();
    println!(
        "# risk_coverage_summary=seed:{seed},strategy:{strategy},aurc_trapezoid:{trapezoidal_aurc:.6},aurc_step:{step_aurc:.6},curve_points:{}",
        points.len()
    );
}

/// Conformal nonconformity is 1 - similarity to the true target. The
/// finite-sample split-conformal quantile uses ceil((n+1)(1-alpha))-th order
/// statistic; if that rank exceeds n, q=+infinity gives the conservative full set.
fn split_conformal_quantile(
    calibration: &[ScoredQuery],
    alpha: f64,
    mode: ConformalScoreMode,
) -> Result<(f64, usize), Box<dyn Error>> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("conformal alpha must be finite and strictly between zero and one".into());
    }
    let mut nonconformity: Vec<f64> = calibration
        .iter()
        .filter(|sample| matches!(sample.kind, QueryKind::Known { .. }))
        .filter_map(|sample| {
            sample
                .target_similarity
                .map(|similarity| conformal_nonconformity(similarity, sample.best_similarity, mode))
        })
        .collect();
    if nonconformity.is_empty() {
        return Err("split conformal needs at least one labeled known calibration query".into());
    }
    nonconformity.sort_by(f64::total_cmp);
    let count = nonconformity.len();
    let rank = (((count + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    let quantile = if rank == 0 || rank > count {
        f64::INFINITY
    } else {
        nonconformity[rank - 1]
    };
    Ok((quantile, count))
}

fn conformal_nonconformity(
    candidate_similarity: f64,
    best_similarity: f64,
    mode: ConformalScoreMode,
) -> f64 {
    match mode {
        ConformalScoreMode::RawSimilarity => 1.0 - candidate_similarity,
        ConformalScoreMode::GapAdjusted { penalty } => {
            (1.0 - candidate_similarity) + penalty * (best_similarity - candidate_similarity)
        }
    }
}

fn score_conformal_sets(
    scored: &[ScoredQuery],
    conformal_q: f64,
    mode: ConformalScoreMode,
) -> Vec<ConformalQueryResult> {
    scored
        .iter()
        .map(|score| {
            let set_size = score
                .prototype_similarities
                .iter()
                .map(|&similarity| conformal_nonconformity(similarity, score.best_similarity, mode))
                .filter(|&nonconformity| nonconformity <= conformal_q)
                .count();
            let covered = score.target_similarity.is_some_and(|similarity| {
                conformal_nonconformity(similarity, score.best_similarity, mode) <= conformal_q
            });
            ConformalQueryResult {
                kind: score.kind,
                labeled: score.target_similarity.is_some(),
                covered,
                set_size,
            }
        })
        .collect()
}

fn tune_dco_score(tuning: &[ScoredQuery], alpha: f64) -> Result<DcoScoreSelection, Box<dyn Error>> {
    let mut candidates = vec![ConformalScoreMode::RawSimilarity];
    candidates.extend(
        DCO_GAP_PENALTY_CANDIDATES
            .iter()
            .copied()
            .skip(1)
            .map(|penalty| ConformalScoreMode::GapAdjusted { penalty }),
    );

    let mut best: Option<DcoScoreSelection> = None;
    for mode in candidates {
        let (quantile, tuning_n) = split_conformal_quantile(tuning, alpha, mode)?;
        let predictions = score_conformal_sets(tuning, quantile, mode);
        let known: Vec<_> = predictions
            .iter()
            .filter(|prediction| {
                prediction.labeled && matches!(prediction.kind, QueryKind::Known { .. })
            })
            .collect();
        if known.is_empty() {
            continue;
        }
        let covered = known.iter().filter(|prediction| prediction.covered).count();
        let coverage = covered as f64 / known.len() as f64;
        if coverage + f64::EPSILON < 1.0 - alpha {
            continue;
        }
        let mean_set_size = known
            .iter()
            .map(|prediction| prediction.set_size)
            .sum::<usize>() as f64
            / known.len() as f64;
        let candidate = DcoScoreSelection {
            mode,
            provisional_quantile: quantile,
            tuning_coverage: coverage,
            mean_set_size,
            tuning_n,
        };
        let should_replace = best.is_none_or(|current| {
            candidate.mean_set_size < current.mean_set_size
                || (candidate.mean_set_size == current.mean_set_size
                    && candidate.tuning_coverage > current.tuning_coverage)
        });
        if should_replace {
            best = Some(candidate);
        }
    }

    best.ok_or_else(|| "DCO tuning found no conformal score with target tuning coverage".into())
}

fn report_conformal_set_summaries(
    seed: u64,
    split: &str,
    alpha: f64,
    quantile: f64,
    calibration_n: usize,
    samples: &[ConformalQueryResult],
) {
    fn emit(
        seed: u64,
        split: &str,
        label: &str,
        alpha: f64,
        quantile: f64,
        calibration_n: usize,
        samples: &[ConformalQueryResult],
    ) {
        if samples.is_empty() {
            return;
        }
        let labeled_n = samples.iter().filter(|sample| sample.labeled).count();
        let covered = samples
            .iter()
            .filter(|sample| sample.labeled && sample.covered)
            .count();
        let set_sum: usize = samples.iter().map(|sample| sample.set_size).sum();
        let singleton = samples.iter().filter(|sample| sample.set_size == 1).count();
        let empty = samples.iter().filter(|sample| sample.set_size == 0).count();
        let coverage = if labeled_n > 0 {
            format!("{:.6}", covered as f64 / labeled_n as f64)
        } else {
            "NA".to_owned()
        };
        println!(
            "# conformal_set=seed:{seed},split:{split},kind:{label},alpha:{alpha:.4},q:{quantile:.6},calibration_known_n:{calibration_n},n:{},labeled_n:{labeled_n},covered:{covered},marginal_coverage:{coverage},mean_set_size:{:.6},singleton_fraction:{:.6},empty_fraction:{:.6}",
            samples.len(),
            set_sum as f64 / samples.len() as f64,
            singleton as f64 / samples.len() as f64,
            empty as f64 / samples.len() as f64
        );
    }

    let known_only: Vec<_> = samples
        .iter()
        .copied()
        .filter(|sample| matches!(sample.kind, QueryKind::Known { .. }))
        .collect();
    emit(
        seed,
        split,
        "known_all",
        alpha,
        quantile,
        calibration_n,
        &known_only,
    );
    emit(
        seed,
        split,
        "all_diagnostic",
        alpha,
        quantile,
        calibration_n,
        samples,
    );
    for noise_index in 0..NOISE_LEVELS.len() {
        let kind = QueryKind::Known { noise_index };
        let subset: Vec<_> = samples
            .iter()
            .copied()
            .filter(|sample| sample.kind == kind)
            .collect();
        emit(
            seed,
            split,
            &format!("known_noise_{noise_index}"),
            alpha,
            quantile,
            calibration_n,
            &subset,
        );
    }
    for (kind, label) in [
        (QueryKind::RandomOod, "random_ood"),
        (QueryKind::AmbiguousMixture, "ambiguous_mixture"),
    ] {
        let subset: Vec<_> = samples
            .iter()
            .copied()
            .filter(|sample| sample.kind == kind)
            .collect();
        emit(seed, split, label, alpha, quantile, calibration_n, &subset);
    }
    for distance_index in 0..NEAR_UNKNOWN_DISTANCES.len() {
        let kind = QueryKind::NearPrototypeUnknown { distance_index };
        let subset: Vec<_> = samples
            .iter()
            .copied()
            .filter(|sample| sample.kind == kind)
            .collect();
        emit(
            seed,
            split,
            &format!("near_unknown_{distance_index}"),
            alpha,
            quantile,
            calibration_n,
            &subset,
        );
    }
}

fn report_identity_collision_controls(
    seed: u64,
    prototypes: &[PatternVector],
    policies: &[(String, DecisionRule)],
) -> Result<(), Box<dyn Error>> {
    // Each vector is scored once. The "unknown identity" variant is derived by
    // changing only its semantic registration label; its cached feature scores
    // are identical by construction, while its correctness label conflicts.
    let not_applicable = ("not_applicable", "NA", "NA", "NA");
    for noise_index in [1usize, 2usize] {
        let noise_fraction = NOISE_LEVELS[noise_index];
        let distance_index = NEAR_UNKNOWN_DISTANCES
            .iter()
            .position(|distance| *distance == noise_fraction)
            .ok_or("identity-collision noise rate missing from near-unknown distances")?;
        let mut rng_state = derive_seed(seed, 0x4944_434f, noise_index as u64);
        let mut known_queries = Vec::with_capacity(NEAR_UNKNOWN_PER_DISTANCE);
        let mut unknown_queries = Vec::with_capacity(NEAR_UNKNOWN_PER_DISTANCE);

        for _ in 0..NEAR_UNKNOWN_PER_DISTANCE {
            let source = sample_bounded(&mut rng_state, prototypes.len());
            let observed =
                prototypes[source].with_noise(noise_fraction, next_u64(&mut rng_state))?;
            known_queries.push(QuerySample {
                vector: observed.clone(),
                target: Some(source),
                kind: QueryKind::Known { noise_index },
            });
            unknown_queries.push(QuerySample {
                vector: observed,
                target: None,
                kind: QueryKind::NearPrototypeUnknown { distance_index },
            });
        }

        if known_queries
            .iter()
            .zip(&unknown_queries)
            .any(|(known, unknown)| known.vector != unknown.vector)
        {
            return Err("identity collision control did not preserve exact vector equality".into());
        }
        let known_scored = score_queries(&known_queries, prototypes)?;
        let unknown_scored: Vec<ScoredQuery> = known_scored
            .iter()
            .zip(&unknown_queries)
            .map(|(known, unknown)| {
                let mut score = known.clone();
                score.correct = false;
                score.target_similarity = None;
                score.kind = unknown.kind;
                score
            })
            .collect();

        for (known, unknown) in known_scored.iter().zip(&unknown_scored) {
            if known.best_similarity != unknown.best_similarity
                || known.top_two_margin != unknown.top_two_margin
                || known.prototype_similarities != unknown.prototype_similarities
            {
                return Err("identity collision control failed feature equality invariant".into());
            }
        }

        for (strategy, rule) in policies {
            for (record_type, label, scored) in [
                (
                    "identifiability_control_known",
                    format!("same_vector_known_noise_{noise_index}"),
                    &known_scored,
                ),
                (
                    "identifiability_control_unknown",
                    format!("same_vector_unknown_noise_{noise_index}"),
                    &unknown_scored,
                ),
            ] {
                write_evaluation_row(
                    record_type,
                    seed,
                    strategy,
                    &label,
                    Some(noise_fraction),
                    *rule,
                    0,
                    0,
                    not_applicable.0,
                    not_applicable.1,
                    not_applicable.2,
                    not_applicable.3,
                    scored.len(),
                    evaluate(scored, *rule),
                );
            }
        }
    }
    Ok(())
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
        "{record_type},{seed},{strategy},{query_kind},{noise},{DIMENSION},{PROTOTYPE_COUNT},{selection_n},{certification_n},{certified},{cert_accepted},{cert_errors},{cert_upper_risk},{test_n},{:.6},{:.6},{},{},{},{},{},{:.6},{},{},{:.6},{:.6},{:.6}",
        rule.min_similarity,
        rule.min_margin,
        result.total,
        result.accepted,
        result.rejected_score_only,
        result.rejected_margin_only,
        result.rejected_both,
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
    let half_width = z * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;
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
        terms.push(log_choose + successes as f64 * log_p + (trials - successes) as f64 * log_q);
    }
    let max_log = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_log.is_finite() {
        return 0.0;
    }
    (max_log
        + terms
            .iter()
            .map(|value| (value - max_log).exp())
            .sum::<f64>()
            .ln())
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
    let half_width = z * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;
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
    fn rejection_reasons_partition_nonaccepted_queries() {
        let samples = vec![
            ScoredQuery {
                best_similarity: 0.9,
                top_two_margin: 0.2,
                correct: true,
                target_similarity: Some(0.9),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 0 },
            },
            ScoredQuery {
                best_similarity: 0.1,
                top_two_margin: 0.2,
                correct: false,
                target_similarity: None,
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::RandomOod,
            },
            ScoredQuery {
                best_similarity: 0.9,
                top_two_margin: 0.01,
                correct: false,
                target_similarity: None,
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::AmbiguousMixture,
            },
            ScoredQuery {
                best_similarity: 0.1,
                top_two_margin: 0.01,
                correct: false,
                target_similarity: None,
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::NearPrototypeUnknown { distance_index: 0 },
            },
        ];
        let result = evaluate(
            &samples,
            DecisionRule {
                min_similarity: 0.5,
                min_margin: 0.05,
            },
        );
        assert_eq!(result.accepted, 1);
        assert_eq!(result.rejected_score_only, 1);
        assert_eq!(result.rejected_margin_only, 1);
        assert_eq!(result.rejected_both, 1);
        assert_eq!(
            result.accepted
                + result.rejected_score_only
                + result.rejected_margin_only
                + result.rejected_both,
            result.total
        );
    }

    #[test]
    fn gap_adjusted_score_penalizes_non_top_candidates() {
        let raw = conformal_nonconformity(0.85, 0.90, ConformalScoreMode::RawSimilarity);
        let top =
            conformal_nonconformity(0.90, 0.90, ConformalScoreMode::GapAdjusted { penalty: 2.0 });
        let lower_rank =
            conformal_nonconformity(0.85, 0.90, ConformalScoreMode::GapAdjusted { penalty: 2.0 });
        assert!((raw - 0.15).abs() < 1e-12);
        assert!((top - 0.10).abs() < 1e-12);
        assert!((lower_rank - 0.25).abs() < 1e-12);
        assert!(lower_rank > raw);
    }

    #[test]
    fn dco_score_tuning_is_deterministic_and_uses_independent_calibration() {
        let prototypes = (0..8)
            .map(|index| PatternVector::seeded_random(256, 5_000 + index as u64))
            .collect::<Result<Vec<_>, _>>()
            .expect("prototype bank");
        let tuning_queries =
            generate_queries(&prototypes, 0x5455_4e45, 512).expect("tuning queries");
        let fresh_calibration_queries =
            generate_queries(&prototypes, 0x4341_4c49, 512).expect("fresh calibration queries");
        let tuning = score_queries(&tuning_queries, &prototypes).expect("tuning scores");
        let fresh_calibration =
            score_queries(&fresh_calibration_queries, &prototypes).expect("calibration scores");

        let first = tune_dco_score(&tuning, CONFORMAL_ALPHA).expect("DCO tuning");
        let second = tune_dco_score(&tuning, CONFORMAL_ALPHA).expect("repeat DCO tuning");
        assert_eq!(first.mode, second.mode);
        assert_eq!(first.provisional_quantile, second.provisional_quantile);
        assert_eq!(first.mean_set_size, second.mean_set_size);
        assert!(first.tuning_coverage >= 1.0 - CONFORMAL_ALPHA);

        // Selection does not receive the fresh-calibration data; calibration is
        // performed only after the DCO score mode is fixed.
        let (fresh_quantile, fresh_count) =
            split_conformal_quantile(&fresh_calibration, CONFORMAL_ALPHA, first.mode)
                .expect("fresh conformal calibration");
        assert!(fresh_count > 0);
        let heldout_sets = score_conformal_sets(&fresh_calibration, fresh_quantile, first.mode);
        let known: Vec<_> = heldout_sets
            .iter()
            .filter(|result| result.labeled && matches!(result.kind, QueryKind::Known { .. }))
            .collect();
        assert!(!known.is_empty());
    }

    #[test]
    fn bonferroni_exact_binomial_certificate_respects_sample_floor() {
        assert!(
            clopper_pearson_upper_bound(0, 124, CERTIFICATION_ALPHA_PER_RULE) > MAX_VALIDATION_RISK
        );
        assert!(
            clopper_pearson_upper_bound(0, 125, CERTIFICATION_ALPHA_PER_RULE)
                <= MAX_VALIDATION_RISK
        );
        assert!(
            clopper_pearson_upper_bound(1, 125, CERTIFICATION_ALPHA_PER_RULE) > MAX_VALIDATION_RISK
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
                target_similarity: Some(0.5),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
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
                target_similarity: Some(0.5),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
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
                target_similarity: Some(0.5),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 0 },
            },
            ScoredQuery {
                best_similarity: 0.8,
                top_two_margin: 0.1,
                correct: false,
                target_similarity: None,
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
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

    #[test]
    fn fixed_sequence_candidates_are_nested_and_predeclared() {
        for mode in [
            CalibrationMode::ScoreOnly,
            CalibrationMode::MarginOnly,
            CalibrationMode::ScoreAndMargin,
        ] {
            let candidates = fixed_sequence_candidates(mode);
            assert!(candidates.len() >= 2);
            for pair in candidates.windows(2) {
                assert!(pair[1].min_similarity <= pair[0].min_similarity);
                assert!(pair[1].min_margin <= pair[0].min_margin);
            }
        }
    }

    #[test]
    fn fixed_sequence_stops_before_a_later_safe_nonmonotone_threshold() {
        // The initial strict candidates accept 400 correct examples at similarity
        // 1.0. At threshold 0.58, 50 errors enter and the risk certificate fails.
        // At the final -1.0 candidate, 2,000 additional correct examples dilute
        // the observed error rate enough to pass again. Fixed-sequence CIC must
        // nevertheless stop at the first failure rather than selecting that later point.
        let mut samples = Vec::with_capacity(2_450);
        samples.extend((0..400).map(|_| ScoredQuery {
            best_similarity: 1.0,
            top_two_margin: 0.95,
            correct: true,
            target_similarity: Some(0.5),
            prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
            kind: QueryKind::Known { noise_index: 0 },
        }));
        samples.extend((0..50).map(|_| ScoredQuery {
            best_similarity: 0.59,
            top_two_margin: 0.1,
            correct: false,
            target_similarity: Some(0.5),
            prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
            kind: QueryKind::AmbiguousMixture,
        }));
        samples.extend((0..2_000).map(|_| ScoredQuery {
            best_similarity: 0.45,
            top_two_margin: 0.05,
            correct: true,
            target_similarity: Some(0.5),
            prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
            kind: QueryKind::Known { noise_index: 1 },
        }));

        let candidates = fixed_sequence_candidates(CalibrationMode::ScoreOnly);
        let result = calibrate_fixed_sequence_rule(&samples, CalibrationMode::ScoreOnly);
        assert!(result.certified_prefix_len > 0);
        assert_eq!(result.stopped_at_index, Some(21));
        assert_eq!(result.selected_candidate_index, Some(20));
        assert!(result.certificate.passed);

        let later_certificate = certify_rule(&samples, *candidates.last().expect("candidate grid"));
        assert!(
            later_certificate.passed,
            "the final candidate should be safe, but fixed-sequence testing must stop earlier"
        );
    }

    #[test]
    fn split_conformal_quantile_uses_labeled_known_order_statistic() {
        let samples = vec![
            ScoredQuery {
                best_similarity: 0.9,
                top_two_margin: 0.2,
                correct: true,
                target_similarity: Some(0.9),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 0 },
            },
            ScoredQuery {
                best_similarity: 0.85,
                top_two_margin: 0.1,
                correct: true,
                target_similarity: Some(0.8),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 1 },
            },
            ScoredQuery {
                best_similarity: 0.7,
                top_two_margin: 0.05,
                correct: true,
                target_similarity: Some(0.7),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 2 },
            },
            ScoredQuery {
                best_similarity: 0.6,
                top_two_margin: 0.03,
                correct: false,
                target_similarity: Some(0.6),
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::Known { noise_index: 3 },
            },
            ScoredQuery {
                best_similarity: 0.99,
                top_two_margin: 0.9,
                correct: false,
                target_similarity: None,
                prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
                kind: QueryKind::RandomOod,
            },
        ];
        let (quantile, calibration_n) =
            split_conformal_quantile(&samples, 0.5, ConformalScoreMode::RawSimilarity)
                .expect("quantile");
        assert_eq!(calibration_n, 4);
        // Scores 1 - sim are [0.1, 0.2, 0.3, 0.4]; ceil((4+1)*.5)=3.
        assert!((quantile - 0.3).abs() < 1e-12);
    }

    #[test]
    fn conformal_returns_sets_and_cannot_separate_identical_identity_vectors() {
        let prototypes = (0..8)
            .map(|index| PatternVector::seeded_random(4_096, 800 + index as u64))
            .collect::<Result<Vec<_>, _>>()
            .expect("prototypes");
        let known = QuerySample {
            vector: prototypes[0].clone(),
            target: Some(0),
            kind: QueryKind::Known { noise_index: 0 },
        };
        let unknown = QuerySample {
            vector: prototypes[0].clone(),
            target: None,
            kind: QueryKind::NearPrototypeUnknown { distance_index: 0 },
        };
        let queries = vec![known, unknown];
        let scores = score_queries(&queries, &prototypes).expect("scores");
        let set_results = score_conformal_sets(&scores, 0.0, ConformalScoreMode::RawSimilarity);
        assert_eq!(set_results[0].set_size, 1);
        assert_eq!(set_results[0].set_size, set_results[1].set_size);
        assert!(set_results[0].covered);
        assert!(!set_results[1].labeled);
    }

    #[test]
    fn conformal_uses_full_set_when_finite_sample_rank_exceeds_calibration_size() {
        let sample = ScoredQuery {
            best_similarity: 1.0,
            top_two_margin: 0.5,
            correct: true,
            target_similarity: Some(1.0),
            prototype_similarities: Arc::<[f64]>::from(Vec::<f64>::new()),
            kind: QueryKind::Known { noise_index: 0 },
        };
        let (quantile, count) =
            split_conformal_quantile(&[sample], 0.05, ConformalScoreMode::RawSimilarity)
                .expect("full-set quantile");
        assert_eq!(count, 1);
        assert!(quantile.is_infinite());
    }

    #[test]
    fn identical_vectors_with_different_registration_labels_are_unidentifiable() {
        let prototypes = (0..8)
            .map(|index| PatternVector::seeded_random(4_096, 100 + index as u64))
            .collect::<Result<Vec<_>, _>>()
            .expect("prototypes");
        let source = 0;
        let noise_index = 1;
        let vector = prototypes[source]
            .with_noise(NOISE_LEVELS[noise_index], 0xabc)
            .expect("noisy observation");
        let known = QuerySample {
            vector: vector.clone(),
            target: Some(source),
            kind: QueryKind::Known { noise_index },
        };
        let unknown = QuerySample {
            vector,
            target: None,
            kind: QueryKind::NearPrototypeUnknown { distance_index: 1 },
        };

        assert_eq!(known.vector, unknown.vector);
        let known_score =
            score_queries(std::slice::from_ref(&known), &prototypes).expect("known score");
        let unknown_score =
            score_queries(std::slice::from_ref(&unknown), &prototypes).expect("unknown score");
        assert_eq!(
            known_score[0].best_similarity,
            unknown_score[0].best_similarity
        );
        assert_eq!(
            known_score[0].top_two_margin,
            unknown_score[0].top_two_margin
        );
        assert_ne!(known_score[0].correct, unknown_score[0].correct);

        for rule in [
            DecisionRule {
                min_similarity: -1.0,
                min_margin: -1.0,
            },
            DecisionRule {
                min_similarity: 0.5,
                min_margin: -1.0,
            },
            DecisionRule {
                min_similarity: -1.0,
                min_margin: 0.05,
            },
            DecisionRule {
                min_similarity: 0.5,
                min_margin: 0.05,
            },
        ] {
            assert_eq!(
                evaluate(&known_score, rule).accepted,
                evaluate(&unknown_score, rule).accepted,
                "same features must lead to the same vector-only acceptance decision"
            );
        }
    }
}
