// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare one global acceptance threshold with a class-wise OSP-inspired threshold map.
//!
//! Restricted policy class:
//!   S_c = {x : argmax_j sim(x, p_j) = c and max_j sim(x, p_j) >= tau_c}.
//! The regions are disjoint by argmax. A dynamic program selects the per-class
//! thresholds to maximize empirical accepted coverage under a predeclared
//! selection-split false-acceptance *mass* budget. A fresh certification split
//! bounds each class's false-acceptance mass; the sum of the upper bounds must
//! fit the global error budget. This is an OSP-inspired score-threshold baseline,
//! not a reproduction of the full One-Sided Prediction algorithm.

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 1_024;
const CLASS_COUNT: usize = 8;
const SELECTION_N: usize = 4_096;
const CERTIFICATION_N: usize = 4_096;
const EVALUATION_N: usize = 8_192;
const TOTAL_ERROR_MASS_BUDGET: f64 = 0.05;
const SELECTION_ERROR_MASS_BUDGET: f64 = 0.025;
const NOISE_LEVELS: [f64; 7] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50];
const SEEDS: [u64; 5] = [17, 29, 41, 53, 67];
// The family includes one global bound and one bound for each class, for every seed.
const CERTIFICATION_ALPHA: f64 =
    0.05 / (SEEDS.len() as f64 * (CLASS_COUNT as f64 + 1.0));

#[derive(Debug, Clone, PartialEq)]
struct ScoredQuery {
    target: Option<usize>,
    predicted_class: usize,
    best_similarity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Evaluation {
    total: usize,
    accepted: usize,
    errors: usize,
}

impl Evaluation {
    fn coverage(self) -> f64 {
        safe_ratio(self.accepted, self.total)
    }

    fn error_mass(self) -> f64 {
        safe_ratio(self.errors, self.total)
    }

    fn selective_risk(self) -> Option<f64> {
        (self.accepted > 0).then_some(self.errors as f64 / self.accepted as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThresholdOption {
    threshold: f64,
    accepted: usize,
    false_positives: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ThresholdPath {
    accepted: usize,
    false_positives: usize,
    thresholds: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ErrorMassCertificate {
    false_positives: usize,
    upper_bound: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    println!("# protocol=classwise-osp-inspired-threshold-map-v1");
    println!(
        "# dimension={DIMENSION};classes={CLASS_COUNT};query_mix=known:random_ood:ambiguous_mixture=7:2:2"
    );
    println!(
        "# selection_n={SELECTION_N};certification_n={CERTIFICATION_N};evaluation_n={EVALUATION_N}"
    );
    println!(
        "# total_error_mass_budget={TOTAL_ERROR_MASS_BUDGET:.4};selection_error_mass_budget={SELECTION_ERROR_MASS_BUDGET:.4}"
    );
    println!(
        "# certificate=one_sided_clopper_pearson;alpha_per_bound={CERTIFICATION_ALPHA:.8};families_per_seed=1_global_plus_{CLASS_COUNT}_classes"
    );
    println!("# comparison=global_scalar_threshold_vs_coverage_optimal_classwise_threshold_map");
    println!(
        "record_type,seed,strategy,class,threshold,selection_n,selection_fp,certification_n,cert_fp,cert_upper_error_mass,cert_sum_upper,certified,test_population,group_n,accepted,coverage_contribution,group_coverage,errors,error_mass,selective_risk,risk95_low,risk95_high"
    );

    for seed in SEEDS {
        let prototypes = (0..CLASS_COUNT)
            .map(|class| {
                PatternVector::seeded_random(
                    DIMENSION,
                    derive_seed(seed, 0x5052_4f54, class as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let selection = score_queries(
            &generate_queries(&prototypes, derive_seed(seed, 0x5345_4c45, 0), SELECTION_N)?,
            &prototypes,
        )?;
        let certification = score_queries(
            &generate_queries(&prototypes, derive_seed(seed, 0x4345_5254, 0), CERTIFICATION_N)?,
            &prototypes,
        )?;
        let evaluation = score_queries(
            &generate_queries(&prototypes, derive_seed(seed, 0x4556_414c, 0), EVALUATION_N)?,
            &prototypes,
        )?;

        let global_threshold =
            fit_global_threshold(&selection, SELECTION_ERROR_MASS_BUDGET);
        let global_selection = evaluate_global(&selection, global_threshold);
        let global_cert = certify_global(&certification, global_threshold);
        let global_test = evaluate_global(&evaluation, global_threshold);
        write_row(
            "overall",
            seed,
            "global_threshold",
            "all",
            Some(global_threshold),
            selection.len(),
            global_selection.errors,
            certification.len(),
            global_cert.false_positives,
            global_cert.upper_bound,
            global_cert.upper_bound,
            if global_cert.upper_bound <= TOTAL_ERROR_MASS_BUDGET { "pass" } else { "fail" },
            evaluation.len(),
            global_test,
            evaluation.len(),
        );

        let classwise_thresholds = fit_classwise_thresholds(
            &selection,
            CLASS_COUNT,
            SELECTION_ERROR_MASS_BUDGET,
        );
        let classwise_selection = evaluate_classwise(&selection, &classwise_thresholds);
        let classwise_certificate =
            certify_classwise(&certification, &classwise_thresholds, CLASS_COUNT);
        let classwise_test = evaluate_classwise(&evaluation, &classwise_thresholds);
        let classwise_upper_sum = classwise_certificate
            .iter()
            .map(|certificate| certificate.upper_bound)
            .sum::<f64>();
        let classwise_pass =
            classwise_upper_sum <= TOTAL_ERROR_MASS_BUDGET + f64::EPSILON;

        write_row(
            "overall",
            seed,
            "classwise_osp_inspired",
            "all",
            None,
            selection.len(),
            classwise_selection.errors,
            certification.len(),
            classwise_certificate.iter().map(|certificate| certificate.false_positives).sum(),
            classwise_upper_sum,
            classwise_upper_sum,
            if classwise_pass { "pass" } else { "fail" },
            evaluation.len(),
            classwise_test,
            evaluation.len(),
        );

        for (class, (threshold, class_cert)) in classwise_thresholds
            .iter()
            .copied()
            .zip(classwise_certificate.iter().copied())
            .enumerate()
        {
            let group: Vec<ScoredQuery> = evaluation
                .iter()
                .filter(|sample| sample.predicted_class == class)
                .cloned()
                .collect();
            let class_test = evaluate_classwise(&group, &classwise_thresholds);
            write_row(
                "class",
                seed,
                "classwise_osp_inspired",
                &class.to_string(),
                Some(threshold),
                selection.len(),
                count_class_false_positives(&selection, class, threshold),
                certification.len(),
                class_cert.false_positives,
                class_cert.upper_bound,
                classwise_upper_sum,
                if classwise_pass { "pass" } else { "fail" },
                evaluation.len(),
                class_test,
                group.len(),
            );
        }

        println!(
            "# seed_summary=seed:{seed},global_threshold:{global_threshold:.6},global_coverage:{:.6},global_error_mass:{:.6},global_conditional_risk:{},global_cert_upper:{:.6},classwise_coverage:{:.6},classwise_error_mass:{:.6},classwise_conditional_risk:{},classwise_cert_upper_sum:{classwise_upper_sum:.6},classwise_certified:{}",
            global_test.coverage(),
            global_test.error_mass(),
            format_optional(global_test.selective_risk()),
            global_cert.upper_bound,
            classwise_test.coverage(),
            classwise_test.error_mass(),
            format_optional(classwise_test.selective_risk()),
            classwise_pass,
        );
    }

    println!("# wall_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn generate_queries(
    prototypes: &[PatternVector],
    seed: u64,
    count: usize,
) -> Result<Vec<(PatternVector, Option<usize>)>, Box<dyn Error>> {
    if prototypes.len() < 2 {
        return Err("classwise threshold benchmark requires at least two prototypes".into());
    }
    let mut state = seed;
    let mut samples = Vec::with_capacity(count);
    let type_weight = 11;
    for _ in 0..count {
        let kind = sample_bounded(&mut state, type_weight);
        if kind < 7 {
            let target = sample_bounded(&mut state, prototypes.len());
            let noise_index = sample_bounded(&mut state, NOISE_LEVELS.len());
            let query = prototypes[target].with_noise(
                NOISE_LEVELS[noise_index],
                next_u64(&mut state),
            )?;
            samples.push((query, Some(target)));
        } else if kind < 9 {
            let query = PatternVector::seeded_random(
                prototypes[0].dimension(),
                next_u64(&mut state),
            )?;
            samples.push((query, None));
        } else {
            let left = sample_bounded(&mut state, prototypes.len());
            let right = (left + 1 + sample_bounded(&mut state, prototypes.len() - 1))
                % prototypes.len();
            let query = PatternVector::bundle(
                &[prototypes[left].clone(), prototypes[right].clone()],
                next_u64(&mut state),
            )?;
            // Mixtures have no unique target, so accepting one label counts as error.
            samples.push((query, None));
        }
    }
    Ok(samples)
}

fn score_queries(
    queries: &[(PatternVector, Option<usize>)],
    prototypes: &[PatternVector],
) -> Result<Vec<ScoredQuery>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(queries.len());
    for (query, target) in queries {
        let mut predicted_class = 0;
        let mut best_similarity = f64::NEG_INFINITY;
        for (class, prototype) in prototypes.iter().enumerate() {
            let similarity = query.similarity(prototype)?;
            // First-maximum tie break, matching the main selective-retrieval runner.
            if similarity > best_similarity {
                predicted_class = class;
                best_similarity = similarity;
            }
        }
        output.push(ScoredQuery {
            target: *target,
            predicted_class,
            best_similarity,
        });
    }
    Ok(output)
}

fn fit_global_threshold(samples: &[ScoredQuery], false_positive_mass_budget: f64) -> f64 {
    let mut best = ThresholdOption {
        threshold: f64::INFINITY,
        accepted: 0,
        false_positives: 0,
    };
    for threshold in candidate_thresholds(samples, None) {
        let result = evaluate_global(samples, threshold);
        if result.error_mass() > false_positive_mass_budget {
            continue;
        }
        if result.accepted > best.accepted
            || (result.accepted == best.accepted && threshold > best.threshold)
        {
            best = ThresholdOption {
                threshold,
                accepted: result.accepted,
                false_positives: result.errors,
            };
        }
    }
    best.threshold
}

/// Select class thresholds by dynamic programming over the integer false-positive
/// budget. For each class, the frontier lists non-dominated (FP count, coverage)
/// choices; DP allocates the total error budget across classes to maximize coverage.
fn fit_classwise_thresholds(
    samples: &[ScoredQuery],
    class_count: usize,
    false_positive_mass_budget: f64,
) -> Vec<f64> {
    if samples.is_empty() || class_count == 0 {
        return vec![f64::INFINITY; class_count];
    }
    let total_false_positive_budget =
        (false_positive_mass_budget * samples.len() as f64).floor() as usize;
    let frontiers: Vec<Vec<ThresholdOption>> = (0..class_count)
        .map(|class| class_frontier(samples, class, false_positive_mass_budget))
        .collect();

    let mut states: Vec<Option<ThresholdPath>> =
        (0..=total_false_positive_budget).map(|_| None).collect();
    states[0] = Some(ThresholdPath {
        accepted: 0,
        false_positives: 0,
        thresholds: Vec::with_capacity(class_count),
    });

    for frontier in frontiers {
        let mut next: Vec<Option<ThresholdPath>> =
            (0..=total_false_positive_budget).map(|_| None).collect();
        for (used_budget, state) in states.iter().enumerate() {
            let Some(state) = state else {
                continue;
            };
            for option in &frontier {
                let new_cost = used_budget + option.false_positives;
                if new_cost > total_false_positive_budget {
                    continue;
                }
                let mut thresholds = state.thresholds.clone();
                thresholds.push(option.threshold);
                let candidate = ThresholdPath {
                    accepted: state.accepted + option.accepted,
                    false_positives: state.false_positives + option.false_positives,
                    thresholds,
                };
                let replace = next[new_cost].as_ref().is_none_or(|current| {
                    candidate.accepted > current.accepted
                        || (candidate.accepted == current.accepted
                            && candidate.false_positives < current.false_positives)
                });
                if replace {
                    next[new_cost] = Some(candidate);
                }
            }
        }
        states = next;
    }

    states
        .into_iter()
        .flatten()
        .max_by(|left, right| {
            left.accepted.cmp(&right.accepted).then_with(|| {
                right
                    .false_positives
                    .cmp(&left.false_positives)
            })
        })
        .map(|best| best.thresholds)
        .unwrap_or_else(|| vec![f64::INFINITY; class_count])
}

fn class_frontier(
    samples: &[ScoredQuery],
    class: usize,
    mass_budget: f64,
) -> Vec<ThresholdOption> {
    let budget = (mass_budget * samples.len() as f64).floor() as usize;
    let mut by_error_count: BTreeMap<usize, ThresholdOption> = BTreeMap::new();
    for threshold in candidate_thresholds(samples, Some(class)) {
        let mut accepted = 0;
        let mut false_positives = 0;
        for sample in samples.iter().filter(|sample| sample.predicted_class == class) {
            if sample.best_similarity >= threshold {
                accepted += 1;
                if sample.target != Some(class) {
                    false_positives += 1;
                }
            }
        }
        if false_positives > budget {
            continue;
        }
        let candidate = ThresholdOption {
            threshold,
            accepted,
            false_positives,
        };
        let replace = by_error_count
            .get(&false_positives)
            .is_none_or(|current| {
                candidate.accepted > current.accepted
                    || (candidate.accepted == current.accepted
                        && candidate.threshold > current.threshold)
            });
        if replace {
            by_error_count.insert(false_positives, candidate);
        }
    }

    let mut frontier = Vec::new();
    let mut best_coverage = 0;
    for candidate in by_error_count.into_values() {
        if candidate.accepted > best_coverage {
            best_coverage = candidate.accepted;
            frontier.push(candidate);
        }
    }
    if frontier.is_empty() {
        frontier.push(ThresholdOption {
            threshold: f64::INFINITY,
            accepted: 0,
            false_positives: 0,
        });
    }
    frontier
}

fn candidate_thresholds(samples: &[ScoredQuery], class: Option<usize>) -> Vec<f64> {
    let mut thresholds = vec![f64::INFINITY];
    thresholds.extend(
        samples
            .iter()
            .filter(|sample| class.is_none_or(|class| sample.predicted_class == class))
            .map(|sample| sample.best_similarity),
    );
    thresholds.sort_by(|left, right| right.total_cmp(left));
    thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());
    thresholds
}

fn evaluate_global(samples: &[ScoredQuery], threshold: f64) -> Evaluation {
    evaluate_with(samples, |sample| sample.best_similarity >= threshold)
}

fn evaluate_classwise(samples: &[ScoredQuery], thresholds: &[f64]) -> Evaluation {
    evaluate_with(samples, |sample| {
        thresholds
            .get(sample.predicted_class)
            .is_some_and(|threshold| sample.best_similarity >= *threshold)
    })
}

fn evaluate_with(samples: &[ScoredQuery], accept: impl Fn(&ScoredQuery) -> bool) -> Evaluation {
    let mut accepted = 0;
    let mut errors = 0;
    for sample in samples {
        if accept(sample) {
            accepted += 1;
            if sample.target != Some(sample.predicted_class) {
                errors += 1;
            }
        }
    }
    Evaluation {
        total: samples.len(),
        accepted,
        errors,
    }
}

fn count_class_false_positives(
    samples: &[ScoredQuery],
    class: usize,
    threshold: f64,
) -> usize {
    samples
        .iter()
        .filter(|sample| {
            sample.predicted_class == class
                && sample.best_similarity >= threshold
                && sample.target != Some(class)
        })
        .count()
}

fn certify_global(
    certification: &[ScoredQuery],
    threshold: f64,
) -> ErrorMassCertificate {
    let result = evaluate_global(certification, threshold);
    ErrorMassCertificate {
        false_positives: result.errors,
        upper_bound: clopper_pearson_upper_bound(
            result.errors,
            certification.len(),
            CERTIFICATION_ALPHA,
        ),
    }
}

fn certify_classwise(
    certification: &[ScoredQuery],
    thresholds: &[f64],
    class_count: usize,
) -> Vec<ErrorMassCertificate> {
    thresholds
        .iter()
        .copied()
        .take(class_count)
        .enumerate()
        .map(|(class, threshold)| {
            let false_positives =
                count_class_false_positives(certification, class, threshold);
            ErrorMassCertificate {
                false_positives,
                upper_bound: clopper_pearson_upper_bound(
                    false_positives,
                    certification.len(),
                    CERTIFICATION_ALPHA,
                ),
            }
        })
        .collect()
}

fn write_row(
    record_type: &str,
    seed: u64,
    strategy: &str,
    class: &str,
    threshold: Option<f64>,
    selection_n: usize,
    selection_fp: usize,
    certification_n: usize,
    certification_fp: usize,
    cert_upper: f64,
    cert_upper_sum: f64,
    certified: &str,
    test_population: usize,
    result: Evaluation,
    group_n: usize,
) {
    let selective_risk = result.selective_risk();
    let (risk_low, risk_high) = wilson_interval_95(result.errors, result.accepted);
    let threshold_text = threshold
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".to_owned());
    println!(
        "{record_type},{seed},{strategy},{class},{threshold_text},{selection_n},{selection_fp},{certification_n},{certification_fp},{cert_upper:.8},{cert_upper_sum:.8},{certified},{test_population},{group_n},{},{:.6},{:.6},{},{:.8},{},{:.6},{:.6}",
        result.accepted,
        safe_ratio(result.accepted, test_population),
        result.coverage(),
        result.errors,
        safe_ratio(result.errors, test_population),
        format_optional(selective_risk),
        risk_low,
        risk_high,
    );
}

fn clopper_pearson_upper_bound(errors: usize, trials: usize, alpha: f64) -> f64 {
    if trials == 0 || errors >= trials || !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return 1.0;
    }
    if errors == 0 {
        return 1.0 - alpha.powf(1.0 / trials as f64);
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..64 {
        let p = (low + high) / 2.0;
        if binomial_cdf_at_most(errors, trials, p) > alpha {
            low = p;
        } else {
            high = p;
        }
    }
    high
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

fn wilson_interval_95(errors: usize, accepted: usize) -> (f64, f64) {
    if accepted == 0 {
        return (0.0, 1.0);
    }
    wilson_interval(errors, accepted, 1.959_963_984_540_054_f64)
}

fn wilson_interval(errors: usize, accepted: usize, z: f64) -> (f64, f64) {
    let n = accepted as f64;
    let p = errors as f64 / n;
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

fn safe_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

fn format_optional(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_else(|| "NA".into())
}

fn sample_bounded(state: &mut u64, upper_exclusive: usize) -> usize {
    let bound = upper_exclusive as u64;
    assert!(bound > 0, "sampling bound must be positive");
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

    fn scored(
        target: Option<usize>,
        predicted_class: usize,
        best_similarity: f64,
    ) -> ScoredQuery {
        ScoredQuery {
            target,
            predicted_class,
            best_similarity,
        }
    }

    #[test]
    fn classwise_dynamic_program_allocates_error_mass_to_improve_coverage() {
        let mut samples = Vec::new();
        samples.extend((0..400).map(|_| scored(Some(0), 0, 0.95)));
        samples.extend((0..100).map(|_| scored(Some(1), 0, 0.75)));
        samples.extend((0..100).map(|_| scored(Some(1), 1, 0.95)));
        samples.extend((0..200).map(|_| scored(Some(0), 1, 0.75)));
        samples.extend((0..300).map(|_| scored(Some(1), 1, 0.65)));

        let budget = 0.20;
        let global = fit_global_threshold(&samples, budget);
        let classwise = fit_classwise_thresholds(&samples, 2, budget);
        let global_eval = evaluate_global(&samples, global);
        let classwise_eval = evaluate_classwise(&samples, &classwise);

        assert_eq!(global_eval.errors, 0);
        assert_eq!(global_eval.accepted, 500);
        assert_eq!(classwise_eval.errors, 200);
        assert_eq!(classwise_eval.accepted, 1_000);
        assert!(classwise_eval.error_mass() <= budget);
    }

    #[test]
    fn classwise_thresholds_are_deterministic_and_bounded_by_selection_budget() {
        let samples = (0..500)
            .map(|index| {
                let class = index % 2;
                let target = if index % 5 == 0 { Some(1 - class) } else { Some(class) };
                scored(target, class, 0.99 - (index % 12) as f64 * 0.05)
            })
            .collect::<Vec<_>>();
        let first = fit_classwise_thresholds(&samples, 2, SELECTION_ERROR_MASS_BUDGET);
        let second = fit_classwise_thresholds(&samples, 2, SELECTION_ERROR_MASS_BUDGET);
        assert_eq!(first, second);
        assert!(
            evaluate_classwise(&samples, &first).error_mass()
                <= SELECTION_ERROR_MASS_BUDGET + f64::EPSILON
        );
    }

    #[test]
    fn one_sided_error_mass_certificate_uses_the_full_query_population() {
        let alpha = 0.001;
        let upper = clopper_pearson_upper_bound(0, 4_096, alpha);
        assert!(upper < 0.005);
        assert!(
            clopper_pearson_upper_bound(20, 4_096, alpha) > upper
        );
    }

    #[test]
    fn query_generation_is_reproducible() {
        let prototypes = (0..CLASS_COUNT)
            .map(|class| PatternVector::seeded_random(128, 10 + class as u64))
            .collect::<Result<Vec<_>, _>>()
            .expect("prototype bank");
        let first = generate_queries(&prototypes, 42, 100).expect("first query set");
        let second = generate_queries(&prototypes, 42, 100).expect("repeat query set");
        assert_eq!(first, second);
    }
}
