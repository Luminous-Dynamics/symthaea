// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic pilot characterization of packed BinaryHV bundle capacity.
//!
//! This is deliberately a narrow first curve: a single 16,384-bit packed
//! bipolar representation, majority bundling, and set-membership queries.
//! It is not a full VSA comparison, a theoretical capacity proof, or a SOTA
//! claim. The output is CSV so exact runner artifacts can be retained.

use symthaea_core::hdc::binary_hv::BinaryHV;

const LOADS: &[usize] = &[1, 3, 7, 15, 31, 63, 127, 255, 511];
const TRIALS: usize = 8;
const NEGATIVE_CALIBRATION_QUERIES: usize = 512;
const NEGATIVE_EVALUATION_QUERIES: usize = 512;

#[derive(Debug, Clone, Copy, Default)]
struct TrialMetrics {
    mean_member_cosine: f64,
    mean_nonmember_cosine: f64,
    auc: f64,
    tpr_at_empirical_one_percent_fpr: f64,
    empirical_false_positive_rate: f64,
}

fn main() {
    println!(
        "load,mean_member_cosine,theoretical_member_cosine,empirical_minus_theory,mean_nonmember_cosine,mean_auc,mean_tpr_at_empirical_1pct_fpr,mean_evaluation_fpr,trial_count,negative_calibration_queries,negative_evaluation_queries"
    );
    for &load in LOADS {
        let metrics = (0..TRIALS)
            .map(|trial| run_trial(load, trial))
            .collect::<Vec<_>>();
        let mean_member = mean(metrics.iter().map(|m| m.mean_member_cosine));
        let mean_nonmember = mean(metrics.iter().map(|m| m.mean_nonmember_cosine));
        let mean_auc = mean(metrics.iter().map(|m| m.auc));
        let mean_tpr = mean(metrics.iter().map(|m| m.tpr_at_empirical_one_percent_fpr));
        let mean_fpr = mean(metrics.iter().map(|m| m.empirical_false_positive_rate));
        let theory_member = theoretical_member_cosine(load);
        let empirical_minus_theory = mean_member - theory_member;
        println!(
            "{load},{mean_member:.8},{theory_member:.8},{empirical_minus_theory:.8},{mean_nonmember:.8},{mean_auc:.8},{mean_tpr:.8},{mean_fpr:.8},{TRIALS},{NEGATIVE_CALIBRATION_QUERIES},{NEGATIVE_EVALUATION_QUERIES}"
        );
    }
}

fn run_trial(load: usize, trial: usize) -> TrialMetrics {
    let members = (0..load)
        .map(|slot| BinaryHV::random(stable_seed(trial, slot, 0xA11CE)))
        .collect::<Vec<_>>();
    let bundled = BinaryHV::bundle_safe(&members);

    let member_scores = members
        .iter()
        .map(|member| f64::from(bundled.cosine_similarity(member)))
        .collect::<Vec<_>>();
    let calibration_nonmembers = (0..NEGATIVE_CALIBRATION_QUERIES)
        .map(|slot| {
            let query = BinaryHV::random(stable_seed(trial, slot, 0xBADC0DE));
            f64::from(bundled.cosine_similarity(&query))
        })
        .collect::<Vec<_>>();
    let evaluation_nonmembers = (0..NEGATIVE_EVALUATION_QUERIES)
        .map(|slot| {
            let query = BinaryHV::random(stable_seed(trial, slot, 0xD15EA5E));
            f64::from(bundled.cosine_similarity(&query))
        })
        .collect::<Vec<_>>();

    let mut sorted_calibration = calibration_nonmembers.clone();
    sorted_calibration.sort_by(f64::total_cmp);
    let threshold_rank = (sorted_calibration.len() * 99).div_ceil(100);
    let threshold_index = threshold_rank
        .saturating_sub(1)
        .min(sorted_calibration.len() - 1);
    let threshold = sorted_calibration[threshold_index];

    // Fit the threshold on calibration negatives; report rates on the
    // independent evaluation-negative set.
    let true_positive_rate = member_scores
        .iter()
        .filter(|score| **score >= threshold)
        .count() as f64
        / member_scores.len() as f64;
    let false_positive_rate = evaluation_nonmembers
        .iter()
        .filter(|score| **score >= threshold)
        .count() as f64
        / evaluation_nonmembers.len() as f64;

    TrialMetrics {
        mean_member_cosine: mean(member_scores.iter().copied()),
        mean_nonmember_cosine: mean(evaluation_nonmembers.iter().copied()),
        auc: pairwise_auc(&member_scores, &evaluation_nonmembers),
        tpr_at_empirical_one_percent_fpr: true_positive_rate,
        empirical_false_positive_rate: false_positive_rate,
    }
}

/// Expected bipolar cosine between one stored member and an odd-load
/// independent majority bundle. For load N = 2m + 1 this is
/// C(2m, m) / 4^m; the asymptotic approximation is sqrt(2 / (pi * N)).
fn theoretical_member_cosine(load: usize) -> f64 {
    assert!(
        load > 0 && load % 2 == 1,
        "majority-bundle load must be positive and odd"
    );
    let pairs = (load - 1) / 2;
    (1..=pairs).fold(1.0, |value, k| value * (2 * k - 1) as f64 / (2 * k) as f64)
}

fn pairwise_auc(positive: &[f64], negative: &[f64]) -> f64 {
    let mut wins = 0.0;
    for positive_score in positive {
        for negative_score in negative {
            if positive_score > negative_score {
                wins += 1.0;
            } else if positive_score == negative_score {
                wins += 0.5;
            }
        }
    }
    wins / (positive.len() * negative.len()) as f64
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| {
        (sum + value, count + 1)
    });
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn stable_seed(trial: usize, slot: usize, domain: u64) -> u64 {
    (trial as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((slot as u64).wrapping_mul(0xD1B54A32D192ED03))
        ^ domain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_item_bundle_round_trips_exactly() {
        let item = BinaryHV::random(0xA11CE);
        let bundle = BinaryHV::bundle_safe(&[item]);
        assert_eq!(bundle, item);
        assert_eq!(bundle.cosine_similarity(&item), 1.0);
    }

    #[test]
    fn member_scores_separate_from_random_nonmembers_for_small_load() {
        let metrics = run_trial(7, 2026);
        assert!(metrics.mean_member_cosine > metrics.mean_nonmember_cosine);
        assert!(metrics.auc > 0.95);
        assert!(metrics.empirical_false_positive_rate <= 0.06);
    }

    #[test]
    fn theoretical_majority_member_similarity_matches_exact_small_loads() {
        assert!((theoretical_member_cosine(1) - 1.0).abs() < 1e-12);
        assert!((theoretical_member_cosine(3) - 0.5).abs() < 1e-12);
        assert!((theoretical_member_cosine(5) - 0.375).abs() < 1e-12);
    }

    #[test]
    fn pairwise_auc_handles_ties() {
        assert_eq!(pairwise_auc(&[0.5, 0.5], &[0.5, 0.5]), 0.5);
        assert_eq!(pairwise_auc(&[1.0], &[0.0]), 1.0);
        assert_eq!(pairwise_auc(&[0.0], &[1.0]), 0.0);
    }
}
