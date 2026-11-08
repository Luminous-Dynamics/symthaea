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
const NEGATIVE_QUERIES: usize = 128;

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
        "load,mean_member_cosine,mean_nonmember_cosine,mean_auc,mean_tpr_at_empirical_1pct_fpr,mean_empirical_fpr,trial_count,negative_queries"
    );
    for &load in LOADS {
        let metrics = (0..TRIALS)
            .map(|trial| run_trial(load, trial))
            .collect::<Vec<_>>();
        let mean_member = mean(metrics.iter().map(|m| m.mean_member_cosine));
        let mean_nonmember = mean(metrics.iter().map(|m| m.mean_nonmember_cosine));
        let mean_auc = mean(metrics.iter().map(|m| m.auc));
        let mean_tpr = mean(
            metrics
                .iter()
                .map(|m| m.tpr_at_empirical_one_percent_fpr),
        );
        let mean_fpr = mean(
            metrics
                .iter()
                .map(|m| m.empirical_false_positive_rate),
        );
        println!(
            "{load},{mean_member:.8},{mean_nonmember:.8},{mean_auc:.8},{mean_tpr:.8},{mean_fpr:.8},{TRIALS},{NEGATIVE_QUERIES}"
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
    let nonmember_scores = (0..NEGATIVE_QUERIES)
        .map(|slot| {
            let query = BinaryHV::random(stable_seed(trial, slot, 0xBADC0DE));
            f64::from(bundled.cosine_similarity(&query))
        })
        .collect::<Vec<_>>();

    let mut sorted_nonmembers = nonmember_scores.clone();
    sorted_nonmembers.sort_by(f64::total_cmp);
    let threshold_index = ((sorted_nonmembers.len() - 1) * 99) / 100;
    let threshold = sorted_nonmembers[threshold_index];

    let true_positive_rate = member_scores
        .iter()
        .filter(|score| **score > threshold)
        .count() as f64
        / member_scores.len() as f64;
    let false_positive_rate = nonmember_scores
        .iter()
        .filter(|score| **score > threshold)
        .count() as f64
        / nonmember_scores.len() as f64;

    TrialMetrics {
        mean_member_cosine: mean(member_scores.iter().copied()),
        mean_nonmember_cosine: mean(nonmember_scores.iter().copied()),
        auc: pairwise_auc(&member_scores, &nonmember_scores),
        tpr_at_empirical_one_percent_fpr: true_positive_rate,
        empirical_false_positive_rate: false_positive_rate,
    }
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
        assert!(metrics.empirical_false_positive_rate <= 0.02);
    }

    #[test]
    fn pairwise_auc_handles_ties() {
        assert_eq!(pairwise_auc(&[0.5, 0.5], &[0.5, 0.5]), 0.5);
        assert_eq!(pairwise_auc(&[1.0], &[0.0]), 1.0);
        assert_eq!(pairwise_auc(&[0.0], &[1.0]), 0.0);
    }
}
