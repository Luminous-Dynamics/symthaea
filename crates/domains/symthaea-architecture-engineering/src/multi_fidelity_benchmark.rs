//! Known-oracle multi-fidelity benchmark for cost-aware architecture search.
//!
//! This is a deterministic synthetic instrument, not a model-training implementation.
//! It deliberately includes rank-preserving and rank-reversing fidelities so the promotion
//! scheduler can be measured against a full-fidelity random baseline at the same total
//! synthetic cost. Synthetic cost units are not FLOPs, wall-clock, dollars, or joules.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::multi_fidelity::{
    plan_promotions, FidelityLevel, FidelityObservation, PromotionPolicy,
};
use super::synthetic_search::{
    enumerate_candidates, evaluate_candidate, exhaustive_oracle, SyntheticCandidate,
    SyntheticLandscape,
};

const LOW_COST: u64 = 1;
const MID_COST: u64 = 4;
const HIGH_COST: u64 = 16;
const FULL_COST: u64 = 64;
const PROMOTION_FRACTION: f64 = 0.25;
const EXPLORATION_SHARE: f64 = 0.25;
const CONFIDENCE_MULTIPLIER: f64 = 1.0;

/// Fidelity-response regimes. All outcomes are synthetic and have an exhaustive oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiFidelityLandscape {
    /// Cheap scores broadly preserve the full-fidelity ordering.
    RankPreserving,
    /// The first two fidelities are intentionally anti-correlated with final quality.
    RankReversal,
    /// Early scores favor a misleading feature prior; signal improves gradually with fidelity.
    LateEmergingQuality,
}

/// One paired trial comparing multi-fidelity promotion and full-fidelity random search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityRun {
    pub landscape: MultiFidelityLandscape,
    pub seed: u64,
    pub total_cost_budget_units: u64,
    pub promotion_policy_cost_units: u64,
    pub random_baseline_cost_units: u64,
    pub promoted_full_fidelity_count: usize,
    pub random_full_fidelity_count: usize,
    pub promotion_best_candidate_id: String,
    pub random_best_candidate_id: String,
    pub promotion_best_quality: f64,
    pub random_best_quality: f64,
    pub promotion_simple_regret: f64,
    pub random_simple_regret: f64,
    pub promotion_hit_global_optimum: bool,
    pub random_hit_global_optimum: bool,
    /// Spearman rank correlation between F0 scores and the known full-fidelity oracle.
    /// This is diagnostic-only and is available because this is a synthetic test.
    pub f0_quality_spearman: f64,
}

/// Aggregate of paired seeds; standard deviations use sample denominator n - 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityAggregate {
    pub landscape: MultiFidelityLandscape,
    pub seed_count: usize,
    pub total_cost_budget_units: u64,
    pub mean_promotion_regret: f64,
    pub mean_random_regret: f64,
    /// Positive values favor multi-fidelity promotion (random regret minus promotion regret).
    pub mean_paired_regret_improvement: f64,
    pub paired_regret_improvement_sample_sd: f64,
    pub promotion_global_optimum_hit_rate: f64,
    pub random_global_optimum_hit_rate: f64,
    pub mean_f0_quality_spearman: f64,
    pub mean_promoted_full_fidelity_count: f64,
    pub mean_random_full_fidelity_count: f64,
}

/// Four cumulative-cost levels. The full-fidelity run is 64 synthetic cost units.
pub fn default_fidelity_ladder() -> Vec<FidelityLevel> {
    vec![
        FidelityLevel {
            id: "f0_proxy".to_owned(),
            cumulative_cost_units: LOW_COST,
        },
        FidelityLevel {
            id: "f1_short".to_owned(),
            cumulative_cost_units: MID_COST,
        },
        FidelityLevel {
            id: "f2_intermediate".to_owned(),
            cumulative_cost_units: HIGH_COST,
        },
        FidelityLevel {
            id: "f3_full".to_owned(),
            cumulative_cost_units: FULL_COST,
        },
    ]
}

/// Run one paired, equal-budget synthetic experiment.
pub fn run_multi_fidelity_trial(
    landscape: MultiFidelityLandscape,
    seed: u64,
) -> Result<MultiFidelityRun, String> {
    let candidates = enumerate_candidates();
    let quality_landscape = quality_landscape(landscape);
    let oracle = exhaustive_oracle(quality_landscape, 10);
    let optimum_ids: HashSet<&str> = oracle
        .global_optimum_ids
        .iter()
        .map(String::as_str)
        .collect();
    let candidate_by_id: HashMap<String, SyntheticCandidate> = candidates
        .iter()
        .copied()
        .map(|candidate| (candidate.id(), candidate))
        .collect();
    let levels = default_fidelity_ladder();

    let mut active = candidates.clone();
    let mut total_cost = (active.len() as u64)
        .checked_mul(levels[0].cumulative_cost_units)
        .ok_or_else(|| "initial fidelity cost overflow".to_owned())?;

    let f0_scores: Vec<f64> = candidates
        .iter()
        .copied()
        .map(|candidate| score_at_fidelity(candidate, landscape, 0, seed).0)
        .collect();
    let true_scores: Vec<f64> = candidates
        .iter()
        .copied()
        .map(|candidate| evaluate_candidate(candidate, quality_landscape).quality)
        .collect();
    let f0_quality_spearman = spearman(&f0_scores, &true_scores);

    // Promotion decisions only see observations from the current fidelity. No later-rung
    // or oracle score is passed into the policy.
    for fidelity_index in 0..(levels.len() - 1) {
        let observations: Vec<FidelityObservation> = active
            .iter()
            .copied()
            .map(|candidate| {
                let (mean_score, standard_error) =
                    score_at_fidelity(candidate, landscape, fidelity_index, seed);
                FidelityObservation {
                    candidate_id: candidate.id(),
                    fidelity_index,
                    mean_score,
                    standard_error,
                }
            })
            .collect();

        let decision = plan_promotions(
            &levels,
            &observations,
            fidelity_index,
            &PromotionPolicy {
                promotion_fraction: PROMOTION_FRACTION,
                exploration_share: EXPLORATION_SHARE,
                confidence_multiplier: CONFIDENCE_MULTIPLIER,
                seed: seed ^ ((fidelity_index as u64 + 1) * 0x9e37_79b9_7f4a_7c15),
            },
        )
        .map_err(|error| format!("fidelity {fidelity_index} promotion failed: {error}"))?;

        total_cost = total_cost
            .checked_add(decision.estimated_incremental_cost_units)
            .ok_or_else(|| "promotion cost overflow".to_owned())?;

        active = decision
            .promoted_candidate_ids
            .iter()
            .map(|id| {
                candidate_by_id
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("scheduler returned unknown candidate {id:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    if total_cost == 0 || active.is_empty() {
        return Err("synthetic scheduler produced no full-fidelity candidates".to_owned());
    }

    // The baseline spends exactly the same cost budget on full-fidelity candidates. Because
    // all synthetic full runs have fixed cost 64, this baseline evaluates floor(budget/64)
    // candidates sampled without replacement, and charges the unused remainder as unused.
    let random_full_fidelity_count = (total_cost / FULL_COST) as usize;
    if random_full_fidelity_count == 0 {
        return Err("budget cannot fund one full-fidelity random evaluation".to_owned());
    }
    let mut random_candidates = candidates.clone();
    shuffle_candidates(&mut random_candidates, seed ^ 0xa076_1d64_78bd_642f);
    random_candidates.truncate(random_full_fidelity_count.min(random_candidates.len()));

    let promotion_best = best_candidate(&active, quality_landscape)?;
    let random_best = best_candidate(&random_candidates, quality_landscape)?;
    let promotion_best_quality = evaluate_candidate(promotion_best, quality_landscape).quality;
    let random_best_quality = evaluate_candidate(random_best, quality_landscape).quality;
    let global_quality = oracle.global_optimum_quality;

    Ok(MultiFidelityRun {
        landscape,
        seed,
        total_cost_budget_units: total_cost,
        promotion_policy_cost_units: total_cost,
        random_baseline_cost_units: (random_candidates.len() as u64) * FULL_COST,
        promoted_full_fidelity_count: active.len(),
        random_full_fidelity_count: random_candidates.len(),
        promotion_best_candidate_id: promotion_best.id(),
        random_best_candidate_id: random_best.id(),
        promotion_best_quality,
        random_best_quality,
        promotion_simple_regret: (global_quality - promotion_best_quality).max(0.0),
        random_simple_regret: (global_quality - random_best_quality).max(0.0),
        promotion_hit_global_optimum: optimum_ids.contains(promotion_best.id().as_str()),
        random_hit_global_optimum: optimum_ids.contains(random_best.id().as_str()),
        f0_quality_spearman,
    })
}

/// Run paired trials at consecutive seeds and summarize regret improvement.
pub fn run_multi_fidelity_sweep(
    landscape: MultiFidelityLandscape,
    seed_count: usize,
) -> Result<(Vec<MultiFidelityRun>, MultiFidelityAggregate), String> {
    if !(2..=10_000).contains(&seed_count) {
        return Err("seed_count must be in 2..=10000".to_owned());
    }

    let mut runs = Vec::with_capacity(seed_count);
    for seed in 1..=seed_count as u64 {
        runs.push(run_multi_fidelity_trial(
            landscape,
            seed.wrapping_add(0x4d46_4944_4500_0000),
        )?);
    }

    let budget = runs
        .first()
        .ok_or_else(|| "no multi-fidelity runs produced".to_owned())?
        .total_cost_budget_units;
    if runs.iter().any(|run| {
        run.landscape != landscape || run.total_cost_budget_units != budget
    }) {
        return Err("aggregate contains a mixed landscape or cost budget".to_owned());
    }

    let paired_improvements: Vec<f64> = runs
        .iter()
        .map(|run| run.random_simple_regret - run.promotion_simple_regret)
        .collect();
    let mean_promotion_regret = mean(
        &runs
            .iter()
            .map(|run| run.promotion_simple_regret)
            .collect::<Vec<_>>(),
    );
    let mean_random_regret = mean(
        &runs
            .iter()
            .map(|run| run.random_simple_regret)
            .collect::<Vec<_>>(),
    );

    let aggregate = MultiFidelityAggregate {
        landscape,
        seed_count: runs.len(),
        total_cost_budget_units: budget,
        mean_promotion_regret,
        mean_random_regret,
        mean_paired_regret_improvement: mean(&paired_improvements),
        paired_regret_improvement_sample_sd: sample_sd(&paired_improvements),
        promotion_global_optimum_hit_rate: runs
            .iter()
            .filter(|run| run.promotion_hit_global_optimum)
            .count() as f64
            / runs.len() as f64,
        random_global_optimum_hit_rate: runs
            .iter()
            .filter(|run| run.random_hit_global_optimum)
            .count() as f64
            / runs.len() as f64,
        mean_f0_quality_spearman: mean(
            &runs
                .iter()
                .map(|run| run.f0_quality_spearman)
                .collect::<Vec<_>>(),
        ),
        mean_promoted_full_fidelity_count: mean(
            &runs
                .iter()
                .map(|run| run.promoted_full_fidelity_count as f64)
                .collect::<Vec<_>>(),
        ),
        mean_random_full_fidelity_count: mean(
            &runs
                .iter()
                .map(|run| run.random_full_fidelity_count as f64)
                .collect::<Vec<_>>(),
        ),
    };
    Ok((runs, aggregate))
}

fn quality_landscape(landscape: MultiFidelityLandscape) -> SyntheticLandscape {
    match landscape {
        MultiFidelityLandscape::RankPreserving | MultiFidelityLandscape::LateEmergingQuality => {
            SyntheticLandscape::SmoothTradeoff
        }
        MultiFidelityLandscape::RankReversal => SyntheticLandscape::Deceptive,
    }
}

/// Return the fidelity score mean and its synthetic standard-error estimate.
///
/// The estimate is intentionally specified by the synthetic generator. It is not a calibrated
/// uncertainty model; a later benchmark version should include miscalibrated-SE controls.
fn score_at_fidelity(
    candidate: SyntheticCandidate,
    landscape: MultiFidelityLandscape,
    fidelity_index: usize,
    seed: u64,
) -> (f64, f64) {
    let quality = evaluate_candidate(candidate, quality_landscape(landscape)).quality;
    let target_distance = 0.12 * (3.0 - candidate.width as f64).abs()
        + 0.11 * (2.0 - candidate.depth as f64).abs()
        + 0.12 * (3.0 - candidate.attention as f64).abs()
        + 0.10 * (1.0 - candidate.recurrence as f64).abs();
    let target_prior = (1.0 - target_distance).clamp(0.0, 1.0);
    let misleading_prior = (1.0 - target_prior).clamp(0.0, 1.0);

    let (expected, noise_scale, base_error) = match landscape {
        MultiFidelityLandscape::RankPreserving => match fidelity_index {
            0 => (0.65 * quality + 0.35 * target_prior, 0.035, 0.025),
            1 => (0.82 * quality + 0.18 * target_prior, 0.020, 0.015),
            2 => (0.96 * quality + 0.04 * target_prior, 0.008, 0.006),
            _ => (quality, 0.0, 0.0),
        },
        MultiFidelityLandscape::RankReversal => match fidelity_index {
            0 => (1.0 - quality, 0.025, 0.020),
            1 => (0.85 * (1.0 - quality) + 0.15 * misleading_prior, 0.018, 0.014),
            2 => (0.55 * quality + 0.45 * target_prior, 0.012, 0.010),
            _ => (quality, 0.0, 0.0),
        },
        MultiFidelityLandscape::LateEmergingQuality => match fidelity_index {
            0 => (0.90 * misleading_prior + 0.10 * quality, 0.030, 0.025),
            1 => (0.62 * misleading_prior + 0.38 * quality, 0.020, 0.018),
            2 => (0.25 * misleading_prior + 0.75 * quality, 0.010, 0.008),
            _ => (quality, 0.0, 0.0),
        },
    };

    let key = candidate
        .width
        .wrapping_mul(64)
        .wrapping_add(candidate.depth.wrapping_mul(16))
        .wrapping_add(candidate.attention.wrapping_mul(4))
        .wrapping_add(candidate.recurrence) as u64;
    let noise = deterministic_noise(
        seed ^ key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (fidelity_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
    ) * noise_scale;
    let heteroscedastic = ((key % 3) as f64) * 0.004;
    let standard_error = (base_error + heteroscedastic).max(0.0);
    ((expected + noise).clamp(0.0, 1.0), standard_error)
}

fn best_candidate(
    candidates: &[SyntheticCandidate],
    landscape: SyntheticLandscape,
) -> Result<SyntheticCandidate, String> {
    candidates
        .iter()
        .copied()
        .max_by(|left, right| {
            evaluate_candidate(*left, landscape)
                .quality
                .total_cmp(&evaluate_candidate(*right, landscape).quality)
                .then_with(|| right.id().cmp(&left.id()))
        })
        .ok_or_else(|| "cannot select the best candidate from an empty set".to_owned())
}

fn shuffle_candidates(candidates: &mut [SyntheticCandidate], mut state: u64) {
    for index in (1..candidates.len()).rev() {
        state = splitmix64(state);
        let selected = (state as usize) % (index + 1);
        candidates.swap(index, selected);
    }
}

fn deterministic_noise(seed: u64) -> f64 {
    let value = splitmix64(seed);
    let unit = (value >> 11) as f64 / ((1u64 << 53) as f64);
    2.0 * unit - 1.0
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]).then_with(|| left.cmp(right)));
    let mut result = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && values[order[start]].total_cmp(&values[order[end]]).is_eq()
        {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for index in start..end {
            result[order[index]] = average_rank;
        }
        start = end;
    }
    result
}

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    pearson(&ranks(left), &ranks(right))
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let count = left.len() as f64;
    let mean_left = left.iter().sum::<f64>() / count;
    let mean_right = right.iter().sum::<f64>() / count;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - mean_left) * (right - mean_right))
        .sum::<f64>();
    let variance_left = left
        .iter()
        .map(|value| (value - mean_left).powi(2))
        .sum::<f64>();
    let variance_right = right
        .iter()
        .map(|value| (value - mean_right).powi(2))
        .sum::<f64>();
    let denominator = (variance_left * variance_right).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    let sum_squares = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>();
    (sum_squares / (values.len() - 1) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fidelity_ladder_has_strictly_increasing_cumulative_cost() {
        let levels = default_fidelity_ladder();
        assert_eq!(levels.len(), 4);
        assert!(levels.windows(2).all(|pair| {
            pair[0].cumulative_cost_units < pair[1].cumulative_cost_units
        }));
    }

    #[test]
    fn promotion_and_random_baseline_use_equal_cost_budget() {
        let run = run_multi_fidelity_trial(MultiFidelityLandscape::RankPreserving, 7).unwrap();
        assert_eq!(run.total_cost_budget_units, 832);
        assert_eq!(run.promotion_policy_cost_units, run.total_cost_budget_units);
        assert_eq!(
            run.random_baseline_cost_units,
            run.total_cost_budget_units
        );
        assert_eq!(run.promoted_full_fidelity_count, 4);
        assert_eq!(run.random_full_fidelity_count, 13);
    }

    #[test]
    fn same_seed_produces_identical_report() {
        let first = run_multi_fidelity_trial(MultiFidelityLandscape::RankPreserving, 991).unwrap();
        let second = run_multi_fidelity_trial(MultiFidelityLandscape::RankPreserving, 991).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rank_reversal_control_has_negative_low_fidelity_rank_correlation() {
        let run = run_multi_fidelity_trial(MultiFidelityLandscape::RankReversal, 29).unwrap();
        assert!(run.f0_quality_spearman < 0.0);
    }

    #[test]
    fn rank_preserving_fidelity_is_more_aligned_than_rank_reversal() {
        let preserving =
            run_multi_fidelity_trial(MultiFidelityLandscape::RankPreserving, 29).unwrap();
        let reversal =
            run_multi_fidelity_trial(MultiFidelityLandscape::RankReversal, 29).unwrap();
        assert!(preserving.f0_quality_spearman > reversal.f0_quality_spearman);
    }

    #[test]
    fn synthetic_sweep_rejects_too_few_or_too_many_seeds() {
        assert!(run_multi_fidelity_sweep(MultiFidelityLandscape::RankPreserving, 1).is_err());
        assert!(run_multi_fidelity_sweep(MultiFidelityLandscape::RankPreserving, 10_001).is_err());
    }

    #[test]
    fn sweep_preserves_paired_seed_count_and_cost_budget() {
        let (runs, aggregate) =
            run_multi_fidelity_sweep(MultiFidelityLandscape::RankPreserving, 3).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(aggregate.seed_count, 3);
        assert_eq!(aggregate.total_cost_budget_units, 832);
        assert!(aggregate.mean_paired_regret_improvement.is_finite());
    }
}
