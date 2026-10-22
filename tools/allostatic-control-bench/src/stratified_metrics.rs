// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stratified, task-balanced analysis of paired normalized control scores.
//!
//! Each task must be mapped onto a predeclared utility scale in [0, 1], where
//! higher is better. Raw task scores with incompatible units must not be pooled.
//! Bootstrap resampling is stratified within each task, while the task suite is
//! treated as fixed; these intervals do not estimate generalization to unseen tasks.

use std::collections::{BTreeMap, BTreeSet};

const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaskEpisode {
    pub seed: u64,
    pub baseline_score: f64,
    pub candidate_score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskSummary {
    pub task: String,
    pub n_pairs: usize,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    pub mean_delta: f64,
    pub iqm_delta: f64,
    /// Fraction of paired episodes won by the candidate; ties count as one half.
    pub paired_win_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StratifiedSummary {
    pub task_count: usize,
    pub total_pairs: usize,
    pub macro_baseline_mean: f64,
    pub macro_candidate_mean: f64,
    pub macro_mean_delta: f64,
    pub macro_mean_delta_ci95_low: f64,
    pub macro_mean_delta_ci95_high: f64,
    pub macro_iqm_delta: f64,
    pub macro_iqm_delta_ci95_low: f64,
    pub macro_iqm_delta_ci95_high: f64,
    pub macro_paired_win_rate: f64,
    pub macro_paired_win_rate_ci95_low: f64,
    pub macro_paired_win_rate_ci95_high: f64,
    pub fraction_tasks_positive_mean_delta: f64,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
}

#[derive(Clone, Copy)]
struct TaskDraw {
    mean_delta: f64,
    iqm_delta: f64,
    paired_win_rate: f64,
}

impl StratifiedSummary {
    pub fn from_tasks(
        tasks: &BTreeMap<String, Vec<TaskEpisode>>,
        bootstrap_replicates: usize,
        random_seed: u64,
    ) -> Result<(Vec<TaskSummary>, Self), String> {
        if tasks.len() < 2 {
            return Err(
                "at least two distinct tasks are required for stratified aggregation".to_owned(),
            );
        }
        if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES {
            return Err(format!(
                "at least {MIN_BOOTSTRAP_REPLICATES} bootstrap replicates are required"
            ));
        }

        let mut task_summaries = Vec::with_capacity(tasks.len());
        let mut deltas_by_task: Vec<Vec<f64>> = Vec::with_capacity(tasks.len());
        let mut total_pairs = 0_usize;

        for (task, episodes) in tasks {
            if !is_slug(task) {
                return Err(format!("invalid task identifier: {task:?}"));
            }
            if episodes.len() < 2 {
                return Err(format!("task {task} needs at least two paired episodes"));
            }

            let mut seeds = BTreeSet::new();
            let mut baseline = Vec::with_capacity(episodes.len());
            let mut candidate = Vec::with_capacity(episodes.len());
            let mut deltas = Vec::with_capacity(episodes.len());

            for episode in episodes {
                if !seeds.insert(episode.seed) {
                    return Err(format!("duplicate seed {} for task {task}", episode.seed));
                }
                if !episode.baseline_score.is_finite()
                    || !episode.candidate_score.is_finite()
                    || !(0.0..=1.0).contains(&episode.baseline_score)
                    || !(0.0..=1.0).contains(&episode.candidate_score)
                {
                    return Err(format!(
                        "task {task} seed {} must have finite normalized scores in [0, 1]",
                        episode.seed
                    ));
                }
                baseline.push(episode.baseline_score);
                candidate.push(episode.candidate_score);
                deltas.push(episode.candidate_score - episode.baseline_score);
            }

            let summary = summarize_task(task, &baseline, &candidate, &deltas);
            total_pairs = total_pairs
                .checked_add(episodes.len())
                .ok_or("total paired episode count overflow")?;
            task_summaries.push(summary);
            deltas_by_task.push(deltas);
        }

        let task_count = task_summaries.len();
        let macro_baseline_mean = mean(
            &task_summaries
                .iter()
                .map(|summary| summary.baseline_mean)
                .collect::<Vec<_>>(),
        );
        let macro_candidate_mean = mean(
            &task_summaries
                .iter()
                .map(|summary| summary.candidate_mean)
                .collect::<Vec<_>>(),
        );
        let macro_mean_delta = mean(
            &task_summaries
                .iter()
                .map(|summary| summary.mean_delta)
                .collect::<Vec<_>>(),
        );
        let macro_iqm_delta = mean(
            &task_summaries
                .iter()
                .map(|summary| summary.iqm_delta)
                .collect::<Vec<_>>(),
        );
        let macro_paired_win_rate = mean(
            &task_summaries
                .iter()
                .map(|summary| summary.paired_win_rate)
                .collect::<Vec<_>>(),
        );
        let fraction_tasks_positive_mean_delta = task_summaries
            .iter()
            .filter(|summary| summary.mean_delta > 0.0)
            .count() as f64
            / task_count as f64;

        let effective_seed = if random_seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            random_seed
        };
        let mut random_state = effective_seed;
        let mut bootstrap_mean = Vec::with_capacity(bootstrap_replicates);
        let mut bootstrap_iqm = Vec::with_capacity(bootstrap_replicates);
        let mut bootstrap_win_rate = Vec::with_capacity(bootstrap_replicates);

        for _ in 0..bootstrap_replicates {
            let mut task_draws = Vec::with_capacity(task_count);
            for deltas in &deltas_by_task {
                let mut sampled_deltas = Vec::with_capacity(deltas.len());

                for _ in 0..deltas.len() {
                    let index = (next_random(&mut random_state) % deltas.len() as u64) as usize;
                    sampled_deltas.push(deltas[index]);
                }
                task_draws.push(TaskDraw {
                    mean_delta: mean(&sampled_deltas),
                    iqm_delta: interquartile_mean(&sampled_deltas),
                    paired_win_rate: paired_win_rate(&sampled_deltas),
                });
            }

            bootstrap_mean.push(mean(
                &task_draws.iter().map(|draw| draw.mean_delta).collect::<Vec<_>>(),
            ));
            bootstrap_iqm.push(mean(
                &task_draws.iter().map(|draw| draw.iqm_delta).collect::<Vec<_>>(),
            ));
            bootstrap_win_rate.push(mean(
                &task_draws
                    .iter()
                    .map(|draw| draw.paired_win_rate)
                    .collect::<Vec<_>>(),
            ));
        }

        for values in [&mut bootstrap_mean, &mut bootstrap_iqm, &mut bootstrap_win_rate] {
            values.sort_by(f64::total_cmp);
        }
        let summary = Self {
            task_count,
            total_pairs,
            macro_baseline_mean,
            macro_candidate_mean,
            macro_mean_delta,
            macro_mean_delta_ci95_low: quantile_sorted(&bootstrap_mean, 0.025),
            macro_mean_delta_ci95_high: quantile_sorted(&bootstrap_mean, 0.975),
            macro_iqm_delta,
            macro_iqm_delta_ci95_low: quantile_sorted(&bootstrap_iqm, 0.025),
            macro_iqm_delta_ci95_high: quantile_sorted(&bootstrap_iqm, 0.975),
            macro_paired_win_rate,
            macro_paired_win_rate_ci95_low: quantile_sorted(&bootstrap_win_rate, 0.025),
            macro_paired_win_rate_ci95_high: quantile_sorted(&bootstrap_win_rate, 0.975),
            fraction_tasks_positive_mean_delta,
            bootstrap_replicates,
            bootstrap_seed: effective_seed,
        };

        let values = [
            summary.macro_baseline_mean,
            summary.macro_candidate_mean,
            summary.macro_mean_delta,
            summary.macro_mean_delta_ci95_low,
            summary.macro_mean_delta_ci95_high,
            summary.macro_iqm_delta,
            summary.macro_iqm_delta_ci95_low,
            summary.macro_iqm_delta_ci95_high,
            summary.macro_paired_win_rate,
            summary.macro_paired_win_rate_ci95_low,
            summary.macro_paired_win_rate_ci95_high,
            summary.fraction_tasks_positive_mean_delta,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err("non-finite stratified statistic; refusing to emit aggregate".to_owned());
        }

        Ok((task_summaries, summary))
    }
}

fn summarize_task(
    task: &str,
    baseline: &[f64],
    candidate: &[f64],
    deltas: &[f64],
) -> TaskSummary {
    TaskSummary {
        task: task.to_owned(),
        n_pairs: deltas.len(),
        baseline_mean: mean(baseline),
        candidate_mean: mean(candidate),
        mean_delta: mean(deltas),
        iqm_delta: interquartile_mean(deltas),
        paired_win_rate: paired_win_rate(deltas),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Integral of the empirical quantile function over [0.25, 0.75], normalized
/// by its 0.5 width. Fractional boundary weights handle sample sizes not divisible by four.
fn interquartile_mean(values: &[f64]) -> f64 {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let mut weighted_sum = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        let start = index as f64 / n;
        let end = (index + 1) as f64 / n;
        let weight = end.min(0.75) - start.max(0.25);
        if weight > 0.0 {
            weighted_sum += value * weight;
        }
    }
    weighted_sum / 0.5
}

fn paired_win_rate(deltas: &[f64]) -> f64 {
    deltas
        .iter()
        .map(|delta| {
            if *delta > 0.0 {
                1.0
            } else if *delta == 0.0 {
                0.5
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / deltas.len() as f64
}

fn quantile_sorted(values: &[f64], probability: f64) -> f64 {
    debug_assert!(!values.is_empty());
    let position = probability.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn is_slug(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| {
                byte.is_ascii_alphanumeric()
                    || byte == b'_'
                    || byte == b'-'
                    || byte == b'.'
            })
}

#[cfg(test)]
mod tests {
    use super::{StratifiedSummary, TaskEpisode};
    use std::collections::BTreeMap;

    fn fixture() -> BTreeMap<String, Vec<TaskEpisode>> {
        BTreeMap::from([
            (
                "cartpole".to_owned(),
                (0..30)
                    .map(|seed| TaskEpisode {
                        seed,
                        baseline_score: 0.4,
                        candidate_score: 0.6,
                    })
                    .collect(),
            ),
            (
                "quadrotor".to_owned(),
                (0..10)
                    .map(|seed| TaskEpisode {
                        seed,
                        baseline_score: 0.7,
                        candidate_score: 0.6,
                    })
                    .collect(),
            ),
        ])
    }

    #[test]
    fn macro_aggregation_weights_tasks_equally() {
        let (_, summary) =
            StratifiedSummary::from_tasks(&fixture(), 1_000, 17).expect("valid task fixture");
        assert!((summary.macro_mean_delta - 0.05).abs() < 1e-12);
        assert!((summary.macro_iqm_delta - 0.05).abs() < 1e-12);
        assert!((summary.macro_paired_win_rate - 0.5).abs() < 1e-12);
        assert_eq!(summary.task_count, 2);
        assert_eq!(summary.total_pairs, 40);
    }

    #[test]
    fn stratified_bootstrap_is_deterministic() {
        let first = StratifiedSummary::from_tasks(&fixture(), 1_000, 17).expect("valid data");
        let second = StratifiedSummary::from_tasks(&fixture(), 1_000, 17).expect("valid data");
        assert_eq!(first, second);
    }

    #[test]
    fn duplicate_seeds_are_rejected_per_task() {
        let mut tasks = fixture();
        tasks
            .get_mut("cartpole")
            .expect("task exists")
            .push(TaskEpisode {
                seed: 0,
                baseline_score: 0.4,
                candidate_score: 0.6,
            });
        assert!(StratifiedSummary::from_tasks(&tasks, 1_000, 17)
            .expect_err("duplicate seed must fail")
            .contains("duplicate"));
    }

    #[test]
    fn non_normalized_scores_are_rejected() {
        let mut tasks = fixture();
        tasks
            .get_mut("cartpole")
            .expect("task exists")[0]
            .candidate_score = 1.01;
        assert!(StratifiedSummary::from_tasks(&tasks, 1_000, 17).is_err());
    }
}
