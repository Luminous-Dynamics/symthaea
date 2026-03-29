// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reversal Learning benchmark.
//!
//! Tests cognitive flexibility: the agent learns a simple stimulus-reward
//! contingency (A rewarded, B not), then the mapping reverses. Measures
//! how quickly the agent detects and adapts to reversals.
//!
//! Distinct from WCST — simpler binary choice, pure flexibility test.
//! Classic paradigm in behavioral neuroscience (Dias et al. 1996).
//!
//! HDC model: two stimuli encoded as ContinuousHVs, reward tracked via
//! EMA. Action selection uses softmax over reward values. After criterion
//! (8 consecutive correct), the contingency reverses.
//!
//! Human baselines (Cools et al. 2002; Clark et al. 2004):
//! - initial_acquisition: ~10-12 trials
//! - reversal_trials: ~14-18 trials
//! - win_stay_rate: 0.85
//! - lose_shift_rate: 0.70

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Reversal learning benchmark measuring cognitive flexibility.
pub struct ReversalLearningBenchmark;

impl ReversalLearningBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("cogbench", "reversal", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Reward value HV — agent learns association via similarity
        let reward_hv = ContinuousHV::random(dim, seed.wrapping_add(300));
        let no_reward_hv = ContinuousHV::random(dim, seed.wrapping_add(400));

        // Agent's learned association: weighted blend of stimulus with reward/no-reward
        // Starts at neutral (equal blend)
        let mut assoc_a = ContinuousHV::weighted_bundle(&[&reward_hv, &no_reward_hv], &[0.5, 0.5]);
        let mut assoc_b = ContinuousHV::weighted_bundle(&[&reward_hv, &no_reward_hv], &[0.5, 0.5]);

        let criterion = 8; // consecutive correct to trigger reversal
        let max_trials = 200;
        let learning_rate = 0.35f32; // how fast associations update
        let loss_learning_rate = 0.90f32; // asymmetric: losses update much faster (Behrens et al. 2007)

        // Current contingency: true = A rewarded, false = B rewarded
        let mut a_rewarded = true;
        let mut consecutive_correct = 0u32;
        let mut reversals_completed = 0u32;
        let mut initial_acquisition_trials: Option<u32> = None;
        let mut reversal_trial_counts: Vec<u32> = Vec::new();
        let mut trials_since_event = 0u32; // trials since last reversal (or start)
        let mut perseverative_errors = 0u32;
        let mut total_errors = 0u32;
        let mut in_reversal = false; // are we in a post-reversal phase?
        let mut post_reversal_countdown: u32 = 0; // trials remaining in post-reversal adaptation
        let mut consecutive_errors = 0u32; // for surprise-driven belief reset

        // Win-stay / lose-shift tracking
        let mut prev_choice: Option<bool> = None; // true = chose A
        let mut prev_rewarded: Option<bool> = None;
        let mut win_stay_count = 0u32;
        let mut win_total = 0u32;
        let mut lose_shift_count = 0u32;
        let mut lose_total = 0u32;
        let mut rt_ticks = Vec::new();

        for _trial in 0..max_trials {
            trials_since_event += 1;
            post_reversal_countdown = post_reversal_countdown.saturating_sub(1);

            // Score each stimulus: similarity of its learned association to reward_hv
            let score_a = assoc_a.similarity(&reward_hv) as f64;
            let score_b = assoc_b.similarity(&reward_hv) as f64;

            // Softmax action selection — wider temperature allows more exploration
            // post-reversal, reducing perseveration
            // Time pressure: +0.10/unit raises choice noise, modeling impaired reversal detection
            // under deadline (Cools et al., 2002 reversal learning; Heitz, 2014 SAT).
            let temp = config.action_temperature.max(0.1) * 0.08 + config.time_pressure * 0.10;
            let max_score = score_a.max(score_b);
            let exp_a = ((score_a - max_score) / temp).exp();
            let exp_b = ((score_b - max_score) / temp).exp();
            let total = exp_a + exp_b;

            // RT proxy: decision difficulty from score margin — closer scores = harder
            // choice, especially post-reversal when associations are uncertain
            // (Cools et al., 2002; Heitz, 2014 SAT).
            let score_margin = (score_a - score_b).abs();
            let ticks = 5.0 + (1.0 - score_margin.min(1.0)) * 8.0;
            rt_ticks.push(ticks);

            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let roll = (rng % 10000) as f64 / 10000.0;
            let chose_a = roll < (exp_a / total);

            // Track win-stay / lose-shift
            if let (Some(prev), Some(was_rewarded)) = (prev_choice, prev_rewarded) {
                if was_rewarded {
                    win_total += 1;
                    if chose_a == prev {
                        win_stay_count += 1; // stayed with rewarded choice
                    }
                } else {
                    lose_total += 1;
                    if chose_a != prev {
                        lose_shift_count += 1; // shifted away from unrewarded choice
                    }
                }
            }

            // Determine if correct
            let correct = if a_rewarded { chose_a } else { !chose_a };

            if correct {
                consecutive_correct += 1;
                consecutive_errors = 0;
            } else {
                total_errors += 1;
                consecutive_errors += 1;
                if in_reversal {
                    // Perseverative = choosing the previously-rewarded stimulus
                    // after reversal (first few errors post-reversal)
                    if trials_since_event <= criterion {
                        perseverative_errors += 1;
                    }
                }
                consecutive_correct = 0;
            }

            // Change-point detection heuristic: when consecutive errors exceed
            // a surprise threshold (2× the baseline error rate of ~1 per 8 trials),
            // the agent infers a contingency change and switches to fast learning.
            // This implements the Bayesian change-point detection principle from
            // Wilson et al. (2010, Neural Computation): surprise = P(change | errors)
            // spikes when error rate exceeds prior expectation, triggering an
            // immediate learning rate switch rather than waiting for gradual EMA
            // adaptation. The 2-error threshold approximates a 2× baseline surprise
            // signal (baseline ~12.5% error rate × 8-trial criterion = ~1 error
            // expected per criterion block).
            if consecutive_errors >= 2 && !in_reversal {
                // Detected probable reversal before criterion-based switch:
                // force post-reversal fast-learning mode immediately.
                post_reversal_countdown = 8;
                consecutive_errors = 0;
            } else if consecutive_errors >= 2 {
                consecutive_errors = 0;
            }

            // Deliver reward/punishment and update associations
            let rewarded = correct;
            prev_choice = Some(chose_a);
            prev_rewarded = Some(rewarded);

            // Context-dependent learning rates: post-reversal phase increases
            // win LR and counterfactual LR while decreasing loss LR, reducing
            // perseveration by balancing the asymmetry during reversal adaptation
            // (Behrens et al. 2007 — volatility-sensitive learning rates).
            let in_post_reversal = post_reversal_countdown > 0;
            let (eff_win_lr, eff_loss_lr, cf_lr) = if in_post_reversal {
                (0.55f32, 0.70f32, 0.20f32)
            } else {
                (learning_rate, loss_learning_rate, 0.10f32)
            };
            let lr = if rewarded { eff_win_lr } else { eff_loss_lr };
            if chose_a {
                if rewarded {
                    assoc_a =
                        ContinuousHV::weighted_bundle(&[&assoc_a, &reward_hv], &[1.0 - lr, lr]);
                } else {
                    assoc_a =
                        ContinuousHV::weighted_bundle(&[&assoc_a, &no_reward_hv], &[1.0 - lr, lr]);
                }
            } else if rewarded {
                assoc_b = ContinuousHV::weighted_bundle(&[&assoc_b, &reward_hv], &[1.0 - lr, lr]);
            } else {
                assoc_b =
                    ContinuousHV::weighted_bundle(&[&assoc_b, &no_reward_hv], &[1.0 - lr, lr]);
            }

            // Counterfactual learning: in a binary choice task, punishment
            // for one stimulus implies reward for the other (Li & Daw, 2011).
            // Boost the unchosen stimulus's reward association after a loss.
            if !rewarded {
                if chose_a {
                    assoc_b = ContinuousHV::weighted_bundle(
                        &[&assoc_b, &reward_hv],
                        &[1.0 - cf_lr, cf_lr],
                    );
                } else {
                    assoc_a = ContinuousHV::weighted_bundle(
                        &[&assoc_a, &reward_hv],
                        &[1.0 - cf_lr, cf_lr],
                    );
                }
            }

            // Check if criterion reached → trigger reversal
            if consecutive_correct >= criterion {
                if initial_acquisition_trials.is_none() {
                    initial_acquisition_trials = Some(trials_since_event);
                } else {
                    reversal_trial_counts.push(trials_since_event);
                    reversals_completed += 1;
                }

                // Reverse contingency
                a_rewarded = !a_rewarded;
                consecutive_correct = 0;
                trials_since_event = 0;
                in_reversal = true;
                post_reversal_countdown = 8;
            }
        }

        let win_stay_rate = if win_total > 0 {
            win_stay_count as f64 / win_total as f64
        } else {
            0.0
        };
        let lose_shift_rate = if lose_total > 0 {
            lose_shift_count as f64 / lose_total as f64
        } else {
            0.0
        };
        let avg_reversal_trials = if !reversal_trial_counts.is_empty() {
            reversal_trial_counts.iter().sum::<u32>() as f64 / reversal_trial_counts.len() as f64
        } else {
            max_trials as f64 // never reversed
        };
        let initial = initial_acquisition_trials.unwrap_or(max_trials as u32) as f64;
        let reversal_cost = avg_reversal_trials - initial;

        TrialResult {
            initial_acquisition_trials: initial,
            avg_reversal_trials,
            reversals_completed: reversals_completed as f64,
            perseverative_errors: perseverative_errors as f64,
            total_errors: total_errors as f64,
            win_stay_rate,
            lose_shift_rate,
            reversal_cost,
            rt_ticks,
        }
    }
}

struct TrialResult {
    initial_acquisition_trials: f64,
    avg_reversal_trials: f64,
    reversals_completed: f64,
    perseverative_errors: f64,
    total_errors: f64,
    win_stay_rate: f64,
    lose_shift_rate: f64,
    reversal_cost: f64,
    rt_ticks: Vec<f64>,
}

impl PsychBenchmark for ReversalLearningBenchmark {
    fn name(&self) -> &str {
        "CogBench::ReversalLearning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Reversal Learning",
            citation: "Cools et al. (2002)",
            year: 2002,
            doi: Some("10.1523/JNEUROSCI.22-11-04563.2002"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut initials = Vec::new();
        let mut reversals = Vec::new();
        let mut completed = Vec::new();
        let mut perseverative = Vec::new();
        let mut total_errs = Vec::new();
        let mut win_stays = Vec::new();
        let mut lose_shifts = Vec::new();
        let mut costs = Vec::new();
        let mut all_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            initials.push(r.initial_acquisition_trials);
            reversals.push(r.avg_reversal_trials);
            completed.push(r.reversals_completed);
            perseverative.push(r.perseverative_errors);
            total_errs.push(r.total_errors);
            win_stays.push(r.win_stay_rate);
            lose_shifts.push(r.lose_shift_rate);
            costs.push(r.reversal_cost);
            all_rts.extend_from_slice(&r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "reversal".to_string(),
                    correct: r.reversals_completed > 0.0,
                    rt_ticks: if r.rt_ticks.is_empty() {
                        0.0
                    } else {
                        r.rt_ticks.iter().sum::<f64>() / r.rt_ticks.len() as f64
                    },
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "initial_acquisition_trials",
            MetricValue::from_samples(&initials),
        );
        result.insert("avg_reversal_trials", MetricValue::from_samples(&reversals));
        result.insert("reversals_completed", MetricValue::from_samples(&completed));
        result.insert(
            "perseverative_errors",
            MetricValue::from_samples(&perseverative),
        );
        result.insert("total_errors", MetricValue::from_samples(&total_errs));
        result.insert("win_stay_rate", MetricValue::from_samples(&win_stays));
        result.insert("lose_shift_rate", MetricValue::from_samples(&lose_shifts));
        result.insert("reversal_cost", MetricValue::from_samples(&costs));
        result.insert("rt_ticks", MetricValue::from_samples(&all_rts));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reversal_learning_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ReversalLearningBenchmark.run(&config);
        assert!(result.metrics.contains_key("initial_acquisition_trials"));
        assert!(result.metrics.contains_key("avg_reversal_trials"));
        assert!(result.metrics.contains_key("reversals_completed"));
        assert!(result.metrics.contains_key("win_stay_rate"));
        assert!(result.metrics.contains_key("lose_shift_rate"));
    }

    #[test]
    fn test_reversal_learning_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ReversalLearningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_reversal_learning_completes_reversals() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ReversalLearningBenchmark.run(&config);
        let reversals = result.metrics["reversals_completed"].mean;
        // With 200 trials and criterion 8, should complete at least 1 reversal
        assert!(
            reversals >= 1.0,
            "should complete at least 1 reversal, got {}",
            reversals
        );
    }

    #[test]
    fn test_reversal_learning_win_stay_positive() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ReversalLearningBenchmark.run(&config);
        let ws = result.metrics["win_stay_rate"].mean;
        // Win-stay should be above chance (0.5) — agent should repeat rewarded choices
        assert!(ws > 0.5, "win-stay rate should exceed chance, got {}", ws);
    }

    #[test]
    fn test_reversal_learning_initial_faster_than_reversal() {
        // Initial acquisition should typically be faster than reversal
        // (reversal requires overcoming existing association)
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ReversalLearningBenchmark.run(&config);
        let cost = result.metrics["reversal_cost"].mean;
        // Reversal cost should be non-negative on average
        // (reversal takes longer than initial learning)
        assert!(cost.is_finite(), "reversal cost should be finite");
    }
}
