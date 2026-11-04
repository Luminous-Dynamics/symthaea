// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Matched-seed experiment over environmental stability and teaching form.
//!
//! This is a synthetic mechanism experiment, not a reanalysis of the PNAS
//! participant data. It uses the published six-generation/12-trial shape and
//! reports simulation uncertainty separately from human-population effects.

use civbench_social_learning_core::environmental_adaptation::{
    simulate, Config, EncodingPolicy, EnvironmentRegime, MessageKind,
};

const REPLICATES: u64 = 256;

#[derive(Default)]
struct OnlineStats {
    n: u64,
    sum: f64,
    sum_squares: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let centered = (self.sum_squares - self.sum * self.sum / n).max(0.0);
        let standard_deviation = (centered / (n - 1.0)).sqrt();
        Some(1.96 * standard_deviation / n.sqrt())
    }
}

#[derive(Default)]
struct GenerationBucket {
    target_hit_rate: OnlineStats,
    optimal_action_rate: OnlineStats,
    payload_bits: OnlineStats,
    procedure_share: OnlineStats,
    outcome_goal_share: OnlineStats,
    causal_model_share: OnlineStats,
    first_optimal_choice: OnlineStats,
    causal_conflicts: OnlineStats,
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn regimes() -> [EnvironmentRegime; 3] {
    [
        EnvironmentRegime::Stable,
        EnvironmentRegime::ReverseEveryGeneration,
        EnvironmentRegime::RandomEveryGeneration,
    ]
}

fn regime_name(regime: EnvironmentRegime) -> &'static str {
    match regime {
        EnvironmentRegime::Stable => "stable",
        EnvironmentRegime::ReverseEveryGeneration => "reverse_each_generation",
        EnvironmentRegime::RandomEveryGeneration => "random_each_generation",
    }
}

fn policies() -> [EncodingPolicy; 4] {
    [
        EncodingPolicy::NoTransmission,
        EncodingPolicy::AlwaysProcedure,
        EncodingPolicy::AlwaysOutcomeGoal,
        EncodingPolicy::AlwaysCausalModel,
        EncodingPolicy::Adaptive,
    ]
}

fn policy_name(policy: EncodingPolicy) -> &'static str {
    match policy {
        EncodingPolicy::NoTransmission => "individual_only_no_message",
        EncodingPolicy::AlwaysProcedure => "procedure_only",
        EncodingPolicy::AlwaysOutcomeGoal => "outcome_goal_only",
        EncodingPolicy::AlwaysCausalModel => "causal_model",
        EncodingPolicy::Adaptive => "volatility_adaptive",
    }
}

fn main() {
    let config = Config::default();
    let mut all_rows: Vec<(EnvironmentRegime, EncodingPolicy, usize, GenerationBucket)> =
        Vec::with_capacity(3 * 4 * config.generations);

    for regime in regimes() {
        for policy in policies() {
            let mut buckets: Vec<GenerationBucket> =
                (0..config.generations).map(|_| GenerationBucket::default()).collect();

            for seed in 0..REPLICATES {
                // Common seed set is used for every policy/regime combination.
                let run_seed = 0xE17A_2026_u64.wrapping_add(seed);
                let run = simulate(regime, policy, config, run_seed)
                    .expect("the frozen reference configuration must validate");

                for metrics in &run.metrics {
                    let bucket = &mut buckets[metrics.generation];
                    bucket.target_hit_rate.push(metrics.target_hit_rate);
                    bucket.optimal_action_rate.push(metrics.optimal_action_rate);
                    bucket.payload_bits.push(metrics.outgoing_payload_bits as f64);
                    bucket.procedure_share.push(
                        if metrics.outgoing_message == MessageKind::Procedure { 1.0 } else { 0.0 }
                    );
                    bucket.outcome_goal_share.push(
                        if metrics.outgoing_message == MessageKind::OutcomeGoal { 1.0 } else { 0.0 }
                    );
                    bucket.causal_model_share.push(
                        if metrics.outgoing_message == MessageKind::CausalModel { 1.0 } else { 0.0 }
                    );
                    if let Some(trial) = metrics.first_optimal_choice_trial {
                        bucket.first_optimal_choice.push(trial as f64);
                    }
                    bucket.causal_conflicts.push(metrics.causal_conflicts as f64);
                }
            }

            for (generation, bucket) in buckets.into_iter().enumerate() {
                all_rows.push((regime, policy, generation, bucket));
            }
        }
    }

    println!(
        "regime,policy,generation,replicates,target_hit_rate_mean,target_hit_rate_ci95,optimal_action_rate_mean,optimal_action_rate_ci95,mean_message_payload_bits,procedure_message_share,outcome_goal_message_share,causal_model_message_share,mean_first_optimal_choice_trial,mean_causal_conflicts"
    );
    for (regime, policy, generation, bucket) in all_rows {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            regime_name(regime),
            policy_name(policy),
            generation,
            REPLICATES,
            fmt(bucket.target_hit_rate.mean()),
            fmt(bucket.target_hit_rate.ci95_half_width()),
            fmt(bucket.optimal_action_rate.mean()),
            fmt(bucket.optimal_action_rate.ci95_half_width()),
            fmt(bucket.payload_bits.mean()),
            fmt(bucket.procedure_share.mean()),
            fmt(bucket.outcome_goal_share.mean()),
            fmt(bucket.causal_model_share.mean()),
            fmt(bucket.first_optimal_choice.mean()),
            fmt(bucket.causal_conflicts.mean()),
        );
    }
}
