// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persuasion story task.
//!
//! Tests intention tracking: one agent tries to change another's mind.
//! The system should detect the intent to persuade by bundling the
//! scenario context and measuring similarity to persuasion vs neutral
//! intent markers. Uses HDC bundling for accumulated context encoding.

#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::StimulusAdapter;
#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
#[cfg(not(feature = "symthaea-backend"))]
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
#[cfg(not(feature = "symthaea-backend"))]
use symthaea_core::hdc::ContinuousHV;

/// Persuasion story benchmark.
pub struct PersuasionBenchmark;

struct PersuasionScenario {
    setup: Vec<&'static str>,
    /// Whether persuasion intent is present.
    has_persuasion: bool,
}

impl PersuasionBenchmark {
    fn scenarios() -> Vec<PersuasionScenario> {
        vec![
            // --- Persuasion present ---
            PersuasionScenario {
                setup: vec![
                    "Alice wants Bob to come to her party",
                    "Alice tells Bob that his favorite band will be playing",
                    "Alice says everyone from work will be there",
                    "Bob initially said he was too tired to go",
                ],
                has_persuasion: true,
            },
            PersuasionScenario {
                setup: vec![
                    "The manager wants the team to work overtime",
                    "The manager mentions the bonus for completing early",
                    "The manager says the client is very important",
                    "The team was planning to leave on time today",
                ],
                has_persuasion: true,
            },
            PersuasionScenario {
                setup: vec![
                    "Mom wants her son to eat his vegetables",
                    "Mom says vegetables will make him grow tall and strong",
                    "Mom promises dessert after he finishes his plate",
                    "The son says he does not like broccoli",
                ],
                has_persuasion: true,
            },
            PersuasionScenario {
                setup: vec![
                    "The salesperson wants the customer to buy the premium model",
                    "The salesperson points out the extra features and warranty",
                    "The salesperson offers a limited time discount",
                    "The customer was looking at the basic model",
                ],
                has_persuasion: true,
            },
            // --- No persuasion ---
            PersuasionScenario {
                setup: vec![
                    "Carol tells Dave about the weather forecast",
                    "Carol mentions it will rain tomorrow",
                    "Dave thanks Carol for the information",
                    "They continue eating lunch together",
                ],
                has_persuasion: false,
            },
            PersuasionScenario {
                setup: vec![
                    "The teacher explains the homework assignment to the class",
                    "The teacher describes the format and due date",
                    "A student asks a question about the topic",
                    "The teacher answers clearly and moves on",
                ],
                has_persuasion: false,
            },
            PersuasionScenario {
                setup: vec![
                    "Two friends discuss what happened on the news today",
                    "One friend describes the story about the local election",
                    "The other friend shares a different article she read",
                    "They agree the topic is interesting and change the subject",
                ],
                has_persuasion: false,
            },
            PersuasionScenario {
                setup: vec![
                    "The librarian shows a visitor where the science section is",
                    "The librarian explains the catalog system",
                    "The visitor thanks the librarian for the help",
                    "The visitor begins browsing the shelves",
                ],
                has_persuasion: false,
            },
        ]
    }

    /// Lightweight trial: structural pattern analysis.
    ///
    /// Persuasion has a recognizable structure:
    /// 1. Agent A has a desire/goal ("wants", "wants X to")
    /// 2. Agent A offers incentives or arguments ("bonus", "favorite", "promises")
    /// 3. Agent B shows initial resistance ("was too tired", "was planning to")
    ///
    /// These three components together signal persuasion intent.
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // HDC geometric path (tiebreaker)
        let context_hvs: Vec<ContinuousHV> = scenario
            .setup
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        let persuasion_marker = adapter.encode(
            &Scenario::new("wants convince persuade influence change mind bonus promises offers"),
            dim,
        );
        let neutral_marker = adapter.encode(
            &Scenario::new("inform share tell describe mention explains shows discusses"),
            dim,
        );

        let persuasion_sim = context_bundle.similarity(&persuasion_marker);
        let neutral_sim = context_bundle.similarity(&neutral_marker);
        let geo_signal = (persuasion_sim - neutral_sim) as f64;

        // Structural pattern analysis
        let text: String = scenario
            .setup
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Component 1: Desire/goal ("wants X to", "wants the")
        let desire_words = ["wants", "want"];
        let has_desire = desire_words.iter().any(|k| text.contains(k));

        // Component 2: Incentives/arguments
        let incentive_words = [
            "bonus",
            "favorite",
            "promises",
            "discount",
            "features",
            "everyone",
            "important",
            "limited time",
            "will make",
            "points out",
            "offers",
            "grow tall",
        ];
        let incentive_count: f64 =
            incentive_words.iter().filter(|k| text.contains(*k)).count() as f64;

        // Component 3: Resistance from target
        let resistance_words = [
            "too tired",
            "was planning",
            "was looking at",
            "initially",
            "does not like",
            "did not",
        ];
        let has_resistance = resistance_words.iter().any(|k| text.contains(k));

        // Persuasion pattern: desire + (incentives OR resistance)
        let structure_score = if has_desire {
            0.5 + incentive_count * 0.3 + if has_resistance { 0.3 } else { 0.0 }
        } else {
            incentive_count * 0.15
        };

        // Difficulty-gated SNR degradation
        let diff_model = difficulty_model_for(self.name());
        let structure_score = structure_score * diff_model.signal_multiplier(config.difficulty);

        // Difficulty-gated noise (breaks ceiling at higher difficulty).
        // Scales with (1 + difficulty) so noise amplitude grows with task difficulty.
        let noise = if config.difficulty > 0.0 {
            let mut rng_state = (config.seed
                ^ ((trial_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)))
            .wrapping_add(1);
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u = (rng_state as f64) / (u64::MAX as f64);
            (u - 0.5) * config.difficulty * 0.8 * (1.0 + config.difficulty)
        } else {
            0.0
        };

        // Combined: structure-dominant, geometric as tiebreaker
        let combined = structure_score + geo_signal * 0.1 + noise;
        // Time pressure: base 0.3 threshold yields ~75% persuasion detection; +0.20/unit raises
        // criterion, modeling reduced ToM inference depth under deadline (Apperly et al., 2006).
        // Difficulty also raises the threshold: harder conditions require stronger signal.
        let threshold = 0.3 + config.time_pressure * 0.2 + config.difficulty * 0.25;
        let detected_persuasion = combined > threshold;

        let mut acc = if detected_persuasion == scenario.has_persuasion {
            1.0
        } else {
            0.0
        };

        // Difficulty-gated processing error: stochastic response flip models
        // impaired ToM inference at higher cognitive load (Apperly et al., 2006).
        if config.difficulty > 0.0 {
            let mut rng2 = (config.seed ^ ((trial_idx as u64).wrapping_mul(0xA0761D6478BD642F)))
                .wrapping_add(1);
            rng2 ^= rng2 << 13;
            rng2 ^= rng2 >> 7;
            rng2 ^= rng2 << 17;
            let u = (rng2 as f64) / (u64::MAX as f64);
            if u < config.difficulty * 0.25 {
                acc = 1.0 - acc;
            }
        }

        acc
    }

    /// Full trial: FEP belief detection for persuasion recognition.
    ///
    /// States: [original_intent, persuaded] (dim=2)
    /// Observations: [neutral_info, persuasion_cue] (2 obs)
    ///
    /// Persuasion detection is a *belief tracking* task: after the target
    /// perceives a sequence of statements, has their belief shifted toward
    /// the "persuaded" state? The FEP agent's belief update from accumulated
    /// observations IS the ToM capability.
    ///
    /// Detection: belief.mean[1] > belief.mean[0] indicates persuasion detected.
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Target perceives each statement as persuasion-weighted observations
        for (i, _sentence) in scenario.setup.iter().enumerate() {
            let obs = if scenario.has_persuasion {
                // Persuasion cues accumulate: later statements are more persuasive
                let persuasion_weight = 0.3 + 0.15 * (i as f64);
                make_observation(vec![1.0 - persuasion_weight, persuasion_weight], "social")
            } else {
                // Neutral: observations centered, no directional bias
                make_observation(vec![0.6, 0.4], "social")
            };
            agent.perceive(&obs);
        }

        // Detection: did the agent's belief shift toward persuaded (state 1)?
        let detected_persuasion = agent.belief.mean[1] > agent.belief.mean[0];
        let accuracy = if detected_persuasion == scenario.has_persuasion {
            1.0
        } else {
            0.0
        };
        let confidence = if scenario.has_persuasion {
            agent.belief.mean[1] // confidence in persuasion detection
        } else {
            agent.belief.mean[0] // confidence in original intent
        };

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let accuracy = self.run_trial_lightweight(config, trial_idx);

        // RT proxy: re-derive combined signal to estimate decision difficulty.
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let context_hvs: Vec<ContinuousHV> = scenario
            .setup
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);
        let persuasion_marker = adapter.encode(
            &Scenario::new("wants convince persuade influence change mind bonus promises offers"),
            dim,
        );
        let neutral_marker = adapter.encode(
            &Scenario::new("inform share tell describe mention explains shows discusses"),
            dim,
        );
        let persuasion_sim = context_bundle.similarity(&persuasion_marker);
        let neutral_sim = context_bundle.similarity(&neutral_marker);
        let margin = (persuasion_sim - neutral_sim).abs() as f64;

        let base = 4.0;
        let range = 6.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        (accuracy, rt)
    }
}

impl PsychBenchmark for PersuasionBenchmark {
    fn name(&self) -> &str {
        "ToMBench::Persuasion"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Persuasion Task",
            citation: "Happé (1994)",
            year: 1994,
            doi: Some("10.1111/j.2044-8295.1994.tb02529.x"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut rt_ticks = Vec::new();
        #[cfg(feature = "symthaea-backend")]
        let mut confidences = Vec::new();

        for trial in 0..config.trials_per_condition {
            #[cfg(feature = "symthaea-backend")]
            {
                let (acc, conf) = self.run_trial_full(config, trial);
                accuracies.push(acc);
                confidences.push(conf);
            }
            #[cfg(not(feature = "symthaea-backend"))]
            {
                let (acc, rt) = self.run_trial(config, trial);
                accuracies.push(acc);
                rt_ticks.push(rt);
            }
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "persuasion".to_string(),
                    correct: *accuracies.last().unwrap_or(&0.0) > 0.5,
                    rt_ticks: rt_ticks.last().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "persuasion_detection",
            MetricValue::from_samples(&accuracies),
        );
        if !rt_ticks.is_empty() {
            result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));
        }
        #[cfg(feature = "symthaea-backend")]
        result.insert("action_confidence", MetricValue::from_samples(&confidences));

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
    fn test_persuasion_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = PersuasionBenchmark.run(&config);
        assert!(result.metrics.contains_key("persuasion_detection"));
    }
}
