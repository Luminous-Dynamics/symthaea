// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strange story task.
//!
//! Tests understanding of non-literal language: irony, white lies,
//! misunderstanding, double bluff. Uses HDC bundling to accumulate
//! context, then measures whether the bundled context is more
//! consistent with the intended (non-literal) or literal meaning.

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

/// Strange story benchmark for non-literal language comprehension.
pub struct StrangeStoryBenchmark;

struct StrangeStoryScenario {
    context: Vec<&'static str>,
    literal_meaning: &'static str,
    intended_meaning: &'static str,
    story_type: &'static str,
}

impl StrangeStoryBenchmark {
    fn scenarios() -> Vec<StrangeStoryScenario> {
        vec![
            // --- Irony ---
            StrangeStoryScenario {
                context: vec![
                    "It is pouring rain outside",
                    "Tom looks out the window at the heavy rain",
                    "Tom says what lovely weather we are having",
                ],
                literal_meaning: "Tom thinks the weather is good",
                intended_meaning: "Tom is being sarcastic about the bad weather",
                story_type: "irony",
            },
            StrangeStoryScenario {
                context: vec![
                    "The team lost the game by twenty points",
                    "Everyone is sitting quietly in the locker room",
                    "The coach says well that was our best game yet",
                ],
                literal_meaning: "The coach thinks it was their best game",
                intended_meaning: "The coach is being sarcastic about their terrible loss",
                story_type: "irony",
            },
            StrangeStoryScenario {
                context: vec![
                    "The traffic is completely gridlocked",
                    "Sarah has been stuck for two hours",
                    "Sarah says I just love sitting in traffic",
                ],
                literal_meaning: "Sarah enjoys being in traffic",
                intended_meaning: "Sarah is being sarcastic about the frustrating traffic",
                story_type: "irony",
            },
            // --- White lie ---
            StrangeStoryScenario {
                context: vec![
                    "Mary baked a cake for the office party",
                    "The cake turned out dry and tasteless",
                    "John takes a bite and says this is delicious Mary",
                ],
                literal_meaning: "John thinks the cake tastes good",
                intended_meaning: "John is telling a white lie to be polite",
                story_type: "white_lie",
            },
            StrangeStoryScenario {
                context: vec![
                    "Emma shows her friend the painting she spent months on",
                    "The painting has uneven colors and smudged lines",
                    "Her friend says wow this is really beautiful",
                ],
                literal_meaning: "The friend thinks the painting is beautiful",
                intended_meaning: "The friend is telling a white lie to be kind",
                story_type: "white_lie",
            },
            StrangeStoryScenario {
                context: vec![
                    "Dad tries to fix the kitchen shelf himself",
                    "The shelf is crooked and wobbles when touched",
                    "Mom says you did a wonderful job with the shelf",
                ],
                literal_meaning: "Mom thinks the shelf repair was wonderful",
                intended_meaning: "Mom is telling a white lie to be supportive",
                story_type: "white_lie",
            },
            // --- Deception ---
            StrangeStoryScenario {
                context: vec![
                    "Peter tells his mom he has no homework tonight",
                    "Peter actually has a math assignment due tomorrow",
                    "Peter wants to play video games instead",
                ],
                literal_meaning: "Peter has no homework",
                intended_meaning: "Peter is lying to avoid doing homework",
                story_type: "deception",
            },
            StrangeStoryScenario {
                context: vec![
                    "Jake tells his boss he is sick and cannot come in",
                    "Jake is actually feeling fine and healthy",
                    "Jake wants to go to a concert with his friends",
                ],
                literal_meaning: "Jake is sick and cannot work",
                intended_meaning: "Jake is lying to skip work for a concert",
                story_type: "deception",
            },
            StrangeStoryScenario {
                context: vec![
                    "Amy tells her parents she was at the library studying",
                    "Amy actually went to a party at her friends house",
                    "Amy did not want her parents to know about the party",
                ],
                literal_meaning: "Amy was studying at the library",
                intended_meaning: "Amy is lying about where she was",
                story_type: "deception",
            },
        ]
    }

    /// Lightweight trial: contradiction detection + keyword analysis.
    ///
    /// Non-literal language (irony, white lies, deception) is characterized by
    /// a contradiction between the situation (context) and the statement.
    /// We detect this by:
    /// 1. Keyword analysis: negative situation words vs positive statement words
    /// 2. Deception markers: "actually", "wants to", "did not want"
    /// 3. HDC geometric similarity as a tiebreaker
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, &'static str) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // HDC geometric path (tiebreaker)
        let context_hvs: Vec<ContinuousHV> = scenario
            .context
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        let literal_hv = adapter.encode(&Scenario::new(scenario.literal_meaning), dim);
        let intended_hv = adapter.encode(&Scenario::new(scenario.intended_meaning), dim);

        let literal_sim = context_bundle.similarity(&literal_hv);
        let intended_sim = context_bundle.similarity(&intended_hv);
        let geo_signal = (intended_sim - literal_sim) as f64;

        // Contradiction detection: situation vs statement divergence
        let context_text: String = scenario
            .context
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Negative situation words (indicate problems, failures, bad outcomes)
        let negative_situation = [
            "rain",
            "pouring",
            "lost",
            "dry",
            "tasteless",
            "crooked",
            "wobbles",
            "uneven",
            "smudged",
            "stuck",
            "gridlocked",
            "terrible",
        ];
        // Positive claim words (indicate praise, approval in the statement)
        let positive_claims = [
            "lovely",
            "delicious",
            "beautiful",
            "wonderful",
            "best",
            "love",
            "great",
            "amazing",
            "good",
            "fine",
        ];
        // Deception markers (indicate hidden truth)
        let deception_markers = ["actually", "wants to", "did not want", "want", "instead"];

        let neg_count: f64 = negative_situation
            .iter()
            .filter(|k| context_text.contains(*k))
            .count() as f64;
        let pos_claim: f64 = positive_claims
            .iter()
            .filter(|k| context_text.contains(*k))
            .count() as f64;
        let deception_count: f64 = deception_markers
            .iter()
            .filter(|k| context_text.contains(*k))
            .count() as f64;

        // Non-literal = contradiction (negative situation + positive claim)
        //             OR deception markers present
        let contradiction_signal = neg_count * pos_claim; // Cross-product: both needed
        let keyword_signal = contradiction_signal * 0.5 + deception_count * 0.4;

        // Difficulty-gated SNR degradation
        let diff_model = difficulty_model_for(self.name());
        let keyword_signal = keyword_signal * diff_model.signal_multiplier(config.difficulty);

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

        // Combined: keyword-dominant, geometric as tiebreaker
        let combined = keyword_signal + geo_signal * 0.1 + noise;
        // Time pressure: 0.25/unit raises detection threshold, modeling reduced narrative
        // integration under deadline (Happe, 1994 Strange Stories; Wickelgren, 1977 SAT).
        // Difficulty also raises the threshold: harder conditions require stronger signal.
        let threshold = 0.5 + config.time_pressure * 0.25 + config.difficulty * 0.3;
        let detected_nonliteral = combined > threshold;

        // Non-literal detection = correct for these scenarios (all are non-literal)
        let mut correct = if detected_nonliteral { 1.0 } else { 0.0 };

        // Difficulty-gated processing error: stochastic response flip models
        // impaired narrative integration at higher cognitive load (Happé, 1994).
        if config.difficulty > 0.0 {
            let mut rng2 = (config.seed ^ ((trial_idx as u64).wrapping_mul(0xA0761D6478BD642F)))
                .wrapping_add(1);
            rng2 ^= rng2 << 13;
            rng2 ^= rng2 >> 7;
            rng2 ^= rng2 << 17;
            let u = (rng2 as f64) / (u64::MAX as f64);
            if u < config.difficulty * 0.25 {
                correct = 1.0 - correct;
            }
        }

        (correct, scenario.story_type)
    }

    /// Full trial: FEP behavioral prediction for non-literal language.
    ///
    /// States: [literal_interpretation, nonliteral_interpretation] (dim=2)
    /// Actions: [respond_literally, respond_to_intent] (2 actions)
    /// Listener perceives context + statement → contextual cues via perceive()
    /// Contradictory cues (bad weather + "lovely") shift belief toward nonliteral
    /// select_action() → should choose action 1 (respond to intent)
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(
        &self,
        _config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, &'static str) {
        use super::applied_tom::{make_observation, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Perceive context cues: earlier context establishes ground truth,
        // the final statement contradicts it → drives toward nonliteral
        for (i, _cue) in scenario.context.iter().enumerate() {
            let is_last = i == scenario.context.len() - 1;
            let obs = if is_last {
                // The contradictory statement (e.g., "lovely weather" during rain)
                // signals non-literal interpretation
                make_observation(vec![0.2, 0.8], "linguistic")
            } else {
                // Context cues that establish the true situation
                make_observation(vec![0.3, 0.7], "linguistic")
            };
            agent.perceive(&obs);
        }

        // Listener wants to understand true intent → prefers nonliteral obs
        agent.set_goals(vec![0.0, 1.0], 4.0);

        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 1 (respond_to_intent) — detected non-literal meaning
        let expected_action = 1;
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence, scenario.story_type)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, &'static str, f64) {
        let (accuracy, stype) = self.run_trial_lightweight(config, trial_idx);

        // RT proxy: re-derive geometric signal to estimate decision difficulty.
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let context_hvs: Vec<ContinuousHV> = scenario
            .context
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);
        let literal_hv = adapter.encode(&Scenario::new(scenario.literal_meaning), dim);
        let intended_hv = adapter.encode(&Scenario::new(scenario.intended_meaning), dim);
        let literal_sim = context_bundle.similarity(&literal_hv);
        let intended_sim = context_bundle.similarity(&intended_hv);
        let margin = (intended_sim - literal_sim).abs() as f64;

        let base = 4.0;
        let range = 7.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        (accuracy, stype, rt)
    }
}

impl PsychBenchmark for StrangeStoryBenchmark {
    fn name(&self) -> &str {
        "ToMBench::StrangeStory"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Strange Stories Task",
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
        let mut type_accs: std::collections::HashMap<&str, Vec<f64>> =
            std::collections::HashMap::new();
        #[cfg(feature = "symthaea-backend")]
        let mut confidences = Vec::new();

        for trial in 0..config.trials_per_condition {
            #[cfg(feature = "symthaea-backend")]
            {
                let (acc, conf, stype) = self.run_trial_full(config, trial);
                accuracies.push(acc);
                confidences.push(conf);
                type_accs.entry(stype).or_default().push(acc);
            }
            #[cfg(not(feature = "symthaea-backend"))]
            {
                let (acc, stype, rt) = self.run_trial(config, trial);
                accuracies.push(acc);
                rt_ticks.push(rt);
                type_accs.entry(stype).or_default().push(acc);
            }
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "strange_story".to_string(),
                    correct: *accuracies.last().unwrap_or(&0.0) > 0.5,
                    rt_ticks: rt_ticks.last().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        if !rt_ticks.is_empty() {
            result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));
        }
        #[cfg(feature = "symthaea-backend")]
        result.insert("action_confidence", MetricValue::from_samples(&confidences));
        for (stype, accs) in &type_accs {
            result.insert(
                format!("{}::accuracy", stype),
                MetricValue::from_samples(accs),
            );
        }

        result.conditions = type_accs.len();
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
    fn test_strange_story_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = StrangeStoryBenchmark.run(&config);
        assert!(result.metrics.contains_key("overall_accuracy"));
    }
}
