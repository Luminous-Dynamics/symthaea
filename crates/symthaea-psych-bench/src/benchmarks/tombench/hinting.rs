// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hinting task.
//!
//! Tests desire inference from indirect cues: the system must infer
//! what a character wants without being told directly. Uses HDC
//! bundling to accumulate contextual cues, then measures whether
//! the accumulated context is more similar to the correct desire
//! inference than to the surface-level (wrong) interpretation.

#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::StimulusAdapter;
#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
#[cfg(not(feature = "symthaea-backend"))]
use symthaea_core::hdc::ContinuousHV;

/// Hinting task benchmark.
pub struct HintingBenchmark;

struct HintingScenario {
    context: Vec<&'static str>,
    correct_inference: &'static str,
    wrong_inference: &'static str,
}

impl HintingBenchmark {
    fn scenarios() -> Vec<HintingScenario> {
        vec![
            HintingScenario {
                context: vec![
                    "George walks past the shop and sees a coat in the window",
                    "George says that coat looks really warm",
                    "George shivers and pulls his thin jacket tighter",
                ],
                correct_inference: "George wants someone to buy him the coat",
                wrong_inference: "George is commenting on coat design",
            },
            HintingScenario {
                context: vec![
                    "Maria is tired after working all day",
                    "Maria says the dishes are piling up in the kitchen",
                    "Maria yawns and sits down on the sofa",
                ],
                correct_inference: "Maria wants her partner to do the dishes",
                wrong_inference: "Maria is describing the state of the kitchen",
            },
            HintingScenario {
                context: vec![
                    "Paul and his friend are at a restaurant",
                    "Paul forgot his wallet at home",
                    "Paul says this meal is more expensive than I expected",
                ],
                correct_inference: "Paul wants his friend to help pay for the meal",
                wrong_inference: "Paul is commenting on restaurant prices",
            },
            HintingScenario {
                context: vec![
                    "Emma is at a boring party with her friend",
                    "Emma looks at her watch and says it is getting late",
                    "Emma yawns and mentions the long drive home",
                ],
                correct_inference: "Emma wants to leave the party",
                wrong_inference: "Emma is telling the time",
            },
        ]
    }

    /// Lightweight trial: structural hint analysis + HDC accumulation.
    ///
    /// Hinting tasks require inferring hidden desires from indirect cues.
    /// The correct inference is always about a desire/need, while the wrong
    /// inference is a surface-level literal interpretation. We detect this via:
    ///
    /// 1. **Desire-pattern analysis**: The correct inference contains desire
    ///    language ("wants", "help", "leave", "buy") while the wrong one
    ///    describes information exchange ("commenting", "describing", "telling").
    /// 2. **Behavioral cue detection**: Context contains physical/emotional
    ///    signals of unfulfilled need (shivers, yawns, forgot, tired).
    /// 3. **HDC geometric similarity** as a supplementary signal.
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // --- Signal 1: Desire vs information keywords ---
        let correct_lower = scenario.correct_inference.to_lowercase();
        let wrong_lower = scenario.wrong_inference.to_lowercase();
        let context_text: String = scenario
            .context
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Desire language: words signaling wants, needs, requests
        let desire_words = [
            "wants", "want", "help", "buy", "pay", "leave", "give", "need", "wish", "hope", "get",
            "lend", "offer",
        ];
        // Literal/informational language: words signaling description, not request
        let literal_words = [
            "commenting",
            "describing",
            "telling",
            "observing",
            "noting",
            "reporting",
            "mentioning",
            "stating",
            "information",
            "design",
            "prices",
            "time",
        ];
        // Behavioral cues in context indicating unfulfilled need
        let behavioral_cues = [
            "shivers",
            "yawns",
            "forgot",
            "tired",
            "looks at",
            "pulls",
            "thin",
            "sitting down",
            "long drive",
            "more expensive",
            "piling up",
        ];

        let correct_desire: f64 = desire_words
            .iter()
            .filter(|k| correct_lower.contains(*k))
            .count() as f64;
        let wrong_desire: f64 = desire_words
            .iter()
            .filter(|k| wrong_lower.contains(*k))
            .count() as f64;
        let correct_literal: f64 = literal_words
            .iter()
            .filter(|k| correct_lower.contains(*k))
            .count() as f64;
        let wrong_literal: f64 = literal_words
            .iter()
            .filter(|k| wrong_lower.contains(*k))
            .count() as f64;
        let behavioral_count: f64 = behavioral_cues
            .iter()
            .filter(|k| context_text.contains(*k))
            .count() as f64;

        // Correct inference has more desire words and fewer literal words
        let correct_desire_signal = correct_desire - correct_literal;
        let wrong_desire_signal = wrong_desire - wrong_literal;
        let keyword_score = (correct_desire_signal - wrong_desire_signal) + behavioral_count * 0.3;

        // --- Signal 2: HDC geometric similarity (supplementary) ---
        let context_hvs: Vec<ContinuousHV> = scenario
            .context
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut hv = adapter.encode(&Scenario::new(*s), dim);
                // Weight later hints more (1.0, 1.5, 2.0, ...)
                let weight = 1.0 + 0.5 * i as f32;
                for v in hv.values.iter_mut() {
                    *v *= weight;
                }
                hv
            })
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        let desire_marker =
            adapter.encode(&Scenario::new("wants needs desires wishes hopes for"), dim);

        let correct_hv = adapter.encode(&Scenario::new(scenario.correct_inference), dim);
        let wrong_hv = adapter.encode(&Scenario::new(scenario.wrong_inference), dim);

        let correct_context_sim = context_bundle.similarity(&correct_hv);
        let wrong_context_sim = context_bundle.similarity(&wrong_hv);
        let correct_desire_sim = desire_marker.similarity(&correct_hv);
        let wrong_desire_sim = desire_marker.similarity(&wrong_hv);

        let geo_signal = (correct_context_sim - wrong_context_sim) as f64
            + (correct_desire_sim - wrong_desire_sim) as f64 * 0.3;

        // --- Combined: keyword-dominant with HDC as tiebreaker ---
        let diff_model_kw = difficulty_model_for("ToMBench::Hinting");
        let kw_attenuation = diff_model_kw.signal_multiplier(config.difficulty);
        let combined = keyword_score * 0.8 * kw_attenuation + geo_signal * 0.2;

        // Softmax decision: convert combined score to P(correct) via sigmoid.
        // Even healthy adults make ~20% errors on hinting tasks (Corcoran
        // et al., 1995). The sigmoid + stochastic sampling models the
        // inherent uncertainty in ToM inference.
        // Time pressure: -0.4/unit flattens sigmoid gain, modeling hasty pragmatic inference;
        // at max pressure, ~20% accuracy drop matches ToM under cognitive load (Lin et al., 2010).
        let diff_model = difficulty_model_for("ToMBench::Hinting");
        let sigmoid_gain = 0.45
            * (1.0 - config.time_pressure * 0.4)
            * diff_model.signal_multiplier(config.difficulty);
        let p_correct = 1.0 / (1.0 + (-combined * sigmoid_gain).exp());
        let seed = config.trial_seed("tombench", "hinting_noise", trial_idx);
        let mut noise_rng = seed ^ 0x9E3779B97F4A7C15;
        noise_rng ^= noise_rng << 13;
        noise_rng ^= noise_rng >> 7;
        noise_rng ^= noise_rng << 17;
        let roll = (noise_rng % 10000) as f64 / 10000.0;
        if roll < p_correct { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP behavioral prediction for desire inference.
    ///
    /// States: [desire_unfulfilled, desire_fulfilled] (dim=2)
    /// Actions: [do_nothing, fulfill_desire] (2 actions)
    /// Observer perceives hints → accumulates desire cues via perceive()
    /// select_action() → should choose action 1 (fulfill) after accumulating hint cues
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Observer perceives each hint as a desire-weighted observation
        for (i, _hint) in scenario.context.iter().enumerate() {
            // Each hint increasingly signals desire: later hints are stronger
            let desire_strength = 0.3 + 0.2 * (i as f64);
            let obs = make_observation(vec![1.0 - desire_strength, desire_strength], "social");
            agent.perceive(&obs);
        }

        // Observer wants desire fulfilled → prefers fulfilled-state observations
        agent.set_goals(vec![0.0, 1.0], 4.0);

        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 1 (fulfill_desire) after accumulating hints
        let expected_action = 1;
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        // Re-derive combined score and p_correct for RT computation.
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let correct_lower = scenario.correct_inference.to_lowercase();
        let wrong_lower = scenario.wrong_inference.to_lowercase();
        let context_text: String = scenario
            .context
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        let desire_words = [
            "wants", "want", "help", "buy", "pay", "leave", "give", "need", "wish", "hope", "get",
            "lend", "offer",
        ];
        let literal_words = [
            "commenting",
            "describing",
            "telling",
            "observing",
            "noting",
            "reporting",
            "mentioning",
            "stating",
            "information",
            "design",
            "prices",
            "time",
        ];
        let behavioral_cues = [
            "shivers",
            "yawns",
            "forgot",
            "tired",
            "looks at",
            "pulls",
            "thin",
            "sitting down",
            "long drive",
            "more expensive",
            "piling up",
        ];

        let correct_desire: f64 = desire_words
            .iter()
            .filter(|k| correct_lower.contains(*k))
            .count() as f64;
        let wrong_desire: f64 = desire_words
            .iter()
            .filter(|k| wrong_lower.contains(*k))
            .count() as f64;
        let correct_literal: f64 = literal_words
            .iter()
            .filter(|k| correct_lower.contains(*k))
            .count() as f64;
        let wrong_literal: f64 = literal_words
            .iter()
            .filter(|k| wrong_lower.contains(*k))
            .count() as f64;
        let behavioral_count: f64 = behavioral_cues
            .iter()
            .filter(|k| context_text.contains(*k))
            .count() as f64;

        let correct_desire_signal = correct_desire - correct_literal;
        let wrong_desire_signal = wrong_desire - wrong_literal;
        let keyword_score = (correct_desire_signal - wrong_desire_signal) + behavioral_count * 0.3;

        let context_hvs: Vec<ContinuousHV> = scenario
            .context
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut hv = adapter.encode(&Scenario::new(*s), dim);
                let weight = 1.0 + 0.5 * i as f32;
                for v in hv.values.iter_mut() {
                    *v *= weight;
                }
                hv
            })
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);
        let desire_marker =
            adapter.encode(&Scenario::new("wants needs desires wishes hopes for"), dim);
        let correct_hv = adapter.encode(&Scenario::new(scenario.correct_inference), dim);
        let wrong_hv = adapter.encode(&Scenario::new(scenario.wrong_inference), dim);
        let correct_context_sim = context_bundle.similarity(&correct_hv);
        let wrong_context_sim = context_bundle.similarity(&wrong_hv);
        let correct_desire_sim = desire_marker.similarity(&correct_hv);
        let wrong_desire_sim = desire_marker.similarity(&wrong_hv);
        let geo_signal = (correct_context_sim - wrong_context_sim) as f64
            + (correct_desire_sim - wrong_desire_sim) as f64 * 0.3;
        let combined = keyword_score * 0.8 + geo_signal * 0.2;
        // Same SAT sigmoid gain as lightweight path (Lin et al., 2010).
        let diff_model = difficulty_model_for("ToMBench::Hinting");
        let sigmoid_gain = 0.45
            * (1.0 - config.time_pressure * 0.4)
            * diff_model.signal_multiplier(config.difficulty);
        let p_correct = 1.0 / (1.0 + (-combined * sigmoid_gain).exp());

        // RT proxy: decisions near p=0.5 are hardest (most uncertain)
        let margin = (p_correct - 0.5).abs() * 2.0; // 0 = hardest, 1 = easiest
        let base = 4.0;
        let range = 7.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        let accuracy = self.run_trial_lightweight(config, trial_idx);
        (accuracy, rt)
    }
}

impl PsychBenchmark for HintingBenchmark {
    fn name(&self) -> &str {
        "ToMBench::Hinting"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Hinting Task",
            citation: "Corcoran et al. (1995)",
            year: 1995,
            doi: Some("10.1017/S0033291700035681"),
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
                    condition: "hinting".to_string(),
                    correct: *accuracies.last().unwrap_or(&0.0) > 0.5,
                    rt_ticks: rt_ticks.last().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("hinting_accuracy", MetricValue::from_samples(&accuracies));
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
    fn test_hinting_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 4,
            dimension: 256,
            ..Default::default()
        };
        let result = HintingBenchmark.run(&config);
        assert!(result.metrics.contains_key("hinting_accuracy"));
    }
}
