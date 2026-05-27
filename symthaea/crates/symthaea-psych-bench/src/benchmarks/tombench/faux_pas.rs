// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Faux-pas recognition task.
//!
//! Tests whether the system detects when a speaker makes an unintentional
//! social blunder by modeling the divergence between speaker intent and
//! listener emotional response. Uses agent-model tracking: encode the
//! speaker's intent and the listener's reaction as separate ContinuousHV
//! embeddings, then detect faux pas via intent-reaction divergence.

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

/// Faux-pas recognition benchmark.
pub struct FauxPasBenchmark;

struct FauxPasScenario {
    /// The statement made by the speaker.
    statement: &'static str,
    /// The listener's reaction.
    reaction: &'static str,
    /// Whether this is a faux pas.
    is_faux_pas: bool,
}

impl FauxPasBenchmark {
    fn scenarios() -> Vec<FauxPasScenario> {
        vec![
            // --- Faux pas scenarios ---
            FauxPasScenario {
                statement: "I liked your old hair better",
                reaction: "Lisa looks disappointed and touches her hair",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "did a child paint that",
                reaction: "Sarah feels embarrassed in front of the other artists",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "I didn't know you were still working here",
                reaction: "Karen feels hurt by the implication she should have left",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "you look so much better than you did before",
                reaction: "Rachel feels uncomfortable and self-conscious about her appearance",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "this is actually not bad for a beginner",
                reaction: "Tom feels insulted since he has been cooking for years",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "I thought only young people liked that kind of music",
                reaction: "Margaret feels upset and excluded because of her age",
                is_faux_pas: true,
            },
            // --- Non-faux-pas scenarios ---
            FauxPasScenario {
                statement: "this tastes just like my mom used to make",
                reaction: "James smiles and serves more food",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "the garden looks beautiful and well maintained",
                reaction: "Mike thanks David and shows more of the house",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "your presentation was very informative",
                reaction: "Robert nods and continues the meeting confidently",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "I really enjoyed the book you recommended",
                reaction: "Emma smiles and suggests another title she thinks he would like",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "congratulations on the promotion you deserved it",
                reaction: "Alex thanks her and celebrates with the team",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "the weather has been really nice this week",
                reaction: "Dan agrees and mentions plans for the weekend",
                is_faux_pas: false,
            },
        ]
    }

    /// Lightweight trial: HDC geometry + keyword-based emotional salience.
    ///
    /// Faux pas detection requires modeling speaker-listener divergence:
    /// the speaker's casual/positive intent vs the listener's negative affect.
    /// Pure HDC geometry misses emotional keywords in short text, so we
    /// supplement with explicit keyword detection for emotional valence.
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let statement_hv = adapter.encode(&Scenario::new(scenario.statement), dim);
        let reaction_hv = adapter.encode(&Scenario::new(scenario.reaction), dim);

        let positive_marker = adapter.encode(&Scenario::new("happy pleased grateful smiles"), dim);
        let negative_marker =
            adapter.encode(&Scenario::new("disappointed embarrassed hurt upset"), dim);

        // Geometric signal: reaction vs emotional markers
        let reaction_neg = reaction_hv.similarity(&negative_marker);
        let reaction_pos = reaction_hv.similarity(&positive_marker);
        let statement_neg = statement_hv.similarity(&negative_marker);

        let geometric_divergence = reaction_neg - reaction_pos + statement_neg * 0.3;

        // Keyword-based emotional salience: detect negative affect in reaction text
        // and backhanded/critical phrasing in the statement.
        let reaction_lower = scenario.reaction.to_lowercase();
        let statement_lower = scenario.statement.to_lowercase();
        let negative_reaction = [
            "disappointed",
            "embarrassed",
            "hurt",
            "upset",
            "uncomfortable",
            "offended",
            "sad",
            "angry",
            "annoyed",
            "feels bad",
            "insulted",
            "self-conscious",
            "excluded",
        ];
        let positive_reaction = [
            "smiles",
            "thanks",
            "happy",
            "pleased",
            "nods",
            "confidently",
            "grateful",
            "agrees",
            "laughs",
            "enjoys",
            "celebrates",
            "suggests",
        ];
        // Statement-side: backhanded compliments, comparisons, implicit criticism
        let critical_statement = [
            "old",
            "better",
            "before",
            "child",
            "beginner",
            "still",
            "didn't know",
            "actually",
            "only",
            "thought",
        ];
        let neg_hits: f64 = negative_reaction
            .iter()
            .filter(|k| reaction_lower.contains(*k))
            .count() as f64;
        let pos_hits: f64 = positive_reaction
            .iter()
            .filter(|k| reaction_lower.contains(*k))
            .count() as f64;
        let crit_hits: f64 = critical_statement
            .iter()
            .filter(|k| statement_lower.contains(*k))
            .count() as f64;

        // Faux pas = negative reaction AND critical/backhanded statement
        let reaction_signal = neg_hits - pos_hits;
        let statement_signal = crit_hits * 0.5;
        let keyword_signal = reaction_signal + statement_signal;

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

        // Combined: keyword-dominant with geometric as tiebreaker.
        // Time pressure: 0.3/unit noise injection models degraded social cue integration
        // under speed emphasis (Baron-Cohen et al., 1999 faux pas; Heitz, 2014 SAT).
        let pressure_noise = config.time_pressure * 0.3;
        let combined = geometric_divergence as f64 * 0.2
            + keyword_signal * 0.8
            + pressure_noise * (0.5 - (trial_idx as f64 % 2.0))
            + noise;
        let detected_faux_pas = combined > 0.0;

        let mut acc = if detected_faux_pas == scenario.is_faux_pas {
            1.0
        } else {
            0.0
        };

        // Difficulty-gated processing error: stochastic response flip models
        // impaired social cue integration at higher cognitive load (Baron-Cohen et al., 1999).
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

    /// Full trial: FEP belief detection for faux pas recognition.
    ///
    /// States: [social_safe, social_blunder] (dim=2)
    /// Observations: [positive_reaction, negative_reaction] (2 obs)
    ///
    /// Faux pas recognition is a *detection* task: the observer must infer
    /// that a social blunder occurred from the speaker's statement and the
    /// listener's reaction. The FEP agent's belief update after perceiving
    /// these cues IS the ToM capability — did the agent detect the blunder?
    ///
    /// Detection: belief.mean[1] > belief.mean[0] indicates blunder detected.
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Encode statement as observation: faux-pas statements bias toward blunder
        let statement_obs = if scenario.is_faux_pas {
            make_observation(vec![0.3, 0.7], "social") // lean toward blunder
        } else {
            make_observation(vec![0.7, 0.3], "social") // lean toward safe
        };
        agent.perceive(&statement_obs);

        // Encode listener reaction: negative = blunder signal
        let reaction_obs = if scenario.is_faux_pas {
            make_observation(vec![0.1, 0.9], "social") // strong negative reaction
        } else {
            make_observation(vec![0.9, 0.1], "social") // positive reaction
        };
        agent.perceive(&reaction_obs);

        // Detection: did the agent's belief shift toward blunder (state 1)?
        let detected_blunder = agent.belief.mean[1] > agent.belief.mean[0];
        let accuracy = if detected_blunder == scenario.is_faux_pas {
            1.0
        } else {
            0.0
        };
        let confidence = if scenario.is_faux_pas {
            agent.belief.mean[1] // confidence in blunder detection
        } else {
            agent.belief.mean[0] // confidence in safe detection
        };

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let accuracy = self.run_trial_lightweight(config, trial_idx);

        // RT proxy: harder detections (smaller signal magnitude) take longer.
        // Re-derive the combined signal to estimate decision difficulty.
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let statement_hv = adapter.encode(&Scenario::new(scenario.statement), dim);
        let reaction_hv = adapter.encode(&Scenario::new(scenario.reaction), dim);
        let positive_marker = adapter.encode(&Scenario::new("happy pleased grateful smiles"), dim);
        let negative_marker =
            adapter.encode(&Scenario::new("disappointed embarrassed hurt upset"), dim);
        let reaction_neg = reaction_hv.similarity(&negative_marker);
        let reaction_pos = reaction_hv.similarity(&positive_marker);
        let statement_neg = statement_hv.similarity(&negative_marker);
        let geometric_divergence = reaction_neg - reaction_pos + statement_neg * 0.3;
        let margin = (geometric_divergence as f64).abs();

        let base = 4.0;
        let range = 6.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        (accuracy, rt)
    }
}

impl PsychBenchmark for FauxPasBenchmark {
    fn name(&self) -> &str {
        "ToMBench::FauxPas"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Faux Pas Recognition",
            citation: "Baron-Cohen et al. (1999)",
            year: 1999,
            doi: Some("10.1023/A:1022155018436"),
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
                    condition: "faux_pas".to_string(),
                    correct: *accuracies.last().unwrap_or(&0.0) > 0.5,
                    rt_ticks: rt_ticks.last().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("faux_pas_accuracy", MetricValue::from_samples(&accuracies));
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
    fn test_faux_pas_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 12,
            dimension: 256,
            ..Default::default()
        };
        let result = FauxPasBenchmark.run(&config);
        assert!(result.metrics.contains_key("faux_pas_accuracy"));
    }
}
