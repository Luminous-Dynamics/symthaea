// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! False belief tasks (Sally-Anne type).
//!
//! Tests whether the system tracks that an agent holds a stale belief
//! about a state of the world that has changed in their absence.
//!
//! Uses a lightweight agent-model approach inspired by SocialCoherence:
//! each character's beliefs are tracked as ContinuousHV embeddings.
//! When a character is absent during a state change, their belief
//! model is NOT updated (stays stale), while the "reality" model IS
//! updated. The test checks whether the system predicts based on the
//! character's belief rather than reality.

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

/// False belief benchmark (Sally-Anne paradigm).
pub struct FalseBeliefBenchmark;

/// A single false-belief scenario.
struct FalseBeliefScenario {
    /// Setup sentences (agent observes object location).
    setup: Vec<&'static str>,
    /// Agent leaves (becomes absent for the change).
    _absence: &'static str,
    /// Object is moved (while agent is absent).
    change: &'static str,
    /// Correct answer: where the agent BELIEVES the object is.
    belief_location: &'static str,
    /// Foil: where the object ACTUALLY is (reality answer).
    reality_location: &'static str,
}

impl FalseBeliefBenchmark {
    fn scenarios() -> Vec<FalseBeliefScenario> {
        vec![
            FalseBeliefScenario {
                setup: vec![
                    "Sally puts her marble in the basket",
                    "Sally sees the marble is in the basket",
                ],
                _absence: "Sally leaves the room",
                change: "Anne moves the marble to the box",
                belief_location: "Sally thinks the marble is in the basket",
                reality_location: "The marble is actually in the box",
            },
            FalseBeliefScenario {
                setup: vec![
                    "John puts his chocolate in the cupboard",
                    "John remembers putting chocolate in the cupboard",
                ],
                _absence: "John goes outside to play",
                change: "Mother moves the chocolate to the drawer",
                belief_location: "John thinks the chocolate is in the cupboard",
                reality_location: "The chocolate is actually in the drawer",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Alice places her book on the shelf",
                    "Alice knows her book is on the shelf",
                ],
                _absence: "Alice goes to school",
                change: "Bob moves the book to the table",
                belief_location: "Alice thinks the book is on the shelf",
                reality_location: "The book is actually on the table",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Tom hides his toy behind the curtain",
                    "Tom saw himself hide the toy behind the curtain",
                ],
                _absence: "Tom goes to the kitchen",
                change: "Emma moves the toy under the bed",
                belief_location: "Tom thinks the toy is behind the curtain",
                reality_location: "The toy is actually under the bed",
            },
        ]
    }

    /// Lightweight trial: HDC geometry + belief tracking (no FEP).
    ///
    /// Combines two complementary signals:
    /// 1. **Structural belief tracking**: Extract the original location from setup
    ///    and the moved-to location from the change event. Since the agent was
    ///    absent during the change, their belief should match the *original*
    ///    location. Check whether the belief answer references the original
    ///    location (from setup) rather than the reality (from change).
    /// 2. **HDC geometric similarity**: Bundle the setup sentences into an
    ///    agent belief vector and compare against the two candidate answers.
    ///    The agent belief vector was never updated with the change, so it
    ///    should be closer to the belief-consistent answer.
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // --- Signal 1: Structural belief tracking via location keywords ---
        //
        // The setup sentences contain the *original* location the agent observed.
        // The change sentence contains the *new* location the object was moved to.
        // The agent's belief should reflect the original location, not the new one.
        //
        // We extract location markers from setup (original) and change (moved-to),
        // then check which answer references the original vs the moved-to location.
        let location_prepositions = ["in the", "on the", "behind the", "under the", "to the"];

        // Extract location from setup text (the agent's believed location)
        let setup_text: String = scenario
            .setup
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        let change_text = scenario.change.to_lowercase();
        let belief_text = scenario.belief_location.to_lowercase();
        let reality_text = scenario.reality_location.to_lowercase();

        // Find the location phrase in setup (original) and change (new)
        let extract_location = |text: &str| -> Option<String> {
            for prep in &location_prepositions {
                if let Some(pos) = text.rfind(prep) {
                    let loc_start = pos + prep.len();
                    // Take the rest of the sentence or up to a comma/period
                    let rest = &text[loc_start..];
                    let end = rest.find([',', '.']).unwrap_or(rest.len());
                    let location = rest[..end].trim().to_string();
                    if !location.is_empty() {
                        return Some(location);
                    }
                }
            }
            None
        };

        let original_location = extract_location(&setup_text);
        let moved_location = extract_location(&change_text);

        let structural_score = match (&original_location, &moved_location) {
            (Some(orig), Some(moved)) => {
                // Belief answer should reference the original location
                let belief_has_orig = belief_text.contains(orig.as_str());
                let belief_has_moved = belief_text.contains(moved.as_str());
                let reality_has_orig = reality_text.contains(orig.as_str());
                let reality_has_moved = reality_text.contains(moved.as_str());

                // Score: +1 if belief answer matches original, -1 if reality does
                let mut score = 0.0f64;
                if belief_has_orig && !belief_has_moved {
                    score += 1.0;
                }
                if reality_has_moved && !reality_has_orig {
                    score += 1.0;
                }
                if belief_has_moved {
                    score -= 1.0;
                }
                if reality_has_orig {
                    score -= 1.0;
                }
                score
            }
            _ => 0.0, // Couldn't extract locations; fall through to HDC
        };

        // --- Signal 2: HDC geometric similarity (original approach) ---
        let mut agent_belief: Option<ContinuousHV> = None;

        for sentence in &scenario.setup {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            agent_belief = Some(match agent_belief {
                Some(prev) => ContinuousHV::bundle_owned(&[prev, hv.clone()]),
                None => hv.clone(),
            });
        }

        let belief_hv = adapter.encode(&Scenario::new(scenario.belief_location), dim);
        let reality_hv = adapter.encode(&Scenario::new(scenario.reality_location), dim);

        let Some(agent) = agent_belief else {
            // No setup sentences; fall back to structural score only
            return if structural_score > 0.0 { 1.0 } else { 0.0 };
        };
        // Encoding noise adds per-comparison noise (individual differences in
        // mentalizing precision; Conway & Engle, 1996 WM and controlled attention).
        let enc_noise = config.effective_noise() as f32;
        let noise_seed = config.trial_seed("tombench", "false_belief", trial_idx);
        let belief_noise_val = {
            let ns = noise_seed.wrapping_add(8000);
            ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let reality_noise_val = {
            let ns = noise_seed.wrapping_add(8001);
            ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let belief_sim = agent.similarity(&belief_hv) + belief_noise_val * enc_noise * 0.35;
        let reality_sim = agent.similarity(&reality_hv) + reality_noise_val * enc_noise * 0.35;
        // Time pressure: 0.15/unit attenuates belief-reality discrimination, modeling reality bias
        // under cognitive load (Birch & Bloom, 2007 curse of knowledge; Wickelgren, 1977 SAT).
        let pressure_noise =
            config.time_pressure * 0.15 * diff_model.interference_multiplier(config.difficulty);
        let geo_signal = (belief_sim - reality_sim) as f64 * (1.0 - pressure_noise);

        // --- Combined: structural + HDC geometry equally weighted ---
        let combined = structural_score * 0.4 + geo_signal * 0.6;
        if combined > 0.0 { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP behavioral prediction from false beliefs.
    ///
    /// States: [marble_at_basket, marble_at_box] (dim=2)
    /// Actions: [go_to_basket, go_to_box] (2 actions)
    /// Sally observes marble in basket → belief = [0.9, 0.1]
    /// Anne moves marble → NO update to Sally's beliefs
    /// Sally wants to find marble → set_goals prefers basket-area obs
    /// select_action() → should return action 0 (go-to-basket) based on false beliefs
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{inject_belief, predict_behavior, social_agent};

        // Create FEP agent modeling Sally's mental state
        let mut agent = social_agent(2, 2, 2);

        // Sally observes marble placed in basket → belief = [0.9, 0.1]
        inject_belief(&mut agent, vec![0.9, 0.1]);

        // Anne moves marble to box while Sally is away → NO update to Sally's beliefs
        // (belief stays frozen at [0.9, 0.1])

        // Sally wants to find the marble → prefers basket-area observations
        agent.set_goals(vec![1.0, 0.0], 4.0);

        // Ask: what would Sally DO given her (false) beliefs?
        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 0 (go-to-basket) because Sally believes marble is there
        let expected_action = 0;
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let accuracy = self.run_trial_lightweight(config, trial_idx);

        // RT proxy: compute decision difficulty from belief vs reality similarity margin.
        // Recompute the geometric signal for RT (lightweight path already computed it
        // but didn't return it). Harder decisions (smaller margin) take longer.
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];
        let mut agent_belief: Option<ContinuousHV> = None;
        for sentence in &scenario.setup {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            agent_belief = Some(match agent_belief {
                Some(prev) => ContinuousHV::bundle_owned(&[prev, hv.clone()]),
                None => hv.clone(),
            });
        }
        let belief_hv = adapter.encode(&Scenario::new(scenario.belief_location), dim);
        let reality_hv = adapter.encode(&Scenario::new(scenario.reality_location), dim);
        let enc_noise2 = config.effective_noise() as f32;
        let seed = config.trial_seed("tombench", "false_belief_rt", trial_idx);
        let rt_noise = {
            let ns = seed.wrapping_add(9000);
            ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let margin = if let Some(ref agent) = agent_belief {
            ((agent.similarity(&belief_hv) - agent.similarity(&reality_hv))
                + rt_noise * enc_noise2 * 0.10)
                .abs() as f64
        } else {
            0.5
        };
        let base = 4.0;
        let range = 6.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        (accuracy, rt)
    }
}

impl PsychBenchmark for FalseBeliefBenchmark {
    fn name(&self) -> &str {
        "ToMBench::FalseBelief"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "False Belief Task",
            citation: "Wimmer & Perner (1983)",
            year: 1983,
            doi: Some("10.1016/0010-0277(83)90004-5"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        let mut rt_ticks = Vec::new();
        #[cfg(feature = "symthaea-backend")]
        let mut confidences = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            #[cfg(feature = "symthaea-backend")]
            {
                let (acc, conf) = self.run_trial_full(config, trial);
                accuracies.push(acc);
                confidences.push(conf);
            }
            #[cfg(not(feature = "symthaea-backend"))]
            {
                let (mut acc, rt) = self.run_trial(config, trial);

                // Attention lapse: on a lapse trial the subject guesses randomly
                // instead of applying their mentalizing ability.
                acc = if config.check_correct(acc > 0.5, "false_belief", trial) {
                    1.0
                } else {
                    0.0
                };

                // Curse of knowledge: stochastic response flip at higher difficulty (Birch & Bloom 2007).
                // Two mechanisms: (1) belief-reality confusion (difficulty * 0.35 flip rate),
                // (2) processing noise that scales with difficulty (breaks ceiling at d>=0.3).
                if config.difficulty > 0.0 {
                    let mut rng_state = (config.seed
                        ^ (trial as u64).wrapping_mul(0x517CC1B727220A95))
                    .wrapping_add(1);
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let u = (rng_state as f64) / (u64::MAX as f64);
                    // Primary flip: confusion between agent's belief and reality
                    if u < config.difficulty * 0.35 {
                        acc = 1.0 - acc;
                    }
                    // Secondary flip: processing noise — independent draw
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let u2 = (rng_state as f64) / (u64::MAX as f64);
                    if u2 < config.difficulty * 0.25 {
                        acc = 1.0 - acc;
                    }
                }

                accuracies.push(acc);
                rt_ticks.push(rt);

                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trial,
                        condition: "false_belief".to_string(),
                        correct: acc > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }
        }

        result.insert(
            "false_belief_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        if !rt_ticks.is_empty() {
            result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));
        }
        #[cfg(feature = "symthaea-backend")]
        result.insert("action_confidence", MetricValue::from_samples(&confidences));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_false_belief_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 4,
            dimension: 256,
            ..Default::default()
        };
        let result = FalseBeliefBenchmark.run(&config);
        assert!(result.metrics.contains_key("false_belief_accuracy"));
    }
}
