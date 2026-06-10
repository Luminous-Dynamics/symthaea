// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Insight Problem Solving benchmark.
//!
//! Measures the "Aha!" moment — sudden restructuring of a problem
//! representation that leads to solution discovery (Bowden & Jung-Beeman,
//! 2003; Ohlsson, 1992). Unlike incremental search, insight involves
//! breaking a mental impasse by re-encoding the problem.
//!
//! HDC implementation: encode a problem as a feature bundle, then simulate
//! iterative re-encoding (adding noise + re-bundling). "Insight" occurs
//! when a re-encoded representation suddenly has high similarity to the
//! solution — modeling representational change (Ohlsson, 1992).
//!
//! Human baselines (Bowden & Jung-Beeman, 2003; Metcalfe & Wiebe, 1987):
//! - insight_accuracy: 0.50 (SD 0.15) — proportion of problems solved via insight
//! - restructuring_depth: 0.40 (SD 0.15) — representational change magnitude

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Insight Problem Solving benchmark (Bowden & Jung-Beeman, 2003).
pub struct InsightProblemBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct InsightScenario {
    #[allow(dead_code)]
    name: &'static str,
    n_problem_features: usize,
    n_solution_features: usize,
    /// How many features overlap between problem and solution
    /// (representing the hidden connection that insight reveals)
    shared_features: usize,
}

impl InsightProblemBenchmark {
    fn scenarios() -> Vec<InsightScenario> {
        vec![
            InsightScenario {
                name: "nine_dots",
                n_problem_features: 6,
                n_solution_features: 5,
                shared_features: 2,
            },
            InsightScenario {
                name: "candle_problem",
                n_problem_features: 5,
                n_solution_features: 4,
                shared_features: 2,
            },
            InsightScenario {
                name: "two_string",
                n_problem_features: 5,
                n_solution_features: 5,
                shared_features: 2,
            },
            InsightScenario {
                name: "matchstick",
                n_problem_features: 4,
                n_solution_features: 4,
                shared_features: 1,
            },
            InsightScenario {
                name: "water_jug",
                n_problem_features: 5,
                n_solution_features: 4,
                shared_features: 2,
            },
            InsightScenario {
                name: "cheap_necklace",
                n_problem_features: 6,
                n_solution_features: 5,
                shared_features: 3,
            },
            InsightScenario {
                name: "coin_triangle",
                n_problem_features: 4,
                n_solution_features: 4,
                shared_features: 1,
            },
            InsightScenario {
                name: "horse_rider",
                n_problem_features: 5,
                n_solution_features: 5,
                shared_features: 2,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];
        let seed = config.trial_seed("creativity", "insight", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let noise_scale = config.effective_noise() as f32;

        // Generate shared role-filler pairs (the hidden connection)
        let shared_roles: Vec<ContinuousHV> = (0..scenario.shared_features)
            .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
            .collect();
        let shared_fillers: Vec<ContinuousHV> = (0..scenario.shared_features)
            .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
            .collect();

        // Build "misleading" features — these are the dominant features in the
        // initial problem encoding that create fixation on the wrong approach.
        // In classic insight problems, the solver's initial frame obscures the
        // solution (e.g., assuming lines must stay within the 9-dot grid).
        let n_misleading = scenario.n_problem_features - scenario.shared_features;
        let mislead_bindings: Vec<ContinuousHV> = (0..n_misleading)
            .map(|_| {
                let r = ContinuousHV::random(dim, next_seed(&mut rng));
                let f = ContinuousHV::random(dim, next_seed(&mut rng));
                r.bind(&f)
            })
            .collect();

        // Problem = heavily weighted misleading features + weakly weighted shared features.
        // This models the initial mental set that obscures the solution path.
        let mut problem_parts: Vec<&ContinuousHV> = Vec::new();
        let mut problem_weights: Vec<f32> = Vec::new();
        // Shared features get low initial weight (hidden connection)
        for role in shared_roles.iter().take(scenario.shared_features) {
            problem_parts.push(role); // use roles directly as proxy
            problem_weights.push(0.15);
        }
        // Misleading features get high weight (mental set / fixation)
        for b in &mislead_bindings {
            problem_parts.push(b);
            problem_weights.push(0.85 / n_misleading as f32);
        }
        // Normalize
        let wsum: f32 = problem_weights.iter().sum();
        for w in &mut problem_weights {
            *w /= wsum;
        }
        let problem = ContinuousHV::weighted_bundle(&problem_parts, &problem_weights);

        // Build the full problem_bindings for re-weighting during search
        let mut problem_bindings = Vec::new();
        for i in 0..scenario.shared_features {
            problem_bindings.push(shared_roles[i].bind(&shared_fillers[i]));
        }
        for b in &mislead_bindings {
            problem_bindings.push(b.clone());
        }

        // Build solution representation (shared features + solution-specific)
        let mut solution_bindings = Vec::new();
        for i in 0..scenario.shared_features {
            solution_bindings.push(shared_roles[i].bind(&shared_fillers[i]));
        }
        for _ in scenario.shared_features..scenario.n_solution_features {
            let role = ContinuousHV::random(dim, next_seed(&mut rng));
            let filler = ContinuousHV::random(dim, next_seed(&mut rng));
            solution_bindings.push(role.bind(&filler));
        }
        let refs: Vec<&ContinuousHV> = solution_bindings.iter().collect();
        let solution = ContinuousHV::bundle(&refs);

        // Generate distractors. "Garden path" distractors share the misleading
        // features with the problem (representing the "obvious but wrong"
        // solution that fixation leads to). Other distractors are random.
        let n_distractors = 6;
        let distractors: Vec<ContinuousHV> = (0..n_distractors)
            .map(|d_idx| {
                if d_idx < 2 {
                    // Garden-path distractor: shares misleading features with problem
                    // (the "obvious" wrong answer that fixation points toward)
                    let mut parts: Vec<ContinuousHV> = mislead_bindings.clone();
                    // Add some unique features
                    for _ in 0..2 {
                        let r = ContinuousHV::random(dim, next_seed(&mut rng));
                        let f = ContinuousHV::random(dim, next_seed(&mut rng));
                        parts.push(r.bind(&f));
                    }
                    let refs: Vec<&ContinuousHV> = parts.iter().collect();
                    ContinuousHV::bundle(&refs)
                } else {
                    // Random distractor
                    let n_feats = scenario.n_solution_features;
                    let bindings: Vec<ContinuousHV> = (0..n_feats)
                        .map(|_| {
                            let r = ContinuousHV::random(dim, next_seed(&mut rng));
                            let f = ContinuousHV::random(dim, next_seed(&mut rng));
                            r.bind(&f)
                        })
                        .collect();
                    let refs: Vec<&ContinuousHV> = bindings.iter().collect();
                    ContinuousHV::bundle(&refs)
                }
            })
            .collect();

        // Simulate iterative re-encoding (impasse → restructuring → insight).
        // Models Ohlsson's (1992) representational change theory: the solver
        // starts with an initial encoding that obscures the solution path.
        // Through successive re-weightings (analogous to "breaking set"),
        // shared features between problem and solution may become prominent,
        // producing a sudden similarity jump — the "Aha!" moment.
        //
        // WM capacity determines the number of restructuring attempts.
        let max_iterations = config.working_memory_capacity.clamp(3, 10);

        let initial_sim = problem.similarity(&solution);
        let mut best_sim = initial_sim;
        let mut best_encoding = problem.clone();
        let mut insight_occurred = false;
        let mut restructuring_depth = 0.0_f64;

        // Each iteration tries a different re-weighting of the problem's
        // feature bindings. Some re-weightings emphasize the shared features
        // (enabling insight), others emphasize problem-specific features
        // (maintaining impasse).
        for iter in 0..max_iterations {
            // Generate re-weighting coefficients for each feature binding.
            // The key insight mechanism: random re-weightings occasionally
            // amplify the shared features, increasing similarity to solution.
            let weights: Vec<f32> = (0..problem_bindings.len())
                .map(|f_idx| {
                    let w_seed = seed.wrapping_add(5000 + iter as u64 * 100 + f_idx as u64);
                    let raw = (w_seed.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                        / (1u64 << 31) as f32;
                    // Weight range [0.2, 1.8] — some features suppressed, others amplified
                    0.2 + raw * 1.6
                })
                .collect();

            // Re-encode problem with new weights
            let weighted_bindings: Vec<ContinuousHV> = problem_bindings
                .iter()
                .zip(weights.iter())
                .map(|(b, &w)| {
                    let scaled: Vec<f32> = b.values.iter().map(|&v| v * w).collect();
                    ContinuousHV { values: scaled }
                })
                .collect();
            let refs: Vec<&ContinuousHV> = weighted_bindings.iter().collect();
            let re_encoded = ContinuousHV::bundle(&refs);

            // Add mild noise (encoding imprecision; scales with config noise)
            let noise_hv = ContinuousHV::random(dim, next_seed(&mut rng));
            let noisy = ContinuousHV::weighted_bundle(
                &[&re_encoded, &noise_hv],
                &[1.0 - noise_scale * 0.08, noise_scale * 0.08],
            );

            let new_sim = noisy.similarity(&solution);

            // Insight: similarity to solution exceeds previous best
            // by at least the threshold (sudden jump, not gradual drift)
            let threshold = 0.02_f32;
            if new_sim > best_sim + threshold {
                insight_occurred = true;
                restructuring_depth = (new_sim - initial_sim).max(0.0) as f64;
                best_sim = new_sim;
                best_encoding = noisy;
            }
        }

        // Final evaluation: does the best re-encoding identify the correct
        // solution over distractors? Compare from the restructured perspective.
        let sol_sim = best_encoding.similarity(&solution);
        let best_dist_sim = distractors
            .iter()
            .map(|d| best_encoding.similarity(d))
            .fold(f32::NEG_INFINITY, f32::max);

        let accuracy = if insight_occurred && sol_sim > best_dist_sim {
            1.0
        } else if sol_sim > best_dist_sim {
            // Solution ranks first even without dramatic insight
            // (incremental progress — Weisberg, 1986)
            0.5
        } else {
            0.0
        };

        // Apply lapse model (scales with config.lapse_rate)
        let acc = if config.lapse_rate > 0.0 {
            let lapse_seed = seed.wrapping_add(9000);
            let r =
                (lapse_seed.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f64 / (1u64 << 31) as f64;
            if r < config.lapse_rate { 0.0 } else { accuracy }
        } else {
            accuracy
        };

        (acc, restructuring_depth)
    }
}

impl PsychBenchmark for InsightProblemBenchmark {
    fn name(&self) -> &str {
        "Creativity::InsightProblem"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Insight Problem Solving",
            citation: "Bowden & Jung-Beeman (2003); Ohlsson (1992)",
            year: 1992,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut depths = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (acc, depth) = self.run_trial(config, trial);
            accuracies.push(acc);
            depths.push(depth);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "insight".to_string(),
                    correct: acc > 0.5,
                    rt_ticks: 0.0,
                    similarity: acc,
                    confidence: depth,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("insight_accuracy", MetricValue::from_samples(&accuracies));
        result.insert("restructuring_depth", MetricValue::from_samples(&depths));

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
    fn test_insight_problem_runs() {
        let config = BenchmarkConfig::default();
        let result = InsightProblemBenchmark.run(&config);
        assert!(result.metrics.contains_key("insight_accuracy"));
        assert!(result.metrics.contains_key("restructuring_depth"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "metric not finite");
        }
    }

    #[test]
    fn test_insight_accuracy_bounded() {
        let config = BenchmarkConfig::default();
        let result = InsightProblemBenchmark.run(&config);
        let acc = result.metrics["insight_accuracy"].mean;
        assert!(acc >= 0.0 && acc <= 1.0, "accuracy {acc} out of bounds");
    }

    #[test]
    fn test_insight_problem_values() {
        let config = BenchmarkConfig::default();
        let result = InsightProblemBenchmark.run(&config);
        for (key, val) in &result.metrics {
            eprintln!("IP {key}: mean={:.4}, sd={:.4}", val.mean, val.std_dev);
        }
    }
}
